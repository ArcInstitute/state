import os
import pickle
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import logging
import yaml
from pathlib import Path
from models import *

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceModule:
    def __init__(self, model_folder: str, cell_set_len: int):
        """
        Initialize the inference module.
        
        Args:
            model_folder: Path to the folder where the model checkpoint and data_module.pkl are stored.
            model_class: The LightningModule class to use for loading the checkpoint.
        """
        self.model_folder = Path(model_folder)

        # Load the pickled data module to obtain properties.
        dm_path = self.model_folder / "data_module.pkl"
        if not os.path.exists(dm_path):
            raise FileNotFoundError(f"Could not find data_module.pkl in {model_folder}")
        with open(dm_path, "rb") as f:
            self.data_module = pickle.load(f)

        # Obtain embed key and perturbation one-hot map from the loaded data module.
        self.embed_key = self.data_module.embed_key  # e.g. "X_uce"
        self.pert_onehot_map = self.data_module.pert_onehot_map  # dictionary: pert_name -> one-hot tensor
        logger.info(f"Loaded perturbation map with {len(self.pert_onehot_map)} perturbations")

        # Load the model checkpoint.
        self.model: PertSetsPerturbationModel = self._load_model()

        if cell_set_len > self.data_module.cell_sentence_len:
            logger.info(f"Requested cell set length {cell_set_len} is greater than the maximum cell sentence length "
                        f"({self.data_module.cell_sentence_len}). Setting cell_set_len to {self.data_module.cell_sentence_len}")
            self.cell_set_len = self.data_module.cell_sentence_len
        else:
            self.cell_set_len = cell_set_len

    def get_control_pert(self):
        return self.data_module.control_pert

    def available_perturbations(self):
        return list(self.pert_onehot_map.keys())
        
    def _load_model(self):
        """
        Load the trained model from the checkpoint file.
        
        Returns:
            The loaded model
        """
        # Find the checkpoint file
        checkpoint_dir = self.model_folder / "checkpoints"
        checkpoint_path = checkpoint_dir / "last.ckpt"
        
        if not checkpoint_path.exists():
            # Try to find the latest checkpoint based on step number
            checkpoint_files = list(checkpoint_dir.glob("step=*.ckpt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
            
            # Sort by step number (assuming filenames contain 'step=X')
            checkpoint_files.sort(key=lambda x: int(str(x).split("step=")[1].split("-")[0]))
            # Get the highest step (last item after sorting)
            checkpoint_path = checkpoint_files[-1]
        
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Get model dimensions from the data module
        var_dims = self.data_module.get_var_dims()
        
        # Determine model type from the config file
        model_config_path = self.model_folder / "config.yaml"
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)
        
        model_class_name = config["model"]["name"]
        ModelClass = self._get_model_class(model_class_name)

        model_kwargs = config["model"]["kwargs"]
        model_init_kwargs = {
            "input_dim": var_dims["input_dim"],
            "hidden_dim": model_kwargs["hidden_dim"],
            "output_dim": var_dims["output_dim"],
            "pert_dim": var_dims["pert_dim"],
            # other model_kwargs keys to pass along:
            **model_kwargs,
        }

        model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
        model.eval()
        return model

    def _get_model_class(self, model_type: str):
        """
        Get the model class based on the model type.
        
        Args:
            model_type: The type of the model
            
        Returns:
            The model class
        """
        model_type = model_type.lower()
        model_classes = {
            "simplesum": SimpleSumPerturbationModel,
            "globalsimplesum": GlobalSimpleSumPerturbationModel,
            "celltypemean": CellTypeMeanModel,
            "embedsum": EmbedSumPerturbationModel,
            "neuralot": PertSetsPerturbationModel, # added as legacy for now for current ckpts. TODO: Remove before release
            "pertsets": PertSetsPerturbationModel,
            "old_neuralot": OldNeuralOTPerturbationModel,
            "decoder_only": DecoderOnlyPerturbationModel,
            "gears": GEARS, 
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_classes[model_type]

    def perturb(self, adata: ad.AnnData, pert_key: str, celltype_key: str) -> ad.AnnData:
        """
        Run inference on an input AnnData and append predictions to .obsm.
        
        Steps:
        1. Verify that required columns exist.
        2. Remove cells with unknown perturbation labels.
        3. Group cells by (cell_type, perturbation) and split each group into chunks
            (each chunk contains at most cell_set_len cells).
        4. For each chunk, create a batch with:
                - "basal": embeddings from adata.obsm[self.embed_key]
                - "pert": repeated one-hot vector for the perturbation.
        5. Run the model (and gene decoder, if available) on each batch.
        6. Restore the original order and add predictions as new keys in .obsm.
        7. Save the modified AnnData.
        
        Args:
            adata: Input AnnData.
            pert_key: Column name for perturbation labels.
            celltype_key: Column name for cell type labels.
            
        Returns:
            AnnData with new obsm entries: {embed_key}_pert and X_hvg_pert (if gene decoder exists).
        """
        # Verify required columns.
        if pert_key not in adata.obs.columns:
            raise ValueError(f"Input AnnData must have a column '{pert_key}' in .obs")
        if celltype_key not in adata.obs.columns:
            raise ValueError(f"Input AnnData must have a column '{celltype_key}' in .obs for grouping.")
        
        # Remove cells with unknown perturbation labels.
        valid_mask = adata.obs[pert_key].isin(self.pert_onehot_map)
        if valid_mask.sum() < adata.n_obs:
            logger.info(f"Removing {adata.n_obs - valid_mask.sum()} cells with unknown perturbation labels.")
            adata = adata[valid_mask].copy()
        
        # Get cell embeddings.
        X_embed = adata.obsm[self.embed_key]
        if not torch.is_tensor(X_embed):
            X_embed = torch.tensor(X_embed, dtype=torch.float32)
        device = next(self.model.parameters()).device
        X_embed = X_embed.to(device)
        
        # Extract perturbation and cell type labels; store original indices.
        pert_labels = adata.obs[pert_key].values
        cell_types = adata.obs[celltype_key].values
        original_indices = np.arange(adata.n_obs)
        
        # Group cells by (cell_type, perturbation).
        groups = {}
        for i, (ct, p) in enumerate(zip(cell_types, pert_labels)):
            key = (ct, p)
            groups.setdefault(key, []).append(i)
        
        latent_predictions = {}
        gene_predictions = {}
        
        from tqdm import tqdm
        # Loop over groups with a progress bar.
        for (ct, p), idx_list in tqdm(list(groups.items()), total=len(groups), desc="Processing groups"):
            onehot = self.pert_onehot_map[p].to(device)
            n = len(idx_list)
            # Process each group in chunks (cell sentences).
            for start in tqdm(range(0, n, self.cell_set_len), 
                            desc=f"Group {ct}-{p}", leave=False):
                chunk_indices = idx_list[start : start + self.cell_set_len]
                basal = X_embed[chunk_indices, :]  # (chunk_size, embed_dim)
                pert_tensor = onehot.unsqueeze(0).repeat(len(chunk_indices), 1)
                batch = {"basal": basal, "pert": pert_tensor}
                with torch.no_grad():
                    batch_pred = self.model.forward(batch, padded=False)
                    latent_out = batch_pred
                    gene_out = self.model.gene_decoder(latent_out) if (hasattr(self.model, "gene_decoder") 
                                                                        and self.model.gene_decoder is not None) else None
                latent_np = latent_out.detach().cpu().numpy()
                if gene_out is not None:
                    gene_np = gene_out.detach().cpu().numpy()
                for i_local, orig_idx in enumerate(chunk_indices):
                    latent_predictions[orig_idx] = latent_np[i_local]
                    if gene_out is not None:
                        gene_predictions[orig_idx] = gene_np[i_local]
        
        # Reconstruct predictions in the original order.
        valid_indices = sorted(latent_predictions.keys())
        latent_array = np.array([latent_predictions[i] for i in valid_indices])
        gene_array = np.array([gene_predictions[i] for i in valid_indices]) if gene_out is not None else None
        
        adata_out = adata[valid_indices].copy()
        adata_out.obsm[self.embed_key + "_pert"] = latent_array
        if gene_array is not None:
            adata_out.obsm["X_hvg_pert"] = gene_array
        return adata_out