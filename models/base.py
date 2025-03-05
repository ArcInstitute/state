import torch
import logging
import wandb

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import numpy as np
import torch.nn as nn

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, List
from lightning.pytorch import LightningModule
from tqdm import tqdm

from models.decoders import DecoderInterface
from models.utils import get_loss_fn
from validation.metrics import compute_metrics

logger = logging.getLogger(__name__)

class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.
    This decoder is trained separately from the main perturbation model.
    
    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Build the layers
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, gene_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x: Latent embeddings of shape [batch_size, latent_dim]
            
        Returns:
            Gene expression predictions of shape [batch_size, gene_dim]
        """
        return self.decoder(x)

class PerturbationModel(ABC, LightningModule):
    """
    Base class for perturbation models that can operate on either raw counts or embeddings.

    Args:
        input_dim: Dimension of input features (genes or embeddings)
        hidden_dim: Hidden dimension for neural network layers
        output_dim: Dimension of output (always gene space)
        pert_dim: Dimension of perturbation embeddings
        dropout: Dropout rate
        lr: Learning rate for optimizer
        loss_fn: Loss function ('mse' or custom nn.Module)
        output_space: 'gene' or 'latent'
        decoder: Optionally a class implementing DecoderInterface, if output is latent
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        decoder: Optional[DecoderInterface] = None,
        gene_names: Optional[List[str]] = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core architecture settings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.embed_key = embed_key
        self.output_space = output_space
        self.batch_size = batch_size
        self.control_pert = control_pert

        # Training settings
        self.decoder = decoder if output_space == "latent" else None
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)

        self.precision = kwargs["precision"]
        self.gene_decoder = None
        if embed_key != "X_hvg" and output_space == "gene":
            if embed_key == "X_scfound":
                hidden_dims = [512, 1024]
            else:
                hidden_dims = [hidden_dim * 2, hidden_dim * 4]

            self.gene_decoder = LatentToGeneDecoder(
                latent_dim=self.output_dim,
                gene_dim=gene_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            
            if self.precision == "bfloat16":
                # Convert to bfloat16 for faster training
                self.gene_decoder = self.gene_decoder.to(torch.bfloat16)
            logger.info(f"Initialized gene decoder for embedding {embed_key} to gene space")
        

        # For caching validation data across steps, if desired
        self.val_cache = defaultdict(list)
        self.test_cache = defaultdict(list)

    def get_precision(self):
        return self.precision

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""
        pass

    @abstractmethod
    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Encode perturbation into latent space."""
        pass

    @abstractmethod
    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Encode gene expression into latent space if needed."""
        pass

    @abstractmethod
    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """Perturb basal expression with perturbation in the co-embedding space."""
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        pred = self(batch)
        
        # Compute main model loss
        main_loss = self.loss_fn(pred, batch["X"])
        self.log("train_loss", main_loss)
        
        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "X_hvg" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model
            
            gene_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["X_hvg"]
            decoder_loss = self.loss_fn(gene_preds, gene_targets)
            
            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss
        
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        pred = self(batch)
        loss = self.loss_fn(pred, batch["X"])

        is_control = self.control_pert in batch["pert_name"]
        if np.random.rand() < 0.1 or is_control:
            self._update_val_cache(batch, pred)
        self.log("val_loss", loss)

        return {"loss": loss, "predictions": pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        """Track decoder performance during validation without training it."""
        if self.gene_decoder is not None and "X_hvg" in batch:
            # Get model predictions from validation step
            latent_preds = outputs["predictions"]
            
            # Get decoder predictions
            gene_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["X_hvg"]
            
            # Compute loss (but don't backprop)
            decoder_loss = self.loss_fn(gene_preds, gene_targets)
            
            # Log the validation metric
            self.log("decoder_val_loss", decoder_loss)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self(batch)
        if self.output_space == "gene":
            if "X_hvg" not in batch:
                raise ValueError("We expected 'X_hvg' to be in batch for gene-level output!")
            target = batch["X_hvg"]
            loss = self.loss_fn(pred, target)
        else:
            loss = self.loss_fn(pred, batch["X"])

        self.log("test_loss", loss, prog_bar=True)
        self._update_test_cache(batch, pred)  # NEW: cache test outputs
        return {
            "preds": pred,  # The distribution's sample
            "X": batch.get("X", None),  # The target gene expression or embedding
            "X_hvg": batch.get("X_hvg", None),  # the true, raw gene expression
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "gem_group": batch.get("gem_group", None),
            "basal": batch.get("basal", None),
        }

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch)
        output_dict = {
            "preds": latent_output,
            "X": batch.get("X", None),
            "X_hvg": batch.get("X_hvg", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "gem_group": batch.get("gem_group", None),
            "basal": batch.get("basal", None),
        }

        if self.gene_decoder is not None:
            gene_preds = self.gene_decoder(latent_output)
            output_dict["gene_preds"] = gene_preds

        return output_dict

    def decode_to_gene_space(self, latent_embeds: torch.Tensor) -> torch.Tensor:
        """
        Decode latent embeddings to gene expression space.
        
        Args:
            latent_embeds: Embeddings in latent space
            
        Returns:
            Gene expression predictions or None if decoder is not available
        """
        if self.gene_decoder is not None:
            return self.gene_decoder(latent_embeds)
        return None

    def configure_optimizers(self):
        """
        Configure a single optimizer for both the main model and the gene decoder.
        """
        # Use a single optimizer for all parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_validation_epoch_end(self) -> None:
        # bypass sanity checkers since we don't add everything to validation cache
        if len(self.val_cache) == 0:
            return

        for k in self.val_cache:
            if k in ("X", "X_hvg", "pred", "gene_pred", "pert", "basal"):
                self.val_cache[k] = np.concatenate(self.val_cache[k])
            else:
                try:
                    self.val_cache[k] = np.concatenate([np.array(obs) for obs in self.val_cache[k]])
                except:
                    # Handle non-array data
                    pass

        # store the non-array, like pert_name, data in adata obs
        obs = pd.DataFrame(
            {k: v for k, v in self.val_cache.items() if k not in ("X", "X_hvg", "pred", "gene_pred", "pert", "basal")}
        )
        adata_real = ad.AnnData(obs=obs, X=self.val_cache["X"])
        adata_pred = ad.AnnData(obs=obs, X=self.val_cache["pred"])

        # Set up gene expression adatas
        adata_real_gene = None
        adata_pred_gene = None
        
        # Check if we have gene expression data
        has_gene_data = "X_hvg" in self.val_cache
        
        # Check if we have decoded gene predictions
        has_gene_preds = "gene_pred" in self.val_cache
        
        # Set up ground truth gene expression adata if available
        if has_gene_data:
            adata_real_gene = ad.AnnData(obs=obs, X=self.val_cache["X_hvg"])
        
        # Set up predicted gene expression adata if available
        if has_gene_preds:
            adata_pred_gene = ad.AnnData(obs=obs, X=self.val_cache["gene_pred"])

        # Standardize logging?
        wandb.define_metric("val", step_metric="epoch")

        # pytorch lightning sanity check runs val for a single batch, which often does
        # not have control so can't compute these metrics
        uniq_perts = obs["pert_name"].unique()
        if self.control_pert not in uniq_perts:
            self.val_cache = defaultdict(list)
            return

        try:
            metrics = compute_metrics(
                adata_pred=adata_pred,
                adata_real=adata_real,  # if output space is gene, this contains the true gene expression anyways, so ignore adata_real_exp
                adata_pred_gene=adata_pred_gene,
                adata_real_gene=adata_real_gene,
                include_dist_metrics=False,
                control_pert=self.control_pert,
                pert_col="pert_name",
                celltype_col="cell_type",
                DE_metric_flag=True,
                class_score_flag=True,
                embed_key=self.embed_key,
                output_space=self.output_space,
            )

            if metrics:
                # Dictionary to store aggregated metrics
                aggregate_metrics = defaultdict(list)
                
                # Collect metrics across all cell types
                for celltype, metrics_df in metrics.items():
                    numeric_df = metrics_df.apply(pd.to_numeric, errors="coerce")
                    celltype_metrics = numeric_df.mean(0).to_dict()
                    
                    # Log individual cell type metrics
                    for k, v in celltype_metrics.items():
                        if np.isfinite(v):
                            self.log(f"val/{k}_{celltype}", v)
                            # Collect for averaging
                            aggregate_metrics[k].append(v)
                
                # Compute and log average metrics across cell types
                for metric_name, values in aggregate_metrics.items():
                    avg_value = np.mean([v for v in values if np.isfinite(v)])
                    if np.isfinite(avg_value):
                        self.log(f"val/{metric_name}", avg_value)
        except:
            # log a warning
            logger.warning("Error in computing metrics during validation epoch.")

        self.val_cache = defaultdict(list)

    def _update_val_cache(self, batch, pred):
        for k in batch:
            if k not in self.val_cache:
                self.val_cache[k] = []

            # add to calculate validation set metrics during training
            if isinstance(batch[k], torch.Tensor):
                self.val_cache[k].append(batch[k].detach().cpu().float().numpy())
            else:
                self.val_cache[k].append(batch[k])

        if "pred" not in self.val_cache:
            self.val_cache["pred"] = []

        self.val_cache["pred"].append(pred.detach().cpu().float().numpy())

        if self.gene_decoder is not None:
            gene_pred = self.gene_decoder(pred)
            if "gene_pred" not in self.val_cache:
                self.val_cache["gene_pred"] = []

            self.val_cache["gene_pred"].append(gene_pred.detach().cpu().float().numpy())

    def _update_test_cache(self, batch, pred):
        for k in batch:
            if k not in self.test_cache:
                self.test_cache[k] = []
            if isinstance(batch[k], torch.Tensor):
                self.test_cache[k].append(batch[k].detach().cpu().float().numpy())
            else:
                self.test_cache[k].append(batch[k])

        if "pred" not in self.test_cache:
            self.test_cache["pred"] = []

        self.test_cache["pred"].append(pred.detach().cpu().float().numpy())

        if self.gene_decoder is not None:
            gene_pred = self.gene_decoder(pred)
            if "gene_pred" not in self.test_cache:
                self.test_cache["gene_pred"] = []

            self.test_cache["gene_pred"].append(gene_pred.detach().cpu().float().numpy())

    def compute_test_metrics(self, dataloader, prefix="test") -> Dict[str, float]:
        """
        Compute model metrics on a given dataloader.
        This consolidates the functionality previously split across test_step and on_test_epoch_end.
        
        Args:
            dataloader: PyTorch DataLoader containing the evaluation data
            prefix: String prefix for metric names (default: "test")
            
        Returns:
            Dictionary of computed metrics
        """
        # Initialize cache for collecting predictions and metadata
        cache = defaultdict(list)
        
        # Collect predictions and data across all batches
        self.eval()  # Ensure model is in eval mode
        with torch.no_grad():
            pbar = tqdm(
                total=len(dataloader),
                desc="Testing",
                unit="batch",
                leave=True,
                position=0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            for batch in dataloader:
                # Move batch to device
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}
                
                # Get predictions in latent space
                latent_pred = self(batch)
                
                # Generate gene predictions if needed
                gene_pred = None
                if self.gene_decoder is not None:
                    gene_pred = self.gene_decoder(latent_pred)

                # Cache predictions and data
                if np.random.rand() < 0.1 / self.batch_size or self.control_pert in batch["pert_name"]:
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            cache[k].append(batch[k].cpu().float().numpy())
                        else:
                            cache[k].append(batch[k])
                    cache["pred"].append(latent_pred.cpu().float().numpy())
                    if gene_pred is not None:
                        cache["gene_pred"].append(gene_pred.cpu().float().numpy())

                pbar.update(1)
            pbar.close()

        # Concatenate all cached arrays
        for k in cache:
            if k in ("X", "X_hvg", "pred", "gene_pred", "pert", "basal"):
                cache[k] = np.concatenate(cache[k])
            else:
                try:
                    cache[k] = np.concatenate([np.array(obs) for obs in cache[k]])
                except:
                    pass

        # Build AnnData objects for metrics computation
        obs = pd.DataFrame({k: v for k, v in cache.items() 
                          if k not in ("X", "X_hvg", "pred", "pert", "basal")})
        adata_real = ad.AnnData(obs=obs, X=cache["X"])
        adata_pred = ad.AnnData(obs=obs, X=cache["pred"])

        adata_real_gene = None
        adata_pred_gene = None
        if "gene_pred" in cache:
            # We have decoded predictions
            adata_real_gene = ad.AnnData(obs=obs, X=cache["X_hvg"])
            adata_pred_gene = ad.AnnData(obs=obs, X=cache["gene_pred"])

        try:
            # Compute metrics using existing compute_metrics function
            metrics = compute_metrics(
                adata_real=adata_real,
                adata_pred=adata_pred,
                adata_pred_gene=adata_pred_gene,
                adata_real_gene=adata_real_gene,
                include_dist_metrics=False,
                control_pert="DMSO_TF",
                pert_col="pert_name",
                celltype_col="cell_type",
                DE_metric_flag=True,
                class_score_flag=True,
                embed_key=self.embed_key,
                output_space=self.output_space,
            )

            # Process metrics
            if metrics:
                aggregate_metrics = defaultdict(list)
                metric_dict = {}
                
                # Process metrics for each cell type
                for celltype, metrics_df in metrics.items():
                    numeric_df = metrics_df.apply(pd.to_numeric, errors="coerce")
                    celltype_metrics = numeric_df.mean(0).to_dict()
                    
                    for k, v in celltype_metrics.items():
                        if np.isfinite(v):
                            metric_name = f"{prefix}/{k}_{celltype}"
                            metric_dict[metric_name] = v
                            aggregate_metrics[k].append(v)
                
                # Compute averages across cell types
                for metric_name, values in aggregate_metrics.items():
                    avg_value = np.mean([v for v in values if np.isfinite(v)])
                    if np.isfinite(avg_value):
                        metric_dict[f"{prefix}/{metric_name}"] = avg_value

                return metric_dict

        except Exception as e:
            logger.warning(f"Error computing metrics: {str(e)}")
            return {}
