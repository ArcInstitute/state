import os
import logging
import torch
import anndata
import h5py as h5
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch import nn

from .nn.model import StateEmbeddingModel
from omegaconf import OmegaConf
from .train.trainer import get_embeddings
from .data import create_dataloader
from .utils import get_embedding_cfg, get_precision_config

log = logging.getLogger(__name__)


class Inference:
    def __init__(self, cfg=None, protein_embeds=None):
        self.model = None
        self.collator = None
        self.protein_embeds = protein_embeds
        self._vci_conf = cfg

    def __load_dataset_meta(self, adata_path):
        with h5.File(adata_path) as h5f:
            attrs = dict(h5f["X"].attrs)
            if "encoding-type" in attrs:  # Fixed: was checking undefined 'adata'
                if attrs["encoding-type"] in ["csr_matrix", "csc_matrix"]:
                    num_cells = attrs["shape"][0]
                    num_genes = attrs["shape"][1]
                elif attrs["encoding-type"] == "array":
                    num_cells = h5f["X"].shape[0]
                    num_genes = h5f["X"].shape[1]
                else:
                    raise ValueError("Input file contains count mtx in non-csr matrix")
            else:
                # No encoding-type specified, try to infer from dataset structure
                if hasattr(h5f["X"], "shape") and len(h5f["X"].shape) == 2:
                    # Treat as dense array - get shape directly from dataset
                    num_cells = h5f["X"].shape[0]
                    num_genes = h5f["X"].shape[1]
                elif all(key in h5f["X"] for key in ["indptr", "indices", "data"]):
                    # Looks like sparse CSR format
                    num_cells = len(h5f["X"]["indptr"]) - 1
                    num_genes = attrs.get(
                        "shape", [0, h5f["X"]["indices"][:].max() + 1 if len(h5f["X"]["indices"]) > 0 else 0]
                    )[1]
                else:
                    raise ValueError("Cannot determine matrix format - no encoding-type and unrecognized structure")

        return {Path(adata_path).stem: (num_cells, num_genes)}

    def _save_data(self, input_adata_path, output_adata_path, obsm_key, data):
        """
        Save data in the output file. This function addresses following cases:
        - output_adata_path does not exist:
          In this case, the function copies the rest of the input file to the
          output file then adds the data to the output file.
        - output_adata_path exists but the dataset does not exist:
          In this case, the function adds the dataset to the output file.
        - output_adata_path exists and the dataset exists:
          In this case, the function resizes the dataset and appends the data to
          the dataset.
        """
        if not os.path.exists(output_adata_path):
            os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
            # Copy rest of the input file to output file
            with h5.File(input_adata_path) as input_h5f:
                with h5.File(output_adata_path, "a") as output_h5f:
                    # Replicate the input data to the output file
                    for _, obj in input_h5f.items():
                        input_h5f.copy(obj, output_h5f)
                    output_h5f.create_dataset(
                        f"/obsm/{obsm_key}", chunks=True, data=data, maxshape=(None, data.shape[1])
                    )
        else:
            with h5.File(output_adata_path, "a") as output_h5f:
                # If the dataset is added to an existing file that does not have the dataset
                if f"/obsm/{obsm_key}" not in output_h5f:
                    output_h5f.create_dataset(
                        f"/obsm/{obsm_key}", chunks=True, data=data, maxshape=(None, data.shape[1])
                    )
                else:
                    output_h5f[f"/obsm/{obsm_key}"].resize(
                        (output_h5f[f"/obsm/{obsm_key}"].shape[0] + data.shape[0]), axis=0
                    )
                    output_h5f[f"/obsm/{obsm_key}"][-data.shape[0] :] = data

    def load_model(self, checkpoint):
        if self.model:
            raise ValueError("Model already initialized")

        # Load and initialize model for eval
        # If no cfg provided, try to restore it from the checkpoint
        cfg_to_use = self._vci_conf
        if cfg_to_use is None:
            try:
                # Need full checkpoint to access embedded config and embeddings
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                # Prefer an explicit yaml snapshot if present
                if isinstance(ckpt, dict) and "cfg_yaml" in ckpt:
                    cfg_to_use = OmegaConf.create(ckpt["cfg_yaml"])  # type: ignore
                # Fallback: try hyper_parameters section
                elif isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
                    hp = ckpt.get("hyper_parameters", {}) or {}
                    if isinstance(hp, dict) and "cfg_yaml" in hp:
                        cfg_to_use = OmegaConf.create(hp["cfg_yaml"])  # type: ignore
                    elif isinstance(hp, dict) and "cfg" in hp and hp["cfg"] is not None:
                        cfg_to_use = OmegaConf.create(hp["cfg"])  # type: ignore
                # Attempt to extract packaged protein embeddings as well
                if isinstance(ckpt, dict) and "protein_embeds_dict" in ckpt and self.protein_embeds is None:
                    self.protein_embeds = ckpt["protein_embeds_dict"]
            except Exception as e:
                log.warning(f"Could not extract config from checkpoint: {e}")

        if cfg_to_use is None:
            raise ValueError("No config found in checkpoint and no override provided. Provide --config to proceed.")

        # Keep internal cfg in sync
        self._vci_conf = cfg_to_use
        self.model = StateEmbeddingModel.load_from_checkpoint(checkpoint, dropout=0.0, strict=False, cfg=self._vci_conf)

        # Convert model to appropriate precision for faster inference
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)
        self.model = self.model.to(precision)

        # Resolve protein embeddings: prefer provided/packaged, then config
        if self.protein_embeds is None:
            # Try to extract from the checkpoint payload even if cfg was provided externally
            try:
                ckpt2 = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if isinstance(ckpt2, dict) and "protein_embeds_dict" in ckpt2:
                    self.protein_embeds = ckpt2["protein_embeds_dict"]
            except Exception:
                pass
        all_pe = self.protein_embeds or get_embeddings(self._vci_conf)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device, dtype=precision)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        if self.protein_embeds is None:
            # Final fallback to config path
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings, weights_only=False)

    def init_from_model(self, model, protein_embeds=None):
        """
        Intended for use during training
        """
        self.model = model
        if protein_embeds:
            self.protein_embeds = protein_embeds
        else:
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf), weights_only=False)

    def get_gene_embedding(self, genes):
        protein_embeds = [self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(5120) for x in genes]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)
        protein_embeds = torch.stack(protein_embeds).to(self.model.device, dtype=precision)
        return self.model.gene_embedding_layer(protein_embeds)

    def encode(self, dataloader, rda=None):
        with torch.no_grad():
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            precision = get_precision_config(device_type=device_type)
            with torch.autocast(device_type=device_type, dtype=precision):
                for i, batch in enumerate(dataloader):
                    _, _, _, emb, ds_emb = self.model._compute_embedding_for_batch(batch)
                    embeddings = emb.detach().cpu().float().numpy()

                    ds_emb = self.model.dataset_embedder(ds_emb)
                    ds_embeddings = ds_emb.detach().cpu().float().numpy()

                    yield embeddings, ds_embeddings

    def encode_adata(
        self,
        input_adata_path: str,
        output_adata_path: str | None = None,
        emb_key: str = "X_emb",
        dataset_name: str | None = None,
        batch_size: int | None = None,
        lancedb_path: str | None = None,
        update_lancedb: bool = False,
        lancedb_batch_size: int = 1000,
    ):
        shape_dict = self.__load_dataset_meta(input_adata_path)
        adata = anndata.read_h5ad(input_adata_path)
        if dataset_name is None:
            dataset_name = Path(input_adata_path).stem

        # Convert to CSR format if needed
        adata = self._convert_to_csr(adata)

        # Auto-detect the best gene column
        gene_column: Optional[str] = self._auto_detect_gene_column(adata)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)

        # Allow overriding batch size for faster inference if more VRAM is available
        dataloader_cfg = self._vci_conf
        if batch_size is not None:
            try:
                dataloader_cfg = OmegaConf.create(OmegaConf.to_container(self._vci_conf, resolve=True))
                # Ensure nested structure exists
                if not hasattr(dataloader_cfg, "model"):
                    dataloader_cfg["model"] = {}
                dataloader_cfg.model.batch_size = int(batch_size)
                log.info(f"Using override batch size: {batch_size}")
            except Exception:
                # Fallback: attempt direct set; if it fails, proceed with original config
                try:
                    dataloader_cfg.model.batch_size = int(batch_size)
                    log.info(f"Using override batch size: {batch_size}")
                except Exception:
                    log.warning("Failed to override batch size; using config default")

        dataloader = create_dataloader(
            dataloader_cfg,
            adata=adata,
            adata_name=dataset_name or "inference",
            shape_dict=shape_dict,
            data_dir=os.path.dirname(input_adata_path),
            shuffle=False,
            protein_embeds=self.protein_embeds,
            precision=precision,
            gene_column=gene_column,
        )

        all_embeddings = []
        all_ds_embeddings = []
        for embeddings, ds_embeddings in tqdm(self.encode(dataloader), total=len(dataloader), desc="Encoding"):
            all_embeddings.append(embeddings)
            if ds_embeddings is not None:
                all_ds_embeddings.append(ds_embeddings)

        # attach this as a numpy array to the adata and write it out
        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        if len(all_ds_embeddings) > 0:
            all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(np.float32)

            # concatenate along axis -1 with all embeddings
            all_embeddings = np.concatenate([all_embeddings, all_ds_embeddings], axis=-1)

        # if output_adata_path is provided, write the adata to the file
        if output_adata_path is not None:
            adata.obsm[emb_key] = all_embeddings
            adata.write_h5ad(output_adata_path)

        # Save to lancedb, if requested
        if lancedb_path is not None:
            from .vectordb import StateVectorDB

            log.info(f"Saving embeddings to LanceDB at {lancedb_path}")
            vector_db = StateVectorDB(lancedb_path)

            # Extract relevant metadata
            metadata = adata.obs.copy()

            # Create or update the database
            vector_db.create_or_update_table(
                embeddings=all_embeddings,
                metadata=metadata,
                embedding_key=emb_key,
                dataset_name=dataset_name or Path(input_adata_path).stem,
                batch_size=lancedb_batch_size,
            )

            log.info(f"Successfully saved {len(all_embeddings)} embeddings to LanceDB")

    def _convert_to_csr(self, adata):
        """Convert the adata.X matrix to CSR format if it's not already."""
        from scipy.sparse import csr_matrix, issparse

        if issparse(adata.X) and not isinstance(adata.X, csr_matrix):
            print(f"Converting {type(adata.X).__name__} to csr_matrix format")
            adata.X = csr_matrix(adata.X)
        return adata

    def _auto_detect_gene_column(self, adata):
        """Auto-detect the gene column with highest overlap with protein embeddings."""
        if self.protein_embeds is None:
            log.warning("No protein embeddings available for auto-detection, using index")
            return None

        protein_genes = set(self.protein_embeds.keys())
        best_column = None
        best_overlap = 0
        best_overlap_pct = 0

        # Check index first
        if hasattr(adata.var, "index"):
            index_genes = set(adata.var.index)
            overlap = len(protein_genes.intersection(index_genes))
            overlap_pct = overlap / len(index_genes) if len(index_genes) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_column = None  # None means use index

        # Check all columns in var
        for col in adata.var.columns:
            col_genes = set(adata.var[col].dropna().astype(str))
            overlap = len(protein_genes.intersection(col_genes))
            overlap_pct = overlap / len(col_genes) if len(col_genes) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_column = col

        if best_column is None:
            log.info(
                f"Auto-detected gene column: var.index (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )
        else:
            log.info(
                f"Auto-detected gene column: var.{best_column} (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )

        return best_column

    def decode_from_file(self, adata_path, emb_key: str, read_depth=None, batch_size=64):
        adata = anndata.read_h5ad(adata_path)
        genes = adata.var.index
        yield from self.decode_from_adata(adata, genes, emb_key, read_depth, batch_size)

    @torch.no_grad()
    def decode_from_adata(self, adata, genes, emb_key: str, read_depth=None, batch_size=64):
        try:
            cell_embs = adata.obsm[emb_key]
        except:
            cell_embs = adata.X

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)
        cell_embs = torch.Tensor(cell_embs).to(self.model.device, dtype=precision)

        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 4.0

        gene_embeds = self.get_gene_embedding(genes)
        with torch.autocast(device_type=device_type, dtype=precision):
            for i in tqdm(range(0, cell_embs.size(0), batch_size), total=int(cell_embs.size(0) // batch_size)):
                cell_embeds_batch = cell_embs[i : i + batch_size]
                task_counts = torch.full(
                    (cell_embeds_batch.shape[0],), read_depth, device=self.model.device, dtype=precision
                )

                ds_emb = cell_embeds_batch[:, -self.model.z_dim_ds :]  # last ten columns are the dataset embeddings
                merged_embs = StateEmbeddingModel.resize_batch(
                    cell_embeds_batch, gene_embeds, task_counts=task_counts, ds_emb=ds_emb
                )
                logprobs_batch = self.model.binary_decoder(merged_embs)
                logprobs_batch = logprobs_batch.detach().cpu().float().numpy()
                yield logprobs_batch.squeeze()
