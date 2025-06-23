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
from .train.trainer import get_embeddings
from .data import create_dataloader
from .utils import get_embedding_cfg

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
                if attrs["encoding-type"] == "csr_matrix":
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
        self.model = StateEmbeddingModel.load_from_checkpoint(checkpoint, strict=False)
        all_pe = self.protein_embeds or get_embeddings(self._vci_conf)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        if self.protein_embeds is None:
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
        protein_embeds = torch.stack(protein_embeds).to(self.model.device)
        return self.model.gene_embedding_layer(protein_embeds)

    def encode(self, dataloader, rda=None):
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                torch.cuda.empty_cache()
                _, _, _, emb, ds_emb = self.model._compute_embedding_for_batch(batch)
                embeddings = emb.detach().cpu().numpy()

                ds_emb = self.model.dataset_embedder(ds_emb)
                ds_embeddings = ds_emb.detach().cpu().numpy()

                yield embeddings, ds_embeddings

    def encode_adata(
        self,
        input_adata_path: str,
        output_adata_path: str,
        emb_key: str = "X_emb",
        dataset_name=None,
        batch_size: int = 32,
    ):
        shape_dict = self.__load_dataset_meta(input_adata_path)
        adata = anndata.read_h5ad(input_adata_path)
        if dataset_name is None:
            dataset_name = Path(input_adata_path).stem

        dataloader = create_dataloader(
            self._vci_conf,
            adata=adata,
            adata_name=dataset_name or "inference",
            shape_dict=shape_dict,
            data_dir=os.path.dirname(input_adata_path),
            shuffle=False,
            protein_embeds=self.protein_embeds,
        )

        all_embeddings = []
        all_ds_embeddings = []
        for embeddings, ds_embeddings in tqdm(self.encode(dataloader), total=len(dataloader), desc="Encoding"):
            all_embeddings.append(embeddings)
            if ds_embeddings is not None:
                all_ds_embeddings.append(ds_embeddings)

        # attach this as a numpy array to the adata and write it out
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        if len(all_ds_embeddings) > 0:
            all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0)

            # concatenate along axis -1 with all embeddings
            all_embeddings = np.concatenate([all_embeddings, all_ds_embeddings], axis=-1)

        adata.obsm[emb_key] = all_embeddings
        adata.write_h5ad(output_adata_path)

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
        cell_embs = torch.Tensor(cell_embs).to(self.model.device)

        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 1000.0

        gene_embeds = self.get_gene_embedding(genes)
        for i in tqdm(range(0, cell_embs.size(0), batch_size), total=int(cell_embs.size(0) // batch_size)):
            cell_embeds_batch = cell_embs[i : i + batch_size]
            if use_rda:
                task_counts = torch.full((cell_embeds_batch.shape[0],), read_depth, device=self.model.device)
            else:
                task_counts = None
            merged_embs = StateEmbeddingModel.resize_batch(cell_embeds_batch, gene_embeds, task_counts)
            logprobs_batch = self.model.binary_decoder(merged_embs)
            logprobs_batch = logprobs_batch.detach().cpu().numpy()
            yield logprobs_batch.squeeze()
