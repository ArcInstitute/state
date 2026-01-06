import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from omegaconf import OmegaConf

from ...emb.finetune_decoder import Finetune

logger = logging.getLogger(__name__)


class FinetuneVCICountsDecoder(nn.Module):
    def __init__(
        self,
        genes=None,
        adata=None,
        # checkpoint: Optional[str] = "/large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt",
        # config: Optional[str] = "/large_storage/ctc/userspace/aadduri/SE-600M/config.yaml",
        checkpoint: Optional[str] = "/home/aadduri/vci_pretrain/vci_1.4.4/vci_1.4.4_v7.ckpt",
        config: Optional[str] = "/home/aadduri/vci_pretrain/vci_1.4.4/config.yaml",
        latent_dim: int = 1034,  # total input dim (cell emb + optional ds emb)
        read_depth: float = 4.0,
        ds_emb_dim: int = 10,  # dataset embedding dim at the tail of input
        hidden_dim: int = 512,
        dropout: float = 0.1,
        basal_residual: bool = False,
        train_binary_decoder: bool = True,
    ):
        super().__init__()
        # Initialize finetune helper and model from a single checkpoint
        if config is None:
            raise ValueError(
                "FinetuneVCICountsDecoder requires a VCI/SE config. Set kwargs.vci_config or env STATE_VCI_CONFIG."
            )
        self.finetune = Finetune(cfg=OmegaConf.load(config), train_binary_decoder=train_binary_decoder)
        self.finetune.load_model(checkpoint)
        # Resolve genes: prefer explicit list; else infer from anndata if provided
        if genes is None and adata is not None:
            try:
                genes = self.finetune.genes_from_adata(adata)
            except Exception as e:
                raise ValueError(f"Failed to infer genes from AnnData: {e}")
        if genes is None:
            raise ValueError("FinetuneVCICountsDecoder requires 'genes' or 'adata' to derive gene names")
        self.genes = genes
        # Keep read_depth as a learnable parameter so decoded counts can adapt
        self.read_depth = nn.Parameter(torch.tensor(read_depth, dtype=torch.float), requires_grad=True)
        self.basal_residual = basal_residual
        self.ds_emb_dim = int(ds_emb_dim) if ds_emb_dim is not None else 0
        self.input_total_dim = int(latent_dim)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.genes)),
        )

        self.gene_decoder_proj = nn.Sequential(
            nn.Linear(len(self.genes), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(self.genes)),
        )

        self.binary_decoder = self.finetune.model.binary_decoder  # type: ignore

        # Validate that all requested genes exist in the pretrained checkpoint's embeddings
        pe = getattr(self.finetune, "protein_embeds", {})
        self.present_mask = [g in pe for g in self.genes]
        self.missing_positions = [i for i, g in enumerate(self.genes) if g not in pe]
        self.missing_genes = [self.genes[i] for i in self.missing_positions]
        total_req = len(self.genes)
        found = total_req - len(self.missing_positions)
        total_pe = len(pe) if hasattr(pe, "__len__") else -1
        miss_pct = (len(self.missing_positions) / total_req) if total_req > 0 else 0.0
        logger.info(
            f"FinetuneVCICountsDecoder gene check: requested={total_req}, found={found}, missing={len(self.missing_positions)} ({miss_pct:.1%}), all_embeddings_size={total_pe}"
        )

        # Create learnable embeddings for missing genes in the post-ESM gene embedding space
        if len(self.missing_positions) > 0:
            # Infer gene embedding output dimension by a dry-run through gene_embedding_layer
            try:
                sample_vec = next(iter(pe.values())).to(self.finetune.model.device)
                if sample_vec.dim() == 1:
                    sample_vec = sample_vec.unsqueeze(0)
                gene_embed_dim = self.finetune.model.gene_embedding_layer(sample_vec).shape[-1]
            except Exception:
                # Conservative fallback
                gene_embed_dim = 1024

            self.missing_table = nn.Embedding(len(self.missing_positions), gene_embed_dim)
            nn.init.normal_(self.missing_table.weight, mean=0.0, std=0.02)
            # For user visibility
            try:
                self.finetune.missing_genes = self.missing_genes
            except Exception:
                pass
        else:
            # Register a dummy buffer so attributes exist
            self.missing_table = None

        # Ensure the wrapped Finetune helper creates its own missing-table parameters
        # prior to Lightning's checkpoint load. Otherwise the checkpoint will contain
        # weights like `gene_decoder.finetune.missing_table.weight` that are absent
        # from a freshly constructed module, triggering "unexpected key" errors.
        try:
            with torch.no_grad():
                self.finetune.get_gene_embedding(self.genes)
        except Exception as exc:
            logger.debug(f"Deferred Finetune missing-table initialization failed: {exc}")

    def gene_dim(self):
        return len(self.genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, total_dim]
        if x.dim() != 3:
            x = x.unsqueeze(0)
        batch_size, seq_len, total_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, total_dim)

        # Split cell and dataset embeddings
        if self.ds_emb_dim > 0:
            cell_embeds = x_flat[:, : total_dim - self.ds_emb_dim]
            ds_emb = x_flat[:, total_dim - self.ds_emb_dim : total_dim]
        else:
            cell_embeds = x_flat
            ds_emb = None

        # Prepare gene embeddings (replace any missing with learned vectors)
        gene_embeds = self.finetune.get_gene_embedding(self.genes)
        if self.missing_table is not None and len(self.missing_positions) > 0:
            device = gene_embeds.device
            learned = self.missing_table.weight.to(device)
            idx = torch.tensor(self.missing_positions, device=device, dtype=torch.long)
            gene_embeds = gene_embeds.clone()
            gene_embeds.index_copy_(0, idx, learned)
        # Ensure embeddings live on the same device as cell_embeds
        if gene_embeds.device != cell_embeds.device:
            gene_embeds = gene_embeds.to(cell_embeds.device)

        # RDA read depth vector (if enabled in SE model)
        use_rda = getattr(self.finetune.model.cfg.model, "rda", False)
        task_counts = None
        if use_rda:
            task_counts = torch.full((cell_embeds.shape[0],), self.read_depth.item(), device=cell_embeds.device)

        # Binary decoder forward with safe dtype handling.
        # - On CUDA: enable bf16 autocast for speed.
        # - On CPU: ensure inputs match decoder weight dtype to avoid BF16/FP32 mismatch.
        device_type = "cuda" if cell_embeds.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
            merged = self.finetune.model.resize_batch(
                cell_embeds=cell_embeds, task_embeds=gene_embeds, task_counts=task_counts, ds_emb=ds_emb
            )

            # Align input dtype with decoder weights when autocast is not active (e.g., CPU path)
            dec_param_dtype = next(self.binary_decoder.parameters()).dtype
            if device_type != "cuda" and merged.dtype != dec_param_dtype:
                merged = merged.to(dec_param_dtype)

            logprobs = self.binary_decoder(merged)
            if logprobs.dim() == 3 and logprobs.size(-1) == 1:
                logprobs = logprobs.squeeze(-1)

        # Reshape back to [B, S, gene_dim]
        decoded_gene = logprobs.view(batch_size, seq_len, len(self.genes))

        # Match dtype for post-decoder projection to avoid mixed-dtype matmul
        proj_param_dtype = next(self.gene_decoder_proj.parameters()).dtype
        if decoded_gene.dtype != proj_param_dtype:
            decoded_gene = decoded_gene.to(proj_param_dtype)
        decoded_gene = decoded_gene + self.gene_decoder_proj(decoded_gene)

        # Optional residual from latent decoder (operates on full input features)
        ld_param_dtype = next(self.latent_decoder.parameters()).dtype
        x_flat_for_ld = x_flat if x_flat.dtype == ld_param_dtype else x_flat.to(ld_param_dtype)
        decoded_x = self.latent_decoder(x_flat_for_ld).view(batch_size, seq_len, len(self.genes))
        return torch.nn.functional.relu(decoded_gene + decoded_x)
