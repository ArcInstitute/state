# src/state/emb/finetune_decoder.py

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from omegaconf import OmegaConf

from vci.nn.model import StateEmbeddingModel
from vci.train.trainer import get_embeddings
from vci.utils import get_embedding_cfg

log = logging.getLogger(__name__)


class Finetune(nn.Module):
    def __init__(
        self,
        cfg: Optional[OmegaConf] = None,
        learning_rate: float = 1e-4,
        read_depth: float = 4.0,
        train_binary_decoder: bool = False,
    ):
        """
        Helper module that loads a pretrained SE/VCI checkpoint and exposes:
          - get_gene_embedding(genes): returns gene/task embeddings with differentiable
            replacement for any genes missing from pretrained protein embeddings
          - get_counts(cell_embs, genes): runs the pretrained binary decoder in a vectorized way

        Args:
            cfg: OmegaConf for the SE model (if not embedded in checkpoint)
            learning_rate: (kept for API compatibility; not used directly here)
            read_depth: initial value for a learnable read depth scalar (if RDA enabled)
        """
        super().__init__()
        self.model: Optional[StateEmbeddingModel] = None
        self.collator = None
        self.protein_embeds: Optional[Dict[str, torch.Tensor]] = None
        self._vci_conf = cfg
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_binary_decoder = train_binary_decoder

        # --- Learnable read-depth scalar used when RDA is enabled ---
        self.read_depth = nn.Parameter(torch.tensor(float(read_depth), dtype=torch.float), requires_grad=True)

        # --- Caching & state for gene embeddings and missing-gene handling ---
        self.cached_gene_embeddings: Dict[Tuple[str, ...], torch.Tensor] = {}

        self.missing_table: Optional[nn.Embedding] = None
        self._last_missing_count: int = 0
        self._last_missing_dim: int = 0

        # Cache present masks and index maps per gene set
        self._present_mask_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self._missing_index_map_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    # -------------------------
    # Loading / setup
    # -------------------------
    def load_model(self, checkpoint: str):
        """
        Load a pre-trained SE model from a single checkpoint path and prepare it.
        Prefers embedded cfg in checkpoint; falls back to provided cfg if needed.
        """
        if self.model is not None:
            raise ValueError("Model already initialized")

        # Resolve configuration: prefer embedded cfg in checkpoint
        cfg_to_use = self._vci_conf
        if cfg_to_use is None:
            try:
                ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "cfg_yaml" in ckpt:
                    cfg_to_use = OmegaConf.create(ckpt["cfg_yaml"])  # type: ignore
                elif isinstance(ckpt, dict) and "hyper_parameters" in ckpt:
                    hp = ckpt.get("hyper_parameters", {}) or {}
                    if isinstance(hp, dict) and len(hp) > 0:
                        try:
                            cfg_to_use = OmegaConf.create(hp["cfg"]) if "cfg" in hp else OmegaConf.create(hp)
                        except Exception:
                            cfg_to_use = OmegaConf.create(hp)
            except Exception as e:
                log.warning(f"Could not extract config from checkpoint: {e}")
        if cfg_to_use is None:
            raise ValueError(
                "No config found in checkpoint and no override provided. "
                "Provide SE cfg or a full checkpoint with embedded config."
            )
        self._vci_conf = cfg_to_use

        # Load model; allow passing cfg to constructor like inference
        self.model = StateEmbeddingModel.load_from_checkpoint(checkpoint, dropout=0.0, strict=False, cfg=self._vci_conf)
        self.device = self.model.device  # type: ignore

        # Try to extract packaged protein embeddings from checkpoint
        packaged_pe = None
        try:
            ckpt2 = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt2, dict) and "protein_embeds_dict" in ckpt2:
                packaged_pe = ckpt2["protein_embeds_dict"]
        except Exception:
            pass

        # Resolve protein embeddings for pe_embedding weights
        all_pe = packaged_pe or get_embeddings(self._vci_conf)
        if isinstance(all_pe, dict):
            # For the model's token embedding table, we only need the stacked array.
            stacked = torch.vstack(list(all_pe.values()))
        else:
            stacked = all_pe
        stacked.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(stacked)  # type: ignore
        self.model.pe_embedding.to(self.device)  # type: ignore

        # Keep a mapping from gene name -> raw protein embedding vector
        self.protein_embeds = packaged_pe
        if self.protein_embeds is None:
            # Fallback to configured path on disk
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings, weights_only=False)

        # Freeze SE model; optionally unfreeze just the binary decoder
        for p in self.model.parameters():  # type: ignore
            p.requires_grad = False
        for p in self.model.binary_decoder.parameters():  # type: ignore
            p.requires_grad = self.train_binary_decoder
        self.model.binary_decoder.train(mode=self.train_binary_decoder)  # type: ignore

    # -------------------------
    # Gene utilities
    # -------------------------
    def _auto_detect_gene_column(self, adata):
        """Auto-detect the gene column with highest overlap with protein embeddings."""
        if self.protein_embeds is None:
            log.warning("No protein embeddings available for auto-detection, using index")
            return None

        protein_genes = set(self.protein_embeds.keys())
        best_column = None
        best_overlap = 0

        # Index first
        index_genes = set(getattr(adata.var, "index", []))
        overlap = len(protein_genes.intersection(index_genes))
        if overlap > best_overlap:
            best_overlap = overlap
            best_column = None  # None => use index

        # All columns
        for col in adata.var.columns:
            try:
                col_vals = adata.var[col].dropna().astype(str)
            except Exception:
                continue
            col_genes = set(col_vals)
            overlap = len(protein_genes.intersection(col_genes))
            if overlap > best_overlap:
                best_overlap = overlap
                best_column = col

        return best_column

    def genes_from_adata(self, adata) -> List[str]:
        """Return list of gene names from AnnData using auto-detected column/index."""
        col = self._auto_detect_gene_column(adata)
        if col is None:
            return list(map(str, adata.var.index.values))
        return list(adata.var[col].astype(str).values)

    def _ensure_missing_table(
        self,
        genes_key: Tuple[str, ...],
        gene_embed_dim: int,
        present_mask: torch.Tensor,
    ):
        """
        Make sure self.missing_table matches the current gene set's missing count & dim.
        Builds a per-position index map (for missing genes) and caches the mask + map.
        """
        # Build / cache index map for missing positions (pos -> 0..(n_missing-1))
        if genes_key in self._missing_index_map_cache and genes_key in self._present_mask_cache:
            return  # already prepared for this gene set

        # Identify missing positions
        present = present_mask.bool().tolist()
        missing_positions = [i for i, ok in enumerate(present) if not ok]
        n_missing = len(missing_positions)

        # Cache mask for this gene set (on device)
        self._present_mask_cache[genes_key] = present_mask

        if n_missing == 0:
            # No missing genes -> trivial index map of zeros (unused)
            self._missing_index_map_cache[genes_key] = torch.zeros(len(genes_key), dtype=torch.long, device=self.device)
            return

        # (Re)create the missing table if shape changed
        if (
            self.missing_table is None
            or self._last_missing_count != n_missing
            or self._last_missing_dim != gene_embed_dim
        ):
            self.missing_table = nn.Embedding(n_missing, gene_embed_dim)
            nn.init.normal_(self.missing_table.weight, mean=0.0, std=0.02)
            # Ensure the embedding table lives on the same device as inputs/masks
            self.missing_table.to(present_mask.device)
            self._last_missing_count = n_missing
            self._last_missing_dim = gene_embed_dim

        # Build a position -> compact missing index map
        inv = {pos: j for j, pos in enumerate(missing_positions)}
        idx_map = [inv.get(i, 0) for i in range(len(genes_key))]
        self._missing_index_map_cache[genes_key] = torch.tensor(idx_map, dtype=torch.long, device=present_mask.device)

    def get_gene_embedding(self, genes: List[str]) -> torch.Tensor:
        """
        Return gene/task embeddings for 'genes'.
        For genes missing from the pretrained protein embeddings dictionary, we replace the
        post-ESM embedding with a learnable vector from `self.missing_table` via torch.where.

        Caching:
          - If no genes are missing, the post-ESM embeddings are cached and reused.
          - If some genes are missing, we recompute each call so gradients flow into
            self.missing_table (no caching of the final tensor).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model(checkpoint) first.")
        if self.protein_embeds is None:
            # Should have been set in load_model; keep a defensive fallback:
            self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings, weights_only=False)

        genes_key = tuple(genes)

        # Fast path: if we saw this gene set before and no missing genes were involved, reuse cache
        if genes_key in self.cached_gene_embeddings:
            return self.cached_gene_embeddings[genes_key].to(self.device)

        # Build a [G, embed_size] tensor of raw protein embeddings (zeros where missing)
        # Determine the raw protein embedding size
        try:
            example_vec = next(iter(self.protein_embeds.values()))
            embed_size = int(example_vec.shape[-1])
        except Exception:
            embed_size = get_embedding_cfg(self._vci_conf).size  # fallback

        raw_list = [
            self.protein_embeds[g] if g in self.protein_embeds else torch.zeros(embed_size)  # type: ignore
            for g in genes
        ]
        protein_embeds = torch.stack(raw_list).to(self.device)

        # Project through the model's gene embedding layer (post-ESM projection)
        gene_embeds_raw = self.model.gene_embedding_layer(protein_embeds)  # type: ignore  # [G, d_model]
        gene_embeds_raw = gene_embeds_raw.to(self.device)
        d_model = int(gene_embeds_raw.shape[-1])

        # Present mask: True where gene exists in pretrained protein_embeds
        present_mask = torch.tensor([g in self.protein_embeds for g in genes], device=self.device).unsqueeze(1)

        # Prepare missing-table and position index map if needed
        self._ensure_missing_table(genes_key, d_model, present_mask.squeeze(1))

        # If we have a non-empty missing_table for this gene set, replace missing rows
        idx_map = self._missing_index_map_cache[genes_key]
        # Safety: if the missing table exists but is on a different device, move it
        if self.missing_table is not None and self.missing_table.weight.device != idx_map.device:
            self.missing_table.to(idx_map.device)
        if self.missing_table is not None and self._last_missing_count > 0:
            learned_full = self.missing_table(idx_map)  # [G, d_model]
            # Differentiable replacement: keep present rows from gene_embeds_raw, else take learned_full
            gene_embeds = torch.where(present_mask, gene_embeds_raw, learned_full)
        else:
            gene_embeds = gene_embeds_raw

        # Cache only when there are no missing genes for this set (so the tensor is static)
        if self._last_missing_count == 0:
            self.cached_gene_embeddings[genes_key] = gene_embeds.detach().clone()

        return gene_embeds

    # -------------------------
    # Counts decoding (vectorized over genes)
    # -------------------------
    def get_counts(self, cell_embs, genes: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Generate predictions with the (pretrained) binary decoder. This is vectorized
        over all genes (no per-gene loops).

        Returns:
            Tensor of shape [Ncells, Ngenes]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model(checkpoint) first.")

        # Convert cell_embs to a tensor on the correct device (no detach here)
        cell_embs = torch.as_tensor(cell_embs, dtype=torch.float, device=self.device)

        # RDA must be enabled to use read_depth
        use_rda = getattr(self.model.cfg.model, "rda", False)  # type: ignore
        assert use_rda, "RDA must be enabled to use get_counts (model.cfg.model.rda == True)."

        # Retrieve (and possibly learn) gene embeddings (with differentiable missing replacement)
        gene_embeds = self.get_gene_embedding(genes)  # [G, d_model]

        outputs = []
        for i in range(0, cell_embs.size(0), batch_size):
            end_idx = min(i + batch_size, cell_embs.size(0))
            cell_batch = cell_embs[i:end_idx]  # [B, E_cell]

            # NOTE: Learnable read depth scalar, expanded to batch (keeps gradient)
            task_counts = self.read_depth.expand(cell_batch.shape[0]).to(cell_batch.dtype).to(cell_batch.device)

            # Build [B, G, *] pairwise features and decode
            merged = self.model.resize_batch(cell_batch, gene_embeds, task_counts)  # type: ignore

            # Align dtype with decoder weights to avoid mixed-precision issues on CPU
            dec_param_dtype = next(self.model.binary_decoder.parameters()).dtype  # type: ignore
            if merged.dtype != dec_param_dtype:
                merged = merged.to(dec_param_dtype)

            logprobs_batch = self.model.binary_decoder(merged)  # type: ignore

            # Squeeze trailing singleton if present: [B, G, 1] -> [B, G]
            if logprobs_batch.dim() == 3 and logprobs_batch.size(-1) == 1:
                logprobs_batch = logprobs_batch.squeeze(-1)

            outputs.append(logprobs_batch)

        return torch.cat(outputs, dim=0)
