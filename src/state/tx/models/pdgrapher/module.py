from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

logger = logging.getLogger(__name__)


@dataclass
class PDGrapherModelConfig:
    """Configuration for the PDGrapher backbone."""

    positional_features_dims: int = 16
    embedding_layer_dim: int = 64
    dim_gnn: int = 64
    n_layers_gnn: int = 2
    n_layers_nn: int = 2
    dropout: float = 0.1
    mode: str = "forward"  # "forward" or "inverse"

    def validate(self) -> None:
        if self.n_layers_gnn < 1:
            raise ValueError("n_layers_gnn must be >= 1")
        if self.n_layers_nn < 1:
            raise ValueError("n_layers_nn must be >= 1")
        if self.mode not in {"forward", "inverse"}:
            raise ValueError("mode must be either 'forward' or 'inverse'")


def _load_edge_index(path: Optional[str]) -> Optional[torch.Tensor]:
    """Load an edge index tensor from a variety of file formats."""

    if path is None:
        return None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Edge index file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".pt", ".pth", ".bin"}:
        edge_index = torch.load(path)
        if isinstance(edge_index, dict) and "edge_index" in edge_index:
            edge_index = edge_index["edge_index"]
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(f"Loaded object from {path} is not a tensor")
        return edge_index

    if ext in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "edge_index" in data.files:
                edge_index = data["edge_index"]
            else:
                # Use the first array that looks like an edge index
                key = data.files[0]
                edge_index = data[key]
        else:
            edge_index = data
        tensor = torch.as_tensor(edge_index)
        if tensor.ndim != 2 or tensor.size(0) != 2:
            raise ValueError(f"Edge index in {path} has invalid shape {tensor.shape}")
        return tensor

    raise ValueError(
        f"Unsupported edge index format '{ext}'. Expected one of .pt, .pth, .bin, .npy or .npz"
    )


def _build_normalized_adjacency(
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    *,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Create a dense, symmetrically normalised adjacency matrix."""

    if edge_index is None:
        adj = torch.eye(num_nodes, dtype=torch.float32)
    else:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges]; got {edge_index.shape}")
        edge_index = edge_index.to(torch.long)
        if edge_index.numel() == 0:
            adj = torch.eye(num_nodes, dtype=torch.float32)
        else:
            src, dst = edge_index
            edges = torch.stack([src, dst], dim=0)
            if add_self_loops:
                loops = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
                loop_index = torch.stack([loops, loops])
                edges = torch.cat([edges, loop_index], dim=1)
            # Add reverse edges to make the graph undirected
            rev_edges = torch.stack([edges[1], edges[0]], dim=0)
            edges = torch.cat([edges, rev_edges], dim=1)
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            adj[edges[0].cpu(), edges[1].cpu()] = 1.0
            # Ensure the matrix is symmetric
            adj = torch.maximum(adj, adj.T)

    deg = adj.sum(dim=1)
    # Avoid division by zero
    deg = deg.clamp_min(1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return norm_adj


class SimpleGCNLayer(nn.Module):
    """A lightweight GCN layer operating on dense adjacency matrices."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply a graph convolution.

        Args:
            x: Tensor of shape [batch, num_nodes, in_dim]
            adjacency: Tensor of shape [batch, num_nodes, num_nodes]
        """

        h = self.linear(x)
        return torch.matmul(adjacency, h)


class PDGrapherBackbone(nn.Module):
    """Backbone shared by the forward and inverse PDGrapher variants."""

    def __init__(
        self,
        num_nodes: int,
        config: PDGrapherModelConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.num_nodes = num_nodes
        self.config = config

        self.primary_embed = nn.Linear(1, config.embedding_layer_dim)
        self.secondary_embed = nn.Linear(1, config.embedding_layer_dim)
        self.positional_embeddings = nn.Embedding(num_nodes, config.positional_features_dims)
        nn.init.normal_(self.positional_embeddings.weight, mean=0.0, std=1.0)

        gcn_input_dim = 2 * config.embedding_layer_dim + config.positional_features_dims
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for _ in range(config.n_layers_gnn):
            self.gcn_layers.append(SimpleGCNLayer(gcn_input_dim, config.dim_gnn))
            self.gcn_norms.append(nn.LayerNorm(config.dim_gnn + 2 * config.embedding_layer_dim))

        self.dropout = nn.Dropout(config.dropout)

        # Build the feed-forward network applied node-wise
        self.mlp_layers = nn.ModuleList()
        self.mlp_norms = nn.ModuleList()
        mlp_input_dim = config.dim_gnn + 2 * config.embedding_layer_dim
        if config.n_layers_nn == 1:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, max(config.dim_gnn // 2, 1)))
            self.mlp_norms.append(nn.LayerNorm(max(config.dim_gnn // 2, 1)))
            hidden_dim = max(config.dim_gnn // 2, 1)
        else:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, config.dim_gnn))
            self.mlp_norms.append(nn.LayerNorm(config.dim_gnn))
            hidden_dim = config.dim_gnn
            for _ in range(config.n_layers_nn - 2):
                self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.mlp_norms.append(nn.LayerNorm(hidden_dim))
            self.mlp_layers.append(nn.Linear(hidden_dim, max(hidden_dim // 2, 1)))
            self.mlp_norms.append(nn.LayerNorm(max(hidden_dim // 2, 1)))
            hidden_dim = max(hidden_dim // 2, 1)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        primary: torch.Tensor,
        secondary: torch.Tensor,
        adjacency: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_nodes = primary.shape
        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes but got tensors with {num_nodes} entries"
            )

        primary_emb = self.primary_embed(primary.unsqueeze(-1))
        secondary_emb = self.secondary_embed(secondary.unsqueeze(-1))
        positional = self.positional_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        base_features = torch.cat([primary_emb, secondary_emb], dim=-1)
        x = torch.cat([base_features, positional], dim=-1)

        # Prepare batch-specific adjacency matrices
        if intervention_mask is not None:
            mask = 1.0 - intervention_mask.float()
            adjacency = adjacency * mask.unsqueeze(1)
        h = x
        for conv, norm in zip(self.gcn_layers, self.gcn_norms):
            conv_out = conv(h, adjacency)
            h = torch.cat([base_features, conv_out], dim=-1)
            h = norm(h)
            h = F.elu(h)
            h = self.dropout(h)

        h = h.view(batch_size * num_nodes, -1)
        for layer, norm in zip(self.mlp_layers, self.mlp_norms):
            h = layer(h)
            h = norm(h)
            h = F.elu(h)
            h = self.dropout(h)

        out = self.output_layer(h)
        return out.view(batch_size, num_nodes)


class PDGrapherLightningModule(LightningModule):
    """Lightning integration for PDGrapher-style forward and inverse models."""

    def __init__(
        self,
        *,
        gene_names: Sequence[str],
        edge_index_path: Optional[str] = None,
        mode: str = "forward",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        positional_features_dims: int = 16,
        embedding_layer_dim: int = 64,
        dim_gnn: int = 64,
        n_layers_gnn: int = 2,
        n_layers_nn: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "edge_index_path": edge_index_path,
                "mode": mode,
                "lr": lr,
                "weight_decay": weight_decay,
                "positional_features_dims": positional_features_dims,
                "embedding_layer_dim": embedding_layer_dim,
                "dim_gnn": dim_gnn,
                "n_layers_gnn": n_layers_gnn,
                "n_layers_nn": n_layers_nn,
                "dropout": dropout,
            }
        )

        self.mode = mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_nodes = len(gene_names)
        self.gene_names = list(gene_names)
        self._gene_lookup = {name.lower(): idx for idx, name in enumerate(self.gene_names)}
        self._warned_perts: set[str] = set()

        edge_index = _load_edge_index(edge_index_path)
        adjacency = _build_normalized_adjacency(edge_index, self.num_nodes)
        self.register_buffer("adjacency", adjacency, persistent=False)

        config = PDGrapherModelConfig(
            positional_features_dims=positional_features_dims,
            embedding_layer_dim=embedding_layer_dim,
            dim_gnn=dim_gnn,
            n_layers_gnn=n_layers_gnn,
            n_layers_nn=n_layers_nn,
            dropout=dropout,
            mode=mode,
        )
        self.backbone = PDGrapherBackbone(self.num_nodes, config)

        if mode == "forward":
            self.loss_fn = nn.MSELoss()
        elif mode == "inverse":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError("mode must be 'forward' or 'inverse'")

        # Attributes expected by downstream tooling
        self.cell_sentence_len = 1
        self.output_space = "gene"

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _ensure_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            # collapse sentence dimension if present
            value = value.view(-1, value.size(-1))
        if value.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {value.shape}")
        return value.float()

    def _prepare_batch(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, torch.Tensor]:
        ctrl = self._ensure_tensor(batch["ctrl_cell_emb"])
        if self.mode == "forward":
            target = batch.get("pert_cell_counts")
            if target is None:
                target = batch.get("pert_cell_emb")
            if target is None:
                raise KeyError("Batch does not contain perturbed expression (pert_cell_emb or pert_cell_counts)")
            target = self._ensure_tensor(target)
        else:
            target = None

        treated = None
        if "pert_cell_emb" in batch:
            treated = self._ensure_tensor(batch["pert_cell_emb"])
        elif "pert_cell_counts" in batch:
            treated = self._ensure_tensor(batch["pert_cell_counts"])

        pert_names = batch.get("pert_name")
        if pert_names is None:
            pert_names = ["" for _ in range(ctrl.size(0))]

        intervention = self._build_intervention_matrix(pert_names, batch.get("pert_emb"))

        return {
            "ctrl": ctrl,
            "treated": treated,
            "target": target,
            "intervention": intervention,
        }

    def _build_intervention_matrix(
        self,
        pert_names: Sequence[str],
        pert_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = len(pert_names)
        mask = torch.zeros((batch_size, self.num_nodes), device=self.device)
        tensor_emb = None
        if pert_emb is not None:
            tensor_emb = pert_emb.to(self.device)

        for i, name in enumerate(pert_names):
            indices = self._resolve_gene_indices(name)
            if not indices and tensor_emb is not None:
                emb_tensor = tensor_emb[i].float()
                if emb_tensor.ndim == 1 and emb_tensor.numel() == self.num_nodes:
                    indices = torch.where(emb_tensor > 0)[0].tolist()
            if not indices:
                key = str(name)
                if key not in self._warned_perts:
                    logger.warning("Could not map perturbation '%s' to gene index", key)
                    self._warned_perts.add(key)
                continue
            mask[i, indices] = 1.0
        return mask

    def _resolve_gene_indices(self, pert_name: str | Sequence[str]) -> List[int]:
        if isinstance(pert_name, (list, tuple)):
            indices: List[int] = []
            for item in pert_name:
                indices.extend(self._resolve_gene_indices(item))
            return indices

        if pert_name is None:
            return []

        name = str(pert_name)
        tokens = re.split(r"[+;,/|]+", name.replace(" ", "+"))
        indices: List[int] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            token_lower = token.lower()
            if token_lower in self._gene_lookup:
                indices.append(self._gene_lookup[token_lower])
            elif token_lower.endswith("_tf") and token_lower[:-3] in self._gene_lookup:
                indices.append(self._gene_lookup[token_lower[:-3]])
            elif token_lower.endswith("_target") and token_lower[:-7] in self._gene_lookup:
                indices.append(self._gene_lookup[token_lower[:-7]])
        return indices

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------
    def _forward_impl(self, batch: dict[str, torch.Tensor | list[str]]):
        batch_tensors = self._prepare_batch(batch)
        ctrl = batch_tensors["ctrl"].to(self.device)
        intervention = batch_tensors["intervention"].to(self.device)
        treated = batch_tensors.get("treated")
        target = batch_tensors.get("target")
        if treated is not None:
            treated = treated.to(self.device)
        if target is not None:
            target = target.to(self.device)

        adjacency = self.adjacency.to(self.device)
        adjacency = adjacency.unsqueeze(0).expand(ctrl.size(0), -1, -1)

        if self.mode == "forward":
            preds = self.backbone(ctrl, intervention, adjacency, intervention)
            return preds, target
        else:
            if treated is None:
                raise KeyError("Batch does not contain treated expression required for inverse mode")
            logits = self.backbone(ctrl, treated, adjacency, intervention)
            return logits, intervention

    def forward(self, batch: dict[str, torch.Tensor | list[str]]):
        preds, _ = self._forward_impl(batch)
        if self.mode == "inverse":
            return torch.sigmoid(preds)
        return preds

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------
    def _compute_inverse_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_matrix = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pos_counts = targets.sum(dim=1, keepdim=True)
        neg_counts = targets.size(1) - pos_counts
        pos_counts = pos_counts.clamp_min(1.0)
        weights = torch.where(targets > 0, neg_counts / pos_counts, torch.ones_like(targets))
        weighted_loss = loss_matrix * weights
        return weighted_loss.mean()

    def training_step(self, batch, batch_idx):
        preds, target = self._forward_impl(batch)
        if target is None:
            raise RuntimeError("Training target is missing")
        if self.mode == "forward":
            loss = self.loss_fn(preds, target)
        else:
            loss = self._compute_inverse_loss(preds, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, target = self._forward_impl(batch)
        if target is None:
            raise RuntimeError("Validation target is missing")
        if self.mode == "forward":
            loss = self.loss_fn(preds, target)
        else:
            loss = self._compute_inverse_loss(preds, target)
        self.log("val_loss", loss, prog_bar=True, sync_dist=False)
        return loss

    def test_step(self, batch, batch_idx):
        preds, target = self._forward_impl(batch)
        if target is None:
            raise RuntimeError("Test target is missing")
        if self.mode == "forward":
            loss = self.loss_fn(preds, target)
        else:
            loss = self._compute_inverse_loss(preds, target)
        self.log("test_loss", loss, prog_bar=True, sync_dist=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        preds, target = self._forward_impl(batch)
        output = {
            "preds": torch.sigmoid(preds) if self.mode == "inverse" else preds,
            "pert_name": batch.get("pert_name"),
            "celltype_name": batch.get("cell_type"),
            "batch": batch.get("batch"),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb"),
        }
        if self.mode == "forward":
            output["pert_cell_emb"] = target
        else:
            output["intervention_target"] = target
        return output
