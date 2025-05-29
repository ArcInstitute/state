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
from models.utils import build_mlp, get_activation_class, get_transformer_backbone

logger = logging.getLogger(__name__)


class GeneWiseDecoder(nn.Module):
    """
    A gene-wise decoder that instantiates a separate MLP for each gene.

    Each gene has its own small network that maps the latent embedding
    (optionally concatenated with batch information) to a predicted count.

    Args:
        latent_dim (int): Input dimension of each per-gene decoder.
        gene_dim (int): Number of genes (i.e. number of separate decoders).
        hidden_dims (List[int], optional): Hidden layer sizes for each decoder.
        dropout (float, optional): Dropout probability.
        softplus (bool, optional): If True, apply Softplus to ensure non-negative outputs.
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.0,
        softplus: bool = False,
    ):
        super().__init__()
        self._gene_dim = gene_dim  # store gene dimension internally

        # Build one MLP per gene. Each decoder maps the latent_dim to a scalar.
        self.decoders = nn.ModuleList(
            [
                build_mlp(
                    latent_dim,
                    1,
                    hidden_dim=512,
                    n_layers=len(hidden_dims),
                    dropout=dropout,
                    activation=get_activation_class("gelu"),
                )
                for _ in range(gene_dim)
            ]
        )
        self.softplus = torch.nn.ReLU() if softplus else None

    def gene_dim(self):
        """Return the output gene dimension for compatibility."""
        return self._gene_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all per-gene decoders.

        x can be of shape (B, latent_dim) or (B, S, latent_dim).
        The output will be of shape (B, gene_dim) or (B, S, gene_dim) respectively.
        """
        # Handle the case of a sequence of latent vectors (e.g. when padded)
        if x.dim() == 3:  # x has shape (B, S, latent_dim)
            B, S, latent_dim = x.shape
            x_flat = x.reshape(-1, latent_dim)  # Flatten to (B*S, latent_dim)
            # Compute each gene’s prediction over the flattened batch
            outputs = [decoder(x_flat) for decoder in self.decoders]  # each is (B*S, 1)
            out = torch.cat(outputs, dim=1)  # (B*S, gene_dim)
            out = out.reshape(B, S, self._gene_dim)
        else:
            # x has shape (B, latent_dim)
            outputs = [decoder(x) for decoder in self.decoders]  # list of (B, 1)
            out = torch.cat(outputs, dim=1)  # (B, gene_dim)

        if self.softplus is not None:
            out = self.softplus(out)
        return out


class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.

    This takes concat([cell embedding, batch onehot]) as the input, and predicts
    counts over all genes as output.

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
        dropout: float = 0.0,
        softplus: bool = False,
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

        # Just add a relu function for now
        if softplus:
            layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        for module in reversed(self.decoder):
            if isinstance(module, nn.Linear):
                return module.out_features
        return None

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
        batch_dim: int = None,
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
        hvg_dim: int = 2001,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core architecture settings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        if kwargs.get("batch_encoder", False):
            self.batch_dim = batch_dim
        else:
            self.batch_dim = None
        self.embed_key = embed_key
        self.output_space = output_space
        self.batch_size = batch_size
        self.control_pert = control_pert

        # Training settings
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)

        # this will either decode to hvg space if output space is a gene,
        # or to transcriptome space if output space is all. done this way to maintain
        # backwards compatibility with the old models
        self.gene_decoder = None
        gene_dim = hvg_dim if output_space == "gene" else gene_dim
        if (embed_key and embed_key != "X_hvg" and output_space == "gene") or (
            embed_key and output_space == "all"
        ):  # we should be able to decode from hvg to all
            if embed_key == "X_scfound":
                if gene_dim > 18000:
                    hidden_dims = [512, 1024, 256]
                else:
                    hidden_dims = [512, 1024]
            elif gene_dim > 18000:  # paper tahoe
                hidden_dims = [1024, 512, 256]
            elif gene_dim > 10000:  # paper replogle
                hidden_dims = [hidden_dim * 2, hidden_dim * 4]  # remove this
            else:
                hidden_dims = [hidden_dim * 2, hidden_dim * 4]

            self.gene_decoder = LatentToGeneDecoder(
                latent_dim=self.output_dim + self.batch_dim if self.batch_dim is not None else self.output_dim,
                # latent_dim=self.output_dim,
                gene_dim=gene_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                softplus=self.hparams.get("softplus", False),
            )
            logger.info(f"Initialized gene decoder for embedding {embed_key} to gene space")

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""
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
        print("self.gene_decoder is not None: ", self.gene_decoder is not None)
        print("X_hvg in batch: ", "X_hvg" in batch)
        if self.gene_decoder is not None and "X_hvg" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            gene_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["X_hvg"]
            decoder_loss = self.loss_fn(gene_preds, gene_targets)

            # Log decoder loss
            print("Decoder loss: ", decoder_loss)
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
        self.log("val_loss", loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        latent_output = self(batch)
        target = batch[self.embed_key]
        loss = self.loss_fn(latent_output, target)

        output_dict = {
            "preds": latent_output,  # The distribution's sample
            "X": batch.get("X", None),  # The target gene expression or embedding
            "X_hvg": batch.get("X_hvg", None),  # the true, raw gene expression
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "gem_group": batch.get("gem_group", None),
            "basal": batch.get("basal", None),
        }

        if self.gene_decoder is not None:
            gene_preds = self.gene_decoder(latent_output)
            output_dict["gene_preds"] = gene_preds
            decoder_loss = self.loss_fn(gene_preds, batch["X_hvg"])
            self.log("test_decoder_loss", decoder_loss, prog_bar=True)

        self.log("test_loss", loss, prog_bar=True)

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

    def decode_to_gene_space(self, latent_embeds: torch.Tensor, basal_expr: None) -> torch.Tensor:
        """
        Decode latent embeddings to gene expression space.

        Args:
            latent_embeds: Embeddings in latent space

        Returns:
            Gene expression predictions or None if decoder is not available
        """
        if self.gene_decoder is not None:
            gene_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                # Add basal expression if provided
                gene_preds += basal_expr
            return gene_preds
        return None

    def configure_optimizers(self):
        """
        Configure a single optimizer for both the main model and the gene decoder.
        """
        # Use a single optimizer for all parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
