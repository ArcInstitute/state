import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from .utils import get_loss_fn

logger = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.

    This takes concat([cell embedding]) as the input, and predicts
    counts over all genes as output.

    This decoder is trained separately from the main perturbation model.

    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        residual_decoder: If True, adds residual connections between every other layer block
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder=False,
    ):
        super().__init__()

        self.residual_decoder = residual_decoder

        if residual_decoder:
            # Build individual blocks for residual connections
            self.blocks = nn.ModuleList()
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
                )
                self.blocks.append(block)
                input_dim = hidden_dim

            # Final output layer
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            # Original implementation without residual connections
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
            # Make sure outputs are non-negative
            layers.append(nn.ReLU())

            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
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
        if self.residual_decoder:
            # Apply blocks with residual connections between every other block
            block_outputs = []
            current = x

            for i, block in enumerate(self.blocks):
                output = block(current)

                # Add residual connection from every other previous block
                # Pattern: blocks 1, 3, 5, ... get residual from blocks 0, 2, 4, ...
                if i >= 1 and i % 2 == 1:  # Odd-indexed blocks (1, 3, 5, ...)
                    residual_idx = i - 1  # Previous even-indexed block
                    output = output + block_outputs[residual_idx]

                block_outputs.append(output)
                current = output

            return self.final_layer(current)
        else:
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
        gene_decoder_lr: Learning rate for gene decoder
        loss_fn: Loss function ('mse' or custom nn.Module)
        output_space: 'gene' or 'latent'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.1,
        lr: float = 1e-4,
        gene_decoder_lr: float = 1e-5,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        gene_names: Optional[List[str]] = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        hvg_dim: int = 2001,
        scheduler_type: Optional[str] = None,
        scheduler_step_size: int = 50,
        scheduler_gamma: float = 0.1,
        scheduler_T_max: int = 100,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_monitor: str = "val_loss",
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
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

        self.residual_decoder = kwargs.get("residual_decoder", False)

        self.embed_key = embed_key
        self.output_space = output_space
        self.batch_size = batch_size
        self.control_pert = control_pert

        # Training settings
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.gene_decoder_lr = gene_decoder_lr
        self.loss_fn = get_loss_fn(loss_fn)

        # Scheduler settings
        self.scheduler_type = scheduler_type
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_T_max = scheduler_T_max
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_monitor = scheduler_monitor

        # Warmup parameters
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor

        # this will either decode to hvg space if output space is a gene,
        # or to transcriptome space if output space is all. done this way to maintain
        # backwards compatibility with the old models
        self.gene_decoder = None
        gene_dim = hvg_dim if output_space == "gene" else gene_dim
        if (embed_key and embed_key != "X_hvg" and output_space == "gene") or (
            embed_key and output_space == "all"
        ):  # we should be able to decode from hvg to all
            if gene_dim > 10000:
                hidden_dims = [1024, 512, 256]
            else:
                if "DMSO_TF" in self.control_pert:
                    if self.residual_decoder:
                        hidden_dims = [2058, 2058, 2058, 2058, 2058]
                    else:
                        hidden_dims = [4096, 2048, 2048]
                elif "PBS" in self.control_pert:
                    hidden_dims = [2048, 1024, 1024]
                else:
                    hidden_dims = [1024, 1024, 512]  # make this config

            self.gene_decoder = LatentToGeneDecoder(
                latent_dim=self.output_dim,
                gene_dim=gene_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                residual_decoder=self.residual_decoder,
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
        main_loss = self.loss_fn(pred, batch["pert_cell_emb"])
        self.log("train_loss", main_loss)

        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets)

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        pred = self(batch)
        loss = self.loss_fn(pred, batch["pert_cell_emb"])

        # TODO: remove unused
        # is_control = self.control_pert in batch["pert_name"]
        self.log("val_loss", loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        latent_output = self(batch)
        target = batch[self.embed_key]
        loss = self.loss_fn(latent_output, target)

        output_dict = {
            "preds": latent_output,  # The distribution's sample
            "pert_cell_emb": batch.get("pert_cell_emb", None),  # The target gene expression or embedding
            "pert_cell_counts": batch.get("pert_cell_counts", None),  # the true, raw gene expression
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds
            decoder_loss = self.loss_fn(pert_cell_counts_preds, batch["pert_cell_counts"])
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
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

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
            pert_cell_counts_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                # Add basal expression if provided
                pert_cell_counts_preds += basal_expr
            return pert_cell_counts_preds
        return None

    def configure_optimizers(self):
        """
        Configure optimizer and optional scheduler for both the main model and the gene decoder.
        
        Supports the following scheduler types:
        - 'step': StepLR - reduces LR by gamma every step_size epochs
        - 'cosine': CosineAnnealingLR - cosine annealing schedule
        - 'plateau': ReduceLROnPlateau - reduces LR when metric plateaus
        - 'exponential': ExponentialLR - exponential decay
        - 'linear': LinearLR - linear decay over T_max epochs
        
        Warmup is supported for all scheduler types except 'plateau'.
        When warmup_epochs > 0, learning rate starts at warmup_start_factor * lr
        and linearly increases to lr over warmup_epochs, then follows the main schedule.
        """
        # Configure optimizer (same as before)
        if self.gene_decoder is not None:
            # Get gene decoder parameters
            gene_decoder_params = list(self.gene_decoder.parameters())

            # Get all other parameters (main model)
            main_model_params = [
                param for name, param in self.named_parameters()
                if not name.startswith("gene_decoder.")
            ]

            # Create parameter groups with different learning rates
            param_groups = [
                {"params": main_model_params, "lr": self.lr},
                {"params": gene_decoder_params, "lr": self.gene_decoder_lr},
            ]

            optimizer = torch.optim.Adam(param_groups)
        else:
            # Use single learning rate if no gene decoder
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # If no scheduler specified, return just the optimizer
        if self.scheduler_type is None:
            return optimizer
        
        # Helper function to create warmup + main scheduler
        def create_scheduler_with_warmup(main_scheduler):
            if self.warmup_epochs > 0:
                # Create warmup scheduler
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.warmup_start_factor,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs
                )
                
                # Combine warmup + main scheduler
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_epochs]
                )
                return scheduler
            else:
                return main_scheduler
        
        # Configure scheduler based on type
        if self.scheduler_type == "step":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.scheduler_step_size, 
                gamma=self.scheduler_gamma
            )
            scheduler = create_scheduler_with_warmup(main_scheduler)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        
        elif self.scheduler_type == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.scheduler_T_max
            )
            scheduler = create_scheduler_with_warmup(main_scheduler)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        
        elif self.scheduler_type == "plateau":
            # Note: Warmup with plateau scheduler is tricky because plateau
            # schedules based on metrics, not epochs. We'll use a custom approach.
            if self.warmup_epochs > 0:
                logger.warning(
                    "Warmup with ReduceLROnPlateau is not directly supported. "
                    "Consider using a different scheduler type for warmup."
                )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        
        elif self.scheduler_type == "exponential":
            main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.scheduler_gamma
            )
            scheduler = create_scheduler_with_warmup(main_scheduler)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        
        elif self.scheduler_type == "linear":
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.scheduler_T_max
            )
            scheduler = create_scheduler_with_warmup(main_scheduler)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                }
            }
        
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}. Using no scheduler.")
            return optimizer
