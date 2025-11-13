# File: models/decoder_only.py

import torch
from geomloss import SamplesLoss

from .base import PerturbationModel
from .utils import get_activation_class


class DecoderOnlyPerturbationModel(PerturbationModel):
    """
    DecoderOnlyPerturbationModel learns to map the ground truth latent embedding
    (provided in batch["pert_cell_emb"]) to the ground truth HVG space (batch["pert_cell_counts"]).

    Unlike the other perturbation models that compute a control mapping (e.g. via a mapping strategy),
    this model simply feeds the latent representation through a decoder network. The loss is computed
    between the decoder output and the target HVG expression.

    It keeps the overall architectural style (and uses the SamplesLoss loss function from geomloss)
    as in the OldNeuralOT model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        n_decoder_layers: int = 2,
        dropout: float = 0.0,
        distributional_loss: str = "energy",
        output_space: str = "gene",
        gene_dim=None,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            output_space=output_space,
            **kwargs,
        )
        self.n_decoder_layers = n_decoder_layers
        self.dropout = dropout
        self.distributional_loss = distributional_loss
        self.cell_sentence_len = kwargs["transformer_backbone_kwargs"]["n_positions"]
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.gene_dim = gene_dim

        # Use the same loss function as OldNeuralOT (e.g. using the MMD loss via geomloss)
        self.loss_fn = SamplesLoss(loss=self.distributional_loss)

    def _build_networks(self):
        pass

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass: use the ground truth latent embedding (batch["pert_cell_emb"]) as the prediction.
        """
        latent = batch["pert_cell_emb"]
        return latent

    def training_step(self, batch, batch_idx):
        """
        Training step: The decoder output is compared against the target HVG expression.
        We assume that when output_space=="gene", the target is in batch["pert_cell_counts"].
        The predictions and targets are reshaped (using a cell sentence length, if provided)
        before computing the loss.
        """
        pred = self(batch)
        # log a zero tensor
        self.log("train_loss", 0.0)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            pert_cell_counts_preds = self.gene_decoder(pred)
            pert_cell_counts_preds = pert_cell_counts_preds.reshape(-1, self.cell_sentence_len, self.gene_dim)
            gene_targets = batch["pert_cell_counts"]
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()
            self.log("decoder_loss", decoder_loss)
        else:
            self.log("decoder_loss", 0.0)
            decoder_loss = None
        return decoder_loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        self.log("val_loss", 0.0)

        return {"loss": None, "predictions": pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        preds = outputs["predictions"]

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            pert_cell_counts_preds = self.gene_decoder(preds)
            gene_targets = batch["pert_cell_counts"]
            pert_cell_counts_preds = pert_cell_counts_preds.reshape(-1, self.cell_sentence_len, self.gene_dim)
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()
            self.log("decoder_val_loss", decoder_loss)

    def test_step(self, batch, batch_idx):
        pred = self(batch)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            pert_cell_counts_preds = self.gene_decoder(pred)
            gene_targets = batch["pert_cell_counts"]
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()
            self.log("decoder_test_loss", decoder_loss)
        return {"loss": None, "predictions": pred}

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch)  # shape [B, ...]
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        pert_cell_counts_preds = self.gene_decoder(latent_output)
        output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
