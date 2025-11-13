from typing import Dict, Optional

import torch
from geomloss import SamplesLoss

from .base import PerturbationModel
from .utils import build_mlp, get_activation_class, get_transformer_backbone


class OldNeuralOTPerturbationModel(PerturbationModel):
    """
    This model:
      1) Projects basal expression and perturbation encodings into a shared latent space.
      2) Uses an OT-based distributional loss (energy, sinkhorn, etc.) from geomloss.
      3) Enables cells to attend to one another, learning a set-to-set function rather than
      a sample-to-sample single-cell map.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_dim: dimension of the input expression (e.g. number of genes or embedding dimension).
            hidden_dim: not necessarily used, but required by PerturbationModel signature.
            output_dim: dimension of the output space (genes or latent).
            pert_dim: dimension of perturbation embedding.
            gpt: e.g. "TranslationTransformerSamplesModel".
            model_kwargs: dictionary passed to that model's constructor.
            loss: choice of distributional metric ("sinkhorn", "energy", etc.).
            **kwargs: anything else to pass up to PerturbationModel or not used.
        """
        # Call the parent PerturbationModel constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.distributional_loss = distributional_loss
        self.cell_sentence_len = self.transformer_backbone_kwargs["n_positions"]
        self.gene_dim = gene_dim

        # Build the distributional loss from geomloss
        self.loss_fn = SamplesLoss(loss=self.distributional_loss)
        # self.loss_fn = LearnableAlignmentLoss()

        # Build the underlying neural OT network
        self._build_networks()

    def _build_networks(self):
        """
        Here we instantiate the actual GPT2-based model or any neuralOT translator
        via your old get_model(model_key, model_kwargs) approach.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Map the input embedding to the hidden space
        self.basal_encoder = build_mlp(
            in_dim=self.input_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        print(self)

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Return the latent perturbed state given the perturbation and basal state.
        """
        pert_embedding = self.encode_perturbation(pert).unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        control_cells = self.encode_basal_expression(basal).unsqueeze(1)  # shape: [batch_size, 1, hidden_dim]
        cls_input = torch.zeros_like(pert_embedding)  # shape: [batch_size, 1, hidden_dim]
        seq_input = torch.cat([pert_embedding, control_cells, cls_input], dim=1)  # shape: [batch_size, 3, hidden_dim]

        # forward pass + extract CLS last hidden state
        prediction = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state[:, -1]

        # add to basal if predicting residual
        if self.predict_residual:
            # treat the actual prediction as a residual sum to basal
            return prediction + control_cells.squeeze(1)
        else:
            return prediction

    def forward(self, batch: dict) -> torch.Tensor:
        """
        The main forward call.
        """
        prediction = self.perturb(batch["pert_emb"], batch["ctrl_cell_emb"])
        output = self.project_out(prediction)

        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        pred = self(batch)
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        # TODO: please improve this, do not assume self.cell_sentence_len for this model
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        main_loss = self.loss_fn(pred, target).mean()
        self.log("train_loss", main_loss)

        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        pred = self(batch)
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)

        return {"loss": loss, "predictions": pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        """Track decoder performance during validation without training it."""
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            # Get model predictions from validation step
            latent_preds = outputs["predictions"]

            # Train decoder to map latent predictions to gene space
            pert_cell_counts_preds = self.gene_decoder(latent_preds)  # verify this is automatically detached
            gene_targets = batch["pert_cell_counts"]

            # Get decoder predictions
            pert_cell_counts_preds = pert_cell_counts_preds.reshape(-1, self.cell_sentence_len, self.gene_dim)
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("decoder_val_loss", decoder_loss)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self.forward(batch, padded=False)
        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)
        pred = pred.reshape(-1, self.output_dim)
        target = target.reshape(-1, self.output_dim)

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
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
