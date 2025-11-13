from typing import Dict, Tuple

import torch
from torch.optim.lr_scheduler import StepLR

from ..base import PerturbationModel
from ._module import scVIModule


class SCVIPerturbationModel(PerturbationModel):
    """
    Implementation of the scVI model. The outputs are always in
    gene expression space.

    Args:
        input_dim: Dimension of input embeddings (either number of genes or latent dim from obsm key)
        hidden_dim: Dimension of hidden layers
        output_dim: Number of genes to predict
        pert_dim: Dimension of perturbation inputs (usually one-hot size)
        decode_intermediate_dim: Optional intermediate dimension for decoder
        n_encoder_layers: Number of layers in encoder (default: 2)
        n_decoder_layers: Number of layers in encoder (default: 2)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Learning rate for optimizer (default: 1e-3)
        loss_fn: Loss function (default: 'nn.MSELoss()')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        n_cell_types: int,
        n_perts: int,
        n_batches: int,
        output_space: str = "gene",
        lr=5e-4,
        wd=1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = None,
        step_size_lr: int = 45,
        do_clip_grad: bool = False,
        gradient_clip_value: float = 3.0,
        check_val_every_n_epoch: int = 5,
        **kwargs,
    ):
        # Register with parent constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            output_space=output_space,
            **kwargs,
        )

        # Set class specific parameters before registering with parent constructor
        self.n_cell_types = n_cell_types
        self.n_perts = n_perts
        self.n_batches = n_batches

        self.n_layers_encoder = kwargs.get("n_layers_encoder", 2)
        self.n_layers_decoder = kwargs.get("n_layers_decoder", 2)
        self.n_hidden_encoder = kwargs.get("n_hidden_encoder", 256)
        self.n_hidden_decoder = kwargs.get("n_hidden_decoder", 256)
        self.n_latent = kwargs.get("n_latent", 64)
        self.recon_loss = kwargs.get("recon_loss", "nb")

        self.use_batch_norm = kwargs.get("use_batch_norm", "both")
        self.use_layer_norm = kwargs.get("use_layer_norm", "none")

        self.pert_embeddings = None  # will be set in _build_networks

        self.dropout_rate_encoder = kwargs.get("dropout_rate_encoder", 0.0)
        self.dropout_rate_decoder = kwargs.get("dropout_rate_decoder", 0.0)
        self.seed = kwargs.get("seed", 0)

        # training params
        self.lr = lr
        self.wd = wd
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        self.step_size_lr = step_size_lr
        self.do_clip_grad = do_clip_grad
        self.gradient_clip_value = gradient_clip_value
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.kwargs = kwargs

        assert self.output_space in ["gene", "all"], "scVI model only supports gene-level or all-level output"

        # Build model components
        self._build_networks()

    def _build_networks(self):
        """
        Build the core components:
        """
        self.module = scVIModule(
            n_genes=self.input_dim,
            n_perts=self.n_perts,
            n_cell_types=self.n_cell_types,
            n_batches=self.n_batches,
            pert_embeddings=self.pert_embeddings,
            n_latent=self.n_latent,
            recon_loss=self.recon_loss,
            n_hidden_encoder=self.n_hidden_encoder,
            n_layers_encoder=self.n_layers_encoder,
            n_hidden_decoder=self.n_hidden_decoder,
            n_layers_decoder=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            dropout_rate_encoder=self.dropout_rate_encoder,
            dropout_rate_decoder=self.dropout_rate_decoder,
            seed=self.seed,
        )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Map perturbation to an effect vector in embedding space."""
        raise NotImplementedError("Perturbation encoding not supported for scVI model")

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Expression is already in embedding space, pass through."""
        raise NotImplementedError("Basal expression encoding not supported for scVI model")

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Given a perturbation and basal embeddings, compute the perturbed embedding.
        """
        # Project perturbation and basal cell state to latent space
        raise NotImplementedError("Perturbation not supported for scVI model")

    @property
    def kl_weight(self):
        slope = 1.0
        if self.n_steps_kl_warmup is not None:
            global_step = self.global_step

            if global_step <= self.n_steps_kl_warmup:
                proportion = global_step / self.n_steps_kl_warmup
                return slope * proportion
            else:
                return slope
        elif self.n_epochs_kl_warmup is not None:
            current_epoch = self.current_epoch

            if current_epoch <= self.n_epochs_kl_warmup:
                proportion = current_epoch / self.n_epochs_kl_warmup
                return slope * proportion
            else:
                return slope
        else:
            return slope

    def extract_batch_tensors(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_pert = batch["pert_cell_emb"]
        x_basal = batch["ctrl_cell_emb"]
        pert = batch["pert_emb"]
        cell_type = batch["cell_type_onehot"]
        batch_ids = batch["batch"]

        # if pert is one-hot, convert to index
        if pert.dim() == 2 and pert.size(1) == self.n_perts:
            pert = pert.argmax(1)

        if cell_type.dim() == 2 and cell_type.size(1) == self.n_cell_types:
            cell_type = cell_type.argmax(1)

        if batch_ids.dim() == 2 and batch_ids.size(1) == self.n_batches:
            batch_ids = batch_ids.argmax(1)

        return x_pert, x_basal, pert, cell_type, batch_ids

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Given

        Args:
            batch: Dictionary containing:
                - pert: Perturbation one-hot
                - basal: Control expression embedding
                - cell_type: Cell type one-hot
                - batch: Batch one-hot
        """
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(x_basal, pert, cell_type, batch_ids)

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        output = getattr(decoder_outputs["px"], output_key)

        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic."""
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(x_basal, pert, cell_type, batch_ids)

        recon_loss, kl_loss = self.module.loss(
            x_pert=x_pert,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        loss = recon_loss + self.kl_weight * kl_loss

        r2_mean, r2_lfc = self.module.r2_metric(
            x_pert=x_pert,
            x_basal=x_basal,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        self.log("KL_weight", self.kl_weight, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        self.log(
            "recon_loss",
            recon_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "r2_mean",
            r2_mean,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "r2_lfc",
            r2_lfc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        if self.global_step % self.step_size_lr * 1000 == 0:
            sch = self.lr_schedulers()
            sch.step()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(x_basal, pert, cell_type, batch_ids)

        recon_loss, kl_loss = self.module.loss(
            x_pert=x_pert,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        r2_mean, r2_lfc = self.module.r2_metric(
            x_pert=x_pert,
            x_basal=x_basal,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        self.log("val_loss", recon_loss + self.kl_weight * kl_loss, prog_bar=True)
        self.log("val_r2_mean", r2_mean, prog_bar=True)
        self.log("val_r2_lfc", r2_lfc, prog_bar=True)

        # is_control = "DMSO_TF" == batch["pert_name"][0] or "non-targeting" == batch["pert_name"][0]
        # if np.random.rand() < 0.1 or is_control:
        #     if self.recon_loss == "gauss":
        #         output_key = "loc"
        #     else:
        #         output_key = "mu"

        #     x_pred = getattr(decoder_outputs["px"], output_key)

        #     self._update_val_cache(batch, x_pred)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(x_basal, pert, cell_type, batch_ids)

        recon_loss, kl_loss = self.module.loss(
            x_pert=x_pert,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        loss = recon_loss + self.kl_weight * kl_loss

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        x_pred = getattr(decoder_outputs["px"], output_key)

        self.log("test_loss", loss, prog_bar=True)
        # self._update_test_cache(batch, x_pred)

        return x_pred

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(x_basal, pert, cell_type, batch_ids)

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        x_pred = getattr(decoder_outputs["px"], output_key)

        return {
            "preds": torch.nan_to_num(torch.log(x_pred + 1), nan=0.0, posinf=1e4, neginf=0.0),
            "pert_cell_counts_preds": torch.nan_to_num(torch.log(x_pred + 1), nan=0.0, posinf=1e4, neginf=0.0),
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_emb": batch.get("pert_emb", None),
            "pert_name": batch.get("pert_name", None),
            "cell_type": batch.get("cell_type", None),
            "cell_type_name": batch.get("cell_type_name", None),
            "batch": batch.get("batch", None),
            "batch_name": batch.get("batch_name", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

    def configure_optimizers(self):
        """Set up optimizer."""
        ae_params = (
            list(filter(lambda p: p.requires_grad, self.module.encoder.parameters()))
            + list(filter(lambda p: p.requires_grad, self.module.decoder.parameters()))
            + list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.pert_embeddings.parameters(),
                )
            )
            + list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.cell_type_embeddings.parameters(),
                )
            )
            + list(filter(lambda p: p.requires_grad, self.module.batch_embeddings.parameters()))
        )

        if self.module.recon_loss in ["zinb", "nb"]:
            ae_params += [self.module.px_r]

        optimizer_autoencoder = torch.optim.Adam(ae_params, lr=self.lr, weight_decay=self.wd)

        scheduler_autoencoder = StepLR(optimizer_autoencoder, step_size=self.step_size_lr, gamma=0.9)

        optimizers = [optimizer_autoencoder]
        schedulers = [scheduler_autoencoder]

        if self.step_size_lr is not None:
            return optimizers, schedulers
        else:
            return optimizers
