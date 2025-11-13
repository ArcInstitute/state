import math
from typing import Dict

import torch
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import accuracy

from ..base import PerturbationModel

from ._module import CPAModule


class CPAPerturbationModel(PerturbationModel):
    """
    Implementation of the CPA model. The outputs are always in
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
        encode_dosage: bool = False,
        dosage_non_linearity: str = "linear",
        lr=5e-4,
        wd=1e-6,
        n_steps_pretrain_ae: int = None,
        n_epochs_pretrain_ae: int = None,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = None,
        n_steps_adv_warmup: int = None,
        n_epochs_adv_warmup: int = None,
        adv_steps: int = 3,
        reg_adv: float = 1.0,
        pen_adv: float = 1.0,
        adv_lr=1e-3,
        adv_wd=1e-6,
        step_size_lr: int = 45,
        do_clip_grad: bool = False,
        gradient_clip_value: float = 3.0,
        adv_loss: str = "cce",
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
        self.encode_dosage = encode_dosage
        self.dosage_non_linearity = dosage_non_linearity

        self.dropout_rate_encoder = kwargs.get("dropout_rate_encoder", 0.0)
        self.dropout_rate_decoder = kwargs.get("dropout_rate_decoder", 0.0)
        self.n_hidden_adv = kwargs.get("n_hidden_adv", 128)
        self.n_layers_adv = kwargs.get("n_layers_adv", 2)
        self.use_norm_adv = kwargs.get("use_norm_adv", "batch")
        self.dropout_rate_adv = kwargs.get("dropout_rate_adv", 0.0)
        self.seed = kwargs.get("seed", 0)

        # training params
        self.lr = lr
        self.wd = wd
        self.n_steps_pretrain_ae = n_steps_pretrain_ae
        self.n_epochs_pretrain_ae = n_epochs_pretrain_ae
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.n_steps_adv_warmup = n_steps_adv_warmup
        self.n_epochs_adv_warmup = n_epochs_adv_warmup
        self.adv_steps = adv_steps
        self.reg_adv = reg_adv
        self.pen_adv = pen_adv
        self.adv_lr = adv_lr
        self.adv_wd = adv_wd

        self.step_size_lr = step_size_lr
        self.do_clip_grad = do_clip_grad
        self.gradient_clip_value = gradient_clip_value
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.kl_weight = 0.0  # disabled for now

        self.kwargs = kwargs

        self.adv_loss = adv_loss.lower()
        self.gamma = kwargs.get("gamma", 2.0)
        if self.adv_loss == "focal":
            # self.adv_loss_fn = FocalLoss(gamma=self.gamma, reduction="mean")
            raise NotImplementedError("Focal loss not implemented for CPA model yet")
        else:
            self.adv_loss_fn = torch.nn.CrossEntropyLoss()

        assert self.output_space == "gene", "CPA model only supports gene-level output"

        self.automatic_optimization = False

        # Build model components
        self._build_networks()

    def _build_networks(self):
        """
        Build the core components:
        """
        self.module = CPAModule(
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
            n_hidden_adv=self.n_hidden_adv,
            n_layers_adv=self.n_layers_adv,
            use_norm_adv=self.use_norm_adv,
            dropout_rate_adv=self.dropout_rate_adv,
            variational=False,
            encode_dosage=self.encode_dosage,
            dosage_non_linearity=self.dosage_non_linearity,
            seed=self.seed,
        )

    def _forward_step(self, batch: Dict[str, torch.Tensor]):
        """
        Given

        Args:
            batch: Dictionary containing:
                - pert: Perturbation one-hot
                - basal: Control expression embedding
                - cell_type: Cell type one-hot
                - batch: Batch one-hot
        """
        basal = batch["ctrl_cell_emb"]
        pert = batch["pert_emb"]
        cell_type = batch["cell_type_onehot"]
        batch_ids = batch["batch"]
        pert_dosages = batch.get("pert_dosage", None)

        # if pert is one-hot, convert to index
        if pert.dim() == 2 and pert.size(1) == self.n_perts:
            pert = pert.argmax(1)

        if cell_type.dim() == 2 and cell_type.size(1) == self.n_cell_types:
            cell_type = cell_type.argmax(1)

        if batch_ids.dim() == 2 and batch_ids.size(1) == self.n_batches:
            batch_ids = batch_ids.argmax(1)

        encoder_outputs, decoder_outputs = self.module.forward(basal, pert, cell_type, batch_ids, pert_dosages)

        return encoder_outputs, decoder_outputs

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Map perturbation to an effect vector in embedding space."""
        raise NotImplementedError("Perturbation encoding not supported for CPA model")

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Expression is already in embedding space, pass through."""
        raise NotImplementedError("Basal expression encoding not supported for CPA model")

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Given a perturbation and basal embeddings, compute the perturbed embedding.
        """
        # Project perturbation and basal cell state to latent space
        raise NotImplementedError("Perturbation not supported for CPA model")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoder_outputs, decoder_outputs = self._forward_step(batch)

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        x_pred = getattr(decoder_outputs["px"], output_key)

        return x_pred

    def adversarial_loss(self, perts, cell_types, batch_ids, z_basal, compute_penalty=True):
        """Computes adversarial classification losses and regularizations"""
        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        adv_logits = self.module.forward_adv(z_basal)
        pert_logits = adv_logits["pert_logits"]

        pert_adv_loss = self.adv_loss_fn(pert_logits, perts.long())
        pert_acc = accuracy(
            pert_logits.argmax(1),
            perts.long().view(
                -1,
            ),
            average="macro",
            task="multiclass",
            num_classes=self.n_perts,
        )

        cell_types_logits = adv_logits["cell_type_logits"]
        cell_types_adv_loss = self.adv_loss_fn(cell_types_logits, cell_types.long())
        cell_types_acc = accuracy(
            cell_types_logits.argmax(1),
            cell_types.long().view(
                -1,
            ),
            average="macro",
            task="multiclass",
            num_classes=self.n_cell_types,
        )

        batch_ids_logits = adv_logits["batch_logits"]
        batch_ids_adv_loss = self.adv_loss_fn(batch_ids_logits, batch_ids.long())
        batch_ids_acc = accuracy(
            batch_ids_logits.argmax(1),
            batch_ids.long().view(
                -1,
            ),
            average="macro",
            task="multiclass",
            num_classes=self.n_batches,
        )

        adv_loss = pert_adv_loss + cell_types_adv_loss + batch_ids_adv_loss
        adv_acc = (pert_acc + cell_types_acc + batch_ids_acc) / 3.0

        if compute_penalty:
            # Penalty losses
            cell_type_penalty = (
                torch.autograd.grad(
                    cell_types_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                .pow(2)
                .mean()
            )

            batch_penalty = (
                torch.autograd.grad(
                    batch_ids_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                .pow(2)
                .mean()
            )

            pert_penalty = (
                torch.autograd.grad(
                    pert_logits.sum(),
                    z_basal,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                .pow(2)
                .mean()
            )

            total_penalty = cell_type_penalty + batch_penalty + pert_penalty

        else:
            total_penalty = torch.tensor(0.0, device=z_basal.device)

        return adv_loss, adv_acc, total_penalty

    @property
    def do_start_adv_training(self):
        if self.n_steps_pretrain_ae is not None:
            return self.global_step > self.n_steps_pretrain_ae
        elif self.n_epochs_pretrain_ae is not None:
            return self.current_epoch > self.n_epochs_pretrain_ae
        else:
            return True

    @property
    def adv_lambda(self):
        slope = self.reg_adv
        if self.n_steps_adv_warmup is not None:
            global_step = self.global_step

            if self.n_steps_pretrain_ae:
                global_step -= self.n_steps_pretrain_ae

            if global_step <= self.n_steps_adv_warmup:
                proportion = global_step / self.n_steps_adv_warmup
                return slope * proportion
            else:
                return slope
        elif self.n_epochs_adv_warmup is not None:
            current_epoch = self.current_epoch

            if self.n_epochs_pretrain_ae:
                current_epoch -= self.n_epochs_pretrain_ae

            if current_epoch <= self.n_epochs_adv_warmup:
                proportion = current_epoch / self.n_epochs_adv_warmup
                return slope * proportion
            else:
                return slope
        else:
            return slope

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic."""
        opt, opt_adv = self.optimizers()

        enc_outputs, dec_outputs = self._forward_step(batch)

        recon_loss, kl_loss = self.module.loss(
            x_pert=batch["pert_cell_emb"],
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        if self.do_start_adv_training:
            if self.adv_steps is None:
                opt.zero_grad()

                z_basal = enc_outputs["z_basal"]

                adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                    perts=batch["pert_emb"].argmax(1),
                    cell_types=batch["cell_type_onehot"].argmax(1),
                    batch_ids=batch["batch"].argmax(1),
                    z_basal=z_basal,
                    compute_penalty=False,
                )

                loss = recon_loss + self.kl_weight * kl_loss - self.adv_lambda * adv_loss

                self.manual_backward(loss)

                if self.do_clip_grad:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.gradient_clip_value,
                        gradient_clip_algorithm="norm",
                    )

                opt.step()

                opt_adv.zero_grad()

                adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                    perts=batch["pert_emb"].argmax(1),
                    cell_types=batch["cell_type_onehot"].argmax(1),
                    batch_ids=batch["batch"].argmax(1),
                    z_basal=z_basal.detach(),
                    compute_penalty=True,
                )

                adv_loss = adv_loss + self.pen_adv * adv_penalty

                self.manual_backward(adv_loss)

                if self.do_clip_grad:
                    self.clip_gradients(
                        opt_adv,
                        gradient_clip_val=self.gradient_clip_value,
                        gradient_clip_algorithm="norm",
                    )

                opt_adv.step()

            elif batch_idx % self.adv_steps == 0:
                opt_adv.zero_grad()

                z_basal = enc_outputs["z_basal"]

                adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                    perts=batch["pert_emb"].argmax(1),
                    cell_types=batch["cell_type_onehot"].argmax(1),
                    batch_ids=batch["batch"].argmax(1),
                    z_basal=z_basal.detach(),
                    compute_penalty=True,
                )

                adv_loss = adv_loss + self.pen_adv * adv_penalty

                loss = adv_loss

                self.manual_backward(adv_loss)

                if self.do_clip_grad:
                    self.clip_gradients(
                        opt_adv,
                        gradient_clip_val=self.gradient_clip_value,
                        gradient_clip_algorithm="norm",
                    )

                opt_adv.step()

            # Model update
            else:
                opt.zero_grad()

                z_basal = enc_outputs["z_basal"]

                adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                    perts=batch["pert_emb"].argmax(1),
                    cell_types=batch["cell_type_onehot"].argmax(1),
                    batch_ids=batch["batch"].argmax(1),
                    z_basal=z_basal,
                    compute_penalty=False,
                )

                loss = recon_loss + self.kl_weight * kl_loss - self.adv_lambda * adv_loss

                self.manual_backward(loss)

                if self.do_clip_grad:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.gradient_clip_value,
                        gradient_clip_algorithm="norm",
                    )

                opt.step()
        else:
            opt.zero_grad()

            z_basal = enc_outputs["z_basal"]

            loss = recon_loss + self.kl_weight * kl_loss

            self.manual_backward(loss)

            if self.do_clip_grad:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.gradient_clip_value,
                    gradient_clip_algorithm="norm",
                )

            opt.step()

            opt_adv.zero_grad()

            adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                perts=batch["pert_emb"].argmax(1),
                cell_types=batch["cell_type_onehot"].argmax(1),
                batch_ids=batch["batch"].argmax(1),
                z_basal=z_basal.detach(),
                compute_penalty=True,
            )

            adv_loss = adv_loss + self.pen_adv * adv_penalty

            self.manual_backward(adv_loss)

            if self.do_clip_grad:
                self.clip_gradients(
                    opt_adv,
                    gradient_clip_val=self.gradient_clip_value,
                    gradient_clip_algorithm="norm",
                )

            opt_adv.step()

        r2_mean, pearson_lfc = self.module.r2_metric(
            x_pert=batch["pert_cell_emb"],
            x_basal=batch["ctrl_cell_emb"],
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        disnt_basal, disnt_after = self.module.disentanglement(
            perts=batch["pert_emb"].argmax(1),
            cell_types=batch["cell_type_onehot"].argmax(1),
            batch_ids=batch["batch"].argmax(1),
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

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
            "pearson_lfc",
            pearson_lfc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "adv_loss",
            adv_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "disnt_basal",
            disnt_basal,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "disnt_after",
            disnt_after,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "adv_acc",
            adv_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        if self.global_step % self.step_size_lr * 1000 == 0:
            sch, sch_adv = self.lr_schedulers()
            sch.step()
            sch_adv.step()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        enc_outputs, dec_outputs = self._forward_step(batch)

        recon_loss, kl_loss = self.module.loss(
            x_pert=batch["pert_cell_emb"],
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        r2_mean, pearson_lfc = self.module.r2_metric(
            x_pert=batch["pert_cell_emb"],
            x_basal=batch["ctrl_cell_emb"],
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        disnt_basal, disnt_after = self.module.disentanglement(
            perts=batch["pert_emb"].argmax(1),
            cell_types=batch["cell_type_onehot"].argmax(1),
            batch_ids=batch["batch"].argmax(1),
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        self.log("val_loss", recon_loss + self.kl_weight * kl_loss, prog_bar=True)
        self.log("val_r2_mean", r2_mean, prog_bar=True)
        self.log("val_pearson_lfc", pearson_lfc, prog_bar=True)
        self.log("es_metric", r2_mean + math.e ** (disnt_after - disnt_basal), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        enc_outputs, dec_outputs = self._forward_step(batch)

        recon_loss, kl_loss = self.module.loss(
            x_pert=batch["pert_cell_emb"],
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        loss = recon_loss + self.kl_weight * kl_loss

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        x_pred = getattr(dec_outputs["px"], output_key)

        self.log("test_loss", loss, prog_bar=True)

        return x_pred

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """

        enc_outputs, dec_outputs = self._forward_step(batch)

        if self.recon_loss == "gauss":
            output_key = "loc"
        else:
            output_key = "mu"

        x_pred = getattr(dec_outputs["px"], output_key)

        outputs = {
            "preds": x_pred,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "X_gene": batch.get("X_gene", None),
            "pert_emb": batch.get("pert_emb", None),
            "pert_name": batch.get("pert_name", None),
            "cell_type": batch.get("cell_type", None),
            "batch": batch.get("batch_name", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        outputs = {k: v for k, v in outputs.items() if v is not None}

        return outputs

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

        adv_params = (
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.perturbation_classifier.parameters(),
                )
            )
            + list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.cell_type_classifier.parameters(),
                )
            )
            + list(filter(lambda p: p.requires_grad, self.module.batch_classifier.parameters()))
        )

        optimizer_adversaries = torch.optim.Adam(adv_params, lr=self.adv_lr, weight_decay=self.adv_wd)
        scheduler_adversaries = StepLR(optimizer_adversaries, step_size=self.step_size_lr, gamma=0.9)

        optimizers = [optimizer_autoencoder, optimizer_adversaries]
        schedulers = [scheduler_autoencoder, scheduler_adversaries]

        if self.step_size_lr is not None:
            return optimizers, schedulers
        else:
            return optimizers
