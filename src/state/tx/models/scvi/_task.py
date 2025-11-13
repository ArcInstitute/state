import math
from collections import defaultdict

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import accuracy

from .._base_modules import FocalLoss
from ._module import CPAModule


class CPATrainer(L.LightningModule):
    def __init__(
        self,
        module: CPAModule,
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
        """Training plan for the CPA model"""
        super().__init__()

        self.module = module

        self.lr = float(lr)
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        self.automatic_optimization = False

        self.wd = float(wd)

        self.n_perts = module.n_perts

        self.n_steps_pretrain_ae = n_steps_pretrain_ae
        self.n_epochs_pretrain_ae = n_epochs_pretrain_ae

        self.n_steps_adv_warmup = n_steps_adv_warmup
        self.n_epochs_adv_warmup = n_epochs_adv_warmup
        self.adv_steps = adv_steps

        self.reg_adv = reg_adv
        self.pen_adv = pen_adv

        self.adv_lr = float(adv_lr)
        self.adv_wd = float(adv_wd)

        self.step_size_lr = step_size_lr

        self.do_clip_grad = do_clip_grad
        self.gradient_clip_value = gradient_clip_value
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.metrics = [
            "recon_loss",
            "KL",
            "disnt_basal",
            "disnt_after",
            "r2_mean",
            "r2_var",
            "adv_loss",
            "penalty_adv",
            "adv_perts",
            "acc_perts",
            "penalty_perts",
        ]

        self.epoch_history = defaultdict(list)

        ## TODO: remove unused
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.adv_loss = adv_loss.lower()
        self.gamma = kwargs.get("gamma", 2.0)
        if self.adv_loss == "focal":
            self.adv_loss_fn = FocalLoss(gamma=self.gamma, reduction="mean")
        else:
            self.adv_loss_fn = nn.CrossEntropyLoss()

    @property
    def kl_weight(self):
        return 0.0

    @property
    def adv_lambda(self):
        slope = self.reg_adv
        if self.n_steps_adv_warmup:
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

    @property
    def do_start_adv_training(self):
        if self.n_steps_pretrain_ae:
            return self.global_step > self.n_steps_pretrain_ae
        elif self.n_epochs_pretrain_ae:
            return self.current_epoch > self.n_epochs_pretrain_ae
        else:
            return True

    def adversarial_loss(self, batch, z_basal, compute_penalty=True):
        """Computes adversarial classification losses and regularizations"""
        if compute_penalty:
            z_basal = z_basal.requires_grad_(True)

        adv_logits = self.module.forward_adv(z_basal)
        perts = batch["pert_emb"].argmax(1)
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

        cell_types = batch["cell_type_onehot"].argmax(1)
        cell_types_logits = adv_logits["cell_type_logits"]
        cell_types_adv_loss = self.adv_loss_fn(cell_types_logits, cell_types.long())
        cell_types_acc = accuracy(
            cell_types_logits.argmax(1),
            cell_types.long().view(
                -1,
            ),
            average="macro",
            task="multiclass",
            num_classes=self.n_perts,
        )

        batch_ids = batch["batch"].argmax(1)
        batch_ids_logits = adv_logits["batch_logits"]
        batch_ids_adv_loss = self.adv_loss_fn(batch_ids_logits, batch_ids.long())
        batch_ids_acc = accuracy(
            batch_ids_logits.argmax(1),
            batch_ids.long().view(
                -1,
            ),
            average="macro",
            task="multiclass",
            num_classes=self.n_perts,
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

    def configure_optimizers(self):
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

    def training_step(self, batch, batch_idx):
        opt, opt_adv = self.optimizers()

        enc_outputs, dec_outputs = self.module.forward(batch)

        recon_loss, kl_loss = self.module.loss(
            batch=batch,
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        if self.do_start_adv_training:
            if self.adv_steps is None:
                opt.zero_grad()

                z_basal = enc_outputs["z_basal"]

                adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                    batch=batch,
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
                    batch=batch,
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
                    batch=batch,
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

            # Model update
            else:
                opt.zero_grad()

                z_basal = enc_outputs["z_basal"]

                adv_loss, adv_acc, adv_penalty = self.adversarial_loss(
                    batch=batch,
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
                batch=batch,
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

        r2_mean, r2_lfc = self.module.r2_metric(batch, enc_outputs, dec_outputs)

        disnt_basal, disnt_after = self.module.disentanglement(batch, enc_outputs, dec_outputs)

        results = {
            "recon_loss": recon_loss.item(),
            "KL": kl_loss.item(),
            "r2_mean": r2_mean,
            "r2_lfc": r2_lfc,
            "adv_loss": adv_loss.item(),
            "adv_acc": adv_acc.item(),
            "penalty_adv": adv_penalty.item(),
            "es_metric": r2_mean + np.e ** (disnt_after - disnt_basal),
            "disnt_basal": disnt_basal,
            "disnt_after": disnt_after,
        }

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

        return results

    def validation_step(self, batch, batch_idx):
        enc_outputs, dec_outputs = self.module.forward(batch)

        recon_loss, kl_loss = self.module.loss(
            batch=batch,
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        r2_mean, r2_lfc = self.module.r2_metric(batch, enc_outputs, dec_outputs)

        disnt_basal, disnt_after = self.module.disentanglement(batch, enc_outputs, dec_outputs)

        self.log("val_r2_mean", r2_mean, prog_bar=True)
        self.log("val_r2_lfc", r2_lfc, prog_bar=True)
        self.log("es_metric", r2_mean + math.e ** (disnt_after - disnt_basal), prog_bar=True)

    def test_step(self, batch, batch_idx):
        enc_outputs, dec_outputs = self.module.forward(batch)

        recon_loss, kl_loss = self.module.loss(
            batch=batch,
            encoder_outputs=enc_outputs,
            decoder_outputs=dec_outputs,
        )

        r2_mean, r2_lfc = self.module.r2_metric(batch, enc_outputs, dec_outputs)

        self.log("test_recon", recon_loss.item(), prog_bar=True)
        self.log("test_r2_mean", r2_mean, prog_bar=True)
        self.log("test_r2_lfc", r2_lfc, prog_bar=True)

        x_pred = self.module.get_expression(batch, n_samples=1)

        return x_pred.detach().cpu().numpy()
