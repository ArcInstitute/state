import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss
from typing import Dict, Optional, Tuple

from .base import PerturbationModel
from .utils import build_mlp, get_activation_class, get_transformer_backbone, apply_lora


logger = logging.getLogger(__name__)


class StateTransitionPerturbationModel(PerturbationModel):
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
        batch_dim: int = None,
        basal_mapping_strategy: str = "random",
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
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # Save or store relevant hyperparams
        self.predict_residual = predict_residual
        self.output_space = output_space
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 1)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 1)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim
        self.nb_loss = bool(kwargs.get("nb_loss", False))
        self.nb_eps = float(kwargs.get("nb_eps", 1e-8))
        default_nb_embed_loss_weight = 1.0 if self.embed_key is not None else 0.0
        self.nb_embed_loss_weight = float(kwargs.get("nb_embed_loss_weight", default_nb_embed_loss_weight))
        if self.nb_loss and self.output_space == "embedding":
            raise ValueError(
                "nb_loss=True is incompatible with output_space='embedding'. "
                "Use output_space='gene' or output_space='all'."
            )
        if self.nb_loss and self.output_space not in {"gene", "all"}:
            raise ValueError(f"nb_loss=True requires output_space in {{'gene', 'all'}}; got {self.output_space!r}.")
        if self.nb_loss:
            if self.gene_decoder is not None:
                logger.info("nb_loss=True: disabling gene_decoder and decoder loss branches.")
            self.gene_decoder = None
            self.gene_decoder_bool = False
            self.decoder_cfg = None
            try:
                self.hparams["gene_decoder_bool"] = False  # type: ignore[index]
                self.hparams["decoder_cfg"] = None  # type: ignore[index]
            except Exception:
                pass

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "se":
            raise ValueError(
                "loss='se' (combined sinkhorn+energy) has been removed. "
                "Use loss='energy', loss='sinkhorn', or loss='mse'."
            )
        elif loss_name == "sinkhorn":
            self.loss_fn = SamplesLoss(loss="sinkhorn", blur=blur)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        self.use_basal_projection = kwargs.get("use_basal_projection", True)

        # Build the underlying neural OT network
        self._build_networks(lora_cfg=kwargs.get("lora", None))
        self.nb_parameter_head: Optional[nn.Module] = None
        self.nb_target_dim = int(self.output_dim)
        if self.nb_loss:
            if self.output_space == "all":
                self.nb_target_dim = int(self.gene_dim if self.gene_dim is not None else self.output_dim)
            elif self.embed_key is not None and self.embed_key != "X_hvg":
                self.nb_target_dim = int(self.hvg_dim if self.hvg_dim is not None else self.output_dim)

            self.nb_parameter_head = build_mlp(
                in_dim=self.output_dim,
                out_dim=self.nb_target_dim * 2,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_decoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
            logger.info(
                "NB loss enabled for state transition model (nb_target_dim=%d, nb_embed_loss_weight=%.3f).",
                self.nb_target_dim,
                self.nb_embed_loss_weight,
            )

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
        self.predict_mean = kwargs.get("predict_mean", False)
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = nn.Embedding(
                num_embeddings=batch_dim,
                embedding_dim=hidden_dim,
            )
            self.batch_dim = batch_dim

        # if the model is outputting to counts space, apply relu
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs["embed_key"] == "X_hvg" or kwargs["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            self.relu = torch.nn.ReLU()

        if kwargs.get("use_batch_token", False) or kwargs.get("batch_predictor", False):
            logger.warning(
                "Batch-token logic has been removed from StateTransitionPerturbationModel. "
                "Ignoring model.kwargs.use_batch_token and model.kwargs.batch_predictor."
            )

        if kwargs.get("confidence_token", False):
            logger.warning(
                "Confidence-token logic has been removed from StateTransitionPerturbationModel. "
                "Ignoring model.kwargs.confidence_token."
            )

        # Backward-compat: accept legacy key `freeze_pert`
        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", kwargs.get("freeze_pert", False))
        if self.freeze_pert_backbone:
            # Freeze backbone base weights but keep LoRA adapter weights (if present) trainable
            for name, param in self.transformer_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # Freeze projection head as before
            for param in self.project_out.parameters():
                param.requires_grad = False

        if kwargs.get("finetune_vci_decoder", False):
            logger.warning(
                "model.kwargs.finetune_vci_decoder is no longer supported. "
                "Ignoring it and using the standard latent-to-gene decoder path."
            )
        print(self)

    def _build_networks(self, lora_cfg=None):
        """
        Here we instantiate the actual GPT2-based model.
        """
        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        # Simple linear layer that maintains the input dimension
        if self.use_basal_projection:
            self.basal_encoder = build_mlp(
                in_dim=self.input_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )
        else:
            self.basal_encoder = nn.Linear(self.input_dim, self.hidden_dim)

        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
        )

        # Optionally wrap backbone with LoRA adapters
        if lora_cfg and lora_cfg.get("enable", False):
            self.transformer_backbone = apply_lora(
                self.transformer_backbone,
                self.transformer_backbone_key,
                lora_cfg,
            )

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def _main_loss_is_expression(self) -> bool:
        if self.nb_loss:
            return True
        return super()._main_loss_is_expression()

    @staticmethod
    def _suspected_discrete_torch(x: torch.Tensor, n_cells: int = 100) -> bool:
        if x.numel() == 0:
            return False
        flat = x.reshape(-1, x.shape[-1])
        top_n = min(flat.shape[0], n_cells)
        rowsum = flat[:top_n].sum(dim=1)
        frac_part = rowsum - rowsum.floor()
        return bool(torch.all(torch.abs(frac_part) < 1e-7))

    @staticmethod
    def _suspected_log_torch(x: torch.Tensor) -> bool:
        if x.numel() == 0:
            return False
        return bool(x.max().item() < 15.0)

    def _to_count_space(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        is_discrete = self._suspected_discrete_torch(x_float)
        is_log = self._suspected_log_torch(x_float)

        if (not is_discrete) and is_log:
            counts = torch.expm1(x_float)
        else:
            counts = x_float

        counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        return counts.clamp_min(0.0).round()

    def _compute_set_library_sizes_from_control(self, ctrl_cells: torch.Tensor) -> torch.Tensor:
        ctrl_counts = self._to_count_space(ctrl_cells)
        per_cell_library_sizes = ctrl_counts.sum(dim=-1)
        per_cell_library_sizes = torch.nan_to_num(per_cell_library_sizes, nan=0.0, posinf=0.0, neginf=0.0)
        per_set_library_sizes = per_cell_library_sizes.median(dim=1, keepdim=True).values.clamp_min(1.0)
        return per_set_library_sizes.unsqueeze(-1)

    def _reshape_sequence_tensor(self, x: torch.Tensor, padded: bool) -> torch.Tensor:
        if padded:
            return x.reshape(-1, self.cell_sentence_len, x.shape[-1])
        return x.reshape(1, -1, x.shape[-1])

    def _get_nb_target_tensor(
        self, batch: Dict[str, torch.Tensor], fallback_target: torch.Tensor, padded: bool
    ) -> torch.Tensor:
        pert_counts = batch.get("pert_cell_counts", None)
        if pert_counts is not None:
            return self._reshape_sequence_tensor(pert_counts.to(fallback_target.device), padded)
        return fallback_target

    def _get_nb_control_tensor_for_library(
        self,
        batch: Dict[str, torch.Tensor],
        fallback_ctrl: torch.Tensor,
        padded: bool,
    ) -> torch.Tensor:
        ctrl_counts = batch.get("ctrl_cell_counts", None)
        if ctrl_counts is not None:
            return self._reshape_sequence_tensor(ctrl_counts.to(fallback_ctrl.device), padded)
        return fallback_ctrl

    def _compute_nb_nll_loss(
        self,
        nb_mean: torch.Tensor,
        nb_dispersion: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target_counts = self._to_count_space(target).to(nb_mean.dtype)
        if nb_mean.shape != nb_dispersion.shape:
            raise RuntimeError(
                f"NB parameter shape mismatch: mean={tuple(nb_mean.shape)} dispersion={tuple(nb_dispersion.shape)}"
            )
        if target_counts.shape[-1] != nb_mean.shape[-1]:
            raise RuntimeError(
                "NB target dimension mismatch: "
                f"target={target_counts.shape[-1]} vs nb_params={nb_mean.shape[-1]}. "
                "Ensure pert_cell_counts has the expected gene dimension for NB training."
            )
        mu = nb_mean.clamp_min(self.nb_eps)
        theta = nb_dispersion.clamp_min(self.nb_eps)
        log_theta_mu_eps = torch.log(theta + mu + self.nb_eps)
        log_nb = (
            theta * (torch.log(theta + self.nb_eps) - log_theta_mu_eps)
            + target_counts * (torch.log(mu + self.nb_eps) - log_theta_mu_eps)
            + torch.lgamma(target_counts + theta)
            - torch.lgamma(theta)
            - torch.lgamma(target_counts + 1)
        )
        recon_loss_all = -log_nb
        return torch.nanmean(recon_loss_all.reshape(recon_loss_all.shape[0], -1), dim=1)

    def forward(
        self,
        batch: dict,
        padded=True,
        return_nb_params: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.

        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension

        The `padded` argument here is set to True if the batch is padded. Otherwise, we
        expect a single batch, so that sentences can vary in length across batches.
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, input_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        # Add encodings in input_dim space, then project to hidden_dim
        combined_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]
        seq_input = combined_input  # Shape: [B, S, hidden_dim]

        if self.batch_encoder is not None:
            # Extract batch indices (assume they are integers or convert from one-hot)
            batch_indices = batch["batch"]

            # Handle one-hot encoded batch indices
            if batch_indices.dim() > 1 and batch_indices.size(-1) == self.batch_dim:
                batch_indices = batch_indices.argmax(-1)

            # Reshape batch indices to match sequence structure
            if padded:
                batch_indices = batch_indices.reshape(-1, self.cell_sentence_len)
            else:
                batch_indices = batch_indices.reshape(1, -1)

            # Get batch embeddings and add to sequence input
            batch_embeddings = self.batch_encoder(batch_indices.long())  # Shape: [B, S, hidden_dim]
            seq_input = seq_input + batch_embeddings

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device
            self.transformer_backbone._attn_implementation = "eager"  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

            # create a [1,1,S,S] mask
            base = torch.eye(seq_length, device=device, dtype=torch.bool).view(1, 1, seq_length, seq_length)

            # Get number of attention heads from model config
            num_heads = self.transformer_backbone.config.num_attention_heads

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, num_heads, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            outputs = self.transformer_backbone(inputs_embeds=seq_input)
            transformer_output = outputs.last_hidden_state

        res_pred = transformer_output

        # add to basal if predicting residual
        if self.predict_residual and self.output_space == "all":
            # Project control_cells to hidden_dim space to match res_pred
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if is_gene_space or (self.gene_decoder is None and not self.nb_loss):
            out_pred = self.relu(out_pred)

        nb_mean = None
        nb_dispersion = None
        if self.nb_loss:
            if self.nb_parameter_head is None:
                raise RuntimeError("nb_loss=True but nb_parameter_head was not initialized.")
            nb_params = self.nb_parameter_head(out_pred)
            px_scale_logits, nb_dispersion_logits = torch.chunk(nb_params, chunks=2, dim=-1)
            px_scale = F.softmax(px_scale_logits, dim=-1)
            ctrl_for_library = self._get_nb_control_tensor_for_library(batch, basal, padded)
            set_library_sizes = self._compute_set_library_sizes_from_control(ctrl_for_library)
            nb_mean = px_scale * set_library_sizes
            nb_dispersion = F.softplus(nb_dispersion_logits) + self.nb_eps

        output = out_pred.reshape(-1, self.output_dim)

        if not self.nb_loss or not return_nb_params:
            return output
        if nb_mean is None or nb_dispersion is None:
            raise RuntimeError("nb_loss=True but NB parameters were not produced in forward().")
        return output, nb_mean.reshape(-1, self.nb_target_dim), nb_dispersion.reshape(-1, self.nb_target_dim)

    def _compute_distribution_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply the primary distributional loss."""
        return self.loss_fn(pred, target)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        if self.nb_loss:
            pred, nb_mean_flat, nb_dispersion_flat = self.forward(batch, padded=padded, return_nb_params=True)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        embedding_aux_loss = None
        if self.nb_loss:
            if padded:
                nb_mean = nb_mean_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
                nb_dispersion = nb_dispersion_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
            else:
                nb_mean = nb_mean_flat.reshape(1, -1, self.nb_target_dim)
                nb_dispersion = nb_dispersion_flat.reshape(1, -1, self.nb_target_dim)

            nb_target = self._get_nb_target_tensor(batch, target, padded)
            per_set_main_losses = self._compute_nb_nll_loss(nb_mean, nb_dispersion, nb_target)
            if self.nb_embed_loss_weight > 0.0:
                embedding_aux_losses = self._compute_distribution_loss(pred, target)
                embedding_aux_loss = torch.nanmean(embedding_aux_losses)
                self.log("train/embedding_loss", embedding_aux_loss)
        else:
            per_set_main_losses = self._compute_distribution_loss(pred, target)
        main_loss = torch.nanmean(per_set_main_losses)
        self.log(self._train_main_loss_key(), main_loss)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss
        if embedding_aux_loss is not None:
            total_loss = total_loss + self.nb_embed_loss_weight * embedding_aux_loss

        # Decoder loss in gene space, if a decoder is configured.
        if (not self.nb_loss) and self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]
            # Train decoder to map latent predictions to gene space

            if self.detach_decoder:
                # with some random change, use the true targets
                if np.random.rand() < 0.1:
                    latent_preds = target.reshape_as(pred).detach()
                else:
                    latent_preds = pred.detach()
            else:
                latent_preds = pred

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            if padded:
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            else:
                gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

            decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
            decoder_loss = decoder_per_set.mean()

            # Log decoder loss
            self.log(self._train_expression_loss_key(), decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.nb_loss:
            pred, nb_mean_flat, nb_dispersion_flat = self.forward(batch, return_nb_params=True)
        else:
            pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        embedding_aux_loss = None
        if self.nb_loss:
            nb_mean = nb_mean_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
            nb_dispersion = nb_dispersion_flat.reshape(-1, self.cell_sentence_len, self.nb_target_dim)
            nb_target = self._get_nb_target_tensor(batch, target, padded=True)
            per_set_main_losses = self._compute_nb_nll_loss(nb_mean, nb_dispersion, nb_target)
            if self.nb_embed_loss_weight > 0.0:
                embedding_aux_losses = self._compute_distribution_loss(pred, target)
                embedding_aux_loss = torch.nanmean(embedding_aux_losses)
                self.log("val/embedding_loss", embedding_aux_loss)
        else:
            per_set_main_losses = self._compute_distribution_loss(pred, target)
        loss = torch.nanmean(per_set_main_losses)
        if embedding_aux_loss is not None:
            loss = loss + self.nb_embed_loss_weight * embedding_aux_loss
        self.log(self._val_main_loss_key(), loss)

        if (not self.nb_loss) and self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                -1, self.cell_sentence_len, self.gene_decoder.gene_dim()
            )
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
            decoder_per_set = self._compute_distribution_loss(pert_cell_counts_preds, gene_targets)
            decoder_loss = decoder_per_set.mean()

            # Log the validation metric
            self.log(self._val_expression_loss_key(), decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.nb_loss:
            pred, nb_mean_flat, nb_dispersion_flat = self.forward(batch, padded=False, return_nb_params=True)
            target = batch["pert_cell_emb"]
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)
            nb_mean = nb_mean_flat.reshape(1, -1, self.nb_target_dim)
            nb_dispersion = nb_dispersion_flat.reshape(1, -1, self.nb_target_dim)
            nb_target = self._get_nb_target_tensor(batch, target, padded=False)
            per_set_main_losses = self._compute_nb_nll_loss(nb_mean, nb_dispersion, nb_target)
            loss = torch.nanmean(per_set_main_losses)
            if self.nb_embed_loss_weight > 0.0:
                embedding_aux_losses = self._compute_distribution_loss(pred, target)
                loss = loss + self.nb_embed_loss_weight * torch.nanmean(embedding_aux_losses)
            self.log("test_loss", loss)
            return
        _ = self.forward(batch, padded=False)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.nb_loss:
            latent_output, nb_mean_flat, nb_dispersion_flat = self.forward(
                batch,
                padded=padded,
                return_nb_params=True,
            )
        else:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "pert_cell_barcode": batch.get("pert_cell_barcode", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
        }

        if self.nb_loss:
            output_dict["pert_cell_counts_preds"] = nb_mean_flat.reshape(-1, self.nb_target_dim)
            output_dict["pert_cell_counts_dispersion"] = nb_dispersion_flat.reshape(-1, self.nb_target_dim)
        elif self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def configure_optimizers(self):
        """
        Configure optimizer and optional cosine LR decay.

        This is intentionally scoped to StateTransitionPerturbationModel only.
        """
        optimizer_name = str(self.hparams.get("optimizer", "adam")).lower()
        base_lr = float(self.hparams.get("lr", self.lr))
        weight_decay = float(self.hparams.get("weight_decay", 0.0))

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Expected one of: adam, adamw.")

        if not bool(self.hparams.get("use_cosine_decay", False)):
            return optimizer

        max_lr_cfg = self.hparams.get("max_lr", None)
        max_lr = float(max_lr_cfg) if max_lr_cfg is not None else base_lr
        if max_lr <= 0:
            raise ValueError(f"max_lr must be > 0 when cosine decay is enabled. Received: {max_lr}")

        decay_steps_cfg = self.hparams.get("lr_decay_steps", None)
        if decay_steps_cfg is None:
            decay_steps = int(self.hparams.get("max_steps", 0))
        else:
            decay_steps = int(decay_steps_cfg)
        if decay_steps <= 0:
            raise ValueError(
                "lr_decay_steps must be a positive integer when cosine decay is enabled "
                "(or training.max_steps must be set > 0)."
            )

        max_lr_fraction = float(self.hparams.get("max_lr_fraction", 0.1))
        if not (0 < max_lr_fraction <= 1.0):
            raise ValueError(f"max_lr_fraction must be in (0, 1]. Received: {max_lr_fraction}")

        min_lr = max_lr * max_lr_fraction
        for param_group in optimizer.param_groups:
            param_group["lr"] = max_lr

        def _lr_lambda(step: int) -> float:
            if step >= decay_steps:
                return max_lr_fraction
            decay_ratio = step / decay_steps
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = min_lr + coeff * (max_lr - min_lr)
            return lr / max_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
