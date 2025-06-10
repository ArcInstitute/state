import logging
from typing import Dict, Optional

import anndata as ad
import numpy as np
import torch
import torch.nn as nn

from geomloss import SamplesLoss
from typing import Tuple

from .base import PerturbationModel
from .decoders import FinetuneVCICountsDecoder
from .decoders_nb import NBDecoder, nb_nll
from .utils import build_mlp, get_activation_class, get_transformer_backbone


logger = logging.getLogger(__name__)

class EffectGatingToken(nn.Module):
    """
    Learnable token that gets appended to the input sequence.
    Its output from the transformer is used to predict a gating scalar (0-1)
    that modulates the perturbation effect.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable gating token embedding
        self.gating_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map gating token output to a pre-sigmoid scalar
        self.gating_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), # Output one scalar for the gate
        )
        # Sigmoid will be applied after projection to get value in (0,1)

    def append_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append gating token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand gating token to batch size
        gating_tokens = self.gating_token_embed.expand(batch_size, -1, -1).to(seq_input.device)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, gating_tokens], dim=1)

    def extract_gating_value_and_main_output(
        self, transformer_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and gating value from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E] (transformer output for original sequence)
            gating_value: Tensor of shape [B, 1] (sigmoid-activated gate value)
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        gating_token_hidden_state = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project gating token output to pre-sigmoid scalar and apply sigmoid
        pre_sigmoid_gate = self.gating_projection(gating_token_hidden_state) # [B, 1, 1]
        gating_value = torch.sigmoid(pre_sigmoid_gate.squeeze(-1))  # [B, 1]

        return main_output, gating_value

class PertSetsPerturbationModel(PerturbationModel):
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
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.cell_sentence_len = kwargs.get("cell_set_len", 256)
        self.decoder_loss_weight = kwargs.get("decoder_weight", 1.0)
        self.regularization = kwargs.get("regularization", 0.0)
        self.detach_decoder = kwargs.get("detach_decoder", False)

        self.transformer_backbone_key = transformer_backbone_key
        self.transformer_backbone_kwargs = transformer_backbone_kwargs
        self.transformer_backbone_kwargs["n_positions"] = self.cell_sentence_len + kwargs.get("extra_tokens", 0)

        self.effect_gating_token = None
        self.gating_loss_fn = None # e.g., MSELoss for the gate
        if kwargs.get("use_effect_gating_token", False):
            # Ensure extra_tokens in transformer_backbone_kwargs is >= 1 if using this token
            if self.transformer_backbone_kwargs.get("n_positions", self.cell_sentence_len) <= self.cell_sentence_len:
                 logger.warning("use_effect_gating_token is True, but transformer n_positions might not account for the extra token. Ensure n_positions = cell_set_len + num_extra_tokens.")

            self.effect_gating_token = EffectGatingToken(hidden_dim=self.hidden_dim, dropout=self.dropout)
            self.gating_loss_fn = nn.MSELoss()

        self.distributional_loss = distributional_loss
        self.gene_dim = gene_dim

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Build the underlying neural OT network
        self._build_networks()

        # Add an optional encoder that introduces a batch variable
        self.batch_encoder = None
        self.batch_dim = None
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

        self.freeze_pert_backbone = kwargs.get("freeze_pert_backbone", False)
        if self.freeze_pert_backbone:
            modules_to_freeze = [
                self.transformer_backbone,
                self.project_out,
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):  # TODO: This will go very soon
            gene_names = []

            if output_space == "gene":
                # hvg's but for which dataset?
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5")
                    # gene_names = temp.var.index.values
            else:
                assert output_space == "all"
                if "DMSO_TF" in control_pert:
                    gene_names = np.load(
                        "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy", allow_pickle=True
                    )
                elif "non-targeting" in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad("/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5")
                    gene_names = temp.var.index.values

            self.gene_decoder = FinetuneVCICountsDecoder(
                genes=gene_names,
                # latent_dim=self.output_dim + (self.batch_dim or 0),
            )

        print(self)

    def _build_networks(self):
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

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """If needed, define how we embed the raw perturbation input."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Define how we embed basal state input, if needed."""
        return self.basal_encoder(expr)

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
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

        # Shape: [B, S, hidden_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)

        seq_input = pert_embedding + control_cells  # Shape: [B, S, hidden_dim]

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

        if self.effect_gating_token is not None:
            # Append gating token: [B, S, E] -> [B, S+1, E]
            seq_input = self.effect_gating_token.append_token(seq_input)

        # forward pass + extract CLS last hidden state
        if self.hparams.get("mask_attn", False):
            batch_size, seq_length, _ = seq_input.shape
            device = seq_input.device

            self.transformer_backbone._attn_implementation = "eager"

            # create a [1,1,S,S] mask (now S+1 if confidence token is used)
            base = torch.eye(seq_length, device=device).view(1, seq_length, seq_length)

            # repeat out to [B,H,S,S]
            attn_mask = base.repeat(batch_size, 1, 1)

            outputs = self.transformer_backbone(inputs_embeds=seq_input, attention_mask=attn_mask)
            transformer_output = outputs.last_hidden_state
        else:
            transformer_output = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state

        # Extract confidence prediction if confidence token was used
        if self.effect_gating_token is not None:
            res_pred, gating_value_alpha = self.effect_gating_token.extract_gating_value_and_main_output(transformer_output)
        else:
            res_pred, gating_value_alpha = transformer_output, None

        if self.predict_residual:
            # transformer_output_main is h_delta_pred (latent delta)
            latent_delta_pred = res_pred
            if gating_value_alpha is not None:
                # gating_value_alpha is [B, 1], latent_delta_pred is [B, S, E]
                # We want to scale the delta for the whole set based on one gate value per set.
                # So, average latent_delta_pred over S, or take CLS-like token if transformer was designed for that.
                # Current transformer backbone processes sequence and `project_out` is applied to each token's output.
                # If we assume the gate applies uniformly to all cells in the set, broadcast:
                gated_latent_delta = gating_value_alpha.unsqueeze(1) * latent_delta_pred # [B,1,1] * [B,S,E] -> [B,S,E]
            else:
                gated_latent_delta = latent_delta_pred
            
            final_latent_representation = control_cells + gated_latent_delta
        else: # Model predicts full state, so we derive delta, gate it, and add back to basal
            latent_full_pred = res_pred
            latent_delta_derived = latent_full_pred - control_cells
            if gating_value_alpha is not None:
                gated_latent_delta = gating_value_alpha.unsqueeze(1) * latent_delta_derived
            else:
                gated_latent_delta = latent_delta_derived
            final_latent_representation = control_cells + gated_latent_delta

        out_pred = self.project_out(final_latent_representation)

        # apply relu if specified and we output to HVG space
        is_gene_space = self.hparams["embed_key"] == "X_hvg" or self.hparams["embed_key"] is None
        if is_gene_space or self.gene_decoder is None:
            out_pred = self.relu(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if self.effect_gating_token is not None:
            return output, gating_value_alpha
        else:
            return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        gating_value_pred = None
        if self.effect_gating_token is not None:
            pred, gating_value_pred = self.forward(batch, padded=padded)
        else:
            pred = self.forward(batch, padded=padded)

        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        main_loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", main_loss)

        # Process decoder if available
        decoder_loss = None
        total_loss = main_loss

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
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

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        if gating_value_pred is not None and self.gating_loss_fn is not None and "ctrl_cell_counts" in batch:
            pert_counts = batch["pert_cell_counts"] # Shape [B*S, gene dim]
            ctrl_counts = batch["ctrl_cell_counts"] # Shape [B*S, gene dim]
            
            num_genes_for_norm = pert_counts.shape[-1]

            # Reshape counts to [B, S, gene dim] to calculate per-set effect norm
            pert_counts_reshaped = pert_counts.reshape(-1, self.cell_sentence_len, num_genes_for_norm)
            ctrl_counts_reshaped = ctrl_counts.reshape(-1, self.cell_sentence_len, num_genes_for_norm)

            # Mean expression for each set in the batch
            mean_pert_counts = pert_counts_reshaped.mean(dim=1) # [B, GeneDim]
            mean_ctrl_counts = ctrl_counts_reshaped.mean(dim=1) # [B, GeneDim]
            
            effect_diff = mean_pert_counts - mean_ctrl_counts # [B, GeneDim]
            effect_norm = torch.linalg.norm(effect_diff, dim=1, ord=2) # [B]

            # Scale effect_norm to target for sigmoid output (0-1)
            target_gating_value = torch.clamp(
                effect_norm, 0.0, 1.0
            ) # Target is [B]

            # gating_value_pred is [B, 1]. Squeeze to [B] for MSELoss.
            gating_value_pred_squeezed = gating_value_pred.squeeze()

            gating_loss = self.gating_loss_fn(gating_value_pred_squeezed, target_gating_value)
            self.log("train/gating_loss", gating_loss)
            self.log("train/effect_norm_mean", effect_norm.mean())

            total_loss = total_loss + 1.0 * gating_loss

        if self.regularization > 0.0:
            ctrl_cell_emb = batch["ctrl_cell_emb"].reshape_as(pred)
            delta = pred - ctrl_cell_emb

            # compute l1 loss
            l1_loss = torch.abs(delta).mean()

            # Log the regularization loss
            self.log("train/l1_regularization", l1_loss)

            # Add regularization to total loss
            total_loss = total_loss + self.regularization * l1_loss

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        if self.effect_gating_token is None:
            pred, gating_value_pred = self.forward(batch), None
        else:
            pred, gating_value_pred = self.forward(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)

        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            gene_targets = batch["pert_cell_counts"]

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["pert_cell_counts"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                # Get decoder predictions
                pert_cell_counts_preds = self.gene_decoder(latent_preds).reshape(
                    -1, self.cell_sentence_len, self.gene_dim
                )
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
                decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets).mean()

            # Log the validation metric
            self.log("val/decoder_loss", decoder_loss)
            loss = loss + self.decoder_loss_weight * decoder_loss

        if gating_value_pred is not None and self.gating_loss_fn is not None and "ctrl_cell_counts" in batch:
            pert_counts = batch["pert_cell_counts"]
            ctrl_counts = batch["ctrl_cell_counts"]
            num_genes_for_norm = pert_counts.shape[-1]

            pert_counts_reshaped = pert_counts.reshape(-1, self.cell_sentence_len, num_genes_for_norm)
            ctrl_counts_reshaped = ctrl_counts.reshape(-1, self.cell_sentence_len, num_genes_for_norm)
            mean_pert_counts = pert_counts_reshaped.mean(dim=1)
            mean_ctrl_counts = ctrl_counts_reshaped.mean(dim=1)
            effect_diff = mean_pert_counts - mean_ctrl_counts
            effect_norm = torch.linalg.norm(effect_diff, dim=1, ord=2)
            target_gating_value = torch.clamp(
                effect_norm / 10.0, 0.0, 1.0
            )
            gating_value_pred_squeezed = gating_value_pred.squeeze()
            gating_loss = self.gating_loss_fn(gating_value_pred_squeezed, target_gating_value)
            self.log("val/gating_loss", gating_loss)
            self.log("val/effect_norm_mean", effect_norm.mean())
            loss = loss + 1.0 * gating_loss

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        if self.effect_gating_token is None:
            pred, gating_value_pred = self.forward(batch), None
        else:
            pred, gating_value_pred = self.forward(batch)

        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        if gating_value_pred is not None and self.gating_loss_fn is not None and "ctrl_cell_counts" in batch:
            pert_counts = batch["pert_cell_counts"]
            ctrl_counts = batch["ctrl_cell_counts"]
            num_genes_for_norm = pert_counts.shape[-1]

            pert_counts_reshaped = pert_counts.reshape(-1, self.cell_sentence_len, num_genes_for_norm)
            ctrl_counts_reshaped = ctrl_counts.reshape(-1, self.cell_sentence_len, num_genes_for_norm)
            mean_pert_counts = pert_counts_reshaped.mean(dim=1)
            mean_ctrl_counts = ctrl_counts_reshaped.mean(dim=1)
            effect_diff = mean_pert_counts - mean_ctrl_counts
            effect_norm = torch.linalg.norm(effect_diff, dim=1, ord=2)
            target_gating_value = torch.clamp(
                effect_norm / 10.0, 0.0, 1.0
            )
            gating_value_pred_squeezed = gating_value_pred.squeeze()
            gating_loss = self.gating_loss_fn(gating_value_pred_squeezed, target_gating_value)
            self.log("val/gating_loss", gating_loss)
            self.log("val/effect_norm_mean", effect_norm.mean())
            loss = loss + 1.0 * gating_loss

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:s
         returning 'preds', 'X', 'pert_name', etc.
        """
        gating_value_alpha = None
        model_output = self.forward(batch, padded=padded)
        if self.effect_gating_token is not None:
            latent_output, gating_value_alpha = model_output
        else:
            latent_output = model_output

        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
            "ctrl_cell_counts": batch.get("ctrl_cell_counts", None),
        }

        if gating_value_alpha is not None:
            # gating_value_alpha is [B,1]. If output is flattened [B*S, D], we might want to repeat/expand it.
            # For predict_step, it's probably best to return it as [B,1] and let downstream handle.
            output_dict["gating_value_pred"] = gating_value_alpha.reshape(-1,1)

        if self.gene_decoder is not None:
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                pert_cell_counts_preds = mu
            else:
                pert_cell_counts_preds = self.gene_decoder(latent_output)

            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict
