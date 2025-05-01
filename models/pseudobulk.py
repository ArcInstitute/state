# File: models/neural_ot.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import anndata as ad
import logging

from collections import defaultdict
from geomloss import SamplesLoss
from typing import Optional, Dict, List

from models.base import PerturbationModel, GeneWiseDecoder
from models.decoders import DecoderInterface, FinetuneVCICountsDecoder
from models.utils import build_mlp, get_activation_class, get_transformer_backbone

from models.decoders_nb import NBDecoder, nb_nll

logger = logging.getLogger(__name__)

class ScaledSamplesLoss(nn.Module):
    """
    A wrapper around SamplesLoss that scales the output by a given factor.
    """
    def __init__(self, base_loss, scale_factor):
        super(ScaledSamplesLoss, self).__init__()
        self.base_loss = base_loss
        self.scale_factor = scale_factor

    def forward(self, x, y):
        return self.base_loss(x, y) / self.scale_factor

class ConfidenceHead(nn.Module):
    """
    A confidence head that predicts the expected loss value for a set of cells.
    
    The confidence head takes the transformer hidden states and predicts a single
    scalar value representing the expected distribution loss.
    """
    
    def __init__(self, hidden_dim, pooling_method='mean', dropout=0.1):
        """
        Initialize the confidence head.
        
        Args:
            hidden_dim: Dimension of the transformer hidden states
            pooling_method: Method to pool the hidden states ('mean', 'max', or 'attention')
            dropout: Dropout rate for the confidence head
        """
        super().__init__()
        self.pooling_method = pooling_method
        
        # If using attention pooling, create an attention mechanism
        if pooling_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # MLP to predict confidence/uncertainty
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)

        )
    
    def forward(self, hidden_states):
        """
        Forward pass of the confidence head.
        
        Args:
            hidden_states: Hidden states from the transformer [B, S, H]
            
        Returns:
            confidence: Predicted confidence value [B, 1]
        """
        # Pool the hidden states based on the specified method
        if self.pooling_method == 'mean':
            # Mean pooling across sequence dimension
            pooled = hidden_states.mean(dim=1)  # [B, H]
        
        elif self.pooling_method == 'max':
            # Max pooling across sequence dimension
            pooled, _ = hidden_states.max(dim=1)  # [B, H]
        
        elif self.pooling_method == 'attention':
            # Attention pooling
            attention_weights = self.attention(hidden_states)  # [B, S, 1]
            attention_weights = F.softmax(attention_weights, dim=1)  # [B, S, 1]
            pooled = torch.sum(hidden_states * attention_weights, dim=1)  # [B, H]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Predict confidence/uncertainty value
        confidence = self.confidence_net(pooled)  # [B, 1]
        
        return confidence

class PseudobulkPerturbationModel(PerturbationModel):
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
        decoder: Optional[DecoderInterface] = None,
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
            decoder=decoder,
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
        self.cell_sentence_len =  self.transformer_backbone_kwargs['n_positions']
        self.gene_dim = gene_dim
        self.batch_dim = batch_dim
        self.residual_decoder = kwargs.get("residual_decoder", False)

        # Build the distributional loss from geomloss
        blur = kwargs.get("blur", 0.05)
        loss_name = kwargs.get("loss", "energy")
        if loss_name == "energy":
            self.loss_fn = SamplesLoss(loss=self.distributional_loss, blur=blur)
        elif loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "scaled_energy":
            scale_factor = 512.0 / float(self.cell_sentence_len)
            base_loss = SamplesLoss(loss=self.distributional_loss, blur=blur)
            self.loss_fn = ScaledSamplesLoss(base_loss, scale_factor)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

        # Build the underlying neural OT network
        self._build_networks()

        self.batch_encoder = None
        if kwargs.get("batch_encoder", False) and batch_dim is not None:
            self.batch_encoder = build_mlp(
                in_dim=batch_dim,
                out_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_encoder_layers,
                dropout=self.dropout,
                activation=self.activation_class,
            )

        # if the model is outputting to counts space, apply softplus
        # otherwise its in embedding space and we don't want to
        is_gene_space = kwargs['embed_key'] == 'X_hvg' or kwargs['embed_key'] is None
        if kwargs.get('softplus', False) and is_gene_space:
            # actually just set this to a relu for now
            self.softplus = torch.nn.ReLU()

        if 'confidence_head' in kwargs and kwargs['confidence_head']:
            self.confidence_head = ConfidenceHead(hidden_dim, pooling_method='attention')
            self.confidence_loss_fn = nn.MSELoss()
        else:
            self.confidence_head = None
            self.confidence_loss_fn = None

        # self.freeze_pert = kwargs.get("freeze_pert", False)
        # if self.freeze_pert:
        #     modules_to_freeze = [
        #         self.pert_encoder,
        #         self.basal_encoder,
        #         self.transformer_backbone,
        #         self.project_out,
        #         self.convolve,
        #     ]
        #     for module in modules_to_freeze:
        #         for param in module.parameters():
        #             param.requires_grad = False

        if kwargs.get("nb_decoder", False):
            self.gene_decoder = NBDecoder(
                latent_dim=self.output_dim + (self.batch_dim or 0),
                gene_dim=gene_dim,
                hidden_dims=[512, 512, 512],
                dropout=self.dropout,
            )

        if kwargs.get("transformer_decoder", False):
            from models.decoders import TransformerLatentToGeneDecoder
            self.gene_decoder = TransformerLatentToGeneDecoder(
                latent_dim=self.output_dim,
                gene_dim=self.gene_dim,
                num_layers=self.n_decoder_layers,
                dropout=self.dropout,
                cell_sentence_len=self.cell_sentence_len,
                softplus=kwargs.get("softplus", False),
            )

        control_pert = kwargs.get("control_pert", "non-targeting")
        if kwargs.get("finetune_vci_decoder", False):
            gene_names = []

            if output_space == 'gene':
                # hvg's but for which dataset?
                if 'DMSO_TF' in control_pert:
                    gene_names = np.load('/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy', allow_pickle=True)
                elif 'non-targeting' in control_pert:
                    temp = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5')
                    gene_names = temp.var.index.values
            else:
                assert output_space == 'all'
                if 'DMSO_TF' in control_pert:
                    gene_names = np.load('/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy', allow_pickle=True)
                elif 'non-targeting' in control_pert:
                    # temp = ad.read_h5ad('/scratch/ctc/ML/vci/paper_replogle/jurkat.h5')
                    # gene_names = temp.var.index.values
                    temp = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/cross_dataset/replogle/jurkat.h5')
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

        self.convolve = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)

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
            pert = batch["pert"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["basal"].reshape(-1, self.cell_sentence_len, self.input_dim)
        else:
            # we are inferencing on a single batch, so accept variable length sentences
            pert = batch["pert"].reshape(1, -1, self.pert_dim)
            basal = batch["basal"].reshape(1, -1, self.input_dim)

        # Shape: [B, S, hidden_dim]
        pert_embedding = self.encode_perturbation(pert)
        control_cells = self.encode_basal_expression(basal)        

        seq_input = torch.cat([pert_embedding, control_cells], dim=2) # Shape: [B, S, 2 * hidden_dim]
        seq_input = self.convolve(seq_input)  # Shape: [B, S, hidden_dim]

        batch_size = seq_input.shape[0]


        if self.batch_encoder is not None:
            if padded:
                batch = batch["gem_group"].reshape(-1, self.cell_sentence_len, self.batch_dim)
            else:
                batch = batch["gem_group"].reshape(1, -1, self.batch_dim)
            
            seq_input = seq_input + self.batch_encoder(batch)  # Shape: [B, S, hidden_dim]
        
        # take the average across the sequence dimension
        seq_input = seq_input.mean(dim=1, keepdim=True)  # Shape: [B, 1, hidden_dim]

        # forward pass + extract CLS last hidden state
        res_pred = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state # Shape: [B, 1, hidden_dim]
        out_dim = res_pred.shape[-1]

        # broadcast to the sequence length
        if padded:
            res_pred = res_pred.expand(batch_size, self.cell_sentence_len, out_dim)  # Shape: [B, S, hidden_dim]
        else:
            res_pred = res_pred.expand(1, -1, out_dim)

        # add to basal if predicting residual
        if self.predict_residual:
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        # apply softplus if specified and we output to HVG space
        is_gene_space = self.hparams['embed_key'] == 'X_hvg' or self.hparams['embed_key'] is None
        if self.hparams.get('softplus', False) and is_gene_space:
            out_pred = self.softplus(out_pred)

        output = out_pred.reshape(-1, self.output_dim)

        if self.confidence_head is not None:
            confidence = self.confidence_head(res_pred)
            return output, confidence
        else:
            return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        confidence_pred = None
        if self.confidence_head is None:
            pred = self.forward(batch, padded=padded)
        else:
            pred, confidence_pred = self.forward(batch, padded=padded)

        target = batch["X"]

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

        if self.gene_decoder is not None and "X_hvg" in batch:
            gene_targets = batch["X_hvg"]
            # Train decoder to map latent predictions to gene space
            latent_preds = pred
            # with torch.no_grad():
            #     latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model
            
            batch_var = batch["gem_group"].reshape(latent_preds.shape[0], latent_preds.shape[1], -1)
            # concatenate on the last axis
            if self.batch_dim is not None and not isinstance(self.gene_decoder, FinetuneVCICountsDecoder):
                latent_preds = torch.cat([latent_preds, batch_var], dim=-1)

            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["X_hvg"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                gene_preds = self.gene_decoder(latent_preds)
                if self.residual_decoder:
                    basal_hvg = batch["basal_hvg"].reshape(gene_preds.shape)
                    gene_preds = gene_preds + basal_hvg.mean(dim=1, keepdim=True).expand_as(gene_preds)
                if padded:
                    gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_decoder.gene_dim())
                else:
                    gene_targets = gene_targets.reshape(1, -1, self.gene_decoder.gene_dim())

                decoder_loss = self.loss_fn(gene_preds, gene_targets).mean()
                
            # Log decoder loss
            self.log("decoder_loss", decoder_loss)
            
            total_loss = total_loss + 0.1 * decoder_loss

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = main_loss.detach().clone().unsqueeze(0)

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred, loss_target)
            self.log("train/confidence_loss", confidence_loss)
            
            # Add to total loss
            total_loss = total_loss + confidence_loss
        
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        confidence_pred = None
        if self.confidence_head is None:
            pred = self.forward(batch)
        else:
            pred, confidence_pred = self(batch)

        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["X"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)

        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)

        if self.gene_decoder is not None and "X_hvg" in batch:
            gene_targets = batch["X_hvg"] 

            # Get model predictions from validation step
            latent_preds = pred

            # Train decoder to map latent predictions to gene space
            batch_var = batch["gem_group"].reshape(latent_preds.shape[0], latent_preds.shape[1], -1)
            # concatenate on the last axis
            if self.batch_dim is not None and not isinstance(self.gene_decoder, FinetuneVCICountsDecoder):
                latent_preds = torch.cat([latent_preds, batch_var], dim=-1)
            if isinstance(self.gene_decoder, NBDecoder):
                mu, theta = self.gene_decoder(latent_preds)
                gene_targets = batch["X_hvg"].reshape_as(mu)
                decoder_loss = nb_nll(gene_targets, mu, theta)
            else:
                gene_preds = self.gene_decoder(latent_preds) # verify this is automatically detached
                if self.residual_decoder:
                    basal_hvg = batch["basal_hvg"].reshape(gene_preds.shape)
                    gene_preds = gene_preds + basal_hvg.mean(dim=1, keepdim=True).expand_as(gene_preds)
                
                # Get decoder predictions
                gene_preds = gene_preds.reshape(-1, self.cell_sentence_len, self.gene_dim)
                gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
                decoder_loss = self.loss_fn(gene_preds, gene_targets).mean()
            
            # Log the validation metric
            self.log("decoder_val_loss", decoder_loss)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone().unsqueeze(0)

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred, loss_target)
            self.log("val/confidence_loss", confidence_loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        confidence_pred = None
        if self.confidence_head is None:
            pred = self.forward(batch, padded=False)
        else:
            pred, confidence_pred = self.forward(batch, padded=False)
        target = batch["X"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

        if confidence_pred is not None:
            # Detach main loss to prevent gradients flowing through it
            loss_target = loss.detach().clone().unsqueeze(0)

            # Compute confidence loss
            confidence_loss = self.confidence_loss_fn(confidence_pred, loss_target)
            self.log("test/confidence_loss", confidence_loss)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        if self.confidence_head is None:
            latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
        else:
            latent_output, confidence_pred = self.forward(batch, padded=padded)

        output_dict = {
            "preds": latent_output,
            "X": batch.get("X", None),
            "X_hvg": batch.get("X_hvg", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "gem_group": batch.get("gem_group", None),
            "basal": batch.get("basal", None),
        }

        basal_hvg = batch.get("basal_hvg", None)

        if self.gene_decoder is not None:
            if latent_output.dim() == 2:
                batch_var = batch["gem_group"].reshape(latent_output.shape[0], -1)
            else:
                batch_var = batch["gem_group"].reshape(latent_output.shape[0], latent_output.shape[1], -1)
            # concatenate on the last axis
            if self.batch_dim is not None and not isinstance(self.gene_decoder, FinetuneVCICountsDecoder):
                latent_output = torch.cat([latent_output, batch_var], dim=-1)
            if isinstance(self.gene_decoder, NBDecoder):
                mu, _ = self.gene_decoder(latent_output)
                gene_preds = mu
            else:
                gene_preds = self.gene_decoder(latent_output)
            if self.residual_decoder and basal_hvg is not None:
                basal_hvg = basal_hvg.reshape(gene_preds.shape)
                gene_preds = gene_preds + basal_hvg.mean(dim=1, keepdim=True).expand_as(gene_preds)
            output_dict["gene_preds"] = gene_preds

        return output_dict

    # def configure_optimizers(self):
    #     read_depth_lr_multiplier = 100

    #     read_depth_params = []
    #     other_params = []

    #     for name, param in self.named_parameters():
    #         if "read_depth" in name:
    #             read_depth_params.append(param)
    #         else:
    #             other_params.append(param)
        
    #     optimizer = torch.optim.Adam([
    #         {'params': other_params, 'lr': self.lr},
    #         {'params': read_depth_params, 'lr': self.lr * read_depth_lr_multiplier}
    #     ])
    #     return optimizer
