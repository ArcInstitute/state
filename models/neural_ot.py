# File: models/neural_ot.py
import torch
import numpy as np

from collections import defaultdict
from geomloss import SamplesLoss
from typing import Optional, Dict, List

from models.base import PerturbationModel
from models.decoders import DecoderInterface
from models.utils import build_mlp, get_activation_class, get_transformer_backbone

def uncollate_batch(batch: Dict[str, torch.Tensor], sentence_len: int) -> List[Dict[str, torch.Tensor]]:
    """
    Uncollates a batch dictionary where each tensor has shape [B*S, ...] into
    a list of S dictionaries with shape [B, ...], where:
    B = batch size (e.g. 512)
    S = sentence length (e.g. 32)
    
    Args:
        batch: Dictionary of tensors, each with first dimension B*S
        sentence_len: The length S to split into
        
    Returns:
        List of S dictionaries, each containing tensors of first dimension B
    """
    total_size = batch['X'].shape[0]
    batch_size = total_size // sentence_len
    
    uncollated_batches = []
    
    for i in range(batch_size):
        start_idx = i * sentence_len
        end_idx = (i + 1) * sentence_len

        current_batch = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                current_batch[key] = tensor[start_idx:end_idx]
            else:
                # Handle non-tensor data (e.g. lists)
                current_batch[key] = tensor[start_idx:end_idx]
                
        uncollated_batches.append(current_batch)
        
    return uncollated_batches

def should_cache_batch(pert_names: List[str], cache_prob: float = 0.01) -> bool:
    """
    Determines if a batch should be cached based on perturbation names
    and random sampling probability.
    
    Args:
        pert_names: List of perturbation names for the batch
        cache_prob: Probability of caching non-control batches
        
    Returns:
        Boolean indicating if batch should be cached
    """
    # Check if this is a control batch
    is_control = pert_names[0] in ["DMSO_TF", "non-targeting"]
    
    # Cache if control or random sample
    return is_control or np.random.rand() < cache_prob

class NeuralOTPerturbationModel(PerturbationModel):
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

        # Build the distributional loss from geomloss
        self.loss_fn = SamplesLoss(loss=self.distributional_loss)
        # self.loss_fn = LearnableAlignmentLoss()

        # Build the underlying neural OT network
        self._build_networks()

        # For caching validation data across steps, if desired
        self.val_cache = defaultdict(list)

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

        precision = self.get_precision()
        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            self.transformer_backbone_key,
            self.transformer_backbone_kwargs,
            precision,
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

    # TODO - add a flexible forward method that can take ragged tensors by sampling by replacement
    # to pad, forward passing, and taking only the original indices to avoid repeated samples in 
    # our test set.

    def forward(self, batch: dict, padded=True) -> torch.Tensor:
        """
        The main forward call. Batch is a flattened sequence of cell sentences,
        which we reshape into sequences of length cell_sentence_len.
        
        Expects input tensors of shape (B, S, N) where:
        B = batch size
        S = sequence length (cell_sentence_len)
        N = feature dimension
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
        
        # forward pass + extract CLS last hidden state
        res_pred = self.transformer_backbone(inputs_embeds=seq_input).last_hidden_state

        # add to basal if predicting residual
        if self.predict_residual:
            # treat the actual prediction as a residual sum to basal
            out_pred = self.project_out(res_pred + control_cells)
        else:
            out_pred = self.project_out(res_pred)

        return out_pred.reshape(-1, self.output_dim)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        # Get model predictions (in latent space)
        pred = self(batch)
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["X"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        main_loss = self.loss_fn(pred, target).mean()
        self.log("train_loss", main_loss)
        
        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "X_hvg" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model
            
            gene_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["X_hvg"]
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(gene_preds, gene_targets).mean()
            
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
        target = batch["X"]
        target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)

        pred = pred.reshape(-1, self.output_dim)
        target = target.reshape(-1, self.output_dim)
        
        # Split batch into sentences
        uncollated_batches = uncollate_batch(batch, self.cell_sentence_len)
        
        # Process each sentence
        for idx, current_batch in enumerate(uncollated_batches):
            if should_cache_batch(current_batch["pert_name"]):
                current_pred = pred[idx*self.cell_sentence_len:(idx+1)*self.cell_sentence_len]
                self._update_val_cache(current_batch, current_pred)

        return {"loss": loss, "predictions": pred}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        """Track decoder performance during validation without training it."""
        if self.gene_decoder is not None and "X_hvg" in batch:
            # Get model predictions from validation step
            latent_preds = outputs["predictions"].detach()

            # Train decoder to map latent predictions to gene space
            gene_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["X_hvg"] 
            
            # Get decoder predictions
            gene_preds = gene_preds.reshape(-1, self.cell_sentence_len, self.gene_dim)
            gene_targets = gene_targets.reshape(-1, self.cell_sentence_len, self.gene_dim)
            decoder_loss = self.loss_fn(gene_preds, gene_targets).mean()
            
            # Log the validation metric
            self.log("decoder_val_loss", decoder_loss)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self.forward(batch, padded=False)
        target = batch["X"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)
        pred = pred.reshape(-1, self.output_dim)
        target = target.reshape(-1, self.output_dim)
        
        # Split batch into sentences
        uncollated_batches = uncollate_batch(batch, self.cell_sentence_len)
        
        # Process each sentence
        for idx, current_batch in enumerate(uncollated_batches):
            if should_cache_batch(current_batch["pert_name"]):
                current_pred = pred[idx*self.cell_sentence_len:(idx+1)*self.cell_sentence_len]
                self._update_test_cache(current_batch, current_pred)

    def predict_step(self, batch, batch_idx, padded=True, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch, padded=padded)  # shape [B, ...]
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
