from typing import Dict

import torch

from .base import PerturbationModel
from .utils import build_mlp, get_activation_class


class EmbedSumPerturbationModel(PerturbationModel):
    """
    Implementation of the EmbedSum model which treats perturbations as learned embeddings
    that are added to control cell representations, which are input as gene expression counts
    or as embeddings from a foundation model (UCE, scGPT, etc). The outputs are always in
    gene expression space.

    This model:
    1. Learns a co-embedding space for perturbations and cell states
    2. Computes perturbation effects in this space
    3. Decoder maps perturbed embeddings to gene expression space

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
        output_space: str = "gene",
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
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.dropout = kwargs.get("dropout", 0.1)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))
        self.kwargs = kwargs

        # Build model components
        self._build_networks()

    def _build_networks(self):
        """
        Build the core components:
        1. Perturbation encoder: maps one-hot to learned embedding
        2. Decoder: maps perturbed embedding to gene space
        """
        # Map perturbation to effect in embedding space
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

        self.project_out = build_mlp(
            in_dim=self.hidden_dim,
            out_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Map perturbation to an effect vector in embedding space."""
        return self.pert_encoder(pert)

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Expression is already in embedding space, pass through."""
        return self.basal_encoder(expr)

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Given a perturbation and basal embeddings, compute the perturbed embedding.
        """
        # Project perturbation and basal cell state to latent space
        perturbation = self.encode_perturbation(pert)
        basal_encoded = self.basal_encoder(basal)

        # Add perturbation to basal embedding
        perturbed_encoded = basal_encoded + perturbation
        return perturbed_encoded

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Given

        Args:
            batch: Dictionary containing:
                - pert: Perturbation one-hot
                - basal: Control expression embedding
        """
        pert = batch["pert_emb"]
        basal = batch["ctrl_cell_emb"]

        # compute perturbed cell state to perturbation/cell co-embedding space
        perturbed_encoded = self.perturb(pert, basal)

        # Decode to gene space or to input cell embedding space
        return self.project_out(perturbed_encoded)
