import logging
import torch
from torch import nn

from vci.nn.model import StateEmbeddingModel
from vci.train.trainer import get_embeddings
from vci.utils import get_embedding_cfg

log = logging.getLogger(__name__)


class Finetune:
    def __init__(self, cfg, learning_rate=1e-4):
        """
        Initialize the Finetune class for fine-tuning the binary decoder of a pre-trained model.

        Parameters:
        -----------
        cfg : OmegaConf
            Configuration object containing model settings
        learning_rate : float
            Learning rate for fine-tuning the binary decoder
        """
        self.model = None
        self.collator = None
        self.protein_embeds = None
        self._vci_conf = cfg
        self.learning_rate = learning_rate
        self.cached_gene_embeddings = {}
        self.device = None

    def load_model(self, checkpoint):
        """
        Load a pre-trained model from a checkpoint and prepare it for fine-tuning.

        Parameters:
        -----------
        checkpoint : str
            Path to the checkpoint file
        """
        if self.model:
            raise ValueError("Model already initialized")

        # Import locally to avoid circular imports

        # Load and initialize model for eval
        self.model = StateEmbeddingModel.load_from_checkpoint(checkpoint, strict=False)

        # Ensure model uses the provided config, not the stored one
        if self._vci_conf is not None:
            self.model.update_config(self._vci_conf)

        self.device = self.model.device

        # Load protein embeddings
        all_pe = get_embeddings(self._vci_conf)
        all_pe.requires_grad = False
        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.device)

        # Load protein embeddings
        self.protein_embeds = torch.load(get_embedding_cfg(self._vci_conf).all_embeddings)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients only for binary decoder
        for param in self.model.binary_decoder.parameters():
            param.requires_grad = False

        # Ensure the binary decoder is in training mode so gradients are enabled.
        self.model.binary_decoder.eval()

    def get_gene_embedding(self, genes):
        """
        Get embeddings for a list of genes, with caching to avoid recomputation.

        Parameters:
        -----------
        genes : list
            List of gene names/identifiers

        Returns:
        --------
        torch.Tensor
            Tensor of gene embeddings
        """
        # Cache key based on genes tuple
        cache_key = tuple(genes)

        # Return cached embeddings if available
        if cache_key in self.cached_gene_embeddings:
            return self.cached_gene_embeddings[cache_key]

        # Compute gene embeddings
        protein_embeds = [self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(5120) for x in genes]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        gene_embeds = self.model.gene_embedding_layer(protein_embeds)

        # Cache and return
        self.cached_gene_embeddings[cache_key] = gene_embeds
        return gene_embeds

    def get_counts(self, cell_embs, genes, read_depth=None, batch_size=32):
        """
        Generate predictions with the binary decoder with gradients enabled.

        Parameters:
        - cell_embs: A tensor or array of cell embeddings.
        - genes: List of gene names.
        - read_depth: Optional read depth for RDA normalization.
        - batch_size: Batch size for processing.

        Returns:
        A single tensor of shape [N, num_genes] where N is the total number of cells.
        """

        # Convert cell_embs to a tensor on the correct device.
        cell_embs = torch.tensor(cell_embs, dtype=torch.float, device=self.device)

        # Check if RDA is enabled.
        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 1000.0

        # Retrieve gene embeddings (cached if available).
        gene_embeds = self.get_gene_embedding(genes)

        # List to collect the output predictions for each batch.
        output_batches = []

        # Loop over cell embeddings in batches.
        for i in range(0, cell_embs.size(0), batch_size):
            # Determine batch indices.
            end_idx = min(i + batch_size, cell_embs.size(0))
            cell_embeds_batch = cell_embs[i:end_idx]

            # Set up task counts if using RDA.
            if use_rda:
                task_counts = torch.full((cell_embeds_batch.shape[0],), read_depth, device=self.device)
            else:
                task_counts = None

            # Resize the batch using the model's method.
            merged_embs = self.model.resize_batch(cell_embeds_batch, gene_embeds, task_counts)

            # Forward pass through the binary decoder.
            logprobs_batch = self.model.binary_decoder(merged_embs)

            # If the output has an extra singleton dimension (e.g., [B, gene_dim, 1]), squeeze it.
            if logprobs_batch.dim() == 3 and logprobs_batch.size(-1) == 1:
                logprobs_batch = logprobs_batch.squeeze(-1)

            output_batches.append(logprobs_batch)

        # Concatenate all batch outputs along the first dimension.
        return torch.cat(output_batches, dim=0)
