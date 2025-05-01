import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from utils import UCEGenePredictor

# set up logger
import logging

logger = logging.getLogger(__name__)

import sys
import os

from vci_pretrain.vci.inference import Inference
from vci_pretrain.vci.finetune_decoder import Finetune

class DecoderInterface(ABC):
    """
    An interface for decoding latent model outputs into DE genes.

    Specific implementations may pass in their own kwargs.
    """

    @abstractmethod
    def compute_de_genes(self, adata_latent, **kwargs):
        """
        Given an AnnData whose .X is in latent space, return a data frame of DE genes.
        """
        pass

class TransformerLatentToGeneDecoder(nn.Module):
    """
    A transformer-based decoder to map latent embeddings (with shape [B, S, latent_dim])
    to gene expression space (with shape [B, S, gene_dim]). It reshapes the input
    appropriately and applies a transformer encoder before a linear output layer.
    """
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        cell_sentence_len: int = 128,
        softplus: bool = False,
        inner_dim=256,
    ):
        super().__init__()
        # Create a transformer encoder layer.
        # Linear layer mapping 
        self.input_layer = nn.Linear(latent_dim, inner_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=inner_dim, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dim = gene_dim
        
        # Linear layer mapping transformer output to gene space.
        self.output_layer = nn.Linear(inner_dim, gene_dim)
        
        self.softplus = softplus
        if softplus:
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def gene_dim(self):
        return self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: latent embeddings of shape [B, S, latent_dim]
        Returns:
            Gene predictions of shape [B, S, gene_dim]
        """
        # Transformer expects input of shape [S, B, latent_dim]
        x = self.input_layer(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FinetuneVCICountsDecoder(nn.Module):
    def __init__(
        self,
        genes,
        # model_loc="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/step=950000.ckpt",
        # config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/tahoe_config.yaml",
        model_loc="/home/aadduri/vci_pretrain/vci_1.4.2.ckpt",
        config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/large_1e-4_rda_tabular_counts_2048/crossds_config.yaml",
        read_depth=70,
        latent_dim=1024, # dimension of pretrained vci model
        hidden_dims=[256, 512], # hidden dimensions of the decoder
        dropout=0.1,
        basal_residual=False,
    ):
        super().__init__()
        self.genes = genes
        self.model_loc = model_loc
        self.config = config
        self.finetune = Finetune(OmegaConf.load(self.config))
        self.finetune.load_model(self.model_loc)
        self.read_depth = nn.Parameter(torch.tensor(read_depth, dtype=torch.float), requires_grad=True)
        self.basal_residual = basal_residual

        # layers = [
        #     nn.Linear(latent_dim, hidden_dims[0]),
        # ]
        
        # self.gene_lora = nn.Sequential(*layers)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], len(self.genes)),
            nn.ReLU(),
        )

        self.binary_decoder = self.finetune.model.binary_decoder
        for param in self.binary_decoder.parameters():
            param.requires_grad = False


    def gene_dim(self):
        return len(self.genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, latent_dim].
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        batch_size, seq_len, latent_dim = x.shape
        x = x.view(batch_size * seq_len, latent_dim)
        
        # Get gene embeddings
        gene_embeds = self.finetune.get_gene_embedding(self.genes)
        
        # Handle RDA task counts
        use_rda = getattr(self.finetune.model.cfg.model, "rda", False)
        # Define your sub-batch size (tweak this based on your available memory)
        sub_batch_size = 32
        logprob_chunks = []  # to store outputs of each sub-batch

        for i in range(0, x.shape[0], sub_batch_size):
            # Get the sub-batch of latent vectors
            x_sub = x[i: i + sub_batch_size]

            # Create task_counts for the sub-batch if needed
            if use_rda:
                # task_counts_sub = torch.full(
                #     (x_sub.shape[0],), self.read_depth, device=x.device
                # )
                task_counts_sub = torch.ones((x_sub.shape[0],), device=x.device) * self.read_depth
            else:
                task_counts_sub = None

            # Compute merged embeddings for the sub-batch
            merged_embs_sub = self.finetune.model.resize_batch(x_sub, gene_embeds, task_counts_sub)

            # Run the binary decoder on the sub-batch
            logprobs_sub = self.binary_decoder(merged_embs_sub)

            # Squeeze the singleton dimension if needed
            if logprobs_sub.dim() == 3 and logprobs_sub.size(-1) == 1:
                logprobs_sub = logprobs_sub.squeeze(-1)

            # Collect the results
            logprob_chunks.append(logprobs_sub)

        # Concatenate the sub-batches back together
        logprobs = torch.cat(logprob_chunks, dim=0)

        # Reshape back to [B, S, gene_dim]
        decoded_gene = logprobs.view(batch_size, seq_len, len(self.genes))
        # decoded_gene = torch.nn.functional.relu(decoded_gene)

        # # normalize the sum of decoded_gene to be read depth
        # decoded_gene = decoded_gene / decoded_gene.sum(dim=2, keepdim=True) * self.read_depth

        # decoded_gene = self.gene_lora(decoded_gene)
        # TODO: fix this to work with basal counts
        
        # add logic for basal_residual:
        decoded_x = self.latent_decoder(x)
        decoded_x = decoded_x.view(batch_size, seq_len, len(self.genes))

        # Pass through the additional decoder layers
        return decoded_gene + decoded_x
    

class VCICountsDecoder(DecoderInterface):
    def __init__(
        self,
        model_loc="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/step=770000.ckpt",
        config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/replogle_config.yaml",
        read_depth=1000,
    ):
        self.model_loc = model_loc
        self.config = config
        self.inference = Inference(OmegaConf.load(self.config))
        self.inference.load_model(self.model_loc)
        self.read_depth = read_depth
    
    def compute_de_genes(self, adata_latent, pert_col, control_pert, genes):
        logger.info("Computing DE genes using VCI counts decoder.")
        # initialize an empty matrix to store the decoded counts
        decoded_counts = np.zeros((adata_latent.shape[0], len(genes)))
        # read from key .X since predictions are stored in .X
        batch_size = 64
        start_idx = 0
        for pred_counts in self.inference.decode_from_adata(adata_latent, genes, 'X', read_depth=self.read_depth, batch_size=batch_size):
            # pred_counts is count data over the gene space (hvg or transcriptome)
            # update the entries of decoded_counts with the predictions
            current_batch_size = pred_counts.shape[0]
            decoded_counts[start_idx:start_idx+current_batch_size, :] = pred_counts
            start_idx += current_batch_size
        

        # undo the log transformation
        decoded_counts = np.expm1(decoded_counts)
        # normalize the total to 10000
        decoded_counts = decoded_counts / decoded_counts.sum(axis=1)[:, None] * 10000

        probs_df = pd.DataFrame(decoded_counts)
        probs_df["pert"] = adata_latent.obs[pert_col].values
        mean_df = probs_df.groupby("pert").mean()

        ctrl = mean_df.loc[control_pert].values
        pert_effects = np.abs(mean_df - ctrl)

        sorted_indices = np.argsort(pert_effects.values, axis=1)[:, ::-1]
        sorted_genes = np.array(genes)[sorted_indices]
        de_genes = pd.DataFrame(sorted_genes)
        de_genes.index = pert_effects.index.values
        return de_genes


class UCELogProbDecoder(DecoderInterface):
    """
    This class decodes UCE embeddings into log probabilities of expression in gene space.
    """

    def __init__(
        self,
        model_loc="/large_storage/ctc/ML/data/cell/misc/model_used_in_paper_33l_8ep_1024t_1280.torch",
    ):
        self.model_loc = model_loc

    def compute_de_genes(self, adata_latent, pert_col, control_pert, genes):
        """
        Compute DE in gene space using UCE predictions, by decoding probability in each gene and
        manipulating the log probs as a statistical test.
        """
        logger.info(f"Computing DE genes using UCE log probs decoder.")
        gene_predictor = UCEGenePredictor(device="cuda:0", model_loc=self.model_loc)
        gene_logprobs = gene_predictor.compute_gene_prob_group_batched(adata_latent.X, genes, batch_size=96)
        probs_df = pd.DataFrame(gene_logprobs)
        probs_df["pert"] = adata_latent.obs[pert_col].values
        mean_df = probs_df.groupby("pert").mean()

        ctrl = mean_df.loc[control_pert].values
        pert_effects = np.abs(mean_df - ctrl)

        sorted_indices = np.argsort(pert_effects.values, axis=1)[:, ::-1]
        sorted_genes = np.array(genes)[sorted_indices]
        de_genes = pd.DataFrame(sorted_genes)
        de_genes.index = pert_effects.index.values
        return de_genes
