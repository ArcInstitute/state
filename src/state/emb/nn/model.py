import warnings

warnings.filterwarnings("ignore")

import math
import logging
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
import lightning as L

import sys

sys.path.append("../../")
sys.path.append("../")

from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss


from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR, ReduceLROnPlateau

from ..utils import (
    get_embedding_cfg,
    get_dataset_cfg,
)
from .loss import WassersteinLoss, KLDivergenceLoss, MMDLoss, TabularLoss


from .flash_transformer import FlashTransformerEncoderLayer
from .flash_transformer import FlashTransformerEncoder


class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))

        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


def nanstd(x):
    return torch.sqrt(torch.nanmean(torch.pow(x - torch.nanmean(x, dim=-1).unsqueeze(-1), 2), dim=-1))


class StateEmbeddingModel(L.LightningModule):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        dropout: float = 0.0,
        warmup_steps: int = 0,
        compiled: bool = False,
        max_lr=4e-4,
        emb_cnt=145469,
        emb_size=5120,
        cfg=None,
        collater=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.compiled = compiled
        self.model_type = "Transformer"
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.max_lr = max_lr
        self.collater = collater
        # Encodes Tokens
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model, bias=True),
            nn.LayerNorm(d_model),  # Moved before activation
            nn.SiLU(),  # Changed to SiLU
        )

        # Create a list of FlashTransformerEncoderLayer instances
        layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) for _ in range(nlayers)]
        self.transformer_encoder = FlashTransformerEncoder(layers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.d_model = d_model
        self.dropout = dropout

        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        self.binary_decoder = nn.Sequential(
            SkipBlock(output_dim + d_model + self.z_dim),
            SkipBlock(output_dim + d_model + self.z_dim),
            nn.Linear(output_dim + d_model + self.z_dim, 1, bias=True),
        )

        if self.cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Encodes Tokens for Decoder
        self.gene_embedding_layer = self.encoder  # reuse this layer

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = (
            None  # TODO: make this cleaner for the type checker, right now it gets set externally after model init
        )
        self.step_ctr = 0

        self.true_top_genes = None
        self.protein_embeds = None

        self._last_val_de_check = 0
        self._last_val_perturbation_check = 0

        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
            self.dataset_embedder = nn.Linear(output_dim, self.z_dim_ds)

            # Assume self.cfg.model.num_datasets is set to the number of unique datasets.
            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )

            # this should be a classification label loss
            self.dataset_loss = nn.CrossEntropyLoss()
        else:
            self.dataset_token = None

    def on_save_checkpoint(self, checkpoint):
        """
        Persist a snapshot of the training config inside the checkpoint so downstream
        inference/eval can run without an external config file.
        """
        try:
            if self.cfg is not None:
                # Store both a YAML snapshot and a resolved container for robustness
                checkpoint["cfg_yaml"] = OmegaConf.to_yaml(self.cfg)
        except Exception:
            # Never block checkpointing if config serialization fails
            pass

        # Also package protein embeddings for standalone inference/transform.
        try:
            if self.protein_embeds is not None:
                pe = self.protein_embeds
            else:
                # Load from configured path as a fallback
                from ..utils import get_embedding_cfg

                pe = torch.load(get_embedding_cfg(self.cfg).all_embeddings, map_location="cpu", weights_only=False)
            # Ensure CPU tensors in the dictionary
            if isinstance(pe, dict):
                cpu_pe = {}
                for k, v in pe.items():
                    try:
                        cpu_pe[k] = v.detach().to("cpu") if hasattr(v, "detach") else torch.tensor(v, device="cpu")
                    except Exception:
                        cpu_pe[k] = v
                checkpoint["protein_embeds_dict"] = cpu_pe
        except Exception:
            # Do not block checkpoint save if embedding packaging fails
            pass

    def _compute_embedding_for_batch(self, batch):
        batch_sentences = batch[0].to(self.device)
        X = batch[1].to(self.device)
        Y = batch[2]
        batch_weights = batch[4]
        mask = batch[5]
        mask = mask.to(torch.bool)
        batch_sentences_counts = batch[7]
        if batch_sentences_counts is not None:
            batch_sentences_counts = batch_sentences_counts.to(self.device)
        dataset_nums = batch[8]
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(self.device)

        # convert the cell sentence and task sentence into embeddings
        batch_sentences = self.pe_embedding(batch_sentences)
        X = self.pe_embedding(X)

        # Normalize token outputs now
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)

        # Add a learnable CLS token to the beginning of the sentence
        batch_sentences[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)

        # Optionally add a learnable dataset token to the end of the sentence
        if self.dataset_token is not None:
            dataset_token = self.dataset_token.expand(batch_sentences.size(0), -1).unsqueeze(1)
            batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
            # concatenate a False to the mask on dim 1
            mask = torch.cat((mask, torch.zeros(mask.size(0), 1, device=mask.device).bool()), dim=1)

        # mask out the genes embeddings that appear in the task sentence
        _, embedding, dataset_emb = self.forward(
            batch_sentences, mask=mask, counts=batch_sentences_counts, dataset_nums=dataset_nums
        )

        X = self.gene_embedding_layer(X)
        return X, Y, batch_weights, embedding, dataset_emb

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(get_embedding_cfg(self.cfg).all_embeddings, weights_only=False)

        protein_embeds = [
            self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(get_embedding_cfg(self.cfg).size)
            for x in genes
        ]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        if protein_embeds.sum() == 0:
            raise ValueError("No gene embeddings found")

        return self.gene_embedding_layer(protein_embeds)

    @staticmethod
    def resize_batch(cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None):
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        if sampled_rda is not None:
            # computes mu and std dev from Y
            reshaped_counts = sampled_rda.unsqueeze(1)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        elif task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        else:
            # Original behavior if total_counts is None
            combine = torch.cat((A, B), dim=2)

        if ds_emb is not None:
            # ds_emb is a tensor of shape (batch_size, 10). concatenate it to the combine tensor
            ds_emb = ds_emb.unsqueeze(1).repeat(1, A.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        return combine

    def forward(self, src: Tensor, mask: Tensor, counts=None, dataset_nums=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        if counts is not None:
            # scFoundation-style soft binning for counts
            counts = counts.unsqueeze(-1)  # now B x H x 1

            # Step 1: Transform count values into bin distribution
            bin_weights = self.count_encoder(counts)  # B x H x 10
            bin_weights = F.softmax(bin_weights, dim=-1)  # Convert to probabilities over bins

            # Step 2: Get bin embeddings
            bin_indices = torch.arange(10, device=self.device)  # 10 bins
            bin_embeddings = self.bin_encoder(bin_indices)  # 10 x d_model

            # Step 3: Compute weighted sum of bin embeddings
            count_emb = torch.matmul(bin_weights, bin_embeddings)

            if self.dataset_token is not None:
                # append B x 1 x d_model to count_emb of all zeros
                dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=self.device)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)  # B x H x d_model

            # Add count embeddings to token embeddings
            src = (
                src + count_emb
            )  # should both be B x H x self.d_model, or B x H + 1 x self.d_model if dataset correction

        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)  # batch x seq_len x 128
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :]  # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1)  # Normalize.

        # we must be in train mode to use dataset correction
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        return gene_output, embedding, dataset_emb

    def _log_nonzero_elements_stats(self, batch_sentences, prefix="trainer"):
        """
        Track and log non-zero elements in the sentence for ablation study.

        This function analyzes the mask tensor to count how many positions in each cell sentence
        contain expressed genes (non-zero) versus unexpressed/padded genes (zero). This is useful
        for understanding how padding with unexpressed gene embeddings affects model learning.

        Args:
            batch_sentences: Boolean tensor of shape (batch_size, seq_len) where True indicates
                  non-zero (expressed) genes and False indicates zero (unexpressed/padded) genes.
            prefix: String prefix for logging (e.g., "trainer" or "validation")

        Logs:
            - {prefix}/avg_nonzero_genes: Average number of non-zero genes per cell
            - {prefix}/nonzero_fraction: Fraction of non-zero genes relative to total slots
        """
        # Count non-zero elements per cell in the batch
        nonzero_counts = batch_sentences.sum(dim=1)  # Sum across sequence dimension
        avg_nonzero = nonzero_counts.float().mean().item()

        # Calculate the fraction of non-zero elements (excluding CLS token)
        total_slots = batch_sentences.shape[1]
        nonzero_fraction = avg_nonzero / total_slots

        # Log the statistics
        self.log(f"{prefix}/avg_nonzero_genes", avg_nonzero)
        self.log(f"{prefix}/nonzero_fraction", nonzero_fraction)

    def shared_step(self, batch, batch_idx):
        logging.info(f"Step {self.global_step} - Batch {batch_idx}")
        X, Y, batch_weights, embs, dataset_embs = self._compute_embedding_for_batch(batch)

        # Track non-zero elements in the sentence
        batch_sentences = batch[7].to(self.device).bool()
        prefix = "trainer" if self.training else "validation"
        self._log_nonzero_elements_stats(batch_sentences, prefix)

        z = embs.unsqueeze(1).repeat(1, X.shape[1], 1)  # CLS token

        if self.z_dim_rd == 1:
            mu = (
                torch.nan_to_num(
                    torch.nanmean(
                        Y.float().masked_fill(Y == 0, float("nan")),  # ignore zeros
                        dim=1,
                    ),
                    nan=0.0,  # if all were 0→NaN, make it 0
                )
                if self.cfg.model.rda
                else None
            )
            reshaped_counts = mu.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, X.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((X, z, reshaped_counts), dim=2)
        else:
            assert self.z_dim_rd == 0
            # Original behavior if total_counts is None
            combine = torch.cat((X, z), dim=2)

        if self.dataset_token is not None and dataset_embs is not None:
            ds_emb = self.dataset_embedder(dataset_embs)
            ds_emb = ds_emb.unsqueeze(1).repeat(1, X.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        # concatenate the counts
        decs = self.binary_decoder(combine)

        if self.cfg.loss.name == "cross_entropy":
            criterion = BCEWithLogitsLoss()
            target = Y
        elif self.cfg.loss.name == "mse":
            criterion = nn.MSELoss()
            target = Y
        elif self.cfg.loss.name == "wasserstein":
            criterion = WassersteinLoss()
            target = Y
        elif self.cfg.loss.name == "kl_divergence":
            criterion = KLDivergenceLoss(apply_normalization=self.cfg.loss.normalization)
            target = batch_weights
        elif self.cfg.loss.name == "mmd":
            kernel = self.cfg.loss.get("kernel", "energy")
            criterion = MMDLoss(kernel=kernel, downsample=self.cfg.model.num_downsample if self.training else 1)
            target = Y
        elif self.cfg.loss.name == "tabular":
            criterion = TabularLoss(
                shared=self.cfg.dataset.S, downsample=self.cfg.model.num_downsample if self.training else 1
            )
            target = Y
        else:
            raise ValueError(f"Loss {self.cfg.loss.name} not supported")

        loss = criterion(decs.squeeze(), target)
        if dataset_embs is not None:
            # use the dataset loss
            dataset_pred = self.dataset_encoder(dataset_embs)  # B x # datasets
            dataset_labels = batch[8].to(self.device).long()

            # self.dataset_loss is a nn.CrossEntropyLoss
            dataset_loss = self.dataset_loss(dataset_pred, dataset_labels)
            if self.training:
                self.log("trainer/dataset_loss", dataset_loss)
                loss = loss + dataset_loss
            else:
                self.log("validation/dataset_loss", dataset_loss)

        sch = self.lr_schedulers()

        for scheduler in sch._schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
        sch._last_lr = [group["lr"] for group in sch._schedulers[-1].optimizer.param_groups]
        return loss

    @torch.compile(disable=True)
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("trainer/train_loss", loss)
        return loss

    @torch.compile(disable=True)
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("validation/val_loss", loss)
        return loss

    def configure_optimizers(self):
        max_lr = self.max_lr
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr, weight_decay=self.cfg.optimizer.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches * 2  # not sure why need to do this

        lr_schedulers = [
            LinearLR(
                optimizer,
                start_factor=self.cfg.optimizer.start,
                end_factor=self.cfg.optimizer.end,
                total_iters=int(0.03 * total_steps),
            )
        ]
        lr_schedulers.append(CosineAnnealingLR(optimizer, eta_min=max_lr * 0.3, T_max=total_steps))
        scheduler = ChainedScheduler(lr_schedulers)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss", "interval": "step", "frequency": 1},
        }

    def update_config(self, new_cfg):
        """Update the model's config after loading from checkpoint."""
        self.cfg = new_cfg
