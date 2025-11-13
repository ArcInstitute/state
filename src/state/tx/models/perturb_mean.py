import logging
from collections import defaultdict
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import PerturbationModel

logger = logging.getLogger(__name__)


class PerturbMeanPerturbationModel(PerturbationModel):
    """
    A map-independent baseline that computes a cell-type-specific
    control mean and a cell-type-specific offset for each (cell type, perturbation).

    Implementation details:
      - We do a single pass over the training dataloader (in on_fit_start) to
        accumulate sums & counts:
        * celltype_ctrl_sum[c_name] and celltype_ctrl_count[c_name]
        * celltype_pert_sum[c_name][p_name] and celltype_pert_count[c_name][p_name]
      - Then we compute:
          ctrl_mean[c_name] = celltype_ctrl_sum[c_name]/celltype_ctrl_count[c_name]
          offset[(c_name, p_name)] = (pert_sum / pert_count) - ctrl_mean[c_name]
      - At inference, for each sample with cell type c and perturbation p,
        we predict: ctrl_mean[c] + offset[(c, p)].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        loss_fn=nn.MSELoss(),
        embed_key: str = None,
        output_space: str = "gene",
        gene_names=None,
        **kwargs,
    ):
        """
        Args:
            input_dim: Size of input features (genes or embedding).
            hidden_dim: Not used here, but required by base-class signature.
            output_dim: Dimension of the output (often #genes).
            pert_dim:  Dimension of perturbation embeddings (not used here).
            n_decoder_layers: (Unused) included for config compatibility.
            dropout: (Unused) included for config compatibility.
            lr: Learning rate if there's anything to optimize. We only keep a dummy param.
            loss_fn: The chosen PyTorch loss function for training (default MSE).
            embed_key: Possibly an embedding key from the datamodule (unused).
            output_space: 'gene' or 'latent' (we handle either; just read from batch["pert_cell_counts"] or batch["pert_cell_emb"]).
            decoder: Possibly a separate decoder if output_space='latent' for final evaluation.
            gene_names: Names of genes if needed for logging.
            kwargs: Catch-all for extra arguments from config.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            dropout=dropout,
            lr=lr,
            loss_fn=loss_fn,
            embed_key=embed_key,
            output_space=output_space,
            gene_names=gene_names,
            **kwargs,
        )

        # We'll store a mean control vector per cell type:
        self.ctrl_mean: Dict[str, torch.Tensor] = {}
        # We'll store an offset for each (cell_type, pert_name):
        self.offsets: Dict[Tuple[str, str], torch.Tensor] = {}

        # A dummy parameter so that Lightning sees something to "optimize"
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.pert_mean_offsets = {}

    def on_fit_start(self):
        """Called by Lightning before training."""
        super().on_fit_start()

        # 1. First compute control means per cell type
        celltype_ctrl_means = {}  # Dict[str, torch.Tensor]

        # 2. And perturbation means per cell type
        celltype_pert_means = defaultdict(dict)  # Dict[str, Dict[str, torch.Tensor]]

        train_loader = self.trainer.datamodule.train_dataloader()
        if train_loader is None:
            logger.warning("No train dataloader found. Cannot compute offsets.")
            return

        # First pass: gather sums per cell type
        celltype_sums = defaultdict(lambda: defaultdict(lambda: {"sum": torch.zeros(self.output_dim), "count": 0}))

        with torch.no_grad():
            for batch in train_loader:
                if (
                    self.embed_key
                    and self.embed_key != "X_hvg"
                    and self.output_space == "gene"
                    or self.embed_key
                    and self.output_space == "all"
                ):
                    X_vals = batch["pert_cell_counts"]
                else:
                    X_vals = batch["pert_cell_emb"]

                X_cpu = X_vals.float().cpu()
                pert_names = batch["pert_name"]
                cell_types = batch["cell_type"]

                for i in range(len(X_cpu)):
                    p_name = str(pert_names[i])
                    ct_name = str(cell_types[i])
                    x_val = X_cpu[i]

                    celltype_sums[ct_name][p_name]["sum"] += x_val
                    celltype_sums[ct_name][p_name]["count"] += 1

            # Now compute means per cell type
            all_ctrl_means = []  # For computing global basal
            for ct_name, pert_dict in celltype_sums.items():
                # Get control mean for this cell type
                ctrl_stats = pert_dict.get(self.control_pert)
                if ctrl_stats is None or ctrl_stats["count"] == 0:
                    logger.warning(f"No control cells found for cell type {ct_name}")
                    continue

                ct_ctrl_mean = ctrl_stats["sum"] / ctrl_stats["count"]
                celltype_ctrl_means[ct_name] = ct_ctrl_mean
                all_ctrl_means.append(ct_ctrl_mean)

                # Compute perturbation means and deltas for this cell type
                for p_name, stats in pert_dict.items():
                    if p_name == self.control_pert:
                        continue

                    if stats["count"] == 0:
                        continue

                    ct_pert_mean = stats["sum"] / stats["count"]
                    ct_delta = ct_pert_mean - ct_ctrl_mean
                    celltype_pert_means[ct_name][p_name] = ct_delta

            # Now average deltas across cell types
            self.pert_mean_offsets = {}
            for p_name in set().union(*[d.keys() for d in celltype_pert_means.values()]):
                # Gather deltas for this perturbation across cell types
                pert_deltas = []
                for ct_name, pert_dict in celltype_pert_means.items():
                    if p_name in pert_dict:
                        pert_deltas.append(pert_dict[p_name])

                if not pert_deltas:
                    continue

                # Average the deltas
                self.pert_mean_offsets[p_name] = torch.stack(pert_deltas).mean(0)

            # Add zero offset for control
            self.pert_mean_offsets[self.control_pert] = torch.zeros(self.output_dim)

            # Compute global basal as mean of cell-type means
            if not all_ctrl_means:
                logger.warning("No control cells found in any cell type. Using zero vector as basal.")
                self.global_basal = torch.zeros(self.output_dim)
            else:
                self.global_basal = torch.stack(all_ctrl_means).mean(0)

            logger.info(f"PerturbMean: computed offsets for {len(self.pert_mean_offsets)} perturbations")

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Now we ignore the cell type entirely.

        For each sample in the batch, we do:
            prediction = global_basal + offset[p_name].
        """
        B = len(batch["pert_name"])
        device = self.dummy_param.device
        pred_out = torch.zeros((B, self.output_dim), device=device)

        for i in range(B):
            p_name = str(batch["pert_name"][i])
            offset_vec = self.pert_mean_offsets.get(p_name, None)
            if offset_vec is None:
                offset_vec = torch.zeros(self.output_dim, device=device)

            pred_out[i] = batch["ctrl_cell_emb"][i] + offset_vec.to(device)

        return pred_out

    def configure_optimizers(self):
        """
        Return an optimizer for our dummy_param if desired, or None if we prefer no updates.
        """
        if len(list(self.parameters())) > 0:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        return None

    def training_step(self, batch, batch_idx):
        """
        We'll compute MSE vs. the ground truth. (Though no real learnable offsets.)
        """
        pred = self(batch)
        if (self.embed_key and self.embed_key != "X_hvg" and self.output_space == "gene") or (
            self.embed_key and self.output_space == "all"
        ):
            target = batch["pert_cell_counts"]
        else:
            target = batch["pert_cell_emb"]
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return None

    def on_save_checkpoint(self, checkpoint):
        """
        Save the global_basal vector and perturbation offsets to the checkpoint.

        Args:
            checkpoint (dict): Checkpoint dictionary to be saved.
        """
        super().on_save_checkpoint(checkpoint)

        # Convert tensors to CPU and NumPy arrays for serialization
        checkpoint["global_basal"] = self.global_basal.cpu().numpy()
        checkpoint["pert_mean_offsets"] = {
            p_name: offset.cpu().numpy() for p_name, offset in self.pert_mean_offsets.items()
        }

        logger.info("PerturbMean: Saved global_basal and pert_mean_offsets to checkpoint.")

    def on_load_checkpoint(self, checkpoint):
        """
        Load the global_basal vector and perturbation offsets from the checkpoint.

        Args:
            checkpoint (dict): Checkpoint dictionary from which to load.
        """
        super().on_load_checkpoint(checkpoint)

        # Load global_basal
        if "global_basal" in checkpoint:
            self.global_basal = torch.tensor(checkpoint["global_basal"], dtype=torch.float32)
            logger.info("PerturbMean: Loaded global_basal from checkpoint.")
        else:
            logger.warning("PerturbMean: No global_basal found in checkpoint. Using zero vector.")
            self.global_basal = torch.zeros(self.output_dim)

        # Load perturbation offsets
        if "pert_mean_offsets" in checkpoint:
            loaded_offsets = {}
            for p_name, offset_np in checkpoint["pert_mean_offsets"].items():
                loaded_offsets[p_name] = torch.tensor(offset_np, dtype=torch.float32)
            self.pert_mean_offsets = loaded_offsets
            logger.info(f"PerturbMean: Loaded offsets for {len(self.pert_mean_offsets)} perturbations from checkpoint.")
        else:
            logger.warning("PerturbMean: No pert_mean_offsets found in checkpoint. All offsets set to zero.")
            self.pert_mean_offsets = {}

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Not really used here, but required by abstract base. We do no param-based encoding."""
        return pert

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """No param-based encoding of basal. Just identity."""
        return expr

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Not used in the normal forward pass, because we look up offset by 'pert_name' strings.
        But we must define it to meet the abstract contract.
        """
        return basal

    def _build_networks(self):
        """
        We don't need any networks to be built.
        """
        pass
