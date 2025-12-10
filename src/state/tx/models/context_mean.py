import logging
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn

from .base import PerturbationModel

logger = logging.getLogger(__name__)


class ContextMeanPerturbationModel(PerturbationModel):
    """
    Baseline model that predicts the perturbed expression for a cell by returning
    the cell-type-specific average expression computed from perturbed cells in the training data.

    Implementation details:
      - During training (in on_fit_start), we iterate over the training dataloader,
        and for each cell type, we accumulate sums and counts for cells with
        pert_name != self.control_pert.
      - For each cell type, we compute:
            celltype_mean = sum(expression) / count
      - At inference, for each sample in the batch, we look up its cell type and return
        the corresponding average perturbed expression.
      - If the cell type of a sample at inference was not observed during training,
        an assertion error is raised.
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
            output_dim: Dimension of the output (typically number of genes).
            pert_dim: Dimension of perturbation embeddings (not used here).
            n_decoder_layers: (Unused) provided for config compatibility.
            dropout: (Unused) provided for config compatibility.
            lr: Learning rate for the optimizer (dummy param only).
            loss_fn: Loss function for training (default MSELoss).
            embed_key: Optional embedding key (unused).
            output_space: 'gene' or 'latent'. Determines which key from the batch to use.
            decoder: Optional separate decoder (unused).
            gene_names: Optional gene names (for logging or reference).
            kwargs: Catch-all for any extra arguments.
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

        # Dictionary to hold the average perturbed expression for each cell type.
        self.celltype_pert_means: Dict[str, torch.Tensor] = {}

        # Dummy parameter so that Lightning sees a parameter to optimize.
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

    def on_fit_start(self):
        """Called by Lightning before training starts.

        Computes, for each cell type, the mean perturbed expression from the training data.
        Only cells with a perturbation (pert_name) different from self.control_pert are used.
        """
        super().on_fit_start()

        train_loader = self.trainer.datamodule.train_dataloader()
        if train_loader is None:
            logger.warning("No train dataloader found. Cannot compute cell type means.")
            return

        # Initialize dictionary to accumulate sum and count for each cell type.
        celltype_sums = defaultdict(lambda: {"sum": torch.zeros(self.output_dim), "count": 0, "control_sum": torch.zeros(self.output_dim), "control_count": 0})

        with torch.no_grad():
            for batch in train_loader:
                # Select the proper expression space
                if (self.embed_key and self.embed_key != "X_hvg" and self.output_space == "gene") or (
                    self.embed_key and self.output_space == "all"
                ):
                    X_vals = batch["pert_cell_counts"]
                else:
                    X_vals = batch["pert_cell_emb"]

                # Ensure the expression values are in float and on CPU
                X_cpu = X_vals.float().cpu()
                pert_names = batch["pert_name"]
                cell_types = batch["cell_type"]

                # Iterate over batch samples and accumulate perturbed and control cells
                for i in range(len(X_cpu)):
                    p_name = str(pert_names[i])
                    ct_name = str(cell_types[i])
                    if p_name == self.control_pert:
                        # Accumulate control cells
                        celltype_sums[ct_name]["control_sum"] += X_cpu[i]
                        celltype_sums[ct_name]["control_count"] += 1
                    else:
                        # Accumulate perturbed cells
                        celltype_sums[ct_name]["sum"] += X_cpu[i]
                        celltype_sums[ct_name]["count"] += 1

        # Compute the mean expression per cell type from the accumulated sums.
        for ct_name, stats in celltype_sums.items():
            if stats["count"] == 0:
                if stats["control_count"] > 0:
                    # Use control cell average as fallback for cell types with no perturbations
                    self.celltype_pert_means[ct_name] = stats["control_sum"] / stats["control_count"]
                    logger.info(f"ContextMean: Using control cell average for cell type '{ct_name}' (no perturbations found, {stats['control_count']} control cells used).")
                else:
                    logger.warning(f"No perturbed or control cells found for cell type {ct_name}.")
                    continue
            else:
                self.celltype_pert_means[ct_name] = stats["sum"] / stats["count"]

        logger.info(
            f"ContextMean: computed average perturbed expression for {len(self.celltype_pert_means)} cell types."
        )

    def configure_optimizers(self):
        """
        Returns an optimizer for our dummy parameter if available.
        """
        if len(list(self.parameters())) > 0:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        return None

    def forward(self, batch: dict) -> torch.Tensor:
        """
        For each sample in the batch:
          - If the cell is a control (pert_name == self.control_pert), return the control cell's expression.
          - Otherwise, look up and return the stored average perturbed expression for its cell type.

        Args:
            batch (dict): Dictionary containing at least the keys "cell_type", "pert_name", and the expression key
                          ("pert_cell_counts" if output_space == "gene", else "pert_cell_emb").

        Returns:
            torch.Tensor: Predicted expression tensor of shape (B, output_dim).
        """
        B = len(batch["cell_type"])
        device = self.dummy_param.device
        # Determine which key to use for the expression values.
        output_key = (
            "pert_cell_counts"
            if self.embed_key
            and ((self.output_space == "gene" and self.embed_key != "X_hvg") or self.output_space == "all")
            else "pert_cell_emb"
        )
        pred_out = torch.zeros((B, self.output_dim), device=device)

        for i in range(B):
            p_name = str(batch["pert_name"][i])
            if p_name == self.control_pert:
                # For control cells, simply return the control cell's expression.
                pred_out[i] = batch[output_key][i].to(device)
            else:
                ct_name = str(batch["cell_type"][i])
                # Assert that the cell type was seen during training.
                assert ct_name in self.celltype_pert_means, f"Cell type '{ct_name}' was not seen during training."
                pred_out[i] = self.celltype_pert_means[ct_name].to(device)

        return pred_out

    def training_step(self, batch, batch_idx):
        """
        Computes the training loss (MSE) for the entire batch.
        For control cells (where pert_name == self.control_pert), the prediction is simply the control cell's expression.
        For perturbed cells, the prediction is the cell type's average perturbed expression computed during on_fit_start.

        Args:
            batch (dict): Batch dictionary containing keys such as "cell_type", "pert_name", and the expression key.
            batch_idx (int): Batch index (unused here).

        Returns:
            torch.Tensor: The computed loss.
        """
        pred = self(batch)
        output_key = (
            "pert_cell_counts"
            if self.embed_key
            and ((self.output_space == "gene" and self.embed_key != "X_hvg") or self.output_space == "all")
            else "pert_cell_emb"
        )
        target = batch[output_key]
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return None

    def on_save_checkpoint(self, checkpoint):
        """
        Save the computed cell type means to the checkpoint.
        """
        super().on_save_checkpoint(checkpoint)
        # Convert each tensor to a CPU numpy array for serialization.
        checkpoint["celltype_pert_means"] = {ct: mean.cpu().numpy() for ct, mean in self.celltype_pert_means.items()}
        logger.info("ContextMean: Saved celltype_pert_means to checkpoint.")

    def on_load_checkpoint(self, checkpoint):
        """
        Load the cell type means from the checkpoint.
        """
        super().on_load_checkpoint(checkpoint)
        if "celltype_pert_means" in checkpoint:
            loaded_means = {}
            for ct, mean_np in checkpoint["celltype_pert_means"].items():
                loaded_means[ct] = torch.tensor(mean_np, dtype=torch.float32)
            self.celltype_pert_means = loaded_means
            logger.info(f"ContextMean: Loaded means for {len(self.celltype_pert_means)} cell types from checkpoint.")
        else:
            logger.warning("ContextMean: No celltype_pert_means found in checkpoint. All means set to empty.")
            self.celltype_pert_means = {}

    def _build_networks(self):
        """
        No networks to build for this baseline model.
        """
        pass
