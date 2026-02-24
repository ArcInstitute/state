import time
import logging
from contextlib import contextmanager
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger as BaseCSVLogger
import csv
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from os.path import join


class RobustCSVLogger(BaseCSVLogger):
    """
    A CSV logger that handles dynamic metrics by allowing new columns to be added during training.
    This fixes the issue where PyTorch Lightning's default CSV logger fails when new metrics
    are added after the CSV file is created.
    """

    def log_metrics(self, metrics, step):
        """Override to handle dynamic metrics gracefully"""
        try:
            super().log_metrics(metrics, step)
        except ValueError as e:
            if "dict contains fields not in fieldnames" in str(e):
                # Recreate the CSV file with the new fieldnames
                self._recreate_csv_with_new_fields(metrics)
                # Try logging again
                super().log_metrics(metrics, step)
            else:
                raise e

    def _recreate_csv_with_new_fields(self, new_metrics):
        """Recreate the CSV file with additional fields to accommodate new metrics"""
        if not hasattr(self.experiment, "metrics_file_path"):
            return

        # Read existing data
        existing_data = []
        csv_file = self.experiment.metrics_file_path

        if os.path.exists(csv_file):
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        # Get all unique fieldnames from existing data and new metrics
        all_fieldnames = set()
        for row in existing_data:
            all_fieldnames.update(row.keys())
        all_fieldnames.update(new_metrics.keys())

        # Sort fieldnames for consistent ordering
        sorted_fieldnames = sorted(all_fieldnames)

        # Rewrite the CSV file with new fieldnames
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
            writer.writeheader()

            # Write existing data (missing fields will be empty)
            for row in existing_data:
                writer.writerow(row)

        # Update the experiment's fieldnames
        self.experiment.metrics_keys = sorted_fieldnames


@contextmanager
def time_it(timer_name: str):
    logging.debug(f"Starting timer {timer_name}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logging.debug(f"Elapsed time {timer_name}: {elapsed_time:.4f} seconds")


def get_loggers(
    output_dir: str,
    name: str,
    wandb_project: str,
    wandb_entity: str,
    local_wandb_dir: str,
    use_wandb: bool = False,
    use_csv: bool = True,  # Enable CSV by default with robust logger
    cfg: dict = None,
):
    """Set up logging to local CSV and optionally WandB."""
    loggers = []

    # Use robust CSV logger that handles dynamic metrics
    if use_csv:
        csv_logger = RobustCSVLogger(save_dir=output_dir, name=name, version=0)
        loggers.append(csv_logger)

    # Add WandB if requested
    if use_wandb:
        try:
            # Check if wandb is available
            import wandb

            wandb_logger = WandbLogger(
                name=name,
                project=wandb_project,
                entity=wandb_entity,
                dir=local_wandb_dir,
                tags=cfg["wandb"].get("tags", []) if cfg else [],
            )
            if cfg is not None:
                wandb_logger.experiment.config.update(cfg)
            loggers.append(wandb_logger)
        except ImportError:
            print("Warning: wandb is not installed. Skipping wandb logging.")
            print("To enable wandb logging, install it with: pip install wandb")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb logger: {e}")
            print("Continuing without wandb logging.")

    # Ensure at least one logger is present
    if not loggers:
        print("Warning: No loggers configured. Adding robust CSV logger as fallback.")
        csv_logger = RobustCSVLogger(save_dir=output_dir, name=name, version=0)
        loggers.append(csv_logger)

    return loggers


def get_checkpoint_callbacks(
    output_dir: str,
    name: str,
    val_freq: int,
    _ckpt_every_n_steps: int,
    monitor_metric: str = "val/expression_loss",
):
    """
    Create checkpoint callbacks based on validation frequency.

    Returns a list of callbacks.
    """
    checkpoint_dir = join(output_dir, name, "checkpoints")
    callbacks = []

    # Save only the best checkpoint (by monitor_metric) plus the latest checkpoint
    best_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        save_last=True,
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        every_n_train_steps=val_freq,
    )
    callbacks.append(best_ckpt)

    return callbacks


def get_lightning_module(model_type: str, data_config: dict, model_config: dict, training_config: dict, var_dims: dict):
    """Create model instance based on config."""
    # combine the model config and training config
    module_config = {**model_config, **training_config}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]
    module_config["batch_size"] = training_config["batch_size"]
    module_config["control_pert"] = data_config.get("control_pert", "non-targeting")

    if data_config["output_space"] == "gene":
        gene_dim = var_dims["hvg_dim"]
    else:
        gene_dim = var_dims["gene_dim"]

    if model_type.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        return EmbedSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "neuralot" or model_type.lower() == "pertsets" or model_type.lower() == "state":
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        return StateTransitionPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            basal_mapping_strategy=data_config["basal_mapping_strategy"],
            **module_config,
        )
    elif model_type.lower() == "globalsimplesum" or model_type.lower() == "perturb_mean":
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        return PerturbMeanPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "celltypemean" or model_type.lower() == "context_mean":
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        return ContextMeanPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        return DecoderOnlyPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "pseudobulk":
        from ...tx.models.pseudobulk import PseudobulkPerturbationModel

        return PseudobulkPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
