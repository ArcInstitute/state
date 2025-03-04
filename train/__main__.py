import os
import shutil
import pickle
import re
from os.path import join, exists
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

from data.utils.modules import get_datamodule
from data.data_modules.tasks import parse_dataset_specs  # TODO-Abhi: Should this move?
from models.decoders import UCELogProbDecoder
from models import (
    SimpleSumPerturbationModel,
    GlobalSimpleSumPerturbationModel,
    CellTypeMeanModel,
    EmbedSumPerturbationModel,
    NeuralOTPerturbationModel,
)
from callbacks import GradNormCallback, PerturbationMagnitudeCallback, TestMetricsCallback

import logging

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")

def get_lightning_module(model_type: str, data_config: dict, model_config: dict, training_config: dict, var_dims: dict, global_config: dict):
    """Create model instance based on config."""
    # combine the model config and training config
    module_config = {**model_config, **training_config}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]
    module_config["batch_size"] = training_config["batch_size"]
    module_config["control_pert"] = data_config.get("control_pert", "non-targeting")
    module_config["bf16"] = global_config.get("bf16", False)

    # if data_config["output_space"] == "gene" and data_config["embed_key"] == "X_uce":
    #     # the model outputs will be in gene space, so no decoder is needed
    #     module_config["decoder"] = None
    # else:
    #     # the model outputs will be in latent space, so a decoder is needed
    #     if module_config["embed_key"] == "X_uce":
    #         # UCE log prob decoder requires gene names
    #         module_config["decoder"] = UCELogProbDecoder()  # TODO-Abhi: try out new decoders here
    #     else:
    #         # Add more decoders here as needed
    #         module_config["decoder"] = None

    if model_type.lower() == "embedsum":
        return EmbedSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=var_dims["gene_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            **module_config,
        )
    elif model_type.lower() == "neuralot":
        return NeuralOTPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=var_dims["gene_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            **module_config,
        )
    elif model_type.lower() == "simplesum":
        return SimpleSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=var_dims["gene_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            **module_config,
        )
    elif model_type.lower() == "globalsimplesum":
        return GlobalSimpleSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=var_dims["gene_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            **module_config,
        )
    elif model_type.lower() == "celltypemean":
        return CellTypeMeanModel(
            input_dim=var_dims["input_dim"],
            gene_dim=var_dims["gene_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            **module_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_latest_step_checkpoint(directory):
    # Get all checkpoint files
    files = os.listdir(directory)
    
    # Extract step numbers using regex, excluding files with 'val_loss'
    step_numbers = []
    for f in files:
        if f.startswith('step=') and 'val_loss' not in f:
            # Extract the number between 'step=' and '.ckpt'
            match = re.search(r'step=(\d+)(?:-v\d+)?\.ckpt', f)
            if match:
                step_numbers.append(int(match.group(1)))
    
    if not step_numbers:
        raise ValueError("No checkpoint files found")
        
    # Get the maximum step number
    max_step = max(step_numbers)
    
    # Construct the checkpoint path
    checkpoint_path = join(directory, f"step={max_step}.ckpt")
    
    return checkpoint_path

def get_loggers(
    output_dir: str,
    name: str,
    wandb_project: str,
    wandb_entity: str,
    local_wandb_dir: str,
    use_wandb: bool = False,
    cfg: dict = None,
) -> List:
    """Set up logging to local CSV and optionally WandB."""
    # Always use CSV logger
    csv_logger = CSVLogger(save_dir=output_dir, name=name, version=0)
    loggers = [csv_logger]

    # Add WandB if requested
    if use_wandb:
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

    return loggers


def get_checkpoint_callbacks(output_dir: str, name: str, val_freq: int, test_freq: int) -> List[ModelCheckpoint]:
    """Create checkpoint callbacks based on validation frequency."""
    checkpoint_dir = join(output_dir, name, "checkpoints")
    callbacks = []

    # Save best checkpoint based on validation loss
    best_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step={step}-val_loss={val_loss:.4f}",
        save_last="link",  # Will create last.ckpt symlink to best checkpoint
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only keep the best checkpoint
        every_n_train_steps=val_freq,
    )
    callbacks.append(best_ckpt)

    # Also save periodic checkpoints (without affecting the "last" symlink)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        save_last=False,  # Don't create/update symlink
        every_n_train_steps=4000,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

    test_metrics_cb = TestMetricsCallback(test_freq=test_freq)
    callbacks.append(test_metrics_cb)

    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    # Convert config to YAML for logging
    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(cfg_yaml)

    # Setup output directory
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg["overwrite"]:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    # Set up wandb directory if needed
    if cfg["use_wandb"]:
        os.makedirs(cfg["wandb"]["local_wandb_dir"], exist_ok=True)

    # if this already exists for a run, then just read it in
    # if exists(join(run_output_dir, "config.yaml")):
    #     with open(join(run_output_dir, "config.yaml"), "r") as f:
    #         cfg = OmegaConf.load(f)
    #     logger.info(f"Config loaded from file.")
    # else:
    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)
    logger.info(f"Config saved to file.")

    # Set random seeds
    if "train_seed" in cfg["training"]:
        pl.seed_everything(cfg["training"]["train_seed"])

    # Use the multi dataset perturbation data module for training perturbation models
    # that involve mapping strageties (e.g., connecting perturbed cells to control cells.)
    if cfg["data"]["name"] == "MultiDatasetPerturbationDataModule":
        # Parse train specs
        if isinstance(cfg["data"]["kwargs"]["train_task"], list):
            cfg["data"]["kwargs"]["train_specs"] = parse_dataset_specs(cfg["data"]["kwargs"]["train_task"])
        else:
            cfg["data"]["kwargs"]["train_specs"] = parse_dataset_specs([cfg["data"]["kwargs"]["train_task"]])

        # Parse test specs
        if isinstance(cfg["data"]["kwargs"]["test_task"], list):
            cfg["data"]["kwargs"]["test_specs"] = parse_dataset_specs(cfg["data"]["kwargs"]["test_task"])
        else:
            cfg["data"]["kwargs"]["test_specs"] = parse_dataset_specs([cfg["data"]["kwargs"]["test_task"]])

    # Initialize data module
    try:
        sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
    except KeyError:
        sentence_len = 512
        
    data_module = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )

    # Special handling for multi-dataset case - TODO-now: revisit this.
    if cfg["data"]["name"] == "MultiDatasetPerturbationDataModule":
        # if the data module already exists, just read it in
        if exists(join(run_output_dir, "data_module.pkl")):
            with open(join(run_output_dir, "data_module.pkl"), "rb") as f:
                data_module = pickle.load(f)
            logger.info(f"Data module loaded from file.")
        else:
            data_module.setup(stage="fit")
            data_module.setup(stage="test")

            # Save data module for reproducibility
            with open(join(run_output_dir, "data_module.pkl"), "wb") as f:
                # TODO-Abhi: only save necessary data
                pickle.dump(data_module, f)
            logger.info(f"Data module loaded and saved.")


    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        data_module.get_var_dims(),
        cfg
    )

    # Set up logging
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=cfg["name"],
        wandb_project=cfg["wandb"]["project"],
        wandb_entity=cfg["wandb"]["entity"],
        local_wandb_dir=cfg["wandb"]["local_wandb_dir"],
        use_wandb=cfg["use_wandb"],
        cfg=cfg,
    )

    # If using wandb, store the run path in a text file for eval
    # that matches the old train_lightning.py logic
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
            with open(wandb_info_path, "w") as f:
                f.write(lg.experiment.path)
            break

    # Set up callbacks
    ckpt_callbacks = get_checkpoint_callbacks(cfg["output_dir"], cfg["name"], cfg["training"]["val_freq"], cfg["training"]["test_freq"])
    callbacks = ckpt_callbacks + [GradNormCallback(), PerturbationMagnitudeCallback()]

    logger.info('Loggers and callbacks set up.')

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator="gpu",
        devices=1,
        max_steps=cfg["training"]["max_steps"],  # for normal models
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
    )

    if cfg.get("bf16", False):
        trainer_kwargs["precision"] = "bf16"
    else:
        trainer_kwargs["precision"] = 32

    # If it's SimpleSum, override to do exactly 1 epoch, ignoring `max_steps`.
    if (cfg["model"]["name"].lower() == "celltypemean" or cfg["model"]["name"].lower() == "globalsimplesum") and cfg[
        "data"
    ]["kwargs"]["output_space"] == "latent":
        trainer_kwargs["max_epochs"] = 1  # do exactly one epoch
        # delete max_steps to avoid conflicts
        del trainer_kwargs["max_steps"]

    # Build trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    # checkpoint_path = get_latest_step_checkpoint(ckpt_callbacks[0].dirpath)
    if not exists(checkpoint_path):
        checkpoint_path = None
        # logging.info('!! No checkpoint found, killing myself !!')
        # return
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    logger.info('Starting trainer fit.')

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    train()
