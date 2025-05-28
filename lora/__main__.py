import json
import os
from pathlib import Path
import shutil
import pickle
import re
from os.path import join, exists
from typing import List
import sys

import hydra
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.plugins.precision import MixedPrecision

from data.utils.modules import get_datamodule
from data.data_modules.tasks import parse_dataset_specs
from models import PertSetsPerturbationModel
from callbacks import GradNormCallback, BatchSpeedMonitorCallback
from lora.lora import get_lora_config, prepare_model_for_lora

import logging

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")

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

def get_checkpoint_callbacks(
    output_dir: str, name: str, val_freq: int, ckpt_every_n_steps: int
) -> List[ModelCheckpoint]:
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

    # Also save periodic checkpoints
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        save_last=False,
        every_n_train_steps=ckpt_every_n_steps,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

    return callbacks

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function for LoRA fine-tuning."""
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

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)

    # Set random seeds
    if "train_seed" in cfg["training"]:
        pl.seed_everything(cfg["training"]["train_seed"])

    # Parse dataset specs if needed
    if cfg["data"]["name"] == "MultiDatasetPerturbationDataModule":
        if isinstance(cfg["data"]["kwargs"]["train_task"], list):
            cfg["data"]["kwargs"]["train_specs"] = parse_dataset_specs(cfg["data"]["kwargs"]["train_task"])
        else:
            cfg["data"]["kwargs"]["train_specs"] = parse_dataset_specs([cfg["data"]["kwargs"]["train_task"]])

        if isinstance(cfg["data"]["kwargs"]["test_task"], list):
            cfg["data"]["kwargs"]["test_specs"] = parse_dataset_specs(cfg["data"]["kwargs"]["test_task"])
        else:
            cfg["data"]["kwargs"]["test_specs"] = parse_dataset_specs([cfg["data"]["kwargs"]["test_task"]])

    # Get data module
    data_module = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=cfg["model"]["kwargs"].get("cell_set_len", 512),
    )

    # Create base model
    model = PertSetsPerturbationModel(
        input_dim=cfg["model"]["kwargs"]["input_dim"],
        hidden_dim=cfg["model"]["kwargs"]["hidden_dim"],
        output_dim=cfg["model"]["kwargs"]["output_dim"],
        pert_dim=cfg["model"]["kwargs"]["pert_dim"],
        batch_dim=cfg["model"]["kwargs"].get("batch_dim"),
        transformer_backbone_key=cfg["model"]["kwargs"]["transformer_backbone_key"],
        transformer_backbone_kwargs=cfg["model"]["kwargs"]["transformer_backbone_kwargs"],
        **cfg["model"]["kwargs"]
    )

    # Load checkpoint if provided
    if "checkpoint_path" in cfg:
        logger.info(f"Loading checkpoint from {cfg['checkpoint_path']}")
        checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu")
        
        # Handle state dict loading
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter out mismatched size parameters
        model_state = model.state_dict()
        filtered_state = {}
        for name, param in state_dict.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    filtered_state[name] = param
                else:
                    logger.warning(f"Skipping parameter {name} due to shape mismatch: checkpoint={param.shape}, model={model_state[name].shape}")
            else:
                logger.warning(f"Skipping parameter {name} as it doesn't exist in the current model")

        # Load the filtered state dict
        model.load_state_dict(filtered_state, strict=False)
        logger.info("Successfully loaded checkpoint")
    else:
        raise ValueError("No checkpoint_path provided in config. LoRA fine-tuning requires a pre-trained checkpoint.")

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

    # Set up callbacks
    callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )

    # Add BatchSpeedMonitorCallback
    batch_speed_monitor = BatchSpeedMonitorCallback()
    callbacks.append(batch_speed_monitor)

    # Build trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=cfg["training"]["max_steps"],
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
    )

    # Check for existing checkpoint to resume from
    checkpoint_path = join(callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    # if a checkpoint does not exist, start with the provided checkpoint
    # this is mainly used for pretrain -> finetune workflows
    manual_init = cfg["model"]["kwargs"].get("init_from", None)
    if checkpoint_path is None and manual_init is not None:
        checkpoint_path = manual_init
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = model.state_dict()
        checkpoint_state = checkpoint["state_dict"]

        pert_encoder_weight_key = "pert_encoder.0.weight"
        if pert_encoder_weight_key in checkpoint_state:
            checkpoint_pert_dim = checkpoint_state[pert_encoder_weight_key].shape[1]
            if checkpoint_pert_dim != model.pert_dim:
                print(
                    f"pert_encoder input dimension mismatch: model.pert_dim = {model.pert_dim} but checkpoint expects {checkpoint_pert_dim}. Overriding model's pert_dim and rebuilding pert_encoder."
                )
                # Rebuild the pert_encoder with the new pert input dimension
                from models.utils import build_mlp

                model.pert_encoder = build_mlp(
                    in_dim=model.pert_dim,
                    out_dim=model.hidden_dim,
                    hidden_dim=model.hidden_dim,
                    n_layers=model.n_encoder_layers,
                    dropout=model.dropout,
                    activation=model.activation_class,
                )

        # Filter out mismatched size parameters
        filtered_state = {}
        for name, param in checkpoint_state.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    filtered_state[name] = param
                else:
                    print(
                        f"Skipping parameter {name} due to shape mismatch: checkpoint={param.shape}, model={model_state[name].shape}"
                    )
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model")

        # Load the filtered state dict
        model.load_state_dict(filtered_state, strict=False)

    # Apply LoRA to the model ONCE, after all checkpoint loading is complete
    model = prepare_model_for_lora(
        model,
        model_type=cfg["model"]["kwargs"]["transformer_backbone_key"],
        lora_config=get_lora_config(
            model_type=cfg["model"]["kwargs"]["transformer_backbone_key"],
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["lora_alpha"],
            lora_dropout=cfg["lora"]["lora_dropout"],
        )
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)

if __name__ == "__main__":
    train()
