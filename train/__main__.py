import os
import shutil
import pickle
import re
from os.path import join, exists
from typing import List
import sys
sys.path.append('./vci_pretrain')

import hydra
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from omegaconf import DictConfig, OmegaConf

from data.utils.modules import get_datamodule
from data.data_modules.tasks import parse_dataset_specs  # TODO-Abhi: Should this move?
from models.decoders import UCELogProbDecoder
from models import (
    SimpleSumPerturbationModel,
    GlobalSimpleSumPerturbationModel,
    CellTypeMeanModel,
    EmbedSumPerturbationModel,
    PertSetsPerturbationModel,
    OldNeuralOTPerturbationModel,
    DecoderOnlyPerturbationModel,
    PseudobulkPerturbationModel
)
from callbacks import GradNormCallback

import logging

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")

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
        return EmbedSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "old_neuralot":
        return OldNeuralOTPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "neuralot" or model_type.lower() == "pertsets":
        return PertSetsPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "simplesum":
        return SimpleSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "globalsimplesum":
        return GlobalSimpleSumPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "celltypemean":
        return CellTypeMeanModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "decoder_only":
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


def get_checkpoint_callbacks(output_dir: str, name: str, val_freq: int, ckpt_every_n_steps: int) -> List[ModelCheckpoint]:
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
        every_n_train_steps=ckpt_every_n_steps,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

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

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        f.write(cfg_yaml)
    logger.info(f"Config saved to file.")

    # Set random seeds
    if "train_seed" in cfg["training"]:
        pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

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

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
        
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
            logger.info("Saving data module...")
            with open(join(run_output_dir, "data_module.pkl"), "wb") as f:
                # TODO-Abhi: only save necessary data
                pickle.dump(data_module, f)
            logger.info(f"Data module saved.")

    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        data_module.get_var_dims(),
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
    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    # callbacks = ckpt_callbacks + [GradNormCallback()]
    callbacks = ckpt_callbacks

    logger.info('Loggers and callbacks set up.')

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        max_steps=cfg["training"]["max_steps"],  # for normal models
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        # profiler='simple',
    )

    # If it's SimpleSum, override to do exactly 1 epoch, ignoring `max_steps`.
    if cfg["model"]["name"].lower() == "celltypemean" or cfg["model"]["name"].lower() == "globalsimplesum":
        trainer_kwargs["max_epochs"] = 1  # do exactly one epoch
        # delete max_steps to avoid conflicts
        del trainer_kwargs["max_steps"]

    # Build trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    logger.info('Starting trainer fit.')

    # if we are training with freeze pert param, manually load the checkpoint
    # if no last.ckpt, start training using the seed file
    if checkpoint_path is None and cfg["model"]["kwargs"].get("freeze_pert", False):
        checkpoint_path = join(ckpt_callbacks[0].dirpath, "seed.ckpt")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state = model.state_dict()
        checkpoint_state = checkpoint['state_dict']

        pert_encoder_weight_key = "pert_encoder.0.weight"
        if pert_encoder_weight_key in checkpoint_state:
            checkpoint_pert_dim = checkpoint_state[pert_encoder_weight_key].shape[1]
            if checkpoint_pert_dim != model.pert_dim:
                print(f"pert_encoder input dimension mismatch: model.pert_dim = {model.pert_dim} but checkpoint expects {checkpoint_pert_dim}. Overriding model's pert_dim and rebuilding pert_encoder.")
                # Update the model's pert_dim
                model.pert_dim = checkpoint_pert_dim
                # Rebuild the pert_encoder with the new input dimension.
                # (Assumes that build_mlp is the utility function used by your model)
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
                    print(f"Skipping parameter {name} due to shape mismatch: checkpoint={param.shape}, model={model_state[name].shape}")
            else:
                print(f"Skipping parameter {name} as it doesn't exist in the current model")
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state, strict=False)

        # Train - for clarity we pass None
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=None,
        )
    else:
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
