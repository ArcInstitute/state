import json
import os
from pathlib import Path
import shutil
import pickle
import re
from os.path import join, exists
from typing import List
import sys

sys.path.append("./vci_pretrain")

import hydra
import torch

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.plugins.precision import MixedPrecision

from vc_load.utils.modules import get_datamodule
from vc_load.data_modules.tasks import parse_dataset_specs

# from models.decoders import UCELogProbDecoder # commented out since it's not used yet
from models import (
    SimpleSumPerturbationModel,
    GlobalSimpleSumPerturbationModel,
    CellTypeMeanModel,
    EmbedSumPerturbationModel,
    PertSetsPerturbationModel,
    OldNeuralOTPerturbationModel,
    DecoderOnlyPerturbationModel,
    PseudobulkPerturbationModel,
    scGPTForPerturbation,
    CPAPerturbationModel,
    SCVIPerturbationModel,
)
from callbacks import GradNormCallback, BatchSpeedMonitorCallback

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
    elif model_type.lower() == "cpa":
        return CPAPerturbationModel(
            input_dim=var_dims["input_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            gene_dim=gene_dim,
            **module_config,
        )
    elif model_type.lower() == "scvi":
        return SCVIPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )
    elif model_type.lower() == "scgpt-chemical" or model_type.lower() == "scgpt-genetic":
        pretrained_path = module_config["pretrained_path"]
        assert pretrained_path is not None, "pretrained_path must be provided for scGPT"

        model_dir = Path(pretrained_path)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"

        model = scGPTForPerturbation(
            ntoken=module_config["ntoken"],
            n_drug_tokens=module_config["n_perts"],  # only used for chemical perturbations
            d_model=module_config["d_model"],
            nhead=module_config["nhead"],
            d_hid=module_config["d_hid"],
            nlayers=module_config["nlayers"],
            nlayers_cls=module_config["n_layers_cls"],
            n_cls=1,
            dropout=module_config["dropout"],
            pad_token_id=module_config["pad_token_id"],
            pad_value=module_config["pad_value"],
            pert_pad_id=module_config["pert_pad_id"],
            do_mvc=module_config["do_MVC"],
            cell_emb_style=module_config["cell_emb_style"],
            mvc_decoder_style=module_config["mvc_decoder_style"],
            use_fast_transformer=module_config["use_fast_transformer"],
            lr=module_config["lr"],
            step_size_lr=module_config["step_size_lr"],
            include_zero_gene=module_config["include_zero_gene"],
            embed_key=module_config["embed_key"],
            perturbation_type=module_config["perturbation_type"],
        )

        load_param_prefixes = module_config["load_param_prefixes"]

        if load_param_prefixes is not None:
            model_dict = model.model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if any([k.startswith(prefix) for prefix in module_config["load_param_prefixes"]])
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")

            model_dict.update(pretrained_dict)
            model.model.load_state_dict(model_dict)
        else:
            try:
                model.model.load_state_dict(torch.load(model_file))
                print(f"Loading all model params from {model_file}")
            except:
                # only load params that are in the model and match the size
                model_dict = model.model.state_dict()
                pretrained_dict = torch.load(model_file)
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape
                }
                for k, v in pretrained_dict.items():
                    print(f"Loading params {k} with shape {v.shape}")

                model_dict.update(pretrained_dict)
                model.model.load_state_dict(model_dict)

        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_latest_step_checkpoint(directory):
    # Get all checkpoint files
    files = os.listdir(directory)

    # Extract step numbers using regex, excluding files with 'val_loss'
    step_numbers = []
    for f in files:
        if f.startswith("step=") and "val_loss" not in f:
            # Extract the number between 'step=' and '.ckpt'
            match = re.search(r"step=(\d+)(?:-v\d+)?\.ckpt", f)
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

    # Set random seeds
    if "train_seed" in cfg["training"]:
        pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # Use the perturbation data module for training perturbation models
    # that involve mapping strageties (e.g., connecting perturbed cells to control cells.)
    if cfg["data"]["name"] == "PerturbationDataModule":
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
        if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
            if "cell_sentence_len" in cfg["model"]["kwargs"] and cfg["model"]["kwargs"]["cell_sentence_len"] > 1:
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]

    if cfg["model"]["name"].lower().startswith("scgpt"):  # scGPT uses log-normalized expression
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
        cfg["data"]["kwargs"]["hvg_names_uns_key"] = (
            "hvg_names" if cfg["data"]["kwargs"]["train_task"] != "replogle" else None
        )  # TODO: better to not hardcode this

        cfg["data"]["kwargs"]["dataset_cls"] = "scGPTPerturbationDataset"

        model_dir = Path(cfg["model"]["kwargs"]["pretrained_path"])

        vocab_file = model_dir / "vocab.json"

        vocab = json.load(open(vocab_file, "r"))
        cfg["model"]["kwargs"]["pad_token_id"] = vocab["<pad>"]
        for s in cfg["model"]["kwargs"]["special_tokens"]:
            if s not in vocab:
                vocab[s] = len(vocab)

        cfg["data"]["kwargs"]["vocab"] = vocab
        cfg["data"]["kwargs"]["perturbation_type"] = cfg["model"]["kwargs"]["perturbation_type"]
        cfg["model"]["kwargs"]["ntoken"] = len(vocab)
        cfg["model"]["kwargs"]["d_model"] = cfg["model"]["kwargs"]["embsize"]

        logger.info(f"Added vocab and hvg_names_uns_key to data kwargs for scGPT")

    elif cfg["model"]["name"].lower() == "cpa" and cfg["model"]["kwargs"]["recon_loss"] == "gauss":
        cfg["data"]["kwargs"]["transform"] = "log-normalize"
    elif cfg["model"]["name"].lower() == "scvi":
        cfg["data"]["kwargs"]["transform"] = None

    data_module = get_datamodule(
        name=cfg["data"]["name"],
        kwargs=cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )

    # Special handling for multi-dataset case - TODO-now: revisit this.
    if cfg["data"]["name"] == "PerturbationDataModule":
        # if the data module already exists, just read it in
        data_module.setup()

        # Save data module for reproducibility
        logger.info("Saving data module...")
        with open(join(run_output_dir, "data_module.pkl"), "wb") as f:
            # TODO-Abhi: only save necessary data
            pickle.dump(data_module, f)
        logger.info(f"Data module saved.")

    if cfg["model"]["name"].lower() in ["cpa", "scvi"] or cfg["model"]["name"].lower().startswith("scgpt"):
        cfg["model"]["kwargs"]["n_cell_types"] = len(data_module.celltype_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(data_module.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(data_module.batch_onehot_map)

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
    # Add BatchSpeedMonitorCallback to log batches per second to wandb
    batch_speed_monitor = BatchSpeedMonitorCallback()
    callbacks = ckpt_callbacks + [batch_speed_monitor]

    logger.info("Loggers and callbacks set up.")

    if cfg["model"]["name"].lower().startswith("scgpt"):
        plugins = [
            MixedPrecision(
                precision="bf16-mixed",
                device="cuda",
            )
        ]
    else:
        plugins = []

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
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"] if cfg["model"]["name"].lower() != "cpa" else None,
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

    logger.info("Starting trainer fit.")

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
