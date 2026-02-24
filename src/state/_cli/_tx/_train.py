import argparse as ap

from omegaconf import DictConfig, OmegaConf


def add_arguments_train(parser: ap.ArgumentParser):
    # Allow remaining args to be passed through to Hydra
    parser.add_argument("hydra_overrides", nargs="*", help="Hydra configuration overrides (e.g., data.batch_size=32)")
    # Add custom help handler
    parser.add_argument("--help", "-h", action="store_true", help="Show configuration help with all parameters")


def run_tx_train(cfg: DictConfig):
    import logging
    import os
    import pickle
    import shutil
    from os.path import exists, join

    import lightning.pytorch as pl
    import torch
    from cell_load.data_modules import PerturbationDataModule
    from cell_load.utils.modules import get_datamodule
    from lightning.pytorch.loggers import WandbLogger

    from ...tx.callbacks import (
        BatchSpeedMonitorCallback,
        CumulativeFLOPSCallback,
        GradNormCallback,
        ModelFLOPSUtilizationCallback,
    )
    from ...tx.utils import get_checkpoint_callbacks, get_lightning_module, get_loggers

    logger = logging.getLogger(__name__)
    torch.set_float32_matmul_precision("medium")

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)

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
    pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        try:
            sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
        except:
            sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["max_position_embeddings"]

    output_space = cfg["data"]["kwargs"].get("output_space", "gene")
    assert output_space in {"embedding", "gene", "all"}, (
        f"data.kwargs.output_space must be one of 'embedding', 'gene', or 'all'; got {output_space!r}"
    )
    nb_loss_enabled = bool(cfg["model"]["kwargs"].get("nb_loss", False))
    if nb_loss_enabled and output_space == "embedding":
        raise ValueError(
            "model.kwargs.nb_loss=True is incompatible with data.kwargs.output_space='embedding'. "
            "Use output_space='gene' or output_space='all'."
        )
    if nb_loss_enabled and output_space not in {"gene", "all"}:
        raise ValueError(
            f"model.kwargs.nb_loss=True requires data.kwargs.output_space in {{'gene', 'all'}}; got {output_space!r}."
        )
    embed_key = cfg["data"]["kwargs"].get("embed_key", None)
    if nb_loss_enabled and embed_key not in {None, "X_hvg"}:
        if not bool(cfg["data"]["kwargs"].get("store_raw_basal", False)):
            logger.warning(
                "nb_loss=True with embed_key=%r requires control counts for library-size estimation; "
                "setting data.kwargs.store_raw_basal=True.",
                embed_key,
            )
            cfg["data"]["kwargs"]["store_raw_basal"] = True

    if output_space == "embedding":
        checkpoint_monitor_metric = "val/embedding_loss"
    else:
        checkpoint_monitor_metric = "val/expression_loss"

    data_module: PerturbationDataModule = get_datamodule(
        cfg["data"]["name"],
        cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )

    data_module.setup(stage="fit")
    if nb_loss_enabled:
        resolved_is_log1p = bool(getattr(data_module, "is_log1p", cfg["data"]["kwargs"].get("is_log1p", False)))
        expected_exp_counts = resolved_is_log1p
        current_exp_counts = bool(getattr(data_module, "exp_counts", False))
        if current_exp_counts != expected_exp_counts:
            logger.warning(
                "nb_loss=True requires exp_counts to follow is_log1p. "
                "Resolved is_log1p=%s, overriding exp_counts %s -> %s.",
                resolved_is_log1p,
                current_exp_counts,
                expected_exp_counts,
            )
            data_module.exp_counts = expected_exp_counts
        else:
            logger.info(
                "nb_loss=True with resolved is_log1p=%s and exp_counts=%s.",
                resolved_is_log1p,
                current_exp_counts,
            )
        cfg["data"]["kwargs"]["is_log1p"] = resolved_is_log1p
        cfg["data"]["kwargs"]["exp_counts"] = expected_exp_counts

    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        # TODO-Abhi: only save necessary data
        data_module.save_state(f)

    dl = data_module.train_dataloader()
    print("num_workers:", dl.num_workers)
    print("batch size:", dl.batch_size)

    var_dims = data_module.get_var_dims()  # {"gene_dim": …, "hvg_dim": …}
    if output_space == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)  # fallback if key missing
    else:
        gene_dim = var_dims.get("gene_dim", 2000)  # fallback if key missing
    latent_dim = var_dims["output_dim"]  # same as model.output_dim
    hidden_dims = cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512])

    if output_space in {"gene", "all"}:
        decoder_cfg = dict(
            latent_dim=latent_dim,
            gene_dim=gene_dim,
            hidden_dims=hidden_dims,
            dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
        )

        # tuck it into the kwargs that will reach the LightningModule
        cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg
    else:
        cfg["model"]["kwargs"].pop("decoder_cfg", None)
        cfg["model"]["kwargs"]["gene_decoder_bool"] = False

    # Save one-hot maps as artifacts instead of storing them in config
    cell_type_onehot_map_path = join(run_output_dir, "cell_type_onehot_map.torch")
    pert_onehot_map_path = join(run_output_dir, "pert_onehot_map.pt")
    batch_onehot_map_path = join(run_output_dir, "batch_onehot_map.torch")
    var_dims_path = join(run_output_dir, "var_dims.pkl")

    torch.save(data_module.cell_type_onehot_map, cell_type_onehot_map_path)
    torch.save(data_module.pert_onehot_map, pert_onehot_map_path)
    torch.save(data_module.batch_onehot_map, batch_onehot_map_path)
    with open(var_dims_path, "wb") as f:
        pickle.dump(var_dims, f)

    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        data_module.get_var_dims(),
    )

    print(
        f"Model created. Estimated params size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3:.2f} GB"
    )
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
        monitor_metric=checkpoint_monitor_metric,
    )
    # Add BatchSpeedMonitorCallback to log batches per second to wandb
    batch_speed_monitor = BatchSpeedMonitorCallback()

    callbacks = ckpt_callbacks + [batch_speed_monitor]

    # Track gradient norm only for state transition model
    if cfg["model"]["name"] == "state":
        callbacks.append(GradNormCallback())

    # Add ModelFLOPSUtilizationCallback to track and log MFU. currently only works for state transition model
    if cfg["training"]["use_mfu"] and cfg["model"]["name"] == "state":
        mfu_available_flops = cfg["training"]["mfu_kwargs"]["available_flops"]
        mfu_use_backward = cfg["training"]["mfu_kwargs"]["use_backward"]
        mfu_logging_interval = cfg["training"]["mfu_kwargs"]["logging_interval"]
        mfu_window_size = cfg["training"]["mfu_kwargs"]["window_size"]
        mfu_cb = ModelFLOPSUtilizationCallback(
            available_flops=mfu_available_flops,
            use_backward=mfu_use_backward,
            logging_interval=mfu_logging_interval,
            cell_set_len=cfg["model"]["kwargs"]["cell_set_len"],
            window_size=mfu_window_size,
        )

        callbacks.append(mfu_cb)

    if "cumulative_flops_use_backward" in cfg["training"] and cfg["model"]["name"] == "state":
        cumulative_flops_use_backward = cfg["training"]["cumulative_flops_use_backward"]
        cumulative_flops_cb = CumulativeFLOPSCallback(use_backward=cumulative_flops_use_backward)
        callbacks.append(cumulative_flops_cb)

    logger.info("Loggers and callbacks set up.")

    plugins = []

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    model_name_lower = cfg["model"]["name"].lower()
    effective_max_steps = cfg["training"]["max_steps"]
    if model_name_lower in {"perturb_mean", "context_mean"}:
        # Mean baselines do not require long training loops; force a short run.
        effective_max_steps = 1
        logger.info(f"Overriding max_steps to {effective_max_steps} for model={model_name_lower}")

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=cfg["training"].get("devices", 1),
        strategy=cfg["training"].get("strategy", "auto"),
        max_steps=effective_max_steps,  # override for mean baselines
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        accumulate_grad_batches=cfg["training"].get("gradient_accumulation_steps", 1),
        use_distributed_sampler=False,
    )

    # Align logging cadence with rolling MFU window (and W&B logging)
    if "log_every_n_steps" in cfg["training"]:
        trainer_kwargs["log_every_n_steps"] = cfg["training"]["log_every_n_steps"]

    # Build trainer
    print(f"Building trainer with kwargs: {trainer_kwargs}")
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer built successfully")

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    print(f"Model device: {next(model.parameters()).device}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    logger.info("Starting trainer fit.")

    # if a checkpoint does not exist, start with the provided checkpoint
    # this is mainly used for pretrain -> finetune workflows
    manual_init = cfg["model"]["kwargs"].get("init_from", None)
    if checkpoint_path is None and manual_init is not None:
        print(f"Loading manual checkpoint from {manual_init}")
        checkpoint_path = manual_init
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = model.state_dict()
        checkpoint_state = checkpoint["state_dict"]

        # Check if output_space differs between current config and checkpoint
        checkpoint_output_space = checkpoint.get("hyper_parameters", {}).get("output_space", "gene")
        current_output_space = cfg["data"]["kwargs"]["output_space"]

        if checkpoint_output_space != current_output_space:
            print(
                f"Output space mismatch: checkpoint has '{checkpoint_output_space}', current config has '{current_output_space}'"
            )
            print("Creating new decoder for the specified output space...")

            if cfg["model"]["kwargs"].get("gene_decoder_bool", True) == False or cfg["model"]["kwargs"].get(
                "nb_loss", False
            ):
                model._decoder_externally_configured = False
            else:
                # Override the decoder_cfg to match the new output_space
                if current_output_space == "gene":
                    new_gene_dim = var_dims.get("hvg_dim", 2000)
                else:  # output_space == "all"
                    new_gene_dim = var_dims.get("gene_dim", 2000)

                new_decoder_cfg = dict(
                    latent_dim=var_dims["output_dim"],
                    gene_dim=new_gene_dim,
                    hidden_dims=cfg["model"]["kwargs"].get("decoder_hidden_dims", [1024, 1024, 512]),
                    dropout=cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
                )

                # Update the model's decoder_cfg and rebuild decoder
                model.decoder_cfg = new_decoder_cfg
                model._build_decoder()
                model._decoder_externally_configured = True  # Mark that decoder was configured externally
                print(f"Created new decoder for output_space='{current_output_space}' with gene_dim={new_gene_dim}")

        pert_encoder_weight_key = "pert_encoder.0.weight"
        if pert_encoder_weight_key in checkpoint_state:
            checkpoint_pert_dim = checkpoint_state[pert_encoder_weight_key].shape[1]

            # if the cell embedding dim doesn't match, or if it was HVGs, rebuild for transfer learning
            if checkpoint_pert_dim != model.pert_dim or cfg["data"]["kwargs"]["embed_key"] == "X_hvg":
                print(
                    f"pert_encoder input dimension mismatch: model.pert_dim = {model.pert_dim} but checkpoint expects {checkpoint_pert_dim}. Overriding model's pert_dim and rebuilding pert_encoder."
                )
                # Rebuild the pert_encoder with the new pert input dimension
                from ...tx.models.utils import build_mlp

                model.pert_encoder = build_mlp(
                    in_dim=model.pert_dim,
                    out_dim=model.hidden_dim,
                    hidden_dim=model.hidden_dim,
                    n_layers=model.n_encoder_layers,
                    dropout=model.dropout,
                    activation=model.activation_class,
                )
            else:
                print("WARNING: pert_encoder will not be rebuilt since input dimension matches")

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
        print("About to call trainer.fit() with manual checkpoint...")

        # Train - for clarity we pass None
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=None,
        )
        print("trainer.fit() completed with manual checkpoint")
    else:
        print(f"About to call trainer.fit() with checkpoint_path={checkpoint_path}")
        # Train
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
        print("trainer.fit() completed")

    print("Training completed, saving final checkpoint...")

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)
