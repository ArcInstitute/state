#!/usr/bin/env python

import argparse
import os
import sys
import pickle
import re
import yaml
import logging
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import wandb

from tqdm import tqdm

# Import the relevant modules from your repository
from models.decoders import UCELogProbDecoder
from data.mapping_strategies import (
    CentroidMappingStrategy,
    ClusteringMappingStrategy,
    BatchMappingStrategy,
    RandomMappingStrategy,
    NearestNeighborMappingStrategy,
    PseudoBulkMappingStrategy,
)
from data.data_modules import MultiDatasetPerturbationDataModule
from validation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    """
    CLI for evaluation. The arguments mirror some of the old script_lightning/eval_lightning.py.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PerturbationModel.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )
    parser.add_argument(
        "--map_type",
        type=str,
        default=None,
        choices=[
            "centroid",
            "clustering",
            "batch",
            "random",
            "nearest",
            "pseudo_nearest",
            "pseudobulk",
        ],
        help="Override the mapping strategy at inference time (optional).",
    )
    return parser.parse_args()


def load_config(cfg_path: str) -> dict:
    """
    Load config from the YAML file that was dumped during training.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config file: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

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
    checkpoint_path = os.path.join(directory, f"step={max_step}.ckpt")
    
    return checkpoint_path

def main():
    args = parse_args()

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory & check for data_module.pkl
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.pkl")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(
            f"Could not find serialized data module at {data_module_path}.\n"
            "Did you remember to pickle it in the training script?"
        )

    # 3. Load the data module
    with open(data_module_path, "rb") as f:
        data_module: MultiDatasetPerturbationDataModule = pickle.load(f)
    logger.info("Loaded data module from %s", data_module_path)

    # If user overrides the mapping strategy
    if args.map_type is not None:
        # Build new mapping strategy
        mapping_cls = {
            "centroid": CentroidMappingStrategy,
            "clustering": ClusteringMappingStrategy,
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
            "nearest": NearestNeighborMappingStrategy,
            "pseudobulk": PseudoBulkMappingStrategy,
        }[args.map_type]

        # Example of typical kwargs you might want to pass (adapt to your needs):
        strategy_kwargs = {
            "random_state": cfg["training"]["train_seed"],
            "n_basal_samples": cfg["data"]["kwargs"]["n_basal_samples"],
            "k_neighbors": cfg["data"]["kwargs"].get("k_neighbors", 10),
            "neighborhood_fraction": cfg["data"]["kwargs"].get("neighborhood_fraction", 0.0),
        }
        data_module.set_inference_mapping_strategy(mapping_cls, **strategy_kwargs)
        logger.info("Inference mapping strategy set to %s", args.map_type)

    # 4. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    # checkpoint_path = get_latest_step_checkpoint(checkpoint_dir)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # The model architecture is determined by the config
    model_class_name = cfg["model"]["name"]  # e.g. "EmbedSum" or "NeuralOT"
    model_kwargs = cfg["model"]["kwargs"]  # dictionary of hyperparams

    # Build the correct class
    if model_class_name.lower() == "embedsum":
        from models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "neuralot":
        from models.neural_ot import NeuralOTPerturbationModel

        ModelClass = NeuralOTPerturbationModel
    elif model_class_name.lower() == "simplesum":
        from models.simple_sum import SimpleSumPerturbationModel

        ModelClass = SimpleSumPerturbationModel  # it would be great if this was automatically kept in sync with the model.__init__
    elif model_class_name.lower() == "globalsimplesum":
        from models.global_simple_sum import GlobalSimpleSumPerturbationModel

        ModelClass = GlobalSimpleSumPerturbationModel
    elif model_class_name.lower() == "celltypemean":
        from models.cell_type_mean import CellTypeMeanModel

        ModelClass = CellTypeMeanModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()  # e.g. input_dim, output_dim, pert_dim
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        # other model_kwargs keys to pass along:
        **model_kwargs,
    }

    # load checkpoint
    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # TODO - add batch size parameter properly
    # data_module.batch_size = 1

    # 5. Run inference on test set
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    
    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    logger.info("Generating predictions on test set using manual loop...")
    preds = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
            # Move each tensor in the batch to the model's device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            batch_preds = model.predict_step(batch, batch_idx, padded=False)
            # Move each tensor in the returned dict to CPU to free GPU memory
            batch_preds = {k: (v.cpu().float() if isinstance(v, torch.Tensor) else v) for k, v in batch_preds.items()}
            preds.append(batch_preds)
    logger.info("Aggregated predictions from manual loop.")

    # Flatten out
    all_preds = []
    all_basals = []
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_reals = []

    # If we have output_space = gene
    all_X_hvg = []
    all_gene_preds = []

    for item in preds:
        # item["preds"] shape: [B, #genes or embed_dim], etc.
        # Some or all might be None
        if item["preds"] is not None:
            all_preds.append(item["preds"].cpu().float().numpy())
        else:
            all_preds.append(None)

        if item["basal"] is not None:
            all_basals.append(item["basal"].cpu().float().numpy())
        else:
            all_basals.append(None)

        # metadata
        if item["pert_name"] is not None:
            # item["pert_name"] might be a list (if in the batch) or a single item
            if isinstance(item["pert_name"], list):
                all_pert_names.extend(item["pert_name"])
            else:
                all_pert_names.append(item["pert_name"])

        if item["celltype_name"] is not None:
            if isinstance(item["celltype_name"], list):
                all_celltypes.extend(item["celltype_name"])
            else:
                all_celltypes.append(item["celltype_name"])

        if item["gem_group"] is not None:
            # gem_group might be a list or single
            if isinstance(item["gem_group"], list):
                all_gem_groups.extend(item["gem_group"])
            elif isinstance(item["gem_group"], torch.Tensor):
                all_gem_groups.extend(item["gem_group"].cpu().float().numpy())
            else:
                all_gem_groups.append(item["gem_group"])

        if item["X"] is not None:
            all_reals.append(item["X"].cpu().float().numpy())
        else:
            all_reals.append(None)

        # Store X_hvg for HVG space ground truth
        if "X_hvg" in item and item["X_hvg"] is not None:
            all_X_hvg.append(item["X_hvg"].cpu().float().numpy())
        else:
            all_X_hvg.append(None)
        
        # Store the decoded gene predictions if available
        if "gene_preds" in item and item["gene_preds"] is not None:
            all_gene_preds.append(item["gene_preds"].cpu().float().numpy())
        else:
            all_gene_preds.append(None)

    # Because each predict_step might have a different batch size, we need to concatenate carefully
    final_preds = np.concatenate([arr for arr in all_preds if arr is not None], axis=0)
    final_reals = np.concatenate([arr for arr in all_reals if arr is not None], axis=0)

    # Handle HVG ground truth if available
    if any(x is not None for x in all_X_hvg):
        final_X_hvg = np.concatenate([x for x in all_X_hvg if x is not None], axis=0)
    else:
        final_X_hvg = None

    # Handle decoded gene predictions if available
    if any(x is not None for x in all_gene_preds):
        final_gene_preds = np.concatenate([x for x in all_gene_preds if x is not None], axis=0)
    else:
        final_gene_preds = None

    # Build adatas for pred and real
    obs = pd.DataFrame({"pert_name": all_pert_names, "celltype_name": all_celltypes, "gem_group": all_gem_groups})

    # Create adata for predictions
    adata_pred = anndata.AnnData(X=final_preds, obs=obs.copy())

    # Create adata for real
    adata_real = anndata.AnnData(X=final_reals, obs=obs.copy())

    # # TODO-Abhi: Remove this before merging, this is to account for a bug with training
    # # that didn't store the decoder
    # if data_module.embed_key == "X_uce" and cfg["data"]["kwargs"]["output_space"] == "latent":
    #     model.decoder = UCELogProbDecoder()
    # else:
    #     model.decoder = None

    # Create adata for real data in gene space (if available)
    adata_real_gene = None
    if final_X_hvg is not None:
        adata_real_gene = anndata.AnnData(X=final_X_hvg, obs=obs.copy())

    # Create adata for gene predictions (if available)
    adata_pred_gene = None
    if final_gene_preds is not None:
        adata_pred_gene = anndata.AnnData(X=final_gene_preds, obs=obs.copy())

    # 6. Compute metrics
    logger.info("Computing metrics for test set...")
    metrics = compute_metrics(
        adata_pred=adata_pred,
        adata_real=adata_real,
        adata_pred_gene=adata_pred_gene,
        adata_real_gene=adata_real_gene,
        include_dist_metrics=True,
        control_pert=data_module.get_control_pert(),
        pert_col="pert_name",
        celltype_col="celltype_name",
        model_loc=None,  # can pass a path if needed
        DE_metric_flag=True,
        class_score_flag=True,
        embed_key=data_module.embed_key,
        output_space=cfg["data"]["kwargs"]["output_space"],  # "gene" or "latent"
        decoder=model.decoder,
        shared_perts=data_module.get_shared_perturbations(),
    )

    # 7. Summarize results
    # The "metrics" object is a dict of celltype -> DataFrame.
    # We'll combine them and get mean.
    summary = []
    for celltype, df in metrics.items():
        if df.empty:
            continue
        for metric_name in df.columns:
            if metric_name in ["DE_pred_genes", "DE_true_genes", "pert"]:  # skip large text
                continue
            metric_vals = df[metric_name].dropna().values
            if len(metric_vals) == 0:
                continue
            summary.append(
                {
                    "celltype": celltype,
                    "metric_name": metric_name,
                    "metric_val": np.mean(metric_vals),
                }
            )

    summary_df = pd.DataFrame(summary)
    logger.info("Summary of metrics:\n" + str(summary_df))

    # 8. Save metrics to CSV
    eval_basedir = os.path.join(
        run_output_dir,
        "eval",
        f"map_{args.map_type or 'train_map'}",
        args.checkpoint.replace(".ckpt", ""),
    )
    os.makedirs(eval_basedir, exist_ok=True)
    metrics_csv_path = os.path.join(eval_basedir, "metrics.csv")
    summary_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Metrics saved to {metrics_csv_path}")

    # Also save the per-pert data frames:
    for celltype, df in metrics.items():
        celltype_csv = os.path.join(eval_basedir, f"metrics_{celltype}.csv")
        df.to_csv(celltype_csv)
    logger.info(f"Per-celltype metrics saved under {eval_basedir}")

    # 9. Optionally store results in wandb
    # Replicate logic from train_lightning where we wrote out the wandb info to a file
    wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
    if os.path.exists(wandb_info_path):
        with open(wandb_info_path, "r") as f:
            wandb_run_path = f.read().strip()
        # Attempt to connect to that run
        try:
            api = wandb.Api()
            eval_run = api.run(wandb_run_path)
            logger.info(f"Logging summary metrics to wandb run {wandb_run_path}")

            for _, row in summary_df.iterrows():
                metric_key = f"test_full/{row['metric_name']}_{row['celltype']}"
                eval_run.summary[metric_key] = float(row["metric_val"])

            # Compute the average value for each metric across cell types
            avg_metrics = summary_df.groupby("metric_name")["metric_val"].mean().reset_index()
            for _, row in avg_metrics.iterrows():
                metric_key = f"test_full/{row['metric_name']}"
                eval_run.summary[metric_key] = float(row["metric_val"])

            eval_run.summary.update()
            logger.info("Metrics updated in wandb run.")
        except wandb.CommError:
            logger.warning("Failed to connect to wandb run. No wandb logging done.")
    else:
        logger.info("No wandb_path.txt found; skipping wandb update.")


if __name__ == "__main__":
    main()