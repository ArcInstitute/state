#!/usr/bin/env python

import argparse
import os
import sys
import pickle
import re
import gc
import yaml
import logging
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from scipy.sparse import csr_matrix
from tqdm import tqdm

# Import the relevant modules from your repository
from models.decoders import UCELogProbDecoder
try:
    from models.decoders import VCICountsDecoder
except ImportError:
    logger.warning("Could not import VCICountsDecoder from models.decoders, submodule may be missing.")

from data.mapping_strategies import (
    BatchMappingStrategy,
    RandomMappingStrategy,
    PseudoBulkMappingStrategy,
)
from vc_load.data_modules import PerturbationDataModule
from validation.metrics import compute_metrics

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
        "--use_uce_decoder",
        action="store_true",
        help="Use the UCELogProbDecoder for decoding the model output (if applicable).",
    )
    parser.add_argument(
        "--use_vci_decoder",
        action="store_true",
        help="Use the VCICountsDecoder for decoding the model output (if applicable).",
    )
    parser.add_argument(
        "--read_depth",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--map_type",
        type=str,
        default="random",
        choices=[
            "batch",
            "random",
            "pseudobulk",
        ],
        help="Override the mapping strategy at inference time (optional).",
    )
    parser.add_argument(
        "--test_time_finetune",
        type=int,
        default=0,
        help="If >0, run test-time fine-tuning for the specified number of epochs on only control cells.",
    )

    return parser.parse_args()

def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device):
    """
    Perform test-time fine-tuning on only control cells.

    For each batch, we check the first sample’s perturbation label (since all
    samples in the batch share the same group). If it equals the control perturbation,
    we process the batch; otherwise, we skip it.

    Arguments:
        model: the loaded model to be fine-tuned.
        dataloader: the test dataloader (which samples cells as in your test split).
        ft_epochs: number of epochs to run fine-tuning.
        control_pert: the control perturbation name (from data.kwargs.control_pert).
        device: the device where the model is located.
    """
    model.train()
    # Use a small learning rate for test-time fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    logger.info(f"Starting test-time fine-tuning for {ft_epochs} epoch(s) on control cells only.")
    for epoch in range(ft_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch+1}/{ft_epochs}", leave=True)
        for batch in pbar:
            # Peek at the first perturbation name in this batch.
            first_pert = (
                batch["pert_name"][0]
                if isinstance(batch["pert_name"], list)
                else batch["pert_name"][0].item()
            )
            if first_pert != control_pert:
                continue

            # Move batch data to the model's device.
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx=0, padded=False)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        logger.info(f"Finetune epoch {epoch+1}/{ft_epochs}, mean loss: {mean_loss}")
    model.eval()

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
        data_module: PerturbationDataModule = pickle.load(f)
    logger.info("Loaded data module from %s", data_module_path)

    # seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # If user overrides the mapping strategy
    if args.map_type is not None:
        # Build new mapping strategy
        mapping_cls = {
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
            "pseudobulk": PseudoBulkMappingStrategy,
        }[args.map_type]

        # Example of typical kwargs you might want to pass (adapt to your needs):
        strategy_kwargs = {
            "random_state": cfg["training"]["train_seed"],
            "n_basal_samples": cfg["data"]["kwargs"]["n_basal_samples"],
            "k_neighbors": cfg["data"]["kwargs"].get("k_neighbors", 10),
            "neighborhood_fraction": cfg["data"]["kwargs"].get("neighborhood_fraction", 0.0),
            "map_controls": cfg["data"]["kwargs"].get("map_controls", False),
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
    elif model_class_name.lower() == "old_neuralot":
        from models.old_neural_ot import OldNeuralOTPerturbationModel
        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() == "neuralot" or model_class_name.lower() == "pertsets":
        from models.pert_sets import PertSetsPerturbationModel
        ModelClass = PertSetsPerturbationModel
    elif model_class_name.lower() == "simplesum":
        from models.simple_sum import SimpleSumPerturbationModel
        ModelClass = SimpleSumPerturbationModel  # it would be great if this was automatically kept in sync with the model.__init__
    elif model_class_name.lower() == "globalsimplesum":
        from models.global_simple_sum import GlobalSimpleSumPerturbationModel
        ModelClass = GlobalSimpleSumPerturbationModel
    elif model_class_name.lower() == "celltypemean":
        from models.cell_type_mean import CellTypeMeanModel
        ModelClass = CellTypeMeanModel
    elif model_class_name.lower() == "decoder_only":
        from models.decoder_only import DecoderOnlyPerturbationModel
        ModelClass = DecoderOnlyPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()  # e.g. input_dim, output_dim, pert_dim
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        # other model_kwargs keys to pass along:
        **model_kwargs,
    }

    # load checkpoint
    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    data_module.batch_size = 1
    if args.test_time_finetune > 0:
        # Get the control perturbation from the data module
        control_pert = data_module.get_control_pert()
        # Run finetuning – use the test dataloader so that you only get cells from the test split.
        test_loader = data_module.test_dataloader()
        run_test_time_finetune(model, test_loader, args.test_time_finetune, control_pert, device=next(model.parameters()).device)
        logger.info("Test-time fine-tuning complete.")

    # 5. Run inference on test set
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    
    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    num_cells = test_loader.batch_sampler.tot_num
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    final_preds = np.empty((num_cells, output_dim), dtype=np.float16)
    final_reals = np.empty((num_cells, output_dim), dtype=np.float16)

    store_raw_expression = (
        data_module.embed_key is not None and 
        data_module.embed_key != "X_hvg" and 
        cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (
        data_module.embed_key is not None and 
        cfg["data"]["kwargs"]["output_space"] == "all"
    )

    if store_raw_expression:
        # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
        if cfg["data"]["kwargs"]["output_space"] == "gene":
            final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float16)
            final_gene_preds = np.empty((num_cells, hvg_dim), dtype=np.float16)
        if cfg["data"]["kwargs"]["output_space"] == "all":
            final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float16)
            final_gene_preds = np.empty((num_cells, gene_dim), dtype=np.float16)
    else:
        # Otherwise, use lists for later concatenation.
        final_X_hvg = None
        final_gene_preds = None

    current_idx = 0

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
            # Move each tensor in the batch to the model's device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
            
            # Get predictions
            batch_preds = model.predict_step(batch, batch_idx, padded=False)
            
            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
            else:
                all_pert_names.append(batch_preds["pert_name"])
            
            # Handle celltype_name
            if isinstance(batch_preds["celltype_name"], list):
                all_celltypes.extend(batch_preds["celltype_name"])
            else:
                all_celltypes.append(batch_preds["celltype_name"])
            
            # Handle gem_group
            if isinstance(batch_preds["gem_group"], list):
                all_gem_groups.extend(batch_preds["gem_group"])
            elif isinstance(batch_preds["gem_group"], torch.Tensor):
                all_gem_groups.extend(batch_preds["gem_group"].cpu().numpy())
            else:
                all_gem_groups.append(batch_preds["gem_group"])
            
            batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float16)
            batch_real_np = batch_preds["X"].cpu().numpy().astype(np.float16)
            batch_size = batch_pred_np.shape[0]
            final_preds[current_idx:current_idx+batch_size, :] = batch_pred_np
            final_reals[current_idx:current_idx+batch_size, :] = batch_real_np
            current_idx += batch_size
            
            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["X_hvg"].cpu().numpy().astype(np.float16)
                final_X_hvg[current_idx-batch_size:current_idx, :] = batch_real_gene_np
            
            # Handle decoded gene predictions if available
            if final_gene_preds is not None:
                batch_gene_pred_np = batch_preds["gene_preds"].cpu().numpy().astype(np.float16)
                final_gene_preds[current_idx-batch_size:current_idx, :] = batch_gene_pred_np


    logger.info("Creating anndatas from predictions from manual loop...")


    # Build pandas DataFrame for obs
    obs = pd.DataFrame({"pert_name": all_pert_names, "celltype_name": all_celltypes, "gem_group": all_gem_groups})

    # Create adata for predictions
    adata_pred = anndata.AnnData(X=final_preds, obs=obs)
    # Create adata for real
    adata_real = anndata.AnnData(X=final_reals, obs=obs)

    if args.use_uce_decoder:
        assert data_module.embed_key == "X_uce", "UCELogProbDecoder can only be used with UCE embeddings."
        logger.info("Using UCELogProbDecoder for decoding.")
        decoder = UCELogProbDecoder()
    elif args.use_vci_decoder:
        assert data_module.embed_key == "X_vci", "VCICountsDecoder can only be used with VCI embeddings."
        logger.info("Using VCICountsDecoder for decoding.")
        decoder = VCICountsDecoder(read_depth=args.read_depth)
    else:
        decoder = None

    # Create adata for real data in gene space (if available)
    adata_real_gene = None
    if final_X_hvg is not None: # either this is available, or we are already working in gene space
        if 'int_counts' in data_module.__dict__ and data_module.int_counts:
            final_X_hvg = np.log1p(final_X_hvg)
        adata_real_gene = anndata.AnnData(X=final_X_hvg, obs=obs)

    # Create adata for gene predictions (if available)
    adata_pred_gene = None
    if final_gene_preds is not None and decoder is None: # otherwise we use UCE log prob decoder
        if 'int_counts' in data_module.__dict__ and data_module.int_counts:
            final_gene_preds = np.log1p(final_gene_preds)
        adata_pred_gene = anndata.AnnData(X=final_gene_preds, obs=obs)

    # # save out adata_real to the output directory
    # adata_real_out = os.path.join(args.output_dir, 'adata_real.h5ad')
    # adata_real.write_h5ad(adata_real_out)
    # logger.info(f"Saved adata_real to {adata_real_out}")
    #
    # adata_pred_out = os.path.join(args.output_dir, 'adata_pred.h5ad')
    # adata_pred.write_h5ad(adata_pred_out)
    # logger.info(f"Saved adata_pred to {adata_pred_out}")

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
        shared_perts=data_module.get_shared_perturbations(),
        decoder=decoder,
        outdir=args.output_dir,
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