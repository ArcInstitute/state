#!/usr/bin/env python

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'vci_pretrain'))

import argparse
import pickle
import yaml
import logging
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the relevant modules from your repository
from models.decoders import UCELogProbDecoder
try:
    from models.decoders import VCICountsDecoder
except ImportError:
    logger.warning("Could not import VCICountsDecoder from models.decoders, submodule may be missing.")

# Dictionary to store layer activations
layer_activations = {}

def get_activation(name):
    def hook(model, input, output):
        layer_activations[name] = output.detach()
    return hook

def register_hooks(model):
    # Clear any existing hooks
    layer_activations.clear()
    
    # Register hooks for each transformer layer
    for i, layer in enumerate(model.transformer_backbone.h):
        layer.register_forward_hook(get_activation(f'layer_{i}'))

def parse_args():
    """
    CLI for evaluation with layer analysis.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PerturbationModel with layer analysis.")
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
        "--data_dir",
        type=str,
        default="/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct_vci_1.4.2",
        help="Directory containing the data plates",
    )
    parser.add_argument(
        "--max_perturbations",
        type=int,
        default=5,
        help="Maximum number of perturbations to process",
    )
    parser.add_argument(
        "--max_cells_per_pert",
        type=int,
        default=100,
        help="Maximum number of cells to process per perturbation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name from config (e.g., pertsets, neuralot, etc.)",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to a model YAML config file to load and merge into the run config.",
    )
    # Accept arbitrary CLI overrides for model.kwargs
    parser.add_argument(
        'overrides',
        nargs=argparse.REMAINDER,
        help="Additional model.kwargs overrides in the form --model.kwargs.KEY=VALUE"
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

def plot_layer_contributions(layer_contributions, perturbation, output_dir):
    """Plot the contribution of each layer to the prediction"""
    plt.figure(figsize=(10, 6))
    layers = list(layer_contributions.keys())
    contributions = list(layer_contributions.values())
    
    plt.bar(layers, contributions)
    plt.title(f'Layer-wise Contributions for {perturbation}')
    plt.xlabel('Transformer Layer')
    plt.ylabel('Average Change in Representation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    layer_analysis_dir = os.path.join(output_dir, 'layer_analysis')
    os.makedirs(layer_analysis_dir, exist_ok=True)
    plt.savefig(os.path.join(layer_analysis_dir, f'{perturbation}_layer_contributions.png'))
    plt.close()

def main():
    args = parse_args()

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Optionally load and merge model config from YAML
    if args.model_config is not None:
        logger.info(f"Merging model config from YAML: {args.model_config}")
        with open(args.model_config, "r") as f:
            model_yaml = yaml.safe_load(f)
        # Merge top-level model keys
        for k, v in model_yaml.items():
            if k == "kwargs":
                # Merge kwargs dict
                cfg["model"].setdefault("kwargs", {})
                cfg["model"]["kwargs"].update(v)
            else:
                cfg["model"][k] = v

    # Optionally override model name from CLI
    if args.model is not None:
        logger.info(f"Overriding model name in config with: {args.model}")
        cfg["model"]["name"] = args.model

    # Parse CLI overrides for model.kwargs
    for override in args.overrides:
        if override.startswith("--model.kwargs."):
            keyval = override[len("--model.kwargs."):]
            if '=' in keyval:
                key, val = keyval.split('=', 1)
                # Try to interpret as int, float, bool, or keep as string
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                cfg["model"].setdefault("kwargs", {})
                cfg["model"]["kwargs"][key] = val
                logger.info(f"Override: model.kwargs.{key} = {val}")

    # 2. Load the data
    logger.info(f"Loading data from {args.data_dir}")
    adata = sc.read_h5ad(os.path.join(args.data_dir, "plate1.h5"), backed='r')
    logger.info(f"Loaded data in backed mode with shape {adata.shape}")

    # Now, to get a small subset in memory:
    adata_subset = adata[:100].to_memory()
    logger.info(f"Subsetted data to shape {adata_subset.shape}")

    # Use adata_subset for the rest of your script
    adata = adata_subset

    # 3. Load the trained model
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # The model architecture is determined by the config
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # --- Extract required dims as in __main__.py ---
    # For inference, we try to infer dims from the config or data
    # Try to get input_dim, output_dim, pert_dim, batch_dim, gene_dim, hvg_dim
    # Fallback to model_kwargs if not present
    def get_dim(key, default=None):
        return model_kwargs.get(key, default)

    input_dim = get_dim("input_dim", adata.shape[1])
    output_dim = get_dim("output_dim", adata.shape[1])
    pert_dim = get_dim("pert_dim", 696)
    batch_dim = get_dim("batch_dim", None)
    gene_dim = get_dim("gene_dim", None)
    hvg_dim = get_dim("hvg_dim", None)

    # Some models require these dims, some don't
    model_init_kwargs = dict()
    if input_dim is not None:
        model_init_kwargs["input_dim"] = input_dim
    if output_dim is not None:
        model_init_kwargs["output_dim"] = output_dim
    if pert_dim is not None:
        model_init_kwargs["pert_dim"] = pert_dim
    if batch_dim is not None:
        model_init_kwargs["batch_dim"] = batch_dim
    if gene_dim is not None:
        model_init_kwargs["gene_dim"] = gene_dim
    if hvg_dim is not None:
        model_init_kwargs["hvg_dim"] = hvg_dim

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
        ModelClass = SimpleSumPerturbationModel
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

    # Initialize model with required dims and model_kwargs
    model = ModelClass(
        **model_init_kwargs,
        **model_kwargs,
    )

    # load checkpoint
    model = ModelClass.load_from_checkpoint(checkpoint_path)
    model.eval()
    logger.info("Model loaded successfully.")

    # Register hooks for layer analysis
    register_hooks(model)

    # 4. Get all unique perturbations and count cells per perturbation
    pert_counts = adata.obs["gene"].value_counts()
    logger.info(f"Found {len(pert_counts)} unique perturbations")
    logger.info(f"Top 5 perturbations by frequency: {pert_counts.head().to_dict()}")

    # Control cells to sample from
    control_pert = "control"
    all_ctrl_cells = adata[adata.obs["gene"] == control_pert].copy()
    logger.info(f"Total control cells available: {all_ctrl_cells.n_obs}")

    # 5. Generate predictions for each perturbation
    all_predictions = []
    all_layer_contributions = {}
    np.random.seed(42)  # For reproducibility

    device = next(model.parameters()).device

    # Limit number of perturbations to process
    perturbations_to_process = pert_counts.index[:args.max_perturbations]
    logger.info(f"Processing {len(perturbations_to_process)} perturbations")

    # Create progress bar for selected perturbations
    for perturbation in tqdm(perturbations_to_process, desc="Processing perturbations"):
        # Get number of cells for this perturbation, but limit it
        n_cells = min(pert_counts[perturbation], args.max_cells_per_pert)

        # Sample this many control cells (with replacement if needed)
        if n_cells <= all_ctrl_cells.n_obs:
            # Sample without replacement if we have enough cells
            sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells, replace=False)
        else:
            # Sample with replacement if we need more cells than available
            sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells, replace=True)

        ctrl_subset = all_ctrl_cells[sample_indices].copy()

        # Set the perturbation
        ctrl_subset.obs["gene"] = perturbation

        # Prepare batch for model
        batch = {
            "X": torch.tensor(ctrl_subset.X, device=device),
            "pert_name": perturbation,
            "celltype_name": ctrl_subset.obs["cell_type"].values,
        }

        # Run inference with layer analysis
        try:
            with torch.no_grad():
                batch_preds = model.predict_step(batch, 0, padded=False)
            
            # Calculate layer-wise contributions
            layer_contributions = {}
            prev_activation = None
            
            for i in range(len(model.transformer_backbone.h)):
                layer_name = f'layer_{i}'
                if layer_name in layer_activations:
                    current_activation = layer_activations[layer_name]
                    
                    if prev_activation is not None:
                        # Calculate how much this layer changed the representation
                        change = torch.norm(current_activation - prev_activation, dim=-1).mean().item()
                        layer_contributions[f'Layer {i}'] = change
                    
                    prev_activation = current_activation

            # Store predictions and layer contributions
            all_predictions.append(batch_preds)
            
            if perturbation not in all_layer_contributions:
                all_layer_contributions[perturbation] = []
            all_layer_contributions[perturbation].append(layer_contributions)

            logger.info(f"Generated predictions for {perturbation}")
            logger.info(f"Layer contributions: {layer_contributions}")

        except Exception as e:
            logger.error(f"Error generating predictions for {perturbation}: {str(e)}")

    # Average layer contributions per perturbation
    avg_layer_contributions = {}
    for pert_name, contributions_list in all_layer_contributions.items():
        avg_contributions = {}
        for layer in contributions_list[0].keys():
            avg_contributions[layer] = np.mean([c[layer] for c in contributions_list])
        avg_layer_contributions[pert_name] = avg_contributions
        
        # Plot layer contributions for this perturbation
        plot_layer_contributions(avg_contributions, pert_name, args.output_dir)

    logger.info("Layer analysis complete. Plots saved in layer_analysis directory.")

if __name__ == "__main__":
    main()