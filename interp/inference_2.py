import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
from typing import Optional, List, Dict, Tuple
import os
import sys
import pandas as pd
import anndata as ad
from tqdm.notebook import tqdm
import os
from pathlib import Path
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory (state-sets) to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'vci_pretrain'))

# Now import from inference_module
from inference_module import InferenceModule

# pert mean model
# pert_mean_baseline_rpe1_fewshot = "/large_storage/ctc/userspace/aadduri/mar5/replogle_hvg_gss/fold4"

# pert sets model trained on HVG space
pert_sets_hvg_ctrl_rpe1_fewshot = "/large_storage/ctc/userspace/dhruvgautam/qwen/qwen_mha_hvg_sm_cs64_hvg_esm_new/qwen_replo_fold4"

pert_sets = "/large_storage/ctc/userspace/dhruvgautam/qwen/tahoe_vci_1.4.4_70m_mha/mha_fold1"

#checkpoints/step=56000.ckpt
# pert sets model trained on HVG space
# pert_sets_scgpt_rpe1_fewshot = "/large_storage/ctc/userspace/aadduri/mar5/replogle_scgpt_samp_ctrl/fold4"

inference_module = InferenceModule(model_folder=pert_sets)

# Read the full AnnData
adata = sc.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct_vci_1.4.4/plate1.h5")

# Stratified sampling: ensure both control and non-control cells are present in the subset
control_pert = None
try:
    control_pert = InferenceModule(model_folder=pert_sets).data_module.control_pert
except Exception as e:
    logger.info(f"Could not determine control perturbation: {e}")
    control_pert = None

if control_pert is not None:
    control_mask = adata.obs["drugname_drugconc"] == control_pert
    control_cells = adata[control_mask]
    non_control_cells = adata[~control_mask]

    n_control = min(50, control_cells.n_obs)  # e.g., 50 control cells
    n_non_control = 512 - n_control
    if n_non_control > 0 and non_control_cells.n_obs > 0:
        n_non_control = min(n_non_control, non_control_cells.n_obs)
    else:
        n_non_control = 0

    if n_control > 0:
        control_sample = control_cells[np.random.choice(control_cells.n_obs, n_control, replace=False)]
    else:
        control_sample = None
    if n_non_control > 0:
        non_control_sample = non_control_cells[np.random.choice(non_control_cells.n_obs, n_non_control, replace=False)]
    else:
        non_control_sample = None

    samples = []
    if control_sample is not None:
        samples.append(control_sample)
    if non_control_sample is not None:
        samples.append(non_control_sample)
    if samples:
        adata_subset = ad.concat(samples)
    else:
        adata_subset = adata[:512].to_memory()  # fallback if something goes wrong
else:
    logger.info("No control perturbation found; using first 512 cells as fallback.")
    adata_subset = adata[:512].to_memory()

logger.info(f"Subsetted data to shape {adata_subset.shape}")

# Use adata_subset for the rest of your script
adata = adata_subset

# tahoe pert sets
# inference_module = InferenceModule(model_folder='/large_storage/ctc/userspace/aadduri/feb28_conc/cv_scgpt/fold1', cell_set_len=512)
# adata = sc.read_h5ad('/large_storage/ctc/userspace/aadduri/datasets/split_tahoe_45_ct_hvg/plate6_split12_emb.h5')

# 1. Get all unique perturbations and count cells per perturbation
logger.info(adata.obs.columns)
pert_counts = adata.obs["drugname_drugconc"].value_counts()
logger.info(f"Found {len(pert_counts)} unique perturbations")
logger.info(f"Top 5 perturbations by frequency: {pert_counts.head().to_dict()}")

# Control cells to sample from
all_ctrl_cells = adata[adata.obs["drugname_drugconc"] == inference_module.data_module.control_pert].copy()
logger.info(f"Total control cells available: {all_ctrl_cells.n_obs}")

# --- BEGIN INTERPRETABILITY CODE ---
layer_activations = {}

def get_activation(name):
    def hook(model, input, output):
        logger.info(f"Hook triggered for {name}")
        # If output is a tuple, take the first element
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        layer_activations[name] = out.detach().cpu()
    return hook

def register_hooks(model):
    layer_activations.clear()
    if hasattr(model, 'transformer_backbone') and hasattr(model.transformer_backbone, 'layers'):
        for i, layer in enumerate(model.transformer_backbone.layers):
            layer.register_forward_hook(get_activation(f'layer_{i}'))
    else:
        logger.info("Model does not have a transformer_backbone with layers to hook.")

def plot_layer_contributions(layer_contributions, perturbation, output_dir):
    plt.figure(figsize=(10, 6))
    layers = list(layer_contributions.keys())
    contributions = list(layer_contributions.values())
    plt.bar(layers, contributions)
    plt.title(f'Layer-wise Contributions for {perturbation}')
    plt.xlabel('Transformer Layer')
    plt.ylabel('Average Change in Representation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    layer_analysis_dir = os.path.join(output_dir, 'layer_analysis')
    os.makedirs(layer_analysis_dir, exist_ok=True)
    plt.savefig(os.path.join(layer_analysis_dir, f'{perturbation}_layer_contributions.png'))
    plt.close()

# Register hooks on the model
register_hooks(inference_module.model)

# Layer analysis for each perturbation
all_layer_contributions = {}
figures_dir = os.path.join(os.path.dirname(__file__), "figures")
layer_analysis_dir = os.path.join(figures_dir, "layer_analysis")
os.makedirs(layer_analysis_dir, exist_ok=True)

# 2. Generate predictions for each perturbation
all_predictions = []
np.random.seed(42)  # For reproducibility

# Create progress bar for all perturbations
for perturbation in pert_counts.index:
    # Get number of cells for this perturbation
    n_cells = pert_counts[perturbation]
    logger.info(f"Generating predictions for {perturbation} with {n_cells} cells")

    # Sample this many control cells (with replacement if needed)
    if n_cells <= all_ctrl_cells.n_obs:
        # Sample without replacement if we have enough cells
        sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells, replace=False)
    else:
        # Sample with replacement if we need more cells than available
        sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells, replace=True)

    ctrl_subset = all_ctrl_cells[sample_indices].copy()

    # Set the perturbation
    ctrl_subset.obs["drugname_drugconc"] = perturbation

    # Clear activations before inference
    layer_activations.clear()

    # Run inference
    try:
        pred_subset = inference_module.perturb(
            ctrl_subset,
            pert_key='drugname_drugconc', # tahoe
            celltype_key='cell_name'
        )

        # Immediately after inference, copy activations
        current_activations = layer_activations.copy()

        # Add metadata to track source perturbation
        pred_subset.obs["original_perturbation"] = perturbation
        pred_subset.obs["is_predicted"] = True
        # Store this batch of predictions
        all_predictions.append(pred_subset)

        logger.info(f"Generated {pred_subset.n_obs} predictions for {perturbation}")

        # --- BEGIN LAYER CONTRIBUTION ANALYSIS FOR THIS PERTURBATION ---
        layer_contributions = {}
        prev_activation = None
        for i in range(len(getattr(inference_module.model.transformer_backbone, 'layers', []))):
            layer_name = f'layer_{i}'
            if layer_name in current_activations:
                current_activation = current_activations[layer_name]
                if prev_activation is not None:
                    # Calculate mean L2 norm change across all cells
                    change = torch.norm(current_activation - prev_activation, dim=-1).mean().item()
                    layer_contributions[f'Layer {i}'] = change
                prev_activation = current_activation
        if layer_contributions:
            all_layer_contributions[perturbation] = layer_contributions
            # plot_layer_contributions(layer_contributions, perturbation, figures_dir)
            logger.info(f"Saved layer contribution plot for {perturbation} to {layer_analysis_dir}")
        else:
            logger.info(f"No layer activations found for {perturbation}")
        # --- END LAYER CONTRIBUTION ANALYSIS ---
    except Exception as e:
        logger.info(f"Error generating predictions for {perturbation}: {str(e)}")

# --- Combine and average layer contributions across all perturbations ---
if all_layer_contributions:
    # Get all layer names (assume all perturbations have the same layers)
    layer_names = list(next(iter(all_layer_contributions.values())).keys())
    # Collect values for each layer
    layer_values = {layer: [] for layer in layer_names}
    for pert, contribs in all_layer_contributions.items():
        for layer in layer_names:
            layer_values[layer].append(contribs.get(layer, np.nan))
    # Compute mean for each layer
    avg_layer_contributions = {layer: np.nanmean(vals) for layer, vals in layer_values.items()}
    logger.info("Average layer contributions across all perturbations:")
    for layer, avg in avg_layer_contributions.items():
        logger.info(f"{layer}: {avg:.4f}")
    # Optionally, plot
    plot_layer_contributions(avg_layer_contributions, "average_across_perturbations", figures_dir)
else:
    logger.info("No layer contributions to average.")

# 3. Combine all predictions into a single AnnData object
if all_predictions:
    # Concatenate all predictions
    pred_adata = ad.concat(all_predictions, merge="same")

    # Make sure we have the same gene set as the original dataset
    if adata.var_names.equals(pred_adata.var_names):
        logger.info("Prediction gene set matches original dataset")
    else:
        logger.info("Warning: prediction gene set differs from original dataset")

    logger.info(f"Generated complete prediction dataset with {pred_adata.n_obs} cells")
    logger.info(f"Original dataset has {adata.n_obs} cells")

    # 4. Basic comparison statistics
    logger.info("\nPerturbations comparison:")
    pred_pert_counts = pred_adata.obs["original_perturbation"].value_counts()
    comparison_df = pd.DataFrame({"Real_cells": pert_counts, "Predicted_cells": pred_pert_counts}).fillna(0)

    # Calculate percentage match
    comparison_df["Cell_count_match"] = comparison_df.apply(
        lambda row: min(row["Real_cells"], row["Predicted_cells"])
        / max(row["Real_cells"], row["Predicted_cells"])
        * 100
        if max(row["Real_cells"], row["Predicted_cells"]) > 0
        else 0,
        axis=1,
    )

    logger.info(comparison_df.sort_values("Real_cells", ascending=False).head(10))

    # Output prediction AnnData for downstream analysis
    pred_adata
else:
    logger.info("No predictions were generated successfully")


# Plot embeddings. we can plot X_hvg vs X_hvg_pert
real_embeddings = adata.obsm["X_hvg"]
pred_embeddings = pred_adata.obsm["X_hvg_pert"]

# # Or we can plot X_scGPT vs X_scGPT_pert
# real_embeddings = adata.obsm['X_scGPT']
# pred_embeddings = pred_adata.obsm['X_scGPT_pert']

# Combine them
combined_embeddings = np.vstack([real_embeddings, pred_embeddings])
sources = ["real"] * adata.n_obs + ["pred"] * pred_adata.n_obs

# Create an integrated AnnData for visualization
integrated = ad.AnnData(X=combined_embeddings)
integrated.obs["source"] = sources
integrated.obs["perturbation"] = list(adata.obs["drugname_drugconc"]) + list(pred_adata.obs["original_perturbation"])

# Ensure the figures directory exists
figures_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figures_dir, exist_ok=True)

sc.pp.pca(integrated)
sc.pp.neighbors(integrated, use_rep="X_pca")
sc.tl.umap(integrated)

# Save UMAP plot to file
umap_fig_path = os.path.join(figures_dir, "umap_real_vs_pred.png")
sc.pl.umap(
    integrated,
    color="source",
    palette={"real": "blue", "pred": "red"},
    show=False,
    save=None,
)
logger.info(f"Saved UMAP plot to {umap_fig_path}")
plt.savefig(umap_fig_path)
plt.close()