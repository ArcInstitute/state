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

# Import model class directly
from models.pertsets import PertSetsPerturbationModel

# Replogle model path
MODEL_DIR = Path("/large_storage/ctc/userspace/rohankshah/preprint/replogle_gpt_31043724/hepg2")

# Load model directly from checkpoint
checkpoint_path = MODEL_DIR / "checkpoints" / "last.ckpt"

if not checkpoint_path.exists():
    # Try other common checkpoint names
    for ckpt_name in ["best.ckpt", "last.ckpt", "epoch=*.ckpt"]:
        ckpt_files = list(MODEL_DIR.glob(f"checkpoints/{ckpt_name}"))
        if ckpt_files:
            checkpoint_path = ckpt_files[0]
            break
    else:
        raise FileNotFoundError(f"Could not find checkpoint in {MODEL_DIR}/checkpoints/")

logger.info(f"Loading model from checkpoint: {checkpoint_path}")
model = PertSetsPerturbationModel.load_from_checkpoint(str(checkpoint_path), strict=False)
model.eval()  # turn off dropout

# Read the replogle data
adata = sc.read_h5ad("/large_storage/ctc/ML/state_sets/replogle/processed.h5")

# Determine embedding key - typically "X_vci" for VCI embeddings or "X_hvg" for HVG expression
embed_key = "X_vci_1.5.2_4"  # Try VCI embeddings first
if embed_key not in adata.obsm:
    embed_key = "X_hvg"  # Fallback to HVG
    if embed_key not in adata.obsm:
        embed_key = "X"  # Last resort - raw expression
        logger.warning("Using raw expression data - model may not work properly")

logger.info(f"Using embedding key: {embed_key}")
logger.info(f"Subsetted data to shape {adata.shape}")

# Find control perturbation
control_pert = "DMSO_TF_24h"  # Default control
if "gene" in adata.obs.columns:
    drugname_counts = adata.obs["gene"].value_counts()
    # Look for common control names
    for potential_control in ["DMSO_TF_24h", "non-targeting", "control", "DMSO"]:
        if potential_control in drugname_counts.index:
            control_pert = potential_control
            break
    else:
        # Use the most common perturbation as control
        control_pert = drugname_counts.index[0]
        logger.warning(f"Could not find standard control, using: {control_pert}")

logger.info(f"Control perturbation: {control_pert}")

# 1. Get all unique perturbations and count cells per perturbation
logger.info(adata.obs.columns)
pert_counts = adata.obs["gene"].value_counts()
logger.info(f"Found {len(pert_counts)} unique perturbations")
logger.info(f"Top 5 perturbations by frequency: {pert_counts.head().to_dict()}")

# Control cells to sample from
all_ctrl_cells = adata[adata.obs["gene"] == control_pert].copy()
logger.info(f"Total control cells available: {all_ctrl_cells.n_obs}")

# --- BEGIN INTERPRETABILITY CODE ---
head_activations = {}

def get_head_activation(name):
    def hook(model, input, output):
        logger.info(f"Head hook triggered for {name}")
        # For attention heads, we typically want the output before concatenation
        # This depends on the specific transformer architecture
        if isinstance(output, tuple):
            # Some attention modules return (attn_output, attn_weights)
            out = output[0]
        else:
            out = output
        head_activations[name] = out.detach().cpu()
    return hook

def register_head_hooks(model):
    head_activations.clear()
    # Updated for GPT-2 architecture using .h instead of .layers
    if hasattr(model, 'transformer_backbone') and hasattr(model.transformer_backbone, 'h'):
        for layer_idx, layer in enumerate(model.transformer_backbone.h):
            # Try to hook into multi-head attention
            if hasattr(layer, 'attn'):
                # GPT-2 uses 'attn' for attention module
                layer.attn.register_forward_hook(get_head_activation(f'layer_{layer_idx}_attn'))
            elif hasattr(layer, 'self_attn') or hasattr(layer, 'attention'):
                # Common attribute names for attention in different architectures
                attn_module = getattr(layer, 'self_attn', None) or getattr(layer, 'attention', None)
                if attn_module is not None:
                    # Hook the attention module output
                    attn_module.register_forward_hook(get_head_activation(f'layer_{layer_idx}_attn'))
            else:
                logger.info(f"Could not find attention module in layer {layer_idx}")
    else:
        logger.info("Model does not have a transformer_backbone with .h layers to hook.")

def calculate_head_contributions(head_activations, num_heads_per_layer=8):
    """
    Calculate contributions for each attention head.
    This assumes the attention output contains concatenated head outputs.
    """
    head_contributions = {}
    
    for layer_attn_name, activation in head_activations.items():
        if activation is None:
            continue
            
        layer_idx = layer_attn_name.split('_')[1]
        
        # Assuming activation shape is [batch_size, seq_len, hidden_dim]
        # Split into individual heads
        batch_size, seq_len, hidden_dim = activation.shape
        head_dim = hidden_dim // num_heads_per_layer
        
        if hidden_dim % num_heads_per_layer != 0:
            logger.warning(f"Hidden dim {hidden_dim} not divisible by num_heads {num_heads_per_layer}")
            continue
            
        # Reshape to separate heads: [batch_size, seq_len, num_heads, head_dim]
        head_outputs = activation.view(batch_size, seq_len, num_heads_per_layer, head_dim)
        
        # Calculate contribution for each head (mean activation magnitude)
        for head_idx in range(num_heads_per_layer):
            head_output = head_outputs[:, :, head_idx, :]  # [batch_size, seq_len, head_dim]
            # Calculate mean L2 norm across batch and sequence dimensions
            contribution = torch.norm(head_output, dim=-1).mean().item()
            head_name = f'L{layer_idx}H{head_idx}'
            head_contributions[head_name] = contribution
    
    return head_contributions

def plot_head_contributions(head_contributions, perturbation, output_dir):
    if not head_contributions:
        logger.warning(f"No head contributions to plot for {perturbation}")
        return
        
    plt.figure(figsize=(15, 8))
    heads = list(head_contributions.keys())
    contributions = list(head_contributions.values())
    
    # Use viridis color map for all heads
    plt.bar(range(len(heads)), contributions, color='viridis')
    plt.title(f'Average Attention Head Magnitudes Across All Perturbations', fontsize=14)
    plt.xlabel('Attention Head', fontsize=12)
    plt.ylabel('Average Activation Magnitude', fontsize=12)
    plt.xticks(range(len(heads)), heads, rotation=45, fontsize=8)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    head_analysis_dir = os.path.join(output_dir, 'head_analysis')
    os.makedirs(head_analysis_dir, exist_ok=True)
    plt.savefig(os.path.join(head_analysis_dir, f'{perturbation}_head_contributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Register hooks on the model for attention heads
register_head_hooks(model)

# Head analysis for each perturbation
all_head_contributions = {}
figures_dir = os.path.join(os.path.dirname(__file__), "figures")
head_analysis_dir = os.path.join(figures_dir, "head_analysis")
os.makedirs(head_analysis_dir, exist_ok=True)

# Try to determine number of attention heads from model architecture
num_heads_per_layer = 8  # Default value, will try to detect automatically
if hasattr(model, 'transformer_backbone'):
    if hasattr(model.transformer_backbone, 'config') and hasattr(model.transformer_backbone.config, 'num_attention_heads'):
        num_heads_per_layer = model.transformer_backbone.config.num_attention_heads
        logger.info(f"Detected {num_heads_per_layer} attention heads per layer from model config")
    elif hasattr(model.transformer_backbone, 'h') and len(model.transformer_backbone.h) > 0:
        first_layer = model.transformer_backbone.h[0]
        # Try to get number of heads from the first layer
        for attr_name in ['attn', 'self_attn', 'attention']:
            if hasattr(first_layer, attr_name):
                attn_module = getattr(first_layer, attr_name)
                if hasattr(attn_module, 'num_heads'):
                    num_heads_per_layer = attn_module.num_heads
                    logger.info(f"Detected {num_heads_per_layer} attention heads per layer")
                    break

# Helper function for direct model inference
def run_model_inference(cells_batch, perturbation_name, cell_sentence_len=64):
    """Run direct model inference on a batch of cells"""
    device = next(model.parameters()).device
    
    # Extract embeddings
    if embed_key in cells_batch.obsm:
        X_embed = torch.tensor(cells_batch.obsm[embed_key], dtype=torch.float32).to(device)
    else:
        X_embed = torch.tensor(cells_batch.X.toarray() if hasattr(cells_batch.X, 'toarray') else cells_batch.X, 
                             dtype=torch.float32).to(device)
    
    # Create perturbation tensor
    pert_dim = model.pert_dim
    n_cells = X_embed.shape[0]
    
    # Create one-hot tensor - for now use control (index 0)
    pert_tensor = torch.zeros((n_cells, pert_dim), device=device)
    pert_tensor[:, 0] = 1  # Set first dimension to 1 for control perturbation
    pert_names = [perturbation_name] * n_cells
    
    # Create batch dictionary
    batch = {
        "ctrl_cell_emb": X_embed,
        "pert_emb": pert_tensor,
        "pert_name": pert_names,
        "batch": torch.zeros((1, cell_sentence_len), device=device)
    }
    
    # Run model forward
    with torch.no_grad():
        output = model.forward(batch, padded=False)
    
    # Create result AnnData - use the predicted embeddings
    if hasattr(output, 'pred_cell_emb'):
        pred_embeddings = output.pred_cell_emb.cpu().numpy()
    elif hasattr(output, 'logits'):
        pred_embeddings = output.logits.cpu().numpy()
    else:
        # Fallback - just use the input embeddings
        pred_embeddings = X_embed.cpu().numpy()
    
    # Create new AnnData with predictions
    pred_adata = ad.AnnData(X=pred_embeddings)
    pred_adata.obs = cells_batch.obs.copy()
    pred_adata.var_names = [f"gene_{i}" for i in range(pred_embeddings.shape[1])]
    
    # Store prediction embeddings
    pred_adata.obsm[f"{embed_key}_pert"] = pred_embeddings
    
    return pred_adata

# 2. Generate predictions for each perturbation
all_predictions = []
np.random.seed(42)  # For reproducibility

# Fixed cell sentence length to match model training
CELL_SENTENCE_LEN = 64  # Use the same as in replogle_split

# Create progress bar for all perturbations
for perturbation in pert_counts.index:
    # Get number of cells for this perturbation
    n_cells = pert_counts[perturbation]
    # Limit to model's maximum sequence length
    n_cells_to_use = min(n_cells, CELL_SENTENCE_LEN)
    logger.info(f"Generating predictions for {perturbation} with {n_cells_to_use} cells (original: {n_cells})")

    # Sample this many control cells (with replacement if needed)
    if n_cells_to_use <= all_ctrl_cells.n_obs:
        # Sample without replacement if we have enough cells
        sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells_to_use, replace=False)
    else:
        # Sample with replacement if we need more cells than available
        sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells_to_use, replace=True)

    ctrl_subset = all_ctrl_cells[sample_indices].copy()

    # Set the perturbation
    ctrl_subset.obs["gene"] = perturbation

    # Clear activations before inference
    head_activations.clear()

    # Run inference
    try:
        pred_subset = run_model_inference(ctrl_subset, perturbation, CELL_SENTENCE_LEN)

        # Immediately after inference, copy activations
        current_activations = head_activations.copy()

        # Add metadata to track source perturbation
        pred_subset.obs["original_perturbation"] = perturbation
        pred_subset.obs["is_predicted"] = True
        # Store this batch of predictions
        all_predictions.append(pred_subset)

        logger.info(f"Generated {pred_subset.n_obs} predictions for {perturbation}")

        # --- BEGIN HEAD CONTRIBUTION ANALYSIS FOR THIS PERTURBATION ---
        head_contributions = calculate_head_contributions(current_activations, num_heads_per_layer)
        
        if head_contributions:
            all_head_contributions[perturbation] = head_contributions
            # plot_head_contributions(head_contributions, perturbation, figures_dir)
            logger.info(f"Calculated head contributions for {perturbation}: {len(head_contributions)} heads")
        else:
            logger.info(f"No head activations found for {perturbation}")
        # --- END HEAD CONTRIBUTION ANALYSIS ---
    except Exception as e:
        logger.info(f"Error generating predictions for {perturbation}: {str(e)}")

# --- Combine and average head contributions across all perturbations ---
if all_head_contributions:
    # Get all head names (assume all perturbations have the same heads)
    head_names = list(next(iter(all_head_contributions.values())).keys())
    # Collect values for each head
    head_values = {head: [] for head in head_names}
    for pert, contribs in all_head_contributions.items():
        for head in head_names:
            head_values[head].append(contribs.get(head, np.nan))
    # Compute mean for each head
    avg_head_contributions = {head: np.nanmean(vals) for head, vals in head_values.items()}
    logger.info("Average head contributions across all perturbations:")
    for head, avg in avg_head_contributions.items():
        logger.info(f"{head}: {avg:.4f}")
    # Plot average contributions
    plot_head_contributions(avg_head_contributions, "average_across_perturbations", figures_dir)
else:
    logger.info("No head contributions to average.")