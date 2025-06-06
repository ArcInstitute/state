#!/usr/bin/env python
"""
calculate_layer3_attention.py – v5
=================================
Updated to analyze attention contribution patterns across multiple samples.
Calculates the direction of contribution from top 32 vs bottom 32 query matrices
for attention heads 1 and 9, and plots distributions with confidence intervals.
"""

from __future__ import annotations

import os
import sys
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict
import statistics

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))

# Import model class directly
from models.pertsets import PertSetsPerturbationModel

MODEL_DIR = Path(
    # "/large_storage/ctc/userspace/aadduri/preprint/replogle_vci_1.5.2_cs64/fold1"
    "/large_storage/ctc/userspace/rohankshah/preprint/replogle_gpt_31043724/hepg2"
)
DATA_PATH = Path(
    "/large_storage/ctc/ML/state_sets/replogle/processed.h5"
)
CELL_SET_LEN = 512   
CONTROL_SAMPLES = 50 
LAYER_IDX = 7
NUM_SAMPLES = 100  # Number of forward passes to average over
TARGET_HEADS = [1, 9]  # Focus on heads 1 and 9
FIG_DIR = Path(__file__).resolve().parent / "figures" / "replogle_split_avg" 
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

# Configure attention outputs
model.transformer_backbone.config._attn_implementation = "sdpa"
model.transformer_backbone._attn_implementation        = "sdpa"
model.transformer_backbone.config.output_attentions = True

if hasattr(model, 'config'):
    model.config.output_attentions = True

# Updated for GPT-2 architecture using .h instead of .layers
for layer in model.transformer_backbone.h:
    if hasattr(layer.attn, 'output_attentions'):
        layer.attn.output_attentions = True

print(f"Model config output_attentions: {getattr(model.transformer_backbone.config, 'output_attentions', 'Not found')}")
print(f"Number of transformer layers: {len(model.transformer_backbone.h)}")
print(f"Target layer index: {LAYER_IDX}")

logger.info(
    "Loaded PertSets model with GPT-2 backbone: %s heads × %s layers",
    model.transformer_backbone.config.num_attention_heads,
    model.transformer_backbone.config.num_hidden_layers,
)

NUM_HEADS = model.transformer_backbone.config.num_attention_heads

# Storage for attention contributions across samples
attention_contributions = {head: {'top_half': [], 'bottom_half': [], 'difference': []} for head in TARGET_HEADS}

# Load data directly
adata_full = sc.read_h5ad(str(DATA_PATH))
logger.info("Using full dataset shape: %s", adata_full.shape)

# Determine embedding key
embed_key = "X_vci_1.5.2_4"  
if embed_key not in adata_full.obsm:
    embed_key = "X_hvg"  
    if embed_key not in adata_full.obsm:
        embed_key = "X" 
        logger.warning("Using raw expression data - model may not work properly")
logger.info(f"Using embedding key: {embed_key}")

# Find control perturbation
control_pert = "DMSO_TF_24h"  
if "gene" in adata_full.obs.columns:
    drugname_counts = adata_full.obs["gene"].value_counts()
    for potential_control in ["DMSO_TF_24h", "non-targeting", "control", "DMSO"]:
        if potential_control in drugname_counts.index:
            control_pert = potential_control
            break
    else:
        control_pert = drugname_counts.index[0]
        logger.warning(f"Could not find standard control, using: {control_pert}")

logger.info("Control perturbation: %s", control_pert)

# Get available cell types
cell_type_counts = adata_full.obs["cell_type"].value_counts()
celltype1, celltype2 = cell_type_counts.index[:2]
logger.info(f"Selected cell types: {celltype1}, {celltype2}")

# Get control cells for each cell type
cells_type1 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype1)].copy()
cells_type2 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype2)].copy()

logger.info(f"Available cells - {celltype1}: {cells_type1.n_obs}, {celltype2}: {cells_type2.n_obs}")

def analyze_attention_sample(attn_weights):
    """
    Analyze attention weights for one sample.
    Calculate and compare contributions from top 32 vs bottom 32 queries.
    
    Returns:
        contributions: dict with head_idx -> {'top_half', 'bottom_half', 'difference'}
    """
    contributions = {}
    
    # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attn_weights.shape
    
    for head_idx in TARGET_HEADS:
        if head_idx >= num_heads:
            continue
            
        # Average across batch dimension
        head_attn = attn_weights[:, head_idx].mean(dim=0)  # [seq_len, seq_len]
        
        # Split queries into top and bottom halves
        mid_point = seq_len // 2
        
        # Calculate mean attention from top half queries (positions 0-31) to all keys
        top_half_contribution = head_attn[:mid_point].mean().item()
        
        # Calculate mean attention from bottom half queries (positions 32-63) to all keys  
        bottom_half_contribution = head_attn[mid_point:].mean().item()
        
        # Calculate the difference (positive = top half has higher attention)
        difference = top_half_contribution - bottom_half_contribution
        
        contributions[head_idx] = {
            'top_half': top_half_contribution,
            'bottom_half': bottom_half_contribution,
            'difference': difference
        }
    
    return contributions

def layer3_hook(_: torch.nn.Module, __, outputs: Tuple[torch.Tensor, torch.Tensor]):
    """Collect and analyze attention matrices."""
    global current_sample_contributions
    
    attn_weights = None
    for i, output in enumerate(outputs):
        if hasattr(output, 'shape') and len(output.shape) == 4:
            B, H, S1, S2 = output.shape
            if S1 == S2:  # Square matrix indicates attention weights
                attn_weights = output.detach().cpu()
                break
    
    if attn_weights is None:
        return
    
    # Analyze this sample's attention
    current_sample_contributions = analyze_attention_sample(attn_weights)

# Run multiple samples
logger.info(f"Running {NUM_SAMPLES} samples for statistical analysis...")

for sample_idx in range(NUM_SAMPLES):
    if sample_idx % 10 == 0:
        logger.info(f"Processing sample {sample_idx + 1}/{NUM_SAMPLES}")
    
    # Sample cells for this iteration
    n_cells_per_type = 64
    
    if cells_type1.n_obs >= n_cells_per_type and cells_type2.n_obs >= n_cells_per_type:
        # Sample cells randomly for each iteration
        idx1 = np.random.choice(cells_type1.n_obs, size=n_cells_per_type, replace=False)
        sampled_type1 = cells_type1[idx1].copy()
        
        # Get embeddings and prepare batch
        device = next(model.parameters()).device
        
        if embed_key in sampled_type1.obsm:
            X_embed = torch.tensor(sampled_type1.obsm[embed_key], dtype=torch.float32).to(device)
        else:
            X_embed = torch.tensor(sampled_type1.X.toarray() if hasattr(sampled_type1.X, 'toarray') else sampled_type1.X, 
                                 dtype=torch.float32).to(device)
        
        # Create dummy perturbation tensor
        pert_dim = model.pert_dim
        n_cells = X_embed.shape[0]
        pert_tensor = torch.zeros((n_cells, pert_dim), device=device)
        pert_tensor[:, 0] = 1  # Control perturbation
        pert_names = [control_pert] * n_cells
        
        batch = {
            "ctrl_cell_emb": X_embed,
            "pert_emb": pert_tensor,
            "pert_name": pert_names,
            "batch": torch.zeros((1, 64), device=device)
        }
        
        # Reset sample contributions
        current_sample_contributions = {}
        
        # Register hook for this sample
        hook_handle = model.transformer_backbone.h[LAYER_IDX].attn.register_forward_hook(layer3_hook)
        
        # Forward pass
        with torch.no_grad():
            batch_pred = model.forward(batch, padded=False)
        
        # Remove hook
        hook_handle.remove()
        
        # Store contributions from this sample
        for head_idx in TARGET_HEADS:
            if head_idx in current_sample_contributions:
                attention_contributions[head_idx]['top_half'].append(
                    current_sample_contributions[head_idx]['top_half']
                )
                attention_contributions[head_idx]['bottom_half'].append(
                    current_sample_contributions[head_idx]['bottom_half']
                )
                attention_contributions[head_idx]['difference'].append(
                    current_sample_contributions[head_idx]['difference']
                )

logger.info("Completed all samples. Creating visualizations...")

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for plot_idx, head_idx in enumerate(TARGET_HEADS):
    # Plot 1: Side-by-side comparison with confidence intervals
    ax1 = axes[0, plot_idx]
    
    top_contributions = attention_contributions[head_idx]['top_half']
    bottom_contributions = attention_contributions[head_idx]['bottom_half']
    differences = attention_contributions[head_idx]['difference']
    
    if len(top_contributions) > 0 and len(bottom_contributions) > 0:
        # Calculate statistics
        top_mean = np.mean(top_contributions)
        bottom_mean = np.mean(bottom_contributions)
        diff_mean = np.mean(differences)
        
        top_sem = stats.sem(top_contributions)
        bottom_sem = stats.sem(bottom_contributions)
        diff_sem = stats.sem(differences)
        
        # 95% confidence intervals
        top_ci = stats.t.interval(0.95, len(top_contributions)-1, loc=top_mean, scale=top_sem)
        bottom_ci = stats.t.interval(0.95, len(bottom_contributions)-1, loc=bottom_mean, scale=bottom_sem)
        
        # Create bar plot with error bars
        categories = ['Top 32\nQueries', 'Bottom 32\nQueries']
        means = [top_mean, bottom_mean]
        errors = [top_mean - top_ci[0], bottom_mean - bottom_ci[0]]
        
        bars = ax1.bar(categories, means, yerr=errors, capsize=5, alpha=0.7, 
                      color=['darkblue', 'darkred'])
        
        # Add individual data points
        ax1.scatter(['Top 32\nQueries'] * len(top_contributions), top_contributions, 
                   alpha=0.3, s=15, color='lightblue', label='Individual samples')
        ax1.scatter(['Bottom 32\nQueries'] * len(bottom_contributions), bottom_contributions, 
                   alpha=0.3, s=15, color='lightcoral')
        
        ax1.set_ylabel('Mean Attention Weight')
        ax1.set_title(f'Head {head_idx}: Top vs Bottom Query Attention\n'
                     f'Mean difference: {diff_mean:.5f} ± {diff_sem:.5f}')
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(top_contributions, bottom_contributions)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(top_contributions) + np.var(bottom_contributions)) / 2)
        cohens_d = (top_mean - bottom_mean) / pooled_std
        
        # Add statistical annotation with significance indicator
        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
            
        ax1.text(0.5, 0.95, f'p = {p_value:.4f} {significance}\nCohen\'s d = {cohens_d:.3f}', 
                transform=ax1.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Distribution of differences (top - bottom)
    ax2 = axes[1, plot_idx]
    
    if len(differences) > 0:
        # Histogram of differences
        ax2.hist(differences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax2.axvline(diff_mean, color='orange', linewidth=2, label=f'Mean = {diff_mean:.5f}')
        
        ax2.set_xlabel('Difference (Top - Bottom)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Head {head_idx}: Distribution of Differences\n'
                     f'(n={len(differences)} samples)')
        ax2.legend()
        
        # One-sample t-test against zero (testing if difference is significantly different from 0)
        t_stat_diff, p_value_diff = stats.ttest_1samp(differences, 0)
        
        ax2.text(0.02, 0.95, f'One-sample t-test vs 0:\np = {p_value_diff:.4f}', 
                transform=ax2.transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        logger.info(f"Head {head_idx} - Top: {top_mean:.6f} ± {top_sem:.6f}")
        logger.info(f"Head {head_idx} - Bottom: {bottom_mean:.6f} ± {bottom_sem:.6f}")
        logger.info(f"Head {head_idx} - Difference: {diff_mean:.6f} ± {diff_sem:.6f}")
        logger.info(f"Head {head_idx} - Two-sample t-test p-value: {p_value:.6f}")
        logger.info(f"Head {head_idx} - One-sample t-test (diff vs 0) p-value: {p_value_diff:.6f}")
        logger.info(f"Head {head_idx} - Cohen's d: {cohens_d:.4f}")

plt.suptitle(f'Layer {LAYER_IDX} Attention Analysis: Top vs Bottom Query Contributions\n'
            f'Statistical Comparison Across {NUM_SAMPLES} Samples', fontsize=16)
plt.tight_layout()

# Save the comprehensive plot
fig_path = FIG_DIR / f"layer{LAYER_IDX}_attention_comprehensive_comparison.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

logger.info(f"Saved comprehensive attention comparison → {fig_path}")

# Create additional violin plot for better distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for plot_idx, head_idx in enumerate(TARGET_HEADS):
    ax = axes[plot_idx]
    
    top_contributions = attention_contributions[head_idx]['top_half']
    bottom_contributions = attention_contributions[head_idx]['bottom_half']
    
    if len(top_contributions) > 0 and len(bottom_contributions) > 0:
        # Prepare data for violin plot
        data_for_violin = [top_contributions, bottom_contributions]
        positions = [1, 2]
        
        # Create violin plot
        parts = ax.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)
        
        # Customize violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Top 32 Queries', 'Bottom 32 Queries'])
        ax.set_ylabel('Attention Weight Distribution')
        ax.set_title(f'Head {head_idx} - Distribution Comparison')
        
        # Add statistical annotation
        t_stat, p_value = stats.ttest_ind(top_contributions, bottom_contributions)
        ax.text(0.5, 0.95, f'p = {p_value:.4f}', transform=ax.transAxes, 
               ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle(f'Layer {LAYER_IDX} Attention Weight Distributions\n'
            f'Query Position Comparison', fontsize=14)
plt.tight_layout()

# Save violin plot
fig_path_violin = FIG_DIR / f"layer{LAYER_IDX}_attention_distributions_violin.png"
plt.savefig(fig_path_violin, dpi=300, bbox_inches='tight')
plt.close()

logger.info(f"Saved attention distributions violin plot → {fig_path_violin}")
logger.info("Analysis completed successfully.")

# Print summary statistics
print("\n" + "="*60)
print("STATISTICAL COMPARISON: TOP vs BOTTOM QUERY CONTRIBUTIONS")
print("="*60)

for head_idx in TARGET_HEADS:
    top_data = attention_contributions[head_idx]['top_half']
    bottom_data = attention_contributions[head_idx]['bottom_half']
    diff_data = attention_contributions[head_idx]['difference']
    
    if len(top_data) > 0 and len(bottom_data) > 0:
        # Basic statistics
        top_mean, top_std = np.mean(top_data), np.std(top_data)
        bottom_mean, bottom_std = np.mean(bottom_data), np.std(bottom_data)
        diff_mean, diff_std = np.mean(diff_data), np.std(diff_data)
        
        # Statistical tests
        t_stat_paired, p_value_paired = stats.ttest_ind(top_data, bottom_data)
        t_stat_diff, p_value_diff = stats.ttest_1samp(diff_data, 0)
        
        # Effect size
        pooled_std = np.sqrt((np.var(top_data) + np.var(bottom_data)) / 2)
        cohens_d = (top_mean - bottom_mean) / pooled_std
        
        # Confidence intervals for difference
        diff_sem = stats.sem(diff_data)
        diff_ci = stats.t.interval(0.95, len(diff_data)-1, loc=diff_mean, scale=diff_sem)
        
        print(f"\nATTENTION HEAD {head_idx}:")
        print(f"  Top 32 queries:     {top_mean:.6f} ± {top_std:.6f}")
        print(f"  Bottom 32 queries:  {bottom_mean:.6f} ± {bottom_std:.6f}")
        print(f"  Difference (T-B):   {diff_mean:.6f} ± {diff_std:.6f}")
        print(f"  95% CI for diff:    [{diff_ci[0]:.6f}, {diff_ci[1]:.6f}]")
        print(f"  \nStatistical Tests:")
        print(f"    Two-sample t-test:  t = {t_stat_paired:.4f}, p = {p_value_paired:.6f}")
        print(f"    One-sample t-test:  t = {t_stat_diff:.4f}, p = {p_value_diff:.6f}")
        print(f"    Effect size (d):    {cohens_d:.4f}")
        
        # Interpretation
        if p_value_diff < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value_diff < 0.01:
            significance = "significant (p < 0.01)"
        elif p_value_diff < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p ≥ 0.05)"
            
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
            
        direction = "higher" if diff_mean > 0 else "lower"
        
        print(f"  \nInterpretation:")
        print(f"    Top half queries have {direction} attention than bottom half")
        print(f"    Difference is {significance}")
        print(f"    Effect size is {effect}")

print("\n" + "="*60)
