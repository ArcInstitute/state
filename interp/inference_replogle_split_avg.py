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
NUM_SAMPLES = 10  # Number of forward passes to average over
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

# Storage for QK similarity metrics across samples
qk_metrics = {
    1: {'pattern_correlation': [], 'pattern_cosine_similarity': [], 'top_mean_qk': [], 'bottom_mean_qk': [],
        'top_queries_avg_qk': [], 'bottom_queries_avg_qk': []},
    9: {'top_to_top': [], 'top_to_bottom': [], 'bottom_to_top': [], 'bottom_to_bottom': [],
        'top_queries_preference': [], 'bottom_queries_preference': [], 
        'positional_preference': [], 'diagonal_preference': [],
        'top_queries_avg_qk': [], 'bottom_queries_avg_qk': []}
}

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

def analyze_qk_similarity_sample(qk_scores):
    """
    Analyze QK similarity scores for one sample.
    Calculate metrics to capture different attention patterns:
    - Head 1: Similarity between top and bottom query attention patterns (vertical line)
    - Head 9: Positional preference (top queries -> top keys, bottom queries -> bottom keys)
    
    Returns:
        metrics: dict with head_idx -> pattern-specific metrics
    """
    metrics = {}
    
    # qk_scores shape: [batch_size, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = qk_scores.shape
    
    for head_idx in TARGET_HEADS:
        if head_idx >= num_heads:
            continue
            
        # Average across batch dimension
        head_qk = qk_scores[:, head_idx].mean(dim=0)  # [seq_len, seq_len]
        
        # Split queries and keys into top and bottom halves
        mid_point = seq_len // 2
        
        top_queries = head_qk[:mid_point]      # [32, 64] - top 32 queries to all keys
        bottom_queries = head_qk[mid_point:]   # [32, 64] - bottom 32 queries to all keys
        
        top_keys = head_qk[:, :mid_point]      # [64, 32] - all queries to top 32 keys
        bottom_keys = head_qk[:, mid_point:]   # [64, 32] - all queries to bottom 32 keys
        
        if head_idx == 1:
            # Head 1 metric: Correlation between top and bottom query patterns
            # High correlation = both query groups attend to same keys (vertical line pattern)
            top_pattern = top_queries.mean(dim=0)    # [64] - average attention from top queries to each key
            bottom_pattern = bottom_queries.mean(dim=0)  # [64] - average attention from bottom queries to each key
            
            # Calculate Pearson correlation between the two patterns
            correlation = torch.corrcoef(torch.stack([top_pattern, bottom_pattern]))[0, 1].item()
            
            # Also calculate cosine similarity as alternative metric
            cosine_sim = torch.nn.functional.cosine_similarity(
                top_pattern.unsqueeze(0), bottom_pattern.unsqueeze(0)
            ).item()
            
            # For both heads: measure the average key position where each query attends most
            # This will show if query groups attend to similar vs different key positions
            top_argmax_positions = top_queries.argmax(dim=1).float()  # [32] - key position each top query attends to most
            bottom_argmax_positions = bottom_queries.argmax(dim=1).float()  # [32] - key position each bottom query attends to most
            
            # Average preferred key position for each query group
            top_avg_key_pos = top_argmax_positions.mean().item()
            bottom_avg_key_pos = bottom_argmax_positions.mean().item()
            
            metrics[head_idx] = {
                'pattern_correlation': correlation,
                'pattern_cosine_similarity': cosine_sim,
                'top_mean_qk': top_pattern.mean().item(),
                'bottom_mean_qk': bottom_pattern.mean().item(),
                'top_queries_avg_qk': top_avg_key_pos,    # Average key position where top queries attend most
                'bottom_queries_avg_qk': bottom_avg_key_pos  # Average key position where bottom queries attend most
            }
            
        elif head_idx == 9:
            # Head 9 metric: Positional preference strength
            # Top queries preferring top keys vs bottom keys
            top_to_top = top_queries[:, :mid_point].mean().item()    # top queries -> top keys
            top_to_bottom = top_queries[:, mid_point:].mean().item() # top queries -> bottom keys
            
            # Bottom queries preferring top keys vs bottom keys  
            bottom_to_top = bottom_queries[:, :mid_point].mean().item()    # bottom queries -> top keys
            bottom_to_bottom = bottom_queries[:, mid_point:].mean().item() # bottom queries -> bottom keys
            
            # Calculate preference scores
            top_queries_preference = top_to_top - top_to_bottom    # positive = prefer top keys
            bottom_queries_preference = bottom_to_bottom - bottom_to_top  # positive = prefer bottom keys
            
            # Overall positional preference (how much the pattern matches expected)
            positional_preference = (top_queries_preference + bottom_queries_preference) / 2
            
            # Cross-diagonal vs main-diagonal preference
            main_diagonal = (top_to_top + bottom_to_bottom) / 2
            cross_diagonal = (top_to_bottom + bottom_to_top) / 2
            diagonal_preference = main_diagonal - cross_diagonal
            
            # For both heads: use the same metric as Head 1 - average key position where each query attends most
            top_argmax_positions = top_queries.argmax(dim=1).float()  # [32] - key position each top query attends to most
            bottom_argmax_positions = bottom_queries.argmax(dim=1).float()  # [32] - key position each bottom query attends to most
            
            # Average preferred key position for each query group
            top_avg_key_pos = top_argmax_positions.mean().item()
            bottom_avg_key_pos = bottom_argmax_positions.mean().item()
            
            metrics[head_idx] = {
                'top_to_top': top_to_top,
                'top_to_bottom': top_to_bottom, 
                'bottom_to_top': bottom_to_top,
                'bottom_to_bottom': bottom_to_bottom,
                'top_queries_preference': top_queries_preference,
                'bottom_queries_preference': bottom_queries_preference,
                'positional_preference': positional_preference,
                'diagonal_preference': diagonal_preference,
                'top_queries_avg_qk': top_avg_key_pos,    # Average key position where top queries attend most
                'bottom_queries_avg_qk': bottom_avg_key_pos  # Average key position where bottom queries attend most
            }
    
    return metrics

def qk_scores_hook(module, input, output):
    """Extract QK scores before softmax from attention mechanism."""
    global current_sample_metrics
    # For GPT-2 style attention, we need to hook into the attention computation
    # The QK scores are computed inside the attention forward pass
    pass

def attention_forward_hook(module, input, output):
    """Hook to capture attention weights and compute QK scores."""
    global current_sample_metrics
    
    # Get the input to attention (hidden states)
    hidden_states = input[0] if isinstance(input, tuple) else input
    
    # Manually compute Q, K matrices to get QK scores
    batch_size, seq_len, embed_dim = hidden_states.shape
    head_dim = embed_dim // module.num_heads
    
    # Get Q, K, V from the combined projection
    qkv = module.c_attn(hidden_states)  # [batch, seq, 3*embed_dim]
    qkv = qkv.view(batch_size, seq_len, 3, module.num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
    
    query, key, value = qkv[0], qkv[1], qkv[2]
    
    # Compute QK scores (before scaling and softmax)
    qk_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch, heads, seq, seq]
    
    # Apply scaling (typically sqrt(head_dim))
    qk_scores = qk_scores / math.sqrt(head_dim)
    
    # Analyze the QK scores
    current_sample_metrics = analyze_qk_similarity_sample(qk_scores.detach().cpu())

# Global variable to store current sample metrics
current_sample_metrics = {}

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
        
        # Reset sample metrics
        current_sample_metrics = {}
        
        # Register hook for this sample
        hook_handle = model.transformer_backbone.h[LAYER_IDX].attn.register_forward_hook(attention_forward_hook)
        
        # Forward pass
        with torch.no_grad():
            batch_pred = model.forward(batch, padded=False)
        
        # Remove hook
        hook_handle.remove()
        
        # Store metrics from this sample
        for head_idx in TARGET_HEADS:
            if head_idx in current_sample_metrics:
                for metric_name, metric_value in current_sample_metrics[head_idx].items():
                    qk_metrics[head_idx][metric_name].append(metric_value)

logger.info("Completed all samples. Creating visualizations...")

# Create comprehensive QK similarity analysis plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Head 1: Pattern similarity metrics (vertical line pattern)
if 1 in qk_metrics and len(qk_metrics[1]['pattern_correlation']) > 0:
    ax1 = axes[0, 0]
    
    correlations = qk_metrics[1]['pattern_correlation']
    cosine_sims = qk_metrics[1]['pattern_cosine_similarity']
    
    # Distribution of pattern correlations
    ax1.hist(correlations, bins=20, alpha=0.7, color='blue', edgecolor='black', label='Correlations')
    corr_mean = np.mean(correlations)
    ax1.axvline(corr_mean, color='red', linewidth=2, label=f'Mean = {corr_mean:.3f}')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Pattern Correlation')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Head 1: Query Pattern Similarity\n'
                 f'Correlation between Top & Bottom Query Patterns\n'
                 f'(n={len(correlations)} samples)')
    ax1.legend()
    
    # Statistical test against zero correlation
    t_stat, p_value = stats.ttest_1samp(correlations, 0)
    ax1.text(0.02, 0.95, f'One-sample t-test vs 0:\np = {p_value:.4f}', 
            transform=ax1.transAxes, va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Head 1: Cosine similarity
if 1 in qk_metrics and len(qk_metrics[1]['pattern_cosine_similarity']) > 0:
    ax2 = axes[1, 0]
    
    cosine_sims = qk_metrics[1]['pattern_cosine_similarity']
    
    ax2.hist(cosine_sims, bins=20, alpha=0.7, color='green', edgecolor='black')
    cosine_mean = np.mean(cosine_sims)
    ax2.axvline(cosine_mean, color='red', linewidth=2, label=f'Mean = {cosine_mean:.3f}')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Head 1: Query Pattern Cosine Similarity\n'
                 f'(n={len(cosine_sims)} samples)')
    ax2.legend()
    
    # Statistical test
    t_stat, p_value = stats.ttest_1samp(cosine_sims, 0)
    ax2.text(0.02, 0.95, f'One-sample t-test vs 0:\np = {p_value:.4f}', 
            transform=ax2.transAxes, va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Head 9: Positional preference
if 9 in qk_metrics and len(qk_metrics[9]['positional_preference']) > 0:
    ax3 = axes[0, 1]
    
    pos_prefs = qk_metrics[9]['positional_preference']
    
    ax3.hist(pos_prefs, bins=20, alpha=0.7, color='purple', edgecolor='black')
    pref_mean = np.mean(pos_prefs)
    ax3.axvline(pref_mean, color='red', linewidth=2, label=f'Mean = {pref_mean:.4f}')
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5, label='No preference')
    
    ax3.set_xlabel('Positional Preference Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Head 9: Positional Preference\n'
                 f'Top→Top & Bottom→Bottom preference\n'
                 f'(n={len(pos_prefs)} samples)')
    ax3.legend()
    
    # Statistical test
    t_stat, p_value = stats.ttest_1samp(pos_prefs, 0)
    ax3.text(0.02, 0.95, f'One-sample t-test vs 0:\np = {p_value:.4f}', 
            transform=ax3.transAxes, va='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Head 9: Detailed positional breakdown
if 9 in qk_metrics and len(qk_metrics[9]['top_to_top']) > 0:
    ax4 = axes[1, 1]
    
    top_to_top = qk_metrics[9]['top_to_top']
    top_to_bottom = qk_metrics[9]['top_to_bottom'] 
    bottom_to_top = qk_metrics[9]['bottom_to_top']
    bottom_to_bottom = qk_metrics[9]['bottom_to_bottom']
    
    # Box plot of all four combinations
    data_for_box = [top_to_top, top_to_bottom, bottom_to_top, bottom_to_bottom]
    positions = [1, 2, 3, 4]
    labels = ['Top→Top', 'Top→Bottom', 'Bottom→Top', 'Bottom→Bottom']
    
    bp = ax4.boxplot(data_for_box, positions=positions, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('QK Similarity Score')
    ax4.set_title(f'Head 9: Detailed Positional Analysis\n'
                 f'Query-Key Position Combinations')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add means as text
    means = [np.mean(data) for data in data_for_box]
    for i, (pos, mean) in enumerate(zip(positions, means)):
        ax4.text(pos, mean, f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle(f'Layer {LAYER_IDX} QK Similarity Analysis: Pattern-Specific Metrics\n'
            f'Head 1: Vertical Line Pattern | Head 9: Positional Preference Pattern', fontsize=14)
plt.tight_layout()

# Save the comprehensive plot
fig_path = FIG_DIR / f"layer{LAYER_IDX}_qk_similarity_analysis.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

logger.info(f"Saved QK similarity analysis → {fig_path}")

# Create violin plots showing QK distributions: Top vs Bottom queries
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Collect all data to determine common y-axis scale
all_qk_scores = []
for head_idx in TARGET_HEADS:
    if head_idx in qk_metrics and len(qk_metrics[head_idx]['top_queries_avg_qk']) > 0:
        all_qk_scores.extend(qk_metrics[head_idx]['top_queries_avg_qk'])
        all_qk_scores.extend(qk_metrics[head_idx]['bottom_queries_avg_qk'])

# Set common y-axis limits
if all_qk_scores:
    y_min = min(all_qk_scores) - 0.1 * (max(all_qk_scores) - min(all_qk_scores))
    y_max = max(all_qk_scores) + 0.1 * (max(all_qk_scores) - min(all_qk_scores))
else:
    y_min, y_max = -1, 1

for plot_idx, head_idx in enumerate(TARGET_HEADS):
    ax = axes[plot_idx]
    
    if head_idx in qk_metrics and len(qk_metrics[head_idx]['top_queries_avg_qk']) > 0:
        top_qk_scores = qk_metrics[head_idx]['top_queries_avg_qk']
        bottom_qk_scores = qk_metrics[head_idx]['bottom_queries_avg_qk']
        
        # Prepare data for violin plot
        data_for_violin = [top_qk_scores, bottom_qk_scores]
        positions = [1, 2]
        
        # Create violin plot
        parts = ax.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        if head_idx == 1:
            # Head 1: Similar colors (should show similar distributions for vertical line pattern)
            colors = ['lightblue', 'lightcyan']
        else:
            # Head 9: Different colors (should show different distributions for positional pattern)
            colors = ['lightblue', 'lightcoral']
            
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Top 32 Queries', 'Bottom 32 Queries'])
        
        ax.set_ylabel('Average Preferred Key Position')
        
        # Set common y-axis limits
        ax.set_ylim(y_min, y_max)
        
        if head_idx == 1:
            ax.set_title(f'Head {head_idx} - Vertical Line Pattern\n'
                        f'Expected: Both groups attend to similar key positions')
        else:
            ax.set_title(f'Head {head_idx} - Positional Preference Pattern\n'
                        f'Expected: Top→top keys (~16), Bottom→bottom keys (~48)')
        
        # Add statistical annotation
        t_stat, p_value = stats.ttest_ind(top_qk_scores, bottom_qk_scores)
        
        # Calculate means for display
        top_mean = np.mean(top_qk_scores)
        bottom_mean = np.mean(bottom_qk_scores)
        diff = top_mean - bottom_mean
        
        # Significance indicator
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        # Color code the p-value box based on expected behavior
        if head_idx == 1:
            # Head 1: we want non-significant (similar distributions)
            box_color = 'lightgreen' if significance == 'ns' else 'lightyellow'
            expected_text = "✓ Good" if significance == 'ns' else "⚠ Unexpected"
        else:
            # Head 9: we want significant (different distributions) 
            box_color = 'lightgreen' if significance != 'ns' else 'lightyellow'
            expected_text = "✓ Good" if significance != 'ns' else "⚠ Unexpected"
        
        ax.text(0.5, 0.95, f'Difference: {diff:.4f}\np = {p_value:.4f} {significance}\n{expected_text}', 
               transform=ax.transAxes, ha='center', va='top', 
               bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
        
        # Add individual sample points with slight jitter for visibility
        x1_jitter = np.random.normal(1, 0.04, len(top_qk_scores))
        x2_jitter = np.random.normal(2, 0.04, len(bottom_qk_scores))
        ax.scatter(x1_jitter, top_qk_scores, alpha=0.4, s=15, color='darkblue')
        ax.scatter(x2_jitter, bottom_qk_scores, alpha=0.4, s=15, color='darkred')

plt.suptitle(f'Layer {LAYER_IDX}: Where Do Query Groups Attend Most?\n'
            f'Key Position (0-31 = top half, 32-63 = bottom half)', fontsize=14)
plt.tight_layout()

# Save violin plot
fig_path_violin = FIG_DIR / f"layer{LAYER_IDX}_qk_distributions_violin.png"
plt.savefig(fig_path_violin, dpi=300, bbox_inches='tight')
plt.close()

logger.info(f"Saved QK distributions violin plot → {fig_path_violin}")
logger.info("Analysis completed successfully.")

# Print summary statistics
print("\n" + "="*70)
print("QK SIMILARITY ANALYSIS: PATTERN-SPECIFIC METRICS")
print("="*70)

# Head 1 Analysis
if 1 in qk_metrics and len(qk_metrics[1]['pattern_correlation']) > 0:
    print(f"\nHEAD 1 - VERTICAL LINE PATTERN ANALYSIS:")
    print("-" * 50)
    
    correlations = qk_metrics[1]['pattern_correlation']
    cosine_sims = qk_metrics[1]['pattern_cosine_similarity']
    
    corr_mean, corr_std = np.mean(correlations), np.std(correlations)
    cosine_mean, cosine_std = np.mean(cosine_sims), np.std(cosine_sims)
    
    # Statistical tests
    t_stat_corr, p_value_corr = stats.ttest_1samp(correlations, 0)
    t_stat_cosine, p_value_cosine = stats.ttest_1samp(cosine_sims, 0)
    
    # Confidence intervals
    corr_sem = stats.sem(correlations)
    cosine_sem = stats.sem(cosine_sims)
    corr_ci = stats.t.interval(0.95, len(correlations)-1, loc=corr_mean, scale=corr_sem)
    cosine_ci = stats.t.interval(0.95, len(cosine_sims)-1, loc=cosine_mean, scale=cosine_sem)
    
    print(f"  Pattern Correlation:     {corr_mean:.6f} ± {corr_std:.6f}")
    print(f"  95% CI for correlation:  [{corr_ci[0]:.6f}, {corr_ci[1]:.6f}]")
    print(f"  Cosine Similarity:       {cosine_mean:.6f} ± {cosine_std:.6f}")
    print(f"  95% CI for cosine sim:   [{cosine_ci[0]:.6f}, {cosine_ci[1]:.6f}]")
    print(f"  \nStatistical Tests (vs 0):")
    print(f"    Correlation t-test:     t = {t_stat_corr:.4f}, p = {p_value_corr:.6f}")
    print(f"    Cosine sim t-test:      t = {t_stat_cosine:.4f}, p = {p_value_cosine:.6f}")
    
    # Interpretation
    if p_value_corr < 0.001:
        corr_significance = "highly significant (p < 0.001)"
    elif p_value_corr < 0.01:
        corr_significance = "significant (p < 0.01)"
    elif p_value_corr < 0.05:
        corr_significance = "significant (p < 0.05)"
    else:
        corr_significance = "not significant (p ≥ 0.05)"
    
    print(f"  \nInterpretation:")
    print(f"    Pattern correlation is {corr_significance}")
    if corr_mean > 0.5:
        print(f"    Strong evidence for vertical line pattern (both query groups attend to same keys)")
    elif corr_mean > 0.3:
        print(f"    Moderate evidence for vertical line pattern")
    elif corr_mean > 0:
        print(f"    Weak evidence for vertical line pattern")
    else:
        print(f"    No evidence for vertical line pattern")
    
    # Add violin plot comparison
    if 'top_queries_avg_qk' in qk_metrics[1] and len(qk_metrics[1]['top_queries_avg_qk']) > 0:
        top_qk = qk_metrics[1]['top_queries_avg_qk']
        bottom_qk = qk_metrics[1]['bottom_queries_avg_qk']
        
        top_qk_mean, bottom_qk_mean = np.mean(top_qk), np.mean(bottom_qk)
        qk_diff = top_qk_mean - bottom_qk_mean
        t_stat_violin, p_value_violin = stats.ttest_ind(top_qk, bottom_qk)
        
        print(f"  \nPreferred Key Position Analysis (for violin plots):")
        print(f"    Top 32 queries prefer key:    {top_qk_mean:.2f} ± {np.std(top_qk):.2f}")
        print(f"    Bottom 32 queries prefer key: {bottom_qk_mean:.2f} ± {np.std(bottom_qk):.2f}")
        print(f"    Difference (T-B):             {qk_diff:.2f}")
        print(f"    Two-sample t-test:            t = {t_stat_violin:.4f}, p = {p_value_violin:.6f}")
        print(f"    Expected: Small difference (both groups attend to similar key positions)")

# Head 9 Analysis
if 9 in qk_metrics and len(qk_metrics[9]['positional_preference']) > 0:
    print(f"\nHEAD 9 - POSITIONAL PREFERENCE ANALYSIS:")
    print("-" * 50)
    
    pos_prefs = qk_metrics[9]['positional_preference']
    diag_prefs = qk_metrics[9]['diagonal_preference']
    top_to_top = qk_metrics[9]['top_to_top']
    top_to_bottom = qk_metrics[9]['top_to_bottom']
    bottom_to_top = qk_metrics[9]['bottom_to_top']
    bottom_to_bottom = qk_metrics[9]['bottom_to_bottom']
    
    # Basic statistics
    pos_mean, pos_std = np.mean(pos_prefs), np.std(pos_prefs)
    diag_mean, diag_std = np.mean(diag_prefs), np.std(diag_prefs)
    
    # Statistical tests
    t_stat_pos, p_value_pos = stats.ttest_1samp(pos_prefs, 0)
    t_stat_diag, p_value_diag = stats.ttest_1samp(diag_prefs, 0)
    
    # Confidence intervals
    pos_sem = stats.sem(pos_prefs)
    pos_ci = stats.t.interval(0.95, len(pos_prefs)-1, loc=pos_mean, scale=pos_sem)
    
    print(f"  Positional Preference:   {pos_mean:.6f} ± {pos_std:.6f}")
    print(f"  95% CI for pos pref:     [{pos_ci[0]:.6f}, {pos_ci[1]:.6f}]")
    print(f"  Diagonal Preference:     {diag_mean:.6f} ± {diag_std:.6f}")
    print(f"  \nDetailed QK Scores:")
    print(f"    Top → Top keys:        {np.mean(top_to_top):.6f} ± {np.std(top_to_top):.6f}")
    print(f"    Top → Bottom keys:     {np.mean(top_to_bottom):.6f} ± {np.std(top_to_bottom):.6f}")
    print(f"    Bottom → Top keys:     {np.mean(bottom_to_top):.6f} ± {np.std(bottom_to_top):.6f}")
    print(f"    Bottom → Bottom keys:  {np.mean(bottom_to_bottom):.6f} ± {np.std(bottom_to_bottom):.6f}")
    print(f"  \nStatistical Tests (vs 0):")
    print(f"    Positional pref t-test: t = {t_stat_pos:.4f}, p = {p_value_pos:.6f}")
    print(f"    Diagonal pref t-test:   t = {t_stat_diag:.4f}, p = {p_value_diag:.6f}")
    
    # Interpretation
    if p_value_pos < 0.001:
        pos_significance = "highly significant (p < 0.001)"
    elif p_value_pos < 0.01:
        pos_significance = "significant (p < 0.01)"
    elif p_value_pos < 0.05:
        pos_significance = "significant (p < 0.05)"
    else:
        pos_significance = "not significant (p ≥ 0.05)"
    
    print(f"  \nInterpretation:")
    print(f"    Positional preference is {pos_significance}")
    if pos_mean > 0.01:
        print(f"    Strong evidence for positional pattern (top queries → top keys, bottom queries → bottom keys)")
    elif pos_mean > 0.005:
        print(f"    Moderate evidence for positional pattern")
    elif pos_mean > 0:
        print(f"    Weak evidence for positional pattern")
    else:
        print(f"    No evidence for positional pattern")
    
    # Add violin plot comparison
    if 'top_queries_avg_qk' in qk_metrics[9] and len(qk_metrics[9]['top_queries_avg_qk']) > 0:
        top_qk = qk_metrics[9]['top_queries_avg_qk']
        bottom_qk = qk_metrics[9]['bottom_queries_avg_qk']
        
        top_qk_mean, bottom_qk_mean = np.mean(top_qk), np.mean(bottom_qk)
        qk_diff = top_qk_mean - bottom_qk_mean
        t_stat_violin, p_value_violin = stats.ttest_ind(top_qk, bottom_qk)
        
        print(f"  \nPreferred Key Position Analysis (for violin plots):")
        print(f"    Top 32 queries prefer key:    {top_qk_mean:.2f} ± {np.std(top_qk):.2f}")
        print(f"    Bottom 32 queries prefer key: {bottom_qk_mean:.2f} ± {np.std(bottom_qk):.2f}")
        print(f"    Difference (T-B):             {qk_diff:.2f}")
        print(f"    Two-sample t-test:            t = {t_stat_violin:.4f}, p = {p_value_violin:.6f}")
        print(f"    Expected: Large difference (top ~16, bottom ~48) for positional pattern")

print("\n" + "="*70)
