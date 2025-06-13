#!/usr/bin/env python
"""
embedding_visualization.py – v6
===============================
Visualizes embedding differences across ALL cell types in the dataset.
Focuses on comprehensive embedding analysis across multiple cell types.

The script now:
* Loads cell data and samples cells from ALL available cell types
* Extracts and compares embeddings between all cell types
* Creates various visualizations to show embedding relationships
* Performs statistical analysis of embedding distributions across cell types
"""

from __future__ import annotations

import os
import sys
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import umap
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))

DATA_PATH = Path(
    "/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct/plate1.h5"
)
FIG_DIR = Path(__file__).resolve().parent / "figures" / "tahoe_embeddings" 
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data directly
adata_full = sc.read_h5ad(str(DATA_PATH))
logger.info("Using full dataset shape: %s", adata_full.shape)

# Debug: Explore the AnnData structure
logger.info("=== AnnData Structure Debug ===")
logger.info(f"Observation columns (adata.obs.columns): {list(adata_full.obs.columns)}")
logger.info(f"Variable columns (adata.var.columns): {list(adata_full.var.columns)}")
logger.info(f"Observation matrices (adata.obsm.keys()): {list(adata_full.obsm.keys())}")
logger.info(f"Unstructured annotations (adata.uns.keys()): {list(adata_full.uns.keys())}")
if hasattr(adata_full, 'layers') and adata_full.layers:
    logger.info(f"Data layers (adata.layers.keys()): {list(adata_full.layers.keys())}")
logger.info("=== End Debug ===\n")

# Determine embedding key - typically "X_vci" for VCI embeddings or "X_hvg" for HVG expression
embed_key = "X_vci_1.5.2_4"  # Try VCI embeddings first
if embed_key not in adata_full.obsm:
    embed_key = "X_hvg"  # Fallback to HVG
    if embed_key not in adata_full.obsm:
        embed_key = "X"  # Last resort - raw expression
        logger.warning("Using raw expression data")
logger.info(f"Using embedding key: {embed_key}")
if embed_key in adata_full.obsm:
    logger.info(f"Embedding shape: {adata_full.obsm[embed_key].shape}")
else:
    logger.info(f"Expression shape: {adata_full.X.shape}")

# Find control perturbation
control_pert = "DMSO_TF_24h"  # Default control
if "drugname_drugconc" in adata_full.obs.columns:
    drugname_counts = adata_full.obs["drugname_drugconc"].value_counts()
    # Look for common control names
    for potential_control in ["DMSO_TF_24h", "non-targeting", "control", "DMSO"]:
        if potential_control in drugname_counts.index:
            control_pert = potential_control
            break   
    else:
        # Use the most common perturbation as control
        control_pert = drugname_counts.index[0]
        logger.warning(f"Could not find standard control, using: {control_pert}")

logger.info("Control perturbation: %s", control_pert)

# Get available cell types and their counts
cell_type_counts = adata_full.obs["cell_name"].value_counts()
logger.info("Available cell types: %s", dict(cell_type_counts))

# Select cell types with sufficient cells (at least 20 cells for meaningful analysis)
min_cells_per_type = 20
valid_cell_types = cell_type_counts[cell_type_counts >= min_cells_per_type].index.tolist()
logger.info(f"Cell types with >= {min_cells_per_type} cells: {valid_cell_types}")

# Limit to top 8 cell types to keep visualizations manageable
max_cell_types = 8
if len(valid_cell_types) > max_cell_types:
    valid_cell_types = valid_cell_types[:max_cell_types]
    logger.info(f"Limited to top {max_cell_types} cell types: {valid_cell_types}")

# Sample cells from each cell type
n_cells_per_type = 20  # Reduced from 32 to handle more cell types
cell_type_data = {}
all_embeddings = []
all_labels = []
all_cell_types = []

logger.info("=== Sampling cells from each cell type ===")
for cell_type in valid_cell_types:
    # Get control cells for this cell type
    cells_this_type = adata_full[(adata_full.obs["drugname_drugconc"] == control_pert) & 
                                (adata_full.obs["cell_name"] == cell_type)].copy()
    
    if cells_this_type.n_obs >= n_cells_per_type:
        # Sample cells
        idx = np.random.choice(cells_this_type.n_obs, size=n_cells_per_type, replace=False)
        sampled_cells = cells_this_type[idx].copy()
        
        # Extract embeddings
        if embed_key in sampled_cells.obsm:
            embeddings = sampled_cells.obsm[embed_key]
        else:
            embeddings = sampled_cells.X.toarray() if hasattr(sampled_cells.X, 'toarray') else sampled_cells.X
        
        cell_type_data[cell_type] = {
            'cells': sampled_cells,
            'embeddings': embeddings,
            'n_cells': n_cells_per_type
        }
        
        all_embeddings.append(embeddings)
        all_labels.extend([cell_type] * n_cells_per_type)
        all_cell_types.extend([cell_type] * n_cells_per_type)
        
        logger.info(f"Sampled {n_cells_per_type} cells of type {cell_type}, embedding shape: {embeddings.shape}")
    else:
        logger.warning(f"Insufficient cells for {cell_type}: {cells_this_type.n_obs} available")

# Combine all embeddings
all_embeddings = np.vstack(all_embeddings)
logger.info(f"Combined embeddings shape: {all_embeddings.shape}")
logger.info(f"Total cell types analyzed: {len(cell_type_data)}")

# === 1. Compute Pairwise Statistics ===
logger.info("=== Computing Pairwise Statistics ===")

cell_types_list = list(cell_type_data.keys())
n_types = len(cell_types_list)

# Compute centroids for each cell type
centroids = {}
for cell_type in cell_types_list:
    centroids[cell_type] = np.mean(cell_type_data[cell_type]['embeddings'], axis=0)

# Compute pairwise distances and similarities between centroids
pairwise_cosine_sim = np.zeros((n_types, n_types))
pairwise_euclidean_dist = np.zeros((n_types, n_types))

for i, ct1 in enumerate(cell_types_list):
    for j, ct2 in enumerate(cell_types_list):
        if i == j:
            pairwise_cosine_sim[i, j] = 1.0
            pairwise_euclidean_dist[i, j] = 0.0
        else:
            cosine_sim = cosine_similarity([centroids[ct1]], [centroids[ct2]])[0, 0]
            euclidean_dist = euclidean_distances([centroids[ct1]], [centroids[ct2]])[0, 0]
            pairwise_cosine_sim[i, j] = cosine_sim
            pairwise_euclidean_dist[i, j] = euclidean_dist

logger.info("Computed pairwise similarities and distances between cell type centroids")

# === 2. Within-group vs Between-group Analysis ===
logger.info("=== Computing Within vs Between Group Statistics ===")

within_group_distances = {}
between_group_distances = {}

# Compute within-group distances
for cell_type in cell_types_list:
    embeddings = cell_type_data[cell_type]['embeddings']
    dists = euclidean_distances(embeddings)
    within_group_distances[cell_type] = dists[np.triu_indices_from(dists, k=1)]

# Compute between-group distances
for i, ct1 in enumerate(cell_types_list):
    for j, ct2 in enumerate(cell_types_list):
        if i < j:  # Only compute upper triangle
            emb1 = cell_type_data[ct1]['embeddings']
            emb2 = cell_type_data[ct2]['embeddings']
            dists = euclidean_distances(emb1, emb2)
            between_group_distances[f"{ct1}_vs_{ct2}"] = dists.flatten()

# === 3. PCA Analysis ===
logger.info("=== Performing PCA Analysis ===")

pca = PCA(n_components=min(20, all_embeddings.shape[1]))
embeddings_pca = pca.fit_transform(all_embeddings)

# Create green color palette for cell types
colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(cell_types_list)))
color_map = dict(zip(cell_types_list, colors))

# Set prettier matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

plt.figure(figsize=(20, 15))

# PCA scatter plot
plt.subplot(3, 3, 1)
for i, cell_type in enumerate(cell_types_list):
    mask = np.array(all_labels) == cell_type
    plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
               c=[color_map[cell_type]], label=cell_type, alpha=0.8, s=35, edgecolors='white', linewidth=0.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA - All Cell Types', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)

# PCA explained variance
plt.subplot(3, 3, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'o-', color='darkgreen', linewidth=2, markersize=6)
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance', fontweight='bold')
plt.fill_between(range(1, len(pca.explained_variance_ratio_) + 1), 
                 np.cumsum(pca.explained_variance_ratio_), alpha=0.3, color='lightgreen')

# Pairwise cosine similarity heatmap
plt.subplot(3, 3, 3)
sns.heatmap(pairwise_cosine_sim, annot=True, fmt='.3f', 
            xticklabels=cell_types_list, yticklabels=cell_types_list,
            cmap='Greens', cbar_kws={'label': 'Cosine Similarity'}, 
            square=True, linewidths=0.5)
plt.title('Pairwise Cosine Similarity\n(Centroid-based)', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Pairwise Euclidean distance heatmap
plt.subplot(3, 3, 4)
sns.heatmap(pairwise_euclidean_dist, annot=True, fmt='.2f',
            xticklabels=cell_types_list, yticklabels=cell_types_list,
            cmap='YlOrRd', cbar_kws={'label': 'Euclidean Distance'}, 
            square=True, linewidths=0.5)
plt.title('Pairwise Euclidean Distance\n(Centroid-based)', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Within-group distance distributions
plt.subplot(3, 3, 5)
for cell_type in cell_types_list:
    plt.hist(within_group_distances[cell_type], bins=15, alpha=0.7, 
             label=cell_type, color=color_map[cell_type], edgecolor='white', linewidth=0.5)
plt.xlabel('Within-group Euclidean Distance')
plt.ylabel('Frequency')
plt.title('Within-group Distance Distributions', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)

# Between-group distance distributions (sample of pairs)
plt.subplot(3, 3, 6)
pair_keys = list(between_group_distances.keys())[:5]  # Show first 5 pairs
green_shades = plt.cm.Greens(np.linspace(0.4, 0.8, len(pair_keys)))
for i, pair_key in enumerate(pair_keys):
    plt.hist(between_group_distances[pair_key], bins=15, alpha=0.7, 
             label=pair_key, color=green_shades[i], edgecolor='white', linewidth=0.5)
plt.xlabel('Between-group Euclidean Distance')
plt.ylabel('Frequency')
plt.title('Between-group Distance Distributions\n(Sample of pairs)', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)

# Hierarchical clustering of cell types
plt.subplot(3, 3, 7)
linkage_matrix = linkage(pairwise_euclidean_dist, method='ward')
dendrogram(linkage_matrix, labels=cell_types_list, orientation='top', 
           color_threshold=0.7*max(linkage_matrix[:,2]), above_threshold_color='darkgreen')
plt.title('Hierarchical Clustering\n(Euclidean Distance)', fontweight='bold')
plt.xticks(rotation=45, ha='right')

# 3D PCA plot
ax = plt.subplot(3, 3, 8, projection='3d')
for i, cell_type in enumerate(cell_types_list):
    mask = np.array(all_labels) == cell_type
    ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], embeddings_pca[mask, 2],
              c=[color_map[cell_type]], label=cell_type, alpha=0.8, s=25, edgecolors='white', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
ax.set_title('3D PCA - All Cell Types', fontweight='bold')
ax.grid(True, alpha=0.3)

# Summary statistics table
plt.subplot(3, 3, 9)
plt.axis('off')
summary_text = "Summary Statistics:\n\n"
summary_text += f"Cell types analyzed: {len(cell_types_list)}\n"
summary_text += f"Cells per type: {n_cells_per_type}\n"
summary_text += f"Total cells: {len(all_labels)}\n"
summary_text += f"Embedding dimension: {all_embeddings.shape[1]}\n"
summary_text += f"PCA variance (3 PCs): {100*np.sum(pca.explained_variance_ratio_[:3]):.1f}%\n\n"

# Add mean distances
mean_within = np.mean([np.mean(within_group_distances[ct]) for ct in cell_types_list])
mean_between = np.mean([np.mean(between_group_distances[k]) for k in between_group_distances.keys()])
summary_text += f"Mean within-group dist: {mean_within:.3f}\n"
summary_text += f"Mean between-group dist: {mean_between:.3f}\n"
summary_text += f"Separation ratio: {mean_between/mean_within:.2f}"

plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top', 
         transform=plt.gca().transAxes, family='monospace')

plt.tight_layout()
fig_path = FIG_DIR / "embedding_all_celltypes_overview.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved overview analysis → %s", fig_path)

# === 4. t-SNE and UMAP Visualizations ===
logger.info("=== Performing t-SNE and UMAP ===")

plt.figure(figsize=(15, 5))

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//3))
embeddings_tsne = tsne.fit_transform(embeddings_pca[:, :10])  # Use first 10 PCs

plt.subplot(1, 3, 1)
for cell_type in cell_types_list:
    mask = np.array(all_labels) == cell_type
    plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
               c=[color_map[cell_type]], label=cell_type, alpha=0.8, s=35, 
               edgecolors='white', linewidth=0.5)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization - All Cell Types', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_embeddings)//4))
embeddings_umap = reducer.fit_transform(embeddings_pca[:, :10])  # Use first 10 PCs

plt.subplot(1, 3, 2)
for cell_type in cell_types_list:
    mask = np.array(all_labels) == cell_type
    plt.scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1], 
               c=[color_map[cell_type]], label=cell_type, alpha=0.8, s=35,
               edgecolors='white', linewidth=0.5)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Visualization - All Cell Types', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)

# Distance comparison: within vs between groups
plt.subplot(1, 3, 3)
within_means = [np.mean(within_group_distances[ct]) for ct in cell_types_list]
within_stds = [np.std(within_group_distances[ct]) for ct in cell_types_list]

x_pos = np.arange(len(cell_types_list))
plt.bar(x_pos, within_means, yerr=within_stds, alpha=0.8, 
        color=[color_map[ct] for ct in cell_types_list], capsize=5, 
        edgecolor='white', linewidth=0.8)
plt.axhline(mean_between, color='darkred', linestyle='--', linewidth=2,
           label=f'Mean between-group: {mean_between:.3f}')
plt.xlabel('Cell Type')
plt.ylabel('Mean Euclidean Distance')
plt.title('Within-group vs Between-group Distances', fontweight='bold')
plt.xticks(x_pos, cell_types_list, rotation=45, ha='right')
plt.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
fig_path = FIG_DIR / "embedding_tsne_umap_distances.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved t-SNE, UMAP and distance analysis → %s", fig_path)

# === 5. Statistical Analysis ===
logger.info("=== Performing Statistical Analysis ===")

# Perform ANOVA across all cell types for each embedding dimension
f_stats = []
p_values_anova = []

for dim in range(all_embeddings.shape[1]):
    groups = [cell_type_data[ct]['embeddings'][:, dim] for ct in cell_types_list]
    f_stat, p_val = stats.f_oneway(*groups)
    f_stats.append(f_stat)
    p_values_anova.append(p_val)

f_stats = np.array(f_stats)
p_values_anova = np.array(p_values_anova)
significant_dims_anova = np.sum(p_values_anova < 0.05)

plt.figure(figsize=(15, 10))

# F-statistic distribution
plt.subplot(2, 3, 1)
plt.hist(f_stats, bins=50, alpha=0.8, color='forestgreen', edgecolor='white', linewidth=0.5)
plt.xlabel('F-statistic')
plt.ylabel('Frequency')
plt.title('ANOVA F-statistics Distribution', fontweight='bold')

# p-value distribution
plt.subplot(2, 3, 2)
plt.hist(p_values_anova, bins=50, alpha=0.8, color='mediumseagreen', edgecolor='white', linewidth=0.5)
plt.axvline(0.05, color='darkred', linestyle='--', linewidth=2, label='p = 0.05')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title('ANOVA p-values Distribution', fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True)

# Most discriminative dimensions
plt.subplot(2, 3, 3)
top_discriminative_dims = np.argsort(f_stats)[-20:]  # Top 20
top_f_stats = f_stats[top_discriminative_dims]
plt.barh(range(len(top_f_stats)), top_f_stats, color='darkseagreen', 
         alpha=0.8, edgecolor='white', linewidth=0.5)
plt.ylabel('Dimension Rank')
plt.xlabel('F-statistic')
plt.title('Top 20 Most Discriminative Dimensions', fontweight='bold')

# Centroid magnitude comparison
plt.subplot(2, 3, 4)
centroid_magnitudes = [np.linalg.norm(centroids[ct]) for ct in cell_types_list]
plt.bar(range(len(cell_types_list)), centroid_magnitudes, 
        color=[color_map[ct] for ct in cell_types_list], alpha=0.8,
        edgecolor='white', linewidth=0.8)
plt.xlabel('Cell Type')
plt.ylabel('Centroid Magnitude')
plt.title('Embedding Centroid Magnitudes', fontweight='bold')
plt.xticks(range(len(cell_types_list)), cell_types_list, rotation=45, ha='right')

# Variance within each cell type
plt.subplot(2, 3, 5)
variances = []
for ct in cell_types_list:
    embeddings = cell_type_data[ct]['embeddings']
    var = np.mean(np.var(embeddings, axis=0))
    variances.append(var)

plt.bar(range(len(cell_types_list)), variances,
        color=[color_map[ct] for ct in cell_types_list], alpha=0.8,
        edgecolor='white', linewidth=0.8)
plt.xlabel('Cell Type')
plt.ylabel('Mean Variance')
plt.title('Mean Embedding Variance by Cell Type', fontweight='bold')
plt.xticks(range(len(cell_types_list)), cell_types_list, rotation=45, ha='right')

# Silhouette-like analysis: ratio of within to between distances
plt.subplot(2, 3, 6)
separation_scores = []
for ct in cell_types_list:
    within_mean = np.mean(within_group_distances[ct])
    # Get mean distance to all other cell types
    between_means = []
    for other_ct in cell_types_list:
        if other_ct != ct:
            pair_key = f"{ct}_vs_{other_ct}" if f"{ct}_vs_{other_ct}" in between_group_distances else f"{other_ct}_vs_{ct}"
            if pair_key in between_group_distances:
                between_means.append(np.mean(between_group_distances[pair_key]))
    
    if between_means:
        between_mean = np.mean(between_means)
        separation_score = between_mean / within_mean
        separation_scores.append(separation_score)
    else:
        separation_scores.append(0)

plt.bar(range(len(cell_types_list)), separation_scores,
        color=[color_map[ct] for ct in cell_types_list], alpha=0.8,
        edgecolor='white', linewidth=0.8)
plt.axhline(1, color='darkred', linestyle='--', linewidth=2, label='Ratio = 1')
plt.xlabel('Cell Type')
plt.ylabel('Between/Within Distance Ratio')
plt.title('Cell Type Separation Scores', fontweight='bold')
plt.xticks(range(len(cell_types_list)), cell_types_list, rotation=45, ha='right')
plt.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
fig_path = FIG_DIR / "embedding_statistical_analysis.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info("Saved statistical analysis → %s", fig_path)

# === 6. Save Detailed Summary ===
logger.info("=== Generating Detailed Summary ===")

summary_stats = {
    'cell_types_analyzed': cell_types_list,
    'n_cells_per_type': n_cells_per_type,
    'total_cells': len(all_labels),
    'embedding_dimension': all_embeddings.shape[1],
    'significant_dimensions_anova': significant_dims_anova,
    'total_dimensions': all_embeddings.shape[1],
    'pca_variance_explained_3pc': np.sum(pca.explained_variance_ratio_[:3]),
    'mean_within_group_distance': mean_within,
    'mean_between_group_distance': mean_between,
    'separation_ratio': mean_between / mean_within,
    'centroid_magnitudes': dict(zip(cell_types_list, centroid_magnitudes)),
    'embedding_variances': dict(zip(cell_types_list, variances)),
    'separation_scores': dict(zip(cell_types_list, separation_scores)),
}

# Save summary to file
summary_path = FIG_DIR / "embedding_all_celltypes_summary.txt"
with open(summary_path, 'w') as f:
    f.write("Embedding Analysis - All Cell Types Summary\n")
    f.write("="*50 + "\n\n")
    
    f.write("BASIC STATISTICS:\n")
    f.write(f"Cell types analyzed: {len(cell_types_list)}\n")
    f.write(f"Cell types: {', '.join(cell_types_list)}\n")
    f.write(f"Cells per type: {n_cells_per_type}\n")
    f.write(f"Total cells: {len(all_labels)}\n")
    f.write(f"Embedding dimension: {all_embeddings.shape[1]}\n\n")
    
    f.write("STATISTICAL ANALYSIS:\n")
    f.write(f"Significant dimensions (ANOVA p < 0.05): {significant_dims_anova}/{all_embeddings.shape[1]} ({100*significant_dims_anova/all_embeddings.shape[1]:.1f}%)\n")
    f.write(f"PCA variance explained (3 PCs): {100*np.sum(pca.explained_variance_ratio_[:3]):.1f}%\n\n")
    
    f.write("DISTANCE ANALYSIS:\n")
    f.write(f"Mean within-group distance: {mean_within:.4f}\n")
    f.write(f"Mean between-group distance: {mean_between:.4f}\n")
    f.write(f"Separation ratio (between/within): {mean_between/mean_within:.2f}\n\n")
    
    f.write("CENTROID MAGNITUDES:\n")
    for ct, mag in zip(cell_types_list, centroid_magnitudes):
        f.write(f"{ct}: {mag:.4f}\n")
    f.write("\n")
    
    f.write("EMBEDDING VARIANCES:\n")
    for ct, var in zip(cell_types_list, variances):
        f.write(f"{ct}: {var:.4f}\n")
    f.write("\n")
    
    f.write("SEPARATION SCORES (between/within ratio):\n")
    for ct, score in zip(cell_types_list, separation_scores):
        f.write(f"{ct}: {score:.2f}\n")

# Save pairwise similarity/distance matrices
np.save(FIG_DIR / "pairwise_cosine_similarity.npy", pairwise_cosine_sim)
np.save(FIG_DIR / "pairwise_euclidean_distance.npy", pairwise_euclidean_dist)

# Save embedding data for further analysis
embedding_data = {
    'embeddings': all_embeddings,
    'labels': all_labels,
    'cell_types': cell_types_list,
    'centroids': centroids,
    'pca_embeddings': embeddings_pca,
    'tsne_embeddings': embeddings_tsne,
    'umap_embeddings': embeddings_umap
}
np.save(FIG_DIR / "embedding_analysis_data.npy", embedding_data)

logger.info("Saved detailed summary → %s", summary_path)
logger.info("Saved numerical data → %s", FIG_DIR)

# === 7. Print Summary ===
print("\n" + "="*70)
print("EMBEDDING ANALYSIS - ALL CELL TYPES SUMMARY")
print("="*70)
print(f"Cell types analyzed: {len(cell_types_list)}")
print(f"Cell types: {', '.join(cell_types_list)}")
print(f"Cells per type: {n_cells_per_type}")
print(f"Total cells analyzed: {len(all_labels)}")
print(f"Embedding dimension: {all_embeddings.shape[1]}")
print()
print("STATISTICAL RESULTS:")
print(f"Significant dimensions (ANOVA): {significant_dims_anova}/{all_embeddings.shape[1]} ({100*significant_dims_anova/all_embeddings.shape[1]:.1f}%)")
print(f"PCA variance explained (3 PCs): {100*np.sum(pca.explained_variance_ratio_[:3]):.1f}%")
print()
print("DISTANCE ANALYSIS:")
print(f"Mean within-group distance: {mean_within:.4f}")
print(f"Mean between-group distance: {mean_between:.4f}")
print(f"Separation ratio: {mean_between/mean_within:.2f}")
print()
print("TOP 3 MOST SEPARATED CELL TYPES:")
sorted_scores = sorted(zip(cell_types_list, separation_scores), key=lambda x: x[1], reverse=True)
for i, (ct, score) in enumerate(sorted_scores[:3]):
    print(f"{i+1}. {ct}: {score:.2f}")
print("="*70)

logger.info("Script completed successfully.")
logger.info(f"Generated comprehensive embedding visualizations in {FIG_DIR}")
