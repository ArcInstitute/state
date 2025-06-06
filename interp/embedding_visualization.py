#!/usr/bin/env python
"""
embedding_visualization.py – v5
===============================
Visualizes the differences between cell embeddings for two groups of cells.
Focuses on embedding analysis rather than attention patterns.

The script now:
* Loads cell data and samples two groups of 32 cells each from different cell types
* Extracts and compares embeddings between the two groups
* Creates various visualizations to show embedding differences
* Performs statistical analysis of embedding distributions
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
import umap

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))

DATA_PATH = Path(
    "/large_storage/ctc/ML/state_sets/replogle/processed.h5"
)
FIG_DIR = Path(__file__).resolve().parent / "figures" / "embedding_comparison" 
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
if "gene" in adata_full.obs.columns:
    drugname_counts = adata_full.obs["gene"].value_counts()
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

# Get available cell types and select the two most abundant ones
cell_type_counts = adata_full.obs["cell_type"].value_counts()
logger.info("Available cell types: %s", list(cell_type_counts.index))

# Select the two most abundant cell types
celltype1, celltype2 = cell_type_counts.index[:2]
logger.info(f"Selected cell types: {celltype1} ({cell_type_counts[celltype1]} available), {celltype2} ({cell_type_counts[celltype2]} available)")

# Get control cells for each cell type
cells_type1 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype1)].copy()
cells_type2 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype2)].copy()

logger.info(f"Available cells - {celltype1}: {cells_type1.n_obs}, {celltype2}: {cells_type2.n_obs}")

# Sample cells - 32 per cell type
n_cells_per_type = 32  

if cells_type1.n_obs >= n_cells_per_type and cells_type2.n_obs >= n_cells_per_type:
    # Sample cells
    idx1 = np.random.choice(cells_type1.n_obs, size=n_cells_per_type, replace=False)
    idx2 = np.random.choice(cells_type2.n_obs, size=n_cells_per_type, replace=False)
    
    sampled_type1 = cells_type1[idx1].copy()
    sampled_type2 = cells_type2[idx2].copy()
    
    logger.info(f"Sampled {sampled_type1.n_obs} cells of type {celltype1}")
    logger.info(f"Sampled {sampled_type2.n_obs} cells of type {celltype2}")
    
    # Extract embeddings for each group
    if embed_key in sampled_type1.obsm:
        embeddings_group1 = sampled_type1.obsm[embed_key]
        embeddings_group2 = sampled_type2.obsm[embed_key]
    else:
        embeddings_group1 = sampled_type1.X.toarray() if hasattr(sampled_type1.X, 'toarray') else sampled_type1.X
        embeddings_group2 = sampled_type2.X.toarray() if hasattr(sampled_type2.X, 'toarray') else sampled_type2.X
    
    logger.info(f"Group 1 embeddings shape: {embeddings_group1.shape}")
    logger.info(f"Group 2 embeddings shape: {embeddings_group2.shape}")
    
    # Combine embeddings for joint visualization
    all_embeddings = np.vstack([embeddings_group1, embeddings_group2])
    labels = ['Group 1'] * n_cells_per_type + ['Group 2'] * n_cells_per_type
    cell_types = [celltype1] * n_cells_per_type + [celltype2] * n_cells_per_type
    
    # === 1. Basic Statistics ===
    logger.info("=== Computing Basic Statistics ===")
    
    # Compute basic stats
    mean_group1 = np.mean(embeddings_group1, axis=0)
    mean_group2 = np.mean(embeddings_group2, axis=0)
    std_group1 = np.std(embeddings_group1, axis=0)
    std_group2 = np.std(embeddings_group2, axis=0)
    
    # Statistical tests
    t_stats, p_values = stats.ttest_ind(embeddings_group1, embeddings_group2, axis=0)
    significant_dims = np.sum(p_values < 0.05)
    
    logger.info(f"Mean embedding magnitude - Group 1: {np.linalg.norm(mean_group1):.4f}, Group 2: {np.linalg.norm(mean_group2):.4f}")
    logger.info(f"Significant dimensions (p < 0.05): {significant_dims} out of {embeddings_group1.shape[1]}")
    
    # === 2. Distance Analysis ===
    logger.info("=== Computing Distance Metrics ===")
    
    # Cosine similarity between group centroids
    centroid_cosine_sim = cosine_similarity([mean_group1], [mean_group2])[0, 0]
    
    # Euclidean distance between group centroids
    centroid_euclidean_dist = euclidean_distances([mean_group1], [mean_group2])[0, 0]
    
    # Within-group and between-group distances
    within_group1_dists = euclidean_distances(embeddings_group1)
    within_group2_dists = euclidean_distances(embeddings_group2)
    between_group_dists = euclidean_distances(embeddings_group1, embeddings_group2)
    
    mean_within_group1 = np.mean(within_group1_dists[np.triu_indices_from(within_group1_dists, k=1)])
    mean_within_group2 = np.mean(within_group2_dists[np.triu_indices_from(within_group2_dists, k=1)])
    mean_between_groups = np.mean(between_group_dists)
    
    logger.info(f"Centroid cosine similarity: {centroid_cosine_sim:.4f}")
    logger.info(f"Centroid Euclidean distance: {centroid_euclidean_dist:.4f}")
    logger.info(f"Mean within-group distance - Group 1: {mean_within_group1:.4f}, Group 2: {mean_within_group2:.4f}")
    logger.info(f"Mean between-group distance: {mean_between_groups:.4f}")
    
    # === 3. Visualization - PCA ===
    logger.info("=== Performing PCA ===")
    
    pca = PCA(n_components=min(10, all_embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(all_embeddings)
    
    plt.figure(figsize=(15, 5))
    
    # PCA scatter plot
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue']
    for i, (label, color) in enumerate(zip(['Group 1', 'Group 2'], colors)):
        mask = np.array(labels) == label
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                   c=color, label=f'{label} ({cell_types[i*n_cells_per_type]})', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA - First Two Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PCA explained variance
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    # Distance distribution
    plt.subplot(1, 3, 3)
    plt.hist(within_group1_dists[np.triu_indices_from(within_group1_dists, k=1)], 
             bins=20, alpha=0.5, label=f'Within {celltype1}', color='red')
    plt.hist(within_group2_dists[np.triu_indices_from(within_group2_dists, k=1)], 
             bins=20, alpha=0.5, label=f'Within {celltype2}', color='blue')
    plt.hist(between_group_dists.flatten(), bins=20, alpha=0.5, 
             label='Between groups', color='green')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIG_DIR / "embedding_comparison_pca.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved PCA analysis → %s", fig_path)
    
    # === 4. Visualization - t-SNE ===
    logger.info("=== Performing t-SNE ===")
    
    # Use PCA embeddings for t-SNE to reduce computation time
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//2))
    embeddings_tsne = tsne.fit_transform(embeddings_pca[:, :10])  # Use first 10 PCs
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    for i, (label, color) in enumerate(zip(['Group 1', 'Group 2'], colors)):
        mask = np.array(labels) == label
        plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                   c=color, label=f'{label} ({cell_types[i*n_cells_per_type]})', alpha=0.7)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # === 5. Visualization - UMAP ===
    logger.info("=== Performing UMAP ===")
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_embeddings)//2))
    embeddings_umap = reducer.fit_transform(embeddings_pca[:, :10])  # Use first 10 PCs
    
    plt.subplot(1, 2, 2)
    for i, (label, color) in enumerate(zip(['Group 1', 'Group 2'], colors)):
        mask = np.array(labels) == label
        plt.scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1], 
                   c=color, label=f'{label} ({cell_types[i*n_cells_per_type]})', alpha=0.7)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title('UMAP Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIG_DIR / "embedding_comparison_tsne_umap.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved t-SNE and UMAP analysis → %s", fig_path)
    
    # === 6. Feature Importance Analysis ===
    logger.info("=== Analyzing Feature Importance ===")
    
    # Find most discriminative dimensions
    abs_t_stats = np.abs(t_stats)
    top_discriminative_dims = np.argsort(abs_t_stats)[-20:]  # Top 20 most discriminative
    
    plt.figure(figsize=(15, 10))
    
    # Heatmap of top discriminative dimensions
    plt.subplot(2, 2, 1)
    discriminative_data = np.vstack([
        embeddings_group1[:, top_discriminative_dims],
        embeddings_group2[:, top_discriminative_dims]
    ])
    group_labels = [f'G1_{i}' for i in range(n_cells_per_type)] + [f'G2_{i}' for i in range(n_cells_per_type)]
    
    sns.heatmap(discriminative_data.T, cmap='viridis', 
                xticklabels=False, yticklabels=[f'Dim_{d}' for d in top_discriminative_dims])
    plt.title('Top 20 Discriminative Dimensions')
    plt.xlabel('Cells (G1: Group 1, G2: Group 2)')
    
    # t-statistic distribution
    plt.subplot(2, 2, 2)
    plt.hist(t_stats, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('t-statistic')
    plt.ylabel('Frequency')
    plt.title('Distribution of t-statistics')
    plt.grid(True, alpha=0.3)
    
    # p-value distribution
    plt.subplot(2, 2, 3)
    plt.hist(p_values, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mean difference per dimension
    plt.subplot(2, 2, 4)
    mean_diff = mean_group2 - mean_group1
    plt.scatter(range(len(mean_diff)), mean_diff, alpha=0.6, s=10)
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Dimension')
    plt.ylabel('Mean Difference (Group 2 - Group 1)')
    plt.title('Mean Embedding Differences')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = FIG_DIR / "embedding_feature_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved feature analysis → %s", fig_path)
    
    # === 7. Summary Statistics ===
    logger.info("=== Summary Statistics ===")
    
    summary_stats = {
        'cell_types': (celltype1, celltype2),
        'n_cells_per_group': n_cells_per_type,
        'embedding_dimension': embeddings_group1.shape[1],
        'centroid_cosine_similarity': centroid_cosine_sim,
        'centroid_euclidean_distance': centroid_euclidean_dist,
        'mean_within_group1_distance': mean_within_group1,
        'mean_within_group2_distance': mean_within_group2,
        'mean_between_group_distance': mean_between_groups,
        'significant_dimensions': significant_dims,
        'total_dimensions': embeddings_group1.shape[1],
        'pca_variance_explained_2pc': np.sum(pca.explained_variance_ratio_[:2]),
    }
    
    # Save summary to file
    summary_path = FIG_DIR / "embedding_comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Embedding Comparison Summary\n")
        f.write("="*30 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("Saved summary statistics → %s", summary_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EMBEDDING COMPARISON SUMMARY")
    print("="*50)
    print(f"Cell types compared: {celltype1} vs {celltype2}")
    print(f"Cells per group: {n_cells_per_type}")
    print(f"Embedding dimension: {embeddings_group1.shape[1]}")
    print(f"Centroid cosine similarity: {centroid_cosine_sim:.4f}")
    print(f"Centroid Euclidean distance: {centroid_euclidean_dist:.4f}")
    print(f"Mean within-group distances: {mean_within_group1:.4f} vs {mean_within_group2:.4f}")
    print(f"Mean between-group distance: {mean_between_groups:.4f}")
    print(f"Significant dimensions: {significant_dims}/{embeddings_group1.shape[1]} ({100*significant_dims/embeddings_group1.shape[1]:.1f}%)")
    print(f"PCA variance explained (2 PCs): {100*np.sum(pca.explained_variance_ratio_[:2]):.1f}%")
    print("="*50)
    
else:
    logger.error(f"Insufficient cells: {celltype1}: {cells_type1.n_obs}, {celltype2}: {cells_type2.n_obs}. Need at least {n_cells_per_type} each.")

logger.info("Script completed successfully.")
logger.info(f"Generated embedding visualizations in {FIG_DIR}")
