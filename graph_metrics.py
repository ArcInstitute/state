import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

def create_cosine_similarity_matrix_efficient(adata, sample_size=None):
    """
    Create cosine similarity matrix for the data
    
    Parameters:
    -----------
    adata : AnnData
        The data to compute similarity for
    sample_size : int, optional
        If provided, randomly sample this many vectors to make computation feasible
    
    Returns:
    --------
    cosine_sim : np.ndarray
        Cosine similarity matrix
    indices : np.ndarray
        Indices of sampled vectors (if sampling was used)
    """
    # Get the data matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Sample if requested
    if sample_size is not None and sample_size < X.shape[0]:
        print(f"Sampling {sample_size} vectors from {X.shape[0]} total vectors")
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        print(f"Sample shape: {X_sample.shape}")
    else:
        X_sample = X
        indices = np.arange(X.shape[0])
        print(f"Using all {X.shape[0]} vectors")
    
    # Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    cosine_sim = cosine_similarity(X_sample)
    print(f"Similarity matrix shape: {cosine_sim.shape}")
    
    return cosine_sim, indices

def create_euclidean_distance_matrix_efficient(adata, sample_size=None):
    """
    Create Euclidean distance matrix for the data
    
    Parameters:
    -----------
    adata : AnnData
        The data to compute distances for
    sample_size : int, optional
        If provided, randomly sample this many vectors to make computation feasible
    
    Returns:
    --------
    euclidean_dist : np.ndarray
        Euclidean distance matrix
    indices : np.ndarray
        Indices of sampled vectors (if sampling was used)
    """
    # Get the data matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Sample if requested
    if sample_size is not None and sample_size < X.shape[0]:
        print(f"Sampling {sample_size} vectors from {X.shape[0]} total vectors")
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        print(f"Sample shape: {X_sample.shape}")
    else:
        X_sample = X
        indices = np.arange(X.shape[0])
        print(f"Using all {X.shape[0]} vectors")
    
    # Compute Euclidean distance matrix
    print("Computing Euclidean distance matrix...")
    euclidean_dist = euclidean_distances(X_sample)
    print(f"Distance matrix shape: {euclidean_dist.shape}")
    
    return euclidean_dist, indices

def create_manhattan_distance_matrix_efficient(adata, sample_size=None):
    """
    Create Euclidean distance matrix for the data
    
    Parameters:
    -----------
    adata : AnnData
        The data to compute distances for
    sample_size : int, optional
        If provided, randomly sample this many vectors to make computation feasible
    
    Returns:
    --------
    euclidean_dist : np.ndarray
        Euclidean distance matrix
    indices : np.ndarray
        Indices of sampled vectors (if sampling was used)
    """
    # Get the data matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Sample if requested
    if sample_size is not None and sample_size < X.shape[0]:
        print(f"Sampling {sample_size} vectors from {X.shape[0]} total vectors")
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        print(f"Sample shape: {X_sample.shape}")
    else:
        X_sample = X
        indices = np.arange(X.shape[0])
        print(f"Using all {X.shape[0]} vectors")
    
    # Compute Manhattan distance matrix
    print("Computing Manhattan distance matrix...")
    manhattan_dist = manhattan_distances(X_sample)
    print(f"Distance matrix shape: {manhattan_dist.shape}")
    
    return manhattan_dist, indices


adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered/rpe1/eval_step=56000.ckpt/adata_real.h5ad')

# Let's start with a smaller sample to test
print("=== COSINE SIMILARITY ANALYSIS ===")
print(f"Real data shape: {adata_real.X.shape}")

cosine_sim_real_sample, indices_real = create_cosine_similarity_matrix_efficient(adata_real, sample_size=1000)

# Analyze the cosine similarity matrix
print("\n=== COSINE SIMILARITY MATRIX STATISTICS ===")
print(f"Matrix shape: {cosine_sim_real_sample.shape}")
print(f"Mean similarity: {cosine_sim_real_sample.mean():.6f}")
print(f"Std similarity: {cosine_sim_real_sample.std():.6f}")
print(f"Min similarity: {cosine_sim_real_sample.min():.6f}")
print(f"Max similarity: {cosine_sim_real_sample.max():.6f}")

# Diagonal should be 1.0 (self-similarity)
diagonal_values = np.diag(cosine_sim_real_sample)
print(f"Diagonal values (should be ~1.0): mean={diagonal_values.mean():.6f}, std={diagonal_values.std():.6f}")

# Off-diagonal similarities (excluding self-similarity)
mask = ~np.eye(cosine_sim_real_sample.shape[0], dtype=bool)
off_diagonal = cosine_sim_real_sample[mask]
print(f"Off-diagonal similarities: mean={off_diagonal.mean():.6f}, std={off_diagonal.std():.6f}")

# Look at distribution of similarities
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"Off-diagonal similarity percentiles:")
for p in percentiles:
    val = np.percentile(off_diagonal, p)
    print(f"  {p:2d}%: {val:.6f}")

# Find most and least similar pairs
flat_indices = np.unravel_index(np.argsort(cosine_sim_real_sample, axis=None), cosine_sim_real_sample.shape)
# Exclude diagonal elements for finding extremes
off_diag_flat = cosine_sim_real_sample[mask]
sorted_off_diag_idx = np.argsort(off_diag_flat)

print(f"\nMost similar pair (excluding self): {off_diag_flat[sorted_off_diag_idx[-1]]:.6f}")
print(f"Least similar pair: {off_diag_flat[sorted_off_diag_idx[0]]:.6f}")



# Compute Euclidean distance matrices for the same samples
print("=== EUCLIDEAN DISTANCE ANALYSIS ===")

# Use same random seed for fair comparison
np.random.seed(42)
euclidean_dist_real_sample, _ = create_euclidean_distance_matrix_efficient(adata_real, sample_size=1000)
manhattan_dist_real_sample, _ = create_manhattan_distance_matrix_efficient(adata_real, sample_size=1000)



# Analyze the Euclidean distance matrix
print("\n=== EUCLIDEAN DISTANCE MATRIX STATISTICS ===")
print(f"Matrix shape: {euclidean_dist_real_sample.shape}")
print(f"Mean distance: {euclidean_dist_real_sample.mean():.6f}")
print(f"Std distance: {euclidean_dist_real_sample.std():.6f}")
print(f"Min distance: {euclidean_dist_real_sample.min():.6f}")
print(f"Max distance: {euclidean_dist_real_sample.max():.6f}")

# Diagonal should be 0.0 (self-distance)
diagonal_values_dist = np.diag(euclidean_dist_real_sample)
print(f"Diagonal values (should be ~0.0): mean={diagonal_values_dist.mean():.6f}, std={diagonal_values_dist.std():.6f}")

# Off-diagonal distances (excluding self-distance)
mask_dist = ~np.eye(euclidean_dist_real_sample.shape[0], dtype=bool)
off_diagonal_dist = euclidean_dist_real_sample[mask_dist]
print(f"Off-diagonal distances: mean={off_diagonal_dist.mean():.6f}, std={off_diagonal_dist.std():.6f}")

# Look at distribution of distances
print(f"Off-diagonal distance percentiles:")
for p in percentiles:
    val = np.percentile(off_diagonal_dist, p)
    print(f"  {p:2d}%: {val:.6f}")

# Find most and least distant pairs
sorted_off_diag_dist_idx = np.argsort(off_diagonal_dist)

print(f"\nMost distant pair: {off_diagonal_dist[sorted_off_diag_dist_idx[-1]]:.6f}")
print(f"Least distant pair (excluding self): {off_diagonal_dist[sorted_off_diag_dist_idx[0]]:.6f}")

print("\n=== MANHATTAN DISTANCE MATRIX STATISTICS ===")
print(f"Matrix shape: {manhattan_dist_real_sample.shape}")
print(f"Mean distance: {manhattan_dist_real_sample.mean():.6f}")
print(f"Std distance: {manhattan_dist_real_sample.std():.6f}")
print(f"Min distance: {manhattan_dist_real_sample.min():.6f}")
print(f"Max distance: {manhattan_dist_real_sample.max():.6f}")

diagonal_values_dist = np.diag(manhattan_dist_real_sample)
print(f"Diagonal values (should be ~0.0): mean={diagonal_values_dist.mean():.6f}, std={diagonal_values_dist.std():.6f}")

# Off-diagonal distances (excluding self-distance)
mask_dist = ~np.eye(manhattan_dist_real_sample.shape[0], dtype=bool)
off_diagonal_dist = manhattan_dist_real_sample[mask_dist]
print(f"Off-diagonal distances: mean={off_diagonal_dist.mean():.6f}, std={off_diagonal_dist.std():.6f}")

# Look at distribution of distances
print(f"Off-diagonal distance percentiles:")
for p in percentiles:
    val = np.percentile(off_diagonal_dist, p)
    print(f"  {p:2d}%: {val:.6f}")

# Find most and least distant pairs
sorted_off_diag_dist_idx = np.argsort(off_diagonal_dist)

print(f"\nMost distant pair: {off_diagonal_dist[sorted_off_diag_dist_idx[-1]]:.6f}")
print(f"Least distant pair (excluding self): {off_diagonal_dist[sorted_off_diag_dist_idx[0]]:.6f}")

# ===============================
# VISUALIZATION SECTION
# ===============================

print("\n=== CREATING VISUALIZATIONS ===")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Heatmap of Cosine Similarity Matrix (subsampled for visibility)
plt.subplot(3, 4, 1)
sample_size_viz = min(100, cosine_sim_real_sample.shape[0])  # Use smaller sample for heatmap visibility
indices_viz = np.random.choice(cosine_sim_real_sample.shape[0], sample_size_viz, replace=False)
cosine_viz = cosine_sim_real_sample[np.ix_(indices_viz, indices_viz)]
sns.heatmap(cosine_viz, cmap='RdYlBu_r', center=0, square=True, 
            cbar_kws={'label': 'Cosine Similarity'})
plt.title(f'Cosine Similarity Matrix\n(Sample of {sample_size_viz} vectors)')
plt.xlabel('Vector Index')
plt.ylabel('Vector Index')

# 2. Distribution of Cosine Similarities (off-diagonal)
plt.subplot(3, 4, 2)
mask = ~np.eye(cosine_sim_real_sample.shape[0], dtype=bool)
off_diagonal_cosine = cosine_sim_real_sample[mask]
plt.hist(off_diagonal_cosine, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(off_diagonal_cosine.mean(), color='red', linestyle='--', 
            label=f'Mean: {off_diagonal_cosine.mean():.3f}')
plt.axvline(off_diagonal_cosine.mean() + off_diagonal_cosine.std(), color='orange', linestyle='-', 
            label=f'Mean + Std: {off_diagonal_cosine.mean() + off_diagonal_cosine.std():.3f}')
plt.axvline(off_diagonal_cosine.mean() - off_diagonal_cosine.std(), color='orange', linestyle='-', 
            label=f'Mean - Std: {off_diagonal_cosine.mean() - off_diagonal_cosine.std():.3f}')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Similarities\n(Off-diagonal values)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Box plot of Cosine Similarities
plt.subplot(3, 4, 3)
plt.boxplot(off_diagonal_cosine, labels=['Cosine Similarity'])
plt.ylabel('Similarity Value')
plt.title('Cosine Similarity\nDistribution Summary')
plt.grid(True, alpha=0.3)

# 4. Cosine Similarity Percentile Plot
plt.subplot(3, 4, 4)
percentiles_range = np.arange(1, 100, 1)
percentile_values = [np.percentile(off_diagonal_cosine, p) for p in percentiles_range]
plt.plot(percentiles_range, percentile_values, linewidth=2)
plt.xlabel('Percentile')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity\nPercentile Curve')
plt.grid(True, alpha=0.3)

# 5. Heatmap of Euclidean Distance Matrix
plt.subplot(3, 4, 5)
euclidean_viz = euclidean_dist_real_sample[np.ix_(indices_viz, indices_viz)]
sns.heatmap(euclidean_viz, cmap='viridis', square=True, 
            cbar_kws={'label': 'Euclidean Distance'})
plt.title(f'Euclidean Distance Matrix\n(Sample of {sample_size_viz} vectors)')
plt.xlabel('Vector Index')
plt.ylabel('Vector Index')

# 6. Distribution of Euclidean Distances (off-diagonal)
plt.subplot(3, 4, 6)
mask_dist = ~np.eye(euclidean_dist_real_sample.shape[0], dtype=bool)
off_diagonal_euclidean = euclidean_dist_real_sample[mask_dist]
plt.hist(off_diagonal_euclidean, bins=50, alpha=0.7, edgecolor='black', color='green')
plt.axvline(off_diagonal_euclidean.mean(), color='red', linestyle='--', 
            label=f'Mean: {off_diagonal_euclidean.mean():.3f}')
plt.axvline(off_diagonal_euclidean.mean() + off_diagonal_euclidean.std(), color='blue', linestyle='-', 
            label=f'Mean + Std: {off_diagonal_euclidean.mean() + off_diagonal_euclidean.std():.3f}')
plt.axvline(off_diagonal_euclidean.mean() - off_diagonal_euclidean.std(), color='blue', linestyle='-', 
            label=f'Mean - Std: {off_diagonal_euclidean.mean() - off_diagonal_euclidean.std():.3f}')
plt.xlabel('Euclidean Distance')
plt.ylabel('Frequency')
plt.title('Distribution of Euclidean Distances\n(Off-diagonal values)')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Box plot of Euclidean Distances
plt.subplot(3, 4, 7)
plt.boxplot(off_diagonal_euclidean, labels=['Euclidean Distance'])
plt.ylabel('Distance Value')
plt.title('Euclidean Distance\nDistribution Summary')
plt.grid(True, alpha=0.3)

# 8. Euclidean Distance Percentile Plot
plt.subplot(3, 4, 8)
percentile_values_euc = [np.percentile(off_diagonal_euclidean, p) for p in percentiles_range]
plt.plot(percentiles_range, percentile_values_euc, linewidth=2, color='green')
plt.xlabel('Percentile')
plt.ylabel('Euclidean Distance')
plt.title('Euclidean Distance\nPercentile Curve')
plt.grid(True, alpha=0.3)

# 9. Heatmap of Manhattan Distance Matrix
plt.subplot(3, 4, 9)
manhattan_viz = manhattan_dist_real_sample[np.ix_(indices_viz, indices_viz)]
sns.heatmap(manhattan_viz, cmap='plasma', square=True, 
            cbar_kws={'label': 'Manhattan Distance'})
plt.title(f'Manhattan Distance Matrix\n(Sample of {sample_size_viz} vectors)')
plt.xlabel('Vector Index')
plt.ylabel('Vector Index')

# 10. Distribution of Manhattan Distances (off-diagonal)
plt.subplot(3, 4, 10)
off_diagonal_manhattan = manhattan_dist_real_sample[mask_dist]
plt.hist(off_diagonal_manhattan, bins=50, alpha=0.7, edgecolor='black', color='orange')
plt.axvline(off_diagonal_manhattan.mean(), color='red', linestyle='--', 
            label=f'Mean: {off_diagonal_manhattan.mean():.3f}')
plt.axvline(off_diagonal_manhattan.mean() + off_diagonal_manhattan.std(), color='purple', linestyle='-', 
            label=f'Mean + Std: {off_diagonal_manhattan.mean() + off_diagonal_manhattan.std():.3f}')
plt.axvline(off_diagonal_manhattan.mean() - off_diagonal_manhattan.std(), color='purple', linestyle='-', 
            label=f'Mean - Std: {off_diagonal_manhattan.mean() - off_diagonal_manhattan.std():.3f}')
plt.xlabel('Manhattan Distance')
plt.ylabel('Frequency')
plt.title('Distribution of Manhattan Distances\n(Off-diagonal values)')
plt.legend()
plt.grid(True, alpha=0.3)

# 11. Box plot of Manhattan Distances
plt.subplot(3, 4, 11)
plt.boxplot(off_diagonal_manhattan, labels=['Manhattan Distance'])
plt.ylabel('Distance Value')
plt.title('Manhattan Distance\nDistribution Summary')
plt.grid(True, alpha=0.3)

# 12. Manhattan Distance Percentile Plot
plt.subplot(3, 4, 12)
percentile_values_man = [np.percentile(off_diagonal_manhattan, p) for p in percentiles_range]
plt.plot(percentiles_range, percentile_values_man, linewidth=2, color='orange')
plt.xlabel('Percentile')
plt.ylabel('Manhattan Distance')
plt.title('Manhattan Distance\nPercentile Curve')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('similarity_distance_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Create a comparative analysis figure
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Comparative box plots
data_for_comparison = [off_diagonal_cosine, off_diagonal_euclidean, off_diagonal_manhattan]
labels_comparison = ['Cosine\nSimilarity', 'Euclidean\nDistance', 'Manhattan\nDistance']
ax1.boxplot(data_for_comparison, labels=labels_comparison)
ax1.set_title('Comparative Distribution Summary')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# 2. Overlaid histograms (normalized)
ax2.hist(off_diagonal_cosine, bins=30, alpha=0.5, label='Cosine Similarity', density=True)
ax2.hist(off_diagonal_euclidean, bins=30, alpha=0.5, label='Euclidean Distance', density=True)
ax2.hist(off_diagonal_manhattan, bins=30, alpha=0.5, label='Manhattan Distance', density=True)
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.set_title('Normalized Distribution Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Summary statistics comparison
metrics_names = ['Cosine Sim', 'Euclidean Dist', 'Manhattan Dist']
means = [off_diagonal_cosine.mean(), off_diagonal_euclidean.mean(), off_diagonal_manhattan.mean()]
stds = [off_diagonal_cosine.std(), off_diagonal_euclidean.std(), off_diagonal_manhattan.std()]

x_pos = np.arange(len(metrics_names))
ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
ax3.set_xlabel('Metric Type')
ax3.set_ylabel('Mean Value')
ax3.set_title('Mean Values with Standard Deviation')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics_names)
ax3.grid(True, alpha=0.3)

# 4. Correlation analysis (if applicable)
# Create a correlation matrix between the three metrics using a subset
n_sample_corr = min(1000, len(off_diagonal_cosine))
indices_corr = np.random.choice(len(off_diagonal_cosine), n_sample_corr, replace=False)

corr_data = np.column_stack([
    off_diagonal_cosine[indices_corr],
    off_diagonal_euclidean[indices_corr], 
    off_diagonal_manhattan[indices_corr]
])
corr_matrix = np.corrcoef(corr_data.T)

im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
ax4.set_xticks(range(3))
ax4.set_yticks(range(3))
ax4.set_xticklabels(metrics_names)
ax4.set_yticklabels(metrics_names)
ax4.set_title('Correlation Matrix Between Metrics')

# Add correlation values to the plot
for i in range(3):
    for j in range(3):
        ax4.text(j, i, f'{corr_matrix[i, j]:.3f}', 
                ha='center', va='center', fontweight='bold')

plt.colorbar(im, ax=ax4, label='Correlation Coefficient')

plt.tight_layout()
plt.savefig('comparative_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics table
print("\n=== SUMMARY STATISTICS TABLE ===")
print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
print("-" * 80)
print(f"{'Cosine Similarity':<20} {off_diagonal_cosine.mean():<12.6f} {off_diagonal_cosine.std():<12.6f} {off_diagonal_cosine.min():<12.6f} {off_diagonal_cosine.max():<12.6f} {np.median(off_diagonal_cosine):<12.6f}")
print(f"{'Euclidean Distance':<20} {off_diagonal_euclidean.mean():<12.6f} {off_diagonal_euclidean.std():<12.6f} {off_diagonal_euclidean.min():<12.6f} {off_diagonal_euclidean.max():<12.6f} {np.median(off_diagonal_euclidean):<12.6f}")
print(f"{'Manhattan Distance':<20} {off_diagonal_manhattan.mean():<12.6f} {off_diagonal_manhattan.std():<12.6f} {off_diagonal_manhattan.min():<12.6f} {off_diagonal_manhattan.max():<12.6f} {np.median(off_diagonal_manhattan):<12.6f}")

print("\n=== VISUALIZATION FILES SAVED ===")
print("- similarity_distance_analysis.pdf: Comprehensive analysis of all three metrics")
print("- comparative_analysis.pdf: Side-by-side comparison of the metrics")
