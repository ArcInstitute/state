import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


adata_real = ad.read_h5ad('/large_storage/ctc/ML/state_sets/replogle_filtered/processed.h5')
#adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered/rpe1/eval_step=56000.ckpt/adata_real.h5ad')
#adata_real = ad.read_h5ad('/large_storage/ctc/datasets/replogle/rpe1_normalized_singlecell_01.h5ad')
#adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered/rpe1/eval_step=56000.ckpt/adata_pred.h5ad')
#adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint_parse/parse_donor_tahoe_best_cs512_2000_state_zeroshot/parse_donor_state_2000/eval_step=112000.ckpt/adata_real.h5ad')
#adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint_redo/tahoe_best_cs256_2000_state/tahoe_best_2000_state/eval_last.ckpt/adata_real.h5ad')

# Choose the grouping column for perturbation
pert_col = 'gene'
cell_type_col = 'cell_type'

# Count the number of cells for each drug and select the bottom 16
drug_counts = adata_real.obs[pert_col][adata_real.obs[cell_type_col] == 'rpe1'].value_counts()
top_drugs = drug_counts.tail(16).index.tolist()

total_num_perts = adata_real.obs[pert_col][adata_real.obs[cell_type_col] == 'rpe1'].nunique()
print(f"Total number of perturbations in the dataset: {total_num_perts}")

num_cells = adata_real.obs[cell_type_col] == 'rpe1'
print(f"Total number of cells in the full dataset: {num_cells}")

perturbations = top_drugs  # Only use these for plotting

# Compute mean per perturbation (mean of all values for all cells in that perturbation)
pert_sums = []
for pert in top_drugs: 
    # mean of all values for all cells in this perturbation
    idx = adata_real.obs[pert_col] == pert
    sum_val = np.array(adata_real.X[idx].sum(axis=1)).flatten()
    pert_sums.append(sum_val)
    print(f"Perturbation: {pert}, Mean: {sum_val.mean():.4f}, Min: {sum_val.min():.4f}, Max: {sum_val.max():.4f}, Std: {sum_val.std():.4f}")
# Print the distribution of means
# pert_sums = np.array(pert_sums)
# print("\nSummary statistics for sum of all values per perturbation:")
# print(f"Min: {pert_sums.min():.4f}")
# print(f"Max: {pert_sums.max():.4f}")
# print(f"Mean: {pert_sums.mean():.4f}")
# print(f"Std: {pert_sums.std():.4f}")

# 1. Per-cell total counts by perturbation
fig, axes = plt.subplots(1, len(perturbations), figsize=(6 * len(perturbations), 4), sharey=True)
for i, pert in enumerate(perturbations):
    idx = adata_real.obs[pert_col] == pert
    cell_counts = np.array(adata_real.X[idx].sum(axis=1)).flatten()
    ax = axes[i] if len(perturbations) > 1 else axes
    sns.histplot(cell_counts, bins=50, kde=True, ax=ax)
    ax.set_title(f'{pert_col}: {pert}')
    ax.set_xlabel('Total Counts per Cell')
    ax.set_ylabel('Number of Cells')
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand_dist_replogle_filtered/cell_counts_distribution_by_perturbation_bottom16.png')
plt.show()

# 2. Per-gene total counts by perturbation
fig, axes = plt.subplots(1, len(perturbations), figsize=(6 * len(perturbations), 4), sharey=True)
for i, pert in enumerate(perturbations):
    idx = adata_real.obs[pert_col] == pert
    gene_counts = np.array(adata_real.X[idx].sum(axis=0)).flatten()
    ax = axes[i] if len(perturbations) > 1 else axes
    sns.histplot(gene_counts, bins=50, kde=True, ax=ax)
    ax.set_title(f'{pert_col}: {pert}')
    ax.set_xlabel('Total Counts per Gene')
    ax.set_ylabel('Number of Genes')
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand_dist_replogle_filtered/gene_counts_distribution_by_perturbation_bottom16.png')
plt.show()

# # 3. Per-gene total counts unlog transformed by perturbation
# fig, axes = plt.subplots(1, len(perturbations), figsize=(6 * len(perturbations), 4), sharey=True)
# for i, pert in enumerate(perturbations):
#     idx = adata_real.obs[pert_col] == pert
#     gene_counts_unlog = np.exp(np.array(adata_real.X[idx].sum(axis=0) - 1).flatten())
#     ax = axes[i] if len(perturbations) > 1 else axes
#     sns.histplot(gene_counts_unlog, bins=50, kde=True, ax=ax)
#     ax.set_title(f'{pert_col}: {pert}')
#     ax.set_xlabel('Total Counts per Gene (unlog)')
#     ax.set_ylabel('Number of Genes')
# plt.tight_layout()
# plt.savefig('/home/dhruvgautam/state-sets/rand_dist/gene_counts_distribution_unlog_by_perturbation.png')
# plt.show()

# 4. Per-gene mean counts by perturbation
fig, axes = plt.subplots(1, len(perturbations), figsize=(6 * len(perturbations), 4), sharey=True)
for i, pert in enumerate(perturbations):
    idx = adata_real.obs[pert_col] == pert
    gene_means = np.array(adata_real.X[idx].mean(axis=0)).flatten()
    ax = axes[i] if len(perturbations) > 1 else axes
    sns.histplot(gene_means, bins=50, kde=True, ax=ax)
    ax.set_title(f'{pert_col}: {pert}')
    ax.set_xlabel('Mean Counts per Gene')
    ax.set_ylabel('Number of Genes')
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand_dist_replogle_filtered/gene_means_distribution_by_perturbation_bottom16.png')
plt.show()

# 5. Per-gene mean counts unlog transformed by perturbation
fig, axes = plt.subplots(1, len(perturbations), figsize=(6 * len(perturbations), 4), sharey=True)
for i, pert in enumerate(perturbations):
    idx = adata_real.obs[pert_col] == pert
    gene_means_unlog = np.exp(np.array(adata_real.X[idx].mean(axis=0) - 1).flatten())
    min_val = np.floor(gene_means_unlog.min() * 10) / 10
    max_val = 2.0
    bins = np.arange(min_val, max_val + 0.05, 0.05)
    ax = axes[i] if len(perturbations) > 1 else axes
    sns.histplot(gene_means_unlog, bins=bins, kde=True, ax=ax)
    ax.set_title(f'{pert_col}: {pert}')
    ax.set_xlabel('Mean Counts per Gene (unlog)')
    ax.set_ylabel('Number of Genes')
    ax.set_xticks(np.arange(min_val, max_val + 0.05, 0.05))
    ax.set_xlim(min_val, max_val)
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand_dist_replogle_filtered/gene_means_distribution_unlog_by_perturbation_bottom16.png')
plt.show()

# Inspect the columns in adata_real.obs
# print('adata_real.obs columns:', adata_real.obs.columns.tolist())

# # Group by 'cytokine' and 'cell_type' and print group sizes
# grouped = adata_real.obs.groupby(['cytokine', 'cell_type'])
# print('Group sizes by cytokine and cell_type:')
# print(grouped.size())

