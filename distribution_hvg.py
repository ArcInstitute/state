import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered/rpe1/eval_step=56000.ckpt/adata_real.h5ad')
#adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered/rpe1/eval_step=56000.ckpt/adata_pred.h5ad')
adata_real = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/preprint_parse/parse_donor_tahoe_best_cs512_2000_state_zeroshot/parse_donor_state_2000/eval_step=112000.ckpt/adata_real.h5ad')


# Visualize the distribution of counts
# Per-cell total counts
cell_counts = np.array(adata_real.X.sum(axis=1)).flatten()
plt.figure(figsize=(8, 4))
sns.histplot(cell_counts, bins=50, kde=True)
plt.title('Distribution of Total Counts per Cell')
plt.xlabel('Total Counts per Cell')
plt.ylabel('Number of Cells')
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand/cell_counts_distribution.png')
plt.show()

# Per-gene total counts
gene_counts = np.array(adata_real.X.sum(axis=0)).flatten()
plt.figure(figsize=(8, 4))
sns.histplot(gene_counts, bins=50, kde=True)
plt.title('Distribution of Total Counts per Gene')
plt.xlabel('Total Counts per Gene')
plt.ylabel('Number of Genes')
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand/gene_counts_distribution.png')
plt.show()

# # Per-gene total counts unlog transformed
# gene_counts_unlog = np.exp(np.array(adata_real.X.sum(axis=0) - 1).flatten())
# plt.figure(figsize=(8, 4))
# sns.histplot(gene_counts_unlog, bins=50, kde=True)
# plt.title('Distribution of Total Counts per Gene (unlog)')
# plt.xlabel('Total Counts per Gene (unlog)')
# plt.ylabel('Number of Genes')
# plt.tight_layout()
# plt.savefig('/home/dhruvgautam/state-sets/rand/gene_counts_distribution_unlog.png')
# plt.show()

# per-gene mean counts
gene_means = np.array(adata_real.X.mean(axis=0)).flatten()
plt.figure(figsize=(8, 4))
sns.histplot(gene_means, bins=50, kde=True)
plt.title('Distribution of Mean Counts per Gene')
plt.xlabel('Mean Counts per Gene')
plt.ylabel('Number of Genes')
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand/gene_means_distribution.png')
plt.show()
# un log transformed per-gene mean counts
gene_means_unlog = np.exp(np.array(adata_real.X.mean(axis=0) - 1).flatten())
plt.figure(figsize=(8, 4))
# Set bin edges to be at 0.1 intervals, with max capped at 2.0
min_val = np.floor(gene_means_unlog.min() * 10) / 10
max_val = 2.0
bins = np.arange(min_val, max_val + 0.05, 0.05)
sns.histplot(gene_means_unlog, bins=bins, kde=True)
plt.title('Distribution of Mean Counts per Gene (unlog)')
plt.xlabel('Mean Counts per Gene (unlog)')
plt.ylabel('Number of Genes')
plt.xticks(np.arange(min_val, max_val + 0.05, 0.05))
plt.xlim(min_val, max_val)
plt.tight_layout()
plt.savefig('/home/dhruvgautam/state-sets/rand/gene_means_distribution_unlog.png')
plt.show()

