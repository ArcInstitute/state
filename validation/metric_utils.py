"""Utility functions for computing metrics."""

import os
import logging
import scipy
import time
import torch
import warnings

import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import multiprocessing as mp

from collections.abc import Iterator
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from typing import Optional
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, auc
from sklearn.metrics.pairwise import cosine_similarity
from adjustpy import adjust
from scipy.stats import ranksums
from tqdm import tqdm

from models.base import DecoderInterface

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_dense(X):
    if scipy.sparse.issparse(X):
        return np.asarray(X.todense())
    else:
        return X


def compute_jaccard(pred, true, ctrl, pred_ctrl):
    """Computes Jaccard score between x and y."""
    set1 = set(pred)
    set2 = set(true)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_wasserstein(pred, true, control, pred_ctrl, epsilon=0.1):
    """Computes transport between x and y via Sinkhorn algorithm."""
    # Compute cost
    # should be preprocessed outside of this function
    # pred = pred.detach().cpu().numpy()
    # true = true.detach().cpu().numpy()

    geom_pred_true = pointcloud.PointCloud(pred, true, epsilon=epsilon)
    ot_prob = linear_problem.LinearProblem(geom_pred_true)

    # Solve ot problem
    solver = sinkhorn.Sinkhorn()
    out_pred_true = solver(ot_prob)

    # Return regularized ot cost
    # converting to float bc returning as np array
    return float(out_pred_true.reg_ot_cost)


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd(pred, true, ctrl, pred_ctrl, gammas=None):
    """Computes MMD between x and y using RBF kernel with multiple gammas."""
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda gamma: safe_mmd(pred, true, gamma), gammas)))


def compute_mse(pred, true, ctrl, pred_ctrl):
    """Computes mean squared error between x and y."""
    return mean_squared_error(pred, true)


def compute_pearson(pred, true, ctrl, pred_ctrl):
    """Computes average pairwise Pearson correlation between batches x and y."""
    return np.mean([np.corrcoef(pred[i], true[j])[0, 1] for i in range(len(pred)) for j in range(len(true)) if i == j])


def compute_pearson_delta(pred, true, ctrl, pred_ctrl):
    """Computes Pearson correlation between pred and true after subtracting control."""
    return pearsonr(pred.mean(0) - ctrl.mean(0), true.mean(0) - ctrl.mean(0))[0]


def compute_pearson_delta_separate_controls(pred, true, ctrl, pred_ctrl):
    """
    Computes Pearson correlation between pred and true after subtracting pred and true controls
    If we have learned a function f(control cells, pert) -> pred, then:
        pred = f(ctrl, pert)
        pred_ctrl = f(ctrl, ctrl_pert)
    true are the real perturbed cells and ctrl are the real control cells
    """
    return pearsonr(pred.mean(0) - pred_ctrl.mean(0), true.mean(0) - ctrl.mean(0))[0]


def compute_pearson_delta_batched(batched_means, weightings):
    """Computes Pearson correlation between pred and true after subtracting control."""
    ## TODO should we use the weightings here

    pred_de = pd.DataFrame(batched_means["pert_pred"] - batched_means["ctrl_pred"])
    pred_de = pred_de.dropna().mean(0)

    true_de = pd.DataFrame(batched_means["pert_true"] - batched_means["ctrl_true"])
    true_de = true_de.dropna().mean(0)

    return pearsonr(pred_de, true_de)[0]


def compute_cosine_similarity(pred, true, ctrl, pred_ctrl):
    """Computes cosine similarity between predictions and the true centroid in a vectorized manner."""
    centroid_true = np.mean(true, axis=0, keepdims=True)  # shape (1, n_features)
    pred_norms = np.linalg.norm(pred, axis=1)             # shape (n_samples,)
    true_norm = np.linalg.norm(centroid_true)
    # Compute dot products in one go
    dot_products = np.dot(pred, centroid_true.T).flatten() # shape (n_samples,)
    cos_sim_scores = dot_products / (pred_norms * true_norm)
    return np.mean(cos_sim_scores)


def compute_cosine_similarity_v2(pred, true, ctrl, pred_ctrl):
    pred_mean = pred.mean(0)
    true_mean = true.mean(0)
    cosine_similarity = np.dot(pred_mean, true_mean) / (np.linalg.norm(pred_mean) * np.linalg.norm(true_mean))
    return cosine_similarity.item()


def get_top_k_de(subset_adata, k):
    return [x[0] for x in subset_adata.uns["rank_genes_groups"]["names"][:k]]


def jaccard_score(x, y):
    set1 = set(x)
    set2 = set(y)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def compute_gene_overlap_pert(pert_pred, pert_true, ctrl_true, ctrl_pred, gene_names=None, k=50):
    # TODO this is too slow

    pert_true = sc.AnnData(pert_true)
    pert_true.obs["condition"] = "pert"
    ctrl_true = sc.AnnData(ctrl_true)
    ctrl_true.obs["condition"] = "ctrl"
    adata_true = sc.concat([pert_true, ctrl_true])
    adata_true.obs = adata_true.obs.reset_index()
    sc.tl.rank_genes_groups(adata_true, groupby="condition")
    true_de = get_top_k_de(adata_true, k)

    pert_pred = sc.AnnData(pert_pred)
    pert_pred.obs["condition"] = "pert"
    ctrl_pred = sc.AnnData(ctrl_pred)
    ctrl_pred.obs["condition"] = "ctrl"
    adata_pred = sc.concat([pert_pred, ctrl_pred])
    adata_pred.obs = adata_pred.obs.reset_index()
    sc.tl.rank_genes_groups(adata_pred, groupby="condition")
    pred_de = get_top_k_de(adata_pred, k)

    overlap = len(set(true_de).intersection(set(pred_de))) / k

    return overlap


def compute_gene_overlap_cross_pert(DE_true, DE_pred, control_pert="non-targeting", k=None, topk=None):
    """
    Compute the fraction of overlapping genes for each perturbation between DE_true and DE_pred.
    
    The overlap is calculated as the number of genes (ignoring NaNs) that appear in both DE_true and 
    DE_pred divided by the appropriate denominator:
    - If k is None and topk is None, this function measures recall of true DE genes, and the denominator 
    is the set of all true genes
    - If k is not None, this function measures topk overlap, and the denominator is min(true DE genes, k)    
    - If topk is not None, this function measure precision at k and the denominator is min(pred DE genes, topk)
    
    Parameters
    ----------
    DE_true : pd.DataFrame
        DataFrame of true differential expression genes where each row corresponds to a perturbation.
    DE_pred : pd.DataFrame
        DataFrame of predicted differential expression genes with the same index as DE_true.
    control_pert : str, optional
        Name of the control perturbation to ignore (default is "non-targeting").
    k : int, optional
        Number of top genes to consider from DE_pred (if more than k genes are present). 
        If k is none, this function measures recall
        
    Returns
    -------
    dict
        A dictionary with perturbation names as keys and the computed overlap fractions as values.
    """

    if k is not None and topk is not None:
        raise ValueError("Please provide only one of k or topk, not both.")

    all_overlaps = {}
    
    # Loop over each perturbation in the predicted DataFrame.
    for c in DE_pred.index:
        # Skip control perturbation or if the perturbation is not in DE_true.
        if c == control_pert or c not in DE_true.index:
            continue
        
        try:
            # Convert the row values to lists while filtering out NaNs.
            true_genes = [gene for gene in DE_true.loc[c].values if pd.notnull(gene)]
            pred_genes = [gene for gene in DE_pred.loc[c].values if pd.notnull(gene)]
            
            # Use only the top k predicted genes (if available).

            if k is not None:
                pred_genes = pred_genes[:k]
                true_genes = true_genes[:k]
            
            elif topk is not None:
                pred_genes = pred_genes[:topk]
            
            # If there are no non-NaN genes in DE_true, skip the computation.
            if not true_genes:
                continue
            
            # Compute the overlap: number of genes common to both, divided by the number of true genes.
            if topk is not None:
                overlap =  len(set(true_genes).intersection(pred_genes)) / len(pred_genes)
            else:
                overlap = len(set(true_genes).intersection(pred_genes)) / len(true_genes)
            all_overlaps[c] = overlap
        except Exception as e:
            # In case of an unexpected error for a given perturbation, we skip it.
            continue

    # Print the average overlap across all perturbations.
    if all_overlaps:
        print("Average DE: ", np.mean(list(all_overlaps.values())))
    else:
        print("No overlaps computed.")

    return all_overlaps

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def compute_sig_gene_counts(DE_true_sig, DE_pred_sig, pert_list):
    """
    For each perturbation in pert_list, count the number of non-NaN significantly DE genes 
    in both the true and predicted DataFrames.
    
    Parameters
    ----------
    DE_true_sig : pd.DataFrame
        Pivoted DataFrame of true significant DE genes (perturbations as index, ranked genes as columns).
    DE_pred_sig : pd.DataFrame
        Pivoted DataFrame of predicted significant DE genes (perturbations as index, ranked genes as columns).
    pert_list : list
        List of perturbation identifiers to compute counts for.
        
    Returns
    -------
    tuple of dicts
        Two dictionaries:
          - true_counts: mapping perturbation -> count of non-NaN true DE genes.
          - pred_counts: mapping perturbation -> count of non-NaN predicted DE genes.
    """
    true_counts = {}
    pred_counts = {}
    
    for p in pert_list:
        # Count non-NaN entries for true significant genes
        if p in DE_true_sig.index:
            true_counts[p] = DE_true_sig.loc[p].dropna().shape[0]
        else:
            true_counts[p] = 0
        
        # Count non-NaN entries for predicted significant genes
        if p in DE_pred_sig.index:
            pred_counts[p] = DE_pred_sig.loc[p].dropna().shape[0]
        else:
            pred_counts[p] = 0
    
    return true_counts, pred_counts


def compute_sig_gene_spearman(true_counts, pred_counts, pert_list):
    """
    Compute the Spearman correlation between the number of significantly DE genes in true and predicted data.
    
    Parameters
    ----------
    true_counts : dict
        Dictionary mapping perturbation -> count of non-NaN true DE genes.
    pred_counts : dict
        Dictionary mapping perturbation -> count of non-NaN predicted DE genes.
    pert_list : list
        List of perturbation identifiers to compute the correlation on.
    
    Returns
    -------
    float
        Spearman correlation coefficient.
    """
    # Prepare lists of counts for the perturbations in the provided order.
    true_list = [true_counts.get(p, 0) for p in pert_list]
    pred_list = [pred_counts.get(p, 0) for p in pert_list]
    
    # Compute and return the Spearman correlation coefficient
    corr, _ = spearmanr(true_list, pred_list)
    return corr


def compute_directionality_agreement(DE_true_df, DE_pred_df, pert_list, p_val_thresh=0.05):
    """
    For each perturbation in pert_list, compute the fraction of overlapping genes that have
    matching fold change directionality, only considering significantly DE genes in the true data 
    (i.e. those with p_value < 0.05).

    Parameters
    ----------
    DE_true_df : pd.DataFrame
        DataFrame of true DE results. Should include columns: 'target', 'feature', 'fold_change',
        and 'p_value'.
    DE_pred_df : pd.DataFrame
        DataFrame of predicted DE results. Should include columns: 'target', 'feature', and 
        'fold_change'.
    pert_list : list
        List of perturbation names to compute the metric for.
    
    Returns
    -------
    dict
        Dictionary mapping each perturbation to the fraction of overlapping genes (from the 
        significantly DE true data) for which the sign of the fold change matches between 
        DE_true_df and DE_pred_df. If no overlapping genes exist, NaN is recorded.
    """
    
    directionality_matches = {}

    for p in pert_list:
        # Filter and merge the relevant rows for each perturbation
        true_sub = DE_true_df[(DE_true_df['target'] == p) & (DE_true_df['p_value'] < p_val_thresh)]
        pred_sub = DE_pred_df[DE_pred_df['target'] == p]

        if true_sub.empty or pred_sub.empty:
            directionality_matches[p] = np.nan
            continue

        # Merge on feature (gene)
        merged = pd.merge(true_sub[['feature', 'fold_change']], 
                          pred_sub[['feature', 'fold_change']], 
                          on='feature', 
                          suffixes=('_true', '_pred'))

        if merged.empty:
            directionality_matches[p] = np.nan
            continue

        # Vectorized sign comparison: True if both FCs are < 1 or both >= 1
        same_direction = ((merged['fold_change_true'] < 1) == (merged['fold_change_pred'] < 1))
        directionality_matches[p] = same_direction.mean()

    return directionality_matches



def compute_DE_for_truth_and_pred(
    adata_real_ct,  # ground truth in gene space or latent space
    adata_pred_ct,  # predicted data in gene space or latent space
    control_pert: str,
    pert_col: str = "gene",
    n_top_genes: int = 2000,  # HVG cutoff
    k_de_genes: int = 50,
    output_space: str = "gene",
    model_decoder: Optional[DecoderInterface] = None,
    outdir=None,
):
    """
    Unify logic for computing DE from both the ground truth and model predictions.

    Steps:
        1) It is assumed that adata_real_ct is always in gene space, unless the model was explicitly trained in latent space only.

        2) If a decoder is present, adata_pred_ct is assumed to be in UCE latent space and DEGs are computed using UCE logprobs.

        3) For each: run rank_genes_groups to get top k DE genes vs. control_pert. Return the final
           lists as a DataFrame: row=pert, columns=ranked DE genes in descending order.

    Returns:
        (DE_true, DE_pred) as two data frames (index=pert_name, columns=top_k DE genes).
    """

    if 'DMSO_TF' in control_pert: # only for tahoe dataset for now
        # attach var names to adata_real_ct, which consists of HVGs
        # real data is either just hvgs, or full transcriptome
        if adata_real_ct.X.shape[1] == 2000: # tahoe hvg
            hvg_gene_names = np.load('/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy', allow_pickle=True)
            adata_real_ct.var.index = hvg_gene_names
        else: # tahoe transcriptome
            gene_names = np.load('/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_names.npy', allow_pickle=True)
            adata_real_ct.var.index = gene_names
    else: # genetic dataset
        if 'non-targeting' in control_pert and adata_real_ct.X.shape[1] == 3609: # I hate this. replogle hvg
            temp = ad.read_h5ad('/large_storage/ctc/userspace/aadduri/datasets/hvg/replogle/jurkat.h5')
            adata_real_ct.var.index = temp.var.index.values
        # add more for replogle gene space, and other datasets, etc
    
    # 2) HVG filtering (applied to each or to the combined data).
    # This happens to the ground truth regardless of input space.
    adata_real_hvg = adata_real_ct
    adata_real_hvg.obs["pert_name"] = pd.Categorical(adata_real_hvg.obs["pert_name"])
    start_true = time.time()
    DE_true_fc, DE_true_pval, DE_true_pval_fc, DE_true_sig_genes, DE_true_df = parallel_compute_de(adata_real_hvg, control_pert, pert_col, k_de_genes, outdir=outdir, split='real')
    print("Time taken for true DE: ", time.time() - start_true)

    start_pred = time.time()
    if model_decoder is not None:
        # This needs to be update to compute all 3 types of de
        pred_de_genes_ranked = model_decoder.compute_de_genes(
            adata_pred_ct,
            pert_col=pert_col,
            control_pert=control_pert,
            genes=adata_real_hvg.var.index.values,
        )
        DE_pred_fc = pred_de_genes_ranked.iloc[:, :k_de_genes]
        DE_pred_pval = DE_pred_fc
        DE_pred_pval_fc = DE_pred_fc
    else:
        # assume adata_pred_ct is already in gene space
        adata_pred_ct.var.index = adata_real_ct.var.index
        adata_pred_gene = adata_pred_ct
        adata_pred_gene.obs.index = adata_pred_gene.obs.index.astype(str)
        adata_pred_hvg = adata_pred_gene
        adata_pred_hvg.obs["pert_name"] = pd.Categorical(adata_real_hvg.obs["pert_name"])
        DE_pred_fc, DE_pred_pval, DE_pred_pval_fc, DE_pred_sig_genes, DE_pred_df = parallel_compute_de(adata_pred_hvg, control_pert, pert_col, k_de_genes, outdir=outdir, split='pred')
    print("Time taken for predicted DE: ", time.time() - start_pred)

    # return DE_true, DE_pred
    return DE_true_fc, DE_pred_fc, DE_true_pval, DE_pred_pval, DE_true_pval_fc, DE_pred_pval_fc, DE_true_sig_genes, DE_pred_sig_genes, DE_true_df, DE_pred_df

def compute_DE_pca(adata_pred, gene_names, pert_col, control_pert, k=50, transform=None):
    """
    Compute differential expression in gene space after decoding from PCA.
    """
    if transform is None:
        raise ValueError("PCA transform required for compute_DE_pca")

    # Decode predictions back to gene space
    decoded_pred = transform.decode(adata_pred.X)

    # Create new anndata with decoded predictions
    decoded_adata = ad.AnnData(X=decoded_pred.cpu().numpy(), obs=adata_pred.obs, var=pd.DataFrame(index=gene_names))

    # Compute DE using scanpy
    sc.tl.rank_genes_groups(decoded_adata, groupby=pert_col, reference=control_pert, rankby_abs=True, n_genes=k)

    # Extract results
    de_genes = pd.DataFrame(decoded_adata.uns["rank_genes_groups"]["names"])
    return de_genes.T

def compute_downstream_DE_metrics(target_gene, pred_df, true_df, p_value_threshold):
    true_target = true_df[true_df['target'] == target_gene]
    pred_target = pred_df[pred_df['target'] == target_gene]

    sig_genes = true_target[true_target['p_value'] < p_value_threshold]['feature'].tolist()

    result = {
        'target_gene': target_gene,
        'spearman': np.nan,
        'pr_auc': np.nan,
        'roc_auc': np.nan,
        'significant_genes_count': len(sig_genes)
    }

    if not sig_genes:
        return result

    true_filtered = true_target[true_target['feature'].isin(sig_genes)]
    pred_filtered = pred_target[pred_target['feature'].isin(sig_genes)]

    merged = pd.merge(
        true_filtered[['feature', 'fold_change']],
        pred_filtered[['feature', 'fold_change']],
        on='feature',
        suffixes=('_true', '_pred')
    )

    # Assume no change for NaN
    merged['fold_change_pred'] = merged['fold_change_pred'].fillna(1)

    if len(merged) > 1:
        try:
            result['spearman'] = spearmanr(merged['fold_change_true'], merged['fold_change_pred'])[0]
        except:
            pass

    true_target = true_target.assign(label=(true_target['p_value'] < p_value_threshold).astype(int))
    merged_curve = pd.merge(true_target[['feature', 'label']], pred_target[['feature', 'p_value']], on='feature')

    if len(merged_curve) > 1 and 0 < merged_curve['label'].sum() < len(merged_curve):
        y_true = merged_curve['label']
        y_scores = -np.log10(merged_curve['p_value'])

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            result['pr_auc'] = auc(recall, precision)
            result['roc_auc'] = auc(fpr, tpr)
        except:
            pass

    return result


def compute_mean_perturbation_effect(adata, pert_col="gene", ctrl_pert="non-targeting"):
    adata_df = adata.to_df()
    adata_df["pert"] = adata.obs[pert_col].values
    mean_df = adata_df.groupby("pert").mean()
    mean_pert_effect = np.abs(mean_df - mean_df.loc[ctrl_pert])
    return mean_pert_effect


def compute_perturbation_id_score(adata_pred, adata_real, pert_col="gene", ctrl_pert="non-targeting"):
    ## Given a specific perturbation, identify which model-predicted
    # perturbation is most similar to it

    ## Compute true mean perturbation effect
    mean_real_effect = compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    mean_pred_effect = compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    all_perts = mean_real_effect.index.values
    ## For each true perturbation effect, find the nearest neighbor predicted
    # perturbation effect

    pred_perts = []
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values
        pred_effects = mean_pred_effect.values
        pred_pert = all_perts[np.argmax(cosine_similarity(real_effect.reshape(1, -1), pred_effects))]
        pred_perts.append(pred_pert)

    accuracy_score = np.sum(pred_perts == all_perts) / len(all_perts)

    return accuracy_score

def compute_perturbation_ranking_score(adata_pred, adata_real, pert_col="gene", ctrl_pert="non-targeting"):
    ## Compute true mean perturbation effect
    mean_real_effect = compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    mean_pred_effect = compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    all_perts = mean_real_effect.index.values

    ranks = []

    ## For each true perturbation effect, compute similarity to all predicted effects
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values.reshape(1, -1)
        pred_effects = mean_pred_effect.values

        # Compute cosine similarities between the real effect and all predicted effects
        similarities = cosine_similarity(real_effect, pred_effects).flatten()

        # Get the rank of the true perturbation based on similarity
        true_pert_index = np.where(all_perts == pert)[0][0]
        sorted_indices = np.argsort(similarities)[::-1]  # Sort in descending order of similarity
        rank_of_true_pert = np.where(sorted_indices == true_pert_index)[0][0]  # 1-based rank

        ranks.append(rank_of_true_pert)

    ## mean normalized rank
    mean_rank = np.mean(ranks) / len(all_perts)

    return mean_rank

# PASTED FROM ARC SEQ #

def parallel_compute_de(adata_gene, control_pert, pert_col, k, outdir=None, split='real'):
    """
    Compute differential expression using parallel_differential_expression,
    returns two DataFrames: one sorted by fold change and one by p-value
    
    Parameters
    ----------
    adata_gene : AnnData
        The annotated data matrix with gene expression data
    control_pert : str
        Name of the control perturbation to use as reference
    pert_col : str
        Column in adata_gene.obs that contains perturbation information
    k : int
        Number of top genes to return for each perturbation
        
    Returns
    -------
    tuple of pd.DataFrame
        Two DataFrames with rows as perturbations and columns as top K genes,
        one sorted by fold change and one by p-value
    """
    import time
    import pandas as pd
    
    # Start timer
    start_time = time.time()
    
    # Filter groups to only include those with more than 1 cell
    group_counts = adata_gene.obs[pert_col].value_counts()
    valid_groups = group_counts[group_counts > 1].index.tolist()
    adata_gene = adata_gene[adata_gene.obs[pert_col].isin(valid_groups)]
    
    # Make sure the control perturbation is included in the valid groups
    if control_pert not in valid_groups:
        raise ValueError(f"Control perturbation '{control_pert}' has fewer than 2 cells")
    
    # Run parallel differential expression
    de_results = parallel_differential_expression(
        adata=adata_gene,
        groups=valid_groups,
        reference=control_pert,
        groupby_key=pert_col,
        num_workers=120,  # Adjust based on your system
        batch_size=1000  # Adjust based on memory constraints
    )

    celltype = adata_gene.obs["celltype_name"].values[0]

    # # Save out the de results
    if outdir is not None:
        outfile = os.path.join(outdir, f"{celltype}_{split}_de_results_{control_pert}.csv")
        # if it doesn't already exist, write it out
        if not os.path.exists(outfile):
            de_results.to_csv(outfile, index=False)
        logger.info(f"Saved DE results to {outfile}")
    # #
    
    logger.info(f"Time taken for parallel_differential_expression: {time.time() - start_time:.2f}s")
    
    # Get top DE genes sorted by fold change
    de_genes_fc = vectorized_topk_de(de_results, control_pert, 200, sort_by='abs_log_fold_change')
    
    # Get top DE genes sorted by p-value
    de_genes_pval = vectorized_topk_de(de_results, control_pert, 200, sort_by='p_value')

    de_genes_pval_fc = vectorized_sig_genes_fc_sort(de_results, control_pert, k=None, pvalue_threshold=0.05)

    de_genes_sig = vectorized_sig_genes_fc_sort(de_results, control_pert, k=None, pvalue_threshold=0.05)
    
    return de_genes_fc, de_genes_pval, de_genes_pval_fc, de_genes_sig, de_results

def _build_shared_matrix(
    data: np.ndarray,
) -> tuple[str, tuple[int, int], np.dtype]:
    """Create a shared memory matrix from a numpy array."""
    shared_matrix = SharedMemory(create=True, size=data.nbytes)
    matrix = np.ndarray(data.shape, dtype=data.dtype, buffer=shared_matrix.buf)
    matrix[:] = data
    return shared_matrix.name, data.shape, data.dtype

def _conclude_shared_memory(name: str):
    """Close and unlink a shared memory."""
    shm = SharedMemory(name=name)
    shm.close()
    shm.unlink()

def _combinations_generator(
    target_masks: dict[str, np.ndarray],
    var_indices: dict[str, int],
    reference: str,
    target_list: list[str],
    feature_list: list[str],
) -> Iterator[tuple]:
    """Generate all combinations of target genes and features."""
    for target in target_list:
        for feature in feature_list:
            yield (
                target_masks[target],
                target_masks[reference],
                var_indices[feature],
                target,
                reference,
                feature,
            )

def _batch_generator(
    combinations: Iterator[tuple],
    batch_size: int,
    num_combinations: int,
) -> Iterator[list[tuple]]:
    """Generate batches of combinations."""
    for _i in range(0, num_combinations, batch_size):
        subset = []
        for _ in range(batch_size):
            try:
                subset.append(next(combinations))
            except StopIteration:
                break
        yield subset

def _process_target_batch_shm(
    batch_tasks: list[tuple],
    shm_name: str,
    shape: tuple[int, int],
    dtype: np.dtype,
) -> list[dict[str, float]]:
    """Process a batch of target gene and feature combinations.

    This is the function that is parallelized across multiple workers.
    """
    # Open shared memory once for the batch
    existing_shm = SharedMemory(name=shm_name)
    matrix = np.ndarray(shape=shape, dtype=dtype, buffer=existing_shm.buf)

    results = []
    for (
        target_mask,
        reference_mask,
        var_index,
        target_name,
        reference_name,
        var_name,
    ) in batch_tasks:
        if target_name == reference_name:
            continue

        x_tgt = matrix[target_mask, var_index]
        x_ref = matrix[reference_mask, var_index]

        μ_tgt = np.mean(x_tgt)
        μ_ref = np.mean(x_ref)

        fc = _fold_change(μ_tgt, μ_ref)
        pcc = _percent_change(μ_tgt, μ_ref)
        rs_result = ranksums(x_tgt, x_ref)

        results.append(
            {
                "target": target_name,
                "reference": reference_name,
                "feature": var_name,
                "target_mean": μ_tgt,
                "reference_mean": μ_ref,
                "percent_change": pcc,
                "fold_change": fc,
                "p_value": rs_result.pvalue,
                "statistic": rs_result.statistic,
            }
        )

    existing_shm.close()
    return results

def parallel_differential_expression(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    num_workers: int = 1,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Calculate differential expression between groups of cells.

    Parameters
    ----------
    adata: ad.AnnData
        Annotated data matrix containing gene expression data
    groups: list[str], optional
        List of groups to compare, defaults to None which compares all groups
    reference: str, optional
        Reference group to compare against, defaults to "non-targeting"
    groupby_key: str, optional
        Key in `adata.obs` to group by, defaults to "target_gene"
    num_workers: int
        Number of workers to use for parallel processing, defaults to 1
    batch_size: int
        Number of combinations to process in each batch, defaults to 100

    Returns
    -------
    pd.DataFrame containing differential expression results for each group and feature
    """
    unique_targets = adata.obs[groupby_key].unique()
    if groups is not None:
        unique_targets = [
            target
            for target in unique_targets
            if target in groups or target == reference
        ]
    unique_features = adata.var.index

    # Precompute the number of combinations and batches
    n_combinations = len(unique_targets) * len(unique_features)
    n_batches = n_combinations // batch_size + 1

    # Precompute masks for each target gene
    logger.info("Precomputing masks for each target gene")
    target_masks = {
        target: _get_obs_mask(
            adata=adata, target_name=target, variable_name=groupby_key
        )
        for target in tqdm(unique_targets, desc="Identifying target masks")
    }

    # Precompute variable index for each feature
    logger.info("Precomputing variable indices for each feature")
    var_indices = {
        feature: idx
        for idx, feature in enumerate(
            tqdm(unique_features, desc="Identifying variable indices")
        )
    }

    # Isolate the data matrix from the AnnData object
    logger.info("Creating shared memory memory matrix for parallel computing")
    (shm_name, shape, dtype) = _build_shared_matrix(data=adata.X.toarray())

    logger.info(f"Creating generator of all combinations: N={n_combinations}")
    combinations = _combinations_generator(
        target_masks=target_masks,
        var_indices=var_indices,
        reference=reference,
        target_list=unique_targets,
        feature_list=unique_features,
    )
    logger.info(f"Creating generator of all batches: N={n_batches}")
    batches = _batch_generator(
        combinations=combinations,
        batch_size=batch_size,
        num_combinations=n_combinations,
    )

    # Partial function for parallel processing
    task_fn = partial(
        _process_target_batch_shm,
        shm_name=shm_name,
        shape=shape,
        dtype=dtype,
    )

    logger.info("Initializing parallel processing pool")
    with mp.Pool(num_workers) as pool:
        logger.info("Processing batches")
        batch_results = list(
            tqdm(
                pool.imap(task_fn, batches),
                total=n_batches,
                desc="Processing batches",
            )
        )

    # Flatten results
    logger.info("Flattening results")
    results = [result for batch in batch_results for result in batch]

    # Close shared memory
    logger.info("Closing shared memory pool")
    _conclude_shared_memory(shm_name)

    dataframe = pd.DataFrame(results)
    dataframe["fdr"] = adjust(dataframe["p_value"].values, method="bh")

    return dataframe

def _get_obs_mask(
    adata: ad.AnnData,
    target_name: str,
    variable_name: str = "target_gene",
) -> np.ndarray:
    """Return a boolean mask for a specific target name in the obs variable."""
    return adata.obs[variable_name] == target_name


def _get_var_index(
    adata: ad.AnnData,
    target_gene: str,
) -> int:
    """Return the index of a specific gene in the var variable.

    Raises
    ------
    ValueError
        If the gene is not found in the dataset.
    """
    var_index = np.flatnonzero(adata.var.index == target_gene)
    if len(var_index) == 0:
        raise ValueError(f"Target gene {target_gene} not found in dataset")
    return var_index[0]

def _fold_change(
    μ_tgt: float,
    μ_ref: float,
) -> float:
    """Calculate the fold change between two means."""
    try:
        return μ_tgt / μ_ref
    except ZeroDivisionError:
        return np.nan

def _percent_change(
    μ_tgt: float,
    μ_ref: float,
) -> float:
    """Calculate the percent change between two means."""
    return (μ_tgt - μ_ref) / μ_ref

def vectorized_topk_de(de_results, control_pert, k, sort_by='abs_log_fold_change'):
    """
    Create a DataFrame with top k DE genes for each perturbation sorted by the specified metric.
    
    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame with differential expression results
    control_pert : str
        Name of the control perturbation
    k : int
        Number of top genes to return for each perturbation
    sort_by : str
        Metric to sort by ('abs_log_fold_change' or 'p_value')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rows as perturbations and columns as top K genes
    """
    # Filter out the control perturbation rows
    df = de_results[de_results['target'] != control_pert]
    
    # Compute absolute fold change (if not already computed)
    df['abs_log_fold_change'] = df['fold_change'].abs()

    if df[sort_by].dtype == 'float16':
        df[sort_by] = df[sort_by].astype('float32')
    
    # Sort direction depends on metric (descending for fold change, ascending for p-value)
    ascending = True if sort_by == 'p_value' else False
    
    # Sort the DataFrame by target and the chosen metric
    df_sorted = df.sort_values(['target', sort_by], ascending=[True, ascending])
    
    # For each target, pick the top k rows
    df_sorted['rank'] = df_sorted.groupby('target').cumcount()
    df_topk = df_sorted[df_sorted['rank'] < k]
    
    # Pivot the DataFrame so that rows are targets and columns are the ranked top genes
    de_genes = df_topk.pivot(index='target', columns='rank', values='feature')
    
    # Optionally, sort the columns so that they are in order from 0 to k-1
    de_genes = de_genes.sort_index(axis=1)
    
    return de_genes

def vectorized_sig_genes_fc_sort(de_results, control_pert, k=None, pvalue_threshold=0.05):
    """
    Create a DataFrame with DE genes for each perturbation.

    Two modes:
    - If k is None: returns all genes passing the significance threshold, sorted 
      by descending absolute fold change.
    - Otherwise, returns only the top k genes sorted by descending absolute fold change.
      For perturbations with fewer than k significant genes, the resulting pivot table
      will contain NaNs in the missing positions.

    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame with differential expression results.
    control_pert : str
        Name of the control perturbation to exclude.
    k : int
        Number of top genes to return for each perturbation (used only if return_all is False).
    pvalue_threshold : float, optional
        p-value cutoff for significance (default is 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame with rows as perturbations and columns as the ranked significant genes.
    """
    # Remove control perturbation rows and compute absolute fold change
    df = de_results[de_results['target'] != control_pert].copy()
    df['abs_fold_change'] = df['fold_change'].abs()

    # Convert types if necessary
    if df['abs_fold_change'].dtype == 'float16':
        df['abs_fold_change'] = df['abs_fold_change'].astype('float32')
    if df['p_value'].dtype == 'float16':
        df['p_value'] = df['p_value'].astype('float32')
    
    # Filter for significant genes only
    df_significant = df[df['p_value'] < pvalue_threshold].copy()
    
    # Sort by target and descending absolute fold change
    df_sorted = df_significant.sort_values(['target', 'abs_fold_change'], ascending=[True, False])
    
    # Within each target, assign a rank based on the ordering
    df_sorted['rank'] = df_sorted.groupby('target').cumcount()
    
    # If not returning all, filter to only the top k genes per target
    if k is not None:
        df_sorted = df_sorted[df_sorted['rank'] < k]
    
    # Pivot the results so that rows are targets and columns are ranked genes.
    # Note: If return_all is False, targets with fewer than k genes will have NaNs in missing positions.
    result_df = df_sorted.pivot(index='target', columns='rank', values='feature')
    result_df = result_df.sort_index(axis=1)
    
    return result_df