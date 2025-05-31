import scipy
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.interpolate import interp1d
from collections import defaultdict
from validation.metric_utils import (
    to_dense,
    compute_mae,
    compute_mse,
    compute_wasserstein,
    compute_mmd,
    compute_pearson_delta,
    compute_pearson_delta_separate_controls,
    compute_cosine_similarity,
    compute_gene_overlap_cross_pert,
    compute_DE_for_truth_and_pred,
    compute_perturbation_ranking_score,
    compute_pearson_delta_batched,
    compute_clustering_agreement,
    compute_downstream_DE_metrics,
    compute_sig_gene_counts,
    compute_sig_gene_spearman,
    compute_directionality_agreement,
)
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import pearsonr
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils import time_it

# setup logger
import logging
import os
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)


## TODO: Redefine this to take as input the mapping information
def compute_metrics(
    adata_pred,  # predictions in uce space
    adata_real,  # true values in uce space
    adata_pred_gene=None,  # predictions in gene space
    adata_real_gene=None,  # true values in gene space
    embed_key=None,
    include_dist_metrics=False,
    control_pert="non-targeting",
    pert_col="pert_name",
    celltype_col="celltype_name",
    batch_col="gem_group",
    model_loc=None,
    DE_metric_flag=True,
    class_score_flag=True,
    output_space="gene",
    decoder=None,
    shared_perts=None,
    outdir=None,  # output directory to store raw de results
):
    pred_celltype_pert_dict = adata_pred.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()
    real_celltype_pert_dict = adata_real.obs.groupby(celltype_col)[pert_col].agg(set).to_dict()

    assert set(pred_celltype_pert_dict.keys()) == set(real_celltype_pert_dict.keys()), (
        "Pred and real adatas do not have identical celltypes"
    )
    for celltype in pred_celltype_pert_dict.keys():
        assert pred_celltype_pert_dict[celltype] == real_celltype_pert_dict[celltype], (
            f"Pred and real adatas have different set of perturbations for {celltype}"
        )

    # compute metrics
    if type(adata_real.obs.index) != pd.core.indexes.range.RangeIndex:
        adata_real.obs = adata_real.obs.reset_index()
    adata_pred.obs = adata_pred.obs.reset_index()

    metrics = {}
    for celltype in pred_celltype_pert_dict:
        with time_it(f"compute_metrics_cell_type_{celltype}"):
            metrics[celltype] = defaultdict(list)

            adata_pred_control = get_samples_by_pert_and_celltype(
                adata_pred,
                pert=control_pert,
                celltype=celltype,
                pert_col=pert_col,
                celltype_col=celltype_col,
            )

            adata_real_control = get_samples_by_pert_and_celltype(
                adata_real,
                pert=control_pert,
                celltype=celltype,
                pert_col=pert_col,
                celltype_col=celltype_col,
            )

            adata_pred_gene_control = None
            adata_real_gene_control = None
            if adata_pred_gene is not None and adata_real_gene is not None:
                adata_pred_gene_control = get_samples_by_pert_and_celltype(
                    adata_pred_gene,
                    pert=control_pert,
                    celltype=celltype,
                    pert_col=pert_col,
                    celltype_col=celltype_col,
                )

                adata_real_gene_control = get_samples_by_pert_and_celltype(
                    adata_real_gene,
                    pert=control_pert,
                    celltype=celltype,
                    pert_col=pert_col,
                    celltype_col=celltype_col,
                )

            # only evaluate on perturbations that are shared between the train and test sets
            if shared_perts:
                all_perts = shared_perts & pred_celltype_pert_dict[celltype]
            else:
                all_perts = pred_celltype_pert_dict[celltype]
            only_perts = [x for x in all_perts if x != control_pert]

            pred_groups = adata_pred.obs[adata_pred.obs[celltype_col] == celltype].groupby(pert_col).indices
            real_groups = adata_real.obs[adata_real.obs[celltype_col] == celltype].groupby(pert_col).indices

            # Get sample indices for count space if gene data is provided
            pred_gene_groups = None
            real_gene_groups = None
            if adata_pred_gene is not None and adata_real_gene is not None:
                pred_gene_groups = (
                    adata_pred_gene.obs[adata_pred_gene.obs[celltype_col] == celltype].groupby(pert_col).indices
                )
                real_gene_groups = (
                    adata_real_gene.obs[adata_real_gene.obs[celltype_col] == celltype].groupby(pert_col).indices
                )

            # use tqdm to track progress
            for pert in tqdm(all_perts, desc=f"Computing metrics for {celltype}", leave=False):
                try:
                    if pert == control_pert:
                        continue

                    with time_it(f"compute_metrics_pert_{pert}"):
                        pred_idx = pred_groups.get(pert, [])
                        true_idx = real_groups.get(pert, [])
                        if len(pred_idx) == 0 or len(true_idx) == 0:
                            continue

                        # Extract data slices in a vectorized way
                        adata_pred_pert = adata_pred[pred_idx]
                        adata_real_pert = adata_real[true_idx]

                        ## Use softmap to generate artificial control distributions
                        pert_idx = adata_pred_pert.obs.index.astype("int").tolist()

                        adata_pred_control.obs.index = pd.Categorical(adata_pred_control.obs.index)
                        adata_real_control.obs.index = pd.Categorical(adata_real_control.obs.index)

                        ## Get the predictions and true values
                        pert_preds = to_dense(adata_pred_pert.X)
                        pert_true = to_dense(adata_real_pert.X)
                        control_true = to_dense(adata_real_control.X)
                        control_preds = to_dense(adata_pred_control.X)

                        ## If matrix is sparse convert to dense
                        try:
                            pert_true = pert_true.toarray()
                            control_true = control_true.toarray()
                        except:
                            pass

                        ## Compute metrics across all batches for a specific perturbation
                        curr_metrics = _compute_metrics_dict(
                            pert_preds,
                            pert_true,
                            control_true,
                            control_preds,
                            suffix="cell_type",
                            include_dist_metrics=include_dist_metrics,
                        )

                        # Add metrics for counts space if gene data is provided
                        if (
                            adata_pred_gene is not None
                            and adata_real_gene is not None
                            and pred_gene_groups is not None
                            and real_gene_groups is not None
                        ):
                            # Get indices for counts space
                            pred_gene_idx = pred_gene_groups.get(pert, [])
                            true_gene_idx = real_gene_groups.get(pert, [])

                            if len(pred_gene_idx) > 0 and len(true_gene_idx) > 0:
                                # Extract data slices for counts space
                                adata_pred_gene_pert = adata_pred_gene[pred_gene_idx]
                                adata_real_gene_pert = adata_real_gene[true_gene_idx]

                                # Get the predictions and true values for counts space
                                pert_preds_gene = to_dense(adata_pred_gene_pert.X)
                                pert_true_gene = to_dense(adata_real_gene_pert.X)
                                control_true_gene = to_dense(adata_real_gene_control.X)
                                control_preds_gene = to_dense(adata_pred_gene_control.X)

                                # If matrix is sparse convert to dense
                                try:
                                    pert_true_gene = pert_true_gene.toarray()
                                    control_true_gene = control_true_gene.toarray()
                                except:
                                    pass

                                # Compute metrics for counts space
                                gene_metrics = _compute_metrics_dict(
                                    pert_preds_gene,
                                    pert_true_gene,
                                    control_true_gene,
                                    control_preds_gene,
                                    suffix="cell_type_counts",
                                    include_dist_metrics=include_dist_metrics,
                                )

                                # Add counts metrics to current metrics
                                curr_metrics.update(gene_metrics)

                        metrics[celltype]["pert"].append(pert)
                        for k, v in curr_metrics.items():
                            metrics[celltype][k].append(v)

                except:
                    print(f"Failed for {celltype} {pert}")
                    pass

            adata_real_ct = adata_real[adata_real.obs[celltype_col] == celltype]
            adata_pred_ct = adata_pred[adata_pred.obs[celltype_col] == celltype]

            # filter adata_real and adata_pred to only include control and shared perturbations
            assert control_pert in all_perts
            all_perts = list(all_perts)
            adata_real_ct.obs[pert_col] = pd.Categorical(adata_real_ct.obs[pert_col])
            adata_real_ct = adata_real_ct[adata_real_ct.obs[pert_col].isin(all_perts)]

            adata_pred_ct.obs[pert_col] = pd.Categorical(adata_pred_ct.obs[pert_col])
            adata_pred_ct = adata_pred_ct[adata_pred_ct.obs[pert_col].isin(all_perts)]

            # gene level metrics may not be available if the output_space was specified to be latent
            adata_real_gene_ct = None
            adata_pred_gene_ct = None
            if adata_real_gene is not None:
                logger.info(f"Using gene expression data for true {celltype}")
                adata_real_gene_ct = adata_real_gene[adata_real_gene.obs[celltype_col] == celltype]
                adata_real_gene_ct.obs[pert_col] = pd.Categorical(adata_real_gene_ct.obs[pert_col])
                adata_real_gene_ct = adata_real_gene_ct[adata_real_gene_ct.obs[pert_col].isin(all_perts)]
            if adata_pred_gene is not None:
                logger.info(f"Using gene expression data for pred {celltype}")
                adata_pred_gene_ct = adata_pred_gene[adata_pred_gene.obs[celltype_col] == celltype]
                adata_pred_gene_ct.obs[pert_col] = pd.Categorical(adata_pred_gene_ct.obs[pert_col])
                adata_pred_gene_ct = adata_pred_gene_ct[adata_pred_gene_ct.obs[pert_col].isin(all_perts)]

            if DE_metric_flag:
                ## Compute differential expression at the full adata level for speed

                # 2) Actually compute DE for both truth & pred
                logger.info(f"Computing DE for 50 genes")
                (
                    DE_true_fc,
                    DE_pred_fc,
                    DE_true_pval,
                    DE_pred_pval,
                    DE_true_pval_fc,
                    DE_pred_pval_fc,
                    DE_true_sig_genes,
                    DE_pred_sig_genes,
                    DE_true_df,
                    DE_pred_df,
                ) = compute_DE_for_truth_and_pred(
                    adata_real_gene_ct or adata_real_ct,
                    adata_pred_gene_ct or adata_pred_ct,
                    control_pert=control_pert,
                    pert_col=pert_col,
                    n_top_genes=2000,  # default HVG
                    output_space=output_space,
                    model_decoder=decoder,
                    outdir=outdir,
                )

                # compute pearson for only significant genes:
                pearson_sig = []
                # for pert in metrics[celltype]["pert"]:
                # use tqdm
                gene_counts = adata_real_gene_ct or adata_real_ct
                pred_gene_counts = adata_pred_gene_ct or adata_pred_ct
                for pert in tqdm(
                    metrics[celltype]["pert"],
                    desc=f"Computing pearson for sig genes across perts in {celltype}",
                    leave=False,
                ):
                    # subset adata for this pert and for control
                    real_pert = gene_counts[gene_counts.obs[pert_col] == pert]
                    pred_pert = pred_gene_counts[pred_gene_counts.obs[pert_col] == pert]
                    real_ctrl = gene_counts[gene_counts.obs[pert_col] == control_pert]
                    pred_ctrl = pred_gene_counts[pred_gene_counts.obs[pert_col] == control_pert]

                    # convert to dense arrays
                    X_true = to_dense(real_pert.X)
                    X_pred = to_dense(pred_pert.X)
                    X_ctrl_true = to_dense(real_ctrl.X)
                    X_ctrl_pred = to_dense(pred_ctrl.X)

                    # get the list of significant genes for this perturbation
                    if pert in DE_true_sig_genes.index:
                        sig_genes = DE_true_sig_genes.loc[pert].dropna().tolist()
                    else:
                        sig_genes = []

                    if not sig_genes:
                        pearson_sig.append(np.nan)
                        continue

                    # map gene names → column indices in the AnnData
                    var_index = list(gene_counts.var.index)
                    gene_cols = [var_index.index(g) for g in sig_genes if g in var_index]
                    if not gene_cols:
                        pearson_sig.append(np.nan)
                        continue

                    # slice out only the sig-gene columns
                    true_sub = X_true[:, gene_cols]
                    pred_sub = X_pred[:, gene_cols]
                    ctrl_true_sub = X_ctrl_true[:, gene_cols]
                    ctrl_pred_sub = X_ctrl_pred[:, gene_cols]

                    # compute the delta‐Pearson
                    # pearson_sig.append(compute_pearson_delta(pred_sub, true_sub, ctrl_true_sub, ctrl_pred_sub))

                metrics[celltype]["pearson_delta_cell_type_sig_genes"] = pearson_sig

                clustering_agreement = compute_clustering_agreement(adata_real, adata_pred, embed_key=None)
                metrics[celltype]["clustering_agreement"] = clustering_agreement

                # Compute overlap for fold change-based DE
                DE_metrics_fc = compute_gene_overlap_cross_pert(DE_true_fc, DE_pred_fc, control_pert=control_pert, k=50)
                metrics[celltype]["DE_fc"] = [DE_metrics_fc.get(p, 0.0) for p in metrics[celltype]["pert"]]
                metrics[celltype]["DE_fc_avg"] = np.mean(list(DE_metrics_fc.values()))

                # Compute overlap for p-value-based DE
                DE_metrics_pval = compute_gene_overlap_cross_pert(
                    DE_true_pval, DE_pred_pval, control_pert=control_pert, k=50
                )
                metrics[celltype]["DE_pval"] = [DE_metrics_pval.get(p, 0.0) for p in metrics[celltype]["pert"]]
                metrics[celltype]["DE_pval_avg"] = np.mean(list(DE_metrics_pval.values()))

                # Compute overlap for fold change-based DE thresholded with p-values
                DE_metrics_pval_fc_50 = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, k=50
                )
                metrics[celltype]["DE_pval_fc_50"] = [
                    DE_metrics_pval_fc_50.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_pval_fc_avg_50"] = np.mean(list(DE_metrics_pval_fc_50.values()))

                DE_metrics_pval_fc_100 = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, k=100
                )
                metrics[celltype]["DE_pval_fc_100"] = [
                    DE_metrics_pval_fc_100.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_pval_fc_avg_100"] = np.mean(list(DE_metrics_pval_fc_100.values()))

                DE_metrics_pval_fc_200 = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, k=200
                )
                metrics[celltype]["DE_pval_fc_200"] = [
                    DE_metrics_pval_fc_200.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_pval_fc_avg_200"] = np.mean(list(DE_metrics_pval_fc_200.values()))

                # Variable k
                DE_metrics_pval_fc_N = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, k=-1
                )
                metrics[celltype]["DE_pval_fc_N"] = [
                    DE_metrics_pval_fc_N.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_pval_fc_avg_N"] = np.mean(list(DE_metrics_pval_fc_N.values()))

                # Compute precision at k for fold change-based DE thresholded with p-values
                DE_metrics_patk_pval_fc_50 = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, topk=50
                )
                metrics[celltype]["DE_patk_pval_fc_50"] = [
                    DE_metrics_patk_pval_fc_50.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_patk_pval_fc_avg_50"] = np.mean(list(DE_metrics_patk_pval_fc_50.values()))

                DE_metrics_patk_pval_fc_100 = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, topk=100
                )
                metrics[celltype]["DE_patk_pval_fc_100"] = [
                    DE_metrics_patk_pval_fc_100.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_patk_pval_fc_avg_100"] = np.mean(list(DE_metrics_patk_pval_fc_100.values()))

                DE_metrics_patk_pval_fc_200 = compute_gene_overlap_cross_pert(
                    DE_true_pval_fc, DE_pred_pval_fc, control_pert=control_pert, topk=200
                )
                metrics[celltype]["DE_patk_pval_fc_200"] = [
                    DE_metrics_patk_pval_fc_200.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_patk_pval_fc_av_200"] = np.mean(list(DE_metrics_patk_pval_fc_200.values()))

                # Compute recall for significant genes
                DE_metrics_sig_genes = compute_gene_overlap_cross_pert(
                    DE_true_sig_genes, DE_pred_sig_genes, control_pert=control_pert
                )
                metrics[celltype]["DE_sig_genes_recall"] = [
                    DE_metrics_sig_genes.get(p, 0.0) for p in metrics[celltype]["pert"]
                ]
                metrics[celltype]["DE_sig_genes_recall_avg"] = np.mean(list(DE_metrics_sig_genes.values()))

                # Record effect sizes
                true_counts, pred_counts = compute_sig_gene_counts(DE_true_sig_genes, DE_pred_sig_genes, only_perts)
                metrics[celltype]["DE_sig_genes_count_true"] = [true_counts.get(p, 0) for p in only_perts]
                metrics[celltype]["DE_sig_genes_count_pred"] = [pred_counts.get(p, 0) for p in only_perts]

                # Compute the Spearman correlation between the counts.
                spearman_corr = compute_sig_gene_spearman(true_counts, pred_counts, only_perts)
                metrics[celltype]["DE_sig_genes_spearman"] = spearman_corr

                # Compute the directionality agreement.
                directionality_agreement = compute_directionality_agreement(DE_true_df, DE_pred_df, only_perts)
                metrics[celltype]["DE_direction_match"] = [directionality_agreement.get(p, np.nan) for p in only_perts]
                metrics[celltype]["DE_direction_match_avg"] = np.nanmean(list(directionality_agreement.values()))

                # Compute clustering overlap

                # Compute the actual top-k gene lists per perturbation
                de_pred_genes_col = []
                de_true_genes_col = []

                for p in metrics[celltype]["pert"]:
                    if p == control_pert:
                        de_pred_genes_col.append("")
                        de_true_genes_col.append("")
                        continue

                    # Retrieve predicted and true DE genes for p, if available
                    if p in DE_pred_pval.index:
                        pred_genes = list(DE_pred_pval.loc[p].values)
                    else:
                        pred_genes = []

                    if p in DE_true_pval.index:
                        true_genes = list(DE_true_pval.loc[p].values)
                    else:
                        true_genes = []

                    # Convert lists to comma-separated strings
                    de_pred_genes_col.append("|".join(pred_genes))
                    de_true_genes_col.append("|".join(true_genes))

                # Store them as new columns
                metrics[celltype]["DE_pred_genes"] = de_pred_genes_col
                metrics[celltype]["DE_true_genes"] = de_true_genes_col

                # Compute additional DE metrics
                print("Computing additional metrics")
                get_downstream_DE_metrics(
                    DE_pred_df, DE_true_df, outdir=outdir, celltype=celltype, n_workers=None, p_value_threshold=0.05
                )

            if class_score_flag:
                ## Compute classification score
                class_score = compute_perturbation_ranking_score(
                    adata_pred_ct,
                    adata_real_ct,
                    pert_col=pert_col,
                    ctrl_pert=control_pert,
                )
                metrics[celltype]["perturbation_id"] = class_score
                metrics[celltype]["perturbation_score"] = 1 - class_score

    try:
        for celltype, stats in metrics.items():
            metrics[celltype] = pd.DataFrame(stats).set_index("pert")
        return metrics
    except Exception as e:
        print(e)
        return metrics


def _compute_metrics_dict(pert_pred, pert_true, ctrl_true, ctrl_pred, suffix="", include_dist_metrics=False):
    metrics = {}
    metrics["mse_" + suffix] = compute_mse(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["mae_" + suffix] = compute_mae(pert_pred, pert_true)
    metrics["pearson_delta_" + suffix] = compute_pearson_delta(pert_pred, pert_true, ctrl_true, ctrl_pred)
    metrics["pearson_delta_sep_ctrls_" + suffix] = compute_pearson_delta_separate_controls(
        pert_pred, pert_true, ctrl_true, ctrl_pred
    )

    metrics["cosine_" + suffix] = compute_cosine_similarity(pert_pred, pert_true, ctrl_true, ctrl_pred)
    if include_dist_metrics:
        # with time_it("compute_wasserstein"):
        #     metrics["wasserstein_" + suffix] = compute_wasserstein(pert_pred, pert_true, ctrl_true, ctrl_pred)
        with time_it("compute_mmd"):
            metrics["mmd_" + suffix] = compute_mmd(pert_pred, pert_true, ctrl_true, ctrl_pred)
    return metrics


def get_samples_by_pert_and_celltype(adata, pert, celltype, pert_col, celltype_col):
    pert_idx = (adata.obs[pert_col] == pert).to_numpy()
    celltype_idx = (adata.obs[celltype_col] == celltype).to_numpy()
    out = adata[pert_idx & celltype_idx]
    return out


def init_worker(global_pred_df, global_true_df):
    global PRED_DF
    global TRUE_DF
    PRED_DF = global_pred_df
    TRUE_DF = global_true_df


def compute_downstream_DE_metrics_parallel(target_gene, p_value_threshold):
    return compute_downstream_DE_metrics(target_gene, PRED_DF, TRUE_DF, p_value_threshold)


def interpolate_curves(curves, x_grid, min_max=(0, 1)):
    """Interpolate a list of curves onto a common x grid."""
    y_values = []
    min_val, max_val = min_max

    for curve in curves:
        x, y = curve

        # Skip if too few points
        if len(x) < 2:
            continue

        # Create interpolation function
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1]))

        # Interpolate y values at the grid points
        interp_y = f(x_grid)

        # Apply bounds
        interp_y = np.clip(interp_y, min_val, max_val)
        y_values.append(interp_y)

    # Convert to numpy array
    if y_values:
        return np.array(y_values)
    return np.array([])


def get_downstream_DE_metrics(DE_pred_df, DE_true_df, outdir, celltype, n_workers=10, p_value_threshold=0.05):
    for df in (DE_pred_df, DE_true_df):
        df["abs_fold_change"] = np.abs(df["fold_change"])
        with np.errstate(divide="ignore"):
            df["log_fold_change"] = np.log10(df["fold_change"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["abs_log_fold_change"] = np.abs(df["log_fold_change"].fillna(0))

    target_genes = DE_true_df["target"].unique()
    os.makedirs(outdir, exist_ok=True)

    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(DE_pred_df, DE_true_df)) as pool:
        func = partial(compute_downstream_DE_metrics_parallel, p_value_threshold=p_value_threshold)
        results = list(tqdm(pool.imap(func, target_genes), total=len(target_genes)))

    # Create the main results DataFrame (without the curve data)
    scalar_results = [
        {k: v for k, v in r.items() if k not in ["recall_raw", "precision_raw", "fpr_raw", "tpr_raw"]} for r in results
    ]
    results_df = pd.DataFrame(scalar_results)

    # Save the main results
    outpath = os.path.join(outdir, f"{celltype}_downstream_de_results.csv")
    results_df.to_csv(outpath, index=False)

    # Get all valid curve data
    pr_curves = [
        (np.array(r["recall_raw"]), np.array(r["precision_raw"]))
        for r in results
        if len(r["recall_raw"]) > 0 and len(r["precision_raw"]) > 0
    ]
    roc_curves = [
        (np.array(r["fpr_raw"]), np.array(r["tpr_raw"]))
        for r in results
        if len(r["fpr_raw"]) > 0 and len(r["tpr_raw"]) > 0
    ]

    # Create common x-axis grids for interpolation
    pr_grid = np.linspace(0, 1, 100)
    roc_grid = np.linspace(0, 1, 100)

    # Interpolate all curves onto the common grid
    pr_interp = interpolate_curves(pr_curves, pr_grid)
    roc_interp = interpolate_curves(roc_curves, roc_grid)

    # Calculate mean and std for each curve type
    pr_mean = np.mean(pr_interp, axis=0) if len(pr_interp) > 0 else np.array([])
    pr_std = np.std(pr_interp, axis=0) if len(pr_interp) > 0 else np.array([])

    roc_mean = np.mean(roc_interp, axis=0) if len(roc_interp) > 0 else np.array([])
    roc_std = np.std(roc_interp, axis=0) if len(roc_interp) > 0 else np.array([])

    # Calculate average AUC values
    mean_pr_auc = auc(pr_grid, pr_mean) if len(pr_mean) > 0 else np.nan
    mean_roc_auc = auc(roc_grid, roc_mean) if len(roc_mean) > 0 else np.nan

    # Save curve data to CSV
    if len(pr_mean) > 0:
        pr_df = pd.DataFrame({"recall": pr_grid, "precision_mean": pr_mean, "precision_std": pr_std})
        pr_df.to_csv(os.path.join(outdir, f"{celltype}_avg_pr_curve.csv"), index=False)

    if len(roc_mean) > 0:
        roc_df = pd.DataFrame({"fpr": roc_grid, "tpr_mean": roc_mean, "tpr_std": roc_std})
        roc_df.to_csv(os.path.join(outdir, f"{celltype}_avg_roc_curve.csv"), index=False)

    # Save mean metrics
    mean_metrics = {
        "mean_aupr": float(mean_pr_auc) if not np.isnan(mean_pr_auc) else None,
        "mean_auroc": float(mean_roc_auc) if not np.isnan(mean_roc_auc) else None,
        "pr_curves_count": len(pr_curves),
        "roc_curves_count": len(roc_curves),
    }

    with open(os.path.join(outdir, f"{celltype}_avg_curve_metrics.json"), "w") as f:
        json.dump(mean_metrics, f)

    # Plot PR curve with shaded std dev
    if len(pr_mean) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pr_grid, pr_mean, "b-", label=f"Mean AUPR = {mean_pr_auc:.3f} (n={len(pr_curves)})")
        ax.fill_between(
            pr_grid,
            np.maximum(0, pr_mean - pr_std),
            np.minimum(1, pr_mean + pr_std),
            color="b",
            alpha=0.2,
            label="±1 std dev",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Average Precision-Recall Curve - {celltype}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"{celltype}_avg_pr_curve.svg"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Plot ROC curve with shaded std dev
    if len(roc_mean) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(roc_grid, roc_mean, "r-", label=f"Mean AUROC = {mean_roc_auc:.3f} (n={len(roc_curves)})")
        ax.fill_between(
            roc_grid,
            np.maximum(0, roc_mean - roc_std),
            np.minimum(1, roc_mean + roc_std),
            color="r",
            alpha=0.2,
            label="±1 std dev",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Diagonal line
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Average ROC Curve - {celltype}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"{celltype}_avg_roc_curve.svg"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Log completion message
    logger.info(f"Completed downstream DE metrics for {celltype}, results saved to {outdir}")

    return results_df


def get_batched_mean(X, batches):
    if scipy.sparse.issparse(X):
        df = pd.DataFrame(X.todense())
    else:
        df = pd.DataFrame(X)

    df["batch"] = batches
    return df.groupby("batch").mean(numeric_only=True)
