import argparse as ap
import os


def add_arguments_eval(parser: ap.ArgumentParser):
    """Add arguments for embedding evaluation CLI."""
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint file")
    parser.add_argument("--adata", required=True, help="Path to AnnData file")
    parser.add_argument(
        "--config",
        required=False,
        help=("Path to configuration override. If omitted, uses the config embedded in the checkpoint."),
    )
    parser.add_argument("--pert-col", default="gene", help="Column name for perturbation labels (default: gene)")
    parser.add_argument(
        "--control-pert", default="non-targeting", help="Control perturbation label (default: non-targeting)"
    )
    parser.add_argument("--gene-column", default="gene_name", help="Column name for gene names (default: gene_name)")
    parser.add_argument("--batch-size", type=int, help="Batch size for model inference (overrides config default)")
    parser.add_argument(
        "--protein-embeddings",
        required=False,
        help=(
            "Path to protein embeddings override (.pt). If omitted, uses embeddings packaged in the checkpoint, or the path from config as fallback."
        ),
    )


def run_emb_eval(args):
    """
    Run embedding evaluation.
    """
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import torch
    from tqdm import tqdm
    from omegaconf import OmegaConf

    def load_config_override(config_path: str | None = None):
        """Load a config override from YAML if provided, else None."""
        if config_path and os.path.exists(config_path):
            return OmegaConf.load(config_path)
        return None

    from ...emb.utils import compute_gene_overlap_cross_pert, get_precision_config
    from ...emb.data import create_dataloader
    from ...emb.inference import Inference

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Loading AnnData: {args.adata}")
    print(f"Perturbation column: {args.pert_col}")
    print(f"Control perturbation: {args.control_pert}")

    if args.config:
        print(f"Using config override: {args.config}")
    else:
        print("No config override provided; will use config embedded in checkpoint")

    # Load configuration override if given; otherwise let Inference load from the checkpoint
    cfg = load_config_override(args.config)

    # Load AnnData
    adata = sc.read_h5ad(args.adata)
    print(f"Loaded AnnData with shape: {adata.shape}")

    # Create inference object and load model
    print("Creating inference object and loading model...")

    # Resolve protein embeddings: explicit override -> use; else let Inference load from checkpoint/config
    protein_embeds = None
    if args.protein_embeddings:
        try:
            protein_embeds = torch.load(args.protein_embeddings, weights_only=False, map_location="cpu")
            print(f"Using protein embeddings override: {args.protein_embeddings}")
        except Exception as e:
            print(f"Error loading protein embeddings override: {e}")
            raise

    inference = Inference(cfg=cfg, protein_embeds=protein_embeds)
    inference.load_model(args.checkpoint)  # populates cfg from checkpoint if cfg was None
    # Ensure cfg is set for downstream usage
    cfg = inference.model.cfg if inference.model is not None else cfg
    # Override batch size if provided
    if args.batch_size and cfg is not None:
        cfg.model.batch_size = args.batch_size
        if inference.model is not None:
            inference.model.update_config(cfg)
        print(f"Using batch size: {args.batch_size}")
    adata = inference._convert_to_csr(adata)

    model = inference.model

    # Type assertion to help linter understand model is not None
    assert model is not None, "Model failed to load from checkpoint"

    print("Model loaded successfully")

    # Create dataloader
    print("Creating dataloader...")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    precision = get_precision_config(device_type=device_type)
    dataloader = create_dataloader(
        cfg,
        adata=adata,
        adata_name="eval_dataset",
        shuffle=False,
        gene_column=args.gene_column,
        precision=precision,
    )

    # Compute embeddings and predictions
    print("Computing embeddings and predictions...")
    emb_batches = []
    ds_emb_batches = []
    logprob_batches = []

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=precision):
            for batch in tqdm(dataloader, desc="Processing batches"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Compute embeddings
                _, _, _, emb, ds_emb = model._compute_embedding_for_batch(batch)

                # Get gene embeddings
                try:
                    gene_embeds = model.get_gene_embedding(adata.var.index)
                except:
                    gene_embeds = model.get_gene_embedding(adata.var["gene_symbols"])

                # Handle dataset embeddings
                if hasattr(model, "dataset_token") and model.dataset_token is not None:
                    ds_emb = model.dataset_embedder(ds_emb)

                # Store embeddings
                if emb is not None:
                    emb_batches.append(emb.detach().cpu().float().numpy())
                if ds_emb is not None:
                    ds_emb_batches.append(ds_emb.detach().cpu().float().numpy())

                # Resize batch and decode
                task_counts = None
                Y = batch[2]
                nan_y = Y.masked_fill(Y == 0, float("nan"))[:, : cfg.dataset.P + cfg.dataset.N]
                task_counts = torch.nanmean(nan_y, dim=1) if cfg.model.rda else None

                # Ensure task_counts is on the same device as other tensors
                if task_counts is not None:
                    task_counts = task_counts.to(model.device)

                merged_embs = model.__class__.resize_batch(emb, gene_embeds, task_counts=task_counts, ds_emb=ds_emb)
                logprobs_batch = model.binary_decoder(merged_embs)
                logprobs_batch = logprobs_batch.detach().cpu().float().numpy()
                logprob_batches.append(logprobs_batch.squeeze())

    # Combine batches
    logprob_batches = np.vstack(logprob_batches)
    emb_combined = np.vstack(emb_batches)
    ds_emb_combined = np.vstack(ds_emb_batches)
    adata.obsm["X_emb"] = np.concatenate([emb_combined, ds_emb_combined], axis=-1)

    # Create predictions DataFrame
    probs_df = pd.DataFrame(logprob_batches)
    probs_df[args.pert_col] = adata.obs[args.pert_col].values

    # Get top-k genes for each perturbation
    k = cfg.validations.diff_exp.top_k_rank
    probs_df = probs_df.groupby(args.pert_col).mean()
    ctrl = probs_df.loc[args.control_pert].values
    pert_effects = np.abs(probs_df - ctrl)
    top_k_indices = np.argsort(pert_effects.values, axis=1)[:, -k:][:, ::-1]
    top_k_genes = np.array(adata.var.index)[top_k_indices]
    pred_de_genes = pd.DataFrame(top_k_genes)
    pred_de_genes.index = pert_effects.index.values

    print(f"Predicted DEGs shape: {pred_de_genes.shape}")

    # Compute ground truth DEGs for ALL genes (for ROC/PR curves)
    print("Computing ground truth DEGs for all genes...")
    adata_copy = adata.copy()  # Don't modify original adata
    sc.pp.log1p(adata_copy)

    # First compute for top k genes (for overlap metric)
    sc.tl.rank_genes_groups(
        adata_copy,
        groupby=args.pert_col,
        reference=args.control_pert,
        rankby_abs=True,
        n_genes=k,
        method=cfg.validations.diff_exp.method,
        use_raw=False,
    )
    true_de_genes = pd.DataFrame(adata_copy.uns["rank_genes_groups"]["names"])
    true_de_genes = true_de_genes.T

    print(f"Ground truth DEGs shape: {true_de_genes.shape}")

    # Compute overlap metrics
    print("Computing gene overlap metrics...")
    de_metrics = compute_gene_overlap_cross_pert(pred_de_genes, true_de_genes, control_pert=args.control_pert, k=k)

    # Now compute for ALL genes (for ROC/PR curves)
    print("Computing statistical tests for all genes...")
    sc.tl.rank_genes_groups(
        adata_copy,
        groupby=args.pert_col,
        reference=args.control_pert,
        rankby_abs=True,
        n_genes=adata_copy.n_vars,  # All genes
        method=cfg.validations.diff_exp.method,
        use_raw=False,
    )

    # Compute ROC and PR curves per perturbation with correct gene alignment
    print("Computing ROC and PR curves per perturbation...")
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    from scipy.stats import sem

    roc_curves = []
    pr_curves = []
    roc_aucs = []
    pr_aucs = []

    # Get ground truth results from scanpy
    names_df = pd.DataFrame(adata_copy.uns["rank_genes_groups"]["names"])
    pvals_df = pd.DataFrame(adata_copy.uns["rank_genes_groups"]["pvals_adj"])

    # Get gene order from original adata
    gene_order = adata.var.index.tolist()

    for pert in pert_effects.index:
        if pert == args.control_pert or pert not in names_df.columns:
            continue

        # Get predicted scores (in original gene order)
        pred_scores = pert_effects.loc[pert].values

        # Get ground truth results for this perturbation (proper alignment)
        pert_col_idx = names_df.columns.get_loc(pert)
        pert_names = names_df.iloc[:, pert_col_idx].values  # Gene names ordered by significance
        pert_pvals = pvals_df.iloc[:, pert_col_idx].values  # P-values in same order

        # Create a mapping from gene name to p-value
        gene_to_pval = dict(zip(pert_names, pert_pvals))

        # Create p-values in the same order as predicted scores (original gene order)
        aligned_pvals = []
        aligned_pred_scores = []

        for i, gene in enumerate(gene_order):
            if gene in gene_to_pval:
                aligned_pvals.append(gene_to_pval[gene])
                aligned_pred_scores.append(pred_scores[i])
            else:
                # If gene not in statistical test results, assign p-value of 1.0 (not significant)
                aligned_pvals.append(1.0)
                aligned_pred_scores.append(pred_scores[i])

        aligned_pvals = np.array(aligned_pvals)
        aligned_pred_scores = np.array(aligned_pred_scores)

        # Create binary labels
        true_labels = (aligned_pvals < 0.05).astype(int)

        # Skip if all labels are the same
        if len(np.unique(true_labels)) < 2:
            continue

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(true_labels, aligned_pred_scores)
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))
        roc_aucs.append(roc_auc)

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(true_labels, aligned_pred_scores)
        pr_auc = auc(recall, precision)
        pr_curves.append((precision, recall))
        pr_aucs.append(pr_auc)

    # Compute and report AUC metrics
    if roc_curves:
        print(f"\nROC AUC: {np.mean(roc_aucs):.4f} ± {sem(roc_aucs):.4f}")
        print(f"PR AUC: {np.mean(pr_aucs):.4f} ± {sem(pr_aucs):.4f}")
    else:
        print("No valid ROC/PR curves could be computed (insufficient variation in labels)")

    # Print overlap results
    mean_overlap = np.array(list(de_metrics.values())).mean()
    print("\nOverlap Results:")
    print(f"Mean gene overlap: {mean_overlap:.4f}")
    print(f"Number of perturbations evaluated: {len(de_metrics)}")

    return de_metrics, mean_overlap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    add_arguments_eval(parser)
    args = parser.parse_args()
    run_emb_eval(args)
