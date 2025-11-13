#!/usr/bin/env python3
import argparse as ap
from typing import Optional
from scipy.sparse import issparse, csc_matrix


def add_arguments_preprocess_infer(parser: ap.ArgumentParser):
    """Add arguments for the preprocess_infer subcommand."""
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output preprocessed AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--control-condition",
        type=str,
        required=True,
        help="Control condition identifier (e.g., \"[('DMSO_TF', 0.0, 'uM')]\")",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        required=True,
        help="Column name containing perturbation information (e.g., 'drugname_drugconc')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        required=False,
        help="obsm key to use/replace instead of X (e.g., 'X_pca')",
    )


def _fast_row_reindex_matrix(X, row_indexer):
    """
    Return X[row_indexer] efficiently for dense or sparse matrices.
    Also convert CSC->CSR once to speed up row gathering on sparse matrices.
    """
    if issparse(X):
        # Row-wise fancy indexing is faster on CSR than CSC.
        if isinstance(X, csc_matrix):
            print("Converting X from CSC to CSR for faster row indexing...")
            X = X.tocsr(copy=True)
        return X[row_indexer]
    else:
        return X[row_indexer, :]


def run_tx_preprocess_infer(
    adata_path: str,
    output_path: str,
    control_condition: str,
    pert_col: str,
    seed: int = 42,
    embed_key: Optional[str] = None,
):
    """
    Preprocess inference data by replacing perturbed cells with control expression.

    This creates a 'control template' where all non-control cells receive expression
    sampled (with replacement) from control cells, while keeping original annotations.
    """
    import logging

    import anndata as ad
    import numpy as np
    # tqdm removed from the hot path; the main speed-up is vectorization, not progress bars.

    logger = logging.getLogger(__name__)

    print(f"Loading AnnData from {adata_path}")
    adata = ad.read_h5ad(adata_path)

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)
    print(f"Set random seed to {seed}")

    # Validate columns/keys upfront
    if pert_col not in adata.obs.columns:
        raise KeyError(f"Column '{pert_col}' not found in adata.obs")

    if embed_key is not None and embed_key not in adata.obsm:
        raise KeyError(f"obsm key '{embed_key}' not found in adata.obsm")

    # Identify control cells
    print(f"Identifying control cells with condition: {control_condition!r}")
    # Use .values to avoid pandas alignment overhead
    col_values = adata.obs[pert_col].values
    control_mask = col_values == control_condition
    control_indices = np.flatnonzero(control_mask)

    print(f"Found {control_indices.size} control cells out of {adata.n_obs} total cells")
    if control_indices.size == 0:
        raise ValueError(f"No control cells found with condition '{control_condition}' in column '{pert_col}'")

    # Compute unique perturbations for logging (no heavy loop per perturbation)
    if hasattr(adata.obs[pert_col], "cat"):
        unique_perturbations = adata.obs[pert_col].cat.categories
    else:
        unique_perturbations = np.unique(col_values)

    non_control_perturbations = [p for p in unique_perturbations if p != control_condition]
    n_non_control_cells = int((~control_mask).sum())

    print(f"Processing {len(non_control_perturbations)} non-control perturbations")

    # Build a source index for every row: control rows map to themselves,
    # non-control rows map to randomly sampled control rows.
    source_idx = np.arange(adata.n_obs, dtype=np.int64)
    if n_non_control_cells > 0:
        sampled_controls = rng.choice(control_indices, size=n_non_control_cells, replace=True)
        source_idx[~control_mask] = sampled_controls

    # Create a copy to preserve original object structure/metadata (matches original behavior)
    adata_modified = adata.copy()

    # Replace data in a single, vectorized operation
    if embed_key is not None:
        emb = adata.obsm[embed_key]
        # emb is expected to be a dense 2D array-like
        adata_modified.obsm[embed_key] = emb[source_idx]
        total_replaced_cells = n_non_control_cells
    else:
        X = adata.X
        adata_modified.X = _fast_row_reindex_matrix(X, source_idx)
        total_replaced_cells = n_non_control_cells

    print(f"Replacement complete! Replaced expression in {total_replaced_cells} cells")
    print(f"Control cells ({control_indices.size}) retain their original expression")

    # Summary log
    print("=" * 60)
    print("PREPROCESSING SUMMARY:")
    print(f"  - Input: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"  - Control condition: {control_condition!r}")
    print(f"  - Control cells: {control_indices.size} (unchanged)")
    print(f"  - Perturbed cells: {total_replaced_cells} (replaced with control expression)")
    print(f"  - Perturbations processed: {len(non_control_perturbations)}")
    if embed_key is not None:
        print(f"  - Using obsm key: {embed_key}")
    else:
        print("  - Using expression matrix (X)")
    print("")
    print("USAGE:")
    print("  The output file contains cells with control expression but original")
    print("  perturbation annotations. When passed through state_transition inference,")
    print("  the model will apply perturbation effects to simulate the original data.")
    print("  Compare: state_transition(output) ≈ original_input")
    print("=" * 60)

    print(f"Saving preprocessed data to {output_path}")
    # Writing can still be I/O-bound; the heavy compute path is now vectorized.
    adata_modified.write_h5ad(output_path)
    print("Preprocessing complete!")
