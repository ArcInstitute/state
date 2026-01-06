import argparse as ap


def add_arguments_preprocess_train(parser: ap.ArgumentParser):
    """Add arguments for the preprocess_train subcommand."""
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
        "--num_hvgs",
        type=int,
        required=True,
        help="Number of highly variable genes to select",
    )


def run_tx_preprocess_train(adata_path: str, output_path: str, num_hvgs: int):
    """
    Preprocess training data by normalizing, log-transforming, and selecting highly variable genes.
    Stores HVG names in .uns["X_hvg_var_names"] for downstream mapping.

    Args:
        adata_path: Path to input AnnData file
        output_path: Path to save preprocessed AnnData file
        num_hvgs: Number of highly variable genes to select
    """
    import logging

    import anndata as ad
    import numpy as np
    import scanpy as sc

    from state.tx.constants import HVG_OBSM_KEY, HVG_VAR_NAMES_KEY

    logger = logging.getLogger(__name__)

    logger.info(f"Loading AnnData from {adata_path}")
    adata = ad.read_h5ad(adata_path)

    logger.info("Normalizing total counts per cell")
    sc.pp.normalize_total(adata)

    logger.info("Applying log1p transformation")
    sc.pp.log1p(adata)

    logger.info(f"Finding top {num_hvgs} highly variable genes")
    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvgs)

    logger.info(f"Storing highly variable genes in .obsm['{HVG_OBSM_KEY}']")
    adata.obsm[HVG_OBSM_KEY] = adata[:, adata.var.highly_variable].X.toarray()

    # Store HVG names alongside X_hvg for downstream gene mapping.
    hvg_gene_names = adata.var_names[adata.var.highly_variable].tolist()
    adata.uns[HVG_VAR_NAMES_KEY] = np.array(hvg_gene_names, dtype=object)
    logger.info(f"Stored {len(hvg_gene_names)} HVG names in adata.uns['{HVG_VAR_NAMES_KEY}']")

    logger.info(f"Saving preprocessed data to {output_path}")
    adata.write_h5ad(output_path)

    logger.info(f"Preprocessing complete. Selected {adata.var.highly_variable.sum()} highly variable genes.")
