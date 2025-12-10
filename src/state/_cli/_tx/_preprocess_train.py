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

    Args:
        adata_path: Path to input AnnData file
        output_path: Path to save preprocessed AnnData file
        num_hvgs: Number of highly variable genes to select
    """
    import logging

    import anndata as ad
    import scanpy as sc

    logger = logging.getLogger(__name__)

    logger.info(f"Loading AnnData from {adata_path}")
    adata = ad.read_h5ad(adata_path)

    logger.info("Normalizing total counts per cell")
    sc.pp.normalize_total(adata)

    logger.info("Applying log1p transformation")
    sc.pp.log1p(adata)

    logger.info(f"Finding top {num_hvgs} highly variable genes")
    sc.pp.highly_variable_genes(adata, n_top_genes=num_hvgs)

    logger.info("Storing highly variable genes in .obsm['X_hvg']")
    adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.toarray()

    logger.info(f"Saving preprocessed data to {output_path}")
    adata.write_h5ad(output_path)

    logger.info(f"Preprocessing complete. Selected {adata.var.highly_variable.sum()} highly variable genes.")
