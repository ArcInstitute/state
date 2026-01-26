import argparse as ap


def add_arguments_sort(parser: ap.ArgumentParser):
    """Add arguments for the sort subcommand."""
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output sorted AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--context-col",
        type=str,
        required=True,
        help="obs column to sort by context (e.g. cell type)",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        required=False,
        default=None,
        help="optional obs column to sort by batch (if omitted, sorts by context + perturbation)",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        required=True,
        help="obs column to sort by perturbation",
    )


def run_tx_sort(args: ap.Namespace):
    import logging
    from pathlib import Path

    import anndata as ad

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    input_path = args.input
    output_path = args.output
    sort_cols = [args.context_col]
    if args.batch_col:
        sort_cols.append(args.batch_col)
    sort_cols.append(args.pert_col)

    logger.info("Loading AnnData from %s", input_path)
    adata = ad.read_h5ad(input_path)

    missing = [col for col in sort_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing obs columns for sorting: {missing}")

    logger.info("Sorting AnnData by columns: %s", sort_cols)
    order = adata.obs.sort_values(by=sort_cols, kind="mergesort").index
    adata_sorted = adata[order].copy()

    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing sorted AnnData to %s", output_path)
    adata_sorted.write_h5ad(output_path)
    logger.info("Sort complete. Wrote %d cells.", adata_sorted.n_obs)
