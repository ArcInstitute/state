import argparse as ap


def add_arguments_query(parser: ap.ArgumentParser):
    """Add arguments for state embedding query CLI."""
    parser.add_argument("--lancedb", required=True, help="Path to existing LanceDB database")
    parser.add_argument("--input", required=True, help="Path to input anndata file with query cells")
    parser.add_argument("--output", required=True, help="Path to output file for results (csv, parquet)")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors to return")
    parser.add_argument("--embed-key", default="X_state", help="Key containing embeddings in input file")
    parser.add_argument("--exclude-distances", action="store_true", help="Exclude vector distances in results")
    parser.add_argument("--filter", type=str, help="Filter expression (e.g., 'cell_type==\"B cell\"')")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for query operations")


def run_emb_query(args: ap.ArgumentParser):
    import logging
    import pandas as pd
    import anndata
    from pathlib import Path

    """
    Query a LanceDB database for similar cells.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from ...emb.vectordb import StateVectorDB

    # check output file extension
    if not args.output.endswith((".csv", ".parquet")):
        raise ValueError("Output file must have a .csv or .parquet extension")

    # Load query cells
    logger.info(f"Loading query cells from {args.input}")
    query_adata = anndata.read_h5ad(args.input)

    # Get embeddings
    if args.embed_key in query_adata.obsm:
        query_embeddings = query_adata.obsm[args.embed_key]
    else:
        raise ValueError(f"Embedding key '{args.embed_key}' not found in input file")

    logger.info(f"Found {len(query_embeddings)} query cells")

    # Connect to database
    vector_db = StateVectorDB(args.lancedb)

    # Get database info
    db_info = vector_db.get_table_info()
    if db_info:
        logger.info(f"Database contains {db_info['num_rows']} cells with {db_info['embedding_dim']}-dim embeddings")

    # Perform batch search
    logger.info(f"Searching for {args.k} nearest neighbors per query cell...")
    results_list = vector_db.batch_search(
        query_vectors=query_embeddings,
        k=args.k,
        filter=args.filter,
        include_distance=not args.exclude_distances,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Add query cell IDs and ranks to results
    all_results = []
    for query_idx, result_df in enumerate(results_list):
        result_df["query_cell_id"] = query_adata.obs.index[query_idx]
        result_df["query_rank"] = range(1, len(result_df) + 1)
        all_results.append(result_df)

    # Combine results
    final_results = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output.endswith(".csv"):
        final_results.to_csv(args.output, index=False)
        logger.info(f"Saved results to {args.output}")
    elif args.output.endswith(".parquet"):
        final_results.to_parquet(args.output, index=False)
        logger.info(f"Saved results to {args.output}")
    else:
        raise ValueError(f"Unsupported output format: {args.output}")


def create_result_anndata(query_adata, results_df, k):
    """Create an anndata object containing query results."""
    # Pivot cell IDs
    cell_ids_pivot = results_df.pivot(index="query_cell_id", columns="query_rank", values="cell_id")
    cell_ids_array = np.array(cell_ids_pivot.values, dtype=str)

    # Handle distances - convert to float64 and handle missing values
    if "vector_distance" in results_df:
        distances_pivot = results_df.pivot(index="query_cell_id", columns="query_rank", values="vector_distance")
        distances_array = np.array(distances_pivot.values, dtype=np.float64)
    else:
        distances_array = None

    # Store match information in uns
    uns_data = {"query_matches": {"cell_ids": cell_ids_array, "distances": distances_array, "k": int(k)}}

    # Create result anndata
    result_adata = query_adata.copy()
    result_adata.uns["lancedb_query_results"] = uns_data

    return result_adata
