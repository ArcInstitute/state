import argparse as ap


def add_arguments_transform(parser: ap.ArgumentParser):
    """Add arguments for state embedding CLI."""
    parser.add_argument("--model-folder", required=True, help="Path to the model checkpoint folder")
    parser.add_argument("--checkpoint", required=False, help="Path to the specific model checkpoint")
    parser.add_argument("--input", required=True, help="Path to input anndata file (h5ad)")
    parser.add_argument("--output", required=False, help="Path to output embedded anndata file (h5ad)")
    parser.add_argument("--embed-key", default="X_state", help="Name of key to store embeddings")
    parser.add_argument("--gene-column", default="gene_name", help="Name of column in var dataframe to use for gene names")
    parser.add_argument("--lancedb", type=str, help="Path to LanceDB database for vector storage")
    parser.add_argument("--lancedb-update", action="store_true", 
                       help="Update existing entries in LanceDB (default: append)")
    parser.add_argument("--lancedb-batch-size", type=int, default=1000,
                       help="Batch size for LanceDB operations")


def run_emb_transform(args: ap.ArgumentParser):
    """
    Compute embeddings for an input anndata file using a pre-trained VCI model checkpoint.
    """
    import glob
    import logging
    import os

    import torch
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from ...emb.inference import Inference

    # check for --output or --lancedb
    if not args.output and not args.lancedb:
        logger.error("Either --output or --lancedb must be provided")
        raise ValueError("Either --output or --lancedb must be provided")

    # look in the model folder with glob for *.ckpt, get the first one, and print it
    model_files = glob.glob(os.path.join(args.model_folder, "*.ckpt"))
    if not model_files:
        logger.error(f"No model checkpoint found in {args.model_folder}")
        raise FileNotFoundError(f"No model checkpoint found in {args.model_folder}")
    if not args.checkpoint:
        args.checkpoint = model_files[-1]
    logger.info(f"Using model checkpoint: {args.checkpoint}")

    # Create inference object
    logger.info("Creating inference object")
    embedding_file = os.path.join(args.model_folder, "protein_embeddings.pt")
    protein_embeds = torch.load(embedding_file, weights_only=False, map_location="cpu")

    config_file = os.path.join(args.model_folder, "config.yaml")
    conf = OmegaConf.load(config_file)

    inferer = Inference(cfg=conf, protein_embeds=protein_embeds)

    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    inferer.load_model(args.checkpoint)

    # Create output directory if it doesn't exist
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

    # Generate embeddings
    logger.info(f"Computing embeddings for {args.input}")
    if args.output:
        logger.info(f"Output will be saved to {args.output}")
    if args.lancedb:
        logger.info(f"Embeddings will be saved to LanceDB at {args.lancedb}")

    inferer.encode_adata(
        input_adata_path=args.input,
        output_adata_path=args.output,
        emb_key=args.embed_key,
        gene_column=args.gene_column,
        lancedb_path=args.lancedb,
        update_lancedb=args.lancedb_update,
        lancedb_batch_size=args.lancedb_batch_size,
    )

    logger.info("Embedding computation completed successfully!")
