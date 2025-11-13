import argparse as ap


def add_arguments_transform(parser: ap.ArgumentParser):
    """Add arguments for state embedding CLI."""
    parser.add_argument("--model-folder", required=True, help="Path to the model checkpoint folder")
    parser.add_argument("--checkpoint", required=False, help="Path to the specific model checkpoint")
    parser.add_argument(
        "--config",
        required=False,
        help=(
            "Path to config override. If omitted, uses the config embedded in the checkpoint; ignores any config in the model folder."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to input anndata file (h5ad)")
    parser.add_argument("--output", required=False, help="Path to output embedded anndata file (h5ad)")
    parser.add_argument("--embed-key", default="X_state", help="Name of key to store embeddings")
    parser.add_argument(
        "--protein-embeddings",
        required=False,
        help=(
            "Path to protein embeddings override (.pt). If omitted, the CLI will look for 'protein_embeddings.pt' in --model-folder, "
            "then fall back to embeddings packaged in the checkpoint, and finally the path from the config."
        ),
    )
    parser.add_argument("--lancedb", type=str, help="Path to LanceDB database for vector storage")
    parser.add_argument(
        "--lancedb-update", action="store_true", help="Update existing entries in LanceDB (default: append)"
    )
    parser.add_argument("--lancedb-batch-size", type=int, default=1000, help="Batch size for LanceDB operations")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for embedding forward pass (overrides config). "
            "Increase to use more VRAM and speed up embedding."
        ),
    )


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
    # Resolve protein embeddings in priority order:
    # 1) Explicit --protein-embeddings
    # 2) Auto-detect 'protein_embeddings.pt' in --model-folder
    # 3) Let Inference load from checkpoint/config
    protein_embeds = None
    if args.protein_embeddings:
        logger.info(f"Using protein embeddings override: {args.protein_embeddings}")
        protein_embeds = torch.load(args.protein_embeddings, weights_only=False, map_location="cpu")
    else:
        # Try auto-detect in model folder
        try:
            exact_path = os.path.join(args.model_folder, "protein_embeddings.pt")
            cand_path = None
            if os.path.exists(exact_path):
                cand_path = exact_path
            else:
                # Consider other variations like protein_embeddings*.pt
                pe_files = sorted(glob.glob(os.path.join(args.model_folder, "protein_embeddings*.pt")))
                if pe_files:
                    # Prefer the lexicographically last to mimic checkpoint selection behavior
                    cand_path = pe_files[-1]
            if cand_path is not None:
                logger.info(
                    f"Found protein embeddings in model folder: {cand_path}. Using these and overriding config."
                )
                protein_embeds = torch.load(cand_path, weights_only=False, map_location="cpu")
        except Exception as e:
            logger.warning(
                f"Failed to load auto-detected protein embeddings: {e}. Will fall back to checkpoint/config."
            )

    # Only use config override if explicitly provided; otherwise use config embedded in the checkpoint
    conf = OmegaConf.load(args.config) if args.config else None
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
        batch_size=args.batch_size if getattr(args, "batch_size", None) is not None else None,
        lancedb_path=args.lancedb,
        update_lancedb=args.lancedb_update,
        lancedb_batch_size=args.lancedb_batch_size,
    )

    logger.info("Embedding computation completed successfully!")
