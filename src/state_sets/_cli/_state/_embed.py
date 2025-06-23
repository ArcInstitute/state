import argparse as ap
import torch


def add_arguments_embed(parser: ap.ArgumentParser):
    """Add arguments for state embedding CLI."""
    parser.add_argument("--model-folder", required=True, help="Path to the model checkpoint folder")
    parser.add_argument("--input", required=True, help="Path to input anndata file (h5ad)")
    parser.add_argument("--output", required=True, help="Path to output embedded anndata file (h5ad)")
    parser.add_argument("--embed-key", default="X_state", help="Name of key to store embeddings")


def run_state_embed(args: ap.ArgumentParser):
    """
    Compute embeddings for an input anndata file using a pre-trained VCI model checkpoint.
    """
    import os
    import glob
    import logging

    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from ...state.inference import Inference

    # look in the model folder with glob for *.ckpt, get the first one, and print it
    model_files = glob.glob(os.path.join(args.model_folder, "*.ckpt"))
    if not model_files:
        logger.error(f"No model checkpoint found in {args.model_folder}")
        raise FileNotFoundError(f"No model checkpoint found in {args.model_folder}")
    args.checkpoint = model_files[0]
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
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Generate embeddings
    logger.info(f"Computing embeddings for {args.input}")
    logger.info(f"Output will be saved to {args.output}")

    inferer.encode_adata(
        input_adata_path=args.input,
        output_adata_path=args.output,
        emb_key=args.embed_key,
    )

    logger.info("Embedding computation completed successfully!")
