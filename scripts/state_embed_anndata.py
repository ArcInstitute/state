#!/usr/bin/env python3
"""
VCI Model Embedding Script

This script computes embeddings for an input anndata file using a pre-trained VCI model checkpoint.
It can be run from any directory and outputs the embedded anndata to a specified location.

Usage:
    python embed_vci.py --checkpoint PATH_TO_CHECKPOINT --input INPUT_ANNDATA --output OUTPUT_ANNDATA

Example:
    python embed_vci.py --checkpoint /path/to/model.ckpt --input data.h5ad --output embedded_data.h5ad
"""

import argparse
import os

from omegaconf import OmegaConf

from state_sets.state.inference import Inference


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compute embeddings for anndata using a VCI model")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint file")
    parser.add_argument("--config", required=True, help="Path to the model training config")
    parser.add_argument("--input", required=True, help="Path to input anndata file (h5ad)")
    parser.add_argument("--output", required=True, help="Path to output embedded anndata file (h5ad)")
    parser.add_argument("--dataset-name", default="perturbation", help="Dataset name to be used in dataloader creation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--filter", action="store_true", help="Filter gene set to our esm embeddings only.")
    parser.add_argument("--embed-key", help="Name of key to store")

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    conf = OmegaConf.load(args.config)
    inferer = Inference(conf)
    inferer.load_model(args.checkpoint)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    inferer.encode_adata(args.input, args.output, emb_key=args.embed_key, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
