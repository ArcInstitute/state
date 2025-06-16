import argparse
import scanpy as sc
import torch
import numpy as np
import os
import pandas as pd

from ...sets.models.pert_sets import PertSetsPerturbationModel

def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument("--embed_key", type=str, default="X_hvg", help="Key in adata.obsm for input features")
    parser.add_argument("--pert_col", type=str, default="drugname_drugconc", help="Column in adata.obs for perturbation labels")
    parser.add_argument("--output", type=str, default=None, help="Path to output AnnData file (.h5ad)")
    parser.add_argument("--celltype_col", type=str, default=None, help="Column in adata.obs for cell type labels (optional)")
    parser.add_argument("--celltypes", type=str, default=None, help="Comma-separated list of cell types to include (optional)")


def run_sets_infer(args):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = PertSetsPerturbationModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    cell_sentence_len = model.cell_sentence_len
    device = next(model.parameters()).device

    # Use model's config for batch prep
    pert_onehot_map = getattr(model, "pert_onehot_map", None)
    pert_dim = model.pert_dim

    # Load AnnData
    logger.info(f"Loading AnnData from: {args.adata}")
    adata = sc.read_h5ad(args.adata)

    # Optionally filter by cell type
    if args.celltype_col is not None and args.celltypes is not None:
        celltypes = [ct.strip() for ct in args.celltypes.split(",")]
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        initial_n = adata.n_obs
        adata = adata[adata.obs[args.celltype_col].isin(celltypes)].copy()
        logger.info(f"Filtered AnnData to {adata.n_obs} cells of types {celltypes} (from {initial_n} cells)")
    elif args.celltype_col is not None:
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        logger.info(f"No cell type filtering applied, but cell type column '{args.celltype_col}' is available.")

    # Get input features
    if args.embed_key in adata.obsm:
        X = adata.obsm[args.embed_key]
        logger.info(f"Using adata.obsm['{args.embed_key}'] as input features: shape {X.shape}")
    else:
        X = adata.X
        logger.info(f"Using adata.X as input features: shape {X.shape}")
    X = torch.tensor(X, dtype=torch.float32).to(device)

    # Prepare perturbation tensor using the model's map
    pert_names = adata.obs[args.pert_col].values
    pert_tensor = torch.zeros((len(pert_names), pert_dim), device=device)
    if pert_onehot_map is not None:
        for idx, name in enumerate(pert_names):
            if name in pert_onehot_map:
                pert_tensor[idx, pert_onehot_map[name]] = 1
            else:
                # Optionally handle unknown perturbations
                pass
    else:
        # Fallback: build map from AnnData (not recommended for production)
        unique_perts = sorted(set(pert_names))
        pert_map = {name: i for i, name in enumerate(unique_perts)}
        for idx, name in enumerate(pert_names):
            pert_tensor[idx, pert_map[name]] = 1

    # Prepare batch
    batch = {
        "ctrl_cell_emb": X,
        "pert_emb": pert_tensor,
        "pert_name": pert_names.tolist(),
        "batch": torch.zeros((1, cell_sentence_len), device=device)
    }
    # when do we need the batch num things

    logger.info("Running inference...")
    with torch.no_grad():
        preds = model.forward(batch)
    preds_np = preds.cpu().numpy()

    # Save predictions to AnnData
    pred_key = "model_preds"
    adata.obsm[pred_key] = preds_np
    output_path = args.output or args.adata.replace(".h5ad", "_with_preds.h5ad")
    adata.write_h5ad(output_path)
    logger.info(f"Saved predictions to {output_path} (in adata.obsm['{pred_key}'])")


def main():
    parser = argparse.ArgumentParser(description="Run inference on AnnData with a trained model checkpoint.")
    add_arguments_infer(parser)
    args = parser.parse_args()
    run_sets_infer(args)

if __name__ == "__main__":
    main()
