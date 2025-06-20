import argparse
import scanpy as sc
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from ...sets.models.pert_sets import PertSetsPerturbationModel

# state-sets sets infer --output_dir /home/aadduri/state-sets/test/ --checkpoint last.ckpt --adata /home/aadduri/state-sets/test/adata.h5ad --pert_col gene 
# state-sets sets infer --output_dir /home/dhruvgautam/state-sets/test/ --checkpoint /large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered_cs32_pretrained/hepg2/checkpoints/step=44000.ckpt --adata /large_storage/ctc/ML/state_sets/replogle/processed.h5 --pert_col gene 

# state-sets sets infer --output_dir /home/dhruvgautam/state-sets/test/ --checkpoint /large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered_cs32_pretrained/hepg2/checkpoints/step=44000.ckpt --adata /large_storage/ctc/ML/state_sets/replogle/processed.h5 --pert_col gene --embed_key X_vci_1.5.2_4

def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument("--embed_key", type=str, default="X_hvg", help="Key in adata.obsm for input features")
    parser.add_argument("--pert_col", type=str, default="drugname_drugconc", help="Column in adata.obs for perturbation labels")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output AnnData file (.h5ad)")
    parser.add_argument("--celltype_col", type=str, default=None, help="Column in adata.obs for cell type labels (optional)")
    parser.add_argument("--celltypes", type=str, default=None, help="Comma-separated list of cell types to include (optional)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference (default: 1000)")


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
    pert_dim = model.pert_dim

    logger.info(f"Using model's cell_sentence_len: {cell_sentence_len}")
    logger.info(f"Using pert_dim: {pert_dim}")

    # Load AnnData
    logger.info(f"Loading AnnData from: {args.adata}")
    adata_full = sc.read_h5ad(args.adata)

    # Define control perturbations to look for
    control_perts = ["DMSO_TF_24h", "non-targeting", "control", "DMSO"]
    
    # Find available control perturbation
    available_perts = set(adata_full.obs[args.pert_col].unique())
    control_pert = None
    for ctrl in control_perts:
        if ctrl in available_perts:
            control_pert = ctrl
            logger.info(f"Using '{ctrl}' as control perturbation")
            break
    
    if control_pert is None:
        # Use the first available perturbation as fallback
        control_pert = list(available_perts)[0]
        logger.warning(f"No standard control found, using '{control_pert}' as control")

    # Get available cell types and select the most abundant one
    if args.celltype_col is not None:
        if args.celltype_col not in adata_full.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        
        if args.celltypes is not None:
            celltypes = [ct.strip() for ct in args.celltypes.split(",")]
            adata_full = adata_full[adata_full.obs[args.celltype_col].isin(celltypes)].copy()
            logger.info(f"Filtered to specified cell types: {celltypes}")
        
        cell_type_counts = adata_full.obs[args.celltype_col].value_counts()
        logger.info("Available cell types: %s", list(cell_type_counts.index))
        celltype1 = cell_type_counts.index[0]
        logger.info(f"Selected cell type: {celltype1} ({cell_type_counts[celltype1]} available)")
        
        # Get control cells for this cell type
        cells_type1 = adata_full[(adata_full.obs[args.pert_col] == control_pert) & 
                                (adata_full.obs[args.celltype_col] == celltype1)].copy()
        logger.info(f"Available control cells - {celltype1}: {cells_type1.n_obs}")
    else:
        # No cell type filtering, use all control cells
        cells_type1 = adata_full[adata_full.obs[args.pert_col] == control_pert].copy()
        logger.info(f"Available control cells: {cells_type1.n_obs}")

    # Use the model's actual cell_sentence_len
    n_cells = cell_sentence_len
    
    if cells_type1.n_obs >= n_cells:
        # Sample cells
        idx1 = np.random.choice(cells_type1.n_obs, size=n_cells, replace=False)
        sampled_cells = cells_type1[idx1].copy()
        
        logger.info(f"Sampled {sampled_cells.n_obs} cells for inference")
        
        # Extract embeddings based on available key
        if args.embed_key in sampled_cells.obsm:
            X_embed = torch.tensor(sampled_cells.obsm[args.embed_key], dtype=torch.float32).to(device)
            logger.info(f"Using adata.obsm['{args.embed_key}'] as input features: shape {X_embed.shape}")
        else:
            X_data = sampled_cells.X.toarray() if hasattr(sampled_cells.X, 'toarray') else sampled_cells.X
            X_embed = torch.tensor(X_data, dtype=torch.float32).to(device)
            logger.info(f"Using adata.X as input features: shape {X_embed.shape}")
        
        # Create simple perturbation tensor - set first dimension to 1 for control
        pert_tensor = torch.zeros((n_cells, pert_dim), device=device)
        pert_tensor[:, 0] = 1  # Set first dimension to 1 for control perturbation
        pert_names = [control_pert] * n_cells
        
        # Create batch dictionary
        batch = {
            "ctrl_cell_emb": X_embed,
            "pert_emb": pert_tensor,
            "pert_name": pert_names,
            "batch": torch.zeros((1, cell_sentence_len), device=device)
        }
        
        logger.info(f"Batch shapes - ctrl_cell_emb: {batch['ctrl_cell_emb'].shape}, pert_emb: {batch['pert_emb'].shape}")
        logger.info(f"Running single forward pass with {n_cells} cells")
        
        # Single forward pass
        with torch.no_grad():
            preds = model.forward(batch, padded=False)
        
        logger.info("Forward pass completed successfully")
        preds_np = preds.cpu().numpy()
        
        # Save predictions to sampled cells
        pred_key = "model_preds"
        sampled_cells.obsm[pred_key] = preds_np
        
    else:
        raise ValueError(f"Not enough control cells available. Need {n_cells}, but only have {cells_type1.n_obs}")

        # Save results
        output_path = args.output_dir or args.adata.replace(".h5ad", "_with_preds.h5ad").replace(".h5", "_with_preds.h5ad")
        sampled_cells.write_h5ad(output_path)
        logger.info(f"Saved predictions to {output_path} (in adata.obsm['{pred_key}'])")


def main():
    parser = argparse.ArgumentParser(description="Run inference on AnnData with a trained model checkpoint.")
    add_arguments_infer(parser)
    args = parser.parse_args()
    run_sets_infer(args)

if __name__ == "__main__":
    main()
