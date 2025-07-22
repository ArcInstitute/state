import argparse


def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, will use model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument("--embed_key", type=str, default=None, help="Key in adata.obsm for input features")
    parser.add_argument(
        "--pert_col", type=str, default="drugname_drugconc", help="Column in adata.obs for perturbation labels"
    )
    parser.add_argument(
        "--batch_col", type=str, default="batch_var", help="Column in adata.obs batch labels"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to output AnnData file (.h5ad)")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model_dir containing the config.yaml file and the pert_onehot_map.pt file that was saved during training.",
    )
    parser.add_argument(
        "--celltype_col", type=str, default=None, help="Column in adata.obs for cell type labels (optional)"
    )
    parser.add_argument(
        "--celltypes", type=str, default=None, help="Comma-separated list of cell types to include (optional)"
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible cell shuffling within perturbation groups")


def run_tx_infer(args):
    import logging
    import os
    import pickle

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from lightning import seed_everything
    from tqdm import tqdm

    from ...tx.models.pert_sets import PertSetsPerturbationModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed if provided
    if args.seed is not None:
        seed_everything(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # Load the config
    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Determine checkpoint path
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
        logger.info(f"No checkpoint provided, using default: {checkpoint_path}")

    # Get perturbation dimensions and mapping from data module
    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)
    pert_dim = var_dims["pert_dim"]

    # Load model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = PertSetsPerturbationModel.load_from_checkpoint(checkpoint_path, strict=False, **cfg['model']['kwargs'])
    model.eval()
    cell_sentence_len = model.cell_sentence_len
    device = next(model.parameters()).device

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
        try:
            X = adata.X.toarray()
        except:
            X = adata.X
        logger.info(f"Using adata.X as input features: shape {X.shape}")

    # Prepare perturbation tensor using the data module's mapping
    pert_names = adata.obs[args.pert_col].values
    pert_tensor = torch.zeros((len(pert_names), pert_dim), device="cpu")  # Keep on CPU initially
    logger.info(f"Perturbation tensor shape: {pert_tensor.shape}")

    # Load perturbation mapping from torch file
    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    pert_onehot_map = torch.load(pert_onehot_map_path, weights_only=False)

    logger.info(f"Data module has {len(pert_onehot_map)} perturbations in mapping")
    logger.info(f"First 10 perturbations in data module: {list(pert_onehot_map.keys())[:10]}")

    unique_pert_names = sorted(set(pert_names))
    logger.info(f"AnnData has {len(unique_pert_names)} unique perturbations")
    logger.info(f"First 10 perturbations in AnnData: {unique_pert_names[:10]}")

    # Load batch mapping from torch file
    batch_onehot_map_path = os.path.join(args.model_dir, "batch_onehot_map.pkl")
    batch_onehot_map = pickle.load(open(batch_onehot_map_path, 'rb'))

    # Create batch name to index mapping
    batch_names = adata.obs[args.batch_col].values
    batch_name_to_idx = {name: idx for idx, name in enumerate(sorted(batch_onehot_map.keys()))}
    
    # Prepare batch indices tensor
    batch_indices_tensor = torch.zeros(len(batch_names), dtype=torch.long, device="cpu")
    
    logger.info(f"Data module has {len(batch_onehot_map)} batches in mapping")
    logger.info(f"First 10 batches in data module: {list(batch_onehot_map.keys())[:10]}")

    unique_batch_names = sorted(set(batch_names))
    logger.info(f"AnnData has {len(unique_batch_names)} unique batches")
    logger.info(f"First 10 batches in AnnData: {unique_batch_names[:10]}")

    # Check overlap for batches
    batch_overlap = set(unique_batch_names) & set(batch_onehot_map.keys())
    logger.info(f"Overlap between AnnData and data module batches: {len(batch_overlap)} batches")
    if len(batch_overlap) < len(unique_batch_names):
        missing_batches = set(unique_batch_names) - set(batch_onehot_map.keys())
        logger.warning(f"Missing batches: {list(missing_batches)[:10]}")

    # Fill batch indices
    batch_matched_count = 0
    default_batch_idx = 0  # Use first batch as default
    for idx, name in enumerate(batch_names):
        if name in batch_name_to_idx:
            batch_indices_tensor[idx] = batch_name_to_idx[name]
            batch_matched_count += 1
        else:
            # Use first available batch as fallback
            batch_indices_tensor[idx] = default_batch_idx

    logger.info(f"Matched {batch_matched_count} out of {len(batch_names)} batches")

    # Check overlap
    overlap = set(unique_pert_names) & set(pert_onehot_map.keys())
    logger.info(f"Overlap between AnnData and data module: {len(overlap)} perturbations")
    if len(overlap) < len(unique_pert_names):
        missing = set(unique_pert_names) - set(pert_onehot_map.keys())
        logger.warning(f"Missing perturbations: {list(missing)[:10]}")

    # Check if there's a control perturbation that might match
    control_pert = cfg["data"]["kwargs"]["control_pert"]
    if args.pert_col == "drugname_drugconc":  # quick hack for tahoe
        control_pert = "[('DMSO_TF', 0.0, 'uM')]"
    logger.info(f"Control perturbation in data module: '{control_pert}'")

    matched_count = 0
    for idx, name in enumerate(pert_names):
        if name in pert_onehot_map:
            pert_tensor[idx] = pert_onehot_map[name]
            matched_count += 1
        else:
            # For now, use control perturbation as fallback
            if control_pert in pert_onehot_map:
                pert_tensor[idx] = pert_onehot_map[control_pert]
            else:
                # Use first available perturbation as fallback
                first_pert = list(pert_onehot_map.keys())[0]
                pert_tensor[idx] = pert_onehot_map[first_pert]

    logger.info(f"Matched {matched_count} out of {len(pert_names)} perturbations")

    # Group cells by perturbation to ensure batches contain cells from same perturbation
    import pandas as pd
    
    # Create a DataFrame for easier grouping
    df = pd.DataFrame({"pert_name": pert_names, "index": range(len(pert_names))})
    grouped = df.groupby("pert_name")
    
    n_samples = len(pert_names)
    batch_size = args.batch_size  # Use user-specified batch size
    
    logger.info(
        f"Running inference on {n_samples} samples grouped by perturbation with batch size {batch_size}..."
    )

    all_preds = []
    processed_samples = 0
    batch_idx = 0

    with torch.no_grad():
        progress_bar = tqdm(total=n_samples, desc="Processing samples", unit="samples")

        # Process each perturbation group
        for pert_name, group in grouped:
            indices = group["index"].values
            
            # Randomize the order of cells within this perturbation group
            if args.seed is not None:
                np.random.shuffle(indices)
            
            group_size = len(indices)
            
            # Process this perturbation group in batches
            for i in range(0, group_size, batch_size):
                batch_indices = indices[i:i + batch_size]
                current_batch_size = len(batch_indices)
                
                # Check if this is an incomplete batch that needs sampling with replacement
                if current_batch_size < batch_size:
                    # Sample with replacement to fill out the batch
                    additional_samples_needed = batch_size - current_batch_size
                    replacement_indices = np.random.choice(batch_indices, size=additional_samples_needed, replace=True)
                    
                    # Combine original indices with replacement indices
                    extended_batch_indices = np.concatenate([batch_indices, replacement_indices])
                    
                    # Get batch data for the extended batch
                    X_batch = torch.tensor(X[extended_batch_indices], dtype=torch.float32).to(device)
                    pert_batch = pert_tensor[extended_batch_indices].to(device)
                    batch_idx_batch = batch_indices_tensor[extended_batch_indices].to(device)
                    pert_names_batch = [pert_names[idx] for idx in extended_batch_indices]
                    
                    # Prepare batch
                    batch = {
                        "ctrl_cell_emb": X_batch,
                        "pert_emb": pert_batch,
                        "pert_name": pert_names_batch,
                        "batch": batch_idx_batch.unsqueeze(0),  # Shape: (1, batch_size)
                    }
                    
                    # Run inference on batch
                    batch_preds = model.predict_step(batch, batch_idx=batch_idx, padded=False)
                    
                    # Extract predictions from the dictionary returned by predict_step
                    if "pert_cell_counts_preds" in batch_preds and batch_preds["pert_cell_counts_preds"] is not None:
                        # Use gene space predictions (from decoder)
                        pred_tensor = batch_preds["pert_cell_counts_preds"]
                    else:
                        # Use latent space predictions
                        pred_tensor = batch_preds["preds"]
                    
                    # Only keep predictions for the original samples (not the replacement samples)
                    original_preds = pred_tensor[:current_batch_size]
                    
                    # Store predictions with their original indices to maintain order
                    batch_preds_with_indices = [(batch_indices[j], original_preds[j].cpu().numpy()) for j in range(current_batch_size)]
                    all_preds.extend(batch_preds_with_indices)
                    
                else:
                    # Full batch - process normally
                    # Get batch data
                    X_batch = torch.tensor(X[batch_indices], dtype=torch.float32).to(device)
                    pert_batch = pert_tensor[batch_indices].to(device)
                    batch_idx_batch = batch_indices_tensor[batch_indices].to(device)
                    pert_names_batch = [pert_names[idx] for idx in batch_indices]

                    # Prepare batch
                    batch = {
                        "ctrl_cell_emb": X_batch,
                        "pert_emb": pert_batch,
                        "pert_name": pert_names_batch,
                        "batch": batch_idx_batch.unsqueeze(0),  # Shape: (1, current_batch_size)
                    }

                    # Run inference on batch
                    batch_preds = model.predict_step(batch, batch_idx=batch_idx, padded=False)

                    # Extract predictions from the dictionary returned by predict_step
                    if "pert_cell_counts_preds" in batch_preds and batch_preds["pert_cell_counts_preds"] is not None:
                        # Use gene space predictions (from decoder)
                        pred_tensor = batch_preds["pert_cell_counts_preds"]
                    else:
                        # Use latent space predictions
                        pred_tensor = batch_preds["preds"]

                    # Store predictions with their original indices to maintain order
                    batch_preds_with_indices = [(batch_indices[j], pred_tensor[j].cpu().numpy()) for j in range(current_batch_size)]
                    all_preds.extend(batch_preds_with_indices)

                # Update progress bar
                progress_bar.update(current_batch_size)
                processed_samples += current_batch_size
                batch_idx += 1

        progress_bar.close()

    # Sort predictions by original index to maintain input order
    all_preds.sort(key=lambda x: x[0])
    preds_np = np.array([pred for _, pred in all_preds])

    # Save predictions to AnnData
    adata.X = preds_np
    output_path = args.output or args.adata.replace(".h5ad", "_with_preds.h5ad")
    adata.write_h5ad(output_path)
    logger.info(f"Saved predictions to {output_path} (in adata.X)")


def main():
    parser = argparse.ArgumentParser(description="Run inference on AnnData with a trained model checkpoint.")
    add_arguments_infer(parser)
    args = parser.parse_args()
    run_tx_infer(args)


if __name__ == "__main__":
    main()
