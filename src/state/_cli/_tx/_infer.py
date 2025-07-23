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


def run_tx_infer(args):
    import logging
    import os
    import pickle

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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
    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
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

    # Process in batches with progress bar
    # Use cell_sentence_len as batch size since model expects this
    n_samples = len(pert_names)
    batch_size = cell_sentence_len  # Model requires this exact batch size
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    logger.info(
        f"Running inference on {n_samples} samples in {n_batches} batches of size {batch_size} (model's cell_sentence_len)..."
    )

    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(total=n_samples, desc="Processing samples", unit="samples")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx

            # Get batch data
            X_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32).to(device)
            pert_batch = pert_tensor[start_idx:end_idx].to(device)
            pert_names_batch = pert_names[start_idx:end_idx].tolist()

            # Pad the batch to cell_sentence_len if it's the last incomplete batch
            if current_batch_size < cell_sentence_len:
                # Pad with zeros for embeddings
                padding_size = cell_sentence_len - current_batch_size
                X_pad = torch.zeros((padding_size, X_batch.shape[1]), device=device)
                X_batch = torch.cat([X_batch, X_pad], dim=0)

                # Pad perturbation tensor with control perturbation
                pert_pad = torch.zeros((padding_size, pert_batch.shape[1]), device=device)
                if control_pert in pert_onehot_map:
                    pert_pad[:] = pert_onehot_map[control_pert].to(device)
                else:
                    pert_pad[:, 0] = 1  # Default to first perturbation
                pert_batch = torch.cat([pert_batch, pert_pad], dim=0)

                # Extend perturbation names
                pert_names_batch.extend([control_pert] * padding_size)

            # Prepare batch - use same format as working code
            batch = {
                "ctrl_cell_emb": X_batch,
                "pert_emb": pert_batch,  # Keep as 2D tensor
                "pert_name": pert_names_batch,
                "batch": torch.zeros((1, cell_sentence_len), device=device),  # Use (1, cell_sentence_len)
            }

            # Run inference on batch using padded=False like in working code
            batch_preds = model.predict_step(batch, batch_idx=batch_idx, padded=False)

            # Extract predictions from the dictionary returned by predict_step
            # Use gene decoder output if available, otherwise use latent predictions
            if "pert_cell_counts_preds" in batch_preds and batch_preds["pert_cell_counts_preds"] is not None:
                # Use gene space predictions (from decoder)
                pred_tensor = batch_preds["pert_cell_counts_preds"]
            else:
                # Use latent space predictions
                pred_tensor = batch_preds["preds"]

            # Only keep predictions for the actual samples (not padding)
            actual_preds = pred_tensor[:current_batch_size]
            all_preds.append(actual_preds.cpu().numpy())

            # Update progress bar
            progress_bar.update(current_batch_size)

        progress_bar.close()

    # Concatenate all predictions
    preds_np = np.concatenate(all_preds, axis=0)

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
