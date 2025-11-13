import argparse
from typing import Dict, List, Optional
import pandas as pd


def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, defaults to model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key in adata.obsm for input features (if None, uses adata.X). If provided, .X will be left untouched in the output file.",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="drugname_drugconc",
        help="Column in adata.obs for perturbation labels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output AnnData file (.h5ad). Defaults to <input>_simulated.h5ad",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the training run directory. Must contain config.yaml, var_dims.pkl, pert_onehot_map.pt, batch_onehot_map.pkl.",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default=None,
        help="Column in adata.obs to group by (defaults to auto-detected cell type column).",
    )
    parser.add_argument(
        "--celltypes",
        type=str,
        default=None,
        help="Comma-separated list of cell types to include (optional).",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Batch column name in adata.obs. If omitted, tries config['data']['kwargs']['batch_col'] then common fallbacks.",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=None,
        help="Override the control perturbation label. If omitted, read from config; for 'drugname_drugconc', defaults to \"[('DMSO_TF', 0.0, 'uM')]\".",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for control sampling (default: 42)",
    )
    parser.add_argument(
        "--max-set-len",
        type=int,
        default=None,
        help="Maximum set length per forward pass. If omitted, uses the model's trained cell_set_len.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )
    parser.add_argument(
        "--tsv",
        type=str,
        default=None,
        help="Path to TSV file with columns 'perturbation' and 'num_cells' to pad the adata with additional perturbation cells copied from random controls.",
    )


def run_tx_infer(args: argparse.Namespace):
    import os
    import pickle
    import warnings

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    # -----------------------
    # Helpers
    # -----------------------
    def load_config(cfg_path: str) -> dict:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def to_dense(mat):
        """Return a dense numpy array for a variety of AnnData .X backends."""
        try:
            import scipy.sparse as sp

            if sp.issparse(mat):
                return mat.toarray()
        except Exception:
            pass
        return np.asarray(mat)

    def pick_first_present(d: "sc.AnnData", candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in d.obs:
                return c
        return None

    def argmax_index_from_any(v, expected_dim: Optional[int]) -> Optional[int]:
        """
        Convert a saved mapping value (one-hot tensor, numpy array, or int) to an index.
        """
        if v is None:
            return None
        try:
            if torch.is_tensor(v):
                if v.ndim == 1:
                    return int(torch.argmax(v).item())
                else:
                    return None
        except Exception:
            pass
        try:
            import numpy as _np

            if isinstance(v, _np.ndarray):
                if v.ndim == 1:
                    return int(v.argmax())
                else:
                    return None
        except Exception:
            pass
        if isinstance(v, (int, np.integer)):
            return int(v)
        return None

    def prepare_batch(
        ctrl_basal_np: np.ndarray,
        pert_onehots: torch.Tensor,
        batch_indices: Optional[torch.Tensor],
        pert_names: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor | List[str]]:
        """
        Construct a model batch with variable-length sentence (B=1, S=T, ...).
        IMPORTANT: All tokens in this batch share the same perturbation.
        """
        X_batch = torch.tensor(ctrl_basal_np, dtype=torch.float32, device=device)  # [T, E_in]
        batch = {
            "ctrl_cell_emb": X_batch,
            "pert_emb": pert_onehots.to(device),  # [T, pert_dim] (same row repeated)
            "pert_name": pert_names,  # list[str], all identical
        }
        if batch_indices is not None:
            batch["batch"] = batch_indices.to(device)  # [T]
        return batch

    def pad_adata_with_tsv(
        adata: "sc.AnnData",
        tsv_path: str,
        pert_col: str,
        control_pert: str,
        rng: np.random.RandomState,
        quiet: bool = False,
    ) -> "sc.AnnData":
        """
        Pad AnnData with additional perturbation cells by copying random control cells
        and updating their perturbation labels according to the TSV specification.

        Args:
            adata: Input AnnData object
            tsv_path: Path to TSV file with 'perturbation' and 'num_cells' columns
            pert_col: Name of perturbation column in adata.obs
            control_pert: Label for control perturbation
            rng: Random number generator for sampling
            quiet: Whether to suppress logging

        Returns:
            AnnData object with padded cells
        """
        # Load TSV file
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        try:
            tsv_df = pd.read_csv(tsv_path, sep="\t")
        except Exception as e:
            raise ValueError(f"Error reading TSV file {tsv_path}: {e}")

        # Validate TSV format
        required_cols = ["perturbation", "num_cells"]
        missing_cols = [col for col in required_cols if col not in tsv_df.columns]
        if missing_cols:
            raise ValueError(
                f"TSV file missing required columns: {missing_cols}. Found columns: {list(tsv_df.columns)}"
            )

        # Find control cells
        ctl_mask = adata.obs[pert_col].astype(str) == str(control_pert)
        control_indices = np.where(ctl_mask)[0]

        if len(control_indices) == 0:
            raise ValueError(f"No control cells found with perturbation '{control_pert}' in column '{pert_col}'")

        if not quiet:
            print(f"Found {len(control_indices)} control cells for padding")

        # Collect cells to add
        new_cells_data = []
        total_to_add = 0

        for _, row in tsv_df.iterrows():
            pert_name = str(row["perturbation"])
            num_cells = int(row["num_cells"])
            total_to_add += num_cells

            if num_cells <= 0:
                continue

            # Sample control cells with replacement
            sampled_indices = rng.choice(control_indices, size=num_cells, replace=True)

            for idx in sampled_indices:
                new_cells_data.append({"original_index": idx, "new_perturbation": pert_name})

        if len(new_cells_data) == 0:
            if not quiet:
                print("No cells to add from TSV file")
            return adata

        if not quiet:
            print(f"Adding {total_to_add} cells from TSV specification")

        # Create new AnnData with padded cells
        original_n_obs = adata.n_obs
        new_n_obs = original_n_obs + len(new_cells_data)

        # Copy X data
        if hasattr(adata.X, "toarray"):  # sparse matrix
            new_X = np.vstack(
                [adata.X.toarray(), adata.X[np.array([cell["original_index"] for cell in new_cells_data])].toarray()]
            )
        else:  # dense matrix
            new_X = np.vstack([adata.X, adata.X[np.array([cell["original_index"] for cell in new_cells_data])]])

        # Copy obs data
        new_obs = adata.obs.copy()
        for i, cell_data in enumerate(new_cells_data):
            orig_idx = cell_data["original_index"]
            new_pert = cell_data["new_perturbation"]

            # Copy the original control cell's metadata
            new_row = adata.obs.iloc[orig_idx].copy()
            # Update perturbation label
            new_row[pert_col] = new_pert

            new_obs.loc[original_n_obs + i] = new_row

        # Copy obsm data
        new_obsm = {}
        for key, matrix in adata.obsm.items():
            padded_matrix = np.vstack([matrix, matrix[np.array([cell["original_index"] for cell in new_cells_data])]])
            new_obsm[key] = padded_matrix

        # Copy varm, uns, var (unchanged)
        new_varm = adata.varm.copy()
        new_uns = adata.uns.copy()
        new_var = adata.var.copy()

        # Create new AnnData object
        import scanpy as sc

        new_adata = sc.AnnData(X=new_X, obs=new_obs, var=new_var, obsm=new_obsm, varm=new_varm, uns=new_uns)

        if not quiet:
            print(f"Padded AnnData: {original_n_obs} -> {new_n_obs} cells")

        return new_adata

    # -----------------------
    # Logging
    # -----------------------
    if not args.quiet:
        print("==> STATE: tx infer (virtual experiment)")

    # -----------------------
    # 1) Load config + dims + mappings
    # -----------------------
    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    if not args.quiet:
        print(f"Loaded config: {config_path}")

    # control_pert
    control_pert = args.control_pert
    if control_pert is None:
        try:
            control_pert = cfg["data"]["kwargs"]["control_pert"]
        except Exception:
            control_pert = None
    if control_pert is None and args.pert_col == "drugname_drugconc":
        control_pert = "[('DMSO_TF', 0.0, 'uM')]"
    if control_pert is None:
        control_pert = "non-targeting"
    if not args.quiet:
        print(f"Control perturbation: {control_pert}")

    # choose cell type column
    if args.celltype_col is None:
        ct_from_cfg = None
        try:
            ct_from_cfg = cfg["data"]["kwargs"].get("cell_type_key", None)
        except Exception:
            pass
        guess = pick_first_present(
            sc.read_h5ad(args.adata),
            candidates=[ct_from_cfg, "cell_type", "celltype", "cellType", "ctype", "celltype_col"]
            if ct_from_cfg
            else ["cell_type", "celltype", "cellType", "ctype", "celltype_col"],
        )
        args.celltype_col = guess
    if not args.quiet:
        print(f"Grouping by cell type column: {args.celltype_col if args.celltype_col else '(not found; no grouping)'}")

    # choose batch column
    if args.batch_col is None:
        try:
            args.batch_col = cfg["data"]["kwargs"].get("batch_col", None)
        except Exception:
            args.batch_col = None

    # dimensionalities
    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    if not os.path.exists(var_dims_path):
        raise FileNotFoundError(f"Missing var_dims.pkl at {var_dims_path}")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)

    pert_dim = var_dims.get("pert_dim")
    batch_dim = var_dims.get("batch_dim", None)

    # mappings
    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_onehot_map_path):
        raise FileNotFoundError(f"Missing pert_onehot_map.pt at {pert_onehot_map_path}")
    pert_onehot_map: Dict[str, torch.Tensor] = torch.load(pert_onehot_map_path, weights_only=False)

    batch_onehot_map_path = os.path.join(args.model_dir, "batch_onehot_map.pkl")
    batch_onehot_map = None
    if os.path.exists(batch_onehot_map_path):
        with open(batch_onehot_map_path, "rb") as f:
            batch_onehot_map = pickle.load(f)

    # -----------------------
    # 2) Load model
    # -----------------------
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
        if not args.quiet:
            print(f"No --checkpoint given, using {checkpoint_path}")

    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = next(model.parameters()).device
    cell_set_len = args.max_set_len if args.max_set_len is not None else getattr(model, "cell_sentence_len", 256)
    uses_batch_encoder = getattr(model, "batch_encoder", None) is not None
    output_space = getattr(model, "output_space", cfg.get("data", {}).get("kwargs", {}).get("output_space", "gene"))

    if not args.quiet:
        print(f"Model device: {device}")
        print(f"Model cell_set_len (max sequence length): {cell_set_len}")
        print(f"Model uses batch encoder: {bool(uses_batch_encoder)}")
        print(f"Model output space: {output_space}")

    # -----------------------
    # 3) Load AnnData
    # -----------------------
    adata = sc.read_h5ad(args.adata)

    # optional TSV padding mode - pad with additional perturbation cells
    if args.tsv:
        if not args.quiet:
            print(f"==> TSV padding mode: loading {args.tsv}")

        # Initialize RNG for padding (separate from inference RNG for reproducibility)
        pad_rng = np.random.RandomState(args.seed)

        adata = pad_adata_with_tsv(
            adata=adata,
            tsv_path=args.tsv,
            pert_col=args.pert_col,
            control_pert=control_pert,
            rng=pad_rng,
            quiet=args.quiet,
        )

    # optional filter by cell types
    if args.celltype_col and args.celltypes:
        keep_cts = [ct.strip() for ct in args.celltypes.split(",")]
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not in adata.obs")
        n0 = adata.n_obs
        adata = adata[adata.obs[args.celltype_col].isin(keep_cts)].copy()
        if not args.quiet:
            print(f"Filtered to {adata.n_obs} cells (from {n0}) for cell types: {keep_cts}")

    # select features: embeddings or genes
    if args.embed_key is None:
        X_in = to_dense(adata.X)  # [N, E_in]
        writes_to = (".X", None)  # write predictions to .X
    else:
        if args.embed_key not in adata.obsm:
            raise KeyError(f"Embedding key '{args.embed_key}' not found in adata.obsm")
        X_in = np.asarray(adata.obsm[args.embed_key])  # [N, E_in]
        writes_to = (".obsm", args.embed_key)  # write predictions to obsm[embed_key]

    if not args.quiet:
        print(
            f"Using {'adata.X' if args.embed_key is None else f'adata.obsm[{args.embed_key!r}]'} as input features: shape {X_in.shape}"
        )

    # pick pert names; ensure they are strings
    if args.pert_col not in adata.obs:
        raise KeyError(f"Perturbation column '{args.pert_col}' not found in adata.obs")
    pert_names_all = adata.obs[args.pert_col].astype(str).values

    # derive batch indices (per-token integers) if needed
    batch_indices_all: Optional[np.ndarray] = None
    if uses_batch_encoder:
        # locate batch column
        batch_col = args.batch_col
        if batch_col is None:
            candidates = ["gem_group", "gemgroup", "batch", "donor", "plate", "experiment", "lane", "batch_id"]
            batch_col = next((c for c in candidates if c in adata.obs), None)
        if batch_col is not None and batch_col in adata.obs:
            raw_labels = adata.obs[batch_col].astype(str).values
            if batch_onehot_map is None:
                warnings.warn(
                    f"Model has a batch encoder, but '{batch_onehot_map_path}' not found. "
                    "Batch info will be ignored; predictions may degrade."
                )
                uses_batch_encoder = False
            else:
                # Convert labels to indices using saved map
                label_to_idx: Dict[str, int] = {}
                for k, v in batch_onehot_map.items():
                    key = str(k)
                    idx = argmax_index_from_any(v, expected_dim=batch_dim)
                    if idx is not None:
                        label_to_idx[key] = idx
                idxs = np.zeros(len(raw_labels), dtype=np.int64)
                misses = 0
                for i, lab in enumerate(raw_labels):
                    if lab in label_to_idx:
                        idxs[i] = label_to_idx[lab]
                    else:
                        misses += 1
                        idxs[i] = 0  # fallback to zero
                if misses and not args.quiet:
                    print(
                        f"Warning: {misses} / {len(raw_labels)} batch labels not found in saved mapping; using index 0 as fallback."
                    )
                batch_indices_all = idxs
        else:
            if not args.quiet:
                print("Batch encoder present, but no batch column found; proceeding without batch indices.")
            uses_batch_encoder = False

    # -----------------------
    # 4) Build control template on the fly & simulate ALL cells (controls included)
    # -----------------------
    rng = np.random.RandomState(args.seed)

    # Identify control vs non-control
    ctl_mask = pert_names_all == str(control_pert)
    n_controls = int(ctl_mask.sum())
    n_total = adata.n_obs
    n_nonctl = n_total - n_controls
    if not args.quiet:
        print(f"Cells: total={n_total}, control={n_controls}, non-control={n_nonctl}")

    # Where we will write predictions (initialize with originals; we overwrite all rows, including controls)
    if writes_to[0] == ".X":
        sim_X = X_in.copy()
        out_target = "X"
    else:
        sim_obsm = X_in.copy()
        out_target = f"obsm['{writes_to[1]}']"

    # Group labels for set-to-set behavior
    if args.celltype_col and args.celltype_col in adata.obs:
        group_labels = adata.obs[args.celltype_col].astype(str).values
        unique_groups = np.unique(group_labels)
    else:
        group_labels = np.array(["__ALL__"] * n_total)
        unique_groups = np.array(["__ALL__"])

    # Control pools (group-specific with fallback to global)
    all_control_indices = np.where(ctl_mask)[0]

    def group_control_indices(group_name: str) -> np.ndarray:
        if group_name == "__ALL__":
            return all_control_indices
        grp_mask = group_labels == group_name
        grp_ctl = np.where(grp_mask & ctl_mask)[0]
        return grp_ctl if len(grp_ctl) > 0 else all_control_indices

    # default pert vector when unmapped label shows up
    if control_pert in pert_onehot_map:
        default_pert_vec = pert_onehot_map[control_pert].float().clone()
    else:
        default_pert_vec = torch.zeros(pert_dim, dtype=torch.float32)
        if pert_dim and pert_dim > 0:
            default_pert_vec[0] = 1.0

    if not args.quiet:
        print("Running virtual experiment (homogeneous per-perturbation forward passes; controls included)...")

    model_device = next(model.parameters()).device
    with torch.no_grad():
        for g in unique_groups:
            grp_idx = np.where(group_labels == g)[0]
            if len(grp_idx) == 0:
                continue

            # control pool for this group (fallback to global if empty)
            grp_ctrl_pool = group_control_indices(g)
            if len(grp_ctrl_pool) == 0:
                if not args.quiet:
                    print(f"Group '{g}': no control cells available anywhere; leaving rows unchanged.")
                continue

            # --- iterate by perturbation so each forward pass is homogeneous ---
            grp_perts = np.unique(pert_names_all[grp_idx])
            POSTFIX_WIDTH = 30
            pbar = tqdm(
                grp_perts,
                desc=f"Group {g}",
                bar_format="{l_bar}{bar}{r_bar}",  # r_bar already has n/total, time, rate, and postfix
                dynamic_ncols=True,
                disable=args.quiet,
            )
            for p in pbar:
                current_postfix = f"Pert: {p}"
                pbar.set_postfix_str(f"{current_postfix:<{POSTFIX_WIDTH}.{POSTFIX_WIDTH}}")

                idxs = grp_idx[pert_names_all[grp_idx] == p]
                if len(idxs) == 0:
                    continue

                # one-hot vector for this perturbation (repeat across window)
                vec = pert_onehot_map.get(p, None)
                if vec is None:
                    vec = default_pert_vec
                    if not args.quiet:
                        print(f"  (group {g}) pert '{p}' not in mapping; using control fallback one-hot.")

                start = 0
                while start < len(idxs):
                    end = min(start + cell_set_len, len(idxs))
                    idx_window = idxs[start:end]
                    win_size = len(idx_window)

                    # 1) Sample matched control basals (with replacement)
                    sampled_ctrl_idx = rng.choice(grp_ctrl_pool, size=win_size, replace=True)
                    ctrl_basal = X_in[sampled_ctrl_idx, :]  # [win, E_in]

                    # 2) Build homogeneous pert one-hots
                    pert_oh = vec.float().unsqueeze(0).repeat(win_size, 1)  # [win, pert_dim]

                    # 3) Batch indices (optional)
                    if uses_batch_encoder and batch_indices_all is not None:
                        bi = torch.tensor(batch_indices_all[idx_window], dtype=torch.long)  # [win]
                    else:
                        bi = None

                    # 4) Forward pass (homogeneous pert in this window)
                    batch = prepare_batch(
                        ctrl_basal_np=ctrl_basal,
                        pert_onehots=pert_oh,
                        batch_indices=bi,
                        pert_names=[p] * win_size,
                        device=model_device,
                    )
                    batch_out = model.predict_step(batch, batch_idx=0, padded=False)

                    # 5) Choose output to write
                    if (
                        writes_to[0] == ".X"
                        and ("pert_cell_counts_preds" in batch_out)
                        and (batch_out["pert_cell_counts_preds"] is not None)
                    ):
                        preds = (
                            batch_out["pert_cell_counts_preds"].detach().cpu().numpy().astype(np.float32)
                        )  # [win, G]
                    else:
                        preds = batch_out["preds"].detach().cpu().numpy().astype(np.float32)  # [win, D]

                    # 6) Write predictions for these rows (controls included)
                    if writes_to[0] == ".X":
                        if preds.shape[1] == sim_X.shape[1]:
                            sim_X[idx_window, :] = preds
                        else:
                            if not args.quiet:
                                print(
                                    f"Dimension mismatch for X (got {preds.shape[1]} vs {sim_X.shape[1]}). "
                                    f"Falling back to adata.obsm['X_state_pred']."
                                )
                            if "X_state_pred" not in adata.obsm:
                                adata.obsm["X_state_pred"] = np.zeros((n_total, preds.shape[1]), dtype=np.float32)
                            adata.obsm["X_state_pred"][idx_window, :] = preds
                            out_target = "obsm['X_state_pred']"
                    else:
                        if preds.shape[1] == sim_obsm.shape[1]:
                            sim_obsm[idx_window, :] = preds
                        else:
                            side_key = f"{writes_to[1]}_pred"
                            if not args.quiet:
                                print(
                                    f"Dimension mismatch for obsm['{writes_to[1]}'] "
                                    f"(got {preds.shape[1]} vs {sim_obsm.shape[1]}). "
                                    f"Writing to adata.obsm['{side_key}'] instead."
                                )
                            if side_key not in adata.obsm:
                                adata.obsm[side_key] = np.zeros((n_total, preds.shape[1]), dtype=np.float32)
                            adata.obsm[side_key][idx_window, :] = preds
                            out_target = f"obsm['{side_key}']"

                    start = end  # next window

    # -----------------------
    # 5) Persist the updated AnnData
    # -----------------------
    if writes_to[0] == ".X":
        if out_target == "X":
            adata.X = sim_X
    else:
        if out_target == f"obsm['{writes_to[1]}']":
            adata.obsm[writes_to[1]] = sim_obsm

    output_path = args.output or args.adata.replace(".h5ad", "_simulated.h5ad")
    adata.write_h5ad(output_path)

    # -----------------------
    # 6) Summary
    # -----------------------
    print("\n=== Inference complete ===")
    print(f"Input cells:         {n_total}")
    print(f"Controls simulated:  {n_controls}")
    print(f"Treated simulated:   {n_nonctl}")
    print(f"Wrote predictions to adata.{out_target}")
    print(f"Saved:               {output_path}")
