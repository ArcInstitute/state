import argparse as ap


def add_arguments_heatmap(parser: ap.ArgumentParser):
    """
    CLI for pathway heatmap analysis with GO MF pathway upregulation.
    """

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )

    parser.add_argument(
        "--test-time-finetune",
        type=int,
        default=0,
        help="If >0, run test-time fine-tuning for the specified number of epochs on only control cells.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="run all metrics, minimal, only de metrics, or only output adatas",
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="If set, only run prediction without evaluation metrics.",
    )

    parser.add_argument(
        "--shared-only",
        action="store_true",
        help=("If set, restrict predictions/evaluation to perturbations shared between train and test (train ∩ test)."),
    )

    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="If set, evaluate the model on the training data rather than on the test data.",
    )
    parser.add_argument(
        "--eval-cell-type",
        type=str,
        default=None,
        help=(
            "If provided, restrict inference and metrics to the specified cell type; applies to train/test loaders."
        ),
    )

    # Optional: apply directional shift on a chosen index using control distributions
    parser.add_argument(
        "--shift-index",
        type=int,
        default=None,
        help="If set, apply a ±2σ shift to this index across core_cells using control distributions.",
    )
    parser.add_argument(
        "--shift-direction",
        type=str,
        default=None,
        choices=["up", "down"],
        help="Direction for the 2σ shift applied to --shift-index. Requires --shift-index.",
    )

    parser.add_argument(
        "--test-time-heat-map",
        action="store_true",
        help="If set, run test-time heat map analysis with position upregulation.",
    )
    parser.add_argument(
        "--phase-one-only",
        action="store_true",
        help="If set, run only phase one to save core cell real embeddings per perturbation.",
    )
    parser.add_argument(
        "--heatmap-output-path",
        type=str,
        default=None,
        help="Path to save the matplotlib heatmap visualization. If not provided, defaults to <results-dir>/position_upregulation_heatmap.png",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results. If not provided, defaults to <output-dir>/eval_<checkpoint>",
    )
    parser.add_argument(
        "--heatmap-snapshots-only",
        action="store_true",
        help=(
            "Compute and persist pathway-upregulated core cell batches without running model inference or generating heatmaps."
        ),
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="/home/dhruvgautam/annotations/replogle_go_annotations.pkl", #/home/dhruvgautam/annotations/var_dims_gene_go_annotations.json
        help="Path to the hvg gene annotations file.",
    )
    parser.add_argument(
        "--annotation-field",
        type=str,
        default="go_reactome_paths",
        help=(
            "Field name in structured annotation data to use for pathway grouping (e.g., 'go_cc_paths'). "
            "Ignored when loading JSON files that map genes directly to pathways."
        ),
    )
    

def run_tx_heatmap(args: ap.ArgumentParser, *, phase_one_only: bool = False):
    import logging
    import os
    import sys
    import copy

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    import json
    import uuid
    from datetime import datetime
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def _prepare_for_serialization(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().copy()
        if isinstance(obj, dict):
            return {k: _prepare_for_serialization(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_prepare_for_serialization(v) for v in obj]
        return obj

    def _save_numpy_snapshot(obj, path, description=None):
        serializable = _prepare_for_serialization(obj)
        try:
            np.save(path, serializable, allow_pickle=True)
            if description:
                logger.info("Saved %s to %s", description, path)
            else:
                logger.info("Saved snapshot to %s", path)
        except Exception as e:
            log_desc = description or "snapshot"
            logger.warning("Failed to save %s to %s: %s", log_desc, path, e)

    def _clone_core_cells(src):
        cloned = {}
        for k, v in src.items():
            if isinstance(v, torch.Tensor):
                cloned[k] = v.clone()
            else:
                try:
                    cloned[k] = copy.deepcopy(v)
                except Exception:
                    cloned[k] = v
        return cloned

    results_dir_default = (
        args.results_dir
        if args.results_dir is not None
        else os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
    )

    snapshots_only = getattr(args, "heatmap_snapshots_only", False)

    torch.multiprocessing.set_sharing_strategy("file_system")

    def _to_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, torch.Tensor):
            try:
                return [x.item() if x.dim() == 0 else x for x in value]
            except Exception:
                return value.tolist()
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    def _resolve_celltype_key(batch):
        candidate_keys = []
        base_key = getattr(data_module, "cell_type_key", None)
        if base_key:
            candidate_keys.append(base_key)
        alias_keys = getattr(data_module, "cell_type_key_aliases", None)
        if isinstance(alias_keys, (list, tuple)):
            candidate_keys.extend(alias_keys)
        alias_keys_alt = getattr(data_module, "celltype_key_aliases", None)
        if isinstance(alias_keys_alt, (list, tuple)):
            candidate_keys.extend(alias_keys_alt)
        candidate_keys.extend(
            [
                "celltype_name",
                "cell_type",
                "celltype",
                "cell_line",
            ]
        )
        seen = set()
        ordered_candidates = []
        for key in candidate_keys:
            if not key or key in seen:
                continue
            seen.add(key)
            ordered_candidates.append(key)
            if key in batch:
                return key, ordered_candidates
        return None, ordered_candidates

    celltype_batches_seen = False

    def _filter_batch_by_celltype(batch, *, target_celltype):
        nonlocal celltype_batches_seen
        if target_celltype is None:
            return batch
        celltype_key, attempted_keys = _resolve_celltype_key(batch)
        if celltype_key is None:
            available_keys = [k for k in batch.keys() if isinstance(k, str)]
            available_preview = ", ".join(sorted(available_keys)[:10])
            raise ValueError(
                "--eval-cell-type requested cell type filtering but none of the expected keys (%s) were present in batch data. Available batch keys: %s%s"
                % (
                    ", ".join(attempted_keys) if attempted_keys else "none",
                    available_preview,
                    "..." if len(available_keys) > 10 else "",
                )
            )

        target_norm = str(target_celltype).lower()
        celltypes = _to_list(batch[celltype_key])
        mask_values = []
        for ct in celltypes:
            try:
                match = str(ct).lower() == target_norm
            except Exception:
                match = False
            mask_values.append(match)

        if not mask_values:
            return None

        mask = torch.tensor(mask_values, dtype=torch.bool)
        if mask.sum().item() == 0:
            return None

        filtered = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                try:
                    mask_device = mask.to(v.device) if mask.device != v.device else mask
                    filtered_val = v[mask_device]
                    if filtered_val.shape[0] == 0:
                        return None
                    filtered[k] = filtered_val
                except Exception:
                    filtered[k] = v
            else:
                vals = _to_list(v)
                mask_list = mask.tolist()
                filtered[k] = [vals[i] for i, keep in enumerate(mask_list[: len(vals)]) if keep]
        celltype_batches_seen = True
        return filtered

    def _iter_batches(dataloader, *, target_celltype=None):
        if target_celltype is None:
            for batch in dataloader:
                yield batch
            return
        for batch in dataloader:
            filtered = _filter_batch_by_celltype(batch, target_celltype=target_celltype)
            if filtered is None:
                continue
            yield filtered

    def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device, *, filter_batch_fn=None):
        """
        Perform test-time fine-tuning on only control cells.
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        logger.info(f"Starting test-time fine-tuning for {ft_epochs} epoch(s) on control cells only.")
        for epoch in range(ft_epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}/{ft_epochs}", leave=True)
            for batch in pbar:
                if filter_batch_fn is not None:
                    batch = filter_batch_fn(batch)
                    if batch is None:
                        continue
                # Check if this batch contains control cells
                first_pert = (
                    batch["pert_name"][0] if isinstance(batch["pert_name"], list) else batch["pert_name"][0].item()
                )
                if first_pert != control_pert:
                    continue

                # Move batch data to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                optimizer.zero_grad()
                loss = model.training_step(batch, batch_idx=0, padded=False)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
            logger.info(f"Finetune epoch {epoch + 1}/{ft_epochs}, mean loss: {mean_loss}")
        model.eval()

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    if not os.path.isabs(run_output_dir):
        run_output_dir = os.path.abspath(run_output_dir)

    if not os.path.exists(run_output_dir):
        inferred_run_dir = args.output_dir
        if os.path.exists(inferred_run_dir):
            logger.warning(
                "Run directory %s not found; falling back to config directory %s",
                run_output_dir,
                inferred_run_dir,
            )
            run_output_dir = inferred_run_dir
        else:
            raise FileNotFoundError(
                "Could not resolve run directory. Checked: %s and %s" % (run_output_dir, inferred_run_dir)
            )

    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    data_module = PerturbationDataModule.load_state(data_module_path)
    data_module.setup(stage="test")
    logger.info("Loaded data module from %s", data_module_path)

    # Seed everything
    pl.seed_everything(cfg["training"]["train_seed"])

    # 3. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    # Determine model class and load
    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    # Import the correct model class
    if model_class_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_class_name.lower() in ["neuralot", "pertsets", "state"]:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel

    elif model_class_name.lower() in ["globalsimplesum", "perturb_mean"]:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_class_name.lower() in ["celltypemean", "context_mean"]:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_class_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # 4. Test-time fine-tuning if requested
    data_module.batch_size = 1
    filter_celltype = getattr(args, "eval_cell_type", None)

    if args.test_time_finetune > 0:
        control_pert = data_module.get_control_pert()
        run_test_time_finetune(
            model,
            data_module.train_dataloader(test=True) if args.eval_train_data else data_module.test_dataloader(),
            args.test_time_finetune,
            control_pert,
            device=next(model.parameters()).device,
            filter_batch_fn=(
                (lambda batch: _filter_batch_by_celltype(batch, target_celltype=filter_celltype))
                if filter_celltype is not None
                else None
            ),
        )
        logger.info("Test-time fine-tuning complete.")

    # 5. Run inference on test set
    data_module.setup(stage="test")
    base_scan_loader = data_module.train_dataloader(test=True) if args.eval_train_data else data_module.test_dataloader()

    celltype_filter = getattr(args, "eval_cell_type", None)

    def _scan_batches():
        return _iter_batches(base_scan_loader, target_celltype=celltype_filter) if celltype_filter is not None else base_scan_loader

    scan_loader = _scan_batches()

    if scan_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    logger.info("Preparing a fixed batch of 64 control cells (core_cells) and enumerating perturbations...")

    control_pert = data_module.get_control_pert()

    # Collect unique perturbation names from the loader without running the model
    unique_perts = []
    seen_perts = set()
    for batch in _scan_batches():
        names = _to_list(batch.get("pert_name", []))
        for n in names:
            if isinstance(n, torch.Tensor):
                try:
                    n = n.item()
                except Exception:
                    n = str(n)
            if n not in seen_perts:
                seen_perts.add(n)
                unique_perts.append(n)

    if celltype_filter is not None and not celltype_batches_seen:
        raise ValueError(
            f"Requested eval cell type '{celltype_filter}' not found in data loader batches; cannot proceed."
        )

    if control_pert in seen_perts:
        logger.info(f"Found {len(unique_perts)} total perturbations (including control '{control_pert}').")
    else:
        logger.warning("Control perturbation not observed in test loader perturbation names.")

    # Build a single fixed batch of exactly 64 control cells
    target_core_n = 64
    core_cells = None
    accum = {}

    def _append_field(store, key, value):
        if key not in store:
            store[key] = []
        store[key].append(value)

    # Iterate again to collect control cells only
    for batch in _scan_batches():
        names = _to_list(batch.get("pert_name", []))
        # Build a mask for control entries when possible
        mask = None
        if len(names) > 0:
            mask = torch.tensor([str(x) == str(control_pert) for x in names], dtype=torch.bool)
            if mask.sum().item() == 0:
                continue
        else:
            # If no names provided in batch, skip (cannot verify control)
            continue

        # Slice each tensor field by mask and accumulate until we have 64
        current_count = 0 if "_count" not in accum else accum["_count"]
        take = min(target_core_n - current_count, int(mask.sum().item()))
        if take <= 0:
            break

        # Identify keys to carry forward; prefer tensors and essential metadata
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                try:
                    vsel = v[mask][:take].detach().clone()
                except Exception:
                    # fallback: try first dimension slice
                    vsel = v[:take].detach().clone()
                _append_field(accum, k, vsel)
            else:
                # For non-tensor fields, convert to list and slice by mask when possible
                vals = _to_list(v)
                try:
                    selected_vals = [vals[i] for i, m in enumerate(mask.tolist()) if m][:take]
                except Exception:
                    selected_vals = vals[:take]
                _append_field(accum, k, selected_vals)

        accum["_count"] = current_count + take
        if accum["_count"] >= target_core_n:
            break

    if accum.get("_count", 0) < target_core_n:
        raise RuntimeError(f"Could not assemble {target_core_n} control cells for core_cells; gathered {accum.get('_count', 0)}.")

    # Collate accumulated pieces into a single batch dict of length 64
    core_cells = {}
    for k, parts in accum.items():
        if k == "_count":
            continue
        if len(parts) == 1:
            val = parts[0]
        else:
            if isinstance(parts[0], torch.Tensor):
                val = torch.cat(parts, dim=0)
            else:
                merged = []
                for p in parts:
                    merged.extend(_to_list(p))
                val = merged
        # Ensure final length == 64
        if isinstance(val, torch.Tensor):
            core_cells[k] = val[:target_core_n]
        else:
            core_cells[k] = _to_list(val)[:target_core_n]

    logger.info(f"Constructed core_cells batch with size {target_core_n}.")

    os.makedirs(results_dir_default, exist_ok=True)
    baseline_core_cells_path = os.path.join(results_dir_default, "core_cells_baseline.npy")
    _save_numpy_snapshot(core_cells, baseline_core_cells_path, description="baseline core_cells batch (control cells)")

    # Compute distributions for each position across ALL control cells in the test loader
    # Strategy: determine a 2D vector key from the first batch, then aggregate all control rows
    vector_key_candidates = ["ctrl_cell_emb", "pert_cell_emb", "X"]
    dist_source_key = None
    # Find key by peeking one batch
    for b in _scan_batches():
        for cand in vector_key_candidates:
            if cand in b and isinstance(b[cand], torch.Tensor) and b[cand].dim() == 2:
                dist_source_key = cand
                break
        if dist_source_key is None:
            # fallback: any 2D tensor
            for k, v in b.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    dist_source_key = k
                    break
        # break after first batch inspected
        break
    if dist_source_key is None:
        raise RuntimeError("Could not find a 2D tensor in test loader batches to compute per-dimension distributions.")

    # Aggregate all control rows for the chosen key
    control_rows = []
    for batch in _scan_batches():
        names = _to_list(batch.get("pert_name", []))
        if len(names) == 0:
            continue
        mask = torch.tensor([str(x) == str(control_pert) for x in names], dtype=torch.bool)
        if mask.sum().item() == 0:
            continue
        vec = batch.get(dist_source_key, None)
        if isinstance(vec, torch.Tensor) and vec.dim() == 2:
            try:
                control_rows.append(vec[mask].detach().cpu().float())
            except Exception:
                # fallback: take leading rows equal to mask sum
                take = int(mask.sum().item())
                if take > 0:
                    control_rows.append(vec[:take].detach().cpu().float())

    if len(control_rows) == 0:
        raise RuntimeError("No control rows found to compute distributions.")

    control_vectors_all = torch.cat(control_rows, dim=0)  # [Nc, D]
    D = control_vectors_all.shape[1]
    if D != 2000:
        logger.warning(f"Expected vector dimension 2000; found {D}. Proceeding with {D} dimensions.")

    control_mean = control_vectors_all.mean(dim=0)
    control_std = control_vectors_all.std(dim=0, unbiased=False).clamp_min(1e-8)

    # Save distributions to results directory later; keep in scope for optional shifting
    distributions = {
        "key": dist_source_key,
        "mean": control_mean.numpy(),
        "std": control_std.numpy(),
        "dim": int(D),
        "num_cells": int(control_vectors_all.shape[0]),
    }

    def apply_shift_to_core_cells(index: int, upregulate: bool):
        """Apply ±2σ shift at a single index across all vectors in core_cells.

        - index: integer in [0, D)
        - upregulate: True for +2σ, False for -2σ
        Operates in-place on the tensor stored at distributions['key'] inside core_cells.
        """
        nonlocal core_cells, distributions
        if index < 0 or index >= distributions["dim"]:
            raise ValueError(f"Index {index} is out of bounds for dimension {distributions['dim']}")
        shift_value = (2.0 if upregulate else -2.0) * float(distributions["std"][index])
        key = distributions["key"]
        tensor = core_cells[key]
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
            raise RuntimeError(f"Core cell field '{key}' is not a 2D tensor")
        tensor[:, index] = tensor[:, index] + shift_value
        core_cells[key] = tensor

    # Optionally apply shift based on CLI flags before running inference
    if args.shift_index is not None:
        if args.shift_direction is None:
            raise ValueError("--shift-direction is required when --shift-index is provided")
        apply_shift_to_core_cells(index=int(args.shift_index), upregulate=(args.shift_direction == "up"))
        logger.info(f"Applied 2σ {'up' if args.shift_direction=='up' else 'down'} shift at index {int(args.shift_index)} across core_cells")

    # Prepare perturbation ordering and, if needed, buffers for forward passes
    perts_order = list(unique_perts)

    if snapshots_only:
        logger.info("Heatmap snapshots flag set; skipping phase-one forward passes through the model.")
        output_dim = var_dims["output_dim"]
        gene_dim = var_dims["gene_dim"]
        hvg_dim = var_dims["hvg_dim"]
        final_preds = None
        final_reals = None
        final_X_hvg = None
        final_pert_cell_counts_preds = None
        normal_preds_per_pert = {}
        real_preds_per_pert = {}
        store_raw_expression = False
    else:
        logger.info("Generating predictions: one forward pass per perturbation on core_cells...")
        num_cells = len(perts_order) * target_core_n
        output_dim = var_dims["output_dim"]
        gene_dim = var_dims["gene_dim"]
        hvg_dim = var_dims["hvg_dim"]

        # Phase 1: Normal inference on all perturbations
        final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
        final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

        # Phase 2: Store normal predictions for distance computation
        normal_preds_per_pert = {}  # pert_name -> [64, output_dim] array
        real_preds_per_pert = {}  # pert_name -> [64, output_dim] array

        store_raw_expression = (
            data_module.embed_key is not None
            and data_module.embed_key != "X_hvg"
            and cfg["data"]["kwargs"]["output_space"] == "gene"
        ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")

        final_X_hvg = None
        final_pert_cell_counts_preds = None
        if store_raw_expression:
            # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
            if cfg["data"]["kwargs"]["output_space"] == "gene":
                final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
            if cfg["data"]["kwargs"]["output_space"] == "all":
                final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)
    device = next(model.parameters()).device

    # Prepare perturbation one-hot/embedding map for the pert encoder
    pert_onehot_map = getattr(data_module, "pert_onehot_map", None)
    if pert_onehot_map is None:
        try:
            map_path = os.path.join(run_output_dir, "pert_onehot_map.pt")
            if os.path.exists(map_path):
                pert_onehot_map = torch.load(map_path, weights_only=False)
            else:
                logger.warning(f"pert_onehot_map.pt not found at {map_path}; proceeding without explicit pert_emb overrides")
                pert_onehot_map = {}
        except Exception as e:
            logger.warning(f"Failed to load pert_onehot_map.pt: {e}")
            pert_onehot_map = {}

    def _prepare_pert_emb(pert_name: str, length: int, device: torch.device):
        vec = None
        try:
            vec = pert_onehot_map.get(pert_name, None)
            if vec is None and control_pert in pert_onehot_map:
                vec = pert_onehot_map[control_pert]
        except Exception:
            vec = None
        if vec is None:
            # Fallback to zeros with model.pert_dim if mapping is unavailable
            pert_dim = getattr(model, "pert_dim", var_dims.get("pert_dim", 0))
            if pert_dim <= 0:
                raise RuntimeError("Could not determine pert_dim to build pert_emb")
            vec = torch.zeros(pert_dim)
        return vec.float().unsqueeze(0).repeat(length, 1).to(device)

    current_idx = 0

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_pert_barcodes = []
    all_ctrl_barcodes = []

    if not snapshots_only:
        with torch.no_grad():
            for p_idx, pert in enumerate(tqdm(perts_order, desc="Predicting", unit="pert")):
                # Build a batch by copying core_cells and swapping perturbation
                batch = {}
                for k, v in core_cells.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.clone().to(device)
                    else:
                        batch[k] = list(v)

                # Overwrite perturbation fields to target pert
                if "pert_name" in batch:
                    batch["pert_name"] = [pert for _ in range(target_core_n)]
                # Best-effort: update any index fields if present and mapping exists
                try:
                    if "pert_idx" in batch and hasattr(data_module, "get_pert_index"):
                        idx_val = int(data_module.get_pert_index(pert))
                        batch["pert_idx"] = torch.tensor([idx_val] * target_core_n, device=device)
                except Exception:
                    pass

                # Ensure perturbation embedding is set for the encoder
                batch["pert_emb"] = _prepare_pert_emb(pert, target_core_n, device)

                batch_preds = model.predict_step(batch, p_idx, padded=False)

                # Extract metadata and data directly from batch_preds
                # Handle pert_name
                batch_pert_names = []
                if isinstance(batch_preds["pert_name"], list):
                    all_pert_names.extend(batch_preds["pert_name"])
                    batch_pert_names = batch_preds["pert_name"]
                else:
                    all_pert_names.append(batch_preds["pert_name"])
                    batch_pert_names = [batch_preds["pert_name"]]

                if "pert_cell_barcode" in batch_preds:
                    if isinstance(batch_preds["pert_cell_barcode"], list):
                        all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.extend(batch_preds.get("ctrl_cell_barcode", [None] * len(batch_preds["pert_cell_barcode"])) )
                    else:
                        all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.append(batch_preds.get("ctrl_cell_barcode", None))

                # Handle celltype_name
                if isinstance(batch_preds["celltype_name"], list):
                    all_celltypes.extend(batch_preds["celltype_name"])
                else:
                    all_celltypes.append(batch_preds["celltype_name"])

                # Handle gem_group
                if isinstance(batch_preds["batch"], list):
                    all_gem_groups.extend([str(x) for x in batch_preds["batch"]])
                elif isinstance(batch_preds["batch"], torch.Tensor):
                    all_gem_groups.extend([str(x) for x in batch_preds["batch"].cpu().numpy()])
                else:
                    all_gem_groups.append(str(batch_preds["batch"]))

                batch_pred_np = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)
                batch_real_np = batch_preds["pert_cell_emb"].detach().cpu().numpy().astype(np.float32)
                batch_size = batch_pred_np.shape[0]
                final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
                final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
                
                # Store normal predictions for this perturbation for distance computation
                normal_preds_per_pert[pert] = batch_pred_np.copy()
                real_preds_per_pert[pert] = batch_real_np.copy()
                
                current_idx += batch_size

                # Handle X_hvg for HVG space ground truth
                if final_X_hvg is not None:
                    batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

                # Handle decoded gene predictions if available
                if final_pert_cell_counts_preds is not None:
                    batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                    final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

        logger.info("Phase 1 complete: Normal inference on all perturbations.")
    
    # Phase 2: Run inference with GO MF pathway groups upregulated (only if requested)
    run_phase_one_only = phase_one_only or getattr(args, "phase_one_only", False)

    if run_phase_one_only and not snapshots_only:
        os.makedirs(results_dir_default, exist_ok=True)
        if not snapshots_only:
            real_preds_path = os.path.join(results_dir_default, "core_cells_real_preds_per_pert.npy")
            try:
                np.save(real_preds_path, real_preds_per_pert, allow_pickle=True)
                logger.info(
                    "Saved real perturbed core cell embeddings for %d perturbations to %s",
                    len(real_preds_per_pert),
                    real_preds_path,
                )
            except Exception as e:
                logger.error("Failed to save core cell real predictions to %s: %s", real_preds_path, e)
                raise
        return

    if args.test_time_heat_map or snapshots_only:
        logger.info("Phase 2: Loading GO MF pathway annotations and running pathway-based upregulation...")

        results_dir = results_dir_default

        # Ensure unique heatmap directory per invocation to avoid overwriting prior outputs
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        unique_suffix = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        heatmap_results_dir = os.path.join(results_dir, "heatmap_runs", unique_suffix)

        os.makedirs(heatmap_results_dir, exist_ok=True)
        annotation_ext = os.path.splitext(args.annotation_path)[1].lower()
        annotation_source_type = "unknown"
        annotation_label = args.annotation_field
        field_suffix = (
            (annotation_label or "pathways").replace('_', '').lower()
            if (annotation_label or "").strip()
            else "pathways"
        )

        # Load gene annotations
        import pickle
        from collections import defaultdict

        pathway_to_genes = defaultdict(list)
        gene_names = var_dims.get("gene_names")
        gene_name_to_index = {str(name): idx for idx, name in enumerate(gene_names)} if gene_names is not None else {}

        if annotation_ext == ".json":
            annotation_source_type = "json"
            annotation_label = os.path.splitext(os.path.basename(args.annotation_path))[0]
            field_suffix = annotation_label.replace('_', '').lower() or "pathways"

            with open(args.annotation_path, 'r') as f:
                gene_annotations = json.load(f)

            if not isinstance(gene_annotations, dict):
                raise ValueError(
                    f"Expected JSON annotation file {args.annotation_path} to map gene names to pathway collections."
                )

            missing_genes = set()
            for gene_name, pathway_data in gene_annotations.items():
                if not pathway_data:
                    continue

                idx = gene_name_to_index.get(str(gene_name))
                if idx is None:
                    missing_genes.add(str(gene_name))
                    continue

                if isinstance(pathway_data, str):
                    pathways = [p.strip() for p in pathway_data.split(';') if p.strip()]
                else:
                    pathways = []
                    for entry in pathway_data:
                        if entry is None:
                            continue
                        pathways.append(str(entry).strip())
                    pathways = [p for p in pathways if p]

                for pathway in pathways:
                    pathway_to_genes[pathway].append(idx)

            if missing_genes:
                sample_missing = ", ".join(sorted(missing_genes)[:5])
                logger.warning(
                    "Skipped %d gene(s) from annotation file not present in model gene names (e.g., %s)",
                    len(missing_genes),
                    sample_missing,
                )
        elif annotation_ext in {".pkl", ".pickle"}:
            annotation_source_type = "pickle"
            field_suffix = (
                (args.annotation_field or "pathways").replace('_', '').lower()
                if (args.annotation_field or "").strip()
                else "pathways"
            )

            with open(args.annotation_path, 'rb') as f:
                gene_annotations = pickle.load(f)

            if not args.annotation_field:
                raise ValueError(
                    "--annotation-field must be provided when loading pickle annotation files."
                )

            for idx, data in gene_annotations.items():
                pathway_data = None
                if isinstance(data, dict):
                    pathway_data = data.get(args.annotation_field)
                else:
                    try:
                        pathway_data = data[args.annotation_field]
                    except (KeyError, TypeError):
                        pathway_data = getattr(data, args.annotation_field, None)

                if not pathway_data:
                    continue

                if isinstance(pathway_data, str):
                    pathways = [p.strip() for p in pathway_data.split(';') if p.strip()]
                else:
                    pathways = [str(p).strip() for p in pathway_data if str(p).strip()]

                try:
                    gene_index = int(idx) - 1
                except (TypeError, ValueError):
                    gene_index = gene_name_to_index.get(str(idx))

                if gene_index is None or gene_index < 0:
                    continue

                for pathway in pathways:
                    pathway_to_genes[pathway].append(gene_index)
        else:
            raise ValueError(
                f"Unsupported annotation file extension '{annotation_ext}' for {args.annotation_path}."
            )
        
        # Filter out pathways with too few genes (less than 3) to avoid noise
        filtered_pathways = {pathway: genes for pathway, genes in pathway_to_genes.items() if len(genes) >= 3}
        
        logger.info(
            "Found %d total pathways from annotation source '%s' (%s)",
            len(pathway_to_genes),
            annotation_label,
            annotation_source_type,
        )
        logger.info(f"Using {len(filtered_pathways)} pathways with 3+ genes for upregulation")
        
        # Initialize heatmap array: [num_pathways, num_perturbations]
        num_pathways = len(filtered_pathways)
        heatmap_distances = np.zeros((num_pathways, len(perts_order)), dtype=np.float32)
        pathway_names = list(filtered_pathways.keys())

        annotation_label_pretty = (annotation_label or "Annotation").replace('_', ' ').strip()
        if annotation_label_pretty:
            annotation_label_pretty = annotation_label_pretty.title()
        else:
            annotation_label_pretty = "Annotation"

        upregulated_preds_path = None
        upregulated_preds_memmap = None
        if not snapshots_only:
            if num_pathways == 0:
                logger.warning("No pathways passed filtering; skipping upregulated prediction storage.")
            else:
                try:
                    upregulated_preds_path = os.path.join(
                        heatmap_results_dir,
                        f"{field_suffix}_pathway_upregulated_preds.npy",
                    )
                    upregulated_preds_memmap = np.memmap(
                        upregulated_preds_path,
                        dtype=np.float32,
                        mode="w+",
                        shape=(num_pathways, len(perts_order), target_core_n, output_dim),
                    )
                except Exception as e:
                    logger.warning("Failed to initialize storage for upregulated predictions: %s", e)
                    upregulated_preds_path = None
                    upregulated_preds_memmap = None
        
        # Create a copy of core_cells for upregulation experiments
        original_core_cells = _clone_core_cells(core_cells)
        
        def apply_pathway_shift_to_core_cells(cell_batch: dict, gene_indices: list, upregulate: bool, target_norm: float = 2.0):
            """Apply shift to multiple gene indices with equivalent euclidean norm across pathways.
            
            This function ensures that all pathways receive the same euclidean norm perturbation:
            1. Compute individual shifts based on 2σ for each gene
            2. Calculate the euclidean norm of the shift vector
            3. Rescale the entire shift vector to match the target euclidean norm
            
            - gene_indices: list of 0-indexed gene positions
            - upregulate: True for positive shift, False for negative shift
            - target_norm: target euclidean norm for the perturbation (default: 2.0)
            Operates in-place on the tensor stored at distributions['key'] inside the provided cell_batch.
            """
            nonlocal distributions
            key = distributions['key']
            tensor = cell_batch[key]
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
                raise RuntimeError(f"Core cell field '{key}' is not a 2D tensor")
            
            if len(gene_indices) == 0:
                return
            
            # Step 1: Compute raw shift values based on 2σ for each gene
            raw_shifts = {}
            for idx in gene_indices:
                if 0 <= idx < distributions["dim"]:
                    base_shift = 2.0 * float(distributions["std"][idx])
                    raw_shifts[idx] = base_shift if upregulate else -base_shift
            
            if len(raw_shifts) == 0:
                return
                
            # Step 2: Calculate euclidean norm of the raw shift vector
            shift_values = np.array(list(raw_shifts.values()))
            current_norm = np.linalg.norm(shift_values)
            
            # Step 3: Rescale to target norm if current norm > 0
            if current_norm > 1e-8:  # Avoid division by zero
                scale_factor = target_norm / current_norm
                
                # Apply rescaled shifts
                for idx, raw_shift in raw_shifts.items():
                    scaled_shift = raw_shift * scale_factor
                    tensor[:, idx] = tensor[:, idx] + scaled_shift
            else:
                # Fallback: if all std deviations are zero, apply uniform shift
                uniform_shift = target_norm / np.sqrt(len(raw_shifts))
                for idx in raw_shifts.keys():
                    shift_value = uniform_shift if upregulate else -uniform_shift
                    tensor[:, idx] = tensor[:, idx] + shift_value

        def compute_pathway_core_cell_snapshots(base_core_cells: dict, pathways: dict) -> list:
            snapshots = []
            for pathway_name, gene_indices in pathways.items():
                shifted_cells = _clone_core_cells(base_core_cells)
                apply_pathway_shift_to_core_cells(shifted_cells, gene_indices, upregulate=True)
                snapshots.append(
                    {
                        "pathway_name": pathway_name,
                        "gene_indices": list(gene_indices),
                        "core_cells": shifted_cells,
                    }
                )
            return snapshots

        shifted_core_cells_path = os.path.join(heatmap_results_dir, f"{field_suffix}_core_cells_upregulated.npy")
        pathway_core_cells_snapshots = compute_pathway_core_cell_snapshots(original_core_cells, filtered_pathways)
        _save_numpy_snapshot(
            pathway_core_cells_snapshots,
            shifted_core_cells_path,
            description=f"core_cells upregulated snapshots ({len(pathway_core_cells_snapshots)} pathways)",
        )
        if len(pathway_core_cells_snapshots) == 0:
            logger.warning("No pathway core cell snapshots generated (0 pathways passed filtering).")

        if not snapshots_only:
            with torch.no_grad():
                for pathway_idx, snapshot in enumerate(
                    tqdm(pathway_core_cells_snapshots, desc="Upregulating pathways", unit="pathway")
                ):
                    core_cells_upregulated = snapshot["core_cells"]
                    # Run inference for all perturbations with this pathway upregulated
                    for p_idx, pert in enumerate(perts_order):
                        # Build batch by copying upregulated core_cells
                        batch = {}
                        for k, v in core_cells_upregulated.items():
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.clone().to(device)
                            else:
                                batch[k] = list(v)

                        # Overwrite perturbation fields
                        if "pert_name" in batch:
                            batch["pert_name"] = [pert for _ in range(target_core_n)]
                        try:
                            if "pert_idx" in batch and hasattr(data_module, "get_pert_index"):
                                idx_val = int(data_module.get_pert_index(pert))
                                batch["pert_idx"] = torch.tensor([idx_val] * target_core_n, device=device)
                        except Exception:
                            pass

                        # Ensure perturbation embedding is set for the encoder
                        batch["pert_emb"] = _prepare_pert_emb(pert, target_core_n, device)

                        # Get predictions with upregulated pathway
                        batch_preds = model.predict_step(batch, p_idx, padded=False)
                        upregulated_preds = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)

                        if upregulated_preds_memmap is not None:
                            upregulated_preds_memmap[pathway_idx, p_idx, :, :] = upregulated_preds

                        # Compute euclidean distance between normal and upregulated predictions
                        normal_preds = normal_preds_per_pert[pert]  # [64, output_dim]
                        distance = np.linalg.norm(upregulated_preds - normal_preds, axis=1).mean()  # Mean across 64 cells
                        heatmap_distances[pathway_idx, p_idx] = distance
        
        logger.info(
            "Phase 2 core cell snapshots ready for %d pathways from annotation source '%s'.",
            len(pathway_core_cells_snapshots),
            annotation_label or annotation_source_type,
        )

        if snapshots_only:
            logger.info("Snapshots-only mode: skipping distance computations, heatmap data, and visualization generation.")
            return

        logger.info(
            "Phase 2 complete: Upregulated inference for all pathways from annotation source '%s'.",
            annotation_label or annotation_source_type,
        )

        if upregulated_preds_memmap is not None:
            upregulated_preds_memmap.flush()
            logger.info(f"Saved upregulated prediction tensors to {upregulated_preds_path}")

        # Create filename based on annotation field
        # Save heatmap data
        try:
            heatmap_path = os.path.join(heatmap_results_dir, f"{field_suffix}_pathway_upregulation_heatmap.npy")
            np.save(heatmap_path, heatmap_distances)
            
            # Save pathway information
            pathway_info_path = os.path.join(heatmap_results_dir, f"{field_suffix}_pathways_info.json")
            pathway_info = {
                "pathway_names": pathway_names,
                "pathway_to_genes": {pathway: genes for pathway, genes in filtered_pathways.items()},
                "total_pathways": len(pathway_to_genes),
                "filtered_pathways": len(filtered_pathways),
                "min_genes_per_pathway": 3
            }
            with open(pathway_info_path, "w") as f:
                json.dump(pathway_info, f, indent=2)
            
            # Save metadata for the heatmap
            heatmap_meta = {
                "shape": [num_pathways, len(perts_order)],
                "description": (
                    f"Euclidean distance heatmap: rows={annotation_label_pretty} pathways, cols=perturbations"
                ),
                "perturbations": perts_order,
                "pathway_names": pathway_names,
                "distance_type": "mean_euclidean_norm_across_64_cells",
                "upregulation": "equivalent_euclidean_norm_perturbation_rescaled_from_2std_per_gene",
                "annotation_field": annotation_label if annotation_source_type != "json" else None,
                "annotation_source_type": annotation_source_type,
                "upregulated_preds_path": upregulated_preds_path,
            }
            heatmap_meta_path = os.path.join(heatmap_results_dir, f"{field_suffix}_pathway_upregulation_heatmap.meta.json")
            with open(heatmap_meta_path, "w") as f:
                json.dump(heatmap_meta, f, indent=2)
            
            logger.info(
                "Saved %s pathway upregulation heatmap to %s",
                annotation_label_pretty,
                heatmap_path,
            )
            logger.info(f"Heatmap shape: {heatmap_distances.shape} (pathways x perturbations)")
        except Exception as e:
            logger.warning(f"Failed to save heatmap data: {e}")
        
        # Create and save matplotlib heatmap visualization
        try:
            # Determine output path for heatmap image
            if args.heatmap_output_path is not None:
                # If user provided a path, make it unique per run as well
                base, ext = os.path.splitext(args.heatmap_output_path)
                heatmap_img_path = f"{base}_{unique_suffix}{ext or '.png'}"
            else:
                heatmap_img_path = os.path.join(heatmap_results_dir, f"{field_suffix}_pathway_upregulation_heatmap.png")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(heatmap_img_path), exist_ok=True)
            
            # Create the heatmap with appropriate size
            fig_width = max(12, len(perts_order) * 0.3)
            fig_height = max(8, num_pathways * 0.05)  # Smaller height per pathway since we have fewer rows
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create heatmap with proper labels
            im = ax.imshow(heatmap_distances, cmap='viridis', aspect='auto')
            
            # Set labels and title
            ax.set_xlabel('Perturbations')
            ax.set_ylabel(f'{args.annotation_field.replace("_", " ").title()} Pathways')
            ax.set_title(f'{args.annotation_field.replace("_", " ").title()} Pathway Upregulation Impact Heatmap\n(Euclidean Distance from Normal Predictions)')
            
            # Set x-axis labels (perturbations)
            ax.set_xticks(range(len(perts_order)))
            ax.set_xticklabels(perts_order, rotation=45, ha='right', fontsize=8)
            
            # Set y-axis labels (pathways) - show pathway names, truncated if too long
            ax.set_yticks(range(num_pathways))
            truncated_pathway_names = []
            for pathway_name in pathway_names:
                # Remove common prefixes and truncate long names
                clean_name = pathway_name
                # Remove common GO prefixes
                for prefix in ['GOMF_', 'GOCC_', 'GOBP_']:
                    clean_name = clean_name.replace(prefix, '')
                if len(clean_name) > 30:
                    clean_name = clean_name[:27] + '...'
                truncated_pathway_names.append(clean_name)
            ax.set_yticklabels(truncated_pathway_names, fontsize=6)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Euclidean Distance', rotation=270, labelpad=20)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(heatmap_img_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close to free memory
            
            logger.info(f"Saved {args.annotation_field} pathway heatmap visualization to {heatmap_img_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create heatmap visualization: {e}")
    else:
        logger.info("Skipping heatmap analysis (--test-time-heat-map not set)")

    logger.info("Creating anndatas from predictions from manual loop...")

    # Build pandas DataFrame for obs and var
    df_dict = {
        data_module.pert_col: all_pert_names,
        data_module.cell_type_key: all_celltypes,
        data_module.batch_col: all_gem_groups,
    }

    if len(all_pert_barcodes) > 0:
        df_dict["pert_cell_barcode"] = all_pert_barcodes
        df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

    obs = pd.DataFrame(df_dict)

    gene_names = var_dims["gene_names"]
    var = pd.DataFrame({"gene_names": gene_names})

    if final_X_hvg is not None:
        if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
            gene_names = np.load(
                "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
            )
            var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - using the decoded gene expression values
        adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs, var=var)
        # Create adata for real - using the true gene expression values
        adata_real = anndata.AnnData(X=final_X_hvg, obs=obs, var=var)

        # add the embedding predictions
        adata_pred.obsm[data_module.embed_key] = final_preds
        adata_real.obsm[data_module.embed_key] = final_reals
        logger.info(f"Added predicted embeddings to adata.obsm['{data_module.embed_key}']")
    else:
        # if len(gene_names) != final_preds.shape[1]:
        #     gene_names = np.load(
        #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
        #     )
        #     var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - model was trained on gene expression space already
        # adata_pred = anndata.AnnData(X=final_preds, obs=obs, var=var)
        adata_pred = anndata.AnnData(X=final_preds, obs=obs)
        # Create adata for real - using the true gene expression values
        # adata_real = anndata.AnnData(X=final_reals, obs=obs, var=var)
        adata_real = anndata.AnnData(X=final_reals, obs=obs)

    # Optionally filter to perturbations seen in at least one training context
    if args.shared_only:
        try:
            shared_perts = data_module.get_shared_perturbations()
            if len(shared_perts) == 0:
                logger.warning("No shared perturbations between train and test; skipping filtering.")
            else:
                logger.info(
                    "Filtering to %d shared perturbations present in train ∩ test.",
                    len(shared_perts),
                )
                mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
                before_n = adata_pred.n_obs
                adata_pred = adata_pred[mask].copy()
                adata_real = adata_real[mask].copy()
                logger.info(
                    "Filtered cells: %d -> %d (kept only seen perturbations)",
                    before_n,
                    adata_pred.n_obs,
                )
        except Exception as e:
            logger.warning(
                "Failed to filter by shared perturbations (%s). Proceeding without filter.",
                str(e),
            )

    # Save the AnnData objects
    results_dir = results_dir_default
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

    adata_pred.write_h5ad(adata_pred_path)
    adata_real.write_h5ad(adata_real_path)

    logger.info(f"Saved adata_pred to {adata_pred_path}")
    logger.info(f"Saved adata_real to {adata_real_path}")

    # Save per-dimension control-cell distributions for reproducibility
    try:
        dist_out = {
            "key": distributions["key"],
            "dim": distributions["dim"],
            "num_cells": distributions["num_cells"],
        }
        dist_out_path = os.path.join(results_dir, "control_distributions.meta.json")
        with open(dist_out_path, "w") as f:
            json.dump(dist_out, f)
        np.save(os.path.join(results_dir, "control_mean.npy"), distributions["mean"])  # [D]
        np.save(os.path.join(results_dir, "control_std.npy"), distributions["std"])    # [D]
        logger.info("Saved control-cell per-dimension mean/std distributions")
    except Exception as e:
        logger.warning(f"Failed to save control-cell distributions: {e}")

    if not args.predict_only:
        # 6. Compute metrics using cell-eval
        logger.info("Computing metrics using cell-eval...")

        control_pert = data_module.get_control_pert()

        ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
        ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

        assert len(ct_split_real) == len(ct_split_pred), (
            f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
        )

        pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
        for ct in ct_split_real.keys():
            real_ct = ct_split_real[ct]
            pred_ct = ct_split_pred[ct]

            evaluator = MetricsEvaluator(
                adata_pred=pred_ct,
                adata_real=real_ct,
                control_pert=control_pert,
                pert_col=data_module.pert_col,
                outdir=results_dir,
                prefix=ct,
                pdex_kwargs=pdex_kwargs,
                batch_size=2048,
            )

            evaluator.compute(
                profile=args.profile,
                metric_configs={
                    "discrimination_score": {
                        "embed_key": data_module.embed_key,
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {},
                    "pearson_edistance": {
                        "embed_key": data_module.embed_key,
                        "n_jobs": -1,  # set to all available cores
                    }
                    if data_module.embed_key and data_module.embed_key != "X_hvg"
                    else {
                        "n_jobs": -1,
                    },
                }
                if data_module.embed_key and data_module.embed_key != "X_hvg"
                else {},
                skip_metrics=["pearson_edistance", "clustering_agreement"],
            )


def save_core_cells_real_preds(args: ap.ArgumentParser):
    """Run only phase one of the heatmap pipeline and persist real core-cell embeddings per perturbation."""
    return run_tx_heatmap(args, phase_one_only=True)
