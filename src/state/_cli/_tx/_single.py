import argparse as ap


def add_arguments_single(parser: ap.ArgumentParser) -> None:
    """CLI for single-pass perturbation analysis on a target cell line."""

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Path to the output_dir containing the config.yaml file that was saved during training."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename relative to the output directory (default: last.ckpt).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "minimal", "de", "anndata"],
        help="Evaluation profile to run after inference.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip metric computation and only run inference.",
    )
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Restrict outputs to perturbations present in both train and test sets.",
    )
    parser.add_argument(
        "--eval-train-data",
        action="store_true",
        help="Evaluate the model on the training data instead of the test data (ignored with --core-cells-path).",
    )
    parser.add_argument(
        "--target-cell-type",
        type=str,
        default=None,
        help=(
            "Optional cell type to construct the base core cells for single perturbations."
            " Ignored when --core-cells-path is provided."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results. Defaults to <output-dir>/eval_<checkpoint>.",
    )
    parser.add_argument(
        "--first-pass-only",
        action="store_true",
        help="Run only the first round of inference and save per-perturbation predictions to a NumPy file.",
    )
    parser.add_argument(
        "--core-cells-path",
        type=str,
        default=None,
        help=(
            "Path to a NumPy .npy file containing a serialized core_cells dictionary to use instead of"
            " constructing a new control batch."
        ),
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help=(
            "Path to a TOML data configuration file to override the data paths in the loaded data module."
        ),
    )
    parser.add_argument(
        "--perturbation-npy",
        type=str,
        default=None,
        help=(
            "Optional path to a NumPy .npy/.npz file (or convertible CSV specification) that maps perturbation"
            " names to explicit encoder vectors. When provided, these vectors override the default pert_onehot_map."
        ),
    )


def run_tx_single(args: ap.ArgumentParser, *, phase_one_only: bool = False) -> None:
    import logging
    import os
    import sys
    import re

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from tqdm import tqdm

    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    phase_one_only = phase_one_only or getattr(args, "first_pass_only", False)

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
        except Exception as exc:
            logger.warning("Failed to save %s to %s: %s", description or "snapshot", path, exc)

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
        if value is None:
            return []
        return [value]

    def _normalize_field(values, length, filler=None):
        items = list(_to_list(values))
        if len(items) == 1 and length > 1:
            items = items * length
        if len(items) < length:
            items.extend([filler] * (length - len(items)))
        elif len(items) > length:
            items = items[:length]
        return items

    def _resolve_celltype_key(batch, module):
        candidate_keys = []
        base_key = getattr(module, "cell_type_key", None)
        if base_key:
            candidate_keys.append(base_key)
        alias_keys = getattr(module, "cell_type_key_aliases", None)
        if isinstance(alias_keys, (list, tuple)):
            candidate_keys.extend(alias_keys)
        alias_keys_alt = getattr(module, "celltype_key_aliases", None)
        if isinstance(alias_keys_alt, (list, tuple)):
            candidate_keys.extend(alias_keys_alt)
        candidate_keys.extend([
            "celltype_name",
            "cell_type",
            "celltype",
            "cell_line",
        ])
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

    def _ensure_tensor_vector(value, *, expected_dim=None, context="perturbation"):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(dtype=torch.float32, device="cpu")
        else:
            try:
                tensor = torch.as_tensor(value, dtype=torch.float32)
            except Exception as exc:
                raise TypeError(f"Could not convert {context} value to tensor: {exc}") from exc
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        if tensor.ndim != 1:
            raise ValueError(
                f"Expected a 1D tensor for {context}, but received shape {tuple(tensor.shape)}."
            )
        if expected_dim is not None and tensor.numel() != expected_dim:
            raise ValueError(
                f"Dimension mismatch for {context}: expected {expected_dim}, received {tensor.numel()}."
            )
        return tensor.contiguous()

    def _normalize_pert_map(raw_map, *, expected_dim=None, label="perturbation map"):
        if raw_map is None:
            return {}
        normalized = {}
        for raw_key, raw_value in raw_map.items():
            key = str(raw_key)
            try:
                tensor = _ensure_tensor_vector(
                    raw_value,
                    expected_dim=expected_dim,
                    context=f"{label}:{key}",
                )
            except Exception as exc:
                raise ValueError(f"Failed to normalize {label} entry '{key}': {exc}") from exc
            normalized[key] = tensor
        return normalized

    def _infer_combination_csv(path):
        directory, filename = os.path.split(path)
        match = re.search(r"max[_-]?drugs[_-]?(\d+)", filename, flags=re.IGNORECASE)
        if not match:
            return None
        suffix = match.group(1)
        candidates = [
            os.path.join(directory, f"average_to_genetic_reconstruction_maxdrugs{suffix}.csv"),
            os.path.join(directory, f"average_to_genetic_reconstruction_maxdrugs{suffix}.CSV"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _parse_combination_spec(spec):
        if spec is None:
            return {}
        if isinstance(spec, (float, np.floating)) and np.isnan(spec):
            return {}
        if not isinstance(spec, str):
            spec = str(spec)
        spec = spec.strip()
        if not spec:
            return {}
        components = {}
        for part in spec.split(";"):
            piece = part.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise ValueError(f"Invalid combination component '{piece}' (missing weight separator).")
            combo_key, weight_str = piece.rsplit(":", 1)
            combo_key = combo_key.strip()
            if not combo_key:
                raise ValueError(f"Invalid combination component '{piece}' (empty key).")
            try:
                weight = float(weight_str.strip())
            except Exception as exc:
                raise ValueError(f"Invalid weight value '{weight_str}' in component '{piece}': {exc}") from exc
            components[combo_key] = components.get(combo_key, 0.0) + weight
        return components

    def _build_map_from_combination_table(csv_path, base_map, expected_dim):
        if expected_dim is None:
            raise ValueError(
                "Cannot construct perturbation vectors from combination table without a known pert_dim."
            )
        if not base_map:
            raise ValueError(
                "Base perturbation map is empty; cannot expand combinations without reference encodings."
            )
        df = pd.read_csv(csv_path)
        constructed = {}
        missing_components = set()
        zero_combo_genes = []
        for row in df.itertuples(index=False):
            gene_name = str(getattr(row, "gene"))
            combo_spec = getattr(row, "combination", "")
            components = _parse_combination_spec(combo_spec)
            if not components:
                zero_combo_genes.append(gene_name)
                continue
            vector = torch.zeros(expected_dim, dtype=torch.float32)
            for combo_key, weight in components.items():
                reference = base_map.get(combo_key)
                if reference is None:
                    missing_components.add(combo_key)
                    continue
                vector = vector + reference.to(dtype=torch.float32) * float(weight)
            constructed[gene_name] = vector
        if missing_components:
            preview = ", ".join(sorted(missing_components)[:10])
            raise KeyError(
                "Encountered %d combination components that were not present in the base perturbation map. "
                "Examples: %s" % (len(missing_components), preview)
            )
        if zero_combo_genes:
            logger.warning(
                "Skipped %d genes with empty combination specifications when building perturbation map from %s.",
                len(zero_combo_genes),
                csv_path,
            )
        logger.info(
            "Constructed %d custom perturbation vectors from %s.",
            len(constructed),
            csv_path,
        )
        return constructed

    def _serialize_pert_map_for_save(pert_map):
        serialized = {}
        for key, tensor in pert_map.items():
            serialized[key] = tensor.detach().cpu().numpy()
        return serialized

    def _load_numpy_mapping(path):
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            return {k: loaded[k] for k in loaded.files}
        if isinstance(loaded, np.ndarray):
            if loaded.dtype == object:
                if loaded.shape == ():
                    return loaded.item()
                mapping = {}
                for entry in loaded.tolist():
                    if isinstance(entry, (tuple, list)) and len(entry) == 2:
                        mapping[str(entry[0])] = entry[1]
                if mapping:
                    return mapping
            raise ValueError(
                f"Unsupported array format when loading perturbation map from {path}."
            )
        if isinstance(loaded, dict):
            return loaded
        raise TypeError(f"Unsupported data type {type(loaded).__name__} in {path}.")

    def _load_custom_perturbation_map(
        path,
        base_map,
        expected_dim,
    ):
        if path is None:
            return None, None

        resolved_path = path
        data = None
        source_description = None
        extension = os.path.splitext(resolved_path)[1].lower()

        if os.path.exists(resolved_path) and extension in {".npy", ".npz"}:
            data = _load_numpy_mapping(resolved_path)
            source_description = resolved_path
        elif os.path.exists(resolved_path) and extension == ".csv":
            data = _build_map_from_combination_table(resolved_path, base_map, expected_dim)
            source_description = resolved_path
        else:
            candidate_csv = _infer_combination_csv(resolved_path)
            if candidate_csv and os.path.exists(candidate_csv):
                data = _build_map_from_combination_table(candidate_csv, base_map, expected_dim)
                source_description = candidate_csv
                if extension in {".npy", ".npz"} or extension == "":
                    try:
                        serializable = _serialize_pert_map_for_save(data)
                        if not os.path.exists(resolved_path):
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
                                tmp_path = tmp_file.name
                            try:
                                np.save(tmp_path, serializable, allow_pickle=True)
                                os.replace(tmp_path, resolved_path)
                                logger.info("Saved converted perturbation map to %s", resolved_path)
                            finally:
                                if os.path.exists(tmp_path):
                                    try:
                                        os.remove(tmp_path)
                                    except OSError:
                                        pass
                        else:
                            logger.debug("Perturbation map already exists at %s; skipping save", resolved_path)
                    except Exception as exc:
                        logger.warning(
                            "Failed to save converted perturbation map to %s: %s",
                            resolved_path,
                            exc,
                        )
            else:
                raise FileNotFoundError(
                    f"Custom perturbation specification {resolved_path} not found. "
                    "Provide an existing .npy/.npz file or a CSV with combination specifications."
                )

        if isinstance(data, dict):
            normalized = _normalize_pert_map(
                data,
                expected_dim=expected_dim,
                label="custom perturbation map",
            )
        else:
            normalized = data

        return normalized, source_description

    torch.multiprocessing.set_sharing_strategy("file_system")

    config_path = os.path.join(args.output_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    logger.info("Loaded config from %s", config_path)

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

    # Override data paths if --data-config is provided
    # We need to modify the saved state before loading the data module
    if getattr(args, "data_config", None) is not None:
        import tempfile
        
        logger.info("Overriding data paths with config from %s", args.data_config)
        
        data_module_path = os.path.join(run_output_dir, "data_module.torch")
        if not os.path.exists(data_module_path):
            raise FileNotFoundError(f"Could not find data module at {data_module_path}")
        
        # Load the raw state dict
        state_dict = torch.load(data_module_path, weights_only=False)
        
        # Override the toml_config_path in the state dict
        state_dict['toml_config_path'] = os.path.abspath(args.data_config)
        logger.info("Updated toml_config_path to: %s", state_dict['toml_config_path'])
        
        # Save to a temporary location in a writable directory
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.torch', delete=False) as tmp:
            temp_path = tmp.name
        torch.save(state_dict, temp_path)
        logger.info("Saved modified state to temp file: %s", temp_path)
        
        # Load the data module from the modified state
        data_module = PerturbationDataModule.load_state(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
    else:
        data_module_path = os.path.join(run_output_dir, "data_module.torch")
        if not os.path.exists(data_module_path):
            raise FileNotFoundError(f"Could not find data module at {data_module_path}")
        data_module = PerturbationDataModule.load_state(data_module_path)
    
    # Only setup data module if we need to load cell data (not using preassembled core cells)
    custom_core_cells_path = getattr(args, "core_cells_path", None)
    if custom_core_cells_path is None:
        data_module.setup(stage="test")
        logger.info("Loaded data module from %s", data_module_path)
    else:
        logger.info("Loaded data module configuration from %s (skipping data setup for preassembled core cells)", data_module_path)

    pl.seed_everything(cfg["training"]["train_seed"])

    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}. Specify --checkpoint with a valid file."
        )
    logger.info("Loading model from %s", checkpoint_path)

    model_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]
    
    # Get var_dims from checkpoint when using preassembled core cells to avoid loading data
    if custom_core_cells_path is not None:
        logger.info("Reading var_dims from checkpoint to avoid loading data")
        ckpt_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        var_dims = {}
        
        # Extract dimensions from model state dict
        if 'state_dict' in ckpt_state:
            state_dict = ckpt_state['state_dict']
            # Try to infer dimensions from model weights
            for key in state_dict.keys():
                if 'pert_emb' in key or 'perturbation_encoder' in key:
                    if 'weight' in key:
                        var_dims['pert_dim'] = state_dict[key].shape[0] if state_dict[key].dim() > 1 else state_dict[key].shape[0]
                if 'cell_encoder' in key or 'encoder.0' in key:
                    if 'weight' in key and 'input_dim' not in var_dims:
                        var_dims['input_dim'] = state_dict[key].shape[1] if state_dict[key].dim() > 1 else state_dict[key].shape[0]
                if 'decoder' in key or 'cell_decoder' in key:
                    if 'weight' in key and 'output_dim' not in var_dims:
                        # Output dim is usually the last layer of decoder
                        if 'decoder.weight' in key or 'cell_decoder.weight' in key:
                            var_dims['output_dim'] = state_dict[key].shape[0]
        
        # Also check hyper_parameters in checkpoint
        if 'hyper_parameters' in ckpt_state:
            hp = ckpt_state['hyper_parameters']
            for dim_key in ['input_dim', 'output_dim', 'pert_dim', 'gene_dim', 'hvg_dim', 'hidden_dim']:
                if dim_key in hp:
                    var_dims[dim_key] = hp[dim_key]
        
        logger.info("Extracted var_dims from checkpoint: %s", var_dims)
        
        if not var_dims or 'input_dim' not in var_dims or 'output_dim' not in var_dims or 'pert_dim' not in var_dims:
            raise RuntimeError(f"Could not extract required dimensions from checkpoint. Got: {var_dims}")
    else:
        var_dims = data_module.get_var_dims()

    if model_name.lower() == "embedsum":
        from ...tx.models.embed_sum import EmbedSumPerturbationModel

        ModelClass = EmbedSumPerturbationModel
    elif model_name.lower() == "old_neuralot":
        from ...tx.models.old_neural_ot import OldNeuralOTPerturbationModel

        ModelClass = OldNeuralOTPerturbationModel
    elif model_name.lower() in {"neuralot", "pertsets", "state"}:
        from ...tx.models.state_transition import StateTransitionPerturbationModel

        ModelClass = StateTransitionPerturbationModel
    elif model_name.lower() in {"globalsimplesum", "perturb_mean"}:
        from ...tx.models.perturb_mean import PerturbMeanPerturbationModel

        ModelClass = PerturbMeanPerturbationModel
    elif model_name.lower() in {"celltypemean", "context_mean"}:
        from ...tx.models.context_mean import ContextMeanPerturbationModel

        ModelClass = ContextMeanPerturbationModel
    elif model_name.lower() == "decoder_only":
        from ...tx.models.decoder_only import DecoderOnlyPerturbationModel

        ModelClass = DecoderOnlyPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_name}")

    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    for optional_key in ("gene_dim", "hvg_dim", "batch_dim"):
        optional_value = var_dims.get(optional_key)
        if optional_value is not None:
            model_init_kwargs[optional_key] = optional_value

    if "hidden_dim" in var_dims and "hidden_dim" not in model_init_kwargs:
        model_init_kwargs["hidden_dim"] = var_dims["hidden_dim"]

    # Add embed_key from data module if not already present
    if "embed_key" not in model_init_kwargs:
        embed_key = getattr(data_module, "embed_key", None) or "latent_embedding"
        model_init_kwargs["embed_key"] = embed_key
    
    # Add output_space from config if not already present
    if "output_space" not in model_init_kwargs:
        output_space = cfg["data"]["kwargs"].get("output_space", "embedding")
        model_init_kwargs["output_space"] = output_space

    # Load checkpoint and handle dimension mismatches
    checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    
    # Handle pert_encoder dimension mismatch
    pert_encoder_weight_key = "pert_encoder.0.weight"
    if pert_encoder_weight_key in checkpoint_state["state_dict"]:
        checkpoint_pert_dim = checkpoint_state["state_dict"][pert_encoder_weight_key].shape[1]
        current_pert_dim = model_init_kwargs.get("pert_dim", var_dims["pert_dim"])
        
        if checkpoint_pert_dim != current_pert_dim:
            logger.warning(
                "pert_encoder dimension mismatch: checkpoint has %d dims, current model needs %d dims. "
                "Using first %d dimensions from checkpoint and zeroing the rest.",
                checkpoint_pert_dim, current_pert_dim, min(checkpoint_pert_dim, current_pert_dim)
            )
            
            # Get checkpoint weights
            checkpoint_weight = checkpoint_state["state_dict"][pert_encoder_weight_key]
            
            # Create new weight tensor with current model dimensions  
            new_weight = torch.zeros(checkpoint_weight.shape[0], current_pert_dim)
            
            # Copy available dimensions (use min of both dimensions)
            min_dim = min(checkpoint_pert_dim, current_pert_dim)
            new_weight[:, :min_dim] = checkpoint_weight[:, :min_dim]
            
            # Update the checkpoint state dict
            checkpoint_state["state_dict"][pert_encoder_weight_key] = new_weight
            
            logger.info(
                "Updated pert_encoder.0.weight: copied %d/%d input dimensions",
                min_dim, current_pert_dim
            )
            
            # Update the model_init_kwargs to match what's actually in the checkpoint
            # but keep our adjusted pert_dim
            model_init_kwargs["pert_dim"] = current_pert_dim
    
    # Handle batch_encoder dimension mismatch
    batch_encoder_weight_key = "batch_encoder.weight"
    if batch_encoder_weight_key in checkpoint_state["state_dict"]:
        checkpoint_batch_dim = checkpoint_state["state_dict"][batch_encoder_weight_key].shape[0]
        current_batch_dim = model_init_kwargs.get("batch_dim")
        
        if current_batch_dim is not None and checkpoint_batch_dim != current_batch_dim:
            logger.warning(
                "batch_encoder dimension mismatch: checkpoint has %d batch categories, current model needs %d. "
                "Using first %d categories from checkpoint and zeroing the rest.",
                checkpoint_batch_dim, current_batch_dim, min(checkpoint_batch_dim, current_batch_dim)
            )
            
            # Get checkpoint weights
            checkpoint_weight = checkpoint_state["state_dict"][batch_encoder_weight_key]
            
            # Create new weight tensor with current model dimensions
            new_weight = torch.zeros(current_batch_dim, checkpoint_weight.shape[1])
            
            # Copy available dimensions (use min of both dimensions)
            min_dim = min(checkpoint_batch_dim, current_batch_dim)
            new_weight[:min_dim, :] = checkpoint_weight[:min_dim, :]
            
            # Update the checkpoint state dict
            checkpoint_state["state_dict"][batch_encoder_weight_key] = new_weight
            
            logger.info(
                "Updated batch_encoder.weight: copied %d/%d batch categories",
                min_dim, current_batch_dim
            )
        elif current_batch_dim is None:
            # If current model doesn't expect batch_encoder, use checkpoint's batch_dim
            model_init_kwargs["batch_dim"] = checkpoint_batch_dim
            logger.info("Using checkpoint's batch_dim: %d", checkpoint_batch_dim)
    
    # Extract additional parameters from checkpoint hyperparameters if available
    if "hyper_parameters" in checkpoint_state:
        hp = checkpoint_state["hyper_parameters"]
        for param_key in ["batch_dim", "cell_sentence_len", "batch_encoder", "predict_mean"]:
            if param_key in hp and param_key not in model_init_kwargs:
                model_init_kwargs[param_key] = hp[param_key]
                logger.debug("Added %s=%s from checkpoint hyperparameters", param_key, hp[param_key])
    
    # Save the modified checkpoint to a temporary file and load from there
    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.ckpt', delete=False) as tmp:
        temp_checkpoint_path = tmp.name
    torch.save(checkpoint_state, temp_checkpoint_path)
    
    try:
        # Load model using Lightning's checkpoint loading mechanism
        model = ModelClass.load_from_checkpoint(
            temp_checkpoint_path,
            **model_init_kwargs,
        )
        model.eval()
        logger.info("Model loaded successfully.")
    finally:
        # Clean up temporary file
        import os
        os.remove(temp_checkpoint_path)

    results_dir_default = (
        args.results_dir
        if args.results_dir is not None
        else os.path.join(args.output_dir, f"eval_{os.path.basename(args.checkpoint)}")
    )

    data_module.batch_size = 1
    target_celltype = getattr(args, "target_cell_type")
    # custom_core_cells_path was already loaded earlier to skip data setup
    if custom_core_cells_path is not None:
        target_celltype = None

    def _create_filtered_loader(module):
        use_preassembled_core = custom_core_cells_path is not None
        eval_train = bool(getattr(args, "eval_train_data", False)) and not use_preassembled_core

        base_loader = (
            module.train_dataloader(test=True)
            if eval_train
            else module.test_dataloader()
        )

        celltype_key, attempted = _resolve_celltype_key({}, module)

        def _generator():
            found_target = False
            for batch in base_loader:
                if use_preassembled_core:
                    found_target = True
                    yield batch
                    continue

                if target_celltype is None:
                    found_target = True
                    yield batch
                    continue

                key = celltype_key
                if key is None:
                    key, attempted_keys = _resolve_celltype_key(batch, module)
                    if key is None:
                        available_keys = [k for k in batch.keys() if isinstance(k, str)]
                        available_preview = ", ".join(sorted(available_keys)[:10])
                        raise ValueError(
                            "--target-cell-type requested filtering but none of the expected keys (%s) were present."
                            " Available batch keys: %s%s"
                            % (
                                ", ".join(attempted_keys) if attempted_keys else "none",
                                available_preview,
                                "..." if len(available_keys) > 10 else "",
                            )
                        )

                celltypes = _to_list(batch[key])
                mask_values = [str(ct).lower() == target_celltype.lower() for ct in celltypes]
                if not mask_values or not any(mask_values):
                    continue

                mask = torch.tensor(mask_values, dtype=torch.bool)
                filtered = {}
                for batch_key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        mask_device = mask.to(value.device)
                        selected = value[mask_device]
                        if selected.shape[0] == 0:
                            continue
                        filtered[batch_key] = selected
                    else:
                        vals = _to_list(value)
                        selected = [vals[idx] for idx, keep in enumerate(mask_values) if keep]
                        if not selected:
                            continue
                        filtered[batch_key] = selected
                if filtered:
                    found_target = True
                    yield filtered

            if target_celltype and not found_target:
                raise ValueError(
                    f"Target cell type '{target_celltype}' not found in any batches for evaluation."
                )

        return _generator()

    # Only create eval_loader if we're not using preassembled core cells
    if custom_core_cells_path is None:
        eval_loader = _create_filtered_loader(data_module)
    else:
        eval_loader = None

    logger.info("Preparing core cells batch and enumerating perturbations...")

    # Get control_pert - when using preassembled core cells, get it from the data_module attributes
    if custom_core_cells_path is not None:
        control_pert = data_module.control_pert
        logger.info("Using control_pert from data_module config: %s", control_pert)
    else:
        control_pert = data_module.get_control_pert()
    
    custom_perturbation_path = getattr(args, "perturbation_npy", None)

    base_perturbation_map_raw = getattr(data_module, "pert_onehot_map", None)
    if base_perturbation_map_raw is None:
        map_path = os.path.join(run_output_dir, "pert_onehot_map.pt")
        if os.path.exists(map_path):
            base_perturbation_map_raw = torch.load(map_path, weights_only=False)
            logger.info("Loaded base pert_onehot_map from %s", map_path)
        else:
            logger.warning("pert_onehot_map.pt not found at %s", map_path)
            base_perturbation_map_raw = {}

    base_perturbation_map = _normalize_pert_map(
        base_perturbation_map_raw,
        label="base perturbation map",
    )
    if base_perturbation_map:
        logger.info("Base perturbation map contains %d entries.", len(base_perturbation_map))
    else:
        logger.warning("Base perturbation map is empty; custom perturbations may be required.")

    base_expected_dim = None
    if base_perturbation_map:
        base_expected_dim = next(iter(base_perturbation_map.values())).numel()

    expected_pert_dim = base_expected_dim
    var_dims_preview = None
    if custom_perturbation_path or base_expected_dim is None:
        try:
            var_dims_preview = data_module.get_var_dims()
        except Exception as exc:
            logger.debug("Unable to preview var_dims prior to perturbation setup: %s", exc)
            var_dims_preview = None
        else:
            if isinstance(var_dims_preview, dict) and var_dims_preview.get("pert_dim") is not None:
                expected_pert_dim = var_dims_preview["pert_dim"]

    custom_perturbation_map = {}
    custom_map_source = None
    if custom_perturbation_path:
        logger.info("Attempting to load custom perturbation vectors from %s", custom_perturbation_path)
        custom_perturbation_map, custom_map_source = _load_custom_perturbation_map(
            custom_perturbation_path,
            base_perturbation_map,
            expected_pert_dim,
        )
        if not custom_perturbation_map:
            raise ValueError(
                f"No perturbation vectors were loaded from {custom_map_source or custom_perturbation_path}."
            )
        expected_pert_dim = next(iter(custom_perturbation_map.values())).numel()
        if control_pert and control_pert not in custom_perturbation_map:
            base_control = base_perturbation_map.get(control_pert)
            if base_control is not None:
                custom_perturbation_map[control_pert] = base_control.clone()
                logger.info(
                    "Added control perturbation '%s' to custom perturbation map using base encoding.",
                    control_pert,
                )
        logger.info(
            "Using custom perturbation map with %d entries (source: %s).",
            len(custom_perturbation_map),
            custom_map_source or custom_perturbation_path,
        )

    if custom_perturbation_map:
        pert_onehot_map = custom_perturbation_map
        fallback_perturbation_map = base_perturbation_map
    else:
        pert_onehot_map = base_perturbation_map
        fallback_perturbation_map = None

    # Use pert_onehot_map to get all perturbations if available, otherwise enumerate from dataloader
    if pert_onehot_map:
        unique_perts = list(pert_onehot_map.keys())
        logger.info(
            "Enumerating %d perturbations from %s perturbation map",
            len(unique_perts),
            "custom" if custom_perturbation_map else "base",
        )
    else:
        unique_perts = []
        seen_perts = set()
        for batch in eval_loader:
            names = _to_list(batch.get("pert_name", []))
            for name in names:
                name_value = name.item() if isinstance(name, torch.Tensor) else str(name)
                if name_value not in seen_perts:
                    seen_perts.add(name_value)
                    unique_perts.append(name_value)
        if not unique_perts:
            raise RuntimeError("No perturbations found in the provided dataloader.")
        logger.warning(
            "Using perturbations from filtered dataloader (%d found). "
            "This may be limited by --target-cell-type filtering. "
            "Consider supplying a perturbation map for complete perturbation coverage.",
            len(unique_perts),
        )

    target_core_n_default = 64

    def _load_core_cells_from_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Core cells file not found: {path}")

        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("Expected a .npy file containing a serialized dictionary; received an .npz archive.")
        if isinstance(loaded, np.ndarray) and loaded.dtype == object:
            if loaded.shape == ():
                loaded = loaded.item()
        if not isinstance(loaded, dict):
            raise ValueError(
                "Serialized core cells must be a dictionary mapping field names to arrays/tensors;"
                f" received type {type(loaded).__name__}."
            )

        converted = {}
        inferred_length = None
        for key, value in loaded.items():
            if isinstance(value, torch.Tensor):
                tensor = value.clone().detach()
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            else:
                tensor = None

            if tensor is not None:
                if tensor.dim() == 0:
                    converted[key] = tensor.item()
                    continue
                if inferred_length is None:
                    inferred_length = tensor.shape[0]
                else:
                    if tensor.shape[0] != inferred_length:
                        raise ValueError(
                            f"Mismatched leading dimensions in core cells: expected {inferred_length},"
                            f" received {tensor.shape[0]} for key '{key}'."
                        )
                converted[key] = tensor
            else:
                values_list = _to_list(value)
                if inferred_length is None:
                    inferred_length = len(values_list)
                elif len(values_list) != inferred_length:
                    raise ValueError(
                        f"Mismatched list length in core cells: expected {inferred_length},"
                        f" received {len(values_list)} for key '{key}'."
                    )
                converted[key] = values_list

        if inferred_length is None or inferred_length == 0:
            raise ValueError("Loaded core cells did not contain any batched entries.")

        for key, value in list(converted.items()):
            if isinstance(value, torch.Tensor):
                converted[key] = value[:inferred_length].clone()
            else:
                converted[key] = value[:inferred_length]

        return converted, inferred_length

    if custom_core_cells_path:
        core_cells, target_core_n = _load_core_cells_from_path(os.path.abspath(custom_core_cells_path))
        
        # Map latent_embedding to ctrl_cell_emb if needed for model compatibility
        if 'latent_embedding' in core_cells and 'ctrl_cell_emb' not in core_cells:
            core_cells['ctrl_cell_emb'] = core_cells['latent_embedding']
            logger.info("Mapped latent_embedding to ctrl_cell_emb for model compatibility")
        
        # Map plate to batch if needed for model compatibility
        if 'plate' in core_cells and 'batch' not in core_cells:
            plate_data = core_cells['plate']
            # Check if plate is a list of strings (tensor representations)
            if isinstance(plate_data, list) and len(plate_data) > 0:
                if isinstance(plate_data[0], str) and 'tensor' in plate_data[0].lower():
                    # Parse string tensor representations
                    import re
                    parsed_tensors = []
                    for plate_str in plate_data:
                        # Extract the numbers from the string representation
                        numbers = re.findall(r'[-+]?\d*\.?\d+', plate_str)
                        parsed_tensors.append(torch.tensor([float(n) for n in numbers]))
                    core_cells['batch'] = torch.stack(parsed_tensors)
                    logger.info("Parsed %d plate strings to batch tensor with shape %s", len(parsed_tensors), core_cells['batch'].shape)
                elif isinstance(plate_data[0], torch.Tensor):
                    core_cells['batch'] = torch.stack(plate_data)
                    logger.info("Stacked %d plate tensors to batch tensor", len(plate_data))
                else:
                    # Assume it's batch indices
                    core_cells['batch'] = torch.tensor(plate_data)
                    logger.info("Converted plate list to batch tensor")
            elif isinstance(plate_data, torch.Tensor):
                core_cells['batch'] = plate_data
                logger.info("Using plate tensor as batch")
            else:
                logger.warning("Could not convert plate to batch, type: %s", type(plate_data))
        
        # Ensure batch tensor matches model's expected batch_dim
        if 'batch' in core_cells and isinstance(core_cells['batch'], torch.Tensor):
            current_batch_dim = core_cells['batch'].shape[-1] if core_cells['batch'].dim() > 1 else core_cells['batch'].max().item() + 1
            expected_batch_dim = model_init_kwargs.get('batch_dim')
            
            if expected_batch_dim is not None and current_batch_dim != expected_batch_dim:
                logger.warning(
                    "Batch dimension mismatch: core cells have %d batch categories, model expects %d. "
                    "Adjusting batch tensor to match model expectations.",
                    current_batch_dim, expected_batch_dim
                )
                
                if core_cells['batch'].dim() > 1:
                    # One-hot encoded batch tensor
                    if current_batch_dim < expected_batch_dim:
                        # Pad with zeros
                        padding_size = expected_batch_dim - current_batch_dim
                        padding = torch.zeros(core_cells['batch'].shape[0], padding_size)
                        core_cells['batch'] = torch.cat([core_cells['batch'], padding], dim=1)
                        logger.info("Padded batch tensor from %d to %d dimensions", current_batch_dim, expected_batch_dim)
                    elif current_batch_dim > expected_batch_dim:
                        # Truncate
                        core_cells['batch'] = core_cells['batch'][:, :expected_batch_dim]
                        logger.info("Truncated batch tensor from %d to %d dimensions", current_batch_dim, expected_batch_dim)
                else:
                    # Index-based batch tensor - convert to one-hot
                    batch_indices = core_cells['batch'].long()
                    # Ensure indices are within expected range
                    batch_indices = torch.clamp(batch_indices, 0, expected_batch_dim - 1)
                    # Convert to one-hot
                    core_cells['batch'] = torch.zeros(batch_indices.shape[0], expected_batch_dim)
                    core_cells['batch'].scatter_(1, batch_indices.unsqueeze(1), 1)
                    logger.info("Converted batch indices to one-hot encoding with %d categories", expected_batch_dim)
        
        logger.info(
            "Loaded custom core_cells batch with size %d from %s.",
            target_core_n,
            custom_core_cells_path,
        )
    else:
        eval_loader = _create_filtered_loader(data_module)
        target_core_n = target_core_n_default
        accum = {}

        def _append_field(store, key, value):
            if key not in store:
                store[key] = []
            store[key].append(value)

        for batch in eval_loader:
            names = _to_list(batch.get("pert_name", []))
            if not names:
                continue
            mask = torch.tensor([str(item) == str(control_pert) for item in names], dtype=torch.bool)
            if mask.sum().item() == 0:
                continue

            current_count = accum.get("_count", 0)
            take = min(target_core_n - current_count, int(mask.sum().item()))
            if take <= 0:
                break

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    mask_device = mask.to(value.device)
                    selected = value[mask_device][:take].detach().clone()
                    _append_field(accum, key, selected)
                else:
                    vals = _to_list(value)
                    selected_vals = [vals[idx] for idx, keep in enumerate(mask.tolist()) if keep][:take]
                    _append_field(accum, key, selected_vals)

            accum["_count"] = current_count + take
            if accum["_count"] >= target_core_n:
                break

        if accum.get("_count", 0) < target_core_n:
            raise RuntimeError(
                f"Could not assemble {target_core_n} control cells; gathered only {accum.get('_count', 0)}."
            )

        core_cells = {}
        for key, parts in accum.items():
            if key == "_count":
                continue
            if len(parts) == 1:
                val = parts[0]
            else:
                if isinstance(parts[0], torch.Tensor):
                    val = torch.cat(parts, dim=0)
                else:
                    merged = []
                    for part in parts:
                        merged.extend(_to_list(part))
                    val = merged
            core_cells[key] = (
                val[:target_core_n]
                if isinstance(val, torch.Tensor)
                else _to_list(val)[:target_core_n]
            )

        logger.info("Constructed core_cells batch with size %d.", target_core_n)

    os.makedirs(results_dir_default, exist_ok=True)
    baseline_path = os.path.join(results_dir_default, "core_cells_baseline.npy")
    _save_numpy_snapshot(core_cells, baseline_path, "baseline core_cells batch")

    perts_order = list(unique_perts)
    num_perts = len(perts_order)
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims.get("gene_dim", 0)
    hvg_dim = var_dims.get("hvg_dim", 0)

    logger.info("Running first-pass predictions across %d perturbations...", num_perts)

    first_pass_preds = np.empty((num_perts, target_core_n, output_dim), dtype=np.float32)
    first_pass_real = np.empty((num_perts, target_core_n, output_dim), dtype=np.float32)

    embed_key = getattr(data_module, "embed_key", None) or "latent_embedding"
    output_space = cfg["data"]["kwargs"].get("output_space", "embedding")
    store_counts = output_space in {"gene", "all"}

    first_pass_counts = None
    first_pass_counts_pred = None
    if store_counts:
        feature_dim = hvg_dim if output_space == "gene" and hvg_dim else gene_dim
        if feature_dim > 0:
            first_pass_counts = np.empty((num_perts, target_core_n, feature_dim), dtype=np.float32)
            first_pass_counts_pred = np.empty((num_perts, target_core_n, feature_dim), dtype=np.float32)
        else:
            store_counts = False

    metadata = {
        "pert_name": [],
        "celltype_name": [],
        "batch": [],
        "pert_cell_barcode": [],
        "ctrl_cell_barcode": [],
    }

    device = next(model.parameters()).device

    # pert_onehot_map was already loaded earlier for perturbation enumeration
    if not pert_onehot_map:
        logger.warning("No perturbation map available; will use zero embeddings for perturbations")
        pert_onehot_map = {}

    def _prepare_pert_emb(pert_name, length):
        vec = pert_onehot_map.get(pert_name)
        if vec is None and fallback_perturbation_map:
            vec = fallback_perturbation_map.get(pert_name)
            if vec is not None:
                logger.debug("Using fallback perturbation vector for %s from base map.", pert_name)
        if vec is None and control_pert:
            if control_pert in pert_onehot_map:
                vec = pert_onehot_map[control_pert]
            elif fallback_perturbation_map and control_pert in fallback_perturbation_map:
                vec = fallback_perturbation_map[control_pert]
        
        pert_dim = getattr(model, "pert_dim", var_dims.get("pert_dim", 0))
        if pert_dim <= 0:
            raise RuntimeError("pert_dim is undefined; cannot create perturbation embedding")
            
        if vec is None:
            # Create zero vector if perturbation not found
            vec = torch.zeros(pert_dim)
            logger.debug("Created zero perturbation vector for %s (not found in pert_onehot_map)", pert_name)
        else:
            vec = vec.clone().detach().cpu()
            # Handle dimension mismatch between pert_onehot_map and model's pert_dim
            if vec.shape[0] != pert_dim:
                if vec.shape[0] > pert_dim:
                    # Truncate if vector is longer than expected (use first pert_dim dimensions)
                    logger.debug("Truncating perturbation vector for %s from %d to %d", pert_name, vec.shape[0], pert_dim)
                    vec = vec[:pert_dim]
                else:
                    # Pad with zeros if vector is shorter than expected
                    logger.debug("Padding perturbation vector for %s from %d to %d", pert_name, vec.shape[0], pert_dim)
                    padding = torch.zeros(pert_dim - vec.shape[0])
                    vec = torch.cat([vec, padding])
        
        return vec.float().unsqueeze(0).repeat(length, 1).to(device)

    with torch.no_grad():
        for p_idx, pert in enumerate(tqdm(perts_order, desc="First pass", unit="pert")):
            batch = {}
            for key, value in core_cells.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.clone().to(device)
                else:
                    batch[key] = list(value)

            if "pert_name" in batch:
                batch["pert_name"] = [pert for _ in range(target_core_n)]
            if "pert_idx" in batch and hasattr(data_module, "get_pert_index"):
                try:
                    idx_val = int(data_module.get_pert_index(pert))
                    batch["pert_idx"] = torch.tensor([idx_val] * target_core_n, device=device)
                except Exception:
                    pass

            batch["pert_emb"] = _prepare_pert_emb(pert, target_core_n)

            batch_preds = model.predict_step(batch, p_idx, padded=False)

            batch_size = batch_preds["preds"].shape[0]
            metadata["pert_name"].extend(_normalize_field(batch_preds.get("pert_name", pert), batch_size, pert))
            metadata["celltype_name"].extend(
                _normalize_field(
                    batch_preds.get("celltype_name"),
                    batch_size,
                    target_celltype,
                )
            )
            metadata["batch"].extend(
                [None if b is None else str(b) for b in _normalize_field(batch_preds.get("batch"), batch_size)]
            )
            metadata["pert_cell_barcode"].extend(
                _normalize_field(batch_preds.get("pert_cell_barcode"), batch_size)
            )
            metadata["ctrl_cell_barcode"].extend(
                _normalize_field(batch_preds.get("ctrl_cell_barcode"), batch_size)
            )

            batch_pred_np = batch_preds["preds"].detach().cpu().numpy().astype(np.float32)
            
            # When using preassembled core cells, pert_cell_emb may not exist (double perturbation case)
            # Use the input ctrl_cell_emb as the "real" baseline in that case
            if batch_preds.get("pert_cell_emb") is not None:
                batch_real_np = batch_preds["pert_cell_emb"].detach().cpu().numpy().astype(np.float32)
            else:
                # Use ctrl_cell_emb (the input embeddings) as the baseline
                batch_real_np = batch["ctrl_cell_emb"].detach().cpu().numpy().astype(np.float32)
                logger.info("Using ctrl_cell_emb as baseline (pert_cell_emb not available)")

            first_pass_preds[p_idx, :, :] = batch_pred_np
            first_pass_real[p_idx, :, :] = batch_real_np

            if store_counts and first_pass_counts is not None and batch_preds.get("pert_cell_counts") is not None:
                counts_np = batch_preds["pert_cell_counts"].detach().cpu().numpy().astype(np.float32)
                first_pass_counts[p_idx, :, :] = counts_np

            if (
                store_counts
                and first_pass_counts_pred is not None
                and batch_preds.get("pert_cell_counts_preds") is not None
            ):
                counts_pred_np = batch_preds["pert_cell_counts_preds"].detach().cpu().numpy().astype(np.float32)
                first_pass_counts_pred[p_idx, :, :] = counts_pred_np

    logger.info("First pass complete across %d perturbations.", num_perts)

    metadata_df = pd.DataFrame(metadata)
    if metadata_df.empty:
        raise RuntimeError("No metadata collected during first pass; cannot proceed.")

    pert_col = getattr(data_module, "pert_col", None) or "perturbation"
    cell_type_col = getattr(data_module, "cell_type_key", None) or "cell_type"
    batch_col = getattr(data_module, "batch_col", None) or "batch"

    obs_df = pd.DataFrame(
        {
            pert_col: metadata_df["pert_name"],
            cell_type_col: metadata_df["celltype_name"],
            batch_col: metadata_df["batch"],
        }
    )
    if metadata_df["pert_cell_barcode"].notna().any():
        obs_df["pert_cell_barcode"] = metadata_df["pert_cell_barcode"]
    if metadata_df["ctrl_cell_barcode"].notna().any():
        obs_df["ctrl_cell_barcode"] = metadata_df["ctrl_cell_barcode"]

    first_pass_pred_flat = first_pass_preds.reshape(num_perts * target_core_n, output_dim)
    first_pass_real_flat = first_pass_real.reshape(num_perts * target_core_n, output_dim)

    if store_counts and first_pass_counts is not None and first_pass_counts_pred is not None:
        feature_dim = first_pass_counts.shape[-1]
        gene_names = var_dims.get("gene_names")
        if gene_names is not None and len(gene_names) == feature_dim:
            var_index = pd.Index([str(name) for name in gene_names], name="gene")
        else:
            var_index = pd.Index([f"feature_{idx}" for idx in range(feature_dim)], name="feature")
        var_df = pd.DataFrame(index=var_index)

        pred_X = first_pass_counts_pred.reshape(num_perts * target_core_n, feature_dim)
        real_X = first_pass_counts.reshape(num_perts * target_core_n, feature_dim)
    else:
        var_index = pd.Index([f"embedding_{idx}" for idx in range(output_dim)], name="embedding")
        var_df = pd.DataFrame(index=var_index)
        pred_X = first_pass_pred_flat
        real_X = first_pass_real_flat

    first_pass_pred_adata = anndata.AnnData(X=pred_X, obs=obs_df.copy(), var=var_df.copy())
    first_pass_real_adata = anndata.AnnData(X=real_X, obs=obs_df.copy(), var=var_df.copy())
    first_pass_pred_adata.obsm[embed_key] = first_pass_pred_flat
    first_pass_real_adata.obsm[embed_key] = first_pass_real_flat

    first_pass_pred_path = os.path.join(results_dir_default, "first_pass_preds.h5ad")
    first_pass_real_path = os.path.join(results_dir_default, "first_pass_real.h5ad")
    first_pass_pred_adata.write_h5ad(first_pass_pred_path)
    first_pass_real_adata.write_h5ad(first_pass_real_path)
    logger.info("Saved first-pass predicted adata to %s", first_pass_pred_path)
    logger.info("Saved first-pass real adata to %s", first_pass_real_path)

    np.save(os.path.join(results_dir_default, "first_pass_preds.npy"), first_pass_preds)
    np.save(os.path.join(results_dir_default, "first_pass_real.npy"), first_pass_real)
    if first_pass_counts is not None:
        np.save(os.path.join(results_dir_default, "first_pass_counts.npy"), first_pass_counts)
    if first_pass_counts_pred is not None:
        np.save(os.path.join(results_dir_default, "first_pass_counts_pred.npy"), first_pass_counts_pred)

    if phase_one_only:
        logger.info("Phase one complete; skipping metrics as requested.")
        return

    if args.predict_only:
        return

    if cell_type_col not in first_pass_real_adata.obs.columns:
        logger.warning(
            "Cell type column '%s' not found in observations; skipping metric computation.",
            cell_type_col,
        )
        return

    control_pert = data_module.get_control_pert()
    ct_split_real = split_anndata_on_celltype(
        adata=first_pass_real_adata,
        celltype_col=cell_type_col,
    )
    ct_split_pred = split_anndata_on_celltype(
        adata=first_pass_pred_adata,
        celltype_col=cell_type_col,
    )

    if len(ct_split_real) != len(ct_split_pred):
        logger.warning(
            "Number of celltypes in real and predicted AnnData objects differ (%d vs %d); skipping metrics.",
            len(ct_split_real),
            len(ct_split_pred),
        )
        return

    # Check if we have enough perturbations for meaningful evaluation
    total_unique_perts = first_pass_real_adata.obs[pert_col].nunique()
    if total_unique_perts < 2:
        logger.warning(
            "Insufficient perturbations for evaluation (%d found). "
            "Differential expression analysis requires at least 2 perturbations. "
            "This may be due to --target-cell-type filtering or missing pert_onehot_map.",
            total_unique_perts
        )
        return

    pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
    for celltype in ct_split_real.keys():
        real_ct = ct_split_real[celltype]
        pred_ct = ct_split_pred[celltype]

        metric_configs = {}
        if data_module.embed_key and data_module.embed_key != "X_hvg":
                metric_configs = {
                "discrimination_score": {"embed_key": embed_key},
                "pearson_edistance": {"embed_key": embed_key, "n_jobs": -1},
            }
        else:
            metric_configs = {"pearson_edistance": {"n_jobs": -1}}

        evaluator = MetricsEvaluator(
            adata_pred=pred_ct,
            adata_real=real_ct,
            control_pert=control_pert,
            pert_col=pert_col,
            outdir=results_dir_default,
            prefix=str(celltype),
            pdex_kwargs=pdex_kwargs,
            batch_size=2048,
        )
        evaluator.compute(
            profile=args.profile,
            metric_configs=metric_configs,
            skip_metrics=["pearson_edistance", "clustering_agreement"],
        )


def save_core_cells_real_preds(args: ap.ArgumentParser) -> None:
    """Run only phase one of the pipeline and persist real core-cell embeddings per perturbation."""
    return run_tx_single(args, phase_one_only=True)
