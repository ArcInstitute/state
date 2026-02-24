import argparse as ap


def add_arguments_predict(parser: ap.ArgumentParser):
    """
    CLI for evaluation using cell-eval metrics.
    """

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--toml",
        type=str,
        default=None,
        help="Optional path to a TOML data config to use instead of the training config.",
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
        "--skip-adatas",
        action="store_true",
        help="If set, skip writing AnnData (.h5ad) outputs and only run metrics/evaluation.",
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
        "--pseudobulk",
        action="store_true",
        help=(
            "If set, aggregate predictions in a streaming fashion into running pseudobulks by "
            "(context, perturbation) before cell-eval."
        ),
    )


def run_tx_predict(args: ap.ArgumentParser):
    import logging
    import os
    import sys

    import anndata
    import lightning.pytorch as pl
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp
    import torch
    import yaml

    # Cell-eval for metrics computation
    from cell_eval import MetricsEvaluator
    from cell_eval.utils import split_anndata_on_celltype
    from cell_load.data_modules import PerturbationDataModule
    from tqdm import tqdm
    from ._utils import normalize_batch_labels

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.predict_only and args.skip_adatas:
        logger.warning("Both --predict-only and --skip-adatas were set; no prediction artifacts will be written.")

    def run_test_time_finetune(model, dataloader, ft_epochs, control_pert, device):
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

    def clip_anndata_values(adata: anndata.AnnData, max_value: float, min_value: float = 0.0) -> None:
        """Clip adata.X values in-place to keep cell-eval scale checks happy."""
        if sp.issparse(adata.X):
            # Clip only the stored data to keep sparsity intact.
            if adata.X.data.size:
                np.clip(adata.X.data, min_value, max_value, out=adata.X.data)
                if hasattr(adata.X, "eliminate_zeros"):
                    adata.X.eliminate_zeros()
        else:
            np.clip(adata.X, min_value, max_value, out=adata.X)

    def get_batch_labels(candidates, batch_size: int):
        batch_labels = None
        for candidate in candidates:
            batch_labels = normalize_batch_labels(candidate, batch_size)
            if batch_labels is not None:
                break
        if batch_labels is None:
            batch_labels = ["None"] * batch_size
        return batch_labels

    def resolve_context_labels(batch: dict, batch_size: int, fallback):
        dataset_labels = None
        for key in ("dataset_name", "dataset"):
            if key in batch and batch.get(key) is not None:
                dataset_labels = normalize_batch_labels(batch.get(key), batch_size)
                if dataset_labels is not None:
                    break
        context_labels = normalize_batch_labels(fallback, batch_size) if fallback is not None else None

        if dataset_labels is not None and context_labels is not None:
            combined = [f"{ds}.{ct}" for ds, ct in zip(dataset_labels, context_labels)]
            return combined, "dataset_name+cell_type"
        if context_labels is not None:
            return context_labels, "cell_type"
        return None, None

    def ensure_list(values, batch_size: int):
        if isinstance(values, list):
            return values
        if isinstance(values, tuple):
            return list(values)
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(values, np.ndarray):
            if values.ndim == 0:
                return [values.item()] * batch_size
            return values.tolist()
        return [values] * batch_size

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    if args.toml:
        data_section = cfg.get("data")
        if data_section is None or "kwargs" not in data_section:
            raise KeyError("The loaded config does not contain data.kwargs, unable to override toml_config_path.")
        cfg["data"]["kwargs"]["toml_config_path"] = args.toml
        logger.info("Overriding data.kwargs.toml_config_path to %s", args.toml)

    # 2. Find run output directory & load data module
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.torch")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(f"Could not find data module at {data_module_path}?")
    data_module = PerturbationDataModule.load_state(data_module_path)
    if args.toml:
        if not os.path.exists(args.toml):
            raise FileNotFoundError(f"Could not find TOML config file at {args.toml}")
        from cell_load.config import ExperimentConfig

        logger.info("Reloading data module configuration from %s", args.toml)
        data_module.toml_config_path = args.toml
        data_module.config = ExperimentConfig.from_toml(args.toml)
        data_module.config.validate()
        data_module.train_datasets = []
        data_module.val_datasets = []
        data_module.test_datasets = []
        data_module._setup_global_maps()
    data_module.setup(stage="test")
    nb_loss_enabled = bool(cfg.get("model", {}).get("kwargs", {}).get("nb_loss", False))
    output_space = cfg.get("data", {}).get("kwargs", {}).get("output_space", "gene")
    if nb_loss_enabled and output_space == "embedding":
        raise ValueError(
            "model.kwargs.nb_loss=True is incompatible with data.kwargs.output_space='embedding'. "
            "Use output_space='gene' or output_space='all'."
        )
    if nb_loss_enabled and output_space not in {"gene", "all"}:
        raise ValueError(
            f"model.kwargs.nb_loss=True requires data.kwargs.output_space in {{'gene', 'all'}}; got {output_space!r}."
        )
    if nb_loss_enabled:
        resolved_is_log1p = bool(getattr(data_module, "is_log1p", cfg["data"]["kwargs"].get("is_log1p", False)))
        expected_exp_counts = resolved_is_log1p
        current_exp_counts = bool(getattr(data_module, "exp_counts", False))
        if current_exp_counts != expected_exp_counts:
            logger.warning(
                "nb_loss=True requires exp_counts to follow is_log1p. "
                "Resolved is_log1p=%s, overriding exp_counts %s -> %s.",
                resolved_is_log1p,
                current_exp_counts,
                expected_exp_counts,
            )
            data_module.exp_counts = expected_exp_counts
        if data_module.embed_key not in {None, "X_hvg"} and not bool(getattr(data_module, "store_raw_basal", False)):
            logger.warning(
                "nb_loss=True with embed_key=%r and store_raw_basal=False. "
                "NB library-size estimation will fall back to ctrl_cell_emb.",
                data_module.embed_key,
            )
        cfg["data"]["kwargs"]["is_log1p"] = resolved_is_log1p
        cfg["data"]["kwargs"]["exp_counts"] = expected_exp_counts
    resolved_exp_counts = bool(getattr(data_module, "exp_counts", cfg["data"]["kwargs"].get("exp_counts", False)))
    metrics_is_log1p = not (nb_loss_enabled or resolved_exp_counts)
    logger.info(
        "Metrics config: setting pdex is_log1p=%s (nb_loss=%s, exp_counts=%s)",
        metrics_is_log1p,
        nb_loss_enabled,
        resolved_exp_counts,
    )
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
    elif model_class_name.lower() == "pseudobulk":
        from ...tx.models.pseudobulk import PseudobulkPerturbationModel

        ModelClass = PseudobulkPerturbationModel
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

    model = ModelClass.load_from_checkpoint(checkpoint_path, weights_only=False, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # 4. Test-time fine-tuning if requested
    if args.test_time_finetune > 0:
        data_module.batch_size = 1
    if args.test_time_finetune > 0:
        control_pert = data_module.get_control_pert()
        if args.eval_train_data:
            test_loader = data_module.train_dataloader(test=True)
        else:
            test_loader = data_module.test_dataloader()

        run_test_time_finetune(
            model, test_loader, args.test_time_finetune, control_pert, device=next(model.parameters()).device
        )
        logger.info("Test-time fine-tuning complete.")

    # 5. Run inference on test set
    data_module.setup(stage="test")
    if args.eval_train_data:
        test_loader = data_module.train_dataloader(test=True)
    else:
        test_loader = data_module.test_dataloader()

    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    num_cells = test_loader.batch_sampler.tot_num
    output_dim = var_dims["output_dim"]
    gene_dim = var_dims["gene_dim"]
    hvg_dim = var_dims["hvg_dim"]

    logger.info("Generating predictions on test set using manual loop...")
    device = next(model.parameters()).device

    cfg_batch_col = cfg.get("data", {}).get("kwargs", {}).get("batch_col", None)
    batch_obs_key = cfg_batch_col or data_module.batch_col
    if batch_obs_key is None:
        batch_obs_key = "batch"

    store_raw_expression = (
        data_module.embed_key is not None
        and data_module.embed_key != "X_hvg"
        and cfg["data"]["kwargs"]["output_space"] == "gene"
    ) or (data_module.embed_key is not None and cfg["data"]["kwargs"]["output_space"] == "all")
    use_count_outputs = store_raw_expression or nb_loss_enabled
    if nb_loss_enabled and not store_raw_expression:
        logger.info(
            "nb_loss=True: forcing prediction artifacts to use NB count outputs even though "
            "store_raw_expression would otherwise be disabled."
        )

    if args.pseudobulk:
        logger.info("Pseudobulk enabled; aggregating running means by (context, perturbation).")

        if args.eval_train_data:
            results_dir = os.path.join(args.output_dir, "eval_train_" + os.path.basename(args.checkpoint))
        else:
            results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
        os.makedirs(results_dir, exist_ok=True)

        pseudo_x_dim = None
        if use_count_outputs:
            if output_space == "gene":
                pseudo_x_dim = hvg_dim
            elif output_space == "all":
                pseudo_x_dim = gene_dim
            else:
                raise ValueError(f"Unsupported output_space for pseudobulk: {output_space}")

        pb_groups: dict[tuple[str, str], dict] = {}
        context_mode = None
        total_cells_seen = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                batch_preds = model.predict_step(batch, batch_idx, padded=False)

                batch_size = batch_preds["preds"].shape[0]
                total_cells_seen += batch_size

                batch_labels = get_batch_labels(
                    (
                        batch.get("batch_name"),
                        batch_preds.get("batch_name"),
                        batch_preds.get("batch"),
                    ),
                    batch_size,
                )

                pert_names = [str(x) for x in ensure_list(batch_preds.get("pert_name"), batch_size)]
                celltypes = [str(x) for x in ensure_list(batch_preds.get("celltype_name"), batch_size)]
                if len(pert_names) != batch_size or len(celltypes) != batch_size:
                    raise ValueError("Mismatch between batch size and pert/celltype metadata lengths.")

                context_labels, detected_mode = resolve_context_labels(batch, batch_size, celltypes)
                if context_labels is None:
                    raise ValueError(
                        "pseudobulk requires dataset_name + cell_type (preferred) or cell_type alone. "
                        f"Check data.kwargs.cell_type_key (current: {data_module.cell_type_key})."
                    )
                context_labels = [str(x) for x in context_labels]
                if context_mode is None:
                    context_mode = detected_mode
                    logger.info("Pseudobulk context source: %s", context_mode)
                elif detected_mode != context_mode:
                    raise ValueError(
                        f"Inconsistent context source during prediction: saw '{detected_mode}' after '{context_mode}'."
                    )

                batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
                batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)

                batch_real_gene_np = None
                batch_gene_pred_np = None
                if use_count_outputs:
                    batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)

                group_to_indices: dict[tuple[str, str], list[int]] = {}
                for idx in range(batch_size):
                    key = (context_labels[idx], pert_names[idx])
                    group_to_indices.setdefault(key, []).append(idx)

                for (context_label, pert_name), idxs in group_to_indices.items():
                    idx_arr = np.asarray(idxs, dtype=np.int64)
                    first_idx = int(idx_arr[0])
                    current_celltype = celltypes[first_idx]
                    current_batch = str(batch_labels[first_idx])

                    entry = pb_groups.get((context_label, pert_name))
                    if entry is None:
                        entry = {
                            "context": context_label,
                            "pert_name": pert_name,
                            "celltype_name": current_celltype,
                            "batch_name": current_batch,
                            "count": 0,
                            "pred_sum": np.zeros(output_dim, dtype=np.float64),
                            "real_sum": np.zeros(output_dim, dtype=np.float64),
                            "x_hvg_sum": np.zeros(pseudo_x_dim, dtype=np.float64) if use_count_outputs else None,
                            "counts_pred_sum": np.zeros(pseudo_x_dim, dtype=np.float64) if use_count_outputs else None,
                        }
                        pb_groups[(context_label, pert_name)] = entry
                    elif entry["celltype_name"] != current_celltype:
                        raise ValueError(
                            f"Inconsistent cell type for context/pert pair ({context_label}, {pert_name}): "
                            f"saw '{current_celltype}' after '{entry['celltype_name']}'."
                        )

                    entry["count"] += int(idx_arr.size)
                    entry["pred_sum"] += batch_pred_np[idx_arr].sum(axis=0, dtype=np.float64)
                    entry["real_sum"] += batch_real_np[idx_arr].sum(axis=0, dtype=np.float64)
                    if use_count_outputs:
                        entry["x_hvg_sum"] += batch_real_gene_np[idx_arr].sum(axis=0, dtype=np.float64)
                        entry["counts_pred_sum"] += batch_gene_pred_np[idx_arr].sum(axis=0, dtype=np.float64)

        if len(pb_groups) == 0:
            logger.warning("No pseudobulk groups were generated. Exiting.")
            sys.exit(0)

        logger.info(
            "Built %d pseudobulk groups from %d cells.",
            len(pb_groups),
            total_cells_seen,
        )

        group_entries = list(pb_groups.values())
        group_entries.sort(key=lambda x: (str(x["celltype_name"]), str(x["context"]), str(x["pert_name"])))
        n_groups = len(group_entries)

        pred_bulk = np.empty((n_groups, output_dim), dtype=np.float32)
        real_bulk = np.empty((n_groups, output_dim), dtype=np.float32)
        pred_x = np.empty((n_groups, pseudo_x_dim), dtype=np.float32) if use_count_outputs else None
        real_x = np.empty((n_groups, pseudo_x_dim), dtype=np.float32) if use_count_outputs else None

        reserved_obs_keys = {
            str(data_module.pert_col),
            str(data_module.cell_type_key),
            str(batch_obs_key),
        }
        if data_module.batch_col:
            reserved_obs_keys.add(str(data_module.batch_col))

        pseudobulk_context_key = "pseudobulk_context"
        while pseudobulk_context_key in reserved_obs_keys:
            pseudobulk_context_key = f"_{pseudobulk_context_key}"
        pseudobulk_n_cells_key = "pseudobulk_n_cells"
        while pseudobulk_n_cells_key in reserved_obs_keys or pseudobulk_n_cells_key == pseudobulk_context_key:
            pseudobulk_n_cells_key = f"_{pseudobulk_n_cells_key}"

        obs_dict = {
            data_module.pert_col: [],
            data_module.cell_type_key: [],
            batch_obs_key: [],
            pseudobulk_context_key: [],
            pseudobulk_n_cells_key: [],
        }
        if data_module.batch_col and data_module.batch_col != batch_obs_key:
            obs_dict[data_module.batch_col] = []

        for idx, entry in enumerate(group_entries):
            count = int(entry["count"])
            denom = float(count)
            pred_bulk[idx, :] = (entry["pred_sum"] / denom).astype(np.float32, copy=False)
            real_bulk[idx, :] = (entry["real_sum"] / denom).astype(np.float32, copy=False)
            if use_count_outputs:
                pred_x[idx, :] = (entry["counts_pred_sum"] / denom).astype(np.float32, copy=False)
                real_x[idx, :] = (entry["x_hvg_sum"] / denom).astype(np.float32, copy=False)

            obs_dict[data_module.pert_col].append(entry["pert_name"])
            obs_dict[data_module.cell_type_key].append(entry["celltype_name"])
            obs_dict[batch_obs_key].append(entry["batch_name"])
            obs_dict[pseudobulk_context_key].append(entry["context"])
            obs_dict[pseudobulk_n_cells_key].append(count)
            if data_module.batch_col and data_module.batch_col != batch_obs_key:
                obs_dict[data_module.batch_col].append(entry["batch_name"])

        obs = pd.DataFrame(obs_dict)
        if use_count_outputs:
            adata_pred = anndata.AnnData(X=pred_x, obs=obs)
            adata_real = anndata.AnnData(X=real_x, obs=obs)
            if data_module.embed_key is not None:
                adata_pred.obsm[data_module.embed_key] = pred_bulk
                adata_real.obsm[data_module.embed_key] = real_bulk
        else:
            adata_pred = anndata.AnnData(X=pred_bulk, obs=obs)
            adata_real = anndata.AnnData(X=real_bulk, obs=obs)

        if nb_loss_enabled:
            logger.info("nb_loss=True in run config; skipping pseudobulk clipping of adata_pred/adata_real X values.")
        else:
            clip_anndata_values(adata_pred, max_value=14.0)
            clip_anndata_values(adata_real, max_value=14.0)
            logger.info("Clipped pseudobulk adata_pred and adata_real X values to [0.0, 14.0].")

        if args.shared_only:
            try:
                shared_perts = data_module.get_shared_perturbations()
                if len(shared_perts) == 0:
                    logger.warning("No shared perturbations between train and test; skipping filtering.")
                else:
                    logger.info(
                        "Filtering pseudobulk rows to %d shared perturbations present in train ∩ test.",
                        len(shared_perts),
                    )
                    mask = adata_pred.obs[data_module.pert_col].isin(shared_perts)
                    before_n = adata_pred.n_obs
                    adata_pred = adata_pred[mask].copy()
                    adata_real = adata_real[mask].copy()
                    logger.info(
                        "Filtered pseudobulk rows: %d -> %d (kept only seen perturbations)",
                        before_n,
                        adata_pred.n_obs,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to filter by shared perturbations (%s). Proceeding without filter.",
                    str(e),
                )

        adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
        adata_real_path = os.path.join(results_dir, "adata_real.h5ad")
        if args.skip_adatas:
            logger.info("Skipping AnnData writes (--skip-adatas).")
        else:
            adata_pred.write_h5ad(adata_pred_path)
            adata_real.write_h5ad(adata_real_path)
            logger.info(f"Saved adata_pred to {adata_pred_path}")
            logger.info(f"Saved adata_real to {adata_real_path}")

        if not args.predict_only:
            logger.info("Computing metrics using cell-eval...")
            control_pert = data_module.get_control_pert()
            ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
            ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

            assert len(ct_split_real) == len(ct_split_pred), (
                f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
            )

            pdex_kwargs = dict(exp_post_agg=True, is_log1p=metrics_is_log1p)
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
        return

    final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
    final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

    final_X_hvg = None
    final_pert_cell_counts_preds = None
    if use_count_outputs:
        # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
        if output_space == "gene":
            final_X_hvg = np.empty((num_cells, hvg_dim), dtype=np.float32)
            final_pert_cell_counts_preds = np.empty((num_cells, hvg_dim), dtype=np.float32)
        if output_space == "all":
            final_X_hvg = np.empty((num_cells, gene_dim), dtype=np.float32)
            final_pert_cell_counts_preds = np.empty((num_cells, gene_dim), dtype=np.float32)

    current_idx = 0

    # Initialize aggregation variables directly
    all_pert_names = []
    all_celltypes = []
    all_gem_groups = []
    all_pert_barcodes = []
    all_ctrl_barcodes = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
            # Move each tensor in the batch to the model's device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Get predictions
            batch_preds = model.predict_step(batch, batch_idx, padded=False)

            # Extract metadata and data directly from batch_preds
            # Handle pert_name
            if isinstance(batch_preds["pert_name"], list):
                all_pert_names.extend(batch_preds["pert_name"])
            else:
                all_pert_names.append(batch_preds["pert_name"])

            if "pert_cell_barcode" in batch_preds:
                if isinstance(batch_preds["pert_cell_barcode"], list):
                    all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.extend(batch_preds["ctrl_cell_barcode"])
                else:
                    all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                    all_ctrl_barcodes.append(batch_preds["ctrl_cell_barcode"])

            # Handle celltype_name
            if isinstance(batch_preds["celltype_name"], list):
                all_celltypes.extend(batch_preds["celltype_name"])
            else:
                all_celltypes.append(batch_preds["celltype_name"])

            batch_size = batch_preds["preds"].shape[0]

            # Handle gem_group - prefer human-readable batch names when available
            batch_labels = get_batch_labels(
                (
                    batch.get("batch_name"),
                    batch_preds.get("batch_name"),
                    batch_preds.get("batch"),
                ),
                batch_size,
            )
            all_gem_groups.extend(batch_labels)

            batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
            batch_real_np = batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
            final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
            final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
            current_idx += batch_size

            # Handle X_hvg for HVG space ground truth
            if final_X_hvg is not None:
                batch_real_gene_np = batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                final_X_hvg[current_idx - batch_size : current_idx, :] = batch_real_gene_np

            # Handle decoded gene predictions if available
            if final_pert_cell_counts_preds is not None:
                batch_gene_pred_np = batch_preds["pert_cell_counts_preds"].cpu().numpy().astype(np.float32)
                final_pert_cell_counts_preds[current_idx - batch_size : current_idx, :] = batch_gene_pred_np

    logger.info("Creating anndatas from predictions from manual loop...")

    # Build pandas DataFrame for obs and var
    print(batch_obs_key)
    df_dict = {
        data_module.pert_col: all_pert_names,
        data_module.cell_type_key: all_celltypes,
        batch_obs_key: all_gem_groups,
    }
    if data_module.batch_col and data_module.batch_col != batch_obs_key:
        print("\t\t STORING BATCH")
        df_dict[data_module.batch_col] = all_gem_groups

    if len(all_pert_barcodes) > 0:
        df_dict["pert_cell_barcode"] = all_pert_barcodes
        df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

    obs = pd.DataFrame(df_dict)

    gene_names = var_dims["gene_names"]
    var = pd.DataFrame({"gene_names": gene_names})

    if final_X_hvg is not None:
        # if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
        #     gene_names = np.load(
        #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
        #     )
        #     var = pd.DataFrame({"gene_names": gene_names})

        # Create adata for predictions - using the decoded gene expression values
        adata_pred = anndata.AnnData(X=final_pert_cell_counts_preds, obs=obs)
        # Create adata for real - using the true gene expression values
        adata_real = anndata.AnnData(X=final_X_hvg, obs=obs)

        # add the embedding predictions
        if data_module.embed_key is not None:
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

    # Clip extreme values to keep cell-eval log1p checks happy unless NB loss is active.
    if nb_loss_enabled:
        logger.info("nb_loss=True in run config; skipping clipping of adata_pred/adata_real X values.")
    else:
        clip_anndata_values(adata_pred, max_value=14.0)
        clip_anndata_values(adata_real, max_value=14.0)
        logger.info("Clipped adata_pred and adata_real X values to [0.0, 14.0] before evaluation.")

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
    if args.eval_train_data:
        results_dir = os.path.join(args.output_dir, "eval_train_" + os.path.basename(args.checkpoint))
    else:
        results_dir = os.path.join(args.output_dir, "eval_" + os.path.basename(args.checkpoint))
    os.makedirs(results_dir, exist_ok=True)
    adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
    adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

    if args.skip_adatas:
        logger.info("Skipping AnnData writes (--skip-adatas).")
    else:
        adata_pred.write_h5ad(adata_pred_path)
        adata_real.write_h5ad(adata_real_path)

        logger.info(f"Saved adata_pred to {adata_pred_path}")
        logger.info(f"Saved adata_real to {adata_real_path}")

    if not args.predict_only:
        # 6. Compute metrics using cell-eval
        logger.info("Computing metrics using cell-eval...")

        control_pert = data_module.get_control_pert()

        ct_split_real = split_anndata_on_celltype(adata=adata_real, celltype_col=data_module.cell_type_key)
        ct_split_pred = split_anndata_on_celltype(adata=adata_pred, celltype_col=data_module.cell_type_key)

        assert len(ct_split_real) == len(ct_split_pred), (
            f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"
        )

        pdex_kwargs = dict(exp_post_agg=True, is_log1p=metrics_is_log1p)

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
