#!/usr/bin/env python
"""Generate per-max-drug heatmaps summarizing Tahoe combination experiments.

The script expects the directory layout produced by `single_tahoe_combination_average.sh`:

    /data/new_heatmaps/comb_tahoe/
        <CELL_NAME>/
            max_drugs_<N>/
                first_pass_preds.npy
                core_cells_baseline.npy
                first_pass_preds.h5ad

For every max-drug slice it aggregates scalar summaries of the predicted embeddings
and their baseline-subtracted perturbation effects, assembling one heatmap per slice.
It also emits a companion heatmap that visualises only the perturbation effect
(`predicted - baseline`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.cluster import hierarchy

try:
    import anndata as ad  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The 'anndata' package is required to inspect perturbation metadata. "
        "Install it or activate the suitable environment before running this script."
    ) from exc


MAX_DRUGS_DEFAULT = (1, 2, 4, 8, 16, 32)
PERT_NAME_COLUMNS = (
    "pert_name",
    "pert",
    "perturbation",
    "drugname_drugconc",
    "drug_name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/new_heatmaps/comb_tahoe"),
        help="Root directory containing per-cell subdirectories (default: /data/new_heatmaps/comb_tahoe).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures. Defaults to <input-root>/plots.",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        nargs="*",
        default=MAX_DRUGS_DEFAULT,
        help="Max-drug bins to plot (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-pert-order",
        type=Path,
        default=None,
        help=(
            "Optional JSON file to cache the perturbation ordering. "
            "Speeds up repeated runs by avoiding re-reading .h5ad files."
        ),
    )
    parser.add_argument(
        "--limit-perts",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of perturbations to display (top-N by effect magnitude across cells). "
            "Leave unset to plot all perturbations."
        ),
    )
    parser.add_argument(
        "--fig-dpi",
        type=int,
        default=200,
        help="Output DPI for saved figures (default: 200).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for raw magnitude heatmaps (default: viridis).",
    )
    parser.add_argument(
        "--diff-cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap for baseline-subtracted heatmaps (default: coolwarm).",
    )
    parser.add_argument(
        "--column-normalize",
        action="store_true",
        help=(
            "Additionally render column-normalized perturbation-effect heatmaps "
            "(each cell type scaled independently to [0, 1])."
        ),
    )
    parser.add_argument(
        "--cluster-summary-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write CSV summaries of early dendrogram splits per heatmap. "
            "If omitted, no summaries are produced."
        ),
    )
    parser.add_argument(
        "--cluster-summary-depth",
        type=int,
        default=2,
        help=(
            "Maximum depth (levels from root) of dendrogram splits to summarise when writing "
            "cluster information CSVs (default: 2)."
        ),
    )
    args = parser.parse_args()
    if args.limit_perts is not None and args.limit_perts <= 0:
        parser.error("--limit-perts must be a positive integer when specified.")
    return args


def discover_cell_dirs(root: Path, max_drugs: Sequence[int]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input root {root} does not exist.")
    cell_dirs: List[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        found = False
        for bin_value in max_drugs:
            candidate = path / f"max_drugs_{bin_value}"
            if candidate.is_dir() and (candidate / "first_pass_preds.npy").exists():
                found = True
                break
        if found:
            cell_dirs.append(path)
    return cell_dirs


def load_perturbation_order(
    cell_dirs: Sequence[Path],
    max_drugs: Sequence[int],
    cache_file: Optional[Path] = None,
) -> List[str]:
    if cache_file and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if isinstance(cached, list) and all(isinstance(item, str) for item in cached):
            return cached  # type: ignore[return-value]

    candidates = []
    for cell_dir in cell_dirs:
        for max_drug in max_drugs:
            h5_path = cell_dir / f"max_drugs_{max_drug}" / "first_pass_preds.h5ad"
            if h5_path.exists():
                candidates.append(h5_path)
        if candidates:
            break

    if not candidates:
        raise RuntimeError(
            "Could not locate any 'first_pass_preds.h5ad' files. "
            "Ensure the combination experiments have completed."
        )

    # Use the first available file to derive perturbation ordering
    pert_names: List[str] = []
    for h5_file in tqdm(candidates, desc="Scanning AnnData files", unit="file"):
        adata = ad.read_h5ad(h5_file, backed="r")
        available_cols = list(adata.obs.columns)
        target_col = None
        for col in PERT_NAME_COLUMNS:
            if col in adata.obs:
                target_col = col
                break
        if target_col is None and available_cols:
            # Heuristic fallback: use the first column with string-like data
            for col in available_cols:
                series = adata.obs[col]
                sample = series.iloc[0] if len(series) else None
                if isinstance(sample, (str, bytes)):
                    target_col = col
                    break

        if target_col is None:
            adata.file.close()
            continue

        obs_names = pd.Index(adata.obs[target_col].astype(str))
        pert_names = obs_names.drop_duplicates().tolist()
        adata.file.close()
        if pert_names:
            break

    if not pert_names:
        raise RuntimeError(
            "Unable to extract perturbation names from available AnnData files. "
            "Checked columns: %s" % ", ".join(PERT_NAME_COLUMNS)
        )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(pert_names, handle)

    return pert_names


def compute_metrics_for_cell(
    cell_dir: Path,
    max_drug: int,
    baseline_key: str = "ctrl_cell_emb",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (raw_norm, baseline_sub_norm) arrays for every perturbation."""
    base_path = cell_dir / f"max_drugs_{max_drug}"
    preds_path = base_path / "first_pass_preds.npy"
    baseline_path = base_path / "core_cells_baseline.npy"

    if not preds_path.exists() or not baseline_path.exists():
        return None

    preds = np.load(preds_path, mmap_mode="r")
    # baseline file stores dictionary
    baseline_payload = np.load(baseline_path, allow_pickle=True).item()
    if baseline_key not in baseline_payload:
        raise KeyError(
            f"Baseline dictionary at {baseline_path} does not contain key '{baseline_key}'. "
            f"Available keys: {list(baseline_payload.keys())}"
        )
    baseline = np.asarray(baseline_payload[baseline_key], dtype=np.float32)
    if baseline.ndim != 2:
        raise ValueError(
            f"Expected baseline tensor to be 2D (<cells>, <features>); got shape {baseline.shape}."
        )

    num_perts = preds.shape[0]
    raw_norm = np.empty(num_perts, dtype=np.float32)
    effect_norm = np.empty(num_perts, dtype=np.float32)

    # Precompute baseline norms once
    for idx in range(num_perts):
        pert_block = np.asarray(preds[idx], dtype=np.float32)
        if pert_block.shape != baseline.shape:
            raise ValueError(
                f"Perturbation block shape {pert_block.shape} does not match baseline shape {baseline.shape} "
                f"for {cell_dir.name} max_drugs_{max_drug} index {idx}."
            )
        raw_norm[idx] = np.linalg.norm(pert_block, axis=1).mean()
        diff = pert_block - baseline
        effect_norm[idx] = np.linalg.norm(diff, axis=1).mean()

    return raw_norm, effect_norm


def column_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1] using per-column min/max (ignoring NaNs)."""
    normalized = matrix.copy()
    with np.errstate(invalid="ignore"):
        col_min = np.nanmin(normalized, axis=0)
        col_max = np.nanmax(normalized, axis=0)
        denom = col_max - col_min
        valid = denom > 1e-12
        if np.any(valid):
            normalized[:, valid] = (normalized[:, valid] - col_min[valid]) / denom[valid]
        if np.any(~valid):
            normalized[:, ~valid] = 0.0
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def assemble_heatmap_data(
    cell_dirs: Sequence[Path],
    max_drugs: Sequence[int],
    pert_order: Sequence[str],
    normalize_columns: bool = False,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Return nested dict: {max_drug: {'raw': matrix, 'effect': matrix}}."""
    num_perts = len(pert_order)
    cell_names = [cell_dir.name for cell_dir in cell_dirs]
    name_to_index = {name: idx for idx, name in enumerate(cell_names)}

    heatmaps: Dict[int, Dict[str, np.ndarray]] = {}
    for max_drug in tqdm(max_drugs, desc="Aggregating heatmaps", unit="bin"):
        raw_matrix = np.full((num_perts, len(cell_dirs)), np.nan, dtype=np.float32)
        effect_matrix = np.full_like(raw_matrix, np.nan)

        for cell_dir in tqdm(
            cell_dirs,
            desc=f"Cells for max_drugs={max_drug}",
            unit="cell",
            leave=False,
        ):
            metrics = compute_metrics_for_cell(cell_dir, max_drug)
            if metrics is None:
                continue
            raw_norm, effect_norm = metrics
            if raw_norm.shape[0] != num_perts:
                raise ValueError(
                    f"Perturbation count mismatch for {cell_dir.name} max_drugs_{max_drug}: "
                    f"{raw_norm.shape[0]} vs expected {num_perts}."
                )
            col_idx = name_to_index[cell_dir.name]
            raw_matrix[:, col_idx] = raw_norm
            effect_matrix[:, col_idx] = effect_norm

        payload: Dict[str, np.ndarray] = {"raw": raw_matrix, "effect": effect_matrix}
        if normalize_columns:
            payload["effect_colnorm"] = column_normalize(effect_matrix)
        heatmaps[max_drug] = payload

    return heatmaps


def maybe_limit_perturbations(
    matrices: Mapping[int, Mapping[str, np.ndarray]],
    pert_names: Sequence[str],
    limit: Optional[int],
) -> Tuple[Sequence[str], Dict[int, Dict[str, np.ndarray]]]:
    if limit is None or limit >= len(pert_names):
        return pert_names, matrices  # type: ignore[return-value]

    # Rank perturbations by global effect magnitude (mean across cells and max_drug bins)
    combined_effects = []
    for max_drug, payload in matrices.items():
        effect_matrix = payload["effect"]
        combined_effects.append(effect_matrix)
    stacked = np.stack(combined_effects, axis=0)  # shape (num_bins, num_perts, num_cells)
    global_scores = np.nanmean(stacked, axis=(0, 2))
    top_indices = np.argsort(global_scores)[::-1][:limit]
    top_indices = np.sort(top_indices)

    reduced_names = [pert_names[idx] for idx in top_indices]
    reduced: Dict[int, Dict[str, np.ndarray]] = {}
    for max_drug, payload in matrices.items():
        reduced[max_drug] = {
            key: value[top_indices, :]
            for key, value in payload.items()
        }

    return reduced_names, reduced


def _collect_leaf_indices(node: hierarchy.ClusterNode) -> List[int]:
    """Return sorted leaf indices beneath a SciPy cluster node."""
    if node.is_leaf():
        return [node.id]
    leaves: List[int] = []
    if node.left is not None:
        leaves.extend(_collect_leaf_indices(node.left))
    if node.right is not None:
        leaves.extend(_collect_leaf_indices(node.right))
    return sorted(leaves)


def _summarize_branch(
    node: hierarchy.ClusterNode,
    labels: Sequence[str],
    data: np.ndarray,
    axis: str,
    split_depth: int,
) -> Dict[str, object]:
    indices = _collect_leaf_indices(node)
    members = [labels[idx] for idx in indices]
    if axis == "perturbation":
        subset = data[np.ix_(indices, range(data.shape[1]))]
        column_means = np.nanmean(subset, axis=0)
        mean_range = float(np.nanmax(column_means) - np.nanmin(column_means))
        axis_range = mean_range
        axis_min = float(np.nanmin(column_means)) if column_means.size else np.nan
        axis_max = float(np.nanmax(column_means)) if column_means.size else np.nan
    else:
        subset = data[np.ix_(range(data.shape[0]), indices)]
        row_means = np.nanmean(subset, axis=1)
        mean_range = float(np.nanmax(row_means) - np.nanmin(row_means))
        axis_range = mean_range
        axis_min = float(np.nanmin(row_means)) if row_means.size else np.nan
        axis_max = float(np.nanmax(row_means)) if row_means.size else np.nan

    summary: Dict[str, object] = {
        "axis": axis,
        "split_depth": split_depth,
        "node_height": float(node.dist),
        "member_count": len(indices),
        "mean_effect": float(np.nanmean(subset)),
        "effect_std": float(np.nanstd(subset)),
        "axis_mean_min": axis_min,
        "axis_mean_max": axis_max,
        "axis_mean_range": axis_range,
        "members": ";".join(members),
    }
    return summary


def summarize_dendrogram(
    linkage: np.ndarray,
    labels: Sequence[str],
    data: np.ndarray,
    axis: str,
    max_depth: int,
) -> List[Dict[str, object]]:
    """Extract summaries for early dendrogram splits up to `max_depth` levels."""
    if linkage.size == 0 or len(labels) <= 1:
        return []

    root = hierarchy.to_tree(linkage, rd=False)
    summaries: List[Dict[str, object]] = []
    queue: List[Tuple[hierarchy.ClusterNode, int]] = [(root, 0)]

    while queue:
        node, depth = queue.pop(0)
        if node.is_leaf() or depth >= max_depth:
            continue
        # Record both child branches for this split
        for child in (node.left, node.right):
            if child is None:
                continue
            summaries.append(
                _summarize_branch(child, labels, data, axis=axis, split_depth=depth + 1)
            )
            if not child.is_leaf():
                queue.append((child, depth + 1))

    # Order summaries by descending split depth then node height (largest first)
    summaries.sort(key=lambda item: (-item["split_depth"], -item["node_height"]))
    return summaries


def write_cluster_summary(
    records: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df["axis_rank"] = df["axis"].map({"cell": 0, "perturbation": 1}).fillna(99)
    df.sort_values(
        by=["axis_rank", "axis", "split_depth", "node_height"],
        ascending=[True, True, False, False],
        inplace=True,
    )
    df.drop(columns=["axis_rank"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_heatmap(
    data: np.ndarray,
    pert_names: Sequence[str],
    cell_names: Sequence[str],
    title: str,
    cmap: str,
    output_path: Path,
    dpi: int = 200,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    summary_depth: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, object]]], List[str], List[str]]:
    """Render and save a single heatmap. Returns (summaries, reordered_pert_names, reordered_cell_names)."""
    df = pd.DataFrame(data, index=pert_names, columns=cell_names)
    height = max(6.0, min(0.02 * len(pert_names), 30.0))
    width = max(8.0, min(0.35 * len(cell_names), 40.0))
    cluster_grid = sns.clustermap(
        df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=(width, height),
        cbar_kws={"label": "Mean L2 norm"},
    )

    if len(cell_names) > 1:
        reordered = cluster_grid.dendrogram_col.reordered_ind
        col_means = df.iloc[:, reordered].mean(axis=0, skipna=True)
        if col_means.iloc[0] > col_means.iloc[-1]:
            cluster_grid.ax_heatmap.invert_xaxis()
            cluster_grid.ax_col_dendrogram.invert_xaxis()

    cluster_grid.ax_heatmap.set_xlabel("Cell Type in Tahoe", fontsize=24)
    cluster_grid.ax_heatmap.set_ylabel("Genetic Perturbation Reconstructed Through Drug Combinations", fontsize=24)
    cluster_grid.ax_heatmap.set_title(title)
    cluster_grid.ax_heatmap.tick_params(axis="x", labelrotation=90)
    cluster_grid.fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_grid.savefig(output_path, dpi=dpi)
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    cluster_grid.savefig(pdf_path, format='pdf')
    plt.close(cluster_grid.fig)

    # Extract reordered indices from the clustermap
    row_order = cluster_grid.dendrogram_row.reordered_ind
    col_order = cluster_grid.dendrogram_col.reordered_ind
    reordered_pert_names = [pert_names[i] for i in row_order]
    reordered_cell_names = [cell_names[i] for i in col_order]

    if summary_depth is not None and summary_depth > 0:
        summaries: List[Dict[str, object]] = []
        row_linkage = cluster_grid.dendrogram_row.linkage
        col_linkage = cluster_grid.dendrogram_col.linkage
        if row_linkage is not None and len(pert_names) > 1:
            summaries.extend(
                summarize_dendrogram(
                    row_linkage, list(pert_names), df.values, axis="perturbation", max_depth=summary_depth
                )
            )
        if col_linkage is not None and len(cell_names) > 1:
            summaries.extend(
                summarize_dendrogram(
                    col_linkage, list(cell_names), df.values, axis="cell", max_depth=summary_depth
                )
            )
        return summaries, reordered_pert_names, reordered_cell_names
    return None, reordered_pert_names, reordered_cell_names


def main() -> None:
    args = parse_args()
    cell_dirs = discover_cell_dirs(args.input_root, args.max_drugs)
    if not cell_dirs:
        raise RuntimeError(f"No cell directories found under {args.input_root}")

    output_dir = args.output_dir or (args.input_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.cluster_summary_dir
    summary_depth: Optional[int] = None
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_depth = max(0, args.cluster_summary_depth)

    pert_order = load_perturbation_order(
        cell_dirs,
        args.max_drugs,
        cache_file=args.cache_pert_order,
    )

    heatmap_payload = assemble_heatmap_data(
        cell_dirs,
        args.max_drugs,
        pert_order,
        normalize_columns=args.column_normalize,
    )

    cell_names = [cell_dir.name for cell_dir in cell_dirs]
    pert_names, reduced_payload = maybe_limit_perturbations(
        heatmap_payload,
        pert_order,
        args.limit_perts,
    )

    # Determine shared color limits for comparability
    effect_values = [
        payload["effect"] for payload in reduced_payload.values()
    ]
    combined_effect = np.concatenate([arr.flatten() for arr in effect_values])
    combined_effect = combined_effect[~np.isnan(combined_effect)]
    effect_max = float(combined_effect.max()) if combined_effect.size else None

    raw_values = [
        payload["raw"] for payload in reduced_payload.values()
    ]
    combined_raw = np.concatenate([arr.flatten() for arr in raw_values])
    combined_raw = combined_raw[~np.isnan(combined_raw)]
    raw_max = float(combined_raw.max()) if combined_raw.size else None

    for max_drug, matrices in sorted(reduced_payload.items()):
        if raw_max:
            raw_path = output_dir / f"heatmap_max_drugs_{max_drug}_raw.png"
            _, _, _ = plot_heatmap(
                matrices["raw"],
                pert_names,
                cell_names,
                title=f"Predicted embedding norm – max_drugs={max_drug}",
                cmap=args.cmap,
                output_path=raw_path,
                dpi=args.fig_dpi,
                vmin=0.0,
                vmax=raw_max,
            )
        effect_path = output_dir / f"heatmap_max_drugs_{max_drug}_effect.png"
        effect_summaries, _, _ = plot_heatmap(
            matrices["effect"],
            pert_names,
            cell_names,
            title=f"Perturbation effect (predicted - baseline) – max_drugs={max_drug}",
            cmap=args.diff_cmap,
            output_path=effect_path,
            dpi=args.fig_dpi,
            vmin=0.0,
            vmax=effect_max,
            summary_depth=summary_depth,
        )
        if summary_dir is not None and effect_summaries:
            summary_path = summary_dir / f"heatmap_max_drugs_{max_drug}_effect_clusters.csv"
            write_cluster_summary(effect_summaries, summary_path)
        if args.column_normalize and "effect_colnorm" in matrices:
            effect_norm_path = output_dir / f"heatmap_max_drugs_{max_drug}_effect_colnorm.png"
            _, clustered_pert_names, clustered_cell_names = plot_heatmap(
                matrices["effect_colnorm"],
                pert_names,
                cell_names,
                title=(
                    "Perturbation effect (baseline subtracted, column-normalized) "
                    f"– max_drugs={max_drug}"
                ),
                cmap=args.diff_cmap,
                output_path=effect_norm_path,
                dpi=args.fig_dpi,
                vmin=0.0,
                vmax=1.0,
            )
            # Save the matrix for max_drugs=32
            if max_drug == 32:
                # Reorder the matrix to match the clustered order
                pert_indices = [list(pert_names).index(name) for name in clustered_pert_names]
                cell_indices = [list(cell_names).index(name) for name in clustered_cell_names]
                clustered_matrix = matrices["effect_colnorm"][np.ix_(pert_indices, cell_indices)]
                
                matrix_path = output_dir / f"heatmap_max_drugs_{max_drug}_effect_colnorm_matrix.npy"
                np.save(matrix_path, clustered_matrix)
                # Save clustered perturbation order as text file (one per line)
                pert_order_path = output_dir / f"heatmap_max_drugs_{max_drug}_effect_colnorm_pert_order.txt"
                with open(pert_order_path, 'w') as f:
                    for pert in clustered_pert_names:
                        f.write(f"{pert}\n")
                # Save clustered cell order as text file (one per line)
                cell_order_path = output_dir / f"heatmap_max_drugs_{max_drug}_effect_colnorm_cell_order.txt"
                with open(cell_order_path, 'w') as f:
                    for cell in clustered_cell_names:
                        f.write(f"{cell}\n")
                print(f"Saved clustered matrix ({clustered_matrix.shape}) to {matrix_path}")
                print(f"Saved clustered perturbation order to {pert_order_path}")
                print(f"Saved clustered cell order to {cell_order_path}")

    # Combined overview for perturbation effects across all bins
    combined_fig = output_dir / "perturbation_effect_overview.png"
    stacked_effects = []
    stacked_columns = []
    for max_drug, matrices in sorted(reduced_payload.items()):
        stacked_effects.append(matrices["effect"])
        stacked_columns.extend(
            [f"{cell}-max{max_drug}" for cell in cell_names]
        )
    if stacked_effects:
        merged = np.concatenate(stacked_effects, axis=1)
        overview_summaries, _, _ = plot_heatmap(
            merged,
            pert_names,
            stacked_columns,
            title="Perturbation effect overview (baseline subtracted)",
            cmap=args.diff_cmap,
            output_path=combined_fig,
            dpi=args.fig_dpi,
            vmin=0.0,
            vmax=effect_max,
            summary_depth=summary_depth,
        )
        if summary_dir is not None and overview_summaries:
            overview_path = summary_dir / "perturbation_effect_overview_clusters.csv"
            write_cluster_summary(overview_summaries, overview_path)
    if args.column_normalize:
        stacked_norm = []
        stacked_norm_cols = []
        for max_drug, matrices in sorted(reduced_payload.items()):
            if "effect_colnorm" not in matrices:
                continue
            stacked_norm.append(matrices["effect_colnorm"])
            stacked_norm_cols.extend(
                [f"{cell}-max{max_drug}" for cell in cell_names]
            )
        if stacked_norm:
            merged_norm = np.concatenate(stacked_norm, axis=1)
            combined_norm_fig = output_dir / "perturbation_effect_overview_colnorm.png"
            _, _, _ = plot_heatmap(
                merged_norm,
                pert_names,
                stacked_norm_cols,
                title="Perturbation effect overview (column-normalized)",
                cmap=args.diff_cmap,
                output_path=combined_norm_fig,
                dpi=args.fig_dpi,
                vmin=0.0,
                vmax=1.0,
            )

    print(f"Saved heatmaps to {output_dir}")


if __name__ == "__main__":
    main()
