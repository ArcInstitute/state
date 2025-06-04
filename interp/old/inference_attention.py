#!/usr/bin/env python


from __future__ import annotations

import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
import pandas as pd

# ─── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# ─── project imports & Paths ----------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'vci_pretrain'))
from inference_module import InferenceModule  # project‑specific import

# ------------------------------------------------------------------------------
# USER‑EDITABLE PATHS -----------------------------------------------------------
MODEL_DIR = Path("/large_storage/ctc/userspace/dhruvgautam/qwen/tahoe_vci_1.4.4_70m_mha/mha_fold1")
DATA_PATH = Path("/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct_vci_1.4.4/plate1.h5")
CELL_SET_LEN = 512  # maximum sentence length
CONTROL_SAMPLES = 50  # how many control cells to sample per perturbation subset
LAYER_IDX = 3  # zero‑based index of Qwen layer to analyse
FIG_DIR = Path(__file__).resolve().parent / "figures" / "layer3_attention"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------------------------

# ─── prepare model & data -------------------------------------------------------
inference_module = InferenceModule(model_folder=str(MODEL_DIR))
model = inference_module.model  # convenience alias

# Ensure attention weights are produced (needed when *not* using the hook)
model.transformer_backbone.config.output_attentions = True
logger.info("Loaded PertSets model with Qwen‑3 backbone: %s heads × %s layers",
            model.transformer_backbone.config.num_attention_heads,
            model.transformer_backbone.config.num_hidden_layers)

# Read AnnData and subset to 512 cells (stratified control / non‑control)
adata_full = sc.read_h5ad(str(DATA_PATH))
control_pert = inference_module.data_module.control_pert  # e.g. "non‑targeting"
logger.info("Control perturbation: %s", control_pert)

if control_pert is not None and control_pert in adata_full.obs["drugname_drugconc"].unique():
    ctrl_mask = adata_full.obs["drugname_drugconc"] == control_pert
    ctrl_cells = adata_full[ctrl_mask]
    non_ctrl_cells = adata_full[~ctrl_mask]

    n_ctrl = min(CONTROL_SAMPLES, ctrl_cells.n_obs)
    n_non_ctrl = CELL_SET_LEN - n_ctrl
    n_non_ctrl = min(n_non_ctrl, non_ctrl_cells.n_obs)

    adata_subset = ad.concat([
        ctrl_cells[np.random.choice(ctrl_cells.n_obs, n_ctrl, replace=False)],
        non_ctrl_cells[np.random.choice(non_ctrl_cells.n_obs, n_non_ctrl, replace=False)],
    ])
else:
    logger.warning("No control perturbation found – taking first %d cells", CELL_SET_LEN)
    adata_subset = adata_full[:CELL_SET_LEN].to_memory()

logger.info("Subsetted data shape: %s", adata_subset.shape)

# ─── global accumulator for attention ------------------------------------------
layer_sum: torch.Tensor | None = None  # cumulative sum of attn weights  [H,S,S]
layer_count: int = 0                   # number of batches aggregated


def layer3_hook(_: torch.nn.Module, __, outputs: Tuple[torch.Tensor, torch.Tensor]):
    """Forward hook to capture (attn_output, attn_weights)."""
    global layer_sum, layer_count
    attn_weights = outputs[1].detach().cpu()  # shape [B,H,S,S]
    batch_sz = attn_weights.shape[0]
    # Sum over batch dimension → [H,S,S]
    attn_sum = attn_weights.sum(0)
    if layer_sum is None:
        layer_sum = attn_sum
    else:
        layer_sum += attn_sum
    layer_count += batch_sz


# Attach hook ONCE
hook_handle = model.transformer_backbone.layers[LAYER_IDX].self_attn.register_forward_hook(layer3_hook)
logger.info("Registered hook on transformer_backbone.layers[%d].self_attn", LAYER_IDX)

# ─── iterate over perturbations -------------------------------------------------
pert_counts = adata_subset.obs["drugname_drugconc"].value_counts()
logger.info("Found %d unique perturbations", len(pert_counts))

all_ctrl_cells = adata_subset[adata_subset.obs["drugname_drugconc"] == control_pert].copy()

for pert in pert_counts.index:
    n_cells = pert_counts[pert]
    logger.info("Running inference for %-30s  (%d cells)", pert, n_cells)

    # sample control cells equal to n_cells (with replacement if needed)
    sample_idx = np.random.choice(all_ctrl_cells.n_obs, size=n_cells, replace=n_cells > all_ctrl_cells.n_obs)
    batch = all_ctrl_cells[sample_idx].copy()
    batch.obs["drugname_drugconc"] = pert  # overwrite control with current perturbation

    # forward pass – hook will collect attention
    _ = inference_module.perturb(batch, pert_key="drugname_drugconc", celltype_key="cell_name")

logger.info("Finished all perturbations – accumulated %d batches", layer_count)
assert layer_sum is not None, "Hook did not run – check layer index"

# ─── normalise & plot -----------------------------------------------------------
avg_attn = layer_sum / layer_count  # [H,S,S]
H, S, _ = avg_attn.shape
cols = 4
rows = math.ceil(H / cols)
plt.figure(figsize=(3 * cols, 3 * rows))
for h in range(H):
    plt.subplot(rows, cols, h + 1)
    sns.heatmap(avg_attn[h], square=True, cbar=False)
    plt.title(f"Head {h}", fontsize=8)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
fig_path = FIG_DIR / "layer3_avg_attention_heads.png"
plt.savefig(fig_path, dpi=300)
plt.close()
logger.info("Saved average attention heat‑maps → %s", fig_path)

# ─── clean‑up -------------------------------------------------------------------
hook_handle.remove()
logger.info("Hook removed, script completed successfully.")
