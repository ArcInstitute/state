#!/usr/bin/env python
"""
calculate_layer3_attention.py – v3
=================================
Fixes the issue where attention was mostly zeros because the model processes
very short sequences (1-7 tokens) rather than 512-token sequences as expected.

The script now:
* Collects attention from all sequences regardless of length
* Uses a dynamic canvas that adapts to the maximum sequence length observed
* Provides better visualization of the actual attention patterns
* Shows per-head statistics and sequence length distributions
"""

from __future__ import annotations

import os
import sys
import math
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent  # adjust if needed
sys.path.append(str(project_root))
sys.path.append(str(project_root / "vci_pretrain"))
from inference_module import InferenceModule  # noqa: E402

MODEL_DIR = Path(
    "/large_storage/ctc/userspace/dhruvgautam/qwen/tahoe_vci_1.4.4_70m_mha/mha_fold1"
)
DATA_PATH = Path(
    "/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct_vci_1.4.4/plate1.h5"
)
CELL_SET_LEN = 512   
CONTROL_SAMPLES = 50 
LAYER_IDX = 3  
FIG_DIR = Path(__file__).resolve().parent / "figures" / "layer3_attention"
FIG_DIR.mkdir(parents=True, exist_ok=True)

inference_module = InferenceModule(model_folder=str(MODEL_DIR))
model = inference_module.model  # alias
model.eval()  # turn off dropout
model.transformer_backbone.config._attn_implementation = "eager"
model.transformer_backbone._attn_implementation        = "eager"

model.transformer_backbone.config.output_attentions = True
if hasattr(model, 'config'):
    model.config.output_attentions = True

for layer in model.transformer_backbone.layers:
    if hasattr(layer.self_attn, 'output_attentions'):
        layer.self_attn.output_attentions = True

print(f"Model config output_attentions: {getattr(model.transformer_backbone.config, 'output_attentions', 'Not found')}")
print(f"Number of transformer layers: {len(model.transformer_backbone.layers)}")
print(f"Target layer index: {LAYER_IDX}")
print(f"Self-attention module type: {type(model.transformer_backbone.layers[LAYER_IDX].self_attn)}")

logger.info(
    "Loaded PertSets model with Qwen‑3 backbone: %s heads × %s layers",
    model.transformer_backbone.config.num_attention_heads,
    model.transformer_backbone.config.num_hidden_layers,
)

# Add debugging for cell sentence length
print(f"Model cell_sentence_len: {model.cell_sentence_len}")
print(f"Data module cell_sentence_len: {inference_module.data_module.cell_sentence_len}")
print(f"InferenceModule cell_set_len: {inference_module.cell_set_len}")

adata_full = sc.read_h5ad(str(DATA_PATH))
control_pert = inference_module.data_module.control_pert  
logger.info("Control perturbation: %s", control_pert)

# Use the full dataset instead of subsetting to get more cells for attention analysis
logger.info("Using full dataset shape: %s", adata_full.shape)

NUM_HEADS = model.transformer_backbone.config.num_attention_heads  # 12

attention_by_length: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
sequence_lengths: List[int] = []
total_sequences: int = 0    


def layer3_hook(_: torch.nn.Module, __, outputs: Tuple[torch.Tensor, torch.Tensor]):
    """Collect attention matrices organized by sequence length."""
    global attention_by_length, sequence_lengths, total_sequences

    print(f"Hook called! Number of outputs: {len(outputs)}")
    print(f"Output shapes: {[o.shape if hasattr(o, 'shape') else type(o) for o in outputs]}")
    
    if len(outputs) < 2:
        print("Warning: Expected at least 2 outputs, but got", len(outputs))
        return
    
    attn_weights = None
    for i, output in enumerate(outputs):
        if hasattr(output, 'shape') and len(output.shape) == 4:
            print(f"Found 4D tensor at outputs[{i}] with shape: {output.shape}")
            # Check if this looks like attention weights [B, H, S, S]
            B, H, S1, S2 = output.shape
            if S1 == S2:  # Square matrix indicates attention weights
                attn_weights = output.detach().cpu()
                print(f"Using outputs[{i}] as attention weights: [B={B}, H={H}, S={S1}]")
                break
    
    if attn_weights is None:
        print("Warning: Could not find attention weights in outputs")
        return

    batch_sz, H, S, _ = attn_weights.shape

    # Check if attention weights contain meaningful values
    print(f"Attention stats: min={attn_weights.min():.6f}, max={attn_weights.max():.6f}, mean={attn_weights.mean():.6f}")
    
    # Store each sequence in the batch separately by length
    for b in range(batch_sz):
        sequence_lengths.append(S)
        for h in range(H):
            attention_by_length[S][h].append(attn_weights[b, h])
    
    total_sequences += batch_sz
    print(f"Added {batch_sz} sequences of length {S} to accumulator (total: {total_sequences})")


# Attach hook once
hook_handle = (
    model.transformer_backbone.layers[LAYER_IDX].self_attn.register_forward_hook(layer3_hook)
)
logger.info(
    "Registered hook on transformer_backbone.layers[%d].self_attn", LAYER_IDX
)

pert_counts = adata_full.obs["drugname_drugconc"].value_counts()
logger.info("Found %d unique perturbations", len(pert_counts))

all_ctrl_cells = adata_full[adata_full.obs["drugname_drugconc"] == control_pert].copy()

for pert in pert_counts.index:
    n_cells = int(pert_counts[pert])
    logger.info("Running inference for %-30s (%d cells)", pert, n_cells)

    # sample control cells equal to n_cells (with replacement if needed)
    sample_idx = np.random.choice(
        all_ctrl_cells.n_obs, size=n_cells, replace=n_cells > all_ctrl_cells.n_obs
    )
    batch = all_ctrl_cells[sample_idx].copy()
    batch.obs["drugname_drugconc"] = pert  # overwrite control with current perturbation

    # Add debugging information
    print(f"DEBUG: Processing perturbation '{pert}' with {n_cells} cells")
    print(f"DEBUG: Batch shape before inference: {batch.shape}")
    print(f"DEBUG: Unique cell types in batch: {batch.obs['cell_name'].nunique()}")
    print(f"DEBUG: Cell type distribution:")
    for cell_type, count in batch.obs['cell_name'].value_counts().items():
        print(f"  {cell_type}: {count} cells")

    # BETTER ATTENTION FIX: Focus on the most abundant cell type for meaningful attention analysis
    # Find the most common cell type in this batch
    most_common_celltype = batch.obs['cell_name'].value_counts().index[0]
    most_common_count = batch.obs['cell_name'].value_counts().iloc[0]
    
    print(f"DEBUG: Most common cell type: {most_common_celltype} ({most_common_count} cells)")
    
    # If the most common cell type has enough cells for meaningful attention analysis
    min_cells_for_attention = 10  # Minimum cells needed for meaningful attention patterns
    if most_common_count >= min_cells_for_attention:
        # Filter to only include the most common cell type
        celltype_mask = batch.obs['cell_name'] == most_common_celltype
        batch_filtered = batch[celltype_mask].copy()
        print(f"DEBUG: Using {batch_filtered.n_obs} cells of type '{most_common_celltype}' for attention analysis")
        
        # forward pass – hook collects attention
        _ = inference_module.perturb(
            batch_filtered, pert_key="drugname_drugconc", celltype_key="cell_name"
        )
    else:
        print(f"DEBUG: Skipping '{pert}' - largest cell type group ({most_common_count}) too small for attention analysis")
        continue

logger.info(
    "Finished all perturbations – accumulated %d sequences", total_sequences
)
assert total_sequences > 0, "Hook did not run – check layer index or data"


# Print statistics about sequence lengths
print(f"\nSequence length distribution:")
from collections import Counter
length_counts = Counter(sequence_lengths)
for length in sorted(length_counts.keys()):
    print(f"  Length {length}: {length_counts[length]} sequences")

# Compute average attention for each sequence length and head
max_length = max(sequence_lengths) if sequence_lengths else 1
print(f"Maximum sequence length observed: {max_length}")

# Create separate plots for each sequence length
for seq_len in sorted(attention_by_length.keys()):
    if seq_len == 1:
        continue  # Skip single-token sequences (they're just identity)
    
    n_sequences = sum(len(attention_by_length[seq_len][h]) for h in range(NUM_HEADS))
    if n_sequences == 0:
        continue
        
    print(f"\nProcessing {n_sequences} attention matrices of length {seq_len}")
    
    # Average attention across all sequences of this length
    avg_attn_by_head = {}
    for h in range(NUM_HEADS):
        if len(attention_by_length[seq_len][h]) > 0:
            # Stack and average all attention matrices for this head and sequence length
            stacked = torch.stack(attention_by_length[seq_len][h])
            avg_attn_by_head[h] = stacked.mean(0)  # [S, S]
        else:
            avg_attn_by_head[h] = torch.zeros(seq_len, seq_len)
    
    # Plot this sequence length
    cols = 4
    rows = math.ceil(NUM_HEADS / cols)
    plt.figure(figsize=(3 * cols, 3 * rows))
    
    for h in range(NUM_HEADS):
        plt.subplot(rows, cols, h + 1)
        attn_head = avg_attn_by_head[h].numpy()
        
        if len(attention_by_length[seq_len][h]) > 0:
            sns.heatmap(
                attn_head, square=True, cbar=True, cmap='viridis', 
                xticklabels=range(seq_len), yticklabels=range(seq_len),
                vmin=0, vmax=1
            )
            plt.xlabel("Key position")
            plt.ylabel("Query position")
            n_matrices = len(attention_by_length[seq_len][h])
            plt.title(f"Head {h} (n={n_matrices})", fontsize=10)
        else:
            plt.text(0.5, 0.5, f"Head {h}\n(No Data)", ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f"Head {h} (Empty)", fontsize=10)
    
    plt.suptitle(f"Layer {LAYER_IDX} Attention - Sequence Length {seq_len}", fontsize=14)
    plt.tight_layout()
    fig_path = FIG_DIR / f"layer{LAYER_IDX}_attention_length{seq_len}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved attention heatmaps for length %d → %s", seq_len, fig_path)

# Create a summary plot showing attention patterns across all lengths
if len(attention_by_length) > 1:
    # Find the most common non-trivial sequence length for the summary
    non_trivial_lengths = [l for l in sequence_lengths if l > 1]
    if non_trivial_lengths:
        most_common_length = Counter(non_trivial_lengths).most_common(1)[0][0]
        
        # Create summary using the most common length
        plt.figure(figsize=(15, 12))
        for h in range(NUM_HEADS):
            plt.subplot(3, 4, h + 1)
            if len(attention_by_length[most_common_length][h]) > 0:
                stacked = torch.stack(attention_by_length[most_common_length][h])
                avg_attn = stacked.mean(0).numpy()
                sns.heatmap(
                    avg_attn, square=True, cbar=True, cmap='viridis',
                    xticklabels=False, yticklabels=False, vmin=0, vmax=1
                )
                n_matrices = len(attention_by_length[most_common_length][h])
                plt.title(f"Head {h} (len={most_common_length}, n={n_matrices})", fontsize=10)
            else:
                plt.text(0.5, 0.5, f"Head {h}\n(No Data)", ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title(f"Head {h} (No Data)", fontsize=10)
        
        plt.suptitle(f"Layer {LAYER_IDX} Average Attention Patterns - Most Common Length ({most_common_length})", fontsize=16)
        plt.tight_layout()
        fig_path = FIG_DIR / f"layer{LAYER_IDX}_attention_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved summary attention heatmap → %s", fig_path)

hook_handle.remove()
logger.info("Hook removed; script completed successfully.")
logger.info(f"Generated attention visualizations in {FIG_DIR}")
