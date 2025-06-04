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
* Updated to work with GPT-2 architecture
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
    "/large_storage/ctc/userspace/aadduri/feb28_conc/cv_norm_vci_1.4.2_samp_ctrl_70m_512/fold2"
)
DATA_PATH = Path(
    "/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct_vci_1.4.2/plate1.h5"
)
CELL_SET_LEN = 512   
CONTROL_SAMPLES = 50 
LAYER_IDX = 1  
FIG_DIR = Path(__file__).resolve().parent / "figures" / "layer1_attention"
FIG_DIR.mkdir(parents=True, exist_ok=True)

inference_module = InferenceModule(model_folder=str(MODEL_DIR))
model = inference_module.model  # alias
model.eval()  # turn off dropout
model.transformer_backbone.config._attn_implementation = "eager"
model.transformer_backbone._attn_implementation        = "eager"

model.transformer_backbone.config.output_attentions = True
if hasattr(model, 'config'):
    model.config.output_attentions = True

# Updated for GPT-2 architecture using .h instead of .layers
for layer in model.transformer_backbone.h:
    if hasattr(layer.attn, 'output_attentions'):
        layer.attn.output_attentions = True

print(f"Model config output_attentions: {getattr(model.transformer_backbone.config, 'output_attentions', 'Not found')}")
print(f"Number of transformer layers: {len(model.transformer_backbone.h)}")
print(f"Target layer index: {LAYER_IDX}")
print(f"Attention module type: {type(model.transformer_backbone.h[LAYER_IDX].attn)}")

logger.info(
    "Loaded PertSets model with GPT-2 backbone: %s heads × %s layers",
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


# Attach hook once - Updated for GPT-2 architecture
hook_handle = (
    model.transformer_backbone.h[LAYER_IDX].attn.register_forward_hook(layer3_hook)
)
logger.info(
    "Registered hook on transformer_backbone.h[%d].attn", LAYER_IDX
)

# NEW: Direct model forward pass with 512 cells (256 of each cell type)
# Get available cell types and select the two most abundant ones
cell_type_counts = adata_full.obs["cell_name"].value_counts()
logger.info("Available cell types: %s", list(cell_type_counts.index))

# Select the two most abundant cell types
celltype1, celltype2 = cell_type_counts.index[:2]
logger.info(f"Selected cell types: {celltype1} ({cell_type_counts[celltype1]} available), {celltype2} ({cell_type_counts[celltype2]} available)")

# Get control cells for each cell type
cells_type1 = adata_full[(adata_full.obs["drugname_drugconc"] == control_pert) & 
                        (adata_full.obs["cell_name"] == celltype1)].copy()
cells_type2 = adata_full[(adata_full.obs["drugname_drugconc"] == control_pert) & 
                        (adata_full.obs["cell_name"] == celltype2)].copy()

logger.info(f"Available cells - {celltype1}: {cells_type1.n_obs}, {celltype2}: {cells_type2.n_obs}")

# Sample 256 cells from each cell type
n_cells_per_type = 256
if cells_type1.n_obs >= n_cells_per_type and cells_type2.n_obs >= n_cells_per_type:
    # Sample cells
    idx1 = np.random.choice(cells_type1.n_obs, size=n_cells_per_type, replace=False)
    idx2 = np.random.choice(cells_type2.n_obs, size=n_cells_per_type, replace=False)
    
    sampled_type1 = cells_type1[idx1].copy()
    sampled_type2 = cells_type2[idx2].copy()
    
    # Combine into a single batch of 512 cells
    combined_batch = ad.concat([sampled_type1, sampled_type2], axis=0)
    
    logger.info(f"Created combined batch with {combined_batch.n_obs} cells")
    logger.info(f"Cell type distribution: {combined_batch.obs['cell_name'].value_counts().to_dict()}")
    
    # Get embeddings and prepare batch manually (bypassing inference module grouping)
    device = next(model.parameters()).device
    X_embed = torch.tensor(combined_batch.obsm[inference_module.embed_key], dtype=torch.float32).to(device)
    
    # Get perturbation one-hot (using control perturbation)
    pert_onehot = inference_module.pert_onehot_map[control_pert].to(device)
    pert_tensor = pert_onehot.unsqueeze(0).repeat(combined_batch.n_obs, 1)  # (512, pert_dim)
    pert_names = [control_pert] * combined_batch.n_obs
    
    # Create batch dictionary
    batch = {
        "basal": X_embed,  # (512, embed_dim)
        "pert": pert_tensor,  # (512, pert_dim)
        "pert_name": pert_names
    }
    
    logger.info(f"Batch shapes - basal: {batch['basal'].shape}, pert: {batch['pert'].shape}")
    logger.info("Running single forward pass with 512 cells (256 per cell type)")
    
    # Single forward pass
    with torch.no_grad():
        if isinstance(model, type(inference_module.model)):  # Check if it's a PertSetsPerturbationModel
            # Call the model directly with the batch
            batch_pred = model.forward(batch, padded=False)
        else:
            batch_pred = model.forward(batch)
    
    logger.info("Forward pass completed successfully")
    
else:
    logger.error(f"Insufficient cells: {celltype1}: {cells_type1.n_obs}, {celltype2}: {cells_type2.n_obs}. Need at least {n_cells_per_type} each.")

logger.info(
    "Finished inference – accumulated %d sequences", total_sequences
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
    fig_path = FIG_DIR / f"split_layer{LAYER_IDX}_attention_length{seq_len}.png"
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
        fig_path = FIG_DIR / f"split_layer{LAYER_IDX}_attention_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved summary attention heatmap → %s", fig_path)

hook_handle.remove()
logger.info("Hook removed; script completed successfully.")
logger.info(f"Generated attention visualizations in {FIG_DIR}")
