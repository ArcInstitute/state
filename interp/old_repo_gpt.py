#!/usr/bin/env python
"""
calculate_layer3_attention.py – v4
=================================
Updated to bypass InferenceModule and data_module.pkl, loading the model directly 
and using dummy perturbation tensors for attention analysis.

The script now:
* Loads the PertSetsPerturbationModel directly from checkpoint
* Uses dummy perturbation tensors instead of relying on data module
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

# Import model class directly
from models.pertsets import PertSetsPerturbationModel

MODEL_DIR = Path(
    # "/large_storage/ctc/userspace/aadduri/preprint/replogle_vci_1.5.2_cs64/fold1"
    # "/large_storage/ctc/userspace/rohankshah/preprint/replogle_gpt_31043724/hepg2"
    # "/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered/hepg2"
    #"/large_storage/ctc/userspace/rohankshah"
    #"/large_storage/ctc/userspace/aadduri/preprint/replogle_hvg_proper_cs32_sm/hepg2"
    "/large_storage/ctc/userspace/aadduri/preprint/replogle_state_proper_cs32_sm/hepg2"
)
DATA_PATH = Path(
    "/large_storage/ctc/ML/state_sets/replogle/processed.h5"
)
CELL_SET_LEN = 512   
CONTROL_SAMPLES = 50 
LAYER_IDX = 3
FIG_DIR = Path(__file__).resolve().parent / "figures" / "replogle_gpt_21_new_cs32_split" 
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load model directly from checkpoint
#checkpoint_path = MODEL_DIR / "step=148000.ckpt"
#checkpoint_path = MODEL_DIR / "checkpoints" / "step=124000.ckpt" # "step=step=104000-val_loss=val_loss=0.0472.ckpt"
checkpoint_path = MODEL_DIR / "checkpoints" / "step=48000.ckpt"

if not checkpoint_path.exists():
    # Try other common checkpoint names
    for ckpt_name in ["best.ckpt", "last.ckpt", "epoch=*.ckpt"]:
        ckpt_files = list(MODEL_DIR.glob(f"checkpoints/{ckpt_name}"))
        if ckpt_files:
            checkpoint_path = ckpt_files[0]
            break
    else:
        raise FileNotFoundError(f"Could not find checkpoint in {MODEL_DIR}/checkpoints/")

logger.info(f"Loading model from checkpoint: {checkpoint_path}")
model = PertSetsPerturbationModel.load_from_checkpoint(str(checkpoint_path), strict=False)
model.eval()  # turn off dropout

# Configure attention outputs
model.transformer_backbone.config._attn_implementation = "sdpa"
model.transformer_backbone._attn_implementation        = "sdpa"
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

# Load data directly
adata_full = sc.read_h5ad(str(DATA_PATH))
logger.info("Using full dataset shape: %s", adata_full.shape)

# Debug: Explore the AnnData structure
logger.info("=== AnnData Structure Debug ===")
logger.info(f"Observation columns (adata.obs.columns): {list(adata_full.obs.columns)}")
logger.info(f"Variable columns (adata.var.columns): {list(adata_full.var.columns)}")
logger.info(f"Observation matrices (adata.obsm.keys()): {list(adata_full.obsm.keys())}")
logger.info(f"Unstructured annotations (adata.uns.keys()): {list(adata_full.uns.keys())}")
if hasattr(adata_full, 'layers') and adata_full.layers:
    logger.info(f"Data layers (adata.layers.keys()): {list(adata_full.layers.keys())}")
logger.info("=== End Debug ===\n")

# Determine embedding key - typically "X_vci" for VCI embeddings or "X_hvg" for HVG expression
embed_key = "X_vci_1.5.2_4"  # Try VCI embeddings first
if embed_key not in adata_full.obsm:
    embed_key = "X_hvg"  # Fallback to HVG
    if embed_key not in adata_full.obsm:
        embed_key = "X"  # Last resort - raw expression
        logger.warning("Using raw expression data - model may not work properly")
logger.info(f"embed_key after: {embed_key}")
logger.info(f"Using embedding key: {embed_key}")
if embed_key in adata_full.obsm:
    logger.info(f"Embedding shape: {adata_full.obsm[embed_key].shape}")
else:
    logger.info(f"Expression shape: {adata_full.X.shape}")

# Find control perturbation
control_pert = "DMSO_TF_24h"  # Default control
if "gene" in adata_full.obs.columns:
    drugname_counts = adata_full.obs["gene"].value_counts()
    # Look for common control names
    for potential_control in ["DMSO_TF_24h", "non-targeting", "control", "DMSO"]:
        if potential_control in drugname_counts.index:
            control_pert = potential_control
            break
    else:
        # Use the most common perturbation as control
        control_pert = drugname_counts.index[0]
        logger.warning(f"Could not find standard control, using: {control_pert}")

logger.info("Control perturbation: %s", control_pert)

NUM_HEADS = model.transformer_backbone.config.num_attention_heads  # 8

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

# Direct model forward pass with 512 cells (256 of each cell type)
# Get available cell types and select the two most abundant ones
cell_type_counts = adata_full.obs["cell_type"].value_counts()
logger.info("Available cell types: %s", list(cell_type_counts.index))

# Select the most abundant cell type
celltype1 = cell_type_counts.index[0]
celltype2 = cell_type_counts.index[1]
logger.info(f"Selected cell type: {celltype1} ({cell_type_counts[celltype1]} available)")

# Get control cells for this cell type
cells_type1 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype1)].copy()
cells_type2 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype2)].copy()

logger.info(f"Available cells - {celltype1}: {cells_type1.n_obs}")

# Use the model's actual cell_sentence_len to avoid position embedding errors
cell_sentence_len = model.cell_sentence_len  # Use model's trained sequence length
logger.info(f"Using model's cell_sentence_len: {cell_sentence_len}")
logger.info(f"Model was trained with cell_sentence_len: {model.cell_sentence_len}")

# Use all cells for the single cell type
n_cells = cell_sentence_len
total_cells = cell_sentence_len  # Use model's trained length

if cells_type2.n_obs >= n_cells // 2:
    # Sample cells
    idx1 = np.random.choice(cells_type1.n_obs, size=n_cells // 2, replace=False)
    idx2 = np.random.choice(cells_type2.n_obs, size=n_cells // 2, replace=False)
    
    sampled_type1 = cells_type1[idx1].copy()
    sampled_type2 = cells_type2[idx2].copy()
    
    # Use single cell type batch
    combined_batch = ad.concat([sampled_type1, sampled_type2], axis=0)
    
    logger.info(f"Created combined batch with {combined_batch.n_obs} cells")
    logger.info(f"Cell type distribution: {combined_batch.obs['cell_type'].value_counts().to_dict()}")
    
    # Get embeddings and prepare batch manually
    device = next(model.parameters()).device
    
    # Extract embeddings based on available key
    if embed_key in combined_batch.obsm:
        X_embed = torch.tensor(combined_batch.obsm[embed_key], dtype=torch.float32).to(device)
    else:
        X_embed = torch.tensor(combined_batch.X.toarray() if hasattr(combined_batch.X, 'toarray') else combined_batch.X, 
                             dtype=torch.float32).to(device)
    
    # Create dummy perturbation tensor
    # Get pert_dim from model
    pert_dim = model.pert_dim
    logger.info(f"Using pert_dim: {pert_dim}")
    
    # Get the number of cells we actually have
    n_cells = X_embed.shape[0]
    logger.info(f"Number of cells: {n_cells}")
    
    # Create one-hot tensor for control perturbation  
    pert_tensor = torch.zeros((n_cells, pert_dim), device=device)
    pert_tensor[:, 0] = 1  # Set first dimension to 1 for control perturbation
    pert_names = [control_pert] * n_cells
    
    # Create batch dictionary - use the correct keys expected by the model
    batch = {
        "ctrl_cell_emb": X_embed,  # (64, embed_dim) - basal/control cell embeddings
        "pert_emb": pert_tensor,  # (64, pert_dim) - perturbation embeddings
        "pert_name": pert_names,
        "batch": torch.zeros((1, cell_sentence_len), device=device)
    }
    
    logger.info(f"Batch shapes - ctrl_cell_emb: {batch['ctrl_cell_emb'].shape}, pert_emb: {batch['pert_emb'].shape}")
    logger.info(f"Running single forward pass with {n_cells} cells of type {celltype1}")
    
    # Single forward pass using padded=True
    with torch.no_grad():
        batch_pred = model.forward(batch, padded=False)
    
    logger.info("Forward pass completed successfully")
    
else:
    logger.error(f"Insufficient cells: {celltype1}: {cells_type1.n_obs}. Need at least {n_cells}.")

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
        max_attn = attn_head.max()
        min_attn = attn_head.min()
        if len(attention_by_length[seq_len][h]) > 0:
            sns.heatmap(
                attn_head, square=True, cbar=True, cmap='Greens', 
                xticklabels=False, yticklabels=False,
                vmin=min_attn, vmax=max_attn
            )
            plt.xlabel("Key position")
            plt.ylabel("Query position")
            n_matrices = len(attention_by_length[seq_len][h])
            plt.title(f"Head {h}", fontsize=10)
        else:
            plt.text(0.5, 0.5, f"Head {h}\n(No Data)", ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title(f"Head {h} (Empty)", fontsize=10)
    
    plt.suptitle(f"Layer {LAYER_IDX} Attention - Cell Set Size {seq_len}", fontsize=14)
    plt.tight_layout()
    fig_path = FIG_DIR / f"no_split_layer{LAYER_IDX}_attention_length{seq_len}.png"
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
                    avg_attn, square=True, cbar=True, cmap='Greens',
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
        fig_path = FIG_DIR / f"no_split_layer{LAYER_IDX}_attention_summary.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved summary attention heatmap → %s", fig_path)

hook_handle.remove()
logger.info("Hook removed; script completed successfully.")
logger.info(f"Generated attention visualizations in {FIG_DIR}")