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
import torch.nn.functional as F

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
    "/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered_cs32_pretrained/hepg2"
)
DATA_PATH = Path(
    "/large_storage/ctc/ML/state_sets/replogle/processed.h5"
)
CELL_SET_LEN = 512   
CONTROL_SAMPLES = 50 
FIG_DIR = Path(__file__).resolve().parent / "figures" / "layer" 
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load model directly from checkpoint
#checkpoint_path = MODEL_DIR / "step=148000.ckpt"
#checkpoint_path = MODEL_DIR / "checkpoints" / "step=124000.ckpt" # "step=step=104000-val_loss=val_loss=0.0472.ckpt"
checkpoint_path = MODEL_DIR / "checkpoints" / "step=44000.ckpt"
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

# Updated for GPT-2 architecture using .layers instead of .h
for layer in model.transformer_backbone.layers:
    if hasattr(layer.self_attn, 'output_attentions'):
        layer.self_attn.output_attentions = True

print(f"Model config output_attentions: {getattr(model.transformer_backbone.config, 'output_attentions', 'Not found')}")
print(f"Number of transformer layers: {len(model.transformer_backbone.layers)}")

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

# Remove attention-specific config and hooks
# Instead, set up hooks for representation change

NUM_LAYERS = len(model.transformer_backbone.layers)

layer_change_magnitudes = [[] for _ in range(NUM_LAYERS)]  # List of lists, one per layer

# Hook to compute L2 norm of representation change per layer

def make_rep_change_hook(layer_idx):
    def hook(module, input, output):
        print(f"[DEBUG] Layer {layer_idx} hook called.")
        print(f"[DEBUG] input type: {type(input)}, len: {len(input)}")
        print(f"[DEBUG] output type: {type(output)}")
        if isinstance(input, tuple) and len(input) > 0:
            inp = input[0]
            print(f"[DEBUG] inp type: {type(inp)}, shape: {getattr(inp, 'shape', None)}")
        else:
            inp = input
            print(f"[DEBUG] inp fallback type: {type(inp)}, shape: {getattr(inp, 'shape', None)}")
        # If output is a tuple, take the first element (hidden states)
        if isinstance(output, tuple):
            out = output[0]
            print(f"[DEBUG] out (from tuple) type: {type(out)}, shape: {getattr(out, 'shape', None)}")
        else:
            out = output
            print(f"[DEBUG] out type: {type(out)}, shape: {getattr(out, 'shape', None)}")
        # Only proceed if both are tensors
        if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
            diff = out - inp
            l2 = torch.norm(diff, dim=-1)  # [B, S]
            mean_l2 = l2.mean().item()
            layer_change_magnitudes[layer_idx].append(mean_l2)
        else:
            print(f"[DEBUG] Skipping L2 calculation for layer {layer_idx} due to non-tensor input/output.")
    return hook

# Register hooks on all layers
handles = []
for i, layer in enumerate(model.transformer_backbone.layers):
    handles.append(layer.register_forward_hook(make_rep_change_hook(i)))

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

# Remove hooks
for h in handles:
    h.remove()

# Aggregate per-layer representation change
mean_change_per_layer = [np.mean(mags) if mags else 0.0 for mags in layer_change_magnitudes]

plt.figure(figsize=(10, 6))
layers = list(range(4))
contributions = mean_change_per_layer
sns.barplot(x=layers, y=contributions, palette="viridis")
plt.title(f'Average Magnitude of Representation Change', fontsize=14)
plt.xlabel('Transformer Layer', fontsize=12)
plt.ylabel('Change in Representation', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
layer_analysis_dir = os.path.join(FIG_DIR, 'layer')
os.makedirs(layer_analysis_dir, exist_ok=True)
plt.savefig(os.path.join(layer_analysis_dir, f'layer_contributions.png'), dpi=300)
plt.close()

# # Plot as bar chart with viridis colormap
# plt.figure(figsize=(10, 6))
# cmap = plt.get_cmap('viridis')
# colors = cmap(np.linspace(0, 1, NUM_LAYERS))
# bars = plt.bar(range(NUM_LAYERS), mean_change_per_layer, color=colors, edgecolor='black')

# plt.xlabel('Layer Index')
# plt.ylabel('Mean L2 Change')
# plt.title('Magnitude of Representation Change by Layer')
# plt.grid(axis='y', linestyle='--', alpha=0.6)

# # Add colorbar for viridis mapping
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=NUM_LAYERS-1))
# sm.set_array([])
# ax = plt.gca()
# cbar = plt.colorbar(sm, ticks=range(0, NUM_LAYERS, max(1, NUM_LAYERS // 8)), ax=ax)
# cbar.set_label('Layer Index (color)')

# fig_path = FIG_DIR / f"representation_change_by_layer.png"
# plt.tight_layout()
# plt.savefig(fig_path, dpi=300, bbox_inches='tight')
# plt.close()
# logger.info(f"Saved representation change plot → {fig_path}")

# logger.info("Script completed successfully.")