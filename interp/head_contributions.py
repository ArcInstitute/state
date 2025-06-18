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
    "/large_storage/ctc/userspace/aadduri/preprint/replogle_llama_21712320_filtered_cs32_pretrained/hepg2"
)
DATA_PATH = Path(
    "/large_storage/ctc/ML/state_sets/replogle/processed.h5"
)
CELL_SET_LEN = 512   
CONTROL_SAMPLES = 50 
FIG_DIR = Path(__file__).resolve().parent / "figures" / "head_contributions"
FIG_DIR.mkdir(parents=True, exist_ok=True) 

# Load model directly from checkpoint
checkpoint_path = MODEL_DIR / "checkpoints" / "step=44000.ckpt"
#checkpoint_path = MODEL_DIR / "step=148000.ckpt"
# checkpoint_path = MODEL_DIR / "checkpoints" / "last.ckpt" # "step=step=104000-val_loss=val_loss=0.0472.ckpt"

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

# Get model dimensions
NUM_LAYERS = model.transformer_backbone.config.num_hidden_layers
NUM_HEADS = model.transformer_backbone.config.num_attention_heads
HIDDEN_DIM = model.transformer_backbone.config.hidden_size
HEAD_DIM = HIDDEN_DIM // NUM_HEADS

logger.info(
    "Loaded PertSets model with GPT-2 backbone: %s heads × %s layers",
    NUM_HEADS,
    NUM_LAYERS,
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

# Storage for head contributions
head_contributions: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
residual_streams: List[torch.Tensor] = []
total_sequences: int = 0

def create_head_contribution_hook(layer_idx: int):
    """Create a hook to capture head contributions for a specific layer."""
    def hook(module: torch.nn.Module, input_tensors: Tuple[torch.Tensor, ...], output):
        global head_contributions, total_sequences
        
        # Handle tuple output (common in attention modules)
        if isinstance(output, tuple):
            # First element is usually the actual output tensor
            actual_output = output[0]
            print(f"Layer {layer_idx}: Received tuple output, using first element with shape {actual_output.shape}")
        else:
            actual_output = output
            print(f"Layer {layer_idx}: Received tensor output with shape {actual_output.shape}")
        
        # Get the attention module's output
        if hasattr(actual_output, 'shape') and len(actual_output.shape) == 3:  # [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = actual_output.shape
            
            # Reshape to separate heads: [batch, seq_len, num_heads, head_dim]
            head_output = actual_output.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
            
            # Store each head's contribution for each sequence in the batch
            for b in range(batch_size):
                for h in range(NUM_HEADS):
                    # Get this head's contribution: [seq_len, head_dim]
                    head_contrib = head_output[b, :, h, :].detach().cpu()
                    head_contributions[layer_idx][h].append(head_contrib)
            
            if layer_idx == 0:  # Only count sequences once
                global total_sequences
                total_sequences += batch_size
                
        else:
            print(f"Layer {layer_idx}: Unexpected output shape: {actual_output.shape if hasattr(actual_output, 'shape') else type(actual_output)}")
    
    return hook

# Register hooks for all layers
hook_handles = []
for layer_idx in range(NUM_LAYERS):
    # Hook the output of the attention mechanism (after projection)
    attention_module = model.transformer_backbone.layers[layer_idx].self_attn
    
    # For GPT-2 style models, we want to hook the output projection
    if hasattr(attention_module, 'out_proj'):
        hook_handle = attention_module.out_proj.register_forward_hook(
            create_head_contribution_hook(layer_idx)
        )
    else:
        # Fallback to hooking the entire attention module
        hook_handle = attention_module.register_forward_hook(
            create_head_contribution_hook(layer_idx)
        )
    
    hook_handles.append(hook_handle)
    logger.info(f"Registered head contribution hook on layer {layer_idx}")

# Direct model forward pass with cells
# Get available cell types and select the most abundant one
cell_type_counts = adata_full.obs["cell_type"].value_counts()
logger.info("Available cell types: %s", list(cell_type_counts.index))

# Select the most abundant cell type
celltype1 = cell_type_counts.index[0]
logger.info(f"Selected cell type: {celltype1} ({cell_type_counts[celltype1]} available)")

# Get control cells for this cell type
cells_type1 = adata_full[(adata_full.obs["gene"] == control_pert) & 
                        (adata_full.obs["cell_type"] == celltype1)].copy()

logger.info(f"Available cells - {celltype1}: {cells_type1.n_obs}")

# Use the model's actual cell_sentence_len to avoid position embedding errors
cell_sentence_len = model.cell_sentence_len  # Use model's trained sequence length
logger.info(f"Using model's cell_sentence_len: {cell_sentence_len}")

# Use all cells for the single cell type
n_cells = cell_sentence_len
total_cells = cell_sentence_len  # Use model's trained length

if cells_type1.n_obs >= n_cells:
    # Sample cells
    idx1 = np.random.choice(cells_type1.n_obs, size=n_cells, replace=False)
    
    sampled_type1 = cells_type1[idx1].copy()
    
    # Use single cell type batch
    combined_batch = sampled_type1
    
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
    
    # Single forward pass
    with torch.no_grad():
        batch_pred = model.forward(batch, padded=False)
    
    logger.info("Forward pass completed successfully")
    
else:
    logger.error(f"Insufficient cells: {celltype1}: {cells_type1.n_obs}. Need at least {n_cells}.")

logger.info("Finished inference – accumulated %d sequences", total_sequences)
assert total_sequences > 0, "Hooks did not run – check model architecture"

# Calculate head contribution statistics
logger.info("Calculating head contribution statistics...")

contribution_stats = {}
for layer_idx in range(NUM_LAYERS):
    contribution_stats[layer_idx] = {}
    
    for head_idx in range(NUM_HEADS):
        if len(head_contributions[layer_idx][head_idx]) > 0:
            # Stack all contributions for this head
            stacked_contribs = torch.stack(head_contributions[layer_idx][head_idx])
            
            # Calculate statistics
            mean_contrib = stacked_contribs.mean(dim=0)  # Average across sequences
            std_contrib = stacked_contribs.std(dim=0)    # Standard deviation
            norm_contrib = torch.norm(stacked_contribs, dim=-1).mean()  # Average L2 norm
            
            contribution_stats[layer_idx][head_idx] = {
                'mean': mean_contrib,
                'std': std_contrib,
                'norm': norm_contrib.item(),
                'shape': mean_contrib.shape,
                'n_sequences': len(head_contributions[layer_idx][head_idx])
            }
            
            logger.info(f"Layer {layer_idx}, Head {head_idx}: norm={norm_contrib:.4f}, shape={mean_contrib.shape}, n={len(head_contributions[layer_idx][head_idx])}")

# Save contribution statistics
output_file = Path(__file__).resolve().parent / "head_contributions.pkl"
import pickle
with open(output_file, 'wb') as f:
    pickle.dump(contribution_stats, f)

logger.info(f"Saved head contribution statistics to {output_file}")

# Create summary of contribution norms
logger.info("\n=== Head Contribution Summary ===")
for layer_idx in range(NUM_LAYERS):
    layer_norms = []
    for head_idx in range(NUM_HEADS):
        if head_idx in contribution_stats[layer_idx]:
            norm = contribution_stats[layer_idx][head_idx]['norm']
            layer_norms.append(f"H{head_idx}:{norm:.3f}")
    
    logger.info(f"Layer {layer_idx}: {', '.join(layer_norms)}")

# Create visualizations
logger.info("Creating head contribution visualizations...")

# Create a single plot with all heads from all layers on x-axis
all_contributions = []
labels = []

# Flatten contributions: iterate through layers, then heads within each layer
for layer_idx in range(NUM_LAYERS):
    for head_idx in range(NUM_HEADS):
        if head_idx in contribution_stats[layer_idx]:
            all_contributions.append(contribution_stats[layer_idx][head_idx]['norm'])
            labels.append(f'L{layer_idx}H{head_idx}')
        else:
            all_contributions.append(0.0)
            labels.append(f'L{layer_idx}H{head_idx}')

total_heads = len(all_contributions)
logger.info(f"Plotting {total_heads} individual head contributions (L×H = {NUM_LAYERS}×{NUM_HEADS})")

# Create the main plot with all head contributions
plt.figure(figsize=(20, 8))
x_positions = range(total_heads)
bars = plt.bar(x_positions, all_contributions, color=plt.cm.viridis(np.linspace(0, 1, total_heads)))

plt.xlabel('Head (Layer × Head)', fontsize=14)
plt.ylabel('Contribution Norm', fontsize=14)
plt.title(f'Individual Head Contributions to Residual Stream\n({NUM_LAYERS} layers × {NUM_HEADS} heads = {total_heads} total)', fontsize=16)

# Set x-axis labels - show every 4th label to avoid overcrowding
label_step = max(1, total_heads // 20)  # Show ~20 labels max
plt.xticks(x_positions[::label_step], labels[::label_step], rotation=45, ha='right')

# Add vertical lines to separate layers
for layer_idx in range(1, NUM_LAYERS):
    plt.axvline(x=layer_idx * NUM_HEADS - 0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig_path = FIG_DIR / "all_head_contributions.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Saved all head contributions plot → {fig_path}")

# Create a line plot version for easier reading
plt.figure(figsize=(20, 8))
plt.plot(x_positions, all_contributions, marker='o', color='darkblue', linewidth=2, markersize=3)

# Color points using viridis
scatter = plt.scatter(x_positions, all_contributions, c=x_positions, cmap='viridis', s=50, zorder=5)
plt.colorbar(scatter, label='Position (Layer × Head order)')

plt.xlabel('Head (Layer × Head)', fontsize=14)
plt.ylabel('Contribution Norm', fontsize=14)
plt.title(f'Individual Head Contributions to Residual Stream (Line Plot)\n({NUM_LAYERS} layers × {NUM_HEADS} heads = {total_heads} total)', fontsize=16)

# Set x-axis labels
plt.xticks(x_positions[::label_step], labels[::label_step], rotation=45, ha='right')

# Add vertical lines to separate layers
for layer_idx in range(1, NUM_LAYERS):
    plt.axvline(x=layer_idx * NUM_HEADS - 0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    # Add layer labels
    plt.text(layer_idx * NUM_HEADS - NUM_HEADS/2, max(all_contributions) * 0.95, f'Layer {layer_idx-1}', 
             ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Add final layer label
plt.text((NUM_LAYERS-1) * NUM_HEADS + NUM_HEADS/2, max(all_contributions) * 0.95, f'Layer {NUM_LAYERS-1}', 
         ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = FIG_DIR / "all_head_contributions_line.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"Saved all head contributions line plot → {fig_path}")

# Remove all hooks
for handle in hook_handles:
    handle.remove()

logger.info("All hooks removed; script completed successfully.")
logger.info(f"Head contribution analysis complete. Results saved to {output_file}")
logger.info(f"Visualizations saved to {FIG_DIR}")