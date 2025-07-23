import argparse as ap
import logging

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


def add_arguments_preprocess_infer(parser: ap.ArgumentParser):
    """Add arguments for the preprocess_infer subcommand."""
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output preprocessed AnnData file (.h5ad)",
    )
    parser.add_argument(
        "--control_condition",
        type=str,
        required=True,
        help="Control condition identifier (e.g., \"[('DMSO_TF', 0.0, 'uM')]\")",
    )
    parser.add_argument(
        "--pert_col",
        type=str,
        required=True,
        help="Column name containing perturbation information (e.g., 'drugname_drugconc')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )


def run_tx_preprocess_infer(
    adata_path: str, 
    output_path: str, 
    control_condition: str, 
    pert_col: str, 
    seed: int = 42
):
    """
    Preprocess inference data by replacing perturbed cells with control expression.
    
    This creates a "control template" where all cells have control expression but retain
    their original perturbation annotations. When passed through state_transition inference,
    the model will apply the perturbation effects to simulate the original data.
    
    Args:
        adata_path: Path to input AnnData file
        output_path: Path to save preprocessed AnnData file
        control_condition: Control condition identifier
        pert_col: Column name containing perturbation information
        seed: Random seed for reproducibility
    """
    logger.info(f"Loading AnnData from {adata_path}")
    adata = ad.read_h5ad(adata_path)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Identify control cells
    logger.info(f"Identifying control cells with condition: {control_condition}")
    control_mask = adata.obs[pert_col] == control_condition
    control_indices = np.where(control_mask)[0]
    
    logger.info(f"Found {len(control_indices)} control cells out of {adata.n_obs} total cells")
    
    if len(control_indices) == 0:
        raise ValueError(f"No control cells found with condition '{control_condition}' in column '{pert_col}'")
    
    # Create a copy of the original data to modify
    logger.info("Creating copy of data for modification")
    adata_modified = adata.copy()
    
    # Get all unique perturbations (non-control)
    if hasattr(adata.obs[pert_col], 'cat'):
        # Categorical column
        unique_perturbations = adata.obs[pert_col].cat.categories
    else:
        # Regular column
        unique_perturbations = adata.obs[pert_col].unique()
    
    non_control_perturbations = [p for p in unique_perturbations if p != control_condition]
    
    logger.info(f"Processing {len(non_control_perturbations)} non-control perturbations...")
    logger.info("Replacing perturbed cell expression with randomly sampled control cells")
    logger.info("This creates a 'control template' where state_transition inference will apply perturbation effects")
    
    total_replaced_cells = 0
    
    # For each non-control perturbation, replace with randomly sampled control cells
    for i, perturbation in enumerate(non_control_perturbations):
        # Find cells with this perturbation
        pert_mask = adata.obs[pert_col] == perturbation
        pert_indices = np.where(pert_mask)[0]
        n_pert_cells = len(pert_indices)
        
        if n_pert_cells > 0:
            # Sample n_pert_cells control cells randomly (with replacement)
            sampled_control_indices = np.random.choice(control_indices, size=n_pert_cells, replace=True)
            
            # Replace the expression data
            adata_modified.X[pert_indices] = adata.X[sampled_control_indices]
            total_replaced_cells += n_pert_cells
            
            # Print progress for every 50 perturbations
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(non_control_perturbations)} perturbations")
    
    logger.info(f"Replacement complete! Replaced expression in {total_replaced_cells} cells")
    logger.info(f"Control cells ({len(control_indices)}) retain their original expression")
    
    # Log the transformation summary
    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY:")
    logger.info(f"  - Input: {adata.n_obs} cells, {adata.n_vars} genes")
    logger.info(f"  - Control condition: {control_condition}")
    logger.info(f"  - Control cells: {len(control_indices)} (unchanged)")
    logger.info(f"  - Perturbed cells: {total_replaced_cells} (replaced with control expression)")
    logger.info(f"  - Perturbations processed: {len(non_control_perturbations)}")
    logger.info("")
    logger.info("USAGE:")
    logger.info("  The output file contains cells with control expression but original")
    logger.info("  perturbation annotations. When passed through state_transition inference,")
    logger.info("  the model will apply perturbation effects to simulate the original data.")
    logger.info("  Compare: state_transition(output) ≈ original_input")
    logger.info("=" * 60)
    
    logger.info(f"Saving preprocessed data to {output_path}")
    adata_modified.write_h5ad(output_path)
    
    logger.info("Preprocessing complete!")