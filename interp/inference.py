import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
from typing import Optional, List, Dict, Tuple
import os
import sys
import pandas as pd
import anndata as ad
from tqdm.notebook import tqdm
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.getcwd()))

# Now import from inference_module
from inference_module import InferenceModule

path = "/large_storage/ctc/userspace/aadduri/feb28_conc/cv_norm_vci_1.4.2_samp_ctrl_70m_512/fold2/checkpoints/last.ckpt" # tahoe 70M

inference_module = InferenceModule(
    model_path=path,
    model_type="pertsets",
    model_kwargs={
        "cell_set_len": 512,
        "hidden_dim": 696,
        "loss": "energy",
        "transformer_backbone_kwargs": {"num_hidden_layers": 8},
    },
)


def main():
    adata = sc.read_h5ad("/large_storage/ctc/userspace/aadduri/datasets/tahoe_45_ct_vci_1.4.4/plate1.h5")

    # 1. Get all unique perturbations and count cells per perturbation
    pert_counts = adata.obs["gene"].value_counts()
    print(f"Found {len(pert_counts)} unique perturbations")
    print(f"Top 5 perturbations by frequency: {pert_counts.head().to_dict()}")

    # Control cells to sample from
    all_ctrl_cells = adata[adata.obs["gene"] == inference_module.data_module.control_pert].copy()
    print(f"Total control cells available: {all_ctrl_cells.n_obs}")

    # 2. Generate predictions for each perturbation
    all_predictions = []
    np.random.seed(42)  # For reproducibility

    # Create progress bar for all perturbations
    for perturbation in pert_counts.index:
        # Get number of cells for this perturbation
        n_cells = pert_counts[perturbation]

        # Sample this many control cells (with replacement if needed)
        if n_cells <= all_ctrl_cells.n_obs:
            # Sample without replacement if we have enough cells
            sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells, replace=False)
        else:
            # Sample with replacement if we need more cells than available
            sample_indices = np.random.choice(np.arange(all_ctrl_cells.n_obs), size=n_cells, replace=True)

        ctrl_subset = all_ctrl_cells[sample_indices].copy()

        # Set the perturbation
        ctrl_subset.obs["gene"] = perturbation

        # Run inference
        try:
            pred_subset = inference_module.perturb(
                ctrl_subset,
                pert_key="gene",  # replogle
                celltype_key="cell_type",
                # pert_key='drugname_drugconc', # tahoe
                # celltype_key='cell_name'
            )

            # Add metadata to track source perturbation
            pred_subset.obs["original_perturbation"] = perturbation
            pred_subset.obs["is_predicted"] = True

            # Store this batch of predictions
            all_predictions.append(pred_subset)

            print(f"Generated {pred_subset.n_obs} predictions for {perturbation}")
        except Exception as e:
            print(f"Error generating predictions for {perturbation}: {str(e)}")

if __name__ == "__main__":
    main()