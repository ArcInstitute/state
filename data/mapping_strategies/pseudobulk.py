from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import torch
from .mapping_strategies import BaseMappingStrategy

logger = logging.getLogger(__name__)


class PseudoBulkMappingStrategy(BaseMappingStrategy):
    """
    Maps perturbed cells to control cells by:
    1) Randomly selecting a control cell for each perturbed cell
    2) Computing a local average (pseudobulk) over k-nearest neighbors within each group

    For controls: averages over k nearest control cells
    For perturbed: averages over k nearest cells with the same perturbation

    The number of neighbors k is determined as a fraction f of the group size:
    - For controls: k_ctrl = max(1, int(f * n_control_cells))
    - For each perturbation: k_pert = max(1, int(f * n_cells_with_pert))
    """

    def __init__(
        self,
        name="pseudobulk",
        random_state=42,
        n_basal_samples=1,
        neighborhood_fraction=0.1,
        **kwargs,
    ):
        super().__init__(name, random_state, n_basal_samples)
        if not (0 <= neighborhood_fraction <= 1):
            raise ValueError("neighborhood_fraction must be in [0,1]")
        self.neighborhood_fraction = neighborhood_fraction
        logger.info(f"Initialized PseudoBulkMappingStrategy with neighborhood_fraction={neighborhood_fraction}")

        # For each split, store:
        self.split_control_indices = {}  # control cell pool
        self.split_control_nn = {}  # NN model for controls
        self.split_pert_indices = {}  # Dict[pert_name -> indices]
        self.split_pert_nn = {}  # Dict[pert_name -> NN model]
        self.stage = "train"  # default to train, which pseudobulks both control and perturbed

        # Cache for precomputed pseudobulks
        self.split_pert_pseudobulks = {}  # Dict[split -> Dict[pert_name -> Dict[idx -> pseudobulk]]]
        self.split_control_pseudobulks = {}  # Dict[split -> Dict[idx -> pseudobulk]]

    def register_split_indices(self, dataset, split, perturbed_indices, control_indices):
        """Same as before but now compute & cache pseudobulks after building NN models."""
        if len(perturbed_indices) == 0 or len(control_indices) == 0:
            return

        # Store control indices
        self.split_control_indices[split] = control_indices

        # Get expression matrix
        if dataset.embed_key:
            X = dataset.h5_file[f"obsm/{dataset.embed_key}"][:]
        else:
            X = np.vstack([dataset.fetch_gene_expression(idx).cpu().float().numpy() for idx in range(len(dataset))])

        # Initialize storage for this split
        self.split_pert_indices[split] = {}
        self.split_pert_nn[split] = {}
        self.split_pert_pseudobulks[split] = {}
        self.split_control_pseudobulks[split] = {}

        # 1. Build control NN model and cache control pseudobulks
        k_ctrl = max(1, int(self.neighborhood_fraction * len(control_indices)))
        control_nn = NearestNeighbors(n_neighbors=k_ctrl)
        control_nn.fit(X[control_indices])
        self.split_control_nn[split] = {"model": control_nn, "k": k_ctrl, "X": X[control_indices]}

        # Cache control pseudobulks
        logger.info(f"[{dataset.name}] Split {split}: Computing control pseudobulks...")
        all_control_neighbors = control_nn.kneighbors(X[control_indices], return_distance=False)
        for i, cell_idx in enumerate(control_indices):
            # neighbor indices (these are row indices relative to X[control_indices])
            ctrl_neighbors = all_control_neighbors[i]
            # map them back to global dataset indices
            ctrl_neighbor_indices = control_indices[ctrl_neighbors]

            # Average expressions
            if dataset.embed_key:
                expr_list = [dataset.fetch_obsm_expression(n_idx, dataset.embed_key) for n_idx in ctrl_neighbor_indices]
            else:
                expr_list = [dataset.fetch_gene_expression(n_idx) for n_idx in ctrl_neighbor_indices]

            self.split_control_pseudobulks[split][cell_idx] = torch.stack(expr_list).mean(0)

        # 2. Group perturbed cells by perturbation, build NN models and cache pseudobulks
        pert_codes = dataset.h5_file[f"obs/{dataset.pert_col}/codes"][perturbed_indices]
        pert_names = dataset.pert_categories[pert_codes]

        logger.info(f"[{dataset.name}] Split {split}: Computing perturbation pseudobulks...")
        for pert_name in np.unique(pert_names):
            # Get indices for this perturbation
            mask = pert_names == pert_name
            pert_indices = perturbed_indices[mask]

            if len(pert_indices) == 0:
                continue

            # Store indices
            self.split_pert_indices[split][pert_name] = pert_indices

            # Build NN model for this perturbation
            k_pert = max(1, int(self.neighborhood_fraction * len(pert_indices)))
            pert_nn = NearestNeighbors(n_neighbors=k_pert)
            pert_nn.fit(X[pert_indices])

            self.split_pert_nn[split][pert_name] = {
                "model": pert_nn,
                "k": k_pert,
                "X": X[pert_indices],
            }

            # Cache pseudobulks for this perturbation
            self.split_pert_pseudobulks[split][pert_name] = {}
            all_pert_neighbors = pert_nn.kneighbors(X[pert_indices], return_distance=False)
            for idx, cell_idx in enumerate(pert_indices):
                pert_neighbors = pert_neighbors = all_pert_neighbors[idx]

                # Map to dataset indices
                pert_neighbor_indices = pert_indices[pert_neighbors]

                # Average expressions
                if dataset.embed_key:
                    expr_list = [dataset.fetch_obsm_expression(i, dataset.embed_key) for i in pert_neighbor_indices]
                else:
                    expr_list = [dataset.fetch_gene_expression(i) for i in pert_neighbor_indices]
                self.split_pert_pseudobulks[split][pert_name][cell_idx] = torch.stack(expr_list).mean(0)

        logger.info(
            f"[{dataset.name}] Split {split}: Cached pseudobulks for {len(control_indices)} controls "
            f"and {len(self.split_pert_nn[split])} perturbations"
        )

    def get_control_indices(self, dataset: "PerturbationDataset", split: str, perturbed_idx: int) -> np.ndarray:
        """Return n_basal_samples random control indices."""
        if split not in self.split_control_indices:
            raise ValueError(f"Split {split} not registered")

        ctrl_indices = self.split_control_indices[split]
        return self.rng.choice(ctrl_indices, size=self.n_basal_samples, replace=True)

    def get_mapped_expressions(self, dataset, split, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Now just lookup precomputed pseudobulks."""
        if split not in self.split_control_pseudobulks:
            raise ValueError(f"Split {split} not registered")

        is_control = dataset.control_mask[idx]

        if is_control:
            # Get precomputed control pseudobulk
            ctrl_expr = self.split_control_pseudobulks[split][idx]
            return ctrl_expr, ctrl_expr
        else:
            # Get perturbation name
            pert_code = dataset.h5_file[f"obs/{dataset.pert_col}/codes"][idx]
            pert_name = dataset.pert_categories[pert_code]

            # Get precomputed pseudobulk for this perturbation
            # TODO: if self.stage logic, make sure that set_inference_strategy correctly updates this to 'inference'
            if self.stage == "train":
                pert_expr = self.split_pert_pseudobulks[split][pert_name][idx]
            else:
                # just use the raw, true expression
                if dataset.embed_key:
                    pert_expr = dataset.fetch_obsm_expression(idx, dataset.embed_key)
                else:
                    pert_expr = dataset.fetch_gene_expression(idx)

            # Get random control & its precomputed pseudobulk
            ctrl_indices = self.get_control_indices(dataset, split, idx)
            control_expr_list = [self.split_control_pseudobulks[split][ctrl_idx] for ctrl_idx in ctrl_indices]
            control_expr = torch.stack(control_expr_list)[0]

            return pert_expr, control_expr
