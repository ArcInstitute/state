"""
PerturbationDataset is used to load perturbation data from h5 files.
Originally, each file was assumed to contain a single cell type.
Now, we remove that assumption so that each file (a plate) may contain
multiple cell types.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import h5py
import numpy as np
import torch
from cell_load.dataset import PerturbationDataset
from cell_load.mapping_strategies import BaseMappingStrategy
from cell_load.utils.data_utils import GlobalH5MetadataCache

logger = logging.getLogger(__name__)


class scGPTPerturbationDataset(PerturbationDataset):
    """
    Dataset class for loading perturbation data from h5 files.

    Each instance handles a single dataset-cell_type combination. Therefore this class is responsible for
    serving a single dataset/cell_type pair. Future improvements will also allow for splitting on
    the perturbation level.

    Currently there are three strategies for mapping basal cells to perturbed cells:
    - "batch": Basal cells are sampled from the same batch as the perturbed cell
    - "random": Basal cells are sampled randomly from the same cell type as the perturbed cell
    - "nearest": Basal cells are sampled from the nearest neighbors of the perturbed cell

    A control cell is always mapped to a perturbed cell within the same dataset and with the same cell type.
    """

    def __init__(
        self,
        name: str,
        h5_path: Union[str, Path],
        mapping_strategy: BaseMappingStrategy,
        pert_onehot_map: Optional[Dict[str, int]] = None,
        cell_type_onehot_map: Optional[Dict[str, int]] = None,
        batch_onehot_map: Optional[Dict[str, int]] = None,
        pert_col: str = "gene",
        cell_type_key: str = "cell_type",
        batch_col: str = "batch",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_uce", "X_pca"] = "X_uce",
        store_raw_expression: bool = False,
        random_state: int = 42,
        should_yield_control_cells: bool = True,
        store_raw_basal: bool = False,
        vocab: Optional[Dict[str, int]] = None,
        hvg_names_uns_key: Optional[str] = None,
        perturbation_type: Literal["chemical", "genetic"] = "chemical",
        **kwargs,
    ):
        """
        Args:
            name: Name of the dataset
            h5_path: Path to the h5 file containing the dataset
            mapping_strategy: Strategy for mapping basal cells to perturbed cells, one of "batch", "random", "nearest"
            pert_onehot_map: Global mapping of perturbation names to one-hot encodings or featurizations
            cell_type_onehot_map: Global mapping of cell type names to one-hot encodings or featurizations
            batch_onehot_map: Global mapping of batch names to one-hot encodings
            pert_col: Column in the h5 file containing perturbation information
            cell_type_key: Column in the h5 file containing cell type information
            batch_col: Column in the h5 file containing batch information
            control_pert: Name of the control perturbation
            embed_key: Key in the h5 file containing the expression data, one of "pert_cell_emb" or "X_uce"
            random_state: Random seed for reproducibility
            pert_tracker: PerturbationTracker instance for tracking valid perturbations
            should_yield_control_cells: If True, control cells will be included in the dataset
        """
        super().__init__(
            name=name,
            h5_path=h5_path,
            mapping_strategy=mapping_strategy,
            pert_onehot_map=pert_onehot_map,
            cell_type_onehot_map=cell_type_onehot_map,
            batch_onehot_map=batch_onehot_map,
            pert_col=pert_col,
            cell_type_key=cell_type_key,
            batch_col=batch_col,
            control_pert=control_pert,
            embed_key=embed_key,
            store_raw_expression=store_raw_expression,
            random_state=random_state,
            should_yield_control_cells=should_yield_control_cells,
            store_raw_basal=store_raw_basal,
            **kwargs,
        )
        self.vocab = vocab
        self.hvg_names_uns_key = hvg_names_uns_key

        assert vocab is not None, "vocab must be provided for scGPTPerturbationDataset"

        self.gene_names = self.get_gene_names()

        self.gene_ids = np.array(
            [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in self.gene_names],
            dtype=int,
        )

        num_invalid_genes = np.sum(self.gene_ids == vocab["<pad>"])
        if num_invalid_genes > 0:
            logger.warning(f"scGPTPerturbationDataset ([{self.name}]) Number of invalid genes: {num_invalid_genes}")

        self.perturbation_type = perturbation_type.lower()

        if self.perturbation_type == "genetic":
            num_genes_X = len(self.gene_names)
            self.pert_flags = {}
            for pert in self.pert_onehot_map.keys():
                self.pert_flags[pert] = np.zeros(num_genes_X)
                if pert in self.gene_names:
                    self.pert_flags[pert][self.gene_names.index(pert)] = 1
                else:
                    logger.warning(
                        f"scGPTPerturbationDataset ([{self.name}]) Perturbation {pert} not found in gene names"
                    )

    def __getitem__(self, idx: int):
        """
        Returns a dictionary with:
            - 'X': the (possibly transformed) expression of the perturbed cell
            - 'basal': the control cell’s expression as chosen by the mapping strategy
            - 'pert': the one-hot encoding (or other featurization) for the perturbation
            - 'pert_name': the perturbation name
            - 'cell_type': the cell type (from the full array)
            - 'gem_group': the batch (as an int or string)

        The index `idx` here is into the filtered set of cells.
        """
        # Map idx to the underlying file index
        underlying_idx = int(self.all_indices[idx])
        split = self._find_split_for_idx(underlying_idx)

        # Get expression from the h5 file.
        # For now, we assume the data is stored in "pert_cell_emb" (could be counts) and/or in obsm (embed_key)
        # (It is up to the downstream code to decide whether to use raw gene expression or a precomputed embedding.)

        pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(self, split, underlying_idx)

        # Get perturbation information using metadata cache
        pert_code = self.metadata_cache.pert_codes[underlying_idx]
        pert_name = self.metadata_cache.pert_categories[pert_code]
        if self.pert_onehot_map is not None:
            # map across all files to a consistent one hot encoding or featurization
            pert_onehot = self.pert_onehot_map[pert_name]
        else:
            pert_onehot = None

        # Get cell type using metadata cache
        cell_type_code = self.metadata_cache.cell_type_codes[underlying_idx]
        cell_type = self.metadata_cache.cell_type_categories[cell_type_code]

        if self.cell_type_onehot_map is not None:
            cell_type_onehot = self.cell_type_onehot_map[cell_type]
        else:
            cell_type_onehot = None

        # Get batch information
        batch_code = self.metadata_cache.batch_codes[underlying_idx]
        batch_name = self.metadata_cache.batch_categories[batch_code]

        if self.batch_onehot_map is not None:
            # map across all files to a consistent one hot encoding or featurization
            batch = self.batch_onehot_map[batch_name]
        else:
            batch = None

        sample = {
            "pert_cell_emb": pert_expr,  # the perturbed cell’s data
            "ctrl_cell_emb": ctrl_expr,  # will be filled in by the mapping strategy
            "pert_emb": pert_onehot,
            "pert_name": pert_name,
            "cell_type": cell_type,
            "cell_type_onehot": cell_type_onehot,
            "batch": batch,
            "batch_name": batch_name,
            "gene_ids": torch.tensor(
                self.gene_ids, dtype=torch.long
            ),  # TODO: should be a more efficient way to do this as this is repeated for every cell
        }

        if "perturbation_type" in self.__dict__ and self.perturbation_type == "genetic":
            sample["pert_flags"] = torch.tensor(self.pert_flags[pert_name], dtype=torch.long)

        # Optionally, if raw gene expression is needed:
        # backwards compatibility for old cktps
        if self.store_raw_expression and self.output_space == "gene":
            sample["pert_cell_counts"] = self.fetch_obsm_expression(underlying_idx, "X_hvg")
        elif self.store_raw_expression and self.output_space == "all":
            sample["pert_cell_counts"] = self.fetch_gene_expression(underlying_idx)
        return sample

    def fetch_obsm_expression(self, idx: int, key: str) -> torch.Tensor:
        row_data = self.h5_file[f"/obsm/{key}"][idx]
        return torch.tensor(row_data, dtype=torch.float32)

    def get_gene_names(self) -> List[str]:
        """
        Get the gene names, which are under adata.var.index, using h5.
        """
        if self.hvg_names_uns_key is not None:  # return hvg names if provided
            hvg_names = self.h5_file[f"uns/{self.hvg_names_uns_key}"][:].astype(str).tolist()
            return hvg_names

        try:
            genes = self.h5_file["var/gene_name"][:].astype(str).tolist()

        # TODO: handle raw exception
        except:
            try:
                categories = self.h5_file["var/gene_name/categories"][:].astype(str)
                codes = self.h5_file["var/gene_name/codes"][:]
                genes = categories[codes].tolist()

            # TODO: handle raw exception
            except:
                genes = self.h5_file["var/_index"][:].astype(str).tolist()

        return genes

    ##############################
    # Static methods
    ##############################
    @staticmethod
    def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
        """
        Custom collate that reshapes data into sequences.
        Safely handles normalization when vectors sum to zero.
        """
        # First do normal collation
        batch_dict = {
            "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
            "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
            "pert_emb": torch.stack([item["pert_emb"] for item in batch]),
            "pert_name": [item["pert_name"] for item in batch],
            "cell_type": [item["cell_type"] for item in batch],
            "cell_type_onehot": torch.stack([item["cell_type_onehot"] for item in batch]),
            "batch": torch.stack([item["batch"] for item in batch]),
            "batch_name": [item["batch_name"] for item in batch],
            "gene_ids": torch.stack([item["gene_ids"] for item in batch]),
        }

        if "pert_flags" in batch[0]:  # only add pert_flags in case of genetic perturbations
            batch_dict["pert_flags"] = torch.stack([item["pert_flags"] for item in batch])

        # If the first sample has "pert_cell_counts", assume the entire batch does
        if "pert_cell_counts" in batch[0]:
            X_hvg = torch.stack([item["pert_cell_counts"] for item in batch])

            # Handle Tahoe dataset (needs log transform)
            if pert_col == "drug" or pert_col == "drugname_drugconc":
                if transform == "log-normalize":
                    library_sizes = X_hvg.sum(
                        dim=1, keepdim=True
                    )  # TODO: Need to replace with library size from all genes
                    # Replace zeros with ones (will result in no change for zero vectors)
                    safe_sizes = torch.where(library_sizes > 0, library_sizes, torch.ones_like(library_sizes) * 10000)
                    X_hvg_norm = X_hvg * 10000 / safe_sizes
                    batch_dict["pert_cell_counts"] = torch.log1p(X_hvg_norm)
                elif transform == "log1p" or transform is True:
                    batch_dict["pert_cell_counts"] = torch.log1p(X_hvg)
                elif int_counts:
                    # this is for log transformed data. let's make it count data
                    batch_dict["pert_cell_counts"] = torch.expm1(X_hvg).round().to(torch.int32)

        # If the first sample has "ctrl_cell_counts", assume the entire batch does
        if "ctrl_cell_counts" in batch[0]:  # either control hvg gene space or 19k gene space
            basal_hvg = torch.stack([item["ctrl_cell_counts"] for item in batch])

            # Handle Tahoe dataset (needs log transform)
            if pert_col == "drug" or pert_col == "drugname_drugconc":
                if transform == "log-normalize":
                    library_sizes = basal_hvg.sum(
                        dim=1, keepdim=True
                    )  # TODO: Need to replace with library size from all genes
                    # Replace zeros with ones (will result in no change for zero vectors)
                    safe_sizes = torch.where(library_sizes > 0, library_sizes, torch.ones_like(library_sizes) * 10000)
                    basal_hvg_norm = basal_hvg * 10000 / safe_sizes
                    batch_dict["ctrl_cell_counts"] = torch.log1p(basal_hvg_norm)
                elif transform == "log1p" or transform is True:
                    batch_dict["ctrl_cell_counts"] = torch.log1p(basal_hvg)
            elif int_counts:
                # this is for log transformed data. let's make it count data
                batch_dict["ctrl_cell_counts"] = torch.expm1(basal_hvg).round().to(torch.int32)
            else:
                batch_dict["ctrl_cell_counts"] = basal_hvg
        # Apply transform if provided
        if transform == "log-normalize":
            X_library_sizes = batch_dict["pert_cell_emb"].sum(dim=1, keepdim=True)
            X_safe_sizes = torch.where(X_library_sizes > 0, X_library_sizes, torch.ones_like(X_library_sizes) * 10000)
            X_norm = batch_dict["pert_cell_emb"] * 10000 / X_safe_sizes
            batch_dict["pert_cell_emb"] = torch.log1p(X_norm)

            # Normalize basal by library size before log transform
            basal_library_sizes = batch_dict["ctrl_cell_emb"].sum(dim=1, keepdim=True)
            basal_safe_sizes = torch.where(
                basal_library_sizes > 0, basal_library_sizes, torch.ones_like(basal_library_sizes) * 10000
            )
            basal_norm = batch_dict["ctrl_cell_emb"] * 10000 / basal_safe_sizes
            batch_dict["ctrl_cell_emb"] = torch.log1p(basal_norm)
        elif transform == "log1p" or transform is True:  # True is for backwards compatibility
            # Original behavior: just log transform without normalization
            batch_dict["pert_cell_emb"] = torch.log1p(batch_dict["pert_cell_emb"])
            batch_dict["ctrl_cell_emb"] = torch.log1p(batch_dict["ctrl_cell_emb"])

        return batch_dict

    def __len__(self) -> int:
        return self.n_cells

    def __getstate__(self):
        """
        Return a dictionary of this dataset's state without the open h5 file object.
        """
        # Copy the object's dict
        state = self.__dict__.copy()
        # Remove the open file object if it exists
        if "h5_file" in state:
            # We'll also store whether it's currently open, so that we can re-open later if needed
            del state["h5_file"]
        return state

    def __setstate__(self, state):
        """
        Reconstruct the dataset after unpickling. Re-open the HDF5 file by path.
        """
        # TODO-Abhi: remove this before release
        self.__dict__.update(state)
        # This ensures that after we unpickle, we have a valid h5_file handle again
        self.h5_file = h5py.File(self.h5_path, "r")
        self.metadata_cache = GlobalH5MetadataCache().get_cache(
            str(self.h5_path),
            self.pert_col,
            self.cell_type_key,
            self.control_pert,
            self.batch_col,
        )
