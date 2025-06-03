"""Dataset invariant util functions."""

"""Helper functions for loading pretrained gene embeddings."""
import warnings

warnings.filterwarnings("ignore")

import logging

import scanpy as sc
import h5py
import torch
import torch.utils.data as data
import numpy as np
import os
import pandas as pd
import anndata
from pathlib import Path
from utils.singleton import Singleton

from typing import Dict, Tuple
from scanpy import AnnData

log = logging.getLogger(__name__)

class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads."""
    
    def __init__(self, h5_path: str, 
                 pert_col: str = 'drug',
                 cell_type_key: str = 'cell_name',
                 control_pert: str = 'DMSO_TF',
                 batch_col: str = 'sample', # replace with plate number
                ):
        """
        Args:
            h5_path: Path to H5 file
            pert_col: Column name for perturbation data
            cell_type_key: Column name for cell type data
            control_pert: Name of control perturbation
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            # Load categories first
            self.pert_categories = safe_decode_array(f[f"obs/{pert_col}/categories"][:])
            self.cell_type_categories = safe_decode_array(f[f"obs/{cell_type_key}/categories"][:])

            # Read batch information
            try:
                # If batch is stored directly as numbers
                raw_batches = f[f"obs/{batch_col}"][:]
                self.batch_is_categorical = False
                self.batch_categories = raw_batches.astype(str)
            except Exception:
                # Otherwise, if stored as a categorical group
                raw_batches = f[f"obs/{batch_col}/categories"][:]
                self.batch_is_categorical = True
                self.batch_categories = safe_decode_array(raw_batches)
            
            # Then load codes
            self.pert_codes = f[f"obs/{pert_col}/codes"][:].astype(np.int32)
            self.cell_type_codes = f[f"obs/{cell_type_key}/codes"][:].astype(np.int32)
            try:
                self.batch_codes = f[f"obs/{batch_col}/codes"][:].astype(np.int32)
            except:
                self.batch_codes = f[f"obs/{batch_col}"][:].astype(np.int32) # if batch is stored directly as numbers
            
            # Create mask for control perturbations
            self.control_pert_code = np.where(self.pert_categories == control_pert)[0][0]
            self.control_mask = self.pert_codes == self.control_pert_code
            
            self.n_cells = len(self.pert_codes)
    
    def get_batch_names(self, indices: np.ndarray) -> np.ndarray:
        """Get batch names for given indices."""
        return self.batch_categories[indices]

    def get_cell_type_names(self, indices: np.ndarray) -> np.ndarray:
        """Get cell type names for given indices."""
        return self.cell_type_categories[indices]

    def get_pert_names(self, indices: np.ndarray) -> np.ndarray:
        """Get perturbation names for given indices."""
        return self.pert_categories[indices]

class GlobalH5MetadataCache(metaclass=Singleton):
    """
    Singleton class to manage a global cache of H5MetadataCache objects.
    
    Usage:
        from utils.data_utils import GlobalH5MetadataCache
        global_cache = GlobalH5MetadataCache()
        cache = global_cache.get_cache(h5_path, pert_col, cell_type_key, control_pert, batch_col)
    """
    def __init__(self):
        self._cache = {}  # dictionary mapping parameter tuple to H5MetadataCache instance

    def get_cache(self, h5_path: str, 
                  pert_col: str = 'drug', 
                  cell_type_key: str = 'cell_name', 
                  control_pert: str = 'DMSO_TF', 
                  batch_col: str = 'drug') -> H5MetadataCache:
        """
        Retrieve the H5MetadataCache for the given file and parameters.
        If it does not exist, it is created and stored.
        """
        key = h5_path
        if key not in self._cache:
            self._cache[key] = H5MetadataCache(h5_path, pert_col, cell_type_key, control_pert, batch_col)
        return self._cache[key]

# A small helper to decode arrays (so we can reuse it in this module if needed)
def safe_decode_array(arr):
    try:
        return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr])
    except Exception:
        return np.array([str(x) for x in arr])

def merge_adata(adata_list):
    """
    Merge a list of AnnData objects.
    """
    # Ensure variable names are strings and unique for each dataset
    for adata in adata_list:
        adata.var.index = adata.var.index.astype(str)
        adata.var_names_make_unique()

    # Find shared genes across all datasets
    shared_genes = get_shared_genes(adata_list)

    # Filter datasets to keep only shared genes and drop duplicate columns
    filtered_adata_list = []
    for adata in adata_list:
        adata = adata[:, shared_genes]
        adata.obs = adata.obs.loc[:, ~adata.obs.columns.duplicated()]
        filtered_adata_list.append(adata)

    # Concatenate all datasets
    merged_adata = sc.concat(filtered_adata_list, join="inner")

    return merged_adata


def collapse_repeated_gene_names(adata):
    ## TODO this is slow, should be optimized
    df = adata.to_df()
    df = df.T.groupby("gene_name").mean().T  # assumption of a gene_name parameter
    return anndata.AnnData(
        X=df.values,
        obs=adata.obs,
        var=df.columns.tolist(),
        obsm=adata.obsm,
        uns=adata.uns,
    )


def use_embedding(adata, embed_key):
    if embed_key is not None:
        embed_adata = sc.AnnData(adata.obsm[embed_key])
        embed_adata.obs = adata.obs
        embed_adata.var = embed_adata.var.reset_index(names=["gene_name"])
        embed_adata.obsm["X"] = adata.X
        embed_adata.uns["gene_names"] = adata.var.index.values.tolist()
        return embed_adata

    else:
        return adata


def get_shared_genes(adata_list):
    """
    Get the shared genes across a list of AnnData objects.
    """
    shared_genes = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        shared_genes.intersection_update(set(adata.var_names))
    shared_genes = list(shared_genes)
    return shared_genes


def get_shared_perts(adata_list, pert_col="gene"):
    """
    Get the shared perturbations between train and test
    """
    shared_perts = set(adata_list[0].obs[pert_col])
    for adata in adata_list[1:]:
        shared_perts.intersection_update(set(adata.obs[pert_col]))
    shared_perts = list(shared_perts)
    return shared_perts


def merge_uce_adata(adata1, adata2, keep_x=True):
    # faster implementation of merge_adata, making some assumptions about the adata structure
    adata1.var.index = adata1.var.index.astype(str)
    adata2.var.index = adata2.var.index.astype(str)

    adata1.var_names_make_unique()
    adata2.var_names_make_unique()

    shared_genes = set(adata1.var_names).intersection(set(adata2.var_names))
    shared_genes = list(shared_genes)

    adata1 = adata1[:, shared_genes]
    adata2 = adata2[:, shared_genes]

    ## Drop duplicate columns
    unique_cols1 = adata1.obs.columns[~adata1.obs.columns.duplicated()]
    unique_cols2 = adata2.obs.columns[~adata2.obs.columns.duplicated()]

    obs = pd.concat([adata1.obs.copy()[unique_cols1], adata2.obs.copy()[unique_cols2]])
    if keep_x:
        var = adata1.var.copy()
        X = np.concatenate(adata1.X, adata2.X)
    else:
        var = None
        X = np.ones((adata1.shape[0] + adata2.shape[0], 1))

    X_uce = np.concatenate([adata1.obsm["X_uce"], adata2.obsm["X_uce"]])

    adata = anndata.AnnData(obs=obs, var=var, X=X, obsm=dict(X_uce=X_uce))

    return adata


def remove_perts_without_expression(
    adata,
    pert_col="gene",
    pert_sep="+",
    ctrl_group="non-targeting",
    verbose=False,
):
    """
    Remove genetic perturbations for which there is no expression data in the adata.X matrix.
    """
    genes_to_remove = set()
    perts_to_remove = set()
    all_pert_genes = set()
    for perturbation in adata.obs[pert_col].unique():
        if perturbation == ctrl_group:
            continue
        for gene in perturbation.split(pert_sep):
            all_pert_genes.add(gene)
            if gene not in adata.var_names:
                genes_to_remove.add(gene)
                perts_to_remove.add(perturbation)

    n_perts = adata.obs[pert_col].nunique()
    n_pert_genes = len(all_pert_genes)

    if verbose:
        log.info(
            f"\tRemoving {len(genes_to_remove)}/{n_pert_genes} genes or {len(perts_to_remove)}/{n_perts} perturbations from the adata..."
        )

    adata = adata[~adata.obs[pert_col].isin(perts_to_remove)]

    return adata


def normalize_data(data):
    """
    Normalize the data to [-1, 1].
    """

    if isinstance(data, sc.AnnData):
        data_X = data.X.copy()
        min_val, max_val = np.min(data_X, axis=0), np.max(data_X, axis=0)
        data.X = 2 * ((data_X - min_val) / (max_val - min_val)) - 1
        return data, min_val, max_val
    else:
        min_val, max_val = np.min(data, axis=0), np.max(data, axis=0)
        data = 2 * ((data - min_val) / (max_val - min_val)) - 1
        return data, min_val, max_val


def unnormalize_data(array, min_val, max_val):
    array = ((array + 1) / 2) * (max_val - min_val) + min_val
    return array


def generate_onehot_map(keys):
    # maps iterable of keys to dictionary mapping unique keys -> onehot encoding
    unique_keys = sorted(list(set(keys)))
    onehot_map = {
        k: torch.nn.functional.one_hot(torch.Tensor([i]).long(), len(unique_keys)).float().squeeze(0)
        for i, k in enumerate(unique_keys)
    }
    return onehot_map


# Reversing the dictionary with non-unique values
def reverse_dict(input_dict):
    reversed_dict = {}
    for key, value in input_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = [key]
        else:
            reversed_dict[value].append(key)
    return reversed_dict


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


def split_perturbations_by_cell_fraction(
    pert_groups: dict,  # {pert_name: np.ndarray of cell indices}
    val_fraction: float,
    rng: np.random.Generator = None,
):
    """
    Partition the set of perturbations into two subsets: 'val' vs 'train',
    such that the fraction of total cells in 'val' is as close as possible
    to val_fraction, using a greedy approach.

    Returns:
        train_perts: list of perturbation names
        val_perts:   list of perturbation names
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # 1) Compute total # of cells across all perturbations
    total_cells = sum(len(indices) for indices in pert_groups.values())
    target_val_cells = val_fraction * total_cells

    # 2) Create a list of (pert_name, size), then shuffle
    pert_size_list = [(p, len(pert_groups[p])) for p in pert_groups.keys()]
    rng.shuffle(pert_size_list)

    # 3) Greedily add perts to the 'val' subset if it brings us closer to the target
    val_perts = []
    current_val_cells = 0
    for pert, size in pert_size_list:
        new_val_cells = current_val_cells + size

        # Compare how close we'd be to target if we add this perturbation vs. skip it
        diff_if_add = abs(new_val_cells - target_val_cells)
        diff_if_skip = abs(current_val_cells - target_val_cells)

        if diff_if_add < diff_if_skip:
            # Adding this perturbation gets us closer to the target fraction
            val_perts.append(pert)
            current_val_cells = new_val_cells
        # else: skip it; it goes to train

    train_perts = list(set(pert_groups.keys()) - set(val_perts))

    return train_perts, val_perts


class SincleCellDataset(data.Dataset):
    def __init__(
        self,
        expression: torch.tensor,
        # Subset to hv genes, count data! cells x genes
        protein_embeddings: torch.tensor,
        # same order as expression, also subset genes x pe
        labels: None,  # optional, tensor of labels
        covar_vals: None,  # tensor of covar values or cpa.yaml
    ) -> None:
        super(SincleCellDataset, self).__init__()

        # Set expression
        self.expression = expression

        row_sums = self.expression.sum(1)  # UMI Counts
        log_norm_count_adj = torch.log1p(self.expression / (self.expression.sum(1)).unsqueeze(1) * torch.tensor(1000))

        # Set log norm and count adjusted expression
        max_vals, max_idx = torch.max(log_norm_count_adj, dim=0)
        self.expression_mod = log_norm_count_adj / max_vals

        # Calculate dropout likliehoods of each gene
        self.dropout_vec = (self.expression == 0).float().mean(0)  # per gene dropout percentages

        # Set data info
        self.num_cells = self.expression.shape[0]
        self.num_genes = self.expression.shape[1]

        # Set optional label info, including categorical covariate index
        self.covar_vals = covar_vals
        self.labels = labels

        # Set protein embeddings
        self.protein_embeddings = protein_embeddings

        self.item_mode = "expression"
        if self.covar_vals is not None:
            self.item_mode = "expression+covar"

    def __getitem__(self, idx):
        if self.item_mode == "expression":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :]
                else:
                    raise IndexError
            else:
                raise NotImplementedError
        elif self.item_mode == "expression+covar":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :], self.covar_vals[idx]
                else:
                    raise IndexError
            else:
                raise NotImplementedError

    def __len__(self) -> int:
        return self.num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


def anndata_to_sc_dataset(
    adata: sc.AnnData,
    species: str = "human",
    labels: list = [],
    covar_col: str = None,
    hv_genes=None,
    embedding_model="ESM2",
) -> (SincleCellDataset, AnnData):
    # Subset to just genes we have embeddings for
    adata, protein_embeddings = load_gene_embeddings_adata(
        adata=adata, species=[species], embedding_model=embedding_model
    )

    if hv_genes is not None:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hv_genes)  # Expects Count Data

        hv_index = adata.var["highly_variable"]
        adata = adata[:, hv_index]  # Subset to hv genes only

        protein_embeddings = protein_embeddings[species][hv_index]
    else:
        protein_embeddings = protein_embeddings[species]
    expression = data_to_torch_X(adata.X)

    covar_vals = None
    if len(labels) > 0:
        assert covar_col is None or covar_col in labels, (
            "Covar needs to be in labels"
        )  # make sure you keep track of covar column!
        labels = adata.obs.loc[:, labels].values

        if covar_col is not None:
            # we have a categorical label to use as covariate
            covar_vals = torch.tensor(pd.Categorical(adata.obs[covar_col]).codes)
    return (
        SincleCellDataset(
            expression=expression,
            protein_embeddings=protein_embeddings,
            labels=labels,
            covar_vals=covar_vals,
        ),
        adata,
    )


def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    """
    Given a :path: to an h5ad,
    """
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()
    log.info(len(np.unique(pe_row_idxs)))

    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")

    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

    dataset_chroms = gene_chrom["spec_chrom"].cat.codes  # now this is correctely indexed by species and chromosome
    log.info("Max Code:", max(dataset_chroms))
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos


def process_raw_anndata(row, additional_filter, root):
    path = row.path
    name = path.replace(".h5ad", "")
    proc_path = path.replace(".h5ad", "_proc.h5ad")
    if os.path.isfile(path + proc_path):
        log.info(f"{name} already processed. Skipping")
        return None, None, None

    log.info(f"Proccessing {name}")

    species = row.species
    covar_col = row.covar_col

    ad = sc.read(path)
    labels = []
    if "cell_type" in ad.obs.columns:
        labels.append("cell_type")

    if covar_col is np.nan or np.isnan(covar_col):
        covar_col = None
    else:
        labels.append(covar_col)

    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)

    dataset, adata = anndata_to_sc_dataset(ad, species=species, labels=labels, covar_col=covar_col, hv_genes=None)
    adata = adata.copy()

    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)

    num_cells = adata.X.shape[0]
    num_genes = adata.X.shape[1]

    adata_path = proc_path
    adata.write(adata_path)

    arr = data_to_torch_X(adata.X).numpy()

    """
    print(arr.max())  # this is a nice check to make sure it's counts
    filename =  f"{name}_counts.npz"
    shape = arr.shape
    print(name, shape)
    fp = np.memmap(filename, dtype='int64', mode='w+', shape=shape)
    fp[:] = arr[:]
    fp.flush()

    if scp != "":
        subprocess.call(["scp", filename, f"{scp}:{filename}"])
        subprocess.call(["scp", adata_path, f"{scp}:{adata_path}"])
    """

    return adata, num_cells, num_genes


def get_species_to_pe(EMBEDDING_DIR):
    """
    Given an embedding directory, return all embeddings as a dictionary coded by species.
    Note: In the current form, this function is written such that the directory needs all of the following species embeddings.
    """
    EMBEDDING_DIR = Path(EMBEDDING_DIR)

    embeddings_paths = {
        "human": EMBEDDING_DIR / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        "mouse": EMBEDDING_DIR / "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
        "frog": EMBEDDING_DIR / "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
        "zebrafish": EMBEDDING_DIR / "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
        "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": EMBEDDING_DIR / "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
        "macaca_fascicularis": EMBEDDING_DIR
        / "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
        "macaca_mulatta": EMBEDDING_DIR / "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
    }
    extra_species = (
        pd.read_csv("./model_files/new_species_protein_embeddings.csv").set_index("species").to_dict()["path"]
    )
    embeddings_paths.update(extra_species)  # adds new species

    species_to_pe = {species: torch.load(pe_dir) for species, pe_dir in embeddings_paths.items()}

    species_to_pe = {species: {k.upper(): v for k, v in pe.items()} for species, pe in species_to_pe.items()}
    return species_to_pe


def get_spec_chrom_csv(
    path="/dfs/project/cross-species/yanay/code/all_to_chrom_pos.csv",
):
    """
    Get the species to chrom csv file
    """
    gene_to_chrom_pos = pd.read_csv(path)
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(
        gene_to_chrom_pos["species"] + "_" + gene_to_chrom_pos["chromosome"]
    )  # add the spec_chrom list
    return gene_to_chrom_pos


EMBEDDING_DIR = Path("model_files/protein_embeddings")
MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = {
    "ESM2": {
        "human": EMBEDDING_DIR / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        "mouse": EMBEDDING_DIR / "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
        "frog": EMBEDDING_DIR / "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
        "zebrafish": EMBEDDING_DIR / "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
        "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": EMBEDDING_DIR / "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
        "macaca_fascicularis": EMBEDDING_DIR
        / "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
        "macaca_mulatta": EMBEDDING_DIR / "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
    }
}

# extra_species = pd.read_csv("./model_files/new_species_protein_embeddings
# .csv").set_index("species").to_dict()["path"]
# MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"].update(extra_species) # adds
# new species


def load_gene_embeddings_adata(
    adata: AnnData, species: list, embedding_model: str
) -> Tuple[AnnData, Dict[str, torch.FloatTensor]]:
    """Loads gene embeddings for all the species/genes in the provided data.

    :param data: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata

    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim).
    """
    # Get species names
    species_names = species
    species_names_set = set(species_names)

    # Get embedding paths for the model
    species_to_gene_embedding_path = MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH[embedding_model]
    available_species = set(species_to_gene_embedding_path)

    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        raise ValueError(f"The following species do not have gene embeddings: {species_names_set - available_species}")

    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(species_to_gene_embedding_path[species]).items()
        }
        for species in species_names
    }

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(
        *[set(gene_symbol_to_embedding) for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()]
    )
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}

    # Subset data to only use genes with embeddings
    adata = adata[:, adata.var_names.isin(genes_to_use)]

    # Set up dictionary mapping species to gene embedding matrix (num_genes, embedding_dim)
    species_to_gene_embeddings = {
        species_name: torch.stack(
            [species_to_gene_symbol_to_embedding[species_name][gene_symbol.lower()] for gene_symbol in adata.var_names]
        )
        for species_name in species_names
    }

    return adata, species_to_gene_embeddings


def equal_subsampling(adata, obs_key, N_min=None):
    """Subsample cells while retaining same class sizes.

    authors: scPerturb

    Classes are given by obs_key pointing to categorical in adata.obs.
    If N_min is given, downsamples to at least this number instead of the number
    of cells in the smallest class and throws out classes with less than N_min cells.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    obs_key: `str` in adata.obs.keys() (default: `perturbation`)
        Key in adata.obs specifying the groups to consider.
    N_min: `int` or `None` (default: `None`)
        If N_min is given, downsamples to at least this number instead of the number
        of cells in the smallest class and throws out classes with less than N_min cells.

    Returns
    -------
    subdata: :class:`~anndata.AnnData`
        Subsampled version of the original annotated data matrix.
    """

    counts = adata.obs[obs_key].value_counts()
    if N_min is not None:
        groups = counts.index[counts >= N_min]  # ignore groups with less than N_min cells to begin with
    else:
        groups = counts.index
    # We select downsampling target counts by min-max, i.e.
    # the largest N such that every group has at least N cells before downsampling.
    N = np.min(counts)
    N = N if N_min == None else np.max([N_min, N])
    # subsample indices per group
    indices = [
        np.random.choice(adata.obs_names[adata.obs[obs_key] == group], size=N, replace=False) for group in groups
    ]
    selection = np.hstack(np.array(indices))
    return adata[selection].copy()


def drop_low_cellcount_perts(adata, obs_key, N_min=0):
    if N_min == 0:
        return adata

    counts = pd.DataFrame(adata.obs[obs_key].value_counts())
    perts_to_keep = counts[counts["count"] > N_min].index.values
    return adata[adata.obs[obs_key].isin(perts_to_keep)]