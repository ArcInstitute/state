import os
import pandas as pd
import numpy as np
import torch

from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
import uuid

import logging
import requests
import time

# Constants for gene mapping
GENE_NAME_ENSEMPLE_MAP = {"GATD3A": "ENSMUSG00000053329", "GATD3B": "ENSG00000160221"}


def convert_gene_symbols_to_ensembl_rest(gene_symbols, species="human"):
    server = "https://grch37.rest.ensembl.org"

    # Map species to its scientific name
    species_map = {"human": "homo_sapiens", "mouse": "mus_musculus", "rat": "rattus_norvegicus"}

    species_name = species_map.get(species.lower(), species)
    gene_to_ensembl = {}

    for symbol in gene_symbols:
        # Construct the URL for the API request
        ext = f"/lookup/symbol/{species_name}/{symbol}?"

        # Make the request
        r = requests.get(server + ext, headers={"Content-Type": "application/json"})

        # Check if the request was successful
        if r.status_code != 200:
            print(f"Failed to retrieve data for {symbol}: {r.status_code}")
            continue

        # Parse the JSON response
        decoded = r.json()

        # Extract the Ensembl ID
        if "id" in decoded:
            gene_to_ensembl[symbol] = decoded["id"]

        # Sleep briefly to avoid overloading the server
        time.sleep(0.1)

    return gene_to_ensembl


def convert_symbols_to_ensembl(adata):
    import mygene

    gene_symbols = adata.var_names.tolist()

    mg = mygene.MyGeneInfo()
    results = mg.querymany(gene_symbols, scopes="symbol", fields="ensembl.gene", species="human")

    symbol_to_ensembl = {}
    for result in results:
        if "ensembl" in result and not result.get("notfound", False):
            if isinstance(result["ensembl"], list):
                symbol_to_ensembl[result["query"]] = result["ensembl"][0]["gene"]
            else:
                symbol_to_ensembl[result["query"]] = result["ensembl"]["gene"]

    for symbol in gene_symbols:
        if symbol_to_ensembl.get(symbol) is None:
            sym_results = convert_gene_symbols_to_ensembl_rest([symbol])
            if len(sym_results) > 0:
                symbol_to_ensembl[symbol] = sym_results[symbol]
                logging.info(f"Converted {symbol} to {symbol_to_ensembl[symbol]} using REST API")

    logging.info("Done...")
    for symbol in gene_symbols:
        if symbol_to_ensembl.get(symbol) is None:
            logging.info(f"{symbol} -> {symbol_to_ensembl.get(symbol, np.nan)}")
            if symbol in GENE_NAME_ENSEMPLE_MAP:
                symbol_to_ensembl[symbol] = GENE_NAME_ENSEMPLE_MAP[symbol]

    # Add the remaining or errored ones manually
    symbol_to_ensembl["PBK"] = "ENSG00000168078"
    return [symbol_to_ensembl.get(symbol, np.nan) for symbol in gene_symbols]


def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


def get_embedding_cfg(cfg):
    return cfg["embeddings"][cfg["embeddings"]["current"]]


def get_dataset_cfg(cfg):
    return cfg["dataset"][cfg["dataset"]["current"]]


def get_precision_config(device_type="cuda"):
    """
    Single source of truth for precision configuration.

    Args:
        device_type: Device type ('cuda' or 'cpu')

    Returns:
        torch.dtype: The precision to use for autocast and model operations.
                    Returns torch.bfloat16 for CUDA, torch.float32 for CPU.
    """
    if device_type == "cuda":
        return torch.bfloat16
    else:
        return torch.float32


def compute_pearson_delta(pred, true, ctrl, ctrl_true):
    """
    pred, true, ctrl, ctrl_true are numpy arrays of shape [n_cells, n_genes],
    or you can pass means if you prefer.

    We'll compute the correlation of (pred.mean - ctrl.mean) with (true.mean - ctrl_true.mean).
    """
    return pearsonr(pred.mean(axis=0) - ctrl.mean(axis=0), true.mean(axis=0) - ctrl_true.mean(axis=0))[0]


def compute_perturbation_ranking_score(adata_pred, adata_real, pert_col="gene", ctrl_pert="non-targeting"):
    """
    1) compute mean perturbation effect for each perturbation in pred and real
    2) measure how well the real perturbation matches the predicted one by rank
    returns the mean normalized rank of the true perturbation
    """
    # Step 1: compute mean effects
    mean_real_effect = _compute_mean_perturbation_effect(adata_real, pert_col, ctrl_pert)
    mean_pred_effect = _compute_mean_perturbation_effect(adata_pred, pert_col, ctrl_pert)
    all_perts = mean_real_effect.index.values

    ranks = []
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values.reshape(1, -1)
        pred_effects = mean_pred_effect.values

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(real_effect, pred_effects).flatten()
        # where is the true row? (the same index in `all_perts`)
        true_pert_index = np.where(all_perts == pert)[0][0]

        # Sort by descending similarity
        sorted_indices = np.argsort(similarities)[::-1]
        # rank of the correct one:
        rank_of_true_pert = np.where(sorted_indices == true_pert_index)[0][0]
        ranks.append(rank_of_true_pert)

    mean_rank = np.mean(ranks) / len(all_perts)
    return mean_rank


def _compute_mean_perturbation_effect(adata, pert_col="gene", ctrl_pert="non-targeting"):
    """
    Helper to compute the mean effect (abs difference from control) for each perturbation.
    Actually we do the absolute difference from control row.
    """
    # shape: adata.X is (#cells, #genes)
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    df = pd.DataFrame(X)
    df[pert_col] = adata.obs[pert_col].values
    mean_df = df.groupby(pert_col).mean(numeric_only=True)
    # difference from control
    return np.abs(mean_df - mean_df.loc[ctrl_pert])


def get_latest_checkpoint(cfg):
    run_name = "exp_{0}_layers_{1}_dmodel_{2}_samples_{3}_max_lr_{4}_op_dim_{5}".format(
        cfg.experiment.name,
        cfg.model.nlayers,
        cfg.model.emsize,
        cfg.dataset.pad_length,
        cfg.optimizer.max_lr,
        cfg.model.output_dim,
    )

    if cfg.experiment.checkpoint.path is None:
        return run_name, None
    chk_dir = os.path.join(cfg.experiment.checkpoint.path, cfg.experiment.name)
    chk = os.path.join(chk_dir, "last.ckpt")
    # chk = os.path.join(chk_dir, 'exp_rda_mmd_counts_1024_layers_4_dmodel_512_samples_1024_max_lr_0.00024_op_dim_512-epoch=1-step=581000.ckpt')
    if not os.path.exists(chk) or len(chk) == 0:
        chk = None

    return run_name, chk


def compute_gene_overlap_cross_pert(DE_pred, DE_true, control_pert="non-targeting", k=50):
    all_overlaps = {}
    for c_gene in DE_pred.index:
        if c_gene == control_pert:
            continue
        all_overlaps[c_gene] = len(set(DE_true.loc[c_gene].values).intersection(set(DE_pred.loc[c_gene].values))) / k

    return all_overlaps


def parse_chk_info(chk):
    chk_filename = Path(chk)
    epoch = chk_filename.stem.split("_")[-1].split("-")[1].split("=")[1]
    steps = chk_filename.stem.split("_")[-1].split("-")[2].split("=")[1]

    return int(epoch), int(steps)


def get_shapes_dict(dataset_path, filter_by_species=None):
    datasets_df = pd.read_csv(dataset_path)

    if filter_by_species is not None:
        datasets_df = datasets_df[datasets_df["species"] == filter_by_species]

    sorted_dataset_names = sorted(datasets_df["names"])
    datasets_df = datasets_df.drop_duplicates()  ## TODO: there should be no duplicates

    de_data_available = "groupid_for_de" in datasets_df.columns

    shapes_dict = {}
    dataset_path_map = {}
    dataset_group_map = {}  # Name of the obs column to be used for retrieing DE scrores

    shapes_dict["dev_immune_mouse"] = (443697, 4786)
    shapes_dict["dev_immune_human"] = (34009, 5566)
    shapes_dict["intestinal_tract_human"] = (69668, 5192)
    shapes_dict["gtex_human"] = (18511, 7109)
    shapes_dict["gut_endoderm_mouse"] = (113043, 6806)
    shapes_dict["luca"] = (249591, 7196)
    shapes_dict.update(
        {
            "madissoon_novel_lung": (190728, 8000),
            "flores_cerebellum_human": (20232, 8000),
            "osuch_gut_human": (272310, 8000),
            "msk_ovarian_human": (929690, 8000),
            "htan_vmuc_dis_epi_human": (65084, 8000),
            "htan_vmuc_val_epi_human": (57564, 8000),
            "htan_vmuc_non_epi_human": (9099, 8000),
            "hao_pbmc_3p_human": (161764, 8000),
            "hao_pbmc_5p_human": (49147, 8000),
            "gao_tumors_human": (36111, 8000),
            "swabrick_breast_human": (92427, 8000),
            "wu_cryo_tumors_human": (105662, 8000),
            "cell_line_het_human": (53513, 8000),
            "bi_allen_metastasis_human": (27787, 8000),
            "zheng68k_human": (68579, 8000),
            "zheng68k_12k_human": (68579, 12000),
            "mouse_embryo_ct": (153597, 12000),
            "regev_gtex_heart": (36574, 8000),
            "tabula_sapiens_heart": (11505, 8000),
            "10k_pbmcs": (11990, 12000),
            "epo_ido": (35834, 12000),
            "tabula_sapiens_kidney": (9641, 8000),
            "tabula_microcebus_kidney": (14592, 8000),
            "tabula_muris_kidney": (2781, 8000),
            "tabula_muris_senis_kidney": (19610, 8000),
            "immune_human": (33506, 8000),
        }
    )

    shapes_dict["zyl_sanes_glaucoma_pig"] = (5901, 6819)
    shapes_dict["parkinsons_macaF"] = (1062, 5103)

    for row in datasets_df.iterrows():
        ngenes = row[1].num_genes
        ncells = row[1].num_cells
        name = row[1].names
        dataset_path_map[name] = row[1].path
        if de_data_available:
            dataset_group_map[name] = datasets_df.set_index("names").loc[name]["groupid_for_de"]
        else:
            # This is for backward compatibility with old datasets CSV
            dataset_group_map[name] = "leiden"

        if not np.isnan(ngenes):
            shapes_dict[name] = (int(ncells), int(ngenes))
        else:
            shapes_dict[name] = (int(ncells), 8000)

    return datasets_df, sorted_dataset_names, shapes_dict, dataset_path_map, dataset_group_map
