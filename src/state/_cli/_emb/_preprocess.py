import argparse as ap
import os
import h5py as h5
import torch
from omegaconf import OmegaConf
from typing import Optional, Tuple


def add_arguments_preprocess(parser: ap.ArgumentParser):
    """Add arguments for embedding preprocessing CLI."""
    parser.add_argument(
        "--profile-name", required=True, help="Name for the new profile (used for both embeddings and dataset)"
    )
    parser.add_argument("--train-csv", required=True, help="Path to training CSV file (species,path,names columns)")
    parser.add_argument("--val-csv", required=True, help="Path to validation CSV file (species,path,names columns)")
    parser.add_argument("--output-dir", required=True, help="Directory to output generated files")
    parser.add_argument(
        "--config-file", default=None, help="Config file to update (default: src/state/configs/state-defaults.yaml)"
    )
    parser.add_argument(
        "--all-embeddings",
        default=None,
        help="Path to existing all_embeddings.pt file (if not provided, creates one-hot embeddings)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel worker processes to use (default: 1)",
    )


def _scan_one(args: Tuple[str, str, bool, Optional[set]]):
    """Top-level worker: scan a single dataset and return stats + genes.

    Returns (name, info_dict)
    """
    name, path, use_onehot, emb_keyset = args
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    field = detect_gene_name_strategy(path, emb_keyset if not use_onehot else None)
    num_cells, num_genes, genes = extract_dataset_info(path, field)
    return name, {"num_cells": num_cells, "num_genes": num_genes, "genes": genes, "gene_field": field}


def run_emb_preprocess(args):
    """
    Preprocess datasets and embeddings to create a new profile.
    """

    import os
    import logging
    import sys
    import pandas as pd
    import torch
    from tqdm import tqdm

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Default config file
    if args.config_file is None:
        args.config_file = "src/state/configs/state-defaults.yaml"

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or create embeddings
    if args.all_embeddings:
        if not os.path.exists(args.all_embeddings):
            log.error(f"All embeddings file not found: {args.all_embeddings}")
            sys.exit(1)
        log.info(f"Loading existing embeddings from {args.all_embeddings}")
        all_embeddings = torch.load(args.all_embeddings)
        if not isinstance(all_embeddings, dict):
            log.error("All embeddings file must be a dict mapping gene names to tensors")
            sys.exit(1)
        all_embeddings = {str(k).upper(): v for k, v in all_embeddings.items()}
        emb_size = next(iter(all_embeddings.values())).shape[0]
        gene_to_idx = {g: i for i, g in enumerate(all_embeddings.keys())}
        use_onehot = False
    else:
        all_embeddings = None
        gene_to_idx = {}
        emb_size = None
        use_onehot = True

    # Load CSVs
    log.info("Loading training and validation CSV files...")
    try:
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
    except Exception as e:
        log.error(f"Error loading CSV files: {e}")
        sys.exit(1)

    for name, df in [("train", train_df), ("val", val_df)]:
        missing = set(["species", "path", "names"]) - set(df.columns)
        if missing:
            log.error(f"{name} CSV missing required columns: {missing}")
            sys.exit(1)

    log.info(f"Processing {len(train_df)} training datasets and {len(val_df)} validation datasets...")

    # Collect per-dataset stats in parallel without shared mutable state
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = []
    for _, row in train_df.iterrows():
        tasks.append((row["names"], row["path"]))
    for _, row in val_df.iterrows():
        tasks.append((row["names"], row["path"]))

    dataset_info = {}
    all_genes = set()
    skipped = []  # (name, path, reason)

    # Use a set of embedding keys for fast membership checks in detection
    emb_keyset = set(all_embeddings.keys()) if all_embeddings is not None else None

    if args.num_threads and args.num_threads > 1 and len(tasks) > 1:
        log.info(f"Scanning datasets with {args.num_threads} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.num_threads) as ex:
            futs = {ex.submit(_scan_one, (n, p, use_onehot, emb_keyset)): (n, p) for n, p in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Scanning datasets"):
                n, p = futs[fut]
                try:
                    name, info = fut.result()
                except Exception as e:
                    # Skip bad files but continue processing others
                    skipped.append((n, p, str(e)))
                    continue
                dataset_info[name] = info
                all_genes.update(info["genes"])
    else:
        log.info("Scanning datasets serially...")
        for n, p in tqdm(tasks, total=len(tasks), desc="Scanning datasets"):
            try:
                name, info = _scan_one((n, p, use_onehot, emb_keyset))
            except Exception as e:
                skipped.append((n, p, str(e)))
                continue
            dataset_info[name] = info
            all_genes.update(info["genes"])

    if skipped:
        log.warning(f"Skipped {len(skipped)} datasets due to errors:")
        for n, p, r in skipped:
            # Print name and full path of all skipped files
            print(f"SKIPPED: name={n} path={p} reason={r}")

    log.info(f"Found {len(all_genes)} unique genes across datasets")

    if use_onehot:
        log.info("Creating one-hot embeddings...")
        all_embeddings = create_onehot_embeddings(all_genes)
        emb_size = len(all_genes)
        # Build deterministic index over sorted gene list
        gene_to_idx = {g: i for i, g in enumerate(sorted(all_genes))}
    else:
        # For provided embeddings, build index over their existing key order
        gene_to_idx = {g: i for i, g in enumerate(all_embeddings.keys())}

    embeddings_file = os.path.join(args.output_dir, f"all_embeddings_{args.profile_name}.pt")
    torch.save(all_embeddings, embeddings_file)
    log.info(f"Saved embeddings to {embeddings_file}")

    # Build mapping tensors and masks now that gene_to_idx is finalized
    for name, info in dataset_info.items():
        genes = info["genes"]
        mapping = []
        mask = []
        for g in genes:
            idx = gene_to_idx.get(g, -1)
            mapping.append(idx)
            mask.append(idx != -1)
        assert mask.count(False) == mapping.count(-1), f"Dataset {name}: mask False count != mapping -1 count"
        info["mapping"] = torch.tensor(mapping, dtype=torch.long)
        info["mask"] = torch.tensor(mask, dtype=torch.bool)

    ds_map = {name: info["mapping"] for name, info in dataset_info.items()}
    mapping_file = os.path.join(args.output_dir, f"ds_emb_mapping_{args.profile_name}.torch")
    torch.save(ds_map, mapping_file)
    log.info(f"Saved dataset mapping to {mapping_file}")

    masks = {name: info["mask"] for name, info in dataset_info.items()}
    masks_file = os.path.join(args.output_dir, f"valid_genes_masks_{args.profile_name}.torch")
    torch.save(masks, masks_file)
    log.info(f"Saved valid gene masks to {masks_file}")

    train_out = create_updated_csv(train_df, dataset_info, args.output_dir, f"train_{args.profile_name}.csv")
    val_out = create_updated_csv(val_df, dataset_info, args.output_dir, f"val_{args.profile_name}.csv")

    total_ds = len(train_df) + len(val_df)
    update_config_file(
        args.config_file,
        args.profile_name,
        embeddings_file,
        mapping_file,
        masks_file,
        train_out,
        val_out,
        emb_size,
        len(all_embeddings),
        total_ds,
    )

    log.info(
        "Preprocessing completed. Run: uv run state emb fit --conf %s embeddings.current=%s dataset.current=%s",
        args.config_file,
        args.profile_name,
        args.profile_name,
    )


def detect_gene_name_strategy(dataset_path, all_embeddings=None):
    """Detect which var field under /var/ holds gene names by max overlap."""
    with h5.File(dataset_path, "r") as f:
        if "var" not in f:
            raise ValueError(f"No var/ in {dataset_path}")

        # candidate fields in order of preference
        fields = [
            fld
            for fld in ("_index", "gene_name", "gene_symbols", "feature_name", "gene_id", "symbol")
            if fld in f["var"]
        ]
        if not fields:
            raise ValueError(f"No gene field found in var/ of {dataset_path}")

        # if no embeddings provided, just pick the first one
        if all_embeddings is None:
            return fields[0]

        best_field = fields[0]
        best_count = -1

        # for each candidate, count how many genes appear in all_embeddings
        for fld in fields:
            grp = f["var"][fld]
            raw = grp["categories"][:] if "categories" in grp else grp[:]
            genes = [
                item.decode("utf-8").upper() if isinstance(item, (bytes, bytearray)) else str(item).upper()
                for item in raw
            ]
            # count overlap
            match_count = sum(1 for g in genes if g in all_embeddings)

            # pick the field with the highest overlap
            if match_count > best_count:
                best_count = match_count
                best_field = fld

        return best_field


def extract_dataset_info(dataset_path, gene_field):
    """Extract num_cells, num_genes, and uppercase gene list, handling categorical codes correctly."""
    with h5.File(dataset_path, "r") as f:
        X = f["X"]
        attrs = dict(X.attrs)
        if "encoding-type" in attrs:
            enc = attrs["encoding-type"]
            if enc in ("csr_matrix", "csc_matrix"):
                num_cells, num_genes = attrs["shape"]
            else:
                num_cells, num_genes = X.shape
        else:
            if hasattr(X, "shape") and len(X.shape) == 2:
                num_cells, num_genes = X.shape
            else:
                num_cells = len(X["indptr"]) - 1
                num_genes = int(X["indices"][:].max()) + 1

        grp = f["var"][gene_field]
        # Handle categorical with codes
        if "categories" in grp and "codes" in grp:
            raw_cats = grp["categories"][:]
            codes = grp["codes"][:]
            cats = [
                c.decode("utf-8").upper() if isinstance(c, (bytes, bytearray)) else str(c).upper() for c in raw_cats
            ]
            genes = [cats[int(code)] for code in codes]
        else:
            raw = grp["categories"][:] if "categories" in grp else grp[:]
            genes = [
                item.decode("utf-8").upper() if isinstance(item, (bytes, bytearray)) else str(item).upper()
                for item in raw
            ]

        if len(genes) != num_genes:
            raise ValueError(f"Gene count mismatch {len(genes)} vs {num_genes}")

    return num_cells, num_genes, genes


def create_onehot_embeddings(all_genes):
    """Make one-hot embeddings for each gene."""
    genes_sorted = sorted(all_genes)
    emb = {}
    for i, g in enumerate(genes_sorted):
        vec = torch.zeros(len(genes_sorted))
        vec[i] = 1.0
        emb[g] = vec
    return emb


def create_updated_csv(df, dataset_info, out_dir, filename):
    """Add num_cells, num_genes, groupid_for_de and save."""
    out = df[df["names"].isin(dataset_info.keys())].copy()
    out["num_cells"] = out["names"].map(lambda n: dataset_info[n]["num_cells"])
    out["num_genes"] = out["names"].map(lambda n: dataset_info[n]["num_genes"])
    out["groupid_for_de"] = "leiden"
    path = os.path.join(out_dir, filename)
    out.to_csv(path, index=False)
    return path


def update_config_file(
    config_path,
    profile_name,
    embeddings_file,
    mapping_file,
    masks_file,
    train_csv,
    val_csv,
    embedding_size,
    num_embeddings,
    num_datasets,
):
    """Insert new profile into config YAML."""
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({})
    if "embeddings" not in cfg:
        cfg.embeddings = {}
    if "dataset" not in cfg:
        cfg.dataset = {}
    cfg.embeddings[profile_name] = {
        "all_embeddings": embeddings_file,
        "ds_emb_mapping": mapping_file,
        "valid_genes_masks": masks_file,
        "size": embedding_size,
        "num": num_embeddings,
    }
    cfg.dataset[profile_name] = {
        "ds_type": "h5ad",
        "train": train_csv,
        "val": val_csv,
        "filter": True,
        "num_datasets": num_datasets,
    }
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    add_arguments_preprocess(parser)
    args = parser.parse_args()
    run_emb_preprocess(args)
