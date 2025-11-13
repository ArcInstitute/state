import scanpy as sc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vci.data import create_dataloader
from vci.eval.emb import cluster_embedding
from vci.utils import compute_gene_overlap_cross_pert
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings


def evaluate_intrinsic(model, cfg, device=None, logger=print, adata=None):
    """
    Standalone evaluation of perturbation effects.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if adata is None:
        adata = sc.read_h5ad(cfg["validations"]["perturbation"]["dataset"])
        adata.X = np.log1p(adata.X)

    if "X_emb" not in adata.obsm.keys():
        dataloader = create_dataloader(
            cfg,
            adata=adata,
            adata_name=cfg["validations"]["perturbation"]["dataset_name"],
            shuffle=False,
            sentence_collator=getattr(model, "collater", None),
        )
        all_embs = []
        for batch in tqdm(
            dataloader, desc=f"Perturbation Embeddings: {cfg['validations']['perturbation']['dataset_name']}"
        ):
            with torch.no_grad():
                _, _, _, emb, _ = model._compute_embedding_for_batch(batch)
                all_embs.append(emb.cpu().detach().numpy())
        all_embs = np.concatenate(all_embs, axis=0)
        adata.obsm["X_emb"] = all_embs

    cluster_embedding(adata, 0, emb_key="X_emb", use_pca=True, job_name=cfg["experiment"]["name"])

    # Run the intrinsic benchmark evaluation
    intrinsic_results = run_intrinsic_benchmark(adata, device, logger)

    return intrinsic_results


def evaluate_de(model, cfg, device=None, logger=print):
    """
    Standalone evaluation of differential expression (DE).

    Returns the anndata annotated with X_emb
    """

    # Get ground truth DE genes
    de_val_adata = sc.read_h5ad(cfg["validations"]["diff_exp"]["dataset"])
    sc.pp.log1p(de_val_adata)
    sc.tl.rank_genes_groups(
        de_val_adata,
        groupby=cfg["validations"]["diff_exp"]["obs_pert_col"],
        reference=cfg["validations"]["diff_exp"]["obs_filter_label"],
        rankby_abs=True,
        n_genes=cfg["validations"]["diff_exp"]["top_k_rank"],
        method=cfg["validations"]["diff_exp"]["method"],
        use_raw=False,
    )
    true_top_genes = pd.DataFrame(de_val_adata.uns["rank_genes_groups"]["names"]).T
    del de_val_adata

    # now for the model
    tmp_adata = sc.read_h5ad(cfg["validations"]["diff_exp"]["dataset"])
    pred_exp = model._predict_exp_for_adata(
        tmp_adata, cfg["validations"]["diff_exp"]["dataset_name"], cfg["validations"]["diff_exp"]["obs_pert_col"]
    )
    torch.cuda.synchronize()
    de_metrics = compute_gene_overlap_cross_pert(
        pred_exp, true_top_genes, k=cfg["validations"]["diff_exp"]["top_k_rank"]
    )
    mean_overlap = float(np.array(list(de_metrics.values())).mean())
    logger(f"DE gene overlap mean: {mean_overlap:.4f}")
    return tmp_adata


class MLPClassifier(nn.Module):
    """N-layer MLP with LayerNorm, GELU, and dropout."""

    def __init__(self, in_dim, hidden_dim, n_classes, n_layers, dropout=0.1):
        super().__init__()
        layers = []
        norms = []
        # first layer: from in_dim → hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        norms.append(nn.LayerNorm(hidden_dim))
        # remaining (n_layers-1) hidden layers: hidden_dim → hidden_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            norms.append(nn.LayerNorm(hidden_dim))

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        for lin, norm in zip(self.layers, self.norms):
            x = self.drop(self.act(norm(lin(x))))
        return self.out(x)


def split_indices_fraction(labels: np.ndarray, val_frac: float, n_groups: int, seed: int):
    """
    For each label in 0..n_groups-1:
      - holdout_count = floor(val_frac * group_size)
      - train = first (group_size - holdout_count)
      - split holdout equally into val/test
    """
    idx = np.arange(len(labels))
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for g in range(n_groups):
        inds = idx[labels == g]
        if len(inds) == 0:
            continue
        rng.shuffle(inds)
        n = len(inds)
        holdout = int(np.floor(val_frac * n))
        if holdout >= n:
            warnings.warn(f"val-split too large for group {g} (size={n}), using minimal split")
            holdout = max(0, n - 1)

        n_train = n - holdout
        n_val = holdout // 2
        n_test = holdout - n_val

        train_idx.extend(inds[:n_train])
        if n_val > 0:
            val_idx.extend(inds[n_train : n_train + n_val])
        if n_test > 0:
            test_idx.extend(inds[n_train + n_val :])

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
    )


def make_loaders(features, labels, train_idx, val_idx, test_idx, batch_size):
    """Create DataLoaders for train/val/test splits."""

    def mk(subset):
        if len(subset) == 0:
            return None
        X = torch.from_numpy(features[subset]).float()
        y = torch.from_numpy(labels[subset]).long()
        return TensorDataset(X, y)

    train_ds = mk(train_idx)
    val_ds = mk(val_idx)
    test_ds = mk(test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None

    return train_loader, val_loader, test_loader


def train_and_select(model, loaders, epochs, lr, device):
    """Train model and select best checkpoint based on validation loss."""
    train_loader, val_loader, _ = loaders

    if train_loader is None:
        raise ValueError("No training data available")

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        # Training
        model.train()
        total_train = 0.0
        with tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [train]", leave=False) as pbar:
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(X), y)
                loss.backward()
                opt.step()
                total_train += loss.item() * X.size(0)
                pbar.set_postfix(loss=loss.item())
        avg_train = total_train / len(train_loader.dataset)

        # Validation
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    total_val += loss_fn(model(X), y).item() * X.size(0)
            avg_val = total_val / len(val_loader.dataset)
        else:
            avg_val = avg_train  # Use training loss if no validation set

        if avg_val < best_val:
            best_val = avg_val
            best_state = model.state_dict().copy()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(model, loader, device):
    """Evaluate model on loader data and return loss, accuracy, and AUROC."""
    if loader is None:
        return float("nan"), float("nan"), float("nan")

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_preds = []
    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Loss
    avg_loss = total_loss / len(loader.dataset)

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # AUROC - use one-vs-rest approach for multiclass
    n_classes = all_probs.shape[1]
    if n_classes > 2:
        # For multiclass, compute one-vs-rest AUROC for each class then average
        auroc_scores = []
        for i in range(n_classes):
            # Create binary labels for current class (1 for current class, 0 for others)
            binary_labels = (all_labels == i).astype(int)
            # Use probability of current class as score
            class_probs = all_probs[:, i]
            try:
                auroc = roc_auc_score(binary_labels, class_probs)
                auroc_scores.append(auroc)
            except ValueError:
                # Skip this class if it has only one label in the test set
                pass
        auroc = np.mean(auroc_scores) if auroc_scores else float("nan")
    else:
        # For binary classification
        try:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            auroc = float("nan")

    return avg_loss, accuracy, auroc


def filter_and_split_by_celltype(
    adata,
    embed_key="X_emb",
    perturb_key="gene",
    cell_type_key="cell_type",
    min_cells_per_perturb=10,
    min_cells_per_celltype=50,
):
    """
    Split AnnData by cell type and filter each subset.
    Returns dictionary of {cell_type: (features, labels, label_names)} for valid cell types.
    """
    # Get unique cell types and their counts
    cell_types = adata.obs[cell_type_key].astype(str)
    cell_type_counts = cell_types.value_counts()

    print("\nCell type distribution:")
    for ct, count in cell_type_counts.items():
        print(f"  {ct}: {count} cells")

    # Filter cell types with sufficient cells
    valid_cell_types = cell_type_counts[cell_type_counts >= min_cells_per_celltype].index.tolist()
    print(f"\nCell types with >= {min_cells_per_celltype} cells: {valid_cell_types}")

    if not valid_cell_types:
        raise ValueError(f"No cell types have >= {min_cells_per_celltype} cells")

    celltype_data = {}

    for cell_type in valid_cell_types:
        print(f"\nProcessing cell type: {cell_type}")

        # Subset data for this cell type
        mask = cell_types == cell_type
        ct_adata = adata[mask].copy()

        # Filter perturbations with sufficient cells in this cell type
        perturb_labels = ct_adata.obs[perturb_key].astype(str).values
        uniq_perturbs, counts = np.unique(perturb_labels, return_counts=True)

        # Keep perturbations with enough cells
        keep_perturbs = set(uniq_perturbs[counts >= min_cells_per_perturb])
        if not keep_perturbs:
            print(f"  Skipping {cell_type}: no perturbations with >= {min_cells_per_perturb} cells")
            continue

        # Filter to keep only valid perturbations
        perturb_mask = np.array([lbl in keep_perturbs for lbl in perturb_labels])
        ct_adata = ct_adata[perturb_mask].copy()

        # Convert perturbation labels to categorical codes
        perturb_cats = ct_adata.obs[perturb_key].astype("category")
        labels = perturb_cats.cat.codes.values
        label_names = list(perturb_cats.cat.categories)

        # Get features
        features = ct_adata.obsm[embed_key]

        print(f"  Final data: {len(labels)} cells, {len(label_names)} perturbations")

        celltype_data[cell_type] = (features, labels, label_names)

    if not celltype_data:
        raise ValueError("No valid cell types found after filtering")

    return celltype_data


def benchmark_single_celltype(
    cell_type,
    features,
    labels,
    label_names,
    device,
    val_split=0.20,
    epochs=5,
    batch_size=128,
    lr=1e-4,
    n_layers=1,
    seed=42,
):
    """Run benchmarking for a single cell type."""
    print(f"\nBenchmarking cell type: {cell_type}")

    # Split data
    tr_idx, va_idx, te_idx = split_indices_fraction(labels, val_split, len(label_names), seed)

    print(f"  Data splits: {len(tr_idx)} train, {len(va_idx)} val, {len(te_idx)} test")

    # Create data loaders
    loaders = make_loaders(features, labels, tr_idx, va_idx, te_idx, batch_size)
    train_loader, val_loader, test_loader = loaders

    if train_loader is None or test_loader is None:
        print(f"  Insufficient data for cell type {cell_type}")
        return None

    # Create and train model
    model = MLPClassifier(in_dim=features.shape[1], hidden_dim=1024, n_classes=len(label_names), n_layers=n_layers).to(
        device
    )

    print(f"  Training model with {sum(p.numel() for p in model.parameters())} parameters...")
    model = train_and_select(model, loaders, epochs, lr, device)

    # Evaluate
    test_loss, test_acc, test_auroc = evaluate_model(model, test_loader, device)

    results = {
        "cell_type": cell_type,
        "loss": test_loss,
        "accuracy": test_acc,
        "auroc": test_auroc,
        "n_perturbations": len(label_names),
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
    }

    print(f"  Results: Loss={test_loss:.4f}, Accuracy={test_acc:.4f}, AUROC={test_auroc:.4f}")

    return results


def run_intrinsic_benchmark(adata, device, logger=print):
    """
    Run the intrinsic benchmark evaluation on embeddings.
    Returns averaged AUROC and Accuracy across all cell types.
    """
    logger("Running intrinsic perturbation benchmark...")

    # Fixed parameters
    embed_key = "X_emb"
    perturb_key = "gene"
    cell_type_key = "cell_type"
    val_split = 0.20
    epochs = 5
    batch_size = 128
    lr = 1e-4
    n_layers = 1
    min_cells_per_perturb = 10
    min_cells_per_celltype = 50
    seed = 42

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split by cell type and filter
    try:
        celltype_data = filter_and_split_by_celltype(
            adata, embed_key, perturb_key, cell_type_key, min_cells_per_perturb, min_cells_per_celltype
        )
    except Exception as e:
        logger(f"Error in data filtering: {e}")
        return {"intrinsic_auroc_mean": float("nan"), "intrinsic_accuracy_mean": float("nan")}

    logger(f"Benchmarking {len(celltype_data)} cell types")

    # Run benchmarking for each cell type
    all_results = []
    for cell_type, (features, labels, label_names) in celltype_data.items():
        result = benchmark_single_celltype(
            cell_type, features, labels, label_names, device, val_split, epochs, batch_size, lr, n_layers, seed
        )
        if result is not None:
            all_results.append(result)

    # Calculate and report averaged metrics
    if not all_results:
        logger("No valid results obtained from intrinsic benchmark!")
        return {"intrinsic_auroc_mean": float("nan"), "intrinsic_accuracy_mean": float("nan")}

    # Calculate averages (excluding NaN values)
    accuracies = [r["accuracy"] for r in all_results if not np.isnan(r["accuracy"])]
    aurocs = [r["auroc"] for r in all_results if not np.isnan(r["auroc"])]

    avg_accuracy = np.mean(accuracies) if accuracies else float("nan")
    avg_auroc = np.mean(aurocs) if aurocs else float("nan")

    # Print individual results
    logger("\nPer-cell-type intrinsic benchmark results:")
    for result in all_results:
        logger(
            f"  {result['cell_type']:15s}: "
            f"Acc={result['accuracy']:.4f}, "
            f"AUROC={result['auroc']:.4f} "
            f"({result['n_perturbations']} perturbations, {result['n_test']} test cells)"
        )

    logger(f"\nIntrinsic benchmark averaged metrics (across {len(all_results)} cell types):")
    logger(f"  Average Accuracy: {avg_accuracy:.4f}")
    logger(f"  Average AUROC: {avg_auroc:.4f}")

    return {"intrinsic_auroc_mean": avg_auroc, "intrinsic_accuracy_mean": avg_accuracy}
