import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import yaml

from state._cli._tx._infer import run_tx_infer
from state._cli._tx._preprocess_train import run_tx_preprocess_train
from state.tx.constants import HVG_VAR_NAMES_KEY


class DummyModel(torch.nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.batch_encoder = None
        self.cell_sentence_len = 2
        self.output_space = "gene"
        self._output_dim = output_dim

    def predict_step(self, batch, batch_idx=0, padded=False):
        ctrl = batch["ctrl_cell_emb"]
        return {"preds": ctrl.clone()}


def _write_model_assets(model_dir: Path, output_dim: int, pert_dim: int = 2):
    config = {
        "data": {
            "kwargs": {
                "control_pert": "ctrl",
                "output_space": "gene",
                "cell_type_key": "cell_type",
            }
        },
        "model": {"kwargs": {}},
    }
    with open(model_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    var_dims = {"pert_dim": pert_dim, "batch_dim": None, "output_dim": output_dim}
    with open(model_dir / "var_dims.pkl", "wb") as f:
        pickle.dump(var_dims, f)

    pert_onehot_map = {
        "ctrl": torch.tensor([1.0, 0.0]),
        "pert": torch.tensor([0.0, 1.0]),
    }
    torch.save(pert_onehot_map, model_dir / "pert_onehot_map.pt")


def _make_args(model_dir: Path, adata_path: Path, output_path: Path, quiet: bool, verbose: bool):
    return argparse.Namespace(
        checkpoint=str(model_dir / "checkpoints" / "final.ckpt"),
        adata=str(adata_path),
        embed_key="X_hvg",
        pert_col="pert",
        output=str(output_path),
        model_dir=str(model_dir),
        celltype_col="cell_type",
        celltypes=None,
        batch_col=None,
        control_pert="ctrl",
        seed=42,
        max_set_len=2,
        quiet=quiet,
        tsv=None,
        verbose=verbose,
    )


def test_infer_preserves_hvg_names(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "checkpoints").mkdir()
    (model_dir / "checkpoints" / "final.ckpt").touch()

    X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    raw = ad.AnnData(X=X)
    raw.obs["pert"] = ["ctrl", "pert", "pert"]
    raw.obs["cell_type"] = ["A", "A", "B"]
    raw.var_names = ["g1", "g2", "g3"]

    raw_path = tmp_path / "raw.h5ad"
    preprocessed_path = tmp_path / "preprocessed.h5ad"
    output_path = tmp_path / "output.h5ad"
    raw.write_h5ad(raw_path)

    run_tx_preprocess_train(str(raw_path), str(preprocessed_path), num_hvgs=2)

    _write_model_assets(model_dir, output_dim=2)

    dummy = DummyModel(output_dim=2)
    monkeypatch.setattr(
        "state.tx.models.state_transition.StateTransitionPerturbationModel.load_from_checkpoint",
        lambda *args, **kwargs: dummy,
    )

    args = _make_args(model_dir, preprocessed_path, output_path, quiet=True, verbose=False)
    run_tx_infer(args)

    out = ad.read_h5ad(output_path)
    assert HVG_VAR_NAMES_KEY in out.uns
    assert len(out.uns[HVG_VAR_NAMES_KEY]) == out.obsm["X_hvg"].shape[1]


def test_infer_warns_when_hvg_missing(monkeypatch, tmp_path, capsys):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "checkpoints").mkdir()
    (model_dir / "checkpoints" / "final.ckpt").touch()

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    adata = ad.AnnData(X=X)
    adata.obs["pert"] = ["ctrl", "pert"]
    adata.obs["cell_type"] = ["A", "A"]
    adata.obsm["X_hvg"] = X.copy()

    adata_path = tmp_path / "input_missing.h5ad"
    output_path = tmp_path / "output_missing.h5ad"
    adata.write_h5ad(adata_path)

    _write_model_assets(model_dir, output_dim=X.shape[1])

    dummy = DummyModel(output_dim=X.shape[1])
    monkeypatch.setattr(
        "state.tx.models.state_transition.StateTransitionPerturbationModel.load_from_checkpoint",
        lambda *args, **kwargs: dummy,
    )

    args = _make_args(model_dir, adata_path, output_path, quiet=False, verbose=True)
    run_tx_infer(args)

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Warning: adata.uns['X_hvg_var_names'] not found" in combined
    assert "HVG names:" in combined
