import types
from pathlib import Path

import anndata as ad
import torch
import yaml

from state._cli._tx._predict import run_tx_predict
from state.tx.constants import HVG_VAR_NAMES_KEY


class DummyBatchSampler:
    def __init__(self, tot_num: int):
        self.tot_num = tot_num


class DummyLoader:
    def __init__(self, batch):
        self.batch_sampler = DummyBatchSampler(batch["pert_cell_emb"].shape[0])
        self._batch = batch

    def __iter__(self):
        yield self._batch


class DummyDataModule:
    def __init__(self, gene_names):
        self.embed_key = "X_hvg"
        self.pert_col = "pert"
        self.cell_type_key = "cell_type"
        self.batch_col = "batch"
        self._gene_names = gene_names
        self.batch_size = 1

    def setup(self, stage="test"):
        return None

    def get_var_dims(self):
        return {
            "input_dim": 2,
            "gene_dim": 2,
            "hvg_dim": 2,
            "output_dim": 2,
            "pert_dim": 2,
            "gene_names": self._gene_names,
        }

    def test_dataloader(self):
        batch = {
            "pert_cell_emb": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "ctrl_cell_emb": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        }
        return DummyLoader(batch)

    def train_dataloader(self, test=False):
        return self.test_dataloader()

    def get_control_pert(self):
        return "ctrl"

    def get_shared_perturbations(self):
        return []


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def predict_step(self, batch, batch_idx, padded=False):
        preds = batch["pert_cell_emb"].clone()
        return {
            "pert_name": ["ctrl", "pert"],
            "celltype_name": ["A", "A"],
            "batch": torch.tensor([0, 0]),
            "preds": preds,
            "pert_cell_emb": batch["pert_cell_emb"],
        }


def _install_dummy_modules(monkeypatch, data_module):
    cell_eval = types.ModuleType("cell_eval")
    cell_eval.MetricsEvaluator = object
    cell_eval_utils = types.ModuleType("cell_eval.utils")
    cell_eval_utils.split_anndata_on_celltype = lambda adata, celltype_col: {"all": adata}
    monkeypatch.setitem(__import__("sys").modules, "cell_eval", cell_eval)
    monkeypatch.setitem(__import__("sys").modules, "cell_eval.utils", cell_eval_utils)

    cell_load = types.ModuleType("cell_load")
    cell_load_data = types.ModuleType("cell_load.data_modules")

    class DummyPerturbationDataModule:
        @staticmethod
        def load_state(_path):
            return data_module

    cell_load_data.PerturbationDataModule = DummyPerturbationDataModule
    monkeypatch.setitem(__import__("sys").modules, "cell_load", cell_load)
    monkeypatch.setitem(__import__("sys").modules, "cell_load.data_modules", cell_load_data)


def test_predict_outputs_hvg_names(monkeypatch, tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    cfg = {
        "output_dir": str(tmp_path),
        "name": "run",
        "training": {"train_seed": 0},
        "model": {"name": "state", "kwargs": {"hidden_dim": 4}},
        "data": {"kwargs": {"output_space": "gene"}},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    run_output_dir = tmp_path / "run"
    run_output_dir.mkdir(exist_ok=True)
    (run_output_dir / "data_module.torch").touch()
    checkpoints_dir = run_output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "last.ckpt").touch()

    gene_names = ["g1", "g2"]
    data_module = DummyDataModule(gene_names)
    _install_dummy_modules(monkeypatch, data_module)

    monkeypatch.setattr(
        "state.tx.models.state_transition.StateTransitionPerturbationModel.load_from_checkpoint",
        lambda *args, **kwargs: DummyModel(),
    )

    args = types.SimpleNamespace(
        output_dir=str(tmp_path),
        checkpoint="last.ckpt",
        test_time_finetune=0,
        profile="anndata",
        predict_only=True,
        shared_only=False,
        eval_train_data=False,
    )

    run_tx_predict(args)

    results_dir = Path(tmp_path) / "eval_last.ckpt"
    adata_pred = ad.read_h5ad(results_dir / "adata_pred.h5ad")
    adata_real = ad.read_h5ad(results_dir / "adata_real.h5ad")

    assert HVG_VAR_NAMES_KEY in adata_pred.uns
    assert HVG_VAR_NAMES_KEY in adata_real.uns
    assert adata_pred.uns[HVG_VAR_NAMES_KEY].tolist() == gene_names
    assert adata_real.uns[HVG_VAR_NAMES_KEY].tolist() == gene_names
