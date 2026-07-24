from types import SimpleNamespace

import pytest
import torch
from torch import nn

from state.tx.models.base import PerturbationModel
from state.tx.models.context_mean import ContextMeanPerturbationModel
from state.tx.models.perturb_mean import PerturbMeanPerturbationModel


class _IdentityPerturbationModel(PerturbationModel):
    def _build_networks(self):
        pass

    def forward(self, batch):
        return batch["ctrl_cell_emb"]


def _model(**kwargs):
    return _IdentityPerturbationModel(
        input_dim=3,
        hidden_dim=3,
        output_dim=3,
        pert_dim=1,
        loss_fn=nn.MSELoss(),
        output_space="all",
        gene_decoder_bool=False,
        **kwargs,
    )


def test_cp10k_log1p_normalizes_each_cell():
    model = _model(
        embed_key="X_hvg",
        log1p_from_raw_counts=True,
        counts_target_sum=10_000,
    )
    counts = torch.tensor([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]])

    actual = model._cp_log1p(counts)
    expected = torch.log1p(torch.tensor([[2_500.0, 7_500.0, 0.0], [0.0, 0.0, 0.0]]))

    torch.testing.assert_close(actual, expected)


def test_count_batch_normalization_does_not_mutate_input():
    model = _model(embed_key="X_hvg", log1p_from_raw_counts=True)
    counts = torch.tensor([[1.0, 3.0, 0.0]])
    batch = {
        "ctrl_cell_emb": counts,
        "pert_cell_emb": counts * 2,
        "pert_cell_counts": counts * 3,
        "pert_emb": torch.tensor([[1.0]]),
    }

    normalized = model._normalize_count_keys(batch)

    assert normalized is not batch
    assert normalized["pert_emb"] is batch["pert_emb"]
    torch.testing.assert_close(batch["ctrl_cell_emb"], counts)
    for key in ("ctrl_cell_emb", "pert_cell_emb", "pert_cell_counts"):
        torch.testing.assert_close(normalized[key].expm1().sum(dim=-1), torch.tensor([10_000.0]))


def test_learned_embeddings_are_not_normalized():
    model = _model(embed_key="X_state", log1p_from_raw_counts=True)
    embedding = torch.tensor([[1.0, -2.0, 3.0]])
    counts = torch.tensor([[1.0, 3.0, 0.0]])
    batch = {
        "ctrl_cell_emb": embedding,
        "pert_cell_emb": embedding,
        "pert_cell_counts": counts,
    }

    normalized = model._normalize_count_keys(batch)

    assert normalized["ctrl_cell_emb"] is embedding
    assert normalized["pert_cell_emb"] is embedding
    torch.testing.assert_close(
        normalized["pert_cell_counts"].expm1().sum(dim=-1),
        torch.tensor([10_000.0]),
    )


def test_normalization_is_disabled_by_default():
    model = _model(embed_key="X_hvg")
    batch = {"pert_cell_emb": torch.tensor([[1.0, 2.0, 3.0]])}

    assert model._normalize_count_keys(batch) is batch


def test_counts_target_sum_must_be_positive():
    with pytest.raises(ValueError, match="counts_target_sum must be positive"):
        _model(
            embed_key="X_hvg",
            log1p_from_raw_counts=True,
            counts_target_sum=0,
        )


def test_predict_step_returns_normalized_real_values():
    model = _model(embed_key="X_hvg", log1p_from_raw_counts=True)
    counts = torch.tensor([[1.0, 3.0, 0.0]])
    batch = {
        "ctrl_cell_emb": counts,
        "pert_cell_emb": counts * 2,
        "pert_cell_counts": counts * 3,
        "pert_name": ["P1"],
        "cell_type": ["CT1"],
    }

    output = model.predict_step(batch, batch_idx=0)

    for key in ("preds", "ctrl_cell_emb", "pert_cell_emb", "pert_cell_counts"):
        torch.testing.assert_close(output[key].expm1().sum(dim=-1), torch.tensor([10_000.0]))


def _raw_mean_batch():
    return {
        "ctrl_cell_emb": torch.tensor([[1.0, 3.0, 0.0], [1.0, 3.0, 0.0]]),
        "pert_cell_emb": torch.tensor([[1.0, 3.0, 0.0], [3.0, 1.0, 0.0]]),
        "pert_cell_counts": torch.tensor([[1.0, 3.0, 0.0], [3.0, 1.0, 0.0]]),
        "pert_name": ["control", "P1"],
        "cell_type": ["CT1", "CT1"],
    }


def _attach_train_batch(model, batch):
    datamodule = SimpleNamespace(train_dataloader=lambda: [batch])
    model._trainer = SimpleNamespace(datamodule=datamodule)


def test_perturb_mean_fits_offsets_in_normalized_space():
    model = PerturbMeanPerturbationModel(
        input_dim=3,
        hidden_dim=3,
        output_dim=3,
        pert_dim=1,
        loss_fn=nn.MSELoss(),
        control_pert="control",
        embed_key="X_hvg",
        output_space="all",
        gene_decoder_bool=False,
        log1p_from_raw_counts=True,
    )
    batch = _raw_mean_batch()
    _attach_train_batch(model, batch)

    model.on_fit_start()

    ctrl = model._cp_log1p(batch["pert_cell_counts"][:1]).squeeze(0)
    pert = model._cp_log1p(batch["pert_cell_counts"][1:]).squeeze(0)
    torch.testing.assert_close(model.global_basal, ctrl)
    torch.testing.assert_close(model.pert_mean_offsets["P1"], pert - ctrl)


def test_context_mean_fits_means_in_normalized_space():
    model = ContextMeanPerturbationModel(
        input_dim=3,
        hidden_dim=3,
        output_dim=3,
        pert_dim=1,
        loss_fn=nn.MSELoss(),
        control_pert="control",
        embed_key="X_hvg",
        output_space="all",
        gene_decoder_bool=False,
        log1p_from_raw_counts=True,
    )
    batch = _raw_mean_batch()
    _attach_train_batch(model, batch)

    model.on_fit_start()

    expected = model._cp_log1p(batch["pert_cell_counts"][1:]).squeeze(0)
    torch.testing.assert_close(model.celltype_pert_means["CT1"], expected)
