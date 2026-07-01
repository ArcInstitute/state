import anndata
import numpy as np
import pandas as pd
import pytest

from state._cli._tx._stream_h5ad import (
    StreamingDenseH5ad,
    select_stream_payload,
    validate_stream_adatas_args,
)


def _obs(n):
    base = pd.DataFrame(
        {"pert": ["a", "b", "a", "c", "b"], "ct": ["x", "x", "y", "y", "x"]}
    )
    return base.iloc[:n].reset_index(drop=True)


def test_roundtrip_matches_in_memory_writer(tmp_path):
    n, d, e = 5, 3, 4
    x = np.arange(n * d, dtype=np.float32).reshape(n, d)
    x[0, 0] = 99.0  # will be clipped to 14
    emb = np.arange(n * e, dtype=np.float32).reshape(n, e)
    obs = _obs(n)

    p = str(tmp_path / "stream.h5ad")
    w = StreamingDenseH5ad(p, n, d, obsm={"X_emb": e}, clip=(0.0, 14.0), chunk_rows=2)
    w.write_block(x[:2], {"X_emb": emb[:2]})
    w.write_block(x[2:], {"X_emb": emb[2:]})
    w.close(obs)

    # reference: the in-memory path clips X only, then writes
    x_ref = x.copy()
    np.clip(x_ref, 0.0, 14.0, out=x_ref)
    obs_ref = obs.copy()
    obs_ref.index = obs_ref.index.astype(str)
    ref = anndata.AnnData(X=x_ref, obs=obs_ref)
    ref.obsm["X_emb"] = emb
    pr = str(tmp_path / "ref.h5ad")
    ref.write_h5ad(pr)

    a = anndata.read_h5ad(p)
    r = anndata.read_h5ad(pr)
    assert a.shape == r.shape
    assert np.array_equal(a.X, r.X)
    assert a.X[0, 0] == 14.0  # clipped
    assert np.array_equal(a.obsm["X_emb"], r.obsm["X_emb"])
    assert a.obsm["X_emb"].max() == emb.max()  # obsm NOT clipped
    assert list(a.var.index) == list(r.var.index) == ["0", "1", "2"]
    assert a.obs.equals(r.obs)


def test_resize_down_for_shared_only(tmp_path):
    n, d = 5, 3
    x = np.arange(n * d, dtype=np.float32).reshape(n, d)
    p = str(tmp_path / "short.h5ad")
    w = StreamingDenseH5ad(p, n, d, clip=None, chunk_rows=2)
    w.write_block(x[:3])  # only 3 of 5 rows actually written
    w.close(_obs(3))
    a = anndata.read_h5ad(p)
    assert a.n_obs == 3
    assert np.array_equal(a.X, x[:3])


def test_empty_dataset(tmp_path):
    # num_cells == 0 (empty test set / fully-masked shared-only) must still write
    # a valid empty h5ad, matching the in-memory path's behavior.
    p = str(tmp_path / "empty.h5ad")
    w = StreamingDenseH5ad(p, 0, 3, obsm={"X_emb": 4}, clip=None)
    w.close(_obs(0))
    a = anndata.read_h5ad(p)
    assert a.shape == (0, 3)
    assert a.obsm["X_emb"].shape == (0, 4)
    assert list(a.var.index) == ["0", "1", "2"]


def test_row_overflow_raises(tmp_path):
    w = StreamingDenseH5ad(str(tmp_path / "o.h5ad"), 2, 3, clip=None)
    with pytest.raises(ValueError):
        w.write_block(np.zeros((3, 3), dtype=np.float32))


def test_obs_length_mismatch_raises(tmp_path):
    w = StreamingDenseH5ad(str(tmp_path / "m.h5ad"), 2, 3, clip=None)
    w.write_block(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        w.close(_obs(1))


def test_select_payload_embedding_space():
    bp = {
        "preds": np.ones((2, 4), dtype=np.float32),
        "pert_cell_emb": np.zeros((2, 4), dtype=np.float32),
    }
    x_pred, x_real, om_pred, om_real = select_stream_payload(bp, False, "X_emb")
    assert np.array_equal(x_pred, bp["preds"])
    assert np.array_equal(x_real, bp["pert_cell_emb"])
    assert om_pred is None and om_real is None


def test_select_payload_gene_space():
    bp = {
        "preds": np.ones((2, 4), dtype=np.float32),
        "pert_cell_emb": np.zeros((2, 4), dtype=np.float32),
        "pert_cell_counts_preds": np.full((2, 6), 3.0, dtype=np.float32),
        "pert_cell_counts": np.full((2, 6), 5.0, dtype=np.float32),
    }
    x_pred, x_real, om_pred, om_real = select_stream_payload(bp, True, "X_emb")
    assert np.array_equal(x_pred, bp["pert_cell_counts_preds"])
    assert np.array_equal(x_real, bp["pert_cell_counts"])
    assert np.array_equal(om_pred["X_emb"], bp["preds"])
    assert np.array_equal(om_real["X_emb"], bp["pert_cell_emb"])


def test_validate_rejects_skip_and_stream():
    class A:
        stream_adatas = True
        skip_adatas = True
    with pytest.raises(ValueError):
        validate_stream_adatas_args(A())


def test_validate_allows_stream_alone():
    class A:
        stream_adatas = True
        skip_adatas = False
    validate_stream_adatas_args(A())  # no raise
