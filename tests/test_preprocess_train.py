import numpy as np
import anndata as ad

from state._cli._tx._preprocess_train import run_tx_preprocess_train
from state.tx.constants import HVG_OBSM_KEY, HVG_VAR_NAMES_KEY


def test_preprocess_train_stores_hvg_names(tmp_path):
    X = np.array(
        [
            [1, 0, 3, 4, 0, 2],
            [2, 1, 0, 1, 0, 3],
            [0, 2, 1, 0, 4, 1],
            [3, 0, 2, 2, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    var_names = [f"gene_{i}" for i in range(X.shape[1])]
    adata = ad.AnnData(X=X)
    adata.var_names = var_names

    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    adata.write_h5ad(input_path)

    run_tx_preprocess_train(str(input_path), str(output_path), num_hvgs=3)

    processed = ad.read_h5ad(output_path)
    assert HVG_OBSM_KEY in processed.obsm
    assert HVG_VAR_NAMES_KEY in processed.uns

    hvg_names = processed.uns[HVG_VAR_NAMES_KEY]
    assert isinstance(hvg_names, np.ndarray)
    assert hvg_names.dtype == object
    assert len(hvg_names) == processed.obsm[HVG_OBSM_KEY].shape[1]
    assert all(isinstance(name, str) for name in hvg_names)

    expected_names = processed.var_names[processed.var.highly_variable].tolist()
    assert set(hvg_names.tolist()) == set(expected_names)
