import numpy as np
import anndata as ad

from state._cli._tx._preprocess_infer import run_tx_preprocess_infer
from state.tx.constants import HVG_VAR_NAMES_KEY


def test_preprocess_infer_preserves_uns(tmp_path):
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    adata = ad.AnnData(X=X)
    adata.obs["pert"] = ["ctrl", "pert"]
    adata.uns[HVG_VAR_NAMES_KEY] = np.array(["g1", "g2"], dtype=object)

    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "output.h5ad"
    adata.write_h5ad(input_path)

    run_tx_preprocess_infer(
        adata_path=str(input_path),
        output_path=str(output_path),
        control_condition="ctrl",
        pert_col="pert",
        seed=0,
        embed_key=None,
    )

    processed = ad.read_h5ad(output_path)
    assert HVG_VAR_NAMES_KEY in processed.uns
    assert processed.uns[HVG_VAR_NAMES_KEY].tolist() == ["g1", "g2"]
