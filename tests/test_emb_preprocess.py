import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from state._cli._emb._preprocess import detect_gene_name_strategy, extract_dataset_info


def _write_h5ad(path, nullable_strings: bool):
    X = sp.csr_matrix(np.ones((4, 3), dtype=np.float32))
    var = pd.DataFrame(
        {"gene_name": pd.array(["TP53", "GAPDH", "ACTB"], dtype="string")},
        index=pd.Index(["E1", "E2", "E3"], dtype="string"),
    )
    if not nullable_strings:
        var["gene_name"] = var["gene_name"].astype(object)
        var.index = var.index.astype(object)
    adata = ad.AnnData(X, obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]), var=var)
    with ad.settings.override(allow_write_nullable_strings=nullable_strings):
        adata.write_h5ad(path)


@pytest.mark.parametrize("nullable_strings", [False, True])
def test_extract_dataset_info_reads_string_columns(tmp_path, nullable_strings):
    path = tmp_path / "data.h5ad"
    _write_h5ad(path, nullable_strings)
    with h5py.File(path, "r") as f:
        # nullable_strings=True is the layout anndata >= 0.11 writes with pandas >= 3
        assert isinstance(f["var/gene_name"], h5py.Group) == nullable_strings

    assert detect_gene_name_strategy(path, {"TP53", "ACTB"}) == "gene_name"
    assert extract_dataset_info(path, "gene_name") == (4, 3, ["TP53", "GAPDH", "ACTB"])
    assert extract_dataset_info(path, "_index") == (4, 3, ["E1", "E2", "E3"])
