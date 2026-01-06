import numpy as np
import anndata as ad

from state.tx.utils.hvg import get_hvg_var_names, validate_hvg_var_names


def test_get_hvg_var_names_prefers_obsm_key():
    adata = ad.AnnData(X=np.ones((2, 2)))
    adata.var_names = ["g1", "g2"]
    adata.uns["X_custom_var_names"] = np.array(["a", "b"], dtype=object)

    names = get_hvg_var_names(adata, obsm_key="X_custom")
    assert names == ["a", "b"]


def test_get_hvg_var_names_falls_back_to_highly_variable():
    adata = ad.AnnData(X=np.ones((3, 3)))
    adata.var_names = ["g1", "g2", "g3"]
    adata.var["highly_variable"] = [True, False, True]

    names = get_hvg_var_names(adata)
    assert names == ["g1", "g3"]


def test_get_hvg_var_names_returns_none_when_missing():
    adata = ad.AnnData(X=np.ones((2, 2)))
    adata.var_names = ["g1", "g2"]

    assert get_hvg_var_names(adata) is None


def test_validate_hvg_var_names():
    adata = ad.AnnData(X=np.ones((2, 2)))
    adata.var_names = ["g1", "g2"]
    assert validate_hvg_var_names(adata) is False

    adata.uns["X_hvg_var_names"] = np.array(["g1", "g2"], dtype=object)
    assert validate_hvg_var_names(adata) is True
