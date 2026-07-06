from __future__ import annotations

from types import TracebackType
from typing import Any, Mapping

import anndata
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata.io import write_elem

# anndata h5ad encoding tags (verified on anndata 0.12.17; guarded by the
# round-trip test in tests/test_stream_h5ad.py).
_ANNDATA_ENCODING = ("anndata", "0.1.0")
_ARRAY_ENCODING = ("array", "0.2.0")
_DICT_ENCODING = ("dict", "0.1.0")
# Empty optional groups anndata.write_h5ad always emits.
_EMPTY_DICT_KEYS = ("layers", "uns", "obsp", "varm", "varp")


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like batch field to a float32 ndarray."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value).astype(np.float32, copy=False)


def select_stream_payload(
    batch_preds: Mapping[str, Any],
    store_raw_expression: bool,
    embed_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    """Pick the X (and obsm) blocks for the pred/real writers from one batch.

    Mirrors the AnnData assembly in ``run_tx_predict``:
      - ``store_raw_expression``: X is the decoded gene expression
        (pred=``pert_cell_counts_preds``, real=``pert_cell_counts``) and the
        embeddings go to ``obsm[embed_key]`` (pred=``preds``, real=``pert_cell_emb``).
      - otherwise: X is the embedding (pred=``preds``, real=``pert_cell_emb``)
        and there is no obsm.

    Returns ``(x_pred, x_real, obsm_pred, obsm_real)`` as float32 ndarrays;
    ``obsm_*`` are ``{embed_key: arr}`` dicts or ``None``.
    """
    preds = _to_numpy(batch_preds["preds"])
    reals = _to_numpy(batch_preds["pert_cell_emb"])
    if store_raw_expression:
        x_pred = _to_numpy(batch_preds["pert_cell_counts_preds"])
        x_real = _to_numpy(batch_preds["pert_cell_counts"])
        return x_pred, x_real, {embed_key: preds}, {embed_key: reals}
    return preds, reals, None, None


def validate_stream_adatas_args(args) -> None:
    """Reject contradictory flag combinations for ``--stream-adatas``."""
    if getattr(args, "stream_adatas", False) and not getattr(args, "predict_only", False):
        raise ValueError(
            "--stream-adatas requires --predict-only because in-process "
            "cell-eval is skipped in streaming mode."
        )
    if getattr(args, "stream_adatas", False) and getattr(args, "skip_adatas", False):
        raise ValueError(
            "--stream-adatas cannot be combined with --skip-adatas: streaming "
            "exists to write the AnnData outputs to disk."
        )


class StreamingDenseH5ad:
    """Incrementally write a dense AnnData (.h5ad) one row-block at a time.

    Bounds peak host memory to a single block instead of the full
    ``(n_obs, n_vars)`` matrix. ``n_obs`` / ``n_vars`` are fixed up front; obs
    and var are written on ``close()``. The produced file is read-compatible
    with ``anndata.AnnData(X=..., obs=..., obsm=...).write_h5ad`` for the same
    data: X (and obsm) are dense float32 arrays, ``var`` carries the default
    string index, and ``obs`` is normalized (string index + strings->categoricals)
    the way ``write_h5ad`` normalizes it.
    """

    def __init__(
        self,
        path: str,
        n_obs: int,
        n_vars: int,
        *,
        dtype: npt.DTypeLike = np.float32,
        chunk_rows: int = 1024,
        obsm: dict[str, int] | None = None,
        clip: tuple[float, float] | None = (0.0, 14.0),
    ) -> None:
        self._path = path
        self._n_obs = int(n_obs)
        self._n_vars = int(n_vars)
        self._clip = clip
        self._cursor = 0
        self._closed = False

        self._file = h5py.File(path, "w")
        (
            self._file.attrs["encoding-type"],
            self._file.attrs["encoding-version"],
        ) = _ANNDATA_ENCODING

        rows = min(chunk_rows, self._n_obs) or 1
        self._x = self._file.create_dataset(
            "X",
            shape=(self._n_obs, self._n_vars),
            maxshape=(self._n_obs, self._n_vars),
            dtype=dtype,
            # A 0-row dataset cannot be chunked (chunk dim > data dim); it needs
            # no chunking anyway since nothing is streamed into it.
            chunks=(rows, self._n_vars) if self._n_obs > 0 else None,
        )
        self._x.attrs["encoding-type"], self._x.attrs["encoding-version"] = _ARRAY_ENCODING

        self._obsm: dict[str, h5py.Dataset] = {}
        if obsm:
            grp = self._file.create_group("obsm")
            grp.attrs["encoding-type"], grp.attrs["encoding-version"] = _DICT_ENCODING
            for key, ncols in obsm.items():
                ncols = int(ncols)
                ds = grp.create_dataset(
                    key,
                    shape=(self._n_obs, ncols),
                    maxshape=(self._n_obs, ncols),
                    dtype=dtype,
                    chunks=(rows, ncols) if self._n_obs > 0 else None,
                )
                ds.attrs["encoding-type"], ds.attrs["encoding-version"] = _ARRAY_ENCODING
                self._obsm[key] = ds

    def write_block(
        self,
        x_block: np.ndarray,
        obsm_blocks: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("write_block called after close()")
        x = np.asarray(x_block, dtype=self._x.dtype)
        n = x.shape[0]
        end = self._cursor + n
        if end > self._n_obs:
            raise ValueError(
                f"row overflow: writing rows up to {end} into n_obs={self._n_obs}"
            )
        if self._clip is not None:
            np.clip(x, self._clip[0], self._clip[1], out=x)
        self._x[self._cursor:end] = x
        for key, ds in self._obsm.items():
            if obsm_blocks is None or key not in obsm_blocks:
                raise ValueError(f"missing obsm block for '{key}'")
            ds[self._cursor:end] = np.asarray(obsm_blocks[key], dtype=ds.dtype)
        self._cursor = end

    def close(self, obs: pd.DataFrame, var: pd.DataFrame | None = None) -> None:
        if self._closed:
            return
        # Shrink to the number of rows actually written (e.g. --shared-only).
        if self._cursor != self._n_obs:
            self._x.resize((self._cursor, self._n_vars))
            for ds in self._obsm.values():
                ds.resize((self._cursor, ds.shape[1]))
        if len(obs) != self._cursor:
            raise ValueError(
                f"obs has {len(obs)} rows but {self._cursor} were written"
            )
        for key in _EMPTY_DICT_KEYS:
            write_elem(self._file, key, {})
        # Normalize obs exactly as AnnData.write_h5ad would (string index +
        # strings->categoricals); write_elem alone does neither. Stringify the
        # index first so anndata does not emit ImplicitModificationWarning.
        obs = obs.copy()
        obs.index = obs.index.astype(str)
        obs_norm = anndata.AnnData(
            X=np.empty((self._cursor, 0), dtype=np.float32), obs=obs
        )
        obs_norm.strings_to_categoricals()
        write_elem(self._file, "obs", obs_norm.obs)
        if var is None:
            var = pd.DataFrame(index=pd.Index([str(i) for i in range(self._n_vars)]))
        write_elem(self._file, "var", var)
        self._file.close()
        self._closed = True

    def __enter__(self) -> "StreamingDenseH5ad":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # On an exception before close(), release the file handle.
        if exc_type is not None and not self._closed:
            self._file.close()
            self._closed = True
