# Migration: HVG Gene Names Stored in AnnData Uns

## Summary

Recent versions of STATE store highly variable gene (HVG) names in `adata.uns["X_hvg_var_names"]`.
This makes it possible for downstream tools to map `adata.obsm["X_hvg"]` columns back to gene IDs.

If you have preprocessed data created before this change, you can backfill the HVG names with the
script below.

## Backward Compatibility

This change is fully backward compatible:

- **Existing preprocessed data**: Inference commands continue to work without modification. A
  non-blocking warning is emitted recommending re-preprocessing, but execution proceeds normally.
- **Existing trained models**: Model checkpoints do not depend on this uns key. Gene names are
  already captured in `var_dims.pkl` at training time.
- **Downstream code**: Code unaware of `X_hvg_var_names` simply ignores it. The obsm matrix
  structure is unchanged.

### Fallback Behavior

When `X_hvg_var_names` is absent, STATE attempts to recover gene names from
`adata.var_names[adata.var.highly_variable]`. This fallback succeeds as long as the
`highly_variable` boolean column remains in `adata.var`.

### When Gene Names Are Unrecoverable

Gene names cannot be recovered if an h5ad file has `X_hvg` in obsm but:

1. No `X_hvg_var_names` in uns, AND
2. No `highly_variable` column in var (e.g., var was subset or modified)

This edge case would already be broken prior to this change. The new feature makes the mapping
explicit rather than implicit.

## Backfill Script

For existing preprocessed files, run the following to add `X_hvg_var_names`:
```python
import anndata as ad
import numpy as np

adata = ad.read_h5ad("your_preprocessed_data.h5ad")

if "X_hvg" in adata.obsm and "X_hvg_var_names" not in adata.uns:
    if "highly_variable" in adata.var.columns:
        hvg_names = adata.var_names[adata.var.highly_variable].tolist()
        adata.uns["X_hvg_var_names"] = np.array(hvg_names, dtype=object)
        adata.write_h5ad("your_preprocessed_data.h5ad")
        print(f"Added {len(hvg_names)} HVG names to uns")
    else:
        print("Cannot backfill: 'highly_variable' column not found in adata.var")
else:
    print("Backfill not needed or X_hvg not present")
```

## Notes

- The uns key is stored as a NumPy array of Python strings for h5ad compatibility.
- Re-running `state tx preprocess_train` with the latest version will populate this automatically.
- The naming convention `{obsm_key}_var_names` allows for multiple obsm matrices with associated
  gene names (e.g., `X_pca_var_names` if needed in the future).