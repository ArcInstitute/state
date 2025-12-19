"""Helpers for retrieving and validating HVG gene names."""

from __future__ import annotations

import logging
from anndata import AnnData

from state.tx.constants import HVG_VAR_NAMES_KEY


logger = logging.getLogger(__name__)


def get_hvg_var_names(adata: AnnData, obsm_key: str = "X_hvg") -> list[str] | None:
    """Return HVG gene names for an embedding.

    Args:
        adata: AnnData to inspect.
        obsm_key: Embedding key to resolve gene names for.

    Returns:
        List of gene names if available, otherwise None.
    """
    version = detect_preprocessing_version(adata, obsm_key=obsm_key)
    if version in {"legacy_uns", "var_only"}:
        logger.warning(
            "Detected legacy HVG metadata for %s. Consider re-running preprocess_train with the latest STATE version.",
            obsm_key,
        )

    derived_key = f"{obsm_key}_var_names"
    if derived_key in adata.uns:
        logger.info("Using HVG var names from adata.uns['%s']", derived_key)
        return list(adata.uns[derived_key])

    if HVG_VAR_NAMES_KEY in adata.uns:
        logger.info("Using HVG var names from adata.uns['%s']", HVG_VAR_NAMES_KEY)
        return list(adata.uns[HVG_VAR_NAMES_KEY])

    if "highly_variable" in adata.var:
        logger.info("Using HVG var names from adata.var['highly_variable']")
        return adata.var_names[adata.var["highly_variable"]].tolist()

    logger.info("No HVG var names available for adata.obsm['%s']", obsm_key)
    return None


def detect_preprocessing_version(adata: AnnData, obsm_key: str = "X_hvg") -> str:
    """Detect the preprocessing metadata format based on uns/var keys.

    Args:
        adata: AnnData to inspect.
        obsm_key: Embedding key to resolve gene names for.

    Returns:
        One of: "current", "legacy_uns", "var_only", or "unknown".
    """
    derived_key = f"{obsm_key}_var_names"
    if derived_key in adata.uns:
        return "current"
    if HVG_VAR_NAMES_KEY in adata.uns:
        return "legacy_uns"
    if "highly_variable" in adata.var:
        return "var_only"
    return "unknown"


def validate_hvg_var_names(adata: AnnData, obsm_key: str = "X_hvg") -> bool:
    """Validate whether HVG gene names can be resolved for an embedding.

    Args:
        adata: AnnData to inspect.
        obsm_key: Embedding key to validate gene names for.

    Returns:
        True when gene names are available, otherwise False.
    """
    return get_hvg_var_names(adata, obsm_key=obsm_key) is not None
