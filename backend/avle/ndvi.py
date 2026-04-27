"""NDVI computation + cloud/shadow masking."""
from __future__ import annotations

import numpy as np

from .config import CONFIG


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED), clamped to [-1, 1]."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    return np.clip(ndvi, -1.0, 1.0)


def cloud_shadow_mask(red: np.ndarray, nir: np.ndarray,
                      threshold: float | None = None) -> np.ndarray:
    """Boolean mask of valid pixels: NIR+RED above a low reflectance threshold."""
    t = CONFIG.ndvi_cloud_min if threshold is None else threshold
    return (red.astype(np.float32) + nir.astype(np.float32)) >= t


def percentile_stretch(ndvi: np.ndarray,
                       low: float = 2, high: float = 98) -> np.ndarray:
    """Per-scene 2nd/98th percentile normalisation used for visualisation."""
    lo = np.nanpercentile(ndvi, low)
    hi = np.nanpercentile(ndvi, high)
    if hi - lo < 1e-9:
        return np.zeros_like(ndvi)
    return np.clip((ndvi - lo) / (hi - lo), 0.0, 1.0)
