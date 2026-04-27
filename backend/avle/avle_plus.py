"""AVLE+  – spatially / temporally weighted vegetation-loss index.

    AVLE+ =  Σ  Loss_mask(x,y) · ΔNDVI(x,y) · S(x,y) · T(x,y) · C(x,y)

Components
----------
S(x,y) : Spatial cluster weight  (K-means on NDVI_t1, k = 5)
T(x,y) : Temporal momentum       (ΔNDVI saturated at 0.3 for 2-date case)
C(x,y) : Canopy-density proxy    (5th/95th percentile stretch of NDVI_t1)

This is Novelty Claim #1 of the paper.  Fully vectorised NumPy, no external
dependencies beyond scikit-learn for K-means.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from .config import CONFIG


# --------------------------------------------------------------------------- #
#  Spatial weight
# --------------------------------------------------------------------------- #
def spatial_cluster_weight(ndvi_t1: np.ndarray,
                           k: int | None = None,
                           random_state: int = 42):
    """Return per-pixel spatial weight S(x,y) ∈ [0,1] and the cluster id map."""
    k = CONFIG.kmeans_k if k is None else k
    flat = ndvi_t1.reshape(-1, 1).astype(np.float32)
    # Sub-sample for faster fitting on huge rasters
    n_fit = min(len(flat), 20_000)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(flat), size=n_fit, replace=False)
    km = KMeans(n_clusters=k, n_init=5, random_state=random_state).fit(flat[idx])
    labels = km.predict(flat).reshape(ndvi_t1.shape)
    cluster_means = np.array(
        [ndvi_t1[labels == c].mean() if (labels == c).any() else 0.0
         for c in range(k)]
    )
    norm = cluster_means.max() if cluster_means.max() > 1e-9 else 1.0
    weight_table = cluster_means / norm
    S = weight_table[labels]
    return S.astype(np.float32), labels.astype(np.int16)


# --------------------------------------------------------------------------- #
#  Temporal weight
# --------------------------------------------------------------------------- #
def temporal_momentum(delta_ndvi: np.ndarray,
                      saturation: float | None = None) -> np.ndarray:
    s = CONFIG.temporal_saturation if saturation is None else saturation
    T = np.clip(delta_ndvi / s, 0.0, 1.0)
    return T.astype(np.float32)


def temporal_momentum_multidate(ndvi_series: np.ndarray) -> np.ndarray:
    """Multi-date variant: normalised negative slope per pixel.

    ndvi_series: array of shape (T, H, W) with at least 3 time-steps.
    """
    assert ndvi_series.ndim == 3 and ndvi_series.shape[0] >= 2
    T, H, W = ndvi_series.shape
    x = np.arange(T).reshape(-1, 1, 1)
    x_mean = x.mean()
    y_mean = ndvi_series.mean(axis=0, keepdims=True)
    num = ((x - x_mean) * (ndvi_series - y_mean)).sum(axis=0)
    den = ((x - x_mean) ** 2).sum()
    slope = num / max(den, 1e-9)
    neg_slope = np.clip(-slope, 0.0, None)
    if neg_slope.max() < 1e-9:
        return np.zeros_like(neg_slope, dtype=np.float32)
    return (neg_slope / neg_slope.max()).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Canopy density
# --------------------------------------------------------------------------- #
def canopy_density(ndvi_t1: np.ndarray) -> np.ndarray:
    p5 = np.nanpercentile(ndvi_t1, 5)
    p95 = np.nanpercentile(ndvi_t1, 95)
    if p95 - p5 < 1e-9:
        return np.zeros_like(ndvi_t1, dtype=np.float32)
    C = (ndvi_t1 - p5) / (p95 - p5)
    return np.clip(C, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
#  AVLE+ aggregator
# --------------------------------------------------------------------------- #
def avle_plus(loss_mask: np.ndarray,
              ndvi_t1:   np.ndarray,
              ndvi_t2:   np.ndarray,
              delta_ndvi: np.ndarray) -> Dict[str, float | np.ndarray]:
    """Full AVLE+ computation.  Returns the aggregated score + component maps."""
    S, cluster_map = spatial_cluster_weight(ndvi_t1)
    T = temporal_momentum(delta_ndvi)
    C = canopy_density(ndvi_t1)

    per_pixel = loss_mask.astype(np.float32) * delta_ndvi * S * T * C
    score = float(per_pixel.sum())

    # Area-normalised score per km² (10 m pixels → 1 px = 100 m²)
    pixel_area_km2 = 1e-4  # 100 m² = 1e-4 km²
    total_km2 = loss_mask.size * pixel_area_km2
    score_per_km2 = score / max(total_km2, 1e-6)

    return {
        "avle_plus":            score,
        "avle_plus_per_km2":    score_per_km2,
        "spatial_weight":       S,
        "temporal_weight":      T,
        "canopy_density":       C,
        "cluster_map":          cluster_map,
        "per_pixel":            per_pixel,
        "loss_pixels":          int(loss_mask.sum()),
        "total_pixels":         int(loss_mask.size),
    }
