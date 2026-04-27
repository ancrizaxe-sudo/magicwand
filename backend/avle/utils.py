"""Shared utilities: NDVI helpers, normalization, hashing, tiling."""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def percentile_normalise(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Stretch values to [0,1] using the (low, high) percentiles; robust to outliers."""
    a = np.asarray(arr, dtype=np.float32)
    lo = np.nanpercentile(a, low)
    hi = np.nanpercentile(a, high)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def morph_open(mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    """Manual morphological opening (erode → dilate) with a square k×k kernel.

    Implemented with pure NumPy slicing so we do not need scipy / cv2.
    """
    m = mask.astype(bool)
    pad = k // 2
    # Erosion: a pixel survives only if all neighbours are True
    eroded = np.ones_like(m, dtype=bool)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
            # Zero the wrapped region
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            eroded &= shifted
    # Dilation of the eroded mask
    dilated = np.zeros_like(m, dtype=bool)
    for dy in range(-pad, pad + 1):
        for dx in range(-pad, pad + 1):
            shifted = np.roll(np.roll(eroded, dy, axis=0), dx, axis=1)
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            dilated |= shifted
    return dilated.astype(np.uint8)


def region_hash(bbox: Tuple[float, float, float, float], t1: str, t2: str) -> str:
    """Deterministic hex hash for (bbox, t1, t2) - used in blockchain logging."""
    key = f"{bbox[0]:.6f}|{bbox[1]:.6f}|{bbox[2]:.6f}|{bbox[3]:.6f}|{t1}|{t2}"
    return "0x" + hashlib.sha256(key.encode()).hexdigest()[:40]


def assign_biome(ndvi_mean: float) -> str:
    """Heuristic biome assignment from mean NDVI.  Used for synthetic labelling
    and for picking the mitigation drift β.  Not a research claim."""
    if ndvi_mean > 0.70:
        return "tropical_moist"
    if ndvi_mean > 0.55:
        return "tropical_dry"
    if ndvi_mean > 0.40:
        return "temperate_broadleaf"
    if ndvi_mean > 0.25:
        return "boreal"
    return "grassland"


def pixel_to_ha(n_pixels: int, resolution_m: float = 10.0) -> float:
    """Sentinel-2 native resolution is 10 m → 1 pixel ≈ 0.01 ha."""
    return float(n_pixels) * (resolution_m * resolution_m) / 10_000.0


# --------------------------------------------------------------------------- #
#  Loss-map cleanup — median filter + connected components + min size
# --------------------------------------------------------------------------- #
def _median_filter(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """3×3 (or k×k) median filter implemented with NumPy stride tricks."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    pad_arr = np.pad(mask.astype(np.uint8), pad, mode="edge")
    windows = sliding_window_view(pad_arr, (k, k))
    return (np.median(windows, axis=(-1, -2)) > 0.5).astype(np.uint8)


def _connected_components(mask: np.ndarray) -> tuple:
    """Flood-fill 4-connected CC labelling, pure NumPy.  Returns (labels, count)."""
    labels = np.zeros_like(mask, dtype=np.int32)
    next_label = 0
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                next_label += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if y < 0 or x < 0 or y >= H or x >= W:
                        continue
                    if visited[y, x] or not mask[y, x]:
                        continue
                    visited[y, x] = True
                    labels[y, x] = next_label
                    stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    return labels, next_label


def clean_loss_mask(raw_mask: np.ndarray,
                    open_kernel: int = 3,
                    min_cluster_px: int = 16) -> np.ndarray:
    """Full loss-map cleanup: median filter → morphological opening →
    connected-component sizing (drop clusters < `min_cluster_px`)."""
    m = (raw_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    m = _median_filter(m, k=3)
    m = morph_open(m, k=open_kernel, iters=1)
    if min_cluster_px <= 0 or m.sum() == 0:
        return m
    labels, n = _connected_components(m)
    if n == 0:
        return m
    # Count pixels per label
    counts = np.bincount(labels.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    for lab in range(1, n + 1):
        keep[lab] = counts[lab] >= min_cluster_px
    return keep[labels].astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Region suitability (WorldCover-style water/snow/non-veg gate)
# --------------------------------------------------------------------------- #
def assess_region_suitability(ndvi_t1: np.ndarray,
                              water_ndvi_fraction: float = 0.65,
                              nonveg_ndvi_fraction: float = 0.80,
                              min_mean_ndvi: float = 0.10) -> dict:
    """Heuristic region-level guard — returns {suitable, reason} dict.

    Rules (Sentinel-2 L2A NDVI at 10 m):
      - >65 % of pixels with NDVI<0 → predominantly open water
      - >80 % of pixels with NDVI<0.15 → non-vegetated (bare rock/desert/snow)
      - mean NDVI below 0.10 → nothing to track
    """
    ndvi = np.asarray(ndvi_t1, dtype=np.float32)
    total = ndvi.size
    water_frac = float((ndvi < 0).sum() / total) if total else 0.0
    nonveg_frac = float((ndvi < 0.15).sum() / total) if total else 0.0
    mean = float(np.nanmean(ndvi))
    if water_frac > water_ndvi_fraction:
        return {"suitable": False, "reason": "predominantly water",
                "water_fraction": water_frac, "mean_ndvi": mean}
    if nonveg_frac > nonveg_ndvi_fraction:
        return {"suitable": False, "reason": "predominantly non-vegetated (snow/desert/rock)",
                "nonveg_fraction": nonveg_frac, "mean_ndvi": mean}
    if mean < min_mean_ndvi:
        return {"suitable": False, "reason": f"mean NDVI {mean:.2f} < {min_mean_ndvi}",
                "mean_ndvi": mean}
    return {"suitable": True, "mean_ndvi": mean,
            "water_fraction": water_frac, "nonveg_fraction": nonveg_frac}
