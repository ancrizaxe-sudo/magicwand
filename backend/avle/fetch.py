"""Sentinel-2 L2A retrieval via Microsoft Planetary Computer STAC.

Strategy:  use the small server-side ``rendered_preview`` asset (returns a
pre-coloured RGB/NIR PNG at a user-specified resolution) so we never pull
a full 10 980 × 10 980 native-resolution COG.  Fallback: deterministic
synthetic scene generator so the pipeline always runs end-to-end.
"""
from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Synthetic fallback — deterministic per (bbox, date)
# --------------------------------------------------------------------------- #
def _synthetic_scene(bbox: Tuple[float, float, float, float],
                     date_iso: str,
                     size: int = 256,
                     disturbance: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    seed = abs(hash(f"{bbox}|{date_iso}")) % (2**32)
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, 1, (size, size))
    f = np.fft.fft2(noise)
    fx, fy = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size))
    mask = np.exp(-((fx**2 + fy**2) / 0.001))
    low = np.real(np.fft.ifft2(f * mask))
    low = (low - low.min()) / (low.max() - low.min() + 1e-9)

    nir = 0.25 + 0.55 * low + rng.normal(0, 0.02, (size, size))
    red = 0.08 + 0.12 * (1 - low) + rng.normal(0, 0.01, (size, size))

    if disturbance > 0:
        cy, cx = size // 2 + int(rng.integers(-40, 40)), size // 2 + int(rng.integers(-40, 40))
        r = int(30 + disturbance * 70)
        yy, xx = np.ogrid[:size, :size]
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        red[patch] += 0.10 * disturbance
        nir[patch] -= 0.35 * disturbance

    return (np.clip(red, 0, 1).astype(np.float32),
            np.clip(nir, 0, 1).astype(np.float32))


# --------------------------------------------------------------------------- #
#  Real fetch — STAC + server-side render (small, fast)
# --------------------------------------------------------------------------- #
def _try_planetary_computer(bbox, date_start, date_end, size=256, timeout=20):
    try:
        import planetary_computer
        import requests
        from PIL import Image
        from pystac_client import Client

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": 30}},
            max_items=3,
        )
        items = list(search.items())
        if not items:
            return None
        item = items[0]

        # Use server-side rendered previews: natural colour (R G B) and NIR-red-green
        # "data:image/png" API returns a rescaled PNG at user-specified pixel size.
        render_root = "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
        common = (
            f"?collection=sentinel-2-l2a&item={item.id}"
            f"&width={size}&height={size}&format=png"
        )
        # Pull each band as a SINGLE-BAND greyscale PNG (no gamma / saturation
        # distortion) so ratios such as NDVI remain physically meaningful.
        url_red = render_root + common + "&assets=B04&rescale=0,4000&colormap_name=greys_r"
        url_nir = render_root + common + "&assets=B08&rescale=0,6000&colormap_name=greys_r"

        r1 = requests.get(url_red, timeout=timeout)
        r2 = requests.get(url_nir, timeout=timeout)
        r1.raise_for_status(); r2.raise_for_status()

        img_red = np.array(Image.open(io.BytesIO(r1.content)).convert("L"),
                           dtype=np.float32) / 255.0
        img_nir = np.array(Image.open(io.BytesIO(r2.content)).convert("L"),
                           dtype=np.float32) / 255.0
        return img_red.astype(np.float32), img_nir.astype(np.float32)
    except Exception as e:
        log.warning("Planetary-Computer fetch failed: %s", e)
        return None


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def fetch_scene(bbox: Tuple[float, float, float, float],
                date_start: str,
                date_end: str,
                size: int = 256,
                disturbance: float = 0.0,
                allow_real: bool = True,
                timeout: int = 15,
                ) -> Tuple[np.ndarray, np.ndarray, str]:
    if allow_real:
        result = _try_planetary_computer(bbox, date_start, date_end,
                                         size=size, timeout=timeout)
        if result is not None:
            red, nir = result
            return red, nir, "planetary-computer"
    red, nir = _synthetic_scene(bbox, f"{date_start}:{date_end}", size,
                                 disturbance=disturbance)
    return red, nir, "synthetic"
