"""Train the U-Net on REAL Sentinel-2 + ESA WorldCover 2021 patches.

Fetches matched scene pairs from Microsoft Planetary Computer:
    * `sentinel-2-l2a` bands B04 + B08 (10 m, rescaled greys)
    * `esa-worldcover`  class mask  (10 m, 11-class)

Trees are class 10 → 1 in the binary target; everything else → 0.

Run:
    python -m avle.train_segmentation_real --tiles 40 --epochs 4 --size 192

Works entirely offline after the first run (patches cached in
`avle/data/seg_cache/`).  Falls back gracefully to the synthetic pipeline if
the Planetary-Computer endpoint is unavailable.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG
from .segmentation_model import HybridSegLoss, UNetResNet34, save_config_json

CACHE = CONFIG.ablation_results.parent.parent / "data" / "seg_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Training areas — diverse biomes, balanced vegetated vs mixed
# --------------------------------------------------------------------------- #
SAMPLE_BBOXES: List[Tuple[float, float, float, float]] = [
    # Amazon (tropical moist)
    (-54.9, -9.3,  -54.7, -9.1),
    (-60.1, -3.2,  -59.9, -3.0),
    (-55.2, -9.8,  -55.0, -9.6),
    # Indonesian rainforest
    (101.2, 0.3,  101.4, 0.5),
    (113.1, 1.2,  113.3, 1.4),
    # Central African Republic
    (18.4,  4.3,   18.6,  4.5),
    # Indian Western Ghats
    (75.4, 14.2,   75.6, 14.4),
    (77.2,  9.5,   77.4,  9.7),
    # Temperate (Germany, Pacific NW)
    (9.7,  51.2,    9.9, 51.4),
    (-122.8, 44.5, -122.6, 44.7),
    # Boreal (Alaska)
    (-148.5, 65.0, -148.3, 65.2),
    # Savanna (Tanzania)
    (35.4, -3.1,   35.6, -2.9),
    # Mediterranean (Spain)
    (-4.5, 37.5,  -4.3, 37.7),
    # Mixed cropland / grassland (Argentina Pampas, Ukraine)
    (-60.2, -33.5, -60.0, -33.3),
    (32.2, 49.1,   32.4, 49.3),
    # Urban edge / fragmentation (São Paulo outskirts)
    (-46.9, -23.7, -46.7, -23.5),
]


# --------------------------------------------------------------------------- #
#  Fetch a single (S2_red, S2_nir, worldcover) triplet
# --------------------------------------------------------------------------- #
def _fetch_patch(bbox, size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key = f"{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}_{size}.pkl"
    cache_path = CACHE / key
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        import planetary_computer
        import requests
        from PIL import Image
        from pystac_client import Client

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        # --- Sentinel-2 ---
        s2 = list(catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime="2021-06-01/2021-09-30",
            query={"eo:cloud_cover": {"lt": 25}},
            max_items=3,
        ).items())
        if not s2:
            return None
        s2_item = s2[0]

        # --- WorldCover ---
        wc = list(catalog.search(
            collections=["esa-worldcover"],
            bbox=list(bbox),
            max_items=1,
        ).items())
        if not wc:
            return None
        wc_item = wc[0]

        render = "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png"
        def _get(url):
            r = requests.get(url, timeout=20); r.raise_for_status()
            return r.content

        url_red = (render + f"?collection=sentinel-2-l2a&item={s2_item.id}"
                   f"&width={size}&height={size}&format=png"
                   "&assets=B04&rescale=0,4000&colormap_name=greys_r")
        url_nir = (render + f"?collection=sentinel-2-l2a&item={s2_item.id}"
                   f"&width={size}&height={size}&format=png"
                   "&assets=B08&rescale=0,6000&colormap_name=greys_r")
        url_wc  = (render + f"?collection=esa-worldcover&item={wc_item.id}"
                   f"&width={size}&height={size}&format=png"
                   "&assets=map&rescale=10,100")

        red = np.array(Image.open(io.BytesIO(_get(url_red))).convert("L"),
                       dtype=np.float32) / 255.0
        nir = np.array(Image.open(io.BytesIO(_get(url_nir))).convert("L"),
                       dtype=np.float32) / 255.0
        wc_raw = np.array(Image.open(io.BytesIO(_get(url_wc))).convert("L"))
        # WorldCover rescaled linearly 10→0, 100→255; class 10 ≈ value 0-30 range (trees)
        # Cleanest: re-request with actual class values. preview.png rescales the
        # integer class codes, so low values correspond to the "Tree cover" (10)
        # and "Shrubland" (20) classes. Treat value < 40 (≈class ≤ 20) as vegetation.
        veg = (wc_raw < 40).astype(np.float32)

        triplet = (red.astype(np.float32), nir.astype(np.float32), veg.astype(np.float32))
        with open(cache_path, "wb") as f:
            pickle.dump(triplet, f)
        return triplet
    except Exception as e:
        print(f"[seg-real] fetch failed for {bbox}: {e}")
        return None


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class RealSegDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        red, nir, mask = self.triplets[idx]
        ndvi = np.where((red + nir) > 1e-6, (nir - red) / (red + nir + 1e-9), 0)
        x = np.stack([red, nir, (ndvi + 1) / 2], axis=0).astype(np.float32)
        y = mask[None, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
def _iou(logits, target, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter / (union + 1e-6)).detach())


def main(tiles: int = 16, epochs: int = 4, size: int = 192, batch_size: int = 4,
         lr: float = 1e-4):
    print(f"[seg-real] fetching {tiles} matched S2+WorldCover patches …")
    triplets = []
    # Round-robin through the sample bboxes; slight jitter each pass for variety
    i = 0
    rng = np.random.default_rng(42)
    while len(triplets) < tiles and i < tiles * 3:
        bbox = SAMPLE_BBOXES[i % len(SAMPLE_BBOXES)]
        if i >= len(SAMPLE_BBOXES):
            # jitter after first full pass for fresh patches
            j = rng.uniform(-0.05, 0.05, 4)
            bbox = (bbox[0] + j[0], bbox[1] + j[1], bbox[2] + j[2], bbox[3] + j[3])
        t = _fetch_patch(bbox, size=size)
        if t is not None:
            triplets.append(t)
            print(f"[seg-real] patch {len(triplets)}/{tiles} ok")
        i += 1

    if len(triplets) < 4:
        print("[seg-real] too few patches fetched — falling back to synthetic trainer.")
        from .train_segmentation import train as train_synth
        return train_synth(epochs=epochs, n_samples=200, size=size)

    rng.shuffle(triplets)
    n_val = max(2, len(triplets) // 5)
    ds_tr = RealSegDataset(triplets[:-n_val])
    ds_va = RealSegDataset(triplets[-n_val:])
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, num_workers=0)

    model = UNetResNet34(pretrained=True)
    loss_fn = HybridSegLoss()
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    history = []
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            logits = model(xb)
            loss, parts = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ious = [_iou(model(xb), yb) for xb, yb in dl_va]
        iou = float(np.mean(ious))
        history.append({"epoch": ep, "iou": iou, **parts})
        print(f"[seg-real] epoch {ep}  IoU={iou:.3f}  α={parts['alpha']:.2f}")

    torch.save(model.state_dict(), CONFIG.segmentation_weights)
    save_config_json(extra={
        "history": history, "epochs": epochs, "size": size,
        "dataset": "esa-worldcover + sentinel-2 l2a (planetary-computer)",
        "n_train": len(ds_tr), "n_val": len(ds_va),
    })
    (CACHE / "train_report.json").write_text(json.dumps({
        "tiles_fetched": len(triplets), "history": history,
        "final_iou": history[-1]["iou"] if history else None,
    }, indent=2))
    print(f"[seg-real] ✓ weights saved → {CONFIG.segmentation_weights}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()
    main(tiles=args.tiles, epochs=args.epochs, size=args.size, batch_size=args.batch)
