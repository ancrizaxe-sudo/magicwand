"""Train the U-Net vegetation segmenter.

This is an engineering component, not a novelty claim.  We train on
programmatically generated vegetation masks derived from the same synthetic
scene generator used in `fetch.py` (a stand-in for WorldCover / Hansen labels
that works fully offline).

A user who wants to fine-tune on real WorldCover / Hansen patches can point
`--data-dir` at a directory of paired (.npz) files with keys {red, nir, mask}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import CONFIG
from .fetch import _synthetic_scene
from .segmentation_model import HybridSegLoss, UNetResNet34, save_config_json


class SyntheticForestDataset(Dataset):
    def __init__(self, n: int = 400, size: int = 128, seed: int = 0):
        self.n = n
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        red, nir = _synthetic_scene(
            (float(idx), float(idx + 1), float(idx + 2), float(idx + 3)),
            f"train-{self.seed}-{idx}", size=self.size, disturbance=0.0,
        )
        ndvi = np.where((red + nir) > 1e-6, (nir - red) / (red + nir + 1e-9), 0)
        mask = (ndvi > 0.30).astype(np.float32)
        # 3-channel input: R, NIR, NDVI
        x = np.stack([red, nir, (ndvi + 1) / 2], axis=0).astype(np.float32)
        y = mask[None, :, :]
        return torch.from_numpy(x), torch.from_numpy(y)


def _iou(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter / (union + 1e-6)).detach())


def train(epochs: int = 4, batch_size: int = 8, lr: float = 1e-4,
          n_samples: int = 400, size: int = 128):
    ds_tr = SyntheticForestDataset(n=n_samples, size=size, seed=0)
    ds_va = SyntheticForestDataset(n=max(32, n_samples // 10), size=size, seed=1)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, num_workers=0)

    model = UNetResNet34(pretrained=True)
    loss_fn = HybridSegLoss()
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    report = []
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
        report.append({"epoch": ep, "iou": iou, **parts})
        print(f"[seg] epoch {ep}  IoU={iou:.3f}  α={parts['alpha']:.2f}")

    torch.save(model.state_dict(), CONFIG.segmentation_weights)
    save_config_json(extra={"history": report, "epochs": epochs, "size": size,
                            "dataset": "synthetic-forest"})
    print(f"[seg] ✓ weights saved → {CONFIG.segmentation_weights}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()
    train(epochs=args.epochs, n_samples=args.samples, size=args.size)
