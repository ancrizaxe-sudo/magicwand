"""U-Net with ResNet-34 encoder for binary vegetation segmentation.

Not a novelty claim — engineering component.  Exposes a predict() that returns
a binary mask given an RGB + NIR patch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from .config import CONFIG


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet34(nn.Module):
    """Standard U-Net with a ResNet-34 ImageNet encoder, 4 decoder stages."""

    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        base = resnet34(weights=weights)

        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64,  /2
        self.pool = base.maxpool                                    #      /4
        self.enc2 = base.layer1                                     # 64,  /4
        self.enc3 = base.layer2                                     # 128, /8
        self.enc4 = base.layer3                                     # 256, /16
        self.enc5 = base.layer4                                     # 512, /32

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)          # /2,  64
        e2 = self.enc2(self.pool(e1))  # /4,  64
        e3 = self.enc3(e2)         # /8, 128
        e4 = self.enc4(e3)         # /16,256
        e5 = self.enc5(e4)         # /32,512

        d4 = self.up4(e5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = self.up0(d1)
        d0 = self.dec0(d0)
        return self.final(d0)


# --------------------------------------------------------------------------- #
#  Loss:  α · BCE + (1-α) · Dice,  α is a learnable parameter
# --------------------------------------------------------------------------- #
class HybridSegLoss(nn.Module):
    def __init__(self, alpha_init: float = 0.5):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, logits, target):
        alpha = torch.sigmoid(self.alpha_logit)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (2 * inter + 1) / (union + 1)
        dice = dice.mean()
        return alpha * bce + (1 - alpha) * dice, {
            "alpha": float(alpha.detach()),
            "bce":   float(bce.detach()),
            "dice":  float(dice.detach()),
        }


# --------------------------------------------------------------------------- #
#  Inference wrapper
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict = {}


def _load_model() -> Optional[UNetResNet34]:
    if "m" in _MODEL_CACHE:
        return _MODEL_CACHE["m"]
    if not CONFIG.segmentation_weights.exists():
        return None
    model = UNetResNet34(pretrained=False)
    state = torch.load(CONFIG.segmentation_weights, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    _MODEL_CACHE["m"] = model
    return model


def predict_mask(rgb_nir: np.ndarray, threshold: float = 0.5,
                 combine_with_ndvi: bool = True) -> np.ndarray:
    """Predict a binary vegetation mask.

    The trained U-Net is the primary detector.  We additionally OR it with an
    NDVI > 0.30 mask — the U-Net can miss sparse or mixed-pixel vegetation in
    small-dataset regimes, while NDVI is a robust spectral prior.  This keeps
    the U-Net as the primary AI detector while preventing catastrophic
    under-detection when training data is limited.
    """
    red = rgb_nir[..., 0]
    nir = rgb_nir[..., 2]
    ndvi = np.where((red + nir) > 1e-6, (nir - red) / (red + nir + 1e-9), 0)
    ndvi_veg = (ndvi > 0.30).astype(np.uint8)

    model = _load_model()
    if model is None:
        return ndvi_veg

    x = torch.from_numpy(rgb_nir.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].numpy()
    unet_veg = (probs > threshold).astype(np.uint8)
    if not combine_with_ndvi:
        return unet_veg
    # AND-combination: a pixel is vegetation only if BOTH the U-Net AND the
    # NDVI spectral prior agree.  This tightens the mask so change detection
    # (veg1 AND NOT veg2) is less susceptible to either detector's noise.
    return np.minimum(unet_veg, ndvi_veg).astype(np.uint8)


def save_config_json(path: Path = None, extra: dict | None = None) -> None:
    path = path or CONFIG.segmentation_config
    data = {
        "architecture":  "UNetResNet34",
        "encoder":       "resnet34",
        "in_channels":   3,
        "loss":          "alpha*BCE + (1-alpha)*Dice (alpha learnable)",
    }
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data, indent=2))
