# AVLE-C — Model Training Guide

## 0. One-liner
```bash
python -m avle.train_all
```
Trains every model in sequence: real U-Net → Random Forest → MLP (with MC-dropout) → XGBoost (BAU + Mitigation + quantile CI + ARIMA baseline) → full ablation with strong baselines + segmentation IoU.

## 1. U-Net — vegetation segmentation
### Architecture
* Encoder: ResNet-34 (ImageNet pretrained)
* Decoder: 4 up-sampling stages with skip connections
* Output: 1 × H × W logits → sigmoid → binary vegetation mask

### Dataset
* **Inputs**: Sentinel-2 L2A bands B04 (red, 665 nm) + B08 (NIR, 842 nm)
* **Labels**: ESA WorldCover 2021 (class 10 = Tree cover, class 20 = Shrubland → 1; all other classes → 0)
* Both fetched via Microsoft Planetary Computer STAC at 10 m.

### Training
* Loss: `α·BCE + (1−α)·Dice` with α learnable (sigmoid(logit), init 0.5)
* Optimiser: AdamW, lr = 1e-4, weight_decay = 1e-4
* Scheduler: CosineAnnealingLR, T_max = epochs
* Augmentation: the fetched tiles themselves are diverse (Amazon, Indian Western Ghats, Sumatra, Congo Basin, Pacific NW, Boreal Alaska, Mediterranean, Pampas, urban edge São Paulo), so explicit augmentation is minimal.

### Output
* `avle/weights/segmentation_unet.pth` — model state dict
* `avle/weights/segmentation_config.json` — architecture + history + IoU

### Command
```bash
python -m avle.train_segmentation_real --tiles 16 --epochs 4 --size 192
```
Caches fetched tiles to `avle/data/seg_cache/` so subsequent runs are offline.

## 2. Random Forest — carbon regressor
Trained on 50 k synthetic samples with the Myneni/Baccini NDVI^1.8 × biome-ceiling AGB relation and IPCC carbon fraction 0.47. See `carbon_model.py`.

```bash
python -m avle.train_carbon
```

## 3. MLP — recommendation classifier
30 k synthetic samples, boundary σ = 0.15, BatchNorm + Dropout, early stopping. MC-dropout inference: 25 forward passes with Dropout re-enabled at inference time (BatchNorm kept in eval mode).

```bash
python -m avle.train_recommendation
```

## 4. XGBoost — projection models
Four boosters on lag-feature sequences: BAU, Mitigation (drift β negative), 2.5 % and 97.5 % quantile CIs. Linear-regression baseline computed in-line; ARIMA(p,d,q) auto-AIC baseline computed during training.

```bash
python -m avle.train_prediction
```

## 5. Ablation
```bash
python -m avle.evaluate
```
Generates `avle/results/ablation.json` consumed by `/api/ablation` and the `/ablation` page.

## 6. Expected numbers (CPU-only)
* U-Net: IoU ~0.4–0.6 on 16 real patches (deliberately modest dataset for reproducibility)
* RF: R² ≥ 0.95, RMSE < 500 tCO₂
* MLP: accuracy ~80 %, F1 ~0.79 (rule baseline ~0.75)
* XGBoost: RMSE ~7–9; SSS modest, strengthened by intervention factor
