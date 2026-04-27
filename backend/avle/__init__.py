"""AVLE-C: Adaptive Vegetation Loss Estimator for Carbon.

End-to-end remote-sensing framework that combines:
  - U-Net vegetation segmentation (PyTorch, ResNet-34 encoder)
  - AVLE+ spatially/temporally weighted loss index
  - Random Forest carbon flux regressor (IPCC Tier-2 synthetic)
  - MLP neural-network recommendation classifier (PyTorch)
  - XGBoost scenario-conditioned projection (BAU / Mitigation + quantile CI)
  - Local EVM blockchain logging (eth-tester / py-evm)

All components are open-source, platform-independent, trainable from scratch.
"""
from .config import CONFIG  # noqa: F401
