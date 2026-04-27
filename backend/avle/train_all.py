"""One-shot training orchestrator. Run once to populate `weights/`.

Usage:
    python -m avle.train_all
"""
from __future__ import annotations

from . import train_carbon, train_prediction, train_recommendation, train_segmentation_real, evaluate


def main():
    # Real U-Net training on Sentinel-2 + ESA WorldCover
    train_segmentation_real.main(tiles=16, epochs=4, size=192, batch_size=4)
    train_carbon.main()
    train_recommendation.main(epochs=30)
    train_prediction.main()
    evaluate.run()


if __name__ == "__main__":
    main()
