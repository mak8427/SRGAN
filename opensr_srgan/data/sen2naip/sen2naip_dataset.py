from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import rasterio as rio
import torch

from opensr_srgan.data.utils.normalizer import Normalizer


class SEN2NAIP(torch.utils.data.Dataset):
    """SEN2NAIP cross-sensor dataset loader backed by a Taco manifest."""

    def __init__(
        self,
        taco_file: str,
        phase: str = "train",
        val_fraction: float = 0.1,
        cfg: Any = None,
        normalization: str = "identity",
    ):
        if phase not in {"train", "val"}:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of: train, val.")
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(
                f"val_fraction must be in (0, 1), received {val_fraction}."
            )

        try:
            import tacoreader
        except ImportError as exc:
            raise ImportError(
                "SEN2NAIP requires 'tacoreader'. Install it via "
                "'pip install tacoreader==0.6.5'."
            ) from exc

        self.dataset = tacoreader.load(taco_file)
        if cfg is None:
            cfg = SimpleNamespace(Data=SimpleNamespace(normalization=normalization))
        self.normalizer = Normalizer(cfg)

        total = len(self.dataset)
        if total < 2:
            raise ValueError(
                "SEN2NAIP dataset requires at least 2 samples to create train/val splits."
            )
        split_idx = int(round(total * (1.0 - val_fraction)))
        split_idx = min(max(split_idx, 1), total - 1)

        if phase == "train":
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, total))

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _to_tensor(data: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(data).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        sample = self.dataset.read(self.indices[idx])
        lr_path = sample.read(0)
        hr_path = sample.read(1)

        with rio.open(lr_path) as src, rio.open(hr_path) as dst:
            lr_data = src.read()
            hr_data = dst.read()

        lr = self._to_tensor(lr_data)
        hr = self._to_tensor(hr_data)

        lr = self.normalizer.normalize(lr)
        hr = self.normalizer.normalize(hr)

        return lr, hr


if __name__ == "__main__":
    ds = SEN2NAIP(
        "/data1/datasets/SEN2NAIP/sen2naipv2-crosssensor.taco",
        phase="train",
        normalization="identity",
    )
