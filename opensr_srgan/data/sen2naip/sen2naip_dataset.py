from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import torch

from opensr_srgan.data.utils.normalizer import Normalizer


class SEN2NAIP(torch.utils.data.Dataset):
    """SEN2NAIP cross-sensor dataset loader backed by a Taco manifest."""

    def __init__(self, config: Any, phase=None, taco_file=None):
        if config is None:
            raise ValueError("SEN2NAIP requires a config object.")

        if isinstance(config, (str, Path)):
            from omegaconf import OmegaConf

            config = OmegaConf.load(config)

        data_cfg = getattr(config, "Data", None)
        if data_cfg is None:
            raise ValueError("SEN2NAIP requires config.Data.")

        if taco_file is None:
            taco_file = getattr(
                data_cfg, "sen2naip_taco_file", DEFAULT_SEN2NAIP_TACO_FILE
            )
        if phase is None:
            phase = getattr(data_cfg, "sen2naip_phase", "train")
        val_fraction = getattr(data_cfg, "sen2naip_val_fraction", 0.1)

        if not taco_file:
            raise ValueError("SEN2NAIP requires config.Data.sen2naip_taco_file.")
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
        self.normalizer = Normalizer(config)

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
    DEFAULT_SEN2NAIP_TACO_FILE = "/data1/datasets/SEN2NAIP/sen2naipv2-crosssensor.taco"
    ds = SEN2NAIP(
        config="opensr_srgan/configs/config_10m.yaml",
        phase="train",
        taco_file=DEFAULT_SEN2NAIP_TACO_FILE,
    )
