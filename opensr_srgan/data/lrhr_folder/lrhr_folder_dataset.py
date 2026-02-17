from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import Dataset

from opensr_srgan.data.utils.normalizer import Normalizer


class LRHRFolderDataset(Dataset):
    """Dataset for paired LR/HR files in a fixed split-by-folder layout.

    Expected structure:

    root_folder/
      train/
        LR/
          image_001.npy
          image_002.npy
        HR/
          image_001.npy
          image_002.npy
      val/
        LR/
        HR/
      test/
        LR/
        HR/

    Notes:
    - The selected ``phase`` must be one of: ``train``, ``val``, ``test``.
    - Files are paired by exact filename between ``LR`` and ``HR``.
    - Every LR file in the selected phase must have a matching HR file.
    - Returned sample format is:
      ``{"LR": torch.Tensor, "HR": torch.Tensor, "filename": str}``.
    - Optional ``normalization`` accepts the same values used in the project
      config (e.g. ``normalise_10k``, ``normalise_10k_signed``,
      ``normalise_s2``, ``sen2_stretch``, ``zero_one_signed``).
    """

    VALID_PHASES = {"train", "val", "test"}

    def __init__(self, root_folder, phase="train", normalization="identity"):
        self.root_folder = Path(root_folder)
        self.phase = phase
        self.normalization = normalization

        cfg = SimpleNamespace(Data=SimpleNamespace(normalization=self.normalization))
        self.normalizer = Normalizer(cfg)

        if phase not in self.VALID_PHASES:
            raise ValueError(
                f"Unknown phase '{phase}'. Expected one of {sorted(self.VALID_PHASES)}."
            )
        if not self.root_folder.is_dir():
            raise FileNotFoundError(f"Dataset root folder '{self.root_folder}' does not exist.")

        phase_dir = self.root_folder / phase
        self.lr_dir = phase_dir / "LR"
        self.hr_dir = phase_dir / "HR"

        if not self.lr_dir.is_dir():
            raise FileNotFoundError(f"Missing LR folder: '{self.lr_dir}'.")
        if not self.hr_dir.is_dir():
            raise FileNotFoundError(f"Missing HR folder: '{self.hr_dir}'.")

        lr_files = sorted([p for p in self.lr_dir.iterdir() if p.is_file()])
        self.pairs = []
        for lr_path in lr_files:
            hr_path = self.hr_dir / lr_path.name
            if not hr_path.is_file():
                raise FileNotFoundError(
                    f"Missing HR counterpart for '{lr_path.name}' in '{self.hr_dir}'."
                )
            self.pairs.append((lr_path, hr_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired files found in '{self.lr_dir}' and '{self.hr_dir}'."
            )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load_array(path):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
        elif suffix == ".npz":
            with np.load(path) as z:
                arr = z[list(z.files)[0]]
        elif suffix in {".tif", ".tiff"}:
            with rasterio.open(path) as src:
                arr = src.read()
                arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC for unified conversion below
        else:
            arr = np.array(Image.open(path))
        return arr

    @staticmethod
    def _to_chw_tensor(arr):
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}.")

        # Heuristic: if first dim is tiny and later dims are spatial, treat as CHW already.
        if not (arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16):
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        return torch.from_numpy(arr.astype(np.float32))

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr = self._to_chw_tensor(self._load_array(lr_path))
        hr = self._to_chw_tensor(self._load_array(hr_path))
        lr = self.normalizer.normalize(lr)
        hr = self.normalizer.normalize(hr)

        return {
            "LR": lr,
            "HR": hr,
            "filename": lr_path.name,
        }
