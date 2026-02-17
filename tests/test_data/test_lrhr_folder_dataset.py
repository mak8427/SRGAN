from pathlib import Path

import numpy as np
import pytest
import torch

from opensr_srgan.data.lrhr_folder.lrhr_folder_dataset import LRHRFolderDataset


def _write_pair(root: Path, phase: str, name: str, lr_shape=(8, 8, 3), hr_shape=(16, 16, 3)):
    (root / phase / "LR").mkdir(parents=True, exist_ok=True)
    (root / phase / "HR").mkdir(parents=True, exist_ok=True)
    np.save(root / phase / "LR" / name, np.zeros(lr_shape, dtype=np.float32))
    np.save(root / phase / "HR" / name, np.ones(hr_shape, dtype=np.float32))


def test_lrhr_folder_dataset_reads_phase_pairs(tmp_path):
    _write_pair(tmp_path, "train", "a.npy")
    _write_pair(tmp_path, "train", "b.npy")
    _write_pair(tmp_path, "val", "v.npy")
    _write_pair(tmp_path, "test", "t.npy")

    ds = LRHRFolderDataset(root_folder=tmp_path, phase="train")
    assert len(ds) == 2

    sample = ds[0]
    assert set(sample.keys()) == {"LR", "HR", "filename"}
    assert isinstance(sample["LR"], torch.Tensor)
    assert isinstance(sample["HR"], torch.Tensor)
    assert sample["LR"].shape == (3, 8, 8)
    assert sample["HR"].shape == (3, 16, 16)


def test_lrhr_folder_dataset_invalid_phase_raises(tmp_path):
    with pytest.raises(ValueError):
        LRHRFolderDataset(root_folder=tmp_path, phase="dev")


def test_lrhr_folder_dataset_missing_pair_raises(tmp_path):
    (tmp_path / "train" / "LR").mkdir(parents=True, exist_ok=True)
    (tmp_path / "train" / "HR").mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "train" / "LR" / "only_lr.npy", np.zeros((8, 8, 1), dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        LRHRFolderDataset(root_folder=tmp_path, phase="train")


def test_lrhr_folder_dataset_applies_normalization(tmp_path):
    (tmp_path / "train" / "LR").mkdir(parents=True, exist_ok=True)
    (tmp_path / "train" / "HR").mkdir(parents=True, exist_ok=True)
    np.save(
        tmp_path / "train" / "LR" / "x.npy",
        np.full((4, 4, 1), 10000.0, dtype=np.float32),
    )
    np.save(
        tmp_path / "train" / "HR" / "x.npy",
        np.full((8, 8, 1), 10000.0, dtype=np.float32),
    )

    ds = LRHRFolderDataset(
        root_folder=tmp_path,
        phase="train",
        normalization="normalise_10k",
    )
    sample = ds[0]

    assert torch.allclose(sample["LR"], torch.ones_like(sample["LR"]))
    assert torch.allclose(sample["HR"], torch.ones_like(sample["HR"]))
