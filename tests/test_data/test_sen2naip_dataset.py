import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opensr_srgan.data.sen2naip.sen2naip_dataset import SEN2NAIP


class _FakeSample:
    def read(self, idx):
        return "lr.tif" if idx == 0 else "hr.tif"


class _FakeTacoDataset:
    def __len__(self):
        return 4

    def read(self, _idx):
        return _FakeSample()


class _FakeRaster:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if self.path == "lr.tif":
            return np.full((1, 2, 2), 5000.0, dtype=np.float32)
        return np.full((1, 2, 2), 10000.0, dtype=np.float32)


@pytest.fixture(autouse=True)
def _stub_external_modules(monkeypatch):
    tacoreader_mod = types.ModuleType("tacoreader")
    tacoreader_mod.load = lambda _path: _FakeTacoDataset()
    monkeypatch.setitem(sys.modules, "tacoreader", tacoreader_mod)

    rasterio_mod = types.ModuleType("rasterio")
    rasterio_mod.open = lambda path: _FakeRaster(path)
    monkeypatch.setitem(sys.modules, "rasterio", rasterio_mod)


def test_sen2naip_uses_config_normalization():
    cfg = SimpleNamespace(
        Data=SimpleNamespace(
            normalization="normalise_10k",
            sen2naip_taco_file="/tmp/fake.taco",
            sen2naip_phase="train",
            sen2naip_val_fraction=0.25,
        )
    )
    ds = SEN2NAIP(cfg)

    lr, hr = ds[0]

    assert ds.normalizer.method == "normalise_10k"
    assert isinstance(lr, torch.Tensor)
    assert isinstance(hr, torch.Tensor)
    assert torch.allclose(lr, torch.full_like(lr, 0.5))
    assert torch.allclose(hr, torch.full_like(hr, 1.0))


def test_sen2naip_requires_taco_file():
    cfg = SimpleNamespace(
        Data=SimpleNamespace(
            normalization="identity",
            sen2naip_phase="train",
            sen2naip_val_fraction=0.25,
        )
    )

    with pytest.raises(ValueError, match="sen2naip_taco_file"):
        SEN2NAIP(cfg)
