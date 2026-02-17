import sys
import types
import runpy

import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from opensr_srgan.data import dataset_selector


class _StubExampleDataset(Dataset):
    """Simple dataset used to intercept ExampleDataset creation in tests."""

    created_args = []

    def __init__(self, folder, phase):
        # record constructor arguments for assertions
        self.__class__.created_args.append((folder, phase))
        self.phase = phase
        # give the train dataset more samples than val to ensure lengths differ
        length = 6 if phase == "train" else 3
        self._data = torch.arange(length, dtype=torch.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # return deterministic single-value tensors for DataLoader consumption
        return self._data[idx]


class _LightningDataModuleStub:
    """Minimal stand-in for ``pytorch_lightning.LightningDataModule``."""

    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def stub_lightning(monkeypatch):
    module = types.ModuleType("pytorch_lightning")
    module.LightningDataModule = _LightningDataModuleStub
    monkeypatch.setitem(sys.modules, "pytorch_lightning", module)
    return module


def _make_config(**data_kwargs):
    """Helper that builds a lightweight config namespace for the selector."""

    data_namespace = types.SimpleNamespace(**data_kwargs)
    generator_namespace = types.SimpleNamespace(scaling_factor=4)
    return types.SimpleNamespace(Data=data_namespace, Generator=generator_namespace)


def test_select_dataset_example_uses_stub(monkeypatch, capsys):
    # Patch the ExampleDataset with our stub so the selector avoids filesystem I/O.
    import opensr_srgan.data.example_data.example_dataset as example_module

    monkeypatch.setattr(example_module, "ExampleDataset", _StubExampleDataset)
    _StubExampleDataset.created_args.clear()

    config = _make_config(
        dataset_type="ExampleDataset",
        train_batch_size=2,
        val_batch_size=3,
        num_workers=0,
    )

    datamodule = dataset_selector.select_dataset(config)

    # Ensure both train and validation datasets were instantiated with the expected path.
    assert _StubExampleDataset.created_args == [
        ("example_dataset/", "train"),
        ("example_dataset/", "val"),
    ]

    # The initialization prints dataset statistics; capture the output to verify it references the selection.
    printed = capsys.readouterr().out
    assert "ExampleDataset" in printed
    assert "training samples" in printed

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 3
    assert train_loader.num_workers == 0
    assert not train_loader.persistent_workers

    first_batch = next(iter(train_loader))
    assert isinstance(first_batch, torch.Tensor)
    assert first_batch.shape[0] == 2


def test_datamodule_from_datasets_uses_defaults():
    train_ds = TensorDataset(torch.arange(10))
    val_ds = TensorDataset(torch.arange(4))

    config = _make_config(
        dataset_type="stub",
        batch_size=5,
        num_workers=1,
        prefetch_factor=3,
    )

    datamodule = dataset_selector.datamodule_from_datasets(config, train_ds, val_ds)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader.batch_size == 5
    assert val_loader.batch_size == 5
    assert train_loader.num_workers == 1
    assert train_loader.prefetch_factor == 3
    assert train_loader.persistent_workers is True
    assert val_loader.prefetch_factor == 3


def test_select_dataset_unknown_raises():
    config = _make_config(dataset_type="does-not-exist")

    with pytest.raises(NotImplementedError) as exc:
        dataset_selector.select_dataset(config)

    assert "does-not-exist" in str(exc.value)


class _StubLegacyDataset(Dataset):
    created_kwargs = []

    def __init__(self, **kwargs):
        self.__class__.created_kwargs.append(kwargs)
        self._data = torch.arange(4, dtype=torch.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _StubWorldWideDataset(Dataset):
    created_kwargs = []

    def __init__(self, **kwargs):
        self.__class__.created_kwargs.append(kwargs)
        self._data = torch.arange(5, dtype=torch.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _StubSEN2NAIPDataset(Dataset):
    created_kwargs = []

    def __init__(self, **kwargs):
        self.__class__.created_kwargs.append(kwargs)
        self._data = torch.arange(6, dtype=torch.float32)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        value = self._data[idx]
        return value.unsqueeze(0), value.unsqueeze(0)


def _install_module(monkeypatch, name, is_package=False, **attrs):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_select_dataset_s2_6b_branch(monkeypatch):
    _StubLegacyDataset.created_kwargs.clear()
    _install_module(monkeypatch, "opensr_srgan.data.SEN2_SAFE", is_package=True)
    _install_module(
        monkeypatch,
        "opensr_srgan.data.SEN2_SAFE.S2_6b_ds",
        S2SAFEDataset=_StubLegacyDataset,
    )

    config = _make_config(
        dataset_type="S2_6b",
        train_batch_size=2,
        val_batch_size=2,
        num_workers=0,
    )
    config.Generator.scaling_factor = 8

    datamodule = dataset_selector.select_dataset(config)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert len(_StubLegacyDataset.created_kwargs) == 2
    assert _StubLegacyDataset.created_kwargs[0]["phase"] == "train"
    assert _StubLegacyDataset.created_kwargs[1]["phase"] == "val"
    assert _StubLegacyDataset.created_kwargs[0]["sr_factor"] == 8
    assert _StubLegacyDataset.created_kwargs[0]["bands_keep"] == [
        "B05_20m",
        "B06_20m",
        "B07_20m",
        "B8A_20m",
        "B11_20m",
        "B12_20m",
    ]
    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2


def test_select_dataset_s2_4b_branch(monkeypatch):
    _StubLegacyDataset.created_kwargs.clear()
    _install_module(monkeypatch, "opensr_srgan.data.SEN2_SAFE", is_package=True)
    _install_module(
        monkeypatch,
        "opensr_srgan.data.SEN2_SAFE.S2_6b_ds",
        S2SAFEDataset=_StubLegacyDataset,
    )

    config = _make_config(
        dataset_type="S2_4b",
        train_batch_size=1,
        val_batch_size=3,
        num_workers=0,
    )

    datamodule = dataset_selector.select_dataset(config)
    _ = datamodule.train_dataloader()
    _ = datamodule.val_dataloader()

    assert len(_StubLegacyDataset.created_kwargs) == 2
    assert _StubLegacyDataset.created_kwargs[0]["bands_keep"] == [
        "B05_10m",
        "B04_10m",
        "B03_10m",
        "B02_10m",
    ]


def test_select_dataset_sisr_ww_branch(monkeypatch):
    _StubWorldWideDataset.created_kwargs.clear()
    _install_module(monkeypatch, "opensr_srgan.data.SISR_WW", is_package=True)
    _install_module(
        monkeypatch,
        "opensr_srgan.data.SISR_WW.SISR_WW_dataset",
        SISRWorldWide=_StubWorldWideDataset,
    )

    config = _make_config(
        dataset_type="SISR_WW",
        train_batch_size=2,
        val_batch_size=2,
        num_workers=0,
    )

    datamodule = dataset_selector.select_dataset(config)
    _ = datamodule.train_dataloader()
    _ = datamodule.val_dataloader()

    assert len(_StubWorldWideDataset.created_kwargs) == 2
    assert _StubWorldWideDataset.created_kwargs[0]["split"] == "train"
    assert _StubWorldWideDataset.created_kwargs[1]["split"] == "val"


def test_select_dataset_sen2naip_branch_wires_normalizer(monkeypatch):
    import opensr_srgan.data.utils.normalizer as normalizer_module

    _StubSEN2NAIPDataset.created_kwargs.clear()
    _install_module(monkeypatch, "opensr_srgan.data.sen2naip", is_package=True)
    _install_module(
        monkeypatch,
        "opensr_srgan.data.sen2naip.sen2naip_dataset",
        SEN2NAIP=_StubSEN2NAIPDataset,
    )

    class _NormalizerStub:
        def __init__(self, _cfg):
            pass

        def normalize(self, tensor):
            return tensor + 1

    monkeypatch.setattr(normalizer_module, "Normalizer", _NormalizerStub)

    config = _make_config(
        dataset_type="sen2naip",
        train_batch_size=2,
        val_batch_size=2,
        num_workers=0,
        sen2naip_taco_file="dummy.taco",
        sen2naip_val_fraction=0.25,
    )

    datamodule = dataset_selector.select_dataset(config)
    _ = datamodule.train_dataloader()
    _ = datamodule.val_dataloader()

    assert len(_StubSEN2NAIPDataset.created_kwargs) == 2
    train_kwargs = _StubSEN2NAIPDataset.created_kwargs[0]
    val_kwargs = _StubSEN2NAIPDataset.created_kwargs[1]
    assert train_kwargs["phase"] == "train"
    assert val_kwargs["phase"] == "val"
    assert train_kwargs["taco_file"] == "dummy.taco"
    assert val_kwargs["val_fraction"] == 0.25
    assert torch.equal(train_kwargs["normalizer"](torch.zeros(1)), torch.ones(1))


def test_dataset_selector_module_main_guard(monkeypatch):
    import opensr_srgan.data.example_data.example_dataset as example_module
    from omegaconf import OmegaConf

    monkeypatch.setattr(example_module, "ExampleDataset", _StubExampleDataset)
    monkeypatch.setattr(
        OmegaConf,
        "load",
        lambda _path: _make_config(
            dataset_type="ExampleDataset",
            train_batch_size=1,
            val_batch_size=1,
            num_workers=0,
        ),
    )

    runpy.run_module("opensr_srgan.data.dataset_selector", run_name="__main__")
