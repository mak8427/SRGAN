import builtins
from pathlib import Path
import types

import pytest
import torch

from opensr_srgan import _factory


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.loaded_state = None
        self.ema = types.SimpleNamespace(load_state_dict=lambda state: state)
        self.is_eval = False

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = (state_dict, strict)

    def eval(self):
        self.is_eval = True
        return self


def test_maybe_download_returns_local_path(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    with _factory._maybe_download(checkpoint) as resolved:
        assert resolved == checkpoint
        assert resolved.read_bytes() == b"checkpoint"


def test_maybe_download_downloads_http(monkeypatch, tmp_path):
    target = tmp_path / "downloaded.ckpt"

    def fake_download(url, filename, progress):
        Path(filename).write_bytes(b"downloaded")

    monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)

    url = "https://example.com/weights.ckpt"
    with _factory._maybe_download(url) as resolved:
        assert resolved.exists()
        assert resolved.read_bytes() == b"downloaded"
        assert resolved.suffix == ".ckpt"


def test_load_from_config_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        _factory.load_from_config(Path("nonexistent.yaml"))


def test_load_from_config_loads_state_and_ema(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test")

    checkpoint_path = tmp_path / "state.ckpt"
    state = {"state_dict": {"weight": torch.tensor([1, 2, 3])}, "ema_state": {"w": 1}}
    torch.save(state, checkpoint_path)

    monkeypatch.setattr(_factory, "SRGAN_model", DummyModel)

    model = _factory.load_from_config(config_path, checkpoint_path)

    assert isinstance(model, DummyModel)
