import importlib
import os
import sys
from types import SimpleNamespace

import pytest


class DummySRGAN:
    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path
        self.eval_called = False
        self.to_device = None
        self.loaded_state = None
        self.map_location = None
        self.ckpt_path = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_device = device
        return self

    def load_state_dict(self, state, strict=False):
        self.loaded_state = (state, strict)

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, map_location=None):
        instance = cls(config_file_path="from_checkpoint")
        instance.map_location = map_location
        instance.ckpt_path = ckpt_path
        return instance


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Ensure we always reload the inference module with the DummySRGAN shim
    monkeypatch.setitem(
        sys.modules,
        "opensr_srgan.model.SRGAN",
        SimpleNamespace(SRGAN_model=DummySRGAN),
    )


@pytest.fixture()
def inference_module():
    return importlib.reload(importlib.import_module("opensr_srgan.inference"))


def test_load_model_defaults_cpu(monkeypatch, inference_module):
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)

    model, device = inference_module.load_model(config_path="cfg.yaml")

    assert device == "cpu"
    assert model.config_file_path == "cfg.yaml"
    assert model.eval_called is True
    assert model.to_device == "cpu"


def test_load_model_with_lightning_checkpoint(monkeypatch, inference_module):
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)

    model, device = inference_module.load_model(
        config_path="cfg.yaml", ckpt_path="weights.ckpt"
    )

    assert device == "cpu"
    assert model.ckpt_path == "weights.ckpt"
    assert model.map_location == "cpu"
    assert model.eval_called is True
    assert model.to_device == "cpu"


def test_load_model_fallback_state_dict(monkeypatch, inference_module):
    class FallbackSRGAN(DummySRGAN):
        @classmethod
        def load_from_checkpoint(cls, *args, **kwargs):
            raise TypeError("force raw state dict")

    loaded_state = {"state_dict": {"layer": 1}}

    monkeypatch.setattr(inference_module, "SRGAN_model", FallbackSRGAN)
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference_module.torch, "load", lambda path, map_location=None: loaded_state)

    model, device = inference_module.load_model(config_path="cfg.yaml", ckpt_path="fallback.ckpt")

    assert device == "cpu"
    assert model.loaded_state == (loaded_state["state_dict"], False)
    assert model.to_device == "cpu"


def test_run_sen2_inference_invokes_pipeline(monkeypatch, inference_module):
    dummy_model = object()

    def fake_load_model(**kwargs):
        return dummy_model, "cpu"

    created_objects = {}

    class DummyProcessor:
        def __init__(self, **kwargs):
            created_objects.update(kwargs)
            self.start_called = False

        def start_super_resolution(self):
            self.start_called = True

    dummy_utils = SimpleNamespace(large_file_processing=DummyProcessor)
    monkeypatch.setitem(sys.modules, "opensr_utils", dummy_utils)
    monkeypatch.setattr(inference_module, "load_model", fake_load_model)

    result = inference_module.run_sen2_inference(
        sen2_path="/tmp/safe",
        config_path="cfg.yaml",
        ckpt_path="weights.ckpt",
        window_size=(64, 64),
        overlap=4,
        eliminate_border_px=1,
        save_preview=True,
        debug=True,
    )

    assert isinstance(result, DummyProcessor)
    assert result.start_called is True
    assert created_objects["root"] == "/tmp/safe"
    assert created_objects["model"] is dummy_model
    assert created_objects["window_size"] == (64, 64)
    assert created_objects["overlap"] == 4
    assert created_objects["eliminate_border_px"] == 1
    assert created_objects["save_preview"] is True
    assert created_objects["debug"] is True
    assert created_objects["gpus"] == []


def test_run_sen2_inference_sets_cuda_devices(monkeypatch, inference_module):
    dummy_model = object()
    monkeypatch.setattr(inference_module, "load_model", lambda **_: (dummy_model, "cuda"))

    class DummyProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.start_called = False

        def start_super_resolution(self):
            self.start_called = True

    dummy_utils = SimpleNamespace(large_file_processing=DummyProcessor)
    monkeypatch.setitem(sys.modules, "opensr_utils", dummy_utils)

    inference_module.run_sen2_inference(
        sen2_path="/tmp/safe",
        gpus=[1, 2],
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"