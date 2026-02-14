import torch
import numpy as np

from opensr_srgan.utils.tensor_conversions import tensor_to_numpy


def test_tensor_to_numpy_handles_non_contiguous():
    tensor = torch.arange(12, dtype=torch.float32).view(3, 4).t()  # non-contiguous
    assert not tensor.is_contiguous()

    result = tensor_to_numpy(tensor)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 3)
    np.testing.assert_array_equal(result, tensor.contiguous().numpy())


def test_tensor_to_numpy_fallback_on_missing_bindings(monkeypatch):
    tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

    def _raise_numpy(_self):
        raise RuntimeError("Numpy is not available")

    monkeypatch.setattr(torch.Tensor, "numpy", _raise_numpy)

    result = tensor_to_numpy(tensor)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.int32))


def test_tensor_to_numpy_reraises_unrelated_runtime_errors(monkeypatch):
    tensor = torch.tensor([1.0], dtype=torch.float32)

    def _raise_other(_self):
        raise RuntimeError("other failure")

    monkeypatch.setattr(torch.Tensor, "numpy", _raise_other)

    with np.testing.assert_raises_regex(RuntimeError, "other failure"):
        tensor_to_numpy(tensor)
