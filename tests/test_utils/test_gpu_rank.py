"""Tests for GPU rank helper utilities."""

# test_is_global_zero.py
import builtins
import types
import pytest
import sys

# Import the function to be tested
from opensr_srgan.utils.gpu_rank import _is_global_zero


def test_returns_true_when_no_torch_and_no_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setitem(builtins.__dict__, "importlib", None)
    assert _is_global_zero() is True


def test_returns_true_when_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _is_global_zero() is True


def test_returns_false_when_not_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _is_global_zero() is False


def test_returns_true_when_world_size_one(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert _is_global_zero() is True


def test_prefers_torch_distributed_rank_when_initialized(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_available", lambda: True, raising=False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True, raising=False)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1, raising=False)
    assert _is_global_zero() is False


def test_distributed_errors_fall_back_to_env(monkeypatch):
    def _raise_boom():
        raise RuntimeError("boom")

    monkeypatch.setattr("torch.distributed.is_available", _raise_boom, raising=False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True, raising=False)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0, raising=False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _is_global_zero() is True
