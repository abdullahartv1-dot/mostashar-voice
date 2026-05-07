"""Tests for ModelManager — lazy load + LRU eviction."""
import pytest
from engine.model_manager import ModelManager


class FakeModel:
    """Stand-in for an ML model — has a vram footprint."""
    def __init__(self, name: str, vram_gb: float):
        self.name = name
        self.vram_gb = vram_gb


def loader_factory(name: str, vram_gb: float):
    """Returns a function that creates a FakeModel; tracks call count."""
    calls = {"n": 0}
    def load():
        calls["n"] += 1
        return FakeModel(name, vram_gb)
    return load, calls


def test_first_get_loads_model():
    mm = ModelManager(vram_budget_gb=20)
    loader, calls = loader_factory("whisper-large", 3)
    model = mm.get("whisper-large", loader, vram_gb=3)
    assert model.name == "whisper-large"
    assert calls["n"] == 1


def test_second_get_uses_cached():
    mm = ModelManager(vram_budget_gb=20)
    loader, calls = loader_factory("whisper-large", 3)
    mm.get("whisper-large", loader, vram_gb=3)
    mm.get("whisper-large", loader, vram_gb=3)
    assert calls["n"] == 1


def test_evicts_lru_when_over_budget():
    mm = ModelManager(vram_budget_gb=10)
    l1, c1 = loader_factory("a", 6)
    l2, c2 = loader_factory("b", 6)  # 6+6=12 > 10, must evict a
    mm.get("a", l1, vram_gb=6)
    mm.get("b", l2, vram_gb=6)
    assert "a" not in mm.loaded
    assert "b" in mm.loaded


def test_get_after_eviction_reloads():
    mm = ModelManager(vram_budget_gb=10)
    l1, c1 = loader_factory("a", 6)
    l2, c2 = loader_factory("b", 6)
    mm.get("a", l1, vram_gb=6)
    mm.get("b", l2, vram_gb=6)  # evicts a
    mm.get("a", l1, vram_gb=6)  # reloads a, evicts b
    assert c1["n"] == 2
    assert "a" in mm.loaded
    assert "b" not in mm.loaded
