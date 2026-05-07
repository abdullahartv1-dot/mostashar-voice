"""Lazy-loading model manager with LRU eviction.

Models are loaded on first request and cached in memory. When loading a new
model would exceed the VRAM budget, the least-recently-used model is evicted
to free space.
"""
import logging
import time
from collections import OrderedDict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, vram_budget_gb: float):
        self.vram_budget_gb = vram_budget_gb
        # Ordered dict: oldest first, newest last (move_to_end on access)
        self.loaded: "OrderedDict[str, dict]" = OrderedDict()

    def _current_vram_gb(self) -> float:
        return sum(entry["vram_gb"] for entry in self.loaded.values())

    def _evict_lru(self) -> None:
        """Remove the oldest entry to free VRAM."""
        if not self.loaded:
            return
        name, entry = self.loaded.popitem(last=False)
        logger.info(f"Evicting LRU model: {name} ({entry['vram_gb']} GB)")
        # Free the model's GPU memory
        del entry["model"]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get(self, name: str, loader: Callable[[], Any], vram_gb: float) -> Any:
        """Get a model by name; load it if not already in memory.

        Args:
            name: unique tool name (e.g. "whisper-large-v3")
            loader: zero-arg function that returns the model object
            vram_gb: estimated VRAM footprint in gigabytes

        Returns:
            The model object.
        """
        # Cache hit: move to end (mark as most recently used)
        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]["model"]

        # Cache miss: ensure budget allows loading
        while self._current_vram_gb() + vram_gb > self.vram_budget_gb:
            if not self.loaded:
                # Single model exceeds budget — load anyway, log warning
                logger.warning(
                    f"Model {name} ({vram_gb} GB) exceeds VRAM budget "
                    f"({self.vram_budget_gb} GB) — loading anyway"
                )
                break
            self._evict_lru()

        t0 = time.time()
        model = loader()
        load_time = time.time() - t0
        logger.info(f"Loaded model {name} in {load_time:.1f}s ({vram_gb} GB)")

        self.loaded[name] = {"model": model, "vram_gb": vram_gb, "loaded_at": time.time()}
        return model
