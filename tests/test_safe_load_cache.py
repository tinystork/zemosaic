import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zemosaic_worker import _safe_load_cache


def test_safe_load_cache_handles_winerror(monkeypatch):
    calls = {"count": 0}
    logs = []

    def fake_load(path, allow_pickle=False, mmap_mode=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("WinError 1455 simulated")
        return np.ones((2, 2, 3), dtype=np.float32)

    def pcb(key, prog=None, lvl=None, **kwargs):
        logs.append((key, kwargs))

    monkeypatch.setattr(np, "load", fake_load)

    arr = _safe_load_cache("dummy.npy", pcb=pcb, tile_id=7)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2, 3)
    # Ensure fallback log was emitted
    assert any(entry[0] == "stack_mem_fallback_memmap_to_ram" for entry in logs)
