import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zemosaic_worker import _safe_load_cache


def test_create_master_tile_memmap_fallbacks(monkeypatch):
    """
    Test that _safe_load_cache handles multiple memmap failures gracefully
    by simulating several OSError(1455) failures and verifying that fallback logs
    are emitted and the function ultimately returns data without crashing.
    """
    calls_per_file = {}
    logs = []

    def fake_load(path, allow_pickle=False, mmap_mode=None):
        """Simulate memmap failures for some files and success for others."""
        if path not in calls_per_file:
            calls_per_file[path] = 0
        calls_per_file[path] += 1

        # Simulate first 3 files failing once with WinError 1455, then succeeding
        file_num = len(calls_per_file) % 5
        if file_num < 3 and calls_per_file[path] == 1:
            # First call for first 3 files: fail with WinError 1455
            raise OSError("WinError 1455 simulated")
        
        # Return valid array on retry or for other files
        return np.ones((10, 10, 3), dtype=np.float32)

    def pcb(key, prog=None, lvl=None, **kwargs):
        logs.append((key, kwargs))

    monkeypatch.setattr(np, "load", fake_load)

    # Test loading multiple files, simulating fallbacks
    for i in range(5):
        arr = _safe_load_cache(f"cache_{i}.npy", pcb=pcb, tile_id=1)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 10, 3)

    # Verify fallback logs were emitted
    fallback_logs = [entry for entry in logs if entry[0] == "stack_mem_fallback_memmap_to_ram"]
    assert len(fallback_logs) >= 3, f"Expected at least 3 fallback logs, got {len(fallback_logs)}"
