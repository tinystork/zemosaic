import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

import zemosaic_align_stack as zas


@pytest.fixture(autouse=True)
def restore_memory_query(monkeypatch):
    original = zas._query_system_memory
    yield
    monkeypatch.setattr(zas, "_query_system_memory", original)


def _make_frames(count: int, shape=(128, 128, 3)):
    return [np.full(shape, fill_value=float(i), dtype=np.float32) for i in range(count)]


def test_winsor_memory_fallback_reduces_frames(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="zemosaic.align_stack")
    monkeypatch.setattr(zas, "_query_system_memory", lambda: (3 * 1024 * 1024, 0))

    frames = _make_frames(8)
    zconfig = SimpleNamespace(
        winsor_auto_fallback_on_memory_error=True,
        winsor_min_frames_per_pass=2,
        winsor_memmap_fallback="never",
        winsor_split_strategy="sequential",
        stack_memmap_enabled=False,
        gui_memmap_enable=False,
    )

    stacked, rejected = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig,
        winsor_max_frames_per_pass=8,
        winsor_max_workers=1,
    )

    assert stacked.shape == frames[0].shape
    assert np.isfinite(stacked).all()
    assert 0.0 <= rejected <= 100.0
    assert any("stack_mem_fallback_reduce_frames" in record.getMessage() for record in caplog.records)


def test_winsor_memmap_cleanup(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="zemosaic.align_stack")
    monkeypatch.setattr(zas, "_query_system_memory", lambda: (256 * 1024, 0))

    created_paths: list[str] = []

    def _fake_named_tempfile(**kwargs):
        class _Tmp:
            def __init__(self, path):
                self.name = path
                self._fh = open(self.name, "wb+")

            def close(self):
                self._fh.close()

        path = tmp_path / f"winsor_{len(created_paths)}.dat"
        created_paths.append(str(path))
        return _Tmp(str(path))

    monkeypatch.setattr(zas.tempfile, "NamedTemporaryFile", _fake_named_tempfile)

    frames = _make_frames(6)
    zconfig = SimpleNamespace(
        winsor_auto_fallback_on_memory_error=True,
        winsor_min_frames_per_pass=2,
        winsor_memmap_fallback="auto",
        winsor_split_strategy="sequential",
        stack_memmap_enabled=True,
        gui_memmap_enable=True,
    )

    stacked, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig,
        winsor_max_frames_per_pass=0,
        winsor_max_workers=1,
    )

    assert stacked.shape == frames[0].shape
    assert any("stack_mem_fallback_memmap" in record.getMessage() for record in caplog.records)
    assert created_paths, "Expected at least one memmap file"
    for path in created_paths:
        assert not os.path.exists(path)


def test_winsor_incremental_after_array_memory_error(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="zemosaic.align_stack")
    monkeypatch.setattr(zas, "_query_system_memory", lambda: (64 * 1024 * 1024, 0))

    error = zas._NumpyArrayMemoryError("simulated failure")

    def _raise_memory_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(zas, "cpu_stack_winsorized", _raise_memory_error)

    frames = _make_frames(6)
    zconfig = SimpleNamespace(
        winsor_auto_fallback_on_memory_error=True,
        winsor_min_frames_per_pass=2,
        winsor_memmap_fallback="never",
        winsor_split_strategy="sequential",
        stack_memmap_enabled=False,
        gui_memmap_enable=False,
    )

    stacked, _ = zas.stack_winsorized_sigma_clip(
        frames,
        zconfig=zconfig,
        winsor_max_frames_per_pass=0,
        winsor_max_workers=1,
    )

    assert stacked.shape == frames[0].shape
    assert any("stack_mem_retry_after_error" in record.getMessage() for record in caplog.records)
    assert not any("stack_mem_retry_memmap" in record.getMessage() for record in caplog.records)
    assert any("stack_mem_retry_incremental" in record.getMessage() for record in caplog.records)
