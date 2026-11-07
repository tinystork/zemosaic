"""Regression witnesses for truthful GPU telemetry and publication alerts."""

from __future__ import annotations

from zemosaic.core.zegrid import gpu_memory
from zemosaic.core.zegrid import run_publication


class _Device:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Runtime:
    @staticmethod
    def memGetInfo():
        return 100, 200


class _Cuda:
    runtime = _Runtime()

    @staticmethod
    def Device(_index):
        return _Device()


class _Pool:
    @staticmethod
    def used_bytes():
        return 10

    @staticmethod
    def total_bytes():
        return 20

    @staticmethod
    def free_bytes():
        return 10


class _PinnedAbsent:
    pass


class _PinnedAvailable:
    used_bytes = 3

    @staticmethod
    def total_bytes():
        return 7


class _PinnedRaising:
    @staticmethod
    def used_bytes():
        raise RuntimeError("counter unavailable")

    total_bytes = 7


class _FakeCupy:
    __name__ = "cupy"
    cuda = _Cuda()

    def __init__(self, pinned):
        self._pinned = pinned

    @staticmethod
    def get_default_memory_pool():
        return _Pool()

    def get_default_pinned_memory_pool(self):
        return self._pinned


def test_pinned_pool_absent_counters_are_truthful_none_without_errors():
    snap = gpu_memory.device_memory_snapshot(_FakeCupy(_PinnedAbsent()))

    assert snap["pinned_pool_used_bytes"] is None
    assert snap["pinned_pool_total_bytes"] is None
    assert not any(token.startswith("pinned_pool_") for token in snap["errors"])


def test_pinned_pool_available_callable_and_value_counters_are_recorded():
    snap = gpu_memory.device_memory_snapshot(_FakeCupy(_PinnedAvailable()))

    assert snap["pinned_pool_used_bytes"] == 3
    assert snap["pinned_pool_total_bytes"] == 7
    assert snap["errors"] == []


def test_pinned_pool_present_raising_counter_stays_visible():
    snap = gpu_memory.device_memory_snapshot(_FakeCupy(_PinnedRaising()))

    assert snap["pinned_pool_used_bytes"] is None
    assert snap["pinned_pool_total_bytes"] == 7
    assert "pinned_pool_used_bytes:RuntimeError" in snap["errors"]


def test_products_text_distinguishes_auxiliary_log_from_delivery_failure():
    payload = {
        "run_id": "run-1",
        "generation": "gen-1",
        "generation_dir": "/archive/gen-1",
        "produced_unix_ns": 1,
        "publication_sha256": "abc",
        "products": {
            run_publication.ROLE_SCIENCE: {
                "path": "/archive/gen-1/science.fits",
                "nbytes": 11,
            },
            run_publication.ROLE_RUN_LOG: {
                "path": "/archive/gen-1/run.log",
                "nbytes": 12,
            },
        },
        "root_files": {},
    }

    text = run_publication._products_text(payload)
    science_line = next(line for line in text.splitlines() if line.startswith("science"))
    run_log_line = next(line for line in text.splitlines() if line.startswith("run_log"))

    assert "root delivery failed" in science_line
    assert "root delivery failed" not in run_log_line
    assert "live auxiliary at output root" in run_log_line
