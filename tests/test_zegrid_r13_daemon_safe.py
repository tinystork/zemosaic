"""ZM-ZEGRID-R13 — daemon-safe parallel map regression guard.

This file reproduces the PRODUCTION crash that broke a real user:

    AssertionError: daemonic processes are not allowed to have children

The product runs the ZeGrid worker inside a DAEMONIC process
(``run_hierarchical_mosaic_process``).  ``multiprocessing`` forbids a daemonic
process from starting child processes (the assert fires in ``BaseProcess.start()``
for EVERY start method — fork, spawn and forkserver alike), so the R12
``ProcessPoolExecutor`` path in :func:`zemosaic.core.zegrid.parallel.pmap`
crashed the whole run ~8 min in.  R12 tests never caught it because pytest runs
in a NON-daemonic process.

These tests run ``pmap`` (and the real gauge) INSIDE a genuinely daemonic child
process, which reproduces the production context even on Linux (the daemonic
assert is start-method agnostic).
"""

from __future__ import annotations

import logging
import multiprocessing

import numpy as np
import pytest

from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid.executor import ExecutorConfig


def _square(x):
    """Module-level (picklable) pure worker."""
    return x * x


class _CaptureHandler(logging.Handler):
    """Collects formatted log messages (used across a fork'd child boundary)."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


_PARALLEL_LOGGER = "zemosaic.core.zegrid.parallel"


def _pmap_child_entry(q, tasks):
    """Runs ``pmap`` inside a daemonic child and reports the outcome via ``q``."""
    assert multiprocessing.current_process().daemon is True, "child must be daemonic"

    cap = _CaptureHandler()
    logger = logging.getLogger(_PARALLEL_LOGGER)
    logger.addHandler(cap)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        results = zpar.pmap(_square, tasks, workers=4)
        q.put({"status": "ok", "results": results, "log": list(cap.messages)})
    except BaseException as exc:  # noqa: BLE001 - surface ANY child crash
        q.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "log": list(cap.messages),
            }
        )
    finally:
        logger.removeHandler(cap)
        logger.setLevel(old_level)


def test_pmap_daemonic_parent_uses_threads_and_matches_serial():
    """Production-context regression guard.

    Run ``pmap`` inside a DAEMONIC child process.  Pre-fix this crashed with
    ``AssertionError: daemonic processes are not allowed to have children``
    (the R12 process-pool path); post-fix it must (i) complete, (ii) return the
    same results as the serial loop, and (iii) select the thread pool.
    """
    tasks = [1, 2, 3, 4, 5, 6]
    serial = [_square(t) for t in tasks]

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_pmap_child_entry, args=(q, tasks), daemon=True)
    p.start()
    p.join(timeout=60)

    assert p.exitcode == 0, f"daemonic child crashed with exit code {p.exitcode}"
    assert not q.empty(), "daemonic child produced no result"
    payload = q.get()

    assert payload["status"] == "ok", payload.get("error")
    assert payload["results"] == serial
    # It must NOT have silently gone serial: the thread pool was actually used.
    assert any("thread pool" in m for m in payload["log"]), payload["log"]


def test_pmap_failsafe_fallback_on_pool_failure(monkeypatch, caplog):
    """A parallel-path failure must degrade to serial with a WARN, never crash."""
    import concurrent.futures

    class _BoomExecutor:
        def __init__(self, *a, **k):
            raise OSError("simulated spawn failure")

    # Force the non-daemonic (process) path, then make that executor blow up.
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _BoomExecutor)

    tasks = [2, 3, 4, 5]
    with caplog.at_level(logging.WARNING, logger=_PARALLEL_LOGGER):
        results = zpar.pmap(_square, tasks, workers=4)

    assert results == [_square(t) for t in tasks]
    assert any(
        "parallel map unavailable" in r.message and "falling back to serial" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Thread-path bit-equality: the real gauge, in a daemonic child (ungated)
# ---------------------------------------------------------------------------

_DITHER = {}


def _decode(f):
    """Module-level decode for the synthetic dithered corpus."""
    return _DITHER[f.frame_id.logical_path]


def _make_tan_wcs(h, w, ra_deg, dec_deg, scale_deg=0.001):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale_deg
    wcs.array_shape = (h, w)
    return wcs


def _build_corpus(n=4):
    """Small deterministic dithered TAN corpus (partial overlap)."""
    h = w = 40
    scale = 0.001
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 12.0**2))
    descs = []
    _DITHER.clear()
    for i in range(n):
        ra = 10.0 + i * 0.004
        wcs = _make_tan_wcs(h, w, ra, 30.0, scale)
        data = (base + rng.normal(0.0, 0.5, (h, w))).astype(np.float32)
        hwc = np.stack([data, data, data], axis=-1)
        fid = f"d{i}.fits"
        _DITHER[fid] = hwc
        descs.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid),
                source_path=f"/x/{fid}",
                shape_hw=(h, w),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return descs


def _gauge_child_entry(q, descs, canvas, config):
    """Runs the real gauge in a daemonic child (thread path) and reports results."""
    assert multiprocessing.current_process().daemon is True, "child must be daemonic"

    cap = _CaptureHandler()
    logger = logging.getLogger(_PARALLEL_LOGGER)
    logger.addHandler(cap)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        new, ids = zphot.compute_global_gauge(
            descs, canvas, _decode, config, None, workers=4
        )
        q.put(
            {
                "status": "ok",
                "reference_index": int(new.reference_index),
                "coefficients": np.asarray(new.coefficients),
                "norm_active": np.asarray(new.norm_active),
                "weights": np.asarray(new.weights),
                "weight_active": np.asarray(new.weight_active),
                "exclusions": [
                    (e.index, e.stage, e.reason, e.detail) for e in new.exclusions
                ],
                "ids": tuple(ids),
                "log": list(cap.messages),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        q.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "log": list(cap.messages),
            }
        )
    finally:
        logger.removeHandler(cap)
        logger.setLevel(old_level)


def test_gauge_thread_path_bit_equal_serial():
    """The real gauge, computed under the THREAD path (daemonic child), is
    bit-identical to the serial build (same reference, flags, coefficients,
    weights, exclusions, frame order)."""
    descs = _build_corpus()
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    ref, ref_ids = zphot.compute_global_gauge(
        descs, canvas, _decode, config, None, workers=1
    )

    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_gauge_child_entry, args=(q, descs, canvas, config), daemon=True
    )
    p.start()
    p.join(timeout=120)

    assert p.exitcode == 0, f"daemonic child crashed with exit code {p.exitcode}"
    assert not q.empty(), "daemonic child produced no result"
    payload = q.get()
    assert payload["status"] == "ok", payload.get("error")

    assert int(ref.reference_index) == payload["reference_index"]
    assert tuple(ref_ids) == payload["ids"]
    np.testing.assert_array_equal(ref.norm_active, payload["norm_active"])
    np.testing.assert_array_equal(ref.weight_active, payload["weight_active"])
    # Bit-equality: coefficients/weights are deterministic pure functions of the
    # task, so thread vs serial are EXACTLY equal (assert_array_equal treats NaN
    # as equal).
    np.testing.assert_array_equal(ref.coefficients, payload["coefficients"])
    np.testing.assert_array_equal(ref.weights, payload["weights"])
    assert tuple((e.index, e.stage, e.reason, e.detail) for e in ref.exclusions) == \
        tuple(payload["exclusions"])
    # Thread path was actually exercised (not a silent serial fallback).
    assert any("thread pool" in m for m in payload["log"]), payload["log"]
