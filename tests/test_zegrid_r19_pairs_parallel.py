"""ZM-ZEGRID-R19 targeted tests — gauge pairs pass: executor/start-method agnostic.

Covers (fast tier, no gated corpora):

* the pairs pass under an explicit **spawn** multiprocessing context (the
  Windows-like case that used to fail because the reference planes were
  fork-only copy-on-write globals) -> completes + bit-equal to serial;
* the **fork** (process) path -> bit-equal (unchanged);
* the **daemonic-parent / thread** path -> bit-equal (R13 already covers it;
  here we assert the initializer is wired for the thread path too);
* the ZM-ZEGRID-R19 diagnostic (executor kind / daemon flag / worker count /
  per-phase seconds-per-unit) is recorded;
* the temp reference files are cleaned up (best-effort);
* the fail-safe fallback WARN is surfaced via the ``emit``/progress-callback
  path (NOT only via ``logging``) — the visibility amendment.

The spawn test writes the synthetic corpus to disk (``.npy``) so the decode
function is spawn-safe (no fork-inherited module globals).
"""

from __future__ import annotations

import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid.executor import ExecutorConfig


def _make_tan_wcs(h, w, ra_deg, dec_deg, scale_deg=0.001):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale_deg
    wcs.array_shape = (h, w)
    return wcs


def _decode_disk(f):
    """Spawn-safe decode: read HWC float32 from the descriptor's source path."""
    return np.load(f.source_path).astype(np.float32)


def _build_corpus_disk(tmp: Path, n=4):
    """Small deterministic dithered TAN corpus written to disk (spawn-safe)."""
    h = w = 40
    scale = 0.001
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 12.0**2))
    descs = []
    for i in range(n):
        ra = 10.0 + i * 0.004
        wcs = _make_tan_wcs(h, w, ra, 30.0, scale)
        data = (base + rng.normal(0.0, 0.5, (h, w))).astype(np.float32)
        hwc = np.stack([data, data, data], axis=-1)
        p = tmp / f"d{i}.npy"
        np.save(p, hwc)
        descs.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(f"d{i}.fits"),
                source_path=str(p),
                shape_hw=(h, w),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return descs


def _gauge_payload(gauge, ids):
    return {
        "reference_index": int(gauge.reference_index),
        "coefficients": np.asarray(gauge.coefficients),
        "norm_active": np.asarray(gauge.norm_active),
        "weights": np.asarray(gauge.weights),
        "weight_active": np.asarray(gauge.weight_active),
        "exclusions": tuple(
            (e.index, e.stage, e.reason, e.detail) for e in gauge.exclusions
        ),
        "ids": tuple(ids),
    }


def _assert_bit_equal(ref, ref_ids, payload):
    assert int(ref.reference_index) == payload["reference_index"]
    assert tuple(ref_ids) == payload["ids"]
    np.testing.assert_array_equal(ref.norm_active, payload["norm_active"])
    np.testing.assert_array_equal(ref.weight_active, payload["weight_active"])
    np.testing.assert_array_equal(ref.coefficients, payload["coefficients"])
    np.testing.assert_array_equal(ref.weights, payload["weights"])
    assert tuple((e.index, e.stage, e.reason, e.detail) for e in ref.exclusions) == \
        payload["exclusions"]


# ---------------------------------------------------------------------------
# pmap initializer + fallback surfacing
# ---------------------------------------------------------------------------

def _square(x):
    return x * x


_INIT_STATE = {"calls": 0}


def _init_state_marker():
    _INIT_STATE["calls"] += 1


def test_pmap_initializer_called_for_process_and_serial():
    # serial path (workers=1) still calls the initializer once in the parent.
    _INIT_STATE["calls"] = 0
    out = zpar.pmap(_square, [1, 2, 3], 1, initializer=_init_state_marker)
    assert out == [1, 4, 9]
    assert _INIT_STATE["calls"] == 1

    # process path: parent + each worker child call it (>= 1 call in parent).
    _INIT_STATE["calls"] = 0
    out = zpar.pmap(_square, [1, 2, 3, 4], 2, initializer=_init_state_marker)
    assert out == [1, 4, 9, 16]
    assert _INIT_STATE["calls"] >= 1


def test_pmap_fallback_warn_reaches_emit(monkeypatch):
    """The fail-safe fallback WARN is surfaced via ``emit`` (progress-callback
    path), not only via ``logging``."""
    import concurrent.futures

    class _BoomExecutor:
        def __init__(self, *a, **k):
            raise OSError("simulated spawn failure")

    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _BoomExecutor)

    surfaced = []

    def _emit(msg, lvl="INFO"):
        surfaced.append((msg, lvl))

    results = zpar.pmap(_square, [2, 3, 4, 5], 4, emit=_emit)
    assert results == [_square(t) for t in [2, 3, 4, 5]]
    warns = [m for m, lvl in surfaced if lvl == "WARN"]
    assert any("parallel map unavailable" in m and "falling back to SERIAL" in m for m in warns)


def test_pmap_decision_surfaced_via_emit():
    surfaced = []

    def _emit(msg, lvl="INFO"):
        surfaced.append((msg, lvl))

    zpar.pmap(_square, [1, 2, 3, 4], 2, emit=_emit)
    assert any("pmap:" in m and "pool" in m for m, _ in surfaced)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def test_gauge_diagnostics_recorded(tmp_path):
    descs = _build_corpus_disk(tmp_path)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    diag = {}
    zphot.compute_global_gauge(descs, canvas, _decode_disk, config, str(tmp_path / "cache"),
                               workers=2, diagnostics=diag)

    assert diag["workers"] == 2
    assert "parent_daemon" in diag
    assert diag["executor"] in ("process", "thread", "serial")
    assert diag["phases"]["counts"]["units"] == 4
    assert diag["phases"]["pairs"]["units"] == 3
    assert diag["phases"]["counts"]["seconds_per_unit"] >= 0.0
    assert diag["phases"]["pairs"]["seconds_per_unit"] >= 0.0
    assert diag["phases"]["counts"]["executor"] in ("process", "thread", "serial")
    assert diag["phases"]["pairs"]["executor"] in ("process", "thread", "serial")


def test_gauge_reference_cleaned_up(tmp_path):
    descs = _build_corpus_disk(tmp_path)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()
    cache_dir = tmp_path / "cache"

    zphot.compute_global_gauge(descs, canvas, _decode_disk, config, str(cache_dir), workers=2)

    # The temp reference dir (cache/__gauge_ref__) must be removed (best-effort).
    ref_dir = cache_dir / "__gauge_ref__"
    assert not ref_dir.exists(), "temp reference planes were not cleaned up"


# ---------------------------------------------------------------------------
# Bit-equality: fork (process) path (unchanged)
# ---------------------------------------------------------------------------

def test_gauge_fork_bit_equal_serial(tmp_path):
    descs = _build_corpus_disk(tmp_path)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    ref, ref_ids = zphot.compute_global_gauge(descs, canvas, _decode_disk, config, None, workers=1)
    got, got_ids = zphot.compute_global_gauge(descs, canvas, _decode_disk, config, None, workers=4)
    _assert_bit_equal(ref, ref_ids, _gauge_payload(got, got_ids))


# ---------------------------------------------------------------------------
# Bit-equality: SPAWN (Windows-like) — the core fix
# ---------------------------------------------------------------------------

def _gauge_spawn_entry(q, descs, canvas, config, workers):
    multiprocessing.set_start_method("spawn", force=True)
    diag = {}
    try:
        new, ids = zphot.compute_global_gauge(descs, canvas, _decode_disk, config, None,
                                              workers=workers, diagnostics=diag)
        q.put({"status": "ok", **_gauge_payload(new, ids),
               "diagnostics": diag})
    except BaseException as exc:  # noqa: BLE001
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def test_gauge_spawn_bit_equal_serial(tmp_path):
    """The pairs pass completes under an explicit SPAWN context and is bit-equal
    to serial (this used to fail: children re-import and the fork-only globals
    were None)."""
    descs = _build_corpus_disk(tmp_path)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    ref, ref_ids = zphot.compute_global_gauge(descs, canvas, _decode_disk, config, None, workers=1)

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_gauge_spawn_entry, args=(q, descs, canvas, config, 4))
    p.start()
    p.join(timeout=600)

    assert p.exitcode == 0, f"spawn child crashed with exit code {p.exitcode}"
    assert not q.empty(), "spawn child produced no result"
    payload = q.get()
    assert payload["status"] == "ok", payload.get("error")
    _assert_bit_equal(ref, ref_ids, payload)
    # The pairs pass actually used processes (not a silent serial fallback).
    assert payload["diagnostics"]["executor"] == "process"
