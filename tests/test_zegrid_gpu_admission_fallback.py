"""ZM-ZEGRID-GPU-ADMISSION-FALLBACK — AUTO dense->bounded on a GPU host refusal.

The stored defect: a dense PLAN is admissible (the 3x3 rescue fits B_plan) while
the ACTIVE GPU admission is refused for host RAM, because the dense per-tile
host bound prices N RESIDENT frames (~14.03 GiB for 341 frames) and the margin
pushes it past B_plan. Before this fix the run died with ``_R55GpuRefused`` and
never tried the bounded profile (2.15 GiB on the SAME demand).

These tests drive the REAL product entry (``run_zegrid_mode``), the REAL initial
bundle planner (``_plan_bundle_auto``) and the REAL bounded GPU decision/admission
(``_r55_active_gpu_decide`` / ``_r55_active_gpu_admit``). Only the DENSE GPU
refusal is injected (deterministically, through the product decision seam), and
the run is stopped at the NEXT phase boundary (the global gauge) so no pixel is
stacked and no CUDA device is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import gpu_readiness as zgr  # noqa: E402
from zemosaic.core.zegrid import resource_envelope as renv  # noqa: E402
from zemosaic.core.zegrid import storage_profile as zsp  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30
_REAL_FLOORS = zal.ScientificFloors
RAW = 4 * GIB
VRAM_FREE = 2 * GIB
VRAM_TOTAL = 4 * GIB


class _StopAfterAdmission(RuntimeError):
    """Deliberate boundary stop: the admission witness is complete."""


def _tiny_floors(**_kw):
    return _REAL_FLOORS(
        min_patch_area_px=1024, min_contributors=1, max_halo_overhead=1e9
    )


def _write_corpus(tmp_path, *, size=256, frames=4):
    """Deterministic 2x2-dithered TAN corpus (no real data, no pixels stacked)."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    h = w = int(size)
    scale = 0.001
    rng = np.random.default_rng(20261025)
    inp = tmp_path / "in"
    inp.mkdir()
    names = []
    for i in range(int(frames)):
        col, row = i % 2, i // 2
        ra = 10.0 + col * (w / 2) * scale
        dec = 30.0 - row * (h / 2) * scale
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        data = (
            100.0
            + 20.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 24.0 ** 2))
            + 3.0 * i
            + rng.normal(0.0, 0.5, (h, w))
        ).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = 10.0
        name = f"frame_{i:04d}.fits"
        hdu.writeto(inp / name, overwrite=True)
        names.append(name)
    (inp / "stack_plan.csv").write_text(
        "file_path,exposure\n" + "".join(f"{n},10.0\n" for n in names),
        encoding="utf-8",
    )
    return inp, tmp_path / "out"


def _readiness():
    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="fake-rtx3070",
        vram_total_bytes=VRAM_TOTAL,
        vram_free_bytes=VRAM_FREE,
    )


def _fake_gpu(monkeypatch):
    probe = {
        "available": True,
        "device": "fake-rtx3070",
        "cupy_version": "14.1.1",
        "vram_total_bytes": VRAM_TOTAL,
        "vram_free_bytes": VRAM_FREE,
        "reason": None,
    }
    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", lambda: dict(probe))
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe", lambda **_kw: _readiness()
    )
    monkeypatch.setattr(
        zz.zgr, "read_vram_free_bytes", lambda device_index=0: int(VRAM_FREE)
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    return probe


@pytest.fixture()
def gpu_seams(monkeypatch):
    """Product GPU seams + a hard stop at the post-admission phase boundary."""
    _fake_gpu(monkeypatch)
    state = {"gauge_calls": 0}

    def _stop_gauge(*_a, **_kw):
        state["gauge_calls"] += 1
        raise _StopAfterAdmission("witness stopped after the ACTIVE GPU admission")

    monkeypatch.setattr(zz.zphot, "compute_global_gauge", _stop_gauge)
    return state


def _drive_decide(monkeypatch, *, refuse_dense=True, reason=None):
    """Wrap the REAL GPU decision: DENSE refuses, BOUNDED delegates for real."""
    real = zz._r55_active_gpu_decide
    calls = []

    def _decide(**kw):
        bundle = kw["bundle"]
        calls.append(
            {
                "profile": zz._STORAGE_EFFECTIVE,
                "bundle_profile": bundle.storage_profile,
                "layout": tuple(int(v) for v in bundle.envelope.layout),
                "cells": len(bundle.cell_ctxs),
                "tasks": len(bundle.cell_tasks),
            }
        )
        if reason is not None:
            raise zz._R55GpuRefused(reason, "injected closed refusal")
        if refuse_dense and zz._STORAGE_EFFECTIVE == zsp.STORAGE_PROFILE_DENSE:
            raise zz._R55GpuRefused(
                renv.REASON_REFUSE_HOST_RAM,
                "dense GPU host bound 14.03 GiB/tile x margin > B_plan "
                "for the resident frame set",
            )
        return real(**kw)

    monkeypatch.setattr(zz, "_r55_active_gpu_decide", _decide)
    return calls


def _drive_admit(monkeypatch):
    real = zz._r55_active_gpu_admit
    admitted = []

    def _admit(*, initial_bundle, decision, **kw):
        result = real(initial_bundle=initial_bundle, decision=decision, **kw)
        admitted.append(
            {
                "bundle": initial_bundle,
                "decision": decision,
                "result": result,
            }
        )
        return result

    monkeypatch.setattr(zz, "_r55_active_gpu_admit", _admit)
    return admitted


def _run(inp, out, *, storage=None, mode="active"):
    zz.run_zegrid_mode(
        str(inp), str(out),
        zconfig=SimpleNamespace(zegrid_scheduler_mode=mode),
        zegrid_halo_px=8,
        use_gpu=True,
        storage_profile=storage,
    )


def _storage():
    return zz._storage_policy_snapshot()


# ---------------------------------------------------------------------------
# Positive witness — dense plan fits, dense GPU refuses, bounded GPU admits.
# ---------------------------------------------------------------------------

def test_auto_gpu_host_refusal_retries_bounded_once_and_admits(
    tmp_path, gpu_seams, monkeypatch,
):
    inp, out = _write_corpus(tmp_path)
    calls = _drive_decide(monkeypatch)
    admitted = _drive_admit(monkeypatch)

    with pytest.raises(_StopAfterAdmission):
        _run(inp, out)

    snap = _storage()
    assert snap["requested"] == "auto"
    assert snap["effective"] == "bounded"
    assert snap["model"] == zsp.STORAGE_MODEL_BOUNDED
    assert snap["fallback_count"] == 1
    assert snap["dense_refusal"]["phase"] == "plan_gpu_admission"
    assert snap["dense_refusal"]["reason_code"] == renv.REASON_REFUSE_HOST_RAM
    assert len(snap["transitions"]) == 1
    assert snap["transitions"][0] == {
        "from": "dense",
        "to": "bounded",
        "phase": "plan_gpu_admission",
        "reason_code": renv.REASON_REFUSE_HOST_RAM,
        "detail": snap["dense_refusal"]["detail"],
    }
    # EXACTLY one dense decision and ONE bounded retry: no second profile switch.
    assert [c["profile"] for c in calls] == ["dense", "bounded"]
    assert calls[0]["bundle_profile"] == "dense"
    assert calls[1]["bundle_profile"] == "bounded"
    # The rebuilt (bounded) initial bundle is the one actually decided/admitted.
    assert calls[1]["layout"] == tuple(
        int(v) for v in admitted[0]["bundle"].envelope.layout
    )
    assert calls[1]["cells"] == len(admitted[0]["bundle"].cell_ctxs)
    assert calls[1]["tasks"] == len(admitted[0]["bundle"].cell_tasks)
    # The bounded GPU plan was REALLY admitted (real decide + real admit).
    assert len(admitted) == 1
    bundle = admitted[0]["bundle"]
    decision = admitted[0]["decision"]
    (_final, _candidate, mapping, _adm, _trail) = admitted[0]["result"]
    assert bundle.storage_profile == "bounded"
    assert bundle.gpu_used is True
    assert mapping["backend"] == "gpu"
    assert mapping["gpu_streaming_tile"] in decision.gpu_tile_ladder
    assert mapping["units_in_flight"] == 1
    assert int(_final.envelope.aggregate_bytes) <= int(
        _final.envelope.budget_bytes
    )
    assert bundle.layout == _final.layout
    assert len(bundle.cell_tasks) == int(bundle.layout["nx"]) * int(
        bundle.layout["ny"]
    )
    # The rebuilt bundle owns EVERY derived consumer (no stale dense artifacts).
    assert bundle.footprints is not None
    assert bundle.cell_ctxs is not None
    assert all(t[0].cell_id for t in bundle.cell_tasks)
    # Exactly ONE fallback: nothing re-activated it later.
    assert zz._STORAGE_FALLBACK_COUNT == 1
    assert len(_storage()["transitions"]) == 1


def test_auto_bounded_gpu_admits_single_owner_inside_the_frozen_ladder(
    tmp_path, gpu_seams, monkeypatch,
):
    """The defect's user-visible contract: the bounded retry runs GPU, not CPU.

    The recorded Caldwell witness reports ``prefetch_units=0``/tile128 because
    the EXACT bounded bound leaves no room for prefetch on that 341-frame
    corpus. This tiny synthetic corpus has ample host headroom, so the bounded
    decision legitimately keeps a positive bounded prefetch; the invariant that
    matters here is that the decision is the REAL bounded one, single-owner,
    inside the frozen declared ladder and inside its own envelope.
    """
    inp, out = _write_corpus(tmp_path)
    _drive_decide(monkeypatch)
    admitted = _drive_admit(monkeypatch)

    with pytest.raises(_StopAfterAdmission):
        _run(inp, out)

    mapping = admitted[0]["result"][2]
    decision = admitted[0]["decision"]
    assert mapping["units_in_flight"] == 1
    assert mapping["single_owner"] is True
    prefetch = int(mapping["prefetch_units"])
    assert 0 <= prefetch <= int(zz._R55E2_PREFETCH_LADDER[-1])
    assert int(decision.qualification["queue_max"]) == prefetch + 1
    assert int(mapping["gpu_streaming_tile"]) >= 1
    assert int(mapping["gpu_streaming_tile"]) in tuple(
        int(t) for t in decision.gpu_tile_ladder
    )
    assert all(
        int(peak) > 0 for _t, peak in decision.gpu_host_bound_by_tile
    )


# ---------------------------------------------------------------------------
# Negative witnesses — the bounded retry must NOT arm outside its contract.
# ---------------------------------------------------------------------------

def test_forced_dense_gpu_host_refusal_stays_terminal(tmp_path, gpu_seams, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    calls = _drive_decide(monkeypatch)
    admitted = _drive_admit(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        _run(inp, out, storage="dense")

    assert "R55-E2 active GPU refused before the gauge/stack" in str(excinfo.value)
    assert not getattr(excinfo.value, "fallback_exhausted", False)
    assert [c["profile"] for c in calls] == ["dense"]
    assert admitted == []
    snap = _storage()
    assert snap["requested"] == "dense"
    assert snap["effective"] == "dense"
    assert snap["fallback_count"] == 0
    assert snap["transitions"] == []
    assert gpu_seams["gauge_calls"] == 0


@pytest.mark.parametrize(
    "reason",
    [
        renv.REASON_REFUSE_VRAM,
        zgr.REASON_GPU_OBSERVATION_FAILED,
        renv.REASON_REFUSE_DISK,
        zgr.REASON_GPU_PARITY_NOT_QUALIFIED,
    ],
)
def test_non_host_ram_gpu_refusal_never_switches_profile(
    tmp_path, gpu_seams, monkeypatch, reason,
):
    inp, out = _write_corpus(tmp_path)
    calls = _drive_decide(monkeypatch, reason=reason)
    admitted = _drive_admit(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        _run(inp, out)

    assert reason in str(excinfo.value)
    assert [c["profile"] for c in calls] == ["dense"]
    assert admitted == []
    snap = _storage()
    assert snap["effective"] == "dense"
    assert snap["fallback_count"] == 0
    assert snap["transitions"] == []
    assert gpu_seams["gauge_calls"] == 0


@pytest.mark.parametrize("mode", ["legacy", "observe"])
def test_legacy_and_observe_never_arm_the_gpu_fallback(
    tmp_path, gpu_seams, monkeypatch, mode,
):
    """No ACTIVE GPU decision is taken, so no profile switch can be armed."""
    inp, out = _write_corpus(tmp_path)
    calls = _drive_decide(monkeypatch)

    class _HugeModel:
        def predict_bound_bytes(self, n, area):
            return 16 * GIB

        def predict_peak_bytes(self, n, area):
            return 16 * GIB

    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            1 * MIB if storage_profile == "bounded" else 8192 * MIB
        ),
    )

    with pytest.raises((zal.LayoutInfeasible, zz.zpar.ModeInfeasible)):
        _run(inp, out, mode=mode)

    assert calls == []
    snap = _storage()
    assert snap["effective"] == "dense"
    assert snap["fallback_count"] == 0
    assert snap["transitions"] == []
    assert gpu_seams["gauge_calls"] == 0


def test_double_refusal_is_terminal_after_exactly_one_bounded_attempt(
    tmp_path, gpu_seams, monkeypatch,
):
    """Bounded is ALSO incapable: one retry, both causes, fallback_exhausted."""
    inp, out = _write_corpus(tmp_path)
    calls = _drive_decide(monkeypatch, reason=renv.REASON_REFUSE_HOST_RAM)
    admitted = _drive_admit(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        _run(inp, out)

    err = excinfo.value
    assert getattr(err, "fallback_exhausted", False) is True
    text = str(err)
    assert "BOTH storage profiles" in text
    assert "R55_REFUSE_HOST_RAM" in text
    assert "dense:" in text and "bounded:" in text
    assert "no second retry" in text
    refusals = err.storage_refusals
    assert refusals["dense"]["reason_code"] == renv.REASON_REFUSE_HOST_RAM
    assert refusals["bounded"]["reason_code"] == renv.REASON_REFUSE_HOST_RAM
    # EXACTLY two decisions (dense + ONE bounded) and no admission attempt.
    assert [c["profile"] for c in calls] == ["dense", "bounded"]
    assert admitted == []
    snap = _storage()
    assert snap["effective"] == "bounded"
    assert snap["fallback_count"] == 1
    assert len(snap["transitions"]) == 1
    assert snap["transitions"][0]["phase"] == "plan_gpu_admission"
    assert gpu_seams["gauge_calls"] == 0
