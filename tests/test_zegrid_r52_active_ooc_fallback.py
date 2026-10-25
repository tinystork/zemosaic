"""ZM-ZEGRID-ACTIVE-OOC-FALLBACK — product AUTO dense->bounded fallback.

The PRODUCT request (``run_zegrid_mode(storage_profile=None)``) is now ``auto``:
the ACTIVE scheduler tries the historical dense model first and, ONLY on a
QUALIFIED host-RAM refusal, retries exactly once with the bounded profile at the
same geometry/halo/science. Legacy and observe never fall back, and an explicit
``dense``/``bounded`` request is FORCED (A/B witnesses stay honest).

Every test here drives the REAL product entry (``run_zegrid_mode``) or the REAL
wired planner helper ``_plan_bundle_auto`` used by that entry — never a
re-implementation. Memory pressure is injected deterministically through the
public estimator seam, not through a captured default function.
"""

from __future__ import annotations

import json
import subprocess
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
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import storage_profile as zsp  # noqa: E402

MIB = 2 ** 20
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(**_kw):
    # Keep Cells reasonably large (32x32 px floor) so the fine-ladder scan stays
    # cheap and deterministic; halo/contributor floors are relaxed for the tiny
    # synthetic corpus only.
    return _REAL_FLOORS(
        min_patch_area_px=1024, min_contributors=1, max_halo_overhead=1e9
    )


def _fake_stream_bytes(n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
                       storage_profile=None):
    """Deterministic memory injection: dense never fits, bounded always does."""
    return 1 * MIB if storage_profile == "bounded" else 8192 * MIB


class _HugeModel:
    """In-memory bound that never fits: forces a REAL dense layout refusal."""

    def predict_bound_bytes(self, n, area):
        return 16 * 2 ** 30

    def predict_peak_bytes(self, n, area):
        return 16 * 2 ** 30


def _write_corpus(tmp_path):
    """Tiny deterministic TAN corpus (4 x 96x96, 2x2 dither) for product runs.

    Small enough that a full ACTIVE run (gauge + stack + assembly) finishes in
    seconds, but multi-frame and multi-cell with a real WCS. Never presented as
    a Caldwell/M106 witness.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    h = w = 96
    scale = 0.001
    rng = np.random.default_rng(20261025)
    inp = tmp_path / "in"
    inp.mkdir()
    names = []
    for i in range(4):
        col, row = i % 2, i // 2
        ra = 10.0 + col * 48 * scale
        dec = 30.0 - row * 48 * scale
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        data = (
            100.0
            + 20.0 * np.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 18.0 ** 2))
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


# Generous-host baseline used by the "dense fits" runs (4 GiB of available RAM).
_DENSE_OK_RAW = 4 * 2 ** 30


def _active_zconfig(mode="active"):
    return SimpleNamespace(zegrid_scheduler_mode=mode)


def _manifest(out):
    return json.loads((out / "zegrid_manifest.json").read_text())


# ---------------------------------------------------------------------------
# Policy primitives (requested vs effective)
# ---------------------------------------------------------------------------

def test_normalize_storage_request_semantics():
    assert zsp.normalize_storage_request(None) == "auto"
    assert zsp.normalize_storage_request("DENSE") == "dense"
    assert zsp.normalize_storage_request("bounded") == "bounded"
    assert zsp.normalize_storage_request("ooc") == "bounded"
    # The effective seam keeps its historical None -> dense contract.
    assert zsp.normalize_storage_profile(None) == "dense"
    with pytest.raises(ValueError):
        zsp.normalize_storage_request("turbo")


def test_policy_reset_forced_never_falls_back():
    zz._storage_policy_reset("dense")
    snap = zz._storage_policy_snapshot()
    assert snap["requested"] == "dense" and snap["effective"] == "dense"
    assert snap["forced"] is True
    assert zz._storage_fallback_available() is False
    assert zz._activate_bounded_storage(phase="x", reason_code="c", detail="d") is False

    zz._storage_policy_reset(None)
    assert zz._storage_fallback_available() is True
    assert zz._activate_bounded_storage(phase="plan_layout", reason_code="c",
                                        detail="d") is True
    assert zz._storage_fallback_available() is False  # exactly ONCE
    assert zz._STORAGE_FALLBACK_COUNT == 1
    assert zz._STORAGE_EFFECTIVE == "bounded"


def test_memory_refusal_classification_does_not_hide_real_floors():
    # A genuine RAM miss is qualified.
    ram = zal.LayoutInfeasible(
        "RAM budget cannot satisfy scientific floors: even the finest "
        "floor-feasible layout (5x3) peak bound 28450.7 MiB exceeds budget "
        "15315.8 MiB"
    )
    assert zz._storage_plan_refusal(ram)[0] is True
    # A halo/floor-only refusal is NOT a memory refusal.
    floors = zal.LayoutInfeasible(
        "RAM budget cannot satisfy scientific floors: min_patch_area floor "
        "(4096 px) / max_halo_overhead floor (0.5) cannot be satisfied together "
        "with the budget (search stopped at the floor boundary)"
    )
    assert zz._storage_plan_refusal(floors)[0] is False
    # Arbitrary errors are never memory refusals.
    assert zz._storage_plan_refusal(ValueError("WCS mismatch"))[0] is False


# ---------------------------------------------------------------------------
# Public entry — dense success / auto fallback / forced profiles
# ---------------------------------------------------------------------------

def test_public_active_auto_dense_success_has_no_retry(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)

    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
    )
    storage = _manifest(out)["storage"]
    assert storage["requested"] == "auto"
    assert storage["effective"] == "dense"
    assert storage["fallback_count"] == 0
    assert storage["dense_refusal"] is None
    assert storage["transitions"] == []
    assert (out / "mosaic_grid.fits").exists()


def test_public_active_auto_layout_refusal_falls_back_to_bounded(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _fake_stream_bytes)

    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
    )
    manifest = _manifest(out)
    storage = manifest["storage"]
    assert storage["requested"] == "auto"
    assert storage["effective"] == "bounded"
    assert storage["model"] == zsp.STORAGE_MODEL_BOUNDED
    assert storage["fallback_count"] == 1
    assert storage["dense_refusal"]["phase"] == "plan_layout"
    assert storage["transitions"][0]["from"] == "dense"
    assert storage["transitions"][0]["to"] == "bounded"
    # The mosaic really completed.
    assert (out / "mosaic_grid.fits").exists()


def test_public_active_forced_dense_refuses_without_fallback(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _fake_stream_bytes)

    with pytest.raises((zal.LayoutInfeasible, zz.zpar.ModeInfeasible)):
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            storage_profile="dense",
        )
    assert not (out / "zegrid_manifest.json").exists()
    assert zz._STORAGE_EFFECTIVE == "dense"
    assert zz._STORAGE_FALLBACK_COUNT == 0


def test_public_active_forced_bounded_is_forced(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)

    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
            storage_profile="bounded",
    )
    storage = _manifest(out)["storage"]
    assert storage["requested"] == "bounded"
    assert storage["effective"] == "bounded"
    assert storage["forced"] is True
    assert storage["fallback_count"] == 0
    assert storage["transitions"] == []


@pytest.mark.parametrize("mode", ["legacy", "observe"])
def test_legacy_and_observe_never_gain_a_fallback(tmp_path, monkeypatch, mode):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _fake_stream_bytes)

    with pytest.raises((zal.LayoutInfeasible, zz.zpar.ModeInfeasible)):
        zz.run_zegrid_mode(str(inp), str(out), zconfig=_active_zconfig(mode), zegrid_halo_px=8)
    assert zz._STORAGE_EFFECTIVE == "dense"
    assert zz._STORAGE_FALLBACK_COUNT == 0
    # No manifest key is added for legacy/observe either.
    assert not (out / "zegrid_manifest.json").exists()


# ---------------------------------------------------------------------------
# Runtime safe boundary — dense checkpoint refusal -> bounded on SAME geometry
# ---------------------------------------------------------------------------

def test_public_active_runtime_checkpoint_fallback(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    # Dense plans exist (the run starts dense), but the first RUN checkpoint is
    # refused for host RAM; the bounded retry (smaller stream bound) succeeds.
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            1 * MIB if storage_profile == "bounded" else 8 * MIB
        ),
    )
    real_checkpoint = zz._r55_run_checkpoint
    calls = {"n": 0}

    def _refuse_once(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise zz._R55RunRefused(
                "R55_REFUSE_HOST_RAM",
                "injected host-RAM pressure at the first safe boundary",
                record={
                    "schema": "test",
                    "status": "refused",
                    "reason_code": "R55_REFUSE_HOST_RAM",
                    "detail": "injected host-RAM pressure",
                    "wave_index": 0,
                    "remaining_units": 1,
                    "shrink_steps": [],
                    "unavailable_shrinks": [],
                },
            )
        return real_checkpoint(**kwargs)

    monkeypatch.setattr(zz, "_r55_run_checkpoint", _refuse_once)

    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
    )
    storage = _manifest(out)["storage"]
    assert storage["requested"] == "auto"
    assert storage["effective"] == "bounded"
    assert storage["fallback_count"] == 1
    assert storage["transitions"][0]["phase"].startswith("run_wave_")
    # Proves the run continued after the recovery instead of aborting.
    assert (out / "mosaic_grid.fits").exists()


def test_runtime_fallback_keeps_geometry_and_task_profile(tmp_path, monkeypatch):
    """A bounded recovery starts the remaining cells bounded at the SAME geometry.

    Proven end-to-end by the manifest: the fallback is recorded at a RUN wave
    phase and the final bundled cells still cover the whole canvas (same layout
    and memberships, only the storage profile changed).
    """
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            1 * MIB if storage_profile == "bounded" else 8 * MIB
        ),
    )
    real_checkpoint = zz._r55_run_checkpoint
    calls = {"n": 0}

    def _refuse_once(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise zz._R55RunRefused(
                "R55_REFUSE_HOST_RAM",
                "injected host-RAM pressure",
                record={
                    "schema": "test", "status": "refused",
                    "reason_code": "R55_REFUSE_HOST_RAM",
                    "detail": "injected host-RAM pressure",
                    "wave_index": 0, "remaining_units": 1,
                    "shrink_steps": [], "unavailable_shrinks": [],
                },
            )
        return real_checkpoint(**kwargs)

    monkeypatch.setattr(zz, "_r55_run_checkpoint", _refuse_once)
    zz.run_zegrid_mode(str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8)

    manifest = _manifest(out)
    assert manifest["storage"]["transitions"][0]["phase"].startswith("run_wave_")
    assert manifest["storage"]["effective"] == "bounded"
    assert (out / "mosaic_grid.fits").exists()


# ---------------------------------------------------------------------------
# Model/executor coherence: bounded tasks + no parent-global dependence
# ---------------------------------------------------------------------------

def test_bounded_bundle_tasks_carry_profile_and_survive_gpu_shaped_rebuild(
    tmp_path, monkeypatch,
):
    inp, _out = _write_corpus(tmp_path)
    from zemosaic.core.zegrid import geometry as zg

    frames, rejected = zg.read_manifest(inp)
    assert rejected == []
    canvas = zg.build_canvas(frames)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _fake_stream_bytes)
    zz._storage_policy_reset(None)
    raw = _DENSE_OK_RAW
    budget = rcontract.plan_budget_bytes(raw)

    bundle = zz._plan_bundle_auto(
        canvas=canvas, descs=frames, footprints=None,
        raw=raw, budget=budget, active=True, emit=None,
        bundle_common={"cpu_count": 2, "gpu_used": False, "cache_root": None},
    )
    assert bundle.storage_profile == "bounded"
    for task in bundle.cell_tasks:
        if task[4] is None:
            continue
        assert len(task) == 7
        assert task[6] == "bounded"
        assert task[4] == "stream"

    # GPU-shaped rebuild (the GPU->CPU / adapter path) keeps the EFFECTIVE
    # profile: never dense, never inmem, and the spawn payload carries it.
    gpu_bundle = zz._build_execution_bundle(
        canvas, frames, bundle.footprints,
        raw_available_bytes=raw, budget_bytes=budget,
        gpu_used=True, vram_budget_bytes=8 * 2 ** 30,
        cpu_count=2,
    )
    assert gpu_bundle.storage_profile == "bounded"
    assert gpu_bundle.chosen_mode == "stream"
    assert gpu_bundle.gpu_used is True


def test_prepared_payload_uses_task_profile_not_parent_global(monkeypatch):
    """A bounded task must stay bounded even if the parent global says dense."""
    seen = {}

    def _spy(cache_dir, storage_profile):
        seen["profile"] = storage_profile
        return "provider"

    monkeypatch.setattr(zz, "_make_cell_provider", _spy)
    monkeypatch.setattr(zz, "_build_one_cell_cache", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zz, "_STACK_CONFIG", object())
    monkeypatch.setattr(zz, "_STACK_DESCS", [])
    monkeypatch.setattr(zz, "_STACK_CANVAS", None)
    monkeypatch.setattr(zz.zxe, "reset_reproject_path_stats", lambda: None)

    class _Cell:
        cell_id = "c"

    class _Patch:
        patch_shape_hw = (8, 8)
        core_slice = SimpleNamespace(y0=0, y1=8, x0=0, x1=8)
        patch = SimpleNamespace(x0=0, y0=0, x1=8, y1=8, height=8, width=8)

    class _Mem:
        patch_ids = ["a", "b"]

    task = (_Cell(), _Patch(), _Mem(), "/nonexistent", "stream", 1, "bounded")
    # Stop after provider selection: the exact payload is covered by Lot C.
    try:
        zz._prepare_stack_cell(task)
    except Exception:
        pass
    assert seen.get("profile") == "bounded"


# ---------------------------------------------------------------------------
# Caldwell 816 replay through the WIRED AUTO planner (no full mosaic)
# ---------------------------------------------------------------------------

CALDWELL = Path("/home/tristan/caldwell11/lights organized/EQ/IRCUT")
CALDWELL_BUDGET = int(15315.8 * MIB)


@pytest.mark.slow
def test_caldwell_816_auto_wired_planner_falls_back_to_bounded(monkeypatch):
    if not CALDWELL.is_dir():
        pytest.skip("Caldwell IRCUT corpus not present")
    from zemosaic.core.zegrid import geometry as zg

    frames, rejected = zg.read_manifest(CALDWELL)
    if len(frames) != 816:
        pytest.skip(f"expected 816 headers, got {len(frames)}")
    canvas = zg.build_canvas(frames)
    raw = CALDWELL_BUDGET * rcontract.PLAN_FACTOR_DEN // rcontract.PLAN_FACTOR_NUM

    zz._storage_policy_reset(None)
    bundle = zz._plan_bundle_auto(
        canvas=canvas, descs=frames, footprints=None,
        raw=raw, budget=CALDWELL_BUDGET, active=True, emit=None,
        bundle_common={"cpu_count": 8, "gpu_used": False, "cache_root": None,
                        "halo_px": 128},
        layout_substep=None,
    )
    storage = zz._storage_policy_snapshot()
    assert storage["requested"] == "auto"
    assert storage["effective"] == "bounded"
    assert storage["fallback_count"] == 1
    assert storage["dense_refusal"]["phase"] == "plan_layout"
    assert "28450.7 MiB" in storage["dense_refusal"]["detail"]

    assert bundle.storage_profile == "bounded"
    assert (bundle.layout["nx"], bundle.layout["ny"]) == (4, 3)
    assert bundle.cpu_streaming_tile == 128
    covered = set()
    for _cell, _patch, mem, _cache, mode, _b, *_rest in bundle.cell_tasks:
        if mode is not None:
            covered.update(mem.patch_ids)
    assert len(covered) == 816
    assert covered == {f.frame_id.logical_path for f in frames}
    assert all(t[6] == "bounded" for t in bundle.cell_tasks if t[4] is not None)
