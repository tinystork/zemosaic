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


def test_memory_refusal_classification_prefers_structured_payload():
    """ZM-ZEGRID-PLAN-REFUSAL-CLASSIFICATION: facts decide, not prose."""
    facts = {
        "kind": "host_ram",
        "source": "coarser_rescue",
        "budget_bytes": int(12844.9 * MIB),
        "required_bytes": int(14444.7 * MIB),
        "candidate": {"nx": 1, "ny": 2},
    }
    # The REAL Caldwell-shaped message names no "exceeds budget" wording; the
    # owner's payload alone must qualify it.
    ram = zal.LayoutInfeasible(
        "requested halo 128px conflicts with the max_halo_overhead floor "
        "(0.5363 > 0.5) at the 4x3 factor-1 layout, and the nearest "
        "floor-feasible COARSER rescue 3x3 needs 14444.7 MiB > budget "
        "12844.9 MiB (in-memory worst 14444.7 MiB; best CPU streaming tile 128)",
        refusal=facts,
    )
    is_memory, code, detail = zz._storage_plan_refusal(ram)
    assert is_memory is True
    assert code == zz.renv.REASON_REFUSE_HOST_RAM
    assert "14444.7 MiB" in detail

    # A geometry payload is NOT eligible even when the message text WOULD match
    # the historical legacy regex.
    geo = zal.LayoutInfeasible(
        "even the finest floor-feasible layout is infeasible: 999.0 MiB "
        "exceeds budget 10.0 MiB (legacy-looking text)",
        refusal={"kind": "geometry", "source": "min_patch_area_floor"},
    )
    geo_is_memory, geo_code, _ = zz._storage_plan_refusal(geo)
    assert geo_is_memory is False
    assert geo_code is None


def test_memory_refusal_classification_is_text_invariant_with_payload():
    """Rewording a message must never change a payload-backed decision."""
    facts = {
        "kind": "host_ram",
        "source": "finest_candidate",
        "budget_bytes": 1 * MIB,
        "required_bytes": 2 * MIB,
        "candidate": {"nx": 4, "ny": 3},
    }
    verbose = zal.LayoutInfeasible(
        "RAM budget cannot satisfy scientific floors: ... exceeds budget ...",
        refusal=facts,
    )
    terse = zal.LayoutInfeasible(
        "plan refused",
        refusal=facts,
    )
    assert zz._storage_plan_refusal(verbose)[0] is True
    assert zz._storage_plan_refusal(verbose)[0] == zz._storage_plan_refusal(terse)[0]
    assert zz._storage_plan_refusal(terse)[1] == zz.renv.REASON_REFUSE_HOST_RAM

    geo_facts = {"kind": "geometry", "source": "halo_conflict_no_coarser_grid"}
    assert zz._storage_plan_refusal(
        zal.LayoutInfeasible("needs 99.0 MiB > budget 1.0 MiB", refusal=geo_facts)
    )[0] is zz._storage_plan_refusal(
        zal.LayoutInfeasible("plan refused", refusal=geo_facts)
    )[0] is False


def test_memory_refusal_classification_legacy_messages_unchanged():
    """No structured payload -> exact historical behaviour (regex + marker)."""
    # Legacy budget wording still qualifies.
    legacy_ram = zal.LayoutInfeasible(
        "RAM budget cannot satisfy scientific floors: even the finest "
        "floor-feasible layout (5x3) peak bound 28450.7 MiB exceeds budget "
        "15315.8 MiB"
    )
    assert legacy_ram.refusal == {}
    assert zz._storage_plan_refusal(legacy_ram)[0] is True
    # Legacy floor wording still does not.
    legacy_floors = zal.LayoutInfeasible(
        "min_patch_area floor (4096 px) cannot be satisfied together with the "
        "budget (search stopped at the floor boundary)"
    )
    assert zz._storage_plan_refusal(legacy_floors)[0] is False
    # The historical memory_exhausted marker still qualifies a rank-window miss.
    window = zal.LayoutInfeasible(
        "no layout candidate lies inside the requested refinement-rank window "
        "(min=12, max=None, cell-count rank)"
    )
    assert zz._storage_plan_refusal(window)[0] is False
    window.memory_exhausted = True
    assert zz._storage_plan_refusal(window)[0] is True


def test_layout_infeasible_refusal_default_is_backwards_compatible():
    exc = zal.LayoutInfeasible("plain reason")
    assert exc.refusal == {}
    assert exc.floors_report == {}
    report = {"x": 1}
    assert zal.LayoutInfeasible("plain", report).floors_report == report


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

def _one_task_waves(monkeypatch):
    monkeypatch.setattr(
        zz.zrt, "next_wave_bound",
        lambda offset, total, units: (offset, min(total, offset + 1)),
    )


def _inject_wave1_latched(monkeypatch, raw, *, at_call=2):
    """Inject one VALID governing latched fact at the wave-1 boundary."""
    real = zz.zrt.PressureLatch.consume_validated
    state = {"n": 0}

    def _consume(self):
        state["n"] += 1
        if state["n"] == at_call:
            return {
                "raw_available_bytes": int(raw),
                "budget_bytes": int(zz.zrt.run_budget_bytes(int(raw))),
            }
        return real(self)

    monkeypatch.setattr(zz.zrt.PressureLatch, "consume_validated", _consume)
    return state


def _pressure_env(tmp_path, monkeypatch, *, dense_mib, bounded_mib, raw):
    inp, out = _write_corpus(tmp_path)
    holder = {"raw": int(raw)}
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: holder["raw"])
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    # Force the DENSE plan to the streaming executor (a huge in-memory model), so
    # its per-cell bound really exceeds the governing wave budget; the bounded
    # profile is stream-only by contract and fits the same budget.
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            int(bounded_mib) * MIB if storage_profile == "bounded"
            else int(dense_mib) * MIB
        ),
    )
    _one_task_waves(monkeypatch)
    return inp, out, holder


def test_runtime_governing_budget_refuses_both_profiles(tmp_path, monkeypatch):
    """The governing (latched) budget is too low for dense AND bounded."""
    inp, out, _h = _pressure_env(
        tmp_path, monkeypatch, dense_mib=1024, bounded_mib=4,
        raw=_DENSE_OK_RAW,
    )
    _inject_wave1_latched(monkeypatch, 8 * MIB)
    with pytest.raises(zz._R55RunRefused) as ei:
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            zegrid_halo_px=8,
        )
    assert ei.value.reason_code == zz.renv.REASON_REFUSE_HOST_RAM
    assert ei.value.storage_fallback_eligible is True
    assert zz._STORAGE_EFFECTIVE == "bounded"
    assert zz._STORAGE_FALLBACK_COUNT == 1
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


def _eligible_refusal(raw, *, wave_index=1, remaining_units=3):
    return zz._R55RunRefused(
        zz.renv.REASON_REFUSE_HOST_RAM,
        "injected measured host-RAM shortage",
        record={
            "schema": zz.zrt.RUN_CONTROL_SCHEMA,
            "status": "refused",
            "stage": "run_checkpoint",
            "reason_code": zz.renv.REASON_REFUSE_HOST_RAM,
            "detail": "injected measured host-RAM shortage",
            "wave_index": int(wave_index),
            "remaining_units": int(remaining_units),
            "host": {
                "raw_available_bytes": int(raw),
                "budget_bytes": int(zz.zrt.run_budget_bytes(int(raw))),
                "factor_num": int(zz.rcontract.ADMIT_FACTOR_NUM),
                "factor_den": int(zz.rcontract.ADMIT_FACTOR_DEN),
            },
            "last_candidate": {"injected": True},
            "shrink_steps": [],
            "unavailable_shrinks": [],
        },
        storage_fallback_eligible=True,
    )


def test_runtime_retry_terminal_refusal_clears_eligibility(
    tmp_path, monkeypatch,
):
    """L-1: a bounded RETRY refusal is terminal (no second fallback)."""
    inp, out, _h = _pressure_env(
        tmp_path, monkeypatch, dense_mib=1024, bounded_mib=4,
        raw=_DENSE_OK_RAW,
    )
    governing = int(1.4 * 2 ** 30)
    _inject_wave1_latched(monkeypatch, governing)
    real_checkpoint = zz._r55_run_checkpoint
    calls = {"n": 0}

    def _checkpoint(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            # Dense wave-1 refusal (valid measured shortage) -> recovery runs.
            raise _eligible_refusal(governing, wave_index=1,
                                    remaining_units=kwargs["remaining_units"])
        if calls["n"] == 3:
            # The bounded RETRY of the SAME boundary refuses as well.
            raise _eligible_refusal(governing, wave_index=1,
                                    remaining_units=kwargs["remaining_units"])
        return real_checkpoint(**kwargs)

    monkeypatch.setattr(zz, "_r55_run_checkpoint", _checkpoint)
    with pytest.raises(zz._R55RunRefused) as ei:
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            zegrid_halo_px=8,
        )
    assert calls["n"] >= 3
    assert ei.value.storage_fallback_eligible is False
    assert ei.value.record["storage_fallback_eligible"] is False
    assert ei.value.record["fallback_exhausted"] is True
    assert zz._STORAGE_EFFECTIVE == "bounded"
    assert zz._STORAGE_FALLBACK_COUNT == 1
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


def test_runtime_bounded_fits_under_governing_latched_budget(
    tmp_path, monkeypatch,
):
    """Dense refuses under the latched budget, bounded REALLY fits it."""
    inp, out, _h = _pressure_env(
        tmp_path, monkeypatch, dense_mib=1024, bounded_mib=4,
        raw=_DENSE_OK_RAW,
    )
    governing = int(1.4 * 2 ** 30)
    state = _inject_wave1_latched(monkeypatch, governing)
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
    )
    assert state["n"] >= 2
    manifest = _manifest(out)
    storage = manifest["storage"]
    assert storage["effective"] == "bounded"
    assert storage["fallback_count"] == 1
    assert storage["transitions"][0]["phase"] == "run_wave_1"
    # The governing latched observation is preserved and really governs the
    # accepted checkpoint of the retried boundary.
    cps = manifest["r55_run_telemetry"]["checkpoints"]
    governed = [
        cp for cp in cps
        if cp.get("latched_host")
        and int(cp["latched_host"]["raw_available_bytes"]) == governing
    ]
    assert governed, cps
    cp = governed[-1]
    assert int(cp["host"]["raw_available_bytes"]) == governing
    assert cp["host_source"] == "fresh+latched"
    # Shrink-only vs the initial admitted candidate (no K/tile/workers growth).
    block = manifest["r55_scheduler"]
    init = block["initial_candidate"]
    final = block["chosen_candidate"]
    assert int(final["units_in_flight"]) <= int(init["units_in_flight"])
    assert int(final["workers_per_unit"]) <= int(init["workers_per_unit"])
    if init.get("tile_px") is not None and final.get("tile_px") is not None:
        assert int(final["tile_px"]) <= int(init["tile_px"])
    assert (out / "mosaic_grid.fits").exists()


def test_runtime_disk_shortage_never_falls_back_public(tmp_path, monkeypatch):
    """F5: a disk shortage at the final CPU rung is DISK, never eligible RAM."""
    inp, out, _h = _pressure_env(
        tmp_path, monkeypatch, dense_mib=1024, bounded_mib=4,
        raw=_DENSE_OK_RAW,
    )
    real_sample = zz._r55_run_checkpoint_sample
    calls = {"n": 0}

    def _sample(disk_root):
        facts = real_sample(disk_root)
        calls["n"] += 1
        if calls["n"] == 2:
            facts["disk"]["free_bytes"] = 1
        return facts

    monkeypatch.setattr(zz, "_r55_run_checkpoint_sample", _sample)
    with pytest.raises(zz._R55RunRefused) as ei:
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            zegrid_halo_px=8,
        )
    assert ei.value.reason_code == zz.renv.REASON_REFUSE_DISK
    assert ei.value.storage_fallback_eligible is False
    assert ei.value.record["classification"]["saw_disk"] is True
    assert zz._STORAGE_EFFECTIVE == "dense"
    assert zz._STORAGE_FALLBACK_COUNT == 0
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


def test_runtime_vram_refusal_stays_vram_and_never_eligible(
    tmp_path, monkeypatch,
):
    """L-3: a VRAM refusal at the final rung keeps its code, never RAM."""
    from dataclasses import replace

    inp, out, _h = _pressure_env(
        tmp_path, monkeypatch, dense_mib=1024, bounded_mib=4,
        raw=_DENSE_OK_RAW,
    )
    real_materialize = zz.zsched.materialize_active_candidate
    real_admit = zz._r55_active_admit
    state = {"armed": False, "vram": 0}

    def _admit(*args, **kwargs):
        result = real_admit(*args, **kwargs)
        state["armed"] = True
        return result

    def _materialize(context, candidate):
        if state["armed"]:
            state["vram"] += 1
            return replace(
                candidate,
                feasible=False,
                reason_code=zz.renv.REASON_REFUSE_VRAM,
                infeasible_reason="injected VRAM shortfall at the last rung",
            )
        return real_materialize(context, candidate)

    monkeypatch.setattr(zz, "_r55_active_admit", _admit)
    monkeypatch.setattr(zz.zsched, "materialize_active_candidate", _materialize)
    monkeypatch.setattr(
        zz.zsched, "next_shrink", lambda current, ctx: (None, None)
    )
    with pytest.raises(zz._R55RunRefused) as ei:
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            zegrid_halo_px=8,
        )
    assert state["vram"] >= 1
    assert ei.value.reason_code == zz.renv.REASON_REFUSE_VRAM
    assert ei.value.storage_fallback_eligible is False
    assert ei.value.record["classification"]["saw_non_memory"] is True
    assert (
        ei.value.record["classification"]["non_memory_reason"]
        == zz.renv.REASON_REFUSE_VRAM
    )
    assert zz._STORAGE_EFFECTIVE == "dense"
    assert zz._STORAGE_FALLBACK_COUNT == 0
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


def test_runtime_recovery_uses_min_of_governing_and_fresh(
    tmp_path, monkeypatch,
):
    """Adversarial: latched HIGH (8 GiB) but fresh LOW (512 MiB) -> min wins."""
    inp, out, holder = _pressure_env(
        tmp_path, monkeypatch, dense_mib=1024, bounded_mib=4,
        raw=_DENSE_OK_RAW,
    )
    # After wave 0 the machine reports only 1.4 GiB (bounded fits, dense floor
    # is ~1364 MiB > the 1.26 GiB governing budget).
    original_sample = zz._r55_run_checkpoint_sample
    calls = {"n": 0}

    def _sample(disk_root):
        calls["n"] += 1
        if calls["n"] == 2:
            holder["raw"] = int(1.4 * 2 ** 30)
        return original_sample(disk_root)

    monkeypatch.setattr(zz, "_r55_run_checkpoint_sample", _sample)
    _inject_wave1_latched(monkeypatch, 8 * 2 ** 30)
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
    )
    manifest = _manifest(out)
    cps = manifest["r55_run_telemetry"]["checkpoints"]
    governed = [
        cp for cp in cps
        if cp.get("latched_host")
        and int(cp["latched_host"]["raw_available_bytes"]) == 8 * 2 ** 30
    ]
    assert governed, cps
    cp = governed[-1]
    # min(8 GiB latched, 1.4 GiB fresh) = 1.4 GiB fresh governs (it is LOWER).
    assert int(cp["host"]["raw_available_bytes"]) == int(1.4 * 2 ** 30)
    assert cp["host_source"] == "fresh"
    assert (out / "mosaic_grid.fits").exists()


def test_runtime_recovery_requires_structured_eligibility():
    """A bare R55_REFUSE_HOST_RAM is NOT enough to switch storage profiles."""
    record = {
        "schema": "test", "status": "refused",
        "reason_code": "R55_REFUSE_HOST_RAM",
        "detail": "malformed observation relabelled as RAM",
        "wave_index": 0, "remaining_units": 3,
        "shrink_steps": [], "unavailable_shrinks": [],
    }
    exc = zz._R55RunRefused(
        "R55_REFUSE_HOST_RAM", "malformed observation", record=record,
    )
    assert exc.storage_fallback_eligible is False
    assert exc.record["storage_fallback_eligible"] is False



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
# Portable transport + cache close/remove
# ---------------------------------------------------------------------------

def test_bounded_task_tuple_is_pickle_portable():
    import pickle

    task = (
        SimpleNamespace(cell_id="c0", row=0, col=0),
        SimpleNamespace(patch_shape_hw=(8, 8)),
        SimpleNamespace(patch_ids=["a", "b"]),
        "/tmp/does-not-matter",
        "stream",
        123,
        "bounded",
    )
    clone = pickle.loads(pickle.dumps(task))
    assert len(clone) == 7
    assert clone[6] == "bounded"
    assert clone[4] == "stream"
    assert clone[0].cell_id == "c0"
    assert clone[2].patch_ids == ["a", "b"]


def test_bounded_cell_closes_and_removes_its_cache(tmp_path):
    """The product cell path closes the provider then deletes the cache."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lot_b_helpers",
        Path(__file__).resolve().parent / "test_zegrid_r52_ooc_storage_lot_b.py",
    )
    lotb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lotb)

    cache = tmp_path / "cache_bounded_close"
    frames, masks, ids = lotb._write_cache(cache)
    cfg = lotb.zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
    with pytest.MonkeyPatch.context() as mp:
        lotb._install_cell_globals(mp, cache, cfg)
        task = (
            lotb._FakeCell(), lotb._FakePatch(), lotb._FakeMem(ids),
            str(cache), "stream", 123, "bounded",
        )
        prepared = zz._prepare_stack_cell(task)
        result = zz._consume_stack_cell(prepared)
    assert result["record"]["storage"]["open_failures"] == 0
    assert result["record"]["storage"]["sink"]["type"] == "BoundedCountingRejectionSink"
    assert not cache.exists(), "bounded cell cache was not removed"


def test_public_auto_double_refusal_is_loud(tmp_path, monkeypatch):
    """AUTO dense fails AND bounded fails -> ONE loud error naming both."""
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    # Both profiles are budget-infeasible; bounded does not magically fit.
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: 8192 * MIB,
    )
    with pytest.raises(zz.zpar.ModeInfeasible) as ei:
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
        )
    text = str(ei.value)
    assert "in either storage profile" in text
    assert "dense:" in text and "bounded:" in text
    assert zz._STORAGE_EFFECTIVE == "bounded"
    assert zz._STORAGE_FALLBACK_COUNT == 1
    assert not (out / "zegrid_manifest.json").exists()
    assert not (out / "mosaic_grid.fits").exists()


# ---------------------------------------------------------------------------
# Pre-gauge ACTIVE admission: typed eligibility + real public retry
# ---------------------------------------------------------------------------

def test_preflight_admission_ram_refusal_falls_back_public(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            1 * MIB if storage_profile == "bounded" else 16 * MIB
        ),
    )
    real_admit = zz._r55_active_admit
    state = {"injected": False}

    def _admit(*args, **kwargs):
        if not state["injected"] and zz._STORAGE_EFFECTIVE == "dense":
            state["injected"] = True
            raise zz.zsched.ActivePlanRejected(
                zz.renv.REASON_REFUSE_HOST_RAM,
                "injected capacity miss before the gauge",
            )
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(zz, "_r55_active_admit", _admit)
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_active_zconfig("active"), zegrid_halo_px=8,
    )
    storage = _manifest(out)["storage"]
    assert state["injected"] is True
    assert storage["effective"] == "bounded"
    assert storage["fallback_count"] == 1
    assert storage["transitions"][0]["phase"] == "admission_preflight"
    assert (out / "mosaic_grid.fits").exists()


def test_preflight_non_memory_error_does_not_fall_back(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            1 * MIB if storage_profile == "bounded" else 16 * MIB
        ),
    )

    def _admit(*args, **kwargs):
        raise zz.zsched.SchedulerConfigError(
            "JUNIOR: invalid scheduler configuration, NOT memory pressure"
        )

    monkeypatch.setattr(zz, "_r55_active_admit", _admit)
    with pytest.raises(RuntimeError):
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            zegrid_halo_px=8,
        )
    assert zz._STORAGE_EFFECTIVE == "dense"
    assert zz._STORAGE_FALLBACK_COUNT == 0
    assert not (out / "zegrid_manifest.json").exists()
    assert not (out / "mosaic_grid.fits").exists()


def test_observation_failure_is_fail_closed_public(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _DENSE_OK_RAW)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile_size=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: (
            1 * MIB if storage_profile == "bounded" else 16 * MIB
        ),
    )
    original = zz.zrt.PressureLatch.consume_validated
    state = {"injected": False}

    def _consume(self):
        if not state["injected"]:
            state["injected"] = True
            self.record_error(RuntimeError("JUNIOR: MemAvailable collection failed"))
        return original(self)

    monkeypatch.setattr(zz.zrt.PressureLatch, "consume_validated", _consume)
    with pytest.raises(zz._R55RunRefused) as ei:
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_active_zconfig("active"),
            zegrid_halo_px=8,
        )
    assert state["injected"] is True
    assert ei.value.reason_code == zz.renv.REASON_REFUSE_HOST_RAM
    assert ei.value.storage_fallback_eligible is False
    assert zz._STORAGE_EFFECTIVE == "dense"
    assert zz._STORAGE_FALLBACK_COUNT == 0
    assert not (out / "zegrid_manifest.json").exists()
    assert not (out / "mosaic_grid.fits").exists()


# ---------------------------------------------------------------------------
# REAL CUDA + bounded provider / counting sink combined path
# ---------------------------------------------------------------------------

CORE_PLANES = (
    "science_core",
    "estimator_weight_sum_core",
    "support_w1_core",
    "support_w2_core",
    "n_eff_support_core",
    "valid_mask_core",
    "surviving_sample_count_core",
)


def test_real_cuda_dense_vs_bounded_provider_and_sink_parity(tmp_path, monkeypatch):
    """The COMBINED path: CuPy backend + file-backed provider + counting sink.

    Not a planner verdict and not the GPU-only canonical suite: the product
    ``_run_cell_stream`` seam runs the canonical engine with ``backend="gpu"``
    on BOTH profiles, so the same device kernels consume the dense memmap
    provider and the bounded provider + fresh counting sink. The seven core
    planes must be bit-identical and the bounded sink must count the whole
    interior partition with real large writes.
    """
    pytest.importorskip("cupy")
    import numpy as np
    from zemosaic.core import canonical_stacking as csx
    from zemosaic.core.zegrid import file_provider as zfp
    from zemosaic.core.zegrid import science_adapter as zsa

    if not csx.canonical_gpu_available():
        pytest.skip("no CUDA device available")

    n, h, w = 9, 512, 512
    rng = np.random.default_rng(11)
    frames, masks = [], []
    for i in range(n):
        f = (
            rng.normal(0, 3, (h, w, 3)).astype(np.float32)
            + np.array([50, 55, 60], np.float32)
            + np.float32(2.0 * i)
        )
        f[0:2, :, :] = np.nan
        frames.append(f)
        m = np.ones((h, w), dtype=bool)
        if i == 3:
            m[:, 40:] = False
        masks.append(m)
    ids = [f"f{i}" for i in range(n)]
    cache = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)

    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell", backend="gpu")
    monkeypatch.setattr(zz, "_STACK_CONFIG", cfg)
    monkeypatch.setattr(zz, "_STACK_GAUGE", None)
    monkeypatch.setattr(zz, "_STACK_GLOBAL_FRAME_IDS", None)
    monkeypatch.setattr(zz, "_STACK_GLOBAL_REF", None)
    monkeypatch.setattr(zz, "_STACK_TILE_SIZE", 64)
    monkeypatch.setattr(zz, "_STACK_CACHE_WORKERS", 1)

    class _Patch:
        cell_id = "c"
        patch_shape_hw = (h, w)
        core_slice = SimpleNamespace(y0=0, y1=h, x0=0, x1=w)
        patch = SimpleNamespace(x0=0, y0=0, x1=w, y1=h, height=h, width=w)

    dense_info, bounded_info = {}, {}
    dense_mt, dense_sres = zz._run_cell_stream(
        cache, _Patch(), cfg, None, tile_size=64,
        storage_profile="dense", storage_info=dense_info,
    )
    bounded_mt, bounded_sres = zz._run_cell_stream(
        cache, _Patch(), cfg, None, tile_size=64,
        storage_profile="bounded", storage_info=bounded_info,
    )

    # The device really executed the rejection/combine stages on BOTH runs.
    for sres in (dense_sres, bounded_sres):
        prov = sres.result.provenance
        assert prov["backend"]["requested"] == "gpu", prov["backend"]
        assert "gpu" in str(prov["backend"]).lower()

    from zemosaic.core.zegrid import assembly as za
    dense_core = za.crop_all_planes_to_core(dense_mt)
    bounded_core = za.crop_all_planes_to_core(bounded_mt)
    for name in CORE_PLANES:
        np.testing.assert_array_equal(
            dense_core[name], bounded_core[name], err_msg=name
        )

    assert dense_info["provider"] == "MemmapCanonicalProvider"
    assert bounded_info["provider"] == "BoundedMemmapCanonicalProvider"
    assert bounded_info["open_failures"] == 0
    sink = bounded_info["sink"]
    assert sink["type"] == "BoundedCountingRejectionSink"
    total_mask_bytes = n * h * w * 3
    assert total_mask_bytes > 4 * MIB
    assert sink["written_cells"] == total_mask_bytes
    assert sink["writes"] > 0
    assert sink["flush_count"] == sink["writes"]
    assert sink["max_write_bytes"] > 0
    # Bounded subdivision: the counting views never exceed the sink block size
    # even though the whole mask is > 4 MiB.
    from zemosaic.core.zegrid import rejection_sink as zrs
    assert sink["max_block_bytes"] <= int(zrs.SINK_BLOCK_BYTES)
    assert sink["written_cells"] // max(1, sink["max_block_bytes"]) > 1
    assert bounded_info["rejection_mask_available"] is False


# ---------------------------------------------------------------------------
# REAL spawn child: bounded profile travels in the task, not the parent global
# ---------------------------------------------------------------------------

class _SpawnCell:
    cell_id = "spawn-c0"
    row = 0
    col = 0


class _SpawnPatch:
    cell_id = "spawn-c0"
    patch_shape_hw = (32, 32)
    core_slice = SimpleNamespace(y0=0, y1=32, x0=0, x1=32)
    patch = SimpleNamespace(x0=0, y0=0, x1=32, y1=32, height=32, width=32)


class _SpawnMem:
    def __init__(self, ids):
        self.patch_ids = list(ids)


def _spawn_bounded_worker(cache_dir, ids, conn):
    """Runs in a FRESH spawn interpreter with a divergent module global."""
    try:
        from pathlib import Path as _P

        from zemosaic import zemosaic_zegrid_mode as _zz
        from zemosaic.core.zegrid import file_provider as _zfp
        from zemosaic.core.zegrid import science_adapter as _zsa

        _zz._STACK_STORAGE_PROFILE = "dense"  # divergent child global
        _zz._STACK_CONFIG = _zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
        _zz._STACK_GAUGE = None
        _zz._STACK_GLOBAL_FRAME_IDS = None
        _zz._STACK_GLOBAL_REF = None
        _zz._build_one_cell_cache = lambda *a, **k: _zfp.load_cache_manifest(a[5])
        task = (
            _SpawnCell(), _SpawnPatch(), _SpawnMem(ids), str(cache_dir),
            "stream", 0, "bounded",
        )
        prepared = _zz._prepare_stack_cell(task)
        result = _zz._consume_stack_cell(prepared)
        conn.send({
            "ok": True,
            "prepared_profile": prepared.get("storage_profile"),
            "record_profile": result["record"].get("storage_profile"),
            "provider": result["record"]["storage"]["provider"],
            "sink": result["record"]["storage"]["sink"]["type"],
            "open_failures": result["record"]["storage"]["open_failures"],
            "cache_exists": _P(cache_dir).exists(),
            "child_global": _zz._STACK_STORAGE_PROFILE,
        })
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def test_bounded_task_runs_in_real_spawn_child(tmp_path):
    import multiprocessing as mp

    import numpy as np
    from zemosaic.core.zegrid import file_provider as zfp

    n, h, w = 4, 32, 32
    rng = np.random.default_rng(21)
    frames = [
        (rng.normal(0, 2, (h, w, 3)).astype(np.float32) + np.float32(10 + i))
        for i in range(n)
    ]
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    ids = [f"spawn{i}" for i in range(n)]
    cache = tmp_path / "spawn_cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_spawn_bounded_worker, args=(str(cache), ids, child_conn)
    )
    proc.start()
    child_conn.close()
    payload = parent_conn.recv() if parent_conn.poll(120) else None
    proc.join(120)
    assert proc.exitcode == 0, payload
    assert payload and payload["ok"], payload
    assert payload["prepared_profile"] == "bounded"
    assert payload["record_profile"] == "bounded"
    assert payload["provider"] == "BoundedMemmapCanonicalProvider"
    assert payload["sink"] == "BoundedCountingRejectionSink"
    assert payload["open_failures"] == 0
    assert payload["cache_exists"] is False
    # The child global said dense; the TASK profile won.
    assert payload["child_global"] == "dense"


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


# ---------------------------------------------------------------------------
# ZM-ZEGRID-PLAN-REFUSAL-CLASSIFICATION — mixed halo/memory witness
# ---------------------------------------------------------------------------

def _write_large_corpus(tmp_path, *, side=512):
    """Deterministic 2x2 mosaic of ``side``-px TAN frames (no local corpus).

    Large enough that the PRODUCT floors (min_patch_area 50000 px,
    max_halo_overhead 0.5) are meaningful at halo 128, and small enough that the
    planner-only witness runs in seconds.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    scale = 0.001
    inp = tmp_path / "large_in"
    inp.mkdir()
    names = []
    for i in range(4):
        col, row = i % 2, i // 2
        ra = 10.0 + col * side * scale
        dec = 30.0 - row * side * scale
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [side / 2.0, side / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (side, side)
        hdu = fits.PrimaryHDU(np.zeros((side, side), dtype=np.float32))
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = 10.0
        name = f"frame_{i:04d}.fits"
        hdu.writeto(inp / name, overwrite=True)
        names.append(name)
    return inp


def test_active_auto_halo_conflict_coarser_rescue_layout_refusal_falls_back_to_bounded(
    tmp_path, monkeypatch
):
    """Halo conflict + coarser rescue over budget => AUTO switches ONCE.

    This is the REAL product decision path (``_plan_bundle_auto``, the planner
    used by ``run_zegrid_mode``) with the PRODUCT scientific floors untouched:
    the factor-1 grid is rejected SOLELY by ``max_halo_overhead``, the nearest
    floor-feasible coarser rescue (1x2) misses the DENSE budget, and the SAME
    request succeeds under bounded. No science is executed (planner only).
    """
    from zemosaic.core.zegrid import geometry as zg

    inp = _write_large_corpus(tmp_path)
    frames, rejected = zg.read_manifest(inp)
    assert len(frames) == 4
    assert rejected == []
    canvas = zg.build_canvas(frames)

    # PRODUCT floors are used verbatim: no relaxation anywhere in this witness.
    floors = zal.ScientificFloors()
    assert floors.min_patch_area_px == 50000
    assert floors.max_halo_overhead == 0.5

    budget = 4096 * MIB
    raw = budget * rcontract.PLAN_FACTOR_DEN // rcontract.PLAN_FACTOR_NUM

    # Memory injection (accepted seam): in-memory never fits; the dense
    # streaming estimate exceeds the budget while the bounded one fits.
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _fake_stream_bytes)

    zz._storage_policy_reset(None)
    bundle = zz._plan_bundle_auto(
        canvas=canvas, descs=frames, footprints=None,
        raw=raw, budget=budget, active=True, emit=None,
        bundle_common={"cpu_count": 2, "gpu_used": False, "cache_root": None,
                       "halo_px": 128},
        layout_substep=None,
    )

    storage = zz._storage_policy_snapshot()
    assert storage["requested"] == "auto"
    assert storage["effective"] == "bounded"
    assert storage["fallback_count"] == 1
    assert len(storage["transitions"]) == 1
    assert storage["transitions"][0]["from"] == "dense"
    assert storage["transitions"][0]["to"] == "bounded"
    assert storage["dense_refusal"]["phase"] == "plan_layout"
    assert storage["dense_refusal"]["reason_code"] == zz.renv.REASON_REFUSE_HOST_RAM
    # Proof the witness exercised the COARSER-rescue memory branch (not a
    # generic "finest candidate" refusal).
    assert "nearest floor-feasible COARSER rescue" in storage["dense_refusal"]["detail"]
    assert "needs 8192.0 MiB > budget 4096.0 MiB" in storage["dense_refusal"]["detail"]

    assert bundle.storage_profile == "bounded"
    assert (bundle.layout["nx"], bundle.layout["ny"]) == (1, 2)
    assert bundle.layout["layout_origin"] == "auto_coarser_rescue"
    assert all(t[6] == "bounded" for t in bundle.cell_tasks if t[4] is not None)
    assert all(t[4] == "stream" for t in bundle.cell_tasks if t[4] is not None)
