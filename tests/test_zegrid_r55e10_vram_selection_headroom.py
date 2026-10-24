"""ZM-ZEGRID-R55-E10 — SELECTION-time VRAM headroom for the ACTIVE GPU tile.

R55-E9 derived the executable GPU ladder from the fresh 70 % VRAM budget, so the
selected tile filled that budget to the razor edge (recorded M106 witness: 506 px,
1.2 MiB of margin). At RUN time ``fresh_vram_revalidation`` re-reads free VRAM once
the CuPy context is live; a ~2 MiB drop was enough to reject that tile and force a
whole-batch CPU retry.

R55-E10 keeps the whole ladder but PRUNES its top to rungs whose exact device
workspace fits 85 % of the fresh 70 % budget (~60 % of free VRAM). Selection only:
the 70 % policy, the frozen ladder and the RUN-time revalidation are unchanged.

These tests pin:

* the recorded M106 witness now selects 448 px (not 506, not 128) with a real
  margin under the 70 % budget (>= 0.7 GiB) and an unchanged host/layout/halo/K;
* the headroom property over a range of free-VRAM values (largest declared rung
  whose exact device peak fits the headroom budget);
* the exact recorded failure scenario (free - ~2 MiB) admits the new tile and
  would have rejected the old one;
* shrink-only pressure is preserved on the pruned ladder;
* the small/large device matrix and honest low-VRAM refusal;
* diagnostics expose the headroom fraction and effective selection budget.

Model-only: no CuPy call, no pixel mosaic, no GPU performance claim.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import gpu_readiness as zgr

MIB = 2 ** 20
GIB = 2 ** 30
LIGHTS = Path("/home/tristan/M106/lights")

# Recorded Windows M106 facts (R55-E8 real run, unchanged by E10).
WITNESS_FRAMES = 66
WITNESS_PATCH_HW = (1767, 1330)
WITNESS_FREE = 7463763968
WITNESS_TOTAL = 8589410304
WITNESS_B_PLAN = 10481732812
WITNESS_RAW = 13102166016
WITNESS_CPU = 16
WITNESS_HALO = 128
WITNESS_VRAM_BUDGET = 5224634777  # floor(free * 70 / 100)
WITNESS_SELECTION_BUDGET = 4440939560  # floor(5224634777 * 85 / 100)
WITNESS_OLD_TILE = 506
WITNESS_SELECTED_TILE = 448
WITNESS_OLD_DEVICE = 5223334912  # exact device workspace of tile506
WITNESS_SELECTED_DEVICE = 4131895552  # exact device workspace of tile448

PATCH_4096 = (4096, 4096)
N_64 = 64


def _readiness(free=WITNESS_FREE, total=WITNESS_TOTAL):
    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="model-only",
        device_count=1,
        device_index=0,
        device_name="simulated-device",
        vram_total_bytes=int(total),
        vram_free_bytes=int(free),
    )


def _probe(free=WITNESS_FREE, total=WITNESS_TOTAL):
    return {
        "available": True,
        "device": "simulated-device",
        "vram_total_bytes": int(total),
        "vram_free_bytes": int(free),
    }


def _device_peak(frames, patch_hw, tile, channels=3):
    side = int(tile) + 2 * int(zgpu.GPU_TAPER_HALO_PX)
    return int(zgpu.gpu_tile_workspace_bytes(int(frames), side, side, channels))


def _selection_shape(free, frames=WITNESS_FRAMES, patch_hw=WITNESS_PATCH_HW):
    """Pure E10 selection shape: policy budget, headroom budget, ladder, peaks."""
    budget = int(zgr.policy_budget_bytes(int(free), zgr.INITIAL_TARGET_PERCENT))
    selection_budget = (
        budget * int(zz._R55E10_VRAM_SELECTION_HEADROOM_PERCENT)
    ) // 100
    raw_max = int(zgpu.choose_gpu_tile_size(int(frames), 3, patch_hw, budget))
    declared = zz._r55_active_gpu_tile_ladder(raw_max)
    peaks = {int(t): _device_peak(frames, patch_hw, t) for t in declared}
    pruned = tuple(t for t in declared if peaks[int(t)] <= selection_budget)
    return budget, selection_budget, raw_max, declared, peaks, pruned


@pytest.fixture(scope="module")
def m106():
    if not LIGHTS.is_dir():
        pytest.skip("M106 lights directory absent")
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    assert len(frames) == WITNESS_FRAMES
    return frames, canvas


@pytest.fixture(autouse=True)
def _deterministic_host(monkeypatch):
    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=4 * 2 ** 40, used=0, free=4 * 2 ** 40),
    )
    monkeypatch.setattr(zz.os, "cpu_count", lambda: WITNESS_CPU)


def _windows_bundle(m106, free=WITNESS_FREE):
    frames, canvas = m106
    return zz._build_execution_bundle(
        canvas,
        frames,
        None,
        raw_available_bytes=WITNESS_RAW,
        budget_bytes=WITNESS_B_PLAN,
        halo_px=WITNESS_HALO,
        cpu_count=WITNESS_CPU,
        gpu_used=True,
        vram_budget_bytes=zgpu.vram_budget_bytes(
            {"available": True, "vram_free_bytes": int(free)}
        ),
        cache_root="/tmp",
    )


def _decide(m106, *, free=WITNESS_FREE, total=WITNESS_TOTAL):
    frames, canvas = m106
    bundle = _windows_bundle(m106, free=free)
    return bundle, zz._r55_active_gpu_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=WITNESS_HALO,
        cache_root="/tmp",
        output_dir="/tmp",
        frame_ids=tuple(f.frame_id.logical_path for f in frames),
        gpu_probe=_probe(free=free, total=total),
        readiness=_readiness(free=free, total=total),
    )


# ---------------------------------------------------------------------------
# A. Recorded M106 witness: real headroom, everything else unchanged.
# ---------------------------------------------------------------------------

def test_e10_witness_selects_a_tile_with_real_headroom(m106):
    bundle, decision = _decide(m106)
    selected = int(decision.mapping["gpu_streaming_tile"])

    # A real tile, not the old 128 cap and not the razor-edge 506 rung.
    assert selected > 128
    assert selected != WITNESS_OLD_TILE
    assert selected == WITNESS_SELECTED_TILE

    # The unfiltered E9 maximum is still reported (auditable), while the
    # DECLARED ladder is the headroom-pruned subset.
    assert int(decision.gpu_tile_max) == WITNESS_OLD_TILE
    assert selected in decision.gpu_tile_ladder
    assert WITNESS_OLD_TILE not in decision.gpu_tile_ladder
    assert int(decision.gpu_selection_budget_bytes) == WITNESS_SELECTION_BUDGET

    peak = int(decision.device_peak_bytes_by_tile[selected])
    assert peak == WITNESS_SELECTED_DEVICE
    assert peak <= decision.gpu_selection_budget_bytes
    # >= ~0.7 GiB margin under the UNCHANGED 70 % budget.
    margin = WITNESS_VRAM_BUDGET - peak
    assert margin >= int(0.7 * GIB), margin
    # ... and the old selection had less than 2 MiB: that was the defect.
    assert WITNESS_VRAM_BUDGET - WITNESS_OLD_DEVICE < 2 * MIB

    # Host plan, layout, halo, frame identity and K=1 are unchanged.
    assert int(decision.plan_set.aggregate_host_peak_bytes) <= WITNESS_B_PLAN
    assert int(decision.plan_set.aggregate_device_peak_bytes) == peak
    assert decision.facts["layout"] == (2, 2)
    assert int(decision.facts["halo_px"]) == WITNESS_HALO
    assert len(decision.facts["frame_ids"]) == WITNESS_FRAMES
    assert decision.facts["frame_ids"] == tuple(
        f.frame_id.logical_path for f in m106[0]
    )
    assert decision.mapping["units_in_flight"] == 1
    assert decision.mapping["single_owner"] is True

    applied = zz._r55_active_gpu_apply(
        bundle, mapping=decision.mapping, decision=decision, stage="preflight"
    )
    assert applied.gpu_streaming_tile == selected
    assert applied.envelope.gpu_tile == selected
    assert applied.envelope.layout == (2, 2)
    assert applied.envelope.aggregate_bytes <= WITNESS_B_PLAN


# ---------------------------------------------------------------------------
# B. Headroom property over a range of fresh free-VRAM observations.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gib", [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48]
)
def test_e10_selection_is_the_largest_rung_that_fits_the_headroom_budget(gib):
    free = int(gib * GIB)
    (
        budget,
        selection_budget,
        raw_max,
        declared,
        peaks,
        pruned,
    ) = _selection_shape(free)

    assert declared, f"unexpectedly empty ladder at {gib} GiB"
    assert selection_budget == budget * 85 // 100
    assert pruned, f"no rung fits the headroom budget at {gib} GiB"
    # Every declared rung keeps its own exact peak and fits the headroom budget.
    assert all(peaks[int(t)] <= selection_budget for t in pruned)
    # The pruned top IS the largest declared rung satisfying the headroom.
    assert pruned[0] == max(t for t in declared if peaks[int(t)] <= selection_budget)
    assert max(pruned) == pruned[0]
    # Nothing above the computed maximum ever survives, and the ladder stays
    # strictly descending with unchanged lower rungs.
    assert all(t <= raw_max for t in pruned)
    assert all(a > b for a, b in zip(pruned, pruned[1:]))
    assert pruned[-1:] == declared[-1:]


def test_e10_pruned_ladder_keeps_the_full_lower_body():
    """Only the razor-edge TOP is removed; the declared body is a suffix."""
    _b, selection_budget, _raw, declared, peaks, pruned = _selection_shape(
        WITNESS_FREE
    )
    assert pruned == tuple(t for t in declared if peaks[int(t)] <= selection_budget)
    assert pruned == declared[declared.index(pruned[0]):]
    assert declared[0] == WITNESS_OLD_TILE
    assert pruned[0] == WITNESS_SELECTED_TILE


def test_e10_decided_ladder_matches_the_pure_headroom_property(m106):
    for gib in (2, 4, 8, 16):
        free = int(gib * GIB)
        _b, selection_budget, raw_max, declared, peaks, pruned = _selection_shape(
            free
        )
        bundle, decision = _decide(
            m106, free=free, total=max(WITNESS_TOTAL, free)
        )
        assert int(decision.gpu_tile_max) == raw_max
        assert tuple(int(t) for t in decision.gpu_tile_ladder) == pruned
        assert int(decision.gpu_selection_budget_bytes) == selection_budget
        selected = int(decision.mapping["gpu_streaming_tile"])
        # Host feasibility may shrink BELOW the ladder top, never above it.
        assert selected in pruned
        assert selected <= pruned[0]
        assert int(decision.device_peak_bytes_by_tile[selected]) <= selection_budget
        assert (
            int(decision.qualification["budget_bytes"]) == _b
            and int(decision.qualification["percent"]) == 70
        )


# ---------------------------------------------------------------------------
# B2. The EXACT recorded RUN-time fluctuation scenario.
# ---------------------------------------------------------------------------

def test_e10_two_mib_drop_at_run_time_keeps_the_selected_tile(m106):
    _bundle, decision = _decide(m106)
    selected = int(decision.mapping["gpu_streaming_tile"])
    selected_peak = int(decision.device_peak_bytes_by_tile[selected])
    old_peak = WITNESS_OLD_DEVICE

    drop = 2 * MIB
    free_after = WITNESS_FREE - drop
    fresh_budget = int(zgr.policy_budget_bytes(free_after, 70))
    assert fresh_budget < WITNESS_VRAM_BUDGET

    # The old razor-edge tile is rejected by the unchanged RUN-time check.
    assert old_peak > fresh_budget
    old = zgr.fresh_vram_revalidation(
        policy=decision.policy,
        current_percent=int(decision.qualification["percent"]),
        free_bytes=free_after,
        device_peak_bytes=old_peak,
    )
    assert old.action == "fallback_cpu"
    assert old.reason_code == zgr.REASON_REFUSE_VRAM

    # The E10 tile survives the same fluctuation with a wide margin.
    assert selected_peak <= fresh_budget
    new = zgr.fresh_vram_revalidation(
        policy=decision.policy,
        current_percent=int(decision.qualification["percent"]),
        free_bytes=free_after,
        device_peak_bytes=selected_peak,
    )
    assert new.action == "continue"

    # Quantify the tolerance: the selected tile only fails after a >1.4 GiB drop.
    minimal_free = -(-selected_peak * 100 // 70)
    assert WITNESS_FREE - minimal_free > 1.4 * GIB


# ---------------------------------------------------------------------------
# C. Pressure stays shrink-only on the pruned ladder.
# ---------------------------------------------------------------------------

def test_e10_fresh_vram_pressure_shrinks_downward_from_the_pruned_top(m106):
    _bundle, decision = _decide(m106)
    chosen = decision.plan_set.chosen
    assert chosen.tile_px == WITNESS_SELECTED_TILE

    # 5.0 GB free at 70 % = 3.5 GB: the next fitting declared rung is 384.
    shrunk = zz._r55_next_gpu_vram_shrink(
        decision, chosen, free_bytes=5_000_000_000, percent=70
    )
    assert shrunk is not None
    assert shrunk.tile_px == 384
    assert shrunk.tile_px < chosen.tile_px

    # VRAM recovery never re-grows the frozen ladder.
    recovered = zz._r55_next_gpu_vram_shrink(
        decision, shrunk, free_bytes=64 * GIB, percent=70
    )
    assert recovered is None or recovered.tile_px < shrunk.tile_px


def test_e10_policy_and_legacy_constants_are_untouched():
    assert zz._R55E10_VRAM_SELECTION_HEADROOM_PERCENT == 85
    assert zgr.INITIAL_TARGET_PERCENT == 70
    assert zgr.VRAM_LADDER == (50, 65, 70, 75)
    assert zgpu.GPU_VRAM_SAFETY_FRACTION == 0.5
    assert zgpu.GPU_TAPER_HALO_PX == 9


# ---------------------------------------------------------------------------
# D. Small/large device matrix + honest low-VRAM refusal.
# ---------------------------------------------------------------------------

def test_e10_low_vram_refuses_with_a_closed_reason(m106):
    frames, canvas = m106
    bundle = _windows_bundle(m106)
    with pytest.raises(zz._R55GpuRefused) as exc:
        zz._r55_active_gpu_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=WITNESS_HALO,
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=tuple(f.frame_id.logical_path for f in frames),
            gpu_probe=_probe(free=20 * MIB),
            readiness=_readiness(free=20 * MIB),
        )
    assert exc.value.reason_code in (
        zgr.REASON_REFUSE_VRAM,
        zgr.REASON_VRAM_BELOW_MINIMUM,
    )


def test_e10_empty_headroom_ladder_refuses_honestly(m106, monkeypatch):
    """If the headroom budget admits no declared rung, refuse (closed reason).

    Not reachable on the recorded geometry (the 64 MiB policy floor is larger
    than the smallest rung); the constant is forced to exercise the guard.
    """
    frames, canvas = m106
    bundle = _windows_bundle(m106)
    monkeypatch.setattr(zz, "_R55E10_VRAM_SELECTION_HEADROOM_PERCENT", 0)
    with pytest.raises(zz._R55GpuRefused) as exc:
        zz._r55_active_gpu_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=WITNESS_HALO,
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=tuple(f.frame_id.logical_path for f in frames),
            gpu_probe=_probe(),
            readiness=_readiness(),
        )
    assert exc.value.reason_code == zgr.REASON_REFUSE_VRAM
    assert "headroom" in exc.value.detail


def test_e10_high_vram_still_exposes_larger_tiles_when_the_patch_permits():
    maxima = {}
    for gib in (2, 4, 8, 16, 24):
        free = int(gib * GIB)
        _b, _s, raw_max, _declared, _peaks, pruned = _selection_shape(
            free, frames=N_64, patch_hw=PATCH_4096
        )
        assert pruned
        maxima[gib] = (raw_max, pruned[0])
        assert pruned[0] <= raw_max
    ordered = [maxima[gib][1] for gib in (2, 4, 8, 16, 24)]
    assert ordered == sorted(ordered)
    assert maxima[24][1] > WITNESS_SELECTED_TILE
    assert maxima[16][1] > 256


# ---------------------------------------------------------------------------
# E. Diagnostics: headroom fraction + effective selection budget.
# ---------------------------------------------------------------------------

def test_e10_diagnostics_report_headroom_and_selection_budget(m106):
    bundle, decision = _decide(m106)
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=decision.mapping, decision=decision, stage="preflight"
    )
    block = zz._r55_gpu_tile_decision_dict(decision, decision.mapping, applied)

    # Existing fields are retained.
    for key in (
        "schema",
        "max_tile_px",
        "declared_ladder",
        "selected_tile_px",
        "selected_prefetch_units",
        "selected_host_added_bound_bytes",
        "selected_host_aggregate_bytes",
        "selected_device_peak_bytes",
        "host_added_bound_bytes_by_tile",
        "device_peak_bytes_by_tile",
        "vram_budget_bytes",
        "vram_percent",
    ):
        assert key in block

    assert block["max_tile_px"] == WITNESS_OLD_TILE
    assert block["selected_tile_px"] == WITNESS_SELECTED_TILE
    assert block["selected_device_peak_bytes"] == WITNESS_SELECTED_DEVICE
    assert block["vram_budget_bytes"] == WITNESS_VRAM_BUDGET
    assert block["vram_percent"] == 70

    # New E10 audit fields.
    assert block["selection_headroom_percent"] == 85
    assert block["selection_headroom_fraction"] == pytest.approx(0.85)
    assert block["selection_budget_bytes"] == WITNESS_SELECTION_BUDGET
    assert block["selected_device_selection_margin_bytes"] == (
        WITNESS_SELECTION_BUDGET - WITNESS_SELECTED_DEVICE
    )
    assert block["selected_device_selection_margin_bytes"] > 0
