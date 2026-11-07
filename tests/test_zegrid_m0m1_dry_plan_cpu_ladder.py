"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M0+M1 — dry-plan + CPU ladder below 32.

Bounded acceptance surface for the first vertical slice:

M1 (CPU ladder below 32):
* the CPU ladder is extended to 8..128 and the GPU ladder is a SEPARATE,
  unchanged 32..128 tuple (GPU below 32 stays refused);
* the scheduler validates the floor PER BACKEND (CPU >= 8, GPU >= 32);
* the canonical streaming executor is BIT-EXACT across the new below-32 rungs
  (taper borders + winsorized rejection + median, and kappa + mean), with no
  relaxed tolerance;
* the real product bundle (``_build_execution_bundle``) selects the below-32
  rung automatically from the existing AUTO path and publishes it to the
  worker initializer seam — no CLI override, no GUI control.

M0 (honest dry-plan):
* the M33 fact replay refuses tile32 at the 10,022 MiB PLAN budget and selects
  an honest below-32 rung;
* non-monotone fixed per-tile overhead is retained as honest candidate facts
  (tile8 can cost more than tile16) while the LARGEST feasible rung still wins;
* refusal classes are explicit (scientific floor / no in-memory plan / no
  bounded RAM tile / disk envelope / external working set required);
* the external working set (M2) is reported as REQUIRED but never as
  implemented, and unpriced disk terms are never zero-by-omission;
* the dry-plan is PURE: no filesystem writes, no gauge, no cache, no stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import dry_plan as zdry
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import scheduler as zs
from zemosaic.core.zegrid import streaming as zstream
from zemosaic.core.zegrid.resource_envelope import InvalidEnvelope

MIB = 2 ** 20


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _dense_profile_and_tile_reset(monkeypatch):
    """Isolate the run-scoped globals this suite touches (collection order)."""
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zz, "_STACK_TILE_SIZE", zz.STREAM_TILE_SIZE)


class _FakeModel:
    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


def _make_tan(i, h, w, scale=0.001, ra_step=0.004):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def _corpus(n=4, h=40, w=40):
    return [_make_tan(i, h, w) for i in range(n)]


_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(*_args, **_kwargs):
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


def _neutralize(monkeypatch):
    from zemosaic.core.zegrid import parallel as zpar

    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)

    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)


# ---------------------------------------------------------------------------
# M1 — ladders and scheduler floors
# ---------------------------------------------------------------------------

def test_cpu_ladder_below_32_gpu_ladder_unchanged():
    assert zz.CPU_STREAMING_TILE_LADDER == (
        128, 112, 96, 80, 64, 48, 32, 24, 20, 16, 12, 10, 8
    )
    assert zz.CPU_STREAMING_TILE_MINIMUM == 8
    assert zz.GPU_STREAMING_TILE_LADDER == (128, 112, 96, 80, 64, 48, 32)
    assert zz.GPU_STREAMING_TILE_MINIMUM == 32
    # The GPU ladder is a strict subset of the CPU ladder (no GPU-only rung).
    assert set(zz.GPU_STREAMING_TILE_LADDER) <= set(zz.CPU_STREAMING_TILE_LADDER)
    assert tuple(zz.CPU_STREAMING_TILE_LADDER) == zs.PRODUCT_TILE_LADDER
    assert tuple(zz.GPU_STREAMING_TILE_LADDER) == zs.GPU_PRODUCT_TILE_LADDER
    assert zs.MIN_CPU_TILE_PX == 8 and zs.MIN_TILE_PX == 32


def _ctx(tile_ladder, backend=zs.BACKEND_CPU, **over):
    base = dict(
        phase=zs.PHASE_CELL_STACK_STREAM,
        backend=backend,
        budget_bytes=1024 * MIB,
        raw_available_bytes=(1024 * MIB * 5) // 4,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=2,
        shared_bytes=1024,
        per_unit_stack_bytes=2048,
        per_unit_cache_bytes=1024,
        per_unit_prefetch_bytes=0,
        disk_free_bytes=10 * MIB,
        layout=(3, 2),
        refinement_rank=6,
        patch_hw=(40, 40),
        halo_px=8,
        channels=3,
        n_frames=4,
        tile_ladder=tuple(tile_ladder),
        prefetch_ladder=(0,),
    )
    base.update(over)
    return zs.PlanContext(**base)


def test_scheduler_accepts_cpu_tiles_below_32():
    candidates = zs.enumerate_candidates(_ctx(zz.CPU_STREAMING_TILE_LADDER))
    assert {c.tile_px for c in candidates if c.feasible} == {
        int(t) for t in zz.CPU_STREAMING_TILE_LADDER
    }


def test_scheduler_rejects_cpu_below_8_and_gpu_below_32():
    with pytest.raises(InvalidEnvelope):
        zs.enumerate_candidates(_ctx((16, 4)))
    with pytest.raises(InvalidEnvelope):
        zs._validate_context(_ctx((32, 16), backend=zs.BACKEND_GPU), allow_gpu=True)
    # GPU >= 32 is still accepted.
    zs._validate_context(_ctx((64, 32), backend=zs.BACKEND_GPU), allow_gpu=True)


# ---------------------------------------------------------------------------
# M1 — exact science parity below 32 (no tolerance, no skip)
# ---------------------------------------------------------------------------

def _ladder_corpus(n=6, h=200, w=200, seed=5):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(5):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        base += 40.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
    gains = np.linspace(0.96, 1.04, n)
    frames = [
        np.asarray(base * gains[i] + rng.normal(0, 2.0, (h, w)), dtype=np.float32)
        for i in range(n)
    ]
    frames[1][10:20, 10:20] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[4][:, w // 2:] = False
    return frames, masks


def _assert_bit_exact(a, b):
    for name in (
        "science", "estimator_weight_sum", "support_w1", "support_w2",
        "n_eff_support", "valid_mask", "surviving_sample_count",
        "rejection_mask",
    ):
        np.testing.assert_array_equal(
            getattr(a, name), getattr(b, name), err_msg=name
        )
    assert a.iterations_used == b.iterations_used
    assert a.initial_sample_count == b.initial_sample_count
    assert a.rejected_sample_count == b.rejected_sample_count
    assert a.rejected_fraction == b.rejected_fraction
    assert a.low_n_cell_count == b.low_n_cell_count
    assert a.degenerate_cell_count == b.degenerate_cell_count
    assert a.rejection_method == b.rejection_method
    assert a.provenance == b.provenance


@pytest.mark.parametrize("tile", (32, 24, 16, 10))
@pytest.mark.parametrize(
    "rejection,combine",
    (("winsorized_sigma_clip", "median"), ("kappa_sigma", "mean")),
)
def test_below_32_tile_science_is_bit_exact(tile, rejection, combine):
    frames, masks = _ladder_corpus()
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks, normalization="sky_mean",
        weighting="noise_variance", rejection=rejection, combine=combine,
        taper="footprint", taper_px=8.0,
    )
    ref = run_canonical_stack(req)
    got = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=tile
    )
    _assert_bit_exact(ref, got)


# ---------------------------------------------------------------------------
# M0 — dry-plan fact replay and honesty
# ---------------------------------------------------------------------------

# 31.7 GiB box replayed at the real M33 PLAN budget: (4 * raw) // 5 == 10,022 MiB.
_M33_RAW = 13_136_035_840
_M33_BUDGET_MIB = 10_022


def _m33_spec(**over):
    base = dict(
        raw_available_bytes=_M33_RAW,
        plan_budget_bytes=None,
        cells=[(22_414, (64, 64))],
        layout=(30, 51),
        refinement_rank=1_530,
        halo_px=8,
        channels=3,
        storage_profile="bounded",
        cpu_logical=8,
        disk_free_bytes=10 ** 12,
        per_unit_cache_bytes=10 ** 9,
        in_memory_bound_by_cell=[204_000 * MIB],
        floors={"min_patch_area_px": 1, "min_contributors": 1,
                "max_halo_overhead": 1e9},
    )
    base.update(over)
    return zdry.DryPlanSpec(**base)


def test_m33_replay_refuses_tile32_at_10022_mib_and_selects_below_32():
    report = zdry.build_dry_plan(_m33_spec())
    assert report["budget"]["plan_budget_bytes"] == _M33_BUDGET_MIB * MIB
    trials = {t["tile"]: t for t in report["cpu"]["trials"]}
    assert trials[32]["feasible"] is False
    assert trials[32]["predicted_bound_bytes"] > _M33_BUDGET_MIB * MIB
    selected = report["cpu"]["selected_tile"]
    assert selected is not None and selected < 32
    assert report["cpu"]["selected_working_set"] == "ram_tile"
    assert report["ok"] is True and report["refusal_class"] is None
    # The selected rung is the LARGEST feasible one (performance-first).
    assert selected == max(t for t, v in trials.items() if v["feasible"])
    assert report["external"]["required"] is False
    assert report["external"]["implemented"] is False


def test_dry_plan_non_monotone_fixed_overhead_is_reported_honestly():
    spec = zdry.DryPlanSpec(
        raw_available_bytes=4 * 2 ** 30,
        plan_budget_bytes=None,
        cells=[(1, (40, 40))],
        layout=(1, 1),
        refinement_rank=1,
        halo_px=8,
        channels=3,
        storage_profile="bounded",
        cpu_logical=4,
        disk_free_bytes=10 ** 12,
        per_unit_cache_bytes=10 ** 6,
    )
    report = zdry.build_dry_plan(spec)
    trials = {t["tile"]: t for t in report["cpu"]["trials"]}
    # The fixed per-tile overhead makes tile8 more expensive than tile16.
    assert trials[8]["predicted_bound_bytes"] > trials[16]["predicted_bound_bytes"]
    # Selection is still the largest feasible rung, never "smallest by default".
    feasible = [t for t, v in trials.items() if v["feasible"]]
    assert report["cpu"]["selected_tile"] == max(feasible)


def test_dry_plan_reports_external_required_but_not_implemented():
    spec = _m33_spec(
        raw_available_bytes=2 * 2 ** 30,
        cells=[(200_000, (128, 128))],
        in_memory_bound_by_cell=[900_000 * MIB],
    )
    report = zdry.build_dry_plan(spec)
    assert report["ok"] is False
    assert report["cpu"]["selected_tile"] is None
    assert report["refusal_class"] == zdry.REFUSAL_EXTERNAL_WORKING_SET_REQUIRED
    assert zdry.REFUSAL_NO_BOUNDED_RAM_TILE in report["refusal_causes"]
    assert zdry.REFUSAL_NO_IN_MEMORY_PLAN in report["refusal_causes"]
    assert report["external"] == {
        "required": True,
        "implemented": False,
        "future_mission": "M2",
        "unpriced_terms": list(zdry._EXTERNAL_UNPRICED_TERMS),
    }


def test_dry_plan_unpriced_disk_is_explicit_never_zero():
    report = zdry.build_dry_plan(_m33_spec(per_unit_cache_bytes=None))
    assert report["ok"] is True  # RAM rung is still identified honestly
    assert report["admissible_for_launch"] is False
    assert report["disk"]["pricing"] == zdry.DISK_UNPRICED
    assert report["disk"]["priced_envelope_bytes"] is None
    assert "external_spool_peak_bytes" in report["disk"]["unknown_terms"]
    assert "external_column_batch_bytes" in report["disk"]["unknown_terms"]


def test_dry_plan_scientific_floor_refuses_before_any_estimate(tmp_path, monkeypatch):
    calls = []
    real = zstream.estimate_streaming_peak_kib

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(zstream, "estimate_streaming_peak_kib", spy)
    before = sorted(p.name for p in Path(tmp_path).iterdir())
    report = zdry.build_dry_plan(
        zdry.DryPlanSpec(
            raw_available_bytes=4 * 2 ** 30,
            plan_budget_bytes=None,
            cells=[(4, (40, 40))],
            layout=(3, 2),
            refinement_rank=6,
            halo_px=8,
            geometry_admissible=False,
            scientific_floor_reason="min_patch_area_px",
        )
    )
    after = sorted(p.name for p in Path(tmp_path).iterdir())
    assert calls == []  # refusal before any scientific estimate
    assert report["refusal_class"] == zdry.REFUSAL_SCIENTIFIC_FLOOR
    assert report["cpu"]["trials"] == []
    assert report["external"]["implemented"] is False
    assert before == after  # no side effect at all
    json.dumps(report)  # strict JSON


def test_dry_plan_is_strict_json_and_side_effect_free(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in Path(tmp_path).iterdir())
    report = zdry.build_dry_plan(_m33_spec())
    after = sorted(p.name for p in Path(tmp_path).iterdir())
    assert before == after
    encoded = json.dumps(report, sort_keys=True)
    assert "external_spool" in encoded
    # Every CPU rung is enumerated honestly (13 rungs, descending).
    assert [t["tile"] for t in report["cpu"]["trials"]] == list(
        zz.CPU_STREAMING_TILE_LADDER
    )


def test_dry_plan_gpu_ladder_is_separate_and_never_below_32():
    report = zdry.build_dry_plan(
        _m33_spec(gpu_ladder=zz.GPU_STREAMING_TILE_LADDER)
    )
    assert report["gpu"]["ladder"] == list(zz.GPU_STREAMING_TILE_LADDER)
    assert report["gpu"]["minimum_tile_size"] == 32
    assert report["gpu"]["selected_tile"] is None
    assert report["cpu"]["selected_tile"] == 20  # CPU choice independent of GPU
    with pytest.raises(ValueError):
        zdry.build_dry_plan(_m33_spec(gpu_ladder=(32, 16)))


# ---------------------------------------------------------------------------
# M1 — real product propagation (bundle + worker initializer seam)
# ---------------------------------------------------------------------------

def test_product_bundle_selects_below_32_and_publishes_executor_tile(monkeypatch):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize(monkeypatch)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    # Values are TOTAL peaks (they include the 340 MiB process baseline), so the
    # ACTIVE facts' ADDED bound is what the scheduler compares to B_plan.
    table = {
        128: 60 * MIB, 112: 60 * MIB, 96: 60 * MIB, 80: 60 * MIB,
        64: 60 * MIB, 48: 60 * MIB, 32: 60 * MIB, 24: 60 * MIB,
        20: 60 * MIB, 16: 8 * MIB, 12: 8 * MIB, 10: 8 * MIB,
        8: 8 * MIB,
    }
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(
            table[int(tile) if tile is not None else zz.STREAM_TILE_SIZE]
        ),
    )
    raw = 64 * MIB
    budget = (4 * raw) // 5
    bundle = zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=raw, budget_bytes=budget,
        cpu_count=8, cache_root=None,
    )
    assert bundle.gpu_used is False
    assert bundle.chosen_mode == "stream"
    assert bundle.cpu_streaming_tile == 16
    assert bundle.cpu_streaming["selected_tile_size"] == 16
    assert bundle.cpu_streaming["ladder"] == list(zz.CPU_STREAMING_TILE_LADDER)
    assert bundle.cpu_streaming["minimum_tile_size"] == 8
    # Every executed stream cell task carries the SELECTED below-32 rung.
    stream_tasks = [t for t in bundle.cell_tasks if t[4] == "stream"]
    assert stream_tasks
    assert all(int(t[5]) == 8 * MIB for t in stream_tasks)
    # The worker initializer seam publishes the SAME tile to the executor.
    zz._init_stack_worker([], None, None, None, (), None, tile_size=bundle.cpu_streaming_tile)
    assert zz._STACK_TILE_SIZE == 16


def test_gui_worker_invocation_has_no_tile_override():
    """The GUI→worker→engine call carries no CLI/manual tile override."""
    import inspect

    from zemosaic import zemosaic_worker as zw

    src = inspect.getsource(zw)
    start = src.index("run_zegrid_mode(\n")
    end = src.index("finally:", start)
    call = src[start:end]
    assert "cpu_tile" not in call
    assert "tile_size" not in call
    # AUTO semantics: storage profile is not a user selector in the worker call.
    assert "storage_profile" not in call
    assert "zegrid_scheduler_mode" in call


def test_active_planner_selects_below_32_and_maps_it(tmp_path, monkeypatch):
    """The default ACTIVE product planner selects the below-32 rung itself."""
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize(monkeypatch)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    # Values are TOTAL peaks (they include the 340 MiB process baseline), so the
    # ACTIVE facts' ADDED bound is what the scheduler compares to B_plan.
    table = {
        128: 700 * MIB, 112: 700 * MIB, 96: 700 * MIB, 80: 700 * MIB,
        64: 700 * MIB, 48: 700 * MIB, 32: 700 * MIB, 24: 700 * MIB,
        20: 700 * MIB, 16: 360 * MIB, 12: 360 * MIB, 10: 360 * MIB,
        8: 360 * MIB,
    }
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(
            table[int(tile) if tile is not None else zz.STREAM_TILE_SIZE]
        ),
    )
    raw = 500 * MIB
    budget = (4 * raw) // 5
    bundle = zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=raw, budget_bytes=budget,
        cpu_count=8, cache_root=None,
    )
    decision = zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=8,
        cache_root=tmp_path,
        output_dir=tmp_path,
        frame_ids=[d.frame_id.logical_path for d in descs],
    )
    assert decision.mapping["cell_task_mode"] == "stream"
    assert int(decision.mapping["cpu_streaming_tile"]) == 16
    assert int(decision.plan_set.chosen.tile_px) == 16
    assert decision.plan_set.mode == zs.MODE_ACTIVE
