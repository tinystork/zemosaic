"""ZM-ZEGRID-R55-E8 — coherent RAM/VRAM adaptation (header-only).

These tests pin the end-to-end contract that failed BEFORE any pixel work:

* the Windows halo128 CPU16 automatic gauge=14 plan is refused by the global
  15 % margin; the product must derive gauge<=12 under the SAME B_plan/margin;
* the Linux halo16 LEGACY/ACTIVE GPU plan must keep the feasible tile128
  instead of discarding a feasible GPU because the VRAM-max tile missed host
  RAM;
* CPU ACTIVE must use each streaming tile's OWN exact host bound (not the
  static t32 maximum);
* worker ladders are bounded, descending and truthfully refuse when even one
  worker cannot fit.

No pixel stack, no CuPy and no full mosaic is built here. Numbers are
model-only; they are not presented as runtime observations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import gpu_readiness as zgr
from zemosaic.core.zegrid import resource_envelope as renv

MIB = 2 ** 20
GIB = 2 ** 30
LIGHTS = Path("/home/tristan/M106/lights")

WINDOWS_B_PLAN = 8992230604
WINDOWS_RAW = (WINDOWS_B_PLAN * 5 + 3) // 4
LINUX_RAW = 2999496704
LINUX_B_PLAN = 2399597363
LINUX_VRAM_BUDGET = 1025310720
LINUX_VRAM_FREE = 2050621440
LINUX_VRAM_TOTAL = 2091253760


def _readiness(free: int, total: int):
    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="model-only",
        device_count=1,
        device_index=0,
        device_name="simulated-rtx3070",
        vram_total_bytes=int(total),
        vram_free_bytes=int(free),
    )


def _probe(free: int, total: int):
    return {
        "available": True,
        "device": "simulated-rtx3070",
        "vram_total_bytes": int(total),
        "vram_free_bytes": int(free),
    }


@pytest.fixture(scope="module")
def m106():
    if not LIGHTS.is_dir():
        pytest.skip("M106 lights directory absent")
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    assert len(frames) == 66
    assert (canvas.width, canvas.height) == (2403, 3278)
    return frames, canvas


@pytest.fixture(autouse=True)
def _large_disk(monkeypatch):
    """Header-only planning needs a deterministic disk observation."""
    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=4 * 2 ** 40, used=0, free=4 * 2 ** 40),
    )


def _bundle(m106, *, raw, budget, halo, cpu, vram, gpu=True):
    frames, canvas = m106
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: cpu)
        return zz._build_execution_bundle(
            canvas,
            frames,
            None,
            raw_available_bytes=int(raw),
            budget_bytes=int(budget),
            halo_px=int(halo),
            cpu_count=int(cpu),
            gpu_used=bool(gpu),
            vram_budget_bytes=(None if vram is None else int(vram)),
            cache_root="/tmp",
        )


def _active_gpu_decide(m106, bundle, *, halo, cpu, free, total):
    frames, canvas = m106
    frame_ids = tuple(f.frame_id.logical_path for f in frames)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: cpu)
        return zz._r55_active_gpu_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=int(halo),
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=frame_ids,
            gpu_probe=_probe(free, total),
            readiness=_readiness(free, total),
        )


def _active_cpu_decide(m106, bundle, *, halo, cpu):
    frames, canvas = m106
    frame_ids = tuple(f.frame_id.logical_path for f in frames)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: cpu)
        return zz._r55_active_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=int(halo),
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=frame_ids,
        )


# ---------------------------------------------------------------------------
# Gate A — Windows CPU16, halo128, automatic workers, ACTIVE GPU
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "free_gib,expected_tile,expected_prefetch",
    [
        # R55-E10: the ACTIVE GPU tile is the VRAM-derived maximum PRUNED to the
        # selection headroom (85 % of the fresh 70 % budget), never max128 and
        # never the razor-edge top rung.
        (2, 192, 2),
        (4, 320, 1),
        (7, 448, 0),
    ],
)
def test_gate_a_windows_automatic_gauge_and_gpu_plan(
    m106, free_gib, expected_tile, expected_prefetch
):
    free = free_gib * GIB
    total = 8 * GIB
    vram = zgpu.vram_budget_bytes({"available": True, "vram_free_bytes": free})
    bundle = _bundle(
        m106,
        raw=WINDOWS_RAW,
        budget=WINDOWS_B_PLAN,
        halo=128,
        cpu=16,
        vram=vram,
    )
    # Automatic (not fixed-at-4) gauge/cache selection, made coherent with the
    # same B_plan + 15 % margin that will validate the plan.
    assert bundle.gauge_workers <= 12
    assert bundle.gauge_workers >= 1
    assert bundle.worker_budget_bytes == (WINDOWS_B_PLAN * 100) // 115

    decision = _active_gpu_decide(
        m106, bundle, halo=128, cpu=16, free=free, total=total
    )
    assert decision.gauge_workers == bundle.gauge_workers
    assert decision.cache_workers == bundle.cache_build_workers
    assert decision.facts["layout"] == (2, 2)
    assert decision.facts["halo_px"] == 128
    assert decision.mapping["backend"] == "gpu"
    tile = int(decision.mapping["gpu_streaming_tile"])
    assert tile == expected_tile
    assert tile > 128
    assert int(decision.gpu_tile_max) >= tile
    assert tile in decision.gpu_tile_ladder
    assert any(int(t) > 128 for t in decision.gpu_tile_ladder)
    assert decision.mapping["units_in_flight"] == 1
    assert decision.mapping["prefetch_units"] == expected_prefetch
    assert decision.plan_set.aggregate_host_peak_bytes <= WINDOWS_B_PLAN
    assert (
        decision.plan_set.aggregate_device_peak_bytes
        <= decision.gpu_facts.vram_budget_bytes
    )
    # The exact selected tile bound reaches the candidate AND the envelope.
    chosen = decision.plan_set.chosen
    added = dict(decision.gpu_host_bound_by_tile)[tile]
    envelope = next(
        env for env in decision.plan_set.phases if env.backend == "gpu"
    )
    assert chosen.host_peak_bytes == (
        bundle.shared_bytes + added + decision.facts["per_unit_cache"]
        * chosen.prefetch_units
    )
    assert envelope.geometry.tile_px == tile
    assert envelope.estimate.peak_bytes == chosen.host_peak_bytes
    # All 66 frame identities survive unchanged (identity gate).
    assert len(decision.facts["frame_ids"]) == 66
    assert decision.facts["cell_ids"] == tuple(
        sorted(cell.cell_id for _r, _c, cell, _p, m in bundle.cell_ctxs if m.patch_ids)
    )


# ---------------------------------------------------------------------------
# Gate B — Linux recorded budgets, CPU8, halo16, LEGACY + ACTIVE GPU
# ---------------------------------------------------------------------------

def test_gate_b_linux_legacy_and_active_keep_the_feasible_gpu_tile(m106):
    bundle = _bundle(
        m106,
        raw=LINUX_RAW,
        budget=LINUX_B_PLAN,
        halo=16,
        cpu=8,
        vram=LINUX_VRAM_BUDGET,
    )
    # LEGACY: the VRAM-max tile misses host RAM, but tile128 is jointly
    # feasible and must be retained instead of a pre-execution CPU fallback.
    assert bundle.gpu_used is True
    assert bundle.gpu_tile_size == 128
    assert bundle.envelope.aggregate_bytes <= LINUX_B_PLAN
    assert bundle.gpu_device_workspace_bytes is not None
    assert bundle.gpu_device_workspace_bytes <= LINUX_VRAM_BUDGET
    trials = bundle.gpu_tile_trials or []
    assert any(
        int(t["tile"]) > 128 and t["fits"] is False for t in trials
    ), trials

    decision = _active_gpu_decide(
        m106,
        bundle,
        halo=16,
        cpu=8,
        free=LINUX_VRAM_FREE,
        total=LINUX_VRAM_TOTAL,
    )
    assert decision.mapping["backend"] == "gpu"
    assert decision.mapping["gpu_streaming_tile"] == 128
    assert decision.plan_set.aggregate_host_peak_bytes <= LINUX_B_PLAN
    assert (
        decision.plan_set.aggregate_device_peak_bytes
        <= LINUX_VRAM_BUDGET
    )
    # The mapped tile reaches the existing GPU runner knobs (wiring proof).
    applied = zz._r55_active_gpu_apply(
        bundle,
        mapping=decision.mapping,
        decision=decision,
        stage="preflight",
    )
    assert applied.gpu_used is True
    assert applied.gpu_tile_size == 128
    assert applied.gpu_streaming_tile == 128
    assert applied.cells_in_flight == 1
    execution = zz._r55_active_execution(applied, decision.mapping)
    assert execution["backend"] == "gpu"
    assert execution["gpu_streaming_tile"] == 128
    assert execution["single_owner"] is True


# ---------------------------------------------------------------------------
# Gate C — CPU ACTIVE exact per-tile bounds (t128/t96 fit, t32 does not)
# ---------------------------------------------------------------------------

def test_gate_c_cpu_active_uses_the_selected_tiles_own_bound(m106):
    bundle = _bundle(
        m106,
        raw=LINUX_RAW,
        budget=LINUX_B_PLAN,
        halo=16,
        cpu=8,
        vram=LINUX_VRAM_BUDGET,
    )
    decision = _active_cpu_decide(m106, bundle, halo=16, cpu=8)
    chosen = decision.plan_set.chosen
    assert chosen.backend == "cpu"
    assert chosen.tile_px in (96, 128)
    bound = decision.facts["stream_added_by_tile"][int(chosen.tile_px)]
    assert chosen.host_peak_bytes == bundle.shared_bytes + bound
    assert (
        decision.facts["stream_worst_by_tile"][32] > LINUX_B_PLAN
    ), "the static t32 maximum must be the value the old path wrongly used"
    envelope = next(
        env
        for env in decision.plan_set.phases
        if env.phase == renv.PHASE_CELL_STACK_STREAM
    )
    assert envelope.geometry.tile_px == chosen.tile_px
    assert envelope.estimate.peak_bytes == chosen.host_peak_bytes
    assert decision.plan_set.plan_hash.startswith("sha256:")
    # Determinism: the same facts produce the same plan and hash.
    again = _active_cpu_decide(m106, bundle, halo=16, cpu=8)
    assert again.plan_set.plan_hash == decision.plan_set.plan_hash
    assert again.mapping == decision.mapping


# ---------------------------------------------------------------------------
# Gate D — worker adaptation adversarials
# ---------------------------------------------------------------------------

def test_margin_inverse_is_integer_exact():
    for budget in (0, 1, 100, 1_000_003, WINDOWS_B_PLAN, LINUX_B_PLAN):
        ceiling = zz._r55_margin_raw_ceiling(budget)
        assert renv.ceil_margin_div(ceiling, 15, 100) <= budget
        assert renv.ceil_margin_div(ceiling + 1, 15, 100) > budget


def test_worker_ladders_are_bounded_descending_and_never_grow():
    for start in (1, 2, 3, 8, 14):
        ladder = zz._r55_worker_ladder(start)
        assert ladder[0] == start and ladder[-1] == 1
        assert list(ladder) == sorted(set(ladder), reverse=True)


def test_worker_ceiling_respects_cpu_and_margin(monkeypatch):
    fp = 640089064
    for cpu in (2, 8, 16, 32):
        monkeypatch.setattr(zz.os, "cpu_count", lambda cpu=cpu: cpu)
        budget = zz._r55_margin_raw_ceiling(WINDOWS_B_PLAN)
        workers = zz._r55_coherent_worker_count(None, budget, fp)
        # The existing R16 seam keeps a 2-worker floor on a healthy multi-core
        # box (cpu=2 therefore uses both cores); memory may still force 1.
        upper = max(1, cpu) if cpu < 4 else max(1, cpu - 2)
        assert 1 <= workers <= upper
        assert workers * fp <= budget or workers == 1


@dataclass
class _Bundle:
    gauge_workers: int
    cache_build_workers: int
    layout: dict
    envelope: object
    shared_bytes: int
    cache_worker_footprint_bytes: int
    cell_specs: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    cpu_fallback_info: dict = field(default_factory=dict)
    cpu_streaming: dict = field(default_factory=dict)
    cpu_streaming_tile: int = 0
    gpu_tile_size: int = 0
    gpu_streaming_tile: int = 0
    gpu_streaming_host_peak_bytes: int = 0
    gpu_used: bool = False
    gpu_pre_execution_fallback: object = None
    cell_tasks: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    chosen_mode: str = "stream"
    cells_in_flight: int = 1
    cache_workers_per_cell: int = 1
    per_cell_footprint: int = 0
    gpu_device_workspace_bytes: object = None
    gpu_tile_trials: list = field(default_factory=list)


def _synthetic_bundle(*, raw, budget, gauge, cache):
    canvas = SimpleNamespace(width=1024, height=1024, canvas_id="r55e8-synth")
    core = SimpleNamespace(x0=0, y0=0, x1=1024, y1=1024)
    cell = SimpleNamespace(cell_id="r0000c0000", core=core)
    patch = SimpleNamespace(
        patch=SimpleNamespace(width=300, height=300),
        patch_shape_hw=(300, 300),
    )
    mem = SimpleNamespace(patch_ids=("f0", "f1"))
    cell_ctxs = [(0, 0, cell, patch, mem)]
    envelope = SimpleNamespace(
        refinement_rank=1,
        pinned=False,
        raw_available_bytes=int(raw),
        factor_num=4,
        factor_den=5,
        budget_bytes=int(budget),
        layout=(1, 1),
    )
    bundle = _Bundle(
        gauge_workers=int(gauge),
        cache_build_workers=int(cache),
        layout={"nx": 1, "ny": 1},
        envelope=envelope,
        shared_bytes=int(zz.rcontract.PROCESS_BASELINE_BYTES),
        cache_worker_footprint_bytes=int(4 * MIB),
    )
    return canvas, cell_ctxs, bundle


def _decide_synth(monkeypatch, *, gauge_fp, cache_fp, gauge, cache, budget, raw):
    monkeypatch.setattr(zz.zpar, "gauge_footprint_bytes", lambda _px: int(gauge_fp))
    monkeypatch.setattr(zz.zpar, "cache_footprint_bytes", lambda _px: int(cache_fp))
    monkeypatch.setattr(zz.os, "cpu_count", lambda: 8)
    canvas, cell_ctxs, bundle = _synthetic_bundle(
        raw=raw, budget=budget, gauge=gauge, cache=cache
    )
    return zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=cell_ctxs,
        canvas=canvas,
        halo_px=16,
        cache_root="/tmp",
        output_dir="/tmp",
        frame_ids=("f0", "f1"),
    )


def test_gauge_and_cache_phases_are_independently_binding(monkeypatch):
    budget = 1500 * MIB
    raw = (budget * 5 + 3) // 4
    # Gauge is the large non-overlapping phase: it must shrink, cache need not.
    gauge_bound = _decide_synth(
        monkeypatch,
        gauge_fp=400 * MIB,
        cache_fp=1 * MIB,
        gauge=8,
        cache=8,
        budget=budget,
        raw=raw,
    )
    assert gauge_bound.gauge_workers < 8
    assert gauge_bound.cache_workers == 8
    # Cache is the large phase: cache shrinks, gauge keeps its useful maximum.
    monkeypatch.undo()
    cache_bound = _decide_synth(
        monkeypatch,
        gauge_fp=1 * MIB,
        cache_fp=400 * MIB,
        gauge=8,
        cache=8,
        budget=budget,
        raw=raw,
    )
    assert cache_bound.cache_workers < 8
    assert cache_bound.gauge_workers == 8


def test_one_worker_still_too_large_refuses_truthfully(monkeypatch):
    budget = 100 * MIB
    raw = (budget * 5 + 3) // 4
    with pytest.raises(zz.zsched.PlanRefused) as exc:
        _decide_synth(
            monkeypatch,
            gauge_fp=400 * MIB,
            cache_fp=1 * MIB,
            gauge=4,
            cache=4,
            budget=budget,
            raw=raw,
        )
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM
    assert "full bounded ladders" in exc.value.detail


# ---------------------------------------------------------------------------
# Gate E — GPU OFF performs no CUDA/VRAM probe
# ---------------------------------------------------------------------------

def test_gpu_off_performs_no_cuda_or_vram_probe(monkeypatch):
    calls = []

    def _explode(*_a, **_kw):
        calls.append("probe")
        raise AssertionError("GPU OFF must not probe CUDA/VRAM")

    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", _explode)
    monkeypatch.setattr(zz.zgpu, "probe_gpu_readiness_safe", _explode, raising=False)
    requested, source, probe, budget = zz._r55_resolve_gpu_probe(False, None)
    assert requested is False
    assert budget is None
    assert calls == []
    assert probe["available"] is False
    assert probe["vram_total_bytes"] is None
    assert probe["vram_free_bytes"] is None
    assert probe["reason"] == "gpu_not_requested"


def test_explicit_gpu_off_beats_realistic_legacy_aliases_without_probing(monkeypatch):
    """Explicit/product OFF must outrank legacy aliases that say True.

    This is the realistic zconfig shape (canonical + legacy keys all True) and
    the R55-E4 precedence contract: ``use_gpu=False`` is authoritative, so the
    E8 resolver returns immediately and performs NO backend/VRAM/readiness probe.
    """
    class _LegacyConfig:
        use_gpu_phase5 = True
        use_gpu_grid = True
        stack_use_gpu = True
        use_gpu_stack = True

    calls = []

    def _record(name):
        def _fn(*_a, **_kw):
            calls.append(name)
            return {"available": True, "vram_free_bytes": 4 * GIB,
                    "vram_total_bytes": 8 * GIB, "device": "must-not-run"}
        return _fn

    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", _record("backend"))
    monkeypatch.setattr(
        zz.zgpu, "probe_gpu_readiness_safe", _record("readiness"), raising=False
    )
    requested, source, probe, budget = zz._r55_resolve_gpu_probe(
        False, _LegacyConfig()
    )
    assert requested is False
    assert source == "argument"
    assert budget is None
    assert calls == []
    assert probe["available"] is False
    assert probe["reason"] == "gpu_not_requested"
    # Sanity: the same aliases WOULD request GPU when no explicit argument wins.
    assert zz.resolve_gpu_preference(None, _LegacyConfig())[0] is True
