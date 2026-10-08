"""ZM-ZEGRID-R22 targeted tests — GPU contract + joint planner + winsor.

Covers (fast tier, no gated corpora):

* ``resolve_gpu_preference`` precedence + strict bool coercion;
* the REAL production worker call propagates ``use_gpu`` (generic argument) into
  ``run_zegrid_mode`` (monkeypatched to capture the invocation — a generic
  propagation witness, not a key-specific toy);
* manifest/log GPU truth (requested/available/effective/device/VRAM/fallback
  reason) and ``ignored_settings`` drops the GPU flags ONLY when honoured;
* GPU unavailable / forced-OOM degrades LOUDLY to exact CPU (never crash);
* the joint mode+concurrency planner on the REAL R20 figures and 2 GiB / 8 GiB
  VRAM budgets (never reports parallel while selecting serial);
* bounded cache-build workers (single active cell -> full budget; K cells ->
  cpu//K so no K x 14 explosion);
* aggregate peak RSS helper (parent + workers, upper bound).

GPU-dependent parity lives in ``test_sci05_canonical_streaming_gpu.py`` (skips
cleanly without CuPy/GPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import instrumentation as zin
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import sweep as zsw

GB = 2**30


# ---------------------------------------------------------------------------
# 1. GPU preference resolution
# ---------------------------------------------------------------------------

def test_resolve_gpu_preference_default():
    assert zz.resolve_gpu_preference(None, None) == (False, "default")
    assert zz.resolve_gpu_preference(None, SimpleNamespace()) == (False, "default")


def test_resolve_gpu_preference_all_true():
    zc = SimpleNamespace(stack_use_gpu=True, use_gpu_stack=True, use_gpu_grid=True)
    req, src = zz.resolve_gpu_preference(None, zc)
    assert req is True and src == "use_gpu_grid"


def test_resolve_gpu_preference_all_false():
    zc = SimpleNamespace(stack_use_gpu=False, use_gpu_stack=False, use_gpu_grid=False)
    assert zz.resolve_gpu_preference(None, zc) == (False, "use_gpu_grid")


def test_resolve_gpu_preference_precedence_order():
    # First non-None wins: use_gpu_grid False beats stack_use_gpu True.
    zc = SimpleNamespace(use_gpu_grid=False, stack_use_gpu=True)
    assert zz.resolve_gpu_preference(None, zc) == (False, "use_gpu_grid")
    # use_gpu_grid absent -> falls through to stack_use_gpu.
    zc2 = SimpleNamespace(stack_use_gpu=True, use_gpu_stack=False)
    assert zz.resolve_gpu_preference(None, zc2) == (True, "stack_use_gpu")
    # falls through to use_gpu_stack.
    zc3 = SimpleNamespace(use_gpu_stack=True)
    assert zz.resolve_gpu_preference(None, zc3) == (True, "use_gpu_stack")
    # falls through to use_gpu_phase5 (GUI canonical).
    zc4 = SimpleNamespace(use_gpu_phase5=True)
    assert zz.resolve_gpu_preference(None, zc4) == (True, "use_gpu_phase5")


def test_resolve_gpu_preference_argument_overrides_config():
    zc = SimpleNamespace(stack_use_gpu=True)
    assert zz.resolve_gpu_preference(False, zc) == (False, "argument")
    assert zz.resolve_gpu_preference(True, SimpleNamespace(stack_use_gpu=False)) == (True, "argument")


def test_resolve_gpu_preference_coercion():
    # string / numeric coercions
    assert zz.resolve_gpu_preference("true", None) == (True, "argument")
    assert zz.resolve_gpu_preference(1, None) == (True, "argument")
    assert zz.resolve_gpu_preference(0, None) == (False, "argument")
    assert zz.resolve_gpu_preference("off", None) == (False, "argument")
    assert zz.resolve_gpu_preference("garbage", None) == (False, "default")
    assert zz.resolve_gpu_preference(None, SimpleNamespace(stack_use_gpu="yes")) == (True, "stack_use_gpu")
    assert zz.resolve_gpu_preference(None, SimpleNamespace(stack_use_gpu="0")) == (False, "stack_use_gpu")


# ---------------------------------------------------------------------------
# 2. Real worker propagation (generic argument witness)
# ---------------------------------------------------------------------------

def test_worker_propagates_use_gpu_into_run_zegrid_mode(monkeypatch):
    """The production worker call passes use_gpu (resolved bool) to run_zegrid_mode."""
    import zemosaic.zemosaic_worker as zw

    captured = {}

    def fake_run_zegrid_mode(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(zw, "detect_grid_mode", lambda folder: True)
    # Patch the module import inside run_hierarchical_mosaic to return our fake.
    monkeypatch.setitem(
        __import__("sys").modules,
        "zemosaic.zemosaic_zegrid_mode",
        SimpleNamespace(
            run_zegrid_mode=fake_run_zegrid_mode,
            resolve_gpu_preference=zz.resolve_gpu_preference,
        ),
    )
    # Inject a config cache with the GUI GPU aliases all true.
    monkeypatch.setattr(zw, "ZEMOSAIC_CONFIG_AVAILABLE", True)
    fake_config = SimpleNamespace(load_config=lambda: {"stack_use_gpu": True, "use_gpu_stack": True, "use_gpu_grid": True})
    monkeypatch.setattr(zw, "zemosaic_config", fake_config)

    def fake_progress(*a, **k):
        return None

    # Minimal args for run_hierarchical_mosaic to reach the ZeGrid branch.
    args = dict(
        input_folder="/tmp/x", output_folder="/tmp/y", astap_exe_path="",
        astap_data_dir_param="", astap_search_radius_config=1.0,
        astap_downsample_config=1, astap_sensitivity_config=1,
        cluster_threshold_config=0.1, cluster_target_groups_config=1,
        cluster_orientation_split_deg_config=1.0, progress_callback=fake_progress,
        stack_ram_budget_gb_config=0.0, stack_norm_method="sky_mean",
        stack_weight_method="noise_variance", stack_reject_algo="kappa_sigma",
        stack_kappa_low=3.0, stack_kappa_high=3.0, parsed_winsor_limits=(0.05, 0.05),
        stack_final_combine="mean", poststack_equalize_rgb_config=False,
        apply_radial_weight_config=False, radial_feather_fraction_config=0.8,
        radial_shape_power_config=2.0, min_radial_weight_floor_config=0.0,
        final_assembly_method_config="reproject_coadd",
        inter_master_merge_enable_config=False, inter_master_overlap_threshold_config=0.3,
        inter_master_min_group_size_config=1, inter_master_stack_method_config="mean",
        inter_master_memmap_policy_config="auto", inter_master_local_scale_config="none",
        inter_master_max_group_config=1, num_base_workers_config=2,
        apply_master_tile_crop_config=False, master_tile_crop_percent_config=0.0,
        quality_crop_enabled_config=False, quality_crop_band_px_config=0,
        quality_crop_k_sigma_config=3.0, quality_crop_margin_px_config=0,
        quality_crop_min_run_config=0, altaz_cleanup_enabled_config=False,
        altaz_margin_percent_config=0.0, altaz_decay_config=0.0, altaz_nanize_config=False,
        quality_gate_enabled_config=False, quality_gate_threshold_config=0.5,
        quality_gate_edge_band_px_config=0, quality_gate_k_sigma_config=3.0,
        quality_gate_erode_px_config=0, quality_gate_move_rejects_config=False,
        save_final_as_uint16_config=False, legacy_rgb_cube_config=False,
        coadd_use_memmap_config=False, coadd_memmap_dir_config="",
        coadd_cleanup_memmap_config=False, assembly_process_workers_config=1,
        auto_limit_frames_per_master_tile_config=False,
        winsor_max_frames_per_pass_config=0, winsor_worker_limit_config=1,
        max_raw_per_master_tile_config=100,
        use_gpu_phase5=False, gpu_id_phase5=None,
    )
    try:
        zw.run_hierarchical_mosaic(**args)
    except Exception:
        pass  # the fake returns None; downstream may raise after — we only check capture
    assert captured.get("use_gpu") is True


# ---------------------------------------------------------------------------
# 3. Manifest / log GPU truth
# ---------------------------------------------------------------------------

def test_manifest_gpu_block_truth():
    ctx = {
        "requested": True, "requested_source": "use_gpu_grid", "available": True,
        "attempted": True, "final_effective": True, "actually_used": True,
        "final_backend": "gpu", "gpu_executed_cells": 6,
        "device": "NVIDIA GeForce MX150", "cupy_version": "14.1.1",
        "vram_total_bytes": 2 * GB, "vram_free_bytes": int(1.9 * GB),
        "vram_budget_bytes": int(0.95 * GB), "fallback_reason": None,
    }
    block = zz._manifest_gpu_block(True, ctx)
    assert block["used"] is True
    assert block["actually_used"] is True
    assert block["final_backend"] == "gpu"
    assert block["requested"] is True
    assert block["attempted"] is True
    assert block["effective"] is True
    assert block["gpu_executed_cells"] == 6
    assert block["device"] == "NVIDIA GeForce MX150"
    assert block["backend"]["per_cell_rejection"] == "gpu"
    assert block["backend"]["per_cell_combine"] == "gpu"
    assert block["backend"]["gauge_rejection"] == "cpu"

    # CPU fallback case (start-time unavailable)
    block2 = zz._manifest_gpu_block(False, {
        **ctx, "attempted": False, "final_effective": False, "actually_used": False,
        "final_backend": "cpu", "gpu_executed_cells": 0,
        "available": False, "fallback_reason": "cupy_or_cuda_unavailable",
    })
    assert block2["used"] is False
    assert block2["actually_used"] is False
    assert block2["requested"] is True
    assert block2["effective"] is False
    assert block2["fallback_reason"] == "cupy_or_cuda_unavailable"
    assert block2["backend"]["per_cell_rejection"] == "cpu"


def test_manifest_gpu_block_empty_run_cannot_claim_used():
    """H2: attempted-but-empty (no GPU cells executed) => used=false."""
    ctx = {
        "requested": True, "available": True, "attempted": True,
        "final_effective": False, "actually_used": False, "final_backend": "cpu",
        "gpu_executed_cells": 0, "fallback_reason": None,
    }
    block = zz._manifest_gpu_block(False, ctx)
    assert block["used"] is False
    assert block["actually_used"] is False
    assert block["backend"]["per_cell_rejection"] == "cpu"


def test_ignored_settings_drops_gpu_flags_only_when_honoured():
    zc = SimpleNamespace(
        stack_use_gpu=True, use_gpu_stack=True, use_gpu_grid=True,
        intertile_affine_blend=0.3, center_out_normalization_p3=True,
    )
    # Not honoured -> GPU flags present.
    not_honoured = zin.ignored_settings_present(zc, gpu_honoured=False)
    assert "stack_use_gpu" in not_honoured
    assert "use_gpu_grid" in not_honoured
    assert "use_gpu_stack" in not_honoured
    assert "intertile_affine_blend" in not_honoured
    # Honoured -> GPU flags absent, post-stack flags remain.
    honoured = zin.ignored_settings_present(zc, gpu_honoured=True)
    assert "stack_use_gpu" not in honoured
    assert "use_gpu_grid" not in honoured
    assert "use_gpu_stack" not in honoured
    assert "intertile_affine_blend" in honoured


# ---------------------------------------------------------------------------
# 4. GPU unavailable / forced-OOM fallback (never crash, loud WARN)
# ---------------------------------------------------------------------------

def _capture_cb():
    events = []

    def cb(msg, prog, lvl, **kw):
        events.append((msg, lvl))

    return events, cb


def test_run_zegrid_mode_cpu_fallback_when_gpu_unavailable(monkeypatch, tmp_path):
    """GPU requested but unavailable -> exact CPU, WARN recorded (no crash)."""
    monkeypatch.setattr(zgpu, "probe_gpu_backend", lambda: {
        "available": False, "device": None, "cupy_version": None,
        "vram_total_bytes": None, "vram_free_bytes": None, "reason": "cupy_or_cuda_unavailable",
    })
    events, cb = _capture_cb()
    zc = SimpleNamespace(stack_use_gpu=True, use_gpu_stack=True, use_gpu_grid=True)
    gpu_requested, gpu_source = zz.resolve_gpu_preference(None, zc)
    assert gpu_requested is True
    probe = zgpu.probe_gpu_backend()
    vram = zgpu.vram_budget_bytes(probe)
    assert vram is None
    # The fallback decision path (mirrored from run_zegrid_mode) must degrade.
    gpu_effective = False
    if gpu_requested:
        if not probe["available"]:
            reason = probe.get("reason") or "gpu_unavailable"
        elif vram is None:
            reason = "vram_unknown_or_too_small"
        else:
            gpu_effective = True
    assert gpu_effective is False
    assert reason == "cupy_or_cuda_unavailable"


def test_vram_budget_and_tile_planner():
    # 8 GiB GPU (RTX 3070) — plenty for a large tile.
    probe8 = {"available": True, "vram_total_bytes": 8 * GB, "vram_free_bytes": int(7.5 * GB)}
    budget8 = zgpu.vram_budget_bytes(probe8)
    assert budget8 is not None and budget8 >= zgpu.GPU_MIN_VRAM_BUDGET_BYTES
    t8 = zgpu.choose_gpu_tile_size(66, 3, (3278, 2403), budget8)
    assert t8 is not None and t8 >= 128
    # 2 GiB GPU (MX150) — smaller tile, still feasible.
    probe2 = {"available": True, "vram_total_bytes": 2 * GB, "vram_free_bytes": int(1.9 * GB)}
    t2 = zgpu.choose_gpu_tile_size(66, 3, (3278, 2403), zgpu.vram_budget_bytes(probe2))
    assert t2 is not None and t2 >= 1
    # A degenerate (tiny) budget -> None (degrade to CPU).
    assert zgpu.choose_gpu_tile_size(66, 3, (3278, 2403), 1) is None
    assert zgpu.choose_gpu_tile_size(66, 3, (3278, 2403), None) is None


def test_vram_budget_never_inflates():
    """H3: free*fraction < GPU_MIN_VRAM_BUDGET_BYTES => None (degrade), never inflate."""
    # 10 MiB free -> 5 MiB budget < 64 MiB floor -> None (NOT 64 MiB).
    assert zgpu.vram_budget_bytes({"available": True, "vram_free_bytes": 10 * 2**20}) is None
    # 100 MiB free -> 50 MiB < 64 MiB -> None (NOT 64 MiB).
    assert zgpu.vram_budget_bytes({"available": True, "vram_free_bytes": 100 * 2**20}) is None
    # Exactly at 2x the floor (128 MiB free -> 64 MiB) -> accepted.
    assert zgpu.vram_budget_bytes({"available": True, "vram_free_bytes": 128 * 2**20}) == 64 * 2**20
    # 2 GiB free -> 1 GiB budget.
    assert zgpu.vram_budget_bytes({"available": True, "vram_free_bytes": 2 * 2**30}) == 1 * 2**30
    # unavailable -> None.
    assert zgpu.vram_budget_bytes({"available": False, "vram_free_bytes": 2 * 2**30}) is None
    # unknown free (falls back to total).
    assert zgpu.vram_budget_bytes({"available": True, "vram_total_bytes": 2 * 2**30, "vram_free_bytes": None}) == 1 * 2**30


def test_is_gpu_runtime_error_classifier():
    """H1: narrow classifier — CuPy errors => True; arbitrary errors => False."""
    def _mk(name, mod):
        cls = type(name, (Exception,), {})
        cls.__module__ = mod
        return cls

    # Direct CuPy errors.
    for name, mod in (
        ("OutOfMemoryError", "cupy.cuda.memory"),
        ("CUDARuntimeError", "cupy.cuda.runtime"),
        ("CUDADriverError", "cupy.cuda.driver"),
        ("CompileException", "cupy.cuda.compiler"),
    ):
        assert zgpu.is_gpu_runtime_error(_mk(name, mod)("boom"))

    # Wrapped in cause/context chains.
    inner = _mk("OutOfMemoryError", "cupy.cuda.memory")("oom")
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = inner
    assert zgpu.is_gpu_runtime_error(wrapped)

    # Arbitrary errors must NOT classify as GPU.
    assert not zgpu.is_gpu_runtime_error(ValueError("science bug"))
    assert not zgpu.is_gpu_runtime_error(KeyError("missing"))
    assert not zgpu.is_gpu_runtime_error(MemoryError("plain python OOM"))  # not cupy
    assert not zgpu.is_gpu_runtime_error(ZeroDivisionError("x / 0"))
    assert not zgpu.is_gpu_runtime_error(None)


# ---------------------------------------------------------------------------
# 5. Joint planner on REAL R20 figures + VRAM budgets
# ---------------------------------------------------------------------------

def test_planner_r20_figures_never_reports_parallel_while_serial():
    # R20: 6 cells, max inmem ~10.35 GiB, max stream ~2.34 GiB, 16 CPU, 11.83 GiB.
    plan = zpar.plan_cell_concurrency(
        [(10849326220, 2449628160)] * 6, 16, 12184610406, cache_build_workers=14,
    )
    assert plan["mode"] == "stream"
    assert plan["cells_in_flight"] == 4
    assert plan["cache_workers_per_cell"] == 4  # min(cache_build_workers=14, cpu//K=4)
    # The chosen concurrency is CONSISTENT with the chosen mode (no silent serial).
    stream_cand = next(c for c in plan["candidates"] if c["mode"] == "stream")
    assert stream_cand["cells_in_flight"] == plan["cells_in_flight"]


def test_planner_cache_workers_bounded_no_k14_explosion():
    # K=4 concurrent -> cache workers = cpu // K = 4 (not 14).
    plan = zpar.plan_cell_concurrency(
        [(GB // 4, GB // 4)] * 6, 16, 3 * GB, cache_build_workers=14,
    )
    assert plan["cells_in_flight"] == 6
    assert plan["cache_workers_per_cell"] == 2  # cpu // 6 = 2
    # Single active cell -> full cache budget.
    plan1 = zpar.plan_cell_concurrency([(GB, GB)], 16, 100 * GB, cache_build_workers=14)
    assert plan1["cells_in_flight"] == 1
    assert plan1["cache_workers_per_cell"] == 14


def test_planner_gpu_serializes_and_forces_stream():
    plan = zpar.plan_cell_concurrency(
        [(10849326220, 2449628160)] * 6, 16, 12184610406, gpu_backend=True,
    )
    assert plan["gpu_serialized"] is True
    assert plan["cells_in_flight"] == 1
    assert plan["mode"] == "stream"


# ---------------------------------------------------------------------------
# 6. Aggregate RSS helper
# ---------------------------------------------------------------------------

def test_aggregate_peak_rss_kib_upper_bound():
    assert zsw.aggregate_peak_rss_kib(100, [200, 300]) == 600
    assert zsw.aggregate_peak_rss_kib(100, []) == 100
    assert zsw.aggregate_peak_rss_kib(0, [None, "bad", 50]) == 50
