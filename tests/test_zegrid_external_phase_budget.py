"""ZM-EXTERNAL-PHASE-BUDGET — sequential phase admission for the external class.

The production failure (Linux LP, 341 frames, ~778 MB available, B_plan
~622.5 MB, ~78 GiB disk) was a PHASE-COUPLING bug:

* the external selector subtracted ``PROCESS_BASELINE_BYTES`` from a budget
  derived from ``psutil.virtual_memory().available`` (already NET of the
  resident process), charging the same residency twice;
* it also subtracted the later ASSEMBLY canvas/halo reserve, even though
  external payloads are durably committed and released before assembly and
  assembly performs its own fresh RAM admission.
* the ACTIVE external plan summed the later assembly phase with the reduction
  phase for the same reason.

These regressions pin the corrected phase-local contract: a minimal serial
external reduction is admitted when its INCREMENTAL working set fits, the later
assembly is admitted separately (anonymous first, then scratch-backed), and a
genuine RAM or DISK shortfall still fails closed with its own reason code.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import bounded_assembly as zbas
from zemosaic.core.zegrid import external_stream as zext
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import resource_envelope as renv
from zemosaic.core.zegrid import scheduler as zsched

from test_zegrid_active_external_wiring import (  # noqa: E402
    GIB,
    MIB,
    _FakeModel,
    _bundle,
    _e2e_config,
    _fits_array,
    _frame_ids,
    _free,
    _tiny_floors,
    _write_e2e_corpus,
)

# The observed LP facts (decimal MB from the production report).
LP_RAW_AVAILABLE = 778_000_000
LP_BUDGET = (4 * LP_RAW_AVAILABLE) // 5
LP_FRAMES = 341
LP_FIXED_OVERLAP = 327_245_088


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)


def _canvas_corpus(n=4, step_px=3600.0):
    """Four 40x40 frames spread over a ~3.6k x 3.6k canvas.

    Only WCS geometry is materialised. At 16 B/px the retained-core assembly
    reserve is ~215 MB, the same order as the observed LP reserve (~312 MB),
    but still smaller than B_plan on its own.
    """
    from astropy.wcs import WCS

    frames = []
    for i in range(n):
        col, row = i % 2, i // 2
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [
            10.0 + col * float(step_px) * 0.001,
            30.0 - row * float(step_px) * 0.001,
        ]
        wcs.wcs.crpix = [20.0, 20.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (40, 40)
        fid = f"p{i}.fits"
        frames.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid),
                source_path=f"/x/{fid}",
                shape_hw=(40, 40),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return frames


def _selector(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    base = dict(
        permitted_tiles=(16,),
        worst_patch_hw=(512, 512),
        max_n_frames=LP_FRAMES,
        channels=3,
        budget_bytes=LP_BUDGET,
        halo_px=32,
        cache_root=str(tmp_path / "cache"),
        storage_profile="bounded",
    )
    base.update(kwargs)
    return zz._select_external_fallback(**base)


# ---------------------------------------------------------------------------
# 1. LP-like budget: later assembly is not reserved during the reduction.
# ---------------------------------------------------------------------------

def test_lp_like_budget_keeps_phase_local_reduction_budget(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 78 * GIB)
    decision = _selector(
        tmp_path, monkeypatch, fixed_overlap_bytes=LP_FIXED_OVERLAP
    )

    assert decision["refused"] is False, decision
    expected = (LP_BUDGET * 100) // 115
    assert decision["reduction_budget_bytes"] == expected
    assert decision["phase_budgets"]["phase"] == "external_reduction"
    assert decision["phase_budgets"]["plan_budget_bytes"] == LP_BUDGET
    assert decision["phase_budgets"]["process_baseline_charged_bytes"] == 0
    assert decision["phase_budgets"]["process_baseline_already_in_available"]
    assert (
        decision["phase_budgets"]["assembly_reserve_excluded_bytes"]
        == LP_FIXED_OVERLAP
    )
    assert (
        decision["phase_budgets"]["assembly_admission"]
        == "separate_at_assembly_time"
    )
    assert decision["column_batch"] >= 1
    assert (
        decision["pricing"]["ram"]["reduction_workspace_bytes"]
        <= decision["reduction_budget_bytes"]
    )

    # The PRE-FIX concurrent formula subtracted both the baseline and the
    # assembly reserve, collapsing the budget to its 1-byte floor.
    legacy = (
        expected
        - rcontract.PROCESS_BASELINE_BYTES
        - LP_FIXED_OVERLAP
    )
    assert legacy <= 0
    assert decision["reduction_budget_bytes"] > legacy


def test_minimum_micro_batch_admitted_when_incremental_fits(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 78 * GIB)
    min_ws = (
        LP_FRAMES
        * 3
        * zext.REDUCTION_BYTES_PER_POSITION_FRAME
    )
    # Smallest budget whose 15/100 margin ceiling still holds ONE pixel batch.
    budget = (min_ws * 115 + 99) // 100
    decision = _selector(
        tmp_path,
        monkeypatch,
        budget_bytes=budget,
        worst_patch_hw=(64, 64),
        permitted_tiles=(8,),
    )
    assert decision["refused"] is False, decision
    # ``column_batch`` is measured in POSITIONS: one pixel x 3 channels.
    assert decision["column_batch"] == 3
    assert decision["pricing"]["ram"]["rows_per_batch"] == 1
    assert decision["pricing"]["ram"]["cols_per_batch"] == 1
    assert (
        decision["pricing"]["ram"]["reduction_workspace_bytes"] == min_ws
    )
    assert decision["reduction_budget_bytes"] >= min_ws


# ---------------------------------------------------------------------------
# 2. Honest refusals: a true RAM floor and a true disk shortfall.
# ---------------------------------------------------------------------------

def test_true_reduction_shortfall_still_refuses_host_ram(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 78 * GIB)
    min_ws = LP_FRAMES * 3 * zext.REDUCTION_BYTES_PER_POSITION_FRAME
    decision = _selector(
        tmp_path,
        monkeypatch,
        budget_bytes=min_ws,
        worst_patch_hw=(64, 64),
        permitted_tiles=(8,),
    )
    assert decision["refused"] is True
    assert decision["reason"] == "reduction_workspace_floor"
    assert decision["free_bytes"] == 78 * GIB
    assert (
        decision["phase_budgets"]["process_baseline_charged_bytes"] == 0
    )
    assert "HOST-RAM floor" in decision["detail"]


def test_insufficient_external_disk_still_refuses_disk(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 1 * MIB)
    decision = _selector(
        tmp_path,
        monkeypatch,
        worst_patch_hw=(512, 512),
        fixed_overlap_bytes=LP_FIXED_OVERLAP,
    )
    assert decision["refused"] is True
    assert decision["reason"] == "disk_envelope_unavailable"
    assert decision["free_bytes"] == 1 * MIB
    assert decision["best_envelope_bytes"] > decision["free_bytes"]


# ---------------------------------------------------------------------------
# 3. Scheduler: the external assembly phase is SEQUENTIAL, not summed.
# ---------------------------------------------------------------------------

def _lp_context(working_set_class):
    shared = rcontract.PROCESS_BASELINE_BYTES
    workspace = LP_FRAMES * 64 * 3 * zext.REDUCTION_BYTES_PER_POSITION_FRAME
    return zsched.PlanContext(
        phase=renv.PHASE_CELL_STACK_STREAM,
        backend=renv.BACKEND_CPU,
        budget_bytes=LP_BUDGET,
        raw_available_bytes=LP_RAW_AVAILABLE,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=16,
        shared_bytes=shared,
        per_unit_stack_bytes=workspace,
        per_unit_cache_bytes=1 * MIB,
        per_unit_prefetch_bytes=0,
        disk_free_bytes=78 * GIB,
        layout=(2, 2),
        refinement_rank=4,
        patch_hw=(96, 96),
        halo_px=32,
        channels=3,
        n_frames=LP_FRAMES,
        tile_ladder=(16,),
        prefetch_ladder=(0,),
        workers_ladder=(1,),
        max_units_in_flight=1,
        working_set_class=working_set_class,
    )


def _lp_requests():
    return (
        zsched.PhaseRequest(
            phase=renv.PHASE_GAUGE,
            components={"g": 1 * MIB},
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_CACHE_BUILD,
            components={"c": 1 * MIB},
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_ASSEMBLY,
            components={
                "halo_band_bytes": LP_FIXED_OVERLAP // 2,
                "retained_core_bytes": LP_FIXED_OVERLAP // 2,
            },
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
            overlaps_with=(),
        ),
        zsched.PhaseRequest(
            phase=renv.PHASE_EXPORT,
            components={"e": 1 * MIB},
            units=1,
            units_in_flight=1,
            workers_per_unit=1,
        ),
    )


def test_external_assembly_is_not_summed_with_reduction():
    ctx = _lp_context(zsched.WORKING_SET_CLASS_EXTERNAL_SPOOL)
    plan_set = zsched.plan_joint(
        zsched.JointPlanInput(contexts=(ctx,)),
        phase_requests=_lp_requests(),
        mode=zsched.MODE_ACTIVE,
    )
    phases = plan_set.phase_map()
    # The reduction and the LATER assembly are separate, non-overlapping
    # groups: only the larger group counts.
    assert phases[renv.PHASE_CELL_STACK_STREAM].overlaps_with == ()
    assert phases[renv.PHASE_ASSEMBLY].overlaps_with == ()
    expected_raw = (
        rcontract.PROCESS_BASELINE_BYTES
        + ctx.per_unit_stack_bytes
    )
    assert plan_set.aggregate_host_peak_bytes_raw == expected_raw
    assert plan_set.aggregate_host_peak_bytes <= LP_BUDGET


def test_ram_class_still_sums_assembly_and_refuses_on_the_same_numbers():
    ctx = _lp_context(zsched.WORKING_SET_CLASS_NONE)
    with pytest.raises(zsched.PlanRefused) as exc:
        zsched.plan_joint(
            zsched.JointPlanInput(contexts=(ctx,)),
            phase_requests=_lp_requests(),
            mode=zsched.MODE_ACTIVE,
        )
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM
    summed = (
        rcontract.PROCESS_BASELINE_BYTES
        + ctx.per_unit_stack_bytes
        + LP_FIXED_OVERLAP
    )
    assert summed > (LP_BUDGET * 100) // 115


# ---------------------------------------------------------------------------
# 4. Product ACTIVE seam: the LP-like bundle admits the external class.
# ---------------------------------------------------------------------------

def test_active_external_bundle_admits_at_lp_like_budget(
    tmp_path, monkeypatch
):
    # Use the REAL shared baseline so the external plan's single charge is
    # visible, and a wide geometry whose retained-core assembly reserve is
    # large enough that summing it would exceed B_plan.
    monkeypatch.setattr(
        zpar,
        "_PER_WORKER_BASELINE_BYTES",
        rcontract.PROCESS_BASELINE_BYTES,
    )
    descs = _canvas_corpus(4)
    bundle = _bundle(
        tmp_path,
        monkeypatch,
        raw=LP_RAW_AVAILABLE,
        stream_table={int(t): 4 * GIB for t in zz.CPU_STREAMING_TILE_LADDER},
        force_external=True,
        descs=descs,
        # Pin the geometry: the point is the LARGE canvas assembly reserve,
        # not an adaptive-layout search over a 4k canvas.
        pinned_layout=(2, 2),
    )
    assert bundle.chosen_mode == "external"
    assert bundle.external_contract["phase_budgets"][
        "process_baseline_charged_bytes"
    ] == 0

    canvas = zg.build_canvas(descs)
    decision = zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=int(bundle.layout.get("halo_px", 8) or 8),
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=_frame_ids(descs),
    )
    assert decision.working_set_class == zz.WORKING_SET_EXTERNAL
    phases = decision.plan_set.phase_map()
    assert phases[renv.PHASE_CELL_STACK_STREAM].overlaps_with == ()
    assert phases[renv.PHASE_ASSEMBLY].overlaps_with == ()
    assert decision.plan_set.aggregate_host_peak_bytes <= LP_BUDGET
    # The assembly reserve alone is real but < B_plan; summing it with the
    # reduction would still exceed the margin ceiling, so this witness proves
    # the phases are NOT summed.
    assembly = phases[renv.PHASE_ASSEMBLY].estimate.peak_bytes
    reduction = phases[renv.PHASE_CELL_STACK_STREAM].estimate.peak_bytes
    ceiling = (LP_BUDGET * 100) // 115
    assert 0 < assembly < ceiling
    assert reduction > 0
    assert reduction + assembly > ceiling
    assert decision.plan_set.aggregate_host_peak_bytes_raw == max(
        reduction, assembly
    )


# ---------------------------------------------------------------------------
# 5. End-to-end: external reduce -> durable commit/release -> separately
#    admitted scratch-backed assembly -> FITS parity.
# ---------------------------------------------------------------------------

def _stored_e2e_config(store_dir):
    """The e2e config plus an explicit durable store (bounded assembly seam).

    A store-less run assembles eagerly in anonymous RAM and never reaches the
    bounded preflight/scratch path. The external acceptance runs the REAL
    durable-store flow so the final assembly is separately admitted.
    """
    config = _e2e_config()
    config.zegrid_durable_store_dir = str(store_dir)
    config.zegrid_durable_store_budget_bytes = 2 * GIB
    config.zegrid_resume_mode = "never"
    return config


@pytest.mark.slow
def test_external_end_to_end_scratch_assembly_parity(tmp_path, monkeypatch):
    inp = _write_e2e_corpus(tmp_path)
    ref_out = tmp_path / "ref"
    ext_out = tmp_path / "ext"

    zz.run_zegrid_mode(
        str(inp),
        str(ref_out),
        zconfig=_stored_e2e_config(tmp_path / "ref-store"),
        storage_profile="bounded",
        cell_gauge_mode="per_cell",
    )
    ref_manifest = json.loads(
        (ref_out / "zegrid_manifest.json").read_text()
    )
    assert ref_manifest["stack_working_set"] == zz.WORKING_SET_RAM_TILE

    backings = []
    scratch_plans = []
    real_guard = zz._assemble_cells_guarded
    real_preflight = zz._bounded_assembly_preflight

    def _guard_spy(canvas, nx, ny, mode, cores, result_by_cid, **kwargs):
        backings.append(str(kwargs.get("canvas_backing")))
        return real_guard(
            canvas, nx, ny, mode, cores, result_by_cid, **kwargs
        )

    def _preflight(canvas, bounded_results, **kwargs):
        # Force the REAL anonymous preflight to refuse by pricing it against a
        # 1-byte budget; the production catch must retry with the REAL
        # scratch-backed allocator and its own fresh admission.
        if str(kwargs.get("canvas_backing")) == "anonymous":
            kwargs = dict(kwargs)
            kwargs["available_bytes"] = 1
        else:
            plan = real_preflight(canvas, bounded_results, **kwargs)
            scratch_plans.append(plan)
            return plan
        return real_preflight(canvas, bounded_results, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(zz, "available_memory_bytes", lambda: 1536 * MIB)
        m.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 6 * GIB)
        m.setattr(zz, "_bounded_assembly_preflight", _preflight)
        m.setattr(zz, "_assemble_cells_guarded", _guard_spy)
        zz.run_zegrid_mode(
            str(inp),
            str(ext_out),
            zconfig=_stored_e2e_config(tmp_path / "ext-store"),
            storage_profile="bounded",
            cell_gauge_mode="per_cell",
        )

    assert backings[0] == "anonymous"
    assert "scratch" in backings
    assert scratch_plans, "the scratch-backed assembly was not preflighted"
    scratch = scratch_plans[-1]
    assert scratch["canvas_backing"] == "scratch"
    assert scratch["canvas_ram_bytes"] == 0
    assert scratch["scratch_bytes"] >= 1
    assert scratch["fits"] is True
    assert scratch["host"]["budget_source"] == (
        "ram_contract.admit_budget_bytes"
    )

    ext_manifest = json.loads(
        (ext_out / "zegrid_manifest.json").read_text()
    )
    assert ext_manifest["stack_working_set"] == zz.WORKING_SET_EXTERNAL
    assert ext_manifest["working_set"]["class"] == zz.WORKING_SET_EXTERNAL
    assert (
        ext_manifest["working_set"]["phase_budgets"][
            "process_baseline_charged_bytes"
        ]
        == 0
    )

    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        np.testing.assert_array_equal(
            _fits_array(ref_out / name),
            _fits_array(ext_out / name),
            err_msg=name,
        )
    ref_cells = [
        (c["row"], c["col"], c["n_contributors"])
        for c in ref_manifest.get("cells", [])
    ]
    ext_cells = [
        (c["row"], c["col"], c["n_contributors"])
        for c in ext_manifest.get("cells", [])
    ]
    assert ext_cells == ref_cells
