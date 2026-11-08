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
    _corpus,
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


def _low_n_corpus(n=3, h=20, w=18, seed=11):
    """Small n-safe corpus (the M4 helper assumes n >= 4)."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(2):
        cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
        base += 8.0 * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2)
        )
    gains = np.linspace(0.99, 1.01, n)
    frames = []
    for i in range(n):
        frame = np.stack(
            [
                base * gains[i] * (0.99 + 0.01 * c)
                + rng.normal(0.0, 0.7, (h, w))
                for c in range(3)
            ],
            axis=-1,
        )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    if n >= 2:
        masks[1][:, w // 2:] = False
    if n >= 3:
        frames[2][2:4, 2:4, :] = np.nan
    return frames, masks


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
        observation_domain=rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE,
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
        <= decision["residual_batch_budget_bytes"]
    )
    # The ONE truthful host peak is <= the margin ceiling, and the batch was
    # sized from the residual AFTER the fixed concurrent host terms.
    host = decision["host_components"]
    assert decision["host_peak_bytes"] == host["peak_bytes"]
    assert decision["host_peak_bytes"] <= decision["phase_budget_bytes"]
    assert host["persistent_result_bytes"] > 0
    assert host["reduce_tile_outputs_bytes"] > 0
    assert host["provider_window_bytes"] > 0
    assert host["write_tile_bytes"] > 0
    assert host["disk_bytes_in_host_peak"] == 0
    assert (
        decision["residual_batch_budget_bytes"]
        == decision["phase_budget_bytes"] - host["reduce_fixed_bytes"]
    )
    assert decision["budget_domain"] == (
        rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE
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
    # A large frame count makes the minimum batch dominate the write-phase
    # fixed terms, so the residual can hold EXACTLY one pixel.
    frames = 100_000
    probe = zext.estimate_external_working_set(
        n_frames=frames,
        height=64,
        width=64,
        channels=3,
        tile_size=8,
        halo_px=32,
        column_batch=1,
    )
    min_batch = int(probe["host"]["reduction_batch_bytes"])
    required = max(
        int(probe["host"]["fixed_peak_bytes"]),
        int(probe["host"]["reduce_fixed_bytes"]) + min_batch,
    )
    # Smallest budget whose 15/100 margin ceiling holds the fixed host terms
    # AND exactly ONE pixel batch.
    budget = (required * 115 + 99) // 100
    decision = _selector(
        tmp_path,
        monkeypatch,
        budget_bytes=budget,
        max_n_frames=frames,
        worst_patch_hw=(64, 64),
        permitted_tiles=(8,),
    )
    assert decision["refused"] is False, decision
    # ``column_batch`` is measured in POSITIONS: one pixel x 3 channels.
    assert decision["column_batch"] == 3
    assert decision["pricing"]["ram"]["rows_per_batch"] == 1
    assert decision["pricing"]["ram"]["cols_per_batch"] == 1
    assert (
        decision["pricing"]["ram"]["reduction_workspace_bytes"] == min_batch
    )
    assert decision["host_peak_bytes"] <= decision["phase_budget_bytes"]
    assert decision["residual_batch_budget_bytes"] >= min_batch


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
# 3b. Truthful host model: fixed terms, disk separation, ghost cache.
# ---------------------------------------------------------------------------

def test_large_patch_outputs_shrink_the_column_batch(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 78 * GIB)
    small = _selector(
        tmp_path, monkeypatch, worst_patch_hw=(128, 128),
        permitted_tiles=(16,),
    )
    large = _selector(
        tmp_path, monkeypatch, worst_patch_hw=(1024, 1024),
        permitted_tiles=(16,),
    )
    assert small["refused"] is False, small
    assert large["refused"] is False, large
    assert (
        large["host_components"]["persistent_result_bytes"]
        > small["host_components"]["persistent_result_bytes"]
    )
    assert (
        large["residual_batch_budget_bytes"]
        < small["residual_batch_budget_bytes"]
    )
    assert large["column_batch"] <= small["column_batch"]
    assert large["host_peak_bytes"] <= large["phase_budget_bytes"]


def test_huge_spool_disk_envelope_does_not_inflate_host_ram(
    tmp_path, monkeypatch
):
    # A large spool (2000 frames x a large patch) has a multi-GiB DISK
    # envelope. With ample disk it is admitted, and the host peak must be
    # strictly lower and independent of those disk bytes.
    _free(monkeypatch, 200 * GIB)
    decision = _selector(
        tmp_path,
        monkeypatch,
        max_n_frames=2000,
        worst_patch_hw=(2048, 2048),
        permitted_tiles=(128,),
        halo_px=64,
    )
    assert decision["refused"] is False, decision
    host = decision["host_components"]
    assert host["disk_bytes_in_host_peak"] == 0
    assert decision["host_peak_bytes"] < decision["disk_envelope_bytes"]
    assert decision["host_peak_bytes"] <= decision["phase_budget_bytes"]
    assert decision["host_peak_bytes"] < 1024 * MIB  # no disk terms leaked in


def test_generic_cache_worker_footprint_is_ghost_for_external(
    tmp_path, monkeypatch
):
    huge = 10 * GIB
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: huge)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", huge)
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)

    # The selector never consults the generic aligned-cache term.
    sel = _selector(tmp_path, monkeypatch)
    assert sel["refused"] is False, sel

    # The product ACTIVE external plan charges the cache phase exactly 0.
    descs = _corpus(4)
    bundle = _bundle(
        tmp_path, monkeypatch, force_external=True, descs=descs,
    )
    assert bundle.chosen_mode == "external"
    decision = zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=zg.build_canvas(descs),
        halo_px=int(bundle.layout.get("halo_px", 8) or 8),
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=_frame_ids(descs),
    )
    cache_env = decision.plan_set.phase_map()[renv.PHASE_CACHE_BUILD]
    assert cache_env.estimate.peak_bytes == 0
    assert (
        decision.plan_set.aggregate_host_peak_bytes
        <= bundle.envelope.budget_bytes
    )


def test_external_observation_domain_is_closed(tmp_path, monkeypatch):
    _free(monkeypatch, 78 * GIB)
    with pytest.raises(rcontract.RamContractError):
        _selector(
            tmp_path, monkeypatch,
            observation_domain="gross_total",
        )
    with pytest.raises(rcontract.RamContractError):
        _selector(
            tmp_path, monkeypatch,
            observation_domain=None,
        )


def test_external_host_component_formula_matches_real_arrays():
    px, c = 40 * 40, 3
    host = zext.external_cell_host_components(
        n_frames=5,
        patch_hw=(40, 40),
        tile_hw=(48, 48),
        interior_hw=(40, 40),
        channels=c,
        column_batch=6,
        rejection_sink=True,
    )
    persistent_arrays = [
        np.empty((40, 40, c), np.float32),
        np.empty((40, 40, c), np.float64),
        np.empty((40, 40, c), bool),
        np.empty((40, 40, c), np.int64),
    ]
    support_arrays = [
        np.empty((40, 40), np.float64),
        np.empty((40, 40), np.float64),
    ]
    assert host["persistent_result_bytes"] == (
        sum(a.nbytes for a in persistent_arrays)
        + sum(a.nbytes for a in support_arrays)
    )
    tile_arrays = [
        np.empty((48, 48, c), np.float32),
        np.empty((48, 48, c), np.float64),
        np.empty((48, 48, c), bool),
        np.empty((48, 48, c), np.int64),
        np.empty((40, 40), np.float64),
        np.empty((40, 40), np.float64),
    ]
    assert host["reduce_tile_outputs_bytes"] == sum(
        a.nbytes for a in tile_arrays
    )
    assert host["provider_window_bytes"] == 40 * 40 * (4 * c + 1)
    assert host["write_tile_bytes"] == 48 * 48 * (
        c * zext.WRITE_TILE_BYTES_PER_PIXEL_CHANNEL
        + zext.WRITE_TILE_AUX_BYTES_PER_PIXEL
    )
    assert host["finalize_n_eff_output_bytes"] == 40 * 40 * 8
    assert host["finalize_working_bytes"] == (
        40 * 40 * zext.FINALIZE_WORKING_BYTES_PER_PIXEL
    )
    assert host["finalize_equalize_bytes"] == 0
    assert host["finalize_total_bytes"] == host["finalize_working_bytes"]
    assert host["phases"]["finalize"] == (
        host["persistent_result_bytes"] + host["finalize_total_bytes"]
    )
    assert host["reduction_batch_bytes"] == zext.external_batch_bytes(5, 6)
    assert host["batch_sample_bytes_per_position_frame"] == (
        zext.EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
    )
    assert host["batch_fixed_position_bytes"] == (
        zext.EXTERNAL_BATCH_FIXED_POSITION_BYTES
    )
    assert host["phases"]["reduce"] == (
        host["persistent_result_bytes"]
        + host["reduce_tile_outputs_bytes"]
        + host["reduction_batch_bytes"]
    )
    assert host["peak_bytes"] == max(host["phases"].values())
    assert host["disk_bytes_in_host_peak"] == 0

    dense = zext.external_cell_host_components(
        n_frames=5,
        patch_hw=(40, 40),
        tile_hw=(48, 48),
        interior_hw=(40, 40),
        channels=c,
        column_batch=6,
        rejection_sink=False,
    )
    assert dense["dense_rejection_mask_bytes"] == 5 * px * c
    assert dense["persistent_result_bytes"] == (
        host["persistent_result_bytes"] + 5 * px * c
    )


def test_result_unit_formula_covers_all_core_planes():
    px, c = 40 * 40, 3
    arrays = [
        np.empty((40, 40, c), np.float32),
        np.empty((40, 40, c), np.float64),
        np.empty((40, 40, c), bool),
        np.empty((40, 40, c), np.int64),
        np.empty((40, 40), np.float64),
        np.empty((40, 40), np.float64),
        np.empty((40, 40), np.float64),
    ]
    expected = sum(a.nbytes for a in arrays) + 7 * 128 + 3 * 8
    assert zz._external_result_unit_bytes(px, c) == expected
    # The old (pre-rework) formula omitted both support planes and n_eff.
    legacy = px * c * 21 + 2 * 8 + 8
    assert zz._external_result_unit_bytes(px, c) > legacy


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
    contract = bundle.external_contract
    assert contract["budget_domain"] == (
        rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE
    )
    assert contract["host_components"]["peak_bytes"] == (
        contract["host_peak_bytes"]
    )
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
    assert reduction == contract["host_peak_bytes"]

    # The product application uses the SAME frozen peak and charges the ghost
    # cache zero. A large disk envelope must not appear in the host aggregate.
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 4 * GIB)
    admitted, _cand, mapping, _outcome, _trail = zz._r55_active_admit(
        initial_bundle=bundle, decision=decision
    )
    assert mapping["cell_task_mode"] == "external"
    assert admitted.envelope.stack_total_bytes == contract["host_peak_bytes"]
    assert admitted.envelope.cache_total_bytes == 0
    assert admitted.envelope.shared_bytes == 0
    assert admitted.envelope.aggregate_bytes == contract["host_peak_bytes"]
    assert contract["host_components"]["disk_bytes_in_host_peak"] == 0


# ---------------------------------------------------------------------------
# 4b. Rework-2 — one frozen peak, bounded finalization, equalization, batch proof.
# ---------------------------------------------------------------------------

def test_one_frozen_host_peak_across_bundle_task_active_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    descs = _corpus(4)
    bundle = _bundle(
        tmp_path, monkeypatch, force_external=True, descs=descs,
    )
    contract = bundle.external_contract
    peak = int(contract["host_peak_bytes"])
    assert contract["batch_sample_bytes_per_position_frame"] == (
        zext.EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
    )
    assert contract["batch_fixed_position_bytes"] == (
        zext.EXTERNAL_BATCH_FIXED_POSITION_BYTES
    )
    assert bundle.per_cell_footprint == peak
    ext_tasks = [t for t in bundle.cell_tasks if t[4] == "external"]
    assert ext_tasks
    assert all(int(t[5]) == peak for t in ext_tasks)
    assert bundle.working_set_block["host_peak_bytes"] == peak
    assert (
        bundle.working_set_block["host_components"]["peak_bytes"] == peak
    )

    decision = zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=zg.build_canvas(descs),
        halo_px=int(bundle.layout.get("halo_px", 8) or 8),
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=_frame_ids(descs),
    )
    stack = decision.plan_set.phase_map()[renv.PHASE_CELL_STACK_STREAM]
    assert stack.estimate.peak_bytes == peak

    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 4 * GIB)
    admitted, _cand, mapping, _outcome, _trail = zz._r55_active_admit(
        initial_bundle=bundle, decision=decision
    )
    assert mapping["cell_task_mode"] == "external"
    assert admitted.per_cell_footprint == peak
    assert admitted.envelope.stack_total_bytes == peak
    assert admitted.envelope.aggregate_bytes == peak
    assert admitted.envelope.cache_total_bytes == 0


def test_bounded_n_eff_is_bit_exact_and_finalize_components_are_priced():
    from zemosaic.core.canonical_streaming import _compute_n_eff_bounded

    rng = np.random.default_rng(20261108)
    w1 = rng.uniform(0.0, 1.0e3, (37, 41))
    w2 = rng.uniform(0.0, 1.0e3, (37, 41))
    # Deliberate edge cases: zero w2, overflow in w1**2, NaN/inf inputs,
    # negative and tiny values.
    w2[0, 0] = 0.0
    w1[1, 1] = 1.0e200
    w2[1, 1] = 1.0
    w1[2, 2] = np.nan
    w2[3, 3] = np.inf
    w1[4, 4] = -3.0
    w2[4, 4] = 2.0
    w1[5, 5] = 0.0
    w2[5, 5] = 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        w1_sq = w1 * w1
        ref = w1_sq / w2
        ratio = w1 / np.sqrt(w2)
        safe = ratio * ratio
    valid = w2 > 0.0
    overflowed = ~np.isfinite(w1_sq)
    ref = np.where(overflowed & valid, safe, ref)
    ref = np.where(
        valid & np.isfinite(ref) & (ref >= 0.0), ref, 0.0
    )
    got = _compute_n_eff_bounded(w1, w2)
    np.testing.assert_array_equal(got, ref)

    px = 40 * 40
    host = zext.external_cell_host_components(
        n_frames=5,
        patch_hw=(40, 40),
        tile_hw=(48, 48),
        interior_hw=(40, 40),
        channels=3,
        column_batch=6,
        equalize_rgb=False,
    )
    assert host["finalize_working_bytes"] == (
        px * zext.FINALIZE_WORKING_BYTES_PER_PIXEL
    )
    assert host["finalize_equalize_bytes"] == 0
    assert host["finalize_total_bytes"] == host["finalize_working_bytes"]
    assert host["phases"]["finalize"] == (
        host["persistent_result_bytes"] + host["finalize_total_bytes"]
    )


def test_equalize_rgb_true_is_priced_in_the_host_peak():
    common = dict(
        n_frames=5,
        patch_hw=(40, 40),
        tile_hw=(48, 48),
        interior_hw=(40, 40),
        channels=3,
        column_batch=6,
    )
    off = zext.external_cell_host_components(**common, equalize_rgb=False)
    on = zext.external_cell_host_components(**common, equalize_rgb=True)
    px = 40 * 40
    assert off["equalize_rgb"] is False
    assert on["equalize_rgb"] is True
    assert off["finalize_equalize_bytes"] == 0
    assert on["finalize_equalize_bytes"] == (
        px * zext.EQUALIZE_RGB_BYTES_PER_PIXEL
    )
    assert on["finalize_total_bytes"] == (
        off["finalize_total_bytes"] + on["finalize_equalize_bytes"]
    )
    assert on["phases"]["finalize"] > off["phases"]["finalize"]
    assert on["peak_bytes"] >= on["phases"]["finalize"]


@pytest.mark.parametrize("n_frames", [1, 2, 3, 101])
def test_batch_two_part_model_bounds_low_n_and_large_n(n_frames):
    """Executable worst-live-set proof at low N and large N.

    The model is ``positions * (n * sample_bytes + fixed_position_bytes)``.
    This test enumerates the REAL ``(n, positions)`` and ``(positions,)`` arrays
    for every rejection/combine pair, including the quantile return/condition
    temporaries that a fixed per-sample coefficient cannot represent.
    """
    positions = 37
    n = int(n_frames)
    common = [
        np.empty((n, positions), np.float32),   # img
        np.empty((n, positions), np.float64),   # weights
        np.empty((n, positions), np.float64),   # orig
        np.empty((n, positions), np.float64),   # wmap
    ] + [np.empty((n, positions), bool) for _ in range(6)]
    common += [
        np.empty((positions,), bool),           # degenerate_b
        np.empty((positions,), np.int64),       # init_sum
    ]
    for rejection in ("none", "kappa_sigma", "winsorized_sigma_clip"):
        for combine in ("mean", "median"):
            n_arrays = []
            fixed = []
            if rejection == "kappa_sigma":
                n_arrays += [
                    np.empty((n, positions), np.float64)  # masked
                    for _ in range(2)                     # + median sort copy
                ]
                n_arrays += [
                    np.empty((n, positions), bool) for _ in range(5)
                ]
                fixed += [
                    np.empty((positions,), np.float64) for _ in range(4)
                ]  # center, std, lower, upper
                fixed += [
                    np.empty((positions,), bool) for _ in range(2)
                ]  # active, degenerate
            elif rejection == "winsorized_sigma_clip":
                n_arrays += [
                    np.empty((n, positions), np.float64)
                    for _ in range(3)  # masked, quantile sort copy, winsor
                ]
                n_arrays += [
                    np.empty((n, positions), bool) for _ in range(6)
                ]  # valid + interval/equality/keep/keep2/and
                fixed += [
                    np.empty((positions,), np.int64) for _ in range(4)
                ]  # count, eff_count, lo, hi
                fixed += [
                    np.empty((positions,), np.float64) for _ in range(12)
                ]  # index/frac/v_lo/v_hi/diff/result/q_low/q_high/center/std/lower/upper
                fixed += [
                    np.empty((positions,), bool) for _ in range(5)
                ]  # conditions, active, degenerate
            if combine == "mean":
                n_arrays += [
                    np.empty((n, positions), np.float64) for _ in range(3)
                ]  # masked, w_contrib, product
                fixed += [
                    np.empty((positions,), np.float64) for _ in range(3)
                ]  # numerator, denominator, estimate
                fixed += [
                    np.empty((positions,), bool) for _ in range(3)
                ]
            else:
                n_arrays += [
                    np.empty((n, positions), np.float64) for _ in range(3)
                ]  # masked, sort copy, median output
                fixed += [np.empty((positions,), np.float64)]
                fixed += [
                    np.empty((positions,), bool) for _ in range(3)
                ]
            live = sum(a.nbytes for a in common + n_arrays + fixed)
            bound = positions * (
                n * zext.EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME
                + zext.EXTERNAL_BATCH_FIXED_POSITION_BYTES
            )
            assert live <= bound, (
                n, rejection, combine, live, bound
            )
            assert bound == zext.external_batch_bytes(n, positions)
            # The request-aware sample-only helper remains <= the sample term.
            assert zext.external_batch_live_bytes_per_position_frame(
                rejection, combine
            ) <= zext.EXTERNAL_BATCH_SAMPLE_BYTES_PER_POSITION_FRAME


def test_equalize_peak_helper_bounds_the_component():
    for channels in (1, 3):
        assert zext.external_equalize_peak_bytes_per_pixel(
            channels
        ) <= zext.EQUALIZE_RGB_BYTES_PER_PIXEL
    assert zext.external_equalize_peak_bytes_per_pixel(3) == 36


@pytest.mark.parametrize("n_frames", [1, 2, 3])
def test_low_n_external_reduction_matches_ram_stream(tmp_path, n_frames):
    """Real low-N external reduction (quantile path) vs the RAM stream."""
    from zemosaic.core.canonical_streaming import (
        InMemoryCanonicalProvider,
        run_canonical_stack_streaming,
    )
    from test_zegrid_m4_external_durability import _request as _m4_request

    frames, masks = _low_n_corpus(n=n_frames)
    request = _m4_request(frames, masks)
    reference = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), request, tile_size=16
    )
    result = zext.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        request,
        tile_size=16,
        column_batch=64,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id=f"low-n-{n_frames}",
        owner_token="OWNER",
        owner_record={"pid": 1},
        cleanup_on_success=True,
    )
    for name in (
        "science",
        "estimator_weight_sum",
        "valid_mask",
        "surviving_sample_count",
        "support_w1",
        "support_w2",
        "n_eff_support",
    ):
        np.testing.assert_array_equal(
            getattr(result, name),
            getattr(reference, name),
            err_msg=f"n={n_frames} {name}",
        )


@pytest.mark.slow
def test_external_end_to_end_equalize_rgb_parity(tmp_path, monkeypatch):
    from dataclasses import replace

    real_science = zz.ExecutorConfig.science_config

    def _equalized_science(self):
        return replace(real_science(self), equalize_rgb=True)

    monkeypatch.setattr(
        zz.ExecutorConfig, "science_config", _equalized_science
    )
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
    with monkeypatch.context() as m:
        m.setattr(zz, "available_memory_bytes", lambda: 1536 * MIB)
        m.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 6 * GIB)
        zz.run_zegrid_mode(
            str(inp),
            str(ext_out),
            zconfig=_stored_e2e_config(tmp_path / "ext-store"),
            storage_profile="bounded",
            cell_gauge_mode="per_cell",
        )

    ext_manifest = json.loads(
        (ext_out / "zegrid_manifest.json").read_text()
    )
    assert ext_manifest["stack_working_set"] == zz.WORKING_SET_EXTERNAL
    ws = ext_manifest["working_set"]
    assert ws["host_components"]["equalize_rgb"] is True
    assert ws["host_components"]["finalize_equalize_bytes"] > 0
    from astropy.io import fits

    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        np.testing.assert_array_equal(
            _fits_array(ref_out / name),
            _fits_array(ext_out / name),
            err_msg=name,
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
    ws = ext_manifest["working_set"]
    assert ws["phase_budgets"]["process_baseline_charged_bytes"] == 0
    assert ws["budget_domain"] == rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE
    assert ws["host_peak_bytes"] == ws["host_components"]["peak_bytes"]
    assert ws["host_components"]["disk_bytes_in_host_peak"] == 0
    assert ws["host_peak_bytes"] <= ws["phase_budgets"]["margin_ceiling_bytes"]

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
