"""ZM-ZEGRID-R32C-GAUGE-FOOTPRINT-WIRING — one memory model for the gauge.

The product refused the Caldwell LP run because the ACTIVE admission still
charged the historical R17 full-canvas gauge bound (870 472 072 B) while the
executed R32c path is bounded + tiled. These regressions pin:

* the exact Caldwell witness (old canvas bound vs the R32c bound);
* the single-source descriptor (bounded pairs, tiled reference, tile 1024);
* MAX of sequential sub-phases (never their sum);
* the strict override returning the historical canvas bound;
* the execution diagnostics carrying the SAME model the planner froze;
* the real external ACTIVE seam: the canvas model refuses, the R32c model
  admits, and a truly larger bbox/tile still fails closed with
  ``R55_REFUSE_HOST_RAM`` naming ``phase=gauge`` and its components.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import photometric as zphot

from test_zegrid_active_external_wiring import (
    GIB,
    MIB,
    _FakeModel,
    _frame_ids,
    _tiny_floors,
)
from test_zegrid_r32a_gauge_perf import (
    _dithered_corpus,
    _run,
    _run_default,
)
from zemosaic.core.zegrid.executor import ExecutorConfig

CALDWELL_CANVAS_PX = 3477 * 4106
CALDWELL_BBOX_PX = 1975 * 1163
CALDWELL_FRAME_PX = 1920 * 1080
CALDWELL_TILE_PX = 1024
CALDWELL_BUDGET = 968_232_140
CALDWELL_OLD_BOUND = 870_472_072
CALDWELL_RAW_CEILING = 841_940_991
CALDWELL_NEW_BOUND = 831_596_340
# The superseded MAX(bbox,frame,tile)*36 model: it omitted the decoded source and
# the canonical/weighting workspaces. Kept ONLY to prove an intermediate budget
# would have been falsely admitted by it.
CALDWELL_OLD_MODEL_BOUND = 439_205_140


# ---------------------------------------------------------------------------
# 1. Caldwell numeric witness (the real product failure).
# ---------------------------------------------------------------------------

def test_caldwell_witness_old_canvas_vs_r32c_bound():
    old = int(zpar.gauge_footprint_bytes(CALDWELL_CANVAS_PX))
    assert old == CALDWELL_OLD_BOUND

    model = zphot.gauge_memory_model(
        canvas_px=CALDWELL_CANVAS_PX,
        max_bbox_px=CALDWELL_BBOX_PX,
        max_frame_px=CALDWELL_FRAME_PX,
        normalization="sky_mean",
        weighting="noise_variance",
    )
    assert model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C
    assert model["bounded_pairs"] is True
    assert model["tiled_reference"] is True
    assert model["tile_px"] == CALDWELL_TILE_PX
    assert model["max_bbox_px"] == CALDWELL_BBOX_PX
    assert model["dominant_area_px"] == CALDWELL_BBOX_PX
    assert model["peak_phase"] == "pairs_normalize"
    assert model["footprint_bytes"] == CALDWELL_NEW_BOUND
    # The decoded source and canonical batch are explicitly charged.
    assert model["phase_peaks"]["decode"]["added_bytes"] == 24 * CALDWELL_FRAME_PX
    assert model["phase_peaks"]["pairs_normalize"]["added_bytes"] == (
        12 * CALDWELL_FRAME_PX + 196 * CALDWELL_BBOX_PX
    )

    # The frozen raw ceiling of the observed B_plan (15/100 margin).
    assert zz._r55_margin_raw_ceiling(CALDWELL_BUDGET) == CALDWELL_RAW_CEILING
    assert CALDWELL_NEW_BOUND <= CALDWELL_RAW_CEILING
    assert CALDWELL_OLD_BOUND > CALDWELL_RAW_CEILING

    # One worker admits under the SAME budget/margin with the corrected bound,
    # while the historical canvas bound is refused.
    assert zz._r55_coherent_worker_count(1, CALDWELL_RAW_CEILING, CALDWELL_NEW_BOUND) == 1
    assert (CALDWELL_NEW_BOUND * 115 + 99) // 100 <= CALDWELL_BUDGET
    assert (CALDWELL_OLD_BOUND * 115 + 99) // 100 > CALDWELL_BUDGET


def test_intermediate_budget_would_have_been_falsely_admitted_by_the_old_model():
    # 600 MB: the superseded MAX-area model's margin fits, but the corrected
    # source+canonical peak does NOT. This is the exact failure mode a MAX-only
    # bound creates.
    budget = 600_000_000
    old_margin = (CALDWELL_OLD_MODEL_BOUND * 115 + 99) // 100
    new_margin = (CALDWELL_NEW_BOUND * 115 + 99) // 100
    assert old_margin <= budget
    assert new_margin > budget
    raw_ceiling = zz._r55_margin_raw_ceiling(budget)
    assert CALDWELL_OLD_MODEL_BOUND <= raw_ceiling
    assert CALDWELL_NEW_BOUND > raw_ceiling


# ---------------------------------------------------------------------------
# 2. Single model: MAX of sequential sub-phases, never their sum.
# ---------------------------------------------------------------------------

def test_model_uses_max_of_sub_phases_not_sum():
    canvas_px = 10_000 * 10_000
    bbox = 500 * 500
    frame = 1920 * 1080
    tile = 1024 * 1024
    model = zphot.gauge_memory_model(
        canvas_px=canvas_px,
        max_bbox_px=bbox,
        max_frame_px=frame,
        tile_px=1024,
        normalization="sky_mean",
        weighting="noise_variance",
    )
    assert model["dominant_area_px"] == max(bbox, frame, min(tile, canvas_px))
    peaks = {
        name: int(entry["added_bytes"])
        for name, entry in model["phase_peaks"].items()
    }
    assert model["subphase_added_bytes"] == max(peaks.values())
    assert model["footprint_bytes"] == (
        int(zpar._PER_WORKER_BASELINE_BYTES) + max(peaks.values())
    )
    # The decoded source is charged in the reproject/decode phases.
    assert peaks["decode"] == 24 * frame
    assert peaks["pairs_reproject"] == 36 * frame + 96 * bbox
    # It is NOT the sum of every sub-phase.
    assert model["subphase_added_bytes"] < sum(peaks.values())

    # A tile larger than the bbox but clipped to a smaller canvas is used as a
    # conservative area, still capped by the canvas.
    small_canvas = 600 * 600
    clipped = zphot.gauge_memory_model(
        canvas_px=small_canvas,
        max_bbox_px=1000,
        max_frame_px=1000,
        tile_px=1024,
    )
    assert clipped["reference_tile_area_px"] == small_canvas
    assert clipped["dominant_area_px"] == small_canvas


# ---------------------------------------------------------------------------
# 3. Strict override keeps the historical full-canvas bound.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounded_pairs": False},
        {"tiled_reference": False},
        {"bounded_pairs": False, "tiled_reference": False},
    ],
)
def test_strict_override_returns_the_canvas_bound(kwargs):
    model = zphot.gauge_memory_model(
        canvas_px=CALDWELL_CANVAS_PX,
        max_bbox_px=CALDWELL_BBOX_PX,
        max_frame_px=CALDWELL_FRAME_PX,
        **kwargs,
    )
    assert model["mode"] == zphot._GAUGE_MEMORY_MODE_CANVAS
    assert model["strict_reason"]
    # Strict is now the SAME sub-phase model with canvas-equivalent areas; it is
    # strictly MORE conservative than the historical baseline+canvas*36.
    assert model["footprint_bytes"] > CALDWELL_OLD_BOUND
    assert model["peak_phase"] == "pairs_normalize"
    assert model["dominant_area_px"] == CALDWELL_CANVAS_PX


def test_unknown_bbox_falls_back_to_canvas_bound():
    model = zphot.gauge_memory_model(canvas_px=CALDWELL_CANVAS_PX)
    assert model["mode"] == zphot._GAUGE_MEMORY_MODE_CANVAS
    assert model["strict_reason"] == "bbox area unknown"
    assert model["footprint_bytes"] > CALDWELL_OLD_BOUND


# ---------------------------------------------------------------------------
# 4. RUN publishes the SAME model the planner froze.
# ---------------------------------------------------------------------------

def test_gauge_execution_diagnostics_carry_the_same_model():
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    planned = zphot.gauge_memory_model(
        canvas,
        descs,
        normalization=str(cfg.normalization),
        weighting=str(cfg.weighting),
    )
    _, _, diag = _run_default(descs, canvas, cfg)
    assert diag["memory_model"]["schema"] == planned["schema"]
    assert diag["memory_model"]["mode"] == zphot._GAUGE_MEMORY_MODE_R32C
    assert diag["memory_model"]["footprint_bytes"] == planned["footprint_bytes"]
    assert diag["memory_model"]["tile_px"] == zphot._GAUGE_REF_TILE_PX

    _, _, strict_diag = _run(
        descs,
        canvas,
        cfg,
        prefilter_pairs=False,
        dedup_reference=False,
        bounded_pairs=False,
        tiled_reference=False,
    )
    assert strict_diag["memory_model"]["mode"] == zphot._GAUGE_MEMORY_MODE_CANVAS
    assert strict_diag["memory_model"]["footprint_bytes"] > int(
        zpar.gauge_footprint_bytes(int(canvas.width) * int(canvas.height))
    )


# ---------------------------------------------------------------------------
# 5. Real ACTIVE external seam: the canvas model refuses, R32c admits.
# ---------------------------------------------------------------------------

def _wide_geometry_corpus():
    """Four 1000x1000 frames spread over a ~5088x5525 canvas (geometry only)."""
    frames = []
    for i, (ra, dec) in enumerate(
        [(10.0, 30.0), (14.5, 30.0), (10.0, 25.5), (14.5, 25.5)]
    ):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [500.0, 500.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (1000, 1000)
        fid = f"g{i}.fits"
        frames.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid),
                source_path=f"/x/{fid}",
                shape_hw=(1000, 1000),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return frames


def _external_bundle(tmp_path, descs, canvas=None):
    canvas = zg.build_canvas(descs) if canvas is None else canvas
    raw = 1280 * MIB  # B_plan = 1024 MiB
    with mock.patch.object(zz, "_STACK_STORAGE_PROFILE", "bounded"), mock.patch.object(
        zal, "ScientificFloors", _tiny_floors
    ), mock.patch.object(
        zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62)
    ), mock.patch.object(
        zz.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100 * GIB, used=0, free=100 * GIB),
    ), mock.patch.object(
        os, "cpu_count", lambda: 8
    ):
        bundle = zz._build_execution_bundle(
            canvas,
            descs,
            None,
            raw_available_bytes=raw,
            budget_bytes=(4 * raw) // 5,
            cpu_count=8,
            cache_root=str(tmp_path / "cache"),
            force_external=True,
            pinned_layout=(4, 4),
            max_refinement_rank=16,
            halo_px=32,
        )
    return bundle, canvas


def _decide(bundle, tmp_path, frame_ids, descs):
    return zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=zg.build_canvas(descs),
        halo_px=32,
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=frame_ids,
    )


def test_active_external_seam_canvas_refuses_r32c_admits(tmp_path):
    descs = _wide_geometry_corpus()
    frame_ids = _frame_ids(descs)
    bundle, canvas = _external_bundle(tmp_path, descs)
    assert bundle.chosen_mode == "external"
    model = bundle.gauge_memory_model
    assert model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C
    new_bound = int(model["footprint_bytes"])
    canvas_bound = int(zpar.gauge_footprint_bytes(canvas.width * canvas.height))
    assert new_bound < canvas_bound

    # The real seam admits the R32c leg and reaches the gauge phase.
    decision = _decide(bundle, tmp_path, frame_ids, descs)
    assert decision.working_set_class == zz.WORKING_SET_EXTERNAL
    assert decision.facts["gauge_footprint_bytes"] == new_bound
    gauge_env = decision.plan_set.phase_map()["gauge"]
    expected_gauge = int(decision.facts["gauge_workers"]) * new_bound
    assert gauge_env.estimate.subtotal_bytes == expected_gauge
    assert (
        gauge_env.estimate.components_dict()["gauge_worker_footprint_bytes"]
        == expected_gauge
    )

    # A/B on the SAME seam: the historical canvas bound is refused.
    old_model = dict(model)
    old_model.update(
        mode=zphot._GAUGE_MEMORY_MODE_CANVAS,
        footprint_bytes=canvas_bound,
        max_bbox_px=int(canvas.width) * int(canvas.height),
        strict_reason="test reproduction of the pre-fix canvas model",
    )
    old_bundle = replace(bundle, gauge_memory_model=old_model)
    with pytest.raises(zz.zsched.PlanRefused) as refused:
        _decide(old_bundle, tmp_path, frame_ids, descs)
    assert refused.value.reason_code == zz.renv.REASON_REFUSE_HOST_RAM
    assert "phase=gauge" in refused.value.detail
    assert f"footprint_bytes={canvas_bound}" in refused.value.detail


# ---------------------------------------------------------------------------
# 6. Fail-closed: a genuinely large bbox still refuses, naming phase=gauge.
# ---------------------------------------------------------------------------

def test_large_bbox_fails_closed_with_explicit_gauge_phase(tmp_path):
    descs = _wide_geometry_corpus()
    frame_ids = _frame_ids(descs)
    bundle, canvas = _external_bundle(tmp_path, descs)
    canvas_px = int(canvas.width) * int(canvas.height)
    huge = zphot.gauge_memory_model(
        canvas_px=canvas_px,
        max_bbox_px=canvas_px,
        max_frame_px=1000 * 1000,
    )
    assert huge["mode"] == zphot._GAUGE_MEMORY_MODE_R32C
    assert huge["dominant_area_px"] == canvas_px
    assert huge["footprint_bytes"] > int(zpar.gauge_footprint_bytes(canvas_px))
    assert huge["footprint_bytes"] > int(bundle.envelope.budget_bytes)

    with pytest.raises(zz.zsched.PlanRefused) as refused:
        _decide(
            replace(bundle, gauge_memory_model=huge),
            tmp_path,
            frame_ids,
            descs,
        )
    assert refused.value.reason_code == zz.renv.REASON_REFUSE_HOST_RAM
    detail = refused.value.detail
    assert "phase=gauge" in detail
    assert f"footprint_bytes={huge['footprint_bytes']}" in detail
    assert f"max_bbox_px={canvas_px}" in detail
    assert "mode=r32c_bounded_tiled" in detail


# ---------------------------------------------------------------------------
# 7. R28 invariant: EXACTLY N WCS projections per bundle build, descriptor
#    transported layout -> bundle -> replans (never re-derived from descs).
# ---------------------------------------------------------------------------

def _small_corpus(n=4):
    frames = []
    for i in range(n):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + i * 0.004, 30.0]
        wcs.wcs.crpix = [20.0, 20.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (40, 40)
        fid = f"d{i}.fits"
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


def _spy_source_polygon(monkeypatch):
    real = zg._source_polygon
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(zg, "_source_polygon", spy)
    return calls


def _small_bundle(tmp_path, descs, *, pinned=None, model=None, footprints=None,
                  restored=None, canvas=None):
    canvas = zg.build_canvas(descs) if canvas is None else canvas
    raw = 2 * GIB
    with mock.patch.object(zz, "_STACK_STORAGE_PROFILE", "bounded"), mock.patch.object(
        zal, "ScientificFloors", _tiny_floors
    ), mock.patch.object(
        zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62)
    ), mock.patch.object(
        zz.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=100 * GIB, used=0, free=100 * GIB),
    ), mock.patch.object(
        os, "cpu_count", lambda: 8
    ):
        return zz._build_execution_bundle(
            canvas,
            descs,
            footprints,
            raw_available_bytes=raw,
            budget_bytes=(4 * raw) // 5,
            cpu_count=8,
            cache_root=str(tmp_path / "cache"),
            pinned_layout=pinned,
            max_refinement_rank=(None if pinned is None else pinned[0] * pinned[1]),
            halo_px=8,
            gauge_memory_model=model,
            restored_geometry=restored,
        )


def test_bundle_projects_exactly_n_and_transports_the_descriptor(tmp_path, monkeypatch):
    descs = _small_corpus()
    canvas = zg.build_canvas(descs)
    calls = _spy_source_polygon(monkeypatch)

    auto = _small_bundle(tmp_path, descs, canvas=canvas)
    assert len(calls) == len(descs)
    assert auto.gauge_memory_model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C

    calls.clear()
    pinned = _small_bundle(tmp_path, descs, pinned=(2, 2), canvas=canvas)
    assert len(calls) == len(descs)
    assert pinned.gauge_memory_model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C

    # Replan with the frozen footprint artifact: ZERO new WCS projections, and
    # the descriptor is the SAME object content (transported verbatim).
    calls.clear()
    replan = _small_bundle(
        tmp_path,
        descs,
        pinned=(2, 2),
        footprints=pinned.footprints,
        model=pinned.gauge_memory_model,
        canvas=canvas,
    )
    assert len(calls) == 0
    assert replan.gauge_memory_model == pinned.gauge_memory_model
    assert replan.gauge_workers == pinned.gauge_workers


def test_external_pinned_bundle_projects_exactly_n(tmp_path, monkeypatch):
    descs = _wide_geometry_corpus()
    canvas = zg.build_canvas(descs)
    calls = _spy_source_polygon(monkeypatch)
    bundle, _canvas = _external_bundle(tmp_path, descs, canvas=canvas)
    assert len(calls) == len(descs)
    assert bundle.gauge_memory_model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C
    assert bundle.chosen_mode == "external"

    calls.clear()
    replan = _small_bundle(
        tmp_path,
        descs,
        pinned=(4, 4),
        footprints=bundle.footprints,
        model=bundle.gauge_memory_model,
        canvas=canvas,
    )
    assert len(calls) == 0
    assert replan.gauge_memory_model == bundle.gauge_memory_model


def test_restored_geometry_fallback_projects_once_and_freezes(tmp_path, monkeypatch):
    descs = _small_corpus()
    canvas = zg.build_canvas(descs)
    base = _small_bundle(tmp_path, descs, pinned=(2, 2), canvas=canvas)
    restored = (canvas, base.layout, base.cell_ctxs)

    calls = _spy_source_polygon(monkeypatch)
    bundle = _small_bundle(tmp_path, descs, restored=restored, canvas=canvas)
    # No footprints artifact exists on the restored path: ONE unique projection
    # pass, then the descriptor is frozen in the bundle.
    assert len(calls) == len(descs)
    assert bundle.gauge_memory_model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C
    assert bundle.gauge_memory_model == base.gauge_memory_model

    calls.clear()
    replan = _small_bundle(
        tmp_path,
        descs,
        restored=restored,
        model=bundle.gauge_memory_model,
        canvas=canvas,
    )
    assert len(calls) == 0
    assert replan.gauge_memory_model == bundle.gauge_memory_model


# ---------------------------------------------------------------------------
# 8. Dense -> bounded retry at the REAL product seam: the artifact + descriptor
#    produced by the REFUSED dense build survive into the bounded retry.
# ---------------------------------------------------------------------------

def test_product_dense_to_bounded_retry_projects_exactly_n(tmp_path, monkeypatch):
    from test_zegrid_r52_active_ooc_fallback import (
        _HugeModel,
        _fake_stream_bytes,
        _write_large_corpus,
    )

    inp = _write_large_corpus(tmp_path)
    frames, rejected = zg.read_manifest(inp)
    assert len(frames) == 4
    assert rejected == []
    canvas = zg.build_canvas(frames)

    budget = 4096 * MIB
    raw = budget * zz.rcontract.PLAN_FACTOR_DEN // zz.rcontract.PLAN_FACTOR_NUM
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _fake_stream_bytes)
    zz._storage_policy_reset(None)

    calls = _spy_source_polygon(monkeypatch)
    bundle = zz._plan_bundle_auto(
        canvas=canvas,
        descs=frames,
        footprints=None,
        raw=raw,
        budget=budget,
        active=True,
        emit=None,
        bundle_common={
            "cpu_count": 2,
            "gpu_used": False,
            "cache_root": None,
            "halo_px": 128,
        },
        layout_substep=None,
    )

    # The refused dense build already produced the footprint artifact +
    # descriptor; the bounded retry MUST reuse them (N total, not 2N).
    assert len(calls) == len(frames)
    storage = zz._storage_policy_snapshot()
    assert storage["effective"] == "bounded"
    assert storage["fallback_count"] == 1
    assert bundle.storage_profile == "bounded"

    expected = zphot.gauge_memory_model(
        canvas,
        bundle.footprints.frames,
        polygons=bundle.footprints.polygons,
        normalization="sky_mean",
        weighting="noise_variance",
    )
    assert bundle.gauge_memory_model == expected
    assert bundle.gauge_memory_model["mode"] == zphot._GAUGE_MEMORY_MODE_R32C


# ---------------------------------------------------------------------------
# 9. L1/L2: never reuse a foreign descriptor; never accept a truncated polygon
#    artifact.
# ---------------------------------------------------------------------------

def test_stale_descriptor_is_rejected_and_recomputed(tmp_path, monkeypatch):
    descs = _small_corpus()
    canvas = zg.build_canvas(descs)
    good = _small_bundle(tmp_path, descs, canvas=canvas)
    canvas_px = int(canvas.width) * int(canvas.height)
    foreign = dict(good.gauge_memory_model)
    foreign["canvas_px"] = canvas_px + 1
    foreign["max_bbox_px"] = int(foreign["max_bbox_px"]) + 1

    calls = _spy_source_polygon(monkeypatch)
    rebuilt = _small_bundle(
        tmp_path,
        descs,
        canvas=canvas,
        footprints=good.footprints,
        model=foreign,
    )
    # The foreign descriptor is NOT reused; the current artifact is re-priced
    # without any new WCS projection.
    assert len(calls) == 0
    assert rebuilt.gauge_memory_model["canvas_px"] == canvas_px
    assert rebuilt.gauge_memory_model["max_bbox_px"] == good.gauge_memory_model["max_bbox_px"]
    assert rebuilt.gauge_memory_model != foreign


def test_polygon_frame_length_mismatch_fails_closed(tmp_path):
    descs = _small_corpus()
    canvas = zg.build_canvas(descs)
    fps = zal.project_footprints(descs, canvas)
    with pytest.raises(ValueError, match="polygons length"):
        zphot.gauge_memory_model(
            canvas,
            descs,
            polygons=fps.polygons[:-1],
        )
