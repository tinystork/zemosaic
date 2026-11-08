"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M3 — automatic working-set wiring.

Bounded product-level acceptance for M3:

* the AUTO path selects ``stream_external`` only when no in-memory plan and no
  RAM streaming tile fits, under the bounded storage profile;
* the disk envelope is priced from the real geometry and a shortfall refuses
  with the dry-plan ``disk_envelope_unavailable`` class before any work;
* the budget-derived column batch and the tile travel in the task payload;
* RAM-feasible bundles keep their historical decision fields (additive-only
  working-set block);
* a frozen working-set class is never silently re-derived on resume;
* the manifest/log block is strict JSON and names its unknown disk terms;
* the GUI/worker call carries no working-set/tile override.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import external_stream as zext
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import run_resume as zres
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20


class _FakeModel:
    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(*_a, **_k):
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


def _make_tan(i, h=40, w=40, scale=0.001, ra_step=0.004):
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


def _corpus(n=4):
    return [_make_tan(i) for i in range(n)]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from zemosaic.core.zegrid import parallel as zpar

    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)


def _free(monkeypatch, free_bytes):
    monkeypatch.setattr(
        zz.shutil, "disk_usage",
        lambda path: SimpleNamespace(total=int(free_bytes), used=0, free=int(free_bytes)),
    )


def _bundle(tmp_path, *, stream_table, raw=2 * 2 ** 30):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    monkeypatch_table = stream_table

    def _fake_est(n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, storage_profile=None):
        return int(monkeypatch_table.get(int(tile), max(monkeypatch_table.values())))

    # Patch through the module global used by the bundle planner.
    zz._estimate_streaming_bytes = _fake_est
    budget = (4 * raw) // 5
    return zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=raw, budget_bytes=budget,
        cpu_count=8, cache_root=str(tmp_path / "cache"),
    )


def test_external_selected_when_all_ram_tiles_refuse_and_disk_fits(tmp_path, monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    _free(monkeypatch, 100 * 2 ** 30)
    table = {int(t): 5 * 2 ** 30 for t in zz.CPU_STREAMING_TILE_LADDER}
    real = zz._estimate_streaming_bytes
    try:
        bundle = _bundle(tmp_path, stream_table=table)
    finally:
        zz._estimate_streaming_bytes = real
    assert bundle.chosen_mode == "external"
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.cpu_streaming_tile in zz.CPU_STREAMING_TILE_LADDER
    block = bundle.working_set_block
    assert block["class"] == zz.WORKING_SET_EXTERNAL
    assert block["tile_size"] == bundle.cpu_streaming_tile
    assert block["column_batch"] >= 1
    assert block["disk"]["spool_peak_bytes_worst_tile"] > 0
    assert block["disk"]["free_bytes_observed"] == 100 * 2 ** 30
    assert block["disk"]["unknown_bytes"] is None
    assert block["disk"]["unknown_terms"]
    # Every non-empty task carries the executable contract as the 8th element.
    external_tasks = [t for t in bundle.cell_tasks if t[4] == "external"]
    assert external_tasks
    for task in external_tasks:
        assert len(task) == 9
        assert task[8] == zz._CELL_INPUT_WINDOWED
        info = task[7]
        assert info["class"] == zz.WORKING_SET_EXTERNAL
        assert int(info["tile_size"]) == bundle.cpu_streaming_tile
        assert int(info["column_batch"]) == block["column_batch"]
        assert Path(info["spool_root"]).name == "external-spool"
        assert int(info["disk_envelope_bytes"]) > 0
    json.dumps(block)


def test_external_selected_under_gpu_detection_when_bounded(
    tmp_path, monkeypatch
):
    """F4: a detected GPU must not block the external working set.

    No GPU tile/plan is feasible, every RAM streaming tile is refused, and the
    bounded profile is active: the layout must be KEPT for the external class
    (no LayoutInfeasible) and ``_select_external_fallback`` must be reached.
    The resulting external pseudo-plan is CPU-only.
    """
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    _free(monkeypatch, 100 * 2 ** 30)
    # GPU detected, but the VRAM ladder yields no candidate -> no device plan.
    monkeypatch.setattr(zz, "_r55_gpu_tile_ladder", lambda *a, **k: ((), None))
    table = {int(t): 5 * 2 ** 30 for t in zz.CPU_STREAMING_TILE_LADDER}
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    real = zz._estimate_streaming_bytes
    zz._estimate_streaming_bytes = (
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: int(table.get(int(tile), max(table.values())))
    )
    try:
        bundle = zz._build_execution_bundle(
            canvas, descs, None,
            raw_available_bytes=2 * 2 ** 30,
            budget_bytes=(4 * 2 * 2 ** 30) // 5,
            gpu_used=True,
            vram_budget_bytes=1 * 2 ** 30,
            cpu_count=8, cache_root=str(tmp_path / "cache"),
        )
    finally:
        zz._estimate_streaming_bytes = real
    assert bundle.chosen_mode == "external"
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.working_set_block["class"] == zz.WORKING_SET_EXTERNAL
    # The external pseudo-plan is CPU-only: no device execution is claimed.
    assert bundle.working_set_block["gpu"]["used"] is False
    external_tasks = [t for t in bundle.cell_tasks if t[4] == "external"]
    assert external_tasks
    for task in external_tasks:
        assert len(task) == 9
        assert task[8] == zz._CELL_INPUT_WINDOWED


def test_external_refused_when_disk_envelope_does_not_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    _free(monkeypatch, 1 * MIB)  # far below any spool peak
    decision = zz._select_external_fallback(
        permitted_tiles=zz.CPU_STREAMING_TILE_LADDER,
        worst_patch_hw=(40, 40),
        max_n_frames=4,
        channels=3,
        budget_bytes=512 * MIB,
        halo_px=9,
        cache_root=str(tmp_path),
        storage_profile="bounded",
        observation_domain=zz.rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE,
    )
    assert decision["refused"] is True
    assert decision["reason"] == "disk_envelope_unavailable"
    assert decision["trials"]
    assert decision["free_bytes"] == 1 * MIB


def test_external_admission_does_not_bill_phantom_aligned_cache(
    tmp_path, monkeypatch
):
    """F2 audit probe: N20k patch 256^2, tile 8, halo 8, 1 GiB volume.

    The bounded external path is SOURCE-BACKED and never writes the aligned
    N x patch cache (~17 GiB here), so that cache must not be billed to the
    disk envelope. Admission must succeed and the published envelope must be
    spool + exact resident staging + result + cleanup margin.
    """
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    gib = 2 ** 30
    _free(monkeypatch, gib)
    n, hw = 20_000, (256, 256)
    decision = zz._select_external_fallback(
        permitted_tiles=(8,),
        worst_patch_hw=hw,
        max_n_frames=n,
        channels=3,
        budget_bytes=1 * gib,
        halo_px=8,
        cache_root=str(tmp_path),
        storage_profile="bounded",
        observation_domain=zz.rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE,
    )
    assert decision["refused"] is False, decision
    assert decision["tile"] == 8
    aligned_cache = zz._r55_cache_unit_bytes(hw[0] * hw[1], n, channels=3)
    # The phantom term the old envelope billed really is > 10 GiB.
    assert aligned_cache > 10 * gib
    # ... and the admitted envelope is now far below both it and the volume.
    assert decision["disk_envelope_bytes"] < gib
    assert decision["disk_envelope_bytes"] < aligned_cache
    pricing = decision["pricing"]
    assert decision["resident_unit_bytes"] == zz._r55_prefetch_resident_bytes(
        storage_profile="bounded",
        per_unit_cache_bytes=aligned_cache,
        worst_patch_px=hw[0] * hw[1],
        channels=3,
    )
    assert decision["disk_envelope_bytes"] == (
        pricing["disk"]["spool_peak_bytes_worst_tile"]
        + decision["resident_unit_bytes"]
        + decision["result_unit_bytes"]
        + pricing["disk"]["cleanup_margin_bytes"]
    )
    # A genuine shortfall now refuses for the REAL disk size only.
    _free(monkeypatch, 100 * MIB)
    short = zz._select_external_fallback(
        permitted_tiles=(8,),
        worst_patch_hw=hw,
        max_n_frames=n,
        channels=3,
        budget_bytes=1 * gib,
        halo_px=8,
        cache_root=str(tmp_path),
        storage_profile="bounded",
        observation_domain=zz.rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE,
    )
    assert short["refused"] is True
    assert short["reason"] == "disk_envelope_unavailable"
    assert short["best_envelope_bytes"] < aligned_cache
    assert all(
        t.get("reason_code") != "aligned_cache" for t in short["trials"]
    )


def test_dense_profile_never_switches_silently(tmp_path):
    decision = zz._select_external_fallback(
        permitted_tiles=zz.CPU_STREAMING_TILE_LADDER,
        worst_patch_hw=(40, 40),
        max_n_frames=4,
        channels=3,
        budget_bytes=512 * MIB,
        halo_px=9,
        cache_root=str(tmp_path),
        storage_profile="dense",
    )
    assert decision["refused"] is True
    assert decision["reason"] == "external_requires_bounded"
    assert "AUTO" in decision["detail"]


def test_ram_feasible_bundle_is_unchanged_and_additive_only(tmp_path, monkeypatch):
    # This is a synthetic tile-selection test (64 MiB), not a physical RAM
    # witness. Match the fixture's other zeroed overheads after A1 joint pricing.
    monkeypatch.setattr(zz.rcontract, "PROCESS_BASELINE_BYTES", 0)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    table = {
        128: 60 * MIB, 112: 60 * MIB, 96: 60 * MIB, 80: 60 * MIB,
        64: 60 * MIB, 48: 60 * MIB, 32: 60 * MIB, 24: 60 * MIB,
        20: 60 * MIB, 16: 8 * MIB, 12: 8 * MIB, 10: 8 * MIB, 8: 8 * MIB,
    }
    real = zz._estimate_streaming_bytes
    try:
        bundle = _bundle(tmp_path, stream_table=table, raw=64 * MIB)
    finally:
        zz._estimate_streaming_bytes = real
    assert bundle.chosen_mode == "stream"
    assert bundle.stack_working_set == zz.WORKING_SET_RAM_TILE
    assert bundle.cpu_streaming_tile == 16
    assert all(len(t) == 6 for t in bundle.cell_tasks)  # dense profile
    block = bundle.working_set_block
    assert block["class"] == zz.WORKING_SET_RAM_TILE
    assert block["disk"] is None
    assert block["column_batch"] is None
    assert block["tile_size"] == 16
    assert block["refusal"] is None


def test_frozen_working_set_class_is_not_re_derived(tmp_path, monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    _free(monkeypatch, 100 * 2 ** 30)
    table = {int(t): 5 * 2 ** 30 for t in zz.CPU_STREAMING_TILE_LADDER}
    frozen = zres.build_admission(
        backend="cpu",
        tile_px=16,
        parity_qualified=True,
        working_set={"class": "ram_tile", "tile_px": 16, "column_batch": None},
    )
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    real = zz._estimate_streaming_bytes
    zz._estimate_streaming_bytes = lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, storage_profile=None: int(
        table.get(int(tile), max(table.values()))
    )
    try:
        with pytest.raises(zpar_mode_infeasible()) as exc:
            zz._build_execution_bundle(
                canvas, descs, None,
                raw_available_bytes=2 * 2 ** 30,
                budget_bytes=(4 * 2 * 2 ** 30) // 5,
                cpu_count=8, cache_root=str(tmp_path / "cache"),
                frozen_admission=frozen,
            )
    finally:
        zz._estimate_streaming_bytes = real
    assert "frozen working-set class" in str(exc.value)


def zpar_mode_infeasible():
    from zemosaic.core.zegrid import parallel as zpar

    return zpar.ModeInfeasible


def test_legacy_frozen_identity_refuses_external(tmp_path, monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))
    _free(monkeypatch, 100 * 2 ** 30)
    table = {int(t): 5 * 2 ** 30 for t in zz.CPU_STREAMING_TILE_LADDER}
    frozen = zres.build_admission(
        backend="cpu", tile_px=16, parity_qualified=True
    )  # no working_set -> legacy
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    real = zz._estimate_streaming_bytes
    zz._estimate_streaming_bytes = lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, storage_profile=None: int(
        table.get(int(tile), max(table.values()))
    )
    try:
        with pytest.raises(zpar_mode_infeasible()) as exc:
            zz._build_execution_bundle(
                canvas, descs, None,
                raw_available_bytes=2 * 2 ** 30,
                budget_bytes=(4 * 2 * 2 ** 30) // 5,
                cpu_count=8, cache_root=str(tmp_path / "cache"),
                frozen_admission=frozen,
            )
    finally:
        zz._estimate_streaming_bytes = real
    assert "legacy run identity" in str(exc.value)


def test_budget_derived_column_batch_matches_helper():
    batch = zext.largest_admissible_column_batch(
        budget_bytes=512 * MIB, n_frames=100, interior_width=20, channels=3
    )
    assert batch % 60 == 0
    assert batch >= 60


def test_task_payload_is_parseable_for_external(tmp_path):
    empty_mem = SimpleNamespace(patch_ids=[])
    task = (
        SimpleNamespace(cell_id="c1", row=0, col=0),
        SimpleNamespace(
            patch=SimpleNamespace(x0=0, y0=0, x1=4, y1=4, width=4, height=4),
            patch_shape_hw=(4, 4),
        ),
        empty_mem,
        str(tmp_path / "cache"),
        "external",
        123,
        "bounded",
        {
            "class": zz.WORKING_SET_EXTERNAL, "tile_size": 16,
            "column_batch": 64, "spool_root": str(tmp_path),
            "disk_envelope_bytes": 1,
        },
    )
    prepared = zz._prepare_stack_cell(task)
    assert "empty_result" in prepared


def test_disk_shortfall_at_cell_boundary_is_typed(tmp_path, monkeypatch):
    _free(monkeypatch, 10)
    with pytest.raises(zext.ExternalDiskShortfall):
        zz._run_cell_external(
            cache_dir=tmp_path,
            patch=None,
            config=None,
            progress_callback=None,
            tile_size=16,
            column_batch=32,
            spool_root=str(tmp_path),
            required_free_bytes=1000,
        )


def test_admission_working_set_roundtrip_and_validation():
    ws = {"class": "external_spool", "tile_px": 16, "column_batch": 64}
    admission = zres.build_admission(
        backend="cpu", tile_px=16, parity_qualified=True, working_set=ws
    )
    assert admission["working_set"] == ws
    legacy = zres.build_admission(
        backend="cpu", tile_px=16, parity_qualified=True
    )
    assert "working_set" not in legacy
    with pytest.raises(zres.ResumeStateError):
        zres.build_admission(
            backend="cpu", tile_px=16, parity_qualified=True,
            working_set={"class": "nope", "tile_px": 16, "column_batch": 4},
        )
    with pytest.raises(zres.ResumeStateError):
        zres.build_admission(
            backend="cpu", tile_px=16, parity_qualified=True,
            working_set={"class": "external_spool", "tile_px": 16},
        )


def test_gui_worker_call_has_no_working_set_or_tile_override():
    from zemosaic import zemosaic_worker as zw

    src = inspect.getsource(zw)
    start = src.index("run_zegrid_mode(\n")
    end = src.index("finally:", start)
    call = src[start:end]
    for forbidden in ("cpu_tile", "tile_size", "working_set", "column_batch",
                      "spool_root", "storage_profile"):
        assert forbidden not in call
    assert "zegrid_scheduler_mode" in call
