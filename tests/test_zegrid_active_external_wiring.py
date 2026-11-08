"""ZM-ACTIVE-EXTERNAL-WIRING — ACTIVE admission of the external working set.

These regressions exercise the REAL product seam:

* the pre-fix ACTIVE CPU decision (RAM inmem+stream contexts) refuses an
  initial external bundle with ``R55_REFUSE_HOST_RAM``;
* the fixed decision selects and admits the external class, materializes the
  executable external tasks and keeps the frozen identity;
* a RAM bundle that passes the preliminary bundle planner but misses the ACTIVE
  joint margin can still select the external class at the SAME geometry;
* a real disk shortfall refuses BEFORE the gauge/stack with its own typed code;
* a frozen external admission resumes as external even when RAM would fit.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import resource_envelope as renv
from zemosaic.core.zegrid import run_resume as zres
from zemosaic.core.zegrid import scheduler as zsched

MIB = 2 ** 20
GIB = 2 ** 30


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


def _wide_corpus(n=4):
    """Four 40x40 frames spread over a ~4k x 4k mosaic canvas.

    Only GEOMETRY is materialised here (no pixels); it gives the assembly phase
    a real retained-core footprint so the ACTIVE joint margin can be exercised
    honestly without a large synthetic pixel run.
    """
    frames = []
    step_px = 9000.0
    for i in range(n):
        col, row = i % 2, i // 2
        from astropy.wcs import WCS

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [
            10.0 + col * step_px * 0.001,
            30.0 - row * step_px * 0.001,
        ]
        wcs.wcs.crpix = [20.0, 20.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (40, 40)
        fid = f"w{i}.fits"
        frames.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid), source_path=f"/x/{fid}",
                shape_hw=(40, 40),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="", instrument="",
            )
        )
    return frames


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
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
        lambda path: SimpleNamespace(
            total=int(free_bytes), used=0, free=int(free_bytes)
        ),
    )


def _patch_stream(monkeypatch, table):
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None: int(
            table.get(int(tile), max(table.values()))
        ),
    )


def _bundle(
    tmp_path, monkeypatch, *, stream_table=None, raw=2 * GIB,
    force_external=False, gauge_workers=None, cache_build_workers=None,
    pinned_layout=None, frozen_admission=None, free_bytes=100 * GIB,
    descs=None, halo_px=None,
):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62))
    _free(monkeypatch, free_bytes)
    if stream_table is not None:
        _patch_stream(monkeypatch, stream_table)
    descs = _corpus() if descs is None else descs
    canvas = zg.build_canvas(descs)
    budget = (4 * raw) // 5
    rank = None if pinned_layout is None else int(pinned_layout[0]) * int(
        pinned_layout[1]
    )
    return zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=raw,
        budget_bytes=budget,
        cpu_count=8,
        cache_root=str(tmp_path / "cache"),
        force_external=force_external,
        gauge_workers=gauge_workers,
        cache_build_workers=cache_build_workers,
        pinned_layout=pinned_layout,
        max_refinement_rank=rank,
        frozen_admission=frozen_admission,
        halo_px=halo_px,
    )


def _decide(bundle, tmp_path, frame_ids, descs=None, halo_px=None):
    return zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=zg.build_canvas(_corpus() if descs is None else descs),
        halo_px=(
            int(bundle.layout.get("halo_px", 8) or 8)
            if halo_px is None else int(halo_px)
        ),
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=frame_ids,
    )


def _frame_ids(descs):
    return tuple(d.frame_id.logical_path for d in descs)


# ---------------------------------------------------------------------------
# 1. Initial EXTERNAL bundle survives ACTIVE planning + materialization.
# ---------------------------------------------------------------------------

def test_external_bundle_survives_active_seam(tmp_path, monkeypatch):
    table = {int(t): 5 * GIB for t in zz.CPU_STREAMING_TILE_LADDER}
    ext_bundle = _bundle(
        tmp_path, monkeypatch, stream_table=table, force_external=True
    )
    assert ext_bundle.chosen_mode == "external"
    assert ext_bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    contract = ext_bundle.external_contract
    assert contract and int(contract["tile_size"]) > 0

    descs = _corpus()
    frame_ids = _frame_ids(descs)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)

    # CONTROL: the PRE-FIX seam is exactly the RAM inmem+stream decision. Feed
    # the same external bundle to it (class forced back to the historical RAM
    # class) and reproduce the confirmed refusal at the real ACTIVE seam.
    from dataclasses import replace as _replace

    ram_control = _replace(
        ext_bundle, stack_working_set=zz.WORKING_SET_RAM_TILE
    )
    with pytest.raises(zsched.PlanRefused) as control:
        _decide(ram_control, tmp_path, frame_ids)
    assert control.value.reason_code == renv.REASON_REFUSE_HOST_RAM

    # FIXED seam: the frozen external class is decided by ACTIVE, admitted and
    # materialized into real external tasks.
    decision = _decide(ext_bundle, tmp_path, frame_ids, descs=descs)
    assert decision.working_set_class == zz.WORKING_SET_EXTERNAL
    assert decision.mapping["cell_task_mode"] == "external"
    assert decision.mapping["units_in_flight"] == 1
    assert decision.mapping["prefetch_units"] == 0

    bundle, candidate, mapping, outcome, trail = zz._r55_active_admit(
        initial_bundle=ext_bundle, decision=decision
    )
    assert bundle.chosen_mode == "external"
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert candidate.working_set_class == zsched.WORKING_SET_CLASS_EXTERNAL_SPOOL
    assert mapping["cell_task_mode"] == "external"
    assert outcome.plan.aggregate_bytes <= outcome.plan.budget_bytes
    external_tasks = [t for t in bundle.cell_tasks if t[4] == "external"]
    assert external_tasks
    for task in external_tasks:
        assert len(task) == 9
        assert task[8] == zz._CELL_INPUT_WINDOWED
        info = task[7]
        assert info["class"] == zz.WORKING_SET_EXTERNAL
        assert int(info["tile_size"]) == int(contract["tile_size"])
        assert int(info["column_batch"]) == int(contract["column_batch"])
        assert Path(info["spool_root"]).name == "external-spool"
        assert int(info["disk_envelope_bytes"]) > 0

    # PREFLIGHT -> LAUNCH continuity: the launch revalidation starts from the
    # admitted external bundle and must keep the SAME class/tile/batch.
    (
        launch_bundle,
        launch_candidate,
        launch_mapping,
        launch_outcome,
        _launch_trail,
    ) = zz._r55_active_revalidate_launch(
        bundle=bundle, candidate=candidate, mapping=mapping, decision=decision,
    )
    assert launch_bundle.chosen_mode == "external"
    assert launch_bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert launch_mapping["cell_task_mode"] == "external"
    assert int(launch_mapping["cpu_streaming_tile"]) == int(
        contract["tile_size"]
    )
    assert launch_bundle.external_contract["column_batch"] == int(
        contract["column_batch"]
    )
    assert launch_outcome.plan.aggregate_bytes <= (
        launch_outcome.plan.budget_bytes
    )



# ---------------------------------------------------------------------------
# 2. RAM bundle passes the preliminary planner but misses the ACTIVE margin.
# ---------------------------------------------------------------------------

def test_ram_bundle_missing_active_margin_selects_external_same_identity(
    tmp_path, monkeypatch
):
    # A real halo-driven assembly (retained cores + halo bands) makes the RAM
    # stack miss the 15 % ACTIVE joint margin while the preliminary bundle
    # (no margin) still fits. The SAME geometry is then priced as external.
    halo = 64
    layout = (1, 1)
    table = {int(t): 1260 * MIB for t in zz.CPU_STREAMING_TILE_LADDER}
    descs = _wide_corpus()
    ram_bundle = _bundle(
        tmp_path, monkeypatch, stream_table=table,
        gauge_workers=1, cache_build_workers=1, descs=descs,
        pinned_layout=layout, halo_px=halo,
    )
    assert ram_bundle.chosen_mode == "stream"
    nx, ny = int(ram_bundle.layout["nx"]), int(ram_bundle.layout["ny"])
    assert (nx, ny) == layout
    ram_cells = tuple(
        sorted(c.cell_id for (_r, _c, c, _p, m) in ram_bundle.cell_ctxs
               if m.patch_ids)
    )
    frame_ids = _frame_ids(descs)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)

    with pytest.raises(zsched.PlanRefused) as refused:
        _decide(ram_bundle, tmp_path, frame_ids, descs=descs, halo_px=halo)
    assert refused.value.reason_code == renv.REASON_REFUSE_HOST_RAM

    # The external class is priced at the SAME pinned geometry/rank/halo.
    ext_bundle = _bundle(
        tmp_path, monkeypatch, stream_table=table, force_external=True,
        gauge_workers=1, cache_build_workers=1,
        pinned_layout=layout, descs=descs, halo_px=halo,
    )
    assert ext_bundle.chosen_mode == "external"
    assert tuple(int(v) for v in ext_bundle.envelope.layout) == layout
    ext_cells = tuple(
        sorted(c.cell_id for (_r, _c, c, _p, m) in ext_bundle.cell_ctxs
               if m.patch_ids)
    )
    assert ext_cells == ram_cells

    decision = _decide(
        ext_bundle, tmp_path, frame_ids, descs=descs, halo_px=halo
    )
    assert decision.working_set_class == zz.WORKING_SET_EXTERNAL
    assert decision.plan_set.identity.cell_ids == ram_cells
    assert decision.plan_set.identity.frame_ids == tuple(sorted(frame_ids))
    assert decision.plan_set.identity.layout == layout
    assert decision.plan_set.identity.halo_px == halo

    bundle, _candidate, _mapping, outcome, _trail = zz._r55_active_admit(
        initial_bundle=ext_bundle, decision=decision
    )
    assert bundle.chosen_mode == "external"
    assert outcome.plan.aggregate_bytes <= outcome.plan.budget_bytes
    assert all(t[4] in (None, "external") for t in bundle.cell_tasks)


# ---------------------------------------------------------------------------
# 4. Negative disk / observation refusals are typed BEFORE any gauge/stack.
# ---------------------------------------------------------------------------

def test_forced_external_disk_shortfall_refuses_typed(tmp_path, monkeypatch):
    with pytest.raises(zsched.PlanRefused) as exc:
        _bundle(
            tmp_path, monkeypatch, stream_table=None,
            force_external=True, free_bytes=1 * MIB,
        )
    assert exc.value.reason_code == renv.REASON_REFUSE_DISK
    assert "disk" in str(exc.value).lower()


def test_forced_external_observation_failure_is_not_hidden(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62))

    def _boom(_path):
        raise OSError("disk observation unavailable")

    monkeypatch.setattr(zz.shutil, "disk_usage", _boom)
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    with pytest.raises(zsched.PlanRefused) as exc:
        zz._build_execution_bundle(
            canvas, descs, None,
            raw_available_bytes=2 * GIB,
            budget_bytes=(4 * 2 * GIB) // 5,
            cpu_count=8,
            cache_root=str(tmp_path / "cache"),
            force_external=True,
        )
    assert exc.value.reason_code == renv.REASON_INTERNAL_ERROR
    assert "disk observation unavailable" in str(exc.value)


# ---------------------------------------------------------------------------
# Resume compatibility: a frozen external class is never re-derived as RAM.
# ---------------------------------------------------------------------------

def test_frozen_external_resumes_external_even_when_ram_would_fit(
    tmp_path, monkeypatch
):
    # RAM would fit comfortably on this fresh observation...
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62))
    _free(monkeypatch, 100 * GIB)
    table = {int(t): 60 * MIB for t in zz.CPU_STREAMING_TILE_LADDER}
    _patch_stream(monkeypatch, table)
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    frozen = zres.build_admission(
        backend="cpu",
        tile_px=16,
        parity_qualified=True,
        working_set={
            "class": zz.WORKING_SET_EXTERNAL,
            "tile_px": 16,
            "column_batch": 64,
        },
    )
    bundle = zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=2 * GIB,
        budget_bytes=(4 * 2 * GIB) // 5,
        cpu_count=8,
        cache_root=str(tmp_path / "cache"),
        frozen_admission=frozen,
    )
    # ... but the FROZEN class wins: never silently re-derived as RAM.
    assert bundle.chosen_mode == "external"
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.external_contract["tile_size"] == 16
    assert bundle.external_contract["column_batch"] == 64
    assert bundle.working_set_block["class"] == zz.WORKING_SET_EXTERNAL
    # The frozen tile/batch really travel in every executable external task.
    for task in bundle.cell_tasks:
        if task[4] is None:
            continue
        assert task[4] == "external"
        assert task[7]["tile_size"] == 16
        assert task[7]["column_batch"] == 64

    # The resumed bundle also survives the REAL ACTIVE decision/admission with
    # the same frozen tile and batch (no silent class change at launch).
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)
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
    resumed, _cand, mapping, outcome, _trail = zz._r55_active_admit(
        initial_bundle=bundle, decision=decision
    )
    assert resumed.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert mapping["cell_task_mode"] == "external"
    assert resumed.external_contract["tile_size"] == 16
    assert resumed.external_contract["column_batch"] == 64
    assert outcome.plan.aggregate_bytes <= outcome.plan.budget_bytes


# ---------------------------------------------------------------------------
# 3. Real hermetic ACTIVE run_zegrid_mode END-TO-END on the external class.
# ---------------------------------------------------------------------------

def _write_e2e_corpus(root, *, frames=6, hw=96, halo=32, step_px=48.0):
    from astropy.io import fits
    from astropy.wcs import WCS

    rng = np.random.default_rng(20261108)
    inp = root / "in"
    inp.mkdir()
    h = w = int(hw)
    step = float(step_px)
    names = []
    for i in range(frames):
        col, row = i % 2, i // 2
        scale = 0.001
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + col * step * scale, 30.0 - row * step * scale]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        data = (
            100.0
            + 20.0 * np.exp(
                -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 20.0 ** 2)
            )
            + 2.0 * i
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
    return inp


def _e2e_config():
    return SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_layout="2x2",
        zegrid_halo_px=32,
        cell_gauge_mode="per_cell",
    )


def _fits_array(path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data)


@pytest.mark.slow
def test_run_zegrid_mode_active_external_end_to_end(tmp_path, monkeypatch):
    inp = _write_e2e_corpus(tmp_path)
    ref_out = tmp_path / "ref"
    ext_out = tmp_path / "ext"

    # Reference: the SAME corpus/layout/halo/config on the bounded RAM stream
    # path (real host memory; no patches).
    zz.run_zegrid_mode(
        str(inp), str(ref_out), zconfig=_e2e_config(),
        storage_profile="bounded", cell_gauge_mode="per_cell",
    )
    ref_manifest = __import__("json").loads(
        (ref_out / "zegrid_manifest.json").read_text()
    )
    assert ref_manifest["stack_working_set"] == zz.WORKING_SET_RAM_TILE

    # External witness: RAM is impossible, a qualified GPU is "detected" and
    # DECLINES on VRAM, disk is ample. The real external executor is spied but
    # always CALLS THROUGH (never stubbed success).
    calls = []
    real_external = zz.zext.run_cell_streaming_external

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("cell_id"))
        return real_external(*args, **kwargs)

    def _decline_gpu(**_kwargs):
        raise zz._R55GpuRefused(
            zz.zgr.REASON_REFUSE_VRAM,
            "synthetic qualified VRAM decline (no physical GPU claim)",
        )

    with monkeypatch.context() as m:
        m.setattr(zz, "available_memory_bytes", lambda: 1536 * MIB)
        m.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 6 * GIB)
        m.setattr(
            zz, "_r55_resolve_gpu_probe",
            lambda *_a, **_k: (
                True,
                "synthetic",
                {
                    "available": True,
                    "device": "synthetic-gpu",
                    "cupy_version": "n/a",
                    "vram_total_bytes": 2 * GIB,
                    "vram_free_bytes": 2 * GIB,
                    "reason": None,
                },
                1 * GIB,
            ),
        )
        m.setattr(zz, "_r55_active_gpu_decide", _decline_gpu)
        m.setattr(zz.zext, "run_cell_streaming_external", _spy)
        zz.run_zegrid_mode(
            str(inp), str(ext_out), zconfig=_e2e_config(),
            storage_profile="bounded", use_gpu=True,
            cell_gauge_mode="per_cell",
        )

    ext_manifest = __import__("json").loads(
        (ext_out / "zegrid_manifest.json").read_text()
    )
    # The EXTERNAL class really governed the run.
    assert ext_manifest["storage"]["effective"] == "bounded"
    assert ext_manifest["stack_working_set"] == zz.WORKING_SET_EXTERNAL
    assert ext_manifest["working_set"]["class"] == zz.WORKING_SET_EXTERNAL
    assert ext_manifest["r55_scheduler"]["mode"] == "active"
    # Truthful requested/effective backend provenance: a qualified GPU was
    # requested and DECLINED on VRAM; the run really executed on CPU.
    gpu = ext_manifest["gpu"]
    assert gpu["requested"] is True
    assert gpu["effective"] is False
    assert gpu["actually_used"] is False
    assert gpu["final_backend"] == "cpu"
    ext_cells = [c for c in ext_manifest.get("cells", []) if c.get("storage")]
    assert ext_cells
    for cell in ext_cells:
        storage = cell["storage"]
        assert storage["working_set"] == zz.WORKING_SET_EXTERNAL
        assert storage["provider"] == "WindowedCanonicalProvider"
        assert storage["external"]["tile_size"] > 0
        assert storage["external"]["column_batch"] > 0
        # No ghost aligned cache: the source-backed path writes none.
        assert int(storage.get("aligned_cache_bytes_written", 0)) == 0
        assert storage["external_spool"]
    # The real external executor was called for the real cells (call-through).
    assert calls and len(calls) == len(ext_cells)

    # Same science/coverage as the RAM reference at the SAME layout/config.
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        np.testing.assert_array_equal(
            _fits_array(ref_out / name), _fits_array(ext_out / name),
            err_msg=name,
        )
    ref_cells = [
        (c["row"], c["col"], c["n_contributors"])
        for c in ref_manifest.get("cells", [])
    ]
    ext_cells_ids = [
        (c["row"], c["col"], c["n_contributors"])
        for c in ext_manifest.get("cells", [])
    ]
    assert ext_cells_ids == ref_cells


@pytest.mark.slow
def test_run_zegrid_mode_external_disk_shortfall_refuses_before_gauge(
    tmp_path, monkeypatch
):
    inp = _write_e2e_corpus(tmp_path)
    out = tmp_path / "out"
    import shutil as _shutil

    real_du = _shutil.disk_usage
    tiny = SimpleNamespace(total=1 * MIB, used=0, free=1 * MIB)

    def _disk_usage(path):
        if "external-spool" in str(path):
            return tiny
        return real_du(path)

    gauge_calls = []
    real_gauge = zz.zphot.compute_global_gauge

    def _gauge_spy(*args, **kwargs):
        gauge_calls.append(1)
        return real_gauge(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(zz.shutil, "disk_usage", _disk_usage)
        m.setattr(zz, "available_memory_bytes", lambda: 1536 * MIB)
        m.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 6 * GIB)
        m.setattr(zz.zphot, "compute_global_gauge", _gauge_spy)
        with pytest.raises(zsched.PlanRefused) as exc:
            zz.run_zegrid_mode(
                str(inp), str(out), zconfig=_e2e_config(),
                storage_profile="bounded", cell_gauge_mode="per_cell",
            )
    assert exc.value.reason_code == renv.REASON_REFUSE_DISK
    assert "disk" in str(exc.value).lower()
    assert gauge_calls == []
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()


# ---------------------------------------------------------------------------
# 2b. Refusal typing: reduction RAM floor vs disk vs malformed frozen identity.
# ---------------------------------------------------------------------------

def test_selector_reports_host_ram_when_minimum_batch_does_not_fit(
    tmp_path, monkeypatch
):
    """Junior probe: ample disk, budget too small for the minimum batch.

    The disk WAS observed and is not the cause; the classification must be a
    HOST-RAM reduction floor, never ``disk_envelope_unavailable``.
    """
    _free(monkeypatch, 77 * GIB)
    decision = zz._select_external_fallback(
        permitted_tiles=(16,),
        worst_patch_hw=(40, 40),
        max_n_frames=100,
        channels=3,
        budget_bytes=1,
        halo_px=8,
        cache_root=str(tmp_path / "cache"),
        storage_profile="bounded",
    )
    assert decision["refused"] is True
    assert decision["reason"] == "reduction_workspace_floor"
    assert decision["free_bytes"] == 77 * GIB
    assert decision["trials"]
    assert all(
        t["reason_code"] == "column_batch_floor" for t in decision["trials"]
    )
    assert "HOST-RAM floor" in decision["detail"]


def test_selector_distinguishes_frozen_batch_ram_from_invalid(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 77 * GIB)
    kwargs = dict(
        permitted_tiles=(16,),
        worst_patch_hw=(40, 40),
        max_n_frames=100,
        channels=3,
        budget_bytes=512 * MIB,
        halo_px=8,
        cache_root=str(tmp_path / "cache"),
        storage_profile="bounded",
    )
    ram = zz._select_external_fallback(
        **kwargs, frozen_tile=16, frozen_batch=10 ** 7
    )
    assert ram["refused"] is True
    assert ram["reason"] == "frozen_batch_ram"
    assert ram["free_bytes"] == 77 * GIB
    assert "HOST RAM" in ram["detail"]

    invalid = zz._select_external_fallback(
        **kwargs, frozen_tile=16, frozen_batch=0
    )
    assert invalid["refused"] is True
    assert invalid["reason"] == "frozen_batch_invalid"
    assert invalid["free_bytes"] == 77 * GIB


def test_builder_maps_external_ram_floor_to_host_ram(tmp_path, monkeypatch):
    monkeypatch.setattr(zz, "_r55_margin_raw_ceiling", lambda *a, **k: 1)
    with pytest.raises(zsched.PlanRefused) as exc:
        _bundle(tmp_path, monkeypatch, stream_table=None, force_external=True)
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM
    assert exc.value.reason_code != renv.REASON_REFUSE_DISK
    assert "reduction" in str(exc.value).lower()


def test_builder_maps_valid_frozen_batch_ram_to_host_ram(
    tmp_path, monkeypatch
):
    frozen = zres.build_admission(
        backend="cpu",
        tile_px=16,
        parity_qualified=True,
        working_set={
            "class": zz.WORKING_SET_EXTERNAL,
            "tile_px": 16,
            "column_batch": 10 ** 9,
        },
    )
    with pytest.raises(zsched.PlanRefused) as exc:
        _bundle(
            tmp_path, monkeypatch, stream_table=None,
            frozen_admission=frozen,
        )
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM
    assert exc.value.reason_code != renv.REASON_REFUSE_DISK
    assert "frozen" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# 2c. NATURAL public M33 transition: RAM bundle accepted -> REAL ACTIVE RAM
#     decision refuses on the joint margin -> the PRODUCTION catch rebuilds and
#     admits the external class at the same pin.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_zegrid_mode_ram_then_real_active_margin_selects_external(
    tmp_path, monkeypatch
):
    # ~3.1k x 3.1k canvas: a real assembly footprint from the pinned 2x2
    # geometry. The preliminary bundle fits B_plan exactly; the 15/100 ACTIVE
    # joint margin needs the assembly overlap and therefore refuses.
    inp = _write_e2e_corpus(
        tmp_path, frames=4, hw=96, step_px=3000.0
    )
    ref_out = tmp_path / "ref"
    ext_out = tmp_path / "ext"
    table = {int(t): 1298 * MIB for t in zz.CPU_STREAMING_TILE_LADDER}

    # Reference: SAME corpus/layout/halo/config on the RAM class under a larger
    # deterministic raw observation and the same deterministic stream facts.
    with monkeypatch.context() as m:
        # Real process baseline: the autouse fixture zeroes it for the small
        # unit witnesses, but the natural margin case must count the SAME
        # baseline the production envelope does.
        m.setattr(
            zpar, "_PER_WORKER_BASELINE_BYTES",
            zz.rcontract.PROCESS_BASELINE_BYTES,
        )
        m.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)
        _patch_stream(m, table)
        zz.run_zegrid_mode(
            str(inp), str(ref_out), zconfig=_e2e_config(),
            storage_profile="bounded", cell_gauge_mode="per_cell",
        )
    ref_manifest = __import__("json").loads(
        (ref_out / "zegrid_manifest.json").read_text()
    )
    assert ref_manifest["stack_working_set"] == zz.WORKING_SET_RAM_TILE
    assert (ref_out / "mosaic_grid.fits").exists()

    builds = []
    decisions = []
    calls = []
    real_build = zz._build_execution_bundle
    real_decide = zz._r55_active_decide
    real_external = zz.zext.run_cell_streaming_external

    def _build_spy(*args, **kwargs):
        bundle = real_build(*args, **kwargs)
        builds.append(
            {
                "force_external": bool(kwargs.get("force_external")),
                "mode": bundle.chosen_mode,
                "working_set": bundle.stack_working_set,
            }
        )
        return bundle

    def _decide_spy(*args, **kwargs):
        decision = real_decide(*args, **kwargs)
        decisions.append(
            {
                "mapping_mode": str(
                    dict(decision.mapping).get("cell_task_mode")
                ),
                "working_set_class": str(decision.working_set_class),
            }
        )
        return decision

    def _external_spy(*args, **kwargs):
        calls.append(kwargs.get("cell_id"))
        return real_external(*args, **kwargs)

    def _decline_gpu(**_kwargs):
        raise zz._R55GpuRefused(
            zz.zgr.REASON_REFUSE_VRAM,
            "synthetic qualified VRAM decline (no physical GPU claim)",
        )

    with monkeypatch.context() as m:
        # A SMALLER raw observation than the reference: the preliminary bundle
        # still fits, but the real 15/100 ACTIVE joint margin does not.
        m.setattr(
            zpar, "_PER_WORKER_BASELINE_BYTES",
            zz.rcontract.PROCESS_BASELINE_BYTES,
        )
        m.setattr(zz, "available_memory_bytes", lambda: 2 * GIB)
        _patch_stream(m, table)
        m.setattr(
            zz, "_r55_resolve_gpu_probe",
            lambda *_a, **_k: (
                True,
                "synthetic",
                {
                    "available": True,
                    "device": "synthetic-gpu",
                    "cupy_version": "n/a",
                    "vram_total_bytes": 2 * GIB,
                    "vram_free_bytes": 2 * GIB,
                    "reason": None,
                },
                1 * MIB,
            ),
        )
        m.setattr(zz, "_r55_active_gpu_decide", _decline_gpu)
        m.setattr(zz, "_build_execution_bundle", _build_spy)
        m.setattr(zz, "_r55_active_decide", _decide_spy)
        m.setattr(zz.zext, "run_cell_streaming_external", _external_spy)
        zz.run_zegrid_mode(
            str(inp), str(ext_out), zconfig=_e2e_config(),
            storage_profile="bounded", use_gpu=True,
            cell_gauge_mode="per_cell",
        )

    # The test itself never asked for external: the PRODUCTION catch owns the
    # transition. The bundle trail proves an initial RAM bundle was accepted and
    # a later rebuild forced the external class.
    assert any(
        b["force_external"] is False and b["mode"] == "stream"
        for b in builds
    ), builds
    assert any(
        b["force_external"] is True
        and b["working_set"] == zz.WORKING_SET_EXTERNAL
        for b in builds
    ), builds
    # A real ACTIVE decision ran and the admitted one is external.
    assert decisions
    assert decisions[-1]["mapping_mode"] == "external"
    assert decisions[-1]["working_set_class"] == zz.WORKING_SET_EXTERNAL

    ext_manifest = __import__("json").loads(
        (ext_out / "zegrid_manifest.json").read_text()
    )
    assert ext_manifest["storage"]["effective"] == "bounded"
    assert ext_manifest["r55_scheduler"]["mode"] == "active"
    assert ext_manifest["stack_working_set"] == zz.WORKING_SET_EXTERNAL
    assert ext_manifest["working_set"]["class"] == zz.WORKING_SET_EXTERNAL
    gpu = ext_manifest["gpu"]
    assert gpu["requested"] is True
    assert gpu["effective"] is False
    assert gpu["actually_used"] is False
    assert gpu["final_backend"] == "cpu"

    ext_cells = [c for c in ext_manifest.get("cells", []) if c.get("storage")]
    assert ext_cells
    for cell in ext_cells:
        storage = cell["storage"]
        assert storage["working_set"] == zz.WORKING_SET_EXTERNAL
        assert storage["provider"] == "WindowedCanonicalProvider"
        assert storage["external"]["tile_size"] > 0
        assert storage["external"]["column_batch"] > 0
        assert int(storage.get("aligned_cache_bytes_written", 0)) == 0
    assert calls and len(calls) == len(ext_cells)

    # SAME corpus/layout/config: stable identity and bit-identical science.
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        np.testing.assert_array_equal(
            _fits_array(ref_out / name), _fits_array(ext_out / name),
            err_msg=name,
        )
    ref_cells = [
        (c["row"], c["col"], c["n_contributors"])
        for c in ref_manifest.get("cells", [])
    ]
    ext_cells_ids = [
        (c["row"], c["col"], c["n_contributors"])
        for c in ext_manifest.get("cells", [])
    ]
    assert ext_cells_ids == ref_cells
    assert ext_manifest["layout"]["nx"] == ref_manifest["layout"]["nx"]
    assert ext_manifest["layout"]["ny"] == ref_manifest["layout"]["ny"]
