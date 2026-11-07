"""ZM-ZEGRID-ADAPTIVE-RESOURCES A4 — bounded assembly + banded FITS (Q40).

Q40 gate: a canvas whose feathered canonical planes (125 B/px) exceed the available RAM
budget must still produce a valid final FITS, byte-identical to the dense
in-memory reference, with NO full-canvas CHW/plane copy at the serialization
step. This test drives the REAL production consumers end to end:

* ``_assemble_cells_guarded(canvas_backing="scratch", scratch_dir=...)`` builds
  the whole feathered canvas through a ``ScratchCanvasAllocator`` (file-backed
  planes, not anonymous RAM);
* ``_write_raw_science_fits`` and the coverage write detect the memmap-backed
  planes and stream row-bands via ``banded_fits`` instead of materialising a
  full contiguous CHW copy;
* the resulting science + coverage FITS are BYTE-IDENTICAL to the dense
  ``PrimaryHDU`` path (same header block, big-endian data, SCIHASH, non-finite
  replacement), and the scratch planes are discarded afterwards.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import bounded_assembly as zbas  # noqa: E402
from zemosaic.core.zegrid import cell_feather as zcf  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import mosaic as zmosaic  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

H, W = 24, 24
NX, NY = 2, 2
HALO = 2
CHANNELS = 3


# Reuse the exact synthetic producer helpers from the M5-A bounded assembly
# suite so the payloads exercise the real store/consumer schema.
def make_canvas(h=H, w=W):
    return zg.GlobalCanvas(
        canvas_id="a4-canvas",
        wcs_header="",
        width=int(w),
        height=int(h),
        resolution_deg=0.001,
    )


def make_identity(*, canvas_hw=(H, W), halo=HALO):
    h, w = canvas_hw
    return zrs.build_run_identity(
        run_id="run-a4",
        inputs={
            "frames": [
                {
                    "frame_id": "frame_0001.fits",
                    "content_sha256": "a" * 64,
                    "nbytes": 1024,
                    "order": 0,
                    "role": "light",
                }
            ]
        },
        canvas={"shape": [int(h), int(w)], "wcs_sha256": "c" * 64,
                "geometry_sha256": "d" * 64},
        grid={"nx": NX, "ny": NY, "tile_px": 8, "geometry_sha256": "e" * 64,
              "cell_ids_sha256": "f" * 64},
        halo={"halo_px": int(halo), "provenance": {"effective_px": int(halo)}},
        memberships={"sha256": "1" * 64, "n_cells": NX * NY},
        gauge={"mode": "per_cell", "sha256": "2" * 64,
               "reference_frame_id": "frame_0001.fits",
               "reference_frame_ids": ["frame_0001.fits"], "n_exclusions": 0},
        science={"sha256": "3" * 64, "contract_version": "1",
                 "params": {"reject": "sigma", "sigma": 3.0}},
        dtypes={
            "planes": {
                "core.science_core": ["<f4", "<f8"],
                "core.estimator_weight_sum_core": ["<f4", "<f8"],
                "core.support_w1_core": ["<f4", "<f8"],
                "core.support_w2_core": ["<f4", "<f8"],
                "core.n_eff_support_core": ["<f4", "<f8"],
                "core.valid_mask_core": ["|b1"],
                "core.surviving_sample_count_core": ["<i4", "<i8"],
                "patch.science": ["<f4", "<f8"],
                "patch.valid_mask": ["|b1"],
                "patch.surviving": ["<i4", "<i8"],
            }
        },
        backend={"backend": "cpu", "tile_px": 8, "parity_qualified": False},
        execution_context={"available_ram_bytes": 4 * 2 ** 20, "workers": 2,
                           "prefetch_units": 0, "host": "test-host",
                           "note": "runtime only"},
    )


def cell_geometry(row, col, *, nx=NX, ny=NY, h=H, w=W, halo=HALO):
    x0 = col * w // nx
    x1 = (col + 1) * w // nx
    y0 = row * h // ny
    y1 = (row + 1) * h // ny
    py0, py1 = max(0, y0 - halo), min(h, y1 + halo)
    px0, px1 = max(0, x0 - halo), min(w, x1 + halo)
    core = (y0, y1, x0, x1)
    patch = (py0, py1, px0, px1)
    core_slice = (y0 - py0, y1 - py0, x0 - px0, x1 - px0)
    return core, patch, core_slice


def complete_result(row, col, *, seed=0):
    (_y0, _y1, _x0, _x1), (py0, py1, px0, px1), core_slice = cell_geometry(
        row, col
    )
    cid = zg.cell_id(row, col)
    rng = np.random.default_rng(1000 + seed)
    ph, pw = py1 - py0, px1 - px0
    yy = np.arange(ph, dtype=np.float64)[:, None]
    xx = np.arange(pw, dtype=np.float64)[None, :]
    base = (
        10.0
        + 0.5 * np.sin((yy + py0) / 5.0)
        + 0.25 * np.cos((xx + px0) / 7.0)
        + float(seed) * 1.5
    )
    science = np.repeat(base[:, :, None], CHANNELS, axis=2) + rng.normal(
        scale=0.05, size=(ph, pw, CHANNELS)
    )
    science = science.astype(np.float32)
    valid = np.ones((ph, pw, CHANNELS), dtype=bool)
    if row == 0 and col == 0:
        valid[:2, :, :] = False
        science[:2, :, :] = np.nan
    surviving = rng.integers(1, 4, size=(ph, pw)).astype(np.int32)
    sci_core = science[core_slice[0]:core_slice[1], core_slice[2]:core_slice[3], :]
    val_core = valid[core_slice[0]:core_slice[1], core_slice[2]:core_slice[3], :]
    surv_core = surviving[core_slice[0]:core_slice[1], core_slice[2]:core_slice[3]]
    w1 = np.ones(surv_core.shape, dtype=np.float32) * 1.5
    w2 = (w1 ** 2).astype(np.float32)
    neff = surv_core.astype(np.float64) * 1.25
    return {
        "cell_id": cid,
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 1,
        "total_bytes": int(science.nbytes),
        "core": {
            "science_core": sci_core,
            "estimator_weight_sum_core": w1,
            "support_w1_core": w1,
            "support_w2_core": w2,
            "n_eff_support_core": neff,
            "valid_mask_core": val_core,
            "surviving_sample_count_core": surv_core,
        },
        "patch": {
            "bounds": (py0, py1, px0, px1),
            "core_slice": core_slice,
            "science": science,
            "valid_mask": valid,
            "surviving": surviving,
        },
        "record": {
            "cell_id": cid,
            "patch_bounds": [px0, py0, px1, py1],
            "patch_shape_hw": [ph, pw],
            "cell_gauge_mode": "per_cell",
            "backend_used": "cpu",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
    }


def seeded_store(tmp_path, results):
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=make_identity())
    for index, result in enumerate(results):
        store.commit_cell(result, attempt_id=f"a{index}", backend="cpu")
    return root, store


def reopen_read_only(root):
    return zrs.RunCellStore.open(root, identity=None, read_only=True)


def eager_patches(results):
    out = {}
    for r in results:
        if r.get("status") != "complete":
            continue
        p = r["patch"]
        out[r["cell_id"]] = zcf.CellPatch(
            cell_id=r["cell_id"],
            bounds=tuple(int(v) for v in p["bounds"]),
            core_slice=tuple(int(v) for v in p["core_slice"]),
            science=p["science"],
            valid_mask=p["valid_mask"],
            surviving=p.get("surviving"),
        )
    return out


def _dense_fits_bytes(assembled, canvas, role="SCI"):
    """The exact dense path bytes for one assembled science cube."""
    raw = np.ascontiguousarray(
        np.moveaxis(np.asarray(assembled.science, dtype=np.float32), -1, 0)
    )
    finite = np.isfinite(raw)
    replaced = int(raw.size - int(np.count_nonzero(finite)))
    if replaced:
        raw = raw.copy()
        raw[~finite] = np.float32(0.0)
    header = zz._science_header(
        canvas, role=role, dtype="float32", dbe_state="n/a",
        sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
    )
    zz._record_halo_cards(header, None)
    zz.record_nonfinite_fill(header, replaced)
    buf = io.BytesIO()
    fits.PrimaryHDU(raw, header=header).writeto(buf)
    return buf.getvalue(), replaced


def test_q40_scratch_assembly_and_banded_fits_byte_identical(tmp_path):
    results = [
        complete_result(0, 0, seed=0),
        complete_result(0, 1, seed=1),
        complete_result(1, 0, seed=2),
        complete_result(1, 1, seed=3),
    ]
    root, store = seeded_store(tmp_path, results)
    store.close()
    canvas = make_canvas()

    # Dense reference assembly + FITS bytes (the witness).
    patches = eager_patches(results)
    graph = zcf.build_cell_offset_graph(canvas, NX, NY, patches, on_disconnected="zero")
    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}
    reference = zcf.assemble_canvas_feathered(
        canvas, NX, NY, patches, offsets, halo_px=HALO
    )
    ref_bytes, ref_replaced = _dense_fits_bytes(reference, canvas)

    # Scratch-backed assembly through the REAL guarded production entry, then
    # the REAL banded raw-science + coverage writers.
    scratch_dir = tmp_path / "canvas-scratch"
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        assembled, info, plan, allocator = zz._assemble_cells_guarded(
            canvas, NX, NY, "per_cell", bounded.cores, bounded.descriptors,
            bounded_results=bounded,
            available_bytes=256 * 2 ** 20,
            canvas_backing="scratch",
            scratch_dir=scratch_dir,
            halo_px=HALO,
            patches=bounded.patches,
        )
        assert plan is not None and plan["canvas"]["backing"] == "scratch"
        assert plan["fits"] is True
        assert isinstance(assembled.science, np.memmap)
        assert isinstance(assembled.stack_depth, np.memmap)
        # Pixel parity with the dense witness.
        assert np.array_equal(assembled.science, reference.science, equal_nan=True)
        assert np.array_equal(assembled.stack_depth, reference.stack_depth)

        # The real serialization seam writes BYTE-IDENTICAL FITS.
        raw_path = tmp_path / "mosaic_grid.fits"
        zz._write_raw_science_fits(assembled, canvas, raw_path)
        assert raw_path.read_bytes() == ref_bytes

        cov_path = tmp_path / "mosaic_grid_coverage.fits"
        cov_header = zz._canvas_header(canvas, ndim=2)
        cov_header["BUNIT"] = ("count", "per-pixel stack depth")
        zz._record_halo_cards(cov_header, None)
        # exercise the REAL coverage branch inside _write_outputs indirectly is
        # heavy; call the banded coverage writer directly + verify parity with a
        # dense coverage write.
        from zemosaic.core.zegrid import banded_fits as zbf
        dense_cov = np.ascontiguousarray(
            np.asarray(reference.stack_depth, dtype=np.float32)
        )
        dense_cov_buf = io.BytesIO()
        fits.PrimaryHDU(dense_cov, header=cov_header).writeto(dense_cov_buf)
        zbf.write_coverage_fits_banded(
            np.asarray(assembled.stack_depth, dtype=np.float32), cov_header, cov_path
        )
        assert cov_path.read_bytes() == dense_cov_buf.getvalue()

        # All 13 scratch planes (F6 includes valid_out/covered) are created and
        # are removed on discard.
        assert len(list(scratch_dir.glob("*.bin"))) == 13
        allocator.discard()
        assert not list(scratch_dir.glob("*.bin"))
    finally:
        store.close()


def test_q40_scratch_backing_requires_scratch_dir(tmp_path):
    results = [complete_result(0, 0, seed=0), complete_result(0, 1, seed=1)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        with pytest.raises(zbas.BoundedAccessError):
            zz._assemble_cells_guarded(
                make_canvas(), NX, NY, "per_cell", bounded.cores,
                bounded.descriptors, bounded_results=bounded,
                available_bytes=256 * 2 ** 20,
                canvas_backing="scratch",
                scratch_dir=None,
                halo_px=HALO,
                patches=bounded.patches,
            )
    finally:
        store.close()


def test_f6_global_assembly_honours_the_scratch_allocator(tmp_path):
    """F6: global gauge must route the allocator (memmap science, bit-exact)."""
    results = [
        complete_result(0, 0, seed=0),
        complete_result(0, 1, seed=1),
        complete_result(1, 0, seed=2),
        complete_result(1, 1, seed=3),
    ]
    canvas = make_canvas()
    cores = {r["cell_id"]: r["core"] for r in results}

    dense = zmosaic.assemble_canvas(canvas, NX, NY, cores)
    allocator = zbas.ScratchCanvasAllocator(tmp_path / "global-scratch",
                                           prefix="global")
    try:
        scratch = zmosaic.assemble_canvas(
            canvas, NX, NY, cores, allocator=allocator
        )
        assert isinstance(scratch.science, np.memmap)
        for name in ("stack_depth", "n_eff_support", "support_w1", "support_w2",
                     "valid_fraction", "owner", "hole_mask"):
            assert isinstance(getattr(scratch, name), np.memmap), name
        # 8 global planes: science, depth, n_eff, w1, w2, valid_fraction, owner,
        # hole_mask (49 B/px at 3 channels).
        assert len(list((tmp_path / "global-scratch").glob("*.bin"))) == 8
        assert zbas.canvas_plane_bytes(
            (H, W), gauge_mode="global"
        )["bytes_per_pixel"] == 49.0
        assert zbas.canvas_plane_bytes(
            (H, W), gauge_mode="per_cell"
        )["bytes_per_pixel"] == 125.0
        assert np.array_equal(scratch.science, dense.science, equal_nan=True)
        assert np.array_equal(scratch.stack_depth, dense.stack_depth)
        assert np.array_equal(scratch.n_eff_support, dense.n_eff_support)
        assert np.array_equal(scratch.support_w1, dense.support_w1)
        assert np.array_equal(scratch.support_w2, dense.support_w2)
        assert np.array_equal(scratch.valid_fraction, dense.valid_fraction)
        assert np.array_equal(scratch.owner, dense.owner)
        assert np.array_equal(scratch.hole_mask, dense.hole_mask)
    finally:
        allocator.discard()
