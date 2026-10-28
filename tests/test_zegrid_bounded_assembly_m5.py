"""ZM-ZEGRID-RESILIENT-RUN-RESUME M5-A — bounded assembly access.

These tests exercise the REAL consumers (``cell_feather`` / ``mosaic`` through
``zemosaic_zegrid_mode._assemble_cells``) against the bounded store path:

* exact parity with the dense in-memory reference (global and per_cell,
  multicell with halos, one empty cell, one disconnected component);
* two successive assemblies after close/reopen: identical pixels, identical
  offsets, identical commit/object digests (no double offset, no restack);
* a real bounded read: no eager load, hard residency cap, read-only mmaps;
* explicit refusal on corruption/I/O/capacity (never a fake success);
* an optional scratch-backed canvas that is bit-identical to ``np.zeros``.

Everything is synthetic and small: the product memory floors are untouched.
Windows is NOT_RUN on this host (documented in the mission report).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import bounded_assembly as zbas  # noqa: E402
from zemosaic.core.zegrid import cell_feather as zcf  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import mosaic as zmosaic  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

H, W = 24, 24
NX, NY = 2, 2
HALO = 2
CHANNELS = 3

DTYPE_CONTRACT = {
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
}


# ---------------------------------------------------------------------------
# Synthetic geometry / payloads (real producer schema)
# ---------------------------------------------------------------------------


def make_canvas(h=H, w=W):
    return zg.GlobalCanvas(
        canvas_id="m5a-canvas",
        wcs_header="",
        width=int(w),
        height=int(h),
        resolution_deg=0.001,
    )


def make_identity(*, nx=NX, ny=NY, canvas_hw=(H, W), halo=HALO, gauge_mode="per_cell"):
    h, w = canvas_hw
    return zrs.build_run_identity(
        run_id="run-m5a",
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
        canvas={
            "shape": [int(h), int(w)],
            "wcs_sha256": "c" * 64,
            "geometry_sha256": "d" * 64,
        },
        grid={
            "nx": int(nx),
            "ny": int(ny),
            "tile_px": 8,
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": "f" * 64,
        },
        halo={"halo_px": int(halo), "provenance": {"effective_px": int(halo)}},
        memberships={"sha256": "1" * 64, "n_cells": int(nx * ny)},
        gauge={
            "mode": gauge_mode,
            "sha256": "2" * 64,
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_ids": ["frame_0001.fits"],
            "n_exclusions": 0,
        },
        science={
            "sha256": "3" * 64,
            "contract_version": "1",
            "params": {"reject": "sigma", "sigma": 3.0},
        },
        dtypes=DTYPE_CONTRACT,
        backend={"backend": "cpu", "tile_px": 8, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": 4 * 2 ** 20,
            "workers": 2,
            "prefetch_units": 0,
            "host": "test-host",
            "note": "runtime only",
        },
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


def complete_result(row, col, *, seed=0, nx=NX, ny=NY, h=H, w=W, halo=HALO):
    (y0, y1, x0, x1), (py0, py1, px0, px1), core_slice = cell_geometry(
        row, col, nx=nx, ny=ny, h=h, w=w, halo=halo
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
    # A deterministic invalid band so shared-valid/feather logic is exercised.
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
        "cache_build_s": 0.01 * (seed + 1),
        "stack_s": 0.02 * (seed + 1),
        "peak_rss_kib": 1024 * (seed + 1),
        "cleanup_failed": False,
        "cache_dir": f"/tmp/m5a-cache-{cid}",
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


def empty_result(row, col, *, nx=NX, ny=NY, h=H, w=W, halo=HALO):
    _, (py0, py1, px0, px1), _ = cell_geometry(
        row, col, nx=nx, ny=ny, h=h, w=w, halo=halo
    )
    cid = zg.cell_id(row, col)
    return {
        "cell_id": cid,
        "status": "empty",
        "backend_used": "cpu",
        "n_frames": 0,
        "total_bytes": 0,
        "record": {
            "cell_id": cid,
            "status": "empty",
            "cell_gauge_mode": "per_cell",
            "n_contributors": 0,
            "patch_bounds": [px0, py0, px1, py1],
            "patch_shape_hw": [py1 - py0, px1 - px0],
            "reference_frame_id": None,
            "excluded": [],
        },
    }


def seeded_store(tmp_path, results, *, identity=None, name="run"):
    root = tmp_path / name
    store = zrs.RunCellStore.open(root, identity=identity or make_identity())
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


def eager_cores(results):
    out = {}
    for r in results:
        if r.get("status") == "complete":
            out[r["cell_id"]] = r["core"]
    return out


def digest_tree(root: Path) -> dict:
    out = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


def assert_same_canvas(a, b):
    for name in (
        "science",
        "stack_depth",
        "n_eff_support",
        "support_w1",
        "support_w2",
        "valid_fraction",
        "owner",
        "hole_mask",
    ):
        left = getattr(a, name)
        right = getattr(b, name)
        assert left.dtype == right.dtype, name
        assert left.shape == right.shape, name
        if left.dtype.kind == "f":
            assert np.array_equal(left, right, equal_nan=True), name
        else:
            assert np.array_equal(left, right), name
    assert list(a.complete_cells) == list(b.complete_cells)
    assert list(a.incomplete_cells) == list(b.incomplete_cells)


# ---------------------------------------------------------------------------
# 1. Real bounded read through the real consumer (per_cell, halos, empty,
#    disconnected component)
# ---------------------------------------------------------------------------


def _three_cell_run(tmp_path):
    """nx=3, ny=1 with the MIDDLE cell empty -> two disconnected components."""
    nx, ny = 3, 1
    identity = make_identity(nx=nx, ny=ny, canvas_hw=(H, W), halo=HALO)
    results = [
        complete_result(0, 0, seed=0, nx=nx, ny=ny),
        empty_result(0, 1, nx=nx, ny=ny),
        complete_result(0, 2, seed=2, nx=nx, ny=ny),
    ]
    root, store = seeded_store(tmp_path, results, identity=identity)
    store.close()
    return nx, ny, results, root


def test_m5a_per_cell_assembly_matches_dense_reference(tmp_path):
    nx, ny, results, root = _three_cell_run(tmp_path)
    canvas = make_canvas()

    reference_patches = eager_patches(results)
    ref_graph = zcf.build_cell_offset_graph(
        canvas, nx, ny, reference_patches, on_disconnected="zero"
    )
    ref_offsets = {
        cid: ref_graph.offsets[i] for i, cid in enumerate(ref_graph.cell_ids)
    }
    reference = zcf.assemble_canvas_feathered(
        canvas, nx, ny, reference_patches, ref_offsets, halo_px=HALO
    )
    assert ref_graph.disconnected, "the middle empty cell must split the graph"

    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded_results = access.results(
            [r["cell_id"] for r in results]
        )
        # NOTHING is read by constructing the views.
        assert access.stats()["loads"] == 0
        assert access.stats()["live_mapped_bytes"] == 0

        assembled, info = zz._assemble_cells(
            canvas,
            nx,
            ny,
            "per_cell",
            bounded_results.cores,
            bounded_results.descriptors,
            halo_px=HALO,
            patches=bounded_results.patches,
        )

        stats = access.stats()
        assert stats["loads"] > 0
        assert stats["peak_live_cells"] <= 2
        assert stats["live_mapped_bytes"] <= stats["peak_live_mapped_bytes"]
        totals = {"store_total_bytes": int(store.report()["counts"]["stored_bytes"])}
        zbas.assert_bounded_usage(
            access.session,
            examined_cells=len(results),
            totals=totals,
        )

        assert_same_canvas(assembled, reference)
        assert info["anchor"] == ref_graph.anchor
        assert info["disconnected"] == list(ref_graph.disconnected)
        assert info["policy"] == ref_graph.policy
        assert info["edges_used"] == ref_graph.edges_used
        for cid in ref_offsets:
            assert np.array_equal(
                np.asarray(info["offsets"][cid], dtype=np.float64),
                np.asarray(ref_offsets[cid], dtype=np.float64),
            ), cid
    finally:
        store.close()


def test_m5a_global_assembly_matches_dense_reference(tmp_path):
    results = [
        complete_result(0, 0, seed=0),
        complete_result(0, 1, seed=1),
        empty_result(1, 0),
    ]
    root, store = seeded_store(tmp_path, results)
    store.close()
    canvas = make_canvas()

    reference = zmosaic.assemble_canvas(
        canvas, NX, NY, eager_cores(results)
    )

    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded_results = access.results([r["cell_id"] for r in results])
        assembled, info = zz._assemble_cells(
            canvas,
            NX,
            NY,
            "global",
            bounded_results.cores,
            bounded_results.descriptors,
        )
        assert info is None
        assert_same_canvas(assembled, reference)
        assert assembled.incomplete_cells == [
            zg.cell_id(1, 0),
            zg.cell_id(1, 1),
        ]
        assert access.stats()["peak_live_cells"] <= 2
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. Two successive assemblies after close/reopen: identical pixels/offsets and
#    unchanged commit+object digests (no double offset, no restack)
# ---------------------------------------------------------------------------


def test_m5a_replay_after_reopen_is_identical_and_commits_untouched(tmp_path):
    nx, ny, results, root = _three_cell_run(tmp_path)
    canvas = make_canvas()
    before = digest_tree(root)

    reference_patches = eager_patches(results)
    ref_graph = zcf.build_cell_offset_graph(
        canvas, nx, ny, reference_patches, on_disconnected="zero"
    )
    ref_offsets = {
        cid: ref_graph.offsets[i] for i, cid in enumerate(ref_graph.cell_ids)
    }
    reference = zcf.assemble_canvas_feathered(
        canvas, nx, ny, reference_patches, ref_offsets, halo_px=HALO
    )

    outs = []
    for _attempt in range(2):
        store = reopen_read_only(root)
        try:
            access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
            bounded_results = access.results([r["cell_id"] for r in results])
            assembled, info = zz._assemble_cells(
                canvas,
                nx,
                ny,
                "per_cell",
                bounded_results.cores,
                bounded_results.descriptors,
                halo_px=HALO,
                patches=bounded_results.patches,
            )
            outs.append((assembled, info, access.stats()))
        finally:
            store.close()
        after = digest_tree(root)
        assert after == before, "assembly must never mutate a durable commit"

    assert_same_canvas(outs[0][0], reference)
    assert_same_canvas(outs[1][0], reference)
    assert_same_canvas(outs[0][0], outs[1][0])
    assert outs[0][1]["offsets"] == outs[1][1]["offsets"]
    assert outs[0][1]["anchor"] == outs[1][1]["anchor"]
    # The arithmetic is applied ONCE to the immutable pre-offset planes: the
    # bounded replay equals the dense reference (a double offset would differ).
    for cid, off in ref_offsets.items():
        assert np.array_equal(
            np.asarray(outs[0][1]["offsets"][cid], dtype=np.float64),
            np.asarray(off, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# 3. Read-only mmap lifetime + hard residency cap
# ---------------------------------------------------------------------------


def test_m5a_payloads_are_read_only_mmaps_with_bounded_residency(tmp_path):
    results = [
        complete_result(0, 0, seed=0),
        complete_result(0, 1, seed=1),
        complete_result(1, 0, seed=2),
        complete_result(1, 1, seed=3),
    ]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        view = access.payloads([r["cell_id"] for r in results])
        assert access.stats()["loads"] == 0  # construction is not eager

        first = view["r0000c0000"]
        for name, array in first["core"].items():
            assert isinstance(array, np.memmap), name
            assert array.flags.writeable is False
        assert first["patch"]["science"].flags.writeable is False
        with pytest.raises(ValueError):
            first["patch"]["science"][0, 0, 0] = 1.0

        # Drop every local reference: loop variables keep a plane alive too.
        del array, name, first
        for cid in view:
            assert "science" in view[cid]["patch"]
        stats = access.stats()
        assert stats["loads"] == 4
        assert stats["peak_live_cells"] == 2
        # 4 loads with cap 2: exactly two documented LRU evictions.
        assert stats["evictions"] == 2
        assert stats["resident_cells"] <= 2
        assert stats["anonymous_copy_bytes"] == 0

        # Lifetime: a view handed out stays readable after close.
        holder = view["r0000c0000"]
        held = holder["patch"]["science"]
        del holder
        access.close()
        assert access.stats()["closed"] is True
        assert np.isfinite(np.asarray(held, dtype=np.float64)).any()
        with pytest.raises(zbas.BoundedLifetimeError):
            view["r0000c0000"]
    finally:
        store.close()


def test_m5a_descriptors_never_read_planes(tmp_path):
    results = [complete_result(0, 0, seed=0), empty_result(0, 1)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        descriptors = access.descriptors([r["cell_id"] for r in results])
        for cid in descriptors:
            meta = descriptors[cid]
            assert "core" not in meta
            assert meta["_store"]["access"] == "descriptor_only"
            assert meta["_store"]["arrays_loaded"] is False
        assert access.stats()["loads"] == 0
        assert access.stats()["verifications"] == 0

        # Metadata the coordinator aggregates is present without any array.
        complete = descriptors["r0000c0000"]
        for key in ("status", "record", "n_frames", "total_bytes", "cache_build_s"):
            assert key in complete, key
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 4. Explicit refusal: corruption / missing object / capacity
# ---------------------------------------------------------------------------


def _corrupt_object_of(root: Path, cell_id: str) -> Path:
    """Flip one byte of a REAL plane object belonging to ``cell_id``."""
    store = reopen_read_only(root)
    try:
        record = store.commit_record(cell_id)
        rel = record.manifest["planes"][0]["path"]
    finally:
        store.close()
    path = Path(root) / rel
    data = bytearray(path.read_bytes())
    assert data
    data[0] = (data[0] + 1) % 256
    try:
        os.chmod(path, 0o644)
    except OSError:  # pragma: no cover - owner can usually chmod
        pass
    path.write_bytes(bytes(data))
    return path


def test_m5a_corrupted_object_is_refused_and_sources_preserved(tmp_path):
    results = [complete_result(0, 0, seed=0), complete_result(0, 1, seed=1)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    victim = _corrupt_object_of(root, "r0000c0000")
    before = digest_tree(root)

    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        with pytest.raises(zrs.StoreIntegrityError):
            access.payloads([r["cell_id"] for r in results])["r0000c0000"]
        # No payload was handed out and the store is intact.
        assert access.stats()["loads"] == 0
        assert access.stats()["live_mapped_bytes"] == 0
    finally:
        store.close()
    assert victim.exists()
    assert digest_tree(root) == before


def test_m5a_missing_object_is_refused(tmp_path):
    results = [complete_result(0, 0, seed=0)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        record = store.commit_record("r0000c0000")
        rel = record.manifest["planes"][0]["path"]
    finally:
        store.close()
    (Path(root) / rel).unlink()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        with pytest.raises(zrs.StoreIntegrityError):
            access.payloads(["r0000c0000"])["r0000c0000"]
    finally:
        store.close()


def test_m5a_capacity_refusal_is_explicit_and_priced(tmp_path):
    plan = zbas.assembly_plan(
        (H, W),
        cell_plane_bytes=[1024, 2048, 4096],
        max_resident_cells=2,
        largest_patch_hw=(16, 16),
    )
    assert plan["cells"]["store_total_bytes"] == 7168
    assert plan["cells"]["resident_mapped_bytes"] == 6144
    assert plan["canvas"]["total_bytes"] == zbas.canvas_plane_bytes((H, W))["total_bytes"]
    assert plan["fits"] is None  # no budget supplied -> no fake verdict

    with pytest.raises(zbas.AssemblyCapacityError) as exc:
        zbas.assembly_plan(
            (H, W),
            cell_plane_bytes=[1024, 2048, 4096],
            max_resident_cells=2,
            available_bytes=1024,
        )
    assert exc.value.required_bytes > exc.value.available_bytes
    assert exc.value.plan["required_bytes"] == exc.value.required_bytes

    # A budget that covers the canvas AND the conservative transient margin
    # (previous-cell overlap + the 1 MiB streaming-hash chunk) fits exactly.
    enough = (
        plan["cells"]["resident_mapped_bytes"]
        + plan["canvas"]["total_bytes"]
        + plan["per_cell_transient_bytes"]
        + plan["graph_bytes"]
    )
    fitting = zbas.assembly_plan(
        (H, W),
        cell_plane_bytes=[1024, 2048, 4096],
        max_resident_cells=2,
        largest_patch_hw=(16, 16),
        available_bytes=enough,
    )
    assert fitting["fits"] is True
    assert fitting["required_bytes"] == enough
    assert fitting["transient"]["hash_chunk_bytes"] == zbas.DEFAULT_HASH_CHUNK_BYTES


# ---------------------------------------------------------------------------
# 5. Optional scratch-backed canvas: bit-identical to the anonymous path
# ---------------------------------------------------------------------------


def test_m5a_scratch_canvas_is_bit_identical(tmp_path):
    results = [
        complete_result(0, 0, seed=0),
        complete_result(0, 1, seed=1),
        complete_result(1, 0, seed=2),
        complete_result(1, 1, seed=3),
    ]
    canvas = make_canvas()
    patches = eager_patches(results)
    graph = zcf.build_cell_offset_graph(
        canvas, NX, NY, patches, on_disconnected="zero"
    )
    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}

    reference = zcf.assemble_canvas_feathered(
        canvas, NX, NY, patches, offsets, halo_px=HALO
    )

    allocator = zbas.ScratchCanvasAllocator(tmp_path / "scratch", prefix="canvas")
    try:
        scratch = zcf.assemble_canvas_feathered(
            canvas, NX, NY, patches, offsets, halo_px=HALO, allocator=allocator
        )
        assert_same_canvas(scratch, reference)
        assert allocator.bytes_allocated() > 0
        assert len(list((tmp_path / "scratch").glob("*.bin"))) == 7
        assert np.array_equal(
            scratch.science, reference.science, equal_nan=True
        )
    finally:
        removed = allocator.discard()
        assert removed == 7
        assert not list((tmp_path / "scratch").glob("*.bin"))


# ---------------------------------------------------------------------------
# 6. Wiring: the mode-level helper hands back a bounded, non-eager view
# ---------------------------------------------------------------------------


def test_m5a_wiring_helper_returns_bounded_results(tmp_path):
    results = [complete_result(0, 0, seed=0), complete_result(0, 1, seed=1)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    state = {}
    try:
        bounded = zz._bounded_store_results(
            store, [r["cell_id"] for r in results], store_state=state
        )
        assert isinstance(bounded, zbas.StoreBackedResults)
        assert state["assembly_access"] is bounded.assembly
        assert bounded.session.stats()["loads"] == 0
        assert [d["cell_id"] for d in bounded] == [r["cell_id"] for r in results]
        assert len(bounded) == 2
        # Metadata iteration must not read a single plane either.
        assert sum(d["n_frames"] for d in bounded) == 2
        assert bounded.session.stats()["loads"] == 0
        bounded.close()
    finally:
        store.close()


def test_m5a_guard_detects_eager_residency(tmp_path):
    results = [complete_result(0, 0, seed=0), complete_result(0, 1, seed=1)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        view = access.payloads([r["cell_id"] for r in results])
        view["r0000c0000"]
        view["r0000c0001"]
        store_total = access.session.cell_bytes("r0000c0000") + access.session.cell_bytes(
            "r0000c0001"
        )
        # Every payload mapped at once: the guard reads the REAL live counters.
        with pytest.raises(zbas.BoundedAccessError):
            zbas.assert_bounded_usage(
                access.session,
                examined_cells=2,
                totals={"store_total_bytes": store_total},
            )
        ok = zbas.assert_bounded_usage(
            access.session,
            examined_cells=2,
            allow_full_residency=True,
        )
        assert ok["peak_live_cells"] == 2
        assert ok["peak_live_mapped_bytes"] == store_total
        assert ok["peak_live_mappings"] == access.stats()["peak_live_mappings"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 7. F1 — production preflight refuses BEFORE consumer/load/allocation
# ---------------------------------------------------------------------------


def _all_complete_2x2():
    return [
        complete_result(0, 0, seed=0),
        complete_result(0, 1, seed=1),
        complete_result(1, 0, seed=2),
        complete_result(1, 1, seed=3),
    ]


def test_m5a_preflight_refuses_before_consumer_load_or_allocation(
    tmp_path, monkeypatch
):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    before = digest_tree(root)
    canvas = make_canvas()

    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        calls = {"assemble": 0, "canvas": 0}
        real_assemble = zz._assemble_cells

        def _spy(*args, **kwargs):
            calls["assemble"] += 1
            return real_assemble(*args, **kwargs)

        monkeypatch.setattr(zz, "_assemble_cells", _spy)
        with pytest.raises(zbas.AssemblyCapacityError) as exc:
            zz._assemble_cells_guarded(
                canvas,
                NX,
                NY,
                "per_cell",
                bounded.cores,
                bounded.descriptors,
                bounded_results=bounded,
                available_bytes=4096,
                halo_px=HALO,
                patches=bounded.patches,
            )
        assert calls["assemble"] == 0, "consumer ran despite the refusal"
        assert exc.value.required_bytes > exc.value.available_bytes
        stats = access.stats()
        assert stats["loads"] == 0
        assert stats["verifications"] == 0
        assert stats["live_mappings"] == 0
        assert stats["peak_live_cells"] == 0
        assert stats["fingerprints_computed"] == 0
        # No canvas allocation either: the refusal is pre-allocation.
        assert "canvas" in exc.value.plan
    finally:
        store.close()
    assert digest_tree(root) == before, "a refusal must never touch the commits"


def test_m5a_preflight_adequate_budget_prices_metadata_only(tmp_path):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    canvas = make_canvas()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        plan = zz._bounded_assembly_preflight(
            canvas, bounded, available_bytes=256 * 2 ** 20
        )
        assert plan["preflight"]["metadata_only"] is True
        assert plan["preflight"]["mappings_opened"] == 0
        assert plan["preflight"]["verifications_after"] == 0
        assert plan["fits"] is True
        assert plan["canvas"]["total_bytes"] == zbas.canvas_plane_bytes(
            (H, W)
        )["total_bytes"]
        assert plan["cells"]["resident_mapped_bytes"] > 0
        # The session's live envelope is armed from the plan (enforcement, not
        # posthoc): the byte cap equals the resident mapping bound.
        assert access.session.max_live_bytes == plan["cells"]["resident_mapped_bytes"]
        assert access.stats()["loads"] == 0
    finally:
        store.close()


def test_m5a_guarded_assembly_matches_dense_reference(tmp_path):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    canvas = make_canvas()

    patches = eager_patches(results)
    graph = zcf.build_cell_offset_graph(
        canvas, NX, NY, patches, on_disconnected="zero"
    )
    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}
    reference = zcf.assemble_canvas_feathered(
        canvas, NX, NY, patches, offsets, halo_px=HALO
    )

    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        assembled, info, plan = zz._assemble_cells_guarded(
            canvas,
            NX,
            NY,
            "per_cell",
            bounded.cores,
            bounded.descriptors,
            bounded_results=bounded,
            available_bytes=256 * 2 ** 20,
            halo_px=HALO,
            patches=bounded.patches,
        )
        assert plan is not None and plan["fits"] is True
        assert_same_canvas(assembled, reference)
        stats = access.stats()
        assert stats["peak_live_cells"] <= 2
        assert stats["peak_live_mapped_bytes"] <= access.session.max_live_bytes
        zbas.assert_bounded_usage(
            access.session,
            examined_cells=len(results),
            totals={
                "store_total_bytes": int(
                    store.report()["counts"]["stored_bytes"]
                )
            },
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 8. F2 — REAL live-mapping instrumentation through the real consumer
# ---------------------------------------------------------------------------


def _install_memmap_probe(monkeypatch):
    import weakref

    import numpy as _np

    probe = {
        "live": 0,
        "bytes": 0,
        "peak": 0,
        "peak_bytes": 0,
        "refs": [],
        "opens": 0,
    }
    real = _np.memmap

    def _factory(*args, **kwargs):
        array = real(*args, **kwargs)
        nbytes = int(array.nbytes)
        probe["opens"] += 1
        probe["live"] += 1
        probe["bytes"] += nbytes
        probe["peak"] = max(probe["peak"], probe["live"])
        probe["peak_bytes"] = max(probe["peak_bytes"], probe["bytes"])

        def _cb(_ref, nbytes=nbytes):
            probe["live"] -= 1
            probe["bytes"] -= nbytes

        probe["refs"].append(weakref.ref(array, _cb))
        return array

    monkeypatch.setattr(zbas.np, "memmap", _factory)
    return probe


@pytest.mark.parametrize("gauge_mode", ["global", "per_cell"])
@pytest.mark.parametrize("grid", [(2, 2), (3, 3)])
def test_m5a_real_consumer_live_peak_matches_stats(
    tmp_path, monkeypatch, gauge_mode, grid
):
    nx, ny = grid
    identity = make_identity(nx=nx, ny=ny, canvas_hw=(H, W), halo=HALO)
    results = [
        complete_result(r, c, seed=r * nx + c, nx=nx, ny=ny)
        for r in range(ny)
        for c in range(nx)
    ]
    root, store = seeded_store(tmp_path, results, identity=identity)
    store.close()
    canvas = make_canvas()
    probe = _install_memmap_probe(monkeypatch)

    store = reopen_read_only(root)
    access = None
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        assembled, info, plan = zz._assemble_cells_guarded(
            canvas,
            nx,
            ny,
            gauge_mode,
            bounded.cores,
            bounded.descriptors,
            bounded_results=bounded,
            available_bytes=256 * 2 ** 20,
            halo_px=HALO,
            patches=(bounded.patches if gauge_mode == "per_cell" else None),
        )
        stats = access.stats()
        # The declared peak IS the measured peak (weakref instrumented).
        assert stats["peak_live_mappings"] == probe["peak"]
        assert stats["peak_live_mapped_bytes"] == probe["peak_bytes"]
        assert stats["peak_live_cells"] <= 2
        assert stats["peak_live_mapped_bytes"] <= access.session.max_live_bytes
        assert assembled.coverage_pixels > 0
        assert plan["preflight"]["payloads_loaded_after"] == 0
    finally:
        if access is not None:
            access.close()
        store.close()
    import gc

    gc.collect()
    # No mapping leaks out of the assembly.
    assert probe["live"] == 0


# ---------------------------------------------------------------------------
# 9. F2 — external retention: eager list fails safely, arrays stay valid
# ---------------------------------------------------------------------------


def test_m5a_eager_retention_fails_safely_without_invalidating_arrays(tmp_path):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        view = access.payloads([r["cell_id"] for r in results])
        first = view["r0000c0000"]
        second = view["r0000c0001"]
        # Both are retained OUTSIDE the session: a third mapping must be refused
        # BEFORE it is opened (never force-closing the retained arrays).
        with pytest.raises(zbas.BoundedLifetimeError):
            view["r0001c0000"]
        stats = access.stats()
        assert stats["live_cells"] == 2
        assert stats["live_mapped_bytes"] <= access.session.cell_bytes(
            "r0000c0000"
        ) + access.session.cell_bytes("r0000c0001")
        # The retained arrays are STILL valid and readable.
        assert np.isfinite(
            np.asarray(first["patch"]["science"], dtype=np.float64)
        ).any()
        assert np.isfinite(
            np.asarray(second["patch"]["science"], dtype=np.float64)
        ).any()
        # A full eager materialisation of the set also fails safely.
        with pytest.raises(zbas.BoundedLifetimeError):
            list(view.values())
        del first, second
        import gc

        gc.collect()
        assert view["r0001c0000"]["cell_id"] == "r0001c0000"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 10. F3 — streaming fingerprint: canonical equivalence, no full copy, once
# ---------------------------------------------------------------------------


def test_m5a_streaming_fingerprint_matches_canonical():
    payload = complete_result(0, 0, seed=7)
    canonical = zrr.fingerprint_result(payload)["payload_sha256"]
    for chunk in (7, 1024, zbas.DEFAULT_HASH_CHUNK_BYTES):
        assert (
            zbas.payload_sha256_streaming(payload, chunk_bytes=chunk) == canonical
        ), chunk

    # Representative dtypes/shapes: float64 science, int64 surviving, 2-D core.
    rng = np.random.default_rng(3)
    payload64 = {
        "cell_id": "r0000c0000",
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 2,
        "total_bytes": 99,
        "core": {
            "science_core": rng.normal(size=(4, 5, 3)).astype(np.float64),
            "estimator_weight_sum_core": rng.normal(size=(4, 5)).astype(np.float64),
            "support_w1_core": rng.normal(size=(4, 5)).astype(np.float64),
            "support_w2_core": rng.normal(size=(4, 5)).astype(np.float64),
            "n_eff_support_core": rng.normal(size=(4, 5)).astype(np.float64),
            "valid_mask_core": rng.integers(0, 2, size=(4, 5, 3)).astype(bool),
            "surviving_sample_count_core": rng.integers(0, 5, size=(4, 5)).astype(np.int64),
        },
        "record": {
            "cell_id": "r0000c0000",
            "patch_bounds": [0, 0, 5, 4],
            "patch_shape_hw": [4, 5],
            "cell_gauge_mode": "global",
            "backend_used": "cpu",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "global",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
    }
    assert zbas.payload_sha256_streaming(payload64) == zrr.fingerprint_result(
        payload64
    )["payload_sha256"]

    empty = empty_result(0, 1)
    assert zbas.payload_sha256_streaming(empty) == zrr.fingerprint_result(empty)[
        "payload_sha256"
    ]


def test_m5a_streaming_fingerprint_does_not_copy_the_plane():
    import tracemalloc

    plane = np.zeros((1024, 1024), dtype=np.float32)  # 4 MiB
    plane[::7, ::11] = 3.0
    nbytes = int(plane.nbytes)
    assert zbas.array_sha256_streaming(plane, chunk_bytes=64 * 1024) == (
        zrr._array_sha256(plane)
    )
    tracemalloc.start()
    try:
        zbas.array_sha256_streaming(plane, chunk_bytes=64 * 1024)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < nbytes // 4, (peak, nbytes)


def test_m5a_verification_and_fingerprint_cached_per_identity(tmp_path):
    results = [complete_result(0, 0, seed=0), complete_result(0, 1, seed=1)]
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        view = access.payloads([r["cell_id"] for r in results])
        view["r0000c0000"]
        view["r0000c0001"]
        assert access.stats()["verifications"] == 2
        assert access.stats()["fingerprints_computed"] == 2
        access.session.release("r0000c0000")
        view["r0000c0000"]  # reload after eviction
        stats = access.stats()
        assert stats["loads"] == 3
        assert stats["verifications"] == 2  # NOT re-verified
        assert stats["fingerprints_computed"] == 2  # NOT re-hashed
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 11. F2 — slices/reopen lifetime and exception cleanup
# ---------------------------------------------------------------------------


def test_m5a_slices_keep_mappings_live_and_are_accounted(tmp_path):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        view = access.payloads([r["cell_id"] for r in results])
        held = view["r0000c0000"]
        slice_view = held["patch"]["science"][4:5, 4:5, :]
        second = view["r0000c0001"]
        # The slice keeps the first cell's mapping alive outside the LRU.
        access.session.release("r0000c0000")
        assert access.stats()["live_cells"] == 2
        with pytest.raises(zbas.BoundedLifetimeError):
            view["r0001c0000"]
        assert np.isfinite(np.asarray(slice_view, dtype=np.float64)).any()
        del slice_view, held, second
        import gc

        gc.collect()
        assert access.stats()["live_cells"] == 0
        assert view["r0001c0000"]["cell_id"] == "r0001c0000"
    finally:
        store.close()


def test_m5a_exception_cleanup_releases_mappings(tmp_path, monkeypatch):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    canvas = make_canvas()
    probe = _install_memmap_probe(monkeypatch)
    store = reopen_read_only(root)
    access = None
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])

        def _boom(*args, **kwargs):
            raise RuntimeError("consumer failure")

        monkeypatch.setattr(zcf, "feather_weight_field", _boom)
        with pytest.raises(RuntimeError, match="consumer failure"):
            zz._assemble_cells_guarded(
                canvas,
                NX,
                NY,
                "per_cell",
                bounded.cores,
                bounded.descriptors,
                bounded_results=bounded,
                available_bytes=256 * 2 ** 20,
                halo_px=HALO,
                patches=bounded.patches,
            )
    finally:
        if access is not None:
            access.close()
        store.close()
    import gc

    gc.collect()
    assert probe["live"] == 0



# ---------------------------------------------------------------------------
# 12. REWORK-2 G1 — the dense offset SOLVE is priced from metadata
# ---------------------------------------------------------------------------


def test_m5a_dense_solve_model_matches_real_solver_dimensions():
    model = zbas.dense_solve_bytes(400, 760, 3)
    # Independent dimensions: A is (E, N-1) float64.
    assert model["cells"] == 400
    assert model["edges"] == 760
    assert model["free"] == 399
    assert model["matrices"]["A_bytes"] == 8 * 760 * 399 == 2425920
    assert model["matrices"]["ATA_bytes"] == 8 * 399 * 399
    # Conservative: A + weighted A + previous-channel A + ATA + solve + margin.
    assert model["total_bytes"] > 3 * model["matrices"]["A_bytes"]
    assert model["total_bytes"] > 6 * 1024 * 1024

    # Global assembly runs no dense solve: exactly zero graph cost.
    plan_global = zbas.assembly_plan(
        (200, 200),
        cell_plane_bytes=[3000] * 400,
        max_resident_cells=2,
        largest_patch_hw=(12, 12),
        gauge_mode="global",
        n_cells_complete=400,
        n_edges=760,
    )
    assert plan_global["graph_bytes"] == 0
    assert plan_global["graph"]["mode"] == "global_no_solve"

    plan_per_cell = zbas.assembly_plan(
        (200, 200),
        cell_plane_bytes=[3000] * 400,
        max_resident_cells=2,
        largest_patch_hw=(12, 12),
        gauge_mode="per_cell",
        n_cells_complete=400,
        n_edges=760,
    )
    assert plan_per_cell["graph_bytes"] == model["total_bytes"]
    assert plan_per_cell["required_bytes"] > plan_global["required_bytes"]


def test_m5a_dense_solve_boundaries():
    # One cell: no free node, no dense matrix.
    one = zbas.dense_solve_bytes(1, 0, 3)
    assert one["free"] == 0
    assert one["matrices"]["A_bytes"] == 0
    assert one["total_bytes"] < 4096
    # Empty set.
    zero = zbas.dense_solve_bytes(0, 0, 3)
    assert zero["total_bytes"] == 0
    # Empty/disconnected identities: edge count is the real restricted grid.
    assert zbas.grid_edge_upper_bound(2, 2, []) == 0
    assert zbas.grid_edge_upper_bound(2, 2, ["r0000c0000"]) == 0
    assert (
        zbas.grid_edge_upper_bound(
            2, 2, ["r0000c0000", "r0001c0000", "r0001c0001"]
        )
        == 2
    )


def test_m5a_underbudget_refused_before_solver_with_large_metadata(
    tmp_path, monkeypatch
):
    """N=400 / E=760 metadata; the guarded entry refuses BEFORE any solve."""
    nx, ny = 20, 20
    h, w = 200, 200
    identity = make_identity(nx=nx, ny=ny, canvas_hw=(h, w), halo=1)
    results = [
        complete_result(r, c, seed=r * nx + c, nx=nx, ny=ny, h=h, w=w, halo=1)
        for r in range(ny)
        for c in range(nx)
    ]
    root, store = seeded_store(tmp_path, results, identity=identity)
    store.close()
    canvas = make_canvas(h=h, w=w)
    before = digest_tree(root)

    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        calls = {"assemble": 0, "solver": 0}
        real_assemble = zz._assemble_cells

        def _spy(*args, **kwargs):
            calls["assemble"] += 1
            return real_assemble(*args, **kwargs)

        real_solve = zcf.solve_additive_offsets

        def _solve_spy(*args, **kwargs):
            calls["solver"] += 1
            return real_solve(*args, **kwargs)

        monkeypatch.setattr(zz, "_assemble_cells", _spy)
        monkeypatch.setattr(zcf, "solve_additive_offsets", _solve_spy)

        with pytest.raises(zbas.AssemblyCapacityError) as exc:
            zz._assemble_cells_guarded(
                canvas,
                nx,
                ny,
                "per_cell",
                bounded.cores,
                bounded.descriptors,
                bounded_results=bounded,
                available_bytes=2 * 1024 * 1024,
                halo_px=1,
                patches=bounded.patches,
            )
        assert calls["assemble"] == 0
        assert calls["solver"] == 0
        plan = exc.value.plan
        # Independent dimensions, not model output alone.
        assert plan["preflight"]["complete_cells"] == 400
        assert plan["preflight"]["solver_inputs"] == {
            "cells": 400,
            "edges": 760,
            "free": 399,
        }
        assert plan["graph"]["matrices"]["A_bytes"] == 8 * 760 * 399
        assert plan["required_bytes"] > 2 * 1024 * 1024
        stats = access.stats()
        assert stats["loads"] == 0
        assert stats["verifications"] == 0
        assert stats["live_mappings"] == 0
        assert stats["fingerprints_computed"] == 0

        # Global mode prices no solve at all (metadata-only, no load).
        # Global mode: with a budget that covers the CANVAS ONLY it must fit
        # (the dense solve is not priced because it never runs).
        plan_global = zz._bounded_assembly_preflight(
            canvas,
            bounded,
            available_bytes=16 * 1024 * 1024,
            gauge_mode="global",
            grid=(nx, ny),
        )
        assert plan_global["graph_bytes"] == 0
        assert access.stats()["loads"] == 0
    finally:
        store.close()
    assert digest_tree(root) == before


def test_m5a_adequate_budget_from_real_plan_allows_assembly(tmp_path):
    results = _all_complete_2x2()
    root, store = seeded_store(tmp_path, results)
    store.close()
    canvas = make_canvas()
    store = reopen_read_only(root)
    try:
        access = zbas.StoreAssemblyAccess(store, max_resident_cells=2)
        bounded = access.results([r["cell_id"] for r in results])
        # Price first (diagnostic mode), then run with exactly that budget.
        plan = zz._bounded_assembly_preflight(
            canvas,
            bounded,
            available_bytes=256 * 2 ** 20,
            gauge_mode="per_cell",
            grid=(NX, NY),
        )
        budget = int(plan["required_bytes"])
        assembled, info, used = zz._assemble_cells_guarded(
            canvas,
            NX,
            NY,
            "per_cell",
            bounded.cores,
            bounded.descriptors,
            bounded_results=bounded,
            available_bytes=budget,
            halo_px=HALO,
            patches=bounded.patches,
        )
        assert used["fits"] is True
        assert used["required_bytes"] == budget
        assert assembled.coverage_pixels > 0
        assert info["offsets"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 13. REWORK-2 G2 — chunk-bounded C-order hashing on non-contiguous arrays
# ---------------------------------------------------------------------------


def _hash_peak(array, chunk):
    import tracemalloc

    tracemalloc.start()
    try:
        digest = zbas.array_sha256_streaming(array, chunk_bytes=chunk)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return digest, int(peak)


@pytest.mark.parametrize(
    "shape,dtype,slicer",
    [
        ((1, 524288), np.float32, lambda a: a[:, ::2]),          # thin + strided
        ((64, 128), np.float64, lambda a: a.T),                 # transposed
        ((64, 128), np.float32, lambda a: a[::-1, ::-1]),       # negative strides
        ((128, 128, 3), np.float32, lambda a: a[::2, ::2, :]),  # channelled
        ((256, 8), np.int32, lambda a: a[:, 1:2]),             # narrow strided
    ],
)
def test_m5a_noncontiguous_hash_is_chunk_bounded(shape, dtype, slicer):
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape).astype(dtype)
    array = slicer(base)
    assert not array.flags.c_contiguous or array.nbytes <= 4096
    canonical = zrr._array_sha256(array)  # reference may copy; outside tracing
    for chunk in (7, 1024, 65536):
        digest, peak = _hash_peak(array, chunk)
        assert digest == canonical, (shape, chunk)
        # The temporary is bounded by the requested chunk (+ fixed overhead),
        # never by the whole plane.
        assert peak <= chunk + 256 * 1024, (shape, chunk, peak, array.nbytes)


def test_m5a_noncontiguous_hash_peak_independent_of_plane_size():
    chunk = 4096
    small = np.zeros((1, 262144), dtype=np.float32)[:, ::2]
    large = np.zeros((1, 1048576), dtype=np.float32)[:, ::2]
    assert large.nbytes >= 4 * small.nbytes
    _d1, peak_small = _hash_peak(small, chunk)
    _d2, peak_large = _hash_peak(large, chunk)
    assert peak_small <= chunk + 256 * 1024
    assert peak_large <= chunk + 256 * 1024
    assert peak_large <= peak_small + 64 * 1024


def test_m5a_contiguous_hash_stays_zero_copy():
    plane = np.zeros((1024, 1024), dtype=np.float32)
    plane[::7, ::11] = 2.0
    _digest, peak = _hash_peak(plane, 64 * 1024)
    assert peak < plane.nbytes // 2


# ---------------------------------------------------------------------------
# 14. REWORK-2 — R37/R40 capture request validation before capture context
# ---------------------------------------------------------------------------


def test_m5a_capture_store_compatibility_guard():
    for r37, r40 in ((True, False), (False, True), (True, True)):
        with pytest.raises(RuntimeError, match="diagnostic capture"):
            zz._validate_capture_store_compatibility(
                True, r37_capture=r37, r40_capture=r40
            )
    # Store-less captures are fully supported (unchanged behaviour).
    for r37, r40 in ((True, False), (False, True), (True, True), (False, False)):
        assert (
            zz._validate_capture_store_compatibility(
                False, r37_capture=r37, r40_capture=r40
            )
            is None
        )
    assert (
        zz._validate_capture_store_compatibility(
            True, r37_capture=False, r40_capture=False
        )
        is None
    )


def test_m5a_capture_guard_runs_before_capture_context_and_run_work():
    import inspect

    source = inspect.getsource(zz._run_single)
    guard = source.index("_validate_capture_store_compatibility(")
    capture_ctx = source.index("_r40_build_seam_context(")
    assemble = source.index("_assemble_cells_guarded(")
    assert guard < capture_ctx
    assert guard < assemble
    # The call is in the MAIN function body (not inside a nested definition):
    # it would otherwise not run before the capture context at all.
    line = source[source.rfind("\n", 0, guard) + 1:source.find("\n", guard)]
    assert line.startswith("    _validate_capture_store_compatibility("), line
    # NOTE: full execution-order proof inside _run_single is NOT_RUN (no product
    # run here); this is a structural guard, the predicate itself is unit-tested.
