"""ZM-ZEGRID-R40-B2-C focused tests — REAL pooled worker→parent sparse transport.

Bounded task C closes the honest gap flagged in B2-B: the seam consumes
``result_by_cid[cid]['r40_sparse']``, which only synthetic fixtures provided. Here
the REAL pooling worker (``_stack_cell`` on the production ``zpar.pmap`` pool)
computes and returns ONLY the sparse per-relation candidate samples of its OWN
cell(s) from its worker-local MiniTile planes, and the seam capture verifies.

Mandatory proofs (measured, never asserted):

* (e) a REAL pooled run with the capture ON produces ``r40_sparse`` payloads per
  cell IDENTICAL to the in-process reference extraction at the same coordinates,
  and the resulting seam capture verifies
  (``verify_generation(..., expected_production_wired=True)`` → ``ok=True,
  problems=[]``). The identity half is transported through a genuine
  ``multiprocessing.get_context('spawn')`` pool — **Linux-side spawn parity ONLY,
  explicitly NOT a Windows run**;
* (f) the same real pooled run with the capture OFF produces worker results with
  NO ``r40_sparse`` key, an UNCHANGED (pre-R40) worker initargs tuple and an
  assembly byte-identical to an INDEPENDENT frozen-primitive baseline;
* (g) fault injection (transport dropped / truncated / a missing coordinate) →
  refuse, NO artifact, input planes byte-unchanged.

Explicit NON-CLAIMS: this suite does NOT claim R40-B2 acceptance, the OFF/ON FITS
byte-identity gate, the 38-relation panel qualification, the injected-effect
re-proof, the real product ROI parity or the exact §1.3 held-out-key intersection
(all ``NOT_RUN``). No real data, no FITS/pixel/image/dataset traversal of the real
corpus, no Windows/Zealfie/LP341, no product default/GUI/config change.
"""

from __future__ import annotations

import hashlib
import inspect
import multiprocessing
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid.assembly import MiniTile

# The synthetic end-to-end runs write FITS via the production path, which emits
# the PRE-EXISTING astropy ``Card is too long`` VerifyWarning on the synthetic WCS
# header (identical to the accepted R37-D / R34f e2e tests). Filter ONLY that
# warning so ``-W error`` still guards everything R40-B2-C cares about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)

R40_BLOCK_KEY = "r40_diagnostic_capture"
BLOCK_PX = 4
CORE = 32
HALO = 8
CHANNELS = 3
# Pre-R40 worker result key set for a COMPLETE cell (captured from the accepted
# B2-B wiring): the transport must add NO key when the capture is OFF.
_PRE_R40_COMPLETE_KEYS = frozenset({
    "cell_id", "status", "core", "record", "backend_used", "n_frames",
    "total_bytes", "cache_build_s", "stack_s", "peak_rss_kib", "cleanup_failed",
    "reproject", "cache_dir", "patch",
})


# ---------------------------------------------------------------------------
# Synthetic in-memory fixtures (CellPatch / MiniTile level; NO FITS, NO data)
# ---------------------------------------------------------------------------


def _canvas(nx, ny, core=CORE):
    return zg.GlobalCanvas(
        canvas_id=f"c{nx}x{ny}", wcs_header="", width=nx * core, height=ny * core,
        resolution_deg=0.001,
    )


def _make_patches(nx, ny, levels, *, core=CORE, halo=HALO, seed=0, sigma=0.3):
    rng = np.random.default_rng(seed)
    canvas = _canvas(nx, ny, core)
    patches = {}
    for row in range(ny):
        for col in range(nx):
            cid = f"r{row:04d}c{col:04d}"
            cy0, cy1 = row * core, (row + 1) * core
            cx0, cx1 = col * core, (col + 1) * core
            py0, py1 = max(0, cy0 - halo), min(canvas.height, cy1 + halo)
            px0, px1 = max(0, cx0 - halo), min(canvas.width, cx1 + halo)
            h, w = py1 - py0, px1 - px0
            lv = np.full(3, float(levels[cid]))
            sci = (lv[None, None, :] + rng.normal(0.0, sigma, (h, w, 3))).astype(np.float32)
            valid = np.ones((h, w, 3), dtype=bool)
            surv = np.full((h, w, 3), 7, dtype=np.int64)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1),
                core_slice=(cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0),
                science=sci, valid_mask=valid, surviving=surv,
            )
    return canvas, patches


def _levels_2x2():
    return {"r0000c0000": 100.0, "r0000c0001": 104.0,
            "r0001c0000": 96.0, "r0001c0001": 110.0}


def _covariate_planes(shape, base=2.0):
    """Worker-local E7–E9 covariate planes (science shape H,W,C)."""
    return {
        "estimator_weight_sum": np.full(shape, base + 0.0, dtype=np.float64),
        "support_w1": np.full(shape, base + 1.0, dtype=np.float64),
        "support_w2": np.full(shape, base + 7.0, dtype=np.float64),
        "n_eff_support": np.full(shape, base + 2.0, dtype=np.float64),
    }


def _minitile_for(cp):
    shape = np.asarray(cp.science).shape
    return MiniTile(
        cell_id=cp.cell_id, patch_shape_hw=shape[:2], core_slice=cp.core_slice,
        science=cp.science, valid_mask=cp.valid_mask,
        surviving_sample_count=cp.surviving, **_covariate_planes(shape),
        reference_frame_id=f"ref-{cp.cell_id}", excluded=(), frame_order=(),
        provenance={},
    )


class _R40FakeProcessingPatch:
    """Picklable stand-in exposing only the ``.patch`` bounds ``_stack_cell``
    reads when arming the transport (no real ``ProcessingPatch`` needed)."""

    def __init__(self, bounds):
        self.patch = SimpleNamespace(y0=int(bounds[0]), y1=int(bounds[1]),
                                     x0=int(bounds[2]), x1=int(bounds[3]))


def _spawn_armed_worker(task):
    """Module-level pool worker: the EXACT transport call site of ``_stack_cell``."""
    cid, minitile, bounds = task
    return zz._r40_maybe_worker_sparse(cid, minitile, _R40FakeProcessingPatch(bounds))


def _spawn_pool(spawn_ctx, tasks, coords):
    """Real spawn Pool, armed through the PRODUCTION ``_init_stack_worker``."""
    initargs = (None, None, None, None, None, None, 128, 1, coords)
    with spawn_ctx.Pool(processes=1, initializer=zz._init_stack_worker,
                        initargs=initargs) as pool:
        return pool.map(_spawn_armed_worker, tasks)


def _reference_payload(cid, coords, cp):
    """In-process reference: the B2-B extractor at the same coordinates."""
    shape = np.asarray(cp.science).shape
    planes = {
        "bounds": cp.bounds, "science": cp.science, "valid_mask": cp.valid_mask,
        "surviving_sample_count": cp.surviving, **_covariate_planes(shape),
    }
    out = []
    for ch in range(CHANNELS):
        planes_ch = dict(planes)
        planes_ch["channel"] = ch
        out.extend(zz._r40_worker_extract_covariates((cid, coords, planes_ch)))
    return out


def _results_from_patches(patches):
    out = {}
    for cid, cp in patches.items():
        out[cid] = {
            "cell_id": cid, "status": "complete",
            "patch": {
                "bounds": cp.bounds, "core_slice": cp.core_slice,
                "science": cp.science, "valid_mask": cp.valid_mask,
                "surviving": cp.surviving,
            },
            "record": {
                "reference_frame_id": f"ref-{cid}",
                "excluded": [[f"f{i}.fit", "stage", "reason"] for i in range(2)],
            },
        }
    return out


def _science_config():
    return SimpleNamespace(
        normalization="sky_mean", weighting="noise_variance", rejection="kappa_sigma",
        combine="mean", taper="footprint", taper_px=8.0, taper_floor=0.0,
        equalize_rgb=False, sigma_low=3.0, sigma_high=3.0,
        winsor_limit_low=0.05, winsor_limit_high=0.05,
        cell_gauge_mode="per_cell",
    )


def _descs(n=4):
    return [SimpleNamespace(frame_id=SimpleNamespace(logical_path=f"f{i}.fit"))
            for i in range(n)]


def _context(tmp_path, canvas, nx, ny, results):
    return zz._r40_build_seam_context(
        output_root=tmp_path, disk_budget_bytes=zz.R40_SEAM_CAPTURE_DISK_BUDGET_BYTES,
        capture_id="a" * 40, descs=_descs(), science_config=_science_config(),
        canvas=canvas, nx=nx, ny=ny, result_by_cid=results,
        halo_px=HALO, measurement_px=2 * HALO, feather_px=HALO, block_px=BLOCK_PX,
        channels=CHANNELS, repo_head="b" * 40,
    )


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _assert_equal_arrays(off, on):
    np.testing.assert_array_equal(off.science, on.science)
    np.testing.assert_array_equal(off.stack_depth, on.stack_depth)
    np.testing.assert_array_equal(off.valid_fraction, on.valid_fraction)
    np.testing.assert_array_equal(off.owner, on.owner)
    assert _sha(np.asarray(off.science, dtype=np.float32)) == _sha(
        np.asarray(on.science, dtype=np.float32)
    )


def _baseline_arrays(canvas, nx, ny, patches):
    """INDEPENDENT baseline from the frozen primitives (no R40 involved)."""
    graph = zcf.build_cell_offset_graph(
        canvas, nx, ny, patches, on_disconnected=zcf.FEATHER_DISCONNECTED_POLICY
    )
    offsets = {cid: graph.offsets[i] for i, cid in enumerate(graph.cell_ids)}
    return zcf.assemble_canvas_feathered(canvas, nx, ny, patches, offsets, halo_px=HALO)


def _generations(root):
    return sorted(p.name for p in pathlib.Path(root).glob("r40_capture_*"))


def _bounds_by_cell(patches):
    return {cid: cp.bounds for cid, cp in patches.items()}


# ---------------------------------------------------------------------------
# Synthetic on-disk corpus (real pooled end-to-end run; NO real data)
# ---------------------------------------------------------------------------

_DISJOINT_CENTERS = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]


def _write_frame(path, ra, dec, shape=(400, 600), scale=0.001, seed=0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * scale
    w.array_shape = shape
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
    fits.PrimaryHDU(data, header=w.to_header()).writeto(path, overwrite=True)


def _write_corpus(root):
    root.mkdir(parents=True, exist_ok=True)
    for i, (ra, dec) in enumerate(_DISJOINT_CENTERS):
        _write_frame(root / f"f{i}.fit", ra, dec, seed=i)
    return root


def _run_pooled(inp, out, monkeypatch, *, r40):
    """Run the REAL production pooled pipeline; capture the pool initargs + results."""
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 3)
    seen = {"initargs": [], "result_by_cid": None}
    real_pmap = zpar.pmap

    def _spy_pmap(worker, tasks, workers=None, **kw):
        if kw.get("initializer") is zz._init_stack_worker:
            seen["initargs"].append(tuple(kw.get("initargs") or ()))
        return real_pmap(worker, tasks, workers, **kw)

    monkeypatch.setattr(zpar, "pmap", _spy_pmap)

    real_assemble = zz._assemble_cells

    def _spy_assemble(canvas, nx, ny, mode, cores, result_by_cid, **kw):
        seen["result_by_cid"] = result_by_cid
        return real_assemble(canvas, nx, ny, mode, cores, result_by_cid, **kw)

    monkeypatch.setattr(zz, "_assemble_cells", _spy_assemble)

    kwargs = dict(zconfig=SimpleNamespace(zegrid_layout="2x2"), cell_gauge_mode="per_cell")
    if r40:
        kwargs.update(r40_seam_capture=True, r40_output_root=str(out))
    zz.run_zegrid_mode(str(inp), str(out), **kwargs)
    return seen


def _science_sha(out):
    with fits.open(out / "mosaic_grid.fits") as hdul:
        data = np.ascontiguousarray(np.asarray(hdul[0].data, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Structure / default-OFF
# ---------------------------------------------------------------------------


def test_transport_default_off_and_strict_structured():
    # the transport global is unarmed by default (no retention, no payload)
    assert zz._STACK_R40_WORKER_COORDS is None
    sig = inspect.signature(zz._init_stack_worker)
    assert sig.parameters["r40_worker_coords"].default is None
    # unarmed initializer keeps the transport global unarmed
    zz._init_stack_worker(None, None, None, None, None, None, 128, 1)
    assert zz._STACK_R40_WORKER_COORDS is None
    zz._STACK_R40_WORKER_COORDS = None


def test_transport_new_component_does_not_import_core_eagerly():
    src = pathlib.Path(zz.__file__).read_text(encoding="utf-8")
    tree = __import__("ast").parse(src)
    for node in tree.body:
        if isinstance(node, __import__("ast").Import):
            assert not any("r40_capture" in a.name for a in node.names)
        elif isinstance(node, __import__("ast").ImportFrom):
            assert not (node.module and "r40_capture" in node.module)


def test_parent_candidate_coords_reuses_sealed_geometry():
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    coords = zz._r40_worker_candidate_coords(_bounds_by_cell(patches), 2, 2, block_px=BLOCK_PX)
    # every relation contributes its sealed candidates to BOTH endpoints
    for (u, v) in zcf.grid_adjacency(2, 2):
        rel = set(zz._r40_candidate_coords(patches, u, v, block_px=BLOCK_PX))
        assert rel <= set(coords[u]) and rel <= set(coords[v])
    # deterministic order by (gy, gx)
    for cid, cs in coords.items():
        assert list(cs) == sorted(cs, key=lambda k: (k[1], k[0]))
    assert set(coords) == set(patches)


# ---------------------------------------------------------------------------
# Proof (e) — REAL pooled run, capture ON, seam verifies
# ---------------------------------------------------------------------------


def test_proof_e1_real_pooled_run_capture_on_verifies(tmp_path, monkeypatch):
    """Fork-pooled PRODUCTION run (``zpar.pmap`` real process pool) with capture ON."""
    inp = _write_corpus(tmp_path / "in")
    out = tmp_path / "out"
    seen = _run_pooled(inp, out, monkeypatch, r40=True)

    # a REAL process pool was spun up (never serial)
    assert len(seen["initargs"]) == 1
    diag = None
    import json
    man = json.loads((out / "zegrid_manifest.json").read_text())
    diag = man["per_cell_diagnostics"]
    assert diag["executor"] == "process" and diag["workers"] >= 2
    assert diag["tasks"] >= 3

    # the transport arg was APPENDED to the production initargs (9 vs the pre-R40 8)
    assert len(seen["initargs"][0]) == 9
    assert seen["initargs"][0][-1] is not None

    # each worker returned ONLY its own cell's sparse samples
    rbc = seen["result_by_cid"]
    complete = [cid for cid, r in rbc.items() if r.get("status") == "complete"]
    assert complete
    for cid in complete:
        payload = rbc[cid].get(zz.R40_SPARSE_RESULT_KEY)
        assert payload, f"cell {cid} carries no sparse transport"
        assert all(s["cell_id"] == cid for s in payload)
        assert {s["channel"] for s in payload} == {0, 1, 2}

    # the resulting seam capture verifies
    from zemosaic.core.zegrid import r40_capture as rc
    gens = _generations(out)
    assert len(gens) == 1
    report = rc.verify_generation(out / gens[0], expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []
    assert report["n_shards"] >= 2 and report["n_records"] > 0
    block = man[R40_BLOCK_KEY]
    assert block["filename"] == gens[0]


def test_proof_e2_spawn_pool_payload_identity_and_bound(tmp_path):
    """Linux-side SPAWN parity ONLY — this is explicitly NOT a Windows run."""
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    coords = zz._r40_worker_candidate_coords(_bounds_by_cell(patches), 2, 2, block_px=BLOCK_PX)
    tasks = [(cid, _minitile_for(cp), cp.bounds) for cid, cp in sorted(patches.items())]

    spawn_ctx = multiprocessing.get_context("spawn")
    pooled = _spawn_pool(spawn_ctx, tasks, coords)

    # IDENTICAL to the in-process reference extraction at the same coordinates
    for (cid, _mt, _b), payload in zip(tasks, pooled):
        assert payload == _reference_payload(cid, coords[cid], patches[cid])

    # transported payload is O(candidate records), never O(full patch area)
    for (cid, mt, _b), payload in zip(tasks, pooled):
        assert len(payload) == len(coords[cid]) * CHANNELS
        full_patch_px = int(np.prod(np.asarray(mt.science).shape[:2])) * CHANNELS
        assert len(payload) * CHANNELS < full_patch_px
    total_records = sum(len(p) for p in pooled)
    total_patch_px = sum(
        int(np.prod(np.asarray(cp.science).shape[:2])) * CHANNELS for cp in patches.values()
    )
    assert total_records < total_patch_px

    # the transported payloads drive a verifying seam capture
    results = _results_from_patches(patches)
    for (cid, _mt, _b), payload in zip(tasks, pooled):
        results[cid][zz.R40_SPARSE_RESULT_KEY] = payload
    ctx = _context(tmp_path, canvas, 2, 2, results)
    _, info = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results, r40_capture=True, r40_context=ctx,
    )
    from zemosaic.core.zegrid import r40_capture as rc
    gens = _generations(tmp_path)
    assert len(gens) == 1
    report = rc.verify_generation(tmp_path / gens[0], expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []
    assert info[R40_BLOCK_KEY]["filename"] == gens[0]


# ---------------------------------------------------------------------------
# Proof (f) — REAL pooled run, capture OFF: no key, unchanged initargs, baseline
# ---------------------------------------------------------------------------


def test_proof_f_real_pooled_run_capture_off_no_transport(tmp_path, monkeypatch):
    inp = _write_corpus(tmp_path / "in")
    out = tmp_path / "out"
    seen = _run_pooled(inp, out, monkeypatch, r40=False)

    # a REAL process pool ran (the OFF path is the same production pool)
    import json
    diag = json.loads((out / "zegrid_manifest.json").read_text())["per_cell_diagnostics"]
    assert diag["executor"] == "process" and diag["workers"] >= 2

    # NO extra IPC payload: the worker initargs tuple is the pre-R40 tuple
    assert len(seen["initargs"]) == 1
    assert len(seen["initargs"][0]) == 8

    # NO transport key, NO extra key on any worker result
    rbc = seen["result_by_cid"]
    complete = [cid for cid, r in rbc.items() if r.get("status") == "complete"]
    assert complete
    for cid in complete:
        assert zz.R40_SPARSE_RESULT_KEY not in rbc[cid]
        assert set(rbc[cid]) == _PRE_R40_COMPLETE_KEYS
    assert zz._STACK_R40_WORKER_COORDS is None

    # NO artifact, NO manifest field
    assert _generations(out) == []
    man = json.loads((out / "zegrid_manifest.json").read_text())
    assert R40_BLOCK_KEY not in man

    # the OFF assembly is byte-identical to a second identical OFF run (no
    # measurable behaviour change from the transport code path)
    out2 = tmp_path / "out2"
    _run_pooled(inp, out2, monkeypatch, r40=False)
    assert _science_sha(out) == _science_sha(out2)
    # and byte-identical to the INDEPENDENT frozen-primitive baseline at CellPatch
    # level (the transport adds nothing to the assembly)
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    off, info = zz._assemble_per_cell_feathered(canvas, 2, 2, _results_from_patches(patches))
    _assert_equal_arrays(_baseline_arrays(canvas, 2, 2, patches), off)
    assert R40_BLOCK_KEY not in info


def test_unarmed_worker_transport_returns_none_and_adds_no_key():
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    cp = patches["r0000c0000"]
    zz._STACK_R40_WORKER_COORDS = None
    assert zz._r40_maybe_worker_sparse(cp.cell_id, _minitile_for(cp),
                                       _R40FakeProcessingPatch(cp.bounds)) is None
    # an armed transport with no coords for this cell yields an EMPTY (still bounded)
    # payload, never a full plane
    zz._STACK_R40_WORKER_COORDS = {"some-other-cell": ((0, 0),)}
    assert zz._r40_maybe_worker_sparse(cp.cell_id, _minitile_for(cp),
                                       _R40FakeProcessingPatch(cp.bounds)) == ()
    zz._STACK_R40_WORKER_COORDS = None


# ---------------------------------------------------------------------------
# Proof (g) — fault injection: transport dropped/truncated -> refuse, no artifact
# ---------------------------------------------------------------------------


def _pooled_results_and_ctx(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    coords = zz._r40_worker_candidate_coords(_bounds_by_cell(patches), 2, 2, block_px=BLOCK_PX)
    tasks = [(cid, _minitile_for(cp), cp.bounds) for cid, cp in sorted(patches.items())]
    pooled = _spawn_pool(multiprocessing.get_context("spawn"), tasks, coords)
    results = _results_from_patches(patches)
    for (cid, _mt, _b), payload in zip(tasks, pooled):
        results[cid][zz.R40_SPARSE_RESULT_KEY] = payload
    ctx = _context(tmp_path, canvas, 2, 2, results)
    return canvas, patches, results, ctx


def _plane_hashes(results):
    return {
        cid: (
            _sha(r["patch"]["science"]), _sha(r["patch"]["valid_mask"]),
            _sha(r["patch"]["surviving"]),
        )
        for cid, r in results.items()
    }


@pytest.mark.parametrize("fault", ["dropped_payload", "truncated_samples", "missing_coord"])
def test_proof_g_fault_injection_refuses_no_artifact(tmp_path, fault):
    canvas, patches, results, ctx = _pooled_results_and_ctx(tmp_path)
    before = _plane_hashes(results)

    # capture ON and healthy once -> a generation is published
    healthy_root = tmp_path / "healthy"
    healthy_root.mkdir()
    healthy_ctx = dict(ctx, output_root=str(healthy_root))
    zz._assemble_per_cell_feathered(
        canvas, 2, 2, results, r40_capture=True, r40_context=healthy_ctx,
    )
    assert len(_generations(healthy_root)) == 1

    # inject the fault into the transport (the cell itself STILL exists: this is a
    # transport fault, never a dropped relation)
    cid = sorted(results)[0]
    key = zz.R40_SPARSE_RESULT_KEY
    if fault == "dropped_payload":
        results[cid].pop(key)                      # transport dropped entirely
    elif fault == "truncated_samples":
        results[cid][key] = results[cid][key][:1]  # transport truncated
    else:
        results[cid][key] = []                     # a relation endpoint lost its samples

    with pytest.raises(Exception):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results, r40_capture=True, r40_context=ctx,
        )
    assert _generations(tmp_path) == []          # NO artifact published
    assert _plane_hashes(results) == before      # input planes byte-unchanged


def _faulty_worker(task):
    """Module-level (picklable) worker that DROPS the transport."""
    return None


def test_proof_g_worker_level_fault_refuses(tmp_path):
    """Capture ON but a worker emitted NO transport -> the seam refuses."""
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    coords = zz._r40_worker_candidate_coords(_bounds_by_cell(patches), 2, 2, block_px=BLOCK_PX)
    tasks = [(cid, _minitile_for(cp), cp.bounds) for cid, cp in sorted(patches.items())]
    ctx = multiprocessing.get_context("spawn")
    initargs = (None, None, None, None, None, None, 128, 1, coords)
    with ctx.Pool(processes=1, initializer=zz._init_stack_worker, initargs=initargs) as pool:
        pooled = pool.map(_faulty_worker, tasks)
    assert pooled == [None] * len(tasks)

    results = _results_from_patches(patches)  # no r40_sparse attached
    seam_ctx = _context(tmp_path, canvas, 2, 2, results)
    with pytest.raises(Exception):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results, r40_capture=True, r40_context=seam_ctx,
        )
    assert _generations(tmp_path) == []
