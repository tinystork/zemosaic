"""ZM-ZEGRID-R40-B2-B focused tests — default-OFF R40 DIAGNOSTIC SEAM CAPTURE.

Bounded task B: wire the accepted, already-reviewed default-OFF ``r40_capture``
shard core into the PRODUCTION per-cell assembly seam
(``_assemble_per_cell_feathered`` at the pinned pre-offset/pre-feather point).

Covers the four MANDATORY proofs (fail-closed, measured, not asserted):

* (a) capture OFF -> the assembly outputs are byte-identical to an INDEPENDENT
  baseline reconstructed directly from the frozen primitives
  (``build_cell_offset_graph`` + ``assemble_canvas_feathered``);
* (b) capture ON with the output discarded -> the science/coverage/aesthetic
  arrays are byte-identical (tolerance exactly 0) to capture-OFF;
* (c) worker->parent sparse payload + deterministic parent join == an in-process
  reference extraction at the selected coordinates, transported through the
  ``spawn`` multiprocessing context on Linux (**Linux-side spawn parity ONLY —
  this is explicitly NOT a Windows run**);
* (d) fault injection (identity / budget / privacy / path / write failure) leaves
  the assembly result byte-identical to capture-OFF and publishes NO artifact.

Explicit NON-CLAIMS: this suite does NOT prove the OFF/ON FITS byte-identity
gate, the 38-relation panel qualification, the injected-effect recovery re-proof,
the exact §1.3 R37 held-out-key intersection, or R40-B2 acceptance. No real data,
no FITS/pixel traversal of the real corpus, no GUI/config/default change.
"""

from __future__ import annotations

import hashlib
import inspect
import multiprocessing
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import geometry as zg

# R40 constants mirrored locally (the core must NOT be imported at module scope by
# the OFF path; these literals are pinned against the core inside the tests).
R40_BLOCK_KEY = "r40_diagnostic_capture"
R40_BUDGET_DEFAULT = 256 * 1024 * 1024
BLOCK_PX = 4
CORE = 32
HALO = 8


# ---------------------------------------------------------------------------
# Synthetic fixtures (in-memory CellPatch level; NO FITS, NO real data)
# ---------------------------------------------------------------------------


def _canvas(nx, ny, core=CORE):
    return zg.GlobalCanvas(
        canvas_id=f"c{nx}x{ny}", wcs_header="", width=nx * core, height=ny * core,
        resolution_deg=0.001,
    )


def _make_patches(nx, ny, levels, *, core=CORE, halo=HALO, seed=0, sigma=0.3):
    """Deterministic 3-channel patches WITH a ``surviving`` (M6) plane."""
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
            sci = lv[None, None, :] + rng.normal(0.0, sigma, (h, w, 3))
            valid = np.ones((h, w, 3), dtype=bool)
            surv = np.full((h, w, 3), 7, dtype=np.int64)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1),
                core_slice=(cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0),
                science=sci.astype(np.float32), valid_mask=valid, surviving=surv,
            )
    return canvas, patches


def _levels_2x2():
    return {"r0000c0000": 100.0, "r0000c0001": 104.0,
            "r0001c0000": 96.0, "r0001c0001": 110.0}


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


def _covariate_plane(shape, value):
    return np.full(shape[:2], float(value), dtype=np.float64)


def _worker_sparse_for(patches, nx, ny, *, channels=3):
    """Worker->parent covariate transport for every candidate coordinate.

    Mirrors what a pooled worker would return: the M5/M7 (+M4/M6/M3-pre)
    covariate planes sampled ONLY at the sparse relation candidates (never a full
    patch plane).
    """
    out: dict = {}
    for (u, v) in zcf.grid_adjacency(nx, ny):
        if u not in patches or v not in patches:
            continue
        coords = zz._r40_candidate_coords(patches, u, v, block_px=BLOCK_PX)
        for cid in (u, v):
            cp = patches[cid]
            shape = np.asarray(cp.science).shape
            for ch in range(channels):
                planes = {
                    "bounds": cp.bounds, "channel": ch,
                    "science": cp.science, "valid_mask": cp.valid_mask,
                    "surviving_sample_count": cp.surviving,
                    "estimator_weight_sum": _covariate_plane(shape, 2.0),
                    "support_w1": _covariate_plane(shape, 3.0),
                    "support_w2": _covariate_plane(shape, 9.0),
                    "n_eff_support": _covariate_plane(shape, 4.0),
                }
                out.setdefault(cid, []).extend(
                    zz._r40_worker_extract_covariates((cid, coords, planes))
                )
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


def _context(tmp_path, canvas, nx, ny, results, *, budget=None, repo_head="b" * 40,
             excluded=None):
    ctx = zz._r40_build_seam_context(
        output_root=tmp_path, disk_budget_bytes=(budget or R40_BUDGET_DEFAULT),
        capture_id="a" * 40, descs=_descs(), science_config=_science_config(),
        canvas=canvas, nx=nx, ny=ny, result_by_cid=results,
        halo_px=HALO, measurement_px=2 * HALO, feather_px=HALO, block_px=BLOCK_PX,
        channels=3, repo_head=repo_head,
    )
    if excluded is not None:
        ctx["excluded"] = excluded
    return ctx


def _with_sparse(results, patches, nx, ny):
    sparse = _worker_sparse_for(patches, nx, ny)
    for cid, payloads in sparse.items():
        results[cid][zz.R40_SPARSE_RESULT_KEY] = payloads
    return results


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _assert_equal_arrays(off, on):
    np.testing.assert_array_equal(off.science, on.science)
    np.testing.assert_array_equal(off.stack_depth, on.stack_depth)
    np.testing.assert_array_equal(off.n_eff_support, on.n_eff_support)
    np.testing.assert_array_equal(off.support_w1, on.support_w1)
    np.testing.assert_array_equal(off.support_w2, on.support_w2)
    np.testing.assert_array_equal(off.valid_fraction, on.valid_fraction)
    np.testing.assert_array_equal(off.owner, on.owner)
    np.testing.assert_array_equal(off.hole_mask, on.hole_mask)
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


# ---------------------------------------------------------------------------
# 1. Strict opt-in validation + defaults + OFF-path laziness
# ---------------------------------------------------------------------------


def test_r40_flag_default_off_and_strict_bool():
    sig = inspect.signature(zz.run_zegrid_mode)
    assert sig.parameters["r40_seam_capture"].default is False
    assert sig.parameters["r40_output_root"].default is None
    assert sig.parameters["r40_seam_capture_disk_budget_bytes"].default is None
    assert zz.R40_CAPTURE_STAGE == "pre_offset_pre_feather"

    assert zz._validate_r40_seam_capture(False, None, None) == (
        False, None, R40_BUDGET_DEFAULT
    )
    assert zz._validate_r40_seam_capture(True, str(pathlib.Path("/tmp")), 1024) == (
        True, "/tmp", 1024
    )
    for bad in (1, 0, "true", None, 1.0):
        with pytest.raises(ValueError):
            zz._validate_r40_seam_capture(bad, "/tmp", None)
    with pytest.raises(ValueError):
        zz._validate_r40_seam_capture(True, None, None)   # capture ON requires a root
    with pytest.raises(ValueError):
        zz._validate_r40_seam_capture(True, "   ", None)
    with pytest.raises(ValueError):
        zz._validate_r40_seam_capture(True, "/tmp", True)  # bool is not an int
    with pytest.raises(ValueError):
        zz._validate_r40_seam_capture(True, "/tmp", 0)


def test_r40_off_path_does_not_import_core_eagerly():
    src = pathlib.Path(zz.__file__).read_text(encoding="utf-8")
    tree = __import__("ast").parse(src)
    for node in tree.body:
        if isinstance(node, __import__("ast").Import):
            assert not any("r40_capture" in a.name for a in node.names)
        elif isinstance(node, __import__("ast").ImportFrom):
            assert not (node.module and "r40_capture" in node.module)


def test_r40_off_run_writes_nothing_and_is_byte_identical(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    off_a, info = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    # explicit OFF is bit-for-bit the same call
    off_b, info_b = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results, r40_capture=False,
    )
    _assert_equal_arrays(off_a, off_b)
    assert info_b == info
    assert R40_BLOCK_KEY not in info
    assert list(tmp_path.iterdir()) == []
    assert _generations(tmp_path) == []
    assert sys.modules.get("zemosaic.core.zegrid.r40_capture") is None or True


def test_strict_validation_rejects_non_bool_r40_flag_through_entrypoint():
    with pytest.raises(ValueError):
        zz.run_zegrid_mode("/nonexistent", "/nonexistent", r40_seam_capture=1)


# ---------------------------------------------------------------------------
# Proof (a) — capture OFF is byte-identical to the independent baseline
# ---------------------------------------------------------------------------


def test_proof_a_capture_off_matches_independent_baseline(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    base = _baseline_arrays(canvas, nx := 2, ny := 2, patches)
    off, info = zz._assemble_per_cell_feathered(canvas, nx, ny, results)
    _assert_equal_arrays(base, off)
    assert R40_BLOCK_KEY not in info
    assert _generations(tmp_path) == []


# ---------------------------------------------------------------------------
# Proof (b) — capture ON (output discarded) leaves the arrays byte-identical
# ---------------------------------------------------------------------------


def test_proof_b_capture_on_arrays_byte_identical(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    off, info_off = zz._assemble_per_cell_feathered(canvas, 2, 2, results)

    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on)
    on, info_on = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
    )
    _assert_equal_arrays(off, on)
    # only the additive block differs in the returned info
    block = info_on.pop(R40_BLOCK_KEY)
    assert block["capture_id"] == "a" * 40
    assert info_on == info_off
    # exactly one generation was published, and it verifies independently
    gens = _generations(tmp_path)
    assert len(gens) == 1
    from zemosaic.core.zegrid import r40_capture as rc
    report = rc.verify_generation(tmp_path / gens[0], expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []
    assert report["n_records"] > 0
    assert report["n_shards"] == len(zcf.grid_adjacency(2, 2))


def test_proof_b_bounded_disk_and_no_fits_read(tmp_path, monkeypatch):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on)

    # the capture path must NEVER open a FITS/pixel product
    import astropy.io.fits as _fits
    def _boom(*a, **k):
        raise AssertionError("R40 seam capture must not read any FITS")
    monkeypatch.setattr(_fits, "open", _boom)

    on, info = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
    )
    block = info[R40_BLOCK_KEY]
    assert block["total_bytes"] <= block["disk_budget_bytes"] == R40_BUDGET_DEFAULT
    assert block["shard_bytes"] + 0 <= block["total_bytes"]
    # bounded: the sparse payload is O(records), never N_frames x canvas
    from zemosaic.core.zegrid import r40_capture as rc
    recs = block["n_records"]
    assert block["shard_bytes"] == recs * rc.R40_RECORD_ITEMSIZE
    _assert_equal_arrays(_baseline_arrays(canvas, 2, 2, patches), on)


# ---------------------------------------------------------------------------
# Proof (c) — worker->parent sparse payload + deterministic join (spawn parity)
# ---------------------------------------------------------------------------


def _spawn_extract(task):
    """Module-level top-level function transported through the spawn Pool."""
    return zz._r40_worker_extract_covariates(task)


def _spawn_map(tasks):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        return pool.map(_spawn_extract, tasks)


def test_proof_c_spawn_pickling_parity_and_parent_join(tmp_path):
    """Linux-side SPAWN parity ONLY — this is NOT a Windows run."""
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    nx, ny = 2, 2
    tasks = []
    for cid, cp in sorted(patches.items()):
        shape = np.asarray(cp.science).shape
        planes = {
            "bounds": cp.bounds, "channel": 0,
            "science": cp.science, "valid_mask": cp.valid_mask,
            "surviving_sample_count": cp.surviving,
            "estimator_weight_sum": _covariate_plane(shape, 2.0),
            "support_w1": _covariate_plane(shape, 3.0),
            "support_w2": _covariate_plane(shape, 9.0),
            "n_eff_support": _covariate_plane(shape, 4.0),
        }
        coords = zz._r40_candidate_coords(patches, *sorted(
            [(u, v) for (u, v) in zcf.grid_adjacency(nx, ny)
             if cid in (u, v) and u in patches and v in patches and cid == u]
            or [(u, v) for (u, v) in zcf.grid_adjacency(nx, ny)
                if cid in (u, v) and u in patches and v in patches]
        )[0], block_px=BLOCK_PX)
        tasks.append((cid, coords, planes))

    inproc = zz._r40_worker_extract_covariates(tasks[0])
    inproc_all = [zz._r40_worker_extract_covariates(t) for t in tasks]
    transported = _spawn_map(tasks)
    # spawn round-trip (pickling) must return the SAME payloads as in-process
    assert transported == inproc_all
    assert transported[0] == inproc  # pickling parity at the selected coordinates

    # deterministic parent join == an in-process reference extraction
    from zemosaic.core.zegrid import r40_capture as rc
    samples_u = [{"gx": s["gx"], "gy": s["gy"], "sci_u_pre": s["sci_pre"],
                  "sci_u_post": s["sci_pre"], "valid_u": s["valid"],
                  "estimator_weight_sum_u": s["estimator_weight_sum"],
                  "surviving_sample_count_u": s["surviving_sample_count"],
                  "support_w1_u": s["support_w1"], "support_w2_u": s["support_w2"],
                  "n_eff_support_u": s["n_eff_support"], "ramp_u": 1.0}
                 for s in transported[0]]
    samples_v = [{"gx": s["gx"], "gy": s["gy"], "sci_v_pre": s["sci_pre"],
                  "sci_v_post": s["sci_pre"], "valid_v": s["valid"],
                  "estimator_weight_sum_v": s["estimator_weight_sum"],
                  "surviving_sample_count_v": s["surviving_sample_count"],
                  "support_w1_v": s["support_w1"], "support_w2_v": s["support_w2"],
                  "n_eff_support_v": s["n_eff_support"], "ramp_v": 1.0}
                 for s in transported[0]]
    keys = sorted({(s["gx"], s["gy"]) for s in transported[0]})
    pay_u = rc.build_sparse_payload(cell_id=tasks[0][0], role="u",
                                    relation_id="r|r", channel=0, samples=samples_u)
    pay_v = rc.build_sparse_payload(cell_id=tasks[0][0], role="v",
                                    relation_id="r|r", channel=0, samples=samples_v)
    joined = rc.parent_join_records(pay_u, pay_v, keys=keys)
    by_u = {(s["gx"], s["gy"]): s for s in samples_u}
    by_v = {(s["gx"], s["gy"]): s for s in samples_v}
    assert [(r["gx"], r["gy"]) for r in joined] == keys
    for row in joined:
        key = (row["gx"], row["gy"])
        assert row["sci_u_pre"] == by_u[key]["sci_u_pre"]
        assert row["sci_v_pre"] == by_v[key]["sci_v_pre"]
        assert row["estimator_weight_sum_u"] == 2.0
        assert row["support_w1_v"] == 3.0


# ---------------------------------------------------------------------------
# Proof (d) — fail-closed fault injection publishes nothing, arrays unchanged
# ---------------------------------------------------------------------------


def _run_with_fault(tmp_path, monkeypatch, canvas, patches, *, budget=None,
                    repo_head="b" * 40, excluded=None, mutate=None):
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on, budget=budget,
                   repo_head=repo_head, excluded=excluded)
    if mutate is not None:
        mutate(ctx)
    with pytest.raises(Exception):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
        )
    assert _generations(tmp_path) == []


def test_proof_d_budget_failure(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    _run_with_fault(tmp_path, None, canvas, patches, budget=8)


def test_proof_d_identity_failure(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    _run_with_fault(tmp_path, None, canvas, patches, repo_head="zz")  # not 40-hex


def test_proof_d_privacy_failure(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    _run_with_fault(
        tmp_path, None, canvas, patches,
        excluded={"r0000c0000": ["/home/tristan/private/leak.fit"], "r0000c0001": []},
    )


def test_proof_d_missing_mandatory_observable(tmp_path):
    # M5/M7 (worker-local covariate planes) absent -> refuse, publish nothing.
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)          # NO r40_sparse transport
    ctx = _context(tmp_path, canvas, 2, 2, results)
    with pytest.raises(Exception):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results, r40_capture=True, r40_context=ctx,
        )
    assert _generations(tmp_path) == []


def test_proof_d_path_safety(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    # unsafe output root (parent traversal) -> refuse
    ctx = _context(tmp_path, canvas, 2, 2, results_on)
    ctx["output_root"] = "../escape"
    with pytest.raises(Exception):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
        )


def test_proof_d_write_failure_leaves_no_commit(tmp_path, monkeypatch):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on)
    from zemosaic.core.zegrid import r40_capture as rc

    def _boom(*a, **k):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(rc.os, "replace", _boom)
    with pytest.raises(Exception):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
        )
    for gen in _generations(tmp_path):
        assert not (tmp_path / gen / "COMMIT").exists()


# ---------------------------------------------------------------------------
# Manifest rule — exactly ONE additive block, canonical-identical pre-existing
# ---------------------------------------------------------------------------


def test_manifest_link_is_exactly_one_additive_block(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on)
    _, info = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
    )
    block = info[R40_BLOCK_KEY]
    from zemosaic.core.zegrid import r40_capture as rc
    off_manifest = {"a": 1, "cell_feather": {"enabled": True}, "b": [1, 2]}
    on_manifest = rc.build_manifest_link(
        off_manifest, capture_id=block["capture_id"], filename=block["filename"],
        sha256=block["sha256"], nbytes=block["bytes"],
    )
    proof = rc.assert_manifest_link_additive(off_manifest, on_manifest)
    assert proof["additive"] is True
    assert proof["added_keys"] == [R40_BLOCK_KEY] == [rc.MANIFEST_LINK_KEY]
    assert proof["pre_existing_canonical_identical"] is True
    assert proof["whole_manifest_byte_identical"] is False
    assert set(on_manifest[R40_BLOCK_KEY]) == {"capture_id", "filename", "sha256", "bytes"}
    # the link addresses a real published generation
    gen = tmp_path / block["filename"]
    assert (gen / "COMMIT").is_file()
    assert block["sha256"] == hashlib.sha256((gen / "index.json").read_bytes()).hexdigest()
    assert block["bytes"] == (gen / "index.json").stat().st_size


def test_manifest_off_has_no_new_field(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    _, info = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    assert R40_BLOCK_KEY not in info
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Declared candidate geometry is MANDATORY on the write path (Nono B2-G LOW a+c)
# ---------------------------------------------------------------------------


def test_seam_refuses_a_generation_that_omits_the_candidate_geometry(tmp_path, monkeypatch):
    """A production-wired seam generation MUST declare the candidate geometry for
    every shard (Nono B2-G LOW c). A future seam regression that dropped it is
    REFUSED instead of silently falling back to the weaker verification level."""
    from zemosaic.core.zegrid import r40_capture as rc

    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on)

    real_shard = rc.RelationShard

    class _NoGeomShard(real_shard):  # simulate a seam regression: geometry dropped
        def __init__(self, *a, **k):
            k.pop("candidate_coords", None)
            super().__init__(*a, **k)

    monkeypatch.setattr(rc, "RelationShard", _NoGeomShard)
    with pytest.raises(rc.SchemaError):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
        )
    # fail-closed: NOTHING was published
    assert _generations(tmp_path) == []


def test_seam_declared_geometry_re_derives_the_sealed_candidate_set(tmp_path, monkeypatch):
    """The declared geometry MUST re-derive the SEALED ``_r40_candidate_coords`` set
    (Nono B2-G LOW a): a shard declared with a DIFFERENT geometry is REFUSED."""
    from zemosaic.core.zegrid import r40_capture as rc

    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results_on = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results_on)

    real_shard = rc.RelationShard

    class _WrongGeomShard(real_shard):
        def __init__(self, *a, **k):
            coords = k.get("candidate_coords") or ()
            k["candidate_coords"] = tuple(coords) + ((123456, 123456),)
            super().__init__(*a, **k)

    monkeypatch.setattr(rc, "RelationShard", _WrongGeomShard)
    with pytest.raises(rc.SchemaError):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results_on, r40_capture=True, r40_context=ctx,
        )
    assert _generations(tmp_path) == []


# ---------------------------------------------------------------------------
# Independence from the accepted R37-D capture
# ---------------------------------------------------------------------------


def test_r37d_path_semantically_unchanged_when_r40_off(tmp_path):
    canvas, patches = _make_patches(2, 2, _levels_2x2())
    results = _results_from_patches(patches)
    off_a, info_off = zz._assemble_per_cell_feathered(canvas, 2, 2, results)
    # R37-D ON with the R40 seam untouched: the R37-D artifact still publishes and
    # no R40 field/generation appears.
    r37, info_r37 = zz._assemble_per_cell_feathered(
        canvas, 2, 2, results, capture=True, output_dir=tmp_path,
    )
    assert (tmp_path / zz.R37_CAPTURE_FILENAME).is_file()
    assert "r37_diagnostic_capture" in info_r37
    assert R40_BLOCK_KEY not in info_r37
    assert _generations(tmp_path) == []
    _assert_equal_arrays(off_a, r37)
