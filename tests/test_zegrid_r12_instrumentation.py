"""ZM-ZEGRID-R12 targeted tests — instrumentation + parallel cache build.

Covers (non-gated, fast):

* the per-phase :class:`Timings` accumulator;
* ignored-settings detection (exact names + ``final_mosaic_dbe_*`` prefix) and
  the explicit GPU-usage note;
* the manifest ``timings`` / ``gpu`` / ``ignored_settings`` blocks (via
  :func:`_write_outputs`);
* the run log written to the output folder (the user's "le log est absent");
* the R11 L1 reference-provenance helper (bookkeeping placeholder labelling).

Gated on real M16 data (skipped when absent): the parallel cache build is
BIT-EQUAL to the serial build (hash of the aligned ``.npy`` cache), and the
chosen worker count is memory-aware.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_streaming import compute_fixed_normalization
from zemosaic.core.zegrid import execution as zxe
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import instrumentation as zin
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid import streaming as zstream
from zemosaic.core.zegrid import sweep as zsw
from zemosaic.core.zegrid.executor import ExecutorConfig

LIGHTS = Path("/home/tristan/M16/quick")
NOOP = lambda *a, **k: None


def _square(x):
    """Module-level (picklable) worker for the pmap order-preservation test."""
    return x * x


def _synthetic_tan_wcs(shape=(10, 10)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


# ---------------------------------------------------------------------------
# Timings accumulator
# ---------------------------------------------------------------------------

def test_timings_accumulate_and_dict():
    t = zin.Timings()
    t.add("setup", 1.0)
    t.add("setup", 0.5)
    t.add("assembly", 2.0)
    assert t.get("setup") == pytest.approx(1.5)
    assert t.total() == pytest.approx(3.5)
    d = t.to_dict()
    assert set(d) == {"setup", "assembly"}
    assert d["setup"] == pytest.approx(1.5)
    lines = t.to_lines()
    assert any("setup" in ln for ln in lines)


def test_timings_timed_context():
    t = zin.Timings()
    with t.timed("x"):
        pass
    assert t.get("x") >= 0.0
    assert "x" in t.to_dict()


# ---------------------------------------------------------------------------
# Ignored settings + GPU note
# ---------------------------------------------------------------------------

def test_ignored_settings_exact_and_prefix():
    z = SimpleNamespace(
        use_gpu_stack=True,
        use_gpu_grid=False,
        stack_use_gpu=True,
        intertile_affine_blend=True,
        center_out_normalization_p3=False,
        enable_poststack_anchor_review=True,
        two_pass_coverage_renorm=True,
        final_mosaic_dbe_enabled=True,
        final_mosaic_dbe_sigma=2.0,
        final_mosaic_dbe_iterations=0,  # falsy -> not reported
    )
    got = zin.ignored_settings_present(z)
    assert got["use_gpu_stack"] is True
    assert "use_gpu_grid" not in got  # False -> absent
    assert got["stack_use_gpu"] is True
    assert got["final_mosaic_dbe_enabled"] is True
    assert got["final_mosaic_dbe_sigma"] == 2.0
    assert "final_mosaic_dbe_iterations" not in got
    assert "center_out_normalization_p3" not in got


def test_ignored_settings_none_and_empty():
    assert zin.ignored_settings_present(None) == {}
    assert zin.ignored_settings_present(SimpleNamespace()) == {}


def test_gpu_note_is_explicit_cpu_only():
    note = zin.describe_gpu_usage()
    assert "CPU" in note
    assert "GPU" in note
    assert zin.GPU_USAGE_NOTE == note


def test_ignored_settings_warning_lines():
    lines = zin.ignored_settings_warning_lines({"use_gpu_stack": True})
    assert any("use_gpu_stack" in ln for ln in lines)
    empty = zin.ignored_settings_warning_lines({})
    assert len(empty) == 1


# ---------------------------------------------------------------------------
# Manifest: timings / gpu / ignored_settings blocks
# ---------------------------------------------------------------------------

def test_manifest_has_timings_gpu_ignored(tmp_path):
    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )

    class _Assembled:
        science = np.zeros((10, 10, 3), dtype=np.float32)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = ["r0000c0000"]
        incomplete_cells = []
        hole_pixels = 0
        coverage_pixels = 100

    descs = [
        zg.FrameDescriptor(
            frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(10, 10),
            wcs_header=_synthetic_tan_wcs().to_header().tostring(),
            header_sha256="", instrument="",
        )
    ]
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    timings = zin.Timings()
    timings.add("setup", 0.1)
    timings.add("cache_build", 0.2)

    _, _, mp = zz._write_outputs(
        _Assembled(), w, 1, 1, tmp_path, descs, {}, [],
        layout, zz.ExecutorConfig().science_config(), 0, {}, None,
        rejected=[], sip_mode="keep", frames_loaded=1,
        global_reference_frame_id="a.fits",
        timings=timings, gpu_used=False,
        ignored_settings={"use_gpu_stack": True},
    )
    m = json.loads(mp.read_text())
    assert m["timings"]["setup"] == pytest.approx(0.1)
    assert m["timings"]["cache_build"] == pytest.approx(0.2)
    assert m["gpu"]["used"] is False
    assert "CPU" in m["gpu"]["note"]
    assert m["ignored_settings"]["use_gpu_stack"] is True
    assert m["outputs"]["run_log"] == zz.RUN_LOG_NAME


# ---------------------------------------------------------------------------
# Run log written to the output folder
# ---------------------------------------------------------------------------

def test_run_log_written(tmp_path):
    timings = zin.Timings()
    timings.add("setup", 0.5)
    timings.add("assembly", 0.25)
    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=20, height=20, resolution_deg=0.001,
    )
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0,
              "layout_source": "auto"}
    zz._write_run_log(
        tmp_path, timings,
        start_ts="2026-10-07T00:00:00",
        frames_loaded=2, n_included=2, n_rejected=0,
        canvas=w, layout=layout, gpu_used=False,
        ignored_settings={"stack_use_gpu": True},
        peak_rss_kib=123, cache_info={"peak_cell_bytes": 5},
        global_reference_frame_id="a.fits",
    )
    log_path = tmp_path / zz.RUN_LOG_NAME
    assert log_path.exists()
    text = log_path.read_text()
    assert "Timings (wall-clock):" in text
    assert "setup:" in text
    assert "GPU usage:" in text
    assert "stack_use_gpu" in text


# ---------------------------------------------------------------------------
# R11 L1 reference provenance (bookkeeping placeholder labelling)
# ---------------------------------------------------------------------------

def test_reference_provenance_anchor():
    prov = zz._reference_provenance("R.fits", ["a.fits", "R.fits", "b.fits"], "R.fits")
    assert prov["reference_frame_role"] == "global_photometric_anchor"
    assert prov["bookkeeping_reference_frame_id"] is None


def test_reference_provenance_bookkeeping_placeholder():
    prov = zz._reference_provenance("R.fits", ["a.fits", "b.fits"], "b.fits")
    assert prov["reference_frame_role"] == "bookkeeping_placeholder"
    assert prov["bookkeeping_reference_frame_id"] == "b.fits"


# ---------------------------------------------------------------------------
# Worker count (memory-aware)
# ---------------------------------------------------------------------------

def test_choose_workers_bounded_and_memory_aware():
    assert zpar.choose_workers(None, 10 * 2**30) >= 1
    # Tight memory -> 1 worker (cannot fit even one ~340 MiB baseline).
    assert zpar.choose_workers(4, int(0.1 * 2**30)) == 1
    # Plenty of memory -> min(requested, cpu).
    cpu = int(np.ceil(zpar.choose_workers(100, 100 * 2**30)))
    assert zpar.choose_workers(4, 100 * 2**30) <= 4
    assert zpar.choose_workers(0, 100 * 2**30) == 1


def test_pmap_serial_and_order_preserving():
    results = zpar.pmap(_square, [1, 2, 3], workers=1)
    assert results == [1, 4, 9]
    results_par = zpar.pmap(_square, [1, 2, 3, 4], workers=2)
    assert results_par == [1, 4, 9, 16]


def test_gpu_note_mentions_cupy_product_init():
    # F3: the GPU note must answer the user's observation that loading "uses the
    # GPU" — the ZeGrid ENGINE is CPU-only, but the PRODUCT worker initialises CuPy.
    note = zin.describe_gpu_usage()
    assert "CPU" in note and "GPU" in note
    assert "CuPy" in note
    assert "zemosaic_worker" in note


def test_ignored_run_args_present_and_described():
    # F2: run_zegrid_mode accepts but ignores these stack/final-mosaic args.
    assert "save_final_as_uint16" in zin.IGNORED_RUN_ARGS
    assert "legacy_rgb_cube" in zin.IGNORED_RUN_ARGS
    assert "apply_radial_weight" in zin.IGNORED_RUN_ARGS
    assert "grid_rgb_equalize" in zin.IGNORED_RUN_ARGS
    assert "use_gpu" in zin.IGNORED_RUN_ARGS
    lines = zin.describe_ignored_run_args({"save_final_as_uint16": True})
    assert any("save_final_as_uint16" in ln for ln in lines)
    empty = zin.describe_ignored_run_args({})
    assert len(empty) == 1


def test_manifest_has_ignored_run_args(tmp_path):
    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )

    class _Assembled:
        science = np.zeros((10, 10, 3), dtype=np.float32)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = ["r0000c0000"]
        incomplete_cells = []
        hole_pixels = 0
        coverage_pixels = 100

    descs = [
        zg.FrameDescriptor(
            frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(10, 10),
            wcs_header=_synthetic_tan_wcs().to_header().tostring(),
            header_sha256="", instrument="",
        )
    ]
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    _, _, mp = zz._write_outputs(
        _Assembled(), w, 1, 1, tmp_path, descs, {}, [],
        layout, zz.ExecutorConfig().science_config(), 0, {}, None,
        rejected=[], sip_mode="keep", frames_loaded=1,
        global_reference_frame_id="a.fits",
        ignored_run_args={"save_final_as_uint16": True},
    )
    m = json.loads(mp.read_text())
    assert m["ignored_run_args"]["save_final_as_uint16"] is True


# ---------------------------------------------------------------------------
# F1: bounded gauge vs full-canvas gauge — NEAR-FULL-OVERLAP (M16) case
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M16 lights directory not present")
def test_bounded_gauge_near_full_overlap_matches_full_canvas(tmp_path):
    """M16 = NEAR-FULL-OVERLAP (bboxes ≈ canvas): bounded gauge matches the
    full-canvas gauge EXACTLY (array-equal). This is the case where the claim of
    exact equality is valid; partial overlap is covered separately (tolerance)."""
    descs, _ = zg.read_manifest(LIGHTS)
    descs = sorted(descs, key=lambda d: d.frame_id)[:6]
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    # OLD (R11): full-canvas aligned cache + compute_fixed_normalization.
    builder = zfp.AlignedCacheBuilder(str(tmp_path / "old_cache"))
    for f in descs:
        hwc = np.asarray(zz._decode_frame_hwc(f, NOOP), dtype=np.float32)
        if hwc.ndim == 2:
            hwc = np.stack([hwc, hwc, hwc], axis=-1)
        chw = np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
        rgb, geom = zxe.reproject_cropped(
            chw, f.wcs(), canvas.wcs(), (canvas.height, canvas.width)
        )
        builder.add(f.frame_id.logical_path, rgb, geom)
    builder.finish()
    prov = zfp.MemmapCanonicalProvider(str(tmp_path / "old_cache"))
    request = zstream.build_streaming_request(config, prov.n_frames)
    old = compute_fixed_normalization(prov, request)

    # NEW (R12): bounded streaming phase-1, no disk cache.
    new, ids = zphot.compute_global_gauge(descs, canvas, zz._gauge_decode, config, None, workers=4)

    assert int(old.reference_index) == int(new.reference_index)
    np.testing.assert_array_equal(old.coefficients, new.coefficients)
    np.testing.assert_array_equal(old.norm_active, new.norm_active)
    np.testing.assert_array_equal(old.weights, new.weights)
    np.testing.assert_array_equal(old.weight_active, new.weight_active)
    assert tuple((e.index, e.stage, e.reason, e.detail) for e in old.exclusions) == \
        tuple((e.index, e.stage, e.reason, e.detail) for e in new.exclusions)
    assert tuple(ids) == tuple(f.frame_id.logical_path for f in descs)


# ---------------------------------------------------------------------------
# F1: bounded gauge vs full-canvas gauge — PARTIAL-OVERLAP case (synthetic)
# ---------------------------------------------------------------------------

_DITHERED_DATA = {}


def _dithered_decode(f):
    """Module-level (picklable) decode for the synthetic dithered corpus."""
    return _DITHERED_DATA[f.frame_id.logical_path]


def _make_dithered_tan_wcs(h, w, ra_deg, dec_deg, scale_deg=0.001):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale_deg
    wcs.array_shape = (h, w)
    return wcs


def _build_dithered_corpus():
    """Synthetic TAN frames with a linear RA dither -> PARTIAL overlap.

    Frames are 48x48 px at 0.001 deg/px with a 6-px RA dither, so each frame's
    footprint bbox is strictly smaller than the canvas (partial overlap). Returns
    ``(descs, offsets)``.
    """
    h = w = 48
    scale = 0.001  # deg/px
    dither_deg = 0.006  # 6 px per frame
    offsets = [0.0, 5.0, 12.0, -4.0, 20.0, -9.0]
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 12.0**2))
    descs = []
    _DITHERED_DATA.clear()
    for i, off in enumerate(offsets):
        ra = 10.0 + i * dither_deg
        wcs = _make_dithered_tan_wcs(h, w, ra, 30.0, scale)
        data = (base + off + rng.normal(0.0, 0.5, (h, w))).astype(np.float32)
        hwc = np.stack([data, data, data], axis=-1)
        fid = f"d{i}.fits"
        _DITHERED_DATA[fid] = hwc
        descs.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid),
                source_path=f"/x/{fid}",
                shape_hw=(h, w),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return descs, offsets


def test_partial_overlap_gauge_equivalent_within_tolerance(tmp_path):
    """PARTIAL OVERLAP (bboxes < canvas): the bounded gauge's discrete outputs
    (reference index, active flags, exclusions) are IDENTICAL to the full-canvas
    gauge, and the continuous coefficients/weights agree within a TIGHT documented
    tolerance (~1e-4 reprojection FP noise, rtol<=1e-3)."""
    descs, offsets = _build_dithered_corpus()
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    # Sanity: partial overlap — each frame's bbox is strictly smaller than canvas.
    for f in descs:
        b = zphot._footprint_bbox(f, canvas)
        assert (b[1] - b[0]) < canvas.height or (b[3] - b[2]) < canvas.width

    # OLD (R11): full-canvas aligned cache + compute_fixed_normalization.
    builder = zfp.AlignedCacheBuilder(str(tmp_path / "old_cache"))
    for f in sorted(descs, key=lambda f: f.frame_id):
        hwc = np.asarray(_dithered_decode(f), dtype=np.float32)
        chw = np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
        rgb, geom = zxe.reproject_cropped(
            chw, f.wcs(), canvas.wcs(), (canvas.height, canvas.width)
        )
        builder.add(f.frame_id.logical_path, rgb, geom)
    builder.finish()
    prov = zfp.MemmapCanonicalProvider(str(tmp_path / "old_cache"))
    request = zstream.build_streaming_request(config, prov.n_frames)
    old = compute_fixed_normalization(prov, request)

    # NEW (R12): bounded streaming phase-1 (union bbox), no disk cache.
    new, ids = zphot.compute_global_gauge(descs, canvas, _dithered_decode, config, None, workers=1)

    # (a) discrete outputs IDENTICAL.
    assert int(old.reference_index) == int(new.reference_index) == 0
    np.testing.assert_array_equal(old.norm_active, new.norm_active)
    np.testing.assert_array_equal(old.weight_active, new.weight_active)
    assert tuple((e.index, e.stage, e.reason, e.detail) for e in old.exclusions) == \
        tuple((e.index, e.stage, e.reason, e.detail) for e in new.exclusions)

    # (b) continuous within TIGHT tolerance (~1e-4 FP noise, rtol<=1e-3).
    assert np.allclose(old.coefficients, new.coefficients, rtol=1e-3, atol=1e-3, equal_nan=True)
    assert np.allclose(old.weights, new.weights, rtol=1e-3, atol=1e-3, equal_nan=True)

    # Pin the numbers (deterministic seed 7). sky_mean slope a == 1.0 (identity);
    # the offsets are NOT exactly -offsets[i] because sky_mean is taken over the
    # PARTIAL common region (the spatial gradient contributes), so they are pinned
    # to the exact computed values.
    pinned_offsets = {
        1: -5.0483836942865,
        2: -12.106878870411919,
        3: 3.623317057887718,
        4: -20.48067739863454,
        5: 8.453891751070742,
    }
    for i, expected in pinned_offsets.items():
        assert new.coefficients[i, 0, 0] == pytest.approx(1.0, abs=1e-6)
        assert new.coefficients[i, 0, 1] == pytest.approx(expected, abs=1e-3)
    pinned_weights = [
        0.9975676058190871, 1.0, 0.9994368570190159,
        0.9948389940794012, 0.9989957281457885, 0.9980533119874564,
    ]
    assert np.allclose(new.weights, pinned_weights, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Parallel cache build is BIT-EQUAL to serial (gated on M16)
# ---------------------------------------------------------------------------

def _dir_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob("*")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M16 lights directory not present")
def test_parallel_cache_bit_equal_serial(tmp_path):
    descs, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(descs)
    nx, ny = 3, 3
    cell_ctxs = []
    for row, col, _b in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        cell_ctxs.append((row, col, cell, patch, mem))
    nonempty = [(c, p, m) for (r, co, c, p, m) in cell_ctxs if m.patch_ids]
    cell, patch, mem = max(nonempty, key=lambda t: len(t[2].patch_ids))

    s_dir = tmp_path / "serial"
    p_dir = tmp_path / "par"
    m1 = zz._build_one_cell_cache(descs, canvas, cell, patch, mem, s_dir, workers=1, reuse_cache=False)
    m2 = zz._build_one_cell_cache(descs, canvas, cell, patch, mem, p_dir, workers=4, reuse_cache=False)

    assert m1["n_frames"] == m2["n_frames"]
    assert m1["frame_ids"] == m2["frame_ids"]
    assert _dir_hash(s_dir) == _dir_hash(p_dir)
    # Determinism: the parallel build is itself repeatable (same hash twice).
    m3 = zz._build_one_cell_cache(descs, canvas, cell, patch, mem, tmp_path / "par2",
                                  workers=4, reuse_cache=False)
    assert _dir_hash(p_dir) == _dir_hash(tmp_path / "par2")
    assert m3["frame_ids"] == m2["frame_ids"]
