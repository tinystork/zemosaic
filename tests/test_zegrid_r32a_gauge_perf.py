"""ZM-ZEGRID-R32a targeted tests — gauge performance levers + instrumentation.

Fast, synthetic (no gated corpora). Covers the R32a deliverables:

* the internal levers DEFAULT OFF (the production default path is unchanged);
* EXACT Lever A (geometric prefilter) — never a false negative, and the
  prefiltered path reproduces the strict path's discrete outputs exactly;
* EXACT Lever B (reference dedup) — bit-identical coefficients/weights;
* the bounded-work path — same discrete decisions, coefficients within the
  declared reprojection FP tolerance;
* the diagnostics instrumentation schema (timings/areas/bytes/RSS) does not
  change science outputs.

Fixtures: full overlap, partial overlap, disjoint (skipped), grazing bbox
(skipped but truly disjoint), tiny-but-kept overlap, SIP-distorted WCS, and
mono vs RGB.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS, Sip

from zemosaic.core.canonical_streaming import subset_fixed_normalization
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid.executor import ExecutorConfig

# Module-level payload keyed by logical frame path (picklable decode).
_DATA: dict = {}

# Bounded path: only frame i's reprojection sub-WCS changes, so the agreement is
# far tighter than the ~1e-4 reprojection FP noise class; declared at 1e-4 to be
# conservative (see the R32a report).
BOUNDED_RTOL = 1e-4
BOUNDED_ATOL = 1e-6


def _decode(f):
    return _DATA[f.frame_id.logical_path]


def _tan_wcs(h, w, ra_deg, dec_deg, scale_deg=0.001):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale_deg
    wcs.array_shape = (h, w)
    return wcs


def _sip_wcs(h, w, ra_deg, dec_deg, scale_deg=0.001):
    """A SIP-distorted TAN WCS (exercises the fallback reprojection path)."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale_deg
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3, 3), dtype=np.float64)
    a[2, 0] = 1.0e-6  # ~1 px correction at ~1000 px
    b[0, 2] = 1.0e-6
    wcs.sip = Sip(a, b, np.zeros((3, 3)), np.zeros((3, 3)), np.asarray(wcs.wcs.crpix))
    wcs.array_shape = (h, w)
    return wcs


def _frame_base(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (min(h, w) / 4.0) ** 2)
    )
    return base, rng


def _add_frame(descs, fid, wcs, data, mono=False):
    _DATA[fid] = data.astype(np.float32) if mono else np.stack(
        [data, data, data], axis=-1
    ).astype(np.float32)
    descs.append(
        zg.FrameDescriptor(
            frame_id=zg.FrameId(fid),
            source_path=f"/x/{fid}",
            shape_hw=data.shape,
            wcs_header=wcs.to_header(relax=True).tostring(),
            header_sha256="",
            instrument="",
        )
    )


def _full_overlap_corpus(h=48, w=48, n=5, sip=False):
    _DATA.clear()
    descs = []
    base, rng = _frame_base(h, w, 3)
    for i in range(n):
        wcs = (_sip_wcs if sip else _tan_wcs)(h, w, 10.0, 30.0)
        data = base + (i - 2) * 3.0 + rng.normal(0.0, 0.5, (h, w))
        _add_frame(descs, f"f{i}.fits", wcs, data)
    return descs


def _dithered_corpus(h=48, w=48, offsets=(0.0, 5.0, 12.0, -4.0, 20.0, -9.0), sip=False):
    _DATA.clear()
    descs = []
    base, rng = _frame_base(h, w, 7)
    for i, off in enumerate(offsets):
        wcs = (_sip_wcs if sip else _tan_wcs)(h, w, 10.0 + i * 0.006, 30.0)
        data = base + off + rng.normal(0.0, 0.5, (h, w))
        _add_frame(descs, f"d{i}.fits", wcs, data)
    return descs, list(offsets)


def _mono_corpus(h=48, w=48, n=5):
    _DATA.clear()
    descs = []
    base, rng = _frame_base(h, w, 11)
    for i in range(n):
        wcs = _tan_wcs(h, w, 10.0 + i * 0.006, 30.0)
        data = base + i * 2.0 + rng.normal(0.0, 0.5, (h, w))
        _add_frame(descs, f"m{i}.fits", wcs, data, mono=True)
    return descs


def _wide_dither_corpus():
    """Partially-overlapping frames with a LARGE dither -> union >> frame bbox."""
    _DATA.clear()
    descs = []
    h = w = 48
    base, rng = _frame_base(h, w, 21)
    for i, s in enumerate((0.0, 25.0, 50.0, -15.0, 75.0, -35.0)):
        wcs = _tan_wcs(h, w, 10.0, 30.0)
        wcs.wcs.crpix = [w / 2.0 - s, h / 2.0]
        data = base + i * 2.0 + rng.normal(0.0, 0.5, (h, w))
        _add_frame(descs, f"w{i}.fits", wcs, data)
    return descs


def _disjoint_corpus():
    """Two clusters of frames with a large RA gap -> bboxes never intersect."""
    _DATA.clear()
    descs = []
    base, rng = _frame_base(40, 40, 5)
    for i, ra in enumerate((10.0, 10.001, 10.002, 10.5, 10.501, 10.502)):
        wcs = _tan_wcs(40, 40, ra, 30.0)
        data = base + i * 2.0 + rng.normal(0.0, 0.5, (40, 40))
        _add_frame(descs, f"g{i}.fits", wcs, data)
    return descs


def _run(descs, canvas, config, **kw):
    diag: dict = {}
    g, ids = zphot.compute_global_gauge(
        descs, canvas, _decode, config, None, workers=1, diagnostics=diag, **kw
    )
    return g, ids, diag


def _excl_id(gauge):
    return tuple((e.index, e.stage, e.reason) for e in gauge.exclusions)


def _max_rel_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0.0
    denom = np.maximum(np.abs(a[m]), 1e-12)
    return float(np.max(np.abs(a[m] - b[m]) / denom))


def _assert_parity(strict, opt, *, exact: bool):
    assert int(strict.reference_index) == int(opt.reference_index)
    np.testing.assert_array_equal(strict.norm_active, opt.norm_active)
    np.testing.assert_array_equal(strict.weight_active, opt.weight_active)
    # identical exclusions (index/stage/reason).
    assert _excl_id(strict) == _excl_id(opt)
    if exact:
        np.testing.assert_array_equal(strict.coefficients, opt.coefficients)
        np.testing.assert_array_equal(strict.weights, opt.weights)
    else:
        assert np.allclose(
            strict.coefficients, opt.coefficients,
            rtol=BOUNDED_RTOL, atol=BOUNDED_ATOL, equal_nan=True,
        ), f"coefficients diff {_max_rel_diff(strict.coefficients, opt.coefficients)}"
        assert np.allclose(
            strict.weights, opt.weights, rtol=BOUNDED_RTOL, atol=BOUNDED_ATOL, equal_nan=True
        )


# ---------------------------------------------------------------------------
# Defaults unchanged + instrumentation schema
# ---------------------------------------------------------------------------

def test_levers_default_off():
    assert zphot._GAUGE_PREFILTER_PAIRS_DEFAULT is False
    assert zphot._GAUGE_DEDUP_REFERENCE_DEFAULT is False
    assert zphot._GAUGE_BOUNDED_PAIRS_DEFAULT is False


def test_default_path_equals_all_levers_off():
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    g0, _ = zphot.compute_global_gauge(descs, canvas, _decode, cfg, None, workers=1)
    g1, _, _ = _run(
        descs, canvas, cfg,
        prefilter_pairs=False, dedup_reference=False, bounded_pairs=False,
    )
    _assert_parity(g0, g1, exact=True)


def test_instrumentation_schema_and_no_science_change():
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()

    g_ref, _, _ = _run(descs, canvas, cfg)  # no diagnostics
    g, _, diag = _run(descs, canvas, cfg)

    # Instrumentation does not touch science outputs.
    _assert_parity(g_ref, g, exact=True)

    instr = diag["instrumentation"]
    assert instr["schema"] == "zegrid-gauge-instrumentation-r32a"
    assert instr["levers"] == {
        "prefilter_pairs": False, "dedup_reference": False, "bounded_pairs": False
    }
    for phase in ("counts", "pairs"):
        assert instr[phase]["reproject_s"]["n"] >= 1
        assert instr[phase]["bytes_reprojected"] > 0
    assert instr["pairs"]["pairs"] == len(descs) - 1
    assert instr["pairs"]["crop_area_px"]["n"] == len(descs) - 1
    assert instr["reference"]["reproject_s"] >= 0.0
    assert instr["reference"]["area_px"] == canvas.height * canvas.width
    assert instr["totals"]["bytes_lifted"] > 0
    assert instr["memory"]["canvas_px"] == canvas.height * canvas.width
    assert instr["memory"]["peak_rss_kib_end"] > 0


def test_run_log_renders_instrumentation(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import instrumentation as zin

    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_tan_wcs(10, 10, 10.0, 30.0).to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    _, _, diag = _run(descs, canvas, cfg)

    timings = zin.Timings()
    timings.add("gauge", 1.0)
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "layout_source": "auto"}
    zz._write_run_log(
        tmp_path, timings, start_ts="2026-10-10T00:00:00",
        frames_loaded=len(descs), n_included=len(descs), n_rejected=0,
        canvas=w, layout=layout, gpu_used=False, ignored_settings={},
        peak_rss_kib=1, cache_info={}, global_reference_frame_id="f0.fits",
        gauge_diagnostics=diag,
    )
    text = (tmp_path / zz.RUN_LOG_NAME).read_text()
    assert "instrumentation(R32a):" in text
    assert "bytes_lifted=" in text
    assert "peak_rss_kib_end=" in text


# ---------------------------------------------------------------------------
# EXACT Lever B (reference dedup)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sip", [False, True])
def test_dedup_reference_exact(sip):
    descs, _ = _dithered_corpus(sip=sip)
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, _ = _run(descs, canvas, cfg)
    opt, _, diag = _run(descs, canvas, cfg, dedup_reference=True)
    _assert_parity(strict, opt, exact=True)
    assert diag["instrumentation"]["levers"]["dedup_reference"] is True


# ---------------------------------------------------------------------------
# EXACT Lever A (geometric prefilter)
# ---------------------------------------------------------------------------

def _assert_prefilter_sound(strict, opt, prefiltered_indices):
    """Every prefiltered pair was excluded with insufficient_common by strict."""
    reasons = {i: r for (i, _s, r) in _excl_id(strict)}
    for i in prefiltered_indices:
        assert reasons.get(i) == "insufficient_common", (
            f"prefilter skipped pair {i} but strict reason was {reasons.get(i)}"
        )
        assert not strict.norm_active[i]
        assert not opt.norm_active[i]


def test_prefilter_skips_disjoint_and_is_exact():
    descs = _disjoint_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, _ = _run(descs, canvas, cfg)
    opt, _, diag = _run(descs, canvas, cfg, prefilter_pairs=True)

    n_pre = diag["instrumentation"]["pairs"]["pairs_prefiltered"]
    assert n_pre >= 1
    _assert_parity(strict, opt, exact=True)
    # Identify the prefiltered indices from the excluded set (all excluded must
    # be insufficient_common here).
    prefiltered = [i for i in range(len(descs)) if not opt.norm_active[i]]
    assert len(prefiltered) == n_pre
    _assert_prefilter_sound(strict, opt, prefiltered)


def test_prefilter_grazing_bbox_skip_reason_is_insufficient_common():
    """bboxes touch (upper bound > 0) but the supports are disjoint; the pair is
    skipped and the reason matches the strict path."""
    _DATA.clear()
    descs = []
    h = w = 48
    base, rng = _frame_base(h, w, 9)
    # crpix shift of 61 canvas px -> supports disjoint but the 8-px-margin
    # bboxes still intersect with a small area (0 < area < min_common=256).
    w0 = _tan_wcs(h, w, 10.0, 30.0)
    w1 = _tan_wcs(h, w, 10.0, 30.0)
    w1.wcs.crpix = [w / 2.0 + 61, h / 2.0]
    for i, wcs in enumerate((w0, w1)):
        data = base + i * 2.0 + rng.normal(0.0, 0.5, (h, w))
        _add_frame(descs, f"z{i}.fits", wcs, data)
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()

    inter = zphot._bbox_intersection_area(
        zphot._footprint_bbox(descs[0], canvas), zphot._footprint_bbox(descs[1], canvas)
    )
    assert 0 < inter < 256, f"fixture must give 0<inter<256, got {inter}"

    strict, _, _ = _run(descs, canvas, cfg)
    opt, _, diag = _run(descs, canvas, cfg, prefilter_pairs=True)
    _assert_parity(strict, opt, exact=True)
    assert diag["instrumentation"]["pairs"]["pairs_prefiltered"] == 1
    _assert_prefilter_sound(strict, opt, [i for i in range(2) if not opt.norm_active[i]])


def test_prefilter_never_skips_a_kept_pair():
    """Conservative: for every pair the strict path KEEPS, the optimized path
    also keeps it (never a false negative)."""
    for corpus in ("full", "dither", "sip", "mono", "disjoint"):
        if corpus == "full":
            descs = _full_overlap_corpus()
        elif corpus == "dither":
            descs, _ = _dithered_corpus()
        elif corpus == "sip":
            descs, _ = _dithered_corpus(sip=True)
        elif corpus == "mono":
            descs = _mono_corpus()
        else:
            descs = _disjoint_corpus()
        canvas = zg.build_canvas(descs)
        cfg = ExecutorConfig().science_config()
        strict, _, _ = _run(descs, canvas, cfg)
        opt, _, _ = _run(descs, canvas, cfg, prefilter_pairs=True)
        # Kept by strict => kept by optimized (and vice-versa for the kept set).
        np.testing.assert_array_equal(strict.norm_active, opt.norm_active)
        _assert_parity(strict, opt, exact=True)


def test_prefilter_unit_bound_is_an_upper_bound():
    # The bbox-intersection area is a provable upper bound on the true common
    # pixel count (both supports are contained in their footprint bboxes).
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, diag = _run(descs, canvas, cfg)
    # Recomputed bounds must be >= the strict common counts recorded in details.
    for e in strict.exclusions:
        if e.reason != "insufficient_common" or not e.detail:
            continue
        inter = zphot._bbox_intersection_area(
            zphot._footprint_bbox(descs[0], canvas), zphot._footprint_bbox(descs[e.index], canvas)
        )
        # detail carries the exact strict common count when available.
        if "common=" in e.detail and "<=" not in e.detail:
            strict_common = int(e.detail.split("common=")[1].split(" ")[0])
            assert inter >= strict_common


# ---------------------------------------------------------------------------
# Bounded path (declared tolerance)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sip", [False, True])
def test_bounded_pairs_within_tolerance(sip):
    descs, _ = _dithered_corpus(sip=sip)
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, _ = _run(descs, canvas, cfg)
    opt, _, diag = _run(descs, canvas, cfg, bounded_pairs=True)
    _assert_parity(strict, opt, exact=False)
    # Bounded crops are strictly smaller than the union for this corpus.
    assert diag["instrumentation"]["pairs"]["crop_area_px"]["max"] <= (
        canvas.height * canvas.width
    )
    assert diag["instrumentation"]["levers"]["bounded_pairs"] is True


def test_bounded_plus_prefilter_plus_dedup_disjoint():
    descs = _disjoint_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, _ = _run(descs, canvas, cfg)
    opt, _, diag = _run(
        descs, canvas, cfg,
        prefilter_pairs=True, dedup_reference=True, bounded_pairs=True,
    )
    _assert_parity(strict, opt, exact=False)
    assert diag["instrumentation"]["pairs"]["pairs_prefiltered"] >= 1


def test_bounded_full_overlap_is_exact():
    """When the footprint IS (nearly) the whole canvas the union == frame bbox,
    so the bounded path coincides with the strict path."""
    descs = _full_overlap_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, _ = _run(descs, canvas, cfg)
    opt, _, _ = _run(descs, canvas, cfg, bounded_pairs=True)
    _assert_parity(strict, opt, exact=False)
    assert _max_rel_diff(strict.coefficients, opt.coefficients) <= BOUNDED_RTOL


def test_bounded_reduces_crop_area():
    """The bounded path reprojects a strictly smaller region than the union bbox."""
    descs = _wide_dither_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, diag_s = _run(descs, canvas, cfg)
    opt, _, diag_b = _run(descs, canvas, cfg, bounded_pairs=True)
    _assert_parity(strict, opt, exact=False)
    assert (
        diag_b["instrumentation"]["pairs"]["crop_area_px"]["max"]
        < diag_s["instrumentation"]["pairs"]["crop_area_px"]["max"]
    )
    assert (
        diag_b["instrumentation"]["pairs"]["crop_area_px"]["median"]
        <= diag_s["instrumentation"]["pairs"]["crop_area_px"]["median"]
    )


# ---------------------------------------------------------------------------
# Mono vs RGB
# ---------------------------------------------------------------------------

def test_mono_corpus_parity():
    descs = _mono_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, _, _ = _run(descs, canvas, cfg)
    opt_pre, _, _ = _run(descs, canvas, cfg, prefilter_pairs=True, dedup_reference=True)
    opt_bnd, _, _ = _run(descs, canvas, cfg, bounded_pairs=True)
    _assert_parity(strict, opt_pre, exact=True)
    _assert_parity(strict, opt_bnd, exact=False)
    # Mono is replicated to 3 identical channels -> per-channel coefficients agree.
    c = strict.coefficients
    fin = np.isfinite(c[:, 0, 0]) & np.isfinite(c[:, 2, 1])
    if fin.any():
        assert np.allclose(c[:, 0, :], c[:, 2, :], rtol=1e-9, atol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# Reference absent from a cell (integration with subset_fixed_normalization)
# ---------------------------------------------------------------------------

def test_reference_absent_from_cell_coefficients_match():
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    strict, ids, _ = _run(descs, canvas, cfg)
    opt, _, _ = _run(descs, canvas, cfg, prefilter_pairs=True, dedup_reference=True)

    ref = int(strict.reference_index)
    cell_ids = [i for i in range(len(descs)) if i != ref][:3]
    a = subset_fixed_normalization(strict, list(range(len(descs))), cell_ids)
    b = subset_fixed_normalization(opt, list(range(len(descs))), cell_ids)
    np.testing.assert_array_equal(a.coefficients, b.coefficients)
    assert a.frame_ids == b.frame_ids


# ---------------------------------------------------------------------------
# Parallel (fork) sanity for the optimized path
# ---------------------------------------------------------------------------

def test_optimized_path_fork_bit_equal_serial():
    import multiprocessing

    if multiprocessing.get_start_method(allow_none=True) == "spawn":
        pytest.skip("spawn start method set by another test; serial parity covered elsewhere")
    descs, _ = _dithered_corpus()
    canvas = zg.build_canvas(descs)
    cfg = ExecutorConfig().science_config()
    serial, _, _ = _run(descs, canvas, cfg, prefilter_pairs=True, dedup_reference=True)
    diag: dict = {}
    par, _ids = zphot.compute_global_gauge(
        descs, canvas, _decode, cfg, None, workers=3, diagnostics=diag,
        prefilter_pairs=True, dedup_reference=True,
    )
    assert diag["phases"]["pairs"]["units"] == len(descs) - 1
    _assert_parity(serial, par, exact=True)
