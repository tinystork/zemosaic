"""ZM-FITS-INTEROP-R26 targeted tests — Linux FITS interoperability.

Product decision (Tristan, supersedes R23's on-disk NaN requirement):
final float RGB science/aesthetic FITS must open in ASIFitsView Linux and
Gwenview. Those readers fail on non-finite (NaN) float samples and on the
ZeGrid coverage's signed-int32 (BITPIX=32) representation.

Contract implemented here:

* At the FINAL SERIALIZATION BOUNDARY only, non-finite float32 samples
  (NaN/+Inf/-Inf) are replaced with ``0.0`` in a copy/view-safe export buffer.
  In-memory science/aesthetic arrays are never mutated; finite samples are
  bit-identical to the pre-sanitization values.
* No-data semantics stay in the coverage / ALPHA products (never removed).
* The replacement is auditable: header cards ``ZNFILL`` (fill value) +
  ``ZNFREPL`` (replaced count) and a HISTORY line; the manifest records
  ``nonfinite_replaced`` / ``nonfinite_fill``.
* ZeGrid ``mosaic_grid_coverage.fits`` is serialized as contiguous float32
  (BITPIX=-32, no BSCALE/BZERO) with the exact pre-fix integer stack-depth
  values.
* Classic/legacy final science + aesthetic call sites opt in via
  ``sanitize_nonfinite=True``; Classic coverage remains float32 unchanged.

These tests are fast-tier and self-contained (temp outputs only).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_utils as zu
from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import geometry as zg


# ---------------------------------------------------------------------------
# 1. Helper unit tests (sanitize_nonfinite_float32 / record_nonfinite_fill)
# ---------------------------------------------------------------------------

def _rgb_hwc(n=4):
    return np.arange(n * n * 3, dtype=np.float32).reshape(n, n, 3)


def test_sanitize_hwc_nan_inf_posinf():
    arr = _rgb_hwc(4)
    arr[0, 0, 0] = np.nan
    arr[1, 1, 1] = np.inf
    arr[2, 2, 2] = -np.inf
    orig = arr.copy()

    out, n = zu.sanitize_nonfinite_float32(arr, fill_value=0.0)

    assert n == 3
    assert np.all(np.isfinite(out))
    assert out[0, 0, 0] == 0.0
    assert out[1, 1, 1] == 0.0
    assert out[2, 2, 2] == 0.0
    # Finite values bit-identical.
    finite = np.isfinite(orig)
    assert np.array_equal(out[finite], orig[finite])
    # Input never mutated.
    assert np.array_equal(arr, orig, equal_nan=True)
    assert np.isnan(arr[0, 0, 0]) and np.isinf(arr[1, 1, 1])


def test_sanitize_chw_rgb():
    arr = _rgb_hwc(4)
    chw = np.ascontiguousarray(np.moveaxis(arr, -1, 0))
    chw[0, 0, 0] = np.nan
    chw[2, 3, 1] = np.inf
    orig = chw.copy()

    out, n = zu.sanitize_nonfinite_float32(chw, fill_value=0.0)

    assert n == 2
    assert np.all(np.isfinite(out))
    finite = np.isfinite(orig)
    assert np.array_equal(out[finite], orig[finite])
    assert np.array_equal(chw, orig, equal_nan=True)


def test_sanitize_2d_float():
    arr = np.linspace(-5, 5, 16, dtype=np.float32).reshape(4, 4)
    arr[0, 0] = np.nan
    arr[3, 3] = np.inf
    orig = arr.copy()

    out, n = zu.sanitize_nonfinite_float32(arr, fill_value=0.0)

    assert n == 2
    assert np.all(np.isfinite(out))
    finite = np.isfinite(orig)
    assert np.array_equal(out[finite], orig[finite])


def test_sanitize_no_nonfinite_unchanged():
    arr = _rgb_hwc(4)
    out, n = zu.sanitize_nonfinite_float32(arr, fill_value=0.0)
    assert n == 0
    assert np.array_equal(out, arr)


def test_record_nonfinite_fill_metadata():
    hdr = fits.Header()
    zu.record_nonfinite_fill(hdr, 7, fill_value=0.0)
    assert hdr["ZNFILL"] == 0.0
    assert hdr["ZNFREPL"] == 7
    assert any("replaced 7 non-finite" in str(h) for h in hdr["HISTORY"])


def test_record_nonfinite_fill_noop_when_zero():
    hdr = fits.Header()
    zu.record_nonfinite_fill(hdr, 0, fill_value=0.0)
    assert "ZNFILL" not in hdr
    assert "ZNFREPL" not in hdr


# ---------------------------------------------------------------------------
# 2. ZeGrid output tests through the real writer functions
# ---------------------------------------------------------------------------

def _synthetic_wcs(shape=(8, 8)):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


def _canvas(shape=(8, 8)):
    return zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_wcs(shape).to_header().tostring(),
        width=shape[1], height=shape[0], resolution_deg=0.001,
    )


class _Assembled:
    def __init__(self):
        h = w = 8
        self.science = np.linspace(10, 200, h * w * 3, dtype=np.float32).reshape(h, w, 3)
        # Inject non-finite holes (uncovered pixels).
        self.science[0, 0, :] = np.nan
        self.science[7, 7, 0] = np.inf
        self.stack_depth = np.arange(h * w, dtype=np.int32).reshape(h, w) % 47  # 0..46
        self.complete_cells = 1
        self.incomplete_cells = 0
        self.hole_pixels = 1
        self.coverage_pixels = h * w - 1


def _layout():
    return {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
            "max_contributors": 1, "predicted_bound_bytes": 0}


def _descs():
    return [
        zg.FrameDescriptor(
            frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(8, 8),
            wcs_header=_synthetic_wcs().to_header().tostring(),
            header_sha256="", instrument="",
        )
    ]


def test_zegrid_raw_science_and_aesthetic_all_finite_on_disk(tmp_path):
    canvas = _canvas()
    a = _Assembled()
    raw_path = tmp_path / "mosaic_grid_science.fits"
    aest_path = tmp_path / "mosaic_grid_aesthetic.fits"
    zz._write_raw_science_fits(a, canvas, raw_path, related_file=aest_path.name)
    zz._write_outputs(
        a, canvas, 1, 1, tmp_path, _descs(), {}, [],
        _layout(), SimpleNamespace(normalization="sky_mean"), 0, {}, None,
        finished_science=a.science, finishing_info={"enabled": False},
        raw_science=a.science, raw_science_path=raw_path,
        aesthetic_path=aest_path, export_aesthetic_fits=True,
    )

    for p in (raw_path, aest_path):
        assert p.exists()
        with fits.open(p) as h:
            d = h[0].data
            assert np.all(np.isfinite(np.asarray(d)))
            assert np.asarray(d).dtype.itemsize == 4  # float32
            assert h[0].header["BITPIX"] == -32
    # In-memory science still carries its NaNs (not mutated).
    assert np.isnan(a.science[0, 0, 0])


def test_zegrid_single_raw_export_off(tmp_path):
    """export OFF -> single ``mosaic_grid.fits`` fallback, finite on disk."""
    canvas = _canvas()
    a = _Assembled()
    zz._write_raw_science_fits(a, canvas, tmp_path / "mosaic_grid.fits")
    with fits.open(tmp_path / "mosaic_grid.fits") as h:
        d = np.asarray(h[0].data)
        assert np.all(np.isfinite(d))
        assert h[0].header["SCIROLE"] == "SCI"
        assert h[0].header["ZNFREPL"] >= 2


def test_zegrid_coverage_float32_bitpix_minus_32(tmp_path):
    canvas = _canvas()
    a = _Assembled()
    raw_path = tmp_path / "mosaic_grid.fits"
    zz._write_raw_science_fits(a, canvas, raw_path)
    zz._write_outputs(
        a, canvas, 1, 1, tmp_path, _descs(), {}, [],
        _layout(), SimpleNamespace(normalization="sky_mean"), 0, {}, None,
        raw_science=a.science, raw_science_path=raw_path,
        aesthetic_path=None, export_aesthetic_fits=False,
    )
    cov = tmp_path / "mosaic_grid_coverage.fits"
    assert cov.exists()
    with fits.open(cov) as h:
        assert h[0].header["BITPIX"] == -32
        assert "BSCALE" not in h[0].header
        assert "BZERO" not in h[0].header
        d = np.asarray(h[0].data)
        assert d.dtype.itemsize == 4
        assert np.all(np.isfinite(d))
        # Exact pre-fix integer values.
        assert np.array_equal(d, a.stack_depth.astype(np.float32))
        assert h[0].header["BUNIT"] == "count"
    # Manifest records float32 coverage + nonfinite audit on raw.
    import json
    m = json.loads((tmp_path / "zegrid_manifest.json").read_text())
    assert m["science_output_contract"]["raw"]["nonfinite_replaced"] >= 2
    assert m["science_output_contract"]["raw"]["nonfinite_fill"] == 0.0


def test_zegrid_manifest_preserves_suffix_references(tmp_path):
    canvas = _canvas()
    a = _Assembled()
    raw_path = tmp_path / "mosaic_grid_sc.fits"
    aest_path = tmp_path / "mosaic_grid_ae.fits"
    zz._write_raw_science_fits(a, canvas, raw_path, related_file=aest_path.name)
    zz._write_outputs(
        a, canvas, 1, 1, tmp_path, _descs(), {}, [],
        _layout(), SimpleNamespace(normalization="sky_mean"), 0, {}, None,
        finished_science=a.science, finishing_info={"enabled": False},
        raw_science=a.science, raw_science_path=raw_path,
        aesthetic_path=aest_path, export_aesthetic_fits=True,
        scientific_fits_suffix="_sc",
    )
    import json
    m = json.loads((tmp_path / "zegrid_manifest.json").read_text())
    assert m["outputs"]["science"] == "mosaic_grid_sc.fits"
    assert m["outputs"]["aesthetic"] == "mosaic_grid_ae.fits"
    assert m["science_reference"] == "mosaic_grid_sc.fits"


# ---------------------------------------------------------------------------
# 3. Classic/legacy production-path propagation (both final-save call sites)
# ---------------------------------------------------------------------------

def _save_fits_calls_in_function(func_name):
    """Return the list of ``save_fits_image(...)`` call blocks in a worker
    function, proving the real call sites opt into ``sanitize_nonfinite=True``.
    """
    from zemosaic import zemosaic_worker as zw
    src = inspect.getsource(getattr(zw, func_name))
    # Find each save_fits_image(...) call block (balanced-paren scan).
    calls = []
    idx = 0
    needle = "save_fits_image("
    while True:
        idx = src.find(needle, idx)
        if idx == -1:
            break
        start = idx
        depth = 0
        j = src.find("(", idx)
        while j < len(src):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        calls.append(src[start:j + 1])
        idx = j + 1
    return calls


def test_classic_legacy_both_science_sites_opt_in():
    """Both duplicated final science save sites pass sanitize_nonfinite=True."""
    sci_calls = _save_fits_calls_in_function("run_hierarchical_mosaic_classic_legacy")
    hier_calls = _save_fits_calls_in_function("run_hierarchical_mosaic")

    # Science call site in classic_legacy (output_path=str(science_fits_path)).
    sci_site = [c for c in sci_calls if "science_fits_path" in c]
    assert sci_site, "classic_legacy science save site not found"
    assert "sanitize_nonfinite=True" in sci_site[0]
    assert "save_as_float=True" in sci_site[0]

    # Science call site in hierarchical (output_path=str(final_fits_path)).
    hier_sci = [c for c in hier_calls if "final_fits_path" in c and "coverage" not in c]
    assert hier_sci, "hierarchical science save site not found"
    assert "sanitize_nonfinite=True" in hier_sci[0]
    assert "save_as_float=True" in hier_sci[0]


def test_classic_legacy_aesthetic_site_opts_in():
    """The Classic/legacy aesthetic save site opts into sanitization."""
    sci_calls = _save_fits_calls_in_function("run_hierarchical_mosaic_classic_legacy")
    aest_site = [c for c in sci_calls if "aesthetic_fits_path" in c]
    assert aest_site, "aesthetic save site not found"
    assert "sanitize_nonfinite=True" in aest_site[0]


def test_classic_coverage_unchanged_float32():
    """Classic coverage save sites remain float32 (no sanitize opt-in)."""
    sci_calls = _save_fits_calls_in_function("run_hierarchical_mosaic_classic_legacy")
    hier_calls = _save_fits_calls_in_function("run_hierarchical_mosaic")
    for c in sci_calls + hier_calls:
        if "coverage" in c and "save_as_float=True" in c:
            assert "sanitize_nonfinite=True" not in c


def test_save_fits_image_opt_in_default_false():
    """The opt-in defaults False so unrelated FITS writes are unchanged."""
    sig = inspect.signature(zu.save_fits_image)
    assert sig.parameters["sanitize_nonfinite"].default is False
