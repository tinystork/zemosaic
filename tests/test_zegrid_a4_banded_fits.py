"""ZM-ZEGRID-ADAPTIVE-RESOURCES A4 — banded streaming FITS serialization.

Q40 support: the banded writer must be BYTE-IDENTICAL to the dense
``fits.PrimaryHDU(np.ascontiguousarray(np.moveaxis(science, -1, 0))).writeto``
path the product has always used, while never materialising the full CHW cube.
It must also reproduce the exact ``SCIHASH`` (sha256 of the sanitized
little-endian CHW bytes) and the non-finite replacement count.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import banded_fits as zb  # noqa: E402

H, W, C = 24, 32, 3


def _wcs_header():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [W // 2, H // 2]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.cdelt = [-1e-4, 1e-4]
    hdr = w.to_header(relax=True)
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H
    hdr["NAXIS3"] = C
    hdr["SCIROLE"] = ("SCI", "science output role")
    return hdr


def _dense_reference(hwc):
    """The exact bytes the product's dense path produces."""
    raw = np.ascontiguousarray(np.moveaxis(np.asarray(hwc, dtype=np.float32), -1, 0))
    finite = np.isfinite(raw)
    replaced = int(raw.size - int(np.count_nonzero(finite)))
    if replaced:
        raw = raw.copy()
        raw[~finite] = np.float32(0.0)
    sha = hashlib.sha256(raw.tobytes()).hexdigest()
    hdr = _wcs_header()
    hdr["SCIHASH"] = (sha, "SHA-256")
    buf = io.BytesIO()
    fits.PrimaryHDU(raw, header=hdr).writeto(buf)
    return buf.getvalue(), sha, replaced


def test_a4_banded_science_fits_is_byte_identical(tmp_path):
    rng = np.random.default_rng(0)
    hwc = rng.standard_normal((H, W, C)).astype(np.float32)
    hwc[1, 2, 0] = np.nan
    hwc[3, 4, 1] = np.inf
    hwc[5, 6, 2] = -np.inf

    dense_bytes, sha, replaced = _dense_reference(hwc)

    sha2, rep2 = zb.science_fits_sha256(hwc)
    assert sha2 == sha
    assert rep2 == replaced

    hdr = _wcs_header()
    hdr["SCIHASH"] = (sha2, "SHA-256")
    out = tmp_path / "science.fits"
    nbytes, rep3 = zb.write_science_fits_banded(hwc, hdr, out)
    assert rep3 == replaced
    assert nbytes == out.stat().st_size
    assert out.read_bytes() == dense_bytes

    # Round-trip: the FITS data reads back to the sanitized dense CHW.
    with fits.open(out) as hdul:
        got = np.asarray(hdul[0].data, dtype="<f4")
    raw = np.ascontiguousarray(np.moveaxis(np.asarray(hwc, dtype=np.float32), -1, 0))
    raw[~np.isfinite(raw)] = np.float32(0.0)
    assert np.array_equal(got, raw)


def test_a4_banded_science_fits_is_atomic_on_failure(tmp_path):
    hwc = np.zeros((H, W, C), dtype=np.float32)
    hdr = _wcs_header()
    hdr["SCIHASH"] = (zb.science_fits_sha256(hwc)[0], "SHA-256")
    out = tmp_path / "science.fits"
    # A pre-existing valid target must survive a failed banded write.
    out.write_bytes(b"sentinel")
    # Force a failure by asking to write to a path whose parent does not exist.
    bad = tmp_path / "no_such_dir" / "science.fits"
    with pytest.raises(FileNotFoundError):
        zb.write_science_fits_banded(hwc, hdr, bad)
    assert out.read_bytes() == b"sentinel"
    assert not (tmp_path / "science.fits.tmp").exists()


def test_a4_banded_normalize_header_places_mandatory_first():
    hdr = _wcs_header()
    block = zb.normalize_header_block(hdr, shape=(W, H, C), bitpix=-32)
    assert len(block) % zb.FITS_BLOCK_BYTES == 0
    text = block.decode("ascii")
    idx_simple = text.find("SIMPLE")
    idx_bitpix = text.find("BITPIX")
    idx_naxis = text.find("NAXIS")
    idx_wcs = text.find("WCSAXES")
    assert 0 <= idx_simple < idx_bitpix < idx_naxis < idx_wcs
    # NAXIS is never duplicated.
    assert text.count("NAXIS   ") == 1


def test_a4_banded_sha256_streams_small_bands(tmp_path):
    # A tiny band (band_rows=1) must still reproduce the exact dense hash.
    rng = np.random.default_rng(1)
    hwc = rng.standard_normal((8, 7, 3)).astype(np.float32)
    _, sha, replaced = _dense_reference(hwc)
    sha_banded, rep = zb.science_fits_sha256(hwc, band_rows=1)
    assert sha_banded == sha
    assert rep == replaced


def test_a4_banded_coverage_fits_is_byte_identical(tmp_path):
    rng = np.random.default_rng(2)
    cov = rng.standard_normal((H, W)).astype(np.float32)
    cov[2, 3] = np.nan

    # Dense reference.
    dense_plane = np.ascontiguousarray(np.asarray(cov, dtype=np.float32))
    finite = np.isfinite(dense_plane)
    replaced = int(dense_plane.size - int(np.count_nonzero(finite)))
    if replaced:
        dense_plane = dense_plane.copy()
        dense_plane[~finite] = np.float32(0.0)
    sha = hashlib.sha256(dense_plane.tobytes()).hexdigest()

    hdr = _wcs_header()
    del hdr["NAXIS"]
    del hdr["NAXIS3"]
    hdr["NAXIS"] = 2
    hdr["BUNIT"] = ("count", "per-pixel stack depth")
    buf = io.BytesIO()
    fits.PrimaryHDU(dense_plane, header=hdr).writeto(buf)
    dense_bytes = buf.getvalue()

    sha2, rep2 = zb.coverage_fits_sha256(cov)
    assert sha2 == sha
    assert rep2 == replaced

    out = tmp_path / "coverage.fits"
    nbytes, rep3 = zb.write_coverage_fits_banded(cov, hdr, out)
    assert rep3 == replaced
    assert nbytes == out.stat().st_size
    assert out.read_bytes() == dense_bytes
    with fits.open(out) as hdul:
        got = np.asarray(hdul[0].data, dtype="<f4")
    assert np.array_equal(got, dense_plane)


# ---------------------------------------------------------------------------
# F1 (ZM-ZEGRID-ADAPTIVE-RESOURCES): _write_outputs never materialises a
# full-canvas (3, H, W) float32 CHW copy, and the science FITS stays identical.
# ---------------------------------------------------------------------------

def test_f1_write_outputs_science_only_has_no_full_canvas_chw_copy(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import geometry as zg

    h, w, c = 16, 20, 3
    rng = np.random.default_rng(7)
    hwc = rng.standard_normal((h, w, c)).astype(np.float32)
    hwc[2, 3, 0] = np.nan
    hwc[4, 5, 1] = np.inf

    sci_path = tmp_path / "science_memmap.bin"
    cov_path = tmp_path / "coverage_memmap.bin"
    science = np.memmap(sci_path, dtype=np.float32, mode="w+", shape=(h, w, c))
    science[...] = hwc
    stack_depth = np.memmap(cov_path, dtype=np.int32, mode="w+", shape=(h, w))
    stack_depth[...] = 3
    assembled = SimpleNamespace(
        science=science,
        stack_depth=stack_depth,
        complete_cells=1,
        incomplete_cells=0,
        hole_pixels=0,
        coverage_pixels=h * w,
    )
    canvas = zg.GlobalCanvas(
        canvas_id="f1", wcs_header=_wcs_header().tostring(),
        width=w, height=h, resolution_deg=0.001,
    )
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 4, "predicted_bound_bytes": 0}

    seen_chw = []

    real_asc = np.ascontiguousarray

    def _recording_asc(arr, *args, **kwargs):
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape == (c, h, w):
            seen_chw.append(tuple(a.shape))
        return real_asc(arr, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", _recording_asc)

    zz._write_outputs(
        assembled, canvas, 1, 1, tmp_path, [], {}, [],
        layout, SimpleNamespace(normalization="sky_mean"), 0, {}, None,
        finished_science=science, raw_science=science,
        raw_science_path=None, aesthetic_path=None,
        export_aesthetic_fits=False,
    )

    assert seen_chw == []  # no full-canvas CHW float32 copy at all
    out = tmp_path / "mosaic_grid.fits"
    assert out.exists()

    # Byte-identical data + SCIHASH to the dense CHW path.
    dense_raw = np.ascontiguousarray(np.moveaxis(hwc, -1, 0))
    dense_raw[~np.isfinite(dense_raw)] = np.float32(0.0)
    with fits.open(out) as hdul:
        got = np.asarray(hdul[0].data, dtype="<f4")
        assert np.array_equal(got, dense_raw)
        assert hdul[0].header["SCIROLE"] == "SCI"
        assert hdul[0].header["SCIHASH"] == hashlib.sha256(
            dense_raw.tobytes()
        ).hexdigest()
    cov = tmp_path / "mosaic_grid_coverage.fits"
    with fits.open(cov) as hdul:
        assert np.array_equal(
            np.asarray(hdul[0].data, dtype=np.float32),
            np.asarray(stack_depth, dtype=np.float32),
        )
