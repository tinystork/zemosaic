"""ZM-ZEGRID-ADAPTIVE-RESOURCES A4 — banded, streaming FITS serialization.

Writes a ``(H, W)`` float32 coverage plane or a ``(H, W, C)`` float32 science
cube as a FITS ``PrimaryHDU`` in row-bands, so no full-canvas CHW contiguous
copy is materialized at the serialization step. The output is **byte-identical**
to the dense ``fits.PrimaryHDU(np.ascontiguousarray(np.moveaxis(science, -1, 0)))
.writeto(...)`` path the product has always used:

* the header block is rebuilt with the mandatory cards (``SIMPLE``, ``BITPIX``,
  ``NAXIS``, ``NAXISn``) placed first, exactly as astropy writes them;
* the data block is big-endian float32 in FITS order (``NAXIS1`` fastest),
  padded to the 2880-byte block boundary;
* the ``SCIHASH`` card is the SHA-256 of the sanitized **little-endian**
  contiguous CHW float32 bytes (the exact bytes ``_array_sha256`` hashes in the
  dense path), computed by streaming the same row-bands.

Nothing here changes science, dtype, WCS, ordering or the sanitization rule:
non-finite samples are still replaced with ``fill_value`` in the serialized
buffer ONLY (never the in-memory array), and the replaced count is returned for
the auditable header/manifest.

Explicit non-claims: no Windows execution on this host; the atomic rename
contract is the same one the product already relies on (``os.replace``).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

FITS_BLOCK_BYTES = 2880
DEFAULT_BAND_ROWS = 64
NONFINITE_FILL_VALUE = 0.0

#: Mandatory FITS keywords that astropy places first and that must never be
#: duplicated when rebuilding the header block.
_MANDATORY_KEYWORDS = ("SIMPLE", "BITPIX", "NAXIS")


def _sanitize_band(arr_f32: np.ndarray, fill_value: float) -> tuple[np.ndarray, int]:
    """Return ``(sanitized, replaced)`` for ONE already-float32 band.

    Bit-for-bit the same rule as ``zemosaic_utils.sanitize_nonfinite_float32``
    but scoped to one band so the serialization can stream. Finite samples are
    never modified; non-finite samples become ``fill_value`` (default 0.0).
    """
    finite = np.isfinite(arr_f32)
    replaced = int(arr_f32.size - int(np.count_nonzero(finite)))
    if replaced == 0:
        return arr_f32, 0
    out = arr_f32.copy()
    out[~finite] = np.float32(fill_value)
    return out, replaced


def _band_planes(shape_ndim: int, channels: int) -> list[tuple[int, int]]:
    """Ordered ``(plane_index, ndim_slot)`` for a FITS data block.

    For a 3-D cube the FITS layout is channel-major: plane 0 = channel 0, plane
    1 = channel 1, plane 2 = channel 2. For a 2-D plane there is one plane.
    """
    if shape_ndim == 2:
        return [(0, 0)]
    return [(c, c) for c in range(channels)]


def normalize_header_block(
    header: fits.Header,
    *,
    shape: tuple[int, ...],
    bitpix: int = -32,
) -> bytes:
    """Rebuild a FITS primary-header block with mandatory cards first.

    ``header`` is the product's already-populated ``fits.Header`` (WCS cards,
    ``SCIROLE``, ``SCIHASH``, halo/non-finite cards). astropy writes ``SIMPLE``,
    ``BITPIX`` and ``NAXIS``/``NAXISn`` first regardless of their position in the
    dict; this helper reproduces that exact serialization without allocating any
    data array, so the caller can stream the data block afterwards.
    """
    out = fits.Header()
    out["SIMPLE"] = (True, "conforms to FITS standard")
    out["BITPIX"] = (int(bitpix), "array data type")
    out["NAXIS"] = (int(len(shape)), "number of array dimensions")
    for i, size in enumerate(shape, start=1):
        out[f"NAXIS{i}"] = int(size)
    for card in header.cards:
        key = str(card.keyword)
        if key in _MANDATORY_KEYWORDS or key.startswith("NAXIS"):
            continue
        try:
            out.append(card)
        except Exception:  # pragma: no cover - defensive card re-append
            out.append((key, card.value, card.comment))
    block = out.tostring().encode("ascii")
    pad = (-len(block)) % FITS_BLOCK_BYTES
    return block + (b" " * pad)


def science_fits_sha256(
    science_hwc: np.ndarray,
    *,
    fill_value: float = NONFINITE_FILL_VALUE,
    band_rows: int = DEFAULT_BAND_ROWS,
) -> tuple[str, int]:
    """Stream the SHA-256 of the sanitized contiguous CHW little-endian bytes.

    ``science_hwc`` is ``(H, W, C)`` float32 (possibly a scratch memmap). The
    dense path hashes ``np.ascontiguousarray(np.moveaxis(science, -1, 0))``
    (little-endian CHW). Streaming the same row-bands in channel-major order
    reproduces those exact bytes without materialising the full CHW cube.

    Returns ``(sha256_hex, replaced_count)``.
    """
    height, _w, channels = science_hwc.shape
    digest = hashlib.sha256()
    replaced = 0
    for ch in range(int(channels)):
        for y0 in range(0, int(height), int(band_rows)):
            band = np.ascontiguousarray(
                science_hwc[y0:y0 + int(band_rows), :, ch], dtype=np.float32
            )
            band, n = _sanitize_band(band, fill_value)
            replaced += n
            digest.update(band.tobytes())
    return digest.hexdigest(), int(replaced)


def write_science_fits_banded(
    science_hwc: np.ndarray,
    header: fits.Header,
    path: str | os.PathLike[str],
    *,
    fill_value: float = NONFINITE_FILL_VALUE,
    band_rows: int = DEFAULT_BAND_ROWS,
) -> tuple[int, int]:
    """Atomically write ``(H, W, C)`` float32 as a channel-major FITS PrimaryHDU.

    The ``SCIHASH`` card must already be present in ``header`` (the caller
    computes it with :func:`science_fits_sha256`). Data is written big-endian in
    row-bands and padded to the FITS block boundary. Returns
    ``(nbytes, replaced_count)``.
    """
    h, w, c = science_hwc.shape
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    header_block = normalize_header_block(
        header, shape=(int(w), int(h), int(c)), bitpix=-32
    )
    replaced = 0
    try:
        with open(tmp, "wb") as fh:
            fh.write(header_block)
            for ch in range(int(c)):
                for y0 in range(0, int(h), int(band_rows)):
                    band = np.ascontiguousarray(
                        science_hwc[y0:y0 + int(band_rows), :, ch], dtype=np.float32
                    )
                    band, n = _sanitize_band(band, fill_value)
                    replaced += n
                    fh.write(band.astype(">f4").tobytes())
            data_len = int(h) * int(w) * int(c) * 4
            pad = (-(int(data_len) % FITS_BLOCK_BYTES)) % FITS_BLOCK_BYTES
            if pad:
                fh.write(b"\x00" * pad)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover - best effort
            pass
        raise
    return int(h) * int(w) * int(c) * 4 + int(len(header_block)) + int(
        (-(int(h) * int(w) * int(c) * 4) % FITS_BLOCK_BYTES) % FITS_BLOCK_BYTES
    ), int(replaced)
