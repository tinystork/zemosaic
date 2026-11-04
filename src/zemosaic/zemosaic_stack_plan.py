"""stack_plan.csv parsing helpers for the ZeGrid engine.

Relocated verbatim (ZM-ZEGRID-R8) from the now-removed legacy ``grid_mode``
module. The legacy Grid engine was REMOVED from the product and is ARCHIVED at
``origin/archive/zegrid-legacy-grid-5.0.0`` (branch tip 83b94f4). This module
retains ONLY the helpers still required by the NEW ZeGrid engine
(``zemosaic_zegrid_mode``): the ``stack_plan.csv`` presence check, the CSV
loader, the per-frame WCS loader, and their small private helpers.

Nothing here re-implements or depends on the legacy Grid pipeline.
"""

from __future__ import annotations

import csv
import errno
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:  # Optional heavy deps – handled gracefully if missing
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    import astropy.units as u

    _ASTROPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    fits = None
    WCS = None
    proj_plane_pixel_scales = None
    u = None
    _ASTROPY_AVAILABLE = False


logger = logging.getLogger("ZeMosaicWorker").getChild("stack_plan")
logger.propagate = True

ProgressCallback = Optional[Callable[[str, object, str], None]]

# ZM-IO-REJECT-01 rework-3: explicit setup/WCS failure classes. Only
# ``terminal_source`` failures may be quarantined; resource/cancellation,
# missing/access and infrastructure/internal failures must never move a source.
WCS_FAILURE_TERMINAL_SOURCE = "terminal_source"
WCS_FAILURE_MISSING_OR_ACCESS = "missing_or_access"
WCS_FAILURE_RESOURCE_OR_CANCELLED = "resource_or_cancelled"
WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL = "infrastructure_or_internal"


@dataclass
class WcsLoadFailure:
    """Structured reason why :func:`load_frame_wcs` returned ``False``.

    The historical ``bool`` return is preserved; callers that need to decide
    whether a source may be quarantined pass ``failure_out=[]`` and inspect the
    last entry.
    """

    kind: str
    path: str
    message: str
    exception_type: str | None = None
    exception: BaseException | None = field(default=None, repr=False, compare=False)

    @property
    def should_quarantine(self) -> bool:
        return self.kind == WCS_FAILURE_TERMINAL_SOURCE


class FrameSetupRefusal(RuntimeError):
    """A non-quarantinable setup/WCS failure the caller must surface."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        path: str | None = None,
        exception_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = str(kind)
        self.path = path
        self.exception_type = exception_type


def classify_setup_failure(exc: BaseException) -> str:
    """Classify one setup/WCS exception without over-claiming source corruption.

    * ``MemoryError`` / cancellation -> resource/cancelled;
    * disappeared/permission/access filesystem errors -> missing/access;
    * clear FITS/header/WCS format errors -> terminal source;
    * everything unexpected (programming errors, missing infrastructure) ->
      infrastructure/internal.
    """
    if isinstance(exc, MemoryError):
        return WCS_FAILURE_RESOURCE_OR_CANCELLED
    if type(exc).__name__ in {"CancelledError", "Cancelled", "TimeoutError"}:
        return WCS_FAILURE_RESOURCE_OR_CANCELLED
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)):
        return WCS_FAILURE_MISSING_OR_ACCESS
    if isinstance(exc, OSError):
        code = getattr(exc, "errno", None)
        if code in {errno.ENOMEM, errno.EAGAIN}:
            return WCS_FAILURE_RESOURCE_OR_CANCELLED
        if code is None:
            # Astropy reports an empty/corrupt FITS file as OSError without an
            # errno; that is a terminal source defect, not an access failure.
            return WCS_FAILURE_TERMINAL_SOURCE
        if code in {
            errno.EACCES,
            errno.EPERM,
            errno.ENOENT,
            errno.ESTALE,
            errno.EIO,
            errno.EINTR,
            errno.ETIMEDOUT,
            errno.EBUSY,
        }:
            return WCS_FAILURE_MISSING_OR_ACCESS
        return WCS_FAILURE_MISSING_OR_ACCESS
    if isinstance(exc, (ValueError, KeyError, IndexError, EOFError, UnicodeError)):
        return WCS_FAILURE_TERMINAL_SOURCE
    if type(exc).__name__ in {"VerifyError", "VerifyWarning"}:
        return WCS_FAILURE_TERMINAL_SOURCE
    if isinstance(exc, (TypeError, AttributeError, AssertionError, RuntimeError)):
        return WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL
    if isinstance(exc, ImportError):
        return WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL
    return WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL


def _emit(msg: str, *, lvl: str = "INFO", callback: ProgressCallback = None, **kwargs) -> None:
    """Emit a `[GRID]` log and mirror it to the worker progress callback."""

    tag = f"[GRID] {msg}"
    level = getattr(logging, str(lvl).upper(), logging.INFO)
    try:
        logger.log(level, tag)
    except Exception:
        pass
    if callback:
        try:
            callback(tag, None, str(lvl).upper(), **kwargs)
        except Exception:
            try:
                logger.debug("Progress callback failed for %s", tag, exc_info=True)
            except Exception:
                pass


def _open_fits_safely(path: Path):
    """Open a FITS file with the robust settings used across Grid mode."""

    if not (_ASTROPY_AVAILABLE and fits):
        raise RuntimeError("Astropy FITS support unavailable")
    return fits.open(path, memmap=False, do_not_scale_image_data=True)


@dataclass
class FrameInfo:
    path: Path
    exposure: float = 1.0
    bortle: str | None = None
    filter_name: str | None = None
    batch_id: str | None = None
    order: int = 0
    mount: str | None = None
    wcs: object | None = None
    shape_hw: tuple[int, int] | None = None
    footprint: tuple[float, float, float, float] | None = None  # (xmin, xmax, ymin, ymax) in global pixels


def detect_grid_mode(input_folder: str | os.PathLike[str]) -> bool:
    """Return True when a stack_plan.csv is present in *input_folder*."""

    try:
        candidate = Path(input_folder).expanduser() / "stack_plan.csv"
    except Exception:
        return False
    return candidate.is_file()


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        val = float(value)
        if math.isfinite(val):
            return val
    except Exception:
        pass
    return default


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_mount(value: object) -> str | None:
    """Normalize mount descriptors to \"EQ\" or \"ALTZ\"."""

    if value is None:
        return None
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if not text:
        return None
    if text in {"EQ", "EQUATORIAL", "GEM", "GERMAN"}:
        return "EQ"
    if text in {"ALTZ", "ALT-AZ", "ALT/AZ", "ALT-AZM", "ALTAZ", "ALTITUDE-AZIMUTH"}:
        return "ALTZ"
    return None


def _resolve_path(base_dir: Path, text: str | os.PathLike[str]) -> Path:
    """Resolve frame paths from stack_plan entries.

    Supports native absolute/relative paths and Windows-exported paths reused on
    POSIX hosts (e.g. ``D:\\...\\file.fit``). In cross-platform cases we try
    sane fallbacks under ``base_dir`` before returning a non-existing candidate.
    """
    raw = str(text).strip().strip('"').strip("'")
    candidates: list[Path] = []

    try:
        candidate = Path(raw)
    except Exception:
        candidate = Path(str(raw))

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(base_dir / candidate)

    # Windows absolute path serialized into CSV and reused on Linux/macOS.
    windows_drive_like = len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()
    windows_unc_like = raw.startswith("\\")
    if windows_drive_like or windows_unc_like or "\\" in raw:
        posixish = raw.replace("\\", "/")
        # Keep only path tail after drive if present: D:/a/b/file.fit -> a/b/file.fit
        if windows_drive_like and ":" in posixish:
            posixish = posixish.split(":", 1)[1].lstrip("/")
        if posixish:
            candidates.append(base_dir / Path(posixish))
        # Last-resort: basename in current input folder
        try:
            basename = Path(posixish).name if posixish else Path(raw).name
        except Exception:
            basename = ""
        if basename:
            candidates.append(base_dir / basename)

    # Return first existing candidate.
    seen: set[str] = set()
    first_resolved: Path | None = None
    for cand in candidates:
        try:
            resolved = cand.expanduser().resolve(strict=False)
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if first_resolved is None:
            first_resolved = resolved
        try:
            if resolved.is_file():
                return resolved
        except Exception:
            pass

    return first_resolved if first_resolved is not None else (base_dir / Path(raw)).expanduser().resolve(strict=False)


def _dialect_from_sample(sample: str) -> csv.Dialect | None:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t ")
    except Exception:
        return None


def load_stack_plan(csv_path: str | os.PathLike[str], *, progress_callback: ProgressCallback = None) -> list[FrameInfo]:
    """Parse ``stack_plan.csv`` and return a list of FrameInfo entries."""

    csv_file = Path(csv_path).expanduser()
    base_dir = csv_file.parent
    if not csv_file.is_file():
        _emit(f"stack_plan.csv not found at {csv_file}", lvl="WARN", callback=progress_callback)
        return []

    try:
        raw_text = csv_file.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        _emit(f"Unable to read stack_plan.csv: {exc}", lvl="ERROR", callback=progress_callback)
        return []

    sample = raw_text[:2048]
    dialect = _dialect_from_sample(sample) or csv.excel
    lines = raw_text.splitlines()
    if not lines:
        _emit("stack_plan.csv is empty", lvl="WARN", callback=progress_callback)
        return []

    reader = csv.DictReader(lines, dialect=dialect)
    frames: list[FrameInfo] = []
    normalized_headers = [h.strip().lower() for h in (reader.fieldnames or [])]
    has_header = bool(normalized_headers)

    def _field(row: dict, keys: Sequence[str], default: str | None = None) -> str | None:
        for key in keys:
            if key in row and row.get(key) not in (None, ""):
                return str(row.get(key)).strip()
        return default

    if not has_header:
        _emit("stack_plan.csv without headers, treating first column as path", lvl="WARN", callback=progress_callback)

    for idx, row in enumerate(reader):
        if not isinstance(row, dict):
            continue
        path_text: str | None
        if has_header:
            path_text = _field(
                row,
                (
                    "file_path",
                    "filepath",
                    "file",
                    "path",
                    "filename",
                    "fits",
                    "raw",
                ),
            )
        else:
            # DictReader with no headers returns None keys; pick the first non-empty value
            values = [v for v in row.values() if v not in (None, "")]
            path_text = str(values[0]).strip() if values else None

        if not path_text:
            _emit(f"Row {idx+1}: missing file path, skipping", lvl="WARN", callback=progress_callback)
            continue

        frame_path = _resolve_path(base_dir, path_text)
        if not frame_path.is_file():
            _emit(f"Row {idx+1}: file not found {frame_path}", lvl="WARN", callback=progress_callback)
            continue

        exposure = _parse_float(
            _field(row, ("exposure", "exp", "exptime", "exposure_s"), default=1.0),
            default=1.0,
        )
        bortle = _field(row, ("bortle", "bortle_class", "sky_quality"))
        filter_name = _field(row, ("filter", "band", "channel"))
        batch_id = _field(row, ("batch_id", "batch", "session", "night"))
        order_val = _field(row, ("order", "seq", "sequence", "index"))
        order = _parse_int(order_val, default=idx)
        mount_raw = _field(row, ("mount", "eqmode", "mount_mode"))
        mount = _normalize_mount(mount_raw)

        frames.append(
            FrameInfo(
                path=frame_path,
                exposure=exposure,
                bortle=bortle,
                filter_name=filter_name,
                batch_id=batch_id,
                order=order,
                mount=mount,
            )
        )

    _emit(f"Loaded {len(frames)} frame(s) from stack_plan.csv", callback=progress_callback)
    return frames


def _extract_pixel_scale_deg(wcs_obj: object) -> float | None:
    if not (_ASTROPY_AVAILABLE and proj_plane_pixel_scales and u):
        return None
    try:
        scales = proj_plane_pixel_scales(wcs_obj)  # type: ignore[arg-type]
        vals = [abs(s.to_value(u.deg)) for s in scales]
        return float(np.mean(vals))
    except Exception:
        try:
            cdelt = np.asarray(getattr(wcs_obj, "wcs", None).cdelt, dtype=float)
            if cdelt.size >= 2:
                return float(np.mean(np.abs(cdelt)))
        except Exception:
            return None
    return None


def _header_int(header, key: str) -> int | None:
    """Return ``header[key]`` as a strict int, or ``None`` when absent/invalid."""
    try:
        value = header.get(key)
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _third_axis_kind(ctype3: str) -> str | None:
    """Classify a *declared* third-axis CTYPE into a coarse, explicit kind.

    Only used to produce clear rejection diagnostics: any non-blank ``CTYPE3``
    means the third axis carries a declared meaning, which is never accepted for
    the RGB channel axis (see :func:`_admit_rgb_chw_cube`).
    """
    base = (ctype3 or "").strip().upper()
    if not base:
        return None
    if base.startswith(("RA---", "DEC--", "GLON", "GLAT", "ELON", "ELAT", "HPLN", "HPLT",
                        "RA", "DEC")):
        return "celestial"
    if base.startswith(("WAVE", "AWAV", "FREQ", "ENER", "VOPT", "VELO", "VRAD", "BETA")):
        return "spectral"
    if base.startswith(("TIME", "UTC", "TAI", "TT", "GPS", "MJD")):
        return "temporal"
    if base.startswith(("STOKES", "POL")):
        return "polarization"
    return "other"


def _admit_rgb_chw_cube(header, shape) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Admission gate for a 3-D primary that is an RGB CHW cube.

    Returns ``(admitted, reason, shape_hw)``:

    * ``(False, None, None)`` — the primary is not 3-D at all: the caller keeps
      the historical 2-D path untouched.
    * ``(True, None, (H, W))`` — the exact admitted class: ``NAXIS=3``,
      ``NAXIS3=3``, data of shape ``(3, H, W)`` consistent with
      ``NAXIS1=W``/``NAXIS2=H``, blank ``CTYPE3``. Only this class may build
      ``WCS(header, naxis=2)``.
    * ``(False, reason, None)`` — a 3-D primary that is *not* this exact class
      (non-RGB, HWC/ambiguous layout, contradictory header dimensions, or a
      declared celestial/spectral/temporal/polarization third axis).

    ``WCSAXES=2`` alone is deliberately NOT accepted as proof of a channel axis:
    the header and pixel dimensions must also agree on the CHW contract.

    ZM-WCS-02: *shape* is an immutable ``tuple`` of axis lengths obtained from
    the header-derived ``PrimaryHDU.shape``. This gate must never receive (or
    touch) a pixel array: the WCS/shape loader is header-only and the pixel
    payload is validated later by the scientific decoder.
    """
    if shape is None:
        if _header_int(header, "NAXIS") == 3:
            return False, "NAXIS=3 primary without readable data (cannot verify CHW layout)", None
        return False, None, None
    try:
        shape = tuple(int(n) for n in shape)
    except Exception:
        return False, "unreadable primary shape (cannot verify CHW layout)", None
    if len(shape) == 2:
        return False, None, None
    if len(shape) != 3:
        return False, f"unsupported primary dimensionality {len(shape)} (expected 2-D or 3-D)", None

    naxis = _header_int(header, "NAXIS")
    naxis3 = _header_int(header, "NAXIS3")
    if naxis != 3:
        return False, f"3-D data with NAXIS={naxis!r} (expected NAXIS=3)", None
    if naxis3 != 3:
        return False, f"third axis NAXIS3={naxis3!r} (expected 3 RGB channels)", None

    channels, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    if channels != 3:
        return (
            False,
            f"leading data axis is {channels}, expected 3 (CHW RGB; HWC/ambiguous layout refused)",
            None,
        )
    if height <= 0 or width <= 0:
        return False, f"non-positive spatial dimensions ({height}, {width})", None

    hdr_w = _header_int(header, "NAXIS1")
    hdr_h = _header_int(header, "NAXIS2")
    if hdr_w != width or hdr_h != height:
        return (
            False,
            f"header dimensions NAXIS1/NAXIS2={hdr_w!r}/{hdr_h!r} contradict data "
            f"shape (3, {height}, {width})",
            None,
        )

    ctype3 = str(header.get("CTYPE3", "") or "").strip()
    if ctype3:
        kind = _third_axis_kind(ctype3) or "other"
        return (
            False,
            f"third axis declared as {kind} (CTYPE3={ctype3!r}); RGB channel axis must be blank",
            None,
        )
    return True, None, (height, width)


def load_frame_wcs(
    frame: FrameInfo,
    *,
    progress_callback: ProgressCallback = None,
    failure_out: list[WcsLoadFailure] | None = None,
) -> bool:
    """Populate frame.wcs and frame.shape_hw if possible.

    Fail-closed contract: the frame fields are cleared on entry and assigned ONLY
    on success, so a ``False`` return always means ``frame.wcs is None`` and
    ``frame.shape_hw is None`` — never a stale state from a previous success.

    ``failure_out`` (optional) receives ONE :class:`WcsLoadFailure` on failure so
    a caller can distinguish a terminal source defect (quarantinable) from a
    missing/access, resource/cancellation or infrastructure/internal failure
    (never quarantinable). The historical ``bool`` return is unchanged.
    """

    # Clear before any possible early return (astropy unavailable, open failure,
    # admission refusal, WCS parsing/qualification, shape, pixel scale).
    frame.wcs = None
    frame.shape_hw = None

    def _fail(kind: str, message: str, exc: BaseException | None = None) -> bool:
        if failure_out is not None:
            failure_out.append(
                WcsLoadFailure(
                    kind=kind,
                    path=str(frame.path),
                    message=message,
                    exception_type=(type(exc).__name__ if exc is not None else None),
                    exception=exc,
                )
            )
        _emit(
            message,
            lvl=("WARN" if kind == WCS_FAILURE_TERMINAL_SOURCE else "ERROR"),
            callback=progress_callback,
        )
        return False

    if not (_ASTROPY_AVAILABLE and fits and WCS):
        return _fail(
            WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL,
            "Astropy not available for WCS parsing",
        )
    try:
        with _open_fits_safely(frame.path) as hdul:
            # ZM-WCS-02: header-only. ``PrimaryHDU.shape`` is derived from the
            # header ``NAXISn`` cards (``_axes``) and never materializes the
            # pixel payload; ``hdul[0].data`` must not be touched here.
            header = hdul[0].header.copy()
            try:
                raw_shape = hdul[0].shape
                shape = tuple(int(n) for n in raw_shape) if raw_shape else None
            except Exception as exc:
                kind = classify_setup_failure(exc)
                if kind != WCS_FAILURE_TERMINAL_SOURCE:
                    return _fail(
                        kind,
                        f"Failed to derive FITS shape for {frame.path}: {exc}",
                        exc,
                    )
                shape = None
    except Exception as exc:
        return _fail(
            classify_setup_failure(exc),
            f"Failed to open FITS {frame.path}: {exc}",
            exc,
        )

    # ZM-ZEGRID-R35A: mount metadata (EQ / ALTZ) is now read from the FITS header
    # itself (EQMODE), replacing the stack_plan ``mount`` column while preserving
    # the exact same EQ/ALTZ semantics downstream.
    try:
        if getattr(frame, "mount", None) is None:
            from .input_inventory import mount_from_header as _mount_from_header

            frame.mount = _mount_from_header(header)
    except Exception as exc:
        if classify_setup_failure(exc) == WCS_FAILURE_RESOURCE_OR_CANCELLED:
            return _fail(
                WCS_FAILURE_RESOURCE_OR_CANCELLED,
                f"Mount metadata read interrupted for {frame.path}: {exc}",
                exc,
            )


    # ZM-WCS-01: a 3-D primary is admitted ONLY as an unambiguous RGB CHW cube
    # (NAXIS=3, NAXIS3=3, (3, H, W), blank CTYPE3). For that single class the
    # celestial WCS lives on axes 1-2, so it is built with ``naxis=2`` (which
    # also keeps SIP working). Every other 3-D primary is refused explicitly.
    # The 2-D primary path below keeps the historical construction unchanged.
    try:
        cube_ok, cube_reason, cube_shape_hw = _admit_rgb_chw_cube(header, shape)
    except Exception as exc:
        return _fail(
            classify_setup_failure(exc),
            f"Failed to admit RGB cube for {frame.path}: {exc}",
            exc,
        )
    if cube_reason is not None:
        return _fail(
            WCS_FAILURE_TERMINAL_SOURCE,
            f"Rejecting frame {frame.path}: {cube_reason}",
        )

    wcs_obj = None
    try:
        if cube_ok:
            wcs_obj = WCS(header, naxis=2)
        else:
            wcs_obj = WCS(header)
    except Exception as exc:
        kind = classify_setup_failure(exc)
        if kind != WCS_FAILURE_TERMINAL_SOURCE:
            return _fail(
                kind,
                f"Failed to parse WCS from {frame.path}: {exc}",
                exc,
            )
        wcs_obj = None
    if (
        wcs_obj is None
        or not getattr(wcs_obj, "is_celestial", False)
        or (cube_ok and getattr(wcs_obj, "pixel_n_dim", None) != 2)
    ):
        return _fail(
            WCS_FAILURE_TERMINAL_SOURCE,
            f"No usable celestial WCS in {frame.path}",
        )
    if cube_ok:
        # DEBUG: the real M45 corpus has 1827 cubes; the admission is pinned by
        # tests, so a per-frame INFO line would only flood the worker log.
        _emit(
            f"RGB CHW cube {frame.path}: celestial WCS built on axes 1-2 (naxis=2)",
            lvl="DEBUG",
            callback=progress_callback,
        )

    shape_hw = None
    try:
        if shape is not None:
            if cube_shape_hw is not None:
                shape_hw = cube_shape_hw
            elif len(shape) == 2:
                shape_hw = (int(shape[0]), int(shape[1]))
            # 3-D primaries are fully handled by the admission gate above.
    except Exception as exc:
        kind = classify_setup_failure(exc)
        if kind != WCS_FAILURE_TERMINAL_SOURCE:
            return _fail(
                kind,
                f"Failed to derive image shape for {frame.path}: {exc}",
                exc,
            )
        shape_hw = None
    if shape_hw is None:
        return _fail(
            WCS_FAILURE_TERMINAL_SOURCE,
            f"Cannot derive image shape for {frame.path}",
        )
    try:
        px_scale = _extract_pixel_scale_deg(wcs_obj)
        if px_scale is None or not math.isfinite(px_scale) or px_scale <= 0:
            return _fail(
                WCS_FAILURE_TERMINAL_SOURCE,
                f"Rejecting frame {frame.path}: invalid pixel scale",
            )
    except Exception as exc:
        return _fail(
            classify_setup_failure(exc),
            f"Rejecting frame {frame.path}: pixel scale check failed ({exc})",
            exc,
        )

    # Success: publish the validated state.
    frame.wcs = wcs_obj
    frame.shape_hw = shape_hw
    return True
