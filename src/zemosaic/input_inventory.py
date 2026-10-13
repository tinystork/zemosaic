"""Shared FITS input inventory + mount metadata (ZM-ZEGRID-R35A).

A single, deterministic inventory of the eligible FITS dataset of a ZeMosaic
input folder.  It reproduces EXACTLY the historical classic scanner semantics
(recursive walk, ``.fit``/``.fits``, deterministic ordering, excluded
directories, output-subtree exclusion, internal/Global-WCS artifacts excluded)
so that ZeGrid and the classic pipeline share ONE scanner instead of diverging.

ZeGrid previously consumed ``stack_plan.csv`` (a ZeAnalyser/ZSSS selection),
which could silently reduce its input.  After R35A the ZeGrid input is this
inventory (optionally narrowed by an EXPLICIT Filter selection made by the
user — never by a stack plan).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

#: Supported FITS suffixes (lowercase).
FITS_SUFFIXES: tuple[str, ...] = (".fit", ".fits")

#: Directory excluded from any inventory (quarantine).
DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset({"unaligned_by_zemosaic"})


def _normcase(path: Any) -> str:
    try:
        return os.path.normcase(str(path))
    except Exception:
        return str(path)


def _is_excluded(path: Path, excluded_dirs: Iterable[str]) -> bool:
    dirs = set(excluded_dirs)
    if not dirs:
        return False
    try:
        parts = set(path.expanduser().resolve(strict=False).parts)
    except Exception:
        try:
            parts = set(path.parts)
        except Exception:
            return False
    return any(part in parts for part in dirs)


def scan_input_fits(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str] | None = None,
    *,
    global_wcs_output_path: str | None = None,
    excluded_dirs: Sequence[str] | None = None,
) -> list[str]:
    """Return the deterministic list of eligible FITS paths under *input_root*.

    Rules (identical to the classic pipeline scanner):

    * recursive walk, ``.fit`` / ``.fits`` only;
    * excluded directories pruned during descent;
    * the *output_root* subtree is pruned (never ingest our own outputs);
    * the Global WCS artifact (basename match) is skipped;
    * deterministic ordering (case-insensitive path sort).
    """
    if input_root is None:
        return []
    try:
        root_path = Path(input_root).expanduser()
    except Exception:
        return []
    if not root_path.is_dir():
        return []

    dirs = tuple(excluded_dirs) if excluded_dirs else tuple(DEFAULT_EXCLUDED_DIRS)

    output_abs_norm = None
    if output_root:
        try:
            output_abs_norm = _normcase(Path(output_root).expanduser().resolve())
        except Exception:
            output_abs_norm = None

    try:
        wcs_out_base = Path(str(global_wcs_output_path or "global_mosaic_wcs.fits")).name.strip().lower()
    except Exception:
        wcs_out_base = "global_mosaic_wcs.fits"

    fits_paths: list[str] = []
    for root_dir, dirnames, filenames in os.walk(str(root_path)):
        current = Path(root_dir)
        keep_dirs: list[str] = []
        for d in dirnames:
            child = current / d
            if _is_excluded(child, dirs):
                continue
            if output_abs_norm is not None:
                try:
                    child_norm = _normcase(child.resolve())
                except Exception:
                    child_norm = _normcase(child)
                if child_norm == output_abs_norm or child_norm.startswith(output_abs_norm + os.sep):
                    continue
            keep_dirs.append(d)
        dirnames[:] = keep_dirs

        try:
            ordered_files = sorted(filenames, key=lambda s: s.lower())
        except Exception:
            ordered_files = list(filenames)

        for name in ordered_files:
            if not name.lower().endswith(FITS_SUFFIXES):
                continue
            full = current / name
            if _is_excluded(full, dirs):
                continue
            if wcs_out_base and name.strip().lower() == wcs_out_base:
                continue
            fits_paths.append(str(full))

    try:
        fits_paths.sort(key=lambda p: p.lower())
    except Exception:
        fits_paths.sort()
    return fits_paths


# ---------------------------------------------------------------------------
# Mount metadata (EQ / ALTZ) from FITS headers
# ---------------------------------------------------------------------------

_EQMODE_KEYS: tuple[str, ...] = ("EQMODE", "MOUNT_MODE", "MOUNTMOD", "MOUNTMODE")


def normalize_mount(value: object) -> str | None:
    """Normalize a mount descriptor to ``"EQ"`` or ``"ALTZ"`` (or ``None``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            iv = int(value)
        except Exception:
            return None
        if iv == 0:
            return "ALTZ"
        if iv == 1:
            return "EQ"
        return None
    try:
        text = str(value).strip().upper()
    except Exception:
        return None
    if not text:
        return None
    if text in {"EQ", "EQUATORIAL", "GEM", "GERMAN", "EQMODE_1"}:
        return "EQ"
    if text in {"ALTZ", "ALT-AZ", "ALT_AZ", "ALT/AZ", "ALT-AZM", "ALTAZ", "ALTITUDE-AZIMUTH", "EQMODE_0"}:
        return "ALTZ"
    return None


def mount_from_header(header: object) -> str | None:
    """Extract ``"EQ"`` / ``"ALTZ"`` from a FITS-like header (or ``None``)."""
    if header is None:
        return None
    for key in _EQMODE_KEYS:
        value = None
        try:
            if hasattr(header, "get"):
                value = header.get(key)
            else:
                value = header[key]  # type: ignore[index]
        except Exception:
            value = None
        if value is None:
            continue
        mount = normalize_mount(value)
        if mount is not None:
            return mount
    return None


def mount_from_fits(path: str | os.PathLike[str]) -> str | None:
    """Best-effort mount read from an on-disk FITS file (never raises)."""
    try:
        from astropy.io import fits  # type: ignore
    except Exception:
        return None
    try:
        with fits.open(str(path), memmap=False, do_not_scale_image_data=True) as hdul:
            return mount_from_header(hdul[0].header)
    except Exception:
        return None


def build_zegrid_frames(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str] | None = None,
    *,
    global_wcs_output_path: str | None = None,
    selection_paths: Sequence[str] | None = None,
    excluded_dirs: Sequence[str] | None = None,
):
    """Build the ZeGrid frame list from the FITS inventory (R35A).

    Replaces the historical ``load_stack_plan()`` feed:

    * the dataset is the FULL eligible FITS inventory of *input_root*;
    * ``selection_paths`` (an EXPLICIT user Filter selection) may narrow it —
      only paths already present in the inventory are kept, preserving the
      deterministic inventory order;
    * ``stack_plan.csv`` never participates.

    Returns ``(frames, info)`` where ``frames`` is a list of
    :class:`zemosaic.zemosaic_stack_plan.FrameInfo` (path + order; WCS/mount are
    populated later from the FITS header) and ``info`` records
    ``discovered`` / ``selected`` / ``used`` for the run log.
    """
    from .zemosaic_stack_plan import FrameInfo

    discovered = scan_input_fits(
        input_root,
        output_root,
        global_wcs_output_path=global_wcs_output_path,
        excluded_dirs=excluded_dirs,
    )
    info: dict[str, object] = {
        "discovered": len(discovered),
        "selection_applied": bool(selection_paths),
        "selected": None,
        "used": len(discovered),
        "selection_unmatched": 0,
    }

    chosen = list(discovered)
    if selection_paths:
        wanted = []
        for p in selection_paths:
            try:
                wanted.append(_normcase(Path(str(p)).expanduser()))
            except Exception:
                continue
        wanted_set = set(wanted)
        matched = [p for p in discovered if _normcase(Path(p).expanduser()) in wanted_set]
        info["selected"] = len(matched)
        if matched:
            chosen = matched
            info["used"] = len(chosen)
        else:
            # A non-empty selection that matches nothing is a real anomaly: keep
            # the full inventory rather than silently running on zero frames, and
            # flag it so the caller can WARN (never a silent reduction/expansion).
            info["selection_unmatched"] = len(wanted_set)
            info["used"] = len(chosen)

    frames = [FrameInfo(path=Path(p), order=idx) for idx, p in enumerate(chosen)]
    return frames, info
