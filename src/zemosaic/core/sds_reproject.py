"""ZM-SDS-R35C-C5 (part A) — SDS CPU reprojection fast-path adapter.

This module accelerates the **SDS frame reprojection / estimation** step by
reusing the EXISTING, already-frozen CPU fast path
:func:`zemosaic.core.zegrid.execution.reproject_cropped` (R21). That wrapper:

* takes the fast ``map_coordinates`` path **only for plain undistorted
  RA---TAN/DEC--TAN WCS** (CD matrix *and* PC+CDELT accepted); and
* **falls back to ``reproject_interp``** for SIP / distorted / non-TAN WCS, or
  on any fast-path error — never approximating, never dropping distortions.

This adapter adds **no new arithmetic**. It only rebuilds the worker's exact
``(patch, weight)`` contract from the ``(RGB, geometric_support)`` pair the R21
wrapper returns:

* ``patch``  = the reprojected RGB float32 with ``NaN`` outside the footprint
  (then ``nan_to_num``-ed by the caller, exactly as the pre-C5 path did);
* ``weight`` = the **binary footprint** (the R21 fast path and
  ``reproject_interp`` both produce a 0/1 footprint for TAN), intersected with
  the per-pixel "all channels finite" mask. This is bit-for-bit the quantity the
  historical ``np.clip(nan_to_num(footprint), 0, 1)`` loop computed, because
  ``reproject_interp`` sets its output to ``NaN`` outside the footprint, so
  ``footprint > 0`` and ``all(isfinite(patch))`` describe the same support.

Scope / safety
--------------
* Used **only** by the SDS reprojection path (``zemosaic_worker._reproject_frame``
  when the global WCS plan mode is ``sds``). Classic/Legacy/ZeGrid dispatch and
  every ``core/canonical_*`` formula are untouched.
* The historical per-frame ``reproject_interp`` loop is KEPT verbatim in the
  worker as the fallback: if this helper returns ``None`` (WCS not eligible, fast
  path error, unexpected shape) the worker runs the unchanged legacy loop.
* The R21 process-local path accumulator (``get_reproject_path_stats``) records
  the fast/fallback calls; :func:`sds_reproject_stats_payload` surfaces it.

Bit-exactness is asserted by ``tests/test_sds_fast_reproject_c5.py`` against the
verbatim legacy loop for synthetic TAN matrices, an adversarial SIP fallback and
real M106 TAN frames.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from zemosaic.core.zegrid.execution import (
    get_reproject_path_stats,
    reproject_cropped,
)

__all__ = [
    "SDS_REPROJECT_FORCE_FALLBACK_ENV",
    "sds_fast_reproject_forced_off",
    "resolve_sds_fast_reproject",
    "reproject_frame_patch_fast",
    "sds_reproject_stats_payload",
]

_LOGGER = logging.getLogger(__name__)

# Process-wide kill switch (A/B): force the SDS reprojection back onto the
# historical ``reproject_interp`` loop. Read at call time so a test can toggle it.
SDS_REPROJECT_FORCE_FALLBACK_ENV = "SDS_REPROJECT_FORCE_FALLBACK"


def sds_fast_reproject_forced_off() -> bool:
    """True when the env kill switch disables the SDS fast reprojection path."""
    return os.environ.get(SDS_REPROJECT_FORCE_FALLBACK_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _coerce_bool(raw, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    try:
        return bool(int(raw))
    except Exception:
        return str(raw).strip().lower() in ("1", "true", "yes", "on")


def resolve_sds_fast_reproject(plan=None, default: bool = True) -> bool:
    """Resolve whether the SDS fast CPU reprojection path is enabled.

    Precedence: the env kill switch wins; then an explicit
    ``sds_fast_reproject`` (top level or ``meta``) in the global WCS plan; else
    ``default`` (``True`` — the parity is bit-exact, see the module docstring).
    """
    if sds_fast_reproject_forced_off():
        return False
    if isinstance(plan, dict):
        raw = plan.get("sds_fast_reproject")
        if raw is None:
            meta = plan.get("meta")
            if isinstance(meta, dict):
                raw = meta.get("sds_fast_reproject")
        if raw is not None:
            return _coerce_bool(raw, default)
    return bool(default)


def reproject_frame_patch_fast(
    frame_hwc: np.ndarray,
    frame_wcs,
    patch_wcs,
    patch_shape_hw: tuple[int, int],
    *,
    emit=None,
):
    """Reproject one SDS frame through the R21 fast path.

    Returns ``(patch_hwc_float32, weight_hw_float32)`` — ``patch`` is
    ``nan_to_num``-ed (NaN/±Inf -> 0), ``weight`` is the binary footprint
    intersected with the finite-support mask (values 0.0/1.0) — or ``None`` when
    the helper is not applicable (unexpected frame layout, reproject
    unavailable, or an internal fast-path failure). ``None`` means "run the
    historical ``reproject_interp`` loop"; it never raises.

    Non-qualified WCS (SIP/distorted/non-TAN) are handled by
    :func:`reproject_cropped` itself, which returns the exact
    ``reproject_interp`` result for them (loud fallback, recorded in the R21
    path stats) — so the science never approximates.
    """
    try:
        if frame_hwc is None or np.asarray(frame_hwc).ndim != 3:
            return None
        shape = (int(patch_shape_hw[0]), int(patch_shape_hw[1]))
        if shape[0] <= 0 or shape[1] <= 0:
            return None
        chw = np.moveaxis(np.asarray(frame_hwc), -1, 0)
        rgb, geom = reproject_cropped(chw, frame_wcs, patch_wcs, shape, emit=emit)
    except Exception as exc:  # noqa: BLE001 - never break the SDS stack
        _LOGGER.warning(
            "[SDS] fast reprojection unavailable, using reproject_interp fallback: %s",
            exc,
        )
        return None

    patch = np.asarray(rgb, dtype=np.float32)
    geom = np.asarray(geom, dtype=bool)
    finite = np.all(np.isfinite(patch), axis=-1)
    weight = np.where(finite, geom, False).astype(np.float32)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    return patch, weight


def sds_reproject_stats_payload() -> dict:
    """Return the process-local R21 reprojection-path counters (fast/fallback)."""
    try:
        return get_reproject_path_stats().to_dict()
    except Exception:  # noqa: BLE001 - observability is never fatal
        return {}
