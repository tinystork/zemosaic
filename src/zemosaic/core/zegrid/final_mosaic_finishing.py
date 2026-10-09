"""ZM-ZEGRID-R23 rework-3 — final-mosaic finishing (legacy light-DBE aesthetic + equalize + hole fill + uint16).

Post-assembly finishing applied to the ASSEMBLED mosaic BEFORE the outputs are
written. Pure ``numpy``/``scipy`` (no new dependency). Opt-in, fail-safe, and
fully documented. This module does **NOT** touch the per-cell / stacking
science: it only post-processes ``AssembledCanvas.science`` (H, W, 3 float32).

Rework-3 contract (human-gate resolved, Tristan):

* **Raw science is the immutable photometric reference** — always written FIRST,
  pre-finishing, never mutated, never clamped/abs'd/offset.

* **Aesthetic output is explicitly visual/non-photometric.** It is produced by
  the *exact legacy light-DBE* algorithm (optionally followed by RGB equalization
  and user-controlled aesthetic hole fill). Its diffuse-flux loss is a documented
  property of the aesthetic branch only.

* **Classic-compatible naming.** The existing ``export_aesthetic_fits`` checkbox
  and ``scientific_fits_suffix`` / ``aesthetic_fits_suffix`` keys (defaults
  ``_science`` / ``_aesthetic``) drive the file layout on the base ``mosaic_grid``:

  - ``export_aesthetic_fits=false`` → ONE primary raw science float32 as
    ``mosaic_grid.fits`` (pre-finishing, SCI role), no aesthetic companion;
  - ``export_aesthetic_fits=true`` → raw science as
    ``mosaic_grid<clean scientific suffix>.fits`` + aesthetic float32 as
    ``mosaic_grid<clean aesthetic suffix>.fits`` (defaults
    ``mosaic_grid_science.fits`` + ``mosaic_grid_aesthetic.fits``).

  Suffix cleaning/collision matches the Classic worker (leading ``_``,
  ``[A-Za-z0-9_-]``, equal-suffix fallback).

* **Legacy light-DBE (exact historical algorithm).** Variation-only correction
  ``corrected = ch_filled - (bg - bg_med)`` with protected bright-object pixels
  restored unchanged, a Gaussian background model (``mode="nearest"``), and the
  historical strength maps (sigma / obj_k / dilation). The DBE candidate retains
  its NaN/coverage mask (uncovered pixels stay NaN).

* **User-controlled aesthetic hole fill.** Runs ONLY when
  ``aesthetic_hole_fill_enabled`` is true and reuses the shared Classic helper;
  when disabled the aesthetic preserves the DBE coverage/NaN mask exactly.

* **Safety guard (no negative explosion, no gross worsening).** After building
  the candidate, if any channel creates a MATERIAL negative-fraction explosion
  or GROSSLY worsens a fixed input-derived background-uniformity metric, the
  whole RGB is returned unchanged (atomic no-op, bit-identical incl. NaNs) with
  a visible reason. The guard never imposes the >=0.90 diffuse-flux requirement.

The honoured user settings:

* ``final_mosaic_dbe_*``  — legacy light-DBE on the assembled mosaic (aesthetic).
* ``grid_rgb_equalize``   — per-channel background (sky) equalization.
* ``aesthetic_hole_fill_*`` — optional visual hole completion (aesthetic only).
* ``save_final_as_uint16``— an integer (uint16) render of the finished (aesthetic) science.

When every setting is disabled the input array is returned **unchanged**
(identity) so the written aesthetic FITS is bit-equal to the raw reference.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Legacy strength presets (EXACT historical Grid mappings).
# ---------------------------------------------------------------------------
#
# From the authoritative legacy reference
# ``grid_mode._apply_grid_final_dbe`` (``git show origin/HEAD:src/zemosaic/grid_mode.py``):
#
#   weak / low     -> sigma=24, obj_k=3.0, dilate=2
#   normal         -> sigma=36, obj_k=2.8, dilate=3
#   strong / high  -> sigma=52, obj_k=2.5, dilate=4
#   aggressive     -> sigma=68, obj_k=2.2, dilate=5
#
# ``sigma`` is the Gaussian background-model sigma (px); ``obj_k`` the compact
# object-detection k-sigma; ``obj_dilate_px`` the object-mask dilation radius.
# These are reproduced EXACTLY; the legacy aliases are accepted.
DBE_STRENGTH_PRESETS: dict[str, dict[str, float | int]] = {
    "weak": {"sigma": 24.0, "obj_k": 3.0, "obj_dilate_px": 2},
    "low": {"sigma": 24.0, "obj_k": 3.0, "obj_dilate_px": 2},
    "normal": {"sigma": 36.0, "obj_k": 2.8, "obj_dilate_px": 3},
    "strong": {"sigma": 52.0, "obj_k": 2.5, "obj_dilate_px": 4},
    "high": {"sigma": 52.0, "obj_k": 2.5, "obj_dilate_px": 4},
    "aggressive": {"sigma": 68.0, "obj_k": 2.2, "obj_dilate_px": 5},
}
DEFAULT_STRENGTH = "normal"

# Canonical strength names (the first in each legacy alias group).
_CANONICAL = {"low": "weak", "high": "strong", "aggressive": "aggressive"}

# Subtraction is always variation-only (factor 1.0); kept explicit/auditable.
DBE_SUBTRACTION_FACTOR = 1.0

# Safety guard (aesthetic output only; does NOT impose science neutrality).
NEG_FRAC_EXPLOSION_ABS = 0.05   # candidate - before negative fraction (absolute) triggering no-op
UNIFORMITY_WORSE_FACTOR = 1.5   # candidate_full_std > before_full_std * this -> no-op
UNIFORMITY_BOXES = 8            # NxN box grid for the fixed input-derived sky metric
UNIFORMITY_MIN_BOX_SAMPLES = 25  # min sky samples per box to count it
UNIFORMITY_MIN_BOXES = 16       # min counted boxes to trust the metric
BOUNDARY_ERODE_PX = 2           # erode coverage boundary before the safety metric


def _safe_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return float(fallback)
        if isinstance(value, str) and not value.strip():
            return float(fallback)
        return float(value)
    except Exception:
        return float(fallback)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        if value is None:
            return int(fallback)
        if isinstance(value, str) and not value.strip():
            return int(fallback)
        return int(float(value))
    except Exception:
        return int(fallback)


# Fixed truthy/falsy string vocabulary (case-insensitive) for boolean coercion.
# Unknown strings, empty/whitespace strings, None, and unrecognised types fall
# back to the field's own default (never silently ``True`` — the classic
# ``bool('false') is True`` foot-gun is explicitly avoided).
_BOOL_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}
_BOOL_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", "disable", "disabled", "none", "null"}


def _coerce_bool(value: Any, fallback: bool, notes: list | None = None) -> bool:
    """Classic-compatible boolean coercion (never a naïve ``bool(value)``).

    Accepts:

    * a real ``bool`` (returned as-is);
    * ``0``/``1`` numerics (0/0.0 → False, anything else → True);
    * case-insensitive strings from a fixed vocabulary —
      ``true/false``, ``yes/no``, ``on/off``, ``enable/disable``
      (plus short aliases ``t/f``, ``y/n``, ``1/0``);
    * ``None``, empty/whitespace strings, and ANY unrecognised value →
      the caller-supplied ``fallback`` (per-field default), so an unknown string
      can never silently become ``True``.

    Unknown (non-empty) strings and unrecognised types are recorded in
    ``notes`` (as ``(value, fallback)`` tuples) for visible surfacing; ``None``
    and empty strings are the normal absent/default path and are not recorded.
    """
    if value is None:
        return bool(fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value not in (0, 0.0)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _BOOL_TRUE_STRINGS:
            return True
        if s in _BOOL_FALSE_STRINGS:
            return False
        # empty/whitespace -> per-field fallback (normal default path).
        if not s:
            return bool(fallback)
        # unknown string -> per-field fallback (visible, never silently True).
        if notes is not None:
            notes.append((value, bool(fallback)))
        return bool(fallback)
    # unrecognised type (list/dict/...) -> per-field fallback.
    if notes is not None:
        notes.append((repr(value), bool(fallback)))
    return bool(fallback)


def _config_get(zconfig: Any, key: str, fallback: Any) -> Any:
    if zconfig is None:
        return fallback
    try:
        return getattr(zconfig, key, fallback)
    except Exception:
        return fallback


def resolve_dbe_strength(zconfig: Any) -> dict:
    """Resolve DBE strength into effective legacy light-DBE params + provenance.

    Returns ``{strength, params_source, params}``.

    * ``strength`` — the effective strength name (canonical:
      ``weak``/``normal``/``strong``/``aggressive``).
    * ``params_source`` — ``preset:<name>`` or ``preset:<alias>``.
    * ``params`` — the effective ``{sigma, obj_k, obj_dilate_px}``.

    Legacy aliases ``low``/``high`` map to their canonical presets. An
    unknown/empty strength falls back to ``normal``. A stored ``custom`` strength
    has NO Gaussian-``sigma`` config key (the legacy block-median
    ``sample_step``/``smoothing`` fields are not meaningful for
    ``legacy_grid_light_dbe``), so ``custom`` falls back to ``normal`` with an
    explicit ``custom_fallback`` record — never a silent reinterpretation, and no
    new setting/control is introduced.
    """
    raw = str(_config_get(zconfig, "final_mosaic_dbe_strength", DEFAULT_STRENGTH)
              or DEFAULT_STRENGTH).strip().lower()

    if raw in DBE_STRENGTH_PRESETS:
        canonical = _CANONICAL.get(raw, raw)
        return {
            "strength": canonical,
            "params_source": f"preset:{raw}",
            "params": dict(DBE_STRENGTH_PRESETS[raw]),
        }

    if raw == "custom":
        # The legacy light-DBE algorithm is defined by (sigma, obj_k, obj_dilate_px).
        # obj_k / obj_dilate_px ARE existing config keys (object-mask params), but
        # the Gaussian background-model scale (sigma) has NO existing config key
        # and cannot be derived from the legacy block-median ``sample_step`` /
        # ``smoothing`` fields without silently reinterpreting them. Per the
        # rework-2 contract (and the product-owner "no new settings" constraint),
        # a stored ``custom`` therefore falls back to the documented ``normal``
        # preset with an explicit WARN — never a silent reinterpretation.
        return {
            "strength": DEFAULT_STRENGTH,
            "params_source": "preset:normal",
            "params": dict(DBE_STRENGTH_PRESETS[DEFAULT_STRENGTH]),
            "custom_fallback": (
                "custom requested but the legacy light-DBE Gaussian sigma has no "
                "existing config key; block-median sample_step/smoothing are not "
                "meaningful for legacy_grid_light_dbe, so fell back to normal "
                "(config preserved, not reinterpreted)"
            ),
        }

    # Invalid / unknown -> documented normal.
    return {
        "strength": DEFAULT_STRENGTH,
        "params_source": "preset:normal",
        "params": dict(DBE_STRENGTH_PRESETS[DEFAULT_STRENGTH]),
    }


def clean_fits_suffix(value: Any, default: str) -> str:
    """Clean a FITS output suffix EXACTLY like the Classic worker.

    Classic reference (``zemosaic_worker.run`` nested ``_clean_suffix``):

    * strip; empty -> ``default``;
    * ensure a single leading ``_``;
    * keep only ``[A-Za-z0-9_-]``;
    * empty after cleaning -> ``default``.

    The collision fallback (aesthetic == scientific -> ``_aesthetic``) is applied
    by :func:`resolve_finishing_config`.
    """
    sfx = str(value or default).strip()
    if not sfx:
        sfx = default
    if not sfx.startswith("_"):
        sfx = f"_{sfx}"
    sfx = "".join(ch for ch in sfx if ch.isalnum() or ch in {"_", "-"})
    if not sfx:
        sfx = default
    return sfx


def resolve_dbe_params(zconfig: Any, strength: str | None = None) -> dict:
    """Resolve effective legacy light-DBE params (back-compat name).

    ``strength`` is accepted for signature compatibility but ignored: strength
    always comes from the config (see :func:`resolve_dbe_strength`).
    """
    return resolve_dbe_strength(zconfig)["params"]


def resolve_finishing_config(
    zconfig: Any,
    *,
    grid_rgb_equalize: bool | None = None,
    save_final_as_uint16: bool | None = None,
) -> dict:
    """Resolve the finishing configuration from the config object + call args.

    Returns a dict:

    * ``dbe_enabled``            — from ``final_mosaic_dbe_enabled`` (default True).
    * ``dbe_strength``           — effective strength (weak/normal/strong/aggressive).
    * ``dbe_params_source``      — ``preset:*`` (or ``preset:normal`` after a
      ``custom`` fallback).
    * ``dbe_params``             — effective legacy params ``{sigma, obj_k, obj_dilate_px}``.
    * ``dbe_subtraction_factor`` — always 1.0 (variation-only correction).
    * ``rgb_equalize``           — from ``grid_rgb_equalize`` (default True).
    * ``save_uint16``            — from ``save_final_as_uint16`` (default False).
    * ``export_aesthetic_fits``  — from the existing ``export_aesthetic_fits`` key
      (default False).
    * ``scientific_fits_suffix`` / ``aesthetic_fits_suffix`` — cleaned Classic suffixes.
    * ``hole_fill_enabled`` / ``hole_fill_max_radius_px`` / ``hole_fill_blend`` /
      ``hole_fill_only_near_seams`` / ``hole_fill_protect_stars_details`` — the
      existing ``aesthetic_hole_fill_*`` settings.
    * ``bool_coercion_fallbacks`` — list of ``{field, value, fallback}`` records
      for any boolean that fell back due to an unknown/typed string (visible,
      never silently ``True``; see :func:`_coerce_bool`).

    All booleans use Classic-compatible coercion (real bool / 0/1 / typed
    strings / empty / None), never a naïve ``bool(value)``.
    """
    # ZM-ZEGRID-R24 (A4): robust, Classic-compatible boolean coercion. Every
    # boolean below is coerced via :func:`_coerce_bool` so a string like
    # ``'false'``/``'off'``/``'no'``/``'disable'`` never silently becomes True
    # (the classic ``bool('false') is True`` foot-gun). Unknown strings fall back
    # to each field's default and are recorded (visible) in
    # ``bool_coercion_fallbacks``. No new config key or GUI control is added.
    bool_coercion_fallbacks: list[dict[str, Any]] = []

    def _cb(field: str, value: Any, fallback: bool) -> bool:
        notes: list[tuple[Any, bool]] = []
        result = _coerce_bool(value, fallback, notes=notes)
        for raw_val, fb in notes:
            bool_coercion_fallbacks.append(
                {"field": field, "value": raw_val, "fallback": bool(fb)}
            )
        return result

    dbe_enabled = _cb("final_mosaic_dbe_enabled",
                      _config_get(zconfig, "final_mosaic_dbe_enabled", None), True)

    strength_info = resolve_dbe_strength(zconfig)

    if grid_rgb_equalize is not None:
        rgb_equalize = _cb("grid_rgb_equalize (run arg)", grid_rgb_equalize, True)
    else:
        rgb_equalize = _cb("grid_rgb_equalize",
                           _config_get(zconfig, "grid_rgb_equalize", None), True)

    if save_final_as_uint16 is not None:
        save_uint16 = _cb("save_final_as_uint16 (run arg)", save_final_as_uint16, False)
    else:
        save_uint16 = _cb("save_final_as_uint16",
                          _config_get(zconfig, "save_final_as_uint16", None), False)

    # ZM-ZEGRID-R23 rework-3 (H1): resolve the EXISTING output naming + hole-fill
    # settings from the same ``zconfig`` the Classic worker uses. These are the
    # established GUI keys; NO new key or control is introduced. The default is
    # ``export_aesthetic_fits=false`` (matches the Classic worker's effective
    # fallback and preserves the old primary-name contract ``mosaic_grid.fits``).
    export_aesthetic_fits = _cb("export_aesthetic_fits",
                                _config_get(zconfig, "export_aesthetic_fits", False), False)
    scientific_fits_suffix = clean_fits_suffix(
        _config_get(zconfig, "scientific_fits_suffix", "_science"), "_science"
    )
    aesthetic_fits_suffix = clean_fits_suffix(
        _config_get(zconfig, "aesthetic_fits_suffix", "_aesthetic"), "_aesthetic"
    )
    if aesthetic_fits_suffix == scientific_fits_suffix:
        aesthetic_fits_suffix = "_aesthetic"

    hole_fill_enabled = _cb("aesthetic_hole_fill_enabled",
                            _config_get(zconfig, "aesthetic_hole_fill_enabled", True), True)
    try:
        hole_fill_max_radius_px = int(_config_get(
            zconfig, "aesthetic_hole_fill_max_radius_px", 64) or 64)
    except Exception:
        hole_fill_max_radius_px = 64
    try:
        hole_fill_blend = float(_config_get(zconfig, "aesthetic_hole_fill_blend", 0.70) or 0.70)
    except Exception:
        hole_fill_blend = 0.70
    hole_fill_only_near_seams = _cb("aesthetic_hole_fill_only_near_seams",
                                    _config_get(zconfig, "aesthetic_hole_fill_only_near_seams", True), True)
    hole_fill_protect_stars_details = _cb("aesthetic_hole_fill_protect_stars_details",
                                          _config_get(zconfig, "aesthetic_hole_fill_protect_stars_details", True), True)

    return {
        "dbe_enabled": bool(dbe_enabled),
        "dbe_strength": strength_info["strength"],
        "dbe_params_source": strength_info["params_source"],
        "dbe_params": strength_info["params"],
        "dbe_custom_fallback": strength_info.get("custom_fallback"),
        "dbe_subtraction_factor": 1.0,
        "rgb_equalize": bool(rgb_equalize),
        "save_uint16": bool(save_uint16),
        "export_aesthetic_fits": bool(export_aesthetic_fits),
        "scientific_fits_suffix": scientific_fits_suffix,
        "aesthetic_fits_suffix": aesthetic_fits_suffix,
        "hole_fill_enabled": bool(hole_fill_enabled),
        "hole_fill_max_radius_px": int(hole_fill_max_radius_px),
        "hole_fill_blend": float(hole_fill_blend),
        "hole_fill_only_near_seams": bool(hole_fill_only_near_seams),
        "hole_fill_protect_stars_details": bool(hole_fill_protect_stars_details),
        "bool_coercion_fallbacks": bool_coercion_fallbacks,
    }


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def _robust_sky(values: np.ndarray) -> tuple[float, float]:
    """Robust (median / MAD) sky level and sigma of a 1-D float array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = float(1.4826 * mad)
    return med, sigma


def _eroded_valid_mask(valid: np.ndarray, iterations: int = BOUNDARY_ERODE_PX) -> np.ndarray:
    """Erode the coverage boundary so edge pixels don't contaminate the metric."""
    from scipy.ndimage import binary_erosion
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid) or iterations <= 0:
        return valid
    return valid & binary_erosion(valid, iterations=int(iterations))


def _uniformity_metric(
    channel: np.ndarray, sky_mask: np.ndarray, boxes: int = UNIFORMITY_BOXES
) -> tuple[dict | None, int]:
    """Fixed input-derived background-uniformity metric (safety guard only).

    Splits the image into ``boxes x boxes`` fixed boxes; for each box computes the
    median of the sky-masked pixels (boxes with fewer than
    ``UNIFORMITY_MIN_BOX_SAMPLES`` samples are skipped). The metric is the std of
    the box medians (robust sigma also reported). Returns
    ``(metric_dict, n_boxes_used)``; the dict is None when fewer than
    ``UNIFORMITY_MIN_BOXES`` boxes qualify (metric untrustworthy).
    """
    h, w = channel.shape
    meds: list[float] = []
    for i in range(boxes):
        for j in range(boxes):
            sl = (slice(i * h // boxes, (i + 1) * h // boxes),
                  slice(j * w // boxes, (j + 1) * w // boxes))
            seg = channel[sl][sky_mask[sl]]
            if seg.size >= UNIFORMITY_MIN_BOX_SAMPLES:
                meds.append(float(np.median(seg)))
    meds = np.asarray(meds, dtype=np.float64)
    if meds.size < UNIFORMITY_MIN_BOXES:
        return None, int(meds.size)
    med = float(np.median(meds))
    robust = float(1.4826 * np.median(np.abs(meds - med)))
    return {"std": float(np.std(meds)), "robust": robust, "n_boxes": int(meds.size)}, int(meds.size)


def _channel_stats(ch: np.ndarray, valid: np.ndarray) -> dict:
    vals = ch[valid]
    return {
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "median": float(np.nanmedian(vals)),
        "neg_frac": float(np.mean(vals < 0)),
    }


# ---------------------------------------------------------------------------
# Legacy light-DBE (exact historical algorithm, aesthetic output)
# ---------------------------------------------------------------------------

def _legacy_light_dbe_channel(
    ch: np.ndarray, valid_hw: np.ndarray, *, sigma: float, obj_k: float, obj_dilate_px: int
) -> tuple[np.ndarray, dict]:
    """Apply the EXACT legacy light-DBE to a single channel (historical algorithm).

    Port of ``grid_mode._apply_grid_final_dbe`` (variation-only correction):

    * per-channel median / MAD / robust sigma;
    * compact-object mask ``ch > median + obj_k * robust_sigma``, dilated;
    * background-only reference ``fill_ref = median(ch[bg])``;
    * ``ch_filled`` fills uncovered pixels with ``fill_ref``;
    * ``ch_model`` replaces protected object pixels with ``fill_ref``;
    * Gaussian background ``bg = gaussian_filter(ch_model, sigma, mode="nearest")``;
    * ``bg_med = median(bg[bg])``;
    * ``corrected = ch_filled - (bg - bg_med)`` (variation-only, sky/DC preserved);
    * protected object pixels restored unchanged (``where(obj, ch_filled, corrected)``).

    Returns ``(ch_out, info)``. ``ch_out`` is NaN over uncovered pixels (the
    caller fills/leaves NaN as the array already had).
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    ch = np.asarray(ch, dtype=np.float32)
    info: dict = {"sigma": float(sigma), "obj_k": float(obj_k),
                  "obj_dilate_px": int(obj_dilate_px)}

    ch_finite = np.isfinite(ch)
    ch_valid = valid_hw & ch_finite
    if not np.any(ch_valid):
        info["reason"] = "no_valid"
        return ch.copy(), info

    median = float(np.nanmedian(ch[ch_valid]))
    mad = float(np.nanmedian(np.abs(ch[ch_valid] - median)))
    robust_sigma = float(1.4826 * mad)
    info["median"] = median
    info["robust_sigma"] = robust_sigma

    obj_thr = float(median + obj_k * robust_sigma)
    obj_mask = ch_valid & (ch > obj_thr)
    if obj_dilate_px > 0:
        obj_mask = binary_dilation(obj_mask, iterations=int(obj_dilate_px))
    info["obj_frac"] = float(np.count_nonzero(obj_mask) / max(1, int(np.count_nonzero(ch_valid))))

    bg_valid = ch_valid & (~obj_mask)
    if not np.any(bg_valid):
        bg_valid = ch_valid

    fill_ref = float(np.nanmedian(ch[bg_valid])) if np.any(bg_valid) else median
    info["fill_ref"] = fill_ref

    ch_filled = np.where(ch_valid, ch, fill_ref).astype(np.float32)
    ch_model = np.where(obj_mask, fill_ref, ch_filled).astype(np.float32)

    bg = gaussian_filter(ch_model, sigma=float(sigma), mode="nearest")
    bg_med = float(np.nanmedian(bg[bg_valid])) if np.any(bg_valid) else fill_ref
    info["bg_med"] = bg_med

    corrected = ch_filled - (bg - bg_med)
    corrected = np.where(obj_mask, ch_filled, corrected).astype(np.float32)

    ch_out = np.where(ch_valid, corrected, np.nan).astype(np.float32)
    return ch_out, info


def apply_dbe(
    science: np.ndarray,
    valid: np.ndarray,
    *,
    strength: str,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """Apply the LEGACY light-DBE (variation-only) to an RGB array (aesthetic).

    ``corrected = ch_filled - (bg - bg_med)`` per channel with protected bright
    object pixels restored unchanged; Gaussian model ``mode="nearest"``; the
    historical strength maps (sigma / obj_k / dilation).

    Safety guard (NOT a science-neutrality gate): after building the candidate,
    if ANY channel creates a MATERIAL negative-fraction explosion or GROSSLY
    worsens a fixed input-derived background-uniformity metric, the whole RGB is
    returned unchanged (atomic no-op) with a visible reason. The guard never
    imposes the >=0.90 diffuse-flux requirement — the aesthetic output is
    allowed to lose diffuse flux by design (the raw science reference is
    untouched and authoritative).

    ``strength`` / ``params`` / ``params_source`` are recorded for the manifest.
    """
    science = np.asarray(science, dtype=np.float32)
    info: dict = {
        "strength": strength,
        "params": dict(params),
        "channels": [],
        "attempted": True,
        "applied": False,
        "reason": "",
        "algorithm": "legacy_grid_light_dbe",
        "guard": {
            "neg_frac_explosion_abs": float(NEG_FRAC_EXPLOSION_ABS),
            "uniformity_worse_factor": float(UNIFORMITY_WORSE_FACTOR),
            "metric": "std of per-box medians over a fixed input-derived valid mask",
            "note": (
                "safety guard only: no-op on negative-fraction explosion or gross "
                "background-uniformity worsening; does NOT require diffuse-flux "
                "neutrality (aesthetic output may lose diffuse flux by design)"
            ),
        },
    }

    if science.ndim != 3 or science.shape[-1] != 3:
        info["reason"] = "non_rgb"
        info["attempted"] = False
        return science.copy(), info

    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        info["reason"] = "no_valid"
        info["attempted"] = False
        return science.copy(), info

    sigma = float(params.get("sigma", 36.0))
    obj_k = float(params.get("obj_k", 2.8))
    obj_dilate_px = int(params.get("obj_dilate_px", 3))

    candidates: list[tuple[int, np.ndarray]] = []
    before_full: list[float] = []
    after_full: list[float] = []
    neg_exploded = False

    for c in range(3):
        ch = science[..., c]
        ch_valid = valid & np.isfinite(ch)
        cinfo: dict = {"channel": int(c), "applied": False}
        cinfo["before"] = _channel_stats(ch, ch_valid) if np.any(ch_valid) else None

        cand, ch_info = _legacy_light_dbe_channel(
            ch, valid, sigma=sigma, obj_k=obj_k, obj_dilate_px=obj_dilate_px
        )
        cinfo.update({k: ch_info[k] for k in ch_info if k != "reason"})
        if ch_info.get("reason"):
            cinfo["reason"] = ch_info["reason"]

        # ZM-ZEGRID-R23 rework-3 (H2): the returned aesthetic must RETAIN the DBE
        # candidate mask (NaN outside valid coverage). A temporary finite plane
        # is used ONLY for scoring the safety guard; it is never returned.
        fill_ref = float(ch_info.get("fill_ref", 0.0))
        cand_score = np.where(np.isfinite(cand), cand, fill_ref).astype(np.float32)

        # negative-fraction + uniformity diagnostics (on the temp finite plane).
        if np.any(ch_valid):
            cinfo["after"] = _channel_stats(cand_score, ch_valid)
            before_neg = float(cinfo["before"]["neg_frac"])
            after_neg = float(cinfo["after"]["neg_frac"])
            cinfo["neg_frac_delta"] = after_neg - before_neg
            if after_neg - before_neg > NEG_FRAC_EXPLOSION_ABS:
                neg_exploded = True

            full_mask = _eroded_valid_mask(ch_valid)
            bm, _ = _uniformity_metric(ch, full_mask)
            am, _ = _uniformity_metric(cand_score, full_mask)
            if bm is not None:
                cinfo["before_full_std"] = bm["std"]
                before_full.append(bm["std"])
            if am is not None:
                cinfo["after_full_std"] = am["std"]
                after_full.append(am["std"])
            if bm is not None and am is not None and bm["std"] > 0:
                cinfo["full_ratio"] = am["std"] / bm["std"]
            else:
                cinfo["full_ratio"] = None

        candidates.append((c, cand))  # NaN-preserving candidate
        info["channels"].append(cinfo)

    # --- safety guard: gross uniformity worsening ---
    uniformity_worse = False
    for b, a in zip(before_full, after_full):
        if b > 0 and a > b * UNIFORMITY_WORSE_FACTOR:
            uniformity_worse = True

    if neg_exploded:
        info["reason"] = "negative_fraction_explosion"
        info["applied"] = False
        return science.copy(), info
    if uniformity_worse:
        info["reason"] = "uniformity_worsened"
        info["applied"] = False
        return science.copy(), info

    out = science.copy()
    for c, cand in candidates:
        out[..., c] = cand
        for cinfo in info["channels"]:
            if cinfo["channel"] == c:
                cinfo["applied"] = True
    info["applied"] = True
    info["reason"] = ""
    return out, info


# ---------------------------------------------------------------------------
# RGB background equalization
# ---------------------------------------------------------------------------

def equalize_rgb(science: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Equalize per-channel background (sky) levels additively.

    Each channel's robust sky median (over the valid coverage) is shifted to the
    three-channel mean, so the channel backgrounds agree while the relative
    per-channel signal (stars) is preserved. Returns ``(equalized, info)``.
    """
    science = np.asarray(science, dtype=np.float32)
    out = science.copy()
    info: dict = {"applied": False, "skies_before": [], "skies_after": [], "target": 0.0}

    if science.ndim != 3 or science.shape[-1] != 3:
        info["reason"] = "non_rgb"
        return out, info

    skies: list[float] = []
    for c in range(3):
        ch = science[..., c]
        v = valid & np.isfinite(ch)
        skies.append(_robust_sky(ch[v])[0] if np.any(v) else 0.0)
    info["skies_before"] = [float(s) for s in skies]

    if not np.any(valid):
        info["reason"] = "no_valid"
        return out, info

    target = float(np.mean(skies))
    info["target"] = float(target)
    for c in range(3):
        offset = float(skies[c] - target)
        v = valid & np.isfinite(science[..., c])
        if np.any(v):
            out[..., c][v] = science[..., c][v] - offset
    info["skies_after"] = [
        float(_robust_sky(out[..., c][valid & np.isfinite(out[..., c])])[0])
        if np.any(valid & np.isfinite(out[..., c])) else 0.0
        for c in range(3)
    ]
    info["applied"] = True
    return out, info


# ---------------------------------------------------------------------------
# uint16 output
# ---------------------------------------------------------------------------

def to_uint16(science: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map the finished (aesthetic) float science to a uint16 array.

    ``uint16 = clip(round(65535 * (v - vmin) / (vmax - vmin)), 0, 65535)`` where
    ``vmin`` / ``vmax`` are the 1st / 99.9th percentile of the valid (finite)
    pixels pooled across channels. Invalid (NaN / hole) pixels map to 0. The
    scaling constants are recorded so the mapping is auditable. This render is
    derived from the finished (aesthetic) float science and is never a scientific
    reference.
    """
    science = np.asarray(science, dtype=np.float32)
    finite = np.isfinite(science)
    info: dict = {"applied": False, "vmin": 0.0, "vmax": 65535.0,
                  "lo_pct": 1.0, "hi_pct": 99.9, "formula": "clip(round(65535*(v-vmin)/(vmax-vmin)),0,65535)"}

    vals = science[finite]
    if vals.size == 0:
        return np.zeros(science.shape[:2] + (3,), dtype=np.uint16), info
    vmin = float(np.percentile(vals, 1.0))
    vmax = float(np.percentile(vals, 99.9))
    if vmax <= vmin:
        vmax = vmin + 1.0
    info["vmin"] = float(vmin)
    info["vmax"] = float(vmax)

    scaled = (science.astype(np.float64) - vmin) / (vmax - vmin) * 65535.0
    scaled = np.clip(np.round(scaled), 0, 65535)
    out = np.zeros(science.shape[:2] + (3,), dtype=np.uint16)
    out[finite] = scaled[finite].astype(np.uint16)
    info["applied"] = True
    return out, info


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class FinishingResult:
    science: np.ndarray            # (H, W, 3) float32 (finished aesthetic, or the input itself when off)
    uint16: np.ndarray | None      # (H, W, 3) uint16, or None
    info: dict                     # full finishing record for the manifest/log


def apply_final_mosaic_finishing(
    science: np.ndarray,
    coverage: np.ndarray,
    *,
    config: dict | None = None,
) -> FinishingResult:
    """Apply the final-mosaic finishing (light-DBE -> RGB equalize -> hole fill -> uint16).

    ``config`` is the dict from :func:`resolve_finishing_config`. When ``config``
    is None (or every feature is disabled) the input array is returned by
    identity and no work is done (bit-equal output). The order is documented:
    light-DBE first, then RGB equalization, then (optional, user-controlled)
    aesthetic hole fill, then the uint16 render.

    The DBE candidate retains its NaN/coverage mask (uncovered pixels stay NaN).
    Hole fill runs ONLY when ``aesthetic_hole_fill_enabled`` is true and reuses
    the shared Classic helper; it never fills when disabled. The uint16 render is
    derived from the post-aesthetic branch (after DBE/RGB/hole fill).
    """
    from .aesthetic_hole_fill import apply_aesthetic_hole_fill

    cfg = dict(config or {})
    science = np.asarray(science, dtype=np.float32)
    coverage = np.asarray(coverage)

    info: dict = {
        "enabled": False,
        "failed": False,
        "failure_reason": "",
        "dbe": {"enabled": False, "applied": False},
        "rgb_equalize": {"enabled": False, "applied": False},
        "hole_fill": {"enabled": False, "applied": False},
        "uint16": {"enabled": False, "written": False},
    }

    # Valid coverage mask (finite + covered). Per-channel finiteness is handled
    # inside each feature; here we build the shared coverage plane.
    valid = (coverage > 0) & np.any(np.isfinite(science), axis=-1)

    dbe_enabled = bool(cfg.get("dbe_enabled", False))
    rgb_equalize = bool(cfg.get("rgb_equalize", False))
    save_uint16 = bool(cfg.get("save_uint16", False))
    hole_fill_enabled = bool(cfg.get("hole_fill_enabled", False))

    info["dbe"]["enabled"] = dbe_enabled
    info["rgb_equalize"]["enabled"] = rgb_equalize
    info["hole_fill"]["enabled"] = hole_fill_enabled
    info["uint16"]["enabled"] = save_uint16
    info["enabled"] = bool(dbe_enabled or rgb_equalize or hole_fill_enabled or save_uint16)

    # When nothing is enabled, return the input by identity (bit-equal path).
    if not info["enabled"]:
        return FinishingResult(science=science, uint16=None, info=info)

    out = science

    if dbe_enabled:
        strength = str(cfg.get("dbe_strength", DEFAULT_STRENGTH))
        params = dict(cfg.get("dbe_params", DBE_STRENGTH_PRESETS[DEFAULT_STRENGTH]))
        out, dbe_info = apply_dbe(out, valid, strength=strength, params=params)
        info["dbe"].update(dbe_info)
        info["dbe"]["enabled"] = True
        info["dbe"]["params_source"] = str(cfg.get("dbe_params_source", "preset:normal"))
        info["dbe"]["subtraction_factor"] = float(cfg.get("dbe_subtraction_factor", 1.0))
        if cfg.get("dbe_custom_fallback"):
            info["dbe"]["custom_fallback"] = str(cfg["dbe_custom_fallback"])

    if rgb_equalize:
        out, rgb_info = equalize_rgb(out, valid)
        info["rgb_equalize"].update(rgb_info)
        info["rgb_equalize"]["enabled"] = True

    if hole_fill_enabled:
        out, hf_info = apply_aesthetic_hole_fill(
            out,
            coverage_hw=coverage,
            enabled=True,
            max_radius_px=int(cfg.get("hole_fill_max_radius_px", 64)),
            blend=float(cfg.get("hole_fill_blend", 0.70)),
            only_near_seams=bool(cfg.get("hole_fill_only_near_seams", True)),
            protect_stars_details=bool(cfg.get("hole_fill_protect_stars_details", True)),
        )
        info["hole_fill"].update(hf_info)
        info["hole_fill"]["enabled"] = True

    uint16_data: np.ndarray | None = None
    if save_uint16:
        uint16_data, u16_info = to_uint16(out, valid)
        info["uint16"].update(u16_info)
        info["uint16"]["enabled"] = True
        info["uint16"]["written"] = bool(u16_info.get("applied", False))

    return FinishingResult(science=out, uint16=uint16_data, info=info)
