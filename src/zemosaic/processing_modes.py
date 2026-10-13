"""Explicit processing-mode resolution for ZeMosaic (ZM-ZEGRID-R35A).

Historically the ONLY way to reach the ZeGrid engine was the *presence* of a
``stack_plan.csv`` inside the input folder, and that same file also *reduced*
the ZeGrid input to its own row subset.  ``stack_plan.csv`` was designed for
ZeAnalyser/ZSSS (quality selection + ordering) and its format must NOT change.

This module makes the running mode an EXPLICIT, single, non-ambiguous decision,
resolved once before any scientific worker runs:

    classic                -> classic raw-frames pipeline
    existing_master_tiles  -> user-provided master tiles
    sds                    -> ZeSupaDupStack raw frames
    zegrid                 -> ZeGrid raw frames (full FITS directory)

``stack_plan.csv`` no longer participates in this decision and no longer feeds
ZeGrid (see :mod:`zemosaic.input_inventory`).

The resolver is deterministic and auditable: an incoherent combination coming
from an old config or a programmatic call is *resolved* by a documented
precedence and reported in the returned info dict (never silently ignored).
"""

from __future__ import annotations

from typing import Any, Mapping

CLASSIC = "classic"
EXISTING_MASTER_TILES = "existing_master_tiles"
SDS = "sds"
ZEGRID = "zegrid"

#: All canonical modes.
PROCESSING_MODES: tuple[str, ...] = (CLASSIC, EXISTING_MASTER_TILES, SDS, ZEGRID)

#: The three mutually-exclusive "special" pipelines (everything else is classic).
SPECIAL_MODES: tuple[str, ...] = (EXISTING_MASTER_TILES, SDS, ZEGRID)

#: Mode selector key (may be ``"auto"`` = derive from the legacy booleans).
MODE_KEY = "processing_mode"

#: Legacy boolean keys (kept for backward compatibility, one per mode).
MASTER_TILES_KEY = "use_existing_master_tiles"
SDS_KEY = "sds_mode_default"
ZEGRID_KEY = "zegrid_mode_enabled"

#: Documented precedence used ONLY when an incoherent combination reaches the
#: engine (e.g. a stale config or a programmatic call).  ``existing_master_tiles``
#: wins because it describes the *kind of input*; ``zegrid`` then ``sds``.
CONFLICT_PRECEDENCE: tuple[str, ...] = (EXISTING_MASTER_TILES, ZEGRID, SDS)

_TRUE_TOKENS = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE_TOKENS = {"0", "false", "no", "off", "disable", "disabled", "", "none"}


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Interpret truthy/falsy representations coming from configs/UI/JSON.

    Unknown values fall back to *default* (never a silent ``True``).
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        return bool(default)
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return getattr(config, key, default)
    except Exception:
        return default


def requested_flags(config: Any) -> dict[str, bool]:
    """Return the three legacy boolean requests (master tiles / SDS / ZeGrid)."""
    return {
        EXISTING_MASTER_TILES: coerce_bool(_get(config, MASTER_TILES_KEY, False), False),
        SDS: coerce_bool(_get(config, SDS_KEY, False), False),
        ZEGRID: coerce_bool(_get(config, ZEGRID_KEY, False), False),
    }


def resolve_processing_mode(config: Any) -> tuple[str, dict[str, Any]]:
    """Resolve ONE canonical processing mode from the config.

    Returns ``(mode, info)`` where ``mode`` is one of :data:`PROCESSING_MODES`
    and ``info`` records the resolution for the run log / manifest:

    * ``requested``  — the raw ``processing_mode`` token (or ``"auto"``);
    * ``flags``      — the three legacy booleans;
    * ``requested_modes`` — the special modes that were actually requested;
    * ``resolved``   — the effective mode;
    * ``source``     — ``"explicit"`` or ``"flags"`` or ``"default"``;
    * ``conflict``   — the requested special modes when more than one was set
      (empty when the state was unambiguous);
    * ``resolution`` — ``"precedence"`` when a conflict was arbitrated.
    """
    raw_mode = _get(config, MODE_KEY, None)
    raw_token = str(raw_mode).strip().lower() if raw_mode is not None else ""
    flags = requested_flags(config)
    requested_specials = [m for m in SPECIAL_MODES if flags.get(m)]

    info: dict[str, Any] = {
        "requested": raw_token or "auto",
        "flags": dict(flags),
        "requested_modes": list(requested_specials),
        "conflict": [],
        "resolution": "",
        "source": "",
        "resolved": CLASSIC,
    }

    if raw_token in PROCESSING_MODES:
        mode = raw_token
        info["source"] = "explicit"
        # An explicit mode must not silently disagree with the legacy booleans.
        inconsistent = [m for m in requested_specials if m != mode]
        if inconsistent:
            info["conflict"] = list(requested_specials)
            info["resolution"] = "explicit_mode_wins"
        info["resolved"] = mode
        return mode, info

    if len(requested_specials) > 1:
        # Incoherent legacy state: arbitrate deterministically and loudly.
        mode = next(m for m in CONFLICT_PRECEDENCE if m in requested_specials)
        info["conflict"] = list(requested_specials)
        info["resolution"] = "precedence"
        info["source"] = "flags"
        info["resolved"] = mode
        return mode, info

    if len(requested_specials) == 1:
        mode = requested_specials[0]
        info["source"] = "flags"
        info["resolved"] = mode
        return mode, info

    info["source"] = "default"
    info["resolved"] = CLASSIC
    return CLASSIC, info


def apply_mode_to_flags(config: dict, mode: str) -> dict:
    """Return *config* updated so the legacy booleans match *mode* exactly.

    Used by the GUI (single logical state) and by config migration so that
    readers still relying on ``use_existing_master_tiles`` / ``sds_mode_default``
    / ``zegrid_mode_enabled`` see a coherent, mutually-exclusive state.
    """
    if mode not in PROCESSING_MODES:
        raise ValueError(f"unknown processing mode {mode!r}")
    updated = dict(config)
    updated[MODE_KEY] = mode
    updated[MASTER_TILES_KEY] = mode == EXISTING_MASTER_TILES
    updated[SDS_KEY] = mode == SDS
    updated[ZEGRID_KEY] = mode == ZEGRID
    return updated


def describe_mode(mode: str) -> str:
    """Short human-readable description (GUI tooltips / run log)."""
    return {
        CLASSIC: "Classic raw frames",
        EXISTING_MASTER_TILES: "Existing master tiles",
        SDS: "ZeSupaDupStack (SDS) raw frames",
        ZEGRID: "ZeGrid raw frames",
    }.get(mode, mode)
