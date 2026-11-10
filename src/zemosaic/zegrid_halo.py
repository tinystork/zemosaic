"""ZM-ZEGRID-R53 — product contract for the anti-checkerboard halo width.

The ZeGrid transition halo (``zegrid_halo_px``) is the ONLY user-selectable
geometric width in this feature. It is expressed in OUTPUT CANVAS pixels and is
used for the real per-Cell patch extension, the contributor inventory, the
per-Cell gauge domain, the offset measurement band (``2 * H``) and the feather
ramp width (``H``).

This module is intentionally tiny and dependency-free so config defaults, the
Qt fallback defaults, the worker transport and the engine all import the SAME
contract (no duplicate literals, no mutable global).

Contract
--------
* Allowed product values: ``{8, 16, 32, 64, 128, 256}``.
* Product default: ``64``.
* ``None`` (key absent / no explicit argument) resolves to the default.
* ``bool`` is REJECTED (``True``/``False`` are not integers here).
* ``float`` is REJECTED, including integral values (``64.0``): no silent
  truncation.
* Numeric strings are accepted ONLY as an ASCII decimal integer with optional
  surrounding whitespace (``"64"`` / ``" 64 "``), for legacy config migration.
  ``"64.0"``, ``"6e1"``, ``"+64"`` and ``""`` are rejected.
* Any other type or any value outside the allowed set fails loudly.

The resolved value is a plain ``int``. Callers MUST pass it explicitly down the
production path; no module constant is mutated at runtime.
"""

from __future__ import annotations

import numbers

__all__ = [
    "ZEGRID_HALO_ALLOWED_PX",
    "ZEGRID_HALO_CONFIG_KEY",
    "ZEGRID_HALO_DEFAULT_PX",
    "HALO_OVERHEAD_POLICY_ADVISORY",
    "HALO_OVERHEAD_POLICY_HARD_FLOOR",
    "ZEGRID_HALO_SOURCE_CONFIG",
    "ZEGRID_HALO_SOURCE_EXPLICIT",
    "ZEGRID_HALO_SOURCE_INTERNAL_ONLY",
    "ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT",
    "ZEGRID_HALO_UNITS",
    "ZeGridHaloError",
    "build_halo_provenance",
    "coerce_halo_user_selected",
    "halo_overhead_policy_for_source",
    "normalize_halo_overhead_policy",
    "resolve_halo_user_selected",
    "resolve_zegrid_halo_px",
    "resolve_run_halo_provenance",
]

ZEGRID_HALO_ALLOWED_PX: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
ZEGRID_HALO_DEFAULT_PX: int = 64
ZEGRID_HALO_CONFIG_KEY: str = "zegrid_halo_px"
ZEGRID_HALO_UNITS: str = "output_canvas_px"

# Stable machine tokens for provenance (never localized, never re-worded).
ZEGRID_HALO_SOURCE_EXPLICIT: str = "explicit_argument"
ZEGRID_HALO_SOURCE_CONFIG: str = "config"
ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT: str = "product_default"
# Only for direct/internal ``_run_single`` calls that predate the product
# provenance object; never emitted by the product entry point.
ZEGRID_HALO_SOURCE_INTERNAL_ONLY: str = "internal_default"

_HALO_SOURCES = frozenset(
    {
        ZEGRID_HALO_SOURCE_EXPLICIT,
        ZEGRID_HALO_SOURCE_CONFIG,
        ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT,
        ZEGRID_HALO_SOURCE_INTERNAL_ONLY,
    }
)


class ZeGridHaloError(ValueError):
    """Raised when a requested ZeGrid halo is not an allowed product value."""


# ---------------------------------------------------------------------------
# ZM-HALO-EXECUTION-DECOUPLING — halo-overhead POLICY (not a geometry width).
#
# ``max_halo_overhead`` is an EFFICIENCY ceiling (halo area vs core area). It is
# NOT a scientific feasibility rule: the halo width is part of the science
# contract (membership, feather, gauge/offset band) and the planner can never
# change it. Two explicit policies decide what an exceedance MEANS:
#
# * ``hard_floor`` — the historical behaviour: an exceedance rejects the
#   candidate layout. Kept for product-default / internal-only halos and for
#   every direct/unit caller that supplies no policy, so low-level behaviour is
#   byte-for-byte unchanged.
# * ``advisory_explicit_halo`` — a USER-REQUESTED halo is honoured exactly. The
#   ratio is still computed, logged and persisted, but only as a diagnostic/cost
#   signal; it never forces a coarser logical Cell and never rejects a layout on
#   its own. Memory admission, the patch-area floor, the contributor floor and
#   the joint PLAN/ACTIVE/RUN envelope remain fail-closed.
HALO_OVERHEAD_POLICY_HARD_FLOOR: str = "hard_floor"
HALO_OVERHEAD_POLICY_ADVISORY: str = "advisory_explicit_halo"

_HALO_OVERHEAD_POLICIES = frozenset(
    {HALO_OVERHEAD_POLICY_HARD_FLOOR, HALO_OVERHEAD_POLICY_ADVISORY}
)


def normalize_halo_overhead_policy(value=None) -> str:
    """Resolve an optional policy token; ``None`` is the historical hard floor."""
    if value is None:
        return HALO_OVERHEAD_POLICY_HARD_FLOOR
    token = str(value)
    if token not in _HALO_OVERHEAD_POLICIES:
        raise ZeGridHaloError(
            f"unknown halo-overhead policy {value!r}; expected one of "
            f"{sorted(_HALO_OVERHEAD_POLICIES)}"
        )
    return token


def halo_overhead_policy_for_source(source: str) -> str:
    """Map a halo PROVENANCE source to the overhead policy.

    A user-selected halo (explicit keyword or config value) is honoured: the
    overhead ratio is advisory only. The product default and the internal
    default keep the historical hard floor, so a run that never requested a
    halo width behaves exactly as before.
    """
    if source == ZEGRID_HALO_SOURCE_EXPLICIT:
        return HALO_OVERHEAD_POLICY_ADVISORY
    if source == ZEGRID_HALO_SOURCE_CONFIG:
        return HALO_OVERHEAD_POLICY_ADVISORY
    if source in (
        ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT,
        ZEGRID_HALO_SOURCE_INTERNAL_ONLY,
    ):
        return HALO_OVERHEAD_POLICY_HARD_FLOOR
    raise ZeGridHaloError(
        f"unknown halo provenance source {source!r}; cannot decide the "
        "halo-overhead policy"
    )


def resolve_zegrid_halo_px(value=None) -> int:
    """Resolve a requested halo width to the validated product value.

    ``value is None`` means "not requested" and yields the product default.
    Everything else is validated strictly (see the module contract); an invalid
    request never falls back silently.
    """
    if value is None:
        return int(ZEGRID_HALO_DEFAULT_PX)

    if isinstance(value, bool):
        raise ZeGridHaloError(
            "zegrid_halo_px must be an integer pixel width, got bool "
            f"{value!r}; allowed values are {ZEGRID_HALO_ALLOWED_PX}"
        )
    if isinstance(value, float):
        raise ZeGridHaloError(
            "zegrid_halo_px must be an integer pixel width, got float "
            f"{value!r}; refusing silent truncation (allowed values are "
            f"{ZEGRID_HALO_ALLOWED_PX})"
        )
    if isinstance(value, str):
        text = value.strip()
        if not text or not text.isascii() or not text.isdigit():
            raise ZeGridHaloError(
                "zegrid_halo_px string must be an ASCII decimal integer "
                f"(for config migration), got {value!r}"
            )
        resolved = int(text, 10)
    elif isinstance(value, numbers.Integral):
        resolved = int(value)
    else:
        raise ZeGridHaloError(
            "zegrid_halo_px must be an integer pixel width, got "
            f"{type(value).__name__} {value!r}; allowed values are "
            f"{ZEGRID_HALO_ALLOWED_PX}"
        )

    if resolved not in ZEGRID_HALO_ALLOWED_PX:
        raise ZeGridHaloError(
            f"zegrid_halo_px={resolved} is not an allowed product value; "
            f"allowed values are {ZEGRID_HALO_ALLOWED_PX}"
        )
    return resolved


def build_halo_provenance(
    requested_px: int | None,
    effective_px: int,
    source: str,
) -> dict:
    """Build the compact, immutable-by-convention provenance object.

    ``requested_px`` is ``None`` ONLY when the product default was used; when a
    caller/config supplied a value it is the NORMALIZED allowed integer (a legacy
    numeric string never leaks into the manifest). ``effective_px`` is the
    validated integer that drives the real geometry.
    """
    effective = resolve_zegrid_halo_px(effective_px)
    if requested_px is not None:
        requested = resolve_zegrid_halo_px(requested_px)
    else:
        requested = None
    if source not in _HALO_SOURCES:
        raise ZeGridHaloError(
            f"unknown halo provenance source {source!r}; expected one of "
            f"{sorted(_HALO_SOURCES)}"
        )
    if source == ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT and requested is not None:
        raise ZeGridHaloError(
            "product_default provenance must carry requested_px=None (the product "
            "default was not a user request)"
        )
    return {
        "requested_px": requested,
        "effective_px": int(effective),
        "source": str(source),
        "units": ZEGRID_HALO_UNITS,
    }


def coerce_halo_user_selected(value):
    """Normalize a persisted/transported user-selection flag.

    Returns ``True``/``False`` for a usable boolean and ``None`` when the value
    carries no usable metadata (absent, invalid or out of contract). ``None`` is
    NEVER treated as a claim of user intent: the caller migrates by value.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        ivalue = int(value)
        if ivalue in (0, 1):
            return bool(ivalue)
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    return None


def resolve_halo_user_selected(raw_intent, halo_value, *, intent_persisted):
    """Resolve the persisted GUI/CLI user-selection signal (migration seam).

    * usable metadata wins, EXCEPT a ``False`` contradicting a non-default width:
      a non-default width is user intent by the migration contract, so the flag
      is normalized UP (the value proves the selection);
    * no usable metadata: a non-default width migrates to ``True`` (legacy user
      intent), while the default 64 migrates to ``False`` (untouched default).

    ``intent_persisted`` documents whether the raw key existed at all; the
    decision itself never invents intent beyond the persisted width.
    """
    intent = coerce_halo_user_selected(raw_intent)
    try:
        value = resolve_zegrid_halo_px(halo_value)
    except ZeGridHaloError:
        # An invalid width blocks the start elsewhere; never claim intent from it.
        return False if intent is None else bool(intent)
    if intent is not None:
        if intent is False and value != int(ZEGRID_HALO_DEFAULT_PX):
            return True
        return bool(intent)
    return bool(value != int(ZEGRID_HALO_DEFAULT_PX))


def resolve_run_halo_provenance(
    explicit_value, config_value, *, explicit_intent=None, config_intent=None
) -> dict:
    """Resolve the ONE run-level provenance from explicit > config > default.

    ``explicit_value is None`` means "not supplied" and ``config_value`` is the
    raw value read ONCE from the effective config object by the caller (never
    re-read later). No mutable global is written.

    ZM-HALO-EXECUTION-DECOUPLING REWORK-1: ``explicit_intent`` /
    ``config_intent`` carry the PERSISTED user-selection signal from the GUI
    (``zegrid_halo_px_user_selected``). The value alone cannot distinguish the
    untouched product default 64 from a user who explicitly selected 64, so
    intent is an independent, additive input:

    * intent ``True``  -> the value is a USER REQUEST (``explicit_argument`` /
      ``config``), whatever the width, including 64;
    * intent ``False`` -> NOT a user request. Only the product default is
      truthful there; a non-default width is a contradictory transport and
      fails closed (callers normalize persisted contradictions separately);
    * intent ``None``  -> legacy/direct call. An explicit keyword keeps the
      historical "explicit argument" semantics (advisory); a legacy config
      value is advisory when it differs from the default and resolves to the
      product default when it equals it.
    """
    _default = int(ZEGRID_HALO_DEFAULT_PX)
    _explicit_intent = coerce_halo_user_selected(explicit_intent)
    _config_intent = coerce_halo_user_selected(config_intent)

    if explicit_value is not None:
        effective = resolve_zegrid_halo_px(explicit_value)
        if _explicit_intent is False:
            if effective != _default:
                raise ZeGridHaloError(
                    "contradictory halo intent: zegrid_halo_px_user_selected="
                    f"False with non-default halo {effective} px; refusing to "
                    "claim or silently discard user intent"
                )
            return build_halo_provenance(
                None, effective, ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT
            )
        # True (user request) and None (legacy/direct explicit keyword) are both
        # explicit-argument provenance; only the overhead policy differs later.
        return build_halo_provenance(
            effective, effective, ZEGRID_HALO_SOURCE_EXPLICIT
        )

    if config_value is not None:
        effective = resolve_zegrid_halo_px(config_value)
        if _config_intent is False and effective != _default:
            # Persisted metadata contradicted a non-default value. Non-default
            # widths are user intent by the migration contract, so honour the
            # value as config provenance instead of refusing the run.
            return build_halo_provenance(
                effective, effective, ZEGRID_HALO_SOURCE_CONFIG
            )
        if _config_intent is False:
            return build_halo_provenance(
                None, effective, ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT
            )
        if _config_intent is None and effective == _default:
            # Legacy/default 64 with NO intent metadata = untouched product
            # default (historical hard floor).
            return build_halo_provenance(
                None, effective, ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT
            )
        return build_halo_provenance(
            effective, effective, ZEGRID_HALO_SOURCE_CONFIG
        )

    effective = resolve_zegrid_halo_px(None)
    return build_halo_provenance(
        None, effective, ZEGRID_HALO_SOURCE_PRODUCT_DEFAULT
    )
