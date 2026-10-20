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
    "ZeGridHaloError",
    "resolve_zegrid_halo_px",
]

ZEGRID_HALO_ALLOWED_PX: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
ZEGRID_HALO_DEFAULT_PX: int = 64
ZEGRID_HALO_CONFIG_KEY: str = "zegrid_halo_px"


class ZeGridHaloError(ValueError):
    """Raised when a requested ZeGrid halo is not an allowed product value."""


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
