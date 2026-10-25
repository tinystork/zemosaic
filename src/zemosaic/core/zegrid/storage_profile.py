"""ZM-ZEGRID-R52-OOC — explicit storage profile (Lot B).

ONE explicit, immutable choice of how a ZeGrid run stores the aligned input and
the rejection diagnostics:

* ``dense`` (default) — the historical behaviour: ``MemmapCanonicalProvider``
  keeps 2N mappings open for the life of a Cell and the streaming executor
  allocates the dense ``(N, H, W, C)`` bool rejection plane. Every existing
  caller, the light M106 path and the in-memory path keep this profile.
* ``bounded`` — the Lot A mechanisms: ``BoundedMemmapCanonicalProvider`` opens,
  copies and closes per access, and a fresh ``BoundedCountingRejectionSink``
  replaces the dense rejection plane for the ZeGrid caller (which consumes the
  scalar rejection diagnostics, not the full mask).

The profile is selected BEFORE the first layout filter and is carried, unchanged,
through PLAN / ADMIT / RUN / fallback. It is explicitly NOT a GUI/config knob in
this lot: the product default is ``dense`` and ``bounded`` is reachable through
the internal ``run_zegrid_mode(storage_profile=...)`` seam (tests/diagnosis).
"""

from __future__ import annotations

__all__ = [
    "STORAGE_PROFILE_DENSE",
    "STORAGE_PROFILE_BOUNDED",
    "STORAGE_PROFILES",
    "STORAGE_MODEL_DENSE",
    "STORAGE_MODEL_BOUNDED",
    "normalize_storage_profile",
    "storage_model_id",
    "is_bounded",
]

STORAGE_PROFILE_DENSE = "dense"
STORAGE_PROFILE_BOUNDED = "bounded"
STORAGE_PROFILES = (STORAGE_PROFILE_DENSE, STORAGE_PROFILE_BOUNDED)

STORAGE_MODEL_DENSE = "dense-streaming-v1"
STORAGE_MODEL_BOUNDED = "bounded-streaming-v1"

_ALIASES = {
    "dense": STORAGE_PROFILE_DENSE,
    "bounded": STORAGE_PROFILE_BOUNDED,
    "ooc": STORAGE_PROFILE_BOUNDED,
    "out-of-core": STORAGE_PROFILE_BOUNDED,
}


def normalize_storage_profile(value, *, default: str = STORAGE_PROFILE_DENSE) -> str:
    """Normalize a storage profile token; reject anything unknown.

    ``None`` -> ``default``. Aliases are deliberately few and explicit; an
    unknown token raises (never a silent fallback that would erase the admitted
    model's identity).
    """
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(
            f"storage_profile must be a string, got {type(value).__name__}"
        )
    token = value.strip().lower()
    if token in _ALIASES:
        return _ALIASES[token]
    raise ValueError(
        f"unknown storage_profile {value!r}; expected one of {STORAGE_PROFILES}"
    )


def storage_model_id(profile: str) -> str:
    """Stable model identifier recorded in plans/manifests."""
    token = normalize_storage_profile(profile)
    return (
        STORAGE_MODEL_BOUNDED
        if token == STORAGE_PROFILE_BOUNDED
        else STORAGE_MODEL_DENSE
    )


def is_bounded(profile: str) -> bool:
    return normalize_storage_profile(profile) == STORAGE_PROFILE_BOUNDED
