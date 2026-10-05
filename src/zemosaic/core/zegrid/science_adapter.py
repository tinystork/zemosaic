"""ZM-ZEGRID-R1 — science adapter: build ONE CanonicalStackRequest + run it.

Thin adapter over SCI-05. No copied/reimplemented science. Imports
``run_canonical_stack`` and the frozen ``CanonicalStackRequest`` /
``CanonicalStackResult`` dataclasses.

Determinism contract:
* All patch contributors are passed in STABLE sorted FrameId order.
* ``reference_index=None`` so the engine auto-selects the greatest valid-support
  frame; the stable ordering makes the index tie deterministic.
* Explicit normalization / weighting / rejection / combine tokens, backend
  ``"cpu"``, taper ``"footprint"`` with ``taper_px=8``, ``taper_floor=0``.
* ``equalize_rgb=False`` (isolate spatial stacking; no per-patch equalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    CanonicalStackResult,
    run_canonical_stack,
)

# Frozen default science configuration for R1. Explicit, not silent defaults.
DEFAULT_NORMALIZATION = "linear_fit"
DEFAULT_WEIGHTING = "noise_variance"
DEFAULT_REJECTION = "kappa_sigma"
DEFAULT_COMBINE = "mean"
DEFAULT_BACKEND = "cpu"
DEFAULT_TAPER = "footprint"
DEFAULT_TAPER_PX = 8.0
DEFAULT_TAPER_FLOOR = 0.0


@dataclass(frozen=True)
class MiniTileScienceConfig:
    """Explicit canonical tokens for one MiniTile stack."""

    normalization: str = DEFAULT_NORMALIZATION
    weighting: str = DEFAULT_WEIGHTING
    rejection: str = DEFAULT_REJECTION
    combine: str = DEFAULT_COMBINE
    backend: str = DEFAULT_BACKEND
    taper: str = DEFAULT_TAPER
    taper_px: float = DEFAULT_TAPER_PX
    taper_floor: float = DEFAULT_TAPER_FLOOR
    reference_index: int | None = None
    equalize_rgb: bool = False


def build_request(
    images: Sequence[object],
    geometric_support: Sequence[object],
    config: MiniTileScienceConfig,
) -> CanonicalStackRequest:
    """Build a canonical request from aligned arrays + geometric support.

    ``images`` and ``geometric_support`` must already be in STABLE sorted
    FrameId order (the caller's responsibility; this adapter does not reorder).
    """
    return CanonicalStackRequest(
        images=list(images),
        geometric_support=list(geometric_support),
        normalization=config.normalization,
        weighting=config.weighting,
        rejection=config.rejection,
        combine=config.combine,
        reference_index=config.reference_index,
        taper=config.taper,
        taper_px=config.taper_px,
        taper_floor=config.taper_floor,
        backend=config.backend,
        equalize_rgb=config.equalize_rgb,
    )


@dataclass(frozen=True)
class MiniTileScienceResult:
    """Canonical result + FrameId-resolved reference/exclusions provenance."""

    result: CanonicalStackResult
    reference_frame_id: str | None
    excluded: tuple[tuple[str, str, str], ...]  # (frame_id, stage, reason)
    frame_order: tuple[str, ...]


def run_minitile_stack(
    images: Sequence[object],
    geometric_support: Sequence[object],
    frame_order: Sequence[str],
    config: MiniTileScienceConfig,
) -> MiniTileScienceResult:
    """Run one canonical stack over all patch contributors in stable order.

    Resolves the engine's integer reference index and exclusion indices back to
    stable FrameIds, so the result carries identity provenance (not just index).
    """
    request = build_request(images, geometric_support, config)
    result = run_canonical_stack(request)

    order = list(frame_order)
    ref_idx = int(result.provenance["reference"]["index"])
    reference_frame_id = order[ref_idx] if 0 <= ref_idx < len(order) else None

    excluded: list[tuple[str, str, str]] = []
    for idx, stage, reason in result.provenance["excluded_frames"]:
        fid = order[idx] if 0 <= idx < len(order) else f"<index {idx}>"
        excluded.append((fid, stage, reason))

    return MiniTileScienceResult(
        result=result,
        reference_frame_id=reference_frame_id,
        excluded=tuple(excluded),
        frame_order=tuple(order),
    )
