"""ZM-ZEGRID-R1 — isolated local MiniTile implementation (one real Cell).

Modules:
* ``geometry``   — manifest, canvas, layout, cell, patch, source ROI plans.
* ``execution``  — local section reads, local WCS reprojection, support map.
* ``science_adapter`` — build ONE CanonicalStackRequest, run canonical stack.
* ``assembly``   — core-slice extraction (no canvas/blend/DBE/equalize).
* ``adequacy``   — explicit witness-adequacy fields (closes Nono M1).
* ``sweep``      — deterministic multi-frame candidate sweep + memory guard.
* ``seam``       — adjacent-Cell continuity diagnostics (diagnostic, no blend).
* ``executor``   — R3 full-layout deterministic executor (row-major, sky_mean).
* ``mosaic``     — R3 deterministic canvas assembly (one-owner, no blend).

Not wired into production dispatch.
"""

from __future__ import annotations

from . import (
    adequacy,
    assembly,
    auto_layout,
    execution,
    executor,
    final_mosaic_finishing,
    geometry,
    mosaic,
    observability,
    photometric,
    science_adapter,
    seam,
    sweep,
)

__all__ = [
    "geometry",
    "execution",
    "science_adapter",
    "assembly",
    "adequacy",
    "sweep",
    "seam",
    "executor",
    "mosaic",
    "auto_layout",
    "photometric",
    "observability",
    "final_mosaic_finishing",
]
