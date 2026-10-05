"""ZM-ZEGRID-R1 — isolated local MiniTile implementation (one real Cell).

Modules:
* ``geometry``   — manifest, canvas, layout, cell, patch, source ROI plans.
* ``execution``  — local section reads, local WCS reprojection, support map.
* ``science_adapter`` — build ONE CanonicalStackRequest, run canonical stack.
* ``assembly``   — core-slice extraction (no canvas/blend/DBE/equalize).
* ``adequacy``   — explicit witness-adequacy fields (closes Nono M1).
* ``sweep``      — deterministic multi-frame candidate sweep + memory guard.
* ``seam``       — adjacent-Cell continuity diagnostics (diagnostic, no blend).

Not wired into production dispatch.
"""

from __future__ import annotations

from . import adequacy, assembly, execution, geometry, science_adapter, seam, sweep

__all__ = [
    "geometry",
    "execution",
    "science_adapter",
    "assembly",
    "adequacy",
    "sweep",
    "seam",
]
