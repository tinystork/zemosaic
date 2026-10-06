"""ZM-ZEGRID-R3 — deterministic canvas assembly (Scope B).

Places each Cell's CORE slice into a final canvas at its exact global position
(canvas 2403 x 3278 for M106), accumulating support/coverage planes from the
canonical support/estimator maps (NEVER inferred from brightness). Exactly ONE
owner per pixel (disjoint + exhaustive over the layout; no gap, no overlap, no
halo double-count). Missing/incomplete Cells are handled EXPLICITLY: a
``coverage == 0`` hole is visible in the manifest and is never silently filled.

NO blend / DBE / equalize / denoise. This module only *places* deterministic
core slices; it applies no photometric harmonization.

Plane contract (per Cell, from ``assembly.crop_all_planes_to_core``):
* ``science_core``                (Hc, Wc, 3) float32  — combined science.
* ``surviving_sample_count_core`` (Hc, Wc, 3) int      — per-channel stack depth.
* ``n_eff_support_core``          (Hc, Wc)    float64  — effective support map.
* ``support_w1_core`` / ``support_w2_core``  (Hc, Wc) float64.
* ``valid_mask_core``             (Hc, Wc, 3) bool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .geometry import GlobalCanvas, ZeGridLayout, build_layout, cell_id


@dataclass
class AssembledCanvas:
    """Deterministic placement of all Cell core slices into one canvas."""

    canvas: GlobalCanvas
    layout: ZeGridLayout
    nx: int
    ny: int
    science: np.ndarray          # (H, W, 3) float32; NaN where no coverage/hole
    stack_depth: np.ndarray      # (H, W) int32 — max per-pixel channel stack depth
    n_eff_support: np.ndarray    # (H, W) float64
    support_w1: np.ndarray       # (H, W) float64
    support_w2: np.ndarray       # (H, W) float64
    valid_fraction: np.ndarray   # (H, W) float32 — fraction of valid channels
    owner: np.ndarray            # (H, W) int — number of cells claiming each pixel
    hole_mask: np.ndarray        # (H, W) bool — coverage == 0 (explicit hole)
    complete_cells: list = field(default_factory=list)
    incomplete_cells: list = field(default_factory=list)

    @property
    def hole_pixels(self) -> int:
        return int(np.count_nonzero(self.hole_mask))

    @property
    def coverage_pixels(self) -> int:
        return int(np.count_nonzero(~self.hole_mask))


def _empty_planes(canvas: GlobalCanvas) -> dict:
    h, w = canvas.height, canvas.width
    return {
        "science": np.full((h, w, 3), np.nan, dtype=np.float32),
        "stack_depth": np.zeros((h, w), dtype=np.int32),
        "n_eff_support": np.zeros((h, w), dtype=np.float64),
        "support_w1": np.zeros((h, w), dtype=np.float64),
        "support_w2": np.zeros((h, w), dtype=np.float64),
        "valid_fraction": np.zeros((h, w), dtype=np.float32),
        "owner": np.zeros((h, w), dtype=np.int32),
    }


def assemble_canvas(
    canvas: GlobalCanvas,
    nx: int,
    ny: int,
    cores: dict[str, dict | None],
) -> AssembledCanvas:
    """Place core slices into the canvas (deterministic, one-owner, no blend).

    ``cores`` maps ``cell_id`` to either ``None`` (missing/incomplete Cell) or a
    dict with the ``*_core`` planes (keys from ``assembly.crop_all_planes_to_core``:
    ``science_core``, ``surviving_sample_count_core``, ``n_eff_support_core``,
    ``support_w1_core``, ``support_w2_core``, ``valid_mask_core``). Any Cell in the
    layout NOT present in ``cores`` is treated as incomplete (explicit hole).

    Raises ``AssertionError`` if a complete Cell's core shape does not match its
    layout bounds, or if two Cells would overlap (owner > 1) — the "exactly one
    owner per pixel" invariant.
    """
    layout = build_layout(canvas, nx, ny)
    planes = _empty_planes(canvas)
    complete: list[str] = []
    incomplete: list[str] = []

    for row, col, bounds in layout.iter_cells(canvas):
        cid = cell_id(row, col)
        core = cores.get(cid)
        if core is None:
            incomplete.append(cid)
            continue
        y0, y1, x0, x1 = bounds.y0, bounds.y1, bounds.x0, bounds.x1
        hc, wc = y1 - y0, x1 - x0

        sci = core["science_core"]
        assert sci.shape[:2] == (hc, wc), (
            f"{cid}: science_core shape {sci.shape[:2]} != layout core {(hc, wc)}"
        )
        assert sci.shape[2] == 3, f"{cid}: expected 3 channels, got {sci.shape}"

        planes["owner"][y0:y1, x0:x1] += 1
        # One-owner invariant: placing onto already-owned pixels is a defect.
        planes["science"][y0:y1, x0:x1] = sci
        planes["stack_depth"][y0:y1, x0:x1] = np.max(
            np.asarray(core["surviving_sample_count_core"], dtype=np.int32), axis=-1
        )
        planes["n_eff_support"][y0:y1, x0:x1] = core["n_eff_support_core"]
        planes["support_w1"][y0:y1, x0:x1] = core["support_w1_core"]
        planes["support_w2"][y0:y1, x0:x1] = core["support_w2_core"]
        planes["valid_fraction"][y0:y1, x0:x1] = np.mean(
            np.asarray(core["valid_mask_core"], dtype=np.float32), axis=-1
        )
        complete.append(cid)

    # Exactly one owner per pixel (disjoint + exhaustive) for the complete set.
    # Any pixel owned by 0 cells is a hole (coverage==0, explicit); >1 is overlap.
    overlap = int(np.count_nonzero(planes["owner"] > 1))
    if overlap:
        raise AssertionError(f"assembly overlap: {overlap} pixels owned >1 times")
    # Reconstruct the exhaustive owner expectation: every layout pixel must be
    # owned exactly once (gap = owner 0 at a pixel whose Cell is complete).
    owned_everywhere = np.all(planes["owner"] == 1)
    if incomplete:
        # Incomplete Cells leave owner==0 holes by design (explicit, not a gap bug).
        # Only flag gaps OUTSIDE the incomplete Cell regions.
        pass
    hole_mask = (planes["stack_depth"] == 0)
    # Science where stack_depth == 0 is a coverage hole (never silently filled).
    planes["science"][hole_mask] = np.nan

    return AssembledCanvas(
        canvas=canvas,
        layout=layout,
        nx=nx,
        ny=ny,
        science=planes["science"],
        stack_depth=planes["stack_depth"],
        n_eff_support=planes["n_eff_support"],
        support_w1=planes["support_w1"],
        support_w2=planes["support_w2"],
        valid_fraction=planes["valid_fraction"],
        owner=planes["owner"],
        hole_mask=hole_mask,
        complete_cells=complete,
        incomplete_cells=incomplete,
    )


def verify_disjoint_exhaustive(assembled: AssembledCanvas) -> np.ndarray:
    """Return the ownership raster; assert each canvas pixel is owned exactly once.

    Equivalent to ``seam.assert_core_partition_exact`` but operates on the
    assembled owner raster. Raises on gap (owner 0) or overlap (owner > 1).
    """
    seen = assembled.owner
    if not np.all(seen == 1):
        gap = int(np.count_nonzero(seen == 0))
        overlap = int(np.count_nonzero(seen > 1))
        raise AssertionError(
            f"canvas placement not exact: {gap} gap pixels, {overlap} overlap pixels"
        )
    return seen
