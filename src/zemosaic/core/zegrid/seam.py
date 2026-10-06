"""ZM-ZEGRID-R2 — adjacent-Cell continuity diagnostics (diagnostic only, NO blend).

Objective C: for the chosen multi-frame Cell, build ONE adjacent neighbour Cell
patch and verify, without implementing any blending/feathering/photometric
harmonization:

1. **Core ownership** — the two Cell cores are DISJOINT and together (with the
   rest of the layout) OWN every canvas pixel exactly once (no gap, no overlap).
2. **No halo double-counting** — the MiniTile is the FULL ProcessingPatch result;
   only the ``core_slice`` crop is placed into final science, so halo
   contributions are never double-counted into any final/core science.
3. **Seam residual** — quantify a diagnostic along the shared boundary: compare
   support/coverage (valid fraction / n_eff / surviving counts and, where
   defined, science values) on the two sides of the boundary inside the
   overlapping halo region.

The seam residual is a **measurement**, not a correction. It quantifies whether
the patch-local statistical domain (per-patch reference / normalization /
rejection) introduces a measurable discontinuity across the boundary. It does
NOT modify science, blend, feather, or harmonize anything.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import GlobalCanvas, ZeGridLayout, build_layout

# Seam context half-width in global pixels (== halo). The two sides of the
# boundary are each ``SEAM_HALF`` pixels wide inside the overlapping halo.
SEAM_HALF = 8


def assert_core_partition_exact(layout: ZeGridLayout, canvas: GlobalCanvas) -> np.ndarray:
    """Return the ownership raster and assert each pixel is owned exactly once.

    Raises ``AssertionError`` if any pixel is owned 0 (gap) or >1 (overlap)
    times. The returned ``(H, W)`` int raster equals 1 everywhere.
    """
    seen = np.zeros((canvas.height, canvas.width), dtype=np.int64)
    for _row, _col, bounds in layout.iter_cells(canvas):
        seen[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] += 1
    if not np.all(seen == 1):
        gap = int(np.count_nonzero(seen == 0))
        overlap = int(np.count_nonzero(seen > 1))
        raise AssertionError(
            f"core partition not exact: {gap} gap pixels, {overlap} overlap pixels"
        )
    return seen


def adjacent_neighbours(row: int, col: int, nx: int, ny: int) -> list[tuple[str, int, int]]:
    """Return valid adjacent cells as ``(direction, row, col)``.

    Directions in priority order: right, bottom, left, top. A cell is adjacent if
    it shares a core boundary (within the layout bounds).
    """
    out: list[tuple[str, int, int]] = []
    if col + 1 < nx:
        out.append(("right", row, col + 1))
    if row + 1 < ny:
        out.append(("bottom", row + 1, col))
    if col - 1 >= 0:
        out.append(("left", row, col - 1))
    if row - 1 >= 0:
        out.append(("top", row - 1, col))
    return out


def _orient(a, b):
    """Return ``(axis, left/top_cell, right/bottom_cell)`` for two adjacent cells."""
    ac, bc = a.core, b.core
    if ac.y0 == bc.y0 and ac.y1 == bc.y1:
        if ac.x0 < bc.x0:
            return "x", a, b
        return "x", b, a
    if ac.x0 == bc.x0 and ac.x1 == bc.x1:
        if ac.y0 < bc.y0:
            return "y", a, b
        return "y", b, a
    raise ValueError(f"cells {a.cell_id}/{b.cell_id} are not axis-aligned adjacent")


def _strip_metrics(science, valid_mask, n_eff, surviving, gy0, gy1, gx0, gx1) -> dict:
    """Summarize coverage/science over a patch-local ``(y,x)`` rectangle.

    ``science``/``valid_mask``/``surviving`` are ``(H, W, C)``; ``n_eff`` is
    ``(H, W)``. All slices are already in the owning MiniTile's patch-local
    coordinates.
    """
    s = science[gy0:gy1, gx0:gx1]
    v = valid_mask[gy0:gy1, gx0:gx1]
    ne = n_eff[gy0:gy1, gx0:gx1]
    sv = surviving[gy0:gy1, gx0:gx1]
    valid_fraction = float(np.mean(v)) if v.size else 0.0
    ne_vals = ne[np.isfinite(ne)]
    mean_n_eff = float(np.mean(ne_vals)) if ne_vals.size else float("nan")
    mean_surviving = float(np.mean(sv)) if sv.size else float("nan")
    science_mean = []
    nch = s.shape[-1] if s.ndim == 3 else 1
    for ch in range(nch):
        if s.ndim == 3:
            vals = s[..., ch][v[..., ch] & np.isfinite(s[..., ch])]
        else:
            vals = s[v & np.isfinite(s)]
        science_mean.append(float(np.mean(vals)) if vals.size else float("nan"))
    return {
        "valid_fraction": valid_fraction,
        "mean_n_eff": mean_n_eff,
        "mean_surviving": mean_surviving,
        "mean_science_per_channel": science_mean,
    }


def compute_seam_diagnostic(
    canvas: GlobalCanvas,
    cell_a,
    patch_a,
    mt_a,
    cell_b,
    patch_b,
    mt_b,
    nx: int = 5,
    ny: int = 4,
    halo_px: int = SEAM_HALF,
) -> dict:
    """Compute the seam diagnostic between two adjacent MiniTiles (no blend).

    ``cell_a``/``cell_b`` are ``ZeGridCell``, ``patch_a``/``patch_b`` are
    ``ProcessingPatch``, and ``mt_a``/``mt_b`` are ``MiniTile`` (full patch
    planes). Cell order is normalized internally (left/top first). Returns a
    JSON-serializable dict with the boundary, per-side coverage/science
    summaries, and the absolute residual. Purely diagnostic.
    """
    axis, lo_cell, hi_cell = _orient(cell_a, cell_b)
    # Map the normalized cell order back to (cell, patch, mt) triples.
    if lo_cell.cell_id == cell_a.cell_id:
        lo_patch, lo_mt = patch_a, mt_a
        hi_patch, hi_mt = patch_b, mt_b
    else:
        lo_patch, lo_mt = patch_b, mt_b
        hi_patch, hi_mt = patch_a, mt_a

    if axis == "x":
        bnd = lo_cell.core.x1  # == hi_cell.core.x0
        gy0, gy1 = lo_cell.core.y0, lo_cell.core.y1
        # low side = lo cell's last ``halo`` core columns; high side = hi's first.
        lo_gx0, lo_gx1 = bnd - halo_px, bnd
        hi_gx0, hi_gx1 = bnd, bnd + halo_px
        lo_slice = (gy0, gy1, lo_gx0, lo_gx1)
        hi_slice = (gy0, gy1, hi_gx0, hi_gx1)
    else:
        bnd = lo_cell.core.y1  # == hi_cell.core.y0
        gx0, gx1 = lo_cell.core.x0, lo_cell.core.x1
        lo_gy0, lo_gy1 = bnd - halo_px, bnd
        hi_gy0, hi_gy1 = bnd, bnd + halo_px
        lo_slice = (lo_gy0, lo_gy1, gx0, gx1)
        hi_slice = (hi_gy0, hi_gy1, gx0, gx1)

    def _side(mt, patch, gy0_, gy1_, gx0_, gx1_):
        py0 = gy0_ - patch.patch.y0
        py1 = gy1_ - patch.patch.y0
        px0 = gx0_ - patch.patch.x0
        px1 = gx1_ - patch.patch.x0
        return _strip_metrics(
            mt.science, mt.valid_mask, mt.n_eff_support,
            mt.surviving_sample_count, py0, py1, px0, px1,
        )

    side_lo = _side(lo_mt, lo_patch, *lo_slice)
    side_hi = _side(hi_mt, hi_patch, *hi_slice)

    def _abs(a, b):
        if isinstance(a, list):
            return [abs(x - y) for x, y in zip(a, b)]
        return abs(a - b)

    residual = {
        "valid_fraction_abs_diff": _abs(
            side_lo["valid_fraction"], side_hi["valid_fraction"]
        ),
        "mean_n_eff_abs_diff": _abs(side_lo["mean_n_eff"], side_hi["mean_n_eff"]),
        "mean_surviving_abs_diff": _abs(
            side_lo["mean_surviving"], side_hi["mean_surviving"]
        ),
        "mean_science_abs_diff_per_channel": _abs(
            side_lo["mean_science_per_channel"], side_hi["mean_science_per_channel"]
        ),
    }

    science_diff = residual["mean_science_abs_diff_per_channel"]
    measurable_science = any(np.isfinite(d) and d > 1e-6 for d in science_diff)
    measurable_coverage = residual["valid_fraction_abs_diff"] > 1e-9 or (
        np.isfinite(residual["mean_n_eff_abs_diff"])
        and residual["mean_n_eff_abs_diff"] > 1e-6
    )
    discontinuity = bool(measurable_science or measurable_coverage)

    return {
        "axis": axis,
        "boundary_global": int(bnd),
        "halo_px": int(halo_px),
        "low_cell": lo_cell.cell_id,
        "high_cell": hi_cell.cell_id,
        "side_low": side_lo,
        "side_high": side_hi,
        "residual": residual,
        "measurable_discontinuity": discontinuity,
    }
