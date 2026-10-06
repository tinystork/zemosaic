"""ZM-ZEGRID-R4 — deterministic RAM-aware Auto layout (Levier 1).

Replaces a FIXED ``Nx x Ny`` with a deterministic, MEASURED, RAM-aware layout
policy. The layout is chosen from a RAM budget instead of a fixed cell count.

Core idea (Levier 1, from ``ZEGRID_MEMORY_NOTES.md``): the peak RSS of one Cell
scales with the patch size and the contributor count (the canonical engine
materialises ``O(N * H * W * C)`` aligned arrays + float64 taper + rejection/
normalization intermediates). Reducing the Cell size reduces the PEAK, while the
TOTAL work is roughly invariant modulo halo overhead. This is a *memory* lever,
not a *compute* lever.

Memory model (DERIVED from measured data, never invented)::

    peak_rss = baseline
             + area_coeff  * patch_area          (per-patch fixed cost, N-independent)
             + frame_coeff * N * patch_area      (per (frame x pixel) materialisation)

* ``baseline``     — process + Python/NumPy/Astropy/SciPy footprint.
* ``area_coeff``   — bytes per patch pixel (N-independent: reprojection target,
  output planes, WCS machinery, etc.).
* ``frame_coeff``  — bytes per (frame x pixel) of the materialised aligned arrays.

The single-slope ``baseline + coeff * N * area`` model fit at ~410k-px patch scale
OVER-predicts small patches (its intercept dominates at ~51k px), which makes Auto
over-refine. The two-term model above is fit on the COMBINED calibration set
(the 20 R3 M106 cells at ~410k px + the 166 tight-run complete cells at ~51k px),
is physically sensible (positive, ~347 MiB process baseline), and stays
conservative at BOTH scales with a 15% safety margin (max positive relative
residual 11.3% on both sets). ``fit_memory_model`` re-derives it from raw records.

``FITTED_MEMORY_MODEL`` holds the frozen fit + the conservative safety margin.

Determinism: same canvas + frames + budget + floors -> same layout. The search
is a fixed, sorted refinement sequence over ``target = median_projected_footprint
/ factor``; no randomness, no filesystem-order dependence.

Budget search uses the CANDIDATE'S EXACT contributor counts (not ``len(frames)``):
for each candidate (nx, ny) the actual per-cell contributor counts are computed
from geometry/membership and the true worst cell (max of
``predict_bound(N_cell, patch_area_cell)``) bounds the peak. This avoids the old
``n_upper = len(frames)`` over-estimate (the true max N varies 61..66 across
candidates) and pairs each N with its real patch area.

Scientific floors (NEVER silently degrade science):
* ``min_patch_area_px``   — a Cell's patch must be large enough for meaningful
  rejection/normalization statistics.
* ``min_contributors``    — the deepest Cell must retain enough contributors for
  a scientifically usable stack ("where relevant").
* ``max_halo_overhead``   — halo area must not dominate the core (efficiency).

If the budget cannot be met while honouring the floors, ``choose_layout`` raises
``LayoutInfeasible`` with an explicit reason (never silently picks a degraded
layout).

KNOWN LIMITATION (uniform partition): ``choose_layout`` selects a uniform
``Nx x Ny`` grid over the whole canvas. For a mosaic whose geometric sky union
covers only a fraction of the canvas (M106 ~84.65%), the outermost cells of a
fine uniform grid can fall entirely OUTSIDE the union and become genuinely EMPTY
(0 contributors). These are reported EXPLICITLY by the executor/assembly (status
``empty``, ``coverage==0`` holes, and ``ownership.exact_one_owner`` is False only
over those data-less cells) — they are never silently filled. This is a known
limitation of the uniform partition model, NOT a memory-model defect; a
non-uniform / footprint-aware partition is deferred (do not change the partition
model here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import geometry as zg
from . import sweep as zsw

# ---------------------------------------------------------------------------
# Memory model — DERIVED (measured), not invented.
# ---------------------------------------------------------------------------

# Two-term model fit by OLS on the COMBINED calibration set:
#   * the 20 ZM-ZEGRID-R3 M106 per-cell records (cell_*.json, ~410k-px patches), and
#   * the 166 complete cells of the ZM-ZEGRID-R4 tight run (~51k-px patches).
#   peak_rss_kib = baseline + area_coeff * patch_area + frame_coeff * N * patch_area
#
#   baseline    = 355,080.24 KiB  (346.76 MiB process baseline)
#   area_coeff  =   1.039765 KiB/px  =>  1064.72 bytes/px  (N-independent)
#   frame_coeff =   0.136787 KiB/(N*px) => 140.07 bytes/(frame*px)
#   R^2 = 0.9799 ; residual std = 107,002 KiB (~104 MiB)
#   max POSITIVE relative residual (dangerous: measured > predicted) = +11.34%
#   max |relative residual| = 24.61% (safe side)
# The 15% safety margin covers +11.34% with headroom.
FITTED_BASELINE_KIB = 355080.2365954309
FITTED_AREA_COEFF_KIB_PER_PX = 1.0397652126
FITTED_FRAME_COEFF_KIB_PER_N_PX = 0.1367872099
FITTED_R2 = 0.97994525
FITTED_RESIDUAL_STD_KIB = 107001.5604
FITTED_MAX_POS_RESID_FRAC = 0.11341365
FITTED_MAX_ABS_RESID_FRAC = 0.24607071

# Conservative safety margin (relative) on the mean prediction. Justified by the
# residual distribution of the COMBINED calibration set: the largest positive
# (dangerous) relative residual is +11.34%; 15% covers it with headroom for
# run-to-run allocator variance. (The R3-only single-slope fit had +12.55% max
# positive; the combined two-term fit is better calibrated and 15% still covers.)
DEFAULT_SAFETY_MARGIN_FRAC = 0.15


@dataclass(frozen=True)
class MemoryModel:
    """Peak-RSS model: ``peak = baseline + area_coeff*area + frame_coeff*N*area``.

    ``area_coeff_bytes_per_px`` is the N-independent per-patch-pixel fixed cost;
    ``frame_coeff_bytes_per_n_px`` is the per-(frame x pixel) materialisation cost.
    ``safety_margin_frac`` is a relative margin on the mean prediction forming the
    conservative bound used for budget enforcement.
    """

    baseline_bytes: float
    area_coeff_bytes_per_px: float
    frame_coeff_bytes_per_n_px: float
    safety_margin_frac: float = DEFAULT_SAFETY_MARGIN_FRAC

    def predict_peak_bytes(self, n_contributors: int, patch_area_px: int) -> float:
        """Mean predicted peak RSS in bytes."""
        return (
            self.baseline_bytes
            + self.area_coeff_bytes_per_px * patch_area_px
            + self.frame_coeff_bytes_per_n_px * n_contributors * patch_area_px
        )

    def predict_bound_bytes(self, n_contributors: int, patch_area_px: int) -> float:
        """Conservative upper bound on peak RSS (mean x (1 + margin))."""
        return self.predict_peak_bytes(n_contributors, patch_area_px) * (1.0 + self.safety_margin_frac)

    def to_dict(self) -> dict:
        return {
            "baseline_bytes": self.baseline_bytes,
            "baseline_kib": self.baseline_bytes / 1024.0,
            "area_coeff_bytes_per_px": self.area_coeff_bytes_per_px,
            "area_coeff_kib_per_px": self.area_coeff_bytes_per_px / 1024.0,
            "frame_coeff_bytes_per_n_px": self.frame_coeff_bytes_per_n_px,
            "frame_coeff_kib_per_n_px": self.frame_coeff_bytes_per_n_px / 1024.0,
            "safety_margin_frac": self.safety_margin_frac,
        }


# The frozen fitted model (DERIVED from the combined R3 + tight calibration set).
FITTED_MEMORY_MODEL = MemoryModel(
    baseline_bytes=FITTED_BASELINE_KIB * 1024.0,
    area_coeff_bytes_per_px=FITTED_AREA_COEFF_KIB_PER_PX * 1024.0,
    frame_coeff_bytes_per_n_px=FITTED_FRAME_COEFF_KIB_PER_N_PX * 1024.0,
    safety_margin_frac=DEFAULT_SAFETY_MARGIN_FRAC,
)

FITTED_MEMORY_MODEL_META = {
    "source": (
        "combined calibration set: ZM-ZEGRID-R3 M106 per-cell records (20 cells, "
        "~410k-px patches) + ZM-ZEGRID-R4 tight-run complete cells (166 cells, "
        "~51k-px patches)"
    ),
    "data_paths": [
        "/home/tristan/zegrid_r3_m106_outputs",
        "/home/tristan/zegrid_r4_m106_tight",
    ],
    "n_cells": 186,
    "canvas": "2403x3278",
    "halo_px": 8,
    "fit_method": "ordinary least squares on peak_rss_kib vs [patch_area, n_patch_contributors*patch_area]",
    "baseline_kib": FITTED_BASELINE_KIB,
    "area_coeff_kib_per_px": FITTED_AREA_COEFF_KIB_PER_PX,
    "frame_coeff_kib_per_n_px": FITTED_FRAME_COEFF_KIB_PER_N_PX,
    "r2": FITTED_R2,
    "residual_std_kib": FITTED_RESIDUAL_STD_KIB,
    "max_positive_residual_frac": FITTED_MAX_POS_RESID_FRAC,
    "max_abs_residual_frac": FITTED_MAX_ABS_RESID_FRAC,
    "safety_margin_frac": DEFAULT_SAFETY_MARGIN_FRAC,
    "note": (
        "Two-term model (baseline + area_coeff*patch_area + frame_coeff*N*patch_area) "
        "fit on the COMBINED large+small patch calibration set. This corrects the "
        "single-slope model's over-prediction at small patches (its ~848 MiB intercept "
        "dominated at ~51k px), while staying conservative at both scales. "
        "area_coeff ~1065 B/px is the N-independent per-patch fixed cost; "
        "frame_coeff ~140 B/(frame*px) is the per-frame-per-pixel materialisation. "
        "The MEMORY_NOTES rough 13-20 bytes/frame/pixel under-counts the full "
        "canonical materialisation and is superseded by this measured value."
    ),
}


def fit_memory_model(records: Sequence[dict]) -> tuple[MemoryModel, dict]:
    """Fit the two-term peak-RSS model from per-cell records.

    Each record is a dict with ``n_patch_contributors`` (int), ``patch`` (list
    ``[x0, y0, x1, y1]``) and ``peak_rss_kib`` (int). Fits::

        peak_rss_kib = baseline + area_coeff * patch_area + frame_coeff * N * patch_area

    Returns the fitted :class:`MemoryModel` (with default margin) and a stats dict
    (R^2, residual std, max positive/abs relative residual) for reporting and
    validation. Deriving the residual stats here (rather than hard-coding them)
    keeps the frozen constants honest and prevents them going stale.
    """
    areas = []
    n_areas = []
    ys = []
    for r in records:
        p = r["patch"]
        area = (p[2] - p[0]) * (p[3] - p[1])
        areas.append(area)
        n_areas.append(r["n_patch_contributors"] * area)
        ys.append(r["peak_rss_kib"])
    X = np.column_stack([np.ones(len(ys)), np.array(areas, dtype=float), np.array(n_areas, dtype=float)])
    y = np.array(ys, dtype=float)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    baseline_kib = float(coef[0])
    area_coeff_kib = float(coef[1])
    frame_coeff_kib = float(coef[2])
    pred = X @ coef
    resid = y - pred
    r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    resid_std = float(resid.std())
    rel = resid / pred
    max_pos_rel = float(rel.max()) if len(rel) else 0.0
    max_abs_rel = float(np.abs(rel).max()) if len(rel) else 0.0
    model = MemoryModel(
        baseline_bytes=baseline_kib * 1024.0,
        area_coeff_bytes_per_px=area_coeff_kib * 1024.0,
        frame_coeff_bytes_per_n_px=frame_coeff_kib * 1024.0,
    )
    stats = {
        "baseline_kib": baseline_kib,
        "area_coeff_kib_per_px": area_coeff_kib,
        "frame_coeff_kib_per_n_px": frame_coeff_kib,
        "r2": r2,
        "residual_std_kib": resid_std,
        "max_positive_residual_frac": max_pos_rel,
        "max_abs_residual_frac": max_abs_rel,
        "n_cells": len(records),
    }
    return model, stats


# ---------------------------------------------------------------------------
# Scientific floors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScientificFloors:
    """Explicit scientific floors for Auto layout (never silently degraded).

    * ``min_patch_area_px`` — minimum Cell patch area for meaningful rejection/
      normalization statistics (a Cell with too few pixels has unreliable stats).
    * ``min_contributors`` — minimum contributors for the deepest Cell ("where
      relevant"); below this a stack is scientifically unusable.
    * ``max_halo_overhead`` — maximum halo area / core area ratio (efficiency
      ceiling; finer Cells pay more halo overhead).

    Defaults are policy floors, documented and configurable (not derived from the
    memory model — they are science choices).
    """

    min_patch_area_px: int = 50_000
    min_contributors: int = 3
    max_halo_overhead: float = 0.50

    def to_dict(self) -> dict:
        return {
            "min_patch_area_px": self.min_patch_area_px,
            "min_contributors": self.min_contributors,
            "max_halo_overhead": self.max_halo_overhead,
        }


class LayoutInfeasible(RuntimeError):
    """Raised when no layout can honour the RAM budget AND the scientific floors."""

    def __init__(self, reason: str, floors_report: dict | None = None):
        super().__init__(reason)
        self.floors_report = floors_report or {}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Refinement factors (coarse -> fine). target_cell = median_footprint / factor.
# Coarsest sensible layout (factor 1.0) = Cell ~= one median projected source
# footprint. Finer factors subdivide to reduce the peak.
REFINEMENT_FACTORS = (
    1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 32.0,
    40.0, 48.0, 64.0,
)


def median_projected_footprint(
    frames: Sequence[zg.FrameDescriptor], canvas: zg.GlobalCanvas
) -> tuple[float, float]:
    """Median projected source bbox ``(width, height)`` in canvas pixels.

    Projects each frame's pixel boundary onto the canvas WCS and takes the
    median bbox width/height (same notion as R0's "median projected source
    bbox"). Deterministic: sorts frames by FrameId first.
    """
    canvas_wcs = canvas.wcs()
    widths: list[float] = []
    heights: list[float] = []
    for f in sorted(frames, key=lambda f: f.frame_id):
        poly = zg._source_polygon(f.shape_hw, f.wcs(), canvas_wcs)
        minx, miny, maxx, maxy = poly.bounds
        widths.append(maxx - minx)
        heights.append(maxy - miny)
    if not widths:
        raise ValueError("no frames for footprint estimation")
    return float(np.median(widths)), float(np.median(heights))


def _nominal_geometry(canvas: zg.GlobalCanvas, nx: int, ny: int, halo_px: int) -> dict:
    """Nominal Cell geometry for a layout (integer math only, deterministic).

    Floor checks use the smallest core (``floor(W/nx) x floor(H/ny)``) so they are
    conservative; the nominal worst patch area is ``ceil(W/nx) x ceil(H/ny)`` plus
    full halo (used only for floors, NOT for the peak bound — the peak bound uses
    exact per-cell geometry).
    """
    W, H = canvas.width, canvas.height
    max_cw = -(-W // nx)          # ceil(W/nx)
    max_ch = -(-H // ny)          # ceil(H/ny)
    min_cw = W // nx              # floor(W/nx)
    min_ch = H // ny              # floor(H/ny)
    max_patch_w = min(W, max_cw + 2 * halo_px)
    max_patch_h = min(H, max_ch + 2 * halo_px)
    min_patch_w = min(W, min_cw + 2 * halo_px)
    min_patch_h = min(H, min_ch + 2 * halo_px)
    max_core_area = max_cw * max_ch
    max_patch_area = max_patch_w * max_patch_h
    min_core_area = min_cw * min_ch
    min_patch_area = min_patch_w * min_patch_h
    halo_overhead = (max_patch_area - max_core_area) / max_core_area if max_core_area else 0.0
    return {
        "nx": nx, "ny": ny,
        "max_core_w": max_cw, "max_core_h": max_ch,
        "min_core_w": min_cw, "min_core_h": min_ch,
        "max_core_area": max_core_area,
        "max_patch_area": max_patch_area,
        "min_core_area": min_core_area,
        "min_patch_area": min_patch_area,
        "halo_overhead": halo_overhead,
    }


def _footprints(frames: Sequence[zg.FrameDescriptor], canvas: zg.GlobalCanvas):
    """Project every frame footprint once; return (frame, polygon) sorted by id."""
    canvas_wcs = canvas.wcs()
    out = []
    for f in sorted(frames, key=lambda f: f.frame_id):
        out.append((f, zg._source_polygon(f.shape_hw, f.wcs(), canvas_wcs)))
    return out


def _cell_contributor_count(footprints, canvas, bounds: zg.GlobalBounds) -> int:
    """Number of frames whose projected footprint intersects a cell's patch rect."""
    rect = zg._rect(bounds)
    n = 0
    for _f, poly in footprints:
        if poly.intersection(rect).area > zg.INTERSECTION_AREA_EPS:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Layout decision + choose_layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayoutCellPrediction:
    """Per-cell memory prediction (exact N from geometry, exact patch area)."""

    cell_id: str
    row: int
    col: int
    n_contributors: int
    patch_area_px: int
    predicted_peak_bytes: float
    predicted_bound_bytes: float


@dataclass
class LayoutDecision:
    """The deterministic layout decision + full provenance."""

    nx: int
    ny: int
    status: str                     # "ok" | "warn"
    ram_budget_bytes: int | None
    available_bytes: int
    median_footprint_w: float
    median_footprint_h: float
    max_contributors: int           # true max N over cells (exact membership)
    refinement_factor: float
    predicted_bound_bytes: float    # conservative bound on the worst cell (exact)
    predicted_mean_bytes: float     # mean prediction on the worst cell (exact)
    max_patch_area: int             # exact worst-cell patch area
    model: dict
    floors: dict
    budget_bound_choice: bool
    warnings: tuple[str, ...] = ()
    cells: tuple[LayoutCellPrediction, ...] = ()

    def to_dict(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "status": self.status,
            "ram_budget_bytes": self.ram_budget_bytes,
            "available_bytes": self.available_bytes,
            "median_footprint_w": self.median_footprint_w,
            "median_footprint_h": self.median_footprint_h,
            "max_contributors": self.max_contributors,
            "refinement_factor": self.refinement_factor,
            "predicted_bound_bytes": self.predicted_bound_bytes,
            "predicted_mean_bytes": self.predicted_mean_bytes,
            "max_patch_area": self.max_patch_area,
            "model": self.model,
            "floors": self.floors,
            "budget_bound_choice": self.budget_bound_choice,
            "warnings": list(self.warnings),
            "cells": [c.__dict__ for c in self.cells],
        }


def _scan_layout(
    canvas: zg.GlobalCanvas,
    footprints,
    nx: int,
    ny: int,
    halo_px: int,
    model: MemoryModel,
) -> tuple[float, float, int, int]:
    """Exact worst-cell peak for a candidate layout.

    Iterates every Cell, computes its exact patch contributor count (geometry
    membership) and exact patch area, and returns
    ``(worst_bound_bytes, worst_mean_bytes, worst_n, worst_patch_area)`` for the
    cell that maximises ``predict_bound_bytes``. Deterministic (row-major).
    """
    layout = zg.build_layout(canvas, nx, ny)
    worst_bound = -1.0
    worst_mean = -1.0
    worst_n = 0
    worst_area = 0
    for row, col, bounds in layout.iter_cells(canvas):
        cid = zg.cell_id(row, col)
        cell = zg.ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col, bounds)
        patch = zg.build_patch(canvas, cell, halo_px)
        n = _cell_contributor_count(footprints, canvas, patch.patch)
        area = patch.patch.width * patch.patch.height
        bound = model.predict_bound_bytes(n, area)
        if bound > worst_bound:
            worst_bound = bound
            worst_mean = model.predict_peak_bytes(n, area)
            worst_n = n
            worst_area = area
    return worst_bound, worst_mean, worst_n, worst_area


def _predict_cells(
    canvas: zg.GlobalCanvas,
    footprints,
    nx: int,
    ny: int,
    halo_px: int,
    model: MemoryModel,
) -> tuple[tuple[LayoutCellPrediction, ...], int]:
    """Exact per-cell predictions (geometry membership + patch area)."""
    layout = zg.build_layout(canvas, nx, ny)
    out = []
    max_n = 0
    for row, col, bounds in layout.iter_cells(canvas):
        cid = zg.cell_id(row, col)
        cell = zg.ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col, bounds)
        patch = zg.build_patch(canvas, cell, halo_px)
        # Count PATCH contributors (frames intersecting the halo-extended patch),
        # matching what the executor actually reads and stacks (n_patch_contributors).
        n = _cell_contributor_count(footprints, canvas, patch.patch)
        max_n = max(max_n, n)
        area = patch.patch.width * patch.patch.height
        out.append(LayoutCellPrediction(
            cell_id=cid, row=row, col=col, n_contributors=n, patch_area_px=area,
            predicted_peak_bytes=model.predict_peak_bytes(n, area),
            predicted_bound_bytes=model.predict_bound_bytes(n, area),
        ))
    return tuple(out), max_n


def choose_layout(
    canvas: zg.GlobalCanvas,
    frames: Sequence[zg.FrameDescriptor],
    ram_budget: int | None,
    floors: ScientificFloors | None = None,
    halo_px: int = zsw.HALO_PX,
    model: MemoryModel | None = None,
) -> LayoutDecision:
    """Choose a deterministic RAM-aware ``(nx, ny)`` layout.

    * ``ram_budget`` — peak-RSS budget in BYTES (``None`` = no budget -> coarsest
      sensible layout, cell ~= median projected footprint).
    * ``floors`` — scientific floors (defaults if ``None``).
    * ``model`` — memory model (defaults to the fitted R3+tight model).

    The chosen layout is the COARSEST (fewest Cells) whose conservative peak
    bound (computed from the candidate's EXACT contributor counts and patch areas)
    fits ``ram_budget`` while honouring the floors. A tighter budget forces a
    finer (smaller-Cell) layout — monotonic. Raises :class:`LayoutInfeasible`
    if no layout honours both the budget and the floors (never silent).
    """
    floors = floors or ScientificFloors()
    model = model or FITTED_MEMORY_MODEL
    available = zsw.read_available_memory()

    mw, mh = median_projected_footprint(frames, canvas)
    if not (mw > 0 and mh > 0):
        raise LayoutInfeasible("median projected footprint is degenerate")

    footprints = _footprints(frames, canvas)

    # Enumerate candidate layouts (coarse -> fine), deduping (nx, ny), computing
    # the EXACT worst-cell bound from geometry/membership for each. The floors are
    # MONOTONIC in refinement: as (nx, ny) grow, min_patch_area shrinks and
    # halo_overhead grows, so once either floor is violated, every finer candidate
    # also violates it -> stop early (never scan the huge fine layouts).
    candidates = []
    seen: set[tuple[int, int]] = set()
    for factor in REFINEMENT_FACTORS:
        nx = max(1, int(math.ceil(canvas.width / (mw / factor))))
        ny = max(1, int(math.ceil(canvas.height / (mh / factor))))
        nx = min(nx, canvas.width)
        ny = min(ny, canvas.height)
        if (nx, ny) in seen:
            continue
        seen.add((nx, ny))

        geom = _nominal_geometry(canvas, nx, ny, halo_px)
        patch_ok = geom["min_patch_area"] >= floors.min_patch_area_px
        halo_ok = geom["halo_overhead"] <= floors.max_halo_overhead
        if not (patch_ok and halo_ok):
            # Floors are monotonic in refinement: finer candidates also fail.
            break

        worst_bound, worst_mean, worst_n, worst_area = _scan_layout(
            canvas, footprints, nx, ny, halo_px, model
        )
        memory_ok = (ram_budget is None) or (worst_bound <= ram_budget)
        candidates.append({
            "factor": factor, "nx": nx, "ny": ny,
            "bound": worst_bound, "mean": worst_mean,
            "worst_n": worst_n, "worst_area": worst_area,
            "memory_ok": memory_ok, "patch_ok": patch_ok, "halo_ok": halo_ok,
            "geom": geom,
        })

    # Pick the coarsest feasible candidate.
    chosen = None
    chosen_index = None
    for i, cand in enumerate(candidates):
        if cand["memory_ok"]:
            chosen = cand
            chosen_index = i
            break

    if chosen is None:
        # Diagnose WHY nothing was feasible (explicit, never silent).
        finest = candidates[-1] if candidates else None
        reasons = []
        if ram_budget is not None and finest is not None and finest["bound"] > ram_budget:
            reasons.append(
                f"even the finest floor-feasible layout ({finest['nx']}x{finest['ny']}) peak bound "
                f"{finest['bound'] / 2**20:.1f} MiB exceeds budget {ram_budget / 2**20:.1f} MiB"
            )
        reasons.append(
            f"min_patch_area floor ({floors.min_patch_area_px} px) / "
            f"max_halo_overhead floor ({floors.max_halo_overhead}) cannot be satisfied "
            f"together with the budget (search stopped at the floor boundary)"
        )
        raise LayoutInfeasible(
            "RAM budget cannot satisfy scientific floors: " + "; ".join(reasons),
            floors_report={
                "finest": {
                    k: finest["geom"][k] for k in ("nx", "ny", "min_patch_area")
                } if finest else {}
            },
        )

    # Post-selection: exact per-cell predictions + min_contributors floor check.
    cells, max_n = _predict_cells(canvas, footprints, chosen["nx"], chosen["ny"], halo_px, model)
    warnings: list[str] = []
    contrib_ok = max_n >= floors.min_contributors
    if not contrib_ok:
        warnings.append(
            f"deepest cell has {max_n} contributors < min_contributors "
            f"({floors.min_contributors}); the layout may be scientifically degraded"
        )

    floors_report = {
        "min_patch_area_px": {
            "value": chosen["geom"]["min_patch_area"], "limit": floors.min_patch_area_px,
            "ok": chosen["patch_ok"],
        },
        "max_halo_overhead": {
            "value": round(chosen["geom"]["halo_overhead"], 6), "limit": floors.max_halo_overhead,
            "ok": chosen["halo_ok"],
        },
        "min_contributors": {"value": max_n, "limit": floors.min_contributors, "ok": contrib_ok},
    }

    # budget_bound_choice: True if the budget actually constrained the choice
    # (i.e. a coarser candidate was rejected specifically because its memory
    # bound exceeded the budget).
    constrained = False
    if ram_budget is not None and chosen_index is not None:
        for cand in candidates[:chosen_index]:
            if not cand["memory_ok"]:
                constrained = True
                break

    return LayoutDecision(
        nx=chosen["nx"],
        ny=chosen["ny"],
        status="ok",
        ram_budget_bytes=ram_budget,
        available_bytes=available,
        median_footprint_w=mw,
        median_footprint_h=mh,
        max_contributors=max_n,
        refinement_factor=chosen["factor"],
        predicted_bound_bytes=chosen["bound"],
        predicted_mean_bytes=chosen["mean"],
        max_patch_area=chosen["worst_area"],
        model={"fitted": FITTED_MEMORY_MODEL_META, "model": model.to_dict()},
        floors=floors_report,
        budget_bound_choice=constrained,
        warnings=tuple(warnings),
        cells=cells,
    )
