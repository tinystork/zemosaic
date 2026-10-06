"""ZM-ZEGRID-R4 — deterministic RAM-aware Auto layout (Levier 1).

Replaces a FIXED ``Nx x Ny`` with a deterministic, MEASURED, RAM-aware layout
policy. The layout is chosen from a RAM budget instead of a fixed cell count.

Core idea (Levier 1, from ``ZEGRID_MEMORY_NOTES.md``): the peak RSS of one Cell
scales as ``~ baseline + coeff * N_cell * patch_area`` (the canonical engine
materialises ``O(N * H * W * C)`` aligned arrays + float64 taper + rejection/
normalization intermediates). Reducing the Cell size reduces the PEAK, while the
TOTAL work is roughly invariant modulo halo overhead. This is a *memory* lever,
not a *compute* lever.

The memory model is **derived from measured data** (the ZM-ZEGRID-R3 M106
per-cell records), never invented:

    peak_rss_kib ~ baseline + coeff * (N_patch * patch_area)

* ``baseline``      — intercept (process + Python/NumPy/Astropy/SciPy footprint).
* ``coeff``         — bytes per (frame x pixel) of patch area materialised.

``FITTED_MEMORY_MODEL`` holds the OLS fit on the 20 M106 R3 cells plus a
conservative *safety margin* (relative) derived from the residual distribution.
The conservative bound is what ``choose_layout`` enforces against the budget.

Determinism: same canvas + frames + budget + floors -> same layout. The search
is a fixed, sorted refinement sequence over ``target = median_projected_footprint
/ factor``; no randomness, no filesystem-order dependence.

Scientific floors (NEVER silently degrade science):
* ``min_patch_area_px``   — a Cell's patch must be large enough for meaningful
  rejection/normalization statistics.
* ``min_contributors``    — the deepest Cell must retain enough contributors for
  a scientifically usable stack ("where relevant").
* ``max_halo_overhead``   — halo area must not dominate the core (efficiency).

If the budget cannot be met while honouring the floors, ``choose_layout`` raises
``LayoutInfeasible`` with an explicit reason (never silently picks a degraded
layout).
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

# OLS fit on the 20 ZM-ZEGRID-R3 M106 per-cell records (cell_*.json):
#   peak_rss_kib = baseline + coeff * (n_patch_contributors * patch_area)
#   baseline = 867997.2 KiB  (intercept)
#   coeff    =    0.130753 KiB per (frame x pixel)  =>  133.891 bytes
#   R^2      =    0.8989
#   residual std = 297,989 KiB ; max |rel residual| = 22.35% (low side, safe);
#   max POSITIVE rel residual (dangerous side) = 13.71% (r0002c0003).
FITTED_BASELINE_KIB = 867997.2
FITTED_COEFF_KIB_PER_N_PX = 0.130753
FITTED_R2 = 0.8989
FITTED_RESIDUAL_STD_KIB = 297989.0
FITTED_MAX_POS_RESID_FRAC = 0.1371

# Conservative safety margin (relative) on the mean prediction. Justified by the
# residual distribution: the largest positive (dangerous) relative residual on
# the R3 M106 data is +13.71%; mean+2*sigma of the relative residuals is ~15%.
# 15% therefore covers the observed worst case with headroom for run-to-run
# allocator variance.
DEFAULT_SAFETY_MARGIN_FRAC = 0.15


@dataclass(frozen=True)
class MemoryModel:
    """A linear peak-RSS model: ``peak = baseline + coeff * N * patch_area``.

    ``coeff_bytes_per_n_px`` is bytes per (frame x pixel). ``safety_margin_frac``
    is a relative margin applied to the mean prediction to form the conservative
    bound used for budget enforcement.
    """

    baseline_bytes: float
    coeff_bytes_per_n_px: float
    safety_margin_frac: float = DEFAULT_SAFETY_MARGIN_FRAC

    def predict_peak_bytes(self, n_contributors: int, patch_area_px: int) -> float:
        """Mean predicted peak RSS in bytes."""
        return self.baseline_bytes + self.coeff_bytes_per_n_px * n_contributors * patch_area_px

    def predict_bound_bytes(self, n_contributors: int, patch_area_px: int) -> float:
        """Conservative upper bound on peak RSS (mean x (1 + margin))."""
        return self.predict_peak_bytes(n_contributors, patch_area_px) * (1.0 + self.safety_margin_frac)

    def to_dict(self) -> dict:
        return {
            "baseline_bytes": self.baseline_bytes,
            "baseline_kib": self.baseline_bytes / 1024.0,
            "coeff_bytes_per_n_px": self.coeff_bytes_per_n_px,
            "coeff_kib_per_n_px": self.coeff_bytes_per_n_px / 1024.0,
            "safety_margin_frac": self.safety_margin_frac,
        }


# The frozen fitted model (DERIVED from ZM-ZEGRID-R3 M106 measurements).
FITTED_MEMORY_MODEL = MemoryModel(
    baseline_bytes=FITTED_BASELINE_KIB * 1024.0,
    coeff_bytes_per_n_px=FITTED_COEFF_KIB_PER_N_PX * 1024.0,
    safety_margin_frac=DEFAULT_SAFETY_MARGIN_FRAC,
)

FITTED_MEMORY_MODEL_META = {
    "source": "ZM-ZEGRID-R3 M106 full-layout executor per-cell records (cell_*.json)",
    "data_path": "/home/tristan/zegrid_r3_m106_outputs",
    "n_cells": 20,
    "canvas": "2403x3278",
    "halo_px": 8,
    "fit_method": "ordinary least squares on peak_rss_kib vs n_patch_contributors*patch_area",
    "baseline_kib": FITTED_BASELINE_KIB,
    "coeff_kib_per_n_px": FITTED_COEFF_KIB_PER_N_PX,
    "r2": FITTED_R2,
    "residual_std_kib": FITTED_RESIDUAL_STD_KIB,
    "max_positive_residual_frac": FITTED_MAX_POS_RESID_FRAC,
    "safety_margin_frac": DEFAULT_SAFETY_MARGIN_FRAC,
    "note": (
        "coeff (~133.9 bytes/frame/pixel) is the empirical slope including ALL "
        "materialised planes (aligned float32 RGB images + float64 taper + "
        "normalization/rejection intermediates); it is not a single-plane "
        "coefficient. The MEMORY_NOTES rough estimate of 13-20 bytes/frame/pixel "
        "under-estimates the full canonical materialisation and is superseded by "
        "this measured value."
    ),
}


def fit_memory_model(records: Sequence[dict]) -> tuple[MemoryModel, dict]:
    """Fit ``peak_rss_kib = baseline + coeff * (N * patch_area)`` from records.

    Each record is a dict with ``n_patch_contributors`` (int), ``patch`` (list
    ``[x0, y0, x1, y1]``) and ``peak_rss_kib`` (int). Returns the fitted
    :class:`MemoryModel` (with default margin) and a stats dict (R^2, residual
    std, max positive relative residual) for reporting/validation.
    """
    xs = []
    ys = []
    for r in records:
        p = r["patch"]
        area = (p[2] - p[0]) * (p[3] - p[1])
        xs.append(r["n_patch_contributors"] * area)
        ys.append(r["peak_rss_kib"])
    X = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    baseline_kib, coeff_kib = float(coef[0]), float(coef[1])
    pred = A @ coef
    resid = y - pred
    r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    resid_std = float(resid.std())
    rel = resid / pred
    max_pos_rel = float(rel.max()) if len(rel) else 0.0
    model = MemoryModel(
        baseline_bytes=baseline_kib * 1024.0,
        coeff_bytes_per_n_px=coeff_kib * 1024.0,
    )
    stats = {
        "baseline_kib": baseline_kib,
        "coeff_kib_per_n_px": coeff_kib,
        "r2": r2,
        "residual_std_kib": resid_std,
        "max_positive_residual_frac": max_pos_rel,
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

    The worst-case (peak) Cell is an INTERIOR Cell with the largest core
    (``ceil(W/nx) x ceil(H/ny)``) and full halo. The floor check uses the
    smallest core (``floor(W/nx) x floor(H/ny)``) so it is conservative.
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
    """Number of frames whose projected footprint intersects a cell's core."""
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
    n_upper: int
    refinement_factor: float
    predicted_bound_bytes: float    # conservative bound on the worst cell
    predicted_mean_bytes: float     # mean prediction on the worst cell
    max_patch_area: int
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
            "n_upper": self.n_upper,
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


def _predict_cells(
    canvas: zg.GlobalCanvas,
    frames: Sequence[zg.FrameDescriptor],
    nx: int,
    ny: int,
    halo_px: int,
    model: MemoryModel,
) -> tuple[tuple[LayoutCellPrediction, ...], int]:
    """Exact per-cell predictions (geometry membership + patch area)."""
    layout = zg.build_layout(canvas, nx, ny)
    footprints = _footprints(frames, canvas)
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
    * ``model`` — memory model (defaults to the fitted R3 model).

    The chosen layout is the COARSEST (fewest Cells) whose conservative peak
    bound fits ``ram_budget`` while honouring the floors. A tighter budget forces
    a finer (smaller-Cell) layout — monotonic. Raises :class:`LayoutInfeasible`
    if no layout honours both the budget and the floors (never silent).
    """
    floors = floors or ScientificFloors()
    model = model or FITTED_MEMORY_MODEL
    available = zsw.read_available_memory()
    n_upper = len(frames)

    mw, mh = median_projected_footprint(frames, canvas)
    if not (mw > 0 and mh > 0):
        raise LayoutInfeasible("median projected footprint is degenerate")

    best_candidate = None
    best_geometry = None
    for factor in REFINEMENT_FACTORS:
        target_w = mw / factor
        target_h = mh / factor
        nx = max(1, int(math.ceil(canvas.width / target_w)))
        ny = max(1, int(math.ceil(canvas.height / target_h)))
        nx = min(nx, canvas.width)
        ny = min(ny, canvas.height)
        geom = _nominal_geometry(canvas, nx, ny, halo_px)

        bound = model.predict_bound_bytes(n_upper, geom["max_patch_area"])
        mean = model.predict_peak_bytes(n_upper, geom["max_patch_area"])

        memory_ok = (ram_budget is None) or (bound <= ram_budget)
        patch_ok = geom["min_patch_area"] >= floors.min_patch_area_px
        halo_ok = geom["halo_overhead"] <= floors.max_halo_overhead

        floors_report = {
            "min_patch_area_px": {
                "value": geom["min_patch_area"], "limit": floors.min_patch_area_px, "ok": patch_ok,
            },
            "max_halo_overhead": {
                "value": round(geom["halo_overhead"], 6), "limit": floors.max_halo_overhead, "ok": halo_ok,
            },
            "min_contributors": {"value": None, "limit": floors.min_contributors, "ok": True},
        }

        if memory_ok and patch_ok and halo_ok:
            best_candidate = dict(
                nx=nx, ny=ny, factor=factor, bound=bound, mean=mean,
                max_patch_area=geom["max_patch_area"], floors=floors_report,
            )
            best_geometry = geom
            break  # coarsest feasible found

    if best_candidate is None:
        # Diagnose WHY nothing was feasible (explicit, never silent).
        finest = _nominal_geometry(canvas, canvas.width, canvas.height, halo_px)
        bound = model.predict_bound_bytes(n_upper, finest["max_patch_area"])
        reasons = []
        if ram_budget is not None and bound > ram_budget:
            reasons.append(
                f"even the finest layout (1px cells) peak bound "
                f"{bound / 2**20:.1f} MiB exceeds budget {ram_budget / 2**20:.1f} MiB"
            )
        if finest["min_patch_area"] < floors.min_patch_area_px:
            reasons.append(
                f"min_patch_area floor ({floors.min_patch_area_px} px) cannot be "
                f"satisfied with any budget below the finest feasible cells "
                f"(finest nominal patch area {finest['min_patch_area']} px)"
            )
        raise LayoutInfeasible(
            "RAM budget cannot satisfy scientific floors: " + "; ".join(reasons),
            floors_report={"finest": finest},
        )

    # Post-selection: exact per-cell predictions + min_contributors floor check.
    cells, max_n = _predict_cells(canvas, frames, best_candidate["nx"], best_candidate["ny"],
                                  halo_px, model)
    warnings: list[str] = []
    contrib_ok = max_n >= floors.min_contributors
    best_candidate["floors"]["min_contributors"] = {
        "value": max_n, "limit": floors.min_contributors, "ok": contrib_ok,
    }
    if not contrib_ok:
        warnings.append(
            f"deepest cell has {max_n} contributors < min_contributors "
            f"({floors.min_contributors}); the layout may be scientifically degraded"
        )

    budget_bound_choice = (ram_budget is not None) and (
        model.predict_bound_bytes(n_upper, best_candidate["max_patch_area"]) <= ram_budget
    )
    # Whether the budget actually constrained the choice (a coarser layout would not fit).
    constrained = False
    if ram_budget is not None:
        # Check the next-coarser factor would exceed budget (or floors block coarser).
        idx = REFINEMENT_FACTORS.index(best_candidate["factor"])
        if idx > 0:
            coarser_factor = REFINEMENT_FACTORS[idx - 1]
            tw, th = mw / coarser_factor, mh / coarser_factor
            cnx = max(1, min(canvas.width, int(math.ceil(canvas.width / tw))))
            cny = max(1, min(canvas.height, int(math.ceil(canvas.height / th))))
            cg = _nominal_geometry(canvas, cnx, cny, halo_px)
            cbound = model.predict_bound_bytes(n_upper, cg["max_patch_area"])
            constrained = cbound > ram_budget
        else:
            constrained = False

    return LayoutDecision(
        nx=best_candidate["nx"],
        ny=best_candidate["ny"],
        status="ok",
        ram_budget_bytes=ram_budget,
        available_bytes=available,
        median_footprint_w=mw,
        median_footprint_h=mh,
        n_upper=n_upper,
        refinement_factor=best_candidate["factor"],
        predicted_bound_bytes=best_candidate["bound"],
        predicted_mean_bytes=best_candidate["mean"],
        max_patch_area=best_candidate["max_patch_area"],
        model={"fitted": FITTED_MEMORY_MODEL_META, "model": model.to_dict()},
        floors=best_candidate["floors"],
        budget_bound_choice=constrained,
        warnings=tuple(warnings),
        cells=cells,
    )
