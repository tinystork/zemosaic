"""ZM-ZEGRID-R37-B — inter-Cell ADDITIVE background model prototype (PURE, UNWIRED).

Isolated Gate-B prototype mandated by ``docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md``.
Never imported by production. Depends only on NumPy/SciPy and the PURE
``cell_offset`` module (reused UNCHANGED as the M0 baseline).

Two additive inter-Cell models on the SAME synthetic corpora:

* **M0 = ``OFFSET_ONLY``** — current R34 scalar additive baseline, reusing
  :func:`zemosaic.core.zegrid.cell_offset.solve_additive_offsets` (per channel,
  changed solver untouched).
* **M1 = ``ROBUST_PLAN``** — per-Cell/channel additive plane
  ``P_i(x,y,c) = a_i,c + b_i,c*x + d_i,c*y`` in GLOBAL normalized canvas
  coordinates ``x,y ∈ [-1,1]``, central anchor ``(a,b,d)=(0,0,0)``.

ADDITIVE-ONLY. No gain, colour matrix, rescaling or hidden multiply.

Discipline encoded here (frozen contract; rework-1 hardening)
------------------------------------------------------------
* Same-global-pixel pair differences; measurement band width SEPARATE from the
  assembly feather width, applied CONSISTENTLY across planes/masks/coords
  (rework H1). A requested band wider than the available overlap is reported as
  ``band_clamped`` — never silently claimed.
* Deterministic 75/25 train/holdout by fixed global block parity; no RNG.
* Robust Huber edge fit (``1.345·1.4826·MAD``), float64.
* Per-channel handling: connectivity, rank, condition and uncertainty are
  computed PER CHANNEL; the global gate takes the WORST channel (rework H2/H3).
  A channel that is under-determined or disconnected yields NaN coefficients and
  an explicit refusal — never a silent zero.
* Sparse whitened LSQR global solve; deterministic per-edge information cap
  (``EDGE_INFO_CAP_MULT``) and bounded graph-level IRLS (rework M3), so one huge
  or contradictory relation cannot dominate.
* Slopes-only regularization over the FROZEN ladder, selected train/holdout only.
  §6.1.6 corner uncertainty is evaluated on the DATA-ONLY system (rework M2);
  the regularized covariance is reported separately as a shrinkage estimate.
* Held-out metrics RMS around ZERO (bias-preserving, symmetric operators, rework
  M1); p90 uses the frozen ``cfg.per_edge_p90_worsen_max``.
* Decision: ``OFFSET_ONLY | ROBUST_PLAN | INSUFFICIENT_EVIDENCE |
  ILL_CONDITIONED | MODEL_MISMATCH | DISCONNECTED``. ``NONLINEAR_CANDIDATE`` is
  research-only (no engine).

Rework-1 finding map: H1 band consistency; H2 no solver exception escapes; H3
per-channel diagnostics; M1 held-out gate; M2 data-only uncertainty; M3 edge
cap + graph IRLS; M4 cycle residuals + refusal reasons; M5 M0 train-only fit;
M6 σΔ→0 branch; M7 provisional mismatch thresholds; L1 densification flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from .cell_offset import (
    DisconnectedOffsetGraphError,
    OffsetEdge,
    grid_adjacency,
    select_central_cell,
    solve_additive_offsets,
)

__all__ = [
    "MODEL_OFFSET_ONLY",
    "MODEL_ROBUST_PLAN",
    "MODEL_NONLINEAR_CANDIDATE",
    "STATE_OFFSET_ONLY",
    "STATE_ROBUST_PLAN",
    "STATE_INSUFFICIENT_EVIDENCE",
    "STATE_ILL_CONDITIONED",
    "STATE_MODEL_MISMATCH",
    "STATE_DISCONNECTED",
    "REGULARIZATION_LADDER",
    "HUBER_DELTA_FACTOR",
    "EDGE_INFO_CAP_MULT",
    "MAX_IRLS",
    "BLOCK_PX",
    "DEFAULT_MEASUREMENT_PX",
    "DEFAULT_FEATHER_PX",
    "GateConfig",
    "PlanCell",
    "EdgeObservation",
    "ModelComparisonResult",
    "normalized_coords",
    "observe_edge",
    "build_observations",
    "solve_offset_baseline",
    "solve_plan_sparse",
    "decide_state",
    "compare_models",
    "gauge_truth_error",
    "apply_plane_correction",
    "plane_field",
    "sparse_scale_witness",
    "nonlinear_study",
]

# ---------------------------------------------------------------------------
# Vocabulary (contract §1, §5)
# ---------------------------------------------------------------------------

MODEL_OFFSET_ONLY = "OFFSET_ONLY"
MODEL_ROBUST_PLAN = "ROBUST_PLAN"
MODEL_NONLINEAR_CANDIDATE = "NONLINEAR_CANDIDATE"

STATE_OFFSET_ONLY = "OFFSET_ONLY"
STATE_ROBUST_PLAN = "ROBUST_PLAN"
STATE_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATE_ILL_CONDITIONED = "ILL_CONDITIONED"
STATE_MODEL_MISMATCH = "MODEL_MISMATCH"
STATE_DISCONNECTED = "DISCONNECTED"

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

HUBER_DELTA_FACTOR = 1.345
MAD_SCALE = 1.4826
BLOCK_PX = 4
DEFAULT_MEASUREMENT_PX = 16
DEFAULT_FEATHER_PX = 8
REGULARIZATION_LADDER: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: Deterministic per-edge information cap (multiple of the median trace).
EDGE_INFO_CAP_MULT = 100.0
#: Bounded graph-level IRLS iterations.
MAX_IRLS = 8
#: Normalized-residual threshold for the bounded relational rejection pass.
EDGE_REJECT_Z = 8.0
#: At most this fraction of a channel's relations may be rejected.
MAX_REJECT_FRACTION = 0.25
_HUBER_MAX_ITER = 60
_HUBER_TOL = 1e-10
_LSQR_ATOL = 1e-13
_LSQR_BTOL = 1e-13
_LSQR_ITERS = 20000
_DENSE_DIAG_LIMIT = 1500
_EIG_TOL_REL = 1e-12
_CORNER_XY = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

PROVENANCE = {
    "module": "zemosaic.core.zegrid.background_model",
    "mission": "ZM-ZEGRID-R37-B",
    "gate": "B",
    "rework": "r1",
    "status": "pure_prototype_unwired",
    "coordinate_system": "global_canvas_normalized_[-1,1]",
    "model": "P_i(x,y,c) = a + b*x + d*y (additive only)",
    "robust_loss": "huber_irls_fixed_1.345_mad_sigma",
    "global_solve": "scipy.sparse.linalg.lsqr_on_whitened_design",
    "global_solve_densified": False,
    "regularization": "slopes_only_ladder_frozen",
    "split": "deterministic_spatial_75_25_block_parity",
    "per_channel_diagnostics": True,
    "edge_information_cap": EDGE_INFO_CAP_MULT,
    "graph_irls_max_iter": MAX_IRLS,
    "edge_reject_z": EDGE_REJECT_Z,
    "mismatch_thresholds_status": "PROVISIONAL_NOT_IN_FROZEN_CONTRACT",
    "mismatch_thresholds": {"level_z": 3.0, "min_fraction": 0.5},
    "references": [
        "Montage (Caltech/IPAC) — offset/slope overlap matching, calibration assumptions",
        "PhotometricMosaic (John Murphy) — scale matching + surface spline (research only)",
    ],
    "no_third_party_code_copied": True,
}


@dataclass(frozen=True)
class GateConfig:
    """Predeclared acceptance thresholds (contract §6). Do not relax post-hoc.

    NOTE: ``mismatch_level_z`` / ``mismatch_min_fraction`` are PROVISIONAL and are
    NOT part of the frozen contract §6; they drive the experimental
    ``MODEL_MISMATCH`` state only and are reported as provisional thresholds.
    """

    min_effective_train: int = 256
    min_tangent_blocks: int = 8
    min_normal_blocks: int = 3
    cond_a_max: float = 1e4
    heldout_ratio_max: float = 0.80
    heldout_abs_sigma_max: float = 1.25
    per_edge_p90_worsen_max: float = 0.05
    corner_unc_sigma_frac: float = 0.5
    corner_unc_correction_frac: float = 0.5
    mismatch_level_z: float = 3.0          # PROVISIONAL (not frozen contract)
    mismatch_min_fraction: float = 0.5     # PROVISIONAL (not frozen contract)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanCell:
    """One Cell's PATCH-level science plane at its GLOBAL position."""

    cell_id: str
    bounds: tuple[int, int, int, int]
    science: np.ndarray
    valid_mask: np.ndarray | None = None

    def channels(self) -> int:
        a = np.asarray(self.science)
        return 1 if a.ndim == 2 else int(a.shape[2])

    def as_3d(self) -> np.ndarray:
        a = np.asarray(self.science, dtype=np.float64)
        if a.ndim == 2:
            a = a[:, :, None]
        return a


def _valid_2d(cell: PlanCell) -> np.ndarray:
    a = np.asarray(cell.science)
    h, w = a.shape[:2]
    if cell.valid_mask is None:
        return np.ones((h, w), dtype=bool)
    v = np.asarray(cell.valid_mask, dtype=bool)
    if v.ndim == 3:
        v = v[..., 0]
    if v.shape != (h, w):
        raise ValueError(f"cell {cell.cell_id}: valid_mask shape {v.shape} != {(h, w)}")
    return v


def normalized_coords(
    bounds: tuple[int, int, int, int], canvas_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Global normalized coordinates of a patch: ``x,y ∈ [-1,1]`` (contract §2)."""
    y0, y1, x0, x1 = bounds
    h, w = int(canvas_hw[0]), int(canvas_hw[1])
    xs = 2.0 * (np.arange(x0, x1, dtype=np.float64) + 0.5) / float(w) - 1.0
    ys = 2.0 * (np.arange(y0, y1, dtype=np.float64) + 0.5) / float(h) - 1.0
    return xs, ys


# ---------------------------------------------------------------------------
# Robust Huber plane fit (deterministic, no RNG)
# ---------------------------------------------------------------------------

def _robust_plane_fit(
    phi: np.ndarray, target: np.ndarray, *, huber_factor: float = HUBER_DELTA_FACTOR
) -> tuple[np.ndarray, np.ndarray, float]:
    n = target.size
    if n == 0:
        return np.zeros(phi.shape[1]), np.zeros(0), 0.0
    coef, *_ = np.linalg.lstsq(phi, target, rcond=None)
    w = np.ones(n, dtype=np.float64)
    for _ in range(_HUBER_MAX_ITER):
        resid = target - phi @ coef
        med = float(np.median(resid))
        sigma = MAD_SCALE * float(np.median(np.abs(resid - med)))
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        cut = huber_factor * sigma
        absr = np.abs(resid)
        w = np.where(absr <= cut, 1.0, cut / np.maximum(absr, 1e-300))
        sw = np.sqrt(w)
        coef_new, *_ = np.linalg.lstsq(phi * sw[:, None], target * sw, rcond=None)
        delta = float(np.max(np.abs(coef_new - coef)))
        coef = coef_new
        if delta <= _HUBER_TOL * max(1.0, float(np.max(np.abs(coef)))):
            break
    resid = target - phi @ coef
    med = float(np.median(resid))
    sigma = MAD_SCALE * float(np.median(np.abs(resid - med)))
    return coef, w, sigma


@dataclass(frozen=True)
class EdgeObservation:
    """One (edge, channel) compact observation (contract §3)."""

    u: str
    v: str
    channel: int
    orientation: str
    n_train: int
    n_holdout: int
    effective_train: float
    tangent_blocks: int
    normal_blocks: int
    robust_sigma: float
    rejection_fraction: float
    theta_global: np.ndarray          # (3,) robust train fit [Δa, Δb, Δd]
    information: np.ndarray           # (3,3) train normal matrix (global coords)
    rhs: np.ndarray                   # (3,) train rhs (global coords)
    level_z: float
    curvature_z: float
    level_mean: float
    local_offset_train: float         # robust median difference on TRAIN only
    measurement_px: int | None        # requested measurement band
    overlap_px: int                   # available overlap width (normal axis)
    band_used_px: int                 # actually observed normal width
    band_clamped: bool                # requested band wider than available overlap
    holdout_xy: np.ndarray            # (m, 2) global normalized coords
    holdout_diff: np.ndarray          # (m,)


def observe_edge(
    cell_u: PlanCell,
    cell_v: PlanCell,
    canvas_hw: tuple[int, int],
    channel: int,
    *,
    block_px: int = BLOCK_PX,
    measurement_px: int | None = None,
    normal_boundary: float | None = None,
    huber_factor: float = HUBER_DELTA_FACTOR,
) -> tuple[EdgeObservation | None, str]:
    """Pure same-global-pixel observation for one (edge, channel).

    Returns ``(observation_or_None, refusal_reason)``. The measurement band is
    applied CONSISTENTLY to the pixel planes, validity masks and coordinate arrays
    (rework H1); a requested band wider than the overlap is reported as clamped.
    """
    uy0, uy1, ux0, ux1 = cell_u.bounds
    vy0, vy1, vx0, vx1 = cell_v.bounds
    y0, y1 = max(uy0, vy0), min(uy1, vy1)
    x0, x1 = max(ux0, vx0), min(ux1, vx1)
    if y1 <= y0 or x1 <= x0:
        return None, "no_overlap"

    su = cell_u.as_3d()[y0 - uy0 : y1 - uy0, x0 - ux0 : x1 - ux0, channel]
    sv = cell_v.as_3d()[y0 - vy0 : y1 - vy0, x0 - vx0 : x1 - vx0, channel]
    vu = _valid_2d(cell_u)[y0 - uy0 : y1 - uy0, x0 - ux0 : x1 - ux0]
    vv = _valid_2d(cell_v)[y0 - vy0 : y1 - vy0, x0 - vx0 : x1 - vx0]
    hh, ww = su.shape

    is_horizontal = abs(vx0 - ux0) >= abs(vy0 - uy0)
    orientation = "h" if is_horizontal else "v"
    overlap_px = int((x1 - x0) if is_horizontal else (y1 - y0))

    # Global pixel-coordinate grids over the overlap window (normal axis applied
    # to the SAME 2-D mask, so planes/masks/coords stay consistent).
    gx = np.arange(x0, x1, dtype=np.float64)[None, :].repeat(hh, 0) + 0.5
    gy = np.arange(y0, y1, dtype=np.float64)[:, None].repeat(ww, 1) + 0.5
    shared = np.isfinite(su) & np.isfinite(sv) & vu & vv

    band_clamped = False
    if measurement_px is not None and normal_boundary is not None:
        half = 0.5 * float(measurement_px)
        normal_val = gx if is_horizontal else gy
        keep = np.abs(normal_val - float(normal_boundary)) <= half
        band_clamped = float(measurement_px) > float(overlap_px) + 1e-9
        shared = shared & keep

    if not np.any(shared):
        return None, ("band_no_support" if measurement_px is not None else "no_shared_valid_pixels")

    gy_l, gx_l = np.nonzero(shared)
    gxp = gx_l + x0
    gyp = gy_l + y0
    xs, ys = normalized_coords((y0, y1, x0, x1), canvas_hw)
    X = xs[gx_l]
    Y = ys[gy_l]
    diff = su[shared] - sv[shared]
    band_used_px = int(np.ptp((gxp if is_horizontal else gyp))) + 1

    bx = gxp // int(block_px)
    by = gyp // int(block_px)
    holdout = (bx % 2 == 1) & (by % 2 == 1)
    train = ~holdout
    n_train = int(np.count_nonzero(train))
    if n_train < 4:
        return None, "insufficient_train_support"

    xbar = float(np.mean(X[train]))
    ybar = float(np.mean(Y[train]))
    phi_c = np.column_stack([np.ones(n_train), X[train] - xbar, Y[train] - ybar])
    tgt = diff[train]
    coef_c, w, sigma = _robust_plane_fit(phi_c, tgt, huber_factor=huber_factor)
    if not np.isfinite(sigma) or sigma <= 1e-10 * max(float(np.max(np.abs(tgt))) if tgt.size else 0.0, 1.0):
        w = np.ones(n_train, dtype=np.float64)
        sigma = 0.0

    sigma_info = sigma if sigma > 0.0 else 1.0
    info_c = (phi_c.T @ (phi_c * w[:, None])) / (sigma_info * sigma_info)
    rhs_c = (phi_c.T @ (w * tgt)) / (sigma_info * sigma_info)

    M = np.array([[1.0, -xbar, -ybar], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Minv = np.array([[1.0, xbar, ybar], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    theta_g = M @ coef_c
    info_g = Minv.T @ info_c @ Minv
    rhs_g = Minv.T @ rhs_c

    resid = tgt - phi_c @ coef_c
    level = 0.5 * (su[shared][train] + sv[shared][train])
    level_c = level - float(np.mean(level))
    radius = (X[train] - xbar) ** 2 + (Y[train] - ybar) ** 2
    radius_c = radius - float(np.mean(radius))

    def _slope_z(reg: np.ndarray, raw: np.ndarray) -> float:
        if sigma <= 0.0:
            return 0.0
        denom = float(np.sum(w * reg * reg))
        if denom <= 0.0:
            return 0.0
        slope = float(np.sum(w * reg * resid) / denom)
        spread = float(np.percentile(raw, 95) - np.percentile(raw, 5))
        return abs(slope) * spread / sigma

    rejection_fraction = (
        float(np.mean(np.abs(resid) > huber_factor * sigma)) if sigma > 0 else 0.0
    )
    tangent_blocks = int(np.unique((gyp if is_horizontal else gxp) // int(block_px)).size)
    normal_blocks = int(np.unique((bx if is_horizontal else by)).size)

    return EdgeObservation(
        u=cell_u.cell_id,
        v=cell_v.cell_id,
        channel=int(channel),
        orientation=orientation,
        n_train=n_train,
        n_holdout=int(np.count_nonzero(holdout)),
        effective_train=float(np.sum(w)),
        tangent_blocks=tangent_blocks,
        normal_blocks=normal_blocks,
        robust_sigma=float(sigma),
        rejection_fraction=rejection_fraction,
        theta_global=theta_g,
        information=info_g,
        rhs=rhs_g,
        level_z=_slope_z(level_c, level),
        curvature_z=_slope_z(radius_c, radius),
        level_mean=float(np.mean(level)),
        local_offset_train=float(np.median(diff[train])),
        measurement_px=(None if measurement_px is None else int(measurement_px)),
        overlap_px=overlap_px,
        band_used_px=band_used_px,
        band_clamped=bool(band_clamped),
        holdout_xy=np.column_stack([X[holdout], Y[holdout]]),
        holdout_diff=diff[holdout],
    ), ""


def _parse_rc(cell_id: str) -> tuple[int, int]:
    return int(cell_id[1:5]), int(cell_id[6:10])


def build_observations(
    cells: Mapping[str, PlanCell],
    nx: int,
    ny: int,
    canvas_hw: tuple[int, int],
    *,
    channels: int | None = None,
    block_px: int = BLOCK_PX,
    measurement_px: int | None = DEFAULT_MEASUREMENT_PX,
) -> tuple[list[EdgeObservation], list[dict]]:
    """Build every (edge, channel) observation. Returns ``(observations, dropped)``.

    ``dropped`` is an explicit inventory of refused relations/channels with a
    reason (rework M4): nothing is silently skipped.
    """
    if channels is None:
        channels = cells[next(iter(cells))].channels()
    w, h = int(canvas_hw[1]), int(canvas_hw[0])
    cell_w = w / float(nx)
    cell_h = h / float(ny)

    out: list[EdgeObservation] = []
    dropped: list[dict] = []
    for (u, v) in grid_adjacency(nx, ny):
        if u not in cells or v not in cells:
            dropped.append({"u": u, "v": v, "channel": None, "reason": "cell_absent"})
            continue
        cu, cv = cells[u], cells[v]
        (ur, uc), (vr, vc) = _parse_rc(u), _parse_rc(v)
        if vr > ur:
            boundary = float((ur + 1) * cell_h)
        else:
            boundary = float((uc + 1) * cell_w)
        for c in range(channels):
            obs, reason = observe_edge(
                cu, cv, canvas_hw, c,
                block_px=block_px,
                measurement_px=measurement_px,
                normal_boundary=boundary,
            )
            if obs is None:
                dropped.append({"u": u, "v": v, "channel": c, "reason": reason})
            else:
                out.append(obs)
    return out, dropped


# ---------------------------------------------------------------------------
# Per-channel graph helpers
# ---------------------------------------------------------------------------

def _reachable_from(anchor: str, nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> set:
    adj: dict[str, set] = {n: set() for n in nodes}
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    seen = {anchor}
    stack = [anchor]
    while stack:
        cur = stack.pop()
        for nb in sorted(adj[cur]):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def _channel_connectivity(
    cell_ids: Sequence[str], observations: Sequence[EdgeObservation], anchor: str, channels: int
) -> dict:
    out = {}
    for c in range(channels):
        edges = [(o.u, o.v) for o in observations if o.channel == c]
        reach = _reachable_from(anchor, cell_ids, edges) if edges else {anchor}
        missing = sorted(set(cell_ids) - reach)
        out[c] = {"edges": len(edges), "connected": len(missing) == 0, "missing": missing}
    return out


# ---------------------------------------------------------------------------
# M0 baseline (reuses the PURE R34c solver UNCHANGED) — per channel, no escape
# ---------------------------------------------------------------------------

def solve_offset_baseline(
    observations: Sequence[EdgeObservation],
    cell_ids: Sequence[str],
    anchor: str,
    channels: int,
    *,
    channel_connected: Mapping[int, bool] | None = None,
) -> np.ndarray:
    """M0 = ``OFFSET_ONLY`` via the UNCHANGED R34c solver, ONE CHANNEL AT A TIME.

    Per-channel isolation prevents a channel with no observation on an edge from
    producing a zero-weight cut (rework H2). Uses the TRAIN-only local offset and
    never fabricates a delta/weight for a missing relation. A disconnected channel
    is solved with ``raise_on_disconnected=False`` (unreachable Cells stay 0); the
    refusal is recorded by the caller via per-channel connectivity.
    """
    n = len(cell_ids)
    index = {cid: i for i, cid in enumerate(cell_ids)}
    offsets = np.zeros((n, channels), dtype=np.float64)
    for c in range(channels):
        obs_c = [o for o in observations if o.channel == c]
        if not obs_c:
            continue
        by_pair: dict[tuple[str, str], list[float]] = {}
        for o in obs_c:
            by_pair.setdefault((o.u, o.v), []).append(float(o.n_train + o.n_holdout))
        edges = [
            OffsetEdge(u, v, np.asarray([o.local_offset_train], dtype=np.float64),
                       np.asarray([max(o.n_train + o.n_holdout, 1)], dtype=np.float64))
            for (u, v) in by_pair
            for o in [next(x for x in obs_c if x.u == u and x.v == v)]
        ]
        try:
            sol = solve_additive_offsets(
                list(cell_ids), edges, anchor, 1, raise_on_disconnected=False
            )
            offsets[:, c] = np.asarray(sol.offsets, dtype=np.float64).ravel()
        except DisconnectedOffsetGraphError:
            # Defensive: never let a solver refusal escape compare_models.
            offsets[:, c] = 0.0
    return offsets


# ---------------------------------------------------------------------------
# Global sparse plan solve (whitened, slopes-only ridge, edge cap + graph IRLS)
# ---------------------------------------------------------------------------

@dataclass
class PlanSolve:
    coefficients: np.ndarray           # (N, C, 3); NaN where a channel is refused
    reg_lambda: float
    rank_data: int | None              # worst channel
    cond_data: float | None            # worst channel
    unknowns: int
    rows: int
    coef_uncertainty: np.ndarray | None            # data-only (gate)
    corner_uncertainty: np.ndarray | None          # data-only (gate)
    coef_uncertainty_regularized: np.ndarray | None
    corner_uncertainty_regularized: np.ndarray | None
    per_channel: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _edge_whitening(info: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(R, trace)`` with ``R^T R = info`` (kept positive eigen-directions)."""
    info = 0.5 * (np.asarray(info, dtype=np.float64) + np.asarray(info, dtype=np.float64).T)
    tr = float(np.trace(info))
    if not np.all(np.isfinite(info)):
        return np.zeros((0, 3)), tr
    evals, evecs = np.linalg.eigh(info)
    if evals.size == 0:
        return np.zeros((0, 3)), tr
    cut = max(float(np.max(evals)) * _EIG_TOL_REL, 0.0)
    keep = [k for k in range(3) if float(evals[k]) > cut and float(evals[k]) > 0.0]
    if not keep:
        return np.zeros((0, 3)), tr
    r = np.vstack([evecs[:, k] * np.sqrt(float(evals[k])) for k in keep])
    return r, tr


def _data_only_covariance(a_data: csr_matrix, limit: int) -> tuple[np.ndarray | None, bool]:
    """Data-only covariance via dense SVD of the DESIGN (explicitly densified).

    Returns ``(cov, densified)``. ``None`` above the column limit → NOT_RUN.
    """
    if a_data.shape[1] == 0 or a_data.shape[0] == 0 or a_data.shape[1] > limit:
        return None, False
    dense = a_data.toarray().astype(np.float64)
    u, s, vt = np.linalg.svd(dense, full_matrices=False)
    if s.size == 0:
        return None, True
    tol = max(dense.shape) * np.finfo(np.float64).eps * float(s[0])
    inv = np.where(s > tol, 1.0 / (s * s), 0.0)
    cov = (vt.T * inv) @ vt
    return cov, True


def _corner_from_cov(cov, cell_slot, anchor, channels):
    n = max(len(cell_slot), 1)
    coef_sigma = np.full((n, 3), np.inf, dtype=np.float64)
    corner_sigma = np.full((n, 4), np.inf, dtype=np.float64)
    for cid, slot in cell_slot.items():
        if cid == anchor:
            coef_sigma[slot, :] = 0.0
            corner_sigma[slot, :] = 0.0
            continue
        base = 3 * slot
        block = cov[base : base + 3, base : base + 3]
        if np.all(np.isfinite(block)):
            coef_sigma[slot] = np.sqrt(np.maximum(np.diag(block), 0.0))
        for ci, (cx, cy) in enumerate(_CORNER_XY):
            phi = np.array([1.0, cx, cy])
            corner_sigma[slot, ci] = np.sqrt(max(float(phi @ block @ phi), 0.0))
    return coef_sigma, corner_sigma


def solve_plan_sparse(
    observations: Sequence[EdgeObservation],
    cell_ids: Sequence[str],
    anchor: str,
    channels: int,
    *,
    reg_lambda: float = 0.0,
    with_uncertainty: bool = True,
    channel_connected: Mapping[int, bool] | None = None,
) -> PlanSolve:
    """Global M1 solve: sparse whitened LSQR per channel, slopes-only ridge.

    Per channel: deterministic information cap + bounded graph IRLS, data-only
    rank/condition, data-only covariance for the gate, regularized covariance as a
    separate shrinkage estimate. Under-determined/disconnected channels get NaN
    coefficients and are refused (never silent zeros).
    """
    if anchor not in cell_ids:
        raise ValueError(f"anchor {anchor!r} not among cells")
    index = {cid: i for i, cid in enumerate(cell_ids)}
    cell_slot = {cid: i for i, cid in enumerate(c for c in cell_ids if c != anchor)}
    n = len(cell_ids)
    coeffs = np.zeros((n, channels, 3), dtype=np.float64)
    unknowns = 3 * max(n - 1, 0)
    n_free = 3 * len(cell_slot)

    coef_unc = None
    corner_unc = None
    coef_unc_reg = None
    corner_unc_reg = None
    per_channel: dict = {}
    totals = {"rows": 0, "diagnostics_densified": False, "covariance_densified": False}
    rank_worst: int | None = None
    cond_worst: float | None = None

    for c in range(channels):
        obs_c = [o for o in observations if o.channel == c]
        connected = True if channel_connected is None else bool(channel_connected.get(c, True))

        # Base whitening + deterministic information cap.
        base = []
        traces = []
        for o in obs_c:
            r, tr = _edge_whitening(o.information)
            base.append((o, r, np.asarray(o.theta_global, dtype=np.float64)))
            traces.append(tr)
        med_tr = float(np.median(traces)) if traces else 0.0
        cap_tr = EDGE_INFO_CAP_MULT * med_tr if med_tr > 0 else 0.0
        scales = []
        edges_capped = 0
        for (o, r, _t), tr in zip(base, traces):
            s = 1.0
            if cap_tr > 0.0 and tr > cap_tr:
                s = cap_tr / tr
                edges_capped += 1
            scales.append(s)

        # Data-only design diagnostics + data-only covariance (gate).
        a_data, _b_data = _assemble(base, scales, np.ones(len(base)), anchor, cell_slot, reg_lambda=0.0)
        rank_c, cond_c = _dense_rank_cond(a_data)
        rt = max(float(np.max(traces)) * _EIG_TOL_REL, 0.0) if traces else 0.0
        densified_diag = a_data.shape[1] <= _DENSE_DIAG_LIMIT and a_data.shape[1] > 0 and a_data.shape[0] > 0
        cov_data = None
        if with_uncertainty:
            cov_data, cov_dens = _data_only_covariance(a_data, _DENSE_DIAG_LIMIT)
            totals["covariance_densified"] = totals["covariance_densified"] or cov_dens
        totals["diagnostics_densified"] = totals["diagnostics_densified"] or densified_diag

        # Bounded graph-level IRLS (deterministic).
        w_edge = np.ones(len(base), dtype=np.float64)
        irls_used = 0
        sol = np.zeros(n_free, dtype=np.float64)
        for it in range(MAX_IRLS):
            a, b = _assemble(base, scales, w_edge, anchor, cell_slot, reg_lambda=reg_lambda)
            sol = _solve_lsqr(a, b)
            t = _edge_norm_residuals(base, scales, w_edge, sol, anchor, cell_slot)
            irls_used = it + 1
            if t.size < 4:
                break
            sig = MAD_SCALE * float(np.median(np.abs(t - np.median(t))))
            if not np.isfinite(sig) or sig <= 1e-12:
                break
            cut = HUBER_DELTA_FACTOR * sig
            w_new = np.where(t <= cut, 1.0, cut / np.maximum(t, 1e-300))
            if float(np.max(np.abs(w_new - w_edge))) <= 1e-6:
                w_edge = w_new
                break
            w_edge = w_new

        # Bounded relational rejection: greedily drop the relation whose removal
        # most reduces the robust loss, only if it materially improves and keeps
        # the channel connected. Bounded by MAX_REJECT_FRACTION.
        edges_rejected: list[int] = []
        max_reject = max(1, int(len(base) * MAX_REJECT_FRACTION))
        for _ in range(max_reject):
            t = _edge_norm_residuals(base, scales, w_edge, sol, anchor, cell_slot)
            sig = MAD_SCALE * float(np.median(np.abs(t - np.median(t))))
            if not (np.isfinite(sig) and sig > 0.0):
                break
            order = [int(e) for e in np.argsort(-t) if t[e] > EDGE_REJECT_Z * sig]
            if not order:
                break
            cur_loss = float(np.sum(t[w_edge > 0.0] ** 2))
            best = None
            best_loss = None
            best_sol = None
            best_w = None
            for e in order:
                w2 = w_edge.copy()
                w2[e] = 0.0
                remaining = [(base[k][0].u, base[k][0].v)
                             for k in range(len(base)) if w2[k] > 0.0]
                reach = _reachable_from(anchor, cell_ids, remaining) if remaining else {anchor}
                if len(reach) != len(set(cell_ids)):
                    continue
                a2, b2 = _assemble(base, scales, w2, anchor, cell_slot, reg_lambda=reg_lambda)
                s2 = _solve_lsqr(a2, b2)
                t2 = _edge_norm_residuals(base, scales, w2, s2, anchor, cell_slot)
                loss = float(np.sum(t2[w2 > 0.0] ** 2))
                if best_loss is None or loss < best_loss:
                    best, best_loss, best_sol, best_w = e, loss, s2, w2
            if best is None or best_loss is None or best_loss >= 0.5 * cur_loss:
                break
            w_edge, sol = best_w, best_sol
            edges_rejected.append(best)
        totals["rows"] += int(reg_lambda > 0) * 2 * (len(cell_slot)) + 0

        # Refusal: under-determined / disconnected channel -> NaN, no silent zero.
        under = (not connected) or (rank_c is not None and rank_c < unknowns and obs_c) or (not obs_c)
        if under:
            coeffs[:, c, :] = np.nan
        else:
            for cid, slot in cell_slot.items():
                coeffs[index[cid], c, :] = sol[3 * slot : 3 * slot + 3]

        if with_uncertainty and cov_data is not None:
            cs, cors = _corner_from_cov(cov_data, cell_slot, anchor, channels)
            if coef_unc is None:
                coef_unc = np.full((n, channels, 3), np.nan)
                corner_unc = np.full((n, channels, 4), np.nan)
            for cid, slot in cell_slot.items():
                coef_unc[index[cid], c, :] = cs[slot]
                corner_unc[index[cid], c, :] = cors[slot]
            coef_unc[index[anchor], c, :] = 0.0
            corner_unc[index[anchor], c, :] = 0.0

        # Regularized covariance (separate shrinkage estimate).
        if with_uncertainty and n_free <= _DENSE_DIAG_LIMIT and n_free > 0 and reg_lambda > 0:
            a_reg, _ = _assemble(base, scales, w_edge, anchor, cell_slot, reg_lambda=reg_lambda)
            cov_reg, _ = _data_only_covariance(a_reg, _DENSE_DIAG_LIMIT)
            if cov_reg is not None:
                cs_r, cors_r = _corner_from_cov(cov_reg, cell_slot, anchor, channels)
                if coef_unc_reg is None:
                    coef_unc_reg = np.full((n, channels, 3), np.nan)
                    corner_unc_reg = np.full((n, channels, 4), np.nan)
                for cid, slot in cell_slot.items():
                    coef_unc_reg[index[cid], c, :] = cs_r[slot]
                    corner_unc_reg[index[cid], c, :] = cors_r[slot]
                coef_unc_reg[index[anchor], c, :] = 0.0
                corner_unc_reg[index[anchor], c, :] = 0.0

        if rank_c is not None and (rank_worst is None or rank_c < rank_worst):
            rank_worst = rank_c
        if cond_c is not None and (cond_worst is None or cond_c > cond_worst):
            cond_worst = cond_c

        per_channel[int(c)] = {
            "connected": bool(connected),
            "observations": len(obs_c),
            "rank": rank_c,
            "cond": None if cond_c is None else float(cond_c),
            "under_determined": bool(under),
            "edges_capped": int(edges_capped),
            "irls_iterations": int(irls_used),
            "edges_rejected": len(edges_rejected),
            "covariance_densified": bool(cov_data is not None),
        }

    return PlanSolve(
        coefficients=coeffs,
        reg_lambda=float(reg_lambda),
        rank_data=rank_worst,
        cond_data=cond_worst,
        unknowns=unknowns,
        rows=totals["rows"],
        coef_uncertainty=coef_unc,
        corner_uncertainty=corner_unc,
        coef_uncertainty_regularized=coef_unc_reg,
        corner_uncertainty_regularized=corner_unc_reg,
        per_channel=per_channel,
        diagnostics={
            "global_solve_densified": False,
            "diagnostics_densified": bool(totals["diagnostics_densified"]),
            "covariance_densified": bool(totals["covariance_densified"]),
            "dense_diag_limit": _DENSE_DIAG_LIMIT,
        },
    )


def _assemble(base, scales, w_edge, anchor, cell_slot, *, reg_lambda: float):
    """Build the whitened sparse design/rhs for one channel.

    ``base`` = list of ``(obs, R0, theta)``. Combined per-edge weight is
    ``scales[e] * w_edge[e]``. Slopes-only ridge rows on non-anchor Cells.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    r = 0
    for e, (o, r0, theta) in enumerate(base):
        if r0.shape[0] == 0:
            continue
        s = float(scales[e]) * float(w_edge[e])
        if s <= 0.0:
            continue
        rad = np.sqrt(s)
        sv = cell_slot.get(o.v)
        su = cell_slot.get(o.u)
        for k in range(r0.shape[0]):
            vec = rad * r0[k]
            for j in range(3):
                if sv is not None:
                    rows.append(r); cols.append(3 * sv + j); vals.append(float(vec[j]))
                if su is not None:
                    rows.append(r); cols.append(3 * su + j); vals.append(-float(vec[j]))
            rhs.append(float(vec @ theta))
            r += 1

    n_free = 3 * max(len(cell_slot), 0)
    if reg_lambda > 0.0:
        s = float(np.sqrt(float(reg_lambda)))
        for cid, slot in cell_slot.items():
            if cid == anchor:
                continue
            for j in (1, 2):
                rows.append(r); cols.append(3 * slot + j); vals.append(s)
                rhs.append(0.0)
                r += 1
    a = csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(r, n_free),
    )
    return a, np.asarray(rhs, dtype=np.float64)


def _edge_norm_residuals(base, scales, w_edge, sol, anchor, cell_slot) -> np.ndarray:
    out = []
    for e, (o, r0, theta) in enumerate(base):
        if r0.shape[0] == 0:
            out.append(0.0)
            continue
        su = cell_slot.get(o.u)
        sv = cell_slot.get(o.v)
        p = np.zeros(3, dtype=np.float64)
        if sv is not None:
            p += sol[3 * sv : 3 * sv + 3]
        if su is not None:
            p -= sol[3 * su : 3 * su + 3]
        rr = r0 @ (p - theta)
        out.append(float(np.linalg.norm(rr) / np.sqrt(r0.shape[0])))
    return np.asarray(out, dtype=np.float64)


def _solve_lsqr(a: csr_matrix, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or a.shape[1] == 0:
        return np.zeros(a.shape[1], dtype=np.float64)
    res = lsqr(a, b, atol=_LSQR_ATOL, btol=_LSQR_BTOL, iter_lim=_LSQR_ITERS)
    return np.asarray(res[0], dtype=np.float64)


def _dense_rank_cond(a: csr_matrix) -> tuple[int | None, float | None]:
    if a.shape[1] == 0:
        return 0, float("inf")
    if a.shape[1] > _DENSE_DIAG_LIMIT:
        return None, None
    dense = a.toarray()
    if dense.size == 0 or dense.shape[0] == 0:
        return 0, float("inf")
    singular = np.linalg.svd(dense, compute_uv=False)
    smax = float(singular[0]) if singular.size else 0.0
    tol = max(dense.shape) * np.finfo(np.float64).eps * smax
    rank = int(np.count_nonzero(singular > tol))
    smin = float(singular[-1]) if singular.size else 0.0
    cond = (smax / smin) if smin > 0.0 else float("inf")
    return rank, cond


# ---------------------------------------------------------------------------
# Held-out metrics (RMS around ZERO; symmetric p90) — rework M1
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def heldout_metrics(
    observations: Sequence[EdgeObservation],
    coeffs_plan: np.ndarray,
    offsets_m0: np.ndarray,
    index: Mapping[str, int],
    channels: int,
) -> dict:
    """Held-out metrics for M0 and M1: RMS around zero (bias-preserving), symmetric
    p90 = ``percentile(abs(residual), 90)``. Same operator for both models.
    """
    r_plan_chunks: list[np.ndarray] = []
    r_m0_chunks: list[np.ndarray] = []
    per_edge: list[dict] = []
    by_orientation: dict[str, dict[str, list[np.ndarray]]] = {
        "h": {"plan": [], "m0": []}, "v": {"plan": [], "m0": []},
    }
    for obs in observations:
        xy = obs.holdout_xy
        if xy.shape[0] == 0:
            continue
        c = obs.channel
        p_u = coeffs_plan[index[obs.u], c, :]
        p_v = coeffs_plan[index[obs.v], c, :]
        phi = np.column_stack([np.ones(xy.shape[0]), xy[:, 0], xy[:, 1]])
        r_plan = obs.holdout_diff - phi @ (p_v - p_u)
        r_m0 = obs.holdout_diff - (offsets_m0[index[obs.v], c] - offsets_m0[index[obs.u], c])
        p90p = float(np.percentile(np.abs(r_plan), 90)) if r_plan.size else float("nan")
        p90m = float(np.percentile(np.abs(r_m0), 90)) if r_m0.size else float("nan")
        per_edge.append({
            "u": obs.u, "v": obs.v, "channel": c, "orientation": obs.orientation,
            "n_holdout": int(xy.shape[0]),
            "p90_plan": p90p, "p90_m0": p90m,
            "rms_plan": _rms(r_plan), "rms_m0": _rms(r_m0),
        })
        r_plan_chunks.append(r_plan)
        r_m0_chunks.append(r_m0)
        by_orientation[obs.orientation]["plan"].append(r_plan)
        by_orientation[obs.orientation]["m0"].append(r_m0)

    def _agg(ch):
        return _rms(np.concatenate(ch)) if ch else float("nan")

    r_plan = _agg(r_plan_chunks)
    r_m0 = _agg(r_m0_chunks)
    ratio = (r_plan / r_m0) if (np.isfinite(r_m0) and r_m0 > 0) else float("inf")
    orientation_ratio: dict[str, float] = {}
    for o, d in by_orientation.items():
        if not d["plan"] and not d["m0"]:
            continue  # orientation absent
        rp, rm = _agg(d["plan"]), _agg(d["m0"])
        orientation_ratio[o] = (rp / rm) if (np.isfinite(rm) and rm > 0) else float("inf")
    return {
        "r_plan": r_plan, "r_offset": r_m0, "ratio": ratio,
        "per_edge": per_edge, "orientation_ratio": orientation_ratio,
    }


# ---------------------------------------------------------------------------
# Model-mismatch detector (PROVISIONAL thresholds, contract §5 state only)
# ---------------------------------------------------------------------------

def detect_mismatch(observations, cfg: GateConfig, *, coeffs=None, index=None):
    usable = [o for o in observations if o.effective_train >= 64 and o.robust_sigma > 0.0]
    if not usable:
        return False, {"usable_edges": 0, "flagged_fraction": 0.0, "median_z": 0.0,
                       "cross_z": 0.0, "thresholds_provisional": True}
    z = np.array([max(o.level_z, o.curvature_z) for o in usable], dtype=np.float64)
    flagged = z >= cfg.mismatch_level_z
    frac = float(np.mean(flagged))
    med = float(np.median(z))
    inband = bool(frac >= cfg.mismatch_min_fraction or med >= cfg.mismatch_level_z)
    cross_z = 0.0
    cross = False
    if coeffs is not None and index is not None and len(usable) >= 4:
        lvl = np.array([o.level_mean for o in usable], dtype=np.float64)
        off_res = np.array([
            o.theta_global[0] - (coeffs[index[o.v], o.channel, 0] - coeffs[index[o.u], o.channel, 0])
            for o in usable
        ], dtype=np.float64)
        sig = float(np.median([o.robust_sigma for o in usable]))
        lvl_c = lvl - float(np.mean(lvl))
        denom = float(np.sum(lvl_c * lvl_c))
        spread = float(np.percentile(lvl, 95) - np.percentile(lvl, 5))
        if denom > 0.0 and sig > 0.0 and np.all(np.isfinite(off_res)):
            slope = float(np.sum(lvl_c * off_res) / denom)
            cross_z = abs(slope) * spread / sig
            cross = cross_z >= cfg.mismatch_level_z
    mismatch = bool(inband or cross)
    return mismatch, {
        "usable_edges": len(usable), "flagged_fraction": frac, "median_z": med,
        "max_z": float(np.max(z)), "inband": inband, "cross_z": cross_z, "cross": cross,
        "thresholds_provisional": True,
        "thresholds": {"level_z": cfg.mismatch_level_z, "min_fraction": cfg.mismatch_min_fraction},
    }


# ---------------------------------------------------------------------------
# Cycle residuals (rework M4)
# ---------------------------------------------------------------------------

def _cycle_residuals(
    observations: Sequence[EdgeObservation],
    cell_ids: Sequence[str],
    nx: int,
    ny: int,
    canvas_hw: tuple[int, int],
    channels: int,
) -> list[dict]:
    """Elementary-grid-cycle offset residuals per channel (observed data).

    For each 2x2 square with all four relations observed on a channel, evaluate
    each fitted edge plane at the cycle centre and sum with alternating signs; a
    consistent graph gives ~0, a contradictory relation leaves a large residual.
    """
    h, w = int(canvas_hw[0]), int(canvas_hw[1])
    cell_w, cell_h = w / float(nx), h / float(ny)
    have: dict[tuple[str, str, int], EdgeObservation] = {}
    for o in observations:
        have[(o.u, o.v, o.channel)] = o
        have[(o.v, o.u, o.channel)] = o

    def _val(o: EdgeObservation, sign: int, px: float, py: float) -> float | None:
        if o is None:
            return None
        x = 2.0 * px / w - 1.0
        y = 2.0 * py / h - 1.0
        t = o.theta_global
        return sign * float(t[0] + t[1] * x + t[2] * y)

    out = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            A = f"r{r:04d}c{c:04d}"
            B = f"r{r:04d}c{c + 1:04d}"
            D = f"r{r + 1:04d}c{c:04d}"
            E = f"r{r + 1:04d}c{c + 1:04d}"
            cx, cy = (c + 1) * cell_w, (r + 1) * cell_h
            for chn in range(channels):
                e_ab = have.get((A, B, chn))
                e_be = have.get((B, E, chn))
                e_de = have.get((D, E, chn))
                e_ad = have.get((A, D, chn))
                if not all(x is not None for x in (e_ab, e_be, e_de, e_ad)):
                    continue
                parts = [
                    _val(e_ab, +1, cx, cy), _val(e_be, +1, cx, cy),
                    _val(e_de, -1, cx, cy), _val(e_ad, -1, cx, cy),
                ]
                sumsq = 0.0
                for o in (e_ab, e_be, e_de, e_ad):
                    sumsq += float(o.robust_sigma) ** 2
                resid = float(sum(parts))
                denom = float(np.sqrt(sumsq)) if sumsq > 0 else float("nan")
                out.append({
                    "cycle": [A, B, E, D], "channel": chn,
                    "offset_residual": resid,
                    "sigma_scale": denom,
                    "z": (abs(resid) / denom) if (np.isfinite(denom) and denom > 0) else None,
                })
    return out


# ---------------------------------------------------------------------------
# Decision (contract §6)
# ---------------------------------------------------------------------------

def decide_state(*, connected: bool, mismatch: bool, gates: Mapping[str, str]) -> str:
    """Precedence: DISCONNECTED > INSUFFICIENT_EVIDENCE > MODEL_MISMATCH >
    OFFSET_ONLY (no held-out gain) > ILL_CONDITIONED > ROBUST_PLAN."""
    if not connected:
        return STATE_DISCONNECTED
    if gates.get("evidence") == "FAIL":
        return STATE_INSUFFICIENT_EVIDENCE
    if mismatch:
        return STATE_MODEL_MISMATCH
    if gates.get("heldout") == "FAIL":
        return STATE_OFFSET_ONLY
    if gates.get("numerics") == "FAIL":
        return STATE_ILL_CONDITIONED
    if gates and all(v == "PASS" for v in gates.values()):
        return STATE_ROBUST_PLAN
    return STATE_OFFSET_ONLY


def _corner_gate(corner_unc, coeffs, sigma_delta, cfg: GateConfig) -> str:
    if corner_unc is None:
        return "NOT_RUN"
    if not np.any(np.isfinite(corner_unc)):
        return "FAIL"
    for i in range(corner_unc.shape[0]):
        for c in range(corner_unc.shape[1]):
            corr = coeffs[i, c]
            if not np.all(np.isfinite(corr)):
                continue
            for ci, (cx, cy) in enumerate(_CORNER_XY):
                unc = corner_unc[i, c, ci]
                if not np.isfinite(unc):
                    return "FAIL"
                magnitude = abs(corr[0] + corr[1] * cx + corr[2] * cy)
                absolute_ok = np.isfinite(sigma_delta) and sigma_delta > 0 and unc <= cfg.corner_unc_sigma_frac * sigma_delta
                relative_ok = magnitude > 0.0 and unc <= cfg.corner_unc_correction_frac * magnitude
                if not (absolute_ok or relative_ok):
                    return "FAIL"
    return "PASS"


# ---------------------------------------------------------------------------
# Application (ADDITIVE-ONLY)
# ---------------------------------------------------------------------------

def plane_field(bounds, canvas_hw, coeffs) -> np.ndarray:
    xs, ys = normalized_coords(bounds, canvas_hw)
    a, b, d = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a + b * xs[None, :] + d * ys[:, None]


def apply_plane_correction(science, coeffs, bounds, canvas_hw) -> np.ndarray:
    """Add the per-channel plane (ADDITIVE-ONLY). Preserves NaN/dims/dtype; no gain."""
    a = np.asarray(science)
    dtype = a.dtype
    af = a.astype(np.float64)
    mono = af.ndim == 2
    af3 = af[:, :, None] if mono else af
    c = af3.shape[2]
    coeff_arr = np.asarray(coeffs, dtype=np.float64)
    if coeff_arr.ndim == 1:
        coeff_arr = np.repeat(coeff_arr[None, :], c, axis=0)
    if coeff_arr.shape != (c, 3):
        raise ValueError(f"coeffs shape {coeff_arr.shape} != {(c, 3)}")
    out = af3.copy()
    finite = np.isfinite(af3)
    for ch in range(c):
        field = plane_field(bounds, canvas_hw, coeff_arr[ch])
        out[:, :, ch] = np.where(finite[:, :, ch], af3[:, :, ch] + field, af3[:, :, ch])
    return (out[:, :, 0] if mono else out).astype(dtype)


# ---------------------------------------------------------------------------
# Nonlinear candidate — documentation only
# ---------------------------------------------------------------------------

def nonlinear_study() -> dict:
    return {
        "model": MODEL_NONLINEAR_CANDIDATE,
        "status": "research_only_no_engine",
        "references": [
            "PhotometricMosaic (John Murphy): photometric scale matching + surface spline",
            "Montage (Caltech/IPAC): planar offset/slope matching with calibration assumptions",
        ],
        "scientific_position": (
            "A per-Cell spline could absorb real sky/IFN and is only identifiable with "
            "independent calibration; it is NOT executable here. Nonlinear residual is "
            "reported as MODEL_MISMATCH needing research/upstream calibration, never "
            "repaired by an additive plane."
        ),
        "credits_proposed": [
            "PhotometricMosaic — John Murphy (methodological reference)",
            "Montage — Caltech/IPAC (methodological reference)",
        ],
    }


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------

@dataclass
class ModelComparisonResult:
    decision: str
    model: str
    cell_ids: tuple[str, ...]
    anchor: str
    channels: int
    offset_offsets: np.ndarray
    plan_coefficients: np.ndarray
    reg_lambda: float
    regularization_ladder: tuple[float, ...]
    heldout: dict
    gates: dict
    rank_data: int | None
    cond_data: float | None
    coef_uncertainty: np.ndarray | None
    corner_uncertainty: np.ndarray | None
    per_channel: dict
    cycles: list
    dropped: list
    edge_stats: list[dict]
    mismatch: dict
    sigma_delta: float
    disconnected: tuple[str, ...]
    truth_diagnostics: dict
    nonlinear_study: dict
    provenance: dict
    refusal_reasons: list[str] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def to_json_dict(self) -> dict:
        def _f(x):
            try:
                return None if not np.isfinite(x) else float(x)
            except TypeError:
                return x

        return {
            "decision": self.decision,
            "model": self.model,
            "anchor": self.anchor,
            "channels": self.channels,
            "cell_ids": list(self.cell_ids),
            "reg_lambda": _f(self.reg_lambda),
            "regularization_ladder": list(self.regularization_ladder),
            "rank_data": self.rank_data,
            "cond_data": None if self.cond_data is None else _f(self.cond_data),
            "sigma_delta": _f(self.sigma_delta),
            "heldout": {
                "r_plan": _f(self.heldout.get("r_plan")),
                "r_offset": _f(self.heldout.get("r_offset")),
                "ratio": _f(self.heldout.get("ratio")),
                "orientation_ratio": {k: _f(v) for k, v in self.heldout.get("orientation_ratio", {}).items()},
                "per_edge": self.heldout.get("per_edge", []),
            },
            "gates": dict(self.gates),
            "mismatch": self.mismatch,
            "per_channel": self.per_channel,
            "cycles": self.cycles,
            "dropped": self.dropped,
            "refusal_reasons": list(self.refusal_reasons),
            "disconnected": list(self.disconnected),
            "truth_diagnostics": self.truth_diagnostics,
            "offset_offsets": np.asarray(self.offset_offsets).tolist(),
            "plan_coefficients": _nan_list(self.plan_coefficients),
            "nonlinear_study": self.nonlinear_study,
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
        }


def _nan_list(a):
    return [[[None if not np.isfinite(x) else float(x) for x in row] for row in ch] for ch in np.asarray(a)]


def _edge_stats_json(o: EdgeObservation) -> dict:
    info = 0.5 * (o.information + o.information.T) if o.information.size else o.information
    reason = ""
    if o.band_clamped:
        reason = "band_clamped_to_overlap"
    return {
        "u": o.u, "v": o.v, "channel": o.channel, "orientation": o.orientation,
        "n_train": o.n_train, "n_holdout": o.n_holdout, "effective_train": o.effective_train,
        "tangent_blocks": o.tangent_blocks, "normal_blocks": o.normal_blocks,
        "robust_sigma": o.robust_sigma, "rejection_fraction": o.rejection_fraction,
        "theta_global": o.theta_global.tolist(),
        "rank_information": int(np.linalg.matrix_rank(o.information)) if o.information.size else 0,
        "leverage_max": (float(np.max(np.linalg.eigvalsh(info))) if info.size else 0.0),
        "level_z": o.level_z, "curvature_z": o.curvature_z,
        "measurement_px": o.measurement_px, "overlap_px": o.overlap_px,
        "band_used_px": o.band_used_px, "band_clamped": o.band_clamped,
        "refusal_reason": reason or None,
    }


def gauge_truth_error(truth_planes, coeffs, cell_ids, anchor, channels) -> dict:
    """Error of the recovered CORRECTION vs truth: expected = plane_anchor - plane_i."""
    index = {cid: i for i, cid in enumerate(cell_ids)}
    truth = np.zeros((len(cell_ids), channels, 3), dtype=np.float64)
    anchor_truth = np.zeros((channels, 3), dtype=np.float64)
    for cid, i in index.items():
        if cid not in truth_planes:
            continue
        t = np.asarray(truth_planes[cid], dtype=np.float64)
        if t.ndim == 1 and t.size == 3:
            t = np.repeat(t[None, :], channels, axis=0)
        truth[i] = t
        if cid == anchor:
            anchor_truth = t
    expected = anchor_truth[None, :, :] - truth
    err = np.abs(np.asarray(coeffs) - expected)
    finite = err[np.isfinite(err)]
    return {
        "max_abs_error": float(np.max(finite)) if finite.size else float("nan"),
        "median_abs_error": float(np.median(finite)) if finite.size else float("nan"),
        "n_nonfinite": int(err.size - finite.size),
    }


def compare_models(
    cells, nx, ny, canvas_hw, *, channels=None, anchor=None,
    measurement_px=DEFAULT_MEASUREMENT_PX, feather_px=DEFAULT_FEATHER_PX,
    block_px=BLOCK_PX, cfg=None, truth_planes=None,
) -> ModelComparisonResult:
    """Evaluate M0 vs M1 on one corpus; returns coefficients, gates and decision."""
    cfg = cfg or GateConfig()
    if channels is None:
        channels = cells[next(iter(cells))].channels()
    cell_ids = tuple(
        f"r{row:04d}c{col:04d}"
        for row in range(ny) for col in range(nx)
        if f"r{row:04d}c{col:04d}" in cells
    )
    if not cell_ids:
        raise ValueError("no cells provided")
    if anchor is None:
        anchor = select_central_cell(nx, ny)
        if anchor not in cells:
            anchor = cell_ids[0]
    index = {cid: i for i, cid in enumerate(cell_ids)}

    obs, dropped = build_observations(
        cells, nx, ny, canvas_hw, channels=channels,
        block_px=block_px, measurement_px=measurement_px,
    )
    chan_conn = _channel_connectivity(cell_ids, obs, anchor, channels)
    disconnected_channels = [c for c, d in chan_conn.items() if not d["connected"]]
    disconnected_cells = sorted({cid for c in disconnected_channels for cid in chan_conn[c]["missing"]})
    connected = not disconnected_channels

    refusal_reasons: list[str] = []
    for c in disconnected_channels:
        refusal_reasons.append(
            f"channel {c} disconnected: {len(chan_conn[c]['missing'])} cell(s) unreachable"
        )

    # M0 per channel (never raises).
    m0 = solve_offset_baseline(obs, cell_ids, anchor, channels,
                               channel_connected={c: d["connected"] for c, d in chan_conn.items()})

    # Ladder selection on train/holdout only.
    ladder_scores: dict[float, float] = {}
    for lam in REGULARIZATION_LADDER:
        sol = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=lam,
                                with_uncertainty=False,
                                channel_connected={c: d["connected"] for c, d in chan_conn.items()})
        ladder_scores[lam] = heldout_metrics(obs, sol.coefficients, m0, index, channels)["r_plan"]
    finite = {k: v for k, v in ladder_scores.items() if np.isfinite(v)}
    best_lambda = max(k for k, v in finite.items() if v <= min(finite.values()) * (1.0 + 1e-9)) if finite else 0.0

    plan = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=best_lambda,
                             with_uncertainty=True,
                             channel_connected={c: d["connected"] for c, d in chan_conn.items()})
    hm = heldout_metrics(obs, plan.coefficients, m0, index, channels)

    sigmas = [o.robust_sigma for o in obs if o.robust_sigma > 0]
    sigma_delta = float(np.median(sigmas)) if sigmas else 0.0
    noise_estimable = bool(np.isfinite(sigma_delta) and sigma_delta > 0)

    mismatch, mismatch_info = detect_mismatch(obs, cfg, coeffs=plan.coefficients, index=index)
    cycles = _cycle_residuals(obs, cell_ids, nx, ny, canvas_hw, channels)

    # §6.1 evidence gate (+ degenerate-noise branch, M6).
    eff_ok = bool(obs) and all(o.effective_train >= cfg.min_effective_train for o in obs)
    blocks_ok = bool(obs) and all(
        o.tangent_blocks >= cfg.min_tangent_blocks and o.normal_blocks >= cfg.min_normal_blocks
        for o in obs
    )
    if not connected:
        evidence = "FAIL"
    elif not noise_estimable:
        evidence = "FAIL"
        refusal_reasons.append("degenerate: noise/uncertainty scale not estimable (sigma_delta==0)")
    else:
        evidence = "PASS" if (eff_ok and blocks_ok) else "FAIL"

    # §6.1 numerics (worst channel; data-only uncertainty).
    finite_unc = True
    if plan.coef_uncertainty is not None and np.any(np.isfinite(plan.coef_uncertainty)):
        finite_unc = bool(np.all(np.isfinite(plan.coef_uncertainty)))
    rank_ok = plan.rank_data is not None and plan.rank_data == plan.unknowns
    cond_ok = plan.cond_data is not None and np.isfinite(plan.cond_data) and plan.cond_data <= cfg.cond_a_max
    corner = _corner_gate(plan.corner_uncertainty, plan.coefficients, sigma_delta, cfg)
    numerics = "PASS" if (rank_ok and cond_ok and finite_unc and corner == "PASS") else "FAIL"

    # §6.2 held-out.
    r_plan, r_off = hm["r_plan"], hm["r_offset"]
    ratio_ok = bool(np.isfinite(r_plan) and np.isfinite(r_off) and r_off > 0 and r_plan <= cfg.heldout_ratio_max * r_off)
    abs_ok = bool(np.isfinite(r_plan) and noise_estimable and r_plan <= cfg.heldout_abs_sigma_max * sigma_delta)
    orient_ok = all(
        np.isfinite(v) and v <= cfg.heldout_ratio_max
        for v in hm["orientation_ratio"].values()
    )
    p90_ok = all(
        (not np.isfinite(e["p90_plan"])) or (e["p90_plan"] <= (1.0 + cfg.per_edge_p90_worsen_max) * e["p90_m0"])
        for e in hm["per_edge"]
    )
    heldout = "PASS" if (ratio_ok and abs_ok and orient_ok and p90_ok) else "FAIL"

    gates = {
        "evidence": evidence,
        "numerics": numerics,
        "heldout": heldout,
        "connected": "PASS" if connected else "FAIL",
        "rank": "PASS" if rank_ok else ("NOT_RUN" if plan.rank_data is None else "FAIL"),
        "condition": "PASS" if cond_ok else ("NOT_RUN" if plan.cond_data is None else "FAIL"),
        "corner_uncertainty": corner,
    }

    truth_diag = {}
    if truth_planes is not None:
        truth_diag = gauge_truth_error(truth_planes, plan.coefficients, cell_ids, anchor, channels)

    decision = decide_state(connected=connected, mismatch=mismatch, gates=gates)
    if decision == STATE_ROBUST_PLAN and not (
        gates["heldout"] == "PASS" and gates["numerics"] == "PASS" and gates["evidence"] == "PASS"
    ):
        decision = STATE_OFFSET_ONLY

    provenance = dict(PROVENANCE)
    provenance.update({
        "measurement_px": measurement_px, "feather_px": feather_px, "block_px": block_px,
        "channels": channels, "n_cells": len(cell_ids), "n_edges_channel": len(obs),
        "anchor": anchor, "selected_reg_lambda": best_lambda,
        "ladder_scores": {str(k): (None if not np.isfinite(v) else float(v)) for k, v in ladder_scores.items()},
        "sigma_delta": None if not np.isfinite(sigma_delta) else float(sigma_delta),
        "noise_estimable": noise_estimable,
        "n_dropped_relations": len(dropped),
        "global_solve_densified": False,
        "diagnostics_densified": bool(plan.diagnostics.get("diagnostics_densified", False)),
        "diagnostics_dense_limit": _DENSE_DIAG_LIMIT,
    })

    return ModelComparisonResult(
        decision=decision,
        model=(MODEL_ROBUST_PLAN if decision == STATE_ROBUST_PLAN else MODEL_OFFSET_ONLY),
        cell_ids=cell_ids, anchor=anchor, channels=channels,
        offset_offsets=m0, plan_coefficients=plan.coefficients,
        reg_lambda=plan.reg_lambda, regularization_ladder=REGULARIZATION_LADDER,
        heldout=hm, gates=gates, rank_data=plan.rank_data, cond_data=plan.cond_data,
        coef_uncertainty=plan.coef_uncertainty, corner_uncertainty=plan.corner_uncertainty,
        per_channel=plan.per_channel, cycles=cycles, dropped=dropped,
        edge_stats=[_edge_stats_json(o) for o in obs], mismatch=mismatch_info,
        sigma_delta=sigma_delta, disconnected=tuple(disconnected_cells),
        truth_diagnostics=truth_diag, nonlinear_study=nonlinear_study(),
        provenance=provenance, refusal_reasons=refusal_reasons,
    )


# ---------------------------------------------------------------------------
# Scalability witness (contract §9)
# ---------------------------------------------------------------------------

def sparse_scale_witness(side: int, *, channels: int = 1, reg_lambda: float = 1e-3,
                         measure: bool = True) -> dict:
    """Compact sparse witness (no images). Reports REAL densification paths."""
    import resource
    import time

    nx = ny = int(side)
    cell_ids = tuple(f"r{r:04d}c{c:04d}" for r in range(ny) for c in range(nx))
    info = np.diag([16.0, 16.0, 4.0])
    obs: list[EdgeObservation] = []
    for (u, v) in grid_adjacency(nx, ny):
        (ur, uc), (vr, vc) = _parse_rc(u), _parse_rc(v)
        theta = np.array([
            0.05 * (((ur + uc) % 11) - 5),
            0.01 * (((ur * 7 + uc * 3) % 9) - 4),
            0.01 * (((ur * 5 + uc * 13) % 9) - 4),
        ])
        obs.append(EdgeObservation(
            u=u, v=v, channel=0, orientation=("h" if vc > uc else "v"),
            n_train=256, n_holdout=0, effective_train=256.0,
            tangent_blocks=8, normal_blocks=4, robust_sigma=1.0, rejection_fraction=0.0,
            theta_global=theta, information=info.copy(), rhs=info @ theta,
            level_z=0.0, curvature_z=0.0, level_mean=0.0, local_offset_train=float(theta[0]),
            measurement_px=None, overlap_px=8, band_used_px=8, band_clamped=False,
            holdout_xy=np.zeros((0, 2)), holdout_diff=np.zeros(0),
        ))
    anchor = select_central_cell(nx, ny)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if measure else 0
    t0 = time.perf_counter()
    plan = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=reg_lambda, with_uncertainty=False)
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if measure else 0
    diag = plan.diagnostics
    return {
        "cells": len(cell_ids), "edges": len(obs), "unknowns": plan.unknowns,
        "regularized_rows": plan.rows, "rank_data": plan.rank_data, "cond_data": plan.cond_data,
        "solve_seconds": dt, "rss_delta_kib": int(max(rss1 - rss0, 0)), "rss_peak_kib": int(rss1),
        "dense_design_bytes_estimate": int(plan.rows * plan.unknowns * 8) if plan.unknowns else 0,
        "method": "scipy.sparse.linalg.lsqr_on_whitened_design",
        "global_solve_densified": False,
        "diagnostics_densified": bool(diag.get("diagnostics_densified", False)),
        "dense_diag_limit": diag.get("dense_diag_limit", _DENSE_DIAG_LIMIT),
    }
