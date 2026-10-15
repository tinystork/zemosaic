"""ZM-ZEGRID-R37-B — inter-Cell ADDITIVE background model prototype (PURE, UNWIRED).

Isolated Gate-B prototype mandated by ``docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md``.
Never imported by production. Depends only on NumPy/SciPy and the PURE
``cell_offset`` module (reused UNCHANGED as the M0 baseline).

Two additive inter-Cell models on the SAME synthetic corpora:

* **M0 = ``OFFSET_ONLY``** — R34 scalar baseline via the unchanged
  ``cell_offset.solve_additive_offsets`` (per channel; NaN + ``applicable=False``
  for a refused channel — never fabricated zeros).
* **M1 = ``ROBUST_PLAN``** — per-Cell/channel additive plane
  ``P_i(x,y,c) = a_i,c + b_i,c*x + d_i,c*y`` (global normalized coords), solved with
  a sparse whitened LSQR, slopes-only ridge, deterministic information cap, bounded
  graph IRLS and bounded connectivity-safe relational rejection.

Discipline encoded here (frozen contract; rework-2 hardening)
-------------------------------------------------------------
* Measurement band applied CONSISTENTLY (H1); clamping reported.
* Per-channel connectivity/rank/condition/uncertainty; worst channel governs (H3).
* The **accepted** relation set is the source of truth for gates (N3/N11): the
  design is rebuilt after rejection, and rank/cond/covariance/rows/uncertainty/
  connectivity/heldout/evidence/occupancy/mismatch/cycles are all computed on the
  accepted set. Rejected/capped edge identities are persisted (JSON-safe).
* **No promotion without cycle redundancy** (N2): an acyclic accepted graph (e.g.
  1xN / Nx1) cannot qualify relational contradictions -> explicit refusal.
* Refused channels get NaN coefficients/offsets/uncertainty (N4/N5); JSON-safe
  serialisation turns non-finite into ``null``.
* ``rows`` = exact final assembled design rows (data + ridge) per channel (N1).
* Robust/rejection knobs are PROVISIONAL, research-only (M7/N8).

Rework map: r0 H1–H3/M1–M7/L1–L4; r1 N1–N11.
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
    "MODEL_OFFSET_ONLY", "MODEL_ROBUST_PLAN", "MODEL_NONLINEAR_CANDIDATE",
    "STATE_OFFSET_ONLY", "STATE_ROBUST_PLAN", "STATE_INSUFFICIENT_EVIDENCE",
    "STATE_ILL_CONDITIONED", "STATE_MODEL_MISMATCH", "STATE_DISCONNECTED",
    "REGULARIZATION_LADDER", "HUBER_DELTA_FACTOR", "EDGE_INFO_CAP_MULT",
    "MAX_IRLS", "EDGE_REJECT_Z", "MAX_REJECT_FRACTION", "REJECT_LOSS_FACTOR",
    "BLOCK_PX", "DEFAULT_MEASUREMENT_PX", "DEFAULT_FEATHER_PX",
    "GateConfig", "PlanCell", "EdgeObservation", "ModelComparisonResult",
    "normalized_coords", "observe_edge", "build_observations",
    "solve_offset_baseline", "solve_plan_sparse", "decide_state", "compare_models",
    "compare_models_from_observations",
    "gauge_truth_error", "apply_plane_correction", "plane_field",
    "sparse_scale_witness", "nonlinear_study",
]

# ---------------------------------------------------------------------------
# Vocabulary
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
# Numerical constants (PROVISIONAL robust knobs are labelled below)
# ---------------------------------------------------------------------------

HUBER_DELTA_FACTOR = 1.345
MAD_SCALE = 1.4826
BLOCK_PX = 4
DEFAULT_MEASUREMENT_PX = 16
DEFAULT_FEATHER_PX = 8
REGULARIZATION_LADDER: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
#: PROVISIONAL (research-only): per-edge information cap (multiple of median trace).
EDGE_INFO_CAP_MULT = 100.0
#: PROVISIONAL: bounded graph-level IRLS iterations.
MAX_IRLS = 8
#: PROVISIONAL: normalized-residual threshold for relational rejection.
EDGE_REJECT_Z = 8.0
#: PROVISIONAL: at most this fraction of a channel's relations may be rejected.
MAX_REJECT_FRACTION = 0.25
#: PROVISIONAL: a rejection must reduce the kept-edge robust loss by this factor.
REJECT_LOSS_FACTOR = 0.5

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
    "rework": "r2",
    "status": "pure_prototype_unwired",
    "coordinate_system": "global_canvas_normalized_[-1,1]",
    "model": "P_i(x,y,c) = a + b*x + d*y (additive only)",
    "robust_loss": "huber_irls_fixed_1.345_mad_sigma",
    "global_solve": "scipy.sparse.linalg.lsqr_on_whitened_design",
    "global_solve_densified": False,
    "regularization": "slopes_only_ladder_frozen",
    "split": "deterministic_spatial_75_25_block_parity",
    "per_channel_diagnostics": True,
    "accepted_set_is_source_of_truth": True,
    "no_promotion_without_cycle_redundancy": True,
    "no_cycle_redundancy_rule_status": "PROVISIONAL_RESEARCH_ONLY_NOT_FROZEN_CONTRACT",
    "cycle_redundancy_method": "deterministic_fundamental_cycle_basis_on_accepted_set",
    "cycle_redundancy_requirement": (
        "per admissible channel: evaluated accepted cycle residuals >= 1 and "
        "evaluated_cycles == cyclomatic_cycles; else explicit cycle_diagnostic_incomplete refusal"
    ),
    "json_non_finite_policy": "all non-finite scalars serialise to null (never 0)",
    "robust_knobs_status": "PROVISIONAL_RESEARCH_ONLY_NOT_FROZEN_CONTRACT",
    "robust_knobs": {
        "edge_information_cap": EDGE_INFO_CAP_MULT,
        "graph_irls_max_iter": MAX_IRLS,
        "edge_reject_z": EDGE_REJECT_Z,
        "max_reject_fraction": MAX_REJECT_FRACTION,
        "reject_acceptance_rule": f"best_loss < {REJECT_LOSS_FACTOR}*current_loss",
    },
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

    ``mismatch_level_z``/``mismatch_min_fraction`` are PROVISIONAL, not frozen
    contract §6; they drive the experimental ``MODEL_MISMATCH`` state only.
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
    mismatch_level_z: float = 3.0
    mismatch_min_fraction: float = 0.5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanCell:
    cell_id: str
    bounds: tuple[int, int, int, int]
    science: np.ndarray
    valid_mask: np.ndarray | None = None

    def channels(self) -> int:
        a = np.asarray(self.science)
        return 1 if a.ndim == 2 else int(a.shape[2])

    def as_3d(self) -> np.ndarray:
        a = np.asarray(self.science, dtype=np.float64)
        return a[:, :, None] if a.ndim == 2 else a


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


def normalized_coords(bounds, canvas_hw):
    y0, y1, x0, x1 = bounds
    h, w = int(canvas_hw[0]), int(canvas_hw[1])
    xs = 2.0 * (np.arange(x0, x1, dtype=np.float64) + 0.5) / float(w) - 1.0
    ys = 2.0 * (np.arange(y0, y1, dtype=np.float64) + 0.5) / float(h) - 1.0
    return xs, ys


# ---------------------------------------------------------------------------
# Robust Huber plane fit
# ---------------------------------------------------------------------------

def _robust_plane_fit(phi, target, *, huber_factor=HUBER_DELTA_FACTOR):
    n = target.size
    if n == 0:
        return np.zeros(phi.shape[1]), np.zeros(0), 0.0
    coef, *_ = np.linalg.lstsq(phi, target, rcond=None)
    w = np.ones(n, dtype=np.float64)
    for _ in range(_HUBER_MAX_ITER):
        resid = target - phi @ coef
        sigma = MAD_SCALE * float(np.median(np.abs(resid - np.median(resid))))
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        cut = huber_factor * sigma
        absr = np.abs(resid)
        w = np.where(absr <= cut, 1.0, cut / np.maximum(absr, 1e-300))
        sw = np.sqrt(w)
        coef_new, *_ = np.linalg.lstsq(phi * sw[:, None], target * sw, rcond=None)
        if float(np.max(np.abs(coef_new - coef))) <= _HUBER_TOL * max(1.0, float(np.max(np.abs(coef)))):
            coef = coef_new
            break
        coef = coef_new
    resid = target - phi @ coef
    sigma = MAD_SCALE * float(np.median(np.abs(resid - np.median(resid))))
    return coef, w, sigma


@dataclass(frozen=True)
class EdgeObservation:
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
    theta_global: np.ndarray
    information: np.ndarray
    rhs: np.ndarray
    level_z: float
    curvature_z: float
    level_mean: float
    local_offset_train: float
    measurement_px: int | None
    overlap_px: int
    band_used_px: int
    band_clamped: bool
    holdout_xy: np.ndarray
    holdout_diff: np.ndarray

    def key(self) -> tuple:
        return (self.u, self.v, self.channel)


def observe_edge(cell_u, cell_v, canvas_hw, channel, *, block_px=BLOCK_PX,
                 measurement_px=None, normal_boundary=None, huber_factor=HUBER_DELTA_FACTOR):
    """Pure same-global-pixel observation. Returns ``(obs_or_None, refusal_reason)``."""
    uy0, uy1, ux0, ux1 = cell_u.bounds
    vy0, vy1, vx0, vx1 = cell_v.bounds
    y0, y1 = max(uy0, vy0), min(uy1, vy1)
    x0, x1 = max(ux0, vx0), min(ux1, vx1)
    if y1 <= y0 or x1 <= x0:
        return None, "no_overlap"

    su = cell_u.as_3d()[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0, channel]
    sv = cell_v.as_3d()[y0 - vy0:y1 - vy0, x0 - vx0:x1 - vx0, channel]
    vu = _valid_2d(cell_u)[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0]
    vv = _valid_2d(cell_v)[y0 - vy0:y1 - vy0, x0 - vx0:x1 - vx0]
    hh, ww = su.shape

    is_horizontal = abs(vx0 - ux0) >= abs(vy0 - uy0)
    orientation = "h" if is_horizontal else "v"
    overlap_px = int((x1 - x0) if is_horizontal else (y1 - y0))

    gx = np.arange(x0, x1, dtype=np.float64)[None, :].repeat(hh, 0) + 0.5
    gy = np.arange(y0, y1, dtype=np.float64)[:, None].repeat(ww, 1) + 0.5
    shared = np.isfinite(su) & np.isfinite(sv) & vu & vv

    band_clamped = False
    if measurement_px is not None and normal_boundary is not None:
        half = 0.5 * float(measurement_px)
        normal_val = gx if is_horizontal else gy
        shared = shared & (np.abs(normal_val - float(normal_boundary)) <= half)
        band_clamped = float(measurement_px) > float(overlap_px) + 1e-9

    if not np.any(shared):
        return None, ("band_no_support" if measurement_px is not None else "no_shared_valid_pixels")

    gy_l, gx_l = np.nonzero(shared)
    gxp = gx_l + x0
    gyp = gy_l + y0
    xs, ys = normalized_coords((y0, y1, x0, x1), canvas_hw)
    X, Y = xs[gx_l], ys[gy_l]
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

    def _slope_z(reg, raw):
        if sigma <= 0.0:
            return 0.0
        denom = float(np.sum(w * reg * reg))
        if denom <= 0.0:
            return 0.0
        slope = float(np.sum(w * reg * resid) / denom)
        return abs(slope) * float(np.percentile(raw, 95) - np.percentile(raw, 5)) / sigma

    rejection_fraction = float(np.mean(np.abs(resid) > huber_factor * sigma)) if sigma > 0 else 0.0
    tangent_blocks = int(np.unique((gyp if is_horizontal else gxp) // int(block_px)).size)
    normal_blocks = int(np.unique((bx if is_horizontal else by)).size)

    return EdgeObservation(
        u=cell_u.cell_id, v=cell_v.cell_id, channel=int(channel), orientation=orientation,
        n_train=n_train, n_holdout=int(np.count_nonzero(holdout)),
        effective_train=float(np.sum(w)), tangent_blocks=tangent_blocks,
        normal_blocks=normal_blocks, robust_sigma=float(sigma),
        rejection_fraction=rejection_fraction, theta_global=theta_g,
        information=info_g, rhs=rhs_g, level_z=_slope_z(level_c, level),
        curvature_z=_slope_z(radius_c, radius), level_mean=float(np.mean(level)),
        local_offset_train=float(np.median(diff[train])),
        measurement_px=(None if measurement_px is None else int(measurement_px)),
        overlap_px=overlap_px, band_used_px=band_used_px, band_clamped=bool(band_clamped),
        holdout_xy=np.column_stack([X[holdout], Y[holdout]]), holdout_diff=diff[holdout],
    ), ""


def _parse_rc(cell_id):
    return int(cell_id[1:5]), int(cell_id[6:10])


def relation_normal_boundary(u, v, nx, ny, canvas_hw) -> float:
    """Frozen relation normal boundary in GLOBAL PIXEL units (§3.2/§1.3).

    ``b = (col(u)+1) * W/nx`` for an ``h`` relation and ``(row(u)+1) * H/ny`` for
    a ``v`` relation, with ``(W, H) = (canvas_hw[1], canvas_hw[0])``. This is the
    EXACT arithmetic the accepted ``build_observations`` has always used, factored
    out so no other caller has to duplicate (and drift from) it.
    """
    w, h = int(canvas_hw[1]), int(canvas_hw[0])
    cell_w, cell_h = w / float(nx), h / float(ny)
    (ur, uc), (vr, vc) = _parse_rc(u), _parse_rc(v)
    return float((ur + 1) * cell_h) if vr > ur else float((uc + 1) * cell_w)


def build_observations(cells, nx, ny, canvas_hw, *, channels=None,
                       block_px=BLOCK_PX, measurement_px=DEFAULT_MEASUREMENT_PX):
    """Returns ``(observations, dropped)``; ``dropped`` is an explicit inventory."""
    if channels is None:
        channels = cells[next(iter(cells))].channels()
    out: list[EdgeObservation] = []
    dropped: list[dict] = []
    for (u, v) in grid_adjacency(nx, ny):
        if u not in cells or v not in cells:
            dropped.append({"u": u, "v": v, "channel": None, "reason": "cell_absent"})
            continue
        cu, cv = cells[u], cells[v]
        boundary = relation_normal_boundary(u, v, nx, ny, canvas_hw)
        for c in range(channels):
            obs, reason = observe_edge(cu, cv, canvas_hw, c, block_px=block_px,
                                       measurement_px=measurement_px, normal_boundary=boundary)
            if obs is None:
                dropped.append({"u": u, "v": v, "channel": c, "reason": reason})
            else:
                out.append(obs)
    return out, dropped


# ---------------------------------------------------------------------------
# Per-channel graph helpers
# ---------------------------------------------------------------------------

def _reachable_from(anchor, nodes, edges):
    adj = {n: set() for n in nodes}
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


def _independent_cycles(nodes, edges, anchor):
    """Cyclomatic number E - V + C over the (accepted) sub-graph."""
    present = {n for e in edges for n in e}
    if not present:
        return 0
    adj = {n: set() for n in present}
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    comps = 0
    unseen = set(present)
    while unseen:
        start = next(iter(unseen))
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps += 1
        unseen -= seen
    return max(len(edges) - len(present) + comps, 0)


def _channel_connectivity(cell_ids, observations, anchor, channels):
    out = {}
    for c in range(channels):
        edges = [(o.u, o.v) for o in observations if o.channel == c]
        reach = _reachable_from(anchor, cell_ids, edges) if edges else {anchor}
        missing = sorted(set(cell_ids) - reach)
        out[c] = {"edges": len(edges), "connected": len(missing) == 0, "missing": missing}
    return out


# ---------------------------------------------------------------------------
# M0 baseline (unchanged R34c solver; per channel; NaN for refused channels)
# ---------------------------------------------------------------------------

def solve_offset_baseline(observations, cell_ids, anchor, channels, *,
                          channel_connected=None):
    """M0 = ``OFFSET_ONLY`` per channel. Refused channels -> NaN (never zeros)."""
    index = {cid: i for i, cid in enumerate(cell_ids)}
    n = len(cell_ids)
    offsets = np.zeros((n, channels), dtype=np.float64)
    for c in range(channels):
        connected = True if channel_connected is None else bool(channel_connected.get(c, True))
        obs_c = [o for o in observations if o.channel == c]
        if (not connected) or (not obs_c):
            offsets[:, c] = np.nan
            continue
        edges = [
            OffsetEdge(o.u, o.v, np.asarray([o.local_offset_train], dtype=np.float64),
                       np.asarray([max(o.n_train + o.n_holdout, 1)], dtype=np.float64))
            for o in obs_c
        ]
        try:
            sol = solve_additive_offsets(list(cell_ids), edges, anchor, 1,
                                         raise_on_disconnected=False)
            col = np.asarray(sol.offsets, dtype=np.float64).ravel()
            offsets[:, c] = col
        except DisconnectedOffsetGraphError:
            offsets[:, c] = np.nan
    return offsets


# ---------------------------------------------------------------------------
# Global sparse plan solve
# ---------------------------------------------------------------------------

@dataclass
class PlanSolve:
    coefficients: np.ndarray
    reg_lambda: float
    rank_data: int | None
    cond_data: float | None
    unknowns: int
    rows: int
    coef_uncertainty: np.ndarray | None
    corner_uncertainty: np.ndarray | None
    coef_uncertainty_regularized: np.ndarray | None
    corner_uncertainty_regularized: np.ndarray | None
    per_channel: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def _edge_whitening(info):
    info = np.asarray(info, dtype=np.float64)
    info = 0.5 * (info + info.T)
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
    return np.vstack([evecs[:, k] * np.sqrt(float(evals[k])) for k in keep]), tr


def _data_only_covariance(a_data, limit):
    """Data-only covariance via dense SVD of the design. Null directions -> inf.

    Returns ``(cov, densified)``; ``None`` above the column limit.
    """
    if a_data.shape[1] == 0 or a_data.shape[0] == 0 or a_data.shape[1] > limit:
        return None, False
    dense = a_data.toarray().astype(np.float64)
    u, s, vt = np.linalg.svd(dense, full_matrices=False)
    if s.size == 0:
        return None, True
    tol = max(dense.shape) * np.finfo(np.float64).eps * float(s[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        # Null singular directions are NON-identifiable: their variance is +inf/
        # undefined, never a confident zero.
        inv = np.where(s > tol, 1.0 / (s * s), np.nan)
        cov = (vt.T * inv) @ vt
    return cov, True


def _corner_from_cov(cov, cell_slot, anchor):
    n = max(len(cell_slot), 1)
    coef_sigma = np.full((n, 3), np.inf, dtype=np.float64)
    corner_sigma = np.full((n, 4), np.inf, dtype=np.float64)
    for cid, slot in cell_slot.items():
        if cid == anchor:
            coef_sigma[slot, :] = 0.0
            corner_sigma[slot, :] = 0.0
            continue
        base = 3 * slot
        block = cov[base:base + 3, base:base + 3]
        coef_sigma[slot] = np.sqrt(np.maximum(np.diag(block), 0.0))
        for ci, (cx, cy) in enumerate(_CORNER_XY):
            phi = np.array([1.0, cx, cy])
            corner_sigma[slot, ci] = np.sqrt(max(float(phi @ block @ phi), 0.0))
    return coef_sigma, corner_sigma


def _dense_rank_cond(a):
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
    return rank, (smax / smin) if smin > 0.0 else float("inf")


def _assemble(base, scales, w_edge, anchor, cell_slot, *, reg_lambda):
    rows, cols, vals, rhs = [], [], [], []
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
    n_data_rows = r
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
    return a, np.asarray(rhs, dtype=np.float64), n_data_rows


def _edge_norm_residuals(base, scales, w_edge, sol, anchor, cell_slot):
    out = []
    for e, (o, r0, theta) in enumerate(base):
        if r0.shape[0] == 0:
            out.append(0.0)
            continue
        su = cell_slot.get(o.u)
        sv = cell_slot.get(o.v)
        p = np.zeros(3, dtype=np.float64)
        if sv is not None:
            p += sol[3 * sv:3 * sv + 3]
        if su is not None:
            p -= sol[3 * su:3 * su + 3]
        out.append(float(np.linalg.norm(r0 @ (p - theta)) / np.sqrt(r0.shape[0])))
    return np.asarray(out, dtype=np.float64)


def _mad(x):
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return MAD_SCALE * float(np.median(np.abs(x - med)))


def _solve_lsqr(a, b):
    if a.shape[0] == 0 or a.shape[1] == 0:
        return np.zeros(a.shape[1], dtype=np.float64)
    res = lsqr(a, b, atol=_LSQR_ATOL, btol=_LSQR_BTOL, iter_lim=_LSQR_ITERS)
    return np.asarray(res[0], dtype=np.float64)


def solve_plan_sparse(observations, cell_ids, anchor, channels, *, reg_lambda=0.0,
                      with_uncertainty=True, channel_connected=None):
    """Global M1 solve. The ACCEPTED relation set is the source of truth.

    Per channel: cap + bounded IRLS + bounded connectivity-safe rejection, then a
    FINAL assembly from which rank/cond/covariance/rows/connectivity/uncertainty are
    recomputed. Rejected/capped identities are persisted. Under-determined channels
    get NaN coefficients/uncertainty.
    """
    if anchor not in cell_ids:
        raise ValueError(f"anchor {anchor!r} not among cells")
    index = {cid: i for i, cid in enumerate(cell_ids)}
    cell_slot = {cid: i for i, cid in enumerate(c for c in cell_ids if c != anchor)}
    n = len(cell_ids)
    coeffs = np.zeros((n, channels, 3), dtype=np.float64)
    unknowns = 3 * max(n - 1, 0)
    n_free = 3 * len(cell_slot)

    coef_unc = coef_unc_reg = corner_unc = corner_unc_reg = None
    per_channel: dict = {}
    totals = {"rows": 0, "diagnostics_densified": False, "covariance_densified": False}
    rank_worst = None
    cond_worst = None

    for c in range(channels):
        obs_c = [o for o in observations if o.channel == c]
        connected_in = True if channel_connected is None else bool(channel_connected.get(c, True))

        base, traces = [], []
        for o in obs_c:
            r, tr = _edge_whitening(o.information)
            base.append((o, r, np.asarray(o.theta_global, dtype=np.float64)))
            traces.append(tr)
        med_tr = float(np.median(traces)) if traces else 0.0
        cap_tr = EDGE_INFO_CAP_MULT * med_tr if med_tr > 0 else 0.0
        scales, capped_idx = [], []
        for e, tr in enumerate(traces):
            if cap_tr > 0.0 and tr > cap_tr:
                scales.append(cap_tr / tr)
                capped_idx.append(e)
            else:
                scales.append(1.0)

        w_edge = np.ones(len(base), dtype=np.float64)
        sol = np.zeros(n_free, dtype=np.float64)
        irls_used = 0
        for it in range(MAX_IRLS):
            # Robust weighting is determined on the DATA-only system so that the
            # accepted set (and its diagnostics) never depends on the ridge λ.
            a, b, _nd = _assemble(base, scales, w_edge, anchor, cell_slot, reg_lambda=0.0)
            sol = _solve_lsqr(a, b)
            irls_used = it + 1
            t = _edge_norm_residuals(base, scales, w_edge, sol, anchor, cell_slot)
            if t.size < 4:
                break
            sig = _mad(t)
            if not (np.isfinite(sig) and sig > 1e-12):
                break
            cut = HUBER_DELTA_FACTOR * sig
            w_new = np.where(t <= cut, 1.0, cut / np.maximum(t, 1e-300))
            converged = float(np.max(np.abs(w_new - w_edge))) <= 1e-6
            w_edge = w_new
            if converged:
                break

        rejected_idx: list[int] = []
        max_reject = max(1, int(len(base) * MAX_REJECT_FRACTION))
        for _ in range(max_reject):
            t = _edge_norm_residuals(base, scales, w_edge, sol, anchor, cell_slot)
            if t.size == 0:  # N6 guard
                break
            sig = _mad(t)
            if not (np.isfinite(sig) and sig > 0.0):
                break
            order = [int(e) for e in np.argsort(-t) if t[e] > EDGE_REJECT_Z * sig]
            if not order:
                break
            cur_loss = float(np.sum(t[w_edge > 0.0] ** 2))
            best = best_loss = best_sol = best_w = None
            for e in order:
                w2 = w_edge.copy()
                w2[e] = 0.0
                remaining = [(base[k][0].u, base[k][0].v) for k in range(len(base)) if w2[k] > 0.0]
                reach = _reachable_from(anchor, cell_ids, remaining) if remaining else {anchor}
                if len(reach) != len(set(cell_ids)):
                    continue
                a2, b2, _nd = _assemble(base, scales, w2, anchor, cell_slot, reg_lambda=0.0)
                s2 = _solve_lsqr(a2, b2)
                t2 = _edge_norm_residuals(base, scales, w2, s2, anchor, cell_slot)
                loss = float(np.sum(t2[w2 > 0.0] ** 2))
                if best_loss is None or loss < best_loss:
                    best, best_loss, best_sol, best_w = e, loss, s2, w2
            if best is None or best_loss is None or best_loss >= REJECT_LOSS_FACTOR * cur_loss:
                break
            w_edge, sol = best_w, best_sol
            rejected_idx.append(best)

        # FINAL assembly (accepted set) — the source of truth.
        a_final, b_final, n_data_rows = _assemble(base, scales, w_edge, anchor, cell_slot,
                                                  reg_lambda=reg_lambda)
        sol = _solve_lsqr(a_final, b_final)
        a_data, _bd, _nd2 = _assemble(base, scales, w_edge, anchor, cell_slot, reg_lambda=0.0)
        rank_c, cond_c = _dense_rank_cond(a_data)
        cov_data, cov_dens = (None, False)
        if with_uncertainty:
            cov_data, cov_dens = _data_only_covariance(a_data, _DENSE_DIAG_LIMIT)

        accepted = [(base[e][0].u, base[e][0].v) for e in range(len(base)) if w_edge[e] > 0.0]
        reach = _reachable_from(anchor, cell_ids, accepted) if accepted else {anchor}
        connected_acc = len(reach) == len(set(cell_ids))
        cycles_acc = _independent_cycles(cell_ids, accepted, anchor)
        under = (not connected_in) or (not accepted) or (not connected_acc) or (
            rank_c is not None and rank_c < unknowns and accepted
        )

        if under:
            coeffs[:, c, :] = np.nan
        else:
            for cid, slot in cell_slot.items():
                coeffs[index[cid], c, :] = sol[3 * slot:3 * slot + 3]

        # Uncertainty arrays are allocated for EVERY channel (NaN when refused),
        # so a refused channel never looks confidently identified.
        if with_uncertainty and n_free > 0:
            if coef_unc is None:
                coef_unc = np.full((n, channels, 3), np.nan)
                corner_unc = np.full((n, channels, 4), np.nan)
            coef_unc[index[anchor], c, :] = 0.0
            corner_unc[index[anchor], c, :] = 0.0
            if cov_data is not None and not under and np.all(np.isfinite(cov_data)):
                cs, cors = _corner_from_cov(cov_data, cell_slot, anchor)
                for cid, slot in cell_slot.items():
                    coef_unc[index[cid], c, :] = np.where(np.isfinite(cs[slot]), cs[slot], np.nan)
                    corner_unc[index[cid], c, :] = np.where(np.isfinite(cors[slot]), cors[slot], np.nan)

        if with_uncertainty and reg_lambda > 0 and n_free <= _DENSE_DIAG_LIMIT and n_free > 0 and not under:
            a_reg, _br, _ndr = _assemble(base, scales, w_edge, anchor, cell_slot, reg_lambda=reg_lambda)
            cov_reg, _ = _data_only_covariance(a_reg, _DENSE_DIAG_LIMIT)
            if cov_reg is not None:
                cs_r, cors_r = _corner_from_cov(cov_reg, cell_slot, anchor)
                if coef_unc_reg is None:
                    coef_unc_reg = np.full((n, channels, 3), np.nan)
                    corner_unc_reg = np.full((n, channels, 4), np.nan)
                for cid, slot in cell_slot.items():
                    coef_unc_reg[index[cid], c, :] = np.where(np.isfinite(cs_r[slot]), cs_r[slot], np.nan)
                    corner_unc_reg[index[cid], c, :] = np.where(np.isfinite(cors_r[slot]), cors_r[slot], np.nan)
                coef_unc_reg[index[anchor], c, :] = 0.0
                corner_unc_reg[index[anchor], c, :] = 0.0

        def _ident(e, reason, status):
            o = base[e][0]
            return {
                "u": o.u, "v": o.v, "channel": o.channel, "reason": reason, "status": status,
                "initial_scale": float(scales[e]), "final_weight": float(w_edge[e]),
            }

        totals["rows"] += int(a_final.shape[0])
        totals["diagnostics_densified"] = totals["diagnostics_densified"] or (
            a_data.shape[1] <= _DENSE_DIAG_LIMIT and a_data.shape[1] > 0 and a_data.shape[0] > 0
        )
        totals["covariance_densified"] = totals["covariance_densified"] or cov_dens
        if rank_c is not None and (rank_worst is None or rank_c < rank_worst):
            rank_worst = rank_c
        if cond_c is not None and (cond_worst is None or cond_c > cond_worst):
            cond_worst = cond_c

        per_channel[int(c)] = {
            "connected": bool(connected_in and connected_acc),
            "observations": len(obs_c),
            "accepted": len(accepted),
            "rejected": len(rejected_idx),
            "capped": len(capped_idx),
            "rank": rank_c,
            "cond": None if cond_c is None else float(cond_c),
            "under_determined": bool(under),
            "rows": int(a_final.shape[0]),
            "data_rows": int(n_data_rows),
            "cycles": int(cycles_acc),
            "irls_iterations": int(irls_used),
            "covariance_densified": bool(cov_data is not None),
            "accepted_keys": [[o[0], o[1]] for o in accepted],
            "rejected_edges": [_ident(e, "relational_rejection", "rejected") for e in rejected_idx],
            "capped_edges": [_ident(e, "information_cap", "capped") for e in capped_idx],
        }

    return PlanSolve(
        coefficients=coeffs, reg_lambda=float(reg_lambda), rank_data=rank_worst,
        cond_data=cond_worst, unknowns=unknowns, rows=int(totals["rows"]),
        coef_uncertainty=coef_unc, corner_uncertainty=corner_unc,
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


def _accepted_key_set(plan: PlanSolve) -> set:
    acc = set()
    for c, d in plan.per_channel.items():
        for (u, v) in d["accepted_keys"]:
            acc.add((u, v, c))
    return acc


# ---------------------------------------------------------------------------
# Held-out metrics (RMS around zero; symmetric p90)
# ---------------------------------------------------------------------------

def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def heldout_metrics(observations, coeffs_plan, offsets_m0, index, channels):
    r_plan_chunks, r_m0_chunks, per_edge = [], [], []
    by_orientation = {"h": {"plan": [], "m0": []}, "v": {"plan": [], "m0": []}}
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
        per_edge.append({
            "u": obs.u, "v": obs.v, "channel": c, "orientation": obs.orientation,
            "n_holdout": int(xy.shape[0]),
            "p90_plan": float(np.percentile(np.abs(r_plan), 90)) if r_plan.size else float("nan"),
            "p90_m0": float(np.percentile(np.abs(r_m0), 90)) if r_m0.size else float("nan"),
            "rms_plan": _rms(r_plan), "rms_m0": _rms(r_m0),
        })
        r_plan_chunks.append(r_plan)
        r_m0_chunks.append(r_m0)
        by_orientation[obs.orientation]["plan"].append(r_plan)
        by_orientation[obs.orientation]["m0"].append(r_m0)

    def _agg(ch):
        return _rms(np.concatenate(ch)) if ch else float("nan")

    r_plan, r_m0 = _agg(r_plan_chunks), _agg(r_m0_chunks)
    ratio = (r_plan / r_m0) if (np.isfinite(r_m0) and r_m0 > 0) else float("inf")
    orientation_ratio = {}
    for o, d in by_orientation.items():
        if not d["plan"] and not d["m0"]:
            continue
        rp, rm = _agg(d["plan"]), _agg(d["m0"])
        orientation_ratio[o] = (rp / rm) if (np.isfinite(rm) and rm > 0) else float("inf")
    return {"r_plan": r_plan, "r_offset": r_m0, "ratio": ratio,
            "per_edge": per_edge, "orientation_ratio": orientation_ratio}


def _orientation_ok(orientation_ratio, max_ratio):
    """A present orientation with a non-finite ratio is a FAIL (contract §6.2.4)."""
    return all(np.isfinite(v) and v <= max_ratio for v in orientation_ratio.values())


def _p90_ok(per_edge, worsen_max):
    """Relative-only p90 gate; non-finite p90 on an included edge is a FAIL."""
    for e in per_edge:
        pp, pm = e["p90_plan"], e["p90_m0"]
        if not np.isfinite(pp) or not np.isfinite(pm):
            return False
        if pp > (1.0 + worsen_max) * pm:
            return False
    return True


# ---------------------------------------------------------------------------
# Mismatch detector (PROVISIONAL)
# ---------------------------------------------------------------------------

def detect_mismatch(observations, cfg, *, coeffs=None, index=None):
    usable = [o for o in observations if o.effective_train >= 64 and o.robust_sigma > 0.0]
    if not usable:
        return False, {"usable_edges": 0, "flagged_fraction": 0.0, "median_z": 0.0,
                       "cross_z": 0.0, "thresholds_provisional": True}
    z = np.array([max(o.level_z, o.curvature_z) for o in usable], dtype=np.float64)
    frac = float(np.mean(z >= cfg.mismatch_level_z))
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
    return bool(inband or cross), {
        "usable_edges": len(usable), "flagged_fraction": frac, "median_z": med,
        "max_z": float(np.max(z)), "inband": inband, "cross_z": cross_z, "cross": cross,
        "thresholds_provisional": True,
        "thresholds": {"level_z": cfg.mismatch_level_z, "min_fraction": cfg.mismatch_min_fraction},
    }


# ---------------------------------------------------------------------------
# Cycle residuals
# ---------------------------------------------------------------------------

def _tree_path(adj, start, goal):
    """Deterministic BFS path (list of nodes) in the spanning tree."""
    prev = {start: None}
    queue = [start]
    while queue:
        cur = queue.pop(0)
        if cur == goal:
            break
        for nb in sorted(adj.get(cur, [])):
            if nb not in prev:
                prev[nb] = cur
                queue.append(nb)
    if goal not in prev:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def _fundamental_cycles(nodes, edges):
    """Deterministic fundamental cycle basis over a simple graph.

    A lexical spanning forest is grown (union-find, edges processed in lexical
    order); each non-tree edge closes exactly one fundamental cycle. The number of
    cycles returned equals the cyclomatic number ``E - V + C``.

    Returns a list of cycles, each a list of directed steps ``(a, b)`` meaning the
    traversal goes from ``a`` to ``b``.
    """
    nodes = list(nodes)
    canonical = sorted({(u, v) if u <= v else (v, u) for (u, v) in edges})
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree_adj: dict = {n: [] for n in nodes}
    cycles = []
    for (u, v) in canonical:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree_adj.setdefault(u, []).append(v)
            tree_adj.setdefault(v, []).append(u)
        else:
            path = _tree_path(tree_adj, u, v)
            steps = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            steps.append((v, u))
            cycles.append(steps)
    return cycles


def _edge_signed_theta(obs_by_key, a, b, channel):
    """Return ``(theta, sign)`` for a directed step a->b (p_b - p_a)."""
    o = obs_by_key.get((a, b, channel))
    if o is not None:
        return np.asarray(o.theta_global, dtype=np.float64), +1, o
    o = obs_by_key.get((b, a, channel))
    if o is not None:
        return np.asarray(o.theta_global, dtype=np.float64), -1, o
    return None, 0, None


def _cycle_residuals(observations, cell_ids, nx, ny, canvas_hw, channels):
    """Cycle residuals on a DETERMINISTIC FUNDAMENTAL CYCLE BASIS (accepted set).

    For every fundamental cycle of each channel's accepted relation graph, traverse
    with native ``u->v`` signs and sum the edge coefficient vectors
    ``theta_global = [a, b, d]``: ``coefficient_residual[3]``. The plane residual is
    evaluated at the deterministic cycle centroid (mean of the member-Cell centres,
    global normalized coords) -> ``offset_residual``. ``sigma_scale`` is
    ``sqrt(sum edge robust_sigma^2)``; ``z = |offset_residual| / sigma_scale``.

    This covers elementary 2x2 squares AND large cycles (e.g. a ladder outer ring).
    """
    h, w = int(canvas_hw[0]), int(canvas_hw[1])
    cell_w, cell_h = w / float(nx), h / float(ny)
    obs_by_key = {}
    for o in observations:
        obs_by_key[(o.u, o.v, o.channel)] = o

    def centre(cid):
        r, c = _parse_rc(cid)
        return 2.0 * (c + 0.5) * cell_w / w - 1.0, 2.0 * (r + 0.5) * cell_h / h - 1.0

    out = []
    cell_set = set(cell_ids)
    for chn in range(channels):
        edges = [(o.u, o.v) for o in observations if o.channel == chn]
        present = sorted({n for e in edges for n in e})
        if not present:
            continue
        for steps in _fundamental_cycles(present, edges):
            coeff_res = np.zeros(3, dtype=np.float64)
            sumsq = 0.0
            nodes = []
            edges_meta = []
            ok = True
            for (a, b) in steps:
                theta, sign, o = _edge_signed_theta(obs_by_key, a, b, chn)
                if theta is None or o is None:
                    ok = False
                    break
                coeff_res = coeff_res + sign * theta
                sumsq += float(o.robust_sigma) ** 2
                edges_meta.append([a, b, int(sign)])
                if a not in nodes:
                    nodes.append(a)
                if b not in nodes:
                    nodes.append(b)
            if not ok:
                continue
            pts = [centre(n) for n in nodes if n in cell_set]
            xc = float(np.mean([p[0] for p in pts]))
            yc = float(np.mean([p[1] for p in pts]))
            offset_residual = float(coeff_res[0] + coeff_res[1] * xc + coeff_res[2] * yc)
            denom = float(np.sqrt(sumsq)) if sumsq > 0 else float("nan")
            out.append({
                "cycle": nodes, "nodes": nodes, "channel": chn, "edges": edges_meta,
                "coefficient_residual": coeff_res.tolist(),
                "offset_residual": offset_residual,
                "centroid": [xc, yc],
                "sigma_scale": denom,
                "z": (abs(offset_residual) / denom) if (np.isfinite(denom) and denom > 0) else None,
                "method": "fundamental_cycle_basis",
            })
    return out


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide_state(*, connected, mismatch, gates):
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


def _corner_gate(corner_unc, coeffs, sigma_delta, cfg):
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
# Application
# ---------------------------------------------------------------------------

def plane_field(bounds, canvas_hw, coeffs):
    xs, ys = normalized_coords(bounds, canvas_hw)
    a, b, d = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a + b * xs[None, :] + d * ys[:, None]


def apply_plane_correction(science, coeffs, bounds, canvas_hw):
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


def nonlinear_study():
    return {
        "model": MODEL_NONLINEAR_CANDIDATE, "status": "research_only_no_engine",
        "references": [
            "PhotometricMosaic (John Murphy): photometric scale matching + surface spline",
            "Montage (Caltech/IPAC): planar offset/slope matching with calibration assumptions",
        ],
        "scientific_position": (
            "A per-Cell spline could absorb real sky/IFN and is only identifiable with "
            "independent calibration; NOT executable here. Nonlinear residual is reported "
            "as MODEL_MISMATCH needing research/upstream calibration, never repaired by a plane."
        ),
        "credits_proposed": [
            "PhotometricMosaic — John Murphy (methodological reference)",
            "Montage — Caltech/IPAC (methodological reference)",
        ],
    }


# ---------------------------------------------------------------------------
# Result payload
# ---------------------------------------------------------------------------

@dataclass
class ModelComparisonResult:
    decision: str
    model: str
    cell_ids: tuple
    anchor: str
    channels: int
    offset_offsets: np.ndarray
    plan_coefficients: np.ndarray
    reg_lambda: float
    regularization_ladder: tuple
    heldout: dict
    gates: dict
    rank_data: int | None
    cond_data: float | None
    coef_uncertainty: np.ndarray | None
    corner_uncertainty: np.ndarray | None
    per_channel: dict
    cycles: list
    cycles_raw: list
    dropped: list
    edge_stats: list
    mismatch: dict
    sigma_delta: float
    disconnected: tuple
    truth_diagnostics: dict
    nonlinear_study: dict
    provenance: dict
    accepted_keys: list = field(default_factory=list)
    refusal_reasons: list = field(default_factory=list)
    notes: tuple = ()

    def to_json_dict(self):
        payload = {
            "decision": self.decision, "model": self.model, "anchor": self.anchor,
            "channels": self.channels, "cell_ids": list(self.cell_ids),
            "reg_lambda": self.reg_lambda,
            "regularization_ladder": list(self.regularization_ladder),
            "rank_data": self.rank_data,
            "cond_data": self.cond_data,
            "sigma_delta": self.sigma_delta,
            "heldout": {
                "r_plan": self.heldout.get("r_plan"),
                "r_offset": self.heldout.get("r_offset"),
                "ratio": self.heldout.get("ratio"),
                "orientation_ratio": dict(self.heldout.get("orientation_ratio", {})),
                "per_edge": self.heldout.get("per_edge", []),
            },
            "gates": dict(self.gates), "mismatch": self.mismatch,
            "per_channel": self.per_channel, "cycles": self.cycles,
            "cycles_raw": self.cycles_raw, "dropped": self.dropped,
            "accepted_keys": self.accepted_keys,
            "refusal_reasons": list(self.refusal_reasons),
            "disconnected": list(self.disconnected),
            "truth_diagnostics": self.truth_diagnostics,
            "offset_offsets": self.offset_offsets,
            "plan_coefficients": self.plan_coefficients,
            "nonlinear_study": self.nonlinear_study, "provenance": dict(self.provenance),
            "notes": list(self.notes),
        }
        return _sanitize(payload)


def _sanitize(obj):
    """Recursive JSON sanitiser: any non-finite float/scalar -> ``None``.

    Handles Mapping/list/tuple/ndarray/numpy scalar/float at any depth; numpy
    bools/ints become Python bool/int. Strings and ``None`` are preserved (never
    masked). Applied to the ENTIRE payload, so ``json.dumps(..., allow_nan=False)``
    always succeeds.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _nan_list(a):
    """JSON-safe nested lists for an array (any ndim), non-finite -> null."""
    return _sanitize(np.asarray(a, dtype=np.float64))


def _edge_stats_json(o: EdgeObservation) -> dict:
    info = 0.5 * (o.information + o.information.T) if o.information.size else o.information
    reason = "band_clamped_to_overlap" if o.band_clamped else None
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
        "refusal_reason": reason,
    }


def gauge_truth_error(truth_planes, coeffs, cell_ids, anchor, channels):
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


def compare_models(cells, nx, ny, canvas_hw, *, channels=None, anchor=None,
                   measurement_px=DEFAULT_MEASUREMENT_PX, feather_px=DEFAULT_FEATHER_PX,
                   block_px=BLOCK_PX, cfg=None, truth_planes=None):
    """Evaluate M0 vs M1. The ACCEPTED relation set is the source of all gates."""
    if channels is None:
        channels = cells[next(iter(cells))].channels()
    cell_ids = tuple(f"r{r:04d}c{c:04d}" for r in range(ny) for c in range(nx)
                     if f"r{r:04d}c{c:04d}" in cells)
    if not cell_ids:
        raise ValueError("no cells provided")
    if anchor is None:
        anchor = select_central_cell(nx, ny)
        if anchor not in cells:
            anchor = cell_ids[0]

    obs_raw, dropped = build_observations(cells, nx, ny, canvas_hw, channels=channels,
                                          block_px=block_px, measurement_px=measurement_px)
    return compare_models_from_observations(
        obs_raw, dropped, cell_ids=cell_ids, nx=nx, ny=ny, canvas_hw=canvas_hw,
        channels=channels, anchor=anchor, measurement_px=measurement_px,
        feather_px=feather_px, block_px=block_px, cfg=cfg, truth_planes=truth_planes,
    )


def compare_models_from_observations(
        obs_raw, dropped, *, cell_ids, nx, ny, canvas_hw, channels, anchor,
        measurement_px=DEFAULT_MEASUREMENT_PX, feather_px=DEFAULT_FEATHER_PX,
        block_px=BLOCK_PX, cfg=None, truth_planes=None):
    """Evaluate M0 vs M1 from ALREADY-BUILT observations (ZM-ZEGRID-R37-C).

    Behaviour-preserving extraction of the tail of :func:`compare_models`: the
    only difference is that the caller supplies ``obs_raw``/``dropped`` (and the
    layout metadata) directly instead of source ``cells``. This is the pure
    conversion helper that lets the OFFLINE diagnostic artifact rerun the
    accepted M0/M1 decisions without source FITS. Gates/thresholds/solver are
    unchanged; ``truth_planes`` is optional exactly as in :func:`compare_models`.
    """
    cfg = cfg or GateConfig()
    if not cell_ids:
        raise ValueError("no cells provided")
    index = {cid: i for i, cid in enumerate(cell_ids)}

    def _filter(plan):
        acc = _accepted_key_set(plan)
        return [o for o in obs_raw if o.key() in acc], acc

    # Ladder selection on train/holdout over each lambda's accepted set.
    scores = {}
    for lam in REGULARIZATION_LADDER:
        p = solve_plan_sparse(obs_raw, cell_ids, anchor, channels, reg_lambda=lam,
                              with_uncertainty=False)
        oa, _ = _filter(p)
        conn = {c: d["connected"] for c, d in p.per_channel.items()}
        m0 = solve_offset_baseline(oa, cell_ids, anchor, channels, channel_connected=conn)
        scores[lam] = heldout_metrics(oa, p.coefficients, m0, index, channels)["r_plan"]
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best_lambda = max(k for k, v in finite.items() if v <= min(finite.values()) * (1.0 + 1e-9)) if finite else 0.0

    plan = solve_plan_sparse(obs_raw, cell_ids, anchor, channels, reg_lambda=best_lambda,
                             with_uncertainty=True)
    obs, accepted_keys = _filter(plan)
    conn = {c: d["connected"] for c, d in plan.per_channel.items()}
    chan_conn = {c: {"connected": d["connected"], "edges": d["accepted"]}
                 for c, d in plan.per_channel.items()}
    disconnected_channels = [c for c, d in chan_conn.items() if not d["connected"]]
    disconnected_cells = sorted({cid for c in disconnected_channels
                                 for cid in _channel_connectivity(cell_ids, obs, anchor, channels)[c]["missing"]})

    m0 = solve_offset_baseline(obs, cell_ids, anchor, channels, channel_connected=conn)
    hm = heldout_metrics(obs, plan.coefficients, m0, index, channels)

    sigmas = [o.robust_sigma for o in obs if o.robust_sigma > 0]
    sigma_delta = float(np.median(sigmas)) if sigmas else 0.0
    noise_estimable = bool(np.isfinite(sigma_delta) and sigma_delta > 0)
    mismatch, mismatch_info = detect_mismatch(obs, cfg, coeffs=plan.coefficients, index=index)
    cycles = _cycle_residuals(obs, cell_ids, nx, ny, canvas_hw, channels)
    cycles_raw = _cycle_residuals(obs_raw, cell_ids, nx, ny, canvas_hw, channels)

    # Evaluated accepted cycles vs the cyclomatic number, per channel (R2-2).
    evaluated_cycles: dict = {}
    for cy in cycles:
        evaluated_cycles[cy["channel"]] = evaluated_cycles.get(cy["channel"], 0) + 1
    cyclomatic_cycles: dict = {}
    for c, d in plan.per_channel.items():
        edges_c = [(o.u, o.v) for o in obs if o.channel == c]
        cyclomatic_cycles[c] = _independent_cycles(cell_ids, edges_c, anchor)

    admissible = [c for c, d in plan.per_channel.items() if d["accepted"] > 0]
    min_cycles = min((evaluated_cycles.get(c, 0) for c in admissible), default=0)
    cycle_ok = bool(admissible)

    refusal_reasons = []
    for c in disconnected_channels:
        refusal_reasons.append(f"channel {c} disconnected on the accepted relation set")
    for c in admissible:
        ev = evaluated_cycles.get(c, 0)
        cy = cyclomatic_cycles.get(c, 0)
        if ev < 1 or ev != cy:
            cycle_ok = False
            refusal_reasons.append(
                f"cycle_diagnostic_incomplete: channel {c} evaluated {ev} vs cyclomatic {cy}")

    eff_ok = bool(obs) and all(o.effective_train >= cfg.min_effective_train for o in obs)
    blocks_ok = bool(obs) and all(
        o.tangent_blocks >= cfg.min_tangent_blocks and o.normal_blocks >= cfg.min_normal_blocks
        for o in obs)
    if not obs:
        evidence = "FAIL"
        refusal_reasons.append("no accepted relations")
    elif not any(d["connected"] for d in plan.per_channel.values()):
        evidence = "FAIL"
    elif not cycle_ok:
        evidence = "FAIL"
        if min_cycles < 1 and not any(r.startswith("no_cycle_redundancy") for r in refusal_reasons):
            refusal_reasons.append("no_cycle_redundancy: accepted graph acyclic (chain)")
    elif not noise_estimable:
        evidence = "FAIL"
        refusal_reasons.append("degenerate: noise/uncertainty scale not estimable (sigma_delta==0)")
    else:
        evidence = "PASS" if (eff_ok and blocks_ok) else "FAIL"

    finite_unc = True
    if plan.coef_uncertainty is not None and np.any(np.isfinite(plan.coef_uncertainty)):
        finite_unc = bool(np.all(np.isfinite(plan.coef_uncertainty)))
    rank_ok = plan.rank_data is not None and plan.rank_data == plan.unknowns
    cond_ok = plan.cond_data is not None and np.isfinite(plan.cond_data) and plan.cond_data <= cfg.cond_a_max
    corner = _corner_gate(plan.corner_uncertainty, plan.coefficients, sigma_delta, cfg)
    numerics = "PASS" if (rank_ok and cond_ok and finite_unc and corner == "PASS") else "FAIL"

    r_plan, r_off = hm["r_plan"], hm["r_offset"]
    ratio_ok = bool(np.isfinite(r_plan) and np.isfinite(r_off) and r_off > 0 and r_plan <= cfg.heldout_ratio_max * r_off)
    abs_ok = bool(np.isfinite(r_plan) and noise_estimable and r_plan <= cfg.heldout_abs_sigma_max * sigma_delta)
    orient_ok = _orientation_ok(hm["orientation_ratio"], cfg.heldout_ratio_max)
    p90_ok = _p90_ok(hm["per_edge"], cfg.per_edge_p90_worsen_max)
    heldout = "PASS" if (ratio_ok and abs_ok and orient_ok and p90_ok) else "FAIL"

    gates = {
        "evidence": evidence, "numerics": numerics, "heldout": heldout,
        "connected": "PASS" if not disconnected_channels else "FAIL",
        "rank": "PASS" if rank_ok else ("NOT_RUN" if plan.rank_data is None else "FAIL"),
        "condition": "PASS" if cond_ok else ("NOT_RUN" if plan.cond_data is None else "FAIL"),
        "corner_uncertainty": corner,
        "cycle_redundancy": "PASS" if cycle_ok else "FAIL",
    }

    truth_diag = {}
    if truth_planes is not None:
        truth_diag = gauge_truth_error(truth_planes, plan.coefficients, cell_ids, anchor, channels)

    decision = decide_state(connected=(not disconnected_channels), mismatch=mismatch, gates=gates)
    if decision == STATE_ROBUST_PLAN and not (
        gates["heldout"] == "PASS" and gates["numerics"] == "PASS"
        and gates["evidence"] == "PASS" and gates["cycle_redundancy"] == "PASS"
    ):
        decision = STATE_OFFSET_ONLY

    per_channel = {int(c): dict(d) for c, d in plan.per_channel.items()}
    for c in per_channel:
        per_channel[c]["m0_applicable"] = bool(c not in disconnected_channels
                                               and per_channel[c]["accepted"] > 0)
        per_channel[c]["cyclomatic_cycles"] = int(cyclomatic_cycles.get(c, 0))
        per_channel[c]["evaluated_cycles"] = int(evaluated_cycles.get(c, 0))

    provenance = dict(PROVENANCE)
    provenance.update({
        "measurement_px": measurement_px, "feather_px": feather_px, "block_px": block_px,
        "channels": channels, "n_cells": len(cell_ids), "n_edges_raw": len(obs_raw),
        "n_edges_accepted": len(obs), "n_dropped_relations": len(dropped),
        "anchor": anchor, "selected_reg_lambda": best_lambda,
        "ladder_scores": {str(k): (None if not np.isfinite(v) else float(v)) for k, v in scores.items()},
        "sigma_delta": None if not np.isfinite(sigma_delta) else float(sigma_delta),
        "noise_estimable": noise_estimable,
        "global_solve_densified": False,
        "diagnostics_densified": bool(plan.diagnostics.get("diagnostics_densified", False)),
        "diagnostics_dense_limit": _DENSE_DIAG_LIMIT,
        "design_rows_total": int(plan.rows),
        "min_cycle_redundancy": int(min_cycles),
        "evaluated_cycles_total": int(sum(evaluated_cycles.values())),
        "cyclomatic_cycles_total": int(sum(cyclomatic_cycles.values())),
    })

    return ModelComparisonResult(
        decision=decision,
        model=(MODEL_ROBUST_PLAN if decision == STATE_ROBUST_PLAN else MODEL_OFFSET_ONLY),
        cell_ids=cell_ids, anchor=anchor, channels=channels,
        offset_offsets=m0, plan_coefficients=plan.coefficients,
        reg_lambda=plan.reg_lambda, regularization_ladder=REGULARIZATION_LADDER,
        heldout=hm, gates=gates, rank_data=plan.rank_data, cond_data=plan.cond_data,
        coef_uncertainty=plan.coef_uncertainty, corner_uncertainty=plan.corner_uncertainty,
        per_channel=per_channel, cycles=cycles, cycles_raw=cycles_raw, dropped=dropped,
        edge_stats=[_edge_stats_json(o) for o in obs_raw], mismatch=mismatch_info,
        sigma_delta=sigma_delta, disconnected=tuple(disconnected_cells),
        truth_diagnostics=truth_diag, nonlinear_study=nonlinear_study(),
        provenance=provenance, accepted_keys=sorted(accepted_keys),
        refusal_reasons=refusal_reasons,
    )


# ---------------------------------------------------------------------------
# Scalability witness
# ---------------------------------------------------------------------------

def sparse_scale_witness(side, *, channels=1, reg_lambda=1e-3, measure=True):
    import resource
    import time

    nx = ny = int(side)
    cell_ids = tuple(f"r{r:04d}c{c:04d}" for r in range(ny) for c in range(nx))
    info = np.diag([16.0, 16.0, 4.0])
    obs = []
    for (u, v) in grid_adjacency(nx, ny):
        (ur, uc), (vr, vc) = _parse_rc(u), _parse_rc(v)
        theta = np.array([0.05 * (((ur + uc) % 11) - 5),
                          0.01 * (((ur * 7 + uc * 3) % 9) - 4),
                          0.01 * (((ur * 5 + uc * 13) % 9) - 4)])
        obs.append(EdgeObservation(
            u=u, v=v, channel=0, orientation=("h" if vc > uc else "v"),
            n_train=256, n_holdout=0, effective_train=256.0, tangent_blocks=8,
            normal_blocks=4, robust_sigma=1.0, rejection_fraction=0.0, theta_global=theta,
            information=info.copy(), rhs=info @ theta, level_z=0.0, curvature_z=0.0,
            level_mean=0.0, local_offset_train=float(theta[0]), measurement_px=None,
            overlap_px=8, band_used_px=8, band_clamped=False,
            holdout_xy=np.zeros((0, 2)), holdout_diff=np.zeros(0)))
    anchor = select_central_cell(nx, ny)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if measure else 0
    t0 = time.perf_counter()
    plan = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=reg_lambda,
                             with_uncertainty=False)
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if measure else 0
    diag = plan.diagnostics
    return {
        "cells": len(cell_ids), "edges": len(obs), "unknowns": plan.unknowns,
        "regularized_rows": int(plan.rows), "rank_data": plan.rank_data,
        "cond_data": plan.cond_data, "solve_seconds": dt,
        "rss_delta_kib": int(max(rss1 - rss0, 0)), "rss_peak_kib": int(rss1),
        "dense_design_bytes_estimate": (int(plan.rows) * plan.unknowns * 8) if plan.unknowns else 0,
        "method": "scipy.sparse.linalg.lsqr_on_whitened_design",
        "global_solve_densified": False,
        "diagnostics_densified": bool(diag.get("diagnostics_densified", False)),
        "dense_diag_limit": diag.get("dense_diag_limit", _DENSE_DIAG_LIMIT),
        "rows_semantics": "sum over channels of final assembled design rows (data + ridge); channels=1",
    }
