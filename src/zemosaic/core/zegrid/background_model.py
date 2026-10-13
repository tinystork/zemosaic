"""ZM-ZEGRID-R37-B — inter-Cell ADDITIVE background model prototype (PURE, UNWIRED).

This module is the ISOLATED Gate-B prototype mandated by
``docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md``. It is **never imported by
production** (``zemosaic_zegrid_mode.py``, ``cell_feather.py``, GUI, config,
dispatcher, canonical SCI-05). It only depends on NumPy/SciPy and the PURE
``cell_offset`` module (reused UNCHANGED as the M0 baseline).

What it builds
--------------
Two candidate inter-Cell models, evaluated on the SAME synthetic corpora:

* **M0 = ``OFFSET_ONLY``** — the current R34 scalar additive baseline, reusing
  :func:`zemosaic.core.zegrid.cell_offset.solve_additive_offsets` (one additive
  level per Cell/channel, central anchor) WITHOUT modification.
* **M1 = ``ROBUST_PLAN``** — a per-Cell/channel additive plane
  ``P_i(x,y,c) = a_i,c + b_i,c*x + d_i,c*y`` in GLOBAL normalized canvas
  coordinates ``x,y ∈ [-1,1]``. The central deterministic anchor Cell has
  ``(a,b,d)=(0,0,0)``.

Both are ADDITIVE-ONLY. No gain, colour matrix, multiplicative normalization or
hidden rescaling is ever applied. Application is ``science += a + b*x + d*y``.

Scientific discipline encoded here (frozen contract, never relaxed)
-------------------------------------------------------------------
* Observation is a **same-global-pixel pair difference** on shared-valid/finite
  pixels only; measurement band width and assembly feather width are SEPARATE.
* Deterministic spatial **75/25 train/holdout** split by fixed block parity — no
  RNG anywhere in the fitting path.
* Robust **Huber** edge fit (fixed transition ``1.345 * robust_sigma``), float64.
* Global solve via **sparse direct least squares** (SciPy ``LSQR``) on a whitened
  design; never a dense global matrix / normal equations.
* Slopes-only regularization over the FROZEN ladder
  ``{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}``, selected on train/holdout only.
* Data-only rank/condition reported SEPARATELY from the regularized solve;
  regularization never disguises absent evidence.
* Decision is one of the contract states:
  ``OFFSET_ONLY | ROBUST_PLAN | INSUFFICIENT_EVIDENCE | ILL_CONDITIONED |
  MODEL_MISMATCH | DISCONNECTED``. ``NONLINEAR_CANDIDATE`` is a research label
  only — there is NO spline engine here.

Scope: research-quality pure block. Adds no feathering, changes no default,
touches no canonical/FITS/production surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from .cell_offset import (
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
# Model / decision vocabulary (contract §1, §5)
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
# Frozen numerical constants (contract §3, §4, §6)
# ---------------------------------------------------------------------------

HUBER_DELTA_FACTOR = 1.345
MAD_SCALE = 1.4826
BLOCK_PX = 4
DEFAULT_MEASUREMENT_PX = 16
DEFAULT_FEATHER_PX = 8
REGULARIZATION_LADDER: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
_HUBER_MAX_ITER = 60
_HUBER_TOL = 1e-10
_LSQR_ATOL = 1e-13
_LSQR_BTOL = 1e-13
_LSQR_ITERS = 20000
_DENSE_DIAG_LIMIT = 1500
_EIG_TOL_REL = 1e-12

#: Module provenance (JSON-safe, contract §8).
PROVENANCE = {
    "module": "zemosaic.core.zegrid.background_model",
    "mission": "ZM-ZEGRID-R37-B",
    "gate": "B",
    "status": "pure_prototype_unwired",
    "coordinate_system": "global_canvas_normalized_[-1,1]",
    "model": "P_i(x,y,c) = a + b*x + d*y (additive only)",
    "robust_loss": "huber_irls_fixed_1.345_mad_sigma",
    "global_solve": "scipy.sparse.linalg.lsqr_on_whitened_design",
    "regularization": "slopes_only_ladder_frozen",
    "split": "deterministic_spatial_75_25_block_parity",
    "references": [
        "Montage (Caltech/IPAC) — offset/slope overlap matching, calibration assumptions",
        "PhotometricMosaic (John Murphy) — scale matching + surface spline (research only)",
    ],
    "no_third_party_code_copied": True,
}


@dataclass(frozen=True)
class GateConfig:
    """Predeclared acceptance thresholds (contract §6). Do not relax post-hoc."""

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
    """One Cell's PATCH-level science plane at its GLOBAL position.

    ``bounds`` = global ``(y0, y1, x0, x1)``. ``science`` is ``(Hp, Wp)`` mono or
    ``(Hp, Wp, C)`` multi-channel (NaN where invalid). ``valid_mask`` is an
    optional channel-invariant ``(Hp, Wp)`` bool plane.
    """

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
    """Huber-IRLS fit of ``target ≈ phi @ coef`` for a compact ``phi`` (n, k).

    Returns ``(coef, weights, robust_sigma)``. Deterministic: plain float64
    reductions, fixed iteration count and tolerance, NO RNG.
    """
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
        scale = max(1.0, float(np.max(np.abs(coef))))
        coef = coef_new
        if delta <= _HUBER_TOL * scale:
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
    theta_global: np.ndarray          # (3,) robust train fit [Delta_a, Delta_b, Delta_d]
    information: np.ndarray           # (3,3) train normal matrix (global coords)
    rhs: np.ndarray                   # (3,) train rhs (global coords)
    level_z: float
    curvature_z: float
    level_mean: float
    local_offset: float               # robust median of the raw band difference
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
) -> EdgeObservation | None:
    """Pure same-global-pixel observation for one (edge, channel).

    Returns ``None`` when the shared-valid support cannot be fit (never a
    fabricated observation). ``normal_boundary`` (global coordinate of the shared
    CORE edge in the normal axis) + ``measurement_px`` restrict the sample to a
    symmetric band; measurement width is independent of the feather width.
    """
    uy0, uy1, ux0, ux1 = cell_u.bounds
    vy0, vy1, vx0, vx1 = cell_v.bounds
    y0, y1 = max(uy0, vy0), min(uy1, vy1)
    x0, x1 = max(ux0, vx0), min(ux1, vx1)
    if y1 <= y0 or x1 <= x0:
        return None

    su = cell_u.as_3d()[y0 - uy0 : y1 - uy0, x0 - ux0 : x1 - ux0, channel]
    sv = cell_v.as_3d()[y0 - vy0 : y1 - vy0, x0 - vx0 : x1 - vx0, channel]
    vu = _valid_2d(cell_u)[y0 - uy0 : y1 - uy0, x0 - ux0 : x1 - ux0]
    vv = _valid_2d(cell_v)[y0 - vy0 : y1 - vy0, x0 - vx0 : x1 - vx0]
    shared = np.isfinite(su) & np.isfinite(sv) & vu & vv
    if not np.any(shared):
        return None

    gy_l, gx_l = np.nonzero(shared)
    gx = gx_l + x0
    gy = gy_l + y0

    xs, ys = normalized_coords((y0, y1, x0, x1), canvas_hw)
    X = xs[gx_l]
    Y = ys[gy_l]
    diff = su[shared] - sv[shared]

    is_horizontal = abs(vx0 - ux0) >= abs(vy0 - uy0)
    orientation = "h" if is_horizontal else "v"
    if measurement_px is not None and normal_boundary is not None:
        half = 0.5 * float(measurement_px)
        normal_val = np.where(is_horizontal, gx, gy).astype(np.float64) + 0.5
        keep = np.abs(normal_val - float(normal_boundary)) <= half
        if not np.any(keep):
            return None
        gx, gy, X, Y, diff = gx[keep], gy[keep], X[keep], Y[keep], diff[keep]

    bx = gx // int(block_px)
    by = gy // int(block_px)
    holdout = (bx % 2 == 1) & (by % 2 == 1)
    train = ~holdout
    n_train = int(np.count_nonzero(train))
    if n_train < 4:
        return None

    xbar = float(np.mean(X[train]))
    ybar = float(np.mean(Y[train]))
    phi_c = np.column_stack([np.ones(n_train), X[train] - xbar, Y[train] - ybar])
    tgt = diff[train]
    coef_c, w, sigma = _robust_plane_fit(phi_c, tgt, huber_factor=huber_factor)
    # Degenerate (effectively noiseless) fit: MAD is at numerical-noise level, so a
    # Huber cut would flag half the samples spuriously. Treat the sample as
    # outlier-free (unit weights) and report an honest zero robust scale.
    _scale = float(np.max(np.abs(tgt))) if tgt.size else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-10 * max(_scale, 1.0):
        w = np.ones(n_train, dtype=np.float64)
        sigma = 0.0

    wphi = phi_c * w[:, None]
    sigma_info = sigma if sigma > 0.0 else 1.0
    info_c = (phi_c.T @ wphi) / (sigma_info * sigma_info)
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

    level_z = _slope_z(level_c, level)
    curvature_z = _slope_z(radius_c, radius)
    rejection_fraction = (
        float(np.mean(np.abs(resid) > huber_factor * sigma)) if sigma > 0 else 0.0
    )
    tangent_blocks = int(np.unique((gy if is_horizontal else gx) // int(block_px)).size)
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
        level_z=level_z,
        curvature_z=curvature_z,
        level_mean=float(np.mean(level)),
        local_offset=float(np.median(diff)),
        holdout_xy=np.column_stack([X[holdout], Y[holdout]]),
        holdout_diff=diff[holdout],
    )


def _parse_rc(cell_id: str) -> tuple[int, int]:
    row = int(cell_id[1:5])
    col = int(cell_id[6:10])
    return row, col


def build_observations(
    cells: Mapping[str, PlanCell],
    nx: int,
    ny: int,
    canvas_hw: tuple[int, int],
    *,
    channels: int | None = None,
    block_px: int = BLOCK_PX,
    measurement_px: int | None = DEFAULT_MEASUREMENT_PX,
) -> list[EdgeObservation]:
    """Build every (edge, channel) observation over the grid adjacency.

    Deterministic order (row-major, right-before-down). The shared CORE boundary
    for an edge is derived from the uniform grid (``canvas_w/nx``, ``canvas_h/ny``).
    """
    if channels is None:
        channels = cells[next(iter(cells))].channels()
    w, h = int(canvas_hw[1]), int(canvas_hw[0])
    cell_w = w / float(nx)
    cell_h = h / float(ny)

    out: list[EdgeObservation] = []
    for (u, v) in grid_adjacency(nx, ny):
        if u not in cells or v not in cells:
            continue
        cu, cv = cells[u], cells[v]
        (ur, uc), (vr, vc) = _parse_rc(u), _parse_rc(v)
        if vr > ur:
            boundary = float((ur + 1) * cell_h)
        else:
            boundary = float((uc + 1) * cell_w)
        for c in range(channels):
            obs = observe_edge(
                cu, cv, canvas_hw, c,
                block_px=block_px,
                measurement_px=measurement_px,
                normal_boundary=boundary,
            )
            if obs is not None:
                out.append(obs)
    return out


# ---------------------------------------------------------------------------
# M0 baseline (reuses the PURE R34c solver UNCHANGED)
# ---------------------------------------------------------------------------

def solve_offset_baseline(
    observations: Sequence[EdgeObservation],
    cell_ids: Sequence[str],
    anchor: str,
    channels: int,
) -> np.ndarray:
    """M0 = ``OFFSET_ONLY`` via the UNCHANGED R34c sparse solver.

    Each (edge, channel) becomes a robust scalar offset observation (the robust
    LOCAL band difference) with support-count weight. Returns ``(N, C)``.
    """
    deltas: dict[tuple[str, str], list[float]] = {}
    weights: dict[tuple[str, str], list[float]] = {}
    for obs in observations:
        key = (obs.u, obs.v)
        deltas.setdefault(key, [0.0] * channels)
        weights.setdefault(key, [0.0] * channels)
        deltas[key][obs.channel] = float(obs.local_offset)
        weights[key][obs.channel] = float(max(obs.n_train + obs.n_holdout, 1))
    edges = [
        OffsetEdge(
            u, v,
            np.asarray(deltas[(u, v)], dtype=np.float64),
            np.asarray(weights[(u, v)] if any(weights[(u, v)]) else [1.0] * channels, dtype=np.float64),
        )
        for (u, v) in deltas
    ]
    sol = solve_additive_offsets(
        list(cell_ids), edges, anchor, channels, raise_on_disconnected=False
    )
    return np.asarray(sol.offsets, dtype=np.float64)


# ---------------------------------------------------------------------------
# Global sparse plan solve (whitened, slopes-only regularization)
# ---------------------------------------------------------------------------

def _assemble_channel_system(
    obs_c: Sequence[EdgeObservation],
    anchor: str,
    *,
    cell_slot: Mapping[str, int],
    reg_lambda: float,
) -> tuple[csr_matrix, np.ndarray, int]:
    """Whitened sparse design/rhs for one channel.

    Each edge contributes ``rank(I_e)`` rows ``R = diag(sqrt(lambda)) V^T`` acting
    on ``p_v - p_u`` (anchor columns dropped). Slopes-only ridge adds
    ``sqrt(reg_lambda)`` rows on the ``b``/``d`` columns of non-anchor Cells.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    r = 0
    for obs in obs_c:
        info = np.asarray(obs.information, dtype=np.float64)
        if not np.all(np.isfinite(info)):
            continue
        info = 0.5 * (info + info.T)
        evals, evecs = np.linalg.eigh(info)
        cut = max(float(np.max(evals)) * _EIG_TOL_REL, 0.0) if evals.size else 0.0
        for k in range(evals.size):
            lam = float(evals[k])
            if lam <= cut or lam <= 0.0:
                continue
            vec = evecs[:, k] * np.sqrt(lam)
            sv = cell_slot.get(obs.v)
            su = cell_slot.get(obs.u)
            for j in range(3):
                if sv is not None:
                    rows.append(r); cols.append(3 * sv + j); vals.append(float(vec[j]))
                if su is not None:
                    rows.append(r); cols.append(3 * su + j); vals.append(-float(vec[j]))
            rhs.append(float(vec @ np.asarray(obs.theta_global, dtype=np.float64)))
            r += 1

    n_free = 3 * max(len(cell_slot), 0)
    if reg_lambda > 0.0:
        s = float(np.sqrt(float(reg_lambda)))
        for cid, slot in cell_slot.items():
            if cid == anchor:
                continue
            base = 3 * slot
            for j in (1, 2):  # slopes only
                rows.append(r); cols.append(base + j); vals.append(s)
                rhs.append(0.0)
                r += 1

    a = csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(r, n_free),
    )
    return a, np.asarray(rhs, dtype=np.float64), n_free


def _solve_lsqr(a: csr_matrix, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or a.shape[1] == 0:
        return np.zeros(a.shape[1], dtype=np.float64)
    res = lsqr(a, b, atol=_LSQR_ATOL, btol=_LSQR_BTOL, iter_lim=_LSQR_ITERS)
    return np.asarray(res[0], dtype=np.float64)


@dataclass
class PlanSolve:
    """Outcome of one global plan solve (all channels)."""

    coefficients: np.ndarray           # (N, C, 3) [a, b, d]; anchor row = 0
    reg_lambda: float
    rank_data: int | None
    cond_data: float | None
    unknowns: int
    rows: int
    coef_uncertainty: np.ndarray | None
    corner_uncertainty: np.ndarray | None
    diagnostics: dict = field(default_factory=dict)


def _dense_rank_cond(a: csr_matrix) -> tuple[int | None, float | None]:
    """Data-only rank/condition of the whitened design (never the normal matrix)."""
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


def _corner_uncertainties(
    cov: np.ndarray, cell_slot: Mapping[str, int], anchor: str, channels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single-channel coefficient + corner sigmas from a covariance matrix."""
    n = max(len(cell_slot), 1)
    coef_sigma = np.full((n, 3), np.inf, dtype=np.float64)
    corner_sigma = np.full((n, 4), np.inf, dtype=np.float64)
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for cid, slot in cell_slot.items():
        if cid == anchor:
            coef_sigma[slot, :] = 0.0
            corner_sigma[slot, :] = 0.0
            continue
        base = 3 * slot
        block = cov[base : base + 3, base : base + 3]
        if np.all(np.isfinite(block)):
            coef_sigma[slot] = np.sqrt(np.maximum(np.diag(block), 0.0))
        for ci, (cx, cy) in enumerate(corners):
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
) -> PlanSolve:
    """Global M1 solve: sparse whitened LSQR per channel, slopes-only ridge.

    Never builds a dense global design or dense normal equations. Data-only
    rank/condition is computed SEPARATELY from the regularized solve.
    """
    if anchor not in cell_ids:
        raise ValueError(f"anchor {anchor!r} not among cells")
    index = {cid: i for i, cid in enumerate(cell_ids)}
    cell_slot = {cid: i for i, cid in enumerate(c for c in cell_ids if c != anchor)}
    n = len(cell_ids)
    coeffs = np.zeros((n, channels, 3), dtype=np.float64)
    coef_unc = None
    corner_unc = None
    rank_data = None
    cond_data = None
    total_rows = 0
    unknowns = 3 * max(n - 1, 0)
    n_free = 3 * max(n - 1, 0)

    for c in range(channels):
        obs_c = [o for o in observations if o.channel == c]
        a, b, _nf = _assemble_channel_system(obs_c, anchor, cell_slot=cell_slot, reg_lambda=reg_lambda)
        total_rows += int(a.shape[0])
        a_data, _, _ = _assemble_channel_system(obs_c, anchor, cell_slot=cell_slot, reg_lambda=0.0)
        r, cd = _dense_rank_cond(a_data)
        if rank_data is None or (r is not None and r > rank_data):
            rank_data = r
        if cond_data is None or (cd is not None and cd > cond_data):
            cond_data = cd

        sol = _solve_lsqr(a, b)
        for cid, slot in cell_slot.items():
            coeffs[index[cid], c, :] = sol[3 * slot : 3 * slot + 3]

        if with_uncertainty and n_free <= _DENSE_DIAG_LIMIT and a.shape[0] > 0 and n_free > 0:
            dense = a.toarray().astype(np.float64)
            normal = dense.T @ dense
            if normal.shape[0] and np.all(np.isfinite(normal)):
                try:
                    cov = np.linalg.inv(normal)
                except np.linalg.LinAlgError:
                    cov = np.linalg.pinv(normal)
                cs, cors = _corner_uncertainties(cov, cell_slot, anchor, channels)
                if coef_unc is None:
                    coef_unc = np.zeros((n, channels, 3), dtype=np.float64)
                    corner_unc = np.zeros((n, channels, 4), dtype=np.float64)
                for cid, slot in cell_slot.items():
                    coef_unc[index[cid], c, :] = cs[slot]
                    corner_unc[index[cid], c, :] = cors[slot]

    return PlanSolve(
        coefficients=coeffs,
        reg_lambda=float(reg_lambda),
        rank_data=rank_data,
        cond_data=cond_data,
        unknowns=unknowns,
        rows=total_rows,
        coef_uncertainty=coef_unc,
        corner_uncertainty=corner_unc,
        diagnostics={"regularized_rows": total_rows},
    )


# ---------------------------------------------------------------------------
# Held-out evaluation (M0 vs M1)
# ---------------------------------------------------------------------------

def _robust_sigma(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("nan")
    med = float(np.median(samples))
    return MAD_SCALE * float(np.median(np.abs(samples - med)))


def heldout_metrics(
    observations: Sequence[EdgeObservation],
    coeffs_plan: np.ndarray,
    offsets_m0: np.ndarray,
    index: Mapping[str, int],
    channels: int,
) -> dict:
    """Held-out residual metrics for M0 (offset) and M1 (plan), robust + p90."""
    resid_plan_all: list[np.ndarray] = []
    resid_m0_all: list[np.ndarray] = []
    per_edge: list[dict] = []
    by_orientation: dict[str, dict[str, list[np.ndarray]]] = {
        "h": {"plan": [], "m0": []},
        "v": {"plan": [], "m0": []},
    }
    for obs in observations:
        xy = obs.holdout_xy
        if xy.shape[0] == 0:
            continue
        c = obs.channel
        p_u = coeffs_plan[index[obs.u], c, :]
        p_v = coeffs_plan[index[obs.v], c, :]
        phi = np.column_stack([np.ones(xy.shape[0]), xy[:, 0], xy[:, 1]])
        pred_plan = phi @ (p_v - p_u)
        pred_m0 = np.full(xy.shape[0], offsets_m0[index[obs.v], c] - offsets_m0[index[obs.u], c])
        r_plan = obs.holdout_diff - pred_plan
        r_m0 = obs.holdout_diff - pred_m0
        p90_plan = float(np.percentile(np.abs(r_plan - np.median(r_plan)), 90))
        p90_m0 = float(np.percentile(np.abs(r_m0), 90))
        per_edge.append({
            "u": obs.u, "v": obs.v, "channel": c, "orientation": obs.orientation,
            "n_holdout": int(xy.shape[0]),
            "p90_plan": p90_plan, "p90_m0": p90_m0,
            "rms_plan": float(np.sqrt(np.mean(r_plan * r_plan))),
            "rms_m0": float(np.sqrt(np.mean(r_m0 * r_m0))),
            "sigma_plan": _robust_sigma(r_plan), "sigma_m0": _robust_sigma(r_m0),
        })
        resid_plan_all.append(r_plan)
        resid_m0_all.append(r_m0)
        by_orientation[obs.orientation]["plan"].append(r_plan)
        by_orientation[obs.orientation]["m0"].append(r_m0)

    def _agg(chunks: list[np.ndarray]) -> float:
        if not chunks:
            return float("nan")
        return _robust_sigma(np.concatenate(chunks))

    r_plan = _agg(resid_plan_all)
    r_m0 = _agg(resid_m0_all)
    ratio = (r_plan / r_m0) if (np.isfinite(r_m0) and r_m0 > 0) else float("inf")

    orientation_ratio: dict[str, float] = {}
    for o, d in by_orientation.items():
        rp, rm = _agg(d["plan"]), _agg(d["m0"])
        orientation_ratio[o] = (rp / rm) if (np.isfinite(rm) and rm > 0) else float("inf")

    return {
        "r_plan": r_plan,
        "r_offset": r_m0,
        "ratio": ratio,
        "per_edge": per_edge,
        "orientation_ratio": orientation_ratio,
    }


# ---------------------------------------------------------------------------
# Model-mismatch detector (non-planar / level-dependent signature, contract §5)
# ---------------------------------------------------------------------------

def detect_mismatch(
    observations: Sequence[EdgeObservation],
    cfg: GateConfig,
    *,
    coeffs: np.ndarray | None = None,
    index: Mapping[str, int] | None = None,
) -> tuple[bool, dict]:
    """Detect additive-plane MODEL_MISMATCH.

    Two independent signatures:

    1. **In-band non-planarity.** Within an edge band, a MULTIPLICATIVE or
       NONLINEAR per-Cell error leaves a residual correlated with the signal
       LEVEL or with radial curvature. Flagged when a majority of usable edges
       show this, or the median signature is strong.
    2. **Cross-edge level dependence.** A per-Cell gain/transparency/spectral
       mismatch scales the residual with the SKY LEVEL seen by the edge. Across a
       canvas whose sky level varies (diffuse IFN/nebulosity), the fitted edge
       offset residual (``theta_0 - (a_v - a_u)``) then correlates with the edge's
       mean level. A pure ADDITIVE per-Cell error does NOT (the common sky cancels
       in the difference). Requires the solved coefficients.

    Identifiability limit (documented): a purely multiplicative mismatch under a
    LOCALLY planar sky with NO global level variation is mathematically
    indistinguishable from an additive plane shift; the detector then correctly
    stays silent. No fabricated mismatch is ever reported.
    """
    usable = [o for o in observations if o.effective_train >= 64 and o.robust_sigma > 0.0]
    if not usable:
        return False, {"usable_edges": 0, "flagged_fraction": 0.0, "median_z": 0.0, "cross_z": 0.0}
    z = np.array([max(o.level_z, o.curvature_z) for o in usable], dtype=np.float64)
    flagged = z >= cfg.mismatch_level_z
    frac = float(np.mean(flagged))
    med = float(np.median(z))
    inband = bool(frac >= cfg.mismatch_min_fraction or med >= cfg.mismatch_level_z)

    cross_z = 0.0
    cross = False
    if coeffs is not None and index is not None and len(usable) >= 4:
        lvl = np.array([o.level_mean for o in usable], dtype=np.float64)
        off_res = np.array(
            [
                o.theta_global[0]
                - (coeffs[index[o.v], o.channel, 0] - coeffs[index[o.u], o.channel, 0])
                for o in usable
            ],
            dtype=np.float64,
        )
        sig = float(np.median([o.robust_sigma for o in usable]))
        lvl_c = lvl - float(np.mean(lvl))
        denom = float(np.sum(lvl_c * lvl_c))
        spread = float(np.percentile(lvl, 95) - np.percentile(lvl, 5))
        if denom > 0.0 and sig > 0.0:
            slope = float(np.sum(lvl_c * off_res) / denom)
            cross_z = abs(slope) * spread / sig
            cross = cross_z >= cfg.mismatch_level_z

    mismatch = bool(inband or cross)
    return mismatch, {
        "usable_edges": len(usable),
        "flagged_fraction": frac,
        "median_z": med,
        "max_z": float(np.max(z)),
        "inband": inband,
        "cross_z": cross_z,
        "cross": cross,
    }


# ---------------------------------------------------------------------------
# Decision (§6.1 evidence/numerics, §6.2 held-out)
# ---------------------------------------------------------------------------

def decide_state(*, connected: bool, mismatch: bool, gates: Mapping[str, str]) -> str:
    """Map predeclared gates to the contract decision state.

    Precedence: DISCONNECTED > INSUFFICIENT_EVIDENCE (support/leverage) >
    MODEL_MISMATCH > OFFSET_ONLY (M0 retained: plan offers no held-out gain) >
    ILL_CONDITIONED (real gain but numerically untrustworthy) > ROBUST_PLAN.
    """
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


def _corner_gate(
    corner_unc: np.ndarray | None,
    coeffs: np.ndarray,
    index: Mapping[str, int],
    sigma_delta: float,
    cfg: GateConfig,
) -> str:
    if corner_unc is None or not np.isfinite(sigma_delta) or sigma_delta <= 0.0:
        return "NOT_RUN"
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    n, channels, _ = corner_unc.shape
    for i in range(n):
        for c in range(channels):
            corr = coeffs[i, c]
            if not np.all(np.isfinite(corr)):
                continue
            for ci, (cx, cy) in enumerate(corners):
                unc = corner_unc[i, c, ci]
                if not np.isfinite(unc):
                    return "FAIL"
                magnitude = abs(corr[0] + corr[1] * cx + corr[2] * cy)
                if unc > cfg.corner_unc_sigma_frac * sigma_delta and (
                    magnitude <= 0.0 or unc > cfg.corner_unc_correction_frac * magnitude
                ):
                    return "FAIL"
    return "PASS"


def _connected_component_check(
    cell_ids: Sequence[str], observations: Sequence[EdgeObservation], anchor: str
) -> tuple[bool, list[str]]:
    adj: dict[str, set] = {c: set() for c in cell_ids}
    for o in observations:
        adj[o.u].add(o.v)
        adj[o.v].add(o.u)
    seen = {anchor}
    stack = [anchor]
    while stack:
        cur = stack.pop()
        for nb in sorted(adj[cur]):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    missing = sorted(set(cell_ids) - seen)
    return (len(missing) == 0), missing


# ---------------------------------------------------------------------------
# Application (ADDITIVE-ONLY, contract §6.3.7/8)
# ---------------------------------------------------------------------------

def plane_field(
    bounds: tuple[int, int, int, int],
    canvas_hw: tuple[int, int],
    coeffs: Sequence[float],
) -> np.ndarray:
    """The additive plane ``a + b*x + d*y`` over a patch (H, W), normalized coords."""
    xs, ys = normalized_coords(bounds, canvas_hw)
    a, b, d = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return a + b * xs[None, :] + d * ys[:, None]


def apply_plane_correction(
    science: np.ndarray,
    coeffs,
    bounds: tuple[int, int, int, int],
    canvas_hw: tuple[int, int],
) -> np.ndarray:
    """Add the per-channel plane to ``science`` (ADDITIVE-ONLY).

    Preserves NaN/non-finite locations, array dimensions and dtype; never applies
    a gain. ``coeffs`` is ``(3,)`` (shared) or ``(C, 3)``.
    """
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
# Nonlinear candidate — DOCUMENTATION / RESEARCH only (no engine)
# ---------------------------------------------------------------------------

def nonlinear_study() -> dict:
    """B3 nonlinear/spline study — documentation only, NO engine (contract §1)."""
    return {
        "model": MODEL_NONLINEAR_CANDIDATE,
        "status": "research_only_no_engine",
        "references": [
            "PhotometricMosaic (John Murphy): photometric scale matching + surface spline for relative gradients",
            "Montage (Caltech/IPAC): planar offset/slope matching with explicit calibration assumptions",
        ],
        "scientific_position": (
            "A per-Cell spline could absorb real sky/IFN and is only identifiable with "
            "independent calibration; it is NOT executable here. Any nonlinear residual "
            "observed by the additive prototype is reported as MODEL_MISMATCH needing "
            "research/upstream calibration, never repaired by an additive plane."
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
    """Complete M0/M1 comparison payload (JSON-safe via :meth:`to_json_dict`)."""

    decision: str
    model: str
    cell_ids: tuple[str, ...]
    anchor: str
    channels: int
    offset_offsets: np.ndarray           # (N, C) M0
    plan_coefficients: np.ndarray        # (N, C, 3) M1
    reg_lambda: float
    regularization_ladder: tuple[float, ...]
    heldout: dict
    gates: dict
    rank_data: int | None
    cond_data: float | None
    coef_uncertainty: np.ndarray | None
    corner_uncertainty: np.ndarray | None
    edge_stats: list[dict]
    mismatch: dict
    sigma_delta: float
    disconnected: tuple[str, ...]
    truth_diagnostics: dict
    nonlinear_study: dict
    provenance: dict
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
            "disconnected": list(self.disconnected),
            "truth_diagnostics": self.truth_diagnostics,
            "offset_offsets": np.asarray(self.offset_offsets).tolist(),
            "plan_coefficients": np.asarray(self.plan_coefficients).tolist(),
            "nonlinear_study": self.nonlinear_study,
            "provenance": dict(self.provenance),
            "notes": list(self.notes),
        }


def _edge_stats_json(o: EdgeObservation) -> dict:
    info = 0.5 * (o.information + o.information.T) if o.information.size else o.information
    return {
        "u": o.u, "v": o.v, "channel": o.channel, "orientation": o.orientation,
        "n_train": o.n_train, "n_holdout": o.n_holdout,
        "effective_train": o.effective_train,
        "tangent_blocks": o.tangent_blocks, "normal_blocks": o.normal_blocks,
        "robust_sigma": o.robust_sigma, "rejection_fraction": o.rejection_fraction,
        "theta_global": o.theta_global.tolist(),
        "rank_information": int(np.linalg.matrix_rank(o.information)) if o.information.size else 0,
        "leverage_max": (float(np.max(np.linalg.eigvalsh(info))) if info.size else 0.0),
        "level_z": o.level_z, "curvature_z": o.curvature_z,
        "refusal_reason": None,
    }


def gauge_truth_error(
    truth_planes: Mapping[str, object],
    coeffs: np.ndarray,
    cell_ids: Sequence[str],
    anchor: str,
    channels: int,
) -> dict:
    """Max/median error of the recovered CORRECTION vs the synthetic truth.

    The fitted coefficients are the additive CORRECTION ``P_i`` that cancels each
    Cell's own plane error. Since ``P_i = -(plane_i - plane_anchor)`` (the anchor
    keeps its plane), the expected correction is ``plane_anchor - plane_i``.
    """
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
    return {
        "max_abs_error": float(np.max(err)) if err.size else 0.0,
        "median_abs_error": float(np.median(err)) if err.size else 0.0,
    }


def compare_models(
    cells: Mapping[str, PlanCell],
    nx: int,
    ny: int,
    canvas_hw: tuple[int, int],
    *,
    channels: int | None = None,
    anchor: str | None = None,
    measurement_px: int | None = DEFAULT_MEASUREMENT_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
    block_px: int = BLOCK_PX,
    cfg: GateConfig | None = None,
    truth_planes: Mapping[str, object] | None = None,
) -> ModelComparisonResult:
    """Evaluate M0 (``OFFSET_ONLY``) vs M1 (``ROBUST_PLAN``) on one corpus.

    Returns fitted coefficients for BOTH models plus the predeclared gate outcomes
    and the resulting contract decision. ``truth_planes`` (optional) maps cell_id
    -> true ``(3,)`` or ``(C, 3)`` plane, producing gauge-fixed recovery diagnostics.
    """
    cfg = cfg or GateConfig()
    if channels is None:
        channels = cells[next(iter(cells))].channels()
    cell_ids = tuple(
        f"r{row:04d}c{col:04d}"
        for row in range(ny)
        for col in range(nx)
        if f"r{row:04d}c{col:04d}" in cells
    )
    if not cell_ids:
        raise ValueError("no cells provided")
    if anchor is None:
        anchor = select_central_cell(nx, ny)
        if anchor not in cells:
            anchor = cell_ids[0]
    index = {cid: i for i, cid in enumerate(cell_ids)}

    obs = build_observations(
        cells, nx, ny, canvas_hw, channels=channels,
        block_px=block_px, measurement_px=measurement_px,
    )
    connected, disconnected = _connected_component_check(cell_ids, obs, anchor)
    if not connected:
        n = len(cell_ids)
        prov = dict(PROVENANCE)
        prov.update({"measurement_px": measurement_px, "feather_px": feather_px,
                     "block_px": block_px, "channels": channels, "n_cells": n,
                     "n_edges_channel": len(obs), "anchor": anchor})
        return ModelComparisonResult(
            decision=STATE_DISCONNECTED,
            model=MODEL_OFFSET_ONLY,
            cell_ids=cell_ids,
            anchor=anchor,
            channels=channels,
            offset_offsets=np.zeros((n, channels), dtype=np.float64),
            plan_coefficients=np.zeros((n, channels, 3), dtype=np.float64),
            reg_lambda=0.0,
            regularization_ladder=REGULARIZATION_LADDER,
            heldout={"r_plan": float("nan"), "r_offset": float("nan"), "ratio": float("inf"),
                     "per_edge": [], "orientation_ratio": {}},
            gates={"evidence": "FAIL", "numerics": "NOT_RUN", "heldout": "NOT_RUN",
                   "connected": "FAIL", "rank": "NOT_RUN", "condition": "NOT_RUN",
                   "corner_uncertainty": "NOT_RUN"},
            rank_data=None, cond_data=None, coef_uncertainty=None, corner_uncertainty=None,
            edge_stats=[_edge_stats_json(o) for o in obs],
            mismatch={"usable_edges": 0, "reason": "disconnected"},
            sigma_delta=float("nan"),
            disconnected=tuple(disconnected),
            truth_diagnostics={},
            nonlinear_study=nonlinear_study(),
            provenance=prov,
            notes=("graph disconnected: no fabricated bridge; M0 retained elsewhere",),
        )
    m0 = solve_offset_baseline(obs, cell_ids, anchor, channels)

    # --- regularization ladder selection: train/holdout ONLY, never truth ---
    ladder_scores: dict[float, float] = {}
    for lam in REGULARIZATION_LADDER:
        sol = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=lam, with_uncertainty=False)
        ladder_scores[lam] = heldout_metrics(obs, sol.coefficients, m0, index, channels)["r_plan"]
    finite = {k: v for k, v in ladder_scores.items() if np.isfinite(v)}
    if finite:
        min_score = min(finite.values())
        best_lambda = max(k for k, v in finite.items() if v <= min_score * (1.0 + 1e-9))
    else:
        best_lambda = 0.0

    plan = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=best_lambda, with_uncertainty=True)
    hm = heldout_metrics(obs, plan.coefficients, m0, index, channels)

    sigma_delta = float(np.median([o.robust_sigma for o in obs])) if obs else float("nan")
    mismatch, mismatch_info = detect_mismatch(obs, cfg, coeffs=plan.coefficients, index=index)

    # --- §6.1 evidence gates ---
    eff_ok = bool(obs) and all(o.effective_train >= cfg.min_effective_train for o in obs)
    blocks_ok = bool(obs) and all(
        o.tangent_blocks >= cfg.min_tangent_blocks and o.normal_blocks >= cfg.min_normal_blocks
        for o in obs
    )
    evidence = "PASS" if (eff_ok and blocks_ok and connected) else "FAIL"

    # --- §6.1 numerics gates ---
    finite_unc = True
    if plan.coef_uncertainty is not None:
        finite_unc = bool(np.all(np.isfinite(plan.coef_uncertainty)))
    rank_ok = plan.rank_data is not None and plan.rank_data == plan.unknowns
    cond_ok = plan.cond_data is not None and np.isfinite(plan.cond_data) and plan.cond_data <= cfg.cond_a_max
    corner = _corner_gate(plan.corner_uncertainty, plan.coefficients, index, sigma_delta, cfg)
    numerics = "PASS" if (rank_ok and cond_ok and finite_unc and corner == "PASS") else "FAIL"

    # --- §6.2 held-out gates ---
    r_plan = hm["r_plan"]
    r_off = hm["r_offset"]
    ratio_ok = bool(np.isfinite(r_plan) and np.isfinite(r_off) and r_off > 0 and r_plan <= cfg.heldout_ratio_max * r_off)
    abs_ok = bool(np.isfinite(r_plan) and np.isfinite(sigma_delta) and sigma_delta > 0 and r_plan <= cfg.heldout_abs_sigma_max * sigma_delta)
    orient_ok = all((not np.isfinite(v)) or v <= cfg.heldout_ratio_max for v in hm["orientation_ratio"].values())
    floor = 0.05 * (sigma_delta if np.isfinite(sigma_delta) else 0.0)
    p90_ok = all(e["p90_plan"] <= 1.05 * e["p90_m0"] + floor for e in hm["per_edge"])
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

    truth_diag: dict = {}
    if truth_planes is not None:
        truth_diag = gauge_truth_error(truth_planes, plan.coefficients, cell_ids, anchor, channels)

    decision = decide_state(connected=connected, mismatch=mismatch, gates=gates)
    if decision == STATE_ROBUST_PLAN and not (
        gates["heldout"] == "PASS" and gates["numerics"] == "PASS" and gates["evidence"] == "PASS"
    ):
        decision = STATE_OFFSET_ONLY

    provenance = dict(PROVENANCE)
    provenance.update({
        "measurement_px": measurement_px,
        "feather_px": feather_px,
        "block_px": block_px,
        "channels": channels,
        "n_cells": len(cell_ids),
        "n_edges_channel": len(obs),
        "anchor": anchor,
        "selected_reg_lambda": best_lambda,
        "ladder_scores": {str(k): (None if not np.isfinite(v) else float(v)) for k, v in ladder_scores.items()},
        "sigma_delta": None if not np.isfinite(sigma_delta) else float(sigma_delta),
    })
    result = ModelComparisonResult(
        decision=decision,
        model=(MODEL_ROBUST_PLAN if decision == STATE_ROBUST_PLAN else MODEL_OFFSET_ONLY),
        cell_ids=cell_ids,
        anchor=anchor,
        channels=channels,
        offset_offsets=m0,
        plan_coefficients=plan.coefficients,
        reg_lambda=plan.reg_lambda,
        regularization_ladder=REGULARIZATION_LADDER,
        heldout=hm,
        gates=gates,
        rank_data=plan.rank_data,
        cond_data=plan.cond_data,
        coef_uncertainty=plan.coef_uncertainty,
        corner_uncertainty=plan.corner_uncertainty,
        edge_stats=[_edge_stats_json(o) for o in obs],
        mismatch=mismatch_info,
        sigma_delta=sigma_delta,
        disconnected=tuple(disconnected),
        truth_diagnostics=truth_diag,
        nonlinear_study=nonlinear_study(),
        provenance=provenance,
    )
    return result


# ---------------------------------------------------------------------------
# Scalability witness (contract §9) — compact sparse graph, NO images
# ---------------------------------------------------------------------------

def sparse_scale_witness(
    side: int,
    *,
    channels: int = 1,
    reg_lambda: float = 1e-3,
    measure: bool = True,
) -> dict:
    """Pure sparse graph construction/solve witness up to 10,000 Cells (§9).

    Builds COMPACT synthetic observations (no images, no FITS, no `N_frames×canvas`)
    on a uniform ``side × side`` grid and solves the same sparse system as M1.
    Returns Cell/edge/row counts, unknown count, timing and peak RSS. Never
    densifies the design.
    """
    import resource
    import time

    from .cell_offset import grid_adjacency

    nx = ny = int(side)
    cell_ids = tuple(f"r{r:04d}c{c:04d}" for r in range(ny) for c in range(nx))
    info = np.diag([16.0, 16.0, 4.0])  # weaker normal slope, like the real band
    obs: list[EdgeObservation] = []
    for (u, v) in grid_adjacency(nx, ny):
        (ur, uc) = _parse_rc(u)
        (vr, vc) = _parse_rc(v)
        theta = np.array([
            0.05 * (((ur + uc) % 11) - 5),
            0.01 * (((ur * 7 + uc * 3) % 9) - 4),
            0.01 * (((ur * 5 + uc * 13) % 9) - 4),
        ])
        obs.append(EdgeObservation(
            u=u, v=v, channel=0,
            orientation=("h" if vc > uc else "v"),
            n_train=256, n_holdout=0, effective_train=256.0,
            tangent_blocks=8, normal_blocks=4,
            robust_sigma=1.0, rejection_fraction=0.0,
            theta_global=theta,
            information=info.copy(),
            rhs=info @ theta,
            level_z=0.0, curvature_z=0.0, level_mean=0.0, local_offset=theta[0],
            holdout_xy=np.zeros((0, 2)), holdout_diff=np.zeros(0),
        ))
    anchor = select_central_cell(nx, ny)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if measure else 0
    t0 = time.perf_counter()
    plan = solve_plan_sparse(obs, cell_ids, anchor, channels, reg_lambda=reg_lambda, with_uncertainty=False)
    dt = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if measure else 0
    return {
        "cells": len(cell_ids),
        "edges": len(obs),
        "unknowns": plan.unknowns,
        "regularized_rows": plan.rows,
        "rank_data": plan.rank_data,
        "cond_data": plan.cond_data,
        "solve_seconds": dt,
        "rss_delta_kib": int(max(rss1 - rss0, 0)),
        "rss_peak_kib": int(rss1),
        "dense_design_bytes_estimate": int(plan.rows * plan.unknowns * 8) if plan.unknowns else 0,
        "method": "scipy.sparse.linalg.lsqr_on_whitened_design",
        "densified": False,
    }
