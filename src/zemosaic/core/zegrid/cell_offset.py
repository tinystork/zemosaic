"""ZM-CELL-PHOTOMETRY-R34C — inter-Cell ADDITIVE offset graph (pure, opt-in).

Given per-Cell self-anchored gauges (R34b), each Cell has its OWN photometric
level (its reference frame's sky). Neighbouring Cells overlap only in their halo
strip (~2x ``HALO_PX`` px), so the mosaic shows small inter-Cell LEVEL steps.

This module is the PURE building block that removes those steps with a SINGLE
per-channel ADDITIVE offset per Cell:

* **Nodes** = Cells (ZeGrid cell ids, e.g. ``r0000c0000``).
* **Edges** = 4-neighbourhood adjacency (adjacent Cells sharing a halo strip).
* **Observation** = a robust sky-difference between the two Cells measured in
  their shared overlap (per channel), mirroring the legacy Phase-5 inter-tile
  photometric matching (``intertile_sky_percentile`` band + ``intertile_robust_
  clip_sigma`` clipping).
* **Solve** = WEIGHTED LEAST SQUARES of the additive offsets with the CENTRAL
  Cell pinned to 0 (``o_anchor = 0``) — a GLOBAL solve, never a sequential chain,
  so no drift accumulates along a path.
* **Anchor** = the Cell nearest the layout centre (deterministic; ties -> lowest
  cell id).

ADDITIVE-ONLY. The model is ``science_cell += offset_cell`` (a pure level shift,
one value per Cell per channel). There is NO per-Cell gain, plane, or any
multiplicative term — a pure level shift is therefore recovered EXACTLY.

Reference STABILITY (R34b's Nono finding I1): the per-Cell gauge result carries a
:class:`~zemosaic.core.zegrid.photometric.CellReferencePin` (reference frame id +
support count), so the level being harmonised is reproducible; a recompute cannot
silently flip a Cell's best frame and shift its level.

Scope: this module is a research-quality pure block. It does NOT wire into
production, does NOT touch the frozen ``canonical_*`` modules, adds NO feathering
(R34d), and changes NO default (R34b's ``cell_gauge_mode`` default stays
``"global"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_SKY_PERCENTILE",
    "DEFAULT_ROBUST_CLIP_SIGMA",
    "MAD_SCALE",
    "robust_sky_level",
    "edge_offset_observation",
    "grid_cell_ids",
    "grid_adjacency",
    "select_central_cell",
    "OffsetEdge",
    "CellOffsetSolution",
    "DisconnectedOffsetGraphError",
    "solve_additive_offsets",
    "chain_offsets",
]

# Robust sky estimation parameters — MIRROR the legacy Phase-5 inter-tile
# photometric matching defaults (``intertile_sky_percentile`` = [30, 70],
# ``intertile_robust_clip_sigma`` = 2.5). ``MAD_SCALE`` converts a MAD to a
# Gaussian-equivalent sigma.
DEFAULT_SKY_PERCENTILE: tuple[float, float] = (30.0, 70.0)
DEFAULT_ROBUST_CLIP_SIGMA: float = 2.5
MAD_SCALE: float = 1.4826


# ---------------------------------------------------------------------------
# Robust sky observation (mirrors legacy intertile_sky_percentile + clip)
# ---------------------------------------------------------------------------

def _as_finite_1d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr[np.isfinite(arr)]


def robust_sky_level(
    values,
    *,
    percentile: tuple[float, float] = DEFAULT_SKY_PERCENTILE,
    clip_sigma: float = DEFAULT_ROBUST_CLIP_SIGMA,
) -> tuple[float, int]:
    """Robust sky LEVEL of a pixel value set (percentile band + MAD clipping).

    Mirrors the legacy inter-tile method: keep the ``[p_lo, p_hi]`` percentile
    band (default 30-70), clip the band at ``clip_sigma`` robust sigma around its
    median (MAD-based), and return the median of the surviving values. Returns
    ``(nan, 0)`` when no finite sample exists (explicit — never fabricated).

    Deterministic: pure float64 reductions, no RNG, no order dependence.
    """
    v = _as_finite_1d(values)
    if v.size == 0:
        return float("nan"), 0
    lo, hi = float(percentile[0]), float(percentile[1])
    if lo > hi:
        lo, hi = hi, lo
    p_lo, p_hi = np.percentile(v, [lo, hi])
    band = v[(v >= p_lo) & (v <= p_hi)]
    if band.size == 0:
        band = v
    med = float(np.median(band))
    mad = float(np.median(np.abs(band - med)))
    sigma = MAD_SCALE * mad
    if sigma > 0.0:
        keep = band[np.abs(band - med) <= float(clip_sigma) * sigma]
        if keep.size > 0:
            band = keep
    return float(np.median(band)), int(band.size)


def edge_offset_observation(
    plane_u,
    plane_v,
    mask=None,
    *,
    percentile: tuple[float, float] = DEFAULT_SKY_PERCENTILE,
    clip_sigma: float = DEFAULT_ROBUST_CLIP_SIGMA,
    reject: float | None = 6.0,
) -> tuple[float, int, float]:
    """Robust per-channel additive offset between two overlapping planes.

    Returns ``(delta, n_overlap, weight)`` where ``delta`` is the robust
    ``sky(plane_u) - sky(plane_v)`` measured on the shared VALID pixels (the value
    ``o_v - o_u`` the solver must satisfy), ``n_overlap`` is the number of shared
    valid pixels, and ``weight`` is the default least-squares weight (the overlap
    pixel count — deterministic, support-driven).

    ``plane_u`` / ``plane_v`` are ``(H, W)`` (mono) or ``(H, W, C)`` (per channel)
    float arrays cropped to the SAME overlap region. ``mask`` (optional) is a 2-D
    bool restricting the overlap. ``reject`` gives an additional pre-clip of
    outlier differences (None disables it). Pixels where either plane is
    non-finite are ignored.
    """
    u = np.asarray(plane_u, dtype=np.float64)
    v = np.asarray(plane_v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"plane shapes differ: {u.shape} vs {v.shape}")
    if u.ndim == 2:
        channels = 1
        u = u[:, :, None]
        v = v[:, :, None]
    elif u.ndim == 3:
        channels = u.shape[2]
    else:
        raise ValueError(f"planes must be 2-D or 3-D, got ndim={u.ndim}")

    h, w = u.shape[:2]
    if mask is None:
        valid = np.ones((h, w), dtype=bool)
    else:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != (h, w):
            raise ValueError(f"mask shape {valid.shape} != plane shape {(h, w)}")

    deltas = np.empty(channels, dtype=np.float64)
    counts = np.empty(channels, dtype=np.int64)
    for c in range(channels):
        uc = u[:, :, c]
        vc = v[:, :, c]
        sel = valid & np.isfinite(uc) & np.isfinite(vc)
        diff = (uc[sel] - vc[sel]) if sel.any() else np.empty(0, dtype=np.float64)
        if reject is not None and diff.size >= 8:
            # One robust pre-clip of the difference cloud (outlier pixels).
            med = float(np.median(diff))
            mad = float(np.median(np.abs(diff - med)))
            sigma = MAD_SCALE * mad
            if sigma > 0.0:
                diff = diff[np.abs(diff - med) <= float(reject) * sigma]
        delta, n = robust_sky_level(
            diff, percentile=percentile, clip_sigma=clip_sigma
        )
        deltas[c] = delta
        counts[c] = n

    if channels == 1:
        n_total = int(counts[0])
        return float(deltas[0]), n_total, float(n_total)
    return deltas, counts, counts.astype(np.float64)


# ---------------------------------------------------------------------------
# Layout topology (deterministic grid + 4-neighbourhood adjacency + anchor)
# ---------------------------------------------------------------------------

def grid_cell_ids(nx: int, ny: int) -> tuple[tuple[int, int, str], ...]:
    """Row-major ``(row, col, cell_id)`` for an ``nx`` x ``ny`` layout."""
    if nx < 1 or ny < 1:
        raise ValueError("nx/ny must be >= 1")
    return tuple(
        (row, col, f"r{row:04d}c{col:04d}")
        for row in range(ny)
        for col in range(nx)
    )


def grid_adjacency(nx: int, ny: int) -> tuple[tuple[str, str], ...]:
    """4-neighbourhood edges of an ``nx`` x ``ny`` grid (right + down neighbours).

    Deterministic order (row-major, right before down). Adjacent Cells share a
    halo strip (~2x ``HALO_PX`` px) across their common core edge.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx/ny must be >= 1")
    edges: list[tuple[str, str]] = []
    for row in range(ny):
        for col in range(nx):
            u = f"r{row:04d}c{col:04d}"
            if col + 1 < nx:
                edges.append((u, f"r{row:04d}c{col + 1:04d}"))
            if row + 1 < ny:
                edges.append((u, f"r{row + 1:04d}c{col:04d}"))
    return tuple(edges)


def select_central_cell(nx: int, ny: int) -> str:
    """The anchor Cell = nearest the layout centre (ties -> lowest cell id).

    Centre is the geometric centre of the cell-index rectangle
    ``((nx-1)/2, (ny-1)/2)``; distance is the squared Euclidean distance of the
    cell index to that centre. Deterministic.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx/ny must be >= 1")
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    best = None
    best_key = None
    for row, col, cid in grid_cell_ids(nx, ny):
        key = ((row - cy) ** 2 + (col - cx) ** 2, cid)
        if best_key is None or key < best_key:
            best_key = key
            best = cid
    return best


# ---------------------------------------------------------------------------
# Additive offset solve (weighted least squares, central anchor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OffsetEdge:
    """One adjacency observation: ``o_v - o_u == delta`` (per channel), weight w.

    ``delta`` is the robust sky difference measured in the shared overlap; the
    solver must place ``o_v - o_u`` at that value. ``weight`` may be a scalar or a
    per-channel ``(C,)`` array (least-squares weight — same units as 1/variance).
    """

    u: str
    v: str
    delta: float | Sequence[float]
    weight: float | Sequence[float] = 1.0
    n_overlap: int | None = None


@dataclass(frozen=True)
class CellOffsetSolution:
    """Global additive offset solution (one additive level per Cell per channel)."""

    anchor: str
    cell_ids: tuple[str, ...]
    offsets: np.ndarray          # (N, C) float64; the anchor row is exactly 0.
    channels: int
    channel_offsets: dict        # convenience: {cell_id: (C,) ndarray}
    weighted_residual_rms: np.ndarray  # (C,) float64
    edges_used: int
    components: int              # ALWAYS 1 (disconnected graphs raise).


class DisconnectedOffsetGraphError(ValueError):
    """The offset graph has a component not reachable from the anchor."""


def _normalize_edges(edges: Iterable[OffsetEdge], channels: int):
    out = []
    for e in edges:
        if not isinstance(e, OffsetEdge):
            raise TypeError(f"expected OffsetEdge, got {type(e).__name__}")
        d = np.asarray(e.delta, dtype=np.float64).ravel()
        if d.size == 1 and channels > 1:
            d = np.repeat(d, channels)
        if d.size != channels:
            raise ValueError(
                f"edge ({e.u},{e.v}) delta has {d.size} values, expected {channels}"
            )
        w = np.asarray(e.weight, dtype=np.float64).ravel()
        if w.size == 1:
            w = np.repeat(w, channels)
        if w.size != channels:
            raise ValueError(
                f"edge ({e.u},{e.v}) weight has {w.size} values, expected {channels}"
            )
        if np.any(w < 0.0):
            raise ValueError(f"edge ({e.u},{e.v}) has a negative weight")
        out.append((e.u, e.v, d, w))
    return out


def _connected_from(anchor: str, nodes: Sequence[str], edges) -> set:
    adj: dict[str, set] = {n: set() for n in nodes}
    for (u, v, _d, _w) in edges:
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


def solve_additive_offsets(
    cell_ids: Sequence[str],
    edges: Iterable[OffsetEdge],
    anchor: str,
    channels: int,
    *,
    raise_on_disconnected: bool = True,
) -> CellOffsetSolution:
    """Solve the GLOBAL additive offset graph (weighted least squares).

    Minimises ``sum_e sum_c w_e,c * (o_v - o_u - delta_e,c)**2`` subject to
    ``o_anchor = 0`` (the CENTRAL Cell). The anchor level is FIXED, so the solve is
    global (least squares over ALL edges): no sequential accumulation, hence no
    drift along a path (unlike a chain solve).

    ADDITIVE-ONLY: one scalar offset per Cell per channel, no gain/plane.

    Args:
        cell_ids: the Cell ids (nodes). Order is preserved in the result; the
            anchor must be one of them.
        edges: :class:`OffsetEdge` observations (``o_v - o_u == delta``).
        anchor: the anchor Cell id (offset forced to 0).
        channels: number of channels (1 for mono, 3 for RGB).

    Raises:
        ValueError: malformed input (unknown endpoint, bad channels, missing
            anchor, non-finite observation).
        DisconnectedOffsetGraphError: a node is not reachable from the anchor
            (explicit failure — never fabricate an offset for an unreachable Cell).
    """
    nodes = list(cell_ids)
    if not nodes:
        raise ValueError("cell_ids is empty")
    if len(set(nodes)) != len(nodes):
        raise ValueError("cell_ids contains duplicates")
    if not isinstance(channels, (int, np.integer)) or isinstance(channels, bool) or channels < 1:
        raise ValueError(f"channels must be an int >= 1, got {channels!r}")
    channels = int(channels)
    if anchor not in set(nodes):
        raise ValueError(f"anchor {anchor!r} is not among the cells")

    norm_edges = _normalize_edges(edges, channels)
    node_set = set(nodes)
    for (u, v, d, _w) in norm_edges:
        if u not in node_set or v not in node_set:
            raise ValueError(f"edge ({u!r},{v!r}) references an unknown cell")
        if u == v:
            raise ValueError(f"self-edge on cell {u!r}")
        if not np.all(np.isfinite(d)):
            raise ValueError(f"edge ({u!r},{v!r}) has a non-finite observation")

    # Connectivity: every node must be reachable from the anchor.
    reachable = _connected_from(anchor, nodes, norm_edges)
    if len(reachable) != len(nodes):
        missing = sorted(node_set - reachable)
        if raise_on_disconnected:
            raise DisconnectedOffsetGraphError(
                "offset graph is disconnected: "
                f"{len(missing)} cell(s) unreachable from anchor {anchor!r}: "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
        # Explicit placeholder: unreachable nodes keep offset 0 (only used by
        # callers that opt into tolerant behaviour).
        missing = sorted(node_set - reachable)
    else:
        missing = []

    # --- weighted least squares, anchor eliminated (o_anchor == 0) ---
    index = {n: i for i, n in enumerate(nodes)}
    free = [n for n in nodes if n != anchor]
    free_index = {n: j for j, n in enumerate(free)}
    nf = len(free)
    offsets = np.zeros((len(nodes), channels), dtype=np.float64)

    if nf == 0:
        # Single-cell graph: the anchor is the only Cell.
        return CellOffsetSolution(
            anchor=anchor,
            cell_ids=tuple(nodes),
            offsets=offsets,
            channels=channels,
            channel_offsets={anchor: offsets[index[anchor]].copy()},
            weighted_residual_rms=np.zeros(channels, dtype=np.float64),
            edges_used=0,
            components=1,
        )

    n_edges = len(norm_edges)
    # Build the sparse structure once; weights are per (edge, channel).
    rows = np.empty((n_edges, 2), dtype=np.int64)
    for r, (u, v, _d, _w) in enumerate(norm_edges):
        rows[r, 0] = -1 if u == anchor else free_index[u]
        rows[r, 1] = -1 if v == anchor else free_index[v]

    residual_rms = np.zeros(channels, dtype=np.float64)
    for c in range(channels):
        delta_c = np.array([d[c] for (_u, _v, d, _w) in norm_edges], dtype=np.float64)
        w_c = np.array([w[c] for (_u, _v, _d, w) in norm_edges], dtype=np.float64)
        a = np.zeros((n_edges, nf), dtype=np.float64)
        for r in range(n_edges):
            cu = rows[r, 0]
            cv = rows[r, 1]
            if cv >= 0:
                a[r, cv] += 1.0
            if cu >= 0:
                a[r, cu] -= 1.0
        # Normal equations: (A^T W A) o = A^T W delta  (anchor terms moved to rhs
        # implicitly because the anchor column is dropped and o_anchor == 0).
        ata = a.T @ (w_c[:, None] * a)
        atb = a.T @ (w_c * delta_c)
        try:
            sol = np.linalg.solve(ata, atb)
        except np.linalg.LinAlgError:
            # A connected graph with positive weights is nonsingular; a singular
            # system means zero/absent weighting on a cut — fail loud, no fake.
            raise DisconnectedOffsetGraphError(
                "offset graph normal equations are singular (a zero-weight cut "
                "disconnects the graph)"
            )
        for n, j in free_index.items():
            offsets[index[n], c] = sol[j]
        # Weighted residual RMS.
        pred = a @ sol
        resid = pred - delta_c
        wsum = float(np.sum(w_c))
        residual_rms[c] = (
            float(np.sqrt(np.sum(w_c * resid * resid) / wsum)) if wsum > 0.0 else 0.0
        )
    offsets[index[anchor], :] = 0.0

    for n in missing:
        offsets[index[n], :] = 0.0

    channel_offsets = {n: offsets[index[n]].copy() for n in nodes}
    return CellOffsetSolution(
        anchor=anchor,
        cell_ids=tuple(nodes),
        offsets=offsets,
        channels=channels,
        channel_offsets=channel_offsets,
        weighted_residual_rms=residual_rms,
        edges_used=n_edges,
        components=1,
    )


def chain_offsets(
    cell_ids: Sequence[str],
    edges: Iterable[OffsetEdge],
    anchor: str,
    channels: int,
) -> np.ndarray:
    """SEQUENTIAL chain solve (accumulating) — for the drift comparison ONLY.

    Walks outward from ``anchor`` along BFS levels and accumulates each edge's
    observation. This is exactly the drift-prone construction the global solve
    replaces; it is provided so tests can show the chain accumulates error along a
    path while :func:`solve_additive_offsets` does not. Not for production.
    """
    nodes = list(cell_ids)
    index = {n: i for i, n in enumerate(nodes)}
    offsets = np.full((len(nodes), channels), np.nan, dtype=np.float64)
    offsets[index[anchor], :] = 0.0
    adj: dict[str, list] = {n: [] for n in nodes}
    for e in edges:
        d = np.asarray(e.delta, dtype=np.float64).ravel()
        if d.size == 1 and channels > 1:
            d = np.repeat(d, channels)
        adj[e.u].append((e.v, d, +1))
        adj[e.v].append((e.u, d, -1))
    seen = {anchor}
    stack = [anchor]
    while stack:
        cur = stack.pop(0)
        for (nb, d, sign) in adj[cur]:
            if nb in seen:
                continue
            # o_nb = o_cur + sign * delta  where delta = o_v - o_u
            offsets[index[nb], :] = offsets[index[cur], :] + sign * d
            seen.add(nb)
            stack.append(nb)
    return offsets
