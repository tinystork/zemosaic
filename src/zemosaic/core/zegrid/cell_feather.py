"""ZM-CELL-PHOTOMETRY-R34D — apply inter-Cell offsets + FEATHERED assembly.

This is the R34d end-to-end building block on top of R34b (per-Cell self-anchored
gauges) and R34c (pure inter-Cell additive offset graph):

    per-Cell gauge (R34b)
      -> per-Cell stack (existing)
      -> robust per-Cell sky + per-edge overlap sky difference (R34c observation)
      -> GLOBAL additive offset solve (R34c, central anchor, pinned references)
      -> apply the per-channel additive offset to each Cell's science
      -> FEATHERED assembly (weighted blend, weights sum to 1 in the halo band)
         replacing the single-owner hard cut.

Design decisions (Nono R34c inputs)
-----------------------------------
* **Additive-only (I4 / binding).** Each Cell's science gets ONE scalar added per
  channel (``science += offset[channel]``). No gain, plane, or multiplicative
  term anywhere.
* **Central anchor.** The offset graph is anchored on ``select_central_cell``
  (offset 0); the assembled result therefore sits on the central Cell's pinned
  reference level. The retained GLOBAL gauge pass is untouched (progress /
  provenance).
* **Edge weight (I1).** The default least-squares weight is the number of SHARED
  VALID overlap pixels (support-count, i.e. "area"). This mirrors the legacy
  Phase-5 inter-tile weighting (``intertile_prune_weight_mode`` = "area") and is
  deterministic; inverse-variance weighting is intentionally NOT used (it needs a
  per-edge variance estimate that is noisier than the support count on thin halo
  strips).
* **Shared-valid semantics (I2), FROZEN.** A pixel contributes to an edge
  observation iff it is valid in BOTH cells (``valid_mask``) AND its science is
  finite in both. The canonical ``valid_mask`` is channel-invariant and already
  encodes geometry + finiteness; object/star masking is upstream and, when
  available, is injected through the optional ``extra_mask`` argument. Same
  semantics are used for the assembly blend weight.
* **Disconnected policy (I3), EXPLICIT.** A Cell (or component) unreachable from
  the anchor is NEVER locally re-anchored and NEVER bridged with a fabricated
  observation. The default policy is to RAISE
  (:class:`~zemosaic.core.zegrid.cell_offset.DisconnectedOffsetGraphError`); a
  documented optional policy ``on_disconnected="zero"`` keeps such Cells at
  offset 0 (their own level, recorded), which is deterministic and fabricates
  nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import mosaic as zmosaic
from . import sweep as zsw
from .cell_offset import (
    CellOffsetSolution,
    DisconnectedOffsetGraphError,
    OffsetEdge,
    edge_offset_observation,
    grid_adjacency,
    robust_sky_level,
    select_central_cell,
    solve_additive_offsets,
)
from .geometry import GlobalCanvas, build_layout, cell_id

__all__ = [
    "DEFAULT_HALO_PX",
    "FEATHER_DISCONNECTED_POLICY",
    "CellPatch",
    "CellFeatherResult",
    "cell_patch_sky",
    "overlap_edge_observation",
    "build_cell_offset_graph",
    "apply_additive_offsets",
    "feather_weight_field",
    "shared_valid_mask",
    "assemble_canvas_feathered",
]

# Halo width (global px) — the per-Cell patch extends ``HALO_PX`` beyond the core.
DEFAULT_HALO_PX = zsw.HALO_PX

# Wired-path disconnected policy (Nono I3). "zero" = Cells unreachable from the
# anchor keep offset 0 (their own self-anchored level); deterministic, recorded,
# NO re-anchor, NO fabricated bridge. The pure R34c solver DEFAULT stays "raise".
FEATHER_DISCONNECTED_POLICY = "zero"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellPatch:
    """One Cell's PATCH-level planes (core + halo) at its global position.

    ``bounds`` is the global patch ``(y0, y1, x0, x1)``; ``core_slice`` is the
    patch-local core ``(y0, y1, x0, x1)``. ``science`` is ``(Hp, Wp, C)`` float
    (NaN outside the valid support); ``valid_mask`` is ``(Hp, Wp, C)`` bool.
    ``surviving`` (optional) is ``(Hp, Wp, C)`` per-channel stack depth.
    """

    cell_id: str
    bounds: tuple[int, int, int, int]
    core_slice: tuple[int, int, int, int]
    science: np.ndarray
    valid_mask: np.ndarray
    surviving: np.ndarray | None = None


@dataclass(frozen=True)
class CellFeatherResult:
    """Diagnostics for the per-Cell offset + feather assembly."""

    anchor: str
    offsets: np.ndarray               # (N, C) additive offsets (anchor row 0)
    cell_ids: tuple[str, ...]
    channels: int
    edges_used: int
    edges_built: int
    weighted_residual_rms: np.ndarray
    disconnected: tuple[str, ...]     # cells kept at offset 0 (on_disconnected="zero")
    policy: str
    per_cell_reference: dict = field(default_factory=dict)  # cell_id -> pinned ref id


# ---------------------------------------------------------------------------
# Robust observation (per-cell sky + per-edge overlap difference)
# ---------------------------------------------------------------------------

def _channels_of(plane) -> int:
    a = np.asarray(plane)
    if a.ndim == 2:
        return 1
    if a.ndim == 3:
        return int(a.shape[2])
    raise ValueError(f"plane must be 2-D or 3-D, got ndim={a.ndim}")


def shared_valid_mask(sci_a, val_a, sci_b, val_b, extra_mask=None) -> np.ndarray:
    """FROZEN shared-valid semantics (R34D, Nono I2).

    A pixel is shared-valid iff it is valid in BOTH cells (channel-invariant
    ``valid_mask``) AND finite in BOTH sciences (all channels). ``extra_mask``
    (2-D bool) can inject an upstream object/star mask. Returns a 2-D bool mask.
    """
    a = np.asarray(sci_a, dtype=np.float64)
    b = np.asarray(sci_b, dtype=np.float64)
    va = np.asarray(val_a, dtype=bool)
    vb = np.asarray(val_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"science shapes differ: {a.shape} vs {b.shape}")
    hw = a.shape[:2]
    fa = np.all(np.isfinite(a), axis=-1) if a.ndim == 3 else np.isfinite(a)
    fb = np.all(np.isfinite(b), axis=-1) if b.ndim == 3 else np.isfinite(b)
    va2 = va[..., 0] if va.ndim == 3 else va
    vb2 = vb[..., 0] if vb.ndim == 3 else vb
    mask = va2 & vb2 & fa & fb
    if extra_mask is not None:
        em = np.asarray(extra_mask, dtype=bool)
        if em.shape != hw:
            raise ValueError(f"extra_mask shape {em.shape} != {(hw)}")
        mask = mask & em
    return mask


def cell_patch_sky(patch: CellPatch, *, use_core=True, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-channel sky level of a Cell (core region by default).

    Returns ``(levels, counts)`` (C,) arrays, using the FROZEN robust observation
    (percentile band + MAD clip). ``use_core=False`` uses the whole patch.
    """
    sci = np.asarray(patch.science, dtype=np.float64)
    val = np.asarray(patch.valid_mask, dtype=bool)
    if val.ndim == 2:
        val = val[:, :, None]
    if use_core:
        y0, y1, x0, x1 = patch.core_slice
        sci = sci[y0:y1, x0:x1]
        val = val[y0:y1, x0:x1]
        if sci.ndim == 2:
            sci = sci[:, :, None]
    if sci.ndim == 2:
        sci = sci[:, :, None]
    c = sci.shape[2]
    levels = np.empty(c, dtype=np.float64)
    counts = np.empty(c, dtype=np.int64)
    for ch in range(c):
        vals = sci[:, :, ch][val[:, :, ch] & np.isfinite(sci[:, :, ch])]
        levels[ch], counts[ch] = robust_sky_level(vals, **kw)
    return levels, counts


def overlap_edge_observation(
    patch_a: CellPatch,
    patch_b: CellPatch,
    *,
    extra_mask=None,
    **kw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust per-channel overlap sky difference between two adjacent Cells.

    Crops both patches to their GLOBAL intersection (the halo band across the
    shared core edge) and returns ``(delta, n_shared, weight)`` (C,) arrays where
    ``delta = sky(patch_a) - sky(patch_b)`` on the SHARED VALID pixels (frozen
    semantics), ``n_shared`` is the shared-valid pixel count (channel-invariant
    here), and ``weight`` is the least-squares weight (support-count / area).
    """
    ay0, ay1, ax0, ax1 = patch_a.bounds
    by0, by1, bx0, bx1 = patch_b.bounds
    y0, y1 = max(ay0, by0), min(ay1, by1)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f"cells {patch_a.cell_id}/{patch_b.cell_id} patches do not overlap"
        )
    a_sci = np.asarray(patch_a.science)[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    a_val = np.asarray(patch_a.valid_mask)[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
    b_sci = np.asarray(patch_b.science)[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    b_val = np.asarray(patch_b.valid_mask)[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]
    mask = shared_valid_mask(a_sci, a_val, b_sci, b_val, extra_mask=extra_mask)
    delta, n, weight = edge_offset_observation(a_sci, b_sci, mask, **kw)
    c = _channels_of(a_sci)
    if c == 1:
        delta = np.asarray([delta], dtype=np.float64)
        n = np.asarray([n], dtype=np.int64)
        weight = np.asarray([weight], dtype=np.float64)
    return np.asarray(delta, np.float64), np.asarray(n, np.int64), np.asarray(weight, np.float64)


# ---------------------------------------------------------------------------
# Offset graph from Cell patches (adjacency + observations + solve)
# ---------------------------------------------------------------------------

def build_cell_offset_graph(
    canvas: GlobalCanvas,
    nx: int,
    ny: int,
    patches: Mapping[str, CellPatch],
    *,
    anchor: str | None = None,
    channels: int | None = None,
    on_disconnected: str = "raise",
    extra_mask=None,
    **obs_kw,
) -> CellFeatherResult:
    """Measure the inter-Cell observations over the halo overlaps and SOLVE.

    Nodes = the Cells present in ``patches``; edges = the 4-neighbourhood pairs
    (from the layout) whose PATCHES both exist; anchor = ``anchor`` or the
    deterministic :func:`select_central_cell`. Returns a :class:`CellFeatherResult`
    carrying the (N, C) additive offsets (anchor row 0) + diagnostics.

    Disconnected cells: ``on_disconnected="raise"`` (default, explicit) or
    ``"zero"`` (documented deterministic policy: unreachable Cells keep offset 0,
    recorded in ``disconnected``; no re-anchor, no bridge).
    """
    if on_disconnected not in ("raise", "zero"):
        raise ValueError(
            f"on_disconnected must be 'raise' or 'zero', got {on_disconnected!r}"
        )
    present = [cid for _r, _c, cid in
               ((row, col, cell_id(row, col)) for row in range(ny) for col in range(nx))
               if cid in patches]
    if not present:
        raise ValueError("no Cell patches provided")
    # Deterministic node order = layout row-major.
    order = present
    if anchor is None:
        anchor = select_central_cell(nx, ny)
    if anchor not in patches:
        # The central layout Cell is missing: fall back deterministically to the
        # first present Cell in row-major order (documented; never random).
        anchor = order[0]
    if channels is None:
        channels = _channels_of(patches[order[0]].science)

    # Adjacency restricted to present cells.
    present_set = set(order)
    edges: list[OffsetEdge] = []
    edges_built = 0
    for (u, v) in grid_adjacency(nx, ny):
        if u not in present_set or v not in present_set:
            continue
        edges_built += 1
        delta, n, weight = overlap_edge_observation(
            patches[u], patches[v], extra_mask=extra_mask, **obs_kw
        )
        if not np.all(np.isfinite(delta)):
            # A fully-invalid overlap yields NaN: skip the edge explicitly (it
            # carries no information) rather than injecting a fabricated value.
            continue
        edges.append(OffsetEdge(u, v, delta, weight, n_overlap=int(np.max(n))))

    sol = solve_additive_offsets(
        order, edges, anchor, channels,
        raise_on_disconnected=(on_disconnected == "raise"),
    )
    disconnected: tuple[str, ...] = ()
    if on_disconnected == "zero" and not _all_reachable(order, edges, anchor):
        disconnected = tuple(sorted(_unreachable(order, edges, anchor)))
    return CellFeatherResult(
        anchor=sol.anchor,
        offsets=sol.offsets,
        cell_ids=sol.cell_ids,
        channels=sol.channels,
        edges_used=sol.edges_used,
        edges_built=edges_built,
        weighted_residual_rms=sol.weighted_residual_rms,
        disconnected=disconnected,
        policy=on_disconnected,
    )


def _adjacency(order, edges):
    adj = {n: set() for n in order}
    for e in edges:
        adj[e.u].add(e.v)
        adj[e.v].add(e.u)
    return adj


def _reachable(order, edges, anchor):
    adj = _adjacency(order, edges)
    seen = {anchor}
    stack = [anchor]
    while stack:
        cur = stack.pop()
        for nb in sorted(adj[cur]):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen


def _all_reachable(order, edges, anchor):
    return len(_reachable(order, edges, anchor)) == len(set(order))


def _unreachable(order, edges, anchor):
    return set(order) - _reachable(order, edges, anchor)


# ---------------------------------------------------------------------------
# Offset application (ADDITIVE-ONLY)
# ---------------------------------------------------------------------------

def apply_additive_offsets(science, offsets, *, channel=None):
    """Return ``science + offsets`` (additive-only, per channel).

    ``offsets`` is a scalar or per-channel ``(C,)`` sequence. NaN (invalid) pixels
    are preserved (never turned into finite values). No gain is ever applied.
    """
    a = np.asarray(science, dtype=np.float64)
    off = np.asarray(offsets, dtype=np.float64).ravel()
    if a.ndim == 2:
        if off.size != 1:
            raise ValueError("offsets must be scalar for a mono (2-D) science plane")
        return a + off[0]
    if a.ndim != 3:
        raise ValueError(f"science must be 2-D or 3-D, got ndim={a.ndim}")
    if off.size not in (1, a.shape[2]):
        raise ValueError(
            f"offsets has {off.size} values, expected 1 or {a.shape[2]}"
        )
    if off.size == 1:
        return a + off[0]
    return a + off[None, None, :]


# ---------------------------------------------------------------------------
# Feather ramp + feathered assembly
# ---------------------------------------------------------------------------

def feather_weight_field(
    shape_hw: tuple[int, int],
    core_slice: tuple[int, int, int, int],
    halo_px: int = DEFAULT_HALO_PX,
    *,
    bounds: tuple[int, int, int, int] | None = None,
    canvas_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Deterministic partition-of-unity feather ramp over a Cell PATCH.

    The ramp is 1 across the core and decays LINEARLY to 0 at each patch border
    that is INTERIOR to the canvas (there a neighbouring Cell takes over); a patch
    border that COINCIDES with the canvas edge does NOT decay (no neighbour can
    take over, so the Cell must remain the sole owner there). Two adjacent Cells'
    ramps cross at their shared core boundary, so the normalized blend transitions
    smoothly across the ~2*halo overlap band.

    ``halo_px <= 0`` yields an all-ones field (no feather → the degenerate
    one-owner cut). ``bounds``/``canvas_hw`` are the global patch bounds and canvas
    size; when omitted, every border decays (used for direct ramp inspection).
    """
    hp, wp = int(shape_hw[0]), int(shape_hw[1])
    if halo_px <= 0:
        return np.ones((hp, wp), dtype=np.float64)
    xs = np.arange(wp, dtype=np.float64)
    ys = np.arange(hp, dtype=np.float64)

    decay_left = decay_top = decay_right = decay_bottom = True
    if bounds is not None and canvas_hw is not None:
        by0, by1, bx0, bx1 = bounds
        ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
        decay_left = bx0 > 0
        decay_top = by0 > 0
        decay_right = bx1 < cw
        decay_bottom = by1 < ch

    dxl = xs / float(halo_px) if decay_left else np.ones_like(xs)
    dxr = ((wp - 1) - xs) / float(halo_px) if decay_right else np.ones_like(xs)
    dx = np.clip(np.minimum(dxl, dxr), 0.0, 1.0)
    dyt = ys / float(halo_px) if decay_top else np.ones_like(ys)
    dyb = ((hp - 1) - ys) / float(halo_px) if decay_bottom else np.ones_like(ys)
    dy = np.clip(np.minimum(dyt, dyb), 0.0, 1.0)
    return dy[:, None] * dx[None, :]


def assemble_canvas_feathered(
    canvas: GlobalCanvas,
    nx: int,
    ny: int,
    patches: Mapping[str, CellPatch],
    offsets: Mapping[str, np.ndarray] | None = None,
    *,
    halo_px: int = DEFAULT_HALO_PX,
    extra_mask=None,
) -> "zmosaic.AssembledCanvas":
    """Feathered assembly: weighted blend (weights sum to 1) replacing the cut.

    Each Cell's PATCH science is blended into the canvas with weight
    ``ramp * valid * finite`` (per channel); the accumulator is normalized per
    pixel (``sum(w)``), so weights sum to 1 EXACTLY where any Cell covers and NO
    signal is attenuated in the core or the overlap band. Where no Cell is valid
    (or ``extra_mask`` excludes it) the output is NaN (explicit hole, never
    fabricated).

    ``offsets`` maps ``cell_id -> (C,)`` additive offsets (from the R34c solve);
    they are applied to the science BEFORE blending (additive-only). Cells absent
    from ``patches`` are recorded as incomplete (explicit hole).
    """
    layout = build_layout(canvas, nx, ny)
    h, w = int(canvas.height), int(canvas.width)
    c = 3
    num = np.zeros((h, w, c), dtype=np.float64)
    wsum = np.zeros((h, w, c), dtype=np.float64)
    stack_depth = np.zeros((h, w), dtype=np.int32)
    owner = np.zeros((h, w), dtype=np.int32)
    support_w1 = np.zeros((h, w), dtype=np.float64)
    support_w2 = np.zeros((h, w), dtype=np.float64)
    n_eff = np.zeros((h, w), dtype=np.float64)
    complete: list[str] = []
    incomplete: list[str] = []

    for row in range(ny):
        for col in range(nx):
            cid = cell_id(row, col)
            cp = patches.get(cid)
            if cp is None:
                incomplete.append(cid)
                continue
            sci = np.asarray(cp.science, dtype=np.float64)
            if sci.ndim == 2:
                sci = np.repeat(sci[:, :, None], c, axis=2)
            if sci.shape[2] != c:
                raise ValueError(
                    f"cell {cid}: expected {c} channels, got {sci.shape[2]}"
                )
            off = offsets.get(cid) if offsets is not None else None
            if off is not None:
                sci = apply_additive_offsets(sci, off)
            val = np.asarray(cp.valid_mask, dtype=bool)
            if val.ndim == 2:
                val = np.repeat(val[:, :, None], c, axis=2)
            finite = np.isfinite(sci)
            ramp = feather_weight_field(
                sci.shape[:2], cp.core_slice, halo_px,
                bounds=cp.bounds, canvas_hw=(h, w),
            )
            wgt = ramp[:, :, None] * (val & finite).astype(np.float64)
            if extra_mask is not None:
                y0, y1, x0, x1 = cp.bounds
                em = np.zeros((y1 - y0, x1 - x0), dtype=bool)
                gsy0, gey0 = max(0, y0), min(h, y1)
                gsx0, gex0 = max(0, x0), min(w, x1)
                em[gsy0 - y0:gey0 - y0, gsx0 - x0:gex0 - x0] = np.asarray(
                    extra_mask, dtype=bool
                )[gsy0:gey0, gsx0:gex0]
                wgt = wgt * em[:, :, None].astype(np.float64)

            y0, y1, x0, x1 = cp.bounds
            gsy0, gey0 = max(0, y0), min(h, y1)
            gsx0, gex0 = max(0, x0), min(w, x1)
            if gey0 <= gsy0 or gex0 <= gsx0:
                complete.append(cid)  # placed (entirely off-canvas defensive)
                continue
            ly0, ly1 = gsy0 - y0, gey0 - y0
            lx0, lx1 = gsx0 - x0, gex0 - x0
            sci_c = sci[ly0:ly1, lx0:lx1]
            wgt_c = wgt[ly0:ly1, lx0:lx1]

            num[gsy0:gey0, gsx0:gex0] += np.where(wgt_c > 0.0, sci_c * wgt_c, 0.0)
            wsum[gsy0:gey0, gsx0:gex0] += wgt_c
            cov = (wgt_c > 0.0).any(axis=-1)
            owner[gsy0:gey0, gsx0:gex0] += cov.astype(np.int32)
            if cp.surviving is not None:
                sv = np.asarray(cp.surviving, dtype=np.int32)[ly0:ly1, lx0:lx1]
                if sv.ndim == 2:
                    sv = np.repeat(sv[:, :, None], c, axis=2)
                sd = np.max(sv, axis=-1)
                tgt = stack_depth[gsy0:gey0, gsx0:gex0]
                np.maximum(tgt, np.where(cov, sd, 0), out=tgt)
            complete.append(cid)

    valid_out = wsum > 0.0
    science = np.full((h, w, c), np.nan, dtype=np.float64)
    np.divide(num, wsum, out=science, where=valid_out)
    science = science.astype(np.float32)

    # Coverage/diagnostics: a pixel is covered when any channel has weight.
    covered = valid_out.any(axis=-1)
    hole_mask = ~covered
    valid_fraction = np.mean(valid_out.astype(np.float32), axis=-1)
    # n_eff / support diagnostics: keep the (unblended) max over contributors is
    # NOT available here; report the coverage-derived n_eff proxy = sum of weights
    # per channel is already the estimator weight; use wsum for support_w1 and
    # owner count for n_eff placeholders that are explicitly documented.
    support_w1 = wsum[:, :, 0].copy()
    support_w2 = (wsum[:, :, 0] ** 2).copy()
    n_eff = owner.astype(np.float64)

    return zmosaic.AssembledCanvas(
        canvas=canvas,
        layout=layout,
        nx=nx,
        ny=ny,
        science=science,
        stack_depth=stack_depth,
        n_eff_support=n_eff,
        support_w1=support_w1,
        support_w2=support_w2,
        valid_fraction=valid_fraction,
        owner=owner,
        hole_mask=hole_mask,
        complete_cells=complete,
        incomplete_cells=incomplete,
    )
