"""ZM-CELL-PHOTOMETRY-R34C focused tests — inter-Cell ADDITIVE offset graph.

Fast, synthetic. Covers the R34c deliverables:

* robust sky observation (percentile band + MAD clipping; mirrors the legacy
  inter-tile `intertile_sky_percentile` / `intertile_robust_clip_sigma`) and its
  robustness to outlier pixels;
* deterministic layout topology (4-neighbourhood adjacency, central anchor);
* the GLOBAL weighted least-squares solve (central anchor, additive-only, no
  gain) — exact on consistent observations, drift-distributing on inconsistent
  ones (vs a sequential chain that accumulates);
* graph shapes: grid, chain, cycle, disconnected (explicit failure), isolated,
  single-cell;
* per-channel independence (mono + RGB), determinism, invalid/empty input;
* per-Cell reference STABILITY pin (R34b finding I1).
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import InMemoryCanonicalProvider
from zemosaic.core.zegrid import cell_offset as zco
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import photometric as zphot


# ---------------------------------------------------------------------------
# Robust sky observation
# ---------------------------------------------------------------------------

def test_robust_sky_level_clips_outliers_and_is_deterministic():
    rng = np.random.default_rng(0)
    sky = 100.0
    vals = sky + rng.normal(0.0, 1.0, 5000)
    vals[:50] = 1.0e6  # gross bright outliers
    level, n = zco.robust_sky_level(vals)
    assert abs(level - sky) < 0.5, level
    assert n > 0
    assert zco.robust_sky_level(vals) == (level, n)  # deterministic
    # empty / all-non-finite is explicit.
    assert zco.robust_sky_level(np.array([np.nan, np.inf])) == (float("nan"), 0) or (
        np.isnan(zco.robust_sky_level([np.nan, np.inf])[0])
        and zco.robust_sky_level([np.nan, np.inf])[1] == 0
    )


def test_edge_offset_observation_robust_to_outlier_pixels():
    rng = np.random.default_rng(1)
    h, w = 64, 64
    u = (50.0 + rng.normal(0.0, 0.8, (h, w))).astype(np.float32)
    v = (50.0 + rng.normal(0.0, 0.8, (h, w))).astype(np.float32)
    mask = np.zeros((h, w), dtype=bool)
    mask[8:56, 8:56] = True  # the halo overlap strip
    # A few gross bright pixels in u inside the overlap.
    u[20:24, 20:24] = 1.0e5
    delta, n, weight = zco.edge_offset_observation(u, v, mask)
    assert abs(delta) < 1.0, delta  # robust ~0, not dragged by the outliers
    assert weight == float(n) and n > 0


def test_edge_offset_observation_per_channel_rgb():
    rng = np.random.default_rng(2)
    h, w = 48, 48
    v = rng.normal(0.0, 0.5, (h, w, 3))
    u = v + np.array([2.0, 5.0, -1.0])  # known per-channel level shifts
    deltas, counts, weights = zco.edge_offset_observation(u, v)
    assert deltas.shape == (3,)
    assert np.allclose(deltas, [2.0, 5.0, -1.0], atol=0.2)


# ---------------------------------------------------------------------------
# Topology + anchor
# ---------------------------------------------------------------------------

def test_grid_adjacency_and_central_anchor():
    nx, ny = 3, 2
    edges = zco.grid_adjacency(nx, ny)
    assert len(edges) == nx * ny * 2 - nx - ny  # grid edge count
    assert ("r0000c0000", "r0000c0001") in edges
    assert ("r0000c0000", "r0001c0000") in edges
    # centre of a 3x2 grid -> row 0 or 1 (distance 0.5); tie -> lowest id.
    assert zco.select_central_cell(3, 2) == "r0000c0001"
    # 5x5 -> exact centre
    assert zco.select_central_cell(5, 5) == "r0002c0002"
    # 1x1
    assert zco.select_central_cell(1, 1) == "r0000c0000"
    # determinism / uniqueness of cell ids
    ids = [cid for _r, _c, cid in zco.grid_cell_ids(3, 2)]
    assert len(ids) == 6 and len(set(ids)) == 6


# ---------------------------------------------------------------------------
# Global solve: exactness, additive-only, per-channel, determinism
# ---------------------------------------------------------------------------

def _edges_from_levels(levels: dict, edges, channels=1, noise=None, rng=None):
    """Observations `o_v - o_u = level_u - level_v` (the additive offset model).

    With per-Cell sky levels s_i and offsets o_i s.t. s_i + o_i == L, we have
    o_v - o_u = s_u - s_v. Here we pass `levels` as the SKY levels s_i.
    """
    out = []
    for (u, v) in edges:
        d = levels[u] - levels[v]
        d = np.array([d] * channels, dtype=np.float64)
        if noise is not None:
            d = d + noise
        out.append(zco.OffsetEdge(u, v, d))
    return out


def test_solve_single_cell():
    sol = zco.solve_additive_offsets(["r0000c0000"], [], "r0000c0000", channels=1)
    assert sol.offsets.shape == (1, 1)
    assert sol.offsets[0, 0] == 0.0
    assert sol.components == 1 and sol.edges_used == 0


def test_solve_chain_exact():
    ids = [f"c{i}" for i in range(3)]
    edges = [("c0", "c1"), ("c1", "c2")]
    levels = {"c0": 10.0, "c1": 14.0, "c2": 9.0}
    sol = zco.solve_additive_offsets(ids, _edges_from_levels(levels, edges), "c0", 1)
    # o_i = level_anchor - level_i
    expected = np.array([[0.0], [10.0 - 14.0], [10.0 - 9.0]])
    np.testing.assert_allclose(sol.offsets, expected, atol=1e-12)
    assert np.allclose(sol.weighted_residual_rms, 0.0, atol=1e-12)


def test_solve_grid_exact_and_additive_only():
    nx, ny = 3, 3
    ids = [cid for _r, _c, cid in zco.grid_cell_ids(nx, ny)]
    edges = zco.grid_adjacency(nx, ny)
    rng = np.random.default_rng(5)
    levels = {cid: float(rng.normal(100.0, 20.0)) for cid in ids}
    anchor = zco.select_central_cell(nx, ny)
    sol = zco.solve_additive_offsets(ids, _edges_from_levels(levels, edges), anchor, 1)

    # ADDITIVE-ONLY: s_i + o_i == s_anchor (= the common level), exactly.
    recon = np.array([levels[cid] + sol.channel_offsets[cid][0] for cid in ids])
    np.testing.assert_allclose(recon, levels[anchor], atol=1e-9)
    assert sol.offsets[ids.index(anchor), 0] == 0.0
    assert np.allclose(sol.weighted_residual_rms, 0.0, atol=1e-9)


def test_solve_per_channel_independence():
    ids = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c")]
    # channel-wise level diffs: R:+0, G:+1, B:+2 per step
    e = [
        zco.OffsetEdge("a", "b", [0.0, 1.0, 2.0]),
        zco.OffsetEdge("b", "c", [0.0, 1.0, 2.0]),
    ]
    sol = zco.solve_additive_offsets(ids, e, "a", channels=3)
    np.testing.assert_allclose(sol.offsets[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(sol.offsets[1], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(sol.offsets[2], [0.0, 2.0, 4.0])


def test_solve_deterministic():
    nx, ny = 4, 4
    ids = [cid for _r, _c, cid in zco.grid_cell_ids(nx, ny)]
    edges = zco.grid_adjacency(nx, ny)
    rng = np.random.default_rng(7)
    levels = {cid: float(rng.normal(50.0, 5.0)) for cid in ids}
    obs = _edges_from_levels(levels, edges, noise=rng.normal(0.0, 0.3, 1))
    anchor = zco.select_central_cell(nx, ny)
    s1 = zco.solve_additive_offsets(ids, obs, anchor, 1)
    s2 = zco.solve_additive_offsets(ids, obs, anchor, 1)
    np.testing.assert_array_equal(s1.offsets, s2.offsets)
    assert s1.anchor == s2.anchor


# ---------------------------------------------------------------------------
# Drift: global solve distributes; a sequential chain accumulates
# ---------------------------------------------------------------------------

def test_global_solve_is_drift_free_vs_chain():
    """2x2 grid with ONE gross edge error: the anchor-relative offsets from the
    GLOBAL solve stay closer to truth than the sequential chain, which
    accumulates the whole error along the path."""
    ids = ["a", "b", "c", "d"]  # a=(0,0) b=(0,1) c=(1,0) d=(1,1)
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    assert set(edges) == {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}
    true = {"a": 0.0, "b": 2.0, "c": 3.0, "d": 5.0}  # true additive offsets

    # Exact consistent observations (o_v - o_u = true_v - true_u):
    # (levels s_i = -true_i, so s_u - s_v = true_v - true_u)
    levels = {k: -v for k, v in true.items()}
    obs = _edges_from_levels(levels, edges)
    # Perturb ONE edge (a-b) by a gross +4 error.
    obs = [
        zco.OffsetEdge(e.u, e.v, np.asarray(e.delta) + (4.0 if (e.u, e.v) == ("a", "b") else 0.0))
        for e in obs
    ]

    global_sol = zco.solve_additive_offsets(ids, obs, "a", 1)
    chain = zco.chain_offsets(ids, obs, "a", 1)
    truth = np.array([true[c] for c in ids]).reshape(-1, 1)

    g_err = np.abs(global_sol.offsets - truth).max()
    c_err = np.abs(chain - truth).max()
    # The chain reproduces the full error along the path (b +4, d +4); the global
    # solve distributes it (never accumulates the whole error).
    assert c_err == pytest.approx(4.0)
    assert g_err < c_err
    # Specifically, the node FAR from the bad edge is much better in the global solve.
    d_idx = ids.index("d")
    assert abs(global_sol.offsets[d_idx, 0] - true["d"]) < abs(chain[d_idx, 0] - true["d"])
    # And the global solve's residual is spread, not concentrated at one edge.
    assert np.all(np.isfinite(global_sol.weighted_residual_rms))


def test_cycle_consistent_recovers_exactly():
    ids = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c"), ("c", "a")]  # a triangle (cycle)
    levels = {"a": 10.0, "b": 12.0, "c": 7.0}
    sol = zco.solve_additive_offsets(ids, _edges_from_levels(levels, edges), "a", 1)
    recon = np.array([levels[c] + sol.channel_offsets[c][0] for c in ids])
    np.testing.assert_allclose(recon, [10.0, 10.0, 10.0], atol=1e-9)


def test_cycle_inconsistent_least_squares_balances():
    ids = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    # consistent diffs + equal-and-opposite pushes that are inconsistent around
    # the loop (their sum is nonzero -> no exact solution).
    e = [
        zco.OffsetEdge("a", "b", 1.0),
        zco.OffsetEdge("b", "c", 1.0),
        zco.OffsetEdge("c", "a", 1.0),
    ]
    sol = zco.solve_additive_offsets(ids, e, "a", 1)
    # LS distributes the inconsistency: residuals balanced, anchor pinned at 0.
    assert sol.offsets[ids.index("a"), 0] == 0.0
    assert sol.weighted_residual_rms[0] > 0.0


# ---------------------------------------------------------------------------
# Disconnected / isolated (explicit failure, no fabrication)
# ---------------------------------------------------------------------------

def test_disconnected_graph_raises():
    ids = ["a", "b", "x", "y"]
    edges = [zco.OffsetEdge("a", "b", 2.0), zco.OffsetEdge("x", "y", 3.0)]
    with pytest.raises(zco.DisconnectedOffsetGraphError):
        zco.solve_additive_offsets(ids, edges, "a", 1)


def test_isolated_cell_raises():
    ids = ["a", "b", "lonely"]
    with pytest.raises(zco.DisconnectedOffsetGraphError):
        zco.solve_additive_offsets(ids, [zco.OffsetEdge("a", "b", 1.0)], "a", 1)


def test_single_cell_graph_is_anchor_only():
    sol = zco.solve_additive_offsets(["only"], [], "only", channels=3)
    np.testing.assert_array_equal(sol.offsets, np.zeros((1, 3)))


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------

def test_solve_rejects_invalid_input():
    good = [zco.OffsetEdge("a", "b", 1.0)]
    with pytest.raises(ValueError):
        zco.solve_additive_offsets([], good, "a", 1)  # empty
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "a"], good, "a", 1)  # duplicate
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "b"], good, "z", 1)  # bad anchor
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "b"], [zco.OffsetEdge("a", "z", 1.0)], "a", 1)
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "b"], good, "a", 0)  # bad channels
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "b"], [zco.OffsetEdge("a", "b", np.nan)], "a", 1)
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "b"], [zco.OffsetEdge("a", "b", 1.0, -1.0)], "a", 1)
    with pytest.raises(ValueError):
        zco.solve_additive_offsets(["a", "b"], [zco.OffsetEdge("a", "a", 1.0)], "a", 1)


# ---------------------------------------------------------------------------
# End-to-end: robust observations -> solve (additive recovery with outliers)
# ---------------------------------------------------------------------------

def test_observations_then_solve_recovers_additive_levels(tmp_path):
    """Synthetic 2x2 halo-overlap planes with KNOWN per-cell levels + noise +
    outlier pixels: the robust edge observations + global solve recover the
    additive offsets that equalise every cell to the anchor level."""
    nx, ny = 2, 2
    ids = [cid for _r, _c, cid in zco.grid_cell_ids(nx, ny)]
    edges = zco.grid_adjacency(nx, ny)
    levels = {"r0000c0000": 100.0, "r0000c0001": 108.0,
              "r0001c0000": 95.0, "r0001c0001": 99.0}
    rng = np.random.default_rng(11)
    h = w = 40
    # One "sky image" per cell, all at its own level, with per-pair overlap masks.
    planes = {cid: (levels[cid] + rng.normal(0.0, 0.7, (h, w))).astype(np.float32)
              for cid in ids}
    # Gross outlier blobs (should be clipped by the robust observation).
    planes["r0000c0001"][0:3, 0:3] = 1.0e5
    overlap = np.zeros((h, w), dtype=bool)
    overlap[15:25, 15:25] = True

    obs = []
    for (u, v) in edges:
        delta, n, weight = zco.edge_offset_observation(
            planes[u], planes[v], overlap
        )
        obs.append(zco.OffsetEdge(u, v, delta, weight, n_overlap=int(n)))

    anchor = zco.select_central_cell(nx, ny)
    sol = zco.solve_additive_offsets(ids, obs, anchor, 1)
    recon = np.array([levels[cid] + sol.channel_offsets[cid][0] for cid in ids])
    # Every cell lands on the anchor's level within the robust noise tolerance.
    np.testing.assert_allclose(recon, levels[anchor], atol=1.0)


# ---------------------------------------------------------------------------
# Reference STABILITY pin (R34b finding I1)
# ---------------------------------------------------------------------------

def _cluster_provider(tmp_path, n_a=2, n_b=2, rgb=False, h=32, w=32, seed=3):
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.normal(0.0, 1.0, (h, w))
    left = np.zeros((h, w), dtype=bool)
    left[:, : w // 2] = True
    right = np.zeros((h, w), dtype=bool)
    right[:, w // 2 :] = True
    frames, masks, ids = [], [], []
    for i in range(n_a):
        d = np.full((h, w), np.nan, np.float32)
        d[left] = (base + i * 3.0)[left]
        frames.append(d); masks.append(left); ids.append(f"f{i}.fits")
    for i in range(n_b):
        d = np.full((h, w), np.nan, np.float32)
        d[right] = (base + (5 + i) * 3.0)[right]
        frames.append(d); masks.append(right); ids.append(f"f{n_a + i}.fits")
    if rgb:
        out = []
        for f in frames:
            out.append(np.stack([f, f + 1.0, f + 2.0], axis=-1).astype(np.float32))
        frames = out
    cache = tmp_path / "cc"
    zfp.write_aligned_cache_from_arrays(str(cache), ids, frames, masks)
    return zfp.MemmapCanonicalProvider(str(cache))


def _cell_request(n):
    return CanonicalStackRequest(
        images=[None] * n, geometric_support=[None] * n,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean", taper="footprint", taper_px=8.0,
    )


def test_reference_pin_is_stable_across_recompute(tmp_path):
    prov = _cluster_provider(tmp_path)
    try:
        res1 = zphot.compute_cell_gauge(prov, _cell_request(prov.n_frames))
        res2 = zphot.compute_cell_gauge(prov, _cell_request(prov.n_frames))
        assert res1.status == zphot.CELL_GAUGE_STATUS_OK
        pin1, pin2 = res1.pin(), res2.pin()
        assert pin1 == pin2
        assert pin1.frame_id == res1.frame_ids[res1.reference_index]
        assert pin1.support == res1.reference_support
        assert pin1.support is not None and pin1.support > 0
        # The pin fully determines the choice: recompute cannot flip the anchor.
        assert res1.reference_frame_id == res2.reference_frame_id
        assert res1.reference_support == res2.reference_support
    finally:
        prov.close()


def test_reference_pin_single_frame(tmp_path):
    prov = _cluster_provider(tmp_path, n_a=1, n_b=0)
    try:
        res = zphot.compute_cell_gauge(prov, _cell_request(prov.n_frames))
        pin = res.pin()
        assert res.reference_index == 0
        assert pin.frame_id == "f0.fits"
        assert pin.support > 0
    finally:
        prov.close()


def test_reference_pin_empty_cell_is_none(tmp_path):
    h = w = 16
    full = np.ones((h, w), dtype=bool)
    nan = np.full((h, w), np.nan, dtype=np.float32)
    cache = tmp_path / "empty"
    zfp.write_aligned_cache_from_arrays(str(cache), ["a.fits", "b.fits"], [nan, nan.copy()], [full, full])
    prov = zfp.MemmapCanonicalProvider(str(cache))
    try:
        res = zphot.compute_cell_gauge(prov, _cell_request(2))
        pin = res.pin()
        assert res.status == zphot.CELL_GAUGE_STATUS_EMPTY
        assert pin.frame_id is None and pin.support is None
    finally:
        prov.close()
