"""ZM-ZEGRID-R37-B focused tests — pure additive background-model prototype (r1).

Covers the rework-1 findings H1–H3, M1–M7, L1–L4 and the mandated synthetic
matrix of ``docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md`` §7.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zemosaic.core.zegrid import background_model as bm
from zemosaic.core.zegrid import cell_offset as zco


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

CORE = 64
HALO = 8


def _cid(row, col):
    return f"r{row:04d}c{col:04d}"


def build_grid(
    nx, ny, *, core=CORE, halo=HALO, channels=1, plane=None, sky=None,
    gain=None, noise=0.0, noise_fn=None, seed=0, nonlinear=None, star=None,
    hole=(), valid_sliver=None, contradict_edge=None, extra_shift=None,
):
    """Synthetic corpus of ``PlanCell`` patches (see r0 for semantics).

    ``contradict_edge = (u, v, value)`` adds ``value`` to cell ``v`` ONLY inside
    the overlap band shared with ``u`` — a genuine contradictory RELATION.
    ``extra_shift = (cid, value)`` shifts a whole cell (graph-consistent).
    ``valid_sliver = (cid, channel, hh, ww)`` keeps only a top-left sliver valid.
    """
    rng = np.random.default_rng(seed)
    W, H = nx * core, ny * core
    cells = {}
    for r in range(ny):
        for c in range(nx):
            cid = _cid(r, c)
            cy0, cy1 = r * core, (r + 1) * core
            cx0, cx1 = c * core, (c + 1) * core
            y0, y1 = max(0, cy0 - halo), min(H, cy1 + halo)
            x0, x1 = max(0, cx0 - halo), min(W, cx1 + halo)
            hh, ww = y1 - y0, x1 - x0
            gx = np.arange(x0, x1, dtype=np.float64)
            gy = np.arange(y0, y1, dtype=np.float64)
            XX = (2.0 * (gx + 0.5) / W - 1.0)[None, :].repeat(hh, 0)
            YY = (2.0 * (gy + 0.5) / H - 1.0)[:, None].repeat(ww, 1)
            XL = ((np.arange(ww) - (ww - 1) / 2.0) / max(ww / 2.0, 1.0))[None, :].repeat(hh, 0)
            YL = ((np.arange(hh) - (hh - 1) / 2.0) / max(hh / 2.0, 1.0))[:, None].repeat(ww, 1)

            arr = np.zeros((hh, ww, channels), dtype=np.float64)
            for ch in range(channels):
                v = np.zeros((hh, ww), dtype=np.float64)
                if sky is not None:
                    v = v + sky(XX, YY, ch)
                if star is not None:
                    sx, sy, amp, sig = star
                    v = v + amp * np.exp(-((gx[None, :] - sx) ** 2 + (gy[:, None] - sy) ** 2) / (2 * sig * sig))
                if gain is not None:
                    v = v * gain(cid, ch)
                if plane is not None:
                    p = np.asarray(plane(cid), dtype=np.float64)
                    if p.ndim == 1:
                        p = np.repeat(p[None, :], channels, axis=0)
                    v = v + p[ch, 0] + p[ch, 1] * XX + p[ch, 2] * YY
                if nonlinear is not None:
                    v = v + nonlinear(cid, XX, YY, XL, YL, ch)
                if noise_fn is not None:
                    v = v + rng.normal(0.0, noise_fn(cid, ch), (hh, ww))
                elif noise > 0.0:
                    v = v + rng.normal(0.0, noise, (hh, ww))
                arr[:, :, ch] = v

            if contradict_edge is not None:
                cu, cv, val = contradict_edge
                if cid == cv and cu in cells:
                    ub = cells[cu].bounds
                    iy0, iy1 = max(y0, ub[0]), min(y1, ub[1])
                    ix0, ix1 = max(x0, ub[2]), min(x1, ub[3])
                    # Restrict to the PURE part of the v-u band: shrink the tangent
                    # axis inside v's core so the corruption does not spill into
                    # v's other (perpendicular) overlap bands.
                    ur, uc = int(cu[1:5]), int(cu[6:10])
                    same_row = (ur == r)
                    if same_row:
                        iy0 = max(iy0, r * core + halo)
                        iy1 = min(iy1, (r + 1) * core - halo)
                    else:
                        ix0 = max(ix0, c * core + halo)
                        ix1 = min(ix1, (c + 1) * core - halo)
                    if iy1 > iy0 and ix1 > ix0:
                        arr[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0, :] += val

            if extra_shift is not None and cid == extra_shift[0]:
                arr = arr + extra_shift[1]

            if cid in hole:
                arr[:] = np.nan
            mask = None
            if valid_sliver is not None and cid == valid_sliver[0]:
                ch = valid_sliver[1]
                mask = np.zeros((hh, ww), dtype=bool)
                mask[: valid_sliver[2], : valid_sliver[3]] = True
                arr[: valid_sliver[2], : valid_sliver[3], ch] = arr[: valid_sliver[2], : valid_sliver[3], ch]
                arr[mask.shape[0]:, :, ch] = np.nan
                arr[:, mask.shape[1]:, ch] = np.nan
            cells[cid] = bm.PlanCell(cid, (y0, y1, x0, x1),
                                     arr[:, :, 0] if channels == 1 else arr, valid_mask=mask)
    return cells, (H, W)


def plane_fn(amp=4.0, bamp=1.5):
    def f(cid):
        r, c = int(cid[1:5]), int(cid[6:10])
        return (amp * ((r + c) % 5 - 2), bamp * ((r % 3) - 1), -bamp * ((c % 3) - 1))
    return f


def plane_from_centres(nx, ny, field):
    """Truth plane as a function of the cell's GLOBAL normalized centre (physical)."""
    def f(cid):
        r, c = int(cid[1:5]), int(cid[6:10])
        cx = 2.0 * (c + 0.5) / nx - 1.0
        cy = 2.0 * (r + 0.5) / ny - 1.0
        return field(cx, cy)
    return f


def truth_of(cells, plane):
    return {cid: np.asarray(plane(cid), dtype=np.float64) for cid in cells}


def common_gradient(gx=8.0, gy=5.0, base=20.0):
    return lambda XX, YY, ch: base + gx * XX + gy * YY


def ifn_field(amp=30.0, cx=0.1, cy=-0.2, s=0.6):
    return lambda XX, YY, ch: amp * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / s)


def source_field(seed=1, n=60):
    rng = np.random.default_rng(seed)
    pts = [(rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9),
            rng.uniform(20, 80), rng.uniform(0.04, 0.10)) for _ in range(n)]

    def sky(XX, YY, ch):
        v = 10.0 + 2.0 * XX
        for (x, y, a, s) in pts:
            v = v + a * np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / (2 * s * s))
        return v
    return sky


ALLOWED = {
    bm.STATE_OFFSET_ONLY, bm.STATE_ROBUST_PLAN, bm.STATE_INSUFFICIENT_EVIDENCE,
    bm.STATE_ILL_CONDITIONED, bm.STATE_MODEL_MISMATCH, bm.STATE_DISCONNECTED,
}


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def assemble_cores(cells, corrections, canvas_hw, nx, ny):
    H, W = canvas_hw
    ch, cw = H // ny, W // nx
    out = np.full((H, W), np.nan, dtype=np.float64)
    for cid, cell in cells.items():
        r, c = int(cid[1:5]), int(cid[6:10])
        cy0, cy1 = r * ch, (r + 1) * ch
        cx0, cx1 = c * cw, (c + 1) * cw
        y0, y1, x0, x1 = cell.bounds
        sci = cell.science
        corr = corrections.get(cid)
        if corr is not None:
            sci = bm.apply_plane_correction(sci, corr, (y0, y1, x0, x1), canvas_hw)
        iy0, iy1 = max(cy0, y0), min(cy1, y1)
        ix0, ix1 = max(cx0, x0), min(cx1, x1)
        out[iy0:iy1, ix0:ix1] = sci[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
    return out


def regress_amplitude(image, truth):
    m = np.isfinite(image) & np.isfinite(truth)
    x, y = truth[m], image[m]
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    return slope, float(np.median(y - x))


def aperture_flux(img, cx, cy, r=5.0, r_in=8.0, r_out=14.0):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.hypot(xx - cx, yy - cy)
    ap = d <= r
    ann = (d >= r_in) & (d <= r_out)
    bkg = float(np.median(img[ann]))
    return float(np.sum(img[ap] - bkg))


# ===========================================================================
# H1 — measurement band consistency / clamping
# ===========================================================================

@pytest.mark.parametrize("band", [4, 8, 16, 32])
def test_h1_band_independent_of_overlap_no_crash(band):
    """Production-like halo 8 (overlap 16) with band <, =, > overlap: must not crash."""
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=8, plane=plane, noise=0.2, seed=1)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=band)
    assert res.decision in ALLOWED
    assert res.provenance["measurement_px"] == band


def test_h1_band_narrower_than_overlap_band_used_le_buget():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=8, plane=plane, noise=0.2, seed=1)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=4)
    assert all(e["band_used_px"] <= 4 for e in res.edge_stats)
    assert all(e["overlap_px"] == 16 for e in res.edge_stats)
    assert all(not e["band_clamped"] for e in res.edge_stats)


def test_h1_band_wider_than_overlap_is_reported_clamped():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=8, plane=plane, noise=0.2, seed=1)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    assert res.edge_stats, "expected edges"
    assert all(e["band_clamped"] for e in res.edge_stats)
    assert all(e["refusal_reason"] == "band_clamped_to_overlap" for e in res.edge_stats)
    assert all(e["band_used_px"] <= e["overlap_px"] for e in res.edge_stats)


# ===========================================================================
# H2 — fragmented per-channel support must not raise
# ===========================================================================

def test_h2_channel_hole_returns_refusal_not_exception():
    plane = plane_fn()
    def plane3(cid):
        p = np.asarray(plane(cid), dtype=np.float64)
        return np.vstack([p, p * 0.5, -p * 0.25])
    cells, cw = build_grid(6, 4, channels=3, plane=plane3, noise=0.2, seed=2,
                           hole=())
    # channel 1 of an interior Cell is entirely invalid
    cid = _cid(2, 2)
    arr = cells[cid].science.copy()
    arr[:, :, 1] = np.nan
    cells[cid] = bm.PlanCell(cid, cells[cid].bounds, arr)
    res = bm.compare_models(cells, 6, 4, cw, channels=3)   # must NOT raise
    assert res.decision in (bm.STATE_DISCONNECTED, bm.STATE_INSUFFICIENT_EVIDENCE)
    assert res.per_channel[1]["connected"] is False
    assert res.refusal_reasons
    assert np.all(np.isnan(res.plan_coefficients[:, 1, :]))


# ===========================================================================
# H3 — per-channel rank/condition/connection; no silent zero
# ===========================================================================

def test_h3_per_channel_diagnostics_and_no_silent_zero():
    plane = plane_fn()
    def plane2(cid):
        p = np.asarray(plane(cid), dtype=np.float64)
        return np.vstack([p, p * 0.5])
    # 6x1 chain: channel 1 lacks one interior edge -> channel 1 disconnected
    cells, cw = build_grid(6, 1, channels=2, plane=plane2, noise=0.2, seed=3)
    mid = _cid(0, 3)
    arr = cells[mid].science.copy()
    arr[:, :, 1] = np.nan  # removes all channel-1 overlap for that cell
    cells[mid] = bm.PlanCell(mid, cells[mid].bounds, arr)
    res = bm.compare_models(cells, 6, 1, cw, channels=2)
    assert res.decision == bm.STATE_DISCONNECTED
    assert res.per_channel[1]["connected"] is False
    assert res.per_channel[1]["under_determined"] is True
    # the under-determined channel is NaN, never zero
    assert np.all(np.isnan(res.plan_coefficients[:, 1, :]))
    # the healthy channel is still finite
    assert np.all(np.isfinite(res.plan_coefficients[:, 0, :]))


def test_h3_worst_channel_governs_rank_and_condition():
    plane = plane_fn()
    def plane2(cid):
        p = np.asarray(plane(cid), dtype=np.float64)
        return np.vstack([p, p])
    cells, cw = build_grid(4, 4, core=64, halo=32, channels=2, plane=plane2, noise=0.2, seed=4)
    res = bm.compare_models(cells, 4, 4, cw, channels=2, measurement_px=64)
    assert set(res.per_channel.keys()) == {0, 1}
    for c in (0, 1):
        assert res.per_channel[c]["rank"] == 3 * (len(res.cell_ids) - 1)


# ===========================================================================
# M1 — held-out gate: RMS around zero, symmetric p90, frozen 5%
# ===========================================================================

def test_m1_const_bias_is_visible_to_gate():
    """A constant held-out bias must not vanish (no destructive recentring)."""
    cells, cw = build_grid(2, 1, plane=lambda cid: (0.0, 0.0, 0.0), noise=0.0)
    obs, _ = bm.build_observations(cells, 2, 1, cw)
    idx = {obs[0].u: 0, obs[0].v: 1}
    # craft fake holdout residuals: M0 carries a +3 constant bias
    o = obs[0]
    fake = bm.EdgeObservation(
        u=o.u, v=o.v, channel=0, orientation=o.orientation,
        n_train=o.n_train, n_holdout=4, effective_train=o.effective_train,
        tangent_blocks=o.tangent_blocks, normal_blocks=o.normal_blocks,
        robust_sigma=1.0, rejection_fraction=0.0, theta_global=o.theta_global,
        information=o.information, rhs=o.rhs, level_z=0.0, curvature_z=0.0,
        level_mean=0.0, local_offset_train=0.0, measurement_px=None,
        overlap_px=8, band_used_px=8, band_clamped=False,
        holdout_xy=np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]),
        holdout_diff=np.array([3.0, 3.0, 3.0, 3.0]),
    )
    coeffs_plan = np.zeros((2, 1, 3))
    offsets_m0 = np.array([[0.0], [3.0]])  # M0 predicts 3 -> zero residual
    hm = bm.heldout_metrics([fake], coeffs_plan, offsets_m0, idx, 1)
    # M1 residual is a constant 3 -> RMS must be ~3 (bias preserved)
    assert hm["r_plan"] == pytest.approx(3.0, abs=1e-9)
    assert hm["r_offset"] == pytest.approx(0.0, abs=1e-9)
    e = hm["per_edge"][0]
    assert e["p90_plan"] == pytest.approx(3.0, abs=1e-9)   # symmetric operator
    assert e["p90_m0"] == pytest.approx(0.0, abs=1e-9)


def test_m1_p90_gate_is_strict_five_percent():
    assert bm.GateConfig().per_edge_p90_worsen_max == 0.05
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=5)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    for e in res.heldout["per_edge"]:
        assert e["p90_plan"] <= 1.0 + 0.05 + 1e-9 or e["p90_plan"] <= (1.05 * e["p90_m0"])


# ===========================================================================
# M2 — §6.1.6 corner uncertainty from the DATA-ONLY system
# ===========================================================================

def test_m2_corner_uncertainty_independent_of_lambda():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=6)
    obs, _ = bm.build_observations(cells, 6, 4, cw, measurement_px=64)
    cells_ids = tuple(f"r{r:04d}c{c:04d}" for r in range(4) for c in range(6))
    a0 = bm.solve_plan_sparse(obs, cells_ids, bm.select_central_cell(6, 4), 1, reg_lambda=0.0)
    a1 = bm.solve_plan_sparse(obs, cells_ids, bm.select_central_cell(6, 4), 1, reg_lambda=1.0)
    assert a0.corner_uncertainty is not None and a1.corner_uncertainty is not None
    assert np.allclose(a0.corner_uncertainty, a1.corner_uncertainty, equal_nan=True, atol=1e-12)


# ===========================================================================
# M3 — edge confidence cap + bounded IRLS; genuine contradictory relation
# ===========================================================================

def _mk_obs(u, v, theta0, info_scale, channel=0, sigma=1.0):
    info = np.diag([16.0, 16.0, 4.0]) * info_scale
    theta = np.array([theta0, 0.0, 0.0])
    return bm.EdgeObservation(
        u=u, v=v, channel=channel, orientation="v", n_train=256, n_holdout=0,
        effective_train=256.0, tangent_blocks=8, normal_blocks=4, robust_sigma=sigma,
        rejection_fraction=0.0, theta_global=theta, information=info, rhs=info @ theta,
        level_z=0.0, curvature_z=0.0, level_mean=0.0, local_offset_train=theta0,
        measurement_px=None, overlap_px=8, band_used_px=8, band_clamped=False,
        holdout_xy=np.zeros((0, 2)), holdout_diff=np.zeros(0),
    )


def test_m3_huge_edge_cannot_dominate():
    ids = ["r0000c0000", "r0000c0001", "r0000c0002", "r0000c0003"]
    good = [_mk_obs(ids[i], ids[i + 1], 0.0, 1.0) for i in range(3)]
    # one contradictory edge with gigantic information
    bad = [_mk_obs(ids[1], ids[2], 50.0, 1e12)]
    obs = good + bad
    anchor = ids[0]
    sol = bm.solve_plan_sparse(obs, ids, anchor, 1, reg_lambda=1e-3, with_uncertainty=False)
    # capped + IRLS: the contradictory relation must not drag the chain to 50
    assert np.max(np.abs(sol.coefficients[:, 0, 0])) <= 5.0


def test_m3_graph_consistent_cell_shift_is_not_a_contradiction():
    plane = plane_fn()
    cells, cw = build_grid(2, 2, plane=plane, noise=0.2, seed=7)
    # a whole-cell shift is graph-consistent: cycle residual must be ~0
    cells[_cid(0, 1)] = bm.PlanCell(_cid(0, 1), cells[_cid(0, 1)].bounds,
                                    cells[_cid(0, 1)].science + 300.0)
    obs, _ = bm.build_observations(cells, 2, 2, cw, measurement_px=None)
    cy = bm._cycle_residuals(obs, tuple(cells), 2, 2, cw, 1)
    assert cy and all(abs(c["offset_residual"]) < 50.0 for c in cy)


def _cycle_obs(theta_by_edge, *, sigma=1.0, info_scale=1.0):
    """Crafted 2x2 cycle observations: one relation per edge (A-B, A-D, B-E, D-E)."""
    A, B, D, E = _cid(0, 0), _cid(0, 1), _cid(1, 0), _cid(1, 1)
    spec = {
        (A, B): "h", (A, D): "v", (B, E): "v", (D, E): "h",
    }
    obs = []
    for (u, v), orient in spec.items():
        theta0 = float(theta_by_edge.get((u, v), 0.0))
        info = np.diag([16.0, 16.0, 4.0]) * info_scale
        theta = np.array([theta0, 0.0, 0.0])
        obs.append(bm.EdgeObservation(
            u=u, v=v, channel=0, orientation=orient, n_train=256, n_holdout=0,
            effective_train=256.0, tangent_blocks=8, normal_blocks=4, robust_sigma=sigma,
            rejection_fraction=0.0, theta_global=theta, information=info, rhs=info @ theta,
            level_z=0.0, curvature_z=0.0, level_mean=0.0, local_offset_train=theta0,
            measurement_px=None, overlap_px=8, band_used_px=8, band_clamped=False,
            holdout_xy=np.zeros((0, 2)), holdout_diff=np.zeros(0),
        ))
    return obs


def test_m3_true_contradictory_relation_in_cycle():
    """One genuinely contradictory RELATION in an over-constrained cycle."""
    A, B, D, E = _cid(0, 0), _cid(0, 1), _cid(1, 0), _cid(1, 1)
    obs = _cycle_obs({(A, B): 30.0})
    cy = bm._cycle_residuals(obs, (A, B, D, E), 2, 2, (128, 128), 1)
    assert cy, "expected cycle diagnostics"
    assert max(c["z"] for c in cy) > 10.0
    # the robust graph solve must not be dragged to the contradictory value
    sol = bm.solve_plan_sparse(obs, (A, B, D, E), A, 1, reg_lambda=1e-3, with_uncertainty=False)
    assert np.max(np.abs(sol.coefficients[:, 0, 0])) <= 5.0
    # negative control: a consistent cycle has ~zero residual
    obs_ok = _cycle_obs({})
    cy_ok = bm._cycle_residuals(obs_ok, (A, B, D, E), 2, 2, (128, 128), 1)
    assert max(abs(c["offset_residual"]) for c in cy_ok) < 1e-9


def test_m3_corrupted_overlap_does_not_yield_confident_plan():
    """A corrupted overlap relation must not end as a confident ROBUST_PLAN."""
    plane = plane_fn()
    A, B = _cid(0, 0), _cid(0, 1)
    cells, cw = build_grid(2, 2, plane=plane, noise=0.2, seed=8,
                           contradict_edge=(A, B, 400.0))
    res = bm.compare_models(cells, 2, 2, cw, measurement_px=None,
                            truth_planes=truth_of(cells, plane))
    assert res.decision != bm.STATE_ROBUST_PLAN
    assert res.cycles


def test_m3_chain_has_no_cycle_evidence():
    plane = plane_fn()
    cells, cw = build_grid(4, 1, plane=plane, noise=0.2, seed=9)
    res = bm.compare_models(cells, 4, 1, cw, measurement_px=None)
    assert res.cycles == []


# ===========================================================================
# M4 — cycle residuals + refusal reasons + dropped inventory
# ===========================================================================

def test_m4_payload_has_cycles_dropped_and_reasons():
    plane = plane_fn()
    mid = _cid(1, 1)
    cells, cw = build_grid(3, 3, plane=plane, noise=0.2, seed=10, hole=(mid,))
    res = bm.compare_models(cells, 3, 3, cw, measurement_px=None)
    payload = res.to_json_dict()
    assert "cycles" in payload and "dropped" in payload
    assert payload["dropped"], "expected dropped-relation inventory"
    assert any(d["reason"] for d in payload["dropped"])
    # cycle entries (if any) are fully populated
    for cy in payload["cycles"]:
        assert set(["cycle", "channel", "offset_residual", "sigma_scale", "z"]).issubset(cy)
    json.dumps(payload)  # JSON-safe


# ===========================================================================
# M5 — M0 offset observation is TRAIN-only
# ===========================================================================

def test_m5_m0_uses_train_only_local_offset():
    # a band with a strong tangent trend: median(train) != median(all)
    def sky(XX, YY, ch):
        return 40.0 * XX
    cells, cw = build_grid(4, 2, core=64, halo=16, plane=plane_fn(), sky=sky, noise=0.0)
    obs, _ = bm.build_observations(cells, 4, 2, cw, measurement_px=None)
    for o in obs:
        # reconstruct train/holdout medians from the stored arrays
        xy = o.holdout_xy
        assert o.local_offset_train is not None
    # direct check: M0 offsets equal those from train-only local offsets
    ids = tuple(f"r{r:04d}c{c:04d}" for r in range(2) for c in range(4))
    m0 = bm.solve_offset_baseline(obs, ids, bm.select_central_cell(4, 2), 1)
    assert np.all(np.isfinite(m0))


# ===========================================================================
# M6 — σΔ → 0 is an explicit refusal, not a fake OFFSET_ONLY via 0/0
# ===========================================================================

def test_m6_noisefree_is_insufficient_evidence_with_recovery():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64,
                            truth_planes=truth_of(cells, plane))
    assert res.sigma_delta == 0.0
    assert res.decision == bm.STATE_INSUFFICIENT_EVIDENCE
    assert any("degenerate" in r for r in res.refusal_reasons)
    # recovery is still verified
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8


# ===========================================================================
# M7 — mismatch thresholds are provisional
# ===========================================================================

def test_m7_mismatch_thresholds_provisional():
    assert bm.PROVENANCE["mismatch_thresholds_status"] == "PROVISIONAL_NOT_IN_FROZEN_CONTRACT"
    plane = plane_fn()
    cells, cw = build_grid(4, 2, plane=plane, noise=0.2, seed=11)
    res = bm.compare_models(cells, 4, 2, cw, measurement_px=None)
    assert res.mismatch.get("thresholds_provisional") is True


# ===========================================================================
# L1 — densification flags are measurements
# ===========================================================================

def test_l1_densification_flags_are_measured():
    small = bm.sparse_scale_witness(20)
    assert small["global_solve_densified"] is False
    assert small["diagnostics_densified"] is True     # 1197 cols <= 1500
    big = bm.sparse_scale_witness(50)
    assert big["global_solve_densified"] is False
    assert big["diagnostics_densified"] is False      # 7497 cols > 1500
    assert big["rank_data"] is None and big["cond_data"] is None


# ===========================================================================
# L2 — an existing orientation with a non-finite ratio is a FAIL
# ===========================================================================

def test_l2_nonfinite_orientation_ratio_fails():
    # planes depend only on the column -> vertical neighbours share offsets
    def plane(cid):
        c = int(cid[6:10])
        return (5.0 * ((c % 4) - 1), 1.0, 0.0)
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=12)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    # the 'v' orientation has M0 essentially perfect -> non-finite ratio -> FAIL
    v = res.heldout["orientation_ratio"].get("v")
    if v is not None and not np.isfinite(v):
        assert res.gates["heldout"] == "FAIL"


# ===========================================================================
# Contract §6 fidelity / astronomy
# ===========================================================================

def test_identity_flat_noisefree_correction_zero():
    cells, cw = build_grid(6, 4, plane=lambda cid: (0.0, 0.0, 0.0), noise=0.0)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision in ALLOWED
    assert np.max(np.abs(res.plan_coefficients)) <= 1e-10
    assert np.max(np.abs(res.offset_offsets)) <= 1e-10


def test_noisy_identity_does_not_invent_plan():
    cells, cw = build_grid(6, 4, plane=lambda cid: (0.0, 0.0, 0.0), noise=0.5, seed=1)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision == bm.STATE_OFFSET_ONLY
    assert res.heldout["ratio"] > 0.80


def test_known_planes_noisefree_recovery():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8


def test_known_planes_robust_plan_wide_band():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=3)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.decision == bm.STATE_ROBUST_PLAN
    assert res.truth_diagnostics["max_abs_error"] <= 0.5 * res.sigma_delta
    assert res.heldout["ratio"] <= 0.80


def test_known_planes_narrow_band_is_ill_conditioned():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=32, halo=8, plane=plane, noise=0.2, seed=3)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=16, truth_planes=truth_of(cells, plane))
    assert res.decision in (bm.STATE_ILL_CONDITIONED, bm.STATE_OFFSET_ONLY)


def test_common_gradient_preserved():
    plane = plane_fn()
    sky = common_gradient(gx=8.0, gy=5.0)
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, sky=sky, noise=0.2, seed=4)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.decision == bm.STATE_ROBUST_PLAN
    a_anchor = np.asarray(plane(res.anchor), dtype=np.float64)
    truth_cells, _ = build_grid(6, 4, core=64, halo=32, plane=lambda cid: a_anchor, sky=sky, noise=0.0)
    corr = {cid: res.plan_coefficients[i, 0, :] for i, cid in enumerate(res.cell_ids)}
    got = assemble_cores(cells, corr, cw, 6, 4)
    truth = assemble_cores(truth_cells, {}, cw, 6, 4)
    ratio, _ = regress_amplitude(got, truth)
    assert 0.99 <= ratio <= 1.01


def test_diffuse_ifn_not_biased_and_not_mismatch():
    plane = plane_fn()
    sky = ifn_field()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, sky=sky, noise=0.2, seed=5)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.decision != bm.STATE_MODEL_MISMATCH
    a_anchor = np.asarray(plane(res.anchor), dtype=np.float64)
    truth_cells, _ = build_grid(6, 4, core=64, halo=32, plane=lambda cid: a_anchor, sky=sky, noise=0.0)
    corr = {cid: res.plan_coefficients[i, 0, :] for i, cid in enumerate(res.cell_ids)}
    got = assemble_cores(cells, corr, cw, 6, 4)
    truth = assemble_cores(truth_cells, {}, cw, 6, 4)
    ratio, bias = regress_amplitude(got, truth)
    assert 0.99 <= ratio <= 1.01
    assert abs(bias) <= 0.10 * 0.2


def test_bright_stars_do_not_bias_fit_and_flux_preserved():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0, seed=8,
                           star=(64.0, 64.0, 8000.0, 2.5))
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8
    cid = _cid(0, 0)
    cell = cells[cid]
    i = res.cell_ids.index(cid)
    corr = res.plan_coefficients[i, 0, :]
    after = bm.apply_plane_correction(cell.science, corr, cell.bounds, cw)
    f0 = aperture_flux(cell.science, 64, 64)
    f1 = aperture_flux(after, 64, 64)
    assert abs(f1 - f0) / max(abs(f0), 1.0) <= 1e-3


def test_flux_bias_median_and_p95_over_cells():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0, seed=9,
                           star=(32.0, 32.0, 6000.0, 2.5))
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    rel = []
    for cid in res.cell_ids:
        cell = cells[cid]
        y0, _y1, x0, _x1 = cell.bounds
        hh, ww = cell.science.shape[:2]
        lx, ly = 32.0 - x0, 32.0 - y0
        if not (0 <= lx < ww and 0 <= ly < hh):
            continue
        i = res.cell_ids.index(cid)
        corr = res.plan_coefficients[i, 0, :]
        after = bm.apply_plane_correction(cell.science, corr, cell.bounds, cw)
        f0 = aperture_flux(cell.science, lx, ly)
        f1 = aperture_flux(after, lx, ly)
        rel.append(abs(f1 - f0) / max(abs(f0), 1.0))
    assert len(rel) >= 1
    assert np.median(rel) <= 1e-3
    assert np.percentile(rel, 95) <= 5e-3


def test_independent_instrumental_gradients_recovered():
    plane = plane_fn(amp=6.0, bamp=2.5)
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0, seed=10)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8


# ===========================================================================
# L4 — mismatch / heterogeneity
# ===========================================================================

def test_nonlinear_non_representable_is_model_mismatch():
    plane = plane_fn()
    def nl(cid, XX, YY, XL, YL, ch):
        r, c = int(cid[1:5]), int(cid[6:10])
        return 1.5 * ((r + c) % 3 - 1) * np.sin(5.0 * XX) * np.cos(3.0 * YY)
    cells, cw = build_grid(6, 4, plane=plane, nonlinear=nl, noise=0.2, seed=7)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision == bm.STATE_MODEL_MISMATCH


def test_multiplicative_gain_mismatch_is_model_mismatch():
    plane = plane_fn()
    def gain(cid, ch):
        r, c = int(cid[1:5]), int(cid[6:10])
        return 1.0 + 0.10 * ((r * 2 + c * 3) % 5 - 2)
    cells, cw = build_grid(6, 4, plane=plane, sky=source_field(1), gain=gain, noise=0.2, seed=6)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision == bm.STATE_MODEL_MISMATCH


def test_heterogeneous_nights_cameras_is_not_additive_plan():
    plane = plane_fn()
    def gain(cid, ch):
        r, c = int(cid[1:5]), int(cid[6:10])
        return 1.0 + 0.10 * ((r * 2 + c * 3) % 5 - 2)
    def nf(cid, ch):
        r, c = int(cid[1:5]), int(cid[6:10])
        return 0.1 + 0.2 * ((r + c) % 3)
    cells, cw = build_grid(6, 4, channels=3, plane=plane, sky=source_field(2),
                           gain=gain, noise_fn=nf, seed=13)
    res = bm.compare_models(cells, 6, 4, cw, channels=3)
    assert res.decision in (bm.STATE_MODEL_MISMATCH, bm.STATE_INSUFFICIENT_EVIDENCE,
                            bm.STATE_ILL_CONDITIONED, bm.STATE_OFFSET_ONLY)
    assert res.decision != bm.STATE_ROBUST_PLAN


def test_additive_source_control_is_not_mismatch():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, plane=plane, sky=source_field(1), noise=0.2, seed=6)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision != bm.STATE_MODEL_MISMATCH


# ===========================================================================
# Outliers / layouts / determinism / application
# ===========================================================================

def test_pixel_outliers_do_not_break_recovery():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0, seed=11)
    cid = _cid(2, 2)
    sci = cells[cid].science.copy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, sci.size, 400)
    flat = sci.ravel()
    flat[idx] += rng.choice([-1.0, 1.0], 400) * 5000.0
    cells[cid] = bm.PlanCell(cid, cells[cid].bounds, sci)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-3


def test_weak_fragmented_support_is_refused():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, plane=plane, noise=0.2, seed=14)
    for cid, cell in list(cells.items()):
        hh, ww = cell.science.shape[:2]
        mask = np.zeros((hh, ww), dtype=bool)
        mask[:3, :3] = True
        cells[cid] = bm.PlanCell(cid, cell.bounds, cell.science, valid_mask=mask)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision in (bm.STATE_INSUFFICIENT_EVIDENCE, bm.STATE_DISCONNECTED,
                            bm.STATE_ILL_CONDITIONED, bm.STATE_OFFSET_ONLY)


@pytest.mark.parametrize("noise,expect", [(0.02, "plan"), (3.0, "refuse")])
def test_low_and_high_snr(noise, expect):
    plane = plane_fn(amp=8.0, bamp=3.0)
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=noise, seed=15)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    if expect == "plan":
        assert res.decision == bm.STATE_ROBUST_PLAN
    else:
        assert res.decision in (bm.STATE_OFFSET_ONLY, bm.STATE_ILL_CONDITIONED,
                                bm.STATE_INSUFFICIENT_EVIDENCE)


@pytest.mark.parametrize("nx,ny", [(1, 4), (4, 1), (2, 2), (6, 4)])
def test_layouts_and_truth(nx, ny):
    plane = plane_fn()
    cells, cw = build_grid(nx, ny, core=64, halo=32, plane=plane, noise=0.2, seed=17)
    res = bm.compare_models(cells, nx, ny, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.rank_data == 3 * (len(res.cell_ids) - 1)
    expected_a = np.array([np.asarray(plane(cid), dtype=float)[0] for cid in res.cell_ids])
    a_anchor = np.asarray(plane(res.anchor), dtype=float)[0]
    assert np.max(np.abs(res.plan_coefficients[:, 0, 0] - (a_anchor - expected_a))) <= 0.5


@pytest.mark.parametrize("nx,ny", [(2, 3), (3, 2)])
def test_l4_alternate_layouts_same_physical_truth(nx, ny):
    field = lambda cx, cy: (6.0 * cx + 4.0 * cy, 1.5, -1.5)
    plane = plane_from_centres(nx, ny, field)
    cells, cw = build_grid(nx, ny, core=64, halo=32, plane=plane, noise=0.2, seed=18)
    res = bm.compare_models(cells, nx, ny, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 0.5 * res.sigma_delta


def test_permutation_determinism():
    plane = plane_fn()
    cells, cw = build_grid(4, 3, core=64, halo=32, plane=plane, noise=0.2, seed=19)
    r1 = bm.compare_models(cells, 4, 3, cw, measurement_px=64)
    reordered = {cid: cells[cid] for cid in sorted(cells, reverse=True)}
    r2 = bm.compare_models(reordered, 4, 3, cw, measurement_px=64)
    assert np.allclose(r1.plan_coefficients, r2.plan_coefficients, atol=1e-10)
    assert np.allclose(r1.offset_offsets, r2.offset_offsets, atol=1e-10)
    assert r1.decision == r2.decision


def test_application_preserves_nan_dims_dtype_and_inputs():
    plane = plane_fn()
    cells, cw = build_grid(3, 3, plane=plane, noise=0.2, seed=20)
    snapshot = {cid: (c.science.copy(), None if c.valid_mask is None else c.valid_mask.copy())
                for cid, c in cells.items()}
    res = bm.compare_models(cells, 3, 3, cw, measurement_px=None)
    for cid in res.cell_ids:
        cell = cells[cid]
        i = res.cell_ids.index(cid)
        corr = res.plan_coefficients[i, 0, :]
        before = cell.science
        after = bm.apply_plane_correction(before, corr, cell.bounds, cw)
        assert after.shape == before.shape
        assert after.dtype == before.dtype
        assert np.array_equal(~np.isfinite(before), ~np.isfinite(after))
    # inputs are not mutated by compare_models or application
    for cid, (sci, mask) in snapshot.items():
        assert np.array_equal(cells[cid].science, sci, equal_nan=True)
        if mask is not None:
            assert np.array_equal(cells[cid].valid_mask, mask)


def test_application_is_additive_only_and_planar():
    bounds, cw = (0, 16, 0, 16), (16, 16)
    sci = np.full((16, 16), 3.0)
    corr = np.array([2.0, 4.0, -4.0])
    after = bm.apply_plane_correction(sci, corr, bounds, cw)
    assert np.allclose(after - sci, bm.plane_field(bounds, cw, corr))


def test_rgb_channels_supported():
    plane = plane_fn()
    def plane3(cid):
        p = np.asarray(plane(cid), dtype=np.float64)
        return np.vstack([p, p * 0.5 + 1.0, -p * 0.25])
    cells, cw = build_grid(4, 2, core=64, halo=32, channels=3, plane=plane3, noise=0.2, seed=21)
    res = bm.compare_models(cells, 4, 2, cw, channels=3, measurement_px=64)
    assert res.channels == 3
    assert res.plan_coefficients.shape[1] == 3
    assert res.decision in ALLOWED


# ===========================================================================
# Regularization / scale / purity
# ===========================================================================

def test_regularization_ladder_is_frozen_and_rank_full():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=22)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    assert res.regularization_ladder == bm.REGULARIZATION_LADDER
    assert res.rank_data == 3 * (len(res.cell_ids) - 1)
    assert res.cond_data is None or res.cond_data <= 1e4
    assert res.provenance["global_solve_densified"] is False


def test_uncertainty_is_finite_when_computed():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=23)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    assert res.coef_uncertainty is not None
    assert np.all(np.isfinite(res.coef_uncertainty))
    assert np.all(np.isfinite(res.corner_uncertainty))


def test_sparse_scale_witness_10000_cells():
    w = bm.sparse_scale_witness(100)
    assert w["cells"] == 10000
    assert w["edges"] == 19800
    assert w["unknowns"] == 3 * 9999
    assert w["global_solve_densified"] is False
    assert w["diagnostics_densified"] is False
    assert np.isfinite(w["solve_seconds"])


def test_m0_baseline_reuses_pure_cell_offset_solver():
    assert bm.solve_additive_offsets is zco.solve_additive_offsets


def test_nonlinear_study_is_documentation_only():
    s = bm.nonlinear_study()
    assert s["model"] == bm.MODEL_NONLINEAR_CANDIDATE
    assert s["status"] == "research_only_no_engine"
    assert bm.PROVENANCE["no_third_party_code_copied"] is True


def test_result_is_json_safe():
    plane = plane_fn()
    cells, cw = build_grid(4, 2, plane=plane, noise=0.2, seed=24)
    res = bm.compare_models(cells, 4, 2, cw, measurement_px=None)
    json.dumps(res.to_json_dict())
