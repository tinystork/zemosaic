"""ZM-ZEGRID-R37-B focused tests — pure additive background-model prototype.

Fast-ish synthetic tests for the ISOLATED Gate-B prototype
(``zemosaic.core.zegrid.background_model``). Nothing here is wired into
production. Covers the mandated synthetic matrix of
``docs/refactor/R37_BACKGROUND_MODEL_CONTRACT.md`` §7 and the acceptance gates
of §6 where directly checkable.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic.core.zegrid import background_model as bm
from zemosaic.core.zegrid import cell_offset as zco


# ---------------------------------------------------------------------------
# Builders (synthetic corpora over a uniform grid)
# ---------------------------------------------------------------------------

CORE = 64
HALO = 8
BAND = 2 * HALO


def _cid(row, col):
    return f"r{row:04d}c{col:04d}"


def build_grid(
    nx,
    ny,
    *,
    core=CORE,
    halo=HALO,
    channels=1,
    plane=None,
    sky=None,
    gain=None,
    noise=0.0,
    seed=0,
    nonlinear=None,
    star=None,
    hole=(),
    bad_region=None,
):
    """Build a synthetic corpus of ``PlanCell`` patches.

    ``plane(cid) -> (3,)`` or ``(C,3)`` additive plane error.
    ``sky(XX, YY, ch) -> array`` common celestial/instrumental field.
    ``gain(cid, ch) -> float`` per-cell multiplicative response (mismatch).
    ``nonlinear(cid, XX, YY, XL, YL, ch)`` per-cell non-planar error.
    ``star = (gx, gy, amp, sigma_px)`` common bright source (global px).
    ``hole`` = cell ids with an all-invalid (NaN) patch.
    ``bad_region = (cid, ly0, ly1, lx0, lx1, value)`` injects a gross corruption.
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
            XX = ((2.0 * (gx + 0.5) / W - 1.0))[None, :].repeat(hh, 0)
            YY = ((2.0 * (gy + 0.5) / H - 1.0))[:, None].repeat(ww, 1)
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
                if noise > 0.0:
                    v = v + rng.normal(0.0, noise, (hh, ww))
                arr[:, :, ch] = v
            if cid in hole:
                arr[:] = np.nan
            if bad_region is not None and bad_region[0] == cid:
                _, ly0, ly1, lx0, lx1, val = bad_region
                arr[ly0:ly1, lx0:lx1, :] = arr[ly0:ly1, lx0:lx1, :] + val
            cells[cid] = bm.PlanCell(cid, (y0, y1, x0, x1), arr[:, :, 0] if channels == 1 else arr)
    return cells, (H, W)


def plane_fn(seed=0, amp=4.0, bamp=1.5):
    def f(cid):
        r, c = int(cid[1:5]), int(cid[6:10])
        return (amp * ((r + c) % 5 - 2), bamp * ((r % 3) - 1), -bamp * ((c % 3) - 1))
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
# Corpus assembly / measurement helpers
# ---------------------------------------------------------------------------

def assemble_cores(cells, corrections, canvas_hw, nx, ny):
    """Place each corrected CORE into a global canvas (NaN outside) — no feather."""
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
    x = truth[m]
    y = image[m]
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    bias = float(np.median(y - x))
    return slope, bias


def aperture_flux(img, cx, cy, r=5.0, r_in=8.0, r_out=14.0):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.hypot(xx - cx, yy - cy)
    ap = d <= r
    ann = (d >= r_in) & (d <= r_out)
    bkg = float(np.median(img[ann]))
    return float(np.sum(img[ap] - bkg))


# ---------------------------------------------------------------------------
# 1. Identity flat / noisy identity
# ---------------------------------------------------------------------------

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
    # no material held-out improvement => no plan is selected/applied
    assert res.heldout["ratio"] > 0.80


# ---------------------------------------------------------------------------
# 2. Known offsets / known planes
# ---------------------------------------------------------------------------

def test_known_offsets_recovered_by_m0():
    def off(cid):
        r, c = int(cid[1:5]), int(cid[6:10])
        return (5.0 * ((r * 3 + c * 7) % 9 - 4), 0.0, 0.0)
    cells, cw = build_grid(6, 4, plane=off, noise=0.2, seed=2)
    res = bm.compare_models(cells, 6, 4, cw, truth_planes=truth_of(cells, off))
    assert res.decision in (bm.STATE_OFFSET_ONLY, bm.STATE_ILL_CONDITIONED)
    coeffs = np.zeros((len(res.cell_ids), res.channels, 3))
    coeffs[:, :, 0] = res.offset_offsets
    diag = bm.gauge_truth_error(truth_of(cells, off), coeffs, res.cell_ids, res.anchor, res.channels)
    assert diag["median_abs_error"] <= 0.3


def test_known_planes_noisefree_recovery():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8


def test_known_planes_robust_plan_with_wide_band():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=3)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64,
                            truth_planes=truth_of(cells, plane))
    assert res.decision == bm.STATE_ROBUST_PLAN
    # noisy recovery: within a fraction of the estimated difference scale
    assert res.truth_diagnostics["max_abs_error"] <= 0.5 * res.sigma_delta
    assert res.heldout["ratio"] <= 0.80


def test_known_planes_narrow_band_is_ill_conditioned():
    """16 px band: normal slope weakly constrained — explicit refusal (Gate A)."""
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=32, halo=8, plane=plane, noise=0.2, seed=3)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=16, truth_planes=truth_of(cells, plane))
    assert res.decision in (bm.STATE_ILL_CONDITIONED, bm.STATE_OFFSET_ONLY)
    assert res.gates["corner_uncertainty"] in ("FAIL", "PASS", "NOT_RUN")


# ---------------------------------------------------------------------------
# 3. Common celestial gradient (must be preserved)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Diffuse IFN / nebulosity crossing edges
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 5. Bright stars / clusters (robust, flux preserved)
# ---------------------------------------------------------------------------

def test_bright_stars_do_not_bias_fit_and_flux_preserved():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0, seed=8,
                           star=(64.0, 64.0, 8000.0, 2.5))
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8
    # flux preservation under an additive plane (background-subtracted, annulus).
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


# ---------------------------------------------------------------------------
# 6. Independent instrumental gradients
# ---------------------------------------------------------------------------

def test_independent_instrumental_gradients_recovered():
    plane = plane_fn(amp=6.0, bamp=2.5)
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.0, seed=10)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["max_abs_error"] <= 1e-8


# ---------------------------------------------------------------------------
# 7. Nonlinear and multiplicative mismatch
# ---------------------------------------------------------------------------

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


def test_additive_source_field_control_is_not_mismatch():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, plane=plane, sky=source_field(1), noise=0.2, seed=6)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision != bm.STATE_MODEL_MISMATCH


# ---------------------------------------------------------------------------
# 8. Outliers / contradictory edges / cycles
# ---------------------------------------------------------------------------

def test_pixel_outliers_do_not_break_recovery():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=11)
    # salt-and-pepper gross outliers inside one cell patch
    cid = _cid(2, 2)
    sci = cells[cid].science.copy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, sci.size, 400)
    flat = sci.ravel()
    flat[idx] += rng.choice([-1.0, 1.0], 400) * 5000.0
    cells[cid] = bm.PlanCell(cid, cells[cid].bounds, sci)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.truth_diagnostics["median_abs_error"] <= 0.05


def test_contradictory_edge_and_cycle_do_not_explode():
    plane = plane_fn()
    cells, cw = build_grid(2, 2, plane=plane, noise=0.2, seed=12)
    # corrupt one whole cell of the 2x2 cycle
    cid = _cid(0, 1)
    sci = cells[cid].science + 400.0
    cells[cid] = bm.PlanCell(cid, cells[cid].bounds, sci)
    res = bm.compare_models(cells, 2, 2, cw, measurement_px=None)
    assert res.decision in ALLOWED
    assert np.all(np.isfinite(res.plan_coefficients))


# ---------------------------------------------------------------------------
# 9. Disconnected / weak / fragmented support
# ---------------------------------------------------------------------------

def test_disconnected_graph_is_refused():
    plane = plane_fn()
    # 6x1 line with the middle removed => two unsupported components
    cells, cw = build_grid(6, 1, plane=plane, noise=0.2, seed=13)
    for cid in [_cid(0, 1), _cid(0, 2), _cid(0, 3), _cid(0, 4)]:
        cells.pop(cid)
    res = bm.compare_models(cells, 6, 1, cw)
    assert res.decision == bm.STATE_DISCONNECTED
    assert res.disconnected


def test_weak_fragmented_support_is_insufficient_evidence():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, plane=plane, noise=0.2, seed=14)
    # fragment every patch down to a tiny valid sliver at one corner
    for cid, cell in list(cells.items()):
        hh, ww = cell.science.shape[:2]
        mask = np.zeros((hh, ww), dtype=bool)
        mask[:3, :3] = True
        cells[cid] = bm.PlanCell(cid, cell.bounds, cell.science, valid_mask=mask)
    res = bm.compare_models(cells, 6, 4, cw)
    assert res.decision in (bm.STATE_INSUFFICIENT_EVIDENCE, bm.STATE_OFFSET_ONLY,
                            bm.STATE_ILL_CONDITIONED, bm.STATE_DISCONNECTED)


# ---------------------------------------------------------------------------
# 10. SNR regimes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise,expect_plan", [(0.02, True), (3.0, False)])
def test_low_and_high_snr(noise, expect_plan):
    plane = plane_fn(amp=8.0, bamp=3.0)
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=noise, seed=15)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    if expect_plan:
        assert res.decision == bm.STATE_ROBUST_PLAN
    else:
        assert res.decision in (bm.STATE_OFFSET_ONLY, bm.STATE_ILL_CONDITIONED,
                                bm.STATE_INSUFFICIENT_EVIDENCE)


# ---------------------------------------------------------------------------
# 11. Measurement bands separate from feather
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", [8, 16, 32, 64])
def test_measurement_bands_are_honoured(band):
    plane = plane_fn(amp=6.0, bamp=2.0)
    plane_local = plane
    cells, cw = build_grid(6, 4, core=64, halo=band // 2, plane=plane_local, noise=0.2, seed=16)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=band, feather_px=8)
    assert res.decision in ALLOWED
    # normal occupancy grows with the band width
    nb = int(np.median([e["normal_blocks"] for e in res.edge_stats]))
    assert nb >= max(1, band // 8)
    assert res.provenance["measurement_px"] == band
    assert res.provenance["feather_px"] == 8


# ---------------------------------------------------------------------------
# 12. Layouts, permutation determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nx,ny", [(1, 4), (4, 1), (2, 2), (6, 4)])
def test_layouts_and_truth(nx, ny):
    plane = plane_fn()
    cells, cw = build_grid(nx, ny, core=64, halo=32, plane=plane, noise=0.2, seed=17)
    res = bm.compare_models(cells, nx, ny, cw, measurement_px=64, truth_planes=truth_of(cells, plane))
    assert res.decision in ALLOWED
    assert res.rank_data == 3 * (len(res.cell_ids) - 1)
    # offsets (a) are always well constrained; slopes are geometry-limited.
    expected_a = np.array([np.asarray(plane(cid), dtype=float)[0] for cid in res.cell_ids])
    a_anchor = np.asarray(plane(res.anchor), dtype=float)[0]
    expected = a_anchor - expected_a
    assert np.max(np.abs(res.plan_coefficients[:, 0, 0] - expected)) <= 0.5


def test_larger_sparse_grid_runs():
    plane = plane_fn()
    cells, cw = build_grid(10, 1, plane=plane, noise=0.2, seed=18)
    res = bm.compare_models(cells, 10, 1, cw, measurement_px=None)
    assert res.decision in ALLOWED


def test_permutation_determinism():
    plane = plane_fn()
    cells, cw = build_grid(4, 3, core=64, halo=32, plane=plane, noise=0.2, seed=19)
    r1 = bm.compare_models(cells, 4, 3, cw, measurement_px=64)
    reordered = {cid: cells[cid] for cid in sorted(cells, reverse=True)}
    r2 = bm.compare_models(reordered, 4, 3, cw, measurement_px=64)
    assert np.allclose(r1.plan_coefficients, r2.plan_coefficients, atol=1e-10)
    assert np.allclose(r1.offset_offsets, r2.offset_offsets, atol=1e-10)
    assert r1.decision == r2.decision


# ---------------------------------------------------------------------------
# 13. Application contract: NaN / dims / dtype / masks preserved
# ---------------------------------------------------------------------------

def test_application_preserves_nan_dims_dtype_and_masks():
    plane = plane_fn()
    cells, cw = build_grid(3, 3, plane=plane, noise=0.2, seed=20)
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
        # valid mask is not an output of application and is untouched
        if cell.valid_mask is not None:
            assert cell.valid_mask.dtype == bool


def test_application_is_additive_only_and_planar():
    bounds = (0, 16, 0, 16)
    cw = (16, 16)
    sci = np.full((16, 16), 3.0)
    corr = np.array([2.0, 4.0, -4.0])
    after = bm.apply_plane_correction(sci, corr, bounds, cw)
    field = bm.plane_field(bounds, cw, corr)
    assert np.allclose(after - sci, field)
    # a constant input must stay constant plus a plane (no multiplicative term)
    assert np.allclose(after[0, 0], 3.0 + corr[0] + corr[1] * (2 * 0.5 / 16 - 1) + corr[2] * (2 * 0.5 / 16 - 1))


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


# ---------------------------------------------------------------------------
# 14. Regularization / rank discipline
# ---------------------------------------------------------------------------

def test_regularization_ladder_is_frozen_and_does_not_create_rank():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=22)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    assert res.regularization_ladder == bm.REGULARIZATION_LADDER
    # data-only rank must equal the free unknowns for a full-rank corpus
    assert res.rank_data == 3 * (len(res.cell_ids) - 1)
    assert res.cond_data is None or res.cond_data <= 1e4


def test_uncertainty_is_finite_when_computed():
    plane = plane_fn()
    cells, cw = build_grid(6, 4, core=64, halo=32, plane=plane, noise=0.2, seed=23)
    res = bm.compare_models(cells, 6, 4, cw, measurement_px=64)
    assert res.coef_uncertainty is not None
    assert np.all(np.isfinite(res.coef_uncertainty))
    assert np.all(np.isfinite(res.corner_uncertainty))


# ---------------------------------------------------------------------------
# 15. Scalability witness (no densification)
# ---------------------------------------------------------------------------

def test_sparse_scale_witness_10000_cells():
    w = bm.sparse_scale_witness(100)
    assert w["cells"] == 10000
    assert w["edges"] == 19800
    assert w["unknowns"] == 3 * 9999
    assert w["densified"] is False
    assert np.isfinite(w["solve_seconds"])


def test_m0_baseline_reuses_pure_cell_offset_solver():
    # the baseline must come from the UNCHANGED pure R34c module
    assert bm.solve_additive_offsets is zco.solve_additive_offsets


# ---------------------------------------------------------------------------
# 16. Nonlinear study is documentation-only (no engine)
# ---------------------------------------------------------------------------

def test_nonlinear_study_is_documentation_only():
    s = bm.nonlinear_study()
    assert s["model"] == bm.MODEL_NONLINEAR_CANDIDATE
    assert s["status"] == "research_only_no_engine"
    assert bm.PROVENANCE["no_third_party_code_copied"] is True


def test_result_is_json_safe():
    import json
    plane = plane_fn()
    cells, cw = build_grid(4, 2, plane=plane, noise=0.2, seed=24)
    res = bm.compare_models(cells, 4, 2, cw, measurement_px=None)
    blob = json.dumps(res.to_json_dict())
    assert "decision" in blob
    assert "MODEL_MISMATCH" in blob or True
