"""ZM-ZEGRID-R37-C focused tests — offline pre-feather diagnostic artifact (r0).

Covers the mandated behaviours of mission ``ZM-ZEGRID-R37-C-DIAGNOSTIC``:
strict/byte-stable JSON, schema rejection, dropped relations, multichannel,
band/clamp metadata, zero-noise semantics, deterministic capped holdout
coverage, round-trip model comparability, footprint estimates and the absence of
any production import/wiring.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import numpy as np
import pytest

from zemosaic.core.zegrid import background_diagnostics as bgd
from zemosaic.core.zegrid import background_model as bm


# ---------------------------------------------------------------------------
# Synthetic builders (compact; mirrors the R37-B corpus semantics)
# ---------------------------------------------------------------------------

CORE = 64
HALO = 8


def _cid(r, c):
    return f"r{r:04d}c{c:04d}"


def plane_fn(amp=4.0, bamp=1.5, channels=1):
    def f(cid):
        r, c = int(cid[1:5]), int(cid[6:10])
        base = np.array([amp * ((r + c) % 5 - 2), bamp * (r % 3 - 1), -bamp * (c % 3 - 1)])
        return np.repeat(base[None, :], channels, axis=0)
    return f


def _grid(nx, ny, *, channels=1, core=CORE, halo=HALO, plane=None, noise=0.2,
          seed=0, hole=()):
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
            XX = (2.0 * (np.arange(x0, x1) + 0.5) / W - 1.0)[None, :].repeat(hh, 0)
            YY = (2.0 * (np.arange(y0, y1) + 0.5) / H - 1.0)[:, None].repeat(ww, 1)
            arr = np.zeros((hh, ww, channels), dtype=np.float64)
            for ch in range(channels):
                v = np.zeros((hh, ww), dtype=np.float64)
                if plane is not None:
                    p = np.asarray(plane(cid), dtype=np.float64).reshape(-1, 3)
                    v = v + p[ch, 0] + p[ch, 1] * XX + p[ch, 2] * YY
                if noise > 0.0:
                    v = v + rng.normal(0.0, noise, (hh, ww))
                arr[:, :, ch] = v
            if cid in hole:
                arr[:] = np.nan
            cells[cid] = bm.PlanCell(cid, (y0, y1, x0, x1),
                                     arr[:, :, 0] if channels == 1 else arr)
    return cells, (H, W)


def _cell_ids(nx, ny):
    return tuple(_cid(r, c) for r in range(ny) for c in range(nx))


def _base_payload():
    cells, cw = _grid(3, 2, plane=plane_fn(), noise=0.2, seed=1)
    return bgd.build_payload(cells, 3, 2, cw)


# ===========================================================================
# 1. Strict JSON, byte stability, insertion-order independence
# ===========================================================================

def test_strict_json_byte_stable_and_order_independent():
    cells, cw = _grid(4, 3, plane=plane_fn(), noise=0.2, seed=1)
    payload = bgd.build_payload(cells, 4, 3, cw)
    json.dumps(payload, allow_nan=False)          # must not raise
    text_a = bgd.dumps(payload)
    text_b = bgd.dumps(bgd.build_payload(cells, 4, 3, cw))
    assert text_a == text_b
    shuffled = dict(reversed(list(cells.items())))
    assert bgd.dumps(bgd.build_payload(shuffled, 4, 3, cw)) == text_a


def test_loads_rejects_non_strict_nan_literal():
    with pytest.raises(bgd.DiagnosticSchemaError):
        bgd.loads('{"schema": "x", "y": NaN}')


# ===========================================================================
# 2. Round-trip reconstruction of the sufficient statistics
# ===========================================================================

def test_roundtrip_reconstructs_sufficient_statistics():
    cells, cw = _grid(3, 2, plane=plane_fn(), noise=0.2, seed=2)
    obs, _dropped = bm.build_observations(cells, 3, 2, cw)
    art = bgd.load_artifact(bgd.build_payload(cells, 3, 2, cw, holdout_cap=10 ** 6))
    assert len(obs) == len(art.observations) > 0
    for a, b in zip(obs, art.observations):
        assert a.key() == b.key()
        for field in ("n_train", "n_holdout", "orientation", "measurement_px",
                      "overlap_px", "band_used_px", "band_clamped"):
            assert getattr(a, field) == getattr(b, field)
        assert np.array_equal(a.theta_global, b.theta_global)
        assert np.array_equal(a.information, b.information)
        assert np.array_equal(a.rhs, b.rhs)
        assert a.robust_sigma == b.robust_sigma
        assert a.local_offset_train == b.local_offset_train
        assert a.effective_train == b.effective_train
        assert np.array_equal(a.holdout_xy, b.holdout_xy)
        assert np.array_equal(a.holdout_diff, b.holdout_diff)


# ===========================================================================
# 3. Schema rejection
# ===========================================================================

def _mutate(fn):
    payload = copy.deepcopy(_base_payload())
    fn(payload)
    return payload


_REJECTIONS = {
    "wrong_schema": lambda p: p.__setitem__("schema", "nope"),
    "wrong_version": lambda p: p.__setitem__("schema_version", "r0"),
    "unknown_top_key": lambda p: p.__setitem__("extra", 1),
    "missing_meta_key": lambda p: p["meta"].pop("anchor"),
    "null_theta": lambda p: p["relations"][0].__setitem__("theta_global", [0.0, None, 0.0]),
    "short_information": lambda p: p["relations"][0].__setitem__("information", [[1.0, 2.0], [3.0, 4.0]]),
    "holdout_len_mismatch": lambda p: p["relations"][0].__setitem__("holdout_diff", [0.0]),
    "channel_out_of_range": lambda p: p["relations"][0].__setitem__("channel", 99),
    "channels_zero": lambda p: p["meta"].__setitem__("channels", 0),
    "anchor_not_a_cell": lambda p: p["meta"].__setitem__("anchor", "r9999c9999"),
    "bool_is_not_int": lambda p: p["relations"][0].__setitem__("holdout_capped", 1),
    "reason_empty": lambda p: p.__setitem__("dropped", [{"u": "a", "v": "b", "channel": None, "reason": ""}]),
    "n_relations_mismatch": lambda p: p["meta"].__setitem__("n_relations", 12345),
    "nonfinite_level_z": lambda p: p["relations"][0].__setitem__("level_z", float("inf")),
}


@pytest.mark.parametrize("name", sorted(_REJECTIONS))
def test_schema_rejection(name):
    payload = _mutate(_REJECTIONS[name])
    with pytest.raises(bgd.DiagnosticSchemaError):
        bgd.load_artifact(payload)


# The unmutated payload must load cleanly (guards the rejection tests above).
def test_base_payload_loads_cleanly():
    art = bgd.load_artifact(_base_payload())
    assert art.meta["n_relations"] == len(art.observations)


def test_dropped_nan_relation_array_rejected_after_strict_dump():
    # A payload built from source is strict (allow_nan=False) by construction.
    payload = _base_payload()
    text = bgd.dumps(payload)
    assert "NaN" not in text and "Infinity" not in text


# ===========================================================================
# 4. Missing / fully-invalid (dropped) relations
# ===========================================================================

def test_missing_and_fully_invalid_cell_are_dropped_explicitly():
    cells, cw = _grid(4, 1, plane=plane_fn(), noise=0.2, seed=4)
    del cells["r0000c0003"]                              # absent cell
    holed = "r0000c0002"
    cells[holed] = bm.PlanCell(holed, cells[holed].bounds,
                               np.full_like(cells[holed].science, np.nan))
    art = bgd.load_artifact(bgd.build_payload(cells, 4, 1, cw))
    reasons = {d["reason"] for d in art.dropped}
    assert "cell_absent" in reasons
    assert any(r == "band_no_support" or r == "no_shared_valid_pixels" for r in reasons)
    assert art.payload["meta"]["n_dropped"] == len(art.dropped)
    # surviving relation between c0000 and c0001 is present and channel-scoped
    assert any(o.u == "r0000c0000" and o.v == "r0000c0001" for o in art.observations)
    assert all(isinstance(d["channel"], (int, type(None))) for d in art.dropped)


# ===========================================================================
# 5. Multichannel
# ===========================================================================

def test_multichannel_roundtrip_and_decision():
    cells, cw = _grid(3, 2, channels=3, plane=plane_fn(channels=3), noise=0.2, seed=5)
    payload = bgd.build_payload(cells, 3, 2, cw, channels=3)
    json.dumps(payload, allow_nan=False)
    art = bgd.load_artifact(payload)
    assert art.channels() == 3
    assert {o.channel for o in art.observations} == {0, 1, 2}
    res = bgd.recompute_decision(art)
    assert set(res.per_channel.keys()) == {0, 1, 2}


# ===========================================================================
# 6. Narrow-band / clamp metadata
# ===========================================================================

@pytest.mark.parametrize("band,expect_clamped", [(4, False), (16, False), (64, True)])
def test_band_and_clamp_metadata(band, expect_clamped):
    cells, cw = _grid(4, 2, plane=plane_fn(), noise=0.2, seed=6)
    payload = bgd.build_payload(cells, 4, 2, cw, measurement_px=band)
    assert payload["meta"]["measurement_px"] == band
    rels = payload["relations"]
    assert rels
    for r in rels:
        assert r["measurement_px"] == band
        assert r["band_clamped"] is expect_clamped
        assert r["band_used_px"] <= band if not expect_clamped else r["band_used_px"] <= r["overlap_px"]
    art = bgd.load_artifact(payload)
    assert art.meta["measurement_px"] == band
    assert bgd.recompute_decision(art).provenance["measurement_px"] == band


# ===========================================================================
# 7. Zero-noise semantics (finite zero, never a fake null)
# ===========================================================================

def test_zero_noise_sigma_is_finite_zero_and_never_null():
    cells, cw = _grid(2, 2, plane=plane_fn(), noise=0.0, seed=0)
    payload = bgd.build_payload(cells, 2, 2, cw)
    rels = payload["relations"]
    assert rels
    for r in rels:
        assert r["robust_sigma"] == 0.0
    art = bgd.load_artifact(payload)
    assert all(o.robust_sigma == 0.0 for o in art.observations)
    res = bgd.recompute_decision(art)
    assert res.provenance["noise_estimable"] is False
    assert res.gates["evidence"] == "FAIL"


# ===========================================================================
# 8. Deterministic capped holdout selection: coverage, no prefix bias
# ===========================================================================

def _holdout_fixture(m=2000, seed=11):
    rng = np.random.default_rng(seed)
    xy = np.column_stack([rng.uniform(-1.0, 1.0, m), rng.uniform(-0.05, 0.05, m)])
    diff = np.arange(m, dtype=np.float64)   # encodes the source index
    return xy, diff


def test_capped_holdout_coverage_and_no_prefix_bias():
    xy, diff = _holdout_fixture()
    cap = 37
    sel_xy, sel_diff, capped = bgd.select_capped_holdout(xy, diff, "h", cap=cap)
    assert capped is True
    assert sel_xy.shape[0] == cap
    idx = sel_diff.astype(int)
    # selection integrity: each kept pair is the true (xy, diff) pair
    assert np.allclose(xy[idx], sel_xy)
    # forced extremes preserve the FULL tangent/normal ranges exactly
    assert sel_xy[:, 1].min() == xy[:, 1].min()
    assert sel_xy[:, 1].max() == xy[:, 1].max()
    assert sel_xy[:, 0].min() == xy[:, 0].min()
    assert sel_xy[:, 0].max() == xy[:, 0].max()
    # not an order-biased prefix
    assert not np.array_equal(np.sort(idx), np.arange(cap))
    # deterministic
    again = bgd.select_capped_holdout(xy, diff, "h", cap=cap)
    assert np.array_equal(again[0], sel_xy) and np.array_equal(again[1], sel_diff)
    # tangent quantile strata coverage (all four quartile bands represented)
    qs = np.quantile(xy[:, 1], [0.25, 0.5, 0.75])
    assert set(np.searchsorted(qs, sel_xy[:, 1], side="right").tolist()) == {0, 1, 2, 3}


def test_capped_holdout_is_identity_below_cap():
    xy, diff = _holdout_fixture(m=50, seed=12)
    sel_xy, sel_diff, capped = bgd.select_capped_holdout(xy, diff, "v", cap=64)
    assert capped is False
    assert np.array_equal(sel_xy, xy) and np.array_equal(sel_diff, diff)


# ===========================================================================
# 9. Round-trip model comparability (2x2 and 2x3)
# ===========================================================================

@pytest.mark.parametrize("nx,ny", [(2, 2), (2, 3)])
def test_roundtrip_coefficients_and_holdout_exact_without_capping(nx, ny):
    cells, cw = _grid(nx, ny, plane=plane_fn(), noise=0.2, seed=7)
    obs, _d = bm.build_observations(cells, nx, ny, cw)
    cell_ids = _cell_ids(nx, ny)
    anchor = bm.select_central_cell(nx, ny)

    src = bm.solve_plan_sparse(obs, cell_ids, anchor, 1, reg_lambda=1e-3, with_uncertainty=False)
    art = bgd.load_artifact(bgd.build_payload(cells, nx, ny, cw, holdout_cap=10 ** 6))
    dst = bm.solve_plan_sparse(list(art.observations), cell_ids, anchor, 1,
                               reg_lambda=1e-3, with_uncertainty=False)
    assert np.allclose(src.coefficients, dst.coefficients, atol=bgd.COEF_ROUNDTRIP_ATOL)

    conn = {c: d["connected"] for c, d in src.per_channel.items()}
    m0a = bm.solve_offset_baseline(obs, cell_ids, anchor, 1, channel_connected=conn)
    m0b = bm.solve_offset_baseline(list(art.observations), cell_ids, anchor, 1, channel_connected=conn)
    assert np.allclose(m0a, m0b, atol=bgd.COEF_ROUNDTRIP_ATOL, equal_nan=True)

    index = {c: i for i, c in enumerate(cell_ids)}
    ha = bm.heldout_metrics(obs, src.coefficients, m0a, index, 1)["r_plan"]
    hb = bm.heldout_metrics(list(art.observations), src.coefficients, m0a, index, 1)["r_plan"]
    assert hb == pytest.approx(ha, rel=1e-12, abs=1e-12)

    src_res = bm.compare_models(cells, nx, ny, cw)
    art_res = bgd.recompute_decision(art)
    assert art_res.decision == src_res.decision
    assert art_res.heldout["r_plan"] == pytest.approx(src_res.heldout["r_plan"], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("nx,ny", [(2, 2), (2, 3)])
def test_capped_holdout_metrics_quantified_within_declared_tolerance(nx, ny):
    cells, cw = _grid(nx, ny, plane=plane_fn(), noise=0.2, seed=7)
    src = bm.compare_models(cells, nx, ny, cw)

    payload = bgd.build_payload(cells, nx, ny, cw, holdout_cap=16)
    assert any(r["holdout_capped"] for r in payload["relations"])
    assert all(r["n_holdout_kept"] <= 16 for r in payload["relations"])
    art = bgd.load_artifact(payload)
    res = bgd.recompute_decision(art)

    delta = abs(res.heldout["r_plan"] - src.heldout["r_plan"]) / abs(src.heldout["r_plan"])
    assert delta <= bgd.HELDOUT_RMS_REL_TOL          # quantified + bounded
    assert res.decision == src.decision


def test_capped_holdout_does_not_change_plan_solve_at_fixed_lambda():
    # M1 coefficients depend only on the sufficient statistics, never the held-out
    # sample, so capping must not move them.
    cells, cw = _grid(3, 2, plane=plane_fn(), noise=0.2, seed=8)
    obs, _d = bm.build_observations(cells, 3, 2, cw)
    cell_ids = _cell_ids(3, 2)
    anchor = bm.select_central_cell(3, 2)
    a = bm.solve_plan_sparse(obs, cell_ids, anchor, 1, reg_lambda=1e-3, with_uncertainty=False)
    art = bgd.load_artifact(bgd.build_payload(cells, 3, 2, cw, holdout_cap=4))
    b = bm.solve_plan_sparse(list(art.observations), cell_ids, anchor, 1,
                             reg_lambda=1e-3, with_uncertainty=False)
    assert all(r["holdout_capped"] for r in art.payload["relations"])
    assert np.array_equal(a.coefficients, b.coefficients)


# ===========================================================================
# 10. No production import / wiring
# ===========================================================================

def test_no_production_import_or_wiring():
    assert "zemosaic.zemosaic_zegrid_mode" not in sys.modules
    src_root = pathlib.Path(bm.__file__).resolve().parents[1]
    offenders = []
    for path in src_root.rglob("*.py"):
        if path.name == "background_diagnostics.py":
            continue
        if "background_diagnostics" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path))
    assert offenders == []


def test_artifact_declares_research_only_provenance():
    art = bgd.load_artifact(_base_payload())
    prov = art.provenance
    assert prov["production_wired"] is False
    assert prov["correction_applied"] is False
    assert prov["mission"] == "ZM-ZEGRID-R37-C"
    assert prov["global_dense_matrix"] is False
    assert prov["no_third_party_code_copied"] is True
    assert prov["references"]


# ===========================================================================
# 11. Footprint estimates (no dense object built)
# ===========================================================================

def test_footprint_estimates_counts_and_scaling():
    w = bgd.canonical_witness_estimates()
    c6 = w["canonical_6x4x3"]
    assert c6["n_cells"] == 24
    assert c6["n_relations"] == 38 * 3          # (6*3 + 4*5) edges x 3 channels
    assert c6["global_dense_matrix"] is False
    assert c6["scaling"] == "O(relations*cap)"
    w10 = w["witness_10000_cells"]
    assert w10["n_cells"] == 10000
    assert w10["n_relations"] == 19800          # 2*100*99
    assert w10["holdout_points_total"] == 19800 * bgd.HOLDOUT_CAP
    # analytic estimate is bounded and scales with relations*cap
    est = bgd.estimate_footprint(n_cells=10, n_relations=100)
    assert est["bytes_per_relation_estimate"] > 0
    assert bgd.estimate_footprint(n_cells=10, n_relations=200)["estimated_bytes"] == 2 * est["estimated_bytes"]


def test_measured_canonical_6x4x3_payload_is_bounded():
    cells, cw = _grid(6, 4, channels=3, plane=plane_fn(channels=3), noise=0.2, seed=8)
    payload = bgd.build_payload(cells, 6, 4, cw, channels=3)
    text = bgd.dumps(payload)
    json.dumps(payload, allow_nan=False)
    assert payload["meta"]["n_relations"] == 114
    est = bgd.canonical_witness_estimates()["canonical_6x4x3"]
    assert 0 < len(text) < 2 * est["estimated_bytes"]
    for r in payload["relations"]:
        assert r["n_holdout_kept"] <= bgd.HOLDOUT_CAP


# ===========================================================================
# 12. File round-trip
# ===========================================================================

def test_dump_and_load_file(tmp_path):
    cells, cw = _grid(2, 2, plane=plane_fn(), noise=0.2, seed=9)
    payload = bgd.build_payload(cells, 2, 2, cw)
    path = tmp_path / "diagnostic.json"
    n = bgd.dump_file(path, payload)
    assert n == path.stat().st_size
    art = bgd.load_file(path)
    assert art.meta["n_relations"] == len(art.observations)
    assert art.anchor() in art.cell_ids()
