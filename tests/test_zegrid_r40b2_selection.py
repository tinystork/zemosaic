"""ZM-ZEGRID-R40-B2-G focused tests — EXACT §1.3 held-out selection + re-proof.

Bounded task G closes the two remaining measurable B2 gaps flagged by Nono B2-F §5
and §3, WITHOUT re-evaluating the CONSUMED B1 v2 cohort:

(1) **Exact §1.3 held-out intersection.** The capture's sample set must BE the
    accepted R37 held-out sample set — not merely its candidate geometry superset.
    It is now recovered through the FROZEN production observation
    (``background_model.observe_edge`` with the frozen measurement band and the
    frozen relation normal boundary) plus the frozen §1.3 integer/block inverse
    (round-half-up, ``INV_TOL = 1e-9``, odd/odd parity verified) — never a raw
    floor of normalized floats, never a box/validity approximation. This suite
    PROVES it by measurement against TWO independent references computed from the
    frozen primitives, and asserts the failure mode is closed: the frozen
    candidate-set digest and the final pair-set digest are both bound into the
    generation index and re-derived here from the frozen rule.

(2) **Injected-effect recovery re-proof** under the ACCEPTED frozen B1 v2 criteria
    snapshot (re-derived in-process; its digest must equal the frozen
    ``B1_CRITERIA_SHA256`` recorded by the implementation) on FRESH, disposable
    synthetic fixtures only. The reserved/consumed v2 cohort is NEVER touched, its
    evaluator is never called, and no bundle/result/marker is read.

(3) **Non-blocking hygiene** guards: the run-volatile classifier must exist in
    exactly ONE shared helper used by both the panel and the B2-D suites.

Explicit NON-CLAIMS (all ``NOT_RUN``): this suite does NOT claim R40-B2 acceptance,
the real product ROI parity (I1a/I1b/I2 through the real ROI path), a real Windows
run, R40-C/R40-D or HUMAN_GATE. No real data; no FITS/pixel/image/dataset
traversal of the real corpus; no Windows816/Zealfie/LP341; no product
default/GUI/config change; no correction/model fitting; the capture stays
default-OFF. The recovered/common-support delta keeps the exact
``ORCHESTRATION_ONLY_NON_GATING`` labelling and gates nothing.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import background_model as zbm
from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import r40_capture as rc

R40_BLOCK_KEY = "r40_diagnostic_capture"
BLOCK_PX = 4
CORE = 32
HALO = 8
CHANNELS = 3
# STRICTLY < 2*halo: the frozen measurement band then removes candidates the raw
# overlap-band geometry would keep, so "exact ⊊ candidate geometry" is MEASURED.
MEASUREMENT_PX = 8
R40_BUDGET_DEFAULT = 256 * 1024 * 1024

_TOOL_PATH = (pathlib.Path(__file__).resolve().parents[1]
              / "tools" / "r40b1_synthetic_calibration.py")
_TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _load_b1_tool():
    name = "r40b1_synthetic_calibration"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic fixtures (in-memory CellPatch level; NO FITS, NO real data)
# ---------------------------------------------------------------------------


def _canvas(nx, ny, core=CORE):
    return zg.GlobalCanvas(
        canvas_id=f"c{nx}x{ny}", wcs_header="", width=nx * core, height=ny * core,
        resolution_deg=0.001,
    )


def _make_patches(nx, ny, levels, *, core=CORE, halo=HALO, seed=0, sigma=0.3,
                  hole_col=None, nan_v_channel=None):
    """Deterministic 3-channel patches carrying holes the frozen rules must honour.

    ``hole_col`` sets ``valid_mask = False`` on one global column of every cell
    (channel-invariant, exactly as the frozen ``_valid_2d`` reads channel 0), and
    ``nan_v_channel`` = ``(channel, gx, gys)`` forces NaN science in ONE channel of
    the ``v`` cell of the h relation, so the held-out set is genuinely
    channel-DEPENDENT (the frozen rule requires finiteness per channel).
    """
    rng = np.random.default_rng(seed)
    canvas = _canvas(nx, ny, core)
    patches = {}
    for row in range(ny):
        for col in range(nx):
            cid = f"r{row:04d}c{col:04d}"
            cy0, cy1 = row * core, (row + 1) * core
            cx0, cx1 = col * core, (col + 1) * core
            py0, py1 = max(0, cy0 - halo), min(canvas.height, cy1 + halo)
            px0, px1 = max(0, cx0 - halo), min(canvas.width, cx1 + halo)
            h, w = py1 - py0, px1 - px0
            lv = np.full(3, float(levels[cid]))
            sci = (lv[None, None, :] + rng.normal(0.0, sigma, (h, w, 3))).astype(np.float32)
            valid = np.ones((h, w, 3), dtype=bool)
            surv = np.full((h, w, 3), 7, dtype=np.int64)
            if hole_col is not None and px0 <= hole_col < px1:
                valid[:, int(hole_col) - px0, :] = False
            if nan_v_channel is not None:
                ch, gx, gys = nan_v_channel
                if cid == f"r0000c{nx - 1:04d}":  # the v cell of the first-row h relation
                    for gy in gys:
                        if py0 <= gy < py1 and px0 <= gx < px1:
                            sci[int(gy) - py0, int(gx) - px0, int(ch)] = np.nan
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1),
                core_slice=(cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0),
                science=sci, valid_mask=valid, surviving=surv,
            )
    return canvas, patches


def _levels_2x2():
    return {"r0000c0000": 100.0, "r0000c0001": 104.0,
            "r0001c0000": 96.0, "r0001c0001": 110.0}


def _hole_col():
    """A hole inside the retained measurement band of the h relation (x = 29)."""
    return 29


def _nan_v_channel():
    """NaN science in channel 1 of the v cell, at exact held-out keys (x=30)."""
    return (1, 30, (20, 21, 22, 23))


def _fixture(nx=2, ny=2):
    canvas, patches = _make_patches(
        nx, ny, _levels_2x2(), hole_col=_hole_col(), nan_v_channel=_nan_v_channel(),
    )
    return canvas, patches


def _results_from_patches(patches):
    out = {}
    for cid, cp in patches.items():
        out[cid] = {
            "cell_id": cid, "status": "complete",
            "patch": {
                "bounds": cp.bounds, "core_slice": cp.core_slice,
                "science": cp.science, "valid_mask": cp.valid_mask,
                "surviving": cp.surviving,
            },
            "record": {
                "reference_frame_id": f"ref-{cid}",
                "excluded": [[f"f{i}.fit", "stage", "reason"] for i in range(2)],
            },
        }
    return out


def _covariate_plane(shape, value):
    return np.full(shape[:2], float(value), dtype=np.float64)


def _worker_sparse_for(patches, nx, ny, *, channels=CHANNELS):
    """Worker->parent transport mirror: the covariate samples at EVERY candidate."""
    out: dict = {}
    for (u, v) in zcf.grid_adjacency(nx, ny):
        if u not in patches or v not in patches:
            continue
        coords = zz._r40_candidate_coords(patches, u, v, block_px=BLOCK_PX)
        for cid in (u, v):
            cp = patches[cid]
            shape = np.asarray(cp.science).shape
            for ch in range(channels):
                planes = {
                    "bounds": cp.bounds, "channel": ch,
                    "science": cp.science, "valid_mask": cp.valid_mask,
                    "surviving_sample_count": cp.surviving,
                    "estimator_weight_sum": _covariate_plane(shape, 2.0),
                    "support_w1": _covariate_plane(shape, 3.0),
                    "support_w2": _covariate_plane(shape, 9.0),
                    "n_eff_support": _covariate_plane(shape, 4.0),
                }
                out.setdefault(cid, []).extend(
                    zz._r40_worker_extract_covariates((cid, coords, planes))
                )
    return out


def _science_config():
    return SimpleNamespace(
        normalization="sky_mean", weighting="noise_variance", rejection="kappa_sigma",
        combine="mean", taper="footprint", taper_px=8.0, taper_floor=0.0,
        equalize_rgb=False, sigma_low=3.0, sigma_high=3.0,
        winsor_limit_low=0.05, winsor_limit_high=0.05,
        cell_gauge_mode="per_cell",
    )


def _descs(n=4):
    return [SimpleNamespace(frame_id=SimpleNamespace(logical_path=f"f{i}.fit"))
            for i in range(n)]


def _context(tmp_path, canvas, nx, ny, results, *, budget=None, repo_head="b" * 40):
    return zz._r40_build_seam_context(
        output_root=tmp_path, disk_budget_bytes=(budget or R40_BUDGET_DEFAULT),
        capture_id="a" * 40, descs=_descs(), science_config=_science_config(),
        canvas=canvas, nx=nx, ny=ny, result_by_cid=results,
        halo_px=HALO, measurement_px=MEASUREMENT_PX, feather_px=HALO, block_px=BLOCK_PX,
        channels=CHANNELS, repo_head=repo_head,
    )


def _with_sparse(results, patches, nx, ny):
    for cid, payloads in _worker_sparse_for(patches, nx, ny).items():
        results[cid][zz.R40_SPARSE_RESULT_KEY] = payloads
    return results


def _generations(root):
    return sorted(pathlib.Path(root).glob("r40_capture_*"))


# ---------------------------------------------------------------------------
# INDEPENDENT references, computed from the frozen primitives only
# ---------------------------------------------------------------------------


def _overlap_band(bounds_u, bounds_v):
    uy0, uy1, ux0, ux1 = bounds_u
    vy0, vy1, vx0, vx1 = bounds_v
    return (max(uy0, vy0), min(uy1, vy1), max(ux0, vx0), min(ux1, vx1))


def _boundary(u, v, nx, ny, canvas):
    """Frozen normal boundary, re-derived INDEPENDENTLY in the test."""
    ur, uc = zz._r40_cell_index(u)
    vr, vc = zz._r40_cell_index(v)
    if vr > ur:
        return float((ur + 1) * (float(canvas.height) / float(ny)))
    return float((uc + 1) * (float(canvas.width) / float(nx)))


def _reference_candidate_geometry(patches, u, v, *, block_px=BLOCK_PX):
    """INDEPENDENT re-derivation of the frozen block-parity candidate geometry."""
    y0, y1, x0, x1 = _overlap_band(patches[u].bounds, patches[v].bounds)
    return tuple(
        (gx, gy) for gy in range(y0, y1) for gx in range(x0, x1)
        if (gx // block_px) % 2 == 1 and (gy // block_px) % 2 == 1
    )


def _reference_exact_keys(canvas, patches, nx, ny, *, block_px=BLOCK_PX,
                          measurement_px=MEASUREMENT_PX):
    """INDEPENDENT dense-raster reference for the exact R37 held-out keys.

    Recomputed from the frozen primitives only (``shared_valid_mask`` + the frozen
    band/parity definition), with NO call into the capture's own selection path.
    """
    out: dict = {}
    for (u, v) in zcf.grid_adjacency(nx, ny):
        if u not in patches or v not in patches:
            continue
        pu, pv = patches[u], patches[v]
        iu, iv = zz._r40_cell_index(u), zz._r40_cell_index(v)
        orientation = "h" if iu[0] == iv[0] else "v"
        boundary = _boundary(u, v, nx, ny, canvas)
        uy0, _, ux0, _ = pu.bounds
        vy0, _, vx0, _ = pv.bounds
        y0, y1, x0, x1 = _overlap_band(pu.bounds, pv.bounds)
        for ch in range(CHANNELS):
            su = np.asarray(pu.science)[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0, ch:ch + 1]
            sv = np.asarray(pv.science)[y0 - vy0:y1 - vy0, x0 - vx0:x1 - vx0, ch:ch + 1]
            vu = np.asarray(pu.valid_mask)[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0, ch:ch + 1]
            vv = np.asarray(pv.valid_mask)[y0 - vy0:y1 - vy0, x0 - vx0:x1 - vx0, ch:ch + 1]
            shared = zcf.shared_valid_mask(su, vu, sv, vv)
            keys = []
            for gy in range(y0, y1):
                for gx in range(x0, x1):
                    if (gx // block_px) % 2 != 1 or (gy // block_px) % 2 != 1:
                        continue
                    normal = (gx + 0.5) if orientation == "h" else (gy + 0.5)
                    if abs(normal - boundary) > (measurement_px / 2.0) + 1e-12:
                        continue
                    if bool(shared[gy - y0, gx - x0]):
                        keys.append((gx, gy))
            out[(u, v, ch)] = tuple(sorted(keys))
    return out


def _frozen_observation_reference(canvas, patches, nx, ny, *, block_px=BLOCK_PX,
                                  measurement_px=MEASUREMENT_PX):
    """Second reference: the FROZEN production observation + the frozen inverse."""
    from zemosaic.core.zegrid import background_diagnostics as _bgd

    out: dict = {}
    canvas_hw = (int(canvas.height), int(canvas.width))
    for (u, v) in zcf.grid_adjacency(nx, ny):
        if u not in patches or v not in patches:
            continue
        for ch in range(CHANNELS):
            obs, _reason = zbm.observe_edge(
                _bgd._as_plan_cell(patches[u]), _bgd._as_plan_cell(patches[v]), canvas_hw,
                ch, block_px=block_px, measurement_px=measurement_px,
                normal_boundary=_boundary(u, v, nx, ny, canvas),
            )
            keys = () if obs is None else rc.heldout_keys_from_xy(
                obs.holdout_xy, canvas_width=int(canvas.width),
                canvas_height=int(canvas.height), block_px=block_px,
            )
            out[(u, v, ch)] = tuple(sorted(keys))
    return out


def _capture_pair_set(gen_dir):
    """Decode the on-disk shards: {(u, v, channel): ((gx, gy), ...)} + the index."""
    gen_dir = pathlib.Path(gen_dir)
    index = json.loads((gen_dir / "index.json").read_text())
    out: dict = {}
    for entry in index["shards"]:
        raw = (gen_dir / entry["filename"]).read_bytes()
        array = np.frombuffer(raw, dtype=rc.R40_RECORD_DTYPE)
        per: dict = {}
        for row in array:
            per.setdefault(int(row["channel"]), []).append(
                (int(row["gx"]), int(row["gy"]))
            )
        for ch, keys in per.items():
            out[(entry["u"], entry["v"], ch)] = tuple(sorted(keys))
    return out, index


def _run_capture(tmp_path, nx=2, ny=2):
    canvas, patches = _fixture(nx, ny)
    results = _with_sparse(_results_from_patches(patches), patches, nx, ny)
    ctx = _context(tmp_path, canvas, nx, ny, results)
    _, info = zz._assemble_per_cell_feathered(
        canvas, nx, ny, results, r40_capture=True, r40_context=ctx,
    )
    gens = _generations(tmp_path)
    assert len(gens) == 1, gens
    return canvas, patches, gens[0], info, ctx


# ---------------------------------------------------------------------------
# 1. The frozen §1.3 inverse itself
# ---------------------------------------------------------------------------


def test_frozen_inverse_round_trips_every_boundary_index():
    for dim in (64, 600):
        for index in sorted({0, 1, BLOCK_PX - 1, BLOCK_PX, BLOCK_PX + 1,
                             dim // 2, dim - 2, dim - 1}):
            assert rc.inverse_index(rc.forward_coord(index, dim), dim) == index


def test_frozen_inverse_refuses_ambiguity_and_out_of_range():
    with pytest.raises(rc.IdentityError):
        rc.inverse_index(rc.forward_coord(10, 400) + 1e-6, 400)
    with pytest.raises(rc.IdentityError):
        rc.inverse_index(1.5, 400)          # recovered index = 499, out of range
    with pytest.raises(rc.IdentityError):
        rc.inverse_index(float("nan"), 400)


def test_heldout_keys_require_the_odd_odd_block_parity():
    W = H = 64
    good = [(float(((gx + 0.5) * 2 / W - 1)), float(((gy + 0.5) * 2 / H - 1)))
            for gx, gy in ((29, 5), (31, 29))]
    assert rc.heldout_keys_from_xy(good, canvas_width=W, canvas_height=H,
                                   block_px=BLOCK_PX) == ((29, 5), (31, 29))
    # a parity-violating (even-block) key is REFUSED, never silently kept
    bad = [(float(((gx + 0.5) * 2 / W - 1)), float(((gy + 0.5) * 2 / H - 1)))
           for gx, gy in ((8, 8),)]
    with pytest.raises(rc.IdentityError):
        rc.heldout_keys_from_xy(bad, canvas_width=W, canvas_height=H, block_px=BLOCK_PX)


# ---------------------------------------------------------------------------
# 2. MEASURED: the exact set is a STRICT subset of the candidate geometry
# ---------------------------------------------------------------------------


def test_exact_selection_is_strictly_smaller_than_the_candidate_geometry():
    """If the capture merely used the candidate geometry (or the pre-change
    candidate ∩ shared-valid approximation), this measurement would fail."""
    canvas, patches = _fixture()
    exact = _reference_exact_keys(canvas, patches, 2, 2)
    strictly_smaller = 0
    for (u, v) in zcf.grid_adjacency(2, 2):
        candidates = set(_reference_candidate_geometry(patches, u, v))
        for ch in range(CHANNELS):
            keys = set(exact[(u, v, ch)])
            assert keys <= candidates, (u, v, ch)
            if len(keys) < len(candidates):
                strictly_smaller += 1
    # measured: EVERY relation-channel (4 adjacency relations x 3 channels) is
    # strictly smaller than the candidate geometry (the frozen measurement band
    # alone removes half the candidates and the validity/finiteness rules remove
    # more)
    n_relations = len(list(zcf.grid_adjacency(2, 2)))
    assert n_relations == 4
    assert strictly_smaller == n_relations * CHANNELS
    # and the exact set is non-vacuous
    assert all(len(k) > 0 for k in exact.values())
    # the fixture really exercises the channel dependence (NaN in channel 1 only)
    h_rel = ("r0000c0000", "r0000c0001")
    assert len(exact[(h_rel[0], h_rel[1], 1)]) < len(exact[(h_rel[0], h_rel[1], 0)])


# ---------------------------------------------------------------------------
# 3. MEASURED: the capture's pair set IS the exact held-out set
# ---------------------------------------------------------------------------


def test_capture_pair_set_equals_the_independent_reference_exact_keys(tmp_path):
    canvas, patches, gen, info, _ctx = _run_capture(tmp_path)
    captured, index = _capture_pair_set(gen)
    reference = _reference_exact_keys(canvas, patches, 2, 2)

    assert set(captured) == set(reference)
    for key, keys in reference.items():
        assert captured[key] == keys, key
    # the capture's records ARE the exact set — not the candidate geometry
    for (u, v, _ch), keys in captured.items():
        candidates = set(_reference_candidate_geometry(patches, u, v))
        assert set(keys) < candidates
    # on-disk bookkeeping agrees with what was measured
    assert index["n_records"] == sum(len(k) for k in reference.values())
    assert info[R40_BLOCK_KEY]["n_records"] == index["n_records"]


def test_capture_pair_set_equals_the_frozen_observation_reference(tmp_path):
    canvas, patches, gen, _info, _ctx = _run_capture(tmp_path)
    captured, _index = _capture_pair_set(gen)
    observation = _frozen_observation_reference(canvas, patches, 2, 2)
    dense = _reference_exact_keys(canvas, patches, 2, 2)
    # the two independent references agree with each other ...
    assert observation == dense
    # ... and the capture IS that set
    assert captured == observation


# ---------------------------------------------------------------------------
# 4. MEASURED: both digests are bound to the frozen rule
# ---------------------------------------------------------------------------


def test_candidate_and_pairset_digests_are_bound_to_the_frozen_rule(tmp_path):
    canvas, patches, gen, _info, _ctx = _run_capture(tmp_path)
    _captured, index = _capture_pair_set(gen)
    reference = _reference_exact_keys(canvas, patches, 2, 2)

    checked = 0
    for entry in index["shards"]:
        u, v = entry["u"], entry["v"]
        # candidate-set digest: recomputed from the INDEPENDENT frozen block-parity
        # geometry of this relation
        geometry = _reference_candidate_geometry(patches, u, v)
        assert entry["candidate_set_digest"] == rc.candidate_set_digest(geometry)
        assert entry["n_candidates"] == len(geometry)
        # pair-set digest: recomputed from the INDEPENDENT exact reference keys
        keys = []
        for ch in range(CHANNELS):
            for (gx, gy) in reference[(u, v, ch)]:
                keys.append(rc.sample_key(entry["relation_id"], ch, gx, gy))
        assert entry["pairset_digest"] == rc.pairset_digest(keys)
        checked += 1
    n_relations = len(list(zcf.grid_adjacency(2, 2)))
    assert checked == len(index["shards"]) == n_relations

    # and the published generation independently verifies, including the frozen
    # odd/odd parity re-derivation from the raw bytes
    report = rc.verify_generation(gen, expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []


# ---------------------------------------------------------------------------
# 5. MEASURED: fail-closed on an incomplete candidate transport
# ---------------------------------------------------------------------------


def test_missing_candidate_for_an_exact_key_refuses_and_publishes_nothing(
    tmp_path, monkeypatch
):
    canvas, patches = _fixture()
    results = _with_sparse(_results_from_patches(patches), patches, 2, 2)
    ctx = _context(tmp_path, canvas, 2, 2, results)
    reference = _reference_exact_keys(canvas, patches, 2, 2)
    dropped = reference[("r0000c0000", "r0000c0001", 0)][0]

    real = zz._r40_candidate_coords

    def _lossy(patches_, u, v, *, block_px):
        return tuple(c for c in real(patches_, u, v, block_px=block_px) if c != dropped)

    monkeypatch.setattr(zz, "_r40_candidate_coords", _lossy)
    with pytest.raises(rc.SchemaError):
        zz._assemble_per_cell_feathered(
            canvas, 2, 2, results, r40_capture=True, r40_context=ctx,
        )
    # fail-closed: NOTHING was published and no partial capture survives
    assert _generations(tmp_path) == []


# ---------------------------------------------------------------------------
# 6. Injected-effect recovery RE-PROOF under the frozen B1 v2 criteria
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def b1():
    return _load_b1_tool()


@pytest.fixture(scope="module")
def frozen_criteria_snapshot(b1):
    """The ACCEPTED B1 v2 criteria snapshot, re-derived IN-PROCESS.

    Re-derivation never touches the consumed cohort/bundle/result/markers: it runs
    ONLY the calibration families and the frozen criteria derivation. The suite
    asserts the resulting digest equals the frozen ``B1_CRITERIA_SHA256`` the
    implementation records, so this IS the accepted snapshot.
    """
    calibration = b1.run_calibration()
    snapshot = b1.freeze_snapshot(
        "criteria", b1.TAG_CRITERIA, b1.derive_frozen_criteria(calibration)
    )
    return snapshot


def _fresh_reproof_specs(b1):
    """FRESH, disposable fixtures (new seeds/group ids; no frozen cohort reuse)."""
    return [
        b1.ScenarioSpec("reproof_g_isolated_moderate", "calibration", 910001,
                        "panel.isolated", ("isolated", "moderate")),
        b1.ScenarioSpec("reproof_g_isolated_strong", "calibration", 910002,
                        "panel.isolated", ("isolated", "strong")),
        b1.ScenarioSpec("reproof_g_null", "calibration", 910003,
                        "panel.null", ("null", "none")),
        b1.ScenarioSpec("reproof_g_confounded", "calibration", 910004,
                        "panel.confounded", ("confounded", "moderate")),
    ]


def _recovery_verdict(evidence, crit):
    """The frozen §15.5-5 injected-effect recovery rule, on the frozen gates."""
    if float(evidence["reduction_rms"]) < float(crit["reduction_rms_min"]):
        return "NO_EFFECT"
    if float(evidence["p90_worsening"]) > float(crit["p90_worsening_max"]):
        return "NO_EFFECT"
    if float(evidence["rms0_sigma_ratio"]) > float(crit["rms0_sigma_gate"]):
        return "NO_EFFECT"
    witnesses = evidence.get("orientation_witnesses") or {}
    need = int(crit["relations_per_orientation_min"])
    for orient in ("h", "v"):
        if int(witnesses.get(orient, 0)) < need:
            return "NOT_IDENTIFIABLE"
    if not bool(evidence.get("signature_consistent")):
        return "NOT_IDENTIFIABLE"
    return "RECOVERED"


def test_frozen_criteria_snapshot_is_the_accepted_b1_v2_criteria(
    b1, frozen_criteria_snapshot
):
    """Record + assert the exact digest of the criteria used by the re-proof."""
    snapshot = frozen_criteria_snapshot
    snapshot.verify()
    assert snapshot.sha256 == rc.B1_CRITERIA_SHA256
    crit = snapshot.frozen["injected_effect_recovery"]
    gates = b1.INHERITED_GATES
    # the frozen criteria and the frozen rule implementation agree EXACTLY
    assert float(crit["reduction_rms_min"]) == float(gates["reduction_min"])
    assert float(crit["p90_worsening_max"]) == float(gates["p90_worsening_max"])
    assert float(crit["rms0_sigma_gate"]) == float(gates["rms0_sigma_gate"])
    assert int(crit["relations_per_orientation_min"]) == \
        int(gates["relations_per_orientation_min"])
    assert int(crit["channel_promote_min"]) == int(gates["channel_promote_min"])
    # the NON-GATING labelling of the recovered delta is preserved EXACTLY
    assert crit["authority"] == "ORCHESTRATION_ONLY"
    assert crit["recovered_delta_is_gating"] is False
    assert crit["injected_amplitude_is_generator_metadata_only"] is True
    assert snapshot.frozen["recovered_delta_diagnostic"]["label"] == \
        "ORCHESTRATION_ONLY_NON_GATING"
    assert snapshot.frozen["recovered_delta_diagnostic"]["consumed_by_h4_admission"] is False


def test_injected_effect_recovery_reproof_under_the_frozen_v2_criteria(
    b1, frozen_criteria_snapshot
):
    crit = frozen_criteria_snapshot.frozen["injected_effect_recovery"]
    verdicts = {}
    for spec in _fresh_reproof_specs(b1):
        sim = b1.simulate_panel(spec)
        evidence = sim["evidence"]
        assert int(evidence["n_relations"]) == int(
            frozen_criteria_snapshot.frozen["h4_controlled_effect_admission"][
                "panel_relation_count"]
        )
        # relation-first + explicit 2-of-3 channel rule is genuinely applied
        for rel in sim["relation_records"]:
            assert len(rel["channel_states"]) == CHANNELS
            if rel["state"] == "PROMOTED":
                assert rel["n_channels_promoted"] >= int(crit["channel_promote_min"])
        verdicts[spec.params[0]] = _recovery_verdict(evidence, crit)

    # the injected effect is RECOVERED on the fresh fixtures, under the frozen gates
    assert verdicts["isolated"] == "RECOVERED"
    # and the null fixture never promotes (no default pass)
    assert verdicts["null"] == "NO_EFFECT"

    # the confounded fixture carries TWO intervention families, so the frozen H4
    # admission rule (one isolated family) can never admit it
    confounded = b1.simulate_panel(_fresh_reproof_specs(b1)[3])["evidence"]
    assert len(confounded["intervention_families"]) != 1
    assert confounded["authority"] == "ORCHESTRATION_ONLY"


# ---------------------------------------------------------------------------
# 7. Hygiene guard: ONE shared run-volatile classifier
# ---------------------------------------------------------------------------


def test_run_volatile_classifier_lives_in_exactly_one_shared_helper():
    helper = importlib.import_module("r40b2_run_volatile")
    for name in ("test_zegrid_r40b2_panel.py", "test_zegrid_r40b2_offon_identity.py"):
        text = (_TESTS_DIR / name).read_text()
        # no local re-definition (the drift risk this task removes) ...
        assert "def _is_run_volatile" not in text, name
        assert "_VOLATILE_PREFIXES = (" not in text, name
        assert "def _prefix_hit" not in text, name
        # ... it IMPORTS the single shared helper
        assert "from r40b2_run_volatile import" in text, name
    # both suites, once collected, bind the IDENTICAL function object
    for mod_name in ("test_zegrid_r40b2_panel", "test_zegrid_r40b2_offon_identity"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            assert mod._is_run_volatile is helper.is_run_volatile, mod_name
            assert mod._is_measurement_volatile is helper.is_measurement_volatile, mod_name
