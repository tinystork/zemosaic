"""ZM-CELL-PHOTOMETRY-R34B targeted tests — per-Cell photometric normalization.

Fast, synthetic (no gated corpora) + one gated M106 parity check. Covers the
R34b deliverables:

* the opt-in ``cell_gauge_mode`` token (``"global"`` default / ``"per_cell"``),
  reachable only via an explicit config/keyword — the default path is unchanged;
* ``compute_cell_gauge`` reuses the FROZEN estimator verbatim and selects the
  Cell's OWN reference (greatest valid support, tie -> lowest sorted index);
* the EXACT Caldwell failure shape (a Cell whose frames are disjoint from any
  global reference) now calibrates in ``per_cell`` mode while the ``global``
  subset path still raises ``cell gauge has no active frame``;
* a zero-support Cell is an explicit, deterministic ``status="empty"`` result
  (never a fabricated gauge, never an exception);
* mono + RGB, reference == only frame, single-frame Cell;
* the ``global`` path dispatches to ``subset_fixed_normalization`` unchanged
  (byte-identical production path) and the per-Cell gauge applied to the stack is
  bit-equal to the frozen per-Cell (``fixed=None``) computation.
"""

from __future__ import annotations

import numpy as np
import pytest

import zemosaic.core.canonical_stacking as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    compute_fixed_normalization,
    run_canonical_stack_streaming,
    subset_fixed_normalization,
)
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid import science_adapter as zs
from zemosaic.core.zegrid.executor import ExecutorConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _request(n, **kw):
    base = dict(
        images=[None] * n,
        geometric_support=[None] * n,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="kappa_sigma",
        combine="mean",
        taper="footprint",
        taper_px=8.0,
    )
    base.update(kw)
    return CanonicalStackRequest(**base)


def _make_frame(base, mask, level, rng, sigma=0.5):
    h, w = base.shape
    data = np.full((h, w), np.nan, dtype=np.float32)
    noise = rng.normal(0.0, sigma, (h, w))
    data[mask] = (base + level + noise)[mask].astype(np.float32)
    return data


def _to_rgb(mono_frames, per_channel_offsets=(0.0, 4.0, 8.0)):
    """Turn mono frames into RGB frames (per-channel additive offsets)."""
    out = []
    for f in mono_frames:
        chans = []
        for off in per_channel_offsets:
            g = np.array(f, dtype=np.float32, copy=True)
            finite = np.isfinite(g)
            g[finite] = g[finite] + np.float32(off)
            chans.append(g)
        out.append(np.stack(chans, axis=-1).astype(np.float32))
    return out


def _cluster_corpus(h=48, w=48, n_a=3, n_b=3, seed=11, rgb=False):
    """Two spatial clusters: A on the LEFT half, B on the RIGHT half (disjoint).

    All frames carry the SAME valid count, so the global reference is the lowest
    index (a cluster-A frame); every cluster-B frame is DISJOINT from it and is
    excluded by the global gauge. This is the Caldwell "no active frame" shape.
    """
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.normal(0.0, 1.0, (h, w))
    left = np.zeros((h, w), dtype=bool)
    left[:, : w // 2] = True
    right = np.zeros((h, w), dtype=bool)
    right[:, w // 2 :] = True

    frames, masks = [], []
    for i in range(n_a):
        frames.append(_make_frame(base, left, i * 3.0, rng))
        masks.append(left)
    for i in range(n_b):
        frames.append(_make_frame(base, right, (5 + i) * 3.0, rng))
        masks.append(right)
    if rgb:
        frames = _to_rgb(frames)
    return frames, masks, left, right, n_a, n_b


# ---------------------------------------------------------------------------
# Mode token (opt-in, default global)
# ---------------------------------------------------------------------------

def test_resolve_cell_gauge_mode_and_default():
    assert zphot.CELL_GAUGE_MODE_DEFAULT == "global"
    assert set(zphot.CELL_GAUGE_MODES) == {"global", "per_cell"}
    assert zphot.resolve_cell_gauge_mode(None) == "global"
    assert zphot.resolve_cell_gauge_mode("global") == "global"
    assert zphot.resolve_cell_gauge_mode("per_cell") == "per_cell"
    assert zphot.resolve_cell_gauge_mode(" PER_CELL ") == "per_cell"
    for bad in ("", "per-cell", "percell", "local", 3, True):
        with pytest.raises(ValueError):
            zphot.resolve_cell_gauge_mode(bad)


def test_science_config_default_mode_is_global():
    assert zs.MiniTileScienceConfig().cell_gauge_mode == "global"
    assert ExecutorConfig().science_config().cell_gauge_mode == "global"


# ---------------------------------------------------------------------------
# Reference selection (deterministic, max support, tie -> lowest index)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rgb", [False, True])
def test_cell_gauge_reference_is_max_support_tiebreak(rgb):
    h, w = 32, 32
    rng = np.random.default_rng(3)
    base = 100.0 + rng.normal(0.0, 1.0, (h, w))
    full = _full_support(h, w)
    half = _full_support(h, w)
    half[:, w // 2 :] = False
    frames = [
        _make_frame(base, full, 0.0, rng),   # index 0: full support
        _make_frame(base, full, 3.0, rng),   # index 1: full support (ties with 0)
        _make_frame(base, half, 6.0, rng),   # index 2: half support -> not the ref
    ]
    masks = [full, full, half]
    if rgb:
        frames = _to_rgb(frames)

    prov = InMemoryCanonicalProvider(frames, masks)
    res = zphot.compute_cell_gauge(prov, _request(3))
    assert res.status == zphot.CELL_GAUGE_STATUS_OK
    assert res.frame_ids is None  # InMemory provider exposes no frame ids
    # greatest support = full; tie (0 vs 1) -> lowest index 0.
    assert res.reference_index == 0
    # deterministic on recompute.
    assert zphot.compute_cell_gauge(prov, _request(3)).reference_index == 0
    # reference carries the identity coefficient.
    ref_c = res.gauge.coefficients[0]
    assert np.allclose(ref_c[:, 0], 1.0)
    assert np.allclose(ref_c[:, 1], 0.0)
    assert res.gauge.frame_ids is None


@pytest.mark.parametrize("rgb", [False, True])
def test_cell_gauge_single_frame_is_its_own_reference(rgb):
    h, w = 24, 24
    rng = np.random.default_rng(5)
    base = 100.0 + rng.normal(0.0, 1.0, (h, w))
    full = _full_support(h, w)
    frames = [_make_frame(base, full, 7.0, rng)]
    if rgb:
        frames = _to_rgb(frames)
    prov = InMemoryCanonicalProvider(frames, [full])
    res = zphot.compute_cell_gauge(prov, _request(1))
    assert res.status == zphot.CELL_GAUGE_STATUS_OK
    assert res.reference_index == 0
    assert res.active_frames == 1
    assert np.allclose(res.gauge.coefficients[0][:, 0], 1.0)
    assert np.allclose(res.gauge.coefficients[0][:, 1], 0.0)


# ---------------------------------------------------------------------------
# Zero-support Cell: explicit + deterministic (never fabricated)
# ---------------------------------------------------------------------------

def test_cell_gauge_zero_support_is_explicit_and_deterministic():
    h, w = 20, 20
    full = _full_support(h, w)
    nan_frame = np.full((h, w), np.nan, dtype=np.float32)
    prov = InMemoryCanonicalProvider([nan_frame, nan_frame.copy()], [full, full])
    res = zphot.compute_cell_gauge(prov, _request(2))
    assert res.status == zphot.CELL_GAUGE_STATUS_EMPTY
    assert res.gauge is None
    assert res.reference_index is None
    assert res.active_frames == 0
    assert res.total_valid_support == 0
    assert res.detail == "zero_valid_support"
    # deterministic: the SAME explicit result, no exception, no fabricated gauge.
    res2 = zphot.compute_cell_gauge(prov, _request(2))
    assert res2 == res


# ---------------------------------------------------------------------------
# The Caldwell failure shape: disjoint-from-global Cell now calibrates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rgb", [False, True])
def test_global_subset_raises_for_disjoint_cell(rgb):
    frames, masks, _left, _right, n_a, n_b = _cluster_corpus(rgb=rgb)
    req = _request(len(frames))
    gauge = compute_fixed_normalization(
        InMemoryCanonicalProvider(frames, masks), req
    )
    # reference is a cluster-A frame; the cluster-B frames are excluded globally.
    assert 0 <= gauge.reference_index < n_a
    cell_ids = list(range(n_a, n_a + n_b))
    assert not gauge.norm_active[cell_ids].any()
    # the pre-R34b global subset path FAILS for this Cell (the exact Caldwell shape):
    with pytest.raises(cs.CanonicalStackValidationError, match="cell gauge has no active frame"):
        subset_fixed_normalization(gauge, list(range(len(frames))), cell_ids)


@pytest.mark.parametrize("rgb", [False, True])
def test_per_cell_mode_calibrates_disjoint_cell(rgb):
    frames, masks, _left, _right, n_a, n_b = _cluster_corpus(rgb=rgb)
    cell_ids = list(range(n_a, n_a + n_b))
    cell_frames = [frames[i] for i in cell_ids]
    cell_masks = [masks[i] for i in cell_ids]

    prov = InMemoryCanonicalProvider(cell_frames, cell_masks)
    res = zphot.compute_cell_gauge(prov, _request(len(cell_ids)))
    assert res.status == zphot.CELL_GAUGE_STATUS_OK
    # Cell-local reference = greatest support, tie -> lowest local index (frame 0).
    assert res.reference_index == 0
    # every contributor is active against the Cell's own reference (no global
    # reference needed): the disjoint Cell is now self-consistent.
    assert res.gauge.norm_active.all()
    assert res.gauge.weight_active.all()
    assert np.allclose(res.gauge.coefficients[0][:, 0], 1.0)
    assert np.allclose(res.gauge.coefficients[0][:, 1], 0.0)
    # The Cell's frames carry DIFFERENT sky levels -> real nonzero offsets.
    offs = res.gauge.coefficients[:, 0, 1]
    assert np.any(np.abs(offs[1:]) > 1.0)


def test_per_cell_gauge_lands_on_cell_reference_level():
    """Self-consistency: stacking the disjoint Cell with its own gauge produces
    science on the CELL reference's sky level (residual ~= 0 across two subsets
    of the same Cell that share the Cell reference)."""
    frames, masks, _left, _right, n_a, n_b = _cluster_corpus()
    cell_ids = list(range(n_a, n_a + n_b))

    def run(ids):
        sub_frames = [frames[i] for i in ids]
        sub_masks = [masks[i] for i in ids]
        prov = InMemoryCanonicalProvider(sub_frames, sub_masks)
        res = zphot.compute_cell_gauge(prov, _request(len(ids)))
        assert res.status == zphot.CELL_GAUGE_STATUS_OK
        out = run_canonical_stack_streaming(
            prov, _request(len(ids)), tile_size=16, fixed=res.gauge
        )
        return float(np.nanmean(out.science))

    full_cell = run(cell_ids)              # frames [0,1,2] of cluster B
    sub_cell = run([cell_ids[0], cell_ids[2]])  # shares the Cell reference (0)
    assert abs(full_cell - sub_cell) < 0.5, (full_cell, sub_cell)


def test_per_cell_gauge_equals_frozen_per_cell_stack():
    """The per-Cell gauge is the FROZEN estimator reused verbatim: applying it is
    bit-equal to the executor's own (fixed=None) per-Cell phase-1 over the same
    provider."""
    frames, masks, _left, _right, n_a, n_b = _cluster_corpus()
    cell_ids = list(range(n_a, n_a + n_b))
    sub_frames = [frames[i] for i in cell_ids]
    sub_masks = [masks[i] for i in cell_ids]
    prov = InMemoryCanonicalProvider(sub_frames, sub_masks)
    req = _request(len(cell_ids))

    res = zphot.compute_cell_gauge(prov, req)
    fixed = run_canonical_stack_streaming(prov, req, tile_size=16, fixed=res.gauge)
    per_cell = run_canonical_stack_streaming(prov, req, tile_size=16)  # fixed=None

    np.testing.assert_array_equal(fixed.science, per_cell.science)
    np.testing.assert_array_equal(fixed.support_w1, per_cell.support_w1)
    np.testing.assert_array_equal(fixed.n_eff_support, per_cell.n_eff_support)
    assert fixed.provenance["reference"]["index"] == per_cell.provenance["reference"]["index"]


# ---------------------------------------------------------------------------
# Dispatcher: global mode unchanged; per_cell mode self-anchors
# ---------------------------------------------------------------------------

def _memmap_provider(tmp_path, frames, masks, ids):
    cache_dir = tmp_path / "cell_cache"
    zfp.write_aligned_cache_from_arrays(str(cache_dir), list(ids), frames, masks)
    return zfp.MemmapCanonicalProvider(str(cache_dir))


def test_resolve_cell_normalization_global_uses_subset(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    frames, masks, _l, _r, n_a, n_b = _cluster_corpus()
    all_ids = [f"f{i}.fits" for i in range(len(frames))]
    gauge = compute_fixed_normalization(
        InMemoryCanonicalProvider(frames, masks), _request(len(frames))
    )
    # The Cell contains ONLY the cluster-B frames (disjoint from the global reference).
    b_ids = all_ids[n_a:]
    prov = _memmap_provider(tmp_path, frames[n_a:], masks[n_a:], b_ids)
    try:
        cfg = zs.MiniTileScienceConfig(normalization="sky_mean")
        with pytest.raises(cs.CanonicalStackValidationError, match="cell gauge has no active frame"):
            zz._resolve_cell_normalization("global", gauge, all_ids, prov, cfg)
    finally:
        prov.close()


def test_resolve_cell_normalization_per_cell(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    frames, masks, _l, _r, n_a, n_b = _cluster_corpus()
    all_ids = [f"f{i}.fits" for i in range(len(frames))]
    prov = _memmap_provider(tmp_path, frames, masks, all_ids)
    try:
        cfg = zs.MiniTileScienceConfig(normalization="sky_mean")
        fixed, status, detail = zz._resolve_cell_normalization(
            "per_cell", None, None, prov, cfg
        )
        assert status == zphot.CELL_GAUGE_STATUS_OK and detail is None
        assert fixed is not None
        assert fixed.coefficients.shape[0] == len(all_ids)
        assert np.allclose(fixed.coefficients[fixed.reference_index][:, 0], 1.0)
        # per_cell path does not need (nor use) the global gauge.
        assert fixed.reference_index == 0
    finally:
        prov.close()


def test_resolve_cell_normalization_per_cell_zero_support(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    h, w = 16, 16
    full = _full_support(h, w)
    nan = np.full((h, w), np.nan, dtype=np.float32)
    ids = ["a.fits", "b.fits"]
    prov = _memmap_provider(tmp_path, [nan, nan.copy()], [full, full], ids)
    try:
        cfg = zs.MiniTileScienceConfig(normalization="sky_mean")
        fixed, status, detail = zz._resolve_cell_normalization(
            "per_cell", None, None, prov, cfg
        )
        assert fixed is None
        assert status == zphot.CELL_GAUGE_STATUS_EMPTY
        assert detail == "zero_valid_support"
    finally:
        prov.close()


# ---------------------------------------------------------------------------
# Provenance + manifest (mode-aware, default unchanged)
# ---------------------------------------------------------------------------

def test_reference_provenance_per_cell_is_local_anchor():
    from zemosaic import zemosaic_zegrid_mode as zz

    prov = zz._reference_provenance("R.fits", ["a.fits", "b.fits"], "b.fits", per_cell=True)
    assert prov["reference_frame_role"] == "cell_local_photometric_anchor"
    assert prov["bookkeeping_reference_frame_id"] is None
    # default (global) behaviour unchanged.
    prov_g = zz._reference_provenance("R.fits", ["a.fits", "b.fits"], "b.fits")
    assert prov_g["reference_frame_role"] == "bookkeeping_placeholder"


def _synthetic_tan_wcs(shape=(10, 10)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


class _Assembled:
    science = np.zeros((10, 10, 3), dtype=np.float32)
    stack_depth = np.ones((10, 10), dtype=np.int32)
    complete_cells = ["r0000c0000"]
    incomplete_cells = []
    hole_pixels = 0
    coverage_pixels = 100


def _manifest_for_mode(tmp_path, mode):
    import json

    from dataclasses import replace

    from zemosaic import zemosaic_zegrid_mode as zz

    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    descs = [zg.FrameDescriptor(
        frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(10, 10),
        wcs_header=_synthetic_tan_wcs().to_header().tostring(), header_sha256="", instrument="",
    )]
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    cfg = replace(ExecutorConfig().science_config(), cell_gauge_mode=mode)
    _, _, manifest_path = zz._write_outputs(
        _Assembled(), w, 1, 1, tmp_path, descs, {}, [],
        layout, cfg, 0, {}, None,
        global_reference_frame_id="a.fits",
    )
    return json.loads(manifest_path.read_text())


def test_manifest_mode_global_default(tmp_path):
    m = _manifest_for_mode(tmp_path, "global")
    assert m["photometric_gauge"]["mode"] == "global"
    assert m["photometric_gauge"]["global_reference_frame_id"] == "a.fits"


def test_manifest_mode_per_cell(tmp_path):
    m = _manifest_for_mode(tmp_path, "per_cell")
    assert m["photometric_gauge"]["mode"] == "per_cell"
    assert "self-anchor" in m["photometric_gauge"]["note"]
    assert "cell_local_photometric_anchor" in m["photometric_gauge"]["provenance_honesty"]


# ---------------------------------------------------------------------------
# Gated M106 — real-data parity of the per-Cell gauge with the global estimator
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_m106_per_cell_gauge_parity_with_global(m106_corpus):
    """On real M106 frames, a Cell whose domain IS the global domain must produce
    the SAME reference and coefficients (within the declared reprojection FP
    tolerance) whether computed by the global gauge or the per-Cell estimator —
    i.e. the per-Cell path reuses the frozen estimator, not a copy."""
    descs, _rejected, _canvas = m106_corpus
    ordered = sorted(descs, key=lambda d: d.frame_id)[:8]
    canvas = zg.build_canvas(ordered)
    cfg = ExecutorConfig().science_config()

    from zemosaic import zemosaic_zegrid_mode as zz

    g, ids = zphot.compute_global_gauge(
        ordered, canvas, zz._gauge_decode, cfg, None, workers=1
    )

    arrays, masks = [], []
    for f in ordered:
        rgb, sup = zphot._reproject_full(zz._gauge_decode, f, canvas)
        arrays.append(rgb)
        masks.append(sup)
    prov = InMemoryCanonicalProvider(arrays, masks)
    res = zphot.compute_cell_gauge(prov, _request(len(arrays)))
    assert res.status == zphot.CELL_GAUGE_STATUS_OK
    assert res.reference_index == g.reference_index
    np.testing.assert_allclose(
        res.gauge.coefficients, g.coefficients, rtol=1e-4, atol=1e-4, equal_nan=True
    )
    np.testing.assert_array_equal(res.gauge.norm_active, g.norm_active)
