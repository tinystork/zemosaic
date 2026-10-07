"""ZM-ZEGRID-R11 targeted tests — global photometric gauge (synthetic, fast).

Covers the core correctness of the global gauge without needing real data:

* the global reference selection is deterministic (greatest valid support, tie
  lowest index);
* the global gauge REUSES the frozen functions: applying the fixed gauge to a
  provider whose domain IS the full footprint is BIT-EQUAL to the per-cell
  computation over that same provider (the "cell == footprint" equivalence);
* ``subset_fixed_normalization`` remaps the reference index + reorders
  coefficients for a Cell's frame subset, and RE-ANCHORS when the global
  reference is not a contributor;
* in-memory vs streaming parity is preserved with a fixed gauge (disk round-trip
  through the memmap provider is bit-exact);
* the R10 advisories are folded in (L1 SIP margin guard, I1 reconciliation).
"""

from __future__ import annotations

import numpy as np
import pytest

import zemosaic.core.canonical_stacking as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_streaming import (
    FixedNormalization,
    InMemoryCanonicalProvider,
    compute_fixed_normalization,
    run_canonical_stack_streaming,
    subset_fixed_normalization,
)
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import file_provider as zfp


def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _corpus(n=7, h=48, w=48, seed=3):
    """Mono frames with offset variation + partial support (mirrors the SCI-05 corpus)."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(5):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        base += 30.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))
    offsets = np.linspace(-5.0, 5.0, n)
    frames = [ (base + offsets[i] + rng.normal(0.0, 1.0, (h, w))).astype(np.float32) for i in range(n) ]
    masks = [_full_support(h, w)] * n
    masks[3][:, w // 2:] = False  # partial support -> a frame with less valid area
    frames[5] = np.full((h, w), np.nan, dtype=np.float32)  # all-NaN -> excluded
    return frames, masks


def _request(arrays, masks, **kw):
    base = dict(normalization="sky_mean", weighting="noise_variance",
                rejection="kappa_sigma", combine="mean", taper="footprint",
                taper_px=8.0)
    base.update(kw)
    return CanonicalStackRequest(images=arrays, geometric_support=masks, **base)


# ---------------------------------------------------------------------------
# Global reference selection (deterministic)
# ---------------------------------------------------------------------------

def test_global_reference_selection_deterministic():
    arrays, masks = _corpus()
    req = _request(arrays, masks)
    prov = InMemoryCanonicalProvider(arrays, masks)
    gauge = compute_fixed_normalization(prov, req)

    # Frame 3 has its right half masked -> fewer valid pixels; frame 5 is all-NaN.
    # The reference is the frame with the greatest valid support (frame 3 loses the
    # right half, so it is NOT the reference; ties resolved to lowest index).
    assert 0 <= gauge.reference_index < len(arrays)
    # Determinism: recompute gives the same reference index.
    gauge2 = compute_fixed_normalization(InMemoryCanonicalProvider(arrays, masks), req)
    assert gauge2.reference_index == gauge.reference_index
    # The reference frame carries the identity coefficient (a=1, b=0).
    ref_c = gauge.coefficients[gauge.reference_index]
    assert np.all(np.isclose(ref_c[:, 0], 1.0))
    assert np.all(np.isclose(ref_c[:, 1], 0.0))


def test_global_reference_greatest_support_tiebreak():
    # Three noisy frames, all full support except frame 2 masked to half -> frame 2 loses.
    h, w = 24, 24
    rng = np.random.default_rng(0)
    a = (100.0 + rng.normal(0.0, 1.0, (h, w))).astype(np.float32)
    arrays = [a.copy(), a.copy(), a.copy()]
    masks = [_full_support(h, w)] * 3
    masks[2][:, w // 2:] = False
    req = _request(arrays, masks)
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(arrays, masks), req)
    # Frames 0 and 1 tie on valid support -> lowest index (0) wins.
    assert gauge.reference_index == 0


# ---------------------------------------------------------------------------
# Fixed gauge == per-cell computation when the cell IS the full footprint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [None, 16, (24, 31)])
def test_fixed_equals_per_cell_over_full_footprint(tile):
    arrays, masks = _corpus()
    req = _request(arrays, masks)
    prov = InMemoryCanonicalProvider(arrays, masks)

    gauge = compute_fixed_normalization(prov, req)
    ref = run_canonical_stack_streaming(prov, req, tile_size=tile)          # per-cell (full footprint)
    st = run_canonical_stack_streaming(prov, req, tile_size=tile, fixed=gauge)  # fixed gauge

    np.testing.assert_array_equal(ref.science, st.science)
    np.testing.assert_array_equal(ref.valid_mask, st.valid_mask)
    np.testing.assert_array_equal(ref.support_w1, st.support_w1)
    np.testing.assert_array_equal(ref.support_w2, st.support_w2)
    np.testing.assert_array_equal(ref.n_eff_support, st.n_eff_support)
    np.testing.assert_array_equal(ref.surviving_sample_count, st.surviving_sample_count)
    assert ref.provenance["reference"]["index"] == st.provenance["reference"]["index"]
    assert ref.provenance["excluded_frames"] == st.provenance["excluded_frames"]


# ---------------------------------------------------------------------------
# subset_fixed_normalization: remap + re-anchor
# ---------------------------------------------------------------------------

def test_subset_remaps_reference_and_coefficients():
    arrays, masks = _corpus()
    req = _request(arrays, masks)
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(arrays, masks), req)

    # Cell contains only frames [1, 3, 0] (reordered) — reference (0) is present.
    cell_ids = [1, 3, 0]
    sub = subset_fixed_normalization(gauge, list(range(len(arrays))), cell_ids)
    assert sub.reference_index == 2  # global frame 0 is now at cell index 2
    np.testing.assert_array_equal(sub.coefficients[2], gauge.coefficients[0])
    np.testing.assert_array_equal(sub.coefficients[0], gauge.coefficients[1])
    np.testing.assert_array_equal(sub.coefficients[1], gauge.coefficients[3])


def test_subset_reanchors_when_reference_absent():
    arrays, masks = _corpus()
    req = _request(arrays, masks)
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(arrays, masks), req)
    ref = gauge.reference_index

    # A cell that does NOT contain the reference frame.
    others = [i for i in range(len(arrays)) if i != ref][:2]
    sub = subset_fixed_normalization(gauge, list(range(len(arrays))), others)

    # The re-anchor: each coefficient is re-expressed relative to the anchor so the
    # photometric scale is unchanged. For sky_mean (a=1), this is an offset
    # difference: b'_i = b_i - b_anchor, and the anchor gets the identity.
    anchor = sub.reference_index
    assert 0 <= anchor < len(others)
    assert np.all(np.isclose(sub.coefficients[anchor, :, 0], 1.0))
    assert np.all(np.isclose(sub.coefficients[anchor, :, 1], 0.0))
    # The offset difference must reproduce the original (a=1 for sky_mean).
    aL = gauge.coefficients[others[anchor], :, 1]
    for j, gi in enumerate(others):
        expected_b = gauge.coefficients[gi, :, 1] - aL
        assert np.allclose(sub.coefficients[j, :, 1], expected_b)


def test_subset_rejects_unknown_frame():
    arrays, masks = _corpus()
    req = _request(arrays, masks)
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(arrays, masks), req)
    with pytest.raises(cs.CanonicalStackValidationError):
        subset_fixed_normalization(gauge, list(range(len(arrays))), [0, 9999])


# ---------------------------------------------------------------------------
# Parity: in-memory vs streaming (memmap round-trip) with a fixed gauge
# ---------------------------------------------------------------------------

def test_parity_inmem_vs_streaming_fixed(tmp_path):
    arrays, masks = _corpus()
    req = _request(arrays, masks)
    prov = InMemoryCanonicalProvider(arrays, masks)
    gauge = compute_fixed_normalization(prov, req)

    inmem = run_canonical_stack_streaming(prov, req, tile_size=16, fixed=gauge)

    # Round-trip the aligned arrays through the memmap provider (disk).
    cache_dir = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(str(cache_dir), [f"f{i}" for i in range(len(arrays))],
                                        arrays, masks)
    memprov = zfp.MemmapCanonicalProvider(str(cache_dir))
    mem_req = _request([None] * memprov.n_frames, [None] * memprov.n_frames)
    streaming = run_canonical_stack_streaming(memprov, mem_req, tile_size=16, fixed=gauge)

    np.testing.assert_array_equal(inmem.science, streaming.science)
    np.testing.assert_array_equal(inmem.support_w1, streaming.support_w1)
    np.testing.assert_array_equal(inmem.support_w2, streaming.support_w2)
    np.testing.assert_array_equal(inmem.n_eff_support, streaming.n_eff_support)
    assert inmem.provenance["reference"]["index"] == streaming.provenance["reference"]["index"]


# ---------------------------------------------------------------------------
# R10 advisories (L1 + I1)
# ---------------------------------------------------------------------------

def test_r10_l1_sip_margin_advisory_guard_present():
    # The SIP footprint margin is documented as CORPUS-VALIDATED (not a formal
    # bound for arbitrary SIP), with an explicit guard constant.
    assert zg.SIP_MARGIN_IS_CORPUS_VALIDATED is True
    src = open(zg.__file__, encoding="utf-8").read()
    assert "CORPUS-VALIDATED" in src
    assert "not a formal bound" in src.lower() or "NOT a formal bound" in src


def test_r10_i1_reconciliation_in_manifest(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    class _Assembled:
        science = np.zeros((10, 10, 3), dtype=np.float32)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = ["r0000c0000"]
        incomplete_cells = []
        hole_pixels = 0
        coverage_pixels = 100

    # Two frames: one valid descriptor + one rejected (bad WCS).
    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    descs = [
        zg.FrameDescriptor(
            frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(10, 10),
            wcs_header=_synthetic_tan_wcs().to_header().tostring(), header_sha256="", instrument="",
        )
    ]
    rejected = [{"path": "bad.fits", "reason": "WCS has no celestial component"}]
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    _, _, manifest_path = zz._write_outputs(
        _Assembled(), w, 1, 1, tmp_path, descs, {}, [],
        layout, zz.ExecutorConfig().science_config(), 0, {}, None,
        rejected=rejected, sip_mode="keep", frames_loaded=2,
        global_reference_frame_id="a.fits",
    )
    import json
    m = json.loads(manifest_path.read_text())
    recon = m["reconciliation"]
    assert recon["frames_included"] == 1
    assert recon["frames_rejected_wcs"] == 1
    assert recon["frames_loaded"] == 2
    assert recon["frames_loaded"] == recon["frames_included"] + recon["frames_rejected_wcs"]
    assert m["photometric_gauge"]["global_reference_frame_id"] == "a.fits"
    assert m["photometric_gauge"]["mode"] == "global"


def _synthetic_tan_wcs(shape=(10, 10)):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w
