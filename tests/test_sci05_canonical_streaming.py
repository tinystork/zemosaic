"""SCI-05 streaming canonical executor — parity / invariance / memory tests.

Proves :func:`zemosaic.core.canonical_streaming.run_canonical_stack_streaming`
is bit-equivalent to :func:`zemosaic.core.canonical_engine.run_canonical_stack`
for the same inputs via the in-memory provider, across the method matrix, and
that its peak memory scales with tile area (not patch area).

No randomness with global seeds, no network, no filesystem writes, no ZSSS import.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import zemosaic.core.canonical_stacking as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)


# ---------------------------------------------------------------------------
# Corpus builder (deterministic)
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _corpus(n=7, h=48, w=48, seed=3):
    """Mono frames with gain/offset variation, stars, noise, partial masks.

    Frame 1 has a hole, frame 4 has a reduced right side, frame 5 is all-NaN
    (zero valid -> excluded), so the corpus exercises partial support and
    exclusions while every method combination succeeds.
    """
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(6):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        amp = 40.0 * rng.uniform(0.7, 1.3)
        base += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))
    gains = np.linspace(0.96, 1.04, n)
    offsets = np.linspace(-3.0, 3.0, n)
    frames = []
    for i in range(n):
        f = base * gains[i] + offsets[i] + rng.normal(0.0, 2.0, (h, w))
        frames.append(f.astype(np.float32))
    frames[1][4:9, 4:9] = np.nan
    masks = [_full_support(h, w)] * n
    masks[4][:, w // 2 :] = False
    frames[5] = np.full((h, w), np.nan, dtype=np.float32)
    return frames, masks


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------

def _assert_bit_exact(ref, st):
    """Assert every comparable plane/scalar is bit-identical."""
    np.testing.assert_array_equal(ref.science, st.science)
    np.testing.assert_array_equal(ref.estimator_weight_sum, st.estimator_weight_sum)
    np.testing.assert_array_equal(ref.valid_mask, st.valid_mask)
    np.testing.assert_array_equal(ref.surviving_sample_count, st.surviving_sample_count)
    np.testing.assert_array_equal(ref.support_w1, st.support_w1)
    np.testing.assert_array_equal(ref.support_w2, st.support_w2)
    np.testing.assert_array_equal(ref.n_eff_support, st.n_eff_support)
    np.testing.assert_array_equal(ref.rejection_mask, st.rejection_mask)
    assert ref.iterations_used == st.iterations_used
    assert ref.initial_sample_count == st.initial_sample_count
    assert ref.rejected_sample_count == st.rejected_sample_count
    assert ref.rejected_fraction == st.rejected_fraction
    assert ref.low_n_cell_count == st.low_n_cell_count
    assert ref.degenerate_cell_count == st.degenerate_cell_count
    assert ref.rejection_method == st.rejection_method
    assert ref.n_frames == st.n_frames
    assert ref.height == st.height
    assert ref.width == st.width
    assert ref.channels == st.channels
    assert ref.original_mono == st.original_mono
    assert ref.original_ndim == st.original_ndim
    assert ref.original_shape == st.original_shape
    assert ref.provenance == st.provenance
    # explicit tolerance report: exact (bit-identical)
    assert np.count_nonzero(~np.isclose(ref.science, st.science, rtol=0, atol=0, equal_nan=True)) == 0
    assert np.count_nonzero(ref.support_w1 != st.support_w1) == 0


def _check(arrays, masks, tile_size, **kw):
    req = CanonicalStackRequest(images=arrays, geometric_support=masks, **kw)
    ref = run_canonical_stack(req)
    st = run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=tile_size)
    _assert_bit_exact(ref, st)
    return ref, st


# ---------------------------------------------------------------------------
# 1. Method-matrix parity (bit-exact vs run_canonical_stack)
# ---------------------------------------------------------------------------

# Full cross-product over the cheap axes (none/sky_mean normalization, none
# weighting) — 2 x 1 x 3 x 2 x 2 = 24 combos.
_MATRIX_CHEAP = [
    (norm, "none", rej, comb, taper)
    for norm in ("none", "sky_mean")
    for rej in ("none", "kappa_sigma", "winsorized_sigma_clip")
    for comb in ("mean", "median")
    for taper in ("none", "footprint")
]

# Representative coverage of linear_fit normalization and noise_variance /
# noise_fwhm weighting (each method appears at least once on every axis).
_MATRIX_REPRESENTATIVE = [
    ("linear_fit", "none", "none", "mean", "none"),
    ("linear_fit", "none", "kappa_sigma", "mean", "footprint"),
    ("linear_fit", "none", "winsorized_sigma_clip", "median", "footprint"),
    ("none", "noise_variance", "kappa_sigma", "mean", "footprint"),
    ("none", "noise_variance", "winsorized_sigma_clip", "median", "none"),
    ("none", "noise_fwhm", "none", "mean", "footprint"),
    ("none", "noise_fwhm", "kappa_sigma", "median", "footprint"),
    ("linear_fit", "noise_variance", "kappa_sigma", "mean", "footprint"),
    ("linear_fit", "noise_variance", "winsorized_sigma_clip", "median", "footprint"),
    ("sky_mean", "noise_fwhm", "winsorized_sigma_clip", "median", "footprint"),
    ("sky_mean", "noise_variance", "kappa_sigma", "mean", "none"),
    ("linear_fit", "noise_fwhm", "kappa_sigma", "mean", "footprint"),
]


@pytest.mark.parametrize("norm,weight,rej,comb,taper", _MATRIX_CHEAP + _MATRIX_REPRESENTATIVE)
def test_parity_matrix(norm, weight, rej, comb, taper):
    arrays, masks = _corpus()
    kw = dict(normalization=norm, weighting=weight, rejection=rej, combine=comb, taper=taper)
    if taper == "footprint":
        kw["taper_px"] = 8.0
    _check(arrays, masks, tile_size=16, **kw)


def test_parity_rgb():
    rng = np.random.default_rng(2)
    n, h, w = 6, 40, 40
    frames = []
    for i in range(n):
        f = rng.normal(0, 3, (h, w, 3)).astype(np.float32) + np.array([50, 55, 60], np.float32) + 2.0 * i
        f[0:3, :, :] = np.nan
        frames.append(f)
    masks = [_full_support(h, w)] * n
    masks[1][10:15, 10:15] = False
    masks[3][:, 20:] = False
    _check(
        frames, masks, tile_size=11,
        normalization="linear_fit", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean", taper="footprint", taper_px=8.0,
    )
    _check(
        frames, masks, tile_size=None,
        normalization="none", weighting="none", rejection="winsorized_sigma_clip",
        combine="median", taper="none", equalize_rgb=True,
    )


def test_parity_explicit_reference_and_exclusions():
    mono = [np.full((20, 20), v, dtype=np.float32) for v in (10.0, 20.0, 30.0)]
    mono[2] = np.full((20, 20), np.nan, dtype=np.float32)
    masks = [_full_support(20, 20)] * 3
    ref, st = _check(
        mono, masks, tile_size=7,
        normalization="linear_fit", weighting="none", rejection="none",
        combine="mean", taper="none", reference_index=1,
    )
    assert ref.provenance["reference"] == {"mode": "explicit", "index": 1}
    # frame 0 constant -> degenerate_ols; frame 2 all-NaN -> zero_valid_support
    reasons = {e[0]: e[2] for e in ref.provenance["excluded_frames"]}
    assert reasons[2] == "zero_valid_support"
    assert reasons[0] == "degenerate_ols"


# ---------------------------------------------------------------------------
# 2. Tile-size invariance
# ---------------------------------------------------------------------------

_TILE_SIZES = [None, 4, 7, 11, 16, 23, 48, 64, (5, 9), (48, 48)]


@pytest.mark.parametrize("tile", _TILE_SIZES)
def test_tile_invariance(tile):
    arrays, masks = _corpus()
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks,
        normalization="linear_fit", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean", taper="footprint", taper_px=8.0,
    )
    ref = run_canonical_stack(req)
    st = run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=tile)
    _assert_bit_exact(ref, st)


def test_tile_invariance_cross_tiles():
    """Two different non-degenerate tile sizes produce identical streaming results."""
    arrays, masks = _corpus()
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks,
        normalization="sky_mean", weighting="none", rejection="winsorized_sigma_clip",
        combine="median", taper="footprint", taper_px=8.0,
    )
    a = run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=8)
    b = run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=19)
    _assert_bit_exact(a, b)


def test_degenerate_tile_equals_full_patch():
    """A tile >= the patch (degenerate streaming) equals the in-memory engine."""
    arrays, masks = _corpus()
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks,
        normalization="linear_fit", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean", taper="footprint", taper_px=8.0,
    )
    st = run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=100)
    ref = run_canonical_stack(req)
    _assert_bit_exact(ref, st)


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    arrays, masks = _corpus()
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks,
        normalization="linear_fit", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="median", taper="footprint", taper_px=8.0,
    )
    prov = InMemoryCanonicalProvider(arrays, masks)
    a = run_canonical_stack_streaming(prov, req, tile_size=13)
    b = run_canonical_stack_streaming(prov, req, tile_size=13)
    _assert_bit_exact(a, b)


def test_provider_no_input_mutation():
    arrays, masks = _corpus()
    arrays_before = [a.copy() for a in arrays]
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks,
        normalization="none", weighting="none", rejection="kappa_sigma",
        combine="mean", taper="footprint", taper_px=8.0,
    )
    run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=16)
    for a, b in zip(arrays, arrays_before):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 4. Validation / non-goal guards
# ---------------------------------------------------------------------------

class TestValidation:
    def test_gpu_backend_raises(self):
        arrays, masks = _corpus()
        req = CanonicalStackRequest(
            images=arrays, geometric_support=masks,
            normalization="none", weighting="none", rejection="none",
            combine="mean", taper="none", backend="gpu",
        )
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=16)

    def test_explicit_taper_raises(self):
        arrays, masks = _corpus()
        explicit = [np.full((48, 48), 0.5, dtype=np.float32) for _ in arrays]
        req = CanonicalStackRequest(
            images=arrays, geometric_support=masks,
            normalization="none", weighting="none", rejection="none",
            combine="mean", taper=explicit,
        )
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=16)

    def test_bad_request_type(self):
        with pytest.raises(cs.CanonicalStackValidationError):
            run_canonical_stack_streaming("not a request", None)

    def test_unknown_method_tokens(self):
        arrays, masks = _corpus()
        for field in ("normalization", "weighting", "rejection", "combine"):
            kw = dict(normalization="none", weighting="none", rejection="none", combine="mean", taper="none")
            kw[field] = "bogus"
            req = CanonicalStackRequest(images=arrays, geometric_support=masks, **kw)
            with pytest.raises(cs.CanonicalStackValidationError):
                run_canonical_stack_streaming(InMemoryCanonicalProvider(arrays, masks), req, tile_size=16)


# ---------------------------------------------------------------------------
# 5. Memory-bound demonstration (heavy, gated)
# ---------------------------------------------------------------------------

_MEM_BENCH = os.path.join(os.path.dirname(__file__), "..", "tools", "zegrid_r5", "mem_bench.py")


def _available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def _run_bench(mode, tile, n, h, w):
    argv = [
        sys.executable, os.path.abspath(_MEM_BENCH),
        "--mode", mode, "--n", str(n), "--h", str(h), "--w", str(w),
    ]
    if tile is not None:
        argv += ["--tile", str(tile)]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"mem_bench {mode} failed: {out.stderr[:2000]}")
    for token in out.stdout.split():
        if token.startswith("added_peak_kib="):
            return int(token.split("=", 1)[1])
    raise RuntimeError(f"mem_bench {mode}: no added_peak_kib in output: {out.stdout!r}")


@pytest.mark.skipif(
    os.environ.get("ZM_STREAMING_HEAVY") != "1",
    reason="opt-in heavy memory-bound test (set ZM_STREAMING_HEAVY=1 to run)",
)
def test_memory_bound_streaming_bounded_by_tile_area():
    # Skip with the reason recorded if the host cannot afford the in-memory
    # control run (which materialises O(N x patch_area) intermediates).
    n, h, w = 30, 1024, 1024  # ~1.05 Mpx patch
    need_mib = 2000  # in-memory control adds ~1.8 GiB of intermediates
    avail = _available_mib()
    if avail < need_mib:
        pytest.skip(
            f"insufficient memory for in-memory control ({avail} MiB available < {need_mib} MiB required)"
        )

    inmem = _run_bench("inmem", None, n, h, w)
    stream_256 = _run_bench("stream", 256, n, h, w)
    stream_512 = _run_bench("stream", 512, n, h, w)

    # Streaming peak must be a small fraction of the in-memory peak, and must
    # scale with tile area (512^2 = 4 x 256^2), not patch area.
    assert stream_256 < inmem / 3, (stream_256, inmem)
    assert stream_512 < inmem / 2, (stream_512, inmem)
    assert stream_256 <= stream_512 + (stream_512 // 4)  # roughly monotone in tile area
