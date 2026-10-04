"""Characterization witness: SCI-03 — Grid mask/weight/all-invalid/alias/winsor_limits contracts.

Pins the *current, deterministic* contracts of the three Grid stacking routes for
masks/weights, zero/negative weights, finite/non-finite samples, per-pixel/all-invalid
behavior, relevant aliases, and ``winsor_limits`` propagation — **without correcting,
harmonizing, or claiming parity**:

1. **Grid CPU** (``grid_mode._stack_weighted_patches``) masks data with ``weight > 0``
   *before* rejection, so any ``weight <= 0`` (zero or negative) finite sample arrives at
   rejection already NaN. After rejection, only finite surviving samples keep their weight
   in ``weight_sum``; mean uses positive weight *magnitude*; median ignores magnitude but
   honors the pre-rejection ``weight > 0`` validity gate. All-invalid (per-pixel or
   whole-tile) → zero tiles, zero ``weight_sum``.
2. **Grid GPU-legacy** (``_stack_weighted_patches_gpu`` with ``stack_core is None``) mirrors
   the CPU mask/rejection ordering through CuPy (NumPy round-trips for rejection) and passes
   ``config.winsor_limits`` to the real WSC helper.
3. **Grid GPU-core** (``_stack_weighted_patches_gpu`` with ``stack_core`` available) passes
   upstream-normalized finite images and **raw/unmasked** ``cp_weights`` into ``stack_core``
   with ``normalize_method='none'`` and **no** ``winsor_limits``; it does **not** pre-mask
   nonpositive weights. ``stack_core`` therefore sees zero/negative-weight finite samples,
   returns NaN at zero ``weight_sum`` (mean), includes finite zero-weight frames in median,
   and does not clamp negative weights — three quantified divergences from Grid CPU/legacy.

This is a **characterization**, not an endorsement and not a parity expectation. Physical GPU
numerical execution is **NOT_RUN**; the GPU paths are exercised only through a hermetic
NumPy-backed fake-CuPy routing seam plus (for the core route) a recording adapter that hands
the captured call to the *real* ``stack_core`` on the CPU backend. That is a
**CORE-CPU-ADAPTER** semantic comparison reached *through* the routing seam, not a GPU
arithmetic claim.

Design notes
------------
* All corpora are tiny deterministic ``float32`` HWC arrays built from explicit values; no
  random/unseeded data, no sleeps, no network, no GPU, no media, no profile writes.
* NaN/finite semantics are asserted with explicit masks (``np.isnan``/``np.isfinite``);
  ``np.testing.assert_equal``/``equal_nan`` are avoided so a real value is never confused with
  a NaN and the route contract is not normalized away by a shared helper.
* Env is isolated via ``monkeypatch`` (``delenv``/``setenv``) and never mutates real user
  config/profile. ``ZEMOSAIC_WSC_IMPL`` is cleared where the default PixInsight path matters.
* Aliases and ``winsor_limits`` propagation are proven with ``*args/**kwargs`` spies and
  recording adapters (call contracts / call counts), not source-line or source-substring
  assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic import grid_mode
from zemosaic import zemosaic_stack_core


# ---------------------------------------------------------------------------
# Hermetic helpers
# ---------------------------------------------------------------------------

_WSC_ENV = "ZEMOSAIC_WSC_IMPL"


def _frame(*values: float) -> np.ndarray:
    """Build a 2x2x1 float32 HWC frame from four scalar pixel values (row-major)."""
    return np.array(
        [[[values[0]], [values[1]]], [[values[2]], [values[3]]]],
        dtype=np.float32,
    )


def _ones_w() -> np.ndarray:
    return np.ones((2, 2, 1), dtype=np.float32)


def _zeros_w() -> np.ndarray:
    return np.zeros((2, 2, 1), dtype=np.float32)


def _cfg(
    *,
    norm: str = "none",
    reject: str = "none",
    combine: str = "mean",
    weight: str = "noise_variance",
) -> grid_mode.GridModeConfig:
    cfg = grid_mode.GridModeConfig()
    cfg.stack_norm_method = norm
    cfg.stack_reject_algo = reject
    cfg.stack_final_combine = combine
    cfg.stack_weight_method = weight
    return cfg


def _core_cfg(**overrides) -> dict:
    cfg = {
        "normalize_method": "none",
        "rejection_algorithm": "none",
        "final_combine_method": "mean",
    }
    cfg.update(overrides)
    return cfg


class _FakeCupy:
    """Hermetic NumPy-backed stand-in for CuPy.

    Used **only** to route ``_stack_weighted_patches_gpu`` through its GPU-core and
    GPU-legacy seams without a physical GPU. Every CuPy call is aliased to NumPy so the seam
    can be exercised deterministically; this is not a claim of GPU numerical execution.
    """

    float32 = np.float32
    float64 = np.float64
    nan = np.nan
    newaxis = np.newaxis
    errstate = np.errstate

    @staticmethod
    def asarray(x, dtype=None):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)

    @staticmethod
    def stack(x, axis=0):
        return np.stack(x, axis=axis)

    @staticmethod
    def where(*args):
        return np.where(*args)

    @staticmethod
    def nan_to_num(x, nan=0.0):
        return np.nan_to_num(x, nan=nan)

    @staticmethod
    def isfinite(x):
        return np.isfinite(x)

    @staticmethod
    def nanmedian(x, axis=None):
        return np.nanmedian(x, axis=axis)

    @staticmethod
    def sum(x, axis=0, **kwargs):
        return np.sum(x, axis=axis, **kwargs)

    @staticmethod
    def any(x, axis=None):
        return np.any(x, axis=axis)

    @staticmethod
    def clip(x, a_min, a_max):
        return np.clip(x, a_min, a_max)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=None):
        return np.ones(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# A. Grid CPU — mask/weight ordering and output semantics
# ---------------------------------------------------------------------------


def test_grid_cpu_masks_nonpositive_weight_before_kappa_rejection(monkeypatch):
    """Prove ``weight <= 0`` finite data becomes NaN *before* kappa rejection.

    The spy captures the exact stack handed to ``_reject_outliers_kappa_sigma``. A finite
    sample with ``weight == 0`` or ``weight < 0`` arrives as NaN (weight-invalid), a finite
    sample with ``weight > 0`` arrives unchanged, and an originally-NaN sample with
    ``weight > 0`` stays NaN (data-invalid) — the two invalidation causes are distinguished
    by their origin, not by the final NaN.
    """
    monkeypatch.delenv(_WSC_ENV, raising=False)

    f0 = _frame(10.0, 20.0, 30.0, 40.0)          # all positive weight, finite
    f1 = _frame(50.0, 60.0, 70.0, 80.0)          # mixed weights below
    f2 = _frame(np.nan, 25.0, 35.0, 45.0)        # pixel(0,0) NaN (data-invalid)

    w0 = _ones_w()
    # pixel(0,0) weight=0 ; pixel(0,1) weight=-1 ; pixel(1,0) weight=2 ; pixel(1,1) weight=1
    w1 = _frame(0.0, -1.0, 2.0, 1.0)
    w2 = _ones_w()

    seen: dict = {}
    real = grid_mode._reject_outliers_kappa_sigma

    def spy(data, lo, hi, progress_callback=None):
        seen["input"] = np.array(data, copy=True)
        return real(data, lo, hi, progress_callback)

    monkeypatch.setattr(grid_mode, "_reject_outliers_kappa_sigma", spy)

    cfg = _cfg(reject="kappa_sigma")
    cfg.stack_kappa_low = 2.0
    cfg.stack_kappa_high = 2.0

    grid_mode._stack_weighted_patches([f0, f1, f2], [w0, w1, w2], cfg, return_weight_sum=True)

    assert seen.get("input") is not None
    inp = seen["input"]  # (3, 2, 2, 1)
    assert inp.shape == (3, 2, 2, 1)

    # weight-invalid: finite data + weight<=0 -> NaN at rejection (converted by gate).
    assert np.isnan(inp[1, 0, 0, 0])   # weight == 0, data was finite 50.0
    assert np.isnan(inp[1, 0, 1, 0])   # weight == -1, data was finite 60.0
    # finite positive-weight samples survive unchanged.
    assert inp[1, 1, 0, 0] == 70.0     # weight == 2
    assert inp[1, 1, 1, 0] == 80.0     # weight == 1
    assert inp[0, 0, 0, 0] == 10.0     # weight == 1
    # data-invalid: NaN data + weight>0 stays NaN (not a weight-gate conversion).
    assert np.isnan(inp[2, 0, 0, 0])
    assert inp[2, 0, 1, 0] == 25.0     # finite, weight>0


def test_grid_cpu_mean_uses_positive_weight_magnitude():
    """Weighted mean uses positive weight magnitude (not a unit gate)."""
    f0 = _frame(10.0, 10.0, 10.0, 10.0)
    f1 = _frame(20.0, 20.0, 20.0, 20.0)
    w0 = _ones_w()
    w1 = np.full((2, 2, 1), 4.0, dtype=np.float32)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f0, f1], [w0, w1], _cfg(), return_weight_sum=True
    )

    # (10*1 + 20*4) / (1+4) = 18.0 — magnitude matters (unweighted mean would be 15.0).
    np.testing.assert_array_equal(result, _frame(18.0, 18.0, 18.0, 18.0))
    np.testing.assert_array_equal(weight_sum, np.full((2, 2, 1), 5.0, dtype=np.float32))


def test_grid_cpu_weight_sum_excludes_rejected_samples(monkeypatch):
    """After rejection, ``weight_sum`` counts only finite surviving samples."""
    monkeypatch.delenv(_WSC_ENV, raising=False)

    # Single pixel, 5 frames: tight cluster of 10.0 with one 40.0 outlier.
    # With kappa low/high == 2.0 the 40.0 outlier is rejected -> NaN.
    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(4)] + [_frame(40.0, 40.0, 40.0, 40.0)]
    weights = [_ones_w() for _ in range(5)]

    cfg = _cfg(reject="kappa_sigma")
    cfg.stack_kappa_low = 2.0
    cfg.stack_kappa_high = 2.0

    result, weight_sum = grid_mode._stack_weighted_patches(frames, weights, cfg, return_weight_sum=True)

    # The rejected 40.0 frame drops out of weight_sum: 4 surviving frames.
    np.testing.assert_array_equal(weight_sum, np.full((2, 2, 1), 4.0, dtype=np.float32))
    # Result is the mean of the four surviving 10.0 samples.
    np.testing.assert_array_equal(result, _frame(10.0, 10.0, 10.0, 10.0))


def test_grid_cpu_median_ignores_magnitude_but_excludes_negative_weight():
    """Median ignores positive magnitude but treats ``weight <= 0`` (incl. negative) as invalid."""
    f0 = _frame(10.0, 10.0, 10.0, 10.0)
    f1 = _frame(100.0, 100.0, 100.0, 100.0)
    w_neg = np.full((2, 2, 1), -1.0, dtype=np.float32)
    w_one = _ones_w()

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f0, f1], [w_neg, w_one], _cfg(combine="median"), return_weight_sum=True
    )

    # weight<0 excludes f0 -> median is f1's values.
    np.testing.assert_array_equal(result, f1)
    np.testing.assert_array_equal(weight_sum, _ones_w())


def test_grid_cpu_mixed_all_invalid_pixels_to_zero():
    """Per-pixel all-invalid (weight-invalid *or* data-invalid) collapses to 0.0."""
    # pixel(0,0): f0 weight=0 (weight-invalid), f1 NaN (data-invalid) -> all invalid -> 0
    # pixel(0,1): both valid -> mean
    # pixel(1,0): f0 valid, f1 weight=-1 -> f1 excluded -> f0 value
    # pixel(1,1): both valid -> mean
    f0 = _frame(10.0, 20.0, 30.0, 40.0)
    f1 = _frame(np.nan, 60.0, 70.0, 80.0)
    w0 = _frame(0.0, 1.0, 1.0, 1.0)
    w1 = _frame(1.0, 1.0, -1.0, 1.0)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f0, f1], [w0, w1], _cfg(), return_weight_sum=True
    )

    expected = _frame(0.0, 40.0, 30.0, 60.0)
    expected_ws = _frame(0.0, 2.0, 1.0, 2.0)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(weight_sum, expected_ws)


def test_grid_cpu_whole_tile_all_invalid_zero_weight_sum():
    """Whole-tile all-invalid -> zero tile with zero weight_sum (already pinned in the
    low-N witness; re-proven here on a negative-weight corpus for the SCI-03 route table)."""
    f0 = _frame(1.0, 2.0, 3.0, 4.0)
    w_neg = np.full((2, 2, 1), -2.0, dtype=np.float32)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f0], [w_neg], _cfg(), return_weight_sum=True
    )

    np.testing.assert_array_equal(result, _zeros_w())
    np.testing.assert_array_equal(weight_sum, _zeros_w())


# ---------------------------------------------------------------------------
# B. Grid GPU-legacy route seam (no physical GPU)
# ---------------------------------------------------------------------------


def _enable_legacy_seam(monkeypatch):
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())
    monkeypatch.setattr(grid_mode, "stack_core", None)


def test_gpu_legacy_seam_masks_nonpositive_weight_before_rejection(monkeypatch):
    """GPU-legacy mirrors CPU: ``weight <= 0`` arrives at rejection as NaN (dynamic spy)."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    _enable_legacy_seam(monkeypatch)

    f0 = _frame(10.0, 20.0, 30.0, 40.0)
    f1 = _frame(50.0, 60.0, 70.0, 80.0)
    w0 = _ones_w()
    w1 = _frame(0.0, -1.0, 2.0, 1.0)

    seen: dict = {}
    real = grid_mode._reject_outliers_kappa_sigma

    def spy(data, lo, hi, progress_callback=None):
        seen["input"] = np.array(data, copy=True)
        return real(data, lo, hi, progress_callback)

    monkeypatch.setattr(grid_mode, "_reject_outliers_kappa_sigma", spy)

    cfg = _cfg(reject="kappa_sigma")
    cfg.stack_kappa_low = 2.0
    cfg.stack_kappa_high = 2.0

    out = grid_mode._stack_weighted_patches_gpu(
        [f0, f1], [w0, w1], cfg, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert seen.get("input") is not None
    inp = seen["input"]
    assert np.isnan(inp[1, 0, 0, 0])   # weight == 0
    assert np.isnan(inp[1, 0, 1, 0])   # weight == -1
    assert inp[1, 1, 0, 0] == 70.0     # weight == 2
    assert inp[1, 1, 1, 0] == 80.0     # weight == 1


def test_gpu_legacy_seam_matches_cpu_numerically(monkeypatch):
    _enable_legacy_seam(monkeypatch)
    f0 = _frame(10.0, 20.0, 30.0, 40.0)
    f1 = _frame(50.0, 60.0, 70.0, 80.0)
    w0 = _ones_w()
    w1 = np.full((2, 2, 1), 3.0, dtype=np.float32)

    cpu_result, cpu_ws = grid_mode._stack_weighted_patches(
        [f0, f1], [w0, w1], _cfg(), return_weight_sum=True
    )

    out = grid_mode._stack_weighted_patches_gpu(
        [f0, f1], [w0, w1], _cfg(), return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert isinstance(out, tuple) and len(out) == 2
    result, weight_sum = out
    assert isinstance(result, np.ndarray)
    assert isinstance(weight_sum, np.ndarray)
    assert result.dtype == np.float32
    assert weight_sum.dtype == np.float32

    # Bounded numeric equivalence to CPU only for cases exercised through the seam.
    np.testing.assert_array_equal(result, cpu_result)
    np.testing.assert_array_equal(weight_sum, cpu_ws)


def test_gpu_legacy_seam_all_zero_weights_returns_zero(monkeypatch):
    """GPU-legacy all-zero weights -> zero tile + zero weight_sum (like CPU, unlike core)."""
    _enable_legacy_seam(monkeypatch)
    f0 = _frame(5.0, 5.0, 5.0, 5.0)
    out = grid_mode._stack_weighted_patches_gpu(
        [f0], [_zeros_w()], _cfg(), return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )
    result, weight_sum = out
    np.testing.assert_array_equal(result, _zeros_w())
    np.testing.assert_array_equal(weight_sum, _zeros_w())


def test_gpu_legacy_seam_median_excludes_zero_weight_frame(monkeypatch):
    """GPU-legacy median excludes a finite zero-weight frame (dynamic seam, not CPU similarity).

    Divergence D2 family: legacy (like CPU) gates median on ``weight>0``, so a finite
    zero-weight frame is dropped from the median; ``stack_core`` would include it.
    """
    _enable_legacy_seam(monkeypatch)
    f0 = _frame(10.0, 10.0, 10.0, 10.0)
    f1 = _frame(100.0, 100.0, 100.0, 100.0)
    w0 = _zeros_w()
    w1 = _ones_w()

    out = grid_mode._stack_weighted_patches_gpu(
        [f0, f1], [w0, w1], _cfg(combine="median"), return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert isinstance(out, tuple) and len(out) == 2
    result, weight_sum = out
    assert isinstance(result, np.ndarray) and result.dtype == np.float32
    assert isinstance(weight_sum, np.ndarray) and weight_sum.dtype == np.float32
    # Zero-weight finite frame is excluded -> median == f1 (100.0), not 55.0.
    np.testing.assert_array_equal(result, f1)
    np.testing.assert_array_equal(weight_sum, _ones_w())


# ---------------------------------------------------------------------------
# C. Grid GPU-core route seam + real core adapter (no physical GPU)
# ---------------------------------------------------------------------------


def _enable_core_adapter(monkeypatch):
    """Route the GPU-core seam into the *real* ``stack_core`` on the CPU backend."""
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())

    real_core = zemosaic_stack_core.stack_core

    def adapter(images=None, weights=None, stack_config=None, backend="cpu", progress_callback=None):
        # Semantic comparison through the routing seam: CPU core semantics, NOT GPU numerics.
        return real_core(images=images, weights=weights, stack_config=stack_config, backend="cpu")

    monkeypatch.setattr(grid_mode, "stack_core", adapter)
    return real_core


def _run_gpu_core(monkeypatch, patches, weights, cfg, **kw):
    monkeypatch.delenv(_WSC_ENV, raising=False)
    _enable_core_adapter(monkeypatch)
    return grid_mode._stack_weighted_patches_gpu(
        patches, weights, cfg, return_weight_sum=True, return_ref_median=True,
        raise_on_gpu_failure=True, gpu_failure_context={}, **kw,
    )


def test_gpu_core_seam_records_raw_weights_and_config(monkeypatch):
    """The recording adapter receives raw/unmasked weights and the exact stack config."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())

    records: list[dict] = []

    def recording_stack_core(images=None, weights=None, stack_config=None, backend="cpu", progress_callback=None):
        records.append({
            "images": images,
            "weights": weights,
            "stack_config": stack_config,
            "backend": backend,
        })
        return (np.zeros((2, 2, 1), dtype=np.float32), 0.0, np.ones((2, 2, 1), dtype=np.float32))

    monkeypatch.setattr(grid_mode, "stack_core", recording_stack_core)

    f0 = _frame(10.0, 20.0, 30.0, 40.0)
    f1 = _frame(50.0, 60.0, 70.0, 80.0)
    w0 = _frame(0.0, -1.0, 2.0, 1.0)
    w1 = _ones_w()

    cfg = _cfg(reject="winsorized_sigma_clip", combine="median")
    cfg.winsor_limits = (0.2, 0.1)
    cfg.stack_kappa_low = 2.5
    cfg.stack_kappa_high = 2.5

    grid_mode._stack_weighted_patches_gpu(
        [f0, f1], [w0, w1], cfg, return_weight_sum=True, return_ref_median=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert len(records) == 1
    call = records[0]
    assert call["backend"] == "gpu"
    cfg_out = call["stack_config"]
    assert cfg_out["normalize_method"] == "none"
    assert cfg_out["rejection_algorithm"] == "winsorized_sigma_clip"
    assert cfg_out["final_combine_method"] == "median"
    assert cfg_out["sigma_clip_low"] == 2.5
    assert cfg_out["sigma_clip_high"] == 2.5
    # GPU-core drops winsor_limits entirely (documented divergence).
    assert "winsor_limits" not in cfg_out

    # Raw weights: zero/negative values are passed through unmasked (not NaN).
    raw = np.stack([np.asarray(w) for w in call["weights"]], axis=0)
    assert raw[0, 0, 0, 0] == 0.0
    assert raw[0, 0, 1, 0] == -1.0
    assert raw[0, 1, 0, 0] == 2.0
    # Images are the normalized finite samples (no NaN pre-masking of the weight<=0 frames).
    imgs = np.stack([np.asarray(im) for im in call["images"]], axis=0)
    assert np.isfinite(imgs).all()
    assert imgs[0, 0, 0, 0] == 10.0  # finite despite weight==0 (not pre-masked)


def test_gpu_core_all_zero_weights_core_nan_vs_cpu_zero(monkeypatch):
    """Divergence 1: all-zero weights -> core NaN (mean) vs Grid CPU zero."""
    f0 = _frame(5.0, 5.0, 5.0, 5.0)

    out = _run_gpu_core(monkeypatch, [f0], [_zeros_w()], _cfg())
    core_result, core_ws = out[0], out[1]
    assert np.isnan(np.asarray(core_result)).all()
    np.testing.assert_array_equal(np.asarray(core_ws), _zeros_w())

    cpu_result, cpu_ws = grid_mode._stack_weighted_patches(
        [f0], [_zeros_w()], _cfg(), return_weight_sum=True
    )
    np.testing.assert_array_equal(cpu_result, _zeros_w())
    np.testing.assert_array_equal(cpu_ws, _zeros_w())


def test_gpu_core_mixed_per_pixel_all_invalid_pixel_is_nan(monkeypatch):
    """GPU-core route + adapter: valid pixels combine; only the all-invalid pixel is NaN
    with zero per-pixel weight_sum (route characterization, not the direct-core witness)."""
    f0 = _frame(10.0, 20.0, 30.0, 40.0)
    f1 = _frame(50.0, 60.0, 70.0, 80.0)
    # pixel(0,0) has weight==0 in both frames -> all-invalid for the core -> NaN.
    w0 = _frame(0.0, 1.0, 1.0, 1.0)
    w1 = _frame(0.0, 1.0, 1.0, 1.0)

    out = _run_gpu_core(monkeypatch, [f0, f1], [w0, w1], _cfg())
    core_result, core_ws = np.asarray(out[0]), np.asarray(out[1])

    # Only the all-invalid pixel is NaN; valid pixels combine as the weighted mean.
    assert np.isnan(core_result[0, 0, 0])
    assert float(core_result[0, 1, 0]) == pytest.approx(40.0)   # (20+60)/2
    assert float(core_result[1, 0, 0]) == pytest.approx(50.0)   # (30+70)/2
    assert float(core_result[1, 1, 0]) == pytest.approx(60.0)   # (40+80)/2
    # Zero per-pixel weight_sum at the all-invalid pixel; 2.0 at valid pixels.
    assert core_ws[0, 0, 0] == 0.0
    np.testing.assert_array_equal(core_ws, _frame(0.0, 2.0, 2.0, 2.0))


def test_gpu_core_median_includes_zero_weight_frame(monkeypatch):
    """Divergence 2: core median includes a finite zero-weight frame; CPU excludes it."""
    f0 = _frame(10.0, 10.0, 10.0, 10.0)
    f1 = _frame(100.0, 100.0, 100.0, 100.0)
    w0 = _zeros_w()
    w1 = _ones_w()

    out = _run_gpu_core(monkeypatch, [f0, f1], [w0, w1], _cfg(combine="median"))
    core_result = np.asarray(out[0])
    # Median of (10, 100) == 55.0 — the zero-weight frame is NOT excluded.
    np.testing.assert_array_equal(core_result, _frame(55.0, 55.0, 55.0, 55.0))

    cpu_result, cpu_ws = grid_mode._stack_weighted_patches(
        [f0, f1], [w0, w1], _cfg(combine="median"), return_weight_sum=True
    )
    # CPU excludes the zero-weight frame -> median == 100.0.
    np.testing.assert_array_equal(cpu_result, f1)
    np.testing.assert_array_equal(cpu_ws, _ones_w())


def test_gpu_core_negative_weight_not_clamped(monkeypatch):
    """Divergence 3: core does not clamp negative weights; CPU/legacy drop them.

    Contract-probing only: production ``process_tile`` builds weight maps from footprint
    clipped to ``[0,1]`` times a scalar frame weight, so negative weights are not normal
    production output.
    """
    f0 = _frame(10.0, 10.0, 10.0, 10.0)
    f1 = _frame(100.0, 100.0, 100.0, 100.0)
    w0 = np.full((2, 2, 1), 2.0, dtype=np.float32)
    w1 = np.full((2, 2, 1), -1.0, dtype=np.float32)

    out = _run_gpu_core(monkeypatch, [f0, f1], [w0, w1], _cfg())
    core_result, core_ws = np.asarray(out[0]), np.asarray(out[1])
    # (10*2 + 100*(-1)) / (2 + (-1)) = -80.0 — negative weight is applied, not clamped.
    np.testing.assert_array_equal(core_result, _frame(-80.0, -80.0, -80.0, -80.0))
    np.testing.assert_array_equal(core_ws, _frame(1.0, 1.0, 1.0, 1.0))

    cpu_result, cpu_ws = grid_mode._stack_weighted_patches(
        [f0, f1], [w0, w1], _cfg(), return_weight_sum=True
    )
    # CPU masks weight<0 -> only frame 0 survives -> 10.0, weight_sum 2.0.
    np.testing.assert_array_equal(cpu_result, f0)
    np.testing.assert_array_equal(cpu_ws, np.full((2, 2, 1), 2.0, dtype=np.float32))


def test_gpu_core_zero_weight_sample_visible_to_core_rejection(monkeypatch):
    """Divergence 4 (ordering): a zero-weight finite sample stays *finite* at core rejection,
    while Grid CPU pre-masks it to NaN before rejection (see test in section A)."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())

    seen: dict = {}
    real_reject = zemosaic_stack_core._reject_outliers_kappa_sigma

    def reject_spy(data, lo, hi, progress_callback=None):
        seen["input"] = np.array(data, copy=True)
        return real_reject(data, lo, hi, progress_callback)

    monkeypatch.setattr(zemosaic_stack_core, "_reject_outliers_kappa_sigma", reject_spy)

    real_core = zemosaic_stack_core.stack_core

    def adapter(images=None, weights=None, stack_config=None, backend="cpu", progress_callback=None):
        return real_core(images=images, weights=weights, stack_config=stack_config, backend="cpu")

    monkeypatch.setattr(grid_mode, "stack_core", adapter)

    f0 = _frame(10.0, 20.0, 30.0, 40.0)
    f1 = _frame(1000.0, 1000.0, 1000.0, 1000.0)  # finite, zero-weight
    w0 = _ones_w()
    w1 = _zeros_w()

    cfg = _cfg(reject="kappa_sigma")
    cfg.stack_kappa_low = 2.0
    cfg.stack_kappa_high = 2.0

    grid_mode._stack_weighted_patches_gpu(
        [f0, f1], [w0, w1], cfg, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert seen.get("input") is not None
    inp = seen["input"]
    # The zero-weight finite sample is STILL finite at core rejection (visible to rejection),
    # unlike Grid CPU/legacy which pre-mask weight<=0 to NaN.
    assert np.isfinite(inp[1, 0, 0, 0])
    assert inp[1, 0, 0, 0] == 1000.0


# ---------------------------------------------------------------------------
# D. Aliases
# ---------------------------------------------------------------------------


def test_grid_cpu_alias_kappa_dispatches_kappa_helper(monkeypatch):
    """Grid CPU alias ``kappa`` -> ``_reject_outliers_kappa_sigma`` (spy call count)."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    seen: dict = {"kappa": 0, "winsor": 0}
    real_k = grid_mode._reject_outliers_kappa_sigma
    real_w = grid_mode._reject_outliers_winsorized_sigma_clip

    monkeypatch.setattr(
        grid_mode, "_reject_outliers_kappa_sigma",
        lambda *a, **k: (seen.__setitem__("kappa", seen["kappa"] + 1), real_k(*a, **k))[1],
    )
    monkeypatch.setattr(
        grid_mode, "_reject_outliers_winsorized_sigma_clip",
        lambda *a, **k: (seen.__setitem__("winsor", seen["winsor"] + 1), real_w(*a, **k))[1],
    )

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(4)] + [_frame(40.0, 40.0, 40.0, 40.0)]
    weights = [_ones_w() for _ in range(5)]
    cfg = _cfg(reject="kappa")
    cfg.stack_kappa_low = 2.0
    cfg.stack_kappa_high = 2.0

    grid_mode._stack_weighted_patches(frames, weights, cfg, return_weight_sum=True)

    assert seen["kappa"] == 1
    assert seen["winsor"] == 0


def test_grid_cpu_alias_winsor_dispatches_wsc_helper(monkeypatch):
    """Grid CPU alias ``winsor`` -> ``_reject_outliers_winsorized_sigma_clip``."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    seen: dict = {"kappa": 0, "winsor": 0}
    real_k = grid_mode._reject_outliers_kappa_sigma
    real_w = grid_mode._reject_outliers_winsorized_sigma_clip

    monkeypatch.setattr(
        grid_mode, "_reject_outliers_kappa_sigma",
        lambda *a, **k: (seen.__setitem__("kappa", seen["kappa"] + 1), real_k(*a, **k))[1],
    )
    monkeypatch.setattr(
        grid_mode, "_reject_outliers_winsorized_sigma_clip",
        lambda *a, **k: (seen.__setitem__("winsor", seen["winsor"] + 1), real_w(*a, **k))[1],
    )

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(5)]
    weights = [_ones_w() for _ in range(5)]
    cfg = _cfg(reject="winsor")
    cfg.winsor_limits = (0.2, 0.1)

    grid_mode._stack_weighted_patches(frames, weights, cfg, return_weight_sum=True)

    assert seen["winsor"] == 1
    assert seen["kappa"] == 0


def test_gpu_legacy_alias_kappa_dispatches_kappa_helper(monkeypatch):
    """GPU-legacy alias ``kappa`` -> ``_reject_outliers_kappa_sigma`` (spy call count)."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    _enable_legacy_seam(monkeypatch)
    seen: dict = {"kappa": 0, "winsor": 0}
    real_k = grid_mode._reject_outliers_kappa_sigma
    real_w = grid_mode._reject_outliers_winsorized_sigma_clip

    monkeypatch.setattr(
        grid_mode, "_reject_outliers_kappa_sigma",
        lambda *a, **k: (seen.__setitem__("kappa", seen["kappa"] + 1), real_k(*a, **k))[1],
    )
    monkeypatch.setattr(
        grid_mode, "_reject_outliers_winsorized_sigma_clip",
        lambda *a, **k: (seen.__setitem__("winsor", seen["winsor"] + 1), real_w(*a, **k))[1],
    )

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(4)] + [_frame(40.0, 40.0, 40.0, 40.0)]
    weights = [_ones_w() for _ in range(5)]
    cfg = _cfg(reject="kappa")
    cfg.stack_kappa_low = 2.0
    cfg.stack_kappa_high = 2.0

    grid_mode._stack_weighted_patches_gpu(
        frames, weights, cfg, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert seen["kappa"] == 1
    assert seen["winsor"] == 0


def test_gpu_legacy_alias_winsor_dispatches_wsc_helper(monkeypatch):
    """GPU-legacy alias ``winsor`` -> ``_reject_outliers_winsorized_sigma_clip``."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    _enable_legacy_seam(monkeypatch)
    seen: dict = {"kappa": 0, "winsor": 0}
    real_k = grid_mode._reject_outliers_kappa_sigma
    real_w = grid_mode._reject_outliers_winsorized_sigma_clip

    monkeypatch.setattr(
        grid_mode, "_reject_outliers_kappa_sigma",
        lambda *a, **k: (seen.__setitem__("kappa", seen["kappa"] + 1), real_k(*a, **k))[1],
    )
    monkeypatch.setattr(
        grid_mode, "_reject_outliers_winsorized_sigma_clip",
        lambda *a, **k: (seen.__setitem__("winsor", seen["winsor"] + 1), real_w(*a, **k))[1],
    )

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(5)]
    weights = [_ones_w() for _ in range(5)]
    cfg = _cfg(reject="winsor")
    cfg.winsor_limits = (0.2, 0.1)

    grid_mode._stack_weighted_patches_gpu(
        frames, weights, cfg, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert seen["winsor"] == 1
    assert seen["kappa"] == 0


def test_gpu_core_forwards_aliases_verbatim_and_core_does_not_dispatch(monkeypatch):
    """GPU-core forwards ``kappa``/``winsor`` verbatim; real core dispatches rejection only
    for the canonical ``kappa_sigma``/``winsorized_sigma_clip`` strings."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())

    forwarded: list[str] = []
    real_core = zemosaic_stack_core.stack_core

    def adapter(images=None, weights=None, stack_config=None, backend="cpu", progress_callback=None):
        forwarded.append(stack_config["rejection_algorithm"])
        return real_core(images=images, weights=weights, stack_config=stack_config, backend="cpu")

    monkeypatch.setattr(grid_mode, "stack_core", adapter)

    # Spy on the core's kappa rejection dispatch.
    kappa_calls: list[int] = []
    real_reject = zemosaic_stack_core._reject_outliers_kappa_sigma

    def reject_spy(data, lo, hi, progress_callback=None):
        kappa_calls.append(1)
        return real_reject(data, lo, hi, progress_callback)

    monkeypatch.setattr(zemosaic_stack_core, "_reject_outliers_kappa_sigma", reject_spy)

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(4)] + [_frame(40.0, 40.0, 40.0, 40.0)]
    weights = [_ones_w() for _ in range(5)]

    cfg_alias = _cfg(reject="kappa")
    cfg_alias.stack_kappa_low = 2.0
    cfg_alias.stack_kappa_high = 2.0
    grid_mode._stack_weighted_patches_gpu(
        frames, weights, cfg_alias, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert forwarded[-1] == "kappa"       # verbatim, not canonicalized
    assert len(kappa_calls) == 0          # core does NOT dispatch for 'kappa'

    cfg_canon = _cfg(reject="kappa_sigma")
    cfg_canon.stack_kappa_low = 2.0
    cfg_canon.stack_kappa_high = 2.0
    grid_mode._stack_weighted_patches_gpu(
        frames, weights, cfg_canon, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert forwarded[-1] == "kappa_sigma"
    assert len(kappa_calls) == 1          # canonical name dispatches


def test_gpu_core_winsor_alias_no_rejection_vs_canonical_rejects(monkeypatch):
    """GPU-core ``winsor`` alias is forwarded verbatim and performs *no* core rejection;
    the canonical ``winsorized_sigma_clip`` executes the simplified core rejection.

    Proven through the real route + core-CPU adapter, recording ``rejected_pct`` and the
    output on a discriminating outlier corpus (four 10.0 frames + one 40.0 outlier at
    sigma 2.0). WSC implementation selection is not involved (the core uses its own
    simplified median/std clip)."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())

    records: list[dict] = []
    real_core = zemosaic_stack_core.stack_core

    def adapter(images=None, weights=None, stack_config=None, backend="cpu", progress_callback=None):
        stacked, rejected_pct, weight_sum = real_core(
            images=images, weights=weights, stack_config=stack_config, backend="cpu"
        )
        records.append({
            "rejection_algorithm": stack_config["rejection_algorithm"],
            "rejected_pct": rejected_pct,
            "result": stacked,
            "weight_sum": weight_sum,
        })
        return stacked, rejected_pct, weight_sum

    monkeypatch.setattr(grid_mode, "stack_core", adapter)

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(4)] + [_frame(40.0, 40.0, 40.0, 40.0)]
    weights = [_ones_w() for _ in range(5)]

    # Alias: forwarded verbatim, no core rejection -> plain mean 16.0, rejected_pct 0.
    cfg_alias = _cfg(reject="winsor")
    cfg_alias.stack_kappa_low = 2.0
    cfg_alias.stack_kappa_high = 2.0
    grid_mode._stack_weighted_patches_gpu(
        frames, weights, cfg_alias, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert records[-1]["rejection_algorithm"] == "winsor"   # verbatim
    assert records[-1]["rejected_pct"] == 0.0               # no core rejection
    np.testing.assert_array_equal(records[-1]["result"], _frame(16.0, 16.0, 16.0, 16.0))

    # Canonical: simplified core rejection runs -> 40.0 outlier rejected, mean 10.0.
    cfg_canon = _cfg(reject="winsorized_sigma_clip")
    cfg_canon.stack_kappa_low = 2.0
    cfg_canon.stack_kappa_high = 2.0
    grid_mode._stack_weighted_patches_gpu(
        frames, weights, cfg_canon, return_weight_sum=True,
        raise_on_gpu_failure=True, gpu_failure_context={},
    )

    assert records[-1]["rejection_algorithm"] == "winsorized_sigma_clip"
    assert records[-1]["rejected_pct"] == pytest.approx(20.0)   # 1 of 5 frames rejected
    np.testing.assert_array_equal(records[-1]["result"], _frame(10.0, 10.0, 10.0, 10.0))


def test_stack_core_raises_on_unsupported_combine():
    """``stack_core`` raises ValueError for an unsupported final-combine method.

    This contrasts with the Grid CPU/legacy branches, which treat any value other than
    ``median`` as mean (no error). The outer GPU wrapper then re-raises (when
    ``raise_on_gpu_failure=True``) or falls back to CPU otherwise.
    """
    f0 = _frame(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        zemosaic_stack_core.stack_core(
            [f0], weights=None,
            stack_config=_core_cfg(final_combine_method="sum"),
            backend="cpu",
        )


def test_grid_cpu_unknown_combine_falls_through_to_mean():
    """Grid CPU treats an unknown final-combine value as mean (no error, not median)."""
    f0 = _frame(10.0, 10.0, 10.0, 10.0)
    f1 = _frame(20.0, 20.0, 20.0, 20.0)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f0, f1], [_ones_w(), _ones_w()], _cfg(combine="sum"), return_weight_sum=True
    )

    np.testing.assert_array_equal(result, _frame(15.0, 15.0, 15.0, 15.0))
    np.testing.assert_array_equal(weight_sum, np.full((2, 2, 1), 2.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# D2. `_compute_frame_weight` aliases
# ---------------------------------------------------------------------------


def _frame_weight(method, patch, exposure=2.0):
    from pathlib import Path
    from zemosaic.grid_mode import FrameInfo

    cfg = grid_mode.GridModeConfig()
    cfg.stack_weight_method = method
    fr = grid_mode.FrameInfo(path=Path("probe.fits"), exposure=exposure)
    footprint = np.ones_like(patch)
    return grid_mode._compute_frame_weight(fr, patch, footprint, cfg)


def _variance_patch() -> np.ndarray:
    return np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)


def test_compute_frame_weight_none_unit_unity_alias_to_exposure():
    """``none``/``unit``/``unity`` all return the exposure factor (no variance term)."""
    patch = _variance_patch()
    for method in ("none", "unit", "unity"):
        assert _frame_weight(method, patch, exposure=2.0) == pytest.approx(2.0)
    # Exposure factor is honored (longer integrations are rewarded).
    assert _frame_weight("unit", patch, exposure=4.0) == pytest.approx(4.0)


def test_compute_frame_weight_noise_fwhm_falls_back_to_variance():
    """``noise_fwhm`` falls back to variance-only (no FWHM estimates in Grid mode)."""
    patch = _variance_patch()
    # std([1,2,3,4]) = 1.118034 ; variance = 1.25 ; inv_var = 0.8 ; weight = 2.0 * 0.8 = 1.6.
    expected = 2.0 * (1.0 / 1.25)
    assert _frame_weight("noise_fwhm", patch, exposure=2.0) == pytest.approx(expected, rel=1e-6)
    # Matches the explicit noise_variance path (same numeric result).
    assert _frame_weight("noise_variance", patch, exposure=2.0) == pytest.approx(expected, rel=1e-6)


def test_compute_frame_weight_unknown_value_falls_through_to_variance():
    """Unknown weight method falls through to variance-only (same as noise_variance)."""
    patch = _variance_patch()
    expected = 2.0 * (1.0 / 1.25)
    assert _frame_weight("bogus_method", patch, exposure=2.0) == pytest.approx(expected, rel=1e-6)


def test_compute_frame_weight_all_nan_returns_exposure():
    """All-NaN patch -> weight is just the exposure factor (no usable variance)."""
    patch = np.full((2, 2, 1), np.nan, dtype=np.float32)
    for method in ("noise_variance", "none", "unit"):
        assert _frame_weight(method, patch, exposure=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# E. `winsor_limits` propagation and behavior
# ---------------------------------------------------------------------------


def test_grid_cpu_passes_winsor_limits_exactly_canonical_and_alias(monkeypatch):
    """Grid CPU passes ``config.winsor_limits`` exactly to the WSC helper for both the
    canonical ``winsorized_sigma_clip`` and the ``winsor`` alias."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    seen: list = []
    real = grid_mode._reject_outliers_winsorized_sigma_clip

    def spy(*args, **kwargs):
        seen.append(args[1])  # winsor_limits_tuple is the 2nd positional arg
        return real(*args, **kwargs)

    monkeypatch.setattr(grid_mode, "_reject_outliers_winsorized_sigma_clip", spy)

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(5)]
    weights = [_ones_w() for _ in range(5)]

    for algo in ("winsorized_sigma_clip", "winsor"):
        cfg = _cfg(reject=algo)
        cfg.winsor_limits = (0.2, 0.1)
        grid_mode._stack_weighted_patches(frames, weights, cfg, return_weight_sum=True)

    assert seen == [(0.2, 0.1), (0.2, 0.1)]


def test_gpu_legacy_passes_winsor_limits_exactly_canonical_and_alias(monkeypatch):
    """GPU-legacy passes ``config.winsor_limits`` exactly to the WSC helper for both routes."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    _enable_legacy_seam(monkeypatch)
    seen: list = []
    real = grid_mode._reject_outliers_winsorized_sigma_clip

    def spy(*args, **kwargs):
        seen.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(grid_mode, "_reject_outliers_winsorized_sigma_clip", spy)

    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(5)]
    weights = [_ones_w() for _ in range(5)]

    for algo in ("winsorized_sigma_clip", "winsor"):
        cfg = _cfg(reject=algo)
        cfg.winsor_limits = (0.2, 0.1)
        grid_mode._stack_weighted_patches_gpu(
            frames, weights, cfg, return_weight_sum=True,
            raise_on_gpu_failure=True, gpu_failure_context={},
        )

    assert seen == [(0.2, 0.1), (0.2, 0.1)]


def test_gpu_core_omits_winsor_limits_and_is_invariant(monkeypatch):
    """GPU-core omits ``winsor_limits`` from the core config, and the simplified core
    winsorized path is invariant to changing Grid ``winsor_limits`` at fixed images/sigmas."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())

    forwarded: list[dict] = []
    real_core = zemosaic_stack_core.stack_core

    def adapter(images=None, weights=None, stack_config=None, backend="cpu", progress_callback=None):
        forwarded.append(dict(stack_config))
        return real_core(images=images, weights=weights, stack_config=stack_config, backend="cpu")

    monkeypatch.setattr(grid_mode, "stack_core", adapter)

    # Discriminating corpus: a mild outlier stack so the simplified winsorized clip runs.
    frames = [_frame(10.0, 10.0, 10.0, 10.0) for _ in range(3)] + [_frame(15.0, 15.0, 15.0, 15.0)]
    weights = [_ones_w() for _ in range(4)]

    results = []
    for limits in ((0.2, 0.1), (0.05, 0.05)):
        cfg = _cfg(reject="winsorized_sigma_clip")
        cfg.winsor_limits = limits
        cfg.stack_kappa_low = 2.5
        cfg.stack_kappa_high = 2.5
        out = grid_mode._stack_weighted_patches_gpu(
            frames, weights, cfg, return_weight_sum=True,
            raise_on_gpu_failure=True, gpu_failure_context={},
        )
        results.append(np.asarray(out[0]))

    # winsor_limits never reaches the core config.
    for cfg_out in forwarded:
        assert "winsor_limits" not in cfg_out

    # Invariant: changing Grid winsor_limits does not change the simplified core output.
    np.testing.assert_array_equal(results[0], results[1])
