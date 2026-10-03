"""Characterization witness: low-N / all-invalid / zero-weight stacking contracts.

Pins the *current* deterministic behavior of the three stacking implementations
before any R2 extraction:

* ``zemosaic.grid_mode._stack_weighted_patches`` (Grid CPU)
* ``zemosaic.zemosaic_stack_core.stack_core`` (shared CPU/GPU core, CPU backend)
* Classic CPU wrappers ``stack_kappa_sigma_clip`` / ``stack_winsorized_sigma_clip``
  (``zemosaic.zemosaic_align_stack``) for documented N<3 behavior.

This is a **characterization**, not an endorsement and not a parity expectation.
The discovered divergences (Grid CPU returns zeros where ``stack_core`` returns
NaN at zero ``weight_sum``; median combine treats ``weight<=0`` as invalid in
Grid CPU but ignores weights entirely in ``stack_core``) are pinned in separate,
discriminating assertions so the difference is explicit rather than normalized
away by a shared helper.

Design notes
------------
* All cases use tiny deterministic ``float32`` HWC arrays (2x2x1) and the CPU
  backend only. No GPU, media, network, user config, or large allocation.
* ``GridModeConfig`` is built directly and normalized/rejection are disabled
  (``stack_norm_method="none"``, ``stack_reject_algo="none"``) unless a case
  specifically exercises the combine method, so the combine contract is what is
  observed, not the normalization/rejection layers.
* NaN semantics are asserted with explicit ``np.isnan`` / ``np.isfinite`` masks;
  ``np.testing.assert_equal`` is avoided for NaN cases (``equal_nan`` would
  obscure whether the value is NaN vs a real float).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from zemosaic import grid_mode
from zemosaic import zemosaic_align_stack
from zemosaic import zemosaic_stack_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(
    *,
    norm: str = "none",
    reject: str = "none",
    combine: str = "mean",
) -> grid_mode.GridModeConfig:
    cfg = grid_mode.GridModeConfig()
    cfg.stack_norm_method = norm
    cfg.stack_reject_algo = reject
    cfg.stack_final_combine = combine
    return cfg


def _frame(*values: float) -> np.ndarray:
    """Build a 2x2x1 float32 HWC frame from four scalar pixel values."""
    return np.array(
        [[[values[0]], [values[1]]], [[values[2]], [values[3]]]],
        dtype=np.float32,
    )


def _ones_w() -> np.ndarray:
    return np.ones((2, 2, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Grid CPU — `_stack_weighted_patches`
# ---------------------------------------------------------------------------


def test_grid_cpu_empty_patches_returns_none():
    # Current return contract for empty input is a bare None (not a tuple).
    assert grid_mode._stack_weighted_patches([], [], _cfg()) is None


def test_grid_cpu_n1_valid_positive_weights_contract():
    frame = _frame(10.0, 20.0, 30.0, 40.0)
    weight = np.full((2, 2, 1), 2.0, dtype=np.float32)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [frame], [weight], _cfg(), return_weight_sum=True
    )

    # shape/dtype contract: HWC float32 preserved for both outputs.
    assert result.shape == (2, 2, 1)
    assert result.dtype == np.float32
    assert weight_sum.shape == (2, 2, 1)
    assert weight_sum.dtype == np.float32
    # value contract: with normalization/rejection disabled and a single frame,
    # the data is returned unchanged.
    np.testing.assert_array_equal(result, frame)
    # weight-sum contract: per-pixel weight is summed unchanged (single frame).
    np.testing.assert_array_equal(weight_sum, np.full((2, 2, 1), 2.0, dtype=np.float32))


def test_grid_cpu_all_zero_weights_returns_zeros():
    frame = _frame(5.0, 5.0, 5.0, 5.0)
    zero_w = np.zeros((2, 2, 1), dtype=np.float32)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [frame], [zero_w], _cfg(), return_weight_sum=True
    )

    assert result.dtype == np.float32
    # Grid CPU: all-invalid (zero weight) -> zero tiles, NOT NaN.
    np.testing.assert_array_equal(result, np.zeros((2, 2, 1), dtype=np.float32))
    np.testing.assert_array_equal(weight_sum, np.zeros((2, 2, 1), dtype=np.float32))


def test_grid_cpu_all_invalid_data_returns_zeros():
    frame = np.full((2, 2, 1), np.nan, dtype=np.float32)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [frame], [_ones_w()], _cfg(), return_weight_sum=True
    )

    # All-NaN data with positive weights still yields zeros (no valid positions).
    np.testing.assert_array_equal(result, np.zeros((2, 2, 1), dtype=np.float32))
    np.testing.assert_array_equal(weight_sum, np.zeros((2, 2, 1), dtype=np.float32))


def test_grid_cpu_mixed_per_pixel_validity_and_all_invalid_pixel():
    # Pixel (0,0) valid in f1 only; (0,1) all-invalid; (1,0) valid f1 only; (1,1) both.
    f1 = _frame(1.0, np.nan, 3.0, 4.0)
    f2 = _frame(np.nan, np.nan, np.nan, 40.0)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f1, f2], [_ones_w(), _ones_w()], _cfg(), return_weight_sum=True
    )

    expected = _frame(1.0, 0.0, 3.0, 22.0)
    expected_ws = _frame(1.0, 0.0, 1.0, 2.0)

    # Grid CPU mean combine: per-pixel valid mean, with all-invalid pixels
    # collapsed to 0.0 (weight_sum clipped to 1e-6), NOT NaN.
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(weight_sum, expected_ws)


def test_grid_cpu_median_ignores_weight_magnitude_but_treats_zero_as_invalid():
    f1 = _frame(1.0, 2.0, 3.0, 4.0)
    f2 = _frame(100.0, 200.0, 300.0, 400.0)
    w_low = np.full((2, 2, 1), 0.1, dtype=np.float32)
    w_high = np.full((2, 2, 1), 100.0, dtype=np.float32)

    result, weight_sum = grid_mode._stack_weighted_patches(
        [f1, f2], [w_low, w_high], _cfg(combine="median"), return_weight_sum=True
    )

    # Magnitude is ignored: median of (1,100)=50.5, (2,200)=101, etc.
    expected = _frame(50.5, 101.0, 151.5, 202.0)
    np.testing.assert_array_equal(result, expected)
    # weight_sum still reflects the summed magnitudes (not used by median).
    np.testing.assert_allclose(weight_sum, np.full((2, 2, 1), 100.1, dtype=np.float32), rtol=1e-6)

    # weight<=0 acts as a validity gate: a zero-weight finite frame is excluded.
    w_zero = np.zeros((2, 2, 1), dtype=np.float32)
    result2, weight_sum2 = grid_mode._stack_weighted_patches(
        [f1, f2], [w_zero, _ones_w()], _cfg(combine="median"), return_weight_sum=True
    )
    # Only f2 (weight=1) survives -> median equals f2's values.
    np.testing.assert_array_equal(result2, f2)
    np.testing.assert_array_equal(weight_sum2, _ones_w())


# ---------------------------------------------------------------------------
# 2. `stack_core` (CPU backend)
# ---------------------------------------------------------------------------


def _core_cfg(**overrides) -> dict:
    cfg = {
        "normalize_method": "none",
        "rejection_algorithm": "none",
        "final_combine_method": "mean",
    }
    cfg.update(overrides)
    return cfg


def test_stack_core_n1_valid_contract():
    frame = _frame(10.0, 20.0, 30.0, 40.0)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [frame], weights=None, stack_config=_core_cfg(), backend="cpu"
    )

    assert stacked.shape == (2, 2, 1)
    assert stacked.dtype == np.float32
    np.testing.assert_array_equal(stacked, frame)
    # No weights -> unit weights -> weight_sum == 1.0 everywhere.
    np.testing.assert_array_equal(weight_sum, _ones_w())
    assert rejected_pct == 0.0


def test_stack_core_all_zero_weights_returns_nan():
    frame = _frame(5.0, 5.0, 5.0, 5.0)
    zero_w = np.zeros((1,), dtype=np.float32)  # scalar per-frame weight (N,)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [frame], weights=zero_w, stack_config=_core_cfg(), backend="cpu"
    )

    # stack_core: zero weight_sum -> NaN (not zeros), unlike Grid CPU.
    assert np.isnan(stacked).all()
    np.testing.assert_array_equal(weight_sum, np.zeros((2, 2, 1), dtype=np.float32))
    assert rejected_pct == 0.0


def test_stack_core_all_invalid_data_returns_nan():
    frame = np.full((2, 2, 1), np.nan, dtype=np.float32)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [frame], weights=None, stack_config=_core_cfg(), backend="cpu"
    )

    assert np.isnan(stacked).all()
    np.testing.assert_array_equal(weight_sum, np.zeros((2, 2, 1), dtype=np.float32))


def test_stack_core_mixed_per_pixel_validity_all_invalid_pixel_is_nan():
    f1 = _frame(1.0, np.nan, 3.0, 4.0)
    f2 = _frame(np.nan, np.nan, np.nan, 40.0)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [f1, f2], weights=None, stack_config=_core_cfg(), backend="cpu"
    )

    # Per-pixel valid mean, but all-invalid pixel -> NaN (not 0.0 as in Grid CPU).
    assert stacked.shape == (2, 2, 1)
    assert stacked.dtype == np.float32
    assert float(stacked[0, 0, 0]) == 1.0
    assert np.isnan(stacked[0, 1, 0])
    assert float(stacked[1, 0, 0]) == 3.0
    assert float(stacked[1, 1, 0]) == 22.0
    expected_ws = _frame(1.0, 0.0, 1.0, 2.0)
    np.testing.assert_array_equal(weight_sum, expected_ws)


def test_stack_core_median_ignores_weights_entirely():
    f1 = _frame(1.0, 2.0, 3.0, 4.0)
    f2 = _frame(100.0, 200.0, 300.0, 400.0)
    # A zero-weight frame with finite data STILL enters the median in stack_core.
    w = np.array([0.0, 1.0], dtype=np.float32)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [f1, f2], weights=w, stack_config=_core_cfg(final_combine_method="median"), backend="cpu"
    )

    # Median of (1,100)=50.5 etc. — the weight=0 frame is NOT excluded.
    expected = _frame(50.5, 101.0, 151.5, 202.0)
    np.testing.assert_array_equal(stacked, expected)
    # weight_sum sums only finite-masked weights; finite data in f1 keeps its
    # weight entry at 0, so weight_sum reflects only the weight=1 frame.
    np.testing.assert_array_equal(weight_sum, _ones_w())


def test_stack_core_median_all_invalid_returns_nan():
    frame = np.full((2, 2, 1), np.nan, dtype=np.float32)

    with pytest.warns(RuntimeWarning):
        stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
            [frame], weights=None, stack_config=_core_cfg(final_combine_method="median"), backend="cpu"
        )

    assert np.isnan(stacked).all()


def test_stack_core_empty_images_raises():
    with pytest.raises(ValueError):
        zemosaic_stack_core.stack_core([], backend="cpu")


def test_stack_core_2d_input_preserves_2d_output():
    frame = np.ones((2, 2), dtype=np.float32)

    stacked, _, _ = zemosaic_stack_core.stack_core(
        [frame], weights=None, stack_config=_core_cfg(), backend="cpu"
    )

    assert stacked.shape == (2, 2)
    assert stacked.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Pinned divergence: Grid CPU zeros vs stack_core NaN at zero weight_sum
# ---------------------------------------------------------------------------


def test_divergence_grid_cpu_zeros_vs_stack_core_nan():
    """Same all-invalid input produces zeros in Grid CPU and NaN in stack_core.

    This is the SCI-03 low-N/all-invalid divergence, pinned explicitly in two
    separate assertions (no shared normalizing helper). It is a
    characterization, not a parity expectation and not a fix.
    """
    frame = np.full((2, 2, 1), np.nan, dtype=np.float32)
    zero_w = np.zeros((2, 2, 1), dtype=np.float32)

    grid_result, grid_ws = grid_mode._stack_weighted_patches(
        [frame], [zero_w], _cfg(), return_weight_sum=True
    )
    core_result, _, core_ws = zemosaic_stack_core.stack_core(
        [frame], weights=np.zeros((1,), dtype=np.float32),
        stack_config=_core_cfg(), backend="cpu",
    )

    # Grid CPU: zeros.
    assert np.isfinite(grid_result).all()
    np.testing.assert_array_equal(grid_result, np.zeros((2, 2, 1), dtype=np.float32))
    # stack_core: NaN.
    assert np.isnan(core_result).all()
    # Both report zero weight_sum.
    np.testing.assert_array_equal(grid_ws, np.zeros((2, 2, 1), dtype=np.float32))
    np.testing.assert_array_equal(core_ws, np.zeros((2, 2, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# 4. Classic low-N — kappa / winsorized wrappers (documented N<3 behavior)
# ---------------------------------------------------------------------------


def test_classic_kappa_sigma_clip_n1_returns_input():
    frame = _frame(1.0, 2.0, 3.0, 4.0)

    stacked, rejected = zemosaic_align_stack.stack_kappa_sigma_clip(
        [frame], weight_method="none", zconfig=None
    )

    assert stacked.dtype == np.float32
    np.testing.assert_array_equal(stacked, frame)
    assert rejected == 0.0


def test_classic_kappa_sigma_clip_n2_mean():
    f1 = _frame(1.0, 2.0, 3.0, 4.0)
    f2 = _frame(3.0, 2.0, 5.0, 4.0)

    stacked, rejected = zemosaic_align_stack.stack_kappa_sigma_clip(
        [f1, f2], weight_method="none", zconfig=None
    )

    # Mean of per-pixel values (sigma clip keeps all at N=2): (1,3)->2, etc.
    expected = _frame(2.0, 2.0, 4.0, 4.0)
    np.testing.assert_array_equal(stacked, expected)
    assert rejected == 0.0


def test_classic_winsorized_n1_forces_cpu_and_returns_input(caplog):
    frame = _frame(1.0, 2.0, 3.0, 4.0)

    with caplog.at_level(logging.WARNING, logger="zemosaic.align_stack"):
        result = zemosaic_align_stack.stack_winsorized_sigma_clip(
            [frame], weight_method="none", zconfig=None
        )

    # Documented N<3 behavior: a warning is emitted and CPU is forced; the
    # stack still returns the (unchanged) single frame plus a 0.0 rejected rate.
    assert any("needs >=3 images" in r.message for r in caplog.records)
    assert isinstance(result, tuple) and len(result) == 2
    stacked, rejected = result
    assert stacked.dtype == np.float32
    np.testing.assert_array_equal(stacked, frame)
    assert rejected == 0.0
