"""Direct ``zemosaic.zemosaic_stack_core.stack_core`` shipped-behaviour contracts.

Restored (ZM-ZEGRID-R8 rework-1) from characterization witnesses that were deleted
alongside the removed legacy Grid. These assertions target the STILL-SHIPPED shared
CPU/GPU stacking core (``zemosaic.zemosaic_stack_core.stack_core``) **directly** and
never depend on the removed ``grid_mode`` module. Every Grid-route-specific case was
dropped; only the direct ``stack_core`` semantics are pinned here.

Pinned semantics (as of the current tree):

* ``normalize_method='linear_fit'`` is UNSUPPORTED and raises
  ``ValueError`` ("unsupported_removed_sci05") — the old median-substitution
  placeholder is gone (SCI-05 Gate F1/N4). ``'none'`` / ``'median'`` remain valid.
* Low-N / all-invalid / zero-weight contracts: ``stack_core`` returns **NaN** at
  zero ``weight_sum`` (unlike the removed Grid CPU, which returned zeros); median
  combine ignores weights entirely; 2-D inputs preserve 2-D outputs; empty input
  raises.
* ``rejection_algorithm='winsorized_sigma_clip'`` is a simplified median/std clip
  (placeholder) — distinct from the PixInsight WSC core (see
  ``test_robust_rejection_contracts.py``).
* ``rejection_algorithm='linear_fit_clip'`` is not a valid ``stack_core`` rejection
  branch and falls through to a no-op (identical to ``'none'``).
* Unsupported ``final_combine_method`` raises ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic import zemosaic_stack_core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 4x5x3 affine corpus (linear-fit semantics section)
_H, _W, _C = 4, 5, 3

_SLOPE = np.array([1.3, 0.8, 1.1], dtype=np.float32)
_INTERCEPT = np.array([20.0, -10.0, 30.0], dtype=np.float32)
_SLOPE2 = np.array([0.6, 1.7, 0.9], dtype=np.float32)
_INTERCEPT2 = np.array([150.0, 5.0, -40.0], dtype=np.float32)


def _ref_patch() -> np.ndarray:
    yy, xx = np.mgrid[0:_H, 0:_W]
    ref = np.empty((_H, _W, _C), dtype=np.float32)
    ref[..., 0] = 100.0 + xx * 10 + yy * 5
    ref[..., 1] = 50.0 + xx * 2 + yy * 3
    ref[..., 2] = 200.0 + xx * 4 + yy * 7
    return ref.astype(np.float32)


def _affine(patch: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return (np.asarray(patch, dtype=np.float32) * slope + intercept).astype(np.float32)


def _affine_corpus() -> list[np.ndarray]:
    ref = _ref_patch()
    return [ref, _affine(ref, _SLOPE, _INTERCEPT), _affine(ref, _SLOPE2, _INTERCEPT2)]


def _ones_weights_hwc(n: int) -> list[np.ndarray]:
    return [np.ones((_H, _W, _C), dtype=np.float32) for _ in range(n)]


def _core_config(normalize: str, *, final: str = "mean") -> dict:
    return {
        "normalize_method": normalize,
        "rejection_algorithm": "none",
        "final_combine_method": final,
    }


def _run_core(images, normalize, *, final="mean"):
    return zemosaic_stack_core.stack_core(
        images=images,
        weights=_ones_weights_hwc(len(images)),
        stack_config=_core_config(normalize, final=final),
        backend="cpu",
    )


# 2x2x1 low-N corpus helpers
def _frame(*values: float) -> np.ndarray:
    return np.array(
        [[[values[0]], [values[1]]], [[values[2]], [values[3]]]],
        dtype=np.float32,
    )


def _ones_w() -> np.ndarray:
    return np.ones((2, 2, 1), dtype=np.float32)


def _core_cfg(**overrides) -> dict:
    cfg = {
        "normalize_method": "none",
        "rejection_algorithm": "none",
        "final_combine_method": "mean",
    }
    cfg.update(overrides)
    return cfg


# 1x1x1 single-pixel corpus helpers (winsorized-sigma-clip section)
IMPULSE = [0.0, 0.0, 0.0, 0.0, 100.0]
NONDEGENERATE = [1.0, 1.0, 1.0, 2.0, 100.0]
GRADIENT = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]


def _px_patches(values):
    return [np.array([[[float(v)]]], dtype=np.float32) for v in values]


def _ones_weights_1x1(n):
    return [np.ones((1, 1, 1), dtype=np.float32) for _ in range(n)]


def _core_config_wsc():
    return {
        "normalize_method": "none",
        "rejection_algorithm": "winsorized_sigma_clip",
        "final_combine_method": "mean",
        "sigma_clip_low": 2.5,
        "sigma_clip_high": 2.5,
    }


def _run_core_wsc(values):
    result, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        images=_px_patches(values),
        weights=_ones_weights_1x1(len(values)),
        stack_config=_core_config_wsc(),
        backend="cpu",
    )
    return np.asarray(result), float(rejected_pct), np.asarray(weight_sum)


# ---------------------------------------------------------------------------
# A. Normalization: ``linear_fit`` is unsupported (raises); ``none``/``median`` valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("final", ["mean", "median"])
def test_stack_core_linear_fit_identical_to_median(final):
    """Gate A/SCI-02: the ``linear_fit`` median placeholder is GONE (Gate F1/N4):
    ``linear_fit`` now raises, never silently becomes median."""
    images = _affine_corpus()
    with pytest.raises(ValueError) as ei:
        _run_core(images, "linear_fit", final=final)
    assert "unsupported_removed_sci05" in str(ei.value)

    med_result, med_rej, _ = _run_core(images, "median", final=final)
    assert med_result.shape == (_H, _W, _C)
    assert float(med_rej) == 0.0


def test_stack_core_linear_fit_differs_from_none():
    """Gate F1 (N4): ``linear_fit`` is unsupported (raises); ``none`` still works."""
    images = _affine_corpus()
    with pytest.raises(ValueError) as ei:
        _run_core(images, "linear_fit")
    assert "unsupported_removed_sci05" in str(ei.value)

    none_result, _, _ = _run_core(images, "none")
    assert none_result.shape == (_H, _W, _C)
    assert none_result.dtype == np.float32


def test_stack_core_linear_fit_raises_never_affine_nor_median():
    """Gate F1 (N4): ``stack_core``'s ``linear_fit`` placeholder is gone (raises);
    it never silently becomes a median placeholder nor a genuine affine mapping."""
    images = _affine_corpus()
    with pytest.raises(ValueError) as ei:
        _run_core(images, "linear_fit")
    assert "unsupported_removed_sci05" in str(ei.value)

    none_result, _, _ = _run_core(images, "none")
    assert none_result.shape == (_H, _W, _C)


def test_stack_core_shapes_dtypes_weights():
    """Pin output shape/dtype/weight_sum/rejected_pct on a supported path (``none``),
    and assert ``linear_fit`` raises (Gate F1/N4)."""
    images = _affine_corpus()
    result, rejected_pct, weight_sum = _run_core(images, "none")

    assert result.shape == (_H, _W, _C)
    assert result.dtype == np.float32
    assert weight_sum.shape == (_H, _W, _C)
    assert weight_sum.dtype == np.float32
    assert float(rejected_pct) == 0.0
    np.testing.assert_allclose(weight_sum, np.full((_H, _W, _C), 3.0, dtype=np.float32))

    with pytest.raises(ValueError) as ei:
        _run_core(images, "linear_fit")
    assert "unsupported_removed_sci05" in str(ei.value)


def test_stack_core_is_internal_not_exported():
    """``stack_core`` is an internal module-level symbol, not part of the public package
    surface (``zemosaic.__all__`` only exposes ``__version__``)."""
    import zemosaic

    assert "stack_core" not in getattr(zemosaic, "__all__", ())
    assert not hasattr(zemosaic, "stack_core")
    assert callable(zemosaic_stack_core.stack_core)


def test_stack_core_has_no_linear_fit_clip_rejection_branch():
    """``stack_core`` only implements ``kappa_sigma`` / ``winsorized_sigma_clip``
    rejection; ``linear_fit_clip`` is not a valid rejection algorithm there and falls
    through to no-op (identical output to rejection='none', rejected_pct == 0)."""
    images = _affine_corpus()

    none_result, none_rej, _ = _run_core(images, "none")
    clip_result, clip_rej, _ = zemosaic_stack_core.stack_core(
        images=images,
        weights=_ones_weights_hwc(len(images)),
        stack_config={
            "normalize_method": "none",
            "rejection_algorithm": "linear_fit_clip",
            "final_combine_method": "mean",
        },
        backend="cpu",
    )

    np.testing.assert_array_equal(clip_result, none_result)
    assert float(clip_rej) == 0.0 == float(none_rej)


def test_stack_core_raises_on_unsupported_combine():
    """``stack_core`` raises ValueError for an unsupported final-combine method."""
    f0 = _frame(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        zemosaic_stack_core.stack_core(
            [f0], weights=None,
            stack_config=_core_cfg(final_combine_method="sum"),
            backend="cpu",
        )


# ---------------------------------------------------------------------------
# B. Low-N / all-invalid / zero-weight contracts
# ---------------------------------------------------------------------------


def test_stack_core_n1_valid_contract():
    frame = _frame(10.0, 20.0, 30.0, 40.0)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [frame], weights=None, stack_config=_core_cfg(), backend="cpu"
    )

    assert stacked.shape == (2, 2, 1)
    assert stacked.dtype == np.float32
    np.testing.assert_array_equal(stacked, frame)
    np.testing.assert_array_equal(weight_sum, _ones_w())
    assert rejected_pct == 0.0


def test_stack_core_all_zero_weights_returns_nan():
    frame = _frame(5.0, 5.0, 5.0, 5.0)
    zero_w = np.zeros((1,), dtype=np.float32)  # scalar per-frame weight (N,)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [frame], weights=zero_w, stack_config=_core_cfg(), backend="cpu"
    )

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
    w = np.array([0.0, 1.0], dtype=np.float32)

    stacked, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        [f1, f2], weights=w, stack_config=_core_cfg(final_combine_method="median"), backend="cpu"
    )

    expected = _frame(50.5, 101.0, 151.5, 202.0)
    np.testing.assert_array_equal(stacked, expected)
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
# C. Winsorized-sigma-clip branch (simplified median/std clip)
# ---------------------------------------------------------------------------


def test_stack_core_wsc_impulse_corpus():
    result, rejected_pct, weight_sum = _run_core_wsc(IMPULSE)

    assert result.shape == (1, 1, 1)
    assert result.dtype == np.float32
    # Simplified median/std clip keeps everything (100 is within +-2.5 std of the
    # median 0) and returns the plain mean.
    assert result[0, 0, 0] == pytest.approx(20.0, abs=1e-5)
    assert rejected_pct == pytest.approx(0.0)
    np.testing.assert_allclose(weight_sum, np.array([[[5.0]]], dtype=np.float32))


def test_stack_core_wsc_nondegenerate_corpus():
    result, rejected_pct, weight_sum = _run_core_wsc(NONDEGENERATE)

    assert result[0, 0, 0] == pytest.approx(1.25, abs=1e-5)
    assert rejected_pct == pytest.approx(20.0)
    np.testing.assert_allclose(weight_sum, np.array([[[4.0]]], dtype=np.float32))


def test_stack_core_wsc_gradient_corpus():
    result, rejected_pct, weight_sum = _run_core_wsc(GRADIENT)

    assert result[0, 0, 0] == pytest.approx(3.0, abs=1e-5)
    assert rejected_pct == pytest.approx(16.666666666666668, abs=1e-4)
    np.testing.assert_allclose(weight_sum, np.array([[[5.0]]], dtype=np.float32))
