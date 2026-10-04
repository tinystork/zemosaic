"""Characterization witness: SCI-01 — Grid winsorized-sigma-clip (WSC) divergence.

Pins the *current, deterministic* numerical divergence among three stacking
routes before any product decision, without correcting, harmonizing, tuning or
choosing a winner:

1. **Grid CPU** and **Grid GPU-legacy fallback** call
   ``_reject_outliers_winsorized_sigma_clip`` **without** an explicit
   ``wsc_impl``, so ``resolve_wsc_impl()`` resolves env > config > default and
   lands on the PixInsight WSC core by default (``ZEMOSAIC_WSC_IMPL`` env can
   override to ``legacy_quantile``).
2. **Grid GPU core** passes ``winsorized_sigma_clip`` through to
   ``zemosaic_stack_core.stack_core``, whose ``winsorized_sigma_clip`` branch is
   a *simplified* median/std sigma clip — neither PixInsight WSC nor legacy
   quantile winsorization.

This is a **characterization**, not an endorsement and not a parity expectation.
Physical GPU numerical execution is **NOT_RUN** here; the GPU paths are exercised
only through hermetic routing seams (a NumPy-backed fake CuPy module plus a
recording ``stack_core`` stand-in) to prove *which* code path and *which* call
contract each route takes, not to claim GPU arithmetic.

Design notes
------------
* All numeric corpora are tiny deterministic ``float32`` HWC ``1x1x1`` stacks
  (single pixel, ``N`` frames), matching the Junior probe values so the pinned
  numbers reproduce exactly and are not copied blindly (they are re-derived
  against the real code paths below).
* Env is isolated with ``monkeypatch`` (``delenv``/``setenv``) and never mutates
  real user config/profile.
* No random corpus, no sleeps, no network, no GPU, no media.
* NaN/finite semantics are asserted with explicit masks; ``equal_nan`` is avoided
  so a real value is never confused with a NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic import grid_mode
from zemosaic import zemosaic_align_stack
from zemosaic import zemosaic_stack_core
from zemosaic.core.robust_rejection import (
    WSC_IMPL_LEGACY,
    WSC_IMPL_PIXINSIGHT,
    resolve_wsc_impl,
    wsc_pixinsight_core,
)


# ---------------------------------------------------------------------------
# Hermetic helpers
# ---------------------------------------------------------------------------

_WSC_ENV = "ZEMOSAIC_WSC_IMPL"
# Matches zemosaic_align_stack._WSC_PIXINSIGHT_MAX_ITERS (pinned at 10).
_PIXINSIGHT_MAX_ITERS = 10

IMPULSE = [0.0, 0.0, 0.0, 0.0, 100.0]
NONDEGENERATE = [1.0, 1.0, 1.0, 2.0, 100.0]
GRADIENT = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]

CORPORA = {
    "impulse": IMPULSE,
    "nondegenerate": NONDEGENERATE,
    "gradient": GRADIENT,
}


def _px_patches(values):
    """Single-pixel HWC ``(1,1,1)`` float32 patches from per-frame scalars."""
    return [np.array([[[float(v)]]], dtype=np.float32) for v in values]


def _ones_weights(n):
    return [np.ones((1, 1, 1), dtype=np.float32) for _ in range(n)]


def _wsc_grid_config():
    cfg = grid_mode.GridModeConfig()
    cfg.stack_norm_method = "none"
    cfg.stack_reject_algo = "winsorized_sigma_clip"
    cfg.stack_final_combine = "mean"
    cfg.winsor_limits = (0.05, 0.05)
    cfg.stack_kappa_low = 2.5
    cfg.stack_kappa_high = 2.5
    return cfg


def _core_config():
    return {
        "normalize_method": "none",
        "rejection_algorithm": "winsorized_sigma_clip",
        "final_combine_method": "mean",
        "sigma_clip_low": 2.5,
        "sigma_clip_high": 2.5,
    }


def _data_stack(values):
    return np.stack([np.asarray(p, dtype=np.float32) for p in _px_patches(values)], axis=0)


def _run_grid_cpu(values):
    """Run the real Grid CPU route; returns ``(result, weight_sum)``."""
    result, weight_sum = grid_mode._stack_weighted_patches(
        _px_patches(values),
        _ones_weights(len(values)),
        _wsc_grid_config(),
        return_weight_sum=True,
    )
    return np.asarray(result), np.asarray(weight_sum)


def _run_core_cpu(values):
    """Run the real shared ``stack_core`` CPU backend; returns
    ``(result, rejected_pct, weight_sum)``."""
    result, rejected_pct, weight_sum = zemosaic_stack_core.stack_core(
        images=_px_patches(values),
        weights=_ones_weights(len(values)),
        stack_config=_core_config(),
        backend="cpu",
    )
    return np.asarray(result), float(rejected_pct), np.asarray(weight_sum)


class _FakeCupy:
    """Hermetic NumPy-backed stand-in for CuPy.

    Used **only** to route ``_stack_weighted_patches_gpu`` through its GPU-core
    and GPU-legacy seams without a physical GPU. This is not a claim of GPU
    numerical execution; every CuPy call is aliased to NumPy so the seam can be
    exercised deterministically.
    """

    float32 = np.float32
    float64 = np.float64
    nan = np.nan
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
    def nanmedian(x):
        return np.nanmedian(x)

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


# ---------------------------------------------------------------------------
# A. Resolver contract (hermetic env)
# ---------------------------------------------------------------------------


def test_resolver_env_absent_defaults_to_pixinsight(monkeypatch):
    monkeypatch.delenv(_WSC_ENV, raising=False)
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_valid_legacy_env_overrides_default(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "legacy_quantile")
    assert resolve_wsc_impl() == WSC_IMPL_LEGACY


def test_resolver_valid_pixinsight_env(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "pixinsight")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_env_is_case_insensitive_and_trimmed(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, " Legacy_Quantile ")
    assert resolve_wsc_impl() == WSC_IMPL_LEGACY
    monkeypatch.setenv(_WSC_ENV, "PIXINSIGHT")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_invalid_env_falls_through_to_default(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "bogus_impl")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT
    monkeypatch.setenv(_WSC_ENV, "")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_never_creates_new_mode_from_invalid_env(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "not-a-real-impl")
    got = resolve_wsc_impl()
    assert got in {WSC_IMPL_PIXINSIGHT, WSC_IMPL_LEGACY}
    assert got == WSC_IMPL_PIXINSIGHT


# ---------------------------------------------------------------------------
# B. Grid CPU dispatch and numerical contract
# ---------------------------------------------------------------------------


def test_grid_cpu_does_not_pass_explicit_wsc_impl(monkeypatch):
    """Prove Grid CPU dispatches through the real helper *without* ``wsc_impl``.

    Dynamic proof, not a source-text assertion. A ``*args/**kwargs`` spy can
    distinguish an omitted keyword from an explicitly-passed ``wsc_impl=None``
    (the latter would bind ``'wsc_impl'`` inside ``kwargs``); the witness asserts
    the keyword is absent, so it fails if production ever starts passing
    ``wsc_impl=None`` explicitly.
    """
    monkeypatch.delenv(_WSC_ENV, raising=False)
    seen = {}
    real = grid_mode._reject_outliers_winsorized_sigma_clip

    def spy(*args, **kwargs):
        seen["wsc_impl_in_kwargs"] = "wsc_impl" in kwargs
        seen["called"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(grid_mode, "_reject_outliers_winsorized_sigma_clip", spy)
    _run_grid_cpu(NONDEGENERATE)

    assert seen.get("called") is True
    assert seen.get("wsc_impl_in_kwargs") is False


@pytest.mark.parametrize("name,values", list(CORPORA.items()))
def test_grid_cpu_env_absent_matches_explicit_pixinsight(monkeypatch, name, values):
    """Grid CPU with env absent must equal Grid CPU with env='pixinsight'."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    result_default, wsum_default = _run_grid_cpu(values)

    monkeypatch.setenv(_WSC_ENV, "pixinsight")
    result_explicit, wsum_explicit = _run_grid_cpu(values)

    np.testing.assert_array_equal(result_default, result_explicit)
    np.testing.assert_array_equal(wsum_default, wsum_explicit)


@pytest.mark.parametrize("name,values", list(CORPORA.items()))
def test_grid_cpu_matches_direct_pixinsight_core(monkeypatch, name, values):
    """Grid CPU (env absent) equals the direct PixInsight WSC core on the same
    stack (all-finite broadcast + equal weights => combined mean == core output)."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    result, weight_sum = _run_grid_cpu(values)

    stack = _data_stack(values)
    pix = wsc_pixinsight_core(
        np,
        stack,
        sigma_low=2.5,
        sigma_high=2.5,
        max_iters=_PIXINSIGHT_MAX_ITERS,
    )

    # Single-pixel corpora: core output is a (1,1,1) array.
    assert pix.shape == (1, 1, 1)
    # Tolerance accounts for float32 accumulation in the Grid combine (the same
    # PixInsight float64 core result rounded twice: once by the core, once by the
    # float32 weighted-mean reduce). Residual is ~1 float32 ULP, not a semantic gap.
    np.testing.assert_allclose(result, pix, rtol=1e-5, atol=1e-6)


def test_grid_cpu_impulse_corpus(monkeypatch):
    monkeypatch.delenv(_WSC_ENV, raising=False)
    result, weight_sum = _run_grid_cpu(IMPULSE)

    assert result.shape == (1, 1, 1)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    # PixInsight WSC collapses the [0,0,0,0,100] stack to ~0 (the 100 outlier is
    # winsorized to ~2.5e-10), NOT to the 20.0 arithmetic mean.
    assert result[0, 0, 0] == pytest.approx(5e-11, abs=1e-9)
    np.testing.assert_allclose(weight_sum, np.array([[[5.0]]], dtype=np.float32))


def test_grid_cpu_nondegenerate_corpus(monkeypatch):
    monkeypatch.delenv(_WSC_ENV, raising=False)
    result, weight_sum = _run_grid_cpu(NONDEGENERATE)

    assert result.shape == (1, 1, 1)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result[0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(weight_sum, np.array([[[5.0]]], dtype=np.float32))


def test_grid_cpu_gradient_corpus(monkeypatch):
    monkeypatch.delenv(_WSC_ENV, raising=False)
    result, weight_sum = _run_grid_cpu(GRADIENT)

    assert result.shape == (1, 1, 1)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result[0, 0, 0] == pytest.approx(6.5740323, abs=1e-4)
    np.testing.assert_allclose(weight_sum, np.array([[[6.0]]], dtype=np.float32))


# ---------------------------------------------------------------------------
# C. Simplified core numerical contract
# ---------------------------------------------------------------------------


def test_core_impulse_corpus():
    result, rejected_pct, weight_sum = _run_core_cpu(IMPULSE)

    assert result.shape == (1, 1, 1)
    assert result.dtype == np.float32
    # Simplified median/std clip keeps everything (100 is within +-2.5 std of the
    # median 0) and returns the plain mean.
    assert result[0, 0, 0] == pytest.approx(20.0, abs=1e-5)
    assert rejected_pct == pytest.approx(0.0)
    np.testing.assert_allclose(weight_sum, np.array([[[5.0]]], dtype=np.float32))


def test_core_nondegenerate_corpus():
    result, rejected_pct, weight_sum = _run_core_cpu(NONDEGENERATE)

    assert result[0, 0, 0] == pytest.approx(1.25, abs=1e-5)
    assert rejected_pct == pytest.approx(20.0)
    np.testing.assert_allclose(weight_sum, np.array([[[4.0]]], dtype=np.float32))


def test_core_gradient_corpus():
    result, rejected_pct, weight_sum = _run_core_cpu(GRADIENT)

    assert result[0, 0, 0] == pytest.approx(3.0, abs=1e-5)
    assert rejected_pct == pytest.approx(16.666666666666668, abs=1e-4)
    np.testing.assert_allclose(weight_sum, np.array([[[5.0]]], dtype=np.float32))


@pytest.mark.parametrize(
    "name,values,expected_delta",
    [
        ("impulse", IMPULSE, 20.0),
        ("nondegenerate", NONDEGENERATE, 0.25),
        ("gradient", GRADIENT, 3.5740323),
    ],
)
def test_core_materially_diverges_from_grid_cpu(monkeypatch, name, values, expected_delta):
    """Quantify the abs delta between simplified ``stack_core`` and default Grid CPU."""
    monkeypatch.delenv(_WSC_ENV, raising=False)
    grid_result, _ = _run_grid_cpu(values)
    core_result, _, _ = _run_core_cpu(values)

    delta = abs(float(core_result[0, 0, 0]) - float(grid_result[0, 0, 0]))
    assert delta == pytest.approx(expected_delta, abs=1e-4)


# ---------------------------------------------------------------------------
# D. Grid GPU-core routing seam (no physical GPU)
# ---------------------------------------------------------------------------


def test_gpu_core_routing_seam_contract(monkeypatch):
    """Invoke ``_stack_weighted_patches_gpu`` through a fake-CuPy seam and capture
    the exact ``stack_core`` call contract without a physical GPU.

    The recording stand-in must be called exactly once; the seam cannot pass
    while bypassing ``stack_core``.
    """
    monkeypatch.delenv(_WSC_ENV, raising=False)
    records: list[dict] = []

    def recording_stack_core(images=None, weights=None, stack_config=None,
                             backend="cpu", progress_callback=None):
        records.append({
            "images": images,
            "weights": weights,
            "stack_config": stack_config,
            "backend": backend,
        })
        return (
            np.zeros((1, 1, 1), dtype=np.float32),
            0.0,
            np.ones((1, 1, 1), dtype=np.float32),
        )

    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())
    monkeypatch.setattr(grid_mode, "stack_core", recording_stack_core)

    values = NONDEGENERATE
    out = grid_mode._stack_weighted_patches_gpu(
        _px_patches(values),
        _ones_weights(len(values)),
        _wsc_grid_config(),
        return_weight_sum=True,
        return_ref_median=True,
        raise_on_gpu_failure=True,
        gpu_failure_context={},
    )

    assert isinstance(out, tuple) and len(out) == 3
    assert len(records) == 1
    call = records[0]

    assert call["backend"] == "gpu"
    cfg = call["stack_config"]
    assert cfg["normalize_method"] == "none"
    assert cfg["rejection_algorithm"] == "winsorized_sigma_clip"
    assert cfg["final_combine_method"] == "mean"
    assert cfg["sigma_clip_low"] == 2.5
    assert cfg["sigma_clip_high"] == 2.5
    # Grid GPU core drops winsor_limits entirely (documented divergence).
    assert "winsor_limits" not in cfg

    assert isinstance(call["images"], list) and len(call["images"]) == len(values)
    assert isinstance(call["weights"], list) and len(call["weights"]) == len(values)


# ---------------------------------------------------------------------------
# E. Grid GPU-legacy routing seam (no physical GPU)
# ---------------------------------------------------------------------------


def test_gpu_legacy_routing_seam_no_explicit_wsc_impl(monkeypatch):
    """Force ``stack_core`` unavailable; the GPU-legacy fallback must route through
    the real helper *without* an explicit ``wsc_impl``.

    Dynamic proof: a ``*args/**kwargs`` spy asserts ``'wsc_impl'`` is absent from
    ``kwargs`` (an explicit ``wsc_impl=None`` would bind the key), so it fails if
    production ever starts passing it explicitly.
    """
    monkeypatch.delenv(_WSC_ENV, raising=False)
    seen = {}
    real = grid_mode._reject_outliers_winsorized_sigma_clip

    def spy(*args, **kwargs):
        seen["wsc_impl_in_kwargs"] = "wsc_impl" in kwargs
        seen["called"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())
    monkeypatch.setattr(grid_mode, "stack_core", None)
    monkeypatch.setattr(grid_mode, "_reject_outliers_winsorized_sigma_clip", spy)

    values = NONDEGENERATE
    out = grid_mode._stack_weighted_patches_gpu(
        _px_patches(values),
        _ones_weights(len(values)),
        _wsc_grid_config(),
        return_weight_sum=True,
        raise_on_gpu_failure=True,
        gpu_failure_context={},
    )

    assert seen.get("called") is True
    assert seen.get("wsc_impl_in_kwargs") is False
    assert isinstance(out, tuple)
    result, weight_sum = np.asarray(out[0]), np.asarray(out[1])
    assert result.shape == (1, 1, 1)
    assert result.dtype == np.float32


def test_gpu_legacy_env_switches_impl(monkeypatch):
    """Valid legacy env changes the GPU-legacy fallback implementation, proving
    env resolution (not a hard-coded path) drives the legacy seam."""
    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())
    monkeypatch.setattr(grid_mode, "stack_core", None)

    values = IMPULSE

    monkeypatch.delenv(_WSC_ENV, raising=False)
    out_default = grid_mode._stack_weighted_patches_gpu(
        _px_patches(values), _ones_weights(len(values)), _wsc_grid_config(),
        return_weight_sum=True, raise_on_gpu_failure=True, gpu_failure_context={},
    )
    result_default = float(np.asarray(out_default[0])[0, 0, 0])

    monkeypatch.setenv(_WSC_ENV, "legacy_quantile")
    out_legacy = grid_mode._stack_weighted_patches_gpu(
        _px_patches(values), _ones_weights(len(values)), _wsc_grid_config(),
        return_weight_sum=True, raise_on_gpu_failure=True, gpu_failure_context={},
    )
    result_legacy = float(np.asarray(out_legacy[0])[0, 0, 0])

    # Default PixInsight collapses the impulse to ~0; legacy rewinsorizes the 100
    # outlier to ~80 and averages to 16. The two are materially different.
    assert result_default == pytest.approx(5e-11, abs=1e-9)
    assert result_legacy == pytest.approx(16.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Legacy quantile numerics (explicitly gated on SciPy + Astropy)
# ---------------------------------------------------------------------------


def test_grid_cpu_legacy_quantile_numerics(monkeypatch):
    """Explicit ``legacy_quantile`` (scipy/astropy path) numerics, distinct from
    both PixInsight default and simplified core.

    Requires the real legacy winsorize + sigma-clipped-stats dependencies so the
    fail-open "return unrejected" fallback in ``_reject_outliers_winsorized_sigma_clip``
    cannot masquerade as legacy qualification.
    """
    if not (
        zemosaic_align_stack.SCIPY_AVAILABLE
        and zemosaic_align_stack.winsorize_func is not None
        and zemosaic_align_stack.SIGMA_CLIP_AVAILABLE
        and zemosaic_align_stack.sigma_clipped_stats_func is not None
    ):
        pytest.skip(
            "legacy_quantile requires scipy.stats.mstats.winsorize and "
            "astropy.stats.sigma_clipped_stats"
        )

    monkeypatch.setenv(_WSC_ENV, "legacy_quantile")

    imp, ws_imp = _run_grid_cpu(IMPULSE)
    nd, ws_nd = _run_grid_cpu(NONDEGENERATE)
    grad, ws_grad = _run_grid_cpu(GRADIENT)

    assert imp[0, 0, 0] == pytest.approx(16.0, abs=1e-5)
    assert nd[0, 0, 0] == pytest.approx(17.08, abs=1e-5)
    assert grad[0, 0, 0] == pytest.approx(15.208333, abs=1e-4)

    np.testing.assert_allclose(ws_imp, np.array([[[5.0]]], dtype=np.float32))
    np.testing.assert_allclose(ws_nd, np.array([[[5.0]]], dtype=np.float32))
    np.testing.assert_allclose(ws_grad, np.array([[[6.0]]], dtype=np.float32))
