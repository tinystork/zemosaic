"""Characterization witness: SCI-02 — ``stack_core`` ``linear_fit`` placeholder semantics.

Pins the *current, deterministic* behavior of ``zemosaic.zemosaic_stack_core.stack_core``
for ``normalize_method='linear_fit'`` before any product decision, without correcting,
unifying, tuning or choosing an algorithm:

1. **Placeholder semantics** — inside ``stack_core``, ``normalize_method='linear_fit'``
   executes the *exact same* code as ``normalize_method='median'`` (per-pixel median across
   the frame axis is subtracted from every frame). It is **not** a genuine per-channel
   affine mapping.
2. **Reachability** — the *only* in-repo production call to ``stack_core`` is
   ``grid_mode._stack_weighted_patches_gpu``, which normalizes upstream via
   ``_normalize_patches_gpu(..., method='linear_fit')`` and then passes
   ``normalize_method='none'`` into ``stack_core``. The placeholder is therefore bypassed
   on that route. Grid CPU never calls ``stack_core``.
3. **Distinct real linear-fit paths** — Grid ``_normalize_patches`` (covariance/variance
   regression) and classic ``_normalize_images_linear_fit`` (percentile-based) are genuine
   affine mappings and are **not** the ``stack_core`` placeholder.
4. **Normalization vs rejection** — the normalization key ``linear_fit`` is distinct from
   the rejection key/function ``linear_fit_clip`` (``_reject_outliers_linear_fit_clip`` /
   ``stack_linear_fit_clip``), which is itself a separate no-op placeholder.

This is a **characterization**, not an endorsement and not a parity expectation. Physical
GPU numerical execution is **NOT_RUN** here; the GPU path is exercised only through a
hermetic NumPy-backed fake-CuPy routing seam plus a recording ``stack_core`` stand-in to
prove *which* call contract the route takes, not to claim GPU arithmetic.

Design notes
------------
* All numeric corpora are tiny deterministic ``float32`` HWC ``4x5x3`` stacks built from
  explicit per-channel ramps and known affine transforms; no random data, no sleeps, no
  network, no GPU, no media, no profile writes.
* The affine corpus is chosen so a *genuine* affine normalization maps target patches back
  to the reference within float32 tolerance, while the median-subtraction placeholder does
  not — a discriminating proof that the placeholder is not an affine mapping.
* ``stack_core`` median vs ``linear_fit`` equivalence is asserted bit-exact
  (``np.array_equal``), because both branches run identical code; value-level closeness is
  used only where a real affine estimate is involved.
* Env is isolated via ``monkeypatch`` and never mutates real user config/profile.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from zemosaic import grid_mode
from zemosaic import zemosaic_align_stack
from zemosaic import zemosaic_config
from zemosaic import zemosaic_stack_core


# ---------------------------------------------------------------------------
# Deterministic affine corpus
# ---------------------------------------------------------------------------

_H, _W, _C = 4, 5, 3  # multi-pixel, >=2 channels, HWC

_SLOPE = np.array([1.3, 0.8, 1.1], dtype=np.float32)
_INTERCEPT = np.array([20.0, -10.0, 30.0], dtype=np.float32)
_SLOPE2 = np.array([0.6, 1.7, 0.9], dtype=np.float32)
_INTERCEPT2 = np.array([150.0, 5.0, -40.0], dtype=np.float32)


def _ref_patch() -> np.ndarray:
    """Deterministic reference patch ``H x W x C`` float32 with per-channel ramps."""
    yy, xx = np.mgrid[0:_H, 0:_W]
    ref = np.empty((_H, _W, _C), dtype=np.float32)
    ref[..., 0] = 100.0 + xx * 10 + yy * 5
    ref[..., 1] = 50.0 + xx * 2 + yy * 3
    ref[..., 2] = 200.0 + xx * 4 + yy * 7
    return ref.astype(np.float32)


def _affine(patch: np.ndarray, slope: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Per-channel affine transform ``patch * slope + intercept`` (broadcast over H/W)."""
    return (np.asarray(patch, dtype=np.float32) * slope + intercept).astype(np.float32)


def _affine_corpus() -> list[np.ndarray]:
    """``[reference, ref*slope1+i1, ref*slope2+i2]`` — a genuine affine family.

    A true per-channel affine normalization maps the two targets back onto the reference;
    median subtraction does not.
    """
    ref = _ref_patch()
    t1 = _affine(ref, _SLOPE, _INTERCEPT)
    t2 = _affine(ref, _SLOPE2, _INTERCEPT2)
    return [ref, t1, t2]


def _ones_weights(n: int) -> list[np.ndarray]:
    return [np.ones((_H, _W, _C), dtype=np.float32) for _ in range(n)]


def _core_config(normalize: str, *, final: str = "mean") -> dict:
    return {
        "normalize_method": normalize,
        "rejection_algorithm": "none",
        "final_combine_method": final,
    }


def _grid_config(norm: str = "linear_fit", reject: str = "none") -> grid_mode.GridModeConfig:
    cfg = grid_mode.GridModeConfig()
    cfg.stack_norm_method = norm
    cfg.stack_reject_algo = reject
    cfg.stack_final_combine = "mean"
    return cfg


def _run_core(images, normalize, *, final="mean"):
    return zemosaic_stack_core.stack_core(
        images=images,
        weights=_ones_weights(len(images)),
        stack_config=_core_config(normalize, final=final),
        backend="cpu",
    )


# ---------------------------------------------------------------------------
# A. Direct stack_core semantics (real CPU implementation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("final", ["mean", "median"])
def test_stack_core_linear_fit_identical_to_median(final):
    """``linear_fit`` runs the exact median-subtraction code; bit-identical to ``median``."""
    images = _affine_corpus()
    lf_result, lf_rej, lf_ws = _run_core(images, "linear_fit", final=final)
    med_result, med_rej, med_ws = _run_core(images, "median", final=final)

    np.testing.assert_array_equal(lf_result, med_result)
    assert float(lf_rej) == float(med_rej) == 0.0
    np.testing.assert_array_equal(lf_ws, med_ws)


def test_stack_core_linear_fit_differs_from_none():
    """The placeholder does change the stack (median subtraction), unlike ``none``."""
    images = _affine_corpus()
    lf_result, _, _ = _run_core(images, "linear_fit")
    none_result, _, _ = _run_core(images, "none")

    assert lf_result.shape == none_result.shape == (_H, _W, _C)
    assert lf_result.dtype == none_result.dtype == np.float32
    # Median subtraction recentres the stack around zero; 'none' keeps the raw mean.
    assert not np.allclose(lf_result, none_result, rtol=1e-4, atol=1e-4)


def test_stack_core_linear_fit_is_not_affine_mapping():
    """Placeholder is NOT a genuine affine mapping.

    On the affine corpus, Grid's real ``_normalize_patches(linear_fit)`` maps every target
    back onto the reference (max err ~3e-5), whereas ``stack_core`` ``linear_fit``
    (median subtraction) does not — its output stays far from the reference.
    """
    ref = _ref_patch()
    images = _affine_corpus()

    normalized, _ref_used = grid_mode._normalize_patches(images, None, method="linear_fit")
    for patch in normalized:
        np.testing.assert_allclose(patch, ref, rtol=1e-3, atol=1e-3)

    core_result, _, _ = _run_core(images, "linear_fit")
    # Median subtraction leaves residual per-pixel offsets; it does NOT collapse the
    # affine family onto the reference (which would be the affine-mapping behavior).
    assert not np.allclose(core_result, ref, rtol=1e-1, atol=1.0)
    assert float(np.max(np.abs(core_result - ref))) > 1.0


def test_stack_core_shapes_dtypes_weights():
    """Pin output shape/dtype/weight_sum/rejected_pct for the placeholder path."""
    images = _affine_corpus()
    result, rejected_pct, weight_sum = _run_core(images, "linear_fit")

    assert result.shape == (_H, _W, _C)
    assert result.dtype == np.float32
    assert weight_sum.shape == (_H, _W, _C)
    assert weight_sum.dtype == np.float32
    assert float(rejected_pct) == 0.0
    # Equal unit weights, no rejection -> weight_sum == N everywhere.
    np.testing.assert_allclose(weight_sum, np.full((_H, _W, _C), 3.0, dtype=np.float32))


# NaN / masked semantics are NOT_RUN here: the corpus is all-finite and does not exercise
# NaN handling or finite-mask behavior.


# ---------------------------------------------------------------------------
# B. Grid CPU upstream path
# ---------------------------------------------------------------------------


def test_grid_cpu_normalize_receives_linear_fit(monkeypatch):
    """Dynamically prove ``_normalize_patches`` is called with ``method='linear_fit'``."""
    images = _affine_corpus()
    seen = {}
    real = grid_mode._normalize_patches

    def spy(patches, reference_median=None, *, method="median"):
        seen["method"] = method
        seen["called"] = True
        return real(patches, reference_median, method=method)

    monkeypatch.setattr(grid_mode, "_normalize_patches", spy)
    grid_mode._stack_weighted_patches(images, _ones_weights(len(images)), _grid_config("linear_fit"))

    assert seen.get("called") is True
    assert seen.get("method") == "linear_fit"


def test_grid_cpu_linear_fit_aligns_affine_corpus():
    """Real Grid CPU upstream normalization maps targets onto the reference; the stacking
    output reflects the aligned patches (== reference for a pure affine family)."""
    ref = _ref_patch()
    images = _affine_corpus()

    normalized, _ = grid_mode._normalize_patches(images, None, method="linear_fit")
    for patch in normalized:
        np.testing.assert_allclose(patch, ref, rtol=1e-3, atol=1e-3)

    result, weight_sum = grid_mode._stack_weighted_patches(
        images, _ones_weights(len(images)), _grid_config("linear_fit"), return_weight_sum=True
    )
    np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1e-3)
    assert result.shape == (_H, _W, _C)
    assert result.dtype == np.float32


def test_grid_cpu_never_calls_stack_core(monkeypatch):
    """Forbidden monkeypatch: ``_stack_weighted_patches`` must not invoke ``stack_core``.

    If the Grid CPU route ever started calling ``stack_core``, this sentinel would fire and
    fail the test. The route completes without touching it.
    """
    images = _affine_corpus()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Grid CPU must not call stack_core")

    monkeypatch.setattr(grid_mode, "stack_core", forbidden)
    result = grid_mode._stack_weighted_patches(
        images, _ones_weights(len(images)), _grid_config("linear_fit")
    )

    assert result is not None
    assert calls == []


# ---------------------------------------------------------------------------
# C. Grid GPU-core routing seam (no physical GPU)
# ---------------------------------------------------------------------------


class _FakeCupy:
    """Hermetic NumPy-backed stand-in for CuPy — routing only, never numerical parity."""

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
    def nanmedian(x):
        return np.nanmedian(x)

    @staticmethod
    def nanmean(x):
        return np.nanmean(x)

    @staticmethod
    def nanvar(x):
        return np.nanvar(x)

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


def test_gpu_core_routing_seam_linear_fit(monkeypatch):
    """Invoke ``_stack_weighted_patches_gpu`` through a fake-CuPy seam and capture the
    exact ``stack_core`` call contract, proving ``linear_fit`` is handled *upstream* and
    the core receives ``normalize_method='none'``.

    The test must fail if the placeholder ``linear_fit`` is ever passed into ``stack_core``
    on this route (the explicit ``!= 'linear_fit'`` assertion).
    """
    images = _affine_corpus()
    ref = _ref_patch()

    # Spy the upstream GPU normalizer: it must receive method='linear_fit'.
    normalize_seen = {}
    real_norm_gpu = grid_mode._normalize_patches_gpu

    def spy_norm_gpu(patches, reference_median=None, *, method="median"):
        normalize_seen["method"] = method
        return real_norm_gpu(patches, reference_median, method=method)

    # Recording stand-in for stack_core.
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
            np.zeros((_H, _W, _C), dtype=np.float32),
            0.0,
            np.ones((_H, _W, _C), dtype=np.float32),
        )

    monkeypatch.setattr(grid_mode, "_CUPY_AVAILABLE", True)
    monkeypatch.setattr(grid_mode, "cp", _FakeCupy())
    monkeypatch.setattr(grid_mode, "_normalize_patches_gpu", spy_norm_gpu)
    monkeypatch.setattr(grid_mode, "stack_core", recording_stack_core)

    cfg = _grid_config("linear_fit")
    cfg.use_gpu = True
    out = grid_mode._stack_weighted_patches_gpu(
        images,
        _ones_weights(len(images)),
        cfg,
        return_weight_sum=True,
        return_ref_median=True,
        raise_on_gpu_failure=True,
        gpu_failure_context={},
    )

    assert isinstance(out, tuple) and len(out) == 3

    # Upstream normalizer saw linear_fit and produced already-normalized inputs.
    assert normalize_seen.get("method") == "linear_fit"
    assert len(records) == 1

    call = records[0]
    assert call["backend"] == "gpu"
    cfg_in = call["stack_config"]
    assert cfg_in["normalize_method"] == "none"
    # Explicit guard: the placeholder must never reach the core on this route.
    assert cfg_in["normalize_method"] != "linear_fit"

    # The images passed to stack_core are the upstream-normalized (aligned) patches.
    assert isinstance(call["images"], list) and len(call["images"]) == len(images)
    for img in call["images"]:
        np.testing.assert_allclose(np.asarray(img), ref, rtol=1e-3, atol=1e-3)

    assert isinstance(call["weights"], list) and len(call["weights"]) == len(images)


# ---------------------------------------------------------------------------
# D. Production reachability map (AST-based, no brittle line numbers)
# ---------------------------------------------------------------------------


def _enclosing_function(tree: ast.AST, node: ast.AST):
    parent = {}
    for n in ast.walk(tree):
        for child in ast.iter_child_nodes(n):
            parent[child] = n
    cur = node
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parent.get(cur)
    return None


def test_stack_core_production_caller_inventory_ast():
    """AST witness: the only production ``stack_core`` call site is
    ``grid_mode._stack_weighted_patches_gpu``.

    Avoids grep/source-string line numbers; records module + enclosing function only.
    """
    src_dir = pathlib.Path(zemosaic_stack_core.__file__).resolve().parent
    call_sites: list[tuple[str, str]] = []

    for path in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "stack_core":
                call_sites.append((path.name, _enclosing_function(tree, node)))

    assert call_sites == [("grid_mode.py", "_stack_weighted_patches_gpu")]


def test_stack_core_is_internal_not_exported():
    """``stack_core`` is an internal module-level symbol, not part of the public package
    surface (``zemosaic.__all__`` only exposes ``__version__``)."""
    import zemosaic

    assert "stack_core" not in getattr(zemosaic, "__all__", ())
    assert not hasattr(zemosaic, "stack_core")
    assert callable(zemosaic_stack_core.stack_core)


# ---------------------------------------------------------------------------
# E. Other real linear-fit paths + normalization vs rejection
# ---------------------------------------------------------------------------


def test_classic_normalize_images_linear_fit_is_real_affine_not_placeholder():
    """Classic ``_normalize_images_linear_fit`` (percentile-based) is a genuine affine
    mapping: it aligns the affine targets back onto the reference, unlike the placeholder."""
    ref = _ref_patch()
    images = _affine_corpus()

    normalized = zemosaic_align_stack._normalize_images_linear_fit(images, reference_index=0)
    assert len(normalized) == len(images)
    np.testing.assert_array_equal(normalized[0], ref)
    for patch in normalized[1:]:
        np.testing.assert_allclose(patch, ref, rtol=1e-3, atol=1e-3)


def test_grid_affine_regression_distinct_from_classic_percentile(monkeypatch):
    """Dynamically prove the two genuine affine normalizers are distinct code paths.

    Grid ``_normalize_patches(linear_fit)`` dispatches through ``_fit_linear_scale``
    (covariance/variance regression); classic ``_normalize_images_linear_fit`` dispatches
    through ``_calculate_robust_stats_for_linear_fit`` (percentiles). Neither uses the other.
    """
    images = _affine_corpus()

    grid_calls = []
    classic_calls = []
    real_fit = grid_mode._fit_linear_scale
    real_stats = zemosaic_align_stack._calculate_robust_stats_for_linear_fit

    def spy_fit(ref_patch, patch):
        grid_calls.append(True)
        return real_fit(ref_patch, patch)

    def spy_stats(*args, **kwargs):
        classic_calls.append(True)
        return real_stats(*args, **kwargs)

    monkeypatch.setattr(grid_mode, "_fit_linear_scale", spy_fit)
    monkeypatch.setattr(zemosaic_align_stack, "_calculate_robust_stats_for_linear_fit", spy_stats)

    grid_mode._normalize_patches(images, None, method="linear_fit")
    zemosaic_align_stack._normalize_images_linear_fit(images, reference_index=0)

    assert grid_calls  # grid uses covariance/variance regression
    assert classic_calls  # classic uses percentile stats
    assert real_fit is not real_stats


def test_linear_fit_clip_rejection_is_separate_noop_placeholder():
    """``_reject_outliers_linear_fit_clip`` is a *rejection* placeholder: it returns the
    input unchanged with an all-True keep mask (does no normalization, no rejection)."""
    stack = np.stack(_affine_corpus(), axis=0)
    out, mask = zemosaic_align_stack._reject_outliers_linear_fit_clip(stack)

    np.testing.assert_array_equal(out, stack)
    assert mask.dtype == bool
    assert bool(np.all(mask))


def test_selecting_normalization_does_not_select_rejection(monkeypatch):
    """Selecting the normalization key ``linear_fit`` never selects the rejection
    key/function ``linear_fit_clip`` (dynamic + structural proof)."""
    images = _affine_corpus()
    reject_calls = []

    def forbidden_reject(*args, **kwargs):
        reject_calls.append((args, kwargs))
        raise AssertionError("normalization must not select linear_fit_clip rejection")

    monkeypatch.setattr(zemosaic_align_stack, "_reject_outliers_linear_fit_clip", forbidden_reject)
    monkeypatch.setattr(zemosaic_align_stack, "stack_linear_fit_clip", forbidden_reject)

    # Grid CPU route with normalization=linear_fit, rejection=none.
    grid_mode._stack_weighted_patches(images, _ones_weights(len(images)), _grid_config("linear_fit", "none"))
    # Shared core route with normalization=linear_fit, rejection=none.
    _run_core(images, "linear_fit")
    # Classic normalize helper (the real affine path) — no rejection involved.
    zemosaic_align_stack._normalize_images_linear_fit(images, reference_index=0)

    assert reject_calls == []

    # Structural: distinct config keys; 'linear_fit' is a normalization value while
    # 'linear_fit_clip' belongs to the rejection family (and stack_core has no such branch).
    assert zemosaic_config.DEFAULT_CONFIG["stacking_normalize_method"] == "linear_fit"
    assert zemosaic_config.DEFAULT_CONFIG["stacking_rejection_algorithm"] == "winsorized_sigma_clip"


def test_stack_core_has_no_linear_fit_clip_rejection_branch():
    """``stack_core`` only implements ``kappa_sigma`` / ``winsorized_sigma_clip`` rejection;
    ``linear_fit_clip`` is not a valid rejection algorithm there and falls through to no-op
    (identical output to rejection='none', rejected_pct == 0)."""
    images = _affine_corpus()

    none_result, none_rej, _ = _run_core(images, "none")
    clip_result, clip_rej, _ = zemosaic_stack_core.stack_core(
        images=images,
        weights=_ones_weights(len(images)),
        stack_config={
            "normalize_method": "none",
            "rejection_algorithm": "linear_fit_clip",
            "final_combine_method": "mean",
        },
        backend="cpu",
    )

    np.testing.assert_array_equal(clip_result, none_result)
    assert float(clip_rej) == 0.0 == float(none_rej)
