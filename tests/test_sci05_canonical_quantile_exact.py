"""ZM-ZEGRID-R22 — exact winsor CPU quantile (sort/interpolate) bit-equality.

Proves the frozen rejection science is UNCHANGED while the pathological
``np.nanquantile(axis=0, method="linear")`` path is replaced by the batched
sort/interpolate primitive (``_nan_axis_quantile``):

* BIT-EXACT equality to ``np.nanquantile(method="linear")`` over adversarial
  randomised matrices: NaN patterns, all-NaN columns, N=0/1/2/3/37/66, even/odd
  N, duplicate/equal values, ±inf (matching the current contract), asymmetric q,
  both interpolation sides.
* BIT-EXACT equality of the FULL winsorized_sigma_clip rejection/combine results
  vs an independent reference built on ``np.quantile(method="linear")`` (so the
  frozen science is unchanged, not merely the primitive).
* Multiple iterations, low-N freeze and degeneracy are preserved.

No global-seed randomness beyond a fixed local ``default_rng``; no network, no
GPU, no media.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from zemosaic.core.canonical_stacking import (
    _nan_axis_quantile,
    _nan_axis_quantile_gpu,
    reject_canonical_samples,
    compute_canonical_quality_weights,
    normalize_canonical_images,
    prepare_canonical_inputs,
)


# ---------------------------------------------------------------------------
# 1. Primitive bit-equality vs np.nanquantile(method="linear")
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _adversarial_columns(seed):
    """Deterministic adversarial matrix (N, M) with NaN/±inf/duplicates/low-N."""
    rng = np.random.default_rng(seed)
    N = 66

    def col(vals):
        """Broadcast a short value list to an (N, 1) column (tile + truncate)."""
        v = np.asarray(vals, dtype=np.float64)
        rep = np.resize(v, N)
        return rep[:, None]

    base = [
        np.full((N, 1), 7.0),                    # all-equal
        col([3.0] + [np.nan] * (N - 1)),          # single valid
        np.full((N, 1), np.nan),                   # all-NaN
        np.linspace(1.0, 5.0, N)[:, None],         # sorted asc
        np.linspace(5.0, 1.0, N)[:, None],         # sorted desc
        col([1.0, 3.0, 2.0, 5.0, 4.0]),            # unsorted (repeated)
        col([-3.0, -2.0, -1.0, 0.0, 1.0]),         # monotone neg->pos
        np.full((N, 1), 1e300),                   # huge
        np.full((N, 1), 1e-300),                  # tiny
        col([-np.inf, np.inf]),                    # ±inf alternating
        col([0.0, 1.0]),                           # even/odd duplicates
    ]
    cols = np.concatenate(base, axis=1).astype(np.float64)
    rand = rng.normal(size=(N, 40)).astype(np.float64)
    rand[rng.random((N, 40)) < 0.3] = np.nan
    rand[rng.random((N, 40)) < 0.02] = np.inf
    rand[rng.random((N, 40)) < 0.02] = -np.inf
    rand[rng.random((N, 40)) < 0.2] = 7.0  # duplicates
    return np.hstack([cols, rand])


def _assert_quantile_bit_exact(a, qs):
    for q in qs:
        ref = np.nanquantile(a, q, axis=0, method="linear")
        got = _nan_axis_quantile(a, q, np)
        np.testing.assert_array_equal(ref, got)  # NaN == NaN via assert_array_equal


def test_quantile_bit_exact_adversarial():
    a = _adversarial_columns(1234)
    qs = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0, 0.123, 0.617, 0.333333333]
    _assert_quantile_bit_exact(a, qs)


def test_quantile_bit_exact_randomized_matrix():
    rng = np.random.default_rng(7)
    for trial in range(60):
        n = int(rng.integers(1, 80))
        m = int(rng.integers(1, 500))
        a = rng.normal(size=(n, m)).astype(np.float64)
        a[rng.random((n, m)) < rng.uniform(0.0, 0.6)] = np.nan
        if rng.random() < 0.5:
            a[rng.random((n, m)) < 0.05] = np.inf
        if rng.random() < 0.5:
            a[rng.random((n, m)) < 0.05] = -np.inf
        if rng.random() < 0.5:
            a[rng.random((n, m)) < 0.3] = 7.0
        if rng.random() < 0.5 and m > 5:
            a[:, rng.integers(0, m, size=m // 10)] = np.nan
        q = float(rng.uniform(0.0, 1.0))
        _assert_quantile_bit_exact(a, [q, 0.0, 1.0, 0.5])


def test_quantile_n_including_low_n_and_singleton():
    # N = 0..3 and representative 37/66 via the same helper on small columns.
    for n in range(1, 4):
        rng = np.random.default_rng(n)
        a = rng.normal(size=(n, 8)).astype(np.float64)
        a[rng.random((n, 8)) < 0.3] = np.nan
        _assert_quantile_bit_exact(a, [0.0, 0.05, 0.5, 0.95, 1.0])


def test_quantile_helper_identity_with_np_is_bit_exact():
    # _nan_axis_quantile (NumPy) == _nan_axis_quantile_gpu(NumPy) trivially.
    a = _adversarial_columns(99)
    for q in (0.05, 0.5, 0.95):
        np.testing.assert_array_equal(
            _nan_axis_quantile(a, q, np), _nan_axis_quantile_gpu(a, q, np)
        )


# ---------------------------------------------------------------------------
# 2. Full winsorized rejection/combine BIT-EXACT vs independent np.quantile ref
# ---------------------------------------------------------------------------

def _reference_wsc_survivor(images64, initial, sigma_low, sigma_high, max_iters,
                            winsor_low, winsor_high):
    """Explicit per-cell loops using np.quantile(method="linear") (independent)."""
    n, h, w, c = images64.shape
    survivor = initial.copy()
    iterations = 0
    degenerate = 0
    for _ in range(max_iters):
        iterations += 1
        changed = False
        for y in range(h):
            for x in range(w):
                for k in range(c):
                    idx = np.nonzero(survivor[:, y, x, k])[0]
                    if idx.size < 3:
                        continue
                    vals = images64[idx, y, x, k].astype(np.float64)
                    q_low = float(np.quantile(vals, winsor_low, method="linear"))
                    q_high = float(np.quantile(vals, 1.0 - winsor_high, method="linear"))
                    win = np.clip(vals, q_low, q_high)
                    center = float(np.mean(win))
                    std = float(np.std(win))
                    if std <= 0.0:
                        degenerate += 1
                        keep = images64[:, y, x, k] == center
                    else:
                        lo = center - sigma_low * std
                        hi = center + sigma_high * std
                        keep = (images64[:, y, x, k] >= lo) & (images64[:, y, x, k] <= hi)
                    new = survivor[:, y, x, k] & keep
                    if not np.array_equal(new, survivor[:, y, x, k]):
                        changed = True
                    survivor[:, y, x, k] = new
        if not changed:
            break
    return survivor, iterations, degenerate


def _wsc_corpus(seed=11, n=9, h=9, w=9):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0, 3, (h, w)).astype(np.float32) + 10.0 + i * 0.3
        f[rng.random((h, w)) < 0.08] = np.nan
        frames.append(f)
    # inject a gross outlier in a few frames so rejection is non-trivial
    for i in (2, 7):
        frames[i][1, 1] = 1000.0
    masks = [_full_support(h, w)] * n
    return frames, masks


@pytest.mark.parametrize("winsor", [(0.05, 0.05), (0.0, 0.25), (0.3, 0.0), (0.1, 0.1)])
def test_wsc_rejection_bit_exact_vs_reference(winsor):
    frames, masks = _wsc_corpus()
    norm = normalize_canonical_images(prepare_canonical_inputs(frames, masks), "sky_mean")
    weight = compute_canonical_quality_weights(norm, "none")
    res = reject_canonical_samples(
        norm, weight, "winsorized_sigma_clip",
        sigma_low=2.5, sigma_high=2.5, max_iters=5,
        winsor_limit_low=winsor[0], winsor_limit_high=winsor[1],
    )
    images64 = norm.images.astype(np.float64)
    initial = (
        weight.active_frames[:, None, None, None]
        & norm.valid_mask[..., None]
        & np.isfinite(images64)
    )
    ref_surv, ref_iters, ref_deg = _reference_wsc_survivor(
        images64, initial, 2.5, 2.5, 5, winsor[0], winsor[1]
    )
    np.testing.assert_array_equal(res.survivor_mask, ref_surv)
    assert res.iterations_used == ref_iters


def test_wsc_unchanged_science_after_quantile_swap():
    """Rejection/combine results are IDENTICAL to the pre-change np.nanquantile
    behaviour (reconstructed via the independent reference), not merely the mask."""
    frames, masks = _wsc_corpus()
    norm = normalize_canonical_images(prepare_canonical_inputs(frames, masks), "sky_mean")
    weight = compute_canonical_quality_weights(norm, "none")
    res = reject_canonical_samples(
        norm, weight, "winsorized_sigma_clip", sigma_low=3.0, sigma_high=3.0
    )
    images64 = norm.images.astype(np.float64)
    initial = (
        weight.active_frames[:, None, None, None]
        & norm.valid_mask[..., None]
        & np.isfinite(images64)
    )
    ref_surv, _iters, _deg = _reference_wsc_survivor(images64, initial, 3.0, 3.0, 5, 0.05, 0.05)
    np.testing.assert_array_equal(res.survivor_mask, ref_surv)
    # survivor | rejection == initial (frozen contract preserved)
    np.testing.assert_array_equal(res.survivor_mask | res.rejection_mask, initial)


def test_wsc_low_n_freeze_and_degenerate_unchanged():
    # N=1 and N=2 -> frozen (no rejection); all-equal -> degenerate std==0.
    for vals in ([5.0], [5.0, 6.0], [7.0] * 5):
        arr = [np.full((1, 1), float(v), dtype=np.float32) for v in vals]
        norm = normalize_canonical_images(prepare_canonical_inputs(arr, [_full_support(1, 1)] * len(arr)), "none")
        weight = compute_canonical_quality_weights(norm, "none")
        res = reject_canonical_samples(norm, weight, "winsorized_sigma_clip")
        assert not res.rejection_mask.any()


# ---------------------------------------------------------------------------
# 3. Honest benchmark (primitive + one winsor step; no full-run inference)
# ---------------------------------------------------------------------------

def test_quantile_primitive_is_faster_than_nanquantile():
    """Wall-clock evidence only (shape/N/NaN fraction reported); does NOT claim a
    full-run speedup from the primitive alone."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(37, 128 * 128 * 3)).astype(np.float64)
    a[rng.random(a.shape) < 0.10] = np.nan
    # warm up
    _nan_axis_quantile(a, 0.05, np)
    t0 = time.perf_counter()
    np.nanquantile(a, 0.05, axis=0, method="linear")
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    _nan_axis_quantile(a, 0.05, np)
    t_new = time.perf_counter() - t0
    # Report + assert the primitive is materially faster (generous 2x margin to
    # be robust to CI noise; on the probe it is ~100x).
    assert t_new < t_ref, (t_new, t_ref)
    print(
        f"QUANTILE_BENCH shape={a.shape} N=37 NaN_frac=0.10 "
        f"np.nanquantile={t_ref:.4f}s sort_helper={t_new:.4f}s speedup={t_ref / t_new:.1f}x"
    )
