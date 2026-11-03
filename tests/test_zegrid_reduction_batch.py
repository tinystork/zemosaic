"""ZM-STACK-REDUCTION-BATCH-20261103 focused regressions.

The stacking rejection/combine path runs several host-side statistics over the
SAME device array. Historically each statistic performed its own ``xp.asnumpy``,
so a winsorized iteration downloaded the identical bytes twice (11 D2H per tile
for 12 reductions in the measured Windows run).

The fix downloads once per array and hands the same host float64 copy to every
reducer. These tests prove, deterministically:

* (a) bit-identical values vs the pre-change (legacy) implementation on
  adversarial matrices (NaN, +/-Inf, all-NaN, N<3, ties, partial masks);
* (b) the D2H call count per tile is reduced, measured from profiler counts and
  from the backend double's ``asnumpy`` counter (never from wall time);
* (c) science/coverage/support/rejection are identical between the existing
  CPU-backed CuPy double and the CPU path.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import zemosaic.core.canonical_stacking as cs
import zemosaic.core.canonical_streaming as cs_streaming
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import stack_phase_profile as zspp


# ---------------------------------------------------------------------------
# Legacy (pre-change) reference implementation, copied verbatim in behaviour.
# ---------------------------------------------------------------------------


def _legacy_nanmean(a, xp, profile=None):
    if xp is np:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=0)
    if profile is not None:
        import time as _time
        _t = _time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        host = xp.asnumpy(a)
    if profile is not None:
        profile.add("d2h_transfer_s", _time.perf_counter() - _t)
        _t = _time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        host = np.nanmean(host, axis=0)
    if profile is not None:
        profile.add("host_reduction_s", _time.perf_counter() - _t)
        _t = _time.perf_counter()
    result = xp.asarray(host)
    if profile is not None:
        profile.add("h2d_transfer_s", _time.perf_counter() - _t)
    return result


def _legacy_nanstd(a, xp, profile=None):
    if xp is np:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanstd(a, axis=0)
    if profile is not None:
        import time as _time
        _t = _time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        host = xp.asnumpy(a)
    if profile is not None:
        profile.add("d2h_transfer_s", _time.perf_counter() - _t)
        _t = _time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        host = np.nanstd(host, axis=0)
    if profile is not None:
        profile.add("host_reduction_s", _time.perf_counter() - _t)
        _t = _time.perf_counter()
    result = xp.asarray(host)
    if profile is not None:
        profile.add("h2d_transfer_s", _time.perf_counter() - _t)
    return result


def _legacy_sum_f64(a, xp, profile=None):
    if xp is np:
        return np.sum(a, axis=0, dtype=np.float64)
    if profile is not None:
        import time as _time
        _t = _time.perf_counter()
    host = xp.asnumpy(a)
    if profile is not None:
        profile.add("d2h_transfer_s", _time.perf_counter() - _t)
        _t = _time.perf_counter()
    result = np.sum(host, axis=0, dtype=np.float64)
    if profile is not None:
        profile.add("host_reduction_s", _time.perf_counter() - _t)
    return result


def _legacy_winsor_step(
    orig2, survivor2, count, sigma_low, sigma_high, winsor_low, winsor_high, xp,
    profile=None,
):
    active = count >= 3
    masked = xp.where(survivor2, orig2, xp.nan)
    q_low = cs._nan_axis_quantile(masked, winsor_low, xp)
    q_high = cs._nan_axis_quantile(masked, 1.0 - winsor_high, xp)
    winsor = xp.clip(masked, q_low[xp.newaxis, :], q_high[xp.newaxis, :])
    center = _legacy_nanmean(winsor, xp, profile=profile)
    std = _legacy_nanstd(winsor, xp, profile=profile)
    keep, degenerate = cs._apply_sigma_interval(
        orig2, center, std, active, sigma_low, sigma_high, xp
    )
    return survivor2 & keep, degenerate


def _legacy_kappa_step(
    orig2, survivor2, count, sigma_low, sigma_high, xp, profile=None
):
    active = count >= 3
    masked = xp.where(survivor2, orig2, xp.nan)
    center = cs._nan_axis_median(masked, xp)
    std = _legacy_nanstd(masked, xp, profile=profile)
    keep, degenerate = cs._apply_sigma_interval(
        orig2, center, std, active, sigma_low, sigma_high, xp
    )
    return survivor2 & keep, degenerate


# ---------------------------------------------------------------------------
# CuPy API double (same shape as tests/test_zegrid_stack_instrumentation.py),
# extended with a truthful ``asnumpy`` (D2H) call counter.
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, block=False, disable_timing=False, interprocess=False):
        del block, interprocess
        self.disable_timing = disable_timing
        self.recorded = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        pass


class _FakeCuda:
    def Event(self, **kwargs):
        return _FakeEvent(**kwargs)

    def get_elapsed_time(self, start, end):
        del start, end
        return 1000.0  # milliseconds -> 1.0 s per tile


class _CountingXP:
    """NumPy-backed module distinct from ``np`` with CuPy-shaped cuda API."""

    float32 = np.float32
    newaxis = np.newaxis

    def __init__(self):
        self.cuda = _FakeCuda()
        self.asnumpy_calls = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, x, *a, **k):
        return np.asarray(x, *a, **k)

    def asnumpy(self, x, *a, **k):
        self.asnumpy_calls += 1
        return np.asarray(x, *a, **k)


def _install_fake_gpu(monkeypatch):
    monkeypatch.setattr(cs_streaming, "canonical_gpu_available", lambda: True)
    monkeypatch.setattr(
        cs_streaming,
        "_get_xp",
        lambda backend: _CountingXP() if backend == "gpu" else np,
    )
    monkeypatch.setattr(
        cs_streaming, "_release_gpu_workspace", lambda xp, **kw: None
    )


# ---------------------------------------------------------------------------
# Adversarial inputs.
# ---------------------------------------------------------------------------


def _adversarial_matrices():
    rng = np.random.default_rng(20_261_103)
    return [
        ("random", rng.normal(5.0, 2.0, (7, 6))),
        ("nan_holes", np.where(rng.random((9, 5)) < 0.35,
                               np.nan, rng.normal(size=(9, 5)))),
        (
            "nan_and_inf",
            np.array(
                [
                    [1.0, np.nan, 2.0, -np.inf, 4.0, 5.0],
                    [3.0, np.nan, np.inf, 2.0, 4.0, 5.0],
                    [np.nan, np.nan, 2.0, 2.0, np.nan, 5.0],
                    [4.0, 7.0, np.nan, 1.0, 1.0, np.inf],
                ],
                dtype=np.float64,
            ),
        ),
        ("all_nan", np.full((4, 4), np.nan)),
        ("n_1", rng.normal(size=(1, 5))),
        ("n_2", rng.normal(size=(2, 5))),
        ("ties_constant", np.full((6, 3), 2.5)),
        ("ties_quantised", np.tile(np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])[:, None], (1, 4))),
        (
            "mixed_sign_magnitude",
            np.array(
                [
                    [1e-12, -1e12, 0.0, 5.0],
                    [1e12, -1e-12, -0.0, 5.0],
                    [3.0, 3.0, 0.0, -5.0],
                ],
                dtype=np.float64,
            ),
        ),
    ]


def _assert_bit_identical(new, legacy):
    np.testing.assert_array_equal(np.asarray(new), np.asarray(legacy))
    assert np.asarray(new).dtype == np.asarray(legacy).dtype


# ---------------------------------------------------------------------------
# (a) bit-identical reductions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,mat", _adversarial_matrices(), ids=lambda v: getattr(v, "shape", ""))
def test_mean_std_sum_bit_identical_to_legacy(name, mat):
    del name
    xp = _CountingXP()
    a = np.array(mat, dtype=np.float64, copy=True)

    _assert_bit_identical(cs._nan_axis_mean(a, np), _legacy_nanmean(a, np))
    _assert_bit_identical(cs._nan_axis_popstd(a, np), _legacy_nanstd(a, np))
    _assert_bit_identical(cs._axis_sum_f64(a, np), _legacy_sum_f64(a, np))

    _assert_bit_identical(
        cs._nan_axis_mean(a, xp), _legacy_nanmean(a, np)
    )
    _assert_bit_identical(
        cs._nan_axis_popstd(a, xp), _legacy_nanstd(a, np)
    )
    _assert_bit_identical(
        cs._axis_sum_f64(a, xp), _legacy_sum_f64(a, np)
    )
    # One array -> exactly one download per single-statistic helper.
    assert xp.asnumpy_calls == 3


def _step_case(n, m, seed):
    rng = np.random.default_rng(seed)
    orig2 = rng.normal(10.0, 3.0, (n, m))
    orig2[rng.random((n, m)) < 0.2] = np.nan
    # partial masks: drop arbitrary samples per column, including <3 survivors
    survivor2 = rng.random((n, m)) > 0.25
    survivor2[:, 0] = True
    survivor2[:, 1] = False
    if m > 2:
        survivor2[:1, 2] = True  # single survivor cell
    if m > 3:
        survivor2[:-1, 3] = True  # n-1 survivors
    orig2[~survivor2] = np.nan
    count = survivor2.sum(axis=0)
    return orig2, survivor2, count


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "params",
    [
        (3.0, 3.0, 0.05, 0.05),
        (2.0, 4.0, 0.25, 0.0),
        (5.0, 1.5, 0.0, 0.49),
    ],
)
def test_winsor_step_bit_identical_and_one_download(seed, params):
    sigma_low, sigma_high, w_low, w_high = params
    orig2, survivor2, count = _step_case(9, 7, seed)
    xp_new = _CountingXP()
    xp_legacy = _CountingXP()
    prof_new = zspp.StackPhaseProfiler(backend="gpu")
    prof_legacy = zspp.StackPhaseProfiler(backend="gpu")

    new = cs._winsorized_sigma_clip_step(
        orig2, survivor2, count, sigma_low, sigma_high, w_low, w_high,
        xp_new, profile=prof_new,
    )
    legacy = _legacy_winsor_step(
        orig2, survivor2, count, sigma_low, sigma_high, w_low, w_high,
        xp_legacy, profile=prof_legacy,
    )

    np.testing.assert_array_equal(new[0], legacy[0])
    np.testing.assert_array_equal(new[1], legacy[1])
    assert new[0].dtype == legacy[0].dtype

    # One iteration: 2 downloads before, 1 after; same host reductions.
    assert xp_legacy.asnumpy_calls == 2
    assert xp_new.asnumpy_calls == 1
    assert prof_new.snapshot()["host_phases"]["d2h_transfer_s"]["count"] == 1
    assert prof_legacy.snapshot()["host_phases"]["d2h_transfer_s"]["count"] == 2
    for snap in (prof_new.snapshot(), prof_legacy.snapshot()):
        assert snap["host_phases"]["host_reduction_s"]["count"] == 2
        assert snap["host_phases"]["h2d_transfer_s"]["count"] == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kappa_step_bit_identical_and_unchanged_downloads(seed):
    orig2, survivor2, count = _step_case(8, 6, seed)
    xp_new = _CountingXP()
    xp_legacy = _CountingXP()
    new = cs._kappa_sigma_step(orig2, survivor2, count, 3.0, 3.0, xp_new)
    legacy = _legacy_kappa_step(orig2, survivor2, count, 3.0, 3.0, xp_legacy)
    np.testing.assert_array_equal(new[0], legacy[0])
    np.testing.assert_array_equal(new[1], legacy[1])
    assert xp_new.asnumpy_calls == xp_legacy.asnumpy_calls == 1


def test_winsor_step_profile_off_is_semantically_neutral():
    orig2, survivor2, count = _step_case(9, 7, 42)
    xp_off = _CountingXP()
    prof = zspp.StackPhaseProfiler(backend="gpu")
    off = cs._winsorized_sigma_clip_step(
        orig2, survivor2, count, 3.0, 3.0, 0.05, 0.05, xp_off
    )
    on = cs._winsorized_sigma_clip_step(
        orig2, survivor2, count, 3.0, 3.0, 0.05, 0.05, _CountingXP(), profile=prof
    )
    np.testing.assert_array_equal(off[0], on[0])
    np.testing.assert_array_equal(off[1], on[1])
    assert xp_off.asnumpy_calls == 1


# ---------------------------------------------------------------------------
# (b)+(c) end-to-end streaming: identical planes, fewer D2H per tile.
# ---------------------------------------------------------------------------


def _corpus(n=7, h=18, w=18, seed=11):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0.0, 1.5, (h, w, 3)).astype(np.float32)
        f += np.array([100.0, 120.0, 140.0], np.float32) + 0.5 * i
        f[rng.random((h, w, 3)) < 0.1] = np.nan
        frames.append(f)
    frames[2][2, 2, :] = 500.0
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[1][5:9, 5:9] = False
    masks[3][:, 12:] = False
    return frames, masks


def _request(frames, masks, backend, combine, rejection):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection=rejection,
        combine=combine,
        taper="footprint",
        backend=backend,
    )


def _run(frames, masks, request, tile_size, phase_profiler=None):
    return run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks),
        request,
        tile_size=tile_size,
        phase_profiler=phase_profiler,
    )


def _assert_planes_equal(a, b):
    np.testing.assert_array_equal(a.science, b.science)
    np.testing.assert_array_equal(a.estimator_weight_sum, b.estimator_weight_sum)
    np.testing.assert_array_equal(a.valid_mask, b.valid_mask)
    np.testing.assert_array_equal(a.surviving_sample_count, b.surviving_sample_count)
    np.testing.assert_array_equal(a.support_w1, b.support_w1)
    np.testing.assert_array_equal(a.support_w2, b.support_w2)
    np.testing.assert_array_equal(a.rejection_mask, b.rejection_mask)
    assert a.initial_sample_count == b.initial_sample_count
    assert a.rejected_sample_count == b.rejected_sample_count
    assert a.rejected_fraction == b.rejected_fraction
    assert a.iterations_used == b.iterations_used
    assert a.low_n_cell_count == b.low_n_cell_count
    assert a.degenerate_cell_count == b.degenerate_cell_count


@pytest.mark.parametrize(
    "combine,rejection",
    [("mean", "winsorized_sigma_clip"), ("median", "kappa_sigma")],
)
def test_streaming_gpu_double_matches_cpu_and_cuts_d2h(monkeypatch, combine, rejection):
    frames, masks = _corpus()
    tile = 6
    cpu = _run(
        frames, masks, _request(frames, masks, "cpu", combine, rejection), tile
    )

    _install_fake_gpu(monkeypatch)

    prof_new = zspp.StackPhaseProfiler(backend="gpu")
    gpu_new = _run(
        frames, masks, _request(frames, masks, "gpu", combine, rejection), tile,
        phase_profiler=prof_new,
    )
    _assert_planes_equal(cpu, gpu_new)

    # Emulate the pre-change behaviour: one download PER reducer.
    def _legacy_download_and_reduce(a, xp, profile, reducers):
        # Pre-change behaviour: EVERY reducer performed its own xp.asnumpy.
        if xp is np:
            return [fn(a) for fn in reducers]
        out = []
        for fn in reducers:
            if profile is not None:
                import time as _time
                _t = _time.perf_counter()
            host = xp.asnumpy(a)
            if profile is not None:
                profile.add("d2h_transfer_s", _time.perf_counter() - _t)
            if profile is not None:
                _t = _time.perf_counter()
            out.append(fn(host))
            if profile is not None:
                profile.add("host_reduction_s", _time.perf_counter() - _t)
        return out

    monkeypatch.setattr(cs, "_download_and_reduce", _legacy_download_and_reduce)
    prof_legacy = zspp.StackPhaseProfiler(backend="gpu")
    gpu_legacy = _run(
        frames, masks, _request(frames, masks, "gpu", combine, rejection), tile,
        phase_profiler=prof_legacy,
    )
    _assert_planes_equal(cpu, gpu_legacy)
    _assert_planes_equal(gpu_new, gpu_legacy)

    snap_new = prof_new.snapshot()
    snap_legacy = prof_legacy.snapshot()
    tiles = snap_new["tiles"]
    assert tiles > 0
    d2h_new = snap_new["host_phases"]["d2h_transfer_s"]["count"]
    d2h_legacy = snap_legacy["host_phases"]["d2h_transfer_s"]["count"]
    if rejection == "winsorized_sigma_clip":
        assert d2h_new < d2h_legacy
    else:
        # kappa_sigma already downloaded once per statistic: unchanged.
        assert d2h_new == d2h_legacy
    # Host reductions themselves are unchanged (2 stats per winsor iteration,
    # 2 sums for mean-combine); only the redundant downloads are removed.
    assert (
        snap_new["host_phases"]["host_reduction_s"]["count"]
        == snap_legacy["host_phases"]["host_reduction_s"]["count"]
    )
    if rejection == "winsorized_sigma_clip":
        # Exactly one redundant download removed per winsorized iteration.
        iters = gpu_new.iterations_used
        assert d2h_legacy - d2h_new == tiles * iters
