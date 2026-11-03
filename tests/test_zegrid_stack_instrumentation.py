"""ZM-STACK-INSTRUMENTATION-20261103 focused regressions.

Covers the two implementation prerequisites:

* ``BoundedPipeline._consume_loop`` counts EVERY ``queue.get`` wait, including
  ``queue.Empty`` timeout polls and the terminal producer-death/stall paths,
  exactly once (the historical bug dropped every empty poll).
* The default-OFF bounded stacking phase profiler is opt-in, bounded, strict
  JSON, and is bit-exact ON vs OFF for science/support/rejection outputs.
"""

from __future__ import annotations

import json
import queue

import numpy as np
import pytest

import zemosaic.core.canonical_streaming as cs_streaming
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import stack_phase_profile as zspp


_EMPTY = object()


class _ScriptedQueue:
    """Deterministic queue.get script (``_EMPTY`` raises queue.Empty)."""

    def __init__(self, script):
        self._script = list(script)
        self.get_calls = 0

    def get(self, timeout=None):  # noqa: ARG002 - API parity
        self.get_calls += 1
        item = self._script.pop(0)
        if item is _EMPTY:
            raise queue.Empty
        return item

    def qsize(self):
        return 0


class _StepClock:
    def __init__(self, start=0.0, step=0.1):
        self._value = float(start)
        self._step = float(step)

    def __call__(self):
        value = self._value
        self._value += self._step
        return value


class _DeadThread:
    name = "dead-producer"

    def is_alive(self):
        return False


def _pipe(clock, **kw):
    params = dict(
        tasks=[0],
        produce=lambda task: {"index": 0, "device_bytes": 8},
        consume=lambda payload: 1,
        release=lambda payload: None,
        prefetch_units=0,
        host_budget_bytes=8,
        device_budget_bytes=8,
        host_bytes=lambda payload: 8,
        device_bytes=lambda payload: 8,
        clock=clock,
    )
    params.update(kw)
    return zpar.BoundedPipeline(**params)


# ---------------------------------------------------------------------------
# 1. Consumer idle accounting
# ---------------------------------------------------------------------------


def test_consumer_idle_counts_every_timeout_poll_exactly_once():
    clock = _StepClock()
    pipe = _pipe(clock)
    # Three timed-out polls, then the terminal sentinel.
    pipe._queue = _ScriptedQueue([_EMPTY, _EMPTY, _EMPTY, {"sentinel": True}])
    pipe._consume_loop()
    # Four queue.get calls, each 0.1 s apart: every wait is counted once.
    assert pipe._queue.get_calls == 4
    assert pipe.telemetry()["consumer_idle_s"] == pytest.approx(0.4)
    assert pipe.telemetry()["consumer_idle_semantics"].startswith(
        "host_consumer_queue_wait_s"
    )
    # Only the successful get samples the queue depth.
    assert pipe.telemetry()["queue_depth_samples"] == 1


def test_consumer_idle_counts_the_poll_that_observes_producer_death():
    clock = _StepClock()
    pipe = _pipe(clock)
    pipe._queue = _ScriptedQueue([_EMPTY])
    pipe._producer_thread = _DeadThread()
    pipe._consume_loop()
    telemetry = pipe.telemetry()
    assert telemetry["consumer_idle_s"] == pytest.approx(0.1)
    assert telemetry["producer_error"].startswith("PipelineError")
    assert telemetry["cancel_reason"] == "producer_exited"


def test_consumer_idle_counts_the_poll_that_trips_the_stall_watchdog():
    clock = _StepClock(step=0.05)
    pipe = _pipe(clock, stall_timeout_s=0.01)
    pipe._last_progress_at = -10.0
    # started=0.0, idle-end=0.05, stall-check=0.10 > deadline.
    pipe._queue = _ScriptedQueue([_EMPTY])
    pipe._consume_loop()
    telemetry = pipe.telemetry()
    assert telemetry["consumer_idle_s"] == pytest.approx(0.05)
    assert telemetry["stalled"] is True
    assert telemetry["cancel_reason"] == "stall_timeout"


class _CancelOnEmptyQueue:
    def __init__(self, pipe):
        self._pipe = pipe
        self.get_calls = 0

    def get(self, timeout=None):  # noqa: ARG002
        self.get_calls += 1
        # Deterministic cancellation DURING the empty poll: the loop top will
        # observe it on the next iteration.
        self._pipe.cancel("empty_poll_cancel")
        raise queue.Empty

    def qsize(self):
        return 0


def test_consumer_idle_counts_cancellation_during_empty_poll():
    clock = _StepClock()
    pipe = _pipe(clock)
    pipe._queue = _CancelOnEmptyQueue(pipe)
    pipe._consume_loop()
    telemetry = pipe.telemetry()
    assert telemetry["consumer_idle_s"] == pytest.approx(0.1)
    assert telemetry["cancel_reason"] == "empty_poll_cancel"
    assert telemetry["producer_error"] is None


# ---------------------------------------------------------------------------
# 2. Profiler bounded / strict JSON / parity
# ---------------------------------------------------------------------------


def _corpus(n=6, h=24, w=24, seed=11):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        frames.append(
            (100.0 + rng.normal(0.0, 1.5, (h, w))).astype(np.float32)
        )
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    frames[1][3:7, 3:7] = np.nan
    return frames, masks


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


def test_profiler_on_is_bit_identical_and_strict_json():
    frames, masks = _corpus()
    request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection="winsorized_sigma_clip",
        combine="median",
        taper="footprint",
    )
    provider_on = InMemoryCanonicalProvider(frames, masks)
    provider_off = InMemoryCanonicalProvider(frames, masks)
    profiler = zspp.StackPhaseProfiler(backend="cpu")
    on = run_canonical_stack_streaming(
        provider_on, request, tile_size=8, phase_profiler=profiler
    )
    off = run_canonical_stack_streaming(provider_off, request, tile_size=8)
    _assert_planes_equal(off, on)

    snap = profiler.snapshot()
    # Strict JSON: no NaN/Inf, plain types only.
    encoded = json.dumps(snap, allow_nan=False, sort_keys=True)
    assert json.loads(encoded)["schema"] == zspp.STACK_PHASE_PROFILE_SCHEMA
    assert snap["tiles"] > 0
    assert snap["backend"] == "cpu"
    assert snap["device"]["combine_stream_span_s"] is None
    assert snap["device"]["unavailable_reason"] == "cpu_backend"
    for name in zspp.HOST_PHASES:
        assert name in snap["host_phases"]
        if name in ("boundary_sync_s", "h2d_transfer_s", "d2h_transfer_s"):
            # No GPU workspace boundary / backend transfer on the CPU path.
            assert snap["host_phases"][name]["count"] == 0
        else:
            assert snap["host_phases"][name]["count"] >= 1


def test_profiler_off_adds_nothing_to_the_result():
    frames, masks = _corpus()
    request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection="none",
        combine="mean",
    )
    result = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), request, tile_size=8
    )
    # The profiler is not part of the canonical result type at all.
    assert not hasattr(result, "stack_phase_profile")


def test_aggregate_is_bounded_strict_json_and_attribution_explicit():
    frames, masks = _corpus()
    request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection="none",
        combine="mean",
    )
    profiler = zspp.StackPhaseProfiler(backend="cpu")
    run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks),
        request,
        tile_size=8,
        phase_profiler=profiler,
    )
    agg = zspp.aggregate([profiler.snapshot()], attribution="current")
    json.dumps(agg, allow_nan=False, sort_keys=True)
    assert agg["cells_profiled"] == 1
    assert agg["attribution"] == "current"
    assert agg["resume_attribution"]["current"] == 1
    assert agg["resume_attribution"]["historical_restored"] == 0

    historical = zspp.aggregate(
        [profiler.snapshot()], attribution="historical_restored"
    )
    assert historical["resume_attribution"]["historical_restored"] == 1
    assert historical["resume_attribution"]["current"] == 0


def test_aggregate_refuses_malformed_entries_without_raising():
    agg = zspp.aggregate([{"schema": "nope"}, None, 42])  # type: ignore[list-item]
    assert agg["cells_profiled"] == 0
    assert agg["cells_malformed"] == 3


def _cpu_profiled_snapshot():
    frames, masks = _corpus()
    request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection="none",
        combine="mean",
    )
    profiler = zspp.StackPhaseProfiler(backend="cpu")
    run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks),
        request,
        tile_size=8,
        phase_profiler=profiler,
    )
    return profiler.snapshot()


def test_aggregate_same_schema_bad_counter_is_atomic_and_never_raises():
    snap = _cpu_profiled_snapshot()
    snap["tiles"] = "bad"
    agg = zspp.aggregate([snap])
    assert agg["cells_profiled"] == 0
    assert agg["cells_malformed"] == 1
    json.dumps(agg, allow_nan=False, sort_keys=True)


def test_aggregate_malformed_bucket_contributes_nothing():
    snap = _cpu_profiled_snapshot()
    # Damage ONE nested bucket; the whole snapshot must be rejected, not
    # partially summed.
    snap["host_phases"]["writes_s"]["count"] = "oops"
    agg = zspp.aggregate([snap])
    assert agg["cells_profiled"] == 0
    assert agg["cells_malformed"] == 1
    assert all(
        bucket["total_s"] == 0.0 and bucket["count"] == 0
        for bucket in agg["host_phases"].values()
    )


def test_aggregate_preserves_perturbation():
    profiler = zspp.StackPhaseProfiler(backend="gpu")
    profiler.add_device_span(0.1)
    profiler.note_extra_sync(2)
    profiler.count_tile(1)
    agg = zspp.aggregate([profiler.snapshot()], attribution="current")
    assert agg["perturbation"]["extra_synchronizations"] == 2
    assert agg["perturbation"]["cuda_event_sync_per_tile"] is True
    assert agg["perturbation"]["cells_with_cuda_event_sync"] == 1


def test_aggregate_records_identity_split_and_missing_profiles():
    profile = _cpu_profiled_snapshot()
    records = [
        {"cell_id": "a", "stack_phase_profile": profile},
        {"cell_id": "b"},  # current pending cell with no profile (empty)
        {"cell_id": "c", "stack_phase_profile": profile},  # restored
        {"cell_id": "d"},  # old restored store without a profile
    ]
    current, historical = zspp.aggregate_records(
        records, current_ids={"a", "b"}
    )
    assert current["cells_in_attempt"] == 2
    assert current["cells_profiled"] == 1
    assert current["cells_without_profile"] == 1
    assert current["resume_attribution"]["current"] == 1
    assert historical["cells_restored"] == 2
    assert historical["cells_profiled"] == 1
    assert historical["cells_without_profile"] == 1
    assert historical["resume_attribution"]["historical_restored"] == 1

    # All-current and all-restored boundaries use the same identity rule.
    all_current, none_historical = zspp.aggregate_records(
        records, current_ids=set(), all_current=True
    )
    assert all_current["cells_profiled"] == 2
    assert none_historical["cells_profiled"] == 0
    none_current, all_historical = zspp.aggregate_records(
        records, current_ids=set()
    )
    assert none_current["cells_profiled"] == 0
    assert all_historical["cells_profiled"] == 2
    json.dumps(
        {"current": current, "historical": historical},
        allow_nan=False,
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# 3. Faithful CPU-backed CuPy API double (constructor + elapsed contract)
# ---------------------------------------------------------------------------


class _FakeEvent:
    """CuPy-shaped Event: keyword signature enforced, NO elapsed_time()."""

    def __init__(self, block=False, disable_timing=False, interprocess=False):
        del block, interprocess
        _FakeCuda.created += 1
        self.disable_timing = disable_timing
        self.recorded = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        _FakeCuda.syncs += 1


class _FakeCuda:
    created = 0
    syncs = 0
    elapsed_calls = 0
    fail_create = False
    fail_elapsed = False

    @classmethod
    def reset(cls):
        cls.created = 0
        cls.syncs = 0
        cls.elapsed_calls = 0
        cls.fail_create = False
        cls.fail_elapsed = False

    def Event(self, **kwargs):
        if _FakeCuda.fail_create:
            raise TypeError("fake Event construction failure")
        return _FakeEvent(**kwargs)

    def get_elapsed_time(self, start, end):
        del start, end
        _FakeCuda.elapsed_calls += 1
        if _FakeCuda.fail_elapsed:
            raise RuntimeError("fake elapsed failure")
        return 1000.0  # milliseconds -> 1.0 s per tile


class _FakeXP:
    """NumPy-backed module distinct from ``np`` with a CuPy-shaped cuda API."""

    cuda = _FakeCuda()
    float32 = np.float32
    newaxis = np.newaxis

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asarray(x, *a, **k):
        return np.asarray(x, *a, **k)

    @staticmethod
    def asnumpy(x, *a, **k):
        return np.asarray(x, *a, **k)


_FAKE_XP = _FakeXP()


def _install_fake_gpu(monkeypatch):
    monkeypatch.setattr(cs_streaming, "canonical_gpu_available", lambda: True)
    monkeypatch.setattr(
        cs_streaming,
        "_get_xp",
        lambda backend: _FAKE_XP if backend == "gpu" else np,
    )
    monkeypatch.setattr(
        cs_streaming, "_release_gpu_workspace", lambda xp, **kw: None
    )


def _gpu_request(frames, masks, *, combine, rejection):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection=rejection,
        combine=combine,
        taper="footprint",
        backend="gpu",
    )


@pytest.mark.parametrize(
    "combine,rejection",
    [("mean", "winsorized_sigma_clip"), ("median", "kappa_sigma")],
)
def test_cpu_backed_cupy_double_parity_on_off(
    monkeypatch, combine, rejection
):
    frames, masks = _corpus()
    cpu_request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection=rejection,
        combine=combine,
        taper="footprint",
        backend="cpu",
    )
    reference = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), cpu_request, tile_size=8
    )
    _install_fake_gpu(monkeypatch)
    gpu_request = _gpu_request(
        frames, masks, combine=combine, rejection=rejection
    )

    _FakeCuda.reset()
    profiler = zspp.StackPhaseProfiler(backend="gpu")
    on = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks),
        gpu_request,
        tile_size=8,
        phase_profiler=profiler,
    )
    _assert_planes_equal(reference, on)
    snap = profiler.snapshot()
    assert snap["backend"] == "gpu"
    assert snap["device"]["unavailable_reason"] is None
    assert snap["device"]["combine_event_pairs"] == snap["tiles"]
    assert snap["device"]["combine_stream_span_s"] == pytest.approx(
        float(snap["tiles"])
    )
    assert snap["perturbation"]["extra_synchronizations"] == snap["tiles"]
    assert snap["host_phases"]["h2d_transfer_s"]["count"] > 0
    assert snap["host_phases"]["d2h_transfer_s"]["count"] > 0
    assert snap["host_phases"]["host_reduction_s"]["count"] > 0
    # The double enforces the real API: two events per tile, one sync each.
    assert _FakeCuda.created == 2 * snap["tiles"]
    assert _FakeCuda.syncs == snap["tiles"]
    assert _FakeCuda.elapsed_calls == snap["tiles"]

    # OFF: no event, no measurement sync, no elapsed query, same science.
    _FakeCuda.reset()
    off = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), gpu_request, tile_size=8
    )
    _assert_planes_equal(reference, off)
    assert _FakeCuda.created == 0
    assert _FakeCuda.syncs == 0
    assert _FakeCuda.elapsed_calls == 0


def test_cupy_event_creation_failure_is_truthful(monkeypatch):
    frames, masks = _corpus()
    _install_fake_gpu(monkeypatch)
    gpu_request = _gpu_request(
        frames, masks, combine="mean", rejection="none"
    )
    _FakeCuda.reset()
    _FakeCuda.fail_create = True
    profiler = zspp.StackPhaseProfiler(backend="gpu")
    run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks),
        gpu_request,
        tile_size=8,
        phase_profiler=profiler,
    )
    snap = profiler.snapshot()
    assert snap["device"]["combine_stream_span_s"] is None
    assert snap["device"]["unavailable_reason"].startswith(
        "cuda_event_unavailable"
    )
    assert snap["perturbation"]["extra_synchronizations"] == 0
    assert snap["perturbation"]["cuda_event_sync_per_tile"] is False
    json.dumps(snap, allow_nan=False, sort_keys=True)


def test_cupy_elapsed_failure_still_counts_real_sync(monkeypatch):
    frames, masks = _corpus()
    _install_fake_gpu(monkeypatch)
    gpu_request = _gpu_request(
        frames, masks, combine="mean", rejection="none"
    )
    _FakeCuda.reset()
    _FakeCuda.fail_elapsed = True
    profiler = zspp.StackPhaseProfiler(backend="gpu")
    run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks),
        gpu_request,
        tile_size=8,
        phase_profiler=profiler,
    )
    snap = profiler.snapshot()
    assert snap["device"]["combine_stream_span_s"] is None
    assert snap["device"]["unavailable_reason"].startswith(
        "cuda_event_elapsed_failed"
    )
    # The synchronization really happened, so it must be counted even though
    # no elapsed value is available.
    assert snap["perturbation"]["extra_synchronizations"] == snap["tiles"]
    assert _FakeCuda.syncs == snap["tiles"]
    json.dumps(snap, allow_nan=False, sort_keys=True)
