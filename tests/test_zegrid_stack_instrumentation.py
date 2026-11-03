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
        if name == "boundary_sync_s":
            # No GPU workspace boundary exists on the CPU backend.
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
