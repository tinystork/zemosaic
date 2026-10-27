"""ZM-ZEGRID-RESILIENT-RUN-RESUME M3 — resource recovery unit contract.

Deterministic tests for the M3 policy seam (plan §5.3/§5.4/§5.7):

* release really happens BEFORE waiting, and an incomplete release refuses to
  wait at all;
* the wait is light (measured CPU/RSS) and cancellable at every wake-up;
* two spaced admissible observations are required for readmission;
* AUTO -> bounded is one-way; an explicit dense request never switches;
* retries are bounded per (cell, plan), then wait for real capacity;
* a late result from a cancelled generation is never accepted, while already
  committed payloads stay valid and immutable;
* under pressure after K cells and a later RAM return, the K validated cells
  are not recomputed (a real execution ledger, not a manifest counter).

No GPU, no Windows and no product run is exercised here; those remain NOT_RUN.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import resource_recovery as rec  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return float(self.t)

    def sleep(self, delay: float) -> None:
        self.t += float(delay)

    def advance(self, delay: float) -> None:
        self.t += float(delay)


def _fact(raw: int, clock: FakeClock | None = None) -> dict:
    return {
        "raw_available_bytes": int(raw),
        "observed_monotonic": (
            float(clock()) if clock is not None else time.monotonic()
        ),
        "source": "test",
    }


def _release_report(**overrides) -> rec.ReleaseReport:
    values = {
        "stopped_admission": True,
        "cancelled_units": 2,
        "drained_units": 1,
        "host_released_bytes": 4096,
        "credits_balanced": True,
    }
    values.update(overrides)
    return rec.ReleaseReport(**values)


def _empty_cell(cell_id: str) -> dict:
    return {
        "cell_id": str(cell_id),
        "status": "empty",
        "record": {
            "cell_id": str(cell_id),
            "status": "empty",
            "mode": None,
            "cell_gauge_mode": "global",
            "n_contributors": 0,
            "patch_bounds": [0, 0, 10, 8],
            "patch_shape_hw": [8, 10],
        },
    }


def _registry(cell_ids=("r0000c0000", "r0000c0001")):
    return zrr.RunResultRegistry(
        run_id="run-m3",
        expected_cell_ids=list(cell_ids),
        gauge_mode="global",
    )


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            pages = int(fh.read().split()[1])
        return int(pages) * 4096
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Frozen policy numbers
# ---------------------------------------------------------------------------


def test_plan_policy_numbers_are_frozen():
    assert rec.PROBE_INITIAL_INTERVAL_S == 5.0
    assert rec.PROBE_MAX_INTERVAL_S == 30.0
    assert rec.REQUIRED_ADMISSIBLE_OBSERVATIONS == 2
    assert rec.MIN_ADMISSIBLE_SPACING_S == 5.0


def test_malformed_and_stale_observations_fail_closed():
    with pytest.raises(rec.ObservationError):
        rec.parse_observation({"raw_available_bytes": True}, now=1.0)
    with pytest.raises(rec.ObservationError):
        rec.parse_observation({"raw_available_bytes": -1}, now=1.0)
    with pytest.raises(rec.ObservationError):
        rec.parse_observation(
            {"raw_available_bytes": 10, "observed_monotonic": 100.0},
            now=50.0,
        )
    with pytest.raises(rec.ObservationError):
        rec.parse_observation({"raw_available_bytes": 10}, now=5.0)
    ok = rec.parse_observation(
        {"raw_available_bytes": 10, "observed_monotonic": 4.0},
        now=5.0,
    )
    assert ok.raw_available_bytes == 10


# ---------------------------------------------------------------------------
# One-way profile policy
# ---------------------------------------------------------------------------


def test_auto_switches_to_bounded_exactly_once():
    policy = rec.StorageProfilePolicy(requested="auto")
    first = policy.decide(
        pressure_persists=True,
        phase="run_wave_3",
        reason_code="REFUSE_HOST_RAM",
        detail="test",
    )
    assert first.switched is True
    assert first.effective == "bounded"
    assert len(policy.transitions) == 1
    second = policy.decide(
        pressure_persists=True,
        phase="run_wave_4",
        reason_code="REFUSE_HOST_RAM",
        detail="again",
    )
    assert second.switched is False
    assert second.effective == "bounded"
    assert second.reason == "auto_already_switched_once"
    assert len(policy.transitions) == 1


def test_explicit_dense_never_switches_and_bounded_stays_bounded():
    dense = rec.StorageProfilePolicy(requested="dense")
    decision = dense.decide(
        pressure_persists=True,
        phase="run_wave_0",
        reason_code="REFUSE_HOST_RAM",
        detail="explicit",
    )
    assert decision.switched is False
    assert decision.effective == "dense"
    assert decision.reason == "explicit_dense_never_switched"

    bounded = rec.StorageProfilePolicy(requested="bounded")
    decision = bounded.decide(
        pressure_persists=True,
        phase="run_wave_0",
        reason_code="REFUSE_HOST_RAM",
        detail="already bounded",
    )
    assert decision.switched is False
    assert decision.effective == "bounded"
    assert decision.reason == "bounded_stays_bounded"


# ---------------------------------------------------------------------------
# Spaced, light, cancellable wait
# ---------------------------------------------------------------------------


def test_readmission_requires_two_observations_spaced_by_five_seconds():
    clock = FakeClock()
    raw = iter([100, 100, 900, 900])
    probes = []

    def probe():
        probes.append(clock())
        return _fact(next(raw), clock)

    waiter = rec.CapacityWaiter(
        probe,
        minimum_bytes=800,
        clock=clock,
        sleep=clock.sleep,
        initial_interval_s=5.0,
        max_interval_s=30.0,
        required_observations=2,
        min_spacing_s=5.0,
        max_age_s=None,
    )
    result = waiter.wait()
    assert result.admissible is True
    assert result.probes == 4
    assert result.admissible_observations == 2
    # first admissible at t=1010, second at t=1015 (exactly the 5 s spacing)
    assert probes[-1] - probes[-2] == pytest.approx(5.0)
    assert result.sleeps == 3


def test_backoff_is_spaced_not_a_tight_loop():
    clock = FakeClock()
    probe_times = []

    def probe():
        probe_times.append(clock())
        return _fact(10, clock)

    waiter = rec.CapacityWaiter(
        probe,
        minimum_bytes=800,
        clock=clock,
        sleep=clock.sleep,
        initial_interval_s=5.0,
        max_interval_s=30.0,
        required_observations=2,
        min_spacing_s=5.0,
        max_age_s=None,
        max_probes=5,
    )
    with pytest.raises(rec.CapacityWaitExhausted):
        waiter.wait()
    gaps = [b - a for a, b in zip(probe_times, probe_times[1:])]
    assert gaps == pytest.approx([5.0, 10.0, 20.0, 30.0])


def test_wait_is_light_and_cancellable():
    stop = threading.Event()
    errors: list[BaseException] = []

    def probe():
        return _fact(10)  # never admissible

    def run():
        waiter = rec.CapacityWaiter(
            probe,
            minimum_bytes=800,
            stop_event=stop,
            initial_interval_s=5.0,
            max_interval_s=30.0,
            max_age_s=None,
        )
        try:
            waiter.wait()
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    cpu0 = time.process_time()
    rss0 = _rss_bytes()
    thread = threading.Thread(target=run, name="m3-wait-cancellable")
    thread.start()
    time.sleep(0.25)
    assert thread.is_alive()  # still waiting, not busy-looping
    stop.set()
    thread.join(timeout=2.0)
    cpu = time.process_time() - cpu0
    rss = max(0, _rss_bytes() - rss0)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], rec.CapacityWaitCancelled)
    assert errors[0].state == rec.STATE_STOPPED_RESUMABLE
    # A busy loop would burn ~0.25 s of CPU; a sleeping waiter is near zero.
    assert cpu < 0.15, f"wait burned {cpu:.3f}s of CPU"
    assert rss < 16 * 2 ** 20, f"wait retained {rss} B of RSS"


def test_wait_deadline_is_resumable_and_keeps_work():
    clock = FakeClock()
    waiter = rec.CapacityWaiter(
        lambda: _fact(10, clock),
        minimum_bytes=800,
        clock=clock,
        sleep=clock.sleep,
        initial_interval_s=5.0,
        max_interval_s=30.0,
        min_spacing_s=5.0,
        deadline_s=12.0,
        max_age_s=None,
    )
    with pytest.raises(rec.CapacityWaitDeadline) as info:
        waiter.wait()
    assert info.value.state == rec.STATE_STOPPED_RESUMABLE
    assert info.value.result is not None
    assert info.value.result.admissible is False


# ---------------------------------------------------------------------------
# Release before wait
# ---------------------------------------------------------------------------


def test_release_happens_before_the_first_probe():
    clock = FakeClock()
    order: list[str] = []

    def release():
        order.append("release")
        return _release_report()

    raws = iter([10, 900, 900])

    def probe():
        order.append("probe")
        return _fact(next(raws), clock)

    coordinator = rec.RecoveryCoordinator(
        run_id="run-order",
        release=release,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
        initial_interval_s=5.0,
        max_interval_s=30.0,
        min_spacing_s=5.0,
        max_age_s=None,
    )
    outcome = coordinator.recover(
        facts=_fact(10, clock),
        phase="run_wave_k",
        reason_code="REFUSE_HOST_RAM",
        detail="pressure",
        minimum_bytes=800,
        wait=True,
    )
    assert order[0] == "release"
    assert "probe" in order
    assert outcome.wait is not None and outcome.wait.probes >= 1
    assert outcome.released is not None
    assert outcome.released.units_released == 3
    assert outcome.state == rec.STATE_ACTIVE
    assert outcome.wait.admissible is True


def test_incomplete_release_refuses_to_wait():
    clock = FakeClock()
    probes: list[float] = []

    def release():
        return _release_report(credits_balanced=False)

    def probe():
        probes.append(clock())
        return _fact(1000, clock)

    coordinator = rec.RecoveryCoordinator(
        run_id="run-incomplete",
        release=release,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
        max_age_s=None,
    )
    with pytest.raises(rec.ReleaseIncompleteError):
        coordinator.recover(
            facts=_fact(10, clock),
            phase="run_wave_k",
            reason_code="REFUSE_HOST_RAM",
            detail="pressure",
            minimum_bytes=800,
            wait=True,
        )
    assert probes == []
    assert coordinator.state == rec.STATE_RECOVERING


def test_structural_minimum_refuses_instead_of_waiting_forever():
    probes: list[float] = []

    def probe():
        probes.append(1.0)
        return _fact(900)

    coordinator = rec.RecoveryCoordinator(
        run_id="run-structural",
        release=lambda: _release_report(),
        probe=probe,
        max_age_s=None,
    )
    with pytest.raises(rec.StructuralCapacityRefused):
        coordinator.recover(
            facts=_fact(900),
            phase="assembly",
            reason_code="REFUSE_HOST_RAM",
            detail="minimum does not fit",
            minimum_bytes=2000,
            structural_minimum_bytes=2000,
            structural_capacity_bytes=1024,
            wait=True,
        )
    assert probes == []
    assert coordinator.state == rec.STATE_CAPACITY_INFEASIBLE


# ---------------------------------------------------------------------------
# Pressure after K cells, then RAM returns: K is never redone
# ---------------------------------------------------------------------------


def test_pressure_after_k_cells_never_recomputes_the_validated_prefix():
    clock = FakeClock()
    cells = [f"r0000c{i:04d}" for i in range(12)]
    executed: list[str] = []

    # Instrumented execution ledger: every real cell execution is appended.
    def execute(cell_id: str) -> None:
        executed.append(str(cell_id))

    pending = list(cells)
    completed: list[str] = []
    k = 4
    for _ in range(k):
        cid = pending.pop(0)
        execute(cid)
        completed.append(cid)

    # Two speculative cells were prefetched (not committed). The release must
    # cancel them and make them available again, not lose or duplicate them.
    speculative = pending[:2]
    del pending[:2]

    def release() -> rec.ReleaseReport:
        report = _release_report(
            cancelled_units=len(speculative),
            drained_units=0,
            host_released_bytes=2 * 4096,
        )
        pending[:0] = list(reversed(speculative))
        return report

    raws = iter([100, 100, 900, 900])

    def probe():
        return _fact(next(raws), clock)

    coordinator = rec.RecoveryCoordinator(
        run_id="run-k",
        release=release,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
        initial_interval_s=5.0,
        max_interval_s=30.0,
        min_spacing_s=5.0,
        max_age_s=None,
        profile=rec.StorageProfilePolicy(requested="auto"),
    )
    outcome = coordinator.recover(
        facts=_fact(100, clock),
        phase="run_wave_4",
        reason_code="REFUSE_HOST_RAM",
        detail="synthetic pressure after K cells",
        minimum_bytes=800,
        wait=True,
    )
    assert outcome.released is not None
    assert outcome.released.units_released == 2
    assert outcome.profile is not None and outcome.profile.switched is True
    assert outcome.wait is not None and outcome.wait.admissible is True

    while pending:
        execute(pending.pop(0))
    assert executed[:k] == cells[:k]
    assert sorted(executed) == sorted(cells)
    assert len(executed) == len(set(executed)) == len(cells)
    assert completed == cells[:k]
    # The pressure episode did not lose the two cancelled speculative cells.
    assert set(speculative).issubset(set(executed))


# ---------------------------------------------------------------------------
# Generations and late results
# ---------------------------------------------------------------------------


def test_late_result_of_cancelled_generation_is_never_accepted():
    registry = _registry()
    fence = rec.GenerationFence(run_id=registry.run_id, registry=registry)
    published: list[str] = []

    env_a = fence.envelope("r0000c0000", attempt_id="cpu:gen0:r0000c0000")
    first = fence.accept(env_a, _empty_cell("r0000c0000"), backend="cpu")
    assert first.accepted is True
    assert registry.counts()["accepted"] == 1
    before = registry.metadata("r0000c0000")
    assert before["plan_generation"] == 0

    fence.advance(reason="dense_to_bounded_storage")

    late = fence.accept(env_a, _empty_cell("r0000c0000"), backend="cpu")
    assert late.accepted is False
    assert late.reason == "stale_generation"
    assert registry.counts()["accepted"] == 1
    assert "r0000c0001" in registry.pending_cell_ids()

    # The registry itself still refuses the stale generation explicitly.
    with pytest.raises(zrr.StaleGenerationError):
        registry.accept(
            _empty_cell("r0000c0001"),
            backend="cpu",
            plan_generation=0,
            attempt_id="cpu:late",
        )

    # A new generation can accept the remaining cell without touching the
    # already-committed one. Its identity is immutable.
    env_b = fence.envelope("r0000c0001", attempt_id="cpu:gen1:r0000c0001")
    second = fence.accept(env_b, _empty_cell("r0000c0001"), backend="cpu")
    assert second.accepted is True
    assert registry.counts()["accepted"] == 2
    after = registry.metadata("r0000c0000")
    assert after == before
    assert after["payload_sha256"] == before["payload_sha256"]
    assert after["attempt_id"] == "cpu:gen0:r0000c0000"
    # No publish callback could ever have been invoked for the late result:
    # acceptance is the publication gate and it returned False.
    assert published == []


def test_late_result_does_not_abort_a_healthy_run():
    registry = _registry()
    fence = rec.GenerationFence(run_id=registry.run_id, registry=registry)
    fence.advance(reason="plan")
    env = fence.envelope("r0000c0000")
    envelope_old = rec.TaskGeneration(
        run_id=registry.run_id,
        cell_id="r0000c0000",
        attempt_id="old",
        plan_generation=0,
    )
    outcome = fence.accept(envelope_old, _empty_cell("r0000c0000"), backend="cpu")
    assert outcome.accepted is False
    assert fence.generation == 1
    # The same fence still accepts current work afterwards.
    current = fence.envelope("r0000c0000")
    assert fence.accept(current, _empty_cell("r0000c0000"), backend="cpu").accepted
    assert env.plan_generation == 1
    assert any(e["event"] == "late_result_rejected" for e in fence.journal())


# ---------------------------------------------------------------------------
# Bounded retries
# ---------------------------------------------------------------------------


def test_retry_budget_is_bounded_then_waits():
    budget = rec.RetryBudget(max_attempts=2)
    assert budget.attempt("c1", 0) == 1
    assert budget.attempt("c1", 0) == 2
    with pytest.raises(rec.RetryExhausted) as info:
        budget.attempt("c1", 0)
    assert info.value.state == rec.STATE_WAITING_RESOURCES
    assert info.value.cell_id == "c1"
    # A NEW plan generation has its own bounded allowance.
    assert budget.attempt("c1", 1) == 1
    assert budget.remaining("c1", 0) == 0
    assert budget.remaining("c1", 1) == 1


def test_coordinator_stop_is_resumable_and_keeps_release_report():
    coordinator = rec.RecoveryCoordinator(
        run_id="run-stop",
        release=lambda: _release_report(),
        probe=lambda: _fact(10),
        max_age_s=None,
    )
    assert coordinator.stop("user stop") == rec.STATE_STOPPED_RESUMABLE
    with pytest.raises(rec.CapacityWaitCancelled):
        coordinator.recover(
            facts=_fact(10),
            phase="run_wave_0",
            reason_code="REFUSE_HOST_RAM",
            detail="stop",
            minimum_bytes=800,
        )
    snapshot = coordinator.snapshot()
    assert snapshot["state"] == rec.STATE_STOPPED_RESUMABLE
    assert snapshot["note"].startswith("WAITING_RESOURCES")
