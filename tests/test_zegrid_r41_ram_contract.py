"""ZM-ZEGRID-R41 Stage 1 — host-RAM contract + admission controller tests.

Scope: :mod:`zemosaic.core.zegrid.ram_contract` ONLY. No product call site is
touched. These tests pin the exact integer budgets, the immutable/auditable
records, the bounded admission loop between the 80 % plan budget and the 90 %
admission ceiling, and the no-growth replanning invariants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import ram_contract as rc  # noqa: E402

GB = 2 ** 30
MIB = 2 ** 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(
    *,
    aggregate: int,
    stack: int | None = None,
    cache: int | None = None,
    raw: int = 20 * GB,
    factor=(rc.PLAN_FACTOR_NUM, rc.PLAN_FACTOR_DEN),
    budget: int | None = None,
    phase: str = rc.PHASE_PLAN,
    layout=(6, 4),
    rank: int = 24,
    cpu_tile: int = 128,
    gpu_tile=None,
    backend: str = "cpu",
    k: int = 1,
    w: int = 1,
    shared: int = 0,
    pinned: bool = False,
    reason: str = "test",
):
    if budget is None:
        budget = (factor[0] * raw) // factor[1]
    return rc.PlanEnvelope(
        phase=phase,
        raw_available_bytes=int(raw),
        factor_num=int(factor[0]),
        factor_den=int(factor[1]),
        budget_bytes=int(budget),
        layout=tuple(layout),
        refinement_rank=int(rank),
        cpu_tile=int(cpu_tile),
        gpu_tile=gpu_tile,
        backend=backend,
        cells_in_flight=int(k),
        cache_workers_per_cell=int(w),
        shared_bytes=int(shared),
        stack_total_bytes=int(aggregate if stack is None else stack),
        cache_total_bytes=int(aggregate if cache is None else cache),
        aggregate_bytes=int(aggregate),
        pinned=bool(pinned),
        reason=reason,
    )


class _Meter:
    """``measure_available`` callback returning a scripted/constant value."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


# ---------------------------------------------------------------------------
# 1. Exact budgets: integer arithmetic, applied once
# ---------------------------------------------------------------------------

def test_budget_formulas_exact_integer():
    assert rc.plan_budget_bytes(100) == (4 * 100) // 5 == 80
    assert rc.admit_budget_bytes(100) == (9 * 100) // 10 == 90
    # odd / non-divisible values floor (never round up)
    assert rc.plan_budget_bytes(101) == 80  # 404 // 5
    assert rc.admit_budget_bytes(101) == 90  # 909 // 10
    assert rc.plan_budget_bytes(1) == 0
    assert rc.admit_budget_bytes(1) == 0


def test_budget_formulas_no_float_precision_on_huge_integers():
    huge = 10 ** 40 + 7
    assert rc.plan_budget_bytes(huge) == (4 * huge) // 5
    assert rc.admit_budget_bytes(huge) == (9 * huge) // 10
    # integer arithmetic must NOT silently degrade to float
    assert rc.plan_budget_bytes(huge) != int((4 * huge) / 5) or (4 * huge) % 5 == 0
    # a value where float64 and int disagree
    tricky = 2 ** 61 + 1
    assert rc.plan_budget_bytes(tricky) == (4 * tricky) // 5
    assert rc.admit_budget_bytes(tricky) == (9 * tricky) // 10


def test_each_factor_applied_once_to_its_own_raw_observation():
    plan_obs = rc.Observation.plan(20 * GB)
    admit_obs = rc.Observation.admit(18 * GB)
    assert plan_obs.budget_bytes == 16 * GB
    assert admit_obs.budget_bytes == (9 * 18 * GB) // 10  # 16.2 GiB, exact int
    # NEVER 0.9 * B_plan
    assert admit_obs.budget_bytes != (9 * plan_obs.budget_bytes) // 10


@pytest.mark.parametrize("bad", [None, 0, -1, -10 ** 9, True, False, 1.5, "100"])
def test_invalid_observations_fail_closed(bad):
    with pytest.raises(rc.InvalidObservation):
        rc.plan_budget_bytes(bad)
    with pytest.raises(rc.InvalidObservation):
        rc.admit_budget_bytes(bad)
    with pytest.raises(rc.InvalidObservation):
        rc.Observation.plan(bad)
    with pytest.raises(rc.InvalidObservation):
        rc.Observation.admit(bad)


def test_none_is_invalid_observation_but_planner_still_allows_unbounded():
    # None is rejected by the CONTRACT ...
    with pytest.raises(rc.InvalidObservation):
        rc.plan_budget_bytes(None)
    # ... but remains a valid 'unbounded' sentinel in the low-level planner API.
    from zemosaic.core.zegrid import parallel as zpar

    plan = zpar.plan_cell_concurrency([(GB, GB)], 8, None)
    assert plan["cells_in_flight"] >= 1


# ---------------------------------------------------------------------------
# 2. Immutable / auditable records + JSON safety
# ---------------------------------------------------------------------------

def test_records_are_frozen_and_json_safe():
    obs = rc.Observation.admit(18 * GB)
    env = _env(aggregate=16 * GB, raw=20 * GB)
    with pytest.raises(Exception):
        obs.raw_available_bytes = 1  # frozen
    with pytest.raises(Exception):
        env.aggregate_bytes = 1  # frozen

    # JSON-safe: tuples become lists, ints stay ints.
    payload = {"observation": obs.to_dict(), "envelope": env.to_dict()}
    text = json.dumps(payload)
    back = json.loads(text)
    assert back["envelope"]["layout"] == [6, 4]
    assert back["envelope"]["aggregate_bytes"] == 16 * GB
    assert isinstance(back["observation"]["budget_bytes"], int)
    assert back["envelope"]["factor_num"] == 4 and back["envelope"]["factor_den"] == 5
    assert "factor" not in back["envelope"]  # L4: no serialized float factor
    assert "factor" not in back["observation"]


def test_observation_records_phase_factor_budget():
    o = rc.Observation.admit(1000)
    assert o.phase == rc.PHASE_ADMIT
    assert (o.factor_num, o.factor_den) == (9, 10)
    assert o.raw_available_bytes == 1000 and o.budget_bytes == 900
    p = rc.Observation.plan(1000)
    assert p.phase == rc.PHASE_PLAN and p.budget_bytes == 800


# ---------------------------------------------------------------------------
# 3. Admission controller: the 20 -> 18 / 17 GiB contract
# ---------------------------------------------------------------------------

def test_20gib_plan_admitted_unchanged_at_18gib_with_tolerance():
    plan = _env(aggregate=16 * GB, raw=20 * GB)
    meter = _Meter(18 * GB)

    def rebuild(b_plan, previous):  # pragma: no cover - must not be called
        raise AssertionError("unchanged plan must pass without a rebuild")

    out = rc.AdmissionController().run(plan, meter, rebuild)
    assert out.plan is plan
    assert out.replans == 0
    assert out.tolerance is True  # 14.4 < 16 <= 16.2 GiB
    assert out.final_observation.raw_available_bytes == 18 * GB
    assert meter.calls == 2  # fresh admission observation + fresh launch observation


def test_20gib_plan_misses_at_17gib_and_rebuilds_at_13_6_never_15_3():
    plan = _env(aggregate=16 * GB, raw=20 * GB)
    meter = _Meter(17 * GB)
    seen = []

    def rebuild(b_plan, previous):
        seen.append(b_plan)
        return _env(aggregate=b_plan, raw=17 * GB)  # fits its own B_plan exactly

    out = rc.AdmissionController().run(plan, meter, rebuild)
    assert seen == [(4 * 17 * GB) // 5]  # 13.6 GiB exactly
    assert seen[0] == (4 * 17 * GB) // 5
    assert seen[0] != (9 * 17 * GB) // 10  # NEVER the admission ceiling
    assert out.replans == 1
    assert out.tolerance is False  # 13.6 <= 13.6 -> no slack used
    assert meter.calls == 3
    assert out.history[-1].aggregate_bytes == seen[0]


def test_tolerance_true_false_and_exact_boundaries():
    # raw=1000 -> B_plan=800, B_admit=900.
    # tolerance TRUE iff 800 < aggregate <= 900.
    plan = _env(aggregate=900, raw=1000)
    out = rc.AdmissionController().run(plan, _Meter(1000), lambda *a: None)
    assert out.tolerance is True  # exact ceiling accepted
    assert out.replans == 0

    plan2 = _env(aggregate=800, raw=1000)
    out2 = rc.AdmissionController().run(plan2, _Meter(1000), lambda *a: None)
    assert out2.tolerance is False  # exactly the plan budget -> no slack


def test_one_byte_over_ceiling_triggers_replan():
    plan = _env(aggregate=901, raw=1000)  # one byte over B_admit=900
    meter = _Meter(1000)
    seen = []

    def rebuild(b_plan, previous):
        seen.append(b_plan)
        return _env(aggregate=b_plan, raw=1000)

    out = rc.AdmissionController().run(plan, meter, rebuild)
    assert seen == [800]  # rebuilt under the PLAN budget from the same raw obs
    assert out.plan.aggregate_bytes == 800
    assert out.replans == 1


def test_exact_ceiling_passes_without_replan():
    plan = _env(aggregate=900, raw=1000)
    calls = _Meter(1000)
    rc.AdmissionController().run(plan, calls, lambda *a: (_ for _ in ()).throw(
        AssertionError("no rebuild expected")
    ))
    assert calls.calls == 2


def test_fresh_observation_before_every_attempt():
    plan = _env(aggregate=16 * GB, raw=20 * GB)
    meter = _Meter(17 * GB)
    n = {"rebuild": 0}

    def rebuild(b_plan, previous):
        n["rebuild"] += 1
        return _env(aggregate=b_plan, raw=17 * GB)

    out = rc.AdmissionController().run(plan, meter, rebuild)
    # 1 miss observation + 1 admission observation + 1 launch observation; the
    # rebuild reuses the MISS observation's raw (not a 4th measurement).
    assert meter.calls == 3
    assert n["rebuild"] == 1
    # every admission attempt was preceded by its own fresh observation
    phases = [o.phase for o in out.observations]
    assert phases[:1] == [rc.PHASE_ADMIT]


def test_max_two_replans_then_quantitative_exception():
    # RAM keeps FALLING: each valid rebuild fits its own B_plan, but the next
    # observation's ceiling is lower still, so the plan misses again.
    plan = _env(aggregate=901, raw=1000)
    raws = iter([1000, 800, 600, 500])
    state = {"raw": 1000}

    def measure():
        state["raw"] = next(raws)
        return state["raw"]

    n = {"rebuild": 0}

    def rebuild(b_plan, previous):
        n["rebuild"] += 1
        raw = state["raw"]  # the raw observation that caused THIS miss
        return _env(aggregate=b_plan, raw=raw, budget=b_plan)

    with pytest.raises(rc.AdmissionExhausted) as exc:
        rc.AdmissionController().run(plan, measure, rebuild)
    assert n["rebuild"] == 2
    assert len(exc.value.attempts) >= 3
    assert "B_admit" in str(exc.value)


def test_l1_launch_observation_is_tagged_launch():
    plan = _env(aggregate=16 * GB, raw=20 * GB)
    out = rc.AdmissionController().run(plan, _Meter(18 * GB), lambda *a: None)
    assert out.final_observation.phase == rc.PHASE_LAUNCH
    assert out.final_observation.factor_num == 9
    assert out.final_observation.factor_den == 10
    assert [o.phase for o in out.observations] == [rc.PHASE_ADMIT, rc.PHASE_LAUNCH]
    assert rc.Observation.launch(1000).budget_bytes == 900


def test_zero_replan_budget_refuses_immediately():
    plan = _env(aggregate=901, raw=1000)
    with pytest.raises(rc.AdmissionExhausted):
        rc.AdmissionController(max_replans=0).run(plan, _Meter(1000), lambda *a: None)


@pytest.mark.parametrize("bad", [0, -5])
def test_zero_or_negative_live_observation_fails_closed_in_controller(bad):
    plan = _env(aggregate=10 * GB, raw=20 * GB)
    with pytest.raises(rc.InvalidObservation):
        rc.AdmissionController().run(plan, _Meter(bad), lambda *a: None)


def test_controller_rejects_growth_blown_by_a_late_observation():
    # Later observation finds MORE RAM, but the rebuild tries to coarsen -> reject.
    plan = _env(aggregate=901, raw=1000, rank=24, cpu_tile=128, k=2)

    def rebuild(b_plan, previous):
        return _env(aggregate=b_plan, raw=1000, rank=6, cpu_tile=256, k=4)

    with pytest.raises(rc.GrowthRejected):
        rc.AdmissionController().run(plan, _Meter(1000), rebuild)


def test_history_preserves_every_observation_and_replan():
    plan = _env(aggregate=16 * GB, raw=20 * GB)
    meter = _Meter(17 * GB)

    def rebuild(b_plan, previous):
        return _env(aggregate=b_plan, raw=17 * GB)

    out = rc.AdmissionController().run(plan, meter, rebuild)
    assert len(out.history) == 1 + out.replans
    assert out.history[0] is plan
    assert any(o.phase == rc.PHASE_PLAN for o in out.observations)
    assert any(o.phase == rc.PHASE_ADMIT for o in out.observations)
    assert json.dumps(out.to_dict())  # whole outcome is JSON-serialisable


# ---------------------------------------------------------------------------
# 4. No-growth guard
# ---------------------------------------------------------------------------

def test_no_growth_accepts_identical_and_finer():
    prev = _env(aggregate=16 * GB, rank=24, cpu_tile=128, k=2)
    rc.assert_no_growth(prev, _env(aggregate=10 * GB, rank=24, cpu_tile=128, k=2))
    rc.assert_no_growth(prev, _env(aggregate=10 * GB, rank=24, cpu_tile=64, k=1))  # finer/smaller


def test_no_growth_rejects_coarsening():
    prev = _env(aggregate=16 * GB, rank=24)
    with pytest.raises(rc.GrowthRejected, match="refinement rank"):
        rc.assert_no_growth(prev, _env(aggregate=10 * GB, rank=12))


def test_no_growth_rejects_tile_growth():
    prev = _env(aggregate=16 * GB, cpu_tile=128, gpu_tile=112)
    with pytest.raises(rc.GrowthRejected, match="cpu tile"):
        rc.assert_no_growth(prev, _env(aggregate=10 * GB, cpu_tile=256))
    with pytest.raises(rc.GrowthRejected, match="gpu tile"):
        rc.assert_no_growth(prev, _env(aggregate=10 * GB, gpu_tile=128))


def test_no_growth_rejects_concurrency_growth():
    prev = _env(aggregate=16 * GB, k=2)
    with pytest.raises(rc.GrowthRejected, match="concurrency increased"):
        rc.assert_no_growth(prev, _env(aggregate=10 * GB, k=4))


def test_no_growth_rejects_pinned_geometry_mutation():
    prev = _env(aggregate=16 * GB, pinned=True, layout=(6, 4))
    with pytest.raises(rc.GrowthRejected, match="pinned layout identity"):
        rc.assert_no_growth(prev, _env(aggregate=10 * GB, pinned=True, layout=(8, 6)))
    with pytest.raises(rc.GrowthRejected, match="pinned flag"):
        rc.assert_no_growth(prev, _env(aggregate=10 * GB, pinned=False, layout=(6, 4)))


def test_tolerance_flag_is_purely_descriptive():
    # exactly B_admit -> admitted with the flag; one above -> NOT admitted.
    assert rc.requires_tolerance(900, 1000) is True
    assert rc.requires_tolerance(901, 1000) is False
    assert rc.requires_tolerance(800, 1000) is False


# ---------------------------------------------------------------------------
# 5. Phase accounting math (integer)
# ---------------------------------------------------------------------------

def test_stack_and_cache_totals_count_shared_once():
    assert rc.stack_phase_total_bytes(400, 2, 100) == 600
    cpc = rc.cache_per_cell_bytes(100, 2, 150)
    assert cpc == 200  # 100 + 2*(150-100)
    assert rc.cache_phase_total_bytes(400, 2, cpc) == 400 + 2 * 200
    assert rc.aggregate_phase_bytes(600, 800) == 800


def test_cache_per_cell_floors_negative_delta_at_zero():
    assert rc.cache_per_cell_bytes(100, 3, 40) == 100  # delta floored to 0


def test_feasible_concurrency_reduces_k_and_refuses_one_cell():
    # shared=400, bound=100 -> K=2 fits 600 exactly; K=1 refusal when bound alone
    k, st, ct, agg, cpc, w = rc.feasible_concurrency(
        budget_bytes=600, shared_bytes=400, stack_per_cell_bound_bytes=100,
        process_baseline_bytes=10, cache_worker_footprint_bytes=10,
        cache_workers_for_k=lambda _k: 1, max_k=5,
    )
    assert (k, st, agg) == (2, 600, 600)
    zero = rc.feasible_concurrency(
        budget_bytes=50, shared_bytes=0, stack_per_cell_bound_bytes=100,
        process_baseline_bytes=10, cache_worker_footprint_bytes=10,
        cache_workers_for_k=lambda _k: 1, max_k=5,
    )
    assert zero[0] == 0 and zero[3] is None  # ONE-CELL REFUSAL, no clamp-to-1


# ---------------------------------------------------------------------------
# 6. H1 — the controller ENFORCES the rebuilt contract (never trusts the callback)
# ---------------------------------------------------------------------------
def _lying_run(candidate_factory):
    # initial aggregate 901 misses B_admit=900 -> a rebuild is requested with
    # B_plan=800 from the SAME raw 1000.
    initial = _env(aggregate=901, raw=1000)
    meter = _Meter(1000)

    def rebuild(b_plan, previous):
        return candidate_factory(b_plan)

    return rc.AdmissionController().run(initial, meter, rebuild)


def test_h1_accepts_a_truthful_rebuild():
    out = _lying_run(lambda b: _env(aggregate=b, raw=1000, budget=b))
    assert out.plan.aggregate_bytes == 800
    assert out.replans == 1


def test_h1_rejects_aggregate_over_b_plan_but_within_admit():
    # B_plan=800, B_admit=900: 801 is inside the admission ceiling but OVER the
    # plan budget -> must FAIL, not bypass the 80 % budget.
    with pytest.raises(rc.InvalidEnvelope, match="B_plan"):
        _lying_run(lambda b: _env(aggregate=b + 1, raw=1000, budget=b))


def test_h1_rejects_aggregate_inconsistent_with_stack_cache():
    with pytest.raises(rc.InvalidEnvelope, match="aggregate_bytes"):
        _lying_run(lambda b: _env(aggregate=0, stack=20 * GB, cache=20 * GB, raw=1000, budget=b))


def test_h1_rejects_wrong_raw():
    with pytest.raises(rc.InvalidEnvelope, match="raw"):
        # raw=1001 still floors to budget 800, so the mismatch is the RAW itself
        _lying_run(lambda b: _env(aggregate=b, raw=1001, budget=b))


def test_h1_rejects_wrong_budget():
    with pytest.raises(rc.InvalidEnvelope):
        _lying_run(lambda b: _env(aggregate=b, raw=1000, budget=b + 1))


def test_h1_rejects_wrong_factor():
    with pytest.raises(rc.InvalidEnvelope, match="factor"):
        _lying_run(
            lambda b: _env(aggregate=b, raw=1000, factor=(9, 10), budget=(9 * 1000) // 10)
        )


def test_h1_rejects_wrong_phase():
    with pytest.raises(rc.InvalidEnvelope, match="phase"):
        _lying_run(lambda b: _env(aggregate=b, raw=1000, budget=b, phase=rc.PHASE_ADMIT))


def test_h1_mismatch_raises_before_any_later_admission():
    # the controller must not silently fall through to a 90 % acceptance
    calls = []

    def rebuild(b_plan, previous):
        calls.append(b_plan)
        return _env(aggregate=b_plan + 1, raw=1000, budget=b_plan)

    with pytest.raises(rc.InvalidEnvelope):
        rc.AdmissionController().run(_env(aggregate=901, raw=1000), _Meter(1000), rebuild)
    assert calls == [800]


# ---------------------------------------------------------------------------
# 7. M1 — fail closed on unsafe numeric inputs
# ---------------------------------------------------------------------------

def test_m1_phase_math_helpers_reject_negative_and_bool():
    for call in (
        lambda: rc.stack_phase_total_bytes(-1, 2, 3),
        lambda: rc.stack_phase_total_bytes(0, True, 3),
        lambda: rc.stack_phase_total_bytes(0, 2, -3),
        lambda: rc.cache_per_cell_bytes(-1, 2, 3),
        lambda: rc.cache_per_cell_bytes(0, -2, 3),
        lambda: rc.cache_phase_total_bytes(-1, 2, 3),
        lambda: rc.aggregate_phase_bytes(-1, 3),
        lambda: rc.aggregate_phase_bytes(True, 3),
    ):
        with pytest.raises(rc.RamContractError):
            call()


def test_m1_feasible_concurrency_rejects_negative_budget_and_callback():
    base = dict(
        shared_bytes=0, stack_per_cell_bound_bytes=1, process_baseline_bytes=1,
        cache_worker_footprint_bytes=1, cache_workers_for_k=lambda _k: 1, max_k=2,
    )
    with pytest.raises(rc.RamContractError):
        rc.feasible_concurrency(budget_bytes=-1, **base)
    with pytest.raises(rc.RamContractError):
        rc.feasible_concurrency(budget_bytes=True, **base)
    with pytest.raises(rc.RamContractError):
        rc.feasible_concurrency(budget_bytes=100, **{**base, "shared_bytes": -1})
    with pytest.raises(rc.RamContractError, match="cache_workers_for_k"):
        rc.feasible_concurrency(
            budget_bytes=100, shared_bytes=0, stack_per_cell_bound_bytes=1,
            process_baseline_bytes=1, cache_worker_footprint_bytes=1,
            cache_workers_for_k=lambda _k: -1, max_k=3,
        )


def test_m1_zero_finite_budget_is_not_unbounded():
    kw = dict(
        shared_bytes=0, stack_per_cell_bound_bytes=1, process_baseline_bytes=1,
        cache_worker_footprint_bytes=1, cache_workers_for_k=lambda _k: 1, max_k=3,
    )
    assert rc.feasible_concurrency(budget_bytes=0, **kw)[0] == 0  # finite 0 refuses
    assert rc.feasible_concurrency(budget_bytes=None, **kw)[0] == 3  # None unbounded
    # finite 0 is coherent ONLY when the aggregate is genuinely 0
    assert rc.feasible_concurrency(
        budget_bytes=0, shared_bytes=0, stack_per_cell_bound_bytes=0,
        process_baseline_bytes=0, cache_worker_footprint_bytes=0,
        cache_workers_for_k=lambda _k: 0, max_k=3,
    )[0] == 3


def test_validate_envelope_structural_rejections():
    rc.validate_envelope(_env(aggregate=16 * GB, raw=20 * GB))  # control
    bad_layouts = [(6,), (6, 4, 2), (0, 4), (6, -4), (6, "4")]
    for layout in bad_layouts:
        with pytest.raises(rc.RamContractError):
            rc.validate_envelope(_env(aggregate=16 * GB, raw=20 * GB, layout=layout))
    with pytest.raises(rc.RamContractError):
        rc.validate_envelope(_env(aggregate=16 * GB, raw=20 * GB, backend="quantum"))
    with pytest.raises(rc.RamContractError):
        rc.validate_envelope(_env(aggregate=16 * GB, raw=20 * GB, rank=-1))
    with pytest.raises(rc.RamContractError):
        rc.validate_envelope(_env(aggregate=16 * GB, raw=20 * GB, shared=17 * GB))


# ---------------------------------------------------------------------------
# 8. L2 — monotonic backend / GPU semantics
# ---------------------------------------------------------------------------

def test_l2_backend_transitions():
    prev = _env(aggregate=16 * GB, backend=rc.BACKEND_GPU, gpu_tile=112)
    # gpu -> cpu degradation is ALLOWED (future fallback stays possible)
    rc.assert_no_growth(prev, _env(aggregate=10 * GB, backend=rc.BACKEND_CPU, gpu_tile=None))
    with pytest.raises(rc.GrowthRejected, match="cpu -> gpu"):
        rc.assert_no_growth(
            _env(aggregate=16 * GB, backend=rc.BACKEND_CPU),
            _env(aggregate=10 * GB, backend=rc.BACKEND_GPU, gpu_tile=112),
        )
    with pytest.raises(rc.GrowthRejected, match="unknown backend"):
        rc.assert_no_growth(
            _env(aggregate=16 * GB, backend="quantum"),
            _env(aggregate=10 * GB, backend="quantum"),
        )


def test_l2_gpu_tile_none_transitions():
    with pytest.raises(rc.GrowthRejected, match="gpu tile introduced"):
        rc.assert_no_growth(
            _env(aggregate=16 * GB, gpu_tile=None), _env(aggregate=10 * GB, gpu_tile=112)
        )
    # value -> None is a degradation and is allowed
    rc.assert_no_growth(
        _env(aggregate=16 * GB, gpu_tile=112), _env(aggregate=10 * GB, gpu_tile=None)
    )


# ---------------------------------------------------------------------------
# 9. L3 — single process-baseline source of truth
# ---------------------------------------------------------------------------

def test_l3_single_process_baseline_source():
    from zemosaic.core.zegrid import parallel as zpar

    assert zpar._PER_WORKER_BASELINE_BYTES == rc.PROCESS_BASELINE_BYTES
    assert rc.PROCESS_BASELINE_BYTES == 340 * 1024 * 1024
