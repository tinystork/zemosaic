"""ZM-ZEGRID-R41 Stage 1 — opt-in phase accounting in ``parallel.py``.

Scope: the OPTIONAL kwargs added to :func:`zemosaic.core.zegrid.parallel.
plan_cell_concurrency`. The default (no phase kwargs) path must be byte-for-byte
unchanged; the opt-in path accounts for the SIMULTANEOUS stack/cache host totals
of the whole in-flight batch. No product call site is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

GB = 2 ** 30
MIB = 2 ** 20
PB = zpar._PER_WORKER_BASELINE_BYTES


def _mib(x: float) -> int:
    return int(round(x * MIB))


# ---------------------------------------------------------------------------
# 1. Default path compatibility (byte-for-byte; NO new keys)
# ---------------------------------------------------------------------------

def test_default_path_has_no_phase_keys_and_matches_r20_fixture():
    plan = zpar.plan_cell_concurrency(
        [(10849326220, 2449628160)] * 6, 16, 12184610406, cache_build_workers=14,
    )
    assert plan["mode"] == "stream"
    assert plan["cells_in_flight"] == 4
    assert plan["cache_workers_per_cell"] == 4
    for banned in (
        "phase_accounting", "shared_bytes", "stack_total_bytes",
        "cache_total_bytes", "aggregate_bytes", "cache_per_cell_bytes",
    ):
        assert banned not in plan


def test_default_planner_expectations_still_hold():
    assert zpar.__dict__  # module importable
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency([(3 * GB, 2 * GB)] * 3, 8, 1 * GB)
    plan = zpar.plan_cell_concurrency([(10 * GB, GB // 2)] * 4, 16, 2 * GB)
    assert plan["mode"] == "stream" and "INFEASIBLE" in plan["chosen_reason"]
    # no-budget historical behaviour
    p0 = zpar.plan_cell_concurrency([(10 * GB, 5 * GB)] * 4, 16, None)
    assert all(c["feasible"] for c in p0["candidates"])


def test_empty_input_schema_coherent_default_and_optin():
    empty = zpar.plan_cell_concurrency([], 8, 10 * GB)
    assert empty["n_cells"] == 0 and empty["candidates"] == []
    optin = zpar.plan_cell_concurrency(
        [], 8, 10 * GB, shared_bytes=5 * GB, cache_worker_footprint_bytes=PB,
    )
    assert optin["n_cells"] == 0 and optin["cells_in_flight"] == 0
    assert optin["aggregate_bytes"] == 5 * GB
    assert optin["phase_accounting"] is True


# ---------------------------------------------------------------------------
# 2. Opt-in activates only when a phase kwarg is supplied
# ---------------------------------------------------------------------------

def test_optin_via_each_trigger():
    common = dict(cell=[(10 * GB, 10 * GB)] * 4, cpu=16, budget=100 * GB)
    for kwargs in (
        {"shared_bytes": 0},
        {"shared_bytes": 4 * GB},
        {"cache_worker_footprint_bytes": PB},
        {"cache_worker_px": 512 * 512},
        {"process_baseline_bytes": 123 * MIB},
    ):
        plan = zpar.plan_cell_concurrency(
            common["cell"], common["cpu"], common["budget"], **kwargs
        )
        assert plan["phase_accounting"] is True
        assert "aggregate_bytes" in plan


def test_derived_cache_worker_footprint_matches_helper():
    px = 700 * 700
    plan = zpar.plan_cell_concurrency(
        [(10 * GB, 10 * GB)] * 2, 8, 100 * GB, cache_worker_px=px,
    )
    assert plan["cache_worker_footprint_bytes"] == zpar.cache_footprint_bytes(px)
    assert plan["cache_worker_footprint_source"].startswith("derived")


# ---------------------------------------------------------------------------
# 3. Simultaneous phase aggregate drives K
# ---------------------------------------------------------------------------

def test_shared_counted_once():
    # shared=40 GiB, per-cell stack=10 GiB, budget=60 GiB -> K=2 (60<=60), K=3=70>60
    plan = zpar.plan_cell_concurrency(
        [(10 * GB, 10 * GB)] * 6, 16, 60 * GB,
        shared_bytes=40 * GB, cache_worker_footprint_bytes=PB,
    )
    assert plan["mode"] == "inmem"
    assert plan["cells_in_flight"] == 2
    assert plan["stack_total_bytes"] == 60 * GB  # shared counted ONCE, not K times
    assert plan["aggregate_bytes"] <= 60 * GB


def test_each_cell_fits_but_phase_aggregate_does_not_reduces_k():
    # Each cell is only 1 GiB, but K of them must fit the aggregate budget.
    plan = zpar.plan_cell_concurrency(
        [(GB, GB)] * 8, 16, 2 * GB,
        shared_bytes=0, cache_worker_footprint_bytes=PB,
    )
    assert plan["cells_in_flight"] == 2  # 2*1 GiB fits; 3 GiB would not
    assert plan["stack_total_bytes"] == 2 * GB
    assert plan["aggregate_bytes"] == 2 * GB


def test_concurrency_reduction_is_real():
    tight = zpar.plan_cell_concurrency(
        [(GB, GB)] * 8, 16, 3 * GB, shared_bytes=0, cache_worker_footprint_bytes=PB,
    )
    loose = zpar.plan_cell_concurrency(
        [(GB, GB)] * 8, 16, 8 * GB, shared_bytes=0, cache_worker_footprint_bytes=PB,
    )
    assert tight["cells_in_flight"] == 3
    assert loose["cells_in_flight"] >= tight["cells_in_flight"]


def test_one_cell_refusal_never_clamps_to_one():
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency(
            [(2 * GB, 3 * GB)], 8, GB,  # even one cell exceeds the 1 GiB budget
            shared_bytes=0, cache_worker_footprint_bytes=PB,
        )


# ---------------------------------------------------------------------------
# 4. Cache footprint: process baseline retained (no r2 undercount)
# ---------------------------------------------------------------------------

def test_cache_per_cell_retains_one_process_baseline_per_active_cell():
    delta = 100
    plan = zpar.plan_cell_concurrency(
        [(GB, GB)] * 6, 16, 100 * GB,
        shared_bytes=2 * GB, cache_worker_footprint_bytes=PB + delta,
    )
    k = plan["cells_in_flight"]
    w = plan["cache_workers_per_cell"]
    assert plan["cache_per_cell_bytes"] == PB + w * delta
    assert plan["cache_total_bytes"] == 2 * GB + k * (PB + w * delta)
    # the DISCARDED r2 undercount omitted one process baseline per active cell
    r2_undercount = 2 * GB + k * w * delta
    assert plan["cache_total_bytes"] != r2_undercount
    assert plan["cache_total_bytes"] - r2_undercount == k * PB


def test_default_cache_footprint_used_when_none_supplied():
    plan = zpar.plan_cell_concurrency(
        [(GB, GB)] * 2, 8, 100 * GB, shared_bytes=0,
    )
    assert plan["cache_worker_footprint_bytes"] == zpar._CACHE_PER_WORKER_FOOTPRINT_BYTES
    assert plan["cache_worker_footprint_source"].startswith("default")


# ---------------------------------------------------------------------------
# 5. GPU host accounting follows the same phase math
# ---------------------------------------------------------------------------

def test_gpu_single_owner_uses_phase_aggregate():
    plan = zpar.plan_cell_concurrency(
        [(10 * GB, 2 * GB)] * 4, 16, 20 * GB, gpu_backend=True,
        shared_bytes=3 * GB, cache_worker_footprint_bytes=PB,
    )
    assert plan["mode"] == "stream"
    assert plan["cells_in_flight"] == 1  # single device owner
    assert plan["gpu_serialized"] is True
    assert plan["stack_total_bytes"] == 3 * GB + 2 * GB
    assert plan["aggregate_bytes"] == max(
        plan["stack_total_bytes"], plan["cache_total_bytes"]
    )
    assert plan["aggregate_bytes"] <= 20 * GB


def test_gpu_refuses_when_phase_aggregate_exceeds_budget():
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency(
            [(10 * GB, 9 * GB)] * 2, 16, 10 * GB, gpu_backend=True,
            shared_bytes=5 * GB, cache_worker_footprint_bytes=PB,
        )


# ---------------------------------------------------------------------------
# 6. Recorded Caldwell deterministic cases (shared/cache disabled)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "label,inmem_mib,stream_mib,budget_mib,should_pass",
    [
        ("5x3", 136220.8, 17229.4, 15574.8, False),
        ("4x3", 164395.4, 18655.4, 17836.1, False),
        ("6x4", 78995.6, 11224.7, 15574.8, True),
        ("6x4@17836", 78995.6, 11224.7, 17836.1, True),
    ],
)
def test_caldwell_recorded_cases(label, inmem_mib, stream_mib, budget_mib, should_pass):
    """Planner evidence (NOT a real-stack claim): with shared=0 and the cache
    increment disabled, the planner reproduces the probe refusals/pass."""
    cells = 15 if label == "5x3" else (12 if label == "4x3" else 24)
    pairs = [(_mib(inmem_mib), _mib(stream_mib))] * cells
    budget = _mib(budget_mib)
    if should_pass:
        plan = zpar.plan_cell_concurrency(
            pairs, 16, budget, shared_bytes=0, cache_worker_footprint_bytes=PB,
        )
        assert plan["mode"] == "stream"
        assert plan["aggregate_bytes"] <= budget
        assert plan["cells_in_flight"] == 1  # a second cell would bust the budget
    else:
        with pytest.raises(zpar.ModeInfeasible):
            zpar.plan_cell_concurrency(
                pairs, 16, budget, shared_bytes=0, cache_worker_footprint_bytes=PB,
            )


# ---------------------------------------------------------------------------
# 7. Unbounded opt-in and result JSON-safety
# ---------------------------------------------------------------------------

def test_optin_unbounded_budget_keeps_max_concurrency():
    plan = zpar.plan_cell_concurrency(
        [(GB, GB)] * 6, 16, None, shared_bytes=1 * GB, cache_worker_footprint_bytes=PB,
    )
    assert plan["cells_in_flight"] == min(14, 6)
    assert plan["feasible"] is True


def test_optin_result_is_json_serialisable():
    import json

    plan = zpar.plan_cell_concurrency(
        [(GB, GB)] * 6, 16, 20 * GB, shared_bytes=GB, cache_worker_px=512 * 512,
    )
    assert json.loads(json.dumps(plan))["aggregate_bytes"] == plan["aggregate_bytes"]


# ---------------------------------------------------------------------------
# 8. M1 — unbounded is ONLY None; finite 0 refuses; unsafe inputs fail closed
# ---------------------------------------------------------------------------

def test_optin_zero_finite_budget_refuses_but_none_is_unbounded():
    # finite 0 is a REAL empty budget -> refuses any positive aggregate
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency(
            [(2 * GB, 2 * GB)], 8, 0, shared_bytes=0, cache_worker_footprint_bytes=PB,
        )
    # None is explicit UNBOUNDED
    plan = zpar.plan_cell_concurrency(
        [(2 * GB, 2 * GB)] * 3, 8, None, shared_bytes=0, cache_worker_footprint_bytes=PB,
    )
    assert plan["cells_in_flight"] == 3
    assert plan["budget_bytes"] is None


def test_optin_empty_plan_shared_vs_finite_budget():
    # shared > finite budget -> refuse
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency(
            [], 8, 1 * GB, shared_bytes=2 * GB, cache_worker_footprint_bytes=PB,
        )
    # shared == 0 with finite 0 budget -> coherent empty plan
    plan = zpar.plan_cell_concurrency(
        [], 8, 0, shared_bytes=0, cache_worker_footprint_bytes=PB,
    )
    assert plan["n_cells"] == 0 and plan["aggregate_bytes"] == 0
    assert plan["cells_in_flight"] == 0


def test_optin_negative_finite_budget_is_refused():
    from zemosaic.core.zegrid import ram_contract as rc

    with pytest.raises(rc.RamContractError):
        zpar.plan_cell_concurrency(
            [(GB, GB)], 8, -1, shared_bytes=0, cache_worker_footprint_bytes=PB,
        )


def test_optin_unsafe_numeric_inputs_fail_closed():
    from zemosaic.core.zegrid import ram_contract as rc

    with pytest.raises(rc.RamContractError):
        zpar.plan_cell_concurrency(
            [(GB, GB)], 8, 10 * GB, shared_bytes=-1, cache_worker_footprint_bytes=PB,
        )
    with pytest.raises(rc.RamContractError):
        zpar.plan_cell_concurrency(
            [(GB, GB)], 8, 10 * GB, shared_bytes=0, cache_worker_footprint_bytes=-1,
        )
    with pytest.raises(rc.RamContractError):
        zpar.plan_cell_concurrency(
            [(-1, GB)], 8, 10 * GB, shared_bytes=0, cache_worker_footprint_bytes=PB,
        )
    with pytest.raises(rc.RamContractError):
        zpar.plan_cell_concurrency(
            [(True, GB)], 8, 10 * GB, shared_bytes=0, cache_worker_footprint_bytes=PB,
        )


# ---------------------------------------------------------------------------
# 9. M-1 (rework-2) — RAW inputs validated BEFORE any int()/max() coercion
# ---------------------------------------------------------------------------

def test_m1_raw_inputs_rejected_before_coercion():
    """Reviewer reproductions: cpu=True, shared=1.9, baseline=2.5,
    cache_build_workers=-3 / True must all FAIL CLOSED (never be coerced)."""
    from zemosaic.core.zegrid import ram_contract as rc

    cells = [(GB, GB)]
    for kwargs in (
        {"shared_bytes": 0, "cache_worker_footprint_bytes": PB, "cpu": True},
        {"shared_bytes": 1.9, "cache_worker_footprint_bytes": PB},
        {"shared_bytes": 0, "process_baseline_bytes": 2.5},
        {"shared_bytes": 0, "cache_worker_footprint_bytes": PB, "cache_build_workers": -3},
        {"shared_bytes": 0, "cache_worker_footprint_bytes": PB, "cache_build_workers": True},
    ):
        cpu = kwargs.pop("cpu", 8)
        with pytest.raises(rc.RamContractError):
            zpar.plan_cell_concurrency(cells, cpu, 10 * GB, **kwargs)


def test_m1_valid_ints_accepted_and_floors_are_conservative():
    # cache_build_workers=0 is a VALID int -> effective floor 1 (conservative)
    plan = zpar.plan_cell_concurrency(
        [(GB, GB)], 8, 10 * GB, shared_bytes=0, cache_build_workers=0,
    )
    assert plan["cache_workers_per_cell"] == 1
    # cpu=0/None are conservative (effective 1), NOT rejected
    for cpu in (0, None):
        p = zpar.plan_cell_concurrency(
            [(GB, GB)], cpu, 10 * GB, shared_bytes=0, cache_worker_footprint_bytes=PB,
        )
        assert p["cells_in_flight"] == 1


# ---------------------------------------------------------------------------
# 10. L-1 (rework-2) — process_baseline_bytes alone activates the opt-in
# ---------------------------------------------------------------------------

def test_l1_baseline_alone_activates_optin_and_is_honoured():
    plan = zpar.plan_cell_concurrency(
        [(GB, GB)], 4, 10 * GB, process_baseline_bytes=999,
    )
    assert plan["phase_accounting"] is True
    assert plan["process_baseline_bytes"] == 999
    assert plan["shared_bytes"] == 0  # shared defaults to 0 after validation


def test_default_branch_unchanged_when_no_phase_kwarg():
    plan = zpar.plan_cell_concurrency([(GB, GB)], 8, 10 * GB)
    assert "phase_accounting" not in plan
    assert "aggregate_bytes" not in plan
