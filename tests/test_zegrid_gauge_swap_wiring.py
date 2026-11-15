"""ZM-GAUGE-SWAP-G2a — product wiring of the per-group gauge credit.

Scope (mission ``ZM-GAUGE-SWAP-G2A-20261115``, contract
``docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md`` §4.7, §10.1–10.3,
§10.5, §10.7, §11 G2a):

* the scheduler's FINAL validator and the gauge plan path use the ONE G1b
  evaluator (no direct ``aggregate_host_peak_bytes(...) > budget``);
* policy OFF/absent is byte-identical to the historical path;
* policy ON + comfortable RAM changes nothing (no credit is granted when it is
  not needed);
* policy ON + constrained synthetic capacity admits the SINGLETON gauge group
  only, at exactly ONE worker, and leaves the stack on ``B_ram``;
* the credit is closed before the cache/stack and is unreachable afterwards;
* restored/imported/per-cell gauges get ZERO credit; the RAM and external_spool
  stack classes receive the SINGLETON gauge credit (G3 rework-1), GPU never does.

No engine run, no ``/proc`` dependency and no benchmark: capacity facts are
injected, exactly like the G1a/G1b suites.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from zemosaic.core.zegrid import gauge_admission as adm
from zemosaic.core.zegrid import gauge_credit as zgcredit
from zemosaic.core.zegrid import gauge_swap_capacity as cap
from zemosaic.core.zegrid import resource_envelope as renv
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20
GIB = 2 ** 30
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
ENGINE_PATH = SRC / "zemosaic" / "zemosaic_zegrid_mode.py"
SCHEDULER_PATH = SRC / "zemosaic" / "core" / "zegrid" / "scheduler.py"
CREDIT_PATH = SRC / "zemosaic" / "core" / "zegrid" / "gauge_credit.py"

CELLS = tuple(f"r0000c{i:04d}" for i in range(10))
FRAMES = tuple(f"f{i}" for i in range(12))

A_RAW = 10 * GIB
B_RAM = (4 * A_RAW) // 5  # 8 GiB: the bundle's own 4/5 plan budget


# ---------------------------------------------------------------------------
# Injected capacity facts and the product session
# ---------------------------------------------------------------------------

def _live_capacity(*, swap_free=6 * GIB, overcommit_mode=0):
    device = cap.SwapDevice(
        name="/swapfile",
        kind="file",
        size_bytes=int(swap_free) + MIB,
        used_bytes=MIB,
        zram=False,
        eligible=True,
        reason="backed_file",
    )
    level = cap.CgroupLevel(
        path="/sys/fs/cgroup",
        is_root=True,
        memory_controller=True,
        ram_status=cap.UNLIMITED,
        ram_headroom=None,
        swap_status=cap.UNLIMITED,
        swap_headroom=None,
        detail="root_cgroup_is_unbounded",
    )
    overcommit = cap.read_overcommit(overcommit_mode, None, None)
    return zgcredit.LiveCapacity(
        devices=(device,),
        cgroup_levels=(level,),
        overcommit=overcommit,
        facts={"schema": "test-injected"},
    )


def _session(**overrides):
    swap_free = int(overrides.pop("swap_free", 6 * GIB))
    overcommit_mode = int(overrides.pop("overcommit_mode", 0))
    data = dict(
        policy=cap.POLICY_BOUNDED_SWAP,
        capacity=_live_capacity(
            swap_free=swap_free, overcommit_mode=overcommit_mode
        ),
        platform="linux",
        restored=False,
        imported=False,
        workers=1,
        cell_gauge_mode="global",
    )
    data.update(overrides)
    return zgcredit.GaugeCreditSession(**data)


def _bundle_like(*, raw=A_RAW, budget=B_RAM):
    return SimpleNamespace(
        envelope=SimpleNamespace(
            raw_available_bytes=int(raw), budget_bytes=int(budget)
        )
    )


def _decision_for(session, *, p_gauge_raw):
    decision, credit = session.decision_for_bundle(
        _bundle_like(), p_gauge_raw=int(p_gauge_raw)
    )
    return decision, credit


# ---------------------------------------------------------------------------
# Envelope / plan fixtures (same frozen topology as the G1b suite)
# ---------------------------------------------------------------------------

def _host():
    return renv.HostEnvelope(
        observation_source="test.available_bytes",
        raw_available_bytes=A_RAW,
        factor_num=4,
        factor_den=5,
        budget_bytes=B_RAM,
        reserve_bytes=0,
        process_baseline_bytes=100,
    )


def _env(phase, peak, *, overlaps=(), backend="cpu"):
    estimate = renv.make_estimate({f"{phase}_bytes": int(peak)})
    return renv.ResourceEnvelope(
        envelope_id=f"{phase}.{backend}",
        phase=phase,
        backend=backend,
        host=_host(),
        device=(
            renv.DeviceEnvelope(
                vram_total_bytes=8 * GIB,
                vram_free_bytes=7 * GIB,
                vram_reserved_estimate_bytes=1 * GIB,
                budget_bytes=6 * GIB,
                device_name="test-gpu",
            )
            if backend == "gpu"
            else None
        ),
        geometry=renv.GeometryEnvelope(
            layout=(2, 2),
            refinement_rank=3,
            cell_id=None,
            patch_hw=(100, 100),
            tile_px=(64 if phase.startswith("cell_stack") else None),
            halo_px=32,
            channels=3,
        ),
        concurrency=renv.ConcurrencyEnvelope(
            units=2, units_in_flight=1, workers_per_unit=1, prefetch_units=0
        ),
        estimate=estimate,
        bound=renv.BoundEnvelope(kind="upper", margin_num=15, margin_den=100),
        accounting=renv.AccountingEnvelope(
            aggregate_rule=renv.AGGREGATE_RULE_MAX_OVER_PHASES,
            host_peak_bytes=estimate.peak_bytes,
            device_peak_bytes=0,
            disk_bytes=0,
            transfer_bytes=0,
        ),
        overlaps_with=tuple(overlaps),
        identity=renv.IdentityEnvelope(run_id="run", plan_hash="", pinned=False),
        reason_code=renv.REASON_UNCALIBRATED,
    )


def _plan(peaks, *, overlaps=(), gauge_backend="cpu"):
    adj = {}
    for a, b in overlaps:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    phases = [
        "gauge",
        "cache_build",
        "cell_stack_stream",
        "assembly",
        "export",
    ]
    return [
        _env(
            phase,
            peaks[phase],
            overlaps=tuple(sorted(adj.get(phase, ()))),
            backend=(gauge_backend if phase == "gauge" else "cpu"),
        )
        for phase in phases
    ]


# ---------------------------------------------------------------------------
# Scheduler context / joint plan fixtures
# ---------------------------------------------------------------------------

def _ctx(**overrides):
    data = dict(
        phase=renv.PHASE_CELL_STACK_STREAM,
        backend=renv.BACKEND_CPU,
        budget_bytes=B_RAM,
        raw_available_bytes=A_RAW,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=len(CELLS),
        shared_bytes=340 * MIB,
        per_unit_stack_bytes=200 * MIB,
        per_unit_cache_bytes=50 * MIB,
        per_unit_prefetch_bytes=50 * MIB,
        disk_free_bytes=100 * GIB,
        layout=(3, 2),
        refinement_rank=6,
        patch_hw=(200, 200),
        halo_px=32,
        channels=3,
        n_frames=50,
        prefetch_ladder=(0,),
        cell_ids=CELLS,
        frame_ids=FRAMES,
    )
    data.update(overrides)
    return zs.PlanContext(**data)


def _requests(*, gauge_bytes):
    return (
        zs.PhaseRequest(
            phase=renv.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": int(gauge_bytes)},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=renv.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 100 * MIB},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=renv.PHASE_ASSEMBLY,
            components={
                "halo_band_bytes": 4 * MIB,
                "retained_core_bytes": 50 * MIB,
            },
            units=1,
            units_in_flight=1,
            overlaps_with=(renv.PHASE_CELL_STACK_STREAM,),
        ),
        zs.PhaseRequest(
            phase=renv.PHASE_EXPORT,
            components={
                "coverage_plane_bytes": 10 * MIB,
                "science_planes_bytes": 30 * MIB,
            },
            units=1,
            units_in_flight=1,
        ),
    )


def _plan_joint(*, gauge_bytes, gauge_credit=None, session=None, stack_bytes=200 * MIB):
    if session is not None:
        _decision, gauge_credit = session.decision_for_bundle(
            _bundle_like(), p_gauge_raw=int(gauge_bytes)
        )
    ctx = _ctx(
        per_unit_stack_bytes=int(stack_bytes), gauge_credit=gauge_credit
    )
    inmem = dataclasses.replace(ctx, phase=renv.PHASE_CELL_STACK_INMEM)
    joint = zs.JointPlanInput(contexts=(inmem, ctx))
    return zs.plan_joint(
        joint,
        phase_requests=_requests(gauge_bytes=gauge_bytes),
        mode=zs.MODE_ACTIVE,
    )


# ---------------------------------------------------------------------------
# (1) Policy resolution: absent = ram_only, unknown = typed refusal
# ---------------------------------------------------------------------------

def test_policy_absent_none_and_empty_are_ram_only():
    assert zgcredit.resolve_gauge_memory_policy(None) == cap.POLICY_RAM_ONLY
    assert (
        zgcredit.resolve_gauge_memory_policy(SimpleNamespace())
        == cap.POLICY_RAM_ONLY
    )
    assert (
        zgcredit.resolve_gauge_memory_policy(
            SimpleNamespace(zegrid_gauge_memory_policy=None)
        )
        == cap.POLICY_RAM_ONLY
    )
    assert (
        zgcredit.resolve_gauge_memory_policy(
            SimpleNamespace(zegrid_gauge_memory_policy="   ")
        )
        == cap.POLICY_RAM_ONLY
    )


def test_policy_bounded_swap_is_accepted_and_unknown_rejected():
    assert (
        zgcredit.resolve_gauge_memory_policy(
            SimpleNamespace(zegrid_gauge_memory_policy="bounded_swap")
        )
        == cap.POLICY_BOUNDED_SWAP
    )
    with pytest.raises(zgcredit.GaugePolicyError):
        zgcredit.resolve_gauge_memory_policy(
            SimpleNamespace(zegrid_gauge_memory_policy="swap_all")
        )
    with pytest.raises(zgcredit.GaugePolicyError):
        zgcredit.resolve_gauge_memory_policy(
            SimpleNamespace(zegrid_gauge_memory_policy=1)
        )


# ---------------------------------------------------------------------------
# (2) Policy OFF / absent: byte-identical scheduler decisions
# ---------------------------------------------------------------------------

def test_policy_off_and_absent_produce_identical_plans_and_hashes():
    baseline = _plan_joint(gauge_bytes=400 * MIB)
    absent = _plan_joint(
        gauge_bytes=400 * MIB, gauge_credit=adm.GaugeCredit(policy=None)
    )
    ram_only = _plan_joint(
        gauge_bytes=400 * MIB,
        gauge_credit=adm.GaugeCredit(policy=cap.POLICY_RAM_ONLY),
    )
    assert baseline.plan_hash == absent.plan_hash == ram_only.plan_hash
    assert baseline.chosen == absent.chosen == ram_only.chosen
    assert baseline.decisions == absent.decisions == ram_only.decisions
    assert (
        baseline.aggregate_host_peak_bytes
        == absent.aggregate_host_peak_bytes
        == ram_only.aggregate_host_peak_bytes
    )
    assert baseline.to_manifest_dict() == absent.to_manifest_dict()


def test_policy_off_refusal_message_is_the_historical_one():
    with pytest.raises(zs.PlanRefused) as baseline_exc:
        _plan_joint(gauge_bytes=9 * GIB)
    with pytest.raises(zs.PlanRefused) as absent_exc:
        _plan_joint(gauge_bytes=9 * GIB, gauge_credit=adm.GaugeCredit(policy=None))
    assert baseline_exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM
    assert absent_exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM
    assert absent_exc.value.detail == baseline_exc.value.detail
    assert "aggregate host peak" in absent_exc.value.detail


def test_credit_session_is_not_created_for_a_ram_only_policy():
    """The engine only builds a session for an explicit bounded_swap policy."""
    assert _session(policy=cap.POLICY_RAM_ONLY).eligible is False
    assert _session(policy=cap.POLICY_RAM_ONLY).decision_for(
        a_raw=A_RAW, p_gauge_raw=9 * GIB
    ) is None


# ---------------------------------------------------------------------------
# (3) Policy ON + comfortable RAM: no decision, worker or hash difference
# ---------------------------------------------------------------------------

def test_policy_on_comfortable_ram_changes_nothing():
    session = _session()
    decision, credit = _decision_for(session, p_gauge_raw=400 * MIB)
    assert decision is not None
    assert decision.credit_needed_bytes == 0
    assert session.grants_credit(decision) is False

    baseline = _plan_joint(gauge_bytes=400 * MIB)
    wired = _plan_joint(gauge_bytes=400 * MIB, session=session)
    assert wired.plan_hash == baseline.plan_hash
    assert wired.chosen == baseline.chosen
    assert wired.decisions == baseline.decisions
    assert wired.aggregate_host_peak_bytes == baseline.aggregate_host_peak_bytes
    gauge_env = wired.phase_map()[renv.PHASE_GAUGE]
    assert int(gauge_env.concurrency.units) == 1
    # The plan's gauge group is still priced exclusively against B_ram: the
    # credit was never applied.
    assert int(gauge_env.estimate.peak_bytes) == 400 * MIB


# ---------------------------------------------------------------------------
# (4) Policy ON + constrained synthetic capacity: gauge admitted, stack RAM-only
# ---------------------------------------------------------------------------

def test_policy_on_constrained_capacity_admits_the_one_worker_gauge():
    session = _session()
    decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    assert decision is not None
    assert decision.b_ram == B_RAM
    assert int(decision.credit_needed_bytes or 0) > 0
    assert session.grants_credit(decision) is True
    assert credit is not None and credit.workers == 1

    with pytest.raises(zs.PlanRefused) as off_exc:
        _plan_joint(gauge_bytes=9 * GIB)
    assert off_exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM

    wired = _plan_joint(gauge_bytes=9 * GIB, session=session)
    gauge_env = wired.phase_map()[renv.PHASE_GAUGE]
    assert int(gauge_env.concurrency.units) == 1
    # The gauge aggregate now legitimately exceeds B_ram; the TRUE peak is
    # published (never reduced by the credit).
    assert wired.aggregate_host_peak_bytes > B_RAM
    decision_obj = adm.admit_validated(
        wired.phases, B_RAM, credit
    )
    gauge_group = [g for g in decision_obj.groups if g.phases == ("gauge",)][0]
    assert gauge_group.credit_applied is True
    assert gauge_group.ceiling_bytes == B_RAM + int(decision.s_credit)


def test_credit_never_saves_a_stack_group_even_when_huge():
    session = _session(swap_free=512 * GIB)
    decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    assert session.grants_credit(decision) is True
    envs = _plan(
        {
            "gauge": 9 * GIB,
            "cache_build": 100 * MIB,
            "cell_stack_stream": 20 * GIB,
            "assembly": 100 * MIB,
            "export": 100 * MIB,
        },
        overlaps=(("cell_stack_stream", "assembly"),),
    )
    result = adm.admit_validated(envs, B_RAM, credit)
    assert result.admitted is False
    gauge_group = [g for g in result.groups if g.phases == ("gauge",)][0]
    stack_group = [
        g for g in result.groups if "cell_stack_stream" in g.phases
    ][0]
    assert gauge_group.admitted is True and gauge_group.credit_applied is True
    assert stack_group.admitted is False
    assert stack_group.ceiling_bytes == B_RAM
    stack_refusal = [r for r in result.refusals if "cell_stack_stream" in r.phases][0]
    assert stack_refusal.code == adm.REASON_REFUSE_HOST_RAM


def test_typed_refusal_code_reaches_the_scheduler_for_an_ineligible_gauge():
    # A singleton gauge that does not fit even WITH the credit: the scheduler
    # must surface the typed GAUGE_* code, not a generic message.
    session = _session(swap_free=1 * MIB)  # below S_MIN_USABLE ⇒ no credit
    decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    assert session.grants_credit(decision) is False
    with pytest.raises(zs.PlanRefused) as exc:
        _plan_joint(gauge_bytes=9 * GIB, session=session)
    assert exc.value.reason_code.startswith("GAUGE_CREDIT_")
    assert exc.value.reason_code in adm.REFUSAL_CODES


# ---------------------------------------------------------------------------
# (5) Closure: no credit after the gauge, RAM-only re-admission
# ---------------------------------------------------------------------------

def _patch_capacity_refresh(monkeypatch, capacity=None):
    """Q3: ``open_for_compute`` re-reads the live facts; inject them."""
    calls = {"n": 0}
    target = _live_capacity() if capacity is None else capacity

    def _collect():
        calls["n"] += 1
        return target

    monkeypatch.setattr(zgcredit, "collect_live_capacity", _collect)
    return calls


def test_credit_is_closed_and_unreachable_before_any_stack_step(monkeypatch):
    _patch_capacity_refresh(monkeypatch)
    session = _session()
    opened = session.open_for_compute(_bundle_like(), p_gauge_raw=9 * GIB)
    assert opened is not None and opened["event"] == zgcredit.EVENT_OPEN
    assert opened["credit_bytes"] > 0
    assert session.credit_for_compute() is not None

    closed = session.close()
    assert closed["event"] == zgcredit.EVENT_CLOSED
    assert closed["closed"] is True
    assert closed["credit_granted_during_gauge"] is True
    assert closed["swap_used_bytes"] is None
    assert "PERMIT" in closed["note"]
    assert session.closed is True
    assert session.credit_for_compute() is None
    assert session.decision_for(a_raw=A_RAW, p_gauge_raw=9 * GIB) is None
    assert session.plan_credit(None, workers=1) is None
    session.require_closed_for_stack()


def test_after_gauge_insufficient_ram_refuses_even_with_a_huge_credit(monkeypatch):
    _patch_capacity_refresh(monkeypatch)
    session = _session()
    session.open_for_compute(_bundle_like(), p_gauge_raw=9 * GIB)
    session.close()
    # The post-gauge path receives NO credit at all (the engine never forwards
    # the session), so an insufficient stack is refused on the RAM ceiling.
    with pytest.raises(zs.PlanRefused) as exc:
        _plan_joint(gauge_bytes=400 * MIB, stack_bytes=12 * GIB)
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM

    # Even a caller that KEPT the pre-close credit cannot buy a stack group.
    _decision, credit = None, adm.GaugeCredit(
        decision=_decision_for(_session(), p_gauge_raw=9 * GIB)[0],
        policy=cap.POLICY_BOUNDED_SWAP,
    )
    envs = _plan(
        {
            "gauge": 9 * GIB,
            "cache_build": 100 * MIB,
            "cell_stack_stream": 20 * GIB,
            "assembly": 100 * MIB,
            "export": 100 * MIB,
        },
        overlaps=(("cell_stack_stream", "assembly"),),
    )
    result = adm.admit_validated(envs, B_RAM, credit)
    assert result.admitted is False
    assert [r.code for r in result.refusals if "cell_stack_stream" in r.phases] == [
        adm.REASON_REFUSE_HOST_RAM
    ]


def test_close_is_idempotent_and_always_emits_the_gauge_credit_closed_event():
    session = _session()
    first = session.close()
    second = session.close()
    assert first["event"] == second["event"] == zgcredit.EVENT_CLOSED
    assert all(
        event["event"] == zgcredit.EVENT_CLOSED
        for event in session.events
    )


# ---------------------------------------------------------------------------
# (6) Mutation: removing ONE propagation point breaks the bounded path
# ---------------------------------------------------------------------------

def test_mutation_removing_session_decision_propagation_fails_closed(monkeypatch):
    session = _session()

    def _no_credit(*_args, **_kwargs):
        return None, None

    monkeypatch.setattr(session, "decision_for_bundle", _no_credit)
    with pytest.raises(zs.PlanRefused) as exc:
        _plan_joint(gauge_bytes=9 * GIB, session=session)
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM


def test_mutation_removing_the_context_field_defeats_the_credit():
    session = _session()
    _, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    assert credit is not None
    ctx = _ctx(per_unit_stack_bytes=200 * MIB, gauge_credit=credit)
    # Remove the ONE propagation point (the PlanContext field).
    stripped = dataclasses.replace(ctx, gauge_credit=None)
    joint = zs.JointPlanInput(
        contexts=(
            dataclasses.replace(stripped, phase=renv.PHASE_CELL_STACK_INMEM),
            stripped,
        )
    )
    with pytest.raises(zs.PlanRefused) as exc:
        zs.plan_joint(
            joint,
            phase_requests=_requests(gauge_bytes=9 * GIB),
            mode=zs.MODE_ACTIVE,
        )
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM


def test_mutation_removing_the_worker_eligibility_fails_closed():
    session = _session(workers=2)
    decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    # The credit arithmetic may exist, but the evaluator refuses a 2-worker
    # gauge with its typed code: the ONE-worker invariant is enforced by the
    # evaluator, not by the caller's good will.
    envs = _plan(
        {
            "gauge": 9 * GIB,
            "cache_build": 100 * MIB,
            "cell_stack_stream": 200 * MIB,
            "assembly": 100 * MIB,
            "export": 100 * MIB,
        },
        overlaps=(("cell_stack_stream", "assembly"),),
    )
    assert credit is not None and credit.workers == 2
    result = adm.admit_validated(envs, B_RAM, credit)
    assert result.admitted is False
    assert result.reason_code == adm.REASON_CREDIT_REQUIRES_ONE_WORKER


# ---------------------------------------------------------------------------
# (6b) REWORK-1 — M1 / L1 / L3 / Q3 / L2
# ---------------------------------------------------------------------------


def test_m1_effective_worker_is_one_whenever_the_plan_applied_the_credit():
    from zemosaic import zemosaic_zegrid_mode as zz

    assert zz._r55_gauge_workers_effective(credit_applied=True, proposed=8) == 1
    assert zz._r55_gauge_workers_effective(credit_applied=True, proposed=1) == 1
    assert zz._r55_gauge_workers_effective(credit_applied=False, proposed=8) == 8


def test_m1_not_needed_at_open_still_runs_one_worker(monkeypatch):
    """The plan applied the credit; the credit became unnecessary at open.

    The gauge MUST still run with one worker: ``not_needed`` only proves the
    fit at one worker, never at the initial ladder value (M1 point 2).
    """
    from zemosaic import zemosaic_zegrid_mode as zz

    _patch_capacity_refresh(monkeypatch, _live_capacity(swap_free=512 * GIB))
    session = _session(swap_free=512 * GIB)
    _decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    assert session.grants_credit(_decision) is True

    # At open the gauge now fits B_ram at ONE worker ⇒ credit not needed.
    opened = session.open_for_compute(_bundle_like(), p_gauge_raw=400 * MIB)
    assert opened is None
    assert session.events[-1]["reason"] == "not_needed"
    # The plan was admitted with the credit: the effective worker is 1.
    assert zz._r55_gauge_workers_effective(
        credit_applied=True, proposed=8
    ) == 1
    # And the engine applies that value BEFORE/independently of the open result.
    source = _function_source(ENGINE_PATH, "_run_single")
    helper_at = source.index("_r55_gauge_workers_effective")
    open_at = source.index("open_for_compute")
    assert helper_at < open_at


def test_m1_ram_contract_diagnostics_publish_the_effective_value():
    from zemosaic import zemosaic_zegrid_mode as zz

    class _Obs:
        def to_dict(self):
            return {"raw": 1}

    class _Outcome:
        observations = ()
        history = ()
        replans = 0
        tolerance = False

        def __init__(self):
            self.final_observation = _Obs()

    def _bundle(gauge_workers):
        envelope = SimpleNamespace(
            raw_available_bytes=A_RAW,
            factor_num=4,
            factor_den=5,
            budget_bytes=B_RAM,
            layout=(2, 2),
            shared_bytes=0,
            stack_total_bytes=0,
            cache_total_bytes=0,
            aggregate_bytes=0,
            backend="cpu",
            to_dict=lambda: {},
        )
        return SimpleNamespace(
            envelope=envelope,
            cache_worker_footprint_bytes=0,
            worst_patch_px=0,
            worker_budget_bytes=0,
            gauge_workers=int(gauge_workers),
            cache_build_workers=4,
            chosen_mode="stream",
            cpu_streaming_tile=64,
            gpu_streaming_tile=None,
            gpu_pre_execution_fallback=None,
            attempts=[],
        )

    initial = _bundle(8)
    final = _bundle(8)  # bundle keeps the historical initial value
    off = zz._ram_contract_diagnostics(_Outcome(), final, initial)
    assert "gauge_workers_effective" not in off
    assert "gauge_credit_applied" not in off
    assert off["gauge_workers"] == 8

    credited = zz._ram_contract_diagnostics(
        _Outcome(),
        final,
        initial,
        **zz._r55_gauge_credit_diagnostics_fields(credit_applied=True),
    )
    assert credited["gauge_workers_effective"] == 1
    assert credited["gauge_credit_applied"] is True
    # The historical field is NOT rewritten: an explicit effective value is
    # published instead (never a misleading "1" stamped on the initial fact).
    assert credited["gauge_workers"] == 8

    # G3 regression: the REAL call site passes the O1 observation too
    # (policy/A/B_ram/S_credit/need/sources). That combination raised
    # TypeError on every credited run before G3; it must be published.
    full = zz._ram_contract_diagnostics(
        _Outcome(),
        final,
        initial,
        **zz._r55_gauge_credit_diagnostics_fields(
            credit_applied=True,
            policy="bounded_swap",
            a_raw_bytes=A_RAW,
            b_ram_bytes=B_RAM,
            s_credit_bytes=123,
            credit_needed_bytes=7,
            sources={"schema": "sources"},
        ),
    )
    assert full["gauge_memory_policy"] == "bounded_swap"
    assert full["a_raw_bytes"] == A_RAW
    assert full["b_ram_bytes"] == B_RAM
    assert full["s_credit_bytes"] == 123
    assert full["credit_needed_bytes"] == 7
    assert full["capacity_sources"] == {"schema": "sources"}


def test_m1_credited_diagnostics_reach_the_product_block():
    source = _function_source(ENGINE_PATH, "_run_single")
    # G4/R1: every ram_contract site now goes through the ONE shared helper
    # (the per-cell overrides used to rebuild the block and drop the O1 fields).
    assert "_gauge_credit_ram_contract_fields(" in source
    assert "credit_applied=_gauge_credit_plan_applied" in source
    assert "_r55_gauge_credit_diagnostics_fields(" not in source


def test_g4_ram_contract_helper_publishes_the_whole_o1_observation():
    """G4/R1: a CREDITED run's ``ram_contract`` carries the O1 fields.

    The helper is the single producer of the additive kwargs of every
    ``_ram_contract_diagnostics`` call site in ``_run_single``, so this
    exercises the exact dict merged into ``per_cell_diagnostics["ram_contract"]``
    (policy, A, B_ram, S_credit, credit_needed, sources) and the OFF-identical
    empty case.
    """
    from zemosaic import zemosaic_zegrid_mode as zz

    decision = SimpleNamespace(
        a_raw=A_RAW,
        b_ram=B_RAM,
        s_credit=123456,
        credit_needed_bytes=654321,
    )
    block = {"sources": {"schema": "gauge-swap-capacity-v1"}}

    fields = zz._gauge_credit_ram_contract_fields(
        credit_applied=True,
        policy="bounded_swap",
        admission_decision=decision,
        block=block,
    )
    assert fields == {
        "gauge_workers_effective": 1,
        "gauge_credit_applied": True,
        "gauge_memory_policy": "bounded_swap",
        "a_raw_bytes": A_RAW,
        "b_ram_bytes": B_RAM,
        "s_credit_bytes": 123456,
        "credit_needed_bytes": 654321,
        "capacity_sources": {"schema": "gauge-swap-capacity-v1"},
    }

    class _Obs:
        def to_dict(self):
            return {"raw": 1}

    class _Outcome:
        observations = ()
        history = ()
        replans = 0
        tolerance = False

        def __init__(self):
            self.final_observation = _Obs()

    envelope = SimpleNamespace(
        raw_available_bytes=A_RAW,
        factor_num=4,
        factor_den=5,
        budget_bytes=B_RAM,
        layout=(2, 2),
        shared_bytes=0,
        stack_total_bytes=0,
        cache_total_bytes=0,
        aggregate_bytes=0,
        backend="cpu",
        to_dict=lambda: {},
    )
    bundle = SimpleNamespace(
        envelope=envelope,
        cache_worker_footprint_bytes=0,
        worst_patch_px=0,
        worker_budget_bytes=0,
        gauge_workers=1,
        cache_build_workers=4,
        chosen_mode="stream",
        cpu_streaming_tile=64,
        gpu_streaming_tile=None,
        gpu_pre_execution_fallback=None,
        attempts=[],
    )
    published = zz._ram_contract_diagnostics(
        _Outcome(), bundle, bundle, **fields
    )
    # The merged block -- what the manifest publishes -- carries every O1 key.
    assert published["gauge_memory_policy"] == "bounded_swap"
    assert published["a_raw_bytes"] == A_RAW
    assert published["b_ram_bytes"] == B_RAM
    assert published["s_credit_bytes"] == 123456
    assert published["credit_needed_bytes"] == 654321
    assert published["capacity_sources"] == {
        "schema": "gauge-swap-capacity-v1"
    }
    assert published["gauge_workers_effective"] == 1
    assert published["gauge_credit_applied"] is True

    # No credit -> NOTHING additive (OFF manifests stay byte-identical).
    assert (
        zz._gauge_credit_ram_contract_fields(
            credit_applied=False,
            policy="bounded_swap",
            admission_decision=decision,
            block=block,
        )
        == {}
    )
    # A credited decision that is unknown (no observation) degrades to the
    # worker pair only -- never an invented value.
    assert zz._gauge_credit_ram_contract_fields(
        credit_applied=True, policy=None, admission_decision=None, block=None
    ) == {"gauge_workers_effective": 1, "gauge_credit_applied": True}


def test_g4_every_ram_contract_site_uses_the_shared_helper():
    """G4/R1: no ``_ram_contract_diagnostics`` call in ``_run_single`` can
    rebuild ``ram_contract`` without the O1 observation again."""
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    run_single = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_single":
            run_single = node
            break
    assert run_single is not None

    calls = []
    for node in ast.walk(run_single):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_ram_contract_diagnostics":
                calls.append(node)
    assert len(calls) == 6, len(calls)  # first block, 2 per-cell, 3 GPU

    for call in calls:
        star_sources = [
            kw.value.func.id
            for kw in call.keywords
            if kw.arg is None
            and isinstance(kw.value, ast.Call)
            and isinstance(kw.value.func, ast.Name)
        ]
        assert star_sources == ["_gauge_credit_ram_contract_fields"], ast.dump(call)

    # The O1 producer is called ONCE (inside the helper), not inline per site.
    module_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_r55_gauge_credit_diagnostics_fields"
    ]
    assert len(module_calls) == 1  # only the G4 helper produces the fields


def test_l1_stale_credit_refusal_is_a_typed_runtime_error():
    err = zgcredit.GaugeCreditError(
        "GAUGE_CREDIT_INSUFFICIENT: stale permit"
    )
    assert isinstance(err, RuntimeError)
    assert err.reason_code == "GAUGE_CREDIT_INSUFFICIENT"
    assert str(err).startswith("GAUGE_CREDIT_INSUFFICIENT:")
    source = _function_source(ENGINE_PATH, "_run_single")
    assert "raise zgcredit.GaugeCreditError(" in source
    assert "raise RuntimeError(\n                            \"GAUGE_CREDIT" not in source


def test_l3_session_has_no_dead_plan_decision_state():
    import dataclasses as _dc

    names = {field.name for field in _dc.fields(zgcredit.GaugeCreditSession)}
    assert "_plan_decision" not in names
    assert "_plan_decision" not in CREDIT_PATH.read_text(encoding="utf-8")
    # The creation-time diagnostic is now emitted after admission, on the
    # EFFECTIVE bundle, not on the initial bundle.
    source = _function_source(ENGINE_PATH, "_run_single")
    assert "_effective_decision" in source
    assert "decision_for_bundle(\n                                _initial_bundle" not in source


def test_q3_capacity_is_refreshed_before_open_and_a_stale_snapshot_fails(monkeypatch):
    session = _session(swap_free=512 * GIB)
    _decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    assert session.grants_credit(_decision) is True

    # Swap was disabled between the plan-time snapshot and the compute.
    stale = _live_capacity(swap_free=0)
    calls = _patch_capacity_refresh(monkeypatch, stale)
    opened = session.open_for_compute(_bundle_like(), p_gauge_raw=9 * GIB)
    assert calls["n"] == 1
    assert session.capacity is stale
    assert session.diagnostics["capacity_refresh_count"] == 1
    # The refreshed facts no longer prove the credit: no permit is issued and
    # the event reports the refreshed status (never the old S_effectif).
    assert opened is None
    assert session.events[-1]["event"] == zgcredit.EVENT_OPEN
    assert session.events[-1]["opened"] is False
    assert session.events[-1]["reason"] != "not_needed"


def test_l2_mutation_removing_m1_enforcement_reintroduces_the_initial_workers(monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    assert zz._r55_gauge_workers_effective(credit_applied=True, proposed=8) == 1
    # Simulate the mutation: identity instead of the M1 enforcement.
    monkeypatch.setattr(
        zz, "_r55_gauge_workers_effective", lambda *, credit_applied, proposed: int(proposed)
    )
    assert zz._r55_gauge_workers_effective(credit_applied=True, proposed=8) == 8


# ---------------------------------------------------------------------------
# (7) Restored / imported / per-cell gauges: zero credit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides",
    [
        {"restored": True},
        {"imported": True},
        {"cell_gauge_mode": "per_cell"},
        {"platform": "win32"},
    ],
)
def test_ineligible_gauge_contexts_get_zero_credit(overrides):
    session = _session(**overrides)
    assert session.eligible is False
    assert session.decision_for(a_raw=A_RAW, p_gauge_raw=9 * GIB) is None
    with pytest.raises(zs.PlanRefused) as exc:
        _plan_joint(gauge_bytes=9 * GIB, session=session)
    assert exc.value.reason_code == renv.REASON_REFUSE_HOST_RAM


def test_restored_gauge_identity_and_hash_are_unchanged():
    session = _session(restored=True)
    baseline = _plan_joint(gauge_bytes=400 * MIB)
    wired = _plan_joint(gauge_bytes=400 * MIB, session=session)
    assert wired.plan_hash == baseline.plan_hash
    assert wired.chosen == baseline.chosen
    assert wired.identity == baseline.identity


# ---------------------------------------------------------------------------
# (8) CPU / external / GPU separately, and no induced credit
# ---------------------------------------------------------------------------

def test_gpu_decide_helpers_never_carry_a_credit():
    """GPU gauge is never credited (unchanged); the EXTERNAL class now is.

    ZM-GAUGE-SWAP-G3 REWORK-1: the pre-stack global gauge is the same
    singleton group whatever the stack class, so ``external_spool`` must
    receive the credit. The GPU path stays credit-free.
    """
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in (
        "_r55_active_admit_external",
        "_r55_active_gpu_decide",
        "_r55_active_gpu_admit",
    ):
        node = funcs[name]
        arg_names = {a.arg for a in node.args.args + node.args.kwonlyargs}
        assert "gauge_credit" not in arg_names
        segment = ast.get_source_segment(source, node) or ""
        assert "gauge_credit" not in segment


def test_external_decide_accepts_the_gauge_credit():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    node = funcs["_r55_active_external_decide"]
    arg_names = {a.arg for a in node.args.args + node.args.kwonlyargs}
    assert "gauge_credit" in arg_names
    segment = ast.get_source_segment(source, node) or ""
    assert "grants_credit" in segment
    # ``_r55_active_decide`` FORWARDS the same object to the external path.
    decide = ast.get_source_segment(source, funcs["_r55_active_decide"]) or ""
    external_call = decide.split("_r55_active_external_decide", 1)[1]
    assert "gauge_credit=gauge_credit" in external_call.split(")", 1)[0]


def test_gpu_gauge_is_not_credited_even_with_a_huge_credit():
    session = _session(swap_free=512 * GIB)
    _decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    envs = _plan(
        {
            "gauge": 9 * GIB,
            "cache_build": 100 * MIB,
            "cell_stack_stream": 200 * MIB,
            "assembly": 100 * MIB,
            "export": 100 * MIB,
        },
        overlaps=(("cell_stack_stream", "assembly"),),
        gauge_backend="gpu",
    )
    result = adm.admit_validated(envs, B_RAM, credit)
    assert result.admitted is False
    gauge_group = [g for g in result.groups if g.phases == ("gauge",)][0]
    assert gauge_group.credit_applied is False
    assert gauge_group.ceiling_bytes == B_RAM


# ---------------------------------------------------------------------------
# (10) The REAL engine decision path: forced 1 worker, credit only for gauge
# ---------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, bound_bytes):
        self._b = int(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


def _tiny_corpus(n=4, h=40, w=40):
    import numpy as np
    from astropy.wcs import WCS

    from zemosaic.core.zegrid import geometry as zg

    out = []
    for i in range(n):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + i * 0.004, 30.0]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (h, w)
        fid = f"d{i}.fits"
        out.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid),
                source_path=f"/x/{fid}",
                shape_hw=(h, w),
                wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="",
                instrument="",
            )
        )
    return out


def test_engine_decide_forces_one_worker_and_credits_only_the_gauge(
    tmp_path, monkeypatch
):
    import os as os_mod

    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import auto_layout as zal
    from zemosaic.core.zegrid import geometry as zg
    from zemosaic.core.zegrid import parallel as zpar

    descs = _tiny_corpus()
    canvas = zg.build_canvas(descs)
    floors = zal.ScientificFloors
    monkeypatch.setattr(
        zal,
        "ScientificFloors",
        lambda *_a, **_k: floors(
            min_patch_area_px=1,
            min_contributors=1,
            max_halo_overhead=1e9,
        ),
    )
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os_mod, "cpu_count", lambda: 3)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(100 * MIB))
    monkeypatch.setattr(
        zz,
        "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: 100 * MIB,
    )
    monkeypatch.setattr(
        zz, "_bundle_gauge_footprint_bytes", lambda bundle, canvas_px: 9 * GIB
    )
    bundle = zz._build_execution_bundle(
        canvas,
        descs,
        None,
        raw_available_bytes=A_RAW,
        budget_bytes=B_RAM,
        cpu_count=8,
        cache_root=str(tmp_path / "cache"),
    )
    kwargs = dict(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=8,
        cache_root=tmp_path,
        output_dir=tmp_path,
        frame_ids=[d.frame_id.logical_path for d in descs],
    )
    with pytest.raises(zs.PlanRefused):
        zz._r55_active_decide(**kwargs)

    decision = zz._r55_active_decide(**kwargs, gauge_credit=_session())
    assert decision.gauge_workers == 1
    gauge_env = decision.plan_set.phase_map()[renv.PHASE_GAUGE]
    assert int(gauge_env.concurrency.units) == 1
    assert decision.facts.get("gauge_credit_applied") is True
    assert decision.plan_set.aggregate_host_peak_bytes >= 9 * GIB


def test_external_decide_credits_the_gauge_and_keeps_the_external_stack_ram_only(
    tmp_path, monkeypatch
):
    """REWORK-1: external_spool receives the gauge credit, only for the gauge.

    Same frozen 1-worker rule as the RAM path; the external stack envelope
    keeps ``B_ram`` and ``credit_applied=False``, and the class stays
    single-owner CPU external.
    """
    import os as os_mod

    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import auto_layout as zal
    from zemosaic.core.zegrid import geometry as zg
    from zemosaic.core.zegrid import parallel as zpar

    descs = _tiny_corpus()
    canvas = zg.build_canvas(descs)
    floors = zal.ScientificFloors
    monkeypatch.setattr(
        zal,
        "ScientificFloors",
        lambda *_a, **_k: floors(
            min_patch_area_px=1,
            min_contributors=1,
            max_halo_overhead=1e9,
        ),
    )
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os_mod, "cpu_count", lambda: 3)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(100 * MIB))
    monkeypatch.setattr(
        zz,
        "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, **kw: 100 * MIB,
    )
    monkeypatch.setattr(
        zz, "_bundle_gauge_footprint_bytes", lambda bundle, canvas_px: 9 * GIB
    )
    bundle = zz._build_execution_bundle(
        canvas,
        descs,
        None,
        raw_available_bytes=A_RAW,
        budget_bytes=B_RAM,
        cpu_count=8,
        cache_root=str(tmp_path / "cache"),
        force_external=True,
    )
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL

    kwargs = dict(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=8,
        cache_root=tmp_path,
        output_dir=tmp_path,
        frame_ids=[d.frame_id.logical_path for d in descs],
    )
    # Without the credit the constrained gauge cannot be admitted (9 GiB raw).
    with pytest.raises(zs.PlanRefused):
        zz._r55_active_decide(**kwargs)

    decision = zz._r55_active_decide(**kwargs, gauge_credit=_session())
    assert decision.working_set_class == zz.WORKING_SET_EXTERNAL
    assert decision.facts.get("gauge_credit_applied") is True
    assert int(decision.gauge_workers) == 1
    assert int(decision.mapping["units_in_flight"]) == 1

    admission = decision.plan_set.admission
    assert admission is not None
    assert admission.execution_digest, "a credited plan must publish its digest"
    assert str(admission.execution_digest).startswith("sha256:")
    groups = {tuple(g.phases): g for g in admission.groups}
    gauge = groups[("gauge",)]
    stack = groups[("cell_stack_stream",)]
    assert gauge.credit_applied is True
    assert gauge.ceiling_bytes == B_RAM + int(admission.credit_bytes)
    assert stack.credit_applied is False
    assert stack.ceiling_bytes == B_RAM
    assert stack.admitted is True


def test_external_stack_over_b_ram_is_refused_even_with_gauge_credit():
    """The credit is group-scoped: it can never save an over-budget external stack."""
    session = _session()
    _decision, credit = _decision_for(session, p_gauge_raw=9 * GIB)
    envs = _plan(
        {
            "gauge": 9 * GIB,
            "cache_build": 100 * MIB,
            "cell_stack_stream": 9 * GIB,
            "assembly": 100 * MIB,
            "export": 100 * MIB,
        }
    )
    result = adm.admit_validated(envs, B_RAM, credit)
    groups = {tuple(g.phases): g for g in result.groups}
    gauge = groups[("gauge",)]
    stack = groups[("cell_stack_stream",)]
    assert gauge.admitted is True and gauge.credit_applied is True
    assert gauge.ceiling_bytes == B_RAM + result.credit_bytes
    assert stack.admitted is False
    assert stack.credit_applied is False
    assert stack.ceiling_bytes == B_RAM
    assert result.reason_code == adm.REASON_REFUSE_HOST_RAM


def test_gauge_swap_block_publishes_execution_digest_only_when_credited():
    from zemosaic import zemosaic_zegrid_mode as zz

    digest = "sha256:" + "a" * 64
    credited = zz._gauge_swap_diagnostics_block(
        policy="bounded_swap",
        session=_session(),
        credit_applied=True,
        gauge_workers_effective=1,
        execution_digest=digest,
    )
    assert credited is not None
    assert credited["execution_digest"] == digest
    # Non-credited bounded_swap: the key is ABSENT (not null).
    plain = zz._gauge_swap_diagnostics_block(
        policy="bounded_swap", session=_session(), credit_applied=False
    )
    assert plain is not None
    assert "execution_digest" not in plain
    # ram_only: no block at all.
    assert (
        zz._gauge_swap_diagnostics_block(
            policy="ram_only", session=_session(), credit_applied=False
        )
        is None
    )


def test_run_single_reads_the_admission_digest_for_the_manifest_block():
    source = _function_source(ENGINE_PATH, "_run_single")
    assert "plan_set" in source and "admission" in source
    assert "execution_digest=_gauge_swap_execution_digest" in source


# ---------------------------------------------------------------------------
# (9) Structural guarantees (AST / source), integer-only arithmetic
# ---------------------------------------------------------------------------

def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def _contains_peak_budget_compare(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Compare):
            continue
        if not any(isinstance(op, ast.Gt) for op in sub.ops):
            continue
        for side in [sub.left, *sub.comparators]:
            for inner in ast.walk(side):
                if isinstance(inner, ast.Call) and isinstance(
                    inner.func, ast.Name
                ) and inner.func.id == "aggregate_host_peak_bytes":
                    return True
    return False


def test_final_validator_has_no_direct_peak_budget_comparison():
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    checked = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "build_plan_set",
            "build_joint_plan_set",
        ):
            checked += 1
            assert _contains_peak_budget_compare(node) is False, node.name
            segment = ast.get_source_segment(
                SCHEDULER_PATH.read_text(encoding="utf-8"), node
            ) or ""
            assert "_admission_decision(" in segment
    assert checked == 2

    helper = _function_source(SCHEDULER_PATH, "_admission_decision")
    assert "admit_validated" in helper
    # Local import only: no module-level import of the evaluator.
    assert "from .gauge_admission import admit_validated" in helper


def test_gauge_plan_path_forwards_the_credit_and_nothing_else_does():
    decide = _function_source(ENGINE_PATH, "_r55_active_decide")
    assert "gauge_credit=None" in decide
    assert "decision_for_bundle" in decide
    assert "grants_credit" in decide
    assert "gauge_credit=_trial_credit" in decide
    contexts = _function_source(ENGINE_PATH, "_r55_active_cpu_contexts")
    assert "gauge_credit" in contexts and "PlanContext" not in contexts
    context_helpers = _function_source(ENGINE_PATH, "_r55_active_context")
    assert "gauge_credit=gauge_credit" in context_helpers


def test_gauge_compute_is_wrapped_by_open_then_finally_close():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    singles = _function_source(ENGINE_PATH, "_run_single")
    open_at = singles.index("open_for_compute")
    compute_at = singles.index("zphot.compute_global_gauge")
    close_at = singles.index("_gauge_credit_session.close()")
    finally_at = singles.index("finally:", compute_at)
    assert open_at < compute_at < finally_at < close_at
    assert "require_closed_for_stack" in singles
    assert "gauge_credit=_gauge_credit_session" in singles
    # The post-gauge revalidation never receives the credit/session.
    revalidate = _function_source(ENGINE_PATH, "_r55_active_revalidate_launch")
    assert "credit" not in revalidate
    assert "gauge_credit" not in revalidate


def test_credit_module_uses_integer_arithmetic_only():
    tree = ast.parse(CREDIT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float constant at line {node.lineno}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            raise AssertionError(f"true division at line {node.lineno}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
            raise AssertionError(f"floor division at line {node.lineno}")


def test_capacity_is_never_read_through_the_root_cgroup_fallback():
    source = CREDIT_PATH.read_text(encoding="utf-8")
    assert 'read_cgroup_levels("/"' not in source
    assert "read_cgroup_levels('/'," not in source
    # A parse failure must land on `levels = None` (unknown), never on "/".
    parse_block = _function_source(CREDIT_PATH, "collect_live_capacity")
    assert "except ValueError" in parse_block
    assert "levels = None" in parse_block
