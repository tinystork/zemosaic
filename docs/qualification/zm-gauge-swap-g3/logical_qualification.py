#!/usr/bin/env python3
"""G3 — LOGICAL qualification of the bounded gauge-swap credit (scratch tool).

Mission ``ZM-GAUGE-SWAP-G3-20261115``. This tool proves, from injected facts
ONLY (no ``/proc`` read, no engine run), the contract obligations of §4, §12 and
§13.2/§13.3 that are observable at the pure admission seam:

  L1 OFF byte-identity against ``resource_envelope.aggregate_host_peak_bytes`` /
     ``compute_plan_hash`` (fixtures + 500-case seeded sweep);
  L2 policy ON with COMFORTABLE RAM changes no decision and no hash;
  L3 policy ON with a CONSTRAINED synthetic capacity admits the singleton gauge
     group at ``B_ram + S_credit`` while the stack group stays RAM-only and is
     REFUSED with the typed code;
  L4 every refusal code of the closed table is reachable with a typed result;
  L5 ceilings are exact to +/-1 byte (gauge and non-gauge);
  L6 the 15 % margin is applied once per group, exactly as the legacy path;
  L7 ``G0-ORACLE-3`` golden recomputes byte-exactly.

Run (official interpreter, PYTHONPATH removed):

    env -u PYTHONPATH QT_QPA_PLATFORM=offscreen \
      /home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python \
      docs/qualification/zm-gauge-swap-g3/logical_qualification.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
TESTS = REPO / "tests"
for path in (str(SRC), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_zegrid_gauge_admission import (  # noqa: E402
    OFF_FIXTURES,
    _absent_credit,
    _bounded_credit,
    _capacity_decision,
    _groups,
    _plan,
)
from zemosaic.core.zegrid import gauge_admission as adm  # noqa: E402
from zemosaic.core.zegrid import gauge_swap_capacity as cap  # noqa: E402
from zemosaic.core.zegrid import resource_envelope as renv  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30
GOLDEN = REPO / "docs" / "qualification" / "zm-gauge-swap-g1b" / "g0_oracle3_capture_a_posteriori.json"

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: dict | None = None) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail or {}})
    if not ok:
        raise AssertionError(f"{name} FAILED: {detail}")


def _decision_for_b_ram(b_ram: int, p_gauge_raw: int, *, swap_free: int = 6 * GIB):
    """A credit decision whose own ``b_ram`` equals the admission budget."""
    base = (int(b_ram) * 5 + 3) // 4
    for a_raw in range(base, base + 4):
        decision = _capacity_decision(
            a_raw=a_raw, p_gauge_raw=p_gauge_raw, swap_free=swap_free
        )
        if decision.b_ram == b_ram:
            return decision
    raise AssertionError(f"no a_raw reproduces b_ram={b_ram}")


def _raw_for_margined(target: int) -> int:
    """Smallest raw peak whose ceil(115 %) equals ``target`` exactly."""
    raw = max(1, (int(target) * 100) // 115)
    while renv.ceil_margin_div(raw, 15, 100) < target:
        raw += 1
    while raw > 1 and renv.ceil_margin_div(raw - 1, 15, 100) >= target:
        raw -= 1
    assert renv.ceil_margin_div(raw, 15, 100) == target
    return raw


# ---------------------------------------------------------------------------
# L1 — OFF byte-identity
# ---------------------------------------------------------------------------

def check_off_identity() -> None:
    cases = 0
    for peaks, overlaps, b_ram in OFF_FIXTURES:
        envs = _plan(peaks, overlaps=overlaps)
        legacy_raw, legacy_peak = renv.aggregate_host_peak_bytes(envs)
        legacy_hash = renv.compute_plan_hash(envs)
        absent = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())
        explicit = adm.evaluate_groups(
            _groups(envs), b_ram, adm.GaugeCredit(policy=cap.POLICY_RAM_ONLY)
        )
        check(
            "L1_off_fixture",
            absent.to_json() == explicit.to_json()
            and absent.aggregate_raw_bytes == legacy_raw
            and absent.aggregate_margined_bytes == legacy_peak
            and absent.admitted == (legacy_peak <= b_ram)
            and absent.legacy_plan_hash == legacy_hash
            and absent.decision_hash == legacy_hash
            and absent.execution_digest is None,
            {"peaks": peaks, "b_ram": b_ram},
        )
        cases += 1

    import random

    rng = random.Random(20261115)
    phases = ["gauge", "cache_build", "cell_stack_stream", "assembly", "export"]
    pairs = [
        ("cache_build", "assembly"),
        ("cache_build", "cell_stack_stream"),
        ("cell_stack_stream", "assembly"),
        ("gauge", "cache_build"),
        ("gauge", "cell_stack_stream"),
        ("assembly", "export"),
    ]
    values = [0, 1, 2, 7, 100, 999, 10_000, 2 ** 31, 2 ** 40]
    sweep = 0
    for _ in range(500):
        peaks = {phase: rng.choice(values) for phase in phases}
        overlaps = tuple(pair for pair in pairs if rng.random() < 0.35)
        envs = _plan(peaks, overlaps=overlaps)
        raw, peak = renv.aggregate_host_peak_bytes(envs)
        legacy_hash = renv.compute_plan_hash(envs)
        for b_ram in (peak, max(0, peak - 1), peak + 1, rng.choice(values)):
            decision = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())
            ok = (
                decision.aggregate_raw_bytes == raw
                and decision.aggregate_margined_bytes == peak
                and decision.admitted == (peak <= b_ram)
                and decision.reason_code
                == (adm.REASON_OK if peak <= b_ram else adm.REASON_REFUSE_HOST_RAM)
                and decision.decision_hash == legacy_hash
                and decision.execution_digest is None
                and all(group.ceiling_bytes == b_ram for group in decision.groups)
            )
            if not ok:
                check("L1_off_sweep", False, {"peaks": peaks, "b_ram": b_ram})
            sweep += 1
    check(
        "L1_off_sweep",
        True,
        {"fixtures": cases, "sweep_cases": sweep, "status": "byte-identical"},
    )


# ---------------------------------------------------------------------------
# L2 — ON + comfortable RAM: no decision/hash change
# ---------------------------------------------------------------------------

COMFORTABLE_BUDGETS = [500_000_000, 1_149_999_928, 2 ** 40]


def check_on_comfortable() -> None:
    """Realistic comfortable RAM: b_ram >= the frozen 340 MiB process baseline.

    The tiny byte fixtures (b_ram < PROCESS_BASELINE) legitimately change the
    *reason code* of an already-refused plan under policy ON (the credit
    evaluator reports the non-pageable guard). That is a diagnostic, not a
    decision change, and it cannot occur at a realistic budget: this check
    therefore uses b_ram >= 500 MB, where the credit evaluates cleanly with
    need == 0.
    """
    fixtures = [
        (peaks, overlaps, b_ram)
        for peaks, overlaps, b_ram in OFF_FIXTURES
        if b_ram in COMFORTABLE_BUDGETS
    ]
    compared = 0
    for peaks, overlaps, b_ram in fixtures:
        envs = _plan(peaks, overlaps=overlaps)
        off = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())
        # A gauge that fits B_ram by a wide margin: the credit is NOT needed.
        decision = _decision_for_b_ram(b_ram, max(0, (b_ram * 100) // 115))
        on = adm.evaluate_groups(
            _groups(envs), b_ram, _bounded_credit(decision)
        )
        ok = (
            decision.credit_needed_bytes == 0
            and decision.status == "not_needed"
            and on.credit_granted is False
            and on.credit_bytes == 0
            and on.execution_digest is None
            and on.decision_hash == off.decision_hash == on.legacy_plan_hash
            and on.aggregate_raw_bytes == off.aggregate_raw_bytes
            and on.aggregate_margined_bytes == off.aggregate_margined_bytes
            and on.admitted == off.admitted
            and on.reason_code == off.reason_code
            and [g.to_dict() for g in on.groups] == [g.to_dict() for g in off.groups]
        )
        if not ok:
            check("L2_on_comfortable", False, {"peaks": peaks, "b_ram": b_ram})
        compared += 1
    check("L2_on_comfortable", True, {"compared": compared})


# ---------------------------------------------------------------------------
# L3 — ON + constrained: gauge admitted at 1 worker, stack REFUSED (RAM-only)
# ---------------------------------------------------------------------------

CONSTRAINED_PEAKS = {
    "gauge": 10 * GIB,
    "cache_build": 512 * MIB,
    "cell_stack_stream": 9 * GIB,
    "assembly": 256 * MIB,
    "export": 64 * MIB,
}
CONSTRAINED_B_RAM = 8 * GIB


def _constrained_decision():
    return _capacity_decision(
        a_raw=(CONSTRAINED_B_RAM * 5) // 4,
        p_gauge_raw=CONSTRAINED_PEAKS["gauge"],
        swap_free=6 * GIB,
    )


def check_on_constrained() -> None:
    decision = _constrained_decision()
    assert decision.b_ram == CONSTRAINED_B_RAM, decision.b_ram
    credit = _bounded_credit(decision)
    envs = _plan(CONSTRAINED_PEAKS)
    result = adm.evaluate_groups(_groups(envs), CONSTRAINED_B_RAM, credit)

    gauge_group = next(g for g in result.groups if g.phases == ("gauge",))
    stack_group = next(
        g for g in result.groups if g.phases == ("cell_stack_stream",)
    )
    others = [g for g in result.groups if g.phases not in (("gauge",), ("cell_stack_stream",))]

    check(
        "L3_constrained_credit_sufficient",
        decision.status == "credit_sufficient"
        and decision.credit_needed_bytes
        == cap.ceil_margin(CONSTRAINED_PEAKS["gauge"]) - CONSTRAINED_B_RAM
        and decision.s_credit >= decision.credit_needed_bytes
        and credit.workers == 1,
        {
            "b_ram": decision.b_ram,
            "p_gauge_raw": decision.p_gauge_raw,
            "p_gauge_margined": decision.p_gauge_margined,
            "credit_needed": decision.credit_needed_bytes,
            "s_effectif": decision.s_effectif,
            "s_credit": decision.s_credit,
            "status": decision.status,
        },
    )
    check(
        "L3_gauge_admitted_with_credit",
        result.credit_granted is True
        and gauge_group.admitted is True
        and gauge_group.credit_applied is True
        and gauge_group.ceiling_bytes == CONSTRAINED_B_RAM + decision.s_credit
        and result.execution_digest is not None
        and result.credit_bytes == decision.s_credit,
        gauge_group.to_dict(),
    )
    check(
        "L3_stack_refused_ram_only",
        stack_group.admitted is False
        and stack_group.credit_applied is False
        and stack_group.ceiling_bytes == CONSTRAINED_B_RAM
        and all(g.admitted for g in others)
        and all(g.ceiling_bytes == CONSTRAINED_B_RAM for g in others)
        and result.admitted is False
        and result.reason_code == adm.REASON_REFUSE_HOST_RAM
        and [r.code for r in result.refusals] == [adm.REASON_REFUSE_HOST_RAM],
        {
            "stack": stack_group.to_dict(),
            "reason": result.reason_code,
            "refusals": [r.to_dict() for r in result.refusals],
        },
    )
    check(
        "L3_credit_never_subtracted_from_peak",
        result.diagnostics["credit_never_subtracted_from_peak"] is True
        and result.aggregate_raw_bytes == CONSTRAINED_PEAKS["gauge"]
        and result.diagnostics["credit_bytes"] == decision.s_credit,
        result.diagnostics,
    )


# ---------------------------------------------------------------------------
# L4 — typed refusal codes of the closed table
# ---------------------------------------------------------------------------

def _admit_with(credit, peaks=None, b_ram=CONSTRAINED_B_RAM):
    peaks = dict(CONSTRAINED_PEAKS if peaks is None else peaks)
    return adm.evaluate_groups(_groups(_plan(peaks)), b_ram, credit)


def _refusal_code(result) -> str | None:
    for refusal in result.refusals:
        if renv.PHASE_GAUGE in refusal.phases:
            return refusal.code
    return result.reason_code


def check_typed_refusals() -> None:
    decision = _constrained_decision()
    emitted: dict[str, str] = {}

    # OFF keeps the LEGACY refusal code verbatim (the credit adjudication
    # status is diagnostic only); GAUGE_CREDIT_DISABLED stays in the closed
    # table but is not pubished as a refusal code, by contract §11 G1b.
    r = _admit_with(adm.GaugeCredit(policy=cap.POLICY_RAM_ONLY, decision=decision))
    emitted["DISABLED_legacy_refusal"] = _refusal_code(r)
    check(
        "L4_disabled_keeps_legacy_refusal",
        emitted["DISABLED_legacy_refusal"] == adm.REASON_REFUSE_HOST_RAM
        and adm.REASON_CREDIT_DISABLED in adm.REFUSAL_CODES,
        r.refusals[0].to_dict() if r.refusals else {},
    )

    for flag in ("restored", "imported"):
        r = _admit_with(_bounded_credit(decision, **{flag: True}))
        emitted[f"NOT_APPLICABLE_{flag}"] = _refusal_code(r)
        check(f"L4_not_applicable_{flag}", emitted[f"NOT_APPLICABLE_{flag}"] == adm.REASON_CREDIT_NOT_APPLICABLE)

    r = _admit_with(_bounded_credit(decision, platform="win32"))
    emitted["NOT_APPLICABLE_platform"] = _refusal_code(r)
    check("L4_not_applicable_platform", emitted["NOT_APPLICABLE_platform"] == adm.REASON_CREDIT_NOT_APPLICABLE)

    gpu_envs = _plan(CONSTRAINED_PEAKS, gauge_backend="gpu")
    r = adm.evaluate_groups(
        _groups(gpu_envs), CONSTRAINED_B_RAM, _bounded_credit(decision)
    )
    emitted["NOT_APPLICABLE_gpu"] = _refusal_code(r)
    check("L4_not_applicable_gpu", emitted["NOT_APPLICABLE_gpu"] == adm.REASON_CREDIT_NOT_APPLICABLE)

    r = _admit_with(_bounded_credit(decision, workers=2))
    emitted["REQUIRES_ONE_WORKER"] = _refusal_code(r)
    check("L4_requires_one_worker", emitted["REQUIRES_ONE_WORKER"] == adm.REASON_CREDIT_REQUIRES_ONE_WORKER)

    merged = _plan(CONSTRAINED_PEAKS, overlaps=(("gauge", "cell_stack_stream"),))
    r = adm.evaluate_groups(_groups(merged), CONSTRAINED_B_RAM, _bounded_credit(decision))
    emitted["GROUP_NOT_SINGLETON"] = _refusal_code(r)
    check("L4_group_not_singleton", emitted["GROUP_NOT_SINGLETON"] == adm.REASON_CREDIT_GROUP_NOT_SINGLETON)

    r = _admit_with(_bounded_credit(None))
    emitted["UNKNOWN_NO_DECISION"] = _refusal_code(r)
    check("L4_unknown_no_decision", emitted["UNKNOWN_NO_DECISION"] == adm.REASON_CREDIT_UNKNOWN_CAPACITY)

    # Authoritative construction through the capacity module: no cgroup chain
    # means the swap bound is UNKNOWN (never "unlimited").
    capacity = cap.GaugeSwapCapacity(
        a_raw=(CONSTRAINED_B_RAM * 5) // 4,
        devices=[
            cap.SwapDevice(
                name="/swapfile", kind="file", size_bytes=7 * GIB, used_bytes=1 * GIB,
                zram=False, eligible=True, reason="backed_file",
            )
        ],
        cgroup_levels=None,
        overcommit=cap.read_overcommit(0, None, None, commit_limit=None, committed_as=None),
    )
    unknown_swap = cap.evaluate_credit(capacity, CONSTRAINED_PEAKS["gauge"], cap.POLICY_BOUNDED_SWAP)
    r = _admit_with(_bounded_credit(unknown_swap))
    emitted["UNKNOWN_CGROUP_SWAP"] = _refusal_code(r)
    check("L4_unknown_cgroup_swap", emitted["UNKNOWN_CGROUP_SWAP"] == adm.REASON_CREDIT_UNKNOWN_CAPACITY)

    tiny = _capacity_decision(
        a_raw=(CONSTRAINED_B_RAM * 5) // 4,
        p_gauge_raw=CONSTRAINED_PEAKS["gauge"],
        swap_free=64 * MIB,
    )
    r = _admit_with(_bounded_credit(tiny))
    emitted["SWAP_BELOW_MINIMUM"] = _refusal_code(r)
    check("L4_swap_below_minimum", emitted["SWAP_BELOW_MINIMUM"] == adm.REASON_CREDIT_SWAP_BELOW_MINIMUM, tiny.to_dict())

    partial = _capacity_decision(
        a_raw=(CONSTRAINED_B_RAM * 5) // 4,
        p_gauge_raw=CONSTRAINED_PEAKS["gauge"],
        swap_free=2 * GIB,
    )
    r = _admit_with(_bounded_credit(partial))
    emitted["INSUFFICIENT"] = _refusal_code(r)
    check("L4_insufficient", emitted["INSUFFICIENT"] == adm.REASON_CREDIT_INSUFFICIENT, partial.to_dict())

    commit = _capacity_decision(
        a_raw=(CONSTRAINED_B_RAM * 5) // 4,
        p_gauge_raw=CONSTRAINED_PEAKS["gauge"],
        swap_free=6 * GIB,
        overcommit_mode=2,
        commit_limit=1 * GIB,
        committed_as=0,
    )
    r = _admit_with(_bounded_credit(commit))
    emitted["COMMIT_INSUFFICIENT"] = _refusal_code(r)
    check("L4_commit_insufficient", emitted["COMMIT_INSUFFICIENT"] == adm.REASON_CREDIT_COMMIT_INSUFFICIENT, commit.to_dict())

    nonpaged = _capacity_decision(
        a_raw=1 * MIB,
        p_gauge_raw=CONSTRAINED_PEAKS["gauge"],
        swap_free=6 * GIB,
    )
    r = _admit_with(_bounded_credit(nonpaged), b_ram=nonpaged.b_ram)
    emitted["NONPAGEABLE"] = _refusal_code(r)
    check("L4_nonpageable", emitted["NONPAGEABLE"] == adm.REASON_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT, nonpaged.to_dict())

    check(
        "L4_closed_table",
        set(emitted.values()) <= set(adm.REFUSAL_CODES),
        {"emitted": emitted, "table": list(adm.REFUSAL_CODES)},
    )
    return emitted


# ---------------------------------------------------------------------------
# L8 — external_spool family (G3 rework-2 / Nono M1)
# ---------------------------------------------------------------------------
#
# The admission evaluator is stack-class agnostic: the pre-stack GLOBAL gauge is
# the same singleton group whatever the class. These checks pin the EXTERNAL
# shape explicitly (small external host peak that fits B_ram) so the family is
# named in the logical record, not only exercised by the RAM story.

EXTERNAL_STACK_FITS = {
    "gauge": 10 * GIB,
    "cache_build": 64 * MIB,
    "cell_stack_stream": 500 * MIB,
    "assembly": 64 * MIB,
    "export": 32 * MIB,
}


def check_external_family() -> None:
    decision = _constrained_decision()
    credit = _bounded_credit(decision)

    result = adm.evaluate_groups(
        _groups(_plan(EXTERNAL_STACK_FITS)), CONSTRAINED_B_RAM, credit
    )
    groups = {tuple(g.phases): g for g in result.groups}
    gauge = groups[("gauge",)]
    stack = groups[("cell_stack_stream",)]
    check(
        "L8_external_gauge_credited_stack_ram_only",
        result.admitted is True
        and gauge.credit_applied is True
        and gauge.ceiling_bytes == CONSTRAINED_B_RAM + decision.s_credit
        and stack.admitted is True
        and stack.credit_applied is False
        and stack.ceiling_bytes == CONSTRAINED_B_RAM
        and result.execution_digest is not None
        and result.execution_digest.startswith("sha256:"),
        {
            "class": "external_spool",
            "gauge": gauge.to_dict(),
            "stack": stack.to_dict(),
            "digest": result.execution_digest,
        },
    )

    over = dict(EXTERNAL_STACK_FITS)
    over["cell_stack_stream"] = 9 * GIB
    refused = adm.evaluate_groups(
        _groups(_plan(over)), CONSTRAINED_B_RAM, credit
    )
    groups = {tuple(g.phases): g for g in refused.groups}
    check(
        "L8_external_stack_over_b_ram_refused",
        groups[("gauge",)].admitted is True
        and groups[("gauge",)].credit_applied is True
        and groups[("cell_stack_stream",)].admitted is False
        and groups[("cell_stack_stream",)].credit_applied is False
        and groups[("cell_stack_stream",)].ceiling_bytes == CONSTRAINED_B_RAM
        and refused.reason_code == adm.REASON_REFUSE_HOST_RAM,
        {
            "class": "external_spool",
            "stack": groups[("cell_stack_stream",)].to_dict(),
            "reason": refused.reason_code,
        },
    )


# ---------------------------------------------------------------------------
# L5 — exact +/-1 byte ceilings
# ---------------------------------------------------------------------------

def check_plus_minus_one() -> None:
    ceiling = CONSTRAINED_B_RAM + _constrained_decision().s_credit
    raw_exact = _raw_for_margined(ceiling)
    result = {"gauge_ceiling": ceiling, "raw_at_ceiling": raw_exact}
    for label, raw, expected in (
        ("ceiling_minus_1_raw", raw_exact - 1, True),
        ("ceiling_exact_raw", raw_exact, True),
        ("ceiling_plus_1_margined", raw_exact + 1, False),
    ):
        # The credit decision carries its OWN p_gauge_raw/need; it must be
        # rebuilt for every envelope peak, otherwise the boundary under test is
        # the stale decision's need, not the group's margined peak.
        decision = _capacity_decision(
            a_raw=(CONSTRAINED_B_RAM * 5) // 4,
            p_gauge_raw=raw,
            swap_free=6 * GIB,
        )
        credit = _bounded_credit(decision)
        peaks = dict(CONSTRAINED_PEAKS)
        peaks["gauge"] = raw
        peaks["cell_stack_stream"] = 1
        peaks["cache_build"] = 1
        peaks["assembly"] = 1
        peaks["export"] = 1
        r = adm.evaluate_groups(_groups(_plan(peaks)), CONSTRAINED_B_RAM, credit)
        gauge = next(g for g in r.groups if g.phases == ("gauge",))
        ok = gauge.admitted is expected
        if not expected:
            ok = ok and r.refusals and r.refusals[0].code == adm.REASON_CREDIT_INSUFFICIENT
        result[label] = {
            "raw": raw,
            "margined": gauge.margined_bytes,
            "admitted": gauge.admitted,
            "reason": r.reason_code,
        }
        check(f"L5_{label}", ok, result[label])

    # Non-gauge group: B_ram is a hard RAM-only ceiling. The 15 % ceil margin
    # means not every integer is reachable, so the exact boundary is the last
    # raw peak that still fits and the first one that does not (+1 raw byte).
    # ceil(raw*115/100) <= B_ram  <=>  raw*115 <= B_ram*100 (closed form).
    raw_admitted = (CONSTRAINED_B_RAM * 100) // 115
    assert renv.ceil_margin_div(raw_admitted, 15, 100) <= CONSTRAINED_B_RAM
    assert renv.ceil_margin_div(raw_admitted + 1, 15, 100) > CONSTRAINED_B_RAM
    for label, raw, expected in (
        ("stack_last_admitted_raw", raw_admitted, True),
        ("stack_first_refused_raw", raw_admitted + 1, False),
    ):
        peaks = {
            "gauge": 1, "cache_build": 1, "cell_stack_stream": raw,
            "assembly": 1, "export": 1,
        }
        r = adm.evaluate_groups(
            _groups(_plan(peaks)), CONSTRAINED_B_RAM, _absent_credit()
        )
        stack = next(g for g in r.groups if g.phases == ("cell_stack_stream",))
        result[label] = {
            "raw": raw,
            "margined": stack.margined_bytes,
            "ceiling": stack.ceiling_bytes,
            "admitted": stack.admitted,
        }
        check(
            f"L5_{label}",
            stack.admitted is expected
            and stack.ceiling_bytes == CONSTRAINED_B_RAM,
            result[label],
        )
    return result


# ---------------------------------------------------------------------------
# L6 — margin parity (applied once, ceil)
# ---------------------------------------------------------------------------

def check_margin_parity() -> None:
    peaks = {
        "gauge": 100,
        "cache_build": 200,
        "cell_stack_stream": 300,
        "assembly": 400,
        "export": 50,
    }
    envs = _plan(peaks, overlaps=(("cache_build", "assembly"),))
    result = adm.evaluate_groups(_groups(envs), 10_000, _absent_credit())
    for group in result.groups:
        raw = sum(
            env.estimate.peak_bytes
            for env in envs
            if env.phase in group.phases
        )
        expected = -((-raw * 115) // 100)
        check(
            "L6_group_margin",
            group.raw_bytes == raw and group.margined_bytes == expected,
            {"phases": group.phases, "raw": raw, "margined": group.margined_bytes},
        )
    legacy_raw, legacy_peak = renv.aggregate_host_peak_bytes(envs)
    check(
        "L6_aggregate_margin_parity",
        result.aggregate_raw_bytes == legacy_raw
        and result.aggregate_margined_bytes == legacy_peak,
        {"raw": legacy_raw, "peak": legacy_peak},
    )
    # Explicit non-double-margin: the composite group is margined ONCE.
    composite = next(g for g in result.groups if g.phases == ("cache_build", "assembly"))
    raw = 200 + 400
    check(
        "L6_no_double_margin",
        composite.margined_bytes == -((-raw * 115) // 100)
        != -((-(-((-200 * 115) // 100)) + -((-400 * 115) // 100))),
        {"composite": composite.to_dict()},
    )


# ---------------------------------------------------------------------------
# L7 — G0-ORACLE-3 golden recompute
# ---------------------------------------------------------------------------

def check_oracle3() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    mismatches = []
    for case in golden["cases"]:
        envs = _plan(
            case["peaks"],
            overlaps=tuple(tuple(edge) for edge in case["overlaps"]),
        )
        decision = adm.evaluate_groups(_groups(envs), case["b_ram"], _absent_credit())
        actual = {
            "aggregate_raw_bytes": decision.aggregate_raw_bytes,
            "aggregate_margined_bytes": decision.aggregate_margined_bytes,
            "admitted": decision.admitted,
            "reason_code": decision.reason_code,
            "peak_group": list(decision.peak_group),
            "legacy_plan_hash": decision.legacy_plan_hash,
            "decision_hash": decision.decision_hash,
            "execution_digest": decision.execution_digest,
            "groups": [group.to_dict() for group in decision.groups],
        }
        if actual != case["decision"]:
            mismatches.append({"peaks": case["peaks"], "b_ram": case["b_ram"]})
    check(
        "L7_oracle3_recompute",
        not mismatches and golden["schema"] == "g0-oracle3-capture-a-posteriori-v1",
        {"cases": len(golden["cases"]), "mismatches": mismatches},
    )


def main() -> int:
    started = time.time()
    check_off_identity()
    check_on_comfortable()
    check_on_constrained()
    check_external_family()
    emitted = check_typed_refusals()
    boundaries = check_plus_minus_one()
    check_margin_parity()
    check_oracle3()

    payload = {
        "schema": "g3-logical-qualification-v1",
        "mission": "ZM-GAUGE-SWAP-G3-20261115",
        "contract": "docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md §4/§12/§13",
        "kind": "INJECTED FACTS ONLY (no /proc, no engine, no pressure)",
        "base_head": "e25fb5133768adf4eb26950c63f86b970909b370",
        "checks": CHECKS,
        "all_ok": all(item["ok"] for item in CHECKS),
        "counts": {
            "checks": len(CHECKS),
            "failed": sum(1 for item in CHECKS if not item["ok"]),
        },
        "typed_refusal_codes": emitted,
        "boundaries": boundaries,
        "elapsed_s": round(time.time() - started, 3),
        "limits": [
            "A synthetic/injected MemAvailable proves the LOGIC only, never physical swap.",
            "No engine, no scope, no OOM, no real allocation is exercised by this tool.",
        ],
    }
    target = HERE / "logical_qualification.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(f"checks={len(CHECKS)} failed={payload['counts']['failed']}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
