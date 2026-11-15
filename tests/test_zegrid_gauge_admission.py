"""G1b — pure per-group admission tests (schema ``zegrid-gauge-admission-v1``).

Scope: :mod:`zemosaic.core.zegrid.gauge_admission` ONLY. No product module is
imported or executed, no ``/proc`` is read and no engine run is started: every
capacity fact is injected as a G1a :class:`CreditDecision`, every phase peak is
injected as a :class:`ResourceEnvelope`.

The behavioral oracles are:

* ``resource_envelope.aggregate_host_peak_bytes`` / ``compute_plan_hash`` for
  the legacy (OFF) path — byte-identical by contract (§11 G1b, §12 G0-ORACLE-3);
* the frozen G0 formula §4.7 (sum intra-group, max inter-groups, one 15 %
  margin per group, per-group ceiling ``B_ram`` / ``B_ram + S_credit``).
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import pytest

from zemosaic.core.zegrid import gauge_admission as adm
from zemosaic.core.zegrid import gauge_swap_capacity as cap
from zemosaic.core.zegrid import resource_envelope as renv

MIB = 1024 ** 2
GIB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
MODULE_PATH = SRC / "zemosaic" / "core" / "zegrid" / "gauge_admission.py"
GOLDEN_PATH = (
    REPO_ROOT / "docs" / "qualification" / "zm-gauge-swap-g1b"
    / "g0_oracle3_capture_a_posteriori.json"
)


# ---------------------------------------------------------------------------
# Envelope fixtures (same shape as tests/test_zegrid_r55_envelope.py)
# ---------------------------------------------------------------------------

def _host():
    raw = 1000
    return renv.HostEnvelope(
        observation_source="test.available_bytes",
        raw_available_bytes=raw,
        factor_num=4,
        factor_den=5,
        budget_bytes=(4 * raw) // 5,
        reserve_bytes=0,
        process_baseline_bytes=100,
    )


def _device():
    return renv.DeviceEnvelope(
        vram_total_bytes=8 * GIB,
        vram_free_bytes=7 * GIB,
        vram_reserved_estimate_bytes=1 * GIB,
        budget_bytes=6 * GIB,
        device_name="test-gpu",
    )


def _env(phase, peak, *, overlaps=(), backend="cpu"):
    estimate = renv.make_estimate({f"{phase}_bytes": int(peak)})
    return renv.ResourceEnvelope(
        envelope_id=f"{phase}.{backend}",
        phase=phase,
        backend=backend,
        host=_host(),
        device=_device() if backend == "gpu" else None,
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


def _symmetric(edges):
    out = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def _plan(peaks, *, overlaps=(), stack_phase="cell_stack_stream", gauge_backend="cpu"):
    """Build a full PLAN set (gauge, cache, one stack, assembly, export)."""
    peaks = dict(peaks)
    adj = _symmetric(overlaps)
    phases = ["gauge", "cache_build", stack_phase, "assembly", "export"]
    return [
        _env(
            phase,
            peaks[phase],
            overlaps=tuple(sorted(adj.get(phase, ()))),
            backend=(gauge_backend if phase == "gauge" else "cpu"),
        )
        for phase in phases
    ]


def _groups(envs):
    return adm.groups_from_envelopes(envs)


def _absent_credit():
    """Policy key absent: treated exactly like ram_only for decisions/hashes."""
    return adm.GaugeCredit(policy=None)


# ---------------------------------------------------------------------------
# Injected G1a credit decisions (no /proc, no file read)
# ---------------------------------------------------------------------------

def _capacity_decision(
    *,
    a_raw=10 * GIB,
    p_gauge_raw=10 * GIB,
    swap_free=6 * GIB,
    overcommit_mode=0,
    commit_limit=None,
    committed_as=None,
):
    device = cap.SwapDevice(
        name="/swapfile",
        kind="file",
        size_bytes=int(swap_free) + 1 * MIB,
        used_bytes=1 * MIB,
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
    overcommit = cap.read_overcommit(
        overcommit_mode,
        None,
        None,
        commit_limit=commit_limit,
        committed_as=committed_as,
    )
    capacity = cap.GaugeSwapCapacity(
        a_raw=int(a_raw),
        devices=[device],
        cgroup_levels=[level],
        overcommit=overcommit,
    )
    return cap.evaluate_credit(capacity, int(p_gauge_raw), cap.POLICY_BOUNDED_SWAP)


def _bounded_credit(decision, **overrides):
    kwargs = dict(decision=decision, policy=cap.POLICY_BOUNDED_SWAP)
    kwargs.update(overrides)
    return adm.GaugeCredit(**kwargs)


GRANTED_DECISION = _capacity_decision()
GRANTED_CREDIT = _bounded_credit(GRANTED_DECISION)
GRANTED_B_RAM = GRANTED_DECISION.b_ram

#: Comfortable RAM: the gauge fits B_ram, so the credit is NOT needed.
COMFORTABLE_DECISION = _capacity_decision(p_gauge_raw=1)
COMFORTABLE_CREDIT = _bounded_credit(COMFORTABLE_DECISION)
COMFORTABLE_PEAKS = {"gauge": 1, "cache_build": 1, "cell_stack_stream": 1,
                     "assembly": 1, "export": 1}


# ---------------------------------------------------------------------------
# (1) Policy OFF is byte-identical to the current path, over N fixtures
# ---------------------------------------------------------------------------

OFF_FIXTURES = [
    # (peaks, overlaps, b_ram)
    ({"gauge": 100, "cache_build": 200, "cell_stack_stream": 300,
      "assembly": 400, "export": 50}, (), 500),
    ({"gauge": 1, "cache_build": 1, "cell_stack_stream": 1,
      "assembly": 1, "export": 1}, (), 1),
    ({"gauge": 10_000, "cache_build": 20_000, "cell_stack_stream": 30_000,
      "assembly": 40_000, "export": 5_000}, (), 100_000),
    ({"gauge": 7, "cache_build": 11, "cell_stack_stream": 13,
      "assembly": 17, "export": 19}, (), 57),
    ({"gauge": 7, "cache_build": 11, "cell_stack_stream": 13,
      "assembly": 17, "export": 19}, (), 56),
    ({"gauge": 123_456_789, "cache_build": 98_765_432,
      "cell_stack_stream": 111_111_111, "assembly": 222_222_222,
      "export": 33_333_333}, (), 500_000_000),
    ({"gauge": 100, "cache_build": 100, "cell_stack_stream": 100,
      "assembly": 100, "export": 100},
     (("cache_build", "assembly"),), 460),
    ({"gauge": 100, "cache_build": 100, "cell_stack_stream": 100,
      "assembly": 100, "export": 100},
     (("cache_build", "assembly"),), 459),
    ({"gauge": 50, "cache_build": 60, "cell_stack_stream": 70,
      "assembly": 80, "export": 90},
     (("cache_build", "cell_stack_stream"), ("cell_stack_stream", "assembly")), 322),
    ({"gauge": 3, "cache_build": 5, "cell_stack_stream": 7,
      "assembly": 11, "export": 13}, (), 44),
    ({"gauge": 0, "cache_build": 0, "cell_stack_stream": 0,
      "assembly": 0, "export": 0}, (), 0),
    ({"gauge": 999_999_937, "cache_build": 1, "cell_stack_stream": 1,
      "assembly": 1, "export": 1}, (), 1_149_999_928),
    ({"gauge": 999_999_937, "cache_build": 1, "cell_stack_stream": 1,
      "assembly": 1, "export": 1}, (), 1_149_999_927),
    ({"gauge": 1, "cache_build": 2, "cell_stack_stream": 3,
      "assembly": 4, "export": 5},
     (("gauge", "cache_build"),), 17),
    ({"gauge": 1, "cache_build": 2, "cell_stack_stream": 3,
      "assembly": 4, "export": 5}, (), 100),
    ({"gauge": 2**40, "cache_build": 2**39, "cell_stack_stream": 2**38,
      "assembly": 2**37, "export": 2**36}, (), 2**40),
]


@pytest.mark.parametrize("peaks,overlaps,b_ram", OFF_FIXTURES)
def test_policy_off_matches_legacy_aggregate_and_hash(peaks, overlaps, b_ram):
    envs = _plan(peaks, overlaps=overlaps)
    legacy_raw, legacy_peak = renv.aggregate_host_peak_bytes(envs)
    legacy_hash = renv.compute_plan_hash(envs)

    decision = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())

    # Numbers: sum intra-group / max inter-groups / one margin.
    assert decision.aggregate_raw_bytes == legacy_raw
    assert decision.aggregate_margined_bytes == legacy_peak
    # Verdict is the current comparison, byte for byte.
    assert decision.admitted == (legacy_peak <= b_ram)
    assert decision.reason_code == (
        adm.REASON_OK if legacy_peak <= b_ram else adm.REASON_REFUSE_HOST_RAM
    )
    # Hashes: no new tolerance, no new payload.
    assert decision.legacy_plan_hash == legacy_hash
    assert decision.scientific_plan_hash == legacy_hash
    assert decision.decision_hash == legacy_hash
    assert decision.execution_digest is None
    assert decision.policy == cap.POLICY_RAM_ONLY
    # Every per-group ceiling is B_ram: no credit is applied when OFF.
    assert all(group.ceiling_bytes == b_ram for group in decision.groups)
    assert decision.credit_granted is False
    assert decision.credit_bytes == 0
    # The true aggregate peak is published, never reduced.
    assert decision.diagnostics["aggregate_raw_bytes"] == legacy_raw
    assert decision.diagnostics["credit_never_subtracted_from_peak"] is True


def test_policy_off_winning_group_matches_independent_argmax():
    peaks = {"gauge": 10, "cache_build": 20, "cell_stack_stream": 30,
             "assembly": 40, "export": 5}
    envs = _plan(peaks)
    expected_groups = renv.overlap_groups(envs)
    per_phase = {env.phase: env.estimate.peak_bytes for env in envs}
    expected_winner = min(
        (group for group in expected_groups
         if sum(per_phase[p] for p in group) == max(
             sum(per_phase[q] for q in g) for g in expected_groups)),
        key=expected_groups.index,
    )

    decision = adm.evaluate_groups(_groups(envs), 1000, _absent_credit())

    assert decision.peak_group == tuple(expected_winner)
    assert decision.peak_group == ("assembly",)


def test_ram_only_policy_string_is_identical_to_absent_policy():
    envs = _plan({"gauge": 100, "cache_build": 200, "cell_stack_stream": 300,
                  "assembly": 400, "export": 50})
    absent = adm.evaluate_groups(_groups(envs), 600, _absent_credit())
    explicit = adm.evaluate_groups(
        _groups(envs), 600, adm.GaugeCredit(policy=cap.POLICY_RAM_ONLY)
    )
    assert absent.to_json() == explicit.to_json()


def test_policy_off_equality_on_seeded_sweep():
    """L6: a small deterministic sweep, versioned in the suite."""
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
    values = [0, 1, 2, 7, 100, 999, 10_000, 2**31, 2**40]
    for _ in range(120):
        peaks = {phase: rng.choice(values) for phase in phases}
        overlaps = tuple(pair for pair in pairs if rng.random() < 0.35)
        envs = _plan(peaks, overlaps=overlaps)
        raw, peak = renv.aggregate_host_peak_bytes(envs)
        legacy_hash = renv.compute_plan_hash(envs)
        for b_ram in (peak, max(0, peak - 1), peak + 1, rng.choice(values)):
            decision = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())
            assert decision.aggregate_raw_bytes == raw
            assert decision.aggregate_margined_bytes == peak
            assert decision.admitted == (peak <= b_ram)
            assert decision.reason_code == (
                adm.REASON_OK if peak <= b_ram else adm.REASON_REFUSE_HOST_RAM
            )
            assert decision.decision_hash == legacy_hash
            assert decision.execution_digest is None


def test_golden_g0_oracle3_capture_a_posteriori_matches_recomputation():
    """L6: the post-hoc G0-ORACLE-3 golden still recomputes exactly."""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["schema"] == "g0-oracle3-capture-a-posteriori-v1"
    assert golden["oracle"] == "G0-ORACLE-3 (decision OFF)"
    assert golden["kind"] == "a_posteriori"
    assert "NOT captured before G1b" in golden["note"]
    assert golden["policy"] == "absent (== ram_only)"

    cases = golden["cases"]
    assert len(cases) == len(OFF_FIXTURES)
    for case, (peaks, overlaps, b_ram) in zip(cases, OFF_FIXTURES):
        assert case["peaks"] == peaks
        assert case["b_ram"] == b_ram
        assert [tuple(edge) for edge in case["overlaps"]] == [tuple(e) for e in overlaps]

        envs = _plan(peaks, overlaps=overlaps)
        decision = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())
        recomputed = {
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
        assert recomputed == case["decision"]


# ---------------------------------------------------------------------------
# (2) The 15 % margin is applied exactly ONCE per group
# ---------------------------------------------------------------------------

def test_margin_applied_once_per_group_after_intra_group_sum():
    # A 3-phase overlap group with 1 B each: sum first, then ONE margin.
    envs = _plan(
        {"gauge": 4, "cache_build": 1, "cell_stack_stream": 1,
         "assembly": 1, "export": 100},
        overlaps=(("cache_build", "cell_stack_stream"), ("cell_stack_stream", "assembly")),
    )
    decision = adm.evaluate_groups(_groups(envs), 10_000, _absent_credit())
    merged = next(
        group for group in decision.groups
        if group.phases == ("cache_build", "cell_stack_stream", "assembly")
    )
    assert merged.raw_bytes == 3
    # Exactly one margin on the group sum: ceil(3 * 115 / 100) == 4.
    # One margin PER PHASE would give ceil(1.15) * 3 == 6 (double reserve).
    assert merged.margined_bytes == 4


def test_margin_not_applied_per_phase():
    envs = _plan(
        {"gauge": 1, "cache_build": 1, "cell_stack_stream": 1,
         "assembly": 1, "export": 1},
        overlaps=(("cache_build", "assembly"),),
    )
    decision = adm.evaluate_groups(_groups(envs), 1000, _absent_credit())
    merged = next(g for g in decision.groups if g.phases == ("cache_build", "assembly"))
    # sum(1,1) = 2 -> ceil(2.3) = 3. Applying the margin per phase would be
    # ceil(1.15) + ceil(1.15) = 2 + 2 = 4 (double reserve).
    assert merged.raw_bytes == 2
    assert merged.margined_bytes == 3


def test_margin_matches_legacy_on_composite_group():
    envs = _plan(
        {"gauge": 37, "cache_build": 1, "cell_stack_stream": 2,
         "assembly": 3, "export": 4},
        overlaps=(("cache_build", "assembly"),),
    )
    decision = adm.evaluate_groups(_groups(envs), 10_000, _absent_credit())
    _raw, peak = renv.aggregate_host_peak_bytes(envs)
    merged = next(g for g in decision.groups if g.phases == ("cache_build", "assembly"))
    assert merged.margined_bytes == renv.ceil_margin_div(merged.raw_bytes, 15, 100)
    assert decision.aggregate_margined_bytes == peak


# ---------------------------------------------------------------------------
# (3) Exact threshold: accepted AT ceiling, refused AT ceiling + 1 B
# ---------------------------------------------------------------------------

def _ceiling_for(peak):
    return renv.ceil_margin_div(peak, 15, 100)


@pytest.mark.parametrize("target", ["gauge", "cache_build", "assembly", "export",
                                    "cell_stack_stream"])
def test_threshold_exact_at_ceiling_and_plus_one_byte(target):
    peaks = {"gauge": 1, "cache_build": 1, "cell_stack_stream": 1,
             "assembly": 1, "export": 1}
    peaks[target] = 100
    envs = _plan(peaks)
    ceiling = _ceiling_for(100)                       # ceil(115) == 115
    assert ceiling == 115

    accepted = adm.evaluate_groups(_groups(envs), ceiling, _absent_credit())
    assert accepted.admitted is True, accepted.to_json()
    assert accepted.reason_code == adm.REASON_OK
    target_group = next(g for g in accepted.groups if target in g.phases)
    assert target_group.margined_bytes == ceiling
    assert target_group.admitted is True

    refused = adm.evaluate_groups(_groups(envs), ceiling - 1, _absent_credit())
    assert refused.admitted is False
    refused_group = next(g for g in refused.groups if target in g.phases)
    assert refused_group.margined_bytes == ceiling      # 115 > 114
    assert refused_group.ceiling_bytes == ceiling - 1
    assert refused_group.admitted is False
    assert refused.reason_code == adm.REASON_REFUSE_HOST_RAM
    assert [(r.actor, r.code) for r in refused.refusals] == [
        (target, adm.REASON_REFUSE_HOST_RAM)
    ]


def test_threshold_off_is_identical_to_legacy_plus_one_byte():
    peaks = {"gauge": 100, "cache_build": 1, "cell_stack_stream": 1,
             "assembly": 1, "export": 1}
    envs = _plan(peaks)
    _raw, peak = renv.aggregate_host_peak_bytes(envs)
    at = adm.evaluate_groups(_groups(envs), peak, _absent_credit())
    above = adm.evaluate_groups(_groups(envs), peak - 1, _absent_credit())
    assert at.admitted is True
    assert above.admitted is False
    assert at.aggregate_margined_bytes == above.aggregate_margined_bytes == peak


# ---------------------------------------------------------------------------
# (4) Credit refusals: non-singleton / restored / imported / workers / platform
# ---------------------------------------------------------------------------

GAUGE_PEAKS = {"gauge": 10 * GIB, "cache_build": 1, "cell_stack_stream": 1,
               "assembly": 1, "export": 1}


def _assert_gauge_needs_credit(decision):
    """Sanity: the gauge genuinely exceeds B_ram, so eligibility is load-bearing."""
    gauge = next(g for g in decision.groups if g.contains_gauge)
    assert gauge.raw_bytes > decision.b_ram_bytes
    assert gauge.ceiling_bytes == decision.b_ram_bytes
    assert gauge.admitted is False


def test_credit_granted_for_eligible_singleton_gauge():
    envs = _plan(GAUGE_PEAKS)
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)

    assert decision.credit_granted is True
    assert decision.credit_refusal_code is None
    assert decision.credit_bytes == GRANTED_DECISION.s_credit > 0
    gauge = next(g for g in decision.groups if g.is_gauge)
    assert gauge.credit_applied is True
    assert gauge.ceiling_bytes == GRANTED_B_RAM + GRANTED_DECISION.s_credit
    assert decision.admitted is True, decision.to_json()
    # The true aggregate peak is NOT reduced by the credit.
    assert decision.aggregate_raw_bytes == 10 * GIB
    assert decision.diagnostics["aggregate_raw_bytes"] == 10 * GIB
    assert decision.diagnostics["credit_bytes"] == GRANTED_DECISION.s_credit


def test_m1_zero_need_never_grants_unused_credit():
    """Rework M1: with swap present but need == 0, no credit is granted."""
    envs = _plan(COMFORTABLE_PEAKS)
    decision = adm.evaluate_groups(
        _groups(envs), COMFORTABLE_DECISION.b_ram, COMFORTABLE_CREDIT
    )

    # The capacity provider WOULD have granted a large credit (swap present).
    assert COMFORTABLE_DECISION.s_credit > 0
    assert COMFORTABLE_DECISION.credit_needed_bytes == 0
    assert COMFORTABLE_DECISION.status == "not_needed"

    assert decision.credit_needed_bytes == 0
    assert decision.credit_granted is False
    assert decision.credit_bytes == 0
    assert decision.credit_refusal_code is None
    assert decision.credit_refusal_detail == "credit_not_needed"
    assert decision.credit_status == "not_needed"
    gauge = next(group for group in decision.groups if group.is_gauge)
    assert gauge.ceiling_bytes == decision.b_ram_bytes
    assert gauge.credit_applied is False
    assert decision.admitted is True
    assert decision.diagnostics["credit_bytes"] == 0
    assert decision.diagnostics["credit_granted"] is False


def test_m2_policy_on_comfortable_ram_keeps_legacy_hash():
    """Rework M2: no applied credit => no execution digest, legacy hash."""
    envs = _plan(COMFORTABLE_PEAKS)
    decision = adm.evaluate_groups(
        _groups(envs), COMFORTABLE_DECISION.b_ram, COMFORTABLE_CREDIT
    )

    assert decision.policy == cap.POLICY_BOUNDED_SWAP
    assert decision.execution_digest is None
    assert decision.decision_hash == decision.legacy_plan_hash
    assert decision.decision_hash == renv.compute_plan_hash(envs)
    # Policy and credit status stay published for observability.
    assert decision.to_dict()["policy"] == cap.POLICY_BOUNDED_SWAP
    assert decision.to_dict()["credit"]["status"] == "not_needed"


def test_l3_missing_decision_b_ram_with_known_a_is_invalid():
    import dataclasses

    broken = dataclasses.replace(GRANTED_DECISION, b_ram=None)
    with pytest.raises(adm.InvalidCredit):
        adm.evaluate_groups(
            _groups(_plan(GAUGE_PEAKS)),
            GRANTED_B_RAM,
            adm.GaugeCredit(decision=broken, policy=cap.POLICY_BOUNDED_SWAP),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"restored": True, "workers": True},
        {"restored": True, "platform": 42},
        {"imported": True, "platform": ""},
        {"workers": 0},
        {"imported": 1},
    ],
)
def test_l4_malformed_credit_fields_raise_before_business_refusal(overrides):
    with pytest.raises(adm.InvalidCredit):
        adm.evaluate_groups(
            _groups(_plan(GAUGE_PEAKS)),
            GRANTED_B_RAM,
            _bounded_credit(GRANTED_DECISION, **overrides),
        )


def test_credit_refused_when_gauge_group_not_singleton():
    envs = _plan(GAUGE_PEAKS, overlaps=(("gauge", "cache_build"),))
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)

    assert decision.credit_granted is False
    assert decision.credit_bytes == 0
    assert decision.credit_refusal_code == adm.REASON_CREDIT_GROUP_NOT_SINGLETON
    _assert_gauge_needs_credit(decision)
    assert decision.admitted is False
    assert decision.reason_code == adm.REASON_CREDIT_GROUP_NOT_SINGLETON
    assert decision.credit_status == "credit_sufficient"   # capacity itself was fine


@pytest.mark.parametrize("flag", ["restored", "imported"])
def test_credit_refused_when_gauge_restored_or_imported(flag):
    envs = _plan(GAUGE_PEAKS)
    credit = _bounded_credit(GRANTED_DECISION, **{flag: True})
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, credit)

    assert decision.credit_granted is False
    assert decision.credit_refusal_code == adm.REASON_CREDIT_NOT_APPLICABLE
    assert decision.credit_refusal_detail == f"gauge_{flag}"
    _assert_gauge_needs_credit(decision)
    assert decision.admitted is False
    assert decision.reason_code == adm.REASON_CREDIT_NOT_APPLICABLE


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_credit_refused_when_workers_not_one(workers):
    envs = _plan(GAUGE_PEAKS)
    credit = _bounded_credit(GRANTED_DECISION, workers=workers)
    if workers < 1:
        with pytest.raises(adm.InvalidCredit):
            adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, credit)
        return
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, credit)
    assert decision.credit_granted is False
    assert decision.credit_refusal_code == adm.REASON_CREDIT_REQUIRES_ONE_WORKER
    _assert_gauge_needs_credit(decision)
    assert decision.reason_code == adm.REASON_CREDIT_REQUIRES_ONE_WORKER


@pytest.mark.parametrize("platform", ["win32", "Windows", "darwin", "freebsd"])
def test_credit_refused_on_non_linux_platform(platform):
    envs = _plan(GAUGE_PEAKS)
    credit = _bounded_credit(GRANTED_DECISION, platform=platform)
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, credit)
    assert decision.credit_granted is False
    assert decision.credit_refusal_code == adm.REASON_CREDIT_NOT_APPLICABLE
    assert decision.credit_refusal_detail == "platform_not_linux"
    _assert_gauge_needs_credit(decision)
    assert decision.reason_code == adm.REASON_CREDIT_NOT_APPLICABLE


def test_credit_not_applicable_for_gpu_gauge():
    envs = _plan(GAUGE_PEAKS, gauge_backend="gpu")
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)
    assert decision.credit_granted is False
    assert decision.credit_refusal_code == adm.REASON_CREDIT_NOT_APPLICABLE
    assert decision.credit_refusal_detail == "gauge_not_compute_backend"
    assert decision.admitted is False


@pytest.mark.parametrize(
    "decision_factory,expected_code,expected_detail",
    [
        (lambda: None,
         adm.REASON_CREDIT_UNKNOWN_CAPACITY, "credit_decision_missing"),
        (lambda: _capacity_decision(swap_free=512 * MIB),
         adm.REASON_CREDIT_SWAP_BELOW_MINIMUM, "effective_swap_below_minimum"),
        (lambda: _capacity_decision(swap_free=3 * GIB),
         adm.REASON_CREDIT_INSUFFICIENT, "credit_below_credit_needed"),
        (lambda: _capacity_decision(
            overcommit_mode=2, commit_limit=1 * GIB, committed_as=0),
         adm.REASON_CREDIT_COMMIT_INSUFFICIENT, "commit_headroom_insufficient"),
        (lambda: _capacity_decision(a_raw=125 * MIB),
         adm.REASON_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT,
         "b_ram_below_non_pageable_new"),
    ],
)
def test_credit_refused_for_capacity_codes(decision_factory, expected_code, expected_detail):
    decision = decision_factory()
    credit = adm.GaugeCredit(decision=decision, policy=cap.POLICY_BOUNDED_SWAP)
    b_ram = GRANTED_B_RAM if decision is None or decision.b_ram is None else decision.b_ram
    envs = _plan(GAUGE_PEAKS)
    result = adm.evaluate_groups(_groups(envs), b_ram, credit)

    assert result.credit_granted is False
    assert result.credit_refusal_code == expected_code
    assert result.credit_refusal_detail == expected_detail
    gauge = next(g for g in result.groups if g.is_gauge)
    assert gauge.ceiling_bytes == b_ram
    assert gauge.admitted is False
    assert result.admitted is False
    assert result.reason_code == expected_code


def test_credit_refused_when_effective_swap_unknown():
    capacity = cap.GaugeSwapCapacity(
        a_raw=10 * GIB,
        devices=[cap.SwapDevice("/swapfile", "file", 6 * GIB, 0, False, True, "ok")],
        cgroup_levels=None,                      # swap cgroup chain unknown
        overcommit=cap.read_overcommit(0, None, None),
    )
    decision = cap.evaluate_credit(capacity, 10 * GIB, cap.POLICY_BOUNDED_SWAP)
    assert decision.cgroup_swap_status == cap.UNKNOWN
    result = adm.evaluate_groups(
        _groups(_plan(GAUGE_PEAKS)),
        GRANTED_B_RAM,
        adm.GaugeCredit(decision=decision, policy=cap.POLICY_BOUNDED_SWAP),
    )
    assert result.credit_refusal_code == adm.REASON_CREDIT_UNKNOWN_CAPACITY
    assert result.credit_refusal_detail == "effective_swap_unknown"


def test_credit_refused_when_mem_available_unknown():
    capacity = cap.GaugeSwapCapacity(
        a_raw=None,
        devices=[cap.SwapDevice("/swapfile", "file", 6 * GIB, 0, False, True, "ok")],
        cgroup_levels=[],
        overcommit=cap.read_overcommit(0, None, None),
    )
    decision = cap.evaluate_credit(capacity, 10 * GIB, cap.POLICY_BOUNDED_SWAP)
    result = adm.evaluate_groups(
        _groups(_plan(GAUGE_PEAKS)),
        GRANTED_B_RAM,
        adm.GaugeCredit(decision=decision, policy=cap.POLICY_BOUNDED_SWAP),
    )
    assert result.credit_refusal_code == adm.REASON_CREDIT_UNKNOWN_CAPACITY
    assert result.credit_refusal_detail == "mem_available_unknown"


# ---------------------------------------------------------------------------
# (5) The credit never saves stack / cache / assembly / export
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phase", ["cache_build", "cell_stack_stream", "assembly", "export"])
def test_credit_never_saves_non_gauge_phase(phase):
    peaks = {"gauge": 10 * GIB, "cache_build": 1, "cell_stack_stream": 1,
             "assembly": 1, "export": 1}
    peaks[phase] = 9 * GIB                    # > B_ram (8 GiB), so it cannot fit
    envs = _plan(peaks)
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)

    assert decision.credit_granted is True     # credit exists for the gauge
    gauge = next(g for g in decision.groups if g.is_gauge)
    assert gauge.admitted is True
    target_group = next(g for g in decision.groups if phase in g.phases)
    assert target_group.ceiling_bytes == GRANTED_B_RAM      # never B_ram + credit
    assert target_group.credit_applied is False
    assert target_group.margined_bytes > GRANTED_B_RAM
    assert target_group.admitted is False
    assert decision.admitted is False
    assert decision.reason_code == adm.REASON_REFUSE_HOST_RAM
    assert [(r.actor, r.code) for r in decision.refusals] == [
        (phase, adm.REASON_REFUSE_HOST_RAM)
    ]


def test_credit_never_saves_a_merged_gauge_stack_group():
    # A gauge that overlaps the stack is refused, never subsidised.
    peaks = {"gauge": 10 * GIB, "cache_build": 1, "cell_stack_stream": 1,
             "assembly": 1, "export": 1}
    envs = _plan(peaks, overlaps=(("gauge", "cell_stack_stream"),))
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)
    merged = next(g for g in decision.groups if "gauge" in g.phases)
    assert merged.phases == ("gauge", "cell_stack_stream")
    assert merged.ceiling_bytes == GRANTED_B_RAM
    assert merged.credit_applied is False
    assert decision.reason_code == adm.REASON_CREDIT_GROUP_NOT_SINGLETON


# ---------------------------------------------------------------------------
# (6) Hashes: legacy preserved OFF, versioned digest ON, science unchanged
# ---------------------------------------------------------------------------

def test_execution_digest_is_versioned_and_science_hash_unchanged():
    envs = _plan(GAUGE_PEAKS)
    off = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, _absent_credit())
    on = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)

    assert off.scientific_plan_hash == on.scientific_plan_hash
    assert off.legacy_plan_hash == on.legacy_plan_hash == renv.compute_plan_hash(envs)
    assert off.execution_digest is None
    assert on.execution_digest is not None
    assert on.execution_digest.startswith("sha256:")
    assert len(on.execution_digest) == len("sha256:") + 64
    assert on.execution_digest != on.legacy_plan_hash
    assert on.decision_hash == on.execution_digest
    assert off.decision_hash == off.legacy_plan_hash


def test_execution_digest_is_stable_and_decision_sensitive():
    envs = _plan(GAUGE_PEAKS)
    first = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)
    second = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)
    assert first.execution_digest == second.execution_digest

    other_plan = dict(GAUGE_PEAKS)
    other_plan["gauge"] = 9 * GIB
    # The decision is repriced for the SAME 9 GiB request: a decision left on
    # the 10 GiB peak is a stale record and is refused (L6, rework-2).
    other_decision = _capacity_decision(p_gauge_raw=9 * GIB)
    other = adm.evaluate_groups(
        _groups(_plan(other_plan)), GRANTED_B_RAM, _bounded_credit(other_decision)
    )
    assert other.execution_digest != first.execution_digest

    refused = adm.evaluate_groups(
        _groups(envs),
        GRANTED_B_RAM,
        _bounded_credit(GRANTED_DECISION, workers=2),
    )
    assert refused.execution_digest != first.execution_digest


def test_stale_credit_decision_p_gauge_is_refused():
    """L6 defence: a decision priced for another gauge peak never widens this one."""
    small_gauge = dict(GAUGE_PEAKS)
    small_gauge["gauge"] = 100 * MIB
    stale = _capacity_decision(p_gauge_raw=9 * GIB)
    with pytest.raises(adm.InvalidCredit) as err:
        adm.evaluate_groups(
            _groups(_plan(small_gauge)), GRANTED_B_RAM, _bounded_credit(stale)
        )
    assert "stale gauge credit decision" in str(err.value)
    # The SAME request with a decision priced for it is admitted normally.
    fresh = _capacity_decision(p_gauge_raw=100 * MIB)
    admitted = adm.evaluate_groups(
        _groups(_plan(small_gauge)), GRANTED_B_RAM, _bounded_credit(fresh)
    )
    assert admitted.groups[0].admitted is True


def test_digest_payload_carries_no_volatile_observation():
    # The digest is audit-only and must not embed timestamps or raw observations.
    envs = _plan(GAUGE_PEAKS)
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, GRANTED_CREDIT)
    payload = decision.to_dict()
    text = json.dumps(payload, sort_keys=True)
    assert "observed_at" not in text
    assert decision.execution_digest == adm.evaluate_groups(
        _groups(envs), GRANTED_B_RAM, GRANTED_CREDIT
    ).execution_digest


# ---------------------------------------------------------------------------
# (7) Typed refusals: closed table, no generic except, fail-closed inputs
# ---------------------------------------------------------------------------

def test_refusal_table_is_closed_and_versioned():
    assert adm.REASON_OK == "R55_OK"
    expected = {
        "R55_REFUSE_HOST_RAM",
        "GAUGE_CREDIT_DISABLED",
        "GAUGE_CREDIT_NOT_APPLICABLE",
        "GAUGE_CREDIT_REQUIRES_ONE_WORKER",
        "GAUGE_CREDIT_GROUP_NOT_SINGLETON",
        "GAUGE_CREDIT_UNKNOWN_CAPACITY",
        "GAUGE_CREDIT_SWAP_BELOW_MINIMUM",
        "GAUGE_CREDIT_INSUFFICIENT",
        "GAUGE_CREDIT_COMMIT_INSUFFICIENT",
        "GAUGE_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT",
    }
    assert adm.REFUSAL_CODES == frozenset(expected)
    assert adm.SCHEMA == "zegrid-gauge-admission-v1"
    assert adm.EXECUTION_SCHEMA == "zegrid-gauge-admission-exec-v1"


def test_every_emitted_code_belongs_to_the_closed_table():
    scenarios = [
        (GAUGE_PEAKS, (), GRANTED_B_RAM, _absent_credit()),
        (GAUGE_PEAKS, (), GRANTED_B_RAM, GRANTED_CREDIT),
        (GAUGE_PEAKS, (("gauge", "cache_build"),), GRANTED_B_RAM, GRANTED_CREDIT),
        (GAUGE_PEAKS, (), GRANTED_B_RAM, _bounded_credit(GRANTED_DECISION, restored=True)),
        (GAUGE_PEAKS, (), GRANTED_B_RAM, _bounded_credit(GRANTED_DECISION, platform="win32")),
        (GAUGE_PEAKS, (), GRANTED_B_RAM, _bounded_credit(GRANTED_DECISION, workers=3)),
    ]
    for peaks, overlaps, b_ram, credit in scenarios:
        decision = adm.evaluate_groups(
            _groups(_plan(peaks, overlaps=overlaps)), b_ram, credit
        )
        assert decision.reason_code in adm.REFUSAL_CODES | {adm.REASON_OK}
        for refusal in decision.refusals:
            assert refusal.code in adm.REFUSAL_CODES
            assert refusal.code != adm.REASON_OK
            assert refusal.actor in renv.PHASES
            assert refusal.margined_bytes > refusal.ceiling_bytes


def test_refusal_decision_roundtrips_as_canonical_json():
    envs = _plan(GAUGE_PEAKS)
    decision = adm.evaluate_groups(_groups(envs), GRANTED_B_RAM, _absent_credit())
    payload = json.loads(decision.to_json())
    assert payload["admitted"] is False
    assert payload["reason_code"] == adm.REASON_REFUSE_HOST_RAM
    assert payload["refusals"][0]["actor"] == "gauge"
    assert payload["refusals"][0]["raw_bytes"] == 10 * GIB
    assert payload["diagnostics"]["credit_never_subtracted_from_peak"] is True


def test_unknown_policy_is_refused_with_a_typed_error():
    envs = _plan(GAUGE_PEAKS)
    for policy in ("swappy", "", "BOUNDED_SWAP "):
        with pytest.raises(adm.InvalidCredit):
            adm.evaluate_groups(
                _groups(envs), GRANTED_B_RAM, adm.GaugeCredit(policy=policy)
            )


def test_split_group_is_refused_by_integrity_check():
    envs = _plan(GAUGE_PEAKS, overlaps=(("cache_build", "assembly"),))
    groups = list(_groups(envs))
    split = []
    for group in groups:
        if len(group) > 1:
            split.extend([(group[0],), (group[1],)])
        else:
            split.append(group)
    with pytest.raises(adm.InvalidGroups):
        adm.evaluate_groups(split, GRANTED_B_RAM, _absent_credit())


def test_gauge_isolated_from_its_overlap_is_refused():
    envs = _plan(GAUGE_PEAKS, overlaps=(("gauge", "cache_build"),))
    groups = _groups(envs)
    fake = []
    for group in groups:
        phases = tuple(env.phase for env in group)
        if phases == ("gauge", "cache_build"):
            fake.append(tuple(env for env in group if env.phase == "gauge"))
            fake.append(tuple(env for env in group if env.phase == "cache_build"))
        else:
            fake.append(group)
    # Every phase is present exactly once, but the grouping is not the frozen
    # connected components: an isolated gauge could otherwise claim singleton.
    with pytest.raises(adm.InvalidGroups):
        adm.evaluate_groups(fake, GRANTED_B_RAM, _absent_credit())


def test_duplicate_phase_is_refused():
    envs = _plan(GAUGE_PEAKS)
    groups = [tuple(group) for group in _groups(envs)]
    groups[1] = groups[1] + (groups[0][0],)
    with pytest.raises(adm.InvalidGroups):
        adm.evaluate_groups(groups, GRANTED_B_RAM, _absent_credit())


def test_invalid_b_ram_is_refused():
    envs = _plan(GAUGE_PEAKS)
    for bad in (True, False, -1, 1.0, "1", None):
        with pytest.raises(adm.InvalidBudget):
            adm.evaluate_groups(_groups(envs), bad, _absent_credit())


def test_inconsistent_credit_decision_is_refused():
    envs = _plan(GAUGE_PEAKS)
    # The credit must have been evaluated against the SAME B_ram: a ceiling
    # built from two different observations is refused, not averaged.
    with pytest.raises(adm.InvalidCredit):
        adm.evaluate_groups(_groups(envs), GRANTED_B_RAM + 1, GRANTED_CREDIT)

    ram_only_decision = cap.evaluate_credit(
        cap.GaugeSwapCapacity(
            a_raw=10 * GIB,
            devices=[],
            cgroup_levels=[],
            overcommit=cap.read_overcommit(0, None, None),
        ),
        10 * GIB,
        cap.POLICY_RAM_ONLY,
    )
    with pytest.raises(adm.InvalidCredit):
        adm.evaluate_groups(
            _groups(envs),
            GRANTED_B_RAM,
            adm.GaugeCredit(decision=ram_only_decision, policy=cap.POLICY_BOUNDED_SWAP),
        )


def test_non_envelope_member_is_refused():
    import dataclasses

    envs = _plan(GAUGE_PEAKS)
    groups = [tuple(group) for group in _groups(envs)]
    groups[0] = groups[0] + ("gauge",)
    with pytest.raises(adm.InvalidGroups):
        adm.evaluate_groups(groups, GRANTED_B_RAM, _absent_credit())

    groups = [tuple(group) for group in _groups(envs)]
    dirty = dataclasses.replace(groups[0][0], phase="not_a_phase")
    groups[0] = (dirty,) + groups[0][1:]
    with pytest.raises(adm.InvalidGroups):
        adm.evaluate_groups(groups, GRANTED_B_RAM, _absent_credit())


# ---------------------------------------------------------------------------
# Purity / no product wiring
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = {
    "__future__",
    "dataclasses",
    "hashlib",
    "json",
    "typing",
    "gauge_swap_capacity",
    "resource_envelope",
}


def test_module_imports_only_pure_contract_modules():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                imported.add((node.module or "").split(".")[0])
            elif node.module:
                imported.add(node.module.split(".")[-1])
            else:
                # ``from . import sibling`` carries the module in ``names``.
                imported.update(alias.name for alias in node.names)
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS


def test_module_never_reads_proc_and_has_no_generic_handler():
    text = MODULE_PATH.read_text(encoding="utf-8")
    # No host access at all: no file open, no path read, no subprocess.
    for forbidden in ("open(", "read_text", "Path(", "subprocess", "os.environ"):
        assert forbidden not in text
    assert "except Exception" not in text
    assert "except:" not in text


#: ZM-GAUGE-SWAP-G2A: the evaluator is now WIRED into the gauge path. Exactly
#: three modules may mention/import it, and every one of them is part of that
#: path: the product edge (``gauge_credit``), the final plan validator
#: (``scheduler``, LOCAL import inside the admission helper) and the engine
#: (``zemosaic_zegrid_mode``, LOCAL import inside ``_run_single``). Every other
#: module — dispatchers, prefetch, reservations, staging, checkpoints RUN,
#: assembly, export, runtime_control — must stay free of it.
ALLOWED_ADMISSION_WIRING = {
    "src/zemosaic/core/zegrid/gauge_credit.py",
    "src/zemosaic/core/zegrid/scheduler.py",
    "src/zemosaic/zemosaic_zegrid_mode.py",
}


def test_only_the_gauge_path_imports_the_admission_evaluator():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == MODULE_PATH:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "gauge_admission" in text and rel not in ALLOWED_ADMISSION_WIRING:
            offenders.append(rel)
    assert offenders == []


def test_scheduler_admission_import_is_local_not_module_level():
    """The final validator imports the evaluator INSIDE the helper.

    A module-level import would pull capacity/admission into every engine path
    that imports the scheduler, which G2a forbids.
    """
    tree = ast.parse(
        (SRC / "zemosaic" / "core" / "zegrid" / "scheduler.py").read_text(
            encoding="utf-8"
        )
    )
    module_level = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and "gauge_admission" in (
            node.module or ""
        ):
            module_level.append(node.module)
        if isinstance(node, ast.Import):
            module_level.extend(
                alias.name
                for alias in node.names
                if "gauge_admission" in alias.name
            )
    assert module_level == []
