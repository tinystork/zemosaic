"""ZM-ZEGRID-R55-B — ResourceEnvelope contract, overlap aggregation and hashing.

Pure unit tests: no product import beyond the contract module, no GPU, no I/O.
"""

from __future__ import annotations

import json

import pytest

from zemosaic.core.zegrid import resource_envelope as re


def _host(raw=1000, num=4, den=5, baseline=100):
    return re.HostEnvelope(
        observation_source="test.available_bytes",
        raw_available_bytes=raw,
        factor_num=num,
        factor_den=den,
        budget_bytes=(num * raw) // den,
        reserve_bytes=0,
        process_baseline_bytes=baseline,
    )


def _env(
    phase,
    components,
    *,
    overlaps=(),
    margin=(15, 100),
    reason=re.REASON_UNCALIBRATED,
    backend="cpu",
    disk=0,
):
    estimate = re.make_estimate(dict(components))
    return re.ResourceEnvelope(
        envelope_id=f"{phase}.{backend}",
        phase=phase,
        backend=backend,
        host=_host(),
        device=None,
        geometry=re.GeometryEnvelope(
            layout=(2, 2),
            refinement_rank=3,
            cell_id=None,
            patch_hw=(100, 100),
            tile_px=(64 if phase.startswith("cell_stack") else None),
            halo_px=32,
            channels=3,
        ),
        concurrency=re.ConcurrencyEnvelope(
            units=2, units_in_flight=1, workers_per_unit=1, prefetch_units=0
        ),
        estimate=estimate,
        bound=re.BoundEnvelope(
            kind="upper",
            margin_num=margin[0],
            margin_den=margin[1],
            confidence="static_bound",
        ),
        accounting=re.AccountingEnvelope(
            aggregate_rule=re.AGGREGATE_RULE_MAX_OVER_PHASES,
            host_peak_bytes=estimate.peak_bytes,
            device_peak_bytes=0,
            disk_bytes=disk,
            transfer_bytes=0,
        ),
        overlaps_with=tuple(overlaps),
        identity=re.IdentityEnvelope(run_id="run", plan_hash="", pinned=False),
        reason_code=reason,
    )


def _set(**overrides):
    envs = [
        _env("gauge", {"gauge_bytes": 100}),
        _env("cache_build", {"cache_bytes": 200}),
        _env("cell_stack_stream", {"stack_bytes": 300}),
        _env("assembly", {"assembly_bytes": 400}),
        _env("export", {"export_bytes": 50}),
    ]
    for phase, updates in overrides.items():
        for i, env in enumerate(envs):
            if env.phase == phase:
                envs[i] = re_apply(env, **updates)
    return envs


def re_apply(env, **updates):
    import dataclasses

    return dataclasses.replace(env, **updates)


# ---------------------------------------------------------------------------
# Integer-only validation
# ---------------------------------------------------------------------------

def test_require_int_rejects_bool_and_negative():
    assert re.require_int(0, "x") == 0
    for bad in (True, False, -1, 1.0, "1", None):
        with pytest.raises(re.InvalidEnvelope):
            re.require_int(bad, "x")


def test_budget_is_exact_integer_floor():
    env = _env("assembly", {"a": 1})
    env.host.validate()  # 1000 * 4 // 5 == 800
    bad = re_apply(
        env,
        host=re.HostEnvelope(
            observation_source="test",
            raw_available_bytes=1000,
            factor_num=4,
            factor_den=5,
            budget_bytes=801,
            reserve_bytes=0,
            process_baseline_bytes=100,
        ),
    )
    with pytest.raises(re.InvalidEnvelope, match="budget_bytes"):
        bad.host.validate()


def test_envelope_rejects_inconsistent_aggregate():
    env = _env("assembly", {"a": 100})
    bad = re_apply(
        env,
        accounting=re_apply(
            env.accounting, host_peak_bytes=env.estimate.peak_bytes + 1
        ),
    )
    with pytest.raises(re.InvalidEnvelope, match="host_peak_bytes"):
        re.validate_envelope(bad)


def test_estimate_components_must_sum_and_no_inner_margin():
    estimate = re.make_estimate({"a": 10, "b": 20})
    assert estimate.subtotal_bytes == 30 and estimate.peak_bytes == 30
    with pytest.raises(re.InvalidEnvelope, match="subtotal"):
        re.EstimateEnvelope(
            method="static_bound",
            calibration_id=None,
            components=(("a", 10),),
            subtotal_bytes=11,
            peak_bytes=11,
        ).validate()
    # A phase peak that already contains the margin is refused (no double reserve).
    with pytest.raises(re.InvalidEnvelope, match="margin"):
        re.EstimateEnvelope(
            method="static_bound",
            calibration_id=None,
            components=(("a", 10),),
            subtotal_bytes=10,
            peak_bytes=12,
        ).validate()
    with pytest.raises(re.InvalidEnvelope, match="lexical"):
        re.EstimateEnvelope(
            method="static_bound",
            calibration_id=None,
            components=(("b", 1), ("a", 1)),
            subtotal_bytes=2,
            peak_bytes=2,
        ).validate()


def test_bool_component_and_negative_reserve_fail_closed():
    with pytest.raises(re.InvalidEnvelope):
        re.make_estimate({"a": True})
    with pytest.raises(re.InvalidEnvelope):
        re.make_estimate({"a": -1})
    env = _env("assembly", {"a": 10})
    with pytest.raises(re.InvalidEnvelope, match="reserve"):
        re_apply(
            env,
            host=re_apply(env.host, reserve_bytes=env.host.budget_bytes + 1),
        ).host.validate()


def test_cpu_envelope_cannot_carry_device():
    env = _env("assembly", {"a": 10})
    device = re.DeviceEnvelope(
        vram_total_bytes=2048,
        vram_free_bytes=1024,
        vram_reserved_estimate_bytes=0,
        budget_bytes=512,
        device_name="fake",
    )
    with pytest.raises(re.InvalidEnvelope, match="device"):
        re.validate_envelope(re_apply(env, device=device))
    with pytest.raises(re.InvalidEnvelope, match="device_peak"):
        re.validate_envelope(
            re_apply(
                env,
                accounting=re_apply(env.accounting, device_peak_bytes=1),
            )
        )


def test_reason_code_must_be_in_closed_set():
    env = _env("assembly", {"a": 10})
    with pytest.raises(re.InvalidEnvelope, match="reason_code"):
        re.validate_envelope(re_apply(env, reason_code="R55_MADE_UP"))


# ---------------------------------------------------------------------------
# overlaps_with is mandatory, symmetric and closed
# ---------------------------------------------------------------------------

def test_overlaps_with_is_a_mandatory_field():
    with pytest.raises(TypeError):
        re.ResourceEnvelope(  # type: ignore[call-arg]
            envelope_id="assembly.cpu",
            phase="assembly",
            backend="cpu",
            host=_host(),
            device=None,
            geometry=re.GeometryEnvelope((2, 2), 1, None, None, None, 8, 3),
            concurrency=re.ConcurrencyEnvelope(1, 1, 1, 0),
            estimate=re.make_estimate({"a": 1}),
            bound=re.BoundEnvelope(),
            accounting=re.AccountingEnvelope(
                re.AGGREGATE_RULE_MAX_OVER_PHASES, 1, 0, 0
            ),
            identity=re.IdentityEnvelope(),
            reason_code=re.REASON_UNCALIBRATED,
        )


def test_overlap_unknown_self_and_duplicate_fail_closed():
    env = _env("assembly", {"a": 10}, overlaps=("nope",))
    with pytest.raises(re.InvalidOverlap, match="unknown phase"):
        re.validate_envelope(env)
    with pytest.raises(re.InvalidOverlap, match="itself"):
        re.validate_envelope(_env("assembly", {"a": 10}, overlaps=("assembly",)))
    with pytest.raises(re.InvalidOverlap, match="duplicates"):
        re.validate_envelope(
            _env("assembly", {"a": 10}, overlaps=("export", "export"))
        )


def test_asymmetric_overlap_fails_closed():
    envs = _set()
    envs = [
        re_apply(e, overlaps_with=("export",)) if e.phase == "assembly" else e
        for e in envs
    ]
    with pytest.raises(re.InvalidOverlap, match="asymmetric"):
        re.validate_envelope_set(envs)


# ---------------------------------------------------------------------------
# Aggregation: max over exclusive groups, explicit sum within a group
# ---------------------------------------------------------------------------

def test_aggregate_is_max_over_mutually_exclusive_phases():
    envs = _set()
    raw, peak = re.aggregate_host_peak_bytes(envs)
    assert raw == 400
    # margin applied exactly once: ceil(400 * 115 / 100) == 460
    assert peak == 460


def test_declared_overlap_group_is_summed():
    envs = [
        re_apply(e, overlaps_with=("export",)) if e.phase == "assembly" else e
        for e in _set()
    ]
    envs = [
        re_apply(e, overlaps_with=("assembly",)) if e.phase == "export" else e
        for e in envs
    ]
    groups = re.overlap_groups(envs)
    assert ("assembly", "export") in groups
    raw, peak = re.aggregate_host_peak_bytes(envs)
    assert raw == 450  # 400 + 50 instead of max(400, 50)
    assert peak == re.ceil_margin_div(450, 15, 100)


def test_transitive_overlap_group_is_consistent():
    envs = _set()
    pairs = {("gauge", "cache_build"), ("cache_build", "assembly")}
    envs = [
        re_apply(
            e,
            overlaps_with=tuple(
                b if a == e.phase else a for a, b in pairs if e.phase in (a, b)
            ),
        )
        for e in envs
    ]
    groups = re.overlap_groups(envs)
    assert ("gauge", "cache_build", "assembly") in groups
    raw, _ = re.aggregate_host_peak_bytes(envs)
    assert raw == 100 + 200 + 400


def test_margin_must_be_uniform_across_phases():
    envs = _set()
    envs = [
        re_apply(e, bound=re.BoundEnvelope(margin_num=15, margin_den=100))
        if e.phase != "export"
        else re_apply(e, bound=re.BoundEnvelope(margin_num=20, margin_den=100))
        for e in envs
    ]
    with pytest.raises(re.InvalidEnvelope, match="margin"):
        re.validate_envelope_set(envs)


def test_required_phases_and_single_stack_phase():
    envs = _set()
    assert len(re.validate_envelope_set(envs)) == 5
    with pytest.raises(re.InvalidEnvelope, match="missing required"):
        re.validate_envelope_set([e for e in envs if e.phase != "assembly"])
    with pytest.raises(re.InvalidEnvelope, match="exactly ONE cell_stack"):
        re.validate_envelope_set(
            envs + [_env("cell_stack_inmem", {"s2": 1})]
        )


def test_gui_preview_is_excluded_from_headless_plans():
    envs = _set() + [_env("gui_preview", {"preview": 10})]
    with pytest.raises(re.InvalidEnvelope, match="gui_preview"):
        re.validate_envelope_set(envs, headless=True)
    # Applicable (interactive) runs may include exactly one preview envelope.
    assert len(re.validate_envelope_set(envs, headless=False)) == 6


def test_disk_aggregation_is_reported_and_not_margin_inflated():
    envs = _set()
    envs = [
        re_apply(e, accounting=re_apply(e.accounting, disk_bytes=10))
        if e.phase == "cache_build" else e
        for e in envs
    ]
    assert re.aggregate_disk_bytes(envs) == 10
    # Disk is a residency term, not a host-RAM peak: no margin is applied.
    assert re.aggregate_host_peak_bytes(envs)[0] == 400


# ---------------------------------------------------------------------------
# Canonical plan hash
# ---------------------------------------------------------------------------

def test_plan_hash_is_canonical_and_stable():
    envs = _set()
    h1 = re.compute_plan_hash(envs)
    h2 = re.compute_plan_hash(list(reversed(envs)))
    assert h1 == h2
    assert h1.startswith("sha256:") and len(h1) == 7 + 64
    # Dictionary insertion order in components cannot change the hash.
    def _with_assembly_order(order):
        return [
            _env("gauge", {"gauge_bytes": 100}),
            _env("cache_build", {"cache_bytes": 200}),
            _env("cell_stack_stream", {"stack_bytes": 300}),
            _env("assembly", order),
            _env("export", {"export_bytes": 50}),
        ]

    assert re.compute_plan_hash(
        _with_assembly_order({"a": 1, "b": 2})
    ) == re.compute_plan_hash(_with_assembly_order({"b": 2, "a": 1}))
    # A real decision change DOES change the hash.
    changed = _set()
    changed = [
        re_apply(e, concurrency=re_apply(e.concurrency, units_in_flight=2, units=2))
        if e.phase == "cell_stack_stream" else e
        for e in changed
    ]
    assert re.compute_plan_hash(changed) != h1


def test_plan_hash_ignores_timestamps_run_id_and_margin_stamp():
    envs = _set()
    base = re.compute_plan_hash(envs)
    envs2 = [
        re_apply(
            e,
            host=re_apply(e.host, observed_at_monotonic_s=123.5),
            identity=re.IdentityEnvelope(run_id="another-run", plan_hash="", pinned=False),
        )
        for e in _set()
    ]
    assert re.compute_plan_hash(envs2) == base


def test_pinned_flag_is_part_of_the_hash():
    envs = _set()
    base = re.compute_plan_hash(envs)
    envs2 = [
        re_apply(e, identity=re_apply(e.identity, pinned=True)) for e in _set()
    ]
    assert re.compute_plan_hash(envs2) != base


def test_envelope_to_dict_is_json_safe():
    env = _env("assembly", {"a": 10})
    blob = json.dumps(env.to_dict(), sort_keys=True)
    assert '"overlaps_with": []' in blob
    assert isinstance(env.to_dict()["estimate"]["components"], dict)
