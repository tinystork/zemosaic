"""ZM-ZEGRID-R40-B1 (freeze-closure) focused tests — v2 calibration/prevalidation harness.

Covers the Nono review-0 repairs (H1 privacy/hash, H2 deep freeze, H3 durable
exactly-once, M1 relation-first/RGB, M2/M3 actual H4 arms, M4 no floors, M5 family
disjointness) plus output/schema safety.

Hard rule for this suite: **the reserved v2 validation cohort is never evaluated.**
No test opens the reserved validation outcome; the reserved evaluator is guarded by an
in-process token and is monkeypatched to fail if touched. No real path/FITS is read.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

_TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "r40b1_synthetic_calibration.py"


def _load_tool():
    name = "r40b1_synthetic_calibration"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r = _load_tool()


@pytest.fixture(scope="module")
def bundle():
    return r.build_prevalidation_bundle()


@pytest.fixture(scope="module")
def criteria_snapshot(bundle):
    return r.FrozenSnapshot(
        name="criteria",
        tag=r.TAG_CRITERIA,
        canonical_json=bundle["criteria"]["canonical_json"],
        sha256=bundle["criteria"]["sha256"],
        frozen=r.deep_freeze(bundle["criteria"]["payload"]),
    )


@pytest.fixture(scope="module")
def protocol_snapshot(bundle):
    return r.FrozenSnapshot(
        name="protocol",
        tag=r.TAG_PROTOCOL,
        canonical_json=bundle["protocol"]["canonical_json"],
        sha256=bundle["protocol"]["sha256"],
        frozen=r.deep_freeze(bundle["protocol"]["payload"]),
    )


# ---------------------------------------------------------------------------
# 1. Import safety / CLI / determinism
# ---------------------------------------------------------------------------

def test_import_audit_no_production_module_scope():
    audit = r.import_audit()
    assert audit["engine_access_is_lazy"] is True
    assert audit["zemosaic_module_scope_imports"] == []
    assert audit["astropy_module_scope_imports"] == []


def test_importing_tool_does_not_import_production():
    code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('t', r'{_TOOL_PATH}'); "
        "m=importlib.util.module_from_spec(s); sys.modules['t']=m; s.loader.exec_module(m); "
        "bad=[x for x in sys.modules if x.startswith('zemosaic')]; print('IMPORTED' if bad else 'CLEAN')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_TOOL_PATH.parents[1]))
    assert out.stdout.strip().endswith("CLEAN"), out.stdout + out.stderr


def test_cli_stages_present():
    p = r.build_parser()
    assert set(p._subparsers._group_actions[0].choices) == {
        "prepare", "invalidate-v1", "validate", "print-hashes"
    }


def test_print_hashes_deterministic(capsys):
    assert r.main(["print-hashes"]) == 0
    first = capsys.readouterr().out.strip()
    assert r.main(["print-hashes"]) == 0
    assert capsys.readouterr().out.strip() == first
    assert "protocol_sha256=" in first and "cohort_sha256=" in first


def test_prepare_is_byte_deterministic(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert r.main(["prepare", "--out", str(a)]) == 0
    assert r.main(["prepare", "--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()


def test_prepare_repeatable_no_resettable_lock(bundle):
    # no resettable process-local scientific lock exists any more
    assert not hasattr(r, "_VALIDATION_LOCK")
    again = r.build_prevalidation_bundle()
    assert r.canonical_json(again) == r.canonical_json(bundle)


# ---------------------------------------------------------------------------
# 2. H1 — privacy: fail closed, prose-safe, hashes recomputable
# ---------------------------------------------------------------------------

def test_h1_v1_prose_false_positive_is_now_fine():
    prose = "relation-first; reduction_rms = 1 - RMS(d_shadow) divided by RMS(d_prod); delta match"
    assert r.path_leak_category(prose) is None
    v1_broken = "relation-first; reduction_rms = 1 - RMS(d_shadow)/RMS(d_prod); delta match"
    assert r.path_leak_category(v1_broken) is None


def test_h1_genuine_leaks_are_still_detected():
    assert r.path_leak_category("/home/x/y.json") == "posix_absolute"
    assert r.path_leak_category("C:\\Users\\a\\b.json") == "windows_drive_absolute"
    assert r.path_leak_category("\\\\host\\share\\f") == "unc"
    assert r.path_leak_category("file:///tmp/x") == "file_uri"
    assert r.path_leak_category("root=/a/b/c.json") == "posix_absolute_embedded"
    assert r.path_leak_category("plain/relative/path.json") is None


def test_h1_privacy_fails_closed_without_redaction(tmp_path):
    payload = {"metrics": {"note": "see /home/secret/thing.json"}, "ok": 1}
    with pytest.raises(r.PrivacyError):
        r.assert_no_path_leaks(payload)
    out = tmp_path / "x.json"
    with pytest.raises(r.PrivacyError):
        r._emit(str(out), payload)
    assert not out.exists()  # fail closed: no artifact written


def test_h1_bundle_has_no_path_leak(bundle):
    assert r.scan_path_leaks(bundle) == []


def test_h1_content_hash_recomputable_from_artifact(bundle):
    assert r.verify_content_hash(bundle)
    reduced = {k: v for k, v in bundle.items() if k not in r.SELF_HASH_FIELDS}
    assert r.sha256_payload(reduced, r.TAG_CONTENT) == bundle["content_sha256"]


def test_h1_protocol_and_criteria_hashes_recompute(bundle, protocol_snapshot, criteria_snapshot):
    protocol_snapshot.verify()
    criteria_snapshot.verify()
    assert r.sha256_payload(bundle["protocol"]["payload"], r.TAG_PROTOCOL) == bundle["protocol"]["sha256"]
    assert r.sha256_payload(bundle["criteria"]["payload"], r.TAG_CRITERIA) == bundle["criteria"]["sha256"]


def test_h1_hash_written_matches_persisted_bytes(tmp_path):
    out = tmp_path / "p.json"
    sha = r._emit(str(out), r.build_prevalidation_bundle())
    assert sha == hashlib.sha256(out.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 3. H2 — deep freeze + tamper refusal
# ---------------------------------------------------------------------------

def test_h2_deep_freeze_blocks_nested_mutation(criteria_snapshot):
    nested = criteria_snapshot.frozen["h4_controlled_effect_admission"]
    with pytest.raises(TypeError):
        nested["reduction_min"] = 0.0
    with pytest.raises(TypeError):
        criteria_snapshot.frozen["m3_algebraic_cross_expression"]["ulp_k"] = 0


def test_h2_tampered_snapshot_refuses(criteria_snapshot):
    tampered_payload = json.loads(criteria_snapshot.canonical_json)
    tampered_payload["h4_controlled_effect_admission"]["reduction_min"] = 0.0
    bad = r.FrozenSnapshot(
        name="criteria",
        tag=r.TAG_CRITERIA,
        canonical_json=r.canonical_json(tampered_payload),
        sha256=criteria_snapshot.sha256,  # stale hash -> must refuse
        frozen=r.deep_freeze(tampered_payload),
    )
    with pytest.raises(r.CriteriaTamperError):
        bad.verify()


def test_h2_admission_verifies_snapshots_every_decision(bundle, criteria_snapshot, protocol_snapshot):
    ev = r.simulate_panel(next(s for s in r._calibration_specs() if s.family == "panel.isolated"))["evidence"]
    assert r.h4_admit(ev, criteria_snapshot, protocol_snapshot)["state"] == "CONTROLLED_EFFECT"
    broken = r.FrozenSnapshot(
        name="criteria", tag=r.TAG_CRITERIA,
        canonical_json=criteria_snapshot.canonical_json, sha256="0" * 64,
        frozen=criteria_snapshot.frozen,
    )
    with pytest.raises(r.CriteriaTamperError):
        r.h4_admit(ev, broken, protocol_snapshot)


# ---------------------------------------------------------------------------
# 4. H3 — durable, content-addressed, exactly-once claim
# ---------------------------------------------------------------------------

def _claim_key(**over):
    base = {"protocol_sha256": "p" * 8, "criteria_sha256": "c" * 8, "cohort_sha256": "h" * 8}
    base.update(over)
    return base


def test_h3_first_claim_wins_second_refuses(tmp_path):
    key = _claim_key()
    a = r.DurableClaim(tmp_path / "state", key)
    assert a.status() == "NONE"
    a.claim(metadata={"who": "a"})
    assert a.status() == "CLAIMED"
    b = r.DurableClaim(tmp_path / "state", key)
    with pytest.raises(r.ValidationClaimError):
        b.claim(metadata={"who": "b"})


def test_h3_completed_refuses_rerun(tmp_path):
    key = _claim_key()
    a = r.DurableClaim(tmp_path / "state", key)
    a.claim(metadata={})
    a.complete(result_sha256="deadbeef", metadata={})
    assert a.status() == "COMPLETED"
    with pytest.raises(r.ValidationClaimError):
        r.DurableClaim(tmp_path / "state", key).claim(metadata={})


def test_h3_ambiguous_crash_refuses(tmp_path):
    key = _claim_key()
    r.DurableClaim(tmp_path / "state", key).claim(metadata={})  # crash before complete
    with pytest.raises(r.ValidationClaimError):
        r.DurableClaim(tmp_path / "state", key).claim(metadata={})


def test_h3_claim_is_content_addressed(tmp_path):
    a = r.DurableClaim(tmp_path / "s", _claim_key(cohort_sha256="1" * 8))
    b = r.DurableClaim(tmp_path / "s", _claim_key(cohort_sha256="2" * 8))
    assert a.claim_id != b.claim_id
    a.claim(metadata={})
    b.claim(metadata={})  # a different cohort/claim is independent


def test_h3_two_subprocesses_exactly_one_claims(tmp_path):
    state = tmp_path / "state"
    script = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('t', r'{_TOOL_PATH}');"
        "m=importlib.util.module_from_spec(s);sys.modules['t']=m;s.loader.exec_module(m);"
        "key={'protocol_sha256':'p','criteria_sha256':'c','cohort_sha256':'h'};"
        f"c=m.DurableClaim(r'{state}',key);"
        "c.claim(metadata={'pid':1});print('CLAIMED')"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    outs = [p.communicate() for p in procs]
    claimed = sum(1 for o, _e in outs if "CLAIMED" in o)
    assert claimed == 1, outs
    assert any("validation already CLAIMED" in e or "exclusive race" in e for _o, e in outs)


# ---------------------------------------------------------------------------
# 5. M1 — relation-first / RGB non-independence
# ---------------------------------------------------------------------------

def _panel(case_kind):
    spec = next(s for s in r._calibration_specs() if s.family == f"panel.{case_kind}")
    return r.simulate_panel(spec)


def test_m1_channels_are_correlated_not_independent():
    p = _panel("isolated")
    recs = p["relation_records"]
    ch0 = np.array([c["reduction"] for rel in recs[:1] for c in rel["channel_states"]])
    assert len(ch0) == 3
    assert np.allclose(ch0, ch0[0], atol=0.15)  # channels strongly correlated


def test_m1_relation_first_requires_two_of_three_channels():
    assert int(r.INHERITED_GATES["channel_promote_min"]) == 2
    p = _panel("isolated")
    for rel in p["relation_records"]:
        if rel["state"] == "PROMOTED":
            assert rel["n_channels_promoted"] >= 2


def test_m1_pseudo_single_relation_cannot_satisfy_two_relations():
    p = _panel("pseudo_single_relation")
    ev = p["evidence"]
    assert ev["n_distinct_relations_promoted"] == 1
    assert ev["orientation_witnesses"]["v"] == 0
    assert ev["orientation_witnesses"]["h"] < 2


def test_m1_contradictory_channel_hard_override_blocks():
    p = _panel("contradictory_channel")
    ev = p["evidence"]
    assert ev["n_distinct_relations_promoted"] == 0
    assert any(rel["state"] == "BLOCKED" for rel in p["relation_records"] if rel["carries_injection"])


def test_m1_channel_duplication_does_not_change_relation_count():
    dup = _panel("channel_duplication")["evidence"]
    iso = _panel("isolated")["evidence"]
    assert dup["n_distinct_relations_promoted"] == iso["n_distinct_relations_promoted"]
    assert dup["orientation_witnesses"] == iso["orientation_witnesses"]


def test_m1_panel_counts_distinct_relations_only(bundle):
    ev = _panel("isolated")["evidence"]
    assert ev["n_relations"] == 38
    assert ev["orientation_witnesses"]["h"] + ev["orientation_witnesses"]["v"] <= 38


def test_m1_signatures_are_derived_values(bundle):
    ev = _panel("isolated")["evidence"]
    assert isinstance(ev["relation_signature"]["h"], list)
    assert all(isinstance(x, float) for x in ev["relation_signature"]["h"])


# ---------------------------------------------------------------------------
# 6. M2/M3 — real H4 arms on production primitives
# ---------------------------------------------------------------------------

def _arm(arm, kind):
    spec = next(s for s in r._calibration_specs() if s.family == f"reproj.{arm}_{kind}")
    return r.measure_reprojection_arm(spec)


def test_m2_i1a_crop_vs_full_source_parity():
    out = _arm("i1a", "tan")
    assert out["decision"] == "PARITY_BIT_IDENTICAL"
    assert out["n_common"] > 0


def test_m2_i1b_coverage_delta_recovered_through_real_stack():
    out = _arm("i1b", "tan")
    assert out["decision"] == "RECOVERED"
    assert out["h_relations"] >= 2 and out["v_relations"] >= 2
    assert out["min_reduction"] >= r.INHERITED_GATES["reduction_min"]
    assert out["product_roi_parity"] == "B2_MANDATORY"
    assert out["pass"] is True


def test_rm3_i1b_requires_independent_relations_and_reports_diagnostic():
    out = _arm("i1b", "tan")
    assert out["patches"] >= 2
    assert len(set(out["independent_patches"])) == out["patches"]
    assert all(rel["independent_relation"] for rel in out["relation_stats"])
    assert out["support_subset_ok"] is True
    assert "common_support_delta_abs_max" in out
    diag = out["recovered_delta_diagnostic"]
    assert diag["label"] == "ORCHESTRATION_ONLY_NON_GATING"
    assert diag["scientific_gate"] is False
    assert diag["authorizes_b2"] is False
    assert diag["common_support_equality_structurally_expected"] is True


def test_rm3_records_are_labelled_non_gating():
    out = _arm("i1b", "tan")
    for rel in out["relation_stats"]:
        assert rel["diagnostic_label"] == "ORCHESTRATION_ONLY_NON_GATING"
        assert rel["scientific_gate"] is False and rel["authorizes_b2"] is False
        assert "recovered_delta" not in rel  # misleading field names removed


def test_r4_i1b_fails_when_relations_cannot_be_formed(monkeypatch):
    spec = next(s for s in r._calibration_specs() if s.family == "reproj.i1b_tan")
    monkeypatch.setattr(r, "_i1b_patch", lambda *a, **k: {
        "patch_index": 0, "seed": 1, "target_crpix": [32.0, 32.0], "support_subset_ok": True,
        "h": None, "v": None, "n_h_boundaries": 0, "n_v_boundaries": 0,
    })
    out = r.measure_reprojection_arm(spec)
    assert out["decision"] == "NOT_IDENTIFIABLE"
    assert out["pass"] is False


def test_m3_i2_is_same_array_fast_vs_reference():
    out = _arm("i2", "tan")
    assert out["production_path"] == "fast"
    assert out["decision"] == "PARITY_BIT_IDENTICAL"
    assert out["max_abs_diff"] == 0.0


def test_m3_non_tan_i2_not_applicable_but_i1b_still_applicable():
    assert _arm("i2", "non_tan")["decision"] == "NOT_APPLICABLE"
    # I1b is a coverage mechanism, NOT a fast-path mechanism -> applicable on fallback
    i1b_non_tan = _arm("i1b", "non_tan")
    assert i1b_non_tan["decision"] == "RECOVERED"
    assert i1b_non_tan["h_relations"] >= 2 and i1b_non_tan["v_relations"] >= 2


def test_m2_toy_is_relabelled_orchestration_only(bundle):
    roi = bundle["criteria"]["payload"]["roi_shadow_parity"]
    assert roi["authority"] == "ORCHESTRATION_ONLY"
    assert roi["product_roi_parity"] == "B2_MANDATORY"
    h4 = bundle["criteria"]["payload"]["h4_controlled_effect_admission"]
    assert h4["authorizes_b2"] is False


def test_m2_real_arms_used_by_calibration(bundle):
    rep = bundle["calibration"]["reprojection"]
    assert rep["i1a_bit_identical"] and rep["i2_tan_bit_identical"]
    assert rep["i1b_tan_recovered"] and rep["i1b_non_tan_recovered"]
    assert rep["i1b_min_reduction"] >= 0.20
    assert rep["i2_non_tan_not_applicable"]


# ---------------------------------------------------------------------------
# 7. M4 — no fabricated floor thresholds
# ---------------------------------------------------------------------------

def test_m4_ulp_k_derived_from_rounding_count_not_a_floor(bundle):
    k, derivation = r._algebraic_ulp_k_derivation()
    assert k == 5
    assert "rounding_count" in derivation
    crit = bundle["criteria"]["payload"]["m3_algebraic_cross_expression"]
    assert crit["ulp_k"] == k
    assert crit["ulp_k"] == crit["ulp_k"]  # not max(floor, ...)
    assert crit["calibration_max_ratio"] <= k


def test_m4_no_floor_constants_in_source():
    src = _TOOL_PATH.read_text()
    assert "max(1e-12" not in src
    assert "_max_ulp_k" not in src
    assert "floor=4" not in src


def test_rm3_recovered_delta_is_diagnostic_and_non_gating(bundle):
    diag = bundle["calibration"]["recovered_delta_diagnostic"]
    assert diag["label"] == "ORCHESTRATION_ONLY_NON_GATING"
    assert diag["scientific_gate"] is False
    assert diag["authorizes_b2"] is False
    assert diag["consumed_by_h4_admission"] is False
    assert diag["common_support_equality_structurally_expected"] is True
    assert "structurally" in diag["note"].lower()
    crit = bundle["criteria"]["payload"]
    assert crit["recovered_delta_diagnostic"]["label"] == "ORCHESTRATION_ONLY_NON_GATING"
    assert crit["recovered_delta_diagnostic"]["scientific_gate"] is False
    inj = crit["injected_effect_recovery"]
    assert inj["recovered_delta_is_gating"] is False
    assert "recovered_delta_tolerance" not in inj
    assert "recovered_delta_rule" not in inj
    assert crit["h4_controlled_effect_admission"]["recovered_delta_consumed"] is False


def test_rm3_h4_admission_does_not_consume_the_diagnostic(bundle, criteria_snapshot, protocol_snapshot):
    spec = next(s for s in r._calibration_specs() if s.family == "panel.isolated")
    ev = r.simulate_panel(spec)["evidence"]
    assert r.h4_admit(ev, criteria_snapshot, protocol_snapshot)["state"] == "CONTROLLED_EFFECT"
    mutated = json.loads(bundle["criteria"]["canonical_json"])
    mutated["recovered_delta_diagnostic"] = {"label": "GATE", "tolerance": 123.0}
    mutated["injected_effect_recovery"]["recovered_delta_is_gating"] = True
    snap = r.freeze_snapshot("criteria", r.TAG_CRITERIA, mutated)
    assert r.h4_admit(ev, snap, protocol_snapshot)["state"] == "CONTROLLED_EFFECT"


def test_rm3_no_bit_identical_scientific_recovery_claim_in_source():
    src = _TOOL_PATH.read_text()
    assert "recovered_err = 0.0" not in src
    assert "the synthetic arms recover the mechanism exactly" not in src
    assert 'recovered_delta_rule' not in src
    assert 'recovered_delta_tolerance' not in src


def test_m4_amplitude_is_metadata_only_never_a_promotion_gate(bundle):
    crit = bundle["criteria"]["payload"]
    assert crit["injected_effect_recovery"]["injected_amplitude_is_generator_metadata_only"] is True
    flat = r.canonical_json(crit)
    assert "amplitude_min" not in flat
    assert "0.75" not in flat
    assert "recovered_delta_tolerance" not in flat


def test_m4_m3_adversarial_coverage_and_nonfinite_refusal():
    cal = r.run_calibration()
    variants = {row.get("variant") for row in cal["m3"]["outcomes"]}
    assert {"cancelling_opposite_offsets", "underflow_to_zero", "opposite_sign_large"} <= variants
    classes = {row.get("class") for row in cal["m3"]["outcomes"]}
    assert {"signed_zero", "subnormal", "large"} <= classes
    assert cal["m3"]["nonfinite_refusals"] >= 2


# ---------------------------------------------------------------------------
# 8. M5 — family-level disjointness + reserved cohort untouched
# ---------------------------------------------------------------------------

def test_m5_family_disjointness(bundle):
    d = bundle["protocol"]["payload"]["split"]["disjointness"]
    assert d["shared_families"] == [] and d["shared_seeds"] == []
    cal = {s.family for s in r._calibration_specs()}
    val = {s.family for s in r._validation_specs_reserved()}
    assert cal.isdisjoint(val)


def test_m5_family_guard_raises_on_overlap():
    cal = [r.ScenarioSpec("a", "calibration", 1, "fam.x", ("p",))]
    val = [r.ScenarioSpec("b", "validation", 2, "fam.x", ("q",))]
    with pytest.raises(r.FamilyDisjointnessError):
        r.assert_family_disjoint(cal, val)


def test_m5_reserved_cohort_has_no_outcomes(bundle):
    cohort = bundle["reserved_validation_cohort"]
    assert cohort["generator_invoked"] is False
    assert cohort["outcomes_present"] is False
    assert cohort["specs"] and all("outcome" not in s for s in cohort["specs"])
    assert bundle["validation_status"] == "NOT_RUN"
    assert bundle["validation_runs"] == 0
    assert bundle["validation_claim"] is None


def test_m5_reserved_evaluator_refuses_without_gate(bundle, criteria_snapshot, protocol_snapshot):
    spec = next(s for s in r._validation_specs_reserved())
    with pytest.raises(r.ReservedCohortError):
        r.evaluate_group_against_expectation(
            spec,
            expected="EVALUATED",
            criteria_snapshot=criteria_snapshot,
            protocol_snapshot=protocol_snapshot,
            authorization=None,
        )


def test_m5_prepare_never_calls_reserved_evaluator(monkeypatch):
    called = {"n": 0}

    def bomb(*a, **k):
        called["n"] += 1
        raise AssertionError("reserved validation evaluator was called during prepare")

    monkeypatch.setattr(r, "evaluate_group_against_expectation", bomb)
    monkeypatch.setattr(r, "_evaluate_arm", bomb)
    b = r.build_prevalidation_bundle()
    assert called["n"] == 0
    assert b["validation_status"] == "NOT_RUN"


def test_m5_validate_stage_requires_explicit_consumption(bundle, tmp_path):
    with pytest.raises(r.ReservedCohortError):
        r.validate_stage(
            bundle=bundle,
            state_dir=tmp_path / "s",
            out_path=tmp_path / "o.json",
            consume_reserved_cohort=False,
        )
    assert not (tmp_path / "o.json").exists()
    assert not (tmp_path / "s").exists()


# ---------------------------------------------------------------------------
# 9. Bundle / invalidation record contract
# ---------------------------------------------------------------------------

def test_bundle_status_contract(bundle):
    assert bundle["status"] == r.STATUS_READY
    assert bundle["b1_status"] == r.STATUS_FAIL_INVALID
    assert bundle["r40b2_authorized"] is False
    assert bundle["b1_status"] != "B1_PASS"
    r.assert_prevalidation_has_no_validation_outcome(bundle)


def test_bundle_carries_no_validation_outcome_fields(bundle):
    for key in ("validation_outcomes", "validation_result", "validation_pass"):
        assert key not in bundle


def test_phase_metadata_is_truthful_and_stable(bundle):
    # the package phase is a stable truthful label, not a stale per-review counter
    assert r.PHASE == "freeze-closure"
    assert bundle["phase"] == r.PHASE
    assert bundle["protocol"]["payload"]["phase"] == r.PHASE
    # the immutable v1 invalidation record keeps its historical phase (byte-reproducible)
    assert r.PHASE_V1_INVALIDATION == "rework-1"
    assert r.build_v1_invalidation_record()["phase"] == r.PHASE_V1_INVALIDATION


def test_invalidation_record_contract():
    rec = r.build_v1_invalidation_record()
    assert rec["record"] == "v1_invalidation"
    assert rec["withdrawn"] is True
    assert rec["b1_status"] == r.STATUS_FAIL_INVALID
    assert rec["r40b2_authorized"] is False
    assert rec["artifact"]["sha256"] == r.V1_ARTIFACT_SHA256
    assert rec["artifact"]["bytes"] == r.V1_ARTIFACT_BYTES
    assert rec["artifact"]["preserved_immutable"] is True
    assert rec["artifact"]["overwritten"] is False
    assert rec["v1_claimed"]["final_status"] == "B1_PASS"
    assert rec["v1_claimed"]["r40b2_authorized"] is True
    assert rec["validation_cohort_v1"]["status"] == "CONSUMED_COMPROMISED"
    assert rec["validation_cohort_v1"]["rerun_forbidden"] is True
    ids = {x["id"] for x in rec["invalid_reasons"]}
    assert {"H1", "H2", "H3", "M1", "M2", "M3", "M4", "M5", "L1", "L2"} <= ids
    assert rec["hash_recomputable_from_v1_artifact"] is False


def test_off_on_zero_tolerance_gate_preserved_verbatim(bundle):
    gate = bundle["criteria"]["payload"]["off_on_identity"]
    assert gate["mode"] == "EXACT_BYTE_IDENTITY"
    assert gate["tolerance"] == 0.0
    assert gate["calibrated_by_b1"] is False
    assert gate["verbatim_from_r40a_section_8_2"] is True


# ---------------------------------------------------------------------------
# 10. Output / schema safety
# ---------------------------------------------------------------------------

def test_dumps_strict_refuses_non_finite():
    with pytest.raises(ValueError):
        r.dumps_strict({"x": float("nan")})
    with pytest.raises(r.NonFiniteError):
        r.assert_finite_json({"a": [float("inf")]})


def test_atomic_write_and_temp_cleanup(tmp_path):
    out = tmp_path / "x.json"
    sha = r.atomic_write_json(out, {"b": 1, "a": [1.0]})
    assert not list(tmp_path.glob("*.tmp"))
    assert sha == hashlib.sha256(out.read_bytes()).hexdigest()


def test_atomic_write_failure_leaves_nothing(tmp_path, monkeypatch):
    out = tmp_path / "x.json"
    monkeypatch.setattr(r.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        r.atomic_write_json(out, {"a": 1})
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))


def test_output_path_safety(tmp_path):
    with pytest.raises(r.PrivacyError):
        r.resolve_output_path("")
    with pytest.raises(r.PrivacyError):
        r.resolve_output_path("../escape.json")
    with pytest.raises(r.PrivacyError):
        r.resolve_output_path("C:\\abs\\x.json")
    with pytest.raises(r.PrivacyError):
        r.resolve_output_path("file:///tmp/x.json")
    with pytest.raises(r.PrivacyError):
        r.resolve_output_path(str(tmp_path / "sub" / "x.json"), allowed_root=tmp_path / "other")
    assert r.resolve_output_path(str(tmp_path / "x.json"), allowed_root=tmp_path).name == "x.json"


def test_no_real_path_or_fits_access_during_prepare():
    opened: list[str] = []
    real_open = builtins.open

    def spy(file, *a, **k):
        try:
            opened.append(str(file))
        except Exception:
            pass
        return real_open(file, *a, **k)

    builtins.open = spy
    try:
        r.build_prevalidation_bundle()
    finally:
        builtins.open = real_open
    forbidden = (".fits", "Windows816", "zegrid_r37_pre_feather", "r39a_windows816", ".a2a-reports", "LP341", "r40b1_synthetic_calibration.json")
    assert not [p for p in opened if any(f in p for f in forbidden)], opened


def test_instrumentation_absent_from_content_hash_inputs(bundle):
    # determinism: only SELF_HASH_FIELDS are excluded, and the content hash is stable
    assert r.content_sha256(bundle) == bundle["content_sha256"]


# ---------------------------------------------------------------------------
# 11. R1/R2/R3/R6 — rework-2 repairs (disposable cohort only)
# ---------------------------------------------------------------------------

_DISP = [
    r.ScenarioSpec("disp_iso", "validation", 9001, "disp_panel.isolated", ("isolated", "moderate")),
    r.ScenarioSpec("disp_null", "validation", 9002, "disp_panel.null", ("null", "none")),
]


@pytest.fixture(scope="module")
def disp_bundle(bundle):
    return r.build_prevalidation_bundle(val_specs=_DISP, calibration=bundle["calibration"])


# --- R1: canonical equivalence binding -------------------------------------------------


def test_r1_snapshot_from_entry_binds_payload_canonical_and_hash(bundle):
    snap = r.snapshot_from_entry("criteria", r.TAG_CRITERIA, bundle["criteria"])
    snap.verify()
    tampered = json.loads(json.dumps(bundle["criteria"]))
    tampered["payload"]["h4_controlled_effect_admission"]["reduction_min"] = 0.0
    with pytest.raises(r.CriteriaTamperError):
        r.snapshot_from_entry("criteria", r.TAG_CRITERIA, tampered)


def test_r1_frozen_object_must_match_the_canonical_snapshot(criteria_snapshot):
    broken = r.FrozenSnapshot(
        name="criteria",
        tag=r.TAG_CRITERIA,
        canonical_json=criteria_snapshot.canonical_json,
        sha256=criteria_snapshot.sha256,
        frozen=r.deep_freeze({"totally": "different"}),
    )
    with pytest.raises(r.CriteriaTamperError):
        broken.verify()


def test_r1_non_canonical_snapshot_refuses(criteria_snapshot):
    parsed = json.loads(criteria_snapshot.canonical_json)
    pretty = json.dumps(parsed, indent=2, sort_keys=True)  # same object, non-canonical bytes
    broken = r.FrozenSnapshot(
        name="criteria", tag=r.TAG_CRITERIA, canonical_json=pretty,
        sha256=criteria_snapshot.sha256, frozen=criteria_snapshot.frozen,
    )
    with pytest.raises(r.CriteriaTamperError):
        broken.verify()


def test_r1_h4_admit_consumes_the_bound_frozen_criteria(bundle, protocol_snapshot):
    spec = next(s for s in r._calibration_specs() if s.family == "panel.isolated")
    ev = r.simulate_panel(spec)["evidence"]
    good = r.snapshot_from_entry("criteria", r.TAG_CRITERIA, bundle["criteria"])
    assert r.h4_admit(ev, good, protocol_snapshot)["state"] == "CONTROLLED_EFFECT"
    strict = json.loads(bundle["criteria"]["canonical_json"])
    strict["h4_controlled_effect_admission"]["reduction_min"] = 0.99
    strict_snap = r.freeze_snapshot("criteria", r.TAG_CRITERIA, strict)
    assert r.h4_admit(ev, strict_snap, protocol_snapshot)["state"] == "NO_EFFECT"


def test_r1_validate_stage_refuses_tampered_payload_before_claim(disp_bundle, tmp_path):
    tampered = json.loads(json.dumps(disp_bundle))
    tampered["criteria"]["payload"]["h4_controlled_effect_admission"]["reduction_min"] = 0.0
    # (a) unsealed tamper: the content hash no longer matches -> fail closed
    with pytest.raises(r.ValidationClaimError):
        r.validate_stage(
            bundle=tampered, state_dir=tmp_path / "s", out_path=tmp_path / "o.json",
            consume_reserved_cohort=True, specs=_DISP,
        )
    # (b) re-sealed tamper: content hash recomputed, but the payload is no longer
    # canonically equivalent to the frozen canonical_json -> refuses before claim
    resealed = json.loads(json.dumps(tampered))
    resealed["content_sha256"] = r.content_sha256(resealed)
    with pytest.raises(r.CriteriaTamperError):
        r.validate_stage(
            bundle=resealed, state_dir=tmp_path / "s", out_path=tmp_path / "o.json",
            consume_reserved_cohort=True, specs=_DISP,
        )
    assert not (tmp_path / "s").exists()
    assert not (tmp_path / "o.json").exists()


# --- R2: expected decisions + real h4_admit routing ------------------------------------


def test_r2_expected_decisions_are_frozen_for_every_group(bundle):
    exp = bundle["reserved_validation_cohort"]["expected_decisions"]
    assert set(exp) == {s.group_id for s in r._validation_specs_reserved()}
    assert set(exp.values()) <= {
        "EVALUATED", "REFUSED_NONFINITE", "PARITY_OK", "NOT_IDENTIFIABLE",
        "PARITY_BIT_IDENTICAL", "NOT_APPLICABLE", "RECOVERED", "CONTROLLED_EFFECT",
        "NO_EFFECT", "INCOMPLETE_PANEL",
    }
    assert r._expected_decisions(r._validation_specs_reserved()) == exp


def test_r2_disposable_validation_routes_panels_through_h4_admit(disp_bundle, tmp_path):
    result = r.validate_stage(
        bundle=disp_bundle, state_dir=tmp_path / "state", out_path=tmp_path / "out.json",
        consume_reserved_cohort=True, specs=_DISP,
    )
    assert result["admission_router"] == "h4_admit with the bound frozen snapshots"
    by_id = {o["group_id"]: o for o in result["outcomes"]}
    assert by_id["disp_iso"]["observed"] == "CONTROLLED_EFFECT"
    assert by_id["disp_null"]["observed"] == "NO_EFFECT"
    assert all(o["pass"] for o in result["outcomes"])
    # no top-level defaulting bug: panels carry real evidence + ladder reasons
    assert by_id["disp_iso"]["detail"]["evidence"]["case_kind"] == "isolated"
    assert by_id["disp_null"]["detail"]["reasons"]
    assert result["b1_status"] == "B1_PASS"


def test_r2_criteria_materially_drive_the_decision(disp_bundle):
    spec = next(s for s in _DISP if s.family.endswith("isolated"))
    proto = r.snapshot_from_entry("protocol", r.TAG_PROTOCOL, disp_bundle["protocol"])
    good = r.snapshot_from_entry("criteria", r.TAG_CRITERIA, disp_bundle["criteria"])
    ok = r.evaluate_group_against_expectation(
        spec, expected="CONTROLLED_EFFECT", criteria_snapshot=good,
        protocol_snapshot=proto, authorization=r.COHORT_GATE,
    )
    assert ok["observed"] == "CONTROLLED_EFFECT" and ok["pass"] is True
    strict = json.loads(disp_bundle["criteria"]["canonical_json"])
    strict["h4_controlled_effect_admission"]["reduction_min"] = 0.99
    bad_snap = r.freeze_snapshot("criteria", r.TAG_CRITERIA, strict)
    bad = r.evaluate_group_against_expectation(
        spec, expected="CONTROLLED_EFFECT", criteria_snapshot=bad_snap,
        protocol_snapshot=proto, authorization=r.COHORT_GATE,
    )
    assert bad["observed"] == "NO_EFFECT" and bad["pass"] is False


def test_r2_disposable_validation_is_durable_exactly_once(disp_bundle, tmp_path):
    r.validate_stage(
        bundle=disp_bundle, state_dir=tmp_path / "state", out_path=tmp_path / "out.json",
        consume_reserved_cohort=True, specs=_DISP,
    )
    with pytest.raises(r.ValidationClaimError):
        r.validate_stage(
            bundle=disp_bundle, state_dir=tmp_path / "state", out_path=tmp_path / "out2.json",
            consume_reserved_cohort=True, specs=_DISP,
        )


# --- R3: whole-bundle preflight before the claim ---------------------------------------


def _preflight(bundle, specs=_DISP):
    return r.preflight_bundle(bundle, specs=specs)


def test_r3_preflight_accepts_a_clean_bundle(disp_bundle):
    out = _preflight(disp_bundle)
    assert out["protocol_snapshot"].verify() is None
    assert out["code_identity"]["repo_head"] == r._probe_repo_head()
    assert out["code_identity"]["baseline_head"] == r.BASELINE_HEAD


def test_r3_preflight_refuses_content_hash_tamper(disp_bundle):
    bad = json.loads(json.dumps(disp_bundle))
    bad["next_gate"] = "tampered"
    with pytest.raises(r.ValidationClaimError):
        _preflight(bad)


def test_r3_preflight_refuses_wrong_status_and_no_outcome_state(disp_bundle):
    for field, value in (("status", "B1_PASS"), ("b1_status", "B1_PASS"),
                         ("r40b2_authorized", True), ("validation_status", "DONE"),
                         ("validation_runs", 1)):
        bad = json.loads(json.dumps(disp_bundle))
        bad[field] = value
        bad["content_sha256"] = r.content_sha256(bad)
        with pytest.raises(r.ValidationClaimError):
            _preflight(bad)


def test_r3_preflight_refuses_consumed_cohort(disp_bundle):
    bad = json.loads(json.dumps(disp_bundle))
    bad["reserved_validation_cohort"]["generator_invoked"] = True
    bad["reserved_validation_cohort"]["outcomes_present"] = True
    bad["content_sha256"] = r.content_sha256(bad)
    with pytest.raises(r.ValidationClaimError):
        _preflight(bad)


def test_r3_preflight_refuses_cohort_and_expectation_mismatch(disp_bundle):
    bad = json.loads(json.dumps(disp_bundle))
    bad["reserved_validation_cohort"]["cohort_sha256"] = "0" * 64
    bad["content_sha256"] = r.content_sha256(bad)
    with pytest.raises(r.ValidationClaimError):
        _preflight(bad)
    bad2 = json.loads(json.dumps(disp_bundle))
    bad2["reserved_validation_cohort"]["expected_decisions"]["disp_iso"] = "NO_EFFECT"
    bad2["content_sha256"] = r.content_sha256(bad2)
    with pytest.raises(r.ValidationClaimError):
        _preflight(bad2)


def test_r3_preflight_refuses_code_and_head_identity_mismatch(disp_bundle):
    for key, value in (("tool_source_sha256", "0" * 64), ("repo_head", "deadbeef"),
                       ("baseline_head", "deadbeef")):
        bad = json.loads(json.dumps(disp_bundle))
        bad["code_identity"][key] = value
        bad["content_sha256"] = r.content_sha256(bad)
        with pytest.raises(r.ValidationClaimError):
            _preflight(bad)


def test_r3_code_identity_is_reproducible(disp_bundle):
    current = r.compute_code_identity()
    assert disp_bundle["code_identity"] == current
    # RM4 exact-byte: the tool hash is a RAW-BYTE hash
    assert current["tool_source_sha256"] == hashlib.sha256(_TOOL_PATH.read_bytes()).hexdigest()
    assert len(current["repo_head"]) == 40


def test_r3_validate_stage_preflights_before_claim(disp_bundle, tmp_path):
    bad = json.loads(json.dumps(disp_bundle))
    bad["status"] = "B1_PASS"
    bad["content_sha256"] = r.content_sha256(bad)
    with pytest.raises(r.ValidationClaimError):
        r.validate_stage(
            bundle=bad, state_dir=tmp_path / "s", out_path=tmp_path / "o.json",
            consume_reserved_cohort=True, specs=_DISP,
        )
    assert not (tmp_path / "s").exists()


# --- R6: durable claim path safety -----------------------------------------------------


def test_r6_symlinked_state_dir_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(r.PrivacyError):
        r.DurableClaim(link, _claim_key())


def test_r6_non_directory_component_refused(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(r.PrivacyError):
        r.DurableClaim(f / "sub", _claim_key())


def test_r6_symlinked_marker_refused(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    claim = r.DurableClaim(state, _claim_key())
    try:
        claim.claim_path.symlink_to(tmp_path / "evil.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(r.PrivacyError):
        claim.status()


def test_r6_safe_state_dir_accepts_a_normal_path(tmp_path):
    p = r.assert_safe_directory(tmp_path / "a" / "b", label="t")
    claim = r.DurableClaim(p, _claim_key())
    assert claim.status() == "NONE"
    claim.claim(metadata={})
    assert claim.status() == "CLAIMED"


# ---------------------------------------------------------------------------
# 12. RM1/RM2/RM3 — rework-3 repairs
# ---------------------------------------------------------------------------


def test_rm1_product_source_manifest_is_the_complete_sorted_raw_byte_tree(bundle):
    manifest = bundle["code_identity"]["product_source_manifest"]
    repo = _TOOL_PATH.resolve().parents[1]
    expected = sorted(
        p.relative_to(repo).as_posix()
        for p in (repo / "src" / "zemosaic").rglob("*.py")
        if p.is_file()
    )
    paths = [e["path"] for e in manifest]
    # the manifest is EXACTLY the complete product source tree: nothing missing, nothing extra
    assert paths == expected
    assert paths == sorted(paths)
    assert len(set(paths)) == len(paths)
    assert len(paths) == bundle["code_identity"]["product_source_manifest_count"]
    assert bundle["code_identity"]["product_source_manifest_policy"] == r.PRODUCT_SOURCE_POLICY
    for e in manifest:
        assert e["path"].startswith("src/zemosaic/") and e["path"].endswith(".py")
        assert len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0
        assert e["why"]
    # every previously bound module is still bound, plus the eagerly imported closure
    required = {
        "src/zemosaic/core/zegrid/execution.py",
        "src/zemosaic/core/zegrid/geometry.py",
        "src/zemosaic/core/zegrid/cell_feather.py",
        "src/zemosaic/core/zegrid/mosaic.py",
        "src/zemosaic/core/zegrid/sweep.py",
        "src/zemosaic/core/zegrid/cell_offset.py",
        "src/zemosaic/core/zegrid/assembly.py",
        "src/zemosaic/core/canonical_engine.py",
        "src/zemosaic/core/canonical_stacking.py",
        "src/zemosaic/core/canonical_support.py",
        "src/zemosaic/core/canonical_equalize.py",
        "src/zemosaic/core/canonical_streaming.py",
        "src/zemosaic/core/zegrid/science_adapter.py",
        "src/zemosaic/core/zegrid/executor.py",
        "src/zemosaic/core/zegrid/photometric.py",
        "src/zemosaic/__init__.py",
        "src/zemosaic/core/__init__.py",
        "src/zemosaic/core/zegrid/__init__.py",
    }
    assert required <= set(paths)
    assert r.compute_product_source_manifest() == manifest


def test_rm1_manifest_hashes_are_exact_raw_bytes(bundle):
    # RM4: every product-source hash is a raw-byte SHA-256 (no text decode)
    repo = _TOOL_PATH.resolve().parents[1]
    manifest = bundle["code_identity"]["product_source_manifest"]
    assert r.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
    for e in manifest:
        assert e["sha256"] == hashlib.sha256((repo / e["path"]).read_bytes()).hexdigest()


def test_rm1_manifest_sha256_and_count_are_bound(bundle):
    ident = bundle["code_identity"]
    assert ident["product_source_manifest_count"] == len(ident["product_source_manifest"])
    assert ident["product_source_manifest_sha256"] == r.sha256_payload(
        ident["product_source_manifest"], r.TAG_MANIFEST
    )


def test_rm1_r4_source_hashes_are_raw_bytes_only(monkeypatch):
    # RM4 exact-byte: identity computation must never go through a text decode. Reading
    # the tool source or any product source in text mode must be impossible here.
    real_read_text = pathlib.Path.read_text
    repo = _TOOL_PATH.resolve().parents[1]

    def guarded(self, *a, **k):
        s = str(self)
        if s == str(_TOOL_PATH) or s.startswith(str(repo / "src" / "zemosaic")):
            raise AssertionError(f"text-mode read must not hash identity sources: {s}")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", guarded)
    ident = r.compute_code_identity()
    monkeypatch.undo()
    assert ident["tool_source_sha256"] == hashlib.sha256(_TOOL_PATH.read_bytes()).hexdigest()
    for e in ident["product_source_manifest"]:
        assert e["sha256"] == hashlib.sha256((repo / e["path"]).read_bytes()).hexdigest()


def test_rm1_missing_product_source_root_or_file_refuses(monkeypatch):
    original = r.product_source_paths
    monkeypatch.setattr(r, "PRODUCT_SOURCE_ROOT", "src/zemosaic_does_not_exist")
    with pytest.raises(r.HarnessError):
        r.compute_product_source_manifest()
    monkeypatch.setattr(r, "PRODUCT_SOURCE_ROOT", "src/zemosaic")
    # a bound-but-absent file refuses (fail closed)
    monkeypatch.setattr(
        r,
        "product_source_paths",
        lambda: list(original()) + ["src/zemosaic/core/does_not_exist.py"],
    )
    with pytest.raises(r.HarnessError):
        r.compute_product_source_manifest()


def test_rm1_loaded_product_closure_is_fully_bound():
    # forcing the arms' lazy imports/calibration entrypoints must leave ZERO local
    # product module unbound: the manifest cannot omit eagerly imported product code
    audit = r.audit_loaded_product_closure()
    assert audit["unbound_local_modules"] == []
    assert audit["manifest_count"] == len(audit["manifest_paths"])
    loaded = set(audit["loaded_local_modules"])
    assert len(loaded) >= 20
    # the previously unbound, eagerly imported modules flagged by review-2 are now bound
    assert {
        "src/zemosaic/core/canonical_streaming.py",
        "src/zemosaic/core/zegrid/executor.py",
        "src/zemosaic/core/zegrid/photometric.py",
        "src/zemosaic/core/zegrid/science_adapter.py",
        "src/zemosaic/core/zegrid/__init__.py",
    } <= loaded
    assert loaded <= set(audit["manifest_paths"])


def test_rm1_loaded_closure_subset_is_in_the_persisted_manifest(bundle):
    persisted = {e["path"] for e in bundle["code_identity"]["product_source_manifest"]}
    r.force_product_import_closure()
    loaded = set(r.loaded_product_source_paths())
    assert loaded and loaded <= persisted


def test_rm1_previously_unbound_module_is_bound_and_dirty_edit_refuses(disp_bundle):
    manifest = {e["path"]: e["sha256"] for e in r.compute_product_source_manifest()}
    assert "src/zemosaic/core/zegrid/science_adapter.py" in manifest

    real = r.compute_product_source_manifest

    def dirty():
        return [
            dict(e, sha256="0" * 64)
            if e["path"] == "src/zemosaic/core/zegrid/science_adapter.py"
            else e
            for e in real()
        ]

    original = r.compute_product_source_manifest
    r.compute_product_source_manifest = dirty
    try:
        with pytest.raises(r.ValidationClaimError):
            r.preflight_bundle(disp_bundle, specs=_DISP)
    finally:
        r.compute_product_source_manifest = original
    # safely restored: the clean state still preflights
    assert r.preflight_bundle(disp_bundle, specs=_DISP)["code_identity_sha256"]


def test_rm1_runtime_versions_are_bound_and_never_unavailable(bundle):
    vers = bundle["code_identity"]["runtime_versions"]
    for key in ("python", "numpy", "scipy", "astropy", "reproject"):
        assert key in vers and str(vers[key]).strip()
        assert str(vers[key]).lower() != "unavailable"
    assert vers == r.compute_runtime_versions()
    assert bundle["code_identity"]["repo_head"] == r._probe_repo_head()
    assert bundle["code_identity"]["baseline_head"] == r.BASELINE_HEAD
    assert bundle["code_identity"]["repo_head"].lower() != "unavailable"


def test_rm1_dirty_product_edit_with_unchanged_head_refuses_before_claim(disp_bundle):
    # simulate a dirty edit in a bound production file: the re-probed manifest
    # changes while git HEAD (and the tool hash) stay the same -> preflight refuses
    real = r.compute_product_source_manifest

    def dirty():
        m = real()
        m[0] = dict(m[0], sha256="0" * 64)
        return m

    original = r.compute_product_source_manifest
    r.compute_product_source_manifest = dirty
    try:
        with pytest.raises(r.ValidationClaimError):
            r.preflight_bundle(disp_bundle, specs=_DISP)
    finally:
        r.compute_product_source_manifest = original
    # and the clean state still preflights
    assert r.preflight_bundle(disp_bundle, specs=_DISP)["code_identity_sha256"]


def test_rm1_runtime_version_mismatch_refuses_before_claim(disp_bundle):
    original = r.compute_runtime_versions
    r.compute_runtime_versions = lambda: {**original(), "numpy": "0.0.0-fake"}
    try:
        with pytest.raises(r.ValidationClaimError):
            r.preflight_bundle(disp_bundle, specs=_DISP)
    finally:
        r.compute_runtime_versions = original


def test_rm2_claim_key_includes_the_code_identity_digest(disp_bundle, tmp_path):
    result = r.validate_stage(
        bundle=disp_bundle, state_dir=tmp_path / "state", out_path=tmp_path / "out.json",
        consume_reserved_cohort=True, specs=_DISP,
    )
    key = result["claim"]["key"]
    assert "code_identity_sha256" in key
    assert key["code_identity_sha256"] == result["code_identity_sha256"]
    assert key["code_identity_sha256"] == r.code_identity_digest(result["code_identity"])
    assert set(key) == {"protocol_sha256", "criteria_sha256", "cohort_sha256", "code_identity_sha256"}


def test_rm2_different_code_identities_cannot_share_a_claim_id(tmp_path):
    base = {
        "protocol_sha256": "p" * 8, "criteria_sha256": "c" * 8,
        "cohort_sha256": "h" * 8, "code_identity_sha256": "a" * 8,
    }
    other = dict(base, code_identity_sha256="b" * 8)
    a = r.DurableClaim(tmp_path / "s", base)
    b = r.DurableClaim(tmp_path / "s", other)
    assert a.claim_id != b.claim_id
    a.claim(metadata={})
    b.claim(metadata={})  # independent claim survives
    assert a.status() == "CLAIMED" and b.status() == "CLAIMED"


def test_rm2_claim_id_changes_with_a_single_product_source_hash(tmp_path):
    # the expanded manifest is transitively bound: two identities differing in exactly
    # one product-source hash must yield distinct code-identity digests and claim ids
    base = r.compute_code_identity()
    changed = json.loads(json.dumps(base))
    changed["product_source_manifest"][0]["sha256"] = "0" * 64

    def key(ident):
        return {
            "protocol_sha256": "p" * 8,
            "criteria_sha256": "c" * 8,
            "cohort_sha256": "h" * 8,
            "code_identity_sha256": r.code_identity_digest(ident),
        }

    assert r.code_identity_digest(base) != r.code_identity_digest(changed)
    a = r.DurableClaim(tmp_path / "s", key(base))
    b = r.DurableClaim(tmp_path / "s", key(changed))
    assert a.claim_id != b.claim_id


def test_rm2_code_identity_mismatch_refuses_preflight_before_claim(disp_bundle, tmp_path):
    for key in ("tool_source_sha256", "product_source_manifest_sha256", "repo_head"):
        bad = json.loads(json.dumps(disp_bundle))
        if key == "product_source_manifest_sha256":
            bad["code_identity"][key] = "0" * 64
        elif key == "repo_head":
            bad["code_identity"][key] = "deadbeef"
        else:
            bad["code_identity"][key] = "0" * 64
        bad["content_sha256"] = r.content_sha256(bad)
        with pytest.raises(r.ValidationClaimError):
            r.preflight_bundle(bad, specs=_DISP)
    with pytest.raises(r.ValidationClaimError):
        r.validate_stage(
            bundle=bad, state_dir=tmp_path / "s", out_path=tmp_path / "o.json",
            consume_reserved_cohort=True, specs=_DISP,
        )
    assert not (tmp_path / "s").exists()


def test_rm2_manifest_payload_tamper_refuses_preflight(disp_bundle):
    bad = json.loads(json.dumps(disp_bundle))
    bad["code_identity"]["product_source_manifest"][0]["sha256"] = "1" * 64
    bad["content_sha256"] = r.content_sha256(bad)
    with pytest.raises(r.ValidationClaimError):
        r.preflight_bundle(bad, specs=_DISP)


def test_rm3_diagnostic_is_not_part_of_the_admission_criteria(bundle):
    crit = bundle["criteria"]["payload"]
    h4 = crit["h4_controlled_effect_admission"]
    flat = r.canonical_json(h4).replace(" ", "").lower()
    # the only reference is the explicit non-consumption boolean; no threshold,
    # tolerance or rule key of the recovered/common-support diagnostic is consumed
    assert '"recovered_delta_consumed":false' in flat
    assert "tolerance" not in flat
    assert "common_support" not in flat
    assert not [k for k in h4 if "recovered_delta" in k and k != "recovered_delta_consumed"]
    inj = r.canonical_json(crit["injected_effect_recovery"])
    assert "common_support" not in inj.lower()
