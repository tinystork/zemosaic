"""ZM-GAUGE-COUNTS-DECOUPLING — distinct counts workers under ONE admission.

Mission ``ZM-GAUGE-COUNTS-DECOUPLING-20261115``. These regressions pin:

* the counts-only per-worker cost is DERIVED from the frozen model peaks
  (``decode``/``counts_reproject``/``counts_prepare``) with the process
  baseline charged exactly once and sequential sub-phases never summed;
* the historical pairs footprint (and therefore the separate reference/canonical
  peak it covers) is unchanged;
* explicit OFF keeps facts / requests / diagnostics byte-identical;
* the explicit ON path prices ``max(Nc*Fc, Np*Fp)`` under the SAME B_plan and
  15 % margin, with Nc reduced by budget and refused when it does not fit;
* a swap credit collapses Nc = Np = 1 strictly;
* a restored gauge never prices a counts pool;
* ``compute_global_gauge(counts_workers=...)`` sizes ONLY the counts pool
  (pairs keeps ``workers``) and the science payload stays bit-identical;
* CPU, external and planned-GPU seams consume the same frozen facts.

No physical GPU, no M33 data, no long benchmark.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_zegrid_r19_pairs_parallel as R19
import test_zegrid_r32a_gauge_perf as R32A
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic import zemosaic_config as zconf
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid.executor import ExecutorConfig

MIB = 2 ** 20
GIB = 2 ** 30
LIGHTS = Path("/home/tristan/M106/lights")


# ---------------------------------------------------------------------------
# Corpus / bundle helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m106():
    if not LIGHTS.is_dir():
        pytest.skip("M106 lights directory absent")
    frames, _rejected = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    return frames, canvas


@pytest.fixture(autouse=True)
def _large_disk(monkeypatch):
    """Deterministic disk observation for planning/admission (header only)."""
    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(total=8 * 2 ** 40, used=0, free=8 * 2 ** 40),
    )


def _bundle(m106, *, raw, budget, cpu=16, halo=16, decoupling=False, gpu=False,
            vram=None):
    frames, canvas = m106
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: cpu)
        mp.setattr(
            zz.shutil,
            "disk_usage",
            lambda _p: SimpleNamespace(
                total=8 * 2 ** 40, used=0, free=8 * 2 ** 40
            ),
        )
        kwargs = {}
        if decoupling is not None:
            kwargs["gauge_counts_decoupling"] = bool(decoupling)
        return zz._build_execution_bundle(
            canvas,
            frames,
            None,
            raw_available_bytes=int(raw),
            budget_bytes=int(budget),
            halo_px=int(halo),
            cpu_count=int(cpu),
            gpu_used=bool(gpu),
            vram_budget_bytes=(None if vram is None else int(vram)),
            cache_root="/tmp",
            **kwargs,
        )


def _decide(m106, bundle, *, halo=16, cpu=16, gauge_restored=False,
            resident=None, gauge_credit=None):
    frames, canvas = m106
    frame_ids = tuple(f.frame_id.logical_path for f in frames)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: cpu)
        return zz._r55_active_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=int(halo),
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=frame_ids,
            gauge_restored=bool(gauge_restored),
            restored_gauge_resident_bytes=resident,
            gauge_credit=gauge_credit,
        )


def _gauge_request(facts):
    return next(
        r for r in zz._r55_active_phase_requests(facts) if r.phase == "gauge"
    )


class _CreditingSession:
    """Minimal credit session: always grants ONE gauge worker."""

    def __init__(self):
        self.calls = []

    def decision_for_bundle(self, bundle, *, p_gauge_raw):
        self.calls.append(("decision", int(p_gauge_raw)))
        return SimpleNamespace(granted=True), 1

    def grants_credit(self, decision):
        return True

    def plan_credit(self, decision, *, workers):
        self.calls.append(("plan", int(workers)))
        return None


# ---------------------------------------------------------------------------
# 1. Model derivation
# ---------------------------------------------------------------------------

def test_counts_footprint_is_baseline_plus_max_counts_peak():
    model = zphot.gauge_memory_model(
        canvas_px=3477 * 4106,
        max_bbox_px=1975 * 1163,
        max_frame_px=1920 * 1080,
        normalization="sky_mean",
        weighting="noise_variance",
    )
    peaks = {
        name: int(entry["added_bytes"])
        for name, entry in model["phase_peaks"].items()
    }
    baseline = int(model["components"]["process_baseline_bytes"])
    expected = baseline + max(
        peaks["decode"], peaks["counts_reproject"], peaks["counts_prepare"]
    )
    assert zphot.gauge_counts_footprint_bytes(model) == expected
    # The pairs bound is untouched and still covers the separate reference
    # phase; the counts leg is strictly smaller on this geometry.
    assert model["footprint_bytes"] == baseline + max(peaks.values())
    assert zphot.gauge_counts_footprint_bytes(model) <= model["footprint_bytes"]
    assert peaks["reference_tile"] > 0


def test_counts_footprint_zero_accounting_and_empty_model():
    zero = {
        "zero_accounting": True,
        "phase_peaks": {},
        "components": {"process_baseline_bytes": 0},
    }
    assert zphot.gauge_counts_footprint_bytes(zero) == 0
    assert zphot.gauge_counts_footprint_bytes({}) == 0
    assert zphot.gauge_counts_footprint_bytes(None) == 0


def test_counts_leg_ignores_pairs_only_workspaces():
    """The counts cost cannot be inflated by canonical/weighting method choice."""
    counts = []
    for normalization, weighting in (
        ("sky_mean", "noise_variance"),
        ("linear_fit", "noise_fwhm"),
    ):
        model = zphot.gauge_memory_model(
            canvas_px=800 * 600,
            max_bbox_px=500 * 400,
            max_frame_px=256 * 256,
            normalization=normalization,
            weighting=weighting,
        )
        value = zphot.gauge_counts_footprint_bytes(model)
        assert value < int(model["footprint_bytes"])
        assert value == int(
            model["components"]["process_baseline_bytes"]
        ) + max(
            int(model["phase_peaks"][name]["added_bytes"])
            for name in ("decode", "counts_reproject", "counts_prepare")
        )
        counts.append(value)
    assert counts[0] == counts[1]


# ---------------------------------------------------------------------------
# 2. Config option: strict, default ON since beta 5.1.4
# ---------------------------------------------------------------------------

def test_config_option_defaults_on_and_is_strict():
    assert zconf.DEFAULT_CONFIG["zegrid_gauge_counts_decoupling"] is True
    assert (
        zconf.validate_zegrid_run_policy({})[
            "zegrid_gauge_counts_decoupling"
        ]
        is True
    )
    assert (
        zconf.validate_zegrid_run_policy(
            {"zegrid_gauge_counts_decoupling": True}
        )["zegrid_gauge_counts_decoupling"]
        is True
    )
    for bad in (1, "true", None):
        with pytest.raises(ValueError):
            zconf.validate_zegrid_run_policy(
                {"zegrid_gauge_counts_decoupling": bad}
            )


def _write_config(tmp_path, payload):
    path = tmp_path / "zemosaic_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_json_to_effective_helper_is_executable(tmp_path):
    """Persisted JSON -> load_config -> namespace -> the SAME pure helper."""
    path = _write_config(
        tmp_path, {"zegrid_gauge_counts_decoupling": True}
    )
    cfg = zconf.load_config(path)
    assert cfg["zegrid_gauge_counts_decoupling"] is True
    ns = SimpleNamespace(**cfg)
    assert zz._resolve_gauge_counts_decoupling(
        ns, zz.zsched.MODE_ACTIVE
    ) is True
    assert zz._resolve_gauge_counts_decoupling(
        ns, zz.zsched.MODE_LEGACY
    ) is False
    assert zz._resolve_gauge_counts_decoupling(
        ns, zz.zsched.MODE_OBSERVE
    ) is False
    # Mapping inputs follow the same rule (no hidden namespace requirement).
    assert zz._resolve_gauge_counts_decoupling(
        {"zegrid_gauge_counts_decoupling": True},
        zz.zsched.MODE_ACTIVE,
    ) is True

    false_path = _write_config(
        tmp_path, {"zegrid_gauge_counts_decoupling": False}
    )
    false_cfg = SimpleNamespace(**zconf.load_config(false_path))
    assert zz._resolve_gauge_counts_decoupling(
        false_cfg, zz.zsched.MODE_ACTIVE
    ) is False

    absent_path = _write_config(tmp_path, {"unrelated": 1})
    absent_cfg = SimpleNamespace(**zconf.load_config(absent_path))
    assert absent_cfg.zegrid_gauge_counts_decoupling is True
    assert zz._resolve_gauge_counts_decoupling(
        absent_cfg, zz.zsched.MODE_ACTIVE
    ) is True

    assert zz._resolve_gauge_counts_decoupling({}, zz.zsched.MODE_ACTIVE) is True
    assert zz._resolve_gauge_counts_decoupling(SimpleNamespace(), zz.zsched.MODE_ACTIVE) is True
    assert zz._resolve_gauge_counts_decoupling(None, zz.zsched.MODE_ACTIVE) is False
    # Fail closed on any non-boolean explicit value; the strict validator is
    # still the authority that REFUSES such a value before this helper.
    assert zz._resolve_gauge_counts_decoupling(
        SimpleNamespace(zegrid_gauge_counts_decoupling=1),
        zz.zsched.MODE_ACTIVE,
    ) is False
    assert zz._resolve_gauge_counts_decoupling(
        SimpleNamespace(zegrid_gauge_counts_decoupling=True),
        "observe",
    ) is False


# ---------------------------------------------------------------------------
# 3. OFF is byte-identical at the facts/request/diagnostic seam
# ---------------------------------------------------------------------------

def test_off_bundle_facts_and_request_keep_historical_shape(m106):
    bundle = _bundle(
        m106,
        raw=(8992230604 * 5 + 3) // 4,
        budget=8992230604,
        decoupling=False,
    )
    assert bundle.gauge_counts_workers is None
    assert bundle.gauge_counts_footprint_bytes is None
    decision = _decide(m106, bundle)
    facts = decision.facts
    assert "gauge_counts_workers" not in facts
    assert "gauge_counts_footprint_bytes" not in facts
    assert facts["gauge_phase_bytes"] == (
        int(facts["gauge_footprint_bytes"]) * int(facts["gauge_workers"])
    )
    request = _gauge_request(facts)
    assert request.components == {
        "gauge_worker_footprint_bytes": (
            int(facts["gauge_footprint_bytes"]) * int(facts["gauge_workers"])
        )
    }
    assert request.units == int(facts["gauge_workers"])
    detail = zz._r55_worker_adaptation_detail(
        facts, decision.worker_trail
    )
    assert "selected_gauge_counts_workers" not in detail
    assert "gauge_sequential_peak_bytes" not in detail


def test_off_plan_hash_and_payload_identical_to_legacy_default(m106):
    raw = (8992230604 * 5 + 3) // 4
    legacy = _bundle(
        m106, raw=raw, budget=8992230604, decoupling=None
    )
    explicit_off = _bundle(
        m106, raw=raw, budget=8992230604, decoupling=False
    )
    assert legacy.gauge_counts_workers is None
    assert explicit_off.gauge_counts_workers is None
    assert legacy.gauge_memory_model == explicit_off.gauge_memory_model
    assert legacy.envelope.to_dict() == explicit_off.envelope.to_dict()
    d_legacy = _decide(m106, legacy)
    d_off = _decide(m106, explicit_off)
    assert d_legacy.plan_set.plan_hash == d_off.plan_set.plan_hash
    assert d_legacy.mapping == d_off.mapping
    assert _gauge_request(d_legacy.facts).components == _gauge_request(
        d_off.facts
    ).components


def test_off_shrink_trial_keeps_historical_diagnostic_payload():
    fp = 8364172360
    facts = dict(
        budget_bytes=16 * GIB,
        margin_num=15,
        margin_den=100,
        gauge_workers=3,
        cache_workers=1,
        cache_footprint_bytes=1024,
        gauge_footprint_bytes=fp,
        gauge_phase_bytes=3 * fp,
        units_total=10,
        canvas_px=75_698_820,
        halo_band_bytes=0,
        retained_core_bytes=0,
        channels=3,
        cell_ids=(),
        frame_ids=(),
        layout=(2, 2),
        refinement_rank=4,
        worst_patch_hw=(10, 10),
        max_n_frames=10,
        disk_free=10 ** 9,
        run_id="r",
        pinned=False,
    )
    trial = zz._r55_trial_with_workers(facts, 1, 1)
    # OFF: no new key and the historical stale phase_bytes is preserved.
    assert "gauge_sequential_peak_bytes" not in trial
    assert trial["gauge_phase_bytes"] == 3 * fp
    assert _gauge_request(trial).components == {
        "gauge_worker_footprint_bytes": fp
    }
    detail = zz._r55_worker_adaptation_detail(trial, [])
    assert "gauge_sequential_peak_bytes" not in detail
    assert detail["gauge_phase"]["phase_bytes"] == 3 * fp


# ---------------------------------------------------------------------------
# 4. ON: derivation, sequential envelope and margin
# ---------------------------------------------------------------------------

def test_on_derives_distinct_counts_workers_under_same_budget(m106):
    budget = 3221225472  # 3 GiB -> Np 3, Nc 4 on the M106 model
    bundle = _bundle(
        m106,
        raw=(budget * 5) // 4,
        budget=budget,
        decoupling=True,
    )
    model = bundle.gauge_memory_model
    fc = int(zphot.gauge_counts_footprint_bytes(model))
    fp = int(model["footprint_bytes"])
    assert bundle.gauge_counts_footprint_bytes == fc
    assert bundle.gauge_counts_workers == (
        zz._r55_coherent_worker_count(None, bundle.worker_budget_bytes, fc)
    )
    assert fc <= fp
    assert bundle.gauge_counts_workers > bundle.gauge_workers
    # ALL chargeable workers fit the margin-aware raw ceiling independently.
    assert bundle.gauge_counts_workers * fc <= bundle.worker_budget_bytes
    assert bundle.gauge_workers * fp <= bundle.worker_budget_bytes

    decision = _decide(m106, bundle)
    facts = decision.facts
    nc = int(facts["gauge_counts_workers"])
    np_ = int(facts["gauge_workers"])
    assert nc == bundle.gauge_counts_workers
    assert facts["gauge_sequential_peak_bytes"] == max(nc * fc, np_ * fp)
    request = _gauge_request(facts)
    assert request.components == {
        "gauge_sequential_peak_bytes": max(nc * fc, np_ * fp)
    }
    # Published concurrency stays the historical pairs metadata; the counts
    # pool is a diagnostic, never a second in-flight unit of the same phase.
    assert request.units == np_
    assert request.units_in_flight == np_
    # The ONE margin at the aggregate boundary still admits (PlanSet publishes
    # the margin-applied aggregate; it must fit the SAME B_plan).
    assert (
        decision.plan_set.aggregate_host_peak_bytes
        <= int(facts["budget_bytes"])
    )
    detail = zz._r55_worker_adaptation_detail(facts, decision.worker_trail)
    assert detail["selected_gauge_counts_workers"] == nc
    assert detail["gauge_counts_footprint_bytes"] == fc
    assert detail["gauge_sequential_peak_bytes"] == max(nc * fc, np_ * fp)


def test_budget_reduces_counts_and_then_refuses(m106):
    small = 1073741824  # 1 GiB -> only ONE counts worker fits
    bundle = _bundle(
        m106,
        raw=(small * 5) // 4,
        budget=small,
        decoupling=True,
    )
    fc = int(bundle.gauge_counts_footprint_bytes)
    assert bundle.gauge_counts_workers == 1
    assert bundle.gauge_workers == 1
    assert fc <= bundle.worker_budget_bytes

    decision = _decide(m106, bundle)
    assert decision.facts["gauge_counts_workers"] == 1

    # An UNACCOUNTED counts count (never produced by the planner) is refused by
    # the SAME admission instead of silently running.
    inflated = replace(bundle, gauge_counts_workers=64)
    with pytest.raises(zz.zsched.PlanRefused) as refused:
        _decide(m106, inflated)
    assert refused.value.reason_code == zz.renv.REASON_REFUSE_HOST_RAM
    assert "counts_workers=64" in refused.value.detail
    assert "max(counts,pairs)" in refused.value.detail


# ---------------------------------------------------------------------------
# 4b. F2 — diagnostics follow the TRIAL's Np (pair->counts dominance)
# ---------------------------------------------------------------------------

def test_shrink_diagnostics_follow_the_effective_sequential_peak():
    """Pair dominates at bundle Np=3; counts dominate after the trial Np=1."""
    fc = 4528480960
    fp = 8364172360
    facts = dict(
        budget_bytes=16 * GIB,
        margin_num=15,
        margin_den=100,
        gauge_workers=3,
        cache_workers=1,
        cache_footprint_bytes=1024,
        gauge_footprint_bytes=fp,
        gauge_phase_bytes=max(4 * fc, 3 * fp),
        gauge_counts_workers=4,
        gauge_counts_footprint_bytes=fc,
        gauge_sequential_peak_bytes=max(4 * fc, 3 * fp),
        units_total=10,
        canvas_px=75_698_820,
        halo_band_bytes=0,
        retained_core_bytes=0,
        channels=3,
        cell_ids=(),
        frame_ids=(),
        layout=(2, 2),
        refinement_rank=4,
        worst_patch_hw=(10, 10),
        max_n_frames=10,
        disk_free=10 ** 9,
        run_id="r",
        pinned=False,
    )
    assert 4 * fc > 1 * fp  # counts dominate after the shrink to Np=1
    assert 3 * fp > 4 * fc  # pair dominates at the bundle Np=3

    assert zz._r55_gauge_phase_peak_bytes(facts) == 3 * fp
    initial_detail = zz._r55_worker_adaptation_detail(facts, [])
    assert initial_detail["gauge_sequential_peak_bytes"] == 3 * fp
    assert initial_detail["gauge_phase"]["phase_bytes"] == 3 * fp

    trial = zz._r55_trial_with_workers(facts, 1, 1)
    assert trial["gauge_phase_bytes"] == 4 * fc
    assert trial["gauge_sequential_peak_bytes"] == 4 * fc
    request = _gauge_request(trial)
    assert request.components == {"gauge_sequential_peak_bytes": 4 * fc}
    assert request.units == 1
    detail = zz._r55_worker_adaptation_detail(trial, [])
    assert detail["gauge_sequential_peak_bytes"] == 4 * fc
    assert detail["gauge_phase"]["phase_bytes"] == 4 * fc
    # The refusal trace names the SAME effective peak, never the stale bundle
    # value nor the pairs leg alone.
    bundle_like = SimpleNamespace(
        gauge_memory_model={
            "mode": "r32c_bounded_tiled",
            "peak_phase": "pairs_normalize",
            "max_bbox_px": 1,
            "tile_px": 1024,
            "canvas_px": 75_698_820,
        }
    )
    trace = zz._gauge_phase_refusal_components(trial, bundle_like, 75_698_820)
    assert f"{4 * fc}" in trace
    assert "max(counts,pairs)" in trace
    assert f"{3 * fp}" not in trace


# ---------------------------------------------------------------------------
# 5. Credit => strictly 1/1
# ---------------------------------------------------------------------------

def test_swap_credit_forces_one_counts_and_one_pairs_worker(m106):
    budget = 3221225472
    bundle = _bundle(
        m106,
        raw=(budget * 5) // 4,
        budget=budget,
        decoupling=True,
    )
    assert bundle.gauge_counts_workers > 1
    session = _CreditingSession()
    decision = _decide(m106, bundle, gauge_credit=session)
    facts = decision.facts
    assert facts["gauge_credit_applied"] is True
    assert facts["gauge_workers"] == 1
    assert facts["gauge_counts_workers"] == 1
    fp = int(facts["gauge_footprint_bytes"])
    assert facts["gauge_phase_bytes"] == fp
    assert facts["gauge_sequential_peak_bytes"] == fp
    detail = zz._r55_worker_adaptation_detail(facts, decision.worker_trail)
    assert detail["gauge_phase"]["phase_bytes"] == fp
    assert detail["gauge_sequential_peak_bytes"] == fp
    request = _gauge_request(facts)
    assert request.components == {"gauge_sequential_peak_bytes": fp}
    # The credit is still priced on the SINGLETON pairs footprint, never the
    # decoupled max.
    assert ("decision", fp) in session.calls


def _force_gauge_shrink(monkeypatch, *, kind, threshold):
    """Force a selection shrink at the FIRST (highest Np) trial only."""
    zs = zz.zsched
    if kind == "cpu":
        real = zs.plan_joint

        def fake(joint, *, phase_requests, mode):
            gauge = next(
                r for r in phase_requests if r.phase == "gauge"
            )
            peak = int(gauge.components.get("gauge_sequential_peak_bytes", 0))
            if peak > threshold:
                raise zs.PlanRefused(
                    zz.renv.REASON_REFUSE_HOST_RAM,
                    "forced F2 shrink at the highest Np",
                )
            return real(joint, phase_requests=phase_requests, mode=mode)

        monkeypatch.setattr(zs, "plan_joint", fake)
        return

    real = zs.build_joint_plan_set

    def fake(joint, *, candidate, phase_requests, mode):
        gauge = next(r for r in phase_requests if r.phase == "gauge")
        peak = int(gauge.components.get("gauge_sequential_peak_bytes", 0))
        if peak > threshold:
            raise zs.PlanRefused(
                zz.renv.REASON_REFUSE_HOST_RAM,
                "forced F2 shrink at the highest Np",
            )
        return real(
            joint,
            candidate=candidate,
            phase_requests=phase_requests,
            mode=mode,
        )

    monkeypatch.setattr(zs, "build_joint_plan_set", fake)


def _assert_trail_dominance(trail, fc, fp):
    values = {}
    for entry in trail:
        np_ = int(entry["gauge_workers"])
        expected = max(fc, np_ * fp)
        assert entry["gauge_phase_bytes"] == expected
        values[np_] = expected
    return values


def test_cpu_shrink_trail_reports_pair_then_counts(m106, monkeypatch):
    """Bundle Np=3 pair-bound; shrink to Np=2 makes counts dominate."""
    budget = 3221225472
    bundle = _bundle(
        m106, raw=(budget * 5) // 4, budget=budget, decoupling=True
    )
    fp = int(bundle.gauge_memory_model["footprint_bytes"])
    fc = 2_000_000_000
    assert bundle.gauge_workers == 3
    assert 3 * fp > fc > 2 * fp
    inflated = replace(
        bundle, gauge_counts_workers=1, gauge_counts_footprint_bytes=fc
    )
    _force_gauge_shrink(monkeypatch, kind="cpu", threshold=2_100_000_000)
    decision = _decide(m106, inflated)
    assert decision.gauge_workers == 2
    facts = decision.facts
    assert facts["gauge_phase_bytes"] == fc
    assert facts["gauge_sequential_peak_bytes"] == fc
    assert _gauge_request(facts).components == {
        "gauge_sequential_peak_bytes": fc
    }
    detail = zz._r55_worker_adaptation_detail(facts, decision.worker_trail)
    assert detail["gauge_sequential_peak_bytes"] == fc
    assert detail["gauge_phase"]["phase_bytes"] == fc
    values = _assert_trail_dominance(decision.worker_trail, fc, fp)
    assert values[3] == 3 * fp  # failed trial: pairs dominate
    assert values[2] == fc      # accepted trial: counts dominate


def test_planned_gpu_shrink_trail_reports_pair_then_counts(m106, monkeypatch):
    from test_zegrid_r55e8_coherent_resource_adaptation import (
        _probe,
        _readiness,
    )

    frames, canvas = m106
    budget = 3221225472
    vram = 2 * GIB
    bundle = _bundle(
        m106,
        raw=(budget * 5) // 4,
        budget=budget,
        halo=16,
        decoupling=True,
        gpu=True,
        vram=vram,
    )
    fp = int(bundle.gauge_memory_model["footprint_bytes"])
    fc = 2_000_000_000
    inflated = replace(
        bundle, gauge_counts_workers=1, gauge_counts_footprint_bytes=fc
    )
    _force_gauge_shrink(monkeypatch, kind="gpu", threshold=2_100_000_000)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: 16)
        mp.setattr(
            zz.shutil,
            "disk_usage",
            lambda _p: SimpleNamespace(
                total=8 * 2 ** 40, used=0, free=8 * 2 ** 40
            ),
        )
        decision = zz._r55_active_gpu_decide(
            bundle=inflated,
            cell_ctxs=inflated.cell_ctxs,
            canvas=canvas,
            halo_px=16,
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=tuple(f.frame_id.logical_path for f in frames),
            gpu_probe=_probe(vram, vram),
            readiness=_readiness(vram, vram),
        )
    assert decision.mapping["backend"] == "gpu"
    assert decision.gauge_workers == 2
    facts = decision.facts
    assert facts["gauge_phase_bytes"] == fc
    assert _gauge_request(facts).components == {
        "gauge_sequential_peak_bytes": fc
    }
    values = _assert_trail_dominance(decision.worker_trail, fc, fp)
    assert values[3] == 3 * fp
    assert values[2] == fc


# ---------------------------------------------------------------------------
# 6. Restored gauge never prices a counts pool
# ---------------------------------------------------------------------------

def test_restored_gauge_excludes_counts_pool(m106):
    budget = 3221225472
    bundle = _bundle(
        m106,
        raw=(budget * 5) // 4,
        budget=budget,
        decoupling=True,
    )
    decision = _decide(
        m106, bundle, gauge_restored=True, resident=96 * MIB
    )
    facts = decision.facts
    assert facts["gauge_restored"] is True
    assert "gauge_counts_workers" not in facts
    request = _gauge_request(facts)
    assert request.components == {
        "restored_gauge_resident_bytes": 96 * MIB
    }
    assert request.units == 1


# ---------------------------------------------------------------------------
# 7. Execution: counts pool only, science payload bit-identical
# ---------------------------------------------------------------------------

def test_compute_global_gauge_counts_workers_only_sizes_counts_pool(
    tmp_path, monkeypatch
):
    descs = R19._build_corpus_disk(tmp_path)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    calls = []
    real_pmap = zphot.zpar.pmap

    def _spy(fn, tasks, workers, **kw):
        calls.append((fn.__name__, int(workers)))
        return real_pmap(fn, tasks, workers, **kw)

    monkeypatch.setattr(zphot.zpar, "pmap", _spy)
    diag = {}
    gauge, ids = zphot.compute_global_gauge(
        descs,
        canvas,
        R19._decode_disk,
        config,
        None,
        workers=1,
        counts_workers=3,
        diagnostics=diag,
    )
    assert calls == [("_counts_worker", 3), ("_pair_worker", 1)]
    assert diag["workers"] == 1
    assert diag["counts_workers"] == 3
    assert diag["phases"]["counts"]["workers"] == 3
    assert "workers" not in diag["phases"]["pairs"]
    # Real process parallelism for the counts pool (no hidden serial fallback).
    assert diag["phases"]["counts"]["executor"] == "process"

    ref, ref_ids = zphot.compute_global_gauge(
        descs, canvas, R19._decode_disk, config, None, workers=1
    )
    R19._assert_bit_equal(ref, ref_ids, R19._gauge_payload(gauge, ids))
    # Default OFF diagnostics keep their historical shape.
    off_diag = {}
    zphot.compute_global_gauge(
        descs,
        canvas,
        R19._decode_disk,
        config,
        None,
        workers=1,
        diagnostics=off_diag,
    )
    assert "counts_workers" not in off_diag
    assert "workers" not in off_diag["phases"]["counts"]


def _spawn_counts_entry(q, descs, canvas, config):
    multiprocessing.set_start_method("spawn", force=True)
    diag = {}
    try:
        gauge, ids = zphot.compute_global_gauge(
            descs,
            canvas,
            R19._decode_disk,
            config,
            None,
            workers=1,
            counts_workers=3,
            diagnostics=diag,
        )
        q.put(
            {
                "status": "ok",
                **R19._gauge_payload(gauge, ids),
                "counts_workers": diag.get("counts_workers"),
                "counts_executor": diag["phases"]["counts"]["executor"],
            }
        )
    except BaseException as exc:  # noqa: BLE001
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def test_counts_workers_spawn_bit_equal(tmp_path):
    descs = R19._build_corpus_disk(tmp_path)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()
    ref, ref_ids = zphot.compute_global_gauge(
        descs, canvas, R19._decode_disk, config, None, workers=1
    )
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_spawn_counts_entry, args=(q, descs, canvas, config)
    )
    proc.start()
    proc.join(timeout=600)
    assert proc.exitcode == 0, f"spawn child crashed: {proc.exitcode}"
    assert not q.empty(), "spawn child produced no result"
    payload = q.get()
    assert payload["status"] == "ok", payload.get("error")
    assert payload["counts_workers"] == 3
    assert payload["counts_executor"] == "process"
    R19._assert_bit_equal(ref, ref_ids, payload)


def test_counts_workers_parity_on_synthetic_dithered_corpus():
    descs, _offsets = R32A._dithered_corpus()
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()
    ref, ref_ids = zphot.compute_global_gauge(
        descs, canvas, R32A._decode, config, None, workers=1
    )
    for nc in (1, 2, 4):
        diag = {}
        got, ids = zphot.compute_global_gauge(
            descs,
            canvas,
            R32A._decode,
            config,
            None,
            workers=1,
            counts_workers=nc,
            diagnostics=diag,
        )
        R19._assert_bit_equal(ref, ref_ids, R19._gauge_payload(got, ids))
        assert diag["counts_workers"] == nc


def test_product_entry_threads_the_option_to_bundle_and_execution():
    """The ONE product wiring: config -> bundle -> decision -> pmap kwarg."""
    import inspect

    entry = inspect.getsource(zz.run_zegrid_mode)
    single = inspect.getsource(zz._run_single)
    assert "validate_zegrid_run_policy" in entry
    # Read once per pass, gated to the ACTIVE scheduler whose admission prices
    # the separate pool.
    assert "_resolve_gauge_counts_decoupling(" in single
    assert "zconfig, scheduler_mode" in single
    # Frozen into EVERY bundle build of the pass (initial + rebuilds/fallbacks).
    assert "gauge_counts_decoupling=bool(_gauge_counts_decoupling)" in single
    # The admitted decision feeds the execution kwarg; OFF passes None.
    assert "counts_workers=_gauge_counts_workers" in single
    assert '_gauge_counts_facts.get("gauge_counts_workers")' in single


# ---------------------------------------------------------------------------
# 8. External CPU class and planned GPU class
# ---------------------------------------------------------------------------

def _external_bundle(tmp_path, descs, *, decoupling):
    from test_zegrid_active_external_wiring import _FakeModel, _tiny_floors

    canvas = zg.build_canvas(descs)
    raw = 2 * GIB
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
        mp.setattr(zal, "ScientificFloors", _tiny_floors)
        mp.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62))
        mp.setattr(
            zz.shutil,
            "disk_usage",
            lambda _p: SimpleNamespace(
                total=100 * GIB, used=0, free=100 * GIB
            ),
        )
        mp.setattr(os, "cpu_count", lambda: 8)
        return zz._build_execution_bundle(
            canvas,
            descs,
            None,
            raw_available_bytes=raw,
            budget_bytes=(4 * raw) // 5,
            cpu_count=8,
            cache_root=str(tmp_path / "cache"),
            force_external=True,
            pinned_layout=(4, 4),
            max_refinement_rank=16,
            halo_px=32,
            gauge_counts_decoupling=bool(decoupling),
        )


def test_external_cpu_class_consumes_counts_facts(tmp_path):
    from test_zegrid_r32c_gauge_footprint_wiring import _wide_geometry_corpus

    descs = _wide_geometry_corpus()
    bundle = _external_bundle(tmp_path, descs, decoupling=True)
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.gauge_counts_workers is not None
    decision = zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=zg.build_canvas(descs),
        halo_px=32,
        cache_root=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "out"),
        frame_ids=tuple(d.frame_id.logical_path for d in descs),
    )
    assert decision.working_set_class == zz.WORKING_SET_EXTERNAL
    facts = decision.facts
    assert facts["gauge_counts_workers"] == bundle.gauge_counts_workers
    request = _gauge_request(facts)
    assert list(request.components) == ["gauge_sequential_peak_bytes"]
    expected = max(
        int(facts["gauge_counts_workers"])
        * int(facts["gauge_counts_footprint_bytes"]),
        int(facts["gauge_workers"]) * int(facts["gauge_footprint_bytes"]),
    )
    assert request.components["gauge_sequential_peak_bytes"] == expected


def test_planned_gpu_class_consumes_counts_facts_without_physical_gpu(m106):
    from test_zegrid_r55e8_coherent_resource_adaptation import (
        _probe,
        _readiness,
    )

    budget = 8992230604
    vram = 2 * GIB
    bundle = _bundle(
        m106,
        raw=(budget * 5 + 3) // 4,
        budget=budget,
        cpu=16,
        halo=128,
        decoupling=True,
        gpu=True,
        vram=vram,
    )
    assert bundle.gauge_counts_workers > 1
    frames, canvas = m106
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz.os, "cpu_count", lambda: 16)
        mp.setattr(
            zz.shutil,
            "disk_usage",
            lambda _p: SimpleNamespace(
                total=8 * 2 ** 40, used=0, free=8 * 2 ** 40
            ),
        )
        decision = zz._r55_active_gpu_decide(
            bundle=bundle,
            cell_ctxs=bundle.cell_ctxs,
            canvas=canvas,
            halo_px=128,
            cache_root="/tmp",
            output_dir="/tmp",
            frame_ids=tuple(f.frame_id.logical_path for f in frames),
            gpu_probe=_probe(vram, vram),
            readiness=_readiness(vram, vram),
        )
    facts = decision.facts
    assert decision.mapping["backend"] == "gpu"
    assert facts["gauge_counts_workers"] == bundle.gauge_counts_workers
    request = _gauge_request(facts)
    expected = max(
        int(facts["gauge_counts_workers"])
        * int(facts["gauge_counts_footprint_bytes"]),
        int(facts["gauge_workers"]) * int(facts["gauge_footprint_bytes"]),
    )
    assert request.components["gauge_sequential_peak_bytes"] == expected
