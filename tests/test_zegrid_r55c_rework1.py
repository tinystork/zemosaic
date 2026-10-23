"""ZM-ZEGRID-R55-C rework-1 — regression, wiring, laziness and semantics.

Covers the confirmed review defects:

1. ``PlanRefused.refusal_manifest_dict`` returns the full block again (closed
   reason, candidate trail, observe/active modes).
2. ``PlanSet.to_manifest_dict`` default-off byte-equivalence and the opt-in
   ``gpu_readiness`` block (unprobed and supplied synthetic snapshots).
3. Product ``zegrid_scheduler_mode=observe`` wiring: additive block on the CPU
   observe path, truthful requested/available/effective distinction, unchanged
   science/coverage FITS, contained probe failure.
4. Real package-import laziness (normal imports, not ``spec_from_file_location``)
   and compatibility of every historical package export.
5. Device-context correctness is covered in the base R55-C test file; here we
   pin the corrected RUN revalidation semantics (per-rung requirements vs one
   fixed requirement).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import gpu_readiness as gr
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import scheduler as zs

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
MODULE = "zemosaic.core.zegrid.gpu_readiness"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_zegrid_r55_observe_only import (  # noqa: E402
    WITNESS_AVAILABLE_BYTES,
    WITNESS_HALO_PX,
    routine_input,  # noqa: F401  (pytest fixture)
)


# ---------------------------------------------------------------------------
# 1. PlanRefused regression
# ---------------------------------------------------------------------------

def _candidate() -> zs.Candidate:
    return zs.Candidate(
        backend=re.BACKEND_CPU,
        tile_px=64,
        units_in_flight=1,
        prefetch_units=0,
        workers_per_unit=1,
        host_peak_bytes=100,
        disk_bytes=0,
        device_peak_bytes=0,
        feasible=False,
        reason_code=re.REASON_REFUSE_HOST_RAM,
        infeasible_reason="synthetic",
    )


def test_refusal_manifest_returns_full_block():
    refusal = zs.PlanRefused(
        re.REASON_REFUSE_HOST_RAM, "synthetic detail", candidates=(_candidate(),)
    )
    block = refusal.refusal_manifest_dict(mode=zs.MODE_OBSERVE, run_id="rid-1")
    assert isinstance(block, dict)
    assert block["schema"] == zs.MANIFEST_SCHEMA
    assert block["mode"] == "observe_only"
    assert block["requested_mode"] == zs.MODE_OBSERVE
    assert block["run_id"] == "rid-1"
    assert block["refused"] is True
    assert block["reason_code"] == re.REASON_REFUSE_HOST_RAM
    assert block["reason_code"] in re.REASON_CODES
    assert block["detail"] == "synthetic detail"
    assert block["plans"] == [_candidate().to_dict()]
    assert block["plans"][0]["tile_px"] == 64


def test_refusal_manifest_active_mode_has_no_observe_only_label():
    refusal = zs.PlanRefused(re.REASON_REFUSE_VRAM, "detail")
    block = refusal.refusal_manifest_dict(mode="legacy", run_id="")
    assert block["mode"] == "active"
    assert block["requested_mode"] == "legacy"
    assert block["plans"] == []
    assert block["reason_code"] in re.REASON_CODES


# ---------------------------------------------------------------------------
# 2. include_gpu_readiness default-off / opt-in
# ---------------------------------------------------------------------------

def _observe_plan() -> zs.PlanSet:
    requests = (
        zs.PhaseRequest(
            phase=re.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": 100},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 100},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_ASSEMBLY,
            components={"halo_band_bytes": 10, "retained_core_bytes": 100},
            units=1,
            units_in_flight=1,
            overlaps_with=(re.PHASE_CELL_STACK_STREAM,),
        ),
        zs.PhaseRequest(
            phase=re.PHASE_EXPORT,
            components={"coverage_plane_bytes": 10, "science_planes_bytes": 30},
            units=1,
            units_in_flight=1,
        ),
    )
    return zs.build_observe_plan(
        phase=re.PHASE_CELL_STACK_STREAM,
        budget_bytes=4 * 2 ** 30,
        raw_available_bytes=5 * 2 ** 30,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=4,
        shared_bytes=340 * 2 ** 20,
        per_unit_stack_bytes=200 * 2 ** 20,
        per_unit_cache_bytes=10 * 2 ** 20,
        per_unit_prefetch_bytes=10 * 2 ** 20,
        disk_free_bytes=100 * 2 ** 30,
        layout=(2, 2),
        refinement_rank=4,
        patch_hw=(300, 300),
        halo_px=32,
        channels=3,
        n_frames=20,
        phase_requests=requests,
        cell_ids=("r0000c0000", "r0000c0001", "r0000c0002", "r0000c0003"),
        frame_ids=("f0", "f1"),
        mode=zs.MODE_OBSERVE,
        run_id="rework-plan",
    )


def _snapshot(available=True, free=2 ** 30, status=None, reason=None) -> gr.GpuReadiness:
    return gr.GpuReadiness(
        status=status or (gr.STATUS_USABLE if available else gr.STATUS_DEPENDENCY_MISSING),
        reason_code=reason or (gr.REASON_GPU_OK if available else gr.REASON_GPU_DEP_MISSING),
        available=available,
        dependency_declared=True,
        dependency_importable=available,
        cupy_version="14.1.1" if available else None,
        device_count=1 if available else 0,
        device_index=0 if available else None,
        device_name="NVIDIA GeForce MX150" if available else None,
        vram_total_bytes=4 * 2 ** 30 if available else None,
        vram_free_bytes=free if available else None,
    )


def test_manifest_default_off_is_byte_equivalent():
    plan = _observe_plan()
    default = plan.to_manifest_dict()
    explicit_false = plan.to_manifest_dict(include_gpu_readiness=False)
    assert "gpu_readiness" not in default
    assert json.dumps(default, sort_keys=True) == json.dumps(
        explicit_false, sort_keys=True
    )


def test_manifest_opt_in_unprobed_and_synthetic_readiness():
    plan = _observe_plan()
    unprobed = plan.to_manifest_dict(include_gpu_readiness=True)
    block = unprobed["gpu_readiness"]
    assert block["readiness"]["status"] == gr.STATUS_NOT_PROBED
    assert block["readiness"]["observed"] is False
    assert block["product"] == {
        "requested_backend": None,
        "available": False,
        "effective_backend": None,
        "policy_applied": False,
    }
    assert block["policy"]["observe_only"] is True

    supplied = plan.to_manifest_dict(
        include_gpu_readiness=True,
        gpu_readiness=_snapshot(available=True),
        requested_backend="gpu",
        effective_backend="cpu",
    )["gpu_readiness"]
    assert supplied["readiness"]["observed"] is True
    assert supplied["readiness"]["available"] is True
    assert supplied["product"] == {
        "requested_backend": "gpu",
        "available": True,
        "effective_backend": "cpu",
        "policy_applied": False,
    }
    # Same structure both times (deterministic key set).
    assert set(unprobed["gpu_readiness"]) == set(supplied)


# ---------------------------------------------------------------------------
# 3. Product observe wiring
# ---------------------------------------------------------------------------

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(inp, out, mode):
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(zz, "available_memory_bytes", lambda: WITNESS_AVAILABLE_BYTES)
        zz.run_zegrid_mode(
            str(inp),
            str(out),
            zegrid_halo_px=WITNESS_HALO_PX,
            zegrid_scheduler_mode=mode,
        )
    finally:
        monkeypatch.undo()


def test_product_observe_carries_gpu_readiness_and_preserves_fits(
    routine_input, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe", lambda **_kw: _snapshot(available=False)
    )
    legacy_out = tmp_path / "legacy"
    observe_out = tmp_path / "observe"
    _run(routine_input, legacy_out, "legacy")
    _run(routine_input, observe_out, "observe")

    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        assert _digest(legacy_out / name) == _digest(observe_out / name), name

    legacy = json.loads((legacy_out / "zegrid_manifest.json").read_text())
    observe = json.loads((observe_out / "zegrid_manifest.json").read_text())
    assert "r55_scheduler" not in legacy
    assert "gpu_readiness" not in legacy

    block = observe["r55_scheduler"]
    assert "gpu_readiness" in block
    gpu = block["gpu_readiness"]
    assert set(gpu) == {
        "schema",
        "observe_only",
        "execution_authority",
        "readiness",
        "product",
        "policy",
        "initial_candidate",
        "candidates",
        "observed_metrics",
        "note",
    }
    assert gpu["observe_only"] is True
    assert gpu["execution_authority"] is False
    assert gpu["readiness"]["status"] == gr.STATUS_DEPENDENCY_MISSING
    assert gpu["readiness"]["reason_code"] == gr.REASON_GPU_DEP_MISSING
    assert gpu["product"] == {
        "requested_backend": "cpu",
        "available": False,
        "effective_backend": "cpu",
        "policy_applied": False,
    }
    assert gpu["policy"]["initial_percent"] == 70
    assert gpu["policy"]["provenance"] == "provisional_observe_only"
    # Deterministic structure: serialising the same block twice is identical.
    assert json.dumps(gpu, sort_keys=True) == json.dumps(
        json.loads(json.dumps(gpu)), sort_keys=True
    )
    # Science/coverage manifest fields are unchanged by the additive block.
    observe_without = dict(observe)
    observe_without.pop("r55_scheduler", None)
    for key in ("r55_scheduler", "timings", "peak_rss_kib", "aggregate_peak_rss_kib",
                "gauge_diagnostics", "per_cell_diagnostics", "reprojection"):
        legacy.pop(key, None)
        observe_without.pop(key, None)
    assert legacy == observe_without


def test_product_helper_requested_available_effective_are_separate():
    cpu = zz._r55_gpu_readiness_observe_block(
        {"requested": False, "available": True}, False, _snapshot(available=True)
    )
    assert cpu["product"] == {
        "requested_backend": "cpu",
        "available": True,
        "effective_backend": "cpu",
        "policy_applied": False,
    }

    requested_gpu_fallback = zz._r55_gpu_readiness_observe_block(
        {"requested": True, "available": False}, False, _snapshot(available=False)
    )
    assert requested_gpu_fallback["product"] == {
        "requested_backend": "gpu",
        "available": False,
        "effective_backend": "cpu",
        "policy_applied": False,
    }

    gpu_effective = zz._r55_gpu_readiness_observe_block(
        {"requested": True, "available": True}, True, _snapshot(available=True)
    )
    assert gpu_effective["product"]["requested_backend"] == "gpu"
    assert gpu_effective["product"]["effective_backend"] == "gpu"
    assert gpu_effective["product"]["available"] is True


def test_product_helper_contains_probe_failure_with_closed_reason(monkeypatch):
    def _boom(**_kwargs):
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(zz.zgr, "probe_gpu_readiness", _boom)
    block = zz._r55_gpu_readiness_observe_block({"requested": True}, False)
    assert block["readiness"]["status"] == gr.STATUS_DEVICE_PROBE_ERROR
    assert block["readiness"]["reason_code"] == gr.REASON_GPU_PROBE_ERROR
    assert block["readiness"]["available"] is False
    assert block["product"]["effective_backend"] == "cpu"


# ---------------------------------------------------------------------------
# 4. Real package-import laziness and export compatibility
# ---------------------------------------------------------------------------

_LAZY_SCRIPT = r"""
import importlib
import sys

targets = [
    "zemosaic.core.zegrid.gpu_readiness",
    "zemosaic.core.zegrid.scheduler",
    "zemosaic.core.zegrid.resource_envelope",
    "zemosaic.core.zegrid.geometry",
]
for name in targets:
    importlib.import_module(name)

cupy_modules = [m for m in sys.modules if m == "cupy" or m.startswith("cupy.")]
assert not cupy_modules, f"CuPy imported eagerly: {cupy_modules[:5]}"
assert "cupy.cuda" not in sys.modules

import zemosaic.core.zegrid as zg
from zemosaic.core.zegrid import assembly as za
from zemosaic.core.zegrid import geometry as zgeo

assert za.__name__ == "zemosaic.core.zegrid.assembly"
assert zgeo.__name__ == "zemosaic.core.zegrid.geometry"
assert zg.assembly is za
legacy = [
    "geometry", "execution", "science_adapter", "assembly", "adequacy",
    "sweep", "seam", "executor", "mosaic", "auto_layout", "photometric",
    "observability", "final_mosaic_finishing",
]
assert sorted(zg.__all__) == sorted(legacy)
for name in legacy:
    module = getattr(zg, name)
    assert module.__name__ == f"zemosaic.core.zegrid.{name}"
for name in ("gpu", "gpu_readiness", "scheduler", "resource_envelope", "parallel"):
    module = getattr(zg, name)
    assert module.__name__ == f"zemosaic.core.zegrid.{name}"
assert "geometry" in dir(zg)
print("LAZY_PACKAGE_OK")
"""


def test_normal_package_import_does_not_import_cupy():
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "LAZY_PACKAGE_OK" in proc.stdout


# ---------------------------------------------------------------------------
# 5. Corrected RUN revalidation semantics
# ---------------------------------------------------------------------------

FREE_1GIB = 2 ** 30
MIB = 2 ** 20


def test_per_rung_requirements_allow_shrink_only_because_workspace_shrank():
    policy = gr.VramPolicy()
    requirements = {
        50: 500 * MIB,
        65: 600 * MIB,
        70: 700 * MIB,
        75: 900 * MIB,  # too big for the 75% budget (768 MiB)
    }
    decision = gr.revalidate_for_unit(
        policy,
        75,
        FREE_1GIB,
        required_bytes_by_percent=requirements,
    )
    assert decision.action == "shrink"
    assert decision.percent == 70
    assert decision.reason_code == gr.REASON_VRAM_SHRINK
    assert decision.percent < 75
    # 70% succeeds because its OWN requirement (700 MiB) shrank below the
    # 75% requirement; it is not a fixed-threshold descent.
    assert policy.candidate(FREE_1GIB, 70).budget_bytes >= 700 * MIB


def test_per_rung_requirements_can_fall_through_to_cpu():
    policy = gr.VramPolicy()
    requirements = {50: 900 * MIB, 65: 900 * MIB, 70: 900 * MIB, 75: 900 * MIB}
    decision = gr.revalidate_for_unit(
        policy, 70, FREE_1GIB, required_bytes_by_percent=requirements
    )
    assert decision.action == "fallback_cpu"
    assert decision.reason_code == gr.REASON_REFUSE_VRAM
    assert decision.whole_batch is True
    assert decision.mixed_partial is False


def test_fixed_impossible_requirement_never_returns_shrink():
    policy = gr.VramPolicy()
    decision = gr.revalidate_for_unit(
        policy, 75, FREE_1GIB, required_bytes=900 * MIB
    )
    assert decision.action == "fallback_cpu"
    assert decision.reason_code == gr.REASON_REFUSE_VRAM
    assert decision.percent is None


def test_requirement_mapping_must_cover_the_whole_ladder():
    policy = gr.VramPolicy()
    with pytest.raises(gr.GpuReadinessError):
        gr.revalidate_for_unit(
            policy,
            70,
            FREE_1GIB,
            required_bytes_by_percent={70: 100 * MIB},
        )


def test_requirements_are_integer_only_and_mutually_exclusive():
    policy = gr.VramPolicy()
    with pytest.raises(gr.InvalidVramObservation):
        gr.revalidate_for_unit(
            policy,
            70,
            FREE_1GIB,
            required_bytes_by_percent={50: 100 * MIB, 65: 100.5 * MIB,
                                        70: 100 * MIB, 75: 100 * MIB},
        )
    with pytest.raises(gr.GpuReadinessError):
        gr.revalidate_for_unit(
            policy,
            70,
            FREE_1GIB,
            required_bytes=100 * MIB,
            required_bytes_by_percent={50: 1, 65: 1, 70: 1, 75: 1},
        )
