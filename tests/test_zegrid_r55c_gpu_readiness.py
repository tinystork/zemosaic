"""ZM-ZEGRID-R55-C — portable GPU readiness + provisional VRAM policy contract.

Acceptance proven here:

* importing the new module never imports CuPy or initialises CUDA (standalone
  import proof; the pre-existing ``reproject`` -> ``dask`` opportunistic CuPy
  import is documented as a baseline limitation, not introduced here);
* synthetic no-CuPy / unimportable / no-device / probe-error / multi-device
  probes are truthful and never crash the CPU path;
* the VRAM policy is integer-only, floor-exact, never inflates, refuses
  bool/float/stale observations, and is provisional/observe-only;
* the candidate ladder and shrink-only transitions are exact, and a GPU
  abandonment is a whole-batch exact CPU fallback with no mixed result;
* the additive manifest block is deterministic and default-off, so R55-B
  observe output stays byte-identical;
* CuPy remains an optional ``gpu`` extra and the active 50% legacy budget is
  untouched.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from zemosaic.core.zegrid import gpu_readiness as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "zemosaic" / "core" / "zegrid" / "gpu_readiness.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# 1. Laziness
# ---------------------------------------------------------------------------

def _standalone_import_script() -> str:
    return (
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('gr_standalone', {str(MODULE_PATH)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['gr_standalone'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'cupy' not in sys.modules, 'cupy imported by gpu_readiness'\n"
        "assert 'cupy.cuda' not in sys.modules, 'cupy.cuda imported'\n"
        "assert mod.SCHEMA == 'zemosaic-r55-gpu-readiness-v1'\n"
        "assert mod.gpu_extra_declared() is True\n"
        "print('LAZY_OK')\n"
    )


def test_importing_gpu_readiness_does_not_import_cupy():
    proc = subprocess.run(
        [sys.executable, "-c", _standalone_import_script()],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "LAZY_OK" in proc.stdout


def test_module_has_no_top_level_cupy_import():
    import ast

    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [getattr(node, "module", "") or ""]
            )
            assert not any(n == "cupy" or n.startswith("cupy.") for n in names), (
                "gpu_readiness must not import cupy at module scope"
            )


# ---------------------------------------------------------------------------
# 2. Synthetic readiness probes
# ---------------------------------------------------------------------------

class _FakeCUDARTError(Exception):
    pass


_FakeCUDARTError.__name__ = "CUDARTError"
_FakeCUDARTError.__module__ = "cupy.cuda.cudart"


def _fake_cupy(
    *,
    count=1,
    name=b"GeForce MX150",
    mem=(2 * 2 ** 30, 4 * 2 ** 30),
    pool=(10, 20),
    pinned=None,
):
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: count,
        getDeviceProperties=lambda idx: {"name": name},
        memGetInfo=lambda: mem,
        driverGetVersion=lambda: 12040,
        runtimeGetVersion=lambda: 12090,
    )
    cupy = types.SimpleNamespace(
        __version__="14.1.1",
        cuda=types.SimpleNamespace(runtime=runtime),
        get_default_memory_pool=lambda: types.SimpleNamespace(
            used_bytes=lambda: pool[0], total_bytes=lambda: pool[1]
        ),
    )
    if pinned is not None:
        cupy.get_default_pinned_memory_pool = lambda: types.SimpleNamespace(
            used_bytes=lambda: pinned[0], total_bytes=lambda: pinned[1]
        )
    return cupy


def _patch_probe(monkeypatch, *, spec=True, cupy=None, canonical=True, importer=None):
    monkeypatch.setattr(gr, "_find_cupy_spec", lambda: object() if spec else None)
    if importer is not None:
        monkeypatch.setattr(gr, "_import_cupy", importer)
    else:
        monkeypatch.setattr(gr, "_import_cupy", lambda: cupy)
    monkeypatch.setattr(gr, "_canonical_available", lambda: canonical)


def test_dependency_missing_is_truthful_and_does_not_import(monkeypatch):
    def _must_not_import():
        raise AssertionError("CuPy must not be imported when the spec is absent")

    monkeypatch.setattr(gr, "_find_cupy_spec", lambda: None)
    monkeypatch.setattr(gr, "_import_cupy", _must_not_import)
    snap = gr.probe_gpu_readiness(declared=True)
    assert snap.status == gr.STATUS_DEPENDENCY_MISSING
    assert snap.reason_code == gr.REASON_GPU_DEP_MISSING
    assert snap.available is False
    assert snap.dependency_declared is True
    assert snap.dependency_importable is False
    snap.to_dict()


def test_unimportable_dependency_is_truthful(monkeypatch):
    def _boom():
        raise ImportError("cupy backend missing")

    _patch_probe(monkeypatch, spec=True, importer=_boom)
    snap = gr.probe_gpu_readiness(declared=True)
    assert snap.status == gr.STATUS_DEPENDENCY_UNIMPORTABLE
    assert snap.reason_code == gr.REASON_GPU_DEP_UNIMPORTABLE
    assert snap.dependency_importable is False
    assert snap.errors == ("cupy_import:ImportError",)


def test_no_device_never_claims_gpu(monkeypatch):
    _patch_probe(monkeypatch, cupy=_fake_cupy(count=0), canonical=False)
    snap = gr.probe_gpu_readiness()
    assert snap.status == gr.STATUS_NO_DEVICE
    assert snap.reason_code == gr.REASON_GPU_NO_DEVICE
    assert snap.available is False
    assert snap.device_count == 0
    assert snap.device_index is None


def test_driver_probe_error_is_classified_without_crashing(monkeypatch):
    cupy = _fake_cupy()
    cupy.cuda.runtime.getDeviceCount = lambda: (_ for _ in ()).throw(_FakeCUDARTError("boom"))
    _patch_probe(monkeypatch, cupy=cupy, canonical=False)
    snap = gr.probe_gpu_readiness()
    assert snap.status == gr.STATUS_DEVICE_PROBE_ERROR
    assert snap.reason_code == gr.REASON_GPU_DRIVER_UNAVAILABLE
    assert snap.available is False
    assert any(e.startswith("getDeviceCount:") for e in snap.errors)


def test_multi_device_metadata_is_safe(monkeypatch):
    cupy = _fake_cupy(count=2, name=b"GeForce MX150", pinned=(0, 0))
    _patch_probe(monkeypatch, cupy=cupy, canonical=True)
    snap = gr.probe_gpu_readiness(device_index=1)
    assert snap.status == gr.STATUS_USABLE
    assert snap.available is True
    assert snap.device_count == 2
    assert snap.device_index == 1
    assert snap.device_name == "GeForce MX150"
    assert snap.cupy_version == "14.1.1"
    assert snap.driver_version == 12040
    assert snap.runtime_version == 12090
    assert snap.vram_total_bytes == 4 * 2 ** 30
    assert snap.vram_free_bytes == 2 * 2 ** 30
    assert snap.pool_used_bytes == 10 and snap.pool_total_bytes == 20
    assert snap.pinned_pool_used_bytes == 0


def test_out_of_range_device_index_is_not_usable(monkeypatch):
    _patch_probe(monkeypatch, cupy=_fake_cupy(count=1), canonical=True)
    snap = gr.probe_gpu_readiness(device_index=2)
    assert snap.available is False
    assert snap.device_index is None


def test_readiness_refuses_inconsistent_snapshot():
    from zemosaic.core.zegrid.gpu_readiness import GpuReadiness, GpuReadinessError

    with pytest.raises(GpuReadinessError):
        GpuReadiness(
            status=gr.STATUS_USABLE,
            reason_code=gr.REASON_GPU_OK,
            available=False,
            dependency_declared=True,
            dependency_importable=True,
        )
    with pytest.raises(GpuReadinessError):
        GpuReadiness(
            status="gpu_maybe",
            reason_code=gr.REASON_GPU_OK,
            available=False,
            dependency_declared=True,
            dependency_importable=True,
        )


# ---------------------------------------------------------------------------
# 3. Pure integer policy
# ---------------------------------------------------------------------------

FREE_1GIB = 2 ** 30


def test_policy_70_percent_is_exact_integer_floor():
    assert gr.policy_budget_bytes(FREE_1GIB, 70) == 751619276
    assert gr.policy_residual_reserve_bytes(FREE_1GIB, 70) == 322122548
    assert gr.VramPolicy().initial(FREE_1GIB).percent == 70
    assert gr.VramPolicy().initial(FREE_1GIB).budget_bytes == 751619276


def test_policy_ladder_is_exact_and_ordered():
    policy = gr.VramPolicy()
    assert policy.ladder == (50, 65, 70, 75)
    expected = {
        50: 536870912,
        65: 697932185,
        70: 751619276,
        75: 805306368,
    }
    got = {c.percent: c.budget_bytes for c in policy.candidates(FREE_1GIB)}
    assert got == expected
    for c in policy.candidates(FREE_1GIB):
        assert c.residual_reserve_bytes == FREE_1GIB - c.budget_bytes
        assert c.eligible is True
        assert c.provenance == "provisional_observe_only"
        assert c.observe_only is True


def test_policy_refuses_bool_and_float_coercion():
    for bad in (True, False, 1.0, "1073741824", None):
        with pytest.raises(gr.InvalidVramObservation):
            gr.policy_budget_bytes(bad, 70)
        with pytest.raises(gr.InvalidVramObservation):
            gr.VramPolicy().candidates(bad)
    for bad_percent in (True, 70.0, "70"):
        with pytest.raises(gr.InvalidVramObservation):
            gr.policy_budget_bytes(FREE_1GIB, bad_percent)


def test_below_minimum_is_ineligible_and_never_inflated():
    free = 64 * 2 ** 20  # 64 MiB free -> 70% is ~44.8 MiB, below the 64 MiB floor
    candidate = gr.VramPolicy().initial(free)
    assert candidate.budget_bytes == (free * 70) // 100
    assert candidate.budget_bytes < gr.MIN_VRAM_BUDGET_BYTES
    assert candidate.eligible is False
    assert candidate.reason_code == gr.REASON_VRAM_BELOW_MINIMUM
    assert candidate.budget_bytes != gr.MIN_VRAM_BUDGET_BYTES


def test_minimum_floor_matches_active_legacy_planner():
    from zemosaic.core.zegrid import gpu as legacy_gpu

    assert gr.MIN_VRAM_BUDGET_BYTES == legacy_gpu.GPU_MIN_VRAM_BUDGET_BYTES
    # The legacy active budget is unchanged by R55-C.
    assert legacy_gpu.GPU_VRAM_SAFETY_FRACTION == 0.5


def test_shrink_ladder_and_growth_rejection():
    policy = gr.VramPolicy()
    assert policy.next_shrink(75, FREE_1GIB).percent == 70
    assert policy.next_shrink(70, FREE_1GIB).percent == 65
    assert policy.next_shrink(65, FREE_1GIB).percent == 50
    assert policy.next_shrink(50, FREE_1GIB) is None
    assert gr.shrink_only(75, 65) == 65
    assert gr.shrink_only(70, 70) == 70
    with pytest.raises(gr.GrowthRejected):
        gr.shrink_only(65, 70)


def test_fresh_observation_is_required():
    policy = gr.VramPolicy()
    for bad in (None, True, 1.0, "1024"):
        with pytest.raises(gr.InvalidVramObservation):
            gr.revalidate_for_unit(policy, 70, bad)


def test_revalidate_continues_when_current_rung_is_eligible():
    decision = gr.revalidate_for_unit(gr.VramPolicy(), 70, FREE_1GIB)
    assert decision.action == "continue"
    assert decision.percent == 70
    assert decision.budget_bytes == 751619276
    assert decision.reason_code == gr.REASON_VRAM_CONTINUE
    assert decision.whole_batch is True
    assert decision.mixed_partial is False


def test_revalidate_shrinks_only_downward_one_step():
    # A 300 MiB reserve gate makes 75% ineligible (only 256 MiB left) while
    # 70% still keeps 307 MiB unclaimed.
    policy = gr.VramPolicy(minimum_reserve_bytes=300 * 2 ** 20)
    decision = gr.revalidate_for_unit(policy, 75, FREE_1GIB)
    assert decision.action == "shrink"
    assert decision.percent == 70
    assert decision.reason_code == gr.REASON_VRAM_SHRINK
    assert decision.percent < 75


def test_revalidate_never_grows_after_pressure():
    policy = gr.VramPolicy()
    decision = gr.revalidate_for_unit(policy, 65, FREE_1GIB)
    assert decision.percent <= 65
    assert decision.action in ("continue", "shrink")


def test_revalidate_falls_back_to_whole_batch_cpu_when_no_rung_fits():
    policy = gr.VramPolicy()
    tiny = gr.MIN_VRAM_BUDGET_BYTES // 2
    decision = gr.revalidate_for_unit(policy, 70, tiny)
    assert decision.action == "fallback_cpu"
    assert decision.percent is None
    assert decision.budget_bytes is None
    assert decision.reason_code == gr.REASON_REFUSE_VRAM
    assert decision.whole_batch is True
    assert decision.mixed_partial is False


def test_revalidate_falls_back_before_execution_when_device_unusable():
    decision = gr.revalidate_for_unit(
        gr.VramPolicy(), 70, FREE_1GIB, device_usable=False
    )
    assert decision.action == "fallback_cpu"
    assert decision.reason_code == gr.REASON_GPU_FALLBACK_PRE
    assert decision.whole_batch is True
    assert decision.mixed_partial is False


def test_policy_provenance_is_provisional_only():
    from zemosaic.core.zegrid.gpu_readiness import GpuReadinessError, VramPolicy

    with pytest.raises(GpuReadinessError):
        VramPolicy(provenance="product_qualified")
    contract = gr.canonical_contract_dict()["policy"]
    assert contract["provenance"] == "provisional_observe_only"
    assert contract["observe_only"] is True
    assert contract["product_qualified"] is False
    assert contract["execution_authority"] is False
    assert contract["initial_percent"] == 70


# ---------------------------------------------------------------------------
# 4. Deterministic additive manifest + packaging
# ---------------------------------------------------------------------------

def test_manifest_is_deterministic_and_default_off():
    block = gr.readiness_manifest_dict(None)
    first = json.dumps(block, sort_keys=True, separators=(",", ":"))
    second = json.dumps(gr.readiness_manifest_dict(None), sort_keys=True, separators=(",", ":"))
    assert first == second
    assert block["readiness"]["status"] == gr.STATUS_NOT_PROBED
    assert block["observe_only"] is True
    assert block["execution_authority"] is False
    assert block["observed_metrics"] is None
    assert block["candidates"] == []
    assert set(block) == {
        "schema",
        "observe_only",
        "execution_authority",
        "readiness",
        "policy",
        "initial_candidate",
        "candidates",
        "observed_metrics",
        "note",
    }
    assert "timestamp" not in json.dumps(block).lower()


def test_manifest_with_observation_reports_candidates():
    snap = gr.GpuReadiness(
        status=gr.STATUS_USABLE,
        reason_code=gr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="GeForce MX150",
        vram_total_bytes=4 * 2 ** 30,
        vram_free_bytes=FREE_1GIB,
    )
    block = gr.readiness_manifest_dict(snap, free_bytes=FREE_1GIB)
    assert block["readiness"]["observed"] is True
    assert block["initial_candidate"]["percent"] == 70
    assert [c["percent"] for c in block["candidates"]] == [50, 65, 70, 75]


def test_scheduler_manifest_is_unchanged_by_default():
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from test_zegrid_r55_observe_only import _manifest_block
    finally:
        sys.path.pop(0)
    block = _manifest_block()
    assert "gpu_readiness" not in block
    assert block["schema"] == "zemosaic-r55-manifest-v1"


def test_optional_extra_declared_and_cupy_not_mandatory():
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    base = " ".join(data["project"]["dependencies"]).lower()
    assert "cupy" not in base
    assert data["project"]["optional-dependencies"]["gpu"] == ["cupy-cuda12x"]
    assert gr.gpu_extra_declared() is True


def test_canonical_contract_dict_is_pure_and_stable():
    a = gr.canonical_contract_dict()
    b = gr.canonical_contract_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["statuses"] == list(gr.STATUSES)
    assert a["reasons"] == sorted(gr.REASONS)
    # Contract evidence is hardware-free: it contains no volatile reading key.
    assert "vram_free_bytes" not in json.dumps(a)
