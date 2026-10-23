"""ZM-ZEGRID-R55-C rework-2 — truthful post-execution provenance.

Proves the final review finding is closed:

* a classified runtime GPU failure that performs the existing exact whole-batch
  CPU rerun leaves ``r55_scheduler.gpu_readiness.product.effective_backend`` at
  ``cpu`` while ``requested_backend`` stays ``gpu``;
* the pre-execution selection is never silently relabelled as execution;
* successful GPU execution stays ``gpu``;
* normal CPU observe is ``cpu`` (directly proven in rework-1 and re-asserted via
  the helper);
* reconciliation is copy-on-write, shape-tolerant, and never fabricates a block
  in legacy mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import gpu_readiness as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_zegrid_r22_runtime_fallback import (  # noqa: E402
    _force_gpu_selection,
    _inject_oom_at_stack_seam,
    _prepare_input,
    _science_sha256,
    routine_corpus,  # noqa: F401  (pytest fixture)
)

WITNESS_AVAILABLE_BYTES = 12 * 2 ** 30
WITNESS_HALO_PX = 32


def _usable_snapshot() -> gr.GpuReadiness:
    return gr.GpuReadiness(
        status=gr.STATUS_USABLE,
        reason_code=gr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="NVIDIA GeForce MX150",
        driver_version=12040,
        runtime_version=12090,
        vram_total_bytes=4 * 2 ** 30,
        vram_free_bytes=2 * 2 ** 30,
    )


def _run(inp, out, *, mode, use_gpu=False, monkeypatch=None):
    mp = monkeypatch or pytest.MonkeyPatch()
    try:
        mp.setattr(zz, "available_memory_bytes", lambda: WITNESS_AVAILABLE_BYTES)
        zz.run_zegrid_mode(
            str(inp),
            str(out),
            zegrid_halo_px=WITNESS_HALO_PX,
            zegrid_scheduler_mode=mode,
            use_gpu=use_gpu,
        )
    finally:
        if monkeypatch is None:
            mp.undo()


def test_runtime_gpu_oom_fallback_reports_cpu_effective_in_observe(
    tmp_path, routine_corpus, monkeypatch
):
    # Forced-CPU scientific baseline (legacy; no R55 block).
    cpu_in, cpu_out = tmp_path / "in_cpu", tmp_path / "out_cpu"
    _prepare_input(cpu_in, routine_corpus)
    _run(cpu_in, cpu_out, mode="legacy")
    cpu_hash = _science_sha256(cpu_out)

    # GPU selected at start, readiness probe says usable, then a classified OOM
    # mid-batch triggers the existing whole-batch exact CPU rerun.
    _force_gpu_selection(monkeypatch)
    _inject_oom_at_stack_seam(monkeypatch)
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe", lambda **_kw: _usable_snapshot()
    )

    g_in, g_out = tmp_path / "in_g", tmp_path / "out_g"
    _prepare_input(g_in, routine_corpus)
    _run(g_in, g_out, mode="observe", use_gpu=True, monkeypatch=monkeypatch)

    # The science output is the successful CPU rerun.
    assert _science_sha256(g_out) == cpu_hash

    manifest = json.loads((g_out / "zegrid_manifest.json").read_text())
    gpu = manifest["gpu"]
    assert gpu["requested"] is True
    # NOTE: ``attempted`` is reset to False by the existing runtime-fallback
    # finalization (pre-existing R22 behaviour); the meaningful post-execution
    # fields are ``actually_used``/``final_backend``/``runtime_fallback``.
    assert gpu["used"] is False
    assert gpu["actually_used"] is False
    assert gpu["final_backend"] == "cpu"
    assert gpu["runtime_fallback"] is not None

    r55 = manifest["r55_scheduler"]
    readiness = r55["gpu_readiness"]
    product = readiness["product"]
    # Requested provenance is preserved; effective is reconciled to the backend
    # that actually produced the output.
    assert product["requested_backend"] == "gpu"
    assert product["available"] is True
    assert product["effective_backend"] == "cpu"
    assert product["policy_applied"] is False
    assert readiness["readiness"]["status"] == gr.STATUS_USABLE
    assert readiness["observe_only"] is True
    assert readiness["execution_authority"] is False


def test_reconcile_helper_covers_gpu_cpu_and_malformed_shapes():
    def _block(effective):
        return {
            "gpu_readiness": {
                "product": {
                    "requested_backend": "gpu",
                    "available": True,
                    "effective_backend": effective,
                    "policy_applied": False,
                }
            }
        }

    # Successful GPU execution stays GPU.
    gpu_ctx = {"actually_used": True, "final_backend": "gpu"}
    out = zz._r55_reconcile_gpu_readiness(_block("gpu"), gpu_ctx)
    assert out["gpu_readiness"]["product"]["effective_backend"] == "gpu"
    assert out["gpu_readiness"]["product"]["requested_backend"] == "gpu"

    # Pre-execution GPU selection but CPU actually produced the output -> CPU.
    stale = _block("gpu")
    out = zz._r55_reconcile_gpu_readiness(
        stale, {"actually_used": False, "final_backend": "cpu"}
    )
    assert out["gpu_readiness"]["product"]["effective_backend"] == "cpu"
    assert out["gpu_readiness"]["product"]["requested_backend"] == "gpu"
    # Copy-on-write: the original pre-execution block is not mutated.
    assert stale["gpu_readiness"]["product"]["effective_backend"] == "gpu"

    # "actually_used" alone is not enough; final_backend must prove GPU.
    out = zz._r55_reconcile_gpu_readiness(
        _block("gpu"), {"actually_used": True, "final_backend": "cpu"}
    )
    assert out["gpu_readiness"]["product"]["effective_backend"] == "cpu"

    # Normal CPU observe is CPU.
    out = zz._r55_reconcile_gpu_readiness(
        _block("cpu"), {"actually_used": False, "final_backend": "cpu"}
    )
    assert out["gpu_readiness"]["product"]["effective_backend"] == "cpu"

    # Legacy / malformed inputs are preserved, never fabricated, never raise.
    assert zz._r55_reconcile_gpu_readiness(None, gpu_ctx) is None
    assert zz._r55_reconcile_gpu_readiness({}, gpu_ctx) == {}
    assert zz._r55_reconcile_gpu_readiness({"gpu_readiness": None}, gpu_ctx) == {
        "gpu_readiness": None
    }
    assert zz._r55_reconcile_gpu_readiness(
        {"gpu_readiness": {"product": "bad"}}, gpu_ctx
    ) == {"gpu_readiness": {"product": "bad"}}
    assert zz._r55_reconcile_gpu_readiness(_block("gpu"), None)[
        "gpu_readiness"
    ]["product"]["effective_backend"] == "cpu"
