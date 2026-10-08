"""ZM-ZEGRID-R22 rework-1 — runtime GPU→CPU whole-batch fallback (H1) end-to-end.

Proves the H1 fix: a CuPy OOM/CUDA error DURING the GPU cell batch degrades to an
exact-CPU WHOLE-BATCH rerun (WARN + recorded), produces bit-identical output to a
forced-CPU baseline, leaves no partial cache / no double-count, and an ARBITRARY
(non-GPU) error still fails (never swallowed). Runs entirely without a physical
GPU by monkeypatching the GPU probe/planner + injecting a classified OOM at the
stack seam.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import gpu as zgpu  # noqa: E402

TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"

# A classified CuPy OOM (module "cupy.*", name "OutOfMemoryError") so
# zgpu.is_gpu_runtime_error() classifies it as a runtime GPU failure.
_FakeCupyOOM = type("OutOfMemoryError", (Exception,), {"__module__": "cupy.cuda.memory"})


@pytest.fixture(scope="module")
def routine_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("r22_runtime_corpus")
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(out)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


def _prepare_input(input_dir, routine_corpus):
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(routine_corpus.glob("*.fits")):
        (input_dir / p.name).write_bytes(p.read_bytes())
    (input_dir / "stack_plan.csv").write_text(
        (routine_corpus / "stack_plan.csv").read_text(), encoding="utf-8"
    )


def _science_sha256(out_dir):
    with fits.open(out_dir / "mosaic_grid.fits") as hdul:
        data = np.ascontiguousarray(np.asarray(hdul[0].data, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


def _force_gpu_selection(monkeypatch):
    """Make the start-time GPU decision succeed WITHOUT a physical GPU."""
    monkeypatch.setattr(zgpu, "probe_gpu_backend", lambda: {
        "available": True, "device": "FakeGPU", "cupy_version": "fake",
        "vram_total_bytes": 8 * 2**30, "vram_free_bytes": int(7.5 * 2**30),
        "reason": None,
    })
    monkeypatch.setattr(zgpu, "vram_budget_bytes", lambda probe: 4 * 2**30)
    monkeypatch.setattr(zgpu, "choose_gpu_tile_size", lambda *a, **k: 128)


def _inject_oom_at_stack_seam(monkeypatch):
    real_run = zz.run_canonical_stack_streaming

    def fake_run(provider, request, tile_size=None, fixed=None):
        if getattr(request, "backend", "cpu") == "gpu":
            raise _FakeCupyOOM("simulated device OOM during phase-2")
        return real_run(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", fake_run)


def test_runtime_gpu_oom_falls_back_to_cpu_whole_batch(tmp_path, routine_corpus, monkeypatch):
    # Forced-CPU baseline (GPU never selected).
    cpu_in, cpu_out = tmp_path / "in_cpu", tmp_path / "out_cpu"
    _prepare_input(cpu_in, routine_corpus)
    zz.run_zegrid_mode(str(cpu_in), str(cpu_out))
    hash_cpu = _science_sha256(cpu_out)

    # GPU selected at start, then a classified OOM mid-batch -> CPU whole-batch rerun.
    _force_gpu_selection(monkeypatch)
    _inject_oom_at_stack_seam(monkeypatch)
    warns = []

    def cb(msg, prog, lvl, **kw):
        warns.append((msg, lvl))

    g_in, g_out = tmp_path / "in_g", tmp_path / "out_g"
    _prepare_input(g_in, routine_corpus)
    zz.run_zegrid_mode(str(g_in), str(g_out), use_gpu=True, progress_callback=cb)

    # 1. bit-identical to the forced-CPU baseline.
    assert _science_sha256(g_out) == hash_cpu

    # 2. loud WARN about the runtime GPU failure.
    assert any("GPU RUNTIME failure" in m and lvl == "WARN" for m, lvl in warns)

    # 3. manifest records the runtime fallback + final exact CPU backend.
    import json
    manifest = json.loads((g_out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    gpu = manifest["gpu"]
    assert gpu["requested"] is True
    assert gpu["attempted"] is True
    assert gpu["used"] is False
    assert gpu["actually_used"] is False
    assert gpu["final_backend"] == "cpu"
    assert gpu["runtime_fallback"] is not None
    assert gpu["runtime_fallback"]["type"] == "OutOfMemoryError"
    assert gpu["backend"]["per_cell_rejection"] == "cpu"
    assert gpu["backend"]["per_cell_combine"] == "cpu"

    # 4. no partial per-cell cache left behind (bounded disk + cleanup).
    cache_root = g_out / zz.CACHE_DIR_NAME
    if cache_root.exists():
        leftovers = [p for p in cache_root.rglob("*.npy") if p.is_file()]
        assert leftovers == [], f"leftover cache files: {leftovers}"

    # 5. no double-count: cell count matches the layout's cells.
    assert len(manifest["cells"]) == manifest["layout"]["nx"] * manifest["layout"]["ny"]


def test_arbitrary_error_not_swallowed(tmp_path, routine_corpus, monkeypatch):
    """H1 boundary: a non-GPU error in the GPU batch must FAIL, not silently
    fall back to CPU (only classified CuPy errors trigger the rerun)."""
    _force_gpu_selection(monkeypatch)
    real_run = zz.run_canonical_stack_streaming

    def fake_run(provider, request, tile_size=None, fixed=None):
        if getattr(request, "backend", "cpu") == "gpu":
            raise ValueError("a genuine science bug, not a GPU error")
        return real_run(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", fake_run)

    g_in, g_out = tmp_path / "in_g", tmp_path / "out_g"
    _prepare_input(g_in, routine_corpus)
    with pytest.raises(ValueError):
        zz.run_zegrid_mode(str(g_in), str(g_out), use_gpu=True)
