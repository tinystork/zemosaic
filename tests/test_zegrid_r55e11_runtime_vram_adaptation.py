"""ZM-ZEGRID-R55-E11 — RUN-time, shrink-only GPU tile adaptation.

Acceptance coverage for the bounded correction:

* device-memory boundary lifecycle: SYNCHRONISE before purging the pool, purge
  only after the device temporaries are unreachable, best-effort on failures,
  never raising and never masking a science error;
* the exact recorded multi-cell pressure sequence (free 7463763968 ->
  5974786048) refuses the old tile but walks to 448 (or the exact next admitted
  rung), and stronger pressure walks to 384;
* no admissible rung -> exactly one whole-batch CPU fallback, no mixed result;
* the runner receives the shrunk tile (payload-carried, not a mutated global);
* the ACTIVE 70 % budget and the legacy 50 % probe budget are distinguishable;
* the original GPU decision survives a runtime fallback;
* science is bit-identical across tiles 128/384/448 (mean and median/rejection).

Model/fake only: no CuPy device is required and no long mosaic is run.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core import canonical_streaming as cstream
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import gpu as zgpu
from zemosaic.core.zegrid import gpu_memory as zgm
from zemosaic.core.zegrid import gpu_readiness as zgr
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import ram_contract as rcontract

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
GIB = 2 ** 30
MIB = 2 ** 20

# Exact recorded Windows M106 R555E facts (R55-E11 diagnostic report).
WITNESS_FREE_0 = 7463763968
WITNESS_FREE_1 = 5974786048
WITNESS_PEAK_506 = 5223334912
WITNESS_PEAK_448 = 4131895552
WITNESS_PEAK_384 = 3075963136
WITNESS_FRAMES = 66
WITNESS_PATCH_HW = (1767, 1330)


# ---------------------------------------------------------------------------
# A/B. Pure RUN-time walk on the exact recorded facts.
# ---------------------------------------------------------------------------

def _decision(tiles=(506, 448, 384), peaks=None):
    peaks = dict(peaks or {
        506: WITNESS_PEAK_506,
        448: WITNESS_PEAK_448,
        384: WITNESS_PEAK_384,
    })
    ladder = tuple(int(t) for t in tiles)
    added_bounds = tuple(
        (
            tile,
            max(
                1,
                int(
                    zz._estimate_streaming_bytes(
                        WITNESS_FRAMES, WITNESS_PATCH_HW, tile
                    )
                )
                - int(rcontract.PROCESS_BASELINE_BYTES),
            ),
        )
        for tile in ladder
    )
    return SimpleNamespace(
        policy=zgr.VramPolicy(),
        qualification={"percent": 70, "budget_bytes": 5_224_634_777},
        device_peak_bytes_by_tile=peaks,
        gpu_tile_ladder=ladder,
        gpu_host_bound_by_tile=added_bounds,
        gpu_max_contributors=WITNESS_FRAMES,
        gpu_worst_patch_hw=WITNESS_PATCH_HW,
    )


def _bundle():
    return SimpleNamespace(
        cache_workers_per_cell=1,
        shared_bytes=340 * MIB,
        cache_worker_footprint_bytes=64 * MIB,
    )


def test_e11_recorded_pressure_walks_506_to_448_not_cpu():
    decision = _decision()
    # The CURRENT (old) tile is refused by the unchanged revalidation...
    current = zz._r55_runtime_tile_check(
        decision,
        tile=506,
        free_bytes=WITNESS_FREE_1,
        host_budget_bytes=12 * GIB,
        bundle=_bundle(),
    )
    assert current["device_ok"] is False
    assert current["vram_action"] == "fallback_cpu"
    assert current["vram_reason_code"] == zgr.REASON_REFUSE_VRAM
    # ...and the walk hands the runner the exact next admitted rung.
    attempts, chosen = zz._r55_runtime_tile_walk(
        decision,
        current_tile=506,
        free_bytes=WITNESS_FREE_1,
        host_budget_bytes=12 * GIB,
        bundle=_bundle(),
    )
    assert chosen == 448
    assert [a["tile_px"] for a in attempts] == [448]
    assert attempts[0]["eligible"] is True
    assert len(attempts) <= len(decision.gpu_tile_ladder) - 1


def test_e11_stronger_pressure_walks_448_to_384():
    # 5.9 GB free at 70 % = 4.130 GB: tile448 (4.1319 GB) no longer fits.
    attempts, chosen = zz._r55_runtime_tile_walk(
        _decision(),
        current_tile=448,
        free_bytes=59_000_000_000 // 10,
        host_budget_bytes=12 * GIB,
        bundle=_bundle(),
    )
    assert chosen == 384, attempts
    assert [a["tile_px"] for a in attempts] == [384]
    assert attempts[0]["eligible"] is True


def test_e11_no_admissible_rung_is_a_single_whole_batch_refusal():
    attempts, chosen = zz._r55_runtime_tile_walk(
        _decision(),
        current_tile=506,
        free_bytes=2 * GIB,
        host_budget_bytes=12 * GIB,
        bundle=_bundle(),
    )
    assert chosen is None
    assert [a["tile_px"] for a in attempts] == [448, 384]
    assert all(not a["eligible"] for a in attempts)


def test_e11_host_pressure_also_walks_downward():
    # Device fits every rung; only the EXACT host aggregate fails at 448.
    decision = _decision()
    cap = zz._r55_gpu_host_aggregate_for_tile(decision, _bundle(), 384) + 1
    assert cap < zz._r55_gpu_host_aggregate_for_tile(
        decision, _bundle(), 448
    )
    attempts, chosen = zz._r55_runtime_tile_walk(
        decision,
        current_tile=506,
        free_bytes=WITNESS_FREE_0,
        host_budget_bytes=cap,
        bundle=_bundle(),
    )
    assert chosen == 384
    assert attempts[0]["tile_px"] == 448
    assert attempts[0]["device_ok"] is True
    assert attempts[0]["host_ok"] is False


def test_e11_walk_never_leaves_the_declared_ladder():
    decision = _decision(tiles=(506, 448, 384))
    _attempts, chosen = zz._r55_runtime_tile_walk(
        decision,
        current_tile=999,  # not a declared rung
        free_bytes=WITNESS_FREE_1,
        host_budget_bytes=12 * GIB,
        bundle=_bundle(),
    )
    assert chosen is None


# ---------------------------------------------------------------------------
# C. Device-memory boundary lifecycle.
# ---------------------------------------------------------------------------

class _FakePool:
    """Matches the REAL CuPy allocator API: byte counters are BOUND METHODS."""

    def __init__(self, log, *, fail=False):
        self.log = log
        self.fail = fail
        self._used = 100
        self._total = 200
        self._free = 100

    def used_bytes(self):
        return self._used

    def total_bytes(self):
        return self._total

    def free_bytes(self):
        return self._free

    def free_all_blocks(self):
        if self.fail:
            raise RuntimeError("pool boom")
        self.log.append("free_all_blocks")
        self._used = 0
        self._total = 100
        self._free = 100


class _FakePinnedPool:
    """Matches the REAL CuPy pinned pool: counters are BOUND METHODS."""

    def __init__(self, log, *, fail=False):
        self.log = log
        self.fail = fail
        self._used = 10
        self._total = 20

    def used_bytes(self):
        return self._used

    def total_bytes(self):
        return self._total

    def free_all_blocks(self):
        if self.fail:
            raise RuntimeError("pinned boom")
        self.log.append("pinned_free_all_blocks")


class _FakeStream:
    def __init__(self, log, *, fail=False):
        self.log = log
        self.fail = fail

    def synchronize(self):
        if self.fail:
            raise RuntimeError("sync boom")
        self.log.append("synchronize")


class _FakeDevice:
    def __init__(self, index):
        self.index = index

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeCupy:
    __name__ = "cupy"

    def __init__(self, log, *, sync_fail=False, pool_fail=False, pinned_fail=False):
        self.log = log
        self.cuda = SimpleNamespace(
            Device=_FakeDevice,
            Stream=SimpleNamespace(null=_FakeStream(log, fail=sync_fail)),
            runtime=SimpleNamespace(memGetInfo=lambda: (3 * GIB, 4 * GIB)),
        )
        self._pool = _FakePool(log, fail=pool_fail)
        self._pinned = _FakePinnedPool(log, fail=pinned_fail)

    def get_default_memory_pool(self):
        return self._pool

    def get_default_pinned_memory_pool(self):
        return self._pinned


def test_e11_boundary_synchronises_before_purging_and_reports_snapshots():
    log: list[str] = []
    xp = _FakeCupy(log)
    record = zgm.gpu_cell_boundary_cleanup(xp, reason="test")
    assert record["sync_ok"] is True
    assert record["purge_ok"] is True
    assert record["cleanup_failures"] == []
    # The historical defect was purge-before-sync. Order is now explicit.
    assert log.index("synchronize") < log.index("free_all_blocks")
    assert log.index("free_all_blocks") < log.index("pinned_free_all_blocks")
    assert record["before"]["mem_get_info_free_bytes"] == 3 * GIB
    assert record["before"]["pool_used_bytes"] == 100
    assert record["after"]["pool_used_bytes"] == 0
    assert record["label"] == zgm.GPU_MEMORY_SNAPSHOT_LABEL
    assert "high_water" not in json.dumps(record).lower().replace(
        "not_high_water", ""
    )


def test_e11_boundary_cleanup_never_raises_and_reports_failures():
    log: list[str] = []
    record = zgm.gpu_cell_boundary_cleanup(
        _FakeCupy(log, sync_fail=True, pool_fail=True)
    )
    assert record["sync_ok"] is False
    assert record["purge_ok"] is False
    assert any(item.startswith("sync:") for item in record["cleanup_failures"])
    assert any(item.startswith("purge:") for item in record["cleanup_failures"])


def test_e11_boundary_is_a_noop_for_numpy():
    import numpy as np

    record = zgm.gpu_cell_boundary_cleanup(np)
    assert record["available"] is False
    assert record["before"] is None and record["after"] is None


def test_e11_legacy_release_helper_uses_the_boundary_helper(monkeypatch):
    calls = []

    def fake_cleanup(xp=None, **kw):
        calls.append((xp, kw))
        return {"ok": True}

    monkeypatch.setattr(cstream.gpu_memory, "gpu_cell_boundary_cleanup", fake_cleanup)
    cstream._release_gpu_workspace("fake-xp", reason="tile_boundary")
    assert calls and calls[0][1]["reason"] == "tile_boundary"
    cstream._release_gpu_workspace(__import__("numpy"))
    assert len(calls) == 1  # NumPy never reaches CuPy


# ---------------------------------------------------------------------------
# D. Truthful per-unit state and budget provenance.
# ---------------------------------------------------------------------------

def test_e11_status_fields_cannot_contradict_the_action():
    state = {
        "tile": 384,
        "shrink_history": [{"from_tile_px": 448, "to_tile_px": 384}],
        "attempts": [{"tile_px": 384, "eligible": True}],
        "units": [
            {
                "unit_index": 0,
                "tile_px": 448,
                "status": "completed",
                "action": "continue",
                "attempted": True,
                "completed": True,
                "discarded": True,
            },
            {
                "unit_index": 1,
                "tile_px": 384,
                "status": "refused",
                "action": "fallback_cpu",
                "attempted": True,
                "completed": False,
                "discarded": True,
            },
        ],
    }
    block = zz._r55_tile_adaptation_from_state(
        state, _decision(), initial_tile=448
    )
    assert block["effective_tile_px"] == 384
    assert block["shrink_count"] == 1
    assert block["intra_cell_oom_supported"] is False
    for unit in block["units"]:
        if unit["status"] == "refused":
            assert unit["action"] == "fallback_cpu"
            assert unit["completed"] is False
        if unit["action"] == "fallback_cpu":
            assert unit["status"] != "completed"


def test_e11_diagnostics_distinguish_active_and_legacy_budgets():
    decision = _decision()
    decision.mapping = {"gpu_streaming_tile": 448}
    decision.gpu_tile_max = 506
    decision.gpu_host_bound_by_tile = (
        (506, 1000),
        (448, 900),
        (384, 800),
    )
    decision.gpu_selection_budget_bytes = 4_440_939_560
    bundle = SimpleNamespace(envelope=SimpleNamespace(aggregate_bytes=123))
    block = zz._r55_gpu_tile_decision_dict(
        decision,
        {"gpu_streaming_tile": 384, "prefetch_units": 1},
        bundle,
        runtime={"shrink_count": 1},
        legacy_probe_budget_bytes=3_731_881_984,
    )
    assert block["vram_percent"] == 70
    assert block["vram_budget_source"] == "active_r55_policy_70"
    assert block["legacy_probe_vram_budget_bytes"] == 3_731_881_984
    assert block["legacy_probe_percent"] == 50
    assert block["initial_tile_px"] == 448
    assert block["effective_tile_px"] == 384
    assert block["runtime_adaptation"]["shrink_count"] == 1
    assert block["engine_provenance"]["active_vram_percent"] == 70


# ---------------------------------------------------------------------------
# E. Science invariance across the shrink rungs 128 / 384 / 448.
# ---------------------------------------------------------------------------

def _corpus(side=576, frames=4):
    import numpy as np

    rng = np.random.default_rng(20261024)
    arrays, masks = [], []
    yy, xx = np.mgrid[0:side, 0:side]
    for i in range(frames):
        base = 100.0 + np.float32(i)
        img = (
            base
            + 10.0 * np.sin(xx / (17.0 + i))
            + 7.0 * np.cos(yy / (23.0 + i))
            + rng.normal(0.0, 0.5, size=(side, side))
        ).astype(np.float32)
        support = np.ones((side, side), dtype=bool)
        support[: 12 + i, :] = False
        arrays.append(img)
        masks.append(support)
    return arrays, masks


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(normalization="none", weighting="none", rejection="none", combine="mean"),
        dict(
            normalization="sky_mean",
            weighting="noise_variance",
            rejection="kappa_sigma",
            combine="median",
        ),
    ],
    ids=["mean", "median-kappa"],
)
def test_e11_science_is_bit_identical_across_128_384_448(kwargs):
    arrays, masks = _corpus()
    request = CanonicalStackRequest(
        images=arrays,
        geometric_support=masks,
        taper="footprint",
        taper_px=8.0,
        **kwargs,
    )
    reference = run_canonical_stack(request)
    results = {}
    for tile in (128, 384, 448):
        results[tile] = run_canonical_stack_streaming(
            InMemoryCanonicalProvider(arrays, masks), request, tile_size=tile
        )
    import numpy as np

    for tile, result in results.items():
        assert result.science.shape == reference.science.shape
        assert np.array_equal(result.science, reference.science, equal_nan=True), tile
        assert np.array_equal(
            result.estimator_weight_sum,
            reference.estimator_weight_sum,
            equal_nan=True,
        ), tile
        assert np.array_equal(result.valid_mask, reference.valid_mask), tile
        assert np.array_equal(
            result.surviving_sample_count,
            reference.surviving_sample_count,
        ), tile
        assert np.array_equal(result.support_w1, reference.support_w1), tile
        assert np.array_equal(result.support_w2, reference.support_w2), tile
    reference_rejection = np.asarray(reference.rejection_mask, dtype=bool)
    for tile, result in results.items():
        assert np.array_equal(
            np.asarray(result.rejection_mask, dtype=bool),
            reference_rejection,
        ), tile


# ---------------------------------------------------------------------------
# F. End-to-end pipeline witness on the routine corpus (fake CUDA, real
#    producer/consumer + ledger + manifest).
# ---------------------------------------------------------------------------

def _readiness():
    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="model-only",
        device_count=1,
        device_index=0,
        device_name="fake-gpu",
        vram_total_bytes=4 * GIB,
        vram_free_bytes=2 * GIB,
    )


def _patch_gpu(monkeypatch, free_vram_reader):
    probe = {
        "available": True,
        "device": "fake-gpu",
        "cupy_version": "model-only",
        "vram_total_bytes": 4 * GIB,
        "vram_free_bytes": 2 * GIB,
        "reason": None,
    }
    # Authority is explicit: no leaked config from another test may turn this
    # witness into a CPU run.
    monkeypatch.setattr(
        zz, "resolve_gpu_preference", lambda *a, **k: (True, "test")
    )
    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", lambda: dict(probe))
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe", lambda **kw: _readiness()
    )
    monkeypatch.setattr(zz.zgr, "read_vram_free_bytes", free_vram_reader)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 12 * GIB)
    # ``_r55_active_gpu_admit`` captures ``available_memory_bytes`` in its
    # default argument at definition time, so patching the module attribute
    # alone does not isolate admission from the host's current free RAM.
    # Inject the same deterministic observation explicitly for these model-only
    # witnesses; RUN-time checkpoints still consume the patched module function.
    real_admit = zz._r55_active_gpu_admit

    def admit_with_fixed_host_observation(**kwargs):
        return real_admit(**kwargs, available_fn=lambda: 12 * GIB)

    monkeypatch.setattr(zz, "_r55_active_gpu_admit", admit_with_fixed_host_observation)


def _sequence(values):
    iterator = iter(values)
    last = values[-1]

    def read(device_index=0):
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass
        return int(last)

    return read


def _fake_gpu_consumer(monkeypatch, seen_tiles):
    real = zz._consume_stack_cell

    def consume(prepared):
        if isinstance(prepared, dict) and "empty_result" in prepared:
            return real(prepared)
        _tile = prepared.get("_stack_tile_size")
        if _tile is None:
            # Post-fallback CPU wave: the REAL CPU semantics must be observed,
            # otherwise the witness would falsely report GPU-backed cells.
            return real(prepared)
        seen_tiles.append(int(_tile))
        cfg = zz._STACK_CONFIG
        zz._STACK_CONFIG = dataclasses.replace(cfg, backend="cpu")
        try:
            result = real(prepared)
        finally:
            zz._STACK_CONFIG = cfg
        result["backend_used"] = "gpu"
        result["record"]["backend_used"] = "gpu"
        return result

    monkeypatch.setattr(zz, "_consume_stack_cell", consume)


@pytest.fixture(scope="module")
def routine_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("r55e11_routine")
    corpus = root / "corpus"
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(corpus)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    inp = root / "input"
    inp.mkdir()
    for path in sorted(corpus.glob("*.fits")):
        shutil.copy2(path, inp / path.name)
    return inp


def _run(inp, out):
    zz.run_zegrid_mode(
        str(inp),
        str(out),
        zegrid_halo_px=32,
        zegrid_scheduler_mode="active",
        use_gpu=True,
    )


def _manifest(out):
    return json.loads((out / "zegrid_manifest.json").read_text())


def test_e11_pipeline_shrinks_and_keeps_gpu(
    routine_input, tmp_path_factory, monkeypatch
):
    _patch_gpu(
        monkeypatch,
        _sequence([2 * GIB, 2 * GIB, 2 * GIB, 400 * MIB]),
    )
    seen_tiles: list[int] = []
    _fake_gpu_consumer(monkeypatch, seen_tiles)
    out = tmp_path_factory.mktemp("r55e11_shrink")
    _run(routine_input, out)

    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    telemetry = manifest["r55_run_telemetry"]
    assert block["status"] == "executed"
    assert "gpu_fallback" not in block
    runtime = telemetry["gpu_tile_adaptation"]
    assert runtime["shrink_count"] >= 1, runtime
    assert runtime["effective_tile_px"] < runtime["initial_tile_px"]
    assert runtime["effective_tile_px"] in runtime["declared_ladder"]
    assert all(
        unit["status"] in ("completed", "empty") for unit in runtime["units"]
    )
    # The RUNNER (not a mutated global) saw the shrunk tile.
    assert seen_tiles and max(seen_tiles) > min(seen_tiles)
    pipe = telemetry["gpu_pipeline"]
    assert pipe["ledger"]["balanced"] is True
    assert pipe["threads_joined"] is True
    assert pipe["producer_alive"] is False
    # Both budgets are explicit and cannot be confused.
    assert block["gpu_tile_decision"]["vram_percent"] == 70
    assert block["gpu_tile_decision"]["vram_budget_source"] == (
        "active_r55_policy_70"
    )
    assert (
        block["gpu_tile_decision"]["legacy_probe_vram_budget_bytes"]
        is not None
    )


def test_e11_no_admissible_tile_falls_back_once_and_preserves_decision(
    routine_input, tmp_path_factory, monkeypatch
):
    # 50 MiB free: not even the smallest declared GPU rung fits the 70 % budget.
    _patch_gpu(
        monkeypatch, _sequence([2 * GIB, 2 * GIB, 2 * GIB, 50 * MIB])
    )
    seen_tiles: list[int] = []
    _fake_gpu_consumer(monkeypatch, seen_tiles)
    out = tmp_path_factory.mktemp("r55e11_fallback")
    _run(routine_input, out)

    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    assert block["status"] == "executed"
    assert "gpu_fallback" in block
    assert block["gpu_fallback"]["reason_code"] in (
        zgr.REASON_REFUSE_VRAM,
        zgr.REASON_GPU_OBSERVATION_FAILED,
    )
    # The ORIGINAL selection survives the fallback (it used to be erased).
    assert "gpu_tile_decision" in block
    assert block["gpu_original_decision"]["reason"] == (
        "runtime_fallback_preserved_selection"
    )
    assert block["gpu_tile_decision"]["selected_tile_px"] > 0
    # No mixed result: every published cell is CPU.
    assert manifest["gpu"]["actually_used"] is False
    assert manifest["gpu"]["gpu_executed_cells"] == 0
    assert all(
        record.get("backend_used") != "gpu"
        for record in manifest.get("cells", [])
    )
    # The GPU RUN diagnostics are preserved on the scheduler block (the CPU
    # retry owns r55_run_telemetry, so the abandoned GPU run must not vanish).
    runtime = block["gpu_tile_runtime"]
    assert runtime["effective_tile_px"] == runtime["initial_tile_px"]
    assert all(
        unit["discarded"] or not unit["completed"] for unit in runtime["units"]
    )
    pipe = block["gpu_fallback_pipeline"]
    assert pipe["ledger"]["balanced"] is True
    assert pipe["threads_joined"] is True
    assert pipe["producer_alive"] is False
    # The immediate failure record distinguishes a VRAM refusal from a generic
    # producer/accounting failure (R55F's unresolved class).
    diag = _failure_diag(out)
    assert diag["category"] == "vram_refusal"
    assert diag["reason_code"] == zgr.REASON_REFUSE_VRAM
    assert diag["pipeline"]["ledger"]["balanced"] is True
    assert diag["tile_state"]["tile_px"] == runtime["initial_tile_px"]


# ---------------------------------------------------------------------------
# G. Immediate failure diagnostics (R55-E11 addendum).
#
# R55F (Windows, 19:27:12) stopped with R55_GPU_PIPELINE_FAILED, no manifest and
# no detailed exception in the accessible artifacts. These tests pin the
# correction: the ORIGINAL exception (type/message/chain/traceback) plus
# producer/consumer/ledger/unit context is persisted BEFORE the whole-batch CPU
# retry, and survives a run that never reaches finalize. Generic producer and
# accounting failures are tested SEPARATELY from VRAM refusal / CUDA OOM.
# ---------------------------------------------------------------------------

def _failure_diag(out: Path) -> dict:
    return json.loads((out / "zegrid_gpu_failure.json").read_text())


def test_e11_producer_failure_diagnostics_survive_without_manifest(
    routine_input, tmp_path_factory, monkeypatch
):
    _patch_gpu(monkeypatch, _sequence([2 * GIB]))
    seen_tiles: list[int] = []
    _fake_gpu_consumer(monkeypatch, seen_tiles)

    def boom(task):
        raise RuntimeError("producer boom")

    monkeypatch.setattr(zz, "_prepare_stack_cell", boom)
    out = tmp_path_factory.mktemp("r55e11_producer_fail")
    # Capture the emission independently of pytest's caplog state (a previous
    # suite can change logger propagation/levels). The product uses the module
    # logger; a local handler observes exactly what the GUI/run-log receives.
    emitted_lines: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            try:
                emitted_lines.append(str(record.getMessage()))
            except Exception:
                pass

    handler = _Capture()
    old_level = zz.logger.level
    zz.logger.addHandler(handler)
    zz.logger.setLevel(logging.WARNING)
    try:
        with pytest.raises(RuntimeError):
            _run(routine_input, out)
    finally:
        zz.logger.removeHandler(handler)
        zz.logger.setLevel(old_level)

    # The run aborted after the fallback and wrote NO manifest...
    assert not (out / "zegrid_manifest.json").exists()
    # ...but the original cause was persisted immediately.
    diag = _failure_diag(out)
    assert diag["schema"] == zz._R55E11_FAILURE_SCHEMA
    assert diag["category"] == "producer_error"
    assert diag["stage"] == "pipeline"
    assert diag["exception"]["type"] == "RuntimeError"
    assert "producer boom" in diag["exception"]["message"]
    assert "producer boom" in diag["exception"]["traceback"]
    assert diag["exception_chain"]
    assert diag["pipeline"]["ledger"]["balanced"] is True
    assert diag["pipeline"]["threads_joined"] is True
    assert diag["pipeline"]["producer_alive"] is False
    assert diag["pipeline"]["producer_error"]
    assert diag["device"] is not None
    assert diag["host"]["parent_memory"] is not None
    # The useful reason reached the GUI/run-log line, not just a reason_code.
    emitted = "\n".join(emitted_lines)
    assert "producer boom" in emitted
    assert "zegrid_gpu_failure.json" in emitted


def test_e11_producer_accounting_failure_is_distinct_and_durable(
    routine_input, tmp_path_factory, monkeypatch
):
    _patch_gpu(monkeypatch, _sequence([2 * GIB]))
    seen_tiles: list[int] = []
    _fake_gpu_consumer(monkeypatch, seen_tiles)
    real_prepare = zz._prepare_stack_cell

    def bad_accounting(task):
        prepared = real_prepare(task)
        if "empty_result" in prepared:
            return prepared
        prepared = dict(prepared)
        prepared["manifest"] = dict(prepared["manifest"])
        # A real post-produce byte-callback/accounting failure (not VRAM/OOM).
        prepared["manifest"]["total_bytes"] = "not-an-int"
        return prepared

    monkeypatch.setattr(zz, "_prepare_stack_cell", bad_accounting)
    out = tmp_path_factory.mktemp("r55e11_accounting_fail")
    with pytest.raises(Exception):
        _run(routine_input, out)

    assert not (out / "zegrid_manifest.json").exists()
    diag = _failure_diag(out)
    assert diag["category"] == "producer_accounting_error"
    assert diag["exception"]["type"] == "PipelineError"
    assert "producer failed" in diag["exception"]["message"]
    # The ORIGINAL cause is preserved through the wrapper chain.
    assert any(
        entry["type"] in ("ValueError", "TypeError")
        for entry in diag["exception_chain"]
    )
    assert diag["extra"]["producer_error"]
    assert diag["pipeline"]["ledger"]["balanced"] is True
    assert diag["pipeline"]["threads_joined"] is True


# ---------------------------------------------------------------------------
# H. Host-credit cache-byte contract (R55F confirmed defect).
#
# The aligned cache is charged to the pipeline as manifest.total_bytes: RGB
# float32 + support bool + one 128-byte .npy header per file. The historical
# estimate counted RGB data only, so a prefetch=0 run could not stage its own
# first cell. These tests use REAL .npy files and the REAL BoundedPipeline
# ledger, and prove consume is reached at prefetch 0/1/2.
# ---------------------------------------------------------------------------

def _real_cache_payload(root: Path, name: str, side: int, frames: int) -> dict:
    import numpy as np

    cache = Path(root) / name
    cache.mkdir(parents=True, exist_ok=True)
    total = 0
    for index in range(int(frames)):
        rgb_path = cache / f"frame_{index:04d}_rgb.npy"
        sup_path = cache / f"frame_{index:04d}_support.npy"
        np.save(rgb_path, np.zeros((int(side), int(side), 3), dtype=np.float32))
        np.save(sup_path, np.zeros((int(side), int(side)), dtype=bool))
        total += int(rgb_path.stat().st_size) + int(sup_path.stat().st_size)
    return {
        "name": str(name),
        "manifest": {"total_bytes": int(total)},
        "cache_dir": str(cache),
    }


def _real_pipeline(payloads, *, budget: int, prefetch: int):
    released = []

    def release(payload):
        released.append(payload["name"])

    pipeline = zpar.BoundedPipeline(
        tasks=list(payloads),
        produce=lambda payload: payload,
        consume=lambda payload: payload["name"],
        release=release,
        prefetch_units=int(prefetch),
        host_budget_bytes=int(budget),
        device_budget_bytes=1,  # zero device bytes charged, explicit >=1 cap
        host_bytes=lambda payload: int(payload["manifest"]["total_bytes"]),
        device_bytes=lambda payload: 0,
    )
    return pipeline, released


@pytest.mark.parametrize(
    "side,frames",
    [(3, 1), (17, 2), (48, 4), (384, 3), (512, 2)],
)
def test_e11_cache_unit_contract_matches_real_npy_files(side, frames, tmp_path):
    payload = _real_cache_payload(tmp_path, "c", side, frames)
    estimated = zz._r55_cache_unit_bytes(side * side, frames)
    # Exact equality for a worst-shape cell: support + both headers included.
    assert estimated == payload["manifest"]["total_bytes"]
    # The old RGB-only estimate is strictly too small (regression witness).
    assert side * side * 3 * 4 * frames < estimated


def test_e11_prefetch0_real_manifest_ledger_reaches_consume(tmp_path):
    payloads = [
        _real_cache_payload(tmp_path, "cell_a", 48, 4),
        _real_cache_payload(tmp_path, "cell_b", 40, 3),
        _real_cache_payload(tmp_path, "cell_c", 17, 2),  # small/header-heavy
    ]
    worst_px = 48 * 48
    max_frames = 4
    contract = zz._r55_cache_unit_bytes(worst_px, max_frames)
    rgb_only = worst_px * 3 * 4 * max_frames
    assert rgb_only < contract

    # Historical defect: the RGB-only budget rejects the first real payload
    # BEFORE consume (exactly the reproduced R55F class).
    bad, bad_released = _real_pipeline(
        payloads, budget=rgb_only, prefetch=0
    )
    with pytest.raises(zpar.PipelineAccountingError):
        bad.run()
    bad_telemetry = bad.telemetry()
    assert bad_telemetry["units_consumed"] == 0
    assert bad_telemetry["units_produced"] >= 1
    assert bad_telemetry["units_released"] == bad_telemetry["units_produced"]
    assert bad_telemetry["ledger"]["balanced"] is True

    # Fixed contract: prefetch=0 reaches consume for EVERY cell.
    good, good_released = _real_pipeline(
        payloads, budget=contract, prefetch=0
    )
    assert good.run() == ["cell_a", "cell_b", "cell_c"]
    telemetry = good.telemetry()
    assert telemetry["units_consumed"] == 3
    assert telemetry["units_released"] == 3
    assert telemetry["ledger"]["balanced"] is True
    assert sorted(good_released) == ["cell_a", "cell_b", "cell_c"]


@pytest.mark.parametrize("prefetch", [1, 2])
def test_e11_prefetch_one_and_two_use_the_same_contract(tmp_path, prefetch):
    payloads = [_real_cache_payload(tmp_path, f"c{i}", 32, 3) for i in range(3)]
    per_unit = zz._r55_cache_unit_bytes(32 * 32, 3)
    budget = per_unit * (int(prefetch) + 1)
    pipeline, released = _real_pipeline(
        payloads, budget=budget, prefetch=prefetch
    )
    assert pipeline.run() == ["c0", "c1", "c2"]
    telemetry = pipeline.telemetry()
    assert telemetry["units_consumed"] == 3
    assert telemetry["ledger"]["balanced"] is True
    assert sorted(released) == ["c0", "c1", "c2"]


def test_e11_genuinely_insufficient_host_fails_before_consume(tmp_path):
    payload = _real_cache_payload(tmp_path, "worst", 48, 4)
    contract = zz._r55_cache_unit_bytes(48 * 48, 4)
    pipeline, released = _real_pipeline(
        [payload], budget=contract - 1, prefetch=0
    )
    with pytest.raises(zpar.PipelineAccountingError):
        pipeline.run()
    telemetry = pipeline.telemetry()
    assert telemetry["units_consumed"] == 0
    assert telemetry["units_produced"] == 1
    assert telemetry["units_released"] == 1
    assert telemetry["ledger"]["balanced"] is True
    assert released == ["worst"]


def test_e11_cancelled_prefetch0_pipeline_releases_everything(tmp_path):
    payload = _real_cache_payload(tmp_path, "cancelled", 32, 2)
    contract = zz._r55_cache_unit_bytes(32 * 32, 2)
    released = []
    holder = {}

    def produce(item):
        # Cancel AFTER producing the first owned unit, before it can be consumed:
        # the pipeline must release it exactly once and stay balanced.
        holder["pipeline"].cancel("test_cancel")
        return item

    pipeline = zpar.BoundedPipeline(
        tasks=[payload],
        produce=produce,
        consume=lambda item: item["name"],
        release=lambda item: released.append(item["name"]),
        prefetch_units=0,
        host_budget_bytes=contract,
        device_budget_bytes=1,
        host_bytes=lambda item: int(item["manifest"]["total_bytes"]),
        device_bytes=lambda item: 0,
    )
    holder["pipeline"] = pipeline
    try:
        pipeline.run()
    except Exception:
        pass
    telemetry = pipeline.telemetry()
    assert telemetry["units_consumed"] == 0
    assert telemetry["units_produced"] == 1
    assert telemetry["units_released"] == 1
    assert telemetry["ledger"]["balanced"] is True
    assert released == ["cancelled"]


def test_e11_product_gpu_pipeline_prefetch0_reaches_consume(
    routine_input, tmp_path_factory, monkeypatch
):
    """Product-level witness: force prefetch=0 and let the real GPU pipeline run.

    This is the exact R55F shape (queue_max=1). The unified cache contract must
    let the first prepared cell be staged and consumed.
    """
    monkeypatch.setattr(zz, "_R55E2_PREFETCH_LADDER", (0,))
    monkeypatch.setattr(zz, "_R55E2_PREFETCH_DEFAULT", 0)
    _patch_gpu(monkeypatch, _sequence([2 * GIB]))
    seen_tiles: list[int] = []
    _fake_gpu_consumer(monkeypatch, seen_tiles)
    out = tmp_path_factory.mktemp("r55e11_prefetch0")
    _run(routine_input, out)

    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    assert block["status"] == "executed"
    assert block["mapping"]["prefetch_units"] == 0
    assert block["gpu_execution"]["queue_max"] == 1
    assert "gpu_fallback" not in block
    telemetry = manifest["r55_run_telemetry"]
    assert telemetry["gpu_pipeline"]["units_consumed"] == len(seen_tiles)
    assert telemetry["gpu_pipeline"]["units_consumed"] > 0
    assert telemetry["gpu_pipeline"]["ledger"]["balanced"] is True
    assert telemetry["gpu_pipeline"]["threads_joined"] is True
    # The failure-only diagnostics file must NOT exist on a healthy run.
    assert not (out / "zegrid_gpu_failure.json").exists()


def test_e11_exact_r55f_scale_credit_mismatch_and_prefetch0():
    """Exact R55F magnitude, no 2 GB file writes.

    The recorded R55F credit was 1,861,287,120 B (RGB-only, 66 frames of
    1767x1330x3 float32) while the produced payload was ~2,014,895,190 B. The
    unified contract includes support bool + both 128-byte .npy headers.
    """
    px = 1767 * 1330
    frames = 66
    rgb_only = px * 3 * 4 * frames
    contract = zz._r55_cache_unit_bytes(px, frames)
    recorded_payload = 2_014_895_190
    assert rgb_only == 1_861_287_120
    assert contract == 2_016_411_276
    assert rgb_only < recorded_payload < contract

    payload = {
        "name": "r55f_scale",
        "manifest": {"total_bytes": recorded_payload},
    }
    # Legacy RGB-only credit: the first unit is refused BEFORE consume.
    bad, released_bad = _real_pipeline([payload], budget=rgb_only, prefetch=0)
    with pytest.raises(zpar.PipelineAccountingError):
        bad.run()
    bad_telemetry = bad.telemetry()
    assert bad_telemetry["units_consumed"] == 0
    assert bad_telemetry["units_released"] == 1
    assert bad_telemetry["ledger"]["balanced"] is True
    assert released_bad == ["r55f_scale"]

    # Unified contract: prefetch=0 stages and consumes the same unit.
    good, released_good = _real_pipeline([payload], budget=contract, prefetch=0)
    assert good.run() == ["r55f_scale"]
    good_telemetry = good.telemetry()
    assert good_telemetry["units_consumed"] == 1
    assert good_telemetry["units_released"] == 1
    assert good_telemetry["ledger"]["balanced"] is True
    assert released_good == ["r55f_scale"]


# ---------------------------------------------------------------------------
# I. Rework-3: stack host baseline counted ONCE (planning/apply/runtime).
#
# Planning freezes decision.gpu_host_bound_by_tile[tile] as the ADDED bound
# (estimate - PROCESS_BASELINE); shared_bytes already carries the baseline once.
# The apply envelope and the RUN aggregate must both use that authoritative
# value, never the TOTAL estimate (which added +356,515,840 B on the witness).
# The cache phase contract is intentionally left unchanged (active cell process
# baseline + shared parent process).
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _ApplyBundle:
    envelope: object
    shared_bytes: int
    cache_worker_footprint_bytes: int
    cache_workers_per_cell: int
    cell_specs: list
    cell_tasks: list
    cpu_fallback_info: dict
    attempts: list
    gauge_workers: int
    cache_build_workers: int
    cpu_streaming: dict = dataclasses.field(default_factory=dict)
    cpu_streaming_tile: int = 0
    gpu_tile_size: int | None = None
    gpu_streaming_tile: int | None = None
    gpu_streaming_host_peak_bytes: int | None = None
    gpu_used: bool = False
    gpu_pre_execution_fallback: dict | None = None
    plan: dict = dataclasses.field(default_factory=dict)
    chosen_mode: str = "stream"
    cells_in_flight: int = 1
    per_cell_footprint: int = 0


def _apply_fixture():
    raw = 13_102_166_016
    factor_num, factor_den = rcontract.ADMIT_FACTOR_NUM, rcontract.ADMIT_FACTOR_DEN
    envelope = SimpleNamespace(
        raw_available_bytes=int(raw),
        factor_num=int(factor_num),
        factor_den=int(factor_den),
        budget_bytes=(int(factor_num) * int(raw)) // int(factor_den),
        layout=(2, 2),
        refinement_rank=0,
        pinned=False,
    )
    n_frames, patch_hw = 66, (1767, 1330)
    thresholds = {}
    for tile in (448, 384):
        total = int(zz._estimate_streaming_bytes(n_frames, patch_hw, tile))
        thresholds[int(tile)] = max(
            1, total - int(rcontract.PROCESS_BASELINE_BYTES)
        )
    decision = SimpleNamespace(
        gpu_context=SimpleNamespace(tile_ladder=(448, 384)),
        gpu_host_bound_by_tile=tuple(
            (int(t), int(thresholds[int(t)])) for t in (448, 384)
        ),
        gauge_workers=1,
        cache_workers=14,
        gpu_max_contributors=n_frames,
        gpu_worst_patch_hw=patch_hw,
        device_peak_bytes_by_tile={
            448: 120_000_000,
            384: 100_000_000,
        },
        policy=zgr.VramPolicy(),
        qualification={"percent": 70, "budget_bytes": 5_000_000_000},
        gpu_tile_ladder=(448, 384),
        facts={
            "concurrent_producer_bytes": 123456,
            "per_unit_prefetch_host": 7890,
        },
    )
    cell = SimpleNamespace(cell_id="c0", row=0, col=0)
    patch = SimpleNamespace(patch_shape_hw=patch_hw)
    mem = SimpleNamespace(patch_ids=[f"f{i}" for i in range(n_frames)])
    bundle = _ApplyBundle(
        envelope=envelope,
        shared_bytes=int(rcontract.PROCESS_BASELINE_BYTES),
        cache_worker_footprint_bytes=(
            int(rcontract.PROCESS_BASELINE_BYTES) + 64 * MIB
        ),
        cache_workers_per_cell=14,
        cell_specs=[(cell, patch, mem, "/tmp/c0", None, 0, 0)],
        cell_tasks=[(cell, patch, mem, "/tmp/c0", None, 0)],
        cpu_fallback_info={},
        attempts=[],
        gauge_workers=1,
        cache_build_workers=14,
    )
    return bundle, decision


def _apply_mapping(tile: int) -> dict:
    return {
        "backend": "gpu",
        "phase": "cell_stack_stream",
        "gpu_streaming_tile": int(tile),
        "workers_per_unit": 14,
        "units_in_flight": 1,
        "prefetch_units": 1,
    }


def test_e11_apply_uses_the_added_stack_bound_and_counts_baseline_once():
    bundle, decision = _apply_fixture()
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=_apply_mapping(448), decision=decision
    )
    added = dict(decision.gpu_host_bound_by_tile)[448]
    total = int(zz._estimate_streaming_bytes(66, (1767, 1330), 448))
    assert applied.envelope.stack_total_bytes == bundle.shared_bytes + added
    assert applied.plan["stack_total_bytes"] == bundle.shared_bytes + added
    assert applied.plan["candidates"][0]["max_bound_bytes"] == added
    assert applied.plan["per_cell_stack_added_bytes"] == added
    assert applied.plan["per_cell_stack_total_bytes"] == total
    # Bundle-level total remains the diagnostic/execution value; the envelope
    # and plan max-bound use the authoritative ADDED bound.
    assert applied.per_cell_footprint == total
    assert applied.gpu_streaming_host_peak_bytes == total
    assert applied.envelope.stack_total_bytes == bundle.shared_bytes + added
    # Old formula used the TOTAL estimate on top of shared -> +one baseline.
    old_stack = rcontract.stack_phase_total_bytes(bundle.shared_bytes, 1, total)
    assert old_stack - applied.envelope.stack_total_bytes == (
        rcontract.PROCESS_BASELINE_BYTES
    )


@pytest.mark.parametrize("tile", [448, 384])
def test_e11_runtime_aggregate_equals_apply_and_does_not_double_count(tile):
    bundle, _decision_fixture = _apply_fixture()
    # One shared decision object (avoid rebuilding maps with a different object).
    decision = _decision_fixture
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=_apply_mapping(tile), decision=decision
    )
    runtime = zz._r55_gpu_host_aggregate_for_tile(
        decision, bundle, tile, prefetch_units=1
    )
    assert runtime == applied.envelope.aggregate_bytes
    added = dict(decision.gpu_host_bound_by_tile)[tile]
    assert applied.envelope.stack_total_bytes == bundle.shared_bytes + added

    total = int(zz._estimate_streaming_bytes(66, (1767, 1330), tile))
    old_stack = rcontract.stack_phase_total_bytes(bundle.shared_bytes, 1, total)
    # UNIFIED PROJECTION: the mapping's p == 1 charges exactly ONE producer
    # owner plus the resident prefetch credit on top of the serial stack.
    prefetch = int(_apply_mapping(tile)["prefetch_units"])
    assert prefetch == 1
    expected = (
        applied.envelope.stack_total_bytes
        + int(decision.facts["concurrent_producer_bytes"])
        + prefetch * int(decision.facts["per_unit_prefetch_host"])
    )
    assert runtime == expected
    assert applied.envelope.aggregate_bytes == expected
    assert applied.envelope.aggregate_rule == (
        rcontract.AGGREGATE_RULE_PROJECTED_HOST_PEAK
    )
    old_max = max(
        applied.envelope.stack_total_bytes, applied.envelope.cache_total_bytes
    )
    # The historical max() model UNDER-counts the coexisting producer owner:
    # under the unified projection the excess over max() is exactly the owner
    # plus the resident credit, minus any cache phase that exceeded the stack.
    assert runtime - old_max == (
        int(decision.facts["concurrent_producer_bytes"])
        + prefetch * int(decision.facts["per_unit_prefetch_host"])
        + applied.envelope.stack_total_bytes
        - old_max
    )
    assert runtime > old_max
    assert old_stack >= applied.envelope.stack_total_bytes
    check = zz._r55_runtime_tile_check(
        decision,
        tile=tile,
        free_bytes=10 * GIB,
        host_budget_bytes=runtime,
        bundle=bundle,
    )
    assert check["host_ok"] is True
    assert check["eligible"] is True


def test_e11_cache_phase_contract_is_unchanged_by_the_stack_fix():
    bundle, decision = _apply_fixture()
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=_apply_mapping(448), decision=decision
    )
    cache_per_cell = rcontract.cache_per_cell_bytes(
        rcontract.PROCESS_BASELINE_BYTES,
        bundle.cache_workers_per_cell,
        bundle.cache_worker_footprint_bytes,
    )
    expected_cache_total = rcontract.cache_phase_total_bytes(
        bundle.shared_bytes, 1, cache_per_cell
    )
    # Cache deliberately counts the active cell process baseline AND shared
    # (parent) process; only the stack double-count was corrected.
    assert applied.envelope.cache_total_bytes == expected_cache_total


def test_e11_added_host_bound_map_is_fail_closed():
    _bundle, _decision = _apply_fixture()
    duplicate = SimpleNamespace(
        gpu_host_bound_by_tile=((448, 1), (448, 2))
    )
    with pytest.raises(ValueError, match="duplicate"):
        zz._r55_gpu_added_host_bound(duplicate, 448)
    missing = SimpleNamespace(gpu_host_bound_by_tile=((448, 1),))
    with pytest.raises(ValueError, match="no authoritative bound"):
        zz._r55_gpu_added_host_bound(missing, 384)
    negative = SimpleNamespace(gpu_host_bound_by_tile=((448, -1),))
    with pytest.raises(ValueError, match="strictly positive"):
        zz._r55_gpu_added_host_bound(negative, 448)
    absent = SimpleNamespace()
    with pytest.raises(ValueError, match="missing"):
        zz._r55_gpu_added_host_bound(absent, 448)
