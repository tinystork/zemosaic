"""ZM-ZEGRID-RESILIENT-RUN-RESUME M1 — real-orchestration conservation test.

This is the M1 acceptance witness (not the M0 historical baseline). It drives
the REAL product path (``run_zegrid_mode`` -> ``_run_single`` -> bounded GPU
pipeline) with the real small FITS corpus and real canonical cell computations,
using only the existing CONTROL-ONLY GPU substitution (the device executor is
the CPU canonical executor; no CuPy). It injects a synthetic available-memory
drop at the REAL product memory boundary after at least two completed cells.

M1 expectations asserted here (M0 expected the opposite: discard + CPU rerun):
* the two accepted cell IDs and their COMPLETE payload fingerprints survive the
  refusal and pipeline/cache cleanup;
* the fingerprints include the per-cell pre-offset/pre-feather patch and the
  photometric reference pin, not only the core hash;
* the coordinator can enumerate the pending sparse cell IDs without recomputing
  or double counting the accepted cells;
* no FITS, coverage or manifest is published;
* the whole-batch CPU rerun is refused (no silent backend mixing);
* explicit disposal releases the retained payload ownership exactly once.

No GPU or Windows qualification is claimed. That remains the separate Windows
harness (NOT_RUN here).
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402


def _load_m0_helpers():
    """Reuse the proven corpus/seam helpers of the M0 harness (no test run)."""
    path = REPO_ROOT / "tests" / "test_zegrid_resume_m0_baseline_witness.py"
    spec = importlib.util.spec_from_file_location("_m0_witness_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


M0 = _load_m0_helpers()

#: Below this live available RAM the 2x2 corpus is not physically admissible.
#: Conservation cases must SKIP explicitly (never run on fabricated capacity).
MIN_LIVE_BYTES = 1200 * 2 ** 20


def _live_available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _require_live_ram() -> int:
    live = _live_available_bytes()
    if live < MIN_LIVE_BYTES:
        pytest.skip(
            "live available RAM is below the 2x2 corpus floor "
            f"({live} < {MIN_LIVE_BYTES}); conservation NOT_RUN on this host"
        )
    return live


class _ClampedObservation:
    """Fresh-min memory observation: no capacity above the start/requested cap.

    Each call re-reads the live value. A bad reading fails closed. The value is
    `min(fresh, initial_cap, requested trajectory)`; pressure injection can only
    lower it, never raise it.
    """

    def __init__(self, *, initial_cap, requested_fn, fresh_fn):
        self.initial_cap = int(initial_cap)
        self.requested_fn = requested_fn
        self.fresh_fn = fresh_fn
        self.values = []

    def __call__(self) -> int:
        fresh = self.fresh_fn()
        if (
            isinstance(fresh, bool)
            or not isinstance(fresh, int)
            or int(fresh) <= 0
        ):
            raise RuntimeError(
                f"fresh available RAM unavailable: {fresh!r}"
            )
        value = int(min(int(fresh), self.initial_cap, int(self.requested_fn())))
        self.values.append(int(value))
        return int(value)


def _require_expected(run: dict, expected, context: str):
    """Return the expected terminal failure or fail/skip for the right reason.

    Only EVIDENCED pre-case resource/setup infeasibility (injection never armed)
    may become NOT_RUN. Any unexpected exception, or a success after the
    injection was verifiably armed, is a regression and must FAIL.
    """
    exc = run.get("exception")
    armed = bool(run.get("armed"))
    evidence = (
        f"accepted={run.get('accepted_cells')}, hits={run.get('hits')}, "
        f"observation_samples={len(run.get('observation_values') or [])}"
    )
    if exc is None:
        if not armed:
            pytest.skip(
                "pre-case resource/setup unavailable; injection was not "
                f"armed and no case was exercised ({evidence})"
            )
        pytest.fail(
            f"injection was armed but no {context} occurred; "
            f"the run completed successfully ({evidence})"
        )
    if not isinstance(exc, expected):
        pre_case_types = (
            zz.zal.LayoutInfeasible,
            zz.zpar.ModeInfeasible,
            rcontract.RamContractError,
            rcontract.AdmissionExhausted,
        )
        if not armed and isinstance(exc, pre_case_types):
            pytest.skip(
                "pre-case resource/setup unavailable before injection "
                f"({type(exc).__name__}: {exc}); {evidence}"
            )
        pytest.fail(
            f"unexpected terminal {type(exc).__name__}: {exc!r} while "
            f"exercising {context} ({evidence})"
        )
    return exc


def _is_cell_batch_worker(worker) -> bool:
    return getattr(worker, "__name__", None) == "_stack_cell"


def _advance_generation_on_cell_batch(
    *, worker, counter, trigger_at, registry_instances, reason="mid_wave_test"
) -> bool:
    """Count CELL-BATCH pmap calls only; advance on the intended one.

    The product also uses ``zpar.pmap`` for cache-frame work. Counting every
    call made the mid-wave injection land on a non-wave/pre-registry call.
    """
    if not _is_cell_batch_worker(worker):
        return False
    counter["n"] += 1
    if not trigger_at or counter["n"] != int(trigger_at):
        return False
    if not registry_instances:
        raise AssertionError(
            "mid-wave injection reached the intended cell-batch pmap but the "
            "run registry does not exist yet"
        )
    for reg in registry_instances:
        reg.advance_generation(reason=reason)
    return True


def _run_m1(
    *,
    base: Path,
    inp: Path,
    arm_after: int = 2,
    stop_after: int | None = None,
    gpu_oom_after: int | None = None,
    fail_accept_at: int | None = None,
    advance_after_pmap_call: int = 0,
    force_wave_size: int | None = None,
    use_gpu: bool = True,
    live_available: int | None = None,
) -> dict:
    sequence: list[dict] = []
    weakrefs: list[weakref.ReferenceType] = []
    memory = {"low": False}
    hits = {
        "memory_scheduled": False,
        "memory_seam": False,
        "stop": False,
        "gpu_oom": False,
        "accept_fail": False,
        "generation": False,
    }
    captured_registries: list = []
    registry_instances: list = []
    accept_count = {"n": 0}
    accept_attempts = {"n": 0}
    gpu_calls = {"n": 0}
    pmap_calls = {"n": 0}
    cell_batch_pmap_calls = {"n": 0}

    live = int(live_available or _live_available_bytes())
    env_live = os.environ.get("ZEMOSAIC_M1_LIVE_AVAILABLE_BYTES")
    if env_live:
        live = min(live, int(env_live))
    initial_cap = max(0, int(live))
    effective_high = min(int(M0.RAW_HIGH), initial_cap)
    effective_low = min(int(M0.RAW_LOW), effective_high)
    def _requested_cap() -> int:
        if memory["low"]:
            hits["memory_seam"] = True
            return effective_low
        return effective_high

    observation = _ClampedObservation(
        initial_cap=initial_cap,
        requested_fn=_requested_cap,
        fresh_fn=_live_available_bytes,
    )

    OOM = type(
        "OutOfMemoryError",
        (RuntimeError,),
        {"__module__": "cupy.cuda.memory"},
    )

    with pytest.MonkeyPatch.context() as mp:
        M0._fake_gpu_seams(mp, memory)
        # REWORK-3: every actual observation seam re-reads live RAM and takes
        # min(fresh, start/env cap, requested trajectory). This is a true
        # no-inflation clamp, not a pinned start snapshot.
        mp.setattr(zz, "available_memory_bytes", observation)
        real_init = zrr.RunResultRegistry.__init__

        def _init(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            registry_instances.append(self)

        mp.setattr(zrr.RunResultRegistry, "__init__", _init)
        real_accept = zrr.RunResultRegistry.accept

        def _accept(self, result, **kwargs):
            accept_attempts["n"] += 1
            if (
                fail_accept_at is not None
                and accept_attempts["n"] == int(fail_accept_at)
            ):
                hits["accept_fail"] = True
                raise zrr.AcceptanceError(
                    "injected internal acceptance/schema failure"
                )
            receipt = real_accept(self, result, **kwargs)
            if receipt.accepted and not receipt.idempotent:
                accept_count["n"] += 1
                captured_registries.append(self)
                if (
                    stop_after is not None
                    and accept_count["n"] >= int(stop_after)
                ):
                    hits["stop"] = True
                    self.request_stop("integration cooperative stop")
            return receipt

        mp.setattr(zrr.RunResultRegistry, "accept", _accept)

        # REWORK-2 mid-wave generation test seam: advance the REAL registry
        # after a chosen pmap call, while the wave results are already computed
        # but before acceptance. A capture-after-dispatch bug would stamp those
        # old results with the new generation.
        real_pmap = zz.zpar.pmap

        def _pmap(*args, **kwargs):
            worker = args[0] if args else kwargs.get("worker")
            pmap_calls["n"] += 1
            out = real_pmap(*args, **kwargs)
            if advance_after_pmap_call and _advance_generation_on_cell_batch(
                worker=worker,
                counter=cell_batch_pmap_calls,
                trigger_at=int(advance_after_pmap_call),
                registry_instances=registry_instances,
                reason="mid_wave_test",
            ):
                hits["generation"] = True
            return out

        mp.setattr(zz.zpar, "pmap", _pmap)

        if force_wave_size is not None:
            # TEST-ONLY deterministic wave bound: smaller waves, same product
            # scheduler/workers/budget. This makes the mid-wave transition
            # executable on a high-RAM host instead of collapsing to one wave.
            _force_wave = int(force_wave_size)

            def _bounded_next_wave(offset, total, units_in_flight):
                return (
                    int(offset),
                    min(int(total), int(offset) + max(1, _force_wave)),
                )

            mp.setattr(zz.zrt, "next_wave_bound", _bounded_next_wave)

        real_stream = zz.run_canonical_stack_streaming

        def _control_stream(
            provider, request, tile_size=zz.STREAM_TILE_SIZE, fixed=None,
            rejection_sink=None,
        ):
            import dataclasses

            return real_stream(
                provider,
                dataclasses.replace(request, backend="cpu"),
                tile_size=tile_size,
                fixed=fixed,
                rejection_sink=rejection_sink,
            )

        mp.setattr(zz, "run_canonical_stack_streaming", _control_stream)
        mp.setattr(zz, "_R55E2_MEASURE_PREFETCH_CAP", 0)

        real_consume = zz._consume_stack_cell

        def _consume(prepared, **kw):
            if gpu_oom_after is not None:
                gpu_calls["n"] += 1
                if gpu_calls["n"] > int(gpu_oom_after):
                    hits["gpu_oom"] = True
                    raise OOM("simulated CuPy OOM after accepted cells")
            out = real_consume(prepared, **kw)
            if isinstance(out, dict) and out.get("status") == "complete":
                fingerprint = zrr.fingerprint_result(out)
                accounting = zrr._buffer_accounting(out)
                sequence.append(
                    {
                        "cell_id": str(out["cell_id"]),
                        "status": "complete",
                        "backend": str(out.get("backend_used")),
                        "fingerprint": fingerprint,
                        "accounting": accounting,
                        "core_sha256": fingerprint["core_hashes"].get(
                            "science_core"
                        ),
                        "patch_sha256": fingerprint["patch_hashes"],
                        "reference_pin": dict(fingerprint["reference_pin"]),
                        "n_frames": int(out.get("n_frames", 0)),
                    }
                )
                weakrefs.append(weakref.ref(out["core"]["science_core"]))
                if len(sequence) >= int(arm_after):
                    memory["low"] = True
                    hits["memory_scheduled"] = True
            return out

        mp.setattr(zz, "_consume_stack_cell", _consume)

        out_dir = base / "out"
        exc: BaseException | None = None
        try:
            zz.run_zegrid_mode(
                str(inp),
                str(out_dir),
                zconfig=SimpleNamespace(
                    zegrid_scheduler_mode="active", zegrid_layout=M0.LAYOUT
                ),
                zegrid_halo_px=M0.HALO,
                use_gpu=bool(use_gpu),
                cell_gauge_mode="per_cell",
            )
        except BaseException as err:  # noqa: BLE001 - refusal IS the witness
            exc = err

    return {
        "sequence": sequence,
        "weakrefs": weakrefs,
        "memory": memory,
        "hits": dict(hits),
        "armed": bool(
            any(
                hits[key]
                for key in (
                    "memory_seam",
                    "stop",
                    "gpu_oom",
                    "accept_fail",
                    "generation",
                )
            )
        ),
        "accepted_cells": int(accept_count["n"]),
        "observation_values": list(observation.values),
        "exception": exc,
        "out_dir": out_dir,
        "pmap_calls": int(pmap_calls["n"]),
        "cell_batch_pmap_calls": int(cell_batch_pmap_calls["n"]),
        "effective_high": int(effective_high),
        "effective_low": int(effective_low),
    }


def test_m1_refusal_preserves_accepted_payloads_and_sparse_pending_set(
    tmp_path, monkeypatch,
):
    _require_live_ram()
    inp, names, hashes = M0._write_corpus(tmp_path)
    run = _run_m1(base=tmp_path / "m1", inp=inp, arm_after=2)

    # --- 1. The refusal is the M1 conservation refusal, not CPU rerun. ------
    exc = run["exception"]
    _require_expected(
        run, zz._R55RunConservedResults, "the K>0 GPU refusal"
    )
    assert not isinstance(exc, rcontract.AdmissionExhausted)
    assert exc.reason_code == "R55_RUN_RESULTS_CONSERVED"
    assert "CPU reuse refused" in str(exc)
    registry = exc.result_registry
    assert registry is not None

    # --- 2. At least two real cells were computed and accepted. -------------
    assert len(run["sequence"]) >= 2
    assert registry.counts()["accepted"] == len(run["sequence"])
    assert registry.counts()["complete"] == len(run["sequence"])
    assert registry.counts()["empty_validated"] == 0
    accepted_ids = registry.accepted_cell_ids()
    assert accepted_ids == [c["cell_id"] for c in run["sequence"]]
    assert registry.pending_cell_ids()  # sparse expected set still has work

    # --- 3. Complete payload fingerprints survive refusal/cleanup. ----------
    for expected in run["sequence"]:
        meta = registry.metadata(expected["cell_id"])
        assert meta is not None
        assert meta["status"] == zrr.STATUS_COMPLETE
        assert meta["backend"] in ("gpu", "cpu")
        assert meta["backend"] == expected["backend"]
        assert meta["payload_sha256"] == expected["fingerprint"]["payload_sha256"]
        assert meta["core_sha256"] == expected["fingerprint"]["core_hashes"]
        # per_cell patch with pre-offset/pre-feather halo is preserved whole.
        assert meta["patch_sha256"] == expected["patch_sha256"]
        assert meta["patch_sha256"] is not None
        assert meta["reference_pin"] == expected["reference_pin"]
        assert meta["logical_bytes"] > 0
        assert meta["retained_bytes"] > 0
        # A view's logical plane bytes may exceed the unique base allocation;
        # the two counters are deliberately separate and must not be conflated.
        # attempt + generation identity is recorded on every accepted result.
        assert meta["attempt_id"].startswith("gpu-pipeline:0:")
        assert meta["plan_generation"] == 0
        assert len(meta["payload_sha256"]) == 64

    # --- 4. Retained bytes are honest and deduplicated across aliases. ------
    expected_buffers: dict[int, int] = {}
    for capture in run["sequence"]:
        for key, entry in capture["accounting"]["buffers"].items():
            expected_buffers[key] = int(entry["bytes"])
    assert registry.retained_bytes == sum(expected_buffers.values())
    assert registry.counts()["retained_buffers"] == len(expected_buffers)

    # --- 5. Coordinator enumeration does not double count or recompute. -----
    counts = registry.counts()
    assert counts["expected"] == counts["accepted"] + counts["pending"]
    pending = registry.pending_cell_ids()
    assert set(pending).isdisjoint(set(accepted_ids))
    # Every cell was computed exactly once; the refused unit never computed.
    assert len(run["sequence"]) == len(set(accepted_ids))
    assert run["memory"]["low"] is True

    # --- 6. Nothing is published; cleanup left no staged payload. -----------
    out_dir = run["out_dir"]
    assert not (out_dir / "mosaic_grid.fits").exists()
    assert not (out_dir / "mosaic_grid_coverage.fits").exists()
    assert not (out_dir / "zegrid_manifest.json").exists()
    cache_root = out_dir / "__zegrid_cache__"
    if cache_root.exists():
        assert [p.name for p in cache_root.rglob("*")] == []
    # The GPU failure diagnostic still exists and records retention, not discard.
    import json

    failure = json.loads(
        (out_dir / "zegrid_gpu_failure.json").read_text(encoding="utf-8")
    )
    completed = [
        u for u in failure["tile_state"]["units"] if u["status"] == "completed"
    ]
    assert len(completed) == len(run["sequence"])
    assert all(u["retained"] is True for u in completed)
    assert all(u["discarded"] is False for u in completed)
    ledger = failure["pipeline"]["ledger"]
    assert ledger["balanced"] is True
    assert ledger["host_leases"] == 0
    assert ledger["device_leases"] == 0
    assert ledger["disk_leases"] == 0
    assert failure["pipeline"]["cleanup_failures"] == []

    # --- 7. Explicit disposal releases ownership exactly once. --------------
    first_ref = run["weakrefs"][0]
    assert first_ref() is not None
    summary = registry.dispose()
    assert summary["released_cells"] == len(run["sequence"])
    assert summary["released_retained_bytes"] == sum(
        expected_buffers.values()
    )
    assert summary["released_logical_bytes"] == sum(
        c["accounting"]["logical_bytes"] for c in run["sequence"]
    )
    assert summary["already_disposed"] is False
    gc.collect()
    assert first_ref() is None
    again = registry.dispose()
    assert again["already_disposed"] is True
    assert again["released_cells"] == 0
    assert registry.counts()["accepted"] == 0


def test_m1_cooperative_stop_ack_preserves_accepted_cells(tmp_path):
    """A requested stop is honoured at the next unit boundary, acknowledged in
    memory, and never erases the accepted results. Durable STOPPED_RESUMABLE is
    explicitly NOT claimed by M1."""
    _require_live_ram()
    inp, names, hashes = M0._write_corpus(tmp_path)
    run = _run_m1(
        base=tmp_path / "m1_stop",
        inp=inp,
        arm_after=99,
        stop_after=1,
        force_wave_size=2,
    )

    exc = run["exception"]
    _require_expected(
        run, zz._R55RunStopRequested, "the cooperative stop boundary"
    )
    registry = exc.result_registry
    assert registry is not None
    assert registry.stop_requested is True
    ack = registry.stop.snapshot()
    assert ack["acknowledged"] is True
    assert ack["ack_state"] == "STOPPED_IN_MEMORY"
    # At least one accepted result survives the stop; pending cells remain a
    # sparse set, and no publication happened.
    assert registry.counts()["accepted"] >= 1
    assert registry.pending_cell_ids()
    assert not (run["out_dir"] / "mosaic_grid.fits").exists()
    assert not (run["out_dir"] / "zegrid_manifest.json").exists()
    assert exc.record["durable_stopped_resumable"] is False
    accepted_before = registry.counts()["accepted"]
    summary = registry.dispose()
    assert summary["released_cells"] == accepted_before
    assert registry.counts()["accepted"] == 0


def test_m1_gpu_oom_after_accepted_cells_is_conserved(tmp_path):
    """K>0 CuPy OOM must conserve, classify as resource shortage, and never
    silently rerun the accepted cells on CPU."""
    _require_live_ram()
    inp, names, hashes = M0._write_corpus(tmp_path)
    run = _run_m1(
        base=tmp_path / "m1_oom",
        inp=inp,
        arm_after=99,
        gpu_oom_after=1,
    )
    exc = run["exception"]
    _require_expected(
        run, zz._R55RunConservedResults, "the K>0 CuPy OOM"
    )
    assert exc.category == "resource_shortage"
    assert "OutOfMemory" in str(exc) or "OutOfMemory" in str(
        exc.record.get("gpu_fallback", {})
    )
    registry = exc.result_registry
    assert registry is not None
    assert registry.counts()["accepted"] >= 1
    assert registry.pending_cell_ids()
    assert not (run["out_dir"] / "mosaic_grid.fits").exists()
    assert not (run["out_dir"] / "zegrid_manifest.json").exists()
    accepted_before = registry.counts()["accepted"]
    summary = registry.dispose()
    assert summary["released_cells"] == accepted_before


def test_m1_gpu_internal_acceptance_failure_is_conserved_and_classified(
    tmp_path,
):
    """A K>0 internal acceptance/schema failure is NOT labelled as a resource
    shortage, keeps the accepted owner handle, and publishes nothing."""
    _require_live_ram()
    inp, names, hashes = M0._write_corpus(tmp_path)
    run = _run_m1(
        base=tmp_path / "m1_accept_fail",
        inp=inp,
        arm_after=99,
        fail_accept_at=2,
    )
    exc = run["exception"]
    _require_expected(
        run, zz._R55RunConservedResults, "the K>0 internal acceptance failure"
    )
    assert exc.category == "acceptance_failure"
    assert exc.reason_code == "R55_RUN_RESULTS_CONSERVED"
    registry = exc.result_registry
    assert registry is not None
    assert registry.counts()["accepted"] == 1
    assert registry.pending_cell_ids()
    assert not (run["out_dir"] / "mosaic_grid.fits").exists()
    assert not (run["out_dir"] / "zegrid_manifest.json").exists()
    summary = registry.dispose()
    assert summary["released_cells"] == 1


def test_m1_cpu_acceptance_failure_preserves_registry_handle(tmp_path):
    """The definite missing-owner path: CPU acceptance failure after K accepted
    cells must expose the registry handle, not only a snapshot/traceback."""
    _require_live_ram()
    inp, names, hashes = M0._write_corpus(tmp_path)
    run = _run_m1(
        base=tmp_path / "m1_cpu_accept_fail",
        inp=inp,
        arm_after=99,
        fail_accept_at=2,
        use_gpu=False,
    )
    exc = run["exception"]
    assert isinstance(exc, zz._R55RunRefused), repr(exc)
    assert exc.reason_code == "R55_RUN_RESULT_ACCEPTANCE"
    assert getattr(exc, "result_registry", None) is not None
    registry = exc.result_registry
    assert registry.counts()["accepted"] == 1
    assert registry.pending_cell_ids()
    assert exc.record["publication"] == {
        "fits": False, "manifest": False, "assembly": False
    }
    assert not (run["out_dir"] / "mosaic_grid.fits").exists()
    summary = registry.dispose()
    assert summary["released_cells"] == 1
    assert registry.counts()["accepted"] == 0


def test_m1_mid_wave_generation_transition_rejects_old_results(tmp_path):
    """A generation change while a CPU wave is in flight must not stamp the
    old results with the new generation, and must keep the previously accepted
    payloads. This is the dispatch-capture regression from REWORK-2."""
    _require_live_ram()
    inp, names, hashes = M0._write_corpus(tmp_path)
    run = _run_m1(
        base=tmp_path / "m1_mid_wave_gen",
        inp=inp,
        arm_after=99,
        advance_after_pmap_call=2,
        force_wave_size=2,
        use_gpu=False,
    )
    # TEST-ONLY smaller waves make this deterministic on an adequate-memory
    # host; product workers/budget/scheduling are unchanged.
    if run["cell_batch_pmap_calls"] < 2:
        pytest.fail(
            "mid-wave injection did not arm: "
            f"cell_batch_pmaps={run['cell_batch_pmap_calls']}, "
            f"all_pmaps={run['pmap_calls']}, force_wave_size=2, "
            f"accepted={run.get('accepted_cells')}, hits={run.get('hits')}"
        )
    exc = run["exception"]
    _require_expected(
        run, zz._R55RunRefused, "the mid-wave generation transition"
    )
    assert exc.reason_code == "R55_RUN_RESULT_ACCEPTANCE"
    assert "stale" in str(exc).lower() or "generation" in str(exc).lower()
    registry = getattr(exc, "result_registry", None)
    assert registry is not None
    # The first wave was accepted before the transition and is preserved; the
    # in-flight second-wave result was rejected instead of stamped gen 1.
    assert registry.counts()["accepted"] >= 1
    assert registry.generation == 1
    first_id = registry.accepted_cell_ids()[0]
    assert registry.metadata(first_id) is not None
    assert registry.pending_cell_ids()
    assert not (run["out_dir"] / "mosaic_grid.fits").exists()
    assert not (run["out_dir"] / "zegrid_manifest.json").exists()
    accepted = registry.counts()["accepted"]
    summary = registry.dispose()
    assert summary["released_cells"] == accepted


def test_clamped_observation_fresh_min_and_fail_closed():
    gib = 2 ** 30
    mib = 2 ** 20
    fresh = iter([2 * gib, 1 * gib, 500 * mib])
    obs = _ClampedObservation(
        initial_cap=1500 * mib,
        requested_fn=lambda: 1600 * mib,
        fresh_fn=lambda: next(fresh),
    )
    assert obs() == 1500 * mib  # fresh above the start cap is clamped
    assert obs() == 1 * gib  # fresh below the cap wins
    assert obs() == 500 * mib  # fresh below the requested trajectory wins
    for bad in (None, 0, -1, True, "123"):
        broken = _ClampedObservation(
            initial_cap=1500 * mib,
            requested_fn=lambda: 700 * mib,
            fresh_fn=lambda bad=bad: bad,
        )
        with pytest.raises(RuntimeError):
            broken()


def test_require_expected_fails_on_unexpected_or_missing_after_arm():
    base_run = {
        "accepted_cells": 0,
        "hits": {"memory": True},
        "observation_values": [1],
    }
    # Unexpected RuntimeError is a regression, never a skip, whether or not the
    # injection fired.
    for armed in (False, True):
        with pytest.raises(pytest.fail.Exception):
            _require_expected(
                {**base_run, "exception": RuntimeError("boom"), "armed": armed},
                zz._R55RunConservedResults,
                "unexpected-error case",
            )
    # A success AFTER a verified armed injection is a missing-refusal failure.
    with pytest.raises(pytest.fail.Exception):
        _require_expected(
            {**base_run, "exception": None, "armed": True},
            zz._R55RunConservedResults,
            "armed-missing case",
        )
    # No injection fired and no terminal: evidenced pre-case NOT_RUN.
    with pytest.raises(pytest.skip.Exception):
        _require_expected(
            {
                "exception": None,
                "armed": False,
                "accepted_cells": 0,
                "hits": {},
                "observation_values": [],
            },
            zz._R55RunConservedResults,
            "pre-case case",
        )


def test_require_expected_accepts_only_the_expected_terminal():
    expected = zz._R55RunConservedResults(
        "R55_RUN_RESULTS_CONSERVED",
        "test",
        record={},
        result_registry=None,
    )
    assert _require_expected(
        {"exception": expected, "armed": True, "accepted_cells": 1},
        zz._R55RunConservedResults,
        "matched case",
    ) is expected


def test_mid_wave_injection_counts_only_cell_batch_pmaps():
    """The injection must ignore cache-frame pmap calls.

    A fake interleaved sequence proves the trigger fires on the 2nd
    ``_stack_cell`` batch, not on unrelated ``_cell_cache_frame_worker`` calls,
    and fails closed if no registry exists when the trigger is reached.
    """

    class _FakeRegistry:
        def __init__(self):
            self.advanced = 0

        def advance_generation(self, reason):
            self.advanced += 1

    def _stack_cell():
        return None

    def _cell_cache_frame_worker():
        return None

    counter = {"n": 0}
    registry = _FakeRegistry()
    fired = []
    for worker in (
        _cell_cache_frame_worker,
        _stack_cell,
        _cell_cache_frame_worker,
        _cell_cache_frame_worker,
        _stack_cell,
        _stack_cell,
    ):
        if _advance_generation_on_cell_batch(
            worker=worker,
            counter=counter,
            trigger_at=2,
            registry_instances=[registry],
            reason="unit",
        ):
            fired.append(counter["n"])

    assert counter["n"] == 3  # three cell batches, three cache-frame calls ignored
    assert fired == [2]  # fired exactly on the second cell batch
    assert registry.advanced == 1
    with pytest.raises(AssertionError):
        _advance_generation_on_cell_batch(
            worker=_stack_cell,
            counter={"n": 0},
            trigger_at=1,
            registry_instances=[],
            reason="unit",
        )
