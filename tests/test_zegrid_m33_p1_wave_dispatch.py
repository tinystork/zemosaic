"""ZM-M33 P0P1 (P1) — PRODUCT-level dispatch witnesses for the RUN batch shrink.

These tests drive the REAL entry point (``run_zegrid_mode``) on a synthetic tiny
corpus (never M33) and observe the REAL dispatch:

* ``_stack_cell`` receives every task the wave loop dispatches (its profile slot,
  its mode slot and the external contract in slot 7);
* ``zext.run_cell_streaming_external`` receives the ``column_batch`` the external
  executor actually runs.

Witnesses:

1. a governed RUN pressure event on the external class shrinks the frozen
   ``column_batch``, persists the transition BEFORE dispatch, and every
   dispatched task/execution carries the successor batch (no 7296 survives);
2. the F2 dense->bounded recovery dispatches the NEW bounded tasks (the stale
   dense task list is gone);
3. the science output is identical between the shrunk run and a reference
   external run with the default batch (batch-size invariance on real data).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import resource_envelope as renv
from zemosaic.core.zegrid import runtime_control as zrt

MIB = 2 ** 20
GIB = 2 ** 30

_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(*_a, **_k):
    """Test-only floors: the synthetic corpus is intentionally tiny."""
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)


def _write_corpus(root, *, frames=6, hw=96, halo=32, step_px=48.0):
    from astropy.io import fits
    from astropy.wcs import WCS

    rng = np.random.default_rng(20261124)
    inp = root / "in"
    inp.mkdir()
    h = w = int(hw)
    step = float(step_px)
    names = []
    for i in range(frames):
        col, row = i % 2, i // 2
        scale = 0.001
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + col * step * scale, 30.0 - row * step * scale]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        data = (
            100.0
            + 20.0 * np.exp(
                -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 20.0 ** 2)
            )
            + 2.0 * i
            + rng.normal(0.0, 0.5, (h, w))
        ).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = 10.0
        name = f"frame_{i:04d}.fits"
        hdu.writeto(inp / name, overwrite=True)
        names.append(name)
    (inp / "stack_plan.csv").write_text(
        "file_path,exposure\n" + "".join(f"{n},10.0\n" for n in names),
        encoding="utf-8",
    )
    return inp


def _config(store_dir, *, gauge_mode="per_cell"):
    return SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=(
            None if store_dir is None else str(store_dir)
        ),
        zegrid_resume_mode="never",
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode=str(gauge_mode),
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )


def _fits_array(path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data)


def _find_state_path(store_root: Path) -> Path:
    states = sorted(store_root.rglob("state.json"))
    assert states, f"no durable state under {store_root}"
    for path in reversed(states):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id"):
            return path
    raise AssertionError(f"no state with a run_id under {store_root}")


def _find_state(store_root: Path) -> dict:
    states = sorted(store_root.rglob("state.json"))
    assert states, f"no durable state under {store_root}"
    # The generation child carries the run state.
    for path in reversed(states):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id"):
            return payload
    raise AssertionError(f"no state with a run_id under {store_root}")


def _external_run(
    *, inp, out, monkeypatch, store, pressure_first_boundary: bool,
    capture: dict, storage_profile="bounded", progress_callback=None,
    pressure_boundaries: int | None = None,
    recorder_failure: bool = False,
    read_state_at_dispatch: bool = False,
    configure=None,
):
    """Run the real product; optionally inject ONE governed pressure event."""
    real_pmap_reserved = zpar.pmap_reserved
    real_pmap = zpar.pmap
    real_checkpoint = zz._r55_run_checkpoint
    from zemosaic.core.zegrid import run_resume as zres

    real_record = zres.record_column_batch_shrink
    boundaries = (
        int(pressure_boundaries) if pressure_boundaries is not None
        else (1 if pressure_first_boundary else 0)
    )
    calls: dict = {
        "external": [], "tasks": [], "checkpoints": [],
        "dispatch_states": [], "record_calls": [],
    }

    def _record_spy(run_root, *, transition, identity_sha256):
        calls["record_calls"].append(dict(transition))
        if recorder_failure:
            raise zres.ResumeStateError(
                "injected: the durable column_batch record cannot be written"
            )
        return real_record(
            run_root, transition=transition, identity_sha256=identity_sha256
        )

    def _state_at_dispatch():
        # Persist-before-dispatch, observed at the REAL dispatch seam: read the
        # durable state from disk BEFORE the executor receives any task.
        state = _find_state(Path(store))
        admission = state.get("admission") or {}
        calls["dispatch_states"].append(
            {
                "admission_batch": (
                    (admission.get("working_set") or {}).get("column_batch")
                ),
                "chain": [
                    (entry.get("from_batch"), entry.get("to_batch"))
                    for entry in (state.get("column_batch_transitions") or [])
                ],
                "plan_generation": state.get("plan_generation"),
            }
        )

    def _capture_tasks(tasks):
        for task in tasks:
            try:
                cell_id = str(task[0].cell_id)
            except (AttributeError, TypeError, IndexError):
                # Another pool call (not a cell task): never a dispatch witness.
                continue
            calls["tasks"].append(
                {
                    "cell_id": cell_id,
                    "mode": task[4],
                    "profile": (task[6] if len(task) > 6 else None),
                    "external": (
                        dict(task[7])
                        if len(task) > 7 and isinstance(task[7], dict)
                        else None
                    ),
                }
            )

    def _pmap_reserved_spy(worker, tasks, results, *args, **kwargs):
        # The parent-side dispatch seam: exactly the tuples handed to the
        # executor, never a re-derivation.
        if read_state_at_dispatch and not calls["dispatch_states"]:
            _state_at_dispatch()
        _capture_tasks(list(tasks))
        return real_pmap_reserved(worker, tasks, results, *args, **kwargs)

    def _pmap_spy(worker, tasks, *args, **kwargs):
        _capture_tasks(list(tasks))
        return real_pmap(worker, tasks, *args, **kwargs)

    def _checkpoint_spy(**kwargs):
        facts = kwargs.get("facts")
        index = len(calls["checkpoints"])
        _pre_bundle = kwargs["bundle"]
        calls.setdefault("pre_checkpoint_candidates", []).append(
            {
                "backend": _pre_bundle.envelope.backend,
                "tile_px": (
                    int(_pre_bundle.cpu_streaming_tile or 0)
                    or int(
                        (
                            getattr(_pre_bundle, "external_contract", None) or {}
                        ).get("tile_size") or 0
                    )
                    or None
                ),
                "units_in_flight": int(_pre_bundle.cells_in_flight),
                "prefetch_units": 0,
                "workers_per_unit": int(_pre_bundle.cache_workers_per_cell),
                "working_set": str(
                    getattr(_pre_bundle, "stack_working_set", "")
                ),
            }
        )
        calls.setdefault("pre_checkpoint_contracts", []).append(
            {
                "class": str(
                    (getattr(_pre_bundle, "external_contract", None) or {})
                    .get("class", "")
                ),
                "column_batch": int(
                    (getattr(_pre_bundle, "external_contract", None) or {})
                    .get("column_batch") or 0
                ),
            }
        )
        if index < boundaries:
            bundle = kwargs["bundle"]
            if str(getattr(bundle, "stack_working_set", "")) == (
                zz.WORKING_SET_EXTERNAL
            ):
                peak = int(
                    zz._r55_external_contract(bundle)["host_peak_bytes"]
                )
            else:
                # A RAM bundle: its own admitted aggregate is the term the
                # margin is applied to.
                peak = int(bundle.envelope.aggregate_bytes)
            # A governed observation whose margin-applied phase budget is just
            # below the frozen peak: a real capacity event, not a fabricated
            # decision. 1.24 * peak -> budget 1.116 * peak -> phase 0.970 * peak.
            raw = int(peak * 1.24)
            facts = dict(facts)
            facts["host"] = dict(facts["host"])
            facts["host"]["raw_available_bytes"] = int(raw)
            facts["host"]["budget_bytes"] = int(zrt.run_budget_bytes(raw))
            facts["host"]["source"] = "injected-run-boundary-pressure"
            kwargs["facts"] = facts
        out = real_checkpoint(**kwargs)
        record = out["record"]
        calls["checkpoints"].append(
            {
                "status": record.get("status"),
                "changed": bool(record.get("changed")),
                "raw": int(record["host"]["raw_available_bytes"]),
                "source": record.get("host_source"),
                "batch_shrink": record.get("batch_shrink"),
                "candidate": dict(record.get("candidate") or {}),
            }
        )
        return out

    with monkeypatch.context() as m:
        m.setattr(zz, "available_memory_bytes", lambda: 1536 * MIB)
        m.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 6 * GIB)
        m.setattr(zpar, "pmap_reserved", _pmap_reserved_spy)
        m.setattr(zpar, "pmap", _pmap_spy)
        m.setattr(zz, "_r55_run_checkpoint", _checkpoint_spy)
        m.setattr(zres, "record_column_batch_shrink", _record_spy)
        if configure is not None:
            configure(m)
        try:
            zz.run_zegrid_mode(
                str(inp),
                str(out),
                progress_callback,
                zconfig=_config(store),
                storage_profile=storage_profile,
                use_gpu=False,
                cell_gauge_mode="per_cell",
            )
        finally:
            # A refused run must still expose what it captured (and prove that
            # NOTHING was dispatched when the boundary refused).
            capture.update(calls)
    capture.update(calls)
    return calls


@pytest.mark.slow
def test_run_batch_shrink_persists_before_dispatch_and_tasks_carry_it(
    tmp_path, monkeypatch
):
    inp = _write_corpus(tmp_path)
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    captured: dict = {}
    _external_run(
        inp=inp, out=out, monkeypatch=monkeypatch, store=store,
        pressure_first_boundary=True, capture=captured,
        read_state_at_dispatch=True,
    )

    # ---- the governed boundary really shrank the frozen batch -------------
    first = captured["checkpoints"][0]
    assert first["status"] == "accepted", first
    assert first["changed"] is True
    batch = first["batch_shrink"]
    assert batch is not None, first
    assert batch["to_batch"] < batch["from_batch"] == batch["frozen_batch"]
    assert (
        batch["admitted_peak_bytes"]
        <= batch["phase_budget_bytes"]
        < batch["required_peak_bytes"]
    )
    assert first["source"] in ("fresh", "fresh+latched")
    assert first["raw"] > 0
    # The class/tile/K tuple never moves: compare against the PRE-checkpoint
    # candidate recorded by the harness before the boundary (never a tautology).
    frozen = captured.get("pre_checkpoint_candidates") or []
    assert frozen, "the harness recorded no pre-checkpoint candidate"
    before = frozen[0]
    assert first["candidate"]["tile_px"] == before["tile_px"] == 128
    assert first["candidate"]["units_in_flight"] == before["units_in_flight"] == 1
    assert first["candidate"]["prefetch_units"] == before["prefetch_units"] == 0
    assert first["candidate"]["workers_per_unit"] == before["workers_per_unit"]
    assert first["candidate"]["backend"] == before["backend"] == "cpu"
    assert captured.get("pre_checkpoint_contracts")[0]["class"] == (
        zz.WORKING_SET_EXTERNAL
    )

    # ---- A: the durable state was ALREADY committed when the executor got
    # ---- the first successor task (read from disk inside the dispatch spy).
    dispatch_states = captured["dispatch_states"]
    assert dispatch_states, "the dispatch seam was never observed"
    first_dispatch = dispatch_states[0]
    assert first_dispatch["admission_batch"] == batch["to_batch"]
    assert first_dispatch["chain"] == [
        (batch["from_batch"], batch["to_batch"])
    ]

    # ---- the DURABLE state was written, with a contiguous chain ----------
    state = _find_state(store)
    admission = state["admission"]
    assert admission["working_set"]["class"] == zz.WORKING_SET_EXTERNAL
    assert admission["working_set"]["column_batch"] == batch["to_batch"]
    chain = state.get("column_batch_transitions") or []
    assert len(chain) == 1
    assert chain[0]["from_batch"] == batch["from_batch"]
    assert chain[0]["to_batch"] == batch["to_batch"]
    assert chain[0]["phase_budget_bytes"] == batch["phase_budget_bytes"]
    assert chain[0]["admitted_peak_bytes"] == batch["admitted_peak_bytes"]
    assert chain[0]["identity_unchanged"] is True

    # ---- EVERY dispatched task and external execution used the successor --
    assert captured["tasks"], "no task was dispatched"
    external_tasks = [
        task for task in captured["tasks"] if task["mode"] == "external"
    ]
    assert external_tasks
    assert all(
        task["external"] is not None and task["external"]["column_batch"]
        == batch["to_batch"]
        for task in external_tasks
    )
    assert not any(
        task["external"] is not None
        and task["external"]["column_batch"] == batch["from_batch"]
        for task in captured["tasks"]
    )

    # ---- publication names the batch that really ran ---------------------
    manifest = json.loads((out / "zegrid_manifest.json").read_text())
    assert manifest["stack_working_set"] == zz.WORKING_SET_EXTERNAL
    assert manifest["working_set"]["column_batch"] == batch["to_batch"]
    assert (
        manifest["working_set"]["batch_transition"]["to_batch"]
        == batch["to_batch"]
    )
    # The deterministic RUN structure carries the transition too.
    cell_batches = {
        int(cell["storage"]["external"]["column_batch"])
        for cell in manifest["cells"]
        if cell.get("storage") and cell["storage"].get("external")
    }
    assert cell_batches == {batch["to_batch"]}, cell_batches
    run_block = manifest["r55_scheduler"]["run"]
    assert "column_batch" in run_block["policy"]["shrink_order"]
    # C1: ONE batch transition is counted EXACTLY once (it is the auditable
    # accepted entry in shrink_steps; there is no second increment).
    assert run_block["shrink_count"] == 1, run_block["shrink_count"]
    assert len(captured["record_calls"]) == 1
    assert any(
        entry.get("batch_shrink") for entry in run_block["checkpoints"]
    )


@pytest.mark.slow
def test_batch_size_does_not_change_the_science(tmp_path, monkeypatch):
    """Same corpus/config, default batch vs shrunk batch: identical pixels."""
    inp = _write_corpus(tmp_path)
    ref_out = tmp_path / "ref"
    srh_out = tmp_path / "shrunk"
    _external_run(
        inp=inp, out=ref_out, monkeypatch=monkeypatch,
        store=ref_out / "zegrid-durable",
        pressure_first_boundary=False, capture={},
    )
    captured: dict = {}
    _external_run(
        inp=inp, out=srh_out, monkeypatch=monkeypatch,
        store=srh_out / "zegrid-durable",
        pressure_first_boundary=True, capture=captured,
    )
    batch = captured["checkpoints"][0]["batch_shrink"]
    assert batch is not None and batch["to_batch"] < batch["from_batch"]
    ref_manifest = json.loads((ref_out / "zegrid_manifest.json").read_text())
    srh_manifest = json.loads((srh_out / "zegrid_manifest.json").read_text())
    assert ref_manifest["stack_working_set"] == zz.WORKING_SET_EXTERNAL
    assert srh_manifest["working_set"]["column_batch"] == batch["to_batch"]
    assert (
        ref_manifest["working_set"]["column_batch"] == batch["from_batch"]
    )
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        np.testing.assert_array_equal(
            _fits_array(ref_out / name),
            _fits_array(srh_out / name),
            err_msg=f"{name}: the staging batch must not change the science",
        )


@pytest.mark.slow
def test_dense_boundary_refusal_dispatches_nothing(tmp_path, monkeypatch):
    """F2 safety: a refused dense boundary must dispatch ZERO stale tasks.

    Observed product behaviour (pre-existing, independent of P1): the
    dense->bounded recovery REFUSES its own rebuild because the scheduler guard
    ``assert_shrink_only`` treats the in-memory -> external class change (and the
    ``None -> 128`` tile introduction) as growth. The recovery is therefore
    unreachable for that class pair today. What P1 must guarantee — and this
    test proves — is that the refusal is NOT silent: no task is dispatched with
    the stale dense profile, and the run stops with the honest typed refusal.

    The shared pending-task remap (F2) is exercised by the column-batch product
    test above, which is a reachable bundle replacement.
    """
    inp = _write_corpus(tmp_path)
    out = tmp_path / "rec"
    store = out / "zegrid-durable"
    captured: dict = {}
    messages: list[str] = []

    def _cb(msg, _cur, lvl, **_kw):
        messages.append(str(msg))

    with pytest.raises(zz._R55RunRefused) as excinfo:
        _external_run(
            inp=inp, out=out, monkeypatch=monkeypatch, store=store,
            pressure_first_boundary=True, capture=captured,
            storage_profile="auto", progress_callback=_cb,
        )
    refusal = excinfo.value
    assert refusal.reason_code == renv.REASON_REFUSE_HOST_RAM
    assert refusal.storage_fallback_eligible is True
    assert captured["tasks"] == []
    assert (out / "zegrid_manifest.json").exists() is False
    joined = " | ".join(messages)
    assert "storage profile AUTO fell back to 'bounded'" in joined
    assert "refused a NON-shrink rebuild" in joined


# ---------------------------------------------------------------------------
# B. Persistence is mandatory: absent/degraded store and recorder failure
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_no_durable_store_refuses_and_dispatches_nothing(tmp_path, monkeypatch):
    """No healthy store => no in-memory-only shrink, zero task dispatched."""
    inp = _write_corpus(tmp_path)
    out = tmp_path / "nostore"
    captured: dict = {}
    with pytest.raises(zz._R55RunRefused) as excinfo:
        _external_run(
            inp=inp, out=out, monkeypatch=monkeypatch,
            store=None,   # no durable store configured at all
            pressure_first_boundary=True, capture=captured,
        )
    refusal = excinfo.value
    assert refusal.reason_code == renv.REASON_INTERNAL_ERROR
    assert "durable store" in refusal.detail
    assert captured["tasks"] == []
    assert captured["record_calls"] == []
    assert not (out / "zegrid_manifest.json").exists()


@pytest.mark.slow
def test_recorder_failure_refuses_before_dispatch(tmp_path, monkeypatch):
    """A refused durable record means: typed refusal, zero task, no success."""
    inp = _write_corpus(tmp_path)
    out = tmp_path / "recfail"
    store = out / "zegrid-durable"
    captured: dict = {}
    with pytest.raises(zz._R55RunRefused) as excinfo:
        _external_run(
            inp=inp, out=out, monkeypatch=monkeypatch, store=store,
            pressure_first_boundary=True, capture=captured,
            recorder_failure=True,
        )
    refusal = excinfo.value
    assert refusal.reason_code == renv.REASON_INTERNAL_ERROR
    assert "REFUSED" in refusal.detail or "could not be written" in refusal.detail
    # The recorder was attempted exactly once and nothing was dispatched.
    assert len(captured["record_calls"]) == 1
    assert captured["tasks"] == []
    assert not (out / "zegrid_manifest.json").exists()
    # No durable trace of a transition was written.
    state = _find_state(store)
    assert not (state.get("column_batch_transitions") or [])
    assert (
        (state.get("admission") or {}).get("working_set") or {}
    ).get("column_batch") != captured["record_calls"][0]["to_batch"]


@pytest.mark.slow
def test_same_boundary_retry_does_not_duplicate_the_durable_transition(
    tmp_path, monkeypatch
):
    """C: durable-level idempotence (real recorder, real state)."""
    from zemosaic.core.zegrid import run_resume as zres

    inp = _write_corpus(tmp_path)
    out = tmp_path / "retry"
    store = out / "zegrid-durable"
    captured: dict = {}
    _external_run(
        inp=inp, out=out, monkeypatch=monkeypatch, store=store,
        pressure_first_boundary=True, capture=captured,
    )
    proposal = dict(captured["record_calls"][0])
    state_path = _find_state_path(store)
    run_root = state_path.parent
    before = zres.read_state(run_root)
    assert len(before["column_batch_transitions"]) == 1
    assert (
        before["admission"]["working_set"]["column_batch"]
        == proposal["to_batch"]
    )
    identity = zres.read_run_identity(run_root) if hasattr(
        zres, "read_run_identity"
    ) else json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    # Replaying the SAME boundary must be refused by the recorder (its
    # from_batch is no longer the current batch) => one logical transition,
    # one chain entry.
    with pytest.raises(zres.ResumeError):
        zres.record_column_batch_shrink(
            run_root,
            transition=proposal,
            identity_sha256=zres._durable_identity_sha256(run_root),
        )
    after = zres.read_state(run_root)
    assert len(after["column_batch_transitions"]) == 1
    assert (
        after["admission"]["working_set"]["column_batch"]
        == proposal["to_batch"]
    )
    assert identity["run_id"] == after["run_id"]


def _durable_store_with_admission(root: Path, *, batch: int) -> Path:
    """A REAL durable store whose frozen admission starts at ``batch``.

    Only the durable bookkeeping is built (identity + validated state): no
    pixel, no cell payload. The recorder and the state validators are the
    production ones.
    """
    from zemosaic.core.zegrid import run_store as zrs
    from zemosaic.core.zegrid import run_resume as zres

    root.mkdir(parents=True, exist_ok=True)
    frames = [
        {
            "frame_id": "frame00000001.fits",
            "content_sha256": "%064x" % 1,
            "nbytes": 1024,
            "order": 0,
            "role": "light",
        },
        {
            "frame_id": "frame00000002.fits",
            "content_sha256": "%064x" % 2,
            "nbytes": 1024,
            "order": 1,
            "role": "light",
        },
    ]
    identity = zrs.build_run_identity(
        run_id="m33-p1-durable-fixture",
        inputs={"frames": frames},
        canvas={
            "shape": [6180, 12249],
            "wcs_sha256": "a" * 64,
            "geometry_sha256": "b" * 64,
        },
        grid={
            "nx": 12,
            "ny": 6,
            "tile_px": 128,
            "geometry_sha256": "c" * 64,
            "cell_ids_sha256": "d" * 64,
        },
        halo={"halo_px": 16},
        memberships={"sha256": "e" * 64, "n_cells": 72},
        gauge={"mode": "global", "sha256": "f" * 64},
        science={"sha256": "1" * 64, "contract_version": "1", "params": {}},
        dtypes={"planes": {"science_core": ["<f4"]}},
        backend={"backend": "cpu", "tile_px": 128, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": 18_941_517_824,
            "workers": 1,
            "prefetch_units": 0,
            "host": "fixture",
            "note": "P1 durable column-batch chain fixture (no pixels)",
        },
    )
    (root / "run.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zres.write_state(
        root,
        run_id=identity["run_id"],
        expected_cell_ids=["r0003c0002"],
        policy=zres.validate_policy(
            {
                "requested": "bounded",
                "effective": "bounded",
                "fallback_count": 0,
                "transitions": [],
            }
        ),
        plan_generation=0,
        finalized=False,
        admission=zres.build_admission(
            backend="cpu",
            tile_px=128,
            parity_qualified=False,
            working_set={
                "class": "external_spool",
                "tile_px": 128,
                "column_batch": int(batch),
            },
        ),
    )
    return root


def test_two_durable_shrinks_form_a_contiguous_chain():
    """D: 7296 -> 6912 -> 6528 through the REAL recorder and state validators.

    The exact M33 numbers are the plan's; the durable bookkeeping is production
    code (``record_column_batch_shrink`` + ``read_state``/``validate_state``).
    A second transition inside ONE product run needs two RUN boundaries, which
    the current one-wave dispatch does not produce for this corpus; that is why
    the two-step chain is proven at the recorder level and the single transition
    + ``shrink_count == 1`` is proven in the product run above.
    """
    from zemosaic.core.zegrid import run_resume as zres

    root = _durable_store_with_admission(
        Path(tempfile.mkdtemp()) / "store", batch=7296
    )
    digest = lambda: zres._durable_identity_sha256(root)  # noqa: E731
    first = {
        "from_batch": 7296, "to_batch": 6912, "frozen_batch": 7296,
        "required_peak_bytes": 15_363_724_306,
        "admitted_peak_bytes": 14_559_855_634,
        "phase_budget_bytes": 14_823_796_557,
        "reason": "RUN phase budget below the frozen host peak (first step)",
    }
    second = {
        "from_batch": 6912, "to_batch": 6528, "frozen_batch": 6912,
        "required_peak_bytes": 14_559_855_634,
        "admitted_peak_bytes": 13_755_986_962,
        "phase_budget_bytes": 14_478_260_869,
        "reason": "RUN phase budget below the reduced host peak (second step)",
    }
    zres.record_column_batch_shrink(
        root, transition=first, identity_sha256=digest()
    )
    zres.record_column_batch_shrink(
        root, transition=second, identity_sha256=digest()
    )
    state = zres.read_state(root)
    chain = state["column_batch_transitions"]
    assert [(e["from_batch"], e["to_batch"]) for e in chain] == [
        (7296, 6912), (6912, 6528)
    ]
    assert state["admission"]["working_set"]["column_batch"] == 6528
    # Contiguity is enforced by the recorder: a NON-contiguous third step and a
    # replayed step are both refused, so the chain cannot be forged or doubled.
    with pytest.raises(zres.ResumeError):
        zres.record_column_batch_shrink(
            root, transition=first, identity_sha256=digest()
        )
    with pytest.raises(zres.ResumeError):
        zres.record_column_batch_shrink(
            root,
            transition={
                "from_batch": 7296, "to_batch": 7296, "frozen_batch": 7296,
                "required_peak_bytes": 1, "admitted_peak_bytes": 1,
                "phase_budget_bytes": 1, "reason": "growth forged",
            },
            identity_sha256=digest(),
        )
    assert len(zres.read_state(root)["column_batch_transitions"]) == 2


@pytest.mark.slow
def test_finalized_state_preserves_the_transition_chain(tmp_path, monkeypatch):
    """E: finalize keeps the durable chain and the reduced admission batch."""
    from zemosaic.core.zegrid import run_resume as zres

    inp = _write_corpus(tmp_path)
    out = tmp_path / "final"
    store = out / "zegrid-durable"
    captured: dict = {}
    _external_run(
        inp=inp, out=out, monkeypatch=monkeypatch, store=store,
        pressure_first_boundary=True, capture=captured,
    )
    run_root = _find_state_path(store).parent
    state = zres.read_state(run_root)
    assert state["finalized"] is True
    batch = captured["record_calls"][0]
    chain = state["column_batch_transitions"]
    assert len(chain) == 1
    assert (chain[0]["from_batch"], chain[0]["to_batch"]) == (
        batch["from_batch"], batch["to_batch"]
    )
    assert state["admission"]["working_set"]["column_batch"] == (
        batch["to_batch"]
    )
    # The resume READER re-validates the same durable facts (no 7296 restore).
    assert zres.validate_state(state)["admission"]["working_set"][
        "column_batch"
    ] == batch["to_batch"]
