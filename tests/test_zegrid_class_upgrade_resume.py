"""ZM-ZEGRID-CLASS-UPGRADE-RESUME — the ONE approved resume class upgrade.

A real M33 run frozen as ``external_spool`` resumes on a 32 GB host using only
39 % of RAM (one Cell in flight, ~3 h ETA) because the external reducer is
single-owner and the run may not leave the frozen class — even though the RAM
streaming (``ram_tile``) class would fit and is bit-identical on real data.

Tristan approved EXACTLY ONE additional recorded transition at the resume
boundary: ``external_spool`` -> ``ram_tile``, and ONLY when a FRESH admission
under the current host proves the target class fits on the SAME frozen
geometry/halo/memberships/science/tile contract. If it does not fit, today's
behaviour is unchanged (stay external; batch-shrink/refusal as today).

These tests pin the recorded observation/envelope numbers, not merely that the
code runs. They also pin the interaction matrix with the class DOWNGRADE and
the SHRINK-ONLY column batch:

* ``inmem`` -> ``ram_tile`` downgrade unchanged;
* ``ram_tile`` -> ``inmem`` still REFUSED;
* ``column_batch`` shrink stays external-only and is INERT on ``ram_tile``;
* a run can never record both an upgrade and a downgrade;
* a fresh run records nothing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

MIB = 2 ** 20
H = W = 8
CHANNELS = 3
CORE = (2, 6, 2, 7)
CELL_IDS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")


# ---------------------------------------------------------------------------
# Planner fixtures (same shape as the class-downgrade suite)
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(*_a, **_k):
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


def _make_tan(i, h=40, w=40, scale=0.001, ra_step=0.004):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zz.os, "cpu_count", lambda: 8)


def _admission(ws_class, *, tile=16, backend="cpu", batch=1):
    return zres.build_admission(
        backend=backend,
        tile_px=int(tile),
        parity_qualified=False,
        working_set={
            "class": str(ws_class),
            "tile_px": 0 if ws_class == "inmem" else int(tile),
            "column_batch": (
                int(batch) if ws_class == "external_spool" else None
            ),
        },
    )


def _bundle(
    tmp_path,
    monkeypatch,
    *,
    inmem_bytes,
    raw,
    ws_class="external_spool",
    stream_table=None,
    bounded=True,
    allow_upgrade=False,
):
    descs = [_make_tan(i) for i in range(4)]
    canvas = zg.build_canvas(descs)
    table = stream_table or {
        int(t): 10 * MIB for t in zz.CPU_STREAMING_TILE_LADDER
    }
    monkeypatch.setattr(
        zz, "_STACK_STORAGE_PROFILE", "bounded" if bounded else "dense"
    )
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(inmem_bytes))
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None, **_kw: int(table[int(tile)]),
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: int(raw))
    return zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=int(raw),
        budget_bytes=(4 * int(raw)) // 5,
        cpu_count=8,
        cache_root=str(tmp_path / "cache"),
        pinned_layout=(2, 2),
        frozen_admission=_admission(ws_class),
        allow_frozen_external_upgrade=bool(allow_upgrade),
    )


# ---------------------------------------------------------------------------
# 1 — the approved upgrade happens exactly when ram_tile freshly fits
# ---------------------------------------------------------------------------


def test_frozen_external_upgrades_to_ram_tile_when_fresh_admission_fits(
    tmp_path, monkeypatch
):
    # budget = 320 MiB; the frozen external spool is single-owner, the fresh
    # ram_tile plan needs 5 MiB at the frozen tile 16. The upgrade must happen
    # and carry the EXACT observation/envelope of the decision.
    raw = 400 * MIB
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=1 << 62, raw=raw,
        ws_class="external_spool",
        stream_table={int(t): 5 * MIB for t in zz.CPU_STREAMING_TILE_LADDER},
        allow_upgrade=True,
    )

    assert bundle.chosen_mode == "stream"
    assert bundle.stack_working_set == zz.WORKING_SET_RAM_TILE
    assert bundle.cpu_streaming_tile == 16
    assert bundle.cells_in_flight >= 1
    assert bundle.external_contract is None

    block = bundle.working_set_block
    assert block["class"] == zz.WORKING_SET_RAM_TILE
    assert block["tile_size"] == 16
    assert block["column_batch"] is None
    assert block["frozen_class"] == zz.WORKING_SET_EXTERNAL

    record = block["transition"]
    assert record["from_class"] == "external_spool"
    assert record["to_class"] == "ram_tile"
    assert record["tile_px"] == 16
    assert record["column_batch"] is None
    assert record["reason"]

    observation = record["ram_observation"]
    assert observation["raw_available_bytes"] == raw
    assert observation["plan_budget_bytes"] == (4 * raw) // 5
    assert observation["frozen_tile_px"] == 16
    assert observation["frozen_column_batch"] == 1
    assert observation["ram_tile_required_bytes"] == 5 * MIB
    assert observation["plan_aggregate_bytes"] == bundle.envelope.aggregate_bytes
    assert observation["cells_in_flight"] == bundle.cells_in_flight
    assert observation["cache_workers_per_cell"] == (
        bundle.cache_workers_per_cell
    )
    # The plan genuinely fits the budget that admitted it.
    assert bundle.envelope.aggregate_bytes <= bundle.envelope.budget_bytes
    assert bundle.envelope.budget_bytes == (4 * raw) // 5

    # The block is strict JSON (it is published verbatim in the manifest).
    json.dumps(block)
    # ... and the layout view carries the SAME explicit record.
    assert bundle.layout["working_set"]["transition"] == record
    assert bundle.layout["working_set"]["frozen_class"] == "external_spool"
    # Only pending Cells execute, on the RAM streaming path.
    assert all(task[4] in (None, "stream") for task in bundle.cell_tasks)


def test_upgrade_keeps_the_frozen_tile_and_never_grows_it(
    tmp_path, monkeypatch
):
    # A smaller stream estimate at 8 px must NOT tempt the resume into a
    # different tile: the frozen tile 16 is the contract (shrink-only).
    raw = 400 * MIB
    table = {int(t): 100 * MIB for t in zz.CPU_STREAMING_TILE_LADDER}
    table[16] = 5 * MIB
    table[8] = 1 * MIB
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=1 << 62, raw=raw,
        ws_class="external_spool", stream_table=table, allow_upgrade=True,
    )
    assert bundle.cpu_streaming_tile == 16
    assert bundle.working_set_block["transition"]["tile_px"] == 16
    assert bundle.working_set_block["transition"]["ram_observation"][
        "frozen_tile_px"
    ] == 16


# ---------------------------------------------------------------------------
# 2 — when ram_tile does NOT fit, nothing changes (today's behaviour)
# ---------------------------------------------------------------------------


def test_frozen_external_stays_external_when_ram_tile_does_not_fit(
    tmp_path, monkeypatch
):
    # Every ladder tile needs 1000 MiB > the 320 MiB plan budget. The fresh
    # admission misses, so the run keeps its frozen external class: no
    # transition, the frozen tile and column batch, one Cell in flight.
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=1 << 62, raw=400 * MIB,
        ws_class="external_spool",
        stream_table={int(t): 1000 * MIB for t in zz.CPU_STREAMING_TILE_LADDER},
        allow_upgrade=True,
    )
    assert bundle.chosen_mode == "external"
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.cells_in_flight == 1
    assert bundle.cpu_streaming_tile == 16
    assert bundle.external_contract["tile_size"] == 16
    assert bundle.external_contract["column_batch"] == 1
    block = bundle.working_set_block
    assert block["class"] == zz.WORKING_SET_EXTERNAL
    assert block["tile_size"] == 16
    assert block["column_batch"] == 1
    assert "transition" not in block
    assert "frozen_class" not in block
    # Exactly today's external execution shape.
    assert all(task[4] in (None, "external") for task in bundle.cell_tasks)


def test_upgrade_is_inert_without_the_resume_flag(tmp_path, monkeypatch):
    # The SAME fitting host, but no resume-boundary permission: today's
    # behaviour is preserved bit-for-bit (forced external, no transition).
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=1 << 62, raw=400 * MIB,
        ws_class="external_spool",
        stream_table={int(t): 5 * MIB for t in zz.CPU_STREAMING_TILE_LADDER},
        allow_upgrade=False,
    )
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.chosen_mode == "external"
    assert "transition" not in bundle.working_set_block


# ---------------------------------------------------------------------------
# 3 — the downgrade is unchanged and the reverse upgrade stays refused
# ---------------------------------------------------------------------------


def test_inmem_downgrade_is_unchanged_by_the_new_upgrade_path(
    tmp_path, monkeypatch
):
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=200 * MIB,
        raw=int(187.5 * MIB), ws_class="inmem", bounded=False,
        allow_upgrade=True,
    )
    assert bundle.chosen_mode == "stream"
    assert bundle.stack_working_set == zz.WORKING_SET_RAM_TILE
    record = bundle.working_set_block["transition"]
    assert record["from_class"] == "inmem"
    assert record["to_class"] == "ram_tile"
    assert record["tile_px"] == 16
    assert record["ram_observation"]["inmem_required_bytes"] == 200 * MIB


def test_ram_tile_to_inmem_is_still_refused_with_the_flag_on(
    tmp_path, monkeypatch
):
    with pytest.raises(zpar.ModeInfeasible) as excinfo:
        _bundle(
            tmp_path, monkeypatch, inmem_bytes=50 * MIB,
            raw=400 * MIB, ws_class="ram_tile", bounded=False,
            allow_upgrade=True,
        )
    assert "cannot be upgraded to 'inmem'" in str(excinfo.value)


def test_transition_predicates_are_disjoint_and_exactly_two_directions():
    assert zz._working_set_upgrade_allowed(
        "external_spool", "ram_tile"
    ) is True
    assert zz._working_set_downgrade_allowed(
        "external_spool", "ram_tile"
    ) is False
    assert zz._working_set_downgrade_allowed("inmem", "ram_tile") is True
    assert zz._working_set_upgrade_allowed("inmem", "ram_tile") is False
    for from_class, to_class in (
        ("ram_tile", "inmem"),
        ("inmem", "external_spool"),
        ("external_spool", "inmem"),
        ("ram_tile", "external_spool"),
    ):
        assert zz._working_set_upgrade_allowed(from_class, to_class) is False
        assert zz._working_set_downgrade_allowed(from_class, to_class) is False


# ---------------------------------------------------------------------------
# 4 — durable state: recorded once, identity and commits untouched
# ---------------------------------------------------------------------------


DTYPE_CONTRACT = {
    "planes": {
        "core.science_core": ["<f4", "<f8"],
        "core.estimator_weight_sum_core": ["<f4", "<f8"],
        "core.support_w1_core": ["<f4", "<f8"],
        "core.support_w2_core": ["<f4", "<f8"],
        "core.n_eff_support_core": ["<f4", "<f8"],
        "core.valid_mask_core": ["|b1"],
        "core.surviving_sample_count_core": ["<i4", "<i8"],
        "patch.science": ["<f4", "<f8"],
        "patch.valid_mask": ["|b1"],
        "patch.surviving": ["<i4", "<i8"],
    }
}


def _identity(*, run_id="run-upgrade", tile_px=16):
    return zrs.build_run_identity(
        run_id=run_id,
        inputs={
            "frames": [
                {
                    "frame_id": "frame_0001.fits",
                    "content_sha256": "a" * 64,
                    "nbytes": 1024,
                    "order": 0,
                    "role": "light",
                },
                {
                    "frame_id": "frame_0002.fits",
                    "content_sha256": "b" * 64,
                    "nbytes": 2048,
                    "order": 1,
                    "role": "light",
                },
            ]
        },
        canvas={"shape": [H, W], "wcs_sha256": "c" * 64, "geometry_sha256": "d" * 64},
        grid={
            "nx": 2,
            "ny": 2,
            "tile_px": int(tile_px),
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": zz._r55_sha256_json(sorted(CELL_IDS)),
        },
        halo={"halo_px": 4, "provenance": {"effective_px": 4}},
        memberships={"sha256": "1" * 64, "n_cells": len(CELL_IDS)},
        gauge={
            "mode": "per_cell",
            "sha256": "2" * 64,
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_ids": ["frame_0001.fits", "frame_0002.fits"],
            "n_exclusions": 0,
        },
        science={
            "sha256": "3" * 64,
            "contract_version": "1",
            "params": {"reject": "sigma"},
        },
        dtypes=DTYPE_CONTRACT,
        backend={
            "backend": "cpu",
            "tile_px": int(tile_px),
            "parity_qualified": False,
        },
        execution_context={
            "available_ram_bytes": 64 * MIB,
            "workers": 2,
            "prefetch_units": 0,
            "host": "test-host",
            "note": "runtime only",
        },
    )


def _complete_result(cell_id="r0000c0000", seed=1):
    rng = np.random.default_rng(seed)
    y0, y1, x0, x1 = CORE
    bases = {
        "science": rng.normal(size=(H, W, CHANNELS)).astype(np.float32),
        "estimator": rng.normal(size=(H, W)).astype(np.float32),
        "w1": rng.normal(size=(H, W)).astype(np.float32),
        "w2": rng.normal(size=(H, W)).astype(np.float32),
        "neff": rng.normal(size=(H, W)).astype(np.float32),
        "valid": rng.integers(0, 2, size=(H, W)).astype(bool),
        "surviving": rng.integers(0, 5, size=(H, W)).astype(np.int32),
    }
    return {
        "cell_id": cell_id,
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 2,
        "total_bytes": 1234,
        "core": {
            "science_core": bases["science"][y0:y1, x0:x1, :],
            "estimator_weight_sum_core": bases["estimator"][y0:y1, x0:x1],
            "support_w1_core": bases["w1"][y0:y1, x0:x1],
            "support_w2_core": bases["w2"][y0:y1, x0:x1],
            "n_eff_support_core": bases["neff"][y0:y1, x0:x1],
            "valid_mask_core": bases["valid"][y0:y1, x0:x1],
            "surviving_sample_count_core": bases["surviving"][y0:y1, x0:x1],
        },
        "record": {
            "cell_id": cell_id,
            "patch_bounds": [0, 0, W, H],
            "patch_shape_hw": [H, W],
            "cell_gauge_mode": "per_cell",
            "backend_used": "cpu",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
        "patch": {
            "bounds": (0, H, 0, W),
            "core_slice": CORE,
            "science": bases["science"],
            "valid_mask": bases["valid"],
            "surviving": bases["surviving"],
        },
    }


def _snapshot(root):
    root = Path(root)
    snap = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "tmp" not in path.relative_to(root).parts:
            snap[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return snap


def _partial_external_store(root, *, batch=1):
    identity = _identity()
    store = zrs.RunCellStore.open(
        root, identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id="upgrade-test",
    )
    try:
        zres.write_state(
            root,
            run_id=store.run_id,
            expected_cell_ids=CELL_IDS,
            policy={
                "requested": "auto",
                "effective": "bounded",
                "fallback_count": 1,
                "transitions": [],
                "dense_refusal": None,
            },
            admission=_admission("external_spool", batch=batch),
        )
        store.commit_cell(
            _complete_result("r0000c0000"),
            attempt_id="cpu-wave:0:0:r0000c0000",
            backend="cpu",
        )
        store.commit_cell(
            _complete_result("r0001c0001", seed=2),
            attempt_id="cpu-wave:0:0:r0001c0001",
            backend="cpu",
        )
        run_id = store.run_id
        identity_sha_before = zrs.identity_sha256(store.identity)
    finally:
        store.close()
    return run_id, identity_sha_before


_UPGRADE_PROPOSAL = {
    "from_class": "external_spool",
    "to_class": "ram_tile",
    "tile_px": 16,
    "column_batch": None,
    "reason": (
        "fresh admission on this host fits the frozen contract; the "
        "single-owner external spool is no longer needed"
    ),
    "ram_observation": {
        "raw_available_bytes": 400 * MIB,
        "plan_budget_bytes": 320 * MIB,
        "frozen_tile_px": 16,
        "frozen_column_batch": 1,
        "ram_tile_required_bytes": 5 * MIB,
        "plan_aggregate_bytes": 6 * MIB,
        "cells_in_flight": 4,
        "cache_workers_per_cell": 2,
    },
}


def _root_with_one_upgrade(tmp_path):
    root = tmp_path / "run"
    _, identity_sha = _partial_external_store(root)
    record = zres.record_working_set_upgrade(
        root, transition=_UPGRADE_PROPOSAL, identity_sha256=identity_sha
    )
    return root, record, identity_sha


def test_state_upgrade_is_recorded_once_and_identity_is_unchanged(tmp_path):
    root = tmp_path / "run"
    run_id, identity_sha_before = _partial_external_store(root)
    run_json_before = (root / "run.json").read_bytes()
    commits_before = _snapshot(root / "commits")

    record = zres.record_working_set_upgrade(
        root, transition=_UPGRADE_PROPOSAL, identity_sha256=identity_sha_before
    )
    assert record["from_class"] == "external_spool"
    assert record["to_class"] == "ram_tile"
    assert record["tile_px"] == 16
    assert record["column_batch"] is None
    assert record["identity_sha256"] == identity_sha_before
    assert record["identity_unchanged"] is True
    # The recorded observation is exactly the fresh admission that justified
    # the move (never a re-derived or rounded number).
    assert record["ram_observation"] == _UPGRADE_PROPOSAL["ram_observation"]

    state = zres.read_state(root)
    # The frozen BACKEND contract is untouched; only the class moved.
    assert state["admission"]["backend"] == "cpu"
    assert state["admission"]["tile_px"] == 16
    assert state["admission"]["parity_qualified"] is False
    assert state["admission"]["working_set"] == {
        "class": "ram_tile", "tile_px": 16, "column_batch": None
    }
    assert len(state["working_set_transitions"]) == 1
    stored = state["working_set_transitions"][0]
    assert stored["ram_observation"] == _UPGRADE_PROPOSAL["ram_observation"]
    assert stored["plan_generation"] == state["plan_generation"]

    # The scientific identity and every commit are byte-identical.
    assert (root / "run.json").read_bytes() == run_json_before
    assert _snapshot(root / "commits") == commits_before

    # Exactly one transition: a second upgrade is refused (the durable class
    # already moved, so the frozen from_class no longer matches).
    with pytest.raises(zres.ResumeStateError):
        zres.record_working_set_upgrade(
            root, transition=_UPGRADE_PROPOSAL,
            identity_sha256=identity_sha_before,
        )
    after = zres.read_state(root)
    assert len(after["working_set_transitions"]) == 1
    assert after["working_set_transitions"][0] == record

    # The resumed store still validates, and only the two pending Cells run.
    boot = zres.ResumeBootstrap.open(root)
    try:
        assert boot.admission["working_set"]["class"] == "ram_tile"
        plan = boot.plan(list(CELL_IDS))
        assert plan["complete"] == 2
        assert plan["pending_count"] == 2
        assert sorted(plan["pending"]) == ["r0000c0001", "r0001c0000"]
    finally:
        boot.close()


def test_run_can_never_record_both_an_upgrade_and_a_downgrade(tmp_path):
    root, record, identity_sha = _root_with_one_upgrade(tmp_path)
    # A later downgrade attempt (or any second class move) is refused.
    with pytest.raises(zres.ResumeStateError):
        zres.record_working_set_downgrade(
            root,
            transition={
                "from_class": "inmem",
                "to_class": "ram_tile",
                "tile_px": 16,
                "column_batch": None,
                "reason": "second move",
                "ram_observation": {"raw_available_bytes": 1},
            },
            identity_sha256=identity_sha,
        )
    assert len(zres.read_state(root)["working_set_transitions"]) == 1

    # And a crafted two-entry state is refused on LOAD.
    state = zres.read_state(root)
    downgrade = {
        "schema": zres.WORKING_SET_TRANSITION_SCHEMA,
        "from_class": "inmem",
        "to_class": "ram_tile",
        "tile_px": 16,
        "column_batch": None,
        "reason": "crafted",
        "ram_observation": {"raw_available_bytes": 1},
        "plan_generation": 0,
        "identity_sha256": record["identity_sha256"],
        "identity_unchanged": True,
        "recorded_unix_ns": 1,
    }
    state["working_set_transitions"] = [dict(record), downgrade]
    with pytest.raises(zres.ResumeStateError) as excinfo:
        zres.validate_state(state)
    assert "exactly one class transition per" in str(excinfo.value)


def test_upgrade_record_survives_policy_rewrites(tmp_path):
    root, record, _ = _root_with_one_upgrade(tmp_path)
    state = zres.read_state(root)
    zres.update_policy(
        root, policy=state["policy"],
        plan_generation=int(state["plan_generation"]),
    )
    after = zres.read_state(root)
    assert after["working_set_transitions"] == [record]
    assert after["admission"]["working_set"]["class"] == "ram_tile"


def test_upgrade_validate_state_requires_a_ram_tile_admission(tmp_path):
    root, record, _ = _root_with_one_upgrade(tmp_path)
    state = zres.read_state(root)
    state["admission"]["working_set"]["class"] = "external_spool"
    state["admission"]["working_set"]["column_batch"] = 1
    with pytest.raises(zres.ResumeStateError) as excinfo:
        zres.validate_state(state)
    assert "does not claim" in str(excinfo.value)


def test_upgrade_record_is_strict_json_and_unknown_directions_fail():
    record = {
        "schema": zres.WORKING_SET_TRANSITION_SCHEMA,
        "from_class": "external_spool",
        "to_class": "ram_tile",
        "tile_px": 16,
        "column_batch": None,
        "reason": "fresh admission fits",
        "ram_observation": {"raw_available_bytes": 1},
        "plan_generation": 0,
        "identity_sha256": "a" * 64,
        "identity_unchanged": True,
        "recorded_unix_ns": 1,
    }
    assert zres.validate_working_set_transition(record)["to_class"] == "ram_tile"
    json.dumps(record)
    for from_class, to_class in (
        ("external_spool", "inmem"),
        ("inmem", "external_spool"),
        ("ram_tile", "inmem"),
    ):
        bad = dict(record)
        bad["from_class"] = from_class
        bad["to_class"] = to_class
        with pytest.raises(zres.ResumeStateError):
            zres.validate_working_set_transition(bad)


# ---------------------------------------------------------------------------
# 5 — the column batch stays external-only and is inert on ram_tile
# ---------------------------------------------------------------------------


def test_column_batch_shrink_is_inert_once_the_run_is_ram_tile(tmp_path):
    root, _, identity_sha = _root_with_one_upgrade(tmp_path)
    proposal = {
        "from_batch": 1,
        "to_batch": 1,
        "frozen_batch": 1,
        "required_peak_bytes": 10 * MIB,
        "admitted_peak_bytes": 5 * MIB,
        "phase_budget_bytes": 8 * MIB,
        "reason": "never on the RAM class",
    }
    with pytest.raises(zres.ResumeStateError) as excinfo:
        zres.record_column_batch_shrink(
            root, transition=proposal, identity_sha256=identity_sha
        )
    assert "carries no column batch" in str(excinfo.value)


def test_historical_batch_chain_is_accepted_beside_the_upgrade(tmp_path):
    """A batch shrunk while external, then upgraded, keeps its history."""
    root = tmp_path / "run"
    _, identity_sha = _partial_external_store(root, batch=4)
    shrink = {
        "from_batch": 4,
        "to_batch": 2,
        "frozen_batch": 4,
        "required_peak_bytes": 10 * MIB,
        "admitted_peak_bytes": 5 * MIB,
        "phase_budget_bytes": 8 * MIB,
        "reason": "external era shrink",
    }
    zres.record_column_batch_shrink(
        root, transition=shrink, identity_sha256=identity_sha
    )
    upgrade = dict(_UPGRADE_PROPOSAL)
    upgrade["ram_observation"] = dict(
        _UPGRADE_PROPOSAL["ram_observation"], frozen_column_batch=2
    )
    zres.record_working_set_upgrade(
        root, transition=upgrade, identity_sha256=identity_sha
    )
    state = zres.read_state(root)
    assert state["admission"]["working_set"]["class"] == "ram_tile"
    assert state["admission"]["working_set"]["column_batch"] is None
    assert len(state["column_batch_transitions"]) == 1
    assert state["column_batch_transitions"][0]["to_batch"] == 2
    # A batch transition on a ram_tile admission still carries no batch.
    with pytest.raises(zres.ResumeStateError):
        zres.record_column_batch_shrink(
            root, transition=shrink, identity_sha256=identity_sha
        )


# ---------------------------------------------------------------------------
# 6 — a fresh run records nothing and plans identically
# ---------------------------------------------------------------------------


def test_fresh_run_never_records_a_transition(tmp_path, monkeypatch):
    descs = [_make_tan(i) for i in range(4)]
    canvas = zg.build_canvas(descs)
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3,
        storage_profile=None, **_kw: 10 * MIB,
    )
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(50 * MIB))
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 400 * MIB)

    def build(allow):
        return zz._build_execution_bundle(
            canvas, descs, None,
            raw_available_bytes=400 * MIB,
            budget_bytes=320 * MIB,
            cpu_count=8,
            cache_root=str(tmp_path / "cache"),
            pinned_layout=(2, 2),
            allow_frozen_external_upgrade=bool(allow),
        )

    baseline = build(False)
    flagged = build(True)
    assert "transition" not in baseline.working_set_block
    assert "transition" not in flagged.working_set_block
    assert baseline.stack_working_set == flagged.stack_working_set
    assert baseline.chosen_mode == flagged.chosen_mode
    assert baseline.cpu_streaming_tile == flagged.cpu_streaming_tile
    assert baseline.plan == flagged.plan
    assert baseline.envelope == flagged.envelope


def test_plan_bundle_auto_threads_the_resume_upgrade_flag(monkeypatch):
    seen = {}

    def fake_build(*_args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(footprints=None, gauge_memory_model=None)

    monkeypatch.setattr(zz, "_build_execution_bundle", fake_build)
    zz._plan_bundle_auto(
        canvas=object(),
        descs=[],
        footprints=None,
        raw=1,
        budget=0,
        frozen_admission=_admission("external_spool"),
        allow_frozen_external_upgrade=True,
        bundle_common={},
    )
    assert seen["allow_frozen_external_upgrade"] is True
    assert seen["frozen_admission"]["working_set"]["class"] == (
        "external_spool"
    )
