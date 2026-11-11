"""ZM-ZEGRID-CLASS-DOWNGRADE-RESUME — the ONE approved resume downgrade.

A run frozen as working-set class ``inmem`` used to become non-resumable as
soon as free RAM dropped, because the planner re-derived ``ram_tile`` and the
guard refused::

    frozen working-set class 'inmem' cannot be re-derived as 'ram_tile'

Tristan approved EXACTLY ONE transition at the resume boundary: ``inmem`` ->
``ram_tile`` (strictly less RAM), recorded explicitly (from, to, reason, RAM
observation, plan generation, identity unchanged) in the durable state, the
manifest and the run log. Everything else stays refused:

* ``ram_tile`` -> ``inmem`` (an upgrade would consume more than was admitted);
* anything -> ``external_spool`` (a different, unproven code path);
* ``external_spool`` -> anything.

The in-memory class is ALSO kept whenever it still fits: the planner's
makespan comparison may prefer streaming for speed, and a resume may not
downgrade gratuitously.

The last test is a production-shaped provider parity regression: one multi-tile
cell (patch 520x520, 14 frames, tile 64) through the in-memory and the bounded
memmap providers must be bit-identical on every published plane.
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
from zemosaic.core.canonical_engine import (  # noqa: E402
    CanonicalStackRequest,
)
from zemosaic.core.canonical_streaming import (  # noqa: E402
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import file_provider as zfp  # noqa: E402
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
# Planner fixtures (same shape as the M3 / frozen-identity suites)
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


def _admission(ws_class, *, tile=16, backend="cpu"):
    return zres.build_admission(
        backend=backend,
        tile_px=int(tile),
        parity_qualified=False,
        working_set={
            "class": str(ws_class),
            "tile_px": 0 if ws_class == "inmem" else int(tile),
            "column_batch": (1 if ws_class == "external_spool" else None),
        },
    )


def _bundle(
    tmp_path, monkeypatch, *, inmem_bytes, raw, ws_class="inmem",
    stream_table=None, bounded=False,
):
    descs = [_make_tan(i) for i in range(4)]
    canvas = zg.build_canvas(descs)
    table = stream_table or {
        int(t): 10 * MIB for t in zz.CPU_STREAMING_TILE_LADDER
    }
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded" if bounded else "dense")
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
    )


# ---------------------------------------------------------------------------
# 1 — the approved downgrade happens ONLY when inmem genuinely cannot fit
# ---------------------------------------------------------------------------


def test_frozen_inmem_downgrades_to_ram_tile_when_inmem_infeasible(
    tmp_path, monkeypatch
):
    # budget = 150 MiB; the frozen in-memory class needs 200 MiB (infeasible),
    # the RAM-tiled class needs 10 MiB per tile (feasible).
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=200 * MIB, raw=int(187.5 * MIB)
    )
    assert bundle.stack_working_set == zz.WORKING_SET_RAM_TILE
    assert bundle.chosen_mode == "stream"
    assert bundle.cpu_streaming_tile == 16
    block = bundle.working_set_block
    assert block["class"] == zz.WORKING_SET_RAM_TILE
    assert block["tile_size"] == 16
    assert block["frozen_class"] == zz.WORKING_SET_INMEM

    record = block["transition"]
    assert record["from_class"] == "inmem"
    assert record["to_class"] == "ram_tile"
    assert record["tile_px"] == 16
    assert record["column_batch"] is None
    assert record["reason"]
    observation = record["ram_observation"]
    assert observation["raw_available_bytes"] == int(187.5 * MIB)
    assert observation["plan_budget_bytes"] == (4 * int(187.5 * MIB)) // 5
    assert observation["inmem_required_bytes"] == 200 * MIB
    assert observation["ram_tile_required_bytes"] == 10 * MIB
    assert observation["deficit_bytes"] == 200 * MIB - (4 * int(187.5 * MIB)) // 5
    # The block is strict JSON (it is published verbatim).
    json.dumps(block)
    # ... and the layout view carries the SAME explicit record.
    assert bundle.layout["working_set"]["transition"] == record
    assert bundle.layout["working_set"]["frozen_class"] == "inmem"
    # Only pending Cells would run: every non-empty task is a stream task.
    assert all(
        task[4] in (None, "stream") for task in bundle.cell_tasks
    )
    # The plan stayed within its own budget.
    assert bundle.envelope.aggregate_bytes <= bundle.envelope.budget_bytes


def test_frozen_inmem_is_kept_when_it_still_fits(tmp_path, monkeypatch):
    """The planner prefers streaming for speed here; the resume must not.

    In-memory 100 MiB admits K=1 under a 150 MiB budget, while streaming
    10 MiB admits K=4 — the makespan comparison picks streaming. A frozen
    ``inmem`` class that still fits is pinned instead: no transition is
    recorded (no gratuitous downgrade).
    """
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=100 * MIB, raw=int(187.5 * MIB)
    )
    assert bundle.chosen_mode == "inmem"
    assert bundle.stack_working_set == zz.WORKING_SET_INMEM
    assert bundle.cells_in_flight == 1
    block = bundle.working_set_block
    assert block["class"] == zz.WORKING_SET_INMEM
    assert "transition" not in block
    assert "frozen_class" not in block
    # The pinned plan is still within the observed budget.
    assert bundle.envelope.aggregate_bytes <= bundle.envelope.budget_bytes


# ---------------------------------------------------------------------------
# 2 — every other direction stays refused, with accurate wording
# ---------------------------------------------------------------------------


def test_frozen_ram_tile_is_never_upgraded_to_inmem(tmp_path, monkeypatch):
    # 50 MiB in-memory bound admits K=4 (makespan 1.0) and beats streaming
    # (1.2): the derived class would be inmem, but the frozen class is ram_tile.
    with pytest.raises(zpar.ModeInfeasible) as excinfo:
        _bundle(
            tmp_path, monkeypatch, inmem_bytes=50 * MIB,
            raw=400 * MIB, ws_class="ram_tile",
        )
    message = str(excinfo.value)
    assert "cannot be upgraded to 'inmem'" in message
    assert "inmem -> ram_tile" in message
    assert "Remedy:" in message


def test_frozen_inmem_is_never_re_derived_as_external(tmp_path, monkeypatch):
    # Bounded profile with no RAM class feasible: the frozen-capacity guard
    # fires BEFORE the external selector, so an inmem-frozen run can never be
    # re-derived as the external spool. That is the structural refusal; the
    # explicit policy wording for the class switch is asserted below.
    monkeypatch.setattr(
        zz.shutil, "disk_usage",
        lambda path: SimpleNamespace(
            total=100 * 1024 * MIB, used=0, free=100 * 1024 * MIB
        ),
    )
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(
            tmp_path, monkeypatch, inmem_bytes=1 << 60,
            raw=int(187.5 * MIB), bounded=True,
            stream_table={
                int(t): 5 * 1024 * MIB for t in zz.CPU_STREAMING_TILE_LADDER
            },
        )
    message = str(excinfo.value)
    assert "class=inmem" in message
    assert "external" not in message.lower()
    # A class-switch refusal is never silent either: it names the reason.
    policy_message = zz._frozen_working_set_mismatch_message(
        "inmem", "external_spool"
    )
    assert "'external_spool'" in policy_message
    assert "inmem -> ram_tile" in policy_message
    assert "not part of the proven parity" in policy_message


def test_external_class_is_never_left_on_resume(tmp_path, monkeypatch):
    # A frozen external run is ALWAYS rebuilt as external (forced), even when
    # RAM would now admit a RAM plan: no silent class change in that direction.
    bundle = _bundle(
        tmp_path, monkeypatch, inmem_bytes=10 * MIB,
        raw=400 * MIB, ws_class="external_spool", bounded=True,
        stream_table={int(t): 5 * MIB for t in zz.CPU_STREAMING_TILE_LADDER},
    )
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert "transition" not in bundle.working_set_block


def test_transition_policy_is_exactly_one_direction():
    assert zz._working_set_downgrade_allowed("inmem", "ram_tile") is True
    for from_class, to_class in (
        ("ram_tile", "inmem"),
        ("inmem", "external_spool"),
        ("external_spool", "ram_tile"),
        ("external_spool", "inmem"),
        ("ram_tile", "external_spool"),
    ):
        assert zz._working_set_downgrade_allowed(
            from_class, to_class
        ) is False
    # The durable validator refuses the same directions.
    bad = {
        "schema": zres.WORKING_SET_TRANSITION_SCHEMA,
        "from_class": "ram_tile",
        "to_class": "inmem",
        "tile_px": 16,
        "column_batch": None,
        "reason": "no",
        "ram_observation": {},
        "plan_generation": 0,
        "identity_sha256": "a" * 64,
        "identity_unchanged": True,
        "recorded_unix_ns": 1,
    }
    with pytest.raises(zres.ResumeStateError):
        zres.validate_working_set_transition(bad)


# ---------------------------------------------------------------------------
# 3 — durable state: exactly one record, identity and commits untouched
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


def _identity(*, run_id="run-downgrade", tile_px=16):
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


def _partial_inmem_store(root):
    identity = _identity()
    store = zrs.RunCellStore.open(
        root, identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id="downgrade-test",
    )
    try:
        zres.write_state(
            root,
            run_id=store.run_id,
            expected_cell_ids=CELL_IDS,
            policy={
                "requested": "auto",
                "effective": "dense",
                "fallback_count": 0,
                "transitions": [],
                "dense_refusal": None,
            },
            admission=_admission("inmem"),
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
        identity_sha_before = store.identity["identity_sha256"] if "identity_sha256" in store.identity else zrs.identity_sha256(store.identity)
    finally:
        store.close()
    return run_id, identity_sha_before


def test_state_downgrade_is_recorded_once_and_identity_is_unchanged(tmp_path):
    root = tmp_path / "run"
    run_id, identity_sha_before = _partial_inmem_store(root)
    run_json_before = (root / "run.json").read_bytes()
    commits_before = _snapshot(root / "commits")

    proposal = {
        "from_class": "inmem",
        "to_class": "ram_tile",
        "tile_px": 16,
        "column_batch": None,
        "reason": (
            "the frozen in-memory working set no longer fits the observed "
            "plan budget"
        ),
        "ram_observation": {
            "raw_available_bytes": 187 * MIB,
            "plan_budget_bytes": 149 * MIB,
            "inmem_required_bytes": 200 * MIB,
            "ram_tile_required_bytes": 10 * MIB,
            "deficit_bytes": 51 * MIB,
        },
    }
    record = zres.record_working_set_downgrade(
        root, transition=proposal, identity_sha256=identity_sha_before
    )
    assert record["from_class"] == "inmem"
    assert record["to_class"] == "ram_tile"
    assert record["tile_px"] == 16
    assert record["identity_sha256"] == identity_sha_before
    assert record["identity_unchanged"] is True

    state = zres.read_state(root)
    # The frozen BACKEND contract is untouched; only the class moved.
    assert state["admission"]["backend"] == "cpu"
    assert state["admission"]["tile_px"] == 16
    assert state["admission"]["parity_qualified"] is False
    assert state["admission"]["working_set"] == {
        "class": "ram_tile", "tile_px": 16, "column_batch": None
    }
    # Explicit, additive record; the original class is never erased.
    assert len(state["working_set_transitions"]) == 1
    stored = state["working_set_transitions"][0]
    assert stored["from_class"] == "inmem"
    assert stored["to_class"] == "ram_tile"
    assert stored["reason"]
    assert stored["ram_observation"] == proposal["ram_observation"]
    assert stored["plan_generation"] == state["plan_generation"]
    assert stored["identity_sha256"] == identity_sha_before

    # The scientific identity and every commit are byte-identical.
    assert (root / "run.json").read_bytes() == run_json_before
    assert _snapshot(root / "commits") == commits_before

    # Exactly one transition: a second attempt is refused.
    with pytest.raises(zres.ResumeStateError):
        zres.record_working_set_downgrade(
            root, transition=proposal, identity_sha256=identity_sha_before
        )

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


def test_transition_trail_survives_state_rewrites(tmp_path):
    """An unrelated policy/finalization rewrite never drops the audit trail."""
    root = tmp_path / "run"
    _, identity_sha_before = _partial_inmem_store(root)
    proposal = {
        "from_class": "inmem",
        "to_class": "ram_tile",
        "tile_px": 16,
        "column_batch": None,
        "reason": "RAM observation forced the strictly-smaller class",
        "ram_observation": {"raw_available_bytes": 1, "plan_budget_bytes": 1},
    }
    record = zres.record_working_set_downgrade(
        root, transition=proposal, identity_sha256=identity_sha_before
    )
    state = zres.read_state(root)
    zres.update_policy(
        root, policy=state["policy"], plan_generation=int(state["plan_generation"])
    )
    after = zres.read_state(root)
    assert after["working_set_transitions"] == [record]
    assert after["admission"]["working_set"]["class"] == "ram_tile"

    boot = zres.ResumeBootstrap.open(root)
    try:
        boot.mark_finalized({})
    finally:
        boot.close()
    final = zres.read_state(root)
    assert final["finalized"] is True
    assert final["working_set_transitions"] == [record]
    assert final["admission"]["working_set"]["class"] == "ram_tile"


def test_working_set_class_is_not_part_of_the_identity_fingerprints(tmp_path):
    root = tmp_path / "run"
    _, identity_sha_before = _partial_inmem_store(root)
    proposal = {
        "from_class": "inmem",
        "to_class": "ram_tile",
        "tile_px": 16,
        "column_batch": None,
        "reason": "RAM observation forced the strictly-smaller class",
        "ram_observation": {"raw_available_bytes": 1, "plan_budget_bytes": 1},
    }
    zres.record_working_set_downgrade(
        root, transition=proposal, identity_sha256=identity_sha_before
    )
    store = zrs.RunCellStore.open(root, identity=None)
    try:
        report = store.report()
        assert "working_set" not in report["identity_fingerprints"]
        assert report["identity_sha256"] == identity_sha_before
        assert zrs.identity_sha256(store.identity) == identity_sha_before
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 4 — production-shaped multi-tile parity: inmem vs bounded memmap
# ---------------------------------------------------------------------------


def _multi_tile_corpus(n=14, h=520, w=520, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        frame = (
            rng.normal(0.0, 3.0, (h, w, 3)).astype(np.float32)
            + np.array([50, 55, 60], np.float32)
            + 1.5 * i
        )
        frame[0:3, :, :] = np.nan
        frames.append(frame)
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[1][40:60, 40:60] = False
    masks[4][:, 300:] = False
    masks[9][200:260, :] = False
    return frames, masks


def test_multi_tile_inmem_vs_bounded_memmap_payload_parity(tmp_path):
    """Patch 520x520, 14 frames, tile 64: identical published payloads.

    This is the production-shaped witness that makes the downgrade safe: the
    ONLY thing that changes is where the same aligned frames are staged.
    """
    frames, masks = _multi_tile_corpus()
    ids = [f"frame_{i:04d}" for i in range(len(frames))]
    cache = tmp_path / "aligned"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)

    request = CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="winsorized_sigma_clip",
        combine="mean",
        taper="footprint",
        taper_px=16.0,
        backend="cpu",
    )

    mem = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), request, tile_size=64
    )
    provider = zfp.MemmapCanonicalProvider(cache, max_open_mappings=64)
    try:
        tiled = run_canonical_stack_streaming(provider, request, tile_size=64)
    finally:
        provider.close()

    assert mem.science.shape == (520, 520, 3)
    np.testing.assert_array_equal(tiled.science, mem.science)
    np.testing.assert_array_equal(
        tiled.estimator_weight_sum, mem.estimator_weight_sum
    )
    np.testing.assert_array_equal(tiled.support_w1, mem.support_w1)
    np.testing.assert_array_equal(tiled.support_w2, mem.support_w2)
    np.testing.assert_array_equal(tiled.n_eff_support, mem.n_eff_support)
    np.testing.assert_array_equal(tiled.valid_mask, mem.valid_mask)
    np.testing.assert_array_equal(
        tiled.surviving_sample_count, mem.surviving_sample_count
    )
    np.testing.assert_array_equal(tiled.rejection_mask, mem.rejection_mask)
    assert tiled.provenance == mem.provenance
    # The bounded provider really used the bounded LRU (not an eager 28-file
    # cache): 14 frames = 28 files, cap 64 -> still eager here; prove the cap
    # is honoured instead by running a second, tighter provider.
    tight = zfp.MemmapCanonicalProvider(cache, max_open_mappings=4)
    try:
        tight_result = run_canonical_stack_streaming(
            tight, request, tile_size=64
        )
        assert tight.peak_open_mappings <= 4
        np.testing.assert_array_equal(tight_result.science, mem.science)
    finally:
        tight.close()
