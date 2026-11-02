"""ZM-ZEGRID-RESILIENT-RUN-RESUME M4 — resume bootstrap contract tests.

Plan §6.4 / §6.6. These tests exercise the resume engine WITHOUT a GPU and
without a full product run: identity-content validation, sparse pending sets,
policy restoration, corruption isolation, explicit selection and the never/auto
semantics. The end-to-end NEW-PROCESS kill/resume proof lives in
``tests/test_zegrid_resume_m4_e2e.py``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402
from zemosaic.core.zegrid import run_artifacts as zart  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.canonical_stacking import FrameExclusion  # noqa: E402
from zemosaic.core.canonical_streaming import (  # noqa: E402
    FixedNormalization,
    subset_fixed_normalization,
)

TOOLS = REPO_ROOT / "tools" / "zegrid_resume_m4"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import m4_resume_harness as HARNESS  # noqa: E402

H = W = 32
CHANNELS = 3
NX = NY = 2
CORE_H = CORE_W = 16
HALO = 4

DTYPES = {
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


def make_identity(
    *,
    run_id="run-m4",
    inputs=None,
    gauge_mode="per_cell",
    gauge_sha="2" * 64,
    memberships_sha="1" * 64,
):
    frames = inputs if inputs is not None else [
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
    return zrs.build_run_identity(
        run_id=run_id,
        inputs={"frames": frames},
        canvas={
            "shape": [H, W],
            "wcs_sha256": "c" * 64,
            "geometry_sha256": "d" * 64,
        },
        grid={
            "nx": NX,
            "ny": NY,
            "tile_px": CORE_H,
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": "f" * 64,
        },
        halo={"halo_px": HALO, "provenance": {"effective_px": HALO}},
        memberships={"sha256": memberships_sha, "n_cells": NX * NY},
        gauge={
            "mode": gauge_mode,
            "sha256": gauge_sha,
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_ids": ["frame_0001.fits", "frame_0002.fits"],
            "n_exclusions": 0,
        },
        science={
            "sha256": "3" * 64,
            "contract_version": "1",
            "params": {"reject": "sigma", "sigma": 3.0},
        },
        dtypes=DTYPES,
        backend={"backend": "cpu", "tile_px": CORE_H, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": 8 * 2 ** 20,
            "workers": 2,
            "prefetch_units": 0,
            "host": "test-host",
            "note": "runtime only",
        },
    )


CELL_IDS = ("r0c0", "r0c1", "r1c0", "r1c1")


def _geometry(row, col):
    cy0, cx0 = row * CORE_H, col * CORE_W
    cy1, cx1 = cy0 + CORE_H, cx0 + CORE_W
    py0, px0 = max(0, cy0 - HALO), max(0, cx0 - HALO)
    py1, px1 = min(H, cy1 + HALO), min(W, cx1 + HALO)
    return (py0, py1, px0, px1), (cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0)


def complete_result(index, *, attempt_seed=0):
    row, col = divmod(index, NX)
    cid = CELL_IDS[index]
    bounds, core_slice = _geometry(row, col)
    py0, py1, px0, px1 = bounds
    ph, pw = py1 - py0, px1 - px0
    rng = np.random.default_rng(1000 + index + attempt_seed)
    science = (
        rng.normal(size=(ph, pw, CHANNELS)).astype(np.float32)
        + np.float32(index)
    )
    valid = np.ones((ph, pw), dtype=bool)
    surviving = np.full((ph, pw), 3, dtype=np.int32)
    y0, y1, x0, x1 = core_slice
    core = {
        "science_core": science[y0:y1, x0:x1, :],
        "estimator_weight_sum_core": np.ones((CORE_H, CORE_W), dtype=np.float32),
        "support_w1_core": np.ones((CORE_H, CORE_W), dtype=np.float32),
        "support_w2_core": np.ones((CORE_H, CORE_W), dtype=np.float32),
        "n_eff_support_core": np.full((CORE_H, CORE_W), 3.0, dtype=np.float32),
        "valid_mask_core": np.ones((CORE_H, CORE_W), dtype=bool),
        "surviving_sample_count_core": np.full(
            (CORE_H, CORE_W), 3, dtype=np.int32
        ),
    }
    return {
        "cell_id": cid,
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 2,
        "total_bytes": 1234,
        "core": core,
        "patch": {
            "bounds": bounds,
            "core_slice": core_slice,
            "science": science,
            "valid_mask": valid,
            "surviving": surviving,
        },
        "record": {
            "cell_id": cid,
            "patch_bounds": [px0, py0, px1, py1],
            "patch_shape_hw": [ph, pw],
            "cell_gauge_mode": "per_cell",
            "backend_used": "cpu",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
    }


def seeded_store(root, *, identity=None, cell_indexes=(0, 1, 2, 3), attempt="a1"):
    """Create a run with state + exactly the requested committed cells."""
    identity = identity or make_identity()
    store = zrs.RunCellStore.open(root, identity=identity)
    try:
        for index in cell_indexes:
            store.commit_cell(complete_result(index), attempt_id=attempt)
    finally:
        store.close()
    zres.write_state(
        Path(root),
        run_id=str(identity["run_id"]),
        expected_cell_ids=CELL_IDS,
        policy={
            "requested": "auto",
            "effective": "bounded",
            "fallback_count": 1,
            "transitions": [
                {
                    "from": "dense",
                    "to": "bounded",
                    "phase": "plan_layout",
                    "reason_code": "REFUSE_HOST_RAM",
                    "detail": "synthetic",
                }
            ],
            "dense_refusal": {
                "phase": "plan_layout",
                "reason_code": "REFUSE_HOST_RAM",
                "detail": "synthetic",
            },
        },
        gauge_sha256=str(identity["science_contract"]["gauge"]["sha256"]),
        note="test state",
    )
    return Path(root)


# ---------------------------------------------------------------------------
# Mode + selection
# ---------------------------------------------------------------------------


def test_m4_mode_is_strict_and_never_by_default():
    assert zres.normalize_resume_mode(None) == "never"
    assert zres.normalize_resume_mode("") == "never"
    assert zres.normalize_resume_mode(" NEVER ") == "never"
    assert zres.normalize_resume_mode("Auto") == "auto"
    for bad in ("force", True, 1, "last"):
        with pytest.raises(zres.ResumeModeError):
            zres.normalize_resume_mode(bad)


def test_m4_discovery_and_explicit_selection(tmp_path):
    assert zres.discover_runs(tmp_path / "absent") == []
    assert zres.select_run(tmp_path / "absent") is None

    a = seeded_store(tmp_path / "a", identity=make_identity(run_id="run-a"))
    b = seeded_store(tmp_path / "b", identity=make_identity(run_id="run-b"))

    candidates = zres.discover_runs(tmp_path)
    assert {c["run_id"] for c in candidates} == {"run-a", "run-b"}
    with pytest.raises(zres.ResumeSelectionRequired) as exc:
        zres.select_run(tmp_path)
    assert len(exc.value.candidates) == 2

    chosen = zres.select_run(tmp_path, "run-b")
    assert chosen["run_id"] == "run-b"
    assert chosen["path"] == str(b)
    with pytest.raises(zres.ResumeNotFound):
        zres.select_run(tmp_path, "run-z")


def test_m4_state_missing_refuses_and_roundtrips(tmp_path):
    root = tmp_path / "run"
    identity = make_identity()
    store = zrs.RunCellStore.open(root, identity=identity)
    store.close()
    with pytest.raises(zres.ResumeStateError):
        zres.ResumeBootstrap.open(root)
    with pytest.raises(zres.ResumeStateError):
        zres.read_state(root)

    seeded_store(root, identity=identity, cell_indexes=())
def _state_roundtrip(root):
    state = zres.read_state(root)
    assert state["expected_cell_ids"] == list(CELL_IDS)
    assert state["policy"]["effective"] == "bounded"
    assert state["policy"]["fallback_count"] == 1
    state["expected_cell_ids"] = ["only", "only"]
    with pytest.raises(zres.ResumeStateError):
        zres.validate_state(state)


def test_m4_state_roundtrip_is_strict(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=())
    _state_roundtrip(root)


# ---------------------------------------------------------------------------
# Policy restoration
# ---------------------------------------------------------------------------


def test_m4_restores_bounded_policy_and_fallback_counter(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0,))
    boot = zres.ResumeBootstrap.open(root)
    try:
        seen = {}
        restored = boot.restore_policy(apply=lambda policy: seen.update(policy))
        assert restored["effective"] == "bounded"
        assert restored["fallback_count"] == 1
        assert restored["requested"] == "auto"
        assert seen["effective"] == "bounded"
        assert seen["transitions"][0]["to"] == "bounded"
    finally:
        boot.close()


# ---------------------------------------------------------------------------
# Sparse completeness
# ---------------------------------------------------------------------------


def test_m4_sparse_pending_is_not_a_prefix(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0, 2))
    boot = zres.ResumeBootstrap.open(root)
    try:
        plan = boot.plan(CELL_IDS)
        assert plan["pending"] == ["r0c1", "r1c1"]
        assert plan["pending_count"] == 2
        assert plan["complete"] == 2
        assert plan["is_complete"] is False
        assert plan["policy"].startswith("sparse")
        # never a 0..K prefix: cell 1 is pending while cell 2 is complete
        assert "r1c0" not in plan["pending"]
    finally:
        boot.close()


def test_m4_complete_run_is_verified_noop(tmp_path):
    root = seeded_store(tmp_path / "run")
    boot = zres.ResumeBootstrap.open(root)
    try:
        plan = boot.plan(CELL_IDS)
        assert plan["is_complete"] is True
        assert plan["pending"] == []
        descriptors = boot.completed_descriptors()
        assert len(descriptors) == 4
        assert all("payload_sha256" in row for row in descriptors)
    finally:
        boot.close()


def test_m4_corrupt_cell_is_isolated_and_recomputed(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0, 1, 2))
    # Corrupt exactly one object of cell r1c0 (index 2).
    store = zrs.RunCellStore.open(root, identity=None, read_only=True)
    try:
        record = store.commit_record("r1c0")
        plane_path = Path(root) / record.manifest["planes"][0]["path"]
    finally:
        store.close()
    os.chmod(plane_path, 0o600)
    plane_path.write_bytes(b"corrupted")

    boot = zres.ResumeBootstrap.open(root)
    try:
        plan = boot.plan(CELL_IDS)
        assert plan["invalid_isolated"] == ["r1c0"]
        assert plan["pending"] == ["r1c0", "r1c1"]
        assert plan["complete"] == 2
        assert plan["invalidated_outputs"] == [
            {"cell_id": "r1c0", "derived": "patch_and_offsets"}
        ]
    finally:
        boot.close()


def test_m4_global_identity_corruption_refuses(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0,))
    (Path(root) / "run.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(zres.ResumeCorruptGlobal):
        zres.ResumeBootstrap.open(root)


def test_m4_foreign_commit_refuses_globally(tmp_path):
    source = seeded_store(
        tmp_path / "other", identity=make_identity(run_id="run-other"),
        cell_indexes=(0,),
    )
    root = seeded_store(tmp_path / "run", cell_indexes=(1,))
    foreign = next((Path(source) / "commits").glob("*.json"))
    shutil.copy2(foreign, Path(root) / "commits" / foreign.name)
    boot = zres.ResumeBootstrap.open(root)
    try:
        with pytest.raises(zres.ResumeCorruptGlobal):
            boot.plan(CELL_IDS)
    finally:
        boot.close()


def test_m4_multiple_attempts_require_explicit_choice(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0, 1), attempt="a1")
    store = zrs.RunCellStore.open(root, identity=None)
    try:
        store.commit_cell(complete_result(0, attempt_seed=5), attempt_id="a2")
    finally:
        store.close()
    boot = zres.ResumeBootstrap.open(root)
    try:
        with pytest.raises(zres.ResumeIncompatible) as exc:
            boot.plan(CELL_IDS)
        assert "attempt" in str(exc.value)
    finally:
        boot.close()


# ---------------------------------------------------------------------------
# Content identity (never name/mtime)
# ---------------------------------------------------------------------------


def _real_inputs(tmp_path):
    paths = {}
    rows = []
    for index, name in enumerate(("f1.fits", "f2.fits")):
        path = tmp_path / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (f"frame-{index}-" * 100).encode("utf-8")
        path.write_bytes(payload)
        sha, nbytes = zrs.content_sha256_file(path)
        paths[name] = str(path)
        rows.append(
            {
                "frame_id": name,
                "content_sha256": sha,
                "nbytes": int(nbytes),
                "order": index,
                "role": "light",
            }
        )
    return rows, paths


def test_m4_inputs_are_validated_by_content_and_relocation(tmp_path):
    rows, paths = _real_inputs(tmp_path)
    root = seeded_store(
        tmp_path / "run", identity=make_identity(inputs=rows), cell_indexes=(0,)
    )
    boot = zres.ResumeBootstrap.open(root)
    try:
        assert boot.verify_inputs(paths)["count"] == 2

        # Relocation is NOT a change: copy to another directory.
        moved = {}
        for name, source in paths.items():
            target = tmp_path / "moved" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            moved[name] = str(target)
        assert boot.verify_inputs(moved)["count"] == 2

        # A changed byte IS a change.
        Path(paths["f1.fits"]).write_bytes(b"changed")
        with pytest.raises(zres.ResumeIncompatible) as exc:
            boot.verify_inputs(paths)
        assert "content sha256" in json.dumps(exc.value.reasons)
    finally:
        boot.close()


def test_m4_new_or_missing_frames_are_refused(tmp_path):
    rows, paths = _real_inputs(tmp_path)
    root = seeded_store(
        tmp_path / "run", identity=make_identity(inputs=rows), cell_indexes=(0,)
    )
    boot = zres.ResumeBootstrap.open(root)
    try:
        with pytest.raises(zres.ResumeIncompatible):
            boot.verify_inputs({"f1.fits": paths["f1.fits"]})
        with pytest.raises(zres.ResumeIncompatible):
            boot.verify_inputs({**paths, "f3.fits": paths["f1.fits"]})
    finally:
        boot.close()


def test_m4_changed_selection_is_refused(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0,))
    boot = zres.ResumeBootstrap.open(root)
    try:
        with pytest.raises(zres.ResumeIncompatible) as exc:
            boot.plan(["r0c0", "r0c1", "r1c0"])
        assert "frozen" in str(exc.value)
    finally:
        boot.close()


def test_m4_identity_mismatch_refuses(tmp_path):
    root = seeded_store(tmp_path / "run", cell_indexes=(0,))
    boot = zres.ResumeBootstrap.open(root)
    try:
        candidate = make_identity(gauge_sha="9" * 64)
        with pytest.raises(zres.ResumeIncompatible) as exc:
            boot.verify_identity(candidate)
        assert "gauge" in str(exc.value)
    finally:
        boot.close()


# ---------------------------------------------------------------------------
# Product wiring seams (no full run)
# ---------------------------------------------------------------------------


def test_m4_default_config_is_never_and_store_is_a_container():
    from zemosaic import zemosaic_config as zconfig_mod
    from zemosaic import zemosaic_zegrid_mode as zz

    defaults = zconfig_mod.DEFAULT_CONFIG
    assert defaults["zegrid_resume_mode"] == "never"
    assert defaults["zegrid_resume_run_id"] is None

    root = Path("/tmp/m4-generation-root")
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Empty container: the first run may use the root itself.
        assert zz._resume_generation_root(tmp, "run-a") == tmp
        (tmp / "run.json").write_text("{}", encoding="utf-8")
        child = zz._resume_generation_root(tmp, "run-a")
        assert child != tmp and child.parent == tmp
        child.mkdir()
        (child / "run.json").write_text("{}", encoding="utf-8")
        second = zz._resume_generation_root(tmp, "run-a")
        assert second not in (tmp, child) and second.parent == tmp


def test_m4_assembly_can_load_a_cell_absent_from_the_registry(tmp_path):
    """A resumed cell is not in this process' registry: reload from disk."""
    root = seeded_store(tmp_path / "run", cell_indexes=(0, 1, 2, 3))
    registry = zrr.RunResultRegistry(
        run_id="run-m4", expected_cell_ids=list(CELL_IDS), gauge_mode="per_cell",
    )
    store = zrs.RunCellStore.open(root, identity=None, read_only=True)
    try:
        assert registry.metadata("r0c0") is None
        payloads = store.ordered_payloads(registry, list(CELL_IDS))
        assert [p["cell_id"] for p in payloads] == list(CELL_IDS)
        assert all(p["status"] == "complete" for p in payloads)
        assert all(p.get("patch") for p in payloads)
        assert all(p["_store"]["source"] == "run_store" for p in payloads)
    finally:
        store.close()
        registry.dispose()


# ---------------------------------------------------------------------------
# Frozen gauge + geometry artifacts (REWORK-1)
# ---------------------------------------------------------------------------


def _artifact_run(tmp_path, *, mutated_membership=False):
    rows = HARNESS.ensure_inputs(tmp_path / "in")
    identity = HARNESS.make_identity("run-artifacts", rows)
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=identity)
    store.close()
    zres.write_state(
        root,
        run_id="run-artifacts",
        expected_cell_ids=HARNESS.CELL_IDS,
        policy={
            "requested": "auto", "effective": "dense",
            "fallback_count": 0, "transitions": [], "dense_refusal": None,
        },
        gauge_sha256=identity["science_contract"]["gauge"]["sha256"],
        note="artifact test",
    )
    gauge = HARNESS.synthetic_gauge()
    zart.write_gauge_artifact(
        root, run_id="run-artifacts", identity=identity, gauge=gauge,
        mode="per_cell", frame_ids=HARNESS.FRAME_IDS,
        reference_frame_id=HARNESS.FRAME_IDS[0],
    )
    canvas, layout, cell_ctxs = HARNESS.synthetic_geometry()
    if mutated_membership:
        _r, _c, cell, patch, _mem = cell_ctxs[0]
        cell_ctxs[0] = (
            _r, _c, cell, patch,
            zg.CellMembership(
                cell_id=cell.cell_id,
                core_ids=("frame_0002.fits",),
                patch_ids=tuple(HARNESS.FRAME_IDS),
            ),
        )
    zart.write_geometry_artifact(
        root, run_id="run-artifacts", identity=identity, canvas=canvas,
        layout=layout, cell_ctxs=cell_ctxs, halo_px=HARNESS.HALO,
    )
    return root, identity


def test_m4_gauge_and_geometry_artifacts_roundtrip(tmp_path):
    root, identity = _artifact_run(tmp_path)
    boot = zres.ResumeBootstrap.open(root)
    try:
        gauge, frame_ids, reference = boot.restore_gauge()
        assert tuple(frame_ids) == HARNESS.FRAME_IDS
        assert reference == HARNESS.FRAME_IDS[0]
        expected = HARNESS.synthetic_gauge()
        assert np.array_equal(gauge.coefficients, expected.coefficients)
        assert np.array_equal(gauge.norm_active, expected.norm_active)
        assert np.array_equal(gauge.weights, expected.weights)
        assert np.array_equal(gauge.weight_active, expected.weight_active)
        assert gauge.reference_index == expected.reference_index
        assert gauge.exclusions == expected.exclusions

        canvas, layout, cell_ctxs = boot.restore_geometry()
        assert [cell.cell_id for (_r, _c, cell, _p, _m) in cell_ctxs] == list(HARNESS.CELL_IDS)
        assert int(layout["nx"]) == HARNESS.NX and int(layout["ny"]) == HARNESS.NY
        assert int(canvas.width) == HARNESS.W and int(canvas.height) == HARNESS.H
        assert [cell.cell_id for (_r, _c, cell, _p, _m) in cell_ctxs] == list(HARNESS.CELL_IDS)
    finally:
        boot.close()


def test_m4_corrupt_gauge_artifact_refuses(tmp_path):
    root, _identity = _artifact_run(tmp_path)
    path = root / zart.GAUGE_FILENAME
    doc = json.loads(path.read_text())
    doc["coefficients"]["values"][0][0][0] = 123.0
    path.write_text(json.dumps(doc))
    with pytest.raises(zres.ResumeCorruptGlobal):
        zart.read_gauge_artifact(root)


def test_m4_tampered_geometry_is_refused_after_self_digest(tmp_path):
    root, identity = _artifact_run(tmp_path, mutated_membership=True)
    # The artifact itself is well-formed (self digest recomputed on write), but
    # its membership content no longer matches the frozen identity.
    with pytest.raises(zres.ResumeIncompatible) as exc:
        zart.restore_geometry(root, identity=identity)
    assert "memberships" in str(exc.value.reasons)


def test_m4_gauge_artifact_is_never_pickle_and_is_self_described(tmp_path):
    root, _identity = _artifact_run(tmp_path)
    raw = (root / zart.GAUGE_FILENAME).read_bytes()
    assert raw.startswith(b"{")
    assert b"\x80\x04" not in raw  # no pickle opcode stream
    assert b"__reduce__" not in raw
    doc = json.loads(raw)
    assert doc["schema"] == zart.GAUGE_SCHEMA
    assert "artifact_sha256" in doc
    assert doc["frame_count"] == len(HARNESS.FRAME_IDS)


# ---------------------------------------------------------------------------
# Gauge JSON strictness / inactive-NaN encoding (ZM-GAUGE-JSON-NONFINITE)
# ---------------------------------------------------------------------------


def _inactive_gauge(*, with_exclusion=True):
    """A gauge whose only excluded frame carries legitimate NaN coefficients.

    The first NaN uses a NON-canonical payload (sign bit set) so a lossy
    "NaN -> np.nan" round-trip would break the frozen array digest.
    """
    coefficients = np.zeros(
        (len(HARNESS.FRAME_IDS), CHANNELS, 2), dtype=np.float64
    )
    coefficients[:, :, 0] = 1.0
    coefficients[1, :, :] = np.nan
    coefficients[1, 0, 0] = np.array(
        [0xFFF8000000000001], dtype=np.uint64
    ).view(np.float64)[0]
    exclusions = (
        (
            FrameExclusion(
                index=1,
                stage="normalization",
                reason="insufficient_common",
                detail="synthetic norm-inactive frame",
            ),
        )
        if with_exclusion
        else ()
    )
    return FixedNormalization(
        reference_index=0,
        coefficients=coefficients,
        norm_active=np.array([True, False]),
        weights=np.array([1.0, 0.0], dtype=np.float64),
        weight_active=np.array([True, False]),
        exclusions=exclusions,
    )


def _inactive_run(tmp_path, *, gauge=None, run_id="run-inactive"):
    gauge = _inactive_gauge() if gauge is None else gauge
    rows = HARNESS.ensure_inputs(tmp_path / "in")
    identity = HARNESS.make_identity(run_id, rows)
    identity["science_contract"]["gauge"]["sha256"] = HARNESS.gauge_sha(gauge)
    identity["science_contract"]["gauge"]["n_exclusions"] = len(gauge.exclusions)
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=identity)
    store.close()
    zres.write_state(
        root,
        run_id=run_id,
        expected_cell_ids=HARNESS.CELL_IDS,
        policy={
            "requested": "auto",
            "effective": "dense",
            "fallback_count": 0,
            "transitions": [],
            "dense_refusal": None,
        },
        gauge_sha256=identity["science_contract"]["gauge"]["sha256"],
        note="inactive-NaN gauge test",
    )
    zart.write_gauge_artifact(
        root,
        run_id=run_id,
        identity=identity,
        gauge=gauge,
        mode="per_cell",
        frame_ids=HARNESS.FRAME_IDS,
        reference_frame_id=HARNESS.FRAME_IDS[0],
    )
    return root, identity, gauge


def _rewrite_with_recomputed_digest(path, mutate):
    doc = json.loads(path.read_text())
    mutate(doc)
    doc.pop("artifact_sha256", None)
    doc["artifact_sha256"] = zart._self_digest(doc)
    path.write_bytes(zrs.canonical_json(doc))


def test_m4_inactive_nan_gauge_roundtrip_preserves_bits_and_fingerprint(tmp_path):
    root, identity, expected = _inactive_run(tmp_path)
    boot = zres.ResumeBootstrap.open(root)
    try:
        gauge, frame_ids, reference = boot.restore_gauge()
    finally:
        boot.close()

    assert tuple(frame_ids) == HARNESS.FRAME_IDS
    assert reference == HARNESS.FRAME_IDS[0]
    assert np.array_equal(
        np.isnan(gauge.coefficients), np.isnan(expected.coefficients)
    )
    # Exact float64 bits (including the NaN payload/sign) must survive.
    assert np.array_equal(
        np.ascontiguousarray(gauge.coefficients).view(np.uint64),
        np.ascontiguousarray(expected.coefficients).view(np.uint64),
    )
    assert np.array_equal(gauge.norm_active, expected.norm_active)
    assert np.array_equal(gauge.weights, expected.weights)
    assert np.array_equal(gauge.weight_active, expected.weight_active)
    assert gauge.exclusions == expected.exclusions
    assert gauge.reference_index == expected.reference_index
    assert (
        zart.gauge_fingerprint(
            gauge=gauge,
            mode="per_cell",
            frame_ids=frame_ids,
            reference_frame_id=reference,
        )
        == identity["science_contract"]["gauge"]["sha256"]
    )


def test_m4_inactive_nan_gauge_json_has_no_non_finite_tokens(tmp_path):
    root, _identity, _gauge = _inactive_run(tmp_path)
    raw = (root / zart.GAUGE_FILENAME).read_bytes()
    # A strict parser must not see ANY NaN/Infinity literal token.
    doc = json.loads(
        raw,
        parse_constant=lambda token: pytest.fail(
            f"strict JSON must not contain a {token!r} token"
        ),
    )
    block = doc["coefficients"]
    assert block["encoding"] == zart.GAUGE_COEFFICIENTS_ENCODING
    assert block["dtype"] == "<f8"
    assert block["shape"] == [len(HARNESS.FRAME_IDS), CHANNELS, 2]
    assert len(block["non_finite"]) == CHANNELS * 2
    assert all(item["kind"] == "nan" for item in block["non_finite"])
    assert all(
        item["float64_bits"].startswith("0x")
        and len(item["float64_bits"]) == 18
        for item in block["non_finite"]
    )
    # Every declared non-finite position is a JSON null sentinel.
    for item in block["non_finite"]:
        i, c, k = item["index"]
        assert block["values"][i][c][k] is None
    assert b"\x80\x04" not in raw  # no pickle


def test_m4_nonfinite_active_gauge_write_is_refused_atomically(tmp_path):
    root, identity, _gauge = _inactive_run(tmp_path)
    path = root / zart.GAUGE_FILENAME
    before = path.read_bytes()

    active_nan = HARNESS.synthetic_gauge()
    active_nan.coefficients[0, 0, 0] = np.nan
    with pytest.raises(zres.ResumeStateError, match="norm-active"):
        zart.write_gauge_artifact(
            root,
            run_id="run-inactive",
            identity=identity,
            gauge=active_nan,
            mode="per_cell",
            frame_ids=HARNESS.FRAME_IDS,
            reference_frame_id=HARNESS.FRAME_IDS[0],
        )
    assert path.read_bytes() == before  # never a partial/incorrect artifact
    assert not list((root / "tmp").glob(".tmp-*"))

    infinite_inactive = _inactive_gauge()
    infinite_inactive.coefficients[1, 0, 0] = np.inf
    with pytest.raises(zres.ResumeStateError, match="infinite"):
        zart.write_gauge_artifact(
            root,
            run_id="run-inactive",
            identity=identity,
            gauge=infinite_inactive,
            mode="per_cell",
            frame_ids=HARNESS.FRAME_IDS,
            reference_frame_id=HARNESS.FRAME_IDS[0],
        )
    assert path.read_bytes() == before


def _tamper_sentinel_to_finite(doc):
    doc["coefficients"]["values"][1][0][0] = 0.0


def _tamper_activate_nan_frame(doc):
    doc["norm_active"][1] = True
    doc["exclusions"] = []


def _tamper_inactive_weight(doc):
    doc["weights"][1] = 0.5


def _tamper_exclusion_index(doc):
    doc["exclusions"][0]["index"] = 99


def _tamper_declared_bits_not_nan(doc):
    doc["coefficients"]["non_finite"][0]["float64_bits"] = "0x3ff0000000000000"


@pytest.mark.parametrize(
    "mutation, match",
    [
        (_tamper_sentinel_to_finite, "non_finite table"),
        (_tamper_activate_nan_frame, "norm-active"),
        (_tamper_inactive_weight, "exactly 0"),
        (_tamper_exclusion_index, "out of range"),
        (_tamper_declared_bits_not_nan, "not NaN"),
    ],
)
def test_m4_gauge_semantic_tampering_is_refused_after_self_digest(
    tmp_path, mutation, match
):
    root, _identity, _gauge = _inactive_run(tmp_path)
    path = root / zart.GAUGE_FILENAME
    _rewrite_with_recomputed_digest(path, mutation)
    with pytest.raises(zres.ResumeCorruptGlobal, match=match):
        zart.read_gauge_artifact(root)


def test_m4_legacy_v1_finite_gauge_is_still_readable_and_restorable(tmp_path):
    gauge = HARNESS.synthetic_gauge()
    rows = HARNESS.ensure_inputs(tmp_path / "in")
    identity = HARNESS.make_identity("run-legacy", rows)
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=identity)
    store.close()
    zres.write_state(
        root,
        run_id="run-legacy",
        expected_cell_ids=HARNESS.CELL_IDS,
        policy={
            "requested": "auto",
            "effective": "dense",
            "fallback_count": 0,
            "transitions": [],
            "dense_refusal": None,
        },
        gauge_sha256=identity["science_contract"]["gauge"]["sha256"],
        note="legacy v1 gauge fixture",
    )
    body = {
        "schema": zart.LEGACY_GAUGE_SCHEMA,
        "schema_version": zart.LEGACY_GAUGE_ARTIFACT_VERSION,
        "run_id": "run-legacy",
        "identity_sha256": zrs.identity_sha256(identity),
        "mode": "per_cell",
        "frame_ids": list(HARNESS.FRAME_IDS),
        "frame_count": len(HARNESS.FRAME_IDS),
        "reference_index": 0,
        "reference_frame_id": HARNESS.FRAME_IDS[0],
        "coefficients": gauge.coefficients.tolist(),
        "norm_active": [True, True],
        "weights": [1.0, 0.5],
        "weight_active": [True, True],
        "exclusions": [],
        "gauge_sha256": HARNESS.gauge_sha(gauge),
        "note": "legacy v1 finite coefficients",
    }
    payload = dict(body)
    payload["artifact_sha256"] = zart._self_digest(body)
    path = root / zart.GAUGE_FILENAME
    path.write_bytes(zrs.canonical_json(payload))

    raw = zart.read_gauge_artifact(root, identity=identity)
    assert raw["schema"] == zart.LEGACY_GAUGE_SCHEMA
    boot = zres.ResumeBootstrap.open(root)
    try:
        restored, frame_ids, reference = boot.restore_gauge()
    finally:
        boot.close()
    assert tuple(frame_ids) == HARNESS.FRAME_IDS
    assert reference == HARNESS.FRAME_IDS[0]
    assert np.array_equal(restored.coefficients, gauge.coefficients)
    assert np.array_equal(restored.norm_active, gauge.norm_active)
    assert np.array_equal(restored.weights, gauge.weights)
    assert np.array_equal(restored.weight_active, gauge.weight_active)

    # Legacy v1 never grows a NaN representation: a NaN token is refused.
    broken = json.loads(path.read_text())
    broken["coefficients"][1][0][0] = float("nan")
    path.write_text(json.dumps(broken))
    with pytest.raises(zres.ResumeCorruptGlobal):
        zart.read_gauge_artifact(root)


def test_m4_geometry_schema_version_is_not_inherited_from_gauge_bump(tmp_path):
    root, _identity = _artifact_run(tmp_path)
    gauge_doc = json.loads((root / zart.GAUGE_FILENAME).read_text())
    geo_doc = json.loads((root / zart.GEOMETRY_FILENAME).read_text())
    assert gauge_doc["schema"] == "zegrid-run-gauge-v2"
    assert gauge_doc["schema_version"] == "2"
    assert geo_doc["schema"] == zart.GEOMETRY_SCHEMA == "zegrid-run-geometry-v1"
    assert geo_doc["schema_version"] == "1"
    assert zart.ARTIFACT_VERSION == "1"
    zart.read_geometry_artifact(root)  # unchanged geometry v1 remains readable


def test_m4_restored_inactive_nan_gauge_feeds_production_subset(tmp_path):
    root, identity, expected = _inactive_run(tmp_path)
    boot = zres.ResumeBootstrap.open(root)
    try:
        gauge, frame_ids, _reference = boot.restore_gauge()
        # The production Cell-normalization consumer must accept the restored
        # gauge (shape/flags/exclusions) and preserve the inactive NaN.
        cell = subset_fixed_normalization(gauge, frame_ids, frame_ids)
        assert cell.frame_ids == tuple(frame_ids)
        assert np.array_equal(
            np.isnan(cell.coefficients), np.isnan(expected.coefficients)
        )
        assert np.array_equal(
            np.ascontiguousarray(cell.coefficients).view(np.uint64),
            np.ascontiguousarray(expected.coefficients).view(np.uint64),
        )
        reference_only = subset_fixed_normalization(
            gauge, frame_ids, (frame_ids[0],)
        )
        assert reference_only.frame_ids == (frame_ids[0],)
        assert reference_only.reference_index == 0
        assert reference_only.norm_active.tolist() == [True]
        assert boot.state["gauge_sha256"] == (
            identity["science_contract"]["gauge"]["sha256"]
        )
    finally:
        boot.close()
