"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 — durable run-store contract tests.

Covers plan §6.1–6.7 without a product run: identity/jauge, atomic publication,
reconstructible index, single-writer lock, sha256 integrity, disk budget,
sparse completeness, idempotent publication and the RAM release hand-off.

The real-orchestration store seam (and the kill matrix) live in
``tests/test_zegrid_resume_m2_store_run.py`` and
``tests/test_zegrid_run_store_m2_kill.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

H, W = 8, 10
CHANNELS = 3
CORE = (2, 6, 2, 7)  # y0, y1, x0, x1

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


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_identity(
    *,
    run_id="run-m2",
    gauge_mode="per_cell",
    ram=4 * 2 ** 20,
    workers=2,
    prefetch=0,
    inputs=None,
    canvas=None,
    grid=None,
    halo=None,
    memberships=None,
    gauge=None,
    science=None,
    backend=None,
    dtypes=None,
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
        canvas=canvas or {
            "shape": [H, W],
            "wcs_sha256": "c" * 64,
            "geometry_sha256": "d" * 64,
        },
        grid=grid or {
            "nx": 1,
            "ny": 1,
            "tile_px": 8,
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": "f" * 64,
        },
        halo=halo or {"halo_px": 2, "provenance": {"effective_px": 2}},
        memberships=memberships or {"sha256": "1" * 64, "n_cells": 1},
        gauge=gauge or {
            "mode": gauge_mode,
            "sha256": "2" * 64,
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_ids": ["frame_0001.fits", "frame_0002.fits"],
            "n_exclusions": 0,
        },
        science=science or {
            "sha256": "3" * 64,
            "contract_version": "1",
            "params": {"reject": "sigma", "sigma": 3.0},
        },
        dtypes=dtypes or DTYPE_CONTRACT,
        backend=backend or {"backend": "cpu", "tile_px": 8, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": int(ram),
            "workers": int(workers),
            "prefetch_units": int(prefetch),
            "host": "test-host",
            "note": "runtime only",
        },
    )


def complete_result(cell_id="r0000c0000", *, seed=1, mode="per_cell", dtypes=None):
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
    result = {
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
            "cell_gauge_mode": mode,
            "backend_used": "cpu",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell" if mode == "per_cell" else "global",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
    }
    if mode == "per_cell":
        result["patch"] = {
            "bounds": (0, H, 0, W),
            "core_slice": CORE,
            "science": bases["science"],
            "valid_mask": bases["valid"],
            "surviving": bases["surviving"],
        }
    return result


def empty_result(cell_id="r0000c0001", *, mode="per_cell"):
    return {
        "cell_id": cell_id,
        "status": "empty",
        "backend_used": "cpu",
        "n_frames": 0,
        "total_bytes": 0,
        "record": {
            "cell_id": cell_id,
            "status": "empty",
            "cell_gauge_mode": mode,
            "n_contributors": 0,
            "patch_bounds": [0, 0, W, H],
            "patch_shape_hw": [H, W],
            "reference_frame_id": None,
            "excluded": [],
        },
    }


def write_owner_record(root: Path, record: dict) -> None:
    """Publish a fabricated OWNER.json (never the locked LOCK file)."""
    (Path(root) / "OWNER.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def store(tmp_path):
    opened = zrs.RunCellStore.open(
        tmp_path / "run",
        identity=make_identity(),
        budget=zrs.DiskBudget(durable_cell_bytes=64 * 2 ** 20),
    )
    yield opened
    opened.close()


# ---------------------------------------------------------------------------
# Family: transaction + idempotence (plan §6.2 / §6.7)
# ---------------------------------------------------------------------------


def test_m2_commit_publishes_verified_planes_and_roundtrips_payload(store):
    result = complete_result()
    receipt = store.commit_cell(result, attempt_id="a1", backend="cpu")
    assert receipt.idempotent is False
    assert receipt.index_updated is True
    assert receipt.stored_bytes > 0
    assert store.verify_cell("r0000c0000")["valid"] is True

    payload = store.load_payload("r0000c0000")
    assert payload["cell_id"] == "r0000c0000"
    assert payload["status"] == "complete"
    assert payload["record"]["patch_bounds"] == [0, 0, W, H]
    assert tuple(payload["patch"]["bounds"]) == result["patch"]["bounds"]
    assert tuple(payload["patch"]["core_slice"]) == result["patch"]["core_slice"]
    zrr.validate_complete_payload(payload, gauge_mode="per_cell")
    assert zrr.fingerprint_result(payload) == zrr.fingerprint_result(result)
    for name, original in result["core"].items():
        loaded = payload["core"][name]
        assert loaded.dtype == original.dtype
        assert loaded.shape == original.shape
        assert loaded.tobytes() == np.ascontiguousarray(original).tobytes()
        assert loaded.flags.writeable is False
    for name in zrr.PATCH_PLANES:
        loaded = payload["patch"][name]
        assert loaded.flags.writeable is False
        assert loaded.tobytes() == np.ascontiguousarray(
            result["patch"][name]
        ).tobytes()


def test_m2_publication_is_idempotent_by_cell_and_attempt(store):
    result = complete_result()
    first = store.commit_cell(result, attempt_id="a1")
    second = store.commit_cell(result, attempt_id="a1")
    assert second.idempotent is True
    assert second.payload_sha256 == first.payload_sha256
    assert store.report()["counts"]["attempts"] == 1


def test_m2_conflicting_payload_for_same_identity_is_refused(store):
    store.commit_cell(complete_result(seed=1), attempt_id="a1")
    conflicting = complete_result(seed=2)
    with pytest.raises(zrs.StoreConflictError):
        store.commit_cell(conflicting, attempt_id="a1")
    # the published commit is untouched
    assert store.verify_cell("r0000c0000")["valid"] is True


def test_m2_empty_validated_cell_publishes_without_arrays(store):
    receipt = store.commit_cell(empty_result(), attempt_id="empty-1")
    assert receipt.status == zrs.STATUS_EMPTY_VALIDATED
    assert receipt.logical_bytes == 0
    payload = store.load_payload("r0000c0001")
    assert payload["status"] == "empty"
    assert "core" not in payload


def test_m2_sparse_completeness_never_uses_a_prefix(store):
    store.commit_cell(complete_result("r0000c0000"), attempt_id="a1")
    store.commit_cell(empty_result("r0003c0007"), attempt_id="a1")
    expected = ["r0000c0000", "r0003c0007", "r0001c0002"]
    report = store.completeness(expected)
    assert report["complete"] == 1
    assert report["empty_validated"] == 1
    assert report["missing"] == ["r0001c0002"]
    assert report["is_complete"] is False
    full = store.completeness(["r0000c0000", "r0003c0007"])
    assert full["is_complete"] is True


# ---------------------------------------------------------------------------
# Family: reconstructible index (plan §6.2 step 6 / §6.4)
# ---------------------------------------------------------------------------


def test_m2_index_is_reconstructible_after_a_lost_index(store):
    store.commit_cell(complete_result("r0000c0000"), attempt_id="a1")
    store.commit_cell(complete_result("r0001c0001", seed=2), attempt_id="a1")
    before = store.index()
    index_path = store.root / "index.json"
    index_path.unlink()
    store.rebuild_index(write=True)
    after = store.index()
    assert after["cells"] == before["cells"]
    assert after["counts"] == before["counts"]


def test_m2_index_is_rebuilt_when_interrupted_between_commit_and_index(tmp_path):
    root = tmp_path / "run"
    first = zrs.RunCellStore.open(root, identity=make_identity())
    first.commit_cell(complete_result("r0000c0000"), attempt_id="a1")
    # Simulate "kill after publication, before index": publish the second cell
    # then restore the pre-commit index bytes.
    saved_index = (root / "index.json").read_bytes()
    first.commit_cell(complete_result("r0001c0001", seed=2), attempt_id="a1")
    first.close()
    (root / "index.json").write_bytes(saved_index)

    reopened = zrs.RunCellStore.open(root, identity=None)
    assert reopened.index_rebuilt is True
    counts = reopened.report()["counts"]
    assert counts["cells"] == 2
    assert sorted(reopened.index()["cells"]) == ["r0000c0000", "r0001c0001"]
    reopened.close()


# ---------------------------------------------------------------------------
# Family: isolated corruption (plan §6.4 / §6.7)
# ---------------------------------------------------------------------------


def test_m2_corrupted_cell_is_invalidated_alone(store):
    store.commit_cell(complete_result("r0000c0000"), attempt_id="a1")
    store.commit_cell(complete_result("r0001c0001", seed=2), attempt_id="a1")
    manifest = json.loads(
        (store.root / "commits").glob("r0000c0000__*.json").__next__().read_text()
    )
    object_path = store.root / manifest["planes"][0]["path"]
    os.chmod(object_path, 0o644)
    data = bytearray(object_path.read_bytes())
    data[0] ^= 0xFF
    object_path.write_bytes(bytes(data))

    bad = store.verify_cell("r0000c0000")
    assert bad["valid"] is False
    assert "digest" in bad["reason"] or "mismatch" in bad["reason"]
    good = store.verify_cell("r0001c0001")
    assert good["valid"] is True
    report = store.completeness(["r0000c0000", "r0001c0001"])
    assert report["invalid"] == ["r0000c0000"]
    assert report["complete"] == 1
    assert report["is_complete"] is False
    # loading a corrupted cell refuses instead of returning partial data
    with pytest.raises(zrs.StoreIntegrityError):
        store.load_payload("r0000c0000")
    # nothing is "repaired" behind our back
    assert store.verify_cell("r0000c0000")["valid"] is False


def test_m2_corrupted_commit_metadata_is_invalid_on_reopen(tmp_path):
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=make_identity())
    store.commit_cell(complete_result("r0000c0000"), attempt_id="a1")
    store.close()
    commit_path = next((root / "commits").glob("r0000c0000__*.json"))
    manifest = json.loads(commit_path.read_text())
    manifest["payload_sha256"] = "0" * 64  # tamper WITHOUT updating commit_sha256
    commit_path.write_text(json.dumps(manifest))
    reopened = zrs.RunCellStore.open(root, identity=None)
    assert reopened.verify_cell("r0000c0000")["valid"] is False
    assert reopened.report()["counts"]["cells"] == 0
    reopened.close()


# ---------------------------------------------------------------------------
# Family: disk budget / disk full (plan §6.5)
# ---------------------------------------------------------------------------


class _Enospc:
    def __init__(self, stage):
        self.stage = stage
        self.hits = []

    def __call__(self, stage, info):
        if stage == self.stage:
            self.hits.append(dict(info))
            raise OSError(28, "No space left on device (injected)")


def test_m2_identity_write_failure_fails_closed(tmp_path):
    fault = _Enospc("identity")
    with pytest.raises(OSError):
        zrs.RunCellStore.open(
            tmp_path / "run", identity=make_identity(), fault_hook=fault
        )
    assert fault.hits


@pytest.mark.parametrize("stage", ["temp_write", "fsync", "commit_rename"])
def test_m2_disk_full_preserves_prior_commits_and_publishes_nothing(
    tmp_path, stage
):
    root = tmp_path / "run"
    fault = _Enospc(stage)
    store = zrs.RunCellStore.open(root, identity=make_identity())
    store.commit_cell(complete_result("r0000c0000"), attempt_id="a1")
    before = store.report()["counts"]
    assert before["cells"] == 1
    # Arm the injection for the SECOND commit only.
    store.fault_hook = fault

    with pytest.raises(zrs.StoreDiskFullError):
        store.commit_cell(complete_result("r0001c0001", seed=2), attempt_id="a1")
    assert fault.hits, "fault injection never reached the requested stage"
    after = store.report()["counts"]
    assert after["cells"] == 1
    assert store.verify_cell("r0000c0000")["valid"] is True
    assert store.verify_cell("r0001c0001")["valid"] is False
    store.close()

    # After the space is freed the same attempt publishes; prior commit intact.
    reopened = zrs.RunCellStore.open(root, identity=None)
    receipt = reopened.commit_cell(
        complete_result("r0001c0001", seed=2), attempt_id="a1"
    )
    assert receipt.idempotent is False
    assert reopened.verify_cell("r0000c0000")["valid"] is True
    assert reopened.completeness(["r0000c0000", "r0001c0001"])["is_complete"]
    reopened.close()


def test_m2_budget_exhaustion_refuses_before_writing(tmp_path):
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(
        root,
        identity=make_identity(),
        budget=zrs.DiskBudget(durable_cell_bytes=1024),
    )
    with pytest.raises(zrs.StoreDiskFullError):
        store.commit_cell(complete_result(), attempt_id="a1")
    assert store.report()["counts"]["cells"] == 0
    assert not list((root / "commits").glob("*.json"))
    store.close()


def test_m2_free_space_margin_is_enforced(tmp_path):
    root = tmp_path / "run"
    free = {"value": 10 ** 9}
    store = zrs.RunCellStore.open(
        root,
        identity=make_identity(),
        budget=zrs.DiskBudget(
            durable_cell_bytes=0, transactional_margin_bytes=256 * 2 ** 20
        ),
        free_bytes_fn=lambda path: free["value"],
    )
    store.commit_cell(complete_result(), attempt_id="a1")
    free["value"] = 200 * 2 ** 20  # below the transactional margin
    with pytest.raises(zrs.StoreDiskFullError):
        store.commit_cell(complete_result("r0001c0001", seed=2), attempt_id="a1")
    store.close()


# ---------------------------------------------------------------------------
# Family: single-writer lock (plan §6.2 / §6.7)
# ---------------------------------------------------------------------------


def test_m2_second_writer_in_same_process_is_refused(tmp_path):
    root = tmp_path / "run"
    first = zrs.RunCellStore.open(root, identity=make_identity())
    with pytest.raises(zrs.StoreLockError):
        zrs.RunCellStore.open(root, identity=make_identity())
    first.close()
    second = zrs.RunCellStore.open(root, identity=make_identity())
    second.close()


def test_m2_orphan_lock_is_recovered(tmp_path):
    root = tmp_path / "run"
    zrs.RunCellStore.open(root, identity=make_identity()).close()
    dead_owner = {
        "schema": zrs.LOCK_SCHEMA,
        "instance_id": "dead-instance",
        "pid": 999_999,
        "process_start_token": "1",
        "boot_id": zrs.boot_id(),
        "host": zrs.socket.gethostname(),
        "session_id": "previous",
        "started_unix_ns": 1,
        "platform": "test",
    }
    write_owner_record(root, dead_owner)
    store = zrs.RunCellStore.open(root, identity=make_identity())
    info = store.report()["lock_info"]
    assert info["orphan_recovered"] is True
    assert info["previous_owner"]["instance_id"] == "dead-instance"
    store.close()


def test_m2_live_process_without_os_lock_is_never_stolen(tmp_path):
    root = tmp_path / "run"
    zrs.RunCellStore.open(root, identity=make_identity()).close()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    try:
        token = zrs.process_start_token(child.pid)
        assert token is not None
        owner = {
            "schema": zrs.LOCK_SCHEMA,
            "instance_id": "live-other-process",
            "pid": int(child.pid),
            "process_start_token": token,
            "boot_id": zrs.boot_id(),
            "host": zrs.socket.gethostname(),
            "session_id": "other",
            "started_unix_ns": int(time.time_ns()),
            "platform": "test",
        }
        write_owner_record(root, owner)
        with pytest.raises(zrs.StoreLockError):
            zrs.RunCellStore.open(root, identity=make_identity())
    finally:
        child.kill()
        child.wait(timeout=10)


def test_m2_concurrent_writer_process_is_refused(tmp_path):
    root = tmp_path / "run"
    ready = tmp_path / "ready"
    # The holder must create the store first, so create it here then reopen in
    # a child that only holds the lock.
    creator = zrs.RunCellStore.open(root, identity=make_identity())
    creator.close()
    holder = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from zemosaic.core.zegrid import run_store as zrs\n"
        f"store = zrs.RunCellStore.open({str(root)!r}, identity=None)\n"
        f"open({str(ready)!r}, 'w').write('ready')\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", holder],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.02)
        if not ready.exists():
            out, err = child.communicate(timeout=5)
            pytest.fail(
                "child writer never acquired the store lock; "
                f"rc={child.returncode} out={out!r} err={err!r}"
            )
        with pytest.raises(zrs.StoreLockError):
            zrs.RunCellStore.open(root, identity=None)
    finally:
        child.kill()
        child.wait(timeout=10)
    # After the holder dies the lock is recoverable.
    store = zrs.RunCellStore.open(root, identity=None)
    assert store.report()["lock_info"]["orphan_recovered"] is True
    store.close()


# ---------------------------------------------------------------------------
# Family: identity compatibility (plan §6.3)
# ---------------------------------------------------------------------------


def test_m2_runtime_context_change_never_invalidates_science(tmp_path):
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=make_identity(ram=4 * 2 ** 20))
    store.commit_cell(complete_result(), attempt_id="a1")
    store.close()
    changed = make_identity(ram=64 * 2 ** 20, workers=16, prefetch=4)
    reopened = zrs.RunCellStore.open(root, identity=changed)
    assert reopened.completeness(["r0000c0000"])["is_complete"] is True
    reopened.close()


def test_m2_relocated_dataset_keeps_identity(tmp_path):
    moved = [
        {
            "frame_id": "/other/place/renamed_0001.fits",
            "content_sha256": "a" * 64,
            "nbytes": 1024,
            "order": 0,
            "role": "light",
        },
        {
            "frame_id": "/other/place/renamed_0002.fits",
            "content_sha256": "b" * 64,
            "nbytes": 2048,
            "order": 1,
            "role": "light",
        },
    ]
    verdict = zrs.check_identity_compatibility(
        make_identity(), make_identity(inputs=moved)
    )
    assert verdict["compatible"] is True


@pytest.mark.parametrize(
    "section,mutate",
    [
        ("inputs", lambda: make_identity(inputs=[
            {"frame_id": "frame_0001.fits", "content_sha256": "9" * 64,
             "nbytes": 1024, "order": 0, "role": "light"},
        ])),
        ("canvas", lambda: make_identity(canvas={
            "shape": [H, W], "wcs_sha256": "9" * 64, "geometry_sha256": "d" * 64,
        })),
        ("grid", lambda: make_identity(grid={
            "nx": 2, "ny": 1, "tile_px": 8,
            "geometry_sha256": "e" * 64, "cell_ids_sha256": "f" * 64,
        })),
        ("halo", lambda: make_identity(halo={"halo_px": 16})),
        ("memberships", lambda: make_identity(
            memberships={"sha256": "9" * 64, "n_cells": 1}
        )),
        ("gauge", lambda: make_identity(gauge={
            "mode": "per_cell", "sha256": "9" * 64,
            "reference_frame_id": "frame_0002.fits",
            "reference_frame_ids": ["frame_0001.fits", "frame_0002.fits"],
            "n_exclusions": 1,
        })),
        ("science", lambda: make_identity(science={
            "sha256": "9" * 64, "contract_version": "1",
            "params": {"reject": "sigma", "sigma": 5.0},
        })),
        ("backend", lambda: make_identity(backend={
            "backend": "gpu", "tile_px": 8, "parity_qualified": False,
        })),
    ],
)
def test_m2_scientific_change_is_refused_with_a_reason(section, mutate, tmp_path):
    root = tmp_path / "run"
    zrs.RunCellStore.open(root, identity=make_identity()).close()
    changed = mutate()
    verdict = zrs.check_identity_compatibility(make_identity(), changed)
    assert verdict["compatible"] is False
    assert section in [r["section"] for r in verdict["reasons"]]
    with pytest.raises(zrs.StoreIdentityMismatch) as err:
        zrs.RunCellStore.open(root, identity=changed)
    assert section in [r["section"] for r in err.value.reasons]
    # nothing was mutated by the refused open
    store = zrs.RunCellStore.open(root, identity=make_identity())
    store.close()


def test_m2_tile_change_without_parity_is_refused(tmp_path):
    root = tmp_path / "run"
    zrs.RunCellStore.open(root, identity=make_identity()).close()
    changed = make_identity(backend={
        "backend": "cpu", "tile_px": 64, "parity_qualified": False,
    })
    with pytest.raises(zrs.StoreIdentityMismatch) as err:
        zrs.RunCellStore.open(root, identity=changed)
    detail = " ".join(
        str(r.get("detail", "")) for r in err.value.reasons
    )
    assert "parity" in detail.lower()


# ---------------------------------------------------------------------------
# Family: format safety (plan §6.2 / §6.7)
# ---------------------------------------------------------------------------


def test_m2_no_pickle_and_no_object_dtype():
    source = Path(zrs.__file__).read_text(encoding="utf-8")
    assert "import pickle" not in source
    assert "pickle." not in source
    assert "np.load" not in source
    result = complete_result()
    result["core"]["science_core"] = np.array([{"a": 1}], dtype=object)
    store = None
    try:
        store = zrs.RunCellStore.open(
            Path("/tmp") / f"zegrid-m2-objdtype-{os.getpid()}",
            identity=make_identity(),
        )
        with pytest.raises(zrs.StoreFormatError):
            store.commit_cell(result, attempt_id="a1")
    finally:
        if store is not None:
            store.close()
            import shutil

            shutil.rmtree(store.root, ignore_errors=True)


def test_m2_unknown_schema_and_unknown_fields_are_refused(tmp_path):
    bad_identity = make_identity()
    bad_identity["schema"] = "other-schema"
    with pytest.raises(zrs.StoreFormatError):
        zrs.RunCellStore.open(tmp_path / "a", identity=bad_identity)

    identity = make_identity()
    identity["science_contract"]["unexpected"] = {"x": 1}
    with pytest.raises(zrs.StoreFormatError):
        zrs.RunCellStore.open(tmp_path / "b", identity=identity)


def test_m2_manifest_path_traversal_is_refused(tmp_path):
    root = tmp_path / "run"
    store = zrs.RunCellStore.open(root, identity=make_identity())
    store.commit_cell(complete_result(), attempt_id="a1")
    store.close()
    commit_path = next((root / "commits").glob("r0000c0000__*.json"))
    manifest = json.loads(commit_path.read_text())
    manifest["planes"][0]["path"] = "../../etc/passwd"
    body = {k: v for k, v in manifest.items() if k != "commit_sha256"}
    manifest["commit_sha256"] = zrs._sha256_bytes(zrs.canonical_json(body))
    commit_path.write_text(json.dumps(manifest))
    reopened = zrs.RunCellStore.open(root, identity=None)
    verdict = reopened.verify_cell("r0000c0000")
    assert verdict["valid"] is False
    entries = reopened.report()["invalid_entries"]
    assert entries, "the escaping manifest must be reported as invalid"
    assert any("escape" in entry["reason"] or "safe" in entry["reason"]
               for entry in entries)
    reopened.close()


def test_m2_cell_id_and_attempt_are_path_confined(store):
    with pytest.raises(zrs.StorePathError):
        store.commit_cell(complete_result("../../etc/passwd"), attempt_id="a1")
    with pytest.raises(zrs.StorePathError):
        zrs.safe_name("..", "cell_id")
    with pytest.raises(zrs.StorePathError):
        zrs.safe_name("a/b", "cell_id")


def test_m2_dtype_outside_contract_is_refused(store):
    result = complete_result()
    result["core"]["support_w1_core"] = result["core"]["support_w1_core"].astype(
        np.int32
    )
    with pytest.raises(zrs.StoreFormatError):
        store.commit_cell(result, attempt_id="a1")


# ---------------------------------------------------------------------------
# Family: release of large arrays + store hand-off (plan §6.1 / M2 deliverable 2)
# ---------------------------------------------------------------------------


def test_m2_release_after_commit_drops_retained_bytes_and_reloads(store):
    registry = zrr.RunResultRegistry(
        run_id=store.run_id,
        expected_cell_ids=["r0000c0000", "r0001c0001"],
        gauge_mode="per_cell",
    )
    first = complete_result("r0000c0000")
    registry.accept(first, backend="cpu", plan_generation=0, attempt_id="a1")
    before = registry.retained_bytes
    assert before > 0

    handoff = store.commit_and_release(
        registry, "r0000c0000", attempt_id="a1", backend="cpu",
        plan_generation=0,
    )
    assert handoff["release"]["released"] is True
    assert registry.retained_bytes < before
    assert registry.counts()["accepted"] == 1
    assert registry.counts()["released_cells"] == 1
    meta = registry.metadata("r0000c0000")
    assert meta["released"] is True
    assert meta["released_bytes"] > 0
    assert registry.get("r0000c0000") is None

    # A released payload may not be silently re-served from the registry.
    with pytest.raises(zrr.PayloadReleasedError):
        registry.ordered_payloads(["r0000c0000"])

    # Idempotent release: a second call changes nothing.
    again = registry.release_payload("r0000c0000")
    assert again["already_released"] is True
    assert again["released_bytes"] == 0

    # The store re-supplies the exact payload in canonical order.
    ordered = store.ordered_payloads(registry, ["r0000c0000"])
    assert ordered[0]["cell_id"] == "r0000c0000"
    assert ordered[0]["patch"]["science"].flags.writeable is False
    registry.dispose()


def test_m2_release_with_digest_mismatch_is_refused(store):
    registry = zrr.RunResultRegistry(
        run_id=store.run_id,
        expected_cell_ids=["r0000c0000"],
        gauge_mode="per_cell",
    )
    registry.accept(
        complete_result(), backend="cpu", plan_generation=0, attempt_id="a1"
    )
    with pytest.raises(zrr.AcceptanceError):
        registry.release_payload(
            "r0000c0000", expected_payload_sha256="0" * 64
        )
    assert registry.retained_bytes > 0
    registry.dispose()


def test_m2_shared_backing_buffer_is_counted_once_across_releases(store):
    shape = (CORE[1] - CORE[0], CORE[3] - CORE[2])
    shared = np.zeros(shape, dtype=np.float32)

    def _registry(mode):
        reg = zrr.RunResultRegistry(
            run_id=store.run_id,
            expected_cell_ids=["r0000c0000", "r0001c0001"],
            gauge_mode="per_cell",
        )
        for index, cell_id in enumerate(("r0000c0000", "r0001c0001")):
            result = complete_result(cell_id, seed=index + 1)
            if mode == "shared":
                result["core"]["support_w1_core"] = shared
            elif mode == "distinct":
                result["core"]["support_w1_core"] = np.zeros(
                    shape, dtype=np.float32
                )
            reg.accept(
                result, backend="cpu", plan_generation=0, attempt_id="a1"
            )
        return reg

    distinct = _registry("distinct")
    shared_reg = _registry("shared")
    # A shared backing allocation is counted ONCE: exactly one 80-byte root
    # disappears from the honest retained total.
    assert (
        distinct.retained_bytes - shared_reg.retained_bytes
        == int(shared.nbytes)
    ), "a shared backing allocation must be counted once"

    before = shared_reg.retained_bytes
    store.commit_and_release(
        shared_reg, "r0000c0000", attempt_id="a1", plan_generation=0
    )
    after_first = shared_reg.retained_bytes
    assert after_first > 0, "the shared base is still owned by the other cell"
    assert before > after_first
    store.commit_and_release(
        shared_reg, "r0001c0001", attempt_id="a1", plan_generation=0
    )
    assert shared_reg.retained_bytes == 0
    distinct.dispose()
    shared_reg.dispose()


def test_m2_commit_and_release_keeps_payload_when_commit_fails(tmp_path):
    fault = _Enospc("commit_rename")
    store = zrs.RunCellStore.open(
        tmp_path / "run", identity=make_identity(), fault_hook=fault
    )
    registry = zrr.RunResultRegistry(
        run_id=store.run_id,
        expected_cell_ids=["r0000c0000"],
        gauge_mode="per_cell",
    )
    registry.accept(
        complete_result(), backend="cpu", plan_generation=0, attempt_id="a1"
    )
    before = registry.retained_bytes
    with pytest.raises(zrs.StoreDiskFullError):
        store.commit_and_release(
            registry, "r0000c0000", attempt_id="a1", plan_generation=0
        )
    # The payload was NOT released because the commit is not durable.
    assert registry.retained_bytes == before
    assert registry.metadata("r0000c0000")["released"] is False
    assert store.verify_cell("r0000c0000")["valid"] is False
    registry.dispose()
    store.close()


def test_m2_reopened_released_payload_preserves_real_feather_assembly(tmp_path):
    """Exercise the actual consumer, without the RAM-gated FITS producer."""
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid.geometry import GlobalCanvas

    root = tmp_path / "assembly_store"
    source = complete_result()
    cid = source["cell_id"]
    canvas = GlobalCanvas(
        canvas_id="m2-roundtrip", wcs_header="", width=W, height=H,
        resolution_deg=0.001,
    )
    baseline, baseline_info = zz._assemble_per_cell_feathered(
        canvas, 1, 1, {cid: source}, halo_px=2,
    )
    registry = zrr.RunResultRegistry(
        run_id="run-m2", expected_cell_ids=[cid], gauge_mode="per_cell",
    )
    registry.accept(source, backend="cpu", plan_generation=0, attempt_id="a1")
    store = zrs.RunCellStore.open(root, identity=make_identity())
    try:
        store.commit_and_release(registry, cid, attempt_id="a1")
        assert registry.retained_bytes == 0
    finally:
        store.close()
    reopened = zrs.RunCellStore.open(root, identity=None, read_only=True)
    try:
        loaded = reopened.ordered_payloads(registry, [cid])[0]
        assembled, info = zz._assemble_per_cell_feathered(
            canvas, 1, 1, {cid: loaded}, halo_px=2,
        )
        assert info == baseline_info
        for name, original in vars(baseline).items():
            actual = getattr(assembled, name)
            if isinstance(original, np.ndarray):
                assert actual.dtype == original.dtype
                assert actual.shape == original.shape
                assert actual.tobytes() == original.tobytes(), name
            else:
                assert actual == original, name
        assert zrr.fingerprint_result(loaded) == zrr.fingerprint_result(source)
    finally:
        reopened.close()
        registry.dispose()


@pytest.mark.parametrize("fault", ["missing_patch", "missing_core_slice", "bad_bounds", "extra_array"])
def test_m2_incomplete_patch_geometry_is_rejected_on_reopen(tmp_path, fault):
    root = tmp_path / "bad_geometry"
    store = zrs.RunCellStore.open(root, identity=make_identity())
    try:
        store.commit_cell(complete_result(), attempt_id="a1")
    finally:
        store.close()
    path = next((root / "commits").glob("*.json"))
    manifest = json.loads(path.read_text())
    meta = manifest["payload_meta"]
    if fault == "missing_patch":
        del meta["patch"]  # exactly the historical incomplete format
    elif fault == "missing_core_slice":
        del meta["patch"]["core_slice"]
    elif fault == "bad_bounds":
        meta["patch"]["bounds"] = [0, H, 0]
    else:
        meta["patch"]["science"] = [[1.0]]
    # Even with a correct envelope digest, an incomplete payload is not usable.
    manifest["commit_sha256"] = zrs._sha256_bytes(zrs.canonical_json(
        {key: value for key, value in manifest.items() if key != "commit_sha256"}
    ))
    path.write_bytes(zrs.canonical_json(manifest))
    reopened = zrs.RunCellStore.open(root, identity=None, read_only=True)
    try:
        assert reopened.report()["counts"]["cells"] == 0
        assert reopened.verify_cell("r0000c0000")["valid"] is False
        assert path.exists(), "rejecting old incomplete data must not delete it"
    finally:
        reopened.close()


def test_m2_loaded_geometry_must_match_original_payload_digest(tmp_path):
    root = tmp_path / "changed_geometry"
    store = zrs.RunCellStore.open(root, identity=make_identity())
    try:
        store.commit_cell(complete_result(), attempt_id="a1")
    finally:
        store.close()
    path = next((root / "commits").glob("*.json"))
    manifest = json.loads(path.read_text())
    # Valid dimensions, wrong placement: shape/array-only checks cannot catch it.
    manifest["payload_meta"]["patch"]["core_slice"] = [1, 5, 2, 7]
    manifest["commit_sha256"] = zrs._sha256_bytes(zrs.canonical_json(
        {key: value for key, value in manifest.items() if key != "commit_sha256"}
    ))
    path.write_bytes(zrs.canonical_json(manifest))
    reopened = zrs.RunCellStore.open(root, identity=None, read_only=True)
    try:
        with pytest.raises(zrs.StoreIntegrityError, match="fingerprint mismatch"):
            reopened.load_payload("r0000c0000")
    finally:
        reopened.close()


@pytest.mark.parametrize("mode", ["global", "per_cell"])
def test_m2_loaded_complete_and_empty_payloads_keep_registry_fingerprints(tmp_path, mode):
    store = zrs.RunCellStore.open(tmp_path / mode, identity=make_identity(gauge_mode=mode))
    try:
        for result in [complete_result(mode=mode), empty_result(mode=mode)]:
            store.commit_cell(result, attempt_id="a1")
            loaded = store.load_payload(result["cell_id"])
            assert zrr.fingerprint_result(loaded) == zrr.fingerprint_result(result)
            assert ("patch" in loaded) == ("patch" in result)
    finally:
        store.close()


def test_m2_gc_finalizer_does_not_deadlock_guard(tmp_path, monkeypatch):
    """REGRESSION: an abandoned store finalizer must not re-enter the live-lock
    guard. A plain non-reentrant Lock deadlocks when ``gc.collect()`` (triggered
    while ``acquire()`` holds the guard) runs ``RunCellStore.__del__``."""
    import gc
    import threading

    root = tmp_path / "gc_reentry"

    def _open_pending_store():
        pending = zrs.RunCellStore(root / "pending")
        pending._lock = zrs.StoreLock(root / "pending" / "LOCK")
        pending.cycle = pending  # keep one GC cycle away from the OS lock

    def _open_cycle():
        pending = zrs.RunCellStore(root / "pending")
        pending._lock = zrs.StoreLock(root / "pending" / "LOCK")
        pending.cycle = pending

    # One store that is still referenced must keep its own lock; a second store
    # that goes out of scope and is collected must not deadlock the guard.
    _open_cycle()  # referenced nowhere afterwards, eligible for GC
    store = zrs.RunCellStore.open(root / "run", identity=make_identity())
    real_write = zrs.StoreLock._write_owner_record

    def collect_then_write(self, record):
        # The finalizer of any garbage RunCellStore runs here, while the caller
        # thread already holds _LIVE_LOCKS_GUARD inside acquire().
        gc.collect()
        return real_write(self, record)

    monkeypatch.setattr(zrs.StoreLock, "_write_owner_record", collect_then_write)
    try:
        store.commit_cell(complete_result(), attempt_id="a1")
        assert store.report()["counts"]["cells"] == 1
    finally:
        store.close()
