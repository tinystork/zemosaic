"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 REWORK-1 — Windows-only defect regressions.

F1: every ``os.open`` in the durable store MUST request binary mode. Windows
translates CRLF<->LF for descriptors opened without ``O_BINARY``; that rewrites
any byte pattern containing 0x0D/0x0A and breaks the sha256 of an array object.

F2: an orphan lock left by a killed process must be recoverable within a
BOUNDED wait, and only when the recorded owner is verifiably dead. A live
owner's lock is never retried into success.

These tests run on POSIX too: F1 is enforced by injecting an ``O_BINARY``
sentinel at call time (POSIX has no such flag, so the sentinel is only
observable through our helper), and F2 is enforced with real child processes and
a lower bound override. Windows execution itself is NOT_RUN here.
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

from zemosaic.core.zegrid import run_store as rs  # noqa: E402

H, W = 8, 10
CORE = (2, 6, 2, 7)

DTYPES = {
    "planes": {
        "core.science_core": ["<f4"],
        "core.estimator_weight_sum_core": ["<f4"],
        "core.support_w1_core": ["<f4"],
        "core.support_w2_core": ["<f4"],
        "core.n_eff_support_core": ["<f4"],
        "core.valid_mask_core": ["|b1"],
        "core.surviving_sample_count_core": ["<i4"],
        "patch.science": ["<f4"],
        "patch.valid_mask": ["|b1"],
        "patch.surviving": ["<i4"],
    }
}


def identity(**overrides):
    common = dict(
        run_id="run-rework1",
        inputs={
            "frames": [
                {
                    "frame_id": "frame_0001.fits",
                    "content_sha256": "a" * 64,
                    "nbytes": 1024,
                    "order": 0,
                    "role": "light",
                }
            ]
        },
        canvas={"shape": [H, W], "wcs_sha256": "c" * 64, "geometry_sha256": "d" * 64},
        grid={
            "nx": 1,
            "ny": 1,
            "tile_px": 8,
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": "f" * 64,
        },
        halo={"halo_px": 2, "provenance": {"effective_px": 2}},
        memberships={"sha256": "1" * 64, "n_cells": 1},
        gauge={"mode": "per_cell", "sha256": "2" * 64},
        science={"sha256": "3" * 64, "contract_version": "1", "params": {"k": 1}},
        dtypes=DTYPES,
        backend={"backend": "cpu", "tile_px": 8, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": 4 * 2 ** 20,
            "workers": 1,
            "prefetch_units": 0,
            "host": "rework1",
            "note": "runtime only",
        },
        created_unix_ns=1,
    )
    common.update(overrides)
    return rs.build_run_identity(**common)


def payload(cell_id="r0000c0000", *, surviving_value=1):
    y0, y1, x0, x1 = CORE
    science = np.zeros((H, W, 3), dtype="<f4")
    valid = np.ones((H, W), dtype=bool)
    surviving = np.full((H, W), int(surviving_value), dtype="<i4")
    return {
        "cell_id": cell_id,
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 1,
        "total_bytes": 10,
        "core": {
            "science_core": science[y0:y1, x0:x1, :],
            "estimator_weight_sum_core": np.zeros((y1 - y0, x1 - x0), "<f4"),
            "support_w1_core": np.zeros((y1 - y0, x1 - x0), "<f4"),
            "support_w2_core": np.zeros((y1 - y0, x1 - x0), "<f4"),
            "n_eff_support_core": np.zeros((y1 - y0, x1 - x0), "<f4"),
            "valid_mask_core": valid[y0:y1, x0:x1],
            "surviving_sample_count_core": surviving[y0:y1, x0:x1],
        },
        "record": {
            "cell_id": cell_id,
            "patch_bounds": [0, 0, W, H],
            "patch_shape_hw": [H, W],
            "cell_gauge_mode": "per_cell",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell",
            "excluded": [],
        },
        "patch": {
            "bounds": (0, H, 0, W),
            "core_slice": CORE,
            "science": science,
            "valid_mask": valid,
            "surviving": surviving,
        },
    }


# ---------------------------------------------------------------------------
# F1 — binary mode at every os.open (read AND write)
# ---------------------------------------------------------------------------

BINARY_SENTINEL = 0x40000000  # a bit POSIX never defines; stripped by the spy


def test_m2_every_os_open_site_requests_binary_mode(tmp_path, monkeypatch):
    root = tmp_path / "run"
    seen: list[tuple[str, int]] = []
    real_open = os.open

    def spy(path, flags, mode=0o777):
        # Record every open that targets the store directory. ``os.open`` is
        # process-global, so unrelated library opens are ignored: the contract is
        # about the durable store's own sites.
        if str(path).startswith(str(root)):
            seen.append((str(path), int(flags)))
        # The sentinel has no meaning for the kernel: strip it, but RECORD that
        # the production code passed it.
        return real_open(str(path), int(flags) & ~BINARY_SENTINEL, mode)

    monkeypatch.setattr(rs.os, "O_BINARY", BINARY_SENTINEL, raising=False)
    monkeypatch.setattr(rs.os, "open", spy)

    store = rs.RunCellStore.open(root, identity=identity(), create=True)
    store.commit_cell(payload(), attempt_id="a1")
    store.close()

    # Four distinct sites: directory fsync (read-only), lock file, atomic
    # manifest/index temp write, content-addressed object write.
    assert len(seen) >= 4, f"expected at least 4 store os.open calls, saw {len(seen)}"
    missing = [
        (path, hex(flags))
        for path, flags in seen
        if not flags & BINARY_SENTINEL
    ]
    assert not missing, f"store os.open without O_BINARY: {missing}"
    # Coverage of every site that EXISTS on this platform. ``fsync_dir`` is a
    # deliberate no-op on Windows (no directory handle to fsync), so the
    # directory-open site is POSIX-only. That must not weaken the flag
    # assertion above, and the test is never skipped.
    assert any(path.endswith("LOCK") for path, _ in seen)
    assert any(".part" in path or ".tmp-" in path for path, _ in seen)
    assert any(os.sep + "objects" + os.sep in path for path, _ in seen)
    if rs._IS_WINDOWS:  # pragma: no cover - exercised on the Windows bench
        assert not any(
            os.path.isdir(path) for path, _ in seen
        ), "fsync_dir must stay a documented no-op on Windows"
    else:
        assert any(
            os.path.isdir(path) for path, _ in seen
        ), "directory fsync open was not exercised on POSIX"
    # And the store is still fully usable with the flag present.
    reopened = rs.RunCellStore.open(root, identity=None, read_only=True)
    assert reopened.verify_cell("r0000c0000")["valid"] is True
    reopened.close()


def test_m2_binary_mode_helper_reads_flag_at_call_time(monkeypatch):
    # The flag must never be cached at import: a future Windows-only regression
    # would then be invisible to this suite.
    monkeypatch.delattr(rs.os, "O_BINARY", raising=False)
    assert rs.binary_flag() == 0
    monkeypatch.setattr(rs.os, "O_BINARY", BINARY_SENTINEL, raising=False)
    assert rs.binary_flag() == BINARY_SENTINEL


def test_m2_crlf_like_payload_roundtrip_is_byte_exact(tmp_path):
    # int32 0x0A0D0D0A -> LE bytes 0A 0D 0D 0A: contains 0x0A, 0x0D and the
    # 0x0D0A sequence whose CRLF translation corrupted the Windows run.
    value = 0x0A0D0D0A
    result = payload(surviving_value=value)
    expected = np.ascontiguousarray(
        result["core"]["surviving_sample_count_core"]
    ).tobytes()
    assert b"\x0d\x0a" in expected, "test payload must contain a CRLF sequence"

    store = rs.RunCellStore.open(tmp_path / "run", identity=identity())
    try:
        receipt = store.commit_cell(result, attempt_id="a1")
        manifest = json.loads(
            next((tmp_path / "run" / "commits").glob("*.json")).read_text()
        )
        plane = next(
            p for p in manifest["planes"]
            if p["name"] == "core.surviving_sample_count_core"
        )
        stored = (tmp_path / "run" / plane["path"]).read_bytes()
        assert stored == expected, "stored bytes were translated on write"
        assert rs._file_sha256(tmp_path / "run" / plane["path"]) == plane["sha256"]
        assert plane["sha256"] == rs._sha256_bytes(expected)
        assert receipt.payload_sha256  # commit itself validated

        loaded = store.load_payload("r0000c0000")
        assert (
            np.ascontiguousarray(
                loaded["core"]["surviving_sample_count_core"]
            ).tobytes()
            == expected
        )
        assert store.verify_cell("r0000c0000")["valid"] is True
    finally:
        store.close()


def test_m2_lock_record_and_placeholder_are_byte_exact(tmp_path):
    assert rs.LOCK_PLACEHOLDER == b"{}\n"
    assert b"\r" not in rs.LOCK_PLACEHOLDER

    root = tmp_path / "run"
    store = rs.RunCellStore.open(root, identity=identity())
    # LOCK is a dedicated lock target: byte-exact, never a record.
    assert (root / "LOCK").read_bytes() == rs.LOCK_PLACEHOLDER
    assert b"\r" not in (root / "LOCK").read_bytes()
    # The owner record is separate and readable through the supported API.
    info = store.lock_info()
    assert info["held_by_this_store"] is True
    assert info["owner"]["state"] == "held"
    assert info["owner"]["schema"] == rs.LOCK_SCHEMA
    assert b"\r" not in (root / "OWNER.json").read_bytes(), (
        "owner record was CRLF-translated"
    )
    store.commit_cell(payload(), attempt_id="a1")
    store.close()

    assert (root / "LOCK").read_bytes() == rs.LOCK_PLACEHOLDER
    released = rs.StoreLock.peek_owner(root)
    assert released is not None and released["state"] == "released"
    assert b"\r" not in (root / "OWNER.json").read_bytes()
    # Canonical JSON has no newlines at all: the index cannot have been translated.
    assert b"\r" not in (root / "index.json").read_bytes()
    assert b"\n" not in (root / "index.json").read_bytes()


# ---------------------------------------------------------------------------
# F2 — bounded orphan-lock recovery (dead owner only)
# ---------------------------------------------------------------------------


def _write_owner(root: Path, *, pid: int, token: str, state: str = "held") -> None:
    # The owner record is a SEPARATE, never-locked file (F1 of REWORK-2).
    (root / "OWNER.json").write_text(
        json.dumps(
            {
                "schema": rs.LOCK_SCHEMA,
                "state": state,
                "instance_id": "previous-instance",
                "pid": int(pid),
                "process_start_token": token,
                "boot_id": rs.boot_id(),
                "host": rs.socket.gethostname(),
                "session_id": "previous",
                "started_unix_ns": 1,
                "platform": "test",
            }
        )
    )


def test_m2_dead_owner_lingering_lock_is_recovered_with_attempts(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    rs.RunCellStore.open(root, identity=identity()).close()
    _write_owner(root, pid=999_999, token="1", state="held")

    real_lock_fd = rs.StoreLock._lock_fd
    failures = {"left": 3}

    def flaky(self, fd):
        if failures["left"] > 0:
            failures["left"] -= 1
            raise rs.StoreLockError("simulated lingering Windows byte-range lock")
        return real_lock_fd(self, fd)

    monkeypatch.setattr(rs.StoreLock, "_lock_fd", flaky)
    monkeypatch.setattr(rs, "LOCK_REACQUIRE_INTERVAL_S", 0.01)

    store = rs.RunCellStore.open(root, identity=identity())
    try:
        info = store.report()["lock_info"]
        assert info["orphan_recovered"] is True
        assert info["previous_owner"]["instance_id"] == "previous-instance"
        assert info["acquire_attempts"] == 4
        assert info["acquire_waited_s"] >= 0.0
        receipt = store.commit_cell(payload(), attempt_id="a1")
        assert receipt.idempotent is False
        assert store.verify_cell("r0000c0000")["valid"] is True
    finally:
        store.close()


def test_m2_exhausted_lock_wait_reports_attempts(tmp_path, monkeypatch):
    root = tmp_path / "run"
    rs.RunCellStore.open(root, identity=identity()).close()
    _write_owner(root, pid=999_999, token="1", state="held")

    def always_busy(self, fd):
        raise rs.StoreLockError("simulated lingering Windows byte-range lock")

    monkeypatch.setattr(rs.StoreLock, "_lock_fd", always_busy)
    monkeypatch.setattr(rs, "LOCK_REACQUIRE_TIMEOUT_S", 0.12)
    monkeypatch.setattr(rs, "LOCK_REACQUIRE_INTERVAL_S", 0.01)

    started = time.monotonic()
    with pytest.raises(rs.StoreLockError) as err:
        rs.RunCellStore.open(root, identity=identity())
    elapsed = time.monotonic() - started
    assert err.value.attempts is not None and err.value.attempts >= 2
    assert "still held after" in str(err.value)
    assert "verifiably dead" in str(err.value)
    assert err.value.last_error
    # A dead-owner wait is bounded and short (never an infinite wait).
    assert elapsed < 2.0


def test_m2_live_owner_lock_is_never_retried_into_success(tmp_path, monkeypatch):
    root = tmp_path / "run"
    rs.RunCellStore.open(root, identity=identity()).close()
    ready = tmp_path / "ready"
    holder = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from zemosaic.core.zegrid import run_store as rs\n"
        f"store = rs.RunCellStore.open({str(root)!r}, identity=None)\n"
        f"open({str(ready)!r}, 'w').write('ready')\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", holder],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    calls = {"n": 0}
    real_lock_fd = rs.StoreLock._lock_fd

    def counting(self, fd):
        calls["n"] += 1
        return real_lock_fd(self, fd)

    monkeypatch.setattr(rs.StoreLock, "_lock_fd", counting)
    try:
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            if child.poll() is not None:
                out, err = child.communicate()
                pytest.fail(f"holder died: rc={child.returncode} {out!r} {err!r}")
            time.sleep(0.02)
        assert ready.exists(), "holder never acquired the lock"

        # F1 (REWORK-2): the owner record MUST be readable while another
        # process holds the lock. It lives outside the locked byte range, so no
        # PermissionError is possible here (that was the Windows defect).
        peeked = rs.StoreLock.peek_owner(root)
        assert peeked is not None, "owner record unreadable while the lock is held"
        assert peeked["state"] == "held"
        assert int(peeked["pid"]) == int(child.pid)
        assert rs.StoreLock.peek_owner(root / "LOCK") == peeked

        started = time.monotonic()
        with pytest.raises(rs.StoreLockError) as err:
            rs.RunCellStore.open(root, identity=None)
        elapsed = time.monotonic() - started
        # A live owner is refused immediately, without a retry loop.
        assert calls["n"] == 1
        assert err.value.attempts == 1
        assert "no retry" in str(err.value)
        assert elapsed < 1.0
    finally:
        child.kill()
        child.wait(timeout=15)

    # Once the owner is really gone, the bounded wait recovers it.
    store = rs.RunCellStore.open(root, identity=None)
    try:
        assert store.report()["lock_info"]["orphan_recovered"] is True
    finally:
        store.close()


def test_m2_peek_owner_never_reads_the_locked_file(tmp_path, monkeypatch):
    root = tmp_path / "run"
    rs.RunCellStore.open(root, identity=identity()).close()

    real_read_bytes = Path.read_bytes
    touched: list[str] = []

    def guarded(self):
        touched.append(str(self))
        if self.name == "LOCK":
            raise PermissionError(13, "simulated Windows locked range")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    record = rs.StoreLock.peek_owner(root)
    assert record is not None and record["state"] == "released"
    assert not [p for p in touched if Path(p).name == "LOCK"], (
        "the supported accessor must never touch the locked LOCK file"
    )
