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
import signal
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
    native_binary = int(getattr(os, "O_BINARY", 0))

    def spy(path, flags, mode=0o777):
        recorded = int(flags)
        if str(path).startswith(str(root)):
            seen.append((str(path), recorded))
        # The sentinel has no meaning for the kernel: strip it and RESTORE the
        # platform's native O_BINARY, so real Windows I/O under the spy stays
        # binary (the regression must not mask the very defect it guards).
        kernel_flags = (recorded & ~BINARY_SENTINEL) | native_binary
        return real_open(str(path), kernel_flags, mode)

    monkeypatch.setattr(rs.os, "O_BINARY", BINARY_SENTINEL, raising=False)
    monkeypatch.setattr(rs.os, "open", spy)

    store = rs.RunCellStore.open(root, identity=identity(), create=True)
    store.commit_cell(payload(), attempt_id="a1")
    store.close()

    assert len(seen) >= 4, f"expected at least 4 store os.open calls, saw {len(seen)}"
    missing = [
        (path, hex(flags))
        for path, flags in seen
        if not flags & BINARY_SENTINEL
    ]
    assert not missing, f"store os.open without O_BINARY: {missing}"

    def _has(predicate):
        return any(predicate(path) for path, _ in seen)

    tmp_marker = os.sep + "tmp" + os.sep
    # (a) dedicated lock target
    assert _has(lambda path: path.endswith("LOCK")), "lock open site not exercised"
    # (b) manifest/identity/index temp write
    assert _has(lambda path: ".tmp-" in os.path.basename(path)), (
        "atomic manifest temp open not exercised"
    )
    # (c) the ACTUAL object write: run/tmp/<session>/obj-*.part (it is renamed
    #     into objects/, never opened there: asserting an objects/ open would
    #     only have matched the POSIX directory fsync by accident).
    assert _has(
        lambda path: tmp_marker in path
        and os.path.basename(path).startswith("obj-")
        and path.endswith(".part")
    ), "content-addressed object temp-write site not exercised"
    # (d) directory fsync exists on POSIX only; it stays a Windows no-op.
    if rs._IS_WINDOWS:  # pragma: no cover - exercised on the Windows bench
        assert not _has(os.path.isdir), (
            "fsync_dir must stay a documented no-op on Windows"
        )
    else:
        assert _has(os.path.isdir), "directory fsync open was not exercised"

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
    # While HELD, LOCK must not be read through another handle (Windows denies
    # it by design). The owner record is the supported, readable surface.
    info = store.lock_info()
    assert info["held_by_this_store"] is True
    assert info["owner"]["state"] == "held"
    assert info["owner"]["schema"] == rs.LOCK_SCHEMA
    assert b"\r" not in (root / "OWNER.json").read_bytes(), (
        "owner record was CRLF-translated"
    )
    store.commit_cell(payload(), attempt_id="a1")
    store.close()

    # After release the lock file is readable again: assert the exact bytes.
    assert (root / "LOCK").read_bytes() == rs.LOCK_PLACEHOLDER
    assert b"\r" not in (root / "LOCK").read_bytes()
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


def _terminate_tree(proc) -> None:
    """Kill the ACTUAL writer, not just a launcher/venv shim."""
    try:
        if os.name == "nt":  # pragma: no cover - Windows path
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=20,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _wait_handshake(ready: Path, proc, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ready.exists():
            try:
                return json.loads(ready.read_text(encoding="utf-8"))
            except Exception:
                pass
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"child exited before ready: rc={proc.returncode} {out!r} {err!r}")
        time.sleep(0.02)
    pytest.fail("child never published its readiness handshake")


def test_m2_live_owner_lock_is_never_retried_into_success(tmp_path, monkeypatch):
    root = tmp_path / "run"
    rs.RunCellStore.open(root, identity=identity()).close()
    ready = tmp_path / "ready.json"
    holder = (
        "import json, os, sys, time\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from zemosaic.core.zegrid import run_store as rs\n"
        f"store = rs.RunCellStore.open({str(root)!r}, identity=None)\n"
        f"fh = open({str(ready)!r}, 'w')\n"
        "json.dump({'pid': os.getpid(),"
        " 'process_start_token': rs.process_start_token(os.getpid()),"
        " 'session_id': str(store.session_id)}, fh)\n"
        "fh.flush(); os.fsync(fh.fileno()); fh.close()\n"
        "time.sleep(30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", holder],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    calls = {"n": 0}
    real_lock_fd = rs.StoreLock._lock_fd

    def counting(self, fd):
        calls["n"] += 1
        return real_lock_fd(self, fd)

    monkeypatch.setattr(rs.StoreLock, "_lock_fd", counting)
    try:
        handshake = _wait_handshake(ready, child)
        holder_pid = int(handshake["pid"])

        # The owner record must be readable while the lock is held, and it must
        # name the ACTUAL interpreter, which may differ from Popen.pid when the
        # executable is a Windows venv launcher.
        peeked = rs.StoreLock.peek_owner(root)
        assert peeked is not None, "owner record unreadable while the lock is held"
        assert peeked["state"] == "held"
        assert int(peeked["pid"]) == holder_pid
        assert rs.owner_is_live(peeked) is True
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
        _terminate_tree(child)

    # Once the ACTUAL holder is gone, the bounded wait recovers the orphan.
    store = rs.RunCellStore.open(root, identity=None)
    try:
        assert store.report()["lock_info"]["orphan_recovered"] is True
        assert int(store.report()["lock_info"]["previous_owner"]["pid"]) == holder_pid
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
