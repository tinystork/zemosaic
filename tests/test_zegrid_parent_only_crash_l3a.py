"""Real parent-only crash witness; Linux fork, not Windows qualification."""
from __future__ import annotations

import hashlib
import os
import signal
import time
from pathlib import Path

import pytest

import test_zegrid_live_commit_l1 as L1
from zemosaic.core.zegrid import run_store as zrs


def _children(pid: int) -> set[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return {int(x) for x in path.read_text().split()}
    except FileNotFoundError:
        return set()


def _tree_digests(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*')) if path.is_file()
    }


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.skipif(not Path('/proc').exists() or not hasattr(os, 'fork'), reason='Linux fork witness')
def test_parent_only_crash_recovers_while_old_worker_is_alive(tmp_path):
    corpus = L1._corpus(tmp_path)
    baseline = tmp_path / 'baseline'
    base, base_status, _ = L1._spawn(tmp_path, corpus, baseline)
    base.join(timeout=900)
    assert not base.is_alive() and L1._status(base_status)['status'] == 'done'

    out = tmp_path / 'interrupted'
    release = tmp_path / 'release-a'
    prior_log = tmp_path / 'prior-executed.jsonl'
    old_returned = tmp_path / 'old-a-returned'
    old, old_status, store = L1._spawn(
        tmp_path, corpus, out, gate_cell='r0000c0000',
        release_path=str(release), executed_log=str(prior_log), gate_timeout_s=600,
        done_marker_cell='r0000c0000', done_marker_path=str(old_returned),
    )
    old_workers: set[int] = set()
    resumed = None
    try:
        commits = L1._wait_until(lambda: L1._commit_files(store), timeout_s=300)
        assert commits and 'r0000c0000' not in commits
        reader = zrs.RunCellStore.open(store, read_only=True, create=False)
        try:
            for cid in commits:
                reader.require_commit(cid)
        finally:
            reader.close()
        old_workers = L1._wait_until(lambda: _children(old.pid), timeout_s=10)
        assert old_workers and all(_alive(pid) for pid in old_workers)
        assert 'r0000c0000' in L1._executed(prior_log)
        assert not release.exists()
        os.kill(old.pid, signal.SIGKILL)  # ONLY parent, never killpg here.
        old.join(timeout=30)
        assert not old.is_alive() and not old_status.exists()
        assert all(_alive(pid) for pid in old_workers)
        assert set(L1._commit_files(store)) == set(commits)

        resume_log = tmp_path / 'resume-executed.jsonl'
        resumed, resume_status, _ = L1._spawn(
            tmp_path, corpus, out, executed_log=str(resume_log), resume_mode='auto'
        )
        resumed.join(timeout=900)
        assert not resumed.is_alive()
        assert L1._status(resume_status)['status'] == 'done'
        assert all(_alive(pid) for pid in old_workers), 'old worker must remain alive through recovery'
        assert set(L1._executed(resume_log)) == set(L1.ALL_CELLS) - set(commits)
        assert 'r0000c0000' in L1._executed(resume_log)
        assert not set(L1._executed(resume_log)) & set(commits)
        assert set(L1._commit_files(store)) == set(L1.ALL_CELLS)
        assert L1._sha256(L1._product(out)) == L1._sha256(L1._product(baseline))
        assert L1._sha256(out / 'mosaic_grid_coverage.fits') == L1._sha256(baseline / 'mosaic_grid_coverage.fits')
        owner = zrs.StoreLock.peek_owner(store)
        assert owner['state'] == 'released'
        run_root = store
        assert (run_root / 'commits').is_dir()
        commits_before = _tree_digests(run_root / 'commits')
        objects_before = _tree_digests(run_root / 'objects')
        science_before = {
            name: L1._sha256(out / name)
            for name in ('mosaic_grid.fits', 'mosaic_grid_coverage.fits')
        }
        assert commits_before and objects_before
        # Let the stale worker complete after the new run. It cannot publish
        # or alter any committed bytes, object payloads, or final science.
        release.write_text('go')
        assert L1._wait_until(old_returned.exists, timeout_s=60)
        assert _tree_digests(run_root / 'commits') == commits_before
        assert _tree_digests(run_root / 'objects') == objects_before
        assert {
            name: L1._sha256(out / name) for name in science_before
        } == science_before
        reader = zrs.RunCellStore.open(store, read_only=True, create=False)
        try:
            for cid in L1.ALL_CELLS:
                reader.require_commit(cid)
        finally:
            reader.close()
        assert set(L1._commit_files(store)) == set(L1.ALL_CELLS)
        assert zrs.StoreLock.peek_owner(store) == owner
    finally:
        release.write_text('go')
        if resumed is not None and resumed.is_alive():
            resumed.kill(); resumed.join(timeout=30)
        if old.is_alive():
            old.kill(); old.join(timeout=30)
        for pid in old_workers:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='POSIX fork witness')
def test_child_release_does_not_unlock_live_parent(tmp_path):
    lock = zrs.StoreLock(tmp_path / 'LOCK')
    lock.acquire()
    pid = os.fork()
    if pid == 0:
        try:
            assert lock.release() is False
            other = zrs.StoreLock(tmp_path / 'LOCK')
            try:
                other.acquire()
            except zrs.StoreLockError:
                os._exit(0)
            os._exit(2)
        except BaseException:
            os._exit(3)
    try:
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert lock.held
        competitor = zrs.StoreLock(tmp_path / 'LOCK')
        with pytest.raises(zrs.StoreLockError):
            competitor.acquire()
        assert zrs.StoreLock.peek_owner(tmp_path)['pid'] == os.getpid()
    finally:
        lock.release()
