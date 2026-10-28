"""ZM-ZEGRID-RESILIENT-RUN-RESUME M5-C — portable member flush (Windows/NTFS).

The defect addressed here is documented Win32 semantics, not an observed NTFS
failure: ``os.fsync`` is ``FlushFileBuffers``, which requires a handle with
write access. ``_flush_member`` must therefore try a write-capable handle first
(``O_RDWR``, never ``O_TRUNC``/``O_CREAT``) and fall back to ``O_RDONLY`` only
where that is supported, instead of skipping the flush or aborting publication
on Windows.

Everything below runs on Linux and is injection-based:

* the simulated Win32 kernel refuses read-only handles on the flush path and
  rejects ``fsync`` on a read-only fd;
* failure of BOTH attempts is proven to abort BEFORE ``publication.json`` and
  BEFORE ``CURRENT.json``, keeping the previous generation authoritative;
* the F4 order (all members -> generation dir -> ancestors -> record ->
  pointer) is proven by instrumentation, not by the presence of calls;
* member bytes/digests and the generation file set are unchanged by the flush;
* ``run_store.fsync_dir`` stays a documented no-op under simulated Windows.

Real Windows/NTFS execution is NOT_RUN on this host.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
TESTS = Path(__file__).resolve().parents[1] / "tests"
for _p in (str(SRC), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import test_zegrid_publication_m5b as m5b  # noqa: E402


class _FdTrace:
    """fd -> (access mode, path) for every ``os.open`` during a test window."""

    def __init__(self):
        self.real_open = os.open
        self.real_fsync = os.fsync
        self.access: dict[int, int] = {}
        self.path: dict[int, str] = {}

    def open(self, path, flags, *args, **kwargs):
        fd = self.real_open(path, flags, *args, **kwargs)
        self.access[fd] = int(flags) & os.O_ACCMODE
        self.path[fd] = str(path)
        return fd




# ---------------------------------------------------------------------------
# a) simulated Win32: write path must be attempted and succeed
# ---------------------------------------------------------------------------


def _install_win32_sim(monkeypatch):
    trace = _FdTrace()
    flushes: list[tuple[str, int]] = []
    read_fsync_attempts: list[str] = []

    def _in_flush_member() -> bool:
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "_flush_member":
                return True
            frame = frame.f_back
        return False

    def _open(path, flags, *args, **kwargs):
        access = int(flags) & os.O_ACCMODE
        # Win32: a handle without write access cannot be flushed, so the
        # simulated kernel refuses to hand one out on the flush path.
        if access == os.O_RDONLY and _in_flush_member():
            raise PermissionError(
                errno.EACCES,
                "simulated Win32: read-only handle cannot be flushed",
                str(path),
            )
        fd = trace.real_open(path, flags, *args, **kwargs)
        trace.access[fd] = access
        trace.path[fd] = str(path)
        return fd

    def _fsync(fd):
        access = trace.access.get(fd)
        path = trace.path.get(fd, "")
        if access == os.O_RDONLY:
            read_fsync_attempts.append(path)
            raise OSError(
                errno.EACCES,
                "simulated Win32 FlushFileBuffers on a read-only handle",
                path,
            )
        if access in (os.O_WRONLY, os.O_RDWR):
            flushes.append((path, access))
        return trace.real_fsync(fd)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "fsync", _fsync)
    # A faithful Win32 run also has run_store's documented directory no-op.
    monkeypatch.setattr(zrs, "_IS_WINDOWS", True)
    return trace, flushes, read_fsync_attempts


def test_m5c_simulated_win32_flushes_every_member_via_write_handle(
    tmp_path, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    pub = m5b._begin(out)
    m5b._stage_generation(out, pub)
    expected = {
        str(pub.generation_dir / filename) for filename in pub.members.values()
    }

    trace, flushes, read_fsync_attempts = _install_win32_sim(monkeypatch)
    gen = pub.commit(required_roles=m5b._required_roles())

    flushed_members = {
        path for path, access in flushes if path in expected and access == os.O_RDWR
    }
    assert flushed_members == expected, (flushed_members, expected)
    assert len(
        [path for path, access in flushes if path in expected and access == os.O_RDWR]
    ) == len(expected)
    # No read-only flush was attempted: the write path carried the whole flush.
    assert read_fsync_attempts == []
    assert gen.pointer_path.is_file()
    assert zpub.resolve_current(out, m5b.RUN_ID).generation == gen.generation

    # The simulation is real: a read-only handle genuinely cannot be flushed.
    member = str(pub.generation_dir / pub.members[zpub.ROLE_SCIENCE])
    fd = trace.real_open(member, os.O_RDONLY)
    try:
        trace.access[fd] = os.O_RDONLY
        trace.path[fd] = member
        with pytest.raises(OSError):
            os.fsync(fd)
    finally:
        os.close(fd)


def test_m5c_write_attempt_precedes_read_fallback(tmp_path, monkeypatch):
    """Fallback is reachable and ordered: write first, read only on failure."""
    out = tmp_path / "out"
    out.mkdir()
    pub = m5b._begin(out)
    m5b._stage_generation(out, pub)
    trace = _FdTrace()
    attempts: list[int] = []
    real_fsync = trace.real_fsync

    def _open(path, flags, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_flush_member":
            attempts.append(int(flags) & os.O_ACCMODE)
        fd = trace.real_open(path, flags, *args, **kwargs)
        trace.access[fd] = int(flags) & os.O_ACCMODE
        trace.path[fd] = str(path)
        return fd

    def _fsync(fd):
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "fsync", _fsync)
    pub.commit(required_roles=m5b._required_roles())

    # One write attempt per member, no read attempt on the nominal POSIX path.
    assert attempts
    assert set(attempts) == {os.O_RDWR}
    assert attempts.count(os.O_RDWR) == len(pub.members)


# ---------------------------------------------------------------------------
# b) both attempts fail: explicit error, nothing published, CURRENT intact
# ---------------------------------------------------------------------------


def test_m5c_total_member_flush_failure_writes_no_pointer(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = m5b._publish(out)
    pointer_before = gen1.pointer_path.read_bytes()
    publication_before = gen1.publication_path.read_bytes()

    pub2 = m5b._begin(out)
    m5b._stage_generation(out, pub2)
    member_paths = {
        str(pub2.generation_dir / filename) for filename in pub2.members.values()
    }
    trace = _FdTrace()
    attempts: list[tuple[str, int]] = []

    def _open(path, flags, *args, **kwargs):
        fd = trace.real_open(path, flags, *args, **kwargs)
        trace.access[fd] = int(flags) & os.O_ACCMODE
        trace.path[fd] = str(path)
        return fd

    def _fsync(fd):
        path = trace.path.get(fd, "")
        access = trace.access.get(fd)
        if path in member_paths:
            attempts.append((path, access))
            raise OSError(errno.EIO, "simulated total member flush failure", path)
        return trace.real_fsync(fd)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "fsync", _fsync)

    with pytest.raises(zpub.PublicationError, match="no pointer was written") as excinfo:
        pub2.commit(required_roles=m5b._required_roles())

    # Both the write and the read-only fallback were attempted for the FIRST
    # member before giving up: the flush is never silently skipped.
    assert [access for _path, access in attempts[:2]] == [os.O_RDWR, os.O_RDONLY]
    assert "member flush failed on both" in str(excinfo.value)

    assert gen1.pointer_path.read_bytes() == pointer_before
    assert gen1.publication_path.read_bytes() == publication_before
    assert not (pub2.generation_dir / zpub.PUBLICATION_FILENAME).exists()
    resolved = zpub.resolve_current(out, m5b.RUN_ID)
    assert resolved.generation == gen1.generation
    assert resolved.publication_sha256 == gen1.publication_sha256


# ---------------------------------------------------------------------------
# c) F4 order on the nominal path, proven by instrumentation
# ---------------------------------------------------------------------------


def test_m5c_f4_order_members_dir_ancestors_record_pointer(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    pub = m5b._begin(out)
    m5b._stage_generation(out, pub)
    expected_members = {
        str(pub.generation_dir / filename) for filename in pub.members.values()
    }
    events: list[tuple[str, str]] = []
    trace = _FdTrace()
    real_replace = os.replace

    def _open(path, flags, *args, **kwargs):
        fd = trace.real_open(path, flags, *args, **kwargs)
        trace.access[fd] = int(flags) & os.O_ACCMODE
        trace.path[fd] = str(path)
        return fd

    def _fsync(fd):
        events.append(("fsync", trace.path.get(fd, "")))
        return trace.real_fsync(fd)

    def _replace(src, dst):
        events.append(("replace", str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "open", _open)
    monkeypatch.setattr(os, "fsync", _fsync)
    monkeypatch.setattr(os, "replace", _replace)

    pub.commit(required_roles=m5b._required_roles())

    gen_dir = str(pub.generation_dir)
    run_root = str(pub.run_root)
    member_idx = [
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path in expected_members
    ]
    dir_idx = [
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path == gen_dir
    ]
    root_idx = [
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path == run_root
    ]
    record_idx = [
        i for i, (kind, path) in enumerate(events)
        if kind == "replace" and path.endswith(zpub.PUBLICATION_FILENAME)
    ]
    pointer_idx = [
        i for i, (kind, path) in enumerate(events)
        if kind == "replace" and path.endswith(zpub.POINTER_FILENAME)
    ]

    # Every member is flushed; the count proves no member was skipped.
    assert sorted(set(events[i][1] for i in member_idx)) == sorted(expected_members)
    assert len(member_idx) == len(expected_members)
    assert dir_idx and root_idx and record_idx and pointer_idx

    last_member = max(member_idx)
    first_dir = min(i for i in dir_idx if i > last_member)
    record = min(record_idx)
    pointer = min(pointer_idx)
    # Ancestor flushes that belong to the pre-record phase (the pointer's own
    # directory fsync happens later, after the record, and is excluded here).
    pre_record_root = [i for i in root_idx if i < record]
    assert pre_record_root
    assert last_member < first_dir
    assert last_member < min(pre_record_root) < record
    assert first_dir < record
    assert record < pointer
    # The record is written before the pointer that names it.
    assert all(i < pointer for i in record_idx)


# ---------------------------------------------------------------------------
# d) the flush mutates nothing: bytes, digests, generation file set
# ---------------------------------------------------------------------------


def test_m5c_flush_preserves_member_bytes_and_generation_file_set(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pub = m5b._begin(out)
    m5b._stage_generation(out, pub)
    gen_dir = pub.generation_dir
    before = {
        filename: (gen_dir / filename).read_bytes()
        for filename in sorted(set(pub.members.values()))
    }
    before_stats = {
        filename: (gen_dir / filename).stat().st_size
        for filename in before
    }

    gen = pub.commit(required_roles=m5b._required_roles())

    after_names = {path.name for path in gen_dir.iterdir() if path.is_file()}
    expected_names = set(pub.members.values()) | {zpub.PUBLICATION_FILENAME}
    assert after_names == expected_names
    assert zpub.STAGING_FILENAME not in after_names
    # No extra file anywhere in the generation directory.
    assert {p.relative_to(gen_dir) for p in gen_dir.rglob("*") if p.is_file()} == {
        Path(name) for name in expected_names
    }
    for name, data in before.items():
        assert (gen_dir / name).read_bytes() == data, name
        assert (gen_dir / name).stat().st_size == before_stats[name], name

    # Record digests describe exactly the bytes on disk after the flush.
    record = json.loads(gen.publication_path.read_text(encoding="utf-8"))
    for role, row in record["members"].items():
        path = gen_dir / row["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"], role
        assert path.stat().st_size == row["nbytes"], role
    pointer = json.loads(gen.pointer_path.read_text(encoding="utf-8"))
    assert pointer["publication_sha256"] == hashlib.sha256(
        gen.publication_path.read_bytes()
    ).hexdigest()


# ---------------------------------------------------------------------------
# e) run_store.fsync_dir stays a documented no-op under simulated Windows
# ---------------------------------------------------------------------------


def test_m5c_fsync_dir_is_noop_under_simulated_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(zrs, "_IS_WINDOWS", True)

    def _must_not_open(*args, **kwargs):
        raise AssertionError("fsync_dir must not open a handle on Windows")

    def _must_not_fsync(*args, **kwargs):
        raise AssertionError("fsync_dir must not fsync on Windows")

    monkeypatch.setattr(os, "open", _must_not_open)
    monkeypatch.setattr(os, "fsync", _must_not_fsync)

    zrs.fsync_dir(tmp_path)
    zrs.fsync_dir(tmp_path / "does-not-exist")
