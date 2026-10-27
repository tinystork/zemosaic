"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 — durable per-cell run store.

This module is the M2 durability seam. It persists the accepted per-cell
payloads of ONE run to disk with an atomic, crash-recoverable publication
protocol, and it releases the corresponding large arrays from the in-memory
M1 registry once a commit is durable.

Scope (plan §6.1–§6.7). M2 owns:

* a frozen RUN IDENTITY: schema/version, input inventory with CONTENT hashes,
  canvas/WCS, grid/geometry, halo, memberships, the complete and atomic gauge
  result, the science contract and the dtype contract. RAM/workers/prefetch
  live in a separate ``execution_context`` block that NEVER participates in
  compatibility;
* per-cell commits: identity, status, bounds/core_slice, contributors, the
  real plan set (science core planes + per-cell pre-offset/pre-feather patch),
  the per-cell photometric reference pin, backend, effective parameters and the
  declared dtype/shape/byte-size/digest of every stored array;
* JSON-ONLY manifests and separate array files with declared dtype/shape.
  No pickle, no content execution, no object dtype. Every path read from a
  manifest is confined to the run directory (traversal refused);
* the mandatory transaction order: reserve space -> write to an attempt-private
  temporary file -> flush+fsync -> verify size/shape/digest -> publish the cell
  commit atomically (rename on the same volume) -> update the global index
  atomically. The index is RECONSTRUCTIBLE from the cell commits: an
  interruption between publication and index update loses nothing;
* a single-writer OS lock whose owner identity is (instance, pid, process start
  token, boot id, host) — never the pid alone. A dead owner's lock is
  recoverable; a live process's lock is never stolen;
* sha256 integrity of array bytes and metadata, verified on every open/load.
  A corrupted cell is invalidated ALONE (no guessing, no automatic repair);
  a corrupted identity/index is reported, never "repaired by guess";
* a disk budget that separates disposable caches, durable cell commits,
  assembly space and a transactional margin, with a free-space preflight and a
  real reservation where the OS offers one. Disk-full preserves prior commits,
  never publishes a partial cell, never deletes checkpoints or user results;
* sparse cell sets: completeness is ``expected == COMPLETE + EMPTY_VALIDATED``
  over the full cell_id set, never a ``0..K`` prefix;
* idempotent publication by ``(cell_id, attempt_id)``: a lost receipt is found
  by transaction lookup, never by recomputation. Two different contents for
  the same validated identity are refused.

Explicit non-claims: this module does NOT implement run restore/resume (M4),
resource wait/backoff or generations (M3), bounded assembly or offsets (M5),
GUI/CLI (M6) or GPU science qualification (M7). It never mutates the science,
the budgets, the floors or the final publication. It never deletes user data
and never retries forever.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import run_registry as zrr

STORE_SCHEMA = "zemosaic-zegrid-run-store-v1"
IDENTITY_SCHEMA = "zemosaic-zegrid-run-identity-v1"
COMMIT_SCHEMA = "zemosaic-zegrid-cell-commit-v1"
INDEX_SCHEMA = "zemosaic-zegrid-run-index-v1"
LOCK_SCHEMA = "zemosaic-zegrid-run-store-lock-v1"
STORE_VERSION = "1"

STATUS_COMPLETE = zrr.STATUS_COMPLETE
STATUS_EMPTY_VALIDATED = zrr.STATUS_EMPTY_VALIDATED

#: Producer status tokens accepted by the store (same contract as M1).
_CELL_STATUS_MAP = {"complete": STATUS_COMPLETE, "empty": STATUS_EMPTY_VALIDATED}

#: Array dtypes the store accepts. Deliberately narrow: no object dtype (no
#: code execution), no complex/float16 (no silent precision contract change),
#: native byte order only.
ALLOWED_DTYPES = frozenset(
    {
        "|b1",
        "|u1",
        "|i1",
        "<u2",
        ">u2",
        "<i2",
        ">i2",
        "<u4",
        ">u4",
        "<i4",
        ">i4",
        "<u8",
        ">u8",
        "<i8",
        ">i8",
        "<f4",
        ">f4",
        "<f8",
        ">f8",
    }
)

#: Identity sections that participate in compatibility. ``execution_context``
#: is deliberately absent: RAM/workers/prefetch must never invalidate science.
IDENTITY_SECTIONS = (
    "version",
    "inputs",
    "canvas",
    "grid",
    "halo",
    "memberships",
    "gauge",
    "science",
    "dtypes",
    "backend",
)

DEFAULT_TRANSACTIONAL_MARGIN_BYTES = 256 * 2 ** 20
MIN_TRANSACTIONAL_MARGIN_BYTES = 16 * 2 ** 20

_SAFE_NAME_RE = None
try:  # pragma: no cover - trivial
    import re as _re

    _SAFE_NAME_RE = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
except Exception:  # pragma: no cover
    _SAFE_NAME_RE = None


class RunStoreError(RuntimeError):
    """Base class for every durable-store contract failure."""


class StoreFormatError(RunStoreError):
    """The manifest/index/identity does not satisfy the strict schema."""


class StorePathError(StoreFormatError):
    """A manifest path escapes the run directory or is not a safe name."""


class StoreLockError(RunStoreError):
    """The single-writer lock is held by a live owner (or cannot be taken).

    ``attempts`` and ``last_error`` are populated when a bounded re-acquire loop
    ran for a verifiably DEAD owner (Windows releases byte-range locks slightly
    after process termination; POSIX ``flock`` does not).
    """

    def __init__(self, message, *, attempts=None, last_error=None):
        super().__init__(message)
        self.attempts = None if attempts is None else int(attempts)
        self.last_error = last_error


class StoreIntegrityError(RunStoreError):
    """Stored bytes/metadata do not match their declared digest."""


class StoreDiskFullError(RunStoreError):
    """The disk budget/margin cannot admit this transaction."""

    def __init__(self, message: str, *, required_bytes: int, free_bytes: int | None,
                 budget_bytes: int | None = None, used_bytes: int | None = None):
        super().__init__(message)
        self.required_bytes = int(required_bytes)
        self.free_bytes = None if free_bytes is None else int(free_bytes)
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.used_bytes = None if used_bytes is None else int(used_bytes)


class StoreIdentityMismatch(RunStoreError):
    """The run identity is incompatible with the stored run. Never resumed."""

    def __init__(self, message: str, *, reasons: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.reasons = [dict(r) for r in reasons]


class StoreConflictError(RunStoreError):
    """A different payload was offered for an already published identity."""


class StoreClosedError(RunStoreError):
    """The store was closed; it accepts nothing further."""


class StoreNotFoundError(RunStoreError):
    """No durable commit exists for the requested cell/attempt."""


# ---------------------------------------------------------------------------
# JSON / strict schema helpers
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """Canonical ASCII JSON bytes. NaN/Inf are rejected (strict JSON)."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise StoreFormatError(f"value is not strict JSON: {exc}") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_mapping(
    value: Any, label: str, required: Sequence[str], optional: Sequence[str] = ()
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StoreFormatError(
            f"{label} must be a JSON object, got {type(value).__name__}"
        )
    keys = set(value.keys())
    if any(not isinstance(k, str) for k in keys):
        raise StoreFormatError(f"{label} has a non-string key")
    missing = [k for k in required if k not in keys]
    unknown = sorted(keys - set(required) - set(optional))
    if missing:
        raise StoreFormatError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise StoreFormatError(
            f"{label} has unknown fields (strict schema): {', '.join(unknown)}"
        )
    return value


def _req_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StoreFormatError(f"{label} must be a non-empty string")
    return value


def _req_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreFormatError(f"{label} must be a strict int, got {value!r}")
    if minimum is not None and value < minimum:
        raise StoreFormatError(f"{label} must be >= {minimum}, got {value}")
    return int(value)


def _req_seq(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise StoreFormatError(f"{label} must be a JSON array")
    return value


def _req_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise StoreFormatError(f"{label} must be a boolean, got {value!r}")
    return value


def json_safe(value: Any, label: str = "value") -> Any:
    """Convert a payload metadata tree to strict JSON or fail closed."""
    import numpy as np

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StoreFormatError(f"{label} contains a non-finite float")
        return float(value)
    if isinstance(value, np.generic):
        return json_safe(value.item(), label)
    if isinstance(value, os.PathLike):
        return str(os.fspath(value))
    if isinstance(value, (list, tuple)):
        return [json_safe(v, f"{label}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StoreFormatError(f"{label} has a non-string key {key!r}")
            out[key] = json_safe(item, f"{label}.{key}")
        return out
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise StoreFormatError(f"{label} holds raw bytes (not JSON storable)")
    raise StoreFormatError(
        f"{label} type {type(value).__name__} is not JSON storable"
    )


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------


def safe_name(value: Any, label: str) -> str:
    """Accept only a single, plain path component (no separators, no '..')."""
    text = _req_str(value, label)
    if _SAFE_NAME_RE is None:  # pragma: no cover - defensive
        raise StorePathError("name validator unavailable")
    if ".." in text or "/" in text or "\\" in text or "\x00" in text:
        raise StorePathError(f"{label} is not a safe path component: {text!r}")
    if not _SAFE_NAME_RE.match(text):
        raise StorePathError(
            f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]*, got {text!r}"
        )
    return text


def confined_path(root: Path, relative: Any, label: str = "path") -> Path:
    """Resolve a manifest-declared path INSIDE ``root`` or refuse.

    Rejects absolute paths, drive letters, UNC paths, ``..`` traversal, NUL
    bytes, empty components and symlinked components that escape the root.
    """
    text = _req_str(relative, label)
    if "\x00" in text:
        raise StorePathError(f"{label} contains a NUL byte")
    normalised = text.replace("\\", "/")
    if normalised.startswith("/") or (len(normalised) > 1 and normalised[1] == ":"):
        raise StorePathError(f"{label} must be relative, got {text!r}")
    parts = [p for p in normalised.split("/")]
    if any(p in ("", ".", "..") for p in parts):
        raise StorePathError(f"{label} has an unsafe component: {text!r}")
    candidate = Path(root)
    for part in parts:
        candidate = candidate / part
    try:
        resolved_root = Path(root).resolve(strict=False)
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(resolved_root)
    except (ValueError, OSError) as exc:
        raise StorePathError(
            f"{label} escapes the run directory: {text!r} ({exc})"
        ) from exc
    # A symlink anywhere on the declared path could redirect the read.
    probe = Path(root)
    for part in parts[:-1]:
        probe = probe / part
        if probe.is_symlink():
            raise StorePathError(f"{label} traverses a symlink: {text!r}")
    if candidate.is_symlink():
        raise StorePathError(f"{label} is a symlink: {text!r}")
    return candidate


# ---------------------------------------------------------------------------
# OS primitives: fsync, atomic write, space reservation, locking
# ---------------------------------------------------------------------------

_IS_WINDOWS = os.name == "nt"


def binary_flag() -> int:
    """The platform's "binary mode" open flag (0 on POSIX).

    Read at CALL time (never cached at import) so a Windows regression test can
    inject a sentinel and prove that every ``os.open`` in this module ORs it.
    """
    return int(getattr(os, "O_BINARY", 0))


def _open_binary(path, flags: int, mode: int = 0o600) -> int:
    """``os.open`` with binary mode FORCED (both read and write paths).

    Windows translates CRLF<->LF for descriptors opened without ``O_BINARY``.
    That silently changes every stored byte pattern containing 0x0D/0x0A and
    corrupts the sha256 of an array object. There is no text-mode open in this
    module: manifests are bytes, arrays are bytes, the lock record is bytes.
    """
    return os.open(str(path), int(flags) | binary_flag(), int(mode))


def fsync_file(fd: int) -> None:
    """Flush one file handle to the OS durable layer."""
    os.fsync(fd)


def fsync_dir(path: Path) -> None:
    """Best-effort directory fsync (POSIX). Windows/NTFS: no-op by design."""
    if _IS_WINDOWS:
        return
    flag = getattr(os, "O_DIRECTORY", 0)
    try:
        fd = _open_binary(path, os.O_RDONLY | flag, 0)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _reserve_space(fd: int, nbytes: int) -> str:
    """Reserve ``nbytes`` in an open file; return the reservation mode.

    POSIX uses ``posix_fallocate`` (a real allocation). Windows uses
    ``SetFileInformationByHandle(FileAllocationInfo)``. When neither is
    available, the file is extended (documented as a weaker guarantee).
    """
    if nbytes <= 0:
        return "empty"
    if hasattr(os, "posix_fallocate") and not _IS_WINDOWS:
        try:
            os.posix_fallocate(fd, 0, int(nbytes))
            return "posix_fallocate"
        except OSError as exc:
            if exc.errno in (getattr(os, "ENOSPC", 28),):
                raise
    if _IS_WINDOWS:
        try:
            import ctypes
            import ctypes.wintypes as wt

            class _FileAllocationInfo(ctypes.Structure):
                _fields_ = [("FileAllocationInfo", ctypes.c_ulonglong)]

            info = _FileAllocationInfo(int(nbytes))
            handle = _get_osfhandle(fd)
            ok = ctypes.windll.kernel32.SetFileInformationByHandle(
                wt.HANDLE(handle), 5, ctypes.byref(info), ctypes.sizeof(info)
            )
            if ok:
                return "windows_file_allocation"
        except Exception:
            pass
    os.ftruncate(fd, int(nbytes))
    return "ftruncate"


def _get_osfhandle(fd: int) -> int:
    import msvcrt  # noqa: F401  (only valid on Windows)

    return msvcrt.get_osfhandle(fd)


def atomic_write_bytes(path: Path, data: bytes, *, tmp_dir: Path) -> None:
    """Write ``data`` to ``path`` atomically (tmp -> fsync -> rename -> dir fsync)."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    fd = _open_binary(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:  # pragma: no cover - defensive
                raise OSError("short write")
            view = view[written:]
        fsync_file(fd)
    finally:
        os.close(fd)
    try:
        os.replace(str(tmp), str(path))
    except BaseException:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    fsync_dir(Path(path).parent)


# -- process / owner identity ------------------------------------------------

_BOOT_ID_CACHE: dict[str, Any] = {"value": None, "loaded": False}


def boot_id() -> str | None:
    if _BOOT_ID_CACHE["loaded"]:
        return _BOOT_ID_CACHE["value"]
    _BOOT_ID_CACHE["loaded"] = True
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        _BOOT_ID_CACHE["value"] = value or None
    except Exception:
        _BOOT_ID_CACHE["value"] = None
    return _BOOT_ID_CACHE["value"]


def process_start_token(pid: int) -> str | None:
    """Return an opaque process start token, or None when unavailable.

    Linux: field 22 of ``/proc/<pid>/stat`` (starttime ticks). Windows:
    ``GetProcessTimes`` creation FILETIME. The token is what makes PID reuse
    detectable: a recycled pid has a different start token.
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if _IS_WINDOWS:
        try:
            import ctypes
            import ctypes.wintypes as wt

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return None
            try:
                created = wt.FILETIME()
                exited = wt.FILETIME()
                kernel = wt.FILETIME()
                user = wt.FILETIME()
                ok = ctypes.windll.kernel32.GetProcessTimes(
                    wt.HANDLE(handle),
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                )
                if not ok:
                    return None
                return f"{int(created.dwHighDateTime)}:{int(created.dwLowDateTime)}"
            finally:
                ctypes.windll.kernel32.CloseHandle(wt.HANDLE(handle))
        except Exception:
            return None
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = raw[raw.rindex(")") + 2:].split()
        return str(tail[19])
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if _IS_WINDOWS:
        try:
            import ctypes
            import ctypes.wintypes as wt

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            try:
                code = wt.DWORD()
                ok = ctypes.windll.kernel32.GetExitCodeProcess(
                    wt.HANDLE(handle), ctypes.byref(code)
                )
                return bool(ok) and int(code.value) == 259  # STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(wt.HANDLE(handle))
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def owner_is_live(owner: Mapping[str, Any]) -> bool:
    """True only when the recorded owner is provably the SAME live process."""
    try:
        host = owner.get("host")
        pid = int(owner.get("pid"))
    except (TypeError, ValueError):
        return False
    if host and str(host) != socket.gethostname():
        # Cannot prove liveness across hosts; conservative (never steal).
        return True
    recorded_boot = owner.get("boot_id")
    if recorded_boot and boot_id() and str(recorded_boot) != str(boot_id()):
        return False  # the machine rebooted: the owner cannot be alive
    if not pid_alive(pid):
        return False
    recorded_start = owner.get("process_start_token")
    if recorded_start:
        current = process_start_token(pid)
        if current is not None and str(current) != str(recorded_start):
            return False  # pid reused
    return True


_LIVE_LOCKS: dict[str, Any] = {}
_LIVE_LOCKS_GUARD = threading.Lock()

#: Bounded re-acquire window, used ONLY when the recorded owner is provably dead
#: (never for a live owner). Windows can release the byte-range lock a little
#: after the process is gone; POSIX flock releases it synchronously.
LOCK_REACQUIRE_TIMEOUT_S = 2.0
LOCK_REACQUIRE_INTERVAL_S = 0.02
#: Byte-exact placeholder written to an empty lock file before byte-range
#: locking (with O_BINARY, so no CRLF translation can ever apply).
LOCK_PLACEHOLDER = b"{}\n"


class StoreLock:
    """Single-writer OS lock with an explicit owner identity.

    The OS lock (``flock`` on POSIX, ``msvcrt.locking`` on Windows) is the
    authority: it is released by the kernel when the owner dies, so an orphan is
    recoverable. The JSON owner record is diagnostic and is used to detect PID
    reuse and to refuse a lock held by a live process.
    """

    def __init__(self, path: Path, *, session_id: str | None = None,
                 instance_id: str | None = None):
        self.path = Path(path)
        self.session_id = session_id
        self.instance_id = instance_id or uuid.uuid4().hex
        self._fd: int | None = None
        self._key = str(self.path.resolve(strict=False))
        self.owner: dict[str, Any] | None = None
        self.previous_owner: dict[str, Any] | None = None
        self.orphan_recovered = False
        self.acquired_unix_ns: int | None = None
        self.last_acquire_attempts = 0
        self.last_acquire_wait_s = 0.0
        self.reservation_mode = "os_lock"

    # -- OS primitive ---------------------------------------------------
    def _lock_fd(self, fd: int) -> None:
        if _IS_WINDOWS:
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise StoreLockError(
                    f"lock {self.path} is held by another writer ({exc})"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise StoreLockError(
                    f"lock {self.path} is held by another writer ({exc})"
                ) from exc

    def _unlock_fd(self, fd: int) -> None:
        try:
            if _IS_WINDOWS:
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass

    # -- owner record ---------------------------------------------------
    def _read_owner_from_fd(self, fd: int) -> dict[str, Any] | None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 65536)
        except OSError:
            return None
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _owner_detail(previous: Mapping[str, Any] | None) -> str:
        if previous is None:
            return "no owner record; the OS lock is held"
        return (
            f"owner pid={previous.get('pid')} "
            f"started={previous.get('started_unix_ns')} "
            f"session={previous.get('session_id')} "
            f"state={previous.get('state', 'held')}"
        )

    @staticmethod
    def _provably_dead(previous: Mapping[str, Any] | None) -> bool:
        """True only for a well-formed HELD record whose process is gone.

        A missing record, a released record or a live process is never treated
        as provably dead — so the bounded wait can never be used to steal (or
        wait out) a live owner.
        """
        if previous is None:
            return False
        if str(previous.get("state", "held")) != "held":
            return False
        return not owner_is_live(previous)

    def acquire(self) -> dict[str, Any]:
        with _LIVE_LOCKS_GUARD:
            if self._key in _LIVE_LOCKS:
                raise StoreLockError(
                    f"store lock {self.path} is already held by this process "
                    f"(instance {_LIVE_LOCKS[self._key].instance_id})"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = _open_binary(self.path, os.O_RDWR | os.O_CREAT)
            if os.fstat(fd).st_size == 0:
                os.write(fd, LOCK_PLACEHOLDER)
                os.lseek(fd, 0, os.SEEK_SET)
            started_monotonic = time.monotonic()
            attempts = 0
            last_error: StoreLockError | None = None
            while True:
                attempts += 1
                try:
                    self._lock_fd(fd)
                    break
                except StoreLockError as exc:
                    last_error = exc
                    previous = self._read_owner_from_fd(fd)
                    if not self._provably_dead(previous):
                        waited = time.monotonic() - started_monotonic
                        os.close(fd)
                        raise StoreLockError(
                            "refusing to write: single-writer lock "
                            f"{self.path} is busy "
                            f"({self._owner_detail(previous)}); "
                            f"{attempts} attempt(s) in {waited:.3f}s "
                            "(no retry: the owner is live or unproven)",
                            attempts=attempts,
                            last_error=str(exc),
                        ) from exc
                    remaining = (
                        float(LOCK_REACQUIRE_TIMEOUT_S) - (
                            time.monotonic() - started_monotonic
                        )
                    )
                    if remaining <= 0:
                        waited = time.monotonic() - started_monotonic
                        os.close(fd)
                        raise StoreLockError(
                            "refusing to write: single-writer lock "
                            f"{self.path} is still held after {attempts} "
                            f"attempt(s) in {waited:.3f}s while the owner is "
                            f"verifiably dead ({self._owner_detail(previous)}); "
                            f"last error: {exc}",
                            attempts=attempts,
                            last_error=str(exc),
                        ) from exc
                    time.sleep(
                        min(float(LOCK_REACQUIRE_INTERVAL_S), remaining)
                    )
            self.last_acquire_attempts = int(attempts)
            self.last_acquire_wait_s = time.monotonic() - started_monotonic
            previous = self._read_owner_from_fd(fd)
            previous_state = (
                str(previous.get("state", "held")) if previous else None
            )
            if (
                previous is not None
                and previous_state != "released"
                and owner_is_live(previous)
            ):
                same_process = (
                    int(previous.get("pid", -1)) == os.getpid()
                    and previous.get("process_start_token")
                    == process_start_token(os.getpid())
                )
                if not same_process:
                    # A live owner without an OS lock is contradictory: refuse
                    # rather than steal. (The OS lock was free, so this is a
                    # foreign/stale record for a process we cannot prove dead.)
                    self._unlock_fd(fd)
                    os.close(fd)
                    raise StoreLockError(
                        "refusing to write: lock record names a live owner "
                        f"(pid={previous.get('pid')}) but no OS lock is held; "
                        "manual diagnosis required"
                    )
            self._fd = fd
            self.previous_owner = dict(previous) if previous else None
            self.orphan_recovered = bool(
                previous is not None
                and previous_state == "held"
                and not owner_is_live(previous)
            )
            self.acquired_unix_ns = int(time.time_ns())
            self.owner = {
                "schema": LOCK_SCHEMA,
                "state": "held",
                "instance_id": str(self.instance_id),
                "pid": int(os.getpid()),
                "process_start_token": process_start_token(os.getpid()),
                "boot_id": boot_id(),
                "host": socket.gethostname(),
                "session_id": self.session_id,
                "started_unix_ns": int(self.acquired_unix_ns),
                "platform": platform.platform(),
            }
            payload = canonical_json(self.owner)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:  # pragma: no cover
                    raise OSError("short lock write")
                view = view[written:]
            fsync_file(fd)
            _LIVE_LOCKS[self._key] = self
            return {
                "path": str(self.path),
                "owner": dict(self.owner),
                "orphan_recovered": bool(self.orphan_recovered),
                "previous_owner": (
                    dict(self.previous_owner) if self.previous_owner else None
                ),
                "acquire_attempts": int(self.last_acquire_attempts),
                "acquire_waited_s": round(float(self.last_acquire_wait_s), 6),
            }

    def release(self) -> bool:
        with _LIVE_LOCKS_GUARD:
            fd = self._fd
            if fd is None:
                return False
            self._fd = None
            _LIVE_LOCKS.pop(self._key, None)
        # Mark the record as cleanly released while the OS lock is still held,
        # so a later writer never confuses a finished owner with a live one.
        try:
            record = dict(self.owner or {})
            record["state"] = "released"
            record["released_unix_ns"] = int(time.time_ns())
            payload = canonical_json(record)
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, payload)
            fsync_file(fd)
        except Exception:
            pass
        try:
            self._unlock_fd(fd)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
        return True

    @property
    def held(self) -> bool:
        return self._fd is not None


# ---------------------------------------------------------------------------
# Disk budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskBudget:
    """Separate disk lines: durable cells, assembly, caches, transactional margin.

    ``durable_cell_bytes`` is the hard ceiling for the per-cell commits of this
    run. ``transactional_margin_bytes`` is free space kept untouched so a
    transaction can always be rolled back without touching prior commits.
    """

    durable_cell_bytes: int
    assembly_bytes: int = 0
    disposable_cache_bytes: int = 0
    transactional_margin_bytes: int = DEFAULT_TRANSACTIONAL_MARGIN_BYTES

    def __post_init__(self) -> None:
        for name in (
            "durable_cell_bytes",
            "assembly_bytes",
            "disposable_cache_bytes",
            "transactional_margin_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DiskBudget.{name} must be a non-negative int")
        if int(self.transactional_margin_bytes) < 0:
            raise ValueError("transactional margin must be >= 0")

    def to_dict(self) -> dict:
        return {
            "durable_cell_bytes": int(self.durable_cell_bytes),
            "assembly_bytes": int(self.assembly_bytes),
            "disposable_cache_bytes": int(self.disposable_cache_bytes),
            "transactional_margin_bytes": int(self.transactional_margin_bytes),
            "total_bytes": int(
                self.durable_cell_bytes
                + self.assembly_bytes
                + self.disposable_cache_bytes
                + self.transactional_margin_bytes
            ),
        }


def default_transactional_margin(durable_cell_bytes: int) -> int:
    """Margin that scales with the durable line but never disappears."""
    scaled = int(int(durable_cell_bytes) // 20)
    return int(max(MIN_TRANSACTIONAL_MARGIN_BYTES, scaled))


# ---------------------------------------------------------------------------
# Run identity (plan §6.1 / §6.3)
# ---------------------------------------------------------------------------

_IDENTITY_FIELDS = (
    "schema",
    "schema_version",
    "run_id",
    "created_unix_ns",
    "science_contract",
    "execution_context",
)


def validate_run_identity(identity: Mapping[str, Any]) -> dict:
    """Strictly validate a run identity and return a normalised copy."""
    _strict_mapping(identity, "run identity", _IDENTITY_FIELDS)
    if identity["schema"] != IDENTITY_SCHEMA:
        raise StoreFormatError(
            f"run identity schema is {identity['schema']!r}, expected "
            f"{IDENTITY_SCHEMA!r} (no force)"
        )
    if str(identity["schema_version"]) != STORE_VERSION:
        raise StoreFormatError(
            f"run identity schema_version {identity['schema_version']!r} is "
            f"not supported by this store ({STORE_VERSION})"
        )
    run_id = _req_str(identity["run_id"], "run identity.run_id")
    created = _req_int(identity["created_unix_ns"], "identity.created_unix_ns")
    contract = _strict_mapping(
        identity["science_contract"],
        "run identity.science_contract",
        (
            "contract_version",
            "inputs",
            "canvas",
            "grid",
            "halo",
            "memberships",
            "gauge",
            "science",
            "dtypes",
            "backend",
        ),
    )
    _req_str(contract["contract_version"], "science_contract.contract_version")
    inputs = _strict_mapping(
        contract["inputs"],
        "science_contract.inputs",
        ("aggregate_sha256", "frames"),
    )
    _req_str(inputs["aggregate_sha256"], "inputs.aggregate_sha256")
    _expected_aggregate = _inputs_aggregate(inputs["frames"])
    if str(inputs["aggregate_sha256"]) != _expected_aggregate:
        raise StoreFormatError(
            "inputs.aggregate_sha256 does not match the CONTENT rows of the "
            "inventory (must be relocation-independent)"
        )
    frames = _req_seq(inputs["frames"], "inputs.frames")
    seen_ids: set[str] = set()
    for index, frame in enumerate(frames):
        entry = _strict_mapping(
            frame,
            f"inputs.frames[{index}]",
            ("frame_id", "content_sha256", "nbytes", "order", "role"),
        )
        fid = _req_str(entry["frame_id"], f"frames[{index}].frame_id")
        if fid in seen_ids:
            raise StoreFormatError(f"inputs.frames has duplicate id {fid!r}")
        seen_ids.add(fid)
        _req_str(entry["content_sha256"], f"frames[{index}].content_sha256")
        _req_int(entry["nbytes"], f"frames[{index}].nbytes", minimum=0)
        _req_int(entry["order"], f"frames[{index}].order", minimum=0)
        _req_str(entry["role"], f"frames[{index}].role")
    canvas = _strict_mapping(
        contract["canvas"], "science_contract.canvas",
        ("shape", "wcs_sha256", "geometry_sha256"),
    )
    shape = _req_seq(canvas["shape"], "canvas.shape")
    if len(shape) != 2:
        raise StoreFormatError("canvas.shape must be [height, width]")
    _req_int(shape[0], "canvas.shape[0]", minimum=1)
    _req_int(shape[1], "canvas.shape[1]", minimum=1)
    _req_str(canvas["wcs_sha256"], "canvas.wcs_sha256")
    _req_str(canvas["geometry_sha256"], "canvas.geometry_sha256")
    grid = _strict_mapping(
        contract["grid"], "science_contract.grid",
        ("nx", "ny", "tile_px", "geometry_sha256", "cell_ids_sha256"),
    )
    _req_int(grid["nx"], "grid.nx", minimum=1)
    _req_int(grid["ny"], "grid.ny", minimum=1)
    _req_int(grid["tile_px"], "grid.tile_px", minimum=1)
    _req_str(grid["geometry_sha256"], "grid.geometry_sha256")
    _req_str(grid["cell_ids_sha256"], "grid.cell_ids_sha256")
    halo = _strict_mapping(
        contract["halo"], "science_contract.halo", ("halo_px",),
        optional=("provenance",),
    )
    _req_int(halo["halo_px"], "halo.halo_px", minimum=0)
    memberships = _strict_mapping(
        contract["memberships"], "science_contract.memberships",
        ("sha256", "n_cells"),
    )
    _req_str(memberships["sha256"], "memberships.sha256")
    _req_int(memberships["n_cells"], "memberships.n_cells", minimum=0)
    gauge = _strict_mapping(
        contract["gauge"], "science_contract.gauge",
        ("mode", "sha256"),
        optional=("reference_frame_id", "reference_frame_ids", "n_exclusions"),
    )
    if gauge["mode"] not in zrr.GAUGE_MODES:
        raise StoreFormatError(
            f"gauge.mode must be one of {zrr.GAUGE_MODES}, got {gauge['mode']!r}"
        )
    _req_str(gauge["sha256"], "gauge.sha256")
    if gauge.get("reference_frame_id") is not None:
        _req_str(gauge["reference_frame_id"], "gauge.reference_frame_id")
    if gauge.get("reference_frame_ids") is not None:
        ids = _req_seq(gauge["reference_frame_ids"], "gauge.reference_frame_ids")
        for index, value in enumerate(ids):
            _req_str(value, f"gauge.reference_frame_ids[{index}]")
    if gauge.get("n_exclusions") is not None:
        _req_int(gauge["n_exclusions"], "gauge.n_exclusions", minimum=0)
    science = _strict_mapping(
        contract["science"], "science_contract.science",
        ("sha256", "contract_version"),
        optional=("params",),
    )
    _req_str(science["sha256"], "science.sha256")
    _req_str(science["contract_version"], "science.contract_version")
    dtypes = _strict_mapping(
        contract["dtypes"], "science_contract.dtypes", ("planes",),
    )
    if not isinstance(dtypes["planes"], Mapping):
        raise StoreFormatError("science_contract.dtypes.planes must be an object")
    planes = _strict_mapping(
        dtypes["planes"], "dtypes.planes", tuple(sorted(dtypes["planes"].keys()))
    )
    if not planes:
        raise StoreFormatError("science_contract.dtypes.planes must not be empty")
    for name, dtype in planes.items():
        allowed = [dtype] if isinstance(dtype, str) else _req_seq(
            dtype, f"dtypes.planes.{name}"
        )
        if not allowed:
            raise StoreFormatError(f"dtypes.planes.{name} declares no dtype")
        for token in allowed:
            token = _req_str(token, f"dtypes.planes.{name} token")
            if token not in ALLOWED_DTYPES:
                raise StoreFormatError(
                    f"dtypes.planes.{name} declares unsupported dtype {token!r}"
                )
    backend = _strict_mapping(
        contract["backend"], "science_contract.backend",
        ("backend", "tile_px", "parity_qualified"),
    )
    _req_str(backend["backend"], "backend.backend")
    _req_int(backend["tile_px"], "backend.tile_px", minimum=1)
    _req_bool(backend["parity_qualified"], "backend.parity_qualified")
    context = _strict_mapping(
        identity["execution_context"],
        "run identity.execution_context",
        ("available_ram_bytes", "workers", "prefetch_units", "host", "note"),
        optional=("platform",),
    )
    for name in ("available_ram_bytes", "workers", "prefetch_units"):
        if context[name] is not None:
            _req_int(context[name], f"execution_context.{name}", minimum=0)
    _req_str(context["host"], "execution_context.host")
    _req_str(context["note"], "execution_context.note")
    normalised = json_safe(identity, "run identity")
    normalised["run_id"] = run_id
    normalised["created_unix_ns"] = int(created)
    return normalised


def build_run_identity(
    *,
    run_id: str,
    inputs: Mapping[str, Any],
    canvas: Mapping[str, Any],
    grid: Mapping[str, Any],
    halo: Mapping[str, Any],
    memberships: Mapping[str, Any],
    gauge: Mapping[str, Any],
    science: Mapping[str, Any],
    dtypes: Mapping[str, Any],
    backend: Mapping[str, Any],
    execution_context: Mapping[str, Any],
    contract_version: str = "1",
    created_unix_ns: int | None = None,
) -> dict:
    """Build and strictly validate a run identity (plan §6.1)."""
    identity = {
        "schema": IDENTITY_SCHEMA,
        "schema_version": STORE_VERSION,
        "run_id": str(run_id),
        "created_unix_ns": int(
            time.time_ns() if created_unix_ns is None else created_unix_ns
        ),
        "science_contract": {
            "contract_version": str(contract_version),
            "inputs": _normalise_inputs(inputs),
            "canvas": json_safe(dict(canvas), "canvas"),
            "grid": json_safe(dict(grid), "grid"),
            "halo": json_safe(dict(halo), "halo"),
            "memberships": json_safe(dict(memberships), "memberships"),
            "gauge": json_safe(dict(gauge), "gauge"),
            "science": json_safe(dict(science), "science"),
            "dtypes": json_safe(dict(dtypes), "dtypes"),
            "backend": json_safe(dict(backend), "backend"),
        },
        "execution_context": json_safe(
            dict(execution_context), "execution_context"
        ),
    }
    return validate_run_identity(identity)


def _normalise_inputs(inputs: Mapping[str, Any]) -> dict:
    """Normalise the inventory and DERIVE its content aggregate."""
    frames = json_safe(list(inputs.get("frames", [])), "inputs.frames")
    return {
        "aggregate_sha256": _inputs_aggregate(frames),
        "frames": frames,
    }


def content_sha256_file(path: str | os.PathLike[str]) -> tuple[str, int]:
    """Streaming content hash of one input file plus its size in bytes.

    Only CONTENT is identity: the path is never part of a fingerprint, so a
    moved dataset is not a different run.
    """
    return _file_sha256(Path(path)), int(Path(path).stat().st_size)


def _inputs_identity_rows(frames: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Content-only inventory rows: a dataset move is NOT a new identity."""
    return [
        [
            str(entry["content_sha256"]),
            int(entry["nbytes"]),
            int(entry["order"]),
            str(entry["role"]),
        ]
        for entry in frames
    ]


def _inputs_aggregate(frames: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(canonical_json(_inputs_identity_rows(frames)))


def identity_sections(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Split a validated identity into compatibility sections."""
    contract = identity["science_contract"]
    inputs = contract["inputs"]
    return {
        "version": {
            "schema": identity["schema"],
            "schema_version": identity["schema_version"],
            "store_version": STORE_VERSION,
            "contract_version": contract["contract_version"],
        },
        "inputs": {
            "aggregate_sha256": inputs["aggregate_sha256"],
            "rows": _inputs_identity_rows(inputs["frames"]),
            "note": (
                "content-only rows: the logical path/frame name is provenance, "
                "never identity"
            ),
        },
        "canvas": contract["canvas"],
        "grid": contract["grid"],
        "halo": contract["halo"],
        "memberships": contract["memberships"],
        "gauge": contract["gauge"],
        "science": contract["science"],
        "dtypes": contract["dtypes"],
        "backend": contract["backend"],
    }


def identity_fingerprints(identity: Mapping[str, Any]) -> dict[str, str]:
    """sha256 of each compatibility section (content, never name/mtime)."""
    sections = identity_sections(identity)
    return {
        name: _sha256_bytes(canonical_json(sections[name]))
        for name in IDENTITY_SECTIONS
    }


def identity_sha256(identity: Mapping[str, Any]) -> str:
    """Fingerprint of the whole scientific identity (execution context excluded)."""
    return _sha256_bytes(
        canonical_json(
            {
                "schema": IDENTITY_SCHEMA,
                "schema_version": STORE_VERSION,
                "science_contract": identity["science_contract"],
            }
        )
    )


_RUNTIME_ONLY_SECTIONS = ("execution_context",)


def check_identity_compatibility(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict:
    """Compare two identities. RAM/workers/prefetch are ignored by design.

    Returns a structured verdict; never raises and never "repairs" a mismatch.
    """
    ref = validate_run_identity(reference)
    cand = validate_run_identity(candidate)
    ref_fp = identity_fingerprints(ref)
    cand_fp = identity_fingerprints(cand)
    reasons: list[dict[str, Any]] = []
    for section in IDENTITY_SECTIONS:
        if ref_fp[section] != cand_fp[section]:
            reasons.append(
                {
                    "section": section,
                    "reference_sha256": ref_fp[section],
                    "candidate_sha256": cand_fp[section],
                    "detail": _section_detail(section),
                }
            )
    same_run_id = str(ref["run_id"]) == str(cand["run_id"])
    if not same_run_id:
        reasons.append(
            {
                "section": "run_id",
                "reference": str(ref["run_id"]),
                "candidate": str(cand["run_id"]),
                "detail": "a different run_id may never resume another run",
            }
        )
    backend_reason = next(
        (r for r in reasons if r["section"] == "backend"), None
    )
    if backend_reason is not None and not bool(
        cand["science_contract"]["backend"]["parity_qualified"]
    ):
        backend_reason["detail"] = (
            "backend/tile change without a qualified parity contract is "
            "refused (V1 conservative policy)"
        )
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "reference_identity_sha256": identity_sha256(ref),
        "candidate_identity_sha256": identity_sha256(cand),
        "ignored_sections": list(_RUNTIME_ONLY_SECTIONS),
        "note": (
            "execution_context (available RAM, workers, prefetch) never "
            "invalidates scientific identity"
        ),
    }


def _section_detail(section: str) -> str:
    return {
        "version": "schema/contract version changed (no force)",
        "inputs": (
            "input inventory content hash/order changed; a dataset moved to "
            "another path is NOT a change (only content matters)"
        ),
        "canvas": "canvas shape/WCS/geometry changed",
        "grid": "grid geometry/tile/cell set changed",
        "halo": "halo width changed",
        "memberships": "per-cell frame memberships changed",
        "gauge": "photometric gauge result/reference/exclusions changed",
        "science": "science contract or effective parameters changed",
        "dtypes": "declared plane dtypes changed",
        "backend": "backend/tile changed (parity must be qualified)",
    }.get(section, "section changed")


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitReceipt:
    run_id: str
    cell_id: str
    attempt_id: str
    status: str
    backend: str
    plan_generation: int
    payload_sha256: str
    logical_bytes: int
    stored_bytes: int
    commit_file: str
    idempotent: bool
    index_updated: bool
    created_unix_ns: int

    def to_dict(self) -> dict:
        return {
            "schema": COMMIT_SCHEMA,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "backend": self.backend,
            "plan_generation": int(self.plan_generation),
            "payload_sha256": self.payload_sha256,
            "logical_bytes": int(self.logical_bytes),
            "stored_bytes": int(self.stored_bytes),
            "commit_file": self.commit_file,
            "idempotent": bool(self.idempotent),
            "index_updated": bool(self.index_updated),
            "created_unix_ns": int(self.created_unix_ns),
        }


@dataclass
class CommitRecord:
    cell_id: str
    attempt_id: str
    status: str
    backend: str
    plan_generation: int
    payload_sha256: str
    logical_bytes: int
    stored_bytes: int
    commit_file: str
    created_unix_ns: int
    manifest: dict = field(repr=False, default_factory=dict)


# ---------------------------------------------------------------------------
# Payload planes
# ---------------------------------------------------------------------------

_CORE_2D = frozenset(zrr.CORE_2D_PLANES)


def _plane_items(result: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return ``(label, array)`` for every durable plane, in a stable order."""
    out: list[tuple[str, Any]] = []
    core = result.get("core")
    if isinstance(core, Mapping):
        for key in zrr.CORE_PLANES:
            if key in core:
                out.append((f"core.{key}", core[key]))
    patch = result.get("patch")
    if isinstance(patch, Mapping):
        for key in zrr.PATCH_PLANES:
            if key in patch:
                out.append((f"patch.{key}", patch[key]))
    return out


def _dtype_token(dtype) -> str:
    token = dtype.str
    if token not in ALLOWED_DTYPES:
        raise StoreFormatError(
            f"dtype {token!r} is not allowed by the durable store "
            f"(no object/complex/non-native; allowed={sorted(ALLOWED_DTYPES)})"
        )
    if not dtype.isnative:
        raise StoreFormatError(
            f"dtype {token!r} is not native byte order; refusing to store"
        )
    return token


def _array_bytes(value: Any, label: str) -> tuple[Any, bytes, str, str]:
    """Return (contiguous array, raw bytes, data sha256, array digest)."""
    import numpy as np

    if not isinstance(value, np.ndarray):
        raise StoreFormatError(
            f"{label} must be a numpy array, got {type(value).__name__}"
        )
    if value.dtype.hasobject:
        raise StoreFormatError(f"{label} has object dtype (refused)")
    _dtype_token(value.dtype)
    contiguous = np.ascontiguousarray(value)
    data = contiguous.tobytes(order="C")
    data_sha = _sha256_bytes(data)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json([int(v) for v in contiguous.shape]))
    digest.update(data)
    return contiguous, data, data_sha, digest.hexdigest()


def _object_relpath(data_sha: str) -> str:
    return f"objects/{data_sha[:2]}/{data_sha}.bin"


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class RunCellStore:
    """Durable per-cell commit store for ONE run directory."""

    def __init__(
        self,
        root: Path,
        *,
        budget: DiskBudget | None = None,
        session_id: str | None = None,
        fault_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
        free_bytes_fn: Callable[[Path], int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.budget = budget
        self.session_id = session_id
        self.fault_hook = fault_hook
        self.free_bytes_fn = free_bytes_fn or (
            lambda path: int(shutil.disk_usage(str(path)).free)
        )
        self._lock: StoreLock | None = None
        self._identity: dict | None = None
        self._identity_sha: str | None = None
        self._commits: dict[str, dict[str, CommitRecord]] = {}
        self._invalid: list[dict] = []
        self._index: dict | None = None
        self._index_errors: list[dict] = []
        self._index_rebuilt = False
        self._index_stale = False
        self._index_generation = 0
        self._stored_bytes = 0
        self._instance_id = uuid.uuid4().hex
        self._closed = False
        self._lock_info: dict | None = None
        self._commit_count = 0
        self._opened_unix_ns = int(time.time_ns())

    # -- properties ------------------------------------------------------
    @property
    def run_id(self) -> str:
        if self._identity is None:
            raise StoreClosedError("store is not open")
        return str(self._identity["run_id"])

    @property
    def identity(self) -> dict:
        if self._identity is None:
            raise StoreClosedError("store is not open")
        return dict(self._identity)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def index_rebuilt(self) -> bool:
        return bool(self._index_rebuilt)

    @property
    def stored_bytes(self) -> int:
        return int(self._stored_bytes)

    @property
    def lock_owner(self) -> dict | None:
        return dict(self._lock.owner) if self._lock and self._lock.owner else None

    # -- paths -----------------------------------------------------------
    def _p(self, *parts: str) -> Path:
        path = self.root
        for part in parts:
            path = path / part
        return path

    @property
    def _tmp_dir(self) -> Path:
        return self._p("tmp", self._instance_id)

    def _fault(self, stage: str, **info: Any) -> None:
        if self.fault_hook is not None:
            self.fault_hook(stage, dict(info))

    # -- open / close ----------------------------------------------------
    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        identity: Mapping[str, Any] | None = None,
        budget: DiskBudget | None = None,
        session_id: str | None = None,
        fault_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
        free_bytes_fn: Callable[[Path], int] | None = None,
        read_only: bool = False,
        create: bool = True,
    ) -> "RunCellStore":
        store = cls(
            Path(root),
            budget=budget,
            session_id=session_id,
            fault_hook=fault_hook,
            free_bytes_fn=free_bytes_fn,
        )
        store._open(
            identity=identity,
            read_only=bool(read_only),
            create=bool(create),
        )
        return store

    def _open(
        self, *, identity: Mapping[str, Any] | None, read_only: bool, create: bool
    ) -> None:
        root = self.root
        if root.exists() and root.is_symlink():
            raise StorePathError(f"store root must not be a symlink: {root}")
        if not root.exists():
            if not create:
                raise StoreNotFoundError(f"store root does not exist: {root}")
            root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise StorePathError(f"store root is not a directory: {root}")
        if not read_only:
            self._lock = StoreLock(
                root / "LOCK",
                session_id=self.session_id,
                instance_id=self._instance_id,
            )
            self._lock_info = self._lock.acquire()
        for name in ("commits", "objects", "tmp"):
            (root / name).mkdir(parents=True, exist_ok=True)
        if not read_only:
            self._tmp_dir.mkdir(parents=True, exist_ok=True)

        candidate = (
            validate_run_identity(identity) if identity is not None else None
        )
        run_json = root / "run.json"
        if run_json.exists():
            stored = self._read_run_json(run_json)
            if candidate is not None:
                verdict = check_identity_compatibility(stored, candidate)
                if not verdict["compatible"]:
                    reasons = "; ".join(
                        f"{r['section']}: {r.get('detail', '')}"
                        for r in verdict["reasons"]
                    )
                    self._cleanup_failed_open()
                    raise StoreIdentityMismatch(
                        "run identity is incompatible with the stored run; "
                        f"refusing silently: {reasons}",
                        reasons=verdict["reasons"],
                    )
            self._identity = stored
        else:
            if candidate is None:
                self._cleanup_failed_open()
                raise StoreFormatError(
                    f"store {root} has no run.json and no identity was supplied"
                )
            if read_only:
                self._cleanup_failed_open()
                raise StoreNotFoundError(f"store {root} has no run.json")
            self._write_run_json(candidate)
            self._identity = candidate
        self._identity_sha = identity_sha256(self._identity)

        self._scan_commits()
        self._load_or_rebuild_index(write=not read_only)

    def _cleanup_failed_open(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _read_run_json(self, path: Path) -> dict:
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise StoreFormatError(
                f"run identity at {path} is not readable JSON: {exc}"
            ) from exc
        return validate_run_identity(value)

    def _write_run_json(self, identity: Mapping[str, Any]) -> None:
        self._fault("identity", path=str(self.root / "run.json"))
        atomic_write_bytes(
            self.root / "run.json",
            canonical_json(identity),
            tmp_dir=self._tmp_dir,
        )

    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        released = False
        if self._lock is not None:
            released = self._lock.release()
            self._lock = None
        return bool(released)

    def __enter__(self) -> "RunCellStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # pragma: no cover - GC safety net
        try:
            self.close()
        except Exception:
            pass

    # -- commits scan ----------------------------------------------------
    def _commit_token(self, cell_id: str, attempt_id: str) -> str:
        return _sha256_bytes(f"{cell_id}\x00{attempt_id}".encode("utf-8"))[:24]

    def _commit_filename(self, cell_id: str, attempt_id: str) -> str:
        return f"{cell_id}__{self._commit_token(cell_id, attempt_id)}.json"

    def _scan_commits(self) -> None:
        commits_dir = self._p("commits")
        self._commits = {}
        self._invalid = []
        self._stored_bytes = 0
        for path in sorted(commits_dir.glob("*.json")):
            try:
                record = self._read_commit_manifest(path)
            except (StoreFormatError, StoreIntegrityError, OSError) as exc:
                self._invalid.append(
                    {"file": path.name, "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            self._commits.setdefault(record.cell_id, {})[
                record.attempt_id
            ] = record
            self._stored_bytes += int(record.stored_bytes)
        self._commit_count = sum(len(v) for v in self._commits.values())

    def _read_commit_manifest(self, path: Path) -> CommitRecord:
        try:
            raw = json.loads(path.read_bytes().decode("utf-8"))
        except Exception as exc:
            raise StoreFormatError(
                f"commit {path.name} is not readable JSON: {exc}"
            ) from exc
        manifest = self._validate_commit_manifest(raw, path.name)
        if str(manifest["run_id"]) != str(self.run_id):
            raise StoreFormatError(
                f"commit {path.name} belongs to run_id {manifest['run_id']!r}, "
                f"not {self.run_id!r}"
            )
        if str(manifest["identity_sha256"]) != str(self._identity_sha):
            raise StoreFormatError(
                f"commit {path.name} belongs to another run identity"
            )
        declared_sha = manifest["commit_sha256"]
        body = {k: v for k, v in manifest.items() if k != "commit_sha256"}
        actual_sha = _sha256_bytes(canonical_json(body))
        if actual_sha != declared_sha:
            raise StoreIntegrityError(
                f"commit {path.name} metadata digest mismatch "
                f"(declared {declared_sha}, actual {actual_sha})"
            )
        return CommitRecord(
            cell_id=manifest["cell_id"],
            attempt_id=manifest["attempt_id"],
            status=manifest["status"],
            backend=manifest["backend"],
            plan_generation=int(manifest["plan_generation"]),
            payload_sha256=manifest["payload_sha256"],
            logical_bytes=int(manifest["logical_bytes"]),
            stored_bytes=int(manifest["stored_bytes"]),
            commit_file=path.name,
            created_unix_ns=int(manifest["created_unix_ns"]),
            manifest=manifest,
        )

    def _validate_commit_manifest(self, raw: Any, label: str) -> dict:
        manifest = _strict_mapping(
            raw,
            f"commit {label}",
            (
                "schema",
                "schema_version",
                "run_id",
                "cell_id",
                "attempt_id",
                "status",
                "backend",
                "plan_generation",
                "identity_sha256",
                "payload_sha256",
                "logical_bytes",
                "stored_bytes",
                "created_unix_ns",
                "planes",
                "payload_meta",
                "commit_sha256",
            ),
        )
        if manifest["schema"] != COMMIT_SCHEMA:
            raise StoreFormatError(
                f"commit {label} schema {manifest['schema']!r} != "
                f"{COMMIT_SCHEMA!r}"
            )
        if str(manifest["schema_version"]) != STORE_VERSION:
            raise StoreFormatError(
                f"commit {label} schema_version {manifest['schema_version']!r} "
                f"is unsupported"
            )
        safe_name(manifest["cell_id"], f"commit {label}.cell_id")
        _req_str(manifest["attempt_id"], f"commit {label}.attempt_id")
        if manifest["status"] not in (STATUS_COMPLETE, STATUS_EMPTY_VALIDATED):
            raise StoreFormatError(
                f"commit {label} has invalid status {manifest['status']!r}"
            )
        _req_str(manifest["backend"], f"commit {label}.backend")
        _req_int(manifest["plan_generation"], f"commit {label}.plan_generation",
                 minimum=0)
        _req_str(manifest["identity_sha256"], f"commit {label}.identity_sha256")
        _req_str(manifest["payload_sha256"], f"commit {label}.payload_sha256")
        _req_int(manifest["logical_bytes"], f"commit {label}.logical_bytes",
                 minimum=0)
        _req_int(manifest["stored_bytes"], f"commit {label}.stored_bytes",
                 minimum=0)
        _req_int(manifest["created_unix_ns"], f"commit {label}.created_unix_ns")
        planes = _req_seq(manifest["planes"], f"commit {label}.planes")
        seen: set[str] = set()
        for index, entry in enumerate(planes):
            plane = _strict_mapping(
                entry,
                f"commit {label}.planes[{index}]",
                ("name", "path", "dtype", "shape", "nbytes", "sha256",
                 "array_sha256"),
            )
            name = _req_str(plane["name"], f"planes[{index}].name")
            if name in seen:
                raise StoreFormatError(
                    f"commit {label} declares plane {name!r} twice"
                )
            seen.add(name)
            confined_path(self.root, plane["path"], f"planes[{index}].path")
            dtype_token = _req_str(plane["dtype"], f"planes[{index}].dtype")
            if dtype_token not in ALLOWED_DTYPES:
                raise StoreFormatError(
                    f"commit {label} plane {name} declares unsupported dtype "
                    f"{dtype_token!r}"
                )
            shape = _req_seq(plane["shape"], f"planes[{index}].shape")
            if not shape or len(shape) > 3:
                raise StoreFormatError(
                    f"commit {label} plane {name} has an invalid shape"
                )
            for dim_index, dim in enumerate(shape):
                _req_int(dim, f"planes[{index}].shape[{dim_index}]", minimum=1)
            nbytes = _req_int(plane["nbytes"], f"planes[{index}].nbytes",
                              minimum=1)
            expected = int(
                _dtype_itemsize(dtype_token) * _prod(int(v) for v in shape)
            )
            if nbytes != expected:
                raise StoreFormatError(
                    f"commit {label} plane {name} declares nbytes={nbytes} but "
                    f"dtype/shape imply {expected}"
                )
            _req_str(plane["sha256"], f"planes[{index}].sha256")
            _req_str(plane["array_sha256"], f"planes[{index}].array_sha256")
        meta = manifest["payload_meta"]
        if not isinstance(meta, Mapping):
            raise StoreFormatError(f"commit {label}.payload_meta must be an object")
        for forbidden in ("core", "patch"):
            if forbidden in meta:
                raise StoreFormatError(
                    f"commit {label}.payload_meta may not carry {forbidden!r}"
                )
        _req_str(manifest["commit_sha256"], f"commit {label}.commit_sha256")
        return dict(manifest)

    # -- index -----------------------------------------------------------
    def _build_index(self, *, generation: int) -> dict:
        cells: dict[str, Any] = {}
        complete = 0
        empty = 0
        for cell_id in sorted(self._commits):
            attempts = []
            for attempt_id in sorted(self._commits[cell_id]):
                record = self._commits[cell_id][attempt_id]
                attempts.append(
                    {
                        "attempt_id": record.attempt_id,
                        "payload_sha256": record.payload_sha256,
                        "status": record.status,
                        "commit_file": record.commit_file,
                        "stored_bytes": int(record.stored_bytes),
                        "created_unix_ns": int(record.created_unix_ns),
                    }
                )
                if record.status == STATUS_COMPLETE:
                    complete += 1
                else:
                    empty += 1
            cells[cell_id] = attempts
        index = {
            "schema": INDEX_SCHEMA,
            "schema_version": STORE_VERSION,
            "run_id": self.run_id,
            "identity_sha256": self._identity_sha,
            "generation": int(generation),
            "updated_unix_ns": int(time.time_ns()),
            "cells": cells,
            "invalid": [
                {"file": entry["file"], "reason": entry["reason"]}
                for entry in self._invalid
            ],
            "counts": {
                "cells": len(self._commits),
                "attempts": int(sum(len(v) for v in self._commits.values())),
                "complete": int(complete),
                "empty_validated": int(empty),
                "stored_bytes": int(self._stored_bytes),
            },
        }
        index["index_sha256"] = _sha256_bytes(
            canonical_json({k: v for k, v in index.items() if k != "index_sha256"})
        )
        return index

    def _write_index(self) -> None:
        self._index = self._build_index(generation=self._index_generation)
        self._fault(
            "index",
            path=str(self.root / "index.json"),
            generation=int(self._index_generation),
            cells=len(self._commits),
        )
        atomic_write_bytes(
            self.root / "index.json",
            canonical_json(self._index),
            tmp_dir=self._tmp_dir,
        )
        self._index_stale = False

    def _load_or_rebuild_index(self, *, write: bool) -> None:
        path = self.root / "index.json"
        stored: dict | None = None
        if path.exists():
            try:
                value = json.loads(path.read_bytes().decode("utf-8"))
                stored = self._validate_index(value)
            except Exception as exc:
                self._index_errors.append(
                    {
                        "file": "index.json",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                stored = None
        expected = self._build_index(
            generation=int((stored or {}).get("generation", 0))
        )
        if stored is not None and self._index_equivalent(stored, expected):
            self._index = stored
            self._index_generation = int(stored["generation"])
            self._index_rebuilt = False
            return
        self._index_generation = int(expected["generation"]) + 1
        self._index = expected
        self._index_rebuilt = True
        if write:
            try:
                self._write_index()
            except OSError as exc:
                self._index_stale = True
                self._index_errors.append(
                    {
                        "file": "index.json",
                        "reason": f"index rewrite failed: {exc}",
                    }
                )

    @staticmethod
    def _index_equivalent(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
        for key in ("run_id", "identity_sha256", "cells", "counts"):
            if canonical_json(stored.get(key)) != canonical_json(expected.get(key)):
                return False
        stored_invalid = sorted(
            (entry.get("file"), entry.get("reason"))
            for entry in stored.get("invalid", [])
        )
        expected_invalid = sorted(
            (entry.get("file"), entry.get("reason"))
            for entry in expected.get("invalid", [])
        )
        return stored_invalid == expected_invalid

    def _validate_index(self, raw: Any) -> dict:
        index = _strict_mapping(
            raw,
            "index",
            (
                "schema",
                "schema_version",
                "run_id",
                "identity_sha256",
                "generation",
                "updated_unix_ns",
                "cells",
                "invalid",
                "counts",
                "index_sha256",
            ),
        )
        if index["schema"] != INDEX_SCHEMA:
            raise StoreFormatError(f"index schema {index['schema']!r} is invalid")
        if str(index["schema_version"]) != STORE_VERSION:
            raise StoreFormatError("index schema_version is unsupported")
        if str(index["run_id"]) != str(self.run_id):
            raise StoreFormatError("index belongs to another run_id")
        if str(index["identity_sha256"]) != str(self._identity_sha):
            raise StoreFormatError(
                "index identity hash disagrees with run.json (tampered or stale)"
            )
        expected_sha = _sha256_bytes(
            canonical_json({k: v for k, v in index.items() if k != "index_sha256"})
        )
        if expected_sha != index["index_sha256"]:
            raise StoreIntegrityError("index digest mismatch")
        _req_int(index["generation"], "index.generation", minimum=0)
        _req_int(index["updated_unix_ns"], "index.updated_unix_ns")
        cells = index["cells"]
        if not isinstance(cells, Mapping):
            raise StoreFormatError("index.cells must be an object")
        for cell_id, attempts in cells.items():
            safe_name(cell_id, "index.cells key")
            _req_seq(attempts, f"index.cells.{cell_id}")
        _req_seq(index["invalid"], "index.invalid")
        counts = index["counts"]
        if not isinstance(counts, Mapping):
            raise StoreFormatError("index.counts must be an object")
        return dict(index)

    def rebuild_index(self, *, write: bool = True) -> dict:
        self._ensure_open()
        self._scan_commits()
        self._index_generation += 1
        if write:
            self._write_index()
        else:
            self._index = self._build_index(generation=self._index_generation)
        return dict(self._index or {})

    def index(self) -> dict:
        self._ensure_open()
        return dict(self._index or {})

    # -- budget ----------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed or self._identity is None:
            raise StoreClosedError("store is closed")
        if self._lock is not None and not self._lock.held:
            raise StoreClosedError("store lock is not held")

    def preflight_bytes(self, needed: int) -> dict:
        """Check the durable line and the free-space margin BEFORE writing."""
        self._ensure_open()
        needed = int(needed)
        budget = self.budget
        used = int(self._stored_bytes)
        if budget is not None and budget.durable_cell_bytes > 0:
            if used + needed > int(budget.durable_cell_bytes):
                raise StoreDiskFullError(
                    "durable cell budget exhausted: "
                    f"used={used} + needed={needed} > "
                    f"budget={budget.durable_cell_bytes}",
                    required_bytes=needed,
                    free_bytes=None,
                    budget_bytes=int(budget.durable_cell_bytes),
                    used_bytes=used,
                )
        margin = (
            int(budget.transactional_margin_bytes)
            if budget is not None
            else 0
        )
        try:
            free = int(self.free_bytes_fn(self.root))
        except Exception as exc:
            raise StoreDiskFullError(
                f"cannot read free space for {self.root}: {exc}",
                required_bytes=needed,
                free_bytes=None,
            ) from exc
        if free - margin < needed:
            raise StoreDiskFullError(
                "insufficient free space preserving the transactional margin: "
                f"free={free} margin={margin} needed={needed}",
                required_bytes=needed,
                free_bytes=free,
                budget_bytes=(
                    None if budget is None else int(budget.durable_cell_bytes)
                ),
                used_bytes=used,
            )
        return {
            "needed_bytes": needed,
            "free_bytes": free,
            "margin_bytes": margin,
            "used_durable_bytes": used,
            "budget_bytes": (
                None if budget is None else int(budget.durable_cell_bytes)
            ),
        }

    # -- commit ----------------------------------------------------------
    def commit_cell(
        self,
        result: Mapping[str, Any],
        *,
        attempt_id: str,
        backend: str | None = None,
        plan_generation: int = 0,
    ) -> CommitReceipt:
        """Durably publish ONE cell result. Idempotent by (cell_id, attempt_id)."""
        self._ensure_open()
        if not isinstance(result, Mapping):
            raise StoreFormatError(
                f"cell result must be a mapping, got {type(result).__name__}"
            )
        cell_id = safe_name(result.get("cell_id"), "result.cell_id")
        attempt = _req_str(attempt_id, "attempt_id")
        if isinstance(plan_generation, bool) or not isinstance(
            plan_generation, int
        ):
            raise StoreFormatError("plan_generation must be a strict int")
        raw_status = str(result.get("status", "")).strip().lower()
        status = _CELL_STATUS_MAP.get(raw_status)
        if status is None:
            raise StoreFormatError(
                f"cell {cell_id} has unsupported status {result.get('status')!r}"
            )
        gauge_mode = str(self._identity["science_contract"]["gauge"]["mode"])
        try:
            if status == STATUS_COMPLETE:
                validated = zrr.validate_complete_payload(
                    result, gauge_mode=gauge_mode
                )
            else:
                validated = zrr.validate_empty_payload(
                    result, gauge_mode=gauge_mode
                )
        except zrr.AcceptanceError as exc:
            # Same real producer schema as M1; the store reports it as a strict
            # format refusal (never a partial commit).
            raise StoreFormatError(
                f"cell {cell_id} payload is invalid for the durable store: {exc}"
            ) from exc
        fingerprint = zrr.fingerprint_result(result)
        if status == STATUS_EMPTY_VALIDATED:
            fingerprint["geometry"]["n_contributors"] = validated["geometry"].get(
                "n_contributors"
            )
            fingerprint["geometry"]["gauge_status"] = validated["geometry"].get(
                "gauge_status"
            )
        payload_sha = str(fingerprint["payload_sha256"])

        # -- idempotence check BEFORE any write ---------------------------
        existing = self._commits.get(cell_id, {}).get(attempt)
        if existing is not None:
            if existing.payload_sha256 == payload_sha:
                verified = self.verify_cell(cell_id, attempt)
                if not verified["valid"]:
                    raise StoreIntegrityError(
                        f"cell {cell_id} attempt {attempt!r} is published but "
                        f"no longer valid: {verified['reason']}"
                    )
                return CommitReceipt(
                    run_id=self.run_id,
                    cell_id=cell_id,
                    attempt_id=attempt,
                    status=existing.status,
                    backend=existing.backend,
                    plan_generation=int(existing.plan_generation),
                    payload_sha256=existing.payload_sha256,
                    logical_bytes=int(existing.logical_bytes),
                    stored_bytes=int(existing.stored_bytes),
                    commit_file=existing.commit_file,
                    idempotent=True,
                    index_updated=not self._index_stale,
                    created_unix_ns=int(existing.created_unix_ns),
                )
            raise StoreConflictError(
                f"cell {cell_id} attempt {attempt!r} is already published with "
                f"a different payload ({existing.payload_sha256} != {payload_sha}); "
                "refusing to overwrite a validated identity"
            )
        # A published commit for the same cell under another attempt is legal
        # only as a distinct attempt identity; the caller decides. We never
        # silently replace it.

        planes_meta: list[dict] = []
        staged: list[tuple[dict, bytes]] = []
        logical_bytes = 0
        array_bytes_total = 0
        for label, value in _plane_items(result):
            contiguous, data, data_sha, array_digest = _array_bytes(value, label)
            entry = {
                "name": label,
                "path": _object_relpath(data_sha),
                "dtype": contiguous.dtype.str,
                "shape": [int(v) for v in contiguous.shape],
                "nbytes": int(len(data)),
                "sha256": data_sha,
                "array_sha256": array_digest,
            }
            planes_meta.append(entry)
            staged.append((entry, data))
            logical_bytes += int(contiguous.nbytes)
            array_bytes_total += int(len(data))

        payload_meta = {
            key: value
            for key, value in result.items()
            if key not in ("core", "patch")
        }
        payload_meta = json_safe(payload_meta, "payload_meta")
        # The declared dtype contract of the run is authoritative: refuse a
        # plane whose dtype disagrees with the frozen contract.
        declared = self._identity["science_contract"]["dtypes"]["planes"]
        for entry in planes_meta:
            expected = declared.get(entry["name"])
            if expected is None:
                raise StoreFormatError(
                    f"plane {entry['name']} is not part of the frozen dtype "
                    "contract of this run"
                )
            tokens = [expected] if isinstance(expected, str) else list(expected)
            if entry["dtype"] not in [str(t) for t in tokens]:
                raise StoreFormatError(
                    f"plane {entry['name']} dtype {entry['dtype']} disagrees "
                    f"with the run contract {tokens}"
                )

        backend_token = str(
            backend
            or result.get("backend_used")
            or payload_meta.get("backend_used")
            or fingerprint["backend"]
            or "unknown"
        )
        created = int(time.time_ns())
        manifest = {
            "schema": COMMIT_SCHEMA,
            "schema_version": STORE_VERSION,
            "run_id": self.run_id,
            "cell_id": cell_id,
            "attempt_id": attempt,
            "status": status,
            "backend": backend_token,
            "plan_generation": int(plan_generation),
            "identity_sha256": self._identity_sha,
            "payload_sha256": payload_sha,
            "logical_bytes": int(logical_bytes),
            "stored_bytes": 0,
            "created_unix_ns": created,
            "planes": planes_meta,
            "payload_meta": payload_meta,
        }
        manifest_bytes = canonical_json(manifest)
        manifest["stored_bytes"] = int(array_bytes_total + len(manifest_bytes) + 256)
        body = {k: v for k, v in manifest.items() if k != "commit_sha256"}
        manifest["commit_sha256"] = _sha256_bytes(canonical_json(body))
        manifest_bytes = canonical_json(manifest)

        preflight = self.preflight_bytes(
            int(array_bytes_total + len(manifest_bytes))
        )

        # -- reserve -> temp -> fsync -> verify ----------------------------
        objects_written: list[Path] = []
        try:
            for entry, data in staged:
                self._write_object(entry, data)
                objects_written.append(self.root / entry["path"])
            # -- publish the cell commit LAST (atomic rename) -------------
            self._fault(
                "commit_rename",
                cell_id=cell_id,
                attempt_id=attempt,
                bytes=int(len(manifest_bytes)),
            )
            target = self._p("commits", self._commit_filename(cell_id, attempt))
            atomic_write_bytes(target, manifest_bytes, tmp_dir=self._tmp_dir)
        except OSError as exc:
            self._cleanup_temp()
            if _is_disk_full(exc):
                raise StoreDiskFullError(
                    "disk full while committing cell "
                    f"{cell_id}: {exc}; prior commits are untouched",
                    required_bytes=int(array_bytes_total + len(manifest_bytes)),
                    free_bytes=None,
                    budget_bytes=(
                        None if self.budget is None
                        else int(self.budget.durable_cell_bytes)
                    ),
                    used_bytes=int(self._stored_bytes),
                ) from exc
            raise StoreFormatError(
                f"commit of cell {cell_id} failed before publication: {exc}"
            ) from exc

        record = CommitRecord(
            cell_id=cell_id,
            attempt_id=attempt,
            status=status,
            backend=backend_token,
            plan_generation=int(plan_generation),
            payload_sha256=payload_sha,
            logical_bytes=int(logical_bytes),
            stored_bytes=int(manifest["stored_bytes"]),
            commit_file=target.name,
            created_unix_ns=created,
            manifest=manifest,
        )
        self._commits.setdefault(cell_id, {})[attempt] = record
        self._stored_bytes += int(record.stored_bytes)
        self._commit_count += 1
        self._cleanup_temp()

        # -- index (reconstructible; a failure here loses nothing) --------
        index_updated = False
        try:
            self._index_generation += 1
            self._write_index()
            index_updated = True
        except OSError as exc:
            self._index_stale = True
            self._index_errors.append(
                {
                    "file": "index.json",
                    "reason": f"index update after commit failed: {exc}",
                }
            )
        return CommitReceipt(
            run_id=self.run_id,
            cell_id=cell_id,
            attempt_id=attempt,
            status=status,
            backend=backend_token,
            plan_generation=int(plan_generation),
            payload_sha256=payload_sha,
            logical_bytes=int(logical_bytes),
            stored_bytes=int(record.stored_bytes),
            commit_file=target.name,
            idempotent=False,
            index_updated=bool(index_updated),
            created_unix_ns=created,
        )

    def _write_object(self, entry: Mapping[str, Any], data: bytes) -> None:
        """Write one content-addressed object with reserve/fsync/verify."""
        final = confined_path(self.root, entry["path"], "object path")
        if final.exists():
            actual = _file_sha256(final)
            if actual != entry["sha256"]:
                raise StoreIntegrityError(
                    f"object {entry['path']} already exists with a different "
                    f"digest ({actual} != {entry['sha256']}); refusing to "
                    "overwrite a referenced immutable artifact"
                )
            if int(final.stat().st_size) != int(entry["nbytes"]):
                raise StoreIntegrityError(
                    f"object {entry['path']} has size {final.stat().st_size} "
                    f"!= declared {entry['nbytes']}"
                )
            return
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_dir / f"obj-{uuid.uuid4().hex}.part"
        self._fault(
            "temp_write",
            path=str(tmp),
            object=entry["path"],
            nbytes=int(entry["nbytes"]),
        )
        fd = _open_binary(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            mode = _reserve_space(fd, int(entry["nbytes"]))
            entry_mode = mode
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:  # pragma: no cover
                    raise OSError("short write")
                view = view[written:]
            self._fault("fsync", path=str(tmp), object=entry["path"])
            fsync_file(fd)
        except OSError as exc:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
            if _is_disk_full(exc):
                raise StoreDiskFullError(
                    f"disk full while writing object {entry['path']}: {exc}",
                    required_bytes=int(entry["nbytes"]),
                    free_bytes=None,
                ) from exc
            raise
        else:
            os.close(fd)
        # -- verify the bytes on disk BEFORE publication -------------------
        actual = _file_sha256(tmp)
        if actual != entry["sha256"]:
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
            raise StoreIntegrityError(
                f"object {entry['path']} failed verification after write "
                f"({actual} != {entry['sha256']})"
            )
        if int(tmp.stat().st_size) != int(entry["nbytes"]):
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
            raise StoreIntegrityError(
                f"object {entry['path']} size mismatch after write"
            )
        os.replace(str(tmp), str(final))
        fsync_dir(final.parent)
        # ``entry_mode`` documents the reservation strength; a weak mode is
        # still safe because the commit manifest is published last.
        try:
            os.chmod(str(final), 0o444)
        except OSError:  # pragma: no cover - Windows ACL edge
            pass

    def _cleanup_temp(self) -> None:
        tmp_dir = self._tmp_dir
        if not tmp_dir.exists():
            return
        for path in tmp_dir.iterdir():
            try:
                if path.is_file():
                    os.unlink(str(path))
            except OSError:
                continue

    # -- reads -----------------------------------------------------------
    def attempts(self, cell_id: str) -> list[str]:
        self._ensure_open()
        return sorted(self._commits.get(str(cell_id), {}))

    def commit_record(self, cell_id: str, attempt_id: str | None = None) -> CommitRecord:
        self._ensure_open()
        cid = str(cell_id)
        records = self._commits.get(cid, {})
        if not records:
            raise StoreNotFoundError(f"no durable commit for cell {cid!r}")
        if attempt_id is None:
            if len(records) > 1:
                raise StoreFormatError(
                    f"cell {cid!r} has {len(records)} attempts; an explicit "
                    "attempt_id is required (never 'last found')"
                )
            return next(iter(records.values()))
        record = records.get(str(attempt_id))
        if record is None:
            raise StoreNotFoundError(
                f"no durable commit for cell {cid!r} attempt {attempt_id!r}"
            )
        return record

    def verify_cell(self, cell_id: str, attempt_id: str | None = None) -> dict:
        """Verify one cell commit and every declared object; never repairs."""
        self._ensure_open()
        try:
            record = self.commit_record(cell_id, attempt_id)
        except RunStoreError as exc:
            return {"valid": False, "reason": f"{type(exc).__name__}: {exc}",
                    "cell_id": str(cell_id)}
        try:
            self._verify_record(record)
        except (StoreFormatError, StoreIntegrityError, OSError) as exc:
            return {
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "cell_id": record.cell_id,
                "attempt_id": record.attempt_id,
            }
        return {
            "valid": True,
            "reason": None,
            "cell_id": record.cell_id,
            "attempt_id": record.attempt_id,
            "planes": len(record.manifest["planes"]),
            "stored_bytes": int(record.stored_bytes),
            "payload_sha256": record.payload_sha256,
        }

    def _verify_record(self, record: CommitRecord) -> None:
        manifest = record.manifest
        if str(manifest["identity_sha256"]) != str(self._identity_sha):
            raise StoreIntegrityError(
                f"commit {record.commit_file} belongs to another run identity"
            )
        if str(manifest["run_id"]) != str(self.run_id):
            raise StoreIntegrityError(
                f"commit {record.commit_file} belongs to another run_id"
            )
        for plane in manifest["planes"]:
            path = confined_path(self.root, plane["path"], "plane path")
            if not path.exists():
                raise StoreIntegrityError(
                    f"cell {record.cell_id}: object {plane['path']} is missing"
                )
            size = int(path.stat().st_size)
            if size != int(plane["nbytes"]):
                raise StoreIntegrityError(
                    f"cell {record.cell_id}: object {plane['path']} size "
                    f"{size} != declared {plane['nbytes']}"
                )
            actual = _file_sha256(path)
            if actual != str(plane["sha256"]):
                raise StoreIntegrityError(
                    f"cell {record.cell_id}: object {plane['path']} digest "
                    f"{actual} != declared {plane['sha256']}"
                )

    def load_payload(self, cell_id: str, attempt_id: str | None = None) -> dict:
        """Load a verified payload. Arrays are read-only; nothing is repaired."""
        self._ensure_open()
        record = self.commit_record(cell_id, attempt_id)
        self._verify_record(record)
        payload: dict[str, Any] = dict(record.manifest["payload_meta"])
        if record.status == STATUS_COMPLETE:
            core: dict[str, Any] = {}
            patch: dict[str, Any] | None = None
            for plane in record.manifest["planes"]:
                array = self._read_array(plane)
                name = str(plane["name"])
                if name.startswith("core."):
                    core[name[len("core."):]] = array
                elif name.startswith("patch."):
                    if patch is None:
                        patch = {}
                    patch[name[len("patch."):]] = array
            payload["core"] = core
            if patch is not None:
                payload["patch"] = patch
        payload["cell_id"] = record.cell_id
        payload["_store"] = {
            "schema": STORE_SCHEMA,
            "run_id": self.run_id,
            "attempt_id": record.attempt_id,
            "payload_sha256": record.payload_sha256,
            "commit_file": record.commit_file,
            "source": "run_store",
        }
        return payload

    def _read_array(self, plane: Mapping[str, Any]):
        import numpy as np

        path = confined_path(self.root, plane["path"], "plane path")
        dtype = np.dtype(str(plane["dtype"]))
        shape = tuple(int(v) for v in plane["shape"])
        expected = int(dtype.itemsize * _prod(shape))
        with open(path, "rb") as fh:
            data = fh.read(expected + 1)
        if len(data) != expected:
            raise StoreIntegrityError(
                f"object {plane['path']} holds {len(data)} bytes, expected "
                f"{expected}"
            )
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        array.flags.writeable = False
        return array

    # -- completeness ----------------------------------------------------
    def completeness(self, expected_cell_ids: Iterable[str] | None = None) -> dict:
        """Exact sparse completeness: expected == COMPLETE + EMPTY_VALIDATED."""
        self._ensure_open()
        if expected_cell_ids is None:
            expected = sorted(self._commits)
        else:
            expected = [str(cid) for cid in expected_cell_ids]
            if len(set(expected)) != len(expected):
                raise StoreFormatError("expected_cell_ids has duplicates")
        complete: list[str] = []
        empty: list[str] = []
        invalid: list[str] = []
        for cell_id in expected:
            records = self._commits.get(cell_id, {})
            if not records:
                continue
            statuses = {r.status for r in records.values()}
            verified = [
                self.verify_cell(cell_id, attempt_id)
                for attempt_id in sorted(records)
            ]
            if not all(v["valid"] for v in verified):
                invalid.append(cell_id)
                continue
            if STATUS_COMPLETE in statuses:
                complete.append(cell_id)
            elif STATUS_EMPTY_VALIDATED in statuses:
                empty.append(cell_id)
        missing = [
            cid for cid in expected if cid not in set(complete) | set(empty)
            and cid not in set(invalid)
        ]
        return {
            "expected": len(expected),
            "complete": len(complete),
            "empty_validated": len(empty),
            "missing": missing,
            "invalid": invalid,
            "is_complete": not missing and not invalid,
            "complete_cell_ids": complete,
            "empty_cell_ids": empty,
            "policy": "sparse cell_id set; never a 0..K prefix",
        }

    def scan(self) -> dict:
        """Report valid and invalid commits. Invalid cells stay isolated."""
        self._ensure_open()
        valid = []
        for cell_id in sorted(self._commits):
            for attempt_id in sorted(self._commits[cell_id]):
                valid.append(self.verify_cell(cell_id, attempt_id))
        return {
            "run_id": self.run_id,
            "identity_sha256": self._identity_sha,
            "valid": [entry for entry in valid if entry["valid"]],
            "invalid": [entry for entry in valid if not entry["valid"]]
            + list(self._invalid),
            "stored_bytes": int(self._stored_bytes),
        }

    def report(self) -> dict:
        """Lightweight audit state (no arrays, safe for diagnostics)."""
        if self._identity is None:
            return {"schema": STORE_SCHEMA, "open": False}
        return {
            "schema": STORE_SCHEMA,
            "open": not self._closed,
            "root": str(self.root),
            "run_id": self.run_id,
            "identity_sha256": self._identity_sha,
            "identity_fingerprints": identity_fingerprints(self._identity),
            "counts": {
                "cells": len(self._commits),
                "attempts": int(self._commit_count),
                "invalid": len(self._invalid),
                "stored_bytes": int(self._stored_bytes),
            },
            "index": {
                "generation": int(self._index_generation),
                "rebuilt_on_open": bool(self._index_rebuilt),
                "stale": bool(self._index_stale),
            },
            "lock": (
                dict(self._lock.owner)
                if self._lock is not None and self._lock.owner is not None
                else None
            ),
            "lock_info": (
                dict(self._lock_info) if self._lock_info is not None else None
            ),
            "budget": (
                None if self.budget is None else self.budget.to_dict()
            ),
            "invalid_entries": [dict(entry) for entry in self._invalid]
            + [dict(entry) for entry in self._index_errors],
            "publication": {
                "no_pickle": True,
                "json_only_manifests": True,
                "path_confined": True,
                "commit_is_last_rename": True,
                "index_reconstructible": True,
            },
        }

    # -- M1 registry hand-off -------------------------------------------
    def commit_and_release(
        self,
        registry: Any,
        cell_id: str,
        *,
        attempt_id: str,
        backend: str | None = None,
        plan_generation: int = 0,
    ) -> dict:
        """Commit a registry-owned cell, then release its large arrays.

        Order is mandatory: the registry payload is released ONLY after the
        commit is published and verified. The release is idempotent and keeps a
        light descriptor so counts stay honest.
        """
        payload = registry.get(cell_id)
        if payload is None:
            raise StoreNotFoundError(
                f"registry holds no payload for cell {cell_id!r}"
            )
        receipt = self.commit_cell(
            payload,
            attempt_id=attempt_id,
            backend=backend,
            plan_generation=plan_generation,
        )
        release = registry.release_payload(
            cell_id, expected_payload_sha256=receipt.payload_sha256
        )
        return {
            "commit": receipt.to_dict(),
            "release": release,
        }

    def ordered_payloads(
        self, registry: Any, cell_ids: Sequence[str]
    ) -> list[Mapping[str, Any]]:
        """Canonical-order payloads, reloading released cells from disk.

        Cells still held by the registry are returned untouched (no copy);
        released cells are read back from their verified commit. This is NOT a
        bounded-assembly implementation (M5); it preserves the exact arrays the
        run already produced.
        """
        out: list[Mapping[str, Any]] = []
        for cell_id in cell_ids:
            meta = registry.metadata(cell_id)
            if meta is None:
                raise StoreNotFoundError(
                    f"cell {cell_id!r} is absent from the registry and the store"
                )
            if meta.get("released"):
                payload = self.load_payload(
                    cell_id, meta.get("attempt_id")
                )
                out.append(payload)
            else:
                payload = registry.get(cell_id)
                if payload is None:
                    raise StoreNotFoundError(
                        f"cell {cell_id!r} is neither retained nor released"
                    )
                out.append(payload)
        return out


def _dtype_itemsize(token: str) -> int:
    import numpy as np

    return int(np.dtype(token).itemsize)


def _prod(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _is_disk_full(exc: BaseException) -> bool:
    import errno as _errno

    code = getattr(exc, "errno", None)
    if code in (
        getattr(_errno, "ENOSPC", 28),
        getattr(_errno, "EDQUOT", 122),
    ):
        return True
    winerror = getattr(exc, "winerror", None)
    return winerror in (39, 112)  # ERROR_HANDLE_DISK_FULL / ERROR_DISK_FULL
