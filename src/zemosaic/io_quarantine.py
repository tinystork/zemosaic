"""ZM-IO-REJECT-01 — legacy source-quarantine contract (shared, dependency-free).

The historical ZeMosaic pipeline moved terminally unusable source frames into
``unaligned_by_zemosaic`` *next to the source itself* (the legacy call passed
``Path(file_path).parent`` as the owner root). That sibling directory is already
excluded by the classic scanner and by :mod:`zemosaic.input_inventory`, so a
rejected source can never be re-ingested by a later run.

This module restores that contract as ONE shared implementation used by both the
worker and the ZeGrid product path. It deliberately has no ZeMosaic imports (no
import cycle) and performs:

* the exact legacy destination name (:data:`UNALIGNED_DIRNAME`);
* a no-overwrite move (a pre-existing destination is NEVER replaced);
* deterministic collision handling (``name__1.fit``, ``name__2.fit``, ...);
* idempotent, thread-safe and process-safe parent-side calls (atomic
  ``link``/``O_EXCL`` claim, plus a process-wide lock for the common parent);
* a structured :class:`QuarantineResult` (status + destination + source).

It also defines the structured errors used when a frame payload cannot be
decoded, so the caller can preserve the path and the root cause instead of
collapsing them into a generic ``RuntimeError``.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

#: Exact historical quarantine directory name. NEVER rename this: the scanner
#: exclusion and user expectations are tied to it.
UNALIGNED_DIRNAME = "unaligned_by_zemosaic"

#: Structured statuses returned by :func:`move_to_unaligned`.
STATUS_MOVED = "moved"
STATUS_MOVED_RENAMED = "moved_renamed"
STATUS_ALREADY_MOVED = "already_moved"
STATUS_SKIPPED_EXCLUDED = "skipped_excluded"
STATUS_MISSING = "missing"
STATUS_FAILED = "failed"

#: Upper bound on deterministic collision suffixes (guards a pathological dir).
MAX_COLLISION_SUFFIX = 10_000

_QUARANTINE_LOCK = threading.RLock()


@dataclass(frozen=True)
class QuarantineResult:
    """Structured outcome of one quarantine attempt."""

    status: str
    source: str
    destination: str | None = None
    collision_index: int = 0
    detail: str | None = None

    @property
    def moved(self) -> bool:
        return self.status in (STATUS_MOVED, STATUS_MOVED_RENAMED, STATUS_ALREADY_MOVED)

    def as_tuple(self) -> tuple[str, Path | None]:
        """Legacy worker-facing ``(status, destination_path)`` shape."""
        return self.status, (Path(self.destination) if self.destination else None)


class FramePayloadError(RuntimeError):
    """A source frame payload cannot be decoded (structured, cause-preserving).

    ``path`` identifies the offending file; ``primary_error`` and ``raw_error``
    keep the two loader attempts as text so the information survives a process
    boundary (``multiprocessing`` pickles exceptions by ``args`` only). The
    original exception chain is preserved in the raising process through
    ``raise ... from raw_exc``.
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        primary_error: str | None = None,
        raw_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.primary_error = primary_error
        self.raw_error = raw_error

    def __reduce__(self):
        return (
            type(self),
            (str(self), self.path, self.primary_error, self.raw_error),
        )


class FramePayloadQuarantineError(RuntimeError):
    """Terminal run error: one source was quarantined; a new plan/run is needed.

    Raised at the ZeGrid parent boundary after the source was moved into
    ``unaligned_by_zemosaic``. The run must stop before producing any science or
    manifest: continuing with the current plan/membership would be incoherent.
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        destination: str | None = None,
        move_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.destination = destination
        self.move_status = move_status

    def __reduce__(self):
        return (
            type(self),
            (str(self), self.path, self.destination, self.move_status),
        )


def describe_exception(exc: BaseException) -> str:
    """Stable ``Type: message`` text for logging and exception payloads."""
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:  # pragma: no cover - defensive
        return repr(exc)


def _resolved(path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except Exception:
        return Path(path)


def _is_in_quarantine(path: Path) -> bool:
    return UNALIGNED_DIRNAME in set(path.parts)


def _collision_name(name: str, index: int) -> str:
    if index <= 0:
        return name
    p = Path(name)
    return f"{p.stem}__{index}{p.suffix}"


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except Exception:
        return False


def _claim_and_move(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst`` without ever replacing an existing destination.

    ``os.link`` gives an atomic no-overwrite claim on POSIX filesystems; the
    ``O_EXCL`` reservation covers platforms/filesystems without hard links.
    Raises ``FileExistsError`` when ``dst`` already exists.
    """
    try:
        os.link(src, dst)
    except FileExistsError:
        raise
    except OSError:
        # Cross-device or hard-link-unsupported filesystem: reserve the exact
        # destination name atomically, then move onto our own reservation.
        fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        try:
            shutil.move(str(src), str(dst))
        except Exception:
            try:
                dst.unlink()
            except Exception:
                pass
            raise
    else:
        # Hard-link claim succeeded: remove the source name. A concurrent caller
        # may already have removed it (same inode), which is a success for us.
        try:
            src.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            raise


def move_to_unaligned(
    src_path,
    owner_root,
    *,
    logger: logging.Logger | None = None,
    max_suffix: int = MAX_COLLISION_SUFFIX,
) -> QuarantineResult:
    """Move ``src_path`` into ``<owner_root>/unaligned_by_zemosaic``.

    Legacy callers pass the SOURCE'S IMMEDIATE PARENT as ``owner_root`` (the
    historical ``_move_to_unaligned_safe(file, Path(file).parent)`` contract),
    which keeps a nested frame in its own folder instead of relocating it to the
    global input root.

    Idempotent and collision-safe. A source already under the quarantine folder
    is left untouched. An existing destination is never overwritten: the move
    uses a deterministic ``stem__N.suffix`` name instead.
    """
    log = logger or logging.getLogger(__name__)
    src = _resolved(src_path)
    root = _resolved(owner_root)

    if _is_in_quarantine(src):
        return QuarantineResult(STATUS_SKIPPED_EXCLUDED, str(src))

    target_dir = root / UNALIGNED_DIRNAME

    with _QUARANTINE_LOCK:
        if not src.exists():
            if target_dir.exists():
                # Is the source already present under its own name (or a known
                # collision name)? Report the effective legacy destination.
                direct = target_dir / src.name
                if direct.exists():
                    return QuarantineResult(
                        STATUS_ALREADY_MOVED, str(src), str(direct)
                    )
            return QuarantineResult(STATUS_MISSING, str(src))

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            try:
                log.warning(
                    "Could not create '%s' under %s: %s",
                    UNALIGNED_DIRNAME,
                    root,
                    exc,
                )
            except Exception:
                pass
            return QuarantineResult(
                STATUS_FAILED, str(src), str(target_dir), detail=describe_exception(exc)
            )

        for index in range(0, max(1, int(max_suffix)) + 1):
            dst = target_dir / _collision_name(src.name, index)
            if dst.exists():
                if _same_file(src, dst):
                    # A concurrent caller already hard-linked/renamed the same
                    # source under this name. The move is effectively done.
                    try:
                        src.unlink()
                    except Exception:
                        pass
                    return QuarantineResult(
                        STATUS_ALREADY_MOVED, str(src), str(dst), collision_index=index
                    )
                continue
            try:
                _claim_and_move(src, dst)
            except FileExistsError:
                # Lost a race to another process: try the next deterministic name.
                continue
            except FileNotFoundError:
                return QuarantineResult(
                    STATUS_ALREADY_MOVED, str(src), str(dst), collision_index=index
                )
            except Exception as exc:
                try:
                    log.warning(
                        "Move to '%s' failed for %s: %s",
                        UNALIGNED_DIRNAME,
                        src,
                        exc,
                    )
                except Exception:
                    pass
                return QuarantineResult(
                    STATUS_FAILED,
                    str(src),
                    str(dst),
                    collision_index=index,
                    detail=describe_exception(exc),
                )
            status = STATUS_MOVED if index == 0 else STATUS_MOVED_RENAMED
            try:
                log.info("Moved to '%s': %s", UNALIGNED_DIRNAME, dst)
            except Exception:
                pass
            return QuarantineResult(status, str(src), str(dst), collision_index=index)

    return QuarantineResult(
        STATUS_FAILED,
        str(src),
        str(target_dir / src.name),
        detail=f"collision suffix limit reached ({max_suffix})",
    )
