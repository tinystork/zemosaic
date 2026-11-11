"""ZM-ZEGRID-FD-EXHAUSTION — descriptor-limit hardening and cache FD budget.

Production background
---------------------
A real 527-frame ZeGrid GPU run died at cell ``r0000c0001`` with::

    OSError: [Errno 24] Too many open files

because the aligned cache stores **2 ``.npy`` files per frame** (RGB float32 +
support bool) and the dense file provider opened **one ``np.load(mmap_mode="r")``
mapping per file at construction**: ~1054 descriptors for ONE cell, in a GUI
process whose soft ``RLIMIT_NOFILE`` was 1024 (hard 524288). That EMFILE is
deterministic, not bad luck.

This module provides the two portable pieces:

* **A — limit hardening** (:func:`ensure_nofile_headroom`): raise the soft
  ``RLIMIT_NOFILE`` toward ``min(hard, FD_TARGET_CEILING)`` at ZeGrid entry /
  worker start. POSIX-guarded, never lowers an existing limit, never crashes on
  denial, and returns an additive record of the effective soft/hard values for
  the run diagnostics.
* **B — honest estimate** (:func:`estimate_fd_budget`,
  :func:`assert_cell_fd_budget`): the worst-cell descriptor budget, computed
  from the real frame count, the 2-files-per-frame cache layout and the bounded
  number of mappings the provider may hold. If it genuinely cannot fit the
  effective soft limit, it raises :class:`FdLimitExceeded` **before** any cache
  write or stack work, with an actionable message — never a mid-run EMFILE.

Nothing here changes science, scheduling, RAM/VRAM accounting or the R56 credit
contract; it only makes descriptor usage bounded and diagnosable.
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = [
    "DEFAULT_MAX_OPEN_CACHE_FILES",
    "FD_SAFETY_MARGIN",
    "FD_TARGET_CEILING",
    "FILES_PER_FRAME",
    "FdLimitExceeded",
    "assert_cell_fd_budget",
    "count_open_fds",
    "default_base_fds",
    "ensure_nofile_headroom",
    "estimate_fd_budget",
    "raise_nofile_soft_limit",
    "read_nofile_limits",
    "reset_cache",
]

LOGGER = logging.getLogger(__name__)

#: Hard ceiling we are willing to raise the soft limit to. Deliberately far
#: below ``RLIM_INFINITY``: a bug that leaks descriptors must still hit a wall.
FD_TARGET_CEILING = 131072

#: Every aligned-cache frame is 2 files (RGB float32 + support bool).
FILES_PER_FRAME = 2

#: Hard cap on simultaneously open aligned-cache files in ONE provider. 64 files
#: = 32 frames; a cell of any size costs at most this many descriptors.
DEFAULT_MAX_OPEN_CACHE_FILES = 64

#: Descriptors the process already needs beyond the cache (logs, sockets, FITS
#: handles, CUDA/GUI descriptors, libraries). Reserved on top of the measured
#: open count so the estimate is conservative, not exact.
FD_SAFETY_MARGIN = 32

#: Worst-case descriptors a cache-build worker holds at once: the decoded source
#: handle(s), the two output ``.npy`` handles and the hashing pass.
FDS_PER_CACHE_WORKER = 6

_FALLBACK_BASE_FDS = 64  # used when /proc is unavailable (non-Linux)

_CACHED_RECORD: dict | None = None


class FdLimitExceeded(RuntimeError):
    """The worst-cell descriptor budget cannot fit the effective soft limit.

    Raised BEFORE any cache write or stack work so the run never dies mid-cell
    with ``EMFILE``.
    """


# ---------------------------------------------------------------------------
# Platform seam
# ---------------------------------------------------------------------------

def _resource_module():
    """Return the ``resource`` module on POSIX, else ``None`` (never raises)."""
    if os.name != "posix":
        return None
    try:
        import resource  # noqa: PLC0415 - POSIX-only, guarded on purpose
    except Exception:  # noqa: BLE001 - absence is a supported state
        return None
    return resource


def _as_limit(value) -> int | None:
    """Normalise a ``getrlimit`` value; ``RLIM_INFINITY`` -> ``None``."""
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue < 0:
        return None
    return ivalue


def read_nofile_limits() -> dict:
    """Read the current ``RLIMIT_NOFILE`` soft/hard pair (never raises)."""
    res = _resource_module()
    record = {
        "platform": sys.platform,
        "supported": res is not None,
        "soft": None,
        "hard": None,
        "infinite_hard": False,
        "error": None,
        "reason": "",
    }
    if res is None:
        record["reason"] = "resource module unavailable (non-POSIX platform)"
        return record
    try:
        soft_raw, hard_raw = res.getrlimit(res.RLIMIT_NOFILE)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail a run
        record["supported"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["reason"] = "getrlimit(RLIMIT_NOFILE) failed"
        return record
    soft = _as_limit(soft_raw)
    hard = _as_limit(hard_raw)
    record["soft"] = soft
    record["hard"] = hard
    record["infinite_hard"] = hard is None
    record["reason"] = "observed"
    return record


def raise_nofile_soft_limit(target: int | None = None) -> dict:
    """Raise the soft ``RLIMIT_NOFILE`` toward ``min(hard, target)``.

    Contract (never violated, in order of precedence):

    * never lowers the current soft limit;
    * never exceeds the hard limit;
    * never raises on denial — the current limit is recorded and the caller
      continues (the FD budget guard is the fail-closed gate, not this);
    * non-POSIX platforms are an explicit recorded no-op.

    Returns an additive record suitable for the run diagnostics::

        {platform, supported, target, soft_before, hard, soft_effective,
         raised, attempted, reason, error}
    """
    res = _resource_module()
    limits = read_nofile_limits()
    requested = FD_TARGET_CEILING if target is None else int(target)
    record = {
        "platform": sys.platform,
        "supported": res is not None,
        "target": requested,
        "soft_before": limits["soft"],
        "hard": limits["hard"],
        "soft_effective": limits["soft"],
        "raised": False,
        "attempted": False,
        "reason": "",
        "error": None,
    }
    if res is None:
        record["reason"] = "no-op: resource module unavailable (non-POSIX platform)"
        return record
    soft = limits["soft"]
    hard = limits["hard"]
    if hard is not None:
        effective_target = min(requested, int(hard))
    else:
        effective_target = requested
    record["target"] = int(effective_target)
    if soft is None:
        record["reason"] = "no-op: soft limit already unlimited"
        return record
    if soft >= effective_target:
        record["reason"] = (
            f"no-op: soft limit {soft} already at/above target {effective_target}"
        )
        return record
    if hard is not None and soft >= int(hard):
        record["reason"] = f"no-op: soft limit {soft} already at the hard limit {hard}"
        return record
    record["attempted"] = True
    try:
        res.setrlimit(
            res.RLIMIT_NOFILE,
            (int(effective_target), res.RLIM_INFINITY if hard is None else int(hard)),
        )
    except (OSError, ValueError, PermissionError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["reason"] = "raise denied; continuing with the current limit"
        record["soft_effective"] = read_nofile_limits()["soft"]
        LOGGER.warning(
            "ZeGrid: could not raise RLIMIT_NOFILE soft limit to %s (%s); "
            "continuing with %s",
            effective_target,
            record["error"],
            record["soft_effective"],
        )
        return record
    after = read_nofile_limits()
    record["soft_effective"] = after["soft"]
    record["hard"] = after["hard"]
    record["raised"] = bool(
        after["soft"] is None
        or (soft is not None and int(after["soft"]) > int(soft))
    )
    record["reason"] = (
        f"raised soft limit {soft} -> {after['soft']}"
        if record["raised"]
        else f"setrlimit accepted but soft limit stayed {after['soft']}"
    )
    return record


def ensure_nofile_headroom(*, target: int | None = None, force: bool = False) -> dict:
    """Attempt the (idempotent, cached) soft-limit raise once per process."""
    global _CACHED_RECORD
    if _CACHED_RECORD is not None and not force:
        return dict(_CACHED_RECORD)
    record = raise_nofile_soft_limit(target)
    _CACHED_RECORD = dict(record)
    return dict(record)


def reset_cache() -> None:
    """Forget the cached headroom record (tests / a deliberately re-checked run)."""
    global _CACHED_RECORD
    _CACHED_RECORD = None


def effective_soft_limit(*, target: int | None = None) -> int | None:
    """Effective soft ``RLIMIT_NOFILE`` after the defensive raise (``None``=unlimited)."""
    ensure_nofile_headroom(target=target)
    return read_nofile_limits()["soft"]


# ---------------------------------------------------------------------------
# FD budget (B)
# ---------------------------------------------------------------------------

def count_open_fds() -> int | None:
    """Number of descriptors this process holds now (Linux ``/proc``), else ``None``."""
    try:
        entries = os.listdir("/proc/self/fd")
    except Exception:  # noqa: BLE001 - best effort only
        return None
    # The listing itself held one descriptor; it is closed before we return.
    return max(0, len(entries) - 1)


def default_base_fds() -> int:
    """Conservative descriptor baseline for the current process."""
    open_now = count_open_fds()
    if open_now is None:
        return _FALLBACK_BASE_FDS + FD_SAFETY_MARGIN
    return int(open_now) + FD_SAFETY_MARGIN


def estimate_fd_budget(
    n_frames: int,
    *,
    files_per_frame: int = FILES_PER_FRAME,
    max_open_files: int = DEFAULT_MAX_OPEN_CACHE_FILES,
    workers: int = 1,
    base_fds: int | None = None,
) -> dict:
    """Worst-cell descriptor estimate with the bounded provider in place.

    ``required = base_fds + min(cache_files, max_open_files) + workers * per-worker``

    The cache holds ``frames x files_per_frame`` files on disk, but the bounded
    provider keeps at most ``max_open_files`` of them mapped at once, so the
    descriptor requirement is independent of the cell's frame count.
    """
    frames = max(0, int(n_frames))
    per_frame = max(1, int(files_per_frame))
    cache_files = frames * per_frame
    cap = max(0, int(max_open_files))
    resident = min(cache_files, cap)
    base = default_base_fds() if base_fds is None else max(0, int(base_fds))
    worker_count = max(0, int(workers))
    worker_fds = worker_count * FDS_PER_CACHE_WORKER
    required = int(base) + int(resident) + int(worker_fds)
    return {
        "frames": frames,
        "files_per_frame": per_frame,
        "cache_files": int(cache_files),
        "resident_cache_files": int(resident),
        "max_open_files": int(cap),
        "base_fds": int(base),
        "cache_workers": int(worker_count),
        "worker_fds": int(worker_fds),
        "required_fds": int(required),
        "bounded": bool(cache_files > resident),
        "fits": True,  # overwritten by the caller once the real limit is known
        "effective_soft": None,
        "hard": None,
    }


def _budget_message(context: str, est: dict, soft: int, hard: int | None) -> str:
    hard_txt = "unlimited" if hard is None else str(hard)
    return (
        f"ZeGrid descriptor budget exceeded BEFORE any work ({context}): this cell "
        f"needs {est['required_fds']} simultaneous descriptors — {est['cache_files']} "
        f"aligned-cache files ({est['frames']} frames x {est['files_per_frame']}), "
        f"at most {est['resident_cache_files']} held open at once by the bounded "
        f"cache provider, {est['base_fds']} already used by this process and "
        f"{est['worker_fds']} reserved for {est['cache_workers']} cache-build "
        f"worker(s) — but the effective RLIMIT_NOFILE soft limit is {soft} "
        f"(hard {hard_txt}). Nothing was written for this cell and no Cell was "
        f"committed. Raise the limit before the run (e.g. `ulimit -n 131072` in "
        f"the launching shell, or `LimitNOFILE=131072` in the systemd unit) and "
        f"rerun; already-committed cells stay reusable."
    )


def assert_cell_fd_budget(
    context: str,
    *,
    n_frames: int,
    files_per_frame: int = FILES_PER_FRAME,
    max_open_files: int = DEFAULT_MAX_OPEN_CACHE_FILES,
    workers: int = 1,
    base_fds: int | None = None,
    effective_soft: int | None = None,
) -> dict:
    """Fail closed BEFORE cache/stack work when the cell cannot fit its FDs.

    ``effective_soft`` overrides the observed limit (tests / controlled hosts).
    Returns the estimate enriched with the observed limits; raises
    :class:`FdLimitExceeded` with an actionable message when it does not fit.
    """
    limits = read_nofile_limits()
    soft = limits["soft"] if effective_soft is None else (
        None if effective_soft is None else int(effective_soft)
    )
    est = estimate_fd_budget(
        n_frames,
        files_per_frame=files_per_frame,
        max_open_files=max_open_files,
        workers=workers,
        base_fds=base_fds,
    )
    est["effective_soft"] = soft
    est["hard"] = limits["hard"]
    if soft is None:
        est["fits"] = True
        return est
    est["fits"] = bool(est["required_fds"] <= soft)
    if not est["fits"]:
        raise FdLimitExceeded(_budget_message(context, est, int(soft), limits["hard"]))
    return est
