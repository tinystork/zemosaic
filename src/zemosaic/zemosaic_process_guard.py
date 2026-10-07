"""Parent-process watchdog for NON-DAEMON worker processes (ZM-ZEGRID-R16).

The product spawns its ZeGrid worker as a ``multiprocessing.Process``.  R16 makes
that worker NON-DAEMON (``daemon=False``) so it may legally create its own child
processes (the R12 process pool — a daemonic process may NOT start children, so
R13 had to fall back to a THREAD pool whose speedup is GIL-capped).

A non-daemon process is NOT auto-terminated when its parent (the GUI) exits, so
``daemon=False`` alone would leak an orphan worker if the GUI dies abruptly
(crash / kill / WM close).  This module restores the "dies with the GUI" property
WITHOUT the daemon flag: a small DAEMON *thread* (threads are always allowed
inside any process) polls the parent pid/process and, when the parent disappears,
signals a clean stop/exit.

Tradeoff (documented):
  * A parent-watchdog thread CANNOT replace ``multiprocessing``'s automatic
    daemon teardown with perfect timing: it is a POLLER, so there is up to one
    ``poll_interval`` of latency before the worker notices the parent is gone.
  * On POSIX the child is reparented to init/subreaper the instant the parent
    dies, so ``os.getppid()`` changes; on Windows ``getppid()`` is frozen at
    process creation and NEVER updates, so we poll the parent process handle
    instead (never signals).  The PID-reuse race on Windows is a documented
    residual risk (a recycled pid can look alive); psutil mitigates but does not
    eliminate it.
  * The clean stop is signalled by ``_thread.interrupt_main()`` (raises
    ``KeyboardInterrupt`` in the main thread -> the worker's existing
    ``except KeyboardInterrupt`` path flushes the queue and returns cleanly).
    If the main thread is stuck in a long C call and does not unwind within the
    grace window, a hard ``os._exit(1)`` guarantees no orphan is ever left
    (skipping Python-level cleanup, but a dead parent cannot read the queue
    anyway).

Detection is FAIL-OPEN: when we cannot prove the parent is dead we assume it is
alive, so the watchdog can never spuriously kill a healthy worker.
"""

from __future__ import annotations

import os
import threading

# Grace period after signalling KeyboardInterrupt before the hard-exit fallback.
_GRACE_S = 5.0


def _windows_parent_alive(pid: int) -> bool:
    """Windows parent-alive check via the process handle (fail-open)."""
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        try:
            return bool(psutil.Process(pid).is_running())
        except Exception:
            return False
    except Exception:
        pass
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == STILL_ACTIVE
    except Exception:
        pass
    return True  # fail-open


def _parent_alive(parent_pid: int) -> bool:
    """Best-effort: is the process ``parent_pid`` still running? (fail-open)."""
    if parent_pid <= 0:
        return True  # unknown parent -> never kill

    # Primary: psutil process poll (works for ANY pid, not just our real parent;
    # ``is_running()`` is False for zombies, so a reaped-but-lingering parent
    # still counts as dead).
    try:
        import psutil

        if not psutil.pid_exists(int(parent_pid)):
            return False
        try:
            return bool(psutil.Process(int(parent_pid)).is_running())
        except Exception:
            return False
    except Exception:
        pass

    if os.name == "nt":
        return _windows_parent_alive(int(parent_pid))

    # POSIX fallback (no psutil): the child is reparented when the parent dies,
    # so a change in getppid() is a reliable death signal; otherwise a reachable
    # pid (kill 0) means alive.
    try:
        if os.getppid() == int(parent_pid):
            return True
    except Exception:
        pass
    try:
        os.kill(int(parent_pid), 0)
        return True
    except OSError:
        return False
    except Exception:
        return True


def _default_parent_exit() -> None:
    """Signal a clean stop, then hard-exit if the main thread lingers."""
    import _thread

    _thread.interrupt_main()
    try:
        threading.main_thread().join(timeout=_GRACE_S)
    except Exception:
        pass
    try:
        if threading.main_thread().is_alive():
            os._exit(1)
    except Exception:
        os._exit(1)


def install_parent_watchdog(
    parent_pid: int | None = None,
    *,
    poll_interval: float = 1.0,
    on_parent_exit: object = None,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Start a daemon thread that stops THIS process when its parent disappears.

    ``parent_pid`` defaults to ``os.getppid()`` (the real parent).  ``poll_interval``
    is the liveness poll period in seconds.  ``on_parent_exit`` (optional callable)
    overrides the default clean-stop action (used by tests).  Returns the started
    daemon thread (never blocks; a missing/unknown parent is fail-open).
    """
    pid = int(parent_pid) if parent_pid is not None else int(os.getppid())
    stop = stop_event if stop_event is not None else threading.Event()
    exit_fn = on_parent_exit if callable(on_parent_exit) else _default_parent_exit
    interval = max(0.05, float(poll_interval))

    def _watch() -> None:
        while not stop.is_set():
            if not _parent_alive(pid):
                try:
                    exit_fn()
                except BaseException:  # noqa: BLE001 - watchdog must never raise
                    pass
                return
            stop.wait(interval)

    thread = threading.Thread(target=_watch, name="ParentWatchdog", daemon=True)
    thread.start()
    return thread
