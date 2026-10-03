"""Real ``spawn`` worker-process lifecycle witness (ZM-ARCH-WITNESS-SPAWN).

This test spawns a *real* child process whose target is the actual
package-qualified ``zemosaic.zemosaic_worker.run_hierarchical_mosaic_process``
callable, using ``multiprocessing.get_context("spawn")`` and a real context
``Queue``.  It is the only test in the suite that proves the production worker
target is picklable/importable under the spawn start method and that the
wrapper's queue protocol is observable across a genuine process boundary.

Scope and limits
----------------
* **Bootstrap/error-path witness only.**  The child is deliberately invoked with
  no scientific arguments, so ``run_hierarchical_mosaic`` raises a ``TypeError``
  for missing required positional arguments inside the wrapper.  This pins the
  current caught-error protocol (``PROCESS_ERROR`` immediately followed by
  ``PROCESS_DONE``) and a clean child exit.  It does **not** witness a successful
  scientific run, GPU/solver work, or GUI end-to-end parity.
* **Not a fake/in-process double.**  The existing
  ``test_dispatch_propagation_witness.py`` exercises the same wrapper in-process
  with a monkeypatched downstream stub; ``test_zesolver_filter_handoff_hg2.py``
  spawns a *helper* target to prove a payload pickles.  This test complements
  both by spawning the *production* target itself.
* ``crash_breadcrumbs_mode="off"``, isolated HOME/XDG/cwd, and no heartbeat
  thread, so the child writes no crash-breadcrumb / heartbeat / state files.
  The only file the child may create is the worker log under the isolated
  XDG config dir (an import-time logger side effect, not this test's concern).
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_worker as zw  # noqa: E402


def _spawn_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("spawn")
    except Exception as exc:  # pragma: no cover - no known platform lacks spawn
        pytest.skip(f"multiprocessing 'spawn' context unavailable: {exc}")


@pytest.fixture
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Isolate HOME/XDG/cwd and make the src-layout package importable in the child.

    The spawned child is a fresh interpreter that only inherits ``os.environ``
    plus the parent ``sys.path`` propagated by ``multiprocessing.spawn``; setting
    ``PYTHONPATH`` explicitly is defense in depth so the child can always import
    ``zemosaic`` regardless of how pytest was invoked.
    """
    work = tmp_path / "work"
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    cache = tmp_path / "cache"
    data = tmp_path / "data"
    for d in (work, home, xdg, cache, data):
        d.mkdir()

    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(SRC) + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
    )
    return tmp_path, work


def _drain_queue(q: multiprocessing.Queue, timeout: float = 10.0) -> list[object]:
    """Read queue messages until PROCESS_DONE (or the queue goes idle)."""
    messages: list[object] = []
    while True:
        try:
            item = q.get(timeout=timeout)
        except Exception:
            break
        messages.append(item)
        if isinstance(item, tuple) and item and item[0] == "PROCESS_DONE":
            break
    return messages


def test_real_spawn_worker_bootstrap_error_protocol(_isolated_env: tuple[Path, Path]) -> None:
    """Spawn the real worker target and pin the caught-error queue protocol."""
    tmp_path, work = _isolated_env
    ctx = _spawn_context()

    q = ctx.Queue()
    p = ctx.Process(
        target=zw.run_hierarchical_mosaic_process,
        args=(q,),
        kwargs={"crash_breadcrumbs_mode": "off"},
        name="ZeMosaicSpawnWitness",
    )

    try:
        p.start()
        p.join(timeout=120)
    finally:
        # Bounded cleanup: only terminate/kill if the child is still alive.
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
        assert not p.is_alive(), "spawned child leaked past cleanup"

    # The wrapper catches the deliberate missing-args TypeError, so the child
    # must exit cleanly with code 0 (no uncaught exception, no import/pickle crash).
    assert p.exitcode == 0, f"expected clean exit (0), got exitcode={p.exitcode}"

    messages = _drain_queue(q)
    keys = [m[0] if isinstance(m, tuple) and m else None for m in messages]

    assert "PROCESS_ERROR" in keys, f"expected PROCESS_ERROR in queue, got {keys!r}"
    assert "PROCESS_DONE" in keys, f"expected PROCESS_DONE in queue, got {keys!r}"

    err_idx = keys.index("PROCESS_ERROR")
    done_idx = keys.index("PROCESS_DONE")
    assert err_idx < done_idx, "PROCESS_ERROR must precede PROCESS_DONE (wrapper finally block)"

    error_payload = messages[err_idx]
    assert isinstance(error_payload, tuple) and len(error_payload) == 4
    assert error_payload[2] == "ERROR"

    error_text = str(error_payload[3]["error"])
    # The error must identify the deliberate missing-required-args failure,
    # proving the child imported and *ran* the real target (not an
    # import/pickle/ModuleNotFound failure, which would abort before the wrapper
    # ever emitted PROCESS_ERROR).
    lowered = error_text.lower()
    assert "missing" in lowered, f"error does not describe missing args: {error_text!r}"
    assert "input_folder" in error_text, f"error does not name a required arg: {error_text!r}"
    for bogus in ("modulenotfound", "pickle", "cannot import", "importerror"):
        assert bogus not in lowered, f"error looks like an import/pickle failure: {error_text!r}"

    # crash_breadcrumbs_mode="off" must suppress breadcrumb/heartbeat/state files
    # anywhere in the isolated tree.
    artifacts = [
        p
        for name in ("worker_crash_breadcrumbs.jsonl", "worker_last_state.json")
        for p in tmp_path.rglob(name)
    ]
    assert artifacts == [], f"crash breadcrumb/state files leaked: {artifacts}"

    # The working directory itself must stay empty (no output dir was created).
    assert list(work.iterdir()) == [], f"unexpected files in cwd: {list(work.iterdir())}"

    # Queue cleanup: close the write end and join the feeder thread.
    q.close()
    q.join_thread()
