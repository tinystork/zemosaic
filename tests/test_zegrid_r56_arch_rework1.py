"""ZM-ZEGRID-R56 ARCH REWORK-1 — product cleanup adapter (F2).

The product passes INDEXED pipeline tasks ``(index, cell_task)`` but the nested
``_cleanup_task`` previously read ``item[3]`` directly, saw a 2-tuple and
silently skipped. Credits were then released while the partial cache remained on
disk. These tests compile the EXACT product callback from the source AST (not a
rewritten equivalent), run the real ``BoundedPipeline`` with real files, and
assert:

* the indexed task is decoded and ONLY the failed cell's cache is removed;
* deletion happens BEFORE the disk credit is released;
* the primary exception survives;
* a failed deletion (including ``_safe_rmtree`` returning False) is REPORTED,
  with the residual file left on disk (never a false success).
"""

from __future__ import annotations

import ast
import copy
import threading
import time
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import parallel as zpar


def _product_cleanup_callback(*, safe_rmtree=None):
    """Compile the EXACT nested ``_cleanup_task`` from the product source."""
    tree = ast.parse(Path(zz.__file__).read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_cleanup_task"
    )
    namespace = {"_safe_rmtree": safe_rmtree or zz._safe_rmtree}
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), zz.__file__, "exec"), namespace)
    return namespace["_cleanup_task"]


def _cell_task(cache_dir):
    return (
        "cell-r0c0",
        "patch",
        "membership",
        Path(cache_dir),
        "stream",
        100,
        "bounded",
    )


def _pipeline(*, cache_dir, callback, produce, tasks=None):
    indexed = [(0, _cell_task(cache_dir))] if tasks is None else tasks
    return zpar.BoundedPipeline(
        tasks=indexed,
        produce=produce,
        consume=lambda payload: payload,
        release=lambda _payload: None,
        prefetch_units=0,
        host_budget_bytes=100,
        device_budget_bytes=1,
        disk_budget_bytes=100,
        disk_enabled=True,
        host_reserve_bytes=lambda _task: 100,
        disk_reserve_bytes=lambda _task: 100,
        cleanup_task=callback,
    )


def test_product_cleanup_removes_indexed_partial_cache_before_credit_release(
    tmp_path,
):
    cache = tmp_path / "cell-cache"
    holder = {}
    observed = {}

    def observing_rmtree(path, progress_callback=None):
        # The disk credit is STILL held while the partial cache is deleted.
        observed["disk_used_before_delete"] = int(
            holder["pipe"].telemetry()["ledger"]["disk_used_bytes"]
        )
        observed["existed_before_delete"] = Path(path).exists()
        return zz._safe_rmtree(path, progress_callback)

    callback = _product_cleanup_callback(safe_rmtree=observing_rmtree)

    def produce(item):
        directory = Path(item[1][3])
        directory.mkdir()
        (directory / "partial.npy").write_bytes(b"partial payload")
        raise RuntimeError("injected partial production failure")

    pipe = _pipeline(cache_dir=cache, callback=callback, produce=produce)
    holder["pipe"] = pipe
    with pytest.raises(zpar.PipelineError) as excinfo:
        pipe.run()

    # Primary cause preserved, not replaced by an accounting error.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "injected partial production failure" in str(excinfo.value)
    # Cleanup ran BEFORE the lease was released, on the real file.
    assert observed == {
        "disk_used_before_delete": 100,
        "existed_before_delete": True,
    }
    assert not cache.exists()
    telemetry = pipe.telemetry()
    assert telemetry["ledger"]["balanced"] is True
    assert telemetry["ledger"]["disk_used_bytes"] == 0
    assert telemetry["cleanup_failures"] == []


def test_product_cleanup_failure_is_reported_and_residue_is_honest(tmp_path):
    cache = tmp_path / "cell-cache"
    callback = _product_cleanup_callback(safe_rmtree=lambda *_a, **_kw: False)

    def produce(item):
        directory = Path(item[1][3])
        directory.mkdir()
        (directory / "partial.npy").write_bytes(b"partial payload")
        raise RuntimeError("injected partial production failure")

    pipe = _pipeline(cache_dir=cache, callback=callback, produce=produce)
    with pytest.raises(zpar.PipelineError) as excinfo:
        pipe.run()

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    # The deletion FAILED: the file must still exist and the failure must be
    # visible in telemetry (never a silent success).
    assert (cache / "partial.npy").exists()
    telemetry = pipe.telemetry()
    assert telemetry["cleanup_failures"]
    assert "partial cache cleanup FAILED" in telemetry["cleanup_failures"][0]
    assert str(cache) in telemetry["cleanup_failures"][0]
    assert telemetry["ledger"]["balanced"] is True


def test_product_cleanup_runs_on_cancelled_partial_production(tmp_path):
    cache = tmp_path / "cell-cache"
    callback = _product_cleanup_callback()
    started = threading.Event()

    def produce(item):
        directory = Path(item[1][3])
        directory.mkdir()
        (directory / "partial.npy").write_bytes(b"partial payload")
        started.set()
        # Wait for the cancellation, then fail inside the producer so the
        # cleanup path runs (same path as partial production).
        time.sleep(0.05)
        raise zpar.PipelineCancelled("cancelled during partial production")

    pipe = _pipeline(cache_dir=cache, callback=callback, produce=produce)

    def _cancel():
        assert started.wait(2.0)
        pipe.cancel("test_cancel_partial_production")

    canceller = threading.Thread(target=_cancel, daemon=True)
    canceller.start()
    with pytest.raises((zpar.PipelineCancelled, zpar.PipelineError)):
        pipe.run()
    canceller.join(timeout=2.0)
    assert not canceller.is_alive()
    assert not cache.exists()
    assert pipe.telemetry()["ledger"]["balanced"] is True
