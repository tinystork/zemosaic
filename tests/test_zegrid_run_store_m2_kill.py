"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 — crash/kill matrix (plan §6.2 / §7).

A real child process is SIGKILLed during each of the four write steps:

(a) temporary array write, (b) fsync, (c) commit publication (atomic rename),
(d) index update.

At every reopening: every visible commit validates, NO partial cell is ever
visible, and the index reconstructs to exactly the set of complete commits.

Linux uses ``SIGKILL``; on Windows the same test kills the child with
``TerminateProcess`` semantics (``Popen.kill``), so the matrix is executable on
both platforms. It is NOT a power-loss claim: rename+fsync ordering is the
guarantee, as documented in the M2 report.
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

from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

H, W = 8, 10
CORE = (2, 6, 2, 7)
STAGES = ("temp_write", "fsync", "commit_rename", "index")
ITERATIONS = 5

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

IDENTITY = zrs.build_run_identity(
    run_id="run-kill-matrix",
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
    memberships={"sha256": "1" * 64, "n_cells": 3},
    gauge={
        "mode": "per_cell",
        "sha256": "2" * 64,
        "reference_frame_id": "frame_0001.fits",
        "reference_frame_ids": ["frame_0001.fits"],
        "n_exclusions": 0,
    },
    science={"sha256": "3" * 64, "contract_version": "1", "params": {"k": 1}},
    dtypes=DTYPES,
    backend={"backend": "cpu", "tile_px": 8, "parity_qualified": False},
    execution_context={
        "available_ram_bytes": 4 * 2 ** 20,
        "workers": 1,
        "prefetch_units": 0,
        "host": "kill-matrix",
        "note": "runtime only",
    },
    created_unix_ns=1,
)

CHILD = r'''
import json, os, sys, time
sys.path.insert(0, {src!r})
from zemosaic.core.zegrid import run_store as zrs
import numpy as np

root = {root!r}
stage = {stage!r}
marker = {marker!r}
identity = json.loads(open({identity_path!r}, encoding="utf-8").read())

H, W = 8, 10
CORE = (2, 6, 2, 7)

def payload(cell_id, seed):
    rng = np.random.default_rng(seed)
    y0, y1, x0, x1 = CORE
    base = {{"science": rng.normal(size=(H, W, 3)).astype("<f4"),
             "w1": rng.normal(size=(H, W)).astype("<f4"),
             "w2": rng.normal(size=(H, W)).astype("<f4"),
             "neff": rng.normal(size=(H, W)).astype("<f4"),
             "est": rng.normal(size=(H, W)).astype("<f4"),
             "valid": np.ones((H, W), bool),
             "surv": np.ones((H, W), "<i4")}}
    return {{
        "cell_id": cell_id, "status": "complete", "backend_used": "cpu",
        "n_frames": 1, "total_bytes": 10,
        "core": {{
            "science_core": base["science"][y0:y1, x0:x1, :],
            "estimator_weight_sum_core": base["est"][y0:y1, x0:x1],
            "support_w1_core": base["w1"][y0:y1, x0:x1],
            "support_w2_core": base["w2"][y0:y1, x0:x1],
            "n_eff_support_core": base["neff"][y0:y1, x0:x1],
            "valid_mask_core": base["valid"][y0:y1, x0:x1],
            "surviving_sample_count_core": base["surv"][y0:y1, x0:x1],
        }},
        "record": {{"cell_id": cell_id, "patch_bounds": [0, 0, W, H],
                   "patch_shape_hw": [H, W], "cell_gauge_mode": "per_cell",
                   "reference_frame_id": "frame_0001.fits",
                   "reference_frame_role": "cell", "excluded": []}},
        "patch": {{"bounds": (0, H, 0, W), "core_slice": CORE,
                  "science": base["science"], "valid_mask": base["valid"],
                  "surviving": base["surv"]}},
    }}

def hook(name, info):
    if name != stage:
        return
    import json as _json
    # Publish the handshake ATOMICALLY: the parent kills us the moment this
    # path exists, so a partially written file would be a test-side race.
    payload = _json.dumps({{"stage": name, "pid": os.getpid()}})
    tmp_marker = marker + ".part"
    with open(tmp_marker, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_marker, marker)
    # Hold the exact transaction point open until the parent kills us.
    while True:
        time.sleep(0.05)

store = zrs.RunCellStore.open(root, identity=identity)
# Two COMPLETE cells are durable before the injection is armed, so the test can
# assert "prior commits survive" as well as "no partial commit is visible".
for i in range(2):
    store.commit_cell(payload("r0000c00%02d" % i, i), attempt_id="a1",
                      backend="cpu", plan_generation=0)
store.fault_hook = hook
store.commit_cell(payload("r0001c0001", 77), attempt_id="a1",
                  backend="cpu", plan_generation=0)
print("COMPLETED_WITHOUT_KILL", flush=True)
'''


def _run_child(tmp_path, stage, iteration):
    root = tmp_path / f"run-{stage}-{iteration}"
    marker = tmp_path / f"marker-{stage}-{iteration}"
    identity_path = tmp_path / f"identity-{stage}-{iteration}.json"
    identity_path.write_text(json.dumps(IDENTITY), encoding="utf-8")
    script = CHILD.format(
        src=str(SRC), root=str(root), stage=stage, marker=str(marker),
        identity_path=str(identity_path),
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    try:
        deadline = time.time() + 60
        while not marker.exists() and time.time() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate()
                pytest.fail(
                    f"child exited before reaching {stage}: rc={proc.returncode} "
                    f"out={out!r} err={err!r}"
                )
            time.sleep(0.01)
        assert marker.exists(), f"child never reached the {stage} step"
        _kill(proc)
    finally:
        if proc.poll() is None:
            _kill(proc)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pytest.fail("child survived SIGKILL")
    handshake = None
    for _ in range(100):
        try:
            handshake = json.loads(marker.read_text(encoding="utf-8"))
            break
        except (ValueError, OSError):
            time.sleep(0.01)
    assert handshake is not None, "kill marker was never readable"
    return root, int(handshake["pid"])


def _kill(proc):
    """Kill the ACTUAL writer (process group on POSIX, tree on Windows).

    A Windows venv launcher has a different pid from the interpreter it starts;
    ``taskkill /T`` covers that tree. On POSIX the child owns its session, so the
    whole group (the real writer) is signalled.
    """
    if os.name == "nt":  # pragma: no cover - Windows path
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except Exception:
                pass


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("iteration", range(ITERATIONS))
def test_m2_kill_during_each_write_step_leaves_only_complete_commits(
    tmp_path, stage, iteration
):
    root, holder_pid = _run_child(tmp_path, stage, iteration)
    store = zrs.RunCellStore.open(root, identity=None)
    try:
        report = store.report()
        scan = store.scan()
        # 1. Every commit the store can see validates; no partial cell.
        for entry in scan["valid"]:
            assert entry["valid"] is True
        for entry in scan["invalid"]:
            # An invalid manifest object is allowed to EXIST as a torn temp,
            # but it must never be reported as a valid commit.
            assert entry.get("valid") is not True

        committed = sorted(store.index()["cells"])
        # The two pre-kill cells are always complete.
        for cell_id in ("r0000c0000", "r0000c0001"):
            verdict = store.verify_cell(cell_id)
            assert verdict["valid"] is True, verdict
        # The target cell is either fully published or absent — never partial.
        target = store.verify_cell("r0001c0001")
        if target["valid"]:
            payload = store.load_payload("r0001c0001")
            assert payload["core"]["science_core"].shape == (4, 5, 3)
        else:
            assert "no durable commit" in str(target["reason"]).lower()
        if stage == "index":
            # Publication happened; only the index may lag. Recovery must find it.
            assert target["valid"] is True
            assert "r0001c0001" in committed
        completeness = store.completeness(
            ["r0000c0000", "r0000c0001", "r0001c0001"]
        )
        assert completeness["invalid"] == []
        assert completeness["complete"] in (2, 3)
        assert completeness["complete"] == completeness["expected"] - len(
            completeness["missing"]
        )
        # 0. The dead owner recorded on disk is the ACTUAL writer that was
        #    killed (never merely a venv launcher pid).
        previous = (store.report().get("lock_info") or {}).get("previous_owner")
        assert previous is not None, "no owner record survived the kill"
        assert int(previous["pid"]) == holder_pid, (
            f"owner record pid {previous.get('pid')} != killed writer {holder_pid}"
        )
        assert zrs.owner_is_live(previous) is False
        # 2. The index is reconstructible and equals the on-disk commit set.
        assert report["index"]["rebuilt_on_open"] in (True, False)
        assert sorted(store.index()["cells"]) == sorted(
            store._commits.keys()
        )
        # 3. Reopening without the index changes nothing about the commit set.
        (root / "index.json").unlink(missing_ok=True)
    finally:
        store.close()

    store2 = zrs.RunCellStore.open(root, identity=None)
    try:
        assert store2.index_rebuilt is True
        assert sorted(store2.index()["cells"]) == committed
        for cell_id in committed:
            assert store2.verify_cell(cell_id)["valid"] is True
    finally:
        store2.close()
