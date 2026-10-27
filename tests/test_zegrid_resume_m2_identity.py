"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 REWORK-2 F3 — identity/open seam unit test.

The real-orchestration store test cannot run on a host below the corpus RAM
floor, so the product seam that builds the run identity and opens the store was
never executed outside Windows. This test exercises that exact seam on any
platform, with the real shapes (descriptors, canvas, bundle, memberships, gauge,
science config), and asserts the store really opens and publishes.

It also pins the two properties the Windows bench depends on:
* a non-finite gauge value or an unhashable provenance value must not abort the
  identity build (they are digested, not serialised raw);
* an opened store writes ``run.json`` and a valid lock/owner pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402


class _FrameId:
    def __init__(self, path: str):
        self.logical_path = path


class _Desc:
    def __init__(self, path: str):
        self.frame_id = _FrameId(path)


class _Bounds:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


class _Cell:
    def __init__(self, cell_id, bounds):
        self.cell_id = cell_id
        self.core = bounds


class _Membership:
    def __init__(self, cell_id, core_ids, patch_ids):
        self.cell_id = cell_id
        self.core_ids = tuple(core_ids)
        self.patch_ids = tuple(patch_ids)


def _fixture(tmp_path):
    frames = []
    for index in range(2):
        path = tmp_path / f"frame_{index:04d}.fits"
        path.write_bytes(b"FITS" + bytes([0x0A, 0x0D]) * 8 + str(index).encode())
        frames.append(_Desc(str(path)))
    canvas = SimpleNamespace(
        height=64,
        width=80,
        wcs_header="WCSLIB: synthetic header",
        canvas_id="canvas-test",
        resolution_deg=0.5,
    )
    cell = _Cell("r0000c0000", _Bounds(0, 0, 40, 32))
    other = _Cell("r0000c0001", _Bounds(40, 0, 80, 32))
    bundle = SimpleNamespace(
        layout={"nx": 2, "ny": 1},
        cpu_streaming_tile=512,
        cell_tasks=[(cell, None, None), (other, None, None)],
        cell_ctxs=[
            (0, 0, cell, None, _Membership(cell.cell_id, ["f0"], ["f0", "f1"])),
            (0, 1, other, None, _Membership(other.cell_id, ["f1"], ["f0", "f1"])),
        ],
        chosen_mode="stream",
    )
    gauge = SimpleNamespace(
        reference_index=0,
        exclusions=[],
        coefficients=np.linspace(0.5, 1.5, 4),
        weights=np.ones(4),
        scalar_quality=float("nan"),  # must be digested, never serialised raw
    )
    science = SimpleNamespace(
        cell_gauge_mode="per_cell",
        contract_version="zegrid-science-v1",
        rejection="sigma",
        sigma=3.0,
        sigma_lower=2.5,
        rejection_algorithm="sds",
    )
    return frames, canvas, bundle, gauge, science


def test_m2_identity_build_then_open_publishes_a_usable_store(tmp_path, monkeypatch):
    frames, canvas, bundle, gauge, science = _fixture(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * 2 ** 20)

    identity = zz._r55_build_run_store_identity(
        run_id="run-seam-unit",
        descs=frames,
        canvas=canvas,
        bundle=bundle,
        halo_px=8,
        halo_provenance={"effective_px": 8, "source": tmp_path / "prov.json"},
        global_gauge=gauge,
        global_frame_ids=[frames[0].frame_id.logical_path, frames[1].frame_id.logical_path],
        gauge_mode="per_cell",
        science_config=science,
        workers=2,
        prefetch_units=0,
    )
    # Identity is strict JSON and relocation-independent.
    sections = zrs.identity_fingerprints(identity)
    assert set(sections) == set(zrs.IDENTITY_SECTIONS)
    assert identity["science_contract"]["grid"]["nx"] == 2
    assert identity["science_contract"]["memberships"]["n_cells"] == 2

    store_dir = tmp_path / "run_store"
    store, error = zz._r55_open_run_store(
        store_dir, identity=identity, budget_bytes=64 * 2 ** 20,
        session_id="run-seam-unit",
    )
    assert error is None, f"store open failed: {type(error).__name__}: {error}"
    assert store is not None
    try:
        assert (store_dir / "run.json").exists()
        report = store.report()
        assert report["run_id"] == "run-seam-unit"
        assert report["counts"]["cells"] == 0
        # The lock and the owner record are SEPARATE files (REWORK-2 F1).
        info = store.lock_info()
        assert info["held_by_this_store"] is True
        assert info["owner"]["pid"] == zz.os.getpid()
        assert Path(info["lock_path"]).name == "LOCK"
        assert Path(info["owner_path"]).name == "OWNER.json"
        assert Path(info["lock_path"]).read_bytes() == zrs.LOCK_PLACEHOLDER
        assert json.loads(Path(info["owner_path"]).read_text())["state"] == "held"
    finally:
        store.close()
    # A closed store leaves a readable "released" record, not a stale lock.
    assert zrs.StoreLock.peek_owner(store_dir)["state"] == "released"


def test_m2_identity_open_is_idempotent_and_reopens(tmp_path, monkeypatch):
    frames, canvas, bundle, gauge, science = _fixture(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * 2 ** 20)
    identity = zz._r55_build_run_store_identity(
        run_id="run-seam-reopen",
        descs=frames,
        canvas=canvas,
        bundle=bundle,
        halo_px=8,
        halo_provenance=None,
        global_gauge=gauge,
        global_frame_ids=[f.frame_id.logical_path for f in frames],
        gauge_mode="per_cell",
        science_config=science,
        workers=1,
        prefetch_units=0,
    )
    store_dir = tmp_path / "run_store"
    first, error = zz._r55_open_run_store(
        store_dir, identity=identity, budget_bytes=0, session_id="a"
    )
    assert error is None and first is not None
    first.close()
    second, error = zz._r55_open_run_store(
        store_dir, identity=identity, budget_bytes=0, session_id="b"
    )
    assert error is None and second is not None
    second.close()
    assert zrs.StoreLock.peek_owner(store_dir)["state"] == "released"
