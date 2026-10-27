"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 REWORK-3 — identity seam (source_path).

The real descriptors carry a PORTABLE logical id (relative to the input root)
and a separate absolute ``source_path`` used by the loader. The durable run
identity must hash the bytes at ``source_path`` and keep the logical id as the
scientific identifier. Hashing the logical id (the REWORK-3 defect) resolved it
against the process cwd: it raised FileNotFoundError on the real run, and could
even digest a same-named decoy.

These tests run without the corpus, so they execute on any host — including the
Linux host where the RAM-gated real-run case is skipped.
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

FRAME_BYTES = {
    "frame_0000.fits": b"FITS-0000" + bytes([0x0A, 0x0D]) * 8,
    "frame_0001.fits": b"FITS-0001" + bytes([0x0D, 0x0A]) * 8,
}
DECOY_BYTES = b"DECOY" + bytes([0x0A]) * 40


class _FrameId:
    def __init__(self, logical_path: str):
        self.logical_path = logical_path


class _Desc:
    """Mirror of ``geometry.FrameDescriptor`` for the identity seam."""

    def __init__(self, logical_id: str, source_path: str | None):
        self.frame_id = _FrameId(logical_id)
        if source_path is not None:
            self.source_path = source_path


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


def _write_frames(root: Path) -> list[_Desc]:
    root.mkdir(parents=True, exist_ok=True)
    descs = []
    for name, payload in FRAME_BYTES.items():
        path = root / name
        path.write_bytes(payload)
        descs.append(_Desc(name, str(path)))
    return descs


def _write_decoys(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in FRAME_BYTES:
        (root / name).write_bytes(DECOY_BYTES)


def _bundle():
    cell = _Cell("r0000c0000", _Bounds(0, 0, 40, 32))
    other = _Cell("r0000c0001", _Bounds(40, 0, 80, 32))
    return SimpleNamespace(
        layout={"nx": 2, "ny": 1},
        cpu_streaming_tile=512,
        cell_tasks=[(cell, None, None), (other, None, None)],
        cell_ctxs=[
            (0, 0, cell, None, _Membership(cell.cell_id, ["f0"], ["f0", "f1"])),
            (0, 1, other, None, _Membership(other.cell_id, ["f1"], ["f0", "f1"])),
        ],
        chosen_mode="stream",
    )


def _canvas():
    return SimpleNamespace(
        height=64,
        width=80,
        wcs_header="WCSLIB: synthetic header",
        canvas_id="canvas-test",
        resolution_deg=0.5,
    )


def _gauge():
    return SimpleNamespace(
        reference_index=0,
        exclusions=[],
        coefficients=np.linspace(0.5, 1.5, 4),
        weights=np.ones(4),
        scalar_quality=float("nan"),  # digested, never serialised raw
    )


def _science():
    return SimpleNamespace(
        cell_gauge_mode="per_cell",
        contract_version="zegrid-science-v1",
        rejection="sigma",
        sigma=3.0,
        sigma_lower=2.5,
        rejection_algorithm="sds",
    )


def _identity(descs, *, run_id="run-seam"):
    return zz._r55_build_run_store_identity(
        run_id=run_id,
        descs=descs,
        canvas=_canvas(),
        bundle=_bundle(),
        halo_px=8,
        halo_provenance={"effective_px": 8, "source": "unit"},
        global_gauge=_gauge(),
        global_frame_ids=[d.frame_id.logical_path for d in descs],
        gauge_mode="per_cell",
        science_config=_science(),
        workers=2,
        prefetch_units=0,
    )


@pytest.fixture(autouse=True)
def _stable_memory(monkeypatch):
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * 2 ** 20)


def test_m2_identity_hashes_source_path_not_the_logical_id(tmp_path, monkeypatch):
    real = _write_frames(tmp_path / "data")
    decoys = tmp_path / "cwd"
    _write_decoys(decoys)

    # The process cwd contains DIFFERENT bytes under the same names.
    monkeypatch.chdir(decoys)
    identity = _identity(real)

    frames = identity["science_contract"]["inputs"]["frames"]
    assert [f["frame_id"] for f in frames] == list(FRAME_BYTES)
    for frame in frames:
        expected = zrs._sha256_bytes(FRAME_BYTES[frame["frame_id"]])
        assert frame["content_sha256"] == expected, (
            f"identity hashed the wrong file for {frame['frame_id']}"
        )
        assert frame["content_sha256"] != zrs._sha256_bytes(DECOY_BYTES)
    # The aggregate is content-only (relocation-independent).
    assert identity["science_contract"]["inputs"]["aggregate_sha256"] == (
        zrs._inputs_aggregate(frames)
    )


def test_m2_identity_relocation_preserves_scientific_fingerprints(tmp_path):
    first = _write_frames(tmp_path / "run_a")
    second = _write_frames(tmp_path / "elsewhere" / "run_b")
    id_a = _identity(first)
    id_b = _identity(second)
    # Same logical ids and same content at a different absolute location.
    assert zrs.identity_fingerprints(id_a) == zrs.identity_fingerprints(id_b)
    assert zrs.check_identity_compatibility(id_a, id_b)["compatible"] is True
    assert all(
        str(frame["frame_id"]) in FRAME_BYTES
        for frame in id_b["science_contract"]["inputs"]["frames"]
    )


def test_m2_identity_missing_source_path_is_explicit(tmp_path):
    desc = _Desc("frame_0000.fits", None)
    with pytest.raises(zz._R55RunStoreIdentityError) as err:
        _identity([desc])
    assert "source_path" in str(err.value)
    assert "guessing" in str(err.value) or "no source_path" in str(err.value)


def test_m2_identity_unreadable_source_is_explicit(tmp_path):
    desc = _Desc("frame_0000.fits", str(tmp_path / "does_not_exist.fits"))
    with pytest.raises(zz._R55RunStoreIdentityError) as err:
        _identity([desc])
    message = str(err.value)
    assert "unreadable" in message
    assert "does_not_exist.fits" in message


def test_m2_identity_build_then_open_publishes_a_usable_store(tmp_path):
    frames = _write_frames(tmp_path / "data")
    identity = _identity(frames, run_id="run-seam-unit")
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
        info = store.lock_info()
        assert info["held_by_this_store"] is True
        assert info["owner"]["pid"] == zz.os.getpid()
        assert Path(info["lock_path"]).name == "LOCK"
        assert Path(info["owner_path"]).name == "OWNER.json"
        # While HELD, never read LOCK through another handle (Windows denies it).
        assert json.loads(Path(info["owner_path"]).read_text())["state"] == "held"
    finally:
        store.close()
    # After release both files are readable and byte-exact.
    assert zrs.StoreLock.peek_owner(store_dir)["state"] == "released"
    assert (store_dir / "LOCK").read_bytes() == zrs.LOCK_PLACEHOLDER


def test_m2_identity_open_is_idempotent_and_reopens(tmp_path):
    frames = _write_frames(tmp_path / "data")
    identity = _identity(frames, run_id="run-seam-reopen")
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
