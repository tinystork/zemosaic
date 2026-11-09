"""ZM-ACTIVE-RESUME-FROZEN-IDENTITY — durable identity from the ACTIVE contract.

The real Caldwell LP run completed the gauge, durably committed two GPU Cells,
then stopped fail-closed on host RAM. The advertised resume failed BEFORE any
stack execution with::

    frozen working-set class 'ram_tile' cannot be re-derived as 'external_spool'

Root cause pinned by this module: ``_r55_build_run_store_identity`` was built
from the provisional CPU planning bundle (``_initial_bundle``) and hard-coded
``backend='cpu'``, while the ACTIVE executor writes Cells through
``_r55_active_bundle`` (GPU). The durable identity therefore described a
different execution contract than the one that produced the immutable commits.

These tests prove, without a product run or a real corpus:

* the new identity/admission/working-set derive from the run-scope bundle and
  a GPU ACTIVE run freezes ``backend='gpu'`` at the executed tile;
* a durable commit whose backend conflicts with the frozen backend is refused
  BEFORE it can become an accepted result;
* a same-backend sparse resume skips committed Cells and executes only pending
  ones, with state/identity/working-set internally consistent;
* a reduced-RAM frozen ``ram_tile`` resume emits the precise frozen-capacity
  refusal instead of a misleading external class mismatch;
* a CPU-frozen resume ignores an offered GPU, and a GPU-frozen run cannot
  silently continue on CPU;
* the known legacy CPU-identity/GPU-commit store is detected explicitly and is
  never silently migrated or mixed.
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
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

MIB = 2**20
H = W = 8
CHANNELS = 3
CORE = (2, 6, 2, 7)  # y0, y1, x0, x1
CELL_IDS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")

DTYPE_CONTRACT = {
    "planes": {
        "core.science_core": ["<f4", "<f8"],
        "core.estimator_weight_sum_core": ["<f4", "<f8"],
        "core.support_w1_core": ["<f4", "<f8"],
        "core.support_w2_core": ["<f4", "<f8"],
        "core.n_eff_support_core": ["<f4", "<f8"],
        "core.valid_mask_core": ["|b1"],
        "core.surviving_sample_count_core": ["<i4", "<i8"],
        "patch.science": ["<f4", "<f8"],
        "patch.valid_mask": ["|b1"],
        "patch.surviving": ["<i4", "<i8"],
    }
}


# ---------------------------------------------------------------------------
# Durable run identity (content is identity)
# ---------------------------------------------------------------------------


def make_identity(
    *, run_id="run-frozen", tile_px=64, backend="cpu", parity=False,
    ram=64 * MIB,
):
    return zrs.build_run_identity(
        run_id=run_id,
        inputs={
            "frames": [
                {
                    "frame_id": "frame_0001.fits",
                    "content_sha256": "a" * 64,
                    "nbytes": 1024,
                    "order": 0,
                    "role": "light",
                },
                {
                    "frame_id": "frame_0002.fits",
                    "content_sha256": "b" * 64,
                    "nbytes": 2048,
                    "order": 1,
                    "role": "light",
                },
            ]
        },
        canvas={"shape": [H, W], "wcs_sha256": "c" * 64, "geometry_sha256": "d" * 64},
        grid={
            "nx": 2,
            "ny": 2,
            "tile_px": int(tile_px),
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": zz._r55_sha256_json(sorted(CELL_IDS)),
        },
        halo={"halo_px": 4, "provenance": {"effective_px": 4}},
        memberships={"sha256": "1" * 64, "n_cells": len(CELL_IDS)},
        gauge={
            "mode": "per_cell",
            "sha256": "2" * 64,
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_ids": ["frame_0001.fits", "frame_0002.fits"],
            "n_exclusions": 0,
        },
        science={
            "sha256": "3" * 64,
            "contract_version": "1",
            "params": {"reject": "sigma"},
        },
        dtypes=DTYPE_CONTRACT,
        backend={
            "backend": str(backend),
            "tile_px": int(tile_px),
            "parity_qualified": bool(parity),
        },
        execution_context={
            "available_ram_bytes": int(ram),
            "workers": 2,
            "prefetch_units": 0,
            "host": "test-host",
            "note": "runtime only",
        },
    )


def complete_result(cell_id="r0000c0000", *, backend="cpu", seed=1):
    rng = np.random.default_rng(seed)
    y0, y1, x0, x1 = CORE
    bases = {
        "science": rng.normal(size=(H, W, CHANNELS)).astype(np.float32),
        "estimator": rng.normal(size=(H, W)).astype(np.float32),
        "w1": rng.normal(size=(H, W)).astype(np.float32),
        "w2": rng.normal(size=(H, W)).astype(np.float32),
        "neff": rng.normal(size=(H, W)).astype(np.float32),
        "valid": rng.integers(0, 2, size=(H, W)).astype(bool),
        "surviving": rng.integers(0, 5, size=(H, W)).astype(np.int32),
    }
    return {
        "cell_id": cell_id,
        "status": "complete",
        "backend_used": str(backend),
        "n_frames": 2,
        "total_bytes": 1234,
        "core": {
            "science_core": bases["science"][y0:y1, x0:x1, :],
            "estimator_weight_sum_core": bases["estimator"][y0:y1, x0:x1],
            "support_w1_core": bases["w1"][y0:y1, x0:x1],
            "support_w2_core": bases["w2"][y0:y1, x0:x1],
            "n_eff_support_core": bases["neff"][y0:y1, x0:x1],
            "valid_mask_core": bases["valid"][y0:y1, x0:x1],
            "surviving_sample_count_core": bases["surviving"][y0:y1, x0:x1],
        },
        "record": {
            "cell_id": cell_id,
            "patch_bounds": [0, 0, W, H],
            "patch_shape_hw": [H, W],
            "cell_gauge_mode": "per_cell",
            "backend_used": str(backend),
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
        "patch": {
            "bounds": (0, H, 0, W),
            "core_slice": CORE,
            "science": bases["science"],
            "valid_mask": bases["valid"],
            "surviving": bases["surviving"],
        },
    }


def _budget():
    return zrs.DiskBudget(durable_cell_bytes=64 * MIB)


def _open_store(root, identity):
    return zrs.RunCellStore.open(
        root, identity=identity, budget=_budget(), session_id="frozen-test"
    )


def _write_state(root, *, run_id, admission):
    return zres.write_state(
        root,
        run_id=str(run_id),
        expected_cell_ids=CELL_IDS,
        policy={
            "requested": "auto",
            "effective": "dense",
            "fallback_count": 0,
            "transitions": [],
            "dense_refusal": None,
        },
        admission=admission,
    )


# ---------------------------------------------------------------------------
# 1 — the NEW identity is frozen from the run-scope bundle, not the CPU one
# ---------------------------------------------------------------------------


class _FrameId:
    def __init__(self, logical_path):
        self.logical_path = logical_path


class _Desc:
    def __init__(self, logical_id, source_path):
        self.frame_id = _FrameId(logical_id)
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


def _scope_bundle(*, cpu_tile=96, gpu_tile=None, parity=False):
    """A real-shaped run buddy: CPU planning values + optional GPU contract."""
    cells = []
    tasks = []
    ctxs = []
    for index, cid in enumerate(CELL_IDS):
        col = index % 2
        row = index // 2
        bounds = _Bounds(col * 40, row * 32, (col + 1) * 40, (row + 1) * 32)
        cell = _Cell(cid, bounds)
        mem = _Membership(cid, ["f0"], ["f0", "f1"])
        cells.append(cell)
        tasks.append((cell, None, None))
        ctxs.append((row, col, cell, None, mem))
    return SimpleNamespace(
        layout={"nx": 2, "ny": 2},
        cpu_streaming_tile=int(cpu_tile),
        gpu_streaming_tile=gpu_tile,
        gpu_used=gpu_tile is not None,
        gpu_parity_qualified=bool(parity),
        cell_tasks=tasks,
        cell_ctxs=ctxs,
        chosen_mode="stream",
        working_set_block={
            "class": zz.WORKING_SET_RAM_TILE,
            "tile_size": int(cpu_tile),
            "column_batch": None,
        },
    )


def _canvas():
    return SimpleNamespace(
        height=64,
        width=80,
        wcs_header="WCSLIB: synthetic header",
        canvas_id="canvas-frozen",
        resolution_deg=0.5,
    )


def _gauge():
    return SimpleNamespace(
        reference_index=0,
        exclusions=[],
        coefficients=np.linspace(0.5, 1.5, 4),
        weights=np.ones(4),
    )


def _science():
    return SimpleNamespace(
        cell_gauge_mode="per_cell",
        contract_version="zegrid-science-v1",
        rejection="sigma",
        sigma=3.0,
    )


def _identity_from_bundle(tmp_path, bundle, *, run_id="run-frozen"):
    frames = []
    for index in range(2):
        path = tmp_path / f"f{index}.fits"
        path.write_bytes((f"frame-{index}-" * 32).encode("utf-8"))
        frames.append(_Desc(f"f{index}.fits", str(path)))
    return zz._r55_build_run_store_identity(
        run_id=run_id,
        descs=frames,
        canvas=_canvas(),
        bundle=bundle,
        halo_px=4,
        halo_provenance={"effective_px": 4, "source": "run_parameter"},
        global_gauge=_gauge(),
        global_frame_ids=[d.frame_id.logical_path for d in frames],
        gauge_mode="per_cell",
        science_config=_science(),
        workers=2,
        prefetch_units=None,
    )


def test_gpu_run_freezes_the_run_scope_gpu_identity_not_the_cpu_plan(tmp_path):
    cpu_initial = _scope_bundle(cpu_tile=96)
    gpu_scope = _scope_bundle(cpu_tile=0, gpu_tile=213, parity=True)

    cpu_identity = _identity_from_bundle(tmp_path, cpu_initial)
    gpu_identity = _identity_from_bundle(tmp_path, gpu_scope)

    cpu_block = cpu_identity["science_contract"]["backend"]
    gpu_block = gpu_identity["science_contract"]["backend"]

    # The provisional CPU planning bundle says cpu/tile96/non-parity (the bug).
    assert (cpu_block["backend"], cpu_block["tile_px"], cpu_block["parity_qualified"]) == (
        "cpu",
        96,
        False,
    )
    # The bundle that ACTUALLY writes Cells is GPU, at its admitted tile.
    assert (gpu_block["backend"], gpu_block["tile_px"], gpu_block["parity_qualified"]) == (
        "gpu",
        213,
        True,
    )
    # grid.tile_px and backend.tile_px can never disagree.
    assert gpu_identity["science_contract"]["grid"]["tile_px"] == 213

    # The frozen admission is a projection of that SAME identity.
    admission = zz._admission_from_identity(gpu_identity)
    assert admission["backend"] == "gpu"
    assert admission["tile_px"] == 213
    assert admission["parity_qualified"] is True

    # The frozen working set is the GPU host staging class at the GPU tile.
    working_set = zz._r55_run_store_working_set(
        gpu_scope, zz._r55_execution_contract_from_bundle(gpu_scope)
    )
    assert working_set == {
        "class": "ram_tile",
        "tile_px": 213,
        "column_batch": None,
    }
    # It is NOT the provisional CPU 96 tile.
    assert working_set["tile_px"] != cpu_block["tile_px"]


def test_cpu_scope_still_freezes_the_cpu_contract(tmp_path):
    identity = _identity_from_bundle(tmp_path, _scope_bundle(cpu_tile=96))
    block = identity["science_contract"]["backend"]
    assert block == {"backend": "cpu", "tile_px": 96, "parity_qualified": False}
    working_set = zz._r55_run_store_working_set(
        _scope_bundle(cpu_tile=96), block
    )
    assert working_set == {"class": "ram_tile", "tile_px": 96, "column_batch": None}


# ---------------------------------------------------------------------------
# 2 — a durable commit may never mix backends under one frozen identity
# ---------------------------------------------------------------------------


def test_cpu_identity_refuses_a_gpu_commit_before_writing(tmp_path):
    root = tmp_path / "run"
    store = _open_store(root, make_identity(tile_px=96))
    try:
        store.commit_cell(
            complete_result("r0000c0000", backend="cpu"),
            attempt_id="cpu-1",
            backend="cpu",
        )
        with pytest.raises(zrs.StoreFormatError) as excinfo:
            store.commit_cell(
                complete_result("r0000c0001", backend="gpu"),
                attempt_id="gpu-1",
                backend="gpu",
            )
        message = str(excinfo.value)
        assert "backend_used='gpu'" in message
        assert "backend='cpu'" in message
        assert "no cell was written" in message
        assert store.attempts("r0000c0001") == []
        # The valid CPU commit is untouched.
        assert store.commit_record("r0000c0000", "cpu-1").backend == "cpu"
    finally:
        store.close()


def test_gpu_identity_refuses_a_cpu_commit_before_writing(tmp_path):
    root = tmp_path / "run"
    store = _open_store(root, make_identity(tile_px=213, backend="gpu", parity=True))
    try:
        store.commit_cell(
            complete_result("r0000c0000", backend="gpu"),
            attempt_id="gpu-1",
            backend="gpu",
        )
        with pytest.raises(zrs.StoreFormatError):
            store.commit_cell(
                complete_result("r0000c0001", backend="cpu"),
                attempt_id="cpu-1",
                backend="cpu",
            )
        assert store.attempts("r0000c0001") == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3 — same-backend sparse resume: state/identity/working-set stay coherent
# ---------------------------------------------------------------------------


def test_sparse_resume_skips_committed_cells_and_keeps_consistency(tmp_path):
    root = tmp_path / "run"
    identity = make_identity(tile_px=64, backend="cpu")
    admission = zres.build_admission(
        backend="cpu", tile_px=64, parity_qualified=False,
        working_set={"class": "ram_tile", "tile_px": 64, "column_batch": None},
    )
    store = _open_store(root, identity)
    try:
        _write_state(root, run_id=store.run_id, admission=admission)
        store.commit_cell(
            complete_result("r0000c0000", backend="cpu"),
            attempt_id="cpu-wave:0:0:r0000c0000",
            backend="cpu",
        )
        store.commit_cell(
            complete_result("r0001c0001", backend="cpu"),
            attempt_id="cpu-wave:0:0:r0001c0001",
            backend="cpu",
        )
    finally:
        store.close()

    boot = zres.ResumeBootstrap.open(root)
    try:
        # state admission and the stored identity are the SAME contract.
        assert boot.admission == admission
        stored_block = boot.identity["science_contract"]["backend"]
        assert stored_block["backend"] == admission["backend"]
        assert stored_block["tile_px"] == admission["tile_px"]
        assert stored_block["parity_qualified"] == admission["parity_qualified"]
        assert admission["working_set"]["class"] == "ram_tile"
        assert admission["working_set"]["tile_px"] == 64

        plan = boot.plan(list(CELL_IDS))
        assert plan["complete"] == 2
        assert plan["pending_count"] == 2
        assert sorted(plan["pending"]) == ["r0000c0001", "r0001c0000"]
        assert sorted(plan["complete_cell_ids"] if "complete_cell_ids" in plan else [
            cid for cid in CELL_IDS if cid not in plan["pending"]
        ]) == ["r0000c0000", "r0001c0001"]

        # The diagnostic counts name the same durable evidence (no re-plan).
        assert boot.retained_counts() == {
            "expected": 4,
            "complete": 2,
            "pending": 2,
        }

        # A process with different free RAM still matches: RAM is runtime-only.
        candidate = make_identity(tile_px=64, backend="cpu", ram=4 * MIB)
        assert boot.verify_identity(candidate)["compatible"] is True
    finally:
        boot.close()


# ---------------------------------------------------------------------------
# 4 — reduced-RAM frozen ram_tile resume: frozen-capacity refusal, no mismatch
# ---------------------------------------------------------------------------

REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(**_kw):
    return REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


class _HugeInmem:
    def predict_bound_bytes(self, n, area):
        return 1 << 40

    def predict_peak_bytes(self, n, area):
        return 1 << 40


def _make_tan(i, h, w, scale, ra_step, source_path):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid),
        source_path=str(source_path),
        shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="",
        instrument="",
    )


@pytest.fixture
def bundle_env(tmp_path, monkeypatch):
    """Tile 128 needs 128 MiB, tile 64 needs 64 MiB; geometry is pinned."""
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeInmem())
    monkeypatch.setattr(
        zz,
        "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3, **_kw: int(tile)
        * 1_000_000,
    )
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(rcontract, "PROCESS_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zz.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 512 * MIB)
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")

    descs = []
    for index in range(3):
        path = tmp_path / f"d{index}.fits"
        path.write_bytes((f"frame-{index}-" * 64).encode("utf-8"))
        descs.append(_make_tan(index, 40, 40, 0.001, 0.004, path))
    canvas = zg.build_canvas(descs)
    return SimpleNamespace(descs=descs, canvas=canvas)


RAW_HIGH = 375_000_000  # plan budget 300 MB -> tile 128 fits
RAW_LOW = 93_750_000    # plan budget 75 MB -> tile 128 refused


def _bundle(env, *, raw, frozen=None, gpu_used=False, vram=None):
    return zz._build_execution_bundle(
        env.canvas,
        env.descs,
        None,
        raw_available_bytes=int(raw),
        budget_bytes=(4 * int(raw)) // 5,
        pinned_layout=(2, 2),
        halo_px=4,
        cpu_count=4,
        cache_root=None,
        gpu_used=bool(gpu_used),
        vram_budget_bytes=vram,
        frozen_admission=frozen,
    )


def _admission(tile_px, backend="cpu", working_set="ram_tile"):
    ws = None
    if working_set is not None:
        ws = {
            "class": str(working_set),
            "tile_px": int(tile_px),
            "column_batch": None,
        }
    return zres.build_admission(
        backend=backend,
        tile_px=int(tile_px),
        parity_qualified=False,
        working_set=ws,
    )


def test_reduced_ram_frozen_ram_tile_emits_capacity_refusal(bundle_env):
    """The observed defect: a misleading external_spool class mismatch."""
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(
            bundle_env,
            raw=RAW_LOW,
            frozen=_admission(128, working_set="ram_tile"),
        )
    exc = excinfo.value
    message = str(exc)
    assert exc.frozen_backend == "cpu"
    assert exc.frozen_tile_px == 128
    assert exc.required_bytes == 128_000_000
    assert exc.available_bytes == 75_000_000
    assert exc.deficit_bytes == 53_000_000
    assert "frozen tile_px=128" in message
    assert "backend=cpu" in message
    assert "class=ram_tile" in message
    # A frozen-capacity refusal is what the operator sees...
    assert "frozen" in message.lower()
    assert "deficit" in message
    assert "never shrunk" in message
    # ...never the misleading class-mismatch message from the old planning order.
    assert "cannot be re-derived as" not in message
    assert "external_spool" not in message


def test_reduced_ram_frozen_ram_tile_never_selects_the_external_class(bundle_env):
    """Under the bounded profile the external selector is never reached."""
    frozen = _admission(128, working_set="ram_tile")
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(bundle_env, raw=RAW_LOW, frozen=frozen)
    # The planner detail is the CPU capacity refusal, not the class mismatch.
    assert "external" not in str(excinfo.value).lower()


def test_frozen_tile_that_fits_again_is_reused(bundle_env):
    """Control: the same frozen tile at adequate RAM plans normally."""
    bundle = _bundle(
        bundle_env, raw=RAW_HIGH, frozen=_admission(128, working_set="ram_tile")
    )
    assert int(bundle.cpu_streaming_tile) == 128
    assert bundle.working_set_block["class"] == "ram_tile"


# ---------------------------------------------------------------------------
# 5 — backend freeze direction: CPU ignores GPU, GPU never continues on CPU
# ---------------------------------------------------------------------------


def test_cpu_frozen_resume_ignores_an_offered_gpu():
    frozen = _admission(64, backend="cpu")
    assert zz._frozen_planning_gpu(frozen, gpu_used=True) is False


def test_gpu_frozen_resume_without_a_gpu_refuses(bundle_env):
    frozen = _admission(64, backend="gpu")
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(bundle_env, raw=RAW_HIGH, frozen=frozen, gpu_used=False)
    assert excinfo.value.frozen_backend == "gpu"
    assert excinfo.value.frozen_tile_px == 64


def test_gpu_frozen_resume_with_a_gpu_selects_gpu():
    frozen = _admission(64, backend="gpu")
    assert zz._frozen_planning_gpu(frozen, gpu_used=True) is True


def test_gpu_tile_cap_never_enlarges_the_resume_ladder():
    """The resume ladder can never grow above the frozen GPU tile."""
    declared = (256, 213, 128, 64)
    assert zz._r55_cap_gpu_tile_ladder(declared, 213) == (213, 128, 64)
    assert zz._r55_cap_gpu_tile_ladder(declared, 64) == (64,)
    # A frozen tile above the fresh maximum is not invented: the ladder stays
    # empty and the caller raises the precise frozen GPU refusal.
    assert zz._r55_cap_gpu_tile_ladder((128, 64), 213) == (128, 64)
    # No frozen admission (fresh run): the declared ladder is untouched.
    assert zz._r55_cap_gpu_tile_ladder(declared, None) == declared
    # Never above the cap, even if the cap is not itself a rung.
    assert zz._r55_cap_gpu_tile_ladder(declared, 200) == (128, 64)


# ---------------------------------------------------------------------------
# 6 — the known legacy CPU-identity/GPU-commit store is detected, never mixed
# ---------------------------------------------------------------------------


def _legacy_store_with_gpu_commits(root):
    """Build the historical artifact: CPU identity + immutable GPU commits.

    The production writer of that build stamped ``backend='gpu'`` while the
    frozen identity said CPU. The strict commit guard now refuses that, so the
    fixture reproduces it at the WRITER level (identity view swapped only for
    the duration of the two commits). Manifests keep the CPU identity sha, which
    is exactly the legacy shape.
    """
    identity = make_identity(tile_px=96, backend="cpu")
    store = _open_store(root, identity)
    original = store._identity
    gpu_view = json.loads(json.dumps(original))
    gpu_view["science_contract"]["backend"]["backend"] = "gpu"
    store._identity = gpu_view
    try:
        store.commit_cell(
            complete_result("r0000c0000", backend="gpu"),
            attempt_id="gpu-wave:0:0:r0000c0000",
            backend="gpu",
        )
        store.commit_cell(
            complete_result("r0001c0001", backend="gpu"),
            attempt_id="gpu-wave:0:0:r0001c0001",
            backend="gpu",
        )
    finally:
        store._identity = original
        store.close()
    return identity


def test_legacy_cpu_identity_with_gpu_commits_is_refused_and_preserved(tmp_path):
    root = tmp_path / "run"
    identity = _legacy_store_with_gpu_commits(root)
    # A legacy store has no frozen admission section in this shape.
    store = zrs.RunCellStore.open(root, identity=None, budget=_budget())
    try:
        _write_state(
            root,
            run_id=store.run_id,
            admission=None,
        )
    finally:
        store.close()

    state_before = (root / "state.json").read_bytes()
    identity_before = (root / "run.json").read_bytes()
    commits_before = sorted((root / "commits").iterdir())
    assert len(commits_before) == 2

    with pytest.raises(zres.ResumeLegacyBackendIdentity) as excinfo:
        zres.ResumeBootstrap.open(root)
    exc = excinfo.value
    assert exc.reason_code == "legacy_inconsistent_backend_identity"
    assert exc.frozen_backend == "cpu"
    assert exc.frozen_tile_px == 96
    assert exc.commit_count == 2
    assert exc.commit_backends == ["gpu"]
    assert "legacy_inconsistent_backend_identity" in str(exc)
    assert "never silently migrated" in str(exc)

    # Nothing was rewritten: identity, state and both commits are preserved.
    assert (root / "run.json").read_bytes() == identity_before
    assert (root / "state.json").read_bytes() == state_before
    assert sorted((root / "commits").iterdir()) == commits_before
    # Lock released by the refusal.
    again = zrs.RunCellStore.open(root, identity=None)
    again.close()


def test_read_only_detector_reports_the_legacy_shape(tmp_path):
    root = tmp_path / "run"
    _legacy_store_with_gpu_commits(root)
    store = zrs.RunCellStore.open(root, identity=None, budget=_budget())
    try:
        report = zres.detect_legacy_backend_identity(store)
    finally:
        store.close()
    assert report["reason_code"] == "legacy_inconsistent_backend_identity"
    assert report["frozen_backend"] == "cpu"
    assert report["commit_backends"] == ["gpu"]
    assert report["recoverable"] is False


def test_consistent_cpu_store_is_not_flagged_as_legacy(tmp_path):
    root = tmp_path / "run"
    identity = make_identity(tile_px=96)
    store = _open_store(root, identity)
    try:
        store.commit_cell(
            complete_result("r0000c0000", backend="cpu"),
            attempt_id="cpu-1",
            backend="cpu",
        )
        assert zres.detect_legacy_backend_identity(store) is None
    finally:
        store.close()
