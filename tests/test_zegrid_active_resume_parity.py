"""ZM-ZEGRID-ACTIVE-RESUME-PARITY — an ``active`` run resumes like ``legacy``.

The durable science identity already contains the effective backend and the
``tile_px`` that backend executes. The defect: those ADMISSION inputs were
RE-DERIVED at every launch from the CURRENT free RAM (``available_ram_bytes``
is injected into the planning facts), so a changed host chose a different tile
and the identity check refused a scientifically resumable run. ``legacy`` was
immune only because its tile is fixed.

These tests pin the fix:
* the admission contract is FROZEN in the durable run state at creation,
  schema-versioned and strictly validated (unknown keys refused);
* a resume REUSES the frozen backend/tile instead of re-deriving them;
* an unhonourable frozen tile FAILS CLOSED with a typed, actionable refusal;
* an old store WITHOUT the section keeps today's conservative behaviour;
* identity refusals carry STORED vs CURRENT values and the remedy.

No GPU, no FITS corpus, no heavy product run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402
from zemosaic.core.zegrid import observability as zobs  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402

TOOLS_M4 = REPO_ROOT / "tools" / "zegrid_resume_m4"
if str(TOOLS_M4) not in sys.path:
    sys.path.insert(0, str(TOOLS_M4))
import m4_resume_harness as HARNESS  # noqa: E402

TINY_CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"

H = W = 32
NX = NY = 2
TILE = 16

DTYPES = {
    "planes": {
        "core.science_core": ["<f4"],
        "patch.science": ["<f4"],
        "patch.valid_mask": ["|b1"],
        "patch.surviving": ["<i4"],
    }
}


def make_identity(
    *, run_id="run-parity", tile_px=TILE, backend="cpu", ram=64 * 2**20,
    parity=False,
):
    """Minimal but REAL run identity (content is identity)."""
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
            "nx": NX,
            "ny": NY,
            "tile_px": int(tile_px),
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": "f" * 64,
        },
        halo={"halo_px": 4, "provenance": {"effective_px": 4}},
        memberships={"sha256": "1" * 64, "n_cells": NX * NY},
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
        dtypes=DTYPES,
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


CELL_IDS = ("r0c0", "r0c1", "r1c0", "r1c1")


def _budget():
    return zrs.DiskBudget(durable_cell_bytes=1 << 20)


def _make_store(root, identity):
    return zrs.RunCellStore.open(root, identity=identity, budget=_budget(), session_id="parity")


def _write_state(root, *, run_id, admission, cells=CELL_IDS):
    return zres.write_state(
        root,
        run_id=run_id,
        expected_cell_ids=cells,
        policy={
            "requested": "auto",
            "effective": "dense",
            "fallback_count": 0,
            "transitions": [],
            "dense_refusal": None,
        },
        admission=admission,
    )


def _policy():
    return {
        "requested": "auto",
        "effective": "dense",
        "fallback_count": 0,
        "transitions": [],
    }


# ---------------------------------------------------------------------------
# 1 — schema: strict, versioned, backward compatible
# ---------------------------------------------------------------------------


def test_admission_is_schema_versioned_and_strict():
    admission = zres.build_admission(
        backend="cpu", tile_px=80, parity_qualified=False
    )
    assert admission == {
        "schema": zres.ADMISSION_SCHEMA,
        "schema_version": zres.ADMISSION_VERSION,
        "backend": "cpu",
        "tile_px": 80,
        "parity_qualified": False,
    }
    # Unknown/extra keys are refused (same discipline as every durable section).
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission({**admission, "extra": 1})
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission({**admission, "backend": "tpu"})
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission({**admission, "tile_px": True})
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission({**admission, "tile_px": 0})
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission({**admission, "schema_version": "9"})
    missing = dict(admission)
    missing.pop("parity_qualified")
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission(missing)


def test_state_without_admission_loads_and_stays_conservative(tmp_path):
    root = tmp_path / "run"
    identity = make_identity()
    store = _make_store(root, identity)
    try:
        _write_state(root, run_id=store.run_id, admission=None)
    finally:
        store.close()
    raw = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert "admission" not in raw, "an unfrozen state must not fabricate a section"

    boot = zres.ResumeBootstrap.open(root)
    try:
        assert boot.admission is None
    finally:
        boot.close()


def test_admission_survives_policy_rewrite(tmp_path):
    root = tmp_path / "run"
    identity = make_identity(tile_px=80)
    store = _make_store(root, identity)
    admission = zres.build_admission(
        backend="cpu", tile_px=80, parity_qualified=False
    )
    try:
        _write_state(root, run_id=store.run_id, admission=admission)
    finally:
        store.close()
    zres.update_policy(
        root, {**_policy(), "effective": "bounded", "fallback_count": 1}
    )
    state = zres.read_state(root)
    assert state["admission"] == admission, "policy rewrite must not touch the freeze"
    assert state["policy"]["effective"] == "bounded"


def test_admission_contradicting_the_stored_identity_is_refused(tmp_path):
    root = tmp_path / "run"
    identity = make_identity(tile_px=80)
    store = _make_store(root, identity)
    try:
        _write_state(
            root,
            run_id=store.run_id,
            admission=zres.build_admission(
                backend="cpu", tile_px=64, parity_qualified=False
            ),
        )
    finally:
        store.close()
    with pytest.raises(zres.ResumeCorruptGlobal) as excinfo:
        zres.ResumeBootstrap.open(root)
    assert "admission" in str(excinfo.value)
    # The refusal released the lock: the store reopens.
    again = zrs.RunCellStore.open(root, identity=None)
    again.close()


# ---------------------------------------------------------------------------
# 2 — resume with a DIFFERENT host RAM reuses the frozen admission
# ---------------------------------------------------------------------------


def test_resume_with_different_ram_is_accepted_by_reusing_the_frozen_tile(tmp_path):
    """The observed defect: same run, different free RAM, tile re-derived."""
    root = tmp_path / "run"
    stored = make_identity(tile_px=80, ram=64 * 2**20)
    store = _make_store(root, stored)
    admission = zres.build_admission(
        backend="cpu", tile_px=80, parity_qualified=False
    )
    try:
        _write_state(root, run_id=store.run_id, admission=admission)
    finally:
        store.close()

    boot = zres.ResumeBootstrap.open(root)
    try:
        frozen = boot.admission
        assert frozen == admission

        # The resumed candidate plan is built FROM the frozen admission, so the
        # tile/backend are reused (never re-derived from the new host's RAM).
        candidate = make_identity(tile_px=frozen["tile_px"], ram=4 * 2**20)
        verdict = boot.verify_identity(candidate)
        assert verdict["compatible"] is True
        assert (
            candidate["science_contract"]["backend"]
            == stored["science_contract"]["backend"]
        )
        assert (
            candidate["science_contract"]["grid"]
            == stored["science_contract"]["grid"]
        )
        assert (
            candidate["execution_context"]["available_ram_bytes"]
            != stored["execution_context"]["available_ram_bytes"]
        ), "the test must really simulate a changed host"

        # What a re-derivation would have produced (the old defect) is refused,
        # and the refusal names STORED vs CURRENT values plus the remedy.
        rederived = make_identity(tile_px=64, ram=4 * 2**20)
        with pytest.raises(zres.ResumeIncompatible) as excinfo:
            boot.verify_identity(rederived)
        message = str(excinfo.value)
        assert "stored=" in message and "current=" in message
        assert "tile_px=80" in message and "tile_px=64" in message
        assert "backend" in message
        assert "expected_cells" in message
        assert "remedy" in message.lower()
    finally:
        boot.close()


def test_legacy_store_under_active_process_is_still_refused_with_detail(tmp_path):
    """A scheduler change is a REAL identity change: still refused."""
    root = tmp_path / "run"
    legacy = make_identity(run_id="legacy-run", tile_px=80, ram=32 * 2**20)
    store = _make_store(root, legacy)
    try:
        # A legacy store predates the frozen-admission section.
        _write_state(root, run_id=store.run_id, admission=None)
    finally:
        store.close()

    boot = zres.ResumeBootstrap.open(root)
    try:
        assert boot.admission is None
        active = make_identity(run_id="legacy-run", tile_px=80, ram=32 * 2**20)
        # Same tile here; the refusal is driven by the frozen legacy grid, so
        # build the candidate with the different active tile the engine chose.
        rederived = make_identity(run_id="legacy-run", tile_px=64, ram=32 * 2**20)
        with pytest.raises(zres.ResumeIncompatible) as excinfo:
            boot.verify_identity(rederived)
        message = str(excinfo.value)
        assert "grid:" in message and "backend:" in message
        assert "stored=" in message and "current=" in message
        assert "remedy" in message.lower()
        # Same-identity candidate is still accepted (no fabricated refusal).
        assert boot.verify_identity(active)["compatible"] is True
    finally:
        boot.close()


# ---------------------------------------------------------------------------
# 3 — the real bundle planner honours the frozen tile (no re-derivation)
# ---------------------------------------------------------------------------

MIB = 2**20
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
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(tile) * 1_000_000,
    )
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 512 * MIB)

    descs = []
    for index in range(3):
        path = tmp_path / f"d{index}.fits"
        path.write_bytes((f"frame-{index}-" * 64).encode("utf-8"))
        descs.append(_make_tan(index, 40, 40, 0.001, 0.004, path))
    canvas = zg.build_canvas(descs)
    return SimpleNamespace(descs=descs, canvas=canvas)


#: raw observation whose 4/5 PLAN budget is the value the tests reason about.
RAW_HIGH = 375_000_000  # -> plan budget 300 MB (tile 128 fits)
RAW_LOW = 87_500_000    # -> plan budget 70 MB (only tile <= 64 fits)


def _bundle(env, *, raw, frozen=None, gpu_used=False, vram=None):
    return zz._build_execution_bundle(
        env.canvas,
        env.descs,
        None,
        raw_available_bytes=int(raw),
        # The envelope contract: budget == plan_budget_bytes(raw) == 4/5 raw.
        budget_bytes=(4 * int(raw)) // 5,
        pinned_layout=(NX, NY),
        halo_px=4,
        cpu_count=4,
        cache_root=None,
        gpu_used=bool(gpu_used),
        vram_budget_bytes=vram,
        frozen_admission=frozen,
    )


def _admission(tile_px, backend="cpu"):
    return zres.build_admission(
        backend=backend, tile_px=int(tile_px), parity_qualified=False
    )


def test_fresh_plan_re_derives_the_tile_from_ram(bundle_env):
    """Control: without a freeze the tile follows free RAM (the defect)."""
    high = _bundle(bundle_env, raw=RAW_HIGH)
    low = _bundle(bundle_env, raw=RAW_LOW)
    assert int(high.cpu_streaming_tile) == 128
    assert int(low.cpu_streaming_tile) == 64


def test_resume_reuses_the_frozen_tile_when_ram_grew(bundle_env):
    """A bigger host must NOT upgrade the tile: the run's own tile is reused."""
    bundle = _bundle(bundle_env, raw=RAW_HIGH, frozen=_admission(64))
    assert int(bundle.cpu_streaming_tile) == 64
    assert int(bundle.cpu_streaming["selected_tile_size"] or 0) in (0, 64, 128)


def test_resume_reuses_the_frozen_tile_when_ram_shrank(bundle_env):
    """A smaller host must NOT shrink the tile either (no silent degradation)."""
    bundle = _bundle(bundle_env, raw=RAW_LOW, frozen=_admission(64))
    assert int(bundle.cpu_streaming_tile) == 64


def test_unhonourable_frozen_tile_fails_closed_with_the_deficit(bundle_env):
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(bundle_env, raw=RAW_LOW, frozen=_admission(128))
    exc = excinfo.value
    assert exc.frozen_tile_px == 128
    assert exc.frozen_backend == "cpu"
    assert exc.required_bytes == 128_000_000
    assert exc.available_bytes == 70_000_000
    assert exc.deficit_bytes == 58_000_000
    message = str(exc)
    assert "frozen tile_px=128" in message
    assert "backend=cpu" in message
    assert "deficit" in message
    assert "NEW output folder" in message
    assert "never shrunk" in message


def test_frozen_tile_outside_the_product_ladder_is_refused(bundle_env):
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(bundle_env, raw=RAW_HIGH, frozen=_admission(100))
    assert excinfo.value.frozen_tile_px == 100
    assert "ladder" in str(excinfo.value).lower()


def test_frozen_gpu_backend_without_a_gpu_is_refused(bundle_env):
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        _bundle(bundle_env, raw=RAW_HIGH, frozen=_admission(64, backend="gpu"))
    assert excinfo.value.frozen_backend == "gpu"
    assert "backend" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4 — the identity produced by the reused admission equals the stored one
# ---------------------------------------------------------------------------


def _identity_from_bundle(env, bundle):
    gauge = SimpleNamespace(reference_index=0, exclusions=())
    return zz._r55_build_run_store_identity(
        run_id="parity-run",
        descs=env.descs,
        canvas=env.canvas,
        bundle=bundle,
        halo_px=4,
        halo_provenance={"effective_px": 4, "source": "run_parameter"},
        global_gauge=gauge,
        global_frame_ids=[str(d.frame_id) for d in env.descs],
        gauge_mode="per_cell",
        science_config=SimpleNamespace(contract_version="zegrid-science-v1"),
        workers=2,
        prefetch_units=None,
    )


def test_resumed_identity_matches_the_stored_identity_despite_new_ram(bundle_env):
    low = _bundle(bundle_env, raw=RAW_LOW)
    stored = _identity_from_bundle(bundle_env, low)
    assert stored["science_contract"]["backend"]["tile_px"] == 64

    # Fresh launch on a bigger host: the tile would be re-derived to 128 and the
    # identity would differ (the observed refusal).
    rederived = _identity_from_bundle(bundle_env, _bundle(bundle_env, raw=RAW_HIGH))
    assert rederived["science_contract"]["backend"]["tile_px"] == 128
    assert zrs.identity_sha256(rederived) != zrs.identity_sha256(stored)
    verdict = zrs.check_identity_compatibility(stored, rederived)
    assert verdict["compatible"] is False
    backend_reason = next(r for r in verdict["reasons"] if r["section"] == "backend")
    assert backend_reason["stored"]["tile_px"] == 64
    assert backend_reason["current"]["tile_px"] == 128

    # Resumed launch: the frozen admission is reused, so the identity matches.
    resumed = _identity_from_bundle(
        bundle_env, _bundle(bundle_env, raw=RAW_HIGH, frozen=_admission(64))
    )
    assert resumed["science_contract"]["backend"]["tile_px"] == 64
    assert zrs.identity_sha256(resumed) == zrs.identity_sha256(stored)
    assert zrs.check_identity_compatibility(stored, resumed)["compatible"] is True


def test_frozen_tile_refusal_mutates_nothing_and_leaves_no_lock(
    tmp_path, bundle_env
):
    """The refusal is read-only and the engine releases the lock before raising."""
    root = tmp_path / "run"
    identity = make_identity(tile_px=128, ram=512 * MIB)
    store = _make_store(root, identity)
    admission = _admission(128)
    try:
        _write_state(root, run_id=store.run_id, admission=admission)
    finally:
        store.close()
    boot = zres.ResumeBootstrap.open(root)
    try:
        before = (root / "state.json").read_bytes()
        commits_before = sorted((root / "commits").iterdir())
        with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
            _bundle(bundle_env, raw=RAW_LOW, frozen=boot.admission)
        # No store mutation and no cell admitted by the refusal itself.
        assert (root / "state.json").read_bytes() == before
        assert sorted((root / "commits").iterdir()) == commits_before
        assert zres.read_state(root)["admission"] == admission
    finally:
        boot.close()
    # Lock released: the store opens again immediately.
    again = zrs.RunCellStore.open(root, identity=None)
    again.close()


def test_engine_releases_the_bootstrap_lock_before_a_frozen_refusal():
    source = (REPO_ROOT / "src" / "zemosaic" / "zemosaic_zegrid_mode.py").read_text(
        encoding="utf-8"
    )
    handler = source.index("except zres.ResumeFrozenAdmissionUnavailable as _frozen_exc:")
    close = source.index("_resume_boot.close()", handler)
    raise_at = source.index("        raise", close)
    assert handler < close < raise_at


# ---------------------------------------------------------------------------
# 5 — the PRODUCTION wiring threads the frozen admission into the real planner
# ---------------------------------------------------------------------------

_HARNESS_HALO = 8  # the product halo ladder accepts 8 (the harness default 4 does not)


class _HarnessFrameInfo:
    """Minimal FrameInfo: only what ``_build_frame_descriptors`` consumes."""

    def __init__(self, name, path):
        from astropy.wcs import WCS

        self.path = str(path)
        self.shape_hw = (HARNESS.H, HARNESS.W)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0, 30.0]
        wcs.wcs.crpix = [HARNESS.W / 2.0, HARNESS.H / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (HARNESS.H, HARNESS.W)
        self.wcs = wcs
        self.mount = "EQ"


def _resumable_harness_run(tmp_path, monkeypatch):
    """A GENUINE durable run (identity + state + gauge/geometry artifacts).

    Built with the M4 harness primitives (the same ones the kill/resume proof
    uses), so ``_run_single`` reaches the REAL planning entry with a REAL
    bootstrap — no mock of the seam under test.
    """
    monkeypatch.setattr(HARNESS, "HALO", _HARNESS_HALO, raising=False)
    input_dir = tmp_path / "in"
    rows = HARNESS.ensure_inputs(input_dir)
    identity = HARNESS.make_identity("parity-wiring", rows)
    root = tmp_path / "store"
    store = zrs.RunCellStore.open(
        root,
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id="parity-wiring",
    )
    backend = identity["science_contract"]["backend"]
    admission = zres.build_admission(
        backend=backend["backend"],
        tile_px=backend["tile_px"],
        parity_qualified=backend["parity_qualified"],
    )
    try:
        zres.write_state(
            root,
            run_id="parity-wiring",
            expected_cell_ids=list(HARNESS.CELL_IDS),
            policy=_policy(),
            gauge_sha256=identity["science_contract"]["gauge"]["sha256"],
            note="parity wiring probe",
            admission=admission,
        )
        from zemosaic.core.zegrid import run_artifacts as zart

        zart.write_gauge_artifact(
            root,
            run_id="parity-wiring",
            identity=identity,
            gauge=HARNESS.synthetic_gauge(),
            mode="per_cell",
            frame_ids=list(HARNESS.FRAME_IDS),
            reference_frame_id=HARNESS.FRAME_IDS[0],
        )
        canvas, layout, cell_ctxs = HARNESS.synthetic_geometry()
        zart.write_geometry_artifact(
            root,
            run_id="parity-wiring",
            identity=identity,
            canvas=canvas,
            layout=layout,
            cell_ctxs=cell_ctxs,
            halo_px=_HARNESS_HALO,
        )
    finally:
        store.close()
    frames_info = [
        _HarnessFrameInfo(name, input_dir / name) for name in HARNESS.FRAME_IDS
    ]
    return SimpleNamespace(
        root=root,
        input_dir=input_dir,
        frames_info=frames_info,
        admission=admission,
        identity=identity,
    )


def test_run_single_threads_the_frozen_admission_into_the_real_plan_request(
    tmp_path, monkeypatch
):
    """C1: the PRODUCTION path must hand the bootstrap's freeze to the planner.

    The real ``_run_single`` runs its real bootstrap (real store, real identity,
    real gauge/geometry restore) and reaches the real ``_plan_request``; only the
    planner entry is replaced by a recorder that stops the run. Replacing the
    threaded value by ``None`` (or re-deriving it) makes this test fail.
    """
    env = _resumable_harness_run(tmp_path, monkeypatch)
    received = {}

    def recorder(**kwargs):
        received.update(kwargs)
        raise zres.ResumeFrozenAdmissionUnavailable(
            "recorded by the wiring probe",
            frozen_backend=str(env.admission["backend"]),
            frozen_tile_px=int(env.admission["tile_px"]),
        )

    monkeypatch.setattr(zz, "_plan_bundle_auto", recorder)
    zconfig = SimpleNamespace(
        zegrid_durable_store_dir=str(env.root),
        zegrid_durable_store_budget_bytes=64 * MIB,
        temporary_work_dir="",
    )
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._run_single(
            env.frames_info,
            env.input_dir,
            tmp_path / "out",
            progress_callback=None,
            science_config=SimpleNamespace(cell_gauge_mode="per_cell"),
            zconfig=zconfig,
            halo_px=_HARNESS_HALO,
            scheduler_mode=zsched.MODE_ACTIVE,
            resume_mode="auto",
        )

    assert received, "the real planner entry was never reached"
    assert received["frozen_admission"] == env.admission, (
        "the production path did not thread the bootstrap's frozen admission"
    )
    assert received["restored_geometry"] is not None
    assert received["active"] is True
    # The typed refusal handler released the single-writer lock.
    again = zrs.RunCellStore.open(env.root, identity=None)
    again.close()


def test_fresh_plan_request_has_no_frozen_admission(tmp_path, monkeypatch):
    """Negative control: a store WITHOUT the section passes ``None`` (conservative)."""
    env = _resumable_harness_run(tmp_path, monkeypatch)
    # Rewrite the state exactly like a pre-parity store (no admission section).
    state = zres.read_state(env.root)
    zres.write_state(
        env.root,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=state["policy"],
        plan_generation=state["plan_generation"],
        finalized=state["finalized"],
        gauge_sha256=state["gauge_sha256"],
        created_unix_ns=state["created_unix_ns"],
        note=state["note"],
    )
    received = {}

    def recorder(**kwargs):
        received.update(kwargs)
        raise zres.ResumeFrozenAdmissionUnavailable(
            "recorded", frozen_backend="cpu", frozen_tile_px=1
        )

    monkeypatch.setattr(zz, "_plan_bundle_auto", recorder)
    zconfig = SimpleNamespace(
        zegrid_durable_store_dir=str(env.root),
        zegrid_durable_store_budget_bytes=64 * MIB,
        temporary_work_dir="",
    )
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._run_single(
            env.frames_info,
            env.input_dir,
            tmp_path / "out",
            progress_callback=None,
            science_config=SimpleNamespace(cell_gauge_mode="per_cell"),
            zconfig=zconfig,
            halo_px=_HARNESS_HALO,
            scheduler_mode=zsched.MODE_ACTIVE,
            resume_mode="auto",
        )
    assert received["frozen_admission"] is None


# ---------------------------------------------------------------------------
# 5b — the FROZEN BACKEND governs the plan in BOTH directions
# ---------------------------------------------------------------------------

VRAM_STUB = 4 * 2**30


def test_cpu_frozen_plan_ignores_an_available_gpu(bundle_env):
    """A CPU-frozen run must plan CPU even when this host offers a GPU."""
    # Control: the same fixture DOES produce a feasible GPU plan when nothing is
    # frozen — this is exactly the defect scenario of the real user run.
    control = _bundle(bundle_env, raw=RAW_HIGH, gpu_used=True, vram=VRAM_STUB)
    assert control.envelope.backend == "gpu"
    assert control.gpu_used is True

    frozen = _bundle(
        bundle_env, raw=RAW_HIGH, frozen=_admission(64), gpu_used=True,
        vram=VRAM_STUB,
    )
    assert frozen.envelope.backend == "cpu"
    assert frozen.gpu_used is False
    assert frozen.gpu_tile_size is None
    assert int(frozen.cpu_streaming_tile) == 64

    # The resumed identity equals the stored one BY CONSTRUCTION.
    plain = _bundle(bundle_env, raw=RAW_HIGH, frozen=_admission(64))
    assert int(plain.cpu_streaming_tile) == 64
    assert zrs.identity_sha256(
        _identity_from_bundle(bundle_env, frozen)
    ) == zrs.identity_sha256(_identity_from_bundle(bundle_env, plain))


def test_run_single_forces_cpu_and_emits_one_gpu_ignored_line(
    tmp_path, monkeypatch
):
    """The launch decision honours the frozen backend (one explicit log line)."""
    env = _resumable_harness_run(tmp_path, monkeypatch)
    received = {}

    def recorder(**kwargs):
        received.update(kwargs)
        raise zres.ResumeFrozenAdmissionUnavailable(
            "recorded by the frozen-backend probe",
            frozen_backend="cpu",
            frozen_tile_px=int(env.admission["tile_px"]),
        )

    monkeypatch.setattr(zz, "_plan_bundle_auto", recorder)
    lines = []

    def progress(msg, *_args, **_kwargs):
        lines.append(str(msg))

    zconfig = SimpleNamespace(
        zegrid_durable_store_dir=str(env.root),
        zegrid_durable_store_budget_bytes=64 * MIB,
        temporary_work_dir="",
    )
    gpu_context = {
        "effective": True,
        "requested": True,
        "available": True,
        "device": "stub-gpu",
        "vram_budget_bytes": VRAM_STUB,
    }
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._run_single(
            env.frames_info,
            env.input_dir,
            tmp_path / "out",
            progress_callback=progress,
            science_config=SimpleNamespace(cell_gauge_mode="per_cell"),
            zconfig=zconfig,
            halo_px=_HARNESS_HALO,
            scheduler_mode=zsched.MODE_ACTIVE,
            resume_mode="auto",
            gpu_context=gpu_context,
        )
    assert received["bundle_common"]["gpu_used"] is False, (
        "the frozen CPU backend did not force the planning backend"
    )
    ignored = [
        line for line in lines
        if "GPU ignored" in line and "CPU-frozen run" in line
    ]
    assert len(ignored) == 1, ignored
    assert "backend=cpu" in ignored[0]
    again = zrs.RunCellStore.open(env.root, identity=None)
    again.close()


def test_pre_parity_store_keeps_the_host_gpu_selection(tmp_path, monkeypatch):
    """No admission section -> today's behaviour (GPU selection untouched)."""
    env = _resumable_harness_run(tmp_path, monkeypatch)
    state = zres.read_state(env.root)
    zres.write_state(
        env.root,
        run_id=state["run_id"],
        expected_cell_ids=state["expected_cell_ids"],
        policy=state["policy"],
        plan_generation=state["plan_generation"],
        finalized=state["finalized"],
        gauge_sha256=state["gauge_sha256"],
        created_unix_ns=state["created_unix_ns"],
        note=state["note"],
    )
    received = {}

    def recorder(**kwargs):
        received.update(kwargs)
        raise zres.ResumeFrozenAdmissionUnavailable(
            "recorded", frozen_backend="cpu", frozen_tile_px=1
        )

    monkeypatch.setattr(zz, "_plan_bundle_auto", recorder)
    lines = []
    zconfig = SimpleNamespace(
        zegrid_durable_store_dir=str(env.root),
        zegrid_durable_store_budget_bytes=64 * MIB,
        temporary_work_dir="",
    )
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._run_single(
            env.frames_info,
            env.input_dir,
            tmp_path / "out",
            progress_callback=lambda msg, *_a, **_k: lines.append(str(msg)),
            science_config=SimpleNamespace(cell_gauge_mode="per_cell"),
            zconfig=zconfig,
            halo_px=_HARNESS_HALO,
            scheduler_mode=zsched.MODE_ACTIVE,
            resume_mode="auto",
            gpu_context={
                "effective": True,
                "requested": True,
                "available": True,
                "device": "stub-gpu",
                "vram_budget_bytes": VRAM_STUB,
            },
        )
    assert received["frozen_admission"] is None
    assert received["bundle_common"]["gpu_used"] is True
    assert not [line for line in lines if "GPU ignored" in line]


# ---------------------------------------------------------------------------
# 6 — the store/state CREATION path freezes the admission once
# ---------------------------------------------------------------------------


def test_admission_from_identity_projects_the_validated_backend_block():
    for tile, parity in ((64, False), (80, True), (128, False)):
        identity = make_identity(tile_px=tile, parity=parity)
        admission = zz._admission_from_identity(identity)
        block = identity["science_contract"]["backend"]
        assert admission == zres.build_admission(
            backend=block["backend"],
            tile_px=block["tile_px"],
            parity_qualified=block["parity_qualified"],
        )
        assert admission["tile_px"] == tile
        assert admission["parity_qualified"] is parity
        # The exact production write/read contract preserves it.
        import tempfile

        root = Path(tempfile.mkdtemp()) / "run"
        store = _make_store(root, identity)
        try:
            _write_state(root, run_id=store.run_id, admission=admission)
        finally:
            store.close()
        assert zres.read_state(root)["admission"] == admission


def test_creation_site_uses_the_identity_projection():
    source = (REPO_ROOT / "src" / "zemosaic" / "zemosaic_zegrid_mode.py").read_text(
        encoding="utf-8"
    )
    assert "admission=_admission_from_identity(_store_identity)" in source


def test_fresh_active_tiny_run_freezes_the_admission_from_its_identity(tmp_path):
    """REAL creation path: a fresh ACTIVE durable run writes the freeze once."""
    corpus = tmp_path / "in"
    proc = subprocess.run(
        [sys.executable, str(TINY_CORPUS_TOOL), str(corpus)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    store_root = tmp_path / "store"
    zconfig = SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store_root),
        zegrid_resume_mode="never",
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )
    zz.run_zegrid_mode(
        str(corpus), str(tmp_path / "out"), zconfig=zconfig, zegrid_halo_px=8,
        use_gpu=False,
    )
    runs = [
        child
        for child in (store_root, *store_root.iterdir())
        if (child / "run.json").is_file()
    ]
    assert len(runs) == 1, runs
    root = runs[0]
    identity = json.loads((root / "run.json").read_text(encoding="utf-8"))
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    block = identity["science_contract"]["backend"]
    assert state["admission"] == {
        "schema": zres.ADMISSION_SCHEMA,
        "schema_version": zres.ADMISSION_VERSION,
        "backend": block["backend"],
        "tile_px": block["tile_px"],
        "parity_qualified": block["parity_qualified"],
    }
    assert state["admission"]["tile_px"] == identity["science_contract"]["grid"]["tile_px"]


# ---------------------------------------------------------------------------
# 6b — R3: progress accounting on resume (frozen committed count)
# ---------------------------------------------------------------------------


def _task_list(cell_ids):
    return [(SimpleNamespace(cell_id=cid),) for cid in cell_ids]


def test_resume_stack_accounting_uses_the_frozen_plan(tmp_path, monkeypatch):
    """A real frozen plan with N commits yields done_initial=N, total unchanged."""
    env = _resumable_harness_run(tmp_path, monkeypatch)
    store = zrs.RunCellStore.open(
        env.root, identity=None, budget=_budget(), session_id="r3-commit",
    )
    registry = zrr.RunResultRegistry(
        run_id="parity-wiring",
        expected_cell_ids=list(HARNESS.CELL_IDS),
        gauge_mode="per_cell",
    )
    try:
        for index in range(2):
            cid = HARNESS.CELL_IDS[index]
            payload = HARNESS.compute_cell(index)
            receipt = registry.accept(
                payload, backend="cpu", plan_generation=0,
                attempt_id=f"r3:{cid}",
            )
            store.commit_and_release(
                registry, cid, attempt_id=receipt.attempt_id,
                backend="cpu", plan_generation=0,
            )
    finally:
        registry.dispose()
        store.close()

    boot = zres.ResumeBootstrap.open(env.root)
    try:
        plan = boot.plan(list(HARNESS.CELL_IDS))
    finally:
        boot.close()
    assert plan["complete"] == 2 and plan["pending_count"] == 2
    accounting = zz._resume_stack_accounting(plan)
    assert accounting["committed"] == 2
    assert accounting["total"] == 4
    assert accounting["pending"] == list(HARNESS.CELL_IDS[2:])
    assert (
        zz._resume_progress_line(2, 4, 2)
        == "ZeGrid resume: 2/4 cells already committed, 2 to stack"
    )
    # Completion: offset + pending == total exactly.
    assert accounting["committed"] + len(accounting["pending"]) == accounting["total"]


def test_stack_item_names_the_executed_pending_cell():
    full = _task_list(["r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001"])
    pending = full[2:]
    # Resume: the ACTUAL executed list (never a full-plan index lookup).
    assert zz._stack_item_cid(pending, 1) == "r0001c0000"
    assert zz._stack_item_cid(pending, 2) == "r0001c0001"
    # Fresh run: the full list is the executed list -> unchanged behaviour.
    assert zz._stack_item_cid(full, 1) == "r0000c0000"
    assert zz._stack_item_cid(full, 4) == "r0001c0001"
    assert zz._stack_item_cid(pending, 0) is None
    assert zz._stack_item_cid(pending, 3) is None


def test_eta_denominator_is_the_remaining_count():
    """The real estimator, fed the offset-adjusted done, yields remaining/rate."""
    eta = zobs.PhaseEta(min_samples=2)
    eta.observe(0.0, 47.0)
    eta.observe(10.0, 57.0)  # 1 cell/s
    assert eta.eta_seconds(108) == pytest.approx(51.0)
    # The pre-fix session-local counter (done=10) would have claimed 98s.
    assert eta.eta_seconds(108) != pytest.approx(98.0)


def test_resume_progress_real_run_starts_from_the_committed_count(tmp_path):
    """END-TO-END: a real resume at 2/4 committed reports 2/4, not 1/4."""
    corpus = tmp_path / "in"
    proc = subprocess.run(
        [sys.executable, str(TINY_CORPUS_TOOL), str(corpus)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    store_root = tmp_path / "store"
    out = tmp_path / "out"
    base = dict(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store_root),
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )
    zz.run_zegrid_mode(
        str(corpus), str(out),
        zconfig=SimpleNamespace(**base, zegrid_resume_mode="never"),
        zegrid_halo_px=8, use_gpu=False,
    )
    log_path = out / "zegrid_run.log"
    fresh_log = log_path.read_text(encoding="utf-8")
    # FRESH run: unchanged (the first cell is reported as 1/4).
    assert "per_cell_stack 1/4" in fresh_log
    assert "ZeGrid resume:" not in fresh_log

    # Crash simulation: the LAST two of the four commits disappear.
    commits = sorted((store_root / "commits").glob("*.json"))
    assert len(commits) == 4, [p.name for p in commits]
    for path in commits[2:]:
        path.unlink()

    lines: list[str] = []
    zz.run_zegrid_mode(
        str(corpus), str(out),
        lambda *args, **_kwargs: lines.append(str(args[0])),
        zconfig=SimpleNamespace(**base, zegrid_resume_mode="auto"),
        zegrid_halo_px=8, use_gpu=False,
    )
    resume_log = log_path.read_text(encoding="utf-8")
    resume_lines = [line for line in lines if "ZeGrid resume:" in line]
    assert resume_lines == [
        "[ZEGRID] ZeGrid resume: 2/4 cells already committed, 2 to stack"
    ], resume_lines
    progress = [
        line for line in resume_log.splitlines()
        if "per_cell_stack" in line and "PROGRESS" in line
    ]
    assert progress, resume_log[-2000:]
    assert "per_cell_stack 2/4" in progress[0]
    assert "item=r0001c0000" in progress[0]
    assert "per_cell_stack 1/4" not in resume_log
    final = [line for line in progress if "per_cell_stack 4/4" in line]
    assert final, progress
    # The completion label is the cell ACTUALLY processed last, not the
    # full-plan index (whose entry would be r0000c0001).
    assert "item=r0001c0001" in final[-1], final[-1]


# ---------------------------------------------------------------------------
# 6c — R4: the published products are DISCOVERABLE at the output root
# ---------------------------------------------------------------------------


def _tiny_durable_run(tmp_path, *, resume_mode="never", progress=None):
    corpus = tmp_path / "in"
    if not corpus.exists():
        proc = subprocess.run(
            [sys.executable, str(TINY_CORPUS_TOOL), str(corpus)],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
    store_root = tmp_path / "store"
    out = tmp_path / "out"
    zconfig = SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store_root),
        zegrid_resume_mode=str(resume_mode),
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )
    zz.run_zegrid_mode(
        str(corpus), str(out), progress,
        zconfig=zconfig, zegrid_halo_px=8, use_gpu=False,
    )
    return store_root, out


def test_published_products_are_visible_at_the_output_root(tmp_path):
    _store, out = _tiny_durable_run(tmp_path)
    marker = out / zpub.PRODUCTS_JSON_FILENAME
    text = out / zpub.PRODUCTS_TEXT_FILENAME
    assert marker.is_file() and text.is_file(), sorted(p.name for p in out.iterdir())
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["schema"] == zpub.PRODUCTS_SCHEMA
    assert payload["generation"] == "gen-1"
    gen_dir = Path(payload["generation_dir"])
    assert gen_dir.is_absolute() and gen_dir.is_dir()
    assert gen_dir.is_relative_to(out.resolve())
    publication = json.loads((gen_dir / "publication.json").read_text(encoding="utf-8"))
    members = publication["members"]
    assert set(payload["products"]) == set(members)
    for role, row in members.items():
        product = payload["products"][role]
        assert product["file"] == row["file"]
        assert product["sha256"] == row["sha256"]
        assert product["nbytes"] == row["nbytes"]
        assert product["path"] == str(gen_dir / row["file"])
    resume_text = text.read_text(encoding="utf-8")
    # ZM-ZEGRID-ROOT-DELIVERABLES: every PRESENT member is named at its
    # delivered ROOT path when delivery succeeded, else at its generation path
    # (aesthetic/uint16 are optional and only listed when produced).
    root_files = payload["root_files"]
    for role, row in members.items():
        delivered = root_files.get(role)
        expected = (
            delivered["path"]
            if isinstance(delivered, dict) and delivered.get("path")
            else str(gen_dir / row["file"])
        )
        assert expected in resume_text
    assert zpub.ROLE_SCIENCE in members
    assert zpub.ROLE_COVERAGE in members
    assert zpub.ROLE_MANIFEST in members
    # The finished FITS/manifest are DELIVERED at the root, byte-identical to
    # the immutable archive, and the two entry points are regular files.
    for role in (zpub.ROLE_SCIENCE, zpub.ROLE_COVERAGE, zpub.ROLE_MANIFEST):
        delivered = root_files[role]
        root_path = Path(delivered["path"])
        assert root_path.parent == out.resolve()
        assert root_path.read_bytes() == (gen_dir / members[role]["file"]).read_bytes()
    assert not marker.is_symlink() and not text.is_symlink()


def test_second_generation_updates_the_visible_pointer_only_after_it_is_durable(
    tmp_path,
):
    store_root, out = _tiny_durable_run(tmp_path)
    marker = out / zpub.PRODUCTS_JSON_FILENAME
    text = out / zpub.PRODUCTS_TEXT_FILENAME
    first = {
        "json": marker.read_bytes(),
        "text": text.read_bytes(),
    }
    first_payload = json.loads(first["json"])
    assert first_payload["generation"] == "gen-1"
    gen1_publication = (
        Path(first_payload["generation_dir"]) / "publication.json"
    ).read_bytes()

    # Simulate a crash with the LAST two cells missing, then resume/republish.
    commits = sorted((store_root / "commits").glob("*.json"))
    assert len(commits) == 4
    for path in commits[2:]:
        path.unlink()
    _tiny_durable_run(tmp_path, resume_mode="auto")

    second_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert second_payload["generation"] == "gen-2"
    gen2_dir = Path(second_payload["generation_dir"])
    assert gen2_dir.name == "gen-2" and gen2_dir.is_dir()
    gen2_publication = json.loads(
        (gen2_dir / "publication.json").read_text(encoding="utf-8")
    )
    for role, row in gen2_publication["members"].items():
        assert second_payload["products"][role]["sha256"] == row["sha256"]
        assert second_payload["products"][role]["nbytes"] == row["nbytes"]
    assert marker.read_bytes() != first["json"]
    assert text.read_bytes() != first["text"]
    # The immutable gen-1 publication was never touched.
    assert (
        Path(first_payload["generation_dir"]) / "publication.json"
    ).read_bytes() == gen1_publication


def test_visible_pointer_failure_keeps_previous_bytes(tmp_path, monkeypatch):
    root = tmp_path / "out"
    root.mkdir()
    gen1 = root / "gen-1"
    gen1.mkdir()
    (gen1 / "science.fits").write_bytes(b"one")
    gen2 = root / "gen-2"
    gen2.mkdir()
    (gen2 / "science.fits").write_bytes(b"two")

    def _generation(name, directory):
        return zpub.PublishedGeneration(
            run_id="r4-run",
            generation=name,
            generation_dir=directory,
            pointer_path=root / "CURRENT.json",
            publication_path=directory / "publication.json",
            publication_sha256="a" * 64,
            members={
                zpub.ROLE_SCIENCE: {
                    "file": "science.fits",
                    "sha256": "b" * 64,
                    "nbytes": 3,
                }
            },
            required_roles=(zpub.ROLE_SCIENCE,),
            optional_roles={},
            created_unix_ns=1,
            identity_sha256=None,
            verified=True,
        )

    zpub.write_visible_products(root, _generation("gen-1", gen1))
    names = (zpub.PRODUCTS_JSON_FILENAME, zpub.PRODUCTS_TEXT_FILENAME)
    before = {name: (root / name).read_bytes() for name in names}

    real_replace = zpub.os.replace

    def flaky(src, dst):
        # Fail the FORWARD replace of the human-readable file; the rollback of
        # the already-replaced JSON must still be allowed to run.
        if (
            Path(dst).name == zpub.PRODUCTS_TEXT_FILENAME
            and not Path(src).name.startswith(f".{zpub.PRODUCTS_TEXT_FILENAME}.rb.")
        ):
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(zpub.os, "replace", flaky)
    with pytest.raises(OSError):
        zpub.write_visible_products(root, _generation("gen-2", gen2))
    after = {name: (root / name).read_bytes() for name in names}
    assert after == before


def test_completion_message_and_open_target_point_at_the_output_root(
    tmp_path, monkeypatch
):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:  # pragma: no cover - app bootstrap
        QApplication([])
    from test_zegrid_scheduler_gui_advanced import _make_window

    out = tmp_path / "out"
    gen = out / ".zegrid_publications" / "run-x" / "gen-1"
    gen.mkdir(parents=True)
    (gen / "mosaic_grid_science.fits").write_bytes(b"science")
    (out / "mosaic_grid_science.fits").write_bytes(b"science")
    (out / zpub.PRODUCTS_JSON_FILENAME).write_text(
        json.dumps(
            {
                "schema": zpub.PRODUCTS_SCHEMA,
                "run_id": "r",
                "generation": "gen-1",
                "generation_dir": str(gen),
                "products": {},
                "root_files": {
                    "science": {
                        "path": str(out / "mosaic_grid_science.fits"),
                        "sha256": hashlib.sha256(b"science").hexdigest(),
                        "nbytes": len(b"science"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    win = _make_window(tmp_path, monkeypatch, {"output_dir": str(out)})
    win._zegrid_active = True
    # ZM-ZEGRID-ROOT-DELIVERABLES: open the OUTPUT ROOT (which now holds the
    # delivered FITS) and name those root files, never the generation subdir.
    assert win._zegrid_products_dir().resolve() == out.resolve()
    assert win._completion_open_target().resolve() == out.resolve()
    message = win._completion_message_text()
    assert "mosaic_grid_science.fits" in message
    assert str(out) in message
    assert str(gen) not in message

    # Classic/SDS/legacy (not a ZeGrid run): today's behaviour, no marker use.
    win._zegrid_active = False
    assert win._completion_open_target().resolve() == out.resolve()
    assert "mosaic_grid_science.fits" not in win._completion_message_text()

    # A ZeGrid run without a publication: output folder, unchanged.
    (out / zpub.PRODUCTS_JSON_FILENAME).unlink()
    win._zegrid_active = True
    assert win._completion_open_target().resolve() == out.resolve()
    assert "mosaic_grid_science.fits" not in win._completion_message_text()


# ---------------------------------------------------------------------------
# 7 — the run snapshot records the EFFECTIVE resume mode
# ---------------------------------------------------------------------------


def test_snapshot_for_run_applies_the_effective_resume_mode(tmp_path, monkeypatch):
    """C3: the PRODUCTION snapshot builder must apply the per-launch decision."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:  # pragma: no cover - app bootstrap
        QApplication([])

    from test_zegrid_scheduler_gui_advanced import _make_window

    win = _make_window(tmp_path, monkeypatch, {"output_dir": str(tmp_path / "out")})
    win._pending_zegrid_resume = {
        "mode": "auto",
        "run_id": "run-1",
        "store_dir": "/tmp/store",
    }
    snapshot = win._snapshot_for_run()
    assert snapshot["zegrid_resume_mode"] == "auto"
    assert snapshot["zegrid_durable_store_dir"] == "/tmp/store"
    assert snapshot["zegrid_resume_run_id"] == "run-1"

    win._pending_zegrid_resume = {"mode": "never", "run_id": None, "store_dir": None}
    fresh = win._snapshot_for_run()
    assert fresh["zegrid_resume_mode"] == "never"
    assert fresh["zegrid_durable_store_dir"] is None


def test_snapshot_records_the_effective_resume_mode():
    gui = pytest.importorskip("zemosaic.zemosaic_gui_qt")

    class _Stub:
        _pending_zegrid_resume = {
            "mode": "auto",
            "run_id": "run-1",
            "store_dir": "/tmp/store",
        }

    snapshot = {
        "zegrid_resume_mode": "never",
        "zegrid_durable_store_dir": None,
        "zegrid_resume_run_id": None,
    }
    out = gui.ZeMosaicQtMainWindow._apply_effective_zegrid_resume_to_snapshot(
        _Stub(), snapshot
    )
    assert out["zegrid_resume_mode"] == "auto"
    assert out["zegrid_durable_store_dir"] == "/tmp/store"
    assert out["zegrid_resume_run_id"] == "run-1"

    # A fresh launch records "never" and no store, exactly like the transport.
    class _Fresh:
        _pending_zegrid_resume = {"mode": "never", "run_id": None, "store_dir": None}

    fresh = gui.ZeMosaicQtMainWindow._apply_effective_zegrid_resume_to_snapshot(
        _Fresh(), {"zegrid_resume_mode": "auto"}
    )
    assert fresh["zegrid_resume_mode"] == "never"
    assert fresh["zegrid_durable_store_dir"] is None


def test_start_processing_uses_the_snapshot_builder_before_logging():
    source = (
        REPO_ROOT / "src" / "zemosaic" / "zemosaic_gui_qt.py"
    ).read_text(encoding="utf-8")
    built = source.index("run_snapshot = self._snapshot_for_run()")
    logged = source.index("self._log_run_config_snapshot(run_snapshot)")
    saved = source.index("self._save_config()", logged)
    assert built < logged < saved
    # The builder is the ONLY place that applies the per-launch decision.
    assert "self._apply_effective_zegrid_resume_to_snapshot(run_snapshot)" not in source
