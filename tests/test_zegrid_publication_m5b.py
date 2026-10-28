"""ZM-ZEGRID-RESILIENT-RUN-RESUME M5-B — coherent generation publication.

These tests exercise the REAL product writers (``_write_outputs``,
``_write_raw_science_fits``, FITS/astropy, ``run_store`` state) against the
run-owned publication generation, then verify through a NEW process when the
scenario requires it.

Covered:
* complete generation (SCI + coverage + manifest + optional aesthetic/uint16):
  real FITS pixels, manifest names, digests, pointer written last;
* previous generation remains authoritative until the pointer moves, including
  a real subprocess killed during finalization;
* missing/corrupt member or companion: explicit refusal, never a fake COMPLETE;
* lost state acknowledgement: publication reconciled without recomputation;
* disk/RAM refusal BEFORE allocation; ENOSPC/write failure preserves the prior
  generation; path escape/symlink refused; foreign files protected;
* store -> bounded assembly -> finishing -> publication -> reopen: identical
  canvas, no double offset, commit digests unchanged.

Windows/GPU/real-pressure are NOT_RUN on this host.
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
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(SRC), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import bounded_assembly as zbas  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import test_zegrid_bounded_assembly_m5 as m5a  # noqa: E402

H = 12
W = 12
CH = 3
RUN_ID = "run-m5b"
POLICY = {
    "requested": "dense",
    "effective": "dense",
    "fallback_count": 0,
    "transitions": [],
    "dense_refusal": None,
}


# ---------------------------------------------------------------------------
# Real product writers
# ---------------------------------------------------------------------------


def _synthetic_wcs(shape=(H, W)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


def _canvas(h=H, w=W):
    return m5a.make_canvas(h=h, w=w)


def _assembled(science=None, stack_depth=None):
    class _A:
        pass

    a = _A()
    a.science = (
        np.linspace(0, 99, H * W * CH, dtype=np.float32).reshape(H, W, CH)
        if science is None
        else np.asarray(science, dtype=np.float32)
    )
    sh, sw = a.science.shape[:2]
    a.stack_depth = (
        np.ones((sh, sw), dtype=np.int32)
        if stack_depth is None
        else np.asarray(stack_depth)
    )
    a.complete_cells = 1
    a.incomplete_cells = 0
    a.hole_pixels = 0
    a.coverage_pixels = H * W
    return a


def _descriptors():
    from zemosaic.core.zegrid import geometry as zg

    return [
        zg.FrameDescriptor(
            frame_id=zg.FrameId("a.fits"),
            source_path="/x/a.fits",
            shape_hw=(H, W),
            wcs_header=_synthetic_wcs().to_header().tostring(),
            header_sha256="",
            instrument="",
        )
    ]


def _layout():
    return {
        "nx": 1,
        "ny": 1,
        "ram_budget_bytes": 0,
        "available_bytes": 0,
        "max_contributors": 1,
        "predicted_bound_bytes": 0,
    }


def _finishing_config(*, export=False, uint16=False):
    return {
        "dbe_enabled": False,
        "rgb_equalize": False,
        "hole_fill_enabled": False,
        "save_uint16": bool(uint16),
        "export_aesthetic_fits": bool(export),
        "scientific_fits_suffix": "_science",
        "aesthetic_fits_suffix": "_aesthetic",
    }


def _members(*, export=False, uint16=False):
    members = {
        zpub.ROLE_SCIENCE: (
            "mosaic_grid_science.fits" if export else "mosaic_grid.fits"
        ),
        zpub.ROLE_COVERAGE: "mosaic_grid_coverage.fits",
        zpub.ROLE_MANIFEST: "zegrid_manifest.json",
        zpub.ROLE_RUN_LOG: "zegrid_run.log",
    }
    if export:
        members[zpub.ROLE_AESTHETIC] = "mosaic_grid_aesthetic.fits"
    if uint16:
        members[zpub.ROLE_UINT16] = "mosaic_grid_uint16.fits"
    return members


def _begin(out, *, run_id=RUN_ID, export=False, uint16=False, ram=8 * 2 ** 30):
    return zpub.PublicationSession.begin(
        out,
        run_id,
        members=_members(export=export, uint16=uint16),
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_config(export=export, uint16=uint16),
        available_ram_bytes=ram,
        n_cells=1,
    )


def _stage_generation(
    out,
    pub,
    *,
    export=False,
    uint16=False,
    science=None,
    finished=None,
):
    assembled = _assembled(science)
    h, w = assembled.science.shape[:2]
    canvas = _canvas(h=h, w=w)
    finished_science = (
        np.asarray(assembled.science, dtype=np.float32).copy()
        if finished is None
        else np.asarray(finished, dtype=np.float32)
    )
    u16 = (
        np.zeros((h, w, CH), dtype=np.uint16) + 4242 if uint16 else None
    )
    raw_path = pub.path_for(zpub.ROLE_SCIENCE)
    zz._write_raw_science_fits(
        assembled,
        canvas,
        raw_path,
        related_file=(
            pub.members[zpub.ROLE_AESTHETIC] if export else None
        ),
    )
    sci, cov, manifest = zz._write_outputs(
        assembled,
        canvas,
        1,
        1,
        out,
        _descriptors(),
        {},
        [],
        _layout(),
        SimpleNamespace(normalization="sky_mean", cell_gauge_mode="per_cell"),
        0,
        {},
        None,
        raw_science_path=raw_path,
        raw_science=assembled.science,
        aesthetic_path=(
            pub.path_for(zpub.ROLE_AESTHETIC) if export else None
        ),
        export_aesthetic_fits=bool(export),
        finished_science=finished_science,
        finishing_info={"enabled": False, "failed": False, "failure_reason": ""},
        fin_uint16=u16,
        scientific_fits_suffix="_science",
        publication=pub,
    )
    (Path(out) / "zegrid_run.log").write_text("live run log\n", encoding="utf-8")
    pub.include_auxiliary(zpub.ROLE_RUN_LOG, Path(out) / "zegrid_run.log")
    return assembled, finished_science, (sci, cov, manifest)


def _required_roles():
    return [
        zpub.ROLE_SCIENCE,
        zpub.ROLE_COVERAGE,
        zpub.ROLE_MANIFEST,
        zpub.ROLE_RUN_LOG,
    ]


def _write_generation(
    out,
    pub,
    *,
    export=False,
    uint16=False,
    science=None,
    finished=None,
):
    assembled, finished_science, paths = _stage_generation(
        out,
        pub,
        export=export,
        uint16=uint16,
        science=science,
        finished=finished,
    )
    optional = {}
    if export:
        optional[zpub.ROLE_AESTHETIC] = True
    if uint16:
        optional[zpub.ROLE_UINT16] = True
    generation = pub.commit(
        required_roles=_required_roles(), optional_roles=optional
    )
    return generation, assembled, finished_science, paths


def _publish(out, *, run_id=RUN_ID, export=False, uint16=False, **kw):
    pub = _begin(out, run_id=run_id, export=export, uint16=uint16)
    return _write_generation(out, pub, export=export, uint16=uint16, **kw)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _world_state(out):
    return {
        str(p.relative_to(out)): _digest(p)
        for p in sorted(Path(out).rglob("*"))
        if p.is_file() and not p.name.startswith(".tmp")
    }


# ---------------------------------------------------------------------------
# 1. Complete generation: real FITS, manifest contract, pointer last
# ---------------------------------------------------------------------------


def test_m5b_complete_generation_is_resolvable_with_real_fits(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, assembled, finished, (sci, cov, manifest) = _publish(out)

    assert gen.generation == "gen-1"
    assert gen.pointer_path.name == zpub.POINTER_FILENAME
    assert gen.pointer_path.is_file()
    assert sci.parent == gen.generation_dir
    assert cov.parent == gen.generation_dir
    assert manifest.parent == gen.generation_dir

    resolved = zpub.resolve_current(out, RUN_ID)
    assert resolved.generation == "gen-1"
    assert set(resolved.members) == {
        zpub.ROLE_SCIENCE,
        zpub.ROLE_COVERAGE,
        zpub.ROLE_MANIFEST,
        zpub.ROLE_RUN_LOG,
    }
    # Real FITS pixels: science is (C,H,W) float32, coverage is the stack depth.
    sci_data = fits.getdata(resolved.path(zpub.ROLE_SCIENCE))
    assert sci_data.dtype.kind == "f" and sci_data.dtype.itemsize == 4
    assert sci_data.shape == (CH, H, W)
    np.testing.assert_array_equal(
        sci_data, np.moveaxis(finished, -1, 0)
    )
    cov_data = fits.getdata(resolved.path(zpub.ROLE_COVERAGE))
    np.testing.assert_array_equal(cov_data, assembled.stack_depth.astype(np.float32))
    payload = json.loads(resolved.path(zpub.ROLE_MANIFEST).read_text())
    assert payload["outputs"]["science"] == "mosaic_grid.fits"
    assert payload["outputs"]["coverage"] == "mosaic_grid_coverage.fits"
    assert payload["outputs"]["aesthetic"] is None
    assert payload["outputs"]["uint16"] is None
    assert payload["science_reference"] == "mosaic_grid.fits"
    # A reader can resolve every member from ONE generation directory.
    for role in resolved.required_roles:
        assert resolved.path(role).parent == resolved.generation_dir


def test_m5b_aesthetic_and_uint16_members_are_real_and_declared(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    finished = np.linspace(5, 50, H * W * CH, dtype=np.float32).reshape(H, W, CH)
    gen, assembled, _finished, _paths = _publish(
        out, export=True, uint16=True, finished=finished
    )
    resolved = zpub.resolve_current(out, RUN_ID)
    assert set(resolved.members) == {
        zpub.ROLE_SCIENCE,
        zpub.ROLE_AESTHETIC,
        zpub.ROLE_COVERAGE,
        zpub.ROLE_UINT16,
        zpub.ROLE_MANIFEST,
        zpub.ROLE_RUN_LOG,
    }
    aest = fits.getdata(resolved.path(zpub.ROLE_AESTHETIC))
    np.testing.assert_array_equal(aest, np.moveaxis(finished, -1, 0))
    u16 = fits.getdata(resolved.path(zpub.ROLE_UINT16))
    assert u16.dtype.kind == "u" and u16.dtype.itemsize == 2
    assert np.all(u16 == 4242)
    payload = json.loads(resolved.path(zpub.ROLE_MANIFEST).read_text())
    assert payload["outputs"]["aesthetic"] == "mosaic_grid_aesthetic.fits"
    assert payload["outputs"]["uint16"] == "mosaic_grid_uint16.fits"
    # SCI is still the immutable raw science, not the finished copy.
    raw = fits.getdata(resolved.path(zpub.ROLE_SCIENCE))
    np.testing.assert_array_equal(raw, np.moveaxis(assembled.science, -1, 0))


def test_m5b_optional_uint16_absent_after_finishing_failure_is_valid(tmp_path):
    """The fail-safe path: uint16 requested but finishing produced none.

    The member is declared in the generation, absent, marked optional=False,
    and the manifest says ``uint16: null`` — a coherent generation, not a
    missing companion.
    """
    out = tmp_path / "out"
    out.mkdir()
    pub = _begin(out, uint16=True)
    _stage_generation(out, pub, uint16=False)
    gen = pub.commit(
        required_roles=_required_roles(),
        optional_roles={zpub.ROLE_UINT16: False},
    )
    resolved = zpub.resolve_current(out, RUN_ID)
    assert zpub.ROLE_UINT16 not in resolved.members
    payload = json.loads(resolved.path(zpub.ROLE_MANIFEST).read_text())
    assert payload["outputs"]["uint16"] is None
    assert resolved.publication_sha256 == gen.publication_sha256


# ---------------------------------------------------------------------------
# 2. Previous generation stays authoritative until the pointer moves
# ---------------------------------------------------------------------------


def test_m5b_previous_generation_authoritative_until_pointer(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen1, _a, _f, _p = _publish(out)
    before = zpub.resolve_current(out, RUN_ID)
    old_digests = {role: row["sha256"] for role, row in before.members.items()}

    pub2 = _begin(out)
    # Stage a complete second generation but do NOT commit yet.
    _stage_generation(out, pub2)
    still = zpub.resolve_current(out, RUN_ID)
    assert still.generation == gen1.generation
    assert {r: row["sha256"] for r, row in still.members.items()} == old_digests
    assert pub2.generation_dir.is_dir()

    pub2.commit(required_roles=_required_roles())
    after = zpub.resolve_current(out, RUN_ID)
    assert after.generation == "gen-2"
    # The old generation is still on disk and complete (never deleted).
    assert gen1.generation_dir.is_dir()
    assert zpub.resolve_current(out, RUN_ID).generation_dir != gen1.generation_dir


def test_m5b_pointer_failure_preserves_previous_generation(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)

    real_atomic = zpub._atomic_write

    def _fail_pointer(path, data, *, tmp_dir):
        if Path(path).name == zpub.POINTER_FILENAME:
            raise OSError("simulated pointer failure")
        return real_atomic(path, data, tmp_dir=tmp_dir)

    monkeypatch.setattr(zpub, "_atomic_write", _fail_pointer)
    # Write the second generation through the product writers, then fail at
    # the pointer flip.
    pub2b = _begin(out)
    _stage_generation(out, pub2b)
    with pytest.raises(OSError, match="pointer failure"):
        pub2b.commit(required_roles=_required_roles())
    still = zpub.resolve_current(out, RUN_ID)
    assert still.generation == gen1.generation
    # The failed staging generation is not referenced by the pointer.
    assert (still.pointer_path.parent / "gen-2").is_dir()


def test_m5b_missing_required_member_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)
    pub2 = _begin(out)
    _stage_generation(out, pub2)
    pub2.path_for(zpub.ROLE_COVERAGE).unlink()
    with pytest.raises(zpub.PublicationIntegrityError, match="required member"):
        pub2.commit(required_roles=_required_roles())
    assert zpub.resolve_current(out, RUN_ID).generation == gen1.generation


def test_m5b_missing_declared_aesthetic_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)
    pub2 = _begin(out, export=True)
    _stage_generation(out, pub2, export=True)
    pub2.path_for(zpub.ROLE_AESTHETIC).unlink()
    with pytest.raises(zpub.PublicationIntegrityError, match="declared optional"):
        pub2.commit(
            required_roles=_required_roles(),
            optional_roles={zpub.ROLE_AESTHETIC: True},
        )
    assert zpub.resolve_current(out, RUN_ID).generation == gen1.generation


def test_m5b_corrupt_member_refused_on_resolve(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    target = zpub.resolve_current(out, RUN_ID).path(zpub.ROLE_SCIENCE)
    with open(target, "r+b") as fh:
        fh.seek(4000)
        fh.write(b"\x00\x01\x02\x03")
    with pytest.raises(zpub.PublicationIntegrityError, match="sha256"):
        zpub.resolve_current(out, RUN_ID)


def test_m5b_same_identity_different_content_is_a_collision(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    pub = zpub.PublicationSession.reattach(
        out,
        RUN_ID,
        gen.generation,
        members=_members(),
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_config(),
    )
    # Replace the SCI bytes with different (still valid) FITS content.
    data = fits.getdata(pub.path_for(zpub.ROLE_SCIENCE))
    fits.PrimaryHDU(np.ascontiguousarray(data + 1.0)).writeto(
        pub.path_for(zpub.ROLE_SCIENCE), overwrite=True
    )
    with pytest.raises(zpub.PublicationCollisionError):
        pub.commit(
            required_roles=[
                zpub.ROLE_SCIENCE,
                zpub.ROLE_COVERAGE,
                zpub.ROLE_MANIFEST,
                zpub.ROLE_RUN_LOG,
            ]
        )


def test_m5b_idempotent_identical_commit(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    pub = zpub.PublicationSession.reattach(
        out,
        RUN_ID,
        gen.generation,
        members=_members(),
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_config(),
    )
    again = pub.commit(
        required_roles=[
            zpub.ROLE_SCIENCE,
            zpub.ROLE_COVERAGE,
            zpub.ROLE_MANIFEST,
            zpub.ROLE_RUN_LOG,
        ]
    )
    assert again.publication_sha256 == gen.publication_sha256
    assert zpub.resolve_current(out, RUN_ID).generation == gen.generation


# ---------------------------------------------------------------------------
# 3. Lost acknowledgement / state reconciliation
# ---------------------------------------------------------------------------


def test_m5b_lost_state_ack_reconciles_from_pointer(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    run_root = tmp_path / "store"
    run_root.mkdir()
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=False,
    )
    gen, *_ = _publish(out)
    verdict = zpub.verify_finalized(out, run_root, RUN_ID)
    assert verdict.published is True
    assert verdict.state_finalized is False
    assert verdict.reason == "publication_durable_state_not_updated"
    # Reconcile by writing the state from the durable generation (no recompute).
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=True,
        outputs=gen.outputs_state(),
    )
    assert zpub.verify_finalized(out, run_root, RUN_ID).state_finalized is True


def test_m5b_state_finalized_without_publication_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    run_root = tmp_path / "store"
    run_root.mkdir()
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=True,
        outputs={"science": {"sha256": "0" * 64, "nbytes": 1}},
    )
    with pytest.raises(zpub.PublicationIntegrityError, match="fake COMPLETE"):
        zpub.verify_finalized(out, run_root, RUN_ID)


def test_m5b_stale_state_ack_is_reconciled_from_durable_publication(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    run_root = tmp_path / "store"
    run_root.mkdir()
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=False,
    )
    gen, *_ = _publish(out)
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=True,
        outputs={
            "science": {"sha256": "f" * 64, "nbytes": 1},
            "coverage": {"sha256": "e" * 64, "nbytes": 1},
            "manifest": {"sha256": "d" * 64, "nbytes": 1},
        },
    )
    verdict = zpub.verify_finalized(out, run_root, RUN_ID)
    assert verdict.published is True
    assert verdict.reason == "publication_durable_state_stale"
    assert verdict.generation.publication_sha256 == gen.publication_sha256
    # Collision/corruption of the generation itself is still an explicit refusal.
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=True,
        outputs=gen.outputs_state(),
    )
    assert zpub.verify_finalized(out, run_root, RUN_ID).reason == "verified_finalized"


# ---------------------------------------------------------------------------
# 4. Capacity admission BEFORE allocation
# ---------------------------------------------------------------------------


def test_m5b_ram_capacity_refused_before_any_generation(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(zpub.PublicationCapacityError) as excinfo:
        zpub.PublicationSession.begin(
            out,
            RUN_ID,
            members=_members(uint16=True),
            shape_hw=(H, W),
            channels=CH,
            finishing_config=_finishing_config(uint16=True),
            available_ram_bytes=1024,
            n_cells=1,
        )
    plan = excinfo.value.plan
    assert plan["admitted"] is False
    assert plan["ram"]["fits"] is False
    assert plan["finishing"]["peak_bytes"] > 0
    assert not zpub.publication_root(out).exists()


def test_m5b_disk_capacity_refused_before_any_generation(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(zpub.PublicationCapacityError) as excinfo:
        zpub.admit_finishing_and_publication(
            output_dir=out,
            shape_hw=(H, W),
            channels=CH,
            finishing_config=_finishing_config(),
            available_ram_bytes=8 * 2 ** 30,
            n_cells=1,
            disk_free_bytes=1024,
        )
    assert excinfo.value.plan["disk"]["fits"] is False
    assert excinfo.value.plan["disk"]["transactional_margin_bytes"] > 0


def test_m5b_finishing_peak_models_options_truthfully():
    off = zpub.finishing_peak_bytes((H, W), finishing_config=_finishing_config())
    on = zpub.finishing_peak_bytes(
        (H, W),
        finishing_config={
            "dbe_enabled": True,
            "rgb_equalize": True,
            "hole_fill_enabled": True,
            "save_uint16": True,
        },
    )
    assert off["enabled"] is False
    assert on["enabled"] is True
    assert on["peak_bytes"] > off["peak_bytes"]
    assert on["terms"]["uint16_float64_scale"] > 0
    # Metadata only: no allocation happened (the model is pure arithmetic).
    assert on["model_margin_percent"] == zpub.MODEL_MARGIN_PERCENT


def test_m5b_unobserved_ram_fails_closed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(zpub.PublicationCapacityError) as excinfo:
        zpub.PublicationSession.begin(
            out,
            RUN_ID,
            members=_members(),
            shape_hw=(H, W),
            channels=CH,
            finishing_config=_finishing_config(),
            available_ram_bytes=None,
            ram_probe_error="RuntimeError: probe failed",
            n_cells=1,
        )
    plan = excinfo.value.plan
    assert plan["admitted"] is False
    assert plan["ram"]["observed"] is False
    assert plan["ram"]["probe_error"] == "RuntimeError: probe failed"
    assert not zpub.publication_root(out).exists()
    # Read-only sizing is still available, but is NEVER labelled admitted.
    sized = zpub.size_finishing_and_publication(
        output_dir=out,
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_config(),
        n_cells=1,
    )
    assert sized["admitted"] is False
    assert sized["admission"] == {"mode": "sizing_only", "admitted": False}
    assert sized["ram"]["observed"] is False


# ---------------------------------------------------------------------------
# 5. Write failure / disk full preserves the previous generation
# ---------------------------------------------------------------------------


def test_m5b_enospc_during_member_write_preserves_previous(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)
    pub2 = _begin(out)
    real = zz._atomic_writeto

    def _fail_after_science(hdu, path):
        if Path(path).name == pub2.members[zpub.ROLE_COVERAGE]:
            raise OSError(28, "No space left on device (simulated)")
        return real(hdu, path)

    monkeypatch.setattr(zz, "_atomic_writeto", _fail_after_science)
    with pytest.raises(OSError, match="No space left"):
        _write_generation(out, pub2)
    assert zpub.resolve_current(out, RUN_ID).generation == gen1.generation


def test_m5b_cleanup_removes_only_own_staging(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)
    # A failed staging generation of THIS run.
    orphan = _begin(out)
    (orphan.path_for(zpub.ROLE_SCIENCE)).write_bytes(b"staging")
    # A foreign file and a foreign staging generation of another run.
    (zpub.publication_root(out) / "user-note.txt").write_text("keep me")
    other = _begin(out, run_id="run-other")
    (other.path_for(zpub.ROLE_SCIENCE)).write_bytes(b"other")
    removed = zpub.cleanup_orphan_generations(out, RUN_ID)
    assert orphan.generation not in removed
    assert orphan.generation_dir.is_dir()
    # A dead owner with a provably disposable staged member set is removable.
    marker_path = orphan.generation_dir / zpub.STAGING_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["pid"] = 999999
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    removed = zpub.cleanup_orphan_generations(out, RUN_ID)
    assert orphan.generation in removed
    assert orphan.generation_dir.exists() is False
    assert other.generation_dir.is_dir()
    assert (zpub.publication_root(out) / "user-note.txt").is_file()
    assert gen1.generation_dir.is_dir()
    assert zpub.resolve_current(out, RUN_ID).generation == gen1.generation


# ---------------------------------------------------------------------------
# 6. Path escape / symlink / foreign file protection
# ---------------------------------------------------------------------------


def test_m5b_member_name_traversal_refused(tmp_path):
    for bad in ("../x.fits", "a/b.fits", "..", "C:\\x.fits", "a\x00b"):
        with pytest.raises(zpub.PublicationPathError):
            zpub.member_filename(bad)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(zpub.PublicationPathError):
        zpub.PublicationSession.begin(
            out,
            RUN_ID,
            members={zpub.ROLE_SCIENCE: "../escape.fits"},
            shape_hw=(H, W),
            channels=CH,
            finishing_config=_finishing_config(),
            available_ram_bytes=8 * 2 ** 30,
        )
    assert not zpub.publication_root(out).exists()


def test_m5b_symlinked_publication_root_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.symlink(target, zpub.publication_root(out))
    with pytest.raises(zpub.PublicationPathError, match="symlink"):
        _begin(out)


def test_m5b_symlinked_member_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pub = _begin(out)
    _stage_generation(out, pub)
    other = tmp_path / "other.fits"
    other.write_bytes(b"foreign")
    pub.path_for(zpub.ROLE_SCIENCE).unlink()
    os.symlink(other, pub.path_for(zpub.ROLE_SCIENCE))
    with pytest.raises(zpub.PublicationPathError, match="symlink"):
        pub.commit(required_roles=_required_roles())


def test_m5b_foreign_files_and_prior_generations_never_deleted(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "user_result.fits").write_bytes(b"user data")
    gen1, *_ = _publish(out)
    _publish(out)
    assert (out / "user_result.fits").read_bytes() == b"user data"
    assert gen1.generation_dir.is_dir()
    assert zpub.resolve_current(out, RUN_ID).generation == "gen-2"


# ---------------------------------------------------------------------------
# 7. Real subprocess killed during finalization, then a new process recovers
# ---------------------------------------------------------------------------


def _harness():
    return REPO_ROOT / "tools" / "zegrid_resume_m5b" / "publication_harness.py"


def _run_harness(*args, **kw):
    return subprocess.run(
        [sys.executable, str(_harness()), *args],
        capture_output=True,
        text=True,
        timeout=180,
        **kw,
    )


def test_m5b_kill_during_finalization_then_new_process(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    base = _run_harness("--out", str(out), "--run-id", RUN_ID, "--mode", "baseline",
                        "--generation", "gen-1")
    assert base.returncode == 0, base.stderr
    old = zpub.resolve_current(out, RUN_ID)

    killed = _run_harness(
        "--out", str(out), "--run-id", RUN_ID, "--mode", "kill-before-pointer",
        "--generation", "gen-2",
    )
    assert killed.returncode == 9, (killed.returncode, killed.stderr)
    still = zpub.resolve_current(out, RUN_ID)
    assert still.generation == "gen-1"
    assert {r: row["sha256"] for r, row in still.members.items()} == {
        r: row["sha256"] for r, row in old.members.items()
    }

    recovered = _run_harness(
        "--out", str(out), "--run-id", RUN_ID, "--mode", "recover",
        "--generation", "gen-3", "--value", "5.0",
    )
    assert recovered.returncode == 0, recovered.stderr
    after = zpub.resolve_current(out, RUN_ID)
    assert after.generation == "gen-3"
    assert after.members[zpub.ROLE_SCIENCE]["sha256"] != old.members[
        zpub.ROLE_SCIENCE
    ]["sha256"]
    assert old.generation_dir.is_dir()
    # The killed generation wrote its publication record before the pointer:
    # F3 conservatively preserves it (possibly committed) after the kill.
    assert (after.pointer_path.parent / "gen-2").is_dir()
    assert (after.pointer_path.parent / "gen-2" / zpub.PUBLICATION_FILENAME).is_file()


def test_m5b_new_process_reopens_same_digests(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    probe = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);"
        "from zemosaic.core.zegrid import run_publication as zpub;"
        "g=zpub.resolve_current(sys.argv[2], sys.argv[3]);"
        "print(json.dumps({'generation':g.generation,'digests':"
        "{k:v['sha256'] for k,v in g.members.items()}}))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe, str(SRC), str(out), RUN_ID],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    assert payload["generation"] == gen.generation
    assert payload["digests"] == {
        role: row["sha256"] for role, row in gen.members.items()
    }


# ---------------------------------------------------------------------------
# 8. Store -> bounded assembly -> finishing -> publication -> reopen
# ---------------------------------------------------------------------------


def test_m5b_store_assembly_publication_reopen_no_restack(tmp_path):
    results = [
        m5a.complete_result(0, 0, seed=0),
        m5a.complete_result(0, 1, seed=1),
        m5a.complete_result(1, 0, seed=2),
        m5a.complete_result(1, 1, seed=3),
    ]
    root, store = m5a.seeded_store(tmp_path, results)
    store.close()
    before = m5a.digest_tree(root)
    canvas = m5a.make_canvas()
    out = tmp_path / "out"
    out.mkdir()

    def _assemble():
        st = m5a.reopen_read_only(root)
        try:
            access = zbas.StoreAssemblyAccess(st, max_resident_cells=2)
            bounded = access.results([r["cell_id"] for r in results])
            assembled, info = zz._assemble_cells(
                canvas,
                m5a.NX,
                m5a.NY,
                "per_cell",
                bounded.cores,
                bounded.descriptors,
                halo_px=m5a.HALO,
                patches=bounded.patches,
            )
            return assembled, info, access.stats()
        finally:
            st.close()

    assembled, info, stats = _assemble()
    assert stats["peak_live_cells"] <= 2
    pub = zpub.PublicationSession.begin(
        out,
        RUN_ID,
        members=_members(),
        shape_hw=(assembled.science.shape[0], assembled.science.shape[1]),
        channels=int(assembled.science.shape[2]),
        finishing_config=_finishing_config(),
        available_ram_bytes=8 * 2 ** 30,
        n_cells=len(results),
    )
    _write_generation(
        out, pub, science=assembled.science, finished=assembled.science
    )
    published = zpub.resolve_current(out, RUN_ID)
    on_disk = fits.getdata(published.path(zpub.ROLE_SCIENCE))
    expected = np.nan_to_num(
        np.moveaxis(assembled.science, -1, 0), nan=0.0, posinf=0.0, neginf=0.0
    )
    np.testing.assert_array_equal(on_disk, expected)

    # A NEW read-only open + assembly must produce the SAME canvas and offsets,
    # and must never mutate a durable commit (no restack, no double offset).
    assembled2, info2, _stats2 = _assemble()
    np.testing.assert_array_equal(assembled.science, assembled2.science)
    assert info["offsets"] == info2["offsets"]
    assert info["anchor"] == info2["anchor"]
    assert m5a.digest_tree(root) == before
    assert zpub.resolve_current(out, RUN_ID).publication_sha256 == (
        published.publication_sha256
    )


# ---------------------------------------------------------------------------
# 9. REWORK-1 — immutable publication records (F2)
# ---------------------------------------------------------------------------


def _reattach(out, gen):
    return zpub.PublicationSession.reattach(
        out,
        RUN_ID,
        gen.generation,
        members=_members(),
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_config(),
    )


def test_m5b_exact_retry_performs_no_write(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    pub = _reattach(out, gen)
    before = {
        "publication": gen.publication_path.read_bytes(),
        "pointer": gen.pointer_path.read_bytes(),
        "members": {
            role: pub.path_for(role).read_bytes() for role in pub.members
        },
    }

    def _boom(*args, **kwargs):
        raise AssertionError("exact idempotent retry must not write")

    monkeypatch.setattr(zpub, "_atomic_write", _boom)
    monkeypatch.setattr(zpub.PublicationSession, "_flush_member", _boom)
    again = pub.commit(required_roles=_required_roles())
    assert again.publication_sha256 == gen.publication_sha256
    assert gen.publication_path.read_bytes() == before["publication"]
    assert gen.pointer_path.read_bytes() == before["pointer"]
    for role, data in before["members"].items():
        assert pub.path_for(role).read_bytes() == data


def test_m5b_retry_contract_change_refused_without_writing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    before_publication = gen.publication_path.read_bytes()
    before_pointer = gen.pointer_path.read_bytes()
    pub = _reattach(out, gen)
    with pytest.raises(zpub.PublicationCollisionError, match="immutable"):
        pub.commit(
            required_roles=_required_roles(),
            metadata={"retry": "changed metadata"},
        )
    assert gen.publication_path.read_bytes() == before_publication
    assert gen.pointer_path.read_bytes() == before_pointer
    assert zpub.resolve_current(out, RUN_ID).publication_sha256 == (
        gen.publication_sha256
    )


def test_m5b_retry_identity_change_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen, *_ = _publish(out)
    pub = zpub.PublicationSession.reattach(
        out,
        RUN_ID,
        gen.generation,
        members=_members(),
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_config(),
        identity_sha256="f" * 64,
    )
    with pytest.raises(zpub.PublicationCollisionError):
        pub.commit(required_roles=_required_roles())
    assert zpub.resolve_current(out, RUN_ID).publication_sha256 == (
        gen.publication_sha256
    )


def test_m5b_stale_retry_cannot_roll_pointer_back(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)
    gen2, *_ = _publish(out)
    pointer_before = gen2.pointer_path.read_bytes()
    pub = _reattach(out, gen1)
    with pytest.raises(zpub.PublicationCollisionError, match="backward"):
        pub.commit(required_roles=_required_roles())
    assert gen2.pointer_path.read_bytes() == pointer_before
    assert zpub.resolve_current(out, RUN_ID).generation == gen2.generation


# ---------------------------------------------------------------------------
# 10. REWORK-1 — cleanup preserves possibly committed generations (F3)
# ---------------------------------------------------------------------------


def test_m5b_residual_marker_on_committed_generation_is_preserved(
    tmp_path, monkeypatch
):
    out = tmp_path / "out"
    out.mkdir()
    pub1 = _begin(out)
    _stage_generation(out, pub1)
    real_unlink = Path.unlink

    def _fail_marker(self, *args, **kwargs):
        if self.name == zpub.STAGING_FILENAME:
            raise OSError("simulated marker unlink failure")
        return real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", _fail_marker)
        gen1 = pub1.commit(required_roles=_required_roles())
    assert (gen1.generation_dir / zpub.STAGING_FILENAME).is_file()
    gen2, *_ = _publish(out)
    pub3 = _begin(out)
    zpub.cleanup_orphan_generations(out, RUN_ID)
    assert gen1.generation_dir.is_dir()
    assert (gen1.generation_dir / zpub.PUBLICATION_FILENAME).is_file()
    assert gen2.generation_dir.is_dir()
    assert zpub.resolve_current(out, RUN_ID).generation == gen2.generation
    assert pub3.generation_dir.is_dir()


# ---------------------------------------------------------------------------
# 11. REWORK-1 — ordered durability before the pointer (F4)
# ---------------------------------------------------------------------------


def test_m5b_members_fsynced_before_publication_and_pointer(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def _traced_fsync(fd):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            target = ""
        events.append(("fsync", target))
        return real_fsync(fd)

    def _traced_replace(src, dst):
        events.append(("replace", str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", _traced_fsync)
    monkeypatch.setattr(os, "replace", _traced_replace)
    gen, *_ = _publish(out)
    science = [
        i for i, (kind, target) in enumerate(events)
        if kind == "fsync" and target.endswith("mosaic_grid.fits")
    ]
    coverage = [
        i for i, (kind, target) in enumerate(events)
        if kind == "fsync" and target.endswith("mosaic_grid_coverage.fits")
    ]
    publication = [
        i for i, (kind, target) in enumerate(events)
        if kind == "replace" and target.endswith(zpub.PUBLICATION_FILENAME)
    ]
    pointer = [
        i for i, (kind, target) in enumerate(events)
        if kind == "replace" and target.endswith(zpub.POINTER_FILENAME)
    ]
    assert science, events
    assert coverage, events
    assert publication, events
    assert pointer, events
    assert max(science + coverage) < min(publication)
    assert max(publication) < min(pointer)


def test_m5b_member_flush_failure_keeps_previous_current(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    gen1, *_ = _publish(out)
    pub2 = _begin(out)
    _stage_generation(out, pub2)
    real_flush = zpub.PublicationSession._flush_member

    def _fail_flush(self, path):
        if Path(path).name == pub2.members[zpub.ROLE_COVERAGE]:
            raise OSError(5, "simulated member flush failure")
        return real_flush(self, path)

    monkeypatch.setattr(zpub.PublicationSession, "_flush_member", _fail_flush)
    with pytest.raises(zpub.PublicationError, match="flush failed"):
        pub2.commit(required_roles=_required_roles())
    assert zpub.resolve_current(out, RUN_ID).generation == gen1.generation
    assert not (
        pub2.generation_dir / zpub.PUBLICATION_FILENAME
    ).exists()


# ---------------------------------------------------------------------------
# 12. REWORK-1 — partial same-generation state is STALE (F5)
# ---------------------------------------------------------------------------


def test_m5b_partial_state_rows_are_stale_not_verified(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    run_root = tmp_path / "store"
    run_root.mkdir()
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=False,
    )
    gen, *_ = _publish(out)
    full = gen.outputs_state()
    partial = {
        "generation": gen.generation,
        "science": full["science"],
    }
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=True,
        outputs=partial,
    )
    verdict = zpub.verify_finalized(out, run_root, RUN_ID)
    assert verdict.reason == "publication_durable_state_stale"
    zres.write_state(
        run_root,
        run_id=RUN_ID,
        expected_cell_ids=["r0000c0000"],
        policy=POLICY,
        finalized=True,
        outputs=full,
    )
    assert zpub.verify_finalized(out, run_root, RUN_ID).reason == (
        "verified_finalized"
    )
