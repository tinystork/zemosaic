"""M5-B REWORK-1 — EXECUTION proofs on the real product path.

These tests do not compose helpers to imitate the pipeline:

* they invoke the REAL ``_run_single`` entry (capture guard at the top, then a
  full store-backed production run);
* a NEW process reopens the store with ``resume_mode=auto`` and every expensive
  seam (layout search, gauge, assembly, finishing, raw/output writes) replaced
  by ``os._exit(77)`` — success proves the finalized no-op reran none of them;
* the publication RAM probe is failed on purpose and must refuse before raw
  science, finishing or a generation exists;
* real finishing (DBE + RGB equalize) runs on the published raw science, with a
  direct off/on parity witness.

Windows/GPU/M3/M31 are NOT_RUN on this host.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid.executor import ExecutorConfig  # noqa: E402

HARNESS = REPO_ROOT / "tools" / "zegrid_resume_m5b" / "run_probe.py"
MARKER = "M5B_JSON:"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_tiny_corpus(root: Path) -> Path:
    """A REAL, deterministic, small corpus (320x320, 6 dithered frames).

    The probe pins the layout to 1x1 (the documented fixture) so the scientific
    patch-area/halo floors are satisfied on a small corpus.

    Small enough that the pinned 2x2 geometry is jointly feasible under the
    real PLAN budget, large enough for a real DBE/equalize finishing pass.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    base = np.linspace(100.0, 200.0, 320 * 320, dtype=np.float32).reshape(320, 320)
    offsets = [(-12, -12), (0, -12), (12, -12), (-12, 12), (0, 12), (12, 12)]
    for index, (dx, dy) in enumerate(offsets):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + dx * 0.001, 30.0 + dy * 0.001]
        wcs.wcs.crpix = [160.0, 160.0]
        wcs.wcs.cd = np.array([[-0.001, 0.0], [0.0, 0.001]])
        data = (
            base
            + rng.normal(0.0, 0.8, (320, 320)).astype(np.float32)
            + np.float32(index)
        )
        fits.PrimaryHDU(data, header=wcs.to_header()).writeto(
            root / f"frame_{index:04d}.fits", overwrite=True
        )
    return root


def _probe(input_dir, out, store, mode, *, tamper="none", fail_repair=False):
    args = [
        sys.executable,
        str(HARNESS),
        "--input", str(input_dir),
        "--out", str(out),
        "--store", str(store),
        "--mode", mode,
    ]
    if tamper != "none":
        args += ["--tamper-state", tamper]
    if fail_repair:
        args += ["--fail-repair"]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            payload = json.loads(line[len(MARKER):])
    return proc, payload


def _digest_tree(root: Path, *, exclude=()) -> dict:
    import hashlib

    excluded = set(exclude)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def _science_config():
    return replace(ExecutorConfig().science_config(), cell_gauge_mode="per_cell")


# ---------------------------------------------------------------------------
# (a) Capture guard executes at the TOP of the real _run_single
# ---------------------------------------------------------------------------


def test_exec_capture_guard_store_r37_r40_before_any_work(tmp_path, monkeypatch):
    calls = {"descriptors": 0}

    def _descriptors(*args, **kwargs):
        calls["descriptors"] += 1
        raise AssertionError("expensive work reached past the capture guard")

    monkeypatch.setattr(zz, "_build_frame_descriptors", _descriptors)
    store = tmp_path / "store"
    science_config = _science_config()

    for flags in ({"r37_diagnostic_capture": True}, {"r40_seam_capture": True}):
        out = tmp_path / ("out-" + next(iter(flags)))
        with pytest.raises(RuntimeError, match="diagnostic capture"):
            zz._run_single(
                [],
                str(tmp_path),
                str(out),
                progress_callback=None,
                science_config=science_config,
                zconfig=SimpleNamespace(zegrid_durable_store_dir=str(store)),
                **flags,
            )
        assert calls["descriptors"] == 0
        assert not zpub.publication_root(out).exists()

    # Store-less captures are unchanged: the guard passes and the next stage is
    # reached (the sentinel proves it).
    out = tmp_path / "out-nostore"
    with pytest.raises(AssertionError, match="expensive work reached"):
        zz._run_single(
            [],
            str(tmp_path),
            str(out),
            progress_callback=None,
            science_config=science_config,
            zconfig=SimpleNamespace(),
            r37_diagnostic_capture=True,
        )
    assert calls["descriptors"] == 1


def test_exec_capture_guard_precedes_an_already_published_noop(tmp_path, monkeypatch):
    # A finalized-looking store/publication must NOT be reached before the
    # guard: a capture cannot bypass refusal through the no-op early return.
    out = tmp_path / "out"
    run_root = (
        zpub.publication_root(out) / "run-run-guarded"
    )
    run_root.mkdir(parents=True)
    pointer = run_root / zpub.POINTER_FILENAME
    pointer.write_text('{"sentinel": "untouched"}\n', encoding="utf-8")
    before = pointer.read_bytes()
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(RuntimeError, match="diagnostic capture"):
        zz._run_single(
            [],
            str(tmp_path),
            str(out),
            progress_callback=None,
            science_config=_science_config(),
            zconfig=SimpleNamespace(zegrid_durable_store_dir=str(store)),
            resume_mode="auto",
            r37_diagnostic_capture=True,
        )
    assert pointer.read_bytes() == before


# ---------------------------------------------------------------------------
# (b) Real production store path + NEW-process finalized no-op
# ---------------------------------------------------------------------------


def test_exec_store_finishing_publication_and_new_process_noop(tmp_path):
    input_dir = _write_tiny_corpus(tmp_path / "input")
    out = tmp_path / "out"
    store = tmp_path / "store"
    proc, new = _probe(input_dir, out, store, "new")
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert new is not None, proc.stdout[-2000:]
    assert new["refused"] is False
    assert new["synthetic_upstream_ram"] is True
    assert any(
        "plan/layout" in item for item in new["synthetic_effects"]
    )
    assert new["state_ack_complete"] is True
    assert new["return_exists"] is True
    assert new["return_inside_generation"] is True
    assert new["return_role"] == zpub.ROLE_AESTHETIC
    assert new["generation"] == "gen-1"
    assert zpub.ROLE_AESTHETIC in new["members"]
    finishing = new["manifest_finishing"]
    assert finishing["enabled"] is True
    assert finishing["failed"] is False
    assert finishing["dbe"]["applied"] is True
    assert finishing["rgb_equalize"]["applied"] is True
    # Real finishing on the published raw science: OFF is the identity, ON
    # differs and preserves the finite/NaN mask; the delivered aesthetic is the
    # real finishing output.
    assert new["finishing_off_identity"] is True
    assert new["finishing_on_differs"] is True
    assert new["finishing_on_mask_preserved"] is True
    assert new["aesthetic_matches_real_finishing"] is True
    assert new["manifest_outputs"]["aesthetic"] == "mosaic_grid_aesthetic.fits"

    # The live run log is a non-authoritative diagnostic appended by every
    # _run_single start (including the no-op); the GENERATION is the authority.
    out_before = _digest_tree(out, exclude={"zegrid_run.log"})
    # LOCK/OWNER are the process lock metadata (updated by any open/close);
    # every checkpoint/gauge/geometry/publication byte must be unchanged.
    store_before = _digest_tree(store, exclude={"LOCK", "OWNER.json"})
    noop_proc, noop = _probe(input_dir, out, store, "noop")
    assert noop_proc.returncode == 0, (
        noop_proc.returncode,
        noop_proc.stdout[-2000:],
        noop_proc.stderr[-2000:],
    )
    assert noop is not None, noop_proc.stdout[-2000:]
    assert noop["generation"] == new["generation"]
    assert noop["return"] == new["return"]
    assert noop["return_role"] == zpub.ROLE_AESTHETIC
    assert noop["publication_sha256"] == new["publication_sha256"]
    assert noop["finalization_reason"] == "verified_finalized"
    assert noop["state_ack_complete"] is True
    # No output, commit, gauge or geometry byte changed; no expensive seam ran
    # (the harness would have exited 77).
    assert _digest_tree(out, exclude={"zegrid_run.log"}) == out_before
    assert _digest_tree(store, exclude={"LOCK", "OWNER.json"}) == store_before


# ---------------------------------------------------------------------------
# (b2) F5 — complete ack repair, idempotence, non-fatal repair failure
# ---------------------------------------------------------------------------


def test_exec_ack_complete_repair_then_stable_noop(tmp_path):
    input_dir = _write_tiny_corpus(tmp_path / "input")
    out = tmp_path / "out"
    store = tmp_path / "store"
    proc, new = _probe(input_dir, out, store, "new")
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert new is not None and new["state_ack_complete"] is True

    run_root = zpub.run_publication_root(out, new["run_id"])
    pointer = run_root / zpub.POINTER_FILENAME
    publication = (
        run_root / new["generation"] / zpub.PUBLICATION_FILENAME
    )
    pointer_before = pointer.read_bytes()
    publication_before = publication.read_bytes()
    out_before = _digest_tree(out, exclude={"zegrid_run.log"})
    store_authority_before = _digest_tree(
        store, exclude={"LOCK", "OWNER.json", "state.json"}
    )

    for tamper in ("partial", "wrong", "lost"):
        run, report = _probe(input_dir, out, store, "noop", tamper=tamper)
        assert run.returncode == 0, (
            tamper,
            run.stdout[-2000:],
            run.stderr[-2000:],
        )
        assert report is not None, (
            tamper,
            run.stdout[-2000:],
        )
        assert report["finalization_reason"] == "verified_finalized", (
            tamper,
            report["finalization_reason"],
            report["state_ack_problems"],
        )
        assert report["state_ack_complete"] is True
        assert report["state_finalized"] is True
        assert report["return"] == new["return"]
        assert report["generation"] == new["generation"]
        assert pointer.read_bytes() == pointer_before
        assert publication.read_bytes() == publication_before
        assert _digest_tree(out, exclude={"zegrid_run.log"}) == out_before
        assert _digest_tree(
            store, exclude={"LOCK", "OWNER.json", "state.json"}
        ) == store_authority_before

    # A correct ack is NOT rewritten on every valid no-op.
    state_path = next(Path(store).glob("**/state.json"))
    state_before = state_path.read_bytes()
    run, report = _probe(input_dir, out, store, "noop")
    assert run.returncode == 0
    assert report["finalization_reason"] == "verified_finalized"
    assert report["state_ack_complete"] is True
    assert state_path.read_bytes() == state_before

    # Repair write failure is non-fatal: valid return, durable authority intact,
    # state remains stale (and is repaired by the NEXT no-op).
    run, report = _probe(
        input_dir, out, store, "noop", tamper="partial", fail_repair=True
    )
    assert run.returncode == 0, (run.stdout[-2000:], run.stderr[-2000:])
    assert report is not None, run.stdout[-2000:]
    assert report["state_ack_complete"] is False
    assert report["return"] == new["return"]
    assert report["generation"] == new["generation"]
    assert "could not be repaired" in (run.stdout + run.stderr)
    assert pointer.read_bytes() == pointer_before
    assert publication.read_bytes() == publication_before
    assert _digest_tree(out, exclude={"zegrid_run.log"}) == out_before

    run, report = _probe(input_dir, out, store, "noop")
    assert run.returncode == 0
    assert report["state_ack_complete"] is True
    assert report["finalization_reason"] == "verified_finalized"
    assert pointer.read_bytes() == pointer_before
    assert publication.read_bytes() == publication_before


# ---------------------------------------------------------------------------
# (c) Publication RAM probe failure refuses BEFORE raw/finishing/generation
# ---------------------------------------------------------------------------


def test_exec_publication_ram_probe_failure_refuses_before_effects(
    tmp_path,
):
    input_dir = _write_tiny_corpus(tmp_path / "input")
    out = tmp_path / "out"
    store = tmp_path / "store"
    proc, payload = _probe(input_dir, out, store, "ram-fail")
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert payload is not None, proc.stdout[-2000:]
    assert payload["refused"] is True
    assert "PublicationCapacityError" in payload["error"]
    plan = payload["plan"]
    assert plan["admitted"] is False
    assert plan["ram"]["observed"] is False
    assert plan["ram"]["probe_error"]
    # BEFORE raw science, finishing output or a generation directory.
    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "mosaic_grid_science.fits").exists()
    assert not (out / "mosaic_grid_aesthetic.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()
    assert not zpub.publication_root(out).exists()
    # The store itself remains (resumable), with no claimed publication.
    assert store.exists()
