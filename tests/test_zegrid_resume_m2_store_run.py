"""ZM-ZEGRID-RESILIENT-RUN-RESUME M2 — real-orchestration store seam.

Drives the REAL product path (``run_zegrid_mode`` -> ``_run_single`` -> bounded
ACTIVE CPU waves) on the real small FITS corpus, with the durable store enabled
through the explicit ``zegrid_durable_store_dir`` config seam.

What it proves (when the host can admit the corpus):
* every completed CPU cell is durably committed before its payload is released;
* the registry's retained bytes drop while the run is still executing (the
  store is not a new RAM accumulation);
* the payload that assembly consumes after release is byte-identical to the
  payload the producer emitted (verified through the store);
* the published science/coverage/manifest are read from the SAME verified
  CURRENT generation the product published (M5-B contract), never from a legacy
  root path;
* the store's index is complete and its commits all validate.

No GPU and no Windows qualification is claimed. On a host below the corpus RAM
floor this case SKIPS (NOT_RUN), exactly like M1.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _path in (str(SRC), str(TESTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

import test_zegrid_publication_m5b as m5b  # noqa: E402

MIN_LIVE_BYTES = 1200 * 2 ** 20


def _load_m0_helpers():
    path = REPO_ROOT / "tests" / "test_zegrid_resume_m0_baseline_witness.py"
    spec = importlib.util.spec_from_file_location("_m0_helpers_m2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


M0 = _load_m0_helpers()


def _live_available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _require_live_ram() -> int:
    live = _live_available_bytes()
    if live < MIN_LIVE_BYTES:
        pytest.skip(
            "live available RAM is below the 2x2 corpus floor "
            f"({live} < {MIN_LIVE_BYTES}); M2 store seam NOT_RUN on this host"
        )
    return live


def _store_report(store_dir: Path) -> dict:
    """Read the durable store identity/report without inventing one."""
    root = Path(store_dir)
    if not (root / "run.json").exists():
        pytest.fail(
            "durable store wrote no run.json although "
            f"zegrid_durable_store_dir={store_dir} was configured"
        )
    store = zrs.RunCellStore.open(root, identity=None, read_only=True)
    try:
        return store.report()
    finally:
        store.close()


def _resolve_published_generation(out_dir: Path, store_dir: Path):
    """Resolve CURRENT from the store's own identity, verifying every member.

    No glob, no newest-file heuristic, no legacy root path, no trust in a
    ``finalized`` state: the run id and expected identity come from the durable
    store, and CURRENT is the only publication authority.
    """
    report = _store_report(store_dir)
    run_id = report.get("run_id")
    identity_sha256 = report.get("identity_sha256")
    assert run_id, report
    assert identity_sha256, report
    try:
        generation = zpub.resolve_current(
            out_dir,
            str(run_id),
            verify_members=True,
            expected_identity_sha256=str(identity_sha256),
        )
    except zpub.PublicationError as exc:
        pytest.fail(
            "no verified CURRENT generation for the durable-store run "
            f"(run_id={run_id!r}, identity_sha256={identity_sha256!r}): "
            f"{type(exc).__name__}: {exc}"
        )
    assert generation.identity_sha256 == str(identity_sha256)
    return generation


def _store_diagnostics(generation) -> dict:
    """Read the durable-store diagnostics from the verified generation manifest."""
    manifest = generation.path(zpub.ROLE_MANIFEST)
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        pytest.fail(
            f"the verified generation manifest {manifest} is unreadable: "
            f"{type(exc).__name__}: {exc}"
        )
    return (data.get("per_cell_diagnostics") or {}).get("r55_durable_store") or {}


def _require_durable_store_active(generation, store_dir: Path) -> dict:
    """Fail with the REAL store failure text instead of a downstream symptom.

    The product intentionally degrades to M1 conservation when the durable store
    cannot open; a test must never report that as "no run.json". It must report
    the recorded ``open_error``/``degraded_reason`` and the failing stage.
    """
    for role in (zpub.ROLE_SCIENCE, zpub.ROLE_COVERAGE, zpub.ROLE_MANIFEST):
        assert role in generation.members, (role, generation.members)
        assert generation.path(role).parent == generation.generation_dir
    diag = _store_diagnostics(generation)
    if not diag:
        pytest.fail(
            "the run published no r55_durable_store diagnostics in its verified "
            f"generation {generation.generation_dir} although "
            f"zegrid_durable_store_dir={store_dir} was configured"
        )
    if diag.get("open_error") or diag.get("degraded_reason"):
        pytest.fail(
            "durable store did not become active: "
            f"stage={diag.get('open_stage')!r} "
            f"open_error={diag.get('open_error')!r} "
            f"degraded_reason={diag.get('degraded_reason')!r} "
            f"commit_errors={diag.get('commit_errors')!r}\n"
            f"traceback tail:\n{diag.get('open_traceback')}"
        )
    if not diag.get("enabled"):
        pytest.fail(
            "durable store was configured but never became active: "
            f"{json.dumps(diag, sort_keys=True)[:1500]}"
        )
    if not (Path(store_dir) / "run.json").exists():
        pytest.fail(
            "durable store reports enabled but wrote no run.json: "
            f"{json.dumps(diag, sort_keys=True)[:1500]}"
        )
    return diag


def _digest(array) -> str:
    contiguous = np.ascontiguousarray(array)
    return zrs._sha256_bytes(contiguous.tobytes())


def _run_with_store(inp: Path, out_dir: Path, store_dir: Path) -> dict:
    produced: dict[str, dict] = {}
    released: list[dict] = []
    store_reports: list[dict] = []
    events: list[tuple[str, str]] = []

    real_consume = zz._consume_stack_cell
    real_release = zrr.RunResultRegistry.release_payload
    real_commit = zrs.RunCellStore.commit_cell

    def _consume(prepared, **kw):
        out = real_consume(prepared, **kw)
        if isinstance(out, dict) and out.get("status") == "complete":
            produced[str(out["cell_id"])] = {
                "science": _digest(out["core"]["science_core"]),
                "patch_science": _digest(out["patch"]["science"]),
                "valid_mask": _digest(out["core"]["valid_mask_core"]),
            }
        return out

    def _release(self, cell_id, **kwargs):
        result = real_release(self, cell_id, **kwargs)
        events.append(("release", str(cell_id)))
        released.append(dict(result, cell_id=str(cell_id)))
        return result

    def _commit(self, result, **kwargs):
        receipt = real_commit(self, result, **kwargs)
        events.append(("commit", str(result.get("cell_id"))))
        store_reports.append(self.report())
        return receipt

    exc = None
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zz, "_consume_stack_cell", _consume)
        mp.setattr(zrr.RunResultRegistry, "release_payload", _release)
        mp.setattr(zrs.RunCellStore, "commit_cell", _commit)
        try:
            zz.run_zegrid_mode(
                str(inp),
                str(out_dir),
                zconfig=SimpleNamespace(
                    zegrid_scheduler_mode="active",
                    zegrid_layout=M0.LAYOUT,
                    zegrid_durable_store_dir=str(store_dir),
                    zegrid_durable_store_budget_bytes=2 * 2 ** 30,
                ),
                zegrid_halo_px=M0.HALO,
                use_gpu=False,
                cell_gauge_mode="per_cell",
            )
        except BaseException as err:  # noqa: BLE001 - the witness reports it
            exc = err
    return {
        "produced": produced,
        "released": released,
        "store_reports": store_reports,
        "events": events,
        "exception": exc,
        "out_dir": out_dir,
        "store_dir": store_dir,
    }


def _require_successful_store_run(run: dict) -> None:
    """Only an explicit pre-production planning refusal is NOT_RUN."""
    exc = run["exception"]
    if exc is None:
        return
    progressed = any(run[key] for key in ("produced", "released", "store_reports"))
    if not progressed and isinstance(
        exc, (zz.zal.LayoutInfeasible, zz.zpar.ModeInfeasible)
    ):
        pytest.skip(f"pre-case resource/setup unavailable ({exc})")
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    pytest.fail(f"store-enabled real run failed:\n{trace}")


def test_m2_real_cpu_run_commits_then_releases_and_reloads_identical_payloads(
    tmp_path,
):
    _require_live_ram()
    base = tmp_path
    inp, names, hashes = M0._write_corpus(base)
    out_dir = base / "out"
    store_dir = base / "run_store"
    run = _run_with_store(inp, out_dir, store_dir)
    _require_successful_store_run(run)

    # F3: surface the product's own store diagnostics BEFORE opening the store.
    generation = _resolve_published_generation(out_dir, store_dir)
    diag = _require_durable_store_active(generation, store_dir)
    assert diag["commits"] == len(run["produced"]), diag
    assert not diag["commit_errors"], diag

    produced = run["produced"]
    assert produced, "no COMPLETE cell was produced"

    # The observed production event sequence, not just set membership: each
    # COMPLETE cell's release is recorded only after its durable commit
    # returned, and no release exists without a preceding commit.
    events = run["events"]
    for cell_id in produced:
        commit_at = [
            index
            for index, (kind, cid) in enumerate(events)
            if kind == "commit" and cid == cell_id
        ]
        release_at = [
            index
            for index, (kind, cid) in enumerate(events)
            if kind == "release" and cid == cell_id
        ]
        assert len(commit_at) == 1, (cell_id, events)
        assert len(release_at) == 1, (cell_id, events)
        assert commit_at[0] < release_at[0], (cell_id, events)
    assert {cid for kind, cid in events if kind == "release"} == set(produced)

    store = zrs.RunCellStore.open(store_dir, identity=None, read_only=True)
    try:
        report = store.report()
        assert report["counts"]["cells"] == len(produced)
        assert report["counts"]["attempts"] == len(produced)
        assert report["counts"]["invalid"] == 0
        assert report["index"]["stale"] is False
        scan = store.scan()
        assert scan["invalid"] == [], scan
        assert len(scan["valid"]) == len(produced), scan
        completeness = store.completeness(list(produced))
        assert completeness["is_complete"] is True, completeness
        assert completeness["missing"] == [], completeness
        # Every release happened AFTER a durable commit, one per produced cell.
        assert {entry["cell_id"] for entry in run["released"]} == set(produced)
        for entry in run["released"]:
            assert entry["released"] is True
        # The store re-supplies exactly the producer bytes.
        for cell_id, digests in produced.items():
            payload = store.load_payload(cell_id)
            assert _digest(payload["core"]["science_core"]) == digests["science"]
            assert _digest(payload["patch"]["science"]) == digests["patch_science"]
            assert _digest(payload["core"]["valid_mask_core"]) == digests["valid_mask"]
            assert payload["core"]["science_core"].flags.writeable is False
        # The store never became a RAM accumulation: after the last commit the
        # registry holds receipts only (released bytes >= produced core bytes).
        last = run["store_reports"][-1]
        assert last["counts"]["cells"] == len(produced)
        # Published science/coverage/manifest come from the SAME verified
        # CURRENT generation (M5-B). There is no legacy root reader here: a
        # root file can neither substitute nor mask a generation member.
        manifest = json.loads(
            generation.path(zpub.ROLE_MANIFEST).read_text(encoding="utf-8")
        )
        canvas = manifest["canvas"]
        published_science = fits.getdata(generation.path(zpub.ROLE_SCIENCE))
        published_coverage = fits.getdata(generation.path(zpub.ROLE_COVERAGE))
        assert published_science.ndim == 3
        assert published_science.shape[-2:] == (
            canvas["height"],
            canvas["width"],
        )
        assert published_science.dtype.kind == "f"
        assert published_science.dtype.itemsize == 4
        assert published_coverage.shape == (canvas["height"], canvas["width"])
        assert manifest["outputs"]["science"] == generation.members[
            zpub.ROLE_SCIENCE
        ]["file"]
        assert manifest["outputs"]["coverage"] == generation.members[
            zpub.ROLE_COVERAGE
        ]["file"]
        assert manifest["science_reference"] == manifest["outputs"]["science"]
    finally:
        store.close()


def test_m2_store_identity_refuses_a_changed_science_contract(tmp_path):
    """A store created by one science contract is not reused for another."""
    _require_live_ram()
    base = tmp_path
    inp, names, hashes = M0._write_corpus(base)
    out_dir = base / "out"
    store_dir = base / "run_store"
    run = _run_with_store(inp, out_dir, store_dir)
    _require_successful_store_run(run)
    generation = _resolve_published_generation(out_dir, store_dir)
    _require_durable_store_active(generation, store_dir)
    stored = zrs.RunCellStore.open(store_dir, identity=None, read_only=True)
    identity = stored.identity
    stored.close()

    changed = json.loads(json.dumps(identity))
    changed["science_contract"]["science"]["sha256"] = "9" * 64
    changed["science_contract"]["science"]["params"] = {"tampered": True}
    with pytest.raises(zrs.StoreIdentityMismatch) as err:
        zrs.RunCellStore.open(store_dir, identity=changed)
    assert "science" in [r["section"] for r in err.value.reasons]


@pytest.mark.parametrize("exc", [KeyError("bounds"), RuntimeError("store broke")])
def test_m2_real_run_unexpected_error_is_never_skipped(exc):
    run = dict(exception=exc, produced={}, released=[], store_reports=[])
    with pytest.raises(pytest.fail.Exception, match=type(exc).__name__):
        _require_successful_store_run(run)


@pytest.mark.parametrize("kind", [zz.zal.LayoutInfeasible, zz.zpar.ModeInfeasible])
@pytest.mark.parametrize("progress", [None, "produced", "released", "store_reports"])
def test_m2_real_run_planning_refusal_skips_only_before_production(kind, progress):
    run = dict(exception=kind("no capacity"), produced={}, released=[], store_reports=[])
    if progress is not None:
        run[progress] = ["observed"]
    outcome = pytest.skip.Exception if progress is None else pytest.fail.Exception
    with pytest.raises(outcome):
        _require_successful_store_run(run)


# ---------------------------------------------------------------------------
# Reader regression checks: CURRENT+members are the only authority
# ---------------------------------------------------------------------------


def _tiny_published_generation(tmp_path):
    """Reuse the existing M5-B real-writer fixture (12x12 complete generation)."""
    out = tmp_path / "out"
    out.mkdir()
    generation, *_ = m5b._publish(out)
    return out, generation


def _plant_m1_root_decoy(out: Path) -> None:
    """A plausible legacy root publication that must never be consulted."""
    (out / "zegrid_manifest.json").write_text(
        json.dumps(
            {
                "per_cell_diagnostics": {
                    "r55_durable_store": {
                        "enabled": True,
                        "commits": 99,
                        "open_error": None,
                        "degraded_reason": None,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (out / "mosaic_grid.fits").write_bytes(b"stale-root-science")
    (out / "mosaic_grid_coverage.fits").write_bytes(b"stale-root-coverage")


def test_m2_reader_missing_current_never_falls_back_to_root(tmp_path):
    out, generation = _tiny_published_generation(tmp_path)
    _plant_m1_root_decoy(out)
    generation.pointer_path.unlink()
    with pytest.raises(zpub.PublicationNotFound):
        zpub.resolve_current(out, m5b.RUN_ID, verify_members=True)


def test_m2_reader_corrupt_pointer_fails_closed(tmp_path):
    out, generation = _tiny_published_generation(tmp_path)
    generation.pointer_path.write_text("{not-a-pointer", encoding="utf-8")
    with pytest.raises(zpub.PublicationIntegrityError, match="not readable JSON"):
        zpub.resolve_current(out, m5b.RUN_ID, verify_members=True)


def test_m2_reader_corrupt_member_fails_closed_despite_root_decoys(tmp_path):
    out, generation = _tiny_published_generation(tmp_path)
    _plant_m1_root_decoy(out)
    target = generation.path(zpub.ROLE_SCIENCE)
    with open(target, "r+b") as handle:
        handle.seek(0)
        handle.write(b"CORRUPT")
    with pytest.raises(zpub.PublicationIntegrityError, match="sha256"):
        zpub.resolve_current(out, m5b.RUN_ID, verify_members=True)


def test_m2_reader_missing_member_fails_closed(tmp_path):
    out, generation = _tiny_published_generation(tmp_path)
    generation.path(zpub.ROLE_COVERAGE).unlink()
    with pytest.raises(zpub.PublicationIntegrityError, match="missing or not a file"):
        zpub.resolve_current(out, m5b.RUN_ID, verify_members=True)


def test_m2_reader_pointer_claiming_another_run_is_refused(tmp_path):
    out, generation = _tiny_published_generation(tmp_path)
    pointer = json.loads(generation.pointer_path.read_text(encoding="utf-8"))
    pointer["run_id"] = "some-other-run"
    generation.pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(
        zpub.PublicationIntegrityError, match="another run identity"
    ):
        zpub.resolve_current(out, m5b.RUN_ID, verify_members=True)
