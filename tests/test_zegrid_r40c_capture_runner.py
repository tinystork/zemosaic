"""ZM-ZEGRID-R40-C focused tests — bounded R40 DEV diagnostic-capture runner CLI.

Covers the mandated R40-C runner behaviours WITHOUT any real corpus read:

* dry-run writes NOTHING (not the input dir, not the output dir) and prints the
  EXACT ``run_zegrid_mode`` kwargs it would use;
* output-safety refusals (non-empty, input/baseline overlap, symlink, traversal)
  mirroring the accepted ``tools/r37_capture_runner.py`` refusal semantics;
* the planned kwargs are asserted against ``inspect.signature(run_zegrid_mode)``;
* the frozen B1 v2 bindings are asserted against the real core;
* the published-generation discovery + independent verification refusal paths;
* the strict-JSON provenance report shape (basenames only, no path leak);
* the NO-FINAL-FITS guarantee (the runner never imports/opens FITS);
* ONE real end-to-end execute on a small synthetic on-disk corpus (real engine,
  real process pool), which is explicitly NOT a Windows run and claims NO
  qualification.

Explicit NON-CLAIMS: this suite does not run the capture on the real corpus, does
not read any real FITS/pixel, claims no Windows result, changes no tolerance,
threshold or gate, and leaves the capture default-OFF everywhere except the
explicit runner seam.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

# The synthetic end-to-end run writes FITS via the production path, which emits the
# PRE-EXISTING astropy ``Card is too long`` VerifyWarning on the synthetic WCS
# header (identical to the accepted R37-D / R40-B2 e2e tests). Filter ONLY that
# warning so ``-W error`` still guards everything R40-C cares about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)

# The tool is a standalone script under tools/; load it by path (registered in
# ``sys.modules`` first so dataclass/annotation resolution works).
_TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "r40_capture_runner.py"


def _load_runner():
    name = "r40_capture_runner"
    spec = importlib.util.spec_from_file_location(name, _TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()

from zemosaic.core.zegrid import r40_capture as c  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

_REPO_HEAD_40 = "0" * 39 + "1"


# ---------------------------------------------------------------------------
# Synthetic on-disk corpus (real engine run; NO real data)
# ---------------------------------------------------------------------------

_DISJOINT_CENTERS = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]
_PIN_MEM_BYTES = 4 * 1024 ** 3


def _write_frame(path, ra, dec, shape=(400, 600), scale=0.001, seed=0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * scale
    w.array_shape = shape
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
    fits.PrimaryHDU(data, header=w.to_header()).writeto(path, overwrite=True)


def _write_corpus(root):
    root.mkdir(parents=True, exist_ok=True)
    for i, (ra, dec) in enumerate(_DISJOINT_CENTERS):
        _write_frame(root / f"f{i}.fit", ra, dec, seed=i)
    return root


# ---------------------------------------------------------------------------
# Synthetic generation fixtures (core-level, in-memory shards)
# ---------------------------------------------------------------------------

def _make_header(u="c0001", v="c0002", orientation="h", **overrides):
    header = {
        "gauge_mode": "canonical_sky_mean",
        "stack_tokens": {
            "normalization": "none", "weighting": "none", "rejection": "none",
            "combine": "mean", "taper": "none", "taper_px": 0, "taper_floor": 0.0,
            "equalize_rgb": False, "sigma_low": -1.0, "sigma_high": -1.0, "winsor": 0.0,
        },
        "reference_frame_id": {u: "r0001c0001", v: "r0001c0002"},
        "frame_order_digest": {u: "0" * 64, v: "1" * 64},
        "excluded": {u: [], v: []},
        "measurement_px": 16, "feather_px": 8, "halo_px": 8, "block_px": 4,
    }
    header.update(overrides)
    return header


def _make_record(channel=0, gx=10, gy=20, **overrides):
    record = {
        "gx": gx, "gy": gy, "channel": channel,
        "sci_u_pre": 1.0, "sci_v_pre": 1.5, "sci_u_post": 1.1, "sci_v_post": 1.6,
        "valid_u": True, "valid_v": True,
        "estimator_weight_sum_u": 2.0, "estimator_weight_sum_v": 2.5,
        "surviving_sample_count_u": 10, "surviving_sample_count_v": 12,
        "support_w1_u": 0.5, "support_w1_v": 0.6,
        "support_w2_u": 0.7, "support_w2_v": 0.8,
        "n_eff_support_u": 3.0, "n_eff_support_v": 3.5,
        "ramp_u": 0.25, "ramp_v": 0.75,
    }
    record.update(overrides)
    return record


def _make_shard(u="c0001", v="c0002", orientation="h", records=()):
    records = records or (_make_record(0, 5, 6),)
    return c.RelationShard(
        u=u, v=v, orientation=orientation,
        header=_make_header(u, v, orientation), records=tuple(records),
    )


def _make_identity(*, production_wired):
    return c.make_capture_identity(
        capture_id="capture-abc",
        source_set_digest="a" * 64,
        n_frames=816,
        layout={"nx": 6, "ny": 4},
        canvas={"width": 4117, "height": 4377, "id": "e8f1877f16f8cca4"},
        gauge_mode="per_cell",
        repo_head=_REPO_HEAD_40,
        source_set_count=816,
        production_wired=production_wired,
    )


def _write_generation(root, *, production_wired=True):
    cfg = c.R40CaptureConfig(enabled=True, output_root=pathlib.Path(root))
    report = c.write_capture_generation(
        root, shards=[_make_shard()],
        identity=_make_identity(production_wired=production_wired), config=cfg,
        expected_production_wired=production_wired,
    )
    return pathlib.Path(root) / report["generation_dir"]


def _fake_invocation(output_dir, capture_root, *, relativized=False):
    return {
        "input_dir": "/private/data/lights",
        "output_dir": str(output_dir),
        "capture_root": str(capture_root),
        "capture_root_relativized": relativized,
        "kwargs": {
            "cell_gauge_mode": "per_cell",
            "r40_seam_capture": True,
            "r40_output_root": str(capture_root),
            "r40_seam_capture_disk_budget_bytes": c.DEFAULT_DISK_BUDGET_BYTES,
        },
    }


# ---------------------------------------------------------------------------
# A. Dry-run: writes nothing, prints the exact plan
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing_and_prints_exact_kwargs(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")          # names only; never opened
    (inp / "f1.fits").write_bytes(b"")
    out = tmp_path / "out"

    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(out)])
    assert rc == runner._EXIT_OK

    # nothing was written: not the output dir, not a generation anywhere
    assert not out.exists()
    assert not list(tmp_path.rglob("r40_capture_*"))

    text = capsys.readouterr().out
    marker = "--- EXACT PLAN (run_zegrid_mode kwargs) ---"
    assert marker in text
    body = text.split(marker, 1)[1].split("\n[r40-runner]", 1)[0]
    plan = json.loads(body)
    assert plan["run_zegrid_mode_kwargs"] == {
        "cell_gauge_mode": "per_cell",
        "r40_seam_capture": True,
        "r40_output_root": str(out),
        "r40_seam_capture_disk_budget_bytes": c.DEFAULT_DISK_BUDGET_BYTES,
    }
    assert plan["execute"] is False
    assert plan["n_source_fits_in_inventory"] == 2
    assert plan["frozen_bindings"]["b1_claim_id"] == runner.B1_CLAIM_ID
    assert "DRY-RUN complete" in text


def test_dry_run_requires_existing_input_dir(tmp_path, capsys):
    rc = runner.main(["--input-dir", str(tmp_path / "nope"),
                      "--output-dir", str(tmp_path / "out")])
    assert rc == runner._EXIT_ERROR
    assert "INPUT ERROR" in capsys.readouterr().err


def test_dry_run_requires_a_fits_inventory(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "readme.txt").write_text("no frames here")
    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(tmp_path / "out")])
    assert rc == runner._EXIT_ERROR
    assert "no FITS frames" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# B. Refusals: gauge mode / holdout cap / disk budget
# ---------------------------------------------------------------------------

def test_cell_gauge_mode_must_be_per_cell(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(tmp_path / "out"),
                      "--cell-gauge-mode", "global"])
    assert rc == runner._EXIT_ERROR
    assert "per_cell" in capsys.readouterr().err


def test_holdout_cap_must_stay_uncapped(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    out = tmp_path / "out"
    # the uncapped sentinel is accepted
    assert runner.main(["--input-dir", str(inp), "--output-dir", str(out),
                        "--holdout-cap", "0"]) == runner._EXIT_OK
    assert not out.exists()
    # any positive cap is REFUSED (never a capped capture)
    assert runner.main(["--input-dir", str(inp), "--output-dir", str(out),
                        "--holdout-cap", "1000000"]) == runner._EXIT_ERROR
    assert "--holdout-cap" in capsys.readouterr().err


def test_disk_budget_must_be_a_positive_int(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    for bad in ("0", "-1"):
        assert runner.main(["--input-dir", str(inp), "--output-dir", str(tmp_path / "out"),
                            "--disk-budget-bytes", bad]) == runner._EXIT_ERROR
    assert "--disk-budget-bytes" in capsys.readouterr().err

    plan_kwargs = runner.resolve_invocation(
        input_dir=str(inp), output_dir=str(tmp_path / "out"),
        cell_gauge_mode="per_cell", holdout_cap=0, disk_budget_bytes=1234,
        core=c, cwd=tmp_path,
    )["kwargs"]
    assert plan_kwargs["r40_seam_capture_disk_budget_bytes"] == 1234


# ---------------------------------------------------------------------------
# C. Output-destination safety
# ---------------------------------------------------------------------------

def test_output_must_be_new_and_empty(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    (out / "leftover.txt").write_text("not empty")
    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(out)])
    assert rc == runner._EXIT_UNSAFE_OUTPUT
    assert "not empty" in capsys.readouterr().err


def test_output_must_not_overlap_input(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    # output INSIDE the input
    assert runner.main(["--input-dir", str(inp), "--output-dir", str(inp / "out")]) \
        == runner._EXIT_UNSAFE_OUTPUT
    # input INSIDE the output
    outer = tmp_path / "outer"
    inner = outer / "in"
    inner.mkdir(parents=True)
    (inner / "f0.fit").write_bytes(b"")
    assert runner.main(["--input-dir", str(inner), "--output-dir", str(outer)]) \
        == runner._EXIT_UNSAFE_OUTPUT
    assert "overlaps" in capsys.readouterr().err
    assert not (inp / "out").exists()


def test_output_must_not_overlap_a_baseline_root(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(baseline / "out"),
                      "--baseline-dir", str(baseline)])
    assert rc == runner._EXIT_UNSAFE_OUTPUT
    assert "protected root" in capsys.readouterr().err


def test_symlinked_output_is_refused(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform without symlinks
        pytest.skip("symlinks are not available on this platform")
    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(link)])
    assert rc == runner._EXIT_UNSAFE_OUTPUT
    assert "symlink" in capsys.readouterr().err


def test_parent_traversal_output_is_refused(tmp_path, capsys):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    rc = runner.main(["--input-dir", str(inp),
                      "--output-dir", os.path.join("..", "out")])
    assert rc == runner._EXIT_UNSAFE_OUTPUT
    assert "traversal" in capsys.readouterr().err


def test_capture_root_relativization_and_refusal(tmp_path, monkeypatch):
    """The frozen core refuses drive-absolute/UNC/traversal capture roots.

    A relative spelling of the SAME physical directory is used when available; when
    only a traversal spelling exists, the runner refuses with an actionable message.
    """

    class _RefuseAbsolute:
        """Stands in for the frozen core: absolute/traversal roots are refused."""

        @staticmethod
        def resolve_fresh_output_root(raw):
            from pathlib import Path
            s = str(raw)
            if os.path.isabs(s) or runner._has_traversal(s):
                raise runner.OutputSafetyError("refused by the frozen core")
            return Path(s)

    fake = _RefuseAbsolute()
    # an absolute output -> the relative spelling is used (same physical directory)
    root_arg, relativized = runner.resolve_capture_root(
        tmp_path / "OUT" / "R40C", core=fake, cwd=tmp_path
    )
    assert relativized is True
    assert root_arg.replace("\\", "/") == "OUT/R40C"

    # a root the core accepts as given is passed through unchanged
    root_arg, relativized = runner.resolve_capture_root(
        "OUT/R40C", core=fake, cwd=tmp_path
    )
    assert (root_arg, relativized) == ("OUT/R40C", False)

    # nothing safe exists (only a traversal spelling) -> refuse with guidance
    with pytest.raises(runner.OutputSafetyError, match="traversal"):
        runner.resolve_capture_root(tmp_path.parent / "elsewhere" / "OUT",
                                    core=fake, cwd=tmp_path)


def test_capture_root_refuses_foreign_drive_or_unc_root():
    # the frozen core (unmodified) refuses a drive-absolute/UNC root; on a
    # non-Windows host the runner must never silently reinterpret such a string.
    with pytest.raises(runner.OutputSafetyError, match="drive|UNC"):
        runner.resolve_capture_root("C:\\Users\\TRISTAN\\OUT\\R40C",
                                    core=c, cwd="/home/tristan")
    with pytest.raises(runner.OutputSafetyError, match="drive|UNC"):
        runner.resolve_capture_root("\\\\host\\share\\OUT", core=c, cwd="/home/tristan")


# ---------------------------------------------------------------------------
# D. kwargs contract + frozen bindings
# ---------------------------------------------------------------------------

def test_kwargs_mapping_asserted_against_signature():
    planned = runner.resolve_invocation(
        input_dir="/data/in", output_dir="/tmp/r40c_out_does_not_matter",
        cell_gauge_mode="per_cell", holdout_cap=0,
        disk_budget_bytes=c.DEFAULT_DISK_BUDGET_BYTES, core=c, cwd="/tmp",
    )["kwargs"]
    assert set(planned) == {
        "cell_gauge_mode", "r40_seam_capture", "r40_output_root",
        "r40_seam_capture_disk_budget_bytes",
    }
    # every planned key is a REAL keyword-only parameter of run_zegrid_mode, with
    # the default-OFF seam exactly as required
    import inspect
    sig = inspect.signature(zz.run_zegrid_mode)
    for name in planned:
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["r40_seam_capture"].default is False
    assert sig.parameters["r40_output_root"].default is None
    assert sig.parameters["r40_seam_capture_disk_budget_bytes"].default is None
    assert zz.R40_SEAM_CAPTURE_DISK_BUDGET_BYTES == c.DEFAULT_DISK_BUDGET_BYTES

    # a bogus kwarg is refused (never silently swallowed)
    with pytest.raises(runner.BindingError):
        runner.assert_kwargs_supported({"r40_seam_captur": True})


def test_frozen_bindings_asserted_against_the_core():
    frozen = runner.assert_frozen_bindings(c)
    assert frozen["b1_protocol_sha256"] == runner.B1_PROTOCOL_SHA256
    assert frozen["b1_criteria_sha256"] == runner.B1_CRITERIA_SHA256
    assert frozen["b1_result_sha256"] == runner.B1_RESULT_SHA256
    assert frozen["b1_claim_id"] == runner.B1_CLAIM_ID
    assert frozen["capture_contract_version"] == "r40-capture-1"

    class _Drifted:
        CAPTURE_CONTRACT_VERSION = "r40-capture-1"
        GENERATION_DIR_PREFIX = "r40_capture_"
        GENERATION_INDEX_BASENAME = "index.json"
        GENERATION_COMMIT_BASENAME = "COMMIT"
        SOURCE_SET_DIGEST_ALGO = runner.SOURCE_SET_DIGEST_ALGO
        DEFAULT_DISK_BUDGET_BYTES = runner.CORE_DEFAULT_DISK_BUDGET_BYTES
        B1_PROTOCOL_SHA256 = runner.B1_PROTOCOL_SHA256
        B1_CRITERIA_SHA256 = runner.B1_CRITERIA_SHA256
        B1_RESULT_SHA256 = runner.B1_RESULT_SHA256
        B1_CLAIM_ID = "deadbeef" * 5  # drifted

    with pytest.raises(runner.BindingError, match="B1_CLAIM_ID"):
        runner.assert_frozen_bindings(_Drifted)


# ---------------------------------------------------------------------------
# E. Published-generation discovery + independent verification (refusal paths)
# ---------------------------------------------------------------------------

def test_find_published_generation_requires_exactly_one_committed(tmp_path):
    root = tmp_path / "gen"
    gen = _write_generation(root)
    assert runner.find_published_generation(root) == gen

    # an uncommitted sibling does not count as published
    (root / "r40_capture_incomplete").mkdir()
    (root / "r40_capture_incomplete" / c.GENERATION_INDEX_BASENAME).write_text("{}")
    assert runner.find_published_generation(root) == gen

    # two committed generations -> refuse (ambiguous evidence)
    other = root / "r40_capture_zzz"
    other.mkdir()
    (other / c.GENERATION_COMMIT_BASENAME).write_text("{}")
    (other / c.GENERATION_INDEX_BASENAME).write_text("{}")
    with pytest.raises(runner.VerificationError, match="exactly ONE"):
        runner.find_published_generation(root)


def test_find_published_generation_requires_a_commit(tmp_path):
    root = tmp_path / "gen"
    root.mkdir()
    (root / "r40_capture_x").mkdir()
    with pytest.raises(runner.VerificationError, match="no committed"):
        runner.find_published_generation(root)
    with pytest.raises(runner.VerificationError, match="does not exist"):
        runner.find_published_generation(tmp_path / "absent")


def test_verification_passes_on_a_healthy_generation(tmp_path):
    gen = _write_generation(tmp_path / "gen")
    report = runner.verify_published_generation(gen, core=c)
    assert report["ok"] is True and report["problems"] == []
    index = runner.read_generation_index(gen)
    assert index["identity"]["capture_id"] == "capture-abc"


def test_verification_refuses_a_tampered_generation(tmp_path):
    gen = _write_generation(tmp_path / "gen")
    index = json.loads((gen / c.GENERATION_INDEX_BASENAME).read_text(encoding="utf-8"))
    shard = gen / index["shards"][0]["filename"]
    shard.write_bytes(shard.read_bytes() + b"\x00")
    with pytest.raises(runner.VerificationError, match="FAILED"):
        runner.verify_published_generation(gen, core=c)


def test_verification_refuses_a_non_production_wired_generation(tmp_path):
    # the runner verifies with expected_production_wired=True, so a default-OFF
    # (production_wired=False) generation must be REFUSED, never accepted.
    gen = _write_generation(tmp_path / "gen", production_wired=False)
    with pytest.raises(runner.VerificationError):
        runner.verify_published_generation(gen, core=c)


def test_verification_refuses_a_missing_index(tmp_path):
    gen = _write_generation(tmp_path / "gen")
    (gen / c.GENERATION_INDEX_BASENAME).unlink()
    with pytest.raises(runner.VerificationError, match="refused|FAILED"):
        runner.verify_published_generation(gen, core=c)


# ---------------------------------------------------------------------------
# F. Provenance report shape + privacy
# ---------------------------------------------------------------------------

def test_provenance_report_shape_and_no_path_leak(tmp_path):
    gen = _write_generation(tmp_path / "gen")
    index = runner.read_generation_index(gen)
    verification = runner.verify_published_generation(gen, core=c)
    invocation = _fake_invocation(tmp_path / "out", tmp_path / "gen")
    report = runner.build_report(
        invocation=invocation,
        command=["python", "r40_capture_runner.py", "--input-dir",
                 runner._basename("/private/data/lights")],
        gen_dir=gen, index=index, verification=verification,
        repo_head=_REPO_HEAD_40,
        tool_sha={"runner": "1" * 64, "zemosaic_zegrid_mode": "2" * 64},
        frozen=runner.assert_frozen_bindings(c),
        input_inventory_fits_count=3,
    )
    assert report["schema"] == runner.REPORT_SCHEMA
    assert report["status"] == "VERIFIED"
    assert report["repo_head"] == _REPO_HEAD_40
    assert report["tool_source_sha256"]["runner"] == "1" * 64
    assert report["input"]["basename"] == "lights"
    assert report["input"]["n_frames"] == 816
    assert report["input"]["source_set_digest"] == "a" * 64
    assert report["input"]["source_set_digest_algo"] == runner.SOURCE_SET_DIGEST_ALGO
    assert report["input"]["source_set_count"] == 816
    assert report["input"]["input_inventory_fits_count"] == 3
    assert report["output"]["basename"] == "out"
    assert report["capture"]["capture_contract_version"] == "r40-capture-1"
    assert report["capture"]["capture_id"] == "capture-abc"
    assert report["capture"]["generation_dirname"] == gen.name
    assert report["capture"]["n_shards"] == 1 and report["capture"]["n_records"] == 1
    assert report["capture"]["generation_sha256"] == verification["index_sha256"]
    assert report["qualification"] == {
        "holdout_capped": False, "qualification": "EXACT", "n_dropped": 0,
    }
    assert report["verification"]["ok"] is True
    assert report["generation_verified"] is True
    assert report["dev_not_qualification"] is True
    assert isinstance(report["command"], list)

    # strict JSON, and refused by the accepted core privacy/finiteness boundary
    text = json.dumps(report, sort_keys=True, allow_nan=False)
    assert "/private/data" not in text
    c.assert_no_path_leaks(report)
    c.assert_finite_json(report, "$.report")


def test_scrub_command_reduces_absolute_paths_to_basenames():
    scrubbed = runner.scrub_command([
        "--input-dir", "/home/tristan/private/lights",
        "--output-dir", "C:\\Users\\TRISTAN\\OUT\\R40C",
        "--disk-budget-bytes", "268435456",
        "--holdout-cap", "0",
    ])
    assert scrubbed == [
        "--input-dir", "lights",
        "--output-dir", "R40C",
        "--disk-budget-bytes", "268435456",
        "--holdout-cap", "0",
    ]


# ---------------------------------------------------------------------------
# G. NO-FINAL-FITS guarantee (static + behavioural)
# ---------------------------------------------------------------------------

def test_runner_imports_no_fits_module():
    source = _TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level.add(node.module.split(".")[0])
    allowed = {"__future__", "argparse", "hashlib", "inspect", "json", "os", "re",
               "sys", "pathlib", "typing", "tempfile"}
    assert module_level <= allowed, module_level - allowed
    assert "astropy" not in module_level and "fits" not in module_level
    assert "zemosaic" not in module_level  # every engine import is lazy
    # ... and the lazy engine imports really exist (nested in functions), so the
    # guarantee above is not vacuous.
    nested = {n.module.split(".")[0] for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom) and n.module}
    assert "zemosaic" in nested


def test_dry_run_works_with_fits_open_poisoned(tmp_path, monkeypatch, capsys):
    def _boom(*args, **kwargs):  # pragma: no cover - only fires on a violation
        raise AssertionError("the R40-C runner must never open a FITS file")

    monkeypatch.setattr(fits, "open", _boom)
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "f0.fit").write_bytes(b"")
    rc = runner.main(["--input-dir", str(inp), "--output-dir", str(tmp_path / "out")])
    assert rc == runner._EXIT_OK
    assert "DRY-RUN complete" in capsys.readouterr().out


def test_verification_path_never_opens_fits(tmp_path, monkeypatch):
    gen = _write_generation(tmp_path / "gen")
    monkeypatch.setattr(fits, "open", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("verification must not read a FITS")))
    found = runner.find_published_generation(gen.parent)
    report = runner.verify_published_generation(found, core=c)
    index = runner.read_generation_index(found)
    assert report["ok"] is True
    assert index["identity"]["qualification"] == "EXACT"


# ---------------------------------------------------------------------------
# H. ONE real end-to-end execute (small synthetic corpus; NOT a Windows run)
# ---------------------------------------------------------------------------

def test_execute_end_to_end_on_synthetic_corpus(tmp_path, monkeypatch, capsys):
    """Real engine + real process pool, capture via the runner's exact seam.

    NOT a Windows run, and it claims NO qualification: it proves only that the
    runner's execute path reconstructs the seam, publishes a real generation,
    verifies it independently and writes the strict-JSON provenance report.
    """
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: _PIN_MEM_BYTES)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 3)
    real_coords = zz._STACK_R40_WORKER_COORDS
    try:
        inp = _write_corpus(tmp_path / "in")
        out = tmp_path / "out"
        rc = runner.main(["--input-dir", str(inp), "--output-dir", str(out), "--execute"])
        assert rc == runner._EXIT_OK
        assert (out / runner.REPORT_NAME).is_file()

        report = json.loads((out / runner.REPORT_NAME).read_text(encoding="utf-8"))
        assert report["generation_verified"] is True
        assert report["verification"]["ok"] is True
        assert report["verification"]["problems"] == []
        assert report["qualification"]["qualification"] == "EXACT"
        assert report["qualification"]["holdout_capped"] is False
        assert report["qualification"]["n_dropped"] == 0
        assert report["capture"]["n_shards"] >= 1
        assert report["capture"]["n_records"] >= 1
        assert report["capture"]["capture_contract_version"] == "r40-capture-1"
        assert len(report["repo_head"]) == 40
        assert report["input"]["basename"] == "in"
        assert report["input"]["n_frames"] >= 1
        assert len(report["input"]["source_set_digest"]) == 64
        assert report["output"]["basename"] == "out"

        # the published generation really is on disk under the output dir and
        # independently re-verifies with the strict production-wired expectation
        gen = runner.find_published_generation(out)
        assert gen.parent == out
        again = c.verify_generation(gen, expected_production_wired=True)
        assert again["ok"] is True and again["problems"] == []
        assert report["capture"]["generation_sha256"] == again["index_sha256"]

        # privacy: no absolute private path survives in the durable report
        c.assert_no_path_leaks(report)
        assert str(tmp_path) not in json.dumps(report)
    finally:
        zz._STACK_R40_WORKER_COORDS = real_coords
