"""ZM-ZEGRID-R55-B — observe-only product integration and propagation contract.

Acceptance proven here:

* ``zegrid_scheduler_mode`` defaults to ``legacy`` and unknown values fail
  visibly (never silently coerced);
* legacy and observe runs are byte-identical on the fixed routine corpus;
* the only manifest delta is the additive ``r55_scheduler`` block, which carries
  predicted envelopes and NO fabricated observed metric;
* the generic GUI/settings -> worker -> engine kwargs/signature guard holds.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import resource_envelope as re
from zemosaic.core.zegrid import scheduler as zs

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
N_FRAMES = 12
# The routine corpus is deliberately SMALL: its canvas cannot satisfy the
# 50000-px / 0.5-halo-overhead scientific floors at the product's 64-px halo on
# a low-RAM host. Halo 32 is a supported product value and keeps this witness
# about byte-identity rather than about host capacity.
WITNESS_HALO_PX = 32
# A FIXED observation makes the layout deterministic even when the CI/dev host
# is under memory pressure. It changes no science: both runs use the same value.
WITNESS_AVAILABLE_BYTES = 12 * 2 ** 30

VOLATILE_MANIFEST_KEYS = (
    "r55_scheduler",
    "timings",
    "peak_rss_kib",
    "aggregate_peak_rss_kib",
    "gauge_diagnostics",
    "per_cell_diagnostics",
    "reprojection",
)


# ---------------------------------------------------------------------------
# Mode contract
# ---------------------------------------------------------------------------

def test_scheduler_mode_defaults_to_legacy_and_rejects_unknown_values():
    assert zs.DEFAULT_MODE == "legacy"
    assert zs.normalize_scheduler_mode(None) == "legacy"
    assert zs.normalize_scheduler_mode("legacy") == "legacy"
    assert zs.normalize_scheduler_mode("observe") == "observe"
    for bad in ("", "auto", "ACTIVE", "OBSERVE", " legacy", True, 0, 1.0):
        with pytest.raises(zs.InvalidSchedulerMode):
            zs.normalize_scheduler_mode(bad)


def test_unknown_mode_fails_visibly_before_any_work(tmp_path):
    with pytest.raises(zs.InvalidSchedulerMode):
        zz.run_zegrid_mode(
            str(tmp_path / "does-not-exist"),
            str(tmp_path / "out"),
            zegrid_scheduler_mode="bogus",
        )
    # No output directory was created: the refusal happened before discovery.
    assert not (tmp_path / "out").exists()


def test_engine_signature_defaults_to_none_and_config_default_is_legacy():
    sig = inspect.signature(zz.run_zegrid_mode)
    assert sig.parameters["zegrid_scheduler_mode"].default is None
    from zemosaic import zemosaic_config

    assert zemosaic_config.DEFAULT_CONFIG["zegrid_scheduler_mode"] == "legacy"


# ---------------------------------------------------------------------------
# Manifest shape / no fabricated metrics
# ---------------------------------------------------------------------------

def _manifest_block():
    requests = (
        zs.PhaseRequest(
            phase=re.PHASE_GAUGE,
            components={"gauge_worker_footprint_bytes": 100},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_CACHE_BUILD,
            components={"cache_build_worker_footprint_bytes": 100},
            units=1,
            units_in_flight=1,
        ),
        zs.PhaseRequest(
            phase=re.PHASE_ASSEMBLY,
            components={"halo_band_bytes": 10, "retained_core_bytes": 100},
            units=1,
            units_in_flight=1,
            overlaps_with=(re.PHASE_CELL_STACK_STREAM,),
        ),
        zs.PhaseRequest(
            phase=re.PHASE_EXPORT,
            components={"coverage_plane_bytes": 10, "science_planes_bytes": 30},
            units=1,
            units_in_flight=1,
        ),
    )
    plan = zs.build_observe_plan(
        phase=re.PHASE_CELL_STACK_STREAM,
        budget_bytes=4 * 2 ** 30,
        raw_available_bytes=5 * 2 ** 30,
        factor_num=4,
        factor_den=5,
        cpu_logical=8,
        units_total=4,
        shared_bytes=340 * 2 ** 20,
        per_unit_stack_bytes=200 * 2 ** 20,
        per_unit_cache_bytes=10 * 2 ** 20,
        per_unit_prefetch_bytes=10 * 2 ** 20,
        disk_free_bytes=100 * 2 ** 30,
        layout=(2, 2),
        refinement_rank=4,
        patch_hw=(300, 300),
        halo_px=32,
        channels=3,
        n_frames=20,
        phase_requests=requests,
        cell_ids=("r0000c0000", "r0000c0001", "r0000c0002", "r0000c0003"),
        frame_ids=("f0", "f1"),
        mode=zs.MODE_OBSERVE,
        run_id="canvas-test",
    )
    return plan.to_manifest_dict()


def test_manifest_has_one_envelope_per_required_phase_and_no_metrics():
    block = _manifest_block()
    assert block["schema"] == "zemosaic-r55-manifest-v1"
    assert block["scheduler_version"] == "r55-b0"
    assert block["mode"] == "observe_only"
    assert block["requested_mode"] == "observe"
    assert set(block["phases"]) == {
        "gauge",
        "cache_build",
        "cell_stack_stream",
        "assembly",
        "export",
    }
    assert "gui_preview" not in block["phases"]
    assert block["observed_metrics"] is None
    assert block["fallbacks"] == []
    assert block["plan_hash"].startswith("sha256:")
    assert block["decisions"] and all(
        d["reason_code"] in re.REASON_CODES for d in block["decisions"]
    )
    assert block["margin"]["applied_once_at"] == "aggregate"
    stack_env = block["phases"]["cell_stack_stream"]["envelope"]
    assembly_env = block["phases"]["assembly"]["envelope"]
    assert stack_env["overlaps_with"] == ["assembly"]
    assert assembly_env["overlaps_with"] == ["cell_stack_stream"]
    stack_peak = stack_env["estimate"]["peak_bytes"]
    assembly_peak = assembly_env["estimate"]["peak_bytes"]
    raw = block["aggregate_host_peak_bytes_raw"]
    assert raw == max(
        block["phases"]["gauge"]["envelope"]["estimate"]["peak_bytes"],
        block["phases"]["cache_build"]["envelope"]["estimate"]["peak_bytes"],
        stack_peak + assembly_peak,
        block["phases"]["export"]["envelope"]["estimate"]["peak_bytes"],
    )
    assert block["aggregate_host_peak_bytes"] == re.ceil_margin_div(raw, 15, 100)
    assert stack_peak == stack_env["estimate"]["subtotal_bytes"]
    assert assembly_peak == assembly_env["estimate"]["subtotal_bytes"]
    # Predicted peaks only: no RSS/VRAM/disk "observed" keys exist.
    text = json.dumps(block, sort_keys=True)
    for forbidden in ("observed_rss", "observed_vram", "observed_peak", "throughput_units_per_s"):
        assert forbidden not in text
    # Deterministic JSON serialization.
    assert json.dumps(block, sort_keys=True) == json.dumps(_manifest_block(), sort_keys=True)


def test_internal_error_block_uses_the_closed_reason_code():
    block = zz._r55_observe_error_block(RuntimeError("synthetic failure"))
    assert block["status"] == "error"
    assert block["reason_code"] == re.REASON_INTERNAL_ERROR
    assert block["reason_code"] in re.REASON_CODES
    assert "observed_metrics" not in block
    assert "plan_hash" not in block


# ---------------------------------------------------------------------------
# GUI / settings -> worker -> engine propagation
# ---------------------------------------------------------------------------

class _RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_generic_gui_to_worker_signature_guard(monkeypatch, tmp_path):
    """Every transported kwargs key must be accepted by the real worker signature.

    This mirrors the dispatch-propagation witness: a capture stub is installed
    with the REAL signature, so a key the GUI sends but the worker does not
    declare would surface as an unexpected kwarg once forwarding is re-enabled.
    A real signature cannot be forged by the stub.
    """
    from zemosaic import zemosaic_worker as zw

    real_sig = inspect.signature(zw.run_hierarchical_mosaic)
    captured: dict = {}

    def stub(**final_kwargs):
        captured.update(final_kwargs)

    stub.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", stub)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    queue = _RecordingQueue()
    zw.run_hierarchical_mosaic_process(
        queue,
        solver_settings_dict=None,
        zegrid_scheduler_mode="observe",
        crash_breadcrumbs_mode="off",
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
    )
    assert captured.get("zegrid_scheduler_mode") == "observe"
    unexpected = set(captured) - set(real_sig.parameters)
    assert not unexpected, f"kwargs not accepted by the worker signature: {sorted(unexpected)}"
    # The engine accepts the same transport key (single validator downstream).
    assert "zegrid_scheduler_mode" in inspect.signature(zz.run_zegrid_mode).parameters


# ---------------------------------------------------------------------------
# Legacy vs observe on the fixed routine corpus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routine_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("r55_routine")
    corpus = root / "corpus"
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(corpus)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    inp = root / "input"
    inp.mkdir()
    for path in sorted(corpus.glob("*.fits")):
        shutil.copy2(path, inp / path.name)
    plan_csv = corpus / "stack_plan.csv"
    if plan_csv.exists():
        shutil.copy2(plan_csv, inp / "stack_plan.csv")
    return inp


def _run(inp, out, mode):
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(zz, "available_memory_bytes", lambda: WITNESS_AVAILABLE_BYTES)
        zz.run_zegrid_mode(
            str(inp),
            str(out),
            zegrid_halo_px=WITNESS_HALO_PX,
            zegrid_scheduler_mode=mode,
        )
    finally:
        monkeypatch.undo()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_and_observe_are_byte_identical_with_only_an_additive_block(
    routine_input, tmp_path
):
    legacy_out = tmp_path / "legacy"
    observe_out = tmp_path / "observe"
    _run(routine_input, legacy_out, "legacy")
    _run(routine_input, observe_out, "observe")

    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        assert _digest(legacy_out / name) == _digest(observe_out / name), name

    legacy = json.loads((legacy_out / "zegrid_manifest.json").read_text())
    observe = json.loads((observe_out / "zegrid_manifest.json").read_text())
    assert "r55_scheduler" not in legacy
    block = observe["r55_scheduler"]
    assert block["mode"] == "observe_only"
    assert block["observed_metrics"] is None
    assert set(block["phases"]) == {
        "gauge",
        "cache_build",
        "cell_stack_inmem",
        "assembly",
        "export",
    }
    assert block["plan_hash"].startswith("sha256:")
    assert block["decisions"][0]["reason_code"] in re.REASON_CODES
    stack_name = next(
        p for p in block["phases"] if p in ("cell_stack_inmem", "cell_stack_stream")
    )
    stack_env = block["phases"][stack_name]["envelope"]
    assembly_env = block["phases"]["assembly"]["envelope"]
    assert stack_env["overlaps_with"] == ["assembly"]
    assert assembly_env["overlaps_with"] == [stack_name]
    assert block["aggregate_host_peak_bytes_raw"] >= (
        stack_env["estimate"]["peak_bytes"] + assembly_env["estimate"]["peak_bytes"]
    )
    assert block["aggregate_host_peak_bytes"] == re.ceil_margin_div(
        block["aggregate_host_peak_bytes_raw"], 15, 100
    )
    # Multi-cell retained cores are counted ONCE in the assembly envelope.
    assert assembly_env["estimate"]["components"]["retained_core_bytes"] > 0
    assert len(block["identity"]["cell_ids"]) > 1

    observe_without_block = dict(observe)
    observe_without_block.pop("r55_scheduler", None)
    for key in VOLATILE_MANIFEST_KEYS:
        legacy.pop(key, None)
        observe_without_block.pop(key, None)
    assert legacy == observe_without_block

    # Structural science fields are preserved exactly.
    for key in ("complete_cells", "incomplete_cells", "coverage_pixels", "layout"):
        assert legacy[key] == observe_without_block[key]


def test_generic_observe_error_is_contained_and_coded(routine_input, tmp_path, monkeypatch):
    """A generic observe failure must not abort or alter legacy execution."""
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: WITNESS_AVAILABLE_BYTES)

    def boom(**_kwargs):
        raise RuntimeError("synthetic observe failure")

    legacy_out = tmp_path / "legacy"
    observe_out = tmp_path / "observe"
    zz.run_zegrid_mode(
        str(routine_input), str(legacy_out),
        zegrid_halo_px=WITNESS_HALO_PX, zegrid_scheduler_mode="legacy",
    )
    with monkeypatch.context() as patch:
        patch.setattr(zz, "_build_r55_observe_block", boom)
        zz.run_zegrid_mode(
            str(routine_input), str(observe_out),
            zegrid_halo_px=WITNESS_HALO_PX, zegrid_scheduler_mode="observe",
        )
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        assert _digest(legacy_out / name) == _digest(observe_out / name), name
    block = json.loads((observe_out / "zegrid_manifest.json").read_text())["r55_scheduler"]
    assert block["status"] == "error"
    assert block["reason_code"] == re.REASON_INTERNAL_ERROR
    assert "observed_metrics" not in block
