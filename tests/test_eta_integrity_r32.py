"""ZM-ETA-INTEGRITY-R32 focused regressions — history integrity + honest ETA.

PURE tests (stdlib only) covering the defects exposed by the real M16 GPU run:

* HISTORY COMPARABILITY (fail closed): contaminated tiny CPU/GPU synthetic
  records are rejected for an M16-like GPU context; genuinely comparable
  same-backend history is accepted and scaled by WORK RATIO; old records missing
  the rich mandatory workload signature are never silently comparable; the
  effective backend is a hard dimension.
* WORK-MEASURE SEMANTICS: ``input_pixels_total`` is the SUM of frame areas
  (never multiplied by ``n_frames`` again); per-cell stack work is the measured
  ``sum(patch_area_px * contributors)``; equal input pixels must not double a
  gauge prior and a 10x stack-work record is rejected.
* HISTORY ISOLATION: the authoritative path seam is used by both writers, the
  pytest session guard keeps every test on temp storage, and a direct
  ``run_zegrid_mode`` run cannot touch the HOME history file.
* EFFECTIVE-BACKEND TRACKING: a GPU -> CPU pre-execution fallback (or final
  actual-backend correction) clears and reselects priors immediately.
* ONE HONEST ETA CONTRACT: the M16 timing trace replay shows seconds (not the
  ~102 s phantom) after ``per_cell_stack`` 4/4, the diagnostic ``cache_build``
  is never future work, and terminal success is exactly 0.

All clocks are injected; no long real-corpus/slow work happens here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from zemosaic import eta_service as zeta
from zemosaic import progress_contract as zprogress
from zemosaic.core.zegrid import observability as zobs

SRC = Path(__file__).resolve().parents[1] / "src" / "zemosaic"

# Real M16 measurement (read-only evidence): 32 frames, 4 cells, GPU.
M16_STAGES = {
    "setup": 0.626,
    "layout": 0.271,
    "gauge": 101.407,
    "per_cell_stack": 292.667,
    "assembly": 1.342,
    "finalize": 2.415,
}
M16_FRAMES = 32
M16_CELLS = 4
# Sanitized geometry proxies (same order of magnitude as a full sensor).
M16_FRAME_PX = 16_000_000          # per-frame input area
M16_CANVAS_PX = 4_800 * 3_600
M16_STACK_WORK = 4 * 12_000_000 * 30  # sum(patch_area * contributors)


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def _work(frames=M16_FRAMES, cells=M16_CELLS, input_pixels=None, stack_work=None,
          canvas_px=M16_CANVAS_PX):
    if input_pixels is None:
        input_pixels = M16_FRAME_PX * frames
    if stack_work is None:
        stack_work = M16_STACK_WORK * frames * cells // (M16_FRAMES * M16_CELLS)
    return zeta.build_workload_signature(
        n_frames=frames,
        cell_count=cells,
        input_pixels_total=input_pixels,
        stack_work_px=stack_work,
        canvas_px=canvas_px,
    )


# ---------------------------------------------------------------------------
# History fixtures
# ---------------------------------------------------------------------------

def _m16_like_record(*, frames=M16_FRAMES, cells=M16_CELLS, backend="gpu", scale=1.0):
    """A genuinely comparable M16-like GPU record (a previous real run)."""
    return zeta.build_zegrid_history_record(
        total_duration_s=sum(M16_STAGES.values()) * scale,
        n_frames=frames,
        cell_count=cells,
        stage_seconds={k: v * scale for k, v in M16_STAGES.items()},
        stage_totals={
            "setup": frames,
            "gauge": 2 * frames - 1,
            "per_cell_stack": cells,
            "finalize": 1,
        },
        backend=backend,
        workload=_work(frames=frames, cells=cells),
    )


def _m16_context(*, backend="gpu", frames=M16_FRAMES, cells=M16_CELLS):
    return {
        "n_frames": frames,
        "cell_count": cells,
        "stage_totals": None,
        "backend": backend,
        "workload": _work(frames=frames, cells=cells),
    }


def _tiny_record(*, frames=4, cells=1, backend="cpu", gauge=0.9, per_cell=1.6):
    """Synthetic direct-engine junk: tiny workload, short stage times."""
    return zeta.build_zegrid_history_record(
        total_duration_s=gauge + per_cell + 0.3,
        n_frames=frames,
        cell_count=cells,
        stage_seconds={
            "setup": 0.05, "layout": 0.02, "gauge": gauge,
            "per_cell_stack": per_cell, "assembly": 0.1, "finalize": 0.05,
        },
        stage_totals={
            "setup": frames,
            "gauge": 2 * frames - 1,
            "per_cell_stack": cells,
            "finalize": 1,
        },
        backend=backend,
        workload=zeta.build_workload_signature(
            n_frames=frames,
            cell_count=cells,
            input_pixels_total=256 * 256 * frames,
            stack_work_px=256 * 256 * frames,
            canvas_px=256 * 256,
        ),
    )


def _contaminated_history() -> list:
    """95 tiny records (the real-world contamination) + a few GPU ones."""
    records = [_tiny_record(backend="cpu") for _ in range(80)]
    records += [_tiny_record(backend="gpu", gauge=4.955, per_cell=1.606) for _ in range(15)]
    return records


# ---------------------------------------------------------------------------
# B1/B2: comparability fails closed
# ---------------------------------------------------------------------------

def test_contaminated_history_rejected_for_m16_gpu_context():
    records = _contaminated_history()
    assert len(records) == 95
    priors, comparable = zeta.select_zegrid_priors(records, **_m16_context())
    assert comparable == 0
    assert priors == {}


def test_same_backend_comparable_history_accepted_and_scaled():
    records = [_m16_like_record()]
    priors, comparable = zeta.select_zegrid_priors(records, **_m16_context())
    assert comparable == 1
    # Work-ratio scaling is 1:1 for an identical workload.
    assert priors["zegrid:gauge"] == pytest.approx(M16_STAGES["gauge"], rel=1e-6)
    assert priors["zegrid:per_cell_stack"] == pytest.approx(
        M16_STAGES["per_cell_stack"], rel=1e-6
    )
    # A neighbouring workload with HALF the frames (same cells) is comparable:
    # its input pixels and measured stack work are both exactly 2x smaller.
    half = [_m16_like_record(frames=16, cells=M16_CELLS)]
    priors2, comparable2 = zeta.select_zegrid_priors(half, **_m16_context())
    assert comparable2 == 1
    assert priors2["zegrid:setup"] == pytest.approx(2 * M16_STAGES["setup"], rel=1e-6)
    assert priors2["zegrid:gauge"] == pytest.approx(2 * M16_STAGES["gauge"], rel=1e-6)
    assert priors2["zegrid:per_cell_stack"] == pytest.approx(
        2 * M16_STAGES["per_cell_stack"], rel=1e-6
    )
    # Halving BOTH frames and cells quarters the measured stack work: that is
    # genuinely not comparable (4x > band) and must be rejected.
    quarter = [_m16_like_record(frames=16, cells=2)]
    assert zeta.select_zegrid_priors(quarter, **_m16_context()) == ({}, 0)


def test_cpu_and_gpu_never_mix():
    cpu = _m16_like_record(backend="cpu")
    # A GPU run must never consume the CPU record, and vice versa.
    assert zeta.select_zegrid_priors([cpu], **_m16_context(backend="gpu")) == ({}, 0)
    gpu = _m16_like_record(backend="gpu")
    assert zeta.select_zegrid_priors([gpu], **_m16_context(backend="cpu")) == ({}, 0)
    # Unknown effective backend fails closed too.
    assert zeta.select_zegrid_priors([gpu], **_m16_context(backend=None)) == ({}, 0)


def test_tiny_gpu_record_rejected_even_with_matching_mode_and_cells():
    tiny = _tiny_record(frames=M16_FRAMES, cells=M16_CELLS, backend="gpu")
    priors, comparable = zeta.select_zegrid_priors([tiny], **_m16_context())
    assert comparable == 0 and priors == {}


# ---------------------------------------------------------------------------
# F1: every rich dimension is MANDATORY on both sides
# ---------------------------------------------------------------------------

def test_incomplete_v3_workload_fails_closed():
    # The exact r0 defect: a v3 record with only n_frames/cell_count passed.
    partial = _m16_like_record()
    partial["workload"] = {"v": 1, "n_frames": M16_FRAMES, "cell_count": M16_CELLS}
    assert zeta.select_zegrid_priors([partial], **_m16_context()) == ({}, 0)


@pytest.mark.parametrize("field", list(zeta.WORKLOAD_REQUIRED_FIELDS))
def test_record_missing_any_required_field_is_rejected(field):
    rec = _m16_like_record()
    rec["workload"].pop(field)
    assert zeta.select_zegrid_priors([rec], **_m16_context()) == ({}, 0)


@pytest.mark.parametrize("field", list(zeta.WORKLOAD_REQUIRED_FIELDS))
def test_current_context_missing_any_required_field_is_rejected(field):
    ctx = _m16_context()
    ctx["workload"].pop(field)
    # n_frames/cell_count may be supplied explicitly; remove them too so the
    # ONLY source is the (incomplete) workload signature.
    if field == "n_frames":
        ctx["n_frames"] = None
    if field == "cell_count":
        ctx["cell_count"] = None
    assert zeta.select_zegrid_priors([_m16_like_record()], **ctx) == ({}, 0)


def test_missing_rich_signature_fails_closed():
    # Old v2-era record: correct backend, NO workload signature.
    legacy_v2 = _m16_like_record()
    legacy_v2.pop("workload")
    legacy_v2["schema_v"] = 2
    assert zeta.select_zegrid_priors([legacy_v2], **_m16_context()) == ({}, 0)

    # Record without a backend dimension.
    no_backend = _m16_like_record()
    no_backend.pop("backend")
    assert zeta.select_zegrid_priors([no_backend], **_m16_context()) == ({}, 0)

    # Unknown signature version.
    bad_version = _m16_like_record()
    bad_version["workload"] = dict(bad_version["workload"], v=99)
    assert zeta.select_zegrid_priors([bad_version], **_m16_context()) == ({}, 0)

    # Incomplete CURRENT context (no work measures) fails closed.
    assert zeta.select_zegrid_priors(
        [_m16_like_record()], backend="gpu", n_frames=32, cell_count=4
    ) == ({}, 0)


# ---------------------------------------------------------------------------
# F2: work-measure semantics and work-ratio scaling
# ---------------------------------------------------------------------------

def test_equal_input_pixels_does_not_double_gauge():
    """16-frame and 32-frame workloads with the SAME total input pixels.

    The gauge prior must stay at 100 s (work ratio 1.0), never become 200 s from
    an item/frame-count ratio. Setup legitimately scales by frame count (2x).
    """
    common_work = _work(frames=M16_FRAMES)
    rec = zeta.build_zegrid_history_record(
        total_duration_s=1000.0,
        n_frames=16,
        cell_count=M16_CELLS,
        stage_seconds={
            "setup": 10.0, "layout": 700.0, "gauge": 100.0,
            "per_cell_stack": 150.0, "assembly": 20.0, "finalize": 15.0,
        },
        # exact item totals deliberately contradict the pixel-work truth.
        stage_totals={"setup": 16, "gauge": 31, "per_cell_stack": M16_CELLS},
        backend="gpu",
        workload=zeta.build_workload_signature(
            n_frames=16,
            cell_count=M16_CELLS,
            input_pixels_total=common_work["input_pixels_total"],
            stack_work_px=common_work["stack_work_px"],
            canvas_px=M16_CANVAS_PX,
        ),
    )
    priors, comparable = zeta.select_zegrid_priors(
        [rec],
        backend="gpu",
        n_frames=M16_FRAMES,
        cell_count=M16_CELLS,
        stage_totals={"setup": 32, "gauge": 63, "per_cell_stack": 4},
        workload=common_work,
    )
    assert comparable == 1
    assert priors["zegrid:gauge"] == pytest.approx(100.0)
    assert priors["zegrid:setup"] == pytest.approx(20.0)  # 2x frames
    # Per-cell stack work is equal -> unscaled, despite the exact cell total.
    assert priors["zegrid:per_cell_stack"] == pytest.approx(150.0)


def test_equal_cells_but_10x_stack_work_is_rejected():
    work = _work()
    rec = _m16_like_record()
    rec["workload"] = dict(rec["workload"], stack_work_px=work["stack_work_px"] * 10)
    priors, comparable = zeta.select_zegrid_priors([rec], **_m16_context())
    assert comparable == 0 and priors == {}


def test_per_cell_prior_scales_with_measured_stack_work():
    work = _work()
    rec = zeta.build_zegrid_history_record(
        total_duration_s=1000.0,
        n_frames=M16_FRAMES,
        cell_count=M16_CELLS,
        stage_seconds={
            "setup": 10.0, "layout": 700.0, "gauge": 100.0,
            "per_cell_stack": 100.0, "assembly": 20.0, "finalize": 15.0,
        },
        backend="gpu",
        workload=zeta.build_workload_signature(
            n_frames=M16_FRAMES,
            cell_count=M16_CELLS,
            input_pixels_total=work["input_pixels_total"],
            stack_work_px=work["stack_work_px"],
            canvas_px=M16_CANVAS_PX,
        ),
    )
    ctx_work = dict(work, stack_work_px=work["stack_work_px"] * 2)
    priors, comparable = zeta.select_zegrid_priors(
        [rec], backend="gpu", workload=ctx_work
    )
    assert comparable == 1
    assert priors["zegrid:per_cell_stack"] == pytest.approx(200.0)
    # Gauge uses input pixels only: unchanged by stack work.
    assert priors["zegrid:gauge"] == pytest.approx(100.0)


def test_assembly_and_finalize_scale_with_canvas():
    work = _work()
    rec = _m16_like_record()
    priors, comparable = zeta.select_zegrid_priors(
        [rec],
        backend="gpu",
        workload=dict(work, canvas_px=M16_CANVAS_PX * 2),
    )
    assert comparable == 1
    assert priors["zegrid:assembly"] == pytest.approx(2 * M16_STAGES["assembly"])
    assert priors["zegrid:finalize"] == pytest.approx(2 * M16_STAGES["finalize"])


# ---------------------------------------------------------------------------
# F3: effective-backend tracking (tracker-level, not grep)
# ---------------------------------------------------------------------------

def _tracker(estimator, records):
    return zeta.EtaContextTracker(
        estimator, records_loader=lambda: list(records), publish=None
    )


def test_backend_fallback_clears_gpu_priors_and_reselects_cpu():
    gpu_rec = _m16_like_record(backend="gpu")
    cpu_rec = _m16_like_record(backend="cpu", scale=2.0)
    est = zeta.HybridEtaEstimator(zprogress.ZEGRID_PLAN, clock=FakeClock())
    tracker = _tracker(est, [gpu_rec, cpu_rec])
    tracker.note_backend("gpu")
    tracker.note_workload(
        n_frames=M16_FRAMES,
        cell_count=M16_CELLS,
        input_pixels_total=M16_FRAME_PX * M16_FRAMES,
        stack_work_px=M16_STACK_WORK,
        canvas_px=M16_CANVAS_PX,
    )
    assert est.diagnostics()["priors"]["zegrid:gauge"] == pytest.approx(
        M16_STAGES["gauge"]
    )
    # Pre-execution / final fallback to CPU: priors now come from the CPU record
    # (2x slower), never from the GPU one.
    assert tracker.note_backend("cpu") is True
    assert est.diagnostics()["backend"] == "cpu"
    assert est.diagnostics()["priors"]["zegrid:gauge"] == pytest.approx(
        2 * M16_STAGES["gauge"]
    )


def test_backend_fallback_clears_priors_when_no_comparable_cpu_record():
    gpu_rec = _m16_like_record(backend="gpu")
    est = zeta.HybridEtaEstimator(zprogress.ZEGRID_PLAN, clock=FakeClock())
    tracker = _tracker(est, [gpu_rec])
    tracker.note_backend("gpu")
    tracker.note_workload(
        n_frames=M16_FRAMES,
        cell_count=M16_CELLS,
        input_pixels_total=M16_FRAME_PX * M16_FRAMES,
        stack_work_px=M16_STACK_WORK,
        canvas_px=M16_CANVAS_PX,
    )
    assert est.diagnostics()["priors"]
    assert tracker.note_backend("cpu") is True
    assert est.diagnostics()["priors"] == {}
    assert est.diagnostics()["comparable_histories"] == 0
    # Idempotent: a repeated identical backend does not churn/reselect.
    assert tracker.note_backend("cpu") is False


def test_tracker_publishes_every_context_change():
    published = []
    est = zeta.HybridEtaEstimator(zprogress.ZEGRID_PLAN, clock=FakeClock())
    tracker = zeta.EtaContextTracker(
        est, records_loader=lambda: [], publish=published.append
    )
    tracker.note_backend("gpu")
    tracker.note_backend("cpu")
    tracker.note_backend("cpu")  # no change -> no publish
    assert [p["backend"] for p in published] == ["gpu", "cpu"]
    tracker.note_workload(n_frames=32, cell_count=4, input_pixels_total=1,
                          stack_work_px=2, canvas_px=3)
    assert published[-1]["input_pixels_total"] == 1
    assert published[-1]["stack_work_px"] == 2


# ---------------------------------------------------------------------------
# A1/A2: path seam + pytest isolation
# ---------------------------------------------------------------------------

def test_pytest_history_is_isolated_under_temp_storage():
    resolved = zeta.eta_history_path()
    home_file = Path.home() / zeta.DEFAULT_ETA_HISTORY_NAME
    assert resolved != home_file
    assert "pytest" in str(resolved) or "tmp" in str(resolved)
    # The legacy worker writer must resolve through the SAME seam.
    from zemosaic import zemosaic_worker as zworker

    assert zworker._eta_history_path() == resolved


def test_default_writer_prefers_env_over_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    sentinel = fake_home / zeta.DEFAULT_ETA_HISTORY_NAME
    sentinel.write_text(json.dumps({"schema": "sentinel", "records": []}), encoding="utf-8")
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    monkeypatch.setattr(zeta.Path, "home", classmethod(lambda cls: fake_home))
    temp_target = tmp_path / "temp_history.json"
    monkeypatch.setenv(zeta.ETA_HISTORY_ENV_VAR, str(temp_target))

    assert zeta.eta_history_path() == temp_target
    assert zeta.append_zegrid_history(_m16_like_record()) is True
    assert temp_target.exists()
    # The HOME sentinel is byte-identical: no test/direct run can reach it.
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


# ---------------------------------------------------------------------------
# C: one honest M16 ETA contract
# ---------------------------------------------------------------------------

def _m16_estimator(clock, *, priors=None, comparable=0):
    est = zeta.HybridEtaEstimator(zprogress.ZEGRID_PLAN, clock=clock)
    if priors is not None:
        est.set_priors(priors, comparable_histories=comparable)
    est.set_context(n_frames=M16_FRAMES, cell_count=M16_CELLS, backend="gpu")
    return est


def _replay_m16(est: zeta.HybridEtaEstimator, clock: FakeClock):
    """Replay the real M16 run-log trace; returns the per_cell_stack 4/4 result."""
    est.on_stage("zegrid:setup", 0, M16_FRAMES)
    clock.advance(M16_STAGES["setup"])
    est.on_stage("zegrid:setup", M16_FRAMES, M16_FRAMES)
    est.on_stage("zegrid:layout", 0, 0)
    clock.advance(M16_STAGES["layout"])
    # Diagnostic cache_build is fused into the per-cell batch: it must never
    # become separate future work.
    est.on_stage("zegrid:cache_build", 0, 0)
    clock.advance(M16_STAGES["gauge"])
    est.on_stage("zegrid:gauge", 63, 63)
    step = M16_STAGES["per_cell_stack"] / 4.0
    for done in range(1, M16_CELLS + 1):
        clock.advance(step)
        last = est.on_stage("zegrid:per_cell_stack", done, M16_CELLS)
    return last


def test_m16_trace_after_stack_completion_is_seconds():
    clock = FakeClock(0.0)
    priors, comparable = zeta.select_zegrid_priors([_m16_like_record()], **_m16_context())
    assert comparable == 1
    est = _m16_estimator(clock, priors=priors, comparable=comparable)
    result = _replay_m16(est, clock)
    assert result.ready is True
    assert result.active_stage == "zegrid:per_cell_stack"
    # Only assembly + finalize remain: bounded by their compatible evidence.
    assert result.future_floor == pytest.approx(
        M16_STAGES["assembly"] + M16_STAGES["finalize"], rel=1e-6
    )
    assert 0.0 < result.remaining_seconds <= 10.0
    assert result.remaining_seconds < 102.0  # the real phantom


def test_m16_trace_terminal_success_is_exactly_zero():
    clock = FakeClock(0.0)
    priors, comparable = zeta.select_zegrid_priors([_m16_like_record()], **_m16_context())
    est = _m16_estimator(clock, priors=priors, comparable=comparable)
    _replay_m16(est, clock)
    clock.advance(M16_STAGES["assembly"])
    est.on_stage("zegrid:assembly", 0, 0)
    clock.advance(M16_STAGES["finalize"])
    est.on_stage("zegrid:finalize", 1, 1)
    done = est.mark_success()
    assert done.remaining_seconds == 0.0
    assert done.terminal == "success"


def test_first_run_without_comparable_prior_is_honest_not_learned():
    clock = FakeClock(0.0)
    # No priors at all: the contaminated history is rejected upstream.
    est = _m16_estimator(clock, priors={}, comparable=0)
    r = est.on_stage("zegrid:setup", 0, M16_FRAMES)
    assert r.ready is False and r.remaining_seconds is None
    result = _replay_m16(est, clock)
    # Live-based, low confidence, and still bounded (no mean-of-completed
    # phantom of ~102 s / 292 s).
    assert result.ready is True
    assert result.confidence == zeta.CONFIDENCE_LOW
    assert result.remaining_seconds < 60.0


def test_cache_build_and_current_phase_are_never_future_work():
    clock = FakeClock(0.0)
    priors, comparable = zeta.select_zegrid_priors([_m16_like_record()], **_m16_context())
    est = _m16_estimator(clock, priors=priors, comparable=comparable)
    est.on_stage("zegrid:setup", 0, M16_FRAMES)
    clock.advance(M16_STAGES["setup"])
    est.on_stage("zegrid:setup", M16_FRAMES, M16_FRAMES)
    est.on_stage("zegrid:layout", 0, 0)
    clock.advance(M16_STAGES["layout"])
    est.on_stage("zegrid:gauge", 63, 63)
    before = est.on_stage("zegrid:per_cell_stack", 3, M16_CELLS)
    # A diagnostic cache_build event must not add future work.
    after = est.on_stage("zegrid:cache_build", 1000, 4000)
    assert after.remaining_seconds == before.remaining_seconds
    # The ACTIVE phase is not also counted as future: the floor only holds
    # assembly + finalize.
    assert before.future_floor == pytest.approx(
        M16_STAGES["assembly"] + M16_STAGES["finalize"], rel=1e-6
    )


def test_hybrid_log_eta_bridge_matches_shared_authority():
    clock = FakeClock(0.0)
    priors, comparable = zeta.select_zegrid_priors([_m16_like_record()], **_m16_context())
    est = _m16_estimator(clock, priors=priors, comparable=comparable)
    bridge = zobs.HybridLogEta(est)
    est.on_stage("zegrid:setup", 0, M16_FRAMES)
    clock.advance(M16_STAGES["setup"])
    est.on_stage("zegrid:setup", M16_FRAMES, M16_FRAMES)
    est.on_stage("zegrid:layout", 0, 0)
    clock.advance(M16_STAGES["layout"])
    est.on_stage("zegrid:gauge", 63, 63)
    for done in range(1, M16_CELLS + 1):
        clock.advance(M16_STAGES["per_cell_stack"] / 4.0)
        bridge.on_stage("zegrid:per_cell_stack", done, M16_CELLS)
    remaining = bridge(9999.0)  # the phase-local ETA is deliberately ignored
    assert remaining is not None
    assert 0.0 < remaining <= 10.0


def test_run_log_wiring_uses_shared_authority_not_global_eta():
    text = (SRC / "zemosaic_zegrid_mode.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    run_single = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_single":
            run_single = node
            break
    assert run_single is not None
    seg = ast.get_source_segment(text, run_single)
    assert "zobs.HybridLogEta(" in seg
    assert "zobs.GlobalEta(" not in seg
    assert "phase_completed(" not in seg
    assert "global_eta=global_eta" in seg
    # Backend transitions are TRACKED facts (pre-execution + final actual).
    assert seg.count("_log_eta_tracker.note_backend(") >= 3
    # The canonical phase order excludes the diagnostic counter and includes
    # the real finalize phase.
    order = _phase_order_tuple(text)
    assert "cache_build" not in order
    assert "finalize" in order


def _phase_order_tuple(source: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_PHASE_ORDER":
                    return ast.literal_eval(node.value)
    raise AssertionError("_PHASE_ORDER not found")


def test_phase_order_matches_progress_plan():
    from zemosaic import zemosaic_zegrid_mode as zz

    plan_ids = tuple(d.id.split(":", 1)[1] for d in zprogress.ZEGRID_PLAN.stages)
    assert zz._PHASE_ORDER == plan_ids
