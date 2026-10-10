"""ZM-PROGRESS-CONTRACT-R29 focused tests — pure deterministic progress contract.

These are PURE tests (no Qt, no heavy imports): they exercise the plan
descriptors, the id normalization, and the deterministic aggregator, plus two
STATIC source-level seam checks (the controller's STAGE_PROGRESS routing and the
ZeGrid finalize-seam ordering) that do not import the Qt GUI or the heavy
production stack. They run under the 1.0 GiB MemAvailable gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from zemosaic import progress_contract as z
from zemosaic.core.zegrid import observability as zobs

SRC = Path(__file__).resolve().parents[1] / "src" / "zemosaic"


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plan", [z.LEGACY_PLAN, z.SDS_PLAN, z.ZEGRID_PLAN])
def test_plan_validates(plan):
    plan.validate()  # raises on duplicate ids / bad weights / order


def test_legacy_plan_mirrors_historical_stage_order_and_weights():
    plan = z.LEGACY_PLAN
    assert [s.id for s in plan.stages] == [
        "phase1", "phase2", "phase3", "phase4", "phase4_5", "phase5", "phase6", "phase7"
    ]
    weights = {s.id: s.weight for s in plan.stages}
    assert weights == {
        "phase1": 30.0, "phase2": 5.0, "phase3": 35.0, "phase4": 5.0,
        "phase4_5": 6.0, "phase5": 9.0, "phase6": 8.0, "phase7": 2.0,
    }


def test_zegrid_plan_has_six_phases_with_finalize_and_merged_cache_build():
    plan = z.ZEGRID_PLAN
    assert [s.id for s in plan.stages] == [
        "zegrid:setup", "zegrid:layout", "zegrid:gauge",
        "zegrid:per_cell_stack", "zegrid:assembly", "zegrid:finalize",
    ]
    # cache_build is an alias of per_cell_stack (merged reality), NOT a 7th stage.
    assert plan.normalize_id("zegrid:cache_build") == "zegrid:per_cell_stack"
    assert plan.stage("zegrid:cache_build") is None
    assert plan.stage("zegrid:per_cell_stack") is not None


def test_sds_plan_has_seven_equal_phases():
    plan = z.SDS_PLAN
    assert len(plan.stages) == 7
    total = sum(s.weight for s in plan.stages)
    assert total == pytest.approx(100.0)


def test_plan_rejects_duplicate_ids():
    bad = z.ModePlan(
        "x",
        (
            z.StageDescriptor("a", "x", 1, "k", 0, 50.0),
            z.StageDescriptor("a", "x", 2, "k", 1, 50.0),
        ),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_plan_rejects_non_normalized_weights():
    bad = z.ModePlan(
        "x",
        (
            z.StageDescriptor("a", "x", 1, "k", 0, 40.0),
            z.StageDescriptor("b", "x", 2, "k", 1, 40.0),
        ),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_plan_rejects_nonpositive_weight():
    bad = z.ModePlan(
        "x",
        (
            z.StageDescriptor("a", "x", 1, "k", 0, 0.0),
            z.StageDescriptor("b", "x", 2, "k", 1, 100.0),
        ),
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_plan_rejects_alias_to_unknown():
    bad = z.ModePlan(
        "x",
        (z.StageDescriptor("a", "x", 1, "k", 0, 100.0),),
        aliases={"alias": "nope"},
    )
    with pytest.raises(ValueError):
        bad.validate()


# ---------------------------------------------------------------------------
# Id normalization (old + new ZeGrid ids; counters never canonical)
# ---------------------------------------------------------------------------

def test_normalize_old_counter_embedded_zegrid_ids():
    plan = z.ZEGRID_PLAN
    assert plan.normalize_id("zegrid:gauge:113/1305") == "zegrid:gauge"
    assert plan.normalize_id("zegrid:gauge:0/100") == "zegrid:gauge"
    assert plan.normalize_id("zegrid:gauge:100/100") == "zegrid:gauge"


def test_normalize_new_stable_zegrid_ids_unchanged():
    plan = z.ZEGRID_PLAN
    for sid in (
        "zegrid:setup", "zegrid:layout", "zegrid:gauge",
        "zegrid:per_cell_stack", "zegrid:assembly", "zegrid:finalize",
    ):
        assert plan.normalize_id(sid) == sid


def test_normalize_counter_embedded_alias_collapses_to_canonical():
    plan = z.ZEGRID_PLAN
    assert plan.normalize_id("zegrid:cache_build:5/10") == "zegrid:per_cell_stack"


def test_counters_never_part_of_canonical_id():
    plan = z.ZEGRID_PLAN
    for sid in ("zegrid:gauge:1/2", "zegrid:setup:3/4", "zegrid:per_cell_stack:99/100"):
        assert "/" not in plan.normalize_id(sid)


def test_legacy_alias_normalization():
    plan = z.LEGACY_PLAN
    assert plan.normalize_id("phase1_scan") == "phase1"
    assert plan.normalize_id("phase5_intertile") == "phase5"
    assert plan.normalize_id("phase5_reproject") == "phase5"
    assert plan.normalize_id("phase4_grid") == "phase4"


def test_unknown_ids_preserved_verbatim():
    assert z.LEGACY_PLAN.normalize_id("stack_winsorized") == "stack_winsorized"
    assert z.ZEGRID_PLAN.normalize_id("bogus:stage") == "bogus:stage"


# ---------------------------------------------------------------------------
# Aggregator: monotonic, local fraction, terminal semantics
# ---------------------------------------------------------------------------

def _full_zegrid_sequence():
    """Return (agg, results) for a complete ZeGrid run ending at finalize 100%."""
    agg = z.ProgressAggregator(z.ZEGRID_PLAN)
    results = []
    for sid, cur, tot in (
        ("zegrid:setup", 100, 100),
        ("zegrid:layout", 500, 500),
        ("zegrid:gauge", 1305, 1305),
        ("zegrid:per_cell_stack", 170, 170),
        ("zegrid:assembly", 170, 170),
        ("zegrid:finalize", 1, 1),
    ):
        results.append(agg.on_stage(sid, cur, tot))
    return agg, results


def test_zegrid_monotonic_global_and_local_fractions():
    agg, results = _full_zegrid_sequence()
    percents = [r.global_percent for r in results]
    assert percents == sorted(percents), "global progress is not monotonic"
    # local fractions are correct for each stage
    assert [r.local_fraction for r in results] == [1.0] * 6
    assert [r.ordinal for r in results] == [1, 2, 3, 4, 5, 6]


def test_zegrid_each_stage_local_100_does_not_set_global_100():
    agg = z.ProgressAggregator(z.ZEGRID_PLAN)
    for sid, cur, tot in (
        ("zegrid:setup", 100, 100),
        ("zegrid:layout", 500, 500),
        ("zegrid:gauge", 1305, 1305),
    ):
        r = agg.on_stage(sid, cur, tot)
        assert r.global_percent < 100.0


def test_zegrid_finalize_local_100_still_below_100_until_success():
    agg, results = _full_zegrid_sequence()
    assert results[-1].known
    assert results[-1].ordinal == 6  # finalize
    assert results[-1].global_percent < 100.0
    assert agg.terminal is None
    agg.mark_success()
    assert agg.global_percent == 100.0
    assert agg.terminal == "success"


def test_mark_success_is_the_only_path_to_100():
    agg, _ = _full_zegrid_sequence()
    assert agg.global_percent < 100.0
    agg.mark_success()
    assert agg.global_percent == 100.0


def test_mark_fail_never_reaches_100():
    agg, _ = _full_zegrid_sequence()
    agg.mark_fail()
    assert agg.global_percent < 100.0
    assert agg.terminal == "fail"


def test_mark_cancel_never_reaches_100():
    agg, _ = _full_zegrid_sequence()
    agg.mark_cancel()
    assert agg.global_percent < 100.0
    assert agg.terminal == "cancel"


def test_stage_transition_marks_prior_stages_complete():
    agg = z.ProgressAggregator(z.ZEGRID_PLAN)
    r = agg.on_stage("zegrid:gauge", 0, 1305)
    # gauge is position 2: setup (0) and layout (1) are proven complete.
    # global = setup(5) + layout(15) + gauge fraction 0 = 20
    assert r.global_percent == pytest.approx(20.0)
    assert agg._fractions["zegrid:setup"] == 1.0
    assert agg._fractions["zegrid:layout"] == 1.0


def test_unknown_stage_never_jumps_progress():
    agg = z.ProgressAggregator(z.ZEGRID_PLAN)
    before = agg.on_stage("zegrid:setup", 50, 100).global_percent
    # An unknown stage with a local 100% must not move global progress.
    r = agg.on_stage("mystery:phase", 100, 100)
    assert r.known is False
    assert r.global_percent == before


def test_reset_clears_run_state():
    agg, _ = _full_zegrid_sequence()
    agg.mark_success()
    assert agg.global_percent == 100.0
    agg.reset()
    assert agg.global_percent == 0.0
    assert agg.terminal is None
    assert agg.current_stage_id is None
    assert agg._fractions == {}


def test_duplicate_and_out_of_order_events_do_not_regress():
    agg = z.ProgressAggregator(z.ZEGRID_PLAN)
    agg.on_stage("zegrid:setup", 80, 100)
    high = agg.on_stage("zegrid:setup", 100, 100).global_percent
    # Duplicate of the final event, then an out-of-order LOWER value: no regress.
    dup = agg.on_stage("zegrid:setup", 100, 100).global_percent
    ooo = agg.on_stage("zegrid:setup", 20, 100).global_percent
    assert dup == high
    assert ooo == high
    assert agg._fractions["zegrid:setup"] == 1.0


def test_no_measurable_total_keeps_stage_floor():
    agg = z.ProgressAggregator(z.ZEGRID_PLAN)
    agg.on_stage("zegrid:setup", 100, 100)  # setup complete -> floor 5.0
    # layout with no measurable total: global stays at the floor.
    r = agg.on_stage("zegrid:layout", 0, 0)
    assert r.known is True
    assert r.global_percent == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Legacy + SDS event sequences
# ---------------------------------------------------------------------------

def test_legacy_sequence_monotonic_and_success_only_100():
    agg = z.ProgressAggregator(z.LEGACY_PLAN)
    percents = []
    for sid, cur, tot in (
        ("phase1_scan", 100, 100),
        ("phase2_cluster", 10, 10),
        ("phase3_master_tiles", 50, 50),
        ("phase4_grid", 20, 20),
        ("phase5_intertile", 30, 30),
        ("phase6", 10, 10),
        ("phase7", 5, 5),
    ):
        percents.append(agg.on_stage(sid, cur, tot).global_percent)
    assert percents == sorted(percents)
    assert percents[-1] < 100.0  # capped until explicit success
    agg.mark_success()
    assert agg.global_percent == 100.0


def test_sds_sequence_monotonic_and_success_only_100():
    agg = z.ProgressAggregator(z.SDS_PLAN)
    percents = []
    for n in range(1, 8):
        percents.append(agg.on_stage(f"sds_phase_{n}", 1, 1).global_percent)
    assert percents == sorted(percents)
    assert percents[-1] < 100.0
    agg.mark_success()
    assert agg.global_percent == 100.0


# ---------------------------------------------------------------------------
# PhaseReporter emits stable id with separate current/total
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.emitted = []
        self.stages = []

    def emit(self, msg, lvl="INFO"):
        self.emitted.append((msg, lvl))

    def stage(self, stage_str, current, total):
        self.stages.append((stage_str, int(current), int(total)))


def test_phase_reporter_emits_stable_id_with_separate_counters():
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, stage=r.stage)
    rep.start("gauge", total=1305, unit="frame-ops")
    rep.progress(113, item_id="pairs:d56")
    rep.end()
    assert r.stages
    # stable id, counters in separate fields, no embedded counter.
    assert ("zegrid:gauge", 0, 1305) in r.stages
    assert ("zegrid:gauge", 113, 1305) in r.stages
    assert ("zegrid:gauge", 1305, 1305) in r.stages
    for stage_str, _c, _t in r.stages:
        assert stage_str == "zegrid:gauge"
        assert "/" not in stage_str


# ---------------------------------------------------------------------------
# STATIC seam checks (no Qt / no heavy production import)
# ---------------------------------------------------------------------------

def _parse_source(rel: str) -> ast.Module:
    path = SRC / rel
    return ast.parse(path.read_text(encoding="utf-8"))


def test_controller_stage_progress_routes_once_without_local_progress():
    tree = _parse_source("zemosaic_gui_qt.py")
    handler = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_payload":
            handler = node
            break
    assert handler is not None, "_handle_payload not found"
    stage_branch = None
    for stmt in handler.body:
        if isinstance(stmt, ast.If):
            if "STAGE_PROGRESS" in ast.unparse(stmt.test):
                stage_branch = stmt
                break
    assert stage_branch is not None, "STAGE_PROGRESS branch not found"
    branch_src = ast.unparse(stage_branch)
    # The structured stage event is routed once to the aggregator; it must NOT
    # emit a raw local percent as global progress.
    assert "stage_progress.emit" in branch_src
    assert "progress_changed.emit" not in branch_src


def test_finalize_seam_ordering_writes_before_completion():
    tree = _parse_source("zemosaic_zegrid_mode.py")
    run_single = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_single":
            run_single = node
            break
    assert run_single is not None, "_run_single not found"

    marks = {
        "finalize_assign": None,
        "finalize_start": None,
        "finalize_end": None,
        "write_raw_science": None,
        "write_outputs": None,
        "write_run_log": None,
        "success_emit": None,
    }
    for node in ast.walk(run_single):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_finalize_rep":
                    marks["finalize_assign"] = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_finalize_rep"
            ):
                if node.func.attr == "start" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value == "finalize":
                        marks["finalize_start"] = node.lineno
                elif node.func.attr == "end":
                    marks["finalize_end"] = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "_write_raw_science_fits":
                marks["write_raw_science"] = node.lineno
            elif name == "_write_outputs":
                marks["write_outputs"] = node.lineno
            elif name == "_write_run_log":
                marks["write_run_log"] = node.lineno
            elif name == "_emit":
                for kw in node.keywords:
                    if kw.arg == "lvl" and isinstance(kw.value, ast.Constant) and kw.value.value == "SUCCESS":
                        marks["success_emit"] = node.lineno

    for k in marks:
        assert marks[k] is not None, f"marker {k!r} not found in _run_single"

    # start before writes; completion (end) only after ALL writes + run log;
    # the terminal SUCCESS emit comes after the finalize completion.
    assert marks["finalize_assign"] < marks["finalize_start"]
    assert marks["finalize_start"] < marks["write_raw_science"]
    assert marks["finalize_start"] < marks["write_outputs"]
    assert marks["write_outputs"] < marks["write_run_log"]
    assert marks["write_run_log"] < marks["finalize_end"]
    assert marks["finalize_end"] < marks["success_emit"]
