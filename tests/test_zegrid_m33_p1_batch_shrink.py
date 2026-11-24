"""ZM-M33 P0P1 (P1) — external RUN column-batch shrink, without pixels.

Covers the block §6 of ``docs/refactor/M33_P0P1_P1_PLAN_20261124.md``:

* F1 — the successor contract is applied EXPLICITLY; the initial bundle is never
  applied unchanged, so the generated tasks carry the new batch;
* F2 — after ANY bundle replacement both task lists are rebuilt by ``cell_id``
  (canonical order, same set, same pending subset), with fail-closed guards;
* the historical M33 fixture now ADMITS a successor 7296 -> 6912 with
  ``admitted_peak 14 559 855 634 <= phase_budget 14 823 796 557``;
* a second pressure step is contiguous (6912 -> 6528);
* the governed fresh+latched minimum decides;
* a disk refusal is never re-labelled and never triggers a batch transition;
* "already fits" and "no smaller batch fits" stay distinguishable;
* the disk envelope / plan hash / working-set block stay coherent.

No pixels, no FITS, no gauge, no CUDA: the geometry/metadata fixture is the
same reconstructed one as P0 (``tools/zegrid_m33_p0_witness.py``).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "zegrid_m33_p0_witness.py"


def _load_tool(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


witness = _load_tool(TOOL_PATH, "zegrid_m33_p0_witness_p1_under_test")

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import external_stream as zext  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import resource_envelope as renv  # noqa: E402
from zemosaic.core.zegrid import runtime_control as zrt  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402

ATT = witness.ATTESTED
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# fixture helpers (real product bundle + decision, attested geometry)
# ---------------------------------------------------------------------------

def _plan_raw() -> int:
    return witness.plan_raw_floor(
        zz, rcontract, renv, int(ATT["host_peak_bytes"])
    )


def _checkpoint(*, run_raw=None, run_disk_free=None, latched=None,
                scratch=None, plan_raw=None, decision=None, bundle=None):
    """Run the REAL checkpoint on the reconstructed M33 fixture."""
    plan_raw = _plan_raw() if plan_raw is None else int(plan_raw)
    evidence = witness.replay(
        plan_raw_bytes=plan_raw,
        run_raw_bytes=run_raw,
        run_disk_free_bytes=run_disk_free,
        latched_raw_bytes=latched,
        include_launch=False,
        scratch=scratch,
    )
    return evidence


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    """The reconstructed fixture: contract + bundle + ACTIVE decision."""
    root = tmp_path_factory.mktemp("m33_p1_fixture")
    plan_raw = _plan_raw()
    with witness.injected_observations(
        raw_available_bytes=plan_raw,
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
    ):
        plan = witness._plan_phase(
            plan_raw_bytes=plan_raw,
            disk_free_bytes=int(ATT["run_disk_free_bytes"]),
            root=root,
        )
    assert plan["refused"] is False
    return plan


def _facts(raw: int, *, disk_free: int | None = None, latched: int | None = None):
    disk = (
        int(ATT["run_disk_free_bytes"]) if disk_free is None else int(disk_free)
    )
    facts = zrt.collect_run_facts(
        root=Path("/tmp"),
        available_fn=lambda: int(raw),
        include_diagnostics=False,
    )
    facts["disk"]["free_bytes"] = int(disk)
    latched_fact = None
    if latched is not None:
        latched_fact = {
            "raw_available_bytes": int(latched),
            "budget_bytes": int(zrt.run_budget_bytes(int(latched))),
            "monotonic_ns": 0,
        }
    return facts, latched_fact


def _phase_budget(raw: int) -> int:
    return int(zz._r55_margin_raw_ceiling(int(zrt.run_budget_bytes(int(raw)))))


# ---------------------------------------------------------------------------
# 1. The successor helper: exact, shrink-only, distinguishable
# ---------------------------------------------------------------------------

def test_successor_reproduces_the_m33_transition(fixture):
    contract = fixture["contract"]
    outcome = zz._r55_external_batch_successor(
        contract=contract,
        phase_budget_bytes=_phase_budget(ATT["run_raw_available_bytes"]),
    )
    assert outcome["already_fits"] is False
    assert outcome["admissible"] is True
    assert outcome["required_peak_bytes"] == ATT["host_peak_bytes"] == 15363724306
    assert outcome["admitted_peak_bytes"] == 14559855634
    assert outcome["phase_budget_bytes"] == 14823796557
    transition = outcome["transition"]
    assert (transition["from_batch"], transition["to_batch"]) == (7296, 6912)
    assert transition["frozen_batch"] == 7296
    assert transition["admitted_peak_bytes"] <= transition["phase_budget_bytes"]
    assert transition["required_peak_bytes"] > transition["phase_budget_bytes"]
    successor = outcome["contract"]
    assert successor["column_batch"] == 6912
    assert successor["effective_column_batch"] == 6912
    assert successor["column_batch_clamped"] is False
    assert successor["host_peak_bytes"] == 14559855634
    assert successor["reduction_workspace_bytes"] == 14469636096
    # disk envelope is batch-INVARIANT: same spool/resident/result/margin terms
    assert (
        successor["disk_envelope_bytes"] == contract["disk_envelope_bytes"]
    )


def test_successor_distinguishes_already_fits_from_no_smaller_batch(fixture):
    contract = fixture["contract"]
    fits = zz._r55_external_batch_successor(
        contract=contract, phase_budget_bytes=int(contract["host_peak_bytes"])
    )
    assert fits == {
        "already_fits": True,
        "admissible": False,
        "required_peak_bytes": int(contract["host_peak_bytes"]),
        "phase_budget_bytes": int(contract["host_peak_bytes"]),
    }
    # A hostile phase budget below the fixed floor: no batch can fit at all.
    hostile = 1000
    none_fits = zz._r55_external_batch_successor(
        contract=contract, phase_budget_bytes=hostile
    )
    assert none_fits["already_fits"] is False
    assert none_fits["admissible"] is False
    assert none_fits["reason_code"] == "no_admissible_column_batch"
    assert none_fits["phase_budget_bytes"] == hostile


def test_successor_refuses_an_inconsistent_contract(fixture):
    contract = dict(fixture["contract"])
    contract["host_peak_bytes"] = int(contract["host_peak_bytes"]) + 1
    with pytest.raises(zsched.SchedulerConfigError):
        zz._r55_external_batch_successor(
            contract=contract, phase_budget_bytes=14823796557
        )


# ---------------------------------------------------------------------------
# 2. The checkpoint admits the successor (historical P0 fixture)
# ---------------------------------------------------------------------------

def test_run_checkpoint_admits_the_successor_instead_of_refusing(tmp_path):
    evidence = _checkpoint(scratch=tmp_path / "replay")
    assert evidence["status"] == "OK"
    checkpoint = evidence["run_checkpoint"]
    assert checkpoint["status"] == "accepted", checkpoint
    record = checkpoint["record"]
    assert record["changed"] is True
    batch = record["batch_shrink"]
    assert (batch["from_batch"], batch["to_batch"]) == (7296, 6912)
    assert batch["admitted_peak_bytes"] == 14559855634
    assert batch["phase_budget_bytes"] == 14823796557
    assert record["phase_budget_bytes"] == 14823796557
    # The scheduler step trail records the batch rung, with the SAME tuple.
    steps = record["shrink_steps"]
    assert [entry["step"] for entry in steps] == ["column_batch"]
    assert steps[0]["accepted"] is True
    assert steps[0]["from"] == steps[0]["to"] == {
        "backend": "cpu", "tile_px": 128, "units_in_flight": 1,
        "prefetch_units": 0, "workers_per_unit": 1,
    }
    # Nothing about the class/tile/backend/K changed.
    assert checkpoint["candidate"]["tile_px"] == 128
    assert checkpoint["candidate"]["units_in_flight"] == 1
    assert checkpoint["mapping"]["cell_task_mode"] == "external"


def test_checkpoint_successor_bundle_tasks_carry_the_new_batch(tmp_path):
    """F1 gate: the generated task (slot 7) and its bound are the successor."""
    root = tmp_path / "fixture"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    facts, _latched = _facts(int(ATT["run_raw_available_bytes"]))
    out = zz._r55_run_checkpoint(
        decision=plan["decision"],
        candidate=plan["decision"].plan_set.chosen,
        mapping=plan["decision"].mapping,
        bundle=plan["bundle"],
        remaining_units=56,
        wave_index=0,
        facts=facts,
    )
    bundle = out["bundle"]
    contract = zz._r55_external_contract(bundle)
    assert contract["column_batch"] == 6912
    assert contract["host_peak_bytes"] == 14559855634
    external_tasks = [
        task for task in bundle.cell_tasks
        if len(task) > 7 and isinstance(task[7], dict)
        and task[7].get("class") == zz.WORKING_SET_EXTERNAL
    ]
    assert len(external_tasks) == 1
    task = external_tasks[0]
    assert task[7]["column_batch"] == 6912
    assert task[7]["effective_column_batch"] == 6912
    assert int(task[5]) == 14559855634
    # No task carrying the frozen batch survives.
    assert all(
        not (len(t) > 7 and isinstance(t[7], dict) and t[7].get("column_batch") == 7296)
        for t in bundle.cell_tasks
    )
    # The envelope is rebound to the governed RUN observation.
    assert int(bundle.envelope.budget_bytes) == int(ATT["run_budget_bytes"])
    assert int(bundle.envelope.aggregate_bytes) == 14559855634


def test_applying_the_unchanged_initial_bundle_would_keep_the_old_batch(fixture):
    """The F1 defect, made executable: the seam is what fixes it."""
    contract = fixture["contract"]
    successor = zz._r55_external_batch_successor(
        contract=contract, phase_budget_bytes=14823796557
    )["contract"]
    decision = fixture["decision"]
    unchanged = zz._r55_active_apply(
        fixture["bundle"], mapping=decision.mapping, facts=decision.facts
    )
    assert zz._r55_external_contract(unchanged)["column_batch"] == 7296
    applied = zz._r55_active_apply(
        fixture["bundle"],
        mapping=decision.mapping,
        facts=decision.facts,
        external_contract=successor,
    )
    assert zz._r55_external_contract(applied)["column_batch"] == 6912
    assert int(applied.envelope.aggregate_bytes) == 14559855634


# ---------------------------------------------------------------------------
# 3. Two consecutive shrinks / pressure governance
# ---------------------------------------------------------------------------

def test_second_pressure_step_is_contiguous(tmp_path):
    root = tmp_path / "fixture2"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    decision = plan["decision"]
    facts1, _ = _facts(int(ATT["run_raw_available_bytes"]))
    first = zz._r55_run_checkpoint(
        decision=decision,
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        bundle=plan["bundle"],
        remaining_units=56,
        wave_index=0,
        facts=facts1,
    )
    assert first["record"]["batch_shrink"]["to_batch"] == 6912

    facts2, _ = _facts(18_000_000_000)
    second = zz._r55_run_checkpoint(
        decision=decision,
        candidate=first["candidate"],
        mapping=first["mapping"],
        bundle=first["bundle"],
        remaining_units=55,
        wave_index=1,
        facts=facts2,
    )
    batch2 = second["record"]["batch_shrink"]
    assert batch2["from_batch"] == 6912
    assert batch2["to_batch"] == 6528
    assert batch2["frozen_batch"] == 6912
    assert (
        batch2["admitted_peak_bytes"]
        <= batch2["phase_budget_bytes"]
        < batch2["required_peak_bytes"]
    )
    task = [
        t for t in second["bundle"].cell_tasks
        if len(t) > 7 and isinstance(t[7], dict)
        and t[7].get("class") == zz.WORKING_SET_EXTERNAL
    ][0]
    assert task[7]["column_batch"] == 6528


def test_latched_minimum_governs_the_transition(tmp_path):
    latched_raw = 18_000_000_000
    facts, latched = _facts(
        int(ATT["run_raw_available_bytes"]), latched=latched_raw
    )
    root = tmp_path / "fixture3"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    decision = plan["decision"]
    out = zz._r55_run_checkpoint(
        decision=decision,
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        bundle=plan["bundle"],
        remaining_units=56,
        wave_index=0,
        facts=facts,
        latched=latched,
    )
    record = out["record"]
    assert record["host_source"] == "fresh+latched"
    assert record["host"]["raw_available_bytes"] == latched_raw
    assert record["host"]["budget_bytes"] == zrt.run_budget_bytes(latched_raw)
    batch = record["batch_shrink"]
    assert batch["phase_budget_bytes"] == _phase_budget(latched_raw)
    assert batch["to_batch"] == 6528          # the latched budget's rung
    assert batch["admitted_peak_bytes"] <= batch["phase_budget_bytes"]


def test_no_pressure_path_is_unchanged_and_transition_free(tmp_path):
    floor = _plan_raw()
    facts, _ = _facts(floor)
    root = tmp_path / "fixture4"
    plan = witness._plan_phase(
        plan_raw_bytes=floor,
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    decision = plan["decision"]
    out = zz._r55_run_checkpoint(
        decision=decision,
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        bundle=plan["bundle"],
        remaining_units=56,
        wave_index=0,
        facts=facts,
    )
    assert out["record"]["status"] == "accepted"
    assert out["record"]["changed"] is False
    assert "batch_shrink" not in out["record"]
    assert out["record"]["shrink_steps"] == []
    assert out["bundle"] is plan["bundle"]      # byte-identical object
    assert zz._r55_external_contract(out["bundle"])["column_batch"] == 7296


# ---------------------------------------------------------------------------
# 4. Disk refusal stays honest; no transition is invented
# ---------------------------------------------------------------------------

def test_disk_shortfall_is_never_relabelled_and_never_shrinks(tmp_path):
    facts, _ = _facts(int(ATT["run_raw_available_bytes"]), disk_free=1 * GIB)
    root = tmp_path / "fixture5"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    decision = plan["decision"]
    with pytest.raises(zz._R55RunRefused) as excinfo:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=plan["bundle"],
            remaining_units=56,
            wave_index=0,
            facts=facts,
        )
    refusal = excinfo.value
    assert refusal.reason_code == renv.REASON_REFUSE_DISK
    assert refusal.record["classification"]["saw_disk"] is True
    assert "batch_shrink" not in refusal.record
    assert "batch_shrink_refusal" not in refusal.record


def test_no_admissible_batch_keeps_the_historical_refusal(tmp_path):
    facts, _ = _facts(120_000_000)
    root = tmp_path / "fixture6"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    decision = plan["decision"]
    with pytest.raises(zz._R55RunRefused) as excinfo:
        zz._r55_run_checkpoint(
            decision=decision,
            candidate=decision.plan_set.chosen,
            mapping=decision.mapping,
            bundle=plan["bundle"],
            remaining_units=56,
            wave_index=0,
            facts=facts,
        )
    refusal = excinfo.value
    assert refusal.reason_code == renv.REASON_REFUSE_HOST_RAM
    record = refusal.record
    assert record["detail"].startswith(
        "no feasible adjacent CPU shrink remains"
    )
    assert record["classification"]["saw_host_ram"] is True
    assert record["classification"]["external_column_batch"] == (
        "no_admissible_column_batch"
    )
    assert record["batch_shrink_refusal"]["already_fits"] is False
    assert record["batch_shrink_refusal"]["reason_code"] == (
        "no_admissible_column_batch"
    )


def test_non_external_bundles_are_untouched(tmp_path):
    """A RAM bundle never reaches the external rung (classification unchanged)."""
    root = tmp_path / "fixture7"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    bundle = plan["bundle"]
    import dataclasses

    ram_bundle = dataclasses.replace(
        bundle,
        external_contract=None,
        stack_working_set=zz.WORKING_SET_RAM_TILE,
    )
    facts, _ = _facts(int(ATT["run_raw_available_bytes"]))
    with pytest.raises(zz._R55RunRefused) as excinfo:
        zz._r55_run_checkpoint(
            decision=plan["decision"],
            candidate=plan["decision"].plan_set.chosen,
            mapping=plan["decision"].mapping,
            bundle=ram_bundle,
            remaining_units=56,
            wave_index=0,
            facts=facts,
        )
    record = excinfo.value.record
    assert "batch_shrink_refusal" not in record
    assert record["classification"]["external_column_batch"] if (
        "external_column_batch" in record["classification"]
    ) else True


# ---------------------------------------------------------------------------
# 5. F2 remap semantics (fail closed, canonical order, sparse pending)
# ---------------------------------------------------------------------------

class _Task:
    def __init__(self, cell_id, batch):
        self.cell = type("C", (), {"cell_id": cell_id})()
        self.payload = batch

    def __getitem__(self, index):
        return [self.cell, None, None, None, "external", 0, "bounded",
                {"column_batch": self.payload}][index]

    def __len__(self):
        return 8


def _tasks(pairs):
    return [_Task(cid, batch) for cid, batch in pairs]


def test_remap_preserves_set_order_and_pending_subset():
    old_all = _tasks([("a", 7296), ("b", 7296), ("c", 7296), ("d", 7296)])
    old_pending = [old_all[0], old_all[2], old_all[3]]
    new_all = _tasks([("d", 6912), ("c", 6912), ("b", 6912), ("a", 6912)])
    out = zz._r55_remap_tasks_after_bundle_replacement(
        old_all_tasks=old_all,
        old_pending_tasks=old_pending,
        new_bundle=type("B", (), {"cell_tasks": new_all})(),
    )
    assert [t[0].cell_id for t in out["all_tasks"]] == ["a", "b", "c", "d"]
    assert [t[0].cell_id for t in out["pending_tasks"]] == ["a", "c", "d"]
    assert [t[7]["column_batch"] for t in out["pending_tasks"]] == [6912] * 3
    assert out["total_tasks"] == 3
    assert out["canonical_cids"] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "new_pairs, message",
    [
        ([("a", 6912), ("b", 6912), ("c", 6912), ("c", 6912)], "duplicated"),
        ([("a", 6912), ("b", 6912), ("c", 6912)], "changed the canonical cell set"),
        (
            [("a", 6912), ("b", 6912), ("c", 6912), ("d", 6912), ("e", 6912)],
            "changed the canonical cell set",
        ),
    ],
)
def test_remap_fails_closed_on_identity_damage(new_pairs, message):
    old_all = _tasks([("a", 7296), ("b", 7296), ("c", 7296), ("d", 7296)])
    with pytest.raises(zz._R55RunRefused) as excinfo:
        zz._r55_remap_tasks_after_bundle_replacement(
            old_all_tasks=old_all,
            old_pending_tasks=old_all[:2],
            new_bundle=type("B", (), {"cell_tasks": _tasks(new_pairs)})(),
        )
    assert message in excinfo.value.detail


def test_remap_detects_a_dropped_pending_cell():
    old_all = _tasks([("a", 7296), ("b", 7296)])
    with pytest.raises(zz._R55RunRefused):
        zz._r55_remap_tasks_after_bundle_replacement(
            old_all_tasks=old_all,
            old_pending_tasks=old_all,
            new_bundle=type("B", (), {"cell_tasks": []})(),
        )


# ---------------------------------------------------------------------------
# 6. Coherence: plan hash + working-set block follow the successor
# ---------------------------------------------------------------------------

def test_working_set_block_and_plan_hash_follow_the_successor(tmp_path):
    facts, _ = _facts(int(ATT["run_raw_available_bytes"]))
    root = tmp_path / "fixture8"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    decision = plan["decision"]
    out = zz._r55_run_checkpoint(
        decision=decision,
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        bundle=plan["bundle"],
        remaining_units=56,
        wave_index=0,
        facts=facts,
    )
    bundle = out["bundle"]
    block = dict(bundle.working_set_block)
    assert block["class"] == zz.WORKING_SET_EXTERNAL
    assert block["column_batch"] == 6912
    assert block["tile_size"] == 128
    durable_like = dict(out["record"]["batch_shrink"])
    bundle.layout.setdefault("working_set", {"class": zz.WORKING_SET_EXTERNAL})
    zz._r55_publish_batch_transition(bundle, durable_like)
    assert bundle.working_set_block["batch_transition"]["to_batch"] == 6912
    assert bundle.layout["working_set"]["column_batch"] == 6912
    # The final plan hash changes with the envelope, and stays valid.
    before = zsched.active_final_plan_hash(
        candidate=decision.plan_set.chosen,
        mapping=decision.mapping,
        identity=decision.plan_set.identity,
        envelope=plan["bundle"].envelope,
    )
    after = zsched.active_final_plan_hash(
        candidate=out["candidate"],
        mapping=out["mapping"],
        identity=decision.plan_set.identity,
        envelope=bundle.envelope,
    )
    assert before != after
    assert len(after) == len(before)


# ---------------------------------------------------------------------------
# 7. Rework-1 regressions: C1 (count once), C4 (layout identity), C5 (sparse)
# ---------------------------------------------------------------------------

def test_one_transition_is_exactly_one_accepted_step(tmp_path):
    """C1: the auditable step exists ONCE and changes no scheduler field."""
    evidence = _checkpoint(scratch=tmp_path / "c1")
    record = evidence["run_checkpoint"]["record"]
    assert record["changed"] is True
    steps = record["shrink_steps"]
    assert len(steps) == 1
    step = steps[0]
    assert step["step"] == "column_batch"
    assert step["accepted"] is True
    assert step.get("changed_kind") == "column_batch_only"
    assert step["from"] == step["to"]           # the tuple really is frozen
    assert step["column_batch"] == {"from": 7296, "to": 6912}
    # The deterministic record carries the transition itself (not a fake step).
    deterministic = evidence["run_checkpoint"].get("record", {})
    assert deterministic["batch_shrink"]["to_batch"] == 6912


def test_successor_gets_its_own_layout_and_the_initial_one_is_untouched(
    tmp_path,
):
    """C4: no shared layout alias; initial bundle keeps the frozen batch."""
    facts, _ = _facts(int(ATT["run_raw_available_bytes"]))
    root = tmp_path / "c4"
    plan = witness._plan_phase(
        plan_raw_bytes=_plan_raw(),
        disk_free_bytes=int(ATT["run_disk_free_bytes"]),
        root=root,
    )
    initial = plan["bundle"]
    assert initial.working_set_block["column_batch"] == 7296
    out = zz._r55_run_checkpoint(
        decision=plan["decision"],
        candidate=plan["decision"].plan_set.chosen,
        mapping=plan["decision"].mapping,
        bundle=initial,
        remaining_units=56,
        wave_index=0,
        facts=facts,
    )
    successor = out["bundle"]
    # Distinct layout objects, and the successor's layout points at ITS block.
    assert successor.layout is not initial.layout
    assert successor.layout["working_set"] is successor.working_set_block
    assert successor.working_set_block["column_batch"] == 6912
    assert successor.external_contract["column_batch"] == 6912
    # Publishing the durable record does NOT need to repair the initial bundle.
    zz._r55_publish_batch_transition(
        successor, dict(out["record"]["batch_shrink"])
    )
    assert successor.working_set_block["batch_transition"]["to_batch"] == 6912
    assert initial.working_set_block["column_batch"] == 7296
    assert initial.layout.get("working_set", {}).get("column_batch") == 7296
    assert initial.external_contract["column_batch"] == 7296
    assert "batch_transition" not in initial.working_set_block


def test_remaining_units_counts_the_sparse_pending_list():
    """C5: the canonical list must never be indexed with a pending offset."""
    class _T:
        def __init__(self, mode):
            self._mode = mode

        def __getitem__(self, index):
            return [None, None, None, None, self._mode][index]

    all_tasks = [
        _T("external"), _T("external"), _T(None), _T("external"),
        _T("external"),
    ]
    # Sparse resume: the empty canonical cell is not in the pending list.
    pending_tasks = [all_tasks[1], all_tasks[3], all_tasks[4]]
    # offset 1 => 2 dispatchable units remain in the PENDING list.
    assert zz._r55_remaining_pending_units(pending_tasks, 1) == 2
    # The historical (wrong) canonical count from a pending offset reports 3.
    assert sum(1 for t in all_tasks[1:] if t[4] is not None) == 3
    assert zz._r55_remaining_pending_units(pending_tasks, 0) == 3
    assert zz._r55_remaining_pending_units(pending_tasks, 99) == 0


def test_resume_selection_reuses_the_durable_reduced_batch(tmp_path):
    """E (reader seam): a resume re-prices the DURABLE batch, never 7296."""
    plan_raw = _plan_raw()
    # A resumed external run at the PLAN observation with frozen_batch=6912:
    # the selector must keep 6912 (no growth, no re-derivation to 7296).
    selected = witness.select_external(
        zz, plan_raw_bytes=plan_raw, tmp_root=tmp_path / "resume",
        frozen_batch=6912,
    )
    assert selected["refused"] is False
    assert selected["column_batch"] == 6912
    assert selected["host_peak_bytes"] == 14559855634
    assert selected["batch_transition"] is None
    # The same frozen batch under the attested RUN observation still fits, so
    # the resume selector keeps it exactly (no growth, no re-derivation).
    selected_run = witness.select_external(
        zz, plan_raw_bytes=int(ATT["run_raw_available_bytes"]),
        tmp_root=tmp_path / "resume_run", frozen_batch=6912,
    )
    assert selected_run["refused"] is False
    assert selected_run["column_batch"] == 6912
    assert selected_run["batch_transition"] is None
    # A LOWER governed observation shrinks it further, never back to 7296.
    selected_low = witness.select_external(
        zz, plan_raw_bytes=18_000_000_000,
        tmp_root=tmp_path / "resume_low", frozen_batch=6912,
    )
    assert selected_low["refused"] is False
    assert selected_low["column_batch"] < 6912
