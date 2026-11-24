"""ZM-M33 P0P1 (P0) — the historical RUN refusal, replayed by PRODUCT code.

Mission ``ZM-M33-P0P1-20261124``, phase P0. No product change, no pixels, no
gauge recomputation, no CUDA allocation.

What these tests prove
----------------------
1. the checkout runs the SAME code observed in the Windows runtime slot
   (sha256 of the four files the diagnostic read);
2. the exact external HOST model reproduces the attested
   ``reduction_workspace_bytes`` / ``host_peak_bytes`` for the attested worst
   Cell (N=21801, patch 1053x1062, tile 128, halo 16), and that the peak does
   not depend on the unattested ``equalize_rgb`` token;
3. the product SELECTOR derives ``column_batch=7296`` by itself from a PLAN
   observation derived by inversion — and derives a strictly smaller batch
   from the attested RUN observation;
4. the ACTIVE decision / mapping / executable bundle are the external
   single-owner class with the frozen tile and batch;
5. the LAUNCH gate ADMITS that bundle under the attested RUN observation while
   the very next RUN-wave checkpoint REFUSES it with
   ``R55_REFUSE_HOST_RAM`` / ``saw_disk=False, saw_host_ram=True,
   saw_non_memory=False`` and an EMPTY adjacent-shrink trail;
6. the same checkpoint ACCEPTS the same bundle when the observation covers the
   15 %-margined peak (fitting control), so the refusal is a real capacity
   attribution and not a modelling artifact;
7. a disk shortfall is reported as ``R55_REFUSE_DISK`` (never re-labelled RAM),
   and a latched low-pressure sample governs the boundary conservatively;
8. the existing ``shrink_frozen_column_batch`` helper already finds the next
   admissible batch (7296 -> 6912) under the RUN phase budget, while the frozen
   ``next_shrink`` order has NO batch rung — the exact missing wiring P1 owns;
9. the GPU tile model refusal is reproduced without CUDA.

Everything is metadata arithmetic. The scratch directory receives no FITS, no
``.npy`` and no spool payload.

P1 NOTE (mission rework: implementation-P1). The historical observation used to
produce a REFUSAL at the RUN wave. P1 wires the existing staging column-batch
shrink at that safe boundary, so the SAME fixture now ADMITS a successor
``7296 -> 6912``. The refusal witness is preserved where it still exists
honestly: a hostile observation where NO smaller batch fits
(``no_admissible_column_batch``) and a disk shortfall (never re-labelled).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "zegrid_m33_p0_witness.py"


def _load_tool(path: Path, name: str):
    """Load the harness by path (never mutate sys.path for other tests)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


witness = _load_tool(TOOL_PATH, "zegrid_m33_p0_witness_under_test")

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import resource_envelope as renv  # noqa: E402
from zemosaic.core.zegrid import runtime_control as zrt  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402

ATT = witness.ATTESTED
GIB = 2 ** 30


def _floor() -> int:
    return witness.plan_raw_floor(
        zz, rcontract, renv, int(ATT["host_peak_bytes"])
    )


# ---------------------------------------------------------------------------
# 1. Provenance — same code OBSERVED (never a claim of immutability)
# ---------------------------------------------------------------------------

def test_checkout_reproduces_the_observed_runtime_code():
    provenance = witness.checkout_provenance()
    # The P0 attribution was made on the HEAD baseline; that hash match is what
    # ``all_match`` means. P1 then modified exactly ONE of the four files, and
    # the tool publishes that divergence instead of hiding it.
    assert provenance["all_match"] is True, provenance
    assert provenance["baseline_head"]["diverged_from_attested"] == []
    assert provenance["working_tree"]["diverged_from_attested"] == [
        "src/zemosaic/zemosaic_zegrid_mode.py"
    ]
    assert "not proof of immutability" in provenance["claim"]


# ---------------------------------------------------------------------------
# 2. Exact external HOST pricing of the attested worst Cell
# ---------------------------------------------------------------------------

def test_external_host_model_reproduces_the_attested_bytes():
    priced = witness.pricing_probe()
    for token, block in priced.items():
        assert block["host_peak_bytes"] == ATT["host_peak_bytes"], token
        assert (
            block["reduction_workspace_bytes"]
            == ATT["reduction_workspace_bytes"]
        ), token
        # The reduction phase is BOTH the peak and the batch workspace: the
        # refusal is the column-batch term, not the persistent planes.
        assert block["phases"]["reduce"] == ATT["host_peak_bytes"]
        assert block["batch_total_bytes"] == ATT["reduction_workspace_bytes"]
        # Attested effective lot: 19 rows x 128 columns x 3 channels.
        assert block["batch_positions"] == ATT["column_batch"]
        assert block["rows_per_batch"] == 19
        assert block["cols_per_batch"] == 128


def test_host_peak_is_invariant_to_the_unattested_equalize_token():
    priced = witness.pricing_probe(equalize_rgb_values=(False, True))
    peaks = {token: block["host_peak_bytes"] for token, block in priced.items()}
    assert len(set(peaks.values())) == 1, peaks
    # Only the (smaller) finalize phase moves, which cannot be the refusal term.
    assert (
        priced["equalize_rgb=True"]["phases"]["finalize"]
        > priced["equalize_rgb=False"]["phases"]["finalize"]
    )
    assert priced["equalize_rgb=True"]["phases"]["reduce"] == ATT["host_peak_bytes"]


# ---------------------------------------------------------------------------
# 3. The product selector derives the attested batch (no hardcoded 7296)
# ---------------------------------------------------------------------------

def test_selector_derives_the_attested_batch_from_the_plan_observation():
    floor = _floor()
    assert rcontract.plan_budget_bytes(floor) == witness.margin_applied_bytes(
        renv, ATT["host_peak_bytes"]
    )
    sweep = witness.selector_sweep(raws=[floor - 1, floor])
    below, at_floor = sweep
    assert below["refused"] is False
    assert below["column_batch"] < ATT["column_batch"], below["column_batch"]
    assert at_floor["refused"] is False
    assert at_floor["column_batch"] == ATT["column_batch"]
    assert at_floor["host_peak_bytes"] == ATT["host_peak_bytes"]
    assert at_floor["effective_column_batch"] == ATT["column_batch"]
    # The RUN observation is BELOW the derived PLAN floor: the frozen contract
    # could not have been re-derived at RUN time.
    assert ATT["run_raw_available_bytes"] < floor


def test_selector_at_the_run_observation_would_not_derive_7296():
    run_raw = int(ATT["run_raw_available_bytes"])
    item = witness.selector_sweep(raws=[run_raw])[0]
    assert item["refused"] is False
    assert item["plan_budget_bytes"] == ATT["run_budget_bytes"]
    assert item["column_batch"] < ATT["column_batch"]
    assert item["host_peak_bytes"] < ATT["host_peak_bytes"]
    assert item["executed_rows_per_batch"] < 19


# ---------------------------------------------------------------------------
# 4. ACTIVE decision / mapping / executable bundle are the frozen external one
# ---------------------------------------------------------------------------

def test_active_decision_and_applied_bundle_are_the_frozen_external_class(tmp_path):
    evidence = witness.replay(scratch=tmp_path / "replay")
    decision = evidence["plan_decision"]
    assert decision["working_set_class"] == zz.WORKING_SET_EXTERNAL
    assert decision["phase"] == zsched.PHASE_CELL_STACK_STREAM
    assert decision["candidate"] == {
        "backend": "cpu", "tile_px": ATT["tile_px"],
        "units_in_flight": 1, "prefetch_units": 0, "workers_per_unit": 1,
    }
    assert decision["mapping"]["cell_task_mode"] == "external"
    assert decision["mapping"]["external_single_owner"] is True
    assert decision["mapping"]["cpu_streaming_tile"] == ATT["tile_px"]
    assert decision["mapping"]["prefetch_units"] == 0
    applied = evidence["applied_bundle"]
    assert applied["aggregate_bytes"] == ATT["host_peak_bytes"]
    assert applied["stack_total_bytes"] == ATT["host_peak_bytes"]
    assert applied["cache_total_bytes"] == 0
    # The real task really carries the frozen contract (bound + batch).
    assert applied["external_tasks"] == 1
    assert applied["external_task_bound_bytes"] == ATT["host_peak_bytes"]
    assert applied["external_task_contract_batch"] == ATT["column_batch"]
    assert evidence["frozen_contract"]["class"] == zz.WORKING_SET_EXTERNAL


# ---------------------------------------------------------------------------
# 5. LAUNCH admits, the FIRST RUN-wave checkpoint refuses
# ---------------------------------------------------------------------------

def test_launch_admits_and_the_run_wave_checkpoint_shrinks_the_batch(tmp_path):
    evidence = witness.replay(scratch=tmp_path / "replay")
    launch = evidence["launch_admission"]
    assert launch["admitted"] is True
    assert launch["plan_aggregate_bytes"] == ATT["host_peak_bytes"]
    for observation in launch["admission_observations"]:
        assert observation["raw_available_bytes"] == ATT["run_raw_available_bytes"]
        assert observation["budget_bytes"] == ATT["run_budget_bytes"]

    checkpoint = evidence["run_checkpoint"]
    assert checkpoint["status"] == "accepted", checkpoint
    record = checkpoint["record"]
    assert record["status"] == "accepted"
    assert record["changed"] is True
    # The RUN observation still governs the boundary decision.
    assert record["host"]["raw_available_bytes"] == ATT["run_raw_available_bytes"]
    assert record["host"]["budget_bytes"] == ATT["run_budget_bytes"]
    assert record["host_source"] == "fresh"
    assert record["disk"]["free_bytes"] == ATT["run_disk_free_bytes"]
    assert record["candidate"]["tile_px"] == ATT["tile_px"]

    # P1: the external staging batch is shrunk under the SAME single margin.
    batch = checkpoint["batch_shrink"]
    assert batch is not None
    assert (batch["from_batch"], batch["to_batch"]) == (7296, 6912)
    assert batch["required_peak_bytes"] == ATT["host_peak_bytes"]
    assert batch["admitted_peak_bytes"] == 14559855634
    assert batch["phase_budget_bytes"] == zz._r55_margin_raw_ceiling(
        ATT["run_budget_bytes"]
    )
    assert batch["required_peak_bytes"] > batch["phase_budget_bytes"] >= (
        batch["admitted_peak_bytes"]
    )
    # The executable task really carries the successor contract.
    after = checkpoint["bundle_after"]
    assert after["column_batch"] == 6912
    assert after["host_peak_bytes"] == 14559855634
    assert after["external_task_batch"] == 6912
    assert after["external_task_bound_bytes"] == 14559855634
    assert after["envelope_budget_bytes"] == ATT["run_budget_bytes"]
    assert after["envelope_aggregate_bytes"] == 14559855634


# ---------------------------------------------------------------------------
# 6. Fitting control: the SAME bundle is accepted at the PLAN floor
# ---------------------------------------------------------------------------

def test_same_bundle_is_accepted_when_the_observation_covers_the_margin():
    floor = _floor()
    evidence = witness.replay(run_raw_bytes=floor)
    checkpoint = evidence["run_checkpoint"]
    assert checkpoint["status"] == "accepted", checkpoint
    assert checkpoint["record"]["changed"] is False
    assert checkpoint["record"]["shrink_steps"] == []
    assert checkpoint["record"]["host"]["raw_available_bytes"] == floor
    assert checkpoint["record"]["host"]["budget_bytes"] == (
        witness.margin_applied_bytes(renv, ATT["host_peak_bytes"])
    )
    # And one byte less raw is still enough only up to the exact inversion.
    assert evidence["launch_admission"]["admitted"] is True


# ---------------------------------------------------------------------------
# 7. Disk shortfall and the governing latched sample
# ---------------------------------------------------------------------------

def test_disk_shortfall_is_refused_as_disk_not_ram():
    evidence = witness.replay(run_disk_free_bytes=1 * GIB, include_launch=False)
    refusal = evidence["run_checkpoint"]
    assert refusal["status"] == "refused"
    assert refusal["reason_code"] == "R55_REFUSE_DISK", refusal
    assert refusal["classification"]["saw_disk"] is True
    assert refusal["classification"]["saw_host_ram"] is False


def test_latched_pressure_governs_the_boundary_conservatively():
    latched = int(ATT["run_raw_available_bytes"]) - 400 * 2 ** 20
    evidence = witness.replay(latched_raw_bytes=latched, include_launch=False)
    checkpoint = evidence["run_checkpoint"]
    record = checkpoint["record"]
    assert checkpoint["status"] == "accepted"
    assert record["host_source"] == "fresh+latched"
    assert record["host"]["raw_available_bytes"] == latched
    assert record["host"]["budget_bytes"] == zrt.run_budget_bytes(latched)
    assert record["fresh_host"]["raw_available_bytes"] == (
        ATT["run_raw_available_bytes"]
    )
    # The latched (lower) observation governs the successor rung.
    batch = checkpoint["batch_shrink"]
    assert batch["phase_budget_bytes"] == zz._r55_margin_raw_ceiling(
        zrt.run_budget_bytes(latched)
    )
    assert batch["to_batch"] < ATT["column_batch"]
    assert batch["admitted_peak_bytes"] <= batch["phase_budget_bytes"]


# ---------------------------------------------------------------------------
# 8. The missing rung: the existing batch-shrink helper vs next_shrink
# ---------------------------------------------------------------------------

def test_existing_batch_shrink_helper_finds_the_next_admissible_batch():
    phase_budget = zz._r55_margin_raw_ceiling(ATT["run_budget_bytes"])
    probe = witness.frozen_batch_shrink_probe(
        phase_budget_bytes=phase_budget
    )
    shrink = probe["shrink"]
    assert shrink is not None, probe
    assert shrink["from_batch"] == ATT["column_batch"]
    assert shrink["to_batch"] < ATT["column_batch"]
    assert shrink["admitted_peak_bytes"] <= phase_budget
    assert shrink["required_peak_bytes"] == ATT["host_peak_bytes"]
    assert shrink["admitted_peak_bytes"] < ATT["host_peak_bytes"]
    # The admitted lot is a real staging lot, not a science change.
    assert shrink["effective_column_batch"] == shrink["to_batch"]


def test_scheduler_shrink_order_is_untouched_and_the_batch_rung_is_additive():
    """P1 adds ONE checkpoint rung; the frozen scheduler order never changes."""
    assert tuple(zsched.SHRINK_ORDER) == (
        "prefetch_units", "workers_per_unit", "units_in_flight", "tile_px",
        "backend",
    )
    assert "column_batch" not in tuple(zsched.SHRINK_ORDER)
    evidence = witness.replay()
    record = evidence["run_checkpoint"]["record"]
    assert [entry["step"] for entry in record["shrink_steps"]] == [
        "column_batch"
    ]
    assert record["candidate"] == {
        "backend": "cpu", "tile_px": ATT["tile_px"],
        "units_in_flight": 1, "prefetch_units": 0, "workers_per_unit": 1,
    }


# ---------------------------------------------------------------------------
# 9. GPU tile model refusal (no CUDA, no allocation)
# ---------------------------------------------------------------------------

def test_gpu_tile_model_refusal_is_reproduced_without_cuda():
    probe = witness.vram_probe()
    assert probe["min_tile_px"] == ATT["gpu_min_tile_px"]
    assert probe["taper_px"] == 18
    assert probe["workspace_at_min_tile_bytes"] == ATT["gpu_t32_workspace_bytes"]
    assert probe["workspace_at_min_tile_bytes"] > ATT["vram_budget_bytes"]
    assert probe["declared_max_tile_px"] == ATT["gpu_declared_max_tile_px"]
    assert probe["declared_max_tile_px"] < probe["min_tile_px"]


# ---------------------------------------------------------------------------
# 10. No pixels, no payload, no CUDA
# ---------------------------------------------------------------------------

def test_replay_touches_no_pixels_and_writes_no_payload(tmp_path):
    scratch = tmp_path / "replay"
    modules_before = set(sys.modules)
    evidence = witness.replay(scratch=scratch)
    assert evidence["status"] == "OK"
    files = [
        p for p in scratch.rglob("*")
        if p.is_file()
    ]
    assert all(
        p.suffix.lower() not in {".fits", ".fit", ".fz", ".npy", ".npz"}
        for p in files
    ), files
    spool = scratch / "external-spool"
    assert spool.is_dir()
    assert not list(spool.iterdir())
    # The replay must not pull a GPU stack in (CPU import laziness).
    new_modules = {
        name for name in set(sys.modules) - modules_before
        if name.split(".")[0] in {"cupy", "cuda", "cupyx"}
    }
    assert new_modules == set(), new_modules


def test_evidence_is_json_serialisable(tmp_path):
    evidence = witness.replay(scratch=tmp_path / "replay")
    payload = json.dumps(evidence, sort_keys=True)
    assert str(ATT["host_peak_bytes"]) in payload
    assert "batch_shrink" in payload
    control = witness.replay(
        run_raw_bytes=120_000_000,
        scratch=tmp_path / "control",
        include_launch=False,
    )
    assert control["run_checkpoint"]["reason_code"] == "R55_REFUSE_HOST_RAM"
    assert control["run_checkpoint"]["classification"][
        "external_column_batch"
    ] == "no_admissible_column_batch"


# ---------------------------------------------------------------------------
# 11. The DOCUMENTED invocation must load THIS checkout (no PYTHONPATH)
# ---------------------------------------------------------------------------

def _clean_env() -> dict:
    """Environment with no inherited PYTHONPATH (the documented invocation)."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def test_documented_command_runs_from_the_worktree_root(tmp_path):
    """``python tools/zegrid_m33_p0_witness.py --out ...`` proves provenance.

    The qualification venv is an editable install of another tree; without the
    checkout guard this command imported the FOREIGN product code and died with
    ``ImportError: cannot import name 'external_stream'``. The guard must pin
    this checkout, remove the foreign entry, and say so in the evidence.
    """
    out = tmp_path / "evidence.json"
    proc = subprocess.run(
        [sys.executable, "tools/zegrid_m33_p0_witness.py", "--out", str(out)],
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    src = (REPO_ROOT / "src").resolve()
    code_source = payload["provenance"]["code_source"]
    assert code_source["locked"] is True
    assert Path(code_source["src"]).resolve() == src
    assert Path(code_source["zemosaic_file"]).resolve() == (
        src / "zemosaic" / "__init__.py"
    )
    assert any(
        "zemosaic/src" in entry
        for entry in code_source["foreign_sys_path_entries_removed"]
    ), code_source
    assert payload["provenance"]["all_match"] is True
    assert payload["status"] == "OK"
    checkpoint = payload["run_checkpoint"]
    assert checkpoint["status"] == "accepted"
    assert checkpoint["batch_shrink"]["to_batch"] < (
        checkpoint["batch_shrink"]["from_batch"]
    )
    assert checkpoint["bundle_after"]["external_task_batch"] == (
        checkpoint["batch_shrink"]["to_batch"]
    )


def test_tool_refuses_a_tree_without_the_local_package(tmp_path):
    """Fail closed BEFORE any evidence when the checkout cannot be pinned."""
    tree = tmp_path / "tree"
    (tree / "tools").mkdir(parents=True)
    shutil.copy2(TOOL_PATH, tree / "tools" / "zegrid_m33_p0_witness.py")
    out = tree / "evidence.json"
    proc = subprocess.run(
        [sys.executable, "tools/zegrid_m33_p0_witness.py", "--out", str(out)],
        cwd=str(tree),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 4, (proc.returncode, proc.stdout, proc.stderr)
    assert "REFUSED" in proc.stderr
    assert "no local package" in proc.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# 12. CLI witnesses: RUN-only disk refusal, and a PARTIAL (PLAN_REFUSED) run
# ---------------------------------------------------------------------------

def _run_cli(args, tmp_path):
    out = tmp_path / "evidence.json"
    proc = subprocess.run(
        [sys.executable, "tools/zegrid_m33_p0_witness.py", *args,
         "--out", str(out)],
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc, out


def test_cli_run_disk_free_reproduces_the_disk_refusal(tmp_path):
    """``--run-disk-free`` touches ONLY the RUN boundary (PLAN stays attested)."""
    proc, out = _run_cli(["--run-disk-free", str(1 << 30)], tmp_path)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "OK"
    assert payload["observations_injected"]["run_disk_free_bytes"] == 1 << 30
    assert payload["observations_injected"]["disk_free_bytes"] == (
        ATT["run_disk_free_bytes"]
    )
    # The PLAN observation is untouched: the selector still derives 7296.
    assert payload["plan_selection"]["column_batch"] == ATT["column_batch"]
    assert payload["plan_selection"]["plan_raw_available_bytes"] == _floor()
    refusal = payload["run_checkpoint"]
    assert refusal["status"] == "refused"
    assert refusal["reason_code"] == "R55_REFUSE_DISK"
    assert refusal["classification"]["saw_disk"] is True
    assert refusal["classification"]["saw_host_ram"] is False
    assert payload["run_facts"]["disk"]["free_bytes"] == 1 << 30


def test_cli_plan_disk_refusal_writes_partial_evidence(tmp_path):
    """A PLAN refusal must exit 2, write partial evidence and never traceback."""
    proc, out = _run_cli(["--disk-free", str(1 << 30)], tmp_path)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "Traceback" not in proc.stderr
    assert "KeyError" not in proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "PLAN_REFUSED"
    assert payload["plan_selection"]["refused"] is True
    assert payload["plan_selection"]["reason"] == "disk_envelope_unavailable"
    # Absent blocks are DECLARED, never fabricated.
    assert "run_checkpoint" not in payload
    assert "pricing_check" not in payload
    assert "run_checkpoint" in payload["partial"]["absent_blocks"]
    assert "run_checkpoint" not in payload["partial"]["present_blocks"]
    # The probes that do not depend on the replay still ran.
    for block in (
        "pricing_probe", "channels_probe", "vram_probe", "selector_sweep",
        "frozen_batch_shrink_probe",
    ):
        assert block in payload
    assert payload["partial"]["note"]


# ---------------------------------------------------------------------------
# 13. ``channels`` is a labelled hypothesis, not a silent constant
# ---------------------------------------------------------------------------

def test_channels_are_a_labelled_reconstructed_hypothesis(tmp_path):
    probe = witness.channels_probe()
    assert probe["channels_used"] == ATT["channels"] == 3
    assert probe["matching_channel_counts"] == [3]
    assert probe["unique"] is True
    assert probe["candidates"]["3"]["host_peak_bytes"] == ATT["host_peak_bytes"]
    assert probe["candidates"]["1"]["host_peak_bytes"] != ATT["host_peak_bytes"]
    assert "RECONSTRUCTED" in probe["status"].upper()
    evidence = witness.replay(scratch=tmp_path / "replay")
    channels = evidence["hypotheses"]["channels"]
    assert channels["value"] == 3
    assert "RECONSTRUCTED" in channels["status"]
    assert evidence["hypotheses"]["plan_observation"]["status"].startswith(
        "RECONSTRUCTED"
    )
    assert evidence["hypotheses"]["non_empty_cells"]["status"] == "NOT REPLAYED"


# ---------------------------------------------------------------------------
# 14. The P1 plan itself must keep the F1/F2 contracts (docs guard)
# ---------------------------------------------------------------------------

def test_p1_plan_states_the_f1_f2_contracts():
    """A regression that deletes the amendments must fail loudly.

    P1 is not implemented yet, so the only executable guarantee available is
    that the plan keeps the two blocking contracts, the pending-task remap
    algorithm and the decided option A.
    """
    plan = (
        REPO_ROOT / "docs" / "refactor" / "M33_P0P1_P1_PLAN_20261124.md"
    ).read_text(encoding="utf-8")
    # F1 — successor contract attached to a NEW bundle, initial bundle forbidden.
    assert "_r55_active_apply" in plan
    assert "NOUVEAU bundle" in plan
    assert "**interdit**" in plan
    assert "mode.py:14371" in plan          # apply() reads the contract itself
    assert "mode.py:6643-6650" in plan      # executor reads slot 7
    # F2 — pending task rebuild after ANY bundle replacement.
    assert "pending_tasks" in plan
    assert "total_tasks" in plan
    assert "fallback dense→bounded" in plan or "dense→bounded" in plan
    assert "mode.py:21911" in plan          # recovery returns the new tasks
    assert "Précondition bloquante (F2)" in plan
    # Decided architect option.
    assert "TRANCHEE : option A" in plan
