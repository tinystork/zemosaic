"""ZM-ZEGRID-FROZEN-BATCH-SHRINK — shrink-only frozen ``column_batch`` resume.

A real M33 run (21,818 frames, 171 Cells) froze an ``external_spool`` admission
with ``column_batch=13440`` and 5 committed Cells. Resuming it on the same host
refused::

    R55_REFUSE_HOST_RAM ... frozen column_batch 13440 yields a host peak of
    28193700118 B but only 14648888809 B remain in the phase-local budget ...
    refusing to shrink the frozen identity silently

The batch is a STAGING parameter, not a science parameter (bit-identical
payloads proven on real M106 data at 8/64/512). This mission allows a
SHRINK-ONLY re-derivation at the resume boundary with the SAME exact host
accounting the refusal already uses, recorded explicitly in the durable state,
the manifest working-set block and ONE WARN log line.

What stays true here:

* never grow, never re-class, never re-tile;
* no smaller batch fits -> today's refusal BYTE-IDENTICAL;
* a fitting frozen batch -> kept as-is, NO transition recorded;
* fresh run (no frozen admission) -> unchanged plan;
* only pending Cells execute (the durable sparse set is untouched).

All peak/budget numbers below are PINNED, not just "it runs".
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import external_stream as xstream  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30
H = W = 8
CHANNELS = 3
CORE = (1, 7, 1, 7)
CELL_IDS = ["r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001"]
DTYPE_CONTRACT = {
    "planes": {
        "core.science_core": ["<f4", "<f8"],
        "core.estimator_weight_sum_core": ["<f4", "<f8"],
        "core.support_w1_core": ["<f4", "<f8"],
        "core.support_w2_core": ["<f4", "<f8"],
        "core.n_eff_support_core": ["<f4", "<f8"],
        "core.valid_mask_core": ["|b1"],
        "core.surviving_sample_count_core": ["<i4", "<i8"],
        "patch.science": ["<f4", "<f8"],
        "patch.valid_mask": ["|b1"],
        "patch.surviving": ["<i4", "<i8"],
    }
}

# The geometry used by the selector tests: 160x160 patch, qualified 128 tile,
# 8 px halo, 100 frames, 3 channels.
_GEOM = dict(
    permitted_tiles=(128,),
    worst_patch_hw=(160, 160),
    max_n_frames=100,
    channels=3,
    halo_px=8,
    storage_profile="bounded",
    observation_domain=rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE,
)
_FROZEN_BATCH = 13440
# PINNED exact host accounting at this geometry (see zext.estimate_*).
_FROZEN_PEAK = 139_355_072
_REDUCE_FIXED = 3_449_792
_HOST_FIXED = 5_909_632
# budget_bytes=11_500_000 -> effective phase budget 10_000_000 (15/100 margin).
_RAW_BUDGET = 11_500_000
_PHASE_BUDGET = 10_000_000
_ADMITTED_BATCH = 384
_ADMITTED_PEAK = 7_332_800


# budget_bytes=200_000_000 -> effective phase budget 173_913_043
# (15/100 margin); the fresh-run inversion yields 16512 positions (43 rows).
_FRESH_RAW_BUDGET = 200_000_000
_FRESH_EFFECTIVE_BUDGET = 173_913_043
_FRESH_BATCH = 16_512


def _free(monkeypatch, free_bytes):
    from types import SimpleNamespace

    monkeypatch.setattr(
        zz.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            total=int(free_bytes), used=0, free=int(free_bytes)
        ),
    )


def _select(tmp_path, monkeypatch, *, budget_bytes, frozen_batch):
    _free(monkeypatch, 77 * GIB)
    return zz._select_external_fallback(
        **_GEOM,
        budget_bytes=int(budget_bytes),
        cache_root=str(tmp_path / "cache"),
        frozen_tile=128,
        frozen_batch=int(frozen_batch),
    )


# ---------------------------------------------------------------------------
# 1. Selector: exact shrink ladder and the recorded numbers.
# ---------------------------------------------------------------------------

def test_frozen_batch_shrinks_to_largest_admissible_smaller_batch(
    tmp_path, monkeypatch
):
    decision = _select(
        tmp_path, monkeypatch, budget_bytes=_RAW_BUDGET,
        frozen_batch=_FROZEN_BATCH,
    )
    assert decision["refused"] is False
    assert decision["tile"] == 128
    # The chosen batch is the largest admissible WHOLE-ROW batch below the
    # frozen one (the existing admissible-batch machinery, reused verbatim).
    assert decision["column_batch"] == _ADMITTED_BATCH
    assert decision["host_peak_bytes"] == _ADMITTED_PEAK
    assert decision["phase_budget_bytes"] == _PHASE_BUDGET
    assert decision["host_fixed_bytes"] == _HOST_FIXED

    transition = decision["batch_transition"]
    assert transition["from_batch"] == _FROZEN_BATCH
    assert transition["to_batch"] == _ADMITTED_BATCH
    assert transition["frozen_batch"] == _FROZEN_BATCH
    assert transition["required_peak_bytes"] == _FROZEN_PEAK
    assert transition["admitted_peak_bytes"] == _ADMITTED_PEAK
    assert transition["phase_budget_bytes"] == _PHASE_BUDGET
    assert transition["residual_batch_budget_bytes"] == (
        _PHASE_BUDGET - _REDUCE_FIXED
    )
    assert transition["reason"]
    assert transition["ladder"][0] == {
        "column_batch": _ADMITTED_BATCH,
        "host_peak_bytes": _ADMITTED_PEAK,
        "fits": True,
    }
    # Never grow: the accepted value is strictly smaller.
    assert transition["to_batch"] < transition["from_batch"]


def test_shrink_helper_never_returns_the_frozen_value_or_greater():
    for budget in (
        6_000_000,
        7_000_000,
        8_000_000,
        10_000_000,
        20_000_000,
        100_000_000,
    ):
        result = xstream.shrink_frozen_column_batch(
            frozen_batch=_FROZEN_BATCH,
            phase_budget_bytes=budget,
            n_frames=100,
            height=160,
            width=160,
            channels=3,
            tile_size=128,
            halo_px=8,
        )
        if result is None:
            continue
        assert 1 <= result["to_batch"] < _FROZEN_BATCH
        assert result["admitted_peak_bytes"] <= budget
        assert result["required_peak_bytes"] > budget


def test_fitting_frozen_batch_is_kept_with_no_transition(tmp_path, monkeypatch):
    # Budget above the frozen peak: the frozen batch is honoured verbatim.
    decision = _select(
        tmp_path, monkeypatch, budget_bytes=200_000_000,
        frozen_batch=_FROZEN_BATCH,
    )
    assert decision["refused"] is False
    assert decision["column_batch"] == _FROZEN_BATCH
    assert decision["host_peak_bytes"] == _FROZEN_PEAK
    assert "batch_transition" not in decision


def test_fresh_external_run_has_no_transition_and_unchanged_shape(
    tmp_path, monkeypatch
):
    _free(monkeypatch, 77 * GIB)
    decision = zz._select_external_fallback(
        **_GEOM,
        budget_bytes=_FRESH_RAW_BUDGET,
        cache_root=str(tmp_path / "cache"),
    )
    assert decision["refused"] is False
    assert "batch_transition" not in decision
    # A fresh run derives its own batch from the residual budget with the SAME
    # admissible-batch machinery as before: the value is exactly the inversion
    # of the phase-local residual (nothing about this mission touched it).
    assert decision["host_peak_bytes"] == xstream.estimate_external_working_set(
        n_frames=100, height=160, width=160, channels=3, tile_size=128,
        halo_px=8, column_batch=_FRESH_BATCH,
    )["host"]["peak_bytes"]
    expected = xstream.largest_admissible_column_batch(
        budget_bytes=_FRESH_EFFECTIVE_BUDGET - _REDUCE_FIXED,
        n_frames=100,
        interior_width=128,
        channels=3,
        fixed_position_bytes=xstream.EXTERNAL_BATCH_FIXED_POSITION_BYTES,
    )
    assert expected == _FRESH_BATCH
    assert decision["column_batch"] == _FRESH_BATCH


# ---------------------------------------------------------------------------
# 2. No smaller batch fits -> today's refusal, byte-identical.
# ---------------------------------------------------------------------------

def test_no_smaller_batch_fits_keeps_the_exact_refusal(tmp_path, monkeypatch):
    # 40 000 frames: even ONE position needs more than the write-phase peak, so
    # the phase budget fits the fixed terms but no batch at all. The frozen
    # batch therefore cannot shrink -> the historical refusal must survive.
    geometry = dict(_GEOM)
    geometry["max_n_frames"] = 40_000
    _free(monkeypatch, 77 * GIB)
    decision = zz._select_external_fallback(
        **geometry,
        # 7 475 000 x 100/115 == 6 500 000 exactly (phase-local budget).
        budget_bytes=7_475_000,
        cache_root=str(tmp_path / "cache"),
        frozen_tile=128,
        frozen_batch=_FROZEN_BATCH,
    )
    assert decision["refused"] is True
    assert decision["reason"] == "frozen_batch_ram"
    assert "batch_transition" not in decision
    assert decision["phase_budgets"]["margin_ceiling_bytes"] == 6_500_000

    pricing = xstream.estimate_external_working_set(
        n_frames=40_000,
        height=160,
        width=160,
        channels=3,
        tile_size=128,
        halo_px=8,
        column_batch=_FROZEN_BATCH,
    )
    host = pricing["host"]
    assert host["peak_bytes"] == 51_619_931_072
    assert decision["host_peak_bytes"] == host["peak_bytes"]
    assert decision["required_reduction_bytes"] == host["reduction_batch_bytes"]
    assert decision["reduction_budget_bytes"] == 6_500_000
    assert decision["free_bytes"] == 77 * GIB
    # Byte-identical wording: the EXACT historical detail string.
    assert decision["detail"] == (
        f"frozen column_batch {_FROZEN_BATCH} yields a host peak of "
        f"{host['peak_bytes']} B but only 6500000 B remain in the "
        "phase-local budget (the process baseline is already excluded from "
        "current available and the later assembly is separately admitted); "
        "refusing to shrink the frozen identity silently (HOST RAM)"
    )
    assert decision["trials"] == []
    assert decision["host_components"] == host


# ---------------------------------------------------------------------------
# 3. Manifest working-set block publishes the transition only when it exists.
# ---------------------------------------------------------------------------

def _manifest_block(decision, tile):
    return zz._working_set_block(
        working_set_class="external_spool",
        tile=int(tile),
        column_batch=int(decision["column_batch"]),
        budget_bytes=_RAW_BUDGET,
        raw_available_bytes=200_000_000,
        layout=(2, 2),
        tile_trials=(),
        gpu_used=False,
        gpu_tile=None,
        external=decision,
    )


def test_manifest_working_set_block_publishes_the_transition(
    tmp_path, monkeypatch
):
    decision = _select(
        tmp_path, monkeypatch, budget_bytes=_RAW_BUDGET,
        frozen_batch=_FROZEN_BATCH,
    )
    block = _manifest_block(decision, 128)
    assert block["column_batch"] == _ADMITTED_BATCH
    assert block["batch_transition"]["from_batch"] == _FROZEN_BATCH
    assert block["batch_transition"]["to_batch"] == _ADMITTED_BATCH
    assert block["batch_transition"]["required_peak_bytes"] == _FROZEN_PEAK
    assert block["batch_transition"]["phase_budget_bytes"] == _PHASE_BUDGET


def test_manifest_working_set_block_is_unchanged_without_a_transition(
    tmp_path, monkeypatch
):
    decision = _select(
        tmp_path, monkeypatch, budget_bytes=200_000_000,
        frozen_batch=_FROZEN_BATCH,
    )
    block = _manifest_block(decision, 128)
    assert "batch_transition" not in block


# ---------------------------------------------------------------------------
# 4. Durable state: explicit record, admission rewrite, audit trail survival.
# ---------------------------------------------------------------------------

def _identity(*, run_id="run-batch-shrink", tile_px=16):
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
            "nx": 2,
            "ny": 2,
            "tile_px": int(tile_px),
            "geometry_sha256": "e" * 64,
            "cell_ids_sha256": zz._r55_sha256_json(sorted(CELL_IDS)),
        },
        halo={"halo_px": 4, "provenance": {"effective_px": 4}},
        memberships={"sha256": "1" * 64, "n_cells": len(CELL_IDS)},
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
        dtypes=DTYPE_CONTRACT,
        backend={
            "backend": "cpu",
            "tile_px": int(tile_px),
            "parity_qualified": False,
        },
        execution_context={
            "available_ram_bytes": 64 * MIB,
            "workers": 2,
            "prefetch_units": 0,
            "host": "test-host",
            "note": "runtime only",
        },
    )


def _complete_result(cell_id="r0000c0000", seed=1):
    rng = np.random.default_rng(seed)
    y0, y1, x0, x1 = CORE
    bases = {
        "science": rng.normal(size=(H, W, CHANNELS)).astype(np.float32),
        "estimator": rng.normal(size=(H, W)).astype(np.float32),
        "w1": rng.normal(size=(H, W)).astype(np.float32),
        "w2": rng.normal(size=(H, W)).astype(np.float32),
        "neff": rng.normal(size=(H, W)).astype(np.float32),
        "valid": rng.integers(0, 2, size=(H, W)).astype(bool),
        "surviving": rng.integers(0, 5, size=(H, W)).astype(np.int32),
    }
    return {
        "cell_id": cell_id,
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 2,
        "total_bytes": 1234,
        "core": {
            "science_core": bases["science"][y0:y1, x0:x1, :],
            "estimator_weight_sum_core": bases["estimator"][y0:y1, x0:x1],
            "support_w1_core": bases["w1"][y0:y1, x0:x1],
            "support_w2_core": bases["w2"][y0:y1, x0:x1],
            "n_eff_support_core": bases["neff"][y0:y1, x0:x1],
            "valid_mask_core": bases["valid"][y0:y1, x0:x1],
            "surviving_sample_count_core": bases["surviving"][y0:y1, x0:x1],
        },
        "record": {
            "cell_id": cell_id,
            "patch_bounds": [0, 0, W, H],
            "patch_shape_hw": [H, W],
            "cell_gauge_mode": "per_cell",
            "backend_used": "cpu",
            "reference_frame_id": "frame_0001.fits",
            "reference_frame_role": "cell",
            "bookkeeping_reference_frame_id": "frame_0001.fits",
            "excluded": [],
        },
        "patch": {
            "bounds": (0, H, 0, W),
            "core_slice": CORE,
            "science": bases["science"],
            "valid_mask": bases["valid"],
            "surviving": bases["surviving"],
        },
    }


def _snapshot(root):
    root = Path(root)
    snap = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "tmp" not in path.relative_to(root).parts:
            snap[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return snap


def _partial_external_store(root, *, frozen_batch=4096):
    identity = _identity()
    store = zrs.RunCellStore.open(
        root,
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * MIB),
        session_id="batch-shrink-test",
    )
    try:
        zres.write_state(
            root,
            run_id=store.run_id,
            expected_cell_ids=CELL_IDS,
            policy={
                "requested": "auto",
                "effective": "bounded",
                "fallback_count": 0,
                "transitions": [],
                "dense_refusal": None,
            },
            admission=zres.build_admission(
                backend="cpu",
                tile_px=16,
                parity_qualified=False,
                working_set={
                    "class": "external_spool",
                    "tile_px": 16,
                    "column_batch": int(frozen_batch),
                },
            ),
        )
        store.commit_cell(
            _complete_result("r0000c0000"),
            attempt_id="cpu-wave:0:0:r0000c0000",
            backend="cpu",
        )
        store.commit_cell(
            _complete_result("r0001c0001", seed=2),
            attempt_id="cpu-wave:0:0:r0001c0001",
            backend="cpu",
        )
        run_id = store.run_id
        identity_sha = zrs.identity_sha256(store.identity)
    finally:
        store.close()
    return run_id, identity_sha


_PROPOSAL = {
    "schema": "zemosaic-zegrid-frozen-batch-shrink-v1",
    "from_batch": 4096,
    "to_batch": 512,
    "frozen_batch": 4096,
    "required_peak_bytes": 28_193_700_118,
    "admitted_peak_bytes": 14_000_000_000,
    "phase_budget_bytes": 14_648_888_809,
    "reason": (
        "frozen column_batch 4096 yields a host peak of 28193700118 B but only "
        "14648888809 B remain in the phase-local budget; the largest smaller "
        "admissible batch 512 fits (staging-only change, identity unchanged)"
    ),
}


def test_state_shrink_is_recorded_and_only_pending_cells_remain(tmp_path):
    root = tmp_path / "run"
    run_id, identity_sha = _partial_external_store(root)
    run_json_before = (root / "run.json").read_bytes()
    commits_before = _snapshot(root / "commits")
    proposal = dict(_PROPOSAL)

    record = zres.record_column_batch_shrink(
        root, transition=proposal, identity_sha256=identity_sha
    )
    assert record["from_batch"] == 4096
    assert record["to_batch"] == 512
    assert record["frozen_batch"] == 4096
    assert record["required_peak_bytes"] == 28_193_700_118
    assert record["admitted_peak_bytes"] == 14_000_000_000
    assert record["phase_budget_bytes"] == 14_648_888_809
    assert record["identity_sha256"] == identity_sha
    assert record["identity_unchanged"] is True
    assert record["reason"] == proposal["reason"]

    state = zres.read_state(root)
    # Class, tile and backend are untouched; ONLY the batch moved.
    assert state["admission"]["backend"] == "cpu"
    assert state["admission"]["tile_px"] == 16
    assert state["admission"]["parity_qualified"] is False
    assert state["admission"]["working_set"] == {
        "class": "external_spool",
        "tile_px": 16,
        "column_batch": 512,
    }
    assert state["column_batch_transitions"] == [record]
    assert record["plan_generation"] == state["plan_generation"]

    # Science identity and every committed Cell are byte-identical.
    assert (root / "run.json").read_bytes() == run_json_before
    assert _snapshot(root / "commits") == commits_before

    # Sparse pending set: exactly the two uncommitted Cells, never 0..K.
    boot = zres.ResumeBootstrap.open(root)
    try:
        assert boot.admission["working_set"]["column_batch"] == 512
        plan = boot.plan(list(CELL_IDS))
        assert plan["complete"] == 2
        assert plan["pending_count"] == 2
        assert sorted(plan["pending"]) == ["r0000c0001", "r0001c0000"]
    finally:
        boot.close()


def test_shrink_trail_survives_state_rewrites(tmp_path):
    root = tmp_path / "run"
    _, identity_sha = _partial_external_store(root)
    record = zres.record_column_batch_shrink(
        root, transition=dict(_PROPOSAL), identity_sha256=identity_sha
    )
    state = zres.read_state(root)
    zres.update_policy(
        root,
        policy=state["policy"],
        plan_generation=int(state["plan_generation"]),
    )
    after = zres.read_state(root)
    assert after["column_batch_transitions"] == [record]
    assert after["admission"]["working_set"]["column_batch"] == 512

    boot = zres.ResumeBootstrap.open(root)
    try:
        boot.mark_finalized({})
    finally:
        boot.close()
    final = zres.read_state(root)
    assert final["finalized"] is True
    assert final["column_batch_transitions"] == [record]
    assert final["admission"]["working_set"]["column_batch"] == 512


def test_second_shrink_is_a_contiguous_chain(tmp_path):
    root = tmp_path / "run"
    _, identity_sha = _partial_external_store(root)
    zres.record_column_batch_shrink(
        root, transition=dict(_PROPOSAL), identity_sha256=identity_sha
    )
    second = dict(_PROPOSAL)
    second.update(
        from_batch=512,
        to_batch=64,
        frozen_batch=512,
        required_peak_bytes=16_000_000_000,
        admitted_peak_bytes=1_000_000_000,
        reason="a later resume needed an even smaller staging batch",
    )
    record = zres.record_column_batch_shrink(
        root, transition=second, identity_sha256=identity_sha
    )
    state = zres.read_state(root)
    assert [r["to_batch"] for r in state["column_batch_transitions"]] == [512, 64]
    assert state["column_batch_transitions"][-1] == record
    assert state["admission"]["working_set"]["column_batch"] == 64

    # A non-contiguous proposal (from_batch != current 64) is refused.
    third = dict(_PROPOSAL)
    third.update(
        from_batch=512,
        to_batch=8,
        frozen_batch=512,
        required_peak_bytes=16_000_000_000,
        admitted_peak_bytes=500_000_000,
    )
    with pytest.raises(zres.ResumeStateError):
        zres.record_column_batch_shrink(
            root, transition=third, identity_sha256=identity_sha
        )


def test_validator_rejects_growth_and_incoherent_accounting():
    good = zres.validate_column_batch_transition(
        {
            "schema": zres.COLUMN_BATCH_TRANSITION_SCHEMA,
            "from_batch": 4096,
            "to_batch": 512,
            "frozen_batch": 4096,
            "required_peak_bytes": 1000,
            "admitted_peak_bytes": 900,
            "phase_budget_bytes": 950,
            "reason": "shrink",
            "plan_generation": 0,
            "identity_sha256": "a" * 64,
            "identity_unchanged": True,
            "recorded_unix_ns": 1,
        }
    )
    assert good["to_batch"] == 512

    bad = dict(good)
    bad.update(to_batch=4096)
    with pytest.raises(zres.ResumeStateError, match="STRICTLY smaller"):
        zres.validate_column_batch_transition(bad)

    bad = dict(good)
    bad.update(frozen_batch=2048)
    with pytest.raises(zres.ResumeStateError, match="must equal from_batch"):
        zres.validate_column_batch_transition(bad)

    bad = dict(good)
    bad.update(required_peak_bytes=900)
    with pytest.raises(zres.ResumeStateError, match="must EXCEED"):
        zres.validate_column_batch_transition(bad)

    bad = dict(good)
    bad.update(admitted_peak_bytes=1000)
    with pytest.raises(zres.ResumeStateError, match="must fit"):
        zres.validate_column_batch_transition(bad)

    bad = dict(good)
    bad.update(identity_unchanged=False)
    with pytest.raises(zres.ResumeStateError, match="identity_unchanged"):
        zres.validate_column_batch_transition(bad)


def test_state_rejects_a_transition_that_disagrees_with_the_admission(tmp_path):
    root = tmp_path / "run"
    _, identity_sha = _partial_external_store(root)
    zres.record_column_batch_shrink(
        root, transition=dict(_PROPOSAL), identity_sha256=identity_sha
    )
    state = zres.read_state(root)
    # Hand-edit the admission back to 4096: the record then describes a batch
    # the admission does not claim.
    state["admission"]["working_set"]["column_batch"] = 4096
    with pytest.raises(zres.ResumeStateError, match="never admitted"):
        zres.validate_state(state)


def test_store_without_the_new_section_stays_valid(tmp_path):
    """Additive schema: an old store (no ``column_batch_transitions``) loads."""
    root = tmp_path / "run"
    _partial_external_store(root)
    state = zres.read_state(root)
    assert "column_batch_transitions" not in state
    zres.validate_state(state)
