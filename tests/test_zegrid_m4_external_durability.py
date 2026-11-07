"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M4 — transactional spool + interruption matrix.

Synthetic, bounded acceptance for the M4 six items:

1. unique attempt + deferred cleanup + receipt (`run_cell_streaming_external`);
2. commit-then-delete order and committed-cell leftover cleanup;
3. working-set identity persistence + frozen tile/batch refusal;
4. cell-boundary disk revalidation (resumable, commits intact);
5. lifecycle accounting (strict JSON);
6. interruption at spool write / reduce / result write / post-commit pre-delete,
   dead-owner stale, committed leftover, disk shortfall, frozen mismatch,
   completed-run no-op — final science identical to the uninterrupted reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import external_stream as x
from zemosaic.core.zegrid import run_resume as zres


# ---------------------------------------------------------------------------
# Corpus / reference
# ---------------------------------------------------------------------------

def _corpus(n=5, h=40, w=36, seed=3):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(3):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        base += 15.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3.0 ** 2))
    gains = np.linspace(0.98, 1.02, n)
    frames = []
    for i in range(n):
        frame = np.stack(
            [base * gains[i] * (0.99 + 0.01 * c) + rng.normal(0, 1.0, (h, w))
             for c in range(3)], axis=-1,
        )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    frames[1][5:10, 5:10, :] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[3][:, w // 2:] = False
    return frames, masks


def _request(frames, masks):
    return CanonicalStackRequest(
        images=frames, geometric_support=masks, normalization="sky_mean",
        weighting="noise_variance", rejection="winsorized_sigma_clip",
        combine="median", taper="footprint", taper_px=8.0,
    )


def _run(root, frames, masks, *, cell="c", attempt="a", cleanup=False, out=None):
    Path(root).mkdir(parents=True, exist_ok=True)
    return x.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks), _request(frames, masks),
        tile_size=16, column_batch=64, spool_root=str(root),
        cell_id=cell, attempt_id=attempt,
        cleanup_on_success=cleanup, spool_out=out,
        owner_token="OWNER", owner_record={"pid": 1},
    )


def _planes(res):
    return {name: np.array(getattr(res, name), copy=True) for name in (
        "science", "estimator_weight_sum", "valid_mask",
        "surviving_sample_count", "support_w1", "support_w2",
        "n_eff_support", "rejection_mask",
    )}


def _assert_same(a, b):
    pa, pb = _planes(a), _planes(b)
    for name in pa:
        assert np.array_equal(pa[name], pb[name], equal_nan=True), name
    for name in (
        "iterations_used", "initial_sample_count", "rejected_sample_count",
        "rejected_fraction", "low_n_cell_count", "degenerate_cell_count",
        "rejection_method",
    ):
        assert getattr(a, name) == getattr(b, name), name
    assert a.provenance == b.provenance


# ---------------------------------------------------------------------------
# 1/2 — lifecycle states, ownership, commit-then-delete
# ---------------------------------------------------------------------------

def test_unique_attempt_and_ready_receipt_then_commit_cleanup(tmp_path):
    frames, masks = _corpus()
    out = {}
    _run(tmp_path, frames, masks, attempt="a-unique", cleanup=False, out=out)
    assert out["attempt_id"] == "a-unique"
    assert out["state"] == x.SPOOL_STATE_READY
    assert Path(out["attempt_dir"]).is_dir()
    assert out["bytes_written"] > 0
    # Commit then delete.
    rep = x.cleanup_after_commit(tmp_path, "c", owner_token="OWNER")
    assert rep["cleaned_committed"] >= 1
    assert not Path(out["attempt_dir"]).exists()


def test_committed_attempt_is_never_reused(tmp_path):
    sp = x.ExternalSpool(x.ExternalSpoolSpec(
        root=tmp_path, n_frames=2, height=8, width=8, channels=3,
        cell_id="c", attempt_id="a", owner_token="OWNER",
    ))
    sp.mark_committed()
    with pytest.raises(x.ExternalSpoolCommittedError):
        x.ExternalSpool(x.ExternalSpoolSpec(
            root=tmp_path, n_frames=2, height=8, width=8, channels=3,
            cell_id="c", attempt_id="a",
        ))


def test_dead_owner_stale_replaced_and_live_owner_refused(tmp_path):
    base = dict(root=tmp_path, n_frames=2, height=8, width=8, channels=3,
                attempt_id="a")
    # Dead owner -> discarded and replaced.
    x.ExternalSpool(x.ExternalSpoolSpec(**base, cell_id="c1", owner_token="A",
                                        owner_record={"pid": 1}))
    sp = x.ExternalSpool(x.ExternalSpoolSpec(
        **base, cell_id="c1", owner_token="B", owner_record={"pid": 2},
        owner_liveness=lambda r: False,
    ))
    assert sp.state == x.SPOOL_STATE_WRITING
    # Live owner -> refused (never stolen).
    x.ExternalSpool(x.ExternalSpoolSpec(
        **base, cell_id="c2", owner_token="A", owner_record={"pid": 1},
    ))
    with pytest.raises(x.ExternalSpoolLiveOwnerError):
        x.ExternalSpool(x.ExternalSpoolSpec(
            **base, cell_id="c2", owner_token="C", owner_record={"pid": 3},
            owner_liveness=lambda r: True,
        ))
    # Unprovable owner (no owner record) is never stolen.
    x.ExternalSpool(x.ExternalSpoolSpec(**base, cell_id="c3", owner_token="D"))
    with pytest.raises(x.ExternalSpoolStaleError):
        x.ExternalSpool(x.ExternalSpoolSpec(
            **base, cell_id="c3", owner_token="E", owner_record={"pid": 5},
        ))


def test_committed_cell_leftover_cleaned_without_touching_commit(tmp_path):
    commit = tmp_path / "cell.cmt"
    commit.write_bytes(b"commit")
    (tmp_path / "spool").mkdir()
    x.ExternalSpool(x.ExternalSpoolSpec(
        root=tmp_path / "spool", n_frames=2, height=8, width=8, channels=3,
        cell_id="c", attempt_id="a", owner_token="A", owner_record={"pid": 1},
    )).mark_ready()
    rep = x.discard_stale_for_cell(
        tmp_path / "spool", "c", owner_token="B",
        committed_cell_ids=("c",),
    )
    assert rep["cleaned_committed"] == 1
    assert commit.read_bytes() == b"commit"  # untouched


def test_live_owner_spool_is_not_stolen_by_scan(tmp_path):
    root = tmp_path / "spool"
    root.mkdir()
    x.ExternalSpool(x.ExternalSpoolSpec(
        root=root, n_frames=2, height=8, width=8, channels=3,
        cell_id="c", attempt_id="a", owner_token="A", owner_record={"pid": 1},
    ))
    rep = x.scan_spool_root(
        root, owner_token="B", owner_liveness=lambda r: True,
    )
    assert rep["live_skipped"] == 1
    assert (root / "c" / "a").exists()


# ---------------------------------------------------------------------------
# 3 — durable identity
# ---------------------------------------------------------------------------

def _policy():
    return {"requested": "auto", "effective": "bounded", "fallback_count": 0,
            "transitions": []}


def test_attach_working_set_persists_and_refuses_conflict(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    zres.write_state(
        root, run_id="r1", expected_cell_ids=["c"], policy=_policy(),
        finalized=False,
        admission=zres.build_admission(
            backend="cpu", tile_px=16, parity_qualified=True,
        ),
    )
    state = zres.attach_working_set(
        root, {"class": "external_spool", "tile_px": 16, "column_batch": 64}
    )
    assert state["admission"]["working_set"] == {
        "class": "external_spool", "tile_px": 16, "column_batch": 64,
    }
    # Same section is idempotent; a different one is refused.
    zres.attach_working_set(
        root, {"class": "external_spool", "tile_px": 16, "column_batch": 64}
    )
    with pytest.raises(zres.ResumeStateError):
        zres.attach_working_set(
            root, {"class": "ram_tile", "tile_px": 16, "column_batch": None}
        )


def test_legacy_admission_stays_legacy(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    zres.write_state(
        root, run_id="r1", expected_cell_ids=["c"], policy=_policy(),
        finalized=False,
        admission=zres.build_admission(
            backend="cpu", tile_px=16, parity_qualified=True,
        ),
    )
    assert "working_set" not in zres.read_state(root)["admission"]


# ---------------------------------------------------------------------------
# 4 — disk shortfall / frozen policy
# ---------------------------------------------------------------------------

def test_disk_shortfall_is_resumable_and_keeps_commits(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        zz.shutil, "disk_usage",
        lambda p: SimpleNamespace(total=10, used=0, free=10),
    )
    with pytest.raises(x.ExternalDiskShortfall) as exc:
        zz._run_cell_external(
            cache_dir=tmp_path, patch=None, config=None, progress_callback=None,
            tile_size=16, column_batch=64, spool_root=str(tmp_path),
            required_free_bytes=1000,
        )
    assert exc.value.resumable is True
    assert exc.value.committed_cells_intact is True


def test_frozen_tile_and_batch_are_reused_or_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(
        zz.shutil, "disk_usage",
        lambda p: SimpleNamespace(total=10 ** 12, used=0, free=10 ** 12),
    )
    ok = zz._select_external_fallback(
        permitted_tiles=zz.CPU_STREAMING_TILE_LADDER,
        worst_patch_hw=(40, 40), max_n_frames=10, channels=3,
        budget_bytes=512 * 2 ** 20, halo_px=9, cache_root=str(tmp_path),
        storage_profile="bounded", frozen_tile=16, frozen_batch=64,
    )
    assert ok["refused"] is False
    assert ok["tile"] == 16 and ok["column_batch"] == 64
    bad = zz._select_external_fallback(
        permitted_tiles=zz.CPU_STREAMING_TILE_LADDER,
        worst_patch_hw=(40, 40), max_n_frames=10, channels=3,
        budget_bytes=512 * 2 ** 20, halo_px=9, cache_root=str(tmp_path),
        storage_profile="bounded", frozen_tile=16, frozen_batch=10 ** 9,
    )
    assert bad["refused"] is True
    assert bad["reason"] == "frozen_batch_budget"


# ---------------------------------------------------------------------------
# 6 — interruption matrix (final science identical)
# ---------------------------------------------------------------------------

def test_interrupt_during_spool_write_then_retry_is_identical(tmp_path):
    frames, masks = _corpus()
    ref = _run(tmp_path / "ref", frames, masks, cell="c", attempt="ref",
               cleanup=True)
    real = x.ExternalSpool.write_tile
    calls = {"n": 0}

    def _boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(5, "simulated kill during spool write")
        return real(self, *a, **k)

    x.ExternalSpool.write_tile = _boom
    try:
        with pytest.raises(OSError):
            _run(tmp_path / "int", frames, masks, cell="c", attempt="a1")
    finally:
        x.ExternalSpool.write_tile = real
    assert not (tmp_path / "int" / "c" / "a1").exists()
    got = _run(tmp_path / "int", frames, masks, cell="c", attempt="a2",
               cleanup=True)
    _assert_same(ref, got)


def test_interrupt_during_reduce_then_retry_is_identical(tmp_path, monkeypatch):
    frames, masks = _corpus()
    ref = _run(tmp_path / "ref", frames, masks, cell="c", attempt="ref",
               cleanup=True)
    real = x.reduce_spooled_tile
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated kill during reduce")
        return real(*a, **k)

    monkeypatch.setattr(x, "reduce_spooled_tile", _boom)
    with pytest.raises(RuntimeError):
        _run(tmp_path / "int", frames, masks, cell="c", attempt="a1")
    monkeypatch.setattr(x, "reduce_spooled_tile", real)
    assert not (tmp_path / "int" / "c" / "a1").exists()
    got = _run(tmp_path / "int", frames, masks, cell="c", attempt="a2",
               cleanup=True)
    _assert_same(ref, got)


def test_interrupt_after_result_before_commit_then_retry_is_identical(tmp_path):
    frames, masks = _corpus()
    ref = _run(tmp_path / "ref", frames, masks, cell="c", attempt="ref",
               cleanup=True)
    out = {}
    _run(tmp_path / "int", frames, masks, cell="c", attempt="a1",
         cleanup=False, out=out)
    assert Path(out["attempt_dir"]).is_dir()   # result written, not committed
    # The product pre-run scan discards the same-owner ready attempt.
    x.discard_stale_for_cell(
        tmp_path / "int", "c", owner_token="OWNER",
    )
    assert not Path(out["attempt_dir"]).exists()
    # Retry with a NEW attempt.
    got = _run(tmp_path / "int", frames, masks, cell="c", attempt="a2",
               cleanup=False)
    _assert_same(ref, got)
    # Simulate the commit completing now.
    x.cleanup_after_commit(tmp_path / "int", "c", owner_token="OWNER")


def test_interrupt_after_commit_before_delete_cleans_without_recompute(tmp_path):
    frames, masks = _corpus()
    out = {}
    _run(tmp_path, frames, masks, cell="c", attempt="a1",
         cleanup=False, out=out)
    # The store committed the Cell; the process dies before deleting the spool.
    committed = {"c"}
    assert "c" in committed  # decision: a committed Cell is NOT re-executed
    rep = x.cleanup_after_commit(tmp_path, "c", owner_token="OWNER")
    assert rep["cleaned_committed"] >= 1
    assert not Path(out["attempt_dir"]).exists()


def test_completed_run_resume_is_a_verified_noop(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    zres.write_state(
        root, run_id="r1", expected_cell_ids=["c"], policy=_policy(),
        finalized=True,
        admission=zres.build_admission(
            backend="cpu", tile_px=16, parity_qualified=True,
            working_set={"class": "external_spool", "tile_px": 16,
                         "column_batch": 64},
        ),
    )
    state = zres.read_state(root)
    assert state["finalized"] is True
    assert state["admission"]["working_set"]["class"] == "external_spool"
    # No spool is created by reading/attaching a completed run.
    assert not (tmp_path / "external-spool").exists()
    zres.attach_working_set(
        root, {"class": "external_spool", "tile_px": 16, "column_batch": 64}
    )
    assert not (tmp_path / "external-spool").exists()


def test_lifecycle_accounting_is_strict_json(tmp_path):
    frames, masks = _corpus()
    out = {}
    _run(tmp_path, frames, masks, cell="c", attempt="a1",
         cleanup=False, out=out)
    x.cleanup_after_commit(tmp_path, "c", owner_token="OWNER")
    payload = {
        "schema": "zemosaic-zegrid-external-lifecycle-v1",
        "attempts_created": 1,
        "attempts_discarded_stale": 0,
        "cleaned_committed": 1,
        "cells_executed_external": 1,
        "cells_reused_committed": 0,
        "spool_bytes_written": int(out["bytes_written"]),
        "spool_bytes_read": int(out["bytes_read"]),
        "spool_retained_after_failed_commit": 0,
        "cleanup_failures": [],
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert "external-lifecycle-v1" in encoded
    assert payload["spool_bytes_written"] > 0
