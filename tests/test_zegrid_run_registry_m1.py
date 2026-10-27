"""ZM-ZEGRID-RESILIENT-RUN-RESUME M1 REWORK-1 — registry contract tests.

Deterministic ownership/transaction/schema tests: no GPU, no product run.
The real orchestration test lives in
``tests/test_zegrid_resume_m1_conservation.py``.
"""

from __future__ import annotations

import gc
import pickle
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for path in (SRC, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402

H, W = 8, 10
CH = 3
CORE = (2, 6, 2, 7)  # y0, y1, x0, x1
CH_, CW_ = CORE[1] - CORE[0], CORE[3] - CORE[2]
PATCH_BOUNDS_XY = [0, 0, W, H]  # real producer convention (x0, y0, x1, y1)


def _bases(seed=1):
    rng = np.random.default_rng(seed)
    return {
        "science": rng.normal(size=(H, W, CH)).astype(np.float32),
        "estimator_weight_sum": rng.normal(size=(H, W)).astype(np.float32),
        "support_w1": rng.normal(size=(H, W)).astype(np.float32),
        "support_w2": rng.normal(size=(H, W)).astype(np.float32),
        "n_eff_support": rng.normal(size=(H, W)).astype(np.float32),
        "valid_mask": rng.integers(0, 2, size=(H, W)).astype(bool),
        "surviving": rng.integers(0, 5, size=(H, W)).astype(np.int32),
    }


def _result(cell_id="r0000c0000", *, backend="cpu", mode="per_cell",
            bases=None, seed=1, patch=True):
    b = _bases(seed) if bases is None else bases
    y0, y1, x0, x1 = CORE
    core = {
        "science_core": b["science"][y0:y1, x0:x1, :],
        "estimator_weight_sum_core": b["estimator_weight_sum"][y0:y1, x0:x1],
        "support_w1_core": b["support_w1"][y0:y1, x0:x1],
        "support_w2_core": b["support_w2"][y0:y1, x0:x1],
        "n_eff_support_core": b["n_eff_support"][y0:y1, x0:x1],
        "valid_mask_core": b["valid_mask"][y0:y1, x0:x1],
        "surviving_sample_count_core": b["surviving"][y0:y1, x0:x1],
    }
    out = {
        "cell_id": cell_id,
        "status": "complete",
        "backend_used": backend,
        "n_frames": 3,
        "total_bytes": 1234,
        "core": core,
        "record": {
            "cell_id": cell_id,
            "patch_bounds": list(PATCH_BOUNDS_XY),
            "patch_shape_hw": [H, W],
            "cell_gauge_mode": mode,
            "backend_used": backend,
            "reference_frame_id": "frame-0000",
            "reference_frame_role": "cell" if mode == "per_cell" else "global",
            "bookkeeping_reference_frame_id": "frame-0000",
            "excluded": [[1, 2, "reason"]],
        },
    }
    if patch:
        out["patch"] = {
            "bounds": (0, H, 0, W),
            "core_slice": CORE,
            "science": b["science"],
            "valid_mask": b["valid_mask"],
            "surviving": b["surviving"],
        }
    return out


def _empty(cell_id="r0000c0001", mode="global"):
    return {
        "cell_id": cell_id,
        "status": "empty",
        "record": {
            "cell_id": cell_id,
            "status": "empty",
            "mode": None,
            "cell_gauge_mode": mode,
            "n_contributors": 0,
            "patch_bounds": list(PATCH_BOUNDS_XY),
            "patch_shape_hw": [H, W],
        },
    }


def _registry(cids=("r0000c0000", "r0000c0001", "r0001c0000"), **kw):
    return zrr.RunResultRegistry(
        run_id="run-test", expected_cell_ids=cids, **kw
    )


def _expected_retained(bases):
    """Unique full allocations retained by the real per_cell alias layout."""
    return int(
        bases["science"].nbytes
        + bases["estimator_weight_sum"].nbytes
        + bases["support_w1"].nbytes
        + bases["support_w2"].nbytes
        + bases["n_eff_support"].nbytes
        + bases["valid_mask"].nbytes
        + bases["surviving"].nbytes
    )


def test_accept_complete_per_cell_preserves_full_payload_and_honest_bytes():
    reg = _registry(gauge_mode="per_cell")
    bases = _bases(seed=5)
    res = _result(bases=bases, backend="gpu")
    before = zrr.fingerprint_result(res)
    receipt = reg.accept(
        res, backend="gpu", plan_generation=0, attempt_id="gpu:0"
    )

    assert receipt.accepted is True
    assert receipt.cell_id == "r0000c0000"
    assert receipt.status == zrr.STATUS_COMPLETE
    assert receipt.payload_sha256 == before["payload_sha256"]
    meta = reg.metadata("r0000c0000")
    assert meta["core_sha256"] == before["core_hashes"]
    assert meta["patch_sha256"] == before["patch_hashes"]
    assert meta["reference_pin"]["reference_frame_id"] == "frame-0000"
    assert meta["geometry"]["bounds"] == [0, H, 0, W]
    assert meta["geometry"]["core_slice"] == list(CORE)
    # Honest accounting: core views + patch planes alias the same allocations.
    assert meta["logical_bytes"] > 0
    assert meta["retained_bytes"] == _expected_retained(bases)
    assert reg.retained_bytes == _expected_retained(bases)
    assert reg.logical_bytes == meta["logical_bytes"]
    assert reg.counts()["retained_buffers"] == 7
    assert reg.pending_cell_ids() == ["r0000c0001", "r0001c0000"]


def test_global_complete_has_no_patch_and_same_retained_alias_truth():
    reg = _registry(gauge_mode="global")
    bases = _bases(seed=6)
    res = _result(bases=bases, mode="global", patch=False)
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:0")
    meta = reg.metadata("r0000c0000")
    assert meta["patch_sha256"] is None
    # Core views still retain the full underlying allocations.
    assert meta["retained_bytes"] == _expected_retained(bases)
    assert reg.counts()["retained_buffers"] == 7


def test_incomplete_complete_is_rejected_without_state_change():
    reg = _registry(gauge_mode="global")
    res = _result(mode="global", patch=False)
    del res["core"]["science_core"]
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:bad")
    assert reg.counts()["accepted"] == 0
    assert reg.retained_bytes == 0
    assert reg.pending_cell_ids() == ["r0000c0000", "r0000c0001", "r0001c0000"]


def test_per_cell_patch_is_mandatory_and_shapes_are_validated():
    reg = _registry(gauge_mode="per_cell")
    res = _result()
    del res["patch"]
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:no")

    bad = _result(seed=2)
    bad["core"]["science_core"] = np.zeros((3, 5, 3), dtype=np.float32)
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(bad, backend="cpu", plan_generation=0, attempt_id="cpu:bad")
    assert reg.counts()["accepted"] == 0


def test_empty_validated_real_schema_and_holes_are_preserved():
    reg = _registry(gauge_mode="global")
    reg.accept(_empty(), backend="cpu", plan_generation=0, attempt_id="cpu:e")
    reg.accept(
        _result("r0001c0000", mode="global", patch=False, seed=3),
        backend="cpu",
        plan_generation=0,
        attempt_id="cpu:c",
    )
    counts = reg.counts()
    assert counts["accepted"] == 2
    assert counts["empty_validated"] == 1
    assert counts["complete"] == 1
    assert reg.pending_cell_ids() == ["r0000c0000"]
    assert reg.metadata("r0000c0001")["logical_bytes"] == 0
    # An empty payload carrying core planes is contradictory.
    bad = _empty("r0000c0000")
    bad["core"] = _result("r0000c0000")["core"]
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(bad, backend="cpu", plan_generation=0, attempt_id="cpu:x")


def test_zero_support_populated_empty_requires_real_gauge_evidence():
    reg = _registry(gauge_mode="global")
    zero_support = _empty("r0000c0000", mode="global")
    zero_support["record"]["n_contributors"] = 3
    zero_support["record"]["gauge_status"] = "empty"
    zero_support["record"]["gauge_detail"] = "no active frame"
    zero_support["gauge_status"] = "empty"
    reg.accept(
        zero_support, backend="cpu", plan_generation=0, attempt_id="cpu:zs"
    )
    meta = reg.metadata("r0000c0000")
    assert meta["status"] == zrr.STATUS_EMPTY_VALIDATED
    assert meta["geometry"]["n_contributors"] == 3
    assert meta["geometry"]["gauge_status"] == "empty"
    assert reg.counts()["empty_validated"] == 1

    # A populated empty WITHOUT the real zero-support evidence is rejected and
    # its contributor count is never rewritten to satisfy the schema.
    bad = _empty("r0000c0001", mode="global")
    bad["record"]["n_contributors"] = 3
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(bad, backend="cpu", plan_generation=0, attempt_id="cpu:bad")
    assert reg.metadata("r0000c0001") is None
    assert reg.counts()["accepted"] == 1


def test_memmap_slice_counts_the_mapping_once(tmp_path):
    path = tmp_path / "mapping.dat"
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(4_000_000,))
    mm[:] = 1
    mm.flush()
    first = mm[:60].reshape(4, 5, 3)
    second = mm[:20].reshape(4, 5)
    res = _result(mode="global", patch=False)
    res["core"]["science_core"] = first
    res["core"]["estimator_weight_sum_core"] = second
    reg = _registry(gauge_mode="global")
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:mm")
    counts = reg.counts()
    assert counts["memmap_bytes"] == mm.nbytes
    assert counts["retained_bytes"] >= mm.nbytes
    assert reg.metadata("r0000c0000")["retained_bytes"] >= mm.nbytes
    reg.dispose()
    del first, second, mm
    path.unlink(missing_ok=True)


def test_memoryview_slice_follows_obj_to_full_owner():
    owner = bytearray(4_000_000)
    view = memoryview(owner)[:240]
    arr = np.frombuffer(view, dtype=np.uint8).reshape(4, 5, 12)
    res = _result(mode="global", patch=False)
    res["core"]["science_core"] = arr
    reg = _registry(gauge_mode="global")
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:mv")
    meta = reg.metadata("r0000c0000")
    assert meta["retained_bytes"] >= 4_000_000
    assert reg.counts()["memmap_bytes"] == 0
    ref = weakref.ref(arr)
    reg.dispose()
    del res, owner, view, arr
    gc.collect()
    assert ref() is None


def test_status_is_part_of_duplicate_identity_and_conflicts_fail_closed():
    reg = _registry(gauge_mode="global")
    empty = _empty("r0000c0000")
    reg.accept(empty, backend="cpu", plan_generation=0, attempt_id="same")
    complete = _result("r0000c0000", mode="global", patch=False)
    with pytest.raises(zrr.DuplicateCellError) as info:
        reg.accept(
            complete, backend="cpu", plan_generation=0, attempt_id="same"
        )
    assert "EMPTY_VALIDATED" in str(info.value)
    assert "COMPLETE" in str(info.value)
    assert reg.metadata("r0000c0000")["status"] == zrr.STATUS_EMPTY_VALIDATED
    assert reg.counts()["empty_validated"] == 1


def test_identical_duplicate_is_idempotent_and_counts_once():
    reg = _registry(gauge_mode="per_cell")
    res = _result()
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="same")
    before_bytes = reg.retained_bytes
    before_logical = reg.logical_bytes
    receipt = reg.accept(
        res, backend="cpu", plan_generation=0, attempt_id="same"
    )
    assert receipt.idempotent is True
    assert reg.retained_bytes == before_bytes
    assert reg.logical_bytes == before_logical
    assert reg.counts()["accepted"] == 1


def test_generation_must_match_current_and_is_captured_at_dispatch():
    reg = _registry(gauge_mode="per_cell")
    reg.accept(
        _result(seed=1), backend="cpu", plan_generation=0, attempt_id="cpu:0"
    )
    gen = reg.advance_generation(reason="dense_to_bounded")
    assert gen == 1
    assert reg.dispatch_generation() == 1
    # Stale and future generations are rejected without state mutation.
    with pytest.raises(zrr.StaleGenerationError):
        reg.accept(
            _result("r0000c0001", seed=2),
            backend="cpu",
            plan_generation=0,
            attempt_id="cpu:late",
        )
    with pytest.raises(zrr.FutureGenerationError):
        reg.accept(
            _result("r0000c0001", seed=2),
            backend="cpu",
            plan_generation=2,
            attempt_id="cpu:future",
        )
    assert reg.pending_cell_ids() == ["r0000c0001", "r0001c0000"]
    # Previous generation payload survives the transition.
    assert reg.metadata("r0000c0000")["payload_sha256"]
    reg.accept(
        _result("r0000c0001", seed=2),
        backend="cpu",
        plan_generation=1,
        attempt_id="cpu:now",
    )
    assert reg.counts()["accepted"] == 2


def test_generation_is_required_and_never_opportunistically_adopted():
    reg = _registry(gauge_mode="per_cell")
    with pytest.raises(TypeError):
        reg.accept(_result(), backend="cpu", attempt_id="cpu:missing")
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(
            _result(), backend="cpu", plan_generation=True, attempt_id="cpu:b"
        )
    assert reg.counts()["accepted"] == 0


def test_staging_failure_leaves_no_partial_record():
    class Boom:
        def __array__(self, *args, **kwargs):
            raise RuntimeError("boom")

    reg = _registry(gauge_mode="per_cell")
    bad = _result()
    bad["core"]["science_core"] = Boom()
    with pytest.raises(zrr.AcceptanceError):
        reg.accept(bad, backend="cpu", plan_generation=0, attempt_id="cpu:b")
    assert reg.counts()["accepted"] == 0
    assert reg.retained_bytes == 0
    assert reg.logical_bytes == 0


def test_unsupported_buffer_owner_fails_closed():
    class UnsupportedOwner(np.ndarray):
        @property
        def base(self):
            return "unsupported-owner"

    reg = _registry(gauge_mode="global")
    res = _result(mode="global", patch=False)
    # A genuinely unsupported base owner must fail closed rather than be
    # counted with a guessed size. (numpy normally normalizes buffers to
    # memoryview, which IS supported and counted by ``_root_buffer``.)
    res["core"]["estimator_weight_sum_core"] = np.zeros(
        (CORE[1] - CORE[0], CORE[3] - CORE[2]), dtype=np.float32
    ).view(UnsupportedOwner)
    with pytest.raises(zrr.UnsupportedBufferOwnerError):
        reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:b")
    assert reg.counts()["accepted"] == 0


def test_tiny_views_and_noncontiguous_views_count_the_base_once():
    reg = _registry(gauge_mode="global")
    big_science = np.ones((1000, 1000, 3), dtype=np.float32)  # 12 MB
    big_plane = np.ones((1000, 1000), dtype=np.float32)  # 4 MB
    big_mask = np.ones((1000, 1000), dtype=bool)
    big_surv = np.ones((1000, 1000), dtype=np.int32)
    y0, y1, x0, x1 = CORE
    res = _result(mode="global", patch=False)
    res["core"] = {
        "science_core": big_science[y0:y1, x0:x1, :],
        # Non-contiguous view: still the same 4 MB base.
        "estimator_weight_sum_core": big_plane[::2, ::2][y0:y1, x0:x1],
        "support_w1_core": big_plane[y0:y1, x0:x1],
        "support_w2_core": big_plane[y0:y1, x0:x1],
        "n_eff_support_core": big_plane[y0:y1, x0:x1],
        "valid_mask_core": big_mask[y0:y1, x0:x1],
        "surviving_sample_count_core": big_surv[y0:y1, x0:x1],
    }
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:v")
    expected = (
        big_science.nbytes
        + big_plane.nbytes
        + big_mask.nbytes
        + big_surv.nbytes
    )
    meta = reg.metadata("r0000c0000")
    assert meta["retained_bytes"] == expected
    assert meta["logical_bytes"] < expected
    assert reg.counts()["retained_buffers"] == 4


def test_aliases_across_cells_are_counted_once_globally():
    reg = _registry(
        cids=("r0000c0000", "r0000c0001"), gauge_mode="per_cell"
    )
    shared = _bases(seed=11)
    reg.accept(
        _result("r0000c0000", bases=shared),
        backend="cpu",
        plan_generation=0,
        attempt_id="cpu:0",
    )
    first = reg.retained_bytes
    reg.accept(
        _result("r0000c0001", bases=shared, seed=11),
        backend="cpu",
        plan_generation=0,
        attempt_id="cpu:1",
    )
    # Two cells, same backing allocations: retained bytes do not double.
    assert reg.retained_bytes == first == _expected_retained(shared)
    assert reg.logical_bytes == 2 * reg.metadata("r0000c0000")["logical_bytes"]


def test_memmap_is_declared_and_counted_conservatively(tmp_path):
    reg = _registry(gauge_mode="global")
    path = tmp_path / "plane.dat"
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(H, W))
    mm[:] = 1.0
    mm.flush()
    res = _result(mode="global", patch=False)
    # Full-size core planes so the mmap root is the only changed plane.
    res["core"] = {
        "science_core": np.zeros((H, W, CH), dtype=np.float32),
        "estimator_weight_sum_core": mm,
        "support_w1_core": np.zeros((H, W), dtype=np.float32),
        "support_w2_core": np.zeros((H, W), dtype=np.float32),
        "n_eff_support_core": np.zeros((H, W), dtype=np.float32),
        "valid_mask_core": np.zeros((H, W), dtype=bool),
        "surviving_sample_count_core": np.zeros((H, W), dtype=np.int32),
    }
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:mm")
    counts = reg.counts()
    assert counts["memmap_bytes"] >= mm.nbytes
    assert counts["retained_bytes"] >= mm.nbytes
    reg.dispose()
    del mm
    path.unlink(missing_ok=True)


def test_out_of_order_acceptance_returns_canonical_payload_order():
    reg = _registry(gauge_mode="per_cell")
    third = _result("r0001c0000", seed=3)
    first = _result("r0000c0000", seed=1)
    second = _result("r0000c0001", seed=2)
    for res, attempt in (
        (third, "cpu:3"), (first, "cpu:1"), (second, "cpu:2")
    ):
        reg.accept(
            res, backend="cpu", plan_generation=0, attempt_id=attempt
        )
    assert [r["cell_id"] for r in reg.ordered_payloads()] == [
        "r0000c0000", "r0000c0001", "r0001c0000"
    ]


def test_dispose_releases_payload_ownership_once_and_is_idempotent():
    reg = _registry(gauge_mode="per_cell")
    res = _result()
    reg.accept(res, backend="cpu", plan_generation=0, attempt_id="cpu:0")
    refs = [
        weakref.ref(res["core"]["science_core"]),
        weakref.ref(res["patch"]["science"]),
    ]
    del res
    assert all(r() is not None for r in refs)
    before = reg.counts()
    summary = reg.dispose()
    assert summary["released_cells"] == 1
    assert summary["released_logical_bytes"] == before["logical_bytes"]
    assert summary["released_retained_bytes"] == before["retained_bytes"]
    assert summary["already_disposed"] is False
    gc.collect()
    assert all(r() is None for r in refs)
    again = reg.dispose()
    assert again["already_disposed"] is True
    assert again["released_cells"] == 0
    assert reg.counts()["accepted"] == 0
    with pytest.raises(zrr.RegistryClosedError):
        reg.accept(
            _result(), backend="cpu", plan_generation=0, attempt_id="cpu:x"
        )


def test_journal_is_bounded_and_holds_no_payload_reference():
    reg = zrr.RunResultRegistry(
        run_id="run-test",
        expected_cell_ids=[f"r0000c{i:04d}" for i in range(6)],
        gauge_mode="per_cell",
        max_events=4,
    )
    for i in range(6):
        reg.accept(
            _result(f"r0000c{i:04d}", seed=i),
            backend="cpu",
            plan_generation=0,
            attempt_id=f"cpu:{i}",
        )
    events = reg.journal_events()
    assert len(events) == 4
    assert reg.snapshot()["journal"]["dropped"] >= 1
    for event in events:
        text = repr(event)
        assert "science_core" not in text
        assert "ndarray" not in text


def test_registry_and_thread_channel_are_explicitly_not_picklable():
    reg = _registry(gauge_mode="per_cell")
    with pytest.raises(TypeError, match="process-local"):
        pickle.dumps(reg)
    with pytest.raises(TypeError, match="SpawnStopChannel"):
        pickle.dumps(zrr.CooperativeStopChannel())


def test_thread_stop_request_ack_is_idempotent_and_retains_payloads():
    reg = _registry(gauge_mode="per_cell")
    reg.accept(
        _result(), backend="cpu", plan_generation=0, attempt_id="cpu:0"
    )
    assert reg.stop_requested is False
    assert reg.request_stop("test stop") is True
    assert reg.request_stop("again") is False
    assert reg.acknowledge_stop("STOPPED_IN_MEMORY", "safe boundary") is True
    assert reg.acknowledge_stop("COMPLETE", "late") is False
    assert reg.stop.wait_ack(0.0) is True
    assert reg.stop.snapshot()["ack_state"] == "STOPPED_IN_MEMORY"
    assert reg.counts()["accepted"] == 1


def test_spawn_stop_channel_real_child_request_ack_and_timeout():
    import multiprocessing as mp

    from spawn_stop_child import child_no_ack, child_request_ack

    ctx = mp.get_context("spawn")
    channel = zrr.SpawnStopChannel.create(ctx)
    child = ctx.Process(target=child_request_ack, args=(channel,))
    child.start()
    try:
        assert channel.request("spawn stop") is True
        assert channel.request("again") is False
        child.join(20)
        assert child.exitcode == 0, "spawn child failed"
        assert channel.wait_ack(2.0) is True
        snap = channel.snapshot()
        assert snap["transport"] == "spawn"
        assert snap["requested_reason"] == "spawn stop"
        assert snap["ack_state"] == "STOPPED_IN_MEMORY"
        assert snap["ack_detail"] == "child saw request"
        assert channel.acknowledge("COMPLETE", "late") is False
    finally:
        child.join(5)
        if child.is_alive():
            child.terminate()
            child.join(5)

    # A child that never acknowledges must not fabricate an ack.
    no_ack_channel = zrr.SpawnStopChannel.create(ctx)
    child2 = ctx.Process(target=child_no_ack, args=(no_ack_channel,))
    child2.start()
    try:
        assert no_ack_channel.request("no ack") is True
        child2.join(20)
        assert child2.exitcode == 0
        assert no_ack_channel.wait_ack(0.2) is False
        assert no_ack_channel.snapshot()["acknowledged"] is False
    finally:
        child2.join(5)
        if child2.is_alive():
            child2.terminate()
            child2.join(5)
