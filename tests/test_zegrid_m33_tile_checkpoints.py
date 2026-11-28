"""ZM-M33-TILE-CHECKPOINTS — durable completed-tile reuse (CPU external path).

Bounded synthetic acceptance for the lot-2 contract:

* a checkpointed uninterrupted run is bit-exact with the historical run and
  stores only reduced interior planes (no N-axis pool);
* a deterministic crash AFTER a tile marker and BEFORE the next tile lets a NEW
  attempt reopen the Cell and skip the already reduced tiles;
* corrupt / truncated / torn / hash-mismatched checkpoints recompute ONLY that
  tile, preserving every other checkpoint;
* incompatible source order / request / gauge / tile geometry are refused
  loudly without erasing the stored checkpoints;
* live/stale owner handling and dense-mask refusal;
* sink counters are restored exactly once (no double count, no lost count);
* the PRODUCT caller ``zemosaic_zegrid_mode._run_cell_external`` is exercised,
  not only the low-level helper.
"""

from __future__ import annotations

import json
import hashlib
import multiprocessing as mp
import os
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    FixedNormalization,
    InMemoryCanonicalProvider,
    _iter_tiles,
    _phase1,
)
from zemosaic.core.zegrid import external_stream as x
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import rejection_sink as zrej
from zemosaic.core.zegrid import run_store as zrs
from zemosaic.core.zegrid import run_publication as zpub
from zemosaic.core.zegrid import tile_checkpoints as ztcp


# ---------------------------------------------------------------------------
# Corpus / request / helpers
# ---------------------------------------------------------------------------

def _corpus(n=4, h=24, w=20, seed=7):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 120.0)
    yy, xx = np.mgrid[0:h, 0:w]
    base += 20.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 5.0 ** 2))
    gains = np.linspace(0.97, 1.03, n)
    frames = []
    for i in range(n):
        frame = np.stack(
            [base * gains[i] * (0.99 + 0.01 * c) + rng.normal(0, 1.0, (h, w))
             for c in range(3)], axis=-1,
        )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    frames[1][4:9, 5:10, :] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[3][:, w // 2:] = False
    return frames, masks


def _request(frames, masks, *, sigma_low=3.0):
    return CanonicalStackRequest(
        images=frames, geometric_support=masks, normalization="sky_mean",
        weighting="noise_variance", rejection="winsorized_sigma_clip",
        combine="mean", taper="footprint", taper_px=2.0,
        sigma_low=sigma_low, sigma_high=3.0,
    )


class _FrameIdProvider:
    """In-memory provider with an explicit frame order (source identity)."""

    def __init__(self, inner, frame_ids):
        self._inner = inner
        self.frame_ids = tuple(str(v) for v in frame_ids)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def close(self):
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            return closer()
        return None


def _provider(frames, masks, frame_ids=None):
    inner = InMemoryCanonicalProvider(frames, masks)
    if frame_ids is None:
        frame_ids = [f"f{i}.fits" for i in range(len(frames))]
    return _FrameIdProvider(inner, frame_ids)


def _fixed(p1):
    return FixedNormalization(
        reference_index=int(p1.reference_index),
        coefficients=np.array(p1.coefficients, copy=True),
        norm_active=np.array(p1.norm_active, copy=True),
        weights=np.array(p1.weights, copy=True),
        weight_active=np.array(p1.weight_active, copy=True),
        exclusions=tuple(p1.exclusions),
    )


def _source_digest(frames, masks, *, label="v1"):
    """Deterministic exact-content digest for the small InMemory witness."""
    digest = hashlib.sha256()
    digest.update(str(label).encode("utf-8"))
    for frame, mask in zip(frames, masks):
        arr = np.ascontiguousarray(frame)
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
        digest.update(arr.tobytes(order="C"))
        m = np.ascontiguousarray(mask, dtype=bool)
        digest.update(m.tobytes(order="C"))
    return digest.hexdigest()


def _run(
    root, frames, masks, *, cell="cellA", tile_size=10, halo=2,
    checkpoint=False, fixed=None, request=None, frame_ids=None,
    sink=None, column_batch=64, source_digest=None, checkpoint_out=None,
    gpu_probe=None,
):
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "spool").mkdir(parents=True, exist_ok=True)
    provider = _provider(frames, masks, frame_ids=frame_ids)
    req = request if request is not None else _request(frames, masks)
    sink = sink if sink is not None else zrej.BoundedCountingRejectionSink(
        (len(frames), frames[0].shape[0], frames[0].shape[1], 3)
    )
    kwargs = {}
    if checkpoint:
        kwargs = {
            "checkpoint_root": str(Path(root) / "checkpoints"),
            "checkpoint_run_id": "run-stable-1",
            "checkpoint_source_digest": (
                _source_digest(frames, masks)
                if source_digest is None else str(source_digest)
            ),
        }
        if checkpoint_out is not None:
            kwargs["checkpoint_out"] = checkpoint_out
    if gpu_probe is not None:
        kwargs["gpu_probe"] = gpu_probe
    return x.run_cell_streaming_external(
        provider, req,
        tile_size=tile_size, column_batch=column_batch,
        spool_root=str(Path(root) / "spool"),
        fixed=fixed, rejection_sink=sink, cell_id=cell, attempt_id="attempt",
        cleanup_on_success=True, owner_token="OWNER",
        owner_record={"pid": 1}, **kwargs,
    ), sink


def _planes(result):
    return {
        name: np.array(getattr(result, name), copy=True)
        for name in (
            "science", "estimator_weight_sum", "valid_mask",
            "surviving_sample_count", "support_w1", "support_w2",
            "n_eff_support",
        )
    }


def _assert_stack_equal(left, right):
    for name, value in _planes(left).items():
        np.testing.assert_array_equal(value, _planes(right)[name], err_msg=name)
    for field in (
        "iterations_used", "initial_sample_count", "rejected_sample_count",
        "low_n_cell_count", "degenerate_cell_count",
    ):
        assert getattr(left, field) == getattr(right, field), field


def _sink_stats(sink):
    return {
        "true_count": int(sink.true_count),
        "written_cells": int(sink.written_cells),
        "writes": int(sink.writes),
        "blocks": int(sink.blocks),
        "max_write_bytes": int(sink.max_write_bytes),
        "max_block_bytes": int(sink.max_block_bytes),
        "flush_count": int(sink.flush_count),
    }


def _count_tiles(h, w, tile_size):
    return sum(1 for _ in _iter_tiles(h, w, tile_size))


# ---------------------------------------------------------------------------
# 1. Uninterrupted checkpointed run is bit-exact and reduced-only
# ---------------------------------------------------------------------------

def test_uninterrupted_checkpointed_run_is_bit_exact(tmp_path):
    frames, masks = _corpus()
    ref, sink_ref = _run(tmp_path / "ref", frames, masks)
    got, sink_got = _run(tmp_path / "cp", frames, masks, checkpoint=True)
    _assert_stack_equal(ref, got)
    assert _sink_stats(sink_ref) == _sink_stats(sink_got)

    cell_dir = Path(tmp_path) / "cp" / "checkpoints" / "cellA"
    manifest = json.loads((cell_dir / ztcp.CELL_MANIFEST_NAME).read_text())
    assert manifest["schema"] == ztcp.CELL_MANIFEST_SCHEMA
    assert manifest["run_id"] == "run-stable-1"
    tiles = sorted((cell_dir / "tiles").iterdir())
    assert len(tiles) == _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)
    # Only the reduced interior is stored (no N-axis tile pool).
    for tile_dir in tiles:
        marker = json.loads((tile_dir / ztcp.TILE_MARKER_NAME).read_text())
        h_int = marker["bounds"][1] - marker["bounds"][0]
        w_int = marker["bounds"][3] - marker["bounds"][2]
        assert marker["arrays"]["science"]["shape"] == [h_int, w_int, 3]
        assert marker["arrays"]["support_w1"]["shape"] == [h_int, w_int]
        assert marker["payload"]["bytes"] < (
            len(frames) * h_int * w_int * 3 * 8
        )
        assert not (tile_dir / "images.npy").exists()


# ---------------------------------------------------------------------------
# 2. Crash AFTER tile K, new attempt skips the completed tiles
# ---------------------------------------------------------------------------

def test_crash_after_marker_resume_skips_completed_tiles(
    tmp_path, monkeypatch
):
    frames, masks = _corpus()
    ref, sink_ref = _run(tmp_path / "ref", frames, masks)
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)
    crash_at = 3

    real_reduce = x.reduce_spooled_tile
    calls = []

    def _crashing_reduce(*args, **kwargs):
        calls.append(1)
        if len(calls) == crash_at + 1:
            raise RuntimeError("synthetic crash after tile marker")
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(x, "reduce_spooled_tile", _crashing_reduce)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _run(tmp_path / "cp", frames, masks, checkpoint=True)
    assert len(calls) == crash_at + 1
    cell_dir = Path(tmp_path) / "cp" / "checkpoints" / "cellA"
    committed = sorted(
        d.name for d in (cell_dir / "tiles").iterdir()
        if (d / ztcp.TILE_MARKER_NAME).exists()
    )
    assert len(committed) == crash_at

    # NEW attempt: no crash, count reductions and provider reads.
    calls.clear()
    reduce_calls = []

    def _counting_reduce(*args, **kwargs):
        reduce_calls.append(1)
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting_reduce)
    got, sink_got = _run(tmp_path / "cp", frames, masks, checkpoint=True)
    assert len(reduce_calls) == total - crash_at
    _assert_stack_equal(ref, got)
    assert _sink_stats(sink_ref) == _sink_stats(sink_got)

    # The existing checkpoints survived the failed attempt (no full purge).
    assert len([
        d for d in (cell_dir / "tiles").iterdir()
        if (d / ztcp.TILE_MARKER_NAME).exists()
    ]) == total


# ---------------------------------------------------------------------------
# 3. Corrupt / torn / hash-mismatched tile -> recompute ONLY that tile
# ---------------------------------------------------------------------------

def _full_checkpoint_run(tmp_path, frames, masks):
    ref, sink_ref = _run(tmp_path / "ref", frames, masks)
    got, sink_got = _run(tmp_path / "cp", frames, masks, checkpoint=True)
    _assert_stack_equal(ref, got)
    return ref, sink_ref


@pytest.mark.parametrize("damage", ["truncate_payload", "remove_marker", "flip_byte"])
def test_corrupt_single_tile_recomputed_only(tmp_path, monkeypatch, damage):
    frames, masks = _corpus()
    ref, sink_ref = _full_checkpoint_run(tmp_path, frames, masks)
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)
    cell_dir = Path(tmp_path) / "cp" / "checkpoints" / "cellA"
    victim = cell_dir / "tiles" / "t000002"
    assert victim.exists()
    if damage == "truncate_payload":
        payload = victim / ztcp.TILE_PAYLOAD_NAME
        payload.write_bytes(payload.read_bytes()[:10])
    elif damage == "remove_marker":
        (victim / ztcp.TILE_MARKER_NAME).unlink()
    else:
        payload = victim / ztcp.TILE_PAYLOAD_NAME
        raw = bytearray(payload.read_bytes())
        raw[0] ^= 0xFF
        payload.write_bytes(bytes(raw))
    # The other checkpoint directories are untouched and still committed.
    others = [
        d for d in (cell_dir / "tiles").iterdir() if d.name != victim.name
    ]
    assert all((d / ztcp.TILE_MARKER_NAME).exists() for d in others)

    real_reduce = x.reduce_spooled_tile
    calls = []

    def _counting_reduce(*args, **kwargs):
        calls.append(1)
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting_reduce)
    got, sink_got = _run(tmp_path / "cp", frames, masks, checkpoint=True)
    # Marker-less tile is treated as absent; torn/hash tiles are recomputed.
    assert len(calls) == 1
    _assert_stack_equal(ref, got)
    assert _sink_stats(sink_ref) == _sink_stats(sink_got)
    # No full purge: every other tile is still present and valid.
    for path in others:
        assert (path / ztcp.TILE_MARKER_NAME).exists()


# ---------------------------------------------------------------------------
# 4. Incompatible contracts refuse loudly, files preserved
# ---------------------------------------------------------------------------

def _expect_refusal(tmp_path, frames, masks, *, request=None, fixed=None,
                    frame_ids=None, tile_size=10, column_batch=64,
                    source_digest=None):
    cell_dir = Path(tmp_path) / "cp" / "checkpoints" / "cellA"
    ignored = {"OWNER.json", "LOCK"}
    before = {
        p.relative_to(cell_dir): p.read_bytes()
        for p in cell_dir.rglob("*")
        if p.is_file() and p.name not in ignored
    }
    with pytest.raises(ztcp.TileCheckpointIncompatible):
        _run(
            tmp_path / "cp", frames, masks, checkpoint=True,
            request=request, fixed=fixed, frame_ids=frame_ids,
            tile_size=tile_size, column_batch=column_batch,
            source_digest=source_digest,
        )
    after = {
        p.relative_to(cell_dir): p.read_bytes()
        for p in cell_dir.rglob("*")
        if p.is_file() and p.name not in ignored
    }
    assert before == after, "an incompatible refusal must never erase or rewrite"


def test_incompatible_request_gauge_source_order_and_geometry_refused(
    tmp_path,
):
    frames, masks = _corpus()
    _full_checkpoint_run(tmp_path, frames, masks)
    provider = _provider(frames, masks)
    base_req = _request(frames, masks)
    p1 = _phase1(provider, base_req)

    # Request (science) change.
    _expect_refusal(
        tmp_path, frames, masks,
        request=_request(frames, masks, sigma_low=2.5),
    )
    # Gauge change (actual phase-1 arrays).
    fixed_b = _fixed(p1)
    fixed_b.coefficients[0, 0, 0] *= 1.5
    _expect_refusal(tmp_path, frames, masks, fixed=fixed_b)
    # Source order change.
    _expect_refusal(
        tmp_path, frames, masks,
        frame_ids=[f"f{i}.fits" for i in reversed(range(len(frames)))],
    )
    # Tile geometry change.
    _expect_refusal(tmp_path, frames, masks, tile_size=12)


def test_source_binding_binds_pixels_and_wcs_not_only_labels(tmp_path):
    """H1: same labels/shape but different pixels or WCS → different binding."""
    path = tmp_path / "f0.fits"
    path.write_bytes(b"pixels-v1")
    desc = zg.FrameDescriptor(
        frame_id=zg.FrameId("f0.fits"), source_path=str(path),
        shape_hw=(4, 4), wcs_header="WCS-A SIP=0", header_sha256="",
        instrument="",
    )
    canvas = SimpleNamespace(
        height=8, width=8, wcs_header="CANVAS-A", canvas_id="c",
        resolution_deg=0.1,
    )
    b1, reason = zz._checkpoint_source_binding([desc], canvas)
    assert reason is None and b1 is not None
    d1 = ztcp.source_binding_digest(b1)

    # Same file, same labels/WCS → same binding (deterministic).
    b1b, _ = zz._checkpoint_source_binding([desc], canvas)
    assert ztcp.source_binding_digest(b1b) == d1

    # Pixel content changes under the SAME name/shape → different binding.
    path.write_bytes(b"pixels-v2-different")
    b2, _ = zz._checkpoint_source_binding([desc], canvas)
    assert ztcp.source_binding_digest(b2) != d1

    # WCS/transform changes with the SAME pixels → different binding.
    path.write_bytes(b"pixels-v1")
    desc_wcs = zg.FrameDescriptor(
        frame_id=zg.FrameId("f0.fits"), source_path=str(path),
        shape_hw=(4, 4), wcs_header="WCS-B SIP=1", header_sha256="",
        instrument="",
    )
    b3, _ = zz._checkpoint_source_binding([desc_wcs], canvas)
    assert ztcp.source_binding_digest(b3) != d1


def test_source_binding_missing_wcs_fails_closed(tmp_path):
    path = tmp_path / "f0.fits"
    path.write_bytes(b"pixels")
    desc = zg.FrameDescriptor(
        frame_id=zg.FrameId("f0.fits"), source_path=str(path),
        shape_hw=(4, 4), wcs_header="", header_sha256="", instrument="",
    )
    canvas = SimpleNamespace(
        height=8, width=8, wcs_header="CANVAS", canvas_id="c",
        resolution_deg=0.1,
    )
    binding, reason = zz._checkpoint_source_binding([desc], canvas)
    assert binding is None
    assert reason and "WCS" in reason


def test_product_caller_disables_on_missing_patch_wcs(tmp_path, monkeypatch):
    frames, masks = _corpus()
    ref, _sink = _run(tmp_path / "ref", frames, masks)
    _product_spawn_install(frames, masks, monkeypatch=monkeypatch)
    (tmp_path / "cp-spool").mkdir(parents=True, exist_ok=True)
    binding = {
        "schema": ztcp.SOURCE_BINDING_SCHEMA,
        "frames": [{"order": 0, "content_sha256": _source_digest(frames, masks)}],
        "canvas": {"wcs_sha256": "canvas-wcs"},
    }
    patch = _product_patch(frames)
    del patch.patch_wcs_header
    info = {}
    _mt, sres = zz._run_cell_external(
        tmp_path / "cache", patch, SimpleNamespace(), None,
        tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "cp-spool"), cell_id="cellW",
        attempt_id="w1", storage_profile="bounded", storage_info=info,
        checkpoint_root=str(tmp_path / "cp-no-wcs"),
        checkpoint_source_binding=binding,
    )
    _assert_stack_equal(ref, sres.result)
    assert info.get("tile_checkpoints") is None


def test_source_binding_change_same_labels_refuses_reuse(tmp_path):
    frames, masks = _corpus()
    _run(tmp_path / "cp", frames, masks, checkpoint=True,
         source_digest="source-A")
    with pytest.raises(ztcp.TileCheckpointIncompatible) as exc:
        _run(tmp_path / "cp", frames, masks, checkpoint=True,
             source_digest="source-B")
    assert any(
        item["section"] == "source_binding" for item in exc.value.reasons
    )


def test_column_batch_change_refused_by_identity(tmp_path):
    frames, masks = _corpus()
    _run(tmp_path / "cp", frames, masks, checkpoint=True, column_batch=64)
    with pytest.raises(ztcp.TileCheckpointIncompatible) as exc:
        _run(tmp_path / "cp", frames, masks, checkpoint=True, column_batch=32)
    assert any(item["section"] == "execution" for item in exc.value.reasons)


def test_gpu_request_with_checkpoints_records_backend(tmp_path, monkeypatch):
    """ZM-M33 lot 3: GPU checkpoints are allowed and carry per-tile provenance.

    This is an explicitly CPU-MOCKED GPU dispatch: ``_resolve_backend`` returns
    ``gpu`` (so the executor takes the GPU branch) while ``_get_xp`` returns
    NumPy (so no real CUDA work happens). It proves the PRODUCT/executor
    integration and the persisted requested/effective backend, NOT a hardware
    GPU result.
    """
    import dataclasses
    import json as _json

    frames, masks = _corpus()
    req = dataclasses.replace(_request(frames, masks), backend="gpu")
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)
    out = {}
    ref, _ = _run(tmp_path / "ref", frames, masks)
    got, _sink = _run(
        tmp_path / "cp", frames, masks, checkpoint=True, request=req,
        checkpoint_out=out,
        gpu_probe={
            "available": True, "vram_total_bytes": 2 * 1024 ** 3,
            "vram_free_bytes": 2 * 1024 ** 3, "reason": None,
        },
    )
    for name, expected in _planes(ref).items():
        np.testing.assert_allclose(
            expected, getattr(got, name), rtol=1e-12, atol=0.0,
            equal_nan=True, err_msg=name,
        )
    assert out["requested_backend"] == "gpu"
    assert out["effective_backend"] == "gpu"
    assert out["sticky_cpu"] is False
    assert out["gpu_tiles"] > 0 and out["cpu_tiles"] == 0
    marker = _json.loads(
        (
            tmp_path / "cp" / "checkpoints" / "cellA" / "tiles"
            / "t000001" / "commit.json"
        ).read_text()
    )
    assert marker["backend"] == {"requested": "gpu", "effective": "gpu"}


# ---------------------------------------------------------------------------
# 5. Ownership: live owner refused, stale owner recovered, dense mask refused
# ---------------------------------------------------------------------------

def _identity(tmp_path, frames, masks, cell="cellA", tile_size=10,
              source_digest=None, column_batch=64):
    provider = _provider(frames, masks)
    req = _request(frames, masks)
    p1 = _phase1(provider, req)
    tiles = [(y0, y1, x0, x1) for (y0, y1, x0, x1) in _iter_tiles(
        frames[0].shape[0], frames[0].shape[1], tile_size
    )]
    max_ih = max(y1 - y0 for (y0, y1, _x0, _x1) in tiles)
    max_iw = max(x1 - x0 for (_y0, _y1, x0, x1) in tiles)
    resolved = x.resolve_column_batch(
        int(column_batch), interior_hw=(max_ih, max_iw), channels=3
    )
    return ztcp.build_tile_identity(
        provider=provider, request=req, p1=p1, cell_id=cell,
        tile_size=tile_size, halo_px=2, tiles=tiles, backend="cpu",
        source_binding_digest=(
            _source_digest(frames, masks)
            if source_digest is None else str(source_digest)
        ),
        column_batch=int(column_batch), resolved_batch=resolved,
    )


def test_live_owner_refused_and_stale_owner_recovered(tmp_path):
    frames, masks = _corpus()
    identity = _identity(tmp_path, frames, masks)
    cell_dir = Path(tmp_path) / "cp" / "cellA"
    cell_dir.mkdir(parents=True)
    live_owner = {
        "schema": "zemosaic-zegrid-store-lock-v1",
        "state": "held",
        "pid": int(os.getppid()),
        "process_start_token": zrs.process_start_token(os.getppid()),
        "boot_id": zrs.boot_id(),
        "host": socket.gethostname(),
    }
    (cell_dir / "OWNER.json").write_text(json.dumps(live_owner))
    with pytest.raises(ztcp.TileCheckpointLocked):
        ztcp.TileCheckpointStore(cell_dir, identity=identity).open()

    stale_owner = dict(live_owner)
    stale_owner["pid"] = 999999
    stale_owner["process_start_token"] = None
    (cell_dir / "OWNER.json").write_text(json.dumps(stale_owner))
    store = ztcp.TileCheckpointStore(cell_dir, identity=identity)
    store.open()
    try:
        assert store._lock.orphan_recovered is True
    finally:
        store.close()


def test_lock_is_released_after_tile_and_finalizer_failures(tmp_path, monkeypatch):
    """H3: the actual executor path releases the lock on MemoryError/finalizer."""
    frames, masks = _corpus()
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)

    real_write = x.ExternalSpool.write_tile

    def _memory_error(*_a, **_k):
        raise MemoryError("synthetic tile allocation failure")

    monkeypatch.setattr(x.ExternalSpool, "write_tile", _memory_error)
    with pytest.raises(MemoryError):
        _run(tmp_path / "cp", frames, masks, checkpoint=True)
    monkeypatch.setattr(x.ExternalSpool, "write_tile", real_write)
    # Same process, same root: a leaked StoreLock would raise TileCheckpointLocked.
    out = {}
    _run(tmp_path / "cp", frames, masks, checkpoint=True, checkpoint_out=out)
    assert out["runtime_complete"] is True

    real_finalize = x._finalize_phase2_outputs

    def _finalizer_error(*_a, **_k):
        raise RuntimeError("synthetic finalizer failure")

    monkeypatch.setattr(x, "_finalize_phase2_outputs", _finalizer_error)
    with pytest.raises(RuntimeError, match="finalizer"):
        _run(tmp_path / "cp2", frames, masks, checkpoint=True)
    monkeypatch.setattr(x, "_finalize_phase2_outputs", real_finalize)
    out2 = {}
    _run(tmp_path / "cp2", frames, masks, checkpoint=True, checkpoint_out=out2)
    # All tiles were checkpointed before the finalizer failure and are reused.
    assert out2["reused_tiles"] == total


def test_lock_is_released_when_checkpoint_out_update_raises(tmp_path):
    """Rework-2: the post-open dict update is inside the lock-owning try."""
    frames, masks = _corpus()
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)
    _run(tmp_path / "cp", frames, masks, checkpoint=True)

    class _ExplodingOut(dict):
        def __init__(self):
            super().__init__()
            self.fired = False

        def update(self, *args, **kwargs):
            if not self.fired:
                self.fired = True
                raise MemoryError("synthetic checkpoint_out update failure")
            return super().update(*args, **kwargs)

    with pytest.raises(MemoryError, match="checkpoint_out"):
        _run(
            tmp_path / "cp", frames, masks, checkpoint=True,
            checkpoint_out=_ExplodingOut(),
        )
    out = {}
    _run(tmp_path / "cp", frames, masks, checkpoint=True, checkpoint_out=out)
    assert out["reused_tiles"] == total


def test_checkpoint_write_failure_is_nonfatal_and_truthful(
    tmp_path, monkeypatch
):
    frames, masks = _corpus()
    ref, _sink_ref = _run(tmp_path / "ref", frames, masks)
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)

    def _fail_save(*_a, **_k):
        raise OSError("synthetic checkpoint disk failure")

    monkeypatch.setattr(ztcp.TileCheckpointStore, "save_tile", _fail_save)
    out = {}
    got, _sink = _run(
        tmp_path / "cp", frames, masks, checkpoint=True, checkpoint_out=out
    )
    _assert_stack_equal(ref, got)
    assert out["runtime_complete"] is True
    assert out["durable_complete"] is False
    assert out["saved_tiles"] == 0
    assert out["write_failures"] == total
    assert out["not_checkpointed_tiles"] == total


def test_dense_rejection_mask_with_checkpoints_is_refused(tmp_path):
    frames, masks = _corpus()
    provider = _provider(frames, masks)
    with pytest.raises(Exception, match="counting rejection sink"):
        x.run_cell_streaming_external(
            provider, _request(frames, masks),
            tile_size=10, column_batch=64,
            spool_root=str(tmp_path / "spool"),
            rejection_sink=None,
            cell_id="cellA", attempt_id="attempt",
            cleanup_on_success=True, owner_token="OWNER",
            owner_record={"pid": 1},
            checkpoint_root=str(tmp_path / "checkpoints"),
        )


def test_cleanup_only_after_commit_hook_respects_live_owner(tmp_path):
    frames, masks = _corpus()
    root = Path(tmp_path) / "cp" / "checkpoints"
    _run(tmp_path / "cp", frames, masks, checkpoint=True)
    cell_dir = root / "cellA"
    assert cell_dir.exists()

    # A live (held) owner must never be removed by the post-commit hook.
    live_owner = {
        "schema": "zemosaic-zegrid-store-lock-v1",
        "state": "held",
        "pid": int(os.getppid()),
        "process_start_token": zrs.process_start_token(os.getppid()),
        "boot_id": zrs.boot_id(),
        "host": socket.gethostname(),
    }
    (cell_dir / "OWNER.json").write_text(json.dumps(live_owner))
    state = {"checkpoint_root": str(root), "checkpoint_cleanup_failures": []}
    report = zz._checkpoint_cleanup_after_commit(state, "cellA")
    assert report["removed"] is False and report["reason"] == "live_owner"
    assert cell_dir.exists()
    assert state["checkpoint_cleanup_failures"][0]["reason"] == "live_owner"

    # A released owner (the real post-commit state) is removable.
    released = dict(live_owner, state="released")
    (cell_dir / "OWNER.json").write_text(json.dumps(released))
    state = {"checkpoint_root": str(root), "checkpoint_cleanup_failures": []}
    report = zz._checkpoint_cleanup_after_commit(state, "cellA")
    assert report["removed"] is True
    assert not cell_dir.exists()
    assert state["checkpoint_cleanup_failures"] == []


def test_spawn_close_then_parent_reopens_and_reads(tmp_path):
    """Windows-like spawn lifecycle: child writes+closes, parent reopens."""
    frames, masks = _corpus()
    identity = _identity(tmp_path, frames, masks)
    root = Path(tmp_path) / "spawn-cp" / "cellA"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_spawn_write_one_checkpoint,
        args=(str(root), identity),
    )
    proc.start()
    proc.join(120)
    assert proc.exitcode == 0

    # The child's owner record is stale now; the parent recovers it and reads
    # the exact reduced arrays it published.
    store = ztcp.TileCheckpointStore(root, identity=identity)
    store.open()
    try:
        loaded = store.load_tile(1, (0, 4, 0, 4))
        assert loaded is not None
        assert loaded["arrays"]["science"].shape == (4, 4, 3)
        assert loaded["arrays"]["science"].dtype == np.float32
        np.testing.assert_array_equal(
            loaded["arrays"]["support_w1"], np.full((4, 4), 2.5, dtype=np.float64)
        )
        assert loaded["counters"]["initial_count"] == 48
    finally:
        store.close()


def _spawn_write_one_checkpoint(root, identity):
    store = ztcp.TileCheckpointStore(Path(root), identity=identity)
    store.open()
    try:
        store.save_tile(
            1, (0, 4, 0, 4),
            arrays={
                "science": np.zeros((4, 4, 3), dtype=np.float32),
                "weight_sum": np.ones((4, 4, 3), dtype=np.float64),
                "valid_mask": np.ones((4, 4, 3), dtype=bool),
                "surviving": np.full((4, 4, 3), 3, dtype=np.int64),
                "support_w1": np.full((4, 4), 2.5, dtype=np.float64),
                "support_w2": np.full((4, 4), 6.25, dtype=np.float64),
            },
            interior_origin=(0, 0),
            interior_shape=(4, 4, 3),
            extended_bounds=(0, 6, 0, 6),
            counters={
                "tile_iters": 2, "initial_count": 48,
                "rejected_count": 4, "low_n_count": 0,
                "degenerate_count": 0,
            },
            batch_geometry={
                "column_batch": 64, "interior_hw": [4, 4],
                "rows_per_batch": 1, "cols_per_batch": 4,
                "n_blocks": 4, "max_block_positions": 12,
            },
            sink_delta={
                "true_count": 4, "written_cells": 48, "writes": 1,
                "blocks": 1, "max_write_bytes_after": 48,
                "max_block_bytes_after": 48,
            },
        )
        store.close()
    except BaseException:
        store.close()
        raise


def test_cpu_durable_commit_cleans_checkpoints_only_after_verify(tmp_path):
    root = tmp_path / "checkpoints"
    cell_dir = root / "cellA"
    cell_dir.mkdir(parents=True)
    (cell_dir / "checkpoint.json").write_text("{}")
    verified = []

    class _Store:
        run_id = "run-1"
        root = tmp_path / "store"

        def commit_and_release(self, registry, cell_id, **_kwargs):
            assert cell_dir.exists(), "checkpoint must exist until commit"
            return {"release": {}}

        def require_commit(self, cell_id, attempt_id):
            verified.append((cell_id, attempt_id))

    state = {
        "store": _Store(), "publishing": True, "commits": 0,
        "released_cells": 0, "released_bytes": 0,
        "degraded_reason": None, "commit_errors": [],
        "checkpoint_root": str(root),
        "checkpoint_cleanup_failures": [],
    }
    receipt = SimpleNamespace(attempt_id="attempt-1")
    zz._cpu_durable_commit(state, object(), "cellA", receipt, plan_generation=1)
    assert verified == [("cellA", "attempt-1")]
    assert not cell_dir.exists()
    assert state["commits"] == 1


def _plane_digests(result):
    planes = {}
    for name in (
        "science", "estimator_weight_sum", "valid_mask",
        "surviving_sample_count", "support_w1", "support_w2",
        "n_eff_support",
    ):
        arr = np.ascontiguousarray(getattr(result, name))
        planes[name] = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
    counters = {
        field: int(getattr(result, field))
        for field in (
            "iterations_used", "initial_sample_count", "rejected_sample_count",
            "low_n_cell_count", "degenerate_cell_count",
        )
    }
    return {"planes": planes, "counters": counters}


def _product_spawn_install(frames, masks, *, monkeypatch=None,
                           child_process=False):
    """Install the product-call doubles.

    Child process: direct assignment is the documented spawned-only path
    (``child_process=True``). Parent tests MUST pass a ``monkeypatch`` so every
    mutated module attribute (including the SHARED zstream/za/zmem modules) is
    restored by pytest teardown. A parent call without either is a hard error:
    the installer has no hidden auto-restore.
    """
    if monkeypatch is None and not child_process:
        raise RuntimeError(
            "_product_spawn_install must be given a monkeypatch in the parent "
            "process (or explicit child_process=True in a spawned worker)"
        )
    req = _request(frames, masks)

    def _make_provider(*_a, **_k):
        return _provider(frames, masks)

    def _build_request(*_a, **_k):
        return req

    def _no_pipeline(**_k):
        return (None, None, None, None)

    def _extract(patch, sres):
        return sres

    class _Sampler:
        def start(self):
            return None

        def mark(self, _label=None):
            return None

        def stop(self):
            return None

        def result(self):
            return None

    attrs = (
        (zz, "_make_cell_provider", _make_provider),
        (zz.zstream, "build_streaming_request", _build_request),
        (zz, "_resolve_m33_preparation_pipeline", _no_pipeline),
        (zz.za, "extract_minitile", _extract),
        (zz.zmem, "PeakMemorySampler", _Sampler),
    )
    if monkeypatch is not None:
        for owner, name, value in attrs:
            monkeypatch.setattr(owner, name, value)
        return
    for owner, name, value in attrs:
        setattr(owner, name, value)


def _product_spawn_worker(root, spool_root, fault_after, source_digest,
                          status_path):
    """Process A (fault_after) / B (None): real _run_cell_external, new PID."""
    calls = []
    try:
        frames, masks = _corpus()
        _product_spawn_install(frames, masks, child_process=True)
        real_reduce = x.reduce_spooled_tile

        def _reduce(*args, **kwargs):
            calls.append(1)
            if fault_after is not None and len(calls) > int(fault_after):
                raise RuntimeError("spawn product crash")
            return real_reduce(*args, **kwargs)

        x.reduce_spooled_tile = _reduce
        Path(spool_root).mkdir(parents=True, exist_ok=True)
        info = {}
        _mt, sres = zz._run_cell_external(
            Path(root) / "cache", _product_patch(frames),
            SimpleNamespace(), None, tile_size=10, column_batch=64,
            spool_root=str(spool_root), cell_id="cellS", attempt_id="a",
            storage_profile="bounded", storage_info=info,
            checkpoint_root=str(root), checkpoint_run_id="run-spawn",
            checkpoint_source_digest=str(source_digest),
        )
        payload = {
            "ok": True, "pid": int(os.getpid()), "calls": int(len(calls)),
            **_plane_digests(sres.result),
            "runtime": dict(info.get("tile_checkpoints") or {}),
        }
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        payload = {
            "ok": False, "pid": int(os.getpid()), "calls": int(len(calls)),
            "error": f"{type(exc).__name__}: {exc}",
        }
    Path(status_path).write_text(json.dumps(payload), encoding="utf-8")


def _run_product_in_spawn(root, spool_root, status, *, fault_after,
                          source_digest):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_product_spawn_worker,
        args=(
            str(root), str(spool_root), fault_after, str(source_digest),
            str(status),
        ),
    )
    proc.start()
    proc.join(180)
    assert proc.exitcode == 0, f"spawn worker exit {proc.exitcode}"
    return json.loads(Path(status).read_text(encoding="utf-8"))


def test_product_doubles_are_scoped_and_restored():
    """Rework-3: parent installs MUST be teardown-scoped, never global."""
    frames, masks = _corpus()
    originals = {
        "provider": zz._make_cell_provider,
        "request": zz.zstream.build_streaming_request,
        "pipeline": zz._resolve_m33_preparation_pipeline,
        "extract": zz.za.extract_minitile,
        "sampler": zz.zmem.PeakMemorySampler,
    }
    with pytest.MonkeyPatch.context() as mp:
        _product_spawn_install(frames, masks, monkeypatch=mp)
        assert zz._make_cell_provider is not originals["provider"]
        assert zz.zstream.build_streaming_request is not originals["request"]
        assert zz._resolve_m33_preparation_pipeline is not originals["pipeline"]
        assert zz.za.extract_minitile is not originals["extract"]
        assert zz.zmem.PeakMemorySampler is not originals["sampler"]
    assert zz._make_cell_provider is originals["provider"]
    assert zz.zstream.build_streaming_request is originals["request"]
    assert zz._resolve_m33_preparation_pipeline is originals["pipeline"]
    assert zz.za.extract_minitile is originals["extract"]
    assert zz.zmem.PeakMemorySampler is originals["sampler"]
    with pytest.raises(RuntimeError, match="monkeypatch"):
        _product_spawn_install(frames, masks)


def test_multiprocess_spawn_product_resume_new_pid(tmp_path, monkeypatch):
    """H5: process A aborts after 2 tiles; process B resumes only 4."""
    frames, masks = _corpus()
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)
    digest = _source_digest(frames, masks)
    root = tmp_path / "spawn-product"
    root.mkdir(parents=True)

    first = _run_product_in_spawn(
        root, tmp_path / "spool-a", tmp_path / "status-a.json",
        fault_after=2, source_digest=digest,
    )
    assert first["ok"] is False and "spawn product crash" in first["error"]
    assert first["calls"] == 3

    second = _run_product_in_spawn(
        root, tmp_path / "spool-b", tmp_path / "status-b.json",
        fault_after=None, source_digest=digest,
    )
    assert second["ok"] is True
    assert second["calls"] == total - 2
    assert second["runtime"]["reused_tiles"] == 2
    assert second["pid"] != first["pid"]
    assert second["pid"] != os.getpid()
    assert first["pid"] != os.getpid()

    # Reference through the SAME product caller in this process (no checkpoints).
    _product_spawn_install(frames, masks, monkeypatch=monkeypatch)
    for name in ("spool-ref",):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    _ref_mt, ref_sres = zz._run_cell_external(
        tmp_path / "cache-ref", _product_patch(frames), SimpleNamespace(),
        None, tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "spool-ref"), cell_id="cellS",
        attempt_id="ref", storage_profile="bounded", storage_info={},
    )
    reference = _plane_digests(ref_sres.result)
    assert second["planes"] == reference["planes"]
    assert second["counters"] == reference["counters"]


def test_product_caller_propagates_frozen_source_binding(tmp_path, monkeypatch):
    frames, masks = _corpus()
    _product_spawn_install(frames, masks, monkeypatch=monkeypatch)
    total = _count_tiles(frames[0].shape[0], frames[0].shape[1], 10)
    binding = {
        "schema": ztcp.SOURCE_BINDING_SCHEMA,
        "frames": [{"order": 0, "content_sha256": _source_digest(frames, masks)}],
        "canvas": {"wcs_sha256": "canvas-wcs"},
    }
    for name in ("cp-spool", "cp-spool2"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    info1 = {}
    zz._run_cell_external(
        tmp_path / "cache", _product_patch(frames), SimpleNamespace(), None,
        tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "cp-spool"), cell_id="cellB",
        attempt_id="b1", storage_profile="bounded", storage_info=info1,
        checkpoint_root=str(tmp_path / "product-binding-checkpoints"),
        checkpoint_run_id=None, checkpoint_source_binding=binding,
    )
    digest1 = info1["tile_checkpoints"]["source_binding_digest"]
    assert digest1
    info2 = {}
    zz._run_cell_external(
        tmp_path / "cache", _product_patch(frames), SimpleNamespace(), None,
        tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "cp-spool2"), cell_id="cellB",
        attempt_id="b2", storage_profile="bounded", storage_info=info2,
        checkpoint_root=str(tmp_path / "product-binding-checkpoints"),
        checkpoint_run_id=None, checkpoint_source_binding=binding,
    )
    assert info2["tile_checkpoints"]["source_binding_digest"] == digest1
    assert info2["tile_checkpoints"]["reused_tiles"] == total


# ---------------------------------------------------------------------------
# 7. PRODUCT caller integration: _run_cell_external, crash + resume
# ---------------------------------------------------------------------------

def _product_patch(frames, halo=2):
    h, w = frames[0].shape[:2]
    return SimpleNamespace(
        patch=SimpleNamespace(height=h, width=w),
        halo_px=halo,
        patch_wcs_header="PATCH-WCS-TEST",
    )


def test_product_run_cell_external_crash_and_resume(tmp_path, monkeypatch):
    frames, masks = _corpus()
    req = _request(frames, masks)
    h, w = frames[0].shape[:2]
    total = _count_tiles(h, w, 10)

    def _make_provider(*_a, **_k):
        return _provider(frames, masks)

    monkeypatch.setattr(zz, "_make_cell_provider", _make_provider)
    monkeypatch.setattr(zz.zstream, "build_streaming_request", lambda *_a, **_k: req)
    monkeypatch.setattr(zz, "_resolve_m33_preparation_pipeline",
                        lambda **_k: (None, None, None, None))
    monkeypatch.setattr(zz.za, "extract_minitile", lambda patch, sres: sres)

    class _Sampler:
        def start(self):
            return None

        def mark(self, _label=None):
            return None

        def stop(self):
            return None

        def result(self):
            return None

    monkeypatch.setattr(zz.zmem, "PeakMemorySampler", _Sampler)

    for name in ("ref_spool", "cp_spool", "cp_spool2"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)

    # Reference through the SAME product caller, no checkpoints.
    ref_mt, ref_sres = zz._run_cell_external(
        tmp_path / "cache", _product_patch(frames), SimpleNamespace(),
        None, tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "ref_spool"), cell_id="cellP",
        attempt_id="a1", storage_profile="bounded", storage_info={},
    )

    real_reduce = x.reduce_spooled_tile
    calls = []

    def _crashing_reduce(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("product synthetic crash")
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(x, "reduce_spooled_tile", _crashing_reduce)
    with pytest.raises(RuntimeError, match="product synthetic crash"):
        zz._run_cell_external(
            tmp_path / "cache", _product_patch(frames), SimpleNamespace(),
            None, tile_size=10, column_batch=64,
            spool_root=str(tmp_path / "cp_spool"), cell_id="cellP",
            attempt_id="a2", storage_profile="bounded", storage_info={},
            checkpoint_root=str(tmp_path / "product-checkpoints"),
            checkpoint_run_id="run-product-1",
            checkpoint_source_digest=_source_digest(frames, masks),
        )
    checkpoint_cell = (
        Path(tmp_path) / "product-checkpoints" / "cellP"
    )
    assert (checkpoint_cell / ztcp.CELL_MANIFEST_NAME).exists()

    reduce_calls = []

    def _counting_reduce(*args, **kwargs):
        reduce_calls.append(1)
        return real_reduce(*args, **kwargs)

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting_reduce)
    storage_info = {}
    _mt, resumed = zz._run_cell_external(
        tmp_path / "cache", _product_patch(frames), SimpleNamespace(),
        None, tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "cp_spool2"), cell_id="cellP",
        attempt_id="a3", storage_profile="bounded", storage_info=storage_info,
        checkpoint_root=str(tmp_path / "product-checkpoints"),
        checkpoint_run_id="run-product-1",
        checkpoint_source_digest=_source_digest(frames, masks),
    )
    # 2 tiles were committed before the crash: only the remaining ones reduce.
    assert len(reduce_calls) == total - 2
    _assert_stack_equal(ref_sres.result, resumed.result)
    report = storage_info["tile_checkpoints"]
    assert report["reused_tiles"] == 2
    assert report["runtime_complete"] is True
    assert report["durable_complete"] is False


def test_product_run_cell_external_dispatches_gpu_mock(tmp_path, monkeypatch):
    """ZM-M33 lot 3: the PRODUCT external caller forwards backend='gpu'.

    Explicitly CPU-mocked (``_resolve_backend -> gpu``, ``_get_xp -> numpy``):
    this proves the frozen task contract reaches the executor and is published
    truthfully, NOT a real-GPU result.
    """
    frames, masks = _corpus()
    h, w = frames[0].shape[:2]

    def _make_provider(*_a, **_k):
        return _provider(frames, masks)

    monkeypatch.setattr(zz, "_make_cell_provider", _make_provider)
    monkeypatch.setattr(
        zz.zstream, "build_streaming_request",
        lambda *_a, **_k: _request(frames, masks),
    )
    monkeypatch.setattr(
        zz, "_resolve_m33_preparation_pipeline",
        lambda **_k: (None, None, None, None),
    )
    monkeypatch.setattr(zz.za, "extract_minitile", lambda patch, sres: sres)

    class _Sampler:
        def start(self):
            return None

        def mark(self, _label=None):
            return None

        def stop(self):
            return None

        def result(self):
            return None

    monkeypatch.setattr(zz.zmem, "PeakMemorySampler", _Sampler)
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)

    (tmp_path / "spool").mkdir(parents=True, exist_ok=True)
    storage_info = {}
    _mt, _sres = zz._run_cell_external(
        tmp_path / "cache", _product_patch(frames), SimpleNamespace(),
        None, tile_size=10, column_batch=64,
        spool_root=str(tmp_path / "spool"), cell_id="cellGPU",
        attempt_id="g1", storage_profile="bounded",
        storage_info=storage_info, backend="gpu",
        device_probe={
            "available": True, "vram_total_bytes": 2 * 1024 ** 3,
            "vram_free_bytes": 2 * 1024 ** 3, "reason": None,
        },
    )
    ext = storage_info["external"]
    assert ext["backend"]["requested"] == "gpu"
    assert ext["backend"]["selected"] == "gpu"
    assert ext["backend"]["effective"] == "gpu"
    assert ext["backend"]["sticky_cpu"] is False
    assert int(ext["backend"]["gpu_tiles"]) > 0
    assert int(ext["backend"]["cpu_tiles"]) == 0
    assert storage_info["gpu_fallback"]["effective_backend"] == "gpu"


def test_checkpoint_binding_reuses_shared_verified_rows(monkeypatch):
    """Rework-3: shared frozen rows avoid a second corpus hash pass.

    The binding must NOT trust an arbitrary list either: the row count is
    validated against the frozen descriptors and a short/malformed list fails
    closed (binding None + loud reason).
    """
    descs = [
        SimpleNamespace(wcs_header="WCS-0", shape_hw=(4, 5)),
        SimpleNamespace(wcs_header="WCS-1", shape_hw=(4, 5)),
    ]
    canvas = SimpleNamespace(
        height=4, width=5, wcs_header="CANVAS", canvas_id="c",
        resolution_deg=0.001,
    )
    rows = [
        {"content_sha256": "a" * 64, "nbytes": 100, "order": 0,
         "role": "science", "frame_id": "f0"},
        {"content_sha256": "b" * 64, "nbytes": 100, "order": 1,
         "role": "science", "frame_id": "f1"},
    ]

    def _must_not_hash(_descs):
        raise AssertionError("shared rows must be used verbatim")

    monkeypatch.setattr(zz, "_r55_frame_input_rows", _must_not_hash)
    binding, reason = zz._checkpoint_source_binding(
        descs, canvas, frame_rows=rows
    )
    assert reason is None
    assert binding["frames"][0]["content_sha256"] == "a" * 64
    assert binding["frames"][1]["content_sha256"] == "b" * 64
    assert binding["canvas"]["wcs_sha256"]

    short, reason2 = zz._checkpoint_source_binding(
        descs, canvas, frame_rows=rows[:1]
    )
    assert short is None
    assert "do not match" in reason2


def _valid_checkpoint_root(tmp_path, frames, masks):
    base = tmp_path / "out"
    owned_root = base / ztcp.TILE_CHECKPOINT_DIRNAME
    cell_dir = ztcp.cell_directory(owned_root, "cellA")
    store = ztcp.TileCheckpointStore(
        cell_dir, identity=_identity(tmp_path, frames, masks), run_id="run-1"
    )
    store.open()
    store.close()
    return base, cell_dir


def test_publication_recognises_only_strictly_owned_checkpoint_residue(
    tmp_path
):
    """Rework-3: valid owned residue is not foreign; forged/links are."""
    frames, masks = _corpus()
    base, cell_dir = _valid_checkpoint_root(tmp_path, frames, masks)
    status = zpub.detect_previous_publication(base)
    _proof = ztcp.inspect_owned_checkpoint_root(
        base / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert ztcp.TILE_CHECKPOINT_DIRNAME not in status["foreign"]
    assert _proof["owned"] is True

    # Unknown content inside a Cell root => NOT owned / still foreign.
    (cell_dir / "evil.txt").write_text("user data")
    status = zpub.detect_previous_publication(base)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status["foreign"]

    # Malformed manifest => NOT owned.
    (cell_dir / "evil.txt").unlink()
    (cell_dir / ztcp.CELL_MANIFEST_NAME).write_text("{}")
    status = zpub.detect_previous_publication(base)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status["foreign"]

    # Symlinked root => NOT owned / foreign.
    base2 = tmp_path / "out2"
    base2.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (base2 / ztcp.TILE_CHECKPOINT_DIRNAME).symlink_to(target)
    status2 = zpub.detect_previous_publication(base2)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status2["foreign"]

    # A live owner protects even a structurally valid root.
    base3, cell3 = _valid_checkpoint_root(tmp_path / "third", frames, masks)
    owner = {
        "schema": "zemosaic-zegrid-store-lock-v1",
        "state": "held",
        "pid": int(os.getpid()),
        "process_start_token": zrs.process_start_token(os.getpid()),
        "boot_id": zrs.boot_id(),
        "host": socket.gethostname(),
    }
    (cell3 / "OWNER.json").write_text(json.dumps(owner))
    proof = ztcp.inspect_owned_checkpoint_root(
        base3 / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert proof["owned"] is False
    assert proof["reason"] == "live_owner"
    status3 = zpub.detect_previous_publication(base3)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status3["foreign"]


def test_cleanup_cell_removes_empty_owned_root(tmp_path):
    frames, masks = _corpus()
    base, _cell = _valid_checkpoint_root(tmp_path, frames, masks)
    owned_root = base / ztcp.TILE_CHECKPOINT_DIRNAME
    result = ztcp.cleanup_cell(owned_root, "cellA")
    assert result["removed"] is True
    assert result["root_removed"] is True
    assert not owned_root.exists()


@pytest.mark.parametrize("marker_value", [[], "x", 5, None, ["a"]])
def test_non_object_tile_marker_is_foreign_not_crash(
    tmp_path, marker_value
):
    """R3-1: any valid-JSON non-object tile marker is FOREIGN, never raises."""
    frames, masks = _corpus()
    base, cell_dir = _valid_checkpoint_root(tmp_path, frames, masks)
    tile_dir = cell_dir / "tiles" / "t000001"
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / ztcp.TILE_MARKER_NAME).write_text(
        json.dumps(marker_value), encoding="utf-8"
    )
    proof = ztcp.inspect_owned_checkpoint_root(
        base / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert proof["owned"] is False
    status = zpub.detect_previous_publication(base)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status["foreign"]


@pytest.mark.parametrize("child", ["tiles", "tmp"])
@pytest.mark.parametrize("target_kind", ["dangling", "real_dir", "real_file"])
def test_linked_tiles_or_tmp_is_foreign_even_when_dangling(
    tmp_path, child, target_kind
):
    """R3-2: symlinks are checked BEFORE existence; dangling links refused."""
    frames, masks = _corpus()
    base, cell_dir = _valid_checkpoint_root(tmp_path, frames, masks)
    current = cell_dir / child
    if current.exists():
        shutil.rmtree(current)
    target = tmp_path / f"target-{child}-{target_kind}"
    if target_kind == "real_dir":
        target.mkdir()
    elif target_kind == "real_file":
        target.write_text("x")
    # dangling: target does not exist
    current.symlink_to(target)
    proof = ztcp.inspect_owned_checkpoint_root(
        base / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert proof["owned"] is False
    assert proof["reason"] == "foreign_cell_content"
    status = zpub.detect_previous_publication(base)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status["foreign"]


@pytest.mark.parametrize("name", ["OWNER.json", "LOCK"])
@pytest.mark.parametrize("kind", ["symlink_file", "symlink_dangling", "directory"])
def test_owner_and_lock_controls_must_be_plain_files(tmp_path, name, kind):
    frames, masks = _corpus()
    base, cell_dir = _valid_checkpoint_root(tmp_path, frames, masks)
    control = cell_dir / name
    if control.exists() or control.is_symlink():
        control.unlink()
    if kind == "directory":
        control.mkdir()
    else:
        target = tmp_path / f"{name}-{kind}"
        if kind == "symlink_file":
            target.write_text("{}")
        control.symlink_to(target)
    proof = ztcp.inspect_owned_checkpoint_root(
        base / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert proof["owned"] is False
    status = zpub.detect_previous_publication(base)
    assert ztcp.TILE_CHECKPOINT_DIRNAME in status["foreign"]


def test_malformed_present_owner_is_foreign_but_missing_owner_is_allowed(tmp_path):
    frames, masks = _corpus()
    base, cell_dir = _valid_checkpoint_root(tmp_path, frames, masks)
    owner = cell_dir / "OWNER.json"
    owner.write_text("[]", encoding="utf-8")
    proof = ztcp.inspect_owned_checkpoint_root(
        base / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert proof["owned"] is False
    assert zpub.detect_previous_publication(base)["foreign"] == [
        ztcp.TILE_CHECKPOINT_DIRNAME
    ]
    # Missing owner is legitimate crash residue: still owned.
    owner.unlink()
    proof2 = ztcp.inspect_owned_checkpoint_root(
        base / ztcp.TILE_CHECKPOINT_DIRNAME
    )
    assert proof2["owned"] is True
