"""ZM-ZEGRID-R52-OOC Lot A — portable bounded storage mechanisms.

Covers the gates of mission ZM-ZEGRID-R52-OOC-STORAGE-LOT-A-20261025:

* A. Dense-vs-bounded-sink science parity (mono/RGB, none/kappa/winsorized,
  tile edges + NaN), including exact scalar aggregates and provenance.
* B. Bounded provider correctness vs ``MemmapCanonicalProvider`` (owned copies,
  truthful diagnostics, second-open failure, idempotent close, no live views).
* C. Counting sink with > 4 MiB write blocks (CPU forced; GPU when available).
* D. No ``O(N*H*W*C)`` bool residency in the sink path (shape-checking sink +
  allocation probe that proves the dense mask is / is not allocated).

No production dispatch is exercised or changed: the dense default remains the
only wired behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

import zemosaic.core.canonical_streaming as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid.rejection_sink import (
    SINK_BLOCK_BYTES,
    BoundedCountingRejectionSink,
)

REJECTIONS = ("none", "kappa_sigma", "winsorized_sigma_clip")


# ---------------------------------------------------------------------------
# Deterministic corpora (tile edges + NaN + partial support)
# ---------------------------------------------------------------------------

def _mono_corpus(n=7, h=48, w=48, seed=3):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(6):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        amp = 40.0 * rng.uniform(0.7, 1.3)
        base += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))
    gains = np.linspace(0.96, 1.04, n)
    offsets = np.linspace(-3.0, 3.0, n)
    frames = []
    for i in range(n):
        f = base * gains[i] + offsets[i] + rng.normal(0.0, 2.0, (h, w))
        frames.append(f.astype(np.float32))
    if n > 1:
        frames[1][4:9, 4:9] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    if n > 4:
        masks[4][:, w // 2:] = False
    if n > 5:
        frames[5] = np.full((h, w), np.nan, dtype=np.float32)
    return frames, masks


def _mono_corpus_with_outliers(n=15, h=32, w=32, seed=5):
    """Corpus whose outlier blocks REALLY reject.

    With the canonical mean/std kappa-sigma test, a lone outlier among N frames
    is only rejected when ``sqrt(N - 1) > kappa``; N=15 with 3.0-sigma limits
    clears that (a 7-frame corpus cannot, whatever the amplitude). Two outlier
    blocks (positive and negative) plus a partial support mask keep the
    rejection path exercised for both kappa and winsorized clipping.
    """
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(5):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        base += 30.0 * rng.uniform(0.7, 1.3) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2)
        )
    gains = np.linspace(0.97, 1.03, n)
    offsets = np.linspace(-2.0, 2.0, n)
    frames = [
        (base * gains[i] + offsets[i] + rng.normal(0.0, 1.5, (h, w))).astype(np.float32)
        for i in range(n)
    ]
    frames[0][6:10, 6:10] += np.float32(50.0)
    frames[1][20:24, 20:24] -= np.float32(50.0)
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[4][:, : w // 3] = False
    return frames, masks


def _rgb_corpus(n=6, h=40, w=40, seed=2):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0, 3, (h, w, 3)).astype(np.float32)
        f += np.array([50, 55, 60], np.float32) + np.float32(2.0 * i)
        f[0:3, :, :] = np.nan
        frames.append(f)
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[1][10:15, 10:15] = False
    masks[3][:, 20:] = False
    frames[min(4, n - 1)][5:8, 5:8, 1] = np.nan  # channel-specific NaN
    return frames, masks


def _request(images, **kw):
    params = dict(
        images=images,
        geometric_support=None,
        normalization="none",
        weighting="none",
        rejection="none",
        combine="mean",
        taper="none",
    )
    params.update(kw)
    return CanonicalStackRequest(**params)


def _assert_same_science(dense, sunk, sink):
    """All science/support/mask planes, scalars and provenance are identical."""
    for name in (
        "science",
        "estimator_weight_sum",
        "valid_mask",
        "surviving_sample_count",
        "support_w1",
        "support_w2",
        "n_eff_support",
    ):
        np.testing.assert_array_equal(
            getattr(dense, name), getattr(sunk, name), err_msg=name
        )
    for name in (
        "iterations_used",
        "initial_sample_count",
        "rejected_sample_count",
        "rejected_fraction",
        "low_n_cell_count",
        "degenerate_cell_count",
        "rejection_method",
        "n_frames",
        "height",
        "width",
        "channels",
        "original_mono",
        "original_ndim",
        "original_shape",
    ):
        assert getattr(dense, name) == getattr(sunk, name), name
    assert dense.provenance == sunk.provenance
    # A count-only sink must not pretend to expose the dense mask.
    assert dense.rejection_mask is not None
    assert sunk.rejection_mask is None
    assert sink.true_count == dense.rejected_sample_count
    assert sink.true_count == int(dense.rejection_mask.sum())
    assert sink.written_cells == int(np.prod(dense.rejection_mask.shape))
    assert sink.flush_count > 0


# ---------------------------------------------------------------------------
# Gate A — dense vs bounded-sink parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rej", REJECTIONS)
def test_sink_parity_mono(rej):
    frames, masks = _mono_corpus()
    req = _request(
        frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection=rej, combine="median", taper="footprint", taper_px=8.0,
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    dense = run_canonical_stack_streaming(provider, req, tile_size=11)
    sink = BoundedCountingRejectionSink((len(frames), 48, 48, 1))
    sunk = run_canonical_stack_streaming(
        provider, req, tile_size=11, rejection_sink=sink
    )
    _assert_same_science(dense, sunk, sink)


@pytest.mark.parametrize("rej", REJECTIONS)
def test_sink_parity_rgb_tile_edges_and_nan(rej):
    frames, masks = _rgb_corpus()
    req = _request(
        frames, geometric_support=masks,
        normalization="linear_fit", weighting="none",
        rejection=rej, combine="mean", taper="footprint", taper_px=6.0,
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    dense = run_canonical_stack_streaming(provider, req, tile_size=(7, 13))
    sink = BoundedCountingRejectionSink((len(frames), 40, 40, 3))
    sunk = run_canonical_stack_streaming(
        provider, req, tile_size=(7, 13), rejection_sink=sink
    )
    _assert_same_science(dense, sunk, sink)


def test_sink_parity_rejects_real_outliers():
    """The parity corpus must actually exercise non-zero rejection counts."""
    frames, masks = _mono_corpus_with_outliers()
    h, w = masks[0].shape
    provider = InMemoryCanonicalProvider(frames, masks)
    for rej in ("kappa_sigma", "winsorized_sigma_clip"):
        req = _request(
            frames, geometric_support=masks, rejection=rej,
            combine="mean", taper="none",
        )
        dense = run_canonical_stack_streaming(provider, req, tile_size=7)
        assert dense.rejected_sample_count > 0, rej
        sink = BoundedCountingRejectionSink((len(frames), h, w, 1))
        sunk = run_canonical_stack_streaming(
            provider, req, tile_size=7, rejection_sink=sink
        )
        _assert_same_science(dense, sunk, sink)


def test_sink_parity_with_fixed_gauge():
    frames, masks = _mono_corpus()
    req = _request(
        frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean", taper="none",
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    gauge = cs.compute_fixed_normalization(provider, req)
    dense = run_canonical_stack_streaming(provider, req, tile_size=10, fixed=gauge)
    sink = BoundedCountingRejectionSink((len(frames), 48, 48, 1))
    sunk = run_canonical_stack_streaming(
        provider, req, tile_size=10, fixed=gauge, rejection_sink=sink
    )
    _assert_same_science(dense, sunk, sink)


class TestSinkValidation:
    def test_wrong_shape_raises(self):
        frames, masks = _mono_corpus()
        req = _request(frames, geometric_support=masks, rejection="none")
        provider = InMemoryCanonicalProvider(frames, masks)
        sink = BoundedCountingRejectionSink((len(frames) - 1, 48, 48, 1))
        with pytest.raises(Exception, match="shape"):
            run_canonical_stack_streaming(provider, req, tile_size=8, rejection_sink=sink)

    def test_wrong_dtype_raises(self):
        frames, masks = _mono_corpus()
        req = _request(frames, geometric_support=masks, rejection="none")
        provider = InMemoryCanonicalProvider(frames, masks)

        class BadSink:
            shape = (len(frames), 48, 48, 1)
            dtype = np.dtype(np.uint8)

            def flush_batch(self):
                pass

        with pytest.raises(Exception, match="dtype"):
            run_canonical_stack_streaming(
                provider, req, tile_size=8, rejection_sink=BadSink()
            )

    def test_missing_flush_raises(self):
        frames, masks = _mono_corpus()
        req = _request(frames, geometric_support=masks, rejection="none")
        provider = InMemoryCanonicalProvider(frames, masks)

        class NoFlush:
            shape = (len(frames), 48, 48, 1)
            dtype = np.dtype(bool)

        with pytest.raises(Exception, match="flush_batch"):
            run_canonical_stack_streaming(
                provider, req, tile_size=8, rejection_sink=NoFlush()
            )

    def test_default_unchanged(self):
        frames, masks = _mono_corpus()
        req = _request(
            frames, geometric_support=masks, rejection="kappa_sigma",
            combine="mean", taper="none",
        )
        provider = InMemoryCanonicalProvider(frames, masks)
        implicit = run_canonical_stack_streaming(provider, req, tile_size=9)
        explicit = run_canonical_stack_streaming(
            provider, req, tile_size=9, rejection_sink=None
        )
        np.testing.assert_array_equal(implicit.rejection_mask, explicit.rejection_mask)
        assert implicit.provenance == explicit.provenance


# ---------------------------------------------------------------------------
# Gate B — bounded provider
# ---------------------------------------------------------------------------

def _write_cache(tmp_path, frames, masks, name="cache"):
    cache = tmp_path / name
    ids = [f"frame_{i}" for i in range(len(frames))]
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    return cache


@pytest.mark.parametrize("rgb", [False, True])
def test_bounded_provider_matches_memmap(tmp_path, rgb):
    frames, masks = (_rgb_corpus() if rgb else _mono_corpus())
    h, w = masks[0].shape
    cache = _write_cache(tmp_path, frames, masks)

    ref = zfp.MemmapCanonicalProvider(cache)
    got = zfp.BoundedMemmapCanonicalProvider(cache)
    try:
        assert got.storage_domain == "aligned-npy-memmap-bounded"
        assert got.n_frames == ref.n_frames
        assert (got.height, got.width, got.channels) == (ref.height, ref.width, ref.channels)
        assert got.original_mono == ref.original_mono
        assert got.frame_ids == ref.frame_ids

        for i in range(len(frames)):
            r_ref, s_ref = ref.get_raw_frame(i)
            r_got, s_got = got.get_raw_frame(i)
            np.testing.assert_array_equal(np.asarray(r_got), np.asarray(r_ref))
            np.testing.assert_array_equal(np.asarray(s_got), np.asarray(s_ref))
            # Owned plain ndarray, never a memmap view.
            assert isinstance(r_got, np.ndarray) and not isinstance(r_got, np.memmap)
            assert isinstance(s_got, np.ndarray) and not isinstance(s_got, np.memmap)

        for bounds in [(0, 10, 0, 10), (3, 31, 5, 22), (0, h, 0, w)]:
            t_ref, v_ref = ref.get_tile(2, *bounds)
            t_got, v_got = got.get_tile(2, *bounds)
            np.testing.assert_array_equal(t_got, t_ref)
            np.testing.assert_array_equal(v_got, v_ref)
    finally:
        got.close()
        ref.close()


def test_bounded_provider_diagnostics_truthful(tmp_path):
    frames, masks = _rgb_corpus(n=4, h=20, w=24)
    cache = _write_cache(tmp_path, frames, masks)
    prov = zfp.BoundedMemmapCanonicalProvider(cache)
    try:
        assert (prov.bytes_read, prov.open_count, prov.open_failures) == (0, 0, 0)
        raw, sup = prov.get_raw_frame(1)
        assert prov.open_count == 2
        assert prov.open_failures == 0
        assert prov.bytes_read == raw.nbytes + sup.nbytes

        before = prov.bytes_read
        tile, valid = prov.get_tile(1, 2, 9, 3, 11)
        assert prov.open_count == 4
        # bytes_read counts the cache bytes copied (raw tile + support), not the
        # prepared validity plane the provider derives afterwards.
        expected_tile_bytes = (9 - 2) * (11 - 3) * 3 * np.dtype(np.float32).itemsize
        expected_sup_bytes = (9 - 2) * (11 - 3) * np.dtype(bool).itemsize
        assert prov.bytes_read - before == expected_tile_bytes + expected_sup_bytes
        assert prov.bytes_read > before
    finally:
        prov.close()


def test_bounded_provider_second_open_failure_closes_first(tmp_path, monkeypatch):
    frames, masks = _mono_corpus(n=3, h=12, w=12)
    cache = _write_cache(tmp_path, frames, masks)
    real_loader = zfp._load_memmap
    opened = []

    def flaky(path):
        if str(path).endswith("_support.npy"):
            raise OSError("simulated support open failure")
        arr = real_loader(path)
        opened.append(arr)
        return arr

    monkeypatch.setattr(zfp, "_load_memmap", flaky)
    prov = zfp.BoundedMemmapCanonicalProvider(cache)
    with pytest.raises(OSError, match="simulated"):
        prov.get_raw_frame(0)
    assert prov.open_count == 1
    assert prov.open_failures == 1
    assert len(opened) == 1
    # The RGB mapping opened first MUST be closed on the error path.
    assert opened[0]._mmap.closed is True

    # And the provider still works once the transport recovers.
    monkeypatch.setattr(zfp, "_load_memmap", real_loader)
    raw, sup = prov.get_raw_frame(1)
    np.testing.assert_array_equal(raw, frames[1])
    np.testing.assert_array_equal(sup, masks[1])


def test_bounded_provider_close_idempotent_and_owned_copies(tmp_path):
    frames, masks = _mono_corpus(n=3, h=16, w=16)
    cache = _write_cache(tmp_path, frames, masks)
    prov = zfp.BoundedMemmapCanonicalProvider(cache)
    raw, sup = prov.get_raw_frame(0)
    tile, valid = prov.get_tile(0, 2, 10, 3, 12)

    prov.close()
    prov.close()  # idempotent, must not raise
    assert prov.closed is True

    # Owned copies stay valid and usable after close.
    np.testing.assert_array_equal(raw, frames[0])
    np.testing.assert_array_equal(sup, masks[0])
    raw[0, 0] = -123.0
    assert raw[0, 0] == -123.0
    assert valid.dtype == np.bool_

    with pytest.raises(RuntimeError, match="closed"):
        prov.get_raw_frame(0)
    with pytest.raises(RuntimeError, match="closed"):
        prov.get_tile(0, 0, 4, 0, 4)


def test_bounded_provider_serves_from_disk_without_live_handles(tmp_path):
    frames, masks = _mono_corpus(n=3, h=16, w=16)
    cache = _write_cache(tmp_path, frames, masks)
    prov = zfp.BoundedMemmapCanonicalProvider(cache)
    raw, _sup = prov.get_raw_frame(0)
    # Patching the served copy must not touch the on-disk cache.
    raw[:] = 0.0
    fresh = zfp.BoundedMemmapCanonicalProvider(cache)
    raw2, _ = fresh.get_raw_frame(0)
    np.testing.assert_array_equal(raw2, frames[0])
    prov.close()
    fresh.close()


# ---------------------------------------------------------------------------
# Gate C — > 4 MiB write blocks (CPU forced; GPU when available)
# ---------------------------------------------------------------------------

def test_counting_sink_handles_large_write_block():
    shape = (2, 1024, 1024, 3)  # 6 MiB bool block
    sink = BoundedCountingRejectionSink(shape)
    block = np.ones(shape, dtype=bool)
    block[0, :, :, :] = False
    sink[:, :, :, :] = block
    assert sink.max_write_bytes == block.nbytes > 4 * 1024 * 1024
    assert sink.max_block_bytes <= SINK_BLOCK_BYTES
    assert sink.blocks > 1
    assert sink.true_count == int(np.count_nonzero(block))
    assert sink.written_cells == block.size


def test_engine_cpu_write_block_exceeding_4mib():
    # RGB, single tile => interior write block = N*H*W*C bytes = ~4.7 MiB.
    n, h, w = 4, 640, 640
    frames, masks = _rgb_corpus(n=n, h=h, w=w, seed=7)
    req = _request(
        frames, geometric_support=masks, rejection="none", combine="mean",
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    dense = run_canonical_stack_streaming(provider, req, tile_size=None)
    sink = BoundedCountingRejectionSink((n, h, w, 3))
    sunk = run_canonical_stack_streaming(
        provider, req, tile_size=None, rejection_sink=sink
    )
    assert sink.max_write_bytes > 4 * 1024 * 1024
    assert sink.max_block_bytes <= SINK_BLOCK_BYTES
    _assert_same_science(dense, sunk, sink)


def test_gpu_streaming_with_sink_if_available():
    pytest.importorskip("cupy")
    from zemosaic.core.canonical_stacking import canonical_gpu_available

    if not canonical_gpu_available():
        pytest.skip("CuPy installed but no usable CUDA device")

    frames, masks = _rgb_corpus(n=5, h=64, w=64, seed=11)
    req = _request(
        frames, geometric_support=masks, normalization="sky_mean", rejection="kappa_sigma",
        combine="mean", taper="footprint", taper_px=4.0, backend="gpu",
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    sink = BoundedCountingRejectionSink((5, 64, 64, 3))
    gpu = run_canonical_stack_streaming(
        provider, req, tile_size=32, rejection_sink=sink
    )
    cpu_req = _request(
        frames, geometric_support=masks, normalization="sky_mean", rejection="kappa_sigma",
        combine="mean", taper="footprint", taper_px=4.0, backend="cpu",
    )
    cpu_sink = BoundedCountingRejectionSink((5, 64, 64, 3))
    cpu = run_canonical_stack_streaming(
        provider, cpu_req, tile_size=32, rejection_sink=cpu_sink
    )
    for name in ("science", "estimator_weight_sum", "valid_mask", "surviving_sample_count",
                 "support_w1", "support_w2", "n_eff_support"):
        np.testing.assert_array_equal(getattr(gpu, name), getattr(cpu, name), err_msg=name)
    assert gpu.initial_sample_count == cpu.initial_sample_count
    assert gpu.rejected_sample_count == cpu.rejected_sample_count
    assert sink.true_count == gpu.rejected_sample_count
    assert sink.max_block_bytes <= SINK_BLOCK_BYTES


# ---------------------------------------------------------------------------
# Gate D — no O(N*H*W*C) residency in the sink path
# ---------------------------------------------------------------------------

class _RecordingSink:
    """Shape-checking proxy: records every block the engine writes."""

    def __init__(self, shape):
        self.inner = BoundedCountingRejectionSink(shape)
        self.shape = self.inner.shape
        self.dtype = self.inner.dtype
        self.records = []

    def __setitem__(self, key, value):
        arr = np.asarray(value)
        self.records.append((tuple(arr.shape), int(arr.nbytes)))
        self.inner[key] = value

    def flush_batch(self):
        self.inner.flush_batch()


def test_sink_never_receives_full_frame_block():
    n, h, w = 6, 48, 48
    frames, masks = _mono_corpus(n=n, h=h, w=w)
    req = _request(
        frames, geometric_support=masks, rejection="kappa_sigma", combine="mean",
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    sink = _RecordingSink((n, h, w, 1))
    run_canonical_stack_streaming(provider, req, tile_size=16, rejection_sink=sink)
    full_frame_bytes = n * h * w * 1
    assert sink.records
    assert max(nbytes for _shape, nbytes in sink.records) < full_frame_bytes
    # Interior tiles of 16x16: each write is exactly N*16*16*1 bytes.
    assert {shape for shape, _nbytes in sink.records} == {(n, 16, 16, 1)}


def test_sink_path_does_not_allocate_dense_mask(monkeypatch):
    n, h, w = 6, 40, 40
    frames, masks = _rgb_corpus(n=n, h=h, w=w)
    req = _request(
        frames, geometric_support=masks, rejection="winsorized_sigma_clip", combine="mean",
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    full_shape = (n, h, w, 3)

    calls = []
    real_zeros = np.zeros

    def probe_zeros(shape, *args, **kwargs):
        try:
            calls.append((tuple(int(v) for v in shape), np.dtype(kwargs.get("dtype", args[0] if args else float))))
        except Exception:
            calls.append((shape, None))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(cs.np, "zeros", probe_zeros)

    # Control: the dense default DOES allocate the (N, H, W, C) bool plane.
    run_canonical_stack_streaming(provider, req, tile_size=10)
    assert any(shape == full_shape and dtype == np.dtype(bool) for shape, dtype in calls)

    # Sink path: no such allocation.
    calls.clear()
    sink = BoundedCountingRejectionSink(full_shape)
    run_canonical_stack_streaming(provider, req, tile_size=10, rejection_sink=sink)
    assert not any(shape == full_shape and dtype == np.dtype(bool) for shape, dtype in calls)
    assert sink.true_count >= 0


def test_counting_sink_rejects_bad_input():
    sink = BoundedCountingRejectionSink((2, 4, 4, 1))
    with pytest.raises(TypeError, match="bool"):
        sink[:, :, :, :] = np.ones((2, 4, 4, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        sink[:, 0:2, 0:2, :] = np.ones((2, 3, 3, 1), dtype=bool)
    with pytest.raises(TypeError, match="4-tuple"):
        sink[0] = np.ones((4, 4, 1), dtype=bool)
    sink.close()
    sink.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        sink[:, :, :, :] = np.ones((2, 4, 4, 1), dtype=bool)
    with pytest.raises(RuntimeError, match="closed"):
        sink.flush_batch()
