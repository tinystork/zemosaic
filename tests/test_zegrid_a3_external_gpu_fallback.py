"""A3 — external working-set reduction GPU→CPU transactional fallback (Q31).

Verifies the OOM-injected fallback contract on the disk-spooled external path:

* a classified CuPy OOM/CUDA error during the GPU reduction re-reduces the SAME
  spooled tile on CPU (bit-exact across backends), producing one validated
  result — no pixel lost, no pixel double-counted;
* a partial sink write made before the OOM is rolled back, so a replayed tile is
  counted exactly once;
* a science/programming error is NEVER reclassified as a GPU failure (it
  propagates, no silent fallback).

These are CPU-side transactionality witnesses: the physical-GPU execution/parity
half of Q31 lives in ``test_zegrid_a3_external_gpu_parity.py``. Skips cleanly
without CuPy.
"""
from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import InMemoryCanonicalProvider
from zemosaic.core.zegrid import external_stream as x
from zemosaic.core.zegrid.rejection_sink import BoundedCountingRejectionSink

# A classified CuPy OOM (module "cupy.*", name "OutOfMemoryError") so
# ``is_gpu_runtime_error`` classifies it as a runtime GPU failure worth a fallback.
_FakeCupyOOM = type("OutOfMemoryError", (Exception,), {"__module__": "cupy.cuda.memory"})

_PLANES = (
    "science",
    "estimator_weight_sum",
    "valid_mask",
    "surviving_sample_count",
    "support_w1",
    "support_w2",
    "n_eff_support",
)


def _corpus(n=9, h=24, w=24, seed=5):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = (
            rng.normal(0, 3, (h, w, 3)).astype(np.float32)
            + np.array([50, 55, 60], np.float32)
            + 2.0 * i
        )
        f[rng.random((h, w, 3)) < 0.1] = np.nan
        frames.append(f)
    frames[2][1, 1, :] = 1000.0  # outlier
    masks = [np.ones((h, w), dtype=bool)] * n
    masks[1][4:8, 4:8] = False
    masks[3][:, 15:] = False
    return frames, masks


def _run(arrays, masks, backend, tmp_path, *, sink=None, gpu_fallback_out=None, **kw):
    req = CanonicalStackRequest(
        images=arrays, geometric_support=masks, backend=backend, **kw
    )
    # ``run_cell_streaming_external`` requires an EXISTING spool root.
    tmp_path.mkdir(parents=True, exist_ok=True)
    return x.run_cell_streaming_external(
        InMemoryCanonicalProvider(arrays, masks),
        req,
        tile_size=16,
        column_batch=7,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id="a",
        rejection_sink=sink,
        gpu_fallback_out=gpu_fallback_out,
    )


def _assert_bit_exact(ref, got):
    assert ref.n_frames == got.n_frames
    assert ref.height == got.height and ref.width == got.width
    assert ref.channels == got.channels
    for name in _PLANES:
        a, b = getattr(ref, name), getattr(got, name)
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        if name in ("valid_mask", "surviving_sample_count") or a.dtype.kind in "iu":
            np.testing.assert_array_equal(a, b, err_msg=name)
        else:
            np.testing.assert_allclose(
                a, b, rtol=1e-12, atol=0.0, equal_nan=True, err_msg=name
            )
    assert ref.iterations_used == got.iterations_used
    assert ref.initial_sample_count == got.initial_sample_count
    assert ref.rejected_sample_count == got.rejected_sample_count
    assert ref.rejected_fraction == got.rejected_fraction
    assert ref.low_n_cell_count == got.low_n_cell_count
    assert ref.degenerate_cell_count == got.degenerate_cell_count


def _sink_counters(sink):
    return {
        "true_count": int(sink.true_count),
        "written_cells": int(sink.written_cells),
        "writes": int(sink.writes),
    }


def test_gpu_oom_falls_back_to_cpu_bit_exact_no_double_count(tmp_path, monkeypatch):
    """A GPU OOM → CPU replay is bit-exact and the sink is counted once."""
    arrays, masks = _corpus()

    # Reference: pure CPU with a fresh counting sink.
    ref_sink = BoundedCountingRejectionSink((len(arrays), 24, 24, 3))
    ref = _run(arrays, masks, "cpu", tmp_path / "ref", sink=ref_sink,
               normalization="sky_mean", weighting="noise_variance",
               rejection="winsorized_sigma_clip", combine="mean", taper="footprint")

    real_reduce = x.reduce_spooled_tile
    state = {"oom": True}

    def _oom_reduce(tile, p1, request, *, column_batch, rejection_sink=None,
                    rejection_mask=None, backend="cpu"):
        if backend == "gpu" and state["oom"]:
            state["oom"] = False
            raise _FakeCupyOOM("simulated device OOM during reduction")
        return real_reduce(
            tile, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _oom_reduce)

    got_sink = BoundedCountingRejectionSink((len(arrays), 24, 24, 3))
    fb = {}
    got = _run(arrays, masks, "gpu", tmp_path / "got", sink=got_sink,
               normalization="sky_mean", weighting="noise_variance",
               rejection="winsorized_sigma_clip", combine="mean", taper="footprint",
               gpu_fallback_out=fb)

    _assert_bit_exact(ref, got)
    # The fallback happened and is observable.
    assert fb["fallback_count"] >= 1
    assert fb["requested_backend"] == "gpu"
    # No double counting: the replayed tile(s) produced the same sink totals as
    # the pure-CPU reference.
    assert _sink_counters(got_sink) == _sink_counters(ref_sink)


def test_partial_sink_write_before_oom_is_rolled_back(tmp_path, monkeypatch):
    """A partial sink write made before the OOM is undone by the checkpoint."""
    arrays, masks = _corpus()
    n, c = len(arrays), 3

    ref_sink = BoundedCountingRejectionSink((n, 24, 24, c))
    ref = _run(arrays, masks, "cpu", tmp_path / "ref", sink=ref_sink,
               normalization="none", weighting="none",
               rejection="none", combine="mean", taper="none")

    real_reduce = x.reduce_spooled_tile
    state = {"oom": True}

    def _partial_oom_reduce(tile, p1, request, *, column_batch, rejection_sink=None,
                            rejection_mask=None, backend="cpu"):
        if backend == "gpu" and state["oom"] and rejection_sink is not None:
            state["oom"] = False
            # Simulate a few interior blocks written BEFORE the OOM: write one
            # bogus bool block at the tile's interior origin, then fail. The
            # executor's checkpoint/rollback must undo this so the CPU replay
            # counts the tile exactly once.
            rejection_sink[:, tile.y0:tile.y0 + 1, tile.x0:tile.x0 + 1, :] = \
                np.ones((n, 1, 1, c), dtype=bool)
            raise _FakeCupyOOM("simulated device OOM mid-reduction")
        return real_reduce(
            tile, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _partial_oom_reduce)

    got_sink = BoundedCountingRejectionSink((n, 24, 24, c))
    fb = {}
    got = _run(arrays, masks, "gpu", tmp_path / "got", sink=got_sink,
               normalization="none", weighting="none",
               rejection="none", combine="mean", taper="none",
               gpu_fallback_out=fb)

    _assert_bit_exact(ref, got)
    assert fb["fallback_count"] >= 1
    # The partial write was rolled back: sink totals match the pure-CPU path.
    assert _sink_counters(got_sink) == _sink_counters(ref_sink)


def test_science_error_is_not_reclassified_as_gpu_failure(tmp_path, monkeypatch):
    """A science/programming error propagates (no silent GPU→CPU fallback)."""
    arrays, masks = _corpus()

    def _boom(tile, p1, request, *, column_batch, rejection_sink=None,
              rejection_mask=None, backend="cpu"):
        if backend == "gpu":
            raise ValueError("science bug, NOT a GPU error")
        return x.reduce_spooled_tile(
            tile, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _boom)
    with pytest.raises(ValueError, match="science bug"):
        _run(arrays, masks, "gpu", tmp_path, normalization="none",
             weighting="none", rejection="none", combine="mean", taper="none")
