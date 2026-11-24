"""ZM-M33-EXT-LIFETIME — bounded lifetime of the external reduction outputs.

Two LOCAL, product-preserving lifetime fixes are witnessed here:

* ``run_cell_streaming_external`` used to keep the previous tile's reduction
  dict (four extended-size planes ``science``/``weight_sum``/``valid_mask``/
  ``surviving`` plus its scalars) alive in the loop-local ``tile`` for the whole
  next tile (write + reduce) AND through the finalize phase. The dict is now
  dropped right after every consumer (persistent cell planes, counters,
  executed geometry) has been fed.
* ``reduce_spooled_tile`` used to keep ``wi`` — the LAST active frame's
  ``(h_int, w_int)`` float64 support map — pinned in its frame for the entire
  micro-block loop. It is now dropped right after the support loop.

The witnesses are RED on the pre-fix baseline and GREEN on the fix:

1. ``test_tile_outputs_freed_before_next_reduce_and_before_finalize``: a
   weakref observer armed OUTSIDE the product sees the previous tile's dict and
   its four planes alive at the entry of the next reduction and at the entry of
   the finalization (baseline), and dead on the fix. The observer keeps ONLY
   weakrefs — never a strong reference that would fake the result.
2. ``test_reducer_holds_no_per_frame_weight_plane_during_blocks``: at EVERY
   ``_gather_block`` call (i.e. at the first gather and inside every block) the
   reducer's own frame is scanned for a live float64 plane bit-equal to a
   per-frame ``q[i] * w_i`` map. The scan is value-based (not a name/grep
   check), so it cannot be satisfied by renaming the local.

All parity checks below compare against the qualified in-RAM canonical path on
the SAME data/params, so "freed" never means "changed": science arrays, masks,
weights, support maps, n_eff and the exact counters stay identical.
"""

from __future__ import annotations

import dataclasses
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.canonical_stacking import CanonicalStackFailure
from zemosaic.core.zegrid import external_stream as xstream
from zemosaic.core.zegrid.rejection_sink import BoundedCountingRejectionSink


# ---------------------------------------------------------------------------
# Corpora / helpers
# ---------------------------------------------------------------------------

def _corpus(n=5, h=45, w=38, seed=3, mono=False):
    """RGB/mono corpus with NaN holes, a zero-weight frame and low-N strips.

    ``h``/``w`` are deliberately NOT multiples of the tile sizes used below so
    the last row/column tile is a remainder tile.
    """
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for _k in range(3):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        base += 20.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2))
    gains = np.linspace(0.97, 1.03, n)
    ch_gains = np.array([0.98, 1.0, 1.03], dtype=np.float64)
    frames = []
    for i in range(n):
        if mono:
            frame = base * gains[i] + rng.normal(0, 1.5, (h, w))
        else:
            frame = np.stack(
                [
                    base * gains[i] * ch_gains[ch] + rng.normal(0, 1.5, (h, w))
                    for ch in range(3)
                ],
                axis=-1,
            )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    # NaN holes (partial validity, identical mask on every channel).
    if mono:
        frames[1][3:9, 4:11] = np.nan
    else:
        frames[1][3:9, 4:11, :] = np.nan
    # A ZERO-WEIGHT frame: fully NaN -> excluded at normalization, q == 0.
    frames[-1][:, :] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    if n > 3:
        masks[3][:, w // 2:] = False
    for i in range(2, n):
        masks[i][:, :2] = False
    return frames, masks


def _request(frames, masks, *, rejection="winsorized_sigma_clip", combine="mean",
             taper="footprint", backend="cpu"):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection=rejection,
        combine=combine,
        taper=taper,
        taper_px=8.0,
        backend=backend,
    )


_PLANES = (
    "science",
    "estimator_weight_sum",
    "valid_mask",
    "surviving_sample_count",
    "support_w1",
    "support_w2",
    "n_eff_support",
    "rejection_mask",
)

_SCALARS = (
    "iterations_used",
    "initial_sample_count",
    "rejected_sample_count",
    "rejected_fraction",
    "low_n_cell_count",
    "degenerate_cell_count",
    "rejection_method",
)


def _assert_planes_exact(ref, got):
    assert ref.n_frames == got.n_frames
    assert ref.height == got.height and ref.width == got.width
    assert ref.channels == got.channels
    for name in _PLANES:
        a = getattr(ref, name)
        b = getattr(got, name)
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert np.array_equal(a, b, equal_nan=True), name
    for name in _SCALARS:
        assert getattr(ref, name) == getattr(got, name), name


def _assert_bit_exact(ref, got):
    _assert_planes_exact(ref, got)
    assert ref.provenance == got.provenance


def _run_external(frames, masks, root, *, tile_size=16, column_batch=6,
                  request=None, **kwargs):
    Path(root).mkdir(parents=True, exist_ok=True)
    return xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        _request(frames, masks) if request is None else request,
        tile_size=tile_size,
        column_batch=column_batch,
        spool_root=str(root),
        cell_id="cell",
        attempt_id="attempt",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# A. Tile-output lifetime: dead before the next reduce and before finalize
# ---------------------------------------------------------------------------

def test_tile_outputs_freed_before_next_reduce_and_before_finalize(
    tmp_path, monkeypatch
):
    frames, masks = _corpus(n=5, h=45, w=38)
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), _request(frames, masks),
        tile_size=16,
    )

    real_reduce = xstream.reduce_spooled_tile
    # Weakrefs ONLY: holding the dict/arrays strongly would keep them alive and
    # make the witness meaningless.
    state = {"refs": None}
    live_at_reduce_entry: list[int] = []
    volumes: list[tuple[int, int]] = []

    def _tracked(tile, p1, request, **kw):
        if state["refs"] is not None:
            live_at_reduce_entry.append(
                sum(1 for r in state["refs"] if r() is not None)
            )
        out = real_reduce(tile, p1, request, **kw)
        state["refs"] = [
            weakref.ref(out["science"]),
            weakref.ref(out["weight_sum"]),
            weakref.ref(out["valid_mask"]),
            weakref.ref(out["surviving"]),
        ]
        c = int(p1.channels)
        cells = int(tile.th) * int(tile.tw) * c
        volumes.append((
            int(out["science"].nbytes + out["weight_sum"].nbytes
                + out["valid_mask"].nbytes + out["surviving"].nbytes),
            cells * (4 + 8 + 1 + 8),  # float32/f64/bool/int64 per cell
        ))
        return out

    live_at_finalize: list[int] = []
    real_finalize = xstream._finalize_phase2_outputs

    def _finalize(*args, **kw):
        live_at_finalize.append(
            0 if state["refs"] is None
            else sum(1 for r in state["refs"] if r() is not None)
        )
        return real_finalize(*args, **kw)

    monkeypatch.setattr(xstream, "reduce_spooled_tile", _tracked)
    monkeypatch.setattr(xstream, "_finalize_phase2_outputs", _finalize)

    got = _run_external(frames, masks, tmp_path, tile_size=16, column_batch=6)

    # The fix must not move a single scientific value.
    _assert_bit_exact(ref, got)
    # 45x38 with tile 16 -> 3x3 tiles: several reductions were observed.
    assert len(live_at_reduce_entry) >= 4, live_at_reduce_entry
    # The freed volume per tile IS the four extended-size tile planes.
    assert volumes and all(actual == expected for actual, expected in volumes), (
        volumes[:3]
    )
    # Baseline: 4 planes of the previous tile still alive at every next entry.
    assert live_at_reduce_entry == [0] * len(live_at_reduce_entry), (
        f"previous tile still alive at next reduce: {live_at_reduce_entry}"
    )
    assert live_at_finalize == [0], (
        f"last tile still alive at finalize: {live_at_finalize}"
    )


# ---------------------------------------------------------------------------
# B. ``wi``: no live per-frame weight plane at the first gather / in blocks
# ---------------------------------------------------------------------------

def test_reducer_holds_no_per_frame_weight_plane_during_blocks(
    tmp_path, monkeypatch
):
    frames, masks = _corpus(n=5, h=45, w=38)
    real_gather = xstream._gather_block
    observations: list[list[int]] = []

    def _probe(images, weights, n, tw, c, iy0, ix0, w_int, a, b, x0, x1):
        # Inspect the REDUCER frame itself from the outside (test-only probe).
        loc = sys._getframe(1).f_locals
        p1 = loc["p1"]
        h_int = int(loc["h_int"])
        q = np.asarray(p1.weights, dtype=np.float64)
        active = np.asarray(p1.weight_active, dtype=bool)
        held: list[int] = []
        for i in np.flatnonzero(active):
            expected = q[i] * np.asarray(
                weights[i][iy0:iy0 + h_int, ix0:ix0 + w_int], dtype=np.float64
            )
            for value in loc.values():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype == np.float64
                    and value.shape == expected.shape
                    and np.array_equal(value, expected)
                ):
                    held.append(int(i))
                    break
        observations.append(held)
        return real_gather(
            images, weights, n, tw, c, iy0, ix0, w_int, a, b, x0, x1
        )

    monkeypatch.setattr(xstream, "_gather_block", _probe)
    got = _run_external(frames, masks, tmp_path, tile_size=16, column_batch=5)

    assert got.science.shape == (45, 38, 3)
    assert len(observations) >= 2, "expected several micro-blocks"
    # Baseline: the last active frame's map is present at EVERY gather.
    assert observations == [[] for _ in observations], (
        "a per-frame weight plane is still referenced by the reducer frame: "
        f"{[o for o in observations if o][:3]}"
    )


def test_reducer_with_inactive_frames_is_bit_exact(tmp_path):
    """Mixed active/inactive frames: the support loop binds ``wi`` only for the
    active ones; parity must be untouched."""
    frames, masks = _corpus(n=6, h=32, w=28)
    frames[-1][:, :] = np.nan  # weight-inactive
    frames[-2][10:14, 10:14] = np.nan
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), _request(frames, masks),
        tile_size=16,
    )
    got = _run_external(
        frames, masks, tmp_path, tile_size=16, column_batch=4, request=_request(frames, masks)
    )
    _assert_bit_exact(ref, got)


def test_all_inactive_frames_are_still_refused_by_phase1(tmp_path):
    """The product FORBIDS an all-inactive set at weighting time; the lifetime
    fix must not replace that typed refusal with a NameError/UnboundLocalError."""
    frames, masks = _corpus(n=3, h=24, w=20)
    blind = [np.zeros((24, 20), dtype=bool) for _ in range(3)]
    with pytest.raises(CanonicalStackFailure):
        xstream.run_cell_streaming_external(
            InMemoryCanonicalProvider(frames, blind),
            _request(frames, blind),
            tile_size=16,
            column_batch=4,
            spool_root=str(tmp_path),
            cell_id="c",
            attempt_id="a",
        )


def test_reducer_tolerates_a_constructed_all_inactive_phase1(tmp_path):
    """Direct reducer call with ``weight_active`` all False: the support loop
    never binds ``wi``, so the post-loop rebind must not raise."""
    frames, masks = _corpus(n=3, h=24, w=20)
    req = _request(frames, masks, rejection="none", combine="mean")
    provider = InMemoryCanonicalProvider(frames, masks)
    p1 = xstream._fixed_phase1(provider, req, None)
    p1_off = dataclasses.replace(
        p1, weight_active=np.zeros(len(frames), dtype=bool)
    )

    taper_kind, taper_px, taper_floor, halo = xstream._resolve_taper(req)
    spec = xstream.ExternalSpoolSpec(
        root=str(tmp_path),
        n_frames=len(frames),
        height=24,
        width=20,
        channels=3,
        halo_px=halo,
        tile_size=None,
        cell_id="c",
        attempt_id="a",
    )
    spool = xstream.ExternalSpool(spec)
    handle = spool.write_tile(
        provider, p1, 0, 24, 0, 20,
        taper_kind=taper_kind, taper_px=taper_px,
        taper_floor=taper_floor, halo_px=halo,
    )
    rejection_mask = np.zeros((len(frames), 24, 20, 3), dtype=bool)
    out = xstream.reduce_spooled_tile(
        handle, p1_off, req,
        column_batch=4,
        rejection_mask=rejection_mask,
        backend="cpu",
    )
    assert np.isnan(out["science"]).all()
    assert np.array_equal(out["weight_sum"], np.zeros((24, 20, 3)))
    assert not out["valid_mask"].any()
    assert np.array_equal(out["surviving"], np.zeros((24, 20, 3), dtype=np.int64))
    assert np.array_equal(out["support_w1"], np.zeros((24, 20)))
    assert out["initial_count"] == 0 and out["rejected_count"] == 0
    assert out["batch_geometry"]["n_blocks"] > 0
    spool.cleanup()


# ---------------------------------------------------------------------------
# C. Parity on the same data / backend / params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile", (24, 16))
@pytest.mark.parametrize("rejection", ("none", "winsorized_sigma_clip"))
@pytest.mark.parametrize("combine", ("mean", "median"))
@pytest.mark.parametrize("column_batch", (5, 1024))
def test_parity_matrix_rgb_multitile_remainder(
    tmp_path, tile, rejection, combine, column_batch
):
    frames, masks = _corpus(n=6, h=45, w=38)
    req = _request(frames, masks, rejection=rejection, combine=combine)
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=tile
    )
    got = _run_external(
        frames, masks, tmp_path, tile_size=tile, column_batch=column_batch,
        request=req,
    )
    _assert_bit_exact(ref, got)


@pytest.mark.parametrize("tile", (32, 16))
def test_parity_mono_uses_the_shared_reducer(tmp_path, tile):
    frames, masks = _corpus(n=5, h=40, w=34, mono=True)
    req = _request(frames, masks, rejection="kappa_sigma", combine="median")
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=tile
    )
    got = _run_external(
        frames, masks, tmp_path, tile_size=tile, column_batch=7, request=req
    )
    _assert_bit_exact(ref, got)
    assert got.science.ndim == 2


def test_parity_single_full_tile_interior_equals_extended(tmp_path):
    frames, masks = _corpus(n=5, h=40, w=36)
    req = _request(frames, masks)
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=None
    )
    got = _run_external(
        frames, masks, tmp_path, tile_size=None, column_batch=1024, request=req
    )
    _assert_bit_exact(ref, got)


@pytest.mark.parametrize("tile", (24, 16))
def test_parity_count_only_rejection_sink(tmp_path, tile):
    frames, masks = _corpus(n=8, h=50, w=44)
    req = _request(frames, masks)
    provider = InMemoryCanonicalProvider(frames, masks)
    dense = run_canonical_stack_streaming(provider, req, tile_size=tile)

    sink_ref = BoundedCountingRejectionSink((len(frames), 50, 44, 3))
    ref = run_canonical_stack_streaming(
        provider, req, tile_size=tile, rejection_sink=sink_ref
    )
    sink_got = BoundedCountingRejectionSink((len(frames), 50, 44, 3))
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    got = xstream.run_cell_streaming_external(
        provider,
        req,
        tile_size=tile,
        column_batch=5,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id="a",
        rejection_sink=sink_got,
    )
    _assert_bit_exact(ref, got)
    assert got.rejection_mask is None and ref.rejection_mask is None
    assert int(sink_got.true_count) == int(sink_ref.true_count)
    assert int(sink_got.true_count) == int(np.count_nonzero(dense.rejection_mask))
    assert int(sink_got.flush_count) == int(sink_ref.flush_count)


# ---------------------------------------------------------------------------
# D. Simulated GPU->CPU fallback: sticky transition + sink rollback unchanged
# ---------------------------------------------------------------------------

# A classified CuPy OOM (module "cupy.*") so ``is_gpu_runtime_error`` treats it
# as a runtime GPU failure worth exactly one CPU replay.
_FakeCupyOOM = type(
    "OutOfMemoryError", (Exception,), {"__module__": "cupy.cuda.memory"}
)


def test_simulated_gpu_fallback_sticky_and_sink_rollback(tmp_path, monkeypatch):
    """CPU-side control-flow witness (no CuPy/hardware needed): the GPU attempt
    OOMs once, the SAME tile is re-reduced on CPU bit-exactly, the sink rollback
    keeps the count exact and every later tile stays on CPU."""
    frames, masks = _corpus(n=6, h=45, w=38)

    ref_sink = BoundedCountingRejectionSink((len(frames), 45, 38, 3))
    ref = _run_external(
        frames, masks, tmp_path / "ref", rejection_sink=ref_sink,
        column_batch=6,
    )

    # Let the request say "gpu" and let the resolver accept it; the real CuPy
    # kernel is never reached because the first GPU call is the injected OOM.
    monkeypatch.setattr(xstream, "_resolve_backend", lambda token: str(token))
    armed = {"oom": True}
    real_reduce = xstream.reduce_spooled_tile
    gpu_calls: list[str] = []

    def _fake(tile, p1, request, *, column_batch, rejection_sink=None,
              rejection_mask=None, backend="cpu", block_probe=None):
        gpu_calls.append(str(backend))
        if backend == "gpu" and armed["oom"]:
            armed["oom"] = False
            raise _FakeCupyOOM("simulated device OOM during reduction")
        return real_reduce(
            tile, p1, request,
            column_batch=column_batch,
            rejection_sink=rejection_sink,
            rejection_mask=rejection_mask,
            backend=backend,
            block_probe=block_probe,
        )

    monkeypatch.setattr(xstream, "reduce_spooled_tile", _fake)

    got_sink = BoundedCountingRejectionSink((len(frames), 45, 38, 3))
    fallback_out: dict = {}
    req_gpu = _request(frames, masks, backend="gpu")
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    got = xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req_gpu,
        tile_size=16,
        column_batch=6,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id="a",
        rejection_sink=got_sink,
        gpu_fallback_out=fallback_out,
    )

    # Same science/planes/scalars as the qualified CPU path (provenance differs
    # ONLY by the requested-vs-effective backend, which is the contract here).
    _assert_planes_exact(ref, got)
    assert got.provenance["backend"]["requested"] == "gpu"
    assert got.provenance["backend"]["rejection"] == "cpu"
    assert got.provenance["backend"]["combine"] == "cpu"
    assert int(got_sink.true_count) == int(ref_sink.true_count)
    # Exactly one GPU attempt, then STICKY CPU for every remaining tile.
    assert gpu_calls[0] == "gpu"
    assert set(gpu_calls[1:]) == {"cpu"}
    assert fallback_out["fallback_count"] == 1
    assert fallback_out["requested_backend"] == "gpu"
