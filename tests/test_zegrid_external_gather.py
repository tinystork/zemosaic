"""ZM-M33-EXT-GATHER — exact vectorised gather of one external micro-block.

``_gather_block`` must return the SAME bytes as the historical per-row Python
loop (slices + ``concatenate`` + ``np.repeat``), just without paying for it.
The oracle used here is a scalar-index loop (one element at a time), which is
independent of both the historical three-slice implementation and the new
grouped 4-D view copies: a shared bug in a slice arithmetic would have to
reproduce itself element by element to pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import external_stream as xstream


# ---------------------------------------------------------------------------
# Independent oracle (scalar index, no vectorised slice arithmetic)
# ---------------------------------------------------------------------------

def _oracle(images, weights, n, tw, c, iy0, ix0, w_int, a, b, x0, x1):
    bw = x1 - x0
    positions = (b - a) * bw * c
    img = np.empty((n, positions), dtype=np.float32)
    wt = np.empty((n, positions), dtype=np.float64)
    for i in range(n):
        p = 0
        for iy in range(a, b):
            for ix in range(x0, x1):
                w = np.float64(weights[i, iy0 + iy, ix0 + ix])
                for ch in range(c):
                    img[i, p] = images[i, iy0 + iy, ix0 + ix, ch]
                    wt[i, p] = w
                    p += 1
    return img, wt


def _sources(n, th, tw, c, seed=11):
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(n, th, tw, c)).astype(np.float32)
    weights = np.abs(rng.normal(size=(n, th, tw))).astype(np.float32)
    # Stress the exact-bit contract: NaN / +-inf / signed zero / zero weight.
    images[0, 0, 0, 0] = np.nan
    images[0, 1, 1, c - 1] = np.inf
    images[0, 2, 2, c - 1] = -np.inf
    images[1, 0, 0, 0] = -0.0
    images[1, 0, 1, 0] = 0.0
    weights[0, 0, 0] = np.nan
    weights[0, 0, 1] = np.inf
    weights[0, 0, 2] = -np.inf
    weights[1, 1, 1] = -0.0
    weights[1, 1, 2] = 0.0
    return images, weights


_NAN_PAYLOAD_BITS = 0x7FC0ABCD


def _payload_nan():
    """A quiet float32 NaN with a NON-canonical mantissa payload."""
    return np.frombuffer(
        np.uint32(_NAN_PAYLOAD_BITS).tobytes(), dtype=np.float32
    )[0]


def _mark_specials(images, weights, c):
    """Put every stress value INSIDE the window [0, 3) x [0, 3)."""
    images[0, 0, 0, 0] = _payload_nan()
    images[0, 0, 1, 0] = np.inf
    images[0, 0, 2, 0] = -np.inf
    images[0, 1, 0, 0] = -0.0
    images[0, 1, 1, 0] = 0.0
    images[0, 2, 2, c - 1] = 1.5
    weights[0, 0, 0] = np.nan
    weights[0, 0, 1] = np.inf
    weights[0, 0, 2] = -np.inf
    weights[0, 1, 1] = -0.0
    weights[0, 2, 2] = 0.0


def _assert_gather_contains_specials(img, wt):
    """The GATHERED bytes must really carry the stress values (not the fixtures)."""
    ub = img.view(np.uint32)
    assert np.isnan(img).any()
    assert (ub == np.uint32(_NAN_PAYLOAD_BITS)).sum() == 1, "NaN payload"
    assert np.isposinf(img).any() and np.isneginf(img).any()
    assert (ub == np.uint32(0x80000000)).any(), "-0.0 image"
    assert (ub == np.uint32(0)).any(), "+0.0 image"
    wb = wt.view(np.uint64)
    assert np.isnan(wt).any()
    assert np.isposinf(wt).any() and np.isneginf(wt).any()
    assert (wb == np.uint64(0x8000000000000000)).any(), "-0.0 weight"
    assert (wb == np.uint64(0)).any(), "+0.0 weight"


def _assert_gather_exact(args):
    images, weights = args[0], args[1]
    ref_img, ref_wt = _oracle(*args)
    got_img, got_wt = xstream._gather_block(*args)
    assert got_img.dtype == np.float32 and got_wt.dtype == np.float64
    assert got_img.shape == ref_img.shape and got_wt.shape == ref_wt.shape
    assert got_img.flags["C_CONTIGUOUS"] and got_wt.flags["C_CONTIGUOUS"]
    # Bit patterns, not just values: NaN payloads, signed zeros and infs must
    # be reproduced byte for byte (equal_nan would not prove that).
    assert got_img.tobytes() == ref_img.tobytes()
    assert got_wt.tobytes() == ref_wt.tobytes()
    return got_img, got_wt


# ---------------------------------------------------------------------------
# 1. Oracle parity over the geometry the product really forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n, th, tw, c, iy0, ix0, w_int, a, b, x0, x1",
    [
        # n subset of the spooled frames (N > n)
        (3, 12, 12, 3, 2, 2, 8, 0, 1, 0, 8),
        # one interior subrow, full interior width
        (5, 12, 12, 3, 2, 2, 8, 0, 1, 0, 8),
        # several interior rows (multiline), full width
        (5, 12, 12, 3, 2, 2, 8, 0, 4, 0, 8),
        # halo offsets on both axes + partial column window
        (5, 12, 12, 3, 3, 3, 6, 1, 3, 2, 5),
        # last partial row and partial last column
        (4, 12, 12, 3, 2, 2, 9, 2, 4, 1, 3),
        # monochrome (C == 1)
        (5, 12, 12, 1, 2, 2, 8, 0, 3, 0, 8),
        # one single pixel (narrowest allowed batch)
        (4, 12, 12, 3, 2, 2, 8, 2, 3, 1, 2),
        # full interior, C == 3, many rows
        (3, 9, 11, 3, 0, 0, 11, 0, 9, 3, 11),
    ],
)
def test_gather_matches_scalar_oracle_bit_exact(
    n, th, tw, c, iy0, ix0, w_int, a, b, x0, x1
):
    images, weights = _sources(n, th, tw, c)
    _assert_gather_exact(
        (images, weights, n, tw, c, iy0, ix0, w_int, a, b, x0, x1)
    )


@pytest.mark.parametrize("c", [1, 3])
def test_gather_window_at_origin_keeps_special_values_bit_exact(c):
    """Window starting at (0, 0) actually gathers NaN/+-inf/-0/0 and the
    non-canonical NaN payload — the fixture values must land in the OUTPUT."""
    n, th, tw = 4, 12, 12
    images, weights = _sources(n, th, tw, c)
    _mark_specials(images, weights, c)
    args = (images, weights, n, tw, c, 0, 0, tw, 0, 3, 0, 3)
    img, wt = _assert_gather_exact(args)
    _assert_gather_contains_specials(img, wt)
    # One payload occurrence: the 4-D copy neither drops nor duplicates it.
    assert (img.view(np.uint32) == np.uint32(_NAN_PAYLOAD_BITS)).sum() == 1


@pytest.mark.parametrize("n", [2, 3])
def test_n_subset_of_five_spooled_frames(n):
    """True N > n: only the first n frames are read, excluded frames never
    leak into the (n, positions) buffers."""
    N, th, tw, c = 5, 10, 12, 3
    images, weights = _sources(N, th, tw, c)
    images[n:, :, :, :] = np.float32(1e30)
    weights[n:, :, :] = np.float32(1e30)
    args = (images, weights, n, tw, c, 1, 1, 10, 0, 3, 0, 10)
    img, wt = _assert_gather_exact(args)
    assert img.shape == (n, 3 * 10 * c) and wt.shape == (n, 3 * 10 * c)
    # Exact sentinel equality: inf/NaN fixtures must not trip the check.
    assert np.count_nonzero(img == np.float32(1e30)) == 0, "excluded frame leaked"
    assert np.count_nonzero(wt == 1e30) == 0, "excluded weight leaked"


def test_n_subset_over_readonly_memmaps(tmp_path):
    N, th, tw, c, n = 5, 10, 12, 3, 2
    images, weights = _sources(N, th, tw, c)
    images[n:, :, :, :] = np.float32(1e30)
    weights[n:, :, :] = np.float32(1e30)
    ip, wp = tmp_path / "images.npy", tmp_path / "weights.npy"
    np.save(ip, images)
    np.save(wp, weights)
    im = np.load(ip, mmap_mode="r")
    wm = np.load(wp, mmap_mode="r")
    assert not im.flags.writeable and not wm.flags.writeable
    args = (im, wm, n, tw, c, 1, 1, 10, 0, 3, 0, 10)
    img, wt = _assert_gather_exact(args)
    assert img.shape == (n, 3 * 10 * c) == wt.shape
    assert np.count_nonzero(img == np.float32(1e30)) == 0


def test_weights_broadcast_matches_cast_then_repeat():
    """The float64 weight column is the exact ``asarray(f64)`` + ``repeat``."""
    n, th, tw, c = 6, 10, 10, 3
    images, weights = _sources(n, th, tw, c)
    a, b, x0, x1, iy0, ix0 = 1, 4, 2, 7, 2, 2
    got_img, got_wt = xstream._gather_block(
        images, weights, n, tw, c, iy0, ix0, 7, a, b, x0, x1
    )
    for i in range(n):
        expected = np.repeat(
            np.asarray(
                weights[i][iy0 + a:iy0 + b, ix0 + x0:ix0 + x1],
                dtype=np.float64,
            ),
            c,
            axis=1,
        ).reshape(-1)
        assert got_wt[i].tobytes() == expected.tobytes()
    assert got_wt.shape == (n, (b - a) * (x1 - x0) * c)


# ---------------------------------------------------------------------------
# 2. Real read-only .npy memmaps (the product spool) and non-contiguous views
# ---------------------------------------------------------------------------

def test_readonly_npy_memmap_sources(tmp_path):
    n, th, tw, c = 4, 11, 13, 3
    images, weights = _sources(n, th, tw, c)
    ip, wp = tmp_path / "images.npy", tmp_path / "weights.npy"
    np.save(ip, images)
    np.save(wp, weights)
    im = np.load(ip, mmap_mode="r")
    wm = np.load(wp, mmap_mode="r")
    assert not im.flags.writeable and not wm.flags.writeable
    args = (im, wm, n, tw, c, 2, 3, 9, 1, 4, 1, 8)
    got_img, got_wt = _assert_gather_exact(args)
    # Fresh owning outputs: never an alias of the read-only spool pages.
    assert not np.shares_memory(got_img, im)
    assert not np.shares_memory(got_wt, wm)


def test_non_contiguous_valid_sources():
    """Strided (but logically valid) sources are copied element for element."""
    n, th, tw, c = 4, 10, 10, 3
    images, weights = _sources(n, th, tw, c)
    big = np.zeros((n, th, 2 * tw, c), dtype=np.float32)
    big[:, :, ::2, :] = images
    images_nc = big[:, :, ::2, :]
    wbig = np.zeros((n, th, 2 * tw), dtype=np.float32)
    wbig[:, :, ::2] = weights
    weights_nc = wbig[:, :, ::2]
    assert not images_nc.flags["C_CONTIGUOUS"]
    assert not weights_nc.flags["C_CONTIGUOUS"]
    _assert_gather_exact(
        (images_nc, weights_nc, n, tw, c, 1, 1, 8, 0, 3, 2, 6)
    )


def test_outputs_are_fresh_owning_c_order_buffers():
    n, th, tw, c = 3, 8, 8, 3
    images, weights = _sources(n, th, tw, c)
    img, wt = xstream._gather_block(
        images, weights, n, tw, c, 0, 0, 8, 0, 3, 0, 8
    )
    for out in (img, wt):
        assert out.flags["OWNDATA"]
        assert out.flags["C_CONTIGUOUS"]
        assert out.base is None
    assert not np.shares_memory(img, images)
    assert not np.shares_memory(wt, weights)


# ---------------------------------------------------------------------------
# 3. Invalid inputs must FAIL, never broadcast a frame/channel silently
# ---------------------------------------------------------------------------

def _valid_sources(n=4, th=12, tw=12, c=3):
    return _sources(n, th, tw, c)


@pytest.mark.parametrize(
    "mutate, args_override",
    [
        # fewer spooled frames than requested n
        ("n_overruns_frames", dict(n=6)),
        # weights spool shorter than images
        ("weights_shorter", dict(w_shape=(3, 12, 12))),
        # channel count mismatch (would broadcast channels)
        ("channels", dict(c=2)),
        # tw mismatch between caller and spooled tile
        ("tw", dict(tw=10)),
        # empty row span
        ("empty_rows", dict(a=3, b=3)),
        # empty column span
        ("empty_cols", dict(x0=2, x1=2)),
        # block columns beyond the interior width
        ("x1_over_interior", dict(x0=0, x1=9)),
        # block rows beyond the tile
        ("rows_beyond_tile", dict(a=0, b=13)),
        # block columns beyond the tile width
        ("cols_beyond_tile", dict(iy0=2, ix0=6, x0=0, x1=8)),
    ],
)
def test_incoherent_gather_requests_raise(mutate, args_override):
    images, weights = _valid_sources()
    args = dict(
        images=images, weights=weights, n=4, tw=12, c=3, iy0=2, ix0=2,
        w_int=8, a=0, b=2, x0=0, x1=8,
    )
    args.update(args_override)
    if mutate == "weights_shorter":
        args["weights"] = np.zeros(args_override["w_shape"], dtype=np.float32)
        del args["w_shape"]
    with pytest.raises(ValueError):
        xstream._gather_block(
            args["images"], args["weights"], args["n"], args["tw"], args["c"],
            args["iy0"], args["ix0"], args["w_int"], args["a"], args["b"],
            args["x0"], args["x1"],
        )


def test_non_4d_images_raise():
    weights = np.zeros((4, 12, 12), dtype=np.float32)
    flat = np.zeros((4, 12, 36), dtype=np.float32)
    with pytest.raises(ValueError):
        xstream._gather_block(flat, weights, 4, 12, 3, 2, 2, 8, 0, 2, 0, 8)


# ---------------------------------------------------------------------------
# 4. Whole external reduction: loop baseline vs vectorised gather, bit exact
# ---------------------------------------------------------------------------

def _corpus(n=5, h=45, w=38, seed=3):
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
        frame = np.stack(
            [base * gains[i] * ch_gains[ch] + rng.normal(0, 1.5, (h, w))
             for ch in range(3)],
            axis=-1,
        )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    frames[1][3:9, 4:11, :] = np.nan
    frames[-1][:, :] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[3][:, w // 2:] = False
    return frames, masks


def _request(frames, masks):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="winsorized_sigma_clip",
        combine="mean",
        taper="footprint",
        taper_px=8.0,
        backend="cpu",
    )


def _run_external(frames, masks, root):
    Path(root).mkdir(parents=True, exist_ok=True)
    return xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        _request(frames, masks),
        tile_size=16,
        column_batch=6,
        spool_root=str(root),
        cell_id="cell",
        attempt_id="attempt",
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


def _assert_result_bit_exact(ref, got):
    assert ref.n_frames == got.n_frames
    assert (ref.height, ref.width, ref.channels) == (
        got.height, got.width, got.channels
    )
    for name in _PLANES:
        a, b = getattr(ref, name), getattr(got, name)
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        assert np.array_equal(a, b, equal_nan=True), name
    for name in _SCALARS:
        assert getattr(ref, name) == getattr(got, name), name
    assert ref.provenance == got.provenance


def test_whole_external_reduction_bit_exact_against_loop_baseline(
    tmp_path, monkeypatch
):
    """Swap ``_gather_block`` for the scalar oracle inside the REAL product
    path and require byte-identical cell results."""
    frames, masks = _corpus()
    ref = _run_external(frames, masks, tmp_path / "ref")

    real = xstream._gather_block
    calls = {"n": 0}

    def _loop_baseline(*args):
        calls["n"] += 1
        return _oracle(*args)

    monkeypatch.setattr(xstream, "_gather_block", _loop_baseline)
    got = _run_external(frames, masks, tmp_path / "got")
    monkeypatch.setattr(xstream, "_gather_block", real)

    assert calls["n"] > 0
    _assert_result_bit_exact(ref, got)
    # And the same result as the qualified in-RAM canonical path.
    _assert_result_bit_exact(
        run_canonical_stack_streaming(
            InMemoryCanonicalProvider(frames, masks), _request(frames, masks),
            tile_size=16,
        ),
        got,
    )
