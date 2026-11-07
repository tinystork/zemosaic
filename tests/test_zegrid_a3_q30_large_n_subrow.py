"""A3 — Q30: canonical reduction at large N with sub-row micro-blocks.

Proves the external working-set reduction, once the 2D micro-block decomposition
is enabled, is bit-exact against the in-memory canonical streaming oracle for
N in {2k, 20k, 50k, 100k} on a BOUNDED surface, while exercising blocks
narrower than one interior row (``cols_per_batch < interior_width``).

A bounded surface is deliberate (the Q30 gate allows it): the oracle materialises
all N frames in RAM, so the patch is kept small so N=100k stays tractable. Memory
*bounds* are the A2/Q20 concern, not Q30.
"""
from __future__ import annotations

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import external_stream as x

_PLANES = (
    "science",
    "estimator_weight_sum",
    "valid_mask",
    "surviving_sample_count",
    "support_w1",
    "support_w2",
    "n_eff_support",
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


def _corpus(n, h=16, w=16, seed=11):
    """Deterministic, varied frames + support (never identical)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 6.0 * np.sin(xx * 0.5) + 3.0 * np.cos(yy * 0.4)
    gains = np.linspace(0.95, 1.05, n)
    ch_gains = np.array([0.98, 1.0, 1.03], dtype=np.float64)
    frames = []
    for i in range(n):
        f = np.stack(
            [base * gains[i] * ch_gains[ch] + rng.normal(0, 1.2, (h, w))
             for ch in range(3)],
            axis=-1,
        ).astype(np.float32)
        if i % 7 == 0:
            f[0:2, 0:2, :] = np.nan
        if i == n - 1:
            f[5, 5, 0] = 1e5  # single-pixel outlier
        frames.append(np.ascontiguousarray(f))
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    for i in range(n):
        if i % 3 == 0:
            masks[i][:, :2] = False  # low-N column
    return frames, masks


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
    for name in _SCALARS:
        assert getattr(ref, name) == getattr(got, name), name


def test_batch_positions_for_subrow_blocks():
    # column_batch below one full row -> a single row split into column chunks.
    rows, cols = x.batch_positions_for(8, 16, 3)
    assert (rows, cols) == (1, 2)
    assert cols < 16  # narrower than one interior row
    # at least one full row -> whole rows.
    assert x.batch_positions_for(48, 16, 3) == (1, 16)
    assert x.batch_positions_for(96, 16, 3) == (2, 16)


@pytest.mark.parametrize("n", [2_000, 20_000, 50_000, 100_000])
def test_q30_large_n_subrow_blocks_bit_exact(tmp_path, n):
    h = w = 16
    frames, masks = _corpus(n, h=h, w=w)
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="none", combine="mean", taper="none",
    )
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=16,
    )
    # column_batch=8 with interior_width=16 forces sub-row blocks (cols=2 < 16).
    got = x.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=16,
        column_batch=8,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id="a",
    )
    _assert_bit_exact(ref, got)
    assert got.n_frames == n


def test_q30_n2000_subrow_blocks_winsorized_bit_exact(tmp_path):
    """A stronger rejection/combine at moderate N with sub-row blocks."""
    n, h, w = 2_000, 24, 24
    frames, masks = _corpus(n, h=h, w=w)
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean", taper="none",
    )
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=16,
    )
    # interior_width=24 -> one row = 72 positions; column_batch=24 -> cols=8 < 24.
    got = x.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=16,
        column_batch=24,
        spool_root=str(tmp_path),
        cell_id="c",
        attempt_id="a",
    )
    _assert_bit_exact(ref, got)
    assert x.batch_positions_for(24, 24, 3) == (1, 8)
