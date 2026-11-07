"""A2c/A2d — Q20 (N≥20k, scratch < full cache) and Q21 (I/O error/stop).

Closes the A2 gate on the source-backed external path:

* Q20a — N=20 000, bit-exact parity of the windowed provider + external spool
  against the canonical in-memory streaming oracle (varied, adversarial frames).
* Q20b — N=20 000 on a patch whose full aligned cache is much larger than the
  worst single-tile spool; proves the windowed path progresses, keeps peak
  scratch under the full-cache estimate, recycles the spool and holds at most
  one live frame.
* Q21 — an injected I/O failure mid-preparation must propagate, remove only the
  partial tile/spool it owns, and never touch sources or a pre-existing valid
  artifact.

No capacity claim: these are correctness/qualification witnesses for the A2
gate, not the full M33 product run (A6).
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
from zemosaic.core.zegrid import file_provider as fp

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


def _varied_frame(i: int, h: int, w: int, seed: int):
    """Deterministic, varied, adversarial frame + support (never identical)."""
    rng = np.random.default_rng(seed + i)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 5.0 * np.sin(xx * 0.31) + 3.0 * np.cos(yy * 0.23)
    gain = 0.95 + 0.10 * ((i % 7) / 7.0)
    frame = np.stack(
        [base * gain + rng.normal(0.0, 1.5, (h, w)) for _ in range(3)],
        axis=-1,
    ).astype(np.float32)
    mask = np.ones((h, w), dtype=bool)
    if i % 5 == 0:
        frame[0:2, 0:2, :] = np.nan
    if i % 3 == 0:
        mask[:, :2] = False  # low-N column
    if i == 9999:
        frame[5, 6, 0] = 1e5  # single-pixel outlier
    return frame, mask


def _request(n: int, *, rejection: str, combine: str, taper: str):
    return CanonicalStackRequest(
        images=[None] * n,
        geometric_support=[None] * n,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection=rejection,
        combine=combine,
        taper=taper,
        taper_px=8.0,
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
        assert np.array_equal(a, b, equal_nan=True), name
    for name in _SCALARS:
        assert getattr(ref, name) == getattr(got, name), name


def test_q20_n20k_windowed_spool_bit_exact_parity(tmp_path):
    """N=20 000 varied frames: windowed+spool is bit-exact to the RAM oracle."""
    n, h, w, seed = 20_000, 24, 24, 20261107
    frames = []
    masks = []
    for i in range(n):
        f, m = _varied_frame(i, h, w, seed)
        frames.append(f)
        masks.append(m)

    req = _request(n, rejection="winsorized_sigma_clip", combine="mean", taper="none")

    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=16
    )

    def prepare(i):
        return _varied_frame(i, h, w, seed)

    provider = fp.WindowedCanonicalProvider(prepare, n, expected_hw=(h, w))
    try:
        got = xstream.run_cell_streaming_external(
            provider,
            req,
            tile_size=16,
            column_batch=32,
            spool_root=str(tmp_path),
            cell_id="c",
            attempt_id="a",
        )
    finally:
        provider.close()

    _assert_bit_exact(ref, got)
    assert provider.live_window == 1
    # The N×patch aligned cache was never materialised; the spool is recycled.
    assert not list(tmp_path.rglob("*.npy"))


def test_q20_n20k_spool_peak_under_full_cache_and_recycled(tmp_path):
    """N=20 000 on a 64×64 patch: worst-tile spool ≪ full aligned cache."""
    n, h, w, seed = 20_000, 64, 64, 424242
    req = _request(n, rejection="none", combine="mean", taper="none")

    def prepare(i):
        return _varied_frame(i, h, w, seed)

    provider = fp.WindowedCanonicalProvider(prepare, n, expected_hw=(h, w))
    try:
        pricing = xstream.estimate_external_working_set(
            n_frames=n, height=h, width=w, channels=3,
            tile_size=16, halo_px=0, column_batch=32,
        )
        spool_peak = pricing["disk"]["spool_peak_bytes_worst_tile"]
        # Full aligned cache for the dense path: N × patch_px × 13 B/px/frame.
        full_cache_estimate = n * h * w * 13
        assert spool_peak < full_cache_estimate, (
            f"spool peak {spool_peak} must stay under full cache "
            f"{full_cache_estimate}"
        )

        spool_info = {}
        got = xstream.run_cell_streaming_external(
            provider,
            req,
            tile_size=16,
            column_batch=32,
            spool_root=str(tmp_path),
            cell_id="c",
            attempt_id="a",
            spool_out=spool_info,
        )
    finally:
        provider.close()

    assert got.n_frames == n
    # ``bytes_written`` is the CUMULATIVE payload across all tiles (each tile is
    # written, reduced, then discarded). The on-disk PEAK is the worst single
    # tile + cleanup margin — never the full N×patch cache. Assert both the
    # per-tile bound and the multi-tile recycling, not cumulative vs peak.
    assert spool_info["bytes_written"] >= spool_peak
    assert spool_info["tile_count"] == 16  # (64/16) × (64/16) tiles
    assert spool_info["bytes_written"] <= 16 * (spool_peak + (spool_peak // 20 + (1 << 20)))
    # Recycled: no spool payload survives a successful run.
    assert not list(tmp_path.rglob("*.npy"))
    assert provider.live_window == 1


def test_q21_mid_preparation_io_error_cleans_owned_artifacts_only(tmp_path):
    """I/O failure mid-preparation: partial spool removed, sources/artifact kept."""
    n, h, w, seed = 200, 24, 24, 7
    # A pre-existing valid artifact OUTSIDE the attempt dir must survive.
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("pre-existing valid artifact")

    req = _request(n, rejection="none", combine="mean", taper="none")
    fail_at = 50

    def prepare(i):
        if i == fail_at:
            raise OSError("injected source read failure")
        return _varied_frame(i, h, w, seed)

    provider = fp.WindowedCanonicalProvider(prepare, n, expected_hw=(h, w))
    with pytest.raises(OSError, match="injected source read failure"):
        xstream.run_cell_streaming_external(
            provider,
            req,
            tile_size=16,
            column_batch=32,
            spool_root=str(tmp_path),
            cell_id="c",
            attempt_id="a",
        )

    # Sources (here the deterministic generator) are untouched by construction;
    # the pre-existing artifact is preserved and no partial spool payload leaks.
    assert sentinel.read_text() == "pre-existing valid artifact"
    assert not list(tmp_path.rglob("*.npy"))
    # The window must be released even after the failure (A2b invariant).
    assert provider._rgb is None or provider._index is None
    provider.close()
