"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M2 — exact external working set.

Acceptance surface for the disk spool + column-row reducer:

* BIT-EXACT parity against ``run_canonical_stack_streaming`` for tiles
  {32, 24, 16, 10}, small/large column batches, rejection {none, kappa_sigma,
  winsorized_sigma_clip} x combine {mean, median}, taper {footprint, none},
  RGB and mono, NaN / partially-valid frames / low-N columns, and the
  count-only rejection sink (no tolerance, no skip);
* the reduction workspace is O(B x N) and INDEPENDENT of patch area;
* the column-batch budget helper is exact at the boundary and refuses honestly;
* spool manifest/lifecycle: SHA-256 manifest, stale-attempt refusal, path
  confinement, truncation/corruption typed errors, cleanup on success;
* disk-full/write failure is typed and leaves no reusable tile spool;
* the M33-like pricing is non-zero, geometry-derived and names its unknowns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import external_stream as xstream
from zemosaic.core.zegrid.rejection_sink import BoundedCountingRejectionSink

MIB = 2 ** 20
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def _corpus(n=6, h=60, w=50, seed=1, mono=False):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for _k in range(3):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        base += 20.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
    gains = np.linspace(0.97, 1.03, n)
    ch_gains = np.array([0.98, 1.0, 1.03], dtype=np.float64)
    frames = []
    for i in range(n):
        if mono:
            frame = base * gains[i] + rng.normal(0, 1.5, (h, w))
        else:
            frame = np.stack(
                [
                    base * gains[i] * ch_gains[ch]
                    + rng.normal(0, 1.5, (h, w))
                    for ch in range(3)
                ],
                axis=-1,
            )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    if mono:
        frames[1][5:10, 5:10] = np.nan
    else:
        frames[1][5:10, 5:10, :] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[3][:, w // 2:] = False
    # A low-N column: only the first two frames see this strip.
    for i in range(2, n):
        masks[i][:, :2] = False
    return frames, masks


def _request(frames, masks, *, rejection, combine, taper):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection=rejection,
        combine=combine,
        taper=taper,
        taper_px=8.0,
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


def _assert_bit_exact(ref, got):
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
    assert ref.provenance == got.provenance


# ---------------------------------------------------------------------------
# 1. Bit-exact parity matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile", (32, 24, 16, 10))
@pytest.mark.parametrize("rejection", ("none", "kappa_sigma", "winsorized_sigma_clip"))
@pytest.mark.parametrize("combine", ("mean", "median"))
@pytest.mark.parametrize("taper", ("footprint", "none"))
@pytest.mark.parametrize("column_batch", (8, 1024))
def test_parity_matrix_rgb(tmp_path, tile, rejection, combine, taper, column_batch):
    frames, masks = _corpus()
    req = _request(
        frames, masks, rejection=rejection, combine=combine, taper=taper
    )
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=tile
    )
    got = xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=tile,
        column_batch=column_batch,
        spool_root=str(tmp_path),
        cell_id="cellA",
        attempt_id="attempt0",
    )
    _assert_bit_exact(ref, got)
    # The attempt spool is cleaned on success.
    assert not (tmp_path / "cellA" / "attempt0").exists()


@pytest.mark.parametrize("tile", (32, 16, 10))
@pytest.mark.parametrize("rejection", ("kappa_sigma", "winsorized_sigma_clip"))
def test_parity_matrix_mono(tmp_path, tile, rejection):
    frames, masks = _corpus(mono=True)
    req = _request(
        frames, masks, rejection=rejection, combine="median", taper="footprint"
    )
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=tile
    )
    got = xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=tile,
        column_batch=7,
        spool_root=str(tmp_path),
    )
    _assert_bit_exact(ref, got)
    assert got.science.ndim == 2  # original mono shape restored


@pytest.mark.parametrize("tile", (24, 16))
@pytest.mark.parametrize("rejection", ("kappa_sigma", "winsorized_sigma_clip"))
def test_parity_count_only_sink(tmp_path, tile, rejection):
    frames, masks = _corpus(n=8, h=50, w=44)
    req = _request(
        frames, masks, rejection=rejection, combine="mean", taper="footprint"
    )
    prov = InMemoryCanonicalProvider(frames, masks)
    dense = run_canonical_stack_streaming(prov, req, tile_size=tile)

    sink_ref = BoundedCountingRejectionSink((len(frames), 50, 44, 3))
    ref_sink = run_canonical_stack_streaming(
        prov, req, tile_size=tile, rejection_sink=sink_ref
    )
    sink_got = BoundedCountingRejectionSink((len(frames), 50, 44, 3))
    got_sink = xstream.run_cell_streaming_external(
        prov,
        req,
        tile_size=tile,
        column_batch=5,
        spool_root=str(tmp_path),
        rejection_sink=sink_got,
    )
    _assert_bit_exact(ref_sink, got_sink)
    assert ref_sink.rejection_mask is None and got_sink.rejection_mask is None
    # Sink counts are the exact dense rejection count.
    assert int(sink_got.true_count) == int(sink_ref.true_count)
    assert int(sink_got.true_count) == int(np.count_nonzero(dense.rejection_mask))
    assert int(sink_got.flush_count) == int(sink_ref.flush_count)


def test_parity_single_full_patch_tile(tmp_path):
    """tile_size=None (one full-patch tile) is the interior==extended case."""
    frames, masks = _corpus(n=5, h=40, w=36)
    req = _request(
        frames, masks, rejection="winsorized_sigma_clip", combine="mean",
        taper="footprint",
    )
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=None
    )
    got = xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=None,
        column_batch=1024,
        spool_root=str(tmp_path),
    )
    _assert_bit_exact(ref, got)


# ---------------------------------------------------------------------------
# 2. N-independence of the reduction workspace
# ---------------------------------------------------------------------------

def test_reduction_workspace_is_bounded_and_patch_independent():
    per = xstream.REDUCTION_BYTES_PER_POSITION_FRAME
    small = xstream.estimate_external_working_set(
        n_frames=4, height=40, width=40, channels=3, tile_size=16,
        halo_px=9, column_batch=64,
    )
    large_n = xstream.estimate_external_working_set(
        n_frames=64, height=40, width=40, channels=3, tile_size=16,
        halo_px=9, column_batch=64,
    )
    large_patch = xstream.estimate_external_working_set(
        n_frames=4, height=400, width=400, channels=3, tile_size=16,
        halo_px=9, column_batch=64,
    )
    p_small = small["ram"]["batch_positions"]
    p_large_n = large_n["ram"]["batch_positions"]
    p_large_patch = large_patch["ram"]["batch_positions"]
    assert p_small == p_large_n == p_large_patch
    assert small["ram"]["reduction_workspace_bytes"] == 4 * p_small * per
    assert large_n["ram"]["reduction_workspace_bytes"] == 64 * p_large_n * per
    # Linear in N, but the PATCH area (400x400 vs 40x40) does not change it.
    assert (
        large_patch["ram"]["reduction_workspace_bytes"]
        == small["ram"]["reduction_workspace_bytes"]
    )
    assert large_patch["ram"]["independent_of_patch_area"] is True
    # Far below N x patch_area float64 (the term the external set removes).
    assert large_n["ram"]["reduction_workspace_bytes"] < 64 * 400 * 400 * 3 * 8


def test_larger_n_stays_bit_exact_with_fixed_batch(tmp_path):
    for n in (4, 12):
        frames, masks = _corpus(n=n, h=48, w=44)
        req = _request(
            frames, masks, rejection="winsorized_sigma_clip", combine="median",
            taper="footprint",
        )
        ref = run_canonical_stack_streaming(
            InMemoryCanonicalProvider(frames, masks), req, tile_size=16
        )
        got = xstream.run_cell_streaming_external(
            InMemoryCanonicalProvider(frames, masks),
            req,
            tile_size=16,
            column_batch=32,
            spool_root=(lambda d: (d.mkdir(), str(d))[1])(tmp_path / f"n{n}"),
            cell_id="c",
            attempt_id="a",
        )
        _assert_bit_exact(ref, got)


# ---------------------------------------------------------------------------
# 3. Column-batch budget helper
# ---------------------------------------------------------------------------

def test_column_batch_budget_exact_boundary():
    n, iw, c = 100, 8, 3
    ppr = iw * c
    per = n * xstream.REDUCTION_BYTES_PER_POSITION_FRAME
    exact = ppr * per
    assert (
        xstream.largest_admissible_column_batch(
            budget_bytes=exact, n_frames=n, interior_width=iw, channels=c
        )
        == ppr
    )
    # F3: the floor is now ONE PIXEL (``c`` positions), not a full row, so a
    # sub-row budget admits a sub-line batch instead of refusing.
    pixel = c
    assert (
        xstream.largest_admissible_column_batch(
            budget_bytes=pixel * per, n_frames=n, interior_width=iw, channels=c
        )
        == pixel
    )
    # Below ONE pixel it still refuses honestly.
    with pytest.raises(xstream.ExternalBudgetError):
        xstream.largest_admissible_column_batch(
            budget_bytes=pixel * per - 1,
            n_frames=n, interior_width=iw, channels=c,
        )


def test_column_batch_budget_sub_line_pixel_granularity():
    """F3 audit probe: the product decider may select a sub-line batch."""
    n, iw, c = 20_000, 32, 3
    per = n * xstream.REDUCTION_BYTES_PER_POSITION_FRAME
    budget = 7_440_000
    ppr = iw * c  # 96 positions: one full row costs 119 040 000 B
    assert ppr * per > budget
    batch = xstream.largest_admissible_column_batch(
        budget_bytes=budget, n_frames=n, interior_width=iw, channels=c
    )
    # 7 440 000 // 1 920 000 == 3 positions == 1 pixel x 3 channels.
    assert batch == 3
    assert 0 < batch < ppr
    assert batch % c == 0
    assert batch * per <= budget
    # One pixel (5 760 000 B) ALWAYS fits the probe budget.
    assert c * per == 5_760_000


def test_column_batch_budget_largest_multiple_and_determinism():
    n, iw, c = 10, 8, 3
    ppr = iw * c
    per = n * xstream.REDUCTION_BYTES_PER_POSITION_FRAME
    budget = per * (ppr * 5 + 1)
    first = xstream.largest_admissible_column_batch(
        budget_bytes=budget, n_frames=n, interior_width=iw, channels=c
    )
    second = xstream.largest_admissible_column_batch(
        budget_bytes=budget, n_frames=n, interior_width=iw, channels=c
    )
    assert first == second == ppr * 5
    assert first * per <= budget


def test_sub_line_column_batch_is_bit_exact_vs_full_row(tmp_path):
    """F3: a decider-selected sub-line batch is bit-exact with full rows."""
    frames, masks = _corpus(n=6, h=60, w=50)
    req = _request(
        frames, masks, rejection="winsorized_sigma_clip", combine="median",
        taper="footprint",
    )
    n = len(frames)
    iw, c = 16, 3  # tile 16 over width 50 -> full interior row width 16
    per = n * xstream.REDUCTION_BYTES_PER_POSITION_FRAME
    ppr = iw * c
    budget = 8 * per  # far below one full row (48 * per)
    batch = xstream.largest_admissible_column_batch(
        budget_bytes=budget, n_frames=n, interior_width=iw, channels=c
    )
    assert 0 < batch < ppr
    rows, cols = xstream.batch_positions_for(batch, iw, c)
    assert rows == 1 and cols < iw  # genuinely narrower than a line

    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=16
    )
    sub = xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=16,
        column_batch=batch,
        spool_root=(lambda d: (d.mkdir(), str(d))[1])(tmp_path / "sub"),
        cell_id="c",
        attempt_id="sub",
    )
    full = xstream.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks),
        req,
        tile_size=16,
        column_batch=ppr,
        spool_root=(lambda d: (d.mkdir(), str(d))[1])(tmp_path / "full"),
        cell_id="c",
        attempt_id="full",
    )
    _assert_bit_exact(ref, sub)
    _assert_bit_exact(full, sub)


# ---------------------------------------------------------------------------
# 4. Spool lifecycle / confinement / corruption
# ---------------------------------------------------------------------------

def _spec(root, **over):
    base = dict(
        root=str(root), n_frames=4, height=20, width=18, channels=3,
        halo_px=3, tile_size=10, cell_id="cellA", attempt_id="attempt0",
        frame_ids=("f0", "f1", "f2", "f3"),
    )
    base.update(over)
    return xstream.ExternalSpoolSpec(**base)


def _write_one_tile(spool, frames, masks):
    req = _request(frames, masks, rejection="none", combine="mean", taper="none")
    p1 = xstream._fixed_phase1(InMemoryCanonicalProvider(frames, masks), req, None)
    return spool.write_tile(
        InMemoryCanonicalProvider(frames, masks), p1, 0, 10, 0, 10,
        taper_kind="none", taper_px=8.0, taper_floor=0.0, halo_px=3,
    )


def test_manifest_roundtrip_and_cleanup(tmp_path):
    frames, masks = _corpus(n=4, h=20, w=18)
    spool = xstream.ExternalSpool(_spec(tmp_path))
    handle = _write_one_tile(spool, frames, masks)
    xstream.validate_tile(handle)
    manifest = spool.finalize_manifest(taper_kind="none", taper_px=8.0, taper_floor=0.0)
    assert manifest["manifest_sha256"]
    loaded = spool.load_manifest()
    assert loaded == manifest
    assert loaded["tile_count"] == 1
    assert loaded["tile_size"] == [10, 10]
    assert loaded["dtypes"] == {"images": "float32", "weights": "float32"}
    attempt = tmp_path / "cellA" / "attempt0"
    assert attempt.exists()
    spool.cleanup()
    assert not attempt.exists()


def test_stale_attempt_is_discarded_by_default_and_refused_on_request(tmp_path):
    stale = tmp_path / "cellA" / "attempt0"
    stale.mkdir(parents=True)
    (stale / "leftover.bin").write_bytes(b"x")
    # M4: an owner-less/dead stale attempt is DISCARDED (fresh attempt), never
    # silently reused as complete.
    spool = xstream.ExternalSpool(_spec(tmp_path))
    assert spool.state == xstream.SPOOL_STATE_WRITING
    spool.cleanup()
    stale.mkdir(parents=True)
    (stale / "leftover.bin").write_bytes(b"x")
    with pytest.raises(xstream.ExternalSpoolStaleError):
        xstream.ExternalSpool(_spec(tmp_path, stale_policy="refuse"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_id": "../escape"},
        {"cell_id": "a/b"},
        {"attempt_id": ".."},
        {"attempt_id": "/abs"},
        {"attempt_id": "with space"},
    ],
)
def test_path_components_are_confined(tmp_path, kwargs):
    with pytest.raises(xstream.ExternalSpoolPathError):
        xstream.ExternalSpool(_spec(tmp_path, **kwargs))


def test_symlink_escape_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "cellA").symlink_to(outside, target_is_directory=True)
    with pytest.raises(xstream.ExternalSpoolPathError):
        xstream.ExternalSpool(_spec(root))


def test_missing_root_is_refused(tmp_path):
    with pytest.raises(xstream.ExternalSpoolPathError):
        xstream.ExternalSpool(_spec(tmp_path / "does-not-exist"))


def test_truncated_spool_is_refused(tmp_path):
    frames, masks = _corpus(n=4, h=20, w=18)
    spool = xstream.ExternalSpool(_spec(tmp_path))
    handle = _write_one_tile(spool, frames, masks)
    path = Path(handle.images_path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    with pytest.raises(xstream.ExternalSpoolCorruptError):
        xstream.validate_tile(handle)


def test_corrupt_manifest_is_refused(tmp_path):
    frames, masks = _corpus(n=4, h=20, w=18)
    spool = xstream.ExternalSpool(_spec(tmp_path))
    _write_one_tile(spool, frames, masks)
    spool.finalize_manifest(taper_kind="none", taper_px=8.0, taper_floor=0.0)
    payload = json.loads(spool.manifest_path.read_text())
    payload["tile_count"] = payload["tile_count"] + 1  # tamper
    spool.manifest_path.write_text(json.dumps(payload))
    with pytest.raises(xstream.ExternalSpoolCorruptError):
        spool.load_manifest()


def test_missing_manifest_is_refused(tmp_path):
    frames, masks = _corpus(n=4, h=20, w=18)
    spool = xstream.ExternalSpool(_spec(tmp_path))
    with pytest.raises(xstream.ExternalSpoolCorruptError):
        spool.load_manifest()


def test_discard_tile_removes_payload_only(tmp_path):
    frames, masks = _corpus(n=4, h=20, w=18)
    spool = xstream.ExternalSpool(_spec(tmp_path))
    handle = _write_one_tile(spool, frames, masks)
    tile_dir = Path(handle.images_path).parent
    spool.discard_tile(handle)
    assert not tile_dir.exists()
    assert (tmp_path / "cellA" / "attempt0").exists()
    spool.cleanup()


# ---------------------------------------------------------------------------
# 5. Disk-full / write failure
# ---------------------------------------------------------------------------

def test_write_error_is_typed_and_leaves_no_spool(tmp_path, monkeypatch):
    frames, masks = _corpus(n=4, h=20, w=18)
    req = _request(frames, masks, rejection="none", combine="mean", taper="none")

    def _boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np.lib.format, "open_memmap", _boom)
    with pytest.raises(xstream.ExternalSpoolWriteError):
        xstream.run_cell_streaming_external(
            InMemoryCanonicalProvider(frames, masks),
            req,
            tile_size=10,
            column_batch=4,
            spool_root=str(tmp_path),
            cell_id="cellA",
            attempt_id="attempt0",
        )
    assert not (tmp_path / "cellA" / "attempt0").exists()


def test_write_error_on_weights_file_removes_partial_tile(tmp_path, monkeypatch):
    frames, masks = _corpus(n=4, h=20, w=18)
    spool = xstream.ExternalSpool(_spec(tmp_path))
    real = np.lib.format.open_memmap
    calls = {"n": 0}

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", _flaky)
    with pytest.raises(xstream.ExternalSpoolWriteError):
        _write_one_tile(spool, frames, masks)
    tiles = list((tmp_path / "cellA" / "attempt0").glob("tile_*"))
    assert tiles == []
    spool.cleanup()


# ---------------------------------------------------------------------------
# 6. M33-like pricing honesty
# ---------------------------------------------------------------------------

def test_m33_pricing_is_nonzero_geometry_derived_and_honest():
    n, h, w, c, tile, halo = 22414, 64, 64, 3, 20, 9
    estimate = xstream.estimate_external_working_set(
        n_frames=n, height=h, width=w, channels=c, tile_size=tile,
        halo_px=halo, column_batch=1024,
    )
    worst = estimate["disk"]["spool_peak_bytes_worst_tile"]
    assert worst > 0
    # Exact geometry: worst extended tile is 38 x 38, f32 images + f32 weights.
    assert estimate["geometry"]["max_extent_hw"] == [38, 38]
    assert worst == n * 38 * 38 * (4 * c + 4)
    assert estimate["disk"]["cumulative_writes_bytes"] > worst
    assert estimate["disk"]["cumulative_reads_bytes"] == estimate["disk"][
        "cumulative_writes_bytes"
    ]
    assert estimate["disk"]["read_amplification"] == 1.0
    assert estimate["disk"]["cleanup_margin_bytes"] > 0
    assert estimate["disk"]["total_peak_bytes"] == (
        worst + estimate["disk"]["cleanup_margin_bytes"]
    )
    assert estimate["disk"]["unknown_terms"] == list(xstream._UNKNOWN_DISK_TERMS)
    assert estimate["disk"]["unknown_bytes"] is None
    assert estimate["ram"]["reduction_workspace_bytes"] == (
        n * estimate["ram"]["batch_positions"]
        * xstream.REDUCTION_BYTES_PER_POSITION_FRAME
    )
    assert estimate["ram"]["reduction_workspace_bytes"] > 0
    # Tile count is geometry-derived (4 x 4 tiles for 64 px at tile 20).
    assert estimate["geometry"]["tile_count"] == 16
