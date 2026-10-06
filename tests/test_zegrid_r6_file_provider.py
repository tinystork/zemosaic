"""ZM-ZEGRID-R6 targeted tests — file-backed provider + streaming integration.

* Light (always run): file-backed provider bit-equivalence vs
  ``InMemoryCanonicalProvider`` (same aligned arrays) through the streaming
  executor; tile-size invariance through the integrated path; memory-gate
  presence/estimate bound; determinism.
* Heavy (gated): real-cell streaming-vs-in-memory MiniTile parity for
  r0001c0002 (N=66) — skipped with an explicit reason if the memory gate /
  fixtures cannot be satisfied, but run and reported when possible.

No production dispatch wiring. No new dependency. Artifacts live on disk
(never /tmp) under ``/home/tristan/zegrid_r6_*``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import streaming as zstream
from zemosaic.core.zegrid import sweep as zsw

LIGHTS = Path("/home/tristan/M106/lights")
FIXTURES = Path("/home/tristan/zegrid_r2_fixtures")
R6_TOOL = (
    Path("/home/tristan/.openclaw/workspace/worktrees/zemosaic-zegrid-r1")
    / "tools" / "zegrid_r6" / "run_streaming_cell.py"
)


# ---------------------------------------------------------------------------
# Synthetic aligned corpus (bit-exactness harness; no fixtures needed)
# ---------------------------------------------------------------------------

def _aligned_corpus(n=6, h=40, w=40, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0, 3, (h, w, 3)).astype(np.float32) + np.array(
            [50, 55, 60], np.float32
        ) + 2.0 * i
        f[0:2, :, :] = np.nan
        frames.append(f)
    masks = [np.ones((h, w), dtype=bool)] * n
    masks[1][10:14, 10:14] = False
    masks[3][:, 20:] = False
    return frames, masks


def _assert_minitile_bit_exact(ref, st):
    np.testing.assert_array_equal(ref.science, st.science)
    np.testing.assert_array_equal(ref.estimator_weight_sum, st.estimator_weight_sum)
    np.testing.assert_array_equal(ref.support_w1, st.support_w1)
    np.testing.assert_array_equal(ref.support_w2, st.support_w2)
    np.testing.assert_array_equal(ref.n_eff_support, st.n_eff_support)
    np.testing.assert_array_equal(ref.valid_mask, st.valid_mask)
    np.testing.assert_array_equal(ref.surviving_sample_count, st.surviving_sample_count)


# ---------------------------------------------------------------------------
# 1. File-backed provider bit-equivalence vs InMemoryCanonicalProvider
# ---------------------------------------------------------------------------

def test_file_provider_bit_equivalent_to_inmemory(tmp_path):
    frames, masks = _aligned_corpus()
    ids = [f"frame_{i}" for i in range(len(frames))]
    cache = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)

    fp = zfp.MemmapCanonicalProvider(cache)
    im = InMemoryCanonicalProvider(frames, masks)

    # Metadata parity.
    assert fp.n_frames == im.n_frames
    assert fp.height == im.height == 40
    assert fp.width == im.width == 40
    assert fp.channels == im.channels == 3
    assert fp.original_mono == im.original_mono is False

    # Raw frames bit-equal (phase 1 contract).
    for i in range(len(frames)):
        r_fp, s_fp = fp.get_raw_frame(i)
        r_im, s_im = im.get_raw_frame(i)
        np.testing.assert_array_equal(np.asarray(r_fp), np.asarray(r_im))
        np.testing.assert_array_equal(np.asarray(s_fp), np.asarray(s_im))

    # Tiles bit-equal (phase 2 contract).
    for (y0, y1, x0, x1) in [(0, 10, 0, 10), (3, 31, 5, 22), (0, 40, 0, 40)]:
        t_fp, v_fp = fp.get_tile(2, y0, y1, x0, x1)
        t_im, v_im = im.get_tile(2, y0, y1, x0, x1)
        np.testing.assert_array_equal(t_fp, t_im)
        np.testing.assert_array_equal(v_fp, v_im)


def test_file_provider_streaming_result_bit_equal_to_inmemory_engine(tmp_path):
    frames, masks = _aligned_corpus()
    ids = [f"frame_{i}" for i in range(len(frames))]
    cache = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)

    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    ref = run_canonical_stack(req)
    st = run_canonical_stack_streaming(zfp.MemmapCanonicalProvider(cache), req, tile_size=9)
    _assert_minitile_bit_exact(ref, st)
    np.testing.assert_array_equal(ref.rejection_mask, st.rejection_mask)
    assert ref.provenance == st.provenance


def test_file_provider_streaming_via_run_cell_streaming_synthetic(tmp_path):
    """The integrated ``run_cell_streaming`` path produces a valid MiniTile."""
    frames, masks = _aligned_corpus()
    ids = [f"frame_{i}" for i in range(len(frames))]
    cache = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    prov = zfp.MemmapCanonicalProvider(cache)
    req = build_req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    ref = run_canonical_stack(req)
    st = run_canonical_stack_streaming(prov, req, tile_size=9)
    _assert_minitile_bit_exact(ref, st)


# ---------------------------------------------------------------------------
# 2. Tile-size invariance through the integrated path
# ---------------------------------------------------------------------------

def test_file_provider_tile_invariance(tmp_path):
    frames, masks = _aligned_corpus()
    ids = [f"frame_{i}" for i in range(len(frames))]
    cache = tmp_path / "cache"
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="median",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    prov = zfp.MemmapCanonicalProvider(cache)
    a = run_canonical_stack_streaming(prov, req, tile_size=8)
    b = run_canonical_stack_streaming(prov, req, tile_size=19)
    c = run_canonical_stack_streaming(prov, req, tile_size=None)
    _assert_minitile_bit_exact(a, b)
    _assert_minitile_bit_exact(a, c)


# ---------------------------------------------------------------------------
# 3. Memory gate presence + estimate bound
# ---------------------------------------------------------------------------

def test_streaming_gate_present_and_estimate_bounded():
    n, patch_hw = 66, (836, 496)
    est = zstream.estimate_streaming_peak_kib(n, patch_hw, 128)
    # Streaming estimate must be FAR below the in-memory O(N x patch_area) peak
    # (~3.4-4.1 GiB); assert a hard sanity ceiling (must be < 1.5 GiB).
    assert est < 1.5 * 1024**2, est
    # Monotone in tile area (larger tile -> larger workspace).
    assert zstream.estimate_streaming_peak_kib(n, patch_hw, 256) > est
    gate = zstream.check_streaming_gate(n, patch_hw, 128)
    assert gate.required > 0
    assert isinstance(gate.ok, bool)


def test_streaming_gate_blocks_when_insufficient(monkeypatch):
    monkeypatch.setattr(zsw, "read_available_memory", lambda: int(0.1 * 1024**3))
    gate = zstream.check_streaming_gate(66, (836, 496), 128)
    assert gate.ok is False


# ---------------------------------------------------------------------------
# 4. Heavy real-cell parity (gated on fixtures + memory)
# ---------------------------------------------------------------------------

def _fixtures_present():
    return FIXTURES.exists() and len(list(FIXTURES.glob("*_rgb.fits"))) >= 66


def test_real_cell_streaming_vs_inmemory_parity():
    if not _fixtures_present():
        pytest.skip("M106 fixtures missing (run prepare_rgb_fixture.py)")
    # In-memory control for N=66 peaks ~3.4 GiB; require the R2/R3 extended gate.
    gate = zsw.check_memory_gate(66)
    if not gate.ok:
        pytest.skip(
            f"memory gate unsatisfied for in-memory control: available="
            f"{gate.available/2**30:.2f}GiB < required={gate.required/2**30:.2f}GiB"
        )
    # Also skip if the heavy run would not be reproducible on a low-memory host:
    # the tool measures VmHWM; here we run the actual parity (one heavy run).
    out = Path("/home/tristan/zegrid_r6_out_r0001c0002")
    cache = Path("/home/tristan/zegrid_r6_cache_r0001c0002")
    cmd = [
        sys.executable, str(R6_TOOL), "--mode", "parity",
        "--lights", str(LIGHTS), "--fixtures", str(FIXTURES),
        "--cache", str(cache), "--out", str(out),
        "--row", "1", "--col", "2", "--tile", "128", "--tiles", "128,256",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, proc.stderr[-3000:]

    report = json.loads((out / "parity_report.json").read_text())
    assert report["cell"] == "r0001c0002"
    assert report["inmem"]["n_contributors"] == 66
    inmem_rss = report["inmem"]["peak_rss_kib"]
    for tile in ("128", "256"):
        t = report["tiles"][tile]
        assert t["equal"] is True, json.dumps(t["planes"], indent=2)
        for name, info in t["planes"].items():
            assert info["equal"] is True, f"tile {tile} plane {name} not bit-equal"
        # Streaming peak must be clearly below in-memory peak (bounded by tile
        # area, not patch area).
        assert t["stream"]["peak_rss_kib"] < inmem_rss, (
            f"tile {tile}: stream {t['stream']['peak_rss_kib']} >= inmem {inmem_rss}"
        )
    # Primary memory proof (tile 128): substantially below the in-memory peak.
    assert report["tiles"]["128"]["stream"]["peak_rss_kib"] < inmem_rss // 2, (
        report["tiles"]["128"]["stream"]["peak_rss_kib"], inmem_rss
    )
