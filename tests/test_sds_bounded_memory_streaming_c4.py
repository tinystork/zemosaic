"""ZM-SDS-R35C-C4 — bounded-memory SDS streaming: parity / memory / memmap tests.

Proves the SDS-specific file/memmap providers of
:mod:`zemosaic.core.sds_streaming` reproduce the current SDS science BIT FOR BIT
(vs the in-memory canonical engine) for several block sizes and scheduling
orders, that the providers are exact for supports/no-data, that memmaps are
closed explicitly, and that the concurrency/tile plan is driven by the EFFECTIVE
backend (CPU) and available RAM.

No network, no writes outside tmp_path, no ZSSS import. `/home/tristan/M106` is
never touched (the real-data witness lives in the durable report scratch).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core import sds_streaming as ss


# ---------------------------------------------------------------------------
# Deterministic corpus builders
# ---------------------------------------------------------------------------

def _final_corpus(n=5, h=72, w=64, c=3, seed=17):
    """Mega-tile-like RGB frames with stars, per-frame offsets and partial cov."""
    rng = np.random.default_rng(seed)
    mosaics, covs = [], []
    for i in range(n):
        m = rng.normal(600.0 + 15.0 * i, 4.0, (h, w, c)).astype(np.float32)
        for _ in range(3):
            cy = int(rng.integers(6, h - 6))
            cx = int(rng.integers(6, w - 6))
            m[cy - 1 : cy + 2, cx - 1 : cx + 2, :] += 350.0
        cov = np.ones((h, w), dtype=np.float32)
        cov[:, : i * 7] = 0.0  # partial footprint
        m[cov <= 0.0] = np.nan
        mosaics.append(m)
        covs.append(cov)
    return mosaics, covs


def _supports(covs):
    return [(np.asarray(c, dtype=np.float32) > 0.0) for c in covs]


def _inmem_final(mosaics, supports, ref_idx, rejection, combine):
    req = CanonicalStackRequest(
        images=mosaics,
        geometric_support=supports,
        normalization="none",
        weighting="none",
        rejection=rejection,
        combine=combine,
        reference_index=ref_idx,
        taper="none",
        backend="cpu",
    )
    return run_canonical_stack(req).science


def _write_patch_entries(tmpdir, n=4, h=110, w=96, c=3, seed=5):
    """Write SDS-style per-lot reprojected patches (data/weight .npy) + bboxes."""
    rng = np.random.default_rng(seed)
    entries = []
    for i in range(n):
        y0 = int(rng.integers(0, h - 40))
        y1 = min(h, y0 + int(rng.integers(50, 80)))
        x0 = int(rng.integers(0, w - 30))
        x1 = min(w, x0 + int(rng.integers(45, 90)))
        d = rng.normal(300.0 + 20.0 * i, 3.0, (y1 - y0, x1 - x0, c)).astype(np.float32)
        wt = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
        wt[:2, :] = 0.0
        dp = Path(tmpdir) / f"d{i}.npy"
        wp = Path(tmpdir) / f"w{i}.npy"
        np.save(dp, d)
        np.save(wp, wt)
        entries.append(
            {
                "data_path": str(dp),
                "weight_path": str(wp),
                "bbox": (int(y0), int(y1), int(x0), int(x1)),
                "frame_id": f"p{i}",
            }
        )
    return entries, h, w, c


def _inmem_band(entries, band_y0, band_y1, width, channels, rejection, combine, ref_idx):
    stack = np.full((len(entries), band_y1 - band_y0, width, channels), np.nan, np.float32)
    wstack = np.zeros((len(entries), band_y1 - band_y0, width), np.float32)
    for idx, e in enumerate(entries):
        d = np.load(e["data_path"])
        wt = np.load(e["weight_path"])
        ys, ye, xs, xe = e["bbox"]
        sy0, sy1 = max(band_y0, ys), min(band_y1, ye)
        if sy0 >= sy1:
            continue
        ly0, ly1 = sy0 - ys, sy1 - ys
        gy0, gy1 = sy0 - band_y0, sy1 - band_y0
        stack[idx, gy0:gy1, xs:xe, :] = d[ly0:ly1]
        wstack[idx, gy0:gy1, xs:xe] = wt[ly0:ly1]
    frames = [stack[i] for i in range(len(entries))]
    supports = [(wstack[i] > 0) for i in range(len(entries))]
    req = CanonicalStackRequest(
        images=frames,
        geometric_support=supports,
        normalization="none",
        weighting="none",
        rejection=rejection,
        combine=combine,
        reference_index=ref_idx,
        taper="none",
        backend="cpu",
    )
    return run_canonical_stack(req)


# ---------------------------------------------------------------------------
# 1. Final-stack parity (bit-exact) across block sizes / scheduling orders
# ---------------------------------------------------------------------------

_TILE_SIZES = [None, 64, 48, 32, 16, 7, (10, 13), (64, 32)]


@pytest.mark.parametrize("tile", _TILE_SIZES)
@pytest.mark.parametrize("rejection,combine", [
    ("winsorized_sigma_clip", "mean"),
    ("kappa_sigma", "mean"),
    ("none", "mean"),
])
def test_final_stack_streaming_bit_parity(tile, rejection, combine):
    mosaics, covs = _final_corpus()
    supports = _supports(covs)
    ref_idx = 0 if np.any(supports[0]) else None
    ref = _inmem_final(mosaics, supports, ref_idx, rejection, combine)
    with tempfile.TemporaryDirectory() as td:
        ss.write_sds_megatile_cache(
            td, [f"m{i}" for i in range(len(mosaics))], mosaics, supports
        )
        with ss.SdsMegatileFileProvider(td) as provider:
            got = ss.run_sds_final_stack_streaming(
                provider,
                {"stack_reject_algo": rejection, "stack_final_combine": combine},
                tile_size=tile,
            )
    assert np.array_equal(ref, got, equal_nan=True)
    assert np.count_nonzero(~np.isclose(ref, got, rtol=0, atol=0, equal_nan=True)) == 0


def test_final_stack_token_mapping_matches_worker_semantics():
    assert ss.resolve_sds_final_stack_tokens(
        {"stack_reject_algo": "winsorized_sigma_clip"}
    ) == ("winsorized_sigma_clip", "mean")
    assert ss.resolve_sds_final_stack_tokens({"stack_reject_algo": "winsor"}) == (
        "winsorized_sigma_clip",
        "mean",
    )
    assert ss.resolve_sds_final_stack_tokens({"stack_reject_algo": "kappa_sigma"}) == (
        "kappa_sigma",
        "mean",
    )
    assert ss.resolve_sds_final_stack_tokens(
        {"stack_reject_algo": "none", "stack_final_combine": "median"}
    ) == ("none", "median")
    assert ss.resolve_sds_final_stack_tokens(
        {"stack_reject_algo": "none", "stack_final_combine": "bogus"}
    ) == ("none", "mean")
    with pytest.raises(ValueError):
        ss.resolve_sds_final_stack_tokens({"stack_reject_algo": "linear_fit_clip"})


def test_final_stack_no_support_returns_all_nan():
    _, covs = _final_corpus(n=3)
    h, w = covs[0].shape
    nan_mosaics = [np.full((h, w, 3), np.nan, np.float32) for _ in covs]
    none_sup = [np.zeros((h, w), bool) for _ in covs]
    with tempfile.TemporaryDirectory() as td:
        ss.write_sds_megatile_cache(td, [f"z{i}" for i in range(3)], nan_mosaics, none_sup)
        with ss.SdsMegatileFileProvider(td) as provider:
            out = ss.run_sds_final_stack_streaming(provider, {"stack_reject_algo": "none"}, tile_size=16)
    assert out.shape == (h, w, 3)
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# 2. Band-stack parity (bit-exact) — brutes -> megatile stage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [None, 32, 16, 7])
@pytest.mark.parametrize("rejection", ["winsorized_sigma_clip", "kappa_sigma", "none"])
def test_band_stack_streaming_bit_parity(tile, rejection):
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td)
        band_y0, band_y1 = 0, h
        ref_idx = ss.sds_band_reference_index(entries, band_y0, band_y1, w, c)
        ref = _inmem_band(entries, band_y0, band_y1, w, c, rejection, "mean", ref_idx)
        ref_sci = np.where(np.isfinite(ref.science), ref.science, np.nan).astype(np.float32)
        ref_w = np.where(np.isfinite(ref.estimator_weight_sum), ref.estimator_weight_sum, 0.0)
        sci, wsum = ss.run_sds_band_stack_streaming(
            entries, band_y0, band_y1, w, c,
            rejection=rejection, combine="mean", reference_index=ref_idx, tile_size=tile,
        )
        assert np.array_equal(ref_sci, sci, equal_nan=True)
        assert np.array_equal(ref_w, wsum)


def test_band_reference_index_rule():
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=3)
        assert ss.sds_band_reference_index(entries, 0, h, w, c) == 0
        # A band above all bboxes has no support for frame 0 -> None.
        top = max(e["bbox"][1] for e in entries)
        assert ss.sds_band_reference_index(entries, top, top + 5, w, c) is None


# ---------------------------------------------------------------------------
# 2b. F1 (rework-0): band streaming fallback is explicit, non-silent, bit-exact
# ---------------------------------------------------------------------------

class _WarnCapture:
    def __init__(self):
        self.events = []

    def __call__(self, key, prog=None, lvl="INFO", **kwargs):
        self.events.append((key, lvl, kwargs))


def test_band_fallback_monkeypatched_is_bit_identical_and_warns(monkeypatch):
    """F1: a forced band-streaming failure must fall back in-memory, byte-identical."""
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=4)
        ref_idx = ss.sds_band_reference_index(entries, 0, h, w, c)
        # Reference = the SAME in-memory algorithm the worker's non-stream path uses.
        ref_sci, ref_w = ss._inmemory_band_stack(
            entries, 0, h, w, c, rejection="winsorized_sigma_clip", combine="mean"
        )

        def _boom(*a, **k):
            raise RuntimeError("forced band streaming failure")

        monkeypatch.setattr(ss, "run_sds_band_stack_streaming", _boom)
        cap = _WarnCapture()
        sci, wsum = ss.run_sds_band_stack_with_fallback(
            entries, 0, h, w, c,
            rejection="winsorized_sigma_clip", combine="mean",
            reference_index=ref_idx, tile_size=32, pcb=cap,
        )
        # Byte-identical to the in-memory path (no tile lost, no science change).
        assert np.array_equal(ref_sci, sci, equal_nan=True)
        assert np.array_equal(ref_w, wsum)
        keys = [(k, lvl) for (k, lvl, _kw) in cap.events]
        assert ("sds_band_stack_streaming_fallback", "WARN") in keys
        warn_kw = [kw for (k, lvl, kw) in cap.events if k == "sds_band_stack_streaming_fallback"][0]
        assert warn_kw["reason"] == "streaming_unavailable_falling_back_in_memory"


def test_band_fallback_not_triggered_when_streaming_succeeds():
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=3)
        ref_idx = ss.sds_band_reference_index(entries, 0, h, w, c)
        cap = _WarnCapture()
        sci, wsum = ss.run_sds_band_stack_with_fallback(
            entries, 0, h, w, c,
            rejection="kappa_sigma", combine="mean",
            reference_index=ref_idx, tile_size=32, pcb=cap,
        )
        assert not any(k == "sds_band_stack_streaming_fallback" for (k, _l, _kw) in cap.events)
        # Still bit-equal to the in-memory reference.
        ref_sci, ref_w = ss._inmemory_band_stack(
            entries, 0, h, w, c, rejection="kappa_sigma", combine="mean"
        )
        assert np.array_equal(ref_sci, sci, equal_nan=True)
        assert np.array_equal(ref_w, wsum)


def test_band_fallback_memoryerror_is_reraised():
    """MemoryError must reach the caller's band-height reduction (not be swallowed)."""
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=3)

        def _oom(*a, **k):
            raise MemoryError("forced")

        import pytest as _pytest

        orig = ss.run_sds_band_stack_streaming
        ss.run_sds_band_stack_streaming = _oom
        try:
            with _pytest.raises(MemoryError):
                ss.run_sds_band_stack_with_fallback(
                    entries, 0, h, w, c, rejection="none", combine="mean",
                    reference_index=None, tile_size=32,
                )
        finally:
            ss.run_sds_band_stack_streaming = orig


# ---------------------------------------------------------------------------
# 2c. I1 (rework-0): mono / HW=1 parity (band + final) + empty-support band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [None, 16, 7])
def test_band_mono_bit_parity(tile):
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=4, c=1)
        assert c == 1
        ref_idx = ss.sds_band_reference_index(entries, 0, h, w, c)
        ref = _inmem_band(entries, 0, h, w, c, "winsorized_sigma_clip", "mean", ref_idx)
        ref_sci = np.where(np.isfinite(ref.science), ref.science, np.nan).astype(np.float32)
        ref_w = np.where(np.isfinite(ref.estimator_weight_sum), ref.estimator_weight_sum, 0.0)
        sci, wsum = ss.run_sds_band_stack_streaming(
            entries, 0, h, w, c,
            rejection="winsorized_sigma_clip", combine="mean", reference_index=ref_idx,
            tile_size=tile,
        )
        assert np.array_equal(ref_sci, sci, equal_nan=True)
        assert np.array_equal(ref_w, wsum)


def test_final_stack_mono_bit_parity():
    rng = np.random.default_rng(23)
    n, h, w = 5, 48, 40
    mosaics, covs = [], []
    for i in range(n):
        m = rng.normal(500.0 + 8.0 * i, 3.0, (h, w)).astype(np.float32)  # 2-D mono
        cov = np.ones((h, w), np.float32)
        cov[:, : i * 6] = 0.0
        m[cov <= 0] = np.nan
        mosaics.append(m)
        covs.append(cov)
    supports = [(np.asarray(c, np.float32) > 0) for c in covs]
    ref_idx = 0 if np.any(supports[0]) else None
    req = CanonicalStackRequest(
        images=mosaics, geometric_support=supports, normalization="none", weighting="none",
        rejection="winsorized_sigma_clip", combine="mean", reference_index=ref_idx,
        taper="none", backend="cpu",
    )
    ref = run_canonical_stack(req).science
    assert ref.ndim == 2
    with tempfile.TemporaryDirectory() as td:
        ss.write_sds_megatile_cache(td, [f"m{i}" for i in range(n)], mosaics, supports)
        with ss.SdsMegatileFileProvider(td) as provider:
            assert provider.original_mono is True
            got = ss.run_sds_final_stack_streaming(
                provider, {"stack_reject_algo": "winsorized_sigma_clip"}, tile_size=13
            )
    assert got.ndim == 2
    assert np.array_equal(ref, got, equal_nan=True)


def test_band_empty_common_support_matches_inmemory():
    """A band with NO support in any entry is refused IDENTICALLY by both paths.

    The frozen canonical engine raises ``CanonicalStackFailure`` when every frame
    has zero valid support, for the in-memory AND the streaming executor. The C4
    requirement is that the streaming path does not diverge: same failure, and the
    fallback helper does not mask it.
    """
    from zemosaic.core.canonical_stacking import CanonicalStackFailure

    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=3)
        top = max(e["bbox"][1] for e in entries)
        band_y0, band_y1 = top + 10, top + 40
        ref_idx = ss.sds_band_reference_index(entries, band_y0, band_y1, w, c)
        assert ref_idx is None
        with pytest.raises(CanonicalStackFailure):
            _inmem_band(entries, band_y0, band_y1, w, c, "none", "mean", ref_idx)
        with pytest.raises(CanonicalStackFailure):
            ss.run_sds_band_stack_streaming(
                entries, band_y0, band_y1, w, c,
                rejection="none", combine="mean", reference_index=ref_idx, tile_size=16,
            )
        # The fallback helper re-raises the SAME failure (no silent success).
        cap = _WarnCapture()
        with pytest.raises(CanonicalStackFailure):
            ss.run_sds_band_stack_with_fallback(
                entries, band_y0, band_y1, w, c,
                rejection="none", combine="mean", reference_index=ref_idx,
                tile_size=16, pcb=cap,
            )


def test_band_provider_taper_guard():
    """I2: the band provider refuses any taper != 'none' explicitly."""
    from zemosaic.core.canonical_stacking import CanonicalStackValidationError

    with pytest.raises(CanonicalStackValidationError):
        ss.SdsPatchBandProvider([], 0, 1, 4, 3, taper="footprint")
    # And the runner forwards the guard.
    with tempfile.TemporaryDirectory() as td:
        entries, h, w, c = _write_patch_entries(td, n=2)
        with pytest.raises(CanonicalStackValidationError):
            ss.run_sds_band_stack_streaming(
                entries, 0, h, w, c, rejection="none", combine="mean",
                reference_index=None, tile_size=16, taper="footprint",
            )


# ---------------------------------------------------------------------------
# 3. Cache round-trip exactness + explicit memmap close
# ---------------------------------------------------------------------------

def test_cache_roundtrip_is_bit_exact_and_closes_memmaps():
    rng = np.random.default_rng(3)
    mosaics = [rng.normal(0, 1, (40, 33, 3)).astype(np.float32) for _ in range(4)]
    mosaics[1][5:9, 5:9, :] = np.nan
    mosaics[1][0, 0, 0] = np.inf  # non-finite payload preserved
    covs = [np.ones((40, 33), np.float32) for _ in range(4)]
    covs[2][:, :5] = 0.0
    supports = _supports(covs)
    with tempfile.TemporaryDirectory() as td:
        ss.write_sds_megatile_cache(td, [f"m{i}" for i in range(4)], mosaics, supports)
        provider = ss.SdsMegatileFileProvider(td)
        for i in range(4):
            raw, sup = provider.get_raw_frame(i)
            np.testing.assert_array_equal(raw, mosaics[i])
            np.testing.assert_array_equal(sup, supports[i])
            assert raw.dtype == np.float32 and sup.dtype == bool
        provider.close()
        assert provider._rgb == [] and provider._sup == []
        # A second close is idempotent.
        provider.close()
        # After close, the scratch files are deletable (Windows WinError-32 guard).
        import shutil

        shutil.rmtree(td)  # must not raise


def test_provider_context_manager_closes_on_error():
    mosaics, covs = _final_corpus(n=3)
    supports = _supports(covs)
    with tempfile.TemporaryDirectory() as td:
        ss.write_sds_megatile_cache(td, [f"m{i}" for i in range(3)], mosaics, supports)
        provider = ss.SdsMegatileFileProvider(td)
        try:
            assert provider._rgb  # open
            raise RuntimeError("boom")
        except RuntimeError:
            pass
        finally:
            provider.close()
        assert provider._rgb == []


# ---------------------------------------------------------------------------
# 4. Concurrency / tile plan driven by effective backend + RAM
# ---------------------------------------------------------------------------

def test_plan_tile_degrades_under_low_ram_and_grows_with_ram():
    low = ss.plan_sds_streaming_tile(7, (3331, 3615), 3, available_bytes=64 * 1024 * 1024)
    high = ss.plan_sds_streaming_tile(7, (3331, 3615), 3, available_bytes=64 * 1024**3)
    assert low >= 64
    assert high > low
    # Never larger than the domain's longest side.
    assert high <= 3615


def test_plan_parallelism_not_raised_by_gpu_preference():
    common = dict(available_bytes=64 * 1024**3, est_bytes_per_megatile=200 * 1024**2, cpu_workers_hint=8)
    cpu_plan = ss.plan_sds_megatile_parallelism(7, requested_use_gpu=False, **common)
    gpu_plan = ss.plan_sds_megatile_parallelism(7, requested_use_gpu=True, **common)
    assert gpu_plan <= cpu_plan  # a GPU preference never RAISES the CPU bound
    # RAM bound respected: ~64 GiB*0.5 / 200 MiB = 163 (>> 7).
    assert 1 <= gpu_plan <= 7


def test_plan_parallelism_ram_bound_is_conservative():
    plan = ss.plan_sds_megatile_parallelism(
        7, requested_use_gpu=False, available_bytes=300 * 1024**2, est_bytes_per_megatile=200 * 1024**2
    )
    assert plan == 1  # one 200 MiB tile at most within half of 300 MiB
    assert ss.plan_sds_megatile_parallelism(1, requested_use_gpu=False, available_bytes=0) == 1


# ---------------------------------------------------------------------------
# 5. Worker wiring guards (static)
# ---------------------------------------------------------------------------

def _worker_source() -> str:
    return (
        Path(__file__).resolve().parents[1] / "src" / "zemosaic" / "zemosaic_worker.py"
    ).read_text()


def test_worker_wires_both_sds_stages_to_streaming_and_is_gated():
    src = _worker_source()
    # Both stages import/call the SDS streaming module.
    assert "zemosaic.core.sds_streaming" in src
    assert "run_sds_final_stack_streaming" in src
    assert "run_sds_band_stack_with_fallback" in src  # F1: fallback-aware band call
    # Stage-1 streaming is gated on a flag (Classic path stays byte-identical).
    assert "sds_stream_band" in src
    assert "sds_stream_band: bool = False" in src
    # Flags default ON for SDS but are overridable.
    assert 'sds_band_stack_streaming' in src
    assert 'sds_final_stack_streaming' in src
    # The non-SDS wrapper must NOT opt in.
    assert "sds_stream_band=True" not in src.replace(" ", "")


def test_worker_does_not_silently_swallow_streaming_errors():
    src = _worker_source()
    # Stage 2 (megatiles -> final): explicit WARN fallback.
    assert "sds_final_stack_streaming_fallback" in src
    assert "streaming_unavailable_falling_back_in_memory" in src
    # Stage 1 (brutes -> megatile): F1 fix — the worker routes through the
    # fallback-aware helper (which owns the WARN + in-memory re-run).
    assert "run_sds_band_stack_with_fallback" in src
    band_src = (
        Path(__file__).resolve().parents[1] / "src" / "zemosaic" / "core" / "sds_streaming.py"
    ).read_text()
    assert "sds_band_stack_streaming_fallback" in band_src
    assert "streaming_unavailable_falling_back_in_memory" in band_src
    assert "def run_sds_band_stack_with_fallback" in band_src
    assert "except MemoryError:\n        raise" in band_src  # OOM still reaches chunk reduction


# ---------------------------------------------------------------------------
# 6. MEASURED memory bound (heavy, opt-in via ZM_SDS_C4_HEAVY=1)
# ---------------------------------------------------------------------------

_MEM_BENCH = os.path.join(
    os.path.dirname(__file__), "..", "tools", "sds_c4", "mem_bench.py"
)


def _available_mib():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


def _run_bench(mode, tile, n, h, w):
    argv = [sys.executable, os.path.abspath(_MEM_BENCH), "--mode", mode,
            "--n", str(n), "--h", str(h), "--w", str(w)]
    if tile is not None:
        argv += ["--tile", str(tile)]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"mem_bench {mode} failed: {out.stderr[:2000]}")
    for tok in out.stdout.split():
        if tok.startswith("added_peak_kib="):
            return int(tok.split("=", 1)[1])
    raise RuntimeError(f"mem_bench {mode}: no added_peak_kib: {out.stdout!r}")


@pytest.mark.skipif(
    os.environ.get("ZM_SDS_C4_HEAVY") != "1",
    reason="opt-in heavy C4 memory-bound test (set ZM_SDS_C4_HEAVY=1 to run)",
)
def test_measured_memory_bound_final_stack():
    n, h, w = 7, 1280, 1280  # SDS final-stack shape class (real: 7 x 3331 x 3615)
    avail = _available_mib()
    if avail < 2600:
        pytest.skip(f"insufficient RAM for in-memory control ({avail} MiB < 2600 MiB)")
    inmem = _run_bench("inmem", None, n, h, w)
    stream_256 = _run_bench("stream", 256, n, h, w)
    stream_128 = _run_bench("stream", 128, n, h, w)
    assert stream_256 < inmem / 3, (stream_256, inmem)
    assert stream_128 < stream_256  # smaller tile -> smaller workspace
