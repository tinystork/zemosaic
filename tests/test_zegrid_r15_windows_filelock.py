"""ZM-ZEGRID-R15 targeted tests — Windows file-lock (WinError 32) safety.

Reproduces WINDOWS semantics on Linux: a memmap/``.npy`` still open when the
cache is deleted raises ``WinError 32`` (deleting an in-use file is a no-op on
Linux, so the R12 tests were green while production crashed on Windows).

Covers (non-gated, fast):

* handle release: ``MemmapCanonicalProvider.close()`` / context manager release
  every memmap's underlying ``mmap`` handle;
* best-effort deletion: ``_safe_rmtree`` / ``file_provider.safe_rmtree`` retry on
  a transient lock, then WARN + continue (never raise) on a persistent lock;
* the cell-stack paths close their provider before the caller deletes the cache;
* results stay BIT-EQUAL (deletion is an optimisation, not part of the science).

Gated (M106): a short end-to-end that deletes the per-cell cache with the
deletion forced to FAIL and asserts the run completes with a WARN and
byte-identical science output.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_streaming import run_canonical_stack_streaming
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import streaming as zstream
from zemosaic.core.zegrid import sweep as zsw

LIGHTS = Path("/home/tristan/M106/lights")


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


def _write_cache(cache, n=6):
    frames, masks = _aligned_corpus(n)
    ids = [f"frame_{i}" for i in range(n)]
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    return frames, masks, ids


def _assert_minitile_bit_exact(ref, st):
    np.testing.assert_array_equal(ref.science, st.science)
    np.testing.assert_array_equal(ref.estimator_weight_sum, st.estimator_weight_sum)
    np.testing.assert_array_equal(ref.support_w1, st.support_w1)
    np.testing.assert_array_equal(ref.support_w2, st.support_w2)
    np.testing.assert_array_equal(ref.n_eff_support, st.n_eff_support)
    np.testing.assert_array_equal(ref.valid_mask, st.valid_mask)
    np.testing.assert_array_equal(ref.surviving_sample_count, st.surviving_sample_count)


class _FakePatch:
    """Minimal ProcessingPatch for the cell-stack helpers (extract_minitile only)."""

    def __init__(self, h, w):
        self.cell_id = "r0000c0000"
        self.patch_shape_hw = (h, w)
        self.core_slice = SimpleNamespace(y0=0, y1=h, x0=0, x1=w)


# ---------------------------------------------------------------------------
# Handle release
# ---------------------------------------------------------------------------

def test_provider_close_releases_memmap_handles(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache)
    prov = zfp.MemmapCanonicalProvider(cache)
    # Every frame has an open memmap base before close.
    assert all(getattr(a, "base", None) is not None for a in prov._rgb)
    prov.close()
    # After close: the arrays are dropped and the handles are closed.
    assert prov._rgb == [] and prov._sup == []
    # The cache can now be deleted (even on Windows the handles are released).
    zfp.safe_rmtree(cache)
    assert not cache.exists()


def test_provider_context_manager_closes(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache)
    with zfp.MemmapCanonicalProvider(cache) as prov:
        assert prov.n_frames == 6
    assert prov._rgb == [] and prov._sup == []


def test_provider_close_idempotent(tmp_path):
    cache = tmp_path / "cache"
    _write_cache(cache)
    prov = zfp.MemmapCanonicalProvider(cache)
    prov.close()
    prov.close()  # must not raise
    assert prov._rgb == []


# ---------------------------------------------------------------------------
# Best-effort deletion (never fatal)
# ---------------------------------------------------------------------------

def test_safe_rmtree_deletes_existing_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    (d / "x.npy").write_bytes(b"1234")
    assert zz._safe_rmtree(d) is True
    assert not d.exists()


def test_safe_rmtree_missing_path_is_ok(tmp_path):
    assert zz._safe_rmtree(tmp_path / "nope") is True


def test_safe_rmtree_retries_then_succeeds(monkeypatch, tmp_path):
    """Simulate a transient Windows lock: first rmtree raises WinError 32, then
    the handle is released and deletion succeeds."""
    d = tmp_path / "cache"
    d.mkdir()
    (d / "x.npy").write_bytes(b"1")

    real = shutil.rmtree
    calls = {"n": 0}

    def flaky(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "[WinError 32] the process cannot access the file")
        return real(path, *a, **k)

    monkeypatch.setattr(zz.shutil, "rmtree", flaky)
    assert zz._safe_rmtree(d) is True
    assert not d.exists()
    assert calls["n"] >= 2


def test_safe_rmtree_non_fatal_warns_and_continues(monkeypatch, tmp_path):
    """A PERSISTENT lock (rmtree always raises) must WARN and continue, never raise."""
    d = tmp_path / "cache"
    d.mkdir()
    (d / "x.npy").write_bytes(b"1")

    def boom(path, *a, **k):
        raise PermissionError(13, "[WinError 32] the process cannot access the file")

    monkeypatch.setattr(zz.shutil, "rmtree", boom)

    warns = []
    result = zz._safe_rmtree(d, progress_callback=lambda m, p, lvl, **kw: warns.append((m, lvl)))
    assert result is False          # deletion did not succeed…
    assert any("cleanup FAILED" in m and lvl == "WARN" for m, lvl in warns)  # …but WARNed
    assert d.exists()               # and did NOT raise (run would continue)


def test_file_provider_safe_rmtree_non_fatal(monkeypatch, tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    (d / "x.npy").write_bytes(b"1")
    monkeypatch.setattr(zfp.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(
        PermissionError(13, "[WinError 32]")))
    assert zfp.safe_rmtree(d) is False
    assert d.exists()


# ---------------------------------------------------------------------------
# Cell-stack paths close the provider before deletion
# ---------------------------------------------------------------------------

def test_run_cell_stream_closes_provider(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _write_cache(cache)
    closed = []
    orig_close = zfp.MemmapCanonicalProvider.close

    def spy_close(self):
        closed.append(True)
        orig_close(self)

    monkeypatch.setattr(zfp.MemmapCanonicalProvider, "close", spy_close)
    cfg = zz.ExecutorConfig().science_config()
    mt, _sres = zz._run_cell_stream(cache, _FakePatch(40, 40), cfg, None)
    assert closed, "provider.close() was not called by _run_cell_stream"
    assert mt.science.shape[:2] == (40, 40)


def test_run_cell_inmem_closes_provider(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _write_cache(cache)
    closed = []
    orig_close = zfp.MemmapCanonicalProvider.close

    def spy_close(self):
        closed.append(True)
        orig_close(self)

    monkeypatch.setattr(zfp.MemmapCanonicalProvider, "close", spy_close)
    cfg = zz.ExecutorConfig().science_config()
    mt, _sres = zz._run_cell_inmem(cache, _FakePatch(40, 40), cfg, None)
    assert closed, "provider.close() was not called by _run_cell_inmem"
    assert mt.science.shape[:2] == (40, 40)


def test_run_cell_inmem_closes_provider_on_read_error(tmp_path, monkeypatch):
    """rework-1 (L1): if the read/materialise loop RAISES, the provider is STILL
    closed (try/finally) so the caller can delete the cache on Windows."""
    cache = tmp_path / "cache"
    _write_cache(cache)
    closed = []
    orig_close = zfp.MemmapCanonicalProvider.close

    def spy_close(self):
        closed.append(True)
        orig_close(self)

    monkeypatch.setattr(zfp.MemmapCanonicalProvider, "close", spy_close)

    def boom(self, i):
        raise RuntimeError("simulated decode/read failure")

    monkeypatch.setattr(zfp.MemmapCanonicalProvider, "get_raw_frame", boom)
    cfg = zz.ExecutorConfig().science_config()
    with pytest.raises(RuntimeError):
        zz._run_cell_inmem(cache, _FakePatch(40, 40), cfg, None)
    # The read loop raised, but the provider handle was STILL released.
    assert closed, "provider.close() was NOT called when the read loop raised"
    # And the cache is then deletable (no lingering handle).
    assert zfp.safe_rmtree(cache) is True
    assert not cache.exists()


def test_cell_stream_result_bit_equal_to_inmemory(tmp_path):
    """Closing the provider in ``_run_cell_stream`` must not change the science."""
    cache = tmp_path / "cache"
    frames, masks, _ids = _write_cache(cache)
    cfg = zz.ExecutorConfig().science_config()

    # In-memory reference engine on the same aligned arrays.
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    ref = run_canonical_stack(req)

    mt, _sres = zz._run_cell_stream(cache, _FakePatch(40, 40), cfg, None)
    _assert_minitile_bit_exact(ref, mt)


def test_run_cell_streaming_closes_provider(tmp_path, monkeypatch):
    """rework-1 (L2): ``run_cell_streaming`` (streaming.py) must close its
    ``MemmapCanonicalProvider``, so it is safe on Windows if promoted."""
    cache = tmp_path / "cache"
    frames, masks, ids = _write_cache(cache)

    # Skip the heavy decode/reproject + membership; serve the pre-written cache.
    class _Cell:
        cell_id = "r0000c0000"

    class _Mem:
        patch_ids = ids  # non-empty

    monkeypatch.setattr(
        zsw, "build_cell_context",
        lambda *a, **k: (_Cell(), _FakePatch(40, 40), _Mem()),
    )
    monkeypatch.setattr(zg, "plan_source_roi", lambda *a, **k: None)
    monkeypatch.setattr(
        zstream, "build_aligned_cache_from_sources",
        lambda *a, **k: zfp.load_cache_manifest(cache),
    )

    closed = []
    orig_close = zfp.MemmapCanonicalProvider.close

    def spy_close(self):
        closed.append(True)
        orig_close(self)

    monkeypatch.setattr(zfp.MemmapCanonicalProvider, "close", spy_close)

    fake_frames = [SimpleNamespace(frame_id=SimpleNamespace(logical_path=i)) for i in ids]
    cfg = zz.ExecutorConfig().science_config()
    res = zstream.run_cell_streaming(
        fake_frames, None, {}, 0, 0, cfg, cache, enforce_gate=False,
    )
    assert closed, "run_cell_streaming did not close its provider"
    # The result is unchanged (valid MiniTile produced through the same path).
    assert res.minitile.science.shape[:2] == (40, 40)


def test_stream_bit_equal_after_provider_close(tmp_path):
    """Explicitly closing a provider (as the cell paths now do) leaves results
    bit-equal — the R15 handle-release does not perturb served data."""
    cache = tmp_path / "cache"
    frames, masks, _ids = _write_cache(cache)
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    ref = run_canonical_stack(req)

    prov = zfp.MemmapCanonicalProvider(cache)
    try:
        st = run_canonical_stack_streaming(prov, req, tile_size=9)
    finally:
        prov.close()
    _assert_minitile_bit_exact(ref, st)


# ---------------------------------------------------------------------------
# L3: manifest carries cache.cleanup_failures (auditable bounded-disk promise)
# ---------------------------------------------------------------------------

def test_manifest_carries_cleanup_failures(tmp_path):
    """rework-1 (L3): the manifest ``cache`` block records ``cleanup_failures``
    (count + failed paths), so a persistent lock is auditable, not just a log line.
    """
    import json

    class _Assembled:
        science = np.zeros((10, 10, 3), dtype=np.float32)
        stack_depth = np.ones((10, 10), dtype=np.int32)
        complete_cells = 1
        incomplete_cells = 0
        hole_pixels = 0
        coverage_pixels = 100

    wcs = zg.WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 30.0]
    wcs.wcs.crpix = [5.0, 5.0]
    wcs.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    wcs.array_shape = (10, 10)
    canvas = zg.GlobalCanvas(
        canvas_id="x", wcs_header=wcs.to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    descs = []

    # Failure case: cache_report records two failed cleanups.
    failed_paths = [str(tmp_path / "a.npy"), str(tmp_path / "b.npy")]
    cache_report = {
        "total_bytes": 100, "n_frame_entries": 2, "peak_cell_bytes": 50,
        "cleanup_failures": failed_paths,
    }
    _, _, manifest_path = zz._write_outputs(
        _Assembled(), canvas, 1, 1, tmp_path, descs, {}, [],
        layout, SimpleNamespace(normalization="sky_mean"), 0, cache_report, None,
    )
    manifest = json.loads(manifest_path.read_text())
    cf = manifest["cache"]["cleanup_failures"]
    assert cf["count"] == 2
    assert cf["paths"] == failed_paths

    # No-failure case: count 0, empty paths.
    cache_report_ok = {"total_bytes": 100, "n_frame_entries": 2, "peak_cell_bytes": 50}
    _, _, manifest_path2 = zz._write_outputs(
        _Assembled(), canvas, 1, 1, tmp_path, descs, {}, [],
        layout, SimpleNamespace(normalization="sky_mean"), 0, cache_report_ok, None,
    )
    manifest2 = json.loads(manifest_path2.read_text())
    cf2 = manifest2["cache"]["cleanup_failures"]
    assert cf2["count"] == 0
    assert cf2["paths"] == []


# ---------------------------------------------------------------------------
# Deletion failure does not change a subsequent identical rebuild (bit-equal)
# ---------------------------------------------------------------------------

def test_deletion_failure_does_not_change_result(tmp_path, monkeypatch):
    """The cache deletion is an optimisation: even when it fails (leftover cache),
    a subsequent identical rebuild reproduces the same aligned bytes, so the
    science is unchanged."""
    cache_a = tmp_path / "a"
    frames, masks, ids = _write_cache(cache_a)

    # Reference: streaming over the freshly-built cache.
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    ref = run_canonical_stack(req)
    prov = zfp.MemmapCanonicalProvider(cache_a)
    try:
        st_ref = run_canonical_stack_streaming(prov, req, tile_size=9)
    finally:
        prov.close()
    _assert_minitile_bit_exact(ref, st_ref)

    # Force deletion to fail on a SECOND cache, leaving leftover files.
    monkeypatch.setattr(zfp.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(
        PermissionError(13, "[WinError 32]")))

    cache_b = tmp_path / "b"
    zfp.write_aligned_cache_from_arrays(cache_b, ids, frames, masks)
    assert zfp.safe_rmtree(cache_b) is False  # deletion failed, cache remains

    # Rebuild the cache over the leftover (best-effort wipe + rewrite), then read
    # it back and confirm the aligned data is bit-identical to the reference.
    monkeypatch.undo()
    zfp.write_aligned_cache_from_arrays(cache_b, ids, frames, masks)
    prov_b = zfp.MemmapCanonicalProvider(cache_b)
    try:
        st_b = run_canonical_stack_streaming(prov_b, req, tile_size=9)
    finally:
        prov_b.close()
    _assert_minitile_bit_exact(ref, st_b)


# ---------------------------------------------------------------------------
# Gated end-to-end: per-cell delete failure -> run completes, science bit-equal
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_end_to_end_delete_failure_non_fatal_and_bit_equal(tmp_path, monkeypatch):
    import shutil as _sh

    frames = sorted(LIGHTS.glob("*.fit"))[:6]
    assert len(frames) == 6

    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in frames:
        _sh.copy2(p, input_dir / p.name)
    lines = ["file_path,exposure"] + [f"{p.name},10.0" for p in frames]
    (input_dir / "stack_plan.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zconfig = SimpleNamespace(zegrid_layout="2x2", zegrid_sip_mode="keep")

    def _run(out, progress_callback=None):
        zz.run_zegrid_mode(
            str(input_dir), str(out), progress_callback=progress_callback,
            zconfig=zconfig, workers=1,
        )
        with __import__("astropy.io.fits", fromlist=["fits"]).open(
            out / "mosaic_grid.fits"
        ) as hdul:
            return bytes(hdul[0].data.tobytes())

    # Control run (deletion succeeds).
    sci_control = _run(tmp_path / "ctrl")

    # Failure run: force every rmtree to raise WinError 32.
    def boom(path, *a, **k):
        raise PermissionError(13, "[WinError 32] the process cannot access the file")

    monkeypatch.setattr(zz.shutil, "rmtree", boom)
    monkeypatch.setattr(zfp.shutil, "rmtree", boom)

    warns = []

    def _cb(m, p, lvl, **kw):
        warns.append((m, lvl))

    sci_fail = _run(tmp_path / "fail", progress_callback=_cb)

    # The run completed (no raise) with byte-identical science.
    assert sci_fail == sci_control
    # And a WARN about the failed cleanup was emitted.
    assert any("cleanup FAILED" in m and lvl == "WARN" for m, lvl in warns)
