"""A2b: real FITS preparation -> source recipe -> cell stack, without N×patch cache.

No capacity claim: these small WCS witnesses qualify transport/preparation/parity.
Large-N scratch/RAM accounting and Windows qualification belong to later gates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS, Sip

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_streaming import compute_fixed_normalization
from zemosaic.core.zegrid import file_provider as fp
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid.science_adapter import MiniTileScienceConfig

PLANES = (
    "science_core", "estimator_weight_sum_core", "support_w1_core",
    "support_w2_core", "n_eff_support_core", "valid_mask_core",
    "surviving_sample_count_core",
)


@pytest.fixture(autouse=True)
def isolate_globals(monkeypatch):
    # Product worker initialization is intentionally global; restore every value
    # so this test cannot contaminate historical cache-only fixture tests.
    for name in tuple(vars(zz)):
        if name.startswith("_STACK_") or name.startswith("_EXTERNAL_"):
            monkeypatch.setattr(zz, name, getattr(zz, name))


def corpus(tmp_path, *, sip=False):
    descs = []
    rng = np.random.default_rng(17)
    for i in range(4):
        shape = (40, 44)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10 + i * 0.001, 30]
        wcs.wcs.crpix = [22.5, 20.5]
        wcs.wcs.cd = np.array([[-0.001, 0.00002], [0.00002, 0.001]])
        if sip:
            a = np.zeros((3, 3))
            b = np.zeros((3, 3))
            a[2, 0], b[0, 2] = 2e-5, -3e-5
            wcs.sip = Sip(a, b, None, None, wcs.wcs.crpix)
            wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        wcs.array_shape = shape
        data = (100 + 2 * i + rng.normal(0, 1, shape)).astype(np.float32)
        data[0:2, :3] = np.nan
        if i == 2:
            data[18, 20] = 1e5
        path = tmp_path / f"f{i}.fits"
        fits.PrimaryHDU(data, header=wcs.to_header(relax=True)).writeto(path)
        descs.append(zg.FrameDescriptor(
            frame_id=zg.FrameId(path.name), source_path=str(path),
            shape_hw=shape, wcs_header=wcs.to_header(relax=True).tostring(),
            header_sha256="", instrument="",
        ))
    canvas = zg.build_canvas(descs)
    layout = zg.build_layout(canvas, 2, 1)
    row, col, core = next(iter(layout.iter_cells(canvas)))
    cell = zg.ZeGridCell(
        cell_id=zg.cell_id(row, col), canvas_id=canvas.canvas_id,
        layout_id=layout.layout_id, row=row, col=col, core=core,
    )
    patch = zg.build_patch(canvas, cell, halo_px=5)
    mem = zg.compute_membership(descs, canvas, cell, patch)
    assert len(mem.patch_ids) == 4
    return descs, canvas, cell, patch, mem


def init(descs, canvas, cfg, *, gauge=None, ids=None):
    zz._init_stack_worker(descs, canvas, cfg, gauge, ids, None, 16, 1)


def no_cache(*args, **kwargs):
    raise AssertionError("source-backed task attempted to build an aligned cache")


def assert_result_equal(a, b):
    for key in PLANES:
        aa, bb = a["core"][key], b["core"][key]
        assert aa.dtype == bb.dtype, key
        np.testing.assert_array_equal(aa, bb, err_msg=key)
        assert np.ascontiguousarray(aa).tobytes() == np.ascontiguousarray(bb).tobytes(), key
    for key in ("reference_frame_id", "excluded", "n_contributors", "cell_gauge_mode"):
        assert a["record"][key] == b["record"][key]


@pytest.mark.parametrize("sip", [False, True])
@pytest.mark.parametrize("gauge_mode,rejection", [
    ("global", "kappa_sigma"), ("per_cell", "winsorized_sigma_clip"),
])
def test_real_fits_source_cell_matches_cache(tmp_path, monkeypatch, sip, gauge_mode, rejection):
    descs, canvas, cell, patch, mem = corpus(tmp_path, sip=sip)
    source_hashes = [hashlib.sha256(Path(f.source_path).read_bytes()).hexdigest() for f in descs]
    cfg = MiniTileScienceConfig(
        normalization="sky_mean", cell_gauge_mode=gauge_mode, rejection=rejection,
    )
    old_cache = tmp_path / "old"
    zz._build_one_cell_cache(descs, canvas, cell, patch, mem, old_cache, workers=1)
    with fp.MemmapCanonicalProvider(old_cache) as provider:
        gauge = compute_fixed_normalization(
            provider, zz.zstream.build_streaming_request(cfg, provider.n_frames)
        )
        ids = list(provider.frame_ids)
    init(descs, canvas, cfg, gauge=gauge, ids=ids)
    # Old bounded cache tuple is still supported, including normalization.
    reference = zz._stack_cell((cell, patch, mem, old_cache, "stream", 1, "bounded"))
    monkeypatch.setattr(zz, "_build_one_cell_cache", no_cache)
    monkeypatch.setattr(zz, "_build_aligned_cache_frame_major", no_cache)
    new_cache = tmp_path / "never-created"
    task = zz._cell_task(cell, patch, mem, new_cache, "stream", 1, "bounded")
    assert len(task) == 9 and zz._task_uses_windowed_sources(task)
    prepared = zz._prepare_stack_cell(pickle.loads(pickle.dumps(task)))
    assert not new_cache.exists()
    assert prepared["manifest"]["total_bytes"] == 0
    assert prepared["gauge_storage"]["provider"] == "WindowedCanonicalProvider"
    assert prepared["source_context"].frame_ids == tuple(ids)
    # No closure/provider/array needs to cross producer -> consumer.
    result = zz._consume_stack_cell(pickle.loads(pickle.dumps(prepared)))
    assert not new_cache.exists()
    assert result["total_bytes"] == 0
    storage = result["record"]["storage"]
    assert storage["provider"] == "WindowedCanonicalProvider"
    assert storage["aligned_cache_bytes_written"] == 0
    assert storage["prepared_count"] > 0
    assert storage["sink"]["type"] == "BoundedCountingRejectionSink"
    assert storage["sink"]["written_cells"] == len(ids) * np.prod(patch.patch_shape_hw) * 3
    assert_result_equal(reference, result)
    assert source_hashes == [hashlib.sha256(Path(f.source_path).read_bytes()).hexdigest() for f in descs]


def test_external_spool_consumes_source_recipe(tmp_path, monkeypatch):
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(normalization="sky_mean", cell_gauge_mode="per_cell")
    init(descs, canvas, cfg)
    ref = zz._stack_cell((cell, patch, mem, tmp_path / "cache-ref", "stream", 1, "bounded"))
    monkeypatch.setattr(zz, "_build_one_cell_cache", no_cache)
    root = tmp_path / "spool"
    root.mkdir()
    external = {
        "tile_size": 16, "column_batch": 48, "spool_root": str(root),
        "disk_envelope_bytes": 0,
    }
    task = zz._cell_task(cell, patch, mem, tmp_path / "no-cache", "external", 1,
                         "bounded", external_info=external)
    got = zz._stack_cell(task)
    assert_result_equal(ref, got)
    assert got["record"]["storage"]["provider"] == "WindowedCanonicalProvider"
    assert not (tmp_path / "no-cache").exists()
    assert not list(root.rglob("*.npy")), "spool payload must be recycled on success"


def test_external_caller_publishes_gpu_fallback(tmp_path, monkeypatch):
    """F8: the product caller collects and publishes the fallback truth."""
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(
        normalization="sky_mean", cell_gauge_mode="per_cell", backend="gpu",
    )
    init(descs, canvas, cfg)
    root = tmp_path / "spool"
    root.mkdir()
    real_reduce = zz.zext.reduce_spooled_tile
    calls = {"gpu": 0, "cpu": 0}

    _FakeCupyOOM = type(
        "OutOfMemoryError", (Exception,), {"__module__": "cupy.cuda.memory"}
    )

    def _always_oom(tile, p1, request, *, column_batch,
                    rejection_sink=None, rejection_mask=None, backend="cpu"):
        calls[backend] += 1
        if backend == "gpu":
            raise _FakeCupyOOM("simulated device OOM")
        return real_reduce(
            tile, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend,
        )

    monkeypatch.setattr(zz.zext, "reduce_spooled_tile", _always_oom)
    expected_tiles = len(
        list(zz.zext._iter_tiles(patch.patch_shape_hw[0],
                                 patch.patch_shape_hw[1], 16))
    )
    external = {
        "tile_size": 16, "column_batch": 48, "spool_root": str(root),
        "disk_envelope_bytes": 0,
    }
    task = zz._cell_task(cell, patch, mem, tmp_path / "no-cache", "external", 1,
                         "bounded", external_info=external)
    got = zz._stack_cell(task)
    storage = got["record"]["storage"]
    assert calls["gpu"] == 1
    assert calls["cpu"] == expected_tiles
    assert storage["gpu_fallback"]["fallback_count"] == 1
    assert storage["gpu_fallback"]["requested_backend"] == "gpu"
    assert storage["gpu_fallback"]["fallback_tiles"] == [(0, 0)]
    assert storage["provider"] == "WindowedCanonicalProvider"


def test_source_recipe_and_worker_initializer_under_spawn(tmp_path):
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(normalization="sky_mean", cell_gauge_mode="per_cell")
    init(descs, canvas, cfg)
    ref = zz._stack_cell((cell, patch, mem, tmp_path / "reference", "stream", 1, "bounded"))
    task = zz._cell_task(cell, patch, mem, tmp_path / "spawn-no-cache", "stream", 1, "bounded")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        initializer=zz._init_stack_worker,
        initargs=(descs, canvas, cfg, None, None, None, 16, 1),
    ) as pool:
        result = pool.submit(zz._stack_cell, task).result(timeout=90)
    assert_result_equal(ref, result)
    assert not (tmp_path / "spawn-no-cache").exists()


def test_source_failure_preserves_existing_cache_and_sources(tmp_path, monkeypatch):
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(normalization="sky_mean", cell_gauge_mode="per_cell")
    init(descs, canvas, cfg)
    cache = tmp_path / "not-owned"
    cache.mkdir()
    sentinel = cache / "keep.txt"
    sentinel.write_text("old artifact")
    task = zz._cell_task(cell, patch, mem, cache, "stream", 1, "bounded")
    monkeypatch.setattr(zz, "_build_one_cell_cache", no_cache)

    def fail(*args, **kwargs):
        raise OSError("source read failed")

    monkeypatch.setattr(zz, "_decode_frame_hwc", fail)
    with pytest.raises(OSError, match="source read failed"):
        zz._prepare_stack_cell(task)
    assert sentinel.read_text() == "old artifact"
    assert all(Path(f.source_path).is_file() for f in descs)


def test_source_strategy_validation_and_cleanup_ownership(tmp_path, monkeypatch):
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(normalization="sky_mean", cell_gauge_mode="per_cell")
    init(descs, canvas, cfg)
    cache = tmp_path / "untouched"
    cache.mkdir()
    marker = cache / "keep"
    marker.write_text("keep")
    task = zz._cell_task(cell, patch, mem, cache, "stream", 1, "bounded")
    monkeypatch.setattr(zz, "_build_one_cell_cache", no_cache)
    prepared = zz._prepare_stack_cell(task)
    assert zz._release_unit_cache(prepared) is False
    zz._consume_stack_cell(prepared)
    assert marker.read_text() == "keep"
    with pytest.raises(RuntimeError, match="bounded"):
        zz._make_cell_provider(cache, "dense", source_context=prepared["source_context"])
    with pytest.raises(ValueError, match="unknown cell input"):
        zz._prepare_stack_cell(task[:8] + ("unknown",))


def test_provider_releases_previous_window_before_preparation():
    holder = {}

    def prepare(i):
        if i:
            assert holder["provider"]._rgb is None
            assert holder["provider"]._index is None
            raise OSError("next frame failed")
        return np.ones((4, 5, 3), np.float32), np.ones((4, 5), bool)

    provider = fp.WindowedCanonicalProvider(prepare, 2, expected_hw=(4, 5))
    holder["provider"] = provider
    with pytest.raises(OSError, match="next frame"):
        provider.get_raw_frame(1)
    assert provider._rgb is None and provider._index is None
    provider.close()


def test_successful_stream_closes_window(tmp_path, monkeypatch):
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(normalization="sky_mean", cell_gauge_mode="per_cell")
    init(descs, canvas, cfg)
    prepared = zz._prepare_stack_cell(
        zz._cell_task(cell, patch, mem, tmp_path / "no-cache", "stream", 1, "bounded")
    )
    providers = []
    real_open = zz._CellSourceContext.open_provider

    def tracked(context):
        p = real_open(context)
        providers.append(p)
        return p

    monkeypatch.setattr(zz._CellSourceContext, "open_provider", tracked)
    zz._consume_stack_cell(prepared)
    assert len(providers) == 1
    assert providers[0]._closed and providers[0]._rgb is None


def test_real_cuda_consumes_source_recipe_bit_exact(tmp_path):
    pytest.importorskip("cupy")
    from zemosaic.core import canonical_stacking as cs
    if not cs.canonical_gpu_available():
        pytest.skip("physical CUDA device unavailable")
    descs, canvas, cell, patch, mem = corpus(tmp_path)
    cfg = MiniTileScienceConfig(
        normalization="sky_mean", cell_gauge_mode="per_cell", backend="gpu",
        rejection="winsorized_sigma_clip",
    )
    init(descs, canvas, cfg)
    cache = tmp_path / "gpu-reference"
    zz._build_one_cell_cache(descs, canvas, cell, patch, mem, cache, workers=1)
    prepared = zz._prepare_stack_cell(
        zz._cell_task(cell, patch, mem, tmp_path / "gpu-no-cache", "stream", 1, "bounded")
    )
    kwargs = dict(tile_size=16, fixed=prepared["cell_fixed"], storage_profile="bounded")
    cache_mt, cache_sres = zz._run_cell_stream(cache, patch, cfg, None, **kwargs)
    info = {}
    source_mt, source_sres = zz._run_cell_stream(
        tmp_path / "gpu-no-cache", patch, cfg, None,
        source_context=prepared["source_context"], storage_info=info, **kwargs,
    )
    for result in (cache_sres.result, source_sres.result):
        assert result.provenance["backend"]["requested"] == "gpu"
    aa, bb = zz.za.crop_all_planes_to_core(cache_mt), zz.za.crop_all_planes_to_core(source_mt)
    for name in PLANES:
        np.testing.assert_array_equal(aa[name], bb[name], err_msg=name)
        assert aa[name].tobytes() == bb[name].tobytes(), name
    assert info["provider"] == "WindowedCanonicalProvider"
    assert not (tmp_path / "gpu-no-cache").exists()
