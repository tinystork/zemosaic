"""ZM-M33-PREP-PIPELINE — bounded two-producer preparation acceptance.

Scientific contract: the pipelined writer must produce the EXACT same spool
payload (and therefore the exact same reduction) as the historical sequential
writer, on real FITS sources including Bayer CFA, mono, non-finite inputs,
unknown CFA fallback, edge/halo windows and non-trivial coefficients.

Adversarial contract: budgets are refused honestly, producer/writer failures
preserve the original exception, out-of-order completion never permutes an
index, and a non-drained producer quarantines the provider set instead of
claiming a successful cleanup.
"""

from __future__ import annotations

import gc
import json
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    _fixed_phase1,
    run_canonical_stack_streaming,
)
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.zegrid import external_stream as zext
from zemosaic.core.zegrid import cell_status as zcs
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import prepare_pipeline as zpp
from zemosaic.core.zegrid import streaming as zstream
from zemosaic.core.zegrid.science_adapter import MiniTileScienceConfig

MIB = 2 ** 20


# ---------------------------------------------------------------------------
# Synthetic provider corpus
# ---------------------------------------------------------------------------

def _inmem_corpus(n=7, h=48, w=40, mono=False, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 15.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / 90.0)
    frames = []
    for i in range(n):
        gain = 0.95 + 0.02 * i
        if mono:
            frame = base * gain + rng.normal(0, 2, (h, w))
        else:
            frame = np.stack(
                [base * gain * (0.98 + 0.02 * c) + rng.normal(0, 2, (h, w))
                 for c in range(3)], axis=-1)
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    frames[1][3:9, 4:10] = np.nan
    if n >= 4:
        # A genuinely INACTIVE contribution: the loader/normalizer excludes it.
        frames[3][:] = np.nan
    masks = [np.ones((h, w), bool) for _ in range(n)]
    return frames, masks


def _request(n, *, mono=False, rejection="winsorized_sigma_clip", combine="mean"):
    return CanonicalStackRequest(
        images=[None] * n,
        geometric_support=[None] * n,
        normalization="linear_fit",
        weighting="noise_variance",
        rejection=rejection,
        combine=combine,
        taper="footprint",
        taper_px=8.0,
    )


def _phase1(frames, masks, request):
    return _fixed_phase1(InMemoryCanonicalProvider(frames, masks), request, None)


def _grant(h, w, tile=24, halo=4, source_hw=None, c=3):
    return zpp.preparation_grant_components(
        patch_hw=(h, w), tile_hw=(tile + 2 * halo, tile + 2 * halo),
        source_hw=source_hw or (h, w), channels=c,
    )["peak_bytes"]


def _run_pipeline_cell(tmp_path, frames, masks, *, tile, column_batch=8,
                       request=None, producers=2, attempt="pipe"):
    n, h, w = len(frames), frames[0].shape[0], frames[0].shape[1]
    c = 1 if frames[0].ndim == 2 else frames[0].shape[-1]
    request = request or _request(n)
    grant = _grant(h, w, tile=tile, source_hw=(h, w), c=c)
    out = {}
    got = zext.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks), request,
        tile_size=tile, column_batch=column_batch,
        spool_root=str(tmp_path), cell_id="c", attempt_id=attempt,
        fixed=None, spool_out=out,
        provider_factory=lambda: InMemoryCanonicalProvider(frames, masks),
        preparation_budget={
            "producer_grant_bytes": grant,
            "producers": int(producers),
            "effective_mode": "auto",
            "reason": "test_forced",
        },
    )
    return got, out, request


def _run_legacy_cell(tmp_path, frames, masks, *, tile, column_batch=8,
                     request=None, attempt="legacy"):
    n, h, w = len(frames), frames[0].shape[0], frames[0].shape[1]
    request = request or _request(n)
    return zext.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks), request,
        tile_size=tile, column_batch=column_batch,
        spool_root=str(tmp_path), cell_id="c", attempt_id=attempt,
        fixed=None,
    )


def _assert_same(ref, got):
    assert np.array_equal(ref.science, got.science, equal_nan=True)
    assert np.array_equal(ref.estimator_weight_sum, got.estimator_weight_sum,
                          equal_nan=True)
    assert np.array_equal(ref.valid_mask, got.valid_mask)
    assert np.array_equal(ref.surviving_sample_count, got.surviving_sample_count)
    assert np.array_equal(ref.support_w1, got.support_w1, equal_nan=True)
    assert np.array_equal(ref.support_w2, got.support_w2, equal_nan=True)
    assert np.array_equal(ref.n_eff_support, got.n_eff_support, equal_nan=True)
    assert ref.iterations_used == got.iterations_used
    assert ref.initial_sample_count == got.initial_sample_count
    assert ref.rejected_sample_count == got.rejected_sample_count
    assert ref.low_n_cell_count == got.low_n_cell_count
    assert ref.degenerate_cell_count == got.degenerate_cell_count


# ---------------------------------------------------------------------------
# 1. Exact pipeline <-> legacy spool/reduction, many windows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile,column_batch,mono", [
    (12, 8, False),
    (12, 1024, False),
    (12, 5, True),
    (14, 64, True),
])
def test_pipeline_science_matches_legacy_rgb_mono(tmp_path, tile, column_batch, mono):
    frames, masks = _inmem_corpus(mono=mono)
    req = _request(len(frames), mono=mono)
    # Non-trivial linear-fit coefficients are produced by phase 1 itself.
    ref = _run_legacy_cell(tmp_path, frames, masks, tile=tile,
                           column_batch=column_batch, request=req)
    got, out, _req = _run_pipeline_cell(
        tmp_path, frames, masks, tile=tile, column_batch=column_batch,
        request=req, attempt="pipe1")
    _assert_same(ref, got)
    assert np.any(~np.isfinite(frames[1][3:9, 4:10]))
    assert not np.isfinite(frames[3]).any()  # inactive frame exercised
    prep = out["preparation_pipeline"]
    assert prep["tiles"] >= 12          # >= 12 real synthetic windows
    assert prep["written"] == prep["tiles"] * len(frames)
    assert prep["producers"] == 2
    assert prep["quarantined"] is False


def test_pipeline_matches_canonical_streaming_reference(tmp_path):
    frames, masks = _inmem_corpus(n=5, h=40, w=36)
    req = _request(len(frames))
    ref = run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=18,
    )
    got, _out, _req = _run_pipeline_cell(
        tmp_path, frames, masks, tile=18, request=req)
    _assert_same(ref, got)


# ---------------------------------------------------------------------------
# 2. Real FITS sources: CFA phases, mono, unknown-CFA full fallback
# ---------------------------------------------------------------------------

def _wcs(shape):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [(shape[1] - 1) / 2 + 0.5, (shape[0] - 1) / 2 + 0.5]
    w.wcs.cd = np.array([[-0.001, 0.00002], [0.00002, 0.001]])
    w.array_shape = shape
    return w


def _fits_corpus(tmp_path, *, bayer=None, mono=False, n=3, shape=(64, 68), seed=5):
    rng = np.random.default_rng(seed)
    descs = []
    for i in range(n):
        w = _wcs(shape)
        w.wcs.crval = [10.0 + i * 0.00033, 30.0 - i * 0.00021]
        header = w.to_header(relax=True)
        path = Path(tmp_path) / f"s{i}.fit"
        if bayer:
            raw = rng.integers(100, 4000, shape).astype(np.uint16)
            header["BAYERPAT"] = bayer
            fits.PrimaryHDU(raw, header=header).writeto(path)
        else:
            arr = (100.0 + 2.0 * i + rng.normal(0, 2, shape)).astype(np.float32)
            if mono:
                arr[0:3, 0:4] = np.nan
            else:
                arr = np.stack([arr, arr * 1.01, arr * 0.99], axis=-1)
                arr[0:3, 0:4, :] = np.nan
            fits.PrimaryHDU(arr, header=header).writeto(path)
        descs.append(zg.FrameDescriptor(
            frame_id=zg.FrameId(path.name), source_path=str(path),
            shape_hw=shape, wcs_header=header.tostring(),
            header_sha256=f"h{i}", instrument="",
        ))
    canvas = zg.build_canvas(descs)
    layout = zg.build_layout(canvas, 1, 1)
    row, col, core = next(iter(layout.iter_cells(canvas)))
    cell = zg.ZeGridCell(
        cell_id=zg.cell_id(row, col), canvas_id=canvas.canvas_id,
        layout_id=layout.layout_id, row=row, col=col, core=core)
    patch = zg.build_patch(canvas, cell, halo_px=4)
    mem = zg.compute_membership(descs, canvas, cell, patch)
    planned = zz._plan_cell_source_frames(descs, canvas, patch, mem)
    ctx = zz._CellSourceContext(
        planned=tuple((f, plan.source_bounds) for f, plan in planned),
        patch_wcs_header=patch.patch_wcs_header,
        patch_hw=tuple(patch.patch_shape_hw))
    return ctx, patch, planned


def _source_cell(tmp_path, ctx, patch, *, tile, pipeline: bool, attempt="src"):
    cfg = MiniTileScienceConfig(
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean", backend="cpu",
        taper="footprint", taper_px=8.0, taper_floor=0.0,
        reference_index=None, equalize_rgb=False,
        sigma_low=3.0, sigma_high=3.0,
        winsor_limit_low=0.05, winsor_limit_high=0.05)
    n = len(ctx.planned)
    request = zstream.build_streaming_request(cfg, n)
    base = ctx.open_provider()
    try:
        h, w = patch.patch_shape_hw
        out = {}
        kwargs = {}
        if pipeline:
            from zemosaic.core.zegrid import prepare_pipeline as _pp
            grant = _pp.preparation_grant_components(
                patch_hw=(h, w),
                tile_hw=(tile + 8, tile + 8),
                source_hw=tuple(max(int(f.shape_hw[k]) for f, _b in ctx.planned)
                                for k in (0, 1)),
                channels=int(base.channels),
            )["peak_bytes"]
            stats_list = []
            registry = _pp.CfaMinMaxRegistry()
            kwargs = {
                "provider_factory": ctx.pipeline_provider_factory(
                    cfa_registry=registry, provider_stats=stats_list),
                "preparation_budget": {
                    "producer_grant_bytes": grant, "producers": 2,
                    "effective_mode": "auto", "reason": "test",
                },
            }
        result = zext.run_cell_streaming_external(
            base, request, tile_size=tile, column_batch=16,
            spool_root=str(tmp_path), cell_id="src", attempt_id=attempt,
            fixed=None, spool_out=out, **kwargs)
        if pipeline:
            out["provider_stats"] = stats_list
            out["cfa_stats"] = registry.stats()
        return result, out
    finally:
        base.close()


@pytest.mark.parametrize("kind", ["GRBG", "RGGB", "GBRG", "BGGR", "mono", "unknown"])
def test_pipeline_real_sources_bit_exact(tmp_path, kind):
    mono = kind == "mono"
    ctx, patch, _planned = _fits_corpus(
        tmp_path,
        bayer=(kind if kind not in ("mono", "unknown") else None),
        mono=mono,
    )
    if kind == "unknown":
        # Unknown CFA pattern forces the whole-frame decode fallback.
        for frame, _b in ctx.planned:
            path = Path(frame.source_path)
            with fits.open(path, mode="update") as hdul:
                hdul[0].header["BAYERPAT"] = "XYZ1"
                hdul.flush()
    ref, _o1 = _source_cell(tmp_path, ctx, patch, tile=18, pipeline=False,
                            attempt="legacy")
    got, out2 = _source_cell(tmp_path, ctx, patch, tile=18, pipeline=True,
                             attempt="pipe")
    _assert_same(ref, got)
    if kind == "unknown":
        stats = out2["cfa_stats"]
        # A 3-D FITS HDU carrying BAYERPAT cannot take the bounded CFA section
        # path at all: every window falls back to the whole-frame decode and no
        # CFA (min,max) is published.
        assert stats["entries"] == 0


def test_cfa_prescan_is_cell_scoped_not_per_window(tmp_path):
    ctx, patch, planned = _fits_corpus(tmp_path, bayer="GRBG", n=3)
    got, out = _source_cell(tmp_path, ctx, patch, tile=16, pipeline=True)
    agg = {}
    for entry in out["provider_stats"]:
        values = entry.window_telemetry() if hasattr(entry, "window_telemetry") else entry
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                agg[key] = agg.get(key, 0) + value
    # Three sources, one paid pre-scan each, shared across all 2*W providers.
    assert out["cfa_stats"]["entries"] == 3
    assert out["cfa_stats"]["computed"] == 3
    assert out["cfa_stats"]["hits"] >= 1
    assert agg["prescans"] == 3
    assert agg["prescans_shared"] > 0
    tiles = int(out["preparation_pipeline"]["tiles"])
    full_cell_per_window = len(planned) * tiles * 64 * 68
    assert agg["source_pixels_decoded"] < full_cell_per_window


# ---------------------------------------------------------------------------
# 3. Out-of-order completion + real barriers (no scheduler assumptions)
# ---------------------------------------------------------------------------

class _OrderProvider(InMemoryCanonicalProvider):
    def __init__(self, frames, masks, ctrl):
        super().__init__(frames, masks)
        self.ctrl = ctrl
        self.closed = False

    def close(self):
        self.closed = True

    def get_tile(self, i, y0, y1, x0, x1):
        if self.ctrl.get("barrier") is not None:
            self.ctrl["barrier"].wait(timeout=5.0)
        if i == 1:
            self.ctrl["one_done"].set()
        if i == 0:
            assert self.ctrl["one_done"].wait(timeout=5.0)
        return super().get_tile(i, y0, y1, x0, x1)


class _TrackedProvider(InMemoryCanonicalProvider):
    def __init__(self, frames, masks):
        super().__init__(frames, masks)
        self.closed = False

    def close(self):
        self.closed = True


def test_out_of_order_completion_never_permutes_index(tmp_path):
    frames, masks = _inmem_corpus(n=4, h=32, w=30)
    req = _request(len(frames))
    p1 = _phase1(frames, masks, req)
    spool = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=4, height=32, width=30, channels=3,
        halo_px=3, tile_size=16, cell_id="c", attempt_id="order"))
    ctrl = {"one_done": threading.Event(), "barrier": None}
    grant = _grant(32, 30, tile=16, halo=3)
    written = []
    handle, stats = zpp.write_tile_two_producers(
        spool, lambda: _OrderProvider(frames, masks, ctrl), p1,
        0, 16, 0, 16,
        grant_bytes=grant, producers=2,
        taper_kind="footprint", taper_px=8.0, taper_floor=0.0, halo_px=3,
    )
    # The writer callback order is not asserted directly here; the PAYLOAD is.
    legacy_spool = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=4, height=32, width=30, channels=3,
        halo_px=3, tile_size=16, cell_id="c", attempt_id="legacy"))
    legacy = legacy_spool.write_tile(
        InMemoryCanonicalProvider(frames, masks), p1, 0, 16, 0, 16,
        taper_kind="footprint", taper_px=8.0, taper_floor=0.0, halo_px=3)
    a = np.load(handle.images_path)
    b = np.load(legacy.images_path)
    assert np.array_equal(a, b, equal_nan=True)
    assert np.array_equal(np.load(handle.weights_path),
                          np.load(legacy.weights_path))
    assert stats["written"] == 4


def test_real_barrier_two_producers_both_active():
    frames, masks = _inmem_corpus(n=2, h=24, w=22)
    req = _request(2)
    p1 = _phase1(frames, masks, req)
    ctrl = {"one_done": threading.Event(), "barrier": threading.Barrier(2)}
    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _OrderProvider(frames, masks, ctrl),
        p1=p1, geometry=(0, 24, 0, 22),
        taper_kind="footprint", taper_px=8.0, taper_floor=0.0,
        grant_bytes=_grant(24, 22, tile=24, halo=0), producers=2,
        drain_timeout_s=2.0,
    )
    seen = []
    stats = pipeline.run(lambda i, f, w: seen.append(i))
    assert sorted(seen) == [0, 1]
    assert stats["provider_opens"] == 2
    assert stats["provider_closes"] == 2


# ---------------------------------------------------------------------------
# 4. Writer is the calling thread; payloads are released
# ---------------------------------------------------------------------------

def test_writer_is_calling_thread_and_payloads_released():
    frames, masks = _inmem_corpus(n=4, h=20, w=18)
    req = _request(4)
    p1 = _phase1(frames, masks, req)
    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _TrackedProvider(frames, masks),
        p1=p1, geometry=(0, 20, 0, 18),
        taper_kind="none", taper_px=8.0, taper_floor=0.0,
        grant_bytes=_grant(20, 18, tile=20, halo=0), producers=2,
        drain_timeout_s=2.0,
    )
    caller = threading.get_ident()
    writer_threads = []
    refs = []

    def _write(i, frame, weights):
        writer_threads.append(threading.get_ident())
        if frame is not None:
            refs.append(weakref.ref(frame))
            refs.append(weakref.ref(weights))

    stats = pipeline.run(_write)
    assert writer_threads and all(t == caller for t in writer_threads)
    assert stats["written"] == 4
    gc.collect()
    assert all(ref() is None for ref in refs)


# ---------------------------------------------------------------------------
# 5. Budget honesty
# ---------------------------------------------------------------------------

def test_credit_refuses_a_grant_larger_than_capacity():
    ledger = zpp.ByteCreditLedger(100)
    with pytest.raises(zpp.PreparationBudgetError):
        ledger.acquire(101)
    ledger.acquire(100)
    ledger.release(100)
    assert ledger.stats()["peak_used_bytes"] == 100


def test_decision_falls_back_when_two_do_not_fit(monkeypatch):
    grant = 100 * MIB
    out = zpp.decide_preparation_pipeline(
        requested_mode="auto", available_bytes=700 * MIB,
        producer_grant_bytes=grant, existing_commitment_bytes=0,
        producers=2, host_reserve_bytes=512 * MIB,
    )
    assert out["effective_mode"] == "legacy"
    assert out["reason"] == "insufficient_host_budget_for_two_producers"
    assert out["single_admissible"] is True
    out2 = zpp.decide_preparation_pipeline(
        requested_mode="auto", available_bytes=300 * MIB,
        producer_grant_bytes=grant, existing_commitment_bytes=0,
        producers=2, host_reserve_bytes=512 * MIB,
    )
    assert out2["reason"] == "insufficient_host_budget_even_single"
    out3 = zpp.decide_preparation_pipeline(
        requested_mode="auto", available_bytes=None,
        producer_grant_bytes=grant, existing_commitment_bytes=0,
    )
    assert out3["reason"] == "host_available_unknown"
    out4 = zpp.decide_preparation_pipeline(
        requested_mode="auto", available_bytes=10 * 1024 ** 3,
        producer_grant_bytes=grant, existing_commitment_bytes=None,
    )
    assert out4["reason"] == "committed_budget_unknown"
    out5 = zpp.decide_preparation_pipeline(
        requested_mode="auto", available_bytes=10 * 1024 ** 3,
        producer_grant_bytes=grant, existing_commitment_bytes=0,
        unqualified_terms=["source_roi_unknown"],
    )
    assert out5["reason"] == "unqualified_budget_term:source_roi_unknown"


def test_factory_without_budget_is_refused(tmp_path):
    frames, masks = _inmem_corpus(n=2, h=16, w=14)
    req = _request(2)
    with pytest.raises(Exception) as exc:
        zext.run_cell_streaming_external(
            InMemoryCanonicalProvider(frames, masks), req, tile_size=8,
            column_batch=4, spool_root=str(tmp_path),
            provider_factory=lambda: InMemoryCanonicalProvider(frames, masks),
        )
    assert "preparation_budget" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. Producer / writer errors, drain and quarantine
# ---------------------------------------------------------------------------

class _FailingProvider(InMemoryCanonicalProvider):
    def __init__(self, frames, masks, fail_at):
        super().__init__(frames, masks)
        self.fail_at = int(fail_at)
        self.closed = False

    def get_tile(self, i, y0, y1, x0, x1):
        if i == self.fail_at:
            raise RuntimeError(f"injected producer failure at {i}")
        return super().get_tile(i, y0, y1, x0, x1)

    def close(self):
        self.closed = True


def _pipeline_for(factory, n, h=24, w=20, *, grant=None, drain=2.0):
    frames, masks = _inmem_corpus(n=n, h=h, w=w)
    req = _request(n)
    p1 = _phase1(frames, masks, req)
    return zpp.PreparationPipeline(
        provider_factory=factory, p1=p1, geometry=(0, h, 0, w),
        taper_kind="none", taper_px=8.0, taper_floor=0.0,
        grant_bytes=grant or _grant(h, w, tile=h, halo=0), producers=2,
        drain_timeout_s=drain,
    ), p1


def test_producer_error_preserves_original_and_releases():
    frames, masks = _inmem_corpus(n=6, h=24, w=20)
    providers = []

    def factory():
        p = _FailingProvider(frames, masks, fail_at=2)
        providers.append(p)
        return p

    pipeline, _p1 = _pipeline_for(factory, 6)
    with pytest.raises(RuntimeError, match="injected producer failure at 2"):
        pipeline.run(lambda i, f, w: None)
    assert pipeline.ledger.stats()["used_bytes"] == 0
    assert all(getattr(p, "closed", False) for p in providers)


def test_writer_error_preserves_original_and_releases():
    frames, masks = _inmem_corpus(n=6, h=24, w=20)
    providers = []

    def factory():
        p = _TrackedProvider(frames, masks)
        providers.append(p)
        return p

    pipeline, _p1 = _pipeline_for(factory, 6)

    def _write(i, f, w):
        if i == 2:
            raise KeyError("injected writer failure")

    with pytest.raises(KeyError, match="injected writer failure"):
        pipeline.run(_write)
    assert pipeline.ledger.stats()["used_bytes"] == 0
    assert all(getattr(p, "closed", False) for p in providers)


class _BlockingProvider(InMemoryCanonicalProvider):
    def __init__(self, frames, masks, block_at, release):
        super().__init__(frames, masks)
        self.block_at = int(block_at)
        self.release = release
        self.entered = threading.Event()
        self.closed = False

    def get_tile(self, i, y0, y1, x0, x1):
        if i == self.block_at:
            self.entered.set()
            assert self.release.wait(timeout=10.0)
        return super().get_tile(i, y0, y1, x0, x1)

    def close(self):
        self.closed = True
        super().close()


def test_incomplete_drain_quarantines_provider_set():
    frames, masks = _inmem_corpus(n=6, h=24, w=20)
    release = threading.Event()
    providers = []

    def factory():
        p = _BlockingProvider(frames, masks, block_at=1, release=release)
        providers.append(p)
        return p

    pipeline, _p1 = _pipeline_for(factory, 6, drain=0.05)

    def _write(i, f, w):
        raise KeyError("writer dies while a producer is blocked")

    with pytest.raises(zpp.PreparationQuarantineError) as exc:
        pipeline.run(_write)
    assert isinstance(exc.value.original, KeyError)
    assert exc.value.alive_producers >= 1
    assert pipeline.quarantined is True
    # The blocked provider was NOT closed as if the drain had succeeded.
    assert any(not p.closed for p in providers)
    release.set()
    time.sleep(0.1)


# ---------------------------------------------------------------------------
# 7. Cell-scope CFA registry single-flight
# ---------------------------------------------------------------------------

def test_product_selector_defaults_auto_only_when_admissible(tmp_path, monkeypatch):
    ctx, patch, _planned = _fits_corpus(tmp_path, n=1)
    h, w = patch.patch_shape_hw
    base = ctx.open_provider()
    tiles = zz.zext._tile_geometry(h, w, 24, 8)[0]
    try:
        monkeypatch.setenv("ZEMOSAIC_M33_PREP_PIPELINE", "legacy")
        fac, budget, info, gate = zz._resolve_m33_preparation_pipeline(
            source_context=ctx, provider=base, tiles=tiles,
            max_th=24 + 8, max_tw=24 + 8, patch=patch, column_batch=16)
        assert fac is None and budget is None
        assert info["reason"] == "requested_legacy_control"

        monkeypatch.setenv("ZEMOSAIC_M33_PREP_PIPELINE", "auto")
        monkeypatch.setattr(
            zz.zrec, "default_capacity_probe",
            lambda: (_ for _ in ()).throw(RuntimeError("no probe")))
        fac, budget, info, gate = zz._resolve_m33_preparation_pipeline(
            source_context=ctx, provider=base, tiles=tiles,
            max_th=24 + 8, max_tw=24 + 8, patch=patch, column_batch=16)
        assert fac is None and info["reason"] == "host_available_unknown"

        monkeypatch.setattr(
            zz.zrec, "default_capacity_probe",
            lambda: {"raw_available_bytes": 64 * 1024 ** 3,
                     "observed_monotonic": time.monotonic(),
                     "source": "test"})
        fac, budget, info, gate = zz._resolve_m33_preparation_pipeline(
            source_context=ctx, provider=base, tiles=tiles,
            max_th=24 + 8, max_tw=24 + 8, patch=patch, column_batch=16)
        assert fac is not None
        assert budget["producers"] == 2
        assert info["effective_mode"] == "auto"
        assert info["decision"]["reason"] == "constructive_budget_admissible"
    finally:
        base.close()


# ---------------------------------------------------------------------------
# 8. Rework-1: stop, partial publish, mapping lifecycle, reuse
# ---------------------------------------------------------------------------

class _CancelProvider(InMemoryCanonicalProvider):
    def __init__(self, frames, masks, at, event):
        super().__init__(frames, masks)
        self.at = int(at)
        self.event = event

    def get_tile(self, i, y0, y1, x0, x1):
        if i == self.at:
            self.event.set()
        return super().get_tile(i, y0, y1, x0, x1)


def test_stop_before_run_refuses_before_factory():
    frames, masks = _inmem_corpus(n=4, h=24, w=20)
    req = _request(4)
    p1 = _phase1(frames, masks, req)
    calls = []

    def factory():
        calls.append(1)
        return _TrackedProvider(frames, masks)

    pipeline = zpp.PreparationPipeline(
        provider_factory=factory, p1=p1, geometry=(0, 24, 0, 20),
        taper_kind="none", taper_px=8.0, taper_floor=0.0,
        grant_bytes=_grant(24, 20, tile=24, halo=0), producers=2,
    )
    pipeline.request_stop("user_stop")
    with pytest.raises(zpp.PreparationAborted, match="user_stop"):
        pipeline.run(lambda i, f, w: None)
    assert calls == []
    assert pipeline.ledger.stats()["reservations"] == 0


def test_integrated_stop_before_and_during_never_publishes(tmp_path):
    frames, masks = _inmem_corpus(n=6, h=20, w=18)
    req = _request(6)
    p1 = _phase1(frames, masks, req)
    grant = _grant(20, 18, tile=20, halo=0)

    # Before start.
    spool = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=6, height=20, width=18, channels=3,
        halo_px=0, tile_size=20, cell_id="c", attempt_id="stop0"))
    ev = threading.Event()
    ev.set()
    with pytest.raises(zpp.PreparationAborted):
        zpp.write_tile_two_producers(
            spool, lambda: _TrackedProvider(frames, masks), p1,
            0, 20, 0, 18, grant_bytes=grant, producers=2,
            taper_kind="none", taper_px=8.0, taper_floor=0.0, halo_px=0,
            cancel_event=ev,
        )
    assert spool._tiles == []
    assert not (spool.attempt_dir / "tile_y000000_x000000").exists()

    # Mid-run: one producer cancels after preparing frame 2.
    spool2 = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=6, height=20, width=18, channels=3,
        halo_px=0, tile_size=20, cell_id="c", attempt_id="stop1"))
    ev2 = threading.Event()
    with pytest.raises(zpp.PreparationAborted):
        zpp.write_tile_two_producers(
            spool2, lambda: _CancelProvider(frames, masks, 2, ev2), p1,
            0, 20, 0, 18, grant_bytes=grant, producers=2,
            taper_kind="none", taper_px=8.0, taper_floor=0.0, halo_px=0,
            cancel_event=ev2, drain_timeout_s=2.0,
        )
    assert spool2._tiles == []
    assert not (spool2.attempt_dir / "tile_y000000_x000000").exists()


def test_publish_failure_removes_partial_tile_and_closes_mapping(tmp_path):
    spool = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=2, height=16, width=14, channels=3,
        halo_px=0, tile_size=16, cell_id="c", attempt_id="pubfail"))
    session = spool.open_tile_writer(0, 16, 0, 14, halo_px=0,
                                     n_frames=2, channels=3)
    original = session.images
    mm = original._mmap

    class _BadFlush:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def flush(self):
            raise OSError(5, "injected flush failure")

    session.images = _BadFlush(original)
    with pytest.raises(OSError, match="injected flush failure"):
        session.publish()
    assert session.tile_dir.exists()  # not silently published
    session.abort()
    assert not session.tile_dir.exists()
    assert mm.closed
    session.abort()  # idempotent
    spool.cleanup()


def test_legacy_second_allocation_closes_first_mapping(tmp_path, monkeypatch):
    frames, masks = _inmem_corpus(n=2, h=24, w=20)
    req = _request(2)
    p1 = _phase1(frames, masks, req)
    spool = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=2, height=24, width=20, channels=3,
        halo_px=0, tile_size=24, cell_id="c", attempt_id="alloc"))
    real = np.lib.format.open_memmap
    opened = []

    def flaky(path, *a, **k):
        if "weights" in str(path):
            raise OSError(28, "No space left on device")
        arr = real(path, *a, **k)
        opened.append(arr)
        return arr

    monkeypatch.setattr(np.lib.format, "open_memmap", flaky)
    with pytest.raises(zext.ExternalSpoolWriteError):
        spool.write_tile(
            InMemoryCanonicalProvider(frames, masks), p1, 0, 24, 0, 20,
            taper_kind="none", taper_px=8.0, taper_floor=0.0, halo_px=0)
    assert opened and opened[0]._mmap.closed
    assert not (spool.attempt_dir / "tile_y000000_x000000").exists()
    spool.cleanup()


def test_pipeline_error_closes_tile_mappings(tmp_path, monkeypatch):
    frames, masks = _inmem_corpus(n=4, h=24, w=20)
    req = _request(4)
    p1 = _phase1(frames, masks, req)
    spool = zext.ExternalSpool(zext.ExternalSpoolSpec(
        root=str(tmp_path), n_frames=4, height=24, width=20, channels=3,
        halo_px=0, tile_size=24, cell_id="c", attempt_id="pipeerr"))
    real = np.lib.format.open_memmap
    opened = []

    def record(path, *a, **k):
        arr = real(path, *a, **k)
        opened.append(arr)
        return arr

    monkeypatch.setattr(np.lib.format, "open_memmap", record)
    with pytest.raises(RuntimeError, match="injected producer failure at 2"):
        zpp.write_tile_two_producers(
            spool, lambda: _FailingProvider(frames, masks, 2), p1,
            0, 24, 0, 20, grant_bytes=_grant(24, 20, tile=24, halo=0),
            producers=2, taper_kind="none", taper_px=8.0,
            taper_floor=0.0, halo_px=0, drain_timeout_s=1.0)
    assert opened and all(a._mmap.closed for a in opened)
    assert not (spool.attempt_dir / "tile_y000000_x000000").exists()
    spool.cleanup()


def test_close_failure_fails_the_pipeline():
    frames, masks = _inmem_corpus(n=3, h=24, w=20)
    req = _request(3)
    p1 = _phase1(frames, masks, req)

    class _CloseBoom(_TrackedProvider):
        def close(self):
            self.closed = True
            raise RuntimeError("injected close failure")

    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _CloseBoom(frames, masks), p1=p1,
        geometry=(0, 24, 0, 20), taper_kind="none", taper_px=8.0,
        taper_floor=0.0, grant_bytes=_grant(24, 20, tile=24, halo=0),
        producers=1, drain_timeout_s=1.0,
    )
    with pytest.raises(RuntimeError, match="injected close failure"):
        pipeline.run(lambda i, f, w: None)
    assert pipeline.ledger.stats()["used_bytes"] == 0


def test_pipeline_is_single_use_and_quarantine_blocks_reuse():
    frames, masks = _inmem_corpus(n=2, h=24, w=20)
    req = _request(2)
    p1 = _phase1(frames, masks, req)
    calls = []

    def factory():
        calls.append(1)
        return _TrackedProvider(frames, masks)

    pipeline = zpp.PreparationPipeline(
        provider_factory=factory, p1=p1, geometry=(0, 24, 0, 20),
        taper_kind="none", taper_px=8.0, taper_floor=0.0,
        grant_bytes=_grant(24, 20, tile=24, halo=0), producers=1,
    )
    pipeline.run(lambda i, f, w: None)
    n_calls = len(calls)
    with pytest.raises(zpp.PreparationAborted, match="already consumed"):
        pipeline.run(lambda i, f, w: None)
    assert len(calls) == n_calls

    # A quarantined object refuses reuse before opening a provider.
    q = zpp.PreparationPipeline(
        provider_factory=factory, p1=p1, geometry=(0, 24, 0, 20),
        taper_kind="none", taper_px=8.0, taper_floor=0.0,
        grant_bytes=_grant(24, 20, tile=24, halo=0), producers=1,
    )
    q._quarantined = True
    q._quarantine_original = RuntimeError("original")
    with pytest.raises(zpp.PreparationQuarantineError):
        q.run(lambda i, f, w: None)
    assert len(calls) == n_calls


def test_payload_released_before_next_write():
    frames, masks = _inmem_corpus(n=6, h=24, w=20)
    req = _request(6)
    p1 = _phase1(frames, masks, req)
    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _TrackedProvider(frames, masks), p1=p1,
        geometry=(0, 24, 0, 20), taper_kind="none", taper_px=8.0,
        taper_floor=0.0, grant_bytes=_grant(24, 20, tile=24, halo=0),
        producers=2,
    )
    refs = []
    violations = []

    def _write(i, frame, weights):
        if frame is None:
            return
        if refs and refs[-1]() is not None:
            violations.append(i)
        refs.append(weakref.ref(frame))

    pipeline.run(_write)
    assert violations == []
    gc.collect()
    assert all(r() is None for r in refs)


def test_progress_heartbeat_is_bounded_and_nonfatal():
    frames, masks = _inmem_corpus(n=32, h=24, w=20)
    req = _request(32)
    p1 = _phase1(frames, masks, req)
    events = []

    def cb(fields):
        events.append(dict(fields))
        if len(events) == 1:
            raise RuntimeError("telemetry must be swallowed")

    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _TrackedProvider(frames, masks), p1=p1,
        geometry=(0, 24, 0, 20), taper_kind="none", taper_px=8.0,
        taper_floor=0.0, grant_bytes=_grant(24, 20, tile=24, halo=0),
        producers=2, progress_callback=cb, progress_every=5,
    )
    stats = pipeline.run(lambda i, f, w: None)
    assert stats["written"] == 32
    assert 2 <= len(events) <= 10
    assert events[-1]["frame_current"] == 32
    assert all(e["frame_total"] == 32 for e in events)
    assert all(e["producers_effective"] == 2 for e in events)


class _SignalProvider(_TrackedProvider):
    def __init__(self, frames, masks, entered):
        super().__init__(frames, masks)
        self.entered = entered

    def get_tile(self, i, y0, y1, x0, x1):
        self.entered.set()
        return super().get_tile(i, y0, y1, x0, x1)


def test_start_failure_drains_queue_and_returns_credits(monkeypatch):
    frames, masks = _inmem_corpus(n=6, h=24, w=20)
    req = _request(6)
    p1 = _phase1(frames, masks, req)
    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _TrackedProvider(frames, masks), p1=p1,
        geometry=(0, 24, 0, 20), taper_kind="none", taper_px=8.0,
        taper_floor=0.0, grant_bytes=_grant(24, 20, tile=24, halo=0),
        producers=2, drain_timeout_s=2.0,
    )
    holder = {"pipeline": pipeline, "calls": 0}
    real_thread = threading.Thread

    class _FakeThread(real_thread):
        def start(self):
            holder["calls"] += 1
            if holder["calls"] == 2:
                deadline = time.perf_counter() + 5.0
                while (holder["pipeline"]._queue.qsize() < 1
                       and time.perf_counter() < deadline):
                    time.sleep(0.001)
                raise RuntimeError("injected_second_start_failure")
            return super().start()

    monkeypatch.setattr(zpp.threading, "Thread", _FakeThread)
    with pytest.raises(RuntimeError, match="injected_second_start_failure"):
        pipeline.run(lambda i, f, w: None)
    stats = pipeline.ledger.stats()
    assert stats["used_bytes"] == 0
    assert pipeline._queue.qsize() == 0
    assert pipeline._stats["ledger"]["used_bytes"] == 0
    assert pipeline._stats["drained_after_start_failure"] >= 1
    assert pipeline._stats["drain_incomplete"] is False


def test_start_failure_with_live_producer_quarantines_and_keeps_costs(monkeypatch):
    frames, masks = _inmem_corpus(n=4, h=24, w=20)
    req = _request(4)
    p1 = _phase1(frames, masks, req)
    entered = threading.Event()
    release = threading.Event()

    class _Blocked(_TrackedProvider):
        def get_tile(self, i, y0, y1, x0, x1):
            entered.set()
            assert release.wait(timeout=10.0)
            return super().get_tile(i, y0, y1, x0, x1)

    pipeline = zpp.PreparationPipeline(
        provider_factory=lambda: _Blocked(frames, masks), p1=p1,
        geometry=(0, 24, 0, 20), taper_kind="none", taper_px=8.0,
        taper_floor=0.0, grant_bytes=_grant(24, 20, tile=24, halo=0),
        producers=2, drain_timeout_s=0.05,
    )
    holder = {"pipeline": pipeline, "calls": 0}
    real_thread = threading.Thread

    class _FakeThread(real_thread):
        def start(self):
            holder["calls"] += 1
            if holder["calls"] == 2:
                assert entered.wait(timeout=5.0)
                raise RuntimeError("injected_second_start_failure")
            return super().start()

    monkeypatch.setattr(zpp.threading, "Thread", _FakeThread)
    with pytest.raises(zpp.PreparationQuarantineError) as exc:
        pipeline.run(lambda i, f, w: None)
    assert exc.value.alive_producers >= 1
    # A live producer's grant is NOT released by the failed start path.
    assert pipeline.ledger.stats()["used_bytes"] > 0
    release.set()
    time.sleep(0.15)


def test_grant_margin_and_domains_are_honest():
    unknown = zpp.preparation_grant_components(
        patch_hw=(20, 18), tile_hw=(20, 18), source_hw=(20, 18), channels=3,
        n_frames=5,
    )
    assert "source_roi_unknown" in unknown["unqualified_terms"]
    known = zpp.preparation_grant_components(
        patch_hw=(20, 18), tile_hw=(20, 18), source_hw=(20, 18), channels=3,
        n_frames=5, source_roi_hw=(20, 18),
    )
    assert known["unqualified_terms"] == []
    assert known["empirical_terms"]
    assert known["non_owned_terms"]
    assert known["peak_without_residual_over_subtotal"] == pytest.approx(2.25)
    big = zpp.preparation_grant_components(
        patch_hw=(20, 18), tile_hw=(20, 18), source_hw=(20, 18), channels=3,
        n_frames=100000, source_roi_hw=(20, 18),
    )
    assert (big["components"]["metadata_cache_bytes"]
            > known["components"]["metadata_cache_bytes"])


def test_tile_boundary_gate_falls_back_sticky(tmp_path):
    frames, masks = _inmem_corpus(n=6, h=40, w=36)
    req = _request(6)
    ref = _run_legacy_cell(tmp_path, frames, masks, tile=12, request=req)
    calls = []
    gates = []

    def factory():
        calls.append(1)
        return _TrackedProvider(frames, masks)

    def gate():
        gates.append(1)
        return {"effective_mode": "legacy", "reason": "test_budget_drop"}

    out = {}
    got = zext.run_cell_streaming_external(
        InMemoryCanonicalProvider(frames, masks), req,
        tile_size=12, column_batch=8, spool_root=str(tmp_path),
        cell_id="c", attempt_id="gate", fixed=None, spool_out=out,
        provider_factory=factory,
        preparation_budget={
            "producer_grant_bytes": _grant(40, 36, tile=12, halo=4),
            "producers": 2, "effective_mode": "auto", "reason": "test",
        },
        preparation_gate=gate,
    )
    _assert_same(ref, got)
    assert len(calls) == 2          # only the first tile opened a pool
    assert len(gates) == 1          # sticky: gate not retried
    prep = out["preparation_pipeline"]
    assert prep["tiles_pipeline"] == 1
    assert prep["tiles_legacy"] > 0
    assert prep["fallback_at_tile"] == 2
    assert prep["sticky_fallback_reason"] == "test_budget_drop"
    assert prep["effective_mode_last"] == "legacy"


def test_real_caller_auto_legacy_and_unknown_committed(tmp_path, monkeypatch):
    ctx, patch, _planned = _fits_corpus(tmp_path, n=2)
    cfg = MiniTileScienceConfig(
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean", backend="cpu",
        taper="footprint", taper_px=8.0, taper_floor=0.0,
        reference_index=None, equalize_rgb=False,
        sigma_low=3.0, sigma_high=3.0,
        winsor_limit_low=0.05, winsor_limit_high=0.05,
    )
    monkeypatch.setattr(
        zz.zrec, "default_capacity_probe",
        lambda: {"raw_available_bytes": 64 * 1024 ** 3,
                 "observed_monotonic": time.monotonic(),
                 "source": "test"})

    monkeypatch.setenv("ZEMOSAIC_M33_PREP_PIPELINE", "auto")
    for _d in ("spool_a", "spool_l", "spool_fb"):
        (tmp_path / _d).mkdir(parents=True, exist_ok=True)
    info_auto = {}
    _mt_a, sres_a = zz._run_cell_external(
        tmp_path / "cache", patch, cfg, None, tile_size=16, column_batch=16,
        spool_root=(tmp_path / "spool_a"), fixed=None,
        storage_profile="bounded", storage_info=info_auto,
        cell_id="c", attempt_id="auto", source_context=ctx,
    )
    prep = info_auto["external"]["preparation_pipeline"]
    assert prep["effective_mode"] == "auto"
    assert prep["producers"] == 2
    assert prep["decision"]["qualified"] is True
    assert prep["decision"]["existing_commitment_bytes"] is not None
    runtime = prep["runtime"]
    assert runtime["tiles_legacy"] == 0
    assert runtime["tiles_pipeline"] == runtime["tiles"]
    assert runtime["gate_probes"] >= 1
    assert runtime["effective_mode_last"] == "auto"

    monkeypatch.setenv("ZEMOSAIC_M33_PREP_PIPELINE", "legacy")
    info_leg = {}
    _mt_l, sres_l = zz._run_cell_external(
        tmp_path / "cache", patch, cfg, None, tile_size=16, column_batch=16,
        spool_root=(tmp_path / "spool_l"), fixed=None,
        storage_profile="bounded", storage_info=info_leg,
        cell_id="c", attempt_id="legacy", source_context=ctx,
    )
    assert info_leg["external"]["preparation_pipeline"]["effective_mode"] == "legacy"
    assert info_leg["external"]["preparation_pipeline"]["reason"] == (
        "requested_legacy_control")
    assert np.array_equal(sres_a.result.science, sres_l.result.science,
                          equal_nan=True)
    assert np.array_equal(sres_a.result.estimator_weight_sum,
                          sres_l.result.estimator_weight_sum, equal_nan=True)

    # Unknown committed commitment => FAIL CLOSED, never assumed zero.
    monkeypatch.setenv("ZEMOSAIC_M33_PREP_PIPELINE", "auto")

    def _boom(*_a, **_k):
        raise RuntimeError("pricing unavailable")

    monkeypatch.setattr(zz.zext, "external_cell_host_components", _boom)
    info_fb = {}
    zz._run_cell_external(
        tmp_path / "cache", patch, cfg, None, tile_size=16, column_batch=16,
        spool_root=(tmp_path / "spool_fb"), fixed=None,
        storage_profile="bounded", storage_info=info_fb,
        cell_id="c", attempt_id="fb", source_context=ctx,
    )
    fb = info_fb["external"]["preparation_pipeline"]
    assert fb["effective_mode"] == "legacy"
    assert fb["reason"] == "committed_budget_unknown"


def test_write_progress_is_visible_end_to_end(tmp_path):
    path = tmp_path / "c1.jsonl"
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=1, cell_id="c1",
        min_interval_s=0.0,
    )
    assert writer.state("RUNNING") is True
    assert writer.progress(
        "write", tile_current=1, tile_total=6,
        frame_current=5, frame_total=100, frame_prepared=5,
        bytes_reserved=123456, bytes_peak=200000, producers_effective=2,
    ) is True
    assert writer.progress(
        "write", tile_current=1, tile_total=6,
        frame_current=50, frame_total=100, frame_prepared=50,
        bytes_reserved=1000, bytes_peak=200000, producers_effective=2,
    ) is True
    lines = [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]
    progress = [l for l in lines if l["kind"] == "PROGRESS"]
    assert len(progress) == 2
    assert progress[0]["frame"] == [5, 100]
    assert progress[1]["frame"] == [50, 100]
    assert progress[1]["frame_prepared"] == 50
    assert progress[1]["bytes_reserved"] == 1000
    assert progress[1]["producers_effective"] == 2

    reader = zcs.LiveTelemetryReader(tmp_path, run_id="run", generation=1)
    events = reader.poll(["c1"])
    states = zcs.LiveCellStates()
    states.reset(["c1"])
    zcs.apply_events(states, events, now=time.monotonic())
    snap = states.snapshot(now=time.monotonic())
    row = next(r for r in snap["rows"] if r["cell_id"] == "c1")
    assert row["state"] == "RUNNING"
    assert row["frame"] == [50, 100]
    assert row["frame_prepared"] == 50
    assert row["bytes_reserved"] == 1000
    assert row["producers_effective"] == 2
    assert row["observed_age_s"] is not None


def test_transition_clears_stale_write_counters():
    states = zcs.LiveCellStates()
    states.reset(["c"])
    states.update("c", "RUNNING", observed_at=0.0)
    states.note_progress(
        "c", phase="write", tile_current=1, tile_total=6,
        frame_current=100, frame_total=100, frame_prepared=100,
        bytes_reserved=1234, bytes_peak=2000, producers_effective=2,
        observed_at=1.0,
    )
    row = next(r for r in states.snapshot(now=1.0)["rows"] if r["cell_id"] == "c")
    assert row["frame"] == [100, 100] and row["producers_effective"] == 2

    # write -> reduce (same tile): no producer/frame/bytes must survive.
    states.note_progress(
        "c", phase="reduce", tile_current=1, tile_total=6,
        block_current=1, block_total=3, observed_at=2.0,
    )
    row = next(r for r in states.snapshot(now=2.0)["rows"] if r["cell_id"] == "c")
    assert row["block"] == [1, 3]
    assert "frame" not in row and "frame_prepared" not in row
    assert "bytes_reserved" not in row and "producers_effective" not in row
    line = zcs.live_cell_line(row)
    assert "frame=" not in line and "producers=" not in line

    # next tile, LEGACY path: only a CURRENT producers=1, no stale frame/bytes.
    states.note_progress(
        "c", phase="write", tile_current=2, tile_total=6,
        producers_effective=1, observed_at=3.0,
    )
    row = next(r for r in states.snapshot(now=3.0)["rows"] if r["cell_id"] == "c")
    assert row["phase"] == "write" and row["tile"] == [2, 6]
    assert row["producers_effective"] == 1
    assert "frame" not in row and "bytes_reserved" not in row

    # Partial frame-only sample keeps the current phase/tile.
    states.note_progress(
        "c", frame_current=7, frame_total=100, observed_at=4.0,
    )
    row = next(r for r in states.snapshot(now=4.0)["rows"] if r["cell_id"] == "c")
    assert row["phase"] == "write" and row["tile"] == [2, 6]
    assert row["frame"] == [7, 100]


def test_journal_transition_does_not_display_stale_counters(tmp_path):
    path = tmp_path / "c.jsonl"
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=1, cell_id="c",
        min_interval_s=-1.0,
    )
    writer.state("RUNNING")
    writer.progress(
        "write", tile_current=1, tile_total=6,
        frame_current=100, frame_total=100, frame_prepared=100,
        bytes_reserved=1234, producers_effective=2,
    )
    writer.progress(
        "reduce", tile_current=1, tile_total=6,
        block_current=1, block_total=3,
    )
    reader = zcs.LiveTelemetryReader(tmp_path, run_id="run", generation=1)
    states = zcs.LiveCellStates()
    states.reset(["c"])
    zcs.apply_events(states, reader.poll(["c"]), now=time.monotonic())
    row = next(r for r in states.snapshot(now=time.monotonic())["rows"]
               if r["cell_id"] == "c")
    assert row["phase"] == "reduce" and row["block"] == [1, 3]
    assert "frame" not in row and "producers_effective" not in row
    assert "frame=" not in zcs.live_cell_line(row)


def test_mode_live_progress_adapter_keeps_frame_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(zz, "_STACK_LIVE_TELEMETRY", {
        "dir": str(tmp_path), "run_id": "run", "generation": 1,
        "interval_s": -1.0,
    })
    monkeypatch.setattr(zz, "_STACK_LIVE_WRITERS", {})
    writer = zz._live_writer_for("c1")
    assert writer is not None
    writer.state("RUNNING")
    zz._live_progress({
        "cell_id": "c1", "phase": "write", "tile_current": 1,
        "tile_total": 6, "frame_current": 5, "frame_total": 100,
        "frame_prepared": 5, "bytes_reserved": 1234, "bytes_peak": 2000,
        "producers_effective": 2,
    })
    zz._live_progress({
        "cell_id": "c1", "phase": "write", "tile_current": 1,
        "tile_total": 6, "frame_current": 50, "frame_total": 100,
        "frame_prepared": 50, "bytes_reserved": 999, "bytes_peak": 2000,
        "producers_effective": 2,
    })
    path = zcs.live_cell_path(tmp_path, "c1")
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    progress = [l for l in lines if l["kind"] == "PROGRESS"]
    assert len(progress) == 2
    assert [p["frame_current"] for p in progress] == [5, 50]
    assert progress[1]["bytes_reserved"] == 999
    assert progress[1]["producers_effective"] == 2


def test_old_progress_events_without_frame_still_work(tmp_path):
    path = tmp_path / "c1.jsonl"
    writer = zcs.LiveTelemetryWriter(
        path, run_id="run", generation=1, cell_id="c1",
        min_interval_s=0.0,
    )
    writer.state("RUNNING")
    assert writer.progress("write", tile_current=2, tile_total=6) is True
    reader = zcs.LiveTelemetryReader(tmp_path, run_id="run", generation=1)
    states = zcs.LiveCellStates()
    states.reset(["c1"])
    zcs.apply_events(states, reader.poll(["c1"]))
    row = next(r for r in states.snapshot()["rows"] if r["cell_id"] == "c1")
    assert row["tile"] == [2, 6]
    assert "frame" not in row


def test_real_caller_descending_probe_falls_back_sticky(tmp_path, monkeypatch):
    ctx, patch, _planned = _fits_corpus(tmp_path, n=2)
    cfg = MiniTileScienceConfig(
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean", backend="cpu",
        taper="footprint", taper_px=8.0, taper_floor=0.0,
        reference_index=None, equalize_rgb=False,
        sigma_low=3.0, sigma_high=3.0,
        winsor_limit_low=0.05, winsor_limit_high=0.05,
    )
    values = [64 * 1024 ** 3, None]

    def _probe():
        return {"raw_available_bytes": (
            values.pop(0) if values else None),
            "observed_monotonic": time.monotonic(), "source": "test"}

    def _probe_or_raise():
        if not values:
            raise RuntimeError("probe unavailable")
        return _probe()

    monkeypatch.setattr(zz.zrec, "default_capacity_probe", _probe_or_raise)
    monkeypatch.setenv("ZEMOSAIC_M33_PREP_PIPELINE", "auto")
    (tmp_path / "spool_dn").mkdir()
    info = {}
    zz._run_cell_external(
        tmp_path / "cache", patch, cfg, None, tile_size=16, column_batch=16,
        spool_root=tmp_path / "spool_dn", fixed=None,
        storage_profile="bounded", storage_info=info,
        cell_id="c", attempt_id="dn", source_context=ctx,
    )
    prep = info["external"]["preparation_pipeline"]
    runtime = prep["runtime"]
    assert prep["effective_mode"] == "auto"          # initial admission
    assert runtime["fallback_at_tile"] == 2
    assert runtime["sticky_fallback_reason"] == "host_available_unknown"
    assert runtime["effective_mode_last"] == "legacy"
    assert runtime["tiles_pipeline"] == 1
    assert runtime["tiles_legacy"] == runtime["tiles"] - 1


def test_cfa_registry_single_flight_publishes_immutably():
    registry = zpp.CfaMinMaxRegistry()
    start = threading.Barrier(2)
    calls = {"n": 0}
    lock = threading.Lock()

    def compute():
        with lock:
            calls["n"] += 1
        time.sleep(0.05)
        return (1.5, 9.5)

    results = []

    def worker():
        start.wait(timeout=5.0)
        results.append(registry.get_or_compute(("f.fits", "h", (10, 10), "p"), compute))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert calls["n"] == 1
    stats = registry.stats()
    assert stats["computed"] == 1
    assert stats["entries"] == 1
    assert stats["hits"] + stats["waited"] >= 1
    values = [r[0] for r in results]
    assert values == [(1.5, 9.5), (1.5, 9.5)]
    assert sorted(r[1] for r in results) == [False, True]
