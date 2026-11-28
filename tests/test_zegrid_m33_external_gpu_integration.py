"""ZM-M33 lot 3 — external GPU product integration (bounded witnesses).

These tests prove the BOUNDED integration, never a hardware qualification:

* the pure external GPU admission prices one ``(N, positions)`` block that
  always holds ALL N contributors, and declines truthfully on unknown VRAM,
  a too-small block, non-ACTIVE scheduling, K>1 or a CPU-frozen run;
* the REAL product external selector + bundle carry requested/selected/
  effective backend, the device/batch/probe record and the CPU fallback;
* the REAL external executor is dispatched to ``backend='gpu'`` through an
  EXPLICIT CPU mock (``_get_xp -> numpy``): no real CUDA work is claimed;
* a GPU Cell that falls back mid-tiles persists per-tile effective backend
  provenance, survives a crash, and a NEW attempt restores the sticky CPU
  fallback instead of relabelling reused tiles as GPU;
* CPU checkpoint identity/content is unchanged (no `backend` key on CPU
  markers), and a classified GPU error falls back while a programming error
  propagates.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import InMemoryCanonicalProvider
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import external_stream as x
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import gpu_external as zgex
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import run_resume as zres
from zemosaic.core.zegrid import run_store as zrs
from zemosaic.core.zegrid.rejection_sink import BoundedCountingRejectionSink

MIB = 2 ** 20
GIB = 2 ** 30

_PLANES = (
    "science",
    "estimator_weight_sum",
    "valid_mask",
    "surviving_sample_count",
    "support_w1",
    "support_w2",
    "n_eff_support",
)

_FakeCupyOOM = type(
    "OutOfMemoryError", (Exception,), {"__module__": "cupy.cuda.memory"}
)


# ---------------------------------------------------------------------------
# Tiny synthetic corpus / request / provider
# ---------------------------------------------------------------------------

def _corpus(n=9, h=17, w=19, seed=5):
    rng = np.random.default_rng(seed)
    base = np.array([50, 55, 60], np.float32)
    frames = []
    for i in range(n):
        chan = np.stack(
            [base[c] + 2.0 * i + rng.normal(0, 3, (h, w)) for c in range(3)],
            axis=-1,
        )
        frames.append(np.ascontiguousarray(chan.astype(np.float32)))
    frames[1][4:7, 5:9, :] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[3][:, w // 2:] = False
    return frames, masks


def _request(frames, masks, *, backend="cpu"):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="winsorized_sigma_clip",
        combine="mean",
        taper="footprint",
        taper_px=2.0,
        sigma_low=3.0,
        sigma_high=3.0,
        backend=backend,
    )


def _source_digest(frames, masks):
    digest = hashlib.sha256()
    for frame, mask in zip(frames, masks):
        arr = np.ascontiguousarray(frame)
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
        digest.update(arr.tobytes(order="C"))
        digest.update(np.ascontiguousarray(mask, dtype=bool).tobytes(order="C"))
    return digest.hexdigest()


def _run_executor(
    root, frames, masks, *, request=None, checkpoint=False, checkpoint_out=None,
    gpu_fallback_out=None, tile_gate=None, tile_size=8, column_batch=64,
    attempt_id="a1", gpu_probe=None,
):
    root = Path(root)
    (root / "spool").mkdir(parents=True, exist_ok=True)
    req = request if request is not None else _request(frames, masks)
    sink = BoundedCountingRejectionSink(
        (len(frames), frames[0].shape[0], frames[0].shape[1], 3)
    )
    provider = InMemoryCanonicalProvider(frames, masks)
    kwargs = {}
    if checkpoint:
        kwargs = {
            "checkpoint_root": str(root / "checkpoints"),
            "checkpoint_run_id": "run-1",
            "checkpoint_source_digest": _source_digest(frames, masks),
        }
        if checkpoint_out is not None:
            kwargs["checkpoint_out"] = checkpoint_out
    return x.run_cell_streaming_external(
        provider,
        req,
        tile_size=int(tile_size),
        column_batch=int(column_batch),
        spool_root=str(root / "spool"),
        rejection_sink=sink,
        cell_id="cellA",
        attempt_id=attempt_id,
        gpu_fallback_out=gpu_fallback_out,
        tile_gate=tile_gate,
        # Explicit CPU-mock test control: never rely on the host's real device.
        gpu_probe=(dict(_PROBE) if gpu_probe is None else gpu_probe),
        **kwargs,
    ), sink


def _assert_planes_equal(ref, got):
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
    assert ref.iterations_used == got.iterations_used
    assert ref.initial_sample_count == got.initial_sample_count
    assert ref.rejected_sample_count == got.rejected_sample_count
    assert ref.low_n_cell_count == got.low_n_cell_count
    assert ref.degenerate_cell_count == got.degenerate_cell_count


# ---------------------------------------------------------------------------
# 1. Pure admission policy
# ---------------------------------------------------------------------------

_PROBE = {
    "available": True,
    "device": "Mock GPU",
    "cupy_version": "0-test",
    "vram_total_bytes": 2 * GIB,
    "vram_free_bytes": 2 * GIB,
    "reason": None,
}


def test_policy_prices_all_n_and_shrinks_only_on_pressure():
    r = zgex.plan_external_gpu(
        requested=True, probe=_PROBE, host_phase_budget_bytes=1 << 40,
        host_batch=969, n_frames=11, interior_hw=(17, 19), channels=3,
        tile_size=16, cells_in_flight=1, active_scheduler=True,
        frozen_backend=None,
    )
    assert r["admitted"] is True
    assert r["batch"]["positions"] == 969
    assert r["device"]["block_bytes"] == zgex.external_gpu_block_bytes(
        n_frames=11, positions=969
    )
    # Every position always carries ALL N contributors: the model is N-scaled.
    assert r["device"]["block_bytes"] >= 11 * 969 * 96

    # Same interior, huge N, small-but-usable device: the block shrinks to one
    # row of whole pixels while STILL holding every contributor.
    tiny_probe = {
        "available": True,
        "vram_total_bytes": 80 * MIB,
        "vram_free_bytes": 80 * MIB,
    }
    s = zgex.plan_external_gpu(
        requested=True, probe=tiny_probe, host_phase_budget_bytes=1 << 40,
        host_batch=969, n_frames=10000, interior_hw=(17, 19), channels=3,
        tile_size=16, cells_in_flight=1, active_scheduler=True,
        frozen_backend=None,
    )
    assert s["admitted"] is True
    assert s["batch"]["clamped"] is True
    assert 0 < s["batch"]["positions"] < 969
    assert s["batch"]["positions"] % 3 == 0
    assert s["device"]["block_bytes"] <= s["device"]["block_budget_bytes"]


def test_policy_unknown_free_and_malformed_probe_never_raise():
    """G1/G2: unknown FREE declines even with a known TOTAL; malformed probes
    decline as ``probe_invalid`` instead of raising. ``numpy.int64`` is a real
    library probe type and is accepted/normalized to a plain JSON int."""
    common = dict(
        requested=True, host_phase_budget_bytes=512 * MIB, host_batch=6528,
        n_frames=20000, interior_hw=(64, 128), channels=3, tile_size=64,
    )
    # G1: a known 8 GiB total does NOT substitute for an unknown free figure.
    r = zgex.plan_external_gpu(
        probe={"available": True, "vram_total_bytes": 8 * GIB,
               "vram_free_bytes": None},
        **common,
    )
    assert r["admitted"] is False
    assert r["reason"] == "vram_unknown"
    json.dumps(r)  # JSON-safe record

    cases = [
        {"available": True, "vram_free_bytes": 1.5},
        {"available": True, "vram_free_bytes": "8GiB"},
        {"available": True, "vram_free_bytes": True},
        {"available": True, "vram_free_bytes": -1},
        {"available": True, "vram_free_bytes": 9 * GIB,
         "vram_total_bytes": 8 * GIB},
        {"available": 1, "vram_free_bytes": 8 * GIB},
        {"available": True, "vram_total_bytes": 1.5,
         "vram_free_bytes": 1 * GIB},
        "not-a-dict",
        None,
    ]
    for probe in cases:
        got = zgex.plan_external_gpu(probe=probe, **common)
        assert got["admitted"] is False, probe
        assert got["reason"] == "probe_invalid", (probe, got["reason"])
        json.dumps(got)

    # A real numpy integer is accepted and normalized to a plain int.
    accepted = zgex.plan_external_gpu(
        probe={"available": True,
               "vram_free_bytes": np.int64(2 * GIB),
               "vram_total_bytes": np.int64(2 * GIB)},
        requested=True, host_phase_budget_bytes=512 * MIB, host_batch=30,
        n_frames=11, interior_hw=(2, 5), channels=3, tile_size=4,
    )
    assert accepted["admitted"] is True
    assert isinstance(accepted["probe"]["vram_free_bytes"], int)
    assert not isinstance(accepted["probe"]["vram_free_bytes"], np.int64)
    json.dumps(accepted)


def test_point_of_use_revalidates_frozen_block_and_never_raises():
    common = dict(n_frames=11, positions=30, interior_hw=(2, 5), channels=3)
    assert zgex.validate_external_gpu_point_of_use(
        probe={"available": True, "vram_free_bytes": None,
               "vram_total_bytes": 8 * GIB}, **common
    )["reason"] == "vram_unknown"
    assert zgex.validate_external_gpu_point_of_use(
        probe={"available": True, "vram_free_bytes": "8GiB"}, **common
    )["reason"] == "probe_invalid"
    assert zgex.validate_external_gpu_point_of_use(
        probe={"available": True, "vram_free_bytes": 100 * MIB,
               "vram_total_bytes": 100 * MIB}, **common
    )["ok"] is True
    # A current free figure too small for the FROZEN block declines; it is not
    # shrunk or re-identified here.
    assert zgex.validate_external_gpu_point_of_use(
        probe={"available": True, "vram_free_bytes": 80 * MIB,
               "vram_total_bytes": 2 * GIB}, n_frames=20000,
        positions=969, interior_hw=(17, 19), channels=3,
    )["reason"] == "vram_block_floor"


def test_policy_declines_truthfully_on_unknown_tiny_k2_frozen_and_non_active():
    common = dict(
        host_phase_budget_bytes=512 * MIB, host_batch=30,
        n_frames=11, interior_hw=(2, 5), channels=3, tile_size=4,
    )
    assert zgex.plan_external_gpu(
        requested=True, probe={"available": True}, **common
    )["reason"] == "vram_unknown"
    assert zgex.plan_external_gpu(
        requested=True,
        probe={"available": False, "reason": "cupy_or_cuda_unavailable"},
        **common
    )["reason"] == "gpu_unavailable"
    assert zgex.plan_external_gpu(
        requested=True,
        probe={"available": True, "vram_free_bytes": 10 * MIB,
               "vram_total_bytes": 10 * MIB}, **common
    )["reason"] == "vram_below_floor"
    assert zgex.plan_external_gpu(
        requested=True, probe=_PROBE, cells_in_flight=2, **common
    )["reason"] == "gpu_owner_concurrency_refused"
    assert zgex.plan_external_gpu(
        requested=True, probe=_PROBE, frozen_backend="cpu", **common
    )["reason"] == "frozen_backend_cpu"
    assert zgex.plan_external_gpu(
        requested=True, probe=_PROBE, active_scheduler=False, **common
    )["reason"] == "scheduler_not_active"
    assert zgex.plan_external_gpu(
        requested=False, probe=_PROBE, **common
    )["reason"] == "gpu_not_requested"
    # A huge N on a small device that cannot hold ONE pixel still declines.
    assert zgex.plan_external_gpu(
        requested=True,
        probe={"available": True, "vram_free_bytes": 80 * MIB,
               "vram_total_bytes": 80 * MIB},
        host_phase_budget_bytes=512 * MIB, host_batch=30, n_frames=10 ** 7,
        interior_hw=(2, 5), channels=3, tile_size=4,
    )["reason"] == "vram_block_floor"


# ---------------------------------------------------------------------------
# 2. REAL product selector / bundle plumbing
# ---------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(*_a, **_k):
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


def _make_tan(i, h=40, w=40, scale=0.001, ra_step=0.004):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


@pytest.fixture
def _product(monkeypatch):
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 62))
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    descs = [_make_tan(i) for i in range(4)]
    canvas = zg.build_canvas(descs)
    return descs, canvas


def _build_bundle(tmp_path, _product, *, probe):
    descs, canvas = _product
    return zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=2 * GIB,
        budget_bytes=rcontract.plan_budget_bytes(2 * GIB),
        cpu_count=8,
        cache_root=str(tmp_path / "cache"),
        force_external=True,
        gpu_used=True,
        gpu_requested=True,
        gpu_probe=probe,
        active_scheduler=True,
        output_dir=str(tmp_path / "out"),
    )


def test_real_bundle_admits_external_gpu_and_keeps_external_working_set(
    tmp_path, _product
):
    bundle = _build_bundle(tmp_path, _product, probe=_PROBE)
    assert bundle.chosen_mode == "external"
    assert bundle.stack_working_set == zz.WORKING_SET_EXTERNAL
    assert bundle.gpu_used is True
    assert bundle.external_contract["backend"] == "gpu"
    assert bundle.external_contract["gpu"]["admitted"] is True
    assert bundle.gpu_pre_execution_fallback is None
    # The external pseudo-plan is device-serialized (K=1) but never the
    # resident whole-tile GPU rung.
    assert bundle.plan["gpu_serialized"] is True
    assert bundle.gpu_streaming_tile is None
    contract = zz._r55_execution_contract_from_bundle(bundle)
    assert contract["backend"] == "gpu"
    assert contract["tile_px"] == bundle.external_contract["tile_size"]
    assert contract["parity_qualified"] is False
    ws = zz._r55_run_store_working_set(bundle, contract)
    assert ws["class"] == zz.WORKING_SET_EXTERNAL
    assert ws["tile_px"] == bundle.external_contract["tile_size"]
    assert ws["column_batch"] == bundle.external_contract["column_batch"]
    # Every executable external task carries the SAME frozen backend fact.
    for task in bundle.cell_tasks:
        if task[4] is None:
            continue
        assert task[4] == "external"
        assert task[7]["backend"] == "gpu"
        assert task[7]["gpu"]["admitted"] is True


def test_real_bundle_declines_gpu_before_frozen_admission(tmp_path, _product):
    bundle = _build_bundle(
        tmp_path, _product,
        probe={"available": False, "reason": "cupy_or_cuda_unavailable"},
    )
    assert bundle.chosen_mode == "external"
    assert bundle.gpu_used is False
    assert bundle.external_contract["backend"] == "cpu"
    assert bundle.external_contract["gpu"]["admitted"] is False
    assert bundle.gpu_pre_execution_fallback["reason"] == "gpu_unavailable"
    contract = zz._r55_execution_contract_from_bundle(bundle)
    assert contract["backend"] == "cpu"
    for task in bundle.cell_tasks:
        if task[4] is None:
            continue
        assert task[7]["backend"] == "cpu"
    # A CPU-frozen resume is NEVER offered the external GPU plan.
    rec = zz._external_gpu_admission(
        {"tile": 16, "column_batch": 64, "executed_column_batch": 64,
         "phase_budget_bytes": 512 * MIB, "equalize_rgb": False,
         "rejection": "winsorized_sigma_clip", "combine": "mean",
         "pricing": {"geometry": {"n_frames": 4, "height": 40, "width": 40,
                                  "channels": 3, "halo_px": 8}}},
        requested=True, probe=_PROBE, n_frames=4, patch_hw=(40, 40),
        channels=3, halo=8, cells_in_flight=1, active_scheduler=True,
        frozen_backend="cpu",
    )
    assert rec["admitted"] is False
    assert rec["reason"] == "frozen_backend_cpu"
    # A GPU-frozen run that selects no GPU is a typed refusal, never CPU.
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._effective_run_backend(
            {"backend": "gpu", "tile_px": 16, "parity_qualified": False},
            gpu_used=False,
        )


# ---------------------------------------------------------------------------
# 3. REAL external executor dispatched to the GPU branch with a CPU mock
# ---------------------------------------------------------------------------

def test_executor_gpu_mock_checkpoint_crash_and_sticky_cpu_resume(
    tmp_path, monkeypatch
):
    frames, masks = _corpus()
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)

    real_reduce = x.reduce_spooled_tile
    failed = {"oom": False}
    calls: list[str] = []

    def _flaky(handle, p1, request, *, column_batch, rejection_sink=None,
               rejection_mask=None, backend="cpu", **kw):
        calls.append(str(backend))
        if (
            backend == "gpu"
            and not failed["oom"]
            and (int(handle.y0), int(handle.x0)) == (0, 8)
        ):
            failed["oom"] = True
            raise _FakeCupyOOM("injected external GPU OOM")
        return real_reduce(
            handle, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend, **kw,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _flaky)
    req = _request(frames, masks, backend="gpu")
    out_a: dict = {}
    fb_a: dict = {}
    gates = {"n": 0}

    def _gate():
        gates["n"] += 1
        if gates["n"] >= 3:
            raise RuntimeError("injected crash before tile 3")

    with pytest.raises(RuntimeError, match="injected crash"):
        _run_executor(
            tmp_path / "gpu", frames, masks, request=req, checkpoint=True,
            checkpoint_out=out_a, gpu_fallback_out=fb_a, tile_gate=_gate,
            attempt_id="a1",
        )
    # Tile 1 GPU, tile 2 fell back and replayed on CPU, THEN the crash.
    assert calls[:3] == ["gpu", "gpu", "cpu"]
    assert fb_a["fallback_count"] == 1
    assert fb_a["effective_backend"] == "cpu"
    assert fb_a["sticky_cpu"] is True
    assert out_a["saved_tiles"] == 2
    assert out_a["gpu_tiles"] == 1 and out_a["cpu_tiles"] == 1
    assert out_a["effective_backend"] == "cpu"
    markers = {}
    for index in (1, 2):
        path = (
            tmp_path / "gpu" / "checkpoints" / "cellA" / "tiles"
            / f"t{index:06d}" / "commit.json"
        )
        markers[index] = json.loads(path.read_text())["backend"]
    assert markers[1] == {"requested": "gpu", "effective": "gpu"}
    assert markers[2] == {"requested": "gpu", "effective": "cpu"}

    # A NEW attempt (new id) reopens the SAME checkpoints. The stored CPU tile
    # restores the sticky fallback: no GPU call may happen again.
    failed["oom"] = True
    calls.clear()
    out_b: dict = {}
    fb_b: dict = {}
    got, _sink = _run_executor(
        tmp_path / "gpu", frames, masks, request=req, checkpoint=True,
        checkpoint_out=out_b, gpu_fallback_out=fb_b, attempt_id="a2",
    )
    assert calls and set(calls) == {"cpu"}, calls
    assert out_b["reused_tiles"] == 2
    assert out_b["restored_sticky_cpu"] is True
    assert out_b["gpu_tiles"] == 1 and out_b["cpu_tiles"] == 8
    assert fb_b["restored_sticky_cpu"] is True
    assert fb_b["effective_backend"] == "cpu"

    # Seven scientific planes/counters stay bit-exact against a CPU reference.
    ref, _ = _run_executor(
        tmp_path / "cpu", frames, masks,
        request=_request(frames, masks, backend="cpu"), attempt_id="ref",
    )
    _assert_planes_equal(ref, got)


def test_executor_programming_error_propagates_and_cpu_marker_has_no_backend(
    tmp_path, monkeypatch
):
    frames, masks = _corpus()
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)
    real_reduce = x.reduce_spooled_tile

    def _boom(handle, p1, request, *, column_batch, **kw):
        raise ValueError("science/programming error")

    monkeypatch.setattr(x, "reduce_spooled_tile", _boom)
    with pytest.raises(ValueError, match="science/programming"):
        _run_executor(
            tmp_path / "gpu", frames, masks,
            request=_request(frames, masks, backend="gpu"),
        )
    # No checkpoint marker was written for the failed tile.
    cell_dir = tmp_path / "gpu" / "checkpoints" / "cellA"
    assert not (cell_dir / "tiles" / "t000001" / "commit.json").exists()

    # CPU checkpoint markers keep their historical shape (no backend key).
    monkeypatch.setattr(x, "reduce_spooled_tile", real_reduce)
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "cpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)
    out = {}
    _run_executor(
        tmp_path / "cpu", frames, masks,
        request=_request(frames, masks, backend="cpu"), checkpoint=True,
        checkpoint_out=out, attempt_id="cpu1",
    )
    marker = json.loads(
        (
            tmp_path / "cpu" / "checkpoints" / "cellA" / "tiles"
            / "t000001" / "commit.json"
        ).read_text()
    )
    assert "backend" not in marker
    assert out["requested_backend"] == "cpu"
    assert out["effective_backend"] == "cpu"


def test_executor_point_of_use_fallback_after_historical_gpu_checkpoints(
    tmp_path, monkeypatch
):
    """G3 point-of-use: historical GPU tiles restore, NEW work declines to CPU.

    Run A checkpoints tiles 1-2 on the GPU then crashes. Run B starts with an
    UNKNOWN current free-VRAM observation: the two historical GPU tiles are
    still reused (they are not rejected just because the current device cannot
    be revalidated), the first NEW tile triggers the fail-closed point check,
    and every remaining computed tile executes on CPU with NO GPU attempt.
    """
    frames, masks = _corpus()
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)
    req = _request(frames, masks, backend="gpu")

    gates = {"n": 0}

    def _gate():
        gates["n"] += 1
        if gates["n"] >= 3:
            raise RuntimeError("injected crash before tile 3")

    out_a, fb_a = {}, {}
    with pytest.raises(RuntimeError, match="injected crash"):
        _run_executor(
            tmp_path / "gpu", frames, masks, request=req, checkpoint=True,
            checkpoint_out=out_a, gpu_fallback_out=fb_a, tile_gate=_gate,
            attempt_id="a1", gpu_probe=dict(_PROBE),
        )
    assert out_a["saved_tiles"] == 2
    assert out_a["gpu_tiles"] == 2

    real_reduce = x.reduce_spooled_tile
    calls = []

    def _counting(handle, p1, request, *, column_batch, rejection_sink=None,
                  rejection_mask=None, backend="cpu", **kw):
        calls.append(str(backend))
        return real_reduce(
            handle, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend, **kw,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting)
    out_b, fb_b = {}, {}
    got, _sink = _run_executor(
        tmp_path / "gpu", frames, masks, request=req, checkpoint=True,
        checkpoint_out=out_b, gpu_fallback_out=fb_b, attempt_id="a2",
        gpu_probe={"available": True, "vram_total_bytes": 2 * GIB,
                   "vram_free_bytes": None},
    )
    assert calls and set(calls) == {"cpu"}, calls
    assert out_b["reused_tiles"] == 2
    assert out_b["gpu_tiles"] == 2
    assert out_b["cpu_tiles"] == 7
    assert out_b["effective_backend"] == "cpu"
    assert out_b["device_fallback"] is True
    assert out_b["device_fallback_reason"] == "vram_unknown"
    assert fb_b["device_fallback"] is True
    assert fb_b["device_revalidation"]["reason"] == "vram_unknown"
    # Historical GPU markers are preserved; new tiles are CPU.
    for index, expected in ((1, "gpu"), (2, "gpu"), (3, "cpu")):
        marker = json.loads(
            (
                tmp_path / "gpu" / "checkpoints" / "cellA" / "tiles"
                / f"t{index:06d}" / "commit.json"
            ).read_text()
        )
        assert marker["backend"]["requested"] == "gpu"
        assert marker["backend"]["effective"] == expected

    # Seven planes/counters remain bit-exact against a CPU reference.
    ref, _ = _run_executor(
        tmp_path / "cpu", frames, masks,
        request=_request(frames, masks, backend="cpu"), attempt_id="ref",
        gpu_probe=dict(_PROBE),
    )
    _assert_planes_equal(ref, got)


def test_product_bundle_passes_real_k_and_refuses_k2(tmp_path, _product):
    """G4: the product passes the REAL plan K (not a literal 1)."""
    captured = []
    real = zz._external_gpu_admission

    def _spy(selection, **kwargs):
        captured.append(int(kwargs.get("cells_in_flight")))
        return real(selection, **kwargs)

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as m:
        m.setattr(zz, "_external_gpu_admission", _spy)
        bundle = _build_bundle(tmp_path, _product, probe=_PROBE)
    assert captured == [1]
    assert bundle.plan["cells_in_flight"] == 1
    assert bundle.plan["gpu_serialized"] is True
    assert int(bundle.cells_in_flight) == 1
    # The product helper itself refuses a simulated K=2 plan; no GPU parallel
    # external path exists.
    rec = zz._external_gpu_admission(
        {"tile": 16, "column_batch": 64, "executed_column_batch": 64,
         "phase_budget_bytes": 512 * MIB, "equalize_rgb": False,
         "rejection": "winsorized_sigma_clip", "combine": "mean",
         "host_peak_bytes": 1 * MIB, "pricing": {"geometry": {
             "n_frames": 4, "height": 40, "width": 40, "channels": 3,
             "halo_px": 8}}},
        requested=True, probe=_PROBE, n_frames=4, patch_hw=(40, 40),
        channels=3, halo=8, cells_in_flight=2, active_scheduler=True,
        frozen_backend=None,
    )
    assert rec["admitted"] is False
    assert rec["reason"] == "gpu_owner_concurrency_refused"


def test_product_builder_runstore_frozen_external_gpu_close_reopen(
    tmp_path, _product
):
    """G5 wiring: REAL builder -> frozen identity -> store close/reopen.

    CPU-only (no GPU execution): asserts the external GPU contract freezes
    backend=gpu + external_spool tile/batch, the admission is valid, and a NEW
    RunCellStore open on the same identity is compatible (resume wiring).
    """
    descs = []
    for i in range(4):
        base = _make_tan(i)
        path = tmp_path / f"d{i}.fits"
        header = fits.Header.fromstring(base.wcs_header)
        fits.PrimaryHDU(
            data=np.full((40, 40), 100.0 + i, dtype=np.float32),
            header=header,
        ).writeto(path, overwrite=True)
        descs.append(dataclasses.replace(base, source_path=str(path)))
    canvas = zg.build_canvas(descs)
    bundle = zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=2 * GIB,
        budget_bytes=rcontract.plan_budget_bytes(2 * GIB),
        cpu_count=8, cache_root=str(tmp_path / "cache"),
        force_external=True, gpu_used=True, gpu_requested=True,
        gpu_probe=_PROBE, active_scheduler=True,
        output_dir=str(tmp_path / "out"),
    )
    assert bundle.gpu_used is True
    science_config = SimpleNamespace(
        backend="gpu", normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean", taper="footprint",
        taper_px=2.0, sigma_low=3.0, sigma_high=3.0, equalize_rgb=False,
        cell_gauge_mode="global", contract_version="zegrid-science-v1",
    )
    gauge = SimpleNamespace(reference_index=0, exclusions=())
    identity = zz._r55_build_run_store_identity(
        run_id="rgpu-run", descs=descs, canvas=canvas, bundle=bundle,
        halo_px=int(bundle.layout.get("halo_px", 8) or 8),
        halo_provenance=None, global_gauge=gauge,
        global_frame_ids=[d.frame_id.logical_path for d in descs],
        gauge_mode="global", science_config=science_config,
        workers=1, prefetch_units=0, effective_backend="gpu",
    )
    backend = identity["science_contract"]["backend"]
    assert backend["backend"] == "gpu"
    assert backend["tile_px"] == bundle.external_contract["tile_size"]
    assert backend["parity_qualified"] is False
    admission = zz._admission_from_identity(identity)
    assert zres.validate_admission(admission)["backend"] == "gpu"
    ws = zz._r55_run_store_working_set(
        bundle, zz._r55_execution_contract_from_bundle(bundle)
    )
    assert ws["class"] == zz.WORKING_SET_EXTERNAL
    assert ws["tile_px"] == bundle.external_contract["tile_size"]
    assert ws["column_batch"] == bundle.external_contract["column_batch"]
    budget = zrs.DiskBudget(
        durable_cell_bytes=1 << 30,
        transactional_margin_bytes=zrs.MIN_TRANSACTIONAL_MARGIN_BYTES,
    )
    store = zrs.RunCellStore.open(
        tmp_path / "store", identity=identity, budget=budget,
        session_id="s1",
    )
    assert store.run_id == "rgpu-run"
    store.close()
    # Close/reopen compatibility is the resume wiring proof (no GPU execution).
    reopened = zrs.RunCellStore.open(
        tmp_path / "store", identity=identity, budget=budget,
        session_id="s2",
    )
    assert reopened.run_id == "rgpu-run"
    reopened.close()


def test_point_of_use_probe_runs_after_write_and_before_gpu_reduce(
    tmp_path, monkeypatch
):
    """M1: probe AFTER the CPU spool write, BEFORE the first GPU reduce."""
    frames, masks = _corpus()
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)

    state = {"write_done": False, "probe_calls": 0,
             "probe_saw_write_done": [], "gpu_reduces": 0}
    real_write = x.ExternalSpool.write_tile

    def _write(self, *args, **kwargs):
        out = real_write(self, *args, **kwargs)
        state["write_done"] = True
        return out

    monkeypatch.setattr(x.ExternalSpool, "write_tile", _write)

    def _probe():
        state["probe_calls"] += 1
        state["probe_saw_write_done"].append(bool(state["write_done"]))
        return {"available": True, "vram_total_bytes": 2 * GIB,
                "vram_free_bytes": None}

    real_reduce = x.reduce_spooled_tile

    def _counting(handle, p1, request, *, column_batch, rejection_sink=None,
                  rejection_mask=None, backend="cpu", **kw):
        if backend == "gpu":
            state["gpu_reduces"] += 1
        return real_reduce(
            handle, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend, **kw,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting)
    out, fb = {}, {}
    got, _sink = _run_executor(
        tmp_path / "gpu", frames, masks,
        request=_request(frames, masks, backend="gpu"), checkpoint=True,
        checkpoint_out=out, gpu_fallback_out=fb, attempt_id="a1",
        gpu_probe=_probe,
    )
    # Exactly ONE point-of-use probe, AFTER the write, and ZERO GPU reduces.
    assert state["probe_calls"] == 1
    assert state["probe_saw_write_done"] == [True]
    assert state["gpu_reduces"] == 0
    assert out["effective_backend"] == "cpu"
    assert out["gpu_tiles"] == 0 and out["cpu_tiles"] == out["tile_total"]
    assert out["device_fallback"] is True
    assert out["device_fallback_reason"] == "vram_unknown"
    assert fb["device_fallback"] is True
    # CPU provenance on every saved marker; requested identity unchanged.
    for index in range(1, out["tile_total"] + 1):
        marker = json.loads(
            (
                tmp_path / "gpu" / "checkpoints" / "cellA" / "tiles"
                / f"t{index:06d}" / "commit.json"
            ).read_text()
        )
        assert marker["backend"] == {"requested": "gpu", "effective": "cpu"}
    ref, _ = _run_executor(
        tmp_path / "cpu", frames, masks,
        request=_request(frames, masks, backend="cpu"), attempt_id="ref",
        gpu_probe=dict(_PROBE),
    )
    _assert_planes_equal(ref, got)
    # Checkpoint cleanup is parent-authorised and removes the Cell directory.
    cell_dir = tmp_path / "gpu" / "checkpoints" / "cellA"
    from zemosaic.core.zegrid import tile_checkpoints as ztcp
    ztcp.cleanup_cell(tmp_path / "gpu" / "checkpoints", "cellA")
    assert not cell_dir.exists()


def test_point_of_use_skipped_when_cell_only_reuses_checkpoints(
    tmp_path, monkeypatch
):
    """M1: an all-reuse attempt performs ZERO device probes and reductions."""
    frames, masks = _corpus()
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)
    req = _request(frames, masks, backend="gpu")
    out_a = {}
    got_a, _ = _run_executor(
        tmp_path / "gpu", frames, masks, request=req, checkpoint=True,
        checkpoint_out=out_a, attempt_id="a1", gpu_probe=dict(_PROBE),
    )
    assert out_a["gpu_tiles"] == out_a["tile_total"]

    state = {"probes": 0, "reduces": 0}
    real_reduce = x.reduce_spooled_tile

    def _counting(*args, **kwargs):
        state["reduces"] += 1
        return real_reduce(*args, **kwargs)

    def _probe():
        state["probes"] += 1
        raise AssertionError("an all-reuse Cell must not probe the device")

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting)
    out_b = {}
    got_b, _ = _run_executor(
        tmp_path / "gpu", frames, masks, request=req, checkpoint=True,
        checkpoint_out=out_b, attempt_id="a2", gpu_probe=_probe,
    )
    assert state["probes"] == 0
    assert state["reduces"] == 0
    assert out_b["reused_tiles"] == out_b["tile_total"]
    assert out_b["effective_backend"] == "gpu"
    _assert_planes_equal(got_a, got_b)


def test_mixed_historical_gpu_after_cpu_hole_restores_honestly(
    tmp_path, monkeypatch
):
    """Adversarial resume: CPU-recomputed tile 1, historical GPU tiles 2..N.

    The verified GPU tiles are restored (historical provenance) while the
    current execution stays sticky CPU for NEW work. A third all-reuse attempt
    restores the mixed set without any probe/GPU reduction and reports truthful
    counts / final effective CPU.
    """
    frames, masks = _corpus()
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "gpu")
    monkeypatch.setattr(x, "_get_xp", lambda _token: np)
    req = _request(frames, masks, backend="gpu")
    root = tmp_path / "gpu"
    cp_root = root / "checkpoints"
    out_a = {}
    _run_executor(
        root, frames, masks, request=req, checkpoint=True,
        checkpoint_out=out_a, attempt_id="a1", gpu_probe=dict(_PROBE),
    )
    total = int(out_a["tile_total"])
    assert out_a["gpu_tiles"] == total
    # Make tile 1 unreusable (the canonical recompute_corrupt path).
    shutil.rmtree(cp_root / "cellA" / "tiles" / "t000001")

    state = {"gpu_reduces": 0}
    real_reduce = x.reduce_spooled_tile

    def _counting(handle, p1, request, *, column_batch, rejection_sink=None,
                  rejection_mask=None, backend="cpu", **kw):
        if backend == "gpu":
            state["gpu_reduces"] += 1
        return real_reduce(
            handle, p1, request, column_batch=column_batch,
            rejection_sink=rejection_sink, rejection_mask=rejection_mask,
            backend=backend, **kw,
        )

    monkeypatch.setattr(x, "reduce_spooled_tile", _counting)
    out_b, fb_b = {}, {}
    got_b, _ = _run_executor(
        root, frames, masks, request=req, checkpoint=True,
        checkpoint_out=out_b, gpu_fallback_out=fb_b, attempt_id="a2",
        gpu_probe={"available": True, "vram_total_bytes": 2 * GIB,
                   "vram_free_bytes": None},
    )
    assert state["gpu_reduces"] == 0, state
    assert out_b["recomputed_tiles"] == 1
    assert out_b["reused_tiles"] == total - 1
    assert out_b["gpu_tiles"] == total - 1
    assert out_b["cpu_tiles"] == 1
    assert out_b["effective_backend"] == "cpu"
    assert out_b["device_fallback"] is True
    assert fb_b["device_fallback_reason"] == "vram_unknown"
    assert out_b["restored_sticky_cpu"] is False
    for index in range(1, total + 1):
        marker = json.loads(
            (
                cp_root / "cellA" / "tiles" / f"t{index:06d}" / "commit.json"
            ).read_text()
        )
        assert marker["backend"]["requested"] == "gpu"
        assert marker["backend"]["effective"] == ("cpu" if index == 1 else "gpu")

    # Third attempt: ALL tiles reused (mixed). Zero probe, zero reduction.
    state2 = {"probes": 0, "reduces": 0}

    def _no_reduce(*args, **kwargs):
        state2["reduces"] += 1
        raise AssertionError("all tiles should be reused")

    def _no_probe():
        state2["probes"] += 1
        raise AssertionError("all-reuse mixed Cell must not probe")

    monkeypatch.setattr(x, "reduce_spooled_tile", _no_reduce)
    out_c = {}
    got_c, _ = _run_executor(
        root, frames, masks, request=req,
        checkpoint=True, checkpoint_out=out_c, attempt_id="a3",
        gpu_probe=_no_probe,
    )
    assert state2 == {"probes": 0, "reduces": 0}
    assert out_c["reused_tiles"] == total
    assert out_c["gpu_tiles"] == total - 1
    assert out_c["cpu_tiles"] == 1
    assert out_c["effective_backend"] == "cpu"
    assert out_c["restored_sticky_cpu"] is True

    monkeypatch.setattr(x, "reduce_spooled_tile", real_reduce)
    monkeypatch.setattr(x, "_resolve_backend", lambda _token: "cpu")
    ref, _ = _run_executor(
        tmp_path / "cpu", frames, masks,
        request=_request(frames, masks, backend="cpu"), attempt_id="ref",
        gpu_probe=dict(_PROBE),
    )
    _assert_planes_equal(ref, got_b)
    _assert_planes_equal(ref, got_c)
