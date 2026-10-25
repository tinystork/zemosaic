"""ZM-ZEGRID-R52-OOC Lot B — bounded storage profile wiring.

Gates covered here (see mission ZM-ZEGRID-R52-OOC-STORAGE-LOT-B-20261025):

* B. Full ZeGrid Cell (per-Cell gauge + stack) through the BOUNDED profile is
  bit-identical to the DENSE profile on the same aligned cache, and really uses
  ``BoundedMemmapCanonicalProvider`` + a fresh ``BoundedCountingRejectionSink``.
* C. No dense ``(N,H,W,C)`` plane allocation in the bounded sink path and no
  persistent memmap mappings during the stack.
* D. Interior partition completeness: ``sink.written_cells == N*H*W*C``.
* E. Profile consistency: bounded plans/mappings carry the bounded model and the
  bounded path never runs the dense provider (and vice versa); in-memory is
  refused under bounded.

The heavy real-corpus Gate A replay lives in
``test_zegrid_r52_ooc_storage_lot_b_gate_a.py`` (slow marker).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import zemosaic.core.canonical_streaming as cs
from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import resource_envelope as renv
from zemosaic.core.zegrid import scheduler as zsched
from zemosaic.core.zegrid import science_adapter as zsa
from zemosaic.core.zegrid import storage_profile as zsp
from zemosaic.core.zegrid import streaming as zstream

CORE_PLANES = (
    "science_core",
    "estimator_weight_sum_core",
    "support_w1_core",
    "support_w2_core",
    "n_eff_support_core",
    "valid_mask_core",
    "surviving_sample_count_core",
)


# ---------------------------------------------------------------------------
# Synthetic aligned corpus + fake Cell/Patch/Mem
# ---------------------------------------------------------------------------

def _corpus(n=6, h=40, w=40, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0, 3, (h, w, 3)).astype(np.float32)
        f += np.array([50, 55, 60], np.float32) + np.float32(2.0 * i)
        f[0:2, :, :] = np.nan
        frames.append(f)
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[1][10:14, 10:14] = False
    masks[3][:, 20:] = False
    return frames, masks


def _write_cache(cache, n=6):
    frames, masks = _corpus(n=n)
    ids = [f"f{i}" for i in range(n)]
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    return frames, masks, ids


class _FakeCell:
    cell_id = "r0000c0000"
    row = 0
    col = 0


class _FakePatch:
    cell_id = "r0000c0000"
    patch_shape_hw = (40, 40)
    core_slice = SimpleNamespace(y0=0, y1=40, x0=0, x1=40)
    patch = SimpleNamespace(x0=0, y0=0, x1=40, y1=40, height=40, width=40)


class _FakeMem:
    def __init__(self, ids):
        self.patch_ids = list(ids)


def _install_cell_globals(monkeypatch, cache_dir, cfg):
    """Make ``_prepare_stack_cell`` use a PREBUILT cache (no reprojection)."""
    monkeypatch.setattr(zz, "_STACK_CONFIG", cfg)
    monkeypatch.setattr(zz, "_STACK_GAUGE", None)
    monkeypatch.setattr(zz, "_STACK_GLOBAL_FRAME_IDS", None)
    monkeypatch.setattr(
        zz, "_build_one_cell_cache",
        lambda *a, **k: zfp.load_cache_manifest(a[5]),
    )


def _run_cell(tmp_path, monkeypatch, profile, cache_name):
    cache = tmp_path / cache_name
    frames, masks, ids = _write_cache(cache)
    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
    _install_cell_globals(monkeypatch, cache, cfg)
    task = (_FakeCell(), _FakePatch(), _FakeMem(ids), str(cache), "stream", 123)
    if profile == zsp.STORAGE_PROFILE_BOUNDED:
        task = task + (profile,)
    prepared = zz._prepare_stack_cell(task)
    result = zz._consume_stack_cell(prepared)
    return prepared, result, frames, masks


# ---------------------------------------------------------------------------
# Gate B — wired gauge+stack bit parity + real bounded mechanisms
# ---------------------------------------------------------------------------

def test_bounded_gauge_and_stack_bit_identical(tmp_path, monkeypatch):
    dense_prep, dense_res, frames, masks = _run_cell(
        tmp_path, monkeypatch, "dense", "cache_dense"
    )
    bound_prep, bound_res, _f2, _m2 = _run_cell(
        tmp_path, monkeypatch, "bounded", "cache_bounded"
    )

    # The gauge half really used the bounded provider under the bounded profile.
    assert dense_prep["gauge_storage"]["provider"] == "MemmapCanonicalProvider"
    assert bound_prep["gauge_storage"]["provider"] == "BoundedMemmapCanonicalProvider"
    assert bound_prep["gauge_storage"]["storage_domain"] == "aligned-npy-memmap-bounded"
    assert bound_prep["gauge_storage"]["open_failures"] == 0
    assert bound_prep["gauge_storage"]["open_count"] > 0
    assert bound_prep["gauge_status"] == "ok"

    # The stack half used the bounded provider + a fresh bounded sink.
    rec = bound_res["record"]
    assert rec["storage_profile"] == "bounded"
    storage = rec["storage"]
    assert storage["provider"] == "BoundedMemmapCanonicalProvider"
    assert storage["sink"]["type"] == "BoundedCountingRejectionSink"
    assert storage["rejection_mask_available"] is False
    assert storage["open_failures"] == 0

    # Science/support/masks are bit-identical.
    for name in CORE_PLANES:
        np.testing.assert_array_equal(
            dense_res["core"][name], bound_res["core"][name], err_msg=name
        )

    # Gate D: complete, single interior partition.
    n, h, w, c = 6, 40, 40, 3
    assert storage["sink"]["written_cells"] == n * h * w * c
    assert storage["sink"]["writes"] > 0
    assert storage["sink"]["flush_count"] == storage["sink"]["writes"]

    # Rejection counts match the dense engine on the same aligned arrays.
    import zemosaic.core.canonical_streaming as csmod
    from zemosaic.core.canonical_streaming import InMemoryCanonicalProvider
    from zemosaic.core.zegrid import streaming as zst

    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
    req = zst.build_streaming_request(cfg, n)
    ref = csmod.run_canonical_stack_streaming(
        InMemoryCanonicalProvider(frames, masks), req, tile_size=zz.STREAM_TILE_SIZE
    )
    assert storage["sink"]["true_count"] == ref.rejected_sample_count
    assert int(ref.rejection_mask.sum()) == ref.rejected_sample_count


def test_bounded_sink_is_fresh_per_cell(tmp_path, monkeypatch):
    first, res1, _f, _m = _run_cell(tmp_path, monkeypatch, "bounded", "c1")
    second, res2, _f2, _m2 = _run_cell(tmp_path, monkeypatch, "bounded", "c2")
    s1 = res1["record"]["storage"]["sink"]
    s2 = res2["record"]["storage"]["sink"]
    # A reused/accumulating sink would report the SUM on the second cell.
    assert s1["written_cells"] == s2["written_cells"] == 6 * 40 * 40 * 3
    assert s1["true_count"] == s2["true_count"]
    assert first["storage_profile"] == second["storage_profile"] == "bounded"


# ---------------------------------------------------------------------------
# Gate C — no dense mask allocation, no persistent mappings
# ---------------------------------------------------------------------------

def test_bounded_path_allocates_no_dense_rejection_plane(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _write_cache(cache)
    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
    full_shape = (6, 40, 40, 3)

    calls = []
    real_zeros = np.zeros

    def probe(shape, *args, **kwargs):
        try:
            calls.append((tuple(int(v) for v in shape), np.dtype(kwargs.get("dtype", float))))
        except Exception:
            pass
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(cs.np, "zeros", probe)

    # Dense control: the full plane IS allocated.
    zz._run_cell_stream(cache, _FakePatch(), cfg, None, tile_size=16,
                        storage_profile="dense")
    assert any(shape == full_shape and dtype == np.dtype(bool) for shape, dtype in calls)

    # Bounded: no full plane.
    calls.clear()
    zz._run_cell_stream(cache, _FakePatch(), cfg, None, tile_size=16,
                        storage_profile="bounded")
    assert not any(shape == full_shape and dtype == np.dtype(bool) for shape, dtype in calls)


def test_bounded_never_holds_persistent_mappings(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _write_cache(cache)
    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")

    live = []
    live_open = []
    real_load = zfp._load_memmap

    def tracking(path):
        # Before a new file is opened, every previously opened mapping must
        # already be closed: only the RGB/support pair of the CURRENT access may
        # coexist. This is the exact opposite of the dense 2N-handle contract.
        live_open[:] = [a for a in live_open if not a._mmap.closed]
        assert len(live_open) <= 1, (
            f"{len(live_open)} mappings were still open when a new access began"
        )
        arr = real_load(path)
        live.append(arr)
        live_open.append(arr)
        return arr

    monkeypatch.setattr(zfp, "_load_memmap", tracking)
    info = {}
    zz._run_cell_stream(cache, _FakePatch(), cfg, None, tile_size=16,
                        storage_profile="bounded", storage_info=info)
    assert live, "the bounded path must open the cache per access"
    assert info["open_count"] == len(live)
    # Every mapping is closed when the provider returns; nothing survives the run.
    assert all(arr._mmap.closed for arr in live)


# ---------------------------------------------------------------------------
# Gate E — profile consistency / no cross-profile execution
# ---------------------------------------------------------------------------

def _ctx(profile):
    return zsched.PlanContext(
        phase=renv.PHASE_CELL_STACK_STREAM,
        backend=renv.BACKEND_CPU,
        budget_bytes=10**12,
        raw_available_bytes=2 * 10**12,
        factor_num=1,
        factor_den=2,
        cpu_logical=8,
        units_total=4,
        shared_bytes=340 * 2**20,
        per_unit_stack_bytes=10**9,
        per_unit_cache_bytes=10**8,
        per_unit_prefetch_bytes=10**8,
        disk_free_bytes=10**12,
        layout=(3, 3),
        refinement_rank=9,
        patch_hw=(1000, 1000),
        halo_px=128,
        channels=3,
        n_frames=50,
        tile_ladder=(128, 96, 64, 32),
        prefetch_ladder=(0,),
        workers_ladder=(1, 2),
        storage_profile=profile,
    )


def test_plan_mapping_carries_the_admitted_model():
    bounded = _ctx("bounded")
    cand = next(c for c in zsched.enumerate_candidates(bounded) if c.feasible)
    assert cand.storage_profile == "bounded"
    mapping = zsched.active_mapping_for_candidate(cand)
    assert mapping["storage"] == {
        "profile": "bounded",
        "model": zsp.STORAGE_MODEL_BOUNDED,
    }
    assert zsched.materialize_active_candidate(bounded, cand).storage_profile == "bounded"

    dense = _ctx("dense")
    dcand = next(c for c in zsched.enumerate_candidates(dense) if c.feasible)
    assert dcand.storage_profile == "dense"
    dmapping = zsched.active_mapping_for_candidate(dcand)
    assert "storage" not in dmapping  # dense plan/mapping bytes unchanged


def test_provider_selection_both_directions(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _write_cache(cache)
    seen = []
    real_make = zz._make_cell_provider

    def spy(cache_dir, profile):
        provider = real_make(cache_dir, profile)
        seen.append((zsp.normalize_storage_profile(profile), type(provider).__name__))
        return provider

    monkeypatch.setattr(zz, "_make_cell_provider", spy)
    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")

    zz._run_cell_stream(cache, _FakePatch(), cfg, None, tile_size=16,
                        storage_profile="dense")
    zz._run_cell_stream(cache, _FakePatch(), cfg, None, tile_size=16,
                        storage_profile="bounded")
    assert seen[-2] == ("dense", "MemmapCanonicalProvider")
    assert seen[-1] == ("bounded", "BoundedMemmapCanonicalProvider")

    with pytest.raises(RuntimeError, match="bounded"):
        zz._run_cell_inmem(cache, _FakePatch(), cfg, None, storage_profile="bounded")


def test_bounded_forces_stream_mode(monkeypatch):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")
    # A budget that would normally select the in-memory path.
    mode, bound = zz._pick_mode(4, 100 * 100, (100, 100), 10**15)
    assert mode == "stream"
    assert bound > 0


# ---------------------------------------------------------------------------
# Model binding — only the two truly eliminated terms change
# ---------------------------------------------------------------------------

def test_bounded_model_removes_only_mask_and_input_residency():
    n, h, w, c = 30, 300, 300, 3
    dense = zstream.estimate_streaming_peak_kib(n, (h, w), 64, c)
    bounded = zstream.estimate_streaming_peak_kib(
        n, (h, w), 64, c, storage_profile="bounded"
    )
    assert bounded < dense
    eliminated_bytes = n * h * w * c + (n - 2) * h * w * (4 * c + 1)
    # Same integer /1024 flooring on both sides: allow <= 2 KiB of rounding.
    assert abs((dense - bounded) * 1024 - eliminated_bytes) <= 2 * 1024
    # Unknown profile never silently falls back.
    with pytest.raises(ValueError, match="unknown storage_profile"):
        zstream.estimate_streaming_peak_kib(n, (h, w), 64, c, storage_profile="nope")
