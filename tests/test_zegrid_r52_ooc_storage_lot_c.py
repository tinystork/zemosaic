"""ZM-ZEGRID-R52-OOC Lot C — Nono Lot B LOW hardening (focused tests).

Covers L-1 (the 1<<62 planner sentinel never appears as a published in-memory
bound; the real value is kept in `inmem_worst_actual`), L-2 (the dense cell
payload does not gain a `storage_profile` key), L-3 (the public entry point
documents the one-run-per-process / non-reentrant contract) and L-4 (dense gauge
storage diagnostics are explicit about unavailable counters).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import science_adapter as zsa

SENTINEL = 1 << 62


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


class _Cell:
    cell_id = "r0000c0000"
    row = 0
    col = 0


class _Patch:
    cell_id = "r0000c0000"
    patch_shape_hw = (40, 40)
    core_slice = SimpleNamespace(y0=0, y1=40, x0=0, x1=40)
    patch = SimpleNamespace(x0=0, y0=0, x1=40, y1=40, height=40, width=40)


class _Mem:
    def __init__(self, ids):
        self.patch_ids = list(ids)


def _prepared(tmp_path, monkeypatch, profile, name):
    cache = tmp_path / name
    frames, masks = _corpus()
    ids = [f"f{i}" for i in range(6)]
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
    monkeypatch.setattr(zz, "_STACK_CONFIG", cfg)
    monkeypatch.setattr(zz, "_STACK_GAUGE", None)
    monkeypatch.setattr(zz, "_STACK_GLOBAL_FRAME_IDS", None)
    monkeypatch.setattr(
        zz, "_build_one_cell_cache",
        lambda *a, **k: zfp.load_cache_manifest(a[5]),
    )
    task = (_Cell(), _Patch(), _Mem(ids), str(cache), "stream", 1)
    if profile == "bounded":
        task = task + (profile,)
    return zz._prepare_stack_cell(task)


def test_dense_cell_payload_has_no_storage_profile_key(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch, "dense", "cache_dense")
    result = zz._consume_stack_cell(prepared)
    assert "storage_profile" not in prepared
    assert "storage_profile" not in result
    assert "storage_profile" not in result["record"]
    assert "storage" not in result["record"]


def test_bounded_cell_payload_keeps_storage_keys(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path, monkeypatch, "bounded", "cache_bounded")
    result = zz._consume_stack_cell(prepared)
    assert prepared["storage_profile"] == "bounded"
    assert result["storage_profile"] == "bounded"
    assert result["record"]["storage_profile"] == "bounded"
    assert result["record"]["storage"]["provider"] == "BoundedMemmapCanonicalProvider"


def test_non_reentrant_contract_is_documented():
    doc = inspect.getdoc(zz.run_zegrid_mode) or ""
    assert "ONE RUN PER PROCESS" in doc
    assert "NOT re-entrant" in doc or "not re-entrant" in doc
    assert "_STACK_STORAGE_PROFILE" in doc


def test_dense_gauge_diagnostics_are_explicit(tmp_path, monkeypatch):
    dense = _prepared(tmp_path, monkeypatch, "dense", "cache_dense")
    bounded = _prepared(tmp_path, monkeypatch, "bounded", "cache_bounded")
    dg = dense["gauge_storage"]
    bg = bounded["gauge_storage"]
    assert dg["counters_available"] is False
    assert dg["storage_domain"] == "aligned-npy-memmap-dense"
    assert dg["bytes_read"] is None and dg["open_count"] is None
    assert bg["counters_available"] is True
    assert bg["storage_domain"] == "aligned-npy-memmap-bounded"
    assert bg["open_count"] > 0


def test_bounded_facts_keep_real_inmem_bound(monkeypatch, tmp_path):
    cell = SimpleNamespace(
        cell_id="c0", row=0, col=0,
        core=SimpleNamespace(x0=0, y0=0, x1=100, y1=100),
    )
    patch = SimpleNamespace(
        patch=SimpleNamespace(x0=0, y0=0, x1=100, y1=100, width=100, height=100),
        patch_shape_hw=(100, 100),
    )
    mem = SimpleNamespace(patch_ids=["a", "b", "c"])
    bundle = SimpleNamespace(
        envelope=SimpleNamespace(
            budget_bytes=8 * 2**30, raw_available_bytes=10 * 2**30,
            factor_num=4, factor_den=5, refinement_rank=1, pinned=False,
        ),
        shared_bytes=340 * 2**20,
        layout={"nx": 1, "ny": 1},
        gauge_workers=1,
        cache_build_workers=1,
    )
    canvas = SimpleNamespace(width=100, height=100, canvas_id="x")
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "bounded")

    facts = zz._r55_active_facts(
        bundle=bundle,
        cell_ctxs=[(0, 0, cell, patch, mem)],
        canvas=canvas,
        halo_px=128,
        cache_root=tmp_path,
        output_dir=tmp_path,
        frame_ids=("a", "b", "c"),
    )
    real = int(zal.FITTED_MEMORY_MODEL.predict_bound_bytes(3, 100 * 100))
    assert facts["inmem_worst_actual"] == max(1, real)
    # L-1: bounded forbids the in-memory stack in the SCHEDULER, so the facts
    # carry the real bound everywhere and no ~4.6 EiB sentinel exists at all.
    assert facts["inmem_worst"] == max(1, real)
    assert facts["inmem_worst"] < 1 << 40
    assert SENTINEL not in facts.values()


def test_bounded_scheduler_forbids_inmem_with_real_bound():
    from zemosaic.core.zegrid import resource_envelope as renv
    from zemosaic.core.zegrid import scheduler as zsched

    base = dict(
        phase=renv.PHASE_CELL_STACK_INMEM,
        backend=renv.BACKEND_CPU,
        budget_bytes=10**12,
        raw_available_bytes=2 * 10**12,
        factor_num=1,
        factor_den=2,
        cpu_logical=8,
        units_total=4,
        shared_bytes=340 * 2**20,
        per_unit_stack_bytes=10**9,
        per_unit_cache_bytes=0,
        per_unit_prefetch_bytes=0,
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
        storage_profile="bounded",
    )
    cands = zsched.enumerate_candidates(zsched.PlanContext(**base))
    assert cands and all(c.tile_px is None for c in cands)
    assert all(not c.feasible for c in cands)
    assert all(c.host_peak_bytes < 1 << 40 for c in cands)
    assert all("bounded" in (c.infeasible_reason or "") for c in cands)
    # Dense in-memory remains admissible under the same comfortable budget.
    base["storage_profile"] = "dense"
    dense = zsched.enumerate_candidates(zsched.PlanContext(**base))
    assert any(c.feasible for c in dense)
