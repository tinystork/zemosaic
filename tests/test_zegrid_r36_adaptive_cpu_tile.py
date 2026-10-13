"""ZM-ZEGRID-R36 — RAM-adaptive CPU streaming tile + joint-planner feasibility.

Covers the R36 deliverables WITHOUT a full M16/M106 pipeline run:

* the adaptive-layout planner selects the LARGEST CPU streaming tile from the
  frozen ladder ``(128,112,96,80,64,48,32)`` whose worst-cell streaming bound fits
  the RAM budget; a layout is admissible iff the in-memory bound fits OR some
  ladder tile fits; auto picks the first/coarsest admissible candidate; the public
  layout dict carries a ``cpu_streaming`` provenance block (never a bare
  ``tile_size``);
* pinned geometry stays EXACTLY ``(nx, ny)`` while its CPU tile adapts, and a
  pinned layout that fits no mode fails loud (minimum tile 32 + budget + the
  deeper disk-backed streaming next step);
* the joint mode/concurrency planner treats a mode whose worst-cell bound
  exceeds the budget as INFEASIBLE (never rescued by ``cells_in_flight == 1``),
  excludes it from the choice, never returns inmem when only stream fits, and
  raises ``ModeInfeasible`` when no mode fits (no-budget behaviour preserved);
* **plan == execution**: the tile selected by the planner is the EXACT tile the
  workers hand to ``run_canonical_stack_streaming`` (initializer/``_STACK_TILE_SIZE``
  seam + the executor call argument);
* the GPU tile policy is untouched, but every GPU->CPU degradation (pre-run VRAM
  infeasible OR a classified runtime error) uses the selected CPU tile, never a
  re-coded 128; the manifest distinguishes ``cpu_streaming_tile`` and
  ``gpu_streaming_tile``;
* the SCI-05 canonical executor is bit-exact across the planner ladder
  128/96/64/32 (science, support, weights, coverage, rejection, provenance).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack  # noqa: E402
from zemosaic.core.canonical_streaming import (  # noqa: E402
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402

TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
GB = 2 ** 30
MIB = 2 ** 20


# ---------------------------------------------------------------------------
# Synthetic corpora (mirrors the R28 helpers; no user data, no pixels)
# ---------------------------------------------------------------------------

def _tiny_floors():
    return zal.ScientificFloors(min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9)


def _make_tan_descriptor(i, h, w, scale, ra_step):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(), header_sha256="", instrument="",
    )


def _build_tan_corpus(n=4, h=40, w=40, scale=0.001, ra_step=0.004):
    return [_make_tan_descriptor(i, h, w, scale, ra_step) for i in range(n)]


class _FakeModel:
    """Test-local stand-in for ``zal.FITTED_MEMORY_MODEL`` (constant bound)."""

    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


def _patch_bounds(monkeypatch, stream_table, inmem_bytes):
    """Control BOTH bound sources deterministically (tile -> stream bytes)."""
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(inmem_bytes))
    biggest = max(stream_table.values()) if stream_table else 0

    def fake_est(n, hw, tile=zz.STREAM_TILE_SIZE, channels=3):
        key = int(tile) if tile is not None else int(zz.STREAM_TILE_SIZE)
        return int(stream_table.get(key, biggest))

    monkeypatch.setattr(zz, "_estimate_streaming_bytes", fake_est)


# ---------------------------------------------------------------------------
# 1. Layout planner — adaptive CPU streaming tile
# ---------------------------------------------------------------------------

def test_comfortable_budget_selects_preferred_tile(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {t: 100 for t in zz.CPU_STREAMING_TILE_LADDER}, 10 ** 12)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15), floors=_tiny_floors(), footprints_out=[],
    )
    cs = layout["cpu_streaming"]
    assert cs["adaptive"] is True
    assert cs["preferred_tile_size"] == 128
    assert cs["selected_tile_size"] == 128
    assert cs["minimum_tile_size"] == 32
    assert "128" in cs["selection_reason"]
    assert cs["trials"][0]["tile"] == 128 and cs["trials"][0]["fits"] is True
    # No bare ``tile_size`` name leaked into the public provenance block.
    assert "tile_size" not in cs


def test_128_and_96_fail_64_selected(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    table = {128: 900, 112: 900, 96: 900, 80: 900, 64: 300, 48: 500, 32: 700}
    _patch_bounds(monkeypatch, table, 10 ** 12)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=400, floors=_tiny_floors(), footprints_out=[],
    )
    cs = layout["cpu_streaming"]
    assert cs["selected_tile_size"] == 64
    assert cs["stream_bound_bytes"] == 300
    # Why 128 dropped must be explicit.
    assert "128" in cs["selection_reason"] and "64" in cs["selection_reason"]
    fits = {t["tile"]: t["fits"] for t in cs["trials"]}
    assert fits[128] is False and fits[96] is False and fits[64] is True


def test_inmem_only_layout_has_no_stream_tile(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {t: 10 ** 6 for t in zz.CPU_STREAMING_TILE_LADDER}, 10)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=1000, floors=_tiny_floors(), footprints_out=[],
    )
    cs = layout["cpu_streaming"]
    assert cs["selected_tile_size"] is None
    assert cs["inmem_bound_bytes"] == 10
    assert "in-memory" in cs["selection_reason"]


def test_no_mode_fits_raises_layout_infeasible(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {t: 10 ** 9 for t in zz.CPU_STREAMING_TILE_LADDER}, 10 ** 9)
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=1000, floors=_tiny_floors(), footprints_out=[],
        )
    msg = str(exc.value)
    assert "MiB" in msg
    assert "32" in msg  # minimum tile stated
    assert "grand" not in msg.lower()  # (sanity: no stray token)
    assert "zegrid_layout" in msg
    assert "disk-backed" in msg or "streaming" in msg


def test_pinned_geometry_unchanged_but_tile_adapts(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {128: 900, 112: 900, 96: 900, 80: 900, 64: 300, 48: 500, 32: 700}, 10 ** 12)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=400, floors=_tiny_floors(),
        pinned_layout=(3, 2), footprints_out=[],
    )
    assert (layout["nx"], layout["ny"]) == (3, 2)
    assert layout["layout_source"] == "pinned"
    assert layout["cpu_streaming"]["selected_tile_size"] == 64


def test_pinned_no_fit_fails_loud(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {t: 10 ** 9 for t in zz.CPU_STREAMING_TILE_LADDER}, 10 ** 9)
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=1000, floors=_tiny_floors(),
            pinned_layout=(3, 2), footprints_out=[],
        )
    assert "32" in str(exc.value)
    assert "zegrid_layout" in str(exc.value)


def test_layout_planner_deterministic(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {128: 900, 112: 900, 96: 900, 80: 900, 64: 300, 48: 500, 32: 700}, 10 ** 12)
    a = zz._choose_layout_mode_aware(canvas, descs, ram_budget=400, floors=_tiny_floors())
    b = zz._choose_layout_mode_aware(canvas, descs, ram_budget=400, floors=_tiny_floors())
    assert a == b


# ---------------------------------------------------------------------------
# 2. Joint planner — infeasible-mode exclusion (no clamp-to-1 rescue)
# ---------------------------------------------------------------------------

def test_infeasible_mode_is_excluded_not_rescued_by_clamp():
    # both max bounds exceed the 2 GiB budget -> the old clamp-to-1 could have
    # reported a "serial" plan; R36 must raise.
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency([(3 * GB, 2 * GB)] * 3, 8, 1 * GB)


def test_inmem_infeasible_stream_feasible_never_returns_inmem():
    plan = zpar.plan_cell_concurrency([(10 * GB, GB // 2)] * 4, 16, 2 * GB)
    assert plan["mode"] == "stream"
    inmem = next(c for c in plan["candidates"] if c["mode"] == "inmem")
    stream = next(c for c in plan["candidates"] if c["mode"] == "stream")
    assert inmem["feasible"] is False and stream["feasible"] is True
    assert inmem["cells_in_flight"] == 0
    assert plan["cells_in_flight"] >= 1
    assert "INFEASIBLE" in plan["chosen_reason"]


def test_no_budget_preserves_behaviour():
    plan = zpar.plan_cell_concurrency([(10 * GB, 5 * GB)] * 4, 16, None)
    assert plan["mode"] in ("inmem", "stream")
    assert plan["cells_in_flight"] >= 1
    assert all(c["feasible"] for c in plan["candidates"])


def test_single_cell_over_budget_raises():
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency([(GB, GB)], 8, GB // 2)


# --- Rework-1 F1/F2/F3 direct ----------------

def test_f1_inmem_only_is_a_valid_plan_not_a_crash():
    """F1: in-memory fits + streaming does NOT -> a VALID inmem plan (no TypeError)."""
    plan = zpar.plan_cell_concurrency([(50, 150)], cpu=8, ram_budget_bytes=100, gpu_backend=False)
    assert plan["mode"] == "inmem"
    assert plan["cells_in_flight"] == 1
    cands = {c["mode"]: c for c in plan["candidates"]}
    assert cands["stream"]["feasible"] is False
    assert cands["stream"]["cells_in_flight"] == 0
    assert cands["stream"]["infeasible_reason"]
    assert cands["inmem"]["feasible"] is True
    assert "INFEASIBLE" in plan["chosen_reason"]


def test_f2_gpu_over_budget_refuses_to_pretend():
    """F2: GPU requested + stream RAM-infeasible -> explicit ModeInfeasible (audited)."""
    with pytest.raises(zpar.ModeInfeasible) as exc:
        zpar.plan_cell_concurrency([(50, 150)], cpu=8, ram_budget_bytes=100, gpu_backend=True)
    assert "RAM" in str(exc.value)
    cands = {c["mode"]: c for c in getattr(exc.value, "candidates", [])}
    assert cands["stream"]["feasible"] is False
    assert cands["stream"]["infeasible_reason"]


def test_f2_gpu_feasible_candidates_are_coherent():
    plan = zpar.plan_cell_concurrency([(50, 150)], cpu=8, ram_budget_bytes=200, gpu_backend=True)
    assert plan["mode"] == "stream" and plan["feasible"] is True
    cands = {c["mode"]: c for c in plan["candidates"]}
    assert cands["stream"]["feasible"] is True
    assert cands["inmem"]["feasible"] is False
    # No-budget keeps the historical unconditional GPU behaviour.
    p0 = zpar.plan_cell_concurrency([(50, 150)], cpu=8, ram_budget_bytes=None, gpu_backend=True)
    assert p0["mode"] == "stream" and p0["feasible"] is True


def test_f3_predicted_bound_is_the_selected_admissible_bound(monkeypatch):
    """F3: an ACCEPTED layout reports the SELECTED bound (<= budget), and the
    legacy preferred-tile (128) bound is kept separately."""
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _patch_bounds(monkeypatch, {128: 900, 112: 900, 96: 900, 80: 900, 64: 300, 48: 500, 32: 700}, 10 ** 12)
    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=400, floors=_tiny_floors(), footprints_out=[],
    )
    assert layout["cpu_streaming"]["selected_tile_size"] == 64
    assert layout["predicted_bound_bytes"] == 300, layout["predicted_bound_bytes"]
    assert layout["predicted_bound_bytes"] <= 400
    assert layout["cpu_streaming"]["preferred_stream_bound_bytes"] == 900


# ---------------------------------------------------------------------------
# 3. SCI-05 canonical executor — bit-exact across the planner ladder
# ---------------------------------------------------------------------------

def _ladder_corpus(n=6, h=200, w=200, seed=5):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(5):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        base += 40.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
    gains = np.linspace(0.96, 1.04, n)
    frames = [np.asarray(base * gains[i] + rng.normal(0, 2.0, (h, w)), dtype=np.float32) for i in range(n)]
    frames[1][10:20, 10:20] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[4][:, w // 2:] = False
    return frames, masks


def _assert_planes_bit_exact(a, b):
    for name in ("science", "estimator_weight_sum", "support_w1", "support_w2",
                 "n_eff_support", "valid_mask", "surviving_sample_count", "rejection_mask"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.iterations_used == b.iterations_used
    assert a.initial_sample_count == b.initial_sample_count
    assert a.rejected_sample_count == b.rejected_sample_count
    assert a.rejected_fraction == b.rejected_fraction
    assert a.low_n_cell_count == b.low_n_cell_count
    assert a.degenerate_cell_count == b.degenerate_cell_count
    assert a.rejection_method == b.rejection_method
    assert a.provenance == b.provenance


@pytest.mark.parametrize("tile", (128, 96, 64, 32))
def test_sci05_ladder_bit_exact(tile):
    frames, masks = _ladder_corpus()
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks, normalization="linear_fit",
        weighting="noise_variance", rejection="kappa_sigma", combine="mean",
        taper="footprint", taper_px=8.0,
    )
    ref = run_canonical_stack(req)
    st = run_canonical_stack_streaming(InMemoryCanonicalProvider(frames, masks), req, tile_size=tile)
    _assert_planes_bit_exact(ref, st)


def test_sci05_ladder_all_tiles_agree():
    frames, masks = _ladder_corpus(n=5, h=180, w=160)
    req = CanonicalStackRequest(
        images=frames, geometric_support=masks, normalization="sky_mean",
        weighting="none", rejection="winsorized_sigma_clip", combine="median",
        taper="footprint", taper_px=8.0,
    )
    ref = run_canonical_stack(req)
    for tile in (128, 96, 64, 32):
        st = run_canonical_stack_streaming(InMemoryCanonicalProvider(frames, masks), req, tile_size=tile)
        _assert_planes_bit_exact(ref, st)


# ---------------------------------------------------------------------------
# 4. plan == execution (routine corpus, serial seam)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def routine_corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("r36_routine_corpus")
    proc = subprocess.run([sys.executable, str(TOOL), str(out)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return out


def _prepare_input(input_dir, routine_corpus):
    input_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(routine_corpus.glob("*.fits")):
        (input_dir / p.name).write_bytes(p.read_bytes())
    (input_dir / "stack_plan.csv").write_text(
        (routine_corpus / "stack_plan.csv").read_text(), encoding="utf-8"
    )


def _force_cpu_tile_64(monkeypatch):
    """Make the planner select CPU tile 64 and force inmem-infeasible/stream."""
    budget = 8192
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: int(budget / zpar.RAM_SAFETY_FRACTION))
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))  # inmem fails
    table = {128: budget * 4, 112: budget * 4, 96: budget * 2, 80: budget * 2,
             64: budget // 2, 48: budget // 3, 32: budget // 4}
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(
                            table.get(int(tile) if tile is not None else 128, budget * 4)))
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)  # serial seam


def test_plan_equals_execution_tile_64(tmp_path, routine_corpus, monkeypatch):
    _force_cpu_tile_64(monkeypatch)
    seen = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        seen.append(int(tile_size) if tile_size is not None else None)
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out))

    manifest = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cpu_streaming"]["selected_tile_size"] == 64
    assert manifest["per_cell_diagnostics"]["cpu_streaming_tile"] == 64
    # The initializer seam published the SAME tile.
    assert zz._STACK_TILE_SIZE == 64
    # The EXACT tile reached the executor (planner64 == execution64).
    assert seen and all(t == 64 for t in seen), seen


def test_manifest_cpu_streaming_block_contract(tmp_path, routine_corpus, monkeypatch):
    _force_cpu_tile_64(monkeypatch)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out))
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    block = m["cpu_streaming"]
    for key in ("adaptive", "preferred_tile_size", "selected_tile_size",
                "minimum_tile_size", "selection_reason", "predicted_peak_bytes",
                "ram_budget_bytes"):
        assert key in block, key
    assert block["preferred_tile_size"] == 128
    assert block["selected_tile_size"] == 64
    assert block["minimum_tile_size"] == 32
    # No BARE ``tile_size`` key in the public provenance block (R36).
    assert "tile_size" not in block
    assert "selected_tile_size" in block and "minimum_tile_size" in block
    run_log = (out / "zegrid_run.log").read_text(encoding="utf-8")
    assert "CPU streaming tile" in run_log


# ---------------------------------------------------------------------------
# 5. GPU tile policy untouched; CPU fallback uses the selected CPU tile
# ---------------------------------------------------------------------------

_FakeCupyOOM = type("OutOfMemoryError", (Exception,), {"__module__": "cupy.cuda.memory"})


def _force_gpu_selection(monkeypatch, gpu_tile=128):
    from zemosaic.core.zegrid import gpu as zgpu
    monkeypatch.setattr(zgpu, "probe_gpu_backend", lambda: {
        "available": True, "device": "FakeGPU", "cupy_version": "fake",
        "vram_total_bytes": 8 * GB, "vram_free_bytes": int(7.5 * GB), "reason": None,
    })
    monkeypatch.setattr(zgpu, "vram_budget_bytes", lambda probe: 4 * GB)
    monkeypatch.setattr(zgpu, "choose_gpu_tile_size", lambda *a, **k: gpu_tile)


def _force_gpu_runtime_tiles(monkeypatch):
    """CPU tile 64; EFFECTIVE GPU tile 48 (< CPU tile, both host-RAM-feasible)."""
    budget = 8192
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: int(budget / zpar.RAM_SAFETY_FRACTION))
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))  # inmem fails
    table = {128: 100000, 112: 100000, 96: 100000, 80: 100000,
             64: 4096, 48: 1000, 32: 500}
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(
                            table.get(int(tile) if tile is not None else 128, 100000)))
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)
    _force_gpu_selection(monkeypatch, gpu_tile=48)


def test_gpu_runtime_fallback_uses_selected_cpu_tile(tmp_path, routine_corpus, monkeypatch):
    """A valid GPU plan (effective GPU tile 48) + runtime GPU OOM -> CPU rerun at the
    SELECTED CPU tile 64 (never a re-coded 128, never the GPU tile)."""
    _force_gpu_runtime_tiles(monkeypatch)
    calls = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backend = getattr(request, "backend", "cpu")
        calls.append((backend, int(tile_size) if tile_size is not None else None))
        if backend == "gpu":
            raise _FakeCupyOOM("simulated runtime OOM")
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)

    # GPU was attempted at its EFFECTIVE tile (48), not 128 and not the CPU 64.
    assert any(b == "gpu" and t == 48 for b, t in calls), calls
    # The CPU rerun used the selected CPU tile 64.
    cpu_tiles = {t for b, t in calls if b == "cpu"}
    assert cpu_tiles == {64}, calls
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert m["gpu"]["cpu_streaming_tile"] == 64
    assert m["gpu"]["gpu_streaming_tile"] == 48
    assert m["gpu"]["runtime_fallback"] is not None


def test_gpu_vram_infeasible_uses_selected_cpu_tile(tmp_path, routine_corpus, monkeypatch):
    _force_cpu_tile_64(monkeypatch)
    _force_gpu_selection(monkeypatch)
    from zemosaic.core.zegrid import gpu as zgpu
    monkeypatch.setattr(zgpu, "choose_gpu_tile_size", lambda *a, **k: None)
    tiles = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        tiles.append(int(tile_size) if tile_size is not None else None)
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)
    assert tiles and all(t == 64 for t in tiles), tiles
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert m["gpu"]["used"] is False
    assert m["gpu"]["cpu_streaming_tile"] == 64
    assert m["gpu"]["gpu_streaming_tile"] is None


def _force_gpu_stream_ram_infeasible_inmem_ok(monkeypatch):
    """GPU stream never fits the RAM budget; in-memory does."""
    budget = 8192
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: int(budget / zpar.RAM_SAFETY_FRACTION))
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(budget // 4))  # inmem fits
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: 10 ** 9)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)


def test_f2_gpu_ram_infeasible_degrades_to_cpu_inmem_before_execution(tmp_path, routine_corpus, monkeypatch):
    """F2 product path: GPU requested but its stream is RAM-infeasible -> loud CPU
    inmem fallback BEFORE any GPU execution; manifest honest; no GPU attempt."""
    _force_gpu_stream_ram_infeasible_inmem_ok(monkeypatch)
    _force_gpu_selection(monkeypatch)
    backends = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backends.append(getattr(request, "backend", "cpu"))
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)

    assert "gpu" not in backends, backends  # NO GPU executor attempt
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert m["gpu"]["used"] is False
    assert m["gpu"]["final_backend"] == "cpu"
    assert m["gpu"]["fallback_reason"] == "gpu_stream_host_ram_infeasible"
    assert m["gpu"]["pre_execution_fallback"]["reason"] == "gpu_stream_host_ram_infeasible"
    assert m["gpu"]["gpu_streaming_tile"] is None
    assert m["per_cell_diagnostics"]["mode"] == "inmem"


def test_f2_both_infeasible_fails_loud_no_gpu_attempt(tmp_path, routine_corpus, monkeypatch):
    budget = 8192
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: int(budget / zpar.RAM_SAFETY_FRACTION))
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 10 ** 9)
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)
    _force_gpu_selection(monkeypatch)
    backends = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backends.append(getattr(request, "backend", "cpu"))
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    with pytest.raises(zal.LayoutInfeasible):
        zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)
    assert backends == [], backends


# --- rework-2 F-M1: GPU gate at the GPU's OWN tile -------------------------------

def test_fm1_gpu_tile128_host_infeasible_refuses_gpu_uses_cpu64(tmp_path, routine_corpus, monkeypatch):
    """F-M1: CPU tile 64 fits host RAM but the GPU's OWN tile 128 does NOT.

    The GPU must be refused BEFORE execution (zero GPU executor call), the CPU is
    re-planned and EXECUTED at the selected CPU tile 64, and the manifest records
    the exact GPU tile/bound + reason.
    """
    _force_cpu_tile_64(monkeypatch)              # CPU tile 64; stream@128 infeasible
    _force_gpu_selection(monkeypatch, gpu_tile=128)
    calls = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        calls.append((getattr(request, "backend", "cpu"),
                      int(tile_size) if tile_size is not None else None))
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)

    assert "gpu" not in {b for b, _t in calls}, calls      # zero GPU executor call
    assert {t for b, t in calls if b == "cpu"} == {64}, calls
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert m["gpu"]["used"] is False
    assert m["gpu"]["final_backend"] == "cpu"
    assert m["gpu"]["attempted"] is False
    assert m["gpu"]["fallback_reason"] == "gpu_stream_host_ram_infeasible"
    pef = m["gpu"]["pre_execution_fallback"]
    assert pef["reason"] == "gpu_stream_host_ram_infeasible"
    assert pef["gpu_streaming_tile"] == 128
    assert pef["gpu_streaming_host_peak_bytes"] == 8192 * 4
    assert pef["ram_budget_bytes"] == 8192
    assert m["gpu"]["gpu_streaming_tile"] is None
    assert m["per_cell_diagnostics"]["cpu_streaming_tile"] == 64


def test_fm1_gpu_effective_tile_fits_is_attempted_single_owner(tmp_path, routine_corpus, monkeypatch):
    """F-M1: the GPU's OWN tile (48) host bound fits -> GPU IS attempted at 48
    (single owner), and a runtime OOM then reruns CPU at the selected tile 64."""
    _force_gpu_runtime_tiles(monkeypatch)   # CPU tile 64, GPU tile 48
    calls = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backend = getattr(request, "backend", "cpu")
        calls.append((backend, int(tile_size) if tile_size is not None else None))
        if backend == "gpu":
            raise _FakeCupyOOM("simulated runtime OOM")
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)

    assert any(b == "gpu" and t == 48 for b, t in calls), calls
    # The gate used the GPU tile 48 bound (1000), NOT the 128 bound (100000).
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert m["gpu"]["gpu_streaming_tile"] == 48
    assert m["gpu"]["gpu_streaming_host_peak_bytes"] == 1000
    assert m["per_cell_diagnostics"]["gpu_streaming_tile"] == 48
    assert m["per_cell_diagnostics"]["gpu_streaming_host_peak_bytes"] == 1000
    assert m["per_cell_diagnostics"]["gpu_serialized"] is True
    assert m["per_cell_diagnostics"]["cells_in_flight"] == 1      # single-owner
    assert m["gpu"]["pre_execution_fallback"] is None


# --- rework-3 F-M2: CPU fallback revalidated at the CURRENT per-cell budget ---------

def _force_asymmetric_budget(monkeypatch, *, available=10240, inmem=10 ** 12, table=None,
                             gpu_tile=48):
    """Model the F-M2 asymmetry: the LAYOUT budget is `available` while the per-cell
    budget is `0.8*available` (RAM can shrink between layout and per-cell time)."""
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: available)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(inmem))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(
                            (table or {}).get(int(tile) if tile is not None else 128, 10 ** 9)))
    monkeypatch.setattr(zpar, "cells_in_flight", lambda *a, **k: 1)
    _force_gpu_selection(monkeypatch, gpu_tile=gpu_tile)


def test_fm2_a_no_cpu_fallback_fails_loud_zero_cpu_attempt(tmp_path, routine_corpus, monkeypatch):
    """F-M2 (A): CPU tile 64 fits the LAYOUT budget but NOT the current per-cell
    budget and inmem does not fit -> NO CPU fallback; GPU (tile 48) is attempted,
    a GPU OOM must FAIL LOUD with ModeInfeasible and ZERO CPU executor call."""
    # Layout budget 10240 -> tile 64 (bound 10000); per-cell budget 8192 -> CPU fails.
    _force_asymmetric_budget(
        monkeypatch, available=10240, inmem=10 ** 12,
        table={128: 100000, 112: 100000, 96: 100000, 80: 100000,
               64: 10000, 48: 1000, 32: 500},
        gpu_tile=48,
    )
    calls = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backend = getattr(request, "backend", "cpu")
        calls.append((backend, int(tile_size) if tile_size is not None else None))
        if backend == "gpu":
            raise _FakeCupyOOM("simulated runtime OOM")
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    msgs = []

    def cb(msg, prog, lvl, **kw):
        msgs.append((msg, lvl))

    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    with pytest.raises(zpar.ModeInfeasible):
        zz.run_zegrid_mode(str(inp), str(out), use_gpu=True, progress_callback=cb)

    # GPU WAS attempted at its effective tile 48; ZERO CPU executor call.
    assert any(b == "gpu" and t == 48 for b, t in calls), calls
    assert not any(b == "cpu" for b, _t in calls), calls
    # A loud explanation was emitted (run fails, so no manifest is expected).
    assert any("refusing an over-budget CPU" in m for m, _l in msgs), msgs[-5:]


def test_fm2_b_cpu_fallback_inmem_is_honoured(tmp_path, routine_corpus, monkeypatch):
    """F-M2 (B): same asymmetry but inmem FITS the current budget -> the CPU fallback
    plan chooses inmem, and after the GPU OOM the rerun actually runs inmem (never an
    over-budget stream tile)."""
    _force_asymmetric_budget(
        monkeypatch, available=10240, inmem=4000,
        table={128: 100000, 112: 100000, 96: 100000, 80: 100000,
               64: 10000, 48: 1000, 32: 500},
        gpu_tile=48,
    )
    calls = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backend = getattr(request, "backend", "cpu")
        calls.append((backend, int(tile_size) if tile_size is not None else None))
        if backend == "gpu":
            raise _FakeCupyOOM("simulated runtime OOM")
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)

    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    fb = m["per_cell_diagnostics"]["cpu_fallback_plan"]
    assert fb["applicable"] is True and fb["feasible"] is True
    assert fb["mode"] == "inmem"
    assert fb["used"] is True
    # The enacted CPU mode/concurrency is honest (inmem, not stream64).
    assert m["per_cell_diagnostics"]["mode"] == "inmem"
    assert m["per_cell_diagnostics"]["cpu_fallback_used"] is True
    assert all(r["mode"] == "inmem" for r in m["cells"] if r.get("status") == "complete")
    # The CPU rerun used the inmem path (tile None for CPU inmem).
    cpu_tiles = [t for b, t in calls if b == "cpu"]
    assert cpu_tiles and all(t is None for t in cpu_tiles), calls


def test_fm2_d_gpu_success_with_cpu_fallback_unavailable(tmp_path, routine_corpus, monkeypatch):
    """F-M2 (D): GPU success is ALLOWED even when no CPU fallback fits the current
    budget; the manifest states the CPU fallback is unavailable (no fake failure)."""
    import dataclasses
    _force_asymmetric_budget(
        monkeypatch, available=10240, inmem=10 ** 12,
        table={128: 100000, 112: 100000, 96: 100000, 80: 100000,
               64: 10000, 48: 1000, 32: 500},
        gpu_tile=48,
    )
    calls = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        backend = getattr(request, "backend", "cpu")
        calls.append((backend, int(tile_size) if tile_size is not None else None))
        if backend == "gpu":
            # Simulate a GPU SUCCESS: run the canonical on the swapped request.
            return real(provider, dataclasses.replace(request, backend="cpu"),
                        tile_size=tile_size, fixed=fixed)
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    inp, out = tmp_path / "in", tmp_path / "out"
    _prepare_input(inp, routine_corpus)
    zz.run_zegrid_mode(str(inp), str(out), use_gpu=True)

    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    assert any(b == "gpu" for b, _t in calls), calls
    assert m["gpu"]["used"] is True
    assert m["gpu"]["final_backend"] == "gpu"
    assert m["gpu"]["runtime_fallback"] is None      # no fake failure
    assert m["per_cell_diagnostics"]["cpu_fallback_plan"]["feasible"] is False
    assert m["per_cell_diagnostics"]["cpu_fallback_used"] is False
