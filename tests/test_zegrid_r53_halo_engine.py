"""ZM-ZEGRID-R53a — engine/config foundation for the anti-checkerboard halo.

Covers the bounded lot-1 contract on the REAL production seams:

* strict resolver (six allowed values, absent -> 64, explicit precedence,
  bool/float/unknown rejection, migration string contract);
* config defaults (product + Qt fallback) carry 64;
* the validated effective H reaches the R41 planner, the real Cell patches /
  membership, the worker task payload and the feathered assembly;
* default 64 changes real patch geometry vs explicit 8, while explicit 8 keeps
  the pre-R53 geometry/assembly;
* bounded replans / CPU fallback preserve H; an inadmissible large H fails
  loudly (never clamped);
* global / per_cell / auto semantics, including the explicit global log.

Heavy science is mocked only where ordering/payloads are asserted; the planner,
geometry and assembly math are the real ones.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic import zegrid_halo as zhalo  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import background_diagnostics as bgd  # noqa: E402
from zemosaic.core.zegrid import cell_feather as zcf  # noqa: E402
from zemosaic.core.zegrid import file_provider as zfp  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import mosaic as zmosaic  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rc  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402

MIB = 2 ** 20
GB = 2 ** 30
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"

_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(**_kw):
    return _REAL_FLOORS(min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9)


class _AreaModel:
    """Bound grows with the (halo'd) patch area."""

    def __init__(self, bytes_per_px):
        self._bpp = float(bytes_per_px)

    def predict_bound_bytes(self, n, area):
        return int(max(1, area) * self._bpp)

    def predict_peak_bytes(self, n, area):
        return self.predict_bound_bytes(n, area)


def _make_tan(i, h=400, w=400, scale=0.001, ra_step=0.05):
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


def _corpus(n=5):
    return [_make_tan(i) for i in range(n)]


def _contexts(fps, canvas, nx, ny, halo):
    return zz._build_chosen_cell_contexts(fps, canvas, nx, ny, halo_px=halo)


# ---------------------------------------------------------------------------
# Resolver matrix / strict contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [8, 16, 32, 64, 128, 256])
def test_resolver_accepts_every_product_value(value):
    assert zhalo.resolve_zegrid_halo_px(value) == value
    assert zhalo.resolve_zegrid_halo_px(str(value)) == value
    assert zhalo.resolve_zegrid_halo_px(f" {value} ") == value


def test_resolver_absent_is_product_default_64():
    assert zhalo.ZEGRID_HALO_DEFAULT_PX == 64
    assert zhalo.resolve_zegrid_halo_px(None) == 64


@pytest.mark.parametrize(
    "value", [True, False, 64.0, 8.0, "64.0", "6e1", "+64", "", " ", "abc", 0, 7, 512]
)
def test_resolver_rejects_loudly(value):
    with pytest.raises(zhalo.ZeGridHaloError):
        zhalo.resolve_zegrid_halo_px(value)


def test_config_defaults_and_qt_fallback_carry_64():
    from zemosaic import zemosaic_config

    assert zemosaic_config.DEFAULT_CONFIG["zegrid_halo_px"] == 64

    src = (REPO_ROOT / "src" / "zemosaic" / "zemosaic_gui_qt.py").read_text(
        encoding="utf-8"
    )
    assert "from .zegrid_halo import ZEGRID_HALO_DEFAULT_PX" in src
    assert '"zegrid_halo_px": ZEGRID_HALO_DEFAULT_PX' in src
    assert '"zegrid_halo_px": 64' not in src


def test_resolution_precedence_is_explicit_arg_then_config_then_default():
    descs = _corpus(2)
    canvas = zg.build_canvas(descs)
    project = zal.project_footprints(descs, canvas)
    fps = [project]

    # Config 16, explicit 8 -> explicit wins; the product seam receives 8.
    seen = {}
    real = zz._build_chosen_cell_contexts

    def spy(*a, **k):
        seen["halo"] = k.get("halo_px")
        return real(*a, **k)

    # Direct seam precedence is exercised by run_zegrid_mode below; here verify
    # the explicit value really reaches the context builder.
    zz._build_chosen_cell_contexts(fps[0], canvas, 2, 2, halo_px=8)
    assert all(p.halo_px == 8 for (_r, _c, _ce, p, _m) in _contexts(fps[0], canvas, 2, 2, 8))

    # Engine-level precedence is logged before any expensive work.
    messages = []
    zconfig = SimpleNamespace(zegrid_halo_px=16)
    with pytest.raises(RuntimeError):
        zz.run_zegrid_mode(
            "/nonexistent-r53-input",
            "/nonexistent-r53-output",
            progress_callback=lambda msg, *a, **k: messages.append(str(msg)),
            zegrid_halo_px=8,
            zconfig=zconfig,
        )
    joined = "\n".join(messages)
    assert "effective=8 px" in joined
    assert "source=explicit argument" in joined


def test_run_zegrid_mode_absent_uses_config_then_default():
    for zconfig, expected, source in [
        (SimpleNamespace(zegrid_halo_px=32), 32, "zegrid_halo_px config"),
        (SimpleNamespace(), 64, "product default"),
    ]:
        messages = []
        with pytest.raises(RuntimeError):
            zz.run_zegrid_mode(
                "/nonexistent-r53-input",
                "/nonexistent-r53-output",
                progress_callback=lambda msg, *a, **k: messages.append(str(msg)),
                zconfig=zconfig,
            )
        joined = "\n".join(messages)
        assert f"effective={expected} px" in joined
        assert f"source={source}" in joined


def test_run_zegrid_mode_invalid_halo_fails_before_discovery(monkeypatch):
    calls = {"discovery": 0}

    def boom(*a, **k):
        calls["discovery"] += 1
        raise AssertionError("discovery must not run")

    from zemosaic import input_inventory as zinv

    monkeypatch.setattr(zinv, "build_zegrid_frames", boom)
    with pytest.raises(zhalo.ZeGridHaloError):
        zz.run_zegrid_mode("/x", "/y", zegrid_halo_px=64.0)
    assert calls["discovery"] == 0


# ---------------------------------------------------------------------------
# Real geometry / membership propagation
# ---------------------------------------------------------------------------

def test_effective_halo_reaches_real_patch_geometry_and_membership():
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    fps = zal.project_footprints(descs, canvas)
    nx = ny = 3
    ctx8 = _contexts(fps, canvas, nx, ny, 8)
    ctx64 = _contexts(fps, canvas, nx, ny, 64)

    # Real patch extension, not a config-only witness.
    assert all(p.halo_px == 8 for (_r, _c, _ce, p, _m) in ctx8)
    assert all(p.halo_px == 64 for (_r, _c, _ce, p, _m) in ctx64)
    interior = [
        (p8, p64)
        for ((r8, c8, _ce8, p8, _m8), (_r64, _c64, _ce64, p64, _m64))
        in zip(ctx8, ctx64)
        if 0 < r8 < ny - 1 and 0 < c8 < nx - 1
    ]
    assert interior, "fixture produced no interior cell"
    for p8, p64 in interior:
        assert p64.patch.width == p8.patch.width + 2 * (64 - 8)
        assert p64.patch.height == p8.patch.height + 2 * (64 - 8)

        def _core_rect(p):
            cs = p.core_slice
            return (
                p.patch.y0 + cs.y0, p.patch.y0 + cs.y1,
                p.patch.x0 + cs.x0, p.patch.x0 + cs.x1,
            )

        # The CORE rectangle is unchanged; only the patch extends further.
        assert _core_rect(p64) == _core_rect(p8)

    # Explicit 8 is exactly the pre-R53 helper default path.
    baseline = zsw.build_cell_contexts_from_footprints(fps, canvas, nx, ny)
    for (r, c, cell, patch, mem), (br, bc, bcell, bpatch, bmem) in zip(ctx8, baseline):
        assert (r, c) == (br, bc)
        assert cell.cell_id == bcell.cell_id
        assert patch.patch.x0 == bpatch.patch.x0
        assert patch.patch.x1 == bpatch.patch.x1
        assert patch.patch.y0 == bpatch.patch.y0
        assert patch.patch.y1 == bpatch.patch.y1
        assert list(mem.patch_ids) == list(bmem.patch_ids)
        assert list(mem.core_ids) == list(bmem.core_ids)


def test_default_64_changes_patch_geometry_relative_to_explicit_8():
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    fps = zal.project_footprints(descs, canvas)
    ctx8 = _contexts(fps, canvas, 3, 3, 8)
    ctx_default = _contexts(fps, canvas, 3, 3, zhalo.resolve_zegrid_halo_px(None))
    assert any(
        p8.patch.width != pd.patch.width or p8.patch.height != pd.patch.height
        for (_r, _c, _ce, p8, _m), (_r2, _c2, _ce2, pd, _m2) in zip(ctx8, ctx_default)
    )
    assert all(p.halo_px == 64 for (_r, _c, _ce, p, _m) in ctx_default)


# ---------------------------------------------------------------------------
# Planner / bundle / replans / fallback
# ---------------------------------------------------------------------------

def _bundle(canvas, descs, *, raw, budget, halo_px, **kw):
    return zz._build_execution_bundle(
        canvas, descs, None, raw_available_bytes=int(raw), budget_bytes=int(budget),
        cpu_count=3, cache_root=None, halo_px=int(halo_px), **kw,
    )


def test_bundle_threads_halo_to_layout_patches_tasks_and_cpu_fallback(monkeypatch):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(1024))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(max(1, int(hw[0]) * int(hw[1])) * 512))
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)

    b = _bundle(canvas, descs, raw=64 * GB, budget=rc.plan_budget_bytes(64 * GB),
                halo_px=32, pinned_layout=(3, 3), gpu_used=True,
                vram_budget_bytes=0)
    # GPU pre-execution fallback happened, but H is unchanged everywhere.
    assert b.gpu_used is False
    assert b.cell_ctxs and b.cell_tasks
    for (_r, _c, _cell, patch, _mem) in b.cell_ctxs:
        assert patch.halo_px == 32
    for task in b.cell_tasks:
        assert task[1].halo_px == 32
    if b.cpu_fallback_tasks:
        for task in b.cpu_fallback_tasks:
            assert task[1].halo_px == 32


def test_bounded_auto_replan_preserves_effective_halo(monkeypatch):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(1024))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(max(1, int(hw[0]) * int(hw[1])) * 1024))
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)

    coarsest = _bundle(canvas, descs, raw=64 * GB, budget=rc.plan_budget_bytes(64 * GB),
                       halo_px=64, max_cells_in_flight=1)
    rank0 = int(coarsest.envelope.refinement_rank)
    raw = int(coarsest.envelope.aggregate_bytes)
    refined = _bundle(canvas, descs, raw=raw, budget=rc.plan_budget_bytes(raw),
                      halo_px=64, max_cells_in_flight=1)
    assert int(refined.envelope.refinement_rank) >= rank0
    for (_r, _c, _cell, patch, _mem) in refined.cell_ctxs:
        assert patch.halo_px == 64


def test_large_halo_infeasible_fails_loudly_never_clamped(monkeypatch):
    descs = _corpus(2)
    canvas = zg.build_canvas(descs)

    def floors(**_kw):
        return _REAL_FLOORS(
            min_patch_area_px=1, min_contributors=1, max_halo_overhead=0.5
        )

    monkeypatch.setattr(zal, "ScientificFloors", floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(1))
    # 2x2 on this canvas: H=256 pushes halo_overhead above the 0.50 floor.
    with pytest.raises(zal.LayoutInfeasible) as exc:
        _bundle(canvas, descs, raw=64 * GB, budget=rc.plan_budget_bytes(64 * GB),
                halo_px=256, pinned_layout=(2, 2))
    assert "halo_overhead" in str(exc.value)
    assert "limit 0.5" in str(exc.value)
    # The same geometry at 8 px is admissible: no hidden clamp of the request.
    ok = _bundle(canvas, descs, raw=64 * GB, budget=rc.plan_budget_bytes(64 * GB),
                 halo_px=8, pinned_layout=(2, 2))
    assert all(p.halo_px == 8 for (_r, _c, _ce, p, _m) in ok.cell_ctxs)


# ---------------------------------------------------------------------------
# Assembly: feather width, diagnostic measurement band, H=8 parity
# ---------------------------------------------------------------------------

CORE = 32


def _make_patches(nx, ny, levels, *, core=CORE, halo=8, seed=0, sigma=0.2):
    rng = np.random.default_rng(seed)
    canvas = zg.GlobalCanvas(
        canvas_id=f"c{nx}x{ny}", wcs_header="", width=nx * core, height=ny * core,
        resolution_deg=0.001,
    )
    patches = {}
    for row in range(ny):
        for col in range(nx):
            cid = f"r{row:04d}c{col:04d}"
            cy0, cy1 = row * core, (row + 1) * core
            cx0, cx1 = col * core, (col + 1) * core
            py0 = max(0, cy0 - halo); py1 = min(canvas.height, cy1 + halo)
            px0 = max(0, cx0 - halo); px1 = min(canvas.width, cx1 + halo)
            h, w = py1 - py0, px1 - px0
            lv = float(levels[cid])
            sci = (lv + rng.normal(0.0, sigma, (h, w, 3))).astype(np.float32)
            valid = np.ones((h, w, 3), dtype=bool)
            core_slice = (cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0)
            patches[cid] = zcf.CellPatch(
                cell_id=cid, bounds=(py0, py1, px0, px1), core_slice=core_slice,
                science=sci, valid_mask=valid,
            )
    return canvas, patches


def _results(patches):
    out = {}
    for cid, cp in patches.items():
        out[cid] = {
            "cell_id": cid, "status": "complete",
            "patch": {
                "bounds": cp.bounds, "core_slice": cp.core_slice,
                "science": cp.science, "valid_mask": cp.valid_mask,
                "surviving": None,
            },
        }
    return out


def test_feather_assembly_and_capture_use_H_and_2H(tmp_path, monkeypatch):
    nx = ny = 2
    halo = 16
    canvas, patches = _make_patches(nx, ny, {f"r{r:04d}c{c:04d}": 10.0 + r + c
                                             for r in range(ny) for c in range(nx)},
                                     halo=halo)
    seen = {}
    real_build = bgd.build_payload
    real_assemble = zcf.assemble_canvas_feathered

    def build_spy(*a, **k):
        seen["measurement"] = k.get("measurement_px")
        seen["feather"] = k.get("feather_px")
        return real_build(*a, **k)

    def assemble_spy(*a, **k):
        seen["assembly_halo"] = k.get("halo_px")
        return real_assemble(*a, **k)

    monkeypatch.setattr(bgd, "build_payload", build_spy)
    monkeypatch.setattr(zcf, "assemble_canvas_feathered", assemble_spy)

    assembled, info = zz._assemble_per_cell_feathered(
        canvas, nx, ny, _results(patches), halo_px=halo, capture=True,
        output_dir=str(tmp_path), holdout_cap=4,
    )
    assert seen["measurement"] == 2 * halo
    assert seen["feather"] == halo
    assert seen["assembly_halo"] == halo
    assert info["halo_px"] == halo
    assert info["r37_diagnostic_capture"]["measurement_px"] == 2 * halo
    assert info["r37_diagnostic_capture"]["feather_px"] == halo
    assert assembled.coverage_pixels > 0


def test_explicit_8_preserves_pre_r53_assembly(monkeypatch):
    nx = ny = 2
    canvas, patches = _make_patches(nx, ny, {f"r{r:04d}c{c:04d}": 10.0 + r + c
                                             for r in range(ny) for c in range(nx)},
                                     halo=8)
    results = _results(patches)
    a8, info8 = zz._assemble_per_cell_feathered(canvas, nx, ny, results, halo_px=8)
    # Pre-R53 helper default (None) is exactly 8.
    a_default, info_default = zz._assemble_per_cell_feathered(
        canvas, nx, ny, results, halo_px=None
    )
    assert np.array_equal(a8.science, a_default.science)
    assert np.array_equal(a8.stack_depth, a_default.stack_depth)
    assert np.array_equal(a8.owner, a_default.owner)
    assert info8["halo_px"] == info_default["halo_px"] == 8


# ---------------------------------------------------------------------------
# Mode semantics
# ---------------------------------------------------------------------------

def test_global_mode_does_not_force_per_cell_and_logs_no_feather():
    nx = ny = 2
    canvas, patches = _make_patches(nx, ny, {f"r{r:04d}c{c:04d}": 10.0
                                             for r in range(ny) for c in range(nx)})
    # Global assembly needs the per-cell core plane dicts, not CellPatch objects.
    cores = {}
    for cid, cp in patches.items():
        cs = cp.core_slice
        sci = cp.science[cs[0]:cs[1], cs[2]:cs[3]]
        h, w = sci.shape[:2]
        cores[cid] = {
            "science_core": sci,
            "surviving_sample_count_core": np.ones((h, w, 3), dtype=np.int32),
            "n_eff_support_core": np.ones((h, w), dtype=np.float64),
            "support_w1_core": np.ones((h, w), dtype=np.float64),
            "support_w2_core": np.ones((h, w), dtype=np.float64),
            "valid_mask_core": cp.valid_mask[cs[0]:cs[1], cs[2]:cs[3]],
        }
    results = _results(patches)
    emitted = []
    assembled, feather_info = zz._assemble_cells(
        canvas, nx, ny, "global", cores, results, halo_px=32,
        emit=lambda m, lvl="INFO": emitted.append(m),
    )
    assert feather_info is None
    expected = zmosaic.assemble_canvas(canvas, nx, ny, cores)
    assert np.array_equal(assembled.science, expected.science)
    assert any("anti-checkerboard feather NOT applied" in m for m in emitted)


def test_per_cell_mode_applies_feather_at_effective_H():
    nx = ny = 2
    halo = 32
    canvas, patches = _make_patches(nx, ny, {f"r{r:04d}c{c:04d}": 10.0
                                             for r in range(ny) for c in range(nx)},
                                     halo=halo)
    cores = {
        cid: cp.science[
            cp.core_slice[0]:cp.core_slice[1], cp.core_slice[2]:cp.core_slice[3]
        ]
        for cid, cp in patches.items()
    }
    _assembled, feather_info = zz._assemble_cells(
        canvas, nx, ny, "per_cell", cores, _results(patches), halo_px=halo,
    )
    assert feather_info is not None
    assert feather_info["halo_px"] == halo


# ---------------------------------------------------------------------------
# End-to-end product witness (real corpus, real workers)
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    subprocess.run([sys.executable, str(TOOL), str(corpus)], check=True,
                   capture_output=True, timeout=180)
    inp = tmp_path / "in"
    inp.mkdir()
    for p in sorted(corpus.glob("*.fits")):
        (inp / p.name).write_bytes(p.read_bytes())
    return inp, tmp_path / "out"


def test_product_path_propagates_H_to_planner_tasks_and_assembly(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    halo = 32
    seen = {"contexts": [], "assembly": [], "bundles": []}
    real_contexts = zz._build_chosen_cell_contexts
    real_assemble = zcf.assemble_canvas_feathered

    def contexts_spy(*a, **k):
        seen["contexts"].append(k.get("halo_px"))
        return real_contexts(*a, **k)

    def assembly_spy(*a, **k):
        seen["assembly"].append(k.get("halo_px"))
        return real_assemble(*a, **k)

    real_bundle = zz._build_execution_bundle

    def bundle_spy(*a, **k):
        b = real_bundle(*a, **k)
        seen["bundles"].append([
            int(p.halo_px) for (_r, _c, _ce, p, _m) in b.cell_ctxs
        ])
        return b

    monkeypatch.setattr(zz, "_build_chosen_cell_contexts", contexts_spy)
    monkeypatch.setattr(zcf, "assemble_canvas_feathered", assembly_spy)
    monkeypatch.setattr(zz, "_build_execution_bundle", bundle_spy)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(1024))

    messages = []
    zz.run_zegrid_mode(
        str(inp), str(out),
        progress_callback=lambda msg, *a, **k: messages.append(str(msg)),
        cell_gauge_mode="per_cell",
        zegrid_halo_px=halo,
        zconfig=SimpleNamespace(zegrid_layout="2x2"),
        r37_diagnostic_capture=True,
    )
    joined = "\n".join(messages)
    assert f"effective={halo} px" in joined
    assert seen["contexts"] and set(seen["contexts"]) == {halo}
    assert seen["bundles"] and all(set(hs) == {halo} for hs in seen["bundles"])
    assert seen["assembly"] == [halo]

    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    feather = m["cell_feather"]
    assert feather["enabled"] is True
    assert feather["halo_px"] == halo
    cap = feather["r37_diagnostic_capture"]
    assert cap["measurement_px"] == 2 * halo
    assert cap["feather_px"] == halo
    assert m["photometric_gauge"]["mode"] == "per_cell"


def test_auto_self_heals_to_per_cell_and_uses_H(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    shape = (400, 600)
    centers = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]
    rows = ["file_path,exposure"]
    for i, (ra, dec) in enumerate(centers):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra, dec]
        w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
        w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
        w.array_shape = shape
        rng = np.random.default_rng(i)
        data = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
        name = f"f{i}.fit"
        from astropy.io import fits
        fits.PrimaryHDU(data, header=w.to_header()).writeto(inp / name, overwrite=True)
        rows.append(f"{name},10.0")
    (inp / "stack_plan.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    out = tmp_path / "out"
    zz.run_zegrid_mode(
        str(inp), str(out),
        zegrid_halo_px=16,
        zconfig=SimpleNamespace(zegrid_layout="2x2"),
    )
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    pg = m["photometric_gauge"]
    assert pg["mode"] == "per_cell"
    assert pg["auto_selection"]["fallback_fired"] is True
    assert m["cell_feather"]["enabled"] is True
    assert m["cell_feather"]["halo_px"] == 16


# ---------------------------------------------------------------------------
# F-1 — cache geometry identity
# ---------------------------------------------------------------------------

def _write_aligned_cache(cache_dir, *, n=20, patch_identity=None, frame_ids=("f0",)):
    builder = zfp.AlignedCacheBuilder(cache_dir, patch_identity=patch_identity)
    rng = np.random.default_rng(1)
    for fid in frame_ids:
        rgb = rng.normal(0.0, 1.0, (n, n, 3)).astype(np.float32)
        sup = np.ones((n, n), dtype=bool)
        builder.add(fid, rgb, sup)
    return builder.finish()


def test_cache_is_complete_expected_hw_and_identity(tmp_path):
    ids = ["f0"]
    cache = tmp_path / "cache"
    _write_aligned_cache(cache, patch_identity="bounds|wcs")

    assert zfp.cache_is_complete(cache, ids) is True
    assert zfp.cache_is_complete(cache, ids, expected_hw=(20, 20)) is True
    for bad in [(21, 20), (20, 21), (20, 0), (0, 20), (20,), (20, 20, 3),
                "20x20", (20.0, 20.0), (True, 20)]:
        assert zfp.cache_is_complete(cache, ids, expected_hw=bad) is False, bad

    assert zfp.cache_is_complete(
        cache, ids, expected_patch_identity="bounds|wcs"
    ) is True
    assert zfp.cache_is_complete(
        cache, ids, expected_patch_identity="other|wcs"
    ) is False
    assert zfp.cache_is_complete(cache, ids, expected_patch_identity="") is False

    # A manifest without patch_identity cannot satisfy an identity expectation.
    manifest = json.loads(zfp.cache_manifest_path(cache).read_text())
    manifest.pop("patch_identity")
    zfp.cache_manifest_path(cache).write_text(json.dumps(manifest))
    assert zfp.cache_is_complete(
        cache, ids, expected_patch_identity="bounds|wcs"
    ) is False

    # Missing/invalid manifest dimensions never raise, and fail the expectation.
    manifest.pop("height", None)
    zfp.cache_manifest_path(cache).write_text(json.dumps(manifest))
    assert zfp.cache_is_complete(cache, ids, expected_hw=(20, 20)) is False
    manifest["height"] = "not-an-int"
    zfp.cache_manifest_path(cache).write_text(json.dumps(manifest))
    assert zfp.cache_is_complete(cache, ids, expected_hw=(20, 20)) is False


def test_cache_is_complete_checks_hashes_after_geometry_match(tmp_path):
    ids = ["f0"]
    cache = tmp_path / "cache"
    _write_aligned_cache(cache, patch_identity="bounds|wcs")
    manifest = zfp.load_cache_manifest(cache)
    frame_file = cache / manifest["frames"][0]["rgb"]
    with open(frame_file, "ab") as fh:
        fh.write(b"corruption")
    assert zfp.cache_is_complete(
        cache, ids, expected_hw=(20, 20),
        expected_patch_identity="bounds|wcs",
    ) is False


def _write_single_frame_corpus(root: Path, h=400, w=400):
    root.mkdir(parents=True, exist_ok=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
    wcs.array_shape = (h, w)
    rng = np.random.default_rng(5)
    data = (100.0 + rng.normal(0, 1, (h, w))).astype(np.float32)
    fits.PrimaryHDU(data, header=wcs.to_header(relax=True)).writeto(
        root / "single.fit", overwrite=True
    )
    descs, _rejected = zg.read_manifest(root)
    return descs


def test_production_cache_never_reuses_inter_halo_geometry(tmp_path):
    descs = _write_single_frame_corpus(tmp_path / "frames")
    canvas = zg.build_canvas(descs)
    fps = zal.project_footprints(descs, canvas)
    nx = ny = 3
    ctx64 = _contexts(fps, canvas, nx, ny, 64)
    ctx8 = _contexts(fps, canvas, nx, ny, 8)
    cache_root = tmp_path / "cache"

    # One frame covers every patch: the ordered frame-id lists are identical, so
    # only the new geometry identity can reject the stale cache.
    assert all(
        list(m64.patch_ids) == list(m8.patch_ids)
        for (_r, _c, _ce, _p64, m64), (_r2, _c2, _ce2, _p8, m8)
        in zip(ctx64, ctx8)
    )

    _dirs64, _m64, rep64 = zz._build_aligned_cache_frame_major(
        descs, canvas, ctx64, cache_root, None
    )
    assert rep64["reused"] == []
    assert set(rep64["rebuilt"]) == {c.cell_id for (_r, _c, c, _p, _m) in ctx64}

    # H=8 against a complete H=64 cache with the SAME frame ids MUST rebuild.
    _dirs8, _m8, rep8 = zz._build_aligned_cache_frame_major(
        descs, canvas, ctx8, cache_root, None
    )
    assert rep8["reused"] == []
    assert set(rep8["rebuilt"]) == set(rep64["rebuilt"])

    # Provider shape follows the ACTUAL H=8 patch, and the core slice fits.
    for (_r, _c, cell, patch, _mem) in ctx8:
        provider = zfp.MemmapCanonicalProvider(cache_root / cell.cell_id)
        try:
            assert (provider.height, provider.width) == patch.patch_shape_hw
            cs = patch.core_slice
            assert cs.y0 >= 0 and cs.x0 >= 0
            assert cs.y1 <= provider.height and cs.x1 <= provider.width
        finally:
            provider.close()

    # Same-H resume still reuses; hashes are still verified by the predicate.
    _dirs8b, _m8b, rep8b = zz._build_aligned_cache_frame_major(
        descs, canvas, ctx8, cache_root, None
    )
    assert rep8b["rebuilt"] == []
    assert set(rep8b["reused"]) == set(rep8["rebuilt"])


# ---------------------------------------------------------------------------
# F-3 — membership witness across H=8 / H=64
# ---------------------------------------------------------------------------

def _tan_descriptor(fid, crval, crpix, shape, scale=0.001):
    h, w = shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(crval[0]), float(crval[1])]
    wcs.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def test_membership_includes_frame_only_beyond_h8_annulus():
    # 300x300 canvas -> 3x3 cores of 100 px. Cell r0001c0001 covers
    # x/y 100..200; its H=8 patch reaches 92..208, its H=64 patch 36..264.
    # The small frame sits at x 210..240, y 150..180: inside H=64, outside H=8.
    base = _tan_descriptor("base.fit", (10.0, 30.0), (150.0, 150.0), (300, 300))
    annulus = _tan_descriptor("annulus.fit", (9.914, 30.015), (15.0, 15.0), (30, 30))
    descs = [base, annulus]
    canvas = zg.build_canvas(descs)
    assert 300 <= canvas.width <= 301 and 300 <= canvas.height <= 301
    fps = zal.project_footprints(descs, canvas)

    ctx8 = _contexts(fps, canvas, 3, 3, 8)
    ctx64 = _contexts(fps, canvas, 3, 3, 64)
    by_id8 = {c.cell_id: m for (_r, _c, c, _p, m) in ctx8}
    by_id64 = {c.cell_id: m for (_r, _c, c, _p, m) in ctx64}

    m8 = by_id8["r0001c0001"]
    m64 = by_id64["r0001c0001"]
    assert "annulus.fit" not in m8.core_ids
    assert "annulus.fit" not in m64.core_ids
    assert "annulus.fit" not in m8.patch_ids
    assert "annulus.fit" in m64.patch_ids
