"""ZM-ZEGRID-R11 M16 measurement tests (gated on real data; reuse persisted caches).

Pins the required evidence on M16 (iteration-speed dataset):

* LAYOUT-INDEPENDENCE: the global gauge makes the assembled science BIT-IDENTICAL
  across two pinned layouts (3x3 vs 2x2) — the exact science SHA-256 is pinned.
* REFERENCE UNIFICATION: BEFORE, every Cell picks its OWN reference (N distinct
  frames); AFTER, every Cell shares ONE global reference frame.
* SEAM METRIC: the inter-cell boundary-step metric (median/max % step between
  adjacent 6-px strips inside cores) is reported BEFORE vs AFTER and improves.

These reuse the disk-backed gauge/cell caches built by ``tools/zegrid_r11/measure.py``
(NEVER /tmp) so they do not re-do the slow full-canvas reprojection. They are
skipped when the M16 lights directory is absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.canonical_streaming import subset_fixed_normalization
from zemosaic.core.zegrid import assembly as za
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic.core.zegrid import mosaic as zmosaic
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid import seam as zsm
from zemosaic.core.zegrid import sweep as zsw
from zemosaic.core.zegrid.executor import ExecutorConfig

LIGHTS = Path("/home/tristan/M16/quick")
CACHE_ROOT = Path("/home/tristan/zegrid_r11_m16_measure")
NOOP = lambda *a, **k: None

# Pinned measurement (M16 quick, 2026-10-07, this machine).
PINNED_SCIENCE_SHA256 = "30306a5a482e88eb700e7071dae1ee5ecf6cf4761a8a017ced352f8264a272f2"
PINNED_GLOBAL_REFERENCE = "Light_M 16_10.0s_LP_20250530-035734.fit"


def _science_hash(science: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(science, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _run_layout(descs, canvas, nx, ny, config, gauge, global_frame_ids, use_gauge):
    cell_ctxs = []
    for row, col, _b in zg.build_layout(canvas, nx, ny).iter_cells(canvas):
        cell, patch, mem = zsw.build_cell_context(descs, canvas, row, col, nx, ny)
        cell_ctxs.append((row, col, cell, patch, mem))
    cores = {}
    ref_ids = {}
    for (row, col, cell, patch, mem) in cell_ctxs:
        cid = cell.cell_id
        if not mem.patch_ids:
            cores[cid] = None
            continue
        cache_dir = CACHE_ROOT / f"cache_{nx}x{ny}" / cid
        provider = zfp.MemmapCanonicalProvider(str(cache_dir))
        cell_frame_ids = list(provider.frame_ids)
        fixed = subset_fixed_normalization(gauge, global_frame_ids, cell_frame_ids) if use_gauge else None
        mt, sres = zz._run_cell_stream(str(cache_dir), patch, config, NOOP, fixed=fixed)
        cores[cid] = za.crop_all_planes_to_core(mt)
        ref_ids[cid] = sres.reference_frame_id
    assembled = zmosaic.assemble_canvas(canvas, nx, ny, cores)
    metric = zsm.compute_boundary_step_metric(canvas, assembled.science, nx, ny, strip_px=6)
    return assembled, ref_ids, metric


@pytest.fixture(scope="module")
def m16_results():
    """Run all layouts once (BEFORE + AFTER + second layout) and cache the results."""
    if not LIGHTS.is_dir():
        pytest.skip("M16 lights directory not present")
    if not (CACHE_ROOT / "__gauge__" / "manifest.json").exists():
        pytest.skip("M16 gauge cache not built (run tools/zegrid_r11/measure.py)")
    descs, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()
    gauge, global_frame_ids = zphot.compute_global_gauge(
        descs, canvas, lambda f: zz._decode_frame_hwc(f, NOOP), config, CACHE_ROOT / "__gauge__"
    )
    a3_before, refs3_before, seam3_before = _run_layout(descs, canvas, 3, 3, config, gauge,
                                                        global_frame_ids, use_gauge=False)
    a3_after, refs3_after, seam3_after = _run_layout(descs, canvas, 3, 3, config, gauge,
                                                     global_frame_ids, use_gauge=True)
    a2_after, _, _ = _run_layout(descs, canvas, 2, 2, config, gauge, global_frame_ids, use_gauge=True)
    return {
        "gauge": gauge,
        "global_frame_ids": global_frame_ids,
        "hash_3x3_before": _science_hash(a3_before.science),
        "hash_3x3_after": _science_hash(a3_after.science),
        "hash_2x2_after": _science_hash(a2_after.science),
        "refs_before": refs3_before,
        "refs_after": refs3_after,
        "seam_before": seam3_before,
        "seam_after": seam3_after,
        "science_3x3_after": a3_after.science,
        "science_2x2_after": a2_after.science,
    }


def test_layout_independence_identical_science(m16_results):
    r = m16_results
    assert r["hash_3x3_after"] == PINNED_SCIENCE_SHA256
    assert r["hash_2x2_after"] == PINNED_SCIENCE_SHA256
    assert r["hash_3x3_before"] != r["hash_3x3_after"]  # the fix changed the science
    np.testing.assert_array_equal(r["science_3x3_after"], r["science_2x2_after"])


def test_reference_unification(m16_results):
    r = m16_results
    # BEFORE: each cell picks its OWN reference (the root cause).
    assert len(set(r["refs_before"].values())) > 1
    # AFTER: every cell shares ONE global reference.
    assert len(set(r["refs_after"].values())) == 1
    assert next(iter(r["refs_after"].values())) == PINNED_GLOBAL_REFERENCE
    assert r["global_frame_ids"][int(r["gauge"].reference_index)] == PINNED_GLOBAL_REFERENCE


def test_seam_metric_improves(m16_results):
    r = m16_results
    before = r["seam_before"]
    after = r["seam_after"]
    assert before["median_pct"] == pytest.approx(0.0706206, abs=1e-5)
    assert before["max_pct"] == pytest.approx(0.1897343, abs=1e-5)
    assert after["median_pct"] == pytest.approx(0.0654814, abs=1e-5)
    assert after["max_pct"] == pytest.approx(0.1825624, abs=1e-5)
    assert after["median_pct"] < before["median_pct"]
    assert after["max_pct"] < before["max_pct"]
