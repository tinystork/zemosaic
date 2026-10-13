"""ZM-CELL-PHOTOMETRY-R34F targeted tests — the ``"auto"`` production default.

``"auto"`` is the DEFAULT ``cell_gauge_mode``: after the global gauge is computed
the run uses ``"global"`` when the global gauge serves EVERY non-empty Cell (the
R11 path, byte-identical, no offsets/feather) and otherwise switches the WHOLE
run to ``"per_cell"`` (R34b gauges + R34c offsets + R34d feather). The user does
nothing — the normal path "just works".

Covers the R34F deliverables:

* ``resolve_cell_gauge_mode`` accepts ``"auto"``/``"global"``/``"per_cell"`` and
  rejects unknown tokens; the default is ``"auto"``;
* ``select_auto_gauge_mode`` mirrors ``subset_fixed_normalization`` EXACTLY (the
  ONLY raise it produces for a populated Cell) — never falsely ``global`` (no
  crash) and never falsely ``per_cell`` when ``global`` would have worked (the
  consistency witness);
* the boundary (a Cell just-active vs just-inactive) and the empty-Cell rule (a
  zero-contributor Cell is NOT a trigger);
* determinism + manifest provenance (``photometric_gauge.auto_selection``);
* a synthetic DISJOINT end-to-end run: ``auto`` self-heals to ``per_cell`` and
  COMPLETES (no ``cell gauge has no active frame``);
* a gated REAL M106 end-to-end: ``auto`` resolves to ``global`` and the science is
  BYTE-IDENTICAL to the explicit ``global`` default (the non-regression gate).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import zemosaic.core.canonical_stacking as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    compute_fixed_normalization,
    subset_fixed_normalization,
)
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import photometric as zphot
from zemosaic.core.zegrid.executor import ExecutorConfig

LIGHTS = Path("/home/tristan/M106/lights")
N_E2E_FRAMES = 6


# ---------------------------------------------------------------------------
# Fixtures / helpers (self-contained; mirror the R34b cluster corpus)
# ---------------------------------------------------------------------------

def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _request(n, **kw):
    base = dict(
        images=[None] * n,
        geometric_support=[None] * n,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="kappa_sigma",
        combine="mean",
        taper="footprint",
        taper_px=8.0,
    )
    base.update(kw)
    return CanonicalStackRequest(**base)


def _make_frame(base, mask, level, rng, sigma=0.5):
    h, w = base.shape
    data = np.full((h, w), np.nan, dtype=np.float32)
    noise = rng.normal(0.0, sigma, (h, w))
    data[mask] = (base + level + noise)[mask].astype(np.float32)
    return data


def _cluster_corpus(h=48, w=48, n_a=3, n_b=3, seed=11):
    """Two spatial clusters (LEFT half = A, RIGHT half = B), disjoint.

    Every frame carries the SAME valid count, so the global reference is the
    lowest index (a cluster-A frame); the cluster-B frames are DISJOINT from it
    and are excluded by the global gauge -> the exact Caldwell shape.
    """
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.normal(0.0, 1.0, (h, w))
    left = np.zeros((h, w), dtype=bool)
    left[:, : w // 2] = True
    right = np.zeros((h, w), dtype=bool)
    right[:, w // 2 :] = True
    frames, masks = [], []
    for i in range(n_a):
        frames.append(_make_frame(base, left, i * 3.0, rng))
        masks.append(left)
    for i in range(n_b):
        frames.append(_make_frame(base, right, (5 + i) * 3.0, rng))
        masks.append(right)
    return frames, masks, left, right, n_a, n_b


def _connected_corpus(h=32, w=32, n=4, seed=7):
    """ALL frames share the full support -> every Cell is globally served."""
    rng = np.random.default_rng(seed)
    base = 100.0 + rng.normal(0.0, 1.0, (h, w))
    full = _full_support(h, w)
    frames = [_make_frame(base, full, i * 2.0, rng) for i in range(n)]
    return frames, [full] * n


def _sha(data):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(data, dtype=np.float32)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Mode token / default (ZM-CELL-PHOTOMETRY-R34F)
# ---------------------------------------------------------------------------

def test_auto_token_and_default():
    assert zphot.CELL_GAUGE_MODE_AUTO == "auto"
    assert zphot.CELL_GAUGE_MODE_DEFAULT == "auto"
    assert set(zphot.CELL_GAUGE_MODES) == {"global", "per_cell", "auto"}
    assert zphot.resolve_cell_gauge_mode(None) == "auto"
    assert zphot.resolve_cell_gauge_mode("auto") == "auto"
    assert zphot.resolve_cell_gauge_mode(" AUTO ") == "auto"
    assert zphot.resolve_cell_gauge_mode("global") == "global"
    assert zphot.resolve_cell_gauge_mode("per_cell") == "per_cell"
    for bad in ("", "per-cell", "percell", "local", "automatic", 3, True):
        with pytest.raises(ValueError):
            zphot.resolve_cell_gauge_mode(bad)


def test_science_config_default_is_auto():
    assert ExecutorConfig().science_config().cell_gauge_mode == "auto"


# ---------------------------------------------------------------------------
# select_auto_gauge_mode — the pure decision
# ---------------------------------------------------------------------------

def test_auto_global_on_connected_corpus():
    frames, masks = _connected_corpus()
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), _request(len(frames)))
    membership = [
        ("cell0", [0, 1]),
        ("cell1", [2, 3]),
    ]
    effective, info = zphot.select_auto_gauge_mode(gauge, list(range(len(frames))), membership)
    assert effective == "global"
    assert info is None


def test_auto_per_cell_on_disjoint_corpus():
    frames, masks, _l, _r, n_a, n_b = _cluster_corpus()
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), _request(len(frames)))
    assert 0 <= gauge.reference_index < n_a  # reference is a cluster-A frame
    membership = [
        ("cellA", list(range(0, n_a))),
        ("cellB", list(range(n_a, n_a + n_b))),
    ]
    effective, info = zphot.select_auto_gauge_mode(gauge, list(range(len(frames))), membership)
    assert effective == "per_cell"
    assert info["trigger_count"] == 1
    assert info["first_cell"] == "cellB"
    assert info["reason"] == "no globally weight-active frame"
    assert info["cells"] == ["cellB"]


def test_auto_boundary_just_active_vs_just_inactive():
    """A Cell that contains the global reference is served; a Cell whose frames
    are all globally inactive is not — the exact boundary of the decision."""
    frames, masks, _l, _r, n_a, n_b = _cluster_corpus()
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), _request(len(frames)))
    # cell_ref contains the global reference index -> served even if its own
    # weight_active were to be false (subset never raises on the ref path).
    ref = int(gauge.reference_index)
    eff_ref, info_ref = zphot.select_auto_gauge_mode(
        gauge, list(range(len(frames))), [("cell_ref", [ref])]
    )
    assert eff_ref == "global" and info_ref is None
    # A Cell with the excluded cluster-B frames only -> not served.
    eff_b, info_b = zphot.select_auto_gauge_mode(
        gauge, list(range(len(frames))), [("cell_b", list(range(n_a, n_a + n_b)))]
    )
    assert eff_b == "per_cell"
    assert info_b["first_cell"] == "cell_b"


def test_auto_empty_cell_is_not_a_trigger():
    frames, masks, _l, _r, n_a, n_b = _cluster_corpus()
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), _request(len(frames)))
    # only the (served) reference Cell + an EMPTY Cell -> still "global".
    ref = int(gauge.reference_index)
    effective, info = zphot.select_auto_gauge_mode(
        gauge, list(range(len(frames))), [("cell_ref", [ref]), ("cell_empty", [])]
    )
    assert effective == "global" and info is None


def test_select_auto_mirrors_subset_raise_exactly():
    """THE consistency witness: over many Cell partitions, ``auto`` picks
    ``"global"`` IFF ``subset_fixed_normalization`` raises for NO populated Cell.

    This is what guarantees ``auto`` never falsely picks global (which would crash)
    and never falsely picks per_cell when global would have worked.
    """
    frames, masks, _l, _r, n_a, n_b = _cluster_corpus(h=40, w=40, n_a=4, n_b=4)
    all_ids = list(range(len(frames)))
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), _request(len(frames)))

    rng = np.random.default_rng(3)
    for _ in range(60):
        # random partition of the frame ids into 1..4 cells (some may be empty)
        labels = rng.integers(0, 4, size=len(all_ids))
        cells: dict = {}
        for fid, lab in zip(all_ids, labels):
            cells.setdefault(f"c{int(lab)}", []).append(fid)
        membership = sorted(cells.items())
        any_raises = False
        for _cid, ids in membership:
            if not ids:
                continue
            try:
                subset_fixed_normalization(gauge, all_ids, ids)
            except cs.CanonicalStackValidationError:
                any_raises = True
        effective, _info = zphot.select_auto_gauge_mode(gauge, all_ids, membership)
        assert (effective == "global") == (not any_raises), (membership, effective, any_raises)


def test_select_auto_deterministic():
    frames, masks, _l, _r, n_a, n_b = _cluster_corpus()
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), _request(len(frames)))
    membership = [("cellA", list(range(0, n_a))), ("cellB", list(range(n_a, n_a + n_b)))]
    assert zphot.select_auto_gauge_mode(gauge, list(range(len(frames))), membership) == \
        zphot.select_auto_gauge_mode(gauge, list(range(len(frames))), membership)


# ---------------------------------------------------------------------------
# Manifest provenance
# ---------------------------------------------------------------------------

class _Assembled:
    science = np.zeros((10, 10, 3), dtype=np.float32)
    stack_depth = np.ones((10, 10), dtype=np.int32)
    complete_cells = ["r0000c0000"]
    incomplete_cells = []
    hole_pixels = 0
    coverage_pixels = 100


def _synthetic_tan_wcs(shape=(10, 10)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


def _manifest(tmp_path, *, mode, auto_gauge_info=None):
    from zemosaic import zemosaic_zegrid_mode as zz

    w = zg.GlobalCanvas(
        canvas_id="x", wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=10, height=10, resolution_deg=0.001,
    )
    descs = [zg.FrameDescriptor(
        frame_id=zg.FrameId("a.fits"), source_path="/x/a.fits", shape_hw=(10, 10),
        wcs_header=_synthetic_tan_wcs().to_header().tostring(), header_sha256="", instrument="",
    )]
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0}
    cfg = replace(ExecutorConfig().science_config(), cell_gauge_mode=mode)
    _, _, manifest_path = zz._write_outputs(
        _Assembled(), w, 1, 1, tmp_path, descs, {}, [],
        layout, cfg, 0, {}, None,
        global_reference_frame_id="a.fits",
        auto_gauge_info=auto_gauge_info,
    )
    return json.loads(manifest_path.read_text())


def test_manifest_records_auto_global(tmp_path):
    auto_info = {
        "requested": "auto", "resolved": "global", "fallback_fired": False,
        "cells_checked": 4, "trigger_count": 0, "first_cell": None,
        "reason": None, "cells": [],
    }
    m = _manifest(tmp_path, mode="global", auto_gauge_info=auto_info)
    pg = m["photometric_gauge"]
    assert pg["mode"] == "global"
    assert pg["requested_mode"] == "auto"
    assert pg["auto_selection"] == auto_info


def test_manifest_records_auto_per_cell_fallback(tmp_path):
    auto_info = {
        "requested": "auto", "resolved": "per_cell", "fallback_fired": True,
        "cells_checked": 8, "trigger_count": 2, "first_cell": "r0000c0002",
        "reason": "no globally weight-active frame", "cells": ["r0000c0002", "r0001c0002"],
    }
    m = _manifest(tmp_path, mode="per_cell", auto_gauge_info=auto_info)
    pg = m["photometric_gauge"]
    assert pg["mode"] == "per_cell"
    assert pg["requested_mode"] == "auto"
    assert pg["auto_selection"]["fallback_fired"] is True
    assert pg["auto_selection"]["first_cell"] == "r0000c0002"


def test_manifest_auto_selection_absent_when_not_auto(tmp_path):
    m = _manifest(tmp_path, mode="global", auto_gauge_info=None)
    assert m["photometric_gauge"]["auto_selection"] == {"enabled": False}


# ---------------------------------------------------------------------------
# End-to-end — synthetic DISJOINT corpus self-heals to per_cell and completes
# ---------------------------------------------------------------------------

def _write_synthetic_cluster_corpus(root: Path, *, shape=(400, 600), scale=0.001):
    """Write a 4-frame, 2-cluster (disjoint) synthetic corpus + stack_plan.csv."""
    root.mkdir(parents=True, exist_ok=True)
    centers = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]
    rows = ["file_path,exposure"]
    for i, (ra, dec) in enumerate(centers):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra, dec]
        w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
        w.wcs.cd = np.array([[-1, 0], [0, 1]]) * scale
        w.array_shape = shape
        rng = np.random.default_rng(i)
        data = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
        name = f"f{i}.fit"
        fits.PrimaryHDU(data, header=w.to_header()).writeto(root / name, overwrite=True)
        rows.append(f"{name},10.0")
    (root / "stack_plan.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_e2e_auto_self_heals_to_per_cell_on_disjoint_corpus(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    inp = tmp_path / "in"
    _write_synthetic_cluster_corpus(inp)
    out = tmp_path / "out"
    # Pin the layout: the auto-layout scientific floors are calibrated for real
    # frames; a synthetic tiny corpus needs an explicit partition.
    zz.run_zegrid_mode(str(inp), str(out), zconfig=SimpleNamespace(zegrid_layout="2x2"))

    m = json.loads((out / "zegrid_manifest.json").read_text())
    pg = m["photometric_gauge"]
    # the default ("auto") resolved to per_cell (global could not serve 2 cells).
    assert pg["mode"] == "per_cell"
    assert pg["requested_mode"] == "auto"
    assert pg["auto_selection"]["fallback_fired"] is True
    assert pg["auto_selection"]["trigger_count"] >= 1
    assert pg["auto_selection"]["reason"] == "no globally weight-active frame"
    # the run COMPLETED (no "cell gauge has no active frame") and feathered.
    assert not m["incomplete_cells"]
    assert m["complete_cells"]
    assert m["cell_feather"]["enabled"] is True
    assert (out / "mosaic_grid.fits").exists()


# ---------------------------------------------------------------------------
# End-to-end — REAL M106 (compact): auto == explicit global, BYTE-IDENTICAL
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_e2e_auto_m106_byte_identical_to_global(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    frames = sorted(LIGHTS.glob("*.fit"))[:N_E2E_FRAMES]
    assert len(frames) == N_E2E_FRAMES

    results = {}
    for tag, kwargs in (("auto", {}), ("global", {"cell_gauge_mode": "global"})):
        inp = tmp_path / f"in_{tag}"
        inp.mkdir(parents=True, exist_ok=True)
        for p in frames:
            shutil.copy2(p, inp / p.name)
        lines = ["file_path,exposure"]
        for p in frames:
            lines.append(f"{p.name},10.0")
        (inp / "stack_plan.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = tmp_path / f"out_{tag}"
        zz.run_zegrid_mode(str(inp), str(out), **kwargs)
        with fits.open(out / "mosaic_grid.fits") as hdul:
            sci = np.asarray(hdul[0].data, dtype=np.float32)
        man = json.loads((out / "zegrid_manifest.json").read_text())
        results[tag] = (sci, man)

    auto_sci, auto_man = results["auto"]
    global_sci, global_man = results["global"]

    # auto (the DEFAULT) resolves to global on the compact M106 corpus...
    assert auto_man["photometric_gauge"]["mode"] == "global"
    assert auto_man["photometric_gauge"]["requested_mode"] == "auto"
    assert auto_man["photometric_gauge"]["auto_selection"]["fallback_fired"] is False
    assert auto_man["photometric_gauge"]["auto_selection"]["trigger_count"] == 0
    # ...and the assembled science is BYTE-IDENTICAL to the explicit global default.
    np.testing.assert_array_equal(auto_sci, global_sci)
    assert _sha(auto_sci) == _sha(global_sci)
    assert auto_man["cell_feather"]["enabled"] is False
