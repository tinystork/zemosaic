"""ZM-ZEGRID-AUTO-EXTERNAL-STREAMING M5 — real product E2E + launchability.

One real `run_zegrid_mode` run on the tiny deterministic corpus where AUTO
selects `stream_external` (forced bounded profile + deliberately tight RAM
planning bound), and one RAM streaming reference on the same pinned geometry.
The published science/coverage must be bit-identical; the manifest must declare
`stack_working_set=external_spool`; no spool residue may remain.

Also asserts the GUI→worker call has no working-set/tile override, so the Qt
launch reaches the engine exactly as today.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal

REPO = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO / "tools" / "zegrid_m6"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from make_tiny_corpus import make_tiny_corpus  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30


class _HugeModel:
    def predict_bound_bytes(self, n, area):
        return 1 << 62

    def predict_peak_bytes(self, n, area):
        return 1 << 62


def _manifests(root):
    path = Path(root) / "zegrid_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _science(root):
    root = Path(root)
    candidates = sorted(root.glob("*science*.fits"))
    if not candidates:
        candidates = sorted(root.glob("*.fits"))
    assert candidates, f"no science FITS under {root}"
    return np.asarray(fits.getdata(candidates[0]))


def _coverage(root):
    root = Path(root)
    candidates = sorted(root.glob("*coverage*.fits"))
    if not candidates:
        candidates = sorted(root.glob("*cov*.fits"))
    assert candidates, f"no coverage FITS under {root}"
    return np.asarray(fits.getdata(candidates[0]))


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("m5_tiny")
    return make_tiny_corpus(root / "in")


def test_product_e2e_external_parity_and_no_residue(tiny_corpus, tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        zegrid_layout="2x2",
        zegrid_scheduler_mode="legacy",
    )
    # RAM reference: no in-memory plan, real estimator -> stream at tile 128.
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())
    # Hermetic: the reference half must not depend on the host's current free
    # RAM (a busy CI host otherwise flips an otherwise deterministic run).
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 64 * GIB)
    zz.run_zegrid_mode(
        str(tiny_corpus), str(tmp_path / "ram"), zegrid_halo_px=8, zconfig=cfg
    )
    ram_manifest = _manifests(tmp_path / "ram")
    ram_sci = _science(tmp_path / "ram")
    ram_cov = _coverage(tmp_path / "ram")
    assert ram_manifest.get("stack_working_set") in (None, "ram_tile")

    # External run: bounded + a planning bound no RAM tile can satisfy.
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GIB)
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes", lambda *a, **k: 10 * GIB
    )
    zz.run_zegrid_mode(
        str(tiny_corpus), str(tmp_path / "ext"), zegrid_halo_px=8,
        zconfig=cfg, storage_profile="bounded",
    )
    ext_manifest = _manifests(tmp_path / "ext")
    assert ext_manifest["stack_working_set"] == "external_spool"
    ws = ext_manifest["working_set"]
    assert ws["class"] == "external_spool"
    assert int(ws["tile_size"]) >= 1
    assert int(ws["column_batch"]) >= 1
    assert int(ws["disk"]["spool_peak_bytes_worst_tile"]) > 0
    assert int(ws["disk"]["disk_envelope_bytes"]) > 0
    lifecycle = ext_manifest["external_lifecycle"]
    assert lifecycle["cells_executed_external"] >= 1
    assert lifecycle["spool_bytes_written"] > 0
    assert lifecycle["cleanup_failures"] == []
    # Science/coverage bit-identical to the RAM stream reference.
    ext_sci = _science(tmp_path / "ext")
    ext_cov = _coverage(tmp_path / "ext")
    assert np.array_equal(ram_sci, ext_sci, equal_nan=True)
    assert np.array_equal(ram_cov, ext_cov, equal_nan=True)
    # No spool residue after success (store-less run cleans in the worker).
    residues = list(Path(tmp_path / "ext").rglob("external-spool"))
    assert residues == []
    assert not list(Path(tmp_path / "ext").rglob("ATTEMPT"))


def test_gui_worker_call_has_no_working_set_override():
    from zemosaic import zemosaic_worker as zw

    src = inspect.getsource(zw)
    start = src.index("run_zegrid_mode(\n")
    end = src.index("finally:", start)
    call = src[start:end]
    for forbidden in (
        "cpu_tile", "tile_size", "working_set", "column_batch",
        "spool_root", "storage_profile",
    ):
        assert forbidden not in call
    assert "zegrid_scheduler_mode" in call
