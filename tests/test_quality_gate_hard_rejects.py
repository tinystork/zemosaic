import os
import sys

import numpy as np


# Ensure repository root is on sys.path so tests can import module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zequalityMT import quality_metrics
from zemosaic_worker import _evaluate_quality_gate_metrics


def _make_edge_blowup_tile(hw=(256, 256), *, edge_px=64, value=1e12) -> np.ndarray:
    h, w = hw
    arr = np.zeros((h, w, 3), dtype=np.float32)
    b = int(edge_px)
    arr[:b, :, :] = value
    arr[-b:, :, :] = value
    arr[:, :b, :] = value
    arr[:, -b:, :] = value
    return arr


def test_zequality_hard_reject_small_dim():
    arr = np.zeros((32, 32, 3), dtype=np.float32)
    m = quality_metrics(arr, edge_band=64, k_sigma=2.5, erode_px=3)
    assert int(m.get("hard_reject", 0)) == 1
    assert m.get("hard_reject_reason") == "small_dim"


def test_zequality_hard_reject_edge_blowup():
    arr = _make_edge_blowup_tile(hw=(256, 256), edge_px=64, value=1e12)
    m = quality_metrics(arr, edge_band=64, k_sigma=2.5, erode_px=3)
    assert int(m.get("hard_reject", 0)) == 1
    assert m.get("hard_reject_reason") == "edge_blowup"
    assert float(m.get("edge_p99", 0.0)) > 1e6
    assert float(m.get("edge_ratio", 0.0)) > 1e6


def test_worker_quality_gate_forces_reject_on_hard_reject():
    arr = _make_edge_blowup_tile(hw=(256, 256), edge_px=64, value=1e12)
    result = _evaluate_quality_gate_metrics(
        1,
        arr,
        enabled=True,
        threshold=999.0,  # would otherwise accept if only score-based
        edge_band=64,
        k_sigma=2.5,
        erode_px=3,
        pcb=None,
        alpha_mask=None,
    )
    assert result is not None
    assert result.get("accepted") is False
    metrics = result.get("metrics") or {}
    assert int(metrics.get("hard_reject", 0)) == 1
    assert metrics.get("hard_reject_reason") == "edge_blowup"

