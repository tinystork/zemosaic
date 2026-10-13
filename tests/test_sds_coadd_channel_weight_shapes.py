"""SDS global-coadd regression tests (found via the R35_A M106 SDS run).

Two defects fixed here:
* ``CanonicalStackResult.estimator_weight_sum`` shares the SCIENCE shape
  ``(H, W, C)`` for RGB; the chunked SDS finalize treated it as ``(H, W)`` and
  crashed on ``np.where(invalid[..., None], ...)`` with mismatched shapes.
* the Qt ``_on_worker_phase_changed`` slot referenced ``stage_label`` before
  assignment on the SDS phase-change path (``UnboundLocalError``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_reduce_rgb_weight_maps_is_per_pixel():
    from zemosaic.zemosaic_worker import _reduce_sds_weight_maps

    w = np.zeros((4, 5, 3), dtype=np.float64)
    w[1, 2, 0] = 3.0
    w[1, 2, 2] = 1.0
    w[3, 4, 1] = 2.0
    valid, cov = _reduce_sds_weight_maps(w)
    assert valid.shape == (4, 5)
    assert cov.shape == (4, 5)
    assert valid[1, 2] and valid[3, 4]
    assert not valid[0, 0]
    assert cov[1, 2] == pytest.approx(3.0)
    assert cov[3, 4] == pytest.approx(2.0)
    assert cov.dtype == np.float32


def test_reduce_mono_weight_maps_unchanged():
    from zemosaic.zemosaic_worker import _reduce_sds_weight_maps

    w = np.array([[0.0, 2.0], [1.0, 0.0]])
    valid, cov = _reduce_sds_weight_maps(w)
    assert valid.shape == (2, 2)
    assert valid[0, 1] and valid[1, 0] and not valid[0, 0]
    assert cov[0, 1] == pytest.approx(2.0)


def test_broadcast_mask_matches_rgb_science_shape():
    """The exact failing expression must now broadcast (regression guard)."""
    from zemosaic.zemosaic_worker import _reduce_sds_weight_maps

    rows, width, channels = 4, 6, 3
    science = np.ones((rows, width, channels), dtype=np.float32)
    weight = np.zeros((rows, width, channels), dtype=np.float64)
    weight[0, :, :] = 1.0
    valid, cov = _reduce_sds_weight_maps(weight)
    out = np.where(valid[..., None], science, np.nan)   # must not raise
    assert out.shape == (rows, width, channels)
    assert np.isfinite(out[0, 0, 0])
    assert not np.isfinite(out[1, 0, 0])
    # coverage is per-pixel (H, W) as the assembler expects.
    assert cov.shape == (rows, width)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_phase_changed_with_sds_phase_does_not_raise(tmp_path, monkeypatch):
    """SDS phase-change payload must not raise UnboundLocalError."""
    import zemosaic.zemosaic_gui_qt as gui
    from types import SimpleNamespace

    monkeypatch.setattr(
        gui,
        "zemosaic_config",
        SimpleNamespace(
            load_config=lambda: {},
            save_config=lambda d: None,
            get_config_path=lambda: tmp_path / "cfg.json",
            DEFAULT_CONFIG={},
        ),
    )
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    win = gui.ZeMosaicQtMainWindow()
    try:
        win._on_worker_phase_changed("phase4_grid", {"sds_phase": True, "current": 1, "total": 10})
        win._on_worker_phase_changed("phase4_grid", {"sds_phase": False, "current": 1, "total": 10})
        win._on_worker_phase_changed("phase4_grid", {})
    finally:
        win.deleteLater()
