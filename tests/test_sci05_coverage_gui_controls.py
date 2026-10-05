"""SCI-05 Gate E5b — coverage GUI controls + legacy radial inert (deterministic).

Headless-safe tests for the Qt coverage controls replacement and the legacy
radial-weighting runtime inertness. No display, network, or filesystem writes.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from zemosaic.zemosaic_align_stack_gpu import _compute_radial_weight_map  # noqa: E402
from zemosaic.zemosaic_gui_qt import ZeMosaicQtMainWindow  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_window():
    window = ZeMosaicQtMainWindow.__new__(ZeMosaicQtMainWindow)
    QMainWindow.__init__(window)  # initialize the Qt base class
    window.config = {
        "coverage_support_taper": True,
        "coverage_aware_reconstruction": False,
    }
    window._config_fields = {}
    window.localizer = types.SimpleNamespace(get=lambda key, fallback: fallback)
    return window


# ---------------------------------------------------------------------------
# Qt coverage controls
# ---------------------------------------------------------------------------

class TestCoverageGuiControls:
    def test_canonical_checkboxes_registered_with_defaults(self):
        window = _make_window()
        window._create_stacking_group()
        fields = window._config_fields

        assert "coverage_support_taper" in fields
        assert "coverage_aware_reconstruction" in fields

        taper = fields["coverage_support_taper"]
        recon = fields["coverage_aware_reconstruction"]
        assert taper["kind"] == "checkbox" and taper["type"] is bool
        assert recon["kind"] == "checkbox" and recon["type"] is bool
        # taper default ON, reconstruction default OFF
        assert bool(taper["widget"].isChecked()) is True
        assert bool(recon["widget"].isChecked()) is False

    def test_legacy_radial_fields_absent(self):
        window = _make_window()
        window._create_stacking_group()
        fields = window._config_fields
        for legacy in ("apply_radial_weight", "radial_feather_fraction",
                       "min_radial_weight_floor"):
            assert legacy not in fields

    def test_settings_round_trip(self):
        window = _make_window()
        window._create_stacking_group()
        taper_widget = window._config_fields["coverage_support_taper"]["widget"]
        recon_widget = window._config_fields["coverage_aware_reconstruction"]["widget"]

        taper_widget.setChecked(False)
        recon_widget.setChecked(True)

        assert window.config["coverage_support_taper"] is False
        assert window.config["coverage_aware_reconstruction"] is True
        # the UI never produces the legacy radial keys
        for legacy in ("apply_radial_weight", "radial_feather_fraction",
                       "min_radial_weight_floor"):
            assert legacy not in window.config


# ---------------------------------------------------------------------------
# Legacy radial weighting inert
# ---------------------------------------------------------------------------

class TestLegacyRadialInert:
    def test_compute_radial_weight_map_returns_none_when_requested(self):
        params = {
            "apply_radial_weight": True,
            "radial_feather_fraction": 0.8,
            "radial_shape_power": 2.0,
        }
        result = _compute_radial_weight_map(100, 100, 3, params, None)
        assert result is None

    def test_compute_radial_weight_map_returns_none_when_not_requested(self):
        params = {"apply_radial_weight": False}
        result = _compute_radial_weight_map(100, 100, 3, params, None)
        assert result is None

    def test_compute_radial_weight_map_returns_none_with_logger(self):
        import logging
        logger = logging.getLogger("zemosaic-test-radial-inert")
        params = {"apply_radial_weight": True}
        result = _compute_radial_weight_map(100, 100, 1, params, logger)
        assert result is None
