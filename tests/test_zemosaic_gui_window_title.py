"""ZM-ZEGRID-MANIFEST-RSS-AND-GUI-VERSION: window title derivation.

The Qt window title must derive ``V<version>`` from ``zemosaic.__version__``
instead of hardcoding a stale number, while keeping the product name
"Tesserae Compositae" and every locale's own wording. Headless/offscreen only;
no real window is shown and no processing starts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LOCALES = ("en", "de", "fr", "is", "nl")

pytestmark = pytest.mark.skipif(
    not (SRC / "zemosaic" / "zemosaic_gui_qt.py").is_file(),
    reason="GUI module unavailable",
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt not installed
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_main_window(tmp_path, monkeypatch):
    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = tmp_path / "zemosaic_config.json"
    fake_cfg = SimpleNamespace(
        load_config=lambda: {},
        save_config=lambda data: None,
        get_config_path=lambda: cfg_path,
        DEFAULT_CONFIG={},
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui, gui.ZeMosaicQtMainWindow()


def test_window_title_derives_version_and_keeps_product_name(tmp_path, monkeypatch):
    import zemosaic

    monkeypatch.setattr(zemosaic, "__version__", "5.1.1", raising=False)
    gui, win = _make_main_window(tmp_path, monkeypatch)
    title = win.windowTitle()
    assert "V5.1.1" in title
    assert "Tesserae Compositae" in title
    assert "V5.0.0" not in title

    # Changing __version__ must change the title: no hardcoded copy remains.
    monkeypatch.setattr(zemosaic, "__version__", "9.9.9", raising=False)
    changed = win._window_title()
    assert "V9.9.9" in changed
    assert "Tesserae Compositae" in changed
    assert "V5.1.1" not in changed

    # Both call sites route through the helper.
    win._refresh_translated_ui()
    assert win.windowTitle() == changed


def test_locale_templates_use_placeholder_and_keep_wording():
    locales_dir = SRC / "zemosaic" / "locales"
    for code in LOCALES:
        data = json.loads((locales_dir / f"{code}.json").read_text(encoding="utf-8"))
        value = data["qt_window_title_preview"]
        assert "V5.0.0" not in value, code
        assert "{version}" in value, code
        assert "Tesserae Compositae" in value, code
    # Product wording/suffixes are untouched.
    de = json.loads((locales_dir / "de.json").read_text(encoding="utf-8"))
    fr = json.loads((locales_dir / "fr.json").read_text(encoding="utf-8"))
    nl = json.loads((locales_dir / "nl.json").read_text(encoding="utf-8"))
    assert de["qt_window_title_preview"].endswith("(Qt-Vorschau)")
    assert fr["qt_window_title_preview"].endswith("(Aperçu Qt)")
    assert nl["qt_window_title_preview"].endswith("(Qt-voorbeeld)")


def test_locale_without_placeholder_and_malformed_template_render_cleanly(
    tmp_path, monkeypatch
):
    import zemosaic

    gui, win = _make_main_window(tmp_path, monkeypatch)
    monkeypatch.setattr(zemosaic, "__version__", "5.1.1", raising=False)

    # es/pl carry no version today: keep their wording verbatim, no braces.
    without = "ZeMosaic (podgląd Qt)"
    assert gui._format_window_title(without, "5.1.1") == without
    monkeypatch.setattr(win.localizer, "get", lambda key, fallback=None: without)
    win._refresh_translated_ui()
    assert win.windowTitle() == without

    # A broken placeholder must never raise or leak raw braces.
    for odd in ("ZeMosaic V{version, Tesserae Compositae", "ZeMosaic V{other} – Tesserae"):
        rendered = gui._format_window_title(odd, "5.1.1")
        assert "{" not in rendered and "}" not in rendered
        assert "V5.1.1" in rendered
        assert "Tesserae Compositae" in rendered
