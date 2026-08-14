"""ZeMosaic application bootstrap.

This is the real, install-safe entry point.  It owns the Qt launch, the
optional opening animation, and GPU cleanup on shutdown.  It deliberately does
*not* manipulate ``sys.path``: all imports are package-relative and all bundled
resources are resolved through :mod:`importlib.resources`.

The historical root ``run_zemosaic.py`` launcher is a thin compatibility
wrapper that only makes the in-repo ``src/`` layout importable and delegates
here.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency guard
    from .cuda_utils import CUPY_AVAILABLE
except Exception:  # pragma: no cover - fallback when cuda_utils is unavailable
    CUPY_AVAILABLE = False

from ._resources import resource_path_optional

_OPENING_GIF_CANDIDATES = (
    "opening.gif",
    "opening.GIF",
    "opening.gif.gif",
    "opening.GIF.GIF",
)
_OPENING_GIF_DEFAULT_DURATION_MS = 6000



def _notify_qt_backend_unavailable(error: Exception) -> None:
    """Inform the user that the Qt backend cannot be used."""

    print(
        "[zemosaic] Unable to load the Qt interface because the optional PySide6 "
        "dependency is missing or broken."
    )
    print("[zemosaic] Install/fix it with `pip install PySide6` to use ZeMosaic.")
    print(f"[zemosaic] Original import error: {error}")


def _determine_backend(argv):
    """Normalize legacy backend CLI flags for the Qt-only runtime."""

    cleaned_args = []
    for arg in argv:
        if arg == "--qt-gui":
            continue
        if arg == "--tk-gui":
            print("[zemosaic] '--tk-gui' is no longer supported; ZeMosaic official runtime is Qt-only.")
            continue
        cleaned_args.append(arg)
    return "qt", cleaned_args, "qt-only"


def _resolve_opening_gif_path() -> Optional[Path]:
    """Return the best-effort path to the opening animation."""

    search_roots: list[Path] = []

    # 1. Bundled package resource (install-safe).
    for name in _OPENING_GIF_CANDIDATES:
        candidate = resource_path_optional("gif", name)
        if candidate is not None and candidate.is_file():
            return candidate

    # 2. Legacy checkout layout: gif/ next to CWD.
    try:
        cwd_root = Path.cwd() / "gif"
        search_roots.append(cwd_root)
    except Exception:
        pass

    for root in search_roots:
        if not root or not root.is_dir():
            continue
        for name in _OPENING_GIF_CANDIDATES:
            candidate = root / name
            if candidate.is_file():
                return candidate
        try:
            for entry in root.iterdir():
                lower_name = entry.name.lower()
                if not lower_name.startswith("opening") or not lower_name.endswith(".gif"):
                    continue
                if entry.is_file():
                    return entry
        except Exception:
            continue

    print(
        "[zemosaic] Opening animation not found. "
        "Ensure the package data includes 'gif/opening.gif'."
    )
    return None


def _estimate_gif_duration_ms(gif_path: Path) -> int:
    """Best-effort estimate of the GIF duration in milliseconds."""

    try:
        from PIL import Image, ImageSequence  # type: ignore
    except Exception:
        return _OPENING_GIF_DEFAULT_DURATION_MS

    total_ms = 0
    try:
        with Image.open(gif_path) as handle:  # type: ignore[attr-defined]
            for frame in ImageSequence.Iterator(handle):  # type: ignore[attr-defined]
                duration = 0
                if hasattr(frame, "info"):
                    duration = frame.info.get("duration", 0)  # type: ignore[arg-type]
                if not isinstance(duration, (int, float)) or duration <= 0:
                    duration = 80
                total_ms += int(duration)
    except Exception as err:
        print(f"[zemosaic] Unable to estimate GIF duration for {gif_path}: {err}")
        return _OPENING_GIF_DEFAULT_DURATION_MS

    return max(total_ms, 1000)


def _play_opening_gif_animation_once() -> None:
    """Display the optional opening animation using the PySide6/QMovie path."""

    gif_path = _resolve_opening_gif_path()
    if gif_path is None:
        return

    try:
        from PySide6.QtCore import QTimer, Qt
        from PySide6.QtGui import QMovie
        from PySide6.QtWidgets import QApplication, QLabel
    except Exception as qt_err:
        print(f"[zemosaic] Unable to import PySide6 for opening animation: {qt_err}")
        return

    app = QApplication.instance()
    owns_app = False
    if app is None:
        try:
            app = QApplication([sys.argv[0], "--zemosaic-opening"])
        except Exception as app_err:
            print(f"[zemosaic] Unable to create QApplication for opening animation: {app_err}")
            return
        owns_app = True
        try:
            app.setProperty("zemosaic_prelaunch_owner", True)
        except Exception:
            pass

    movie = QMovie(str(gif_path))
    if not movie.isValid():
        print(f"[zemosaic] Opening animation at {gif_path} is not a valid GIF.")
        if owns_app:
            try:
                app.quit()
            except Exception:
                pass
        return

    movie.setCacheMode(QMovie.CacheAll)
    manual_loop_control = False
    loop_setter = getattr(movie, "setLoopCount", None)
    if callable(loop_setter):
        try:
            loop_setter(1)
        except Exception as err:
            manual_loop_control = True
            print(f"[zemosaic] Unable to enforce single loop for animation: {err}")
    else:
        manual_loop_control = True
        print("[zemosaic] PySide6 build lacks QMovie.setLoopCount; animation may loop more than once.")

    try:
        movie.jumpToFrame(0)
    except Exception:
        pass
    frame_rect = movie.frameRect()
    frame_width = frame_rect.width() if frame_rect and frame_rect.width() > 0 else None
    frame_height = frame_rect.height() if frame_rect and frame_rect.height() > 0 else None

    splash = QLabel()
    splash.setWindowFlag(Qt.SplashScreen)
    splash.setWindowFlag(Qt.FramelessWindowHint)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint)
    splash.setAttribute(Qt.WA_TranslucentBackground)
    splash.setStyleSheet("background-color: black; border: 1px solid #111;")
    splash.setMovie(movie)
    splash.setScaledContents(False)

    if frame_width and frame_height:
        splash.resize(frame_width, frame_height)
    else:
        pix = movie.currentPixmap()
        if not pix.isNull():
            splash.resize(pix.size())
        else:
            splash.resize(512, 288)

    screen = app.primaryScreen()
    if screen is not None:
        geometry = screen.availableGeometry()
        pos_x = geometry.x() + max(int((geometry.width() - splash.width()) / 2), 0)
        pos_y = geometry.y() + max(int((geometry.height() - splash.height()) / 2), 0)
        splash.move(pos_x, pos_y)

    splash.show()

    finished = {"done": False}

    def _teardown() -> None:
        if finished["done"]:
            return
        finished["done"] = True
        try:
            movie.stop()
        except Exception:
            pass
        try:
            splash.close()
        except Exception:
            pass

    try:
        movie.finished.connect(_teardown)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        def _on_state_changed(state):  # type: ignore[override]
            from PySide6.QtGui import QMovie as _QMovie

            if state == _QMovie.NotRunning:
                _teardown()

        movie.stateChanged.connect(_on_state_changed)  # type: ignore[attr-defined]
    except Exception:
        pass

    duration_ms = _estimate_gif_duration_ms(gif_path)

    if manual_loop_control:
        loop_tracker = {"started": False}

        def _on_frame_changed(frame_number: int) -> None:
            if frame_number <= 0 and loop_tracker["started"]:
                _teardown()
            elif frame_number >= 0:
                loop_tracker["started"] = True

        try:
            movie.frameChanged.connect(_on_frame_changed)  # type: ignore[attr-defined]
        except Exception:
            pass

    QTimer.singleShot(max(duration_ms, 1000), _teardown)
    QTimer.singleShot(60000, _teardown)  # Safety net in case the GIF never ends

    movie.start()

    try:
        app.processEvents()
    except Exception:
        pass

    if owns_app:
        try:
            app.setProperty("zemosaic_prelaunch_owner", True)
        except Exception:
            pass


def main(argv=None) -> int:
    """Launch the ZeMosaic application and return the process exit code."""

    multiprocessing.freeze_support()

    if argv is None:
        argv = sys.argv[1:]

    _backend, cleaned_args, _backend_source = _determine_backend(argv)
    if cleaned_args != argv:
        sys.argv = [sys.argv[0], *cleaned_args]

    try:
        try:
            from .zemosaic_gui_qt import run_qt_main
        except ImportError as qt_import_error:
            _notify_qt_backend_unavailable(qt_import_error)
            return 1

        print("[zemosaic] Launching ZeMosaic with the Qt backend.")
        _play_opening_gif_animation_once()
        exit_code = run_qt_main()
        return exit_code
    finally:
        # Clean up GPU resources to avoid CUDA errors at shutdown.
        if CUPY_AVAILABLE and "cupy" in sys.modules:
            print("[zemosaic] Cleaning up GPU resources...")
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                print("[zemosaic] GPU resources cleaned up successfully.")
            except Exception as e:
                print(f"[zemosaic] Error during GPU cleanup: {e}")


if __name__ == "__main__":
    sys.exit(main())
