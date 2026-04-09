"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)         ║
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System   ║
║              (aka ChatGPT, Grand Maître du ciselage de code)                      ║
║                                                                                   ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description :                                                                     ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,                ║
║   dans le but noble de transformer des nuages de photons en art                   ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,                        ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.             ║
║   (le karma des développeurs en dépend).                                          ║
║                                                                                   ║
║ Avertissement :                                                                   ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le                    ║
║   développement de ce code.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau)              ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System      ║
║           (aka ChatGPT, Grand Master of Code Chiseling)                           ║
║                                                                                   ║
║ License : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description:                                                                      ║
║   This program was forged under the sacred light of pixels and                    ║
║   caffeine, with the noble intent of turning clouds of photons into               ║
║   astronomical art. If you use it, please consider saying “thanks,”               ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —                  ║
║   developer karma depends on it.                                                  ║
║                                                                                   ║
║ Disclaimer:                                                                       ║
║   No AIs or butter knives were harmed in the making of this code.                 ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""
import sys  # Ajout pour sys.path et sys.modules
import multiprocessing
import os
import importlib.util
import platform
from pathlib import Path
from typing import Optional
# import reproject # L'import direct ici n'est pas crucial, mais ne fait pas de mal

# Tenter d'importer CUPY_AVAILABLE pour le nettoyage des ressources GPU
try:
    # Cet import est placé ici pour s'assurer qu'il est disponible pour le bloc finally.
    from cuda_utils import CUPY_AVAILABLE
except ImportError:
    # Si cuda_utils n'existe pas ou a un problème, on suppose que cupy n'est pas utilisé.
    CUPY_AVAILABLE = False

# --- Impression de débogage initiale ---
print("--- run_zemosaic.py: DÉBUT DES IMPORTS ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Chemin de travail actuel (CWD): {sys.path[0]}") # sys.path[0] est généralement le dossier du script

_system_name = platform.system().lower()
if _system_name == "darwin":
    print("[Info] macOS détecté : CUDA/GPU désactivé si indisponible. Mode CPU.")

# S'assurer que le dossier parent est dans sys.path afin que les imports relatifs
# dans les modules du package fonctionnent même lorsque ce script est exécuté
# directement.
_package_dir = Path(__file__).resolve().parent
_parent_dir = _package_dir.parent
_parent_dir_str = str(_parent_dir)
if _parent_dir_str not in sys.path:
    sys.path.insert(0, _parent_dir_str)

print("--- run_zemosaic.py: FIN DES IMPORTS ---")
print("DEBUG (run_zemosaic): sys.path complet:\n" + "\n".join(sys.path))
print("-" * 50)


def _is_pyside6_available() -> bool:
    """Return True if the optional PySide6 dependency is installed.

    Using importlib.util.find_spec avoids importing the whole Qt stack
    just to check for availability and keeps startup overhead minimal.
    """
    try:
        spec = importlib.util.find_spec("PySide6")
    except Exception:
        spec = None
    if spec is not None:
        return True
    # Some environments (particularly when Python is launched from a
    # different venv or from a frozen EXE) may not populate the importlib
    # cache correctly even though PySide6 is installed.  As a fallback we
    # attempt a direct import and report success if it works.
    try:  # pragma: no cover - optional dependency probe
        import PySide6  # type: ignore  # noqa: F401  (only for availability check)
    except Exception:
        return False
    return True


def _notify_qt_backend_unavailable(error: Exception) -> None:
    """Inform the user that the Qt backend cannot be used."""

    print(
        "[run_zemosaic] Unable to load the Qt interface because the optional PySide6 "
        "dependency is missing or broken."
    )
    print("[run_zemosaic] Install/fix it with `pip install PySide6` to use ZeMosaic.")
    print(f"[run_zemosaic] Original import error: {error}")


def _determine_backend(argv):
    """Normalize legacy backend CLI flags for Qt-only runtime."""
    cleaned_args = []
    for arg in argv:
        if arg == "--qt-gui":
            continue
        if arg == "--tk-gui":
            print("[run_zemosaic] '--tk-gui' is no longer supported; ZeMosaic official runtime is Qt-only.")
            continue
        cleaned_args.append(arg)
    return "qt", cleaned_args, "qt-only"


_OPENING_GIF_CANDIDATES = (
    "opening.gif",
    "opening.GIF",
    "opening.gif.gif",
    "opening.GIF.GIF",
)
_OPENING_GIF_DEFAULT_DURATION_MS = 6000


def _resolve_opening_gif_path() -> Optional[Path]:
    """Return the best-effort Path to the opening animation."""

    base_dir: Optional[Path] = None
    try:
        from zemosaic_utils import get_app_base_dir  # type: ignore
    except Exception:
        get_app_base_dir = None  # type: ignore
    if callable(get_app_base_dir):
        try:
            base_dir = Path(get_app_base_dir())
        except Exception:
            base_dir = None
    if base_dir is None:
        try:
            base_dir = Path(__file__).resolve().parent
        except Exception:
            base_dir = Path.cwd()

    search_roots: list[Path] = []
    gif_root = base_dir / "gif"
    if gif_root not in search_roots:
        search_roots.append(gif_root)
    try:
        cwd_root = Path.cwd() / "gif"
        if cwd_root not in search_roots:
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
        "[run_zemosaic] Opening animation not found. "
        "Ensure 'gif/opening.gif' exists next to run_zemosaic.py."
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
        print(f"[run_zemosaic] Unable to estimate GIF duration for {gif_path}: {err}")
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
        print(f"[run_zemosaic] Unable to import PySide6 for opening animation: {qt_err}")
        return

    app = QApplication.instance()
    owns_app = False
    if app is None:
        try:
            app = QApplication([sys.argv[0], "--zemosaic-opening"])
        except Exception as app_err:
            print(f"[run_zemosaic] Unable to create QApplication for opening animation: {app_err}")
            return
        owns_app = True
        try:
            app.setProperty("zemosaic_prelaunch_owner", True)
        except Exception:
            pass

    movie = QMovie(str(gif_path))
    if not movie.isValid():
        print(f"[run_zemosaic] Opening animation at {gif_path} is not a valid GIF.")
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
            print(f"[run_zemosaic] Unable to enforce single loop for animation: {err}")
    else:
        manual_loop_control = True
        print("[run_zemosaic] PySide6 build lacks QMovie.setLoopCount; animation may loop more than once.")

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

    # Ensure pending events are processed so the splash can appear promptly.
    try:
        app.processEvents()
    except Exception:
        pass

    # When we created a temporary QApplication, keep it alive so that the main
    # Qt backend can reuse it without blocking here.
    if owns_app:
        try:
            app.setProperty("zemosaic_prelaunch_owner", True)
        except Exception:
            pass


def main(argv=None):
    """Fonction principale pour lancer l'application ZeMosaic."""
    print("--- run_zemosaic.py: Entrée dans main() ---")

    if argv is None:
        argv = sys.argv[1:]

    _backend, cleaned_args, _backend_source = _determine_backend(argv)
    if cleaned_args != argv:
        sys.argv = [sys.argv[0], *cleaned_args]

    try:
        try:
            from zemosaic.zemosaic_gui_qt import run_qt_main
        except ImportError as qt_import_error:
            _notify_qt_backend_unavailable(qt_import_error)
            return 1

        print("[run_zemosaic] Launching ZeMosaic with the Qt backend.")
        _play_opening_gif_animation_once()
        exit_code = run_qt_main()
        return exit_code
    finally:
        # Nettoyage des ressources GPU pour éviter les erreurs CUDA à la fermeture.
        # Cette opération est effectuée inconditionnellement si cupy a été chargé.
        if CUPY_AVAILABLE and "cupy" in sys.modules:
            print("[run_zemosaic] Nettoyage des ressources GPU...")
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                print("[run_zemosaic] Ressources GPU nettoyées avec succès.")
            except Exception as e:
                print(f"[run_zemosaic] Erreur lors du nettoyage GPU : {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Lancement de ZeMosaic via run_zemosaic.py (__name__ == '__main__')...")
    main()
    print("ZeMosaic terminé (sortie de __main__).")
