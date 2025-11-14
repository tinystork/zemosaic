# zemosaic/run_zemosaic.py
"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import sys  # Ajout pour sys.path et sys.modules
import multiprocessing
import os
import platform
# import reproject # L'import direct ici n'est pas crucial, mais ne fait pas de mal
import tkinter as tk
from tkinter import messagebox  # Nécessaire pour la messagebox d'erreur critique

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
_package_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_package_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Essayer d'importer la classe GUI et la variable de disponibilité du worker
try:
    from zemosaic.zemosaic_gui import ZeMosaicGUI, ZEMOSAIC_WORKER_AVAILABLE
    print("--- run_zemosaic.py: Import de zemosaic_gui RÉUSSI ---")

    # Vérifier le module zemosaic_worker si la GUI dit qu'il est disponible
    if ZEMOSAIC_WORKER_AVAILABLE:
        try:
            # Importer le worker via le package pour que ses imports relatifs fonctionnent
            from zemosaic import zemosaic_worker
            print(
                f"DEBUG (run_zemosaic): zemosaic_worker chargé depuis: {zemosaic_worker.__file__}"
            )
            if 'zemosaic.zemosaic_worker' in sys.modules:
                print(
                    "DEBUG (run_zemosaic): sys.modules['zemosaic.zemosaic_worker'] pointe vers: "
                    f"{sys.modules['zemosaic.zemosaic_worker'].__file__}"
                )
            elif 'zemosaic_worker' in sys.modules:
                # Importe sous l'ancien nom si déjà présent
                print(
                    "DEBUG (run_zemosaic): module importé sous le nom 'zemosaic_worker', fichier: "
                    f"{sys.modules['zemosaic_worker'].__file__}"
                )
            else:
                print(
                    "DEBUG (run_zemosaic): zemosaic_worker n'est pas dans sys.modules après import direct (étrange)."
                )
        except ImportError as e_worker_direct:
            print(
                f"ERREUR (run_zemosaic): Échec de l'import direct de zemosaic_worker pour débogage: {e_worker_direct}"
            )
        except AttributeError:
            print(
                "ERREUR (run_zemosaic): zemosaic_worker importé mais n'a pas d'attribut __file__ (très étrange)."
            )

except ImportError as e:
    print(f"ERREUR CRITIQUE (run_zemosaic): Impossible d'importer ZeMosaicGUI depuis zemosaic_gui.py: {e}")
    print("  Veuillez vérifier que zemosaic_gui.py est présent et que toutes ses dépendances Python sont installées.")
    
    try:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Erreur de Lancement Fatale",
                             f"Impossible d'importer le module GUI principal (zemosaic_gui.py).\n"
                             f"Erreur: {e}\n\n"
                             "Veuillez vérifier les logs console pour plus de détails.")
        root_err.destroy()
    except Exception as tk_err:
        print(f"  Erreur Tkinter lors de la tentative d'affichage de la messagebox: {tk_err}")
    
    ZEMOSAIC_WORKER_AVAILABLE = False 
    ZeMosaicGUI = None

print("--- run_zemosaic.py: FIN DES IMPORTS ---")
print("DEBUG (run_zemosaic): sys.path complet:\n" + "\n".join(sys.path))
print("-" * 50)


def _notify_qt_backend_unavailable(error: Exception) -> None:
    """Inform the user that the Qt backend cannot be used."""

    print(
        "[run_zemosaic] Unable to load the Qt interface because the optional PySide6 "
        "dependency is missing."
    )
    print("[run_zemosaic] Install it with `pip install PySide6` to enable the Qt GUI.")
    print(f"[run_zemosaic] Original import error: {error}")
    print("[run_zemosaic] Falling back to the classic Tk interface.")

    try:
        root_warning = tk.Tk()
        root_warning.withdraw()
        messagebox.showwarning(
            "ZeMosaic Qt backend unavailable",
            (
                "The PySide6 dependency required for the Qt interface could not be imported.\n\n"
                "ZeMosaic will continue with the classic Tk interface instead.\n\n"
                "Install PySide6 with `pip install PySide6` if you wish to use the Qt GUI."
            ),
        )
        root_warning.destroy()
    except Exception as tk_error:
        print(f"[run_zemosaic] Unable to display Tk warning dialog: {tk_error}")


def _determine_backend(argv):
    """Determine which GUI backend should be used.

    The selection prioritises explicit command-line flags over the environment
    variable so that users can temporarily override their default preference.
    Use ``--qt-gui`` to force the Qt backend or ``--tk-gui`` to force the
    classic Tk interface regardless of :envvar:`ZEMOSAIC_GUI_BACKEND`.
    """

    requested_backend = os.environ.get("ZEMOSAIC_GUI_BACKEND", "tk")
    cleaned_args = []
    for arg in argv:
        if arg == "--qt-gui":
            requested_backend = "qt"
            continue
        if arg == "--tk-gui":
            requested_backend = "tk"
            continue
        cleaned_args.append(arg)

    backend = (requested_backend or "tk").strip().lower()
    if backend not in {"tk", "qt"}:
        print(
            f"[run_zemosaic] Backend '{requested_backend}' is not supported. "
            "Falling back to Tk."
        )
        backend = "tk"

    return backend, cleaned_args


def main(argv=None):
    """Fonction principale pour lancer l'application ZeMosaic."""
    print("--- run_zemosaic.py: Entrée dans main() ---")

    if argv is None:
        argv = sys.argv[1:]

    backend, cleaned_args = _determine_backend(argv)
    if cleaned_args != argv:
        sys.argv = [sys.argv[0], *cleaned_args]

    if backend == "qt":
        try:
            from zemosaic.zemosaic_gui_qt import run_qt_main
        except ImportError as qt_import_error:
            _notify_qt_backend_unavailable(qt_import_error)
            backend = "tk"
        else:
            print("[run_zemosaic] Launching ZeMosaic with the Qt backend.")
            return run_qt_main()

    # Vérification de sys.modules au début de main
    if 'zemosaic_worker' in sys.modules:
        print(f"DEBUG (main): 'zemosaic_worker' EST dans sys.modules. Chemin: {sys.modules['zemosaic_worker'].__file__}")
    else:
        print("DEBUG (main): 'zemosaic_worker' N'EST PAS dans sys.modules au début de main.")


    if not ZeMosaicGUI: 
        print("ZeMosaic ne peut pas démarrer car la classe GUI (ZeMosaicGUI) n'a pas pu être chargée.")
        return

    if not ZEMOSAIC_WORKER_AVAILABLE:
        print("Avertissement (run_zemosaic main): Le module worker (zemosaic_worker.py) n'est pas disponible ou n'a pas pu être importé correctement par zemosaic_gui.py.")
        
        root_temp_err_worker = tk.Tk()
        root_temp_err_worker.withdraw() 
        messagebox.showerror("Erreur de Lancement Critique (Worker)",
                             "Le module 'zemosaic_worker.py' est introuvable ou contient une erreur d'importation.\n"
                             "L'application ZeMosaic ne peut pas démarrer correctement.\n\n"
                             "Veuillez vérifier les logs console pour plus de détails.")
        root_temp_err_worker.destroy()
        return 

    print("DEBUG (main): ZEMOSAIC_WORKER_AVAILABLE est True. Tentative de création de l'interface graphique.")
    if backend != "qt":
        print("[run_zemosaic] Launching ZeMosaic with the Tk backend.")

    root = tk.Tk()
    app = ZeMosaicGUI(root)
    root.mainloop()
    print("--- run_zemosaic.py: mainloop() terminée ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Lancement de ZeMosaic via run_zemosaic.py (__name__ == '__main__')...")
    main()
    print("ZeMosaic terminé (sortie de __main__).")