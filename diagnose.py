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
"""
ZeMosaic — GUI de diagnostic (macOS/Windows/Linux)
- Vérifie les dépendances clés
- Teste l'import de zemosaic_worker (direct et package)
- Affiche un rapport clair avec options de copie/sauvegarde
- Bascule automatiquement en mode console si Tkinter indisponible
"""

import sys, os, importlib, traceback
from pathlib import Path
from datetime import datetime

# ---------------------------
# Paramètres et constantes
# ---------------------------
RECOMMENDED_PY = "3.11–3.12"
MODULES_CORE = [
    # NumPy/SciPy stack
    "numpy", "scipy",
    # Astro stack
    "astropy", "reproject", "photutils", "astroalign",
    # Vision
    "cv2",
    # I/O / large arrays
    "zarr",
    # Optionnel GPU
    "cupy",
    # GUI / images
    "tkinter", "PIL",
]
PROJECT_HINT_FILES = ["run_zemosaic.py", "zemosaic_worker.py", "zemosaic_gui.py"]


def run_checks(verbose=False):
    lines = []
    add = lines.append

    add("=== ZeMosaic Environment Diagnostic ===")
    add(f"Timestamp       : {datetime.now().isoformat(timespec='seconds')}")
    add(f"Python exe      : {sys.executable}")
    add(f"Python version  : {sys.version.split()[0]}")
    add(f"Platform        : {sys.platform}")
    add(f"CWD             : {os.getcwd()}")
    add("")

    # Conseil version Python
    try:
        maj, minr = sys.version_info.major, sys.version_info.minor
        if not (maj == 3 and 11 <= minr <= 12):
            add(f"⚠️ Recommandation : utilisez Python {RECOMMENDED_PY} pour des wheels stables (actuel: {maj}.{minr}).")
            add("")
    except Exception:
        pass

    # Détection du dossier projet (heuristique)
    cwd = Path.cwd()
    project_root = cwd
    for guess in [cwd, *cwd.parents]:
        if any((guess / name).exists() for name in PROJECT_HINT_FILES):
            project_root = guess
            break
    add(f"Projet détecté  : {project_root}")
    add("")

    # 1) Dépendances
    add("[1] Vérification des dépendances :")
    missing = []
    present = []
    for mod in MODULES_CORE:
        try:
            spec = importlib.util.find_spec(mod)
            if spec is None:
                missing.append(mod)
                add(f"  ❌ {mod:<10} : manquant")
            else:
                present.append(mod)
                add(f"  ✅ {mod:<10} : OK")
        except Exception as e:
            missing.append(mod)
            add(f"  ❌ {mod:<10} : erreur d'inspection -> {e.__class__.__name__}: {e}")
            if verbose:
                add("".join(traceback.format_exc(limit=2)).rstrip())
    add("")

    # 2) Test import worker
    add("[2] Test d'import de zemosaic_worker :")
    # Assure que le parent du projet est dans sys.path (utile pour `python -m` ou exécution directe)
    parent = str(project_root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    ok_worker = False
    err_msgs = []

    # a) import direct si le fichier est à la racine
    try:
        with pushd(project_root):
            import importlib as _il
            if (project_root / "zemosaic_worker.py").exists():
                _il.invalidate_caches()
                __import__("zemosaic_worker")
                add("  ✅ Import direct `zemosaic_worker` : RÉUSSI")
                ok_worker = True
            else:
                add("  ℹ️  Import direct ignoré (fichier zemosaic_worker.py non trouvé à la racine).")
    except Exception as e1:
        err = f"  ⚠️  Échec import direct : {e1.__class__.__name__}: {e1}"
        err_msgs.append(err); add(err)
        if verbose:
            add("".join(traceback.format_exc()).rstrip())

    # b) import package zemosaic.zemosaic_worker
    if not ok_worker:
        try:
            import importlib as _il
            _il.invalidate_caches()
            __import__("zemosaic.zemosaic_worker")
            add("  ✅ Import package `zemosaic.zemosaic_worker` : RÉUSSI")
            ok_worker = True
        except Exception as e2:
            err = f"  ❌ Import package échoué : {e2.__class__.__name__}: {e2}"
            err_msgs.append(err); add(err)
            if verbose:
                add("".join(traceback.format_exc()).rstrip())

    add("")
    add("=== Résumé ===")
    if missing:
        add(f"{len(missing)} module(s) manquant(s) : {', '.join(missing)}")
        add("➡️  Installez-les dans votre environnement courant :")
        add(f"    pip install {' '.join(missing)}")
    else:
        add("🎉 Toutes les dépendances essentielles sont présentes.")

    if ok_worker:
        add("🧩 Le module worker se charge correctement.")
    else:
        add("🚫 Le module worker ne s'est pas chargé.")
        add("   Indices :")
        add("   - Vérifiez la version de Python (3.11–3.12 recommandé).")
        add("   - Assurez-vous d'exécuter depuis le dossier parent et/ou utilisez :")
        add("       python -m zemosaic.run_zemosaic")
        add("   - Si des modules manquent ci-dessus, installez-les puis réessayez.")

    add("\n=== Fin du diagnostic ===")
    return "\n".join(lines)


# Utilitaire pour changer de dossier temporairement
class pushd:
    def __init__(self, new_dir):
        self.new = str(new_dir)
        self.old = os.getcwd()
    def __enter__(self):
        os.chdir(self.new)
    def __exit__(self, exc_type, exc, tb):
        os.chdir(self.old)


# ---------------------------
# GUI (Tkinter) ou fallback
# ---------------------------
def main_gui():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
    except Exception:
        # Fallback console si Tkinter indispo
        print("Tkinter indisponible — bascule en mode console.\n")
        print(run_checks(verbose=True))
        return

    root = tk.Tk()
    root.title("ZeMosaic — Diagnostic")
    root.geometry("820x560")

    # Frame top (actions)
    top = ttk.Frame(root, padding=8)
    top.pack(side="top", fill="x")

    verbose_var = tk.BooleanVar(value=False)

    ttk.Label(top, text=f"Python {sys.version.split()[0]}  |  Recommandé : {RECOMMENDED_PY}").pack(side="left")
    ttk.Checkbutton(top, text="Verbose", variable=verbose_var).pack(side="left", padx=10)

    def do_run():
        report = run_checks(verbose=verbose_var.get())
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", report)
        text.configure(state="disabled")

    def do_copy():
        try:
            root.clipboard_clear()
            root.clipboard_append(text.get("1.0", "end-1c"))
            messagebox.showinfo("Copié", "Rapport copié dans le presse-papiers.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de copier : {e}")

    def do_save():
        path = filedialog.asksaveasfilename(
            title="Enregistrer le rapport",
            defaultextension=".txt",
            filetypes=[("Texte", "*.txt"), ("Tous fichiers", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text.get("1.0", "end-1c"))
            messagebox.showinfo("Enregistré", f"Rapport sauvegardé : {path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'enregistrer : {e}")

    ttk.Button(top, text="Lancer le test", command=do_run).pack(side="right")
    ttk.Button(top, text="Enregistrer…", command=do_save).pack(side="right", padx=6)
    ttk.Button(top, text="Copier", command=do_copy).pack(side="right")

    # Zone texte
    frm = ttk.Frame(root, padding=(8, 0, 8, 8))
    frm.pack(fill="both", expand=True)
    text = tk.Text(frm, wrap="word")
    text.pack(side="left", fill="both", expand=True)
    text.insert("1.0", "Cliquez sur « Lancer le test » pour démarrer le diagnostic…")
    text.configure(state="disabled")
    scroll = ttk.Scrollbar(frm, command=text.yview)
    scroll.pack(side="right", fill="y")
    text.configure(yscrollcommand=scroll.set)

    root.mainloop()


if __name__ == "__main__":
    main_gui()
