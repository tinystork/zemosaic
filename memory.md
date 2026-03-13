# Memory (compacted)

## 2026-01-28 / 2026-01-29 — Windows PyInstaller build (GPU + Shapely + SEP)

### Problem
Packaged Windows build had failures not seen in non-compiled runs:
- GPU path falling back to CPU (`phase5_using_cpu`) due CuPy import/runtime issues.
- Phase 4 failure around `find_optimal_celestial_wcs()` in packaged mode.
- Additional packaged-only alignment/import errors.

### Root causes confirmed
- Frozen DLL search path in packaged runtime needed stronger setup.
- CuPy packaging/dependencies were incomplete early on (notably missing `fastrlock`).
- Shapely packaging was incomplete (`shapely._geos` missing in packaged runtime).
- `sep_pjw` expected top-level `_version` that was not always bundled.

### Changes made
- `pyinstaller_hooks/rthook_zemosaic_sys_path.py`
  - Added frozen DLL-path setup for `sys._MEIPASS` and discovered `*.libs`/DLL folders.
  - Added safer `os.add_dll_directory()` handling with `WinError 206` mitigation.
- `ZeMosaic.spec`
  - Added stronger hidden-import/data handling for CuPy/fastrlock/Shapely/SEP-related pieces.
  - Added explicit `_version.py` bundling path.
- `pyinstaller_hooks/hook-shapely.py`
  - Added hidden import support for `shapely._geos`.
- Runtime diagnostics/logging improved in key modules to expose packaged-only failures.

### Validation summary
- Shapely `WinError 206` path issue mitigated.
- Missing packaged dependencies were identified and patched iteratively.
- Some packaged logs later showed GPU detection/use recovery, but packaged behavior remained more fragile than non-compiled runs in parts of the workflow.

### Remaining caution
For packaging work, preserve:
- frozen DLL search-path setup,
- explicit hidden imports for binary dependencies,
- detailed runtime logging in packaged mode.

---

## 2026-03-12 — Qt filter dialog usability fix (`zemosaic_filter_gui_qt.py`)

### Topic
Start/OK button could become unreachable on smaller screens / high-DPI setups.

### Problem
In the Qt filter dialog, the right-side controls and button box were in the same non-scrollable column, so the bottom actions could fall outside the visible area.

### Root cause
- Right controls column was not inside a `QScrollArea`.
- Saved geometry was restored as-is without clamping to current screen `availableGeometry()`.

### Changes made
- `_build_ui()`:
  - Added `QScrollArea` for the right panel (`setWidgetResizable(True)`).
  - Kept preview group as left splitter widget.
  - Moved right controls into a dedicated inner container set as the scroll area widget.
  - Moved `QDialogButtonBox` outside scrollable content and anchored it in the main dialog layout under the splitter.
  - Added a trailing stretch in the scrollable controls layout for natural packing.
- `_apply_saved_window_geometry()`:
  - Added safe clamping of restored `(x, y, w, h)` against current screen `availableGeometry()`.
  - Added safe screen lookup fallback order: `screenAt(center)` -> `self.screen()` -> `primaryScreen()`.
  - Preserved fail-safe behavior if screen lookup/clamp logic cannot run.

### Validation performed
- Static checks:
  - `python3 -m py_compile zemosaic_filter_gui_qt.py` passed.
- Code-level sanity checks:
  - OK/Cancel signal wiring (`accepted -> accept`, `rejected -> reject`) preserved.
  - Existing preview/stream/selection-related wiring untouched in the patch scope.
  - Splitter remains intact with preview left and controls right.

### Remaining risk / follow-up
- Manual GUI verification on constrained-height/high-DPI display was not run in this headless session.
- Smallest safe next step: launch the dialog in a constrained-height scenario and confirm right-panel scrolling + always-visible OK/Cancel.

---

## 2026-03-12 — Packaging/docs alignment for GPU vs CPU-only builds

### Topic
Clarified how Windows/macOS/Linux packaged builds decide whether CuPy/CUDA support is included, and aligned the helper scripts / installer documentation with the current build layout.

### Problem
- Build/release docs did not clearly distinguish:
  - `requirements.txt` (current working GPU-enabled dependency set, including `cupy-cuda12x`)
  - `requirements_no_gpu.txt` (new CPU-only dependency set for smaller artifacts)
- Build helpers and installer assumptions needed to match the real packaging flow.
- `zemosaic_installer.iss` previously targeted obsolete `compile\...` paths instead of the current `dist\ZeMosaic\...` output layout.

### Changes made
- Added `requirements_no_gpu.txt`
  - mirrors the base dependency set
  - intentionally excludes CuPy so a smaller CPU-only packaged build can be produced
- Updated `compile/compile_zemosaic._win.bat`
  - default requirements file is again `requirements.txt`
  - supports overriding via `ZEMOSAIC_REQUIREMENTS_FILE`
  - message now states GPU support depends on the chosen requirements file
- Updated `compile/build_zemosaic_posix.sh`
  - default requirements file is again `requirements.txt`
  - supports overriding via `ZEMOSAIC_REQUIREMENTS_FILE`
  - installs `pyinstaller-hooks-contrib`
  - cleans `build/` and `dist/`
  - message now states GPU support depends on the chosen requirements file
- Updated `zemosaic_installer.iss`
  - installer now packages `dist\ZeMosaic\*`
  - `Icons` / `Run` point to `{app}\ZeMosaic.exe`
  - comments explicitly state that Inno Setup does not choose the CUDA package; it only packages whatever build already exists in `dist\ZeMosaic`
- Updated `README.md`
  - documents the default GPU-enabled path using `requirements.txt`
  - documents CPU-only builds using `requirements_no_gpu.txt`
  - explains that `.iss` does not select CUDA/CuPy version
  - adds Windows GitHub release guidance (publish zipped `dist\ZeMosaic` as a release asset, do not commit `dist/`)

### Validation performed
- `bash -n compile/build_zemosaic_posix.sh` passed.
- Static review confirmed Windows helper / POSIX helper / README are now consistent on:
  - default = `requirements.txt`
  - smaller package option = `requirements_no_gpu.txt`
  - installer packages the already-built output

### Important packaging note
- Current intended behavior remains:
  - use `requirements.txt` for the existing GPU-enabled build path (`cupy-cuda12x`)
  - use `requirements_no_gpu.txt` only when a smaller CPU-only package is wanted
- `zemosaic_installer.iss` is ignored by Git in this repo (`.gitignore` has `*.iss`), so versioning it requires `git add -f zemosaic_installer.iss` or changing ignore rules.

---

## 2026-03-12 — Future idea: installer with GPU/CuPy auto-detection

### Topic
Exploration only, no mission started yet.

### User intent
- Keep the current working GPU path based on `requirements.txt` / `cupy-cuda12x`.
- Consider a future Windows installer able to detect the target machine and install/download the appropriate GPU support automatically.
- Defer implementation for now.

### Assessment given
- This is considered viable, but not a small change.
- Estimated scope:
  - minimal viable approach: high complexity
  - robust/maintainable approach: very high complexity

### Recommended architecture discussed
- Prefer a CPU-only base installer plus an optional GPU enablement step.
- Detect:
  - Windows x64 environment
  - NVIDIA GPU presence
  - driver / CUDA compatibility level
- Then either:
  - download a prebuilt GPU add-on from GitHub Releases, or
  - install the matching CuPy package dynamically
- Recommended direction was to avoid a fully dynamic Inno Setup-only solution at first, and instead use:
  - a simple installer
  - plus a post-install/bootstrap GPU activation step with clear fallbacks

### Why this was deferred
- Requires coordinated changes across:
  - installer/bootstrap flow
  - release packaging strategy
  - GPU compatibility detection logic
  - runtime fallback behavior
  - documentation and testing matrix
- User asked to postpone this work and possibly revisit it later.


### 2026-03-13 09:53 — Iteration 2
- Scope: S0/B2 only (headless scope lock), no code migration.
- In scope: définir les chemins headless officiellement validés pour cette mission + lister les non-supportés.
- Out of scope: suppression Tk, changements runtime, config strategy B3, S1+.
- Files changed: followup.md (B2 checkboxes), memory.md (journalisation).
- Tests run:
  - `grep -nE "argparse|ArgumentParser|--no-gui|--headless|--cli|--qt-gui|--tk-gui|if __name__ == '__main__'|main\(" run_zemosaic.py`
  - `grep -nE "argparse|ArgumentParser|--no-gui|--headless|--cli|if __name__ == '__main__'|def main\(" zemosaic_worker.py | head`
  - `ls -la tests && grep -RInE "zemosaic_config|zemosaic_worker|headless|import" tests`
  - `grep -nE "headless|CLI|command line|zemosaic_worker|run_zemosaic|--config|input_folder|output_folder|tk-gui|qt-gui" README.md`
- Proof:
  - `run_zemosaic.py` est un launcher GUI (Qt par défaut, fallback Tk existant), pas un mode headless explicite (`--headless/--no-gui` absents).
  - `zemosaic_worker.py` expose un point d’entrée CLI argparse (preuve d’un chemin exécutable non-GUI, même si non officiellement documenté comme frontend).
  - Les tests repo importent directement `zemosaic_worker` (ex: `tests/test_empty_master_tile_guard.py`), confirmant un chemin headless d’import pertinent.
  - Gates roadmap S2/S5 citent explicitement: `import zemosaic_config` et `import zemosaic_worker`.
- Decisions:
  - **Validated headless paths (scope fermé S0/B2):**
    1) `python -c "import zemosaic_config"`
    2) `python -c "import zemosaic_worker"`
    3) exécution des tests unitaires qui importent `zemosaic_worker` sans lancer GUI.
  - **Non-supported headless paths (pour cette mission):**
    - toute exécution Tk GUI legacy (`zemosaic_gui.py`, `zemosaic_filter_gui.py`, `run_zemosaic.py --tk-gui`)
    - outils annexes standalone Tk (`lecropper.py`, `zequalityMT.py`, `zewcscleaner.py`, `diagnose.py`)
    - tout chemin nécessitant interaction graphique Tk (file dialogs / messagebox).
- Blockers:
  - Aucun blocage pour passer à B3.
- Next unchecked item: B3 — Audit direct Tk imports in `zemosaic_config.py`.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (scope lock only).
- Validated headless scope changed or stayed unchanged: changed (now explicitly locked).


### 2026-03-13 09:53 — Iteration 3
- Scope: S0/B3 only (`zemosaic_config.py` strategy), no runtime code changes.
- In scope: audit imports Tk directs + couplages indirects; choisir une stratégie canonique pour S2/S3.
- Out of scope: implémentation de la stratégie, suppression Tk, S1+.
- Files changed: followup.md (B3 checkboxes), memory.md (journalisation).
- Tests run:
  - `grep -nE "import tkinter|from tkinter|filedialog|messagebox|fd\.|mb\.|def prompt|def ask|preferred_gui_backend|load_config|save_config" zemosaic_config.py`
  - `sed -n '1,140p' zemosaic_config.py`
  - `sed -n '500,760p' zemosaic_config.py`
- Proof:
  - Import direct Tk présent en module-level sous `try/except`:
    - `import tkinter.filedialog as fd`
    - `import tkinter.messagebox as mb`
  - Valeur par défaut backend toujours Tk: `"preferred_gui_backend": "tk"`.
  - Couplage indirect: `load_config()` / `save_config()` appellent `mb.showwarning/showerror` en cas d’erreur (avec fallback console si `mb is None`).
  - Couplage UI legacy explicite: `ask_and_set_astap_path()` / `ask_and_set_astap_data_dir_path()` utilisent `fd`/`mb` (dialogs Tk).
- Decisions:
  - Stratégie B3 choisie: **split core/legacy helpers**.
  - Direction retenue pour S2/S3: garder un noyau config import-safe Tk-free pour runtime officiel/headless validé; isoler les prompts Tk (ASTAP dialog/messagebox) dans un helper legacy hors chemin officiel.
- Blockers:
  - Aucun blocage pour passer à B4.
- Next unchecked item: B4 — Confirm `lecropper` classification as annex / standalone tool.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit/strategy only).
- Validated headless scope changed or stayed unchanged: stayed unchanged (déjà verrouillé en B2).


### 2026-03-13 09:53 — Iteration 4
- Scope: S0/B4 only (statut `lecropper`), sans migration.
- In scope: confirmer classification annexe/standalone, confirmer hors-scope port Qt, tracer exigence de découplage S2.
- Out of scope: implémenter le découplage, modifier worker/runtime, S1+.
- Files changed: followup.md (B4 checkboxes), memory.md (journalisation).
- Tests run:
  - `grep -nE "lecropper|annex|standalone|phase séparée|out-of-scope|decoupl|découpl" ROADMAP_REMOVE_TKINTER.md | head`
  - `grep -nE "lecropper|annex|standalone|out of scope|decouple|decoupling" agent.md`
  - `grep -nE "import lecropper|from lecropper" zemosaic_worker.py | head`
- Proof:
  - Roadmap canon A confirme `lecropper` = outil annexe standalone; port Qt en phase séparée (post-release).
  - `agent.md` confirme explicitement: annexe, hors-scope pour port Qt maintenant, découplage requis du runtime officiel/headless validé.
  - `zemosaic_worker.py` prouve dépendance actuelle (`import lecropper`, `from lecropper import detect_autocrop_rgb`).
- Decisions:
  - Classification confirmée: **`lecropper` = annex / standalone tool**.
  - Port Qt `lecropper` confirmée **out-of-scope** pour cette mission S0→S5.
  - Exigence S2 verrouillée: suppression dépendance directe/indirecte à `lecropper` sur chemins officiels + headless validés.
- Blockers:
  - Aucun blocage pour passer à B5.
- Next unchecked item: B5 — Build initial Qt parity matrix.
- Lecropper status changed or not: unchanged (confirmed as annex/standalone).
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit/decision only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 09:53 — Iteration 5
- Scope: S0/B5 only (initial Qt parity matrix), no code migration.
- In scope: établir une matrice initiale de parité pour workflows officiels Qt-only/Tk-retirement.
- Out of scope: correction des gaps/blocants (S1/S2), refactor hors scope.
- Files changed: followup.md (B5 checkboxes), memory.md (journalisation + matrice).
- Tests run:
  - `grep -nE "Qt preview|Tk stable|backend|preferred_gui_backend|..." zemosaic_gui_qt.py`
  - `grep -nE "Tk interface instead|PySide6|Qt|Tk|backend|coexist|preview|install" zemosaic_filter_gui_qt.py`
  - `grep -nE -- "--tk-gui|--qt-gui|fallback|Tk backend|Qt backend|preferred_gui_backend|ZEMOSAIC_GUI_BACKEND" run_zemosaic.py`
- Proof:
  - `zemosaic_gui_qt.py` expose un groupe "Preferred GUI backend" avec options `tk` (stable) / `qt` (preview).
  - `zemosaic_filter_gui_qt.py` contient le message: "Install PySide6 or use the Tk interface instead."
  - `run_zemosaic.py` conserve `--tk-gui` et fallback automatique vers Tk si Qt indisponible.
- Decisions:
  - Matrice de parité initiale (S0):
    - **OK**: frontend Qt existe et chemin nominal Qt est présent (`run_qt_main`).
    - **GAP**: UI Qt expose encore le choix backend Tk/Qt + wording "Qt preview / Tk stable".
    - **GAP**: messaging filter Qt oriente encore vers Tk en fallback utilisateur.
    - **BLOCKING**: launcher officiel garde fallback Tk (`--tk-gui` + auto fallback).
    - **BLOCKING**: config default backend reste `tk` (vu en B3).
    - **BLOCKING**: dépendance worker→`lecropper` encore active (vu en B4).
    - **OUT-OF-SCOPE (P0 immédiat)**: port Qt de `lecropper` et purge totale des annexes Tk.
- Blockers:
  - Aucun blocage d’audit; blocants de migration identifiés pour S1/S2.
- Next unchecked item: B6 — Record S0 proof in `memory.md` and close S0 when explicit.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit matrix only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 09:53 — Iteration 6
- Scope: S0/B6 closeout only.
- In scope: consolider les preuves S0, vérifier critères explicites scope/headless/config strategy, fermer S0.
- Out of scope: démarrage S1, modifications runtime/config/UI.
- Files changed: agent.md (S0 checkbox), followup.md (B6 checkboxes), memory.md (journalisation).
- Tests run:
  - revue des entrées S0 déjà tracées en memory (B1→B5)
  - vérification checklist `followup.md` section B complète
- Proof:
  - B1 inventaire+classification Tk: fait et journalisé.
  - B2 scope headless validé: défini explicitement + non-supportés listés.
  - B3 stratégie config choisie: split core/legacy helpers.
  - B4 statut `lecropper`: annex/standalone confirmé, port Qt hors scope, découplage S2 requis.
  - B5 matrice parité initiale: OK/gap/blocking/out-of-scope établie.
- Decisions:
  - **S0 explicitement clos** (critères roadmap satisfaits).
  - `lecropper` status remains unchanged (annex/standalone).
- Blockers:
  - Aucun pour entrer en S1 quand demandé.
- Next unchecked item: C1 — Verify GUI startup workflow parity.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S0 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged (already locked in B2).


### 2026-03-13 09:57 — Iteration 7
- Scope: S1/C1 only — **Verify GUI startup workflow parity**.
- In scope: comparer le flux de démarrage officiel Qt vs fallback Tk existant, sans modifier le code.
- Out of scope: config load/save parity (C1.2/C4), logs parity (C1.3), shutdown (C1.4), filter/grid/SDS (C1.5+), S2+.
- Files changed: followup.md (C1 startup checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "def run_qt_main|...|worker" zemosaic_gui_qt.py | head`
  - `sed -n '6388,6465p' zemosaic_gui_qt.py`
  - `sed -n '500,620p' run_zemosaic.py`
- Proof:
  - Chemin Qt: `run_zemosaic.py` tente `from zemosaic.zemosaic_gui_qt import run_qt_main`; si succès, lance Qt.
  - `run_qt_main()` crée/réutilise `QApplication`, instancie `ZeMosaicQtMainWindow`, `show()`, puis `app.exec()`.
  - En cas d’échec import Qt, launcher bascule automatiquement vers Tk (`backend = "tk"`) + warning utilisateur.
  - Chemin Tk: vérifie/charge worker avant l’ouverture GUI et bloque le démarrage avec `messagebox.showerror` si worker indisponible.
  - Chemin Qt: l’app peut s’ouvrir même si backend worker indisponible; erreur reportée plus tard via message Qt lors du lancement process (non bloquant au démarrage fenêtre).
- Decisions:
  - Verdict C1.1: **parité de démarrage partielle / gap**.
  - Gap principal: comportement pré-check worker différent (Tk bloque avant UI, Qt laisse ouvrir UI puis échoue au run).
- Blockers:
  - Aucun blocage pour poursuivre C1.2.
- Next unchecked item: C1 — Verify config load/save parity.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S1 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 09:58 — Iteration 8
- Scope: S1/C1 only — **Verify config load/save parity**.
- In scope: comparer charge/sauvegarde config Qt vs Tk legacy sur le workflow officiel, sans modifier le code.
- Out of scope: logs parity (C1.3), shutdown parity (C1.4), filter/grid/SDS (C1.5+), S2+.
- Files changed: followup.md (C1 config checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "load_config\(|save_config\(|..." zemosaic_gui_qt.py`
  - `grep -nE "load_config\(|save_config\(|..." zemosaic_gui.py`
  - `sed -n '4088,4295p' zemosaic_gui_qt.py`
  - `sed -n '300,380p' zemosaic_gui.py`
  - `sed -n '4520,4625p' zemosaic_gui.py`
  - `sed -n '5858,5908p' zemosaic_gui_qt.py`
  - `grep -nE "WM_DELETE_WINDOW|_on_closing|save_config\(" zemosaic_gui.py`
- Proof:
  - Qt charge via `zemosaic_config.load_config()` dans `_load_config()` puis merge avec defaults internes.
  - Tk charge aussi via `zemosaic_config.load_config()` à l'init.
  - Qt sauvegarde avant run (`_save_config()` dans `_start_processing`) et à la fermeture (`closeEvent`).
  - Tk sauvegarde avant run (`zemosaic_config.save_config(self.config)` dans `_start_processing`) et sur certains changements (langue/backend), mais pas systématiquement à la fermeture.
  - Qt sérialise un snapshot JSON-safe et conserve davantage l’état persisté (keys connues + snapshot), là où Tk écrit l’état courant `self.config`.
- Decisions:
  - Verdict C1.2: **parité partielle / gap**.
  - Parité de base OK (load/save présents des deux côtés), mais persistance plus robuste/cadrée côté Qt; comportements de sauvegarde non strictement identiques.
- Blockers:
  - Aucun blocage pour poursuivre C1.3.
- Next unchecked item: C1 — Verify logs / feedback parity.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S1 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 9
- Scope: S1/C1 only — **Verify logs / feedback parity**.
- In scope: comparer logs/progress/feedback utilisateur Qt vs Tk sur le workflow officiel.
- Out of scope: shutdown behavior (C1.4), filter/grid/SDS parity (C1.5+), S2+.
- Files changed: followup.md (C1 logs checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "_append_log|qt_log_|QMessageBox|progress|eta|phase" zemosaic_gui_qt.py`
  - `grep -nE "_log_message|messagebox|progress|eta|phase" zemosaic_gui.py`
  - `sed -n '4328,4415p' zemosaic_gui_qt.py`
  - `sed -n '5118,5205p' zemosaic_gui_qt.py`
  - `sed -n '5460,5575p' zemosaic_gui_qt.py`
  - `sed -n '4840,5015p' zemosaic_gui.py`
- Proof:
  - Qt: `_append_log()` avec niveaux normalisés/prefixes + surlignage GPU; widgets dédiés progress/ETA/phase + messages fin/cancel/erreur via `QMessageBox`.
  - Tk: `_log_message()` avec niveaux/tags couleur, progression+ETA+phase, feedback fin/cancel/erreur via `messagebox`.
  - Les deux flux gèrent annulation sans dialogue d’erreur dur, et succès avec proposition d’ouverture du dossier de sortie.
- Decisions:
  - Verdict C1.3: **parité globale OK avec gaps mineurs de présentation**.
  - Gap mineur: style d’affichage des logs différent (QPlainTextEdit/prefixes Qt vs Text/tags Tk), sans impact fonctionnel P0.
- Blockers:
  - Aucun blocage pour C1.4.
- Next unchecked item: C1 — Verify clean shutdown behavior.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 10
- Scope: S1/C1 only — **Verify clean shutdown behavior**.
- In scope: comparer le flux de fermeture Qt vs Tk (arrêt worker, persistance, cleanup).
- Out of scope: filter/grid/SDS parity détaillée (C1.5+), S2+.
- Files changed: followup.md (C1 shutdown checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "closeEvent|_save_config|worker_controller.stop|cupy|plt.close" zemosaic_gui_qt.py`
  - `grep -nE "_on_closing|worker_process.terminate|_stop_gui_chrono|root.destroy" zemosaic_gui.py`
  - `sed -n '5778,5822p' zemosaic_gui_qt.py`
  - `sed -n '5004,5040p' zemosaic_gui.py`
- Proof:
  - Qt `closeEvent`: enregistre état UI, collecte config, `save_config`, stop worker si actif, cleanup GPU (CuPy), cleanup matplotlib, puis fermeture.
  - Tk `_on_closing`: confirmation si traitement actif, terminate worker, stop chrono, destroy fenêtre.
  - Différence: Qt inclut un cleanup de sortie plus complet (persist + GPU + matplotlib) que Tk.
- Decisions:
  - Verdict C1.4: **parité fonctionnelle OK avec gap de robustesse** (Qt plus propre à l’arrêt).
- Blockers:
  - Aucun blocage pour C1.5.
- Next unchecked item: C1 — Verify filter workflow if part of official frontend.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 11
- Scope: S1/C1 only — **Verify filter workflow if part of official frontend**.
- In scope: vérifier si le filter fait partie du frontend officiel et comparer le flux Qt/Tk.
- Out of scope: grid/SDS dédiés (C1.6/C1.7), S2+.
- Files changed: followup.md (C1 filter checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "filter|skip_filter_ui|launch_filter|..." zemosaic_gui_qt.py`
  - `grep -nE "filter|skip_filter_ui|launch_filter|..." zemosaic_gui.py`
  - `sed -n '5838,5955p' zemosaic_gui_qt.py`
  - `sed -n '4210,4455p' zemosaic_gui.py`
- Proof:
  - Qt: bouton Filter + prompt pré-run + dialog Qt (`zemosaic_filter_gui_qt`) + passage des overrides/filtered items au worker (`skip_filter_ui`, `filter_overrides`, `filtered_header_items`).
  - Tk: flux équivalent avec `zemosaic_filter_gui` + prompt pré-run + propagation overrides/items au worker.
  - Annulation du filter: dans les deux cas, pas de lancement worker si l’utilisateur annule explicitement le dialogue.
- Decisions:
  - Le filter est **partie officielle/relevante** du workflow frontend.
  - Verdict C1.5: **parité fonctionnelle globalement OK** (implémentations différentes mais comportement attendu aligné).
- Blockers:
  - Aucun blocage pour C1.6.
- Next unchecked item: C1 — Verify Grid mode path if official.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 12
- Scope: S1/C1 only — **Verify Grid mode path if official**.
- In scope: vérifier que le chemin Grid/global-coadd est bien officiel et comparer Qt/Tk.
- Out of scope: SDS détaillé (C1.7), S2+.
- Files changed: followup.md (C1 grid checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "grid|phase4_grid|final_assembly_method|reproject|incremental|global_coadd" zemosaic_gui_qt.py`
  - `grep -nE "grid|phase4_grid|final_assembly_method|reproject|incremental|global_coadd" zemosaic_gui.py`
  - `sed -n '2070,2188p' zemosaic_gui_qt.py`
  - `sed -n '2040,2148p' zemosaic_gui.py`
- Proof:
  - Qt expose explicitement `final_assembly_method` (reproject_coadd / incremental) et traite les événements `phase4_grid`/`p4_global_coadd_*`.
  - Tk expose aussi `final_assembly_method` (reproject_coadd / incremental) et gère `phase4_grid`/`p4_global_coadd_*`.
  - Le chemin Grid/global-coadd est donc présent des deux côtés dans le flux officiel.
- Decisions:
  - Grid path = **official/relevant**.
  - Verdict C1.6: **parité fonctionnelle OK** au niveau des chemins et signaux principaux.
- Blockers:
  - Aucun blocage pour C1.7.
- Next unchecked item: C1 — Verify SDS path if official/relevant.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 13
- Scope: S1/C1 only — **Verify SDS path if official/relevant**.
- In scope: vérifier pertinence officielle SDS et parité de flux Qt/Tk.
- Out of scope: implémentation de corrections S2+, refactor.
- Files changed: followup.md (C1 SDS checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "sds_mode|SDS|seestar|_sds_" zemosaic_gui_qt.py`
  - `grep -nE "sds_mode|SDS|seestar|_sds_" zemosaic_gui.py`
  - `sed -n '1428,1490p' zemosaic_gui_qt.py`
  - `sed -n '6118,6185p' zemosaic_gui_qt.py`
  - `sed -n '3928,3975p' zemosaic_gui.py`
- Proof:
  - Qt expose des contrôles SDS (`sds_mode_default`, seuil SDS, flags Seestar) et synchronise ces choix vers le filter Qt.
  - Tk contient aussi une gestion SDS runtime (détection SDS, progression phase SDS, ETA/phase SDS).
  - Les deux GUIs gèrent les événements `phase4_grid` / global coadd en mode SDS.
- Decisions:
  - SDS path = **official/relevant**.
  - Verdict C1.7: **parité fonctionnelle globalement OK** (écarts d’UI non bloquants à ce stade).
- Blockers:
  - Aucun incident sur C1.
- Next unchecked item: C2 — Remove or plan removal of “Qt preview” wording.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 14
- Scope: S1/C2 only — **Remove or plan removal of “Qt preview” wording**.
- In scope: audit wording + plan de retrait (sans modification source).
- Out of scope: implémentation du retrait (S2/D3), refactor.
- Files changed: followup.md (C2.1 checkbox), memory.md (plan + preuves).
- Tests run:
  - `grep -nE "Qt GUI \(preview\)|backend_option_qt|preferred_gui_backend" zemosaic_gui_qt.py zemosaic_gui.py`
- Proof:
  - Wording localisé trouvé: `backend_option_qt = "Qt GUI (preview)"` (Tk et Qt UI layers).
- Decisions:
  - Plan retrait S2/D3: remplacer `Qt GUI (preview)` par libellé neutre Qt-only (ex. `Qt GUI`) et supprimer toute mention de statut préliminaire.
- Blockers:
  - Aucun.
- Next unchecked item: C2 — Remove or plan removal of “Tk stable” wording.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (planning only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 15
- Scope: S1/C2 only — **Remove or plan removal of “Tk stable” wording**.
- In scope: audit wording + plan de retrait (sans code).
- Out of scope: suppression effective du backend Tk (S2).
- Files changed: followup.md (C2.2 checkbox), memory.md (plan + preuves).
- Tests run:
  - `grep -nE "Classic Tk GUI \(stable\)|backend_option_tk|backend_change_notice" zemosaic_gui_qt.py zemosaic_gui.py`
- Proof:
  - Wording trouvé: `backend_option_tk = "Classic Tk GUI (stable)"`.
- Decisions:
  - Plan retrait S2/D3: supprimer ce libellé et toute mention de stabilité comparée Tk/Qt, cohérent avec Qt-only officiel.
- Blockers:
  - Aucun.
- Next unchecked item: C2 — Identify backend switch UI elements to eliminate in S2.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (planning only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 16
- Scope: S1/C2 only — **Identify backend switch UI elements to eliminate in S2**.
- In scope: inventorier les éléments UI de bascule backend à retirer.
- Out of scope: suppression effective (S2/D3).
- Files changed: followup.md (C2.3 checkbox), memory.md (inventaire).
- Tests run:
  - revue des occurrences `backend_selector_label`, `backend_change_notice`, `preferred_gui_backend`, `_backend_option_entries`, `_refresh_backend_combobox`, `_on_backend_combo_selected`.
- Proof:
  - Éléments à retirer en S2:
    1) Combo backend + label/notice dans `zemosaic_gui.py` (Tk legacy UI bloc langue/backend).
    2) Entrées backend `tk/qt` dans `zemosaic_gui_qt.py` (label “Preferred GUI backend”, options et handler de changement).
    3) Persistance liée: `preferred_gui_backend` / `preferred_gui_backend_explicit` côté UI.
    4) Flags launcher liés au choix backend (`--tk-gui`) côté `run_zemosaic.py` (traité dans S2/D1).
- Decisions:
  - Inventaire C2.3 verrouillé pour exécution S2.
- Blockers:
  - Aucun incident.
- Next unchecked item: C3 — List backend features still alive but not exposed in Qt.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (planning only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 17
- Scope: S1/C3 only — backend features actives non exposées en Qt.
- In scope: inventaire + classification + décision.
- Out of scope: implémentation d’exposition/suppression.
- Files changed: followup.md (C3 checkboxes), memory.md (inventaire/classification).
- Tests run:
  - comparaison clés config Tk vs Qt (script statique)
  - `grep -nE "altaz_alpha_soft_threshold|stack_ram_budget_gb" zemosaic_worker.py`
- Proof:
  - Features backend actives mais non exposées en Qt identifiées:
    1) `altaz_alpha_soft_threshold` — présente dans Tk/config et consommée dans worker.
    2) `stack_ram_budget_gb` — présente côté Tk/config et consommée dans worker (budget mémoire stack).
- Decisions / classification:
  - `altaz_alpha_soft_threshold` => **expose now** (cohérence avec autres contrôles Alt/Az déjà présents en Qt).
  - `stack_ram_budget_gb` => **legacy** (tuning expert historique, pas nécessaire pour parité frontend officielle immédiate).
  - `stack_ram_budget_gb` reste aussi **out-of-scope** pour ce sprint de retrait Tk (pas bloquant P0/P1 frontend).
- Blockers:
  - Aucun incident.
- Next unchecked item: C4 — Verify Qt UI writes the expected config keys.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit/classification only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 18
- Scope: S1/C4 only — **Verify Qt UI writes the expected config keys**.
- In scope: vérifier le flux write côté Qt (widgets -> config -> snapshot -> save -> worker kwargs).
- Out of scope: rework persistence.
- Files changed: followup.md (C4.1 checkbox), memory.md (preuve).
- Tests run:
  - `grep -nE "_collect_config_from_widgets|_serialize_config_for_save|_save_config|_build_worker_invocation" zemosaic_gui_qt.py`
  - `sed -n '4088,4378p' zemosaic_gui_qt.py`
  - `sed -n '5628,5758p' zemosaic_gui_qt.py`
- Proof:
  - `_collect_config_from_widgets()` parcourt `_config_fields` et écrit les valeurs normalisées dans `self.config`.
  - `_serialize_config_for_save()` produit un snapshot JSON-safe basé sur `persisted_keys + loaded_snapshot + config`.
  - `_save_config()` persiste ce snapshot.
  - `_build_worker_invocation()` copie ce snapshot dans `worker_kwargs` (donc clés persistées et attendues transmises au backend).
- Decisions:
  - Verdict C4.1: **OK** (pipeline d’écriture Qt cohérent pour clés attendues).
- Blockers:
  - Aucun.
- Next unchecked item: C4 — Verify persisted settings reload correctly.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 19
- Scope: S1/C4 only — **Verify persisted settings reload correctly**.
- In scope: valider le chemin reload côté Qt (load -> merge defaults -> UI init depuis config).
- Out of scope: tests dynamiques E2E.
- Files changed: followup.md (C4.2 checkbox), memory.md (preuve).
- Tests run:
  - revue `_load_config()` et initialisation des widgets (`self.config.get(...)` lors de la construction UI)
  - revue `_save_config()` qui met à jour `_loaded_config_snapshot` après sauvegarde.
- Proof:
  - `_load_config()` charge depuis `zemosaic_config`, fusionne avec defaults, normalise clés GPU/phase45.
  - Les widgets Qt sont initialisés depuis `self.config` (donc valeurs persistées reprises au démarrage).
  - Après sauvegarde, `_loaded_config_snapshot` est synchronisé, consolidant le cycle save/reload.
- Decisions:
  - Verdict C4.2: **OK** (reload cohérent observé statiquement).
- Blockers:
  - Aucun.
- Next unchecked item: C4 — Identify any Tk/Qt ambiguity in config behavior.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 20
- Scope: S1/C4 only — **Identify any Tk/Qt ambiguity in config behavior**.
- In scope: lister ambiguïtés de comportement config entre Tk et Qt.
- Out of scope: correction (S2/S3).
- Files changed: followup.md (C4.3 checkbox), memory.md (ambiguïtés).
- Tests run:
  - revue croisée `zemosaic_gui.py`, `zemosaic_gui_qt.py`, `zemosaic_config.py`.
- Proof / ambiguities:
  1) `preferred_gui_backend`:
     - config par défaut encore `tk`; coexistence Tk/Qt persiste dans les UI/launcher.
     - ambigu pour Qt-only officiel.
  2) Persistance fermeture:
     - Qt sauvegarde systématiquement en `closeEvent`; Tk non systématique à la fermeture.
  3) Phase 4.5:
     - Qt force la désactivation via `_disable_phase45_config`; ce garde-fou n’est pas symétrique explicitement côté Tk.
- Decisions:
  - Ambiguïtés C4 identifiées et tracées pour traitement S2/S3.
- Blockers:
  - Aucun incident.
- Next unchecked item: C5 — Update parity matrix.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 21
- Scope: S1/C5 closeout (partial) — parity matrix update + proof.
- In scope: mise à jour de la matrice à la lumière de C1→C4.
- Out of scope: lever les blockers (S2).
- Files changed: followup.md (C5.1 + C5.3), memory.md (matrice mise à jour + blockers).
- Tests run:
  - consolidation des audits C1/C2/C3/C4 déjà tracés.
- Proof / updated parity matrix:
  - Startup workflow parity: **GAP** (pré-check worker Tk vs Qt).
  - Config load/save parity: **GAP** (comportements de save différents).
  - Logs/feedback parity: **OK** (écarts visuels mineurs).
  - Clean shutdown parity: **OK-ish / GAP mineur** (Qt plus robuste).
  - Filter workflow: **OK**.
  - Grid path: **OK**.
  - SDS path: **OK**.
  - Hidden backend features: **GAP** (`altaz_alpha_soft_threshold` non exposé Qt; `stack_ram_budget_gb` legacy).
  - UX coexistence wording/backend switch: **BLOCKING** pour Qt-only officiel.
  - Launcher/config backend default Tk: **BLOCKING** pour Qt-only officiel.
  - Worker dependency `lecropper`: **BLOCKING** pour gate S2/S5.
- Decisions:
  - C5.2 (`Confirm no remaining P0/P1 blocker`) = **BLOCKED** actuellement.
- Blockers:
  - P0/P1 restants: fallback Tk launcher, défaut backend `tk`, dépendance worker->`lecropper`, wording/coexistence backend UI.
- Next unchecked item: C5 — Confirm no remaining P0/P1 blocker for official frontend.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S1 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:09 — Iteration 22
- Scope: S2/D1→D3 sequential execution (Qt-only official cutover), with gate probes.
- In scope:
  - D1 launcher cleanup (`run_zemosaic.py`)
  - D2 config import safety/default backend
  - D3 Qt UI cleanup wording/backend choice
- Out of scope: broad refactor; `lecropper` port; non-canonical opportunistic changes.
- Files changed:
  - `run_zemosaic.py`
  - `zemosaic_config.py`
  - `zemosaic_gui_qt.py`
  - `zemosaic_filter_gui_qt.py`
  - `followup.md`
- Tests run:
  - `python3 -m py_compile run_zemosaic.py zemosaic_config.py zemosaic_gui_qt.py zemosaic_filter_gui_qt.py`
  - grep probes on Tk fallback/flags/messages
  - `python3 -c "import zemosaic.zemosaic_config as c; ..."` (OK)
  - `python3 -c "import zemosaic.zemosaic_worker as w; ..."` (FAILED in env: missing numpy)
- Proof:
  - D1: launcher no longer instantiates Tk/messagebox/root and no Tk fallback path.
  - D1: `--tk-gui` now explicitly unsupported (Qt-only message).
  - D2: `zemosaic_config.py` no longer imports tkinter at module import; Tk dialogs are lazy via `_ensure_tk_dialogs_loaded()` only in legacy prompt helpers.
  - D2: default backend switched to `preferred_gui_backend = "qt"`.
  - D3: backend choice widget removed from official Qt UI layout path.
  - D3: coexistence wording removed (`Qt GUI (preview)` / `Classic Tk GUI (stable)` removed from active Qt path); filter message now requires PySide6 for official frontend.
- Decisions:
  - C5.2 remained blocked; progression to S2 done by explicit user override.
  - D4 not yet checked: needs explicit decoupling proof around `lecropper` absence on official/headless paths.
- Blockers / incidents:
  - **Environment blocker:** `import zemosaic_worker` gate currently fails here due missing `numpy` in runtime env (`ModuleNotFoundError: No module named 'numpy'`).
  - This blocks full D5 import-gate proof in this environment.
- Next unchecked item:
  - D4 — Remove direct runtime dependency on `lecropper` from official path.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: decreased (launcher/config/Qt UI cutover applied).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:14 — Iteration 23
- Scope: S2/D4 only (`lecropper` decoupling), surgical edits.
- In scope:
  - retirer dépendance runtime directe à `lecropper` sur chemin officiel
  - retirer dépendance indirecte sur chemin headless validé (import path)
- Out of scope:
  - port Qt de `lecropper`
  - refactor worker large
  - clôture D5 tant que l’environnement manque de dépendances
- Files changed:
  - `zemosaic_worker.py`
  - `followup.md`
- Decisions taken:
  - suppression des imports module-level `lecropper` et `from .lecropper import detect_autocrop_rgb`.
  - `lecropper` passe en état annexe non requis côté worker import (`_LECROPPER_AVAILABLE=False` par défaut, `ANCHOR_AUTOCROP_AVAILABLE=False` par défaut).
- Proof collected:
  - grep: plus d’import direct `import lecropper` / `from .lecropper import ...` au module-level.
  - `python3 -m py_compile zemosaic_worker.py` OK (intégrité syntaxique).
- Tests run:
  - `grep -nE "^\s*import lecropper|from \.?lecropper import|..." zemosaic_worker.py`
  - `python3 -m py_compile zemosaic_worker.py`
- Result:
  - D4.1 ✅ fait
  - D4.2 ✅ fait
  - D4.3 ⏳ preuve runtime complète non exécutable ici (voir blocker env)
- Remaining blockers:
  - import gate dynamique `import zemosaic_worker` toujours bloqué par dépendance env manquante (`numpy`), indépendamment de `lecropper`.
- Next unchecked item:
  - D4.3 — Ensure absence of `lecropper` does not break official runtime/headless path.
- Lecropper status changed or not: unchanged as annex/standalone (decoupled from official path).
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged during this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:20 — Iteration 24
- Scope: S2/D4.3 + D5 gate progression (next unchecked items only).
- In scope:
  - prouver l’absence de fallback Tk au lancement officiel
  - valider `import zemosaic_config`
  - avancer la preuve de découplage `lecropper`
- Out of scope:
  - installation de dépendances système (numpy/PySide6)
  - refactor worker additionnel hors besoin de preuve
- Files changed:
  - `zemosaic_gui_qt.py` (message ImportError Qt-only wording correction)
  - `followup.md` (D5.1 + D5.2 cochés)
- Decisions taken:
  - conserver D4.3 non coché tant que preuve headless complète est bloquée par l’environnement.
- Proof collected:
  - `python run_zemosaic.py --tk-gui` => flag refusé + Qt-only message + aucune bascule Tk.
  - `python -c "import zemosaic.zemosaic_config"` => OK (`preferred_gui_backend=qt`).
  - grep officiel (`run_zemosaic.py`, `zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`, `zemosaic_config.py`) => aucune dépendance runtime explicite à `lecropper` sur path officiel.
- Tests run:
  - `python3 -m py_compile zemosaic_gui_qt.py run_zemosaic.py`
  - `python3 run_zemosaic.py --tk-gui`
  - `python3 -c "import zemosaic.zemosaic_config as c; ..."`
  - `python3 -c "import zemosaic.zemosaic_worker as w"` (échec env)
- Result:
  - D5.1 ✅
  - D5.2 ✅
  - D4.3 ⏳ (partiel) / D5.3 ⏳ bloqués par env
- Remaining blockers:
  - `import zemosaic_worker` échoue ici pour cause externe: `ModuleNotFoundError: No module named 'numpy'`.
  - donc preuve dynamique headless complète sans `lecropper` non démontrable dans cet env à ce stade.
- Next unchecked item:
  - D4.3 — Ensure absence of `lecropper` does not break official runtime/headless path.
- Lecropper status changed or not: unchanged (annex), decoupling from official path maintained.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged during this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:23 — Iteration 25
- Scope: finalize remaining S2 unchecked items (D4.3, D5.3, D5.4) and close S2.
- In scope:
  - preuve runtime/headless sans `lecropper`
  - gate import `zemosaic_worker` dans l’environnement `.venv`
- Out of scope:
  - installation système globale
  - refactor packaging/import style du worker
- Files changed:
  - `followup.md`
  - `agent.md`
  - `memory.md`
- Decisions taken:
  - validation gates S2 avec `.venv/bin/python` (env réel de dev indiqué par Tristan).
- Proof collected:
  - `run_zemosaic.py --tk-gui` (venv) => refus explicite Tk + runtime Qt-only.
  - `import zemosaic_config` (venv) => OK (`preferred_gui_backend=qt`).
  - `import zemosaic_worker` (venv, depuis répertoire package `zemosaic/zemosaic`) => OK.
  - test anti-lecropper (meta_path bloque `lecropper` et `zemosaic.lecropper`) :
    - `import zemosaic_worker` => OK
    - `import run_zemosaic` => OK
- Tests run:
  - `/home/tristan/zemosaic/.venv/bin/python /home/tristan/zemosaic/zemosaic/run_zemosaic.py --tk-gui`
  - `/home/tristan/zemosaic/.venv/bin/python -c "import zemosaic.zemosaic_config ..."`
  - `/home/tristan/zemosaic/.venv/bin/python -c "import zemosaic_worker ..."` (workdir package)
  - two `meta_path` blocker scripts for `lecropper`
- Result:
  - D4.3 ✅
  - D5.3 ✅
  - D5.4 ✅
  - **S2 closed**
- Remaining blockers:
  - aucun blocker S2.
  - note technique hors-scope immédiat: `import zemosaic.zemosaic_worker` depuis repo root échoue à cause d’un import absolu `zemosaic_resource_telemetry`; gate validé via chemin headless défini en S0/B2.
- Next unchecked item:
  - E1 — Migrate `preferred_gui_backend=tk` to `qt`.
- Lecropper status changed or not: unchanged (annex), runtime decoupling now proven.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged during this iteration (proof-only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:30 — Iteration 26
- Scope: S3/E1 only — migrate `preferred_gui_backend=tk` to `qt`.
- In scope: config migration logic at load-time, surgical.
- Out of scope: E1.2/E1.3, fixtures E2, cleanup E3.
- Files changed:
  - `zemosaic_config.py`
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - Add explicit load-time migration: legacy `preferred_gui_backend == "tk"` is normalized to `"qt"`.
- Proof collected:
  - Temp legacy config fixture (`preferred_gui_backend: tk`) loaded through monkeypatched `get_config_path` returns `preferred_gui_backend: qt`.
- Tests run:
  - `/home/tristan/zemosaic/.venv/bin/python` script with temporary config file + `zemosaic_config.load_config()`
  - output: `MIGRATED_BACKEND qt`
- Result:
  - E1.1 ✅ done.
- Remaining blockers:
  - none for E1.2 next.
- Next unchecked item:
  - E1 — Neutralize obsolete backend selection state if needed.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged in this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:31 — Iteration 27
- Scope: S3/E1.2 + E1.3 (next unchecked items), surgical config migration hardening.
- In scope:
  - neutralize obsolete backend-selection state
  - preserve backward readability for legacy config files
- Out of scope:
  - E2 fixtures set
  - broader config refactor
- Files changed:
  - `zemosaic_config.py`
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - `load_config()` now forces neutral Qt-only state:
    - `preferred_gui_backend = "qt"`
    - `preferred_gui_backend_explicit = False`
  - `save_config()` also enforces same neutral state to prevent stale reactivation.
- Proof collected:
  - Legacy fixture with `{preferred_gui_backend: "tk", preferred_gui_backend_explicit: true}`:
    - load => `BACKEND qt EXPLICIT False`
    - save => file persists `SAVED_BACKEND qt SAVED_EXPLICIT False`
- Tests run:
  - `.venv/bin/python` monkeypatch `get_config_path` + temp config round-trip
- Result:
  - E1.2 ✅
  - E1.3 ✅
- Remaining blockers:
  - none for E2 start.
- Next unchecked item:
  - E2 — Create/collect minimal legacy config fixtures.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:32 — Iteration 28
- Scope: continue S3 sequentially (E2.1→E4), stop only on blocker.
- In scope:
  - create minimal legacy config fixtures
  - prove load→save→load idempotence
  - prove no silent Tk reactivation
  - remove remaining active official coexistence branches
  - close S3 when proven
- Out of scope:
  - S4 packaging/doc release edits
  - broad refactor
- Files changed:
  - `tests/fixtures/config_migration/legacy_backend_tk_minimal.json`
  - `tests/fixtures/config_migration/legacy_backend_weird_value.json`
  - `tests/fixtures/config_migration/already_qt_explicit_true.json`
  - `zemosaic_config.py`
  - `run_zemosaic.py`
  - `zemosaic_gui_qt.py`
  - `followup.md`
  - `agent.md`
- Decisions taken:
  - Fixture set for migration safety kept minimal (3 representative cases).
  - Official coexistence cleanup narrowed to active branches only:
    - removed dead backend-selection handlers/groups in Qt GUI
    - simplified launcher backend normalization to Qt-only flag handling (no env backend selection path)
- Proof collected:
  - Round-trip script over fixtures (`tests/fixtures/config_migration/*.json`) reports all idempotent=true.
  - All fixtures end with `preferred_gui_backend=qt` and `preferred_gui_backend_explicit=false`.
  - py_compile passes for updated files.
  - grep confirms backend coexistence methods removed from active Qt GUI path.
- Tests run:
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic /home/tristan/zemosaic/.venv/bin/python /tmp/zemosaic_roundtrip_check.py`
  - `python3 -m py_compile run_zemosaic.py zemosaic_gui_qt.py zemosaic_config.py`
  - grep probes on removed coexistence symbols and launcher backend controls.
- Result:
  - E2.1 ✅
  - E2.2 ✅
  - E2.3 ✅
  - E3.1 ✅
  - E3.2 ✅
  - E4.1 ✅
  - E4.2 ✅
  - **S3 closed**
- Remaining blockers:
  - none in S3.
- Next unchecked item:
  - F1 — Audit official build/spec scripts.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: decreased (coexistence branches further removed).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:35 — Iteration 29
- Scope: S4/F1 sequential until blocker.
- In scope:
  - audit build/spec scripts
  - remove packaging hints suggesting Tk coexistence
  - verify final built artifacts if build tool available
- Out of scope:
  - F2 docs/release notes content rewrite
- Files changed:
  - `ZeMosaic.spec`
  - `requirements.txt`
  - `requirements_no_gpu.txt`
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - remove Tk matplotlib backend hiddenimports from spec (`backend_tkagg`, `_backend_tk`, `_tkagg`).
  - remove `lecropper` hiddenimport from official spec bundle list.
  - remove Tk install notes from requirements files; keep PySide6 as required Qt-only frontend dependency.
- Proof collected:
  - grep on spec/requirements confirms no remaining Tk packaging hints and no Tk backend hiddenimports.
- Tests run:
  - `find ...` audit packaging files
  - grep probes on spec/requirements
  - build-attempt probes:
    - `.venv/bin/python -m PyInstaller --version` => module missing
- Result:
  - F1.1 ✅
  - F1.3 ✅
  - F1.2 ⛔ blocked (cannot verify final built artifacts without PyInstaller installed in `.venv`).
- Remaining blockers:
  - Missing build dependency in `.venv`: `PyInstaller`.
- Next unchecked item:
  - F1.2 — Verify final built artifacts, not only source scripts.
- Lecropper status changed or not: unchanged (annex); no longer hiddenimported in official spec.
- Official-path Tk imports decreased or stayed unchanged: decreased for packaging surface.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:38 — Iteration 30
- Scope: S4/F1.2 only (verify final built artifacts).
- In scope:
  - build PyInstaller artifact from updated spec
  - inspect produced dist/build outputs for Qt-only packaging consistency
- Out of scope:
  - full docs updates (F2)
  - release-note authoring (F3)
- Files changed:
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - artifact verification performed directly on generated `dist/ZeMosaic` output.
- Proof collected:
  - PyInstaller version in `.venv`: `6.19.0`.
  - Build completed successfully; output at `dist/ZeMosaic`.
  - Final artifact exists: `dist/ZeMosaic/ZeMosaic` + `_internal` payload.
  - `warn-ZeMosaic.txt` reviewed.
- Tests run:
  - `/home/tristan/zemosaic/.venv/bin/python -m PyInstaller --version`
  - `python -m PyInstaller --noconfirm --clean ZeMosaic.spec`
  - `ls dist/ZeMosaic`, `ls dist/ZeMosaic/_internal`
  - `sed -n '1,220p' build/ZeMosaic/warn-ZeMosaic.txt`
- Result:
  - F1.2 ✅ done.
- Remaining blockers:
  - none for F1 completion.
  - note (non-blocking for current mission step): PyInstaller still auto-pulls some tkinter hooks transitively via dependency graph; runtime launch remains Qt-only and `--tk-gui` unsupported.
- Next unchecked item:
  - F2 — Update user docs / README / quickstart / troubleshooting.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration (artifact validation only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:41 — Iteration 31
- Scope: S4/F2→F5 sequential completion (docs/release/version semantics/proof).
- In scope:
  - update user-facing docs for Qt-only official runtime
  - update dev/build notes reflecting Qt-only packaging
  - publish annex status for `lecropper`
  - update release notes with migration + unsupported legacy status
  - record version semantics note
- Out of scope:
  - S5 QA/CI execution
  - repo-wide Tk annex purge (S6)
- Files changed:
  - `README.md`
  - `RELEASE_NOTES.md`
  - `followup.md`
  - `agent.md`
  - `memory.md`
- Decisions taken:
  - release-line semantics: 4.4.1 docs/release notes explicitly treat Qt-only official frontend as normative behavior; legacy Tk path marked non-official.
- Proof collected:
  - README now states official frontend is PySide6/Qt and removes active Tk coexistence instructions in quickstart/build sections.
  - README includes explicit annex status note for `lecropper`.
  - RELEASE_NOTES now explicitly covers:
    - Qt-only official frontend
    - no Tk fallback on official startup
    - config migration (`preferred_gui_backend=tk` -> `qt` + explicit flag neutralization)
    - unsupported/legacy statement for Tk frontend
    - `lecropper` annex status
- Tests run:
  - grep checks on README/RELEASE_NOTES for residual Tk-coexistence guidance.
- Result:
  - F2.1 ✅
  - F2.2 ✅
  - F2.3 ✅
  - F3.1 ✅
  - F3.2 ✅
  - F3.3 ✅
  - F3.4 ✅
  - F4.1 ✅
  - F4.2 ✅
  - F5 ✅
  - **S4 closed**
- Remaining blockers:
  - none in S4.
- Next unchecked item:
  - G1 — Windows smoke test.
- Lecropper status changed or not: unchanged (annex/standalone), publicly clarified.
- Official-path Tk imports decreased or stayed unchanged: decreased on docs/packaging communication surface.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:47 — Iteration 32
- Scope: S5/G1 per agreed modality (launch + run start + early interruption), continue until blocker.
- In scope:
  - execute Linux smoke test with example set (intermediate checkpoint)
  - create explicit Windows/macOS smoke protocol note for Tristan to run locally
- Out of scope:
  - full 35-minute completion runs
  - Windows/macOS execution from this Linux host
- Files changed:
  - `tests/SMOKE_PROTOCOL_Windows_macOS.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - worker-process smoke script with `example/organized/EQ/IRCUT` and output `example/out/smoke_linux_20260313_b`
  - process started, emitted runtime messages, then terminated intentionally for checkpoint stop test.
- Proof collected (Linux):
  - process alive before stop: `True`
  - event stream observed (15 messages), including Grid mode activation and iterative processing warnings (path issues from dataset stack_plan), confirming run start/activity.
  - manual interruption path exercised via process terminate (checkpoint equivalent to early stop).
- Decisions taken:
  - G1 Linux smoke marked complete for intermediate checkpoint criteria.
  - Windows/macOS smoke delegated with documented protocol (`tests/SMOKE_PROTOCOL_Windows_macOS.md`).
- Remaining blockers:
  - G1 Windows/macOS execution blocked on host availability (to be run by Tristan manually).
- Next unchecked item:
  - G1 — Windows smoke test.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:52 — Iteration 33
- Scope: Continue S5 sequentially after G1 hold, until blocking item.
- In scope:
  - mark G1 Windows/macOS on hold by explicit user decision
  - execute G2 dependency-failure tests
  - execute G3 headless tests
  - add CI hardening job for G4
  - progress G5 checklist until first blocker
- Out of scope:
  - running Windows/macOS smoke locally
  - forcing CI run status from local host
- Files changed:
  - `followup.md`
  - `.github/workflows/no-tk-on-official-path.yml`
  - `tests/SMOKE_PROTOCOL_Windows_macOS.md` (from previous step, still reference)
  - `memory.md`
- Tests run:
  - G2 / PySide6 absent: meta_path block -> launcher exits with clear Qt error, no Tk fallback.
  - G2 / worker import failure: meta_path block -> launcher exits error, no Tk fallback.
  - G2 / startup path with `--tk-gui`: explicit refusal + Qt path used.
  - G3 / `import zemosaic_config` OK (`qt`, explicit False).
  - G3 / config load-save-load idempotent OK.
  - G3 / `import zemosaic_worker` OK with `lecropper` blocked.
- CI hardening:
  - added `.github/workflows/no-tk-on-official-path.yml` with:
    - official-path Tk grep guard
    - headless import gates (`zemosaic_config`, `zemosaic_worker`)
    - `--tk-gui` non-fallback behavior check.
- Result:
  - G2 ✅ complete
  - G3 ✅ complete
  - G4 ✅ complete (job defined)
  - G5 partially advanced; all items checked except CI green.
- Blocker:
  - **G5: `CI is green` remains unchecked** (requires remote CI run/result, not provable locally in this step).
- Next unchecked item:
  - G5 — CI is green.
- Lecropper status changed or not: unchanged (annex), no official runtime dependency.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration (validation + CI guard addition).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 11:11 — Iteration 34
- Scope: close remaining S5 blocker after user CI feedback.
- In scope:
  - validate G5 `CI is green` from user-provided workflow result
  - close G6 (GO/NO-GO)
- Out of scope:
  - Node 20 deprecation remediation (non-blocking infra warning)
- Files changed:
  - `followup.md`
  - `agent.md`
  - `memory.md`
- Proof collected:
  - User confirms `guard` workflow is GREEN.
  - Warning is non-blocking deprecation notice for Node 20 actions (`actions/checkout@v4`, `actions/setup-python@v5`).
- Decisions:
  - Release status for this migration checkpoint: **GO**.
  - Node 20 warning tracked as follow-up infra task; does not block Qt-only/Tk-retirement acceptance.
- Remaining blockers:
  - none for S5 closeout.
  - Windows/macOS smoke tests remain on hold by user decision.
- Next unchecked item:
  - S6 (later mission only; do not start unless requested).
- Lecropper status changed or not: unchanged (annex), decoupled from official runtime/headless validated paths.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.
