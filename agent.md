# Mission: Corriger le build Windows PyInstaller (GPU indisponible + crash Shapely)

## Constat (logs)
Comparer 2 logs :
- `dist/ZeMosaic/zemosaic_worker.log` (app compilée) :
  - `phase5_using_cpu`
  - `[GPU_SAFETY] ... reasons=...gpu_unavailable`
  - crash en Phase 4 (Final Grid) : `WinError 206` sur `...\\_internal\\shapely.libs` pendant `import shapely` (via `reproject`).
- `zemosaic_worker.log` (run Python non compilé) :
  - `phase5_using_gpu`
  - `gpu_info_summary` avec des valeurs non-None
  - pas de crash shapely en Phase 4.

## Diagnostic
1) **GPU**
- La détection GPU passe par `parallel_utils.detect_parallel_capabilities()` → `_probe_gpu()` (CuPy).
- Dans le build PyInstaller, CuPy n’arrive pas à s’initialiser (ou ne voit pas ses DLL) → `gpu_available=False`.
- Ensuite `zemosaic_gpu_safety.apply_gpu_safety_to_parallel_plan()` force `plan.use_gpu=False` et Phase 5 tombe en CPU.

2) **Shapely**
- `reproject.mosaicking.wcs_helpers.find_optimal_celestial_wcs()` importe Shapely.
- Shapely (delvewheel) appelle `os.add_dll_directory(<...>\\shapely.libs)` et ça échoue avec `WinError 206`.
- Résultat : `import shapely` échoue → Phase 4 échoue → run stoppé.

## Objectif
- Le build compilé doit :
  - ne plus crasher sur Shapely en Phase 4.
  - détecter/utiliser le GPU quand disponible (et rester robuste en fallback CPU).

## Plan d’action (à exécuter dans ce repo)
1) [X] **Rendre les DLL “bundlées” discoverables**
   - Modifier `pyinstaller_hooks/rthook_zemosaic_sys_path.py` :
     - ajouter `sys._MEIPASS` (onedir: `dist/ZeMosaic/_internal`) au DLL search path via `os.add_dll_directory()`.
     - ajouter aussi chaque dossier `*.libs` sous `sys._MEIPASS`.
     - répercuter ces chemins dans `PATH` (pour les sous-process).

2) [X] **Durcir `os.add_dll_directory()` pour WinError 206**
   - Dans `pyinstaller_hooks/rthook_zemosaic_sys_path.py` :
     - wrapper `os.add_dll_directory()` pour retenter avec un chemin “extended-length” (`\\\\?\\...`) sur `WinError 206`.
     - optionnel: éviter un crash si Shapely tente d’ajouter `shapely.libs` alors qu’on a relocalisé ses DLL (voir étape 3).

3) [X] **Stopper la dépendance à `shapely.libs`**
   - Modifier `ZeMosaic.spec` :
     - après `a = Analysis(...)`, relocaliser les binaires dont la destination commence par `shapely.libs\\` vers `shapely\\`.
     - but : les DLL GEOS se retrouvent à côté des `.pyd` de Shapely, et Shapely n’a plus besoin d’ajouter `shapely.libs`.

4) [X] **Rebuild Windows**
   - Lancer `compile\\compile_zemosaic._win.bat` (mode onedir recommandé) ou :
     - `pyinstaller --noconfirm --clean ZeMosaic.spec`

5) [X] **Validation (smoke test)**
   - Relancer un dataset minimal qui passe par Phase 4 + 5.
   - Vérifier dans `dist/ZeMosaic/zemosaic_worker.log` :
     - plus de `WinError 206` sur `shapely.libs`.
     - `gpu_info_summary` a des valeurs et `phase5_using_gpu` apparaît (si GPU dispo).

6) [X] **Écrire un `memory.md` à la fin**
   - Créer `memory.md` à la racine du repo avec :
     - résumé des changements (fichiers touchés + pourquoi),
     - commande de build Windows utilisée,
     - comment valider (quoi chercher dans les logs),
     - points de vigilance (onefile vs onedir, CUDA_PATH, etc.).

## Contraintes
- Ne pas rendre le GPU obligatoire : fallback CPU doit rester OK.
- Ne pas toucher aux algorithmes (stacking, WCS, reprojection) : uniquement build/runtime hooks.
