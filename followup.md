# Follow-up: checklist de debug/build (Windows PyInstaller)

## 1) [X] Vérifier le symptôme dans les logs
- App compilée : `dist/ZeMosaic/zemosaic_worker.log`
  - Chercher :
    - `phase5_using_cpu`
    - `[GPU_SAFETY] ... gpu_unavailable`
    - `WinError 206` + `shapely.libs`
- Run Python : `zemosaic_worker.log`
  - Chercher :
    - `phase5_using_gpu`
    - `gpu_info_summary`

## 2) [X] Pourquoi “GPU indisponible” arrive en build
Rappel : `parallel_utils._probe_gpu()` dépend de CuPy.
Si CuPy n’arrive pas à importer/charger ses DLL (CUDA), `gpu_available=False` et le safety clamp coupe le GPU.

À vérifier après correctif :
- que le runtime hook ajoute bien `sys._MEIPASS` au DLL search path.
- que les DLL CUDA effectivement bundlées dans `_internal` sont trouvables.

## 3) [X] Pourquoi Shapely crash en build
Le crash est déclenché par Shapely au moment où il exécute le patch delvewheel :
- `os.add_dll_directory(<...>\\shapely.libs)` → `WinError 206`

Stratégies de correction (préférence dans cet ordre) :
1) wrapper `os.add_dll_directory()` pour retenter en “extended-length path”.
2) relocaliser les DLL GEOS vers `shapely/` dans `ZeMosaic.spec` (post-`Analysis`).

## 4) [X] Commandes de build Windows (onedir recommandé)
Dans PowerShell, depuis la racine du projet :
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib
pyinstaller --noconfirm --clean ZeMosaic.spec
```

Ou via :
```powershell
compile\compile_zemosaic._win.bat
```

## 5) [X] Smoke-test minimal (packagé)
1) Lancer `dist\ZeMosaic\ZeMosaic.exe`
2) Lancer un run qui atteint Phase 4 puis Phase 5.
3) Vérifier `dist\ZeMosaic\zemosaic_worker.log` :
   - pas de `WinError 206`
   - GPU : `phase5_using_gpu` + `gpu_info_summary` non-None (si GPU dispo)

## 6) [X] Si ça échoue encore
Actions “diagnostic” à faire dans le code (petites, ciblées) :
- Ajouter un log INFO dans `parallel_utils` quand CuPy est indisponible (capturer l’exception d’import).
- Ajouter un log INFO dans le runtime hook indiquant :
  - `sys._MEIPASS`
  - les dossiers ajoutés via `os.add_dll_directory`

## 7) [X] Mémo obligatoire
À la fin, créer `memory.md` (résumé humain des changements et de la validation).
