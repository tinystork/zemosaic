# Memory: build Windows PyInstaller (GPU + Shapely)

Date: 2026-01-28

## Problème observé
Sur la version compilée (PyInstaller / Windows / onedir) :
- `dist/ZeMosaic/zemosaic_worker.log` montrait `phase5_using_cpu` et `[GPU_SAFETY] ... gpu_unavailable` (GPU non détecté / non utilisé).
- Crash en Phase 4 (Final Grid) lors de l’import de Shapely via `reproject` :
  `FileNotFoundError: [WinError 206] ... add_dll_directory(...\\_internal\\shapely.libs)`.

Le run Python non compilé (`zemosaic_worker.log` à la racine) utilisait bien le GPU (`phase5_using_gpu` + `gpu_info_summary`).

## Hypothèses / causes
- GPU : `parallel_utils._probe_gpu()` dépend de CuPy. Dans le build PyInstaller, CuPy peut échouer à importer/charger ses DLL CUDA si le DLL search path ne contient pas le dossier `_internal` (=`sys._MEIPASS`) où PyInstaller place des DLL.
- Shapely : Shapely (delvewheel) appelle `os.add_dll_directory(<...>\\shapely.libs)` ; selon la résolution de chemin côté Windows, ça peut déclencher `WinError 206`.

## Changements effectués

### `pyinstaller_hooks/rthook_zemosaic_sys_path.py`
- Ajout d’un patch de `os.add_dll_directory()` :
  - retry avec un chemin “extended-length” (`\\\\?\\...`) quand l’appel échoue avec `WinError 206`.
  - évite de crasher si `shapely.libs` sous `sys._MEIPASS` ne peut pas être ajouté (utile si on relocalise les DLL Shapely).
- Ajout d’un setup “frozen DLL search path” :
  - ajoute `sys._MEIPASS` au DLL search path via `os.add_dll_directory()` et le préfixe dans `PATH`.
  - ajoute aussi chaque dossier `*.libs` directement sous `sys._MEIPASS`.

Objectif : rendre les DLL bundlées (dont CUDA pour CuPy) trouvables et stabiliser les imports de wheels delvewheel.

### `ZeMosaic.spec`
- Ajout d’un post-traitement Windows après `Analysis(...)` :
  - relocalise les binaires dont la destination commence par `shapely.libs\\` vers `shapely\\`.

Objectif : mettre les DLL GEOS à côté des `.pyd` de Shapely pour ne plus dépendre de `shapely.libs` (et réduire les risques liés à `add_dll_directory`).

### Docs / pilotage
- `agent.md` et `followup.md` réécrits pour guider le diagnostic et la validation (anciens contenus supprimés).

## Comment valider (après rebuild Windows)
1) Rebuild : `pyinstaller --noconfirm --clean ZeMosaic.spec` (onedir conseillé).
2) Lancer un run qui atteint Phase 4 + 5.
3) Vérifier dans `dist/ZeMosaic/zemosaic_worker.log` :
   - plus de `WinError 206 ... shapely.libs`
   - GPU : `gpu_info_summary` avec valeurs non-None + `phase5_using_gpu` (si GPU dispo).

## Notes
- Si la machine n’a pas de GPU/CUDA fonctionnel, le fallback CPU doit rester OK (ne pas rendre le GPU obligatoire).

## Mise à jour (2026-01-28)
- Confirmation que les correctifs runtime hook (`pyinstaller_hooks/rthook_zemosaic_sys_path.py`) et la relocalisation Shapely (`ZeMosaic.spec`) sont en place.
- Checklist `agent.md` mise à jour avec [X] pour les étapes 1–3 et 6.
- Aucun rebuild ni smoke-test lancé dans cette session (étapes 4–5 restantes).

## Mise à jour (2026-01-28, diagnostics)
- Ajout de logs de diagnostic :
  - `parallel_utils.py` : log INFO quand l’import de CuPy échoue (message d’erreur capturé).
  - `pyinstaller_hooks/rthook_zemosaic_sys_path.py` : log INFO de `sys._MEIPASS` et des dossiers ajoutés via `os.add_dll_directory()` (avec sortie stderr).
- Checklist `followup.md` étape 6 cochée.
-Programme recompilé par l'utilisateur les logs sont disponibles ici : 
  Log de la version compilée : 
  F:\seescan\env\zemosaic\dist\ZeMosaic\zemosaic_worker.log
  L log de la version non compilée est ci : 
  F:\seescan\env\zemosaic\zemosaic_worker.log

## Mise à jour (2026-01-28, build + test)
- Rebuild Windows lancé avec : `pyinstaller --noconfirm --clean ZeMosaic.spec`.
- Smoke-test packagé lancé puis interrompu (le problème persiste en compilé).
- Logs annoncés par l’utilisateur :
  - Compilé : `F:\seescan\env\zemosaic\dist\ZeMosaic\zemosaic_worker.log`
  - Non compilé : `F:\seescan\env\zemosaic\zemosaic_worker.log`
- À analyser dès que les extraits de log sont fournis.

## Mise à jour (2026-01-28, extrait logs)
- Build compilé : `phase5_using_cpu` + `[GPU_SAFETY] ... reasons=hybrid_graphics,battery_present,gpu_unavailable`.
- Aucun `WinError 206` ni `shapely.libs` détecté dans l’extrait filtré.
- Build non compilé : `gpu_info_summary` avec VRAM + `phase5_using_gpu`.
- Les logs `[rthook]` et `CuPy unavailable` n’apparaissent pas dans `zemosaic_worker.log` (probable absence de routage vers le logger worker).

## Mise à jour (2026-01-28, diagnostics GPU supplémentaires)
- `parallel_utils.py` : logs GPU routés vers le logger `ZeMosaicWorker` + ajout de messages d’erreur détaillés (getDeviceCount, device properties, memGetInfo).
- `pyinstaller_hooks/rthook_zemosaic_sys_path.py` : ajout des dossiers contenant des DLL sous `cupy/` et `cupy_backends/` (scan `*.dll`) au DLL search path / `PATH`.

## Mise à jour (2026-01-28, constat CuPy manquant)
- Log compilé : `GPU probe: CuPy unavailable (ImportError: ...)` → CuPy indisponible dans le build.
- Aucune DLL trouvée sous `dist\ZeMosaic\_internal\cupy\` (dossier vide) → CuPy non bundlé.
- Correctif build ajouté :
  - `ZeMosaic.spec` : avertissement explicite si CuPy absent + option `ZEMOSAIC_REQUIRE_CUPY=1` pour rendre CuPy obligatoire au build.
  - `compile/compile_zemosaic._win.bat` : installation optionnelle de CuPy via `ZEMOSAIC_CUPY_PKG`.

## Mise à jour (2026-01-28, stack CUDA confirmée)
- Script `tests/test_version_gpu.py` (hors build) :
  - CuPy 13.4.1
  - CUDA Runtime 12.8 / Driver 12.9
  - nvcc 12.8
  - GPU : NVIDIA GeForce RTX 3070 (8 GB)
- Conclusion : la pile CUDA fonctionne en Python non compilé ; le build PyInstaller n’embarque pas CuPy.

## Mise à jour (2026-01-28, erreurs CPU en build)
- En build packagé, AlignGroup produit `TypeError: Input type for source not supported` pendant la Phase 3 (alignement).
- Le run non compilé ne présente pas ces erreurs.
- Prochaine action : activer `ZEMOSAIC_LOG_LEVEL=DEBUG` sur le build packagé pour capturer `AlignGroup Traceback` et identifier la dépendance fautive.

## Mise à jour (2026-01-28, warnings PyInstaller CuPy)
- Warnings build PyInstaller : DLLs introuvables `cuTENSOR.dll` et `cudnn64_8.dll` (dépendances de modules CuPy).
- Interprétation : ces libs sont optionnelles (cuTENSOR/cuDNN). Pas bloquant si on ne les utilise pas, mais si CuPy tente de charger ces modules, l’import peut échouer.

## Mise à jour (2026-01-28, build ZEMOSAIC_REQUIRE_CUPY)
- Build lancé avec `ZEMOSAIC_REQUIRE_CUPY=1` depuis Python 3.13 (install système).
- PyInstaller analyse bien de nombreux hidden imports CuPy (cupy._core, cupy_backends, cupyx.*), donc CuPy est présent dans l’environnement de build.
- Warnings cuTENSOR/cudnn toujours présents (libs non trouvées au build).

## Mise à jour (2026-01-28, diagnostic ImportError CuPy)
- Le log runtime indique `ImportError: Failed to import CuPy` (message générique).
- CuPy `.pyd` présents dans `_internal\cupy\` et `CUDA_PATH` est défini.
- Ajout d’un log détaillé dans `parallel_utils.py` pour capturer la chaîne d’erreurs (cause/context/traceback) + infos CUDA_PATH/PATH.

## Mise à jour (2026-01-28, cause CuPy)
- ImportError détaillé : `ModuleNotFoundError: No module named 'fastrlock'`.
- Correctif: bundler explicitement `fastrlock` via `ZeMosaic.spec` (hiddenimports + collect_all).

## Mise à jour (2026-01-28, logs comparés)
- `zemosaic_worker_compiled.log` : GPU détecté + utilisé, mais le run reste en Phase 3 (Master Tiles) et n’atteint pas Phase 4/5 dans le log.
- Nombreuses alertes : ~224 occurrences de `salvage mode (n=1)` et ~72 `run_warn_phase3_alignment_retry_abandoned` → alignement intra-tuile échoue fréquemment.
- `zemosaic_worker.log` non compilé est très court (61 lignes) et ne contient pas ces alertes → comparaison directe non concluante (run différent ou log tronqué).

## Mise à jour (2026-01-28, analyse logs repo)
- `zemosaic_worker_compiled.log` : Phase 4 démarre puis échoue immédiatement avec
  `calcgrid_error_find_optimal_wcs_unavailable` puis `run_error_phase4_grid_calc_failed`.
- `zemosaic_worker_uncompiled.log` : Phase 4 se termine (`run_info_phase4_finished`) puis Phase 5 démarre.
- Conclusion : en build packagé, l’appel à `find_optimal_celestial_wcs()` lève un `ImportError` (dépendance manquante/lazy-import dans `reproject`), ce qui bloque la Phase 4.

## Mise à jour (2026-01-28, correctif de log)
- `zemosaic_worker.py` : le `except ImportError` autour de `find_optimal_celestial_wcs()` logge désormais le message d’exception + traceback pour identifier le module manquant au runtime packagé.

## Mise à jour (2026-01-29, cause réelle Phase 4)
- `zemosaic_worker_compiled.log` (racine) confirme l’ImportError exact :
  `ModuleNotFoundError: No module named 'shapely._geos'` pendant `find_optimal_celestial_wcs`.
- Impact : la Phase 4 (grid final) échoue en build packagé, alors que la version non compilée passe.

## Mise à jour (2026-01-29, fix packaging Shapely)
- `pyinstaller_hooks/hook-shapely.py` : ajout de `hiddenimports` pour `shapely._geos` (et variantes si présentes) afin d’embarquer l’extension C manquante dans le build packagé.

## Mise à jour (2026-01-29, état logs + alignement)
- `zemosaic_worker_compiled.log` (racine) : ~240 occurrences de `salvage mode` et ~120 `run_warn_phase3_alignment_retry_abandoned` → alignement intra-tuile en échec dans le build packagé.
- `zemosaic_worker_uncompiled.log` : 0 occurrence de ces warnings (run non compilé OK).
- L’erreur runtime Phase 4 est bien `ModuleNotFoundError: No module named 'shapely._geos'` dans le build packagé ; l’utilisateur confirme que `_geos.cp313-win_amd64.pyd` est maintenant présent sous `dist\ZeMosaic\_internal\shapely`.

## Mise à jour (2026-01-29, instrumentation AlignGroup)
- `zemosaic_align_stack.py` : en cas d’exception dans `align_images_in_group`,
  ajout d’un log **ERROR** dans le logger `ZeMosaicWorker` avec le type/dtype/shape des images
  + traceback complet. Objectif : identifier la source exacte du `TypeError: Input type for source not supported`.

## Mise à jour (2026-01-29, cause AlignGroup confirmée)
- Logs packagés montrent que l’exception provient de `astroalign` → `sep_pjw` :
  `ModuleNotFoundError: No module named '_version'`, suivi de `TypeError: Input type for source not supported`.
- Root cause : le module top-level `_version` (utilisé par `sep_pjw`) n’est pas bundle par PyInstaller.

## Mise à jour (2026-01-29, fix packaging SEP)
- `ZeMosaic.spec` : ajout d’un bloc optionnel pour `sep` :
  - `hiddenimports` inclut `sep`, `sep_pjw`, et `_version` si présent.
  - `collect_all("sep")` pour embarquer wrapper + fichiers associés.

## Mise à jour (2026-01-29, stub _version pour sep_pjw)
- Ajout d’un `_version.py` à la racine du repo (stub) qui expose `__version__` + `get_versions()`.
- Objectif : satisfaire `sep_pjw` en build packagé si `_version` n’est pas fourni par PyInstaller.

## Mise à jour (2026-01-29, forçage bundle _version)
- `ZeMosaic.spec` : ajout explicite de `_version.py` dans `datas` (destination racine `_internal`),
  pour garantir que `import _version` fonctionne en build packagé.

## Mise à jour (2026-01-29, stub _version corrigé)
- Log packagé : `ImportError: cannot import name version` depuis `sep_pjw`.
- `_version.py` mis à jour pour exposer `version = __version__` (en plus de `__version__` et `get_versions()`).
