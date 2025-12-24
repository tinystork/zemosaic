# Mission — Intertile silent hang/crash hardening (Windows)

## Constat
On observe un arrêt silencieux pendant "Phase5 Intertile Pairs" vers ~348/4342 sans exception.
Le log s’arrête après:
- [Intertile] ... pairs=xxxx
- [Intertile] Parallel: threadpool workers=15 -> 4 ...
Aucune ligne "Intertile done" ni "[SUCCESS]" ensuite.

Hypothèse forte: deadlock/crash natif dans des appels multi-thread à:
- reproject_interp (astropy.wcs / wcslib / reproject)
- cv2.resize
Le code actuel exécute ces appels dans un ThreadPoolExecutor par paire.

## Objectifs (priorité)
1) Rendre l’Intertile robuste sur Windows: éviter hang/crash silencieux.
2) Ajouter de la télémétrie minimale pour diagnostiquer si ça rehange.
3) Patch minimal, pas de refactor massif, pas d’UI nouvelle.

## Fichiers ciblés
- zemosaic_utils.py (compute_intertile_affine_calibration, _process_overlap_pair)

## Plan
### A) Sécurité d’exécution (Windows)
Implémenter un mode "safe" automatique sur Windows:
- Si platform=win32 et pairs >= 2000 (ou >= 4000) et preview >= 512:
  => FORCER effective_workers = 1 pour l’intertile
  => logger clairement: "[Intertile] SAFE_MODE: forcing single-worker on Windows to avoid native deadlocks in reproject"
Justification: stabilité > perf. Ne pas exposer ça à l’UI.

Alternative si tu veux garder un peu de parallélisme:
- Introduire un `threading.Lock()` global autour des appels reproject_interp (et éventuellement création WCS)
  => laisse le pool, mais sérialise la partie dangereuse (souvent suffisant contre deadlocks wcslib).
Choisir l’option la plus minimale/fiable.

### B) Watchdog + trace en cas de hang
Activer faulthandler dans ce module (ou juste dans la fonction):
- `import faulthandler`
- `faulthandler.enable(all_threads=True, file=open(<temp>/faulthandler_intertile.log,"w"))`
- `faulthandler.dump_traceback_later(600, repeat=True)` pendant l’intertile
But: si hang, on récupère des stacks Python (même si natif, parfois utile).

### C) Heartbeat de progression dans les logs
Dans la boucle as_completed:
- toutes les N paires (ex: 25 ou 50), logger: "[Intertile] progress pairs_done=X/Y"
Ça permet de voir dans le .log où ça s’arrête, sans se fier uniquement au GUI.

## Contraintes
- Ne pas toucher aux comportements "batch size = 0" et "batch size > 1" ailleurs.
- Patch localisé, pas de refactor global.
- Conserver la logique actuelle de calcul des overlaps et des gains.

## Livraison
- Patch git-ready
- Un mini test manuel décrit (run dataset existant, vérifier que Intertile termine sur Windows)
