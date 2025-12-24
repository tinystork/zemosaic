# followup.md

## Notes d’implémentation (guidelines)
- Garder le helper **local** au module intertile si possible.
- Toutes les dépendances optionnelles (psutil, cv2) doivent être **guarded** :
  - pas d’import global obligatoire si ça risque de casser sur certaines machines.
- Ne pas modifier la logique générale de calcul des paires (pas besoin).

## Scénarios de test (rapides)
1) **Cas repro**
- Dataset où le log indiquait :
  - workers=15
  - pairs≈4344
  - preview=512
Attendu :
- log devient `workers=15 -> 4/6 ... pairs=4344 preview=512`
- run termine sans crash.

2) **Petit dataset**
- pairs < 200
Attendu :
- workers effectif reste “raisonnable” (jusqu’à 6–8)
- pas de régression perf notable

3) **RAM faible (simulation)**
- Forcer le helper via test (available_mb=3000)
Attendu :
- clamp à 2–4

## Checklist patch
- [x] Localiser exactement où est loggé `[Intertile] Parallel: threadpool workers=...`
- [x] Ajouter `compute_intertile_workers_limit(...)` (ou nom similaire)
- [x] Utiliser `effective_workers` dans `ThreadPoolExecutor`
- [x] Ajouter log “requested -> effective + reasons”
- [x] (Optionnel) `cv2.setNumThreads(1)` best-effort pendant Intertile
- [x] Vérifier lint/format, pas d’imports cassants
- [x] Run local rapide si possible

## Commit message suggéré
`Clamp intertile ThreadPool workers automatically to prevent Windows crashes on large pair counts`

## Si tu hésites sur le cap (4/6/8)
- Préférer la stabilité : 4–6 sur Windows quand pairs >= 4000.
- Sur Linux/macOS : 6–8 max.
