# followup.md — Validation & mesures (Auto-organize perf)

## 1) Repro perf [x]
1. Ouvrir le Filter raw frames (Qt).
2. Charger un dataset “lourd” (≈ 1000+ frames).
3. (Optionnel) Faire “Analyse” si c’est ton flux habituel.
4. Cliquer “Auto-organize Master Tiles”.
5. Noter:
   - durée totale (chronomètre)
   - CPU global approx (gestionnaire tâches)
   - si overlay disparaît bien à la fin

## 2) Collecte logs [x]
Récupérer:
- `zemosaic_filter.log` (celui que tu fournis habituellement)
- Optionnel: capture de l’Activity log dans la fenêtre Qt

Attendu dans le log:
- `[AutoGroupTiming] build_candidates=...s prefetch_eqmode=...s clustering=...s autosplit=...s borrowing=...s auto_optimiser=...s total=...s`
- `[AutoOptimiser] start ...`
- `[AutoOptimiser] final ...`

## 3) Scénarios de non-régression [x]
A) Dataset small (50–200 frames): auto-organize doit rester correct.
B) Dataset mix EQ/ALTZ: les groupes ne doivent pas fusionner entre signatures.
C) Dataset avec EQMODE manquant: doit fonctionner (cache json et fallback header).
D) Erreur volontaire: sélectionner 0 frames / bbox vide:
   - overlay ne doit pas rester bloqué
   - message clair dans log/status

## 4) Comparaison qualité [x]
Comparer “avant/après”:
- nb de groupes final
- distribution des sizes (min/median/max)
- cohérence visuelle dans Sky Preview (pas de regroupements absurdes)

## 5) Si encore lent [ ]
Si `build_candidates` domine:
- activer le préchargement headers en ThreadPool (étape 3 du plan).
Si `clustering` domine:
- on aura besoin d’optimiser côté worker (hors scope de ce patch).