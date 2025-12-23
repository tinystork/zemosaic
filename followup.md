# Follow-up — Validation & checks (Phase 5 Intertile multithread)

## 1) Vérification dans les logs
Lancer un run avec logging=Debug.
Attendu dans le log :
- Une ligne du type :
  `[Intertile] Parallel: threadpool workers=... pairs=... preview=...`

Comparer le timing entre :
- `[Intertile] Using: ... pairs=...`
et
- la fin de l’étape intertile (ou le prochain gros jalon après l’intertile).

## 2) Vérification comportement GUI
- La barre de progression "Phase5 Intertile Pairs" doit continuer à avancer régulièrement.
- Pas de spam excessif (si nécessaire, n’updater le progress_callback que toutes les X paires, ex: 1% ou toutes les 5 paires).

## 3) Vérification perf OS
Sur Windows Task Manager :
- Pendant Intertile : CPU doit monter sensiblement (plusieurs cœurs actifs).
- GPU peut rester bas (normal) : l’objectif ici est CPU, pas GPU.

## 4) Vérification résultat
- Comparer la mosaïque finale avant/après patch :
  - pas de changement de géométrie
  - pas de bandes/artefacts nouveaux
  - photometric match toujours appliqué

## 5) Fallback sécurité (si problème)
- Si crash ou instabilité : forcer `cpu_workers=1` (en retenant processing_threads=1) et confirmer que le séquentiel marche toujours.
- Garder un chemin séquentiel simple et fiable.
