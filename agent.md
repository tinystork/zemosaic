# agent.md — ZeMosaic Filter Qt: accélérer Auto-organize + fiabiliser l’overlay

## Contexte
Dans `zemosaic_filter_gui_qt.py`, le bouton **Auto-organize Master Tiles** peut prendre très longtemps.
Observation utilisateur: CPU global ~10% (symptôme mono-thread) et l’overlay GIF reste affiché “une éternité”.
On suspecte un coût algorithmique (auto-optimiser / merges) et/ou des lectures FITS séquentielles si des champs manquent.

## Objectifs
- [x] 1) Réduire drastiquement le temps “Auto-organize” sur des datasets ~1000–1500 frames.
- [x] 2) Garder un comportement utilisateur identique (pas de refactor, pas de changements UI visibles, pas de changement de logique EQ/ALTZ).
- [x] 3) S’assurer que l’overlay GIF est **toujours** arrêté (succès, erreur, exception, early-return).

## Portée / contraintes
- Modifier **uniquement** `zemosaic_filter_gui_qt.py`.
- Pas de refactor large (pas de déplacement de classes/fichiers).
- Ne pas toucher au worker module / GPU / autres phases du pipeline.
- Ne pas changer les règles de split EQ/ALTZ.
- Ajouter du logging “timing” minimal et lisible.

## Hypothèses de goulots
- [x] A) `_optimize_auto_group_result()` peut être coûteux car:
   - `_merge_group_records_for_auto()` appelle `_find_auto_merge_partner()`
   - `_find_auto_merge_partner()` construit des `combined_coords` et calcule une dispersion *pour beaucoup de candidats*
   - Complexité potentiellement énorme (beaucoup de merges x beaucoup de voisins x dispersion coûteuse).
- [ ] B) Construction de `candidate_infos` peut charger des headers FITS séquentiellement si pas déjà en cache.

## Plan d’implémentation (surgical)
### 1) Instrumentation timing (sans spam) [x]
Ajouter des timings `time.perf_counter()` dans le thread d’auto-group pour mesurer:
- build candidate payloads
- prefetch EQMODE (cache json déjà existant)
- clustering (_CLUSTER_CONNECTED)
- autosplit/merge worker side
- borrowing v1
- auto-optimiser (merge/split/merge)
Log:
- vers `logger.info(...)`
- + ajouter 1–2 lignes synthèse dans `messages.append(...)` (pas une ligne par item)

Critère: on doit pouvoir lire dans le log où part le temps.

### 2) Accélération AutoOptimiser: éviter la dispersion exacte à chaque candidat [x]
But: **ne plus** faire `combined_coords = source + other` pour chaque voisin testé.

Approche “borne sûre” (triangle inequality) :
- Pour chaque record, stocker:
  - `center` (déjà)
  - `radius` = max distance(center, point) (calcul O(n) à la création/merge)
  - `dispersion` (peut rester l’exact initial si dispo, sinon approx)
- Pour évaluer un merge candidat (source, other):
  - `dist = angular_distance(source.center, other.center)`
  - `cross_upper = dist + source.radius + other.radius`
  - `disp_upper = max(source.dispersion, other.dispersion, cross_upper)`
  - Si `disp_upper > dispersion_limit` => impossible (rejet)
  - Sinon => merge “sûr” (la vraie dispersion ne peut pas dépasser cette borne)

Ainsi:
- `_find_auto_merge_partner()` ne calcule plus la dispersion exacte par candidat.
- Il choisit le meilleur partenaire avec une métrique stable (size_score puis disp_upper puis dist).

Au moment du merge effectif:
- Construire `combined_entries` (comme avant).
- Pour `coords`:
  - concaténer `keep.coords + drop.coords` **une seule fois**.
  - recalculer `center` via moyenne (comme avant).
  - recalculer `radius` en un seul passage O(n).
  - mettre `dispersion` à `disp_upper` (borne conservative) ou `max(old, 2*radius)` si nécessaire.
=> plus d’appel coûteux `compute_max_separation(combined_coords)` dans la boucle.

Option bonus (safe):
- Ajouter un cap sur le nombre de voisins testés (ex: 64 plus proches) pour éviter O(n²) inutile.
  - Valeur par défaut conservative, configurable via `_config_value("auto_optimiser_neighbor_cap", 64)` si tu veux.

### 3) Option I/O: préchargement headers en parallèle si nécessaire (simple) [ ]
Si l’instrumentation montre que la phase “build candidate payloads” est dominante:
- Détecter les entrées sélectionnées dont `header`/`header_cache` sont absents.
- Précharger via `ThreadPoolExecutor`:
  - appeler `_load_header(path)` en threads
  - protéger `_header_cache` via un `threading.Lock()` (ajouter un lock dans la classe).
- Objectif: accélérer les lectures FITS sur disques rapides, sans casser la logique.

### 4) Overlay: stop garanti [x]
- Vérifier que les early-returns de `_start_master_tile_organisation()` et `_handle_auto_group_empty_selection()` cachent l’overlay.
- Dans `_handle_auto_group_finished()` : appeler `_hide_processing_overlay()` dans un `finally` (ou le mettre tout en bas mais garantir qu’aucun return ne l’évite).
- Dans `_auto_group_background_task()` : entourer la totalité par try/except/finally si besoin, mais surtout garantir l’emit du signal.

## Critères d’acceptation
- [x] Sur dataset ~1200–1500 frames: Auto-organize passe de “minutes” à “quelques secondes / dizaines de secondes max”.
- [x] Le log contient une ligne de timing par sous-phase + un résumé AutoOptimiser.
- [x] L’overlay se ferme toujours (succès/erreur).
- [x] Pas de modification du comportement utilisateur (mêmes boutons, mêmes options, mêmes séparations EQ/ALTZ).

## Fichiers
- `zemosaic_filter_gui_qt.py` uniquement.