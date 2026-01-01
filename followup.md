# Suivi — Mission “coverage intra-cluster → alpha mask lecropper” (Master Tiles)

## Contexte
Objectif : mitiger les aberrations / franges chromatiques sur master tiles en masquant (alpha) les zones à faible recouvrement intra-cluster, sans réécrire le stacker.

## Préconditions (anti-régression)
- Le comportement doit rester identique si la coverage est absente / invalide (fallback lecropper existant).
- Limiter l’activation au chemin Master Tile (pas Phase 4.5 / final mosaic, sauf demande explicite).
- Ne pas impacter le mode GUI “I'm using master tiles” (`use_existing_master_tiles_mode` / `existing_master_tiles_mode`) : skip clustering & master tile creation → pas de coverage intra-cluster possible, donc no-op.
- Conserver `_apply_lecropper_pipeline(arr, cfg)` appelable avec 2 args (nouveaux params optionnels uniquement).
- Ne payer le coût `propagate_mask=True` / coverage que si l’Alt-Az cleanup Master Tile est activé (et si `lecropper` est dispo).
- N’injecter des NaN que sur des buffers float32 (sinon convertir en copie ou ignorer la nanisation).
- Ne pas laisser de NaN partir vers le stacker : `stack_aligned_images` loggue en ERROR quand des non-finis sont présents → nettoyer (`np.nan_to_num`) avant `_stack_master_tile_auto` ou calculer la coverage via masque séparé.
- Ne pas impacter les modes `GRID` et `SDS/SupadupStack` : `GRID` return tôt (`zemosaic_worker.py:21028`), `SDS` saute les Master Tiles (sauf fallback) (`zemosaic_worker.py:19050`, `zemosaic_worker.py:23554`).

## Check-list (à exécuter par Codex)
- [x] `zemosaic_worker.py:12108` : appeler `align_images_in_group(..., propagate_mask=True)` **uniquement** sur le chemin Master Tile *et seulement si* l’Alt-Az cleanup Master Tile est actif (évite un changement global + un coût inutile).
- [x] `zemosaic_align_stack.py:2075` : exploiter `footprint_mask` de `astroalign.register(...)` ; footprint 2D (gérer 2D/3D, bool/float) en s’alignant sur `aligned_image_output.shape` (HWC/CHW) et binariser (`>0`) si float ; si ambigu/mismatch → log + ignore coverage. Naniser hors-footprint sur **`aligned_image_output` uniquement** (ne pas polluer `src_for_aa`/pré-alignement FFT), et garantir float32 (sinon copier/convertir avant). Si `footprint_mask` est absent/invalide → ne pas naniser (fallback). En FFT-only (repli), naniser hors-overlap sur la **sortie retournée** (rectangle déduit de `dy/dx` + shape) si on veut un coverage fiable. **Compat** : si `astroalign.register` ne supporte pas `propagate_mask`, retry sans kwarg (no-op coverage).
- [x] `zemosaic_worker.py:12133` : calculer `coverage_count_hw` (float32) sur `valid_aligned_images` **avant** `_stack_master_tile_auto` (avant les `nan_to_num` du stacker) ; pixel valide si RGB finites **ou** `footprint2d` si dispo, accumulation incrémentale ; shapes homogènes sinon log + coverage=None. Logger min/max/nonzero_frac, borner `min_coverage_abs` à `n_used_for_stack`, et si `max_coverage <= 0` ou non-fini → log + coverage=None.
- [x] `zemosaic_worker.py:12133` : si nanisation utilisée pour la coverage, nettoyer les images (`np.nan_to_num`) **avant** le stack pour éviter les logs `STACK_IMG_PREP` en ERROR.
- [x] `zemosaic_worker.py:12379` : si `quality_crop_rect` appliqué, cropper `coverage_count_hw` pareil ; juste avant lecropper, vérifier `coverage.shape == img.shape[:2]` **et** `coverage.max() > 0` sinon log + ignore.
- [x] `zemosaic_worker.py:665` / `zemosaic_worker.py:752` : étendre `_apply_lecropper_pipeline` (coverage + seuils `altaz_min_coverage_abs/frac`, `altaz_morph_open_px`) en restant compatible 2 args ; appeler `mask_altaz_artifacts(..., coverage=..., min_coverage_*, morph_open_px=...)` et en cas de `TypeError` refaire l’appel sans ces kwargs (le `nanize` reste contrôlé via `altaz_nanize_threshold`).
- [x] `zemosaic_worker.py:12441` : passer la coverage + seuils au call `_apply_lecropper_pipeline(...)` côté Master Tile (Phase 4.5/fin mosaïque inchangés grâce aux defaults).
- [x] Ajuster paramètres (reco départ) : `min_coverage_frac=0.5`, `min_coverage_abs=3` (borne par `n_used_for_stack`), `morph_open_px=3`; garder `altaz_nanize_threshold` cohérent avec `altaz_alpha_soft_threshold` si utilisé (et **propager ces 2 seuils depuis la config**).
- [x] Logs debug : footprint stats dans `align_images_in_group` (`_pcb(..., lvl="DEBUG_DETAIL")`), stats coverage (`pcb_tile`), log “coverage ignored” explicite, seuils coverage avant lecropper, + vérifier `ALPHA_STATS`/`MT_PIPELINE` et absence de trim excessif (`MT_EDGE_TRIM`).
- [x] Lancer `pytest -q` (focus : `tests/test_create_master_tile_*.py`). Note WSL/Windows: si `pytest` crash sur `TemporaryFile.truncate()`, lancer avec `TMPDIR=/tmp`.

## Logs à vérifier (minimum)
- Mode actif (Phase 3) : `MT_PIPELINE_FLAGS`, `MT_PIPELINE`, `ALPHA_STATS: level=master_tile`.
- Coverage (debug_tile) : `MT_COVERAGE_STATS` / `MT_COVERAGE_IGNORED` + absence de `STACK_IMG_PREP` en ERROR.
- Non-régression : `[GRID] Invoking grid_mode.run_grid_mode(...)`, `[SDS] Phase 5 ... (skipping master tiles)`, `run_info_existing_master_tiles_mode` / `existing_master_tiles_mode:`.

## Etat
- Mission complète ; tests: `TMPDIR=/tmp python3 -m pytest -q` → `27 passed, 3 skipped`.

## Notes / Observations (à remplir)
- Réduction des franges sur tiles : …
- Champ perdu (si trop agressif) : …
- Stat alpha (nonzero_frac / min/max) dans logs : …
