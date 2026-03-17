# followup.md

# ZeMosaic — Follow-up checklist
## Mission seamless mosaic + viewer preview quality

Legend:
- `[ ]` not done
- `[x]` done
- `[~]` partial
- `BLOCKED:` reason

Reference dataset:
- `/home/tristan/zemosaic/zemosaic/example/out/ref/`

---

## A. Discipline mission

- [x] Lire `agent.md`, `followup.md`, `memory.md` avant chaque itération
- [x] Travailler sur le prochain item non coché (patchs chirurgicaux)
- [x] Prouver chaque claim (logs + outputs + diff visuel)
- [x] Mettre à jour `memory.md` à chaque itération significative

---

## B. Audit initial (code reality check)

### B1. Viewer PNG stretch — état actuel confirmé
- [x] Identifier le point d’entrée preview final dans `zemosaic_worker.py` (Phase 6)
- [x] Confirmer paramètres hardcodés actuels preview:
  - `preview_p_low = 2.5`
  - `preview_p_high = 99.8`
  - `preview_asinh_a = 20.0`
- [x] Confirmer downscale preview cap (`max_preview_dim = 4000`) et masquage alpha/NaN
- [x] Confirmer dépendance à `stretch_auto_asifits_like(_gpu)` dans `zemosaic_utils.py`

### B2. Seams inter-tuiles — mécanismes existants confirmés
- [x] `zemosaic_worker.py` (classic/incremental/reproject):
  - intertile affine calibration (`intertile_*`),
  - background matching,
  - recenter global,
  - radial feather parity.
- [x] `grid_mode.py`:
  - overlap graph + régressions d’overlap,
  - solve gain/offset global,
  - blending overlap (laplacian pyramid + fallback weighted blend),
  - fusion unique vs overlap regions.
- [x] Lister les leviers de config existants à réutiliser avant d’en créer de nouveaux

### B3. Baseline de référence (avant patch)
- Preuves exportées:
  - `example/out/ref/preview_baseline_metrics_2026-03-15.md`
  - `example/out/ref/preview_baseline_metrics_2026-03-15.json`
- [x] Capturer comparaison visuelle/metrics sur `example/out/ref`:
  - visibilité des seams,
  - clipping hautes lumières PNG,
  - rendu du fond (propreté/perception)
- [x] Établir un mini tableau baseline par mode (au moins classic + ZeGrid)

---

## C. Sprint 1 prioritaire — PNG viewer (quick win)

> Ordre validé avec Tristan: traiter d’abord le viewer PNG (plus simple/rapide), puis enclencher la 1ère tentative seams.

### C1. Plan de correction viewer
- [x] Remplacer les constantes preview hardcodées par paramètres config-gated:
  - `preview_png_p_low`
  - `preview_png_p_high`
  - `preview_png_asinh_a`
  - `preview_png_max_dim`
- [x] Définir defaults conservateurs orientés rendu naturel (moins brûlé)
- [x] Ajouter logs explicites des paramètres réellement appliqués au run

### C2. Implémentation viewer
- [x] Implémenter lecture config + fallback propre
- [x] Préserver compat GPU/CPU stretch
- [x] Préserver alpha-masking/NaN behavior

### C3. Validation viewer
- Retour run utilisateur (latest): preview passé de “brûlé” à “trop sombre”, puis “un peu trop lumineux” après retunes successives.
- Cible actuelle: compromis intermédiaire stable (lisible sans brûler les coeurs).
- Retour run utilisateur: amélioration preview confirmée mais coeurs étoiles/galaxies encore brûlés.
- Retune v2 préparé (JSON-only): `preview_png_p_low=0.3`, `preview_png_p_high=99.97`, `preview_png_asinh_a=0.1`.
- [x] Run A (baseline actuel)
- [~] Run B (nouveau preset conservateur)
- [ ] Vérifier:
  - moins de blancs brûlés,
  - fond moins agressif,
  - pas de régression FITS (science inchangée)
- [~] Décision rapide: keep / tune / revert

---

## D. Sprint 2 — Seams (première tentative contrôlée)

### D1. Priorisation du chemin le plus impactant
- Cible seams #1 retenue: chaîne **classic/reproject-like MT14 preview outputs** (seam proxy le plus élevé sur baseline ref).
- Hypothèse dominante (en cours de preuve): compensation photométrique inter-tuiles insuffisante localement + transition overlap/weight trop lisible sur zones à gradient de fond.
- [x] Choisir cible #1 seams (ZeGrid ou reproject classic) selon baseline visuelle
- [~] Isoler la cause dominante:
  - mismatch local de fond,
  - transition de poids,
  - limite overlap regression,
  - feather inadapté

### D2. Correctif seams v1 (minimal risk)
- Retour run utilisateur: seams toujours très visibles, composante bleue sur-corrigée.
- Retune seams v2 préparé (JSON-only): `poststack_equalize_rgb=false`, `intertile_affine_blend=0.65`, `intertile_recenter_clip=[0.92,1.08]`, `apply_radial_weight=true`, `radial_feather_fraction=0.90`.
- Levier D2 v1 implémenté (JSON only, no UI): `intertile_affine_blend` (0..1), appliqué sur corrections gain/offset inter-tuiles avant assemblage.
- Valeur posée pour validation terrain: `intertile_affine_blend=0.8`.
- [x] Implémenter un seul levier principal à la fois (config-gated)
- [x] Conserver garde-fous anti-surcorrection
- [~] Ajouter logs comparables avant/après (seam delta local)

### D3. Validation seams v1
- Diagnostic-safe profile préparé pour isolation color drift post-refactor:
  - `poststack_equalize_rgb=false`
  - `final_mosaic_rgb_equalize_enabled=false`
  - `final_mosaic_black_point_equalize_enabled=false`
  - `final_mosaic_dbe_enabled=false`
  - seams retune: `intertile_affine_blend=0.50`, `intertile_recenter_clip=[0.95,1.05]`, `intertile_overlap_min=0.10`, `intertile_robust_clip_sigma=2.0`, `radial_feather_fraction=0.92`
- [ ] Run comparatif avant/après sur dataset ref
- [ ] Évaluer réduction visible des coutures
- [ ] Vérifier absence de nouveaux artefacts (banding, halos de transition)

---

## E. Extension inter-modes (après v1 validée)

- [ ] Reporter la correction seams (si pertinente) vers autres modes sans copier brutalement
- [ ] Vérifier comportement sur:
  - Classique
  - Existing master tiles
  - SDS
  - ZeGrid
- [ ] Ajuster par-mode uniquement si nécessaire

---

## F. Non-régression transversale

- [ ] FITS science inchangés dans leur logique (pas de stretch destructif)
- [ ] Coverage/alpha cohérents
- [ ] Preview PNG généré sans erreur sur tous modes testés
- [ ] Aucun crash worker/GUI introduit
- [ ] Tests ciblés passants (unit/smoke)

---

## G. Clôture mission

- [ ] Rapport final seamless + preview
- [ ] Paramètres finaux recommandés (defaults + cas marginaux)
- [ ] GO / NO-GO production
- [ ] Mise à jour durable `memory.md` (synthèse exploitable debug prod)


## H. Dossier `poststack_equalize_rgb` (drift chromatique)

- [x] Confirmer corrélation terrain: `poststack_equalize_rgb=false` supprime les aberrations de courbe RGB sur dataset test
- [x] Isoler cause algorithmique actuelle:
  - médianes globales par canal sur sous-stack
  - absence de masque robuste fond/objets
  - gains non bornés assez strictement pour dataset pauvre
- [x] Proposer/implémenter version robuste v2:
  - masque fond valide + exclusion objets brillants
  - clip gain conservateur (`[0.95,1.05]` par défaut)
  - no-op si fiabilité insuffisante (samples/overlap)
- [x] Ajouter télémétrie explicite: `samples`, `mask_coverage`, `raw_gains`, `clipped_gains`, `applied/no-op`
- [x] Politique produit jusqu’à validation:
  - `poststack_equalize_rgb=false` par défaut
  - documentation claire du risque “drift chromatique”


---

## I. Réévaluation 2026-03-16 (dataset plus lourd, focus seams)

### I1. Ajustement de proposition
- [x] Réévaluer la proposition précédente à la lumière du log lourd (`zemosaic_worker.log`)
- [x] Corriger le point de design: conserver `intertile_overlap_min=0.05` (ne pas monter à 0.10 sur ce dataset)
- [x] Formaliser un profil unique "VISUAL_SEAMLESS_v1" documenté (voir agent.md addendum)

### I2. Paramètres du profil VISUAL_SEAMLESS_v1
- [x] `poststack_equalize_rgb=false`
- [x] `intertile_affine_blend=0.40`
- [x] `intertile_recenter_clip=[0.96,1.04]`
- [x] `intertile_overlap_min=0.05`
- [x] `intertile_robust_clip_sigma=2.0`
- [x] `apply_radial_weight=true`
- [x] `radial_feather_fraction=0.94`
- [x] `radial_shape_power=2.6`
- [x] `final_mosaic_dbe_enabled=true`
- [x] `final_mosaic_dbe_strength=normal`
- [x] `final_mosaic_dbe_smoothing=0.75`
- [x] `final_mosaic_dbe_sample_step=20`
- [x] `final_mosaic_dbe_obj_dilate_px=4`
- [x] `preview_png_apply_wb=false`
- [x] `preview_png_p_low=0.40`
- [x] `preview_png_p_high=99.93`
- [x] `preview_png_asinh_a=0.14`

### I3. Suite planifiée (reportée)
- [ ] Implémenter un pass optionnel "seam-heal low-frequency" (rendu visuel)
- [ ] Valider A/B: science conservatrice vs rendu visuel (2 runs max)
- [ ] Décider si preset "VISUAL_SEAMLESS_v1" devient preset GUI explicite

### I4. Guardrails — future Visual Seam Heal (do not lose sight of this)
- [ ] Prepare a GUI option before `Save final mosaic as uint16`
- [ ] Keep this feature explicitly visual-only (not a science/FITS correction)
- [ ] Preserve strict separation between Classic / Existing master tiles / SDS / ZeGrid
- [ ] Implement as config-gated, conservative, OFF by default
- [ ] Prefer Phase 6 / final visual output integration, not upstream stack math
- [ ] Be extremely careful with RGB drift; luma-first approach preferred for V1
- [ ] Avoid halos, banding, and hard local corrections
- [ ] Add every persisted key to `DEFAULT_CONFIG`
- [ ] Update `memory.md` after each significant iteration
