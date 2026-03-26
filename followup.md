# followup.md

## Existing content

# followup.md

# ZeMosaic — Follow-up checklist
## Seam root-cause elimination (Classic first)

Legend:
- `[ ]` not done
- `[x]` done
- `[~]` partial
- `BLOCKED:` reason

Reference principle:
- Work on the next unchecked item.
- Patch surgically.
- Prove every claim.
- Update `memory.md` after each significant iteration.

---

## A. Discipline

- [ ] Lire `agent.md`, `followup.md`, `memory.md` avant chaque itération
- [ ] Travailler sur le prochain item non coché
- [ ] Patchs chirurgicaux uniquement
- [ ] Prouver chaque claim par logs + images + comparaison
- [ ] Mettre à jour `memory.md` à chaque itération significative
- [ ] Ne pas toucher au comportement stable `batch size = 0` / `batch size > 1`
- [ ] Ne pas élargir le scope aux autres modes tant que Classic n’est pas proprement instrumenté

---

## B. Freeze baseline and assets

- [ ] Geler le dataset de référence principal “gros dataset hétérogène”
- [ ] Geler les sorties benchmark déjà produites comme baseline
- [ ] Identifier clairement les fichiers de comparaison:
  - weighting OFF
  - weighting ON actuel
  - coverage maps
  - diagnostics V3
- [ ] Documenter dans `memory.md` quels outputs servent désormais de référence officielle

---

## C. Instrumentation — real pipeline first

### C1. Graphe réellement retenu
- [ ] Exporter le graphe final réellement retenu par le pipeline worker
- [ ] Exporter pour chaque arête retenue:
  - tile_i
  - tile_j
  - score / strength
  - overlap
  - métriques photométriques utiles
- [ ] Exporter les arêtes rejetées mais fortes
- [ ] Logger explicitement les raisons principales de rejet si disponible

### C2. Solve photométrique
- [ ] Exporter l’ancre réellement choisie
- [ ] Exporter les gains / offsets par tuile avant/après solve
- [ ] Exporter les résidus par arête après harmonisation
- [ ] Exporter un résumé global:
  - residual median
  - residual p95
  - worst edges
  - overlap_mean des paires actives

### C3. Weighting final
- [ ] Exporter les poids finaux réellement appliqués par tuile
- [ ] Logger leur source exacte:
  - `MT_NFRAMES`
  - fallback
  - autre source éventuelle
- [ ] En mode “existing master tiles”, logger:
  - nombre de tuiles avec poids valide
  - nombre de tuiles fallback=1.0
- [ ] Produire une `weighted_coverage_map`
- [ ] Produire une `winner_map` / `dominant_tile_map`

### C4. Corrélation visuelle
- [ ] Vérifier si les seams visibles suivent:
  - la winner map
  - la weighted coverage map
  - certaines arêtes à fort résidu
- [ ] Documenter conclusion courte dans `memory.md`

---

## D. Revalidation of current hypothesis

- [ ] Confirmer ou infirmer que le problème principal est:
  - graphe trop simplifié
  - homogénéisation insuffisante
  - weighting trop dominateur
  - combinaison des trois
- [ ] Éviter toute conclusion basée uniquement sur simulation externe
- [ ] Prioriser les preuves issues du pipeline réel ZeMosaic

---

## E. Graph rework experiments

### E1. Pruning
- [ ] Identifier précisément la logique actuelle de top-K / pruning utilisée sur le gros dataset
- [ ] Tester une variante moins brutale
- [ ] Tester une variante top-K adaptative selon densité locale
- [ ] Vérifier si des arêtes fortes aujourd’hui rejetées doivent être conservées

### E2. Scoring des arêtes
- [ ] Étudier un score combiné:
  - overlap
  - stabilité photométrique
  - cohérence temporelle
- [ ] Éviter qu’une arête géométriquement correcte mais photométriquement aberrante domine
- [ ] Éviter aussi de jeter trop tôt des arêtes photométriquement utiles

### E3. Cohortes temporelles
- [ ] Évaluer si le dataset doit être traité avec une conscience temporelle plus explicite
- [ ] Tester l’idée de cohortes/session groups si l’étalement temporel reste une cause forte
- [ ] Documenter GO / NO-GO sur cette piste

---

## F. Weighting V4

### F1. Politique conservatoire immédiate
- [ ] Considérer `master tile weighting = OFF` comme baseline conservatoire tant que V4 n’est pas validé
- [ ] Documenter clairement si cette position doit devenir le défaut temporaire

### F2. Design V4
- [ ] Introduire une compression forte de dynamique:
  - `sqrt`
  - `log`
  - soft-cap robuste
  - ou équivalent
- [ ] Éviter qu’une tuile “forte” domine massivement une tuile “faible”
- [ ] Réduire la domination globale au centre des tuiles
- [ ] Favoriser un plateau intérieur plus neutre
- [ ] Garder le weighting surtout utile dans les zones de couture / feather

### F3. Guardrails V4
- [ ] Ajouter un plafond automatique si les résidus photométriques d’une tuile restent élevés
- [ ] Ajouter une pénalité temporelle si une tuile est très atypique
- [ ] Tout rendre config-gated
- [ ] Ajouter toute clé persistée à `DEFAULT_CONFIG`

### F4. Telemetry V4
- [ ] Logger avant/après:
  - distribution des poids
  - min / median / max
  - nombre de tuiles plafonnées / pénalisées
  - effet sur winner map
  - effet sur résidus / seams proxy

---

## G. Validation protocol

- [ ] Comparer sur mêmes master tiles:
  - OFF
  - ON actuel
  - ON V4
- [ ] Comparer visuellement:
  - seams rectangulaires
  - fond
  - homogénéité locale
  - couleur
- [ ] Comparer quantitativement:
  - dispersion photométrique par overlap
  - résidu moyen par arête
  - residual p95
  - variation de fond par tuile
  - intensité de domination locale

---

## H. Visual seam-heal — postponed, not mainline

- [ ] Ne pas implémenter le seam-heal low-frequency avant d’avoir traité C/D/F
- [ ] Garder l’idée vivante uniquement comme finition visuelle
- [ ] Si réactivé plus tard:
  - luma-first
  - visual-only
  - OFF par défaut
  - Phase 6 / output final
  - pas de modification science/FITS

---

## I. Non-regression

- [ ] Pas de régression géométrique
- [ ] Pas de dérive couleur induite
- [ ] Pas de halos / banding / zones molles
- [ ] Pas de crash worker/GUI
- [ ] Pas de casse sur Classic / Existing master tiles
- [ ] Pas de changement indésirable de comportement batch stable

---

## J. Mission close

- [ ] Root cause prouvée de façon crédible
- [ ] Pipeline réel instrumenté
- [ ] Graph rework et/ou weighting V4 validés ou clairement rejetés
- [ ] Baseline conservatoire documentée
- [ ] `memory.md` mis à jour avec synthèse GO / NO-GO

---

## K. 2026-03-20 — Proto V4 / Pruning runtime / Discipline anti-régression

### K1. Proto V4 (baseline de test)
- [x] Ajouter clés config persistées `tile_weight_v4_*` dans `DEFAULT_CONFIG`
- [x] Brancher proto V4 en mode config-gated (OFF par défaut)
- [ ] Ajouter pénalité résidu photométrique par tuile (guardrail qualité)
- [ ] Ajouter pénalité temporelle optionnelle

### K2. Pruning runtime configurable
- [x] Ajouter `intertile_prune_k` (runtime)
- [x] Ajouter `intertile_prune_weight_mode` (`area|strength|hybrid`)
- [x] Brancher ces paramètres dans le calcul intertile réel
- [ ] Vérifier logs terrain: `Pair pruning summary ... K=... mode=...`

### K3. RUN A/B mini-dataset
- [x] Préparer RUN A (V4 OFF explicite + prune explicite)
- [ ] Exécuter RUN A et archiver sortie/logs/crops
- [ ] Préparer RUN B (V4 ON, autres paramètres constants)
- [ ] Exécuter RUN B et comparer à RUN A

### K4. Discipline anti-régression (obligatoire)
- [ ] Pour chaque patch Classic, lister la zone partagée touchée
- [ ] Refaire un smoke check ZeGrid + SDS après patch Classic
- [ ] Reporter preuve dans `memory.md` (compile/tests/run + risque résiduel)


---

## L. Mission supplémentaire — Fiabiliser le Quality Gate

Objectif: empêcher qu’un run « techniquement réussi » sorte un résultat visuellement/scientifiquement invalide (master tile corrompue, dynamique aberrante, stats incohérentes) sans alerte bloquante.

### L1. Définir les signaux de corruption / dérive
- [ ] Définir des seuils robustes par tuile:
  - médiane canal (R/G/B)
  - MAD canal
  - ratio max/median
  - fraction de pixels nuls / NaN
- [ ] Ajouter détection d’outlier robuste (IQR/MAD) sur la distribution inter-tiles
- [ ] Marquer explicitement les tuiles suspectes dans les logs (ex: `QUALITY_GATE_TILE_OUTLIER`)

### L2. Gates pré-assemblage (hard fail configurable)
- [ ] Ajouter un quality gate avant Phase 5 (reproject/coadd)
- [ ] Si tuile extrême détectée:
  - mode `warn`: continuer + alerte forte
  - mode `fail`: stopper run proprement avec message actionnable
- [ ] Exposer en config:
  - `quality_gate_enabled`
  - `quality_gate_mode` (`warn|fail`)
  - `quality_gate_tile_sigma_threshold` (ou équivalent robuste)

### L3. Gates post-assemblage (sanity mosaic)
- [ ] Ajouter contrôles globaux après assemblage:
  - plage dynamique globale
  - fraction NaN
  - cohérence inter-canaux (ratios robustes)
- [ ] Lever alerte bloquante si profil incompatible avec baseline dataset
- [ ] Exporter un résumé machine-readable (`quality_gate_report.json`)

### L4. Existing master tiles — durcissement spécifique
- [ ] Vérifier cohérence stricte entre index tuiles attendues et présentes
- [ ] Refuser silencieux impossible: si “trou + outlier extrême”, alerte explicite obligatoire
- [ ] Logger source des tiles et stats minimales lors du chargement (`tile_id -> median/mad/min/max`)

### L5. UX opérateur / observabilité
- [ ] Ajouter messages GUI clairs:
  - nombre de tuiles OK
  - nombre suspectes
  - action recommandée
- [ ] Ajouter résumé fin de run:
  - `QUALITY_GATE_STATUS=PASS|WARN|FAIL`
  - liste courte des tuiles concernées

### L6. Validation et non-régression
- [ ] Cas test nominal: dataset propre => PASS sans faux positif
- [ ] Cas test corruption connue (master tile saturée) => WARN/FAIL attendu
- [ ] Vérifier non-régression Classic / ZeGrid / SDS (smoke minimal)
- [ ] Documenter les seuils retenus et le rationnel dans `memory.md`

### L0. Alignement 2026-03-23 (constats validés)
- [x] Confirmer que le quality gate existant est bien câblé (GUI -> config -> worker -> accept/reject)
- [x] Confirmer que le quality gate actuel s'applique en création de master tiles (Phase 3)
- [x] Confirmer que le mode `use_existing_master_tiles=true` n'est pas actuellement protégé par ce gate
- [x] Confirmer sur logs/config terrain (`NGC6888_2`, `NGC6888_3`) que `quality_gate_enabled=false`
- [x] Confirmer root cause terrain: master tile outlier (`master_tile_125`) peut contaminer tout le coadd en existing-master mode

### L7. Recentrage mission (quality gate au service du pivot principal)
- [x] Ajouter un **pré-check quality gate dédié existing master tiles** avant Phase 5
- [x] Définir politique claire en existing mode:
  - `warn` = continuer mais signaler tuiles suspectes explicitement
  - `fail` = arrêt propre avant coadd si outlier critique
- [x] Garantir qu'un outlier photométrique massif (type `master_tile_125`) ne passe plus silencieusement
- [x] Journaliser un résumé de validation des masters existantes (count ok/suspect/reject) dans les logs run
- [ ] Garder le scope chirurgical: pas de refactor large, pas de dérive hors mission graph/weighting


## Imported from followup_prepivot_20260317.md

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


## Imported from CHANGELOG.md

## Unreleased
## Fixed
- **Legacy stacking color cast (green tint)**  
  Fixed a long-standing green color bias in legacy mode caused by **mixed CPU/GPU execution paths within the same RGB stack**.  
  The issue occurred when different color channels were processed through different backends (CPU vs GPU), leading to subtle but cumulative normalization divergences.
  ➤ The stacking pipeline is now **backend-consistent per stack**: a given stack is processed entirely on CPU or entirely on GPU.  
  ➤ Partial CPU/GPU channel splitting is explicitly forbidden to guarantee photometric and color integrity.
  This change restores color parity with historical legacy results and stabilizes RGB behavior across runs.

### Fixed
* Robust FITS writing for ASTAP CLI (colour order normalised)
* Filter window now opens once via the streaming scanner path in the GUI
* Restored lightweight WCS footprint previews on large datasets with updated guidance copy
* Winsorized Phase 3 stacking now auto-recovers from NumPy `ArrayMemoryError` by reducing batch sizes or streaming from disk
* Suppressed ASTAP "Access violation" pop-ups by coordinating solver launches through a cooperative inter-process lock (opt-out via `ZEMOSAIC_ASTAP_DISABLE_IPC_LOCK`)
* Hardened ASTAP concurrency handling: Windows `SetErrorMode`, per-image locks, isolated working dirs, rate limiting, and automatic retries (tunable via `ZEMOSAIC_ASTAP_RATE_*` / `ZEMOSAIC_ASTAP_RETRIES` / `ZEMOSAIC_ASTAP_BACKOFF_SEC`) keep the GUI responsive even when multiple tools solve simultaneously
### Added
* Solver option "Convert to Luminance" to force mono before plate-solve
* Added configurable `winsor_worker_limit` (CLI `--winsor-workers` / `-W` and GUI field)
* Added `winsor_max_frames_per_pass` streaming limit for Winsorized rejection (GUI/config)
* Manual frame cap via `max_raw_per_master_tile` (CLI/GUI/config)
* Automatic memory fallback controls for Winsorized stacking (`winsor_auto_fallback_on_memory_error`, `winsor_min_frames_per_pass`, `winsor_memmap_fallback`, `winsor_split_strategy`)
* Fixed incremental assembly with reproject>=0.11
* 16-bit FITS export now writes a 2D luminance primary with R/G/B extensions for broader viewer compatibility, with a legacy RGB cube available from Advanced options
* Added cross-platform GUI icon fallback using existing PNG assets
* Automatic ASTAP executable/data path detection across Windows/macOS/Linux, with environment variable overrides exposed via `zemosaic_config.detect_astap_installation`
* POSIX PyInstaller helper script (`compile/build_zemosaic_posix.sh`) plus macOS/Linux setup documentation

### Changed
* Improved cross-platform startup: GUI icon fallback, optional dependency handling, and GPU detection now degrade gracefully on macOS/Linux (CUDA acceleration remains Windows-only)
* `requirements.txt` no longer lists the unsupported `tk` wheel and only installs `wmi` on Windows, eliminating pip failures on macOS/Linux


## Imported from README.md

# 🌌 ZeMosaic

**ZeMosaic** is an open-source tool for assembling **large astronomical mosaics** from FITS images, with particular support for all-in-one sensors like the **Seestar S50**.

It was born out of a need from an astrophotography Discord community called the seestar collective stacking tens of **thousands of FITS images** into clean wide-field mosaics — a task where most existing tools struggled with scale, automation, or quality.

---

## 🚀 Key Features

> Note: `lecropper` remains an annex/standalone legacy tool and is not part of the official runtime path.

- Astrometric alignment using **ASTAP**
- Smart tile grouping and automatic clustering
- Configurable stacking with:
  - **Noise-based weighting** (1/σ²)
  - **Kappa-Sigma** and **Winsorized** rejection
  - Radial feathering to blend tile borders
- Two mosaic assembly modes:
  - `Reproject & Coadd` (high quality, RAM-intensive)
  - `Incremental` (low memory, scalable)
- Stretch preview generation (ASIFits-style)
- Official GUI built with **PySide6 (Qt)**, fully translatable (EN/FR)
- Flexible FITS export with configurable `axis_order` (default `HWC`) and
  proper `BSCALE`/`BZERO` for float images
- Option to save the final mosaic as 16-bit integer FITS
- Phase-specific auto-tuning of worker threads (alignment capped at 50% of CPU threads)
- Process-based parallelization for final mosaic assembly (both `reproject_coadd` and `incremental`)

- Configurable `assembly_process_workers` to tune process count for assembly (used by both methods)

- Optional CUDA acceleration for the Mosaic-First reprojection+coadd path (Phase 4). When
  `use_gpu_phase5` is enabled and a compatible CUDA device is detected, ZeMosaic now leverages the GPU
  for mean, median, winsorized sigma-clip, and kappa-sigma stacking modes.


---

## How It Works: ZeAnalyser & Grid Mode

### ZeAnalyser: Frame Quality Analysis and Selection

ZeAnalyser is the analysis engine used by ZeMosaic to evaluate the quality of individual frames before stacking.
Its goal is simple: keep signal, reject noise and pathological frames, without requiring calibration files or manual tuning.

For each input frame, ZeAnalyser computes a set of objective image quality metrics, such as:

*   Star detection and count (robust to noise and gradients)
*   Star shape statistics (eccentricity / elongation indicators)
*   Global sharpness / structure metrics
*   Noise and background behavior
*   Optional SNR-related estimations

These metrics are combined to:

*   Reject unusable frames (e.g., tracking errors, clouds, severe blur)
*   Weight or filter frames consistently across large datasets
*   Ensure homogeneous data quality before stacking

ZeAnalyser operates fully automatically and is designed to scale efficiently to tens of thousands of frames, making it suitable for long multi-night Seestar and traditional imaging sessions.

### Grid Mode: Mosaic-First Processing Strategy

Grid Mode introduces a mosaic-first approach, specifically designed for wide fields, large sky coverage, and datasets with variable overlap.

Instead of stacking everything into a single reference frame, the field of view is divided into a regular grid of tiles. Each tile is processed independently before being reassembled into the final mosaic.

Key steps in Grid Mode:

1.  **Spatial Partitioning**
    The sky coverage is divided into overlapping grid tiles. Each input frame contributes only to the tiles it actually covers.

2.  **Local Analysis with ZeAnalyser**
    ZeAnalyser is applied per tile, not globally. This allows for local quality decisions: a frame may be rejected in one tile but accepted in another. Local seeing, tracking, or distortion issues are handled naturally.

3.  **Independent Tile Stacking**
    Each tile is stacked using only frames validated for that tile. This improves local sharpness and signal consistency, while reducing edge artifacts and uneven coverage.

4.  **Tile Cropping and Normalization**
    Invalid or low-coverage borders are automatically trimmed. Tile intensity and background are normalized prior to reprojection.

5.  **Final Mosaic Assembly**
    All tiles are reprojected using WCS information. The final mosaic is assembled with consistent geometry and color behavior.

### Why Grid Mode Matters

Grid Mode solves several classic problems of wide-field and mosaic stacking:

*   Uneven frame overlap
*   Local tracking distortions
*   Field rotation and edge degradation
*   Quality variations across large datasets

By combining ZeAnalyser’s per-frame analysis with Grid Mode’s spatially aware stacking, ZeMosaic achieves:

*   Better local sharpness
*   Reduced ghosting and duplication artifacts
*   More stable mosaics on large or imperfect datasets
*   A processing strategy that remains robust as dataset size grows

### Design Philosophy

ZeAnalyser and Grid Mode are intentionally designed to be:

*   **Automatic** – minimal user tuning required
*   **Deterministic** – same input, same output
*   **Scalable** – from a few hundred to tens of thousands of frames
*   **Instrument-agnostic** – optimized for Seestar but not limited to it

Together, they form the backbone of ZeMosaic’s modern stacking pipeline.

---

## Fonctionnement : ZeAnalyser et Mode Grille (Français)

### ZeAnalyser : Analyse et Sélection de la Qualité des Images

ZeAnalyser est le moteur d'analyse utilisé par ZeMosaic pour évaluer la qualité de chaque image individuelle avant l'empilement. Son objectif est simple : conserver le signal, rejeter le bruit et les images inutilisables, sans nécessiter de fichiers de calibration ou de réglages manuels.

Pour chaque image, ZeAnalyser calcule un ensemble de métriques objectives de qualité, telles que :

*   Détection et comptage d'étoiles (robuste au bruit et aux gradients)
*   Statistiques sur la forme des étoiles (indicateurs d'excentricité / d'élongation)
*   Métrique globale de netteté / structure
*   Analyse du bruit et du comportement du fond de ciel
*   Estimations optionnelles liées au rapport Signal/Bruit (SNR)

Ces métriques sont combinées pour :

*   Rejeter les images inexploitables (ex: erreurs de suivi, nuages, flou important)
*   Pondérer ou filtrer les images de manière cohérente sur de grands ensembles de données
*   Assurer une qualité de données homogène avant l'empilement

ZeAnalyser fonctionne de manière entièrement automatique et est conçu pour traiter efficacement des dizaines de milliers d'images, le rendant adapté aux longues sessions d'imagerie multi-nuits avec un Seestar ou un équipement traditionnel.

### Mode Grille : Stratégie de Traitement « Mosaïque d'Abord »

Le Mode Grille (Grid Mode) introduit une approche centrée sur la mosaïque, spécialement conçue pour les grands champs, les vastes couvertures célestes et les ensembles de données avec un chevauchement variable.

Au lieu d'empiler toutes les images en une seule trame de référence, le champ de vision est divisé en une grille régulière de tuiles. Chaque tuile est traitée indépendamment avant d'être réassemblée dans la mosaïque finale.

Étapes clés du Mode Grille :

1.  **Partitionnement Spatial**
    La couverture céleste est divisée en tuiles de grille qui se chevauchent. Chaque image ne contribue qu'aux tuiles qu'elle recouvre réellement.

2.  **Analyse Locale avec ZeAnalyser**
    ZeAnalyser est appliqué à chaque tuile individuellement, et non globalement. Cela permet des décisions de qualité locales : une image peut être rejetée pour une tuile mais acceptée pour une autre. Les problèmes locaux de seeing, de suivi ou de distorsion sont ainsi gérés naturellement.

3.  **Empilement Indépendant des Tuiles**
    Chaque tuile est empilée en utilisant uniquement les images validées pour cette tuile spécifique. Cela améliore la netteté locale et la cohérence du signal, tout en réduisant les artefacts de bord et les couvertures inégales.

4.  **Rognage et Normalisation des Tuiles**
    Les bordures invalides ou à faible couverture sont automatiquement rognées. L'intensité et le fond de ciel de chaque tuile sont normalisés avant la reprojection.

5.  **Assemblage Final de la Mosaïque**
    Toutes les tuiles sont reprojetées en utilisant leurs informations WCS. La mosaïque finale est assemblée avec une géométrie et une colorimétrie cohérentes.

### Pourquoi le Mode Grille est Important

Le Mode Grille résout plusieurs problèmes classiques de l'empilement de grands champs et de mosaïques :

*   Chevauchement inégal des images
*   Distorsions de suivi locales
*   Rotation de champ et dégradation des bords
*   Variations de qualité sur de grands ensembles de données

En combinant l'analyse par image de ZeAnalyser avec l'empilement spatialisé du Mode Grille, ZeMosaic obtient :

*   Une meilleure netteté locale
*   Une réduction des artefacts de « ghosting » (images fantômes) et de duplication
*   Des mosaïques plus stables sur des ensembles de données volumineux ou imparfaits
*   Une stratégie de traitement qui reste robuste à mesure que la taille de l'ensemble de données augmente

### Philosophie de Conception

ZeAnalyser et le Mode Grille sont intentionnellement conçus pour être :

*   **Automatiques** – Réglages manuels minimaux requis
*   **Déterministes** – Mêmes entrées, mêmes résultats
*   **Évolutifs** (*Scalable*) – De quelques centaines à des dizaines de milliers d'images
*   **Indépendants de l'instrument** – Optimisés pour le Seestar mais non limités à celui-ci

Ensemble, ils forment la colonne vertébrale du pipeline d'empilement moderne de ZeMosaic.

---

Quality Crop (edge artifact removal)

ZeMosaic includes an optional Quality Crop step designed to automatically remove low-quality borders that can appear after alignment/reprojection (dark rims, stretched edges, noisy bands, stacking seams, etc.). The idea is to analyze the image edges and crop away regions that statistically look “worse” than the interior.

Parameters

Enable quality crop (default: OFF)
Turns the whole feature on/off.
When OFF, ZeMosaic keeps the full tile image and does not run any edge quality analysis.

Band width (px) (default: 32)
Defines the thickness (in pixels) of the edge bands inspected for quality.
ZeMosaic analyzes borders within this width (top/bottom/left/right) to decide where quality drops.

K-sigma (default: 2.0)
Controls the sigma threshold used to decide whether a pixel/run is considered “bad” compared to expected background statistics.
Lower values = more aggressive cropping (more pixels flagged as outliers).
Higher values = more conservative cropping.

Minimum run (default: 2)
Sets the minimum length (in pixels) of a continuous bad segment before it is considered meaningful.
This helps ignore isolated bad pixels and prevents overreacting to tiny defects.

Margin (px) (default: 8)
Adds a safety margin (in pixels) when cropping.
Once a low-quality edge region is detected, ZeMosaic crops slightly deeper by this amount to avoid leaving a thin residual artifact line.

Practical guidance

If you still see obvious borders/seams, try increasing Band width slightly (e.g. 48–64) and/or lowering K-sigma (e.g. 1.5–1.8).

If you feel ZeMosaic crops too much, increase K-sigma or increase Minimum run.

---

## 📷 Requirements

### Mandatory:

- Python ≥ 3.9  
- [ASTAP](https://www.hnsky.org/astap.htm) installed with G17/H17 star catalogs

### Recommended Python packages:

```bash
pip install numpy astropy reproject opencv-python photutils scipy psutil
```
The worker originally required `DirectoryStore`, removed in `zarr>=3`.
ZeMosaic now falls back to `LocalStore`, and skips the old
`LRUStoreCache` wrapper when running against Zarr 3.
Both Zarr 2.x and 3.x are supported (tested on Python 3.11+).

🧠 Inspired by PixInsight
ZeMosaic draws strong inspiration from the image integration strategies of PixInsight, developed by Juan Conejero at Pleiades Astrophoto.

Specifically, the implementations of:

Noise Variance Weighting (1/σ²)

Kappa-Sigma and Winsorized Rejection

Radial feather blending

...are adapted from methods described in:

📖 PixInsight 1.6.1 – New ImageIntegration Features
Juan Conejero, 2010
Forum thread

🙏 We gratefully acknowledge Juan Conejero's contributions to astronomical image processing.

🛠 Dependencies
ZeMosaic uses several powerful open-source Python libraries:

numpy and scipy for numerical processing

astropy for FITS I/O and WCS handling

reproject for celestial reprojection

opencv-python for debayering

photutils for source detection and background estimation

psutil for memory monitoring

PySide6 (Qt) for the official graphical user interface

> **Note (Linux/macOS):** The official ZeMosaic frontend is Qt-only and requires `PySide6`.

📦 Installation & Usage
1. 🔧 Install Python dependencies
If you have a local clone of the repository, make sure you're in the project folder, then run:

pip install -r requirements.txt
💡 Requirements are mostly flexible. ZeMosaic now supports both zarr 2.x and
3.x, automatically falling back to `LocalStore` when `DirectoryStore` is
unavailable. The project is tested with Python 3.11+.

If you prefer to install manually:

pip install numpy astropy reproject opencv-python photutils scipy psutil

2. 🚀 Launch ZeMosaic
Once the dependencies are installed:
python run_zemosaic.py

The GUI will open. From there:

Select your input folder (with raw FITS images)

Choose your output folder

Configure ASTAP paths and options

Adjust stacking & mosaic settings

Click "Start Hierarchical Mosaic"

📁 Requirements Summary
✅ Python 3.9 or newer

✅ ASTAP installed + star catalogs (D50 or H18)

✅ FITS images (ideally calibrated, debayered or raw from Seestar)
✅ Python multiprocessing enabled (ProcessPoolExecutor is used for assembly)

✅ `assembly_process_workers` can be set in `zemosaic_config.json` to control
   how many processes handle final mosaic assembly (0 = auto, applies to both methods)


🖥️ How to Run
After installing Python and dependencies:

python run_zemosaic.py
Use the GUI to:

Choose your input/output folders

Configure ASTAP paths

Select stacking and assembly options

Click Start Hierarchical Mosaic

### Official Qt interface

ZeMosaic now uses a PySide6/Qt interface as the only official frontend.

To run the official frontend:

1. Install the optional dependency:

   ```bash
   pip install PySide6
   ```

2. Launch ZeMosaic with either of the following options:

   ```bash
   python run_zemosaic.py
   ```

If PySide6 is unavailable, ZeMosaic reports a clear startup error. No Tk fallback is used on the official path.

#### Automatic ZeAnalyser / Beforehand Tool Discovery (Qt GUI)
To enable the `Analyse` button, install a compatible analysis tool in the parent directory of `zemosaic/`. ZeMosaic auto-detects them at startup.

**Discovery Rules & UI Behavior:**
1.  It first checks for a `zeanalyser/` directory. If found, the **Analyse** button is enabled, using **ZeAnalyser** as the backend.
2.  If not found, it looks for `seestar/beforehand/`. If this directory exists, the button is enabled, using the legacy **Beforehand** backend.
3.  If neither is found, the `Analyse` button is not displayed, keeping the UI clean.

The button's tooltip will always indicate which backend is active. If both are installed, ZeAnalyser takes priority.

#### Découverte automatique des outils ZeAnalyser / Beforehand (IUG Qt)
Pour activer le bouton `Analyser`, installez un outil d'analyse compatible dans le répertoire parent de `zemosaic/`. ZeMosaic les détecte automatiquement au démarrage.

**Règles de découverte et comportement de l'interface :**
1.  Le logiciel vérifie d'abord la présence d'un répertoire `zeanalyser/`. S'il est trouvé, le bouton **Analyser** est activé et utilise le moteur **ZeAnalyser**.
2.  Sinon, il recherche `seestar/beforehand/`. Si ce répertoire existe, le bouton est activé et utilise le moteur historique **Beforehand**.
3.  Si aucun des deux n'est trouvé, le bouton `Analyser` n'est pas affiché, gardant l'interface épurée.

L'info-bulle du bouton indiquera toujours quel moteur est actif. Si les deux sont installés, ZeAnalyser est prioritaire.

### Force Seestar workflow checkbox

The Main tab of both GUIs exposes two related toggles for Seestar datasets:

- **Auto-detect Seestar frames** stays on by default and inspects the FITS `INSTRUME`
  header (or any instrument hint provided by the filter UI). When the label
  contains “Seestar/S50/S30”, ZeMosaic enters the Seestar/Mosaic-First workflow
  automatically.
- **Force Seestar workflow** is a manual override. When it is checked, the filter
  dialog and the worker assume the Mosaic-First path regardless of what the FITS
  headers say. The filter always prepares/reuses the global WCS descriptor,
  exports the `global_wcs_meta`/FITS/JSON paths, and sets the workflow mode to
  `seestar`, so the worker skips the classic per-master-tile stack even if the
  dataset mixes instruments or carries incomplete metadata.

Enable this override whenever the automatic detection fails (e.g. FITS files
stripped of `INSTRUME`), or when you deliberately want to run non-Seestar data
through the Seestar-optimized Mosaic-First pipeline. Disabling it reverts to the
classic workflow unless the headers clearly advertise Seestar frames.

### Global WCS auto-cropping (English)

ZeMosaic can optionally trim the Mosaic-first canvas so the exported FITS only contains sky regions with real coverage.  
Enable this by setting `global_wcs_autocrop_enabled` to `true` inside `zemosaic_config.py` (see the `DEFAULT_CONFIG` block near the other `global_wcs_*` keys) or in your personal `zemosaic_config.json`.  
Once enabled, the worker inspects the Phase 5 coverage map, removes empty borders, and shifts the global WCS `CRPIX`/`NAXIS` values automatically so downstream tools see the reduced frame.  
Use `global_wcs_autocrop_margin_px` (same section) to keep a safety border in pixels—default is 64 px.

### Recadrage automatique du WCS global (Français)

ZeMosaic peut rogner automatiquement la mosaïque finale (mode Mosaic-first) pour ne conserver que la zone réellement couverte par les images.  
Activez `global_wcs_autocrop_enabled` en le passant à `true` dans `zemosaic_config.py` (section `DEFAULT_CONFIG`, proche des autres clés `global_wcs_*`) ou dans votre fichier `zemosaic_config.json`.  
Une fois l’option activée, l’ouvrier analyse la carte de couverture de la Phase 5, supprime les bordures vides et ajuste `CRPIX` / `NAXIS` du WCS global afin que les outils en aval utilisent la toile réduite.  
La marge de sécurité se règle avec `global_wcs_autocrop_margin_px` (en pixels, 64 px par défaut).

### GPU helper for Phase 4

Setting `use_gpu_phase5` to `true` (via the worker configuration or overrides) now enables the CUDA helper
for the entire Mosaic-First reprojection+coadd stage. If a supported GPU is available, ZeMosaic will run
mean, median, winsorized, and kappa-sigma global stacking directly on the GPU and automatically fall back
to the CPU only when the helper is unavailable. For integration testing you can optionally set
`gpu_helper_verify_tolerance` in the global plan to log the max per-pixel delta between the GPU and CPU
reference implementation.

### macOS quickstart

1. Install Python 3.11+ from [python.org](https://www.python.org/downloads/).
2. Download the macOS ASTAP `.dmg`, drag `ASTAP.app` into `/Applications`, then install the star catalogs
   (D50/H17) under `/Library/Application Support/ASTAP` or `~/Library/Application Support/ASTAP`.
3. Inside the project folder run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   python3 run_zemosaic.py
   ```

4. ZeMosaic now auto-detects `/Applications/ASTAP.app/Contents/MacOS/astap` and the associated catalog
   directories, but you can override them at any time:

   ```bash
   export ASTAP_EXE=/Applications/ASTAP.app/Contents/MacOS/astap
   export ASTAP_DATA_DIR="/Library/Application Support/ASTAP"
   ```

### Linux quickstart

1. Install Python, pip, and build tooling via your package manager. Example for Debian/Ubuntu:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip python3-dev build-essential
   ```

   Fedora/RHEL users can run `sudo dnf install python3 python3-venv python3-pip`.

2. Install the ASTAP Linux package (from https://www.hnsky.org/astap.htm) and the desired star catalogs.
   The default installer places binaries under `/usr/bin/astap` and data under `/opt/astap`.

3. Create a virtual environment and launch ZeMosaic:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   python3 run_zemosaic.py
   ```

4. ASTAP locations are discovered automatically on `/usr/bin`, `/usr/local/bin`, and `/opt/astap`, but
   you can pin custom installs:

   ```bash
   export ASTAP_EXE=/usr/local/bin/astap
   export ASTAP_DATA_DIR=/opt/astap
   ```

### ASTAP path detection & overrides

`zemosaic_config.py` now validates the stored executable/data paths on startup and, if needed, scans the
common installation directories for Windows (`Program Files`), macOS (`/Applications/ASTAP*.app`), and
Linux (`/usr/bin`, `/opt/astap`, etc.). The following environment variables are respected ahead of the
auto-detected paths:

- `ASTAP_EXE`, `ASTAP_BIN`, or `ASTAP_PATH` for the binary
- `ASTAP_DATA_DIR`, `ASTAP_STAR_DB`, or `ASTAP_DATABASE` for the star catalogs

You can inspect what was found by running:

```bash
python - <<'PY'
from zemosaic_config import detect_astap_installation
print(detect_astap_installation())
PY
```

### ASTAP concurrency guard

Running more than one ASTAP instance in parallel can trigger the `Access violation` pop-up shown by the
native solver. To keep ZeMosaic/ZeSeestarStacker runs unattended, every ASTAP call now cooperates through
an inter-process lock:

- The configured `astap_max_instances` value (GUI + `zemosaic_config.json`) becomes a global cap shared by
  every ZeMosaic utility launched on the same machine. Extra workers simply wait for a free slot.
- Progress logs mention when we are waiting for another process and resume automatically once the lock is
  released, eliminating the Windows dialog.
- Advanced users who prefer the legacy behaviour can set `ZEMOSAIC_ASTAP_DISABLE_IPC_LOCK=1` before
  launching the tools. Override the lock directory with `ZEMOSAIC_ASTAP_LOCK_DIR=/path/to/tmp` if the
  default `%TEMP%/zemosaic_astap_slots` is not suitable (portable drives, RAM disks, etc.).
- SetErrorMode is enabled on Windows so ASTAP cannot raise modal crash boxes, and additional guards keep the
  solver healthy under bursty workloads: per-image file locks, a configurable rate limiter
  (`ZEMOSAIC_ASTAP_RATE_SEC` / `ZEMOSAIC_ASTAP_RATE_BURST`), and automatic retries with back-off
  (`ZEMOSAIC_ASTAP_RETRIES`, `ZEMOSAIC_ASTAP_BACKOFF_SEC`).

🔧 Build & Compilation (Windows)

🇬🇧 Instructions (English)
1. Install Python 3.13 (x64) from python.org.
2. From the project root:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install --upgrade pyinstaller
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   The default output is `dist/ZeMosaic/ZeMosaic.exe` (onedir).
   This path keeps the current GPU-enabled setup because `requirements.txt` currently includes `cupy-cuda12x`.

   If you want a CPU-only package with a much smaller payload:

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

   Or manually:

   ```powershell
   python -m pip install -r requirements_no_gpu.txt
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

3. Optional onefile build:

   ```powershell
   set ZEMOSAIC_BUILD_MODE=onefile
   set ZEMOSAIC_RUNTIME_TMPDIR=C:\Temp
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   Output becomes `dist/ZeMosaic.exe` (onefile).

4. Optional debug build (console ON) for diagnostics:

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=1
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   Clear the env var for release builds (console OFF):

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=
   ```

5. Helper script (uses the same spec):

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

Notes:
- Resources `locales/`, `icon/`, and `gif/` are bundled via `ZeMosaic.spec`.
- `PySide6` is required for the official packaged frontend (Qt-only).
- `matplotlib` is optional: if missing, the Qt filter preview is disabled.
- `cupy-cuda12x` is optional: if missing (or if NVIDIA drivers are missing/incompatible, or if required CUDA DLLs are not present on the target machine), ZeMosaic falls back to CPU. On Windows this often means having CUDA Toolkit (or at least its runtime DLLs) available via `%CUDA_PATH%\\bin`/`PATH`.
- `requirements.txt` keeps the current working GPU setup (`cupy-cuda12x`). `requirements_no_gpu.txt` is available for CPU-only builds when you want a smaller artifact.
- Prefer onedir for reliability; onefile can hit Windows path length issues (example: Shapely `WinError 206`), mitigated by using a short `ZEMOSAIC_RUNTIME_TMPDIR` like `C:\Temp` and/or enabling long paths in Windows.
- `zemosaic_installer.iss` does not choose the CUDA package. It simply packages `dist\\ZeMosaic\\*`. If you built with `requirements.txt`, the installer packages the GPU-enabled build. If you built with `requirements_no_gpu.txt`, it packages the CPU-only build.
- The relevant `.iss` lines are the `[Files]` entry pointing to `dist\\ZeMosaic\\*`, plus `[Icons]` / `[Run]` pointing to `{app}\\ZeMosaic.exe`. You normally do not need to change them for a different CUDA version.

Mini smoke-test (packaged build):
- Launch `dist/ZeMosaic/ZeMosaic.exe` (or `dist/ZeMosaic.exe` in onefile mode).
- Confirm the UI starts without crash and icons load (window icon / toolbar icons).
- Switch language (if available) to confirm `locales/` loads.
- Pick an input folder and start a small run to confirm the worker starts and writes output.

🇫🇷 Instructions (Francais)
1. Installez Python 3.13 (x64) depuis python.org.
2. Depuis la racine du projet :

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install --upgrade pyinstaller
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   La sortie par défaut est `dist/ZeMosaic/ZeMosaic.exe` (onedir).
   Ce chemin conserve le build GPU actuel car `requirements.txt` contient actuellement `cupy-cuda12x`.

   Si vous voulez un paquet CPU-only beaucoup plus léger :

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

   Ou en manuel :

   ```powershell
   python -m pip install -r requirements_no_gpu.txt
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

3. Build onefile optionnel :

   ```powershell
   set ZEMOSAIC_BUILD_MODE=onefile
   set ZEMOSAIC_RUNTIME_TMPDIR=C:\Temp
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   La sortie devient `dist/ZeMosaic.exe` (onefile).

4. Build debug optionnel (console ON) pour diagnostiquer :

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=1
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   Remettez l'env var a vide pour la release (console OFF) :

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=
   ```

5. Script helper (meme spec) :

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

Notes :
- Les ressources `locales/`, `icon/`, et `gif/` sont embarquees via `ZeMosaic.spec`.
- `PySide6` est requis pour l'interface officielle packagée (Qt-only).
- `matplotlib` est optionnel : s'il manque, l'aperçu (Qt filter preview) est désactivé.
- `cupy-cuda12x` est optionnel : s'il manque (ou si les drivers NVIDIA sont absents/incompatibles, ou si les DLL CUDA nécessaires ne sont pas présentes sur la machine cible), ZeMosaic retombe en CPU. Sous Windows cela implique souvent CUDA Toolkit (ou au minimum ses DLL runtime) accessibles via `%CUDA_PATH%\\bin`/`PATH`.
- `requirements.txt` conserve le setup GPU actuel qui fonctionne (`cupy-cuda12x`). `requirements_no_gpu.txt` est disponible pour produire des builds CPU-only plus petits.
- Préférez onedir pour la fiabilité ; onefile peut déclencher des soucis de longueur de chemin Windows (ex: Shapely `WinError 206`), atténués via un `ZEMOSAIC_RUNTIME_TMPDIR` court comme `C:\Temp` et/ou l'activation des long paths dans Windows.
- `zemosaic_installer.iss` ne choisit pas le paquet CUDA. Il empaquette simplement `dist\\ZeMosaic\\*`. Si vous avez buildé avec `requirements.txt`, l'installateur emballera la version GPU. Si vous avez buildé avec `requirements_no_gpu.txt`, il emballera la version CPU-only.
- Les lignes `.iss` pertinentes sont l'entrée `[Files]` qui pointe sur `dist\\ZeMosaic\\*`, ainsi que `[Icons]` / `[Run]` qui pointent sur `{app}\\ZeMosaic.exe`. En pratique vous n'avez pas à les changer pour une autre version de CUDA.

Mini smoke-test (build packagé) :
- Lancez `dist/ZeMosaic/ZeMosaic.exe` (ou `dist/ZeMosaic.exe` en onefile).
- Vérifiez que l'UI démarre sans crash et que les icônes se chargent (icône fenêtre / toolbar).
- Changez la langue (si dispo) pour confirmer le chargement de `locales/`.
- Choisissez un dossier d'entrée et lancez un petit run pour valider le démarrage du worker et l'écriture de sortie.

🛠️ Build & Compilation (macOS/Linux)
🇬🇧 Instructions (English)

1. Ensure you have a virtual environment (`python3 -m venv .venv`), then activate it: `source .venv/bin/activate`.
2. Install the runtime requirements once: `python3 -m pip install -r requirements.txt`.
3. Make the helper executable and launch it:

   ```bash
   chmod +x compile/build_zemosaic_posix.sh
   ./compile/build_zemosaic_posix.sh
   ```

   The script installs/updates PyInstaller inside `.venv` and produces `dist/zemosaic`.
   By default it uses `requirements.txt`, which preserves the current GPU-enabled setup when `cupy-cuda12x` is available for the platform.

   For a CPU-only build:

   ```bash
   ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt ./compile/build_zemosaic_posix.sh
   ```

🇫🇷 Instructions (Français)

1. Créez/activez votre environnement virtuel (`python3 -m venv .venv` puis `source .venv/bin/activate`).
2. Installez les dépendances (`python3 -m pip install -r requirements.txt`).
3. Rendez le script exécutable puis lancez-le :

   ```bash
   chmod +x compile/build_zemosaic_posix.sh
   ./compile/build_zemosaic_posix.sh
   ```

   L'exécutable macOS/Linux sera généré dans `dist/zemosaic`.
   Par défaut le script utilise `requirements.txt`, ce qui conserve le setup GPU actuel quand `cupy-cuda12x` existe pour la plateforme.

   Pour un build CPU-only :

   ```bash
   ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt ./compile/build_zemosaic_posix.sh
   ```

### Windows GitHub release

1. Build the Windows onedir package:

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

2. Optionally build a CPU-only variant if the GPU bundle is too large:

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

3. Create a zip from `dist\ZeMosaic`.
4. Publish that zip as a GitHub Release asset rather than committing `dist/` to the repository.

Notes:
- GitHub blocks regular Git files above 100 MiB, so `dist/` should not be committed.
- GitHub Release assets are the right place for Windows binaries.
- The installer script `zemosaic_installer.iss` packages `dist\ZeMosaic\*` exactly as built.

### Release Windows sur GitHub

1. Générez le build Windows onedir :

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

2. En option, générez une variante CPU-only si le bundle GPU est trop gros :

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

3. Créez une archive zip à partir de `dist\ZeMosaic`.
4. Publiez ce zip dans une GitHub Release au lieu de versionner `dist/` dans le dépôt.

Notes :
- GitHub bloque les fichiers Git classiques au-delà de 100 MiB, donc `dist/` ne doit pas être commit.
- Les GitHub Releases sont le bon endroit pour publier les binaires Windows.
- `zemosaic_installer.iss` empaquette exactement `dist\ZeMosaic\*` tel qu'il a été construit.

### Memory-mapped coadd (enabled by default)

```jsonc
{
  "final_assembly_method": "reproject_coadd",
  "coadd_use_memmap": true,
  "coadd_memmap_dir": "D:/ZeMosaic_memmap",
  "coadd_cleanup_memmap": true
  "assembly_process_workers": 0
}
```
`assembly_process_workers` also defines how many workers the incremental method uses.
A final mosaic of 20 000 × 20 000 px in RGB needs ≈ 4.8 GB
(4 × H × W × float32). Make sure the target disk/SSD has enough space.
Hot pixel masks detected during preprocessing are also written to the temporary
cache directory to further reduce memory usage.

### Memory-saving parameters

The configuration file exposes a few options to control memory consumption:

- `auto_limit_frames_per_master_tile` – automatically split raw stacks based on available RAM.
- `max_raw_per_master_tile` – manual cap on raw frames stacked per master tile (0 disables).
- `winsor_worker_limit` – maximum parallel workers during the Winsorized rejection step.
- `winsor_max_frames_per_pass` – maximum frames processed at once during Winsorized rejection (0 keeps previous behaviour).
- `winsor_auto_fallback_on_memory_error` – proactively halve the batch size, then fall back to disk streaming when NumPy cannot allocate RAM.
- `winsor_min_frames_per_pass` – lower bound for the streaming fallback (default 4).
- `winsor_memmap_fallback` – `auto` (default) activates disk-backed memmap only when needed, `always` forces it, `never` keeps pure RAM processing.
- `winsor_split_strategy` – choose `sequential` (default) or `roundrobin` chunk scheduling to balance memory pressure across large stacks.

Machines with < 16 GB RAM benefit from setting `winsor_max_frames_per_pass` to 32–48 and keeping the automatic fallback enabled. En dessous de 16 Gio de RAM, conservez l’option de secours automatique activée et limitez les passes à 32–48 images ; le mode `memmap` sera déclenché si nécessaire pour éviter les erreurs `ArrayMemoryError`.

6 ▸ Quick CLI example
```bash
run_zemosaic.py \
  --final_assembly_method reproject_coadd \
  --coadd_memmap_dir D:/ZeMosaic_memmap \
  --coadd_cleanup_memmap \
  --assembly_process_workers 4
```




🧪 Troubleshooting
If astrometric solving fails:

Check ASTAP path and data catalogs

Ensure your images contain enough stars

Use a Search Radius of ~3.0°

Watch zemosaic_worker.log for full tracebacks

📎 License
ZeMosaic is licensed under GPLv3 — feel free to use, adapt, and contribute.

🤝 Contributions
Feature requests, bug reports, and pull requests are welcome!
Please include log files and test data if possible when reporting issues.

🌠 Happy mosaicking!


## Advanced color option: Final Mosaic RGB Equalization

ZeMosaic includes an optional final color balancing step on the assembled mosaic:

- Config key: `final_mosaic_rgb_equalize_enabled`
- Recommended default: `false` (conservative)
- Activation: GUI toggle or direct JSON edit

When enabled, this step can help recover global RGB neutrality on some difficult/marginal datasets.
On already well-balanced datasets, it may bring little or no visible benefit.

Related keys:
- `final_mosaic_rgb_equalize_clip_enabled`
- `final_mosaic_rgb_equalize_gain_clip`
- `existing_master_tiles_final_rgb_equalize_gain_clip`
- `sds_enable_final_rgb_equalize`
- `sds_final_rgb_equalize_gain_clip`

Practical note:
Keep it OFF by default for stable production behavior, and enable it as a targeted tool when color drift appears on specific datasets.

---


## Imported from RELEASE_NOTES.md

# ZeMosaic 4.4.1

## Qt-only / Tk retirement update

- Qt (PySide6) is now the only official frontend runtime path.
- Official startup no longer falls back to Tk (`--tk-gui` unsupported on official path).
- `zemosaic_config` migration normalizes legacy `preferred_gui_backend=tk` to `qt` and neutralizes obsolete backend-selection state.
- `lecropper` remains an annex/standalone legacy tool and is decoupled from official runtime/headless validated paths.

## Compatibility / unsupported legacy

- Legacy Tk frontend is not an official runtime path.
- Full repo-wide removal of all Tk annex tools is out-of-scope for this release line.


## Imported from fix_regression.md

# fix_regression.md

## Mission
Rétablir le fonctionnement des **3 voies exclusives** après le refactor (héritage `agent.md` / `followup.md` / `memory.md`) **sans casser d'autres chemins**:
- voie classique
- voie grid_mode
- voie SDS

⚠️ Règle de cadrage: ces modes sont **mutuellement exclusifs**. Ils ne doivent pas coexister dans un même run.

Contrainte de mission:
- correctifs chirurgicaux
- preuve avant/après
- non-régression explicite sur les chemins non ciblés

---

## Contexte initial (2026-03-14)
Erreur remontée sur dernier run SDS:
- `NameError: name 'existing_master_tiles_results' is not defined`
- trace observée dans `run_hierarchical_mosaic` (`zemosaic_worker.py:26488`)

Observation clé:
- Le run va jusqu'à la fin de la Phase 2, puis crash au passage vers la phase suivante.

---

## Clarification architecture modes (ajout 2026-03-14 09:29)
Les chemins d'exécution à considérer sont:
1. **Mode classique**
2. **Mode grid_mode**
3. **Mode SDS**

Ces 3 modes sont distincts et exclusifs:
- pas de scénario “SDS + grid_mode”
- pas de mélange de logique entre branches
- la validation doit se faire **mode par mode**, pas en combinatoire croisée SDS/grid.

---

## Journal d'itérations

### Iteration 1 — Audit initial et cadrage (2026-03-14 09:25)
**Fait:**
1. Lecture des fichiers de continuité mission:
   - `agent.md`
   - `followup.md`
   - `memory.md`
2. Lecture du log du run SDS en échec (`zemosaic_worker.log`).
3. Inspection statique de `zemosaic_worker.py` autour de la stacktrace.
4. Vérification des occurrences de `existing_master_tiles_results` dans le fichier.

**Constats techniques:**
- Dans `run_hierarchical_mosaic` (nouveau chemin), `existing_master_tiles_results` est **utilisée** mais non définie localement avant usage.
- Dans `run_hierarchical_mosaic_classic_legacy`, la variable est bien initialisée en amont.
- Symptomatique d'une régression de refactor/copier-coller entre chemins legacy et nouveau pipeline.

**Impact probable:**
- crash précoce quand le code atteint l'initialisation `master_tiles_results_list = list(existing_master_tiles_results)`.
- risque de toucher plusieurs voies d'exécution si ce bloc est partagé.

**Ce qui reste à faire:**
1. Localiser le meilleur point d'initialisation/garde de cette variable dans `run_hierarchical_mosaic`.
2. Corriger minimalement la portée/initialisation sans modifier la logique scientifique.
3. Vérifier les 3 voies exclusives (classique / grid / SDS) pour éviter une casse collatérale.
4. Ajouter un test de non-régression ciblant ce scénario (`NameError` impossible).
5. Documenter preuve avant/après ici et dans `memory.md`.

---

## Plan d'action (amendé)
1. **Repro contrôlée**
   - rejouer le scénario SDS qui crash
   - confirmer le point exact de rupture
2. **Patch minimal de portée**
   - introduire une initialisation sûre de `existing_master_tiles_results` dans `run_hierarchical_mosaic`
   - éviter toute modification de comportement hors besoin
3. **Validation par mode (exclusifs)**
   - Run en voie classique
   - Run en voie grid_mode
   - Run en voie SDS
4. **Tests de non-régression**
   - ajout test dédié contre `NameError` sur ce bloc
   - exécution des tests ciblés puis subset pertinent
5. **Clôture de correction**
   - résumé du diff
   - risques résiduels
   - checklist “fait / restant” à jour

---

## Checklist opérationnelle
- [x] Audit des logs du run SDS en échec
- [x] Lecture `agent.md` / `followup.md` / `memory.md`
- [x] Identification du point de rupture (`existing_master_tiles_results`)
- [x] Cadrage explicite: 3 modes exclusifs (classique / grid / SDS)
- [x] Correctif code appliqué
- [x] Repro validée post-fix (plus de NameError)
- [ ] Validation multi-mode non-régression (classique / grid / SDS)
- [x] Test automatisé ajouté
- [ ] Clôture et preuve finale documentées

### Iteration 2 — Correctif SDS ciblé `NameError` + garde de non-régression (2026-03-14 09:33)
**Fait:**
1. Correctif appliqué dans `run_hierarchical_mosaic` (chemin SDS/grid):
   - remplacement de l'initialisation fautive
   - `master_tiles_results_list` est maintenant initialisée explicitement à `[]`.
2. Vérification de l'autre occurrence dans le chemin legacy:
   - conservée inchangée (`list(existing_master_tiles_results)`) pour ne pas altérer la voie classique legacy.
3. Ajout d'un test source-contract de non-régression:
   - `test_run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol`
   - garantit que `run_hierarchical_mosaic` ne dépend plus du symbole non défini.

**Validation effectuée:**
- `python3 -m py_compile zemosaic_worker.py`
- `python3 -m py_compile tests/test_phase3_adaptive_invariants.py`
- `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol"` -> `1 passed`

**Note de traçabilité:**
- une première édition a touché la mauvaise occurrence (legacy), immédiatement corrigée dans la même itération pour isoler strictement le patch sur le chemin SDS/grid.

**Ce qui reste à faire:**
1. Rejouer un run SDS réel via GUI (toggle SDS actif) et confirmer disparition du `NameError`.
2. Vérifier que la voie classique et la voie grid_mode ne régressent pas (smoke rapide par mode).
3. Documenter preuve finale avant clôture.

### Iteration 3 — Fix SDS broadcast mismatch en Phase 5 polish (2026-03-14 10:11)
**Contexte erreur utilisateur:**
- `ValueError: operands could not be broadcast together with shapes (1099,32) (3330,3748) ()`
- traceback dans `_apply_final_mosaic_quality_pipeline` pendant `_finalize_sds_global_mosaic`.

**Diagnostic:**
- En SDS global polish, la quality-crop (lecropper) peut recadrer `final_mosaic_data` (ex: `1099x32`).
- La coverage map reste sur la géométrie globale SDS (`3330x3748`).
- Puis `np.where(keep_mask > 0, final_mosaic_coverage, 0.0)` échoue par mismatch de dimensions.

**Correctif appliqué (ciblé SDS):**
- Dans `_finalize_sds_global_mosaic`, création d'une copie locale `sds_pipeline_cfg`.
- Désactivation explicite de `quality_crop_enabled` uniquement pour le polish SDS global.
- Ajout d'un log `phase5_sds_quality_crop_disabled` pour traçabilité.

**Pourquoi ce choix:**
- Le mode SDS s'appuie sur une géométrie globale (descriptor WCS) qui doit rester cohérente entre mosaic/coverage/alpha.
- Le quality-crop est géométrie-changing; en SDS global il introduit une incohérence de shape.
- Correctif chirurgical: on préserve alt-az cleanup et le reste du pipeline, on neutralise seulement la partie recadrage en SDS polish.

**Validation effectuée:**
- `python3 -m py_compile zemosaic_worker.py tests/test_phase3_adaptive_invariants.py`
- `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish"` -> `2 passed`

**Reste à faire:**
1. Rejouer run SDS via GUI (toggle SDS actif) et confirmer disparition de l'erreur broadcast.
2. Vérifier sortie finale SDS générée correctement.
3. Puis smoke rapide classique/grid.

### Iteration 4 — SDS VRAM hardening sur global coadd helper GPU (2026-03-14 10:38)
**Contexte log:**
- répétitions de warnings en SDS:
  - `GPU helper error: Out of memory allocating ...`
  - fallback immédiat CPU via `helper_failed`.

**Diagnostic:**
- Ce n'est pas une fuite mémoire prouvée dans le log; c'est un OOM runtime sur `gpu_reproject` avec hints chunk/rows trop agressifs pour certaines passes/canaux.
- Le chemin helper SDS n'avait pas la même logique de retry+tightening adaptatif que d'autres chemins GPU (voie classique/P3).

**Correctif appliqué (priorité VRAM):**
- Dans `_attempt_gpu_helper_route`:
  1. Ajout d'un retry GPU local par canal (`max_gpu_helper_retries = 3`).
  2. Sur OOM détecté (`_is_gpu_oom_error`):
     - purge des pools CuPy (`free_cupy_memory_pools`) si dispo,
     - réduction adaptative des hints GPU:
       - `rows_per_chunk` divisé par 2 (plancher 32),
       - `max_chunk_bytes` divisé par 2 (plancher 32MB),
     - réessai GPU avant fallback CPU global.
  3. Persistance intra-run des hints réduits via `plan_rows_gpu_hint` / `plan_chunk_gpu_hint` (pour les canaux suivants du même lot).
  4. Ajout de log de traçabilité: `global_coadd_gpu_oom_retry`.

**Validation effectuée:**
- `python3 -m py_compile zemosaic_worker.py`
- test source-contract ajouté:
  - `test_sds_global_gpu_helper_has_oom_retry_with_chunk_tightening`
- `pytest` ciblé: `3 passed`.

**Couleurs (constat rapide post-log):**
- Les stats RGB P6/P7 du log restent tri-canaux (pas mono explicite).
- Analyse rapide du PNG/FITS de sortie: image bien RGB, canaux proches mais non identiques.
- Suspicion actuelle: rendu/perception (stretch/équilibrage), pas conversion N&B stricte.

**Reste à faire:**
1. Re-run SDS et vérifier présence/efficacité de `global_coadd_gpu_oom_retry`.
2. Vérifier diminution des bascules CPU sur OOM.
3. Si rendu toujours "gris", traiter la chaîne d'affichage (preview stretch / RGB equalize) sans impacter la data FITS scientifique.

### Iteration 5 — Validation SDS post-fix + audit préliminaire ZeGrid (2026-03-14 11:04)
**Lecture log demandée (`zemosaic_worker.log`):**
- Le run SDS observé va jusqu'au bout:
  - `run_success_mosaic_saved`
  - `run_success_preview_saved_auto_asifits`
  - `run_success_processing_completed`
  - `Run Hierarchical Mosaic COMPLETED`
- Plus de crash `NameError` ni `broadcast` sur le scénario courant.
- Le marqueur SDS phase5 est présent et cohérent (`phase5_sds_polish_start`), sortie FITS+preview générée.

**Conclusion opérationnelle SDS (sur ce dataset):**
- SDS peut être considéré **réparé fonctionnellement** sur le flux principal.

**Piste qualitative ajoutée au plan (super-tiles):**
- Point à investiguer ensuite: harmonisation/normalisation inter-brutes dans une super_tile pour réduire les écarts photométriques intra-lot.
- Pistes techniques évidentes à prioriser:
  1. normalisation robuste par frame avant stack (médiane/échelle sur zone valide, avec clip sigma),
  2. correction de fond local par frame (offset + gain doux) avant coadd,
  3. contrôle qualité par frame (rejet ou down-weight des outliers de fond/couleur/FWHM),
  4. suivi de métriques de cohérence inter-canaux et inter-frames dans les logs.

**Audit ZeGrid (grid mode) — état actuel:**
- Vérifications statiques:
  - `grid_mode.py`, `zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py` compilent (`py_compile` OK).
  - signature `grid_mode.run_grid_mode(...)` compatible avec l'appel effectué depuis `run_hierarchical_mosaic`.
  - import `grid_mode` OK dans l'environnement venv du projet.
- Smoke minimal exécuté:
  - appel `run_grid_mode` sur dossier vide -> échec contrôlé attendu (`RuntimeError: Grid mode failed: no frames loaded`), sans crash interne inattendu.
- Limite d'audit:
  - pas de run ZeGrid réel récent dans le log fourni; impossible de certifier fonctionnel en production sans dataset grid dédié.

**Action restante ZeGrid:**
1. lancer un run réel ZeGrid sur dataset compatible grid,
2. valider fin de run + sortie FITS/preview,
3. confirmer absence de régression post-refactor.

### Iteration 6 — Fix ZeGrid `stack_plan.csv` Windows paths (2026-03-14 11:11)
**Erreur analysée (log confirmé):**
- `Grid mode failed: no frames loaded`
- Bien que `stack_plan.csv` soit présent, chaque ligne était rejetée avec `file not found`.
- Les entrées CSV contiennent des chemins Windows absolus (ex. `D:\ASTRO\...\Light_*.fit`).
- Sur Linux, l'ancien resolveur concaténait ces chaînes à `base_dir`, produisant des chemins invalides.

**Correctif appliqué (ciblé ZeGrid):**
- Renforcement de `_resolve_path(...)` dans `grid_mode.py` pour gérer les chemins cross-plateforme:
  1. support chemins natifs absolus/relatifs (comportement existant préservé),
  2. détection de formats Windows (`D:\...`, UNC, backslashes),
  3. fallback intelligent sous `base_dir`:
     - tentative avec la queue du chemin,
     - tentative par basename (`base_dir/<filename>`),
  4. sélection du premier candidat existant.

**Validation locale:**
- Relecture du CSV réel problématique:
  - avant fix: `frames=0`
  - après fix: `frames=24`, `all_exist=True`
- Test ajouté:
  - `tests/test_grid_mode_stack_plan_paths.py`
  - couvre la résolution de chemins Windows depuis `stack_plan.csv`.
- Résultats tests:
  - `pytest tests/test_grid_mode_stack_plan_paths.py` -> `1 passed`
  - revalidation des garde-fous SDS déjà en place -> `3 passed`

**Impact/régression:**
- Patch limité à `grid_mode.py` (chemin ZeGrid).
- Aucune modification des branches SDS/classique.

**Reste à faire:**
1. Relancer un run ZeGrid réel (avec `stack_plan.csv` actuel) pour confirmer fin de pipeline en conditions GUI.
2. Vérifier sortie FITS/preview + absence d'erreurs `file not found` dans log.

### Iteration 7 — Vérification activité GPU en mode "I'm using master tiles" (2026-03-14 14:18)
**Demande:** vérifier si la Phase 5 annoncée "GPU" utilise réellement le backend GPU en mode existing master tiles.

**Constats log (run 14:02→14:11):**
- Message d'activation présent: `phase5_using_gpu` + `Phase 5 reprojection will use the GPU`.
- Plan GPU confirmé: `Phase5 VRAM budget ... phase5_gpu=1 plan.use_gpu=1`.
- Aucune trace de fallback runtime (`gpu_fallback_runtime_error`) pendant la reprojection.
- Backend two-pass confirmé GPU:
  - `[TwoPass] Normalized coverage blur applied ... using cupy_chunk/cupy_chunk`
  - `[TwoPass] Two-pass reprojection backend: gpu_all=True`
- Télémétrie GPU:
  - `gpu_used_mb` passe ~38→42MB en reproject, puis ~132MB pendant blur/gains,
  - `gpu_util_percent` non disponible (`None`) car source `cupy_meminfo`.

**Conclusion:**
- Le chemin GPU est bien emprunté pour cette exécution.
- L'impression "pas d'activité GPU" vient surtout de la télémétrie actuelle (mémoire disponible mais pas %utilisation), pas d'un fallback CPU caché détecté dans ce run.

**Piste d'amélioration (observabilité):**
1. ajouter un log explicite de succès backend (`gpu_reproject_backend_active`) au début/fin de chaque canal,
2. exposer compteurs chunks GPU traités,
3. intégrer une sonde util% via NVML quand disponible (au lieu de `cupy_meminfo` seul).


---

## M. 2026-03-26 — Plan opérationnel « normalisation globale inter‑tuiles »

### M1. Implémentation V1 (offset-only)
- [x] Ajouter extraction robuste des contraintes overlap pour `b_t` (offset)
- [x] Construire système global des contraintes (toutes arêtes actives)
- [x] Fixer une tuile de référence (`b_ref=0`)
- [x] Résoudre et appliquer `b_t` **avant reprojection**
- [x] Logger diagnostics:
  - nb contraintes retenues/rejetées
  - residual median / p95 post-correction
  - worst overlaps
- [x] Constat terrain: **impasse M1** sur dataset hétérogène (seams quasi inchangées D/E)

### M2. Implémentation V2 (gain+offset)
- [x] Étendre solve à `a_t + b_t` avec ancre (`a_ref=1`, `b_ref=0`)
- [x] Ajouter clamps conservateurs (`a_min/a_max`, `b_min/b_max`)
- [x] Ajouter régularisation légère sur `a_t` autour de 1.0
- [x] Rejeter outliers robustement (pairs hors bornes + IRLS robuste)
- [ ] Lancer run C (M2 ON, M1 OFF) et comparer métriques seams vs run E

### M3. Intégration pipeline
- [ ] Vérifier ordre strict:
  1) normalisation globale inter-tuiles
  2) reprojection
  3) tile weighting
  4) blend final
- [ ] Interdire toute permutation silencieuse de cet ordre

### M4. Weighting et finition
- [ ] Conserver tile weighting (rôle de blend, pas de correction photométrique)
- [ ] Ajouter feather multibande léger optionnel (OFF par défaut)
- [ ] Vérifier absence de halo/banding/softening excessif

### M5. Protocole de validation
- [ ] A/B/C sur même dataset:
  - A = baseline actuel
  - B = normalisation V1 (offset-only)
  - C = normalisation V2 (a+b)
- [ ] Mesurer:
  - seam amplitude sur overlaps
  - residual median/p95
  - homogénéité de fond
  - stabilité flux étoiles communes
- [ ] Décider GO/NO-GO pour activation par défaut

