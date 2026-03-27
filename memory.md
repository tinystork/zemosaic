# memory.md

## Existing content

# Memory (curated)

Last curated: 2026-03-15
Purpose: keep durable technical history + production-relevant decisions, remove transient debug noise.
Full previous log archived in: `memory_full_archive_20260315-1024.md`.

---

## 1) Project-level durable decisions

- Official GUI stack is Qt/PySide6.
- Tk legacy paths were progressively retired from official runtime paths (kept only where explicitly legacy/annexed).
- Three operational processing families remain distinct and must not be merged conceptually:
  1) Classic
  2) ZeGrid
  3) SDS
- Safety rule maintained through missions: mode-by-mode surgical patches, no broad transversal rewrites.

---

## 2) Packaging and runtime robustness (historical)

### Windows packaged runtime (PyInstaller)
- Main issues handled:
  - DLL search-path fragility in frozen mode
  - CuPy/fastrlock packaging misses
  - Shapely binary hidden-import misses
  - SEP top-level `_version` packaging issue
- Durable outcome:
  - stronger runtime hooks/spec for packaged dependencies
  - better packaged-mode diagnostics

### Build documentation alignment
- Clear split documented between:
  - `requirements.txt` (GPU-enabled path)
  - `requirements_no_gpu.txt` (CPU-only smaller package)
- Installer behavior clarified (packages existing `dist/` output, does not decide CUDA flavor itself).

---

## 3) GUI / UX durable fixes

### Qt filter dialog accessibility
- Right-side controls moved into scrollable area.
- Action buttons (OK/Cancel) kept permanently accessible.
- Saved geometry restore clamped against available screen geometry.

---

## 4) Mission: multi-mode quality harmonization (normalization / RGB / DBE)

Reference scope files:
- `agent.md`
- `followup.md`

### Mission intent
Bring non-classic modes closer to classic quality behavior while preserving non-regression and mode-specific architecture.

### Status
- E (SDS quality harmonization): completed.
- F (ZeGrid DBE priority): completed and field-validated.

---

## 5) Final RGB Equalization policy (important)

### Capability retained
`final_mosaic_rgb_equalize_enabled` remains available and documented for marginal datasets.

### Current production preference (Tristan)
- Set to OFF in user JSON config (`~/.config/ZeMosaic/zemosaic_config.json`).
- Rationale: limited benefit on current dataset, keep as optional targeted lever.

### Related tuning keys to remember
- `final_mosaic_rgb_equalize_clip_enabled`
- `final_mosaic_rgb_equalize_gain_clip`
- `existing_master_tiles_final_rgb_equalize_gain_clip`
- `sds_enable_final_rgb_equalize`
- `sds_final_rgb_equalize_gain_clip`

---

## 6) SDS branch — durable fixes and controls

### Stabilization fixes kept
- Regression fixes around SDS-specific path integrity (name/shape guards, geometry-safe behavior, OOM resilience).
- Geometry-changing crop prevented in SDS finalization when incompatible with global descriptor constraints.

### Optional SDS equalization controls (default conservative)
- `sds_enable_final_rgb_equalize` (default false)
- `sds_final_rgb_equalize_gain_clip`
- `sds_enable_final_black_point_equalize` (default false)

### Durable rule
SDS global geometry invariants are non-negotiable; quality improvements must remain geometry-safe.

---

## 7) ZeGrid branch — DBE implementation history and final state

### Initial gap (confirmed)
- Worker Phase 6 DBE path is bypassed in ZeGrid route.
- Grid mode had DBE hook logging but no effective final DBE processing at that point.

### Implemented resolution
- Added effective ZeGrid final DBE step in `grid_mode.py` before final mosaic save.
- Added explicit DBE status logging (`applied/skipped`, reason, strength/sigma/channels).
- DBE applied on valid-covered regions only.

### Halo artifact mitigation
Observed issue: dark halos around bright stars in ZeGrid only.
Root causes:
- simpler ZeGrid DBE behavior vs classic/SDS worker DBE
- GUI label mapping mismatch (`weak/strong` not aligned with grid-side aliases)

Fixes delivered:
- GUI strength alias alignment (`weak/normal/strong` now mapped correctly for ZeGrid)
- bright-object protection in DBE model estimation
- avoidance of correction over protected bright cores

### Validation status
- Unit tests added/updated for ZeGrid DBE behavior and strength aliases.
- Real ZeGrid run validated visually by Tristan as successful.
- No blocking ERROR observed in end-of-run checks.

---

## 8) Production-relevant variables / switches (quick reference)

### Global / classic-facing
- `final_mosaic_rgb_equalize_enabled`
- `final_mosaic_rgb_equalize_clip_enabled`
- `final_mosaic_rgb_equalize_gain_clip`
- `final_mosaic_dbe_enabled`
- `final_mosaic_dbe_strength`
- `final_mosaic_dbe_obj_k`
- `final_mosaic_dbe_obj_dilate_px`
- `final_mosaic_dbe_sample_step`
- `final_mosaic_dbe_smoothing`

### Existing master tiles
- `existing_master_tiles_rgb_balance_prephase5`
- `existing_master_tiles_rgb_balance_gain_clip`
- `existing_master_tiles_rgb_balance_min_pixels`
- `existing_master_tiles_final_rgb_equalize_gain_clip`

### SDS
- `sds_enable_final_rgb_equalize`
- `sds_final_rgb_equalize_gain_clip`
- `sds_enable_final_black_point_equalize`
- `sds_mode_default`
- `sds_coverage_threshold`
- `sds_min_coverage_keep`

### ZeGrid
- `poststack_equalize_rgb` / grid RGB equalization path
- `final_mosaic_dbe_enabled`
- `final_mosaic_dbe_strength`

---

## 9) Known durable guardrails for future debug

- Always reason mode-by-mode (Classic / SDS / ZeGrid).
- Validate with artifacts + logs, not assumptions.
- Keep new quality controls conservative by default (opt-in where uncertain).
- Do not remove safety guardrails without replacement + explicit rationale.
- Prefer reversible/tunable config switches to hard behavior changes.

---

## 10) Current mission-level conclusion

- Multi-mode quality mission reached practical completion on current dataset.
- ZeGrid DBE is now operational and aligned with GUI toggles.
- Final RGB equalization remains available but intentionally OFF by default for current production use.

## 2026-03-15 10:39 — Mission kickoff: seamless + preview

- Nouveau scope validé: couture inter-tuiles + preview PNG trop stretchée.
- Baseline de référence confirmée dans `example/out/ref/`.
- Audit rapide mécanismes existants réalisé:
  - Preview final en Phase6 (`zemosaic_worker.py`) avec paramètres actuellement hardcodés (`p_low=2.5`, `p_high=99.8`, `asinh_a=20.0`, downscale cap 4000).
  - Stretch backend via `stretch_auto_asifits_like(_gpu)` (`zemosaic_utils.py`).
  - Seams: intertile affine/background matching/recenter côté worker; overlap regression + laplacian pyramid blending côté `grid_mode.py`.
- Décision de séquencement: traiter d'abord le viewer PNG (quick win), puis première tentative seams.
- `followup.md` remplacé par une nouvelle checklist alignée au nouveau `agent.md`.

## 2026-03-15 10:42 — Preview PNG power-user JSON tuning (no UI change)

- Implémenté dans `zemosaic_worker.py` la lecture de paramètres preview PNG via JSON uniquement (power-user):
  - `preview_png_p_low`
  - `preview_png_p_high`
  - `preview_png_asinh_a`
  - `preview_png_max_dim`
- Valeurs par défaut conservatrices intégrées côté worker:
  - `p_low=1.0`, `p_high=99.9`, `asinh_a=12.0`, `max_dim=3200`
- Ajout d'un log explicite `run_info_preview_png_params` pour traçabilité run.
- Config utilisateur mise à jour (`~/.config/ZeMosaic/zemosaic_config.json`) avec ces valeurs raisonnables.

## 2026-03-15 11:05 — Audit refresh + vérification impl preview JSON + avancement checklist

- Relecture mission effectuée: `agent.md` + `followup.md`.
- Vérification implémentation dernière action (preview JSON power-user) OK:
  - clés worker détectées: `preview_png_p_low`, `preview_png_p_high`, `preview_png_asinh_a`, `preview_png_max_dim`
  - log runtime présent: `run_info_preview_png_params`
  - config user présente: `1.0 / 99.9 / 12.0 / 3200`
  - `py_compile zemosaic_worker.py` OK
- Baseline PNG ref capturée et exportée:
  - `example/out/ref/preview_baseline_metrics_2026-03-15.md`
  - `example/out/ref/preview_baseline_metrics_2026-03-15.json`
- `followup.md` mis à jour:
  - B3 ✅
  - C1 ✅
  - C2 ✅
  - C3 Run A ✅
  - D1 cible #1 choisie ✅ (MT14 / classic-reproject-like path)
- Prochain item actif: D1.2 isoler la cause dominante des seams sur cible #1.

## 2026-03-15 18:57 — D2 seams v1 implémenté (JSON power-user, no UI)

- Nouveau levier seams minimal-risk ajouté: `intertile_affine_blend` (borne [0..1], défaut code=1.0).
- Intention: réduire la visibilité des coutures en atténuant les corrections photométriques inter-tuiles trop fortes:
  - `gain' = 1 + (gain-1)*blend`
  - `offset' = offset*blend`
- Intégration:
  - helper `_blend_affine_corrections(...)`
  - appliqué sur chemins `assemble_final_mosaic_incremental` et `assemble_final_mosaic_reproject_coadd` après composition anchor shift.
  - logs ajoutés:
    - `run_info_incremental_affine_blend`
    - `assemble_info_intertile_affine_blend`
    (incluent medians before/after sur |gain-1| et |offset|).
- Garde-fous anti-surcorrection:
  - clamp strict du blend à [0,1]
  - blend≈1 => comportement legacy inchangé
  - blend=0 => neutralisation des corrections intertile
- Paramètre branché en config user pour run terrain: `intertile_affine_blend = 0.8`.
- Validation statique: `python3 -m py_compile zemosaic_worker.py` OK.
- Prochaines validations terrain prévues (par Tristan en un seul lot):
  - C3 Run B (viewer PNG nouveau preset),
  - D3 run comparatif seams v1 sur ref.

## 2026-03-15 19:32 — Clarification RGB-EQ + persistance des clés power-user JSON

- Clarification: `[RGB-EQ] poststack_equalize_rgb enabled=True` ≠ `final_mosaic_rgb_equalize_enabled`.
  - `poststack_equalize_rgb` = equalization post-stack (amont/chemin tuiles)
  - `final_mosaic_rgb_equalize_enabled` = equalization RGB finale mosaïque
- Constat run récent: `final_mosaic_rgb_equalize_enabled` était repassé à `true` dans snapshot; remis à `false` dans config user.
- Cause de non-persistance des clés power-user ajoutées précédemment: `save_config` ne conserve que les clés présentes dans `DEFAULT_CONFIG` (`zemosaic_config.py`).
- Correctif structurel appliqué:
  - ajout à `DEFAULT_CONFIG` de `preview_png_p_low`, `preview_png_p_high`, `preview_png_asinh_a`, `preview_png_max_dim`
  - ajout à `DEFAULT_CONFIG` de `intertile_affine_blend`
- Config user réappliquée:
  - `final_mosaic_rgb_equalize_enabled=false`
  - `poststack_equalize_rgb=true` (inchangé)
  - `preview_png_*` conservateurs
  - `intertile_affine_blend=0.8`
- Validation statique: `py_compile zemosaic_config.py zemosaic_worker.py` OK.

## 2026-03-15 20:01 — Feedback terrain + retune preset v2

Feedback utilisateur après run:
- Preview: mieux, mais coeurs étoiles/galaxies encore brûlés.
- Final images: composante bleue sur-corrigée.
- Seams: toujours nettement visibles.

Actions immédiates (JSON-only, sans UI):
- Viewer anti-burn v2:
  - `preview_png_p_low=0.3`
  - `preview_png_p_high=99.97`
  - `preview_png_asinh_a=0.1`
- Color safety:
  - `poststack_equalize_rgb=false`
  - `final_mosaic_rgb_equalize_enabled=false`
- Seams retune v2:
  - `intertile_affine_blend=0.65`
  - `intertile_recenter_clip=[0.92,1.08]`
  - `apply_radial_weight=true`
  - `radial_feather_fraction=0.90`

Intention: réduire clipping hautes lumières preview, limiter sur-correction chromatique (bleu), adoucir jonctions inter-tuiles par pondération radiale + corrections intertile moins agressives.

## 2026-03-15 20:40 — Isolation pass: low/mid ADU color drift + seams persistent

Symptôme confirmé user:
- divergence RGB sur courbe descendante (200–600 ADU) apparue post-refactor,
- seams toujours visibles,
- preview encore un peu brûlé (coeurs).

Hypothèse principale pour dérive couleur: combinaison de traitements finaux (poststack RGB-eq / black-point equalization / DBE) au-delà de l'assemblage master tiles.

Actions immédiates (preset diagnostic-safe JSON, no UI):
- `poststack_equalize_rgb=false`
- `final_mosaic_rgb_equalize_enabled=false`
- `final_mosaic_black_point_equalize_enabled=false`
- `final_mosaic_dbe_enabled=false`
- preview anti-burn affiné: `preview_png_asinh_a=0.08` (p_low=0.3, p_high=99.97)
- seams retune v3: `intertile_affine_blend=0.50`, `intertile_recenter_clip=[0.95,1.05]`, `intertile_overlap_min=0.10`, `intertile_robust_clip_sigma=2.0`, `radial_feather_fraction=0.92`

Correctif structurel persistance config:
- ajout à `DEFAULT_CONFIG` des clés
  - `final_mosaic_black_point_equalize_enabled`
  - `final_mosaic_black_point_percentile`
  pour éviter pertes silencieuses au save.

## 2026-03-15 21:59 — Post-run diag: preview still burned, colors shifted, no clear log markers

- Run `run_mt_diag_v3` produced artifacts in `example/out/run_mt_diag_v3/` but no dedicated `run_config_snapshot.json` in that folder.
- CLI run path currently does not emit the same rich GUI log markers into `zemosaic_filter.log`, limiting postmortem traceability.
- Observed symptom persistence confirmed by user.

Immediate corrective actions applied:
1) Preview color oddities isolation
- Added JSON power-user key: `preview_png_apply_wb` (default `False`) and wired in preview stretch calls.
- Rationale: auto white-balance in preview path can create non-physical color shifts and perceived burn in bright cores.

2) Preview anti-burn retune
- `preview_png_p_low=0.25`
- `preview_png_p_high=99.995`
- `preview_png_asinh_a=0.15`
- `preview_png_apply_wb=false`

3) Final color drift isolation (existing-master path)
- `existing_master_tiles_rgb_balance_prephase5=false`
- Rationale: master tiles already color-balanced by user observation; this prephase balance can reintroduce channel divergence on descending curves.

Files updated:
- `zemosaic_worker.py` (preview apply_wb config wiring)
- `zemosaic_config.py` (`DEFAULT_CONFIG` includes `preview_png_apply_wb`)
- `example/diagnostic_profile_v3.json` (new values)
- user config mirror updated similarly.

## 2026-03-15 22:10 — CLI logging wired + first actionable root-cause from worker log

- CLI path fixed to actually execute `run_hierarchical_mosaic(...)` from `__main__` (previously parsed args but did not run, causing silent no-output behavior).
- CLI now writes worker logs to `~/.config/ZeMosaic/zemosaic_worker.log` with live progress markers.
- Evidence from worker log (existing-master diagnostic run):
  - `existing_master_tiles_mode: anchor gain applied to tile ... gains=(0.674,0.665,0.649)`
  - later `gains=(0.529,0.518,0.506)`
  - strong per-channel anchor gains likely source of RGB drift in low/mid ADU.
  - two-pass coverage renorm still active in phase5 (`[TwoPass] Second pass requested ...`).
- Hypothesis sharpened: color drift is dominated by photometric anchor/intertile gain application (and possibly two-pass renorm interaction), not preview-only transform.

## 2026-03-15 22:15 — Targeted fix: existing-master anchor photometry guard + CLI rerun command

- Added config-gated control for existing-master anchor photometry in `assemble_final_mosaic_reproject_coadd`:
  - `existing_master_tiles_anchor_photometry_enabled` (default True)
  - `existing_master_tiles_anchor_gain_clip` (default [0.90, 1.10])
- Best-effort anchor now skips when disabled, with explicit log marker.
- Hard gain clip tightened from historical `(0.5, 2.0)` to config-driven conservative clip.
- Diagnostic profile and user config set to isolate drift:
  - `existing_master_tiles_anchor_photometry_enabled=false`
  - `existing_master_tiles_anchor_gain_clip=[0.95,1.05]`
- Expectation: suppress large per-tile anchor gains that can induce RGB divergence in low/mid ADU.

## 2026-03-15 22:34 — Decision: restore anchor + full raw GUI run; black-point hard guard

- User feedback: disabling anchor did not fix FITS color drift; preview became too dark.
- Decision: restore existing-master anchor controls to default enabled for normal behavior.
- Strong clue from previous log: black-point equalization still applied even when expected disabled.
- Safety patch: black-point application now requires both
  - `final_mosaic_black_point_equalize_enabled == True`
  - `final_mosaic_black_point_percentile > 0.0`
  (patched in all three finalization call sites).
- Current config prepared for next validation:
  - `existing_master_tiles_anchor_photometry_enabled=true`
  - `existing_master_tiles_anchor_gain_clip=[0.90,1.10]`
  - `final_mosaic_black_point_equalize_enabled=false`
  - `final_mosaic_black_point_percentile=0.0`

## 2026-03-15 23:10 — Confirmed drift path + black-point coercion bug fix

- User A/B confirms disabling `poststack_equalize_rgb` reduces RGB drift symptoms.
- Root bug found in shared phase5 pipeline: `final_mosaic_black_point_percentile` parsing used `or 0.1`, forcing `0.0 -> 0.1` and reapplying black-point pedestal unexpectedly.
- Fix applied:
  - black-point default fallback in shared phase5 set to disabled (`False`) when missing
  - removed `or 0.1` coercion so explicit `0.0` remains `0.0`
  - existing guards (`enabled` and `percentile>0`) now effective in practice.
- Preview retune for less darkness while keeping anti-burn:
  - `preview_png_p_low=0.20`, `preview_png_p_high=99.995`, `preview_png_asinh_a=0.35`, `preview_png_apply_wb=false`.

## 2026-03-15 23:35 — Confirmation user: poststack RGB-eq likely culprit; preview dark retune

- User run GUI with `poststack_equalize_rgb=False`: courbe sans aberration + FITS visuellement cohérent.
- Confirms hypothesis: sub-stack RGB equalization path can induce chromatic drift on this dataset.
- Preview still too dark; retuned JSON preview preset to brighter-safe:
  - `preview_png_p_low=0.5`, `preview_png_p_high=99.9`, `preview_png_asinh_a=0.10`, `preview_png_apply_wb=false`.
- Kept color-safe final toggles OFF:
  - `final_mosaic_rgb_equalize_enabled=false`
  - `final_mosaic_black_point_equalize_enabled=false`, `final_mosaic_black_point_percentile=0.0`.

## 2026-03-16 00:10 — Config path unification for git-friendly workflow

- Updated `zemosaic_config.py` config path resolution precedence:
  1) `ZEMOSAIC_CONFIG_PATH` (explicit override)
  2) project-local `zemosaic_config.json` (repo root) when present
  3) fallback user config `~/.config/ZeMosaic/zemosaic_config.json`
- Effect: GUI/worker now read+write project config by default in repo runtime, so settings are versionable/pushable in git.

## 2026-03-16 00:14 — Revert config/log location to original ~/.config behavior

- Reverted `zemosaic_config.py` path resolution: active config path is again `~/.config/ZeMosaic/zemosaic_config.json` only (original behavior).
- Confirmed worker log path remains original: `~/.config/ZeMosaic/zemosaic_worker.log`.

## 2026-03-16 00:21 — Repo-local config/log paths restored on request

- `zemosaic_config.py`: `get_config_path()` now points to repo-local `.../zemosaic/zemosaic/zemosaic_config.json`.
- `zemosaic_worker.py`: worker log path now points to repo-local `.../zemosaic/zemosaic/zemosaic_worker.log`.
- One-time sync performed from `~/.config/ZeMosaic/zemosaic_config.json` to repo config file to preserve current runtime settings.

## 2026-03-16 00:42 — Constat consolidé + plan mission (session handoff)

Constats validés terrain:
- Seams classiques toujours visibles (dataset pauvre + confirmé aussi sur set plus complet).
- `poststack_equalize_rgb` est le principal suspect de drift chromatique:
  - quand `poststack_equalize_rgb=False`, courbe RGB redevient cohérente et FITS visuellement acceptable.
- Preview PNG: problème de compromis tonemap (tour à tour brûlé / trop sombre / un peu trop lumineux selon preset).

Analyse technique (code):
- `poststack_equalize_rgb` actuel repose sur `equalize_rgb_medians_inplace` (médiane globale par canal, gains appliqués au sous-stack entier).
- Cette logique est conceptuellement fragile sur mosaïques hétérogènes (fond non uniforme, couverture partielle, objets brillants), ce qui explique la sur-correction R/B observée.

Décision de mission:
- Traiter `poststack_equalize_rgb` comme chantier dédié (robustification v2) et non simple tuning de coefficient.
- Maintenir par défaut `poststack_equalize_rgb=False` jusqu’à validation robuste.
- Agent/followup amendés pour inclure ce dossier explicitement + critères de release gate.

Contexte config/log:
- Chemins rétablis sur repo (à la demande user):
  - config active: `/home/tristan/zemosaic/zemosaic/zemosaic_config.json`
  - worker log: `/home/tristan/zemosaic/zemosaic/zemosaic_worker.log`

## 2026-03-16 00:55 — Implémentation H validée: `poststack_equalize_rgb` robust v2 + télémétrie

Implémentation réalisée (CPU+GPU+defaults):

- `zemosaic_align_stack.py`
  - `equalize_rgb_medians_inplace` refactoré en mode robuste/conservateur:
    - masque de fond robuste basé sur luminance percentiles (`poststack_rgb_equalize_bg_percentile`, défaut `[5,85]`),
    - exclusion implicite des objets brillants/hot pixels via masque percentile + validité RGB,
    - clip gain conservateur (`poststack_rgb_equalize_gain_clip`, défaut `[0.95,1.05]`),
    - garde-fous fiabilité (`poststack_rgb_equalize_min_samples`, défaut `5000`; `poststack_rgb_equalize_min_coverage`, défaut `0.01`),
    - no-op explicite si fiabilité insuffisante.
  - `_poststack_rgb_equalization` mis à jour:
    - défaut `poststack_equalize_rgb=False`,
    - télémétrie enrichie: `samples`, `mask_coverage`, `raw_gains`, `clipped_gains`, `decision`, `applied`.

- `zemosaic_align_stack_gpu.py`
  - chemin GPU aligné sur la logique robuste CPU (appel helper CPU quand dispo),
  - mêmes champs de télémétrie / décisions no-op, logs homogènes.

- Politique produit par défaut (jusqu’à validation terrain):
  - `poststack_equalize_rgb=False` par défaut dans `zemosaic_config.py`,
  - fallback defaults GUI alignés à `False` (`zemosaic_gui_qt.py`, `zemosaic_gui.py`).

- Nouvelles clés JSON power-user documentées dans `DEFAULT_CONFIG`:
  - `poststack_rgb_equalize_gain_clip`
  - `poststack_rgb_equalize_bg_percentile`
  - `poststack_rgb_equalize_min_samples`
  - `poststack_rgb_equalize_min_coverage`

Suivi mission:
- `followup.md` section H cochée sur implémentation v2 + télémétrie + politique default OFF.


## 2026-03-16 00:58 — Preset terrain appliqué (UX case unique GUI)

- Contexte: l’utilisateur final n’a qu’une case à cocher (`poststack_equalize_rgb`) dans la GUI.
- Action: preset JSON "safe" fixé dans `zemosaic_config.json` pour rendre la case auto-portée sans réglages cachés:
  - `poststack_equalize_rgb=false` (par défaut)
  - `poststack_rgb_equalize_gain_clip=[0.95,1.05]`
  - `poststack_rgb_equalize_bg_percentile=[5.0,85.0]`
  - `poststack_rgb_equalize_min_samples=5000`
  - `poststack_rgb_equalize_min_coverage=0.01`
- Effet produit attendu:
  - OFF: comportement stable (pas de drift induit)
  - ON: algorithme robuste v2 + garde-fous/no-op automatique si fiabilité insuffisante.


## 2026-03-16 13:28 — Réévaluation seams (dataset plus lourd) + nouvelle proposition

Demande utilisateur: réévaluer la proposition anti-patchwork puis mettre à jour `agent.md`, `followup.md`, `memory.md`.

Décision réévaluée:
- Amélioration visuelle des seams jugée **possible et réaliste**.
- Correction de la proposition initiale: **ne pas** monter `intertile_overlap_min` à `0.10` sur ce dataset; garder `0.05` pour conserver assez d'information d'overlap.

Profil documenté "VISUAL_SEAMLESS_v1":
- `poststack_equalize_rgb=false`
- `intertile_affine_blend=0.40`
- `intertile_recenter_clip=[0.96,1.04]`
- `intertile_overlap_min=0.05`
- `intertile_robust_clip_sigma=2.0`
- `apply_radial_weight=true`
- `radial_feather_fraction=0.94`
- `radial_shape_power=2.6`
- `final_mosaic_dbe_enabled=true`
- `final_mosaic_dbe_strength=normal`
- `final_mosaic_dbe_smoothing=0.75`
- `final_mosaic_dbe_sample_step=20`
- `final_mosaic_dbe_obj_dilate_px=4`
- `preview_png_apply_wb=false`
- `preview_png_p_low=0.40`
- `preview_png_p_high=99.93`
- `preview_png_asinh_a=0.14`

Nouvelle proposition (reportée pour itération ultérieure):
- pass optionnel "seam-heal low-frequency" dédié au rendu visuel (non-science), activable par preset.

## 2026-03-17 — Mission pivot: seams root-cause reorientation

Pivot stratégique de la mission seams.

Constat consolidé:
- le problème principal ne semble pas être un manque global de recouvrement géométrique entre master tiles;
- le graphe brut est dense, mais le graphe réellement exploité après pruning est nettement plus maigre;
- les master tiles sont hétérogènes (coverage / sessions / qualité locale / résidus);
- le master-tile weighting actuel peut imprimer la géométrie des tuiles dans le rendu final au lieu de dissoudre les disparités.

Décision:
- rétrograder le tuning purement visuel et le futur seam-heal low-frequency en finition seulement;
- réorienter la mission principale vers:
  1. instrumentation du pipeline réel,
  2. rework du graphe photométrique,
  3. weighting V4 plus conservateur et moins dominateur.

Consigne durable:
- Classic d’abord;
- patchs chirurgicaux;
- tout nouveau levier sensible config-gated;
- `memory.md` à jour à chaque itération significative.

## 2026-03-20 — Proto V4 + centralisation de la connaissance de régression

Décision utilisateur: centraliser les rappels critiques dans `memory.md` pour éviter l'éparpillement (au lieu de dépendre de notes isolées).

### Ce qui a été fait aujourd'hui
- Ajout d'un proto **Weighting V4** config-gated (OFF par défaut):
  - `tile_weight_v4_enabled`
  - `tile_weight_v4_curve`
  - `tile_weight_v4_strength`
  - `tile_weight_v4_min`
  - `tile_weight_v4_max`
- Pruning intertile rendu configurable en runtime:
  - `intertile_prune_k`
  - `intertile_prune_weight_mode` (`area|strength|hybrid`)
- RUN A préparé dans `zemosaic_config.json` (V4 OFF + prune explicite).
- Bug de régression corrigé juste après patch: `NameError: intertile_prune_k_config is not defined` (signature `run_hierarchical_mosaic` complétée).

### Leçon durable (issue de `fix_regression.md`)
Le pipeline est **intriqué**: un patch local sur la voie Classic peut casser SDS/ZeGrid ou des wrappers partagés.

Règle durable à garder en tête avant chaque modif Classic:
1. Repérer les fonctions partagées touchées (surtout `zemosaic_worker.py` / utilitaires communs).
2. Préférer les changements config-gated et réversibles.
3. Exiger preuve minimale multi-voies (Classic + ZeGrid + SDS) après patch.
4. Documenter immédiatement dans `memory.md` le diff, le risque, et la preuve.

### Source de vérité opérationnelle
- `memory.md` devient la référence centrale pour:
  - décisions techniques durables,
  - alertes de régression connues,
  - discipline de validation après patch.


## 2026-03-23 — Incident NGC6888 (existing master tiles) + clarification quality gate

### Résultat terrain validé
- Run `NGC6888_3` produit une image correcte (seams absents) après exclusion des tuiles master aberrantes.
- Sortie validée: `zemosaic_MT124_R0.fits` + preview associée.

### Root cause opérationnelle observée
- Le run dégradé ("bouillie grise") provenait d'une master tile extrême dans les masters existants:
  - `master_tile_125.fits` avec médianes ~`1.44e8` (ordre de grandeur anormal vs autres tuiles ~`1e2..1e3`).
- Corrélation log confirmée côté worker:
  - outlier massif sur la tuile finale chargée (`idx=124` dans le run à 125 tuiles),
  - deltas photométriques TwoPass extrêmes (`TwoPassWorst`),
  - contamination du coadd final.

### Quality gate — état réel du câblage
Constat important pour la mission principale:
1. Le quality gate ZeQualityMT est bien présent et câblé (GUI -> config -> worker -> décision accept/reject).
2. Mais il s'applique au moment de la création de master tiles (Phase 3), pas à la validation des masters déjà existantes chargées depuis disque.
3. Sur les runs `NGC6888_2` / `NGC6888_3`, il était de toute façon désactivé (`quality_gate_enabled=false` dans `run_config_snapshot.json`).
4. Si le module `zequalityMT` est indisponible, le worker continue en mode dégradé (warning + pas de gate bloquant).

### Conséquence mission
- Le quality gate actuel ne protège pas le flux `use_existing_master_tiles=true` contre une tuile master photométriquement aberrante.
- Priorité mission: fiabiliser ce maillon sans dévier du pivot principal (instrumentation réelle + graph/weighting), via un garde-fou spécifique au chargement des existing master tiles avant Phase 5.

## 2026-03-23 12:28+ — Implémentation one-pass: quality gate pour existing master tiles

Livraison réalisée en une passe (scope chirurgical, mission principale):

### Changements code
- `zemosaic_worker.py`
  - Ajout d'un pré-check qualité dédié aux **existing master tiles** avant activation du mode existing:
    - `_existing_master_tile_robust_stats(...)`
    - `_scan_existing_master_tiles_quality(...)`
  - Le pré-check calcule des stats robustes par tuile (valid_frac, median, max_abs, ratio max/median, z-score robuste sur log-médiane inter-tiles).
  - Ajout d'une politique de décision configurable:
    - `existing_master_tiles_quality_gate_mode = warn|fail`
    - en `warn`: run continue avec alertes explicites
    - en `fail`: arrêt propre avant coadd (`run_error_existing_master_tiles_quality_gate_failed`)
  - Logs ajoutés:
    - `run_info_existing_master_tiles_quality_precheck`
    - `run_warn_existing_master_tile_suspect`

- `zemosaic_config.py`
  - Ajout des defaults persistés:
    - `existing_master_tiles_quality_gate_enabled` (default `True`)
    - `existing_master_tiles_quality_gate_mode` (default `"warn"`)
    - `existing_master_tiles_quality_gate_sigma_threshold` (default `8.0`)
    - `existing_master_tiles_quality_gate_ratio_threshold` (default `5000.0`)
    - `existing_master_tiles_quality_gate_min_valid_frac` (default `0.05`)

- `zemosaic_config.json`
  - clés ajoutées pour activer le nouveau pré-check en runtime sans UI additionnelle.

### Validation technique
- Compilation statique OK:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_config.py`
- Vérification présence câblage/logs par grep: OK (`run_info_existing_master_tiles_quality_precheck`, `run_warn_existing_master_tile_suspect`, mode `warn|fail`).

### Impact mission
- Le flux `use_existing_master_tiles=true` n'est plus aveugle aux outliers photométriques massifs.
- Le risque "bouillie grise" due à une master tile catastrophique est désormais détecté explicitement avant Phase 5 (warn/fail selon politique).


## Imported from memory_compacted.md

# Memory (compacted)

## 2026-01-28 / 2026-01-29 — Windows PyInstaller build (GPU + Shapely + SEP)

### Problem
The Windows packaged build showed multiple issues not present in the non-compiled Python run:
- GPU not used in packaged mode (`phase5_using_cpu`, GPU unavailable / CuPy import failures)
- Phase 4 grid failure during `find_optimal_celestial_wcs()`
- Additional packaged-only alignment/runtime import errors

### Root causes confirmed
- Packaged runtime DLL search path was incomplete for frozen execution.
- CuPy was not correctly bundled at first; later a missing dependency was identified:
  - `fastrlock` missing caused CuPy import failure in packaged mode.
- Shapely packaging was incomplete:
  - runtime/import issues around bundled DLL resolution
  - later confirmed missing module: `shapely._geos`
- `sep_pjw` packaging was incomplete in the packaged build:
  - missing top-level `_version` import caused downstream alignment/runtime failures.

### Changes made
- `pyinstaller_hooks/rthook_zemosaic_sys_path.py`
  - added frozen DLL-path setup for `sys._MEIPASS`
  - added support for discovered `*.libs` folders and DLL-containing subfolders
  - added safer `os.add_dll_directory()` handling, including `WinError 206` mitigation
- `ZeMosaic.spec`
  - added packaging adjustments for Shapely
  - added stronger CuPy bundling / checks
  - added explicit handling for `fastrlock`
  - added optional packaging support for `sep`, `sep_pjw`, and `_version`
  - added explicit bundling of `_version.py`
- `pyinstaller_hooks/hook-shapely.py`
  - added hidden import support for `shapely._geos`
- Logging/instrumentation improved in relevant modules to expose packaged-only failures more clearly.

### Validation / observed status
- Shapely `WinError 206` issue was mitigated.
- CuPy root cause was narrowed down and at least one missing dependency (`fastrlock`) was fixed.
- Phase 4 packaged failure was traced to missing `shapely._geos`.
- Packaged alignment issues were traced to missing `_version` for `sep_pjw`.
- At one point GPU detection/use appeared to recover in packaged logs, but packaged runs still showed additional Phase 3 / Phase 4 instability compared with non-compiled runs.

### Remaining caution
This thread involved several packaged-only import/runtime issues. Any future packaging work should preserve:
- frozen DLL search-path setup
- explicit hidden imports for fragile binary packages
- detailed packaged runtime logging

---

## 2026-03-12 — Qt filter dialog usability issue (small screens / scaling)

### Problem
A user reported that the **Start / OK** button in `zemosaic_filter_gui_qt.py` can be unreachable on some screen sizes or DPI scaling settings because it ends up outside the visible dialog area.

### Root cause identified
- The filter dialog right-side controls are stacked directly in a layout with **no `QScrollArea`**.
- The saved dialog geometry is restored without clamping to the current screen bounds.

### Planned fix
- Put the **right-side controls column** inside a `QScrollArea`.
- Keep the **OK/Cancel button box fixed outside** the scrollable area at the bottom of the dialog.
- Clamp restored geometry to the current screen `availableGeometry()` before applying it.

### Session status
- Root cause reviewed in code.
- Mission files prepared (`agent.md`, `followup.md`).
- No code change applied yet in this session.


## Imported from memory_full_archive_20260315-1024.md

# Memory (compacted)

## 2026-01-28 / 2026-01-29 — Windows PyInstaller build (GPU + Shapely + SEP)

### Problem
Packaged Windows build had failures not seen in non-compiled runs:
- GPU path falling back to CPU (`phase5_using_cpu`) due CuPy import/runtime issues.
- Phase 4 failure around `find_optimal_celestial_wcs()` in packaged mode.
- Additional packaged-only alignment/import errors.

### Root causes confirmed
- Frozen DLL search path in packaged runtime needed stronger setup.
- CuPy packaging/dependencies were incomplete early on (notably missing `fastrlock`).
- Shapely packaging was incomplete (`shapely._geos` missing in packaged runtime).
- `sep_pjw` expected top-level `_version` that was not always bundled.

### Changes made
- `pyinstaller_hooks/rthook_zemosaic_sys_path.py`
  - Added frozen DLL-path setup for `sys._MEIPASS` and discovered `*.libs`/DLL folders.
  - Added safer `os.add_dll_directory()` handling with `WinError 206` mitigation.
- `ZeMosaic.spec`
  - Added stronger hidden-import/data handling for CuPy/fastrlock/Shapely/SEP-related pieces.
  - Added explicit `_version.py` bundling path.
- `pyinstaller_hooks/hook-shapely.py`
  - Added hidden import support for `shapely._geos`.
- Runtime diagnostics/logging improved in key modules to expose packaged-only failures.

### Validation summary
- Shapely `WinError 206` path issue mitigated.
- Missing packaged dependencies were identified and patched iteratively.
- Some packaged logs later showed GPU detection/use recovery, but packaged behavior remained more fragile than non-compiled runs in parts of the workflow.

### Remaining caution
For packaging work, preserve:
- frozen DLL search-path setup,
- explicit hidden imports for binary dependencies,
- detailed runtime logging in packaged mode.

---

## 2026-03-12 — Qt filter dialog usability fix (`zemosaic_filter_gui_qt.py`)

### Topic
Start/OK button could become unreachable on smaller screens / high-DPI setups.

### Problem
In the Qt filter dialog, the right-side controls and button box were in the same non-scrollable column, so the bottom actions could fall outside the visible area.

### Root cause
- Right controls column was not inside a `QScrollArea`.
- Saved geometry was restored as-is without clamping to current screen `availableGeometry()`.

### Changes made
- `_build_ui()`:
  - Added `QScrollArea` for the right panel (`setWidgetResizable(True)`).
  - Kept preview group as left splitter widget.
  - Moved right controls into a dedicated inner container set as the scroll area widget.
  - Moved `QDialogButtonBox` outside scrollable content and anchored it in the main dialog layout under the splitter.
  - Added a trailing stretch in the scrollable controls layout for natural packing.
- `_apply_saved_window_geometry()`:
  - Added safe clamping of restored `(x, y, w, h)` against current screen `availableGeometry()`.
  - Added safe screen lookup fallback order: `screenAt(center)` -> `self.screen()` -> `primaryScreen()`.
  - Preserved fail-safe behavior if screen lookup/clamp logic cannot run.

### Validation performed
- Static checks:
  - `python3 -m py_compile zemosaic_filter_gui_qt.py` passed.
- Code-level sanity checks:
  - OK/Cancel signal wiring (`accepted -> accept`, `rejected -> reject`) preserved.
  - Existing preview/stream/selection-related wiring untouched in the patch scope.
  - Splitter remains intact with preview left and controls right.

### Remaining risk / follow-up
- Manual GUI verification on constrained-height/high-DPI display was not run in this headless session.
- Smallest safe next step: launch the dialog in a constrained-height scenario and confirm right-panel scrolling + always-visible OK/Cancel.

---

## 2026-03-12 — Packaging/docs alignment for GPU vs CPU-only builds

### Topic
Clarified how Windows/macOS/Linux packaged builds decide whether CuPy/CUDA support is included, and aligned the helper scripts / installer documentation with the current build layout.

### Problem
- Build/release docs did not clearly distinguish:
  - `requirements.txt` (current working GPU-enabled dependency set, including `cupy-cuda12x`)
  - `requirements_no_gpu.txt` (new CPU-only dependency set for smaller artifacts)
- Build helpers and installer assumptions needed to match the real packaging flow.
- `zemosaic_installer.iss` previously targeted obsolete `compile\...` paths instead of the current `dist\ZeMosaic\...` output layout.

### Changes made
- Added `requirements_no_gpu.txt`
  - mirrors the base dependency set
  - intentionally excludes CuPy so a smaller CPU-only packaged build can be produced
- Updated `compile/compile_zemosaic._win.bat`
  - default requirements file is again `requirements.txt`
  - supports overriding via `ZEMOSAIC_REQUIREMENTS_FILE`
  - message now states GPU support depends on the chosen requirements file
- Updated `compile/build_zemosaic_posix.sh`
  - default requirements file is again `requirements.txt`
  - supports overriding via `ZEMOSAIC_REQUIREMENTS_FILE`
  - installs `pyinstaller-hooks-contrib`
  - cleans `build/` and `dist/`
  - message now states GPU support depends on the chosen requirements file
- Updated `zemosaic_installer.iss`
  - installer now packages `dist\ZeMosaic\*`
  - `Icons` / `Run` point to `{app}\ZeMosaic.exe`
  - comments explicitly state that Inno Setup does not choose the CUDA package; it only packages whatever build already exists in `dist\ZeMosaic`
- Updated `README.md`
  - documents the default GPU-enabled path using `requirements.txt`
  - documents CPU-only builds using `requirements_no_gpu.txt`
  - explains that `.iss` does not select CUDA/CuPy version
  - adds Windows GitHub release guidance (publish zipped `dist\ZeMosaic` as a release asset, do not commit `dist/`)

### Validation performed
- `bash -n compile/build_zemosaic_posix.sh` passed.
- Static review confirmed Windows helper / POSIX helper / README are now consistent on:
  - default = `requirements.txt`
  - smaller package option = `requirements_no_gpu.txt`
  - installer packages the already-built output

### Important packaging note
- Current intended behavior remains:
  - use `requirements.txt` for the existing GPU-enabled build path (`cupy-cuda12x`)
  - use `requirements_no_gpu.txt` only when a smaller CPU-only package is wanted
- `zemosaic_installer.iss` is ignored by Git in this repo (`.gitignore` has `*.iss`), so versioning it requires `git add -f zemosaic_installer.iss` or changing ignore rules.

---

## 2026-03-12 — Future idea: installer with GPU/CuPy auto-detection

### Topic
Exploration only, no mission started yet.

### User intent
- Keep the current working GPU path based on `requirements.txt` / `cupy-cuda12x`.
- Consider a future Windows installer able to detect the target machine and install/download the appropriate GPU support automatically.
- Defer implementation for now.

### Assessment given
- This is considered viable, but not a small change.
- Estimated scope:
  - minimal viable approach: high complexity
  - robust/maintainable approach: very high complexity

### Recommended architecture discussed
- Prefer a CPU-only base installer plus an optional GPU enablement step.
- Detect:
  - Windows x64 environment
  - NVIDIA GPU presence
  - driver / CUDA compatibility level
- Then either:
  - download a prebuilt GPU add-on from GitHub Releases, or
  - install the matching CuPy package dynamically
- Recommended direction was to avoid a fully dynamic Inno Setup-only solution at first, and instead use:
  - a simple installer
  - plus a post-install/bootstrap GPU activation step with clear fallbacks

### Why this was deferred
- Requires coordinated changes across:
  - installer/bootstrap flow
  - release packaging strategy
  - GPU compatibility detection logic
  - runtime fallback behavior
  - documentation and testing matrix
- User asked to postpone this work and possibly revisit it later.


### 2026-03-13 09:53 — Iteration 2
- Scope: S0/B2 only (headless scope lock), no code migration.
- In scope: définir les chemins headless officiellement validés pour cette mission + lister les non-supportés.
- Out of scope: suppression Tk, changements runtime, config strategy B3, S1+.
- Files changed: followup.md (B2 checkboxes), memory.md (journalisation).
- Tests run:
  - `grep -nE "argparse|ArgumentParser|--no-gui|--headless|--cli|--qt-gui|--tk-gui|if __name__ == '__main__'|main\(" run_zemosaic.py`
  - `grep -nE "argparse|ArgumentParser|--no-gui|--headless|--cli|if __name__ == '__main__'|def main\(" zemosaic_worker.py | head`
  - `ls -la tests && grep -RInE "zemosaic_config|zemosaic_worker|headless|import" tests`
  - `grep -nE "headless|CLI|command line|zemosaic_worker|run_zemosaic|--config|input_folder|output_folder|tk-gui|qt-gui" README.md`
- Proof:
  - `run_zemosaic.py` est un launcher GUI (Qt par défaut, fallback Tk existant), pas un mode headless explicite (`--headless/--no-gui` absents).
  - `zemosaic_worker.py` expose un point d’entrée CLI argparse (preuve d’un chemin exécutable non-GUI, même si non officiellement documenté comme frontend).
  - Les tests repo importent directement `zemosaic_worker` (ex: `tests/test_empty_master_tile_guard.py`), confirmant un chemin headless d’import pertinent.
  - Gates roadmap S2/S5 citent explicitement: `import zemosaic_config` et `import zemosaic_worker`.
- Decisions:
  - **Validated headless paths (scope fermé S0/B2):**
    1) `python -c "import zemosaic_config"`
    2) `python -c "import zemosaic_worker"`
    3) exécution des tests unitaires qui importent `zemosaic_worker` sans lancer GUI.
  - **Non-supported headless paths (pour cette mission):**
    - toute exécution Tk GUI legacy (`zemosaic_gui.py`, `zemosaic_filter_gui.py`, `run_zemosaic.py --tk-gui`)
    - outils annexes standalone Tk (`lecropper.py`, `zequalityMT.py`, `zewcscleaner.py`, `diagnose.py`)
    - tout chemin nécessitant interaction graphique Tk (file dialogs / messagebox).
- Blockers:
  - Aucun blocage pour passer à B3.
- Next unchecked item: B3 — Audit direct Tk imports in `zemosaic_config.py`.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (scope lock only).
- Validated headless scope changed or stayed unchanged: changed (now explicitly locked).


### 2026-03-13 09:53 — Iteration 3
- Scope: S0/B3 only (`zemosaic_config.py` strategy), no runtime code changes.
- In scope: audit imports Tk directs + couplages indirects; choisir une stratégie canonique pour S2/S3.
- Out of scope: implémentation de la stratégie, suppression Tk, S1+.
- Files changed: followup.md (B3 checkboxes), memory.md (journalisation).
- Tests run:
  - `grep -nE "import tkinter|from tkinter|filedialog|messagebox|fd\.|mb\.|def prompt|def ask|preferred_gui_backend|load_config|save_config" zemosaic_config.py`
  - `sed -n '1,140p' zemosaic_config.py`
  - `sed -n '500,760p' zemosaic_config.py`
- Proof:
  - Import direct Tk présent en module-level sous `try/except`:
    - `import tkinter.filedialog as fd`
    - `import tkinter.messagebox as mb`
  - Valeur par défaut backend toujours Tk: `"preferred_gui_backend": "tk"`.
  - Couplage indirect: `load_config()` / `save_config()` appellent `mb.showwarning/showerror` en cas d’erreur (avec fallback console si `mb is None`).
  - Couplage UI legacy explicite: `ask_and_set_astap_path()` / `ask_and_set_astap_data_dir_path()` utilisent `fd`/`mb` (dialogs Tk).
- Decisions:
  - Stratégie B3 choisie: **split core/legacy helpers**.
  - Direction retenue pour S2/S3: garder un noyau config import-safe Tk-free pour runtime officiel/headless validé; isoler les prompts Tk (ASTAP dialog/messagebox) dans un helper legacy hors chemin officiel.
- Blockers:
  - Aucun blocage pour passer à B4.
- Next unchecked item: B4 — Confirm `lecropper` classification as annex / standalone tool.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit/strategy only).
- Validated headless scope changed or stayed unchanged: stayed unchanged (déjà verrouillé en B2).


### 2026-03-13 09:53 — Iteration 4
- Scope: S0/B4 only (statut `lecropper`), sans migration.
- In scope: confirmer classification annexe/standalone, confirmer hors-scope port Qt, tracer exigence de découplage S2.
- Out of scope: implémenter le découplage, modifier worker/runtime, S1+.
- Files changed: followup.md (B4 checkboxes), memory.md (journalisation).
- Tests run:
  - `grep -nE "lecropper|annex|standalone|phase séparée|out-of-scope|decoupl|découpl" ROADMAP_REMOVE_TKINTER.md | head`
  - `grep -nE "lecropper|annex|standalone|out of scope|decouple|decoupling" agent.md`
  - `grep -nE "import lecropper|from lecropper" zemosaic_worker.py | head`
- Proof:
  - Roadmap canon A confirme `lecropper` = outil annexe standalone; port Qt en phase séparée (post-release).
  - `agent.md` confirme explicitement: annexe, hors-scope pour port Qt maintenant, découplage requis du runtime officiel/headless validé.
  - `zemosaic_worker.py` prouve dépendance actuelle (`import lecropper`, `from lecropper import detect_autocrop_rgb`).

---

## 2026-03-13 — GPU memory management follow-up (Phase 3 focus)

### Context
- Initial investigation started from low apparent GPU usage in Phase 5 (`resource_telemetry.csv` + worker logs).
- Phase 5 on AC was updated so its dynamic VRAM budget no longer stays clamped by the inherited global `plan_cap` when the machine is plugged in and not in safe mode.
- Follow-up discussion then focused on whether Phase 3 suffers from the same issue and whether it should be relaxed too.

### Confirmed findings
- Phase 3 shares the same root safety clamp as Phase 5 on hybrid/battery-capable laptops:
  - global GPU safety commonly clamps `gpu_max_chunk_bytes` to `128 MB`,
  - logs showed Phase 3 running with `gpu_max_chunk_bytes=134217728` and `gpu_rows_per_chunk=256`,
  - older Phase 5 runs under the same guard showed much smaller effective row chunks there, so the symptom is more severe in Phase 5 than in Phase 3.
- Phase 3 is not structurally identical to Phase 5:
  - Phase 3 builds many master tiles and can process several tile tasks concurrently,
  - Phase 5 is much more serial and therefore benefits more directly from “bigger VRAM per task”.

### Important risk assessment for Phase 3
- Relaxing Phase 3 VRAM policy is riskier than Phase 5 because Phase 3 can run multiple master-tile tasks in parallel.
- One overly aggressive GPU budget can trigger:
  - intermittent GPU OOM,
  - repeated chunk backoff,
  - GPU disablement for the remainder of Phase 3 after repeated failures.
- Therefore, “same AC relaxation as Phase 5” should **not** be applied blindly to Phase 3.

### What seems realistic vs illusory
- Dynamic evaluation **per master tile start** is realistic and useful:
  - current free VRAM,
  - available RAM,
  - tile dimensions,
  - number of frames in the tile,
  - current GPU task concurrency.
- Dynamic evaluation per raw file is not worth it.
- Dynamic throttling of effective Phase 3 concurrency already exists via runtime-updated semaphores; growing the executor beyond its initial max without rebuilding it is not currently how Phase 3 works.

### Best direction discussed for Phase 3
- If the goal is to maximize throughput for very large datasets, Phase 3 should likely prefer:
  - **low GPU concurrency + larger per-task chunks**
  - rather than many simultaneous GPU-backed master tiles.
- Safer design direction:
  1. reevaluate GPU budget at the start of each master tile,
  2. decide CPU vs GPU for that tile dynamically,
  3. recompute `gpu_rows_per_chunk` for that tile,
  4. add a dedicated Phase 3 GPU semaphore (likely `1` on hybrid laptops).
- In short:
  - Phase 5: aggressive AC VRAM usage is a good lever.
  - Phase 3: use a more conservative/dynamic policy unless GPU concurrency is explicitly serialized.

### Throughput intuition: VRAM vs parallelization
- For Phase 3, simply maximizing VRAM per task is not automatically better than parallelization.
- The main gain from more VRAM comes from reducing chunking overhead.
- Once chunking overhead is low enough, pushing many GPU tasks in parallel often hurts more than it helps on an 8 GB laptop GPU.
- Practical conclusion from the discussion:
  - prefer one “fat” GPU stack at a time over several competing GPU stacks.

### Order-of-magnitude estimate discussed
- For a hypothetical master tile with `3000` inputs at `5 MiB` each, with Phase 3 chunking by rows:
  - the relevant simple-path estimate is:
    - `chunk_gpu_bytes ~= n_frames * frame_bytes * (rows_per_chunk / frame_height)`
  - with `rows_per_chunk=256`, `3000 x 5 MiB` gives roughly:
    - about `7.3 GiB` if frame height is `512`,
    - about `4.9 GiB` if frame height is `768`,
    - about `3.7 GiB` if frame height is `1024`,
    - about `2.5 GiB` if frame height is `1500`.
- This estimate is already large for a single GPU task on an 8 GB card and does **not** include all temporary allocations.
- For the WSC/PixInsight-style GPU path, the memory pressure is much worse; `3000` frames with `256` rows would be far beyond what an 8 GB GPU can tolerate without strong chunk reduction or CPU/streaming fallback.
- Conclusion from that example:
  - `256` rows can be borderline even for one task,
  - parallel GPU handling of several such master tiles would very likely saturate VRAM.

### Current status
- Phase 5 AC budget behavior was changed.
- No Phase 3 VRAM policy change has been implemented yet.
- If Phase 3 work resumes in a later session, the recommended next step is:
  - design a per-master-tile dynamic GPU budget with explicit GPU concurrency control, instead of copying the Phase 5 policy directly.
- Decisions:
  - Classification confirmée: **`lecropper` = annex / standalone tool**.
  - Port Qt `lecropper` confirmée **out-of-scope** pour cette mission S0→S5.
  - Exigence S2 verrouillée: suppression dépendance directe/indirecte à `lecropper` sur chemins officiels + headless validés.
- Blockers:
  - Aucun blocage pour passer à B5.
- Next unchecked item: B5 — Build initial Qt parity matrix.
- Lecropper status changed or not: unchanged (confirmed as annex/standalone).
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit/decision only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 09:53 — Iteration 5
- Scope: S0/B5 only (initial Qt parity matrix), no code migration.
- In scope: établir une matrice initiale de parité pour workflows officiels Qt-only/Tk-retirement.
- Out of scope: correction des gaps/blocants (S1/S2), refactor hors scope.
- Files changed: followup.md (B5 checkboxes), memory.md (journalisation + matrice).
- Tests run:
  - `grep -nE "Qt preview|Tk stable|backend|preferred_gui_backend|..." zemosaic_gui_qt.py`
  - `grep -nE "Tk interface instead|PySide6|Qt|Tk|backend|coexist|preview|install" zemosaic_filter_gui_qt.py`
  - `grep -nE -- "--tk-gui|--qt-gui|fallback|Tk backend|Qt backend|preferred_gui_backend|ZEMOSAIC_GUI_BACKEND" run_zemosaic.py`
- Proof:
  - `zemosaic_gui_qt.py` expose un groupe "Preferred GUI backend" avec options `tk` (stable) / `qt` (preview).
  - `zemosaic_filter_gui_qt.py` contient le message: "Install PySide6 or use the Tk interface instead."
  - `run_zemosaic.py` conserve `--tk-gui` et fallback automatique vers Tk si Qt indisponible.
- Decisions:
  - Matrice de parité initiale (S0):
    - **OK**: frontend Qt existe et chemin nominal Qt est présent (`run_qt_main`).
    - **GAP**: UI Qt expose encore le choix backend Tk/Qt + wording "Qt preview / Tk stable".
    - **GAP**: messaging filter Qt oriente encore vers Tk en fallback utilisateur.
    - **BLOCKING**: launcher officiel garde fallback Tk (`--tk-gui` + auto fallback).
    - **BLOCKING**: config default backend reste `tk` (vu en B3).
    - **BLOCKING**: dépendance worker→`lecropper` encore active (vu en B4).
    - **OUT-OF-SCOPE (P0 immédiat)**: port Qt de `lecropper` et purge totale des annexes Tk.
- Blockers:
  - Aucun blocage d’audit; blocants de migration identifiés pour S1/S2.
- Next unchecked item: B6 — Record S0 proof in `memory.md` and close S0 when explicit.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit matrix only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 09:53 — Iteration 6
- Scope: S0/B6 closeout only.
- In scope: consolider les preuves S0, vérifier critères explicites scope/headless/config strategy, fermer S0.
- Out of scope: démarrage S1, modifications runtime/config/UI.
- Files changed: agent.md (S0 checkbox), followup.md (B6 checkboxes), memory.md (journalisation).
- Tests run:
  - revue des entrées S0 déjà tracées en memory (B1→B5)
  - vérification checklist `followup.md` section B complète
- Proof:
  - B1 inventaire+classification Tk: fait et journalisé.
  - B2 scope headless validé: défini explicitement + non-supportés listés.
  - B3 stratégie config choisie: split core/legacy helpers.
  - B4 statut `lecropper`: annex/standalone confirmé, port Qt hors scope, découplage S2 requis.
  - B5 matrice parité initiale: OK/gap/blocking/out-of-scope établie.
- Decisions:
  - **S0 explicitement clos** (critères roadmap satisfaits).
  - `lecropper` status remains unchanged (annex/standalone).
- Blockers:
  - Aucun pour entrer en S1 quand demandé.
- Next unchecked item: C1 — Verify GUI startup workflow parity.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S0 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged (already locked in B2).


### 2026-03-13 09:57 — Iteration 7
- Scope: S1/C1 only — **Verify GUI startup workflow parity**.
- In scope: comparer le flux de démarrage officiel Qt vs fallback Tk existant, sans modifier le code.
- Out of scope: config load/save parity (C1.2/C4), logs parity (C1.3), shutdown (C1.4), filter/grid/SDS (C1.5+), S2+.
- Files changed: followup.md (C1 startup checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "def run_qt_main|...|worker" zemosaic_gui_qt.py | head`
  - `sed -n '6388,6465p' zemosaic_gui_qt.py`
  - `sed -n '500,620p' run_zemosaic.py`
- Proof:
  - Chemin Qt: `run_zemosaic.py` tente `from zemosaic.zemosaic_gui_qt import run_qt_main`; si succès, lance Qt.
  - `run_qt_main()` crée/réutilise `QApplication`, instancie `ZeMosaicQtMainWindow`, `show()`, puis `app.exec()`.
  - En cas d’échec import Qt, launcher bascule automatiquement vers Tk (`backend = "tk"`) + warning utilisateur.
  - Chemin Tk: vérifie/charge worker avant l’ouverture GUI et bloque le démarrage avec `messagebox.showerror` si worker indisponible.
  - Chemin Qt: l’app peut s’ouvrir même si backend worker indisponible; erreur reportée plus tard via message Qt lors du lancement process (non bloquant au démarrage fenêtre).
- Decisions:
  - Verdict C1.1: **parité de démarrage partielle / gap**.
  - Gap principal: comportement pré-check worker différent (Tk bloque avant UI, Qt laisse ouvrir UI puis échoue au run).
- Blockers:
  - Aucun blocage pour poursuivre C1.2.
- Next unchecked item: C1 — Verify config load/save parity.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S1 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 09:58 — Iteration 8
- Scope: S1/C1 only — **Verify config load/save parity**.
- In scope: comparer charge/sauvegarde config Qt vs Tk legacy sur le workflow officiel, sans modifier le code.
- Out of scope: logs parity (C1.3), shutdown parity (C1.4), filter/grid/SDS (C1.5+), S2+.
- Files changed: followup.md (C1 config checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "load_config\(|save_config\(|..." zemosaic_gui_qt.py`
  - `grep -nE "load_config\(|save_config\(|..." zemosaic_gui.py`
  - `sed -n '4088,4295p' zemosaic_gui_qt.py`
  - `sed -n '300,380p' zemosaic_gui.py`
  - `sed -n '4520,4625p' zemosaic_gui.py`
  - `sed -n '5858,5908p' zemosaic_gui_qt.py`
  - `grep -nE "WM_DELETE_WINDOW|_on_closing|save_config\(" zemosaic_gui.py`
- Proof:
  - Qt charge via `zemosaic_config.load_config()` dans `_load_config()` puis merge avec defaults internes.
  - Tk charge aussi via `zemosaic_config.load_config()` à l'init.
  - Qt sauvegarde avant run (`_save_config()` dans `_start_processing`) et à la fermeture (`closeEvent`).
  - Tk sauvegarde avant run (`zemosaic_config.save_config(self.config)` dans `_start_processing`) et sur certains changements (langue/backend), mais pas systématiquement à la fermeture.
  - Qt sérialise un snapshot JSON-safe et conserve davantage l’état persisté (keys connues + snapshot), là où Tk écrit l’état courant `self.config`.
- Decisions:
  - Verdict C1.2: **parité partielle / gap**.
  - Parité de base OK (load/save présents des deux côtés), mais persistance plus robuste/cadrée côté Qt; comportements de sauvegarde non strictement identiques.
- Blockers:
  - Aucun blocage pour poursuivre C1.3.
- Next unchecked item: C1 — Verify logs / feedback parity.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S1 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 9
- Scope: S1/C1 only — **Verify logs / feedback parity**.
- In scope: comparer logs/progress/feedback utilisateur Qt vs Tk sur le workflow officiel.
- Out of scope: shutdown behavior (C1.4), filter/grid/SDS parity (C1.5+), S2+.
- Files changed: followup.md (C1 logs checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "_append_log|qt_log_|QMessageBox|progress|eta|phase" zemosaic_gui_qt.py`
  - `grep -nE "_log_message|messagebox|progress|eta|phase" zemosaic_gui.py`
  - `sed -n '4328,4415p' zemosaic_gui_qt.py`
  - `sed -n '5118,5205p' zemosaic_gui_qt.py`
  - `sed -n '5460,5575p' zemosaic_gui_qt.py`
  - `sed -n '4840,5015p' zemosaic_gui.py`
- Proof:
  - Qt: `_append_log()` avec niveaux normalisés/prefixes + surlignage GPU; widgets dédiés progress/ETA/phase + messages fin/cancel/erreur via `QMessageBox`.
  - Tk: `_log_message()` avec niveaux/tags couleur, progression+ETA+phase, feedback fin/cancel/erreur via `messagebox`.
  - Les deux flux gèrent annulation sans dialogue d’erreur dur, et succès avec proposition d’ouverture du dossier de sortie.
- Decisions:
  - Verdict C1.3: **parité globale OK avec gaps mineurs de présentation**.
  - Gap mineur: style d’affichage des logs différent (QPlainTextEdit/prefixes Qt vs Text/tags Tk), sans impact fonctionnel P0.
- Blockers:
  - Aucun blocage pour C1.4.
- Next unchecked item: C1 — Verify clean shutdown behavior.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 10
- Scope: S1/C1 only — **Verify clean shutdown behavior**.
- In scope: comparer le flux de fermeture Qt vs Tk (arrêt worker, persistance, cleanup).
- Out of scope: filter/grid/SDS parity détaillée (C1.5+), S2+.
- Files changed: followup.md (C1 shutdown checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "closeEvent|_save_config|worker_controller.stop|cupy|plt.close" zemosaic_gui_qt.py`
  - `grep -nE "_on_closing|worker_process.terminate|_stop_gui_chrono|root.destroy" zemosaic_gui.py`
  - `sed -n '5778,5822p' zemosaic_gui_qt.py`
  - `sed -n '5004,5040p' zemosaic_gui.py`
- Proof:
  - Qt `closeEvent`: enregistre état UI, collecte config, `save_config`, stop worker si actif, cleanup GPU (CuPy), cleanup matplotlib, puis fermeture.
  - Tk `_on_closing`: confirmation si traitement actif, terminate worker, stop chrono, destroy fenêtre.
  - Différence: Qt inclut un cleanup de sortie plus complet (persist + GPU + matplotlib) que Tk.
- Decisions:
  - Verdict C1.4: **parité fonctionnelle OK avec gap de robustesse** (Qt plus propre à l’arrêt).
- Blockers:
  - Aucun blocage pour C1.5.
- Next unchecked item: C1 — Verify filter workflow if part of official frontend.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 11
- Scope: S1/C1 only — **Verify filter workflow if part of official frontend**.
- In scope: vérifier si le filter fait partie du frontend officiel et comparer le flux Qt/Tk.
- Out of scope: grid/SDS dédiés (C1.6/C1.7), S2+.
- Files changed: followup.md (C1 filter checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "filter|skip_filter_ui|launch_filter|..." zemosaic_gui_qt.py`
  - `grep -nE "filter|skip_filter_ui|launch_filter|..." zemosaic_gui.py`
  - `sed -n '5838,5955p' zemosaic_gui_qt.py`
  - `sed -n '4210,4455p' zemosaic_gui.py`
- Proof:
  - Qt: bouton Filter + prompt pré-run + dialog Qt (`zemosaic_filter_gui_qt`) + passage des overrides/filtered items au worker (`skip_filter_ui`, `filter_overrides`, `filtered_header_items`).
  - Tk: flux équivalent avec `zemosaic_filter_gui` + prompt pré-run + propagation overrides/items au worker.
  - Annulation du filter: dans les deux cas, pas de lancement worker si l’utilisateur annule explicitement le dialogue.
- Decisions:
  - Le filter est **partie officielle/relevante** du workflow frontend.
  - Verdict C1.5: **parité fonctionnelle globalement OK** (implémentations différentes mais comportement attendu aligné).
- Blockers:
  - Aucun blocage pour C1.6.
- Next unchecked item: C1 — Verify Grid mode path if official.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 12
- Scope: S1/C1 only — **Verify Grid mode path if official**.
- In scope: vérifier que le chemin Grid/global-coadd est bien officiel et comparer Qt/Tk.
- Out of scope: SDS détaillé (C1.7), S2+.
- Files changed: followup.md (C1 grid checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "grid|phase4_grid|final_assembly_method|reproject|incremental|global_coadd" zemosaic_gui_qt.py`
  - `grep -nE "grid|phase4_grid|final_assembly_method|reproject|incremental|global_coadd" zemosaic_gui.py`
  - `sed -n '2070,2188p' zemosaic_gui_qt.py`
  - `sed -n '2040,2148p' zemosaic_gui.py`
- Proof:
  - Qt expose explicitement `final_assembly_method` (reproject_coadd / incremental) et traite les événements `phase4_grid`/`p4_global_coadd_*`.
  - Tk expose aussi `final_assembly_method` (reproject_coadd / incremental) et gère `phase4_grid`/`p4_global_coadd_*`.
  - Le chemin Grid/global-coadd est donc présent des deux côtés dans le flux officiel.
- Decisions:
  - Grid path = **official/relevant**.
  - Verdict C1.6: **parité fonctionnelle OK** au niveau des chemins et signaux principaux.
- Blockers:
  - Aucun blocage pour C1.7.
- Next unchecked item: C1 — Verify SDS path if official/relevant.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 13
- Scope: S1/C1 only — **Verify SDS path if official/relevant**.
- In scope: vérifier pertinence officielle SDS et parité de flux Qt/Tk.
- Out of scope: implémentation de corrections S2+, refactor.
- Files changed: followup.md (C1 SDS checkbox), memory.md (journalisation).
- Tests run:
  - `grep -nE "sds_mode|SDS|seestar|_sds_" zemosaic_gui_qt.py`
  - `grep -nE "sds_mode|SDS|seestar|_sds_" zemosaic_gui.py`
  - `sed -n '1428,1490p' zemosaic_gui_qt.py`
  - `sed -n '6118,6185p' zemosaic_gui_qt.py`
  - `sed -n '3928,3975p' zemosaic_gui.py`
- Proof:
  - Qt expose des contrôles SDS (`sds_mode_default`, seuil SDS, flags Seestar) et synchronise ces choix vers le filter Qt.
  - Tk contient aussi une gestion SDS runtime (détection SDS, progression phase SDS, ETA/phase SDS).
  - Les deux GUIs gèrent les événements `phase4_grid` / global coadd en mode SDS.
- Decisions:
  - SDS path = **official/relevant**.
  - Verdict C1.7: **parité fonctionnelle globalement OK** (écarts d’UI non bloquants à ce stade).
- Blockers:
  - Aucun incident sur C1.
- Next unchecked item: C2 — Remove or plan removal of “Qt preview” wording.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 14
- Scope: S1/C2 only — **Remove or plan removal of “Qt preview” wording**.
- In scope: audit wording + plan de retrait (sans modification source).
- Out of scope: implémentation du retrait (S2/D3), refactor.
- Files changed: followup.md (C2.1 checkbox), memory.md (plan + preuves).
- Tests run:
  - `grep -nE "Qt GUI \(preview\)|backend_option_qt|preferred_gui_backend" zemosaic_gui_qt.py zemosaic_gui.py`
- Proof:
  - Wording localisé trouvé: `backend_option_qt = "Qt GUI (preview)"` (Tk et Qt UI layers).
- Decisions:
  - Plan retrait S2/D3: remplacer `Qt GUI (preview)` par libellé neutre Qt-only (ex. `Qt GUI`) et supprimer toute mention de statut préliminaire.
- Blockers:
  - Aucun.
- Next unchecked item: C2 — Remove or plan removal of “Tk stable” wording.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (planning only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 15
- Scope: S1/C2 only — **Remove or plan removal of “Tk stable” wording**.
- In scope: audit wording + plan de retrait (sans code).
- Out of scope: suppression effective du backend Tk (S2).
- Files changed: followup.md (C2.2 checkbox), memory.md (plan + preuves).
- Tests run:
  - `grep -nE "Classic Tk GUI \(stable\)|backend_option_tk|backend_change_notice" zemosaic_gui_qt.py zemosaic_gui.py`
- Proof:
  - Wording trouvé: `backend_option_tk = "Classic Tk GUI (stable)"`.
- Decisions:
  - Plan retrait S2/D3: supprimer ce libellé et toute mention de stabilité comparée Tk/Qt, cohérent avec Qt-only officiel.
- Blockers:
  - Aucun.
- Next unchecked item: C2 — Identify backend switch UI elements to eliminate in S2.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (planning only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 16
- Scope: S1/C2 only — **Identify backend switch UI elements to eliminate in S2**.
- In scope: inventorier les éléments UI de bascule backend à retirer.
- Out of scope: suppression effective (S2/D3).
- Files changed: followup.md (C2.3 checkbox), memory.md (inventaire).
- Tests run:
  - revue des occurrences `backend_selector_label`, `backend_change_notice`, `preferred_gui_backend`, `_backend_option_entries`, `_refresh_backend_combobox`, `_on_backend_combo_selected`.
- Proof:
  - Éléments à retirer en S2:
    1) Combo backend + label/notice dans `zemosaic_gui.py` (Tk legacy UI bloc langue/backend).
    2) Entrées backend `tk/qt` dans `zemosaic_gui_qt.py` (label “Preferred GUI backend”, options et handler de changement).
    3) Persistance liée: `preferred_gui_backend` / `preferred_gui_backend_explicit` côté UI.
    4) Flags launcher liés au choix backend (`--tk-gui`) côté `run_zemosaic.py` (traité dans S2/D1).
- Decisions:
  - Inventaire C2.3 verrouillé pour exécution S2.
- Blockers:
  - Aucun incident.
- Next unchecked item: C3 — List backend features still alive but not exposed in Qt.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (planning only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 17
- Scope: S1/C3 only — backend features actives non exposées en Qt.
- In scope: inventaire + classification + décision.
- Out of scope: implémentation d’exposition/suppression.
- Files changed: followup.md (C3 checkboxes), memory.md (inventaire/classification).
- Tests run:
  - comparaison clés config Tk vs Qt (script statique)
  - `grep -nE "altaz_alpha_soft_threshold|stack_ram_budget_gb" zemosaic_worker.py`
- Proof:
  - Features backend actives mais non exposées en Qt identifiées:
    1) `altaz_alpha_soft_threshold` — présente dans Tk/config et consommée dans worker.
    2) `stack_ram_budget_gb` — présente côté Tk/config et consommée dans worker (budget mémoire stack).
- Decisions / classification:
  - `altaz_alpha_soft_threshold` => **expose now** (cohérence avec autres contrôles Alt/Az déjà présents en Qt).
  - `stack_ram_budget_gb` => **legacy** (tuning expert historique, pas nécessaire pour parité frontend officielle immédiate).
  - `stack_ram_budget_gb` reste aussi **out-of-scope** pour ce sprint de retrait Tk (pas bloquant P0/P1 frontend).
- Blockers:
  - Aucun incident.
- Next unchecked item: C4 — Verify Qt UI writes the expected config keys.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit/classification only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 18
- Scope: S1/C4 only — **Verify Qt UI writes the expected config keys**.
- In scope: vérifier le flux write côté Qt (widgets -> config -> snapshot -> save -> worker kwargs).
- Out of scope: rework persistence.
- Files changed: followup.md (C4.1 checkbox), memory.md (preuve).
- Tests run:
  - `grep -nE "_collect_config_from_widgets|_serialize_config_for_save|_save_config|_build_worker_invocation" zemosaic_gui_qt.py`
  - `sed -n '4088,4378p' zemosaic_gui_qt.py`
  - `sed -n '5628,5758p' zemosaic_gui_qt.py`
- Proof:
  - `_collect_config_from_widgets()` parcourt `_config_fields` et écrit les valeurs normalisées dans `self.config`.
  - `_serialize_config_for_save()` produit un snapshot JSON-safe basé sur `persisted_keys + loaded_snapshot + config`.
  - `_save_config()` persiste ce snapshot.
  - `_build_worker_invocation()` copie ce snapshot dans `worker_kwargs` (donc clés persistées et attendues transmises au backend).
- Decisions:
  - Verdict C4.1: **OK** (pipeline d’écriture Qt cohérent pour clés attendues).
- Blockers:
  - Aucun.
- Next unchecked item: C4 — Verify persisted settings reload correctly.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 19
- Scope: S1/C4 only — **Verify persisted settings reload correctly**.
- In scope: valider le chemin reload côté Qt (load -> merge defaults -> UI init depuis config).
- Out of scope: tests dynamiques E2E.
- Files changed: followup.md (C4.2 checkbox), memory.md (preuve).
- Tests run:
  - revue `_load_config()` et initialisation des widgets (`self.config.get(...)` lors de la construction UI)
  - revue `_save_config()` qui met à jour `_loaded_config_snapshot` après sauvegarde.
- Proof:
  - `_load_config()` charge depuis `zemosaic_config`, fusionne avec defaults, normalise clés GPU/phase45.
  - Les widgets Qt sont initialisés depuis `self.config` (donc valeurs persistées reprises au démarrage).
  - Après sauvegarde, `_loaded_config_snapshot` est synchronisé, consolidant le cycle save/reload.
- Decisions:
  - Verdict C4.2: **OK** (reload cohérent observé statiquement).
- Blockers:
  - Aucun.
- Next unchecked item: C4 — Identify any Tk/Qt ambiguity in config behavior.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 20
- Scope: S1/C4 only — **Identify any Tk/Qt ambiguity in config behavior**.
- In scope: lister ambiguïtés de comportement config entre Tk et Qt.
- Out of scope: correction (S2/S3).
- Files changed: followup.md (C4.3 checkbox), memory.md (ambiguïtés).
- Tests run:
  - revue croisée `zemosaic_gui.py`, `zemosaic_gui_qt.py`, `zemosaic_config.py`.
- Proof / ambiguities:
  1) `preferred_gui_backend`:
     - config par défaut encore `tk`; coexistence Tk/Qt persiste dans les UI/launcher.
     - ambigu pour Qt-only officiel.
  2) Persistance fermeture:
     - Qt sauvegarde systématiquement en `closeEvent`; Tk non systématique à la fermeture.
  3) Phase 4.5:
     - Qt force la désactivation via `_disable_phase45_config`; ce garde-fou n’est pas symétrique explicitement côté Tk.
- Decisions:
  - Ambiguïtés C4 identifiées et tracées pour traitement S2/S3.
- Blockers:
  - Aucun incident.
- Next unchecked item: C5 — Update parity matrix.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:01 — Iteration 21
- Scope: S1/C5 closeout (partial) — parity matrix update + proof.
- In scope: mise à jour de la matrice à la lumière de C1→C4.
- Out of scope: lever les blockers (S2).
- Files changed: followup.md (C5.1 + C5.3), memory.md (matrice mise à jour + blockers).
- Tests run:
  - consolidation des audits C1/C2/C3/C4 déjà tracés.
- Proof / updated parity matrix:
  - Startup workflow parity: **GAP** (pré-check worker Tk vs Qt).
  - Config load/save parity: **GAP** (comportements de save différents).
  - Logs/feedback parity: **OK** (écarts visuels mineurs).
  - Clean shutdown parity: **OK-ish / GAP mineur** (Qt plus robuste).
  - Filter workflow: **OK**.
  - Grid path: **OK**.
  - SDS path: **OK**.
  - Hidden backend features: **GAP** (`altaz_alpha_soft_threshold` non exposé Qt; `stack_ram_budget_gb` legacy).
  - UX coexistence wording/backend switch: **BLOCKING** pour Qt-only officiel.
  - Launcher/config backend default Tk: **BLOCKING** pour Qt-only officiel.
  - Worker dependency `lecropper`: **BLOCKING** pour gate S2/S5.
- Decisions:
  - C5.2 (`Confirm no remaining P0/P1 blocker`) = **BLOCKED** actuellement.
- Blockers:
  - P0/P1 restants: fallback Tk launcher, défaut backend `tk`, dépendance worker->`lecropper`, wording/coexistence backend UI.
- Next unchecked item: C5 — Confirm no remaining P0/P1 blocker for official frontend.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged (S1 audit only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:09 — Iteration 22
- Scope: S2/D1→D3 sequential execution (Qt-only official cutover), with gate probes.
- In scope:
  - D1 launcher cleanup (`run_zemosaic.py`)
  - D2 config import safety/default backend
  - D3 Qt UI cleanup wording/backend choice
- Out of scope: broad refactor; `lecropper` port; non-canonical opportunistic changes.
- Files changed:
  - `run_zemosaic.py`
  - `zemosaic_config.py`
  - `zemosaic_gui_qt.py`
  - `zemosaic_filter_gui_qt.py`
  - `followup.md`
- Tests run:
  - `python3 -m py_compile run_zemosaic.py zemosaic_config.py zemosaic_gui_qt.py zemosaic_filter_gui_qt.py`
  - grep probes on Tk fallback/flags/messages
  - `python3 -c "import zemosaic.zemosaic_config as c; ..."` (OK)
  - `python3 -c "import zemosaic.zemosaic_worker as w; ..."` (FAILED in env: missing numpy)
- Proof:
  - D1: launcher no longer instantiates Tk/messagebox/root and no Tk fallback path.
  - D1: `--tk-gui` now explicitly unsupported (Qt-only message).
  - D2: `zemosaic_config.py` no longer imports tkinter at module import; Tk dialogs are lazy via `_ensure_tk_dialogs_loaded()` only in legacy prompt helpers.
  - D2: default backend switched to `preferred_gui_backend = "qt"`.
  - D3: backend choice widget removed from official Qt UI layout path.
  - D3: coexistence wording removed (`Qt GUI (preview)` / `Classic Tk GUI (stable)` removed from active Qt path); filter message now requires PySide6 for official frontend.
- Decisions:
  - C5.2 remained blocked; progression to S2 done by explicit user override.
  - D4 not yet checked: needs explicit decoupling proof around `lecropper` absence on official/headless paths.
- Blockers / incidents:
  - **Environment blocker:** `import zemosaic_worker` gate currently fails here due missing `numpy` in runtime env (`ModuleNotFoundError: No module named 'numpy'`).
  - This blocks full D5 import-gate proof in this environment.
- Next unchecked item:
  - D4 — Remove direct runtime dependency on `lecropper` from official path.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: decreased (launcher/config/Qt UI cutover applied).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:14 — Iteration 23
- Scope: S2/D4 only (`lecropper` decoupling), surgical edits.
- In scope:
  - retirer dépendance runtime directe à `lecropper` sur chemin officiel
  - retirer dépendance indirecte sur chemin headless validé (import path)
- Out of scope:
  - port Qt de `lecropper`
  - refactor worker large
  - clôture D5 tant que l’environnement manque de dépendances
- Files changed:
  - `zemosaic_worker.py`
  - `followup.md`
- Decisions taken:
  - suppression des imports module-level `lecropper` et `from .lecropper import detect_autocrop_rgb`.
  - `lecropper` passe en état annexe non requis côté worker import (`_LECROPPER_AVAILABLE=False` par défaut, `ANCHOR_AUTOCROP_AVAILABLE=False` par défaut).
- Proof collected:
  - grep: plus d’import direct `import lecropper` / `from .lecropper import ...` au module-level.
  - `python3 -m py_compile zemosaic_worker.py` OK (intégrité syntaxique).
- Tests run:
  - `grep -nE "^\s*import lecropper|from \.?lecropper import|..." zemosaic_worker.py`
  - `python3 -m py_compile zemosaic_worker.py`
- Result:
  - D4.1 ✅ fait
  - D4.2 ✅ fait
  - D4.3 ⏳ preuve runtime complète non exécutable ici (voir blocker env)
- Remaining blockers:
  - import gate dynamique `import zemosaic_worker` toujours bloqué par dépendance env manquante (`numpy`), indépendamment de `lecropper`.
- Next unchecked item:
  - D4.3 — Ensure absence of `lecropper` does not break official runtime/headless path.
- Lecropper status changed or not: unchanged as annex/standalone (decoupled from official path).
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged during this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:20 — Iteration 24
- Scope: S2/D4.3 + D5 gate progression (next unchecked items only).
- In scope:
  - prouver l’absence de fallback Tk au lancement officiel
  - valider `import zemosaic_config`
  - avancer la preuve de découplage `lecropper`
- Out of scope:
  - installation de dépendances système (numpy/PySide6)
  - refactor worker additionnel hors besoin de preuve
- Files changed:
  - `zemosaic_gui_qt.py` (message ImportError Qt-only wording correction)
  - `followup.md` (D5.1 + D5.2 cochés)
- Decisions taken:
  - conserver D4.3 non coché tant que preuve headless complète est bloquée par l’environnement.
- Proof collected:
  - `python run_zemosaic.py --tk-gui` => flag refusé + Qt-only message + aucune bascule Tk.
  - `python -c "import zemosaic.zemosaic_config"` => OK (`preferred_gui_backend=qt`).
  - grep officiel (`run_zemosaic.py`, `zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`, `zemosaic_config.py`) => aucune dépendance runtime explicite à `lecropper` sur path officiel.
- Tests run:
  - `python3 -m py_compile zemosaic_gui_qt.py run_zemosaic.py`
  - `python3 run_zemosaic.py --tk-gui`
  - `python3 -c "import zemosaic.zemosaic_config as c; ..."`
  - `python3 -c "import zemosaic.zemosaic_worker as w"` (échec env)
- Result:
  - D5.1 ✅
  - D5.2 ✅
  - D4.3 ⏳ (partiel) / D5.3 ⏳ bloqués par env
- Remaining blockers:
  - `import zemosaic_worker` échoue ici pour cause externe: `ModuleNotFoundError: No module named 'numpy'`.
  - donc preuve dynamique headless complète sans `lecropper` non démontrable dans cet env à ce stade.
- Next unchecked item:
  - D4.3 — Ensure absence of `lecropper` does not break official runtime/headless path.
- Lecropper status changed or not: unchanged (annex), decoupling from official path maintained.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged during this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:23 — Iteration 25
- Scope: finalize remaining S2 unchecked items (D4.3, D5.3, D5.4) and close S2.
- In scope:
  - preuve runtime/headless sans `lecropper`
  - gate import `zemosaic_worker` dans l’environnement `.venv`
- Out of scope:
  - installation système globale
  - refactor packaging/import style du worker
- Files changed:
  - `followup.md`
  - `agent.md`
  - `memory.md`
- Decisions taken:
  - validation gates S2 avec `.venv/bin/python` (env réel de dev indiqué par Tristan).
- Proof collected:
  - `run_zemosaic.py --tk-gui` (venv) => refus explicite Tk + runtime Qt-only.
  - `import zemosaic_config` (venv) => OK (`preferred_gui_backend=qt`).
  - `import zemosaic_worker` (venv, depuis répertoire package `zemosaic/zemosaic`) => OK.
  - test anti-lecropper (meta_path bloque `lecropper` et `zemosaic.lecropper`) :
    - `import zemosaic_worker` => OK
    - `import run_zemosaic` => OK
- Tests run:
  - `/home/tristan/zemosaic/.venv/bin/python /home/tristan/zemosaic/zemosaic/run_zemosaic.py --tk-gui`
  - `/home/tristan/zemosaic/.venv/bin/python -c "import zemosaic.zemosaic_config ..."`
  - `/home/tristan/zemosaic/.venv/bin/python -c "import zemosaic_worker ..."` (workdir package)
  - two `meta_path` blocker scripts for `lecropper`
- Result:
  - D4.3 ✅
  - D5.3 ✅
  - D5.4 ✅
  - **S2 closed**
- Remaining blockers:
  - aucun blocker S2.
  - note technique hors-scope immédiat: `import zemosaic.zemosaic_worker` depuis repo root échoue à cause d’un import absolu `zemosaic_resource_telemetry`; gate validé via chemin headless défini en S0/B2.
- Next unchecked item:
  - E1 — Migrate `preferred_gui_backend=tk` to `qt`.
- Lecropper status changed or not: unchanged (annex), runtime decoupling now proven.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged during this iteration (proof-only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:30 — Iteration 26
- Scope: S3/E1 only — migrate `preferred_gui_backend=tk` to `qt`.
- In scope: config migration logic at load-time, surgical.
- Out of scope: E1.2/E1.3, fixtures E2, cleanup E3.
- Files changed:
  - `zemosaic_config.py`
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - Add explicit load-time migration: legacy `preferred_gui_backend == "tk"` is normalized to `"qt"`.
- Proof collected:
  - Temp legacy config fixture (`preferred_gui_backend: tk`) loaded through monkeypatched `get_config_path` returns `preferred_gui_backend: qt`.
- Tests run:
  - `/home/tristan/zemosaic/.venv/bin/python` script with temporary config file + `zemosaic_config.load_config()`
  - output: `MIGRATED_BACKEND qt`
- Result:
  - E1.1 ✅ done.
- Remaining blockers:
  - none for E1.2 next.
- Next unchecked item:
  - E1 — Neutralize obsolete backend selection state if needed.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged in this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:31 — Iteration 27
- Scope: S3/E1.2 + E1.3 (next unchecked items), surgical config migration hardening.
- In scope:
  - neutralize obsolete backend-selection state
  - preserve backward readability for legacy config files
- Out of scope:
  - E2 fixtures set
  - broader config refactor
- Files changed:
  - `zemosaic_config.py`
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - `load_config()` now forces neutral Qt-only state:
    - `preferred_gui_backend = "qt"`
    - `preferred_gui_backend_explicit = False`
  - `save_config()` also enforces same neutral state to prevent stale reactivation.
- Proof collected:
  - Legacy fixture with `{preferred_gui_backend: "tk", preferred_gui_backend_explicit: true}`:
    - load => `BACKEND qt EXPLICIT False`
    - save => file persists `SAVED_BACKEND qt SAVED_EXPLICIT False`
- Tests run:
  - `.venv/bin/python` monkeypatch `get_config_path` + temp config round-trip
- Result:
  - E1.2 ✅
  - E1.3 ✅
- Remaining blockers:
  - none for E2 start.
- Next unchecked item:
  - E2 — Create/collect minimal legacy config fixtures.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:32 — Iteration 28
- Scope: continue S3 sequentially (E2.1→E4), stop only on blocker.
- In scope:
  - create minimal legacy config fixtures
  - prove load→save→load idempotence
  - prove no silent Tk reactivation
  - remove remaining active official coexistence branches
  - close S3 when proven
- Out of scope:
  - S4 packaging/doc release edits
  - broad refactor
- Files changed:
  - `tests/fixtures/config_migration/legacy_backend_tk_minimal.json`
  - `tests/fixtures/config_migration/legacy_backend_weird_value.json`
  - `tests/fixtures/config_migration/already_qt_explicit_true.json`
  - `zemosaic_config.py`
  - `run_zemosaic.py`
  - `zemosaic_gui_qt.py`
  - `followup.md`
  - `agent.md`
- Decisions taken:
  - Fixture set for migration safety kept minimal (3 representative cases).
  - Official coexistence cleanup narrowed to active branches only:
    - removed dead backend-selection handlers/groups in Qt GUI
    - simplified launcher backend normalization to Qt-only flag handling (no env backend selection path)
- Proof collected:
  - Round-trip script over fixtures (`tests/fixtures/config_migration/*.json`) reports all idempotent=true.
  - All fixtures end with `preferred_gui_backend=qt` and `preferred_gui_backend_explicit=false`.
  - py_compile passes for updated files.
  - grep confirms backend coexistence methods removed from active Qt GUI path.
- Tests run:
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic /home/tristan/zemosaic/.venv/bin/python /tmp/zemosaic_roundtrip_check.py`
  - `python3 -m py_compile run_zemosaic.py zemosaic_gui_qt.py zemosaic_config.py`
  - grep probes on removed coexistence symbols and launcher backend controls.
- Result:
  - E2.1 ✅
  - E2.2 ✅
  - E2.3 ✅
  - E3.1 ✅
  - E3.2 ✅
  - E4.1 ✅
  - E4.2 ✅
  - **S3 closed**
- Remaining blockers:
  - none in S3.
- Next unchecked item:
  - F1 — Audit official build/spec scripts.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: decreased (coexistence branches further removed).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:35 — Iteration 29
- Scope: S4/F1 sequential until blocker.
- In scope:
  - audit build/spec scripts
  - remove packaging hints suggesting Tk coexistence
  - verify final built artifacts if build tool available
- Out of scope:
  - F2 docs/release notes content rewrite
- Files changed:
  - `ZeMosaic.spec`
  - `requirements.txt`
  - `requirements_no_gpu.txt`
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - remove Tk matplotlib backend hiddenimports from spec (`backend_tkagg`, `_backend_tk`, `_tkagg`).
  - remove `lecropper` hiddenimport from official spec bundle list.
  - remove Tk install notes from requirements files; keep PySide6 as required Qt-only frontend dependency.
- Proof collected:
  - grep on spec/requirements confirms no remaining Tk packaging hints and no Tk backend hiddenimports.
- Tests run:
  - `find ...` audit packaging files
  - grep probes on spec/requirements
  - build-attempt probes:
    - `.venv/bin/python -m PyInstaller --version` => module missing
- Result:
  - F1.1 ✅
  - F1.3 ✅
  - F1.2 ⛔ blocked (cannot verify final built artifacts without PyInstaller installed in `.venv`).
- Remaining blockers:
  - Missing build dependency in `.venv`: `PyInstaller`.
- Next unchecked item:
  - F1.2 — Verify final built artifacts, not only source scripts.
- Lecropper status changed or not: unchanged (annex); no longer hiddenimported in official spec.
- Official-path Tk imports decreased or stayed unchanged: decreased for packaging surface.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:38 — Iteration 30
- Scope: S4/F1.2 only (verify final built artifacts).
- In scope:
  - build PyInstaller artifact from updated spec
  - inspect produced dist/build outputs for Qt-only packaging consistency
- Out of scope:
  - full docs updates (F2)
  - release-note authoring (F3)
- Files changed:
  - `followup.md`
  - `memory.md`
- Decisions taken:
  - artifact verification performed directly on generated `dist/ZeMosaic` output.
- Proof collected:
  - PyInstaller version in `.venv`: `6.19.0`.
  - Build completed successfully; output at `dist/ZeMosaic`.
  - Final artifact exists: `dist/ZeMosaic/ZeMosaic` + `_internal` payload.
  - `warn-ZeMosaic.txt` reviewed.
- Tests run:
  - `/home/tristan/zemosaic/.venv/bin/python -m PyInstaller --version`
  - `python -m PyInstaller --noconfirm --clean ZeMosaic.spec`
  - `ls dist/ZeMosaic`, `ls dist/ZeMosaic/_internal`
  - `sed -n '1,220p' build/ZeMosaic/warn-ZeMosaic.txt`
- Result:
  - F1.2 ✅ done.
- Remaining blockers:
  - none for F1 completion.
  - note (non-blocking for current mission step): PyInstaller still auto-pulls some tkinter hooks transitively via dependency graph; runtime launch remains Qt-only and `--tk-gui` unsupported.
- Next unchecked item:
  - F2 — Update user docs / README / quickstart / troubleshooting.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration (artifact validation only).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:41 — Iteration 31
- Scope: S4/F2→F5 sequential completion (docs/release/version semantics/proof).
- In scope:
  - update user-facing docs for Qt-only official runtime
  - update dev/build notes reflecting Qt-only packaging
  - publish annex status for `lecropper`
  - update release notes with migration + unsupported legacy status
  - record version semantics note
- Out of scope:
  - S5 QA/CI execution
  - repo-wide Tk annex purge (S6)
- Files changed:
  - `README.md`
  - `RELEASE_NOTES.md`
  - `followup.md`
  - `agent.md`
  - `memory.md`
- Decisions taken:
  - release-line semantics: 4.4.1 docs/release notes explicitly treat Qt-only official frontend as normative behavior; legacy Tk path marked non-official.
- Proof collected:
  - README now states official frontend is PySide6/Qt and removes active Tk coexistence instructions in quickstart/build sections.
  - README includes explicit annex status note for `lecropper`.
  - RELEASE_NOTES now explicitly covers:
    - Qt-only official frontend
    - no Tk fallback on official startup
    - config migration (`preferred_gui_backend=tk` -> `qt` + explicit flag neutralization)
    - unsupported/legacy statement for Tk frontend
    - `lecropper` annex status
- Tests run:
  - grep checks on README/RELEASE_NOTES for residual Tk-coexistence guidance.
- Result:
  - F2.1 ✅
  - F2.2 ✅
  - F2.3 ✅
  - F3.1 ✅
  - F3.2 ✅
  - F3.3 ✅
  - F3.4 ✅
  - F4.1 ✅
  - F4.2 ✅
  - F5 ✅
  - **S4 closed**
- Remaining blockers:
  - none in S4.
- Next unchecked item:
  - G1 — Windows smoke test.
- Lecropper status changed or not: unchanged (annex/standalone), publicly clarified.
- Official-path Tk imports decreased or stayed unchanged: decreased on docs/packaging communication surface.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:47 — Iteration 32
- Scope: S5/G1 per agreed modality (launch + run start + early interruption), continue until blocker.
- In scope:
  - execute Linux smoke test with example set (intermediate checkpoint)
  - create explicit Windows/macOS smoke protocol note for Tristan to run locally
- Out of scope:
  - full 35-minute completion runs
  - Windows/macOS execution from this Linux host
- Files changed:
  - `tests/SMOKE_PROTOCOL_Windows_macOS.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - worker-process smoke script with `example/organized/EQ/IRCUT` and output `example/out/smoke_linux_20260313_b`
  - process started, emitted runtime messages, then terminated intentionally for checkpoint stop test.
- Proof collected (Linux):
  - process alive before stop: `True`
  - event stream observed (15 messages), including Grid mode activation and iterative processing warnings (path issues from dataset stack_plan), confirming run start/activity.
  - manual interruption path exercised via process terminate (checkpoint equivalent to early stop).
- Decisions taken:
  - G1 Linux smoke marked complete for intermediate checkpoint criteria.
  - Windows/macOS smoke delegated with documented protocol (`tests/SMOKE_PROTOCOL_Windows_macOS.md`).
- Remaining blockers:
  - G1 Windows/macOS execution blocked on host availability (to be run by Tristan manually).
- Next unchecked item:
  - G1 — Windows smoke test.
- Lecropper status changed or not: unchanged.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 10:52 — Iteration 33
- Scope: Continue S5 sequentially after G1 hold, until blocking item.
- In scope:
  - mark G1 Windows/macOS on hold by explicit user decision
  - execute G2 dependency-failure tests
  - execute G3 headless tests
  - add CI hardening job for G4
  - progress G5 checklist until first blocker
- Out of scope:
  - running Windows/macOS smoke locally
  - forcing CI run status from local host
- Files changed:
  - `followup.md`
  - `.github/workflows/no-tk-on-official-path.yml`
  - `tests/SMOKE_PROTOCOL_Windows_macOS.md` (from previous step, still reference)
  - `memory.md`
- Tests run:
  - G2 / PySide6 absent: meta_path block -> launcher exits with clear Qt error, no Tk fallback.
  - G2 / worker import failure: meta_path block -> launcher exits error, no Tk fallback.
  - G2 / startup path with `--tk-gui`: explicit refusal + Qt path used.
  - G3 / `import zemosaic_config` OK (`qt`, explicit False).
  - G3 / config load-save-load idempotent OK.
  - G3 / `import zemosaic_worker` OK with `lecropper` blocked.
- CI hardening:
  - added `.github/workflows/no-tk-on-official-path.yml` with:
    - official-path Tk grep guard
    - headless import gates (`zemosaic_config`, `zemosaic_worker`)
    - `--tk-gui` non-fallback behavior check.
- Result:
  - G2 ✅ complete
  - G3 ✅ complete
  - G4 ✅ complete (job defined)
  - G5 partially advanced; all items checked except CI green.
- Blocker:
  - **G5: `CI is green` remains unchecked** (requires remote CI run/result, not provable locally in this step).
- Next unchecked item:
  - G5 — CI is green.
- Lecropper status changed or not: unchanged (annex), no official runtime dependency.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration (validation + CI guard addition).
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 11:11 — Iteration 34
- Scope: close remaining S5 blocker after user CI feedback.
- In scope:
  - validate G5 `CI is green` from user-provided workflow result
  - close G6 (GO/NO-GO)
- Out of scope:
  - Node 20 deprecation remediation (non-blocking infra warning)
- Files changed:
  - `followup.md`
  - `agent.md`
  - `memory.md`
- Proof collected:
  - User confirms `guard` workflow is GREEN.
  - Warning is non-blocking deprecation notice for Node 20 actions (`actions/checkout@v4`, `actions/setup-python@v5`).
- Decisions:
  - Release status for this migration checkpoint: **GO**.
  - Node 20 warning tracked as follow-up infra task; does not block Qt-only/Tk-retirement acceptance.
- Remaining blockers:
  - none for S5 closeout.
  - Windows/macOS smoke tests remain on hold by user decision.
- Next unchecked item:
  - S6 (later mission only; do not start unless requested).
- Lecropper status changed or not: unchanged (annex), decoupled from official runtime/headless validated paths.
- Official-path Tk imports decreased or stayed unchanged: stayed unchanged this iteration.
- Validated headless scope changed or stayed unchanged: stayed unchanged.


### 2026-03-13 12:00 — Iteration 35
- Scope: mission handoff and plan reset only.
- In scope:
  - explicitly close the previous active mission as completed
  - replace `agent.md` and `followup.md` with a new mission plan for Phase 3 adaptive RAM control
  - lock the new scientific invariant for the next mission direction
- Out of scope:
  - Phase 3 code changes
  - scheduler changes
  - telemetry changes
  - tests beyond document review
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `sed -n '1,260p' agent.md`
  - `sed -n '1,320p' followup.md`
  - `tail -n 80 memory.md`
- Proof:
  - Previous mission status was already recorded as **GO** in Iteration 34.
  - `agent.md` was replaced with a new mission focused on Phase 3 adaptive RAM control without dropping raw frames.
  - `followup.md` was replaced with a new checklist aligned to that mission.
- Decisions:
  - Previous Qt-only/Tk-retirement mission is considered completed for this checkpoint.
  - New active mission is: **Phase 3 adaptive RAM control without dropping raw frames**.
  - The new mission invariant is locked at plan level: adapt memory via launch control and working-set reduction, not by silently discarding raw inputs.
- Blockers:
  - none at planning stage.
- Next unchecked item:
  - B1 — Write the invariant explicitly and audit the current Phase 2/3 behavior against it.
- All-raw-frames invariant changed or stayed unchanged: changed (now explicitly locked as the central mission invariant).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (planning only).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (planning only).


### 2026-03-13 17:19 — Iteration 36
- Scope: Documentation hardening of mission files to improve clarity and prevent regression.
- In scope:
  - strengthen invariant as testable contract in `agent.md`
  - add explicit exactness policy matrix per combine/rejection mode in `agent.md`
  - add anti-oscillation controller guardrails in `followup.md`
  - add critical non-regression gates in `followup.md`
- Out of scope:
  - runtime code changes
  - scheduler implementation changes
  - telemetry implementation changes
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - static content verification by direct file edit and section-presence checks
- Proof:
  - mission docs now include explicit equality invariants for logical membership
  - mission docs now include mode exactness policy table and no-approximation-by-default rule
  - checklist now includes explicit hysteresis/cooldown/rate-limit constraints
  - checklist now includes critical non-regression gates prior to mission gate
- Decisions:
  - documentation-only hardening accepted
- Blockers:
  - none
- Next unchecked item:
  - B1. Scientific invariant (execution checklist remains open; no mission implementation performed)
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (clarified and made testable).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (docs only).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (docs only).


### 2026-03-13 17:22 — Iteration 37
- Scope: Mission-document precision pass for operational clarity and regression prevention.
- In scope:
  - add numeric default controller profile in `agent.md`
  - add explicit RAM signal source-of-truth policy in `agent.md`
  - add ordered degradation pseudocode and extreme-pressure behavior in `agent.md`
  - add mission glossary in `agent.md`
  - refine checklist wording in `followup.md` (scientific outputs scope, anti-thrashing criterion, minimum-limit pressure behavior, RAM signal definition)
- Out of scope:
  - runtime code changes
  - scheduler implementation changes
  - telemetry implementation changes
  - test execution
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - static mission-file consistency review after edits
- Proof:
  - mission docs now include explicit default thresholds and cooldown/rate constraints
  - mission docs now include single RAM decision surface requirement
  - mission docs now include mandatory degradation order pseudocode and extreme-pressure fallback
  - mission docs now include glossary and clarified non-regression output scope
- Decisions:
  - documentation hardening accepted; implementation scope intentionally unchanged
- Blockers:
  - none
- Next unchecked item:
  - B1. Scientific invariant (execution checklist still open; no implementation performed)
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (further clarified).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (docs only).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (docs only).


### 2026-03-13 17:26 — Iteration 38
- Scope: Terminology harmonization pass across mission documents (no behavior change).
- In scope:
  - align `followup.md` wording with `agent.md` glossary (`logical stack`, `logical stack membership`)
  - align `agent.md` phrasing for preservation target terminology
- Out of scope:
  - runtime code changes
  - scheduler/controller logic changes
  - test execution
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - static textual consistency check by direct diff review
- Proof:
  - mission files now use consistent glossary-aligned terms for membership invariants
- Decisions:
  - keep semantics unchanged; wording-only normalization
- Blockers:
  - none
- Next unchecked item:
  - B1. Scientific invariant
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (wording harmonized only).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 17:46 — Iteration 39
- Scope: S0 baseline audit and invariant lock (no runtime behavior change).
- In scope:
  - baseline audit of Phase 2 splitting behavior in active worker path
  - baseline audit of Phase 3 launch/concurrency/memory levers
  - explicit acceptable vs non-acceptable splitting criteria for this mission
  - lock invariant status in mission tracking checklists
- Out of scope:
  - runtime code changes
  - scheduler refactor
  - telemetry implementation changes
  - test execution on datasets
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - static code-path audit with file/line references
- Proof:
  - Invariant explicitly locked at mission level (agent): logical stack preserved; adaptation constrained to working-set/parallelism.
  - Phase 2 baseline (active run path `run_hierarchical_mosaic`):
    - preplan strict mapping enabled (`zemosaic_worker.py:25103-25142`), prevents runtime auto-limit split (`25632`).
    - RAM-budget recluster/split path exists (`25178`, helper `10675`).
    - auto-split helper path exists (`25484`, helper `10547`).
    - manual cap split exists (`25509-25517`).
    - overlap batching can duplicate membership across batches (`25532-25555`, helper `10447`) and is therefore not acceptable under strict per-logical-stack equality invariant.
  - Phase 3 baseline (active run path):
    - worker count derived then boosted/capped by CPU+IO+RAM heuristics (`26379-26479`).
    - runtime monitor adapts IO/CPU-driven limits but not RAM-driven limits (`26503+`).
    - monitor currently swaps semaphore objects at runtime (`26575`) while tasks use global semaphore acquire/release in `create_master_tile` (`13417`, `13464`, `13778`, `14510`), which is concurrency-fragile.
    - eager submission still present: all initial groups submitted before completion loop (`26761` then `26770`).
    - retries are immediately submitted and increase total work (`26850+`).
  - Per-tile memory levers baseline:
    - `winsor_max_frames_per_pass` propagated from config (`25581`) and used in stackers (`zemosaic_align_stack.py:1728+`).
    - preemptive streaming limit in CPU stack path (`zemosaic_worker.py:12802-12815`).
    - winsor memory plan (stream/memmap/incremental fallback) (`zemosaic_align_stack.py:1778-1849`, `1929`, `2110+`).
    - row/chunk sizing from parallel plan hints (`zemosaic_align_stack.py:237-266`, `966`, `1015`, `1943-1955`).
    - GPU budget interaction via Phase 3 GPU retry+plan shrink (`zemosaic_worker.py:12928-13051`).
- Decisions:
  - Acceptable for this mission: splitting/partitioning that preserves full raw-frame union without overlap duplication.
  - Not acceptable for invariant-gated mode: overlap batching that duplicates frame membership across logical stacks.
  - S0 considered complete: invariant + current gaps are explicit and documented.
- Blockers:
  - none for S0 closeout.
- Next unchecked item:
  - C1. Controller policy (S1 design).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (now baseline-audited and explicitly gated).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (audit only).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (audit only).


### 2026-03-13 17:54 — Iteration 40
- Scope: S1 adaptive strategy design sign-off (design-only, explicit decision tree).
- In scope:
  - controller policy around 80% RAM target
  - hysteresis/cooldown/rate-limit policy
  - lazy scheduler and retry queue semantics
  - scientific preservation policy by combine/rejection mode
  - telemetry/log contract for adaptation explainability
- Out of scope:
  - runtime code changes
  - tests on datasets
  - implementation of controller loop
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - design consistency check against S0 baseline findings
- Proof (target design):
  - Controller signal/policy:
    - primary decision signal: system RAM `used_percent`
    - supporting signals: available bytes, swap activity
    - target zone around 80%; thresholds: high=82%, low=72%
    - minimum cooldown between level changes: 10s
    - rate-limit: max 6 adaptation-level changes/min
  - Escalation order (mandatory):
    - L1: reduce active Phase 3 launch budget (future launches only)
    - L2: reduce per-pass frame budget (`winsor_max_frames_per_pass`)
    - L3: reduce row/chunk size (parallel-plan row/chunk hints)
    - L4: serialize special paths (GPU or specific reject/combine path) only if prior levels insufficient
  - Recovery order: inverse and gradual (L4→L1) once below low threshold and cooldown elapsed.
  - Scheduler model:
    - pending queue + lazy dispatcher; never eager-submit all groups
    - dispatcher invariant: `active_futures <= launch_budget` for new admissions
    - retries re-enter queue (no bypass), preserving launch-budget control
  - Scientific preservation policy:
    - mean / weighted mean: frame-pass and row-chunk adaptation allowed only with full logical membership preserved
    - median: no approximation by default; do not enable frame-dropping/truncation adaptation
    - winsorized sigma clip: adaptation allowed only via exact full-membership paths (stream/memmap/chunk) using all logical inputs
    - other rejection modes: conservative path unless exactness is validated explicitly
  - Exactness limitations (explicit):
    - no approximation-enabled mode is permitted by default
    - any mode with unproven exactness under adaptation stays on conservative behavior until proven
  - Telemetry/log contract:
    - required adaptation event fields: `phase`, `action`, `reason`, `ram_used_pct`, `ram_available_mb`, `swap_used_pct`, `launch_budget_before/after`, `frames_per_pass_before/after`, `rows_chunk_before/after`, `pending_queue_len`, `active_futures`, `cooldown_state`
    - emit on every adaptation decision and every recovery step
- Decisions:
  - S1 design is explicit and approved as implementation target for S2→S4.
  - next implementation step is S2 scheduler refactor (lazy launch + launch budget enforcement).
- Blockers:
  - none for S1 closeout.
- Next unchecked item:
  - D1. Replace eager launch.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (design constraints reinforced).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (design-only this iteration).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (design-only this iteration).


### 2026-03-13 18:09 — Iteration 41
- Scope: S2 scheduler refactor (lazy Phase 3 admission + runtime launch budget control).
- In scope:
  - replace eager Phase 3 all-at-once submission with lazy queue-based admission
  - enforce runtime-controllable launch budget on future admissions
  - route retry groups back through same admission queue
  - keep progress/ETA/telemetry flow semantics as close as possible
- Out of scope:
  - RAM-based controller redesign (S4)
  - per-tile frames/chunk dynamic tuning (S3)
  - scientific algorithm changes
- Files changed:
  - `zemosaic_worker.py`
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_align_stack.py`
  - static pattern verification (lazy queue + runtime launch budget markers)
- Proof:
  - Eager submission removed; Phase 3 now builds `pending_launch_queue` and dispatches lazily (`zemosaic_worker.py:26792+`, `26805+`, `26819+`).
  - Admission invariant enforced for new launches: dispatcher only launches while `len(pending_futures) < launch_budget` (`26808-26812`).
  - Runtime launch budget introduced as shared state (`runtime_launch_limit`, `26529`) and updated by monitor (`26600-26605`), with explicit adaptation logs (`26614-26616`).
  - Retries re-enter queue instead of immediate direct submit (`26910-26913`), so retries no longer bypass launch control.
  - Loop condition now drains both active and pending work (`26819`), preserving completion semantics with lazy admission.
  - Removed runtime semaphore object hot-swapping for Phase 3 worker throttle path in this section; launch-budget control now drives admission decisions directly.
- Decisions:
  - S2 accepted: launch throttling is now real for not-yet-started tasks (lazy admission + mutable launch budget).
  - Keep cache-read semaphore adaptation as-is for now; deeper controller unification deferred to S4.
- Blockers:
  - none for S2 closeout.
- Next unchecked item:
  - E1. Frame-pass adaptation (S3).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (scheduler-only changes).
- Phase 3 launch control changed or stayed unchanged: changed (lazy admission + runtime launch budget).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (deferred to S3/S4).


### 2026-03-13 18:27 — Iteration 42
- Scope: S3 partial implementation (E1/E2) — per-tile adaptive working-set sizing in Phase 3.
- In scope:
  - add per-tile dynamic `winsor_max_frames_per_pass` computation under RAM pressure
  - add per-tile dynamic row/chunk downsizing via `ParallelPlan` replacement under RAM pressure
  - wire adaptive values into `_stack_master_tile_auto` call path
- Out of scope:
  - runtime RAM controller hysteresis loop redesign (S4)
  - full exactness validation by reject/combine mode (E3)
  - test-suite additions (S5)
- Files changed:
  - `zemosaic_worker.py`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_align_stack.py`
  - static grep verification of adaptive symbols/log wiring (`P3_MEM_ADAPT_TILE`, adaptive pass/chunk args)
- Proof:
  - per-tile adaptive pass sizing added before stack call (`zemosaic_worker.py:13692+`), based on live RAM percent/available bytes and tile frame memory estimate.
  - per-tile adaptive chunk sizing added via `replace(parallel_plan, ...)` with CPU+GPU rows/chunk tightening under pressure (`13666+`).
  - stack invocation now consumes adaptive values (`13816`, `13819`) rather than static inputs.
  - no frame list truncation or membership filtering introduced in adaptation block; adaptation only changes working-set knobs.
  - structured adaptation log emitted per tile under pressure (`P3_MEM_ADAPT_TILE`, `13782+`).
- Decisions:
  - E1/E2 marked done; E3/E4 remain open pending exactness review and broader proof.
- Blockers:
  - none for E1/E2 closeout.
- Next unchecked item:
  - E3. Exactness review by combine/rejection mode under adaptation.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (no membership-trimming logic added).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged this iteration (S2 already completed).
- Working-set adaptation changed or stayed unchanged: changed (per-tile pass/chunk adaptive controls added).


### 2026-03-13 18:37 — Iteration 43
- Scope: S3 closeout (E3/E4) — exactness policy gating + proof of logical membership preservation.
- In scope:
  - enforce conservative mode gating for per-tile adaptation according to exactness confidence
  - document/emit limitation when mode exactness is not validated
  - record proof of no logical-frame-membership mutation by adaptation block
- Out of scope:
  - S4 runtime controller redesign
  - S5 non-regression test-suite creation
- Files changed:
  - `zemosaic_worker.py`
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_align_stack.py`
  - static grep verification for exactness gating/log fields (`pass_adapt_enabled`, `chunk_adapt_enabled`, `P3_MEM_ADAPT_MODE_CONSERVATIVE`)
- Proof:
  - exactness/conservatism policy now explicit in code at tile adaptation point:
    - pass adaptation enabled only for `winsorized_sigma_clip` (`zemosaic_worker.py:13709`)
    - chunk adaptation enabled only for `{winsorized_sigma_clip, kappa_sigma, linear_fit_clip}` (`13710`)
    - unsupported/unvalidated mode combinations emit conservative-policy log (`P3_MEM_ADAPT_MODE_CONSERVATIVE`, `13811+`)
  - adaptation telemetry now includes `reject_algo`, `combine_mode`, and enabled flags (`13797-13800`) to make exactness decisions auditable.
  - logical membership preservation: adaptation block does not trim/reorder/drop `valid_aligned_images`; it only adjusts pass/chunk knobs passed into `_stack_master_tile_auto` (`winsor_max_frames_per_pass` and `parallel_plan`).
- Exactness review outcome (current scope):
  - validated/adapted:
    - winsorized sigma clip: pass + chunk adaptation
    - kappa sigma / linear fit clip: chunk adaptation only
  - conservative (adaptation limited until validated):
    - other/unlisted rejection/combine paths, including median-centric paths not explicitly validated for adaptive knobs
- Decisions:
  - S3 marked complete with conservative exactness gating and documented limitations.
- Blockers:
  - none for S3 closeout.
- Next unchecked item:
  - F1. Runtime RAM sampling (S4).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (no frame-membership mutation).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged this iteration (S2 behavior retained).
- Working-set adaptation changed or stayed unchanged: changed (exactness-gated adaptation behavior finalized for S3).


### 2026-03-13 18:49 — Iteration 44
- Scope: S4 implementation/closeout — runtime RAM controller + hysteresis + observability for Phase 3 admission.
- In scope:
  - extend real-time Phase 3 monitor with RAM sampling and RAM-level state machine
  - add hysteresis thresholds, cooldown, and max changes/min anti-thrashing controls
  - implement explicit admission pause behavior at critical RAM pressure with alert log
  - expose adaptation decision context in runtime logs (RAM/CPU/IO + reasons)
  - make launch budget consumer pause-aware (budget can be 0 during short admission pause)
- Out of scope:
  - S5 test-suite implementation
  - deep redesign of GPU stack internals
- Files changed:
  - `zemosaic_worker.py`
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_align_stack.py`
  - static grep verification of S4 knobs/signals/logs (`phase3_ram_low_pct`, `phase3_adapt_cooldown_s`, `phase3_adapt_max_changes_per_min`, `phase3_admission_pause_s`, `RAM_ADAPT_RT`, `pause_until`)
- Proof:
  - runtime controller now samples RAM every monitor cycle and tracks `ram_level` with hysteresis (`high/low/critical`).
  - anti-oscillation controls implemented:
    - cooldown between budget changes (`phase3_adapt_cooldown_s`, default 10s)
    - rate-limit (`phase3_adapt_max_changes_per_min`, default 6/min)
  - critical-pressure behavior implemented:
    - when RAM stays critical and budget already minimal, controller sets short admission pause (`pause_until`) and emits explicit `RAM_ADAPT_RT: admission_pause ...` log.
  - launch admission path now honors pause window:
    - `_current_launch_budget()` can return `0` while pause is active
    - lazy dispatcher skips new submissions during pause, preserving in-flight work safety
  - decision observability improved:
    - budget-change logs include CPU/IO/RAM context and RAM-level transitions
    - hold logs emitted when change blocked by cooldown/rate-limit (`RAM_ADAPT_RT: launch_budget_hold ...`)
  - GPU policy decision:
    - keep existing narrow Phase 3 GPU guardrails (retry+plan-shrink+hard-disable path) without adding a second independent GPU-concurrency subsystem.
- Decisions:
  - S4 marked complete: runtime decisions are now pressure-driven, bounded, and explainable in logs.
- Blockers:
  - none for S4 closeout.
- Next unchecked item:
  - G1. Invariant tests (S5).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (controller only impacts admission/working-set knobs).
- Phase 3 launch control changed or stayed unchanged: changed (pause-aware launch budget with hysteresis/rate-limited updates).
- Working-set adaptation changed or stayed unchanged: stayed unchanged this iteration (S3 behavior retained; S4 controls admission dynamics).


### 2026-03-13 18:18 — Iteration 45
- Scope: S5 implementation pass (G1/G2/G3) — add and run non-regression tests for Phase 3 adaptive behavior.
- In scope:
  - strengthen `tests/test_phase3_adaptive_invariants.py` with invariant + pressure-response + compatibility coverage
  - execute targeted and full test suite in `.venv`
  - update `followup.md` checkboxes for completed S5 sub-items
- Out of scope:
  - dataset-level baseline/output-count integration run (G3b-1)
  - explicit stable-pressure anti-thrashing scenario test harness (G3b-5)
  - final mission closeout decision (G5)
- Files changed:
  - `tests/test_phase3_adaptive_invariants.py`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py` → `7 passed`
  - `../.venv/bin/python -m pytest -q tests` → `29 passed, 7 warnings`
- Proof:
  - G1 covered by numeric invariance tests + source-contract guard ensuring adaptation block does not mutate logical frame membership list.
  - G2 covered by source-contract tests asserting runtime RAM backoff/hysteresis/cooldown/rate-limit/recovery markers and pass/chunk shrink contracts.
  - G3 covered by tests for retry queue re-entry (no bypass direct submit), adaptation telemetry key presence/stability, and mode-gating markers (conservative policy for unvalidated modes).
  - All added checks are green in local `.venv` test run.
- Decisions:
  - Marked G1, G2, G3 as complete in `followup.md`.
  - Marked G3b retry-membership + log-parseability items complete; kept baseline-output-count and anti-thrashing scenario items open pending dedicated harness/dataset run.
- Blockers:
  - No fast deterministic in-repo fixture currently encodes a full baseline-vs-adaptive Phase 3 output-count integration comparison at mission level.
- Next unchecked item:
  - G3b: "Same number of scientific Phase 3 outputs as baseline for identical dataset/config".
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (tests/documentation only, no runtime algorithm mutation this iteration).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (validation iteration only).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (validation iteration only).


### 2026-03-13 18:21 — Iteration 46
- Scope: S5 deeper validation pass — add dynamic Phase 3 tile test under forced RAM pressure.
- In scope:
  - add integration-style unit test exercising `create_master_tile` with synthetic cached frames
  - assert full logical membership reaches stack call under adaptation pressure
  - assert reject/combine mode passthrough is unchanged under adaptation
  - rerun targeted + full tests
- Out of scope:
  - full mission-level scheduler baseline-vs-adaptive output-count integration benchmark
  - direct runtime anti-thrashing scenario replay of nested monitor loop
- Files changed:
  - `tests/test_phase3_adaptive_invariants.py`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k create_master_tile_adaptation_preserves_membership_and_modes`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py` → `8 passed`
  - `../.venv/bin/python -m pytest -q tests` → `30 passed, 14 warnings`
- Proof:
  - New test forces high RAM pressure (`psutil.virtual_memory` patched) and checks `create_master_tile` forwards all loaded frames (`captured["n_images"] == len(group)`) to `_stack_master_tile_auto` (no silent membership drop in adaptation path).
  - Test confirms no silent mode drift: `stack_reject_algo` and `stack_final_combine` are forwarded unchanged under adaptation.
  - Test confirms working-set knobs tighten under pressure (`winsor_max_frames_per_pass`, `rows_per_chunk`, `max_chunk_bytes`) while logical membership remains intact.
- Decisions:
  - Marked G3b item "No silent change of combine/rejection mode" as complete.
  - Kept G3b output-count parity and anti-thrashing scenario items open pending dedicated harness/scenario-run coverage.
- Blockers:
  - Current unit tests do not yet execute the full Phase 3 scheduler loop end-to-end with baseline/adaptive parity accounting on identical dataset/config.
- Next unchecked item:
  - G3b: "Same number of scientific Phase 3 outputs as baseline for identical dataset/config".
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (validation-only changes).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged (validation-only changes).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (validation-only changes).


### 2026-03-13 18:31 — Iteration 47
- Scope: Post-abort integrity verification + S5 closeout (G3b/G4/G5) for Phase 3 adaptive mission.
- In scope:
  - verify interruption point after anti-thrashing helper refactor
  - check for code-loss/regression risk (compile + targeted + full tests)
  - complete remaining S5 gates in `agent.md` / `followup.md` based on evidence
- Out of scope:
  - new algorithmic behavior changes beyond no-op guard refactor already in progress
  - optional S6 refinements
- Files changed:
  - `zemosaic_worker.py` (already modified before this resume; helper retained)
  - `tests/test_phase3_adaptive_invariants.py`
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_align_stack.py tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py` → `12 passed`
  - `../.venv/bin/python -m pytest -q tests` → `34 passed, 21 warnings`
- Proof:
  - Prior interruption root cause was not functional failure but run abort during full-suite rerun; rerun now completed successfully.
  - Anti-thrashing guard refactor is structurally consistent:
    - helper defined once: `_phase3_budget_change_block_reason(...)`
    - called in both Phase 3 monitor loops (2 call sites)
    - centralizes cooldown + changes/min gating without modifying decision policy semantics
  - Retry scheduling invariant remains preserved (`pending_launch_queue.append(...)` present; forbidden direct retry submit absent).
  - Runtime admission pause behavior remains intact (`pause_until` branch returns launch budget 0 during pause window).
  - Added/kept S5 tests cover:
    - logical membership preservation under adaptation
    - output-count parity baseline vs pressure (master-tile level synthetic run)
    - no silent reject/combine mode drift
    - anti-thrashing guard behavior (cooldown/rate-limit)
    - telemetry/log marker stability
- Decisions:
  - S5 marked complete in `agent.md`.
  - Remaining `followup.md` S5 checkboxes (G3b, G4, G5) marked complete.
  - Mission status for merge gate: **GO** (based on current automated evidence).
- Blockers:
  - none.
- Next unchecked item:
  - S6 optional refinements only (out of base mission).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (validation/guard-refactor only; no frame-membership drop logic introduced).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged in policy semantics (guard logic extracted to helper; behavior validated by tests).
- Working-set adaptation changed or stayed unchanged: stayed unchanged (tests/closeout only).


### 2026-03-13 21:56 — Iteration 48
- Scope: Add mission tracking for post-intertile slowdown diagnosis requested by Tristan.
- In scope:
  - update `agent.md` with add-on mission objective/rules/deliverables
  - update `followup.md` with actionable checklist section I
- Out of scope:
  - implementation/optimization changes (planning-only update)
- Files changed:
  - `agent.md`
  - `followup.md`
  - `memory.md`
- Tests run:
  - none (docs/checklist update only)
- Proof:
  - Added "Active add-on mission (2026-03-13) — Phase 5 slowdown after intertile anchor event" to `agent.md`.
  - Added section `I. Add-on mission — Phase 5 slowdown after intertile anchor event` in `followup.md` with I1..I5 checklist.
- Decisions:
  - Keep prior Phase 3 mission closed; track new work as add-on mission focused on runtime performance diagnosis and optimization.
- Blockers:
  - none
- Next unchecked item:
  - I1. Reproduction and timeline.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 22:13 — Iteration 49
- Scope: Implement display FITS companion export for black-looking scientific FITS in viewers.
- In scope:
  - add automatic Phase 6 export of `*_display.fits`
  - keep scientific FITS unchanged
  - add regression source-contract test
- Out of scope:
  - changes to scientific assembly math or CPU/GPU processing logic
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k display_fits_companion` -> `1 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `38 passed, 21 warnings`
- Proof:
  - Phase 6 now writes `zemosaic_*_display.fits` via preview-like robust stretch (`p_low=2.5`, `p_high=99.8`, `asinh_a=20`) and uint16 export helper.
  - Scientific FITS path remains unchanged (`zemosaic_*_R*.fits` still float scientific output).
  - Display header tags added: `ZMDISPF`, `ZMDPLWR`, `ZMDPHIG`, `ZMDPASH`.
- Decisions:
  - Keep display companion always enabled (no new config gate) per user request.
- Blockers:
  - none
- Next unchecked item:
  - User validation on real run output readability in external FITS viewers.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 22:28 — Iteration 50
- Scope: ETA reliability fix (CPU/GPU modes) requested by Tristan.
- In scope:
  - analyze ETA formulas and runtime hooks
  - remove ETA double-counting patterns
  - add ETA smoothing to reduce jumps
  - reduce misleading intertile heuristic ETA seed impact
  - add ETA regression/source-contract tests
- Out of scope:
  - deep runtime benchmark campaign on multiple datasets
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "eta_seconds_from_progress or eta_smoothing or eta_uses_max_not_double_count_sum"` -> `3 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `41 passed, 21 warnings`
- Proof:
  - Added `_eta_seconds_from_progress(...)` helper using guarded global progress ratio.
  - Added `_eta_smooth_seconds(...)` low-pass ETA filter with outlier jump bounding.
  - Updated ETA composition from additive (`local + global`) to conservative max (`max(local, global)`) in prep/channel/Phase1/Phase3 paths.
  - Updated both runtime `update_gui_eta(...)` closures to use smoothing state before emitting `ETA_UPDATE`.
  - Intertile heuristic now emits debug seed (`[Intertile][ETA seed]`) instead of forcing direct `ETA_UPDATE`, reducing false spikes.
- Decisions:
  - Prioritize ETA stability/consistency over aggressive early precision.
- Blockers:
  - Two-pass long spans still depend on stage progress cadence; further granularity tuning can be added after real-run feedback.
- Next unchecked item:
  - Validate ETA drift on real production run and tune smoothing alpha if needed.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 22:38 — Iteration 51
- Scope: ETA hotfix after user report of severe underestimation (00:00:30 → 00:02:10 on a known 20–30 min dataset).
- In scope:
  - inspect live ETA/log progression
  - add conservative phase-cost ETA model
  - combine local/progress/model ETA with safe max
  - keep smoothing and remove misleading intertile ETA forcing
- Out of scope:
  - persistent per-dataset historical ETA learning
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "eta_phase_model_conservative_vs_progress_in_phase3_midrun or source_contract_eta_uses_phase_model"` -> `2 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `43 passed, 21 warnings`
- Proof:
  - Live log showed early ETA collapse in P1/P3 despite long known total runtime.
  - Added `_eta_seconds_from_phase_model(...)` with conservative phase shares (P5 dominant).
  - Updated ETA computation paths (P1/P3/P5 prep/P5 channels) to include phase-model ETA and take `max(local, global, phase_model)`.
  - Existing ETA smoothing retained.
- Decisions:
  - Favor conservative realistic ETA over optimistic early ETA.
- Blockers:
  - This fix affects next run; current running process keeps previous ETA logic until restart.
- Next unchecked item:
  - Validate ETA realism on next full run and tune phase shares if needed.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 22:47 — Iteration 52
- Scope: Phase 3 ETA-only refinement requested by Tristan (online extrapolation from known master-tile durations).
- In scope:
  - Phase 3 ETA prediction by tile runtime (n_frames-aware)
  - live EWMA update as tiles complete
  - remaining wall-time projection using running+queued jobs and effective parallelism
- Out of scope:
  - any scientific pipeline/stacking behavior change
- Files changed:
  - `zemosaic_worker.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "eta_"` -> `5 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `43 passed, 21 warnings`
- Proof:
  - Added `_estimate_parallel_makespan_seconds(...)` helper.
  - Phase 3 now tracks per-tile start time + frame count at submission.
  - On completion, runtime updates global EWMA and per-frame-count EWMA model.
  - ETA Phase 3 now includes model-based remaining makespan across running + queued tiles, accounting for effective concurrency.
  - ETA update remains ETA-only; no data path modifications.
- Decisions:
  - Keep conservative `max(avg-based, model-based, global-progress, phase-model)` for Phase 3 ETA display.
- Blockers:
  - Current already-running process may need worker reload to pick up latest ETA code, depending on launch timing.
- Next unchecked item:
  - Validate live Phase 3 ETA coherence on current/restarted run.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 22:53 — Iteration 53
- Scope: Fix persistent Phase 3 ETA incoherence after live user feedback.
- In scope:
  - inspect live log from re-run
  - identify ETA smoothing pathology across phase transitions
  - patch ETA smoothing to allow regime-change reset
  - keep ETA-only changes
- Out of scope:
  - scientific/image processing behavior
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "eta_"` -> `6 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `44 passed, 21 warnings`
- Proof:
  - Live log showed Phase1 ETA ending ~00:00:28 then Phase3 ETA climbing slowly (00:00:44 -> 00:01:06 -> 00:04:10), inconsistent with conservative phase-model ETA.
  - Root cause: `_eta_smooth_seconds` hard clamp (`high=old*2.5+30`) prevented legitimate upward jumps at phase transitions.
  - Patch now performs phase-shift detection and immediate reset on large regime changes.
- Decisions:
  - Preserve smoothing for normal fluctuations, but never suppress valid phase-transition jumps.
- Blockers:
  - Current run started before this patch load; effect applies to next run/restart only.
- Next unchecked item:
  - Verify Phase 3 ETA on next fresh run with updated smoothing-reset logic.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-13 23:11 — Iteration 54
- Scope: Phase 5 intertile-pairs ETA coherence fix requested by Tristan after live log verification.
- In scope:
  - add direct ETA updates from intertile pair progression callbacks
  - keep reproject/scientific behavior unchanged
  - add regression source-contract test
- Out of scope:
  - broader Phase 5 ETA model retuning
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "intertile_pairs_emit_eta_updates or eta_"` -> `7 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `45 passed, 21 warnings`
- Proof:
  - `_intertile_progress_bridge` now handles structured callbacks `phase5_intertile_pairs` and emits `ETA_UPDATE:HH:MM:SS` derived from done/total pairs and elapsed time.
  - Added compatibility parsing for textual `pairs_done=X/Y` messages.
  - ETA emission is throttled (first/last, every 5 pairs, or >=2s) and smoothed with `_eta_smooth_seconds`.
- Decisions:
  - Keep this patch ETA-only and scoped to intertile pair stage to avoid touching scientific compute paths.
- Blockers:
  - Running process must use this updated code path (restart/new run) for visible effect.
- Next unchecked item:
  - Validate live intertile ETA trend in next run logs.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-14 00:03 — Iteration 55
- Scope: Gate display FITS companion behind config flag to restore previous default output behavior.
- In scope:
  - add `save_display_fits` config flag (default disabled)
  - condition Phase 6 `*_display.fits` export on that flag
  - keep scientific FITS export unchanged
  - update tests for source contract
- Out of scope:
  - GUI toggle wiring for the new flag (manual config/default behavior only)
- Files changed:
  - `zemosaic_worker.py`
  - `zemosaic_config.py`
  - `zemosaic_config.json`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "display_fits_companion or intertile_pairs_emit_eta_updates or eta_"` -> `8 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `45 passed, 21 warnings`
- Proof:
  - Phase 6 now reads `save_display_fits_config = bool(getattr(zconfig, "save_display_fits", False))`.
  - `*_display.fits` is generated only when `save_display_fits` is enabled.
  - Default config now includes `"save_display_fits": false`.
- Decisions:
  - Keep legacy/default behavior (no display companion) unless explicitly enabled by config.
- Blockers:
  - none
- Next unchecked item:
  - optional: expose `save_display_fits` in GUI settings if desired.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.


### 2026-03-14 00:26 — Iteration 56
- Scope: Reduce GPU OOM frequency on low-VRAM hosts (TinyDebian MX150) while preserving full GPU usage on stronger desktops.
- In scope:
  - Phase 3 WSC GPU preflight made VRAM-tier adaptive
  - Phase 3 OOM backoff persists conservative row-cap across subsequent chunks/tiles
  - Phase 5 GPU reproject OOM retry persists tightened chunk hints across retries/channels
  - add source-contract tests for new OOM-adaptive logic
- Out of scope:
  - disabling GPU globally
  - scientific algorithm changes
- Files changed:
  - `zemosaic_align_stack_gpu.py`
  - `zemosaic_utils.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `memory.md`
- Tests run:
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "phase3_wsc_oom_backoff_persists_rows_cap or phase5_gpu_oom_hint_persists_across_retries"` -> `2 passed`
  - `../.venv/bin/python -m pytest -q tests` -> `47 passed, 21 warnings`
- Proof:
  - Added `_wsc_budget_fraction_by_vram(total_vram_bytes, n_frames)` with tiered fractions (low VRAM => lower budget fraction).
  - Added run-scoped WSC OOM memory (`_WSC_DYNAMIC_ROWS_CAP`, `_WSC_OOM_EVENTS`) so one OOM tightens future Phase 3 chunking automatically.
  - Phase 5 reproject now has persistent OOM hints (`_PHASE5_GPU_OOM_HINT`) applied before calls and tightened on each OOM retry.
  - High-VRAM machines remain effectively unconstrained unless OOM actually occurs.
- Decisions:
  - Keep fallback architecture unchanged (GPU retry/backoff first, CPU fallback still available) and improve proactive sizing + persistence.
- Blockers:
  - Patch applies on next run (current run process may still be on old code if not restarted after edit).
- Next unchecked item:
  - Validate OOM incidence drop on TinyDebian with same reference dataset; compare wall-time impact.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 08:32 — Iteration 57
- Scope: surgical FITS compliance fix for RGB 3D exports (master tiles + final mosaic paths) to remove `CTYPE3/WCSAXES` inconsistency.
- In scope:
  - map exact write sites for Primary header/WCSAXES/CTYPE3/EXTNAME/ALPHA
  - patch shared FITS writer behavior with minimal diff
  - validate with `fitsverify` on representative master tile + final mosaic outputs
- Out of scope:
  - pipeline refactor
  - scientific data-path changes
- Files changed:
  - `zemosaic_utils.py`
- Tests run:
  - synthetic bad-file baseline showing original error class:
    - `fitsverify /tmp/zem_fitsverify/before_bug_equivalent.fits` -> `Keyword CTYPE3: index 3 is not in range 1-2 (WCSAXES)`
  - representative exports through real helper (`save_fits_image`, float RGB cube, WCSAXES=2):
    - `fitsverify /tmp/zem_fitsverify/master_tile_repr.fits` -> 0 warnings / 0 errors
    - `fitsverify /tmp/zem_fitsverify/final_mosaic_repr.fits` -> 0 warnings / 0 errors
- Proof / mapping:
  - common writer confirmed: `zemosaic_worker.py` master tiles + final mosaic both call `zemosaic_utils.save_fits_image(..., save_as_float=True, axis_order="HWC", alpha_mask=...)`.
  - responsible sites in shared writer:
    - float RGB path added `CTYPE3='RGB'` and `EXTNAME='RGB'` in Primary when 3D
    - legacy RGB cube helper also added `CTYPE3='RGB'` and `EXTNAME='RGB'`
    - ALPHA extension writing remains separate (`ImageHDU(..., name='ALPHA')`)
- Patch summary:
  - added `_header_declares_2d_wcs(header)`
  - when header declares `WCSAXES=2`, Primary RGB 3D path now removes `CTYPE3` instead of setting it
  - removed `EXTNAME` from Primary in those RGB Primary paths (keeps ALPHA extension naming untouched)
  - preserved `WCSAXES=2`, `CTYPE1/2`, `CRPIX*`, `CRVAL*`, `PC*`, `CUNIT*` etc.
- Result:
  - `CTYPE3/WCSAXES` error class corrected on representative master/final outputs.
- Risks:
  - low/contained: header hygiene change in shared writer affects all RGB 3D Primary exports using this helper; Windows compatibility expected to improve (no intentional data-layout change).
- Next unchecked item:
  - optional real-dataset A/B check in ASIFitsViewer Windows/Linux after this patch.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 09:25 — Iteration 58
- Scope: Regression SDS/grid_mode audit and remediation planning only (no code fix yet).
- In scope:
  - read mission continuity docs (`agent.md`, `followup.md`, `memory.md`)
  - inspect latest SDS failure logs
  - map probable refactor regression point
  - create persistent tracking file for this new regression mission
- Out of scope:
  - applying code patches
  - running post-fix validation matrix
- Files changed:
  - `fix_regression.md` (new)
  - `memory.md`
- Tests run:
  - log inspection: `/home/tristan/.config/ZeMosaic/zemosaic_worker.log`
  - static code inspection around failing lines in `zemosaic_worker.py`
  - occurrence scan for `existing_master_tiles_results`
- Proof:
  - crash reproduced in logs at `run_hierarchical_mosaic` after Phase 2 with:
    - `NameError: name 'existing_master_tiles_results' is not defined`
  - in `run_hierarchical_mosaic` the variable is referenced but not initialized in-function.
  - same symbol is properly initialized in `run_hierarchical_mosaic_classic_legacy`, indicating a likely refactor scope regression.
- Decisions:
  - keep fix surgical and scope-limited to variable initialization/guarding in the new path.
  - add a dedicated mission tracker `fix_regression.md` with iteration log + TODO list.
- Blockers:
  - none for starting the patch phase.
- Next unchecked item:
  - Apply minimal fix in `run_hierarchical_mosaic`, then run SDS/grid + non-SDS smoke validation matrix.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged (analysis-only iteration).
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 09:33 — Iteration 59
- Scope: SDS restoration only (priority user), fix NameError regression in new `run_hierarchical_mosaic` path.
- In scope:
  - patch `existing_master_tiles_results` undefined-symbol regression
  - keep mode architecture constraint explicit (classic/grid/SDS are mutually exclusive)
  - add non-regression test guard for this exact failure class
- Out of scope:
  - broad refactor
  - algorithmic changes
  - full end-to-end GUI run validation matrix (pending next step)
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `fix_regression.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol"` -> `1 passed`
- Proof:
  - In `run_hierarchical_mosaic`, bootstrap now uses `master_tiles_results_list = []`, removing dependency on undefined `existing_master_tiles_results`.
  - Legacy path occurrence remains unchanged to avoid collateral behavior drift.
  - Dedicated source-contract test now enforces this invariant in CI/local runs.
- Decisions:
  - Keep fix surgical and SDS-focused.
  - Validate via real SDS GUI toggle run next, then smoke classic/grid separately.
- Blockers:
  - none for running post-fix SDS reproduction.
- Next unchecked item:
  - Re-run SDS mode from GUI and confirm no `NameError` at Phase2->Phase3 handoff.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 10:11 — Iteration 60
- Scope: SDS-only regression fix for broadcast shape mismatch in Phase 5 global polish.
- In scope:
  - analyze new SDS crash reported in latest worker log
  - patch SDS finalization path to preserve geometry consistency
  - add non-regression marker test
- Out of scope:
  - CPU fallback behavior analysis (explicitly ignored per user request)
  - non-SDS algorithm changes
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `fix_regression.md`
  - `memory.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish"` -> `2 passed`
- Proof:
  - traceback pointed to `np.where(keep_mask > 0, final_mosaic_coverage, 0.0)` with shape mismatch `(1099,32)` vs `(3330,3748)`.
  - root cause: SDS global polish allowed lecropper quality crop (geometry-changing) on mosaic while coverage stayed at global descriptor size.
  - fix: in `_finalize_sds_global_mosaic`, force `quality_crop_enabled=False` in local SDS pipeline config and log marker `phase5_sds_quality_crop_disabled`.
- Decisions:
  - keep SDS geometry stable in global polish; preserve non-cropping parts of lecropper pipeline (alt-az cleanup path).
- Blockers:
  - none for runtime validation.
- Next unchecked item:
  - rerun real SDS GUI flow and confirm no broadcast error + successful output completion.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 10:38 — Iteration 61
- Scope: SDS priority fix on VRAM/OOM behavior in global coadd helper path; color check secondary.
- In scope:
  - inspect latest SDS log OOM pattern
  - port adaptive GPU OOM handling concept to SDS helper route
  - add non-regression source-contract test
  - quick objective color sanity check on outputs
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `fix_regression.md`
  - `memory.md`
- Technical changes:
  - `_attempt_gpu_helper_route` now declares `nonlocal plan_rows_gpu_hint, plan_chunk_gpu_hint`
  - added per-channel GPU retry loop (`max_gpu_helper_retries=3`)
  - on OOM: free CuPy pools when available, halve rows/chunk hints, log `global_coadd_gpu_oom_retry`, retry GPU before CPU fallback
  - tightened hints persist during the same run/batch for next channels
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py`
  - `python3 -m py_compile tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "sds_global_gpu_helper_has_oom_retry_with_chunk_tightening or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish or run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol"` -> `3 passed`
- Color sanity check:
  - PNG preview stats (RGBA) confirm RGB channels are not identical (non-zero inter-channel diffs)
  - FITS final stats also confirm 3-channel data present; no strict grayscale conversion detected
- Decisions:
  - classify current issue as GPU OOM pressure handling first; not proven VRAM leak from logs
  - defer color pipeline change until post-rerun evidence confirms persistent visual desaturation issue
- Next unchecked item:
  - rerun SDS and verify reduced CPU fallbacks + inspect visual rendering path if user still perceives grayscale

### 2026-03-14 11:04 — Iteration 62
- Scope: read latest SDS log, declare operational status, add qualitative next-step, and audit ZeGrid health.
- SDS log verdict:
  - latest run completed end-to-end (`run_success_mosaic_saved`, preview saved, processing completed, final COMPLETED banner)
  - no reappearance of previous blocking regressions (`existing_master_tiles_results` NameError / broadcast mismatch)
- Decision:
  - mark SDS as functionally repaired for current dataset/workflow.
- Added mission next-step:
  - investigate qualitative inter-frame harmonization inside each super-tile (normalization/photometric consistency) as follow-up improvement.
- ZeGrid audit performed:
  - static compile checks OK (`grid_mode.py`, `zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`)
  - runtime signature compatibility confirmed between worker call site and `grid_mode.run_grid_mode`
  - import works in project venv
  - smoke on empty folder yields controlled expected error (`Grid mode failed: no frames loaded`), no unexpected crash
- Limitation:
  - no recent real ZeGrid run in worker log, so production health remains to validate on a true grid dataset.
- Files changed:
  - `fix_regression.md`
  - `memory.md`
- Next unchecked item:
  - execute one real ZeGrid run and validate completion + output artifacts.

### 2026-03-14 11:11 — Iteration 63
- Scope: fix ZeGrid failure `Grid mode failed: no frames loaded` with `stack_plan.csv` present.
- Root cause from worker log:
  - CSV rows used Windows absolute paths (`D:\...`) while run host is Linux.
  - grid parser resolved paths as `base_dir / "D:\..."`, so every row became `file not found`.
- Code change:
  - hardened `grid_mode._resolve_path` with cross-platform fallbacks for Windows-style paths:
    - detect drive/UNC/backslash formats,
    - try tail-relative and basename resolution under CSV folder,
    - keep existing native absolute/relative behavior.
- Evidence:
  - real problematic CSV re-check:
    - before: `frames=0`
    - after: `frames=24`, all files resolved and existing.
- Tests:
  - added `tests/test_grid_mode_stack_plan_paths.py`
  - `pytest -q tests/test_grid_mode_stack_plan_paths.py` -> `1 passed`
  - SDS non-regression guard tests rerun -> `3 passed`
- Files changed:
  - `grid_mode.py`
  - `tests/test_grid_mode_stack_plan_paths.py`
  - `fix_regression.md`
  - `memory.md`
- Next unchecked item:
  - run full ZeGrid from GUI on real dataset and verify end-to-end completion artifacts.

### 2026-03-14 14:18 — Iteration 64
- Scope: verify real GPU activity for Phase 5 reproject in existing_master_tiles mode.
- Findings from worker log:
  - GPU mode was requested and accepted (`phase5_using_gpu`, phase5 plan.use_gpu=1).
  - no `gpu_fallback_runtime_error` seen during run.
  - two-pass renorm explicitly reports GPU backend (`cupy_chunk/cupy_chunk`, `gpu_all=True`).
  - GPU memory telemetry changed (38→42MB reprojection, ~132MB blur/gains), indicating GPU allocations.
  - `gpu_util_percent` remains unavailable (`None`) because telemetry source is `cupy_meminfo` (memory-only).
- Conclusion:
  - GPU path is active in this run; apparent "no activity" likely due to limited util telemetry, not CPU-only fallback.
- Next optional improvement:
  - add explicit success logs/counters for GPU reproject chunks + NVML utilization probe when available.
- Files changed:
  - `fix_regression.md`
  - `memory.md`

### 2026-03-14 15:28 — Post-mortem refactor (garde-fou architecture modes)
- Constat durable: l’architecture ZeMosaic repose sur **3 modes distincts et exclusifs**:
  1. **Classique**
  2. **ZeGrid** (déclenché par `stack_plan.csv`)
  3. **SDS / ZeSupaDupStack**
- Ces modes partagent une partie d’infrastructure, mais leurs chemins d’exécution divergent rapidement (préparation, phases intermédiaires, assemblage/finalisation, contraintes de géométrie, gestion GPU/CPU).
- Le refactor précédent a montré que de petites modifications “transverses” (variables partagées, initialisations, pipeline Phase 5, résolution de chemins) peuvent casser un mode sans casser les autres.
- Règle à conserver pour toute refactorisation future:
  - **penser mode-par-mode** (Classique, ZeGrid, SDS),
  - valider explicitement chaque mode,
  - éviter les hypothèses implicites de compatibilité entre modes,
  - maintenir des garde-fous/tests de non-régression dédiés par mode,
  - traiter les spécificités cross-plateforme (ex. chemins Windows dans ZeGrid sur hôte Linux).
- Objectif: faire évoluer le code sans réintroduire de régressions silencieuses sur ces trois voies.

### 2026-03-14 15:49 — Iteration 1 (nouvelle mission qualité, audit-only)
- Scope: lancement mission harmonisation qualitative multi-modes (sans changement code pipeline).
- In scope:
  - vérifier le garde-fou RGB finaleq actuellement désactivé
  - vérifier présence des artefacts de référence de baseline
  - initialiser la checklist mission (`followup.md`) avec preuves
- Out of scope:
  - modification de comportement runtime
  - réactivation effective de la final RGB equalization
  - implémentation DBE ZeGrid
- Files changed:
  - `agent.md` (nouvelle mission)
  - `followup.md` (checklist initiale + items audit cochés)
  - `memory.md`
- Tests run / preuves:
  - scan code `zemosaic_worker.py`: garde-fou confirmé
    - commentaire explicite: "We temporarily disable final mosaic RGB equalization"
    - mention explicite de "strong green cast"
  - vérification artefacts baseline: 15/15 présents dans `example/out/`:
    - global_mosaic_wcs.{fits,json}
    - mosaic_grid.{fits,coverage.fits}
    - resource_telemetry.csv
    - run_config_snapshot.json
    - zemosaic_MT0_R30 (+ coverage + preview)
    - zemosaic_MT14_R0 (+ coverage + preview)
    - zemosaic_MT14_R30 (+ coverage + preview)
- Decisions:
  - mission démarre en audit-first, sans patch code, pour réduire le risque de régression.
  - baseline d’artefacts gelée comme référence de comparaison qualitative.
- Blockers:
  - aucun.
- Next unchecked item:
  - `B3` cartographie inter-modes détaillée (normalisation / RGB / DBE / preview).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 15:52 — Iteration 2 (mission qualité multi-modes, B3 cartographie)
- Scope: B3 uniquement (audit inter-modes), sans modification du pipeline runtime.
- In scope:
  - cartographier par mode: normalisation photométrique, RGB equalization, DBE final, preview/stretch
  - produire preuves code (lignes/fonctions)
  - cocher B3 dans `followup.md`
- Out of scope:
  - réactiver la final RGB equalization
  - implémenter DBE ZeGrid
  - lancer des runs de validation visuelle
- Files changed:
  - `followup.md` (B3.1→B3.4 cochés)
  - `memory.md`
- Tests run / commandes:
  - `grep -n "run_grid_mode\|sds_mode\|use_existing_master_tiles_mode" zemosaic_worker.py`
  - `nl -ba zemosaic_worker.py | sed -n '22220,22345p'`
  - `nl -ba zemosaic_worker.py | sed -n '22770,22875p'`
  - `nl -ba zemosaic_worker.py | sed -n '10070,10118p'`
  - `nl -ba zemosaic_worker.py | sed -n '10190,10270p'`
  - `nl -ba zemosaic_worker.py | sed -n '23320,23425p'`
  - `nl -ba zemosaic_worker.py | sed -n '23795,24010p'`
  - `nl -ba zemosaic_worker.py | sed -n '26590,26870p'`
  - `nl -ba zemosaic_worker.py | sed -n '28096,28215p'`
  - `nl -ba zemosaic_worker.py | sed -n '28405,28640p'`
  - `nl -ba grid_mode.py | sed -n '2295,2340p'`
  - `nl -ba grid_mode.py | sed -n '3060,3245p'`
  - `nl -ba grid_mode.py | sed -n '3580,3670p'`
  - `nl -ba grid_mode.py | sed -n '3668,3715p'`
  - `grep -n "stretch_auto_asifits_like" grid_mode.py`
- Cartographie B3 (résultat):

  | Mode | Normalisation photométrique | RGB equalization | DBE final | Preview/stretch |
  |---|---|---|---|---|
  | Classique | **Y** | **Partiel** | **Y** | **Y** |
  | Existing master tiles | **Partiel** | **Partiel** | **Y** | **Y** |
  | SDS | **Partiel** | **Partiel** | **Y** | **Y** |
  | ZeGrid | **Y** | **Y** | **N (hook only)** | **Partiel/N** |

- Preuves détaillées:
  - Routage des modes:
    - ZeGrid est routé séparément via `run_grid_mode` (`zemosaic_worker.py:24431-24484`).
    - Classique passe par `run_hierarchical_mosaic_classic_legacy` si `not grid and not sds` (`24497-24514`).
    - SDS suit un chemin dédié (`26590+`, `26687-26732`).
  - Normalisation photométrique:
    - Classique: center-out P3 préparé (`22231-22335`) + intertile photometric match P5 (`10092`, `10114`).
    - Existing master tiles: pas de Phase 3 locale (préchargement tuiles existantes, `21710+`, puis branche `if not use_existing_master_tiles_mode` évitée), mais P5 intertile reste actif (`23292+`, `10092`).
    - SDS: normalisation photométrique des méga-tiles (`_normalize_sds_megatiles_photometry`, `31017-31024`) + finition SDS dédiée (`26687+`).
    - ZeGrid: scaling photométrique inter-tiles explicite (`grid_mode.py:3083-3206`).
  - RGB equalization:
    - Classique: equalization post-stack au niveau master-tile (`poststack_equalize_rgb` au submit `22727`; trace/effects `14129-14144`).
    - Existing master tiles: Phase 3 locale sautée => pas de poststack RGB local; final RGB equalization mosaïque explicitement désactivée (`10199-10217`).
    - SDS: final RGB equalization mosaïque aussi désactivée (`10199-10217`), et chemin SDS nominal contourne la logique P3 classique.
    - ZeGrid: equalization post-stack tile optionnelle (`grid_mode.py:2318-2326`) + equalization mosaïque ZeGrid (`3621-3643`).
  - DBE final:
    - Classique/Existing: DBE Phase 6 active (gate + appel `_apply_final_mosaic_dbe_per_channel`) (`23383-23407`, `23406+`).
    - SDS: même DBE Phase 6 sur le chemin SDS (`28124-28207`).
    - ZeGrid: flag DBE transmis (`4302-4319`) mais hook seulement loggé (`"bypassing worker Phase 6"`, `3653-3661`) sans appel DBE effectif dans `grid_mode.py`.
  - Preview/stretch:
    - Classique/Existing: preview Phase 6 + `stretch_auto_asifits_like` + export PNG (`23812`, `23927-23964`).
    - SDS: preview/stretch identique sur chemin SDS (`28434`, `28548-28584`).
    - ZeGrid: sauvegarde FITS/science/viewer (`3668-3712`) mais pas d’appel `stretch_auto_asifits_like` ni pipeline preview PNG dans `grid_mode.py`.
- Decisions:
  - B3 considéré terminé (audit cartographié + preuves traçables).
  - Prochaine étape mission: C1 (essai contrôlé de réactivation final RGB equalization derrière flag).
- Blockers:
  - aucun pour passer à C1.
- Next unchecked item:
  - C1 — Implémentation contrôlée de la réactivation (flag explicite, default conservateur, logs clairs).
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 20:56 — Iteration 3 (Q1 C1 impl + C2 Run A baseline log)
- Scope: traiter la demande utilisateur (audit du dernier log RGB-final OFF) puis exécuter le prochain item non coché (C1), sans lancer de run.
- In scope:
  - analyser le dernier run OFF dans `~/.config/ZeMosaic/zemosaic_worker.log`
  - implémenter C1 (réactivation contrôlée final RGB equalization)
  - conserver default conservateur
  - ajouter logs explicites ON/OFF
  - poser la base pour Run B (flag on)
- Out of scope:
  - run de validation flag ON
  - tuning color final après comparaison
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `followup.md`
  - `memory.md`
- Log audit (Run A OFF, baseline):
  - run confirmé terminé: `run_success_processing_completed`, durée ~582.73s.
  - mode: `existing_master_tiles_mode` + pipeline classique legacy.
  - aucun marqueur d’application final RGB equalization (normal car bloc désactivé), black-point égalisation active:
    - `[BlackPoint] Final mosaic RGB pedestal subtracted ...`
  - DBE final actif en phase 6 (`[DBE] applied=True ...`).
- Code changes C1 (réactivation contrôlée):
  - réactivation du hook final RGB eq derrière `final_mosaic_rgb_equalize_enabled` dans le chemin partagé phase 5 (`zemosaic_worker.py:10207-10243`).
  - parité ajoutée dans l’autre chemin finalisation (`zemosaic_worker.py:23558-23594`).
  - logs explicites ajoutés:
    - OFF: `final mosaic skipped: disabled by config ...`
    - ON+SDS: `enabled=True but sds_mode_phase5=True`
    - ON: `final mosaic gate: enabled=True applied=... gains=... target_median=...`
- Valeur par défaut conservatrice:
  - inchangée et explicite dans config: `final_mosaic_rgb_equalize_enabled = False` (`zemosaic_config.py`).
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py`
  - `python3 -m py_compile tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "final_mosaic_rgb_equalization_gate_and_logging_markers_present or run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish"` -> `3 passed`
- followup status update:
  - C1.1 ✅
  - C1.2 ✅
  - C1.3 ✅
  - C2 Run A (flag off baseline) ✅
- Next unchecked item:
  - C2 Run B (flag on) comparaison.
- All-raw-frames invariant changed or stayed unchanged: stayed unchanged.
- Phase 3 launch control changed or stayed unchanged: stayed unchanged.
- Working-set adaptation changed or stayed unchanged: stayed unchanged.

### 2026-03-14 21:03 — Iteration 4 (préparation Run B Q1)
- Scope: bascule opératoire demandée par Tristan pour exécuter Run B avec final RGB equalization ON.
- In scope:
  - activer le flag runtime sans changer le défaut code/config global conservateur
- Files changed:
  - `/home/tristan/.config/ZeMosaic/zemosaic_config.json`
- Action:
  - `final_mosaic_rgb_equalize_enabled` passé de `false` à `true` dans la config utilisateur active.
- Proof:
  - relecture JSON immédiate -> valeur `True` confirmée.
- Notes:
  - aucun autre paramètre modifié.
  - après Run B, prévoir remise à `false` si on reste en posture conservatrice.

### 2026-03-14 21:20 — Iteration 5 (hotfix NameError Run B)
- Scope: corriger crash runtime remonté pendant Run B: `name 'final_mosaic_rgb_equalize_enabled' is not defined`.
- Root cause:
  - dans `run_hierarchical_mosaic_classic_legacy`, le hook final RGB-eq utilisait `final_mosaic_rgb_equalize_enabled` sans initialisation locale.
  - le bloc avait été ajouté en C1 mais sans normaliser le flag dans ce scope.
- Fix applied (surgical):
  - initialisation locale ajoutée en entrée de fonction:
    - coercion depuis `final_mosaic_rgb_equalize_enabled_config`
    - fallback `zconfig.final_mosaic_rgb_equalize_enabled`
    - défaut conservateur `False`
  - aucun autre comportement pipeline modifié.
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py`
  - `python3 -m py_compile tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "classic_legacy_initializes_final_rgb_equalize_flag_before_use or final_mosaic_rgb_equalization_gate_and_logging_markers_present"` -> `2 passed`
- Status:
  - hotfix prêt; relancer Run B requis (même config).

### 2026-03-14 21:38 — Iteration 6 (hotfix NameError sds_mode_phase5)
- Scope: corriger crash runtime remonté pendant Run B: `name 'sds_mode_phase5' is not defined`.
- Root cause:
  - le hook final RGB-eq/black-point en phase finale du chemin classique référençait `sds_mode_phase5` sans initialisation locale.
- Fix applied (surgical):
  - ajout de `sds_mode_phase5 = bool(sds_mode_flag)` juste après la résolution de `sds_mode_flag` dans `run_hierarchical_mosaic_classic_legacy`.
  - pas de changement d’algorithme; uniquement robustesse d’état local.
- Files changed:
  - `zemosaic_worker.py`
  - `tests/test_phase3_adaptive_invariants.py`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py tests/test_phase3_adaptive_invariants.py`
  - `../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "classic_legacy_initializes_final_rgb_equalize_flag_before_use or final_mosaic_rgb_equalization_gate_and_logging_markers_present"` -> `2 passed`
- Status:
  - hotfix prêt, relance du run possible immédiatement.

### 2026-03-14 21:40 — Iteration 7 (vérif Grid mode + hotfix variable SDS)
- Scope: sécuriser aussi le chemin grid suite aux NameError signalés.
- Actions:
  - hotfix local dans `run_hierarchical_mosaic_classic_legacy`: `sds_mode_phase5 = bool(sds_mode_flag)`.
  - vérification du chemin Grid:
    - `run_hierarchical_mosaic` route `grid_mode.run_grid_mode(...)` avec variables définies.
    - `grid_mode.py` ne dépend pas de `sds_mode_phase5`.
- Validation:
  - `python3 -m py_compile zemosaic_worker.py grid_mode.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_grid_mode_stack_plan_paths.py tests/test_phase3_adaptive_invariants.py -k "classic_legacy_initializes_final_rgb_equalize_flag_before_use or final_mosaic_rgb_equalization_gate_and_logging_markers_present or grid_mode"`
  - résultat: `3 passed`.
- Note:
  - premier essai de tests grid a échoué en collecte (PYTHONPATH manquant), corrigé sans changement code.

### 2026-03-14 22:06 — Iteration 8 (Q1 decision=tune + D1 démarré)
- Trigger: Tristan valide l’analyse visuelle (bleu trop présent) et demande passage en mode tune puis étape suivante.
- Run B status/logs:
  - run completed (`run_success_processing_completed`, ~771.66s)
  - final RGB-eq applied on final mosaic with aggressive gains: `(R=0.634967, G=1.000000, B=2.352305)`
  - impact observé: réduction du vert mais dominante bleue/magenta excessive.
- Decision C2:
  - keep/tune/revert => **tune**.
- Tune implementation (surgical):
  - updated `_apply_final_mosaic_rgb_equalization` to apply conservative gain clipping after helper output.
  - defaults:
    - `final_mosaic_rgb_equalize_clip_enabled = True`
    - `final_mosaic_rgb_equalize_gain_clip = [0.80, 1.25]`
  - logs added: `[RGB-EQ][TUNE] gain clip applied ... raw=(...) clipped=(...)`.
  - preserved behavior when no over-gain: no change.
- Files changed:
  - `zemosaic_worker.py`
  - `zemosaic_config.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `followup.md`
- Tests run:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_config.py grid_mode.py tests/test_phase3_adaptive_invariants.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py tests/test_grid_mode_stack_plan_paths.py -k "final_mosaic_rgb_equalization_gate_and_logging_markers_present or classic_legacy_initializes_final_rgb_equalize_flag_before_use or grid_mode"`
  - result: `3 passed`.
- D1 (mode existing master tiles) — gap analysis summary:
  - missing vs classic: Phase3 local poststack RGB-eq is absent when reusing MT; final RGB-eq on mosaic was disabled historically (now tunable/re-enabled).
  - already covered by two-pass/affine: phase5 intertile matching (including affine/two-pass coverage renorm), DBE phase6, black-point equalization, preview/stretch exports.
- Next unchecked item:
  - D2.1 ajouter mécanisme(s) manquant(s) sans perturber two-pass/affine.
- Runtime config aligned for next run (`~/.config/ZeMosaic/zemosaic_config.json`):
  - `final_mosaic_rgb_equalize_enabled=true`
  - `final_mosaic_rgb_equalize_clip_enabled=true`
  - `final_mosaic_rgb_equalize_gain_clip=[0.8, 1.25]`

### 2026-03-14 23:05 — Iteration 9 (Q2 D2.1 existing-master harmonisation)
- Trigger: Tristan valide visuellement le mode tune (léger rouge perçu) et autorise poursuite mission autonome.
- Scientific reading (run tune latest):
  - final RGB-eq gains now moderate (`R=1.0427, G=1.0, B=0.9607`), no extreme blue boost.
  - residual warm bias mild (~1%) plausible on low-coverage testset.
- D2.1 implementation (existing master tiles mode):
  - Added conservative **pre-phase5 per-tile RGB balance** in `assemble_final_mosaic_reproject_coadd`, executed only when `existing_master_tiles_mode=True`.
  - Purpose: emulate missing classic Phase3 poststack RGB harmonization when reusing prebuilt master tiles.
  - Defaults (new config keys):
    - `existing_master_tiles_rgb_balance_prephase5 = True`
    - `existing_master_tiles_rgb_balance_gain_clip = [0.90, 1.10]`
    - `existing_master_tiles_rgb_balance_min_pixels = 5000`
  - Behavior:
    - derive valid-pixel mask from alpha/coverage,
    - compute per-tile RGB medians,
    - apply clipped gains toward tile-internal neutral target,
    - log per-tile gains + summary,
    - fail-safe: continue unchanged on errors.
- Files changed:
  - `zemosaic_worker.py`
  - `zemosaic_config.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `followup.md` (D2.1 checked)
- Tests:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_config.py grid_mode.py tests/test_phase3_adaptive_invariants.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py tests/test_grid_mode_stack_plan_paths.py -k "existing_master_tiles_prephase5_rgb_balance_markers_present or final_mosaic_rgb_equalization_gate_and_logging_markers_present or classic_legacy_initializes_final_rgb_equalize_flag_before_use or grid_mode"`
  - result: `4 passed`
- Next step requested from user:
  - Run validation for D2.2/D3 (compat outputs FITS/coverage/preview + before/after qualitative check).

### 2026-03-14 23:25 — Iteration 10 (blue bias mitigation for existing-master run)
- Trigger: user reports residual blue dominance on latest existing-master run.
- Diagnosis from logs/science:
  - pre-phase5 existing-master rgb balance is very mild (gains ~1.00±0.01), not main culprit.
  - final RGB-eq remained the dominant lever with clip-edge behavior on prior run (`raw=(0.741,1.000,1.537) -> clipped=(0.800,1.000,1.250)`).
- Fix/tune applied (targeted to existing-master mode only):
  - added override of final RGB-eq gain clip in classic legacy path when `use_existing_master_tiles_mode=True`:
    - config key: `existing_master_tiles_final_rgb_equalize_gain_clip`
    - default: `[0.90, 1.10]`
  - worker logs explicit marker:
    - `existing_master_tiles_mode: overriding final RGB-eq gain clip to (...)`
- Files changed:
  - `zemosaic_worker.py`
  - `zemosaic_config.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - runtime config `~/.config/ZeMosaic/zemosaic_config.json` updated with `existing_master_tiles_final_rgb_equalize_gain_clip=[0.9,1.1]`.
- Validation:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_config.py tests/test_phase3_adaptive_invariants.py grid_mode.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py tests/test_grid_mode_stack_plan_paths.py -k "existing_master_tiles_overrides_final_rgb_clip_markers_present or existing_master_tiles_prephase5_rgb_balance_markers_present or final_mosaic_rgb_equalization_gate_and_logging_markers_present or grid_mode"`
  - result: `4 passed`.
- Next action:
  - request new validation run to confirm reduced blue in highlights while keeping global neutrality.
- 2026-03-15 01:30: existing-master final RGB-eq clip tightened on runtime config to `[0.95, 1.05]` (from `[0.90, 1.10]`) for next validation run.

### 2026-03-15 01:54 — Iteration 11 (D2.2 + D3 + E1 analysis)
- Scope: continue mission autonomously on next unchecked items.
- Completed checklist items:
  - D2.2 compatibility FITS/coverage/preview ✅
  - D3 run before/after + non-regression check ✅
  - E1 SDS gap analysis + geometry constraints isolation ✅
- D2.2/D3 evidence:
  - latest run completed successfully with no ERROR/Traceback in current log slice.
  - run markers present: `run_success_mosaic_saved` + `run_success_processing_completed`.
  - output artifacts present and valid:
    - `zemosaic_MT14_R0.fits` (HDU0 shape `(3,3035,2205)`, ALPHA ext present)
    - `zemosaic_MT14_R0_coverage.fits` (shape `(3035,2205)`)
    - `zemosaic_MT14_R0_preview.png`
  - comparative runs before/after tune available (`R30` vs `R0`) and analyzed for color drift/highlights.
- SDS E1 (qualitative differences vs classic):
  - SDS has dedicated photometric normalization stage on mega-tiles (`_normalize_sds_megatiles_photometry`).
  - SDS finalize path uses `_finalize_sds_global_mosaic` with SDS-specific masking/polish logic.
  - final RGB equalization is explicitly skipped when `sds_mode_phase5=True` in shared phase5 gate and phase6 gate.
  - black-point equalization is also gated off for SDS (`and not sds_mode_phase5`).
  - DBE final remains available in phase6 worker path.
- SDS geometry/global constraints isolated:
  - SDS polish must preserve global descriptor geometry `(H,W)`.
  - quality-crop is explicitly disabled in SDS finalize (`phase5_sds_quality_crop_disabled`) to prevent shape/broadcast breakage.
  - SDS low-coverage masking + nanization depends on coverage/alpha consistency.
- Next unchecked item now:
  - E2.1 — add missing normalization/equalization for SDS in a compatible, non-regressive way.

### 2026-03-15 01:54 — Iteration 12 (Q3 E2 implementation, SDS-compatible equalization controls)
- Scope: continue mission autonomously on next unchecked item (E2) after E1 completion.
- Changes implemented (surgical, opt-in for SDS):
  - Added SDS-specific final equalization controls inside `_run_shared_phase45_phase5_pipeline`:
    - `sds_enable_final_rgb_equalize` (default `False`)
    - `sds_final_rgb_equalize_gain_clip` (default `[0.95, 1.05]`)
    - `sds_enable_final_black_point_equalize` (default `False`)
  - Behavior:
    - when `sds_mode_phase5=True` and SDS RGB flag is OFF -> preserved legacy skip behavior.
    - when SDS RGB flag is ON -> final RGB-eq is allowed in phase5 with SDS-specific clip override (logged).
    - black-point gate now allows SDS only when SDS black-point flag is ON.
  - No default behavior change for SDS runs (all new flags default to False).
- Files changed:
  - `zemosaic_worker.py`
  - `zemosaic_config.py`
  - `tests/test_phase3_adaptive_invariants.py`
  - `followup.md`
- Tests/validation:
  - `python3 -m py_compile zemosaic_worker.py zemosaic_config.py tests/test_phase3_adaptive_invariants.py grid_mode.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py tests/test_grid_mode_stack_plan_paths.py -k "final_mosaic_rgb_equalization_gate_and_logging_markers_present or existing_master_tiles_prephase5_rgb_balance_markers_present or existing_master_tiles_overrides_final_rgb_clip_markers_present or grid_mode or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish"` -> `5 passed`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_phase3_adaptive_invariants.py -k "sds_global_gpu_helper_has_oom_retry_with_chunk_tightening or sds_finalize_disables_geometry_changing_quality_crop_in_phase5_polish or classic_legacy_initializes_final_rgb_equalize_flag_before_use or run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol"` -> `4 passed`
- Checklist advanced:
  - E2.1 ✅
  - E2.2 ✅
- Next required user action for mission progression:
  - E3 validation run in SDS mode (full run + color/background coherence check).
- Note runtime: `existing_master_tiles_final_rgb_equalize_gain_clip` was observed back to `[0.9, 1.1]` after run snapshot; re-applied to `[0.95, 1.05]` in `~/.config/ZeMosaic/zemosaic_config.json`.

### 2026-03-15 07:51 — Iteration 13 (pilotage: clôture E + démarrage F1 ZeGrid DBE)
- Demande explicite Tristan: section E (SDS) peut être clôturée.
- `followup.md` mis à jour: E3.1 ✅, E3.2 ✅.
- Démarrage F1 (ZeGrid DBE):
  - `zemosaic_worker.py` route ZeGrid via `grid_mode.run_grid_mode(...)` puis `return` immédiat (Phase 6 worker non exécutée pour ZeGrid).
  - `grid_mode.py` lit bien `final_mosaic_dbe_enabled` et logge un hook DBE, mais n'applique pas de DBE effectif à la mosaïque avant `save_fits_image(...)`.
- Verdict F1 provisoire: gap DBE ZeGrid confirmé (hook informatif sans traitement DBE).

### 2026-03-15 07:54 — Iteration 14 (F2 ZeGrid DBE implémenté, logs explicites)
- Scope: Q4/F2 ZeGrid DBE (implémentation chirurgicale dans `grid_mode.py`).
- Implémentation:
  - ajout helper `_apply_grid_final_dbe(...)` (correction finale fond par canal RGB via modèle gaussien large, intention équivalente DBE).
  - `assemble_tiles(...)` applique désormais DBE ZeGrid quand `final_mosaic_dbe_enabled=True` avant sauvegarde FITS.
  - force valid-mask via `weight_sum>0` pour ne pas corriger hors zones couvertes.
  - ajout paramètre `final_mosaic_dbe_strength` (repris depuis config `final_mosaic_dbe_strength`, défaut `normal`).
- Logs ajoutés:
  - `[DBE] grid_mode: applied=<bool> strength=<...> sigma=<...> channels=<...> reason=<...>`
  - `[DBE] grid_mode: skipped (final_mosaic_dbe_enabled=False)`
- Tests/preuves:
  - `python3 -m py_compile grid_mode.py tests/test_grid_mode_dbe.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_grid_mode_dbe.py tests/test_grid_mode_stack_plan_paths.py` -> `3 passed`
- Checklist `followup.md` mise à jour:
  - F1.1 ✅, F1.2 ✅
  - F2.1 ✅, F2.2 ✅
- Next: F3 run ZeGrid réel complet + vérif sorties/non-régression.

### 2026-03-15 09:10 — Iteration 15 (ZeGrid DBE tuning anti-halos + mapping GUI)
- Trigger: halos noirs observés autour des étoiles brillantes uniquement en mode ZeGrid.
- Root cause confirmé:
  - ZeGrid DBE path plus simple que worker DBE classique/SDS (plus sensible à la sur-soustraction autour des objets brillants).
  - mapping force DBE GUI non aligné (`weak/strong` GUI vs `low/high` grid).
- Correctifs implémentés (chirurgicaux, grid only):
  - alias force DBE: `weak->24`, `normal->36`, `strong->52` (avec compat `low/high/aggressive`).
  - protection objets brillants avant modèle fond: détection robuste (median+MAD), dilation, exclusion du masque objets du modèle.
  - application de correction fond limitée au fond; zones objets protégées laissées inchangées pour éviter halos noirs.
- Validation:
  - `python3 -m py_compile grid_mode.py tests/test_grid_mode_dbe.py`
  - `PYTHONPATH=/home/tristan/zemosaic/zemosaic ../.venv/bin/python -m pytest -q tests/test_grid_mode_dbe.py tests/test_grid_mode_stack_plan_paths.py` -> `5 passed`
- Next step: run ZeGrid réel (F3) pour vérifier disparition des halos en dataset terrain.

### 2026-03-15 10:16 — Iteration 16 (F3 validation terrain ZeGrid + constat mission)
- Retour utilisateur: résultat ZeGrid jugé très bon à l'oeil, mission perçue comme complète.
- Vérifications log/artifacts rapides:
  - `zemosaic_filter.log`: aucun `ERROR` détecté sur la fin de session.
  - artifacts ZeGrid générés à 10:07:
    - `example/out/mosaic_grid.fits`
    - `example/out/mosaic_grid_coverage.fits`
    - `example/out/resource_telemetry.csv`
  - snapshot config confirme DBE actif via GUI: `final_mosaic_dbe_enabled=true`, `final_mosaic_dbe_strength="normal"`.
- Checklist: F3.1 ✅, F3.2 ✅.
- Statut: Q4 (ZeGrid DBE) opérationnel et validé en run réel + validation visuelle utilisateur.

### 2026-03-26 — Pivot intertile: impasse M1 confirmée, M2 implémenté

- Analyse runs D/E: M1 offset-only n'a pas réduit les seams de façon utile sur le dataset hétérogène (variation surtout en offset global, pas en continuité locale visible).
- Décision: M1 rétrogradé en baseline diagnostic; passage en priorité à M2 (gain+offset robuste).
- Implémentation M2 livrée:
  - nouveau solveur robuste `solve_global_affine_v2` dans `zemosaic_utils.py` (ancre, IRLS, rejet paires hors bornes, régularisation gain, clamps gain/offset),
  - nouveau mode intertile config-gated: `intertile_gain_offset_v2` + paramètres (`intertile_gain_prior_lambda`, `intertile_gain_clip`, `intertile_offset_clip`, `intertile_pair_gain_clip`, `intertile_pair_offset_abs_max`, `intertile_max_irls_iters`),
  - logs dédiés: `M2 gain+offset solve ...` + `M2 worst[...]`.
- Config de test préparée dans `zemosaic_config.json` pour run M2 direct:
  - `intertile_gain_offset_v2=true`,
  - `intertile_offset_only_v1=false`,
  - `intertile_photometric_match=true`,
  - `intertile_global_recenter=false`,
  - clips/garde-fous conservateurs activés.
- Prochaine étape: lancer run M2 (C) et comparer métriques/logs vs run E.


## 2026-03-27 13:05 — Smoking-gun confirmé et patch appliqué (weighting Phase 5)

Contexte run G/H (existing master tiles):
- G (weighting ON) montrait des seams amplifiées.
- `tile_weights_final.csv` exposait un écart massif entre `tile_weight_raw` (jusqu'à ~602) et `tile_weight_effective` (~0.91..1.35 avec V4).

Root cause code confirmé dans `assemble_final_mosaic_reproject_coadd`:
- Le pipeline calculait bien `tile_weight_effective` (V4),
- mais au moment d'appeler `reproject_and_coadd_wrapper`, il réinjectait `entry["tile_weight"]` (raw) au lieu de l'effectif.
- Conséquence: V4 était loggé/exporté mais non réellement appliqué au coadd final (domination locale + seams renforcées).

Patch chirurgical appliqué:
- ajout helper local `_resolve_runtime_tile_weight(entry_obj)` qui préfère `tile_weight_effective`, fallback `tile_weight`.
- remplacement des trois points de consommation runtime des poids:
  1) `tile_weights_for_sources` (intertile affine source weighting)
  2) `collect_tile_data` (payload utilisé pour passes suivantes)
  3) `tile_weights_for_entries` (appel principal `reproject_and_coadd_wrapper`)

Validation rapide:
- `python3 -m py_compile zemosaic_worker.py tests/test_phase3_adaptive_invariants.py` ✅
- test source-contract ajouté: `test_phase5_tile_weighting_prefers_effective_weights_when_available`
- `pytest -q tests/test_phase3_adaptive_invariants.py -k "tile_weighting_prefers_effective_weights_when_available or sds_global_gpu_helper_has_oom_retry_with_chunk_tightening or run_hierarchical_master_tiles_bootstrap_avoids_undefined_existing_tiles_symbol"` ✅ (3 passed)

Risque résiduel:
- patch ciblé Classic/Phase5, mais zone partagée large; garder smoke check Classic/ZeGrid/SDS minimal après prochain run terrain.
