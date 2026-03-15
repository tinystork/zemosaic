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
