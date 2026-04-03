# followup.md

# ZeMosaic — Follow-up exécutable (avril 2026)
## Mission: stabilité cross-platform + filtrage fiable + rendu consistant

Légende:
- `[ ]` à faire
- `[~]` en cours / partiel
- `[x]` validé
- `BLOCKED:` dépendance manquante

---

## 0) Règles d’exécution (non négociables)

- [ ] Travailler **dataset par dataset**, un changement majeur à la fois
- [ ] Conserver séparation **science vs visuel**
- [ ] Ne pas casser le comportement stable `batch size = 0` / `>1`
- [ ] Toute évolution sensible = config-gated + défaut conservateur
- [ ] Preuves obligatoires: logs + artifacts + comparaison avant/après
- [ ] Mettre `memory.md` à jour après chaque itération significative

---

## 1) État actuel consolidé (référence)

### 1.1 Instrumentation et socle
- [x] Exports diagnostics intertile (`intertile_graph_*`)
- [x] `weighted_coverage_map.fits`
- [x] `winner_map.fits` + `winner_index.csv`
- [x] Propagation runtime `intertile_prune_k` / `intertile_prune_weight_mode`

### 1.2 Stabilité mémoire Phase 3
- [x] Working-set adaptatif CPU/GPU
- [x] Watchdog anti-livelock + relance coopérative
- [x] Auto-profiling RAM/swap
- [x] Hotfix `ram_critical_pct` scope

### 1.3 Logging cross-platform
- [~] Patch anti `PermissionError logging.flush()` (spawn Windows) appliqué
- [ ] Validation terrain complète Windows + Linux + macOS

### 1.4 Point de fragilité confirmé
- [~] Quality gate peut rejeter 100% des master tiles sur certains edge-cases
- [~] Mécanisme de recovery ajouté si tout est rejeté (à valider terrain)

---

## 2) Priorités P0 (à finir avant toute nouvelle brique)

## P0-A — Fiabilité run (cross-platform)

Objectif: un run complet ne doit plus échouer pour des raisons de logging/process/mémoire non fatales.

- [ ] Windows: run complet sans spam `PermissionError` logging
- [ ] Linux: run complet sans régression depuis patchs récents
- [ ] macOS: smoke run (ou validation proxy si machine non dispo)
- [ ] Vérifier que `PROCESS_ERROR` n’est plus déclenché par side effects logging
- [ ] Vérifier que les workers enfants ne perdent pas de logs utiles

**Critère GO P0-A**
- 3 runs complets (Win + Linux + 1 autre OS) sans crash infrastructure.

---

## P0-B — Quality gate robuste (anti “all rejected”)

Objectif: filtrer correctement sans tuer le run ni sur-filtrer les datasets difficiles.

- [ ] Vérifier le recovery “all rejected” sur dataset edge-case (M106_5)
- [ ] Logger explicitement:
  - `tiles_evaluated`
  - `tiles_rejected`
  - `tiles_recovered`
  - distribution des scores (p50/p90/p95/max)
- [ ] Ajouter mode de sécurité:
  - `quality_gate_max_reject_fraction` (ex: 0.6 par défaut)
  - au-delà: downgrade en pondération plutôt que rejet dur
- [ ] Introduire zone borderline configurable:
  - `accept` / `borderline(weight_penalty)` / `reject_hard`
- [ ] Conserver `move_rejects` mais éviter qu’il vide tout le set exploitable

**Critère GO P0-B**
- Aucun run ne finit en `run_error_phase3_no_master_tiles_created` à cause d’un gate trop strict.

---

## P0-C — Consistance couleur / photométrie (RGB drift)

Objectif: expliquer et stabiliser les décalages couleur même quand RGB-eq est OFF.

- [ ] Matrice A/B contrôlée (mêmes tuiles):
  1. `center_out=OFF`, `intertile_photometric_match=OFF`
  2. `center_out=ON`,  `intertile_photometric_match=OFF`
  3. `center_out=OFF`, `intertile_photometric_match=ON`
  4. `center_out=ON`,  `intertile_photometric_match=ON`
- [ ] Logger gains/offsets intertile et métriques de dérive chroma (au moins synthèse)
- [ ] Identifier la combinaison responsable du drift perceptif après stretch
- [ ] Définir preset conservateur par défaut si drift non maîtrisé

**Critère GO P0-C**
- Cause principale du drift identifiée + contournement stable documenté.

---

## 3) Priorités P1 (après P0)

## P1-A — Quality gate dynamique dataset-aware

- [ ] Remplacer seuil absolu pur par composant relatif (quantiles/MAD)
- [ ] Ajouter cap de rejet global + fallback pondéré
- [ ] Export rapport par run (`quality_gate_report.json`):
  - threshold_effective
  - reject_fraction
  - borderline_fraction
  - top outliers

## P1-B — Weighting V4 finalisation

- [ ] Ajouter pénalité résidu photométrique par tuile
- [ ] Ajouter pénalité temporelle optionnelle
- [ ] Télémétrie complète avant/après V4
- [ ] Validation OFF vs ON actuel vs ON V4 sur mêmes master tiles

## P1-C — Revalidation graphe photométrique

- [ ] Vérifier pruning effectif sur datasets hétérogènes
- [ ] Confirmer que les arêtes utiles ne sont pas éliminées trop tôt
- [ ] Corréler seams visibles avec winner/coverage/résidus

---

## 4) Protocole de validation standard (obligatoire)

Pour chaque run candidat:
- [ ] Conserver `run_config_snapshot.json`
- [ ] Conserver `zemosaic_worker.log`
- [ ] Conserver preview + FITS + maps diagnostics
- [ ] Générer résumé run:
  - succès/échec
  - temps total
  - incidents (OOM/livelock retries/reject ratio)
  - métriques quality gate
  - métriques photométriques intertile

Comparaison minimale requise:
- [ ] `baseline conservatrice`
- [ ] `candidate`
- [ ] verdict GO/NO-GO basé sur faits (pas uniquement visuel)

---

## 5) Backlog ciblé immédiat (prochaines itérations)

### Sprint 1 (immédiat)
- [ ] Valider fix cross-platform logging (Win + Linux)
- [ ] Valider recovery all-rejected sur M106_5
- [ ] Ajouter logs synthèse quality gate (counts + distribution)

### Sprint 2
- [ ] Implémenter cap `quality_gate_max_reject_fraction`
- [ ] Implémenter mode borderline pondéré
- [ ] Lancer A/B court NGC6888_17 et M106_5

### Sprint 3
- [ ] Matrice couleur A/B center_out/intertile
- [ ] Décision de preset par défaut anti-drift

---

## 6) Definition of Done (mission)

Mission considérée close uniquement si:

- [ ] Stabilité opérationnelle confirmée cross-platform
- [ ] Plus d’échec run causé par “all rejected” quality gate
- [ ] Filtrage jugé cohérent sur au moins 2 datasets contrastés
- [ ] Cause du drift couleur identifiée + mitigation documentée
- [ ] Validation comparative OFF / ON / V4 complète
- [ ] `memory.md` contient la synthèse finale et la décision produit



## Progress update (2026-04-03)

- [x] Export solve/intertile enrichi: `intertile_photometric_solve.csv`, `intertile_residuals.csv`, `intertile_tile_residual_summary.csv`
- [x] Diagnostics solve ajoutés à `intertile_graph_summary.json` (M2/M1 constraints + residual stats)
- [x] Weighting V4 étendu: pénalité résiduelle et pénalité temporelle (config-gated)
- [x] Télémétrie V4 ajoutée: `tile_weights_v4_telemetry.csv` + `tile_weights_v4_summary.json`
- [x] Protocole OFF/ON/ON+V4 formalisé: `validation_protocol_phase5_weighting.md`
- [ ] Validation comparative complète (OFF vs ON vs ON+V4) encore à exécuter
- [ ] Non-régression multi-modes (classic/existing/SDS/ZeGrid) encore à clôturer
