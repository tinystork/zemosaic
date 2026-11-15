# GPU per-cell product — 2026-11-15

Mission: ZM-GPU-PERCELL-PRODUCT-20261115

- [x] Tristan autorise intégration GUI expérimentale défaut OFF puis beta après gates/revue, test Windows manuel sans CLI.
- [x] Baseline feat/zegrid-adaptive-resources HEAD/origin-beta 6df6a50d95ae82e1e887b85394d20a6c75c8ba0a (5.1.5), préexistants conservés.
- [x] Gate baseline lancé ; 5 échecs sélectionnés constatés en cache pytest, sortie tronquée : aucun compte exact revendiqué. Coco doit préciser la causalité avant modifications.
- [x] Coco reset nouvelle mission déjà effectué et vérifié Reset seq1 ; aucune mission reçue lors de première tentative environnement indisponible. Ne pas reset à nouveau.
- [x] Mandat durable créé sous .a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.mission.md.
- [x] Dispatch Coco ACCEPTED/queued, runId 49553419-e178-4a30-9b78-eabcf0669cfc.
- [ ] Implémentation r0, rapport puis callback exact source.
- [ ] Vérification Junior puis Nono read-only, corrections ciblées sans ping-pong.
- [ ] Publication beta par Junior après gates.
- [ ] Essai physique Windows par Tristan, sortie neuve.

Rapport attendu: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.coco.r0.md

Qualification précédente harnais ACCEPT local ; Windows physique NOT_RUN, node raw-command-mismatch. Aucun gain GPU promis. Aucun changement runtime/M33/SH2 référence/sauvegardes, aucun commit/push/bump par Coco. Toutes les docs/modifications préexistantes protégées.

Prochaine étape: attendre COCO_REPORT_READY puis vérification indépendante Junior, sans polling ni reset.

---

## Implémentation r0 (Coco, 2026-11-15)

### Décisions de conception

| Sujet | Décision |
| --- | --- |
| Clé persistée | `zegrid_gpu_percell_tiles: bool = False` (DEFAULT_CONFIG + politique stricte + GUI). Absente = OFF. |
| Portée | Uniquement ZeGrid **ACTIVE** + **GPU**. GPU OFF gagne toujours ; la case est désactivée mais sa valeur enregistrée n'est jamais effacée. |
| Planification | Chaque cellule reçoit sa propre échelle ACTIVE (`choose_gpu_tile_size(n_i,3,hw_i,budget)` → `_r55_active_gpu_tile_ladder` → élagage E10 exact), ses cartes device/hôte exactes et sa réserve. Aucun couple `max(n)` × `max(area)` en ON. |
| Admission | La cellule est admise sur la plus grande tuile dont (a) le workspace device exact tient dans la sélection E10, (b) le pic hôte projeté exact (shared une fois + stack ajouté + owner producteur réel à prefetch≥1 + crédit résident réel) tient dans le budget avec la marge 15/100, (c) la réserve mesurée est un **plancher** explicite. |
| Référence scheduler | La décision scheduler (infrastructure) est construite sur **une cellule réelle** : celle dont le pic hôte admis est le plus grand (départage aire/profondeur/cell_id). La décision référente est recalculée si le plan désigne une autre cellule. |
| Exécution | Un état de tuile **par cellule** ; revalidation fraîche RAM/VRAM avant chaque unité ; rétrécissement strictement sur l'échelle déclarée de la cellule ; aucune croissance, aucun second scheduler, K=1. |
| Reprise | Extension optionnelle `percell_tiles` dans `admission` ; `schema_version="2"` uniquement quand elle est présente. v1 reste lu à l'identique ; v2 est strict, couverture exacte vérifiée contre les memberships. |
| Store gelé ON + case OFF | **Refus explicite** avant exécution/écriture (jamais de réinterprétation). |
| Store gelé v1 + case ON | Reste **legacy** avec diagnostic explicite (aucun replan silencieux). |
| Shrink durable | La tuile réellement exécutée est écrite dans le commit (`percell_tile_px`) ; une reprise plafonne chaque cellule commitée à cette tuile (minimum, jamais croissance). |
| Migration | Aucune migration de store réel. |

### Fichiers

* `src/zemosaic/core/zegrid/percell_tile.py` (nouveau) — plan pur, parseur strict, caps de reprise.
* `src/zemosaic/core/zegrid/run_resume.py` — admission v2 optionnelle, préservation dans `attach_working_set`, lecteur des caps commités.
* `src/zemosaic/zemosaic_zegrid_mode.py` — wrapper de décision ON/OFF, apply par cellule, exécution par cellule, manifeste, refus de reprise, propagation.
* `src/zemosaic/zemosaic_config.py` — clé, validation booléenne stricte.
* `src/zemosaic/zemosaic_gui_qt.py` — case « Tuiles GPU adaptatives par cellule — expérimental », tooltip, transport GUI.
* `src/zemosaic/zemosaic_worker.py` — transport explicite des clés de politique ajoutées (`gpu_percell`, `stack_phase_profiling`, `gauge_counts_decoupling`).
* `src/zemosaic/locales/en.json`, `fr.json` — libellés EN/FR.
* `tests/test_zegrid_gpu_percell_tiles.py` (nouveau) — 33 tests (dont régressions rework-1).
* `tests/test_zegrid_r55e7...`, `r55e11...`, `test_zegrid_resume_complete_gpu_rerun.py` — stubs alignés sur le contrat produit (champs `gpu_parity_qualified`, `gauge_counts_*`) et seam CPU du noyau GPU réellement utilisé.

### Validation locale (hôte sans GPU)

* `PYTHONPATH=src QT_QPA_PLATFORM=offscreen python3 -m pytest -q` sur le sous-ensemble baseline (3 fichiers) : **53 passed** (baseline initiale : 5 failed / 48 passed).
* Suite affectée combinée (ladder, host bound, partial resume, per-cell, e11, e2 prefetch, e8, admission fallback, gauge counts) : **184 passed**.
* Nouveau fichier per-cell : **33 passed**, incluant plan hétérogène, refus, schéma v1/v2 strict, propagation Qt→worker→engine, exécution de bout en bout avec noyau GPU simulé, science OFF/ON bit-identique (FITS), reprise ON/OFF.
* Tests GUI policy : 57 passed, 1 échec **préexistant** (`tile_weight_v4_strength` absent de la signature worker, indépendant de cette option) ; test tiny run isolé non concluant pour la même raison de dérive préexistante.
* Stop/resume Qt : 104 passed, 2 tests sensibles à l'ordre ; chacun passe seul (40 s et 17 s), aucun chemin per-cell n'est emprunté quand l'option est OFF.

### Limites

* GPU physique Windows : **NOT_RUN** (propriété de Tristan). Aucun gain en secondes revendiqué.
* La parité device↔device à tuile variable n'est pas prouvée.
* La sélection de la cellule de référence maximise le pic hôte admis ; un cas pathologique où une autre cellule aurait un crédit prefetch plus grand n'est pas couvert par une preuve physique.

### Prochaine étape

Vérification indépendante Junior (gates locaux, revue du schéma v2 et des tests), puis revue Nono en lecture seule. Publication bêta par Junior seulement après acceptation. Essai manuel Windows par Tristan selon `ZEGRID_GPU_PERCELL_BETA_GUIDE_20261115.md`.

## Junior r0 independent review
Coco implementation callback verified. Gate independent142PASS/155.40s, diffcheckPASS, HEAD6df6a50unchanged. NOT ACCEPTED: reservation floor lost at runtime (4GiB admitted vs required6.33GB) and four incoherent admission cases accepted reproduced, proof .a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.junior.r0-repro.json. Pending-cell shrink persistence gap by source; no stop/crash regression proving durable cap. Review Nono r0 ACCEPTED queued run3ed31fc7-fb74-4944-83c0-f4708d69e631. Await callback, no polling/reset. Beta NOT published, Windows NOT_RUN, M33/sauvegardes intacts.


## Junior adjudication Nono review-0: REWORK-1
- [x] Nono callback/provenance/mission/report checked; FINDINGS confirmed independently.
- [x] D1 reservation-floor runtime gap and four D2 corrupt contracts reproduced; D3 pending-cell shrink durability gap confirmed by source (crash regression required).
- [ ] Coco rework-1: D1/D2/D3 + Q1/Q2/Q3 bounded followups, report coco.r1.md.
- [ ] Junior independent gates, then Nono followup. Beta NOT published, physical Windows NOT_RUN.
Report: .a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.junior.review0.md. Same mission/session, no reset. Corrective iteration1/3.

Coco REWORK-1 dispatch ACCEPTED/queued runId fff54170-1346-4065-8522-fd042886b538, same agent:coco:main preserved. Report expected .a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.coco.r1.md. Await exact-source callback without polling/reset; then independent verification and Nono followup.

---

## REWORK-1 corrections (2026-11-15, Coco r1)

Nono review-0 findings + Junior adjudication: D1 HIGH, D2/D3 MEDIUM, Q1/Q2/Q3.

| Item | Correction | Evidence |
| --- | --- | --- |
| **D1** runtime lost reservation floor | One authoritative `PerCellEntry.admitted_host_bound(tile) = max(raw model bound, reservation − baseline)`; used by PLAN, APPLY and the module-level runtime context factory `_r55_percell_runtime_context`. The context no longer exports raw `host_bound_by_tile`. | `test_d1_runtime_context_enforces_the_reservation_floor_at_every_rung`: every rung refuses a budget one byte below its floor with abundant VRAM. |
| **D2** incomplete v2 validation | `plan_from_admission` now refuses: selected `device_workspace_bytes` ≠ map[tile]; selected peak > selection; `host_aggregate_bytes` > host budget; `selection_budget` > `device_budget`. `validate_admission` refuses CPU backend + per-cell extension. Strict integer parsing happens before arithmetic. Nonselected rungs are not rejected merely for exceeding selection. | four dedicated regressions + CPU extension test. |
| **D3** pending shrink not durable | A separate atomic `zegrid_percell_shrink_caps.json` (schema `zegrid-percell-shrink-caps-v1`) is written **before** the shrunken unit executes, even when the Cell is still pending; repeated shrinks keep the monotone minimum. Resume reads it, validates run id/policy/every Cell/ladder/planned tile strictly, merges with committed-tile caps, and seeds the start tile. Persistence failure raises; it never silently proceeds. | `test_d3_stop_after_shrink_before_commit_restarts_at_the_cap` removes a Cell's commit after writing its cap and verifies the resumed effective tile ≤ cap while all other commits stay byte-identical. |
| **Q1** device budget provenance | `_R55GpuActiveDecision.gpu_device_budget_bytes` is now the ACTUAL VramPolicy 70 % budget; the planner no longer reconstructs it from the qualification percent. Per-Cell declared ladders are pruned with the exact product E10 rule (declared rung only if its exact device workspace fits the 85 % selection budget). | plan tests + full ladder/host/VRAM suites. |
| **Q2** producer/prefetch owners | Production planning now supplies REAL per-Cell cache-owner and prefetch-residency maps (derived from each Cell's patch/depth and the coherent cache worker count). The scheduler reference Cell is the real Cell carrying the largest owner/residency facts, so run-wide cache/prefetch credit covers every Cell without a fictitious `max(n)×max(area)`. | `test_q2_per_cell_producer_and_prefetch_owners_are_priced_per_cell`. |
| **Q3** OFF payload bytes | `percell_tiles` is omitted (not `null`) from attempts, run block, run telemetry and scheduler block when OFF. | OFF attempt + OFF manifest witnesses. |

Corrections are bounded; no margin/budget relaxation, no second scheduler, no science/geometry change, no commit/push.

### Validation rework-1

* Core gate (ladder + host bound + partial resume + per-cell): **86 passed**.
* Neighbor gate (e11 + e2 prefetch + e8 + admission fallback + gauge counts): **110 passed**.
* GUI policy + scheduler: **58 passed**.
* Qt stop/resume: **78 passed**, 2 order/infrastructure failures (local `_observe` function cannot be pickled by a pool; no per-Cell path when OFF). The per-cell D3 crash-after-shrink test passes.
* `py_compile` and `git diff --check`: PASS.

Windows physical GPU remains **NOT_RUN**. Next: Junior independent gate, then Nono follow-up.

## Junior independent rework-1 gate / Nono review-1
- [x] Coco r1 callback verified; D1/D2 guards and D3 write-before-execute inspected; diff check PASS, 12 source/test SHA256 unchanged through gate.
- [x] Combined independent 254 tests: 253 PASS/1 FAIL190.20s; failed GUI tiny-run rejected explicit host budget423339622 B. Isolated first FAIL, published HEAD isolated PASS8.37s then candidate isolated PASS6.35s under improved available RAM. Resource-sensitive, not all-green combined nor proven regression.
- [x] New percell suite33 tests PASS within combined gate; physical GPU NOT_RUN.
- [x] Nono review-1 accepted queued runId eee14056-572d-46cf-90c9-2d8b81a1591e; report .a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.nono.r1.md expected. Q2 heterogeneous dense owner/reference + D3 injected-cap test limitations explicitly requested.
- [ ] Await callback without polling, adjudicate then accept or bounded rework (iteration1/3). Beta NOT published; no commit/push/bump/node/runtime/M33.
Evidence: .a2a-reports/ZM-GPU-PERCELL-PRODUCT-20261115.junior.r1.md and .sha256.

## Adjudication finale Junior — ACCEPT pour bêta expérimentale 5.1.6

- [x] Reprise explicitement demandée par Tristan ; push beta autorisé.
- [x] Nono review-1 terminé ACCEPT ; rapport durable lu. History confirme une
  erreur de livraison du callback SESSION_WORK_START_CHANGED (pas de revue en cours).
- [x] 12/12 empreintes source/tests identiques au gate indépendant ; HEAD6df6a50
  et origin/beta identiques avant publication ; main af49d7c inchangé.
- [x] D1/D2/D3 clos après inspection indépendante ; observations LOW acceptées
  pour expérimentation défaut OFF et documentées dans le guide.
- [x] Gate antérieur : 253/254 PASS, puis GUI tiny-run PASS isolé sur candidat
  et témoin. Pas de prétention suite combinée intégralement verte.
- [x] Comparaison Qt de publication : candidat 2 FAIL (refus mémoire477079142B,
  fonction locale non picklable) ; témoin HEAD 1 FAIL (stop non acquitté),
  1 PASS. Cas pickling candidat répété seul PASS14.10s. Ces résultats ne prouvent
  PAS les deux mêmes échecs préexistants ; ils montrent des limites ressources/ordre.
  Fichier de test inchangé (blob35542144f6b4239f4b1d409d4d1cf25355fb26e0).
- [x] Science GPU physique / Windows : NOT_RUN, aucun gain garanti.
- [x] Gate packaging+titre après bump : 20 passed en 5.19s ; diff check PASS.
- Publication autorisée : commit ciblé et push fast-forward origin/beta par Junior.

Preuves Qt : /tmp/zm-percell-release-qt-{candidate,baseline,isolated}.log.
Aucun run node/Windows/M33/SH2 lancé. Préexistants hors périmètre conservés.
