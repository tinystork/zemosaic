# GPU per-Cell finalization fix — 2026-11-16 (REWORK-1)

État de la mission `ZM-GPU-PERCELL-FINALIZE-FIX-20261116` (beta 5.1.6,
`feat/zegrid-adaptive-resources`, base `67f882c`). Aucun accès Windows/store
réel n'a été effectué depuis cette machine.

## 0. Erratum r0 (correction explicite, rien n'est effacé)

Le rapport r0 affirmait que `run_resume.percell_committed_tiles` lisait
« toujours une carte vide » parce que `percell_tile_px` serait un frère de
`record`. **C'est faux.** Mesure refaite sur un store synthétique réel (9
commits) : le reader imbriqué historique (`payload_meta["record"]
["percell_tile_px"]`) retourne **9/9** tuiles, car le writer de commit injecte
la clé dans `record`; la clé existe **aussi** au niveau supérieur de
`payload_meta` (copie du résultat). L'observation « vide » du r0 venait d'un
appel de test passant un `Path` au lieu d'un store ouvert (exception avalée).
L'élargissement du reader reste (robustesse inoffensive aux deux formes), mais
il n'est **pas** la cause du `PipelineAccountingError` de reprise partielle :
celle-ci était la borne de réservation run-wide, corrigée en §3.

## 1. Checklist

- [x] Cause racine reproduite sur la vraie entrée produit (hash v1 strict vs
      couple référence/max hétérogène).
- [x] Hash d'audit hétérogène déterministe (schéma v2) ; `active_final_plan_hash`
      v1 intact et toujours strict ; OFF/legacy/CPU byte-identiques.
- [x] Bloc `gpu_tile_decision` recadré sur la décision de référence.
- [x] **M1** reprise sur la VRAIE forme de crash finalisation (génération
      staging + `.staging.json` valide à owner PID mort + FITS science
      seulement, ni manifest ni pointeur) : publication réussie, zéro
      cellule/cache/gauge, commits/admission/caps inchangés, no-op au rejeu
      répété. Refus/préservation des cas live/foreign/autre-run couverts.
- [x] **M2** aucune tuile planifiée étiquetée « mesurée » : preuve durable
      exigée pour CHAQUE Cellule d'un rejeu complet, refus qualifié sinon (pas
      de `CURRENT`, pas de products, commits intacts) ; provenance par entrée
      (`executed_this_attempt` vs `frozen_plan_after_durable_caps`).
- [x] **M3** commentaires/lecteur/doc/rapport corrigés.
- [x] **L1** docstring du hash exacte (référence stricte + enveloppe run-wide
      vérifiée séparément, jamais « validée par le scheduler »).
- [x] **L2** `plan_hash_origin='recomputed_at_replay'` + note, aucun télémetrie
      historique inventée.
- [x] **L3** capacité du ledger host par-Cellule (≤ budget admis), test
      d'invariant + contre-exemple reproduit.
- [x] **L4** gardes HARD gauge + cache (pas seulement stack) dans le test de
      reprise complète.
- [x] Vérification indépendante Junior : 59 passed (161,51 s), empreintes des
      4 fichiers inchangées ; revue Nono review-1 ACCEPT, 4 tests ciblés passés.
- [x] Acceptation locale Junior (2026-11-16), mission close, patch non commité.
- [ ] Publication bêta et essai Windows : étapes séparées, non effectuées.

## 2. Symptôme Windows (lecture seule)

90 Cellules attendues, 90 commits GPU COMPLETS, IDs exacts ; assemblage
5,188 s ; `state.finalized=false`, `outputs={}`, aucun `CURRENT.json`,
science de staging gen1 seulement.

```
_run_single:26869 -> _r55_build_active_block:16894
  -> scheduler.active_final_plan_hash:4037
mapping.gpu_streaming_tile=320 != candidate.tile_px=32
```

## 3. Cause racine et défauts de cohérence

Le runner publiait `mapping.gpu_streaming_tile = max(tuiles effectives)` alors
que le candidat restait celui de la Cellule de **référence** ; le hash v1 exige
`mapping == active_gpu_mapping_for_candidate(candidate)` et refuse à raison.

Défauts distincts révélés par l'hétérogénéité :

1. v1 ne peut pas exprimer un run hétérogène (référence + tuiles effectives).
2. `_r55_gpu_tile_decision_dict` indexait les tables de la référence avec le
   tuile max d'une autre Cellule (`KeyError`).
3. Reprise complète : décision effacée → **aucun** bloc scheduler publié.
4. Reprise partielle : crédit host/disk run-wide (référence) < borne réelle
   d'une Cellule plus profonde → `PipelineAccountingError` (host puis disk),
   et capacité du ledger bornée par le crédit de référence.

## 4. Conception (rework inclus)

* `scheduler.active_final_plan_hash` : inchangé, strict, utilisé pour
  l'homogène. `percell_tile.final_plan_hash` (v2) l'appelle pour la moitié
  référence, puis lie le plan gelé et les tuiles effectives (couverture exacte,
  rung déclaré, jamais de croissance), et vérifie séparément l'enveloppe
  run-wide (`gpu_tile == max effectif`, K=1, prefetch/workers).
* Finalisation : schéma v1 ou v2 selon le plan ; `gpu_tile_decision` garde sa
  portée de référence + champs `scope/reference_tile_px/
  run_level_max_effective_tile_px/percell_tiles_differ`.
* Reprise complète : `_r55_freeze_complete_replay_decision` gèle la décision
  de référence + le plan + les tuiles **mesurées** des commits (strict : toute
  Cellule sans preuve ⇒ `_R55PerCellTileEvidenceError`, rien n'est publié) et
  reprojette l'enveloppe par arithmétique pure. Le bloc publié porte
  `plan_hash_origin=recomputed_at_replay`.
* Crédit pipeline : réserves host/disk **par Cellule** ; capacité host du
  ledger =
  `min(budget_admis, max(crédit_référence, pire_réserve_par_Cellule) × queue_max)`
  — jamais au-dessus du budget admis, aucun budget augmenté.
* Provenance des tuiles : `executed_this_attempt` pour les Cellules empilées,
  `frozen_plan_after_durable_caps` pour les Cellules restaurées (jamais
  « mesuré »).

## 5. Preuves de non-régression (commandes exactes)

* `python -m pytest tests/test_zegrid_gpu_percell_tiles.py -q`
  → **40 passed** (36 existants + 4 rework), 87 s.
* `python -m pytest tests/test_zegrid_resume_complete_gpu_rerun.py
  tests/test_zegrid_gpu_partial_resume.py
  tests/test_zegrid_publication_execution_m5b.py
  tests/test_zegrid_gpu_durable_integration.py -q` → **19 passed**, 77 s.
* `python -m pytest tests/test_zegrid_r55e2_active_gpu_prefetch.py
  tests/test_zegrid_r55e11_runtime_vram_adaptation.py
  tests/test_zegrid_r55d2_cpu_active.py
  tests/test_zegrid_per_cell_scheduler.py -q` → **156 passed**, 159 s.

Échecs **préexistants** avec preuve brute conservée (worktree détaché propre à
`67f882c`, puis arbre courant, mêmes 3 fichiers) :

* `.a2a-reports/ZM-GPU-PERCELL-FINALIZE-FIX-20261116.r1.baseline67f882c.txt`
  → `14 failed, 54 passed`.
* `.a2a-reports/ZM-GPU-PERCELL-FINALIZE-FIX-20261116.r1.worktree.txt`
  → `14 failed, 54 passed`.
* `diff` des lignes `FAILED` : **identiques** (aucune régression attribuée à ce
  correctif). Les échecs portent sur des scénarios de pression/gauge injectée
  (`r55e1_run_checkpoints`, `resume_telemetry_truth`) et une constante legacy
  0.5 vs 0.7 (`r55e10`), hors périmètre modifié.

## 6. Prérequis de reprise manuelle (Windows)

1. Ne jamais éditer, déplacer ni supprimer le store réel, les commits, la
   science de staging ni un éventuel `CURRENT.json`.
2. Reprendre avec la MÊME version corrigée, la MÊME identité scientifique et
   l'option `zegrid_gpu_percell_tiles` ON (store v2 gelé refusé si OFF).
3. Run entièrement commité ⇒ zéro empilement ; assemblage depuis les commits
   immuables puis génération atomique. `executed_this_attempt=0` attendu.
4. La génération de staging laissée par le crash doit être celle du MÊME run,
   avec un `.staging.json` valide et un owner PID mort ; elle est alors nettoyée
   par le nettoyage conçu et remplacée par une nouvelle génération. Un staging
   vivant, étranger, ou d'un autre run est refusé et **préservé**.
5. L'admission GPU de référence est encore ré-évaluée avant que la reprise
   n'apprenne que 0 Cellule est pending : l'hôte doit encore satisfaire
   l'admission gelée (VRAM libre ≥ plan admis, tuile ≤ tuile gelée).
6. Un run déjà finalisé + publication vérifiée est un no-op vérifié.

## 7. Limites

* Télémétrie appareil de la tentative originale non reconstructible (elle n'a
  jamais été publiée) : le rejeu publie le hash exact recalculé, le plan gelé,
  les tuiles commitées et l'enveloppe, et le déclare (`plan_hash_origin`).
* La Cellule réelle de référence n'est pas persistée ; aucun identifiant n'est
  fabriqué.
* Profil stockage `dense` : la borne par Cellule peut dépasser le crédit de
  référence (n×aire) ; la capacité du ledger le couvre désormais sans dépasser
  le budget admis. Profil `bounded` : la référence (plus grande aire) majore
  déjà toutes les Cellules.
* Aucun essai Windows/M33/SH2 : `NOT_RUN`. Aucune exécution CUDA physique.

## 8. Clôture Junior

ACCEPT local après revue Nono review-1 et vérification finale HEAD/branche/statut,
empreintes et diff check. Aucun changement code après le gate indépendant 59 PASS.
Contrôles stricts conservés ; réserves LOW du test (assertion tautologique, faible
risque de recyclage du PID) non bloquantes et consignées. Les 14 échecs des autres
suites restent explicitement hors gate vert, sans déclaration globale zéro régression.
Rapport complet : `.a2a-reports/ZM-GPU-PERCELL-FINALIZE-FIX-20261116.junior.acceptance.md`
dans le workspace.

Windows réel : **90 cellules**, toujours NOT_RUN et intact. Le test synthétique
emploie 9 cellules. Une reprise écrit une publication : ce n’est pas une opération
lecture seule. Une éventuelle qualification physique devra être séparément autorisée
et protéger l’original, idéalement sur copie complète. Aucun commit/push/bump ni
changement des politiques RAM/VRAM effectué.

## 9. Publication beta autorisée — 2026-11-16

Tristan autorise la publication du correctif accepté sur `beta`, explicitement
sans bump de version. Version conservée : **5.1.6**. Périmètre : les quatre
fichiers code/tests vérifiés ci-dessus et ce document ; les autres travaux
locaux restent exclus. Aucun tag/release ni modification du run Windows.
