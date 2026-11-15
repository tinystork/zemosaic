# ZeGrid gauge swap — pilotage nocturne G0→G4

Mandat : Tristan autorise toute la mission sur la nuit, revue et corrections comprises.
Point demandé demain sur ce chantier et M33. Ne pas toucher M33 ni les runtimes.
Base : feat/zegrid-adaptive-resources @ e25fb5133768adf4eb26950c63f86b970909b370.
Plan : ZEGRID_GAUGE_BOUNDED_SWAP_PLAN_20261115.md (v3, sections 11–13 prioritaires).

## Checklist
- [x] G0 contrat opérationnel et baseline (ACCEPT LOCAL Junior, décisions Annexe A tranchées)
- [x] G1a capacité Linux (ACCEPT LOCAL Junior)
- [x] G1b admission par groupes (ACCEPT LOCAL Junior)
- [x] G2a/G2b câblage public/UX/Stop et réadmission RAM-only (ACCEPT LOCAL Junior)
- [x] G3 qualification logique + physique isolée, parité et reprise (Nono review-1 ACCEPT)
- [x] G4 clôture : R1 corrigé + testé, fiche de clôture, documentation utilisateur,
      état de pilotage final (rapport `.a2a-reports/ZM-GAUGE-SWAP-G4-20261115.coco.r0.md`)

## Lot actif
Revue Nono de G4 — mission ZM-GAUGE-SWAP-G4-20261115. G4 ACCEPT LOCAL Junior (264 tests).
Rapport attendu : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-GAUGE-SWAP-G4-20261115.nono.r0.md
Dispatch ACCEPTED, runId 0f943863-1a4c-4b89-8945-2875baeb4459, target queued.
Attendre callback exact source, sans polling. Clôture définitive Junior après verdict favorable.
Aucune publication effectuée : pas de push/bump/tag/build (à demander à Tristan).

## Clôture — verdict final

Verdict composite : **ACCEPT local — « qualified with limits »**.

- Qualification : `docs/qualification/zm-gauge-swap-g3/ZM_GAUGE_SWAP_G3_CLOSURE_QUALIFICATION_20261115.md`
  (Q1–Q10 avec libellés exacts, limites complètes, décisions d'architecte).
- Documentation utilisateur :
  `docs/ZEGRID_GAUGE_SWAP_OPTION_UTILISATEUR_20261115.md`.
- R1 (LOW, préexistant) **corrigé** : les champs O1 (`gauge_memory_policy`,
  `a_raw_bytes`, `b_ram_bytes`, `s_credit_bytes`, `credit_needed_bytes`,
  `capacity_sources`) sont produits par un helper unique
  (`_gauge_credit_ram_contract_fields`) appliqué aux six sites
  `_ram_contract_diagnostics` de `_run_single` ; les deux écrasements per-cell
  ne peuvent plus les perdre. Tests :
  `test_g4_ram_contract_helper_publishes_the_whole_o1_observation`,
  `test_g4_every_ram_contract_site_uses_the_shared_helper`.
- OFF/absent reste byte-identique : le helper renvoie `{}` sans crédit.
- Non-régression ciblée G1/G2/G3 : `291 passed` (wiring, g2b, admission,
  capacité, r41 ram_contract, r55 envelope), interpréteur officiel,
  `PYTHONPATH` retiré, Qt `offscreen`.

### Prochaines étapes
1. Revue Junior + Nono du lot G4 (R1, fiche, doc utilisateur, état).
2. **Décision Tristan** sur la publication d'une beta (push `origin/beta` exclu
   d'office : jamais fait par le worker).
3. Dettes de mesure listées dans la fiche (§6) : baseline unique `Q1`/`Q2`,
   `Q7` réutilisation, `Q4` thrash, suites Qt Stop/Resume non exécutées.
4. Ne pas toucher M33, Windows, runtimes ni données pendant la clôture.

## Protocole de continuation Junior
À chaque callback : vérifier provenance/mission/path, lire le rapport, inspecter
le vrai diff/HEAD/status, vérifier les résultats importants. Faire revoir les
contrats risqués et le code par Nono asynchrone ; findings confirmés renvoyés au
même Coco avec rapport r1/r2/r3, sans reset. Après acceptation du lot, reset Coco
pour le nouveau lot indépendant et dispatch suivant avec callback source exact.
Continuer sans redemander confirmation pour le périmètre déjà autorisé. Maximum
3 corrections par lot. Terminer en REPLY_SKIP si nouveau worker lancé depuis
un callback. Clôture ACCEPT/BLOCKED/HUMAN_GATE seulement avec preuves.

## Invariants / limites
90 % du swap disque effectif, pas de plafond fixe Gio, pas de double cache/zram.
Jauge globale uniquement, RAM-only ailleurs, science/backend/identités inchangés.
Pas de commit/push/merge/release/déploiement ; préserver fichiers préexistants.
Ne pas changer la configuration système, swap, swappiness ou limites hôte.
Tests de pression uniquement isolés, bornés et réversibles ; pas d'OOM volontaire
sur l'hôte. Incapacité matérielle = gate explicite, jamais preuve simulée.
Windows/GPU physiques non annoncés qualifiés sans accès et test réels.

## Changements préexistants
M docs/refactor/ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md
?? docs/qualification/ZM-RAM-TILE-TO-EXTERNAL-QUAL-20261114.md
?? docs/qualification/zm-ram-tile-external-qual/
?? docs/refactor/ZEGRID_GAUGE_BOUNDED_SWAP_PLAN_20261115.md
?? docs/refactor/ZEGRID_THROUGHPUT_M33_FLIGHT_PLAN_20261108.md

## G0-ORACLE-3 — absence de capture préalable (constat L6) et golden a posteriori
Constat : `G0-ORACLE-3` (§12 : `plan_hash`, tuples de décision, `aggregate_bytes`,
reason codes d'une fixture synthétique) **n'a pas été capturé avant G1b** comme le
contrat l'exigeait. Aucun fichier golden préexistant n'existe dans le dépôt.
Compensation (capture a posteriori, explicitement étiquetée comme telle) :
- JSON : `docs/qualification/zm-gauge-swap-g1b/g0_oracle3_capture_a_posteriori.json`
  (schéma `g0-oracle3-capture-a-posteriori-v1`, 16 fixtures déterministes OFF) ;
- outil de capture reproductible :
  `docs/qualification/zm-gauge-swap-g1b/capture_g0_oracle3.py` ;
- verrou : `tests/test_zegrid_gauge_admission.py::test_golden_g0_oracle3_capture_a_posteriori_matches_recomputation`
  recalcule chaque cas et exige l'égalité exacte ; toute évolution volontaire des
  fixtures exige une régénération explicite du golden.
Portée : preuve de non-régression OFF, **pas** un oracle pré-enregistré. Aucun seuil
nouveau n'est introduit ; le hash legacy reste `resource_envelope.compute_plan_hash`.
