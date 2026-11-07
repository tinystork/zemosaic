# ZeGrid Adaptive Resources — fiche de reprise

Mise à jour : 2026-11-07  
Programme : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107`  
État : **AUDIT_FINDINGS — qualification produit incomplète**
Contrat : [plan v2, sections 10–14](ZEGRID_ADAPTIVE_RESOURCE_ENGINE_PLAN_20261107.md)

## Verdict courant après audit transversal (2026-11-07)

**L'audit est terminé; les gates produit A4/A5 ne sont pas satisfaits et A6 n'est pas qualifié.**
Les acceptations et le journal ci-dessous sont historiques, supersédés pour leur portée produit
par le [rapport final d'audit](../qualification/zegrid-adaptive-resources/POST_PUBLICATION_AUDIT_20261107.md).
A2/A3 ont des primitives valides mais des manques d'intégration; A1d reste différé.
54 tests ciblés passent; le comparatif réel5150040/418ba43 reproduit les mêmes9 échecs connus.
Pas de modification produit ni nouvelle publication pendant l'audit.
Le run Windows M106 `mem_fix` confirme GUI/ACTIVE/GPU70 % et publication, mais seulement en profil
`dense`/`ram_tile`; il révèle aussi F11–F14 (identité de reprise, télémétrie et faux WARN).
**Prochaine action unique : rework borné « reprise et alertes honnêtes » (F11–F14), puis contrat
admission/provider et choix automatique sous-ligne (F2/F3).** Différer le gros M33.

## Autorisation réelle

- Lancement complet A0→A6 reçu de Tristan le 2026-11-07 : « tu as les commandes,
  rendez-vous à la fin pour un test sur le GUI ».
- Publication sur `beta` autorisée **en fin de mission seulement**, après
  acceptation A6 ; aucune publication intermédiaire.
- Gate humain final : test GUI interactif à la fin, non simulé par Junior.
- Portée : progression séquentielle A0→A6 dans le périmètre du plan, commits
  locaux autorisés dans le worktree de mission ; push beta réservé à la fin.
- Aucun push/main, tag, release, installation/remplacement runtime, écrasement
  d'anciens résultats, migration destructive, changement scientifique ou
  nouvelle dépendance majeure sans décision explicite.

## Localisation et baseline

| Champ | Valeur vérifiée à la préparation |
| --- | --- |
| Dossier documentaire d'entrée | `/home/tristan/.openclaw/workspace/projects/zemosaic/docs/refactor/` |
| Worktree d'implémentation de cette mission | `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources` |
| Branche d'implémentation de cette mission | `feat/zegrid-adaptive-resources` |
| SHA de départ envisagé | `515004078cfb64b327fb02aeaa323e48cca62bd1` (ref locale origin/beta) |
| Sources auditées | `/home/tristan/.openclaw/workspace/worktrees/zemosaic-auto-external-streaming` |
| État des sources auditées | HEAD `5150040`, propre lors de la vérification ; ancien worktree de mission, ne pas réutiliser implicitement |
| Checkout documentaire ancien | `/home/tristan/.openclaw/workspace/projects/zemosaic`, HEAD `ec708a1`, documents locaux modifiés/non suivis |
| Source active unique | `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources` |
| Index de preuves d'exécution | `docs/qualification/zegrid-adaptive-resources/` (worktree de mission) — créé, en cours de remplissage A0 |

Les champs non créés sont intentionnels avant lancement, pas des preuves manquantes
à inventer. Tous les champs requis par le gate A0 devront être remplis avant A1.

## Tableau de progression

| Lot | Statut | Tâche en cours | Verdict Junior / preuves |
| --- | --- | --- | --- |
| A0 | ACCEPTED | A0-T01 | Coco r0+r1+r2, revue Nono FINDINGS→corrigé, vérifié Junior |
| A1 | ACCEPTED (a+b+c) ; A1d DÉFÉRÉ | A1-T02 | a+b+c acceptés (5 commits) ; A1d → mission séparée après A2 |
| A2 | ACCEPTED | A2-T01 | A2a provider + A2b câblage source (8e56013+83f2dde) + A2c/A2d Q20/Q21 (89e1391) ; 304+ tests verts |
| A3 | ACCEPTED (Q30+Q31) ; Q32 BLOCKED (Windows) | A3-T01 | 2D micro-blocs (b61f9fe) + GPU externe (2ce487c) + fallback transactionnel (2d065ff) + témoins Q30 (bab2b0b) et Q31 (0c2af51+1014e30) ; Q32 (cellule M33) → A6/Windows |
| A4 | ACCEPTED (Q40+Q41) | A4-T01 | 8 commits : allocateur 11 plans (c07270f), banded_fits (9548f19), canvas_backing (476465f), câblage banded (04351a6), guarded scratch (bdbdc06), fallback production + discard (f097144), Q40 (380dcd2), Q41 (15d999e) |
| A5 | ACCEPTED (Q50+Q51) | A5-T01 | croissance bornée inverse du shrink (39e350c) + Q50 (5 tests) ; Q51 couvert par resume_m4/p0_resume_integrity/m3_recovery (79 tests verts) |
| A6 | RUNNING (Q60+Q64) ; Q61 BLOCKED | A6-T01 | Q60 vrais runs E2E 1/2/4 Gio bit-identiques (e9f8729, 4 tests) ; Q64 propagation GUI headless (15 tests) ; Q61 M33 Windows + Q62 Caldwell + Q63 M106/M16 reportés au test direct beta de Tristan |

**Dernier gate accepté : A5_ACCEPTED (Q50+Q51).** A6 en cours (Q60/Q64 faits,
Q61 BLOCKED, Q62/Q63 reportés au test direct beta).

## Suivi worker à renseigner seulement au dispatch réel

- Tâche stable : aucune.
- Worker / session : aucun assigné à ce programme.
- Statut dispatch : aucun.
- Source Junior exacte de callback : non établie ; utiliser la provenance réelle.
- Rapport attendu : aucun.
- Itération corrective : 0 ; maximum 3 par tâche indépendante.
- Dernière preuve de fin d'activité : sans objet pour ce programme.
- Ne pas renseigner une session supposée ou un faux identifiant de run.

## Décisions figées et limites

- Tous les contributeurs et la science conservés ; pas de modification automatique
  du halo/rejet/dtype, pas de sous-stacks scientifiques approximatifs.
- M0–M6 sont des acquis à réutiliser, pas à refaire.
- Objectif petit budget : 1–32 Gio selon le contrat §12, socle moteur inclus.
- Exécution CUDA physique requise, ainsi que sortie FITS finale et grand canevas.
- Les coefficients de la référence ne prouvent pas la faisabilité de ZeGrid.
- A0 fige corpus, mesures, seuils, oracles et commandes avant A1 ; aucun gate
  assoupli après échec.
- Progression automatique après lancement complet seulement si le lot précédent
  est accepté ; arrêt pour dépassement du périmètre ou preuve indispensable absente.
- Pas de suite exhaustive ni de run M33 complet relancés à chaque correction.

## État Windows et données

- Témoin M106 `C:\\Users\\JarvisJunior\\Desktop\\zemosaic out\\M106\\mem_fix` inspecté :
  terminé, 66/66 frames, ACTIVE/GPU, profil dense/ram_tile, sorties racine vérifiées.
- Ce témoin ne force ni external, ni scratch, ni OOM, ni croissance dynamique; il ne qualifie pas
  les grands corpus. Alertes matérielles ajoutées au rapport final sous F11–F14.

- Dossier communiqué : `C:\Users\JarvisJunior\Desktop\M33\ircut`.
- Dernière tentative de lecture : `Invalid system.run request: raw-command-mismatch`.
- Tristan a signalé Windows planté ; cause non diagnostiquée ici.
- Connectivité actuelle, runtime et derniers logs : **NON REVÉRIFIÉS**.
- N=21 818 vs ancien inventaire 22 414 : à réconcilier en A0, pas à choisir au hasard.
- Espace scratch disponible, chemins sources réels et empreintes : à relever en A0.
- Aucun runtime/dossier de données Windows modifié par cette préparation.

## Prochaine action antérieure — historique, supersédée par l’audit

A4-T01 : assemblage 121 B/px + écriture FITS transactionnelle/hachage streaming (Q40 canevas > budget, Q41 finitions/preview disque presque plein). Q32 (cellule M33 profonde) reporté à A6/Q61 (Windows indisponible).

Après lancement complet : consigner son autorisation et sa portée ; vérifier la
baseline et l'absence de conflit de travail ; créer le worktree isolé ; y copier
le contrat v2 et cette fiche ; inscrire sa localisation active ici ; préparer
puis déléguer la première tâche A0 bornée selon AGENTS.md. Aucun travail A1
avant A0_ACCEPTED. Si Tristan lance seulement A0, ne pas extrapoler à A6.

## Journal documentaire

- 2026-11-07 : A6 en cours — Q60 vrais runs E2E synthétiques 1/2/4 Gio
  herméticisés (e9f8729, 4 tests) : FITS+manifest valides, coverage > 0, science/
  coverage bit-identiques entre budgets (la preuve petit budget est un run réel,
  pas un dry-plan). Q64 : propagation GUI→backend headless vérifiée
  (test_dispatch_propagation_witness 15 tests + r53c_gui_halo + r55e4_gui_gpu_toggle).
  Q61 (M33 Windows GPU) BLOCKED : corpus Windows indisponible. Q62 (Caldwell
  halo 128) et Q63 (M106/M16) : corpus réels présents, runs lourds reportés au
  test direct beta. NB : 2 tests GUI policy (`m6_gui_policy`) échouent AUSSI à
  bab2b0b (préexistant).

- 2026-11-07 : A5 ACCEPTED (Q50+Q51) — croissance bornée `next_grow`/`grow_sequence`/
  `assert_grow_only`/`assert_joint_grow_only` + `GROW_ORDER` (inverse exact du
  shrink : tile_px -> units_in_flight -> workers_per_unit -> prefetch_units,
  borné par les mêmes contraintes d'énumération, sans promotion CPU->GPU — le
  backend reste une porte à sens unique L2). Commit 39e350c. Q50 : 5 tests verts
  (inverse exact depuis le maximum du treillis, gardes refusent tout shrink,
  budget bas n'achète jamais plus). Q51 : 79 tests verts resume_m4/p0_resume_integrity/
  m3_recovery (corruption isolée/refusée, zéro double comptage). NB : 6 tests
  `resume_m1_conservation` échouent AUSSI à bab2b0b (préexistant).

- 2026-11-07 : A4 ACCEPTED (Q40+Q41) — 8 commits. (1) `assemble_canvas_feathered`
  adosse désormais les 11 plans canoniques (121 B/px) à l'allocateur, y compris
  science_work/science_out/valid_fraction/hole_mask, et écrit support_w1/w2/n_eff
  en place (bit-identique). (2) `banded_fits.py` : écriture FITS science/coverage
  par bandes de lignes, byte-identique au chemin dense `PrimaryHDU` (bloc header
  mandatoires d'abord, data big-endian, SCIHASH = sha256 des octets CHW
  little-endian sanitisés, compte non-fini). (3) `canvas_backing=scratch` tarife
  le canevas en disque (scratch) au lieu de RAM. (4) `_assemble_cells_guarded`
  accepte scratch + scratch_dir, retourne l'allocateur. (5) `_run_single` retombe
  en scratch quand le canevas anonyme ne tient pas, et libère le scratch après
  sérialisation. Q40 (380dcd2, byte-identique) + Q41 (15d999e, finitions jamais
  désactivées en silence sur disque presque plein). 130 tests verts (banded,
  bounded, publication, e2e). NB : `test_explicit_h8_array_parity_with_accepted_r53a
  [per_cell]` échoue AUSSI à bab2b0b (préexistant, pas une régression A4).

- 2026-11-07 : A3 ACCEPTED (Q30+Q31) — livrables 2D micro-blocs (b61f9fe), GPU externe bit-exact (2ce487c+0c2af51), fallback transactionnel OOM→CPU + rollback sink (2d065ff+1014e30). Q30 N=2k/20k/50k/100k blocs < une ligne bit-exact (bab2b0b, 6 tests, ~9 min). Q32 (cellule M33 profonde) BLOCKED : corpus Windows indisponible, reporté à A6/Q61.

- 2026-11-07 : A2c/A2d DONE + ACCEPTED — témoins Q20 (N=20 000 bit-exact vs oracle RAM + quota scratch pire-tuile ≪ cache aligné complet, recyclage spool) et Q21 (erreur I/O mi-préparation → nettoyage limité aux artefacts possédés). Commit 89e1391. Q20a prend ~8 min (N=20k winsorized) ; garder en témoin lent.


- 2026-11-07 : A2b IMPLÉMENTÉ PAR JUNIOR (Tristan a demandé le plumbing direct). Transport source sérialisable (_CellSourceContext) + _prepare_cell_source_frame (décodeur/WCS identiques au cache) + _cell_task 9-tuple borné qui saute _build_one_cell_cache. Provider WindowedCanonicalProvider ouvert depuis la recette ; fermeture/cleanup sans supprimer un cache non possédé. Commits 8e56013 (code) + 83f2dde (tests). 304 tests verts (parité FITS TAN/TAN-SIP, spool externe, spawn, CUDA, reprise, durabilité). Harnais corrigés : m3_working_set + _tiny_durable_run figeaient désormais la RAM (socle 340 Mio introduit en A1).

- 2026-11-07 : A1 COMPLET (a+b+c) — A1a marge unique, A1b sonde jointe au layout, A1c VRAM 70 %. Commit bf0f832 : garde enforce_joint (sonde sautée en reprise figée) + recalibration du fixture de reprise (baseline=0, budgets 300/75 MB, assertions deficit). 173 tests reprise verts. A1d (schéma v2 identité) DÉFÉRÉ à une mission séparée après A2 (le contrat actuel tile/backend-in-identity reste en vigueur).

- 2026-11-07 : A1c ACCEPTED — politique VRAM unifiée à 70 % (GPU_VRAM_SAFETY_PERCENT, arithmétique entière, alias conservé), commit d1c66de, 115 tests. A1d BLOCKED par Coco (décision schéma) → Junior tranche Option A scoped.

- 2026-11-07 : A1-T01 (A1a+A1b) ACCEPTED après rework-1 (M1 libellé assemblage 121 B/px déféré à A4 + L1/L2/L3). 3 commits (cb0f0fa, e29f7db, 9933e2a), 10+300 tests verts, rejeu M33 refusé au layout, aucun changement science.

- 2026-11-07 : A1-T01 (A1a+A1b) Coco DONE — `_layout_joint_plan_probe` (layout admis seulement si l'agrégat joint complet tient, marge 15/100 appliquée une fois via build_plan_set). Vérifié Junior : 2 commits (cb0f0fa, e29f7db), 10+288 tests verts, rejeu M33 refusé au layout avec motif numérique, aucun fichier science touché. `test_zegrid_r41b_product_wiring.py` pend sur la baseline 5150040 aussi (préexistant, pas une régression A1).

- 2026-11-07 : A0-T01 Coco DONE (r0/r1/r2), revue Nono review-0 FINDINGS (1 MEDIUM + 3 LOW) → rework-2 corrigé, vérifié Junior. **A0_ACCEPTED** (défaut admission + limite GPU reproduits/quantifiés ; blockers corpus/Windows reportés NOT_RUN/BLOCKED).

- 2026-11-07 : A0-T01 Coco DONE (r0), vérifié Junior → rework-1 (CuPy env + résidu spool) → Coco DONE (r1), corrections vérifiées. A0 en revue Nono.

- 2026-11-07 : lancement A0→A6 reçu ; worktree `zemosaic-adaptive-resources` créé depuis `5150040`, Coco reset (tokens=0). A0-T01 en préparation.

- 2026-11-07 : plan initial issu de la réévaluation de beta et de la référence.
- 2026-11-07 : à la demande de Tristan, renforcement v2 : autorisation explicite,
  limites techniques, gates A0, mesures RAM/VRAM/disque, matrice Q00–Q64,
  protocole de reprise et clôture A6. Documentation seulement.

À chaque étape réelle, ajouter le changement d'état, la tâche, les chemins des
preuves et la prochaine action. Préserver les anciens verdicts et échecs.
