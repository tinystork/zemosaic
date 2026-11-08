# ZeGrid Adaptive Resources — fiche de reprise

Mise à jour : 2026-11-07  
Programme : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107`  
État : **REWORK F1–F10 appliqué — qualification produit encore PARTIELLE (runs réels restants)**
Contrat : [plan v2, sections 10–14](ZEGRID_ADAPTIVE_RESOURCE_ENGINE_PLAN_20261107.md)

## Réouverture ciblée M33 — 2026-11-08 (prioritaire sur les statuts historiques)

- Défaut confirmé sur beta `c73e6d3`, moteur Windows identique : ACTIVE ne sélectionne ni ne matérialise le working-set external après refus RAM/GPU. Le run M33 passe le layout puis refuse avant jauge/stack.
- Reproduction Junior : bundle external accepté, puis `_r55_active_decide` refuse RAM sans appeler le sélecteur external. Les tests F4 s'arrêtent au bundle ; Q60 utilise legacy. Qualification produit toujours PARTIELLE.
- Tristan autorise la correction et la délégation si substantielle. Mission bornée `ZM-ACTIVE-EXTERNAL-WIRING-20261108`, branche `feat/zegrid-adaptive-resources`, baseline `c73e6d37d3bf1d23a217dbffb03fb551b39db12d` propre avant cette note.
- Coco réinitialisé (nouvelle mission indépendante, tokens=0), délégation préparée. Brief : `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ACTIVE-EXTERNAL-WIRING-20261108.brief.md`.
- Rapport attendu : `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ACTIVE-EXTERNAL-WIRING-20261108.coco.r0.md`. Après callback : vérification Junior puis revue Nono ; ne pas clôturer sur le seul rapport worker.
- Baseline ciblée Junior : deux tests de sélection external (CPU/GPU détecté) passent, `2 passed in 38.78s`; cela ne valide pas la jonction ACTIVE.
- Aucun push/install/runtime Windows/full M33 lancé pour ce correctif. A1d et F14 restent hors périmètre.

## Acceptation ciblée Junior — 2026-11-08

- Mission `ZM-ACTIVE-EXTERNAL-WIRING-20261108` : **ACCEPT**, HEAD `38636ceeec82dddacbd3b9a372e854bf635c87fe`.
- Revue indépendante Nono r1 : ACCEPT. Aucun finding matériel restant ; vérification Junior du diff et du dépôt effectuée.
- Validation indépendante Junior conservée : **55 passed, 792 warnings, 364.80s**. Nono : transition publique naturelle 1 passed + typage 4 passed. Pas de nouvelle exécution revendiquée à la reprise.
- Acceptation limitée au câblage ACTIVE/external : **qualification produit Windows/M33/GUI toujours PARTIELLE**, run M33 complet NON EXÉCUTÉ. Aucun push/install/relance dans cette reprise.
- Publication de ce correctif sur `beta` explicitement autorisée par Tristan le 2026-11-08 ; cette publication ne vaut pas qualification Windows/M33 ni autorisation d’installation.

## Implémentation worker ZM-ACTIVE-EXTERNAL-WIRING — 2026-11-08

- Classe de working-set `external_spool` portée par `PlanContext`/`Candidate` (champ additif ; vide = classe RAM historique) : K=1, prefetch=0, résidence disque = enveloppe spool réelle.
- `_r55_active_decide` route un bundle external vers `_r55_active_external_decide` (même `plan_joint`, mêmes phases, même marge 15/100) ; `_r55_active_apply`/`_r55_active_admit` matérialisent les tâches external et refusent tout changement de classe (shrink/grow inclus).
- Sélection pré-gauge : un refus RAM bounded sélectionne la classe external au MÊME pin (géométrie/halo/frames), sans seconde transition de profil ; un refus disque reste typé `R55_REFUSE_DISK` AVANT jauge/stack.
- Pricing external aligné sur la marge (espace de réduction + overlap assembly réservés), aucun cache aligné fantôme ; la taille de lot ne remplit plus le budget brut.
- Reprise : une admission gelée `external_spool` reconstruit toujours la classe external (tile/lot gelés), jamais RAM, même si la RAM fraîche suffirait.
- Preuves r0 : `tests/test_zegrid_active_external_wiring.py` (7 tests, dont E2E ACTIVE réel avec refus GPU injecté et exécuteur external spy call-through) ; suites ciblées M3/R55D2/resume/run_store/r52/F5/E8/A1 vertes. Deux échecs `test_zegrid_m0m1_dry_plan_cpu_ladder` reproduits à l'identique sur baseline `c73e6d3` (pré-existants, hors périmètre).
- REWORK-1 (revue Nono/Junior) :
  - ajout d'une régression PUBLIQUE `run_zegrid_mode` du cas M33 naturel : bundle RAM initial accepté, décision ACTIVE réelle refusée par la marge joint 15/100, puis le bloc de production sélectionne/construit l'external au MÊME pin (layout/cellules/frames/halo), exécution provider/spool/réduction réelle, FITS+manifeste, science/coverage bit-identiques à la référence RAM ; le test n'appelle jamais `force_external` et vérifie la trace des rebuilds (stream -> external), le backend demandé GPU / effectif CPU et l'identité stable.
  - typage honnête des refus : plancher RAM de réduction (`reduction_workspace_floor`) -> `R55_REFUSE_HOST_RAM` (jamais DISK), spool réel trop gros -> `R55_REFUSE_DISK`, observation disque indisponible -> `R55_INTERNAL_ERROR`, lot gelé valide trop gros -> HOST_RAM, lot gelé malformé -> INTERNAL (identité/config). Régressions sélecteur + builder.
  - provenance backend : la classe external force le backend CPU des cellules et publie requested=GPU / effective=CPU / actually_used=false (aucune exécution GPU revendiquée).
  - Nono W2 (ladder mono-rung au contrôle de mapping) non appliqué : suggestion sans défaut actif, hors périmètre minimal. W5/W6 écartés par l'arbitrage Junior (messages/diagnostic, F3 pixel-granulaire déjà présent).
- Preuves REWORK-1 : module `test_zegrid_active_external_wiring.py` **12 passed** ; suites ciblées R55D2/resume/run_store/r52/M3/M2/M4 **346 passed** (m4 mis à jour vers `frozen_batch_ram`). Les deux échecs m0m1 restent pré-existants. La régression publique utilise des faits de ressources déterministes (budget mémoire simulé, estimateur de flux fixé) et le socle processus réel ; ce n'est PAS une qualification physique ni un run complet.
- Run M33 production : **NON EXÉCUTÉ**. Qualification Windows toujours requise ; ce rapport worker n'est pas une acceptation produit.

## Verdict historique après audit transversal (2026-11-07)

**Les défauts matériels F1–F13 de l'audit sont corrigés et vérifiés indépendamment** (commits
locaux sur `feat/zegrid-adaptive-resources`) : F2/F3 (admission véridique + sous-ligne) `b251643`,
F4/F8 (external sous GPU + fallback sticky/provenance) `aad9f64`, F1/F6/F7 (publication bandée +
scratch global/per_cell + cleanup honnête) `0ed7cb2`, F5 (grow A5 câblé) `3c81aaf`, et
F11/F12/F13 (reprise + télémétrie + faux échec run_log) `9fd304b`+`0cd0080` déjà poussés `91a425b`.
**Les gates produit restent PARTIELS** : Q40/Q41/Q50/Q60/Q64 exigent des runs réels sur machine
contrainte (grand canevas M33, rampe 4→1→4 matérielle, GUI ACTIVE→external GPU→scratch→FITS) qui
n'ont pas encore tourné. F14 (WARN trop bruyants, LOW) est exclu par décision Tristan. A1d (schéma
d'identité v2) reste différé. Le run de contrôle M33 GPU de Tristan (sortie
`C:\Users\JarvisJunior\Desktop\zemosaic out\M33`) confirmait AVANT correction le refus F4
(external verrouillé sous GPU) ; à rejouer après ces corrections.

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

**Dernier gate accepté : A5_ACCEPTED (Q50+Q51).** A6 partiel (Q60/Q64 PARTIELS,
Q61 BLOCKED, Q62/Q63 reportés au test direct beta). Rework F1–F13 appliqué.

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
