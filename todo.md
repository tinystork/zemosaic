# ZeMosaic — Architectural Cleanup R0–R3

## Statut et autorisation

**STATUT TERMINAL — R0–R3 CLOS / PROMOTED BETA+MAIN / TECHNICAL ACCEPT / HUMAN SCIENCE ACCEPT**, le 2026-10-04.

La mission R0–R3 est **CLOS** et **promue** à travers `beta` puis `main`. Version **inchangée** (`4.7.0`) ; aucun bump/tag/release/déploiement. **Aucune action de refactor active ne subsiste** pour cette mission.

Références et preuves de promotion (exactes) :

- PR #420 refactor → beta : https://github.com/tinystork/zemosaic/pull/420 — merge `bbe63d4edc65d4d762a6d42e59081f6fe7130697`.
- PR #421 Windows intertile safeguard → beta : https://github.com/tinystork/zemosaic/pull/421 — merge/current beta `14607d94342a308672b5be04d0db5a562fc564ab`.
- PR #422 beta → main : https://github.com/tinystork/zemosaic/pull/422 — merge/current main `940655a7202568dad8dd8fa18ea5f12bfea284f8`.
- `origin/main` et `origin/beta` diffèrent en commit ID mais ont des **arbres exactement identiques**.
- Suite post-main en worktree détaché : **445 passed, 0 failed, 261 warnings**.
- M106 : critère mécanique bitwise verrouillé `INCONCLUSIVE/HOLD` ; verdict humain/scientifique Tristan `ACCEPT` (2026-10-04 14:17 Europe/Paris). Distinction préservée.
- Preuve incident Windows (domaine de crash, PAS la couche native exacte) : `/home/tristan/M106/faulthandler_intertile.log` — `0xc0000374`, exactement six ThreadPool workers dans `_process_overlap_pair -> reproject_interp`, stacks concentrés reproject/Astropy WCS, parent attend sur futures, OpenCV absent des stacks capturés.
- Preuve mitigation Windows : `/home/tristan/M106/outwinsequential/` — workers 14→1, token `windows_reproject_wcs_serial`, mode séquentiel, 279/279 paires, 26/26 Phase 5, WORKER_DONE/run succès, artefacts FITS/preview ; intertile séquentiel ~144.1s sur 1570.1s de run complet.
- Optimisation différée (NON implémentée) : élagage top-K séquentiel des paires. Pour la simulation M106, K=8 → 279→155 paires, graphe connecté, estimé linéaire ~144→80s. K=8 ne change PAS le compte workers Windows (toujours effective_workers=1 / aucun ThreadPool). Comparaison scientifique vs graphe complet requise. Backlog `PERF-01`, pas une feature complétée.

Historique (archivé, inchangé) :

Tristan a autorisé le lancement de la mission. R0 (archéologie + baseline ciblée)
est terminé et accepté par Junior après revue indépendante Nono `review-3: ACCEPT`.
Les témoins pré-extraction TEST-01/03/04/05/06/07 sont en place. R1 est clos sans
suppression : aucun candidat n'a franchi le seuil PROVEN DEAD. Le scope R2 **borné**
accepté est clos pour CETTE mission : lot 1 (helpers de regroupement, commit `8b7a979`),
lot 2A (témoin crash-breadcrumb, commit `5d7920d`), lot 2B (moteur crash-breadcrumb
stateless, commit `1282cfe`) — tous Nono `review-0: ACCEPT` + acceptation Junior.
Ceci ne prétend PAS que le worker de ~38k lignes est « terminé » : les extractions
arbitraires sont volontairement évitées et toute décomposition future exige une nouvelle
mission bornée (témoins/critères/revue). L'audit R3 post-R2 et `FINAL_REPORT.md` sont
**acceptés techniquement** après Nono `review-0: ACCEPT` + acceptation Junior.
M106 a ensuite été exécutée sur le corpus privé de 66 FITS : géométrie/coverage/winner/weighted
bit-identiques, variations science/aesthetic bornées par une variabilité même-build plus grande.
Après contrôle plein format sans anomalie visible, Tristan a rendu `HUMAN_VISUAL_VERDICT=ACCEPT`
le 2026-10-04 à 14:17 Europe/Paris. Le dossier de preuve est conservé sous
`/home/tristan/M106/gate_evidence_20261004/`.

- [x] Vérifier l'identité du dépôt et actualiser les références distantes.
- [x] Vérifier base, version et propreté initiale.
- [x] Vérifier les principales hypothèses de la mission par lecture du code.
- [x] Créer la branche locale dédiée depuis le SHA exact.
- [x] Écrire le plan, ses corrections et ses gates.
- [x] Recevoir l'instruction de lancer la mission.
- [x] Exécuter et accepter R0 (archéologie + baseline, Nono `review-3: ACCEPT`).
- [x] Clore R1 sans suppression : aucun candidat PROVEN DEAD après témoins et revue.
- [x] Témoins de caractérisation pré-R2 complets (TEST-01/03/04/05/06 clos).
- [x] Finaliser la carte des contrats R3 (baseline PRE-R2) : `docs/refactor/STACKING_CONTRACTS_R3.md` accepté comme freeze de comportement/contrats après Nono `review-1: ACCEPT` et acceptation Junior.
- [x] Exécuter le scope R2 borné accepté (lot 1 `8b7a979`, lot 2A `5d7920d`, lot 2B `1282cfe` — Nono review-0 ACCEPT + Junior).
- [x] Rédiger l'audit R3 post-R2 + `FINAL_REPORT.md`.
- [x] Obtenir l'acceptation R3 technique finale (Nono `review-0: ACCEPT` + Junior).
- [x] Obtenir l'acceptation scientifique manuelle de Tristan sur M106 (`ACCEPT`, 2026-10-04 14:17 Europe/Paris).

> Les puces `- ARCHIVAL/GATE —` qui suivent dans ce fichier sont des éléments historiques ou des gates réutilisables pour de futures missions bornées — **PAS des tâches actives** de la mission R0–R3 (close).

## Backlog actif (missions séparées, non bloquantes)

La mission R0–R3 est close. Les éléments ci-dessous sont des **missions distinctes** à lancer séparément ; aucun n'est implémenté ici.

| Priorité | ID | Sujet | Posture |
| --- | --- | --- | --- |
| **CLOS** | SCI-01 | Grid CPU / legacy WSC vs `stack_core` GPU simplifié — témoin/quantification numérique, sans correction groupée | **FAIT** (ZM-SCI-01-GRID-WSC-CHAR-20261004) — CLOS-NO-FIX, voir ligne SCI-01 §10 |
| **CLOS** | SCI-02 | Placeholder `linear_fit` dans `stack_core` — caractérisation seule (callers effectifs ; distinguer normalisation vs rejet), sans correction groupée | **FAIT** (ZM-SCI-02-STACKCORE-LINEARFIT-CHAR-20261004) — CLOS-NO-FIX, voir ligne SCI-02 §10 |
| **CLOS** | SCI-03 | Masques/poids Grid CPU/GPU, all-invalid, aliases, winsor_limits — caractérisation seule | **FAIT** (ZM-SCI-03-GRID-MASK-WEIGHT-CHAR-20261004) — CLOS-NO-FIX, voir ligne SCI-03 §10 |
| **CLOS** | SCI-04 | variantes Classic/SDS/Phase 4.5 low-N/chunking — caractérisation seule | **FAIT** (ZM-SCI-04-CLASSIC-SDS-PHASE45-VARIANTS-20261004) — CLOS-NO-FIX, voir ligne SCI-04 §10 |
| P2 | ARCH-01/02/03/04/05 + SCI-07 | dettes architecture/inconnues/dormantes (invocations Phase 4.5, contrats externes/frozen, `cuda_utils` SUSPECTED DEAD, Tk legacy dormant, duplication SDS Classic, helpers affine 4.5 absents) | missions séparées |
| **CLOS** | TEST-02 | ancien diagnostic manuel `test_version_gpu.py` (0 test collecté), désormais remplacé par la détection/planification/probe GPU de ZeAlfie | **RÉSOLU** — fichier supprimé ; aucune couverture pytest ni fonction runtime ZeMosaic retirée |
| P3 | Plateforme/parité | parité distribution restante (macOS, frozen/PyInstaller/external solver plus large, CPU↔GPU) ; revendication Windows bornée au chemin standalone M106 observé | missions séparées |
| P3 (différé) | PERF-01 | élagage top-K séquentiel K=8 (279→155 paires, graphe connecté, ~144→80s estimé) ; ne change PAS effective_workers=1 / aucun ThreadPool ; comparaison science vs graphe complet requise | différé, non implémenté |

## 1. Base canonique et branche

| Élément | État constaté lors de la préparation |
| --- | --- |
| Dépôt | `https://github.com/tinystork/zemosaic.git` |
| Checkout réel | `/home/tristan/.openclaw/workspace/projects/zemosaic` |
| Chemin de l'énoncé | `~/zemosaic/zemosaic` absent sur cette machine |
| BASE SHA | `c03d0bb965d073b12ad9978094327829f0d0c366` |
| `origin/main`, `origin/beta`, HEAD initial | Tous égaux à BASE SHA après `git fetch origin --prune` |
| Branche initiale | `beta`, worktree propre |
| Branche préparée | `refactor/zm-architecture-cleanup-r0-r3` |
| HEAD après checkpoint documentaire | `035119eb72266eeb8279a63fd244beff803ec213` |
| Version | `4.7.0`, `src/zemosaic/__init__.py`; concorde avec `version.txt` |
| Checkpoint de préparation | `035119e docs: plan ZeMosaic architectural cleanup` |

Ne pas réinitialiser une branche existante, rebaser, fusionner ou avancer la base
opportunistement. Aucun push, merge, tag, release, bump de version ou modification
de `main`/`beta`. Les commits futurs restent petits et locaux.

### Gate de reprise

- [x] Relire ce fichier et les instructions locales; vérifier qu'aucun worker n'est actif.
- [x] Refaire `git fetch origin --prune`, `git status --short`,
  `git branch --show-current`, `git rev-parse origin/main origin/beta HEAD`,
  `git log --oneline --decorate -10` et `git worktree list` dans le vrai checkout.
- [x] Si une branche distante diffère de BASE SHA : STOP et rapport, sans changement de base.
  (vérifié : origin/main == origin/beta == BASE SHA après fetch, aucun écart)
- [x] Préserver toute modification nouvelle de Tristan, ne rien stasher/effacer implicitement.
- [x] Résoudre le checkpoint documentaire : le commit local
  `docs: plan ZeMosaic architectural cleanup` (`035119e`) existe déjà et le worktree
  était propre avant R0. Ne pas le supprimer pour satisfaire artificiellement la gate.
  Aucun commit de produit/test créé pendant R0.
- [x] Garder BASE SHA comme référence scientifique immuable même après les commits docs/tests.

## 2. Avis de cohérence — constats ciblés, pas R0 complet

La mission est cohérente avec le code de cette base. Les avertissements Classic,
Phase 4.5, helpers Tk et divergence Grid `winsorized_sigma_clip` sont fondés. Les précisions suivantes
font partie du plan corrigé. Les numéros de ligne ci-dessous désignent BASE SHA.

| Sujet | Preuve statique et conséquence |
| --- | --- |
| Layout/entrée | `pyproject.toml` déclare `[project.gui-scripts] zemosaic = "zemosaic._app:main"`, pas `console_scripts`. Sources dans `src/zemosaic/`. Ajouter `python -m zemosaic` à la carte. |
| Application | `_app.py:main` impose Qt, appelle `freeze_support`, puis `zemosaic_gui_qt.run_qt_main`; GUI crée le process ciblant `run_hierarchical_mosaic_process`, qui appelle le dispatcher. |
| Wrapper racine | `run_zemosaic.py` délègue bien à `_app.main`, MAIS `ZeMosaic.spec:160` l'utilise aussi comme entrée PyInstaller. Pas un candidat mort; préserver également `_version.py` (stub sep_pjw), hooks et hiddenimports. |
| Dispatch | `zemosaic_worker.py:30898–30983` : Grid détecté d'abord; exception/import Grid manquant = arrêt SANS fallback Classic. Sinon SDS résolu par overrides/config; hors SDS, appel de `run_hierarchical_mosaic_classic_legacy`. |
| Phase 4.5 | Défaut False dans `zemosaic_config.py:200`; GUI Qt impose False (`zemosaic_gui_qt.py:4180`, `:4338`), filtre Qt aussi (`zemosaic_filter_gui_qt.py:1663`). Le helper partagé contient une branche active si `phase45_options.enable` (`zemosaic_worker.py:11041`); appelants Classic/SDS présents. L'accessibilité par invocation SUPPORTED reste à établir. |
| Phase 5 partagée | `_run_shared_phase45_phase5_pipeline` exécute aussi la Phase 5 lorsque 4.5 est désactivée. Ne jamais supprimer ce helper au motif que 4.5 serait obsolète. |
| Filtre historique | `zemosaic_filter_gui_qt.py:448` importe les trois helpers nommés dans la mission, avec copies de secours en cas d'échec. Worker : imports dynamiques de `launch_filter_interface` à `:25615` et `:31844`. Tk est importé dans certaines fonctions, pas simplement à l'import de tout le module. |
| Couplage retour | Le filtre Qt prend aussi des helpers du worker (`:434`); les extractions doivent examiner ce graphe dans les deux sens et les références capturées à l'import. |
| Grid CPU/GPU | `grid_mode.py:1913` utilise les rejets établis; `:1984` / `:2018` appelle `stack_core` côté GPU. Le core contient un `winsorized_sigma_clip` simplifié (`zemosaic_stack_core.py:325`), distinct du WSC PixInsight. Divergence structurelle confirmée, impact numérique NON MESURÉ ici. |
| Nuance linear_fit | Le placeholder existe à `zemosaic_stack_core.py:294`, mais Grid GPU normalise en amont puis passe `normalize_method='none'` (`grid_mode.py:2012`). Ne pas lui attribuer sans preuve ce placeholder. Cartographier séparément normalisation linear-fit et rejet linear-fit. |
| Monolithe | `zemosaic_worker.py` : 38 882 lignes, 1 736 598 octets. La taille n'est pas un critère de succès en elle-même. |
| Témoins Phase 3 | 11 appels `pytest.importorskip("zemosaic_worker")` persistent dans `tests/test_phase3_adaptive_invariants.py`, alors que le package interdit les alias plats (`tests/test_packaging.py`, NamespaceTests). Risque de skips silencieux; aucun résultat pytest revendiqué à ce stade. |
| Tests structurels | Des tests extraient des sous-chaînes/commentaires du worker. Une extraction peut casser leur localisation sans casser le comportement. Migrer les preuves sans affaiblir les assertions, avec revue. |
| Garde CI Tk | `.github/workflows/no-tk-on-official-path.yml` interdit seulement certains imports Tk directs. Ce n'est PAS une preuve d'absence de dépendance transitive ou de fallback Tk. |

## 3. Compléments nécessaires à la mission

- [x] **Définir les chemins supportés en R0** : GUI installée, `python -m`, checkout,
  distribution frozen, éventuelle API programmatique documentée, scripts externes.
  Une fonction importable n'est pas automatiquement publique; l'absence de caller
  local ne prouve pas son abandon. En cas de doute : UNKNOWN, conserver.
  (documenté dans ARCHAEOLOGY_R0.md §1)
- [x] **Préparer R3 dès R0** pour les zones touchées, puis le compléter en fin de
  mission. La carte du stacking et ses témoins doivent précéder une extraction,
  pas être découverts après R2.
  (STACKING_CONTRACTS_R3.md préliminaire créé)
- ARCHIVAL/GATE — **Définir une comparaison avant/après par chemin**, à entrées/config/environnement
  identiques, distincte d'une comparaison CPU/GPU. Bit-identité lorsque déterministe;
  sinon tolérances explicites justifiées sur la baseline, définies AVANT changement,
  jamais élargies pour faire passer le candidat. Comparer masques et couverture
  autant que les pixels; ne pas exiger une parité entre deux chemins déjà divergents.
- [x] **Inventorier l'environnement réel** : Python, dépendances/versions, OS,
  GPU/driver/CUDA/CuPy, BLAS/threads, graines, disponibilité solveurs/catalogues.
  Distinguer backend demandé, effectivement exécuté et fallback. Un skip GPU ou
  un mock CuPy ne vaut pas qualification GPU physique.
  (documenté dans ARCHAEOLOGY_R0.md §0.1 ; CuPy runtime OK sur MX150, nvcc absent)
- [x] **Isoler les tests** : répertoires temporaires pour profil/config/cwd/XDG,
  copies de FITS; pas de modification du profil ni des brutes de Tristan, pas de
  téléchargement/catalogue/réseau implicite. Consigner passed/failed/skipped/xfail,
  collecte, warnings utiles et raisons. Vérifier les 11 imports plats avant de
  compter ces tests comme témoins; ne pas réintroduire d'alias produit pour eux.
  (isolé sous /tmp/zm-r0-home + /tmp/zm-r0-xdg ; 11 skips confirmés, cf. TEST-01)
- ARCHIVAL/GATE — **Préserver GUI → process → moteur** : valeurs et précédence des paramètres,
  pas seulement clés/signatures. Le wrapper renomme, parse, suffixe puis filtre
  silencieusement les kwargs via `inspect.signature` (`worker:36251` et suivantes).
  Exiger un témoin de propagation et un petit témoin process réel si cette zone bouge.
- ARCHIVAL/GATE — **Préserver concurrence/lifecycle** : `spawn`, objets picklables et chemins
  d'import, queues/protocoles, callbacks, globals/closures, initialisation CUDA,
  annulation/arrêt, nettoyage threads/process/memmap/FITS. Pas de passage fork↔spawn
  ni changement d'ordre des opérations/imports sous couvert de déplacement.
- ARCHIVAL/GATE — **Préserver les acquis 4.7.0** : WCS filtré transmis en mémoire, absence de
  second solve, choix write-WCS respecté, annulation sans déplacement de source,
  fermeture du filtre et arrêt worker sans fallback scientifique involontaire.
- ARCHIVAL/GATE — **Préserver caches/reprise** : formats et signatures, paths, réutilisation de
  master tiles, checkpoints Phase 1/5, invalidation et run repris vs neuf; pas de
  migration incidente ni effacement des artefacts de référence.
- ARCHIVAL/GATE — **Étendre la matrice distribution** : wheel installé hors checkout/CWD,
  ressources et locales, PyInstaller/hiddenimports/hooks, lancement Windows/macOS/Linux,
  CPU sans CuPy et intégrations absentes. Une plateforme non testée reste NOT_RUN.
- ARCHIVAL/GATE — **Clarifier les STOP** : divergence CPU/GPU préexistante = geler la
  consolidation/parité concernée, documenter et continuer ailleurs; changement
  nouveau avant/après = bloquer le lot. Aucune correction scientifique incidente.
- ARCHIVAL/GATE — **Borner R2** après R0 : liste ordonnée de lots et critères mesurables validés
  par Junior/Nono. Pas de promesse de finir le monolithe « dans la nuit », pas de
  nombre arbitraire de fichiers ou lignes. Limiter chaque correction à 3 REWORK.

## 4. Invariants / hors périmètre

Science gelée : seuils/rejets WSC et Kappa/Sigma, photométrie, WCS, ZeSolver/ZeNear,
reprojection, poids des tuiles, background matching, sémantique FITS, géométrie Grid,
science SDS. Aucun algorithme/performance/ordre de réduction numérique « amélioré ».

Préserver formes/axes HWC-CHW, dtype et conversions, unités, NaN/Inf, alpha continu,
masques/coverage/poids et low-N. La Phase 3 adaptative reste single-tile, toutes les
brutes contribuent à la même tuile logique; retries/chunking ne deviennent pas un
split physique. Conserver les budgets/retries/fallbacks et leur observabilité.

Hors périmètre : Direct Tiled Co-add, Mini Tiles, nouvelle sémantique Stack Plan,
sélection ZeAnalyser, acquisition Seestar, refonte Grid, ajout de dépendance majeure.

Standalone impératif : ZeAlfie non requis au runtime, intégrations optionnelles,
SolverPort/adapters publics conservés, aucun import/probing de dépôt frère.

## 5. R0 — archéologie complète (acceptée localement)

- [x] Créer `docs/refactor/ARCHAEOLOGY_R0.md`, ancré à BASE SHA, symboles et preuves.
- [x] Dessiner le graphe application : entrée installée / module / wrapper / frozen
  → bootstrap Qt → GUI/filtre → wrapper process → dispatcher → Classic/SDS/Grid.
- [x] Détailler précédence Grid/SDS, erreurs/fallbacks et payloads GUI/filter.
- [x] Cartographier Classic, SDS, Grid et 4.5 si accessible : appelant →
  normalisation → pondération → rejet → combine → backend réel/fallback.
  (4.5 corrigé jusqu'au rework-3 : flux A–E, CPU-only, normalisation active,
  branche alpha-weighted, gain inter-super et branches dormantes documentés)
- [x] Couvrir Filter Qt/Tk, SolverPort/ZeSolver/ASTAP, lancement ZeAnalyser,
  FITS I/O, WCS, reprojection, photométrie, assembly, checkpoint/resume,
  preview/progress, ressources/GPU, modules `core/robust_rejection` et `cuda_utils`.
  (ARCHAEOLOGY_R0.md §7 — cartes data-plane ajoutées en rework-1)
- [x] Inventorier globals, caches, imports dynamiques, side effects et dépendances
  inverses worker↔GUI avant d'identifier des modules extractibles.
- [x] Classer chaque module ET chemin significatif : ACTIVE, COMPATIBILITY,
  TEST / DIAGNOSTIC, DORMANT BUT REACHABLE, SUSPECTED DEAD, PROVEN DEAD ou UNKNOWN.
  Inclure preuves, appelants, contrat supporté, témoins, risque et revue.
- [x] Établir le statut supporté de Phase 4.5 : preuve de chemin et témoin ciblé;
  jamais « False dans Qt = mort ». Ne pas changer les defaults pour la tester.
  (conclusion : DORMANT BUT REACHABLE / programme seulement, désactivé sur GUI Qt)
- [x] Séparer helpers métier/Tk/fallbacks dans le filtre, sans suppression immédiate.
- [x] Établir une baseline rapide, isolée et honnête (cf. section 6).
- [x] Ajouter avant extraction les petits témoins de comportement absents.
  (non exécuté en R0 — interdit d'éditer les tests dans cette itération)
  TEST-01 clos séparément (mission ZM-ARCH-TEST01-PHASE3-IMPORTS-20261003) : imports
  plats Phase 3 → qualifiés, 33 pass/0 skip. Témoin dispatch RÉSOLU séparément
  (mission ZM-ARCH-WITNESS-DISPATCH-20261003, cf. TEST-03). Les témoins
  low-N/all-invalid, spawn réel et cache/reprise sont clos par TEST-04/05/06.
- [x] Produire premiers tableaux R3, anomalies scientifiques, UNKNOWN/STOP et
  ordre proposé des extractions avec critères de sortie de chaque lot.
- [x] Revue Nono indépendante de R0; toute classification PROVEN DEAD contestée
  redevient UNKNOWN ou SUSPECTED DEAD avant R1.
  (trois cycles de correction docs-only; Nono `review-3: ACCEPT`, puis ACCEPT Junior)

PROVEN DEAD exige, selon le composant : aucun caller supporté, dépendance d'import,
chemin Qt/CLI/package/API publique, import dynamique, dépendance packaging/resource,
contrat de test pertinent ou besoin de compatibilité. Recherche AST/texte seule
insuffisante. Aucun candidat n'est déclaré PROVEN DEAD dans cette préparation.

## 6. Inventaire initial des témoins — baseline R0 exécutée

Résultats exacts, durées et skips de R0 : `docs/refactor/ARCHAEOLOGY_R0.md` §12.
Pour les prochains témoins, utiliser l'interpréteur de l'environnement constaté et
`python -m pytest -q -ra <fichiers>` sous isolation; enregistrer commande exacte,
SHA, environnement, durée, compteurs et contrat prouvé. Groupes sélectionnés selon
le diff, pas toute la suite à chaque déplacement.

| Témoins existants à inspecter/exécuter | Contrat / précaution |
| --- | --- |
| `tests/test_packaging.py` | Entrée gui-scripts, namespace, CWD, ressources, migration config, CPU-only; ajouter la preuve wheel réelle si nécessaire |
| `tests/test_phase3_adaptive_invariants.py` | Identité/membership d'une tuile, WSC par passes, backoff, ETA; auditer skips plats et tests texte |
| `tests/test_grid_mode_dbe.py` | DBE actuel et protection des étoiles, pas parité complète du stacking |
| `tests/test_grid_mode_stack_plan_paths.py` | Résolution actuelle des chemins du CSV; ne prouve pas à lui seul toute sa sémantique |
| `tests/test_solver_port.py`, `tests/test_solver_port_integration.py`, `tests/test_zesolver_adapter.py`, `tests/test_zesolver_hardening.py` | Frontière solveur, contrats optionnels et comportement actuel |
| `tests/test_zesolver_filter_handoff.py`, `tests/test_zesolver_filter_handoff_hg2.py`, `tests/test_zesolver_filter_qt.py` | Filtre → GUI → process → Phase 1, WCS en mémoire et cycle Qt |
| `tests/test_zesolver_filter_cancel_hg2.py` | Annulation/fermeture pendant solve, non seulement chemin succès |
| `tests/test_zesoftware_interop.py`, `tests/test_zeanalyser_launch.py` | Standalone et lancement via contrats installés |
| `tests/test_cupy_platform_guard.py`, `tests/test_phase5_vram_budget.py`, `tests/test_resource_telemetry.py` | GPU optionnel, gardes CuPy/NVRTC, budgets et télémétrie ; l'ancien diagnostic manuel `test_version_gpu.py` a été retiré sous TEST-02 sans perte de couverture |
| `tests/SMOKE_PROTOCOL_Windows_macOS.md` et garde CI Qt | Compléments plateforme, ne pas annoncer PASS sans exécution |

- ARCHIVAL/GATE — Compléter les trous : dispatch réel, paramètres aval, absence de Tk sur
  chemins officiels, cache/reprise, stacking petits tableaux et erreurs/fallbacks.
- ARCHIVAL/GATE — Préserver les différences existantes avec des tests de caractérisation;
  séparer une parité attendue mais fausse des gates de non-régression. Si xfail
  nécessaire : ciblé/strict, témoin d'échec et entrée SCIENCE, jamais xfail global.
- ARCHIVAL/GATE — Tests structurels déplacés : adapter leur localisation ou les remplacer par
  un témoin comportemental équivalent avec revue; ne pas masquer un vrai échec.
- ARCHIVAL/GATE — Une validation globale consolidée à la fin d'un jalon significatif est utile;
  pas de full suite répétitive après chaque petit diff.

## 7. R1 — supprimer seulement le code prouvé mort

**Décision R1 (2026-10-04) : CLOS, AUCUNE SUPPRESSION.** R0 §14 et la revue Nono
acceptée n'identifient aucun PROVEN DEAD. Les témoins ajoutés depuis R0 n'ont fait
passer aucun SUSPECTED DEAD au niveau PROVEN DEAD; `ARCH-03` reste donc conservé.
Les checklists ci-dessous sont sans objet pour ce passage R1 et restent le gate à
réutiliser si une preuve nouvelle apparaît. Aucun diff produit/test de suppression,
aucun commit `refactor: remove ...`.

Pour CHAQUE unité, checklist à copier dans son rapport :

- ARCHIVAL/GATE — Preuves R0 + revue Nono acceptées, contrats/packaging/compatibilité exclus.
- ARCHIVAL/GATE — Témoins ciblés avant modification, résultats conservés.
- ARCHIVAL/GATE — Petit diff mécanique; aucune refonte nécessaire.
- ARCHIVAL/GATE — Témoins après, `git diff --check`, inspection des fichiers inattendus.
- ARCHIVAL/GATE — Revue Nono du diff final, vérification indépendante Junior.
- ARCHIVAL/GATE — Commit local distinct `refactor: remove proven-dead <specific thing>`.

Ne pas présumer morts Classic legacy, filtre historique, Phase 4.5, wrapper,
`_version.py`, modules CUDA « doublons », scripts de build ou diagnostics utilisés.
Si aucune suppression sûre n'existe, R1 peut se terminer sans suppression; documenter.

## 8. R2 — extractions bornées et mécaniques

Direction conceptuelle, pas architecture nouvelle imposée : prepare → dispatch →
geometry/solve → group → stack → photometry → reproject/compose → write.

Ordre CANDIDAT à confirmer en R0 : helpers purs réellement partagés du filtre,
puis petites responsabilités transversales aux contrats connus; ne pas commencer
par réécrire/déplacer en bloc Classic/SDS. La priorité exacte dépend des témoins,
des dépendances et du risque, pas du nombre de lignes.

### Premier lot R2 borné (gelé avant R2 — à ne pas implémenter dans cette mission)

Le premier lot R2 est **borné** aux trois helpers purs réellement partagés du filtre :
`_merge_small_groups`, `_split_group_by_orientation`, `_circular_dispersion_deg`.

Critères d'acceptation du lot :
- Nouveau module neutre (sans import Tk) hébergeant ces trois helpers.
- Imports/exports de compatibilité legacy préservés (signatures/symboles/sémantique identiques).
- Le chemin officiel Qt n'importe plus le module legacy Tk pour ces trois helpers.
- Tests de comportement exacts : wrap circulaire, PA invalide, min-size, cap, ordre.
- Aucun changement worker/science ; pas de modification de comportement numérique.
- Tests d'import / Qt / package pertinents.
- Revue Nono du diff exact.

Ne pas implémenter ce lot ici (mission R3 baseline freeze).

**Statut R2 lot 1 — DONE, revue Nono `review-0: ACCEPT`, commit local `8b7a979`** (mission `ZM-ARCH-R2-LOT1-GROUPING-HELPERS-20261004`, 2026-10-04) :
- Nouveau module neutre `src/zemosaic/core/grouping_helpers.py` (imports `math`/`typing`/`collections.abc` uniquement, aucun import Tk/Qt/`zemosaic_filter_gui`) hébergeant les trois helpers canoniques `_merge_small_groups`, `_split_group_by_orientation`, `_circular_dispersion_deg` **déplacés verbatim** + leurs deps privés `_group_center_deg`, `_angular_sep_deg`, `_circ_delta_deg` (non réutilisés ailleurs dans `zemosaic_filter_gui.py`).
- `zemosaic_filter_gui.py` réexporte les six noms depuis le module neutre (mêmes objets, identité `is` vérifiée).
- `zemosaic_filter_gui_qt.py` importe les trois helpers depuis `core.grouping_helpers` (chemin officiel) ; les copies fallback inline Qt restent intactes et distinctes (garde `if _tk_* is None` conservée).
- Témoin : `tests/test_grouping_helpers_r2_lot1.py` (25 pass). Suite post-edit : 140 pass. Diff body byte-identique vérifié par AST.

**Seam extraction Q2** : la variante Qt `_split_group_by_orientation_key` / `split_clusters_by_orientation` / `_split_group_by_mount_mode` (buckets gloutons, algorithmes DIFFÉRENTS des canoniques) reste volontairement NON consolidée ; de même les trois copies fallback inline Qt sont des algorithmes distincts (mean-angle/max-deviation, buckets gloutons, log texte différent) laissés en fallback, jamais unifiés avec les canoniques. Toute consolidation future = décision scientifique séparée, pas une extraction mécanique.

**Statut R2 lot 2B — DONE, revue Nono `review-0: ACCEPT`, acceptation Junior** (mission `ZM-ARCH-R2-LOT2B-CRASH-BREADCRUMB-EXTRACTION-20261004`, 2026-10-04, base HEAD `5d7920d`) :
- Moteur crash-breadcrumb extrait mécaniquement dans `src/zemosaic/core/crash_breadcrumbs.py` (neutre, sans état runtime breadcrumb, imports stdlib seulement — `json`/`os`/`time`/`datetime`/`pathlib`/`typing` — aucun import `zemosaic_worker`/GUI/Qt/Tk/GPU/CuPy/config/solver/science, aucune dépendance nouvelle).
- `zemosaic_worker.py` conserve les 4 globals observables `_CRASH_BREADCRUMB_LOCK` / `_CRASH_BREADCRUMB_PATH` / `_CRASH_STATE_PATH` / `_CRASH_BREADCRUMB_MODE` comme source de vérité de compatibilité ; `_configure_crash_breadcrumbs` / `_safe_runtime_snapshot` / `_emit_crash_breadcrumb` deviennent des adaptateurs minces (mêmes noms/signatures) qui lisent les globals au CALL TIME (assignation directe + monkeypatch conservent leur effet exact).
- Le thread heartbeat, l'install/restore des signaux, le protocole de queue et TOUS les call sites `_emit_crash_breadcrumb` restent dans le worker (NON extraits) ; `run_hierarchical_mosaic_process` lit toujours `_CRASH_BREADCRUMB_PATH`/`_CRASH_STATE_PATH` directement pour le payload `PROCESS_ERROR`.
- Témoin étendu : `tests/test_crash_breadcrumbs_characterization_witness.py` 47 → 53 pass/0 skip (6 assertions d'extraction ajoutées, aucun affaiblissement des 47 existantes). Combined new+dispatch+spawn+packaging+grouping+phase3 = 143 pass/0 fail. `git diff --check` OK, compile/AST OK.
- Lot 2A (commit `5d7920d`) = témoin de caractérisation de référence ; 2B ne le modifie pas.

Pour CHAQUE extraction :

- ARCHIVAL/GATE — Identifier comportement exact, tous appelants/imports/monkeypatchs, globals,
  lifecycle et frontière proposée sans cycle ni dépendance lourde nouvelle.
- ARCHIVAL/GATE — Définir critère d'acceptation : responsabilité nommée/testable localement,
  contrat identique, dépendances connues; pas simple éclatement arbitraire.
- ARCHIVAL/GATE — Témoins avant; déplacer mécaniquement, conserver signatures/symboles/semantics
  lorsque nécessaires (shim/réexport seulement si justifié et testé).
- ARCHIVAL/GATE — Préserver ordre des calculs, structures, copies/vues et effets de bord.
- ARCHIVAL/GATE — Tests ciblés/import/process/package pertinents après, diff-check, revue Nono,
  contrôle indépendant Junior; commit distinct après acceptation du diff exact.
- ARCHIVAL/GATE — Mettre à jour carte, témoins et ce TODO avant la tâche suivante.

Duplication : distinguer copie exacte, helper partagé, variante scientifique,
compatibilité et comportement de mode. Consolider uniquement les vrais doublons
mécaniques. Toute différence numérique/masques/fallbacks → STOP consolidation,
FOLLOW-UP SCIENCE. Pas de fusion Classic/SDS simplement parce qu'ils se ressemblent.

## 9. R3 — carte des contrats de stacking

- [x] Créer `docs/refactor/STACKING_CONTRACTS_R3.md` — **baseline PRE-R2 acceptée** après Nono `review-1: ACCEPT` et acceptation Junior, cf. mission `ZM-ARCH-R3-BASELINE-FREEZE-20261004`.
- [x] Lignes : Classic CPU/GPU, SDS (CPU réel, pas de SDS GPU), Grid CPU/GPU core/GPU legacy
  fallback, Phase 4.5 (branches alpha-weighted et configured-rejection).
- [x] Pour chacun : median/mean, Kappa-Sigma, WSC, linear-fit NORMALISATION et
  linear-fit REJET distincts; modes non supportés documentés, pas inventés.
- [x] Colonnes : entrée/caller, config/aliases/précédence, implémentation,
  normalisation, pondération scalaire/pixel, NaN/Inf, masque/rejet, dtype,
  axes/canaux, low-N/all-invalid, paramètres effectivement propagés,
  fallback/import/erreur/OOM, backend réel, tests et limites des preuves.
- [x] Tracer Grid CPU, GPU core et GPU legacy fallback si core indisponible;
  `core/robust_rejection.py` mappé sur ses appelants réels (align_stack / align_stack_gpu).
- [x] Aucune parité non mesurée; échecs préexistants et limites matériel conservés sans
  modifier la science.
- [x] **Audit et rapport R3 post-R2** — `FINAL_REPORT.md` et
  `STACKING_CONTRACTS_R3.md` **POST-R2 AUDITED — ACCEPTED** après Nono
  `review-0: ACCEPT` + Junior ; aucune extraction R2 ne touche stacking
  math/order/weights/rejection/WCS/FITS/science. M106 est désormais acceptée par Tristan.

## 10. TODO / FOLLOW-UP — SCIENCE et inconnues

| ID | Sujet | Statut initial / suite, hors corrections R0–R3 |
| --- | --- | --- |
| SCI-01 | Grid CPU / Grid GPU legacy appellent `_reject_outliers_winsorized_sigma_clip` SANS `wsc_impl` explicite → helper résout env/config/default et dispatche vers PixInsight WSC PAR DÉFAUT (`pixinsight`; env peut choisir `legacy_quantile`). GPU core `stack_core` winsorized = médian/σ simplifié (ni WSC, ni winsorization) | **CARACTÉRISÉ / CLOS-NO-FIX** (mission ZM-SCI-01-GRID-WSC-CHAR-20261004) : divergence conservée ET quantifiée — Grid CPU/legacy = WSC PixInsight par défaut vs core = médian/σ simplifié. Preuve dynamique : `tests/test_grid_wsc_characterization.py` (26 pass/0 skip) + `docs/refactor/SCI_01_GRID_WSC_CHARACTERIZATION.md` (matrice numérique : delta abs 0.25→20.0 selon corpus). Kappa-sigma n'est PAS divergent sur GPU ; GPU physique NOT_RUN (seams hermétiques fake-CuPy uniquement, pas de parité CPU↔GPU). Aucune correction ; toute décision scientifique (unifier `stack_core` sur PixInsight après qualification GPU physique, ou exposer une politique Grid explicite) reste un gate humain. |
| SCI-02 | Placeholder linear_fit dans core | **CARACTÉRISÉ / CLOS-NO-FIX** (mission ZM-SCI-02-STACKCORE-LINEARFIT-CHAR-20261004) : `stack_core` `normalize_method='linear_fit'` = placeholder exécutant le code identique à `median` (soustraction du médian par pixel), prouvé bit-exact et NON-affine (résidu max 240.33 vs 7.6e-06 pour le vrai linear-fit Grid). Un SEUL caller de production : `grid_mode._stack_weighted_patches_gpu` (inventaire AST), qui normalise EN AMONT (`_normalize_patches_gpu(method='linear_fit')`) puis passe `normalize_method='none'` au core → placeholder contourné en production. Grid CPU ne call jamais `stack_core` (monkeypatch interdit). Chemins linear-fit réels distincts : Grid = régression covariance/variance (`_fit_linear_scale`), classic = percentiles (`_normalize_images_linear_fit`/`_calculate_robust_stats_for_linear_fit`), les deux affines et ≠ placeholder. Normalisation `linear_fit` ≠ rejet `linear_fit_clip` (`_reject_outliers_linear_fit_clip`/`stack_linear_fit_clip`, placeholder rejet no-op). Preuve dynamique : `tests/test_stack_core_linear_fit_characterization.py` (16 pass/0 skip) + `docs/refactor/SCI_02_STACK_CORE_LINEAR_FIT_CHARACTERIZATION.md`. GPU physique NOT_RUN (seam hermétique fake-CuPy seulement) ; NaN/mask NOT_RUN ; callers externes hors contrat API = NOT_RUN. Aucune correction ; options de décision (garder documenté / retirer l'option core après audit API / implémenter un vrai affine core sous mission séparée) = gate humain. |
| SCI-03 | Masques/poids Grid CPU/GPU, all-invalid, aliases et winsor_limits | **CARACTÉRISÉ / CLOS-NO-FIX** (mission ZM-SCI-03-GRID-MASK-WEIGHT-CHAR-20261004) : Grid CPU et GPU-legacy masquent `weight<=0` AVANT rejet (`data_stack = where(weight>0, data, nan)`), mean utilise la MAGNITUDE positive des poids, median ignore la magnitude mais gate sur `weight>0`, all-invalid (par-pixel et tuile entière) → zéro + `weight_sum=0`. Grid GPU-core transmet les poids RAW (non masqués, zéro/négatifs inclus) à `stack_core` sans `winsor_limits` : mean → NaN à `weight_sum=0`, median inclut les frames à poids nul, poids négatifs NON clampés (`[10,100]`×`[2,-1]` → −80 vs 10 CPU). Aliases : CPU/legacy honorent `kappa`/`winsor` → helpers ; core forward verbatim → AUCUN rejet (seuls `kappa_sigma`/`winsorized_sigma_clip` exacts dispatchent). `_compute_frame_weight` : `none|unit|unity` → exposition seule ; `noise_fwhm` et valeur inconnue → variance-only ; NaN total → exposition. `winsor_limits` propagé EXACTEMENT (sentinel (0.2,0.1)) sur CPU/legacy (canonical + alias) ; DROP par le core (invariant). Preuve dynamique : `tests/test_grid_mask_weight_characterization.py` (31 pass/0 skip) + `docs/refactor/SCI_03_GRID_MASK_WEIGHT_CHARACTERIZATION.md` (3 divergences quantifiées + winsor_limits). GPU physique NOT_RUN (seams fake-CuPy + adaptateur core-CPU, pas de parité CPU↔GPU). Poids négatifs = sondage de contrat uniquement (`process_tile` clippe footprint `[0,1]` → poids production non négatifs). Aucune correction ; options (garder documenté / aligner core pre-mask/invalid sous mission science/API / canonicaliser aliases + propager winsor_limits sous mission compat) = gate humain. |
| SCI-04 | Variantes Classic/SDS/Phase 4.5 / low-N / chunking | **CARACTÉRISÉ / CLOS-NO-FIX** (mission ZM-SCI-04-CLASSIC-SDS-PHASE45-VARIANTS-20261004) : inventaire routes/différences + contrats purs SDS + différences normalisation/empilement Phase 4.5 vs Classic quantifiés, SANS consolidation/unification/refactor/fix/changement de défaut. Route inventory AST/import (dispatcher `run_hierarchical_mosaic` ≠ legacy `run_hierarchical_mosaic_classic_legacy`, appels Grid/legacy prouvés) ; 5 helpers SDS purs dynamiquement épinglés (`_mask_sds_low_coverage_pixels`, `_sanitize_sds_megatile_payload`, `_sds_compute_tile_payload`, `_sds_choose_reference_index`, `_normalize_sds_megatiles_photometry`) ; Phase 4.5 branch alpha-weighted inline (weighted-mean `nan_to_num`+clip `[0,1]`) vs branch rejet (wrappers réels ; `stack_kappa_sigma` ABSENT → retombe sur `stack_kappa_sigma_clip`) vs Classic (mêmes wrappers + `stack_aligned_images`) ; normalisation pre-stack `linear_fit` (slope clip `[0.25,4.0]`, intercept sur slope NON clipé, gate `max(5000,1%)` px → tiny no-op) vs `sky_mean` (percentile + delta médian, sans gate px) caractérisées par RECONSTRUCTION test-only (bloc inline non importable) ; low-N inventaire cross-ref TEST-04/SCI-03 (classic N≥3 reste non couvert) ; 3 mécanismes de chunking distingués (`_apply_safe_dynamic_chunk_profile` dynamique tous modes + fallback invalide, boucle Phase 4.5 `max_group` statique, budget VRAM Phase 5 dynamique, DBE RBF chunked statique). Preuve dynamique : `tests/test_classic_sds_phase45_variants_characterization.py` (34 pass/0 skip) + `docs/refactor/SCI_04_CLASSIC_SDS_PHASE45_VARIANTS.md`. NOT_RUN : exécution Phase 4.5 réelle, SDS sur données réelles, GPU physique/Grid GPU core, DBE heavy compute, classic N≥3, parité CPU↔GPU. Résolution drapeau SDS = STATIC-only (inline dupliqué ARCH-05). Aucune correction ; prérequis de consolidation inventoriés (gate humain). |
| SCI-05 | Canonical stacking + couverture (support positif / footprint taper / coadd) — Gate A archéologie + contrat | **GATE A ARCHAEOLOGY ACCEPT (Nono review-2) + GATE A2 DECISION FREEZE ACCEPT (Nono review-1 + Junior) ; Gate B OUVERT** (missions ZM-SCI-05-GATE-A-ARCHAEOLOGY-CONTRACT-20261005 + ZM-SCI-05-GATE-A2-CONTRACT-FREEZE-20261005) : archéologie GUI→runtime bornée/non-exhaustive mais acceptée comme preuve factuelle (Nono review-2) + témoins rouges hermetiques acceptés, SANS changement science/runtime/config/GUI/deps/version, SANS port couverture ; docs archéologie Gate A commités dans l'ancêtre `1b5c742`, freeze A2 accepté et intégré par le commit de clôture courant. Matrice `docs/science/SCI05_ARCHAEOLOGY_MATRIX.md` (renommage GUI→worker, 9 routes, symboles exécutés par colonne, verdict CANONICAL/DIVERGENT/PLACEHOLDER/SILENT_SCIENCE_FALLBACK/SILENT_SCIENCE_DEGRADATION/UNSUPPORTED/NOT_REACHABLE/NOT_RUN + classe preuve STATIC/DYNAMIC/RECONSTRUCTION/ROUTING_SEAM/PHYSICAL_GPU_NOT_RUN, sémantique stage-local vs outcome pipeline) ; contrat `docs/science/SCI05_CANONICAL_STACKING_CONTRACT.md` = `JUNIOR SCIENTIFIC ACCEPT — IMPLEMENTATION CONTRACT` (décisions résolues N1-N4/W1-W3/S1-S3/R1-R3/C1-C2/G1, tableau §16, pas DRAFT/registre ; pipeline gelé ALIGNED→NORM→WEIGHTS→SUPPORT→REJECT→COMBINE→CanonicalStackResult ; inventaire donor SHA `9b891de…` — `s_i = geometric*quality*footprint_taper`, `SUP_W1/W2`, `N_eff=W1²/W2`, `make_footprint_taper(px=8,floor=0)`, render final OFF ; défauts frais Tristan : taper ON, reconstruction OFF, all-invalid doit être NaN/invalid ou sentinel délibéré (PAS zéro arbitraire) ; discrepancy donor settings_state render=True vs engine False rapportée) ; témoin `tests/test_sci05_archaeology_characterization.py` (24 pass/0 skip) : tokens/labels Qt + radial legacy, placeholder `stack_core` linear_fit bit-exact == median + winsorized simplifié vs PixInsight WSC, `linear_fit_clip` no-op, `noise_fwhm` Classic = substitution variance SIFF photutils absent / no-weighting si star-free / partiel `1e-6`+`1.0` (vrai chemin réel + seam partiel) — TOUTES branches dégradées classées SILENT_SCIENCE_FALLBACK/DEGRADATION (politique absolue Tristan), Grid = variance-only séparé (aussi silent fallback), poids zéro/négatif + median ignore + all-invalid cross-ref, carte center-radial échoue l'invariance par translation vs concept footprint taper (RECONSTRUCTION, sans import donor), dispatch global-coadd par symbole exécuté (reconstruction dimensionnellement exacte N,H,W,C), gaps logging/provenance nommés (scan récursif rglob). Review-0 FINDINGS de fidélité/clarté résolus sans réouverture science ; review-1 ACCEPT. **Gate A2 accepté au commit `bc839ea`.** **Gate B1 ACCEPT (Nono review-0 + Junior), commit `be4039b`** (mission ZM-SCI-05-GATE-B1-CANONICAL-NORMALIZATION-20261005) : nouvelle couche pure CPU `src/zemosaic/core/canonical_stacking.py` (validation entrée canonique NHWC float32, sélection de référence, normalisation `none`/`linear_fit`/`sky_mean`, revalidation finie post-transform sans saturation), témoin `tests/test_sci05_canonical_normalization.py` (61 pass/0 skip), régressions 123 pass + packaging 17 pass, AUCUN branchement caller production. **Gate B2 ACCEPT (Nono review-1 + Junior), commit `d680d02`** (mission ZM-SCI-05-GATE-B2-CANONICAL-WEIGHTING-20261005) : `compute_canonical_quality_weights` + `canonical_noise_fwhm_available` + `CanonicalWeightingResult` (méthodes réelles `none`/`noise_variance`/`noise_fwhm`, luminance Rec.709 exacte, σ robuste sigma-clipped, FWHM Photutils `SourceCatalog.fwhm` déterministe sans fallback/deblend/seconde passe, préflight API Photutils incompatible) ; témoin `tests/test_sci05_canonical_weighting.py` (30 pass/0 skip), B1+B2 91 pass, régressions 123 pass + packaging 17 pass ; correction contract W1 `equivalent_fwhm`→`fwhm` (circulaire/seconds-moments égaux). **Gate C1 ACCEPT (Nono review-0 + Junior), commit `97d551b`** (mission ZM-SCI-05-GATE-C1-CANONICAL-REJECTION-20261005) : `CanonicalRejectionResult` + `reject_canonical_samples` (`none`/`kappa_sigma`/vrai `winsorized_sigma_clip`, masques NHWC par canal, poids ignorés hors gate actif, diagnostics bornés, `linear_fit_clip` rejeté `unsupported_removed_sci05`) ; témoin `tests/test_sci05_canonical_rejection.py` (41 pass/0 skip), C1+B1+B2 132 pass, régressions 123 pass + packaging 17 pass, fuzz indépendant Nono 600/600. **Gate C2 ACCEPT (Nono review-0 + Junior), commit `1d49bb9`** (mission ZM-SCI-05-GATE-C2-CANONICAL-COMBINE-20261005) : `CanonicalCombineResult` + `combine_canonical_samples` (`mean`/`median`, consomme la carte explicite pré-rejet `w_i=q*m*a` `(N,H,W)` float64 — PAS de `a=1` par défaut car taper ON ; validation poids `[0,1]`/zéro invalide-inactif/`<=q` ; weights partagés par canal, survivor canal-spécifique ; masque valide exact `estimator_weight_sum>0` ; mean=Σw, median=count ; restoration HW/HWC1/RGB ; invariant post-cast sans Inf-valide + `nonfinite_output_count`) ; témoin `tests/test_sci05_canonical_combine.py` (44 pass/0 skip), C2+B1+B2+C1 176 pass, régressions 123 pass + packaging 17 pass. **Gate D ACCEPT (Nono review-0 + review-1 R1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-D-CANONICAL-BACKEND-PARITY-20261005) : rejection+combine backend-neutral via un seul algorithme `xp`-générique (NumPy CPU default / CuPy GPU opt-in `backend="cpu"|"gpu"`, PAS de fallback silencieux, `canonical_gpu_available()` lazy) ; résultats GPU host-convertis (contrat résultat inchangé) ; `cupy.nanquantile` absent → équivalent NaN-aware **bit-exact** (forme `_lerp` two-sided NumPy — R1) ; témoin `tests/test_sci05_canonical_gpu_parity.py` (30 pass/0 skip, MX150 physique + monkeypatch unavailable + bit-exact quantile) ; C1/C2 CPU byte-identiques (176 pass), régressions 123 pass + packaging 17 pass. **Gate E1 ACCEPT (Nono review-0 + Junior), prêt au commit local** (mission ZM-SCI-05-GATE-E1-CANONICAL-SUPPORT-TAPER-20261005) : nouveau module pur CPU `src/zemosaic/core/canonical_support.py` — `make_footprint_taper` (EDT primary + chamfer fallback, jamais radial), `PositiveSupportAccumulator`/`accumulate_support_pair` (paire atomique `SUP_W1`/`SUP_W2` float64 fail-before-mutation + `N_eff` dérivé), `build_canonical_estimator_weights` (carte explicite `(N,H,W)` float64 `w=q*m*a` consommée par C2) ; témoin `tests/test_sci05_canonical_support.py` (39 pass/0 skip). **Gate E2 ACCEPT (Nono review-0 + review-1 R1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-E2-CANONICAL-ENGINE-20261005) : nouveau module `src/zemosaic/core/canonical_engine.py` — `CanonicalStackRequest`/`CanonicalStackResult` + `run_canonical_stack` (assemble B1→B2→E1→C1→C2, support accumulé pré-rejet donc rejection-indépendant, provenance bornée, `equalize_rgb=True` → erreur deferred explicite, backend `gpu` forwardé C1/C2, provenance backend **per-stage** — B1/B2/support CPU, C1/C2 demandé) ; témoin `tests/test_sci05_canonical_engine.py` (26 pass/0 skip). **Gate E3 ACCEPT (Nono review-0 + Junior), prêt au commit local** (mission ZM-SCI-05-GATE-E3-CANONICAL-COVERAGE-RENDER-20261005) : nouveau module pur CPU `src/zemosaic/core/canonical_render.py` — `coverage_aware_render` (formule donor-exacte `B+D` detail blend, `alpha=clip(1-N_eff/n_ref,0,1)`), `render_preview` + `coverage_render_event` ; preview-only, pur (ne mute jamais science/support), no-gain/no-inpaint ; témoin `tests/test_sci05_canonical_render.py` (16 pass/0 skip). **Gate E4 ACCEPT (Nono review-0 + Junior), prêt au commit local** (mission ZM-SCI-05-GATE-E4-CANONICAL-RGB-EQUALIZE-20261005) : nouveau module pur `src/zemosaic/core/canonical_equalize.py` — `equalize_rgb_medians_canonical`/`equalize_rgb_medians_copy` (port exact décision J, out-of-place, parity prouvée vs `equalize_rgb_medians_inplace`) + engine wiring `equalize_rgb` (post-combine, RGB-only, applied/no-op provenance) ; témoins `tests/test_sci05_canonical_equalize.py` (17) + engine equalize path. **Gate E5a ACCEPT (Nono review-0 + Junior), prêt au commit local** (mission ZM-SCI-05-GATE-E5A-COVERAGE-SETTINGS-MIGRATION-20261005) : `zemosaic_config.py` — deux booleans publics `coverage_support_taper=True`/`coverage_aware_reconstruction=False` + `migrate_coverage_settings` (pur, idempotent, AUCUN mapping de valeurs legacy radial → taper px/floor, legacy radial inert) intégré non-breaking dans `load_config` ; 4 clés locale × 7 fichiers ; témoin `tests/test_sci05_coverage_settings_migration.py` (14 pass/0 skip). **Gate E5b ACCEPT (Nono review-0 + Junior), prêt au commit local** (mission ZM-SCI-05-GATE-E5B-COVERAGE-GUI-CONTROLS-20261005) : `zemosaic_gui_qt.py` — retrait des widgets radiaux legacy + deux checkboxes canoniques (`coverage_support_taper` default True / `coverage_aware_reconstruction` default False) liées à config + clés locale ; `zemosaic_gui.py` (Tk) — mêmes booleans + radial forcé inerte ; `zemosaic_align_stack_gpu.py::_compute_radial_weight_map` — no-op retournant None (radial runtime-inert) ; témoin `tests/test_sci05_coverage_gui_controls.py` (6 pass/0 skip). **Gate F1 ACCEPT (Nono review-0 + review-1 R1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-F1-REMOVE-PLACEHOLDERS-20261005) : décision R3 — `linear_fit_clip` retiré des choix Qt/Tk + les 2 sites worker lèvent `unsupported_removed_sci05` (jamais de migration silencieuse) + token GPU aligné + helpers `stack_linear_fit_clip`/`_reject_outliers_linear_fit_clip` marqués legacy/unreachable ; décision N4 — `stack_core` `linear_fit` lève `unsupported_removed_sci05` (placeholder médiane supprimé) ; témoin `tests/test_sci05_gate_f1_placeholder_removal.py` + mises à jour bornées des témoins. **R1 (Nono optional, plié — garde centralisée)** : `_validate_rejection_token` dans `stack_aligned_images` (normalise strip/lower + valide — `linear_fit_clip` case/whitespace → `unsupported_removed_sci05`, token inconnu → erreur explicite, supported `none`/`kappa_sigma`/`winsorized_sigma_clip` inchangé bit-identique). **Gate F2 ACCEPT (Nono review-0 + review-1 R1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-F2-CLASSIC-ROUTE-CONVERGENCE-20261005) : `align_images_in_group` surface les footprints géométriques (`return_footprints=True`, backward-compatible 2-tuple conservé) + propagation UNCONDITIONNELLE (décision R1/Option A) ; worker threads les footprints (`valid_aligned_images`/`valid_footprints` → `_stack_master_tile_auto` → `_stack_master_tile_cpu` → `stack_aligned_images(geometric_support=…)`) ; `stack_aligned_images` route via `run_canonical_stack` (backend cpu, supported methods, `coverage_support_taper` consommé pour la carte estimator-weight, legacy radial inert, render preview-only, frame sans footprint honnête exclue) ; témoin `tests/test_sci05_gate_f2_classic_route.py` (10 pass/0 skip). SDS/Grid/Phase4.5/coadd = prochains lots F ; normalisation/weighting GPU parity différé ; AUCUNE convergence caller SDS/Grid/Phase4.5. **Gate F3 ACCEPT (Nono review-0 F1 + review-1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-F3-RADIAL-INERT-ALL-ROUTES-20261005) : points d'application radiaux restants rendus inertes (`stack_aligned_images` legacy `final_radial_weights_list`, worker tile-feather `base_weight` + Phase-5 `radial2d`) — `apply_radial_weight=True` bit-identique à `False` ; `ZMT_RADW` toujours False, `ZMT_RADF`/`ZMT_RADP` omis ; clés deprecated lisibles-inertes ; Grid n'a jamais appliqué de carte radiale ; GPU reste inerte (E5b) ; témoin `tests/test_sci05_gate_f3_radial_inert.py` (8 pass/0 skip). **F1 (Nono MEDIUM, plié — header honnête)** : headers mosaïque finale `STK_RADW` toujours False (inert), `STK_RADFF`/`STK_RADPW`/`STK_RADFLR` omis (jamais de claim radial appliqué). **Gate F4 ACCEPT (Nono review-0 R1 + review-1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-F4-GRID-ROUTE-CONVERGENCE-20261005) : `_stack_weighted_patches`/`_stack_weighted_patches_gpu` routent via `run_canonical_stack` (backend cpu) quand le support géométrique est threadé (footprints WCS-reprojection × alpha de `_reproject_frame_to_tile`, binarisées `> 0`, jamais dérivées brightness/NaN) ; `coverage_support_taper` consommé ; GPU route la scène canonique sur CPU (explicite — B1/B2/support CPU-only) ; témoin `tests/test_sci05_gate_f4_grid_route.py` (9 pass/0 skip). SDS/Phase4.5/coadd = prochains lots F. **Gate F5 ACCEPT (Nono review-0 F1 + review-1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-F5-GLOBAL-COADD-CONVERGENCE-20261005) : label `Winsorized` du global coadd (`global_coadd_method == "winsorized"`) route vers `run_canonical_stack` (`winsorized_sigma_clip`, par chunk spatial, backend cpu) ; le clip percentile-winsorized divergent (`np.nanpercentile`) est RETIRÉ (pas d'approximation same-label) ; support géométrique = footprint reprojeté (`> 0`), normalization/weighting `none`, combine `mean` ; limite notée : `kappa_sigma` global-coadd utilise `coadd_k=2.0` (vs sigma canonique 3.0) — lot ultérieur ; témoin `tests/test_sci05_gate_f5_global_coadd.py` (5 pass/0 skip). SDS/Phase4.5 = prochains lots F. **F1 (Nono MEDIUM, plié — option a, convergence des 4 labels)** : les 4 labels `global_coadd_method` (`Mean`/`Median`/`Kappa-Sigma`/`Winsorized`) routent tous via `run_canonical_stack` (`_finalize_chunked`, par chunk spatial, backend cpu) — `mean`→none/mean, `median`→none/median, `kappa_sigma`→kappa_sigma (sigma canonique 3.0)/mean, `winsorized`→winsorized_sigma_clip/mean ; finalizers ad-hoc `_finalize_mean`/`_finalize_kappa_sigma` + clip percentile RETIRÉS du dispatch ; R2 : `coadd_k` et `winsor_limits` accepted-but-INERT (défauts gelés canoniques) ; témoin `tests/test_sci05_gate_f5_global_coadd.py` (7 pass/0 skip) + maj bornée témoin archéologie `test_finalizer_dispatch_symbols`. SDS/Phase4.5 = prochains lots F. **R2 (Junior, plié — divergence backend GPU-helper)** : le chemin helper GPU legacy (`reproject_and_coadd_wrapper` avec `combine_function=coadd_method`, `coadd_k`, `winsor_limits`) est DÉSACTIVÉ pour le global coadd (`gpu_helper_supported=False` + log explicite `global_coadd_helper_legacy_disabled_canonical_only`) — CPU et GPU exécutent la scène canonique CPU, label identique → sémantique canonique ; `coadd_k`/`winsor_limits` inert partout ; témoin `tests/test_sci05_gate_f5_global_coadd.py` (8 pass/0 skip). SDS/Phase4.5 = prochains lots F. **Gate F6 ACCEPT (Nono review-0 + review-1, Junior), prêt au commit local** (mission ZM-SCI-05-GATE-F6-SDS-PHASE45-CONVERGENCE-20261005) : D1 (gain inter-master documenté comme opération explicite non-canonique), D2 (référence SDS alignée N1 canonique — max valid support count), D3 (constante dérivée 1% retirée) ; A1 (extension additive du noyau : `CanonicalStackRequest.taper` accepte un taper explicite (N,H,W)/(H,W) `[0,1]`) ; A2/D4 (combine alpha-weighted Phase 4.5 → `run_canonical_stack` avec alpha en taper explicite) ; A3/D5 (appels stack legacy → rejection canonique avec footprint WCS honnête ; sous-chemin sans footprint honnête = deferred non-silencieux) ; témoin `tests/test_sci05_gate_f6_sds_phase45.py` (7 pass/0 skip) + `tests/test_sci05_canonical_engine.py` (+5 A1). SDS/Phase4.5 = convergés. **R3 (Junior, plié — dispatch legacy WSC/Kappa résiduel)** : `_stack_master_tile_cpu` et `_stack_mosaics` n'appellent plus les wrappers legacy `stack_winsorized_sigma_clip`/`stack_kappa_sigma_clip` ; tous les algos de rejection supportés routent via le moteur canonique (`stack_aligned_images`/`run_canonical_stack`) avec support géométrique honnête (footprint/coverage WCS) ; wrappers legacy marqués LEGACY/UNREACHABLE (align_stack) ; sous-chemin `_stack_mosaics` sans footprint honnête = deferred non-silencieux (`sds_final_no_honest_footprint_deferred`) ; témoin `tests/test_sci05_gate_f6_sds_phase45.py` (10 pass/0 skip). **R4 (Nono LOW, plié — pas de combine non-canonique sur le sous-chemin deferred)** : le sous-chemin SDS sans footprint honnête SKIP désormais (retourne NaN, event non-silencieux `sds_final_no_honest_footprint_deferred` avec count/reason) — plus de `np.nanmean`/`np.nanmedian` non-canonique fabriqué ; chemin canonique inchangé ; témoin `tests/test_sci05_gate_f6_sds_phase45.py` (11 pass/0 skip). **Gate G ACCEPT (Nono review-0), SCI-05 CLOSED — prêt PR vers beta (approbation Tristan requise)** (mission ZM-SCI-05-GATE-G-FINAL-AUDIT-20261005) : fix borné des doubles de test `align_images_in_group` (`tests/test_phase3_adaptive_invariants.py`) acceptant `return_footprints`/`**kwargs` (retour 3-tuple avec footprints honnêtes) ; suite complète 975 pass/0 fail ; audit final : AUCUNE divergence supportée restante (pas de clip percentile-winsorized stacking, pas de constante epsilon stacking, pas de sentinelle zéro, pas d'appel legacy WSC/kappa/linear, pas de `make_radial_weight_map` live, pas de fallback backend silencieux).** |
| TEST-01 | 11 imports plats importorskip Phase 3 | RÉSOLU (mission ZM-ARCH-TEST01-PHASE3-IMPORTS-20261003) : imports plats → qualifiés (`zemosaic.zemosaic_worker`, `zemosaic.parallel_utils`) dans `tests/test_phase3_adaptive_invariants.py`, sans alias produit. Résultat témoin : baseline historique 22 pass/11 skip → 33 pass/0 skip ; `tests/test_packaging.py` 17 pass. |
| ARCH-01 | Invocations supportées de Phase 4.5 / fallback Tk | UNKNOWN jusqu'à preuve/revue; conserver |
| ARCH-02 | Contrats externes / frozen / API programmatique | Inventaire incomplet; ambiguïté bloque une suppression, pas toute R0 |
| MANUAL-01 | M106 avant/après sur GPU réel | RÉSOLU : trois runs cupy/MX150 (BASE, CANDIDATE, repeat même-build), preuves sous `/home/tristan/M106/gate_evidence_20261004/`; technique `INCONCLUSIVE/HOLD` sous critère bit-identique, verdict visuel/scientifique Tristan `ACCEPT`. Les autres plateformes restent NOT_RUN. |
| TEST-02 | Ancien `tests/test_version_gpu.py` : script d'impression CUDA sans `test_*` (0 item collecté), sans appelant produit | **RÉSOLU / SUPPRIMÉ** : diagnostic manuel devenu redondant avec ZeAlfie (détection hôte, planification `NVIDIA_CUDA`, closure accélérée et probe CuPy/NVRTC). Sa suppression ne retire aucun test ni garde runtime ZeMosaic ; `test_cupy_platform_guard.py`, les tests de budgets/télémétrie et les gardes GPU produit restent en place. |
| TEST-03 | Témoin de propagation dispatch GUI/config → `run_hierarchical_mosaic_process` → `run_hierarchical_mosaic` (kwargs effectifs) | RÉSOLU (mission ZM-ARCH-WITNESS-DISPATCH-20261003) : nouveau `tests/test_dispatch_propagation_witness.py`, 14 pass/0 skip en 4.63s. Caractérise rename GUI→worker (stacking_* → stack_*), parsing `stacking_winsor_limits`→`parsed_winsor_limits` (tuple, fallback (0.05,0.05)), promotion suffixe `_config`, drop silencieux des kwargs inconnus (risque architectural), défauts `stack_ram_budget_gb_config=0.0`/`num_base_workers_config=0`, préservation des valeurs falsy (False/0/""/numériques), invocation unique sans travail lourd. Aucun fix produit ; les autres témoins alors ouverts sont désormais suivis par TEST-04/05/06. |
| TEST-04 | Témoin low-N / all-invalid / zero-weight du stacking (Grid CPU vs `stack_core` vs classic N<3) | RÉSOLU (mission ZM-ARCH-WITNESS-STACK-EDGES-20261003) : nouveau `tests/test_stacking_low_n_all_invalid_witness.py`, 18 pass/0 skip en 2.15s, CPU uniquement, float32 HWC 2×2×1. Épingle : Grid CPU all-invalid → zéros vs `stack_core` → NaN (pixel all-invalid 0.0 vs NaN en mean) ; median Grid CPU ignore la magnitude mais traite `weight<=0` comme invalide, `stack_core` ignore totalement les poids en median ; classic kappa N=1/N=2 et winsorized N=1 (warning « needs >=3 images ; forcing CPU ») renvoient des stacks valides `rejected=0.0`. Divergence SCI-03 caractérisée, PAS corrigée. Restent ouverts (non testés) : parité CPU↔GPU, Grid GPU `stack_core`, classic N≥3 / SDS, `linear_fit` numériques, Phase 4.5 exécution ; cache/reprise et spawn sont désormais couverts par TEST-06/05. |
| TEST-05 | Témoin spawn réel du worker package-qualifié (`zemosaic.zemosaic_worker.run_hierarchical_mosaic_process`) | RÉSOLU (mission ZM-ARCH-WITNESS-SPAWN-20261003) : nouveau `tests/test_spawn_worker_process_witness.py`, 1 pass/0 skip en ~6.5s (stable sur 3 exécutions, aucun sleep arbitraire). Spawn réel via `multiprocessing.get_context("spawn")` + `Queue` réelle, cible = fonction produit package-qualifiée (prouve pickling/import sous spawn, pas un double in-process). Invocation volontairement sans arguments scientifiques → `TypeError` (args requis manquants) attrapé par le wrapper → protocole `PROCESS_ERROR` (le champ `error` identifie « missing … input_folder », pas un échec import/pickle) suivi de `PROCESS_DONE` (finally), sortie propre exitcode=0, aucun enfant fuité. `crash_breadcrumbs_mode="off"`, HOME/XDG/cwd/temp isolés, pas de heartbeat, aucun fichier breadcrumb/state, cwd vide, queue fermée/jointe. Ne témoigne PAS d'une exécution scientifique réussie, ni GPU/solver, ni parité GUI end-to-end, ni plateformes hors environnement exécuté (Linux x64). Cache/reprise est désormais couvert par TEST-06 ; l'exécution scientifique réussie reste NOT_RUN. |
| TEST-06 | Témoin cache/reprise/checkpoint Classic (`_safe_load_cache`, checkpoint Phase 5, reprise/écriture Phase 1) | RÉSOLU (mission ZM-ARCH-WITNESS-CACHE-RESUME-20261004) : nouveau `tests/test_cache_resume_characterization_witness.py`, 18 pass/0 skip en ~2.8s, CPU uniquement, petits tableaux + Header/WCS Astropy réels sous `tmp_path`. Épingle : `_safe_load_cache` charge en memmap `mmap_mode="r"` (retourne `np.memmap`) sans pickle ; retente une fois sans memmap sur OSError WinError 1455 avec callback `stack_mem_fallback_memmap_to_ram` (lvl WARN) ; relance les OSError non-1455 et les échecs du fallback. Checkpoint Phase 5 : écrit/relit un artefact mosaïque HWC float32 + coverage/alpha HW, manifest `schema_version=1`/`pipeline=classic_legacy`, rejet signature/schema/pipeline/output-shape ; méthode vérifiée seulement si stockée non vide, counts seulement pour attentes entières positives. Mosaïque manquante/corrompue/mauvaise dimension rejette tout ; coverage/alpha manquants ou mal dimensionnés → None (mosaïque valide encore chargée), normalisation singleton trailing channel `(H,W,1)→(H,W)`, aucun `.tmp` résiduel après succès. Reprise Phase 1 : écrit manifest + `phase1_processed_info.json` + `phase1.done` ; mode auto signature exacte → `(True, entries, "ok")` avec Header/WCS reconstruits ; auto mismatch rejeté / force mismatch procède avec warning ; cache partiel → `(False, entries, raison)` avec compteurs ; raisons épinglées (manifest/schema/pipeline/processed-info/manifest missing). Seam test-only par reconstruction des code objects imbriqués via `run_hierarchical_mosaic_classic_legacy.__wrapped__.__code__` (identité prouvée, aucune copie de formule). Reste NOT_RUN : entrée mosaïque non-float32 (champ dtype manifest pré-cast), branches permissives méthode vide/count attendu non positif, rétention/refcount par tuile, nettoyage `run_end`, réutilisation master tiles, égalité scientifique reprise-vs-neuf, interruption/crash recovery, sémantique filesystem cross-platform. |
| TEST-07 | Témoin crash-breadcrumb / last-state / heartbeat (`_configure_crash_breadcrumbs`, `_safe_runtime_snapshot`, `_emit_crash_breadcrumb`, lifecycle `run_hierarchical_mosaic_process`) | RÉSOLU (mission ZM-ARCH-R2-LOT2A-CRASH-BREADCRUMB-WITNESS-20261004) : nouveau `tests/test_crash_breadcrumbs_characterization_witness.py`, 37→47 pass/0 skip en ~3.7s (stable sur 2 exécutions), CPU uniquement, aucun GPU requis (fakes patchés), isolation HOME/XDG/cwd/`tmp_path`. Épingle : `_configure_crash_breadcrumbs` modes valides always/errors_only → chemins exacts `<out>/worker_crash_breadcrumbs.jsonl` + `<out>/worker_last_state.json` + création du répertoire ; mode off → chemins None sans créer le répertoire ; mode invalide → fallback always ; output None/"" → chemins None ; échec best-effort (str() raise) → chemins None ; les 3 globals restent observables. `_safe_runtime_snapshot` : pid/ppid/ts_unix toujours présents ; champs RAM psutil (ram_used_mb/ram_total_mb/ram_pct) présents si `virtual_memory()` OK, absents (fail-open) si raise ; champs GPU absents à BASE (car `zemosaic_utils` n'a pas `get_gpu_vram_info`), présents si fake helper patché, absents (fail-open) si probe GPU raise. `_emit_crash_breadcrumb` : JSONL append-only + last-state remplacé ; record = event + iso (`utcnow().isoformat()+"Z"`) + snapshot + payload (payload override les clés snapshot : `pid`/`ppid`/`ts_unix` témoignés) ; filtre `errors_only` sur sous-chaîne ERROR/EXCEPTION/CRASH insensible à la casse (cas encadrés `WORKER_ERROR_X`/`foo_exception_bar`/`preCRASHpost` passent, quasi-ratés `ERR`/`EROR`/`EXCEPTON`/`CRSH` non) ; off/sans-chemin = no-op ; `default=str` ; échecs d'écriture JSONL/state (indépendants — l'un survit pendant que l'autre échoue — et simultanés) et échec du lock avalés. Concurrence : 8 threads × 25 events → 200 lignes JSONL valides non entrelacées, last-state valide, join borné par thread (aucun thread vivant avant lecture). Lifecycle in-process (runner scientifique monkeypatché + queue double) : off → aucun fichier/thread, seul PROCESS_DONE ; always → WORKER_START + WORKER_DONE (`graceful_stop=False`), thread heartbeat démarré puis terminé, handlers de signal restaurés ; exception contrôlée → WORKER_START + WORKER_EXCEPTION(error+traceback) + WORKER_DONE, queue PROCESS_ERROR(error/breadcrumb_path/last_state_path) puis PROCESS_DONE ; errors_only supprime le lifecycle non-erreur (seul WORKER_EXCEPTION en cas d'erreur, aucun fichier sur run propre) ; cadence heartbeat témoignée par événement borné (WORKER_HEARTBEAT entre START et DONE, intervalle borné ≥0.5s), pas de sleep arbitraire. Reste NOT_RUN : cadence/timing précis du heartbeat sous charge (seule existence+ordre témoignés), probe GPU physique (fakes seulement), spawn réel avec crash mode on, exécution scientifique réussie, Windows/macOS. |
| ARCH-03 | `core/cuda_utils.py::enforce_nvidia_gpu` sans importeur trouvé ; `cuda_utils.py::gpu_supported/enforce_nvidia_gpu` sans caller (mais `CUPY_AVAILABLE` vivant via `_app.py`) | SUSPECTED DEAD — vérifier avant R1 ; ne pas supprimer sans preuve |
| ARCH-04 | `zemosaic_gui.py` (Tk legacy) sans importeur dans src ni dans `ZeMosaic.spec` hiddenimports ; `--tk-gui` rejeté par `_app._determine_backend` | DORMANT BUT REACHABLE — conserver jusqu'à preuve contraire |
| ARCH-05 | `run_hierarchical_mosaic_classic_legacy` contient son propre bloc de résolution SDS (`worker:23690-23715`) + helpers SDS partagés, en plus du dispatcher `run_hierarchical_mosaic` | Duplication à documenter avant toute extraction Classic/SDS ; ne pas consolider sans témoin |
| ARCH-06 | `stack_core` réutilisé par Grid GPU ; `linear_fit` = placeholder médian ; winsorized GPU simplifié vs CPU établi | SCI-01/02 confirmés ; Grid GPU normalise en amont et passe `none` au core (`grid_mode:2012`) |
| SCI-07 | Helpers Phase 4.5 `estimate_affine_photometry` / `apply_affine_photometry` / `micro_align_stack` ABSENTS à BASE | Runtime `hasattr` `False False False` ; gates `worker:7412-7414` éteignent micro-align/intra-group affine/legacy affine/global-affine inter-super (`:8034`,`:7701-7705`,`:7766`,`:8084`,`:8095`,`:8151-8154`,`:8865`) → DORMANT/UNREACHABLE, PAS PROVEN DEAD. Seule normalisation 4.5 ACTIVE = `linear_fit`/`sky_mean` pré-stack (`worker:8202-8290`) + gain-only inter-super post-stack (`:8622-8832`). Ne pas documenter comme couverture photométrique exécutée |
| PERF-01 | Élagage top-K séquentiel des paires (Windows séquentiel) | DIFFÉRÉ / NON implémenté : pour la simulation M106, K=8 → 279→155 paires, graphe connecté, estimé linéaire ~144→80s. K=8 ne change PAS effective_workers=1 / aucun ThreadPool. Comparaison scientifique vs graphe complet requise avant toute activation. Backlog, pas une feature complétée. |

Pour toute découverte ajouter : ID, SHA, chemin/caller, attendu vs observé,
commande/témoin/artefacts, impact, raison de non-correction, prochain pas unique.

## 11. Orchestration et discipline Git (lors du lancement seulement)

Junior dirige et accepte; Coco implémente des lots bornés, Nono révise indépendamment.
Pas de délégation déclenchée par la seule présence de ce fichier.

- ARCHIVAL/GATE — Une mission_id stable par lot indépendant, rapports durables distincts par
  itération sous `/home/tristan/.openclaw/workspace/.a2a-reports/`.
- ARCHIVAL/GATE — Coco neuf/reset seulement à la frontière d'une mission indépendante, après
  preuve d'inactivité; conserver son contexte pendant revue et REWORK (maximum 3).
  Un contexte Nono frais peut servir un lot indépendant, jamais effacer un actif.
- ARCHIVAL/GATE — Transport direct `sessions_send(timeoutSeconds=0)`, callback vers l'exact
  `sourceSession` fourni par OpenClaw, rapport écrit avant callback, puis REPLY_SKIP;
  pas de polling, spawn natif ou attente synchrone Coco/Nono. Inclure le contrat
  intégral de callback des instructions workspace dans chaque délégation.
- ARCHIVAL/GATE — Junior vérifie dépôt/diff/tests/artefacts indépendamment; Nono ne modifie pas
  le dépôt lors d'une revue. Une revue R0 n'exempte pas les futurs diffs de revue.
- ARCHIVAL/GATE — Inspecter branche/HEAD/status avant et après chaque lot; préserver les
  changements tiers. Tout commit code/test : témoins, diff-check, revue Nono du
  diff exact avant commit; nouveau delta après revue = revue à compléter.
- ARCHIVAL/GATE — Commits locaux petits/réversibles : docs, témoin, suppression ou extraction
  distincts. Pas de « cleanup everything », pas de réécriture d'historique.
- ARCHIVAL/GATE — Gates humaines : changement de périmètre/architecture fondamentale/API
  supportée, retrait de fonctionnalité, migration destructive/dépendance majeure,
  publication. Aucun push/merge/tag/release autorisé par cette mission.

## 12. Comparaison M106, clôture et rapport final

- [x] AVANT refactor scientifique/structurel : localiser le corpus historique
  « M106 de l'enfer », créer manifeste de fichiers/hash/config/ordre, capturer
  dépendances et backend réel; conserver référence BASE SHA, logs et sorties
  dans un espace distinct, sans écraser/modifier les brutes de référence.
- [x] Préserver sorties intermédiaires pertinentes : nombre/identité des tuiles,
  FITS data/WCS/headers scientifiques, poids/alpha/coverage, NaN, rejets, stats et
  diagnostics des trous/seams. Identifier métadonnées volatiles séparément.
- [x] Comparer candidat et base avec même corpus/config/matériel; un résultat
  simplement « joli » ne prouve pas la science. M106 ne remplace pas les témoins
  des modes/options qu'il n'exerce pas (SDS/Grid/GPU/Phase 4.5 notamment).
- [x] Tristan réalise/valide l'acceptation manuelle finale; baseline indisponible
  ou test non passé = HOLD explicite, jamais promotion automatique.
- [x] Écrire `docs/refactor/FINAL_REPORT.md` : BASE SHA, branche, HEAD final, statut
  worktree, synthèse R0/classifications, suppressions et preuves, extractions,
  contrats R3, anomalies intentionnellement NON corrigées, tests/commandes/durées/
  résultats/skips, revues Nono, commits, TODO restants et gates manuelles.
  (accepté techniquement le 2026-10-04 après Nono review-0 + Junior ; le commit docs
  est séparé, HEAD d'implémentation audité `1282cfe`.)
- [x] Distinguer le résultat mécanique `INCONCLUSIVE/HOLD` du verdict final **HUMAN SCIENCE ACCEPT**. Même après M106,
  une publication nécessite une autorisation distincte.

Succès = chemin officiel évident, code mort démontré, responsabilités moins
couplées/testables, science préservée et anomalies isolées. Pas « beaucoup de
fichiers » ni « moins de lignes à tout prix ».

> Archaeology first. Mechanical refactor second. Science unchanged. M106 last.
