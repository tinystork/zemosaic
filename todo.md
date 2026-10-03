# ZeMosaic — Architectural Cleanup R0–R3

## Statut et autorisation

**PRÉPARÉ — NON DÉMARRÉ**, le 2026-10-03, à la demande de Tristan.

Ce fichier conserve la mission et les compléments issus d'une lecture ciblée du
code. Ce n'est ni le rapport d'archéologie R0, ni une preuve de validation
scientifique. Ne pas lancer R0–R3, Coco/Nono, pytest, builds ou M106 sans une
nouvelle instruction de lancement. Aucune suppression/extraction réalisée.

- [x] Vérifier l'identité du dépôt et actualiser les références distantes.
- [x] Vérifier base, version et propreté initiale.
- [x] Vérifier les principales hypothèses de la mission par lecture du code.
- [x] Créer la branche locale dédiée depuis le SHA exact.
- [x] Écrire le plan, ses corrections et ses gates.
- [ ] Recevoir l'instruction de lancer la mission.
- [ ] Exécuter R0 et obtenir sa revue indépendante.
- [ ] Exécuter les seules suppressions R1 prouvées sûres.
- [ ] Exécuter les extractions R2 acceptées une par une.
- [ ] Finaliser la carte des contrats R3 et le rapport.
- [ ] Obtenir l'acceptation scientifique manuelle de Tristan sur M106.

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
| HEAD préparé | BASE SHA, aucun commit de préparation |
| Version | `4.7.0`, `src/zemosaic/__init__.py`; concorde avec `version.txt` |
| Changements de préparation | Ce seul `todo.md`, laissé non commité |

Ne pas réinitialiser une branche existante, rebaser, fusionner ou avancer la base
opportunistement. Aucun push, merge, tag, release, bump de version ou modification
de `main`/`beta`. Les commits futurs restent petits et locaux.

### Gate de reprise

- [ ] Relire ce fichier et les instructions locales; vérifier qu'aucun worker n'est actif.
- [ ] Refaire `git fetch origin --prune`, `git status --short`,
  `git branch --show-current`, `git rev-parse origin/main origin/beta HEAD`,
  `git log --oneline --decorate -10` et `git worktree list` dans le vrai checkout.
- [ ] Si une branche distante diffère de BASE SHA : STOP et rapport, sans changement de base.
- [ ] Préserver toute modification nouvelle de Tristan, ne rien stasher/effacer implicitement.
- [ ] Résoudre le checkpoint documentaire : ce `todo.md` rend volontairement le
  worktree non propre après préparation. Lors du lancement autorisé, inspecter et
  enregistrer ce document seul dans un commit local `docs: plan ZeMosaic architectural cleanup`;
  puis vérifier le worktree propre AVANT les travaux R0/tests. Ne pas le supprimer
  pour satisfaire artificiellement la gate. Aucun commit n'a été créé aujourd'hui.
- [ ] Garder BASE SHA comme référence scientifique immuable même après les commits docs/tests.

## 2. Avis de cohérence — constats ciblés, pas R0 complet

La mission est cohérente avec le code de cette base. Les avertissements Classic,
Phase 4.5, helpers Tk et divergence Grid WSC sont fondés. Les précisions suivantes
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
| Grid CPU/GPU | `grid_mode.py:1913` utilise les rejets établis; `:1984` / `:2018` appelle `stack_core` côté GPU. Le core contient WSC simplifié (`zemosaic_stack_core.py:325`). Divergence structurelle confirmée, impact numérique NON MESURÉ ici. |
| Nuance linear_fit | Le placeholder existe à `zemosaic_stack_core.py:294`, mais Grid GPU normalise en amont puis passe `normalize_method='none'` (`grid_mode.py:2012`). Ne pas lui attribuer sans preuve ce placeholder. Cartographier séparément normalisation linear-fit et rejet linear-fit. |
| Monolithe | `zemosaic_worker.py` : 38 882 lignes, 1 736 598 octets. La taille n'est pas un critère de succès en elle-même. |
| Témoins Phase 3 | 11 appels `pytest.importorskip("zemosaic_worker")` persistent dans `tests/test_phase3_adaptive_invariants.py`, alors que le package interdit les alias plats (`tests/test_packaging.py`, NamespaceTests). Risque de skips silencieux; aucun résultat pytest revendiqué à ce stade. |
| Tests structurels | Des tests extraient des sous-chaînes/commentaires du worker. Une extraction peut casser leur localisation sans casser le comportement. Migrer les preuves sans affaiblir les assertions, avec revue. |
| Garde CI Tk | `.github/workflows/no-tk-on-official-path.yml` interdit seulement certains imports Tk directs. Ce n'est PAS une preuve d'absence de dépendance transitive ou de fallback Tk. |

## 3. Compléments nécessaires à la mission

- [ ] **Définir les chemins supportés en R0** : GUI installée, `python -m`, checkout,
  distribution frozen, éventuelle API programmatique documentée, scripts externes.
  Une fonction importable n'est pas automatiquement publique; l'absence de caller
  local ne prouve pas son abandon. En cas de doute : UNKNOWN, conserver.
- [ ] **Préparer R3 dès R0** pour les zones touchées, puis le compléter en fin de
  mission. La carte du stacking et ses témoins doivent précéder une extraction,
  pas être découverts après R2.
- [ ] **Définir une comparaison avant/après par chemin**, à entrées/config/environnement
  identiques, distincte d'une comparaison CPU/GPU. Bit-identité lorsque déterministe;
  sinon tolérances explicites justifiées sur la baseline, définies AVANT changement,
  jamais élargies pour faire passer le candidat. Comparer masques et couverture
  autant que les pixels; ne pas exiger une parité entre deux chemins déjà divergents.
- [ ] **Inventorier l'environnement réel** : Python, dépendances/versions, OS,
  GPU/driver/CUDA/CuPy, BLAS/threads, graines, disponibilité solveurs/catalogues.
  Distinguer backend demandé, effectivement exécuté et fallback. Un skip GPU ou
  un mock CuPy ne vaut pas qualification GPU physique.
- [ ] **Isoler les tests** : répertoires temporaires pour profil/config/cwd/XDG,
  copies de FITS; pas de modification du profil ni des brutes de Tristan, pas de
  téléchargement/catalogue/réseau implicite. Consigner passed/failed/skipped/xfail,
  collecte, warnings utiles et raisons. Vérifier les 11 imports plats avant de
  compter ces tests comme témoins; ne pas réintroduire d'alias produit pour eux.
- [ ] **Préserver GUI → process → moteur** : valeurs et précédence des paramètres,
  pas seulement clés/signatures. Le wrapper renomme, parse, suffixe puis filtre
  silencieusement les kwargs via `inspect.signature` (`worker:36251` et suivantes).
  Exiger un témoin de propagation et un petit témoin process réel si cette zone bouge.
- [ ] **Préserver concurrence/lifecycle** : `spawn`, objets picklables et chemins
  d'import, queues/protocoles, callbacks, globals/closures, initialisation CUDA,
  annulation/arrêt, nettoyage threads/process/memmap/FITS. Pas de passage fork↔spawn
  ni changement d'ordre des opérations/imports sous couvert de déplacement.
- [ ] **Préserver les acquis 4.7.0** : WCS filtré transmis en mémoire, absence de
  second solve, choix write-WCS respecté, annulation sans déplacement de source,
  fermeture du filtre et arrêt worker sans fallback scientifique involontaire.
- [ ] **Préserver caches/reprise** : formats et signatures, paths, réutilisation de
  master tiles, checkpoints Phase 1/5, invalidation et run repris vs neuf; pas de
  migration incidente ni effacement des artefacts de référence.
- [ ] **Étendre la matrice distribution** : wheel installé hors checkout/CWD,
  ressources et locales, PyInstaller/hiddenimports/hooks, lancement Windows/macOS/Linux,
  CPU sans CuPy et intégrations absentes. Une plateforme non testée reste NOT_RUN.
- [ ] **Clarifier les STOP** : divergence CPU/GPU préexistante = geler la
  consolidation/parité concernée, documenter et continuer ailleurs; changement
  nouveau avant/après = bloquer le lot. Aucune correction scientifique incidente.
- [ ] **Borner R2** après R0 : liste ordonnée de lots et critères mesurables validés
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

## 5. R0 — archéologie complète (à exécuter ultérieurement)

- [ ] Créer `docs/refactor/ARCHAEOLOGY_R0.md`, ancré à BASE SHA, symboles et preuves.
- [ ] Dessiner le graphe application : entrée installée / module / wrapper / frozen
  → bootstrap Qt → GUI/filtre → wrapper process → dispatcher → Classic/SDS/Grid.
- [ ] Détailler précédence Grid/SDS, erreurs/fallbacks et payloads GUI/filter.
- [ ] Cartographier Classic, SDS, Grid et 4.5 si accessible : appelant →
  normalisation → pondération → rejet → combine → backend réel/fallback.
- [ ] Couvrir Filter Qt/Tk, SolverPort/ZeSolver/ASTAP, lancement ZeAnalyser,
  FITS I/O, WCS, reprojection, photométrie, assembly, checkpoint/resume,
  preview/progress, ressources/GPU, modules `core/robust_rejection` et `cuda_utils`.
- [ ] Inventorier globals, caches, imports dynamiques, side effects et dépendances
  inverses worker↔GUI avant d'identifier des modules extractibles.
- [ ] Classer chaque module ET chemin significatif : ACTIVE, COMPATIBILITY,
  TEST / DIAGNOSTIC, DORMANT BUT REACHABLE, SUSPECTED DEAD, PROVEN DEAD ou UNKNOWN.
  Inclure preuves, appelants, contrat supporté, témoins, risque et revue.
- [ ] Établir le statut supporté de Phase 4.5 : preuve de chemin et témoin ciblé;
  jamais « False dans Qt = mort ». Ne pas changer les defaults pour la tester.
- [ ] Séparer helpers métier/Tk/fallbacks dans le filtre, sans suppression immédiate.
- [ ] Établir une baseline rapide, isolée et honnête (cf. section 6).
- [ ] Ajouter avant extraction les petits témoins de comportement absents.
- [ ] Produire premiers tableaux R3, anomalies scientifiques, UNKNOWN/STOP et
  ordre proposé des extractions avec critères de sortie de chaque lot.
- [ ] Revue Nono indépendante de R0; toute classification PROVEN DEAD contestée
  redevient UNKNOWN ou SUSPECTED DEAD avant R1.

PROVEN DEAD exige, selon le composant : aucun caller supporté, dépendance d'import,
chemin Qt/CLI/package/API publique, import dynamique, dépendance packaging/resource,
contrat de test pertinent ou besoin de compatibilité. Recherche AST/texte seule
insuffisante. Aucun candidat n'est déclaré PROVEN DEAD dans cette préparation.

## 6. Inventaire initial des témoins — tous NOT_RUN ici

Commandes futures : utiliser l'interpréteur de l'environnement constaté et
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
| `tests/test_cupy_platform_guard.py`, `tests/test_version_gpu.py`, `tests/test_phase5_vram_budget.py`, `tests/test_resource_telemetry.py` | GPU optionnel, imports/absence NVRTC, budgets et télémétrie |
| `tests/SMOKE_PROTOCOL_Windows_macOS.md` et garde CI Qt | Compléments plateforme, ne pas annoncer PASS sans exécution |

- [ ] Compléter les trous : dispatch réel, paramètres aval, absence de Tk sur
  chemins officiels, cache/reprise, stacking petits tableaux et erreurs/fallbacks.
- [ ] Préserver les différences existantes avec des tests de caractérisation;
  séparer une parité attendue mais fausse des gates de non-régression. Si xfail
  nécessaire : ciblé/strict, témoin d'échec et entrée SCIENCE, jamais xfail global.
- [ ] Tests structurels déplacés : adapter leur localisation ou les remplacer par
  un témoin comportemental équivalent avec revue; ne pas masquer un vrai échec.
- [ ] Une validation globale consolidée à la fin d'un jalon significatif est utile;
  pas de full suite répétitive après chaque petit diff.

## 7. R1 — supprimer seulement le code prouvé mort

Pour CHAQUE unité, checklist à copier dans son rapport :

- [ ] Preuves R0 + revue Nono acceptées, contrats/packaging/compatibilité exclus.
- [ ] Témoins ciblés avant modification, résultats conservés.
- [ ] Petit diff mécanique; aucune refonte nécessaire.
- [ ] Témoins après, `git diff --check`, inspection des fichiers inattendus.
- [ ] Revue Nono du diff final, vérification indépendante Junior.
- [ ] Commit local distinct `refactor: remove proven-dead <specific thing>`.

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

Pour CHAQUE extraction :

- [ ] Identifier comportement exact, tous appelants/imports/monkeypatchs, globals,
  lifecycle et frontière proposée sans cycle ni dépendance lourde nouvelle.
- [ ] Définir critère d'acceptation : responsabilité nommée/testable localement,
  contrat identique, dépendances connues; pas simple éclatement arbitraire.
- [ ] Témoins avant; déplacer mécaniquement, conserver signatures/symboles/semantics
  lorsque nécessaires (shim/réexport seulement si justifié et testé).
- [ ] Préserver ordre des calculs, structures, copies/vues et effets de bord.
- [ ] Tests ciblés/import/process/package pertinents après, diff-check, revue Nono,
  contrôle indépendant Junior; commit distinct après acceptation du diff exact.
- [ ] Mettre à jour carte, témoins et ce TODO avant la tâche suivante.

Duplication : distinguer copie exacte, helper partagé, variante scientifique,
compatibilité et comportement de mode. Consolider uniquement les vrais doublons
mécaniques. Toute différence numérique/masques/fallbacks → STOP consolidation,
FOLLOW-UP SCIENCE. Pas de fusion Classic/SDS simplement parce qu'ils se ressemblent.

## 9. R3 — carte des contrats de stacking

- [ ] Créer `docs/refactor/STACKING_CONTRACTS_R3.md` (version préliminaire en R0).
- [ ] Lignes : Classic CPU/GPU, SDS CPU/GPU si réel, Grid CPU/GPU, Phase 4.5 si accessible.
- [ ] Pour chacun : median/mean, Kappa-Sigma, WSC, linear-fit NORMALISATION et
  linear-fit REJET distincts; documenter les modes non supportés, pas les inventer.
- [ ] Colonnes : entrée/caller, config/aliases/précédence, implémentation,
  normalisation, pondération scalaire/pixel, NaN/Inf, masque/rejet, dtype,
  axes/canaux, low-N/all-invalid, paramètres effectivement propagés,
  fallback/import/erreur/OOM, backend réel, tests et limites des preuves.
- [ ] Tracer Grid CPU, GPU core et GPU legacy fallback si core indisponible;
  inclure `core/robust_rejection.py` dans les chemins qui l'utilisent réellement.
- [ ] Ne déclarer une parité que mesurée; conserver échecs préexistants et
  limites de matériel sans modifier la science pour rendre le tableau vert.

## 10. TODO / FOLLOW-UP — SCIENCE et inconnues

| ID | Sujet | Statut initial / suite, hors corrections R0–R3 |
| --- | --- | --- |
| SCI-01 | Grid CPU WSC établi vs GPU core WSC simplifié | Divergence de code confirmée; construire témoin reproductible, quantifier, ne pas corriger |
| SCI-02 | Placeholder linear_fit dans core | Présent; Grid GPU passe none au core. Identifier tout caller effectif et distinguer normalisation/rejet |
| SCI-03 | Masques/poids Grid CPU/GPU, all-invalid, aliases et winsor_limits | À caractériser : CPU masque les poids non positifs avant rejet; configuration transmise au core différente. Pas de conclusion de parité ni de correctif ici |
| SCI-04 | Variantes Classic/SDS/Phase 4.5 / low-N / chunking | Différences à inventorier avant toute consolidation |
| TEST-01 | 11 imports plats importorskip Phase 3 | Risque de couverture absente confirmé dans le source; mesurer collecte/skips et réparer le témoin, pas le namespace produit |
| ARCH-01 | Invocations supportées de Phase 4.5 / fallback Tk | UNKNOWN jusqu'à preuve/revue; conserver |
| ARCH-02 | Contrats externes / frozen / API programmatique | Inventaire incomplet; ambiguïté bloque une suppression, pas toute R0 |
| MANUAL-01 | M106 avant/après et plateformes/GPU réels | NOT_RUN; nécessaires aux validations qu'ils prétendent établir |

Pour toute découverte ajouter : ID, SHA, chemin/caller, attendu vs observé,
commande/témoin/artefacts, impact, raison de non-correction, prochain pas unique.

## 11. Orchestration et discipline Git (lors du lancement seulement)

Junior dirige et accepte; Coco implémente des lots bornés, Nono révise indépendamment.
Pas de délégation déclenchée par la seule présence de ce fichier.

- [ ] Une mission_id stable par lot indépendant, rapports durables distincts par
  itération sous `/home/tristan/.openclaw/workspace/.a2a-reports/`.
- [ ] Coco neuf/reset seulement à la frontière d'une mission indépendante, après
  preuve d'inactivité; conserver son contexte pendant revue et REWORK (maximum 3).
  Un contexte Nono frais peut servir un lot indépendant, jamais effacer un actif.
- [ ] Transport direct `sessions_send(timeoutSeconds=0)`, callback vers l'exact
  `sourceSession` fourni par OpenClaw, rapport écrit avant callback, puis REPLY_SKIP;
  pas de polling, spawn natif ou attente synchrone Coco/Nono. Inclure le contrat
  intégral de callback des instructions workspace dans chaque délégation.
- [ ] Junior vérifie dépôt/diff/tests/artefacts indépendamment; Nono ne modifie pas
  le dépôt lors d'une revue. Une revue R0 n'exempte pas les futurs diffs de revue.
- [ ] Inspecter branche/HEAD/status avant et après chaque lot; préserver les
  changements tiers. Tout commit code/test : témoins, diff-check, revue Nono du
  diff exact avant commit; nouveau delta après revue = revue à compléter.
- [ ] Commits locaux petits/réversibles : docs, témoin, suppression ou extraction
  distincts. Pas de « cleanup everything », pas de réécriture d'historique.
- [ ] Gates humaines : changement de périmètre/architecture fondamentale/API
  supportée, retrait de fonctionnalité, migration destructive/dépendance majeure,
  publication. Aucun push/merge/tag/release autorisé par cette mission.

## 12. Comparaison M106, clôture et rapport final

- [ ] AVANT refactor scientifique/structurel : localiser le corpus historique
  « M106 de l'enfer », créer manifeste de fichiers/hash/config/ordre, capturer
  dépendances et backend réel; conserver référence BASE SHA, logs et sorties
  dans un espace distinct, sans écraser/modifier les brutes de référence.
- [ ] Préserver sorties intermédiaires pertinentes : nombre/identité des tuiles,
  FITS data/WCS/headers scientifiques, poids/alpha/coverage, NaN, rejets, stats et
  diagnostics des trous/seams. Identifier métadonnées volatiles séparément.
- [ ] Comparer candidat et base avec même corpus/config/matériel; un résultat
  simplement « joli » ne prouve pas la science. M106 ne remplace pas les témoins
  des modes/options qu'il n'exerce pas (SDS/Grid/GPU/Phase 4.5 notamment).
- [ ] Tristan réalise/valide l'acceptation manuelle finale; baseline indisponible
  ou test non passé = HOLD explicite, jamais promotion automatique.
- [ ] Écrire `docs/refactor/FINAL_REPORT.md` : BASE SHA, branche, HEAD final, statut
  worktree, synthèse R0/classifications, suppressions et preuves, extractions,
  contrats R3, anomalies intentionnellement NON corrigées, tests/commandes/durées/
  résultats/skips, revues Nono, commits, TODO restants et gates manuelles.
- [ ] Distinguer ACCEPT technique local, PARTIAL/BLOCKED et ACCEPT scientifique
  humain. Même après M106, une publication nécessite une autorisation distincte.

Succès = chemin officiel évident, code mort démontré, responsabilités moins
couplées/testables, science préservée et anomalies isolées. Pas « beaucoup de
fichiers » ni « moins de lignes à tout prix ».

> Archaeology first. Mechanical refactor second. Science unchanged. M106 last.
