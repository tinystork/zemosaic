# M33 recovery robustness — état consolidé au 2026-11-28

Mission `ZM-M33-DRAIN-DURABLE-20261127` — branche `fix/m33-drain-durable-results`,
baseline/HEAD `9cd04c93f9952d62b8dddd00f74b0c7f7d372109` (beta vérifié), working
tree propre au départ. **Aucun commit, aucun push, aucune publication : ce
document décrit un état de travail, pas une clôture.** Le run Windows M33 en
cours reste protégé : aucune lecture de nœud, aucun arrêt, aucune installation,
aucun accès dataset/store, aucune récupération des résultats en RAM.

## 1. Défaut observé (baseline, non réparé par ce lot)

`pmap` (et `pmap_reserved`) collectaient le premier échec worker via
`_raise_collect_failure` / `_consume_finished_siblings` : seuls les résultats
**déjà terminés** étaient consommés, les tâches PENDING annulées, puis
`shutdown(wait=True)` attendait les workers déjà démarrés **sans jamais
consommer leurs résultats tardifs**. Sur M33 (MemoryError fatal tuile 59, quatre
pairs continuant 62–74/81, GUI figée à 28 commits), les cellules démarrées qui
terminaient après le fatal étaient donc calculées puis JETÉES : ni commit
durable, ni trace exploitable.

Ce lot ne corrige PAS le MemoryError ni l'OOM : il répare le **cycle de vie de
l'échec** (drain). Aucun checkpoint de tuile n'est implémenté ici.

## 2. Contrat d'acceptation (défini AVANT les éditions)

1. Premier échec worker : annuler les tâches réellement PENDING, ne jamais
   re-soumettre ni rejouer, remonter l'état fatal immédiatement (`on_fatal`
   avant le drain), puis **continuer à recevoir et consommer durablement** les
   tâches déjà démarrées qui réussissent, exactement une fois chacune.
2. L'erreur ORIGINALE reste primaire et finit TOUJOURS par être levée
   (aucun faux PASS). Les échecs secondaires (worker ou consommateur) sont
   enregistrés explicitement dans `meta`, sans masquer l'original.
3. Propriétaire unique du shutdown (aucun second `shutdown(wait=True)`).
4. Heartbeat et progression restent émis pendant le drain ; les reçus sont
   véridiques (`COMPUTED` ≠ `COMMITTED`, comptage par réception).
5. Un échec de commit du consommateur durable n'est ni avalé ni converti en
   succès. Politique explicite et fail-closed : après un échec consommateur, le
   drain NE consomme PLUS les résultats suivants (le store durable est dégradé
   et l'intake suivant refuserait de toute façon : `StoreCommitNotDurable` /
   `_R55DurableStoreUnavailable`). Ces résultats reçus sont identifiés comme
   **non commités** dans `meta["uncommitted_outputs"]`.
6. Les commits déjà acquis restent intacts (aucune réécriture, aucun rollback).
7. Périmètre : `core/zegrid/parallel.py` (`pmap`, `pmap_reserved`,
   télémétrie minimale). Aucun changement de maths/science/jauge/ordre des
   frames/chunks/identité. Aucune dépendance, aucun commit, aucun push.

## 3. Checklist

- [x] État de base inspecté : branche `fix/m33-drain-durable-results`,
      HEAD `9cd04c9`, worktree propre.
- [x] Contrat d'acceptation écrit ci-dessus AVANT édition.
- [x] Régression rouge déterministe écrite AVANT le correctif
      (`tests/test_zegrid_m33_drain_durable.py`) :
      - thread réel : worker fautif + pair bloqué qui ne termine qu'APRÈS
        `on_fatal`, pair consommé + reçu durable (vrai `RunCellStore`) avant
        que l'erreur originale ne s'échappe ; exactement une fois ; consommateur
        dans le thread appelant ;
      - même témoin en `multiprocessing` **spawn** (style Windows, petit
        synthétique, portes fichiers) ;
      - `pmap_reserved` : tâches PENDING jamais exécutées, paire démarrée
        drainée/consommée (thread + spawn) ;
      - échec worker secondaire, échec consommateur primaire (arrêt explicite
        des commits suivants), échec consommateur secondaire pendant le drain,
        pool cassé (`BrokenProcessPool`), succès inchangé, aucun rejeu sérial,
        heartbeat/progression pendant le drain, métadonnées de reçus.
- [x] Correctif implémenté (lot 1 uniquement) dans
      `src/zemosaic/core/zegrid/parallel.py` :
      - `pmap` : `_raise_collect_failure` surface `_fatal` AVANT tout attente,
        annule uniquement les futurs qui n'ont jamais démarré (`cancel()`
        vrai), puis `_drain_started()` reçoit/consomme chaque tâche déjà
        démarrée via `_collect` (marquage `recorded` = exactement une fois),
        avec cadence heartbeat et progression ; l'erreur originale est levée
        ensuite (type préservé, `ResultConsumeError` pour le consommateur).
      - `pmap_reserved` : `_drain_cells_after_failure` (fatal d'abord, annulation
        des futurs non démarrés, drain consommé des cellules démarrées, tâches
        jamais dispatchées nommées `pending_never_dispatched`) ; shutdown unique
        conservé dans le `finally`.
      - Politique explicite fail-closed : après un échec consommateur
        (`StoreCommitNotDurable`/`_R55DurableStoreUnavailable`),
        `consume_policy = "stopped: …"` et plus aucun commit n'est tenté ; les
        résultats reçus sont listés dans `uncommitted_outputs` (jamais comptés
        comme commités). Les échecs secondaires restent dans
        `sibling_worker_failures` / `consume_failures`. Télémétrie additive :
        `drain` (started/seconds/primary/cancelled/consumed).
- [x] Validation ciblée exécutée :
      - rouge AVANT correctif : `pytest tests/test_zegrid_m33_drain_durable.py`
        → **9 failed, 2 passed in 71.03s** (le pair démarré n'était jamais
        consommé ; `meta["drain"]` inexistant).
      - vert APRÈS correctif : même fichier → **11 passed in 12.92s** ;
        `tests/test_zegrid_live_commit_l1.py` + `tests/test_zegrid_live_obs_l2.py`
        + nouveau fichier → **72 passed in 104.02s**.
      - suites pmap/télémétrie existantes (`r12`, `r13`, `r19`, `r40b2`,
        `per_cell_scheduler`, `io_reject01`, `r14`) → **120 passed,
        4 skipped, 3 failed** ; les 3 échecs (`r40b2` E1/F, `r14`
        `test_pmap_progress_seam_preserves_results`) échouent À L'IDENTIQUE sur
        l'arbre propre au HEAD `9cd04c9` (vérifié par `git archive`) :
        préexistants, aucune régression.
      - batch `m33_p1_wave_dispatch` + `resume_m1_conservation` +
        `dispatch_propagation_witness` + `fd_exhaustion` + `commit_integrity`
        + nouveau fichier → **50 passed, 13 skipped in 63.92s**.
      - `r55e1_run_checkpoints` → **36 passed, 6 failed** ; les mêmes 6 échecs
        sont reproduits sur l'arbre HEAD propre → préexistants.
      - Détail complet et commandes dans
        `.a2a-reports/ZM-M33-DRAIN-DURABLE-20261127.coco.r0.md`.

## 3bis. Rework-1 (2026-11-27) — F1/F2 de la revue Nono r0

Deux trous de chemins d'échec secondaires, reproduits indépendamment par Nono
(`/tmp/nono_drain_repro_a.py`, `/tmp/nono_drain_repro_b.py`), ont été corrigés
sans élargir le périmètre :

- **F1 — échec partiel de `pool.submit`.** Chemin `pmap` (boucle de soumission)
  et les DEUX sites de `pool.submit` de `pmap_reserved` : l'échec passe
  désormais par le même cycle `_fatal → cancel des non-démarrés → drain +
  consommation` avant de lever `ParallelPoolError` (cause originale et index
  conservés, aucun replay, aucune soumission après l'échec). Bookkeeping
  véridique : `submit_failed_index`/`never_submitted`/`submit_failure` pour
  `pmap`, `submit_failures` + `pending_never_dispatched` pour `pmap_reserved`
  (et `dispatches` ne compte que les soumissions réussies, `_dispatch` est
  appelé APRÈS un `submit` réussi).
- **F2 — échec de consommateur dans le MÊME lot terminé.** La politique est
  fermée immédiatement dans le `except` du consommateur (`pmap`) et dans la
  boucle de collecte de `pmap_reserved` : plus aucun sibling du même lot n'est
  consommé après le premier échec de commit, il est nommé dans
  `uncommitted_outputs`. Le compteur de progression ne compte que les reçus
  réellement délivrés.
- `_consume_finished_siblings` reste (tests directs existants) mais est
  explicitement marqué « test-only, plus sur le chemin de production » (N1).

Nouveaux tests déterministes (sans sleep-course) : soumission partielle
`pmap`/`pmap_reserved` (pool factice qui réussit `submit` #1 puis échoue ; la
tâche 0 est drainée/consommée dans le thread appelant) et échec consommateur
dans un lot `ALL_COMPLETED` forcé (`pmap` + `pmap_reserved`, un seul appel
consommateur, sibling nommé uncommitted).

Preuves rework-1 : repros Nono rejouées → F1 `consumed=[0]`, `drain.started=True`,
`submit_failed_index=1` ; F2 `calls=[0]`, `uncommitted=[{index:1,…}]`. Suite
nouvelle → **15 passed**. `live_commit_l1 + live_obs_l2 + nouveau` → **76 passed**.
Batch ciblé r12/r13/per_cell/io_reject/m33_p1 → **85 passed, 9 skipped** ;
r19/fd_exhaustion/commit_integrity → **27 passed**. Aucun commit/push.

## 4. Lot 2 — checkpoints de tuile (état 2026-11-27)

- [x] **`durable_tile_checkpoints` : IMPLÉMENTÉ (lot 2 séparé).** Voir
      `docs/refactor/M33_TILE_CHECKPOINTS_DESIGN_20261127.md`. Le nouvel exécuteur
      externe écrit, pour chaque tuile réduite, un checkpoint typé atomique
      (payload réduit + marqueur) avant de jeter le spool de la tuile ; une
      nouvelle tentative réouvre la Cell et réutilise les tuiles déjà validées
      (identité source/science/jauge/backend/géométrie identique, sinon refus
      explicite). Le nettoyage n'a lieu qu'après le commit durable final de la
      Cell ; sans commit durable, les checkpoints réduits bornés sont conservés
      (nettoyage manuel documenté).
- [ ] Le MemoryError/OOM M33 (tuile 59) n'est ni expliqué ni corrigé ici ; les
      checkpoints ne rendent pas les tuiles 74/81 de l'ancien run récupérables
      (ce run n'a écrit aucun checkpoint).
- [ ] Aucun scheduler mémoire global, aucun GPU, aucun rework du live-running 0.

Tests lot 2 : `tests/test_zegrid_m33_tile_checkpoints.py` (23 passed) plus
suites ciblées externes/préparation/sink/resume (voir le rapport de mission
`.a2a-reports/ZM-M33-TILE-CHECKPOINTS-20261127.coco.r2.md`).

Rework-1 (revue Nono r0) : l'identité lie maintenant le contenu source + WCS
(empreinte gelée transmise par l'appelant produit), le batch résolu et un
run-id non-nullable strict ; le lock est ouvert après toutes les allocations et
le constructeur de spool, puis possédé par un `try/finally` inconditionnel
(jusqu'au finaliseur) ; le lot est CPU-only ; un témoin multiprocessus spawn
démontre la reprise par un nouveau PID ; la sémantique `complete` est renommée
`runtime_complete` (non durable) avec télémétrie `saved_tiles`/`write_failures`.

Rework-2 (durcissement final) : `Store.open()` et le dictionnaire/logger qui
suit sont désormais DANS le `try/finally` propriétaire du lock ; un échec de
`checkpoint_out.update` juste après l'ouverture ne fuit plus. Le helper WCS
refuse une preuve vide (WCS/transform ou `shape_hw` absents désactivent avec
raison). Le cache de binding est lié à l'identité des objets gelés (immuables).
Le schéma `source_binding` est l'époque de contrat explicite ; aucun hash de
version décodeur/politique n'est revendiqué aujourd'hui. Un test réel
builder → task-info et un run externe forcé sans digest injecté prouvent la
propagation et la persistance du binding source.

## Acceptation Junior lot 1

Lot1 ACCEPT après revue Nono r1 et reproductions indépendantes Junior. `never_submitted` de pmap inclut l’index dont submit échoue ; `pending_never_dispatched` du dispatcher réservé l’exclut (il figure dans `submit_failures`). État local sans commit/push ; OOM et checkpoints internes non corrigés par lot1. Lot2 checkpoints par tuile autorisé ensuite, avant intégration GPU externe et publication beta.

## Acceptation Junior lot 2

Checkpoints CPU acceptés après revue Nono r1, fermeture inconditionnelle du verrou et témoin réel builder→manifest (r2), puis correction de pollution des tests (r3). Junior : 26 tests combinés passent (84,01 s, 1 avertissement FITS). Aucun test Windows ni récupération des anciennes tuiles M33 revendiqués. État local sans commit/push. Prochain lot autorisé : intégration GPU externe, puis publication beta après validation.

## 5. Lot 3 — intégration GPU externe (état 2026-11-27)

Détail de conception : `docs/refactor/M33_EXTERNAL_GPU_INTEGRATION_DESIGN_20261127.md`.

- [x] **Admission GPU externe typée : IMPLÉMENTÉE.** Nouveau module
      `core/zegrid/gpu_external.py` (pur, sans CuPy). Un bloc est
      `N x positions` et contient TOUJOURS les N contributeurs de chaque pixel ;
      l'inversion entière est `positions * (N*96 + 512)`, plafonnée au batch
      hôte déjà admis et au relief intérieur. Budget = 90 % du VRAM libre
      courant − réserve fixe déclarée de 16 MiB. Le coefficient 96 B/cellule
      est une borne empirique supérieure auditée (pas un pic prouvé).
- [x] **Refus honnêtes avant admission gelée.** GPU indisponible, VRAM
      inconnue, sous le plancher, bloc d'un pixel impossible, K>1, scheduler
      non-ACTIVE, backend CPU gelé ou absence de demande → enregistrement typé
      `admitted=false` + raison, puis chemin CPU externe inchangé.
- [x] **Produit : request → planner/bundle → executor.** Le contrat externe
      transporte `backend` et le bloc `gpu` (probe/modèle/batch/raison).
      `_run_cell_external` impose le backend du contrat gelé au request
      (jamais un global périmé). Les tâches externes portent le même fait.
- [x] **Identité et reprise.** Un run GPU externe neuf gèle `backend=gpu` avec
      `working_set=external_spool` (tile + batch). Un run CPU gelé refuse
      l'injection GPU (`frozen_backend_cpu`) et garde l'époque CPU acceptée.
      Un run GPU gelé qui ne peut plus admettre le GPU échoue avec le refus gelé
      typé, jamais un CPU silencieux.
- [x] **Checkpoints et provenance par tuile.** Les marqueurs GPU portent
      `backend {requested, effective}` ; les marqueurs CPU restent inchangés
      (pas de clé `backend`, identité CPU non invalidée). Une tuile réutilisée
      effective CPU restaure l'état sticky CPU ; une tuile GPU après sticky CPU
      est refusée (non-monotone). Compteurs/sink/géométrie restent exacts une
      fois. `checkpoint_out`/`gpu_fallback_out` sont publiés en cours de Cell
      pour rester véridiques après un crash.
- [x] **Journal/manifeste/UI.** `gpu.external` publie
      requested/selected/effective/raison/batch/probe/modèle/réserve ; le
      record par Cell publie le backend EFFECTIF (un fallback n'est jamais
      compté GPU) ; `backend_used` et la conservation suivent cette vérité.
- [x] **Tests bornés.** `tests/test_zegrid_m33_external_gpu_integration.py`
      (6 passed) : modèle pur, refus, builder réel, dispatch executor GPU via
      mock CPU EXPLICITE, OOM classifié + crash + nouvelle tentative sticky CPU,
      erreur de programmation propagée, marqueur CPU inchangé. Suites lot1+lot2+lot3 :
      45 passed. Batch externe élargi (fallback/parité GPU, lifetimes, phase log,
      batch truth, phase budget, model measure, M4 durabilité, source plumbing,
      M33 P1 wave dispatch) : 238 passed, 1 skipped.
- [x] **Matériel réel minuscule (optionnel autorisé).** MX150 2 GiB, CuPy
      14.1.1, synthétique 11×17×19, tuile 8, batch 64 : tentative A GPU réelle
      9/9 tuiles GPU ; tentative B sur le nouveau chemin de checkpoints 9/9
      réutilisées, 0 réduction GPU ; 7 plans bit-exacts vs référence CPU.
      VRAM échantillonnée seulement, aucune qualification vitesse/mémoire.
      Artefact : `.a2a-reports/ZM-M33-EXTERNAL-GPU-INTEGRATION-20261127.coco.r0.artifacts/`.
- [ ] Aucune optimisation de prefetch de blocs, aucun changement de
      reprojection/CFA/maths, aucune formule whole-tile appliquée aux blocs
      externes, aucun nouveau réglage GUI. Aucun commit/push/installation.
- Échecs de baseline suivis (non causés par ce lot) : M6/resume envelope et
      anciens tests GPU résident host-RAM de
      `test_zegrid_active_external_wiring.py` (dont le seam legacy remplacé par
      l'admission externe lot3) ; le test `auto_external` a été mis à jour vers
      le nouveau seam et passe. Le test e2e externe passe seul et dans les
      sélections ciblées ; la suite complète garde les échecs baseline connus.

### Rework-1 (revue Nono r0)

- [x] **G1** : une VRAM LIBRE inconnue décline (`vram_unknown`) même si le TOTAL
      est connu ; le total n'est plus un substitut silencieux.
- [x] **G2** : sonde malformée (bool/float/str/négatif/`free>total`/`available`
      non booléen) => `probe_invalid`, sans exception ; `numpy.int64` accepté et
      normalisé en int JSON. Les paramètres d'appel invalides restent des erreurs
      de programmation.
- [x] **G3** : audit d'allocation corrigé (`degenerate_seen` est un vecteur
      `(positions,)`; `orig2` n'est pas libéré avant le combine) ; le coefficient
      96 B est requalifié en POLITIQUE conservatrice empiriquement non qualifiée,
      pas en borne prouvée ; la réserve 16 MiB n'est pas une garantie. Nouvelle
      revalidation point-of-use avant la PREMIÈRE réduction GPU réelle de chaque
      Cell (jamais pour une Cell qui ne fait que réutiliser des checkpoints) :
      free actuel + bloc gelé + réserve, sinon repli CPU collant sans tentative
      d'allocation GPU, identité/batch gelés inchangés. Les tuiles GPU
      historiques restent réutilisables ; les nouvelles tuiles sont CPU.
- [x] **G4** : le K réel du plan externe est lu (`plan["cells_in_flight"]`) et
      passé à l'admission ; K=2 est refusé par le helper produit, et le plan
      externe reste `gpu_serialized` K=1 (aucun chemin GPU parallèle).
- [x] **G5** : témoin hardware r1 en comparaison EXACTE (dtype/shape, positions
      NaN, octets bruts des éléments non-NaN, compteurs exacts), étiqueté parité
      directe d'exécuteur (pas produit complet). Ajout d'un test produit réel
      builder → identité → RunCellStore fermé/réouvert (CPU only) prouvant la
      compatibilité de l'identité GPU externe gelée.
- Tests : `tests/test_zegrid_m33_external_gpu_integration.py` 11 passed ;
      lot1+lot2+lot3 51 passed ; régressions GPU/externes ciblées 78 passed.
      Artefact r1 :
      `.a2a-reports/ZM-M33-EXTERNAL-GPU-INTEGRATION-20261127.coco.r1.artifacts/`.
      Les revendications baseline r0 restent inchangées (provenance r0).

### Rework-2 (correction runtime bornée)

- [x] **M1 — moment du point-of-use.** La revalidation n'est plus avant l'écriture
      du spool CPU : elle est DANS le `try` de réduction, APRÈS l'écriture /
      jonction producteurs et IMMÉDIATEMENT avant la première branche de
      réduction GPU réelle. Latch unique par Cell, probe callable frais, aucun
      changement de batch/identité. En cas de refus, la tuile déjà écrite est
      réduite sur CPU et les tuiles NON calculées restent CPU ; aucune allocation
      GPU. Un test à hook d'écriture réel prouve `probe_after_write=True`,
      `gpu_reduces=0`, provenance CPU des checkpoints, parité 7 plans et
      nettoyage. Un chemin 100 % réutilisation fait 0 probe et 0 réduction.
- [x] **Coin de reprise mixte.** Un ensemble complet de checkpoints GPU, tuile 1
      corrompue/supprimée, nouvelle tentative à VRAM basse : la tuile 1 est
      recalculée CPU, les tuiles GPU historiques 2..N sont restaurées (provenance
      par tuile distincte de l'état sticky courant), les NOUVELLES tuiles restent
      CPU, l'identité demandée `gpu` est inchangée, les compteurs mixtes et le
      backend effectif final CPU sont véridiques. Une troisième tentative
      restaure le mélange sans probe/réduction GPU. Les refus
      requested-mismatch/marqueur invalide restent.
- Tests : `tests/test_zegrid_m33_external_gpu_integration.py` 14 passed ;
      lot1+lot2+lot3 54 passed ; régressions GPU/externes ciblées 78 passed
      (relance propre). Aucune campagne baseline 319 s ni suite complète.
      Artefacts r0/r1 inchangés ; rapport r2 :
      `.a2a-reports/ZM-M33-EXTERNAL-GPU-INTEGRATION-20261127.coco.r2.md`.

## 6. Validation finale (2026-11-27/28) — statut local, pas acceptation

- Fast tier demandé (`pytest -m "not slow" -q --maxfail=12`, timeout 1200 s) :
  **TIMEOUT (exit 124)** après ~21 % (5813 tests collectés). 5 échecs observés,
  tous préexistants : `tests/test_locales_parity.py::
  test_locale_has_exactly_the_reference_keys[de|es|is|nl|pl]`
  (clés `qt_field_zegrid_gpu_percell_*` absentes des catalogues), reproduits à
  l'identique sur l'archive `git archive HEAD` dans `/tmp/m33base`. Aucun nouvel
  échec observé. Le junit fast n'a pas été produit (process tué par timeout).
  Log : `.a2a-reports/ZM-M33-EXTERNAL-GPU-INTEGRATION-20261127.final-fast.log`.
- Gate intégré combiné (lots 1/2/3 + external/m2/m4/prepare/rejection/batch
  truth/GPU a3/commit/live/e2e) : **371 passed** en 270,91 s, 0 failed.
  Log + junit : `...final-combined.log`, `...final-combined.junit.xml`.
- Gates d'import headless officiels (`PYTHONPATH=src`, workflow
  `no-tk-on-official-path`) : TK guard PASS, imports config/worker PASS,
  `--tk-gui` backend Qt PASS.
- Manifeste SHA-256 des fichiers modifiés/nouveaux :
  `...coco.validation.files.sha256`. HEAD inchangé
  `9cd04c93f9952d62b8dddd00f74b0c7f7d372109`; aucun commit/push.
- **Statut de validation : BLOCKED pour la complétude du fast tier** (timeout,
  preuve manquante). Le gate intégré et les gates d'import sont verts. Windows
  natif et run produit réel complet restent NOTRUN. Junior garde l'autorité
  d'acceptation/publication.

## 7. Validation2 — couverture fast chunkée (statut local)

- Inventaire fast exact : 5813 node IDs collectés (`...final-fast.collected.txt`).
- Couverture : 368 IDs déjà PASS dans le JUnit combiné (mapping exact ; 3 tests
  slow-only exclus), 5 locales baseline connues, 5440 restants exécutés en 12
  chunks séquentiels (logs `-v` + JUnit par chunk) :
  **5075 passed, 292 failed, 60 errors, 13 skipped, 0 unfinished**.
  Union : 5813/5813 accounted. Ce n'est PAS un run single-process du gate fast.
- Baseline ciblée des 352 échecs/erreurs sur l'archive HEAD `/tmp/m33base` :
  timeout à 20 % (89/352). 84 échouent aussi au HEAD (baseline), 5 passent au
  HEAD. Deux nouveaux échecs confirmés et isolés :
  `test_zegrid_gauge_import.py::test_rework1_import_hashes_each_frame_exactly_once`
  (8 hashs pour 4 frames, double passe) et
  `test_zegrid_m6_cli.py::test_overwrite_is_explicit_and_recorded`
  (`.zegrid-tile-checkpoints` laissé dans l'output, refusé comme contenu
  étranger). `m6_stop` échoue aussi au HEAD ; `m6_gui_policy` et
  `r23_science_dbe` passent seuls sur la branche (sensibles à l'ordre).
  263 nœuds restent non classés (timeout baseline), listés dans
  `...final-fast.coverage_final.json`.
- Aucun code/test modifié pendant la validation. HEAD inchangé
  `9cd04c93f9952d62b8dddd00f74b0c7f7d372109`; aucun commit/push.
- **Statut : BLOCKED** — nouveaux échecs non expliqués et classification
  baseline incomplète. Windows natif et run produit réel complet restent NOTRUN.
  Junior garde l'autorité d'acceptation/publication.


## 8. Rework-3 — régressions d'intégration corrigées (statut local)

- **Hash frames une seule fois** : `_run_single` calcule une fois les lignes
  gelées (`_r55_frame_input_rows`) et les partage entre vérification d'import,
  identité durable et binding des checkpoints. `frame_rows` est validé
  (longueur/champs) ; liste courte/malformée => binding désactivé avec raison,
  jamais un binding affaibli. Cache limité à un run ; aucun cache inter-run.
- **Résidu de checkpoints et publication** : la racine possédée est supprimée
  quand elle devient vide après le commit final. Si un résidu doit rester
  (run incomplet), `detect_previous_publication` ne le reconnaît que via une
  preuve d'appartenance stricte (`inspect_owned_checkpoint_root`) : répertoire
  simple, manifestes typés valides, noms tile/tmp du produit, aucun owner vivant,
  aucun lien/entrée étrangère. Un contenu forgé/étranger/malformé ou un lien
  reste protégé. Pas d'ignore par nom.
- Tests : les deux régressions nommées passent (2 passed), module gauge-import
  31 passed, module CLI 19 passed + 1 échec baseline confirmé au HEAD,
  checkpoints/GPU/drain 57 passed, publication + régressions 70 passed.
  Aucun commit/push ; HEAD inchangé.

## 9. Rework-4 — hardening inspecteur d'ownership

- Marqueur de tuile JSON non-objet (`[]`, chaîne, nombre, `null`) => foreign
  propre, jamais d'exception, via inspecteur ET `detect_previous_publication`.
- `tiles`/`tmp` testés `is_symlink()` AVANT `exists()` : lien réel ou cassé
  refusé sans traversée/mutation.
- `LOCK`/`OWNER.json` présents doivent être des fichiers simples (pas de lien
  cassé/réel, pas de répertoire). `OWNER.json` malformé présent => ownership non
  prouvable (foreign) ; owner ABSENT = résidu de crash légitime, toujours owned
  si le reste est valide. Owners vivants protégés.
- Limite documentée : ownership STRUCTUREL seulement, pas validation de
  réutilisabilité scientifique ni provenance cryptographique.

## 10. Validation3 — classification des 352 échecs fast (statut local)

- Inventaire fast 5813/5813 accounted (368 PASS combiné exact + 5 locales
  baseline connues + 5440 chunkés : 5075 PASS, 292 FAILED, 60 ERROR, 13 SKIP,
  0 unfinished).
- 352 nœuds en échec sur la branche, classés avec preuve HEAD :
  - **344 baseline** (mêmes catégories/erreurs au HEAD ; premier groupe 85,
    reste 259) — causes dominantes dans les 230 JUnit disponibles :
    AssertionError 70, RuntimeError 56, IdentityError 38, InvalidEnvelope 23,
    NameError 16, TypeError 7, StopIteration 7, etc.
  - **2 régressions nouvelles corrigées en r3** : hash frame unique et overwrite
    CLI `.zegrid-tile-checkpoints` ; tests existants verts (r3).
  - **6 baseline-PASS au HEAD et verts isolément sur la branche** :
    order-sensitive/transient (dont root_deliverables, flaky une fois puis vert,
    et routine e2e 28,30 s).
  - **0 nouveau échec inexpliqué / non classé**.
- Arbre baseline vérifié : `git archive HEAD` frais `/tmp/m33head3` identique à
  `/tmp/m33base` (seuls `__pycache__` diffèrent), imports pointés sur cet arbre,
  jamais sur `src` de la branche.
- Témoin hash réel différé : RAM disponible ~806 MiB, insuffisante pour le gate
  réel ; dernier run adéquat r3 = 2 passed. Les 58PASS/1ERROR Junior étaient
  `R55_REFUSE_HOST_RAM` avant l'assertion de hash, non verts.
- Aucun code/test modifié ; HEAD inchangé. Statut validation : classification
  complète, aucun nouvel échec inexpliqué ; témoin hash réel reste différé pour
  ressources. Windows/corpus/GPU réel non exécutés.

## 11. Validation4 — correction d'évidence (statut local)

- Nouveau ledger orthogonal `...classification_evidence4.json` (352 nœuds) :
  - **176 trace_same** (dont 171 avec traces HEAD+branche et 5 status-only
    recheckés isolément : même signature InvalidEnvelope exacte) ;
  - **15 trace_different_environment** (R55 RAM/InvalidEnvelope, valeurs RSS
    mesurées, refus de ressource) — aucune différence sémantique ;
  - **153 status_only** (109 sans trace baseline, 44 sans trace branche) ;
  - **2 resolved_fixed_r3** (hash unique, overwrite checkpoints) ;
  - **4 resolved_environment_flake** (dont root_deliverables, R55 RAM) ;
  - **2 resolved_uncaused_pass_alone** (routine e2e, m6 gui policy) ;
  - **0 trace_different_semantics / 0 non classé**.
- Traces disponibles : branche 305/352, baseline 230/352 ; 186 ont les deux
  (171 same +15 env-diff). Les 122 sans trace baseline et 44 sans trace branche
  sont listés dans
  `...classification_evidence4.missing_trace_ids.txt`.
- `root_deliverables` reclassé flake d'environnement (R55 RAM), pas ordre.
- Le compteur « 344 même échec » est remplacé par la carte ci-dessus : la
  parité de statut n'est pas une preuve de même cause.
- Witness hash produit réel : **DEFERRED** (MemAvailable ~672 MiB < 1,5 GiB) ;
  dernier run adéquat r3 = 2 passed, non rejoué.
- Aucun code/test modifié ; HEAD inchangé. Ni Windows/corpus/GPU réel.

## Junior gate historique — blocage RAM désormais levé

État historique, remplacé par la décision ci-dessous. Implementation and Nono review4 accepted. Beta publication remained BLOCKED_RESOURCE: unmodified real-product single-hash witness must be rerun with adequate MemAvailable (conservative 1.5 GiB; last 613 MiB). Corrected failure evidence is 176 same signatures / 15 environment-different / 153 status-only / 8 resolved; not a green full suite or 344 proven same semantic defects. Detailed gate and exact next step: `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-M33-EXTERNAL-GPU-INTEGRATION-20261127.gate-state.md`. No commit/push/Windows/M33 modification; existing scoped beta authorization persists.

## Décision Junior — intégration beta acceptée, 2026-11-28

Tristan a libéré la RAM de tinydebian : MemAvailable 1843 MiB avant le test.
Le test réel **inchangé** `test_rework1_import_hashes_each_frame_exactly_once`
a été exécuté indépendamment par Junior : **1 passed, 29 warnings in 9.85s**
(warnings fork Python 3.13 et commentaires FITS tronqués). Le blocage ressource
est levé. Les deux régressions d’intégration (double hash et overwrite CLI)
et les défauts d’ownership sont corrigés et revus ; Nono r4 ACCEPT, module
checkpoints Junior 46 PASS.

Acceptation de l’intégration des trois lots (drain durable, reprise par tuile,
voie GPU externe) et publication beta dans le périmètre autorisé par Tristan.
La suite fast complète n’est **pas** déclarée verte : la carte d’évidence §11
reste la limite de preuve pour les échecs préexistants/statut-seul/ressources.
Aucune modification de l’installation Windows, du run M33 ou des datasets.
La preuve matérielle GPU existante porte sur l’exécuteur direct MX150 ; aucun
benchmark produit complet Windows/RTX ni gain de vitesse garanti. Les nouveaux
checkpoints ne rendent pas récupérable rétroactivement le travail d’un ancien
run qui ne les écrivait pas. Le défaut OOM initial n’est pas annoncé corrigé.

Les sections précédentes constituent l’historique de validation ; cette décision
remplace les états BLOCKED_RESOURCE/DEFERRED du témoin de hash.
