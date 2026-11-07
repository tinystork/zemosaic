# Audit final Junior — ZeGrid Adaptive Resources A0–A6

Date : 2026-11-07. Mission : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107`.
Phase : audit transversal post-publication, revue Nono indépendante + vérification Junior.
**Audit terminé : FINDINGS. Qualification produit : BLOCKED / INCOMPLETE.**
Ce verdict remplace les acceptations trop larges d'A4/A5 et les PASS produit Q40/Q41/Q50/Q60.
Il ne nie pas les parités et les primitives réellement prouvées.

## Périmètre et état

- Dépôt : `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources`.
- Branche : `feat/zegrid-adaptive-resources`.
- HEAD audité : `418ba4395d96b62997673bcbd85559964aeee1f9`; origin/beta local identique.
- Baseline réelle : `515004078cfb64b327fb02aeaa323e48cca62bd1`.
- Diff : 28 commits, 58 fichiers, +7481/-237.
- Revue : `.a2a-reports/ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107.nono.audit.r0.md`.
- Aucun changement produit/config/runtime, commit, push, rollback ou modification des corpus.
- Seuls ce rapport et les annotations de statut STATE/matrice sont ajoutés localement au dépôt par l'audit.

## Verdict exécutif

Le moteur est bien appelé depuis la GUI, mais plusieurs mécanismes testés isolément ne sont pas
intégrés au parcours utilisateur. Les tests verts ne satisfont donc pas les gates du plan.
Différer le gros M33 jusqu'aux corrections d'intégration et leurs preuves ciblées.
Aucun rollback général recommandé : conserver les primitives valides, corriger le câblage et les modèles.
L'affirmation précédente de Junior « tout est câblé automatiquement, seul Windows reste » était trop forte.

## Vérification Windows M106 lancée depuis le parcours utilisateur

Run inspecté : `C:\\Users\\JarvisJunior\\Desktop\\zemosaic out\\M106\\mem_fix`.
Il est terminé avec succès (66/66 frames, 2 cellules complètes, zéro rejet, sorties immuables
et copies racine vérifiées). Le parcours sélectionne bien `processing_mode=zegrid`,
`zegrid_scheduler_mode=active`, GPU CuPy sur RTX 3070 et budget VRAM à 70 %. Le layout
automatique a rejeté 3x2 à cause du halo 256 puis admis 2x1 : comportement attendu.

Ce run ne prouve toutefois pas les chemins difficiles de la mission : son profil effectif est
`dense-streaming-v1` / `ram_tile`, sans external streaming, micro-bloc sous-ligne, scratch canevas,
OOM GPU ni croissance A5. Le mode de jauge `global` utilisé prend par ailleurs le chemin d'assemblage
qui ignore encore l'allocateur scratch (F6). Il constitue donc un bon témoin GUI/ACTIVE/GPU et un
témoin de publication, pas une fermeture de F1–F10.

Les avertissements « product settings ignored » et « accepted-but-ignored run arguments » décrivent
pour l'essentiel des options volontairement hors contrat ZeGrid : ils n'altèrent pas ce résultat.
Leur niveau WARN est néanmoins trop bruyant lorsque seules des valeurs par défaut sont concernées
(F14). Les trois anomalies ci-dessous sont, elles, des défauts confirmés à inclure au rework.

## Défauts matériels retenus

### F1 — HIGH — Copies plein canevas dans la sortie produit (A4)

`src/zemosaic/zemosaic_zegrid_mode.py:19573`, `_write_outputs`, appelé par `_run_single:18764`.
Même science-only et science déjà écrite, deux copies CHW RGB float32 et un cast coverage sont
créés inconditionnellement : **>=28 B/pixel**, hors sanitization, hash et second cast coverage.
Pour 75 698 820 pixels : 2 119 566 960 B, soit environ **1,97 Gio** supplémentaires minimum.
Probe memmap 8x9x3 : allocations indépendantes 288 + 864 + 864 B, aucune ne partage les sources.
La branche banded coverage est ATTEINTE, mais reçoit une copie complète float32 préalablement créée.
La sanitization et le hash doivent également être traités en bandes.
Classification : dette préexistante non fermée par A4; violation du contrat livré, pas nouvelle formule scientifique.

### F2 — HIGH — Cache supprimé encore facturé par l'admission (A2)

`zemosaic_zegrid_mode.py:5854`, `_select_external_fallback` utilise `_r55_cache_unit_bytes` puis
ajoute le N×patch à l'enveloppe, alors que les tâches bounded sont source-backed et n'écrivent plus ce cache.
Probe N20k, patch256² RGB, tile8, halo8, 1Gio disque simulé : spool184320000 B,
cache imaginaire17044480000 B, enveloppe17242144792 B => refus injustifié.
Le helper intervient aussi dans les faits ACTIVE/sonde jointe; auditer les domaines RAM/staging/disque
avant de mettre arbitrairement tous les termes à zéro. Les transitoires décodage/reprojection restent réels.
Classification : intégration incomplète introduite par changement de provider A2.

### F3 — HIGH — Sous-ligne non sélectionnable automatiquement (A3)

`core/zegrid/external_stream.py:1096`, `largest_admissible_column_batch` conserve un minimum
`interior_width*channels`, contrairement au réducteur 2D désormais capable de blocs sous-ligne.
Probe N20k width32 RGB budget7440000 B : refuse minimum59520000 B, alors qu'un pixelRGB coûte1860000 B.
Les tests directs peuvent choisir le petit batch; le décideur produit ne peut pas.
Classification : capacité primitive présente / sélection manquante.

### F4 — HIGH — GPU externe absent de la sélection produit (A3)

`zemosaic_zegrid_mode.py:6523` ne sélectionne external que sous `not effective_gpu`;
le pseudo-plan est CPU-only. La primitive GPU a bien passé ses tests physiques, mais aucune
sélection produit external+GPU sous micro-budget VRAM n'est démontrée/câblée.
Ne pas assimiler la case GPU transmise par la GUI à un backend externe réellement utilisé.

### F5 — HIGH — Croissance A5 non appelée par l'exécution

`core/zegrid/scheduler.py:4111`: next_grow/grow_sequence ont seulement des appels en tests/dans le module.
Pas de rampe produit 4→1→4, hystérésis ou reprise re-dimensionnée démontrée.
`_replace_candidate:3998` conserve les anciens coûts et le drapeau feasible; les helpers sont
structurels, NON des admissions: un simple branchement serait insuffisant.
A1d différé laisse tile/backend dans le contrat scientifique gelé et limite la reprise avec autre budget/backend.
Classification : mécanisme partiel; A5 ne peut rester ACCEPTED.

### F6 — HIGH — Scratch incomplet / modèle de résidence non démontré (A4)

`zemosaic_zegrid_mode.py:4340`: le chemin global ignore l'allocator et appelle
`mosaic.assemble_canvas` sans l'argument. Probe guarded avec scratch demandé : **zéro appel allocator**,
science ndarray, pas memmap. Le modèle scratch peut pourtant retirer le canevas de la RAM.
Coût global réel de base : 7 plans totalisant48 B/px + hole_mask1 B/px, hors transitoires;
**pas121 B/px**, qui décrit le chemin feathered per_cell.
`cell_feather.py:724`: même per_cell crée valid_out, covered et valid_out.astype(float32) plein canevas.
`bounded_assembly.py:1267`: scratch reclassifie tout le canevas hors RAM sans borne prouvée de résidence
mmap; pas de réservation/vérification de disque libre pour ces plans dans ce chemin inspecté.
A4 doit couvrir global ET per_cell, transitoires, pages résidentes, disque et publication, pas seulement
fournir un allocateur memmap. Pas de dépassement RSS physique provoqué ici.

### F7 — MEDIUM — Nettoyage scratch silencieusement incomplet

`core/zegrid/bounded_assembly.py:1484`: discard unlink avant fermeture effective des mappings;
OSError avalé, listes effacées même en cas d'échec. Probe PermissionError simulant verrou Windows :
removed=0, fichier encore présent, suivi interne vide, mapping encore ouvert. Le caller ne journalise
pas ce zéro comme échec. Cleanup A4 uniquement en succès, pas finally englobant assemblage/finalisation.
Classification : défaut de gestion d'échec confirmé sous injection Linux; comportement natif Windows
non exécuté. Primitive préexistante nouvellement utilisée en produit.

### F8 — MEDIUM — Fallback GPU externe : provenance erronée et retries répétés

`core/zegrid/external_stream.py:1483` et `:1608`: token reste gpu après repli CPU, transmis à
finalize_streaming_result. Probe 4 tuiles, chaque GPU échoue de façon injectée : 4 tentatives GPU +
4 CPU; provenance annonce rejection=gpu/combine=gpu alors que toutes les réductions ont été CPU.
Le dict optionnel gpu_fallback_out reçoit la vérité, mais le caller produit ne le collecte pas.
Classification : défaut introduit A3; latent tant que F4 empêche external GPU produit.
Pas de boucle infinie démontrée: retries bornés par nombre de tuiles, mais absence de sticky fallback
et de journal produit fidèle. À corriger AVANT de brancher F4.

### F9 — MEDIUM / risque de débit — Repréparation complète répétée

`WindowedCanonicalProvider._load/get_tile` et `_prepare_cell_source_frame` : fenêtre d'une frame/patch,
PAS fenêtre spatiale. Source entière décodée puis patch entier reprojeté à chaque changement d'index.
Probe 3 frames et 4 tuiles :18 préparations (6 par frame). Les12 préparations de phase2 répètent
le travail par tuile, en plus des passes de préparation. C'est permis comme fallback mais son
amplification réelle d'I/O/débit n'est pas qualifiée sur M33. `bytes_read` compte les octets alignés
préparés, pas les lectures sources physiques. Aucun facteur de lenteur réel n'est annoncé.

### F10 — HIGH — Gates fermés sur preuves insuffisantes

- Q40 : 24x24 =>69696 B de plans, budget256MiB, scratch forcé; pas canevas>budget, pas `_write_outputs`,
  pas repli automatique bout-en-bout ni plafond RSS.
- Q41 : modèles/refus de publication; aucune finition/preview réelle sous ressources contraintes.
- Q50 : API/treillis, pas rampe dynamique/restart produit.
- Q60 : quatre frames, legacy explicitement, 1/2/4Gio observés simulés; ni plafond RSS/swap ni GUI ACTIVE
  ni preuve tous nouveaux chemins. Ce sont bien des petits runs E2E, mais pas la qualification produit.
- Q64 : propagation valide; insuffisante pour GUI ACTIVE→external GPU→scratch→FITS.
- Q61 BLOCKED; Q62/Q63 réels non lancés, corpus locaux disponibles.
- Matrice/STATE contradictoires et ancien A0_RUNNING. Annotations d'audit ajoutées sans effacer historique.

### F11 — HIGH — Identité working-set de reprise non gelée (défaut observé Windows)

Le run M106 émet : `NameError: name '_current_bundle' is not defined` lors de l'attachement de
l'identité working-set. `_current_bundle` est un helper local de `_run_r55_active_gpu_units`, mais
le bloc durable externe à cette fonction l'appelle aussi (`zemosaic_zegrid_mode.py:17116` et suivants).
L'exception est avalée et le run frais finit correctement, mais la reprise reste volontairement
conservatrice au lieu de réutiliser l'identité exacte annoncée. Ce n'est pas un simple bruit de log.

### F12 — MEDIUM — Télémétrie CuPy produit de faux AttributeError répétés

Le manifest M106 contient des erreurs répétées `pinned_pool_used_bytes:AttributeError` et
`pinned_pool_total_bytes:AttributeError`. CuPy 14.1.1 n'expose normalement pas ces méthodes sur
`PinnedMemoryPool`; `gpu_readiness.py` le gère déjà honnêtement, mais `gpu_memory.py:151-163` fait
un `getattr` non gardé et transforme l'absence d'une API optionnelle en erreur. L'exécution GPU et
les résultats ne sont pas affectés, mais la télémétrie d'alerte est fausse et masque les vraies erreurs.

### F13 — MEDIUM — Le résumé produits annonce à tort un échec du run log

`ROOT_DELIVERABLE_ROLES` exclut explicitement `run_log` car le log vivant est déjà écrit à la racine.
Pourtant `_products_text` applique à tous les rôles visibles absents de `root_files` le suffixe
`[generation copy, root delivery failed]`. Le M106 affiche donc un échec qui n'a pas eu lieu.
Les produits science/aesthetic/coverage/manifest sont bien livrés et vérifiés; corriger la sémantique
du résumé et conserver le vrai message d'échec pour un rôle racine effectivement attendu.

### F14 — LOW — Avertissements attendus trop bruyants

Le parcours GUI remonte au niveau WARN des paramètres volontairement ignorés, y compris plusieurs
valeurs par défaut (`False`, `mean`, `noise_variance`). Garder une trace structurée et avertir pour
un choix utilisateur non supporté est utile; présenter les valeurs par défaut comme des alertes
actionnables ne l'est pas. Reclasser/filtrer sans supprimer l'information de provenance.

## Points vérifiés / exclusions de faux positifs

1. **Non-régression des neuf échecs connus maintenant établie pour ces témoins** :
   baseline5150040:50 passed,9 failed en41.98s; HEAD418ba43:50 passed,9 failed en38.92s.
   Mêmes tests et causes: hash r53b identique en échec;6 resume_m1 (seams phase_profiler/OOM);
   2 gui policy (None vs auto et zegrid_stack_phase_profiling hors signature).
   Le précédent comparatif bab2b0b était insuffisant; le nouveau corrige cette preuve.
   Ceci ne prouve pas l'absence de toute régression sur tout le produit.
2. **Writer banded atomique pour l'échec rename testé** : os.replace injecté en échec conserve
   l'ancienne sortie et supprime .tmp. Code flush+fsync puis replace confirmé. Parité bytes/hash
   couverte par tests ciblés. Pas de garantie crash/panne électrique Windows déduite.
3. **Correction revue Nono N1** : conversion en variable locale stack_depth ne remplace pas
   assembled.stack_depth. La branche coverage banded n'est PAS inatteignable. Défaut réel = copies
   préalables/cast supplémentaire, confirmé par probe d'interception du writer.
4. **Correction revue Nono N7** : global n'alloue pas les11 plans121 B/px du feathered, mais
   48 B/px de base +1 B/px hole_mask et transitoires. Ignorer allocator reste un défaut matériel.
5. Nono n'a pas ré-exécuté les tests; Junior les a exécutés et a vérifié indépendamment ses nouveaux
   findings. Les deux erreurs de harnais exploratoire (kwargs request manquants, spool_root inexistant)
   ont été corrigées; ce n'étaient pas des défauts produit et elles ne figurent pas dans les findings.

## Validation effectivement exécutée

- Venv projet Python3.13, QT_QPA_PLATFORM=offscreen, PYTHONPATH=src.
- 54 tests ciblés A1–A6 :54 passed,565 warnings en32.66s, aucun skip (témoins GPU physiques inclus).
- Comparatif baseline/HEAD sur r53b per_cell, resume_m1_conservation, m6_gui_policy : résultats ci-dessus.
- Probes déterministes petits tableaux/mocks: copies, faux refus disque, plancher ligne, allocator global,
  branche coverage, cleanup verrou simulé, atomicité rename, nombre préparations, provenance fallback.
- Inspection du run Windows M106 `mem_fix` : config/manifest/publication/logs/breadcrumbs, intégrité
  des sorties racine et de la génération, sélection ACTIVE/GPU/dense, défauts F11–F14 ci-dessus.
- Aucune suite exhaustive, aucun benchmark lent N20k/Q30 rejoué, aucun M33/Caldwell complet,
  aucun dépassement mémoire ou disque volontaire. Le M106 est un run Windows natif, mais ne force
  aucun des chemins bounded/external/scratch/OOM qui restent à qualifier.

Preuves durables sous `/home/tristan/.openclaw/workspace/.a2a-reports/` :
- `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107.audit.targeted-tests.log`
- `.audit.probes.json` (même préfixe)
- `.audit.baseline-tests.log`, `.audit.head-regression-tests.log`
- `.audit.review-probes.py`, `.audit.review-probes.json`
- `.audit.junior.r0.md`, `.nono.audit.r0.md`, `.audit.junior.final.md`

## Statut de qualification corrigé (autorité Junior)

| Lot | Statut actuel |
| --- | --- |
| A0 | Défaut d'admission reproduit et limites quantifiées conservés; inventaire/Windows partiels connus |
| A1 | a/b/c testés; identité d différée; PLAN/ADMIT/RUN global non certifié |
| A2 | Primitives/parité conservées; intégration ressources à corriger (F2/F9) |
| A3 | Primitives CPU/GPU et rollback testés; intégration automatique non acceptée (F3/F4/F8) |
| A4 | Gate produit NON SATISFAIT (F1/F6/F7/F10) |
| A5 | Gate produit NON SATISFAIT (F5/F10) |
| A6 | NON QUALIFIÉ, pas seulement attente d'un test humain Windows |

## Corrections recommandées, séquentielles et mesurables

1. **Contrat de reprise et alertes honnêtes** : corriger F11–F14 avant les runs lourds pour que
   les diagnostics Windows distinguent un vrai défaut d'une option volontairement ignorée.
   Gate: run synthétique/GUI ACTIVE sans NameError ni faux AttributeError/échec de livraison;
   identité working-set persistée et reprise vérifiée.
2. **Admission/source et micro-blocs** : un seul contrat du provider réellement utilisé;
   supprimer cache fictif sans oublier les transitoires réels; sélection2D au pixelRGB minimum.
   Gate: vrai bundle produit N20k sous quota disque avec tous contributeurs et preuve plan=exécution.
3. **Assemblage/publication** : global/per_cell, scratch réservé et borné, finalisation+hash+coverage
   sans copie pleine taille, cleanup explicite sur succès/échec. Gate: E2E ACTIVE, canevas>budget
   matériellement contraint, science/coverage identiques et finitions réellement exercées.
4. **GPU externe** : budget VRAM complet, fallback sticky/provenance réelle, sélection depuis GUI.
   Gate: GPU physique microblocs sous plancher ancien puis OOM contrôlé, résultat unique et backend vrai.
5. **Adaptation/reprise** : résoudre contrat identité A1d puis plans de croissance re-tarifés,
   hystérésis/frontières sûres. Gate: rampe réelle 4→1→4 + restart/GPU absent, commits conservés.
6. **Qualification finale** : prouver chaque nouveau chemin via parcours GUI ACTIVE, contrôler RSS/swap/
   disque/handles/I/O puis M106/M16, Caldwell128 et M33 Windows. Ne pas fermer les gates avec seulement
   des assertions de modèles; ne pas promettre six budgets physiques sans machine/contrainte adaptée.

L'audit est terminé; aucune correction substantielle n'a été lancée dans ce volet de vérification.
Le prochain lot recommandé est (1), en conservant ce mission_id et la traçabilité des corrections.

## Corrections appliquées — lot 1 (F11–F13) — 2026-11-07

Les trois défauts du lot « contrat de reprise et alertes honnêtes » ont été
corrigés, vérifiés et committés localement sur `feat/zegrid-adaptive-resources`
(pas encore poussé au moment de la rédaction) :

### F11 — HIGH — identité working-set de reprise non gelée — CORRIGÉ

- `src/zemosaic/zemosaic_zegrid_mode.py` introduit
  `_run_scope_bundle(active_bundle, final_bundle)`, remplace le helper local
  `_current_bundle` par l'accesseur `_gpu_current_bundle`, et route les six
  références extérieures invalides vers `_run_scope_bundle(_r55_active_bundle, _final)`.
- Commit : `9fd304b` (Coco), vérifié par Junior.
- `rg "_current_bundle\("` ne montre plus aucune référence résiduelle ;
  `git diff --check` vide.

### F12 — MEDIUM — faux AttributeError pinned pool CuPy — CORRIGÉ

- `src/zemosaic/core/zegrid/gpu_memory.py` : l'absence d'un compteur
  pinned-memory est désormais traitée comme télémétrie absente (`None`), pas
  comme erreur ; un attribut présent qui lève est reporté honnêtement.
- Commit : `0cd0080` (Junior).

### F13 — MEDIUM — faux « root delivery failed » sur run_log — CORRIGÉ

- `src/zemosaic/core/zegrid/run_publication.py` : `_products_text` distingue le
  `run_log` auxiliaire (copie d'archive, log vivant à la racine) d'un échec de
  livraison d'un rôle racine effectivement attendu.
- Commit : `0cd0080` (Junior).

### Témoin de reprise ACTIVE mis à jour

- `tests/test_zegrid_active_resume_parity.py::test_fresh_active_tiny_run_freezes_the_admission_from_its_identity`
  vérifie désormais que le `working_set` gelé reproduit le bloc annoncé par le
  manifest du run (classe + `tile_size`→`tile_px` + `column_batch`), au lieu d'un
  littéral `inmem`/`tile_px:0` erroné.

### Validation

- `tests/test_zegrid_active_resume_parity.py tests/test_zegrid_m5_product_e2e.py
  tests/test_zegrid_r55e11_runtime_vram_adaptation.py tests/test_zegrid_r55c_gpu_readiness.py
  tests/test_zegrid_publication_execution_m5b.py` → **105 passed**.
- `tests/test_zegrid_f11_run_scope_bundle.py tests/test_zegrid_alert_integrity_f12_f13.py
  tests/test_zegrid_m4_external_durability.py tests/test_zegrid_publication_m5b.py`
  → **79 passed**.

F14 (WARN attendus trop bruyants, LOW) reste à corriger dans un lot ultérieur.
Les lots F1–F10 restent ouverts.

## Corrections appliquées — lots F1–F10 (hors F14) — 2026-11-07

Tous les défauts matériels de l'audit ont été corrigés et vérifiés
indépendamment par Junior, committés localement sur
`feat/zegrid-adaptive-resources` (aucun push au moment de la rédaction) :

| Défaut | Contenu | Commit | Vérification |
| --- | --- | --- | --- |
| F2 (HIGH) | enveloppe disque external ne facture plus le cache aligné N×patch fantôme ; `resident_unit` = staging résident exact | `b251643` | suite requise 170 passed/1 skipped |
| F3 (HIGH) | `largest_admissible_column_batch` plancher abaissé au pixel (granularité `channels`), bit-exact vs ligne | `b251643` | idem |
| F4 (HIGH) | external sélectionnable sous GPU (`allow_external=is_bounded`, gate sans `not effective_gpu`), pseudo-plan CPU-only | `aad9f64` | 172 passed (venv CuPy) |
| F8 (MEDIUM) | sticky fallback (1 transition GPU→CPU/cellule), provenance `requested=gpu`/`rejection=combine=cpu`, caller publie `gpu_fallback` | `aad9f64` | idem |
| F1 (HIGH) | `_write_outputs` sans copie plein canevas (hash+écriture bandés) | `0ed7cb2` | 29 passed + 148 passed élargie |
| F6 (HIGH) | allocator scratch routé global + per_cell, modèle gauge-mode aware (49 B/px global vs 125 B/px per_cell) | `0ed7cb2` | idem |
| F7 (MEDIUM) | `discard()` reporte/leve sur échec, close avant unlink, cleanup dans `finally` | `0ed7cb2` | idem |
| F5 (HIGH) | grow A5 câblé dans l'admission ACTIVE préflight (hystérésis, borné, trail, garde A1d pinned) | `3c81aaf` | 118 passed + 74 passed |

F10 (honnêteté de qualification) : la matrice `05_QUALIFICATION_MATRIX.md` et la
fiche `ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md` ont été corrigées pour
ne plus marquer `PASS` des gates aux preuves partielles (Q40/Q41/Q50/Q60/Q64
passent en `PARTIEL`). Ces corrections lèvent les faux refus et la dette de
copies mémoire, mais ne ferment pas à elles seules les gates produit : les runs
réels sur machine contrainte (grand canevas M33, rampe 4→1→4 matérielle,
GUI ACTIVE→external GPU→scratch→FITS) restent à exécuter. F14 (WARN, LOW) est
exclu du périmètre par décision Tristan. A1d (schéma d'identité v2) reste différé.
