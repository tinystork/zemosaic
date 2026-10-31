## Junior — ZM-ZEGRID-COMMIT-INTEGRITY-20261031 : ACCEPT (2026-10-31)

Incident `/home/tristan/M106/OUT/RESUME_NEW` : store créé 01:33:40 avec 20 cellules attendues, `commits/` et `objects/` VIDES, index à zéro, mais `per_cell_stack` annoncé 0/20 → 19/20 pendant ~14 min, puis échec à l'assemblage `bounded_assembly.py:434 → run_store.py:2609 StoreNotFoundError: no durable commit for cell 'r0000c0000'`.

Cause structurelle trouvée : la progression était émise par le callback de fin de worker (`mode.py:12149 _wave_progress`), donc AVANT la boucle de commit du parent (`mode.py:12680+ accept → commit_and_release`). Un chemin de commit dégradé paraissait donc sain pendant des minutes. Le code actuel commite correctement sur runs réels (tests r4/visibles verts) ⇒ l'incident lui-même reste NON reproduit ; c'est la fenêtre silencieuse qui est fermée.

Livré : `StoreCommitNotDurable` (`run_store.py:189`), `require_commit` (`:2648`), vérification après `commit_and_release` (`mode.py:12726`), échec de commit TERMINAL, progression avancée seulement après commit vérifié (`_durable_wave` `mode.py:12556`), fail-fast avant assemblage `_require_durable_commits_before_assembly` (`mode.py:145`, appel `:15581`), lignes `durable_commits=N` au départ et à la fin.

Vérification Junior indépendante :
- hashes conformes : `run_store e92a470e…`, `mode f2f8d69e…`, `tests 447c75ef…`
- nouvelle suite **6 passed** ; combiné parity+visible+commit+resume_m4+classic **92 passed**
- **Probe A** (vérification par cellule retirée, syntaxe `ast.parse` OK) ⇒ `test_silent_commit_fails_at_the_first_cell` échoue ; hash restauré.
- **Probe B** (fail-fast zéro-commit désactivé, syntaxe OK) ⇒ `test_zero_commit_store_refuses_assembly` échoue ; hash restauré.
- rejeu : m6 unified + qt_stop_resume **95 passed, 1 deselected** ; gui_policy **48 passed** ; diff-check OK ; HEAD `87e98a46…` inchangé, 24 entrées sales, aucun commit.

Limite documentée : la vérification par cellule est câblée sur la voie ACTIVE CPU (le chemin de l'incident) ; la voie GPU garde une progression pilotée par le registre, couverte par le garde zéro-commit avant assemblage.

## Junior — ZM-ZEGRID-VISIBLE-ARTIFACTS-20261031 : ACCEPT (2026-10-31)

Demande produit de Tristan : ZeMosaic est open source, un utilisateur doit voir ce que le logiciel produit ; les dossiers de travail cachés doivent devenir visibles.

Renommages livrés :
- `.zegrid_publications` → `zegrid_publications` (`run_publication.py:65`, legacy en `:66`), découverte `discover_publication_root` (`:631`), `legacy_publication_in_use` (`:650`), écriture forcée visible `run_publication_root(for_write=True)`.
- `.zegrid-durable` → `zegrid-durable` (`zemosaic_gui_qt.py:2684`, legacy `:2685`), découverte `_zegrid_store_discovery_root` (`:2716`).
- `__zegrid_cache__` → `zegrid-cache` (`zemosaic_zegrid_mode.py:139`, legacy `:140`), marqueur de propriété `.zegrid-cache-owner`.
Règle : nouveaux runs écrivent UNIQUEMENT les noms visibles ; les anciens dossiers cachés restent lisibles en secours avec une ligne de log explicite ; aucun déplacement/suppression/migration ; collision avec du contenu étranger ⇒ refus fail-closed. Listes de propriété (publication + cache) reconnaissent les deux noms ; CLI aligné.

Vérification Junior indépendante :
- hashes conformes : `run_publication b159bd3c…`, `mode f95419eb…`, `gui fe339b18…`, `cli dce57a4a…`, `tests 5316ca5b…`
- nouvelle suite **13 passed** ; combiné parity+visible+resume_m4+classic **86 passed**
- **Probe A** (renommage annulé) ⇒ `test_fresh_run_writes_only_visible_work_directories` échoue ; hash restauré.
- **Probe B** (fallback legacy supprimé, syntaxe vérifiée par `ast.parse`) ⇒ `test_legacy_layout_resumes_and_verifies_without_being_moved` échoue ; hash restauré.
- rejeu : m6 unified + qt_stop_resume **95 passed, 1 deselected** ; gui_policy **48 passed** ; diff-check OK ; HEAD `87e98a46…` inchangé, 23 entrées sales, aucun commit.

Limites assumées : le dossier existant `/home/tristan/M106/OUT/RESUME` garde ses noms cachés (pas de migration) ; un `__zegrid_cache__` résiduel peut subsister ; le marqueur de propriété du cache reste un petit fichier caché dans un dossier visible.

## Junior — r4 produits publiés visibles : ACCEPT (2026-10-31)

Défaut réel : run `/home/tristan/M106/OUT/RESUME` finalisé (108/108, gen-1, verified) mais produits uniquement dans `.zegrid_publications/run-<id>/gen-1/` (science 94,5 Mo, aesthetic 94,5 Mo, coverage 31,5 Mo, manifest 493 Ko). Racine de sortie vide ; GUI « Processing completed successfully » puis propose d'ouvrir `output_dir` (la racine) ⇒ impression de run sans fichier.

r4 : `write_visible_products` (`run_publication.py:2020`) écrit `zemosaic_products.json` + `PRODUCTS.txt` à la racine, pointant la génération immuable, après que `CURRENT.json` est durable ; `_replace_visible_pair` (`:1948`) staging puis deux `os.replace` avec rollback ; appels `mode.py:15968` (après commit) et `mode.py:14398` (reprise d'un run déjà complet). GUI : `_zegrid_products_dir` (`:3421`), `_completion_message_text` (`:3451`), `_completion_open_target` (`:3468`) ⇒ message nommant le dossier produits et ouverture de la génération, pas de la racine. Classic/SDS/legacy inchangés.

Vérification Junior indépendante :
- hashes conformes : `run_publication 0f81c498…`, `mode d4984604…`, `gui f9abcaeb…`, `tests e406c24d…`
- parity suite **34 passed**
- **Probe I** (écriture des marqueurs désactivée) ⇒ `test_published_products_are_visible_at_the_output_root` échoue ; hash restauré.
- **Probe J** : premier essai INVALIDE (motif sur 3 lignes, remplacement d'une seule ⇒ SyntaxError, échec non probant) ; refaite proprement, syntaxe vérifiée par `ast.parse`, puis échec sémantique du test ; hash restauré.
- rejeu : resume_m4 + classic **39 passed** ; m6 unified + qt_stop_resume **95 passed, 1 deselected** ; gui_policy **48 passed** ; diff-check OK ; HEAD `87e98a46…` inchangé, 22 entrées sales, aucun commit.

Note : les marqueurs seront écrits pour le run déjà complet au prochain lancement de ce dossier (chemin idempotent `mode.py:14398`, sans recalcul). Aucun FITS copié, aucun lien symbolique.

## Junior — r3 comptabilité de progression : ACCEPT (2026-10-31)

Défaut confirmé par mesure sur le run live : le compteur de `per_cell_stack` repartait de 1/108 et le champ `item=` venait du préfixe du plan, alors que le moteur sautait correctement les cellules commitées. Mesures : 00:38:05 → 55 commits (47 anciennes + 8) affiché `9/108 item=r0000c0008` ; 00:38:50 → 57 commits affiché `11/108 item=r0000c0010`. Aucun fichier de commit ancien réécrit (mtimes 00:26–00:27 intacts) ⇒ aucun recalcul, uniquement un affichage faux et un ETA calculé sur le mauvais reste.

r3 : `_resume_stack_accounting` (`mode.py:2264`, committed = complete + empty_validated, invariant committed + pending == total), `_resume_progress_line` (`:2285`), `_stack_item_cid` (`:2293`), offset appliqué à `_stack_report` (`:11392-11406`, `:11643`, `:11995`, offset posé `:14337`). ETA inchangé mais alimenté par le `done` corrigé.

Vérification Junior indépendante :
- hashes conformes : `mode.py 12c97e16…`, `tests 5d57a262…`
- parity suite **30 passed**
- **Probe G** (offset remis à 0) ⇒ `test_resume_progress_real_run_starts_from_the_committed_count` échoue ; hash restauré.
- **Probe H** (label d'index de plan restauré) ⇒ même test échoue ; hash restauré.
- rejeu : resume_m4 + classic **39 passed** ; m6 unified + qt_stop_resume **95 passed, 1 deselected** ; gui_policy **48 passed** ; diff-check OK ; HEAD `87e98a46…` inchangé, 22 entrées sales.

Run live `/home/tristan/M106/OUT/RESUME` : arrêté à 00:42:20 par **WORKER_COOPERATIVE_STOP** (arrêt demandé, pas un crash) à **71/108 cellules commitées**. Store intact, admission cpu/tile80, reprise possible. Au prochain lancement le log devra afficher « 71/108 cells already committed, 37 to stack » et la progression démarrer à 71/108.

## Junior — r2 backend gelé : ACCEPT (2026-10-31)

Défaut réel observé par Tristan : store gelé `cpu/tile80`, relance avec GPU activé ⇒ refus « a GPU backend was selected for a CPU-frozen run ». Cause : le backend gelé n'était appliqué que dans un sens (`frozen gpu` sans GPU ⇒ refus) ; `frozen cpu` avec GPU sélectionné n'était pas forcé ⇒ plan GPU ⇒ refus d'identité.

r2 : helper unique `_frozen_planning_gpu` (`mode.py:2264`) utilisé par les deux entrées de planification (décision de lancement `mode.py:10238`, builder `mode.py:4857`). `frozen cpu` ⇒ `effective_gpu=False`, une ligne de log « GPU ignored — CPU-frozen run ». `frozen gpu` sans GPU ⇒ refus typé inchangé. Store sans section et runs neufs ⇒ comportement inchangé. `legacy` intact.

Vérification Junior indépendante :
- hashes conformes : `mode.py 3d7b6380…`, `tests ab77ae5c…`, `run_resume 79dffcbf…`, `run_store d19d7169…`, `gui 3b3862a6…`
- parity suite **26 passed**
- **Probe E rejouée** : `frozen cpu` remis à `bool(gpu_used)` ⇒ 2 tests échouent ; hash restauré exact.
- **Probe F rejouée** : refus GPU neutralisé ⇒ `test_frozen_gpu_backend_without_a_gpu_is_refused` échoue ; hash restauré exact.
- Rejeu : resume_m4 + classic **39 passed** ; m6 unified + qt_stop_resume (hors synthetic) **95 passed, 1 deselected** ; gui_policy **48 passed** ; `git diff --check` OK ; HEAD `87e98a46…` inchangé, 22 entrées sales, aucun commit.

Limite explicite : aucun run GPU réel, GPU simulé disponible dans les tests. Le store `/home/tristan/M106/OUT/RESUME` (admission cpu/tile80) devrait désormais reprendre en CPU sans décocher le GPU — à condition de lancer le code du worktree (`PYTHONPATH=$PWD/src zemosaic`), pas le beta périmé de l'install éditable.

## Junior — ZM-ZEGRID-ACTIVE-RESUME-PARITY-20261030 : ACCEPT (2026-10-30)

Nono review-0 = ACCEPT-WITH-CONDITIONS (C1 threading exhaustif, C2 refus propres, C3 non-vacuité). C2 fermée structurellement. **C1 et C3 échouaient réellement** : sonde Junior sur r0 — couper `frozen_admission=_frozen_admission` dans `_plan_request` laissait **17/17 verts**. La suite prouvait que le planificateur respecte une admission gelée qu'on lui donne, pas que la production la transmet.

Rework r1 (même mission, même session) : 6 tests ajoutés (23 au total), helper `_admission_from_identity` (`mode.py:9572`), site de création `mode.py:14437`, `_snapshot_for_run` (`gui.py:9477`) comme point d'appel unique.

Vérification Junior indépendante :
- hashes conformes : `mode.py 6ab4fe53…`, `gui.py 3b3862a6…`, `tests f2b17df7…`, `run_resume 79dffcbf…`, `run_store d19d7169…`
- **Probe A rejouée par Junior** : transmission coupée ⇒ `test_run_single_threads_the_frozen_admission_into_the_real_plan_request` **échoue** (1 failed, 1 passed) ; fichier restauré au hash exact.
- **Probe D rejouée par Junior** : `_apply_effective_zegrid_resume_to_snapshot` retiré ⇒ `test_snapshot_for_run_applies_the_effective_resume_mode` **échoue** ; hash restauré exact.
- Rejeu : parity **23 passed** ; resume_m4 + classic witness **39 passed** ; m6 unified + qt_stop_resume (hors synthetic) **95 passed, 1 deselected** ; gui_policy **48 passed** ; `git diff --check` OK ; HEAD `87e98a46b60afc201999eb0db1e9c0057c05ba3d` inchangé, aucun commit.

Livré : admission (backend/tile_px/parity_qualified) gelée une fois à la création depuis l'identité validée, réutilisée en reprise au lieu d'être redérivée de la RAM courante ; ladder restreinte à la tuile gelée ; refus typé avec requis/disponible/déficit et remède ; stores anciens sans section ⇒ comportement conservateur inchangé ; `legacy` intact.

Réserve connue, non bloquante : sensibilité d'ordre inter-modules préexistante (`test_real_tiny_run_uses_gui_policy_values` échoue seulement quand plusieurs suites M6 tournent dans le même process). Reproduite à l'identique sur la source r0 exacte ⇒ antérieure, hors périmètre.

## Junior — active non reprenable comme legacy : diagnostic + mission bloquée (2026-10-30)

Défaut confirmé sur `/home/tristan/M106/OUT/RESUME` : identité gelée `backend.cpu`, `tile_px=80`, `parity_qualified=false`, 108 cellules, canvas 3278x2403 ; refus à `mode.py:13968 _resume_boot.verify_identity` → `run_resume.py:577`, sections `grid` + `backend`.

Cause : `run_resume.validate_policy` ne gèle que la politique de stockage (`requested/effective/fallback_count/transitions/dense_refusal`). Les entrées d'admission qui font partie de l'identité scientifique (`backend`, `tile_px`) sont **re-dérivées à chaque lancement** depuis les faits machine (`available_ram_bytes` injecté dans les faits de planification, `mode.py:9412`, PlanContext `~5948`). Une RAM libre différente ⇒ tuile différente ⇒ refus d'une reprise pourtant scientifiquement valide. `legacy` y échappe seulement parce que sa tuile est fixe.

Nettoyage : le dossier n'était pas vide — `.zegrid-durable` (692K, run du 11:50) subsistait. Déplacé en `.zegrid-durable.stale-20261030-120602` (récupérable, rien supprimé). Le dossier est maintenant réellement vierge.

Mission `ZM-ZEGRID-ACTIVE-RESUME-PARITY-20261030` rédigée (brief `.a2a-reports/...coco-mission.md`) : geler backend/tile_px/parity dans l'état durable, les réutiliser en reprise au lieu de les re-dériver, refus typé actionnable si la tuile gelée ne tient plus, compatibilité ascendante des stores sans la nouvelle section, message de refus avec valeurs stockées vs courantes, correction de l'incohérence `zegrid_resume_mode` (snapshot `never` alors que la porte AUTO a été franchie), `legacy` intact.

**BLOQUÉ** : `sessions.reset` sur `agent:coco:main` a échoué 3 fois (`gateway request timeout for sessions.reset`), alors que `openclaw status` montre le gateway actif (2026.10.4, 122 sessions). Protocole : pas de nouvelle mission indépendante sans reset, pas de transport alternatif inventé. Mission staged, non dispatchée. Options : réessayer le reset plus tard, ou autorisation explicite de dispatcher sans reset.

## Junior — ZM-UNIFIED-STOP-SAVE-CAPABILITY-20261030 : ACCEPT (2026-10-30)

Nono review-0 = ACCEPT avec 3 LOW. Les trois conditions ont été fermées par Junior avant acceptation :

- **L-1 (non-vacuité)** : sonde contrôlée sur `zemosaic_zegrid_mode.py` — garde désactivé temporairement, 5 tests échouent (`test_engine_refuses_durable_resume_without_active_scheduler` ×4 + `test_engine_guard_does_not_create_a_missing_store`). Fichier restauré, hash identique à celui du rapport Coco (`e2cab6ae…`). La suite ne passe donc pas à vide.
- **L-2 (texte)** : `qt_zegrid_stop_saved` n'est émis que dans la branche `ack_state == "STOPPED_RESUMABLE"` ; `STOPPED_IN_MEMORY` dit explicitement « NOT resumable » ; la voie générique appelle `worker_controller.stop(graceful=True)` et n'affirme jamais de reprise.
- **L-3 (flake SIGTERM)** : `tests/test_zegrid_m6_stop.py` est déjà suivi à HEAD (`87e98a4`), donc antérieur à ce delta et hors périmètre de la mission. Passes isolée + 3 runs combinés = vert. Reporté comme tâche séparée de stabilisation (attente fixe → poll borné), non requalifié en succès.

État final : HEAD `87e98a46b60afc201999eb0db1e9c0057c05ba3d` inchangé, **aucun commit**, `beta`/`main` intactes. 20 entrées sales (15 modifiés + 5 non suivis), dont 18 de base préservées ; 2 ajouts de cette mission (`run_resume.py`, `tests/test_zegrid_m6_unified_stop_capability.py`).

Capacité effective livrée : bouton unique, `ZEGRID_DURABLE` réservé à zegrid+active+checkpoints ON (gelé au lancement, transporté à l'enfant), tout le reste en arrêt gracieux historique ; refus typé du moteur pour une reprise durable hors `active`, avant tout store ou verrou.

Restent ouverts et non bloquants : commit/push des deltas (non autorisés), preuve synthétique lourde stop→reprise en nouveau processus (bloquée par la RAM disponible), bench M3/NTFS Windows en attente du retour de Tristan.

## Junior — vérification r0 du Stop-and-Save unifié (2026-10-30)

Rapport Coco `.coco.r0.md` (DONE) relu et diff inspecté. HEAD toujours `87e98a46b60afc201999eb0db1e9c0057c05ba3d`, aucun commit, 20 entrées sales dont les 18 de base préservées.

Garde-fou moteur vérifié en source : erreur typée `ResumeSchedulerUnsupported` (`core/zegrid/run_resume.py:81`), refus dans `_run_single` (`zemosaic_zegrid_mode.py:9757-9796`) placé après `normalize_resume_mode` et AVANT résolution du store, `_validate_capture_store_compatibility`, `ResumeBootstrap.open`, `select_run` et `RunCellStore.open`. Message actionable, pas de fallback silencieux, chemin `active` non touché.

Résolveur de capacité vérifié : `_resolve_stop_capability` (`zemosaic_gui_qt.py:2712`) exige zegrid + scheduler `active` + checkpoints ON, et bascule en fail-closed sur valeur de scheduler non normalisable. Transport (`:9415`) envoie toujours `zegrid_stop_capability` et n'expédie les kwargs durables que pour `zegrid_durable`.

Rejeu indépendant : nouvelle suite + caractérisation Classic = **61 passed** ; `test_zegrid_m6_qt_stop_resume.py -k "not synthetic"` = **51 passed, 1 skipped, 1 deselected** ; `test_zegrid_resume_m4.py` = **21 passed** ; `gui_policy + m6_stop` = **49 passed ×3**.

**Flake observé** : un run combiné antérieur a fait échouer `test_sigterm_stops_resumably_with_intact_durable_commits` ; il passe isolément et sur 3 runs combinés consécutifs. Signalé à Nono comme risque à qualifier (timing pré-existant vs régression introduite), non requalifié comme succès.

Revue Nono review-0 lancée, accepted `c81d6594-7560-4f90-b6f0-32e7f83e541c`, rapport attendu `.nono.r1.md`. Aucun commit/push.
## Junior — plan Stop-and-Save unifié revu par Nono, mission confiée à Coco (2026-10-30)

Défaut confirmé : run ZeGrid avec scheduler `legacy` persisté + checkpoints ON => le GUI prépare un plan durable et expose Stop and Save, mais registre/canal/store n'existent qu'en `active`. Aucun consommateur, aucun store, aucun ack.

Plan `.a2a-reports/ZM-UNIFIED-STOP-SAVE-CAPABILITY-20261030.junior.plan.r0.md` : quatre capacités gelées par lancement (ZEGRID_DURABLE vs LEGACY_GRACEFUL), préflight/transport durable strictement réservés à active+checkpoints, bouton unique routé par capacité, textes honnêtes, aucune migration automatique de legacy/observe, préservation des caches Phase 1/Phase 5 Classic.

Nono design review-0 = ACCEPT AVEC AMENDEMENTS (A1–A4 + 7 cas de matrice). A1 vérifié par Junior et il révèle un vrai trou : l'ouverture du store frais est bien gatée `active` (via result_registry), MAIS le bootstrap de reprise ouvre un store dès `resume_mode=auto` sans condition de scheduler. Le moteur doit donc refuser explicitement resume-auto/run_id hors `active` avant tout store ou verrou.

Mission `ZM-UNIFIED-STOP-SAVE-CAPABILITY-20261030` confiée à Coco (reset confirmé) accepted/queued `21c4db58-6f6e-4cef-a020-d66e8e94cbcb`. Brief `.a2a-reports/...coco-mission.md`, rapport attendu `.coco.r0.md`, exact-source callback. Aucun commit/push ; aucun run utilisateur interrompu. Après callback : vérification Junior puis Nono.

## Junior — code review accepted; full validation BLOCKED_RESOURCE (2026-10-30)

Nono review3 ACCEPT read; only STATE changed since reviewed snapshot. Junior separately read GUI callsite/tests omitted by Nono, already executed independent86pass1resource-skip2deselected21.50s incl actual F1/F2 closures. Code-review ACCEPT scoped; mission NOT fullyaccepted: required C2 successful final-tree new-process resume pending, earlier attempt was RAM-refused (NOT an unattempted test). CurrentRAM0.86GiB, no retry/guard reduction/process termination. No new implementation warranted, no agents pending.

Terminal state BLOCKED_RESOURCE for qualification. Next single step when RAM genuinely sufficient: replay only synthetic GUI stop/new-process resume and retain complete result. User's real-corpus Linux test separate; Windows not prerequisite for LinuxGUI.18dirty files preserved/HEAD87e98a46feature; no commit/push/merge. Usage doc corrected for actual engine refusal scope, exact worktree launch and uncommitted/status limitations (docs-only). Durable decision `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.junior.review3-decision.md`. Callback final private, no source-channel send.

## Junior — r4 F1/F2 verified; final review3 queued (2026-10-30)

User-authorized extension delivered. All15Coco hashes matched; only Junior docstring correction afterwards (stale all-members requirement), no executable change.18expected dirty files/feature HEAD87e98a46unchanged. Independent Qt+publication+own repros **86passed1skipped2deselected47warnings21.50s**; injected artifact failure test skipped on resource-preflight path (exact exception text not retained); no claim this exercised init error anew. Heavy synthetic/new-process and real12x384 excluded, MemAvailable1.01GiB. C2 remains pending after prior attempted RAM refusal.

F1 empty/partial-stage resume and F2 unrelated-stage beside realCURRENT verified fixed by both persistent tests and own former repros, no mutations. C1 real publication proof closed. Evidence `.junior.r4.md`, `.review3-snapshot.json`, `.review3-brief.md`, `zm_qt_r4_junior_tests.log`, `zm_qt_r4_edges.py` in workspace .a2a-reports. Nono same-context review3 accepted/queued `ca29af93-080c-4790-98ce-7368172d43e7`; expected `.nono.r3.md`, read-only/lightweight. No polling/reset; wait callback. No commit/push/acceptance yet, no further correction automatically authorized after r4.

## Junior — user authorizes bounded extension; REWORK4 queued (2026-10-30)

Tristan: « oui tu peux reprendre » after iteration-cap HUMAN_GATE. One additional bounded correction F1/F2 authorized; same mission ZM-ZEGRID-M6-QT-STOP-RESUME-20261030/Coco context, no reset. Branch feat/zegrid-resilient-run-resume HEAD87e98a46 unchanged,18expected dirty files verified.

REWORK4 accepted/queued `000ff96d-0281-4715-beca-9263138399d0`. Brief workspace `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.rework4.md`, expected `.coco.r4.md`. Scope permit valid dead-owner stage with zero/subset existing members (ownership, NOT completeness); refuse unrelated staged run beside validCURRENT; persistent real fixtures with read-only/no-mutation proofs. No engine/science/guard/default/legacy change. C2 successful independent final-tree new-process test still pending actual sufficient RAM; no heavy retry under shortage. Next callback: independent tests+F1/F2 probes, then focused Nono review if closed. NOT accepted/no commit/push; no polling.

## Junior — final r3 checked; HUMAN_GATE iteration cap (2026-10-30)

All15reported hashes match; feature HEAD87e98a46 unchanged,18expected dirty files. Junior81passed2deselected45warnings22.10s on Qt stop/resume + publication suite. Real CURRENT/corruption test and nested sibling fix verified. Additional2probes confirm REMAINING defects: (F1) same-run stage afterbegin with missing output members refused despite valid dead-owner marker and engine-owned orphan cleanup; (F2) unrelated run staging ignored beside selected validCURRENT. No data mutated/lost. Evidence mission `.junior.r3.md`, `zm_qt_r3_edges.py/.log`, `zm_qt_r3_junior_tests.log` in workspace .a2a-reports. Snapshot `.r3-final-snapshot.json` excludes subsequent STATE update.

Three corrective iterations exhausted: HUMAN_GATE/NOTaccepted per AGENTS iteration cap, NO fourth dispatch/reset/self-fix. Recommend explicit one-round extension limited to F1 partial-member subset + F2 all-root ownership validation, then independent tests/review. No worker pending. C2 successful final-tree new-process resume still pending after earlier RAM refusal; available1.45GiB, no heavy retry/guard change. No commit/push/full-suite/Windows/GPU/user-data claim. Current callback source policy keeps final private; no source-channel message sent.

## Junior — conditional review adjudicated; final REWORK3 queued (2026-10-30)

Nono review2 ACCEPT callback means ACCEPT-WITH-CONDITIONS. Junior real12x12 publication probe closes C1 once: actual CURRENT/digests accepted, real science corruption refused, hashes unchanged by preflight. Corrected setup after initial missing-directory probe failures; final3passed2warnings3.44s. Evidence workspace `.a2a-reports/zm_qt_review2_probe.py` and log; adjudication mission `.junior.review2.md`.

Two additional confirmed gaps: legitimate same-run orphan stage/noCURRENT refuses same-output GUI resume though engine finalization verifier accepts incomplete state; nested store ancestor allows hidden unrelated .zegrid_publications sibling. Bounded third/final Coco correction SAMEcontext accepted/queued `ea95fcfd-3508-46ef-9c62-322ca69cb40c`: strict read-only staging ownership validation, nested guard, persistent real-publication/adversarial tests. Brief mission `.rework3.md`, expected `.coco.r3.md`. No fourth corrective iteration. Source-reviewed normal listener waits for process exit (no confirmed new normal ack race); late commit cleanup concern not independently reproduced, no wider engine edits requested.

MemAvailable0.84GiB: no costly integration retry, no lower guards. C2 successful independent final-tree new-process resume still pending after previous RAM refusal. NOT accepted/no commit/push. Prior Qt policy preserved, HEAD87e98a46/feature unchanged. Await callback/no polling or reset.

## Junior — interruption recovered; targeted review2 queued (2026-10-30)

Human requested recovery. Read durable Nono review1 after provider callback failure: requested source closures were NOT inspected, no new confirmed defect; not acceptance. Repository still feature HEAD87e98a46 with18expected dirty files, prior Qt policy delta preserved. Fresh MemAvailable1.16GiB: no heavy retry, no guard changes. Final-tree new-process completion gate remains pending after prior RAM refusal.

Same Nono context review2 accepted/queued `78ff1d80-cd30-4a22-9c04-f3c585b23737`, exact source ranges supplied in `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.review2-brief.md`. Expected `.nono.r2.md`; lightweight review only. No agent reset/polling, no new Coco iteration. Two corrections used. Still NOT accepted, no commit/push. Await callback, then adjudicate findings and resource-dependent validation.

## Junior — r2 verified, replay RAM refusal; Nono review1 queued (2026-10-30)

Mission `ZM-ZEGRID-M6-QT-STOP-RESUME-20261030`, HEAD87e98a46/feature unchanged;18expected dirty files now include authorized engine change. All15Coco hashes match; diffcheckclean. Source fixes R1/R2/ack retention+early typed artifact refusal inspected. Junior174test progress: ONE failure, synthetic second-new-process resume rejects `R55_REFUSE_HOST_RAM`; other entries pass incl real12x384NaN early refusal+lock release. Final aggregate/duration truncated by tool, no invented green summary. Pre-run MemAvailable1.583GiB printed (not gated; should have deferred expensive integration); no guards weakened or repeated run. Earlier160pass was r1, not r2.

Artifacts `/var/tmp/openclaw-tristan/pytest-of-tristan/pytest-47/test_synthetic_gui_built_stop_0`: originalstore3commits/finalizedFalse, observedcopystore4/finalizedTrue; execution observer proof passed, second spawned completion remains validation gate. No lost data or successful-final-publication claim. Preserve and rerun only with concrete resource improvement.

Nono same-context review1 accepted/queued `1e89f558-4469-493e-b0af-7e0e28e010ef`; expected `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.nono.r1.md`. Review1snapshot/evidence junior.r2.md + review1-brief.md. Check prior issue closure and actual cleanup/staged-publication/ownership paths with lightweight probes; no heavy rerun under tight RAM. Two Coco corrections used; final correction only for consolidated material findings. Still NOTaccepted/no commit/push/user corpus/Windows/pressure. Await callback/no polling.

## Junior — Nono FINDINGS adjudicated; REWORK-2 queued (2026-10-30)

Mission `ZM-ZEGRID-M6-QT-STOP-RESUME-20261030`, HEAD87e98a46 unchanged; review snapshot unchanged except STATE. Nono confirms R1 resume output bypass/R2 pending close-latch and engine store-existence issue, no runtime probes. Junior corrects review imprecisions: engine file NOT modified; Junior NaN failure not reproduced (today NOT_RUN before generation due MemAvailable<2GiB); final publication is NOT prerequisite for assembly; blanket publication refusal must not break matching crash-after-CURRENT reconciliation. Durable adjudication `.junior.review0.md`.

Second bounded correction SAME Coco/no reset accepted/queued `22ac92df-1763-44c2-8885-9af362ed5bfb`: resume-specific verified output/ownership checks incl nested-store ancestor; pending spawn Save-close latch + fast ack/done handling; authorized minimal engine explicit durable-artifact failure refusal before cells, lock cleanup, no strictJSON/science weakening, no all-payload RAM fallback; regression proofs. Brief `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.rework2.md`, expected `.coco.r2.md`. Two corrective iterations used; no polling/reset. After callback independently validate then Nono follow-up. Still NOT accepted, no commit/push/user-corpus/pressure/Windows claims.

## Junior — Qt stop/resume r1 verified; independent review queued (2026-10-30)

Mission `ZM-ZEGRID-M6-QT-STOP-RESUME-20261030`, same HEAD87e98a46. Coco recovery completed; all14reported hashes match, diffcheckclean, expected17dirty files. Junior **160passed,35warnings in115.09s** (Qt policy/stop-resume/config/scheduler/GPU-toggle + M4 + CLI), includes actual tiny new-process resume plus transparent dispatcher pending-only execution on copied store. NOT accepted: independent probes confirm resume branch bypasses foreign-output preflight, and Save-and-close during pending ZeGrid spawn loses request (deferredFalse/armedFalse/unavailable). Engine NaN-gauge degradation remains unresolved: publishingFalse but store-backed results selected on store existence; source checked, routine failure not independently replayed.

Nono read-only review0 accepted/queued `cb15670f-bf24-48bc-a65c-88a9d3b83615`. Evidence `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.junior.r1.md`, review0-brief.md and review0-snapshot.json. Expected `.nono.r0.md`. Review consolidates confirmed R1/R2 + ack/finish races/output ownership and minimum safe engine handling before second Coco correction. One corrective iteration used (runtime recovery same iteration); keep Coco context/no reset. No commit/push, no user-corpus/pressure/Windows tests. Await callback, no polling.

## Junior — REWORK-1 runtime interruption recovery (2026-10-30)

Mission `ZM-ZEGRID-M6-QT-STOP-RESUME-20261030`. Explicit sessions_send delivery failure: session rebound. Authorized recovery: expected coco.r1 report ABSENT; one history read proves ebeaeb13 DID receive/rework, then terminal provider failure `Auth profile "deepseek:manual" is temporarily unavailable for deepseek/deepseek-v4-flash.` Last test observer failed ResumeIncompatible (test direct-call science mismatch, not a license to weaken identity). Partial edits preserved incl new run_publication.py + zegrid_cli.py changes; HEAD87e98a46 unchanged. No validation/acceptance of partial tree.

ONE same-session continuation, no reset/model/transport change: accepted/queued `444986fe-63b2-49e3-9945-8dc81f692022`; completion NOT yet proved. Brief `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.recovery1.md`; new unique report expected `.coco.r1-recovery1.md`, phase still rework-1 (same corrective iteration). Finish F1-F4+triage, real dispatch observer proof, CLI safety regressions; preserve all prior changes. Await callback, no polling/retry loop. If same provider failure recurs, preserve/report runtime blocker rather than silently switch provider.

## Junior — Qt stop/resume REWORK-1 queued (2026-10-30)

Mission `ZM-ZEGRID-M6-QT-STOP-RESUME-20261030`, HEAD87e98a46 unchanged. Coco r0 DONE inspected; Junior **87passed,10warnings in59.06s** incl synthetic spawned stop/new-process resume; diffcheck clean and expected15dirty files incl prior accepted baseline. NOT accepted: closeEvent blocks GUI20sec then auto-kills (ack-before-exit also legacy kill); invalid store preflight computes but ignores reason without candidates (own3probes confirmed); foreign/publication output guards missing (own foreign-file probe admitted launch); integration resume kwargs bypass GUI selection and no execution-level no-recompute proof. Routine12x384 durable failure reported by Coco remains untriaged, not proved preexisting.

Same Coco context, first correction accepted/queued `ebeaeb13-2df5-40b2-a605-f0700f895fd3`. Durable findings+brief `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.{junior.r0,rework1}.md`; await `.coco.r1.md` exact-source callback. Scope async safe close/no auto-kill, complete fail-closed preflight/data preservation, real GUI second-launch+execution proof and legacy regression, exact routine failure triage. No Nono yet (confirmed implementation defects). No commit/push; no real-corpus/physical pressure/Windows qualification. No polling/reset between corrections.

## Junior — Qt Stop-and-Save + Resume implementation launched (2026-10-30)

Mission `ZM-ZEGRID-M6-QT-STOP-RESUME-20261030`. User explicitly expands GUI integration: existing Stop and Save for ZeGrid only, preserve other modes, offer restart/crash resume from recognized incomplete checkpoints. Previous Qt-six-policy ACCEPT remains uncommitted; HEAD87e98a46 unchanged. Baseline snapshot in workspace .a2a-reports. Junior fresh targeted baseline: **69 passed,18warnings in14.06s** (GUI policy + M4 resume).

Coco reset (counters0) then async accepted/queued `5a744772-80f9-429f-a0e0-f5fb4cf9a3f0`. Brief/report `.a2a-reports/ZM-ZEGRID-M6-QT-STOP-RESUME-20261030.{brief,coco.r0}.md`. Scope: explicit opt-in durable GUI checkpoints (defaultOFF), spawn-safe cooperative stop with ack/no automatic4sec kill, frozen-mode routing, honest status/close lifecycle, explicit valid incomplete-run selection and refusal/no-overwrite protections, real bounded synthetic spawn stop/new-process resume proof, legacy regression checks. No worker commits or pushes. Wait exact-source callback, then Junior independent checks and Nono review; no polling.

User real-corpus Linux qualification remains user-owned; synthetic integration authorized. M3 pressure and Windows/NTFS remain unqualified; no RAM guard/science/global default changes. Existing GUI SIGTERM+4sec termination is inadequate for cooperative save and must remain confined to legacy behavior.

## Junior — M6 Qt six-key wiring ACCEPTED locally (2026-10-29)

Mission `ZM-ZEGRID-M6-QT-POLICY-20261029` terminal ACCEPT after two corrections. Nono review0 ACCEPT-WITH-CONDITIONS: C2 already satisfied (None/default vs literalauto docs+roundtrip); C1 now closed by persistent real-loader/recovery/migration/QTest/save regression. Junior final **112 passed, 1 deselected in11.15s**; earlier **221passed** includes real tiny GUI->dispatcher->engine first-run. Counts overlap; do not sum. Product11hashes unchanged since review; only test changed in r2, plus Junior STATE. Actual-dispatch kwargs signature assertion strengthened.

Accepted delta remains UNCOMMITTED/UNPUSHED in `/home/tristan/.openclaw/workspace/worktrees/zemosaic-resilient-run-resume`, branch `feat/zegrid-resilient-run-resume`, HEAD`87e98a46b60afc201999eb0db1e9c0057c05ba3d`. Expected13files, diffcheckclean. No merge or activation; main projects checkout untouched. Full fast tier not green (one preexisting AST defect proven; other broadF/E untriaged), so this is scoped Qt acceptance only. Durable acceptance `.a2a-reports/ZM-ZEGRID-M6-QT-POLICY-20261029.junior.acceptance.md`.

No active delegation remains. Next: Tristan's own full non-seamed Linux restart/resume, NOT_RUN here. M3physical/WindowsNTFS remain open. GUI Stop-and-Save/resume selector not included. Usage: `docs/refactor/ZEGRID_M6_QT_POLICY_USAGE_20261029.md`.

## Junior — Nono review0 conditions, test-only REWORK2 (2026-10-29)

Mission `ZM-ZEGRID-M6-QT-POLICY-20261029`: Nono ACCEPT-WITH-CONDITIONS; C1 persistent real-loader/recovery/unrelated-migration test genuinely missing, C2 None-vs-auto documentation/roundtrip ALREADY satisfied in usage doc18-23 and parameterized tests. Implementation hashes unchanged; only Junior STATE differs. Prior221passes and own real-file QTest probe remain valid. No new product defect found.

Same Coco/no reset TEST-ONLY REWORK2 accepted/queued `63675e85-ed40-43b9-8bf7-ec6a7dbc7f1a`: port real-file/QTest probe into persistent suite, correct inaccurate r1 evidence claim, strengthen actual-dispatch signature test proportionately. No product/locale/doc/default changes; no full suite or user-owned Linuxrestart qualification. Expected `.a2a-reports/ZM-ZEGRID-M6-QT-POLICY-20261029.coco.r2.md`. Two corrections used (of3); no polling. Then independently replay added tests/check hashes and close conditions. Not accepted/committed/pushed yet.

## Junior — M6 Qt r1 verified, Nono review-0 queued (2026-10-29)

Mission `ZM-ZEGRID-M6-QT-POLICY-20261029`, HEAD87e98a46 unchanged. Independent replay **199+22 = 221 passed** (15.57s +75.38s), including actual tiny GUI-built invocation -> dispatcher -> engine producing2x2 despite disk1x1 and existing CLI/stop regressions. F1 explicit auto and F2 persisted-type coercion closed on tests/probes. Own real-loader/QTest/file-save probe passes (raw invalid values preserved, unrelated backend/GPU migrations retained, explicit edits accepted). Twelve implementation hashes match report; diffcheck clean.

Coco report discrepancy: claimed persistent real strict-loader recovery+unrelated migration test is absent; Junior probe proves it once, not suite coverage. Generic bind_partial test also weak (prefilters keys). Sent to Nono for focused read-only review alongside code. Nono accepted/queued `b79e9ddc-b256-41a2-a7bd-da2d9a52dd7d`; expected `.a2a-reports/ZM-ZEGRID-M6-QT-POLICY-20261029.nono.r0.md`, exact-source callback. One Coco correction used; preserve same context, no polling. Broad fast tier NOT green (one AST failure proven baseline, others untriaged). No commit/push or acceptance yet. Complete non-seamed Linux restart/resume remains user-owned/NOT_RUN; M3physical/NTFS also open.

## Junior — M6 Qt REWORK-1 queued (2026-10-29)

Mission `ZM-ZEGRID-M6-QT-POLICY-20261029`, HEAD `87e98a46` unchanged. Junior targeted replay **177 passed, 1 deselected, 23 warnings in 17.49s**, but independent Qt probes confirm F1 literal valid gauge `auto` absent from combo (false invalid warning) and F2 strict-invalid persisted numeric strings silently converted into accepted numeric values. Scope corrections plus true null-clearing-to-engine and generic invocation-contract tests. Actual engine gauge default is `auto` (initial brief parenthetical global was stale); preserve None and literal auto separately, do not change science.

Same Coco context, no reset: REWORK-1 accepted/queued `4004c040-0e6b-403a-ba20-936e20d238ae`; expected `.a2a-reports/ZM-ZEGRID-M6-QT-POLICY-20261029.coco.r1.md`. Junior evidence and brief are siblings. On callback: independent checks, then Nono. One of three corrections used; not accepted, no commit/push. Broad-tier AST failure reproduced on identical HEAD source, other reported F/E/timeouts untriaged; no full fast-tier-green claim. Complete non-seamed Linux restart/resume remains user-owned/NOT_RUN.

## Junior — M6 Qt policy wiring in flight (2026-10-29)

Mission `ZM-ZEGRID-M6-QT-POLICY-20261029`, baseline `87e98a46b60afc201999eb0db1e9c0057c05ba3d`, clean feature worktree before delegation. User explicitly requests the six Qt keys and reserves the complete non-seamed Linux restart/resume for his own later run.

- Junior baseline: config-policy + scheduler Qt offscreen **33 passed in 4.40s**.
- Coco reset confirmed counters zero; async dispatch accepted/queued `ffea56d9-7a18-4d2e-8b67-62bfc3fb6835`.
- Scope: Advanced controls for recovery enabled/deadline, durable budget, gauge mode, layout and SIP; exact defaults unchanged; typed round-trip and actual GUI -> worker -> engine propagation; focused regressions and bounded first-run smoke only. No restart/resume qualification, pressure, global config changes or worker commit/push.
- Durable brief/report under workspace `.a2a-reports/ZM-ZEGRID-M6-QT-POLICY-20261029.{brief,coco.r0}.md`. Await exact-source callback, independently verify, then Nono review. No polling; retain Coco context through any rework.
- Full Linux restart/resume remains USER-OWNED/NOT_RUN; M3 physical and Windows/NTFS remain NOT_RUN. Qt Stop-and-Save/resume selector are not implied by the six policy fields.

## Junior — M6 ACCEPTED and published (2026-10-29)

Mission `ZM-ZEGRID-M6-PRODUCT-SEAM-20261029`, base `69409fcb14d1fd13111b541a1fea15fad0630141`. Nono review-1 ACCEPT (C1 closed, C2/C3/C4 intact) after Junior re-probed the exact data-loss case on the r2 tree.

Delivered, defaults unchanged (recovery `False`, deadline `None`, store budget `None`, resume `never`):
- six run-policy keys declared in `DEFAULT_CONFIG` with strict `validate_zegrid_run_policy` called once at `run_zegrid_mode` entry — no coercion, legacy strings refused explicitly (architect decision: strict, no 0/1 migration, because a truthy string could silently enable recovery);
- `python -m zemosaic.zegrid_cli`: real headless entry, no Qt import, exit 0/2/3, fail-closed option refusals, disk-verified publication, optional `--overwrite` recorded in the summary and refused with `--resume-mode auto` or foreign content, pre-run rejection of an existing publication in the output folder (root, `.zegrid_publications`, `grid_EQ`/`grid_ALTZ` children);
- cooperative `SIGTERM`/`SIGINT`: sets the channel, finishes the safe boundary, returns `STOPPED_RESUMABLE` with intact atomic commits and no publication; bounded grace watchdog; Windows signals NOT_RUN;
- `format_bounded_assembly_close` reads the real `peak_live_*` counters, so the close warning after a successful close is genuinely fixed.

Evidence: **45 passed** (config-policy+close-stats, CLI incl. overwrite/freshness, cooperative stop). Junior's own runs: fresh output exit 0 (`6848e939…`/`fa7a79d0…`), second run exit 2 with byte-identical prior files, `--overwrite` exit 0 with `overwrite=true`, foreign content exit 2, `--overwrite`+resume exit 2.

Still NOT_RUN / out of scope: Qt settings wiring for the six keys, Windows signal semantics, per-cell bounded-assembly close integration (PLAN budget on the tiny corpus), GPU, M31, large corpus. No activation by default, no merge to `beta`/`main`.

## Junior — M6 r0 delivered and verified, Nono review-0 in flight (2026-10-29)

Mission `ZM-ZEGRID-M6-PRODUCT-SEAM-20261029`, HEAD `69409fcb14d1fd13111b541a1fea15fad0630141`. Product diff: `zemosaic_config.py +151` (six declared keys, strict `validate_zegrid_run_policy`), `zemosaic_zegrid_mode.py +92/-12` (validation at entry, optional `stop_channel`, durable-vs-in-memory stop classification, close-log formatter), new `src/zemosaic/zegrid_cli.py`, four test files, `tools/zegrid_m6/make_tiny_corpus.py`. No commit/push.

Defaults unchanged: recovery False, deadline None, store budget None, gauge/layout None, sip keep; `zegrid_resume_mode` still never. Junior replayed **38 passed in 25.56 s** and ran the CLI itself: corpus generator + `python -m zemosaic.zegrid_cli` -> exit 0, `COMPLETE in 6.7 s`, FITS+coverage+manifest+aesthetic, science `6848e939...`, coverage `f43ffd94...` identical to Coco r0 (reproducible).

Pieces now real: a headless entry that does not import Qt and refuses contradictory options; a cooperative stop that reaches a safe boundary and reports `STOPPED_RESUMABLE` with intact atomic commits and no publication; a close log that reads the real counters.

Open questions sent to review: whether the new validation could reject a legitimate legacy persisted config, and whether the CLI could accept a stale manifest in a reused output folder. Nono review-0 accepted/queued `cd0e54ac-1480-45cc-8f5a-b30086840614`. Limits unchanged: GUI wiring, Windows signals, per-cell close integration (PLAN budget), GPU/M31, large corpus all NOT_RUN. No activation, no merge.

## Junior — M3 bricks ACCEPTED and published; host-capacity decision pending (2026-10-29)

Nono review-0 = ACCEPT-WITH-CONDITIONS. Condition C1 (prove the six runner invariants before any pressure) is closed by Junior line-read plus the focused tests: single owned child spawned once, 64 MiB bounded mmap steps, hard 512 MiB whole-process cap, fixed `GUARD_ORDER` (pressure RSS/reserve -> wait exit -> cooperative stop -> owned kill classified FAIL), one-shot unix-socket IPC with a <=2 s ack deadline and timeout => release + NOT_RUN, cleanup only `os.kill` on the owned PID (no killpg/pkill/killall), no cgroup/sysctl/swap/cache manipulation, experiment mode refusing before any child/socket/mmap and reporting `REVIEWED_FLOW_NOT_EXECUTED_IN_THIS_MISSION` even when granted. `30 passed` replayed.

Published on the feature branch as `635be6e261fd4520f138a82a815903ef9bedfbc5` (parent `7d0b759`); `origin/beta=231b597867` and `origin/main=434961a8ec` unchanged.

Measured brick: bounded reference 1600x1600/2x2 PASS in 138.8 s, peak product tree 1,139,167,232 B, min MemAvailable 2,013,675,520 B, `X_b = 1,133,114,368 B` (agrees across `r55_scheduler.execution.aggregate_bytes` and `per_cell_diagnostics.ram_contract.aggregate_bytes`). Artifacts live outside any checkout at `/home/tristan/.openclaw/workspace/.m3-linux-runner-20261029/`.

Blocking measurement: preflight with `X_b` is `NOT_READY` with `NO_SAFE_PRESSURE_DOSE` and `PRESSURE_CAP_INSUFFICIENT`; refusing that minimum needs a ~1.95 GB dose against a 512 MiB cap, and the corpus that would land in the refusable window (~2300 px) needs 1.5-1.8 GB of tree plus the 1 GiB reserve against 2.7-3.0 GB live. This 7.7 GiB host cannot host the experiment under the agreed contract. Physical pressure remains NOT_RUN; no threshold, reserve, cap or science value was lowered. Awaiting an explicit host-capacity decision (larger host, explicitly reviewed alternative strategy, or defer). No activation, no M6, no beta/main merge.

## Coco — M3 runner/X_b continuation r1: 1600² measured, NOT_READY (host-fit) (2026-10-29)

No physical pressure. Artifacts moved out of the product checkout to `/home/tristan/.openclaw/workspace/.m3-linux-runner-20261029/`; `projects/zemosaic` is clean of the run directory. Runner now defaults to an artifact base outside any git checkout and refuses a root inside one without an explicit `--allow-inside-git`.

Measured no-pressure bounded reference (1600² frames, dither ×2, grid 2×2, halo 64, fresh ext4 roots): PASS 138.8 s, 4/4 cells, 4 commits, 4 waves, peak tree RSS 1,139,167,232 B, min MemAvailable 2,013,675,520 B, **`X_b = 1,133,114,368 B`** (`r55_scheduler.execution.aggregate_bytes`). Dense at 1024² remains 1,034,015,744 B; the 1024² bounded minimum was 960,876,544 B.

With `X_b = 1,133,114,368`, `required_raw = 1,259,015,965 B >` the 1 GiB reserve (condition 4 now holds) but is still below `A-P = 2,668,462,080 B`, so the bounded retry would fit after pressure: preflight is **NOT_READY** (`NO_SAFE_PRESSURE_DOSE`, `PRESSURE_CAP_INSUFFICIENT`). Reaching the M3 wait on this host needs ≈2.2–2.5 GB aggregate (~2300 px canvas), where product tree + 1 GiB reserve ≈ live `A` with no margin — the corpus that makes M3 reachable likely does not fit this 7.7 GiB host without weakening the reserve or the science. Pressure experiment still NOT_RUN.

## Junior — M3 preflight+protocol ACCEPTED and published; runner/X_b mission running (2026-10-29)

Nono review-1 ACCEPT (F4 closed on the protocol text; F1/F2/F3 untouched and conforming). Junior had independently replayed **26 passed (2.11 s)** and confirmed all four protocol/tool/test hashes unchanged before acceptance. Published on the feature branch as `7d0b7596ac6425f99c1787379a8d40092ed3fb38`; `origin/beta=231b597867` and `origin/main=434961a8ec` unchanged.

Accepted scope is PREPARATION ONLY: read-only fail-closed feasibility tool (strict units/ranges, ancestor cgroup walk, reserve >=1 GiB, pressure/cap <=512 MiB rejected in CLI, owned exclusive JSON) plus the reviewable protocol (hook before the original sampler `:11681`, strict attribution and exclusions, one-shot bounded IPC, no fact rewriting, pre-arm latch disqualified, barrier trace required for PASS). No physical pressure, no product run, no runner at acceptance time, no GUI/CLI claim.

Next mission `ZM-ZEGRID-M3-LINUX-RUNNER-20261029` dispatched in a fresh Coco context: implement the transparent exec/wait/commit/barrier observers and the owned pressure runner (experiment mode refusing by default), then measure the real bounded minimum `X_b` with a no-pressure dense+bounded reference on real ext4 and re-run the preflight with `X_b`. Accepted/queued `bbe497b3-14a6-4087-a3b6-5172135950be`; report expected `.a2a-reports/ZM-ZEGRID-M3-LINUX-RUNNER-20261029.coco.r0.md`. Host about 2.7 GiB available after the user freed memory — every observation volatile, stop rather than force.

Still unqualified: physical M3 pressure, full FITS product, Windows/NTFS, BLAS/deep contributors, GPU, M31, already-bounded/repeated-pressure product path. No default activation. HEAD `7d0b759` is the accepted baseline until the next milestone lands.

## Junior — M3 protocol r2 verified, Nono review-1 in flight (2026-10-29)

REWORK-2 continuation finished the interrupted 01:40 work: candidate-wave/exactly-once wording, latched-fact rule, barrier-trace fields, 5 protocol-consistency tests, fresh preflight, report. Junior independently replayed **26 passed (2.11s)** and confirmed all five file hashes match; preflight tool and its 21 tests are byte-identical to r1, so F1/F2/F3 stay conforming and untouched.

Protocol now states the correct order for the designated safe wave: hook -> original unmodified `_r55_run_checkpoint_sample` (:11681) -> original unmodified `_r55_run_checkpoint` (:11712/:11779), with explicit exclusion of heartbeat (:11576), M3 coordinator probe (:11405), post-wait resampler (:11919) and GPU sites (:12705/:13036); one-shot bounded IPC (<=2 s proposal) owned by the parent observer, timeout/refusal => release + NOT_RUN; no fabrication/rewrite/clamp; pre-arm latched fact disqualified unless `latched_host.monotonic_ns >= pressure_applied`; PASS requires barrier trace `hook < pressure_applied <= sampler` plus post-arm governing fact.

Fresh real preflight `NOT_READY` at 2697.0 MiB available, sole blocker `BOUNDED_AGGREGATE_NOT_MEASURED` (512 MiB dose clears the reserve at that snapshot). Volatile observation, not an M3 pass.

Nono review-1 accepted/queued `e0c64fe3-cc91-4eb2-9156-e6b574f73a80`; report expected `.a2a-reports/ZM-ZEGRID-M3-LINUX-PREFLIGHT-20261028.nono.r1.md`. HEAD `a18dd12` unchanged, no production change, no pressure, no commit/push. Still missing after acceptance: transparent execution/wait/commit observers + one-shot pressure runner, real bounded-minimum measurement, then the reviewed experiment. Physical M3, FITS, Windows/NTFS, BLAS, GPU and M31 remain unqualified.

## Current status — M3 preflight REWORK-2 delivered (2026-10-29), NOT an M3 pass

Protocol-only F4 correction completed; preflight F1/F2/F3 unchanged and conforming.
The pressure handshake now runs **before** the original safe-wave
`_r55_run_checkpoint_sample(disk_root)` (`mode.py:11681`), not merely before
`_r55_run_checkpoint`: barrier hook -> original sampler (unmodified) -> original
checkpoint with those facts. It is attributed to the exact safe-wave call site,
never heartbeat/M3-recovery/M3-post-wait/GPU samplers, arms exactly once only
after K durable commits, uses one-shot bounded parent-owned IPC with explicit
ack/timeout => NOT_RUN, never fabricates/rewrites facts, and documents the
residual race. PASS requires `hook < pressure_applied <= sampler` for the same
wave, no pre-arm latched fact governing, and zero post-release execution of K.

Validation: **26 passed** (21 existing preflight + 5 protocol-contract tests),
`py_compile` OK, `git diff --check` OK. Fresh read-only preflight (unique JSON
`preflight.rework2.20261029T072352Z.json`): **NOT_READY**, exit 2, sole blocker
`BOUNDED_AGGREGATE_NOT_MEASURED`; `MemAvailable` 2,697.0 MiB at snapshot,
`A-512 MiB` = 2,185.0 MiB (reserve passes), ext4 `/` 22,273.6 MiB free, cgroup
chain known-unlimited with ancestor evidence. This is one volatile snapshot, not
a universal host fact or reservation and not an M3 qualification. No pressure,
product run, production change, commit or push. `X_b` remains unmeasured;
execution/wait/commit observers and the runner remain unimplemented;
already-bounded/repeated-pressure remains a separate product gap; namespace
scope remains ENGINE-only. Next: measure `X_b` at the pressure-governing raw
with a no-pressure bounded reference, then re-run preflight with
`--bounded-aggregate-bytes`. Historical Junior/Nono and prior status below are
preserved.

---

## Junior — M3 protocol REWORK-2 in flight (2026-10-29)

Nono review-0 = FINDINGS: preflight tool conforming (F1/F2/F3 closed), one HIGH confirmed — the protocol's pressure barrier runs after `_r55_run_checkpoint_sample` (mode.py:11681) so it cannot govern the current wave. Junior had reached the same conclusion independently. Bounded protocol-only correction dispatched to the same Coco context (rework-2, 2/3), accepted/queued `880504e9-1a93-4d87-9fc3-f2c446905cf9`; report expected `.a2a-reports/ZM-ZEGRID-M3-LINUX-PREFLIGHT-20261028.coco.r2.md`.

Held constant: HEAD `a18dd12e4cc2775fc397d5bcfe9c334bbaffb205`, four tools/tests/docs files, no production change, no pressure, no runner, no commit/push. Real preflight still `NOT_READY` (1386.3 MiB at last snapshot: 512 MiB dose breaches the 1 GiB reserve by 149.7 MiB, and the bounded minimum is unmeasured). Preflight tests 21 passed. Nothing accepted or published in this mission yet; M3 physical pressure, Windows/NTFS, full FITS, BLAS, GPU and M31 remain unqualified.

## Junior — M3 preflight r1 verified, independent review pending (2026-10-28)

21testsPASS7.80s. Independentfixtures confirm inheritedcgroup/capfloor/telemetryfalseREADY fixed; CLIweakreserve exits3. RealpreflightNOT_READY at1386.3MiB available: selected512MiB breachesreserve and boundedminimumunknown. No pressure/productrun. Residual protocol issue: proposed handshake wraps checkpoint AFTER facts already sampled; currentwave may use prepressurefacts. Nono review to verify and scope correction. Evidence `.a2a-reports/ZM-ZEGRID-M3-LINUX-PREFLIGHT-20261028.junior.r1.md` and probes/preflight-r1. No acceptance/commit/push, HEADa18dd12 unchanged.

## Junior status — M3 preflight REWORK-1 required (2026-10-28)

12tests pass but independent probes confirm false READY_TO_PLAN: inheritedcgroup limit ignored, reserve/pressure policy bypass through parameters, inconsistentRAMfacts accepted. Protocol also needs mandatory actual-compute evidence for zero-recompute, not just unchangedcommitIDs. No physicalpressure/productrun executed. ActualJuniorpreflight1829.3MiB available, reserve arithmeticfits512MiB but boundedminimumunknown => NOT_READY; no demonstrated need to changehost. Same-contextboundedrework next; details `.a2a-reports/ZM-ZEGRID-M3-LINUX-PREFLIGHT-20261028.junior.r0.md`. HEADa18dd12 remains baseline; RSS accepted/published unchanged.

## Current status — M3 preflight REWORK-1 delivered (2026-10-28), NOT an M3 pass

Mission `ZM-ZEGRID-M3-LINUX-PREFLIGHT-20261028` rework-1 fixed the four
Junior-confirmed false-readiness/proof gaps without pressure, product run,
production change, commit or push. Tool/tests/protocol only; accepted RSS
sources/tests untouched.

- **F1 cgroup:** preflight now walks leaf→mount-root ancestors, takes the minimum
  finite `memory.max`, records ancestor paths/limits, accepts a missing
  `memory.max` only at the mount root, and fails closed on any other
  missing/unreadable/malformed ancestor. Unknown is never unlimited.
- **F2 contract:** public CLI/evaluator enforce reserve ≥1 GiB and selected
  dose/cap ≤512 MiB; weaker overrides exit 3 / NOT_READY. Only stricter values
  are accepted. A safe smaller dose window is reported; 512 MiB is a maximum,
  not a mandatory dose.
- **F3 telemetry:** required fields reject unknown units, non-positive
  `MemTotal`, `MemAvailable > MemTotal`, and `SwapFree > SwapTotal`; inconsistent
  facts cannot yield READY.
- **F4 protocol:** zero compute of committed K is now mandatory, proven by a
  test-only execution trace around the real `_stack_cell`/`_prepare`/`_consume`
  path, a live `RecoveryCoordinator` journal trace, and a durable commit trace.
  No PASS without them. Polling-until-K is replaced by a bounded test-only
  handshake at the real `_r55_run_checkpoint` safe boundary. Already-bounded or
  fallback-consumed pressure remains a separate unqualified product gap.
  Env/cmdline capture removed; process fields allowlisted; 512 MiB includes
  process overhead; sampling-latency caveat and product-tree ceiling added.

Validation: **21 passed** focused tests; `py_compile` OK; `git diff --check` OK.
Fresh read-only preflight (unique JSON
`preflight.rework1.final.20261028T221958Z.json`): **NOT_READY**, exit 2, sole
host blocker `BOUNDED_AGGREGATE_NOT_MEASURED`; `MemAvailable` 1,678.2 MiB,
`A-512 MiB` = 1,166.2 MiB (reserve passes at this snapshot), ext4 `/` 22,233.6
MiB free, cgroup chain known-unlimited with ancestor evidence. Other snapshots
(0.8 GiB, 1.47 GiB, Junior 1,829.3 MiB) differ and are preserved as observations,
not universal facts or reservations. No host move is mandated by the 512 MiB
maximum; the next step is a no-pressure bounded reference to measure `X_b` at the
pressure-governing raw, then re-run preflight with `--bounded-aggregate-bytes`.
Historical RSS acceptance below is retained.

---

## Current status — RSS Linux ACCEPTED by Junior (2026-10-28)

Nono review-1 ACCEPT; F1/F2/F3 + parity-seed stop defect closed. Junior verified exact4file snapshot, **79passed/0skip/0fail**, real guarded matrix **PASS85.495s**. Cases512²/16cells,1024²/16cells,1024²/64cells; only256² dense parity (8planes+offsets) proven. Peak lifetime sampledRSS497,184,768B (**474.15MiB**, not497MiB), minimumhost1,283,166,208B, cap2, immutable storecontent,0live mappings afterclose. Five independent failed-seed cases stop beforeparity. No production/science/schema/default changes. Two correctiveiterations used.

Scope: synthetic producer, real durablestore/guardedconsumer, per_cell only,20ms sampled observations; parenthash warms pagecache. NOT a universal RSS bound. FullFITS/restart, physicalM3, Windows/NTFS, hugecanvas/BLAS/deepcontributors, GPU/M31 still NOT_RUN. No default activation/M6. Evidence in agent workspace `.a2a-reports/ZM-ZEGRID-RSS-LINUX-20261028.acceptance.md`, `.junior.r2.md`, `.nono.r1.md`; raw artefacts `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028-junior-r2/run-20261028T215419Z/`.

Next: prepare a bounded real-pressure M3 Linux protocol/preflight; no pressure workload until the protocol/guards have been reviewed.

---
## Historical review progression (superseded by acceptance above)

## Current Junior status — RSS REWORK-2 verified, Nono review pending (2026-10-28)

Independent **79 passed**, 0skip/0fail (83.25s). Real guarded Linux matrix **PASS** (85.495s):512²/16cells,1024²/16cells,1024²/64cells; separate256² dense parity8planes+offsets equal. Peak lifetime sampledRSS497,184,768B; minimum observed host reserve1,283,166,208B; max live cells2, store content unchanged, live mappings0 after close. Independent five failure probes stop before parity; success control launches. Evidence `.a2a-reports/ZM-ZEGRID-RSS-LINUX-20261028.junior.r2.md`; artefacts `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028-junior-r2/run-20261028T215419Z/`. Nono review required before acceptance. Synthetic scope only, NOT physicalM3/FITSfull/Windows/BLAS/GPU/M31; no activation/commit/push. HEAD8564070 unchanged.

### Previous orchestration status (superseded)

## Junior recovery — RSS REWORK-2 pending (2026-10-28)

Independent replay: **71 passed**, 55.77s. REWORK-1 not accepted: parity seed safety-stop still launches parity child before validating seed result (mocked return137/HOST_RESERVE). Final FAIL preserved, but stop invariant violated. Evidence: `.a2a-reports/ZM-ZEGRID-RSS-LINUX-20261028.junior.r1.md` and `.junior-probes-r1.json` in agent workspace. Bounded second correction; no production change or qualification expansion. Junior matrix replay deferred until guard repaired. HEAD8564070 unchanged.

## Current status — M5-B RSS Linux REWORK-2 — implemented, NOT accepted (2026-10-28)

Mission `ZM-ZEGRID-RSS-LINUX-20261028`, HEAD `8564070bf534d90037cbeb3d420303d9b86da264`, branche `feat/zegrid-resilient-run-resume`. Aucun commit/push/produit/science/store-schema/default. REWORK-2 corrige le défaut résiduel F1 confirmé par Junior : la branche parité lançait l'enfant `parity` avant de valider le résultat/processus du seed de parité.

Correction bornée : le seed de parité est maintenant validé immédiatement (timeout, guard breach, returncode non nul, JSON absent/invalide ou statut ≠ `SEEDED` ⇒ **FAIL**, aucun lancement de `parity`, preuve du seed conservée, un seul rapport parité, FAIL collant). Les flags de processus sont validés, pas seulement le returncode, pour le seed ET le résultat `parity` (exit 0 n'efface pas un timeout/guard enregistré). Le deadline global est revérifié après le seed ; épuisement honnête sans échec enfant ⇒ **NOT_RUN/PARTIAL**, jamais un FAIL écrasé. La précédence failure-before-refusal est conservée. Tests d'injection orchestrateur ajoutés (seed parité nonzero/timeout/guard/JSON manquant-invalide ⇒ pas d'appel `parity` ; chemin succès ⇒ `parity` lancé ; deadline épuisée ⇒ NOT_RUN sans lancement).

Validation finale : **42 passed** (harnais RSS), **79 passed** (harnais + M5-A), **39 passed** (M5-B publication/execution), `git diff --check` OK, exécutable `/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python` (3.13.5). Matrice réelle corrigée sur base neuve `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028-r2`, statut **PASS**, 47.348 s, disque 107,449,414 B, MemAvailable départ 1,999,347,712 B :

| cas | canvas/grille | modèle requis B | pic vie B | pic assemblage B | post-close RSS B | HWM B | échantillons RSS/hôte/assemblage | cellule pic | mapped pic B | elapsed s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c512-g4x4 | 512²/4×4 | 38,292,584 | 408,424,448 | 408,424,448 | 402,423,808 | 412,143,616 | 185/186/19 | 2 | 2,065,920 | 0.387 |
| c1024-g4x4 | 1024²/4×4 | 148,147,304 | 487,436,288 | 487,436,288 | 460,591,104 | 503,554,048 | 214/215/40 | 2 | 7,923,200 | 0.820 |
| c1024-g8x8 | 1024²/8×8 | 133,720,632 | 494,325,760 | 494,325,760 | 456,863,744 | 498,294,784 | 275/276/81 | 2 | 2,065,920 | 1.644 |

Chaque cas : admission réelle `ram_contract.admit_budget_bytes`, cap=2, enveloppe live armée, offsets+feather per_cell, commits et hash de CONTENU du store identiques avant/après, timelines ordonnées, arrays valides après close, `live_cells==0`. Parité dense **uniquement 256×256/4×4** : `PARITY_PASS`, `parity_launched=true`, seed et parity rc0, huit plans et offsets identiques. Les digests 512/1024 sont des sorties observées, pas une preuve d'égalité dense.

Preuves : `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028-r2/run-20261028T214814Z/` (`matrix.json`, `summary.json`, `measure.json`, `parity-256-g4x4/parity.json`, logs). Hashes : rss_bench `18e460ba…`, test RSS `c2e57ce3…`, rss_probe `9543ad3f…`. R0/R1 conservés intacts.

NOT_RUN / non qualifié : FITS complet, M3 pression physique, Windows/NTFS, BLAS-scale, GPU/M31, jauge globale mesurée. M5-B RSS **non accepté** avant vérification Junior + Nono. Aucune activation/M6.

---
## Historique — M5-B RSS Linux REWORK-1 (corrigé ensuite par REWORK-2)

## Current status — M5-B RSS Linux REWORK-1 — implemented, NOT accepted (2026-10-28)

Mission `ZM-ZEGRID-RSS-LINUX-20261028`, HEAD `8564070bf534d90037cbeb3d420303d9b86da264`, branche `feat/zegrid-resilient-run-resume`. Aucun commit/push, aucun changement produit/science/store. REWORK-1 corrige F1/F2/F3 de la revue Junior+Nono ; les nombres r0 restent des observations provisoires et sont remplacés par ceux ci-dessous.

**F1 (classification/précédence).** `cmd_measure` ne produit `REFUSED` que pour un `AssemblyCapacityError` typé, avant tout chargement/vérification/mapping, avec bloc `refusal` explicite (type, phase `preflight`, compteurs zéro, plan) ; une exception générique ou un refus après progrès écrit `ERROR`, traceback et code enfant non nul (3/4). `classify_measurement` applique d'abord timeout/guard/returncode (FAIL), ne convertit un REFUSED valide en NOT_RUN qu'après validation séparée, et refuse les REFUSED incomplets/spoofés. La matrice conserve un FAIL (plus de FAIL→PARTIAL), écrit exactement un enregistrement par cas, et ne lance pas la parité après un arrêt de sécurité/échec.

**F2 (preuve de surveillance).** Le parent compte des échantillons RSS/hôte réellement valides (`valid_rss_samples`, `valid_host_samples`, `valid_paired_samples`, `valid_assembly_samples`), exige ≥2 échantillons appariés et ≥2 dans la fenêtre d'assemblage, un pic >0, une réserve hôte valide et une fenêtre d'assemblage observée. Télémesure inconnue d'un enfant vivant ⇒ abandon explicite après une courte tolérance de course de sortie ; sortie normale non faussement échouée ; absence de preuve ⇒ NOT_RUN, jamais PASS. Les checks nommés (`REQUIRED_CHECKS`) ne peuvent être vides/partiels ; `assert_real_disk` refuse explicitement un fstype inconnu.

**F3 (temporalité/immuabilité).** L'enfant prend ses snapshots de commits/métadonnées avant l'assemblage et après la fermeture ; `checks.store_metadata_unchanged` est dérivé. Le parent streame le hash de CONTENU de tous les fichiers du store avant spawn et après sortie (hors RSS mesuré ; cache de pages réchauffé et documenté), et `classify_measurement` exige l'égalité. `post_close_current_rss_bytes` est capturé après `access.close()+gc.collect()`. Pic échantillonné de durée de vie et pic de fenêtre d'assemblage sont distincts. Timeline ordonnée vérifiée ; test d'injection qui modifie des octets de payload détecté.

Matrice corrigée (enfant réel, store réel, admission réelle ; base `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028-r1`), statut **PASS**, 48.714 s, disque 107,435,622 B :

| cas | canvas/grille | modèle requis B | pic échantillonné vie B | delta vie B | pic assemblage B | delta assemblage B | post-close RSS B | HWM B | échantillons RSS/hôte/assemblage | cellule pic | mapped pic B | elapsed s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c512-g4x4 | 512²/4×4 | 38,292,584 | 411,766,784 | 27,873,280 | 410,308,608 | 26,415,104 | 404,307,968 | 414,048,256 | 181/182/18 | 2 | 2,065,920 | 0.370 |
| c1024-g4x4 | 1024²/4×4 | 148,147,304 | 487,596,032 | 103,686,144 | 487,596,032 | 103,686,144 | 463,904,768 | 503,779,328 | 213/214/40 | 2 | 7,923,200 | 0.820 |
| c1024-g8x8 | 1024²/8×8 | 133,720,632 | 497,516,544 | 112,353,280 | 497,516,544 | 112,353,280 | 457,035,776 | 497,860,608 | 316/317/109 | 2 | 2,065,920 | 2.256 |

Chaque cas : budget réel `ram_contract.admit_budget_bytes` (MemAvailable au départ 3,188,166,656 B), cap=2, enveloppe live armée, offsets+feather per_cell, commits et hash de contenu du store identiques avant/après, arrays retournés valides après close, `live_cells==0`. Parité dense **uniquement 256×256/4×4** : `PARITY_PASS`, huit plans et offsets identiques. Les digests 512/1024 sont des sorties observées, PAS une preuve d'égalité dense.

Preuves : `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028-r1/run-20261028T212811Z/` (`matrix.json`, `summary.json` à générer, `measure.json`, `parity-256-g4x4/parity.json`, logs). Tests : **34 passed** (harnais), **71 passed** (harnais+M5-A), **39 passed** (M5-B publication). Exécutable Python `/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python` (3.13.5). Hashes : rss_bench `ae095125…`, test RSS `80cc9e59…`, rss_probe `9543ad3f…`, STATE `554f16a5…`.

NOT_RUN / non qualifié : FITS complet, M3 pression physique, Windows/NTFS, BLAS-scale, GPU/M31, jauge globale mesurée (seul le plan metadata est montré). Identité du store synthétique (`execution_context.available_ram_bytes=4 MiB`), pas la RAM mesurée. M5-B RSS **non accepté** avant vérification Junior + Nono. Aucune activation par défaut/M6.

---
## Historique — M5-B RSS Linux r0 (PROVISOIRE, remplacé par REWORK-1)

## Current status — RSS Linux r0 NON ACCEPTÉ, revue Nono en cours (2026-10-28)

Mission `ZM-ZEGRID-RSS-LINUX-20261028`, HEAD `8564070bf534d90037cbeb3d420303d9b86da264`. Livraison bornée tools/tests/docs, aucun produit/commit/push. Junior **55 passed, zéro skip/échec, 44.01s**, mais sondes adversariales confirment : REFUSED+crash/timeout -> NOT_RUN; absence totale de mesures de surveillance -> PASS; checks={} -> PASS. Source : exceptions génériques converties REFUSED, FAIL matrice rétrogradé PARTIAL après cas stoppé, double append cas; hashes enfant avant/après tous calculés après, RSS annoncé post-close mesuré avant close. Parent conserve une vraie comparaison des records de commits avant/après, mais pas preuve complète d'immuabilité des payloads. Parité dense réelle limitée au256, pas512/1024.

Nombres de la matrice Coco conservés comme observations PROVISOIRES, pas qualification acceptée. Grande matrice non rejouée Junior avant correction des gardes/preuves. Nono review0 accepted/queued `e3363baa-d857-498d-a2fa-0823815253ef`; attendu workspace `.a2a-reports/ZM-ZEGRID-RSS-LINUX-20261028.nono.r0.md`. Rapport Junior/sondes/snapshot/log même préfixe. Attendre callback sans polling puis REWORK borné SAME Coco sans reset; 0/3 corrections consommées. Windows/M3physique/FITS complet/BLAS/GPU/M31 restent exclus, aucune activation.

---
## Historique — livraison Coco avant vérification indépendante

## Current status — M5-B RSS Linux (bounded larger canvases) — implemented, NOT accepted (2026-10-28)

Mission `ZM-ZEGRID-RSS-LINUX-20261028` (implementation, no commit/push). The
legacy `tools/zegrid_resume_m5b/rss_probe.py` (24x24 / 4 cells, produced
payloads retained, `require_budget=False`) is now explicitly labelled a
tiny NON-QUALIFYING witness. New `tools/zegrid_resume_m5b/rss_bench.py` adds a
bounded measurement harness plus focused tests.

Measured matrix (Linux ext4, own artifact dir under the workspace, fresh child
per assembly; store created in a SEPARATE child one cell at a time):

| case | canvas/grid | model required | sampled peak RSS | delta vs pre | post HWM | peak live cells | mapped peak | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c512-g4x4 | 512² / 4×4 | 38,292,584 B | 208,879,616 B | 26,460,160 B | 213,303,296 B | 2 | 2,065,920 B | 0.378 s |
| c1024-g4x4 | 1024² / 4×4 | 148,147,304 B | 302,063,616 B | 119,820,288 B | 302,002,176 B | 2 | 7,923,200 B | 0.823 s |
| c1024-g8x8 | 1024² / 8×8 | 133,720,632 B | 280,768,512 B | 97,734,656 B | 297,365,504 B | 2 | 2,065,920 B | 1.578 s |

Each case used the REAL `ram_contract.admit_budget_bytes` (fresh
`available_memory_bytes`, ~3.3 GiB observed at run time), the product
`_assemble_cells_guarded` path with cap=2 and an armed live-byte envelope, and
per_cell offsets+feather. Commits and output digests were identical
before/after; returned arrays survived session close; `live_cells==0` after
close. A separate-process parity case (256² / 4×4) matched the eager dense
reference on all eight canvas planes and the offset graph exactly.

Evidence: `/home/tristan/.openclaw/workspace/.m5b-rss-linux-20261028/run-20261028T210950Z/`
(`matrix.json`, per-case `measure.json`, `parity-256-g4x4/parity.json`, logs).
Focused harness tests: 18 passed; M5-A assembly baseline 37 passed. Whole run
33.1 s, disk 107,406,726 B (budgets 300 s / 512 MiB).

NOT qualified: full FITS product, physical M3 waiting/recovery, Windows/NTFS,
BLAS-scale/deep-contributor canvases, and global-gauge measurement (only the
metadata plan is shown). Synthetic producer identity `execution_context` is a
4 MiB fixture, NOT host RAM. Host `MemAvailable` moved from ~1.3 GiB before the
matrix to ~3.3 GiB during it (external process release): fresh observations
were used and no value was inflated. M5-B RSS evidence is NOT accepted until
Junior verification + Nono review. No commit/push/default activation/M6.

---

## Current status — M2-READER ACCEPTÉ, intégration Linux vérifiée (2026-10-28)

Mission `ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028` close : Nono review-0 ACCEPT puis décision Junior ACCEPT. Les deux échecs étaient dus au lecteur de manifeste racine périmé depuis M5-B, pas à un défaut établi du store. Le témoin résout maintenant l'identité exacte du store et CURRENT vérifié, sans repli racine. Code produit inchangé.

Junior : **210 passed, zéro skip/échec** (99 M2/store/IO/crash/identité + 111 M4/M5). Les deux tests M2 réels passent, avec ordre commit durable avant release observé, rechargement core/patch/mask byte-identique et immutable, index complet, refus de contrat scientifique modifié. Sondes du helper : pointeur absent/membre corrompu malgré leurres/identité différente refusés. Le hash du test est inchangé depuis ces rejeux. Aucune parité scientifique finale store vs no-store déduite des seules formes FITS.

Git : `feat/zegrid-resilient-run-resume` HEAD `fb268cb2df9892a9a783a0f72e29546bfdeae246`; lot tests M2 + STATE accepté ; Tristan a autorisé le commit et le push sur cette seule branche feature le 2026-10-28. Le SHA de publication sera consigné dans le rapport durable après vérification distante. Rapport final workspace `.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028.junior.acceptance.md`. Aucun worker actif pour cette mission.

Suivis non bloquants : tests négatifs du helper à pérenniser (sondes indépendantes déjà faites), commentaires de diagnostic à clarifier, log close `peak_resident_*` à aligner sur `peak_live_*` (fermeture effective déjà exécutée, pas de fuite démontrée). Qualifications restantes : pression physique M3, Windows/NTFS réel (dual boot possible), reprise produit complète sans seams, grands RSS/BLAS, GPU/M31. Pas d'activation par défaut/M6. Les preuves historiques ne sont pas effacées.

---
## Historique — vérification et revue avant acceptation

## Current status — M2-READER vérifié Junior, revue Nono en cours (2026-10-28)

Mission `ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028`, HEAD `fb268cb2df9892a9a783a0f72e29546bfdeae246`. Diff borné aux tests M2 + STATE, sans code produit/commit/push. Junior : **99 passed** (M2/store/IO/crash/identité, 43.15s) + **111 passed** (M4/M5, 136.96s) = **210 passed, zéro skip/échec**. Les deux vrais tests M2 exécutés. Sondes du nouveau helper sur copies des sorties réelles : génération valide acceptée, pointeur absent/membre corrompu malgré leurres racine/identité différente refusés. Preuves `.a2a-reports/...M2-READER-20261028.junior.r0.md`, logs et JSON associés.

Nono review-0 accepted/queued `ad0d6946-94c1-4f2f-a512-4d4925b05d81`; attendu `.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028.nono.r0.md`. Attendre callback sans polling. Pas d'acceptation finale avant revue. 0 correction consommée; même contexte Coco à conserver.

Suivi distinct : log close utilise des clés périmées `peak_resident_*` alors que stats expose `peak_live_*`; close est exécuté avant le warning de formatage. Pas de fuite démontrée, correction diagnostic à traiter séparément. M3pression/Windows/qualification produit complète sans seams/grandsRSS restent ouverts; aucune activation.

---
## Historique — livraison Coco avant vérification Junior

## Current status — M2 lecteur par génération corrigé, tests d'intégration rejoués PASS (2026-10-28)

Mission `ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028`, base `fb268cb2df9892a9a783a0f72e29546bfdeae246`, branche `feat/zegrid-resilient-run-resume`. Aucun commit/push.

Cause racine établie : le témoin M2 lisait encore le manifeste racine obsolète `out/zegrid_manifest.json`. Le produit n'a PAS échoué et la dégradation vers M1 n'est PAS établie. Correction bornée du lecteur de test : le run est résolu depuis l'identité du store durable (`run_id`, `identity_sha256`), puis `run_publication.resolve_current(verify_members=True, expected_identity_sha256=...)`; manifeste/science/coverage sont lus dans CETTE génération vérifiée, sans glob, sans fichier le plus récent, sans fallback racine et sans confiance dans `state.finalized` seul. L'ordre commit durable -> release est désormais prouvé par la séquence d'événements réellement observée (un commit par cellule COMPLETE, release seulement après), pas par la seule présence des fichiers. Ajout de 5 contrôles lecteur : pointeur absent, pointeur corrompu, membre corrompu malgré des leurres racine, membre absent, pointeur revendiquant un autre run — tous échouent fermé.

Résultats exacts : `test_zegrid_resume_m2_store_run.py` 17 passed, 0 skipped, les DEUX tests réels exécutés (4 cellules, 4 commits, 4 releases, 55 370 541 B libérés, 0 invalid ; 2 803 776 KiB disponibles, soit ~2,67 GiB) ; store/IO M2 82 passed ; M4 21 passed ; M4 e2e 8 passed ; M5 assembly+publication 82 passed ; `git diff --check` OK.

Statut : M2 NON ACCEPTÉ jusqu'à vérification Junior puis revue indépendante Nono. Les qualifications historiques M2/Windows ne sont pas effacées. Restent NON qualifiés : pression physique M3, Windows/NTFS réel, produit complet sans seams, grands RSS/BLAS, GPU, M31. NTFS n'est pas l'unique condition avant activation ; aucune activation/M6.

---
## Historique — dispatch initial M2-READER (2026-10-28)

Mission `ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028`, base `fb268cb2df9892a9a783a0f72e29546bfdeae246`, branche `feat/zegrid-resilient-run-resume`. Autorisation Tristan de poursuivre M2.

Correction du diagnostic précédent : la dégradation vers M1 n'est PAS établie. Junior a reproduit les deux échecs avec RAM disponible (~2809 MiB) : les tests cherchent encore `out/zegrid_manifest.json`, chemin obsolète depuis la publication par générations M5-B. Pour chacun des deux runs, CURRENT résolu avec vérification de tous les membres et de l'identité : store actif, 4 cellules committées, 4 libérées (55 370 541 B), 0 invalid, aucune erreur ouverture/commit/dégradation. Les diagnostics existent dans le manifeste de la génération. Les assertions ultérieures (parité payload, ordre release, refus identité) doivent encore être exécutées : pas d'acceptation anticipée.

Baseline : `FF..........`, 2 échecs lecteur ancien / 10 pass. Preuves workspace `.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028.junior-baseline.json`; sorties conservées `/tmp/zm-m2-investigation-20261028-junior/`.

Coco réinitialisé avant cette nouvelle mission (tokens0), dispatch accepted/queued `6dbd6fa8-3af7-495a-a5d4-284ce57acfb4`. Scope : lecteur de test lié à CURRENT+identité, sans fallback racine, maintien/renforcement des preuves scientifiques et refus corruption. Rapport attendu `.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M2-READER-20261028.coco.r0.md`. Puis vérification Junior et revue indépendante Nono, sans polling. Aucun commit/push demandé au worker.

M5-C code déjà publié; M2 intégration en vérification, sans effacer les qualifications historiques. Windows/NTFS, pression physique M3, qualification produit sans seams et grands RSS restent NON qualifiés; NTFS n'est pas l'unique condition avant activation. Pas d'activation/M6.

---
## Historique — diagnostic antérieur corrigé ci-dessus

## Current status — M5-C flush portable ACCEPTÉ, M2 durable-store révélé en échec (2026-10-28)

M5-C ACCEPTÉ (code + tests) et publié. `_flush_member` (`run_publication.py`) ouvre
désormais `O_RDWR|O_BINARY` (sans O_TRUNC/O_CREAT/O_WRONLY) puis repli `O_RDONLY`
sur OSError; échec des deux => `PublicationError` avant `publication.json` et avant
pointeur, CURRENT précédent intact, message « no pointer was written ». Ordre F4
inchangé (membres -> répertoire génération -> ancêtres -> record -> pointeur).
Motif: sous Win32 `os.fsync` = `FlushFileBuffers`, qui exige un handle avec droit
d'écriture; `run_store.fsync_dir` garde son no-op Windows documenté (hors périmètre).

Preuves Junior (rejouées): 6 passed (nouveau `test_zegrid_publication_m5c_flush_portable.py`),
76 passed (M5 assembly+publication+execution), 104 passed (R23/R24/R25/R18),
`git diff --check` OK. Tests d'injection: simulation Win32 (refus lecture seule =>
flush par handle écriture), échec total => aucun pointeur, ordre F4 instrumenté,
octets/digests/jeu de fichiers inchangés, `fsync_dir` no-op Windows.

RÉVÉLATION M2 (pré-existante, PAS une régression M5-C): les 2 tests
`test_m2_real_cpu_run_commits_then_releases_and_reloads_identical_payloads` et
`test_m2_store_identity_refuses_a_changed_science_contract` étaient IGNORÉS faute
de RAM (`MIN_LIVE_BYTES=1200 MiB`). Avec ~1.9-2.1 GiB libres ils s'exécutent et
ÉCHOUENT: « the run published no r55_durable_store diagnostics although
zegrid_durable_store_dir=... was configured ». Reproduction À L'IDENTIQUE sur
worktree pristine `17af7f8` sans M5-C => défaut pré-existant du chemin produit
store durable (dégradation probable vers M1), jamais qualifié jusqu'ici car masqué
par le skip. M2 reste NON accepté; mission d'investigation à ouvrir.

NOT_RUN: Windows/NTFS réel (sémantique Win32 simulée seulement), coupure électrique
réelle, GPU, M3 pression réelle, M31, RSS BLAS-scale, activation par défaut.
Aucune activation: `zegrid_resource_recovery_enabled=false`, `zegrid_resume_mode=never`.

---
## Historique — les états ci-dessous ne remplacent pas le statut courant

### Dispatch REWORK-2 — Junior (2026-10-28)

Nono review-1 = ACCEPT-WITH-CONDITIONS, pas acceptation finale Junior. F1–F4 fermés; Junior exige les deux corrections bornées restantes avant clôture: F5 ack complet (sha seul ne suffit pas), tests vrais repair puis no-op stable/échec repair nonfatal; F6 provenance exacte du témoin RAM8GiB (global upstream, incluant assemblage, pas PLAN-only) ou seam planning restreint, fixture1x1. Sept hashes code/tests/tools inchangés, diff-checkPASS; preuves Junior218passed2skipped et sondes restent valides.

SAME mission ZM-ZEGRID-RESILIENT-RUN-RESUME-M5B-20261028, SAME Coco sans reset. REWORK-2 (2/3) accepted/queued865c3343-bd94-40b3-92b8-4a9e9481ccc4. Brief workspace .a2a-reports/...M5B-20261028.rework2-brief.md; attendu .coco.r2.md exact-source callback. Prochain pas: vérification Junior puis Nono; pas de polling. M5-B NON accepté. HEADb092768, aucune publication git/commit/activationM6. Pression réelle M3, budgets produit sans seam, Windows/GPU/M31 restent NOT_RUN; aucune qualification de coupure électrique revendiquée.



## Current status — M5-B REWORK-1 vérifié, Nono review-1 en file (2026-10-28)

Junior a rejoué séquentiellement: M5A/B 75 passed (61.50s), M2/M4 39 passed +2 skipped (75.33s), finitions/export 104 passed (82.09s): **218 passed, 2 skipped, aucun échec**. Sondes indépendantes F1–F4 PASS: RAM inconnue refusée, retry exact zéro écriture/collision metadata immuable, ancienne génération avec marker résiduel conservée, flush FITS avant record puis pointeur (preuve d'ordre Linux).

Restent à arbitrer: ack partiel avec seul sha256 valide encore déclaré verified_finalized (autorité fichiers indépendante intacte); provenance du harnais 8GiB global affecte aussi admission assemblage, pas seulement PLAN; tests repair ack production incomplets. Vrai _run_single+finitions+publication et nouveau-processus no-op exécutés sous seam ressource upstream explicite; PAS qualification budget réelle. Nouveau run_id store-safe inspecté, régressions ciblées passent.

Mission ZM-ZEGRID-RESILIENT-RUN-RESUME-M5B-20261028, 1/3 corrections consommées, HEAD b092768, delta attendu, pas commit/push. Nono review-1 accepted/queued aa950173-cc61-4679-83a2-f17d626ee885, rapport attendu workspace .a2a-reports/...M5B-20261028.nono.r1.md. Preuves Junior .junior.r1.md/.junior-probes-r1.py/json et snapshot7fichiers .review-snapshot-r1.json. Attendre callback sans polling; puis adjudication et éventuel REWORK-2 SAME Coco sans reset. M5-B NON accepté. M3 pression réelle/M4 produit sans seam/Windows/GPU/M31 restent NOT_RUN; aucune activation M6.

---
## Historique — les états suivants ne remplacent pas le statut courant

## Current status — M5-B REWORK-1 CODE LIVRÉ, NON ACCEPTÉ (2026-10-28)

Corrections F1–F6 livrées. Aucun commit/push. Baseline `b092768e79c7833af622c0bc1c808c17b6dbefd3` inchangée.

- **F1 fail-closed** : `size_finishing_and_publication()` (pur, `admitted=false`, `sizing_only`) séparé de `admit_finishing_and_publication()` ; RAM inconnue/invalide/échec de sonde ⇒ `PublicationCapacityError` avec plan honnête AVANT toute génération/allocation ; aucun override. `_run_single` refuse avant écriture raw/finitions/génération. `reattach` read-only ne crée jamais de publication.
- **F2 immuabilité** : une génération portant `publication.json` est immuable. Retry exact = validation + retour, ZÉRO écriture (ni membre, ni record, ni pointeur). Tout changement sémantique (identité/rôles/options/metadata/canvas/digests) ⇒ collision sans écriture. Retry d'une génération plus ancienne ⇒ refus (pas de retour arrière du pointeur). Crash avant pointeur ⇒ seul le pointeur est écrit, record intact.
- **F3 cleanup** : `publication.json` et un marqueur résiduel ne sont plus une preuve de staging jetable. Seuls sont supprimés les `gen-N` du même run, sans record, marqueur avec membres énumérés, propriétaire mort, contenu strictement membre+marker. Le reste (committed, inconnu, étranger) est conservé.
- **F4 durabilité ordonnée** : fsync de CHAQUE membre FITS, puis `publication.json`, puis répertoires/génération/ancêtres, puis pointeur. Échec de flush avant pointeur ⇒ erreur explicite, CURRENT précédent intact. `atomic_write_bytes` fsyncait déjà son record (pas de revendication contraire).
- **F5 no-op produit** : le no-op `_run_single` retourne le MÊME type de chemin qu'un succès normal (esthétique si présent, sinon SCI) depuis la génération vérifiée, avec notification SUCCESS terminale et libération des ressources ; l'ack état est réparé s'il est absent OU périmé, sans republier/restacker ; les lignes requises manquantes/partielles ne sont jamais « consistentes ».
- **F6 preuves d'exécution réelles** : nouveau `tests/test_zegrid_publication_execution_m5b.py` (4 tests) : garde capture en TÊTE de `_run_single` (store+R37/R40 refusés avant descripteurs/effets, no-store passe, no-op ne contourne pas) ; run produit store-backed réel (`_run_single`, scheduler ACTIVE, petit corpus 4×320²) avec assemblage borné, finitions DBE+equalize réelles, sérialisation et publication, puis NOUVEAU processus `resume_mode=auto` où les seams coûteux (`layout`, `gauge`, `assembly`, `finishing`, écritures) tuent le process (exit 77) : no-op vérifié, chemin esthétique retourné dans la génération, digests sortie/commits/gauge/geometry inchangés ; échec de sonde RAM ⇒ refus avant raw/finitions/génération ; parité finitions off/on réelle sur la science publiée. Le run_id interne du registre (`zegrid:<ts>:...`) est normalisé en composant store-safe avant identité/état/publication (sinon le store ACTIVE ne pouvait pas s'ouvrir) ; identité et reprise utilisent le même id normalisé.

Tests exécutés : **75 passed** (M5A 37 + M5B 34 + exécution 4), **39 passed/2 skipped** (M2/M4 store+e2e), **104 passed** (R23/R24/R25/R18 finitions/export). Les 3 échecs R23 observés en lançant deux suites en parallèle sont la sensibilité préexistante RAM/ordre (rejoués seuls : 104 passed) ; ils ne sont pas causés par M5-B et ne sont pas masqués. Un seam de test limite l'observation RAM du PLAN (hôte contraint) : l'admission de PUBLICATION reste une observation réelle ; ce seam est documenté et jamais une configuration produit.

NOT_RUN : Windows/NTFS, GPU, pression réelle M3, produit complet M4, M31, qualification M5 ; RSS 4×24² reste une observation étroite, pas une borne BLAS. Aucune activation par défaut. Prochaine étape : vérification Junior + Nono.


---
## Historique — les états ci-dessous ne remplacent pas le statut courant

### Dispatch REWORK-1 — Junior (2026-10-28)

Nono review-0 FINDINGS vérifié par Junior: F1–F4 confirmés par sondes indépendantes, F5 retour no-op/ack stale et F6 preuves d'exécution manquantes retenus. Snapshot des cinq fichiers code/tests/tools inchangé, diff-check PASS; preuves précédentes 63 passed + sondes restent valides. Correction à la revue: publication.json est déjà fsync via atomic_write_bytes, ce sont les membres FITS et l'ordre complet à corriger. Retry identique => réutilisation sans écriture; changement de contrat => refus/nouvelle génération, pas mutation de l'autorité. Pas d'override RAM inconnue, pas de réécriture systématique des états finalisés valides.

Mission ZM-ZEGRID-RESILIENT-RUN-RESUME-M5B-20261028, SAME Coco sans reset, première correction (1/3) accepted/queued run 025a3248-1c6f-42d4-83f8-518f313e426d. Brief durable workspace .a2a-reports/...M5B-20261028.rework1-brief.md, rapport attendu .coco.r1.md exact-source callback. Scope publication/admission/cleanup durable + vrai chemin production et garde capture. Prochain pas: callback, vérification indépendante puis revue Nono; aucun polling. M5-B NON accepté. Pas commit/push/activation/M6. M3 pression réelle, M4 produit/Windows, M5 Windows/GPU/M31 restent NOT_RUN.


---
## Current status — M5-B r0 NON ACCEPTÉ, revue Nono en file (2026-10-28)

Callback Coco reçu: le timeout initial n'avait pas empêché le lancement. HEAD b092768e79c7833af622c0bc1c808c17b6dbefd3, delta attendu seulement, aucun commit/push. Junior a rejoué M5A+B: **63 passed, 26 warnings, 36.72s**, diff-check PASS.

Défauts reproduits par Junior: (F1) RAM inconnue admise, même modèle 9.634GB; (F2) retry modifie publication courante avant pointeur, panne pointer => digest mismatch; (F3) cleanup supprime ancienne génération committée si marker post-commit non supprimé; (F4) pas de fsync FITS avant marqueur final (trace réelle). Source: no-op _run_single retourne None/ack stale non réparé; tests intégration/garde capture requis encore manquants. M5-B pas accepté, 0 correction consommée.

Mission ZM-ZEGRID-RESILIENT-RUN-RESUME-M5B-20261028; rapports/preuves/snapshot sous workspace .a2a-reports. Nono review-0 accepted/queued run 8eee5c30-7d65-4580-903c-44cac75f2116, rapport attendu .nono.r0.md via callback exact source. Prochain pas: disposition indépendante Nono puis REWORK-1 borné à la MÊME session Coco, sans reset ni polling. Qualifications pression réelle M3 / produit+Windows M4 / Windows M5 / M31 restent NOT_RUN; aucune activation.


---
## Historique — M5-A et états antérieurs

## Current status — M5-A CODE ACCEPTÉ (2026-10-28)

Nono review-2 = ACCEPT-WITH-CONDITIONS, aucun HIGH/MEDIUM bloquant. G1 fermé (le modèle facture exactement l'allocation réelle du solveur dense : A=2 425 920 B à N400/E760, total 11 754 664 B, zéro coût en global). G2 fermé (hash non contigu réellement borné : 4 308 B au lieu de 2 098 088 B pour 1 MiB strided; digests canoniques identiques; contigu zéro-copie). Garde capture ordonnée avant les travaux coûteux. Cap 2 cellules et vues externes jamais fermées de force conservés. Junior a rejoué 155 passed / 5 skipped et reproduit les deux corrections par sondes indépendantes.

Périmètre M5-A: accès read-only borné au store (mmap, vérification+fingerprint une fois par identité de session), préflight de capacité métadonnées-seul branché avant tout chargement/graphe/canvas, assemblage scientifique exact (global et per_cell) et rejeu idempotent après reopen. Aucune activation par défaut, aucun changement de science/schéma/budget.

CONDITIONS portées en M5-B (non acquises): mesure RSS/pic réelle de la borne (BLAS/workspace/Nfree), test d'exécution de la garde capture (store+R37/R40 refusé, sans store accepté), documentation du coût CPU du hash non contigu hors chemin chaud, conservation du cap/rétention, pas d'activation par défaut.

NOT_RUN: produit complet avec store, Windows, GPU, pression réelle M3, M31, finitions/publication (M5-B). M5 n'est PAS complet.



Corrections 1 et 2 livrées. Junior a rejoué 155 passed / 5 skipped (45.95 s; 148 passed hors tests RAM-gated) et vérifié par sondes indépendantes: (G1) le modèle du solveur dense facture exactement l'allocation réelle (A=2 425 920 B à N400/E760) avec un total conservateur 11 754 664 B et zéro coût en mode global; (G2) le hash non contigu est réellement borné (pic 4 308 B au lieu de 2 098 088 B pour 1 MiB strided, chunk 1 024) avec digests canoniques identiques; la garde capture s'exécute avant les effets de bord. Aucune modification de science, de schéma/identité de store, de défaut d'activation.

Nono review-2 read-only en file (run b46bf446-5dde-44f5-8245-42b59a279b96). M5-A reste non accepté jusqu'à son verdict; M5-B (finitions + publication transactionnelle multifichiers) non commencé. NOT_RUN: produit complet avec store, Windows, GPU, pression réelle M3, M31. Aucun commit/push. Historique ci-dessous conservé.

## Current status — M5-A REWORK-2 queued (2026-10-28)

Nono review-1 FINDINGS adjudicated: dense graph underpricing and noncontiguous hash chunk violation confirmed by Junior probes/source; four-cell eager retention already covered. No proof that historical R37/R40 stores are incompatible: correction limited to testing/ordering current store+capture request guard, no blanket store rejection. Four code/test hashes unchanged, STATE orchestration-only differences understood.

Second correction dispatched to SAME Coco session (no reset), accepted/queued run5720f65d-2714-4e3b-8e1d-3f9d7cc362d0. Mission ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028; durable .rework2-brief.md and .junior-review1-disposition.md under workspace .a2a-reports. Expected .coco.r2.md exact-source callback; no polling. Then independent verification + Nono follow-up. Scope mode-aware dense-solve model, bounded noncontig hash, early tested capture guard; science/defaults/store identity unchanged.

M5-A NOT accepted, M5-B not started, M3 realpressure/M4 fullproduct/Windows NOT_RUN. No commit/push.

## Current status — M5-A r1 verified, Nono review-1 queued (2026-10-28)

Junior replay: 142 passed / 5 skipped (15.47s); independent 3x3 float32/float64 global/per_cell dense parity PASS, mapped-cell peak2. Preflight now before consumer; lifetime enforcement improved. NOT ACCEPTED: graph model linear96B/cell misses actual dense solver matrices. For400cells, A alone2,425,920B > total advertised model1,137,070B (graph38,400B). Noncontiguous hash also copies outer slice (2MiB observed for1KiB chunk); current production C-order maps limit applicability of that second issue. Evidence in workspace .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028.junior.r1.md and .junior-probes-r1.json.

Nono read-only review-1 accepted/queued run21534f86-4a2f-402d-afc0-54fa88dc2c08; expected .nono.r1.md via exact-source callback, no polling. Then consolidate bounded REWORK-2 to SAME Coco context if findings confirmed. 1/3 corrective iterations used. M5-B not started, M3 realpressure/M4 fullproduct/Windows still NOT_RUN; no commit/push. Older status sections below are historical.

## Current status — M5-A REWORK-1 queued (2026-10-28)

Nono review-0 FINDINGS confirmed by Junior: production preflight missing, real mapping lifetime under-counted, repeated full-plane fingerprint copy unpriced. Code/test hashes unchanged from Junior 129 passed / 5 skipped and live-mapping probes; only orchestration STATE note differs. No acceptance.

Mission ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028 continues in SAME Coco session, no reset. First corrective iteration dispatched asynchronously, accepted/queued run ae9b6d8c-fde8-47a6-8d94-be1ae9ca39aa. Durable brief: workspace .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028.rework1-brief.md; expected report .coco.r1.md via exact-source callback. Correct preflight + actual safe lifetime enforcement + bounded integrity hash and model; no science/default changes. Await callback, no polling, then independent verification and Nono follow-up.

M5-B not started; M3 physical pressure unqualified; M4 product/Windows NOT_RUN. No commit/push. Older implementation hard-cap/verify-once claims below are historical, NOT accepted evidence.

## M5-A REWORK-2 (Coco, 2026-10-28) — code-only, NOT accepted, M5-B not started

REWORK-2 closes review-1 findings G1/G2 and narrows the R37/R40 guard:

- **G1 (HIGH)** — `bounded_assembly.dense_solve_bytes()` prices the REAL
  canonical dense least-squares solve from metadata only: `A(E, Nfree)`,
  weighted `A`, previous-channel `A`, `AᵀA`, LAPACK copies, edge vectors,
  per-node copies and a documented 25 % BLAS margin. Reference N=400/E=760:
  `A` = 8·760·399 = **2 425 920 B** (previous model priced the WHOLE graph at
  38 400 B). Recovery N=400: graph bound **11 754 664 B** (vs total plan
  1 137 070 B before). Edges are the real 4-neighbour pairs of the selected
  cells (`grid_edge_upper_bound`); `global` mode prices EXACTLY 0 (no offset
  solve runs). Refusal carries the full preflight provenance.
- **G2 (MEDIUM)** — `array_sha256_streaming` now hashes non-contiguous arrays
  (strided / transposed / negative stride / channel slice) through blocks of at
  most `chunk_bytes`: the last axis is walked in element blocks, every leading
  index in C order. No `np.ascontiguousarray` of an arbitrary outer slice, no
  flatten/ravel copy. Contiguous (production mmap) path stays zero-copy.
  Canonical digest unchanged; tracemalloc peak bounded by the chunk.
- **R37/R40** — `_validate_capture_store_compatibility()` is a pure predicate
  called right after the store is known (before any expensive run work and
  before the R40 capture context); a store-active capture request fails
  explicitly, store-less captures are unchanged, no store identity/schema
  rejection was added (no evidence justified it).
- Manifest `r55_bounded_assembly.preflight` now publishes the solver inputs
  (cells/edges/free), gauge mode, grid and the metadata-only/refused facts.
- Evidence (REWORK-2): M5 file **37 passed** (26.41 s); combined M5 + M2 store +
  M2 IO + M4 + feather/offset/R3 assembly **155 passed, 5 skipped** (36.82 s);
  `git diff --check` OK. Including a N=400 tiny-patch store: the guarded entry
  refuses on a 2 MiB budget with ZERO loads, verifications and solver calls.
- NOT_RUN: full product run with the store, Windows, real-pressure assembly,
  M31; execution order inside `_run_single` for the capture guard is verified
  structurally + by predicate tests only. M3/M4 qualifications unchanged
  (NOT_RUN). M5 not complete.

## Current status — M5-A dispatched (2026-10-28)

Night mission resumed after the sessions.reset tool-registry blocker. Reset confirmed by fresh Coco history (Reset seq1) despite RPC timeout; dispatch accepted/queued run 45ea85bd-db7a-42f2-875c-dc5111339794.

Active mission: ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028. M5-A is the bounded store-read/scientific-assembly slice (immutable checkpoints, exact parity, bounded working set); M5-B will cover final finishing and coherent multi-file publication. M5 is NOT accepted yet. Baseline HEAD 26be05550789a5f52bf49308072ef15842e3b0b3 clean before delegation; two store/feather baseline tests pass (3.79s).

Expected report: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028.coco.r0.md. Next: callback, independent Junior verification, then Nono review. M3 real-pressure and M4 actual product/Windows qualifications remain NOT_RUN; no default activation and no M31 run.

## M5-A implementation + REWORK-1 (Coco, 2026-10-28) — code-only, NOT accepted, NOT M5

M5-A (bounded store read + immutable scientific assembly) implemented on
`feat/zegrid-resilient-run-resume` from `26be055`: no commit, no push.
REWORK-1 corrected the three Nono/Junior findings (F1/F2/F3); the r0 claims
below that REWORK-1 replaced (a hard "LRU cap = 2" as the bound, "verify-once"
as a blanket statement, and an "exact" working-set model / "700 MiB" canvas)
are OBSOLETE and must not be used as evidence.

- New isolated module `core/zegrid/bounded_assembly.py`:
  - `CellReadSession`: read-only `np.memmap` planes, integrity + canonical
    fingerprint checked **once per immutable identity**
    `(cell_id, attempt_id, payload_sha256)` per session, with the hash
    **streamed** in fixed chunks (`array_sha256_streaming` /
    `payload_sha256_streaming`, byte-identical to `run_registry`), never a
    full-plane copy.
  - REAL live-mapping accounting through weak references: `live_cells`,
    `live_mappings`, `live_mapped_bytes` and their peaks count external views,
    slices and reopen of the same cell. Enforcement happens BEFORE a new
    mapping is opened (LRU refs dropped first, then an explicit
    `BoundedLifetimeError` if external retention still exceeds the envelope).
    Mappings a consumer holds are NEVER force-closed.
  - `preflight_assembly()`: metadata-only, fail-closed capacity gate
    (commit manifests + canvas price + conservative transient + hash chunk +
    graph). The production path REQUIREs a real budget; `require_budget=False`
    is the explicit diagnostic mode.
  - `assembly_plan()`: conservative bound (not a measured RSS); canvas is
    121 B/px at 3 channels (953,121,114 B ≈ 909 MiB for 2403×3278).
  - `assert_bounded_usage()` reads the real live counters; an eager
    `list(view.values())` or externally retained set now FAILS safely instead
    of passing the guard.
- Wiring: the store path no longer calls `ordered_payloads` eagerly;
  `_assemble_cells_guarded()` runs the preflight BEFORE `_assemble_cells` (so
  before any payload load, graph solve or canvas allocation) and propagation is
  an explicit `AssemblyCapacityError` (no false COMPLETE, commits preserved).
  `cell_feather.assemble_canvas_feathered` gained an OPTIONAL `allocator`
  parameter (default `np.zeros`, numeric path identical). Store-less path
  unchanged.
- Manifest: `per_cell_diagnostics["r55_bounded_assembly"]` carries the
  preflight plan (enforcement), the live-mapping statistics and a guard verdict
  (evidence, not enforcement).
- Local evidence (REWORK-1): `tests/test_zegrid_bounded_assembly_m5.py`
  **24 passed**; M2 store 48, M2 IO 8, M4 resume 21, feather/offset/R3 assembly
  41 passed + 5 skipped; combined **142 passed, 5 skipped** (14.44 s).
- NOT_RUN: real product end-to-end with the store, Windows, real-pressure
  assembly, M31. M5-B (final generation + last marker, multifile atomic
  publication, kill at finalization) NOT started. M5 is NOT claimed complete.

## Historical M5-A r0 note (superseded by REWORK-1 — do not use as evidence)

M5-A (borned store read + immutable scientific assembly) implemented on
`feat/zegrid-resilient-run-resume` from `26be055`: no commit, no push.

- New isolated module `core/zegrid/bounded_assembly.py`: `CellReadSession`
  (verify-once + read-only `np.memmap` planes, hard LRU residency cap = 2),
  bounded `Mapping` views (`descriptors`/`cores`/`patches`), an honest
  working-set plan (`assembly_plan`, exact canvas accumulator bytes) with an
  explicit `AssemblyCapacityError` refusal when a budget is supplied, a defect
  guard (`assert_bounded_usage`) and an optional scratch-backed canvas
  allocator (not activated by default).
- Wiring: the durable-store assembly path no longer calls
  `ordered_payloads` eagerly; `_run_single` receives metadata descriptors and
  bounded views, `_assemble_cells(..., patches=...)` receives the bounded patch
  view. Store-less path unchanged (eager dict, same code). `cell_feather.
  assemble_canvas_feathered` gained an OPTIONAL `allocator` parameter (default
  `np.zeros`, numeric path identical).
- Manifest: `per_cell_diagnostics["r55_bounded_assembly"]` publishes the
  session stats + working-set model + guard verdict (no payload, no pixel).
- Local evidence: `tests/test_zegrid_bounded_assembly_m5.py` 11 passed;
  M2 store 48 passed; M4 resume 21 passed; feather/offset/R3 assembly 41 passed
  + 5 skipped; M2 IO 8 passed. Two `test_zegrid_r40b2_pooled` failures are
  PRE-EXISTING (reproduced identically in a clean worktree at `26be055`).
- NOT_RUN: product end-to-end with the store, Windows, real-pressure assembly,
  M31. M5-B (final generation + last marker, multifile atomic publication, kill
  at finalization) NOT started. M5 is NOT claimed complete.

## Current status — M4 code ACCEPTÉ (ACCEPT-WITH-CONDITIONS, qualification NOT_RUN)

Nono review-0 = ACCEPT-WITH-CONDITIONS, aucun HIGH/MEDIUM. M4 (reprise de run) commitée `1801d70` et poussée sur `feat/zegrid-resilient-run-resume`. Jauge complète + géométrie figée persistées et RESTAURÉES (jamais re-dérivées); mismatch => ResumeIncompatible; reprise sur ensemble clairsemé; run complet no-op; never par défaut; politique dense→bounded restaurée.

NON QUALIFIÉ (NOT_RUN, déclaré): chemin produit complet _run_single de reprise, GPU, Windows, pression RAM réelle, FITS réelle. Kill seam produit DIFFÉRÉ à M7 (décision architecte). La reprise réelle n'est PAS revendiquée; reprise reste défaut `never` sans qualification dédiée.

Prochaine porte: M5 (assemblage reprenable), puis M6 (GUI/CLI + activation), puis M7 (qualification Windows incl. tranche verticale M2–M4 + kill seam produit + pression réelle M3).

# ZeGrid resilient run — project state

## M1 — QUALIFIÉ (banc Windows réel, 27/10 19:42Z)
- Banc `m1_probe_02`: `status=PASS`, exit 0, 237 s, 33 tests passés, **0 skip**, 0 échec,
  `registry_tests_passed=23`, `required_conservation_missing=[]`, plafond artefacts non atteint.
- Les **six** cas de conservation d'orchestration réelle ont tourné et passé sur l'hôte Windows,
  dont `test_m1_mid_wave_generation_transition_rejects_old_results`.
- Source exécutée: HEAD `48a072e835a52598b73b98de144fbe2424a465a7`, arbre propre, `SourceRoot`
  = arbre de travail (pas le produit installé). Produit inchangé pendant le run
  (`mode.py` 2152c6ca…, `run_registry.py` 1ba47fa6…, avant == après).
- Périmètre qualifié: conservation **en mémoire pendant le run**, mode `WINDOWS_M1_CONTROL_ONLY_CPU`
  (exécuteur GPU remplacé par l'exécuteur CPU canonique).
- NON qualifié / NON mesuré: GPU et science GPU (`gpu_qualified=false`), parité avec le produit
  installé (`installed_runtime_used=false`), A/B science sans pression (`NOT_RUN`), sauvegarde et
  reprise disque (M2), GUI, mélange CPU/GPU.
- Revue indépendante: Nono review-2 `ACCEPT-WITH-BENCH-PENDING` -> condition de banc remplie;
  review-3 `ACCEPT` du delta de test `48a072e`.
- Conclusion: **M1 accepté et qualifié dans son périmètre déclaré**. M2 peut démarrer.

## M1 ACCEPT-WITH-BENCH-PENDING + branche publiée (27/10 18:5x)
- Nono review-2: **ACCEPT-WITH-BENCH-PENDING**. Gate de preuve fail-closed confirmé sur 4 fermetures (F1 skip bénin seulement, F2 précédence échec>skip, F3 six IDs exacts, F4 observation fresh-min). Aucun défaut HIGH/MEDIUM restant. Réserves LOW/hors périmètre: dédup `status`, `mmap.size()` conservateur, A/B NOT_RUN.
- Autorisation Tristan: push au retour de Nono sans problème. Commit `d0905dec68fd5a87c6a93ff5fa294a2b6848e3b0` (21 fichiers) + branche `feat/zegrid-resilient-run-resume` poussée sur origin. `beta` intacte à `231b597867b4d251afac6b767000c8a37e2cf3b0`. Arbre propre après commit.
- Conditions non négociables posées par Nono: M2 ne démarre pas avant banc Windows M1 PASS sur les 6 cas; le rapport d'acceptation doit déclarer A/B NOT_RUN et Windows M1 bench NOT_RUN.
- Prochaine action = Tristan lance le banc Windows; ensuite qualification puis M2.

## M1 r3 reprise après coupure — vérifié, revue finale en cours
- Coco r3 livré 18:27 (callback CALLBACK_FAILED auth). Junior repris: produit HASH-IDENTIQUE au snapshot r1 (mode.py+run_registry.py), seuls3 fichiers tests/tools changés.
- Rejoué: guards+registry+conservation38passed6skipped3.74s; M0+zero-support41passed1skip18.96s. Sondes Junior indépendantes: failed+skip=>FAIL, error+skip=>FAIL, nonzero+allPASSED=>FAIL, zero+printedFAILED=>FAIL, cap/timeout prioritaires, skips-only=>NOT_RUN, required-absent=>FAIL, unexpected exc armé ou non=>FAIL, armed-success=>FAIL, unarmed=>SKIP. Défaut reproduit hier (failed+skip=>NOT_RUN) FERMÉ.
- Vérifié code: six IDs requis exacts incl mid-wave, classify_probe_outcome précéde échec>skip, _ClampedObservation min(fresh,cap,requested) par lecture, force_wave_size test-only patch zrt.next_wave_bound.
- Nono review-2 (confirmation finale) dispatch accepted/queued run602d5b32-694d-4ee6-8e19-8ccc83df51e9, expected .nono.r2.md. Snapshot .review-snapshot-r2.json.
- BLOQUANT RESTANT: les6 cas d'orchestration réelle n'ont JAMAIS tourné (RAM vive632MB < plancher1258MB). Banc Windows prêt mais code M1 uniquement sur Linux non poussé. Attente décision Tristan pour push branche feature (PAS beta) afin qu'il pull et lance le probe.

## M1 REWORK-3 FINAL — implementation complete, awaiting verification
- Nono review-1 F1–F4 corrected in the same session: `_require_expected` skips ONLY evidenced pre-case unavailability and FAILS unexpected exceptions or armed-success; the probe uses a pure failure-before-skip `classify_probe_outcome` with exact SIX required node IDs; the mid-wave generation case is required and forced to deterministic two-cell waves; every memory seam re-reads live RAM and returns `min(fresh, start/env cap, requested trajectory)` with fail-closed invalid readings. Status duplicate comparison already includes `status`.
- Rework-3 local evidence: probe guards + registry + conservation unit helpers **38 passed / 6 skipped**; conservation file **3 passed / 6 skipped** (live 1,218,064,384 < 1,258,291,200); M0 preservation + zero-support producer **41 passed / 1 skipped**; focused GPU/CPU/assembly **218 passed / 7 skipped** (product unchanged since rework-2).
- The six real-orchestration conservation cases are **NOT_RUN on this host** for capacity; no fake RAM. No-pressure A/B remains NOT_RUN (candidate standalone PASS at 1,306,472,448 B with science 3772b6cd…/coverage 49ed57f7…; baseline NOT_RUN at 845,000,704/825,798,656 B). Windows M1 NOT_RUN; known adequate Python `C:\Users\JarvisJunior\projects\zemosaic\.venv\Scripts\python.exe`.
- Report .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.coco.r3.md. No commit/push; M1 not accepted by Coco. Next: Junior replay/diff + Nono final confirmation + adequate-host six-case/A-B bench. No fourth iteration; remaining defects must be reported as BLOCKED.

## M1 REWORK-2 verification — Nono review-1 active
- Junior independently74passed7skipped35.19s:6integration skips LIVE654MB<1258MB,1slow skip. Fixed probes: populated zero-support preserved, memmap/memoryview4MB accounted4MB, invalid-output no overwrite. Source CPUgeneration before dispatch, active cap kill confirmed; actual spawn/cap subprocess tests passed.
- Remaining confirmed helper masks unexpected RuntimeError as Skipped; runner skip precedence masks mixed failure;6thmid-wave case absent required list and may not run high-RAM singlewave; live observation pinned once. No-pressure A/B and WindowsM1 NOT_RUN. No M1 acceptance.
- Nono review-1 read-only dispatched accepted/queued run b78f4958-6d89-4ea6-a584-449e86dfaf88, expected .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.nono.r1.md; exact11file .review-snapshot-r1.json. Junior .junior.r2.md/.junior-probes-r2.json. Await callback no polling; consolidate before final REWORK-3 (2/3 corrections used). Coco session retained. No commit/push/deploy/M2; no user action yet.

## M1 REWORK-2 active — residual regressions corrected (Coco implementation)
- Coco r2 implementation completed in the same session: zero-support populated EMPTY_VALIDATED preserved with real `gauge_status="empty"` evidence and real contributor counts; memmap slices follow the mapping and memoryview follows `.obj` (4 MB counted once, `memmap_bytes` labelled separately); rejected/existing/outside/symlink probe paths are never written; live cap/deadline watcher kills the owned process tree; CPU wave generation is captured BEFORE `_run_cell_batch`; new M1 execution is clamped to live RAM and skips are NOT_RUN (Windows probe requires the five conservation node IDs PASSED).
- Rework-2 local evidence: registry 23 passed; probe guards 10 passed; zero-support producer 20 passed/1 skipped; combined M1/M0 54 passed/6 skipped (K>0 conservation NOT_RUN on this low-memory host); focused GPU/CPU/assembly 218 passed/7 skipped. Candidate live no-pressure PASS (science 3772b6cd…, coverage 49ed57f7…); baseline live run NOT_RUN at 845,000,704 B / 825,798,656 B → A/B NOT_RUN, no inflation. Windows M1 NOT_RUN.
- Report .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.coco.r2.md. No commit/push; M1 not accepted by Coco. Next: Junior replays invariant probes/tests and exercises K>0 conservation on an adequate host; then Nono confirmation. No M2.

## M1 REWORK-1 active — Junior disposition of Nono review-0
- Nono FINDINGS received; Junior confirms F1/F2/F3/F5 from executed probes + source, all 8 review-snapshot hashes unchanged.
- F4 narrowed: fix actual CooperativeStopChannel spawn transport; registry remains process-local, no manager/shared payload requirement. F6 definite CPU acceptance error lacks registry handle; AdmissionExhausted variant unproven. Progress-before-accept and new bench boundaries need correction.
- First corrective iteration dispatched to SAME agent:coco:main, no reset. Accepted/queued run 87607dc6-6dc6-46a5-9562-649bb4dfce88.
- Brief .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.rework1-brief.md; expected durable report .coco.r1.md, exact-source callback. Await callback without polling.
- Scope: schema/status/generation validation, allocation-aware retained bytes, real spawn stop test, CPU failure ownership/GPU progress and error coverage, honest fresh bounded bench/no RAM inflation. No M2/commit/push/install edits. Windows M1 NOT_RUN, M1 not accepted. No user intervention needed yet.

## Active mission — M1 (2026-10-27)

Tristan authorized M1 implementation and asks Junior to run bench checks when possible, requesting manual Windows execution only when needed.
- Mission: ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.
- Coco session previous mission closed; reset RPC timed out, one history read confirmed fresh Reset-only transcript seq1. No redispatch/reset loop.
- Dispatch accepted/queued, run ac0889f6-144e-45b3-a459-ad935ac76dcb, timeoutSeconds=0.
- Baseline HEAD231b597867b4d251afac6b767000c8a37e2cf3b0, feat/zegrid-resilient-run-resume. Existing untracked M0 artifacts preserved.
- Scope: runtime registry/atomic acceptance, sparse IDs, payload preservation, ownership/generations/cancellation groundwork, focused tests and M1 bench. No full disk resume, no unqualified GPU/CPU reuse, no M31 or installed-runtime changes, no commit/push.
- Brief: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.brief.md.
- Coco implemented M1 in the worktree; durable report written to .a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M1-20261027.coco.r0.md and callback emitted. Status pending Junior independent verification + Nono review; M1 NOT accepted by Coco.
- M1 deliverable: `core/zegrid/run_registry.py` (run-owned sparse registry, validated atomic accept against the real producer schema, complete core+halo-patch+reference-pin fingerprints, status-aware duplicate identity, future/stale generation fencing, honest unique-buffer accounting, bounded journal, spawn-safe stop endpoint, explicit disposal) + coordinator seams (GPU receipts, CPU wave acceptance, conserved refusal after K>0 with category, stop checks, success disposal/manifest snapshot).
- REWORK-1 (2026-10-27): Nono F1–F5 corrected; F4 limited to the stop-channel transport per Junior disposition (registry stays process-local). CPU acceptance failure now carries the registry handle; GPU K>0 OOM/internal-acceptance is classified distinctly; GPU progress counts only non-idempotent acceptance; no-pressure bench no longer inflates RAM; Windows probe rewritten with NEW zgprobe-only output, basetemp/cache confinement, explicit Python, live artifact cap, timeout and source provenance.
- Evidence (rework-1 final): M1 registry 20 + real conservation 5 + probe guards 6 + M0 witness/harness 21 = **52 passed, 64.18s**. R55-E1 checkpoint file **42 passed, 144.95s**. Focused GPU/CPU/assembly regression **218 passed, 7 skipped, 364.16s**. No-pressure live-memory A/B **NOT_RUN** (live 559,968,256 B; 2x2 needs 781.1 MiB vs 441.6 MiB budget) — no synthetic inflation. Windows probe **NOT_RUN** locally.
- Historical counts: M0 was 21 tests; the pre-rework combined M1+M0 run was 33 tests (not 32).
- No commit/push. Next: Junior replays invariant probes/tests and Nono confirms; then M2 durable store.

## Current status — Windows witness verified (2026-10-27)

- M0 baseline reproduction ACCEPTED on real Windows GPU. Tristan ran the harness; Junior read witness.json and child.log remotely and checked the supplied parent summary.
- RTX 3070 Laptop GPU, CuPy 14.1.1; 2 complete cells with backend_used=gpu and recorded core hashes. R55_REFUSE_HOST_RAM then discarded=true for both, CPU AdmissionExhausted (843255808 > 660602880 bytes). All 15 witness checks true; no OOM; zero leases/staging leftovers; no science publication.
- Parent PASS / EXPECTED_BASELINE_FAILURE, exit 0, 61.092 s, no timeout/cap breach, 8567842 artifact bytes. Main module and listed dependencies match installed runtime, no override.
- Evidence: C:\Users\JarvisJunior\Desktop\zgprobe\m0_witness_out (witness.json, run_summary.json, child.log, out/zegrid_gpu_failure.json).
- Scope: synthetic available-memory drop, not physical exhaustion. Small 8-frame corpus, tile 409 (M31 used tile 32). Confirms the failure mechanism, not full M31 equivalence, science parity, recovery or durable resume.
- Sufficient evidence to design/implement M1 runtime result conservation; no further broad baseline campaign needed first. Acceptance must test preserved cell IDs/payload hashes, no duplicate work/counting, safe buffer ownership/late-result handling. Cross-backend reuse remains disallowed until parity is qualified.
- M1 NOT launched in this evaluation turn. Later gates remain physical-pressure/RSS, durable save-stop-resume and scientific comparison, assembly/scaling, Windows GUI. No M31 relaunch, product change, commit or push.

Earlier NOT_RUN and pending-review entries below are historical and superseded by this status.

2026-10-27. Tristan authorized starting the short validation task after plan
review2. Mission `ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027`.
Baseline fresh origin/beta `231b597867b4d251afac6b767000c8a37e2cf3b0`.
Branch `feat/zegrid-resilient-run-resume`, isolated worktree.
No M31 restart; no implemented resume claimed; no commit/push/deploy.

## M0 REWORK-1 status (Nono review-0 F1–F7)

- [x] F1 provenance equality enforced in parent + child (actual imported module
      and key dependency hashes); override is explicitly non-qualifying.
- [x] F2 parent fail-closed: returncode==0 AND witness PASS AND
      EXPECTED_BASELINE_FAILURE AND no cap breach; nonzero child exit wins.
- [x] F3 full witness predicate + conditional `gpu_qualified` (real Windows GPU
      + backend_used=gpu + all checks + verified provenance only).
- [x] F4 injected high clamped by live RAM; mode derived
      (LOCAL_CONTROL_CPU / WINDOWS_REAL_GPU / WINDOWS_NO_GPU).
- [x] F5 4 B/px conservative estimate, streamed child log, live cap watcher,
      coherent CAP_EXCEEDED/exit 4 (never PASS).
- [x] F6 harness unit tests (provenance, fail-closed, cap, timeout, NOT_RUN,
      predicate, qualification); local witness second-run terminal outcome.
      No-pressure real-memory success control is out of scope on this host:
      live RAM (~0.6–0.8 GiB) is below the product PLAN budget, so it refuses
      before any cell. No gpu_qualified claim is made.
- [x] F7 README + docs: control-only CPU label, 1 GiB wording, tile 409 vs
      M31 tile 32 explanation.
- [x] Local control-only witness re-run (see report r1).
- [x] Local NOT_RUN smoke re-run: exit 2, mode LOCAL_CONTROL_CPU,
      gpu_qualified=false, provenance_verified=false, stage=gpu_readiness.
- [x] M0 tests: 21 passed (1 control witness + 20 harness) in 19.18s.
- [ ] Windows real-GPU run (still NOT_RUN on this host; Junior must run it).
- [ ] Junior/Nono re-review; M0 acceptance.

## M0 status — evidence first (r0)

- [x] Baseline focused tests run: 145 passed in 293.80 s (5 focused files; no
      full suite, no long benchmark).
- [x] Baseline failure witness RUN on the REAL orchestration path with real
      floor-respecting FITS and real cell computations:
      `tests/test_zegrid_resume_m0_baseline_witness.py`.
      Result: 2 completed cells, synthetic memory drop, `_R55GpuFallback`
      (`R55_REFUSE_HOST_RAM`), completed results `discarded=true`,
      `AdmissionExhausted` on CPU re-admission, no OOM, no FITS/manifest.
      Mode is **CONTROL_ONLY_FAKE_GPU** (device executor substituted by the
      real CPU canonical executor); this is not GPU qualification.
- [x] Determinism: two consecutive runs produced identical cell IDs and core
      SHA-256 (wall times excluded).
- [x] Windows REAL-GPU witness script + PowerShell launcher prepared under
      `tools/zegrid_resume_m0/`, with fresh `zgprobe` subfolder, installed
      module SHA-256 provenance, ≤300 s owned-process timeout, ≤1 GiB new
      artifacts, stage labelling, expected-vs-unexpected classification.
      **NOT_RUN on this host** (no CuPy); no GPU proof.
- [x] Contracts/inventory documented in
      `docs/refactor/ZEGRID_RESILIENT_RUN_RESUME_M0_20261027.md`
      (producers/consumers, checkpoint fields, state machines, generations,
      sparse sets, ownership, commit-before-release, cancellation, per-phase
      resource envelope, T0–T4 matrix, M1–M4 vertical slice).
- [x] Plan document preserved unchanged in this worktree; original main-project
      plan untouched. No code correction to the plan was substantiated in M0.
- [ ] Junior independent verification of the witness/harness.
- [ ] Nono review.
- [ ] M1 implementation (not started; M0 does not authorize it).

## Known limits / decisions

- Real GPU/Windows witness NOT_RUN locally. Junior must run it after inspecting
  the harness; it refuses to run without installed-module provenance unless
  explicitly labelled unverified.
- Two decisions await Junior: (1) checkpoint storage format (recommended
  directory-per-run/cell + strict JSON + separate arrays, no pickle),
  (2) mixed CPU/GPU resume policy (recommended: no mixing without qualified
  parity; preserve GPU commits and wait for admissible GPU).
- Assembly/finishing peak RAM is not yet a first-class measured envelope row;
  M5 must measure it before a large run.
- `AdmissionExhausted` structured aggregate/budget attributes are missing; M0
  records the parsed message and recommends structured fields in M2.

## Next step

Junior verifies the M0 witness + contracts and runs the Windows harness; Nono
reviews. Only then decide whether to start M1. No commit/push authorized.

## Junior r0 verification

- [x] Local witness replay: 1 passed / 36 warnings / 21.16 s (control-only).
- [ ] M0 acceptance: pending independent review and harness corrections.
- [ ] Windows execution: withheld; provenance equality not enforced, weak PASS predicates, nonzero-child exit ignored, resource cap only post-run.
- Junior report: `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027.junior.r0.md`.
- Routine architecture choices resolved: strict JSON + separate numeric array checkpoint files with atomic commit marker; no cross-backend reuse before parity qualification. No additional human gate needed for these choices.

## Junior r0 verification + Nono review-0

- [x] Junior replay local witness: 1 passed / 21.16 s (control-only CPU).
- [x] Nono review-0 = FINDINGS (4 HIGH, 2 MEDIUM, 2 LOW): harness not fail-closed.
- [x] Junior confirmed F1-F5 independently (provenance not enforced; parent PASS on nonzero child exit; weak child predicate + gpu_qualified unconditionally true; RAW_HIGH 16GiB no live clamp + hardcoded mode; post-hoc artifact cap + 2 B/px estimate + stdout in RAM).
- [x] REWORK-1 dispatched to same Coco session (run 9264fbce). Scope: harness/tests/docs only; F1-F7; no product edit, no Windows run, no push.
- [ ] REWORK-1 verification + Nono follow-up review.
- [ ] Windows real-GPU execution (withheld until harness fail-closed).

## Junior REWORK-1 verification

- [x] Replay 21 tests (witness+harness) 19.58s; fixes read and confirmed fail-closed.
- [ ] Nono review-1 confirmation.
- [ ] M0 acceptance.
- [ ] Windows real-GPU witness run.

## M0 ACCEPTED

- [x] Nono review-1 = ACCEPT (F1-F7 closed; 1 MEDIUM label provenance sans --installed-src-root; 3 LOW).
- [x] M0 accepted for declared scope (short baseline failure witness + contracts).
- [ ] Windows real-GPU witness run (provenance complete via --installed-src-root).
- [ ] M1 (runtime conservation) — NOT started; separate mission, needs Tristan go.

## M0 Windows harness deployment

- [x] Harness + launcher copied to C:\\Users\\JarvisJunior\\Desktop\\zgprobe\\m0_witness_tool (py 37948 B SHA 8426f9bb, ps1 2324 B); py_compile OK; psutil 7.2.2.
- [x] Installed runtime = ZeAlfie slot rt-54e8c7a8ba26 (module SHA 8ccc629 = beta 231b597 LF).
- [ ] Real-GPU witness run (Tristan foreground; node cannot hold 300 s).

## M2 — durable store (implementation, 2026-10-27)

- [x] `run_store.py`: frozen run identity/jauge, atomic per-cell commits, reconstructible index, single-writer OS lock (owner = pid + start token + boot id), sha256 integrity, disk budget with reservation/margin, sparse completeness, idempotence by `(cell_id, attempt_id)`.
- [x] M1 registry: additive `release_payload` (idempotent, refcounted, digest-checked) + `PayloadReleasedError`; no M1 semantic weakened.
- [x] Product seam: CPU accept → durable commit → RAM release; store failure degrades to M1 conservation (no release, no durable claim); assembly reloads released cells from the verified store; publication/science/budgets/floors untouched.
- [x] Bench `tools/zegrid_resume_m2/` (fail-closed; skip ⇒ NOT_RUN).
- [x] Linux: 39 store-contract + 20 kill-matrix cases pass; M1 (23+4) and M0 (1+20) non-regression pass.
- [ ] NOT_RUN here: store-enabled real CPU run (2 cases) and M1 real-orchestration (6 cases) — host below the 1,200 MiB corpus floor; **all Windows execution** (bench shipped).
- [ ] Junior verification + Nono review. Coco does not accept M2.

## M2 — CURRENT state after REWORK-3 (supersedes the earlier M2 line above)

- Active milestone: M2 (durable store) — implemented, **not accepted**.
- Applied: atomic per-cell commits, reconstructible index, single-writer OS lock
  (`LOCK` dedicated file + separate `OWNER.json`), sha256 integrity, disk budget,
  sparse completeness, idempotent `(cell_id, attempt_id)` publication, RAM
  release after commit; REWORK-1 (`O_BINARY` at every open, bounded dead-owner
  lock wait); REWORK-2 (lock/owner split, `Mapping` import, open-stage
  diagnostics); REWORK-3 (identity hashes `source_path` while keeping the
  portable logical id; faithful lock/PID/binary-site tests; ranked bench
  diagnostics).
- Correction: the earlier r2 statement "tests no longer read LOCK at all" was
  wrong; two direct reads remained and are now removed.
- Linux evidence: store contract 39, I/O 8, kill 20, identity seam 6, bench
  diagnostics 5, M1 registry 23, M0 witness/harness 21 — 0 failures. The two
  real-run store cases and 6 M1 conservation cases remain **NOT_RUN** on this
  host (RAM floor), never qualified.
- Pending gates: short Windows focused proofs on the candidate commit, then the
  full Windows bench (`exit 0`, `status=PASS`, zero skip); Nono review; Junior
  acceptance. No Windows PASS claim exists.

### Junior M5-A r0 — independent gate (2026-10-28)

- NOT ACCEPTED pending correction/review: 129 tests passed, 5 skipped; connected 2x2 exact parity passed.
- Confirmed: no production capacity preflight before assembly; actual mapped-cell peak 3 vs reported cap 2, and eager 4-payload retention passes the guard. Hash/transient pricing also needs review.
- Nono review-0 asynchronously accepted; durable Junior evidence in `.a2a-reports/ZM-ZEGRID-RESILIENT-RUN-RESUME-M5A-20261028.junior.r0.md` (workspace). Snapshot preceded this orchestration-only note.
- Next: verify Nono findings, bounded REWORK-1 to same Coco mission session. M5-B not started; real-pressure and Windows gates unchanged. No commit/push.
