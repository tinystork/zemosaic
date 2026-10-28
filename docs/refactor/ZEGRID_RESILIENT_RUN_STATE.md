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
