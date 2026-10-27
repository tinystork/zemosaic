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
