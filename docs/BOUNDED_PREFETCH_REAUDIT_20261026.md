# Bounded prefetch — independent publication audit, 2026-10-26

## Decision: BLOCKED — do not push

This audit supersedes the earlier statement that the architectural correction
was complete and functional with only Windows runtime evidence outstanding.
Tristan authorized beta publication only if the re-audit confirmed readiness;
that condition is NOT satisfied. No push, merge, tag or implementation change.

- Mission: ZM-ZEGRID-BOUNDED-PREFETCH-20261025
- Candidate: perf/zegrid-bounded-prefetch @ 873abb1b333beee19ba1bf9d6fe2ff1452b13878
- Remote beta, freshly fetched: 3fa088cf3409f7d19978e28dee171df4bc50bfce
- Candidate is a descendant of remote beta, seven unpublished commits.
- Candidate worktree was clean before this audit documentation was added.
- The separate main checkout's local beta is 48a0b65, NOT the current remote
  beta; do not conflate them or reset that checkout/untracked user files.

## Independently verified positives and limits

The seven-module order (R56 bounded credit, R56 rework1, E2, E5, E7, E8, E9)
passes: **147 passed, 1 skipped in 79.64s**, using the project's Python venv.
The faithful LP341 model witness is included: tile192/p2/workers14,
aggregate 14,016,615,629 bytes; this remains MODEL evidence, not measured
Windows runtime capacity or performance.

The persisted 18-module result is Coco's **426 passed / 3 skipped**, not a
new independent full-matrix run. This audit did not rerun it after discovering
reproducible release blockers. `git diff --check origin/beta...HEAD` passed.

The real-memory Linux CUDA artifact explicitly has `all_passed=false`,
`candidate_gpu_prefetch_ge1=false`, `candidate_overlap_proved=false`, with
scientific equality true. No new real GPU campaign was performed in this audit.
The old pinned-memory artifact, where even control p0 overlapped, is NOT proof
for the corrected p0-serial contract. No measured LP341 speedup is established.

## F1 — HIGH: admission shrink lost in final GPU candidate

Source: src/zemosaic/zemosaic_zegrid_mode.py, especially lines 11171-11175 and
11687-11708 at the audited commit.

The runner receives `gpu_candidate` from admission but constructs effective
candidates from `gpu_decision.plan_set.chosen` (the original PLAN candidate).
When admission reduces p2 to p0 and runtime tile does not change, its final
mapping stays p0 but the final candidate reverts to p2. Final manifest hash
validation fails with `ActivePlanRejected: R55_INTERNAL_ERROR`.
A runtime tile shrink also starts from the original chosen candidate, so the
same family of errors can restore stale resource fields; this second path
requires dedicated regression coverage rather than assumed equivalence.

Reproduction: reuse the existing E2 routine-corpus integration test, including
its CPU-backed GPU simulation; inject admission-only host pressure with the
real admission algorithm. Raw=1,019,115,909 bytes; admission performs two
shrinks, tile417/workers6/p2 -> tile417/workers6/p0. The product then fails at
`active_final_plan_hash`, expected prefetch2 versus actual prefetch0. This is a
real product control-flow integration test with simulated GPU and injected RAM,
NOT a physical GPU/OOM test.

The stale-candidate code is also present in origin/beta, so it is pre-existing,
not attributed as introduced by the night commits. Nevertheless it invalidates
the prior closure claim. Late-binding the memory observer fixes the test seam,
not this real pressure-handling defect. The new seam test merely inspects the
default and invokes the patched observer directly; it never exercises this
admission-to-finalization path.

## F2 — HIGH: partial-produce cleanup receives the wrong task shape

Source: zemosaic_zegrid_mode.py lines 11476-11503.

The pipeline gets `tasks=indexed` (each item is `(index, task)`); production
unwraps `item[1]`, but `_cleanup_task` directly tests `len(task)>=4` and reads
`task[3]`. It therefore silently does nothing on the two-element indexed item.
If cache production raises after writing a file, credit release still occurs.

Reproduction: compile the EXACT nested callback from the product AST and pass
it to the real BoundedPipeline with the product's indexed task shape. An
injected producer writes a real partial file then raises. The file survives,
while host/disk used bytes become zero and ledger `balanced=true`.
This is an isolated exact-product-callback/pipeline reproduction, not a full
mosaic integration run; the wiring mismatch was independently inspected in the
product source. Existing generic cleanup tests use their own callback and do
not cover this adapter.

## Correction plan (prepared, not dispatched)

1. **F1: preserve admitted execution identity.** Use the admitted/current
   candidate as the authority throughout runtime; derive all tile reductions
   from it, preserving reduced prefetch/workers/K. Keep mapping, candidate,
   envelope and final hash coherent. Never weaken hash assertions or restore
   resources when RAM/VRAM recovers.
2. **F2: correct indexed task decoding.** Unwrap the pipeline item before
   cleanup; operate only on the failed cell's cache. Preserve the primary
   exception, report cleanup failure explicitly, and do not pretend that disk
   bytes are removed when deletion fails. Exercise the actual product wiring
   with a real partial file and verify cleanup precedes credit release.
3. **Regression gates.** Turn both audit probes into maintained regressions;
   add admission p2->p0 with unchanged runtime tile, admission shrink followed
   by runtime tile shrink, recovery-without-regrowth, cancellation/partial
   production and cleanup-failure cases. Keep science, frame eligibility,
   halo128, host80/90, GPU70/85, legacy50 and single GPU owner unchanged.
4. **Validation and independent review.** Run targeted new regressions, the
   seven-module order, the documented 18-module matrix and project's fast gate.
   Have Nono review the exact final diff read-only; Junior independently checks
   material findings and accepts only after blockers are resolved. No long
   performance campaign is needed to prove these control-flow fixes.
5. **Evidence/documentation cleanup.** Supersede stale r0/intermediate claims
   in BOUNDED_PREFETCH_MISSION.md and measurements README. Clearly separate
   historical pinned runs, current model evidence, real p0 smoke and Windows
   p>=1 NOT_RUN. Record exact commands/counts and environment for each run.
6. **Publication gate.** Recheck fresh remote beta and exact candidate range.
   No force push, no main update, no unrelated local-beta changes. Only a
   verified candidate may be promoted within Tristan's conditional mandate.
   Then Tristan's Windows LP341 halo128 test must measure actual p>=1 overlap,
   correct completion, RAM/VRAM, and scientific parity; no promised speedup.

## Durable evidence

Directory: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-R56-AUDIT-20261026/

- test_review_probes.py: two reproducible reviewer probes, no product edits.
- probes.log: **2 failed in 26.39s**, each at the intended defect assertion.
- seven-modules.log: **147 passed, 1 skipped**.

Probe command (from candidate worktree):

```sh
PYTHONPATH=src:tests /home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python -m pytest -q -s /home/tristan/.openclaw/workspace/.a2a-reports/ZM-R56-AUDIT-20261026/test_review_probes.py
```

## Follow-up — correction authorized, 2026-10-26

Tristan explicitly asked to launch this correction plan. ARCH REWORK-1 is
prepared for the existing Coco mission session, same mission id, baseline
873abb1. Scope: F1/F2, product regressions, evidence cleanup and validation.
Publication remains BLOCKED pending implementation, Junior verification and
Nono review. No scientific/budget policy changes are authorized by this step.
Durable brief: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-BOUNDED-PREFETCH-20261025.arch-rework1-brief.md
Expected report: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-BOUNDED-PREFETCH-20261025.coco.arch-r1.md

## Follow-up — ARCH REWORK-1 implementation, 2026-10-26

The two reproduced blockers are fixed at HEAD `4c703d9` (on `873abb1`), same
mission, local branch only. F1 preserves the admitted candidate/mapping through
runtime tile reduction and finalization (no more p2->p0 loss); F2 decodes the
indexed pipeline task, deletes only the failed cell's cache BEFORE credit
release, and reports failed deletions instead of claiming success. Maintained
regressions cover both, including the runtime-tile-shrink variant and
real-partial-file cleanup. Reviewer probes: 2 passed. Seven-module: 149/1.
18-module: 430/1. The project fast gate (1200 s) times out at 49 % with two
order-dependent `test_zegrid_r18_final_finishing.py` failures that pass alone
and reproduce on a clean `873abb1` archive in the same order: pre-existing, not
caused by F1/F2. Publication remains BLOCKED pending Junior verification and
Nono review; no push/merge/tag/install. Historical pinned-4 GiB evidence remains
superseded and is not p0 control proof.

## Follow-up — ARCH REWORK-2, 2026-10-26

At HEAD `ba38097` (on `303a6c2`) the Nono MEDIUM is corrected: the post-shrink
shrink-history and RUN-check projection now use the effective p0 mapping/bundle
with an explicit `prefetch_units`; the tile-384 witness publishes 793 183 232 B
instead of the stale 1 228 748 400 B. Maintained regressions cover the full
p2-admission -> p0 -> runtime tile-shrink path, including history/checkpoint
projection, K1/workers preservation and exact science parity. The external
post-shrink probe passes unchanged.

Fast gate: the R18 finishing failures were an ambient host-RAM dependency,
reproduced at low RAM on beta 3fa088c and on the candidate, fixed by a test-only
12 GiB host pin. The fast gate still times out at 51 % (1200 s) with a
pre-existing R37 diagnostic-capture host-pressure cluster (176 passed/47 errors
alone; 7/7 error reproducer on beta 3fa088c). Publication-gate residual is
explicit; no all-green claim, no push.
