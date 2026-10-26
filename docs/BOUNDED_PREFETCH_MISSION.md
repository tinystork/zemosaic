# Bounded prefetch — ZM-ZEGRID-BOUNDED-PREFETCH-20261025

Status: implementation r0 complete; awaiting Junior verification, then Nono review.
Baseline: 3fa088cf3409f7d19978e28dee171df4bc50bfce (branch `perf/zegrid-bounded-prefetch`).
Objective: profile-aware coherent host/disk credit with prefetch>=1, real
producer/consumer overlap, before/after sampled process-tree RSS and VRAM on
LP341 halo128 sky_mean. Existing historical memory data are baseline context,
not continuous runtime peak proof.

Full brief: /home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-BOUNDED-PREFETCH-20261025.brief.md

## Changed contract (implemented)

The prefetch credit used to conflate two resources in ONE number:
`per_unit_prefetch_bytes == per_unit_cache` (on-disk aligned-cache bytes) was
billed against the RESIDENT HOST budget. That is only true for the DENSE
provider (2N live memmaps). Under the BOUNDED provider no mapping outlives an
access, so a queued slot retains metadata plus at most one transient owned copy.

* `_R55_PREFETCH_CREDIT_SCHEMA = "r56-bounded-prefetch-credit-v1"`.
* `_r55_prefetch_resident_bytes(...)` derives the RESIDENT host credit:
  dense -> `per_unit_cache` (byte-identical to the old contract);
  bounded -> `worst_patch_px * (channels*4 + 1)` (one worst-patch RGB float32
  plane + its bool support plane). Never zero, never a guessed tile slice.
* `PlanContext.per_unit_prefetch_disk_bytes` (optional) carries the EXACT
  on-disk bytes per prefetch slot. `None` keeps the historical single-number
  semantics byte for byte, so every existing plan/hash is unchanged.
  `disk_residency_bytes` uses the disk field; `host_peak` uses the host field.
* RUN: `BufferLedger` gained an independent `disk` dimension
  (`PIPELINE_SCHEMA_V2 = "zemosaic-r56-bounded-pipeline-v2"`). The producer
  reserves host staging bytes and on-disk bytes separately; `balanced` covers
  all three dimensions. Disk budget is half the free volume at PLAN time and is
  re-declared at RUN; an impossible request fails closed.
* Dense keeps the historical host credit (`manifest.total_bytes`), so dense
  plans and hashes are byte-identical.

## Overlap proof (implemented)

`BoundedPipeline` records the `(start, end)` interval of each `produce()` and
`consume()` CALL on one monotonic clock and publishes `observed_overlap`:
`producer_work_s`, `consumer_work_s`, `wall_s`, `intersection_s`, `overlap_units`,
`overlap_proved`. A nonzero intersection means both sides performed work
concurrently. It is NOT a queue-capacity or thread-count claim, and no CUDA
kernel overlap is inferred from an enclosing CPU function duration. The RUN
block publishes the same witness under `run.overlap`.

## Measurement runner

`tools/zegrid_r56_prefetch_campaign.py` — one instrumented runner, two arms
(`control` forces a SHRINK-ONLY prefetch cap of 0, `candidate` uses the planner).
`--available-bytes` pins the RAM observation for both arms so the PLAN is
identical and only the run depth differs (recorded in the output).
It samples process-tree RSS via psutil (portable, never /proc) and device VRAM via
`cupy.cuda.runtime.memGetInfo` plus the CuPy default pool, with period/source/
caveats, and keeps reservations separate from observations. Unavailable values
are `null` with a reason; zeros are never fabricated.

Recorded artifact and interpretation:
`docs/measurements/ZM-ZEGRID-BOUNDED-PREFETCH-20261025/`.

## Validation executed

See the mission report `.a2a-reports/ZM-ZEGRID-BOUNDED-PREFETCH-20261025.coco.r0.md`
for the exact commands and results. Summary:

* new gate module `tests/test_zegrid_r56_bounded_prefetch_credit.py` (LP341
  numbers, profile-aware credit, disk separation, disk refusal, ledger, overlap);
* product RUN coherence assertion added to the E2 module;
* cross-test `_STACK_STORAGE_PROFILE` leakage fixed in `tests/conftest.py`
  (autouse snapshot/restore) — expectations unchanged;
* R55 E2/E5/E7/E8/E9/E10/E11 + R52 fallback/storage and GPU admission modules run.

## Outstanding gates

* LP341 halo128 sky_mean before/after on a capable Windows node: **NOT_RUN**
  (no capable node/tool here; this host cannot build the halo128 LP geometry).
* Full LP runtime overlap: **NOT_RUN**. Planning-level prefetch>=1 is proved.
* The Linux MX150 halo64 smoke proves real overlap and identical science; it is
  NOT Windows proof and NOT an LP-scale memory win.

## Next step

Junior verifies diff/tests/artifacts, then requests Nono read-only review. If a
capable Windows node is later available, run the recorder with `--halo 128` and
the LP341 inputs for the before/after measurement.

## REWORK-1 (2026-10-26) — corrected contract

Starting HEAD 4c2ae77 -> 1b0faaa (local). Findings F1-F4 from Junior/Nono review-0.

Checklist:
- [x] F1 pre-acquisition BEFORE produce (predicted host/disk lease, reconcile down,
      underestimate fails closed, partial-produce/cancel release once, balanced).
- [x] F2 simultaneous ownership: sequential producer phases -> MAX(cache build,
      cell normalization); initial global gauge NOT added; shared counted once;
      concurrent_producer_bytes in scheduler host peak + envelope estimate;
      GPU envelope aggregate_rule = simultaneous_stack_plus_cache_minus_shared.
- [x] F3 campaign runner v2: fresh subprocess per arm, refuse occupied dirs,
      downward-only RAM clamp, honest uncontrolled-OS-cache label, correct
      pinned-pool bytes, non-null science/participation, candidate-only
      GPU/prefetch>=1/overlap gates, exit status follows gates.
- [x] F4 fresh disk observation before staging, own retained bytes added back,
      half-free policy, SHRINK-ONLY budget, refuse before production.
- [ ] Clean combined matrix: 438 passed, 1 skipped, 2 failed (E2 OOM + RUN
      coherence). Cause unproven; leading hypothesis = RUN admission reads the
      real 1.2-1.8 GiB host RAM. NOT hidden, not worked around.
- [ ] Corrected A/B campaign executed end-to-end: NOT_RUN on this host.
- [ ] LP341 H128 sky_mean runtime / Windows: NOT_RUN.
- [ ] Re-derive LP341 prefetch>=1 under the F2 model at the real budget.

Contract change is user-visible: tight budgets can now shrink tile/prefetch or
refuse the GPU instead of admitting a max-only plan (E8 gate B now refuses).
That is the intended honest outcome pending Junior's decision.

## REWORK-2 (2026-10-26) — partial, honest checklist

- [x] A: disk liveness — host credit first, fresh disk observation AFTER the
      host wait, temporary own saturation WAITS (cancellable) instead of a
      terminal refusal, structural/impossible single request refuses, observation
      failure is a structured refusal, explicit `disk_enabled` distinguishes
      legacy-omitted from enabled-zero, `cleanup_task` runs BEFORE credit release.
- [x] B(partial): `concurrent_producer_bytes` recomputed per TRIAL worker count;
      early GPU feasibility now probes the whole worker ladder before refusing.
- [ ] B(remaining): unify candidate vs apply/runtime projection (p0 serial under
      pre-acquisition), faithful LP341 product wiring witness (current helper
      still omits the producer term).
- [x] C(partial): recorder accepts the REAL manifest schema (complete_cells list,
      incomplete_cells []), downward-only FRESH RAM clamp with observer restore,
      geometry-only comparability.
- [ ] C(remaining): adapter integration test from a real manifest, final-backend /
      consumed-cells gate, real feasible small bounded CUDA run.
- [ ] D: combined-matrix failure root cause NOT established.
- [ ] New required adversarial tests (ordinary full-slot wait, external decline,
      shrink during wait, cancelled wait, zero enabled, real-file partial cleanup,
      real-schema recorder adapter) NOT yet added.

## ARCH iteration (2026-10-26, HEAD ffe75bc)

The three PLAN/ADMIT/RUN host projections are now ONE published function
(`ram_contract.projected_host_peak_bytes`):

```
shared + stack_added + (producer_owner if prefetch >= 1 else 0)
       + prefetch * resident_per_slot
```

* `aggregate_rule = "projected_host_peak_v1"` is published on the ACTIVE GPU
  envelope together with `prefetch_units`, `producer_owner_bytes` and
  `prefetch_resident_per_slot_bytes`;
* the scheduler CPU+GPU candidate, the GPU envelope aggregate and the RUN tile
  revalidation all call it, so the three phases publish the SAME value;
* at `prefetch == 0` the queue is full while the consumer stacks, the producer
  is BLOCKED and is NOT charged (serial plan). At `p >= 1` exactly ONE producer
  owner plus the resident prefetch credit are charged;
* witnesses E7/E8/E9/E11 and the R56 F2 scheduler tests were reconciled in one
  pass to this truth (p0 plans became cheaper, so E7/E8's largest rung moved
  384 -> 448 and the Linux H16 ACTIVE plan is now ADMITTED at p0 instead of
  being refused by an unconditional producer charge). Every delta is semantic;
  no budget/policy (80/90, 70/85, legacy 50, K1, shrink-only, science, halo)
  was relaxed.

Faithful LP341 witness (`tests/test_zegrid_r56_bounded_prefetch_credit.py`):
REAL product decision, published facts -> tile 192 / p2 / workers 14,
aggregate 14 016 615 629 B < B_plan 14 687 984 025 B, limited by the gauge
phase. MODEL result, not a runtime LP341 run.

Real-memory bounded CUDA smoke (H64, MX150): GPU executed 6/6 cells with
identical outputs, but the real ~2.27 GB observation admits `prefetch=0` only,
so there is no overlap claim. See `docs/measurements/.../README.md`.

Combined 18-module matrix at this HEAD: 426 passed, 3 skipped
(`matrix-18-modules-final.txt`). The two r1 E2 reds DID reproduce in a 7-module
order and are now root-caused and fixed: `_r55_active_gpu_admit` captured
`available_memory_bytes` as an import-time default, bypassing the tests' 12 GiB
pin; the admission then observed the REAL host, shrank prefetch 2 -> 0 on a
low-RAM moment, and the final plan hash rejected the shrunk mapping against the
unshrunk candidate. The observer is now late-bound at call time (same product
behaviour) and a regression guard pins the seam.

## ARCH REWORK-1 (2026-10-26) — F1/F2 fixed, HEAD 4c703d9

Two publication blockers reproduced by the independent audit are fixed with
maintained product regressions:

1. **F1 (admitted identity).** `_run_r55_active_gpu_units` now uses the ADMITTED
   candidate/mapping throughout (runtime tile shrink materialized from the
   current candidate; finalization uses the current candidate + bundle). The
   previous code rebuilt from `plan_set.chosen`, so admission p2->p0 was lost at
   finalization and the exact-mapping hash rejected it. Regressions cover
   admission p2->p0 with no tile change and admission shrink followed by a
   runtime tile shrink (reduced prefetch/workers/K preserved, no regrowth).
2. **F2 (cleanup adapter).** The nested cleanup callback now unwraps the indexed
   `(index, cell_task)` pipeline item, deletes ONLY the failed cell's cache
   BEFORE credit release, and raises a reported failure if `_safe_rmtree`
   returns False. Regressions use a real partial file, observe the disk credit
   still held during deletion, and cover failure residue + cancellation.

Validation: reviewer probes 2 passed; seven-module 149 passed/1 skipped;
18-module 430 passed/1 skipped. The project fast gate (1200 s) times out at 49 %
with two order-dependent `test_zegrid_r18_final_finishing.py` failures; both
pass alone and are reproduced identically on a clean `873abb1` archive in the
same order, so they are PRE-EXISTING and not caused by this work. No push,
merge, beta/main change, tag, install or GUI change.

Evidence corrections: the historical pinned-4 GiB smoke is superseded and is
NOT p0 control proof; LP341 p2 is MODEL; the real H64 CUDA run is p0 only;
Windows halo128 overlap is NOT_RUN. Logs are in
`docs/measurements/ZM-ZEGRID-BOUNDED-PREFETCH-20261025/`.
