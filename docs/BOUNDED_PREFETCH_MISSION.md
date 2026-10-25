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
