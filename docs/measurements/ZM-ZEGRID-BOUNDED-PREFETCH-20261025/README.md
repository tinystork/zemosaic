# ZM-ZEGRID-BOUNDED-PREFETCH-20261025 — measured artifacts

Owner: Coco (implementation). Acceptance: Junior after verification, then Nono review.

## What is in here

`linux-cuda-smoke-halo64-pinned.json` — output of the common instrumented runner
`tools/zegrid_r56_prefetch_campaign.py` on the ONLY GPU available to this
worktree (NVIDIA GeForce MX150, 2 GiB, Linux). It is a SHORT bounded CUDA smoke,
NOT the LP341 corpus and NOT a Windows run.

Reproduce:

```
PYTHONPATH=src python tools/zegrid_r56_prefetch_campaign.py \
  --input <12x768px synthetic corpus> --out <scratch> \
  --halo 64 --gpu --storage-profile bounded \
  --available-bytes 4294967296 --sample-period 0.1
```

`--available-bytes` PINS the host-RAM observation for BOTH arms. Without it the
planner re-observes RAM per run; on this host (1.2–1.8 GiB available, the
OpenClaw gateway itself holds ~1.6 GiB RSS) the admitted plan drifted between
arms, which makes an A/B meaningless. The pin is recorded in the artifact as
`memory_observation_pinned_bytes`; RSS/VRAM remain real observations.

## Observed result (this artifact)

| quantity | control (cap 0) | candidate (planner) |
|---|---|---|
| PLAN prefetch | 2 | 2 |
| RUN prefetch (effective) | 0 | 2 |
| queue_max | 1 | 3 |
| host credit peak (reservation) | 7 817 082 B | 23 451 246 B |
| disk peak (reservation) | 93 808 056 B | 179 224 404 B |
| device peak (reservation) | 974 984 704 B | 974 984 704 B |
| sampled process-tree RSS peak | 5 514 735 616 B | 5 312 466 944 B |
| sampled device free VRAM min | 1 522 139 136 B | 1 522 139 136 B |
| producer work | 9.912 s | 10.027 s |
| consumer work | 19.103 s | 18.404 s |
| observed overlap intersection | 8.857 s | 8.869 s |

Gates: `scientific_outputs_identical=true`, `frames_identical=true`,
`complete_cells_identical=true`, `real_overlap_proved=true`,
`observed_memory_available=true`, `vram_observed=true`.

## How to read it (and how NOT to)

* The host/disk/device numbers are RESERVATIONS from the product's credit model
  and pipeline ledger. They are not RSS, not VRAM and not a high-water mark.
* The RSS number is a SUM over the sampled process tree at a 0.1 s period. Shared
  pages are counted once per process, so it is an aggregate UPPER bound that can
  legitimately exceed physical RAM. It is not a process RSS and not an OS peak.
* VRAM free/total is DEVICE-WIDE; `vram_pool_used_peak_bytes` is this process'
  CuPy default pool only. Boundary snapshots in the product remain instantaneous.
* The candidate's ~200 MiB lower sampled RSS is within run-to-run noise on this
  host and is NOT reported as a memory win. No performance claim is made.
* Halo here is 64 px (chosen so the existing halo-overhead floor admits the
  geometry on a 1.2 GiB box). The LP341 halo128 invariant is unchanged in code;
  its runtime proof is NOT_RUN on this hardware.

## Outstanding gates (not satisfied here)

* LP341 halo128 sky_mean before/after on a capable Windows node: **NOT_RUN**.
  No such node/tool is available in this worktree; this host cannot even build
  the halo128 LP geometry (PLAN budget ~959 MiB).
* LP341 planning at prefetch>=1 is proved at PLAN level by
  `tests/test_zegrid_r56_bounded_prefetch_credit.py` using the exact published
  Windows numbers (341 frames, patch 1625x1415, cache 10 193 216 671 B,
  B_plan 14 687 984 025 B).


## ARCH iteration (ZM-ZEGRID-BOUNDED-PREFETCH-20261025, 2026-10-26)

HEAD: `ffe75bc` (on top of the verified `d1ea296`). Branch
`perf/zegrid-bounded-prefetch`, local only.

### `linux-cuda-real-memory-halo64.json` — REAL-memory bounded CUDA smoke

Reproduce:

```
PYTHONPATH=src <project-venv>/bin/python tools/zegrid_r56_prefetch_campaign.py \
  --input /tmp/r56_big/input --out <fresh-scratch> \
  --halo 64 --gpu --storage-profile bounded --sample-period 0.1
```

No `--max-available-bytes`: the host observation is the REAL 2 265 784 320 B
re-observed per arm. Both arms executed on the GPU: `final_backend="gpu"`,
6/6 consumed cells, identical scientific outputs and frame identities, VRAM
observed. The planner admitted `prefetch=0` at this real budget (tile 384 after
the RUN shrink, workers 3, aggregate 876 522 496 B), so there is NO overlap
claim and `candidate_gpu_prefetch_ge1=false`, `candidate_overlap_proved=false`.
This is an honest NEGATIVE result at H64, not an LP/Windows proof.

### `matrix-18-modules-final.txt` — combined matrix

`426 passed, 3 skipped` in 298.15 s at HEAD with the late-binding seam fix (18 modules: R56 x2,
E2/E5/E7/E8/E9/E10/E11, R52 A/B/B-gateA/C, GPU admission fallback, R41b,
planner determinism, envelope, E6 large-halo layout). The r1 reds DID reproduce in a 7-module order and are now ROOT-CAUSED:
`_r55_active_gpu_admit` captured `available_memory_bytes` as an IMPORT-TIME
default, so the tests' 12 GiB pin was bypassed and the admission observed the
REAL host. On a low-RAM moment it shrank prefetch 2 -> 0 and the final plan hash
rejected the shrunk mapping against the unshrunk candidate. Fixed by late
binding the observer at call time + a regression guard; the same 7-module order
is green (146 passed, 1 skipped) and the full matrix is 426 passed, 3 skipped.

### LP341 faithful product witness (MODEL, not a Windows run)

`tests/test_zegrid_r56_bounded_prefetch_credit.py` drives the REAL product
decision with the published LP341 facts (341 frames, halo128, 3x3, worst patch
1625x1415, bounded, VRAM free 7 463 763 968 B, B_plan 14 687 984 025 B):
tile 192, K=1, workers 14, **prefetch 2**, aggregate 14 016 615 629 B
(95.4 % of B_plan) limited by the GAUGE phase (14 x 870 597 244 B); the
cache-build owner (1 515 400 840 B) is charged at p>=1, the normalization bound
is not binding. This is a model/contract result, not a runtime LP341 run.

## ARCH REWORK-1 (2026-10-26) — publication blockers F1/F2 fixed

HEAD: `4c703d9` (on `873abb1`), branch `perf/zegrid-bounded-prefetch`, local.

### F1 — admitted GPU identity preserved

`_run_r55_active_gpu_units` now carries the ADMITTED candidate/mapping
(`gpu_candidate` from admission) through the whole run. A runtime tile shrink is
materialized from the CURRENT candidate (preserving reduced prefetch/workers/K),
and finalization uses the current candidate + current bundle. Previously both
places rebuilt from `plan_set.chosen`, so an admission p2->p0 was lost and the
final hash rejected `mapping prefetch 0` against `candidate prefetch 2`.
Regressions: `test_admission_p2_to_p0_survives_final_manifest`,
`test_admission_shrink_then_runtime_tile_shrink_keeps_reduced_identity` in
`tests/test_zegrid_r55e2_active_gpu_prefetch.py`.

### F2 — indexed cleanup task decoded

The product pipeline task is `(index, cell_task)`; the nested cleanup callback
read `item[3]` directly, saw a 2-tuple and silently skipped, so credits were
released while the partial cache survived. It now unwraps the indexed item,
deletes ONLY the failed cell's cache, and RAISES a reported failure when
`_safe_rmtree` returns False (the pipeline records it in `cleanup_failures`
while preserving the primary exception). Regressions:
`tests/test_zegrid_r56_arch_rework1.py` (real partial file, cleanup BEFORE
credit release, failure residue, cancellation).

### Validation (exact, logs in this directory)

- Reviewer probes (unchanged audit file):
  `PYTHONPATH=src:tests <venv>/bin/python -m pytest -q -s
  .a2a-reports/ZM-R56-AUDIT-20261026/test_review_probes.py`
  -> **2 passed** (`arch-r1-reviewer-probes.txt`).
- Seven-module order (R56 bounded, R56 rework1, E2, E5, E7, E8, E9)
  -> **149 passed, 1 skipped** (`arch-r1-seven-modules.txt`).
- 18-module matrix -> **430 passed, 1 skipped** (`arch-r1-matrix-18.txt`).
- Fast gate `pytest -m 'not slow' -q`, 1200 s timeout -> **exit 124 at 49 %**
  with 2 failures, both in `tests/test_zegrid_r18_final_finishing.py`
  (`arch-r1-fast-gate-timeout.txt`, `arch-r1-fast-gate-first-failure.txt`,
  `arch-r1-fast-gate-second-failure.txt`). Both tests pass alone. A clean
  baseline archive of `873abb1` reproduces BOTH in the same fast-gate order
  (first: 1 failed/1271 passed/2 skipped; second with the first deselected:
  same counts) — `arch-r1-fast-gate-baseline-first.txt`,
  `arch-r1-fast-gate-baseline-second.txt`. They are PRE-EXISTING
  order-dependent failures, NOT regressions from F1/F2.

### Evidence corrections (binding, supersede earlier wording)

- The historical `linux-cuda-smoke-halo64-pinned.json` (pinned 4 GiB) is
  **SUPERSEDED** and is **NOT** corrected-p0 control proof: in that run even the
  cap-0 control overlapped, which contradicts the corrected p0-serial contract.
  It must not be cited as p0 evidence or as a memory/performance win.
- LP341 `tile192/p2/workers14` remains **MODEL** evidence, not a runtime result.
- The real-memory H64 CUDA artifact is real GPU execution but admitted **p0
  only**; no overlap was measured.
- Windows/LP341 halo128 runtime overlap remains **NOT_RUN**.
