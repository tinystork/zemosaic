# ZM-MEMORY-90PCT-BETA — 90 % RAM/VRAM beta ceilings

- **Mission:** `ZM-MEMORY-90PCT-BETA-20261116`
- **Phase:** implementation
- **Status:** DONE (implementation + fast policy gate; no commit/push/version bump)
- **Date:** 2026-11-16
- **Repo:** `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources`
- **Branch:** `feat/zegrid-adaptive-resources`
- **Initial HEAD:** `86c632affb588a8ba6107af964321f3749b58f52`
- **Product version:** `5.1.6` unchanged

## Purpose

Raise the effective initial host-RAM and VRAM ceilings to **90 %** on beta for a
direct physical user comparison. This is a constant-only policy change: no new
flags, GUI, config keys, runtime abstractions, or planning machinery.

## Policy before / after

| Constant | Before | After | Applied to |
|---|---:|---:|---|
| `ram_contract.PLAN_FACTOR_NUM/DEN` | `4/5` (80 %) | `9/10` (90 %) | RAW **available** RAM, plan build |
| `ram_contract.ADMIT_FACTOR_NUM/DEN` | `9/10` (90 %) | `9/10` (90 %) | RAW **available** RAM, admission |
| `gpu.GPU_VRAM_SAFETY_PERCENT` | `70` | `90` | Fresh **free** VRAM |
| `gpu_readiness` standalone fallback | `70` | `90` | Standalone module load only |
| `gpu_readiness.INITIAL_TARGET_PERCENT` | `70` | `90` | `VramPolicy.initial_percent` |
| `gpu_readiness.VRAM_LADDER` | `(50, 65, 70, 75)` | `(50, 65, 70, 75, 90)` | Ascending rungs, historical shrink rungs preserved |
| `_R55E10_VRAM_SELECTION_HEADROOM_PERCENT` | `85` | `100` | Selection budget = full fresh policy budget |
| Effective selection ceiling | `0.85 x 0.70 = 59.5 %` | `1.00 x 0.90 = 90 %` | No second haircut |

The plan factor is still applied **exactly once** with integer arithmetic:
`B_plan = (9 * raw_available) // 10`. Admission remains
`B_admit = (9 * raw_available) // 10` from a **fresh** raw observation before
every admission attempt and again before launch. `Observation.plan`,
`Observation.admit` and `Observation.launch` keep their separate factors and
observations.

## Protections retained

- Fresh RAM observation before every admission/launch; no cached budget.
- Rebuild under the **same** miss observation's `B_plan`; the rebuild callback is
  still validated against that raw/factor/budget and the aggregate.
- No-growth envelope validation; admission still refuses to enlarge layout,
  tile, concurrency or backend.
- Shrink-only VRAM ladder with the historical rungs. Order is now
  `90 -> 75 -> 70 -> 65 -> 50 -> CPU`.
- `fresh_vram_revalidation` remains the RUN-time hard backstop.
- Whole-batch CPU fallback on GPU OOM/refusal; no mixed partial result.
- Integer-only VRAM policy, 64 MiB minimum floor, never inflated up to the floor.
- Hashes, science paths and product version untouched.

## Effective remaining margins

- **Host RAM:** 10 % of the raw *available* memory is left unclaimed by both the
  plan and admission budgets. Because `B_plan == B_admit`, the former 80–90 %
  admission-only tolerance band is **empty** (0 percentage points between plan
  and admission). `requires_tolerance()` is consequently always `False`.
- **VRAM:** 10 % of fresh free VRAM is left unclaimed. Selection now uses the full
  90 % budget; the former additional selection haircut is gone.
- **Device estimate risk:** the per-cell workspace is still an estimate. The
  protections are the fresh RUN revalidation, shrink-only ladder, and whole-batch
  CPU fallback. This beta deliberately accepts the reduced margin for physical
  comparison.

## Secondary RAM constants — inspected, intentionally unchanged

- `INMEM_AVAILABLE_FRAC = 0.80` (`zemosaic_zegrid_mode.py`): used by the legacy
  per-cell `_pick_mode` in-memory-vs-streaming selector, **not** by the ACTIVE
  PLAN/ADMIT contract. No active source call site was found for `_pick_mode`.
- `parallel.RAM_SAFETY_FRACTION = 0.8`: legacy scheduler/per-worker helper
  constant, not used by the ACTIVE RAM contract. No production source call site
  was found; it appears only in the module and historical tests/docs.

Neither constant is on the ACTIVE `B_plan` / `B_admit` path, so changing them
would alter unrelated legacy behavior without raising the active ceiling.

## Tests

Fast policy gate (run):

```text
PYTHONPATH=src python3 -m pytest -q \
  tests/test_zegrid_memory_90pct_beta.py \
  tests/test_zegrid_a1c_vram_policy.py \
  tests/test_zegrid_r55e10_vram_selection_headroom.py::test_e10_policy_and_legacy_constants_match_90pct_beta
12 passed in 1.17s
```

Targeted diagnostics/import checks (run):

```text
PYTHONPATH=src python3 -m pytest -q \
  tests/test_zegrid_r55e11_runtime_vram_adaptation.py::test_e11_diagnostics_distinguish_active_and_legacy_budgets \
  tests/test_zegrid_r55e2_active_gpu_prefetch.py::test_cpu_modules_do_not_import_cupy
2 passed in 2.29s
```

Compilation:

```text
PYTHONPATH=src python3 -m py_compile \
  src/zemosaic/core/zegrid/ram_contract.py \
  src/zemosaic/core/zegrid/gpu.py \
  src/zemosaic/core/zegrid/gpu_readiness.py \
  src/zemosaic/core/zegrid/scheduler.py \
  src/zemosaic/zemosaic_zegrid_mode.py \
  tests/test_zegrid_memory_90pct_beta.py
COMPILE_OK
```

No GPU run, no integration run, no product mosaic, no large pytest campaign.

### Known stale historical tests (NOT claimed green)

These encode the previous 80 %-RAM / 70 %-VRAM policy and were deliberately not
rewritten in this bounded mission:

- `tests/test_zegrid_r41_ram_contract.py` — extensive hardcoded 800/900 budgets,
  tolerance-band and replan expectations for plan 80 % / admit 90 %.
- `tests/test_zegrid_r41b_product_wiring.py` — historical 80 %-plan trap scenario
  and old budget expectations.
- `tests/test_zegrid_r55c_gpu_readiness.py` — policy/ladder/manifest tests pinned
  to 70 %, `(50, 65, 70, 75)` and initial 70 %. Only the directly coupled constant
  assertions were updated.
- `tests/test_zegrid_r55e10_vram_selection_headroom.py` — only the constant block
  and headroom diagnostics assertions were updated; the recorded M106 witness
  numbers and margins are 70 %/85 %-specific and remain stale.
- `tests/test_zegrid_r22_gpu_contract.py` — assertions are dynamic and should
  still pass, but its `44.8 MiB` comment is 70 %-specific.

## Files changed

- `src/zemosaic/core/zegrid/ram_contract.py` — PLAN factor 9/10; live contract
  docs and factor-based error text.
- `src/zemosaic/core/zegrid/gpu.py` — `GPU_VRAM_SAFETY_PERCENT = 90`.
- `src/zemosaic/core/zegrid/gpu_readiness.py` — standalone fallback 90, ladder
  `(50, 65, 70, 75, 90)`, initial 90, live shrink-order/docs.
- `src/zemosaic/core/zegrid/scheduler.py` — live PLAN-observation comment only.
- `src/zemosaic/zemosaic_zegrid_mode.py` — selection headroom 100; live policy
  comments/manifest note updated to 90 %.
- `tests/test_zegrid_memory_90pct_beta.py` — new small fast policy test.
- `tests/test_zegrid_a1c_vram_policy.py`,
  `tests/test_zegrid_r55c_gpu_readiness.py`,
  `tests/test_zegrid_r55e10_vram_selection_headroom.py`,
  `tests/test_zegrid_r55e11_runtime_vram_adaptation.py`,
  `tests/test_zegrid_r55e2_active_gpu_prefetch.py` — directly coupled constant
  assertions/expectations updated only.

Preserved unrelated dirty docs and untracked qualification artifacts from the
previous mission. No commit, no push, no version bump.

## Next step

Junior review and publication, then the user's direct beta physical comparison.
If the comparison rejects the margin, revert the same constants (PLAN 4/5,
VRAM 70, ladder without 90, selection 85) without touching science paths.

## Junior acceptance / beta publication authorization

ACCEPT for the explicitly requested beta experiment (2026-11-16).
Independent short gate: **14 passed in 2.63s**, using the five policy tests,
A1c policy suite, E10 constants, E11 budget provenance, and lazy-CuPy import
check listed above. Source diff independently inspected; no science change,
no physical GPU/Windows run, no broad regression campaign.

The 90 % values are **budget ceilings, not utilization targets**. Existing
resource-envelope estimate inflation (`DEFAULT_MARGIN_NUM/DEN = 15/100`)
remains: the scheduler compares margin-applied modeled peaks to the 90 %
budget. Legacy/per-cell CPU reservation dispatch also retains its 10 %
decision headroom (`RESERVATION_BUDGET_HEADROOM_FRAC = 0.10`); this does not
reintroduce the removed ACTIVE GPU selection haircut. Offline `dry_plan`
keeps its historical 4/5 default and is not the executing ACTIVE contract.
These mechanisms are deliberately outside the requested ceiling-only change.

Authorized publication: only this policy patch, directly coupled tests and
this document, to `beta`; **5.1.6 unchanged**, no tag/release, original Windows
run untouched. Historical hardcoded-policy tests above are not claimed green.
