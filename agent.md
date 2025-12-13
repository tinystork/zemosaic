Voici un couple **agent.md / followup.md** prêt à coller pour Codex. Objectif : **corriger définitivement le “vert” en mode legacy** en **unifiant CPU/GPU** sur la couverture (coverage) et en **empêchant le split GPU/CPU par canal** en Phase 5 (Two-Pass).

---

## agent.md

````md
# Mission — Fix “Green Tint” in Legacy Mode (Phase 5 Two-Pass) by Unifying CPU/GPU Coverage + Avoid Per-Channel Split

## Context
We have a persistent green tint regression in **legacy/classic mode** mosaics. Logs show the RGB balance is healthy before Two-Pass, then becomes strongly green after Two-Pass coverage renormalization (Phase 5). The root cause is a **CPU/GPU split per channel during Two-Pass reprojection**, combined with **inconsistent meaning/scaling of the returned `coverage` map between GPU and CPU paths**.

Key evidence from logs:
- Two-Pass channel reprojection uses **gpu=True for channel 1**, then **gpu=False for channels 2/3**.
- GPU `coverage` max is ~0.70 while CPU `coverage` max is 12.0 → not the same unit/scale.
- After Two-Pass, G/R ratio explodes (~4x), causing green tint.

## Non-goals / Constraints
- Do **NOT** create a new logging system. A GUI log-level dropdown already exists (PySide) and must be respected. Only use the existing logger and its levels.
- Do **NOT** change anything outside the Two-Pass coverage logic and the GPU reproject coverage scaling.
- Preserve existing behavior regarding “batch size = 0” and “batch size > 1” (do not touch).
- Keep CPU fallback behavior, but make it *coherent across channels*.

## Targets (files/functions)
1) `/zemosaic_utils.py`
   - `gpu_reproject_and_coadd_impl(...)`
   - Ensure GPU coverage returned has the **same semantics** as CPU `reproject_and_coadd` footprint:
     - CPU footprint is effectively the **sum of weights** (can exceed 1, e.g. 12 overlaps)
     - GPU currently normalizes in mean path: `weight_sum / n_inputs` clipped to [0,1] (BAD)

2) `/zemosaic_worker.py`
   - `run_second_pass_coverage_renorm(...)`
   - Internal `_process_channel(ch_idx, use_gpu_flag)`
   - Remove/avoid the “only first channel gets GPU” assignment. Two-Pass must use **one backend for all channels**:
     - If GPU is selected: attempt **GPU for all channels sequentially**
     - If any GPU failure/fallback happens: rerun **ALL channels on CPU** (not mixed)

## Required Changes
### Backend Safety Rule — Phase 5 (Two-Pass Coverage Renorm)

Two-Pass coverage renormalization is **not backend-mix safe**.

Therefore, enforce the following invariant:

- During Phase 5, **all RGB channels MUST be processed with the same backend**
  (GPU or CPU).
- If `use_gpu=True` and any channel fails, falls back, or cannot use GPU:
  - Abort the current Two-Pass attempt
  - Log a single warning
  - Rerun Phase 5 with **CPU for ALL channels**
- Mixed backend execution (e.g. R/G on GPU and B on CPU) is **explicitly forbidden**
  even if GPU and CPU coverage semantics are aligned.

Rationale:
- Two-Pass applies a coverage-based renormalization shared across channels.
- Even small backend-dependent differences (interpolation, NaN handling,
  accumulation order) can introduce chromatic drift.
- Enforcing a single backend guarantees photometric coherence and long-term stability.

This rule applies **only** to Phase 5 (Two-Pass coverage renorm).
Other pipeline phases may continue to use independent GPU/CPU logic as currently implemented.

### A) Unify GPU coverage semantics with CPU footprint
In `zemosaic_utils.py`, inside `gpu_reproject_and_coadd_impl`, for the **mean/fast path**:
- Replace the normalized coverage:
  - current: `coverage_gpu = clip(weight_sum_gpu / max(1,n_inputs), 0..1)`
- With: coverage being the raw weight sum (matching CPU footprint semantics):
  - `coverage_gpu = weight_sum_gpu`
- Also sanitize to finite float32 (nan/inf -> 0), similar to other GPU paths.

This makes GPU coverage max reflect overlaps/weights (e.g. up to ~12), matching CPU.

### B) Remove per-channel GPU assignment (no split CPU/GPU across channels)
In `zemosaic_worker.py`, in `run_second_pass_coverage_renorm`:
- Replace the logic that assigns GPU only to the first channel:
  ```py
  gpu_assigned = False
  for ch in range(n_channels):
      use_gpu_flag = bool(use_gpu and not gpu_assigned)
      if use_gpu_flag:
          gpu_assigned = True
      channel_tasks.append((ch, use_gpu_flag))
````

* With a policy:

  * If `use_gpu` is True:

    * Process channels **sequentially** with `use_gpu_flag=True` for all channels.
    * If any channel errors and falls back, abort and rerun all channels on CPU to avoid mixing.
  * If `use_gpu` is False:

    * Process channels on CPU (you may keep threads, but ensure no GPU is used).

Implementation hint:

* Write a small helper:

  * `_reproject_all_channels(use_gpu_flag: bool) -> (mosaic_channels, coverage_channels)`
* If GPU attempt fails for any channel:

  * log a warning once (no spam)
  * rerun `_reproject_all_channels(False)`
* Ensure logs show per-channel `cov_stats(min,max)` and a summary line:

  * `[TwoPass] Coverage backend: gpu_all=True` or `gpu_all=False`

### C) Keep existing telemetry/log-level usage

* Use the existing logger (already configured).
* Do not add new handlers/files.
* Log only at DEBUG for detailed stats.

## Acceptance Criteria

* In Two-Pass, all channels use the same backend. No more “gpu=True then gpu=False” per channel.
* GPU `coverage` stats can exceed 1 and resemble CPU (e.g. max ~12 when overlaps exist).
* Post Two-Pass RGB ratios no longer explode (green tint gone in legacy mode).
* No regression in CPU-only runs.
* No changes to unrelated pipeline phases.

## Minimal Manual Test

1. Run a legacy/classic mosaic where Two-Pass is enabled and GPU is available.
2. Confirm in logs:

   * Channel reprojection lines all show `gpu=True` OR all show `gpu=False`
   * Coverage stats max are on comparable scale across channels
3. Verify final mosaic no longer has strong green tint.

## Deliverables

* Code changes in the two files above.
* No new files unless strictly necessary.
* Keep diffs tight.


