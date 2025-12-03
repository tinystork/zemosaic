# Mission: Phase 3 “Mode C” – Automatic GPU with strict fallback to CPU-only

## Objective

Restore and solidify the **Phase 3 GPU stacker** using `zemosaic_align_stack_gpu.py`, with a
simple **automatic behaviour**:

- No new GUI options.
- Internally, Phase 3 works in **Mode C = AUTO**:

  1. If a sane GPU is present → **use GPU for all Phase 3 stacks** (with VRAM-aware chunking).
  2. On **first GPU failure** (CuPy unavailable, GPU smoke test fails, OOM, `GPUStackingError`, etc.):
     - attempt **one explicit retry on GPU** after freeing pools / tightening chunking if possible;
     - if the retry still fails → **disable GPU for the rest of the run** and
       **recompute the current tile on CPU**.
  3. Once the GPU has been declared “unusable”, **all remaining Phase 3 stacks run on the CPU**
     using the existing CPU/memmap/streaming fallback chain.

The goals are:

- keep **exactly the same quality** as the CPU reference path,
- avoid mixing “half-GPU / half-CPU / partially failed” tiles,
- avoid catastrophic fallbacks that create coverage holes or weird seams (cf. the Andromeda example).

---

## Constraints

- **No new GUI controls**.
  - Tk and Qt GUIs keep their current GPU toggle/semantics if they exist.
  - Phase 3 behaviour is automatic given the existing config + hardware.

- **No change** to SDS/non-SDS logic, clustering, ZeQualityMT, Lecropper, or Phase 5.
  This mission focuses on **Phase 3 stacking only**.

- Keep the GPU logic self-contained:
  - all CuPy imports and GPU helpers live in `zemosaic_align_stack_gpu.py` or `cuda_utils.py`;
  - `zemosaic_worker.py` must remain importable on CPU-only machines.

- Preserve existing **memmap / streaming / chunking** behaviour in the CPU path:
  - GPU should accelerate Phase 3,
  - but must never degrade global quality when it fails → we fall back to the existing CPU stacker.

---

## Relevant files

You will likely need to work in:

- `zemosaic_worker.py`  (Phase 3 orchestration, `create_master_tile`, etc.)
- `zemosaic_align_stack.py`  (CPU stacker and streaming/memmap logic)
- `zemosaic_align_stack_gpu.py`  (new GPU stacker for Phase 3 — already present again)
- `parallel_utils.py`  (ParallelPlan, chunk/autotune info)
- `cuda_utils.py`  (optional helpers to check GPU health, flush pools, etc.)
- `zemosaic_utils.py`  (GPU pool helpers used in other places)

Please **do not** modify GUI files (`zemosaic_gui*.py`) or Phase 5 / two-pass code in this mission.

---

## Target behaviour in detail

### 1. Centralize Phase 3 GPU usability

Implement / use a small set of helpers so that Phase 3 can ask:

- “Is GPU globally usable for Phase 3?” (result cached for this run)
- “Should this particular tile use GPU, or has GPU been disabled already?”

You can base this on the existing `_gpu_is_usable` in `zemosaic_align_stack_gpu.py`:

- Ensure `_gpu_is_usable(logger)`:
  - checks CuPy availability,
  - does a tiny allocation,
  - caches the result (`_GPU_HEALTH_CHECKED`, `_GPU_HEALTHY`).

On the worker side (`zemosaic_worker.py`), maintain **two Phase-3-specific flags**:

- `phase3_gpu_allowed: bool` – initial candidate based on:
  - `_gpu_is_usable(logger)`,
  - existing config/ParallelPlan (if `plan.use_gpu` is False, you can short-circuit).
- `phase3_gpu_hard_disabled: bool` – once set to `True`, GPU is no longer attempted
  for any subsequent tiles.

These flags should live in a scope shared by all Phase 3 stacks within a run
(e.g. part of the worker object, or `phase3_state` dict threaded through Phase 3).

### 2. Route Phase 3 stacks through the GPU helper (Mode C behaviour)

In `zemosaic_worker.py`:

- Identify the Phase 3 stack entry point (`create_master_tile(...)`) which currently
  stacks a list of aligned frames using the CPU stacker.

Refactor it so that:

1. The **CPU implementation** is clearly isolated:
   - e.g. `_stack_master_tile_cpu(...)` that calls into `zemosaic_align_stack`
     (with streaming/memmap/autochunk exactly as today).

2. The **GPU attempt** uses `gpu_stack_from_paths(...)` or `gpu_stack_from_arrays(...)`
   from `zemosaic_align_stack_gpu.py`:

   - Pass:
     - the aligned images or their cached `.npy` paths (depending on how Phase 3 stores them),
     - `stacking_params` / `stack_cfg` already used by the CPU stacker,
     - `parallel_plan` (if available) so `_resolve_rows_per_chunk(...)` can compute
       rows per chunk from `gpu_max_chunk_bytes` / `gpu_rows_per_chunk`,
     - the Phase 3 logger and `pcb_tile` callback if available (for chunk telemetry).

   - Respect `GPUStackingError`:
     - this is the signal that the GPU stacker cannot provide a valid image
       (NaNs, all zeros, unsupported algo, etc.).

3. Implement **Mode C logic**:

   ```python
   def _stack_master_tile_auto(...):
       # phase3_gpu_allowed & phase3_gpu_hard_disabled come from surrounding state
       if phase3_gpu_allowed and not phase3_gpu_hard_disabled:
           # First GPU attempt
           try:
               result_gpu, meta_gpu = gpu_stack_from_paths(
                   image_descriptors,
                   stacking_params,
                   parallel_plan=parallel_plan,
                   logger=logger,
                   pcb_tile=pcb_tile,
               )
               used_gpu = True
               return result_gpu, meta_gpu, used_gpu
           except GPUStackingError as exc:
               # GPU path produced unusable data; mark it as “suspect” but not yet hard-disabled
               logger.warning("[P3][GPU] Stack failed (GPUStackingError): %s -- retrying once on GPU", exc)
               # Optional: try freeing CuPy pools / tightening chunk size here
               # (see section 3 below), then one more GPU attempt...
           except Exception as exc:
               logger.warning("[P3][GPU] Unexpected GPU error: %s -- retrying once on GPU", exc, exc_info=True)
               # same retry logic
           # Second GPU attempt (after clean-up). If it fails again:
           try:
               # same call as above, potentially with smaller chunk
               result_gpu, meta_gpu = gpu_stack_from_paths(...)
               used_gpu = True
               return result_gpu, meta_gpu, used_gpu
           except Exception as exc:
               logger.error("[P3][GPU] Second GPU attempt failed; disabling Phase 3 GPU for this run: %s", exc, exc_info=True)
               phase3_gpu_hard_disabled = True
               # fall through to CPU
       # CPU fallback (either GPU disabled, or both attempts failed)
       used_gpu = False
       result_cpu, meta_cpu = _stack_master_tile_cpu(...)
       return result_cpu, meta_cpu, used_gpu
````

* The *current* tile is always recomputed on CPU after GPU failure,
  so we never end up with a half-empty super-tile or dropped frames.
* Once `phase3_gpu_hard_disabled` is `True`, all subsequent tiles go directly to CPU.

4. Ensure the rest of `create_master_tile(...)` is agnostic:

   * It just receives the stacked array (`master_data`) + metadata,
     feeds it through Lecropper, quality crop, alpha mask, etc.,
     regardless of whether it came from GPU or CPU.

### 3. Chunking and retry refinement

To minimise OOM and maximise VRAM usage, use existing helpers instead of ad-hoc sizes:

* In `zemosaic_align_stack_gpu.py`:

  * `_resolve_rows_per_chunk(...)` already uses `ParallelPlan` (`gpu_rows_per_chunk`, `gpu_max_chunk_bytes`)
    and defaults (`DEFAULT_GPU_MAX_CHUNK_BYTES`, `DEFAULT_GPU_ROWS_PER_CHUNK`).

* Ensure the worker actually passes a `parallel_plan` to the GPU stacker when available.

For the **second GPU attempt** after failure:

* If the error is clearly memory-related (e.g. `cp.cuda.memory.OutOfMemoryError`),
  or you catch a dedicated helper like `cuda_utils.is_oom_error(exc)`:

  * try to:

    * free CuPy pools (`zemosaic_utils.free_cupy_memory_pools()` or equivalent in `cuda_utils`),
    * reduce the effective chunk size:

      * either by constructing a new `ParallelPlan` with smaller `gpu_max_chunk_bytes`,
      * or by passing an explicit `rows_per_chunk` override to the GPU stacker
        (you can extend `gpu_stack_from_arrays` kwargs to accept `rows_per_chunk_override`
        if needed—keep it optional and backwards-compatible).

* If the second attempt still fails:

  * **do not try a third time**; set `phase3_gpu_hard_disabled = True`
    and stick to CPU for the rest of the run.

### 4. Logging and telemetry

Use the existing Phase 3 logging/pcb hooks to emit useful but not noisy messages:

* On the first successful GPU tile:

  * `"[P3][GPU] Using GPU stacker for Phase 3 (mode=auto)"`.
* On each tile (INFO_DETAIL level via `pcb_tile`):

  * `phase3_gpu_chunk_summary` is already implemented in `zemosaic_align_stack_gpu.py` and should be used.
* On GPU fallback and hard disable:

  * log once at WARNING/ERROR level that GPU has been disabled for the run,
    including a short summary of the exception type (`OOM`, `GPUStackingError`, etc.).

This will help diagnose why the system fell back to CPU without flooding the logs.

### 5. Testing / guardrails

Implement and run at least the following tests or manual scenarios:

1. **CPU-only machine (no CuPy / no CUDA):**

   * The worker imports fine.
   * Phase 3 runs entirely on CPU (no attempt to import/use CuPy).
   * Resulting mosaics are identical to current CPU reference.

2. **GPU machine, healthy GPU:**

   * Run a dataset with thousands of frames (like the Andromeda example).
   * Confirm:

     * Phase 3 uses GPU for all tiles (no hard-disable),
     * GPU chunking is used (see `phase3_gpu_chunk_summary` rows_per_chunk in logs),
     * No gaps / holes appear in the final mosaic compared to CPU.

3. **Simulated GPU failure:**

   * Temporarily force a `GPUStackingError` (e.g. by raising in the GPU helper for one test run),
     or simulate an OOM.
   * Confirm:

     * GPU is attempted once, retried once, then globally disabled,
     * the tile where the failure happened is still produced via CPU,
     * the rest of the run proceeds CPU-only, without artifacts.

Do not change any behaviour related to:

* Phase 4/5,
* SDS/non-SDS group logic,
* ZeQualityMT thresholds,
* WCS or Lecropper semantics.

