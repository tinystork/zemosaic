# Follow-up checklist – Phase 3 AUTO GPU (Mode C)

Work through the steps in order and tick them as you go.

---

## 1. Understand current Phase 3 stacker wiring

- [ ] Open `zemosaic_worker.py` and locate **Phase 3** code:
  - `create_master_tile(...)`,
  - any helpers directly responsible for stacking aligned frames into a master tile.
- [ ] Identify where the **CPU stacker** in `zemosaic_align_stack.py` is called.
- [ ] Confirm how `ParallelPlan` (from `parallel_utils`) is obtained and whether it is already
      available in Phase 3 context.

---

## 2. Cleanly isolate the CPU Phase 3 stacker

- [ ] In `zemosaic_worker.py`, extract the existing CPU stacking logic into a dedicated helper:

  - Example: `_stack_master_tile_cpu(image_descriptors, stacking_params, parallel_plan, logger, pcb_tile, ...)`.

- [ ] Ensure this helper:
  - uses **exactly the same behaviour** as the current Phase 3 implementation:
    - memmap vs in-memory,
    - streaming over rows or tiles,
    - kappa/winsor/linear-fit clipping,
    - final combine (mean/median/min/max),
    - RGB equalization, etc.
  - returns a tuple `(stacked_array, stack_metadata)` where:
    - `stacked_array` is `float32`, contiguous, shape `(H, W, C)` or `(H, W)`,
    - `stack_metadata` contains whatever the CPU path already produces.

- [ ] Replace direct CPU stacking calls in `create_master_tile(...)` with calls to `_stack_master_tile_cpu(...)`
      so the current behaviour is preserved.

---

## 3. Wire up the GPU helper for Phase 3

- [ ] Open `zemosaic_align_stack_gpu.py` and inspect:

  - `GPUStackingError`,
  - `_gpu_is_usable(logger)`,
  - `gpu_stack_from_arrays(...)`,
  - `gpu_stack_from_paths(...)`.

- [ ] In `zemosaic_worker.py`:

  - [ ] Add a **lazy import** block at top-level:

    ```python
    try:
        from zemosaic_align_stack_gpu import (
            gpu_stack_from_paths as _p3_gpu_stack_from_paths,
            GPUStackingError as _P3GPUStackingError,
            _gpu_is_usable as _p3_gpu_is_usable,
        )
        _P3_GPU_HELPERS_AVAILABLE = True
    except Exception:
        _p3_gpu_stack_from_paths = None
        class _P3GPUStackingError(RuntimeError):
            pass
        def _p3_gpu_is_usable(logger=None):
            return False
        _P3_GPU_HELPERS_AVAILABLE = False
    ```

  - [ ] Ensure this does **not** raise on CPU-only machines.

- [ ] Introduce a small Phase-3-specific state holder, e.g. at module or worker level:

  ```python
  _P3_GPU_STATE = {
      "allowed": True,          # initial intent
      "hard_disabled": False,   # becomes True after repeated failure
      "health_checked": False,
      "healthy": False,
  }
````

---

## 4. Implement Mode C decision logic

* [ ] Add a helper in `zemosaic_worker.py`:

  ```python
  def _phase3_gpu_candidate(parallel_plan, logger) -> bool:
      if _P3_GPU_STATE["hard_disabled"]:
          return False
      if not _P3_GPU_HELPERS_AVAILABLE:
          return False
      # Optionally respect ParallelPlan.use_gpu if present:
      try:
          if parallel_plan is not None and not getattr(parallel_plan, "use_gpu", True):
              return False
      except Exception:
          pass
      if not _P3_GPU_STATE["health_checked"]:
          healthy = bool(_p3_gpu_is_usable(logger))
          _P3_GPU_STATE["healthy"] = healthy
          _P3_GPU_STATE["health_checked"] = True
      return _P3_GPU_STATE["healthy"]
  ```

* [ ] Use this helper in the new `_stack_master_tile_auto(...)` to decide if GPU should be attempted.

---

## 5. Implement `_stack_master_tile_auto(...)` with strict retry

* [ ] Add a new helper, e.g.:

  ```python
  def _stack_master_tile_auto(image_descriptors, stacking_params, parallel_plan, logger, pcb_tile, zconfig, ...):
      # 1) Decide if GPU is a candidate
      use_gpu_candidate = _phase3_gpu_candidate(parallel_plan, logger)

      # 2) GPU attempt + optional retry
      if use_gpu_candidate and _p3_gpu_stack_from_paths is not None:
          # First attempt
          try:
              stacked, meta = _p3_gpu_stack_from_paths(
                  image_descriptors,
                  stacking_params,
                  parallel_plan=parallel_plan,
                  logger=logger,
                  pcb_tile=pcb_tile,
                  zconfig=zconfig,
              )
              return stacked, meta, True
          except _P3GPUStackingError as exc:
              if logger:
                  logger.warning("[P3][GPU] GPUStackingError on first attempt: %s -- retrying once", exc)
          except Exception as exc:
              if logger:
                  logger.warning("[P3][GPU] Unexpected GPU error on first attempt: %s -- retrying once", exc, exc_info=True)

          # Optional: free pools / tighten chunk before second attempt (next step)
          # Second attempt
          try:
              stacked, meta = _p3_gpu_stack_from_paths(
                  image_descriptors,
                  stacking_params,
                  parallel_plan=parallel_plan,
                  logger=logger,
                  pcb_tile=pcb_tile,
                  zconfig=zconfig,
              )
              return stacked, meta, True
          except Exception as exc:
              if logger:
                  logger.error(
                      "[P3][GPU] Second GPU attempt failed; disabling Phase 3 GPU for this run: %s",
                      exc,
                      exc_info=True,
                  )
              _P3_GPU_STATE["hard_disabled"] = True

      # 3) CPU fallback (either GPU disabled or both attempts failed)
      stacked_cpu, meta_cpu = _stack_master_tile_cpu(
          image_descriptors,
          stacking_params,
          parallel_plan=parallel_plan,
          logger=logger,
          pcb_tile=pcb_tile,
          zconfig=zconfig,
      )
      return stacked_cpu, meta_cpu, False
  ```

* [ ] Replace all direct Phase 3 stack calls in `create_master_tile(...)` with `_stack_master_tile_auto(...)`.

---

## 6. Improve retry for OOM via chunk/pool tuning

* [ ] Open `cuda_utils.py` and `zemosaic_utils.py` and look for helpers like:

  * `ensure_cupy_pool_initialized`,
  * `free_cupy_memory_pools`,
  * `gpu_memory_sufficient` or similar.

* [ ] Between the first and second GPU attempts, do the following **if the error looks memory-related**:

  * [ ] Detect OOM errors using either:

    * specific exception classes (`cp.cuda.memory.OutOfMemoryError`),
    * or a utility like `cuda_utils.is_oom_error(exc)` if present.

  * [ ] If OOM:

    * Call `free_cupy_memory_pools()` / equivalent to release cached VRAM.
    * Optionally recompute a stricter `ParallelPlan` for the retry
      (smaller `gpu_max_chunk_bytes` / `gpu_rows_per_chunk`), or pass an override
      `rows_per_chunk` kwarg to the GPU stacker if supported.

* [ ] Ensure that this retry logic is safe and does not crash on CPU-only machines
  (all GPU helpers must be behind guards).

---

## 7. Preserve downstream Phase 3 behaviour

* [ ] Verify that the output of `_stack_master_tile_auto(...)` has the same shape/dtype
  as the CPU stacker and is passed through:

  * Lecropper pipeline,
  * quality cropping,
  * alpha mask generation,
  * metadata/telemetry.

* [ ] If the GPU path adds extra metadata (e.g. `stack_metadata["rgb_equalization"]`),
  ensure this doesn’t conflict with existing dictionaries.

---

## 8. Logging & diagnostics

* [ ] Add a single INFO log at the start of Phase 3 when GPU is used:

  * `"[P3][GPU] Phase 3 GPU auto mode enabled (mode=C, candidate=True)"`.

* [ ] Ensure `zemosaic_align_stack_gpu.py` emits `phase3_gpu_chunk_summary`
  via `pcb_tile` when available.

* [ ] Ensure that when GPU is hard-disabled:

  * the worker logs something like:

    * `"[P3][GPU] GPU disabled for remaining Phase 3 tiles after repeated failures."`.

---

## 9. Test scenarios

* [ ] **CPU-only test** (no CuPy / no CUDA):

  * Run a representative dataset.
  * Confirm:

    * Phase 3 runs fully on CPU,
    * output is unchanged compared to pre-GPU code.

* [ ] **Happy GPU path**:

  * On a CUDA machine, run a large dataset (~10k+ frames).
  * Confirm:

    * GPU is used for all tiles,
    * chunk sizes make sense in `phase3_gpu_chunk_summary`,
    * final mosaic shows no coverage holes or strange seams.

* [ ] **Forced GPU failure**:

  * Temporarily inject a `raise GPUStackingError("test")` in the GPU stacker
    for one tile, or force an OOM.
  * Confirm:

    * GPU is attempted once, then retried once,
    * afterwards, GPU is hard-disabled and the entire remainder of the run is CPU-only,
    * the tile for which GPU failed is correctly produced via CPU.

If any change would require touching Phase 5 or GUI files, stop and document
your reasoning instead of modifying them in this mission.
