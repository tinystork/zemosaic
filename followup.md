# FOLLOW-UP TASKS — GPU / MEMORY REVIEW CHECKLIST

This checklist guides you step by step through the GPU/memory review and improvements.

## 0. Read and understand existing GPU helpers

- [x] Open `zemosaic_utils.py` and identify:
  - `GPU_AVAILABLE`, `gpu_is_available`
  - `ensure_cupy_pool_initialized`, `free_cupy_memory_pools`
  - `gpu_memory_sufficient`
  - GPU reprojection functions (`gpu_assemble_final_mosaic_reproject_coadd`, `gpu_assemble_final_mosaic_incremental`, `reproject_and_coadd_wrapper`, etc.)
  - GPU image processing helpers (`_percentiles_gpu`, `detect_and_correct_hot_pixels_gpu`, `estimate_background_map_gpu`, etc.)
- [x] Confirm they are imported and used in:
  - `zemosaic_align_stack.py`
  - `zemosaic_worker.py`
  - Any other modules using GPU paths.

Goal: **build a mental map** of where GPU is used and where memory checks already exist.


## 1. Strengthen and centralise GPU memory checks

### 1.1 `gpu_memory_sufficient`

- [x] In `zemosaic_utils.py`, review `gpu_memory_sufficient(estimated_bytes, safety_fraction=0.75)`.
- [x] Ensure it:
  - Returns `True` when `GPU_AVAILABLE` is false (so CPU code does not get blocked).
  - Uses `memGetInfo()` correctly: compare `estimated_bytes` against `free_bytes * safety_fraction`.
  - Handles any exceptions (driver issues, etc.) by being permissive, not by disabling GPU entirely.
- [x] Optionally:
  - Add a small helper to clamp `safety_fraction` between `0.1` and `0.9` to avoid extreme values.
  - Add a debug log (guarded by an env var or config flag) when a check fails, so we can see in logs why GPU was refused.

### 1.2 CuPy pool management

- [x] Ensure `ensure_cupy_pool_initialized()`:
  - Is idempotent (uses a `_done` flag).
  - Optionally accepts a `device_id` but does not crash on invalid IDs.
- [x] Ensure `free_cupy_memory_pools()`:
  - Is called at the end of heavy GPU sections (e.g. GPU stacking, GPU reprojection).
  - Frees both device and pinned memory pools via `cp.get_default_memory_pool().free_all_blocks()` and similar.

If missing, **add** calls to `free_cupy_memory_pools()` in GPU-heavy functions’ `finally` blocks.


## 2. GPU usage in stacking (`zemosaic_align_stack.py`)

### 2.1 Usage of GPU helpers

- [x] Confirm `zemosaic_align_stack.py` imports `zemosaic_utils` and retrieves:
  - `ensure_cupy_pool_initialized`
  - `free_cupy_memory_pools`
  - `gpu_memory_sufficient`
- [x] Confirm helper functions exist:
  - `_ensure_gpu_pool()`
  - `_free_gpu_pools()`
  - `_has_gpu_budget(estimated_bytes: int) -> bool`

If any of these are missing or duplicate, **harmonise** them to use the central helpers from `zemosaic_utils.py`.

### 2.2 GPU winsorised / kappa-sigma stacking

For GPU stack functions (`gpu_stack_winsorized`, `gpu_stack_kappa` etc.):

- [x] Before allocating large arrays (e.g. `(N, H, W)` frames, masks, intermediate buffers):
  - Compute `estimated_bytes` realistically.
  - Use `_has_gpu_budget(estimated_bytes)` to decide if the GPU path is allowed.
- [x] If `_has_gpu_budget` returns `False`:
  - Log a GPU fallback message (see section 4).
  - Immediately fall back to CPU stacking (`cpu_stack_winsorized` / `cpu_stack_kappa` etc.).
- [x] Wrap GPU code in `try / except` to catch:
  - `cupy.cuda.memory.OutOfMemoryError`
  - Any other GPU-related exceptions
- [x] On such exceptions:
  - Log a `gpu_fallback_runtime_error` message.
  - Fall back to CPU stacking.

### 2.3 Chunking logic

- [x] Review `_iter_row_chunks(total_rows, frames, width, itemsize, max_chunk_bytes)`:
  - Ensure it is used in GPU stacking when dealing with very large images.
  - If GPU stacking still tries to allocate giant arrays in one go, refactor it to process row chunks instead.
- [x] Tie `max_chunk_bytes` to:
  - A reasonable default (e.g. 128–256 MB).
  - A dynamic limit based on `gpumemory_sufficient` and `memGetInfo()`, to adapt to available memory.

Goal: GPU stacking should **never** exit with a raw OOM; it should either:
- reduce chunk size automatically, or
- fall back to CPU with a clear log.


## 3. GPU usage in final mosaic assembly (`zemosaic_utils.py` + `zemosaic_worker.py`)

### 3.1 GPU reprojection and coadd functions

- [x] In `zemosaic_utils.py`, locate GPU reprojection/coadd functions (e.g. `gpu_assemble_final_mosaic_reproject_coadd`, `gpu_assemble_final_mosaic_incremental`, `gpu_reproject_and_coadd_impl`, `reproject_and_coadd_wrapper`).
- [x] Ensure they:
  - Check `gpu_is_available()` first.
  - Estimate memory usage for:
    - Mosaic accumulators (`H x W` float32 arrays for sum/weights).
    - Any per-tile buffers.
  - Use `gpu_memory_sufficient` with a reasonable safety margin (e.g. `0.7–0.8`).
- [x] If memory is insufficient **before** starting GPU work:
  - Log `gpu_fallback_insufficient_memory`.
  - Call the CPU implementation (`cpu_reproject_and_coadd` or equivalent).

- [x] Wrap the main GPU code in `try / except` and on exceptions:
  - Log `gpu_fallback_runtime_error`.
  - Fall back to CPU if `allow_cpu_fallback` is `True`.
  - If `allow_cpu_fallback` is `False`, re-raise the exception so the caller can surface a proper error.

- [x] Ensure that GPU-only kwargs (like `tile_affine_corrections`, `rows_per_chunk`, GPU-specific tuning parameters) are stripped from kwargs before calling the CPU function.

### 3.2 Worker integration

In `zemosaic_worker.py`:

- [x] Identify where the worker chooses final assembly method and GPU usage (Phase 5):
  - `final_assembly_method` (`reproject`, `incremental`, etc.)
  - `use_gpu_phase5` or similar flags taken from `global_plan` / config.
- [x] Ensure the worker:
  - Passes `use_gpu=True/False` to the reprojection wrappers based on config and availability.
  - Logs when GPU is requested but `gpu_is_available()` is false (`gpu_fallback_unavailable`).
  - Logs when GPU is requested but memory is insufficient or a runtime error occurs.

Goal: from logs alone, a user should see clearly whether:
- Phase 5 ran on GPU or CPU,
- and if CPU, exactly why.


## 4. Logging and localisation

### 4.1 Logging keys

- [x] Add new message keys in the logging / localization layer (if not already present), such as:
  - `gpu_info_summary` — “GPU detected: {name}, VRAM: {total_mb:.0f} MB, free: {free_mb:.0f} MB.”
  - `gpu_fallback_unavailable` — “GPU helpers requested but CuPy is not available. Falling back to CPU.”
  - `gpu_fallback_insufficient_memory` — “GPU memory guard: estimated {estimated_mb:.1f} MB vs allowed {allowed_mb:.1f} MB. Falling back to CPU.”
  - `gpu_fallback_runtime_error` — “GPU processing error: {error}. Falling back to CPU.”
- [x] Implement these in both:
  - `locales/en.json`
  - `locales/fr.json`

### 4.2 Integration in callbacks

- [x] Use the existing progress callback (`pcb`) / logger in `zemosaic_worker.py` to emit these messages with appropriate log levels:
  - INFO or INFO_DETAIL for informational messages.
  - WARN for fallbacks that still complete the run.
  - ERROR only if the whole run fails.

- [x] Where GPU helpers are used outside the worker (e.g. in `zemosaic_utils.py` utility scripts or GUIs), use Python logging (`logging.getLogger(__name__)`) with consistent messages.

Goal: no GPU-related decision (use or fallback) should be **silent**.


## 5. GUI and config alignment

### 5.1 Tk GUI (`zemosaic_gui.py`)

- [x] Ensure GPU-related options in the Tk GUI:
  - Read and write `use_gpu_phase5` in the configuration.
  - Synchronise legacy flags (`stack_use_gpu`, `use_gpu_stack`) to this canonical flag via `_synchronize_legacy_gpu_flags`.
- [x] Confirm that when the Tk GUI launches a run:
  - The config handed to the worker clearly indicates whether GPU is requested for Phase 5 and/or stacking.

### 5.2 Qt GUI (`zemosaic_gui_qt.py`)

- [x] Mirror the same logic as in Tk:
  - Maintain a canonical GPU flag in the Qt config.
  - Ensure the “Use GPU” checkbox/selector sets the correct fields.
- [x] If there is a GPU selector combobox (CPU/GPU auto/off), ensure:
  - Values map cleanly to boolean `use_gpu_phase5` (or similar).
  - Changes are persisted and restored from the config.

### 5.3 Filter Qt GUI (`zemosaic_filter_gui_qt.py`)

- [x] If Mosaic-First / special modes (like ZeSupaDupStack) include GPU hints:
  - Ensure these hints are passed to the worker in a way consistent with the new GPU policy.
- [x] Do not introduce duplicate GPU flags; piggyback on the same `use_gpu_phase5` / `stack_use_gpu` values.


## 6. Optional: GPU diagnostics helper

If you deem it low risk and helpful:

- [x] Add an optional “GPU diagnostics” helper in `zemosaic_utils.py` that:
  - Queries device name, total and free VRAM.
  - Logs a `gpu_info_summary` message once per run.
- [x] Call it from the worker at the very beginning of `run_hierarchical_mosaic` when GPU is enabled in config.

This is optional but can greatly help users understand what the GPU is doing.


## 7. Final sanity checks

- [x] Run the test scenarios (or create synthetic ones) for:
  - CPU-only system (no CuPy).
  - GPU present but small VRAM (simulate with aggressive safety fractions and large inputs).
  - GPU present with plenty of VRAM.
- [x] Confirm from logs that:
  - GPU is used when reasonable.
  - CPU fallback happens only with clear, understandable reasons.
  - No uncaught OOM or cryptic CuPy errors reach the user.

Once all boxes are checked, the GPU/memory review can be considered complete.
