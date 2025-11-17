# AGENT MISSION — GLOBAL GPU / MEMORY REVIEW FOR ZEMOSAIC

You are an autonomous coding agent working on the **ZeMosaic / ZeSeestarStacker** project.

The repository contains at least the following relevant modules:

- `run_zemosaic.py`
- `zemosaic_gui.py` (Tk)
- `zemosaic_gui_qt.py` (Qt)
- `zemosaic_filter_gui_qt.py` (Qt filter dialog)
- `zemosaic_worker.py`
- `zemosaic_utils.py`
- `zemosaic_align_stack.py`
- `cuda_utils.py` (if present)
- `solver_settings.py`
- `lecropper.py`
- `zequalityMT.py`
- Localization: `en.json`, `fr.json`
- Config: `zemosaic_config.py`

Your mission is to perform a **global review and hardening of GPU usage and GPU memory management** across the project, so that:

1. **The NVIDIA GPU is used whenever reasonably possible** when the user enables “use GPU” in the GUI/config.
2. **CPU fallback only happens for valid technical reasons** (no CuPy, hard memory limit exceeded, or actual GPU error).
3. **GPU memory usage is bounded and predictable**, using chunking / streaming strategies instead of failing with OOM.
4. **Fallbacks are clearly logged**, with localized messages, so the user understands why the GPU was not used.

You MUST preserve:
- Existing user-facing behaviour and options as much as possible.
- Backward compatibility with CPU-only systems.
- The current chunking / streaming logic, improving it where needed but not removing it.

The user already implemented:
- A CuPy availability flag and helpers in `zemosaic_utils.py` (`GPU_AVAILABLE`, `gpu_is_available`, `ensure_cupy_pool_initialized`, `free_cupy_memory_pools`, `gpu_memory_sufficient`, etc.).
- GPU-accelerated reprojection / coadd helpers (`gpu_assemble_final_mosaic_reproject_coadd`, `gpu_assemble_final_mosaic_incremental`, `reproject_and_coadd_wrapper`).
- GPU helpers in `zemosaic_align_stack.py` for stacking (GPU winsorized/kappa-sigma, GPU percentiles).
- GUI toggles and config mappings for `use_gpu_phase5` / `stack_use_gpu` / `use_gpu_stack`.

Your job is to:

- **Audit** all GPU-related code paths.
- **Align** them on a consistent GPU/CPU decision model.
- **Improve** memory checks and chunk sizing so that genuine GPU usage is maximised while remaining safe.
- **Instrument** them with clear logging and localisation keys for fallbacks.


## GLOBAL PRINCIPLES

Follow these principles in all GPU-related code:

1. **Single source of truth for GPU availability & memory**
   - Use `zemosaic_utils.gpu_is_available()` to decide if the GPU can be used.
   - Use `zemosaic_utils.gpu_memory_sufficient(estimated_bytes, safety_fraction=...)` as the standard way to check if a planned allocation is “safe enough”.
   - Use `ensure_cupy_pool_initialized()` before heavy CuPy use and `free_cupy_memory_pools()` after you are done with big allocations.

2. **Try GPU first when requested and available**
   - When a config / GUI option “use GPU” is set (`use_gpu_phase5`, `stack_use_gpu`, `use_gpu_stack`, or equivalent), and `gpu_is_available()` returns `True`, try the GPU path.
   - Only skip the GPU path if:
     - `gpu_memory_sufficient(...)` returns `False` for the estimated workload, or
     - The GPU code raised an exception (e.g. `cupy.cuda.memory.OutOfMemoryError` or other runtime error).

3. **Controlled & logged fallback**
   - When you skip GPU *before* running it (due to insufficient estimated memory), log a **structured message** through the existing progress / log callback with a dedicated key such as:
     - `gpu_fallback_insufficient_memory`
     - `gpu_fallback_unavailable`
   - When GPU execution fails at runtime and you fall back to CPU, log with:
     - `gpu_fallback_runtime_error`
   - Make sure these keys exist in `en.json` and `fr.json` with clear user-facing texts.

4. **Chunking / streaming instead of hard failure**
   - For large stacks and mosaics:
     - Prefer **row/area chunking** or **frame streaming** instead of allocating massive monolithic arrays.
     - Reuse and improve existing chunking helpers (`_iter_row_chunks`, `rows_per_chunk`, `max_chunk_bytes`, etc.).
   - Don’t remove existing chunking logic; instead:
     - Tie it more explicitly to `gpu_memory_sufficient`.
     - Dynamically adjust chunk size based on `memGetInfo()` when running on GPU.

5. **No silent GPU disabling**
   - Avoid conditions like `if not GPU_AVAILABLE: return CPU_path` when, in practice, `GPU_AVAILABLE` is true but a conservative guard is preventing GPU usage.
   - If you really have to force CPU (e.g. incompatible platform, env override such as `ZEMOSAIC_FORCE_CPU_INTERTILE`), log this fact with a clear, localized message.

6. **Thread-safety and multi-process safety**
   - `ensure_cupy_pool_initialized()` must remain **idempotent** and safe when called from multiple workers / processes.
   - Avoid global state that might break in multi-processing (e.g. changing device mid-run) unless there is already a proven pattern in the code.
   - Do not introduce any global GPU state that would break the worker model (`run_hierarchical_mosaic_process`).

7. **No breaking changes to external modules**
   - Do **not** modify the `seestar/core/stack_methods.py` external module.
   - Do **not** change ASTAP / astrometry.net invocation semantics.
   - Do **not** change FITS headers semantics or WCS I/O logic.


## SCOPE OF THE REVIEW

You must at least review and potentially modify:

1. `zemosaic_utils.py`
   - GPU helpers: `GPU_AVAILABLE`, `gpu_is_available`, `ensure_cupy_pool_initialized`, `free_cupy_memory_pools`, `gpu_memory_sufficient`, `_percentiles_gpu`, `detect_and_correct_hot_pixels_gpu`, `estimate_background_map_gpu`, and all GPU reprojection/assembly helpers.
   - Ensure:
     - Consistent use of memory guards before big allocations.
     - Proper use of CuPy memory pools.
     - Clear logging when GPU paths fail and fall back to CPU.

2. `zemosaic_align_stack.py`
   - GPU stack functions: `gpu_stack_winsorized`, any GPU helpers (`_gpu_nanpercentile`, etc.).
   - Chunking function `_iter_row_chunks(...)`.
   - Integration with `zemosaic_utils` for GPU memory checks and pool management.
   - Ensure GPU stacking:
     - Uses `_has_gpu_budget` (or equivalent) based on `gpu_memory_sufficient`.
     - Falls back to CPU only when strictly necessary and logs the reason.

3. `zemosaic_worker.py`
   - Final assembly logic: calls to `gpu_assemble_final_mosaic_reproject_coadd`, `gpu_assemble_final_mosaic_incremental`, `reproject_and_coadd_wrapper`, and related options (`final_assembly_method`, `use_gpu_phase5`, etc.).
   - Stacking plan and winsor streaming configuration (`winsor_max_frames_per_pass`, `winsor_worker_limit`).
   - Ensure the worker:
     - Passes correct GPU flags to helpers.
     - Logs when GPU is requested but unavailable, or when it has to fall back due to memory.

4. GUI frontends
   - `zemosaic_gui.py` (Tk) and `zemosaic_gui_qt.py` (Qt):
     - Ensure GPU-related options (`use_gpu_phase5`, `stack_use_gpu`, `use_gpu_stack`, etc.) are:
       - Coherent (synchronised between legacy and new names).
       - Persisted properly in `zemosaic_config.DEFAULT_CONFIG`.
       - Correctly propagated to the worker.
   - `zemosaic_filter_gui_qt.py`:
     - If any GPU options or hints are present (e.g. for Mosaic-First / ZeSupaDupStack modes), ensure they are consistent with the global GPU policy.

5. `cuda_utils.py` (if present)
   - Review this file for any ad-hoc GPU logic (device selection, memory checks, custom kernels).
   - Either:
     - Plug it cleanly into the central `zemosaic_utils` GPU helpers, or
     - Clearly isolate it as a low-level helper used by the other modules, without duplicating configuration logic.


## DELIVERABLES

Your modifications must:

1. **Strengthen GPU memory management**
   - Make `gpu_memory_sufficient` the standard guard for big allocations.
   - Use dynamic chunk sizing based on `memGetInfo()` where appropriate.
   - Ensure `free_cupy_memory_pools()` is systematically called after heavy GPU sections.

2. **Minimise unnecessary CPU fallbacks**
   - Only fall back to CPU when:
     - CuPy is missing / cannot be imported.
     - `gpu_memory_sufficient` says there is not enough memory even with chunking.
     - A GPU call raises an error (especially `OutOfMemoryError`).
   - Avoid overly conservative conditions that disable GPU even when it could work with smaller chunks.

3. **Improve logging and localisation**
   - Add/ensure localization keys for GPU-related events:
     - GPU availability summary at start of run.
     - GPU use vs CPU use for Phase 5 and stacking.
     - Each fallback reason.
   - Use existing logging infrastructure (progress callbacks, `log_key_*` style messages, etc.).

4. **Document new behaviour where needed**
   - Add inline comments near GPU decisions explaining:
     - Why a fallback may happen.
     - How to override behaviour via config or environment variables.
   - If you add new config keys or env vars (for example, to tweak GPU safety margins), document them in:
     - `zemosaic_config.py` defaults.
     - The GUI where appropriate (with tooltips or labels).


## STYLE & CONSTRAINTS

- Respect the existing coding style, logging patterns, and naming conventions.
- Keep functions importable on systems **without** CuPy installed (guard imports, use `importlib.util.find_spec`, etc.).
- Do not introduce heavy new dependencies.
- Keep all public APIs backward compatible unless explicitly indicated otherwise.


## SUCCESS CRITERIA

The GPU / memory review is complete when:

1. You can trace a clear, consistent decision path from GUI config ➜ worker ➜ GPU helper for both stacking and final assembly.
2. GPU is used by default when:
   - The user requested it,
   - CuPy is installed and available,
   - Memory is sufficient (according to `gpu_memory_sufficient` and chunking strategies).
3. CPU fallback reasons are always logged and localised, and there are no “silent” fallbacks caused by overly defensive code.
4. All GPU functions cleanly release memory via CuPy pools after heavy sections, and large allocations are guarded by reasonable checks.
