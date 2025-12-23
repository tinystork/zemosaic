# agent.md — Mission: GPU Safety Layer (ZeMosaic, repo-current)

You are working on ZeMosaic (Windows/Linux/macOS). The codebase already contains:
- ResourceTelemetryController in both zemosaic_worker.py and grid_mode.py.
- Parallel auto-tuning via parallel_utils.ParallelPlan (auto_tune_parallel_plan, detect_parallel_capabilities).
- Phase 5 GPU selection via gui config keys gpu_selector / gpu_id_phase5 and worker env masking (CUDA_VISIBLE_DEVICES).
- Phase 3 GPU stacking has a GPU candidate gate + retry + shrink-parallel-plan fallback.

This mission is to ADD a defensive “GPU Safety Layer” that reduces Windows TDR/OS freezes
(especially hybrid Intel+NVIDIA laptops) WITHOUT changing scientific algorithms and WITHOUT adding user-facing options.

## Primary objective
Create a centralized runtime GPU safety/policy module and route GPU decisions through it,
so GPU usage becomes auto-adaptive and conservative on risky systems.

## Files you MUST consider (existing integration points)
- parallel_utils.py: ParallelCapabilities + ParallelPlan + auto_tune_parallel_plan()
- zemosaic_worker.py:
  - global_parallel_plan creation (auto_tune_parallel_plan(kind="global"...))
  - phase5 parallel_plan_phase5 (auto_tune_parallel_plan(kind="global_reproject"...))
  - phase5 GPU init uses gpu_id_phase5 + CUDA_VISIBLE_DEVICES masking
  - phase3 GPU candidate gate uses parallel_plan.use_gpu and retries with _shrink_parallel_plan_for_gpu
  - ResourceTelemetryController context includes plan fields (use_gpu, gpu_rows_per_chunk, gpu_max_chunk_bytes, use_gpu_phase5)
- grid_mode.py: has its own GPU concurrency limiter (_compute_gpu_concurrency) + telemetry
- zemosaic_align_stack_gpu.py: GPU stacking chunk sizing (gpu_rows_per_chunk, gpu_max_chunk_bytes) but no explicit sync/timeout guard
- zemosaic_gui_qt.py: GPU selector stores gpu_selector and gpu_id_phase5 (do not change GUI unless absolutely needed)

## Deliverables
- [x] Add ONE new module: zemosaic_gpu_safety.py (or gpu_safety.py if you prefer, but keep imports stable).
- [x] Minimal integration changes in:
  - [ ] parallel_utils.py (optional, if you integrate policy inside auto_tune_parallel_plan)
  - [x] zemosaic_worker.py (required)
  - [x] grid_mode.py (required)
  - [x] zemosaic_align_stack_gpu.py (recommended for “short-kernel” guardrails)
- [x] Logging/observability: decisions must be logged via existing logger + progress_callback + telemetry context.

## Constraints (STRICT)
- Do NOT change scientific algorithms/results (only resource usage strategy).
- Do NOT add new user-facing GUI options or config fields.
- Do NOT refactor unrelated parts of the worker.
- Keep Linux/macOS behavior intact.
- Always prefer safety over speed on Windows.

---

# Implementation details

## 1) Create zemosaic_gpu_safety.py [x]
Implement a small, dependency-safe policy layer with best-effort probing (all guarded by try/except).

Suggested API:

- dataclass GpuRuntimeContext:
  - os_name / platform_system
  - gpu_available (bool)
  - gpu_name (str|None)
  - gpu_vendor (str: "nvidia"/"intel"/"amd"/"apple"/"unknown")
  - vram_total_bytes, vram_free_bytes (int|None)
  - has_battery (bool|None)
  - is_windows (bool)
  - is_hybrid_graphics (bool|None)  # best-effort: multiple controllers, or "intel"+"nvidia" heuristics
  - safe_mode (bool)
  - reasons (list[str])

- probe_gpu_runtime_context(*, preferred_gpu_id: int|None = None) -> GpuRuntimeContext
  - Use parallel_utils.detect_parallel_capabilities() if available to get gpu_name + VRAM.
  - has_battery: psutil.sensors_battery() if available, else Windows-only WMI Win32_Battery if wmi installed.
  - hybrid: Windows-only WMI Win32_VideoController list (if wmi installed); detect both Intel and NVIDIA in names.
  - vendor: parse from gpu_name lowercased.

- apply_gpu_safety_to_parallel_plan(
    plan: ParallelPlan | None,
    caps: ParallelCapabilities | None,
    config: Mapping[str, Any] | None,
    *,
    operation: str,
    logger: logging.Logger | None = None,
  ) -> tuple[ParallelPlan | None, GpuRuntimeContext]
  - Decide safe_mode triggers:
    - safe_mode = is_windows AND (has_battery is True OR is_hybrid_graphics is True)
    - If vendor == "intel" and no discrete GPU detected => disable GPU (plan.use_gpu=False)
  - Enforce conservative clamps (only in safe_mode):
    - plan.use_gpu may remain True but clamp GPU chunking:
      - gpu_max_chunk_bytes <= 256MB (or smaller if VRAM small)
      - gpu_rows_per_chunk <= 128 (and >= 32)
    - Additionally clamp effective VRAM fraction (internally) to <= 0.6 if safe_mode.
  - Always append reasons and log them once (INFO).

- apply_gpu_safety_to_phase5_flag(
    use_gpu_phase5_flag: bool,
    ctx: GpuRuntimeContext,
    *,
    logger: logging.Logger | None = None,
  ) -> bool
  - If safe_mode and vendor/VRAM unknown => force disable for Phase 5 (return False).
  - If ctx indicates “intel-only” => force disable.
  - Otherwise keep.

Also expose:
- get_env_safe_mode_flag(ctx) -> bool
  - If safe_mode, set os.environ["ZEMOSAIC_GPU_SAFE_MODE"]="1" (debug-only internal env, not user-facing)

## 2) Integrate in zemosaic_worker.py (REQUIRED) [x]
Goal: ensure BOTH global plan and phase5 plan go through the safety layer, and Phase 5 GPU flag is guarded.

Integration points:
- After global_parallel_plan = auto_tune_parallel_plan(kind="global"...):
  - call apply_gpu_safety_to_parallel_plan(global_parallel_plan, parallel_caps, worker_config_cache, operation="global")
  - store the returned plan back into worker_config_cache["parallel_plan"] and zconfig.parallel_plan
  - keep telemetry context working (it already reads plan attributes).
- After parallel_plan_phase5 = auto_tune_parallel_plan(kind="global_reproject"...):
  - call apply_gpu_safety_to_parallel_plan(parallel_plan_phase5, caps_for_phase5, worker_config_cache, operation="global_reproject")
  - store into worker_config_cache["parallel_plan_phase5"] and zconfig.parallel_plan_phase5
- Before finalizing use_gpu_phase5_flag:
  - create/probe ctx using preferred_gpu_id=gpu_id_phase5
  - use use_gpu_phase5_flag = apply_gpu_safety_to_phase5_flag(use_gpu_phase5_flag, ctx)
  - if ctx.safe_mode: set env var ZEMOSAIC_GPU_SAFE_MODE=1
- Logging:
  - Emit one concise summary line:
    "[GPU_SAFETY] safe_mode={0/1} vendor=... hybrid=... battery=... vram_free_mb=... -> phase5_gpu={0/1}, plan.use_gpu={0/1}, gpu_max_chunk_mb=..."
  - Flush logger handlers after this summary (best-effort) to survive OS-level crashes.

## 3) Integrate in grid_mode.py (REQUIRED) [x]
grid_mode currently computes GPU concurrency via _compute_gpu_concurrency(stack_chunk_budget_mb).

Change minimally:
- Probe ctx via zemosaic_gpu_safety.probe_gpu_runtime_context()
- If ctx.safe_mode:
  - force concurrency=1
  - optionally reduce stack_chunk_budget_mb effective value in concurrency calc (e.g. multiply by 0.6)
  - Log one line:
    "[GPU_SAFETY][GRID] safe_mode=... -> chosen_concurrency=..."

Do NOT refactor the whole grid_mode pipeline.

## 4) Hardening in zemosaic_align_stack_gpu.py (RECOMMENDED) [x]
Without changing results:
- When ZEMOSAIC_GPU_SAFE_MODE=1:
  - cap rows_per_chunk to a smaller value (e.g. min(current, 64 or 128))
  - after each chunk combine, call cp.cuda.Stream.null.synchronize()
  - track per-chunk wall time; if a single chunk exceeds a conservative threshold (e.g. 2–3 seconds),
    raise GPUStackingError to trigger the existing CPU fallback in zemosaic_worker.

This does not change math; it only reduces risk of long kernels / driver watchdog triggers.

## Acceptance criteria / Definition of done
- No new GUI options.
- On Windows hybrid laptops, the logs show safe_mode engaged and GPU usage reduced/disabled automatically.
- Phase 3 and Phase 5 complete without OS freezes on the problematic dataset (or they auto-fallback to CPU).
- On Linux desktop, behavior remains effectively unchanged (GPU still used when available).
- All modified modules import cleanly on machines without CuPy/WMI.

## Quick sanity checks you must run
- python -m py_compile zemosaic_gpu_safety.py parallel_utils.py zemosaic_worker.py grid_mode.py zemosaic_align_stack_gpu.py
- Run a CPU-only config (gpu_selector = CPU) to ensure no regressions.
- Run GPU-enabled config on a safe machine; confirm plan.use_gpu still True when not in safe_mode.
