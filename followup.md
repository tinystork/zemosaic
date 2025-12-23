# followup.md — Guidance & guardrails for implementation

## Step-by-step plan (keep commits small)

### Step 1 — Add zemosaic_gpu_safety.py
- [x] Implement GpuRuntimeContext + probe_gpu_runtime_context()
- [x] Implement apply_gpu_safety_to_parallel_plan() returning (plan, ctx)
- [x] Implement apply_gpu_safety_to_phase5_flag()
- [x] All imports must be optional:
  - [x] psutil might exist in worker; still guard sensors_battery()
  - [x] wmi is optional; Windows-only; guard import
  - [x] cupy is optional; DO NOT import cupy here unless you absolutely need it

### Step 2 — Wire into zemosaic_worker.py
Touch only the narrow points:
- [x] Right after global auto_tune_parallel_plan(kind="global")
- [x] Right after phase5 auto_tune_parallel_plan(kind="global_reproject")
- [x] Right before computing/using use_gpu_phase5_flag
- [x] Ensure zconfig.parallel_plan / parallel_plan_phase5 updated with the “safe” plan so telemetry context keeps working
- [x] Add one log summary line + flush handlers

### Step 3 — Wire into grid_mode.py
- [x] Keep existing _compute_gpu_concurrency but feed it a safer budget/concurrency when ctx.safe_mode
- [x] Log one summary line

### Step 4 — Optional hardening in zemosaic_align_stack_gpu.py
- [x] If env ZEMOSAIC_GPU_SAFE_MODE=1:
  - [x] reduce rows_per_chunk cap
  - [x] synchronize after each chunk
  - [x] add per-chunk timeout -> raise GPUStackingError (worker already falls back)

## What NOT to do
- Do not change stacking math, weights, sigma-kappa, winsorization logic, or output formats.
- Do not invent a new config schema.
- Do not refactor Phase 3/Phase 5 architecture beyond inserting policy calls.

## Logging format (keep it grep-friendly)
Use consistent prefixes:
- [GPU_SAFETY] ... (worker/global/phase5)
- [GPU_SAFETY][GRID] ... (grid_mode)
- [GPU_SAFETY][P3] ... (optional)

Example:
[GPU_SAFETY] safe_mode=1 vendor=nvidia hybrid=1 battery=1 vram_free_mb=5120 -> phase5_gpu=0 plan.use_gpu=0 gpu_max_chunk_mb=256 gpu_rows=128 reason="win+laptop/hybrid"

## Verification hints
- Confirm that in safe_mode, Phase 5 shows "phase5_using_cpu" even if GPU was selected (expected).
- Confirm that Phase 3 GPU candidate respects plan.use_gpu=False and goes CPU without trying GPU.
- Confirm that on non-Windows machines, ctx.safe_mode is false and nothing is clamped.

## Rollback safety
If something breaks, the safest temporary fallback is:
- keep the new module
- disable its integration calls (feature becomes inert)
This should make revert easy.
