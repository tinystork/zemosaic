# Mission: Grid Mode - Transparent GPU concurrency limiter (auto VRAM) - NO UI / NO new user params

## Problem
Grid mode can run many workers in parallel (auto=0 chooses a high CPU-based count).
Even with per-tile chunking (stack_chunk_budget_mb), multiple workers may enter the GPU stacking section concurrently, causing CuPy OOM (cudaErrorMemoryAllocation) and pinned memory OOM.

## Goal
Make GPU usage stable and automatic WITHOUT adding any new user-facing parameter:
- Keep existing worker logic (auto=0) untouched.
- Add an internal GPU concurrency limiter (Semaphore).
- Compute the maximum number of concurrent GPU stacks automatically based on free VRAM at runtime.
- Do not change mosaic outputs (only scheduling/stability/logging).

## Constraints
- NO REFACTOR: surgical patch.
- Prefer touching only grid_mode.py.
- Must not introduce new GUI options or config fields required by users.
- Works even if CuPy missing / GPU disabled (no regression).
- Existing GPU fallback-to-CPU remains unchanged.

## Implementation (grid_mode.py)
### 1) Compute internal GPU concurrency limit
When GPU stacking path is enabled:
- Try import cupy as cp
- Query free/total VRAM:
  - free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
  - free_mb = free_bytes / (1024**2), total_mb likewise

Estimate memory per concurrent GPU stack worker:
- stack_chunk_budget_mb already exists in code (used for chunk sizing)
- Heuristic (conservative):
  - safety_mult = 2.5
  - fixed_overhead_mb = 512
  - per_worker_mb = stack_chunk_budget_mb * safety_mult + fixed_overhead_mb
- Keep headroom:
  - usable_mb = free_mb * 0.80

Compute:
- auto_n = floor(usable_mb / per_worker_mb)
- gpu_concurrency = clamp(auto_n, 1..4)  (cap to 4 for safety)
- If VRAM query fails, default gpu_concurrency = 1

Important:
- This value is INTERNAL ONLY; no new settings, no new UI.

### 2) Add a semaphore to guard only the GPU stacking section
Create `gpu_semaphore = threading.Semaphore(gpu_concurrency)` in the Grid Mode run entrypoint (or local closure scope).
Wrap only the call(s) to `_stack_weighted_patches_gpu(...)` (e.g. inside `flush_chunk()`):
- `with gpu_semaphore:`
    - run GPU stacking
    - optional `cp.cuda.Device().synchronize()` (guarded, avoid always-on sync)
    - free CuPy pools after stacking:
      - cp.get_default_memory_pool().free_all_blocks()
      - cp.get_default_pinned_memory_pool().free_all_blocks()

### 3) Logging
Add concise informational logs (not too verbose):
- At Grid Mode start (GPU enabled):
  - "GPU concurrency limiter: free/total VRAM, stack_chunk_budget_mb, per_worker_est_mb, chosen_concurrency"
- Debug logs around GPU stack enter/exit are OK but keep them debug-level.

### 4) Safety / fallback
- If CuPy import or memGetInfo fails: gpu_concurrency=1.
- If GPU stack throws, existing fallback to CPU stays.
- Ensure semaphore release on exceptions via context manager.

## Acceptance criteria
- Same Grid Mode dataset no longer crashes with CuPy OOM when GPU is enabled.
- Worker count auto=0 unchanged, but GPU stacks limited to chosen_concurrency.
- Logs show computed limiter values.
- No output changes vs successful GPU runs (when they already succeed).

## Files
- grid_mode.py only (preferred).
