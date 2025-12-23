# Mission: GPU safety — chunk cap that scales with VRAM (safe_mode)

## Context
Currently, zemosaic_gpu_safety._clamp_gpu_chunks() forces a fixed safe_mode ceiling of 256 MB, and then applies min(...) with VRAM total/free. This means even "real" NVIDIA GPUs with large VRAM are hard-capped to 256 MB when safe_mode triggers (e.g., Windows laptop + hybrid + battery).

We want to keep safety, but allow the cap to scale upward on large VRAM GPUs, while still honoring VRAM free and keeping a reasonable hard max.

## Scope / Constraints
- **No refactor**
- Touch **only**: `zemosaic_gpu_safety.py`
- Keep existing behavior when VRAM is unknown.
- Preserve minimum chunk bytes floor (32 MB).
- Preserve existing logging format (can add more info but don’t break).
- Do NOT change the meaning of safe_mode detection in this patch.

## Implementation plan
1. In `_clamp_gpu_chunks(plan, ctx)`:
   - Replace fixed `cap_bytes = 256 MB` logic with:
     - `base_cap = 256MB` by default
     - if `ctx.vram_total_bytes` is known:
       - `scaled = int(ctx.vram_total_bytes * 0.10)`  (10% of total VRAM)
       - `base_cap = max(256MB, scaled)`
       - `base_cap = min(base_cap, 2GB)` (hard cap)
     - if `ctx.vram_free_bytes` is known:
       - `base_cap = min(base_cap, int(ctx.vram_free_bytes * 0.80))`
     - `cap_bytes = max(base_cap, 32MB)` (floor)
   - Then clamp `plan.gpu_max_chunk_bytes` to `cap_bytes` (same semantics as today).
2. Keep the existing `gpu_rows_per_chunk` clamp behavior unchanged in this patch.
3. Ensure logging still prints `gpu_chunk_mb=...` and add optional debug comment lines only if needed (avoid noisy logs).

## Acceptance criteria
- In safe_mode on a GPU with >8 GB VRAM, `gpu_max_chunk_bytes` can be >256 MB (up to 2 GB), but never exceeds 80% of free VRAM.
- On systems with unknown VRAM: behavior remains essentially the same (defaults to 256 MB then floored to 32 MB).
- No other modules changed.

## Quick test (manual)
- Run any pipeline that triggers apply_gpu_safety_to_parallel_plan() with safe_mode enabled.
- Confirm in logs: `[GPU_SAFETY] ... gpu_chunk_mb=` reflects a value >256 on large VRAM GPUs (if free VRAM allows).
