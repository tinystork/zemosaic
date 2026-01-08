# Mission: Phase 3 GPU WSC — Dynamic VRAM preflight + channel-wise execution (critical)

## Context
On large master tiles (e.g. N≈151 frames, RGB, weights_block=(N,1,1,3)), the GPU PixInsight-like WSC path can trigger CuPy OOM:
- Example symptom: "Out of memory allocating ~2GB ...; falling back to CPU for WSC."
- On Windows/WDDM this may also cause paging/freezes before OOM.

The current Phase 3 GPU stacker:
- computes rows_per_chunk using a float32-only estimate (bytes_per_row = width*channels*4*N),
- enforces MIN_GPU_ROWS_PER_CHUNK = 32,
- calls wsc_pixinsight_core() on the full (N, rows, W, C) block, where internals use float64 and allocate several large temporaries.

This massively underestimates VRAM usage for WSC, and RGB multiplies memory further.

## Goals (must)
1. [x] Prevent VRAM explosions on GPU PixInsight WSC in Phase 3.
2. [x] Add a **dynamic VRAM preflight** for WSC:
   - Use cp.cuda.runtime.memGetInfo() to get free VRAM at runtime.
   - Convert that into a safe rows_per_chunk for WSC (conservative headroom).
   - Allow rows_per_chunk to go below 32 for WSC (down to 1).
3. [x] Implement **channel-wise WSC execution** for RGB:
   - Process R, then G, then B independently on the GPU.
   - This reduces peak VRAM by ~3x and is functionally equivalent (WSC does not mix channels).
   - Maintain CPU/GPU parity (float64 internal math stays inside wsc_pixinsight_core).
4. [x] Add an **OOM backoff loop** for WSC chunks:
   - If an OOM happens despite preflight, halve rows_per_chunk, free CuPy pools, and retry the same row_start.
   - Only fallback to CPU if rows_per_chunk == 1 still OOMs (or after a bounded number of retries).
5. [x] Add clear logging so end-users and devs understand what happened:
   - Preflight chosen rows_per_chunk, N/W/C, free VRAM, budget, estimated bytes/row, channel-wise enabled.
   - On backoff: log the new rows_per_chunk and retry count.
   - On CPU fallback: log that GPU is not possible even at rows=1.

## Non-goals / Guardrails (do NOT do these)
- Do NOT change the WSC math in robust_rejection.py (no algorithm changes).
- Do NOT change other rejection paths (kappa, legacy winsor, etc.).
- Do NOT modify GUI, config, or add new user-facing options.
- Do NOT refactor unrelated code.
- Do NOT alter existing behavior for batch size logic (batch size = 0 and batch size > 1 behavior must remain untouched).
- Keep output dtype/shape identical to the current GPU stacker contract.

## Files (allowed edits)
- ✅ zemosaic_align_stack_gpu.py ONLY

## Implementation Plan (exact, surgical)

### A) [x] Add WSC VRAM preflight helper (new small functions)
In zemosaic_align_stack_gpu.py, add a helper near _resolve_rows_per_chunk:

1) `_wsc_estimate_bytes_per_row(n_frames, width, dtype_bytes=8, overhead=12) -> int`
- For channel-wise WSC, treat channels as 1.
- dtype_bytes must assume float64 internals (8 bytes).
- overhead is conservative to cover Xf, Xw, r, masks, huber weights, optional weights broadcast, etc.
- Start with overhead=12 (conservative but not insane).

2) `_resolve_rows_per_chunk_wsc(cp, height, width, channels, n_frames, plan, logger) -> int`
- Read existing rows hint / max bytes as an upper bound (don’t ignore plan), but WSC can further clamp down.
- Query VRAM: `free_b, total_b = cp.cuda.runtime.memGetInfo()`
- Choose budget: `budget = min(max_bytes if max_bytes>0 else free_b, int(free_b * 0.55))`
  (headroom ~45% to avoid WDDM paging)
- Compute rows_budget = budget // bytes_per_row_est
- rows_final = clamp(rows_budget, min=1, max=height)
- IMPORTANT: do not apply MIN_GPU_ROWS_PER_CHUNK=32 for WSC. min must be 1.
- If GPU safe mode is enabled, you may still cap rows_final, but do not increase it.

- Log once per tile (or once per call):
  `[P3][WSC][VRAM_PREFLIGHT] N=.. W=.. C=.. free=..MiB budget=..MiB est_row=..MiB rows=.. channelwise=yes`

### B) [x] Use WSC-aware rows_per_chunk
Currently rows_per_chunk is computed before algo/wsc_pixinsight is known.
Do NOT refactor heavily. Do one of these minimal approaches:

Option 1 (preferred): keep existing `_resolve_rows_per_chunk(...)` call, then if wsc_pixinsight:
- override: `rows_per_chunk = min(rows_per_chunk, _resolve_rows_per_chunk_wsc(...))`
- ensure `rows_per_chunk = max(1, rows_per_chunk)`

This preserves existing behavior for non-WSC.

### C) [x] Implement channel-wise WSC in the chunk loop
Inside the `if wsc_pixinsight:` branch in the for-loop:

Replace the single call:
- `data_gpu = cp.asarray(chunk_cpu, dtype=cp.float32)`
- `chunk_out, stats = wsc_pixinsight_core(cp, data_gpu, ..., weights_block=wsc_weights_block_gpu, return_stats=True)`
- `stacked[row_start:row_end] = cp.asnumpy(chunk_out)`

With channel-wise logic:

Pseudo:
- Determine `C = channels`
- For c in range(C):
  - Create CPU view: `cpu_c = chunk_cpu[..., c]` if C>1 else `chunk_cpu[..., 0]` or `chunk_cpu[:, :, :, 0]` accordingly.
    (Target shape must be (N, rows, W) for wsc_pixinsight_core.)
  - Upload: `data_gpu_c = cp.asarray(cpu_c, dtype=cp.float32)`
  - Slice weights_block for this channel:
    - If weights_block_gpu is None: pass None
    - Else if weights_block_gpu.ndim==4 and weights_block_gpu.shape[-1]==C:
        use `wb_c = weights_block_gpu[..., c]`  # shape (N,1,1)
      else:
        use `wb_c = weights_block_gpu` (already broadcastable, e.g. (N,), (N,1,1))
  - Call WSC core on 3D stack:
    `out_c, stats_c = wsc_pixinsight_core(cp, data_gpu_c, ..., weights_block=wb_c, return_stats=True)`
  - Download and store:
    `stacked[row_start:row_end, :, c] = cp.asnumpy(out_c)`

Stats accumulation:
- Sum clip_low_count/clip_high_count/valid_count across channels.
- Keep iters_used = max(iters_used across channels).
- (Fractions computed later from totals will remain meaningful.)

### D) [x] Add OOM backoff retry (WSC only)
Wrap the WSC chunk processing (per row_start) in a retry loop:

- `current_rows = rows_per_chunk` initially
- Attempt to process [row_start:row_end] with current_rows
- Catch `cp.cuda.memory.OutOfMemoryError` (and also generic Exception where str contains "Out of memory" as fallback)
  - Free CuPy pools:
    - `cp.get_default_memory_pool().free_all_blocks()`
    - `cp.get_default_pinned_memory_pool().free_all_blocks()` (safe optional)
  - Reduce: `current_rows = max(1, current_rows // 2)`
  - Log:
    `[P3][WSC][VRAM_BACKOFF] oom retry=k rows->current_rows row_start=...`
  - If current_rows == 1 and still OOM after retry => break and trigger CPU fallback (see below).
- IMPORTANT: ensure you retry the same row_start (do not skip rows, do not corrupt output).
- Bound retries (e.g. max 6-8 retries) to avoid infinite loops.

When backoff reduces rows, update the loop behavior safely:
- Easiest: convert the outer `for row_start in range(...)` into a `while row_start < height` only in the WSC branch,
  OR keep the for-loop but if you change current_rows you must re-run the same row_start and not advance.
Minimal approach:
- For WSC path only, replace the for-loop with a manual while-loop (local to this function), keeping non-WSC loop unchanged.

### E) [x] CPU fallback only when truly necessary
Current code falls back to CPU on any exception in the big try/except.
Keep that, but make it more specific for WSC:
- If after preflight + backoff you still cannot process even a 1-row chunk, THEN fallback CPU.
- Log a final reason:
  `[P3][WSC][CPU_FALLBACK] reason=vram_exhausted rows=1 ...`

## Acceptance Criteria
- Large RGB WSC stacks no longer crash GPU with OOM in normal cases; instead rows_per_chunk is reduced and processing completes on GPU.
- Peak VRAM is significantly reduced for RGB (channel-wise enabled).
- CPU fallback for WSC happens only if GPU cannot even process rows_per_chunk=1 after freeing pools.
- Non-WSC algorithms are unaffected (identical behavior).
- Logging clearly shows preflight, channel-wise mode, and any backoff.

## Notes
- Channel-wise WSC should be numerically identical to 4D WSC because all operations reduce along axis=0 (frames) and never mix channels.
- Keep output float32 as currently returned by wsc_pixinsight_core.
