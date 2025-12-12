# Mission: Fix SDS GPU fallback (CuPy missing nanpercentile/nanquantile) WITHOUT touching other pipelines

## High-level goal
Restore GPU path for SDS / SupaDupStack (mega-tile pipeline) by removing the hard dependency on `cupy.nanpercentile` / `cupy.nanquantile`.

The current run falls back to CPU with:
- "module 'cupy' has no attribute 'nanpercentile'"
- then "CuPy missing nanpercentile/nanquantile"
This must be fixed so the GPU helper route does not fail on systems where CuPy lacks these functions.

## Non-goals / strict constraints
- DO NOT modify classic pipeline behavior.
- DO NOT modify grid mode behavior.
- DO NOT refactor unrelated code or change algorithms outside SDS helper compatibility.
- Keep changes minimal and localized.
- Do not change batch-size semantics (especially the known-good behavior “batch size = 0” and “batch size > 1”).

## Where the bug is
`zemosaic_utils._sds_cp_nanpercentile()` currently:
- uses `cp.nanpercentile` if present
- else uses `cp.nanquantile` if present
- else raises RuntimeError("CuPy missing nanpercentile/nanquantile")
On some installations, CuPy lacks both -> GPU helper raises -> worker falls back to CPU.

## Required fix
Implement a **third fallback** in `_sds_cp_nanpercentile()`:
- If both `nanpercentile` and `nanquantile` are missing, compute NaN-ignoring percentiles using CuPy primitives (no NumPy fallback).
- Must support scalar percentile (float) and small percentile arrays (e.g., two values for winsorization).
- Must support `axis=0` (this is the SDS winsorized path use case).

Suggested algorithm (NaN-safe percentile via sort):
1. Let `x = arr_gpu`.
2. Build `finite = cp.isfinite(x)`.
3. Replace non-finite values with `+inf` so they sort to the end: `x2 = cp.where(finite, x, cp.inf)`.
4. Sort along `axis=0`: `xs = cp.sort(x2, axis=axis)`.
5. Count finite per pixel: `cnt = cp.sum(finite, axis=axis)`.
6. Compute index k per pixel for percentile p:
   - k = floor((p/100) * (cnt-1)) clipped to [0, max(cnt-1,0)]
7. Use `cp.take_along_axis(xs, k[None,...], axis=0)` to gather.
8. For `cnt==0`, return 0.0 (or cp.nan then later caller nan_to_num; pick what matches current behavior best).

Edge cases:
- Works when cnt==1
- Works when many NaNs
- Must not crash if `cp.errstate` is missing (use existing `_xp_errstate()` patterns elsewhere; do not introduce new cp.errstate usage).

## Files to modify
- `zemosaic_utils.py` (only): implement the extra fallback in `_sds_cp_nanpercentile()`.

## Acceptance criteria
- SDS run no longer falls back to CPU due to nanpercentile/nanquantile missing.
- GPU helper route (gpu_reproject) completes at least on the provided small example (10 frames / 3 channels).
- No behavior changes in classic mode and grid mode.

## Logging (minimal)
- If using third fallback, optionally log once at DEBUG level (guarded) like:
  "CuPy lacks nanpercentile/nanquantile -> using SDS sort-based nanpercentile fallback"
Do not spam per-chunk.

## Manual test
Run the same example that currently triggers fallback and verify logs no longer show:
- gpu_fallback_runtime_error with nanpercentile/nanquantile
and verify GPU path is used.

## Deliverables
- One commit with message: "Fix SDS GPU nanpercentile fallback"
