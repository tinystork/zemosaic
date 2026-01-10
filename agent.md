# Mission: Fix WSC (PixInsight-like) memory blowups on large clusters (Phase 3 master tiles)

## Context
On large clusters, Phase 3 stacking fails with:
- GPU pinned-memory OOM then `cudaErrorAlreadyMapped`
- CPU fallback OOM in `wsc_pixinsight_core()` trying to allocate ~8.18 GiB for shape (N, rows, W, 3) float64.

Example failure:
shape (649, 256, 2203, 3) float64 => ~8.18 GiB (before extra masks/buffers).

## Root causes (must address)
1. WSC CPU fallback builds a full RGB 4D chunk: (N, rows, W, 3) then converts to float64 and uses xp.where -> massive allocations.
2. WSC rows preflight `_resolve_rows_per_chunk_wsc()` allows `rows_hint` to override safety when `rows_budget <= 0` (falls back to rows_cap instead of 1).
3. Current chunking is "rows only": memory still scales with N. For N up to 20000, even rows=1 is too big if we stack all frames at once.
4. GPU errors `cudaErrorAlreadyMapped` after hostAlloc OOM should be treated as OOM-like and trigger pool purge + fallback, not a repeated failure loop.

## Scope / files
- `zemosaic_align_stack_gpu.py`
  - [x] Fix `_resolve_rows_per_chunk_wsc()` hint logic (hint is a cap, never a fallback when budget <= 0).
  - [x] Ensure GPU WSC builds per-channel chunks without creating full RGB 4D `chunk_cpu` when possible.
  - [x] Update CPU fallback `_wsc_pixinsight_stack_cpu()` to be channelwise and to use the new streaming WSC for large N.
  - [x] Treat `cudaErrorAlreadyMapped` (and similar pinned-memory errors) as OOM-like: purge memory pools and fallback cleanly.

- `core/robust_rejection.py` (or wherever `wsc_pixinsight_core` lives in repo)
  - [x] Reduce allocations: replace `Xf = xp.where(valid, Xf, xp.nan)` with in-place masking when backend supports it.
  - [x] Add a new streaming implementation that does NOT require materializing (N, rows, W[,C]) for large N:
    - [x] `wsc_pixinsight_core_streaming_numpy(frames, rows_slice, channel, ...)` or equivalent.
    - [x] Must support optional frame weights (broadcast forms currently accepted) and sky_offsets.
    - [x] Must preserve NaNs as invalid mask.

- `zemosaic_align_stack.py`
  - [x] Update `_wsc_pixinsight_stack_numpy()` to use channelwise and optionally the streaming WSC for large N.

## Hard constraints
- NO regression on SDS mode and Grid mode (do not change their algorithms/paths).
- Keep existing behavior for batch size = 0 and batch size > 1 (do not modify that logic).
- Keep existing WSC small-stack behavior and parity tests intact for small N (streaming activates only when needed).

## Implementation plan (suggested)
### [x] A) Fix rows preflight (critical, small patch)
In `_resolve_rows_per_chunk_wsc()`:
- Compute `rows_budget = floor(budget / bytes_per_row)`.
- If `rows_budget <= 0`, set `rows_final = 1`.
- If `rows_hint` exists, apply `rows_final = min(rows_final, rows_hint)` (hint is cap).
- Ensure logs reflect actual `rows_final`.

### [x] B) Make CPU fallback channelwise
In `_wsc_pixinsight_stack_cpu()`:
- If frames are RGB, loop over channels:
  - Build per-channel chunk `(N, rows, W)` not `(N, rows, W, 3)`.
  - Pass `weights_block[..., c]` if weights are channelwise.
- This must work with `sky_offsets`.

### [x] C) Add streaming WSC for large N
Add a streaming code path used when:
- N is large (e.g. > 512), OR
- estimated chunk bytes exceed a configurable fraction of available RAM.

Streaming algorithm requirements:
- Must not stack all frames at once.
- Use sample-based init for (m, sigma) to avoid full-stack median/MAD:
  - sample frames deterministically via linspace indices capped at e.g. 256.
- For each iteration:
  - Pass1: compute winsorized mean (weighted/unweighted) -> m_new (accumulate sums over frame blocks).
  - Pass2: compute sigma_new (Huber IRLS style) -> sigma_new (accumulate numer/denom over frame blocks).
- Stop on convergence similar to existing tolerances.
- Output: for pixels with <2 valid samples use fallback_mean (computed via streaming count/sum).
- Return stats (clip counts can be approximated or computed in a final pass; keep existing stats fields).

### [x] D) GPU pinned-memory error hardening
- Extend `_wsc_is_oom_error()` to treat `cudaErrorAlreadyMapped` / pinned hostAlloc failures as OOM-like.
- On such errors: purge `cp.get_default_memory_pool().free_all_blocks()` and `cp.get_default_pinned_memory_pool().free_all_blocks()`, synchronize, then either backoff rows or fallback to CPU streaming.

## Acceptance criteria
1. Large cluster case no longer allocates multi-GB arrays on CPU fallback; must complete without MemoryError.
2. `_resolve_rows_per_chunk_wsc()` returns 1 (or small) for big N even if `gpu_rows_per_chunk` hint is 256.
3. For small N (<= ~64), outputs remain unchanged vs current implementation (within existing parity thresholds).
4. No changes in SDS and Grid mode results or code paths.
5. Logs clearly indicate when streaming WSC is used:
   - `[P3][WSC][STREAM] enabled=1 reason=N>threshold N=... block=... rows=...`

## Notes
Prefer surgical changes. Add small unit tests if repo has a tests folder; otherwise add a lightweight self-test function guarded by `if __name__ == "__main__":` only if acceptable.
