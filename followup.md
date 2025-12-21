# Follow-up: Transparent GPU concurrency limiter (Grid Mode)

## Code checklist
- [x] No new user-facing parameter added (no GUI/config requirement).
- [x] gpu_concurrency computed only when GPU path is active.
- [x] memGetInfo guarded; fallback concurrency=1.
- [x] semaphore scope correct (shared by all workers in the run).
- [x] semaphore wraps ONLY GPU stacking call(s).
- [x] fallback-to-CPU behavior unchanged.

## Heuristic
- [x] safety_mult=2.5, fixed_overhead_mb=512, headroom=20%.
- [x] concurrency = clamp(floor(usable_mb/per_worker_mb), 1..4).
- [x] Log free/total VRAM + chosen concurrency.

## Pool cleanup
- [x] After GPU stack: free default memory pool blocks.
- [x] After GPU stack: free pinned memory pool blocks.
- [ ] Synchronize only if needed (avoid global perf hit).

## Validation
- [ ] Run Grid mode with auto workers (0), GPU enabled.
- [ ] Confirm no OOM.
- [ ] Confirm logs show limiter and computed values.
- [ ] Confirm output matches baseline (visual / stats).
