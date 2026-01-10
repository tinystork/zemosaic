# Follow-up checklist: Validate WSC memory fixes (no regressions)

## [ ] 1) Quick sanity: rows preflight cannot be overridden by hint
- Create/force a ParallelPlan with `gpu_rows_per_chunk=256`.
- Run a WSC stack with large N (or simulate with small images but big N).
Expected log (example):
- `[P3][WSC][VRAM_PREFLIGHT] ... N=649 ... rows=1 ...`
NOT rows=256.

## [ ] 2) Validate CPU fallback is channelwise
Trigger GPU failure intentionally (e.g. set very low VRAM budget / safe mode) then check logs:
- Should see per-channel processing in CPU fallback (or streaming log).
- Must NOT attempt to allocate shape (N, rows, W, 3) float64.

## [ ] 3) Validate streaming WSC activates for huge N
Use synthetic test (small images, huge N like 20000):
- Build frames of shape (H=2, W=64, C=3) float32, add some NaNs.
- Call the WSC stack path.
Expected:
- streaming enabled log appears
- process completes without MemoryError
- output has correct shape and finite values where expected.

## [ ] 4) Regression check on small stacks (parity)
- Run existing WSC parity check (cpu vs gpu) on small stacks.
Expected:
- parity still OK within existing mode thresholds.
- streaming should NOT activate.

## [ ] 5) Integration smoke tests (critical)
- Run one SDS mosaic (small dataset) end-to-end.
- Run one Grid mode mosaic end-to-end.
Expected:
- identical outputs (or same failure/success behavior as before), no code-path changes.

## [ ] 6) Confirm pinned-memory hardening
On a GPU machine, try to reproduce the old pinned error path:
- If `cudaErrorAlreadyMapped` occurs, it must be handled like OOM:
  - pools freed
  - either backoff rows or clean fallback to CPU streaming
  - no infinite retry loop

## Deliverables to include in PR
- Short commit message summary:
  - "Fix WSC rows hint override + channelwise CPU fallback"
  - "Add streaming WSC for large N to cap memory"
  - "Harden GPU pinned-memory errors (AlreadyMapped treated as OOM-like)"
- Mention explicitly: "No changes to SDS or Grid code paths."
