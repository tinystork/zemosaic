# Follow-up checklist: Phase 3 GPU WSC VRAM guard + channel-wise execution

## Implementation
- [x] Edit ONLY `zemosaic_align_stack_gpu.py` (no other files).
- [x] Add WSC VRAM preflight helpers:
  - [x] `_wsc_estimate_bytes_per_row(...)`
  - [x] `_resolve_rows_per_chunk_wsc(cp, ...)` using memGetInfo + conservative headroom
  - [x] Log `[P3][WSC][VRAM_PREFLIGHT] ... channelwise=yes`
- [x] Ensure WSC can use rows_per_chunk < 32 (down to 1) without changing global MIN_GPU_ROWS_PER_CHUNK for other algos.
- [x] Implement channel-wise WSC for RGB:
  - [x] Upload/process (N,rows,W) per channel instead of (N,rows,W,C)
  - [x] Slice `weights_block` per channel when shape is (N,1,1,C)
  - [x] Store into `stacked[row_start:row_end, :, c]`
  - [x] Accumulate WSC stats across channels (sum counts, max iters_used)
- [x] Add OOM backoff retry for WSC chunks:
  - [x] Catch `cp.cuda.memory.OutOfMemoryError`
  - [x] Free CuPy pools
  - [x] Halve rows_per_chunk and retry same row_start
  - [x] Bound retries (no infinite loops)
  - [x] Log `[P3][WSC][VRAM_BACKOFF] ...`
- [x] CPU fallback only if still OOM at rows=1:
  - [x] Log `[P3][WSC][CPU_FALLBACK] reason=vram_exhausted ...`

## Regression checks
- [ ] Non-WSC GPU paths unchanged (kappa, legacy winsor, etc.).
- [ ] Output shape/dtype unchanged (float32, (H,W,C) / (H,W) when drop_channel).
- [ ] No changes to GUI/config.
- [ ] Confirm batch size behavior remains unchanged (batch size = 0 and batch size > 1 untouched).

## Manual validation (recommended)
- [ ] Re-run the problematic large master tile (e.g. N~151 RGB WSC) and verify:
  - [ ] No GPU OOM
  - [ ] No WDDM freeze
  - [ ] Preflight log shows reduced rows_per_chunk
  - [ ] GPU path completes without CPU fallback (or only falls back if truly impossible)
