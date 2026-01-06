# Follow-up checklist — TRUE PixInsight-like WSC (CPU+GPU) with zero-regression guardrails

## 0) Safety rails before coding
- [x] Identify ALL callsites where reject_algo can be "winsorized_sigma_clip":
      - Phase 4.5: `zemosaic_worker.py` → `zemosaic_align_stack.stack_winsorized_sigma_clip()`
      - Phase 3 GPU stacker: `zemosaic_align_stack_gpu.py`
      - CPU stack core path: `zemosaic_align_stack.stack_aligned_images()`
- [x] Confirm current code uses:
      - quantile-ish winsorization (winsor_limits)
      - std / sigma_clipped_stats
      - possible frame-splitting streaming plan
- [x] Write down the current defaults used by GUI (sigma_low/high, winsor_limits, etc.) and ensure they remain stable.
      - Defaults (from `zemosaic_config.py`): kappa_low/high=3.0, winsor_limits="0.05,0.05", winsor_worker_limit=10,
        winsor_max_frames_per_pass=0, winsor_min_frames_per_pass=4, winsor_auto_fallback_on_memory_error=True,
        winsor_memmap_fallback="auto", winsor_split_strategy="sequential".

## 1) Implement the single shared core (backend-agnostic)
- [x] Place core implementation in a neutral location (`core/robust_rejection.py`) to prevent import cycles.
- [x] Confirm both `zemosaic_align_stack.py` and `zemosaic_align_stack_gpu.py` import and use the *same* helper from this location.
- [x] Add `wsc_pixinsight_core(xp, X_block, ...)` exactly per spec (median+MAD init, then clamp→winsorized mean→winsorized sigma iterations).
- [x] NaN/inf policy implemented (missing samples ignored per spec: valid mask + neutral replacement for median, or compaction).
- [x] Sigma floor (1e-10) implemented.
- [x] Convergence criteria implemented (eps_m/eps_s + stable bounds).

## 2) Wire CPU path (Phase 4.5 + CPU stack core)
- [x] Update `zemosaic_align_stack.stack_winsorized_sigma_clip()` to use spatial chunking only and call the shared core.
- [x] Ensure NO frames_per_pass / streaming-by-frames is used for pixinsight WSC.
- [x] Update `stack_aligned_images()` rejection path for winsorized_sigma_clip to route to the same shared core (or call stack_winsorized_sigma_clip internally).

## 3) Wire GPU path(s) (both places!)
- [x] Phase 4.5 GPU usage:
      - Ensure the WSC GPU route also calls the same shared core with `xp=cupy`.
- [x] Phase 3 GPU stacker (`zemosaic_align_stack_gpu.py`):
      - Ensure reject_algo winsorized_sigma_clip calls the same shared core (via import) and does NOT use quantiles/partition in pixinsight mode.
- [x] If any GPU failure or parity risk: fallback to CPU for WSC only (log one line).
- [x] On GPU OOM/exception, fallback to CPU (WSC only) and continue the pipeline without crashing.

## 4) Legacy mode (optional but recommended)
- [x] Add env/config switch `ZEMOSAIC_WSC_IMPL=pixinsight|legacy_quantile`.
- [x] Default pixinsight.
- [x] legacy_quantile keeps the previous behavior unchanged (do not “tweak” it).

## 5) Parity + determinism enforcement
- [x] Ensure CPU and GPU use the same dtype strategy (document it).
- [x] Add strict parity test:
      - small seeded stack → CPU output float32 and GPU output float32 must be identical (max_abs_diff == 0).
- [x] Add a quick dev script entry (or extend existing compare script) to compare CPU/GPU WSC on synthetic data.

## 6) Quality tests
- [ ] Cosmic ray suppression test passes.
- [ ] IFN-like faint signal test:
      - WSC must not be worse than kappa-sigma (define a numeric criterion: e.g. mean signal preserved within X% and noise not inflated).
- [ ] Existing test suite passes unchanged.

## 7) Real-data validation checklist
- [ ] Run one representative mosaic dataset:
      - Compare master tiles WSC vs kappa-sigma visually.
      - Check background texture (IFN), and check outlier removal (sat trails/cosmic rays).
- [ ] Confirm runtime is sane (no pathological slowdowns).
- [ ] Confirm GPU still used where expected; if WSC falls back to CPU, log clearly why.

## Done definition (must check all)
- [ ] No GUI changes.
- [ ] No batch size semantics change.
- [ ] No regressions in other algorithms/modes.
- [ ] True PixInsight-like WSC behavior (no quantile core).
- [ ] CPU/GPU parity strict test passes (max_abs_diff == 0 on float32).

## Local validation commands (must run before marking done)
- python -m pytest -q
- python compare_cpu_gpu_stack.py   (or equivalent dev script) on a small seeded synthetic stack:
    - verify WSC CPU vs GPU parity per env: STRICT (default) or NUMERIC
- run one real dataset in WSC and kappa-sigma to visually confirm faint background is preserved
