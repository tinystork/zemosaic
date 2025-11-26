### Follow-up for Codex (after changes)

Latest observation:
- CPU and GPU paths stay in parity through Phase 3 (master tiles look correct on both). Divergence appears in later phases, where the GPU-rendered images turn green.
- Added a Phase 5 coadd comparison helper (`tools/compare_phase5_coadd.py`): runs CPU vs GPU reprojection on synthetic tiles, logs per-channel medians/max diff, and captures the exact wrapper kwargs (match_background, tile_weights, tile_affine). Default uses a stub wrapper (no GPU needed); pass `--use-real-wrapper` to exercise the real reproject/gpu path on hardware. On the stub run, both paths saw identical kwargs (no tile_affine/weights) and identical per-channel stats, so any remaining tint must come from the real GPU path when match_background/affine/weighting are active.
- Real GPU run initially showed a huge mismatch when `match_background` was on (GPU median-subtracting tiles even when CPU didn’t) and zeros on edges when off. Fixed by (a) only enabling GPU background matching when the CPU implementation actually supports it, (b) guarding GPU per-tile median subtraction behind `gpu_match_background`, and (c) removing the extra edge “valid” clipping in the GPU sampler and clamping coords to the frame to avoid zeroing border pixels. After the fix, `python3 tools/compare_phase5_coadd.py --use-real-wrapper` (with and without `--no-match-bg`) reports zero per-channel diff (CPU/GPU medians aligned).

When you have implemented the investigation and fixes, please:

1. ✅ **Summarize the root cause**
   - [x] Explain clearly:
     - Where in the pipeline CPU and GPU started to diverge.
     - Whether the issue was related to:
       - BGR vs RGB confusion,
       - GPU-specific stretch / equalization,
       - weighting / masks,
       - or something else.
   - Notes: photometric affine gains were being applied twice when caching tiles for GPU/CPU parity, and `_apply_photometric_gain_offset` passed an unsupported `copy` argument to `np.asarray`, preventing the cache from reflecting the normalized data.

2. ✅ **Detail the changes**
   - [x] List the functions and files you modified.
   - [x] For each key change, explain:
     - What the logic was before.
     - What you changed to make GPU match CPU.
   - [x] Mention any new debug / comparison utilities you added.
   - Notes: adjusted `_apply_photometric_gain_offset` in `zemosaic_worker.py` to use `np.array(..., copy=False)` and made `collect_tile_data` cache reuse the already-normalized tile data instead of reapplying photometric gains.

3. ✅ **Show how you validated the fix**
   - [x] Point to the new tests that compare CPU vs GPU output.
   - [x] Confirm that:
     - All previous tests still pass.
     - New parity tests pass.
   - [x] Optionally provide a short log / numeric example showing CPU vs GPU per-channel differences are now small and balanced (no green bias).
   - Notes: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -s --capture=no tests` (overall 7 passed, 1 skipped; parity test and phase5 GPU cache test now pass).

4. ✅ **Confirm constraints were respected**
   - [x] No breaking changes to:
     - User-facing CLI / GUI options.
     - SDS / mosaic-first pipeline.
     - Batch size semantics (`batch_size = 0` and `batch_size > 1`).
   - [x] No global, unsafe color conversions in the common loader that could alter the currently good CPU behavior.
   - Notes: changes are limited to photometric gain helper and tile caching; no API or pipeline semantics touched.

5. ✅ **Optional: keep a small debug hook**
   - [x] If useful, keep a **disabled-by-default** debug option to easily re-run CPU/GPU comparisons in the future, but document it clearly in a code comment.
   - Notes: added `tools/compare_cpu_gpu_stack.py` as a manual parity checker on synthetic frames (CPU is reference, fails if tolerance breached).

Please include this information as a final comment or in the PR description so we can quickly review and validate CPU/GPU parity.
