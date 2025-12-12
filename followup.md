# Follow-up checklist: SDS GPU nanpercentile fallback

## 1) Patch implementation (zemosaic_utils.py)
- [x] Locate `_sds_cp_nanpercentile(arr_gpu, percentiles, axis=None)`
- [x] Keep existing branches:
      - cp.nanpercentile if available
      - cp.nanquantile if available
- [x] Add third fallback:
      - supports scalar p and small array/list p
      - supports axis=0 (primary use case)
      - ignores NaNs (do NOT treat NaNs as zeros)
      - uses cp.sort + cp.take_along_axis (or equivalent) + per-pixel finite counts
      - handles cnt==0 safely

## 2) Correctness notes
- [x] Percentile definition: use floor-based index on sorted finite samples
- [x] For cnt==0: return 0.0 (float32) to match downstream nan_to_num behavior
- [x] Ensure output dtype is float32 where reasonable

## 3) Guardrails
- [x] DO NOT touch other modules (no edits in worker, grid_mode, align_stack_cpu, etc.)
- [x] DO NOT change any algorithm knobs / defaults outside this function
- [x] No new dependencies

## 4) Smoke test (minimal)
- [ ] Add a tiny internal self-test block (optional) or run interactive snippet:
      - Create a cp array shaped (N,H,W) with some NaNs
      - Request percentiles 5 and 95 along axis=0
      - Ensure it returns finite arrays and no exception
- [ ] Run the existing SDS example run; confirm logs show GPU helper completing and no fallback error.

## 5) Expected log outcome
Previously:
- gpu_fallback_runtime_error (nanpercentile missing)
- global_coadd_info_helper_cpu_resume (reason helper_failed)

After patch:
- No gpu_fallback_runtime_error for nanpercentile/nanquantile
- global coadd finishes on helper route.

## 6) Commit
- [x] Commit message: "Fix SDS GPU nanpercentile fallback"
