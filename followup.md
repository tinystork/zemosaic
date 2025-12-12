# Follow-up checklist: SDS GPU nanpercentile fallback

## 1) Patch implementation (zemosaic_utils.py)
- [ ] Locate `_sds_cp_nanpercentile(arr_gpu, percentiles, axis=None)`
- [ ] Keep existing branches:
      - cp.nanpercentile if available
      - cp.nanquantile if available
- [ ] Add third fallback:
      - supports scalar p and small array/list p
      - supports axis=0 (primary use case)
      - ignores NaNs (do NOT treat NaNs as zeros)
      - uses cp.sort + cp.take_along_axis (or equivalent) + per-pixel finite counts
      - handles cnt==0 safely

## 2) Correctness notes
- [ ] Percentile definition: use floor-based index on sorted finite samples
- [ ] For cnt==0: return 0.0 (float32) to match downstream nan_to_num behavior
- [ ] Ensure output dtype is float32 where reasonable

## 3) Guardrails
- [ ] DO NOT touch other modules (no edits in worker, grid_mode, align_stack_cpu, etc.)
- [ ] DO NOT change any algorithm knobs / defaults outside this function
- [ ] No new dependencies

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
- [ ] Commit message: "Fix SDS GPU nanpercentile fallback"
