# Follow-up Checklist — Fix Two-Pass CPU/GPU Split + Coverage Semantics

## 0) Guardrails
- [ ] Do NOT touch batch-size behavior (batch size = 0 vs >1 must remain unchanged).
- [ ] Do NOT introduce any new logging system/handlers/files. The GUI already has a log-level dropdown; respect existing logger usage.
- [ ] Keep changes limited to:
  - `zemosaic_utils.py` (`gpu_reproject_and_coadd_impl`)
  - `zemosaic_worker.py` (`run_second_pass_coverage_renorm`)

---

## 1) Patch A — GPU coverage must match CPU footprint semantics

### File: `zemosaic_utils.py`
### Function: `gpu_reproject_and_coadd_impl(...)`

#### Locate mean/fast path where coverage is computed:
Current buggy code (mean path):
- `coverage_gpu = cp.clip(weight_sum_gpu / float(max(1, n_inputs)), 0.0, 1.0)`

#### Change to:
- `coverage_gpu = weight_sum_gpu`
- Sanitize:
  - `coverage_gpu = cp.nan_to_num(coverage_gpu, nan=0.0, posinf=0.0, neginf=0.0)`
- Ensure returned dtype float32 (already done at return).

#### Why
CPU `reproject_and_coadd` returns footprint/coverage in “sum of weights” units (can exceed 1).
GPU must return the same to prevent Two-Pass gain/renorm from applying different effective scaling per channel.

#### Quick self-check
- [ ] Grep to ensure no other normalization exists in GPU mean path.
- [ ] Ensure winsor/kappa-sigma paths already return non-normalized coverage (leave them as-is).

---

## 2) Patch B — Two-Pass must never mix backends per channel

### File: `zemosaic_worker.py`
### Function: `run_second_pass_coverage_renorm(...)`

#### Find channel assignment block
It currently builds `channel_tasks` where only one channel gets GPU:
- `gpu_assigned = False ... use_gpu_flag = use_gpu and not gpu_assigned`

#### Replace with coherent backend strategy
Implement logic:

**If `n_channels == 1`:**
- Keep existing behavior: try GPU if requested, fallback to CPU if needed.

**If `n_channels > 1`:**
- If `use_gpu` is True:
  1) Attempt **GPU for all channels** (sequential, not threaded).
  2) If any channel fails or falls back, log one warning and rerun **ALL channels CPU**.
- If `use_gpu` is False:
  - Run CPU for all channels (threaded ok).

#### Implementation pattern (recommended)
Create helper:
- `_run_channels(use_gpu_flag: bool) -> (mosaic_channels, coverage_channels)`
  - loops channels 0..n-1
  - calls `_process_channel(ch, use_gpu_flag)`
  - collects arrays
  - raises on first failure

Then:
- try:
  - if use_gpu: `_run_channels(True)`
- except:
  - warning: `[TwoPass] GPU reprojection failed on one or more channels; rerunning all channels on CPU to avoid mixed backend`
  - `_run_channels(False)`

#### Logging expectations
At DEBUG:
- [ ] Per-channel log already prints `cov_stats(min,max)`; keep it.
Add one summary line (DEBUG or INFO):
- [ ] `[TwoPass] Two-pass reprojection backend: gpu_all=True` (or False)

---

## 3) Sanity checks (fast)
- [ ] Run `python -m py_compile zemosaic_utils.py zemosaic_worker.py`
- [ ] Search for the old “gpu_assigned” logic; ensure removed.
- [ ] Search for the old GPU coverage normalization in mean path; ensure removed.

---

## 4) Minimal runtime test (single dataset)
Run legacy/classic mosaic with Two-Pass enabled and GPU available.

Confirm logs:
- [ ] All channels report `gpu=True` OR all channels report `gpu=False` (never mixed).
- [ ] `cov_stats max` are comparable scale across channels (e.g. >1 when overlaps exist; not 0.7 on one channel and 12 on others).
- [ ] After Two-Pass, RGB ratios do not explode (no strong green cast).

---

## 5) Done Definition
- [ ] Green tint regression resolved in legacy mode.
- [ ] No behavior change outside Two-Pass / GPU coverage semantics.
- [ ] CPU-only still works.
- [ ] No new logging system added.
````

