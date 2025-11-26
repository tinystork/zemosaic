✅ agent.md — Ready for Codex Max / High

Mission: Investigate and fix the **quality difference between CPU and GPU pipelines** so that, for the same input, **GPU output matches CPU output** (same geometry and colors, up to small numerical noise). The GPU path should be **a faster drop-in replacement**, not a different look.

Branch / scope: current ZeMosaic working branch (do **not** refactor the whole project, focus on CPU vs GPU parity for stacking & mosaic).

---

## 🎯 Goals

- [ ] Find **where** and **why** the GPU result diverges from the CPU result.
- [ ] Ensure **identical stacking logic** on CPU and GPU (same math, same reference frame, same photometric behavior).
- [ ] Fix the GPU pipeline so that:
  - [ ] There is **no green tint** or color bias vs CPU.
  - [ ] There is no geometric distortion / misalignment specific to GPU.
  - [ ] On a given dataset, CPU and GPU produce **visually identical** images, with only small floating-point differences.
- [ ] Provide a **small comparison harness + tests** to keep CPU/GPU in sync over time.

---

## 🧩 Context

The project has two stacking/mosaic pipelines:

- **CPU reference**:
  - Main logic in:  
    - `zemosaic_align_stack.py`
    - `zemosaic_worker.py` (orchestrates phases, preview generation, etc.)
  - This path is considered **correct** in terms of geometry, photometry, and color balance.

- **GPU pipeline**:
  - Main logic in:
    - `zemosaic_align_stack_gpu.py`
    - Any GPU-specific helper or wrapper (e.g. in `parallel_utils.py` or similar).
  - The GPU path should be **mathematically equivalent** to the CPU path, only faster.

Current issue:

- The **GPU path sometimes produces greenish / color-shifted images**, while the CPU path yields correct colors.
- The expectation is: **under the same configuration (same input data, same params), CPU and GPU outputs should match**.

Important note:

- Some external analysis suggested a potential **BGR vs RGB mix-up** (e.g., OpenCV vs FITS channel order). That hypothesis is **unconfirmed**.
- The CPU path currently looks good, so any change that would affect the **common loader / shared code** must be **carefully validated** and not break CPU output.

---

## 🗂️ Files to inspect (non-exhaustive, but high priority)

- [ ] `zemosaic_align_stack.py`  
- [ ] `zemosaic_align_stack_gpu.py`  
- [ ] `zemosaic_worker.py`  
- [ ] `zemosaic_time_utils.py` (if used in Phase 3/5 orchestration)
- [ ] `parallel_utils.py` (for GPU orchestration / CPU/GPU auto-tune)
- [ ] `zemosaic_config.py` (global GPU toggles, phase-specific flags)
- [ ] Any GPU helper modules referenced by `zemosaic_align_stack_gpu.py`
- [ ] Image loading / preview:
  - `zemosaic_utils.py` (e.g. `load_and_validate_fits`, `debayer_image`, etc.)
  - Any preview / PNG/PNG-export functions in `zemosaic_worker.py`

---

## 🔍 Tasks (step-by-step, with checkboxes)

### 1. Map CPU vs GPU pipelines

- [ ] Identify all code paths involved when:
  - CPU stacking is used.
  - GPU stacking is used.
- [ ] For each **phase** (especially stacking / mosaic / stretch / RGB normalization), list:
  - [ ] The CPU function(s) called.
  - [ ] The GPU function(s) called.
  - [ ] Any differences in parameters, defaults, numeric order of operations.

Goal: have a clear diagram of **which CPU function corresponds to which GPU function**, and where they diverge.

---

### 2. Build a comparison harness (CPU vs GPU)

Create a small, self-contained **debug / test utility** (can be under `tests/` or a dedicated module) that:

- [ ] Takes a small set of FITS tiles (real or synthetic).
- [ ] Runs **the CPU pipeline** on that set.
- [ ] Runs **the GPU pipeline** on the exact same set, with the same configuration.
- [ ] Compares outputs:
  - [ ] Per-channel statistics (min, max, mean, median) for R, G, B.
  - [ ] Per-channel histograms or percentiles.
  - [ ] Pixel-wise differences: max abs diff, mean abs diff, maybe a norm per channel.
- [ ] Emits a clear summary:
  - e.g. “Max difference per channel (R,G,B) = (…)”,
  - “Median value per channel (CPU vs GPU)”,
  - “Any channel significantly off compared to the other two?”

You can expose this as:

- a small Python function used by tests, and/or
- a CLI/debug entry point (e.g. `python -m zemosaic.debug.compare_cpu_gpu ...`).

---

### 3. Locate the source(s) of divergence

Using the comparison harness, drill down phase by phase:

- [ ] Compare **stacked tiles** CPU vs GPU *before* any stretch/normalization.
- [ ] Compare **RGB equalization / white balance** steps CPU vs GPU.
- [ ] Compare **final stretched images** CPU vs GPU (what the user actually sees).
- [ ] Check:
  - [ ] Are the differences already visible on the stacked data?
  - [ ] Or do they appear during equalization / stretching / background subtraction?
  - [ ] Is there a systematic excess in the **G channel** vs R/B on the GPU path?

Specifically investigate:

- [ ] Any GPU-specific implementation of **RGB normalization / equalization**  
  (e.g. functions named like `_poststack_rgb_equalization` or similar).
- [ ] Any GPU-specific **stretch** code  
  (e.g. `stretch_auto_asifits_like_gpu` vs the CPU `stretch_auto_asifits_like`).
- [ ] Any difference in the way **weights** or **masks** are applied on CPU vs GPU.

If an earlier hypothesis about **BGR vs RGB** is correct:

- [ ] Confirm it with **hard evidence**:
  - Show that at some entry point, data is in BGR for GPU but in RGB for CPU.
  - Show that a channel swap explains the green tint.
- [ ] Only then, design a **minimal, controlled fix** (see Constraints below).

---

### 4. Design and implement the fix

Once the root cause is identified:

- [ ] Align the GPU logic with the CPU logic:
  - Same order of operations,
  - Same formulas,
  - Same default parameters.
- [ ] If the bug is a color space issue (e.g. BGR vs RGB):
  - [ ] Add a **local, explicit conversion** at the correct boundary (e.g. just before feeding OpenCV, or right after retrieving OpenCV output).
  - [ ] Do **not** add a global “convert all 3-channel images BGR→RGB” in a shared loader unless absolutely proven necessary and validated against the CPU path.
- [ ] If the bug is in a GPU-specific stretch or equalization:
  - [ ] Modify the GPU implementation so it matches the CPU math.
  - [ ] Add comments explaining the rationale and link to the CPU reference implementation.

---

### 5. Add regression tests for CPU/GPU parity

Add automated tests (e.g. with `pytest`) that:

- [ ] Use a small deterministic test dataset (synthetic or bundled FITS tiles).
- [ ] Run both CPU and GPU pipelines with the same configuration.
- [ ] Assert that:
  - [ ] CPU and GPU outputs have the same **shape** and dtype.
  - [ ] Per-channel difference (R,G,B) is below a reasonable epsilon (for example, `max_abs_diff < 1e-3` or similar, depending on scaling).
  - [ ] There is no systematic channel imbalance (e.g. G much higher on GPU than CPU).
- [ ] Flag clearly that the CPU path is the **reference**.

These tests should fail on the current broken state and pass after the fix.

---

## 🚫 Constraints / Non-goals

- [ ] **Do not change** the overall user-facing API, CLI options, or GUI behavior (except maybe adding a debug flag / advanced expert option).
- [ ] **Do not break** existing behavior of:
  - SDS / mosaic-first pipeline.
  - Batch size logic, especially the semantics of:
    - `batch_size = 0`
    - `batch_size > 1`
- [ ] **Do not** refactor unrelated parts of the codebase or introduce large architecture changes.
- [ ] Avoid “magic” global conversions (e.g. blind `BGR→RGB` on any 3-channel array) that could silently change good CPU behavior.
- [ ] Keep changes as **local and well-documented** as possible.

---

## ✅ Acceptance Criteria

- [ ] On the same dataset (same parameters), CPU and GPU images are **visually indistinguishable**:
  - same framing,
  - same background,
  - same color balance (no greenish cast from GPU).
- [ ] The comparison harness reports **small numeric differences only**, consistent with float math / GPU vs CPU differences.
- [ ] New tests for CPU/GPU parity are in place and **pass**.
- [ ] Existing tests still pass.
- [ ] The GPU path remains clearly faster than CPU for realistic workloads.

Please summarize in the PR / final note:

- What the precise root cause was.
- What code was changed.
- How the new tests guarantee CPU/GPU parity going forward.
