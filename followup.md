# followup.md

## Step-by-step tasks for fixing RGB normalization in the GPU Phase 5 path

Use this checklist to implement and verify the changes.

---

### 1. Understand the CPU photometric normalization in Phase 5

- [x] In `zemosaic_worker.py` and `zemosaic_utils.py`, locate:
  - The non-SDS Phase 5 pipeline:
    - Tile/mega-tile loading,
    - Inter-tile overlap analysis,
    - Final `reproject & coadd` call.
- [x] Identify the exact sequence of CPU operations that:
  - compute RGB gains / background levels for tiles,
  - apply these gains,
  - perform any coverage-based renormalization.
- [x] Add brief comments/docstrings summarizing this sequence for future reference.

---

### 2. Inspect the GPU Phase 5 path and compare

- [x] Find the GPU-enabled path in Phase 5:
  - where `use_gpu=True` or equivalent is passed,
  - or where GPU-specific functions are called for reprojection/coadd.
- [x] Compare with CPU path:
  - note which normalization steps are **present** in CPU but **missing or different** in GPU.
- [x] Pay particular attention to:
  - per-tile RGB gains,
  - background matching,
  - coverage-based gain maps.

---

### 3. Create shared photometric normalization helpers

- [x] Refactor existing CPU normalization code into reusable helpers, for example:

  ```python
  def compute_photometric_gains_and_offsets(...):
      ...

  def apply_photometric_normalization(image_or_tiles, gains, backend):
      ...
(Use actual function names and signatures of the project.)

 Ensure helpers:

encapsulate the full normalization logic (RGB gains, offsets, clipping, etc.),

 can work with both NumPy arrays and CuPy arrays, or at least can be called from both CPU and GPU flows.

- [x] Integrate normalization helpers into the GPU path
 Modify the GPU Phase 5 path so that:

it calls compute_* helpers to derive the same gains/offsets as CPU,

it applies these gains using apply_* helpers to the GPU tiles or mosaic.

 Decide the best placement:

pre-warp per-tile normalization (recommended if that’s what CPU does),

post-warp normalization on the full mosaic (if CPU does that),

or a combination, depending on current design.

 If implementing full normalization on GPU is too complex:

implement a hybrid approach:

run GPU for heavy reprojection,

transfer intermediate result to CPU for final normalization using existing NumPy code,

and document this choice.

- [x] Ensure two-pass coverage renorm parity
 Inspect the two-pass coverage renorm implementation:

where coverage maps are built,

how gains are derived from coverage,

how they are applied.

 Ensure:

GPU path uses the same coverage-based math and parameters as CPU,

or falls back to CPU for this part if necessary.

Avoid having a “simplified” GPU renorm that changes behaviour.

- [x] Preserve robustness and fallback
 Wherever GPU normalization is introduced:

wrap GPU operations in a try/except catching GPU-specific exceptions.

 On failure:

log a single clear message indicating GPU normalization failure and fallback to CPU,

disable GPU for subsequent Phase 5 operations in this run (if a global flag exists),

recompute the affected step on CPU to ensure a valid result.

- [x] Add tests for photometric parity
 Create a small synthetic test dataset with:

multiple overlapping tiles,

non-trivial RGB values (e.g. different brightness/color per tile).

 Write a test that:

runs Phase 5 CPU-only to produce mosaic_cpu,

runs Phase 5 with GPU enabled to produce mosaic_gpu (mock GPU presence if needed),

asserts:

mosaic_cpu.shape == mosaic_gpu.shape,

np.allclose(mosaic_cpu, mosaic_gpu, rtol=1e-3, atol=1e-3) (or similarly tight thresholds),

and/or bounds the maximum per-channel difference.

 Add a test (or extend an existing one) that ensures CPU-only results are unchanged versus previous baseline (if previous reference outputs exist).

8. Manual validation and telemetry sanity check
 Run a small real-world dataset with:

GPU disabled → obtain a CPU reference mosaic.

GPU enabled → obtain a GPU mosaic.

 Compare visually:

check for seams, color shifts, background jumps.

 Inspect telemetry/resource monitor:

confirm that Phase 5 still loads the GPU as expected,

and that no abnormal CPU-only fallback is occurring unexpectedly.

When all tasks are complete:

The GPU Phase 5 path must produce mosaics that are photometrically equivalent to the CPU path (no extra seams, no color drifts),

Enabling GPU should be a pure performance optimization, not a change in scientific behaviour.
