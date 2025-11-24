# agent.md

## Mission: Restore RGB photometric normalization in the GPU Phase 5 path

You are working on **ZeMosaic**, an astrophotography mosaic/stacking pipeline written in Python.

The project recently introduced/expanded a **GPU path in Phase 5** (“Reproject & Coadd”, including the optional two-pass coverage renormalization).  

The user observes the following:

- When Phase 5 is run on **CPU**, the final mosaic has:
  - smooth transitions between tiles,
  - consistent RGB balance and background levels.
- When Phase 5 uses the **GPU** path:
  - seams / boundaries between tiles become visible,
  - local color/brightness mismatches appear,
  - suggesting that the **RGB photometric normalization / background matching is not applied (or not applied correctly) in the GPU branch**.

Your mission is to:

1. **Audit Phase 5 CPU vs GPU implementations** and identify where **RGB photometric normalization and background matching** are applied in the CPU pipeline.
2. **Ensure the GPU path applies *exactly the same* normalization logic**, or a mathematically equivalent implementation, so that:
   - enabling GPU does **not** change scientific/photometric behaviour,
   - only performance changes.
3. **Keep the global workflow, SDS logic and GUI semantics unchanged**:
   - SDS vs non-SDS pipelines must behave identically as before,
   - only the internal CPU/GPU computation and normalization implementation is allowed to change.

The goal:  
> **With GPU enabled, Phase 5 should produce a mosaic that is visually and numerically very close to the CPU reference (no additional seams / color shifts), while still benefiting from GPU acceleration.**

---

## Scope

Focus specifically on:

- **Phase 5 non-SDS final mosaic assembly**:
  - The main `reproject & coadd` step that combines tiles into the final RGB mosaic.
- **Phase 5 two-pass coverage renorm (if enabled)**:
  - Any additional per-pixel / per-region gain adjustment based on coverage maps that affects photometric balance.
- **The photometric normalization steps**, including (names may differ in code):
  - inter-tile RGB gain computation,
  - local / overlap-based renormalization,
  - background matching / offset corrections,
  - any scalar or per-channel gain applied to tiles before or after reprojection.

You are **not** allowed to:

- Change how tiles/mega-tiles/lots are defined or grouped.
- Change which frames belong to which tile or mega-tile.
- Change the scientific formulae / design of the normalization (unless you are simply rewriting them in a GPU-compatible way).
- Change GUI/CLI options semantics.

---

## Non-goals / constraints

- ❌ Do **not** alter:
  - SDS (ZeSupaDupStack) logical pipeline,
  - the way master tiles / mega-tiles are created,
  - overlap / coverage thresholds.
- ❌ Do **not** introduce new GUI options or break existing ones.
- ❌ Do **not** degrade CPU behaviour; CPU-only runs must produce **unchanged results**.

- ✅ It is acceptable to:
  - Factor common normalization logic into shared helpers,
  - Introduce GPU-friendly vectorized code (CuPy-based),
  - Use a hybrid approach (GPU reprojection + CPU normalization) if that is simpler/safer.

- ✅ Numeric equality does **not** have to be bit-perfect:
  - Small differences due to float precision / interpolation or order of operations are acceptable,
  - But **visible seams / major RGB shifts are not**.

---

## Key files and entry points

(Names may differ slightly, adjust to actual repo.)

- **Phase orchestration / worker**  
  `zemosaic_worker.py`
  - Phase 5 orchestration:
    - Non-SDS `reproject & coadd` call,
    - Optional two-pass coverage renorm.
  - Look for log lines like:
    - `[INFO] [Intertile] Overlap pairs at min_overlap=...`
    - `assemble_info_finished_reproject_coadd`
    - `Phase 5 finished: Reproject & Coadd completed. Mosaic shape (HWC): ...`

- **Photometric normalization & coadd helpers**  
  `zemosaic_utils.py` (and possibly related modules)
  - Functions for:
    - building/interpreting coverage maps,
    - computing/ applying RGB gains,
    - matching backgrounds,
    - performing coadd (CPU and GPU).
  - Look for functions / blocks relating to:
    - `photometric`, `gain`, `rgb`, `background`, `match`, `renorm`, `coverage`.

- **Align/stack CPU & GPU implementation**  
  `zemosaic_align_stack.py`  
  `zemosaic_align_stack_gpu.py`
  - Show how CPU and GPU paths share or diverge in logic.
  - Provide patterns for:
    - using CuPy vs NumPy,
    - ensuring identical math on CPU & GPU,
    - fallback logic.

- **Parallel/plan & GPU use**  
  `parallel_utils.py`
  - `detect_parallel_capabilities`, `auto_tune_parallel_plan`, etc.
  - Phase 5 GPU decisions should be consistent with this (but this mission is about **normalization parity**, not re-tuning parallelism).

- **Config & GUI**  
  `zemosaic_config.py`
  - Config flags controlling:
    - GPU usage,
    - two-pass coverage renorm,
    - any photometric/normalization options.
  - **Do not change semantics**, only ensure GPU uses the same flags.

  `zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`
  - Existing GPU toggle(s) and SDS options; you don’t need to modify the GUI for this mission, only be aware of how the GPU is enabled.

---

## Desired behaviour (high-level design)

### 1. Map the CPU photometric normalization pipeline (Phase 5)

Identify, for **non-SDS Phase 5 (CPU path)**:

1. Where per-tile / per-channel gains are computed:
   - Example: overlap-based RGB gain estimation, background level measurements, etc.
2. Where those gains are applied:
   - before reprojection (on tile data),
   - during coadd,
   - or after coadd using coverage/gain maps.
3. Any coverage-based renormalization (two-pass coverage renorm or similar) affecting final RGB/bkg.

Create a **clear sequence** (e.g. comments or docstring) describing:

> “CPU Phase 5 does:  
>  1) compute per-tile RGB gains,  
>  2) apply them to each tile,  
>  3) run reproject & coadd,  
>  4) optionally run second-pass coverage renorm.”

This will be the **reference behaviour**.

---

### 2. Map the GPU Phase 5 path and identify gaps

For the **GPU Phase 5 path** (non-SDS), identify:

- How the GPU branch is selected (e.g. `use_gpu=True` argument, helper, or condition).
- Which steps are done on GPU:
  - reprojection (warp),
  - coadd,
  - any weighting,
  - and whether normalization is applied at all.
- Where, if anywhere, the current GPU path **skips**:
  - RGB gain computation,
  - RGB gain application,
  - background matching.

Do the same check for the GPU-like path of the **two-pass coverage renorm** if it exists.

The typical problem you need to confirm/fix:

> GPU path goes straight from raw tiles → reprojection → coadd  
> without applying the CPU photometric normalization logic.

---

### 3. Refactor or introduce shared normalization helpers

Introduce **shared, well-defined helpers** that perform the photometric normalization, e.g.:

```python
def compute_tile_rgb_gains(...):
    """Compute RGB gains / offsets for each tile, using the same math as CPU Phase 5."""
    ...

def apply_tile_rgb_gains(image, gains, backend="numpy"):
    """Apply per-channel gains to an image using NumPy or CuPy, depending on backend."""
    ...
This is only an example; adjust names and signatures to actual code.

Key points:

The same core math (formulas, weighting, clipping, etc.) must be used for both CPU and GPU paths.

The helper may accept a backend / array type or detect whether arrays are NumPy (CPU) or CuPy (GPU) and act accordingly.

You may reuse existing CPU-only code by:

generalizing it to work with array-like objects,

or branching internally on np vs cp.

4. Ensure GPU path invokes the same normalization logic
Modify the GPU Phase 5 flow so that:

It calls the same normalization helpers as the CPU path, with:

the same inputs (coverage maps, overlap metrics, etc.),

the same configuration (clip limits, sigma, thresholds).

The normalization is applied in the same stage as the CPU pipeline:

either pre-warp on each tile,

or post-warp using coverage-based maps,

or both, depending on existing design.

If it is hard to implement everything on GPU, a hybrid approach is acceptable:

e.g.:

use GPU for reprojection + coadd,

then transfer to CPU for a final normalization pass with the exact same NumPy-based code as CPU,

as long as the photometric result matches the CPU reference (performance may be slightly reduced, but correctness is mandatory).

5. Keep two-pass coverage renorm consistent
If the project implements two-pass coverage renorm:

Ensure CPU and GPU flows use the same logic and parameters for:

coverage smoothing (sigma_px),

gain clipping,

gain map application.

If GPU currently bypasses or oversimplifies this step, bring it to feature parity with CPU.

6. Robustness and fallback
Re-use the existing GPU→CPU fallback mechanism (if implemented) so that:

If GPU normalization runs out of memory or fails:

log a clear warning,

disable GPU for the rest of the run,

recompute Phase 5 on CPU with full normalization.

Do not leave the user with a partially normalized GPU mosaic.

Testing expectations
Add or update tests to validate:

CPU vs GPU photometric parity:

Build a small synthetic dataset with multiple tiles and overlaps.

Run Phase 5:

once with CPU-only,

once with GPU enabled.

Compare the resulting mosaics:

same shape, same WCS,

similar pixel values.

It’s acceptable to check:

np.allclose(cpu, gpu, rtol=1e-3, atol=1e-3) or similar,

and/or assert that max per-channel difference is below a small threshold.

No visible RGB bias:

Optionally compute statistics per tile region in the final mosaic:

mean / median in overlaps should be close between CPU and GPU runs.

No regression for CPU-only:

Existing tests for Phase 5 and SDS should pass unchanged.

Document (in comments or a short dev note) the assumption:

“GPU Phase 5 now uses the same photometric normalization as CPU; enabling GPU should only change performance, not photometric behaviour.”