# Agent — SDS Global Mosaic = Final Image (Phase 5 = Polish + Save)

## 1. Mission

Refactor the SDS (Super-Deep Stacking / Mosaic-First) pipeline so that:

- **When SDS is ON**:  
  - The pipeline is strictly:
    > `lights → mega-tiles → global super-stack = final mosaic`  
    - Phase 5 no longer re-runs a full `reproject_and_coadd` over all mega-tiles.
    - Phase 5 becomes a **polish + save** stage operating on the already-built global SDS mosaic.
  - Low-coverage, noisy borders are **masked out** early (coverage-based threshold).

- **When SDS is OFF**:  
  - Preserve the **historical behavior**:
    > master tiles are built, Phase 3 & 5 work as before, with the standard `assemble_final_mosaic_reproject_coadd`.
  - No regression, no change in visual results (aside from normal noise).

The main goal is to:
- **keep the dense, well-covered core**,  
- **drop very low-coverage “skirts” (mostly noise and seams)**,  
- and **avoid redundant reproject+coadd passes** when SDS already produced a proper global mosaic.

---

## 2. Current situation (summary)

Today, in SDS mode, the logs show something like:

- `P4 - Mosaic-First global coadd finished (kappa_sigma) ... grid 3621×3333px, 6 image(s).`  
  → SDS builds a **global mosaic from mega-tiles** using a Mosaic-First `reproject_and_coadd`.

- Then, **Phase 5** still runs:
  - `Phase 5: Final assembly (Reproject & Coadd)...`
  - `run_info_phase5_finished_reproject_coadd ... shape: (...)`
  → This is a **full reproject+coadd** over the tiles/mega-tiles again, as in the non-SDS pathway.

- At save time, we see:
  - `SAVE_DEBUG: Données image_data reçues - Shape: (3333, 2661, 3) ...`
  → auto-crop + coverage logic is applied there.

So the SDS idea is already present (mega-tiles + Mosaic-First coadd), but **Phase 5 remains too “heavy”** in SDS mode and still behaves like the generic pipeline.

---

## 3. Target behavior

### 3.1 SDS ON

Logical pipeline must be:

```text
Lights (unit frames)
  → SDS grouping / batch processing
    → Mega-tiles (each = stack of unit frames on global WCS)
      → Mosaic-First global coadd (super-stack of mega-tiles)
        → Coverage-based masking / cropping
          → Phase 5: polish (renorm/IBN, auto-crop, alpha, save) on the global mosaic
````

Concretely:

* After **Mosaic-First global coadd** in SDS:

  * Compute / normalize a **coverage map** `coverage_HW`:

    * Either in [0, 1] (fraction of max contributions) or in “number of frames” + normalized variant.
  * Apply a **minimum coverage threshold** `min_coverage_keep`:

    * Pixels below threshold:

      * `final_mosaic_data_HWC[...] = NaN`
      * `final_coverage_HW[...] = 0.0`
  * Expose these as:

    * `final_mosaic_data_HWC` (float32, shape `(H, W, 3)` or `(H, W)` depending on channels)
    * `final_coverage_HW` (float32, shape `(H, W)`)

* **Phase 5 in SDS mode**:

  * **Does NOT** call `assemble_final_mosaic_reproject_coadd` on the list of tiles/mega-tiles.
  * Instead, Phase 5 takes:

    * `final_mosaic_data_HWC`
    * `final_coverage_HW`
  * And applies:

    * global renorm / IBN (if enabled),
    * coverage-based auto-crop (as today),
    * alpha mask building (if applicable),
    * FITS + coverage + preview export.

### 3.2 SDS OFF

* Keep the **legacy behavior** unchanged:

  * Master tiles are built.
  * Phase 3 / Phase 5 run the classic:

    * `assemble_final_mosaic_reproject_coadd(...)`
    * and associated coverage/alpha/crop logic.
* This must preserve:

  * Historical stacking/coverage behavior.
  * All existing options (including batch_size=0 / >1 semantics, GPU/CPU fallback, etc.).

---

## 4. Scope of changes

Likely impacted files (names based on current project structure):

* `zemosaic_worker.py` (or equivalent):

  * Orchestration of phases 1–7.
  * SDS branching logic (SDS ON/OFF).
  * Calls to:

    * `assemble_global_mosaic_sds` / `assemble_global_mosaic_first`,
    * `assemble_final_mosaic_reproject_coadd`,
    * save functions (FITS, coverage, preview).

* `zemosaic_align_stack.py` (or similar):

  * Implement or adjust:

    * SDS global mosaic coadd,
    * coverage computation/normalization,
    * handling of `final_mosaic_data_HWC` + `final_coverage_HW`.

* `zemosaic_config.py` / `solver_settings.py`:

  * New SDS-specific config parameter(s):

    * `min_coverage_keep` (float in [0,1], default e.g. 0.4 or 0.5).
    * Optionally: distinct `min_coverage_cropping` if needed.

* Possibly: `zemosaic_filter_gui_qt.py` (optional, not mandatory in this mission):

  * If you expose `min_coverage_keep` in GUI (advanced SDS options).
  * This is a **nice-to-have**, not required for this mission.

---

## 5. Constraints & Non-goals

* **Do not touch**:

  * Non-SDS stacking logic (historical pathway).
  * `batch_size=0` vs `batch_size>1` behavior.
  * GPU/CPU fallback logic and chunking.
* Avoid:

  * Changing existing default visual behavior when SDS is **OFF**.
* Performance:

  * The new SDS path must remain streaming / memory-aware (no huge in-RAM copies unnecessarily).

Non-goals (for this mission):

* No change to:

  * GUI layout / UX (beyond optional config binding).
  * Solver / WCS solving behavior.
  * Non-SDS photometric normalization algorithms (IBN, etc.).

---

## 6. Success criteria

* When SDS is **ON**:

  * Logs clearly show:

    * Construction of mega-tiles from unit frames.
    * A global Mosaic-First coadd on mega-tiles.
    * Phase 5 operating in **“polish only”** mode, not launching a full `reproject_and_coadd` again.
  * The final FITS + coverage:

    * Have low-coverage skirts removed (NaN / 0 coverage).
    * Show reduced noise and fewer visible seams on borders.

* When SDS is **OFF**:

  * Logs remain identical (or very close) to the historical behavior.
  * Final outputs remain visually consistent with previous versions.

* Code:

  * Clear branching on SDS ON/OFF at the worker/orchestration level.
  * New coverage-threshold parameter is documented and has a safe default.

````

