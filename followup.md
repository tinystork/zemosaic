## `followup.md`

```markdown
# Follow-up: Classic mode still shows green tint – focus on master-tile lecropper pipeline

## Current status

- We have:
  - Disabled final mosaic RGB equalisation.
  - Restored Phase 5 call arguments to match the reference worker for classic mode (`enable_lecropper_pipeline=False`, `enable_master_tile_crop=False`).
- Despite these changes, the classic / non-grid mosaic still shows a **strong green tint**.
- Logs show:
  - `poststack_equalize_rgb` is applied on master tiles.
  - Quality crop logs `MT_CROP: quality-based rect=...` appear in Phase 3.
- The visual result is still a **green, noisy background**, which suggests that:
  - The problem originates at the **master-tile level** (background / gradients / alt-az artefacts),
  - Or colour manipulations are still being compounded across stages.

The user’s priority is:

> “Just get a production-ready classic mode again, like the old worker, even if we don’t fully understand every detail.”

---

## Key observation

The known-good worker `zemosaic_worker_non grid_ok.py`:

- Applies **center-out normalisation + lecropper + quality crop + alt-az cleanup** on **each master tile**.
- Does **not** apply aggressive RGB equalisation on the final mosaic.
- Produces clean, neutral backgrounds.

The unified worker `zemosaic_worker.py` is very similar, but the classic path may:

- Use slightly different flags (e.g. `quality_crop_enabled_tile`, `quality_gate_enabled_tile`),
- Potentially bypass or weaken the lecropper pipeline for some tiles,
- Or apply per-tile overrides that disable quality crop / alt-az cleanup unintentionally.

Since changing Phase 5 and disabling final RGB-EQ had **no visible effect**, the remaining suspect is:

> **The master-tile lecropper / alt-az pipeline not being faithfully replicated for classic mode.**

---

## What remains to be done

1. **Align master-tile lecropper pipeline with the reference worker**, focusing on:

   - `pipeline_cfg` contents (keys and types):
     - `quality_crop_enabled`
     - `quality_crop_band_px`
     - `quality_crop_k_sigma`
     - `quality_crop_margin_px`
     - `quality_crop_min_run`
     - `altaz_cleanup_enabled`
     - `altaz_margin_percent`
     - `altaz_decay`
     - `altaz_nanize`
   - Ensuring master tiles always call `_apply_lecropper_pipeline(...)` in classic mode, without accidental short-circuit.

2. Ensure that any per-tile flags (`quality_crop_enabled_tile`, `quality_gate_enabled_tile`, etc.):

   - Default to the **global config** in classic mode,
   - Do not silently disable the lecropper pipeline unless explicitly required.

3. Add a clear log line at the end of `create_master_tile(...)` summarising:

   - `lecropper_applied` (True/False),
   - `quality_crop_enabled_tile`,
   - `altaz_cleanup_enabled`.

4. After implementing this, run a classic test and confirm:

   - Logs show `lecropper_applied=True` for all master tiles.
   - The mosaic background / colour is back to a normal, neutral look.

---

## Guardrails

- Do not touch Grid Mode or SDS-specific paths.
- Do not re-enable final RGB equalisation.
- Do not introduce new colour transformations beyond what exists in the reference worker.

The main objective is to **mirror the reference worker’s master-tile processing** in `zemosaic_worker.py` for classic mode, so that colour balance and background behaviour match the previously validated implementation.
