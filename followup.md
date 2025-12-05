## ✅ `followup.md`

```markdown
# Follow-up – Grid mode assembly & RGB consistency

## Task checklist

### 1. Enrich `assemble_tiles(...)` logging

- [ ] Add detailed logs for each early-return condition:
  - [ ] Astropy/FITS unavailable (with flags).
  - [ ] `tiles_list` empty (with counts and sample paths).
  - [ ] `tile_infos` empty (with summary: io_fail, channel_mismatch, empty_mask, kept=0).
  - [ ] `weight_sum` all zeros (with counts and hint).
- [ ] Track per-tile failure reasons with counters:
  - [ ] I/O failures.
  - [ ] Channel mismatches.
  - [ ] Empty valid masks.
- [ ] Emit a summary line after building `tile_infos`.
- [ ] Ensure all logs use `_emit(...)` and follow the existing `[GRID]` / logging style.

### 2. Relax logic & salvage assembly

- [ ] Modify the `weight_sum` check to:
  - [ ] Log a warning when `weight_sum` is all zeros.
  - [ ] Attempt a salvage assembly using a simple, relaxed placement of tiles into the mosaic.
  - [ ] Only `return None` if both nominal and salvage paths fail to write any data.
- [ ] Add a log indicating when the salvage path is used:
  - [ ] `"Assembly: salvage assembly succeeded..."` or similar.
- [ ] Keep overlap/photometry logic intact for the normal successful path.

### 3. RGB equalization coherence (Grid vs classic)

- [ ] Audit current RGB equalization flags and usage in:
  - [ ] `zemosaic_worker.py` (classic pipeline, phase 3+5).
  - [ ] `grid_mode.py` (`grid_rgb_equalize`, `grid_post_equalize_rgb`).
- [ ] Introduce a single, explicit boolean for “final RGB equalization enabled?”:
  - [ ] Derive it from config/GUI in `run_hierarchical_mosaic(...)`.
  - [ ] Use it for the classic pipeline’s final assembly (`RGBEqualize`).
  - [ ] Pass it explicitly to `grid_mode.run_grid_mode(..., grid_rgb_equalize=...)`.
- [ ] Adjust config overlay in `grid_mode.run_grid_mode(...)` so that:
  - [ ] The final `grid_rgb_equalize` state is well-defined and logged.
- [ ] Add minimal sanity logs:
  - [ ] In `run_hierarchical_mosaic(...)`: log `[GRID] Invoking grid_mode with RGBEqualize=...`.
  - [ ] In `assemble_tiles(...)`: log the effective RGB equalization state when calling `grid_post_equalize_rgb(...)`.

### 4. Tests / validation

- [ ] Happy-path grid run with RGB equalization ON.
- [ ] Dataset with a few broken tiles:
  - [ ] Confirm they are skipped and mosaic is still produced.
- [ ] Edge-case where masks/overlaps zero out `weight_sum`:
  - [ ] Confirm salvage path is attempted and logged.
- [ ] Compare ON/OFF RGB equalization across:
  - [ ] Classic pipeline.
  - [ ] Grid mode.
  - [ ] Ensure behaviours are consistent with the chosen toggle.

## Notes / Decisions

- [ ] Document any behavioural change that could affect existing users (e.g. salvaged mosaics where previous versions aborted).
- [ ] If any unexpected side-effects appear, describe them here and propose follow-up tasks.
````

Si tu veux, je peux ensuite t’aider à préparer un petit jeu d’images “test grid mode” pour valider rapidement les cas happy-path / salvage avant que tu pushes sur `V4WIP` 😄
