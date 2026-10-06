# ZM-ZEGRID-R8 — Legacy Grid removal + helper relocation

Mission: `ZM-ZEGRID-R8`
Phase: implementation
Date: 2026-10-06
Author: Coco (implementation worker), for Junior (acceptance authority) and Nono (independent reviewer)

## Decision

The historical Grid engine (`src/zemosaic/grid_mode.py`, ~4650 lines) never worked in
production and is **REMOVED from the product** (Tristan's decision). It is **ARCHIVED**
on the pushed branch:

```
origin/archive/zegrid-legacy-grid-5.0.0   (branch tip 83b94f4)
```

This branch is historical reference only and is **not** to be modified.

## What changed

1. **Deleted** `src/zemosaic/grid_mode.py` entirely.
2. **Relocated** (verbatim, behaviour-preserving) the helpers still required by the
   new ZeGrid engine:
   - `_load_image_with_optional_alpha` (product decoder = `load_and_validate_fits` +
     `debayer_image`) → `zemosaic.zemosaic_utils.load_image_with_optional_alpha`
     (public name; exact behaviour preserved).
   - `FrameInfo`, `load_stack_plan`, `_load_frame_wcs` (now `load_frame_wcs`),
     `detect_grid_mode`, and their private helpers (`_parse_float`, `_parse_int`,
     `_normalize_mount`, `_resolve_path`, `_dialect_from_sample`, `_open_fits_safely`,
     `_extract_pixel_scale_deg`) → new module `zemosaic.zemosaic_stack_plan`.
3. **Worker** (`zemosaic.zemosaic_worker`): removed the `grid_engine` setting,
   `_resolve_grid_engine`, and the legacy `run_grid_mode` branch. A detected
   `stack_plan.csv` now routes **directly** to `zemosaic_zegrid_mode.run_zegrid_mode`
   (normalization still resolved via `_resolve_zegrid_normalization`). There is no
   silent fallback (nothing to fall back to).
4. **ZeGrid engine** (`zemosaic.zemosaic_zegrid_mode`) now imports the relocated
   helpers from their new homes (no dependency on the removed `grid_mode`).

## The new ZeGrid engine (unchanged behaviour)

`stack_plan.csv` → `zemosaic_zegrid_mode.run_zegrid_mode` (R7, verified). The R7
end-to-end output for a pinned layout is unchanged by this removal.
