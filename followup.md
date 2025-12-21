# followup.md — Test plan & validation

## Quick manual tests (Windows / Qt)
### Test A — Happy path (existing master tiles)
1) Prepare a folder containing a known-good set of master tiles:
   - Example: copy `out/zemosaic_temp_master_tiles/master_tile_*.fits` from a previous successful run.
2) Launch ZeMosaic Qt.
3) Set **Input folder** to that folder.
4) Set **Output folder** to a new empty folder.
5) Enable: **“I’m using master tiles …”**
6) Keep default assembly: `Reproject co-add`, keep intertile options ON if you want.
7) Run.

Expected:
- Log shows message that phases 0–3 are skipped due to existing master tiles.
- No filter/clustering window is invoked.
- Output final mosaic is produced and looks consistent with the source tiles.
- Intermediate phase45 outputs (if any) go to the normal temp output area, not inside the input folder.

### Test B — Fallback path (invalid WCS)
1) Use an input folder containing FITS without a valid celestial WCS (or remove WCS keywords).
2) Enable “I’m using master tiles …”.
3) Run.

Expected:
- Warning appears: insufficient valid master tiles with WCS.
- ZeMosaic continues in normal mode (phases 0–3 run).
- No crash.

### Test C — Regression (toggle off)
1) Run a standard dataset with raws (your usual workflow).
2) Toggle OFF.
3) Compare logs and outputs with the previous behavior.

Expected:
- Identical behavior (no changes in clustering/master tile steps, same outputs).

---

## Developer checks
- Confirm the new config key is present in `DEFAULT_CONFIG` and is saved/loaded.
- Confirm `zemosaic_worker.py` receives `use_existing_master_tiles_config` and branches only when True.
- Ensure WCS validation uses `validate_wcs_header()` and does not silently accept garbage headers.
- Ensure any UI disabling/enabling doesn’t break layout or crash if groups are missing.

---

## What to include in your PR message
Title: `Qt: Add "use existing master tiles" shortcut mode`

Bullet points:
- Adds GUI toggle to start pipeline from already-resolved master tiles
- Validates WCS and falls back safely when invalid
- No behavior change when toggle is off
