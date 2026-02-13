# Iteration note (2026-02-13, DBE ON/OFF root-cause + observability logs)

## What changed
- Analyzed worker logs:
  - `zemosaic_worker_dbe_off.log`
  - `zemosaic_worker_dbe_on.log`
  - Found both runs reported DBE execution:
    - `[DBE] applied=True ...` in both OFF and ON logs.
- Updated `zemosaic_config.py`:
  - Added missing default key:
    - `final_mosaic_dbe_enabled: True`
- Updated `zemosaic_worker.py` (both Phase 6 save paths):
  - Added explicit DBE gate log before apply:
    - `[DBE] phase6 gate enabled=<bool> explicit_cfg=<bool> raw_value=<repr>`
  - Added explicit disabled log:
    - `[DBE] skipped: disabled by config (final_mosaic_dbe_enabled=False)`
  - Added preflight validity-mask diagnostics:
    - mask source (`alpha_final` / `coverage` / `finite_fallback`)
    - valid pixel counts/fraction
    - mosaic shape
  - Extended applied log with richer metrics:
    - `obj_k`, `valid_px`, `total_px`, `model`, `time_ms`, and `mean_abs_bg_sub`
- Extended `_apply_final_mosaic_dbe_per_channel(...)` return metadata:
  - `valid_pixels`, `total_pixels`, `model_shape`, `mean_abs_bg_sub`

## Why
- Root cause for "no ON/OFF difference" was config persistence:
  - GUI wrote `final_mosaic_dbe_enabled`, but `save_config()` filters by `DEFAULT_CONFIG`.
  - Because the key was missing from `DEFAULT_CONFIG`, it was dropped on save.
  - Worker then fell back to `getattr(..., True)`, effectively forcing DBE ON.
- Added high-signal DBE logs to make future diagnosis unambiguous from a single run log.

## How tested
- Log inspection:
  - `rg -n "\\[DBE\\]|P6_PRE_EXPORT|phase6" zemosaic_worker_dbe_off.log zemosaic_worker_dbe_on.log`
  - Confirmed both OFF/ON logs previously showed `[DBE] applied=True`.
- Config persistence probe:
  - Verified `final_mosaic_dbe_enabled` now survives load/save roundtrip.
- Syntax checks:
  - `python -m py_compile zemosaic_config.py zemosaic_worker.py`

## Known limitations / risks
- End-to-end rerun after this fix is still required to capture new logs in fresh ON/OFF runs.
- DBE helper still uses coarse Gaussian model; this iteration focused on gating/persistence and logging, not algorithm redesign.

## Next item
- Re-run one dataset with DBE OFF then ON and confirm logs diverge as expected:
  - OFF: `[DBE] skipped: disabled by config ...`
  - ON: `[DBE] applied=True ...`

# Iteration note (2026-02-13)

## What changed
- Updated `followup.md`:
  - Marked `Meta / Process (MANDATORY)` item as complete:
    - ``[x] `memory.md` updated with: changes, why, tests, limitations, next step``
- Updated `memory.md` (this file) with the required fields for this iteration.

## Why
- The next unchecked follow-up item was the mandatory process requirement to update `memory.md`.
- Completing it now keeps the checklist state accurate before moving on to code-level DBE tasks.

## How tested
- Command: `Get-Content -Raw followup.md`
  - Verified the first checklist item is now `[x]`.
- Command: `Get-Content -Raw memory.md`
  - Verified this file contains: what changed, why, tests, limitations/risks, and next step.

## Known limitations / risks
- No runtime or GUI behavior changed in this iteration.
- DBE feature work remains unimplemented; checklist code/test items are still pending.

## Next item
- `GUI: checkbox stored in config key final_mosaic_dbe_enabled`

# Iteration note (2026-02-13, GUI DBE config key)

## What changed
- Updated `zemosaic_gui.py`:
  - Added `self.final_mosaic_dbe_var` (`tk.BooleanVar`) with default from config key `final_mosaic_dbe_enabled` and fallback `True`.
  - Added checkbox in `final_assembly_options_frame`:
    - label: `Dynamic Background Extraction (DBE) on final mosaic`
    - variable: `self.final_mosaic_dbe_var`
  - Persisted setting during run launch:
    - `self.config["final_mosaic_dbe_enabled"] = bool(self.final_mosaic_dbe_var.get())`
- Updated `followup.md`:
  - Marked `GUI: checkbox stored in config key final_mosaic_dbe_enabled` as `[x]`.

## Why
- Implements the next unchecked follow-up item only: expose final-mosaic DBE control in GUI and store it in config.
- Defaulting to `True` preserves the intended default-ON behavior for this feature flag.

## How tested
- `python -m py_compile zemosaic_gui.py`
  - Expected/observed: no syntax errors.
- `rg -n "final_mosaic_dbe_var|final_mosaic_dbe_enabled|Dynamic Background Extraction \\(DBE\\) on final mosaic" zemosaic_gui.py`
  - Verified:
    - variable initialization exists
    - checkbox exists in final assembly UI
    - config persistence assignment exists
- `Get-Content -Raw followup.md`
  - Verified only this target item was newly checked.

## Known limitations / risks
- This iteration does not implement worker-side DBE execution yet; the checkbox currently only stores configuration.
- No end-to-end mosaic run was executed in this iteration, so functional DBE effect is not yet validated.

## Next item
- `Worker: reads zconfig.final_mosaic_dbe_enabled with default True`

# Iteration note (2026-02-13, DBE header keywords only when applied)

## What changed
- Updated `zemosaic_worker.py`:
  - Extended `_apply_final_mosaic_dbe_per_channel(...)` metadata with `obj_k` (default `3.0`) so DBE header fields have explicit source values.
  - In both Phase 6 DBE hook blocks, added FITS header assignments only inside `if dbe_info.get("applied")`:
    - `ZMDBE = True`
    - `ZMDBE_DS = int(ds_factor)`
    - `ZMDBE_K = float(obj_k)`
    - `ZMDBE_SIG = float(blur_sigma)`
  - No `ZMDBE*` write occurs in the non-applied path.
- Updated `followup.md`:
  - Marked `Header keywords written only if applied (ZMDBE, ZMDBE_DS, ZMDBE_K, ZMDBE_SIG)` as `[x]`.

## Why
- This was the next unchecked checklist item.
- Header metadata now records DBE usage parameters while respecting the requirement to emit keys only when DBE actually applied.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass (no syntax errors).
- `rg -n 'if isinstance\\(dbe_info, dict\\) and dbe_info.get\\(\"applied\"\\)|ZMDBE|ZMDBE_DS|ZMDBE_K|ZMDBE_SIG|obj_k' zemosaic_worker.py`
  - Result: confirmed `ZMDBE*` assignments are under `dbe_info.get("applied")` in both Phase 6 save paths.
- `Get-Content zemosaic_worker.py` line slices:
  - `22184..22214` and `26486..26516`
  - Result: visually confirmed guarded header writes.
- Runtime probe (inline Python):
  - Called `_apply_final_mosaic_dbe_per_channel(...)` and verified metadata includes `obj_k`.

## Known limitations / risks
- End-to-end FITS export run was not executed in this iteration, so on-disk header presence is validated by code-path inspection, not full pipeline output.
- Manual SDS/classic/grid visual regression checks remain pending.

## Next item
- `Run with DBE ON/OFF, confirm only background changes.`

# Iteration note (2026-02-13, DBE valid mask uses alpha/coverage priority)

## What changed
- Updated `zemosaic_worker.py` in both Phase 6 DBE hook blocks:
  - Replaced `valid_mask_hw=None` with a computed `dbe_valid_mask_hw` using priority:
    1. `alpha_final > 0` (if shape matches final mosaic)
    2. `final_mosaic_coverage_HW > 0` (if shape matches final mosaic)
    3. fallback: finite-pixel mask from `final_mosaic_data_HWC`
  - Passed `valid_mask_hw=dbe_valid_mask_hw` into `_apply_final_mosaic_dbe_per_channel(...)`.
- Updated `followup.md`:
  - Marked `Uses alpha_final / coverage to avoid touching invalid mosaic areas` as `[x]`.

## Why
- This was the next unchecked checklist item.
- Ensures DBE does not alter invalid zones and follows the intended alpha/coverage validity priority.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass (no syntax errors).
- `rg -n "dbe_valid_mask_hw|alpha_final|final_mosaic_coverage_HW|valid_mask_hw=dbe_valid_mask_hw" zemosaic_worker.py`
  - Result: confirmed both Phase 6 save paths now compute and pass mask using alpha/coverage/fallback.
- Runtime probe (inline Python) calling `_apply_final_mosaic_dbe_per_channel(...)` with explicit `valid_mask_hw`:
  - Observed:
    - `applied=True`
    - `applied_channels=3`
    - `unchanged_invalid=True`
    - `changed_valid=True`

## Known limitations / risks
- DBE FITS header keywords (`ZMDBE`, `ZMDBE_DS`, `ZMDBE_K`, `ZMDBE_SIG`) are not yet written.
- Checklist still contains manual end-to-end runs not executed yet.

## Next item
- `Header keywords written only if applied (ZMDBE, ZMDBE_DS, ZMDBE_K, ZMDBE_SIG)`

# Iteration note (2026-02-13, consistency audit + per-channel DBE application)

## What changed
- Consistency/regression audit performed before new edits:
  - Re-validated current DBE-related wiring in Tk GUI, Qt GUI, worker Phase 6 hooks, and grid mode hook.
  - Recompiled key modules to check syntax regressions.
- Updated `zemosaic_worker.py`:
  - Added `_apply_final_mosaic_dbe_per_channel(...)` helper.
  - DBE subtraction is applied per-channel (`for chan_idx in range(3)`) with per-channel background models only (2D channel buffers), avoiding a full HWC background buffer allocation.
  - Hooked helper into both existing Phase 6 DBE hook points (both save-path blocks).
  - Added DBE applied/skipped logs for the new per-channel mode.
- Updated `followup.md`:
  - Marked `Per-channel application (no full HWC background buffer)` as `[x]`.

## Why
- The next unchecked checklist item was per-channel DBE application without full HWC background buffering.
- This implementation keeps memory behavior aligned with the requirement while preserving existing fail-open behavior.

## How tested
- Consistency/regression checks:
  - `python -m py_compile zemosaic_gui.py zemosaic_gui_qt.py zemosaic_worker.py grid_mode.py zemosaic_utils.py`
  - `rg -n "final_mosaic_dbe_enabled|\\[DBE\\]|P6_PRE_EXPORT|DBE hook placement is unambiguous" followup.md zemosaic_gui.py zemosaic_gui_qt.py zemosaic_worker.py grid_mode.py`
- New implementation checks:
  - `python -m py_compile zemosaic_worker.py grid_mode.py zemosaic_gui.py zemosaic_gui_qt.py`
  - `rg -n "def _apply_final_mosaic_dbe_per_channel|mode=per_channel|bg_full = cv2.resize|for chan_idx in range\\(3\\)" zemosaic_worker.py`
  - Runtime probe:
    - inline Python script calling `zemosaic_worker._apply_final_mosaic_dbe_per_channel(...)` on synthetic RGB data.
    - Observed: `applied=True`, `applied_channels=3`, `dtype=float32`, `nan_preserved=True`.

## Known limitations / risks
- Current per-channel DBE call uses fallback validity only (`valid_mask_hw=None` at call site), so alpha/coverage-driven masking is not yet implemented.
- FITS header DBE keywords (`ZMDBE*`) are still not written.

## Next item
- `Uses alpha_final / coverage to avoid touching invalid mosaic areas`

# Iteration note (2026-02-13, checklist parent: DBE hook placement unambiguous)

## What changed
- Updated `followup.md`:
  - Marked the parent row as complete:
    - `DBE hook placement is unambiguous`
- Updated `memory.md` (this file) for this iteration record.

## Why
- This was the next unchecked checklist item in order.
- Both required subitems under that parent row were already completed and verified, so the parent row is now legitimately complete.

## How tested
- `Select-String -Path followup.md -Pattern "DBE hook placement is unambiguous|zemosaic_worker.py: Phase 6|Grid mode: either confirmed"`
  - Verified parent row existed and both child rows were `[x]` before marking parent.
- `rg -n "\\[DBE\\] phase6 hook point reached \\(worker path\\)|\\[DBE\\] grid_mode: bypassing worker Phase 6" zemosaic_worker.py grid_mode.py`
  - Verified worker and grid hook markers are present in code.
- `Get-Content -Raw followup.md`
  - Verified the parent row is now `[x]`.

## Known limitations / risks
- No new functional DBE processing was added in this iteration; this is checklist-state consolidation based on prior completed code.
- Remaining DBE algorithm and export metadata items are still pending.

## Next item
- `Per-channel application (no full HWC background buffer)`

# Iteration note (2026-02-13, grid-mode DBE hook / no silent omission)

## What changed
- Updated `grid_mode.py`:
  - Extended `assemble_tiles(...)` signature with:
    - `final_mosaic_dbe_enabled: bool = True`
  - Added explicit grid-mode DBE hook logging right before final save:
    - enabled:
      - `[DBE] grid_mode: bypassing worker Phase 6, DBE hook point is in grid_mode final save path`
    - disabled:
      - `[DBE] grid_mode: disabled by config (final_mosaic_dbe_enabled=False)`
  - In `run_grid_mode()` / `_run_single_grid(...)`, derived `grid_dbe_flag` from:
    - `zconfig.final_mosaic_dbe_enabled` (preferred)
    - fallback `cfg_disk["final_mosaic_dbe_enabled"]`
    - fallback default `True`
  - Passed the resolved flag into `assemble_tiles(...)`.
- Updated `followup.md`:
  - Marked only this subitem `[x]`:
    - `Grid mode: either confirmed to pass through same save path OR explicit hook in grid_mode.py (no silent omission)`

## Why
- Completes the next unchecked grid-mode hook item by making DBE handling in grid mode explicit, with no silent bypass of worker Phase 6.
- Keeps change minimal: hook/log + config wiring only, no DBE pixel-processing yet.

## How tested
- `python -m py_compile grid_mode.py`
  - Result: pass (no syntax errors).
- `rg -n "final_mosaic_dbe_enabled|\\[DBE\\] grid_mode:|bypassing worker Phase 6|assemble_tiles\\(" grid_mode.py`
  - Result: confirmed new flag parameter, hook logs, config-read handoff, and call-site wiring.
- `Get-Content -Raw followup.md`
  - Result: confirmed only the targeted grid-mode subitem was newly marked `[x]`.

## Known limitations / risks
- Grid-mode DBE hook currently logs and routes config only; actual DBE subtraction algorithm is still pending.
- Top-level checklist row `DBE hook placement is unambiguous` remains unchecked in file state despite both subitems now checked.

## Next item
- `DBE hook placement is unambiguous:` (top-level checklist row in `followup.md`)

# Iteration note (2026-02-13, worker DBE config read)

## What changed
- Updated `zemosaic_worker.py`:
  - Added Phase 6 config read in both save-path blocks, immediately after `_dbg_rgb_stats("P6_PRE_EXPORT", ...)`:
    - `final_mosaic_dbe_enabled = bool(getattr(zconfig, "final_mosaic_dbe_enabled", True))`
  - Locations:
    - around `zemosaic_worker.py:22028`
    - around `zemosaic_worker.py:26261`
- Updated `followup.md`:
  - Marked `Worker: reads zconfig.final_mosaic_dbe_enabled with default True` as `[x]`.

## Why
- Implements the next unchecked worker item with minimal scope: ensure worker reads the DBE enable flag from runtime config and defaults to `True` when missing.
- This read is required before wiring the actual DBE application hook.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass (no syntax errors).
- `rg -n 'final_mosaic_dbe_enabled = bool\\(getattr\\(zconfig, "final_mosaic_dbe_enabled", True\\)\\)' zemosaic_worker.py`
  - Result: confirmed the read exists in both Phase 6 save blocks.
- `Get-Content -Raw followup.md`
  - Result: confirmed only this checklist item was newly marked `[x]`.

## Known limitations / risks
- DBE processing hook itself is not implemented yet; flag is read but not yet applied to mosaic data.
- No end-to-end run (classic/SDS/grid) was executed in this iteration.

## Next item
- `zemosaic_worker.py: Phase 6, immediately after _dbg_rgb_stats("P6_PRE_EXPORT", ...)` (DBE hook placement)

# Iteration note (2026-02-13, DBE hook placement in worker Phase 6)

## What changed
- Updated `zemosaic_worker.py`:
  - Added explicit DBE hook marker code in both Phase 6 save-path blocks, immediately after `_dbg_rgb_stats("P6_PRE_EXPORT", ...)`:
    - reads `final_mosaic_dbe_enabled = bool(getattr(zconfig, "final_mosaic_dbe_enabled", True))`
    - emits debug marker when enabled:
      - `logger.debug("[DBE] phase6 hook point reached (worker path)")`
  - Added inline comment to make placement intent explicit:
    - `DBE hook placement: post-P6_PRE_EXPORT stats, pre-export processing/writes.`
- Updated `followup.md`:
  - Marked only this subitem as complete:
    - `zemosaic_worker.py: Phase 6, immediately after _dbg_rgb_stats("P6_PRE_EXPORT", ...)`

## Why
- Completes the next unchecked follow-up item by making DBE hook location explicit and unambiguous in worker Phase 6.
- Keeps behavior minimal and non-invasive while preparing for actual DBE processing integration.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass (no syntax errors).
- `rg -n 'P6_PRE_EXPORT|final_mosaic_dbe_enabled = bool\\(getattr\\(zconfig, "final_mosaic_dbe_enabled", True\\)\\)|\\[DBE\\] phase6 hook point reached \\(worker path\\)' zemosaic_worker.py`
  - Result: confirmed both Phase 6 blocks contain stats marker + DBE hook lines.
- `Get-Content zemosaic_worker.py` line slices:
  - `22016..22034` and `26252..26270`
  - Result: confirmed hook block is placed directly after `_dbg_rgb_stats(...)` in both paths.
- `Get-Content -Raw followup.md`
  - Result: confirmed only this checklist line was newly marked `[x]`.

## Known limitations / risks
- This iteration adds hook placement only; DBE processing algorithm and application are still pending.
- Added `[DBE]` log is debug-level only and not yet the final INFO metrics log requested by later checklist items.

## Next item
- `Grid mode: either confirmed to pass through same save path OR explicit hook in grid_mode.py (no silent omission)`

# Iteration note (2026-02-13, Qt GUI DBE config key)

## What changed
- Updated `zemosaic_gui_qt.py`:
  - Added `final_mosaic_dbe_enabled` checkbox to the "Final assembly & output" group via `_register_checkbox(...)`.
  - Label text: `Dynamic Background Extraction (DBE) on final mosaic`.
  - Added fallback default config entry: `"final_mosaic_dbe_enabled": True` in `_baseline_default_config`.

## Why
- Mirrors the same DBE GUI/config behavior already added in Tk so Qt users can control the same feature flag.
- Default `True` keeps intended default-ON behavior when no prior config value exists.

## How tested
- `python -m py_compile zemosaic_gui_qt.py`
  - Result: pass (no syntax errors).
- `rg -n "final_mosaic_dbe_enabled|qt_field_final_mosaic_dbe|Dynamic Background Extraction \\(DBE\\) on final mosaic" zemosaic_gui_qt.py`
  - Verified checkbox key registration, label, and default value are present.
- `rg -n "_collect_config_from_widgets\\(\\)|_save_config\\(\\)" zemosaic_gui_qt.py`
  - Verified existing Qt flow collects widget-bound config keys and saves config.

## Known limitations / risks
- Worker-side DBE execution is still pending; this change only wires Qt UI/config.
- No full Qt GUI manual launch test was run in this iteration.

## Next item
- `Worker: reads zconfig.final_mosaic_dbe_enabled with default True`
