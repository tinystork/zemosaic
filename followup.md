# FOLLOW-UP INSTRUCTIONS — SDS / ZeSupaDupStack Batch Policy Fix

This document gives you a **step-by-step plan** to implement the SDS fixes described in `agent.md`.

Please follow the steps in order and keep changes minimal outside the SDS code paths.


---

## 1. Add SDS batch size config keys

**File:** `zemosaic_config.py`

1. Locate `DEFAULT_CONFIG` (the main dict with config defaults).
2. Add two new entries with safe defaults:

   - `sds_min_batch_size`: `5`
   - `sds_target_batch_size`: `10`

3. If there is any config load/merge logic elsewhere, ensure that:
   - If a user config file predates these keys, the defaults still apply.
   - Values are coerced to integers and at least 1 when used downstream.


---

## 2. Update SDS batch building in Qt Filter GUI

**File:** `zemosaic_filter_gui_qt.py`

### 2.1. Locate current SDS helpers

1. Find the SDS preview / grouping code, e.g.:

   - `_build_sds_batches_for_indices(...)`
   - the part of `_compute_auto_groups` or similar that calls it when SDS is enabled.
   - the code that serializes SDS groups into `overrides["preplan_master_groups"]`.

2. Confirm how the current SDS coverage grouping is implemented (grid, WCS descriptor, coverage_threshold).

### 2.2. Implement coverage + min + target batch policy

1. Introduce a helper in the Qt filter module, something like:

   ```python
   def _build_sds_batches_with_policy(
       entries: list[dict],
       descriptor: dict,
       coverage_threshold: float,
       min_batch_size: int,
       target_batch_size: int,
       logger_fn: Optional[Callable[[str], None]] = None,
   ) -> list[list[dict]]:
       ...
This helper must:

Reuse the existing coverage grid logic used by SDS (no new math).

Apply the algorithm described in section 3.2 of agent.md:

Add frames one by one, updating coverage.

Close batch when:

len(batch) >= min_batch_size and coverage >= threshold, OR

len(batch) >= target_batch_size (forced close).

Merge final leftover batch with the previous one if too small, etc.

Ensure you read sds_min_batch_size / sds_target_batch_size from the config/overrides in the same way sds_coverage_threshold is read:

Use safe defaults (5 / 10) if missing or invalid.

Use max(1, int(value)) to avoid zeros or negatives.

Replace the existing SDS preview function _build_sds_batches_for_indices(...) to call this new helper:

You can:

Either refactor _build_sds_batches_for_indices to become a small wrapper that calls _build_sds_batches_with_policy.

Or integrate the policy directly into the existing function if it’s easier.

Make sure the returned structure (a list of groups/indices) remains fully compatible with:

The tree view that displays SDS groups in the Filter GUI.

The serialization step that writes these groups into overrides["preplan_master_groups"].

2.3. Logging in Filter GUI
Where SDS preview batches are computed, log at least:

The coverage threshold.

The min and target batch sizes.

The number of SDS batches and their sizes.

Example (textual, later localized):

"SDS preview: thr=0.92, min=5, target=10 → 7 batches [10, 9, 8, ...]"

If the Filter GUI uses a local logger or text widget, route this message there.

3. Update SDS batch building in worker
File: zemosaic_worker.py

3.1. Locate SDS runtime function
Find assemble_global_mosaic_sds(...).

Identify where SDS:

Builds its list of Seestar entries (entry_infos or similar).

Builds SDS batches (coverage grid logic).

Logs sds_error_no_valid_batches and falls back to Mosaic-First.

3.2. Create a worker-side batch policy helper
Introduce a helper function near the SDS code, e.g.:

python
Copier le code
def _build_sds_batches_runtime(
    entry_infos: list[dict],
    global_plan: dict,
    coverage_threshold: float,
    min_batch_size: int,
    target_batch_size: int,
    logger: Optional[logging.Logger],
    pcb_fn: Optional[Callable[..., None]],
) -> list[list[dict]]:
    ...
Implement inside this helper the same algorithm as the Qt side:

Same definition of coverage grid and footprint.

Same closure rules:

len(batch) >= min_batch_size and coverage >= threshold → close.

len(batch) >= target_batch_size → force close.

Same merging of small final leftover batch.

Ensure consistency:

Batches are constructed over the same set/order of entry_infos as before.

If entry_infos is empty or invalid, return an empty list.

3.3. Wire the helper into assemble_global_mosaic_sds
At the beginning of assemble_global_mosaic_sds, read:

sds_coverage_threshold

sds_min_batch_size

sds_target_batch_size

from the config (using the same configuration access pattern as other options).

Before building SDS batches, log the policy via:

The worker logger.

And/or pcb_fn (if available) with a dedicated log key (e.g. "sds_info_batch_policy").

Replace the existing batch building logic in assemble_global_mosaic_sds with _build_sds_batches_runtime(...).

After batches are built:

If not batches:

Log sds_error_no_valid_batches with a clear reason.

Fallback to Mosaic-First as done today.

Else:

Log how many batches and their sizes (sds_debug_batch_coverage_summary style payload).

Proceed to run _assemble_global_mosaic_first_impl on each batch and stack the results exactly as before.

3.4. Respect preplan groups when possible
If preplan_master_groups is passed via overrides and SDS is active:

Check whether union of all preplan_master_groups indices matches the set of indices for entry_infos (or at least a significant subset).

If yes:

Optionally use preplan groups directly as SDS batches:

i.e. convert group indices → entry objects and skip coverage regrouping.

If no:

Keep using coverage-based grouping but log that preplan SDS groups could not be reused.

Important: Do not break existing non-SDS uses of preplan_master_groups.

4. Localization updates (only if used)
Files: locales/en.json, locales/fr.json

If you introduced new SDS messages that are user-visible (GUI or log pane), create matching keys in:

en.json

fr.json

Use the existing naming pattern, e.g.:

"sds_log_batch_policy"

"sds_log_batch_summary"

Ensure:

The Qt GUI and/or worker use these keys through the localization layer.

No raw English or French strings remain hard-coded for new messages that are visible to users.

5. Sanity checks & non-regression
After implementation, conceptually verify:

SDS ON, Seestar data, many frames:

Worker logs show:

sds_info_batch_policy with coverage, min, target values.

sds_debug_batch_coverage_summary or equivalent summarizing several batches with sizes ≥ sds_min_batch_size.

Filter GUI preview shows a similar batch structure.

The run finishes using SDS (no sds_error_no_valid_batches).

SDS OFF or non-Seestar series:

No SDS batch policy logs.

Behaviour identical to the previous Mosaic-First / classic pipeline.

Edge case (few frames, e.g. 1–3):

SDS still produces 1 small batch.

No crash; fallback only if coverage grid or WCS is invalid.

No change in:

Phase names and ETA logic in zemosaic_gui_qt.py.

Two-pass coverage renormalization.

Phase 4.5 / inter-master merging.

Classic Tk GUI.

If any of these checks fail, refine the SDS helper functions but keep changes strictly local to SDS logic.