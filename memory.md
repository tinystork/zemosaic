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

# Iteration note (2026-02-13, Qt DBE translation keys audit/fix)

## What changed
- Audited DBE translation keys used by `zemosaic_gui_qt.py`:
  - `qt_field_final_mosaic_dbe`
  - `qt_field_final_mosaic_dbe_strength`
  - `qt_dbe_strength_weak`
  - `qt_dbe_strength_normal`
  - `qt_dbe_strength_strong`
- Added missing keys to all locale JSON files:
  - `locales/en.json`
  - `locales/fr.json`
  - `locales/es.json`
  - `locales/pl.json`
  - `locales/de.json`
  - `locales/nl.json`
  - `locales/is.json`

## Why
- The DBE Qt feature had UI `_tr(...)` keys in code, but those keys were missing from locale JSON resources.
- Without keys in locale files, translations would rely on fallbacks instead of language-specific strings.

## How tested
- Presence check:
  - `rg -n '"qt_field_final_mosaic_dbe"|"qt_field_final_mosaic_dbe_strength"|"qt_dbe_strength_weak"|"qt_dbe_strength_normal"|"qt_dbe_strength_strong"' locales\\en.json locales\\fr.json locales\\es.json locales\\pl.json locales\\de.json locales\\nl.json locales\\is.json`
  - Result: all keys present in all seven locale files.
- JSON validity + key completeness:
  - Inline Python (`json.load`) over all locale files.
  - Result:
    - `en ok`
    - `fr ok`
    - `es ok`
    - `pl ok`
    - `de ok`
    - `nl ok`
    - `is ok`
- Syntax checks:
  - `python -m py_compile zemosaic_gui_qt.py locales\\zemosaic_localization.py`
  - Result: pass.

## Known limitations / risks
- Translations were added with best-effort wording; linguistic tuning by native speakers may still improve phrasing.
- No full GUI language switch walkthrough was run in this iteration.

## Next item
- Optional: manually switch Qt language to each locale and visually confirm DBE labels in the "Final assembly & output" section.

# Iteration note (2026-02-13, validation from zemosaic_worker_dbe_on_strong.log)

## What changed
- Reviewed `zemosaic_worker_dbe_on_strong.log` from the latest strong-preset test run.
- Updated `followup.md`:
  - Marked done criteria item as complete:
    - `Worker uses surface-fit DBE with ordered robust fallback (RBF -> gaussian -> skip)`

## Why
- The latest log confirms DBE runs in strong preset and successfully uses the RBF model on this dataset (no gaussian fallback taken in this run).

## How tested
- Command:
  - `rg -n "\[DBE\]|ERROR|CRITICAL|Traceback|Exception|FAILED" zemosaic_worker_dbe_on_strong.log`
- Key evidence:
  - `phase6 config strength=strong source=preset:strong obj_k=2.20 obj_dilate_px=4 sample_step=16 smoothing=0.250`
  - DEBUG shows memory-safe RBF path active:
    - `rbf_sample_cap_dynamic ...`
    - `rbf_eval_chunked ...`
  - Final applied log:
    - `applied=True ... strength=strong ... n_samples=840 ... model=rbf_thin_plate ... channels=3`
  - No `rbf_failed -> gaussian_fallback` in this latest run.

## Known limitations / risks
- This confirms DBE ON strong behavior for one run; it does not replace remaining manual checks:
  - DBE OFF skip path
  - preset switching comparison
  - forced fit failure fallback/skip verification at full pipeline level.

## Next item
- `Manual user test - DBE OFF: verify DBE skipped cleanly`

# Iteration note (2026-02-13, RBF MemoryError root-cause + mitigation)

## What changed
- Investigated `zemosaic_worker_dbe_on_strong.log` fallback behavior.
  - Confirmed run-level DBE settings were correct:
    - `strength=strong source=preset:strong obj_k=2.20 obj_dilate_px=4 sample_step=16 smoothing=0.250`
  - Confirmed per-channel RBF failures were `MemoryError` and all channels fell back to gaussian.
- Updated `zemosaic_worker.py` in `_apply_final_mosaic_dbe_per_channel(...)`:
  - Added memory/time guard for RBF sample count on large model grids:
    - dynamic cap based on `model_h * model_w` (`max_eval_pairs_rbf = 120_000_000`).
  - Replaced full-grid `rbf_model(xx, yy)` evaluation with chunked evaluation:
    - bounded by `max_eval_pairs_chunk = 8_000_000`,
    - evaluates flattened grid in chunks and reshapes back to `(model_h, model_w)`.
  - Added DEBUG diagnostics:
    - `rbf_sample_cap_dynamic` when dynamic sample cap is applied,
    - `rbf_eval_chunked` when chunked evaluation is used.

## Why
- Root cause:
  - SciPy `Rbf.__call__` on the full low-res grid allocates a large distance matrix of size approximately:
    - `n_eval_points x n_samples`.
  - On strong-run dimensions this becomes very large and can trigger `MemoryError`, causing fallback.
- Fix strategy:
  - keep RBF fit path, but bound evaluation memory by chunking and bound evaluation complexity by dynamic sample cap.

## How tested
- Compilation:
  - `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- Functional smoke:
  - Inline Python call to `_apply_final_mosaic_dbe_per_channel(...)` on synthetic data.
  - Result:
    - `smoke_applied True`
    - `smoke_model rbf_thin_plate`
    - `smoke_samples 1122`
- Large-grid probe (approximate problematic regime):
  - Inline Python call on `1024x1024` synthetic mosaic with DEBUG capture.
  - Result:
    - `large_applied True`
    - `large_model rbf_thin_plate`
    - `large_samples_used 342`
    - `large_has_memerr_fallback False`
    - `large_has_chunk_log True`
- Fallback integrity (forced RBF failure):
  - Monkeypatched `scipy.interpolate.Rbf` to raise `MemoryError`.
  - Result:
    - `fallback_applied True`
    - `fallback_model gaussian_fallback`
    - `fallback_reason0 rbf_failed:MemoryError`

## Known limitations / risks
- Dynamic sample capping may slightly smooth away small-scale background structure on very large mosaics (tradeoff for stability/performance).
- End-to-end validation on your real dataset is still required to confirm RBF now stays active more often under your runtime constraints.

## Next item
- Run the same strong dataset again and compare:
  - previous log: `rbf_failed:MemoryError` fallback,
  - expected now: chunked RBF path (`rbf_eval_chunked` DEBUG) and more `model=rbf_thin_plate` outcomes.

# Iteration note (2026-02-13, manual log check DBE ON strong)

## What changed
- Reviewed `zemosaic_worker_dbe_on_strong.log` for DBE phase-6 behavior on a real run.
- Updated `followup.md`:
  - Marked `[x]` for:
    - `Manual user test (reduced dataset) - DBE ON: verify logs show model + strength + params + n_samples`

## Why
- This run provides direct user-run evidence for section E DBE-ON logging expectations.

## How tested
- Command:
  - `rg -n "\[DBE\]|ERROR|Traceback|Exception|FAILED|CRITICAL" zemosaic_worker_dbe_on_strong.log`
- Key evidence found:
  - `phase6 gate enabled=True`
  - `phase6 config strength=strong source=preset:strong obj_k=2.20 obj_dilate_px=4 sample_step=16 smoothing=0.250`
  - `rbf_failed -> gaussian_fallback` on channels 0/1/2 (`MemoryError`)
  - `applied=True ... strength=strong ... n_samples=3715 ... model=gaussian_fallback ... channels=3`

## Known limitations / risks
- This log confirms DBE ON behavior and ordered RBF->gaussian fallback.
- It does not validate DBE OFF behavior, preset switching comparisons, or gaussian failure -> skip path.

## Next item
- `Manual user test - DBE OFF: verify DBE skipped cleanly`

# Iteration note (2026-02-13, Phase 6 DBE wiring presets + headers)

## What changed
- Updated `zemosaic_worker.py`:
  - Added `_DBE_STRENGTH_PRESETS` and `_resolve_final_mosaic_dbe_params_from_config(config_obj)` to resolve DBE parameters from:
    - presets (`weak` / `normal` / `strong`),
    - `custom` config values (`final_mosaic_dbe_obj_k`, `final_mosaic_dbe_obj_dilate_px`, `final_mosaic_dbe_sample_step`, `final_mosaic_dbe_smoothing`).
  - In both Phase 6 DBE blocks:
    - resolved DBE parameters from config,
    - passed resolved values to `_apply_final_mosaic_dbe_per_channel(...)` (`obj_k`, `obj_dilate_px`, `sample_step`, `smoothing`, `strength`).
  - Enriched applied/failed DBE logs with:
    - `strength`, `obj_k`, `obj_dilate_px`, `sample_step`, `smoothing`, `n_samples`, and `model`.
  - Extended FITS header writes (applied path only) in both blocks:
    - `ZMDBE_STR`, `ZMDBE_DIL`, `ZMDBE_STP`, `ZMDBE_SMO`, `ZMDBE_MDL`.
- Updated `followup.md` section D:
  - checked all 4 items under `### D — Hook Phase 6 wiring (x2 blocks)`.

## Why
- Section D required wiring advanced DBE config/preset values into both Phase 6 hook blocks, plus richer observability and optional header metadata.
- The new resolver keeps mapping logic centralized and consistent across both Phase 6 save paths.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- `@' ... '@ | python -` inline probe:
  - Verified resolver mapping:
    - `weak -> (4.0, 2, 32, 1.0)`
    - `normal -> (3.0, 3, 24, 0.6)`
    - `strong -> (2.2, 4, 16, 0.25)`
    - `custom -> uses explicit config values`
    - invalid strength -> fallback `normal`.
  - Verified DBE apply call with resolved `custom` values returns info containing expected:
    - `strength=custom`
    - configured params
    - `model=rbf_thin_plate`
    - sample counts populated.
- Static wiring checks:
  - `rg -n -F 'valid_mask_hw=dbe_valid_mask_hw' zemosaic_worker.py`
    - Result: 2 occurrences (both Phase 6 blocks).
  - `rg -n -F 'obj_k=float(dbe_params.get(\"obj_k\"' zemosaic_worker.py`
  - `rg -n -F 'obj_dilate_px=int(dbe_params.get(\"obj_dilate_px\"' zemosaic_worker.py`
  - `rg -n -F 'sample_step=int(dbe_params.get(\"sample_step\"' zemosaic_worker.py`
  - `rg -n -F 'smoothing=float(dbe_params.get(\"smoothing\"' zemosaic_worker.py`
  - `rg -n -F 'strength=str(dbe_params.get(\"strength\"' zemosaic_worker.py`
    - Result: each appears in both Phase 6 blocks.
  - `rg -n -F 'final_header[\"ZMDBE_STR\"]' zemosaic_worker.py`
  - `rg -n -F 'final_header[\"ZMDBE_DIL\"]' zemosaic_worker.py`
  - `rg -n -F 'final_header[\"ZMDBE_STP\"]' zemosaic_worker.py`
  - `rg -n -F 'final_header[\"ZMDBE_SMO\"]' zemosaic_worker.py`
  - `rg -n -F 'final_header[\"ZMDBE_MDL\"]' zemosaic_worker.py`
    - Result: each header key appears in both Phase 6 blocks.

## Expected log line example
- Applied:
  - `[DBE] applied=True mode=per_channel strength=normal obj_k=3.00 obj_dilate_px=3 sample_step=24 smoothing=0.600 ds_factor=... blur_sigma=... valid_frac=... valid_px=... total_px=... n_samples=... mean_abs_bg_sub=... model=rbf_thin_plate channels=3 time_ms=...`

## Known limitations / risks
- No full end-to-end reduced dataset run was executed in this iteration; manual smoke checks in section E remain pending.
- FITS key length > 8 may be represented as `HIERARCH` by writer depending on backend behavior (existing code already uses longer keys elsewhere).

## Next item
- `Manual user test (reduced dataset) - DBE ON: verify logs show model + strength + params + n_samples.`

# Iteration note (2026-02-13, DBE algorithm details + limits summary)

## What changed
- Added a dedicated DBE-v2 algorithm summary in `memory.md` with explicit implementation details and operational limits.
- Summary recorded:
  - Per-channel processing pipeline at low-res:
    - valid mask handling,
    - object detection (`median + MAD`, `thr = median + obj_k * 1.4826 * MAD`),
    - object dilation (`obj_dilate_px`),
    - background grid sampling (`sample_step`, local median window),
    - surface fit attempt (SciPy RBF thin-plate, `smooth=smoothing`),
    - ordered fallback (`RBF -> gaussian -> skip`),
    - upsample + subtract on valid pixels only.
  - Key limits/guards:
    - `max_samples_rbf = 2000`
    - `min_samples_rbf = 30`
    - `sample_step >= 1`
    - fail-open if gaussian also fails.
  - Main DBE info fields now emitted:
    - `model`, `strength`, `params`, `sample_counts`, `fallback_info`,
    - plus per-channel counts/fallback arrays and mask statistics.
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Update memory.md with algorithm details + limits.`

## Why
- This was the next unchecked checklist item in section C.
- A dedicated summary section makes current DBE behavior and limits immediately discoverable without scanning multiple iteration notes.

## How tested
- `rg -n "DBE algorithm details \\+ limits summary|max_samples_rbf|min_samples_rbf|RBF -> gaussian -> skip|sample_counts|fallback_info|obj_k \\* 1.4826" memory.md`
  - Result: confirmed the new summary section includes required algorithm details and limit values.
- `rg -n "Update `memory.md` with algorithm details \\+ limits\\." followup.md`
  - Result: confirmed checklist line exists and was updated to `[x]`.

## Known limitations / risks
- This iteration is documentation-only; no worker behavior changed.
- Phase 6 config/preset wiring remains pending in section D.

## Next item
- `In BOTH phase6 DBE blocks in zemosaic_worker.py, read config values and pass them into DBE.`

# Iteration note (2026-02-13, DBE DEBUG fallback transition logs)

## What changed
- Updated `zemosaic_worker.py` in `_apply_final_mosaic_dbe_per_channel(...)` to emit explicit DEBUG transition logs for fallback flow:
  - `"[DBE] rbf_failed -> gaussian_fallback chan=%d reason=%s"`
  - `"[DBE] gaussian_failed -> dbe_skipped chan=%d reason=%s"`
- Logging is gated by `logger.isEnabledFor(logging.DEBUG)` (`dbe_debug`) so no extra INFO noise was added.
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Add DEBUG logs for fallback transitions and failure reasons.`

## Why
- This was the next unchecked item in section C.
- It makes fallback path transitions explicit in DEBUG mode as required by `agent.md`, without changing algorithm behavior.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- `rg -n "rbf_failed -> gaussian_fallback|gaussian_failed -> dbe_skipped|dbe_debug" zemosaic_worker.py`
  - Result: confirmed the new DEBUG transition logs and debug-gate are present.
- Synthetic runtime probe with debug log capture:
  - Captured logger output with DEBUG level.
  - Forced RBF failure (`scipy.interpolate.Rbf` monkeypatch) then forced gaussian failure (`cv2.GaussianBlur` monkeypatch).
  - Observed:
    - `has_rbf_transition True`
    - `has_gauss_transition True`
    - `info1_model gaussian_fallback`
    - `info2_applied False`
    - `info2_reason no_channel_applied`

## Known limitations / risks
- These are function-level DEBUG logs; phase6 call-site wiring/log enrichment is still pending in section D.
- No manual dataset smoke run executed in this iteration.

## Next item
- `Update memory.md with algorithm details + limits.`

# Iteration note (2026-02-13, DBE info dict extension)

## What changed
- Updated `zemosaic_worker.py` in `_apply_final_mosaic_dbe_per_channel(...)` to extend DBE metadata output (`dbe_info`) with the requested structure:
  - `model` (already present, retained and populated as `rbf_thin_plate` / `gaussian_fallback` / `mixed` / `none`),
  - `strength` (new),
  - `params` (new, includes DBE parameters and derived shape/downsample context),
  - `sample_counts` (new, per-channel + totals for raw/used sample counts),
  - `fallback_info` (new, per-channel fallback summary + counts).
- Added optional function parameter:
  - `strength: str = "normal"`
  - This updates metadata only in this iteration (wiring from phase6 config remains pending in section D).
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Extend DBE info dict: model, strength, params, sample counts, fallback info.`

## Why
- This was the next unchecked item in section C.
- It standardizes metadata for downstream logging and validation without changing DBE math paths.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- `rg -n '"strength"|"params"|"sample_counts"|"fallback_info"|"model": "none"' zemosaic_worker.py`
  - Result: confirmed field declarations and population paths.
- Synthetic runtime probe (inline Python) calling `_apply_final_mosaic_dbe_per_channel(...)` with `strength='strong'`:
  - `has_model True rbf_thin_plate`
  - `has_strength True strong`
  - `has_params True ['blur_sigma', 'ds_factor', 'max_model_side', 'model_shape', 'obj_dilate_px', 'obj_k', 'sample_step', 'smoothing']`
  - `has_sample_counts True {'per_channel_raw': [255, 255, 255], 'per_channel_used': [255, 255, 255], 'total_raw': 765, 'total_used': 765}`
  - `has_fallback_info True {'per_channel': [], 'counts': {}, 'any_fallback': False}`
  - `fallback_per_channel_len 3`
  - `sample_counts_per_channel_len 3`

## Known limitations / risks
- `fallback_info["per_channel"]` currently stores only non-empty summary entries; it is intentionally compact and not one-to-one with channel count.
- Phase 6 call-site wiring for `strength` and other advanced params is still pending (section D).

## Next item
- `Add DEBUG logs for fallback transitions and failure reasons.`

# Iteration note (2026-02-13, per-channel processing verified / no HWC model buffer)

## What changed
- No worker algorithm code changes were required for this item.
- Verified in `zemosaic_worker.py` that DBE processing remains per-channel and does not allocate a full HWC background model buffer:
  - loop is per channel (`for chan_idx in range(3)`),
  - background models are 2D per channel (`bg_lr`, `bg_full`),
  - subtraction is applied on channel slices (`channel[use_mask] = ...`).
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Keep per-channel processing (no HWC model allocation).`

## Why
- This was the next unchecked item in section C.
- The current implementation already satisfies this memory constraint, so reimplementation was unnecessary and avoided risk.

## How tested
- `rg -n "for chan_idx in range\\(3\\)|channel = mosaic\\[\\.\\.\\., chan_idx\\]|bg_lr|bg_full|model_shape|np\\.dstack|np\\.stack|np\\.zeros\\(\\(.*3\\)|np\\.empty\\(\\(.*3\\)" zemosaic_worker.py`
  - Result: DBE function uses per-channel loop and 2D model arrays (`bg_lr`, `bg_full`); no HWC model allocation inside DBE path.
- `Get-Content zemosaic_worker.py | Select-Object -Index (2818..3065)`
  - Result: manual code-path verification of per-channel processing and 2D background handling.
- Runtime probe (inline Python):
  - Called `_apply_final_mosaic_dbe_per_channel(...)` on synthetic `HWC` input and verified:
    - `applied=True`
    - `applied_channels=3`
    - output shape preserved as `(H, W, 3)`,
    - per-channel model reporting via `model_per_channel`.
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.

## Known limitations / risks
- This check confirms per-channel processing semantics by code-path inspection and probe, not by full memory profiler instrumentation.
- Next DBE item (info dict completeness) is still pending and may require additional metadata fields (e.g., strength/params harmonization at call-site level).

## Next item
- `Extend DBE info dict: model, strength, params, sample counts, fallback info.`

# Iteration note (2026-02-13, worker RBF surface-fit + ordered fallback)

## What changed
- Updated `zemosaic_worker.py` in `_apply_final_mosaic_dbe_per_channel(...)`:
  - Added `smoothing` parameter (default `0.6`) for RBF thin-plate fit.
  - Implemented surface fit with `scipy.interpolate.Rbf(..., function="thin_plate", smooth=smoothing)` using sampled low-res background points.
  - Added sample cap for RBF fit: `max_samples_rbf = 2000`.
  - Added minimum sample threshold for RBF: `min_samples_rbf = 30`.
  - Implemented required fail-safe model order per channel:
    - `RBF` -> `gaussian_fallback` -> `skip channel`.
  - Preserved fail-open behavior if gaussian fallback also fails (no crash; channel skipped safely).
  - Added metadata fields for verification/diagnostics:
    - `bg_sample_count_used_lr`
    - `model_per_channel`
    - `fallback_per_channel`
    - aggregate `model`.
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Implement surface fit using SciPy RBF thin-plate + smoothing` (and its listed fallback/cap expectations).

## Why
- This was the next unchecked item in section C.
- It establishes the requested surface-fit background model while maintaining robust fallback behavior and bounded sample count.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- `rg -n "scipy_rbf|rbf_thin_plate|max_samples_rbf|min_samples_rbf|gaussian_fallback|bg_sample_count_used_lr|model_per_channel|fallback_per_channel" zemosaic_worker.py`
  - Result: confirmed RBF fit, cap/threshold, fallback path, and metadata fields.
- Synthetic runtime probe (inline Python) with four scenarios:
  1. Normal RBF path (`sample_step=2`):
     - `case1_model rbf_thin_plate`
     - `case1_model_per_channel ['rbf_thin_plate', 'rbf_thin_plate', 'rbf_thin_plate']`
     - `case1_used_samples [2000, 2000, 2000]`
     - `case1_cap_ok True`
  2. Too few samples (`sample_step=512`) -> gaussian fallback:
     - `case2_model gaussian_fallback`
     - `case2_fallback ['rbf_too_few_samples', ...]`
  3. Forced RBF exception (monkeypatch `scipy.interpolate.Rbf`) -> gaussian fallback:
     - `case3_model gaussian_fallback`
     - `case3_fallback ['rbf_failed:RuntimeError', ...]`
  4. Forced RBF failure + forced gaussian failure (monkeypatch both `Rbf` and `cv2.GaussianBlur`) -> safe skip:
     - `case4_applied False`
     - `case4_reason no_channel_applied`
     - `case4_model_per_channel ['skipped', 'skipped', 'skipped']`
     - `case4_fallback ['rbf_failed:RuntimeError|gaussian_failed:RuntimeError', ...]`

## Known limitations / risks
- Phase 6 wiring still does not pass `smoothing` from config/presets into DBE call sites (defaults are used at current call sites).
- DEBUG log transitions in worker call-site logs are still pending (separate checklist item).

## Next item
- `Keep per-channel processing (no HWC model allocation).`

# Iteration note (2026-02-13, worker background sampling grid + robust local median)

## What changed
- Updated `zemosaic_worker.py` in `_apply_final_mosaic_dbe_per_channel(...)`:
  - Added `sample_step` parameter (default `24`).
  - Implemented background sampling on a regular low-res grid:
    - iterate points with `range(0, model_h, sample_step)` and `range(0, model_w, sample_step)`,
    - use local window radius `sample_step // 2`,
    - keep only pixels from `bg_mask_lr`,
    - compute per-sample value as robust local median (`np.nanmedian(local_vals)`).
  - Integrated sampled grid into current gaussian pipeline:
    - build sparse sampled map,
    - fill missing cells with median sample value,
    - then run current gaussian smoothing.
  - Added sampling metadata to `dbe_info`:
    - `sample_step`
    - `bg_sample_count_lr` (per-channel sample counts).
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Implement background sampling on grid (sample_step) with robust local median.`

## Why
- This was the next unchecked item in section C.
- Sampling explicit background points is required groundwork before surface fitting, while keeping the existing gaussian model path intact for minimal risk.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- `rg -n "sample_step|sample_radius|bg_sample_count_lr|local_bg_mask|np\\.nanmedian\\(local_vals\\)|range\\(0, model_h, sample_step_i\\)" zemosaic_worker.py`
  - Result: confirmed grid-step loops, local-median sampling, and metadata fields are present.
- Synthetic runtime probe (inline Python) with same image and two sampling steps:
  - Called `_apply_final_mosaic_dbe_per_channel(...)` with `sample_step=8` and `sample_step=32`.
  - Observed:
    - `len_counts_step8 3`
    - `len_counts_step32 3`
    - `counts_step8 [567, 567, 567]`
    - `counts_step32 [36, 36, 36]`
    - `more_samples_with_smaller_step True`
    - `sample_step_recorded_step8 8`
    - `sample_step_recorded_step32 32`
    - `applied_step8 True`

## Known limitations / risks
- This iteration does not yet implement RBF/thin-plate fitting; gaussian smoothing is still the model step.
- Phase 6 wiring still does not pass `sample_step` from presets/config into DBE call sites.

## Next item
- `Implement surface fit using SciPy RBF thin-plate + smoothing (with fallback order).`

# Iteration note (2026-02-13, worker low-res object mask + dilation)

## What changed
- Updated `zemosaic_worker.py` in `_apply_final_mosaic_dbe_per_channel(...)`:
  - Added parameter `obj_dilate_px` (default `3`).
  - Implemented low-res object mask computation per channel using:
    - `median` and `MAD` on valid low-res pixels,
    - robust sigma `1.4826 * MAD`,
    - threshold `thr = median + obj_k * robust_sigma`,
    - raw object mask: `channel_lr > thr`,
    - dilation with `cv2.dilate` (elliptical kernel sized from `obj_dilate_px`).
  - Implemented `bg_mask_lr = valid_lr & ~object_mask_dilated` and used it for background-only gaussian modeling input.
  - Added DBE info fields for observability/testing:
    - `obj_dilate_px`
    - `obj_mask_lr_raw_pixels`
    - `obj_mask_lr_dilated_pixels`
    - `bg_mask_lr_pixels`
    - `obj_thr_lr`
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Implement object mask in low-res using median+MAD threshold + dilation.`

## Why
- This was the next unchecked checklist item in section C.
- It introduces the required robust low-res object/background separation step while keeping the rest of the DBE pipeline unchanged and minimal.

## How tested
- `python -m py_compile zemosaic_worker.py`
  - Result: pass.
- `rg -n "obj_dilate_px|obj_mask_lr_raw_pixels|obj_mask_lr_dilated_pixels|bg_mask_lr_pixels|obj_thr_lr|1.4826|cv2\\.dilate|bg_mask_lr" zemosaic_worker.py`
  - Result: confirmed threshold, MAD scaling, dilation, bg-mask logic, and info fields are present.
- Synthetic runtime probe:
  - Command: inline Python script calling `_apply_final_mosaic_dbe_per_channel(...)` twice on a synthetic image (bright square object), once with `obj_dilate_px=0`, once with `obj_dilate_px=3`.
  - Observed output:
    - `no_dil_raw_ch0 64`
    - `no_dil_dilated_ch0 64`
    - `dil_raw_ch0 64`
    - `dil_dilated_ch0 180`
    - `dil_gt_no_dil True`
    - `thr_logged True`
    - `bg_mask_logged True`
    - `applied True`

## Known limitations / risks
- This iteration does not yet add grid sampling (`sample_step`) or RBF fitting; gaussian model is still used.
- Phase 6 wiring does not yet pass `obj_dilate_px` from config/presets; current behavior uses function default when called.

## Next item
- `Implement background sampling on grid (sample_step) with robust local median.`

# Iteration note (2026-02-13, Qt DBE strength presets UI)

## What changed
- Updated `zemosaic_gui_qt.py` in `_create_final_assembly_group(...)`:
  - Added a `QComboBox` labeled `DBE strength` directly below the existing `final_mosaic_dbe_enabled` checkbox row.
  - Added exactly three visible preset options in GUI:
    - `Weak` (`weak`)
    - `Normal` (`normal`)
    - `Strong` (`strong`)
  - Bound combobox value to config key `final_mosaic_dbe_strength` via `_config_fields`.
  - Added enable/disable wiring so DBE strength controls follow `final_mosaic_dbe_enabled` checkbox state.
  - Kept `custom` config-only behavior:
    - no `Custom` option is shown in the combobox,
    - if loaded config uses `final_mosaic_dbe_strength="custom"`, that value is preserved until user explicitly changes the combobox.
- Updated `followup.md` under `### B — Qt GUI (presets)`:
  - Marked all four B items as `[x]`.

## Why
- This completes the remaining Qt GUI preset requirements with minimal targeted edits in the final-assembly UI section only.
- Preserving existing `custom` values avoids clobbering power-user JSON settings while still keeping GUI exposure limited to Weak/Normal/Strong.

## How tested
- `python -m py_compile zemosaic_gui_qt.py`
  - Result: pass.
- `rg -n "final_mosaic_dbe_strength|qt_dbe_strength_weak|qt_dbe_strength_normal|qt_dbe_strength_strong|DBE strength" zemosaic_gui_qt.py`
  - Result: combobox label, three preset options, and config binding are present.
- `rg -n "_update_dbe_strength_enabled|dbe_checkbox\\.toggled\\.connect|setEnabled\\(enabled\\)" zemosaic_gui_qt.py`
  - Result: combobox enabled state is explicitly tied to DBE checkbox toggle.
- `rg -n "custom|initial_custom_strength|preserve_custom|user_changed" zemosaic_gui_qt.py`
  - Result: preservation logic for config-only `custom` exists; no GUI `Custom` option added.
- `Get-Content zemosaic_gui_qt.py | Select-Object -Index (2492..2576)`
  - Result: visual verification that DBE checkbox row is followed by DBE strength row in final assembly section.

## Known limitations / risks
- No interactive Qt launch screenshot was captured in this iteration; verification is code/static-level.
- Worker-side use of `final_mosaic_dbe_strength` presets is still pending (GUI now captures/preserves it).

## UI notes
- UI location:
  - File: `zemosaic_gui_qt.py`
  - Function: `_create_final_assembly_group(...)`
  - Section: "Final assembly & output", near `final_mosaic_dbe_enabled`.
- Screenshot:
  - Not captured in this iteration.

## Next item
- `Implement object mask in low-res using median+MAD threshold + dilation.`

# Iteration note (2026-02-13, DBE key names + preset mapping documented)

## What changed
- Updated `memory.md` with the chosen DBE-v2 config key names and preset mapping values.
- Documented key names:
  - `final_mosaic_dbe_strength`
  - `final_mosaic_dbe_obj_k`
  - `final_mosaic_dbe_obj_dilate_px`
  - `final_mosaic_dbe_sample_step`
  - `final_mosaic_dbe_smoothing`
- Documented preset mapping (low-res space):
  - `weak`: `obj_k=4.0`, `obj_dilate_px=2`, `sample_step=32`, `smoothing=1.0`
  - `normal` (default): `obj_k=3.0`, `obj_dilate_px=3`, `sample_step=24`, `smoothing=0.6`
  - `strong`: `obj_k=2.2`, `obj_dilate_px=4`, `sample_step=16`, `smoothing=0.25`
  - `custom`: values read strictly from config.
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Update memory.md with chosen key names + preset mapping.`

## Why
- This was the next unchecked checklist item.
- Having one explicit mapping reference in `memory.md` avoids ambiguity before wiring UI preset selection and worker parameter routing.

## How tested
- `rg -n "Update `memory.md` with chosen key names \\+ preset mapping\\.|### A — Configuration" followup.md`
  - Result: item is now `[x]` under section A.
- `rg -n "final_mosaic_dbe_strength|final_mosaic_dbe_obj_k|final_mosaic_dbe_obj_dilate_px|final_mosaic_dbe_sample_step|final_mosaic_dbe_smoothing|weak|normal|strong|custom" memory.md`
  - Result: key names and all four preset labels/values are present in this new iteration note.
- Cross-check against source instruction:
  - `rg -n "Mapping des presets|Weak:|Normal \\(default\\):|Strong:|Custom:" agent.md`
  - Result: mapping values match `agent.md`.

## Known limitations / risks
- This iteration is documentation-only; no runtime behavior changed.
- Qt combobox UI and worker preset application are still pending.

## Next item
- `In zemosaic_gui_qt.py, add combobox "DBE strength" next to DBE checkbox: Weak / Normal / Strong.`

# Iteration note (2026-02-13, Qt fallback defaults for DBE-v2 keys)

## What changed
- Updated `zemosaic_gui_qt.py` in `_baseline_default_config()` fallback defaults:
  - Added `final_mosaic_dbe_strength: "normal"`
  - Added `final_mosaic_dbe_obj_k: 3.0`
  - Added `final_mosaic_dbe_obj_dilate_px: 3`
  - Added `final_mosaic_dbe_sample_step: 24`
  - Added `final_mosaic_dbe_smoothing: 0.6`
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Add same defaults in Qt fallback defaults (zemosaic_gui_qt.py default config dict).`

## Why
- This was the next unchecked checklist item.
- Keeps Qt fallback defaults aligned with `zemosaic_config.py` so config load/save behavior remains consistent when fallback values are used.

## How tested
- `python -m py_compile zemosaic_gui_qt.py`
  - Result: pass.
- `rg -n "final_mosaic_dbe_strength|final_mosaic_dbe_obj_k|final_mosaic_dbe_obj_dilate_px|final_mosaic_dbe_sample_step|final_mosaic_dbe_smoothing" zemosaic_gui_qt.py`
  - Result: all five keys present at `zemosaic_gui_qt.py:3977-3981`.
- `rg -n "final_mosaic_dbe_strength|final_mosaic_dbe_obj_k|final_mosaic_dbe_obj_dilate_px|final_mosaic_dbe_sample_step|final_mosaic_dbe_smoothing" zemosaic_config.py`
  - Result: confirmed same values exist in `zemosaic_config.py:220-224` for parity.

## Known limitations / risks
- This iteration updates config defaults only; Qt DBE strength combobox is still not implemented.
- Worker wiring for strength/custom parameter usage is still pending.

## Next item
- `Update memory.md with chosen key names + preset mapping.`

# Iteration note (2026-02-13, config defaults for DBE-v2 keys)

## What changed
- Updated `zemosaic_config.py` `DEFAULT_CONFIG` with five new DBE-v2 keys:
  - `final_mosaic_dbe_strength`: `"normal"`
  - `final_mosaic_dbe_obj_k`: `3.0`
  - `final_mosaic_dbe_obj_dilate_px`: `3`
  - `final_mosaic_dbe_sample_step`: `24`
  - `final_mosaic_dbe_smoothing`: `0.6`
- Updated `followup.md`:
  - Marked only this item as complete:
    - `Add new config keys in zemosaic_config.py defaults: ...`

## Why
- This was the next unchecked checklist item.
- These defaults establish the baseline "Normal" preset values in persistent config, so save/load paths can carry DBE-v2 parameters.

## How tested
- `python -m py_compile zemosaic_config.py`
  - Result: pass.
- `rg -n "final_mosaic_dbe_strength|final_mosaic_dbe_obj_k|final_mosaic_dbe_obj_dilate_px|final_mosaic_dbe_sample_step|final_mosaic_dbe_smoothing" zemosaic_config.py`
  - Result: all five keys found in `DEFAULT_CONFIG` (lines 220-224).
- Inline runtime probe:
  - Command:
    - `@' ... import zemosaic_config as c ... print(DEFAULT_CONFIG values) ... '@ | python -`
  - Result:
    - `final_mosaic_dbe_strength='normal'`
    - `final_mosaic_dbe_obj_k=3.0`
    - `final_mosaic_dbe_obj_dilate_px=3`
    - `final_mosaic_dbe_sample_step=24`
    - `final_mosaic_dbe_smoothing=0.6`

## Known limitations / risks
- Qt fallback defaults are not yet updated in this iteration (`zemosaic_gui_qt.py` still pending).
- Worker wiring for these new keys is still pending.

## Next item
- `Add same defaults in Qt fallback defaults (zemosaic_gui_qt.py default config dict).`

# Iteration note (2026-02-13, preflight doubtful-item code audit)

## What changed
- Ran a targeted preflight audit for doubtful DBE-v2 checklist items before any implementation edits:
  - Checked whether new DBE config keys already exist in `zemosaic_config.py`.
  - Checked whether Qt fallback defaults and UI already expose DBE strength presets in `zemosaic_gui_qt.py`.
  - Checked whether worker already contains RBF/thin-plate + ordered fallback markers in `zemosaic_worker.py`.
- Updated `followup.md`:
  - Marked only this item as complete:
    - `For each doubtful item, run a quick code check before implementation (avoid duplicate work).`

## Why
- The next unchecked checklist row was the mandatory preflight process rule.
- Completing this audit prevents duplicate implementation work and confirms which large DBE-v2 items are still genuinely pending.

## How tested
- `rg -n "final_mosaic_dbe_strength|final_mosaic_dbe_obj_k|final_mosaic_dbe_obj_dilate_px|final_mosaic_dbe_sample_step|final_mosaic_dbe_smoothing" zemosaic_config.py`
  - Result: no matches (keys not yet present in config defaults).
- `rg -n "final_mosaic_dbe_strength|final_mosaic_dbe_obj_k|final_mosaic_dbe_obj_dilate_px|final_mosaic_dbe_sample_step|final_mosaic_dbe_smoothing" zemosaic_gui_qt.py`
  - Result: no matches (advanced DBE keys not yet present in Qt fallback defaults).
- `rg -n "QComboBox|DBE strength|Weak|Normal|Strong" zemosaic_gui_qt.py`
  - Result: generic `QComboBox` usage found, but no DBE strength preset control found.
- `rg -n "scipy|Rbf|thin_plate|gaussian_fallback|rbf_failed|gaussian_failed|obj_dilate_px|sample_step|smoothing|final_mosaic_dbe_strength" zemosaic_worker.py`
  - Result: no DBE RBF/preset/fallback markers found (only unrelated scipy gaussian usage).
- `python -m py_compile zemosaic_config.py zemosaic_gui_qt.py zemosaic_worker.py`
  - Result: pass (no syntax errors).

## Known limitations / risks
- This iteration is verification-only; no DBE feature behavior changed.
- The remaining DBE-v2 tasks are still pending implementation and runtime validation.

## Next item
- `Add new config keys in zemosaic_config.py defaults: final_mosaic_dbe_strength, final_mosaic_dbe_obj_k, final_mosaic_dbe_obj_dilate_px, final_mosaic_dbe_sample_step, final_mosaic_dbe_smoothing`

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
# Iteration note (2026-02-21, startup backend policy: Qt default, Tk fallback-only)

## What changed
- Updated `run_zemosaic.py` backend selection policy:
  - Removed config-driven startup backend selection from launcher path.
  - Kept only explicit overrides for startup backend:
    - CLI flags: `--qt-gui` / `--tk-gui`
    - env var: `ZEMOSAIC_GUI_BACKEND=qt|tk`
  - Enforced default startup backend to Qt (`zemosaic_gui_qt.py`) when no explicit override is present.
- Improved startup logging clarity:
  - Explicit Tk override logs now state source (`cli` or `env`).
  - Automatic fallback to Tk keeps an explicit fallback log path when Qt import fails.
- Removed now-unused launcher imports/helpers tied to startup config backend preference parsing.

## Why
- Requested behavior is: start on PySide6/Qt by default, and use Tkinter only as fallback when Qt is unavailable (or when explicitly forced for diagnostics).
- Previous logic could select Tk from stored config preference, which conflicted with this policy.

## How tested
- Syntax check:
  - `python3 -m py_compile run_zemosaic.py`
  - Result: pass.
- Static verification:
  - Confirmed `_determine_backend(...)` now returns Qt by default unless CLI/env explicitly requests otherwise.
  - Confirmed fallback message path still logs:
    - `[run_zemosaic] Falling back to the classic Tk interface.`
  - Confirmed Tk launch log distinguishes:
    - explicit override (`cli`/`env`) vs automatic fallback.

## Known limitations / risks
- Any stored `preferred_gui_backend` value in config is now intentionally ignored at launcher startup.
- If users relied on persistent Tk startup via config, they must now use `--tk-gui` or `ZEMOSAIC_GUI_BACKEND=tk`.

## Next item
- Optional: align README startup-backend wording to explicitly document the new startup precedence:
  - `CLI/env override > Qt default > Tk fallback on Qt import failure`.
