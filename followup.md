# ZeMosaic – Grid / Survey mode hardening – Follow-up

This file tracks the progress of the Grid mode mission described in `agent.md`.

## Checklist

### A. Grid mode logging integration

- [x] A1 – Replace the `NullHandler`-based logger in `grid_mode.py` so that `_emit(...)` logs go to the same place as `ZeMosaicWorker`:
  - Use either the `"ZeMosaicWorker"` logger or ensure `"zemosaic.grid_mode"` propagates to a non-null parent.
  - Remove any `NullHandler` that swallows messages.
- [x] A2 – Ensure `_emit(...)` consistently prefixes messages with `"[GRID]"` (if not already done).
- [x] A3 – Add a small logging smoke test (e.g. `tests/test_grid_mode_logging.py`) that verifies `_emit(...)` output is captured by a handler attached to the main logger.
  - Notes: Grid-mode logger now uses the `ZeMosaicWorker` hierarchy directly, and a pytest smoke test captures the `[GRID]` message via `caplog`.

### B. Assembly robustness and explicit logs

- [ ] B1 – In `assemble_tiles(...)`, add/verify logs for:
  - Computing `tiles_list` and handling the case where `len(tiles_list) == 0`.
  - I/O failures for tiles (`Assembly: failed to read ...`).
  - Channel mismatch skips.
  - Empty valid-mask skips.
  - Final summary of kept tiles and failure counts.
- [ ] B2 – Ensure the “no tile infos” case:
  - Logs an explicit error summarizing `attempted`, `io_fail`, `channel_mismatch`, `empty_mask`, `kept=0`.
  - Returns `None` (no exceptions raised from inside `assemble_tiles`).
- [ ] B3 – Make salvage mode logging explicit:
  - Log when salvage starts (no valid data in initial mosaic).
  - Log whether salvage succeeded or failed.
  - On salvage failure, return `None` with a clear log message.
- [ ] B4 – On successful assembly:
  - Log the final mosaic `shape` and `dtype` before writing FITS.
- [ ] B5 – Add at least one test case exercising assembly robustness:
  - Case with mixed valid and invalid tiles → assembly succeeds and logs skips.
  - Case with all tiles invalid → `assemble_tiles(...)` returns `None` and logs a detailed failure summary.

### C. RGB equalization parity (Grid vs classic)

- [ ] C1 – Review and, if needed, refactor `grid_post_equalize_rgb(...)` in `grid_mode.py` so that:
  - It uses `equalize_rgb_medians_inplace(...)` from `zemosaic_align_stack.py` where available.
  - Its behaviour (medians, gains) matches the classic pipeline’s logic.
- [ ] C2 – Add detailed logs around RGB equalization in Grid mode:
  - Before calling `grid_post_equalize_rgb(...)`:
    - Log that RGB equalization is being invoked, with mosaic shape and weight shape.
  - Inside `grid_post_equalize_rgb(...)`:
    - Log when equalization is applied, including gains, medians, and target.
    - Log all skip reasons (non-RGB, no valid pixels, missing channel, invalid target, error).
- [ ] C3 – Ensure `grid_rgb_equalize` flag precedence is well-defined and documented:
  - Config (`grid_rgb_equalize` on disk) vs. parameter vs. default.
  - Log the final effective value and its source as `enabled=..., source=...`.
- [ ] C4 – In `zemosaic_worker.py`:
  - Introduce a clearly named `grid_rgb_equalize_flag` derived from the same config/UI semantics as `poststack_equalize_rgb` (or sensibly documented).
  - Log the value and source in the Grid branch.
  - Pass it explicitly as `grid_rgb_equalize=grid_rgb_equalize_flag` to `grid_mode.run_grid_mode(...)`.
- [ ] C5 – Add a unit test (e.g. `tests/test_grid_mode_rgb_equalize.py`) to compare:
  - `equalize_rgb_medians_inplace(arr_classic)` vs. `grid_post_equalize_rgb(arr_grid, weight_sum=None, ...)`.
  - Assert that resulting channel medians are equal within a small tolerance.

### D. Worker fallback behaviour

- [ ] D1 – Confirm that `run_hierarchical_mosaic(...)`:
  - Detects Grid mode using `detect_grid_mode(...)`.
  - Logs a line like `"[GRID] Invoking grid_mode.run_grid_mode(...) with grid_rgb_equalize=..., stack_norm=..., ..."`.
  - Calls `grid_mode.run_grid_mode(...)` inside a `try` block.
  - On success: returns early (does not run classic pipeline).
- [ ] D2 – On Grid mode exceptions:
  - Logs an ERROR with `exc_info=True` and a clear message:
    - `"[GRID] Grid/Survey mode failed, continuing with classic pipeline"`.
  - Then continues with the classic pipeline unchanged.
- [ ] D3 – (Optional) Add a small regression test or harness:
  - Using a fake Grid project, verify that:
    - Partial tile failures don’t break Grid mode.
    - Total failure leads to a logged fallback to classic pipeline.

### E. Regression & sanity checks

- [ ] E1 – Run the existing test suite; fix any regressions caused by changes in logging or imports.
- [ ] E2 – Verify that a **non-Grid** project behaves identically to the previous version:
  - Same logs (aside from harmless extra debug).
  - Same outputs.
- [ ] E3 – Verify that classic pipeline RGB equalization (`poststack_equalize_rgb`) is unchanged.
- [ ] E4 – Optionally, run a small real-world Grid project (if available) and inspect:
  - `zemosaic_worker.log` for the new `[GRID]` messages.
  - Final mosaic colours vs. classic pipeline (sanity check for RGB parity).

---

## Notes / Journal

Use this section to jot down important decisions, gotchas, or future ideas.

- [ ] (Example) Consider a future config key to *not* raise on empty mosaic in Grid mode but silently skip Grid and go classic, if that ever becomes desirable.

