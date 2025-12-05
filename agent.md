# ZeMosaic – Grid / Survey mode hardening (logging, robustness, RGB parity)

> TL;DR for Codex:
> - Make Grid/Survey mode **talk in the main log**.
> - Make assembly **skip bad tiles instead of silently dying**, and be very explicit when it really has to give up.
> - Make sure RGB equalization in Grid mode is **strictly coherent** with the classic pipeline (same logic, same gains, same switches).
> - Keep the rest of the workflow and GUI **unchanged**.

---

## 0. How to run this mission

1. Read this `agent.md` fully.
2. Open and read `followup.md`.
3. Work on **the first unchecked item** in `followup.md`.
4. When you finish an item:
   - Mark it `[x]` in `followup.md`.
   - Add a short note if useful (what changed, key design choices).
5. Repeat from step 3 until all items are checked or clearly blocked.

---

## 1. Context

ZeMosaic has a **Grid / Survey mode** driven by `stack_plan.csv`:

- `zemosaic_worker.run_hierarchical_mosaic()`:
  - Detects Grid mode.
  - Calls `grid_mode.run_grid_mode(...)`.
  - If Grid mode throws, it logs an error and **falls back to the classic pipeline**.

- `grid_mode.run_grid_mode()`:
  - Reads `stack_plan.csv` and configuration from disk.
  - Stacks tiles into “master tiles”.
  - Calls `assemble_tiles(...)` to build the final mosaic.
  - If `assemble_tiles(...)` returns `None`, it raises `RuntimeError("Grid mode failed during assembly")`.

### Problems observed by the user

1. **Logging is almost silent for Grid mode**
   - `grid_mode.py` uses a logger `logging.getLogger("zemosaic.grid_mode")` plus `NullHandler`.
   - Result: all `_emit("[GRID] ...")` logs are swallowed and **do not appear** in `zemosaic_worker.log`.
   - When Grid mode fails, the user only sees:
     - `[GRID] Grid/Survey mode failed, continuing with classic pipeline`
     - plus the final `RuntimeError`, but **no detail** on *why* assembly failed.

2. **Assembly error handling is too opaque**
   - `assemble_tiles()` already tries to be robust:
     - Skips tiles that can’t be read (I/O errors).
     - Skips tiles with empty valid masks.
     - Only returns `None` when **no tile is usable** or when **no pixels are valid** even after the “salvage” attempt.
   - BUT, since the logger is effectively muted, the user doesn’t see:
     - “No tiles to assemble…”
     - “Unable to read any tile…”
     - “salvage assembly failed…”
   - From the user’s perspective, Grid mode just “fails during assembly” with no explanation.

3. **RGB equalization parity is unclear**
   - Classic pipeline:
     - Uses `poststack_equalize_rgb` / `equalize_rgb_medians_inplace(...)` in `zemosaic_align_stack.py`.
   - Grid mode:
     - Has `grid_post_equalize_rgb(...)` in `grid_mode.py`.
     - It already tries to reuse `equalize_rgb_medians_inplace` when available.
   - The user sees a **red shift in the final Grid mosaic** (not in master tiles) and is not sure Grid mode is actually using the same logic as the classic pipeline, or whether the `grid_rgb_equalize` flag is being respected like `poststack_equalize_rgb`.

---

## 2. Goals

### 2.1 Logging: make Grid mode transparent and debuggable

- All important Grid mode events must be visible in the **same log** as the rest of ZeMosaic (`ZeMosaicWorker` logger).
- In particular, the user must see (with a `[GRID]` prefix):

  - When Grid mode is detected/activated.
  - The **effective value** of `grid_rgb_equalize` and its source (**config vs parameter vs default**).
  - A summary of stacking/assembly (number of tiles, failures, skips).
  - Clear reasons when `assemble_tiles()` returns `None`:
    - No tiles found on disk.
    - All tiles unreadable.
    - All tiles fully masked / no valid pixels even after salvage.
  - When RGB equalization is applied or skipped, and why.

### 2.2 Robust assembly: skip bad tiles, avoid unnecessary hard failure

- Keep the current global behaviour (if **nothing** usable is left → assembly fails → Grid mode raises → worker falls back to classic pipeline).
- Within `assemble_tiles()`:
  - **Skip** tiles that fail I/O or have invalid masks or channel mismatch.
  - **Only** give up if **no usable tile** remains or if, after salvage, there are still no valid pixels.
- Make sure the logs explicitly describe:

  - `io_failures`, channel mismatches, empty masks.
  - The number of tiles kept.
  - When salvage mode is entered and its outcome.

### 2.3 RGB equalization parity with classic pipeline

- Ensure Grid mode uses **the same RGB equalization logic** as the classic pipeline:

  - Reuse `equalize_rgb_medians_inplace(...)` where possible.
  - If it’s not available, emulate the same behaviour (median per channel, same target definition, similar gain computation).

- The **effective flag** controlling RGB equalization in Grid mode must:

  - Be clearly logged (effective value + source).
  - Map sensibly to the existing configuration, so a user who toggles RGB equalization in the GUI gets **consistent behaviour** in classic and Grid modes.

- Add a small test to show that, given the same RGB mosaic:

  - Applying Grid mode’s `grid_post_equalize_rgb(...)` and applying the classic `equalize_rgb_medians_inplace(...)` lead to **equivalent medians / gains** within a reasonable tolerance.

---

## 3. Scope and non-goals

**In scope**

- `grid_mode.py`:
  - Logging wiring.
  - `_emit(...)` behaviour.
  - `assemble_tiles(...)` messages and robustness.
  - `grid_post_equalize_rgb(...)` and `grid_rgb_equalize` handling.

- `zemosaic_worker.py`:
  - Logging around Grid mode invocation.
  - Passing the RGB equalization flag / config to `grid_mode.run_grid_mode(...)` in a clear, explicit way if needed.
  - Ensuring Grid mode failure is logged cleanly before falling back to classic pipeline.

- Tests:
  - New unit-style tests in `tests/` targeting:
    - Logging behaviour (at least smoke-level).
    - Assembly robustness when some tiles are broken.
    - RGB equalization parity.

**Out of scope**

- No changes to the **GUI**:
  - Don’t add or remove GUI controls or tabs for Grid mode or RGB equalization.
- No changes to the **overall hierarchical pipeline logic** outside the Grid branch:
  - Do not alter clustering, ASTAP calls, or classic stacking behaviour.
- No changes to `stack_plan.csv` format.

---

## 4. Files to inspect and possibly modify

> File paths are relative to the ZeMosaic project root.

- `grid_mode.py`
  - Logger setup (`logger = logging.getLogger("zemosaic.grid_mode")` + `NullHandler`).
  - `_emit(...)` helper.
  - `run_grid_mode(...)`.
  - `assemble_tiles(...)`.
  - `grid_post_equalize_rgb(...)`.

- `zemosaic_worker.py`
  - `run_hierarchical_mosaic(...)`:
    - The branch that detects Grid mode and calls `grid_mode.run_grid_mode(...)`.
    - Logging around this branch.

- (If relevant / already present) configuration helpers:
  - `zemosaic_config.py` or any module that carries `grid_rgb_equalize` / `poststack_equalize_rgb` config to the worker.

- Tests (create if missing):
  - `tests/test_grid_mode_logging.py`
  - `tests/test_grid_mode_rgb_equalize.py`
  - Or a single `tests/test_grid_mode.py` with multiple test functions.

---

## 5. Detailed tasks

### Task A – Wire Grid mode logging into `ZeMosaicWorker` logger

**Goal:** ensure `_emit(...)` writes to the same place as the main worker logger, and keep the `[GRID]` prefix convention.

1. In `grid_mode.py`, update the logger setup:

   - **Current** (approx):

     ```python
     logger = logging.getLogger("zemosaic.grid_mode")
     if not logger.handlers:
         logger.addHandler(logging.NullHandler())
     ```

   - **Target behaviour**:

     - Use the **same logger name** as the worker, e.g. `"ZeMosaicWorker"`, OR
     - Keep `"zemosaic.grid_mode"` but do **not** attach a `NullHandler` so messages propagate up to the root logger configured by the worker.

   - Make sure `_emit(...)` always uses this logger and, if not already the case, prepends `"[GRID]"` to messages for easy filtering.

2. Verify that `run_hierarchical_mosaic(...)` in `zemosaic_worker.py` **does not** create a conflicting handler that could silence child loggers.

3. Add/adjust a smoke test (e.g. `tests/test_grid_mode_logging.py`) that:

   - Creates a temporary logger with a `StringIO` handler.
   - Temporarily sets `logging.getLogger("ZeMosaicWorker")` (or `"zemosaic.grid_mode"`, depending on your choice) to use that handler.
   - Calls a small piece of Grid code that uses `_emit("some message")`.
   - Asserts that the handler captured a line containing `"[GRID]"` and the expected message.

### Task B – Make assembly logging explicit and robust

**Goal:** provide detailed explanations in the logs for all assembly outcomes.

1. In `assemble_tiles(...)`:

   - Ensure there are explicit `_emit(...)` calls at key decision points:

     - When deriving `tiles_list` from `tiles_seq`:
       - If `tiles_list` is empty:
         - Log something like:
           - `"[GRID] No tiles to assemble (attempted=N, len(tiles_list)=0). Sample output paths: ..."`
         - Return `None` (this will lead to Grid mode raising).

     - Each time a tile fails I/O:
       - `"[GRID] Assembly: failed to read {t.output_path} ({exc})"`

     - Each time a tile is skipped due to channel mismatch:
       - `"[GRID] Assembly: tile {t.tile_id} skipped due to channel mismatch..."`

     - Each time a tile has empty valid mask:
       - `"[GRID] Assembly: tile {t.tile_id} has empty valid-mask, skipping"`

     - When we finally have `len(tile_infos)` usable tiles:
       - `"[GRID] Assembly: {len(tile_infos)} tiles kept (io_fail=..., channel_mismatch=..., empty_mask=...)"`

2. For the “no tile infos” case (all tiles failed / skipped):

   - Keep returning `None`.
   - Make the log message **very explicit** that the issue is with tile inputs, not some hidden crash.

3. For the “no valid pixels” / salvage path:

   - Before salvage:
     - Log:
       - `"[GRID] Assembly: no valid tile data written to mosaic, entering salvage mode..."`

   - At the end of salvage:
     - If salvage also fails (no pixels):
       - Log:
         - `"[GRID] Assembly: salvage assembly failed (no valid tile data after salvage)"`

   - This should end with `return None`, leaving `run_grid_mode()` to decide to raise.

4. On successful assembly:

   - After computing `mosaic` and *before* writing FITS:

     - Log shape and dtype:

       - `"[GRID] Assembly: mosaic built with shape={mosaic.shape}, dtype={mosaic.dtype}"`

   - After applying RGB equalization (if enabled) – see Task C.

### Task C – RGB equalization parity between Grid mode and classic pipeline

**Goal:** ensure Grid mode’s RGB equalization is coherent with `equalize_rgb_medians_inplace(...)` and clearly controlled/logged.

1. In `grid_post_equalize_rgb(...)` (already present in `grid_mode.py`):

   - Confirm that it:

     - Checks that `mosaic` is 3D with 3 channels.
     - Computes medians per channel on **valid pixels**.
     - Uses `equalize_rgb_medians_inplace(...)` when available.
     - Logs when equalization is:
       - Applied (with gains and medians).
       - Skipped (non-RGB mosaic, no valid pixels, missing channel, invalid target, error, etc.).

   - If the function currently mixes its own median-based gains with `equalize_rgb_medians_inplace`, **simplify** to:

     - Either:
       - Call `equalize_rgb_medians_inplace(...)` directly and log the gains it returns.
       - Or ensure that the logic is mathematically equivalent.

   - Keep the log message explicit, for example:

     - `"[GRID] RGB equalization: applied (reused classic poststack_equalize_rgb); gains=(...), medians=(...), target=..."`

2. Ensure the `grid_rgb_equalize` flag behaviour is coherent:

   - In `run_grid_mode(...)`, you already read configuration from disk with something like:

     ```python
     cfg_disk = _load_config_from_disk()
     rgb_source = "param"
     rgb_cfg = cfg_disk.get("grid_rgb_equalize", grid_rgb_equalize)
     if "grid_rgb_equalize" in cfg_disk:
         rgb_source = "config"
     ...
     grid_rgb_equalize = bool(...)
     ```

   - Keep or refine this precedence clearly:

     1. **Config / on-disk** `grid_rgb_equalize` (if key exists).
     2. Otherwise, the parameter `grid_rgb_equalize` passed from the worker.
     3. Fallback to a sensible default (`True`).

   - Emit a log line summarizing the final decision:

     - `"[GRID] RGB equalization: enabled={grid_rgb_equalize} (source=config|param|default)"`

3. In `assemble_tiles(...)`:

   - Around the call to `grid_post_equalize_rgb(...)`:

     - If `grid_rgb_equalize` is `True` and mosaic is RGB:

       - Log a DEBUG/INFO line **before** calling:

         - `"[GRID] RGB equalization: calling grid_post_equalize_rgb (shape=..., weight_shape=...)"`

     - If `grid_rgb_equalize` is `False` or mosaic is not RGB:

       - Log explicitly that RGB equalization is skipped and why.

4. In `zemosaic_worker.py` (Grid branch):

   - Ensure that when calling `grid_mode.run_grid_mode(...)`, you:

     - Have a clearly named variable for the flag you *intend* to use in Grid mode, e.g. `grid_rgb_equalize_flag`.
     - Log its value and source (config / GUI / default).
     - Pass it explicitly as `grid_rgb_equalize=grid_rgb_equalize_flag` to `run_grid_mode(...)`.

   - This does **not** have to override the on-disk config if you still want the on-disk config to win; but the semantics must be documented in comments.

5. Add a small test (e.g. in `tests/test_grid_mode_rgb_equalize.py`) that:

   - Creates a synthetic RGB mosaic `arr` (NumPy array).
   - Makes a copy (`arr_classic`) and applies `equalize_rgb_medians_inplace(...)`.
   - Makes another copy (`arr_grid`) and applies `grid_post_equalize_rgb(arr_grid, weight_sum=None, ...)`.
   - Checks that:

     - The medians per channel after classic and grid equalization are equal **within a small tolerance** (e.g. `1e-5`).
     - The signs of gains and general direction (balancing channels) are the same.

   - Skip the test if `equalize_rgb_medians_inplace` is not importable.

### Task D – Confirm worker behaviour and fallback

**Goal:** ensure the worker logs Grid failures clearly and then proceeds with classic pipeline without regression.

1. In `run_hierarchical_mosaic(...)`:

   - Make sure the Grid-branch looks like:

     - Detect Grid mode via `detect_grid_mode(...)`.
     - Log something like:
       - `"[GRID] Invoking grid_mode.run_grid_mode(...) with grid_rgb_equalize={...}, stack_norm='{...}', stack_weight='{...}', etc."`
     - Call `grid_mode.run_grid_mode(...)` in a `try` block.
     - On success:
       - Return early (Grid pipeline handled everything).
     - On exception:
       - Log an ERROR with `exc_info=True` and a clear message:
         - `"[GRID] Grid/Survey mode failed, continuing with classic pipeline"`
       - Then proceed into the classic pipeline path.

2. Ensure no behavioural changes for **non-grid** sessions.

3. (Optional but nice) Add a tiny functional test or test harness:

   - Using a very small artificial `input_folder` with a fake `stack_plan.csv` that references both valid and invalid tiles.
   - Ensure that:
     - Grid mode logs the errors for bad tiles.
     - Grid mode doesn’t crash due to a few broken tiles.
     - For the “all tiles broken” case, Grid mode fails and worker logs fallback to classic pipeline.

---

## 6. Acceptance criteria

The mission is **done** when:

1. **Logging**

   - Running a Grid-mode project produces in `zemosaic_worker.log`:
     - `[GRID]` messages:
       - Activation of Grid mode.
       - Summaries of tile assembly (tiles kept, io_failures, etc.).
       - Clear reasons when assembly returns `None`.
       - Explicit RGB equalization status and gains when applied.

2. **Assembly robustness**

   - Manually breaking one or a few tiles in `stack_plan.csv`:
     - Does **not** make Grid mode abort immediately.
     - Logs tile-level problems and continues with remaining tiles.
   - If **all** tiles are broken or invalid:
     - `assemble_tiles(...)` returns `None`.
     - `run_grid_mode(...)` raises `RuntimeError("Grid mode failed during assembly")`.
     - Worker logs the failure and runs the classic pipeline.

3. **RGB parity**

   - For a test RGB mosaic:
     - Grid mode’s `grid_post_equalize_rgb(...)` and classic `equalize_rgb_medians_inplace(...)` produce equivalent channel medians (within tolerance).
   - The log clearly shows:
     - Whether RGB equalization is enabled.
     - Where that decision came from (config vs param vs default).
     - The gains applied when equalization is active.

4. **No regressions**

   - Non-grid runs behave exactly as before.
   - Classic pipeline RGB equalization is unchanged.
   - Existing tests still pass, and new tests are added/passing.

---

## 7. Style and safety notes

- Keep the humour and comments already present in the codebase, but don’t sacrifice clarity.
- Avoid introducing new global state.
- Keep imports **local** where possible to limit side effects.
- Any new config keys should have:
  - A sensible default.
  - A short comment explaining their role.

---
**End of agent.md**
