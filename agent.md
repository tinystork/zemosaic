# agent.md — Mission: Fix "silent success" when Phase 5 (Reproject) aborts (OOM/crash)

## Context (symptom)
On some runs (often Phase 5 Reproject), the worker process stops abruptly (likely OOM / OS kill / native crash).
The GUI then prints:
- `[INFO] assemble_info_finished_reproject_coadd`
- `[SUCCESS] Processing completed successfully.`
even though Phase 5 did NOT finish (tiles counter stops mid-way) and outputs may be partial.

Root causes:
1) Qt GUI decides success based on `_had_error` only. If worker dies without emitting `PROCESS_ERROR`, GUI shows SUCCESS.
2) Worker code may catch exceptions in Phase 5 and `return None` instead of raising, preventing `PROCESS_ERROR` propagation.

## Goal
Make the app **never claim SUCCESS** when the worker process terminated abnormally or when Phase 5 failed internally.

### Acceptance Criteria
- If worker exits with non-zero exit code (crash/kill/OOM), GUI must report an error (not SUCCESS), with a clear log line.
- If Phase 5 errors inside Python (MemoryError, BrokenProcessPool, etc.), the worker must emit `PROCESS_ERROR` and GUI must report failure.
- Success message must only be shown when the run truly completed and Phase 5 produced valid outputs.

## Constraints
- Minimal, surgical patch.
- No behavioral changes to normal successful runs.
- Do not redesign pipeline; only fix error reporting + propagation.
- Preserve existing cancel/stop behavior: do not misreport user-cancel as crash.

## Files in scope (expected)
- `zemosaic_gui_qt.py`  (Qt worker finalization / listener finished)
- `zemosaic_worker.py`  (Phase 5 exception handling + completion logging)

Avoid touching unrelated modules unless strictly required.

---

## Task A — GUI: detect abnormal worker termination via exit code
### Where
In `zemosaic_gui_qt.py`, class handling the process + listener, method similar to:
- `ZeMosaicQtWorker._on_listener_finished()` (or equivalent finalization hook)

### What to implement
- When listener finishes, read `self._process.exitcode`.
- If `exitcode` is not `0` (and not `None`), and run was not cancelled/stopped:
  - set `_had_error=True`
  - set `_last_error` to something explicit: `"Worker process terminated unexpectedly (exitcode=X). Likely OOM/crash."`
  - emit an ERROR log line into the GUI log (if there is a signal for that; keep minimal)
  - final `success` must become False.

### Important
- Do NOT flag as error if user requested stop/cancel (check `_stop_requested`, `_cancelled` or equivalents).
- On Windows, exitcode might be positive; on POSIX it could be negative (signal). Treat any nonzero as crash.

---

## Task B — Worker: Phase 5 must not "return None" on fatal errors
### Where
In `zemosaic_worker.py`, function:
- `assemble_final_mosaic_reproject_coadd(...)` (or same role)

### What to implement
1) If the internal Phase 5 call fails, do NOT `return None, None, None`.
   - Replace with `raise` after logging.
2) Add a guard before printing / emitting:
   - `"assemble_info_finished_reproject_coadd"`
   If mosaic/coverage is missing (`None`) or clearly incomplete, raise a `RuntimeError`.

Rationale: exceptions must propagate to the top-level worker run loop where `PROCESS_ERROR` is emitted.

---

## Task C — Ensure top-level worker emits PROCESS_ERROR on uncaught exceptions
### Where
In the main worker entry/run method (often `run()` in the worker process).

### What to implement (only if missing)
- Ensure there is a `try/except Exception as e:` around the full processing pipeline that emits:
  - `PROCESS_ERROR` (payload includes `error=str(e)` + maybe `traceback`)
  - then exits with non-zero status (or just lets exception kill process; GUI will catch via exitcode anyway)

If this mechanism already exists, do not change it.

---

## Task D — Logging correctness
- Ensure `[SUCCESS] Processing completed successfully.` is only emitted when `success=True`.
- If run failed, show a single clear `[ERROR]` line in GUI log:
  - either from `PROCESS_ERROR` payload OR from `exitcode` detection.

---

## Manual test plan (must be done)
1) **Normal success run**: verify unchanged behavior; GUI shows SUCCESS.
2) **Simulated crash** (lightweight):
   - Add a temporary dev-only code path OR a tiny internal test hook (if one already exists) that forces the worker to `os._exit(137)` mid-way (DO NOT ship this hook enabled by default).
   - Confirm GUI reports error and does not show SUCCESS.
3) **Simulated Phase 5 exception**:
   - Force a `MemoryError` / raise `RuntimeError("test")` inside Phase 5 code path (dev-only, disabled by default).
   - Confirm `PROCESS_ERROR` is shown; GUI shows error.

If adding test hooks is too invasive, skip code hooks and provide clear instructions how to simulate by manually killing the worker process while running; GUI must show error (exitcode != 0).

---

## Deliverables
- Patch in the two files above.
- Brief note in code comments explaining why exitcode detection is required (silent kill cases).
- Keep diff small and readable.

## Definition of Done
- Abnormal termination never results in SUCCESS.
- Internal Phase 5 failure never results in silent completion.
- No regressions on normal runs / cancel runs.
