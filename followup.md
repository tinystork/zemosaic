# followup.md — What to report back after implementation

## 1) Summary of changes (bullet list)
- GUI: where exitcode is checked + how error is surfaced
- Worker: where exceptions are re-raised + guard preventing "finished" log on invalid outputs
- Any changes to PROCESS_ERROR emission (only if necessary)

## 2) Exact code locations
Provide file + function names + short snippet around the key modifications:
- `zemosaic_gui_qt.py`: `_on_listener_finished` (or equivalent)
- `zemosaic_worker.py`: `assemble_final_mosaic_reproject_coadd` (or equivalent)
- Top-level worker loop if modified

## 3) Logs from validation
Paste three log excerpts:

### A) Successful run (expected SUCCESS)
- Show last ~20 lines including the final SUCCESS line.

### B) Crash/kill simulation (expected ERROR)
- Either simulated exit or manual termination.
- Show GUI log line that indicates exitcode and that SUCCESS is absent.

### C) Phase 5 exception path (expected PROCESS_ERROR)
- Show `PROCESS_ERROR` surfacing in GUI log.
- Confirm no `assemble_info_finished_reproject_coadd` nor SUCCESS is printed.

## 4) Behavior checks
Answer yes/no:
- Cancel/Stop still shows “cancelled/stopped” (not crash).
- Nonzero exitcode now always yields error.
- No false positives on normal run.

## 5) Final output artifacts (optional but useful)
- If mosaic `.dat` / coverage `.dat` can remain partially created after crash, confirm GUI now reports failure anyway.
- If you added any extra guard that checks for completeness, describe it.

## 6) Provide the diff
Attach `git diff` output for the touched files only.
