# Smoke protocol — Windows/macOS (intermediate refactor checkpoint)

Goal for this checkpoint (not full validation):
1. App launches.
2. A run starts without blocking.
3. Manual interruption is clean.

Dataset:
- Use the small `example/` dataset as agreed.

## Steps (Windows/macOS)

1. Start the app from the current branch build/runtime.
2. Load input/output folders from `example` test set.
3. Start a run.
4. Wait until you see clear processing activity (logs/progress moving, worker started).
5. Stop/cancel manually after startup confirmation (no need to run full 35 min).
6. Confirm clean interruption:
   - app remains responsive,
   - no crash/freeze,
   - no inconsistent fatal popup,
   - worker process stops.

## Pass criteria (checkpoint)

- PASS if launch + start + clean interrupt are confirmed.
- FAIL if launch crashes, run cannot start, or stop hangs/crashes.

## Report template

- Platform: Windows / macOS
- Launch: PASS/FAIL
- Run start: PASS/FAIL
- Clean interrupt: PASS/FAIL
- Notes/log excerpt (if FAIL):
