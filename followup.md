# Follow-up checklist (DBE final mosaic)

## Meta / Process (MANDATORY)
- [x] `memory.md` updated with: changes, why, tests, limitations, next step

## Code review
- [x] GUI: checkbox stored in config key `final_mosaic_dbe_enabled` :contentReference[oaicite:3]{index=3}
- [x] Worker: reads `zconfig.final_mosaic_dbe_enabled` with default True :contentReference[oaicite:4]{index=4}
- [x] DBE hook placement is unambiguous:
  - [x] zemosaic_worker.py: Phase 6, immediately after `_dbg_rgb_stats("P6_PRE_EXPORT", ...)` :contentReference[oaicite:5]{index=5}
  - [x] Grid mode: either confirmed to pass through same save path OR explicit hook in grid_mode.py (no silent omission) :contentReference[oaicite:6]{index=6}
- [x] Per-channel application (no full HWC background buffer)
- [x] Uses `alpha_final` / coverage to avoid touching invalid mosaic areas :contentReference[oaicite:7]{index=7}
- [x] Header keywords written only if applied (ZMDBE, ZMDBE_DS, ZMDBE_K, ZMDBE_SIG)

## Manual tests
1) Classic small dataset (fast)
- [ ] Run with DBE ON/OFF, confirm only background changes.
- [ ] Confirm FITS saved + viewer FITS (if enabled) + ALPHA ext exists.

2) SDS mode
- [ ] Run SDS with DBE ON, ensure no crash and output looks sane.

3) Grid mode
- [ ] Run a small grid mode case:
  - [ ] either DBE executes (and logs) OR logs a clear “not applicable/bypassed” message
  - [ ] no regression in grid exports

## Logs to verify
- [ ] `P6_PRE_EXPORT` present :contentReference[oaicite:8]{index=8}
- [ ] `[DBE] ...` line present when enabled
- [ ] No ERROR; only WARN if DBE fails and run completes

## Performance sanity
- [ ] Downsample factor computed and reasonable (longest side <= 1024 in model space)
- [ ] No massive RAM spikes (check memory logs around Phase 6)
