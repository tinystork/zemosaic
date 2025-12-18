# Follow-up Checklist (Codex)

## 1) Locate the correct insertion point
- [x] In `zemosaic_filter_gui_qt.py`, find the pipeline section where initial clusters/groups are produced and optional splitting/merging happens; insert EQMODE split immediately after initial `groups` is built and before any merge/rebalance that might re-mix entries (e.g., `groups = [sub for g in groups for sub in split_group_by_eqmode(g)]`).

## 2) Implement header-only EQMODE read (fast)
- [x] Use astropy to read FITS header only (no image data), caching results if available; resolve path via entry keys (path/file/filename) and parse `EQMODE` safely, returning `"EQ"` for 1, `"ALTZ"` for 0, or `None` otherwise.

## 3) Keep orientation split intact
- [x] After EQMODE splitting, ensure the existing orientation split code still runs, at minimum for ALTZ groups (and optionally UNKNOWN) without restricting it to EQ only.

## 4) Ensure merges do NOT remix EQ/ALTZ
- [x] Guard later merge logic so groups from different EQMODE buckets are not merged; use a `group_eqmode_signature` helper returning `"EQ"` / `"ALTZ"` / `"MIXED"` / `"UNKNOWN"` and merge only when signatures match or both are UNKNOWN (apply only if there is a remix risk; keep surgical).

## 5) Logging
- [x] Add 1–2 log lines when a mixed group is detected and split, optionally including resulting group counts (e.g., `logger.info("eqmode_split: mixed group -> EQ=%d ALTZ=%d UNKNOWN=%d (splitting)", n_eq, n_altz, n_unk)`).

## 6) Manual test protocol (must be done)
- [ ] Dataset A: Seestar mixed EQMODE (0/1) that currently produces colored wedges. Expected: log shows split; resulting master tiles are coherent with no obvious “triangles”.
- [ ] Dataset B: Non-Seestar images (no EQMODE). Expected: no eqmode logs; same group count behavior as before (except existing orientation split).

## 7) Output
- [x] Return a git diff for `zemosaic_filter_gui_qt.py` only and a short note pointing to the insertion point plus which keys were used for file paths in entries.
