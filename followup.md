# Follow-up Checklist (Codex)

## 1) Locate the correct insertion point
In `zemosaic_filter_gui_qt.py`, find the pipeline section where:
- initial clusters/groups are produced (list of groups)
- then optional splitting/merging happens (orientation split, merge small groups, etc.)

Insert EQMODE split:
- immediately after initial `groups` is built
- before any merge/rebalance that might re-mix entries

Implementation detail:
- `groups = [sub for g in groups for sub in split_group_by_eqmode(g)]`

## 2) Implement header-only EQMODE read (fast)
Preferred:
- use astropy to read FITS header only (no image data)
- cache result per file path if there’s an existing metadata cache dict (optional but nice)

Pseudo:
- path = entry.get("path") or entry.get("file") or entry["filename"] (use actual project key)
- header = fits.getheader(path, 0)
- v = header.get("EQMODE", None)
- parse int safely

Return values:
- "EQ" for 1
- "ALTZ" for 0
- None otherwise

## 3) Keep orientation split intact
After EQMODE splitting, ensure the existing orientation split code still runs.
Important:
- Don’t restrict orientation split to EQ only by mistake.
- At minimum, apply orientation split to ALTZ groups (and optionally UNKNOWN).

## 4) Ensure merges do NOT remix EQ/ALTZ
If there is logic later that merges small groups:
- add a guard: never merge groups of different eqmode buckets when EQMODE is known.
Minimal approach:
- When EQMODE split happened, resulting groups are already pure; ensure merge logic doesn’t ignore that.
If merge logic is aggressive, add a cheap check:
- `group_eqmode_signature(group)` -> returns "EQ" / "ALTZ" / "MIXED" / "UNKNOWN"
- only merge when signatures match or both UNKNOWN.

(Do this only if you observe an actual remix risk; keep it surgical.)

## 5) Logging
Add 1–2 log lines max:
- when mixed group detected and split
- optionally counts of resulting groups

Example:
`logger.info("eqmode_split: mixed group -> EQ=%d ALTZ=%d UNKNOWN=%d (splitting)", n_eq, n_altz, n_unk)`

## 6) Manual test protocol (must be done)
- Dataset A: Seestar mixed EQMODE (0/1) that currently produces colored wedges.
  Expected:
  - log shows split
  - resulting master tiles are coherent; no obvious “triangles”
- Dataset B: Non-Seestar images (no EQMODE)
  Expected:
  - no eqmode logs
  - same group count behavior as before (except existing orientation split)

## 7) Output
Return:
- a git diff for `zemosaic_filter_gui_qt.py` only
- short note pointing to the insertion point + which keys were used for file path in entries
