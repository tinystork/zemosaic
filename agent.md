# Mission: Prevent mixing Seestar ALTZ and EQ frames inside the same master_tile groups (Qt filter stage)

## Context / Problem
Some datasets contain a mix of Seestar frames captured in Alt-Az mode and Equatorial mode.
When these get clustered together (same sky location), the frames have incompatible orientation/field rotation
and stacking produces severe artifacts (large colored wedges, gradients, etc.).

Seestar FITS headers contain:
- EQMODE = 1  -> Equatorial mode
- EQMODE = 0  -> Alt-Az mode

We already have an orientation-based split mechanism in the Qt filter GUI which is still useful,
especially for Alt-Az clusters.

## Goal (Transparent to user)
Inside the Qt filter stage (before master_tiles are built), automatically split any cluster/group that mixes
Seestar EQ and ALTZ frames, using the FITS header keyword EQMODE when available.

Fallback behavior:
- If EQMODE is not present (non-Seestar images), do NOT change behavior: rely on existing orientation split only.

Keep existing orientation split logic intact and still applied for ALTZ clusters (and optionally UNKNOWN).

## Scope / Constraints
- **Modify only one file**: `zemosaic_filter_gui_qt.py`
- No refactor, no new UI controls required.
- Preserve existing pipeline behavior for non-Seestar datasets.
- Add clear, minimal logging so users can understand when a split happens.
- Avoid heavy IO: prefer header-only reads (Astropy header read) where possible.

## Implementation Plan (Surgical)
1) Add a small helper to extract EQMODE from an entry:
   - Return "EQ" if EQMODE==1, "ALTZ" if EQMODE==0.
   - Return None if absent/unreadable.

2) Add a group splitter:
   - Input: `group: list[dict]` (entries/files)
   - Partition into buckets: EQ / ALTZ / UNKNOWN (UNKNOWN = no EQMODE)
   - If group contains BOTH EQ and ALTZ:
       - Emit a log line: `eqmode_split: group mixed (EQ=%d ALTZ=%d UNKNOWN=%d) -> split`
       - Return up to 3 groups (skip empty buckets)
     Else:
       - Return [group] unchanged.

3) Insert this split at the correct stage:
   - After initial cluster/group formation
   - Before any "merge small groups" or "group rebalance" logic that could remix frames.
   - Ensure downstream code processes the flattened list of groups.

4) Keep orientation split:
   - After eqmode split, run existing orientation split logic as currently done.
   - Ensure it still triggers for ALTZ groups (and optionally UNKNOWN).

## Acceptance Criteria
A) Mixed dataset (Seestar only) with EQMODE 0 and 1 frames near same sky location:
   - No master_tile group contains a mix of EQMODE=0 and EQMODE=1 frames.
   - Log shows at least one `eqmode_split:` line.

B) Non-Seestar dataset (no EQMODE keyword):
   - No new splitting occurs because of EQMODE (behavior unchanged).
   - Existing orientation split continues to work as before.

C) Alt-Az only Seestar dataset (EQMODE=0):
   - Orientation split behavior remains active and unchanged.

D) No regressions:
   - Pipeline completes; no crashes due to missing headers, unreadable FITS, or unexpected EQMODE values.

## Notes
- Treat EQMODE as authoritative when present.
- Do not attempt to infer mount mode from other keywords.
- Keep changes minimal: small helper functions + one insertion point + logs.
