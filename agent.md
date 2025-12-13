# ZeMosaic — Borrowing v1 (coverage-first preplan) — Deterministic, no-gain

## Mission
Implement "Borrowing v1" as a *small deterministic post-process* on the auto-grouping result:
- **No “gain”**, no optimization, no changes to clustering rules.
- Only duplicate-assign some *border* images from a group `G` into a single best neighbor group `H`.
- Must be **reproducible** run-to-run (stable ordering).
- Must be **bounded** (quota) and must not create new UI controls.
- Must reuse the existing logging system and existing log level dropdown (Qt).

Scope:
- Apply borrowing only to **coverage-first preplan** outputs (the path that logs "Coverage-first preplan ready").
- Apply in both filter UIs:
  - `zemosaic_filter_gui.py` (Tk)
  - `zemosaic_filter_gui_qt.py` (Qt)

Non-goals:
- No “alpha” weighting, no adaptive quota, no multi-borrow to 2+ neighbors.
- No footprint/overlap geometry, no WCS intersection checks.
- No new logger, no new settings UI.

## Why
Some coverage-first groupings under-sample overlaps at group borders. Borrowing duplicates borderline frames into adjacent groups to improve overlap without changing clustering or downstream pipeline.

## Borrowing v1 — Rule (final)
Parameters (internal constants, no UI):
- `BORROW_ENABLE = True`
- `BORROW_MAX_NEIGHBOR_PER_IMAGE = 1` (enforced by "img borrowed at most once")
- `BORROW_IN_QUOTA_FRAC = 0.25`  (cap borrowed-in per destination group H, based on initial size)
- `BORROW_BORDER_FRAC = 0.20`    (border images are those within the outer 20% of the robust radius)
- `NEIGHBOR_K = 4`
- `BORROW_RADIUS_PCTL = 90`      (robust radius = Pctl of distances to group center)
- `BORROW_RADIUS_EPS = 1e-6`
- Anti-pollution guard: only borrow if `dist(img, center(H)) < dist(img, center(G))`

Required data per group `G` (computed from group members):
- `center(G)` = mean of member image centers (2D)
- `radius(G)` = percentile(BORROW_RADIUS_PCTL) of `dist(center(img), center(G))`
- `neighbors(G)` = K nearest groups by distance between `center(G)` and `center(H)`

Border image definition:
- `img` is border if `dist(center(img), center(G)) >= (1 - BORROW_BORDER_FRAC) * max(radius(G), BORROW_RADIUS_EPS)`

Borrowing:
For each group `G`:
- Take border images in deterministic order (sort by stable key, e.g. filepath string).
- For each border image `img`:
  - Choose `H` among `neighbors(G)` minimizing `dist(center(img), center(H))`.
  - Borrow `img` into `H` if all conditions hold:
    - `img` not already in `members(H)`
    - `borrowed_in_count(H) < ceil(BORROW_IN_QUOTA_FRAC * initial_size(H))`
    - `img` not already borrowed to any other group
    - Anti-pollution guard: `dist(img, center(H)) < dist(img, center(G))`
  - If OK: append `img` into `members(H)` and tag it `borrowed=True` (internal tag only).

Important:
- **Quota uses initial_size(H)**, not updated size.
- Borrowing is **duplicate assignment**; do not remove from source group.
- Stable ordering for groups and images.

## Integration points (surgical)
We do not touch downstream pipeline. We only mutate the auto-grouping payload just before it is applied/serialized into `preplan_master_groups`.

### Tk: `zemosaic_filter_gui.py`
Locate the auto-grouping code path that builds:
- `final_groups`
- `result_payload = {"final_groups": final_groups, ...}`
Insert borrowing *just before* `result_payload` is created (or right after `final_groups` is finalized and before applying result), but only when coverage-first preplan is the chosen mode.

### Qt: `zemosaic_filter_gui_qt.py`
Locate `_compute_auto_groups(...)` producing `final_groups` and `result_payload`, then `_apply_auto_group_result(...)`.
Insert borrowing in `_compute_auto_groups(...)` after final groups are computed and before the payload is returned/applied.
No new UI; reuse existing controls and logger.

## Logging (baseline + small extras)
Do NOT create a new logger. Reuse existing `logger` passed/available in these modules.
Only log when borrowing is enabled and executed.

Minimum baseline logs:
- Per destination group H: `initial_size`, `borrowed_in`, `final_size`
- Global totals: `borrowed_unique_images`, `borrowed_total_assignments`
- Top 5 sample: `img -> from_group -> to_group`

Add these low-cost extras:
- `border_candidate_images_total`
- `borrow_attempts_total`, `borrow_success_total`
- Borrow distance stats: mean and p90 of `dist(img, center(H))` for successful borrows

These logs must respect the existing log level dropdown in Qt; no new toggle.

## Data structures
Groups are lists of image identifiers/paths.
Use a stable key for deterministic ordering:
- Prefer full filepath string (normalized) or whatever the current grouping uses consistently.

Image center:
- Use existing per-image center/footprint center when available in the auto-grouping context.
- If only 2D XY centers are available, use them.
- If only RA/Dec is available, use those as 2D.

Do not introduce heavy dependencies.

## Tests / Acceptance
1) Run the filter twice on the same dataset:
- Borrowing outputs identical group compositions (deterministic).
2) Quota respected:
- For every group H: borrowed_in <= ceil(frac * initial_size(H)).
3) No duplicates inside a group.
4) Borrowed image assigned to at most 1 neighbor.
5) Anti-pollution guard prevents nonsensical borrows.
6) When borrowing disabled or not coverage-first: behavior is unchanged.

## Files to modify
- `zemosaic_filter_gui.py`
- `zemosaic_filter_gui_qt.py`
Optionally a small shared helper in `zemosaic_utils.py` (only if it reduces duplication cleanly).

## Deliverable
- Implement Borrowing v1 in both Tk and Qt paths for coverage-first preplan.
- Add baseline logs.
- No other refactors.
