# Follow-up — Borrowing v1 (coverage-first) — Implementation Checklist

## 0) Safety rails
- [x] Do NOT change clustering logic, only post-process `final_groups`.
- [x] Do NOT add UI toggles or settings.
- [x] Do NOT add a new logging system; reuse existing logger and Qt log-level dropdown behavior.
- [x] Keep deterministic ordering everywhere.

## 1) Locate "coverage-first preplan" auto-grouping output (Tk)
File: `zemosaic_filter_gui.py`
- [x] Find where `final_groups` is produced (after merge/sanity filters).
- [x] Identify the log line or boolean that indicates coverage-first preplan path (e.g. "Coverage-first preplan ready").
- [x] Insert borrowing call **after** `final_groups` is finalized and **before** `result_payload` is created / `_apply_result`.

Implementation note:
- Borrowing function should take `(final_groups, image_centers, logger)` and return new `final_groups` (or mutate in place, but be explicit).

## 2) Locate auto-grouping output (Qt)
File: `zemosaic_filter_gui_qt.py`
- [x] Find `_compute_auto_groups(...)` where `final_groups` and `result_payload` are created.
- [x] Apply borrowing **inside** `_compute_auto_groups(...)` after `final_groups` computed and before payload return.
- [x] Ensure logs honor current logger level (existing dropdown).

## 3) Implement the core borrowing function
Preferred: a small helper function in the same module (or `zemosaic_utils.py` if shared cleanly).

Function shape (suggested):
- `def apply_borrowing_v1(final_groups, image_centers, *, logger, neighbor_k=4, border_frac=0.20, quota_frac=0.25, radius_pctl=90, eps=1e-6):`
Return:
- updated `final_groups`
- stats dict for logging

### 3.1 Deterministic ordering
- [x] Sort groups by stable group id/index.
- [x] Sort images within a group by stable key (filepath string).
- [x] Ensure `neighbors(G)` computed from stable centers, tie-break by group id for equal distances.

### 3.2 Compute group centers and radii
For each group `G`:
- [x] `center(G)` = mean of member centers
- [x] distances = `dist(center(img), center(G))`
- [x] `radius(G)` = percentile(distances, radius_pctl), with `max(radius, eps)`
- [x] If group has <2 valid centers: handle gracefully
  - radius = eps
  - border set will be empty or based on trivial check (do not crash)

### 3.3 Compute K nearest neighbors
For each `G`:
- [x] `neighbors(G)` = K closest other groups by dist(center(G), center(H))
- [x] If fewer than K other groups: use all available.

### 3.4 Mark border images
- [x] border if `dist(img, center(G)) >= (1 - border_frac) * radius(G)`

### 3.5 Borrow loop
For each group `G` (stable order), for each border image `img` (stable order):
- [x] Choose `H` = argmin_{H in neighbors(G)} dist(img, center(H))
- [x] Conditions:
  - [x] img not already in H
  - [x] `borrowed_in_count(H) < ceil(quota_frac * initial_size(H))`
  - [x] img not already borrowed anywhere
  - [x] anti-pollution: `dist(img, center(H)) < dist(img, center(G))`
- [x] If ok: append to H and record mapping `img -> G -> H`

### 3.6 Stats for logs
Track:
- [x] per H: initial_size, borrowed_in, final_size
- [x] global: borrowed_unique_images, borrowed_total_assignments
- [x] global: border_candidate_images_total
- [x] attempts + successes
- [x] successful borrow distances: mean + p90
- [x] top 5 borrow examples

## 4) Logging requirements
- [x] Do not spam; single block summary is enough.
- [x] Must be gated: only when borrowing executed.
- [x] Use `logger.info` for summary; use `logger.debug` for top5 examples or extended stats if needed.

Suggested log format:
- "Borrowing v1: candidates=X attempts=Y success=Z unique_imgs=U assignments=A dist_mean=... dist_p90=..."
- Per group H line: "Borrowing v1 group H: initial=.. borrowed_in=.. final=.."
- Top5 lines: "Borrowing v1 sample: img=... from=G to=H"

## 5) No behavior change when not coverage-first
- [x] Ensure borrowing is only applied in coverage-first preplan path.
- [x] If that flag isn’t explicit, key off the same mode/branch that currently prints "Coverage-first preplan ready".

## 6) Validation steps
Run twice (same dataset, same settings):
- [ ] Group compositions identical (deterministic).
- [ ] No duplicate images within a single group.
- [ ] No image borrowed to > 1 neighbor.
- [ ] Quota respected for all groups.
- [ ] Borrowing block appears in logs and shows sensible counts.

## 7) Out of scope guard
- [x] Do not touch worker pipeline, stack_core, reprojection, normalization, or mosaic assembly.
- [x] Do not modify GPU/CPU paths.
- [x] Do not introduce “gain” or adaptive heuristics.

## 8) Completion criteria
- [x] Borrowing v1 active only for coverage-first preplan in Tk + Qt.
- [x] Logs baseline present.
- [x] Deterministic and bounded.
- [x] No UI changes.
