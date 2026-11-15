# ZM-LAYOUT-PRODUCT-20261115 — Windows A/B protocol (ZeAlfie → ZeMosaic)

Status: **PROTOCOL ONLY — not executed by Coco, not executed by Junior.**
Coco does not launch Windows runs, does not update the runtime, does not touch
the M33 run and does not copy the corpus.

Rework-1 (Nono M1/M2/M3) corrections are applied below; the affected statements
are marked **[M1]**, **[M2]**, **[M3]**.

## 0. Authorization and ownership

| step | owner |
|---|---|
| local implementation, parity tests, local probe evidence | Coco (this mission) |
| independent verification | Junior |
| code review | Nono |
| merge / version bump / beta publication | Junior (Tristan has authorized `origin/beta`) |
| **candidate Windows run through ZeAlfie → ZeMosaic** | **Tristan launches it himself, after beta publication** |
| comparison of the two output trees | Junior (read-only), never an automatic launch |
| acceptance | Junior, then Tristan |

## 1. Artifacts supplied by Tristan (2026-11-15)

| role | path | rule |
|---|---|---|
| reference (beta 5.1.4 baseline) output | `D:\SH2-101_mosaic_sub\out\newgauge` | **READ-ONLY**, never modified, never re-run into |
| candidate output (run launched by Tristan) | `D:\SH2-101_mosaic_sub\out\newgaugev2` | intentionally a DISTINCT output tree |
| input corpus | `D:\SH2-101_mosaic_sub\lights` | immutable; read-only listing verified 3188 files / 13 248 817 920 B (FITS count still to confirm) |
| M33 protection | `D:\M33\S50\out\fresh` | must not be disturbed; no concurrent run |

No output directory is written inside `lights`. `newgauge` is only read.

## 2. Preconditions (checked before anything else)

1. **NO run is active on `D:\M33\S50\out\fresh`.** The Windows A/B is strictly
   **serialized**: the M33 run must be finished (or stopped by Tristan) before a
   second engine launch of any kind.
2. **[M2] The worker log is GLOBAL and TRUNCATED.** `zemosaic_worker.py`
   resolves ONE per-user path — `%APPDATA%\ZeMosaic\logs\zemosaic_worker.log` —
   and opens it with `mode='w'` at import/startup. A second launch (M33 run, A/B
   run, or a CLI run) therefore **truncates** the previous run's worker log.
   Distinct output directories do NOT protect it. Consequences:
   - never run two engine instances at once (no M33 + A/B concurrency);
   - once the M33 run is idle, **archive** `%APPDATA%\ZeMosaic\logs\zemosaic_worker.log`
     (copy to a dated file) **before** launching any other engine/CLI, so the M33
     log is preserved as evidence;
   - after the baseline (`newgauge`) phase is over, archive its worker log the
     same way before the candidate run;
   - the logging implementation is NOT modified by this mission.
3. `newgauge` (reference) is complete and frozen: record its tree fingerprint
   (§5) BEFORE the candidate run so a later change is detectable.
4. The candidate build is the exact beta artifact Junior published; record the
   version string and the source commit.
5. Corpus fingerprint (file count + total bytes; `Get-ChildItem -Recurse`).
6. Disk space for the candidate output tree, outside `lights`.

## 3. Conditions (what must match, what intentionally differs)

**[M3] Must match — the effective SETTINGS, not the paths:**

- input folder = `D:\SH2-101_mosaic_sub\lights`;
- cell layout policy (auto/pinned) and, when pinned, the exact `NXxNY`;
- halo width, photometric gauge mode, weighting, normalization, rejection
  options, storage profile, workers, GPU on/off and the GPU tile policy;
- **an explicit effective RAM budget**. In `auto` the chosen grid depends on the
  RAM available at that moment, so two runs at different times may legitimately
  differ. The comparison is CONTROLLED only with the same explicit effective
  budget (bytes), recorded on both sides;
- the same launcher build (ZeAlfie); only the ZeMosaic engine build differs.

**[M3] Intentionally DIFFERENT — recorded, never treated as a mismatch:**

- output folder (`newgauge` reference vs `newgaugev2` candidate), run ids, run
  timestamps, absolute paths, provenance/path-echoed fields, worker log path,
  timings, and additive diagnostics (e.g. the `layout index:` lines introduced by
  this mission, which only appear when an index was built).

### Budget recovery rule (no re-running the reference)

The baseline `newgauge` tree is read-only and must NOT be re-run or overwritten.

1. If the reference run recorded an explicit effective budget (its manifest /
   geometry artifact `layout.ram_budget_bytes`), use exactly that value for the
   candidate run.
2. If the reference ran in `auto` (no explicit budget), the recorded
   `layout.ram_budget_bytes` (if any) is the value to pin on the candidate when
   the candidate launcher allows it.
3. If the effective budget cannot be recovered or pinned, the comparison is
   **observational/confounded**, NOT a controlled A/B: report it as such, and do
   NOT declare a parity failure because the grids differ.
4. Missing baseline settings (same rule): if an invariant of §3 cannot be
   recovered from the reference tree, the comparison is observational for that
   dimension; do not modify, re-run or overwrite `newgauge`.

## 4. Evidence to capture per run

1. **Decision log lines** — the whole `layout scan:` sequence
   (`candidate NxM ACCEPTED/REJECTED`, `CHOSEN NxM …`, `layout rescue:` if any)
   and the whole `layout load:` sequence when the load pass engages. **[M3]**
   `layout index:` lines are diagnostic only: compare them separately (they
   exist only when an index was built) and never as decision lines.
2. **Frozen geometry artifacts** — the per-cell predictions in the geometry /
   manifest (`cells`: `cell_id`, `n`, `patch_area`, `predicted_*`) and the
   `layout` block (`nx`, `ny`, `predicted_bound_bytes`, `max_contributors`,
   `cpu_streaming.selected_tile_size`, `cpu_streaming.trials`, `layout_origin`,
   `membership_load` if present).
3. **Timings** — layout-phase wall time (log timestamps) and total run time;
   per-cell `peak_rss_kib` if the engine persisted it.
4. **Science spot-check** — a handful of Cells (first, heaviest by `n`, one
   empty/halo-only): identical frame membership, identical `n_contributors`,
   identical reference frame.

## 5. Comparison procedure (Junior, read-only)

Per output tree, compute and compare — **[M3] field by field, never a whole-file
manifest equality** (provenance, paths, timings and additive diagnostics differ
by design):

- normalized layout log lines (timestamps stripped): SHA-256;
- geometry/manifest `layout` block and `cells` block: canonical JSON (sorted
  keys), SHA-256 and field-by-field equality;
- chosen `(nx, ny)`, `max_contributors`, `selected_tile_size`, `layout_origin`;
- per-cell `n` vector in row-major order: element-wise equality;
- wall time and RSS: reported as-is (single runs are indicative, not medians);
- **separately**: `layout index:` diagnostic lines (presence/absence and
  counters), absolute paths, run ids, timings — expected to differ, never a
  failure.

**Pass criteria** — identical chosen grid; identical `cells` block (including
`n` per cell); identical `layout` block; identical `cpu_streaming` decision;
identical decision log lines; identical science spot-check. Timing/RSS/`layout
index:` differences are *reported*, not required.

**Fail criteria** — any difference in the chosen grid, a per-cell `n`, the
`layout` block or the decision log lines, PROVIDED the §3 invariants (including
an explicit identical effective budget) were actually in force. Otherwise the run
is observational (§3.3) and cannot be scored as a parity failure.

## 6. No decision-only pre-check exists **[M1]**

There is **no** decision-only entry point in this product: `python -m
zemosaic.zegrid_cli` calls `run_zegrid_mode(...)`, i.e. the FULL engine
(stacking included), and writes an output tree at the same cost as a product
run. Do **not** use it as a "cheap pre-check", do not add such a mode in this
mission, and never launch it while M33 is active (worker-log truncation, §2).

## 7. What is already proven locally (so the Windows run is a confirmation)

- Full-dict decision hashes identical between `a59e72c` and the candidate on the
  real M106 corpus (3 budgets) and a synthetic 2000-frame corpus (2 budgets);
- 10 reachable chooser paths (auto, tight budget, pinned, rank window, coarser
  rescue, advisory halo, bounded/joint, external fallback, refusal, load-aware)
  produce identical decision dictionaries AND identical decision log lines with
  the exact index ON and OFF (only the dedicated `layout index:` diagnostics
  differ — they are not decision lines and are not in the layout dict);
- the geometry-only patch is the same rectangle as a real `build_patch`
  (bounds, core slice, `(H, W)`), and the scan never serializes a WCS header;
- the index is a bounded one-off cost (≈ 4.6 MiB / 0.09 s at 21 818 polygons);
  the retained per-Cell reuse list is ≈ 1.28 MB at 2613 Cells.

See `docs/qualification/zm-layout-product-20261115/README.md` and the mission
report `.a2a-reports/ZM-LAYOUT-PRODUCT-20261115.coco.r1.md`.
