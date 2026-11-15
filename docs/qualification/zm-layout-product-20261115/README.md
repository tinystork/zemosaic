# ZM-LAYOUT-PRODUCT-20261115 — phase 2 product integration evidence

Mission: integrate the **phase 2** layout-scan acceleration into the REAL
mode-aware chooser, with parity first and no science/decision change.
Document revised in rework-1 (Nono review-0: M1/M2/M3 in the Windows protocol,
L1–L4 in measurement/observability).

Bounded scope of the change (product, `src/`):

1. **Geometry-only candidate scan.** The candidate scan needs a Cell's patch
   *rectangle* (membership) and its `(H, W)` (streaming bound). It used to build
   a full `ProcessingPatch` per Cell per candidate — a deep-copied WCS plus a
   serialized FITS header — and read neither. The rectangle now comes from
   `geometry.patch_geometry()` (integer arithmetic), which is ALSO the single
   source of truth used by `geometry.build_patch()`, so the two cannot disagree.
2. **Exact footprint index, built once per layout choice.** `FootprintIndex`
   (`auto_layout`) prefilters the contributor predicate by bounding box; the
   production predicate `poly.intersection(rect).area > INTERSECTION_AREA_EPS`
   still counts. It is optional (small corpora keep the full sweep), falls back
   to a pure-Python `sorted` backend when Shapely < 2 / no `STRtree`, and any
   failing query falls back to the full exact sweep.
3. **Chosen-scan reuse.** The per-Cell dicts produced by the scan that admitted
   the chosen grid (or by the load pass's adopted candidate) are reused instead
   of re-running the identical pass.
4. **Index observability (rework-1 L1).** The index state is published in the
   REAL run log under its own prefix `layout index:` (backend, polygons,
   indexed, uncertain, queries, query_failures, build time, fallback reason), at
   construction AND at every exit of the choice (completion or refusal), WARN
   when degraded. These lines are NOT `layout scan:` / `layout load:` decision
   lines and are NOT part of the layout dict.

**Explicitly deferred (not implemented): all-modes early rejection ("(B)")**
of the qualification prototype. A sound certificate has to be exact for the
in-memory mode, EVERY CPU streaming ladder tile, the bounded profile and the
joint aggregate, and it must not perturb the exact numeric diagnostics that the
refusal messages publish (the "finest candidate" refusal prints the exact
`inmem_worst_bytes`, `stream_worst_by_tile[MIN]` and cheaper bound of that very
candidate). A partial certificate would either be unsound or would change the
terminal diagnostics of the refusal path. Deferring it is the bounded choice;
the qualification numbers for (B) remain conditional and are not used here.

## Files

| file | role |
|---|---|
| `probe_product_scan.py` | bounded A/B probe of the REAL chooser (wall medians, post-call RSS maximum, decision hash, index build/native cost, retained per-Cell cost) |
| `evidence/baseline.json` | probe output in a pristine `a59e72c` worktree |
| `evidence/candidate.json` | probe output in the candidate tree |
| `evidence/retained_cells.json` | retained per-Cell object cost at the largest qualified grid (67x39) |

Product tests: `tests/test_zegrid_layout_product_20261115.py` (62 tests).

```bash
cd /home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources
PYTHONPATH=src pytest -q -p no:cacheprovider tests/test_zegrid_layout_product_20261115.py
PYTHONPATH=src python docs/qualification/zm-layout-product-20261115/probe_product_scan.py \
    --out docs/qualification/zm-layout-product-20261115/evidence/candidate.json
PYTHONPATH=src python docs/qualification/zm-layout-product-20261115/probe_product_scan.py \
    --only-retained --out docs/qualification/zm-layout-product-20261115/evidence/retained_cells.json
```

The baseline evidence was produced by the SAME script file executed with the
baseline tree's `PYTHONPATH` (exact commands in the mission report).

## Decision parity (highest-value result)

The probe hashes the FULL returned layout dict (cells included). Baseline and
candidate produce the **same SHA-256 for every case**, i.e. identical chosen
grid, identical per-Cell `n`/`patch_area`/`patch_hw`, identical in-memory and
streaming bounds, identical selected tile, identical floors block, identical
load provenance:

| case | chosen | signature (both trees) |
|---|---|---|
| M106, no budget | 3x2 | `b67ac406b1fa…` |
| M106, 1 GiB | 8x6 | `29c9ba4c2dd6…` |
| M106, 2 GiB | 4x3 | `3ffd4a48331c…` |
| dense 2000 frames, 1 GiB | 6x7 | `364dd41930b5…` |
| dense 2000 frames, 512 MiB | 21x23 | `b2f840f3fb87…` |

**10 parity paths** are compared on the REAL chooser with the index ON versus
OFF — auto, tight budget, pinned, refinement-rank window, coarser rescue,
advisory explicit halo, bounded/joint, external fallback, infeasible refusal and
the load-aware pass. Each compares the full layout dict, the refusal payload and
the DECISION log lines verbatim; the dedicated `layout index:` diagnostics are
excluded from that comparison by design (they exist only when an index was
built) and are asserted separately.

## Measured wall time (median, same host, same script, read-only M106)

| case | baseline `a59e72c` | candidate | ratio | repeats |
|---|---:|---:|---:|---:|
| M106, no budget | 0.2101 s | 0.1885 s | 1.11x | 5 |
| M106, 1 GiB | 1.6054 s | 0.6139 s | 2.62x | 5 |
| M106, 2 GiB | 0.3534 s | 0.2399 s | 1.47x | 5 |
| dense 2000, 1 GiB | 6.8702 s | 6.4586 s | 1.06x | 3 |
| dense 2000, 512 MiB | 19.4983 s | 10.9406 s | 1.78x | 3 |

Read this honestly:

* these are LOCAL Linux medians on synthetic + one real 66-frame corpus; no
  Windows, no M33, no ZeAlfie path was exercised here;
* the 512 MiB dense case is scan-bound (21x23 = 483 Cells x 2000 frames x
  several candidates): 1.78x;
* the dense 1 GiB case is **projection-bound**: `cProfile` attributes 16.9 s of
  its 17.2 s to `project_footprints` (one WCS pass per frame), which this
  mission does not touch. Its 1.06x is the scan part only, and there is NO
  universal speedup claim;
* `count` work excluded the index query before this mission; it is now included,
  so the wall numbers are the real end-to-end choice.

## Memory: what is actually measured (rework-1 L2)

`rss_after_bytes_max` is the **maximum of RSS samples taken immediately AFTER
each complete chooser call** — NOT a transient peak during the scan (no
intra-scan sampler is installed). The key was renamed from `rss_peak_bytes`
in rework-1 with the SAME numbers; the corrected claim is "post-call RSS is
unchanged within noise on every case", not "peak RSS unchanged".

| case | baseline post-call RSS max | candidate post-call RSS max |
|---|---:|---:|
| M106, no budget | 175 MiB | 175 MiB |
| M106, 1 GiB | 180 MiB | 179 MiB |
| M106, 2 GiB | 183 MiB | 182 MiB |
| dense 2000, 1 GiB | 280 MiB | 280 MiB |
| dense 2000, 512 MiB | 337 MiB | 334 MiB |

### Index cost (native, not tracemalloc)

| corpus | backend | build | RSS delta | Python structures |
|---|---|---:|---:|---:|
| real M106 (66) | STRtree | 0.30 ms | +8 KiB | 3.4 KiB |
| real M106 (66) | sorted | 0.24 ms | 0 | 4.6 KiB |
| synthetic 1024 | STRtree | 4.1 ms | 0 | 43 KiB |
| synthetic 1024 | sorted | 3.8 ms | 0 | 60 KiB |
| synthetic 21818 | STRtree | 87.6 ms | +4.6 MiB | 0.97 MiB |
| synthetic 21818 | sorted | 82.2 ms | +3.0 MiB | 1.34 MiB |

At the historical M33 scale (21 818 footprints) the index is a one-off ~4.6 MiB
and ~0.09 s.

### Retained per-Cell list (rework-1 L4)

The chose-scan reuse retains ONE candidate's per-Cell dicts (the forward scan's
last scanned grid, or the load pass's adopted grid): **O(cells)** dictionaries.
Measured on the largest existing qualified grid (M106 canvas, 67x39 = **2 613
Cells**), with a deep `sys.getsizeof` over the retained list plus the
`dict(c)` worst-cell copy:

| corpus | grid | cells | retained Python bytes | bytes/cell |
|---|---|---:|---:|---:|
| real M106 (66 frames) | 67x39 | 2 613 | **1 278 296 B ≈ 1.22 MiB** | 489 |
| synthetic dense 2000 | 67x39 | 2 613 | 1 279 416 B ≈ 1.22 MiB | 490 |

Scope: Python objects only (dicts, str keys, ints, one `list` per Cell); this is
NOT RSS and NOT a transient peak. Extrapolation (formula, not a measurement): a
200x200 grid (40 000 Cells) would retain ≈ 19.6 MB.

## Exactness / soundness contracts

* `query(rect)` returns a **superset** of the polygons whose
  `intersection(rect).area` is positive (a positive-area intersection implies
  overlapping bounding boxes). The production predicate decides every count.
* Polygons with **no usable bounding box** (empty, non-finite bounds) and
  polygons with **non-finite coordinates** are excluded from the tree and
  re-checked by the exact predicate on EVERY query. A NaN footprint still
  raises the same `GEOSException` as before. (`Shapely.bounds` silently drops a
  NaN vertex, so the bounding box alone is NOT a sound criterion: this is why
  coordinate finiteness is classified explicitly, vectorized via
  `shapely.get_coordinates` — ~8 ms at 21 818 polygons.)
* `FootprintIndex.count()` retries the historical full sweep in ascending
  polygon order if any geometry operation raises.
* **Scope of that guarantee (rework-1 L3):** for a FINITE-but-INVALID polygon
  (e.g. a bowtie) whose bbox does NOT overlap the rect, the indexed path
  evaluates no overlay for it. With the GEOS in use this is outcome-equivalent
  (a bbox-disjoint overlay short-circuits to an empty geometry instead of
  raising) and both paths count it as 0; when the bbox DOES overlap, both paths
  raise the same `GEOSException`. The claimed guarantee is the exact COUNT plus
  the preserved non-finite/uncertain error path — not "no engine can ever
  raise". Both branches are pinned by a test.
* An index built from a different footprint artifact is refused
  (`ValueError`), never silently used.
* `_build_footprint_index` is skipped below
  `FOOTPRINT_INDEX_MIN_POLYGONS = 128` footprints: measured per-Cell query
  overhead is 0.83x–0.98x (i.e. a slowdown) on small/overlapping corpora and
  there is nothing to filter.

## Rework-1 review resolutions

| finding | resolution |
|---|---|
| **M1** fictitious decision-only CLI pre-check | protocol §6 deleted/requalified: `zegrid_cli` runs the FULL engine; no new CLI/mode added; never to be used as a pre-check or while M33 runs |
| **M2** global worker log truncated (`mode='w'`) | protocol §2 corrected: per-user `%APPDATA%\ZeMosaic\logs\zemosaic_worker.log`, truncation on any new engine launch, distinct output dirs do NOT protect it; strict serialization + archive the worker log once idle; logging implementation untouched |
| **M3** contradictory conditions | protocol §3 rewritten: output folder/provenance/paths/timings/additive diagnostics are intentionally different; identical SETTINGS list; an explicit identical effective RAM budget is required, with a documented recovery rule for an `auto` baseline (pin the recorded effective budget; otherwise the comparison is observational/confounded, never scored as a parity failure); Tristan launches the candidate, Junior never launches a Windows run |
| **L1** index not observable | `layout index:` diagnostic in the real run log (build + every completion/refusal, WARN when degraded), outside the decision lines and outside the layout dict; 4 dedicated tests incl. fallback/query-failure visibility and decision invariance |
| **L2** `rss_peak_bytes` was not a peak | renamed to `rss_after_bytes_max` (same numbers, probe + evidence), claim corrected to post-call RSS; no expensive benchmark re-run |
| **L3** over-strong error-equivalence claim | docstrings narrowed; focused bowtie test pins both branches (overlapping bbox raises in both paths, disjoint bbox counts exactly with no raise) |
| **L4** unbounded/unmeasured retention | documented O(cells) with a measured retained-Python-cost at 67x39 (2 613 Cells ≈ 1.22 MiB) in a bounded micro-probe |

## Limits

* No Windows, no M33, no ZeAlfie/ZeMosaic GUI, no real 21 818-frame corpus, no
  pixels, no science comparison beyond identical memberships/decisions.
* (B) early rejection is NOT integrated (see above).
* The load-aware pass is not accelerated beyond the shared prefilter; the joint
  aggregate probe and the WCS footprint projection are untouched.
* Synthetic dense corpora are projection-bound; their gains do not represent the
  M33 run.
* Transient intra-scan memory peaks are not measured (only post-call RSS and
  retained Python object cost).
