# ZeMosaic — R3 Post-R2 Technical Audit / Final Report

- **mission_id:** `ZM-ARCH-R3-POST-R2-AUDIT-20261004`
- **phase:** implementation (docs + validation only)
- **status:** `LOCAL TECHNICAL ACCEPT / HUMAN SCIENCE ACCEPT`
- **date:** 2026-10-04
- **author:** Coco (implementation worker) for Junior (architect) + Tristan (final human authority)
- **repository:** `/home/tristan/.openclaw/workspace/projects/zemosaic`
- **branch:** `refactor/zm-architecture-cleanup-r0-r3`

This document closes the bounded R2 scope for **this mission** and records the R3
post-R2 audit. It was **locally accepted on technical grounds** after Nono `review-0:
ACCEPT` and Junior acceptance. M106 was subsequently executed as a separate mission and
received Tristan's explicit visual/scientific `ACCEPT` on 2026-10-04 at 14:17 Europe/Paris.
No publication gate (push/merge/tag/release) is authorized by that acceptance.

---

## 1. Hard gate (verified first, read-only)

| Check | Expected | Observed |
| --- | --- | --- |
| Branch | `refactor/zm-architecture-cleanup-r0-r3` | ✅ |
| HEAD | `1282cfe8901fac86ef21751991dcd98ae63679d2` | ✅ |
| Worktree | clean | ✅ (`git status --porcelain=v1` empty) |
| `origin/main` | `c03d0bb965d073b12ad9978094327829f0d0c366` | ✅ |
| `origin/beta` | `c03d0bb965d073b12ad9978094327829f0d0c366` | ✅ |
| canonical base | `c03d0bb965d073b12ad9978094327829f0d0c366` | ✅ (all three equal) |
| Remote | `https://github.com/tinystork/zemosaic.git` | ✅ |
| Worktrees | single checkout @ `1282cfe` | ✅ |
| Version | `4.7.0` (`__init__.py`, `version.txt`, `pyproject.toml` attr) | ✅ consistent |

No unexpected deviation ⇒ gate **PASS**, no reset/stash/switch/repair.

## 2. Audited implementation HEAD

- **Canonical base:** `c03d0bb965d073b12ad9978094327829f0d0c366`
- **pre-R2 baseline (freeze):** `7c469562cb40521bb4f4d8a4ab511a5099ce17a4`
- **Audited implementation HEAD:** `1282cfe8901fac86ef21751991dcd98ae63679d2`
  (`refactor: extract crash breadcrumb engine` — R2 lot 2B)
- The **report/document commit for this mission comes later** and is **not** part of the
  audited implementation HEAD. The audited implementation HEAD remains `1282cfe`. Junior will
  report the later docs commit separately; this report deliberately does **not** fabricate an
  impossible self-referential final SHA.

## 3. Chronological commits (canonical base → HEAD)

```
035119e docs: plan ZeMosaic architectural cleanup
93eb3a8 docs: record ZeMosaic R0 architecture archaeology
1655392 test: restore Phase 3 adaptive witnesses
b42c0a6 test: characterize worker dispatch propagation
b7cb83e test: characterize stacking edge contracts
efad088 test: witness spawned worker lifecycle
f4915d1 test: characterize cache and resume contracts
e975c93 docs: close R1 with no proven-dead deletion
7c46956 docs: freeze pre-R2 stacking contracts
8b7a979 refactor: extract shared filter grouping helpers into neutral module   (R2 lot 1)
5d7920d test: characterize crash breadcrumb lifecycle                          (R2 lot 2A)
1282cfe refactor: extract crash breadcrumb engine                              (R2 lot 2B)
```

pre-R2 baseline (`7c46956`) → HEAD is exactly the last three commits: `8b7a979`, `5d7920d`,
`1282cfe`.

## 4. File inventory (canonical base → HEAD) — classification

15 files changed, `+4993 / −322`.

| File | Delta (numstat +/−) | Classification |
| --- | --- | --- |
| `docs/refactor/ARCHAEOLOGY_R0.md` | A (+798/−0) | docs |
| `docs/refactor/STACKING_CONTRACTS_R3.md` | A (+514/−0) | docs |
| `todo.md` | A (+401/−0) | docs |
| `src/zemosaic/core/grouping_helpers.py` | A (+253/−0) | production — mechanical extraction (R2 lot 1) |
| `src/zemosaic/core/crash_breadcrumbs.py` | A (+184/−0) | production — mechanical extraction (R2 lot 2B) |
| `src/zemosaic/zemosaic_filter_gui.py` | M (+8/−238) | production — re-export shim (R2 lot 1) |
| `src/zemosaic/zemosaic_filter_gui_qt.py` | M (+1/−1) | production — import path (R2 lot 1) |
| `src/zemosaic/zemosaic_worker.py` | M (+20/−70) | production — thin adapters (R2 lot 2B) |
| `tests/test_phase3_adaptive_invariants.py` | M (+13/−13) | test (TEST-01 import fix, pre-R2) |
| `tests/test_dispatch_propagation_witness.py` | A (+346/−0) | test (pre-R2) |
| `tests/test_stacking_low_n_all_invalid_witness.py` | A (+371/−0) | test (pre-R2) |
| `tests/test_spawn_worker_process_witness.py` | A (+167/−0) | test (pre-R2) |
| `tests/test_cache_resume_characterization_witness.py` | A (+734/−0) | test (pre-R2) |
| `tests/test_grouping_helpers_r2_lot1.py` | A (+268/−0) | test (R2 lot 1) |
| `tests/test_crash_breadcrumbs_characterization_witness.py` | A (+915/−0) | test (R2 lot 2A/2B) |

Totals: **+4993 / −322** (exact `git diff --numstat` canonical base → implementation HEAD).

**pre-R2 baseline → HEAD (the R2 delta only):** 9 files —
`M ARCHAEOLOGY_R0.md`, `A core/grouping_helpers.py`, `A core/crash_breadcrumbs.py`,
`M zemosaic_filter_gui.py`, `M zemosaic_filter_gui_qt.py`, `M zemosaic_worker.py`,
`A test_grouping_helpers_r2_lot1.py`, `A test_crash_breadcrumbs_characterization_witness.py`,
`M todo.md`.

**No unexpected file, dependency, version, default, config, API, or science change.** No
`pyproject.toml` / `setup` / `requirements` / workflow / package-manifest touched. The only
production deltas are the two declared mechanical extractions and their compatibility seams.

## 5. R0 classification outcome (unchanged, not rewritten)

R0 archaeology accepted: Nono `review-3: ACCEPT`, then Junior acceptance. No component was
promoted to PROVEN DEAD; the "PROVEN DEAD list" is **None**. Phase 4.5 affine/micro-align
helpers remain DORMANT/UNREACHABLE (not dead). See `ARCHAEOLOGY_R0.md` and
`.a2a-reports/ZM-ARCH-CLEANUP-R0-20261003.*`.

## 6. R1 no-deletion outcome (unchanged, not rewritten)

R1 **CLOSED WITH NO DELETION** (commit `e975c93`, docs-only). No candidate crossed the
PROVEN DEAD threshold, so no `refactor: remove …` commit exists. `ARCH-03`
(`core/cuda_utils.py::enforce_nvidia_gpu`) remains SUSPECTED DEAD (not deleted); `ARCH-04`
(`zemosaic_gui.py` Tk legacy) remains DORMANT BUT REACHABLE.

## 7. R2 exact extractions and compatibility seams

R2 was **closed for this mission** after exactly three bounded, architect-approved units. This
does **not** claim the ~38k-line worker is "finished"; it deliberately avoids arbitrary
extractions. Future decomposition requires a new bounded mission with witnesses/criteria/review.

### Lot 1 — shared filter grouping helpers (commit `8b7a979`, Nono review-0 ACCEPT)

- New neutral module `src/zemosaic/core/grouping_helpers.py` (imports `math`/`typing`/
  `collections.abc` only; no Tk/Qt/`zemosaic_filter_gui`). Holds the six canonical helpers
  `_group_center_deg`, `_angular_sep_deg`, `_merge_small_groups`, `_circ_delta_deg`,
  `_circular_dispersion_deg`, `_split_group_by_orientation` moved **verbatim** (AST
  byte-identical, verified by Nono).
- `zemosaic_filter_gui.py` re-exports the six names (same objects, `is`-identity verified).
- `zemosaic_filter_gui_qt.py` official path imports the three named helpers from
  `core.grouping_helpers` (single import-line change); the inline Qt fallback copies and the
  separate Qt variant algorithms (`_split_group_by_orientation_key`, `split_clusters_by_orientation`,
  `_split_group_by_mount_mode`) are **intentionally NOT consolidated** (distinct algorithms).
- Witness `tests/test_grouping_helpers_r2_lot1.py`: 25 pass.
- Compatibility seam: legacy re-export shim preserves every existing import path.

### Lot 2A — crash-breadcrumb characterization witness (commit `5d7920d`, Nono review-0 ACCEPT)

- `tests/test_crash_breadcrumbs_characterization_witness.py` (47 pass) pins the pre-extraction
  behavior of `_configure_crash_breadcrumbs` / `_safe_runtime_snapshot` / `_emit_crash_breadcrumb`
  and the in-process `run_hierarchical_mosaic_process` lifecycle. No product-code change in 2A.

### Lot 2B — stateless crash-breadcrumb engine (commit `1282cfe`, Nono review-0 ACCEPT)

- New neutral core `src/zemosaic/core/crash_breadcrumbs.py` (stdlib only —
  `json`/`os`/`time`/`datetime`/`pathlib`/`typing`; no `zemosaic_worker`/GUI/Qt/Tk/GPU/CuPy/config/
  solver/science import; no new dependency; no mutable breadcrumb paths/mode/lock singleton).
- `zemosaic_worker.py` **remains the single compatibility source of truth** for the four
  observable globals `_CRASH_BREADCRUMB_LOCK` / `_CRASH_BREADCRUMB_PATH` /
  `_CRASH_STATE_PATH` / `_CRASH_BREADCRUMB_MODE`. `_configure_crash_breadcrumbs` /
  `_safe_runtime_snapshot` / `_emit_crash_breadcrumb` are now thin adapters (same names/
  signatures) reading worker globals at CALL TIME, so direct assignment + monkeypatch keep
  their exact effect. Heartbeat thread, signal install/restore, queue protocol, wrapper
  context, and all `_emit_crash_breadcrumb` call sites were **not** extracted.
- Witness extended 47 → 53 pass/0 skip (6 additive extraction assertions, none weakened).
- Compatibility seam: worker-owned globals + call-time read = behavior frozen.

## 8. R3 stacking-contract conclusion (post-R2)

The R3 stacking contract map (`STACKING_CONTRACTS_R3.md`) was frozen pre-R2 (Nono review-1
ACCEPT + Junior acceptance). Post-R2 re-verification confirms **neither extraction touches
stacking math/order/weights/rejection/WCS/FITS/science**:

- Lot 1 (`grouping_helpers`) is a **filter GUI** concern (group merge/split/dispersion):
  the extracted module is never imported by a stacking path and changes no numeric behavior.
- Lot 2B (`crash_breadcrumbs`) is a **crash-autopsy side-channel** (append-only JSONL +
  last-state + RAM/VRAM snapshot), orthogonal to stacking.

The map remains valid: all eight stacking rows (Classic CPU/GPU, SDS CPU, Grid CPU/GPU-core/
GPU-legacy, Phase 4.5 alpha + configured-rejection) and the SCI-01/02/03 divergence pinning are
unchanged. **Status: POST-R2 AUDITED — ACCEPTED** after Nono `review-0: ACCEPT` and
Junior acceptance. M106 scientific acceptance was subsequently granted by Tristan; see §12.

## 9. Validation (exact commands / results)

Environment: repo `.venv/bin/python` 3.13.5, pytest 9.1.1, Linux x64 (TINYDEBIAN). All runs
isolated with `HOME=/tmp/zm-r3-home`, `XDG_CONFIG_HOME=/tmp/zm-r3-xdg`,
`XDG_DATA_HOME=/tmp/zm-r3-data`, `XDG_CACHE_HOME=/tmp/zm-r3-cache`, `TMPDIR=/tmp/zm-r3-tmp`,
`QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, `PYTHONDONTWRITEBYTECODE=1`.

### A/B — full pytest suite (subsumes the R2 combined 143-baseline)

```
.venv/bin/python -m pytest -q -ra
→ 349 passed, 0 failed, 0 skipped, 261 warnings in 33.44s
```

The R2 combined suite (143) was **not** blindly duplicated: the full suite subsumes it and
reports 349 pass (53 crash + 14 dispatch + 1 spawn + 17 packaging + 25 grouping + 33 phase3 +
remaining solver/grid/cache/stack/telemetry tests). No timeout, no rerun needed.

Warnings breakdown (all pre-existing, none a failure; exact category sum = 261):
- **229** × `DeprecationWarning: datetime.utcnow()` at `core/crash_breadcrumbs.py:161` (iso field).
- **9** × `DeprecationWarning: datetime.utcnow()` at `zemosaic_worker.py:24541` (Phase-5 checkpoint `created_utc`).
- **5** × `DeprecationWarning: datetime.utcnow()` at `zemosaic_worker.py:24805` (Phase-1 resume `created_utc`).
- **14** × astropy FITS `VerifyWarning` HIERARCH cards (7 keyword types × 2 phase-3 tests).
- **4** × `DeprecationWarning: deprecated logging.warn` at `zemosaic_filter_gui_qt.py:7300`.

Total 229 + 9 + 5 + 14 + 4 = **261**. None is a test failure.

### C — packaging / wheel

- `python -m build` **available** (build 1.5.0), but the build backend `setuptools.build_meta`
  was **not** importable in the repo `.venv` (setuptools absent). Worked around **without
  network/install** by putting the system setuptools (78.1.1,
  `/usr/lib/python3/dist-packages`) on `PYTHONPATH`:

```
PYTHONPATH=/usr/lib/python3/dist-packages .venv/bin/python -m build --wheel \
  --outdir /tmp/zm-r3-wheelbuild --no-isolation
→ Successfully built zemosaic-4.7.0-py3-none-any.whl
```

- Wheel contents verified to include `zemosaic/core/grouping_helpers.py`,
  `zemosaic/core/crash_breadcrumbs.py`, `zemosaic/zemosaic_worker.py`, and the official
  entrypoint `zemosaic-4.7.0.dist-info/entry_points.txt` = `[gui_scripts] zemosaic = zemosaic._app:main`.
- Import from the extracted wheel **outside the checkout** with the full dependency set (repo
  `.venv`) — all pass: `zemosaic` (4.7.0), `zemosaic.core.grouping_helpers`,
  `zemosaic.core.crash_breadcrumbs`, `zemosaic.zemosaic_worker`, and
  `zemosaic_filter_gui._merge_small_groups.__module__ == "zemosaic.core.grouping_helpers"`.
- Isolated temp venv (`--system-site-packages` + `pip install --no-deps --no-index <wheel>`)
  import result (recorded exactly): `zemosaic`, `grouping_helpers`, `crash_breadcrumbs` import
  OK; `zemosaic.zemosaic_worker` fails with `ModuleNotFoundError: No module named 'zarr'`
  because `--no-deps` omits the declared runtime dependency `zarr>=2` (and other science deps).
  This is an **artifact of the no-deps install, not a packaging defect** of the extracted
  modules; the full-deps import (above) proves the wheel is complete. No network used, no
  user profile touched.

### D — static / cleanliness

- `git diff --check` → clean (exit 0).
- `py_compile` on `grouping_helpers.py`, `crash_breadcrumbs.py`, `zemosaic_worker.py`,
  `zemosaic_filter_gui.py`, `zemosaic_filter_gui_qt.py` → OK.
- Version consistency: package `__version__ = "4.7.0"` in `src/zemosaic/__init__.py`;
  `version.txt` is multi-line and its **first line** is `ZeMosaic 4.7.0`; `pyproject.toml`
  uses a **dynamic** `version = { attr = "zemosaic.__version__" }` (read from the package
  attribute). All three agree on 4.7.0; `version.txt` is not a bare single-line `4.7.0`.
- Final candidate worktree state (`git status --porcelain=v1`): exactly the 4 authorized
  docs — ` M docs/refactor/ARCHAEOLOGY_R0.md`, ` M docs/refactor/STACKING_CONTRACTS_R3.md`,
  ` M todo.md`, `?? docs/refactor/FINAL_REPORT.md`. **No generated build/test artifact entered
  the repo** beyond these authorized docs; `build/`, `*.egg-info/`, `__pycache__/`,
  `.pytest_cache/` remain pre-existing git-ignored entries, not tracked. (Clean-at-start is
  recorded in the hard gate, §1; the final state here is the 4-doc candidate worktree.)

## 10. All Nono reviews (durable reports, re-checked — not rewritten)

| Mission | Nono verdict |
| --- | --- |
| R0 cleanup `ZM-ARCH-CLEANUP-R0-20261003` | review-3 **ACCEPT** |
| R3 baseline freeze `ZM-ARCH-R3-BASELINE-FREEZE-20261004` | review-1 **ACCEPT** |
| TEST-01 `ZM-ARCH-TEST01-PHASE3-IMPORTS-20261003` | review-0 **ACCEPT** |
| WITNESS dispatch `ZM-ARCH-WITNESS-DISPATCH-20261003` | review-0 **ACCEPT** |
| WITNESS stack-edges `ZM-ARCH-WITNESS-STACK-EDGES-20261003` | review-0 **ACCEPT** |
| WITNESS spawn `ZM-ARCH-WITNESS-SPAWN-20261003` | review-0 **ACCEPT** |
| WITNESS cache/resume `ZM-ARCH-WITNESS-CACHE-RESUME-20261004` | review-0 **ACCEPT** |
| R2 lot 1 `ZM-ARCH-R2-LOT1-GROUPING-HELPERS-20261004` | review-0 **ACCEPT** |
| R2 lot 2A `ZM-ARCH-R2-LOT2A-CRASH-BREADCRUMB-WITNESS-20261004` | review-0 **ACCEPT** |
| R2 lot 2B `ZM-ARCH-R2-LOT2B-CRASH-BREADCRUMB-EXTRACTION-20261004` | review-0 **ACCEPT** |

All verdicts verified against their durable report files under
`/home/tristan/.openclaw/workspace/.a2a-reports/`. No history rewritten, no overclaim.

## 11. Anomalies intentionally NOT fixed (open / NOT_RUN / platform limits)

Preserved unchanged (see `todo.md` §10 for full ledger):

- **SCI-01** — Grid CPU/GPU-legacy use PixInsight WSC by default vs `stack_core` simplified
  median/σ winsorized; numeric impact unmeasured. NOT_RUN.
- **SCI-02** — `stack_core` `linear_fit` placeholder; Grid GPU normalizes upstream and passes
  `none`. NOT_RUN.
- **SCI-03** — Grid CPU zeros vs `stack_core` NaN for all-invalid pixels (pinned divergence,
  not fixed). NOT_RUN for CPU↔GPU.
- **SCI-04** — Classic/SDS/Phase 4.5 low-N/chunking variants not inventoried.
- **ARCH-01** — Phase 4.5 supported invocations / Tk fallback UNKNOWN.
- **ARCH-02** — external/frozen/programmatic-API contracts incomplete.
- **ARCH-03** — `cuda_utils.enforce_nvidia_gpu` SUSPECTED DEAD (kept).
- **ARCH-04** — `zemosaic_gui.py` Tk legacy DORMANT (kept).
- **ARCH-05** — Classic/SDS SDS-block duplication (documented, not consolidated).
- **ARCH-06** — Grid GPU `stack_core` reuse + linear_fit placeholder (SCI-01/02 confirmation).
- **SCI-07** — Phase 4.5 affine/micro-align helpers absent at BASE (DORMANT/UNREACHABLE).
- **TEST-02** — `test_version_gpu.py` is a diagnostic script (0 items collected), not a test.
- **MANUAL-01** — M106 before/after: completed on cupy/MX150 with BASE, CANDIDATE and a
  same-build repeat; Tristan visual/scientific verdict `ACCEPT`. Other platforms remain NOT_RUN.
- No CPU/GPU parity is claimed. The original R3 validation used fakes and did not execute GPU
  stacking; the later M106 gate exercised real cupy/MX150 for this corpus only and does not prove
  backend parity. No external solver/catalogue, network, PyInstaller build, or Windows/macOS
  qualification was performed; those remain NOT_RUN.

## 12. M106 manual gate (COMPLETED — HUMAN ACCEPT)

M106 was executed later under mission `ZM-ARCH-M106-GATE-20261004` against the private
66-FITS corpus. Immutable source snapshots, manifests, adapted/effective configs, logs,
telemetry, FITS comparisons and visual artifacts are preserved under
`/home/tristan/M106/gate_evidence_20261004/`.

- BASE `c03d0bb`, CANDIDATE `1282cfe`, plus one same-build candidate repeat ran sequentially
  through cupy on the MX150 with the same corpus, 26-group preplan and isolated outputs.
- Coverage, winner map and weighted coverage map are array-bit-identical; shapes, WCS and
  NaN masks match. Science/aesthetic are not bit-identical.
- Same-build variability is comparable to or larger than BASE↔CANDIDATE. The locked mechanical
  criterion therefore remains `INCONCLUSIVE/HOLD`; no threshold was loosened post hoc and no
  refactor-attributable science regression was demonstrated.
- Tristan inspected the full-resolution result, reported no visible issue, and explicitly set
  `HUMAN_VISUAL_VERDICT=ACCEPT` on 2026-10-04 at 14:17 Europe/Paris.

Final M106 project gate: **HUMAN SCIENCE ACCEPT**. This does not authorize publication.

## 13. Publication gates

- **No commit** in this mission. **No push / merge / tag / release / version bump.** No
  `src/` or `tests/` change. The report/document commit is created **later** by Junior and
  reported separately; the audited implementation HEAD stays `1282cfe`.

## 14. Next steps

1. The technical docs/report checkpoint is commit `97c08f1`; commit this M106 acceptance addendum
   separately. Neither documentation commit changes the audited implementation HEAD `1282cfe`.
2. M106 is complete and human-accepted. The only remaining project gate is a separate,
   explicit publication decision (push/merge/tag/release remain unauthorized).

## 15. Limitations / review readiness

- Final state: **LOCAL TECHNICAL ACCEPT / HUMAN SCIENCE ACCEPT** (Nono review-0,
  Junior acceptance, then Tristan M106 visual/scientific acceptance).
- This report is a docs-only artifact; it cannot self-reference its own commit SHA.
- All observed facts above are reproduced from live git/pytest/wheel evidence; unverified
  hypotheses are labelled UNKNOWN/NOT_RUN.

---

## 16. Post-report promotion and Windows closure addendum — 2026-10-04

> **Operationally supersedes** the earlier publication gate / next-step state in §13–§14.
> Sections 1–15 remain the historical audit-time report and are **not** rewritten. Statements
> that publication was not yet authorized were true at audit time; the promotion below is a
> separate, later product-owner action and does not retcon those statements.

### Terminal mission status

**R0–R3 CLOSED / TECHNICAL ACCEPT / HUMAN SCIENCE ACCEPT / PROMOTED MAIN+BETA.** The locked
mechanical bitwise result remains `INCONCLUSIVE/HOLD`; Tristan's human/scientific verdict is
`ACCEPT` (2026-10-04 14:17 Europe/Paris). This distinction is preserved.

### Promotion references (exact)

| Item | Ref / SHA |
| --- | --- |
| PR #420 refactor → beta | https://github.com/tinystork/zemosaic/pull/420 — merge `bbe63d4edc65d4d762a6d42e59081f6fe7130697` |
| PR #421 Windows intertile safeguard → beta | https://github.com/tinystork/zemosaic/pull/421 — merge/current beta `14607d94342a308672b5be04d0db5a562fc564ab` |
| PR #422 beta → main | https://github.com/tinystork/zemosaic/pull/422 — merge/current main `940655a7202568dad8dd8fa18ea5f12bfea284f8` |
| Version | `4.7.0` (unchanged; no bump) |
| Tree equality | `origin/main` and `origin/beta` commit IDs differ; trees are **exactly identical** |

### Runtime and test facts

- Post-main detached-worktree suite: **445 passed, 0 failed, 261 warnings**.
- No tag, release, deployment, or version bump was performed.
- **Windows incident proof (bounded conclusion):** `/home/tristan/M106/faulthandler_intertile.log`
  records `0xc0000374` with exactly six ThreadPool workers in `_process_overlap_pair -> reproject_interp`;
  captured worker stacks concentrate in reproject/Astropy WCS; parent waits on futures; OpenCV is
  absent from the captured worker stacks. This proves the crash **domain** (intertile reprojection
  under concurrency) — **not** the exact defective native layer, **nor** that a shared WCS identity
  is the necessary cause.
- **Windows mitigation proof:** `/home/tristan/M106/outwinsequential/` logs workers 14→1, token
  `windows_reproject_wcs_serial`, sequential mode, 279/279 pairs, 26/26 Phase 5, WORKER_DONE/run
  success, FITS/preview artifacts. Intertile sequential ~144.1s of the 1570.1s full run.

### Remaining backlog (separate, non-blocking missions)

Nothing below is implemented as part of this closure.

- **SCI-01 (P1, recommended next)** — numerical Grid CPU/legacy WSC vs GPU `stack_core` simplified
  witness/quantification; no correction bundled.
- **SCI-02 / SCI-03 / SCI-04** — scientific debts (linear_fit placeholder; Grid CPU/GPU masks/
  weights/all-invalid/aliases; Classic/SDS/Phase 4.5 low-N/chunking variants).
- **ARCH-01/02/03/04/05 + SCI-07** — architecture/unknown/dormant debts.
- **TEST-02** — diagnostic-script cleanup (`test_version_gpu.py`).
- **Platform/distribution parity** — macOS, broader frozen/PyInstaller/external solver, CPU↔GPU;
  Windows claim is bound to the observed standalone M106 path.
- **PERF-01 (deferred)** — sequential top-K pair pruning; K=8 → 279→155 pairs, graph connected,
  ideal linear estimate ~144→80s. K=8 MUST NOT change the Windows worker count (still
  effective_workers=1 / no ThreadPool). Not implemented/active; requires science comparison
  against the full graph.

No CPU/GPU parity, macOS qualification, bitwise science identity, or dead-code proof is claimed.
