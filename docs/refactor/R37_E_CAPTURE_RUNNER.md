# R37-E — Reproducible diagnostic-capture runner (CLI)

**Mission:** ZM-ZEGRID-R37-E-RUNNER · **Phase:** implementation · **Iteration:** r0
**Baseline:** `beta` @ `0117a6392963633969a00b0d94770296799b8392` (R37-C/D accepted).
**Status:** IMPLEMENTED. Adds a standalone, dry-run-by-default CLI runner. **No
production runtime, default or GUI change.**

---

## 1. Why

The accepted R37-D capture seam publishes the R37-C pre-offset/pre-feather
artifact into a run's output directory — but only when a run is deliberately
started with `r37_diagnostic_capture=True`. The canonical baseline
(`R34F`: 653 selected frame_ids out of 816 files, 6x4 layout, effective
`per_cell`, canvas 4117x4075) must be reproduced **exactly** so the captured
evidence corresponds to the accepted relations.

Reproducing it by hand is fragile: the exact frame subset, the pinned layout, the
gauge mode and the capture opt-in all have to be reconstructed coherently from a
prior run's `zegrid_manifest.json` + `run_config_snapshot.json`. The original
capable host was a Windows box (RTX 3070, ~16 GiB); this Linux host cannot fit the
6x4 layout at all (measured: min CPU tile-32 worst-cell bound **8561.5 MiB** vs
~1.3 GiB available). R37-E provides one documented command to run the capture
safely on a capable machine, with a hard preflight that refuses an infeasible host.

## 2. What this adds

* `tools/r37_capture_runner.py` — a generic, dry-run-by-default CLI.
* `tests/test_zegrid_r37e_capture_runner.py` — focused suite.
* this document.

**No** production module, default, GUI wiring, correction application, new
dependency or network access. The runner drives the **already-accepted**
`run_zegrid_mode(...)` entry point directly (no shell interpolation, no hidden
subprocess, no GUI/config activation).

## 3. CLI contract

```
python tools/r37_capture_runner.py \
    --input-dir PATH            # folder containing the source FITS
    --baseline-run-dir PATH     # contains zegrid_manifest.json + run_config_snapshot.json
    --output-dir PATH           # NEW/empty destination; never the baseline dir
    [--holdout-cap N]           # default 1000000 (exact-heldout target)
    [--gpu {auto,on,off}]       # default auto (prefers baseline use_gpu_grid)
    [--execute]                 # required for any write / full run
    [--skip-memory-preflight]   # inspection dry-run ONLY (refused with --execute)
```

Exit codes: `0` ok · `1` validation/other error · `2` memory preflight FAIL ·
`3` unsafe output destination.

* **Dry-run (default).** Performs the real descriptor/canvas/pinned-layout memory
  preflight using the **same accepted ZeGrid functions** as a production run, and
  writes **nothing** — not in the input dir, not in the baseline dir, not in the
  output dir.
* **`--execute`.** Additionally requires a new/empty output dir that cannot
  overwrite the input or the baseline, and a **successful real memory preflight**.
  On success it verifies the artifact↔manifest linkage and writes a compact
  verification report. If the atomic report write fails, the run returns nonzero
  and never prints `VERIFICATION PASS`.
* **Memory-safety contract.** `--execute` **must** have a PASSing preflight.
  `--skip-memory-preflight` is an **inspection-only** affordance: it reports
  `preflight SKIPPED` (never `PASS`), is refused outright when combined with
  `--execute` (before any engine work), and there is deliberately **no
  `--force-infeasible`**.

### Examples

bash (Linux/macOS):

```bash
python tools/r37_capture_runner.py \
    --input-dir "/path/to/lights organized/EQ/IRCUT" \
    --baseline-run-dir "/path/to/zegrid_out/R34F" \
    --output-dir "/path/to/zegrid_out/R37E_CAPTURE" \
    --gpu auto --holdout-cap 1000000                     # dry-run / preflight

python tools/r37_capture_runner.py \
    --input-dir "/path/to/lights organized/EQ/IRCUT" \
    --baseline-run-dir "/path/to/zegrid_out/R34F" \
    --output-dir "/path/to/zegrid_out/R37E_CAPTURE" \
    --gpu auto --holdout-cap 1000000 --execute           # real capture + verify
```

PowerShell (Windows; example paths only):

```powershell
python .\tools\r37_capture_runner.py `
    --input-dir "C:\Users\TRISTAN\Desktop\astrophoto\caldwell_11\lights organized\EQ\IRCUT" `
    --baseline-run-dir "C:\Users\TRISTAN\Desktop\astrophoto\caldwell_11\OUT\R34F" `
    --output-dir "C:\Users\TRISTAN\Desktop\astrophoto\caldwell_11\OUT\R37E_CAPTURE" `
    --gpu auto --holdout-cap 1000000

python .\tools\r37_capture_runner.py `
    --input-dir "C:\Users\TRISTAN\Desktop\astrophoto\caldwell_11\lights organized\EQ\IRCUT" `
    --baseline-run-dir "C:\Users\TRISTAN\Desktop\astrophoto\caldwell_11\OUT\R34F" `
    --output-dir "C:\Users\TRISTAN\Desktop\astrophoto\caldwell_11\OUT\R37E_CAPTURE" `
    --gpu auto --holdout-cap 1000000 --execute
```

The tool uses `pathlib` throughout, handles spaces in paths, and contains **no**
hard-coded Caldwell paths (the examples above are documentation only).

## 4. Behaviour

### 4.1 Baseline reconstruction

* Strict-load `zegrid_manifest.json` + `run_config_snapshot.json` (NaN/Infinity
  literals rejected). Require: `engine == "zegrid"`, `cell_feather.enabled is
  True`, non-empty unique string `frame_ids`, `n_frames == len(frame_ids)`, valid
  positive canvas/layout, and an effective `photometric_gauge.mode == "per_cell"`.
* Resolve each manifest `frame_id` to **exactly one** file under `--input-dir`
  (relative-path match first, then unique-basename), rejecting path traversal,
  absolute ids, missing ids, ambiguous basenames, duplicate resolution and any
  resolution escaping the input dir. Deterministic; the manifest order is kept and
  the exact full paths are passed as `selection_paths` (so the accepted
  inventory-scanner selection semantics apply unchanged: an exact 653-of-816
  subset).
* Pin the baseline `nx×ny` through a **copied** `SimpleNamespace` config
  (`zegrid_layout = "NXxNY"`); the source snapshot mapping is never mutated.
* Explicitly request `cell_gauge_mode="per_cell"`, `r37_diagnostic_capture=True`
  and the requested `--holdout-cap`; reconstruct normalization, rejection algo,
  kappa and winsor limits (strict tuple parsing) from the manifest/snapshot.
  All effective parameters are printed before execution.

### 4.2 Memory / capability preflight (no stacking, no writes)

Uses the accepted `_build_frame_descriptors` → `geometry.build_canvas` →
`_choose_layout_mode_aware` (pinned layout) path, plus the R36 CPU streaming-tile
ladder (`_iter_cell_bounds_mode_aware` + `_estimate_streaming_bytes` +
`_select_cpu_streaming_tile`). It reports candidate/selected frame count, rejected
WCS, canvas, pinned layout, live available RAM, the CPU tile ladder / predicted
bound, the GPU probe summary, and **PASS/FAIL**.

If the pinned layout cannot fit, the runner returns exit code `2` and creates no
output. `--execute` refuses on a failed preflight. There is deliberately **no
`--force-infeasible`**: forcing a pinned layout that cannot fit only invites
swap-thrash/OOM. `--skip-memory-preflight` is an inspection-only affordance and
is **refused with `--execute`** before any engine work; on a skip dry-run the
verdict is reported as `SKIPPED`, never `PASS`.

Measured on the current weak host (dry-run, real baseline, no writes):

```
budget 1490.5 MiB; pinned 6x4
CPU tile ladder worst-cell bounds (MiB): 128→9087, 112→8608, 96→8196,
  80→7871, 64→7709, 48→7789, 32→8561.5   (none fits)
FAIL: LayoutInfeasible: pinned layout 6x4 cannot fit any mode within budget
```

### 4.3 Execution safety

`--execute` refuses if the output equals/contains/is-contained-by the baseline or
input dir, or if it exists and is non-empty. It then calls `run_zegrid_mode`
directly with the reconstructed keyword arguments (capture ON, explicit
parameters). `run_zegrid_mode`'s progress callback is forwarded human-readably to
the console.

### 4.4 Post-run verification

**Topology (fail-closed).** Exactly one of these is accepted: (a) a single root
`zegrid_manifest.json`, or (b) recognized `grid_EQ` + `grid_ALTZ` pass manifests
with **no** root manifest. Any other `zegrid_manifest.json` anywhere under the
output dir (any depth, symlink-safe — external symlink trees are never followed)
is reported as an unknown manifest and fails; root+passes coexistence is
stale/ambiguous; a missing half of the mount pair fails.

After a successful run the runner strict-loads the artifacts + manifests and
checks, per pass:

* `manifest_links_artifact` on all four identity keys (`capture_id`, `filename`,
  `sha256`, `bytes`) — fed with **disk/artifact-derived** `sha256`/`bytes`, never
  values copied from the same manifest (non-tautological); the explicit
  disk-equality checks are kept too;
* capture id / filename / on-disk sha256 / on-disk bytes agreement;
* artifact provenance truthfulness: `mission == "ZM-ZEGRID-R37-D"`,
  `capture_stage == "pre_offset_pre_feather"`, `production_wired is True`,
  `correction_applied is False`, effective gauge `per_cell`;
* **independent exact/capped/qualification truth** derived from the strict-loaded
  artifact (`heldout_capped`, `heldout_exact()`, expected qualification) and
  compared against all three manifest capture claims; any mismatch fails;
* **source-set provenance**: `background_diagnostics.source_set_digest()` is
  recomputed from that pass's canonical relative logical ids and required to
  match BOTH the artifact `provenance.capture.source_set_digest` AND the manifest
  capture-block `source_set_digest`; both source-set-digest **algos** must equal
  `SOURCE_SET_DIGEST_ALGO`; the artifact and manifest source counts must equal the
  pass's unique logical-id count. Count-only or same-basename substitution fails;
* layout facts: manifest and artifact `nx×ny` equal the baseline.

**Identity** is the canonical POSIX relative logical path under the input root
(the same id the engine emits), never a basename; malformed out-manifest ids
(absolute / drive-absolute / UNC / traversal / empty / `.` components /
non-canonical spelling) are rejected rather than basename-normalised.

**Frame accounting:** single pass — the manifest logical-id set must equal the
exact selection (count + digest); multi-pass — each pass is internally consistent
(`n_frames == len(unique logical ids)`, artifact source count == that pass count),
the pass sets are **disjoint**, and their **union** equals the exact selected set.

**Run-level qualification is the aggregate over EVERY pass**: `heldout_exact =
all(pass artifact exact)` and `heldout_capped = any(pass capped)`; the run is
`EXACT` iff every pass is exact, otherwise `RESEARCH_ONLY_HELDOUT_CAPPED`. Per-pass
qualification/digest detail is retained, and a multi-pass run never presents a
single pass's sha/capture id as if it were the whole run (an explicit aggregate
summary + per-pass list is used instead).

It writes `r37_capture_verification.json` **atomically** into the output dir with
the command parameters, invocation parameters, preflight summary, the linkage
verdict, the aggregate qualification, the artifact digest/bytes (single pass) or
per-pass digests (multi-pass), and the repo HEAD / package version when available.
The report carries **basenames, counts and digests only**: absolute
source/temp/report paths — whole-value **and embedded after text**, POSIX /
Windows / UNC, including paths with spaces — are scrubbed at the durable report
boundary via the known roots (input, baseline, output, temporary work dir) plus
generic patterns (execution may still use the real paths). Any post-run
verification failure — or a failed atomic report write — is a nonzero exit (and a
failed write never prints `VERIFICATION PASS`).

## 5. Tests

`tests/test_zegrid_r37e_capture_runner.py` (run under `-W error`) covers: baseline
manifest/config validation (engine / cell_feather / frame_ids / canvas / layout /
effective `per_cell`, strict JSON), exact subset resolution on small fixtures with
missing / ambiguous-basename / path-traversal / Windows-like (incl. drive-absolute
and UNC) ids, strict winsor+sigma parsing, copy-without-mutation pinned config,
dry-run performs no writes and never calls the engine, failed preflight returns
nonzero and creates no output, `--skip-memory-preflight --execute` is refused
without ever running the engine, output-safety refusals, execution argument
reconstruction via monkeypatch, artifact-derived exact/capped/qualification
verification with manifest-tamper regressions (all three fields), source-set
provenance (digest/algo/count tamper, independently), canonical relative logical
identity (valid nested same-basename single- and multi-pass; substitution /
missing / extra / duplicate / malformed-id failures), the aggregate qualification
(one EXACT pass + one CAPPED pass ⇒ linkable but overall capped/not-exact),
topology fail-closed (nested unknown, root+passes, missing peer, symlink),
a failed verification-report write returning nonzero without a `PASS` claim,
report sanitization (rejected-WCS detail basename-only; embedded POSIX/Windows/UNC
text and paths with spaces scrubbed; strict JSON), relative `.git` `gitdir:`
resolution, non-tautological linkage (manifest `sha256`/`bytes` tamper fails),
atomic verification report (no temp residue), and a **small real synthetic run**
end-to-end through the CLI. A top-level import audit asserts the tool imports no
`zemosaic` module at module scope (all lazy), so importing the runner never
activates a capture or a run.

## 6. Limits / non-goals

* No production runtime/default/GUI change; diagnostic-only.
* No correction application, no `apply_plane_correction` wiring.
* No new dependency, no network, no push, no real Caldwell run performed here.
* Producing the real capture artifact requires the capable (Windows) machine; the
  weak host intentionally fails the preflight with the measured blocker and writes
  nothing.
