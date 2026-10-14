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
* `tests/test_zegrid_r37e_capture_runner.py` — focused tests (45).
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
    [--skip-memory-preflight]   # LOUD, explicit; never hides an infeasible host
```

Exit codes: `0` ok · `1` validation/other error · `2` memory preflight FAIL ·
`3` unsafe output destination.

* **Dry-run (default).** Performs the real descriptor/canvas/pinned-layout memory
  preflight using the **same accepted ZeGrid functions** as a production run, and
  writes **nothing** — not in the input dir, not in the baseline dir, not in the
  output dir.
* **`--execute`.** Additionally requires a new/empty output dir that cannot
  overwrite the input or the baseline, and a PASSing preflight. On success it
  verifies the artifact↔manifest linkage and writes a compact verification report.

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
swap-thrash/OOM. `--skip-memory-preflight` exists only as an explicit, loud
opt-out; it does not force an infeasible execution.

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

After a successful run the runner strict-loads the artifact + the completed
manifest (in the base and any mount-pass `grid_EQ`/`grid_ALTZ` dirs) and checks:

* `manifest_links_artifact` on all four identity keys (`capture_id`, `filename`,
  `sha256`, `bytes`);
* capture id / filename / on-disk sha256 / on-disk bytes agreement;
* artifact provenance truthfulness: `mission == "ZM-ZEGRID-R37-D"`,
  `capture_stage == "pre_offset_pre_feather"`, `production_wired is True`,
  `correction_applied is False`, effective gauge `per_cell`;
* layout and frame facts: manifest and artifact `nx×ny` equal the baseline, the
  manifest frame set matches the resolved selection, and the artifact/manifest
  source-frame counts equal the selection count;
* qualification is reported and stated as **EXACT** or **CAPPED** (never claimed
  exact before verifying).

It writes `r37_capture_verification.json` **atomically** into the output dir with
the command parameters, invocation parameters, preflight summary, the linkage
verdict, the artifact digest/bytes/qualification, and the repo HEAD / package
version when available. The report carries **basenames, counts and digests only**
— no credentials and no private absolute input paths. Any post-run verification
failure is a nonzero exit.

## 5. Tests

`tests/test_zegrid_r37e_capture_runner.py` (45 tests, run under `-W error`)
covers: baseline manifest/config validation (engine / cell_feather / frame_ids /
canvas / layout / effective `per_cell`, strict JSON), exact subset resolution on
small fixtures with missing / ambiguous-basename / path-traversal / Windows-like
ids, strict winsor+sigma parsing, copy-without-mutation pinned config, dry-run
performs no writes and never calls the engine, failed preflight returns nonzero
and creates no output, output-safety refusals, execution argument reconstruction
via monkeypatch, post-run linkage PASS/FAIL (missing / tampered manifest / tampered
artifact bytes), atomic verification report (no temp residue), and a **small real
synthetic run** end-to-end through the CLI. A top-level import audit asserts the
tool imports no `zemosaic` module at module scope (all lazy), so importing the
runner never activates a capture or a run.

## 6. Limits / non-goals

* No production runtime/default/GUI change; diagnostic-only.
* No correction application, no `apply_plane_correction` wiring.
* No new dependency, no network, no push, no real Caldwell run performed here.
* Producing the real capture artifact requires the capable (Windows) machine; the
  weak host intentionally fails the preflight with the measured blocker and writes
  nothing.
