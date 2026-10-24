# ZeGrid Windows beta witness (R55-E3)

Status: **readiness only — `WINDOWS_NOT_RUN`**. This guide lets a Windows user
exercise the opt-in ZeGrid scheduler and capture reviewable evidence. It is not a
performance benchmark, a hardware qualification, or a promise of speed. Windows
and RTX remain **unqualified** until a real result is imported and reviewed.

## 1. Prerequisites

- Windows 10/11 x64, ZeMosaic beta build (CPU-only build is supported).
- Optional GPU run: NVIDIA GPU (e.g. RTX 3070) with a current driver, plus a
  ZeMosaic build that includes the optional CuPy extra. CuPy is never a base
  dependency and is not required for CPU runs. If CuPy is installed, the
  historical `parallel_utils` module may import it even on a CPU run; that is
  pre-existing and does not by itself activate GPU execution.
- A bounded corpus: roughly 8–30 FITS frames with a small canvas. Do not start
  with a full night or a multi-gigabyte dataset.
- Free disk for outputs, cache and logs. Close other GPU-heavy applications.

## 2. Exact settings

Main tab:

1. Check **“Let me use ZeGrid”**.
2. Keep **Anti-checkerboard smoothing** at `32 px` or the recommended `64 px`.
3. Set **ZeGrid scheduler (advanced)** to **“Active — experimental beta”**.

System tab:

4. Check **“Use GPU acceleration when available”** and select the intended GPU.
5. Leave the **Hybrid GPU stability guard** enabled. Do not change the VRAM
   policy or RAM guards.

`Active` does **not** enable the GPU by itself: the GPU toggle above is still
required. To capture a CPU reference, use scheduler **“Standard (legacy)”** and
leave the GPU toggle off.

## 3. Runs to capture

Run the same bounded corpus three times into separate output folders:

| Run | Scheduler | GPU toggle | Purpose |
| --- | --- | --- | --- |
| A | Standard (legacy) | off | CPU reference for parity |
| B | Active — experimental beta | off | Active CPU path |
| C | Active — experimental beta | on | Active GPU path (optional) |

Run C requires a working CUDA device and the optional CuPy build. If CuPy or the
device is missing, the engine must refuse or degrade honestly with a recorded
reason; it must never crash silently or silently claim GPU execution.

## 4. Artefacts to keep per run

- `mosaic_grid.fits`
- `mosaic_grid_coverage.fits`
- `zegrid_manifest.json`
- the run log (`run.log` or the GUI log export)
- the saved configuration snapshot
- SHA-256 hashes of every artifact

For the GPU run, also write `windows_result.json` next to the artifacts using the
contract in `tools/zegrid_r55/windows_result_check.py`:

```json
{
  "schema": "zemosaic-r55e3-windows-result-v1",
  "status": "DONE",
  "os": {"name": "Windows", "version": "11 23H2", "arch": "AMD64"},
  "gpu": {"name": "NVIDIA GeForce RTX 3070", "driver": "551.xx",
          "runtime": "12.x", "vram_total_bytes": 8589934592,
          "vram_free_bytes_at_start": 7516192768},
  "config": {"zegrid_mode": true, "use_gpu": true,
             "zegrid_scheduler_mode": "active", "zegrid_halo_px": 64},
  "run": {"corpus_frames": 12, "peak_rss_bytes": 2147483648,
          "ram_budget_bytes": 34359738368},
  "fallback": {"used": false, "reason_code": null},
  "parity_reference": {"mosaic_grid.fits": "<sha256 of run A>",
                       "mosaic_grid_coverage.fits": "<sha256 of run A>"},
  "artifacts": {"zegrid_manifest.json": "<sha256>", "run.log": "<sha256>"}
}
```

Never paste secrets, API keys, account names or personal paths into a report.

## 5. Checks

1. **Parity** — for runs B and C, the two FITS files must be byte-identical to
   run A (same corpus/config). Compare SHA-256 digests.
2. **Manifest** — `r55_scheduler.mode` and `requested_mode` are `active`,
   `status` is `executed`, and `identity` matches run A’s layout, cell IDs,
   frame IDs and halo.
3. **GPU mapping** — for run C, `r55_scheduler.mapping.backend` is `gpu`,
   `units_in_flight` is `1`, and `prefetch_units` is within the bounded ladder.
4. **Thread lifecycle** — `r55_run_telemetry.gpu_pipeline.threads_joined` is
   `true`, the ledger is balanced, and `queue_max == prefetch_units + 1`.
5. **RAM** — record `peak_rss_bytes`; it must stay below the run RAM budget.
6. **VRAM** — record total/free VRAM at start; `device_peak_bytes` must stay
   within the 70% policy budget.
7. **Fallback honesty** — if a fallback occurred, the manifest publishes the CPU
   backend, does not claim GPU cell execution, and records a non-empty reason.

Run the read-only checker:

```bash
python tools/zegrid_r55/windows_result_check.py --result <captured-run-dir>
```

A missing or inconsistent result reports `WINDOWS_NOT_RUN` /
`WINDOWS_RESULT_INVALID` and exits non-zero. A structurally valid result reports
`WINDOWS_RESULT_VALID_UNQUALIFIED` with `qualified: false`.

## 6. Reporting

Attach the artefacts or their hashes, the checker verdict, Windows version, GPU
model, driver, RAM/VRAM observations, and the exact corpus size. Until then the
project status remains **`WINDOWS_NOT_RUN` / non qualifié**. No performance or
qualification claim may be made before a real reviewed result.
