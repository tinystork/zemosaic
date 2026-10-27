# M0 Windows witness harness — read this first

This folder contains the **REAL-GPU Windows witness** for mission
`ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027`, rework-1.

It is **not** a resume implementation, a product fix, a scientific
qualification, or a GPU qualification. It is a fail-closed reproduction harness.

## What it proves

With a real Windows GPU and a real small FITS corpus, it reproduces:

1. at least two cells complete (real `backend_used == "gpu"`);
2. a synthetic available-RAM drop is injected at the real product boundary
   (`zemosaic_zegrid_mode.available_memory_bytes`);
3. the ACTIVE GPU pipeline raises `_R55GpuFallback` (`R55_REFUSE_HOST_RAM`) and
   marks every completed unit `discarded=true`;
4. no FITS/coverage/manifest is published; cleanup failures are empty; the
   lease ledger is balanced with zero host/device/disk leases;
5. the whole-batch CPU re-admission fails with `AdmissionExhausted`;
6. no OOM is involved.

`PASS` means the **baseline failure was reproduced for the expected reason**.
It never means the product succeeded, resume exists, or science/GPU
qualification was obtained.

## Local Linux control test is different

`tests/test_zegrid_resume_m0_baseline_witness.py` is a **LOCAL CONTROL-ONLY**
test. It drives the real orchestration and real canonical cell computations but
replaces the GPU device executor with the CPU executor. It is labelled
`CONTROL_ONLY_FAKE_GPU`; its 16 GiB high observation is a synthetic decision
budget, not a physical-memory claim. It is not a GPU or scientific
qualification.

The Windows harness (`m0_windows_witness.py`) is the only path that can claim
`gpu_qualified=true`, and only when:

* `mode == WINDOWS_REAL_GPU`;
* GPU readiness/probe are available;
* every recorded completed cell has `backend_used == "gpu"`;
* all witness checks pass (`EXPECTED_BASELINE_FAILURE`);
* provenance is verified (installed module SHA-256 == the **actual imported**
  module file SHA-256).

## Fail-closed rules (rework-1)

- Parent success requires **child returncode == 0 AND witness.status == PASS
  AND classification == EXPECTED_BASELINE_FAILURE AND no cap breach**. A
  nonzero child exit is authoritative; the witness alone is never trusted.
- Provenance equality is enforced in parent **and** child. A mismatch refuses
  unless `--allow-unverified-provenance` is explicit; the witness then records
  `provenance_verified=false` and `qualification=NON_QUALIFYING_CONTROL`.
- The injected high observation is `min(16 GiB, live_available)`; both values
  and `synthetic_observation_capped_by_live_ram` are recorded.
- The 1 GiB cap is watched **during** the run; exceeding it kills the owned
  process tree and yields `CAP_EXCEEDED` (exit 4), never `PASS`.
- Child stdout/stderr is streamed to `child.log`, not buffered unbounded.
- Timeout (≤300 s) kills the owned process tree and yields `TIMEOUT` (exit 3).
- No persistent service/task; no installed-product edit; original data is never
  deleted.

## Artifact cap

The hard cap is **1 GiB of new artifacts per invocation**. The preflight floor
is conservative, not a measured peak:

- FITS frames at 4 bytes/pixel (float32);
- support planes, bounded cache, canvas/output planes, logs/diagnostics;
- current estimate ≈ 100 MiB (the function returns the exact number).

A run is refused if `--max-new-bytes` is below that estimate. The cap is
live-watched and checked post-run; status and exit stay coherent.

## Why the witness tile is 409 while M31 used tile 32

M31 had a 17×11 grid and a large multi-cell canvas; its admitted ladder bottomed
out at tile 32 with prefetch 0. This M0 corpus is 8 × 512 px with a pinned 2×2
layout and a deliberately small canvas. The admitted VRAM/host ladder therefore
resolves to tile 409. The witness reproduces the **host-RAM discard and CPU
re-admission failure**, not M31's exact tile/prefetch geometry. Do not compare
tile numbers as if the geometry were identical.

## Run

```powershell
.\run_m0_windows_witness.ps1 `
  -SrcRoot "C:\path\to\worktree\src" `
  -InstalledModule "C:\path\to\installed\zemosaic_zegrid_mode.py" `
  -OutputRoot "C:\Users\JarvisJunior\Desktop\zgprobe\m0_witness_01"
```

Optional: `-InstalledSrcRoot` for dependency-tree provenance equality; otherwise
only the main module hash is compared, while all actual dependency hashes are
recorded for audit.

Exit codes:

- `0` — expected baseline failure reproduced (all gates passed; this is NOT
  product success);
- `1` — unexpected failure / child nonzero / no witness / classification wrong;
- `2` — NOT_RUN (no GPU or live RAM below the witness envelope);
- `3` — TIMEOUT (owned process tree killed);
- `4` — artifact cap exceeded (owned process tree killed; never PASS).
