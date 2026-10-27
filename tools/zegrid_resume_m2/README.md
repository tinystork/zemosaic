# M2 — durable run store bench (source tree only)

Fail-closed bench for the M2 durability milestone: per-cell atomic commits, a
reconstructible index, a single-writer OS lock, sha256 integrity, a disk budget
and the release of large finished arrays after commit.

## What it runs (real OS primitives, source worktree only)

| Suite | File | Covers |
|---|---|---|
| `m2_store_contract` | `tests/test_zegrid_run_store_m2.py` | identity/jauge, atomic publication, idempotence `(cell_id, attempt_id)`, reconstructible index, isolated corruption, lock (live/orphan/concurrent), disk budget + ENOSPC injection, sparse completeness, release hand-off |
| `m2_kill_matrix` | `tests/test_zegrid_run_store_m2_kill.py` | a child process is KILLED during each of the four write steps (temp write, fsync, commit rename, index), 5 iterations each: only complete commits are recoverable |
| `m2_real_store_run` | `tests/test_zegrid_resume_m2_store_run.py` | REAL `run_zegrid_mode` CPU path with the store enabled: commit, RAM release, byte-identical reload, unchanged publication |
| `m1_registry` | `tests/test_zegrid_run_registry_m1.py` | M1 non-regression (ownership/transaction/generation contract) |
| `m1_conservation` | `tests/test_zegrid_resume_m1_conservation.py` | M1 non-regression (real orchestration conservation) |
| `m0_witness` | `tests/test_zegrid_resume_m0_baseline_witness.py` | M0 non-regression (baseline witness) |
| `m0_harness` | `tests/test_zegrid_resume_m0_windows_harness.py` | M0 non-regression (fail-closed harness) |

## Verdict (fail-closed)

`status=PASS` requires zero failures, zero errors and **zero skips**, plus the
full 20-case kill matrix and >= 35 store-contract cases. Any skip => `NOT_RUN`.
Timeout => `TIMEOUT`. Artifact cap => `CAP_EXCEEDED`. Guard refusal => exit 6.

Exit codes: `0 PASS`, `2 NOT_RUN`, `3 FAIL`, `4 TIMEOUT`, `5 CAP_EXCEEDED`,
`6 GUARD`.

## Windows

```powershell
powershell -ExecutionPolicy Bypass -File `
  C:\path\to\worktree\tools\zegrid_resume_m2\run_m2_bench.ps1 `
  -SourceRoot C:\path\to\worktree `
  -OutputRoot C:\Users\JarvisJunior\Desktop\zgprobe\m2_bench
```

Requirements: Python 3.11+ with `numpy`, `astropy`, `psutil`, `pytest`; the
worktree containing this file and `src/`. No CuPy and no GPU are used.

On Windows the kill matrix kills the child through `taskkill /T /F`
(`TerminateProcess`) and the single-writer lock uses `msvcrt.locking`; on Linux
the same code path uses `SIGKILL` and `flock`. Both are real OS primitives —
this is not a simulated kill.

## Linux

```bash
python3 tools/zegrid_resume_m2/m2_bench.py \
  --source-root "$PWD" \
  --output-root /tmp/zgprobe/m2_bench \
  --zgprobe-root /tmp/zgprobe
```

## What it does NOT prove

* No run restore/resume engine (M4), no resource wait/backoff or generations
  (M3), no bounded assembly/offsets (M5).
* No GPU science/parity qualification (M7) and no installed-runtime provenance.
* rename+fsync ordering is not a power-loss guarantee: the bench kills a
  process, it does not cut power.
* A skipped RAM-gated real-run case is reported as NOT_RUN, never as PASS.
