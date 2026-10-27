# M1 — bounded Windows probe (separate source tree only)

This is the M1 runtime-conservation probe for a Windows machine. It runs the
repository tests from an explicit **source worktree**, never from an installed
ZeMosaic runtime, and writes only under a fresh `zgprobe` directory.

## What it proves

- `tests/test_zegrid_run_registry_m1.py` — ownership/transaction semantics:
  complete payload fingerprints (core + per-cell halo patch + reference pin),
  sparse IDs, idempotent duplicate, conflicting duplicate rejection, stale
  generation, empty validated holes, staging failure, bounded journal,
  explicit disposal, stop request/ack (thread and spawn-context events).
- `tests/test_zegrid_resume_m1_conservation.py` — the REAL orchestration path
  with real cell computations and injected synthetic memory pressure after two
  accepted cells: accepted IDs and full payload hashes survive the refusal,
  sparse pending IDs are enumerable, no CPU reuse, no publication, explicit
  disposal; plus a cooperative-stop variant.

## What it does NOT prove

- No GPU qualification. The conservation test substitutes the GPU device
  executor with the CPU canonical executor (`CONTROL_ONLY_FAKE_GPU`), exactly
  like the M0 control witness. The harness labels itself
  `WINDOWS_M1_CONTROL_ONLY_CPU / NOT_GPU_QUALIFIED`.
- It does not compare against the installed product runtime. If you also run
  the same probe on the installed baseline, label that result
  `NOT_VERIFIED_INSTALLED_RUNTIME`; do not claim installed-runtime provenance.
- No durable save/restart. M1 conservation is in-memory during one run.

## Requirements

- Python 3.11+ with the source tree's dependencies (`numpy`, `astropy`,
  `psutil`, `pytest`; no CuPy needed for this probe).
- The worktree containing this file and `src/`.
- No installed ZeMosaic files are modified or imported.

## Command (bounded, fail-closed)

```powershell
powershell -ExecutionPolicy Bypass -File `
  C:\path\to\worktree\tools\zegrid_resume_m1\run_m1_windows_probe.ps1 `
  -SourceRoot C:\path\to\worktree `
  -OutputRoot C:\Users\JarvisJunior\Desktop\zgprobe\m1_probe
```

The script writes `probe_summary.json` and `pytest.log` in `OutputRoot`,
returns exit 0 only for `status=PASS`, and returns 2 for `NOT_RUN`
(missing pytest/source), 3 for `FAIL`, 4 for timeout. Maximum run time is
900 seconds.

## Linux equivalent (used for the M1 report)

```bash
python3 -m pytest -q -p no:cacheprovider \
  tests/test_zegrid_run_registry_m1.py \
  tests/test_zegrid_resume_m1_conservation.py
```

No-pressure science/coverage identity control (baseline vs M1 source):

```bash
python3 tools/zegrid_resume_m1/m1_no_pressure_control.py \
  --src-root <baseline-or-m1>/src \
  --output-root /tmp/m1_np \
  --json-out /tmp/m1_np.json
```
