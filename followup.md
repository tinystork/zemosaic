# Follow-up — SDS CPU/GPU Consistency & Color Safety

## 0) Guardrails
- [x] Do not modify legacy/classic mode.
- [x] Do not alter normalization algorithms or math.
- [x] Do not add new logging handlers or files.
- [x] Respect existing GUI log-level control.

---

## 1) Identify SDS Critical Stages
Search SDS / grid code for stages that:
- [x] call `reproject_and_coadd` or GPU equivalents
- [ ] compute statistics (mean, sigma, weights, coverage)
- [ ] apply `linear_fit` or similar normalization

Document stage names internally (no user-facing docs needed).

Identified stage:
- `sds_global_coadd` — `_assemble_global_mosaic_first_impl` (GPU helper path wraps `reproject_and_coadd`).

---

## 2) Enforce All-or-Nothing Backend Policy
For each identified SDS stage:

- [ ] Ensure backend selection happens **once per stage**, not per image/channel.
- [ ] If GPU is enabled:
  - [ ] Attempt full-stage GPU execution.
  - [ ] On any GPU failure:
    - [ ] Abort stage
    - [ ] Log single warning
    - [ ] Rerun entire stage on CPU
- [ ] Remove/avoid any logic that enables partial GPU usage.

---

## 3) Logging Verification
For each SDS stage:

- [ ] One log line indicates backend used:
  - `gpu_all=True` or `gpu_all=False`
- [ ] Fallback message appears only once per stage.
- [ ] No log spam.

---

## 4) Quick Validation Run
- [ ] Run SDS with GPU enabled.
- [ ] Confirm logs show consistent backend per stage.
- [ ] Verify final mosaic has no chromatic drift.
- [ ] Run SDS CPU-only as control (result unchanged).

---

## 5) Done Definition
- [ ] SDS pipeline is backend-coherent by construction.
- [ ] No possibility of silent CPU/GPU mixing remains.
- [ ] Color stability is guaranteed long-term.
