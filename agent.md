# Mission — SDS Color Safety: Enforce CPU/GPU Backend Consistency

## Context
A critical color regression (green tint) was fixed in legacy mode by identifying
a CPU/GPU backend split during coverage-based renormalization (Two-Pass).
Although SDS (grid / SDS mode) mainly applies normalization *between images*
(e.g. linear_fit), similar risks exist whenever SDS executes:

- statistical operations (coadd, rejection, weighting),
- coverage / footprint usage,
- reprojection with possible GPU fallback.

Even when normalization is conceptually “between images”, backend mixing
(CPU for some images/channels, GPU for others) can introduce subtle numerical
differences leading to chromatic drift.

This mission ensures SDS is **architecturally protected** against such issues.

## Scope
SDS pipeline only (grid / SDS code paths).
Legacy / classic mode is explicitly out of scope (already fixed).

## Key Rule (Invariant)
For any SDS processing step that:
- depends on statistics (mean, sigma-clip, winsorized, weights),
- or uses coverage / footprint / weight_sum,
- or applies normalization (e.g. linear_fit),

👉 **ALL RGB channels and ALL images involved in that step MUST use the same backend**:
- either CPU for all,
- or GPU for all.

Backend mixing within a single step is strictly forbidden.

## Required Changes

### 1) Backend Policy: “All-or-Nothing” (SDS)
For each SDS processing stage that can run on GPU:
- If `use_gpu=True`:
  - Attempt GPU for the entire stage.
  - If any GPU error or fallback occurs:
    - Abort the stage.
    - Log a single warning.
    - Rerun the stage entirely on CPU.
- Never allow partial GPU usage (no per-image or per-channel split).

This policy must be enforced even if GPU and CPU implementations are
mathematically equivalent.

### 2) Explicit Backend Logging (Low Noise)
Add **one concise log line per SDS stage** (INFO or DEBUG, existing logger only):

- `[SDS] stage=<name> backend_policy=all_or_nothing gpu_all=True`
- or, on fallback:
  - `[SDS] stage=<name> GPU failed → rerun all CPU`

No new logging system must be introduced.
Respect the existing GUI log-level dropdown.

### 3) No Algorithmic Changes
- Do NOT change the math of:
  - linear_fit normalization,
  - rejection algorithms,
  - weighting or stacking logic.
- This mission is about **backend coherence**, not algorithm tuning.

## Non-goals / Constraints
- Do NOT touch legacy/classic mode.
- Do NOT change batch-size semantics (batch size = 0 vs >1).
- Do NOT introduce new config options or UI controls.
- Do NOT add new normalization steps.

## Acceptance Criteria
- SDS never mixes CPU and GPU within a single processing stage.
- Logs clearly indicate which backend was used for each SDS stage.
- No color drift or green tint appears in SDS outputs.
- CPU-only runs remain unchanged.
- GPU-enabled runs either succeed fully on GPU or cleanly fallback to CPU.

## Deliverables
- Minimal code changes enforcing the backend invariant.
- Tight diffs, no refactors.
