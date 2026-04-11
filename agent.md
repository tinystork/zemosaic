# agent.md — ZeMosaic mission brief (aesthetic-first, speed-safe)

## Product objective (updated)
Deliver a mosaic output that is:
1. **visually smooth and homogeneous** (background + nebulosity, minimal seams/patches/holes),
2. **easy to edit** in Siril / PixInsight / Seti Astro Suite,
3. while preserving ZeMosaic’s core advantage: **high-throughput processing of very large datasets**.

This mission explicitly prioritizes aesthetic usability for most users, **without sacrificing** scientific integrity or pipeline speed.

---

## Output contract (mandatory)

### Output A — Scientific
- canonical FITS, physically faithful as much as possible,
- never silently altered by aesthetic-only operations,
- remains the reference output.

### Output B — Aesthetic
- dedicated visually-optimized FITS,
- may include seam suppression + local hole-fill/inpainting,
- intended for downstream editing workflows.

Naming recommendation:
- `*_science.fits`
- `*_aesthetic.fits`

---

## Implemented foundations (already done)

### A) Patchwork suppressor
- config + pipeline hook + logs (`[Patchwork]`) available.

### B) Underconstrained intertile guardrail
- sparse-graph detection + safe fallback + logs (`[IntertileGuard]`) available.

These remain active as baseline protections.

---

## New Objective C — Dual FITS export
Status: **in progress (core runtime implemented)**

### Requirements
- [ ] Add `export_aesthetic_fits` switch.
- [ ] Export both science and aesthetic FITS when enabled.
- [ ] Guarantee no overwrite ambiguity (safe filenames).
- [ ] Add metadata/header tags identifying branch + parameters.

Suggested keys:
```json
"export_aesthetic_fits": false,
"scientific_fits_suffix": "_science",
"aesthetic_fits_suffix": "_aesthetic"
```

---

## New Objective D — Aesthetic hole-fill / seam completion
Status: **in progress (core runtime implemented)**

Purpose:
- remove residual “holes” and patch discontinuities in visual branch,
- keep edits local and low-frequency aware,
- avoid destructive star/core smearing.

### V1 behavior (aesthetic branch only)
- detect invalid/near-invalid holes from coverage/alpha maps,
- fill locally using seam-aware inpainting / low-frequency completion,
- feather transitions to avoid hard patches,
- protect compact high-frequency structures (stars, sharp filaments).

Suggested keys:
```json
"aesthetic_hole_fill_enabled": true,
"aesthetic_hole_fill_max_radius_px": 64,
"aesthetic_hole_fill_blend": 0.7,
"aesthetic_hole_fill_only_near_seams": true
```

---

## Throughput constraint (non-negotiable)

Any new aesthetic module must respect large-scale throughput:
- O(N) / tiled-friendly memory behavior,
- no heavy global optimization loops in default mode,
- one-pass or bounded multi-pass,
- optional stronger mode allowed only as explicit opt-in.

Default should stay **fast + robust**.

---

## Validation matrix (must complete)

### Sparse pathological case (existing master tiles)
- [ ] baseline (A/B off)
- [ ] A+B on
- [ ] A+B+C (dual export)
- [ ] A+B+C+D (with hole-fill)

### Dense normal case
- [ ] regression check runtime
- [ ] regression check visual integrity

### Performance checks
- [ ] runtime overhead delta (%)
- [ ] peak RAM delta
- [ ] output size delta

---

## Acceptance criteria

### Visual (aesthetic branch)
- visibly reduced seams/patches,
- problematic holes substantially reduced or visually neutralized,
- output judged “ready to edit” in Siril/PixInsight/Seti Astro Suite.

### Integrity (science branch)
- scientific FITS unchanged in semantics vs baseline branch.

### Performance
- no unacceptable slowdown on large runs,
- throughput profile remains compatible with thousands-of-frames workflows.

---

## Priority order
1. Objective C (dual FITS export)
2. Objective D (hole-fill visual completion)
3. tuning presets (Balanced / Strong)
4. finalize default profile for production.

---

## New mission addendum (2026-04-10) — Markarian stability/performance track

### Immediate priority (before memory/chunk tuning)
Fix ETA behavior in GUI so runtime estimates remain trustworthy during long runs.

Rationale:
- Current Markarian stress runs show ETA drift/jumps and poor user trust in progress forecasting.
- Any throughput optimization (dynamic chunking, memory scaling) must be measured against a reliable ETA baseline.

### Priority sequence (updated)
1. **ETA reliability in GUI** (first, mandatory)
2. Dynamic auto-chunking SAFE mode (RAM/VRAM adaptive with guardrails)
3. Throughput tuning validation (A/B vs baseline)
4. Production default decision

### ETA reliability requirements
- Stable ETA smoothing (avoid oscillation/yo-yo).
- Distinguish per-phase ETA vs global ETA.
- Re-baseline ETA when entering a new phase (P3/P5 especially).
- Avoid reporting unrealistically short ETA early in long phases.
- Keep logs that explain ETA updates (`source`, `window`, `confidence`).

### Hard safety constraints (non-negotiable)
- Preserve ability to process very large datasets (target up to ~10k images) without OOM.
- Keep adaptive memory reserves for system stability (do not consume all RAM/VRAM).
- Any aggressive tuning must remain opt-in.

### Acceptance criteria for this addendum
- ETA in GUI remains coherent and usable on long Markarian-like runs.
- No regression in run stability.
- Then only: start SAFE dynamic chunking rollout.

---

## Dedicated mission (new, 2026-04-10 evening) — Progressive Resume + Crash Recovery

### Product decision
This is now a dedicated mission, not a side-task.

Goal: make ZeMosaic resumable by stages, tolerant to partial losses, and robust after interruption/crash.

### Scope (target behavior)
1. Phase 1 partial reuse (rebuild only missing/invalid cache entries).
2. Reuse compatible stack/group plan when possible.
3. Reuse valid master tiles; rebuild only missing/corrupted/obsolete tiles.
4. Resume from highest valid level (final > masters > plan > phase1 partial > full restart).
5. Add explicit user actions: **Quit and Save Progress** (priority) and **Pause** (second step).

### Architecture baseline
State folder (proposed): `.zemosaic_state/`
- `run_state.json`
- `phase1_manifest.json`
- `stack_plan_manifest.json`
- `master_tiles_manifest.json`
- `final_assembly_manifest.json`

Each manifest must answer:
- artifact exists?
- artifact compatible with current run signature?

### Non-negotiable rules
- Never invalidate all cache for one missing file.
- Invalidate locally, not globally.
- Use atomic writes for manifests.
- Never mark a stage done before outputs are fully written + validated.

### Execution plan and tracking
- [ ] Design note (short) for run signature, manifests, validation, invalidation, atomicity, partial resume paths.
- [x] Refactor `_try_resume_phase1(...)` from all-or-nothing to partial rebuild mode.
- [x] Add official stack-plan reuse with compatibility validation.
- [x] Add official master-tiles reuse with per-tile validation.
- [x] Add `Quit and Save Progress` (clean stop + flush manifests/state).
- [ ] Add `Pause` behavior (safe suspension semantics).
- [ ] Add resume logs (`phase1 partial reuse X/Y`, `plan reused`, `master reused A/B`, `rebuild tiles [...]`).
- [ ] Add interrupted-run scenarios/tests (phase1 hole, after grouping, mid-phase3, battery/quit-save, local corruption).

### Completion definition
Mission completes when partial resume works by stage on large datasets without restarting from zero after local cache loss.


## Crash-track addendum (2026-04-11/12) — Intertile fail-fast on Windows

New forensic evidence collected from crash artifacts:
- Python faulthandler stack lands in intertile overlap calibration path (`_process_overlap_pair` -> `reproject_interp` -> `astropy.wcs` transforms).
- Windows dump indicates fast-fail style termination (`0xC0000409`) with crash context in `PySide6/Qt6Core` for the supervising process.

Operational observation:
- `intertile_preview_size=256` still crashed on Markarian D.
- `intertile_preview_size=128` appears stable on the same workload.

Decision direction:
- Prioritize an adaptive intertile preview guardrail with risk-based tier selection and fallback ladder (`512 -> 256 -> 128`) on failure.
- Keep this on the crash-track critical path for dense overlap datasets.

