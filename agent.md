# agent.md

# ZeMosaic — Mission Codex
## Phase 3 adaptive RAM control without dropping raw frames

Date: 2026-03-13
Owner: Tristan / ZeMosaic core
Mission mode: design-first / proof-driven / surgical

## Mission objective

Design and implement a **Phase 3 adaptive memory-control mechanism** that:

- preserves the **original scientific content** of each master tile
- keeps **all raw frames** assigned to a logical master tile
- prefers **parallel execution** when resources allow it
- targets about **80% RAM usage** during Phase 3
- reduces **effective memory pressure** before reducing scientific depth

This mission is explicitly **not** about discarding frames to fit memory.

The priority order is locked:

1. keep all raw frames
2. reduce effective Phase 3 concurrency if RAM pressure rises
3. reduce per-task working-set size through streaming/chunking
4. only if strictly required, split execution into sub-passes that still preserve all raw inputs

The Qt/Tk retirement mission is considered completed for this checkpoint and is no longer the active mission.

---

## Non-negotiable execution rules

1. **Never trade image quality for memory by silently dropping raw frames.**
2. **Treat a master tile's raw membership as scientifically fixed unless the user explicitly asks otherwise.**
3. **Prefer adaptive scheduling and out-of-core strategies over dataset reduction.**
4. **Work only on the next unchecked item.**
5. **Keep diffs surgical and local to Phase 3 behavior unless a dependency requires otherwise.**
6. **Do not refactor unrelated parts of the pipeline.**
7. **Always update `memory.md` after each meaningful iteration.**
8. **Always mark completed items with `[x]` in both `agent.md` and `followup.md`.**
9. **Never declare a step complete without proof.**
10. **If an exact all-frames method is impossible for a rejection mode, document that explicitly before changing behavior.**

---

## Scope

### In scope for this mission
- Phase 2 to Phase 3 handoff when it affects Phase 3 memory behavior
- Phase 3 scheduler behavior
- Phase 3 runtime RAM adaptation
- master-tile submission strategy
- per-tile dynamic chunk sizing
- per-tile dynamic frame-pass sizing
- telemetry needed to support and validate adaptation
- tests covering adaptive RAM behavior
- documentation in `memory.md`, `agent.md`, and `followup.md`

### Explicitly out of scope for this mission
- reducing image quality by dropping frames
- changing scientific grouping rules for mosaic coverage
- unrelated GUI work
- Qt/Tk migration follow-up
- broad cleanup of Phase 4/5 unless strictly required by shared utilities
- changing combine/rejection math without necessity and proof

---

## Core architectural rule

For this mission, distinguish strictly between:

- **logical stack size**: number of raw frames scientifically belonging to a master tile
- **working-set size**: number of frames/rows resident in RAM at once

The mission may reduce only the **working-set size** and/or **active parallelism**.
It must preserve the **logical stack size**.

## Invariant test contract (must remain true)

For every logical master tile, adaptive behavior must preserve scientific membership exactly:

- `logical_input_count_before == logical_input_count_after`
- `logical_input_frame_ids_before == logical_input_frame_ids_after`
- zero tolerance for silent raw-frame dropping in adaptive mode

These checks are release-gating invariants, not best-effort goals.


---

## Current understanding to validate and refine

Current code already contains partial building blocks:

- Phase 2 can auto-split groups by a frame limit derived from memory heuristics
- Phase 3 can cap worker count from RAM heuristics
- Phase 3 can adapt effective concurrency at runtime based on CPU and I/O pressure
- telemetry already exposes RAM availability

What is still missing is the full adaptive loop that:

- reacts to **runtime RAM pressure**
- limits how many master tiles are started
- recomputes per-task memory budgets dynamically
- preserves full logical-stack frame membership in the tile
- avoids eager submission of all Phase 3 tasks at once

Do not assume the current implementation already satisfies that contract until audited.

## Exactness policy by combine/rejection mode

Before enabling adaptive behavior per mode, classify exactness explicitly:

| Mode | Exact streaming/chunked implementation available? | Approximate fallback allowed by default? | Mandatory logging when adaptation applies |
|---|---|---|---|
| mean / weighted mean | yes (expected) | no | yes |
| median | to validate explicitly | no | yes |
| winsorized sigma clip | to validate explicitly | no | yes |
| other active rejection modes | to validate explicitly | no | yes |

Rules:
- no approximation is enabled silently
- if exactness is not proven for a mode, keep conservative behavior and document limitation first
- any mode-specific limitation must be recorded before changing runtime behavior


---

## Required design direction

The preferred implementation direction is:

1. **lazy Phase 3 scheduling**
2. **runtime RAM controller with hysteresis**
3. **adaptive per-tile working-set sizing**
4. **logical-stack-preserving execution**

That means:

- do not drop frames
- do not silently shrink scientific tiles
- do not rely only on a static pre-phase memory estimate
- do not rely only on semaphores if tasks are already all submitted


## Default controller profile (initial tuning, adjustable)

Unless overridden by explicit runtime configuration, start from:

- RAM target: `80%`
- high-pressure threshold: `82%`
- recovery threshold: `72%`
- minimum adaptation cooldown: `10s`
- maximum adaptation-level changes: `6 / minute`

These are starting defaults for stable behavior on hybrid laptops and must remain configurable.

---

## RAM pressure signal source (single decision surface)

Adaptation decisions must be driven by one explicit decision surface, documented and testable:

- primary signal: system memory pressure (`used_percent`)
- supporting signals: available RAM bytes and swap activity (if present)
- process-local memory may be logged for diagnostics, but controller decisions must remain consistent with the chosen primary signal

Do not mix incompatible RAM definitions silently across code paths.

## Proof requirements

Each completed step must include proof such as:

- code-path audit with file/line references
- logs showing runtime adaptation decisions
- tests proving no raw-frame loss in adaptive mode
- tests proving RAM backpressure reduces concurrency before scientific scope
- tests proving chunk/pass reductions preserve logical input membership
- before/after notes in `memory.md`

If a claim depends on a limitation of a combine/rejection method, document that limitation explicitly.

---

## Ordered degradation policy (operational pseudocode)

```
if RAM > high_threshold:
  1) reduce future active launches (lazy scheduler budget)
  2) reduce per-pass frames (working-set shrink)
  3) reduce rows/chunk size (working-set shrink)
  4) if special path requires, serialize that path narrowly

if RAM remains high after minimum limits reached:
  - stop admitting new Phase 3 launches temporarily
  - keep in-flight tasks safe and observable
  - emit explicit pressure alert/telemetry event

if RAM < recovery_threshold and cooldown elapsed:
  - gradually restore chunk/pass/launch budgets
```

The order is mandatory unless an explicit mode limitation is documented first.

## Phase execution contract

## Glossary (mission-local terms)

- **logical stack**: full set of raw frames scientifically assigned to a master tile
- **working-set**: subset of data resident in RAM at one instant (passes/chunks)
- **active launch**: Phase 3 tile job admitted for execution
- **tile pass**: one processing pass over a subset of tile frames

### [x] S0 — Baseline audit and invariant lock
Goal:
- lock the scientific invariant: **all raw frames preserved**
- audit how Phase 2 and Phase 3 currently control memory
- identify where current logic can violate the intended invariant

Expected outputs:
- exact map of current Phase 3 memory levers
- explicit distinction between:
  - concurrency limiting
  - chunk limiting
  - frame-pass limiting
  - group splitting
- explicit invariant statement recorded in `memory.md`

Hard prohibitions:
- no implementation yet
- no silent behavior change

### [x] S1 — Adaptive strategy design
Goal:
- define the target adaptive controller before editing code

Expected outputs:
- control strategy for RAM target around 80%
- hysteresis thresholds
- lazy scheduling plan
- decision tree for:
  - reduce active tiles
  - reduce per-pass frames
  - reduce rows per chunk
  - serialize GPU usage if relevant
- exact policy for preserving all raw frames

Hard prohibition:
- do not implement heuristics before they are written down

### [x] S2 — Scheduler refactor for Phase 3 launch control
Goal:
- stop eager launch behavior from undermining runtime adaptation

Required work:
- replace eager all-at-once submission with lazy scheduling
- control how many master-tile jobs are launched
- preserve existing retry behavior
- preserve progress reporting semantics as much as practical

Mandatory gate:
- runtime must be able to reduce future task launches without changing logical tile content

### [x] S3 — Per-tile adaptive working-set control
Goal:
- make each Phase 3 task adapt its RAM footprint while preserving all frames

Required work:
- dynamic per-tile `winsor_max_frames_per_pass` policy
- dynamic per-tile chunk/row policy
- explicit handling for exact vs non-exact rejection/combine modes
- ensure all raw frames remain part of the logical stack

Mandatory gate:
- adaptation changes only working-set size, not logical membership

### [x] S4 — Runtime RAM controller and telemetry
Goal:
- use observed RAM pressure, not only static estimates

Required work:
- periodic RAM sampling
- hysteresis-based controller
- structured logs / telemetry fields for adaptation decisions
- optional dedicated GPU concurrency policy if Phase 3 GPU path needs serialization

Mandatory gate:
- logs/telemetry make adaptation decisions explainable

### [x] S5 — Validation and non-regression tests
Goal:
- prove the adaptive design preserves quality intent while reducing memory risk

Required work:
- tests for no raw-frame loss
- tests for concurrency backoff under RAM pressure
- tests for pass/chunk shrink under RAM pressure
- tests for stable recovery when pressure drops
- tests for retry path compatibility

Release gate for this mission:
1. no silent raw-frame dropping
2. Phase 3 can reduce active launches under RAM pressure
3. per-tile working-set adapts dynamically
4. telemetry/logs explain adaptation
5. tests cover the invariant

### [ ] S6 — Optional refinement
Not part of the base mission unless needed after S5.

Possible later refinements:
- smarter predictive memory model
- per-mode exact streaming optimization
- dedicated GPU semaphore for Phase 3



---

## Active add-on mission (2026-03-13) — Phase 5 slowdown after intertile anchor event

Context trigger:
- During real runs, after log event
  - `[Intertile] Anchor selection biased: anchor=...`
  users observe a strong slowdown and suspect GPU is no longer used.

Mission objective:
- Diagnose and optimize the post-intertile Phase 5 path, with focus on:
  1. intertile calibration execution mode (single-worker fallback vs parallel fallback)
  2. Two-Pass coverage renorm performance (`stage=gains`)
  3. explicit GPU/CPU activity attribution in telemetry and logs
- Preserve scientific output behavior unless explicitly approved otherwise.

Non-negotiable rules for this add-on mission:
1. Do not disable intertile correction globally as a speed shortcut.
2. Do not reduce scientific tile membership or output fidelity to gain speed.
3. Keep changes surgical to intertile / Phase 5 / two-pass performance paths.
4. Every optimization must include before/after proof in `memory.md`.
5. If a fallback is required, prefer safe multi-worker thread fallback over forced single-worker when correctness allows.

Required deliverables:
- Root-cause map of where time is spent after anchor selection.
- Clear classification: GPU-active stages vs CPU-bound stages.
- Optimization patch(es) for the dominant bottleneck(s), starting with:
  - intertile fallback mode,
  - two-pass gains computation path.
- Updated telemetry keys/messages making the runtime state unambiguous.
- Non-regression tests (correctness + behavioral contract) and benchmark notes.
