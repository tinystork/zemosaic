# followup.md

# ZeMosaic — Follow-up checklist
## Phase 3 adaptive RAM control without dropping raw frames

Legend:
- `[ ]` not done
- `[x]` done
- `[~]` partially done / needs follow-up
- `BLOCKED:` explain precisely why

---

## A. Mission discipline

- [x] Read `agent.md`, `followup.md`, and `memory.md` before editing
- [x] Replace previous mission plan with the new active mission
- [x] Record in `memory.md` that the previous mission is considered completed
- [x] Work only on the next unchecked item
- [x] Keep changes surgical
- [x] Do not refactor unrelated code
- [x] Update `memory.md` at the end of each meaningful iteration
- [x] Mark completed items in both `agent.md` and `followup.md`

---

## B. S0 — Baseline audit and invariant lock

### B1. Scientific invariant
- [x] Write the invariant explicitly:
  - [x] all raw frames assigned to a logical master tile must remain part of that tile's logical stack
  - [x] adaptation may reduce only working-set size and/or active parallelism
  - [x] no silent quality downgrade via frame dropping

### B2. Current Phase 2 behavior
- [x] Audit current group splitting in Phase 2
- [x] Determine whether current auto-limit logic changes logical master-tile content
- [x] Record exactly when Phase 2 splitting is acceptable vs not acceptable for this mission

### B3. Current Phase 3 behavior
- [x] Audit current Phase 3 worker selection
- [x] Audit current runtime semaphore adaptation
- [x] Audit current eager task submission behavior
- [x] Audit current per-tile memory levers:
  - [x] `winsor_max_frames_per_pass`
  - [x] row/chunk sizing
  - [x] memmap/streaming flags
  - [x] GPU budget interactions if relevant

### B4. Baseline proof
- [x] Record baseline findings in `memory.md`
- [x] Mark S0 complete only when the invariant and current gaps are explicit

---

## C. S1 — Adaptive strategy design

### C1. Controller policy
- [x] Define target RAM policy around 80%
- [x] Define hysteresis thresholds
- [x] Define escalation order:
  - [x] reduce active Phase 3 launches
  - [x] reduce per-pass frames
  - [x] reduce rows per chunk
  - [x] serialize special paths if required

### C2. Scheduler policy
- [x] Decide lazy scheduling model for Phase 3
- [x] Decide queue semantics for pending groups
- [x] Decide how retries re-enter the queue

### C3. Scientific preservation policy
- [x] Define what "preserve all raw frames" means for:
  - [x] mean / weighted mean
  - [x] median
  - [x] winsorized sigma clip
  - [x] other active rejection modes
- [x] Explicitly document any exactness limitations

### C4. Telemetry policy
- [x] Define which adaptation decisions must be logged
- [x] Define which telemetry fields must be emitted

### C5. Design sign-off
- [x] Record the target design in `memory.md`
- [x] Mark S1 complete only when the decision tree is explicit

---

## D. S2 — Scheduler refactor for Phase 3 launch control

### D1. Replace eager launch
- [x] Stop submitting all Phase 3 groups immediately
- [x] Introduce a pending queue / lazy launcher
- [x] Preserve existing completion/progress behavior as closely as practical

### D2. Adaptive launch budget
- [x] Add a controllable active-launch limit
- [x] Ensure the limit can shrink at runtime for not-yet-started work
- [x] Ensure the limit can grow again when memory pressure drops

### D3. Retry compatibility
- [x] Preserve retry-group scheduling
- [x] Ensure retries do not bypass launch control

### D4. Proof
- [x] Record scheduler behavior proof in `memory.md`
- [x] Mark S2 complete only when runtime launch throttling is real, not only nominal

---

## E. S3 — Per-tile adaptive working-set control

### E1. Frame-pass adaptation
- [x] Add dynamic per-tile `winsor_max_frames_per_pass` logic
- [x] Ensure it reduces RAM footprint without dropping raw membership

### E2. Chunk adaptation
- [x] Add dynamic per-tile row/chunk sizing
- [x] Ensure chunk adaptation reacts to actual tile characteristics

### E3. Exactness review
- [x] Verify whether each combine/rejection mode remains exact under the chosen adaptation
- [x] If not exact, document the limitation before enabling the mode

### E4. Proof
- [x] Record per-tile adaptation proof in `memory.md`
- [x] Mark S3 complete only when logical frame membership remains unchanged

---

## F. S4 — Runtime RAM controller and telemetry

### F1. Runtime RAM sampling
- [x] Add or reuse periodic RAM sampling for Phase 3 decisions
- [x] Base decisions on observed pressure, not only static startup estimates
- [x] Define and document the primary RAM pressure signal used for decisions

### F2. Hysteresis and stability
- [x] Avoid oscillation with hysteresis / cooldown rules
- [x] Define minimum cooldown between adaptation level changes
- [x] Define explicit high/low thresholds (hysteresis gap)
- [x] Define maximum adaptation-level changes per minute
- [x] Verify recovery behavior when RAM pressure decreases
- [x] Define behavior when minimum limits are reached but pressure remains high (admission pause + explicit alert)

### F3. Telemetry and logs
- [x] Emit structured logs for adaptation actions
- [x] Emit telemetry fields needed for later validation

### F4. Optional GPU policy
- [x] Decide whether Phase 3 needs a dedicated GPU concurrency guard
- [x] If yes, implement and document it narrowly

### F5. Proof
- [x] Record runtime-controller proof in `memory.md`
- [x] Mark S4 complete only when decisions are observable and explainable

---

## G. S5 — Validation and non-regression tests

### G1. Invariant tests
- [x] Add tests proving no silent raw-frame dropping
- [x] Add tests proving logical stack membership is preserved under adaptation

### G2. Pressure-response tests
- [x] Add tests proving active-launch backoff under RAM pressure
- [x] Add tests proving per-pass/chunk shrink under RAM pressure
- [x] Add tests proving recovery when pressure drops

### G3. Compatibility tests
- [x] Add tests for retry-path compatibility
- [x] Add tests for telemetry/log integrity
- [x] Add tests for affected combine/rejection modes

### G3b. Critical non-regression gates
- [x] Same number of scientific Phase 3 outputs as baseline for identical dataset/config (excluding temporary files/logs)
- [x] No silent change of combine/rejection mode
- [x] Retry path preserves logical stack membership
- [x] Adaptation logs are present and machine-parseable with stable keys
- [x] No adaptation thrashing beyond defined limit on stable-pressure scenario

### G4. Mission gate
- [x] no silent raw-frame dropping
- [x] runtime launch throttling works
- [x] working-set adaptation works
- [x] telemetry/logging is sufficient
- [x] tests cover the invariant

### G5. Closeout
- [x] Record final proof in `memory.md`
- [x] Mark mission status as GO or NO-GO for merge

---

## H. Required per-iteration report format

At the end of each iteration, append to `memory.md`:

### YYYY-MM-DD HH:MM — Iteration N
- Scope:
- In scope:
- Out of scope:
- Files changed:
- Tests run:
- Proof:
- Decisions:
- Blockers:
- Next unchecked item:

Mandatory:
- mention whether the all-raw-frames invariant changed or stayed unchanged
- mention whether Phase 3 launch control changed or stayed unchanged
- mention whether working-set adaptation changed or stayed unchanged



---

## I. Add-on mission — Phase 5 slowdown after intertile anchor event

### I1. Reproduction and timeline
- [ ] Capture timeline from `Anchor selection biased` to Phase 5 completion
- [ ] Split runtime into segments: intertile solve / affine application / reproject / two-pass gains
- [ ] Record where wall-time spikes

### I2. GPU/CPU attribution clarity
- [ ] Confirm GPU enablement decisions (`GPU_SAFETY`, plan flags, runtime flags)
- [ ] Add explicit logs for CPU-bound sub-phases to avoid false "GPU dropped" diagnosis
- [ ] Ensure telemetry clearly separates GPU-accelerated and CPU-bound stages

### I3. Intertile fallback optimization
- [ ] Audit current fallback (`daemonic processes... -> single worker`)
- [ ] Implement safe multi-worker fallback path (threads) when process pool is unavailable
- [ ] Preserve numerical behavior and anchor-selection correctness

### I4. Two-pass gains optimization
- [ ] Profile `compute_per_tile_gains_from_coverage` path on representative run
- [ ] Reduce overhead in gains stage (executor mode/chunk strategy/progress cadence)
- [ ] Keep gain semantics unchanged unless explicitly documented and approved

### I5. Validation and proof
- [ ] Add/extend tests for intertile fallback behavior
- [ ] Add/extend tests for two-pass gains behavioral invariants
- [ ] Record before/after timing + utilization evidence in `memory.md`
- [ ] Mark GO/NO-GO for merge with explicit risk note
