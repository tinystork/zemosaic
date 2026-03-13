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
- [ ] Write the invariant explicitly:
  - [ ] all raw frames assigned to a logical master tile must remain part of that tile's scientific stack
  - [ ] adaptation may reduce only working-set size and/or active parallelism
  - [ ] no silent quality downgrade via frame dropping

### B2. Current Phase 2 behavior
- [ ] Audit current group splitting in Phase 2
- [ ] Determine whether current auto-limit logic changes logical master-tile content
- [ ] Record exactly when Phase 2 splitting is acceptable vs not acceptable for this mission

### B3. Current Phase 3 behavior
- [ ] Audit current Phase 3 worker selection
- [ ] Audit current runtime semaphore adaptation
- [ ] Audit current eager task submission behavior
- [ ] Audit current per-tile memory levers:
  - [ ] `winsor_max_frames_per_pass`
  - [ ] row/chunk sizing
  - [ ] memmap/streaming flags
  - [ ] GPU budget interactions if relevant

### B4. Baseline proof
- [ ] Record baseline findings in `memory.md`
- [ ] Mark S0 complete only when the invariant and current gaps are explicit

---

## C. S1 — Adaptive strategy design

### C1. Controller policy
- [ ] Define target RAM policy around 80%
- [ ] Define hysteresis thresholds
- [ ] Define escalation order:
  - [ ] reduce active Phase 3 launches
  - [ ] reduce per-pass frames
  - [ ] reduce rows per chunk
  - [ ] serialize special paths if required

### C2. Scheduler policy
- [ ] Decide lazy scheduling model for Phase 3
- [ ] Decide queue semantics for pending groups
- [ ] Decide how retries re-enter the queue

### C3. Scientific preservation policy
- [ ] Define what "preserve all raw frames" means for:
  - [ ] mean / weighted mean
  - [ ] median
  - [ ] winsorized sigma clip
  - [ ] other active rejection modes
- [ ] Explicitly document any exactness limitations

### C4. Telemetry policy
- [ ] Define which adaptation decisions must be logged
- [ ] Define which telemetry fields must be emitted

### C5. Design sign-off
- [ ] Record the target design in `memory.md`
- [ ] Mark S1 complete only when the decision tree is explicit

---

## D. S2 — Scheduler refactor for Phase 3 launch control

### D1. Replace eager launch
- [ ] Stop submitting all Phase 3 groups immediately
- [ ] Introduce a pending queue / lazy launcher
- [ ] Preserve existing completion/progress behavior as closely as practical

### D2. Adaptive launch budget
- [ ] Add a controllable active-launch limit
- [ ] Ensure the limit can shrink at runtime for not-yet-started work
- [ ] Ensure the limit can grow again when memory pressure drops

### D3. Retry compatibility
- [ ] Preserve retry-group scheduling
- [ ] Ensure retries do not bypass launch control

### D4. Proof
- [ ] Record scheduler behavior proof in `memory.md`
- [ ] Mark S2 complete only when runtime launch throttling is real, not only nominal

---

## E. S3 — Per-tile adaptive working-set control

### E1. Frame-pass adaptation
- [ ] Add dynamic per-tile `winsor_max_frames_per_pass` logic
- [ ] Ensure it reduces RAM footprint without dropping raw membership

### E2. Chunk adaptation
- [ ] Add dynamic per-tile row/chunk sizing
- [ ] Ensure chunk adaptation reacts to actual tile characteristics

### E3. Exactness review
- [ ] Verify whether each combine/rejection mode remains exact under the chosen adaptation
- [ ] If not exact, document the limitation before enabling the mode

### E4. Proof
- [ ] Record per-tile adaptation proof in `memory.md`
- [ ] Mark S3 complete only when logical frame membership remains unchanged

---

## F. S4 — Runtime RAM controller and telemetry

### F1. Runtime RAM sampling
- [ ] Add or reuse periodic RAM sampling for Phase 3 decisions
- [ ] Base decisions on observed pressure, not only static startup estimates

### F2. Hysteresis and stability
- [ ] Avoid oscillation with hysteresis / cooldown rules
- [ ] Verify recovery behavior when RAM pressure decreases

### F3. Telemetry and logs
- [ ] Emit structured logs for adaptation actions
- [ ] Emit telemetry fields needed for later validation

### F4. Optional GPU policy
- [ ] Decide whether Phase 3 needs a dedicated GPU concurrency guard
- [ ] If yes, implement and document it narrowly

### F5. Proof
- [ ] Record runtime-controller proof in `memory.md`
- [ ] Mark S4 complete only when decisions are observable and explainable

---

## G. S5 — Validation and non-regression tests

### G1. Invariant tests
- [ ] Add tests proving no silent raw-frame dropping
- [ ] Add tests proving logical membership is preserved under adaptation

### G2. Pressure-response tests
- [ ] Add tests proving active-launch backoff under RAM pressure
- [ ] Add tests proving per-pass/chunk shrink under RAM pressure
- [ ] Add tests proving recovery when pressure drops

### G3. Compatibility tests
- [ ] Add tests for retry-path compatibility
- [ ] Add tests for telemetry/log integrity
- [ ] Add tests for affected combine/rejection modes

### G4. Mission gate
- [ ] no silent raw-frame dropping
- [ ] runtime launch throttling works
- [ ] working-set adaptation works
- [ ] telemetry/logging is sufficient
- [ ] tests cover the invariant

### G5. Closeout
- [ ] Record final proof in `memory.md`
- [ ] Mark mission status as GO or NO-GO for merge

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

