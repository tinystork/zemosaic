# followup.md

# ZeMosaic — Follow-up checklist
## Qt-only official runtime / Tk retirement

Legend:
- `[ ]` not done
- `[x]` done
- `[~]` partially done / needs follow-up
- `BLOCKED:` explain precisely why

---

## A. Mission discipline

- [x] Read `agent.md`, `followup.md`, and `memory.md` before editing code
- [x] Work only on the next unchecked item
- [x] Confirm this iteration used the canonical roadmap section (A) as source of truth
- [x] Keep changes surgical
- [x] Do not refactor unrelated code
- [x] Update `memory.md` at the end of the iteration
- [x] Mark completed items in both `agent.md` and `followup.md`
- [x] If `memory.md` is missing, use `MIGRATION_LOG.md`

---

## B. S0 — Audit and scope lock

### B1. Tk inventory
- [x] Search all Tk usage (`tkinter`, `messagebox`, `filedialog`, `ttk`, `Tk()`, TkAgg, wrappers)
- [x] Produce per-file classification:
  - [x] official runtime
  - [x] validated headless
  - [x] legacy GUI
  - [x] standalone utility
  - [x] build/doc/test only

### B2. Headless scope lock
- [x] Define the exact validated headless paths for this mission
- [x] Explicitly list non-supported headless paths
- [x] Record this in `memory.md`

### B3. `zemosaic_config.py` strategy
- [x] Audit direct Tk imports in `zemosaic_config.py`
- [x] Audit indirect Tk coupling risks from config helpers
- [x] Choose one strategy and record it:
  - [ ] Tk-free module
  - [x] split core/legacy helpers
  - [ ] strict lazy import outside official path

### B4. `lecropper` status
- [x] Confirm `lecropper` classification as annex / standalone tool
- [x] Confirm that `lecropper` port is out of scope for this mission
- [x] Record the decoupling requirement for S2

### B5. Initial parity matrix
- [x] Build initial Qt parity matrix
- [x] Mark each item as:
  - [x] OK
  - [x] gap
  - [x] blocking
  - [x] out-of-scope

### B6. S0 closeout
- [x] Record S0 proof in `memory.md`
- [x] Mark S0 complete only when scope/headless/config strategy are explicit

---

## C. S1 — Strict Qt parity

### C1. Official workflow review
- [x] Verify GUI startup workflow parity
- [x] Verify config load/save parity
- [x] Verify logs / feedback parity
- [x] Verify clean shutdown behavior
- [x] Verify filter workflow if part of official frontend
- [x] Verify Grid mode path if official
- [x] Verify SDS path if official/relevant

### C2. UX cleanup prep
- [x] Remove or plan removal of “Qt preview” wording
- [x] Remove or plan removal of “Tk stable” wording
- [x] Identify backend switch UI elements to eliminate in S2

### C3. Hidden backend features
- [x] List backend features still alive but not exposed in Qt
- [x] For each, classify:
  - [x] expose now
  - [x] out-of-scope
  - [x] legacy
- [x] Record decisions in `memory.md`

### C4. Persistence parity
- [x] Verify Qt UI writes the expected config keys
- [x] Verify persisted settings reload correctly
- [x] Identify any Tk/Qt ambiguity in config behavior

### C5. S1 closeout
- [x] Update parity matrix
- [ ] Confirm no remaining P0/P1 blocker for official frontend
- [x] Record proof in `memory.md`

---

## D. S2 — Official runtime Qt-only cutover

### D1. Launcher cleanup
- [x] Remove Tk fallback from `run_zemosaic.py`
- [x] Remove `--tk-gui` or equivalent official-path Tk switch if present
- [x] Remove Tk startup message boxes / root creation
- [x] Ensure startup error handling uses console / Qt / neutral path only

### D2. Config import safety
- [x] Make `zemosaic_config.py` import-safe without Tk
- [x] Remove or isolate Tk-dependent helpers from config official/headless path
- [x] Change official default backend from `tk` to `qt`

### D3. Qt UI cleanup
- [x] Remove backend choice from official Qt UI
- [x] Remove coexistence wording from `zemosaic_gui_qt.py`
- [x] Remove “install PySide6 or use Tk interface instead” wording from `zemosaic_filter_gui_qt.py`
- [x] Harden official messaging: PySide6 is required for official frontend

### D4. `lecropper` decoupling
- [x] Remove direct runtime dependency on `lecropper` from official path
- [x] Remove indirect runtime dependency on `lecropper` from validated headless path
- [x] Ensure absence of `lecropper` does not break official runtime/headless path

### D5. Mandatory gate
- [x] Test official launch path without Tk fallback
- [x] Test `python -c "import zemosaic_config"`
- [x] Test validated `import zemosaic_worker`
- [x] Test official path without `lecropper`
- [x] Record proof in `memory.md`

---

## E. S3 — Config migration and cleanup

### E1. Legacy config migration
- [x] Migrate `preferred_gui_backend=tk` to `qt`
- [x] Neutralize obsolete backend selection state if needed
- [x] Preserve backward readability where required

### E2. Round-trip safety
- [x] Create/collect minimal legacy config fixtures
- [x] Verify load → save → load idempotence
- [x] Verify no silent Tk reactivation through config

### E3. Official cleanup
- [x] Remove active official branches tied to Tk backend coexistence
- [x] Keep cleanup narrow and migration-focused

### E4. S3 closeout
- [x] Record fixture results in `memory.md`
- [x] Mark S3 complete only when round-trip is proven

---

## F. S4 — Packaging / docs / release notes

### F1. Build / packaging
- [x] Audit official build/spec scripts
- [x] Verify final built artifacts, not only source scripts
- [x] Ensure official packaging no longer suggests Tk coexistence

### F2. Docs
- [x] Update user docs / README / quickstart / troubleshooting
- [x] Update dev/build notes
- [x] Add explicit public status of annex tools including `lecropper`

### F3. Release notes
- [x] State clearly that Qt is the only official frontend
- [x] Describe config migration behavior
- [x] Describe unsupported / legacy status
- [x] Clarify `lecropper` status

### F4. Versioning / semantics
- [x] Record target release/version semantics for this breaking change
- [x] Add note to `memory.md`

### F5. S4 closeout
- [x] Record packaging/doc proof in `memory.md`

---

## G. S5 — Final validation / QA / CI

### G1. Smoke tests
- [~] Windows smoke test (ON HOLD by Tristan, 2026-03-13)
- [x] Linux smoke test
- [~] macOS smoke test if officially supported (ON HOLD by Tristan, 2026-03-13)

### G2. Dependency failure tests
- [x] PySide6 absent
- [x] worker import failure
- [x] GUI startup error path
- [x] verify no Tk fallback in all such cases

### G3. Headless tests
- [x] `import zemosaic_config`
- [x] config load/save
- [x] idempotent round-trip
- [x] `import zemosaic_worker`
- [x] confirm no `lecropper` dependency leak

### G4. CI hardening
- [x] Add or update CI job: `no-Tk-on-official-path`
- [x] Ensure CI fails on official-path Tk import regression
- [x] Ensure CI fails on validated-headless Tk regression

### G5. GO / NO-GO checklist
- [x] official runtime has no Tk fallback
- [x] `zemosaic_config.py` imports without Tk
- [x] validated headless paths do not import Tk
- [x] `lecropper` is no longer an official runtime dependency
- [x] config migration is idempotent
- [x] docs and release notes are aligned
- [x] CI is green

### G6. S5 closeout
- [x] Record final QA proof in `memory.md`
- [x] Mark release status as GO or NO-GO

---

## H. S6 — Later mission only (do not start now unless requested)

### H1. Decide final fate of Tk annexes
- [ ] `lecropper` = legacy standalone frozen
- [ ] or `lecropper` = dedicated Qt port later
- [ ] or `lecropper` = deprecated / removed
- [ ] owner + due date recorded

### H2. Other annexes
- [ ] `zemosaic_filter_gui.py`
- [ ] `zequalityMT.py`
- [ ] other Tk wrappers/helpers

---

## I. Required per-iteration report format

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
- mention whether `lecropper` status changed or not
- mention whether official-path Tk imports decreased or stayed unchanged
- mention whether validated headless scope changed or stayed unchanged

---

## J. Stop conditions

Stop and report instead of improvising if:
- the next item requires broad refactor outside migration scope
- the headless scope is ambiguous
- `lecropper` port work would get mixed into official cutover
- proof is missing
- tests contradict the roadmap assumptions
