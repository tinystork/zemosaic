# agent.md

# ZeMosaic — Mission Codex
## Qt-only official runtime + Tk retirement (phase 1)

Date: 2026-03-13  
Owner: Tristan / ZeMosaic core  
Mission mode: surgical / no-refactor / proof-driven

## Mission objective

Make the **official ZeMosaic runtime** Qt-only and remove any **Tk dependency from official and validated headless paths**, without mixing this work with a full repo-wide Tk purge.

This mission must follow the roadmap order:

**S0 → S1 → S2 → S3 → S4 → S5**, then optionally **S6** later.

Do **not** start S6 now unless Tristan explicitly asks for it.

The architectural rules are locked:

- official frontend must be **100% Qt**
- official runtime and validated headless paths must be **Tk-free**
- `lecropper.py` is treated as an **annex / standalone tool**
- `lecropper.py` must be **decoupled from official runtime now**
- porting `lecropper` to Qt is a **separate later mission**, not part of this one

Reference roadmap: `ROADMAP_REMOVE_TKINTER.md`.

Canonical source-of-truth for execution:
- Use **section A (Canon executable)** as normative execution contract.
- Treat sections B/C as historical reference only.
- If wording conflicts, section A prevails.

---

## Non-negotiable execution rules

1. **Work only on the next unchecked item.**
2. **Keep diffs surgical.**
3. **Do not refactor unrelated code.**
4. **Do not change behavior outside Tk/Qt migration scope unless required to preserve current behavior.**
5. **Always update `memory.md` after each meaningful iteration.**
6. **Always mark completed items with `[x]` in both `agent.md` and `followup.md`.**
7. If `memory.md` does not exist, use `MIGRATION_LOG.md`.
8. In every report, explicitly state:
   - what was in scope
   - what was out of scope
   - what proof was collected
   - what remains blocked / open
9. Never declare success without proof.
10. Never mix `lecropper` Qt port work into the official cutover unless Tristan explicitly changes scope.

---

## Current baseline assumptions to verify first

The roadmap indicates these likely current issues and they must be confirmed before modification:

- `run_zemosaic.py` still imports Tk and still contains Tk fallback behavior
- `zemosaic_config.py` still defaults GUI backend to `tk`
- `zemosaic_gui_qt.py` still exposes Tk/Qt coexistence language and/or backend switching
- `zemosaic_filter_gui_qt.py` still mentions “use the Tk interface instead”
- `zemosaic_worker.py` still imports `lecropper`
- `lecropper.py` is Tk-based but is considered annex scope, not official frontend scope

Do not assume more than that until audited.

---

## Scope

### In scope for this mission
- `run_zemosaic.py`
- `zemosaic_gui_qt.py`
- `zemosaic_filter_gui_qt.py`
- `zemosaic_config.py`
- official startup error handling
- official packaging/build/docs/release-note adjustments
- validated headless paths
- config migration for legacy users
- CI/QA checks for “no-Tk-on-official-path”

### Explicitly out of scope for this mission
- full repo-wide Tk removal
- Qt port of `lecropper.py`
- aggressive cleanup/refactor of legacy tools
- aesthetic or architectural refactors unrelated to Tk retirement
- changing standalone utility behavior unless needed for runtime decoupling

### Special rule for `lecropper`
`lecropper.py` may remain standalone/Tk for now, but it must stop being a dependency of the official runtime and validated headless paths.

---

## Required artifacts to maintain during the mission

Keep these updated as the mission progresses:

- `agent.md`
- `followup.md`
- `memory.md` (or `MIGRATION_LOG.md` if absent)

Optional but recommended if they do not exist yet:
- `QT_ONLY_P0_CHECKLIST.md`
- `TK_EXCEPTION_REGISTER.md`
- `OFFICIAL_TEST_MATRIX.md`

If you create these optional files, keep them minimal and practical.

---

## Proof requirements

You must collect proof for each completed step.

Examples of acceptable proof:
- grep/ripgrep results showing official-path Tk imports removed
- import tests such as:
  - `python -c "import zemosaic_config"`
  - `python -c "import zemosaic_worker"`
- config migration round-trip tests
- launch-path checks showing no Tk fallback
- CI or local test output
- concise before/after notes in `memory.md`

Never mark an item done without attaching proof in the iteration notes.

---

## Phase execution contract

### [x] S0 — Audit and scope lock
Goal:
- produce a reliable inventory of Tk usage
- classify each usage by criticality
- freeze validated headless scope
- decide config strategy
- confirm `lecropper` status as annex

Expected outputs:
- list of Tk imports/usages by file
- classification:
  - official runtime
  - validated headless
  - legacy GUI
  - standalone utility
  - build/doc/test only
- explicit validated headless matrix
- chosen strategy for `zemosaic_config.py`
- explicit note that `lecropper` is annex and not part of P0 UI port

Hard prohibitions:
- no Tk deletion yet
- no `lecropper` port work
- no opportunistic refactor

### [ ] S1 — Strict Qt parity for official workflows
Goal:
- make sure no critical official workflow still depends on Tk
- remove “Qt preview” positioning
- verify config persistence parity
- classify backend features not exposed in Qt

Expected outputs:
- updated parity matrix
- no remaining P0/P1 blocker for official frontend
- explicit decisions for hidden backend features:
  - expose
  - accept as out-of-scope
  - classify as legacy

Hard prohibition:
- do not remove launcher fallback yet until S1 is truly validated

### [x] S2 — Official runtime Qt-only cutover
Goal:
- remove Tk from official nominal + startup error paths
- remove Tk fallback
- make config import-safe without Tk
- decouple `lecropper`

Required work:
- remove Tk fallback from `run_zemosaic.py`
- remove Tk message boxes from startup paths
- make `zemosaic_config.py` import-safe without Tk
- switch official default backend to Qt
- remove backend-switch exposure from official Qt UI
- harden PySide6-required messaging
- ensure `lecropper` is not imported directly or indirectly on official/headless validated paths

Mandatory gate:
In an environment without Tk, all of the following must pass:
- official startup path does not fallback to Tk
- `import zemosaic_config` works
- `import zemosaic_worker` works on validated headless paths
- absence of `lecropper` does not break official runtime

### [x] S3 — Config migration and official cleanup
Goal:
- migrate legacy config cleanly
- neutralize obsolete Tk backend state
- ensure idempotent round-trip

Required work:
- migrate `preferred_gui_backend=tk` to `qt`
- neutralize obsolete backend-switch state if needed
- preserve backward readability
- ensure load/save/load round-trip is idempotent
- remove active official references to Tk-specific config behavior

Mandatory proof:
- config fixtures
- rewritten config examples
- round-trip verification

### [x] S4 — Packaging / docs / release notes
Goal:
- align product, packaging and docs with the new official reality

Required work:
- clean official spec/build scripts
- verify final artifacts, not only scripts
- update user docs
- update dev docs
- prepare release notes
- state public status of annex tools including `lecropper`

Important:
This is where versioning / release semantics should be made explicit if needed.

### [ ] S5 — Final validation / QA / CI
Goal:
- prove the official release path is genuinely Qt-only

Required work:
- smoke tests on supported OSes
- legacy config migration fixtures
- missing dependency tests (PySide6 absent, startup errors, etc.)
- validated headless tests
- CI “no-Tk-on-official-path”

Release gate:
Do not declare GO unless all 7 conditions are satisfied:
1. official runtime has no Tk fallback
2. `zemosaic_config.py` imports without Tk
3. validated headless paths do not import Tk
4. `lecropper` and other Tk tools are not official runtime dependencies
5. config migration is idempotent
6. docs and release notes are aligned
7. CI is green

### [ ] S6 — Annex tools / final `lecropper` decision
Not part of the current mission unless explicitly requested later.

If opened later, options are:
- legacy standalone frozen
- Qt port in a separate dedicated mission
- deprecation/removal

---

## Required iteration protocol

At the start of each iteration:
1. Read `agent.md`, `followup.md`, and `memory.md` if present.
2. Work only on the next unchecked item.
3. Restate the immediate sub-scope internally before editing.

At the end of each iteration:
1. Update `memory.md`.
2. Mark completed checkbox(es) in `agent.md` and `followup.md`.
3. Add proof summary.
4. Add blockers / risks / next step.
5. Explicitly state whether `lecropper` status changed or remained unchanged.

---

## Mandatory `memory.md` update format

Append a compact entry after each meaningful iteration using this structure:

### YYYY-MM-DD HH:MM — Iteration N
- Scope:
- In scope:
- Out of scope:
- Files changed:
- Decisions taken:
- Proof collected:
- Tests run:
- Result:
- Remaining blockers:
- Next unchecked item:

Keep entries concise and cumulative.
Do not rewrite history.
Do not omit failures: log them.

---

## Success definition

This mission is successful only if:
- the official runtime is Qt-only
- validated headless paths are Tk-free
- config migration is safe and idempotent
- `lecropper` is decoupled from official runtime
- docs/build/release notes match reality
- proof exists for each claim

If any of those is missing, the mission is not complete.
