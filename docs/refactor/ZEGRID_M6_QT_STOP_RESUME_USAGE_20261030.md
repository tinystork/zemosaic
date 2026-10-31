# ZeGrid Qt — Stop and Save Progress + Resume (ZM-ZEGRID-M6-QT-STOP-RESUME)

Date: 2026-10-30
Scope: Qt GUI integration plus minimal typed engine refusals for failed durable
initialization or degraded-store assembly. Store schemas, scientific values,
budgets and engine/CLI defaults are unchanged.

## What this adds

* **ZeGrid checkpoint saving (opt-in, default OFF)** in *Advanced → ZeGrid run
  policy*: `Save resumable checkpoints during the run`.
  - OFF: the GUI ZeGrid run uses **no durable store** (the persisted config file
    is left untouched, so the CLI keeps its own value).
  - ON: the run commits every completed cell to a durable store **while it
    runs** (not merely when the button is pressed later).
* **Deterministic store**: empty *Checkpoint store folder* means
  `<selected output folder>/.zegrid-durable`; a value there selects an explicit
  existing store folder. The resolved path and byte budget are shown inline.
* **Stop and Save Progress** for a ZeGrid run requests a **spawn-safe
  cooperative stop**. The worker and the queue listener stay alive until the
  engine reaches a safe boundary and acknowledges. There is **no automatic
  4-second terminate** for this operation, and a request alone is never shown as
  "saved".
* **Resume offer** on every ZeGrid start (independent of the checkbox): a
  recognized **incomplete** durable generation in the chosen store is listed and
  the user must choose *Resume…*, *Start new run* or *Cancel*. Multiple
  generations are listed explicitly; nothing is selected by creation date.
  Complete, corrupt, legacy (no frozen state) or live-locked generations are
  never offered as unfinished.

## Exact Linux GUI steps (fresh checkpoint-enabled run → stop → reopen → resume)

1. Start the Qt GUI from this checkout:
   ```bash
   cd /home/tristan/.openclaw/workspace/worktrees/zemosaic-resilient-run-resume
   /home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python run_zemosaic.py
   ```
2. Select the input folder and an output folder.
3. Enable **ZeGrid** and open **Advanced → ZeGrid run policy**.
4. Tick **Save resumable checkpoints during the run**. Leave *Checkpoint store
   folder* empty for `<output>/.zegrid-durable`, or enter a separate existing
   folder. Optionally set an exact **Durable store budget (bytes)**
   (1 GiB = 1073741824).
5. Start the run. The log states that the durable run store is ACTIVE and the
   store path; cells are committed as they complete.
6. Click **Stop and Save Progress**. The log reports that the worker keeps
   running until the engine acknowledges. When the engine acknowledges
   `STOPPED_RESUMABLE`, the log says the completed cells are committed and
   resumable. If it acknowledges `STOPPED_IN_MEMORY`, the run stopped safely but
   is **not** resumable. Without an acknowledgement the run is reported as
   **not saved**.
7. Closing the window while a ZeGrid run is active asks explicitly: *Stop and
   Save, then close* (the window stays open and responsive until the engine
   acknowledges AND the worker exits; nothing is force-killed), *Force stop and
   close* (explicitly reported as NOT saved), or *Cancel*. The ordinary Stop
   button remains usable as the force escape hatch while a save is pending.
8. Close and reopen the GUI (same output folder / same store folder).
9. Start a ZeGrid run again. The **Unfinished ZeGrid run found** dialog lists the
   recognized generation(s). Choose **Resume…** (select explicitly if several
   are listed). The log confirms the explicit selection; the engine restores the
   frozen gauge/geometry/state, verifies input content, and admits only the
   remaining cells.

## Fresh-launch preflight (fail closed)

Before any fresh ZeGrid spawn, the GUI refuses (and creates/modifies nothing):

* an unsafe checkpoint store when checkpoint saving is ON: store == output,
  store inside the inputs, an existing file, or a symlinked store root;
* an unusable output: missing selection, path is a file, output == input, or
  output inside the input;
* an output that already contains a previous ZeGrid publication
  (`zegrid_manifest.json`, `mosaic_grid*.fits`, `.zegrid_publications/`) or any
  foreign content that is not ours to delete.

GUI-owned leftovers (`zegrid_run.log`, `__zegrid_cache__`,
`run_config_snapshot.json`, telemetry/crash artifacts, and the selected store
folder) do not block a retry. With checkpoint saving OFF the store path is not
judged, but the output checks still apply. There is no implicit overwrite.

A RESUME is checked against the output too: a verified publication generation
of the SELECTED run is allowed (so a crash after the durable CURRENT pointer but
before the finalized state ack can still reconcile through the engine's no-op
path). A same-run **crash staging** generation without `CURRENT.json` is also
allowed when its ownership is proven read-only: same run/generation marker,
schema-valid, a DEAD owner pid, no symlink/path escape, no foreign/undeclared
content, and no other run under `.zegrid_publications`. Ownership is NOT
completeness: zero or a partial subset of declared plain members (including a
truncated mid-write file) is allowed, because a real crash may happen right
after staging begins. Staged bytes are never treated as COMPLETE; the engine
still verifies identity/commits and publishes. A publication record with missing
declared members, or a marker-less complete record, is conservatively refused
and left untouched (never deleted to force a resume). The publication root is
checked for BOTH the pointer and staged-only branches: another run's staging or
any foreign entry is refused even beside a valid `CURRENT.json`. Another run's
publication, root-level legacy publication evidence, a live staging owner,
corrupt/unverifiable same-run evidence, or foreign content is refused before
dispatch without modifying any byte. A nested store does not hide unrelated
siblings under its ancestor, including a nested `.zegrid_publications`
directory.

## Crash recovery limitations (honest)

* A checkpoint exists only for cells the engine has already **committed** before
  the crash. In-flight and not-yet-started cells are recomputed on resume.
* A crash is not a "stop": there is no acknowledgement. On reopen the GUI offers
  a run whose `state.json` says it is not finalized and whose store is not
  locked; that is a candidate, not a guarantee. The engine still revalidates
  identity, content and compatibility and may refuse explicitly.
* A **locked** generation (a live writer) is shown as locked and is never
  resumed by the GUI; the engine refuses it too. Stale locks left by a killed
  process are recovered by the engine's store lock, not by the GUI.
* Corrupt or foreign stores are never "blessed": they are reported and the
  engine refuses them. No saved data is deleted or modified by the GUI.
* Only one writer per generation is allowed. Do not point two GUIs at the same
  generation.
* **NaN in the frozen gauge/resume artifact (bounded fail-closed handling):**
  some corpora produce a `NaN` in the frozen gauge/resume payload, which the
  strict store JSON writer refuses. A checkpoint-enabled run now REFUSES before
  any cell admission with an explicit error (`durable checkpoint initialization
  FAILED ... not strict JSON ...`), closes the store (releasing the lock) and
  preserves every file; it makes no ACTIVE/resumable claim and publishes
  nothing. Scientific values are never rewritten to force a store. The 12-frame
  384 px routine corpus reproduces this path (see the mission report). Such a
  corpus cannot be checkpointed/resumed with the current store schema; the GUI
  reports the refusal honestly. If a store is degraded later (e.g. a commit
  failure), assembly refuses with a typed error instead of reading missing
  commits.
* This document makes **no Windows-qualified claims**. The synthetic tests are
  Linux/offscreen and use a tiny deterministic corpus; the real-corpus
  qualification remains the user's.

## Validation status (2026-10-30, final r4 tree)

Code review accepted; full stop/new-process-resume qualification is still
**pending available RAM**. Junior targeted checks: 86 passed, 1 resource-preflight
skip, 2 deselected. The earlier new-process resume attempt was RAM-refused, not
a successful qualification. Do not lower memory guards to force a pass.

The GUI changes are currently **uncommitted/unpushed in the worktree above**.
Switching another checkout to the feature branch alone does not include them.
The test on the user's real corpus remains a separate qualification.
