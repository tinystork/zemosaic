## ZeGrid run policy in the Qt Advanced tab (2026-10-29)

Mission `ZM-ZEGRID-M6-QT-POLICY-20261029`. This note documents the six editable
run-policy keys added to **Advanced → ZeGrid run policy**. It is a GUI wiring
note, not a restart/resume qualification report.

### Controls

| Control | Config key | Value and unit | Default |
|---|---|---|---|
| Wait for resources during the run | `zegrid_resource_recovery_enabled` | boolean | `False` |
| Recovery deadline (seconds) | `zegrid_resource_recovery_deadline_s` | positive seconds, fractions accepted; empty = no deadline | `None` |
| Durable store budget (bytes) | `zegrid_durable_store_budget_bytes` | exact positive integer bytes; empty = engine default; values above 2 GiB accepted | `None` |
| Cell gauge mode | `zegrid_cell_gauge_mode` | `None` (engine-default sentinel), `auto` (explicit), `global`, `per_cell` | `None` |
| Layout (NXxNY) | `zegrid_layout` | explicit `NXxNY` with positive dimensions; empty = automatic | `None` |
| SIP mode | `zegrid_sip_mode` | `keep` (apply SIP), `strip` (legacy plain TAN) | `keep` |

The gauge control has FOUR distinct stored values. `None` is the absent-value
sentinel ("Engine default"); it currently resolves to the engine's auto strategy
but is stored separately. `auto` is the literal token and is shown as a separate
"Auto (explicit)" choice. `global` and `per_cell` force the corresponding path.
Both `None` and `auto` resolve to the same engine strategy today, but the GUI
never rewrites one into the other. No gauge strategy is changed by this wiring.

The durable-store budget is entered and stored as bytes, not GiB, so the value
is transmitted exactly. Example: 1 GiB = `1073741824`; 6 GiB = `6442450944`.
No rounding, clamping, or int32 narrowing is applied.

### Behaviour

- The controls are enabled only when the effective processing mode is ZeGrid
  and no run is active. They are disabled in Classic, SDS and existing-master
  modes without affecting those modes, and disabling never erases a stored
  value.
- With resource recovery OFF, the recovery deadline is disabled but keeps its
  persisted value. Recovery means waiting for memory/disk pressure to ease
  **during the same run**; it is not restart resume.
- Invalid persisted or typed values stay visible (raw text or empty combo plus
  an inline warning). The common validator, `validate_zegrid_run_policy`, is the
  single authority; an invalid value blocks a ZeGrid start with a warning and is
  never silently replaced by a default. A non-ZeGrid start is not blocked.
- Strict persisted types are preserved. A persisted JSON string such as
  `"12.5"` or `"6442450944"` is INVALID for the numeric keys, even though the
  text looks numeric. Loading the widgets, collecting, saving, changing the
  active mode or language, or building the worker invocation does NOT convert it
  to a float/int; the raw string and its warning stay until the user deliberately
  edits the field. Whitespace-only/empty strings, booleans and non-numerics are
  handled the same way.
- Once the user deliberately edits a field into a valid typed value, normal
  typed parsing applies and the warning clears. No automatic migration or silent
  coercion is performed.
- Values round-trip through load/edit/save/reload. The GUI invocation carries
  the typed values explicitly to the worker, and the worker applies them to the
  engine config, so a different persisted/global file cannot silently win.
- If the strict config loader refuses a persisted policy value, the GUI recovers
  ONLY the six raw policy values. It loads a sanitised copy of the same file
  through the normal loader first, so every unrelated migration/validation still
  runs, then overlays the raw six values so they remain visible and editable.
  Any other load failure keeps the historical defaults-only behaviour.

### What this wiring does NOT add

- No GUI Stop-and-Save button and no GUI resume selector.
- No automatic enablement: the defaults above are unchanged (`resume` remains
  `never`).
- No change to store schema, science, RAM thresholds, GPU behavior, Windows
  behavior, or M31.

### Manual resume (not qualified in this mission)

The headless CLI already exposes the launch-time resume policy. For a later
manual attempt, with an existing durable store and a compatible stopped run:

```bash
python -m zemosaic.zegrid_cli \
  --input <frames-folder> \
  --output <output-folder> \
  --durable-store-dir <store-root> \
  --resume-mode auto \
  --resume-run-id <explicit-generation-if-needed>
```

`--resume-mode auto` requires `--durable-store-dir`; `--resume-run-id` requires
`--resume-mode auto`. These options are documented here for the user's own
manual restart/resume work. They were **not** exercised by this mission.

### Evidence and limits

Evidence for this wiring: focused Qt offscreen tests (four-way gauge
round-trip, typed round-trip, null clearing at the real engine boundary,
multi-GiB budget, fractional deadline, strict persisted-type preservation,
invalid-input refusal and correction, mode/language switching, actual
GUI-invocation → worker-signature binding, and GUI → worker → engine
propagation with a deliberately different persisted config) plus one real tiny
4-frame / 256×256 / 2×2 CPU first-run smoke through the GUI-built invocation,
dispatcher and real engine. The smoke only verifies first-run plumbing.

USER-OWNED / NOT_RUN here: the complete non-seamed Linux restart/resume run,
M3 physical memory pressure, Windows/NTFS resume behaviour, full-corpus and GPU
qualification.
