
## 🧾 `followup.md` – Checklist pour vérifier que Codex a bien fait le job

```markdown
# Follow-up: Auto vs Manual master-tile organisation and optimiser

## 1. What we asked

We requested:

- A **new “Manual-organize Master Tiles” button** that preserves the **old behaviour** of “Auto-organize Master Tiles”.
- A redefined **“Auto-organize Master Tiles”** which:
  - maximises the number of raw frames per master tile (up to the configured max),
  - minimises the number of groups,
  - and keeps groups spatially coherent, using existing clustering helpers.

## 2. What to verify in your changes

Please confirm the following points:

- [ ] The Qt filter dialog now shows **two buttons**:
  - [ ] “Auto-organize Master Tiles”
  - [ ] “Manual-organize Master Tiles”
- [ ] The **Manual** button:
  - [ ] Calls a dedicated helper that encapsulates the legacy behaviour.
  - [ ] Produces the same group structures and logs as the old auto-organise.
- [ ] The **Auto** button:
  - [ ] Builds or retrieves initial groups.
  - [ ] Runs an optimisation step that:
    - [ ] Respects `max_raw_frames_per_master_tile`.
    - [ ] Avoids tiny groups when other options exist (`min_safe_stack` / `target_stack`).
    - [ ] Reduces the total group count where possible.
    - [ ] Keeps groups spatially coherent.
  - [ ] Writes its result into the same group state structure used by the UI.
  - [ ] Refreshes the Sky Preview and groups tree correctly.

## 3. Regression checks

Please double-check:

- [ ] No frames are lost: every selected frame is assigned to some group in both modes.
- [ ] Tk filter UI still works (if any shared helpers were changed).
- [ ] The Coverage Map tab remains unchanged and correct.
- [ ] Sky Preview updates work with both buttons, and (if implemented) group colouring still corresponds to the underlying grouping.
- [ ] There are no new warnings or tracebacks in normal use.

## 4. Behaviour sanity tests

On a real Seestar dataset (with hundreds/thousands of frames):

- [ ] Manual mode yields a grouping comparable to previous versions of ZeMosaic.
- [ ] Auto mode generally:
  - [ ] Creates **fewer** groups than manual mode (when parameters allow it).
  - [ ] Produces groups whose sizes gravitate toward the configured max frames per master tile.
  - [ ] Avoids strange, widely scattered groups.

If any of these expectations are not met, please refine the optimiser logic and re-test.
