# Follow-up checklist — Qt filter dialog scroll fix

## Objective
Make the filter dialog in `zemosaic_filter_gui_qt.py` usable on smaller screens / high-DPI displays by ensuring the controls area can scroll and the bottom action buttons remain reachable.

## Constraints
- Surgical diff only
- No broad refactor
- No regression in preview / stream mode / selection workflow
- Keep splitter behavior intact
- Update `memory.md` before stopping

## Tasks
- [x] Review `_build_ui()` layout structure and confirm why the button box can fall outside the visible window.
- [x] Wrap the **right-side controls column** in a `QScrollArea` with a dedicated inner container.
- [x] Keep the preview group on the left side of the splitter.
- [x] Move `QDialogButtonBox` out of the scrollable area and anchor it at the bottom of the main dialog layout.
- [x] Preserve existing controls wiring and layout proportions as much as possible.
- [x] Update `_apply_saved_window_geometry()` to clamp restored geometry to the current screen available area.
- [x] Sanity-check that OK / Cancel still work and that no obvious regression was introduced.
- [ ] If feasible, verify behavior on a constrained-height window or equivalent scenario.
- [x] Update `memory.md` with a compact summary of the issue, root cause, fix, and validation.
- [x] Compact older `memory.md` entries where possible without losing important conclusions.

## Done definition
The task is complete only when all of the following are true:
- The right panel can scroll when the dialog is too short.
- The OK/Cancel row stays visible.
- Restored geometry is clamped to screen bounds.
- `followup.md` is updated with `[x]` for completed work.
- `memory.md` has been updated and compacted.

## Notes for the next iteration
If anything blocks completion, record exactly:
- what remains unfinished,
- why it is unfinished,
- the smallest safe next step.

Current remaining item:
- Manual constrained-height verification remains unfinished because this session is headless (no interactive Qt display to exercise the dialog UI directly).
- Smallest safe next step: run the Qt filter dialog once on a constrained-height display (or high-DPI scaling scenario), confirm right-side scrolling and always-visible OK/Cancel, then mark the last checkbox.
