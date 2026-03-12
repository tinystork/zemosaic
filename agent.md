# Mission: fix inaccessible Start/OK button in Qt filter dialog

## Context
A user reported that in `zemosaic_filter_gui_qt.py`, the **Start / OK** button can become unreachable on some screens or DPI scaling settings because it ends up below the visible area of the dialog.

Initial code review indicates two likely causes:
1. The filter dialog builds its controls column directly in a `QVBoxLayout` with no `QScrollArea`, so when vertical space is insufficient the bottom action area simply falls outside the window.
2. `_apply_saved_window_geometry()` restores the saved geometry as-is, without clamping it to the current screen's available geometry. A geometry saved on a larger display can therefore reopen too large or partly off-screen on a smaller one.

## Goal
Make the filter dialog reliably usable on smaller screens and high-DPI setups, with **no regression** to preview behavior, layout balance, or dialog actions.

## Scope
Primary target:
- `zemosaic_filter_gui_qt.py`

Allowed if strictly needed:
- tiny helper methods in the same file only

Avoid touching unrelated files unless absolutely necessary.

## Required outcome
Implement the safest low-regression fix:

1. Keep the overall dialog structure and existing widgets.
2. Put the **right-hand controls column** inside a `QScrollArea`.
3. Keep the **OK/Cancel button box fixed outside the scroll area**, at the bottom of the dialog, so it remains visible.
4. Clamp restored window geometry to the current screen's `availableGeometry()` before applying it.
5. Preserve the existing preview pane, splitter, labels, activity log, tree, options, and dialog behavior.

## Important constraints
- **No broad refactor.** Use a surgical diff.
- Do **not** change the meaning or wiring of existing controls.
- Do **not** break preview tabs, coverage preview, activity log, image tree, or options persistence.
- Do **not** introduce regressions for stream mode, selection actions, or accept/reject behavior.
- Keep the splitter behavior intact.
- Prefer readability over cleverness.

## Implementation guidance
### A. `_build_ui()`
Refactor the right-side controls assembly like this:
- Create a `QScrollArea` for the controls side.
- Create a dedicated inner `controls_container` widget and its `QVBoxLayout`.
- Set `setWidgetResizable(True)` on the scroll area.
- Add the scroll area to the splitter instead of adding the raw controls container directly.
- Keep the preview group as the left widget of the splitter.
- Move the `QDialogButtonBox` out of the scrollable controls layout and add it directly to the main dialog layout after the splitter.
- Keep a stretch inside the scrollable controls layout so the controls pack naturally.

Desired UX:
- On tall screens, the dialog should look essentially unchanged.
- On short screens, the right-side controls should scroll while the bottom OK/Cancel row remains reachable.

### B. `_apply_saved_window_geometry()`
Before calling `setGeometry(...)`:
- Obtain the saved `(x, y, width, height)`.
- Determine the current screen's available geometry.
- Clamp width/height so they do not exceed the available area.
- Clamp x/y so the window remains fully reachable on screen.
- Fail safely if screen lookup is unavailable.

### C. Keep the patch small
Do not redesign the dialog. This is a usability fix, not a UI rewrite.

## Validation checklist
Please validate as much as possible directly in code and, if runnable, manually:

1. Dialog opens normally on standard resolution.
2. Preview panel still appears and splitter still resizes.
3. Right-side content scrolls when vertical height is constrained.
4. OK/Cancel remain visible and clickable.
5. Accept / reject still work.
6. Restored geometry no longer places the dialog partly off-screen or too tall for the current display.
7. No obvious regression in stream mode or image filtering workflow.

## Deliverables
Update:
- `zemosaic_filter_gui_qt.py`
- `followup.md`
- `memory.md`

## Mandatory project tracking updates
Before finishing:
1. Mark completed items in `followup.md` with `[x]`.
2. Add a concise entry to `memory.md` describing:
   - the root cause,
   - what was changed,
   - what was validated,
   - any remaining risk or follow-up.
3. **Compact `memory.md` if it has become too verbose.** Keep the historical trace, but rewrite long step-by-step logs into short status summaries so future reads consume fewer tokens.

## Suggested memory style
Prefer this compact structure:
- Date
- Topic
- Problem
- Root cause
- Changes made
- Validation
- Remaining items

Avoid dumping repeated intermediate attempts once the conclusion is known.
