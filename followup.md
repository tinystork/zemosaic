# Follow-up — Qt GUI tooltip for existing master tiles photometry limitation

## Status
- [x] Tooltip added for existing master tiles mode in Qt GUI

## Goal
Inform users **at the point of use** that photometric normalization is limited
when using pre-existing master tiles.

Documentation is not sufficient; the GUI must carry this information.

## Scope
- Qt GUI only (`zemosaic_gui_qt.py`)
- Tooltip only (no modal dialog, no warning popup)
- No behavior change
- No Tk changes

## Implementation Details

### Target UI element
Checkbox:
- "I'm using master tiles (skip clustering_master tile creation)"

### Tooltip content (EN)
"Photometric normalization is limited in this mode.
Geometry will be correct, but residual brightness differences between tiles may remain.
For best photometric quality, build master tiles inside ZeMosaic."

### Tooltip content (FR)
"Dans ce mode, la normalisation photométrique est limitée.
La géométrie sera correcte, mais des différences de luminosité peuvent subsister entre les tuiles.
Pour une qualité photométrique optimale, générez les master tiles dans ZeMosaic."

### Localization
- Use existing localization system
- Add new localization keys if needed
- Do NOT hardcode strings

### Visual behavior
- Tooltip only
- No icon
- No color change
- No warning triangle
- Subtle and informative

## Non-goals
- No README update
- No online documentation link
- No blocking warning

## Success Criteria
- Tooltip visible on hover
- Clear expectation setting
- No added friction to workflow
- Zero impact on non-Qt GUI
