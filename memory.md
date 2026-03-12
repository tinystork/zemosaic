# Memory (compacted)

## 2026-01-28 / 2026-01-29 — Windows PyInstaller build (GPU + Shapely + SEP)

### Problem
Packaged Windows build had failures not seen in non-compiled runs:
- GPU path falling back to CPU (`phase5_using_cpu`) due CuPy import/runtime issues.
- Phase 4 failure around `find_optimal_celestial_wcs()` in packaged mode.
- Additional packaged-only alignment/import errors.

### Root causes confirmed
- Frozen DLL search path in packaged runtime needed stronger setup.
- CuPy packaging/dependencies were incomplete early on (notably missing `fastrlock`).
- Shapely packaging was incomplete (`shapely._geos` missing in packaged runtime).
- `sep_pjw` expected top-level `_version` that was not always bundled.

### Changes made
- `pyinstaller_hooks/rthook_zemosaic_sys_path.py`
  - Added frozen DLL-path setup for `sys._MEIPASS` and discovered `*.libs`/DLL folders.
  - Added safer `os.add_dll_directory()` handling with `WinError 206` mitigation.
- `ZeMosaic.spec`
  - Added stronger hidden-import/data handling for CuPy/fastrlock/Shapely/SEP-related pieces.
  - Added explicit `_version.py` bundling path.
- `pyinstaller_hooks/hook-shapely.py`
  - Added hidden import support for `shapely._geos`.
- Runtime diagnostics/logging improved in key modules to expose packaged-only failures.

### Validation summary
- Shapely `WinError 206` path issue mitigated.
- Missing packaged dependencies were identified and patched iteratively.
- Some packaged logs later showed GPU detection/use recovery, but packaged behavior remained more fragile than non-compiled runs in parts of the workflow.

### Remaining caution
For packaging work, preserve:
- frozen DLL search-path setup,
- explicit hidden imports for binary dependencies,
- detailed runtime logging in packaged mode.

---

## 2026-03-12 — Qt filter dialog usability fix (`zemosaic_filter_gui_qt.py`)

### Topic
Start/OK button could become unreachable on smaller screens / high-DPI setups.

### Problem
In the Qt filter dialog, the right-side controls and button box were in the same non-scrollable column, so the bottom actions could fall outside the visible area.

### Root cause
- Right controls column was not inside a `QScrollArea`.
- Saved geometry was restored as-is without clamping to current screen `availableGeometry()`.

### Changes made
- `_build_ui()`:
  - Added `QScrollArea` for the right panel (`setWidgetResizable(True)`).
  - Kept preview group as left splitter widget.
  - Moved right controls into a dedicated inner container set as the scroll area widget.
  - Moved `QDialogButtonBox` outside scrollable content and anchored it in the main dialog layout under the splitter.
  - Added a trailing stretch in the scrollable controls layout for natural packing.
- `_apply_saved_window_geometry()`:
  - Added safe clamping of restored `(x, y, w, h)` against current screen `availableGeometry()`.
  - Added safe screen lookup fallback order: `screenAt(center)` -> `self.screen()` -> `primaryScreen()`.
  - Preserved fail-safe behavior if screen lookup/clamp logic cannot run.

### Validation performed
- Static checks:
  - `python3 -m py_compile zemosaic_filter_gui_qt.py` passed.
- Code-level sanity checks:
  - OK/Cancel signal wiring (`accepted -> accept`, `rejected -> reject`) preserved.
  - Existing preview/stream/selection-related wiring untouched in the patch scope.
  - Splitter remains intact with preview left and controls right.

### Remaining risk / follow-up
- Manual GUI verification on constrained-height/high-DPI display was not run in this headless session.
- Smallest safe next step: launch the dialog in a constrained-height scenario and confirm right-panel scrolling + always-visible OK/Cancel.
