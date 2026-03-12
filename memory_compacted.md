# Memory (compacted)

## 2026-01-28 / 2026-01-29 — Windows PyInstaller build (GPU + Shapely + SEP)

### Problem
The Windows packaged build showed multiple issues not present in the non-compiled Python run:
- GPU not used in packaged mode (`phase5_using_cpu`, GPU unavailable / CuPy import failures)
- Phase 4 grid failure during `find_optimal_celestial_wcs()`
- Additional packaged-only alignment/runtime import errors

### Root causes confirmed
- Packaged runtime DLL search path was incomplete for frozen execution.
- CuPy was not correctly bundled at first; later a missing dependency was identified:
  - `fastrlock` missing caused CuPy import failure in packaged mode.
- Shapely packaging was incomplete:
  - runtime/import issues around bundled DLL resolution
  - later confirmed missing module: `shapely._geos`
- `sep_pjw` packaging was incomplete in the packaged build:
  - missing top-level `_version` import caused downstream alignment/runtime failures.

### Changes made
- `pyinstaller_hooks/rthook_zemosaic_sys_path.py`
  - added frozen DLL-path setup for `sys._MEIPASS`
  - added support for discovered `*.libs` folders and DLL-containing subfolders
  - added safer `os.add_dll_directory()` handling, including `WinError 206` mitigation
- `ZeMosaic.spec`
  - added packaging adjustments for Shapely
  - added stronger CuPy bundling / checks
  - added explicit handling for `fastrlock`
  - added optional packaging support for `sep`, `sep_pjw`, and `_version`
  - added explicit bundling of `_version.py`
- `pyinstaller_hooks/hook-shapely.py`
  - added hidden import support for `shapely._geos`
- Logging/instrumentation improved in relevant modules to expose packaged-only failures more clearly.

### Validation / observed status
- Shapely `WinError 206` issue was mitigated.
- CuPy root cause was narrowed down and at least one missing dependency (`fastrlock`) was fixed.
- Phase 4 packaged failure was traced to missing `shapely._geos`.
- Packaged alignment issues were traced to missing `_version` for `sep_pjw`.
- At one point GPU detection/use appeared to recover in packaged logs, but packaged runs still showed additional Phase 3 / Phase 4 instability compared with non-compiled runs.

### Remaining caution
This thread involved several packaged-only import/runtime issues. Any future packaging work should preserve:
- frozen DLL search-path setup
- explicit hidden imports for fragile binary packages
- detailed packaged runtime logging

---

## 2026-03-12 — Qt filter dialog usability issue (small screens / scaling)

### Problem
A user reported that the **Start / OK** button in `zemosaic_filter_gui_qt.py` can be unreachable on some screen sizes or DPI scaling settings because it ends up outside the visible dialog area.

### Root cause identified
- The filter dialog right-side controls are stacked directly in a layout with **no `QScrollArea`**.
- The saved dialog geometry is restored without clamping to the current screen bounds.

### Planned fix
- Put the **right-side controls column** inside a `QScrollArea`.
- Keep the **OK/Cancel button box fixed outside** the scrollable area at the bottom of the dialog.
- Clamp restored geometry to the current screen `availableGeometry()` before applying it.

### Session status
- Root cause reviewed in code.
- Mission files prepared (`agent.md`, `followup.md`).
- No code change applied yet in this session.
