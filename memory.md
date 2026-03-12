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

---

## 2026-03-12 — Packaging/docs alignment for GPU vs CPU-only builds

### Topic
Clarified how Windows/macOS/Linux packaged builds decide whether CuPy/CUDA support is included, and aligned the helper scripts / installer documentation with the current build layout.

### Problem
- Build/release docs did not clearly distinguish:
  - `requirements.txt` (current working GPU-enabled dependency set, including `cupy-cuda12x`)
  - `requirements_no_gpu.txt` (new CPU-only dependency set for smaller artifacts)
- Build helpers and installer assumptions needed to match the real packaging flow.
- `zemosaic_installer.iss` previously targeted obsolete `compile\...` paths instead of the current `dist\ZeMosaic\...` output layout.

### Changes made
- Added `requirements_no_gpu.txt`
  - mirrors the base dependency set
  - intentionally excludes CuPy so a smaller CPU-only packaged build can be produced
- Updated `compile/compile_zemosaic._win.bat`
  - default requirements file is again `requirements.txt`
  - supports overriding via `ZEMOSAIC_REQUIREMENTS_FILE`
  - message now states GPU support depends on the chosen requirements file
- Updated `compile/build_zemosaic_posix.sh`
  - default requirements file is again `requirements.txt`
  - supports overriding via `ZEMOSAIC_REQUIREMENTS_FILE`
  - installs `pyinstaller-hooks-contrib`
  - cleans `build/` and `dist/`
  - message now states GPU support depends on the chosen requirements file
- Updated `zemosaic_installer.iss`
  - installer now packages `dist\ZeMosaic\*`
  - `Icons` / `Run` point to `{app}\ZeMosaic.exe`
  - comments explicitly state that Inno Setup does not choose the CUDA package; it only packages whatever build already exists in `dist\ZeMosaic`
- Updated `README.md`
  - documents the default GPU-enabled path using `requirements.txt`
  - documents CPU-only builds using `requirements_no_gpu.txt`
  - explains that `.iss` does not select CUDA/CuPy version
  - adds Windows GitHub release guidance (publish zipped `dist\ZeMosaic` as a release asset, do not commit `dist/`)

### Validation performed
- `bash -n compile/build_zemosaic_posix.sh` passed.
- Static review confirmed Windows helper / POSIX helper / README are now consistent on:
  - default = `requirements.txt`
  - smaller package option = `requirements_no_gpu.txt`
  - installer packages the already-built output

### Important packaging note
- Current intended behavior remains:
  - use `requirements.txt` for the existing GPU-enabled build path (`cupy-cuda12x`)
  - use `requirements_no_gpu.txt` only when a smaller CPU-only package is wanted
- `zemosaic_installer.iss` is ignored by Git in this repo (`.gitignore` has `*.iss`), so versioning it requires `git add -f zemosaic_installer.iss` or changing ignore rules.

---

## 2026-03-12 — Future idea: installer with GPU/CuPy auto-detection

### Topic
Exploration only, no mission started yet.

### User intent
- Keep the current working GPU path based on `requirements.txt` / `cupy-cuda12x`.
- Consider a future Windows installer able to detect the target machine and install/download the appropriate GPU support automatically.
- Defer implementation for now.

### Assessment given
- This is considered viable, but not a small change.
- Estimated scope:
  - minimal viable approach: high complexity
  - robust/maintainable approach: very high complexity

### Recommended architecture discussed
- Prefer a CPU-only base installer plus an optional GPU enablement step.
- Detect:
  - Windows x64 environment
  - NVIDIA GPU presence
  - driver / CUDA compatibility level
- Then either:
  - download a prebuilt GPU add-on from GitHub Releases, or
  - install the matching CuPy package dynamically
- Recommended direction was to avoid a fully dynamic Inno Setup-only solution at first, and instead use:
  - a simple installer
  - plus a post-install/bootstrap GPU activation step with clear fallbacks

### Why this was deferred
- Requires coordinated changes across:
  - installer/bootstrap flow
  - release packaging strategy
  - GPU compatibility detection logic
  - runtime fallback behavior
  - documentation and testing matrix
- User asked to postpone this work and possibly revisit it later.
