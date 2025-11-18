


```markdown
# FOLLOW-UP TASKS — RESTORE ICONS FOR TK & QT GUIS

This file lists concrete, ordered tasks for fixing the icons.  
Do **not** change anything beyond what is described here.

---

## 1. Centralise base directory detection

**File:** `zemosaic_utils.py` (or a similar core utility module if it already exists)

1. Add (or update) a function:

   ```python
   from pathlib import Path
   import sys
   import os

   def get_app_base_dir() -> Path:
       """
       Return the root directory where ZeMosaic resources live (including the 'icon' folder).
       Must be robust across:
       - source checkout,
       - installed package,
       - frozen/PyInstaller builds.
       """
       # 1) PyInstaller / frozen
       if getattr(sys, "frozen", False):
           base = Path(sys.executable).resolve().parent
           # If a 'zemosaic' subfolder exists (onedir), prefer it.
           candidate = base / "zemosaic"
           return candidate if candidate.is_dir() else base

       # 2) Try to locate the installed package
       try:
           import importlib.util
           spec = importlib.util.find_spec("zemosaic")
           if spec and spec.origin:
               return Path(spec.origin).resolve().parent
       except Exception:
           pass

       # 3) Fallback: current file directory
       try:
           return Path(__file__).resolve().parent
       except Exception:
           return Path(os.getcwd())
````

2. Ensure this function:

   * has no side effects on import,
   * does not raise on failure (always returns some `Path`).

3. If `get_app_base_dir` already exists, **update** it to match this behaviour instead of creating duplicates.

---

## 2. Normalise Tk icon application in the main GUI

**File:** `zemosaic_gui.py`

1. Import the helper:

   ```python
   from zemosaic_utils import get_app_base_dir  # adjust import path if needed
   ```

2. Replace the current direct icon logic in `ZeMosaicGUI.__init__`:

   ```python
   # OLD (example)
   base_path = os.path.dirname(os.path.abspath(__file__))
   icon_path = os.path.join(base_path, "icon", "zemosaic.ico")
   # ...
   ```

   with a call to a small local helper that mirrors `_apply_zemosaic_icon_to_tk`:

   ```python
   def _apply_zemosaic_icon_to_tk(window):
       import platform
       from tkinter import PhotoImage
       from pathlib import Path

       system_name = platform.system().lower()
       is_windows = system_name == "windows"

       base_path = get_app_base_dir()
       icon_dir = base_path / "icon"

       ico_path = icon_dir / "zemosaic.ico"
       png_candidates = [
           icon_dir / "zemosaic_64x64.png",
           icon_dir / "zemosaic_icon.png",
           icon_dir / "zemosaic.png",
       ]

       try:
           if is_windows and ico_path.is_file():
               window.iconbitmap(default=str(ico_path))
           else:
               png_path = next((p for p in png_candidates if p.is_file()), None)
               if png_path:
                   photo = PhotoImage(file=str(png_path))
                   window.iconphoto(True, photo)
       except Exception as exc:
           print(f"[TkMain] Unable to apply ZeMosaic icon: {exc}")
   ```

3. In `ZeMosaicGUI.__init__`, call:

   ```python
   try:
       _apply_zemosaic_icon_to_tk(self.root)
   except Exception as e_icon:
       print(f"AVERT GUI: icône ZeMosaic non appliquée ({e_icon})")
   ```

4. Remove the old hard-coded `iconbitmap` / `os.path.dirname(__file__)` code to avoid conflicting behaviours.

---

## 3. Reuse the same Tk icon logic in the Filter GUI

**File:** `zemosaic_filter_gui.py`

1. Inside the function that defines `_apply_zemosaic_icon_to_tk(window)` (already present),
   change the implementation to use the same pattern as in `zemosaic_gui.py`:

   * Use `get_app_base_dir()` instead of `get_app_base_dir` being undefined or `__file__`/`os.getcwd()`.
   * Use the same list of candidate files and platform rules.

   Example:

   ```python
   from zemosaic_utils import get_app_base_dir  # at top-level of the module

   def _apply_zemosaic_icon_to_tk(window):
       try:
           import platform
           from tkinter import PhotoImage
           from pathlib import Path

           system_name = platform.system().lower()
           is_windows = system_name == "windows"

           base_path = get_app_base_dir()
           icon_dir = base_path / "icon"

           ico_path = icon_dir / "zemosaic.ico"
           png_candidates = [
               icon_dir / "zemosaic_64x64.png",
               icon_dir / "zemosaic_icon.png",
               icon_dir / "zemosaic.png",
           ]

           if is_windows and ico_path.is_file():
               window.iconbitmap(default=str(ico_path))
           else:
               png_path = next((p for p in png_candidates if p.is_file()), None)
               if png_path:
                   photo = PhotoImage(file=str(png_path))
                   window.iconphoto(True, photo)
       except Exception as exc:
           print(f"[FilterGUI] Impossible d'appliquer l'icône ZeMosaic: {exc}")
   ```

2. Keep the call `_apply_zemosaic_icon_to_tk(root)` as it is, right after creating the Tk window.

3. Do not modify any other parts of the filter GUI (plots, WCS logic, etc.).

---

## 4. Fix and unify Qt icon loading in the main Qt GUI

**File:** `zemosaic_gui_qt.py`

1. Import the helper:

   ```python
   from zemosaic_utils import get_app_base_dir  # adjust import if needed
   from pathlib import Path
   ```

2. Ensure `_load_zemosaic_qicon()` is defined as:

   ```python
   from PySide6.QtGui import QIcon

   def _load_zemosaic_qicon() -> QIcon | None:
       try:
           icon_dir = get_app_base_dir() / "icon"
       except Exception:
           return None

       candidates = [
           icon_dir / "zemosaic.ico",
           icon_dir / "zemosaic_64x64.png",
           icon_dir / "zemosaic_icon.png",
           icon_dir / "zemosaic.png",
       ]

       for path in candidates:
           try:
               if path.is_file():
                   return QIcon(str(path))
           except Exception:
               continue
       return None
   ```

3. In `ZeMosaicQtMainWindow.__init__` (already calling `_load_zemosaic_qicon()`),
   keep the logic:

   ```python
   icon = _load_zemosaic_qicon()
   if icon is not None:
       self.setWindowIcon(icon)
   ```

4. Do not modify any other part of `ZeMosaicQtMainWindow` or the Qt main UI.

---

## 5. Apply the same Qt icon logic in the Qt Filter GUI

**File:** `zemosaic_filter_gui_qt.py`

1. Import `get_app_base_dir` and `QIcon` at top-level if not already done:

   ```python
   from zemosaic_utils import get_app_base_dir  # adjust import path if needed
   from PySide6.QtGui import QIcon
   ```

2. Ensure this module also has a `_load_zemosaic_qicon()` identical (or trivially shared)
   with the one in `zemosaic_gui_qt.py`:

   ```python
   def _load_zemosaic_qicon() -> QIcon | None:
       try:
           icon_dir = get_app_base_dir() / "icon"
       except Exception:
           return None

       candidates = [
           icon_dir / "zemosaic.ico",
           icon_dir / "zemosaic_64x64.png",
           icon_dir / "zemosaic_icon.png",
           icon_dir / "zemosaic.png",
       ]

       for path in candidates:
           try:
               if path.is_file():
                   return QIcon(str(path))
           except Exception:
               continue
       return None
   ```

3. In the main dialog class of the Qt filter GUI (e.g. `ZeMosaicQtFilterDialog` or equivalent),
   call this helper in `__init__`:

   ```python
   icon = _load_zemosaic_qicon()
   if icon is not None:
       self.setWindowIcon(icon)
   ```

   Place this at the start of `__init__`, before other UI setup if possible.

4. Do not change any other filter logic (tree, groups, WCS, etc.).

---

## 6. Optional: apply icons to other Tk tools (only if trivial)

*Only if it can be done without risk and minimal code:*

* For tools like `zequalityMT.py` or `lecropper.py` that create a Tk window,
  you may reuse `_apply_zemosaic_icon_to_tk(window)` by:

  * importing `get_app_base_dir`,
  * duplicating the minimal helper locally,
  * calling it right after `tk.Tk()` or `tk.Toplevel()`.

If in doubt, **skip this step** to avoid regressions.

---

## 7. Manual sanity checks (no code changes)

After code edits:

1. Run `python run_zemosaic.py --backend=tk`

   * Confirm the main Tk window has the icon (where the OS supports it).
   * Open the Tk filter GUI and confirm the icon is also applied.

2. Run `python run_zemosaic.py --backend=qt`

   * Confirm the main Qt window has the icon.
   * Open the Qt filter GUI and confirm the icon is also applied.

3. Temporarily rename the `icon/` directory and confirm:

   * The application still starts,
   * Only warnings are printed (no crashes).

Do **not** commit additional behavioural or cosmetic changes as part of this task.

```
