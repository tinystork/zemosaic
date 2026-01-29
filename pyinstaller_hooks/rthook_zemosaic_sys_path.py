"""PyInstaller runtime hook for ZeMosaic.

Goal: avoid importing stray modules from the user's TEMP directory.

`run_zemosaic.py` prepends the parent directory of its location to `sys.path`.
In onefile builds, the app is extracted under `%TEMP%\\_MEIxxxxx`, so the parent
directory becomes `%TEMP%`. If `%TEMP%` is prepended, any accidental files like
`%TEMP%\\zemosaic_worker.py` can shadow the bundled modules and crash at startup.

This hook runs *before* the entry-point script and ensures the `%TEMP%` parent
directory is present (so `run_zemosaic.py` won't re-insert it at position 0),
but only at the end of `sys.path`.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path


LOGGER = logging.getLogger("zemosaic.rthook")


def _rthook_info(message: str) -> None:
    try:
        LOGGER.info(message)
    except Exception:
        pass
    try:
        sys.stderr.write(f"[rthook] {message}\n")
    except Exception:
        pass


def _patch_add_dll_directory_for_long_paths() -> None:
    """Patch `os.add_dll_directory()` to retry with extended-length paths on WinError 206.

    This mainly targets frozen apps started from:
    - subst drives
    - deep paths under user profiles
    - environments where the resolved path exceeds MAX_PATH

    It keeps behavior identical unless the first call fails with WinError 206.
    """

    if os.name != "nt":
        return

    original_add_dll_directory = getattr(os, "add_dll_directory", None)
    if original_add_dll_directory is None:
        return

    if getattr(original_add_dll_directory, "_zemosaic_patched", False):
        return

    def to_extended_length_path(path: str) -> str:
        if path.startswith("\\\\?\\"):
            return path
        if path.startswith("\\\\"):
            # UNC path: \\server\share\dir -> \\?\UNC\server\share\dir
            return "\\\\?\\UNC\\" + path.lstrip("\\")
        return "\\\\?\\" + path

    class _NoopAddedDllDirectory:
        def close(self) -> None:  # pragma: no cover - defensive
            return

        def __enter__(self):  # pragma: no cover - defensive
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - defensive
            self.close()

    def is_shapely_libs_dir(path: str) -> bool:
        if not getattr(sys, "frozen", False):
            return False
        meipass = getattr(sys, "_MEIPASS", None)
        if not meipass:
            return False
        try:
            candidate = str(path).replace("/", "\\").rstrip("\\")
            internal = str(meipass).replace("/", "\\").rstrip("\\")
            if not candidate.lower().startswith(internal.lower()):
                return False
            return Path(candidate).name.lower() == "shapely.libs"
        except Exception:
            return False

    def patched_add_dll_directory(path: str):  # type: ignore[override]
        try:
            return original_add_dll_directory(path)
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror == 206:
                try:
                    resolved = str(Path(path).resolve())
                except Exception:
                    resolved = path
                extended = to_extended_length_path(resolved)
                if extended != path:
                    try:
                        return original_add_dll_directory(extended)
                    except OSError as exc2:
                        winerror = getattr(exc2, "winerror", winerror)

            # In frozen builds we may deliberately relocate DLLs out of `*.libs` folders
            # (e.g. Shapely) so the directory can be missing or un-addable; do not crash.
            if winerror in (2, 3, 206) and is_shapely_libs_dir(path):
                return _NoopAddedDllDirectory()

            raise

    patched_add_dll_directory._zemosaic_patched = True  # type: ignore[attr-defined]
    os.add_dll_directory = patched_add_dll_directory  # type: ignore[assignment]


_patch_add_dll_directory_for_long_paths()


def _ensure_frozen_internal_dll_search_path() -> None:
    """Make bundled DLLs discoverable in frozen Windows builds.

    PyInstaller onedir puts the runtime under `<app>/_internal` and sets `sys._MEIPASS` to it.
    Many third-party wheels (CuPy, Shapely, rasterio, ...) rely on `os.add_dll_directory()`
    to add their `.libs` folders or wheel lib directories.

    If the internal folder is not on the DLL search path, GPU detection can fail (CuPy import),
    and some packages can crash at import time.
    """

    if os.name != "nt":
        return
    if not getattr(sys, "frozen", False):
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    internal_dir = Path(meipass)
    if not internal_dir.is_dir():
        return

    _rthook_info(f"Frozen MEIPASS: {internal_dir}")
    added_dirs: set[str] = set()

    def _add_dll_dir(path: Path) -> None:
        try:
            key = str(path)
        except Exception:
            return
        if key in added_dirs:
            return
        try:
            add_dll_directory(key)
            added_dirs.add(key)
            _rthook_info(f"Added DLL dir: {key}")
        except Exception:
            return
        try:
            os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

    # Ensure the internal root is searchable (CuPy bundles many CUDA DLLs at that level when frozen).
    _add_dll_dir(internal_dir)

    # Many wheels ship native deps in `*.libs` folders (delvewheel).
    try:
        for libs_dir in sorted(internal_dir.glob("*.libs")):
            if not libs_dir.is_dir():
                continue
            _add_dll_dir(libs_dir)
    except Exception:
        pass

    # CuPy wheels often bundle CUDA runtime DLLs under `cupy/.data/**`.
    try:
        for pkg_name in ("cupy", "cupy_backends"):
            pkg_dir = internal_dir / pkg_name
            if not pkg_dir.is_dir():
                continue
            try:
                for dll_path in pkg_dir.rglob("*.dll"):
                    _add_dll_dir(dll_path.parent)
            except Exception:
                continue
    except Exception:
        pass


_ensure_frozen_internal_dll_search_path()


def _move_meipass_parent_to_sys_path_end() -> None:
    if not getattr(sys, "frozen", False):
        return
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return
    try:
        temp_parent = str(Path(meipass).resolve().parent)
    except Exception:
        return

    # Remove all occurrences first (including accidental position 0), then
    # append once at the end so it cannot shadow bundled modules.
    sys.path[:] = [p for p in sys.path if p != temp_parent]
    sys.path.append(temp_parent)


_move_meipass_parent_to_sys_path_end()

def _ensure_cuda_dll_search_path() -> None:
    """Help CuPy find CUDA DLLs in frozen apps on Windows.

    CuPy may rely on CUDA runtime/toolkit DLLs (e.g. `cudart64_*.dll`, `nvrtc64_*.dll`)
    that are *not* bundled by default. When users have CUDA Toolkit installed, those
    DLLs typically live under `%CUDA_PATH%\\bin`.

    In some frozen contexts, the DLL search path might not include that directory.
    """

    if os.name != "nt":
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return

    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        # Fallback: CUDA_PATH_V12_8, CUDA_PATH_V12_7, ...
        candidates: list[tuple[tuple[int, ...], str]] = []
        pattern = re.compile(r"^CUDA_PATH_V(\d+)(?:_(\d+))?$", re.IGNORECASE)
        for key, value in os.environ.items():
            if not value:
                continue
            match = pattern.match(key)
            if not match:
                continue
            major = int(match.group(1))
            minor = int(match.group(2) or 0)
            candidates.append(((major, minor), value))
        cuda_path = max(candidates, default=None, key=lambda item: item[0])[1] if candidates else None
    if not cuda_path:
        return

    bin_dir = Path(cuda_path) / "bin"
    if not bin_dir.is_dir():
        return

    try:
        add_dll_directory(str(bin_dir))
        _rthook_info(f"Added DLL dir: {bin_dir}")
    except Exception:
        return

    # Also prepend to PATH for sub-processes (worker) and non-PEP587 loaders.
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_ensure_cuda_dll_search_path()


def _patch_find_spec_for_cupy() -> None:
    """Make `importlib.util.find_spec("cupy")` reliable in frozen apps.

    Some frozen environments can import optional deps successfully but still
    return None from `find_spec()`. ZeMosaic uses `find_spec("cupy")` to decide
    whether GPU is available; this patch makes that probe fall back to a direct
    import for CuPy only.
    """

    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, package: str | None = None):  # type: ignore[override]
        spec = original_find_spec(name, package)
        if spec is not None:
            return spec
        if name != "cupy":
            return None
        try:
            import cupy  # type: ignore
        except Exception:
            return None
        return getattr(cupy, "__spec__", None) or original_find_spec(name, package)

    importlib.util.find_spec = patched_find_spec  # type: ignore[assignment]


_patch_find_spec_for_cupy()
