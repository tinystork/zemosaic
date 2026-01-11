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
import os
import sys
from pathlib import Path


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
        candidates: list[str] = []
        for key, value in os.environ.items():
            if key.upper().startswith("CUDA_PATH_V") and value:
                candidates.append(value)
        cuda_path = sorted(candidates)[-1] if candidates else None
    if not cuda_path:
        return

    bin_dir = Path(cuda_path) / "bin"
    if not bin_dir.is_dir():
        return

    try:
        add_dll_directory(str(bin_dir))
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
