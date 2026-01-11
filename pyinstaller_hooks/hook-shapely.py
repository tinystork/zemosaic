"""PyInstaller hook for Shapely (ZeMosaic).

Root issue: Shapely wheels ship native DLLs under `shapely.libs` and add that
directory via `os.add_dll_directory()` at import time (delvewheel patch).

In onefile builds, that call can fail on some Windows setups (WinError 206),
crashing the app when Shapely is imported (e.g. via `reproject`).

Workaround: bundle Shapely's DLLs *next to the Shapely extension module* (the
`shapely/` package directory). Then the Windows loader can resolve dependencies
from the extension module's directory without needing `shapely.libs`, and the
delvewheel patch becomes a no-op because `shapely.libs` is absent.
"""

from __future__ import annotations

from PyInstaller.utils.hooks import collect_dynamic_libs

# Place all collected DLLs under `shapely/` instead of `shapely.libs/`.
binaries = [(src, "shapely") for (src, _dest) in collect_dynamic_libs("shapely")]

