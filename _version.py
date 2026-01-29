"""Compatibility stub for sep_pjw in frozen builds.

sep_pjw imports a top-level `_version` module at runtime. In some PyInstaller
builds this module is missing, causing alignment failures. This stub provides
the minimal API expected by versioneer-style code.
"""

from __future__ import annotations

from typing import Any


def _detect_version() -> str:
    try:
        try:
            from importlib.metadata import version as _pkg_version
        except Exception:
            from importlib_metadata import version as _pkg_version  # type: ignore
        return _pkg_version("sep")
    except Exception:
        pass
    try:
        import sep  # type: ignore

        return str(getattr(sep, "__version__", "0+unknown"))
    except Exception:
        return "0+unknown"


__version__ = _detect_version()
# sep_pjw expects `from _version import version`
version = __version__


def get_versions() -> dict[str, Any]:
    return {"version": __version__}
