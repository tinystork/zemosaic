"""Package-aware resource resolution for ZeMosaic.

All bundled data (locales JSON, icon, opening GIF) is resolved through
:mod:`importlib.resources` so it keeps working from arbitrary working
directories and when the package is installed (including zipped wheels).

For normal (unzipped) installs, :func:`resource_path` returns the on-disk
``pathlib.Path``.  For zipped wheels it materializes the resource into a
per-user cache directory so APIs that require a real file path (e.g. QMovie,
QIcon) can consume it.  In PyInstaller-frozen builds it prefers the extracted
``_MEIPASS`` tree.
"""

from __future__ import annotations

import importlib.resources as _resources
import os
import sys
from pathlib import Path
from typing import Optional

PACKAGE = "zemosaic"


def _package_files():
    """Return the importlib.resources anchor for the ZeMosaic package."""
    return _resources.files(PACKAGE)


def _cache_root() -> Path:
    """Return the per-user cache directory used to materialize resources."""
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        root = Path(base)
    else:
        root = Path.home() / ".cache"
    return root / "zemosaic" / "resources"


def _frozen_candidate(*parts: str) -> Optional[Path]:
    """Return the extracted ``_MEIPASS`` path for a resource when frozen."""

    if not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    candidates = [
        Path(meipass) / PACKAGE / Path(*parts),
        Path(meipass) / Path(*parts),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resource_path(*parts: str) -> Path:
    """Return a filesystem path for the requested package resource.

    The resource must exist inside the ``zemosaic`` package (e.g.
    ``resource_path("locales", "en.json")``).  Raises ``FileNotFoundError`` if
    the resource is missing.
    """

    frozen = _frozen_candidate(*parts)
    if frozen is not None:
        return frozen

    traversable = _package_files().joinpath(*parts)
    if isinstance(traversable, Path):
        if not traversable.exists():
            raise FileNotFoundError(str(traversable))
        return traversable

    if not traversable.exists():
        raise FileNotFoundError("/".join(parts))

    # Zip-backed Traversable: extract once into a stable cache directory.
    target = _cache_root() / Path(*parts)
    if traversable.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if not target.exists():
        target.write_bytes(traversable.read_bytes())
    return target


def resource_path_optional(*parts: str) -> Optional[Path]:
    """Like :func:`resource_path` but return ``None`` when missing."""

    try:
        return resource_path(*parts)
    except (FileNotFoundError, ModuleNotFoundError, OSError, TypeError, ValueError):
        return None
