"""Lightweight wrappers around :mod:`os.path` helpers.

These functions centralize the handful of quick path checks that the
application performs (case-folding tokens, existence checks, etc.) so the
rest of the code base no longer calls :mod:`os.path` directly.
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

StrPath = str | PathLike[str] | Path

__all__ = [
    "casefold_path",
    "expand_to_path",
    "normpath_segments",
    "safe_path_exists",
    "safe_path_getsize",
    "safe_path_isdir",
    "safe_path_isfile",
]


def _coerce_to_string(pathish: StrPath | None, *, expanduser: bool = True) -> str:
    """Return a normalized string representation for *pathish*."""

    if pathish is None:
        return ""

    try:
        text = os.fspath(pathish)
    except TypeError:
        try:
            text = str(pathish)
        except Exception:
            return ""

    if isinstance(text, bytes):
        try:
            text = text.decode()
        except Exception:
            text = text.decode(errors="ignore")

    if not text:
        return ""

    if expanduser:
        try:
            text = os.path.expanduser(text)
        except Exception:
            pass
    return text


def safe_path_exists(pathish: StrPath | None, *, expanduser: bool = True) -> bool:
    """Best-effort ``os.path.exists`` wrapper that tolerates bad inputs."""

    text = _coerce_to_string(pathish, expanduser=expanduser)
    if not text:
        return False
    try:
        return os.path.exists(text)
    except Exception:
        return False


def safe_path_isfile(pathish: StrPath | None, *, expanduser: bool = True) -> bool:
    """Return True when *pathish* points to an existing file."""

    text = _coerce_to_string(pathish, expanduser=expanduser)
    if not text:
        return False
    try:
        return os.path.isfile(text)
    except Exception:
        return False


def safe_path_isdir(pathish: StrPath | None, *, expanduser: bool = True) -> bool:
    """Return True when *pathish* refers to a directory."""

    text = _coerce_to_string(pathish, expanduser=expanduser)
    if not text:
        return False
    try:
        return os.path.isdir(text)
    except Exception:
        return False


def safe_path_getsize(
    pathish: StrPath | None,
    *,
    expanduser: bool = True,
    default: int = 0,
) -> int:
    """Return the size of *pathish* in bytes, or *default* on failure."""

    text = _coerce_to_string(pathish, expanduser=expanduser)
    if not text:
        return default
    try:
        return int(os.path.getsize(text))
    except Exception:
        return default


def casefold_path(
    pathish: StrPath | None,
    *,
    absolute: bool = False,
    expanduser: bool = False,
) -> str:
    """Return a case-folded representation of *pathish* suitable for dict keys."""

    text = _coerce_to_string(pathish, expanduser=expanduser)
    if not text:
        return ""
    if absolute:
        try:
            text = os.path.abspath(text)
        except Exception:
            pass
    try:
        return os.path.normcase(text)
    except Exception:
        try:
            return os.path.normcase(str(pathish))
        except Exception:
            return ""


def normpath_segments(
    pathish: StrPath | None,
    *,
    absolute: bool = False,
    expanduser: bool = True,
) -> Tuple[str, ...]:
    """Return the normalized directory components for *pathish*."""

    text = _coerce_to_string(pathish, expanduser=expanduser)
    if not text:
        return ()
    try:
        normalized = os.path.normpath(os.path.abspath(text) if absolute else text)
    except Exception:
        normalized = text

    parts: list[str] = []
    for segment in normalized.split(os.sep):
        segment = segment.strip()
        if segment and segment != ".":
            parts.append(segment)
    return tuple(parts)


def expand_to_path(
    pathish: StrPath | None,
    *,
    expanduser: bool = True,
    expandvars: bool = True,
    resolve: bool = False,
) -> Optional[Path]:
    """Return a :class:`Path` object from *pathish* with best-effort expansion."""

    text = _coerce_to_string(pathish, expanduser=False)
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    if expanduser:
        try:
            text = os.path.expanduser(text)
        except Exception:
            pass
    if expandvars:
        try:
            text = os.path.expandvars(text)
        except Exception:
            pass
    try:
        path_obj = Path(text)
    except Exception:
        return None
    if resolve:
        try:
            return path_obj.resolve(strict=False)
        except Exception:
            return path_obj
    return path_obj
