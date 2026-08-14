"""ZeMosaic — open-source astronomical mosaic builder.

The official runtime is the Qt GUI, launched via :func:`zemosaic._app.main`
(the ``zemosaic`` console script) or ``python -m zemosaic``.

Bundled data (locales, icon, opening GIF) lives inside this package and is
resolved with :mod:`importlib.resources`; user configuration and logs live
outside the package (see ``zemosaic.zemosaic_utils``).
"""

from __future__ import annotations

__version__ = "4.5.0"

__all__ = ["__version__"]
