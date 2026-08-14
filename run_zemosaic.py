"""Historical ZeMosaic launcher — thin compatibility wrapper.

This file exists so a checkout can still be started the old way
(``python run_zemosaic.py``) without installing the package.  It only makes the
in-repo ``src/`` layout importable and then delegates to the real bootstrap
``zemosaic._app:main``.

For a proper installation use ``pip install .`` and run the ``zemosaic``
console script (or ``python -m zemosaic``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

_src_str = str(_SRC)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from zemosaic._app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
