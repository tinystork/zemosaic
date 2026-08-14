"""Enable ``python -m zemosaic`` as an alias for the ``zemosaic`` entry point."""

from __future__ import annotations

import sys

from ._app import main

if __name__ == "__main__":
    sys.exit(main())
