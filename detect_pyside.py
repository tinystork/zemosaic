"""Small helper to debug PySide6 detection issues.

Run with the same interpreter you use for ZeMosaic, e.g.::

    python detect_pyside.py

It prints the interpreter path, sys.path entries, and whether importlib can find
PySide6.  Share its output when debugging backend selection problems.
"""
from importlib import util
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("sys.path entries:")
for entry in sys.path:
    print("  ", entry)

spec = util.find_spec("PySide6")
print("\nimportlib.util.find_spec('PySide6') ->", spec)
if spec is not None:
    print("PySide6 origin:", spec.origin)
else:
    try:
        import PySide6  # type: ignore  # noqa: F401
    except Exception as exc:
        print("Direct import failed:", exc)
    else:
        print("Direct import succeeded via fallback import")
