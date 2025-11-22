"""Compatibility shim for legacy imports.

The canonical implementation now lives in ``core.cuda_utils`` to keep shared
helpers in a single place.  Both import paths continue to work to avoid
breaking existing code.
"""

from core.cuda_utils import *  

