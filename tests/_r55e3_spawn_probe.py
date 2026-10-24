"""Hermetic spawn probe for R55-E3 (no heavy imports at module import time).

This helper is imported by a ``multiprocessing`` spawn child. Keeping it free of
package-level imports lets the child prove that the R55 modules themselves are
CuPy-lazy before it imports the historical engine.
"""

from __future__ import annotations

import multiprocessing as mp
import sys


def probe(queue) -> None:
    result = {
        "start_method": mp.get_start_method(),
        "cupy_after_r55_import": None,
        "cupy_after_engine_import": None,
        "engine_import_ok": False,
        "tokens": [],
        "invalid_refused": False,
    }
    from zemosaic.core.zegrid import gpu_readiness as child_gr  # noqa: F401
    from zemosaic.core.zegrid import parallel as child_zp  # noqa: F401
    from zemosaic.core.zegrid import scheduler as child_zs

    result["cupy_after_r55_import"] = "cupy" in sys.modules
    for token in child_zs.MODES:
        result["tokens"].append(child_zs.normalize_scheduler_mode(token))
    try:
        child_zs.normalize_scheduler_mode("BOGUS")
    except child_zs.InvalidSchedulerMode:
        result["invalid_refused"] = True

    # The historical engine import may import CuPy when installed (pre-existing
    # ``parallel_utils`` behaviour). Recorded, never denied.
    from zemosaic import zemosaic_zegrid_mode as child_zz  # noqa: F401

    result["engine_import_ok"] = True
    result["cupy_after_engine_import"] = "cupy" in sys.modules
    queue.put(result)
