#!/usr/bin/env python3
"""M5-B controlled RSS probe — bounded assembly in an ISOLATED process.

Measures, for one small synthetic store (4 cells, 24x24, 3 channels):

* baseline RSS after imports ONLY (fresh process, no payload);
* RSS immediately before the bounded assembly (store + preflight, metadata);
* peak RSS right after the real feathered assembly;
* observed delta, the process virtual size and the session's own mapped-byte
  counter, next to the metadata-only preflight model (``required_bytes``).

The probe is deliberately small and bounded (no OS stress, no large corpus).
``ru_maxrss`` already includes Python/NumPy/astropy imports and every touched
mmap page, so the delta is an OBSERVATION for this host, never a universal
guarantee or a substitute for the model bound.
"""

from __future__ import annotations

import json
import resource
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _statm_kib() -> dict:
    try:
        fields = Path("/proc/self/statm").read_text().split()
        page = Path("/proc/self/statm")
        page_kib = resource.getpagesize() // 1024
        return {
            "virtual_kib": int(fields[0]) * page_kib,
            "resident_kib": int(fields[1]) * page_kib,
        }
    except Exception as exc:  # pragma: no cover - non-Linux
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    import test_zegrid_bounded_assembly_m5 as m5a

    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import bounded_assembly as zbas

    baseline = _rss_kib()
    baseline_statm = _statm_kib()
    results = [
        m5a.complete_result(0, 0, seed=0),
        m5a.complete_result(0, 1, seed=1),
        m5a.complete_result(1, 0, seed=2),
        m5a.complete_result(1, 1, seed=3),
    ]
    with tempfile.TemporaryDirectory(prefix="m5b-rss-") as tmp:
        tmp = Path(tmp)
        root, store = m5a.seeded_store(tmp, results)
        store.close()
        canvas = m5a.make_canvas()
        st = m5a.reopen_read_only(root)
        try:
            access = zbas.StoreAssemblyAccess(st, max_resident_cells=2)
            bounded = access.results([r["cell_id"] for r in results])
            plan = zbas.preflight_assembly(
                bounded.session,
                [r["cell_id"] for r in results],
                canvas_hw=(m5a.H, m5a.W),
                available_bytes=None,
                require_budget=False,
                gauge_mode="per_cell",
                grid=(m5a.NX, m5a.NY),
            )
            before = _rss_kib()
            before_statm = _statm_kib()
            assembled, info = zz._assemble_cells(
                canvas,
                m5a.NX,
                m5a.NY,
                "per_cell",
                bounded.cores,
                bounded.descriptors,
                halo_px=m5a.HALO,
                patches=bounded.patches,
            )
            peak = _rss_kib()
            peak_statm = _statm_kib()
            stats = access.stats()
            report = {
                "schema": "zemosaic.zegrid.m5b.rss_probe.v1",
                "corpus": {
                    "cells": len(results),
                    "canvas_hw": [m5a.H, m5a.W],
                    "channels": m5a.CHANNELS,
                    "halo_px": m5a.HALO,
                },
                "rss_kib": {
                    "baseline_imports": baseline,
                    "before_assembly": before,
                    "peak_after_assembly": peak,
                    "delta_before_to_peak": int(peak - before),
                },
                "statm_kib": {
                    "baseline": baseline_statm,
                    "before_assembly": before_statm,
                    "peak_after_assembly": peak_statm,
                },
                "assembly": {
                    "model_required_bytes": int(plan["required_bytes"]),
                    "model_canvas_bytes": int(plan["canvas"]["total_bytes"]),
                    "model_resident_mapped_bytes": int(
                        plan["cells"]["resident_mapped_bytes"]
                    ),
                    "peak_live_cells": int(stats["peak_live_cells"]),
                    "peak_live_mapped_bytes": int(stats["peak_live_mapped_bytes"]),
                    "offsets_applied": bool(info.get("offsets")),
                    "science_shape": list(np.shape(assembled.science)),
                },
                "note": (
                    "single-host observation in a fresh process; ru_maxrss "
                    "includes imports and touched mmap pages. It is not a "
                    "universal guarantee and does not replace the model bound."
                ),
            }
        finally:
            st.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
