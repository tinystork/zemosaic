#!/usr/bin/env python3
"""Short isolated A/B witness: gauge counts/pairs worker decoupling.

Mission ZM-GAUGE-COUNTS-DECOUPLING-20261115 (phase implementation).

This is a MEASUREMENT witness, not a product run and not a Windows proof:

* real local FITS (M106 lights subset) decoded by the PRODUCT decoder;
* same inputs/config for every arm, only the worker counts change;
* bit-for-bit payload parity against the default OFF arm;
* the counts/pairs sub-phase seconds are taken from the EXISTING diagnostics;
* parent + child CPU time from ``resource.getrusage`` (Linux only).

It intentionally stays small (default 10 frames, 3 bounded repetitions) so it
never competes with the live M33 run and never stresses RAM. It writes NOTHING
into any product output directory: the gauge cache is a throwaway temp dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import photometric as zphot  # noqa: E402
from zemosaic.core.zegrid.executor import ExecutorConfig  # noqa: E402

LIGHTS = Path("/home/tristan/M106/lights")


def _hash_array(value) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _payload(gauge, ids) -> dict:
    return {
        "reference_index": int(gauge.reference_index),
        "ids": tuple(str(v) for v in ids),
        "coefficients_sha256": _hash_array(gauge.coefficients),
        "weights_sha256": _hash_array(gauge.weights),
        "norm_active_sha256": _hash_array(gauge.norm_active),
        "weight_active_sha256": _hash_array(gauge.weight_active),
        "exclusions": tuple(
            (int(e.index), str(e.stage), str(e.reason), str(e.detail))
            for e in gauge.exclusions
        ),
    }


def _cpu():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        float(ru.ru_utime + ru.ru_stime),
        float(rc.ru_utime + rc.ru_stime),
    )


def _one(descs, canvas, config, *, workers, counts_workers):
    diag: dict = {}
    cache = tempfile.mkdtemp(prefix="zm-gauge-counts-ab-")
    p0, c0 = _cpu()
    t0 = time.perf_counter()
    try:
        gauge, ids = zphot.compute_global_gauge(
            descs,
            canvas,
            zz._gauge_decode,
            config,
            cache,
            workers=int(workers),
            counts_workers=counts_workers,
            diagnostics=diag,
        )
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    wall = time.perf_counter() - t0
    p1, c1 = _cpu()
    phases = diag.get("phases") or {}
    return {
        "workers": int(workers),
        "counts_workers": (
            None if counts_workers is None else int(counts_workers)
        ),
        "payload": _payload(gauge, ids),
        "diagnostics": {
            "workers": diag.get("workers"),
            "counts_workers": diag.get("counts_workers"),
            "executor": diag.get("executor"),
            "fallback": diag.get("fallback"),
            "phases": {
                name: {
                    "executor": (phases.get(name) or {}).get("executor"),
                    "units": (phases.get(name) or {}).get("units"),
                    "seconds": (phases.get(name) or {}).get("seconds"),
                    "seconds_per_unit": (phases.get(name) or {}).get(
                        "seconds_per_unit"
                    ),
                    "workers": (phases.get(name) or {}).get("workers"),
                    "fallback_reason": (phases.get(name) or {}).get(
                        "fallback_reason"
                    ),
                }
                for name in ("counts", "pairs")
            },
            "instrumentation_memory": (
                (diag.get("instrumentation") or {}).get("memory") or {}
            ),
            "instrumentation_totals": (
                (diag.get("instrumentation") or {}).get("totals") or {}
            ),
        },
        "wall_s": wall,
        "parent_cpu_s": p1 - p0,
        "children_cpu_s": c1 - c0,
    }


def _median_run(runs, key):
    values = [r[key] for r in runs if r.get(key) is not None]
    return statistics.median(values) if values else None


def _spawn_payload(q, descs, canvas, config):
    multiprocessing.set_start_method("spawn", force=True)
    try:
        result = _one(descs, canvas, config, workers=1, counts_workers=2)
        q.put({"status": "ok", "result": result})
    except BaseException as exc:  # noqa: BLE001
        q.put({"status": "error", "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if not LIGHTS.is_dir():
        print(f"absent corpus: {LIGHTS}", file=sys.stderr)
        return 2
    frames, _rejected = zg.read_manifest(LIGHTS)
    descs = sorted(frames, key=lambda f: f.frame_id)[: int(args.frames)]
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()
    model = zphot.gauge_memory_model(
        canvas,
        descs,
        normalization=getattr(config, "normalization", None),
        weighting=getattr(config, "weighting", None),
    )
    fc = int(zphot.gauge_counts_footprint_bytes(model))

    arms = [
        ("off_w1", 1, None),
        ("on_nc2_w1", 1, 2),
        ("on_nc4_w1", 1, 4),
        ("on_nc2_pairs2", 2, 2),
    ]
    runs: dict[str, list[dict]] = {name: [] for name, _w, _nc in arms}
    for _rep in range(int(args.reps)):
        for name, workers, counts in arms:
            runs[name].append(
                _one(descs, canvas, config, workers=workers, counts_workers=counts)
            )

    ref = runs["off_w1"][0]["payload"]
    summary = {}
    for name, workers, counts in arms:
        entries = runs[name]
        payload = entries[0]["payload"]
        parity = all(entry["payload"] == ref for entry in entries)
        phase_medians = {}
        for phase in ("counts", "pairs"):
            phase_medians[phase] = statistics.median(
                entry["diagnostics"]["phases"][phase]["seconds"] or 0.0
                for entry in entries
            )
        summary[name] = {
            "workers": workers,
            "counts_workers": counts,
            "reps": len(entries),
            "payload_bit_equal_to_off": bool(parity),
            "reference_index": payload["reference_index"],
            "ids": payload["ids"],
            "exclusions": payload["exclusions"],
            "coefficients_sha256": payload["coefficients_sha256"],
            "weights_sha256": payload["weights_sha256"],
            "norm_active_sha256": payload["norm_active_sha256"],
            "weight_active_sha256": payload["weight_active_sha256"],
            "wall_s_median": _median_run(entries, "wall_s"),
            "parent_cpu_s_median": _median_run(entries, "parent_cpu_s"),
            "children_cpu_s_median": _median_run(entries, "children_cpu_s"),
            "counts_seconds_median": phase_medians["counts"],
            "pairs_seconds_median": phase_medians["pairs"],
            "counts_executor": entries[0]["diagnostics"]["phases"]["counts"]["executor"],
            "pairs_executor": entries[0]["diagnostics"]["phases"]["pairs"]["executor"],
            "fallback": entries[0]["diagnostics"]["fallback"],
            "counts_worker_field": entries[0]["diagnostics"]["phases"]["counts"].get("workers"),
            "pairs_worker_field": entries[0]["diagnostics"]["phases"]["pairs"].get("workers"),
            "memory": entries[0]["diagnostics"]["instrumentation_memory"],
        }

    # Bounded SPAWN witness (Linux spawn is NOT the Windows qualification).
    spawn_result = {"status": "not_run"}
    try:
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(
            target=_spawn_payload, args=(q, descs[:4], zg.build_canvas(descs[:4]), config)
        )
        proc.start()
        proc.join(timeout=900)
        if proc.exitcode != 0 or q.empty():
            spawn_result = {
                "status": "error",
                "exitcode": proc.exitcode,
                "error": "spawn child crashed or produced no result",
            }
        else:
            payload = q.get()
            if payload.get("status") != "ok":
                spawn_result = payload
            else:
                spawn_result = {
                    "status": "ok",
                    "counts_executor": payload["result"]["diagnostics"]["phases"]["counts"]["executor"],
                    "counts_workers": payload["result"]["diagnostics"].get("counts_workers"),
                    "wall_s": payload["result"]["wall_s"],
                }
    except BaseException as exc:  # noqa: BLE001
        spawn_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    result = {
        "schema": "zm-gauge-counts-decoupling-ab-v1",
        "mission": "ZM-GAUGE-COUNTS-DECOUPLING-20261115",
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "corpus": {
            "source": str(LIGHTS),
            "frames": len(descs),
            "canvas": [int(canvas.width), int(canvas.height)],
            "normalization": getattr(config, "normalization", None),
            "weighting": getattr(config, "weighting", None),
        },
        "model": {
            "pairs_footprint_bytes": int(model["footprint_bytes"]),
            "counts_footprint_bytes": fc,
            "peak_phase": model["peak_phase"],
            "max_bbox_px": model.get("max_bbox_px"),
            "max_frame_px": model.get("max_frame_px"),
        },
        "spawn_probe": spawn_result,
        "arms": summary,
        "note": (
            "Model/planning numbers on this local subset are NOT the M33 "
            "admission. This measures the execution decoupling only; Windows "
            "spawn/psutil behaviour remains unproven until a physical Windows "
            "qualification."
        ),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
