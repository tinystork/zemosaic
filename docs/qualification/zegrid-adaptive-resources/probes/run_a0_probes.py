#!/usr/bin/env python3
"""A0 probes — ZeGrid adaptive resources (diagnostic only, no product code).

Usage (from the mission worktree, PYTHONPATH=src):

    python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py \
        --phase env|model|witness-ext|witness-inmem|caldwell-inventory [--out DIR]

This script never modifies the corpus, never writes into a run output
directory, and never imports the GUI. It only reads product modules and
(optionally) hashes source files read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

MIB = 2 ** 20
GIB = 2 ** 30


def _hwm_bytes() -> int:
    """Peak resident set of THIS process (Linux /proc/self/status VmHWM)."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def phase_env() -> dict:
    import platform

    import numpy as np

    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
        "rss_after_import_bytes": _rss_bytes(),
        "hwm_after_import_bytes": _hwm_bytes(),
    }
    try:
        import cupy  # noqa: F401

        info["cupy"] = cupy.__version__
        info["cuda_runtime"] = int(cupy.cuda.runtime.runtimeGetVersion())
        info["cuda_device_count"] = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001 - probe must report, not fail
        info["cupy"] = None
        info["cupy_error"] = f"{type(exc).__name__}: {exc}"
    return info


def phase_model() -> dict:
    """Geometry/model pricing with the REAL product functions (no pixels)."""
    from zemosaic.core.zegrid import external_stream as xstream
    from zemosaic.core.zegrid.gpu import (
        GPU_TAPER_HALO_PX,
        GPU_VRAM_SAFETY_FRACTION,
        gpu_tile_workspace_bytes,
    )
    from zemosaic.core.zegrid.streaming import (
        STREAMING_BASELINE_KIB,
        TILE_WORKSPACE_BYTES_PER_CELL,
        estimate_streaming_peak_kib,
    )

    out = {
        "constants": {
            "streaming_baseline_kib": STREAMING_BASELINE_KIB,
            "tile_workspace_bytes_per_cell": TILE_WORKSPACE_BYTES_PER_CELL,
            "gpu_workspace_bytes_per_cell": 96,
            "gpu_taper_halo_px": GPU_TAPER_HALO_PX,
            "gpu_vram_safety_fraction": GPU_VRAM_SAFETY_FRACTION,
            "external_bytes_per_position_frame": (
                xstream.REDUCTION_BYTES_PER_POSITION_FRAME
            ),
            "external_spool_bytes_per_pos_frame": "4*channels + 4",
        },
        "scenarios": {},
    }

    # M33: 22 414 entries (older inventory) and 21 818 retained (last run).
    for label, n in (("m33_old_inventory_22414", 22414), ("m33_last_run_21818", 21818)):
        tile = 20
        halo = 9
        ext_hw = 2 * halo + tile  # 38
        patch = (200, 200)  # tile 20 is authoritative; patch only affects O(patch) terms
        chan = 3

        ext = xstream.estimate_external_working_set(
            n_frames=n,
            height=patch[0],
            width=patch[1],
            channels=chan,
            tile_size=tile,
            halo_px=halo,
            column_batch=1024,
        )
        bounded_kib = estimate_streaming_peak_kib(
            n, patch, tile, chan, storage_profile="bounded"
        )
        dense_kib = estimate_streaming_peak_kib(
            n, patch, tile, chan, storage_profile="dense"
        )
        gpu_bytes = gpu_tile_workspace_bytes(n, ext_hw, ext_hw, chan)
        samples = n * ext_hw * ext_hw * chan

        out["scenarios"][label] = {
            "n_frames": n,
            "tile_interior_px": tile,
            "extended_tile_px": ext_hw,
            "extended_area": ext_hw * ext_hw,
            "samples_n_x_ext_area_x_channels": samples,
            "cpu_streaming_bounded_peak_bytes": int(bounded_kib) * 1024,
            "cpu_streaming_bounded_bytes_per_sample": (
                int(bounded_kib) * 1024 / samples
            ),
            "cpu_streaming_dense_peak_bytes": int(dense_kib) * 1024,
            "gpu_tile_workspace_bytes": gpu_bytes,
            "gpu_bytes_per_sample": gpu_bytes / samples,
            "external_spool_peak_worst_tile_bytes": ext["disk"][
                "spool_peak_bytes_worst_tile"
            ],
            "external_reduction_workspace_bytes": ext["ram"][
                "reduction_workspace_bytes"
            ],
            "external_geometry": ext["geometry"],
            "external_rejection_block_if_dense_bytes": ext["ram"][
                "rejection_block_bytes_if_dense"
            ],
            "external_unknown_terms": ext["disk"]["unknown_terms"],
        }

    # Audit-reported bounds (2026-11-07 read-only audit of the two Windows runs)
    reported = {
        "run1_1013": {
            "budget_mib": 13973,
            "layout": [7, 12],
            "tile_px": 24,
            "accepted_bound_mib": 13883,
            "vram_budget_bytes": 5_220_000_000,
        },
        "run2_1238": {
            "budget_mib": 9931,
            "layout": [12, 20],
            "tile_px": 20,
            "accepted_bound_mib": 9289,
        },
    }
    for key, val in reported.items():
        samples = 21818 * (val["tile_px"] + 2 * 9) ** 2 * 3
        val["implied_bytes_per_sample"] = (
            val["accepted_bound_mib"] * MIB / samples
        )
        out["scenarios"].setdefault("audit_reported", {})[key] = val

    out["gauge_threshold"] = {
        "formula": "gauge_workers * (PROCESS_BASELINE + canvas_px*(3*8+3*4))",
        "process_baseline_bytes": 340 * MIB,
        "bytes_per_canvas_px_per_worker": 36,
    }
    return out


def _corpus(n=48, h=64, w=64, seed=7):
    import numpy as np

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.full((h, w), 100.0)
    for _ in range(4):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        base += 25.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2))
    frames = []
    gains = np.linspace(0.95, 1.05, n)
    for i in range(n):
        frame = np.stack(
            [base * gains[i] * [0.97, 1.0, 1.04][c] + rng.normal(0, 1.0, (h, w))
             for c in range(3)],
            axis=-1,
        )
        frames.append(np.ascontiguousarray(frame.astype(np.float32)))
    # adversarial but deterministic: NaNs, saturated outliers, partial masks
    frames[1][4:9, 4:9, :] = np.nan
    frames[2][10:14, 40:44, :] = 1.0e6
    frames[3][:, :, 0] *= 0.5
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[5][:, w // 3:] = False
    for i in range(6, n):
        masks[i][:, :3] = False
    return frames, masks


def _request(frames, masks):
    from zemosaic.core.canonical_engine import CanonicalStackRequest

    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="winsorized_sigma_clip",
        combine="mean",
        taper="footprint",
        taper_px=8.0,
    )


def _summary(name, result, seconds, base_rss, hwm):
    import numpy as np

    return {
        "phase": name,
        "seconds": round(seconds, 3),
        "rss_before_bytes": int(base_rss),
        "vmhwm_bytes": int(hwm),
        "peak_delta_over_pre_arrays_bytes": int(hwm - base_rss),
        "n_frames": int(result.n_frames),
        "shape": list(result.science.shape),
        "science_dtype": str(result.science.dtype),
        "science_sha256": hashlib.sha256(
            np.ascontiguousarray(result.science).tobytes()
        ).hexdigest(),
        "counters": {
            "iterations_used": int(result.iterations_used),
            "initial_sample_count": int(result.initial_sample_count),
            "rejected_sample_count": int(result.rejected_sample_count),
            "rejected_fraction": float(result.rejected_fraction),
            "low_n_cell_count": int(result.low_n_cell_count),
            "degenerate_cell_count": int(result.degenerate_cell_count),
        },
        "surviving_count_sha256": hashlib.sha256(
            np.ascontiguousarray(result.surviving_sample_count).tobytes()
        ).hexdigest(),
        "support_w1_sum": float(np.sum(result.support_w1)),
        "support_w2_sum": float(np.sum(result.support_w2)),
    }


def phase_witness_inmem() -> dict:
    from zemosaic.core.canonical_engine import run_canonical_stack

    frames, masks = _corpus()
    base = _rss_bytes()
    t0 = time.perf_counter()
    res = run_canonical_stack(_request(frames, masks))
    dt = time.perf_counter() - t0
    out = _summary("witness-inmem", res, dt, base, _hwm_bytes())
    out["corpus_sha256"] = hashlib.sha256(
        b"".join(f.tobytes() for f in frames)
    ).hexdigest()
    return out


def phase_witness_ext() -> dict:
    from zemosaic.core.canonical_streaming import (
        InMemoryCanonicalProvider,
        run_canonical_stack_streaming,
    )
    from zemosaic.core.zegrid import external_stream as xstream

    frames, masks = _corpus()
    req = _request(frames, masks)
    base = _rss_bytes()
    with tempfile.TemporaryDirectory(prefix="a0-spool-") as tmp:
        t0 = time.perf_counter()
        res = xstream.run_cell_streaming_external(
            InMemoryCanonicalProvider(frames, masks),
            req,
            tile_size=16,
            column_batch=1024,
            spool_root=tmp,
            cell_id="a0cell",
            attempt_id="a0attempt",
        )
        dt = time.perf_counter() - t0
        residue = []
        for dirpath, dirnames, filenames in os.walk(tmp):
            rel = os.path.relpath(dirpath, tmp)
            for d in dirnames:
                residue.append({"rel": d if rel == "." else os.path.join(rel, d), "type": "dir"})
            for f in filenames:
                residue.append({"rel": f if rel == "." else os.path.join(rel, f),
                                "type": "file",
                                "bytes": os.path.getsize(os.path.join(dirpath, f))})
    out = _summary("witness-ext", res, dt, base, _hwm_bytes())
    files = [r for r in residue if r["type"] == "file"]
    dirs = [r for r in residue if r["type"] == "dir"]
    out["spool_residue_entries"] = len(residue)
    out["spool_residue_files"] = len(files)
    out["spool_residue_dirs"] = [r["rel"] for r in dirs]
    out["spool_residue_is_empty_dir_only"] = bool(not files)
    out["corpus_sha256"] = hashlib.sha256(
        b"".join(f.tobytes() for f in frames)
    ).hexdigest()
    return out


def phase_caldwell_inventory(root: str, out_dir: str) -> dict:
    root_p = Path(root)
    files = sorted(
        p for p in root_p.iterdir()
        if p.is_file() and p.suffix.lower() in {".fit", ".fits"}
    )
    digest = hashlib.sha256()
    total = 0
    rows = []
    t0 = time.perf_counter()
    for p in files:
        h = hashlib.sha256()
        size = 0
        with open(p, "rb") as fh:
            while True:
                chunk = fh.read(4 * MIB)
                if not chunk:
                    break
                size += len(chunk)
                h.update(chunk)
        total += size
        line = {"name": p.name, "bytes": size, "sha256": h.hexdigest()}
        rows.append(line)
        digest.update((p.name + "\0" + line["sha256"] + "\n").encode())
    dt = time.perf_counter() - t0
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "caldwell_ircut_inventory.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    tokens = {}
    for r in rows:
        for tok in ("IRCUT", "_LP_", "Ha", "OIII", "SII", "LIGHT", "Light"):
            if tok in r["name"]:
                tokens[tok] = tokens.get(tok, 0) + 1
    return {
        "phase": "caldwell-inventory",
        "root": str(root_p),
        "n_files": len(rows),
        "total_bytes": total,
        "seconds": round(dt, 3),
        "ordered_inventory_sha256": digest.hexdigest(),
        "first": rows[0]["name"] if rows else None,
        "last": rows[-1]["name"] if rows else None,
        "filename_tokens": tokens,
        "inventory_path": str(out_path / "caldwell_ircut_inventory.jsonl"),
    }



def phase_admission() -> dict:
    """Reproduce the last M33 joint-admission arithmetic from LOGGED values.

    Inputs are the values printed by the real run (2026-11-07 12:38-12:53):
    canvas_px, 240 cells, chosen layout 12x20, CPU streaming tile 20,
    B_plan = 9931.5 MiB, stream@20 = 9289.1 MiB. The per-phase footprints use
    the REAL product formulas (parallel.gauge_footprint_bytes /
    cache_footprint_bytes). No pixel work, no GUI import.

    L3 (rework-2): these inputs are HARDCODED from the logged run on purpose.
    They are anchored by the Windows SHA-256s in the A0 README. A1 MUST read
    ``run_config_snapshot.json`` and ``resource_telemetry.csv`` directly (or a
    versioned replay file) instead of copying literals; otherwise a new run
    could be audited with stale numbers.
    """
    from zemosaic.core.zegrid.parallel import (
        cache_footprint_bytes,
        gauge_footprint_bytes,
    )
    from zemosaic.core.zegrid.ram_contract import (
        PLAN_FACTOR_DEN,
        PLAN_FACTOR_NUM,
        PROCESS_BASELINE_BYTES,
        cache_per_cell_bytes,
    )
    from zemosaic.core.zegrid.streaming import estimate_streaming_peak_kib

    # Hardcoded from the logged run (see L3 note above): A1 must parse the
    # snapshot/telemetry instead of reusing these literals.
    canvas_px = 75698820
    cells = 240
    patch_px = canvas_px // cells
    budget_mib = 9931.5
    stack_mib = 9289.1
    budget = int(budget_mib * MIB)
    stack = int(stack_mib * MIB)

    best = None
    for h in range(600, 660):
        for w in (500, 501, 502):
            kib = estimate_streaming_peak_kib(
                21818, (h, w), 20, 3, storage_profile="bounded"
            )
            err = abs(kib / 1024 - stack_mib)
            if best is None or err < best["err_mib"]:
                best = {
                    "patch_hw": [h, w],
                    "patch_px": h * w,
                    "estimate_mib": kib / 1024,
                    "err_mib": err,
                }

    gauge1 = gauge_footprint_bytes(canvas_px)
    cache1 = cache_footprint_bytes(patch_px)
    cache_cell = cache_per_cell_bytes(PROCESS_BASELINE_BYTES, 1, cache1)
    shared = PROCESS_BASELINE_BYTES
    raw_min = shared + gauge1 + cache_cell + stack
    margin_num, margin_den = 15, 100
    agg_min = (raw_min * (margin_den + margin_num)) // margin_den

    return {
        "logged": {
            "canvas_px": canvas_px,
            "cells": cells,
            "layout": [12, 20],
            "tile_px": 20,
            "n_frames": 21818,
            "budget_mib": budget_mib,
            "stream_at_20_mib": stack_mib,
            "ram_available_at_setup_mib": 12414.3125,
            "plan_factor": [PLAN_FACTOR_NUM, PLAN_FACTOR_DEN],
            "margin": [margin_num, margin_den],
        },
        "estimator_fit": best,
        "footprints": {
            "process_baseline_bytes": PROCESS_BASELINE_BYTES,
            "gauge_1_worker_bytes": gauge1,
            "cache_1_worker_bytes": cache1,
            "cache_per_cell_1_worker_bytes": cache_cell,
            "patch_px_nominal": patch_px,
        },
        "admission": {
            "max_raw_bytes_for_budget": budget * margin_den // (margin_den + margin_num),
            "max_raw_mib": (budget * margin_den // (margin_den + margin_num)) / MIB,
            "stack_mib": stack_mib,
            "minimum_raw_mib": raw_min / MIB,
            "minimum_aggregate_mib": agg_min / MIB,
            "minimum_aggregate_exceeds_budget": bool(agg_min > budget),
            "headroom_before_margin_mib": (budget - stack) / MIB,
            "note": (
                "stack is charged once and the aggregate adds the 15% margin "
                "plus gauge/cache phases; the PLAN-accepted stack alone leaves "
                "less than the margin requires"
            ),
        },
    }


def phase_gpu_smoke() -> dict:
    """Minimal CuPy smoke: device count, memGetInfo, one small allocation.

    This is NOT a parity test. It only proves the Linux CuPy/CUDA runtime is
    usable for A3. Run it with the project venv interpreter.
    """
    out = {"phase": "gpu-smoke", "status": "NOT_RUN"}
    try:
        import cupy as cp
        import numpy as np

        out["cupy"] = cp.__version__
        out["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
        ndev = int(cp.cuda.runtime.getDeviceCount())
        out["cuda_device_count"] = ndev
        if ndev < 1:
            out["status"] = "NO_DEVICE"
            return out
        props = cp.cuda.runtime.getDeviceProperties(0)
        out["device_name"] = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        out["compute_capability"] = [int(props["major"]), int(props["minor"])]
        free_before, total = cp.cuda.runtime.memGetInfo()
        out["mem_free_before_bytes"] = int(free_before)
        out["mem_total_bytes"] = int(total)
        nbytes = 64 * 1024 * 1024
        a = cp.empty(nbytes // 4, dtype=cp.float32)
        a.fill(1.0)
        total_sum = float(cp.asnumpy(cp.sum(a, dtype=cp.float64)))
        cp.cuda.Stream.null.synchronize()
        out["allocation_bytes"] = nbytes
        out["sum"] = total_sum
        out["sum_expected"] = float(nbytes // 4)
        del a
        cp.get_default_memory_pool().free_all_blocks()
        free_after, _ = cp.cuda.runtime.memGetInfo()
        out["mem_free_after_bytes"] = int(free_after)
        out["status"] = "PASS_SMOKE" if abs(total_sum - nbytes // 4) < 1e-6 else "FAILED_SUM"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "FAILED"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        required=True,
        choices=["env", "model", "admission", "witness-ext", "witness-inmem", "caldwell-inventory", "gpu-smoke"],
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--caldwell-root",
        default="/home/tristan/caldwell11/lights organized/EQ/IRCUT",
    )
    args = ap.parse_args()
    t_start = time.perf_counter()
    if args.phase == "env":
        result = phase_env()
    elif args.phase == "model":
        result = phase_model()
    elif args.phase == "admission":
        result = phase_admission()
    elif args.phase == "witness-ext":
        result = phase_witness_ext()
    elif args.phase == "witness-inmem":
        result = phase_witness_inmem()
    elif args.phase == "gpu-smoke":
        result = phase_gpu_smoke()
    else:
        result = phase_caldwell_inventory(args.caldwell_root, args.out or ".")
    result["wall_seconds_total"] = round(time.perf_counter() - t_start, 3)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
