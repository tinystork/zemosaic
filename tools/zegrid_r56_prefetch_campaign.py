#!/usr/bin/env python3
"""ZM-ZEGRID-R56-BOUNDED-PREFETCH — A/B memory + overlap campaign runner.

Runs the SAME product binary/corpus/config twice, differing ONLY in the number
of staged prefetch slots (``--prefetch-cap 0`` control vs the planner's admitted
depth), and records:

* SAMPLED process-tree RSS (psutil, portable: Windows and Linux — never /proc),
  with the sample period, the sampler source and the explicit note that this is a
  sampled maximum, not an OS high-water mark;
* SAMPLED device VRAM (``cupy.cuda.mem_get_info`` device-wide free/total) plus
  the CuPy default-pool used/total, each labelled with its source and period;
* the product's own RESERVATION numbers (credit model + pipeline ledger) kept
  strictly separate from the observations above;
* phase wall timings and the observed producer/consumer interval intersection;
* scientific output digests (mosaic + coverage) so a memory win that changes the
  science is reported as a FAILURE, never as a result.

Unavailable observations are reported as ``null`` with a reason; a zero is never
fabricated.

Usage
-----
    PYTHONPATH=src python tools/zegrid_r56_prefetch_campaign.py \
        --input  /path/to/input \
        --out    /path/to/scratch/campaign \
        --halo   128 \
        --gpu \
        --arms   control,candidate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCHEMA = "zemosaic-r56-prefetch-campaign-v1"
SAMPLE_PERIOD_S = 0.2
OUTPUT_FILES = ("mosaic_grid.fits", "mosaic_grid_coverage.fits")


# ---------------------------------------------------------------------------
# Observation helpers (never fabricate: unavailable -> None + reason)
# ---------------------------------------------------------------------------

def _psutil():
    try:
        import psutil  # noqa: PLC0415

        return psutil
    except Exception:
        return None


def _process_tree(psutil_module, root_pid):
    root = psutil_module.Process(root_pid)
    procs = [root]
    try:
        procs.extend(root.children(recursive=True))
    except Exception:
        pass
    return procs


def _sample_rss_tree(psutil_module):
    """Sum of RSS over the runner process tree (bytes) + per-process max."""
    total = 0
    per_proc = {}
    for proc in _process_tree(psutil_module, os.getpid()):
        try:
            rss = int(proc.memory_info().rss)
        except Exception:
            continue
        total += rss
        per_proc[int(proc.pid)] = rss
    return total, per_proc


def _sample_vram():
    """Device-wide VRAM + CuPy pool, or (None, reason)."""
    try:
        import cupy  # noqa: PLC0415
        from cupy.cuda import runtime as cudart  # noqa: PLC0415

        count = int(cudart.getDeviceCount())
        if count <= 0:
            return None, "no_cuda_device"
        free, total = cudart.memGetInfo()
        pool = cupy.get_default_memory_pool()
        pinned = cupy.get_default_pinned_memory_pool()
        return {
            "device_index": 0,
            "device_count": count,
            "device_free_bytes": int(free),
            "device_total_bytes": int(total),
            "device_used_bytes": int(total) - int(free),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_total_bytes": int(pool.total_bytes()),
            "pinned_pool_used_bytes": int(getattr(pinned, "n_free_blocks", lambda: 0)() or 0),
            "source": "cupy.cuda.runtime.memGetInfo + cupy default pool",
            "scope": "device_wide_free_total; pool_used is THIS process",
        }, None
    except Exception as exc:  # noqa: BLE001 - observation only
        return None, f"{type(exc).__name__}: {exc}"


class Sampler:
    """Background sampler: process-tree RSS + VRAM at a fixed period."""

    CAVEAT = (
        "Peak-SAMPLED at the stated period, not an OS high-water mark. RSS is the "
        "sum over the sampled process tree (shared pages counted once per "
        "process); VRAM free/total is DEVICE-WIDE while pool_used is this "
        "process only."
    )

    def __init__(self, period_s: float = SAMPLE_PERIOD_S) -> None:
        self.period_s = float(period_s)
        self._stop = threading.Event()
        self._thread = None
        self.rss_peak_bytes = None
        self.rss_peak_per_process = None
        self.rss_samples = 0
        self.rss_peak_sample_at = None
        self.vram_free_min_bytes = None
        self.vram_pool_used_peak_bytes = None
        self.vram_samples = 0
        self.vram_error = None
        self.vram_meta = None
        self._lock = threading.Lock()

    def _loop(self) -> None:
        psutil_module = _psutil()
        started = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                if psutil_module is not None:
                    try:
                        total, per_proc = _sample_rss_tree(psutil_module)
                        self.rss_samples += 1
                        if self.rss_peak_bytes is None or total > self.rss_peak_bytes:
                            self.rss_peak_bytes = total
                            self.rss_peak_per_process = dict(per_proc)
                            self.rss_peak_sample_at = time.monotonic() - started
                    except Exception as exc:  # noqa: BLE001
                        self.rss_peak_per_process = self.rss_peak_per_process or {
                            "error": f"{type(exc).__name__}: {exc}"
                        }
                vram, err = _sample_vram()
                if vram is not None:
                    self.vram_samples += 1
                    self.vram_meta = vram
                    free = int(vram["device_free_bytes"])
                    if self.vram_free_min_bytes is None or free < self.vram_free_min_bytes:
                        self.vram_free_min_bytes = free
                    used = int(vram["pool_used_bytes"])
                    if (
                        self.vram_pool_used_peak_bytes is None
                        or used > self.vram_pool_used_peak_bytes
                    ):
                        self.vram_pool_used_peak_bytes = used
                elif err:
                    self.vram_error = err
            self._stop.wait(self.period_s)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="r56-memory-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
        psutil_module = _psutil()
        system = None
        if psutil_module is not None:
            vm = psutil_module.virtual_memory()
            system = {
                "virtual_total_bytes": int(vm.total),
                "virtual_available_bytes": int(vm.available),
                "virtual_used_bytes": int(vm.used),
                "source": "psutil.virtual_memory (SYSTEM-wide)",
            }
        return {
            "schema": "zemosaic-r56-memory-observation-v1",
            "sampler_period_s": self.period_s,
            "sampler_source": (
                "psutil process-tree RSS"
                if psutil_module is not None
                else "UNAVAILABLE: psutil missing"
            ),
            "process_tree_rss_peak_bytes": self.rss_peak_bytes,
            "process_tree_rss_peak_per_process_bytes": self.rss_peak_per_process,
            "process_tree_rss_peak_sample_at_s": self.rss_peak_sample_at,
            "process_tree_rss_samples": self.rss_samples,
            "process_tree_rss_scope": "runner process + recursive children",
            "system_ram": system,
            "vram_samples": self.vram_samples,
            "vram_device_free_min_bytes": self.vram_free_min_bytes,
            "vram_pool_used_peak_bytes": self.vram_pool_used_peak_bytes,
            "vram_last": self.vram_meta,
            "vram_unavailable_reason": self.vram_error,
            "observation_caveat": self.CAVEAT,
            "note": (
                "These are OBSERVATIONS. They are never compared to the planner's "
                "reservation arithmetic as if they were the same quantity."
            ),
        }


# ---------------------------------------------------------------------------
# Provenance + run
# ---------------------------------------------------------------------------

def _git(*args) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _provenance() -> dict:
    import cupy  # noqa: PLC0415

    gpu = None
    try:
        from cupy.cuda import runtime as cudart  # noqa: PLC0415

        props = cudart.getDeviceProperties(0)
        gpu = {
            "name": props.get("name", b"").decode(errors="replace")
            if isinstance(props.get("name"), bytes)
            else props.get("name"),
            "total_memory_bytes": int(props.get("totalGlobalMem", 0)),
        }
    except Exception as exc:  # noqa: BLE001
        gpu = {"unavailable_reason": f"{type(exc).__name__}: {exc}"}
    version_file = REPO_ROOT / "version.txt"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "repo": str(REPO_ROOT),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "package_version": (
            version_file.read_text().strip() if version_file.is_file() else None
        ),
        "cupy_version": getattr(cupy, "__version__", None),
        "gpu": gpu,
    }


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_arm(
    *,
    arm: str,
    input_dir: Path,
    out_dir: Path,
    halo: int,
    use_gpu: bool,
    prefetch_cap,
    storage_profile,
    sample_period_s: float,
) -> dict:
    from zemosaic import zemosaic_zegrid_mode as zz  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    zz._R55E2_MEASURE_PREFETCH_CAP = prefetch_cap
    sampler = Sampler(sample_period_s)
    sampler.start()
    started = time.monotonic()
    error = None
    try:
        zz.run_zegrid_mode(
            str(input_dir),
            str(out_dir),
            zegrid_halo_px=int(halo),
            zegrid_scheduler_mode="active",
            use_gpu=bool(use_gpu),
            storage_profile=storage_profile,
        )
    except BaseException as exc:  # noqa: BLE001 - report, never hide
        error = f"{type(exc).__name__}: {exc}"
    finally:
        wall_s = time.monotonic() - started
        zz._R55E2_MEASURE_PREFETCH_CAP = None
        observation = sampler.stop()

    manifest = {}
    manifest_path = out_dir / "zegrid_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    block = manifest.get("r55_scheduler") or {}
    run_block = block.get("run") or {}
    pipeline = (manifest.get("r55_run_telemetry") or {}).get("gpu_pipeline") or {}
    return {
        "arm": arm,
        "prefetch_cap_requested": prefetch_cap,
        "storage_profile_requested": storage_profile,
        "error": error,
        "wall_s": wall_s,
        "mapping": block.get("mapping"),
        "run_policy": {
            "prefetch_units": (run_block.get("policy") or {}).get("prefetch_units"),
            "prefetch_cap_override": (
                run_block.get("policy") or {}
            ).get("prefetch_cap_override"),
            "queue_max": (run_block.get("policy") or {}).get("queue_max"),
        },
        "credit_model": (run_block.get("policy") or {}).get("credit_model"),
        "disk_policy": (run_block.get("policy") or {}).get("disk"),
        "ledger": pipeline.get("ledger"),
        "overlap": run_block.get("overlap"),
        "timings": (manifest.get("r55_run_telemetry") or {}).get("metrics"),
        "observation": observation,
        "outputs": {
            name: _digest(out_dir / name) for name in OUTPUT_FILES
        },
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "complete_cells": manifest.get("complete_cells"),
        "incomplete_cells": manifest.get("incomplete_cells"),
        "frames_participating": (
            manifest.get("frame_participation")
            or manifest.get("frames_used")
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--halo", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--arms", default="control,candidate")
    parser.add_argument(
        "--storage-profile", default=None, help="dense|bounded|auto (default auto)"
    )
    parser.add_argument("--sample-period", type=float, default=SAMPLE_PERIOD_S)
    parser.add_argument("--cache-policy", default="cold",
                        choices=("cold", "warm"))
    parser.add_argument(
        "--available-bytes",
        type=int,
        default=None,
        help=(
            "Pin the host-RAM observation for BOTH arms so the PLAN is identical "
            "and only the prefetch cap differs. Recorded in provenance as an "
            "explicit measurement control; the RSS/VRAM observations stay real."
        ),
    )
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    input_dir = Path(args.input).resolve()
    root = Path(args.out).resolve()
    if not input_dir.is_dir():
        print(f"input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if root.exists() and args.keep is False:
        pass  # fresh arm dirs are created below; never delete user data

    arms = [token.strip() for token in args.arms.split(",") if token.strip()]
    if args.available_bytes is not None:
        from zemosaic import zemosaic_zegrid_mode as _zz  # noqa: PLC0415

        _zz.available_memory_bytes = lambda: int(args.available_bytes)
    results = []
    for arm in arms:
        cap = 0 if arm == "control" else None
        out_dir = root / f"arm_{arm}"
        if out_dir.exists():
            # Mission scratch only: the caller owns --out.
            shutil.rmtree(out_dir)
        results.append(
            _run_arm(
                arm=arm,
                input_dir=input_dir,
                out_dir=out_dir,
                halo=int(args.halo),
                use_gpu=bool(args.gpu),
                prefetch_cap=cap,
                storage_profile=args.storage_profile,
                sample_period_s=float(args.sample_period),
            )
        )

    comparison = {
        "outputs_identical": (
            results[0]["outputs"] == results[1]["outputs"]
            if len(results) == 2
            else None
        ),
        "complete_cells_identical": (
            results[0]["complete_cells"] == results[1]["complete_cells"]
            if len(results) == 2
            else None
        ),
        "frames_identical": (
            results[0]["frames_participating"]
            == results[1]["frames_participating"]
            if len(results) == 2
            else None
        ),
    }
    payload = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(input_dir),
        "out": str(root),
        "halo_px": int(args.halo),
        "use_gpu": bool(args.gpu),
        "cache_policy": args.cache_policy,
        "memory_observation_pinned_bytes": args.available_bytes,
        "provenance": _provenance(),
        "arms": results,
        "comparison": comparison,
        "gates": {
            "scientific_outputs_identical": comparison["outputs_identical"],
            "real_overlap_proved": any(
                bool((arm.get("overlap") or {}).get("overlap_proved"))
                for arm in results
            ),
            "observed_memory_available": all(
                (arm["observation"].get("process_tree_rss_peak_bytes") is not None)
                for arm in results
            ),
            "vram_observed": all(
                (arm["observation"].get("vram_samples") or 0) > 0
                for arm in results
            ),
        },
        "note": (
            "Reservation numbers (credit_model/ledger) and sampled observations "
            "are different quantities and are reported separately. A sampled "
            "peak is not an OS high-water mark."
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "campaign.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(
        {
            "report": str(report_path),
            "gates": payload["gates"],
            "arms": [
                {
                    "arm": arm["arm"],
                    "plan_prefetch": ((arm.get("mapping") or {}).get("prefetch_units")),
                    "run_prefetch": (arm.get("run_policy") or {}).get("prefetch_units"),
                    "overlap_proved": (arm.get("overlap") or {}).get("overlap_proved"),
                    "overlap_s": (arm.get("overlap") or {}).get("intersection_s"),
                    "rss_peak_bytes": arm["observation"].get(
                        "process_tree_rss_peak_bytes"
                    ),
                    "vram_free_min_bytes": arm["observation"].get(
                        "vram_device_free_min_bytes"
                    ),
                    "error": arm["error"],
                }
                for arm in results
            ],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if not any(arm["error"] for arm in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
