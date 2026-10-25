#!/usr/bin/env python3
"""ZM-ZEGRID-R56-BOUNDED-PREFETCH — corrected A/B memory + overlap recorder.

REWORK-1 (F3) corrections:
* every arm runs in its OWN subprocess (fresh interpreter, fresh output dir);
* a destination that already exists/non-empty is REFUSED, never deleted;
* the RAM control can only clamp DOWN against the real observation
  (``effective = min(requested, real_available)``); it can never raise it;
* gates require NON-NULL, COMPLETE science artifacts and participation, the
  CANDIDATE arm to run with prefetch>=1 on the GPU with no CPU fallback, and the
  CANDIDATE's own observed overlap — control-only overlap never qualifies;
* ``cache_policy`` is reported honestly: a fresh process does NOT flush the OS
  file cache, so cold state is not claimed;
* CuPy pinned-pool bytes are reported as bytes only when the API gives bytes.

Unavailable observations are ``null`` with a reason. A zero is never fabricated.
Reservations (credit model / ledger) are reported separately from observations.

Usage::

    PYTHONPATH=src python tools/zegrid_r56_prefetch_campaign.py \
        --input <corpus> --out <fresh/scratch> --halo 64 --gpu \
        --storage-profile bounded [--max-available-bytes N]
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

SCHEMA = "zemosaic-r56-prefetch-campaign-v2"
ARM_SCHEMA = "zemosaic-r56-campaign-arm-v2"
SAMPLE_PERIOD_S = 0.2
OUTPUT_FILES = ("mosaic_grid.fits", "mosaic_grid_coverage.fits")


# ---------------------------------------------------------------------------
# Observation helpers
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


def _pool_bytes(pool, candidates):
    for name in candidates:
        fn = getattr(pool, name, None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
    return None


def _sample_vram():
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
            "pool_used_bytes": _pool_bytes(pool, ("used_bytes",)),
            "pool_total_bytes": _pool_bytes(pool, ("total_bytes",)),
            # CuPy's PinnedMemoryPool has n_free_blocks() (a COUNT) and, on
            # recent versions, used_bytes(). Never label a count as bytes.
            "pinned_pool_used_bytes": _pool_bytes(pinned, ("used_bytes",)),
            "pinned_pool_free_blocks": _pool_bytes(pinned, ("n_free_blocks",)),
            "source": "cupy.cuda.runtime.memGetInfo + cupy default pools",
            "scope": "device_wide_free_total; pool_used is THIS process only",
        }, None
    except Exception as exc:  # noqa: BLE001 - observation only
        return None, f"{type(exc).__name__}: {exc}"


class Sampler:
    CAVEAT = (
        "Peak-SAMPLED at the stated period, not an OS high-water mark. RSS is the "
        "sum over the sampled process tree (shared pages counted once per "
        "process), so it can exceed physical RAM. VRAM free/total is DEVICE-WIDE "
        "while pool_used is this process only."
    )

    def __init__(self, period_s: float = SAMPLE_PERIOD_S) -> None:
        self.period_s = float(period_s)
        self._stop = threading.Event()
        self._thread = None
        self.rss_peak_bytes = None
        self.rss_peak_per_process = None
        self.rss_samples = 0
        self.vram_free_min_bytes = None
        self.vram_pool_used_peak_bytes = None
        self.vram_samples = 0
        self.vram_error = None
        self.vram_last = None
        self._lock = threading.Lock()

    def _loop(self) -> None:
        psutil_module = _psutil()
        while not self._stop.is_set():
            with self._lock:
                if psutil_module is not None:
                    try:
                        total, per_proc = _sample_rss_tree(psutil_module)
                        self.rss_samples += 1
                        if self.rss_peak_bytes is None or total > self.rss_peak_bytes:
                            self.rss_peak_bytes = total
                            self.rss_peak_per_process = dict(per_proc)
                    except Exception as exc:  # noqa: BLE001
                        self.rss_peak_per_process = {
                            "error": f"{type(exc).__name__}: {exc}"
                        }
                vram, err = _sample_vram()
                if vram is not None:
                    self.vram_samples += 1
                    self.vram_last = vram
                    free = int(vram["device_free_bytes"])
                    if self.vram_free_min_bytes is None or free < self.vram_free_min_bytes:
                        self.vram_free_min_bytes = free
                    used = vram.get("pool_used_bytes")
                    if used is not None and (
                        self.vram_pool_used_peak_bytes is None
                        or int(used) > self.vram_pool_used_peak_bytes
                    ):
                        self.vram_pool_used_peak_bytes = int(used)
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
            "process_tree_rss_samples": self.rss_samples,
            "process_tree_rss_scope": "runner process + recursive children",
            "system_ram": system,
            "vram_samples": self.vram_samples,
            "vram_device_free_min_bytes": self.vram_free_min_bytes,
            "vram_pool_used_peak_bytes": self.vram_pool_used_peak_bytes,
            "vram_last": self.vram_last,
            "vram_unavailable_reason": self.vram_error,
            "observation_caveat": self.CAVEAT,
        }


# ---------------------------------------------------------------------------
# Provenance + one arm
# ---------------------------------------------------------------------------

def _git(*args) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _provenance() -> dict:
    cupy_version = None
    gpu = None
    try:
        import cupy  # noqa: PLC0415
        from cupy.cuda import runtime as cudart  # noqa: PLC0415

        cupy_version = getattr(cupy, "__version__", None)
        props = cudart.getDeviceProperties(0)
        name = props.get("name")
        gpu = {
            "name": name.decode(errors="replace") if isinstance(name, bytes) else name,
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
        "source_sha256": hashlib.sha256(
            (SRC / "zemosaic" / "zemosaic_zegrid_mode.py").read_bytes()
        ).hexdigest(),
        "package_version": (
            version_file.read_text().strip() if version_file.is_file() else None
        ),
        "cupy_version": cupy_version,
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
    max_available_bytes,
) -> dict:
    from zemosaic import zemosaic_zegrid_mode as zz  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=False)
    zz._R55E2_MEASURE_PREFETCH_CAP = prefetch_cap
    if max_available_bytes is not None:
        real = zz.available_memory_bytes()
        effective = min(int(max_available_bytes), int(real))
        zz.available_memory_bytes = lambda: int(effective)
    else:
        effective = None
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
            stack_norm_method="sky_mean",
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
        "schema": ARM_SCHEMA,
        "arm": arm,
        "pid": os.getpid(),
        "prefetch_cap_requested": prefetch_cap,
        "memory_observation_max_bytes": max_available_bytes,
        "memory_observation_effective_bytes": effective,
        "storage_profile_requested": storage_profile,
        "error": error,
        "wall_s": wall_s,
        "mapping": block.get("mapping"),
        "gpu_execution": block.get("gpu_execution"),
        "gpu_fallback": block.get("gpu_fallback"),
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
        "outputs": {name: _digest(out_dir / name) for name in OUTPUT_FILES},
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "complete_cells": manifest.get("complete_cells"),
        "incomplete_cells": manifest.get("incomplete_cells"),
        "frame_ids": manifest.get("frame_ids"),
        "n_frames": manifest.get("n_frames"),
        "layout": manifest.get("layout"),
        "normalization": manifest.get("normalization"),
        "observation": observation,
    }


# ---------------------------------------------------------------------------
# Gates (non-null, candidate-specific, comparable)
# ---------------------------------------------------------------------------

def _arm_complete(arm: dict) -> bool:
    if arm.get("error") is not None:
        return False
    outputs = arm.get("outputs") or {}
    if any(not outputs.get(name) for name in OUTPUT_FILES):
        return False
    if arm.get("complete_cells") is None:
        return False
    if arm.get("incomplete_cells") not in (0, None):
        return False
    frames = arm.get("frame_ids")
    if not isinstance(frames, list) or not frames:
        return False
    return True


def _arm_gpu_prefetch(arm: dict) -> bool:
    mapping = arm.get("mapping") or {}
    policy = arm.get("run_policy") or {}
    gpu_exec = arm.get("gpu_execution") or {}
    fallback = arm.get("gpu_fallback")
    return (
        str(mapping.get("backend")) == "gpu"
        and bool(gpu_exec.get("qualified"))
        and fallback in (None, {})
        and (policy.get("prefetch_units") or 0) >= 1
    )


def evaluate(control: dict, candidate: dict) -> dict:
    comparable = (
        control.get("layout") == candidate.get("layout")
        and control.get("n_frames") == candidate.get("n_frames")
        and control.get("normalization") == candidate.get("normalization")
        and _arm_complete(control)
        and _arm_complete(candidate)
    )
    outputs_identical = (
        comparable
        and all(
            control["outputs"].get(name)
            and control["outputs"].get(name) == candidate["outputs"].get(name)
            for name in OUTPUT_FILES
        )
    )
    frames_identical = (
        comparable
        and isinstance(control.get("frame_ids"), list)
        and control.get("frame_ids") == candidate.get("frame_ids")
    )
    overlap_candidate = bool(
        (candidate.get("overlap") or {}).get("overlap_proved")
        and float((candidate.get("overlap") or {}).get("intersection_s") or 0) > 0
    )
    gates = {
        "control_complete": _arm_complete(control),
        "candidate_complete": _arm_complete(candidate),
        "arms_comparable": bool(comparable),
        "scientific_outputs_identical": bool(outputs_identical),
        "frames_identical": bool(frames_identical),
        "candidate_gpu_prefetch_ge1": _arm_gpu_prefetch(candidate),
        "candidate_overlap_proved": overlap_candidate,
        "observed_memory_available": bool(
            control["observation"].get("process_tree_rss_peak_bytes") is not None
            and candidate["observation"].get("process_tree_rss_peak_bytes") is not None
        ),
        "vram_observed": bool(
            (control["observation"].get("vram_samples") or 0) > 0
            and (candidate["observation"].get("vram_samples") or 0) > 0
        ),
        "fresh_processes": control.get("pid") != candidate.get("pid"),
    }
    gates["all_passed"] = all(bool(v) for v in gates.values())
    return gates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _worker_main(args) -> int:
    arm = _run_arm(
        arm=args.arm_worker,
        input_dir=Path(args.input).resolve(),
        out_dir=Path(args.arm_out).resolve(),
        halo=int(args.halo),
        use_gpu=bool(args.gpu),
        prefetch_cap=(0 if args.arm_worker == "control" else None),
        storage_profile=args.storage_profile,
        sample_period_s=float(args.sample_period),
        max_available_bytes=args.max_available_bytes,
    )
    Path(args.arm_json).write_text(json.dumps(arm, indent=2, sort_keys=True))
    return 0 if arm["error"] is None else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--halo", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--storage-profile", default=None)
    parser.add_argument("--sample-period", type=float, default=SAMPLE_PERIOD_S)
    parser.add_argument(
        "--max-available-bytes",
        type=int,
        default=None,
        help=(
            "Upper bound for the host-RAM control. The runner uses "
            "min(requested, REAL available) so it can never inflate capacity."
        ),
    )
    parser.add_argument("--cache-policy-note", default=None)
    parser.add_argument("--arm-worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arm-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arm-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.arm_worker:
        return _worker_main(args)

    input_dir = Path(args.input).resolve()
    root = Path(args.out).resolve()
    if not input_dir.is_dir():
        print(f"input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if root.exists() and any(root.iterdir()):
        print(
            f"refusing occupied output directory (never deleted): {root}",
            file=sys.stderr,
        )
        return 3
    root.mkdir(parents=True, exist_ok=True)

    arms = {}
    for arm_name in ("control", "candidate"):
        arm_dir = root / f"arm_{arm_name}"
        arm_json = root / f"arm_{arm_name}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input", str(input_dir),
            "--out", str(root),
            "--halo", str(args.halo),
            "--sample-period", str(args.sample_period),
            "--arm-worker", arm_name,
            "--arm-out", str(arm_dir),
            "--arm-json", str(arm_json),
        ]
        if args.gpu:
            cmd.append("--gpu")
        if args.storage_profile:
            cmd += ["--storage-profile", str(args.storage_profile)]
        if args.max_available_bytes is not None:
            cmd += ["--max-available-bytes", str(args.max_available_bytes)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not arm_json.is_file():
            print(
                f"arm {arm_name} produced no JSON (rc={proc.returncode}): "
                f"{proc.stderr[-1500:]}",
                file=sys.stderr,
            )
            return 4
        arms[arm_name] = json.loads(arm_json.read_text())

    gates = evaluate(arms["control"], arms["candidate"])
    payload = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(input_dir),
        "out": str(root),
        "halo_px": int(args.halo),
        "use_gpu": bool(args.gpu),
        "config": {"stack_norm_method": "sky_mean", "scheduler_mode": "active"},
        "cache_policy": {
            "requested": args.cache_policy_note or "unspecified",
            "effective": "uncontrolled_os_file_cache",
            "note": (
                "A fresh process does NOT flush the OS file cache. No cold state "
                "is claimed and no global cache was flushed."
            ),
        },
        "provenance": _provenance(),
        "arms": arms,
        "gates": gates,
        "note": (
            "Reservation numbers (credit_model/ledger) and sampled observations "
            "are different quantities. A sampled peak is not an OS high-water mark."
        ),
    }
    report_path = root / "campaign.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(
        {
            "report": str(report_path),
            "gates": gates,
            "arms": [
                {
                    "arm": name,
                    "run_prefetch": (arm.get("run_policy") or {}).get("prefetch_units"),
                    "backend": (arm.get("mapping") or {}).get("backend"),
                    "overlap_proved": (arm.get("overlap") or {}).get("overlap_proved"),
                    "rss_peak_bytes": arm["observation"].get(
                        "process_tree_rss_peak_bytes"
                    ),
                    "error": arm.get("error"),
                }
                for name, arm in arms.items()
            ],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
