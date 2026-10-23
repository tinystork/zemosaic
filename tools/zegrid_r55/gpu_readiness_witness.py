#!/usr/bin/env python3
"""R55-C bounded real GPU readiness / allocator / parity witness (MX150).

Observe-only. Explicit CuPy initialisation is authorised for THIS witness only;
the product modules remain lazy and no GPU scheduling is activated.

Bounds (hard-coded, never exceeded):
* one device, device 0;
* a tiny deterministic kernel/round-trip (4096 float32 elements);
* a bounded synthetic canonical-style masked-mean CPU/GPU parity check;
* sequential live allocations of 64 MiB then 256 MiB (never more than 256 MiB
  live at once), each synchronised, released, and pool-freed;
* never 50-75% of VRAM, no FITS corpus, no full cell/mosaic, no benchmark.

Any GPU error aborts with status BLOCKED/FAILED and is recorded verbatim; there
is NO CPU fallback inside the witness.

Usage:
    python tools/zegrid_r55/gpu_readiness_witness.py --out FILE.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

MIB = 1024 * 1024
MAX_LIVE_ALLOC_BYTES = 256 * MIB
KERNEL_ELEMENTS = 4096
PARITY_SHAPE = (4, 32, 32, 3)
PARITY_SEED = 20261023


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mem(cp) -> dict:
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    out = {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
    }
    try:
        pinned = cp.get_default_pinned_memory_pool()
        used_fn = getattr(pinned, "used_bytes", None)
        total_fn = getattr(pinned, "total_bytes", None)
        if callable(used_fn):
            out["pinned_pool_used_bytes"] = int(used_fn())
        if callable(total_fn):
            out["pinned_pool_total_bytes"] = int(total_fn())
    except Exception:
        pass
    return out


def _cleanup(cp) -> None:
    gc.collect()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    cp.cuda.Stream.null.synchronize()
    gc.collect()


def run_witness() -> dict:
    import numpy as np

    from zemosaic.core.zegrid import gpu_readiness as gr

    report: dict = {
        "schema": "zemosaic-r55c-gpu-witness-v1",
        "phase": "implementation",
        "mission": "ZM-ZEGRID-R55-C-GPU-READINESS-VRAM-POLICY",
        "observe_only": True,
        "execution_authority": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bounds": {
            "device_index": 0,
            "kernel_elements": KERNEL_ELEMENTS,
            "allocations_bytes": [64 * MIB, 256 * MIB],
            "max_live_alloc_bytes": MAX_LIVE_ALLOC_BYTES,
            "corpus": None,
            "benchmark": False,
        },
        "checks": {},
        "steps": [],
    }

    readiness = gr.probe_gpu_readiness(device_index=0)
    report["readiness"] = readiness.to_dict()
    report["policy"] = gr.readiness_manifest_dict(
        readiness, free_bytes=readiness.vram_free_bytes
    )
    if not readiness.available:
        report["status"] = "BLOCKED"
        report["blocked_reason"] = (
            f"GPU not usable: {readiness.status}/{readiness.reason_code}"
        )
        return report

    import cupy as cp

    def step(name: str, fn):
        entry = {"step": name, "ok": False}
        try:
            value = fn()
            entry["ok"] = True
            entry["result"] = value
        except Exception as exc:  # record verbatim, never hide
            entry["error"] = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
            report["steps"].append(entry)
            raise
        report["steps"].append(entry)
        return value

    report["before"] = _mem(cp)

    # 1. Tiny deterministic kernel / host round-trip.
    def kernel():
        host = np.arange(KERNEL_ELEMENTS, dtype=np.float32)
        device = cp.asarray(host)
        out = cp.asnumpy(device * cp.float32(2.0) + cp.float32(1.0))
        expected = host * 2.0 + 1.0
        exact = bool(np.array_equal(out, expected))
        total = float(cp.asnumpy(cp.sum(device, dtype=cp.float64)))
        return {
            "exact_round_trip": exact,
            "sum_float64": total,
            "sum_expected": float(np.sum(host, dtype=np.float64)),
        }

    step("tiny_kernel_round_trip", kernel)

    # 2. Bounded synthetic canonical-style masked-mean parity (no product data).
    def parity():
        rng = np.random.default_rng(PARITY_SEED)
        images = rng.standard_normal(PARITY_SHAPE).astype(np.float32)
        weights = rng.random(PARITY_SHAPE).astype(np.float64)
        mask = rng.random(PARITY_SHAPE[:3]) > 0.2
        w = weights * mask[..., None]
        num_cpu = (images.astype(np.float64) * w).sum(axis=0)
        den_cpu = w.sum(axis=0)
        mean_cpu = np.divide(num_cpu, den_cpu, out=np.zeros_like(num_cpu), where=den_cpu > 0)
        g_images = cp.asarray(images)
        g_w = cp.asarray(w)
        g_num = (g_images.astype(cp.float64) * g_w).sum(axis=0)
        g_den = g_w.sum(axis=0)
        mean_gpu = cp.asnumpy(
            cp.where(g_den > 0, g_num / cp.maximum(g_den, 1e-30), 0.0)
        )
        diff = np.abs(mean_gpu - mean_cpu)
        return {
            "shape": list(PARITY_SHAPE),
            "max_abs_diff": float(diff.max()),
            "exact_equal": bool(np.array_equal(mean_gpu, mean_cpu)),
            "allclose_1e12": bool(np.allclose(mean_gpu, mean_cpu, rtol=0, atol=1e-12)),
            "finite": bool(np.isfinite(mean_gpu).all()),
        }

    step("synthetic_masked_mean_parity", parity)

    # 3. Bounded sequential allocator retention + cleanup.
    allocations = []
    for size in (64 * MIB, 256 * MIB):
        if size > MAX_LIVE_ALLOC_BYTES:
            raise AssertionError("witness allocation exceeds the authorised bound")

        def alloc(size=size):
            before = _mem(cp)
            buf = cp.empty(size, dtype=cp.uint8)
            buf[:] = 7
            cp.cuda.Stream.null.synchronize()
            during = _mem(cp)
            del buf
            _cleanup(cp)
            after = _mem(cp)
            return {
                "requested_bytes": int(size),
                "before": before,
                "during": during,
                "after": after,
                "pool_released": after["pool_used_bytes"] == 0,
                "free_recovered": after["free_bytes"] >= before["free_bytes"] - 16 * MIB,
            }

        entry = step(f"alloc_{size // MIB}MiB", alloc)
        allocations.append(entry)

    _cleanup(cp)
    report["allocations"] = allocations
    report["after"] = _mem(cp)

    checks = {
        "readiness_usable": readiness.available,
        "cupy_version": readiness.cupy_version,
        "device_name": readiness.device_name,
        "driver_version": readiness.driver_version,
        "runtime_version": readiness.runtime_version,
        "vram_total_bytes": readiness.vram_total_bytes,
        "vram_free_bytes": readiness.vram_free_bytes,
        "kernel_ok": report["steps"][0]["ok"],
        "kernel_exact": report["steps"][0]["result"]["exact_round_trip"],
        "parity_allclose": report["steps"][1]["result"]["allclose_1e12"],
        "parity_exact": report["steps"][1]["result"]["exact_equal"],
        "allocations_pool_released": all(a["pool_released"] for a in allocations),
        "allocations_free_recovered": all(a["free_recovered"] for a in allocations),
        "max_live_alloc_bytes": max(a["requested_bytes"] for a in allocations),
        "cleanup_pool_used_bytes": report["after"]["pool_used_bytes"],
    }
    report["checks"] = checks
    report["status"] = (
        "DONE"
        if all(
            [
                checks["kernel_ok"],
                checks["kernel_exact"],
                checks["parity_allclose"],
                checks["allocations_pool_released"],
            ]
        )
        else "FAILED"
    )
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_witness()
    except Exception as exc:
        report = {
            "schema": "zemosaic-r55c-gpu-witness-v1",
            "mission": "ZM-ZEGRID-R55-C-GPU-READINESS-VRAM-POLICY",
            "status": "FAILED",
            "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    data = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    out.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    out.with_name(out.name + ".sha256").write_text(
        f"{digest}  {out.name}\n", encoding="ascii"
    )
    print(f"status={report.get('status')} out={out} sha256={digest}")
    return 0 if report.get("status") == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
