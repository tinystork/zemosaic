#!/usr/bin/env python3
"""M5-B bounded RSS bench — larger canvases, real store, real admission.

Why this exists
---------------
``rss_probe.py`` measured a 24x24 / 4-cell corpus whose produced payloads were
retained in the same process and whose assembly called the RAW consumer with
``require_budget=False``. That is a useful tiny witness, but it is NOT evidence
that the production bounded path fits a larger canvas.

This harness measures a bounded larger-canvas assembly through the SAME product
entry the run uses:

* the durable store is created by a SEPARATE child process, one cell at a time
  (no list of produced payloads is retained in the producer);
* the measured child is a FRESH process that opens the immutable store
  read-only, takes a baseline AFTER imports, then calls the real
  ``_assemble_cells_guarded`` with ``bounded_results`` — so the preflight runs
  with a FRESH real ``available_memory_bytes()`` observation, the canonical
  ``ram_contract`` margin, cap=2, and the session's live-byte envelope is armed
  BEFORE the consumer opens any payload;
* the parent samples the child's ``VmRSS``/``VmHWM``/``VmSize`` and the host's
  ``MemAvailable`` during the child's life, and aborts only its OWN child if the
  reserve or the per-child RSS cap is crossed;
* commits and outputs are hashed before/after; cap, cleanup and returned-array
  lifetime are checked explicitly.

Scientific parity against an eager dense reference runs in ANOTHER process on a
SMALLER case, so the dense payload list never enters the measured process.

Everything is synthetic and deterministic: a real store, the real bounded
assembly consumer and the real admission arithmetic — NOT the full FITS
pipeline, not physical M3 waiting/recovery, not Windows/NTFS, not a BLAS-scale
universal bound. The stored identity's ``execution_context.available_ram_bytes``
is a SYNTHETIC fixture value (4 MiB), never the measured host RAM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform as _platform
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"
for _p in (str(SRC), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import test_zegrid_bounded_assembly_m5 as m5a  # noqa: E402

MARKER_PREFIX = "M5B_RSS_JSON:"
PROGRESS_PREFIX = "M5B_RSS_PROGRESS:"
SCHEMA_MEASURE = "zemosaic.zegrid.m5b.rss_bench.measure.v1"
SCHEMA_SEED = "zemosaic.zegrid.m5b.rss_bench.seed.v1"
SCHEMA_PREDICT = "zemosaic.zegrid.m5b.rss_bench.predict.v1"
SCHEMA_PARITY = "zemosaic.zegrid.m5b.rss_bench.parity.v1"
SCHEMA_CASE = "zemosaic.zegrid.m5b.rss_bench.case.v1"
SCHEMA_MATRIX = "zemosaic.zegrid.m5b.rss_bench.matrix.v1"

MAX_CANVAS = 1024
MAX_CELLS = 64
MAX_HALO = 64
MAX_CHILD_SECONDS = 90.0
MAX_TOTAL_SECONDS = 300.0
MAX_DISK_BYTES = 512 * 2 ** 20
MAX_CASE_DISK_BYTES = 256 * 2 ** 20
MIN_RESERVE_BYTES = 1 * 2 ** 30
MAX_CHILD_RSS_BYTES = 768 * 2 ** 20
SAMPLE_INTERVAL_S = 0.02
STDOUT_CAP_BYTES = 256 * 1024
MEMORY_FS = {"tmpfs", "devtmpfs", "ramfs"}

OUTPUT_PLANES = (
    "science",
    "stack_depth",
    "n_eff_support",
    "support_w1",
    "support_w2",
    "valid_fraction",
    "owner",
    "hole_mask",
)

REQUIRED_CHECKS = (
    "assembly_completed",
    "offsets_applied",
    "cap_two",
    "peak_live_cells_within_cap",
    "live_budget_armed",
    "preflight_metadata_only",
    "real_budget_observed",
    "model_fits_admission",
    "returned_arrays_alive_after_close",
    "session_closed",
    "close_released_mappings",
    "finite_measurements",
    "store_metadata_unchanged",
    "qualified_path",
    "output_planes_present",
    "timeline_ordered",
)

MIN_VALID_SAMPLES = 2

WORKSPACE_ROOT = Path("/home/tristan/.openclaw/workspace").resolve()
DEFAULT_BASE = WORKSPACE_ROOT / ".m5b-rss-linux-20261028"

DEFAULT_CASES = [
    {"name": "c512-g4x4", "canvas": (512, 512), "grid": (4, 4), "halo": 8},
    {"name": "c1024-g4x4", "canvas": (1024, 1024), "grid": (4, 4), "halo": 8},
    {"name": "c1024-g8x8", "canvas": (1024, 1024), "grid": (8, 8), "halo": 8},
]
PARITY_CASE = {"name": "parity-256-g4x4", "canvas": (256, 256), "grid": (4, 4), "halo": 8}


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def seed_for(row: int, col: int, nx: int) -> int:
    """Deterministic corpus seed for one cell (reported in provenance)."""
    return int(row) * int(nx) + int(col)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_digest(root: Path) -> str:
    """Digest of run identity + every commit record (payload identity included)."""
    root = Path(root)
    digest = hashlib.sha256()
    run_json = root / "run.json"
    if run_json.is_file():
        digest.update(b"run.json\0")
        digest.update(run_json.read_bytes())
    commits = sorted((root / "commits").rglob("*.json"))
    for path in commits:
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def tree_stat_digest(root: Path) -> str:
    """Digest of (relative path, size, mtime_ns) for every regular file."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        st = path.stat()
        rel = path.relative_to(root).as_posix()
        digest.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\0".encode("utf-8"))
    return digest.hexdigest()


def store_content_digest(root: Path) -> str:
    """Streaming content digest of EVERY store file (payload bytes included).

    Run by the PARENT before spawning and after the measured child exits, so
    the hashing pages never enter the measured child's RSS. It warms the page
    cache; that is reported, not hidden.
    """
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def dir_bytes(root: Path) -> int:
    total = 0
    root = Path(root)
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += int(path.stat().st_size)
        except OSError:
            continue
    return int(total)


def _read_proc_mem(pid: int) -> dict:
    """Current/peak RSS and virtual size for ONE process (Linux)."""
    out = {
        "available": False,
        "rss_bytes": None,
        "hwm_rss_bytes": None,
        "virtual_bytes": None,
        "virtual_peak_bytes": None,
    }
    try:
        text = Path(f"/proc/{int(pid)}/status").read_text()
    except OSError:
        return out
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    try:
        out["rss_bytes"] = int(fields["VmRSS"].split()[0]) * 1024
        out["hwm_rss_bytes"] = int(fields["VmHWM"].split()[0]) * 1024
        out["virtual_bytes"] = int(fields["VmSize"].split()[0]) * 1024
        out["virtual_peak_bytes"] = int(fields["VmPeak"].split()[0]) * 1024
        out["available"] = True
    except (KeyError, ValueError, IndexError):
        return out
    return out


def _statm(pid: int) -> dict:
    try:
        fields = Path(f"/proc/{int(pid)}/statm").read_text().split()
        page_kib = resource.getpagesize() // 1024
        names = ("size", "resident", "shared", "text", "lib", "data", "dt")
        return {
            name: int(fields[index]) * page_kib
            for index, name in enumerate(names)
            if index < len(fields)
        }
    except Exception as exc:  # pragma: no cover - non-Linux/dead process
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _self_snapshot() -> dict:
    pid = os.getpid()
    mem = _read_proc_mem(pid)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "sample_unix_ns": time.time_ns(),
        "current_rss_bytes": int(mem["rss_bytes"] or 0),
        "hwm_rss_bytes": int(mem["hwm_rss_bytes"] or 0),
        "virtual_bytes": int(mem["virtual_bytes"] or 0),
        "virtual_peak_bytes": int(mem["virtual_peak_bytes"] or 0),
        "statm_kib": _statm(pid),
        "ru_maxrss_kib": int(usage.ru_maxrss),
        "cpu_user_s": float(usage.ru_utime),
        "cpu_system_s": float(usage.ru_stime),
        "note": (
            "ru_maxrss/VmHWM include imports and every touched mmap page; they "
            "are process high-water marks, not the incremental assembly peak"
        ),
    }


def _host_meminfo() -> dict:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.strip().split()[0]) * 1024
    except Exception as exc:  # pragma: no cover
        return {"available_for_bench": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available_for_bench": True,
        "mem_total_bytes": values.get("MemTotal"),
        "mem_free_bytes": values.get("MemFree"),
        "mem_available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def _fresh_available_bytes() -> int:
    value = _host_meminfo().get("mem_available_bytes")
    if not isinstance(value, int) or value <= 0:
        raise RuntimeError("host MemAvailable is unavailable or invalid")
    return int(value)


def _fs_info(path: Path) -> dict:
    """Filesystem type/source for an EXISTING path (or its nearest ancestor)."""
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    info = {"path": str(probe), "fstype": None, "source": None, "mountpoint": None}
    try:
        out = subprocess.run(
            ["findmnt", "-T", str(probe), "-n", "-o", "FSTYPE,SOURCE,TARGET"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        line = (out.stdout or "").strip().splitlines()
        if out.returncode == 0 and line:
            parts = line[0].split()
            if parts:
                info["fstype"] = parts[0]
            if len(parts) > 1:
                info["source"] = parts[1]
            if len(parts) > 2:
                info["mountpoint"] = parts[2]
    except (OSError, subprocess.SubprocessError):
        pass
    if info["fstype"] is None:
        try:
            best = None
            target = str(probe)
            for line in Path("/proc/mounts").read_text().splitlines():
                fields = line.split()
                if len(fields) < 2:
                    continue
                mp = fields[1]
                if target == mp or target.startswith(mp.rstrip("/") + "/"):
                    if best is None or len(mp) > len(best[1]):
                        best = (fields[2], fields[0], mp)
            if best is not None:
                info["fstype"], info["source"], info["mountpoint"] = best
        except OSError:
            pass
    try:
        st = os.statvfs(str(probe))
        info["free_bytes"] = int(st.f_bavail) * int(st.f_frsize)
        info["total_bytes"] = int(st.f_blocks) * int(st.f_frsize)
    except OSError:
        info["free_bytes"] = None
        info["total_bytes"] = None
    return info


def assert_real_disk(path: Path) -> dict:
    """Refuse a RAM-backed or UNKNOWN filesystem.

    An undetected fstype is NOT evidence of a real disk; it must be an explicit
    refusal rather than a permissive default.
    """
    info = _fs_info(path)
    fstype = str(info.get("fstype") or "").lower()
    if not fstype:
        raise RuntimeError(
            f"refusing {path}: filesystem type is unknown; cannot certify a "
            "real disk (no permissive default)"
        )
    if fstype in MEMORY_FS:
        raise RuntimeError(
            f"refusing {path}: filesystem {fstype!r} at {info.get('mountpoint')} "
            "is RAM-backed; the store would consume RAM and confound pressure"
        )
    if info.get("free_bytes") is not None and int(info["free_bytes"]) <= 0:
        raise RuntimeError(f"refusing {path}: no free bytes reported")
    info["real_disk"] = True
    return info


def _ensure_under_workspace(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise RuntimeError(
            f"artifact base {resolved} is outside the owned workspace "
            f"{WORKSPACE_ROOT}; refusing to create stores elsewhere"
        ) from None
    return resolved


def parse_canvas(value: str) -> tuple[int, int]:
    try:
        h, w = (int(part) for part in str(value).lower().split("x", 1))
    except Exception:
        raise argparse.ArgumentTypeError(f"canvas must be HxW, got {value!r}") from None
    return h, w


def parse_grid(value: str) -> tuple[int, int]:
    try:
        nx, ny = (int(part) for part in str(value).lower().split("x", 1))
    except Exception:
        raise argparse.ArgumentTypeError(f"grid must be NxM, got {value!r}") from None
    return nx, ny


def validate_case(case: dict) -> dict:
    h, w = int(case["canvas"][0]), int(case["canvas"][1])
    nx, ny = int(case["grid"][0]), int(case["grid"][1])
    halo = int(case.get("halo", 8))
    if h <= 0 or w <= 0 or max(h, w) > MAX_CANVAS:
        raise ValueError(f"case {case.get('name')}: canvas {h}x{w} exceeds {MAX_CANVAS}")
    if nx <= 0 or ny <= 0 or nx * ny > MAX_CELLS:
        raise ValueError(f"case {case.get('name')}: grid {nx}x{ny} exceeds {MAX_CELLS} cells")
    if halo <= 0 or halo > MAX_HALO:
        raise ValueError(f"case {case.get('name')}: halo {halo} outside 1..{MAX_HALO}")
    return {
        "name": str(case["name"]),
        "canvas": (h, w),
        "grid": (nx, ny),
        "halo": halo,
        "cells": int(nx * ny),
    }


def _case_argv(verb: str, spec: dict, root: Path, **extra) -> list[str]:
    h, w = spec["canvas"]
    nx, ny = spec["grid"]
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(verb),
        "--root",
        str(root),
        "--canvas",
        f"{h}x{w}",
        "--grid",
        f"{nx}x{ny}",
        "--halo",
        str(int(spec["halo"])),
    ]
    for key, value in extra.items():
        if value is None or value is False:
            continue
        flag = "--" + str(key).replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


# ---------------------------------------------------------------------------
# Output digests / labels (shared by the measured child and parity)
# ---------------------------------------------------------------------------


def canvas_output_digest(assembled) -> dict:
    """Digest every returned canvas plane (shape/dtype/bytes)."""
    names = (
        "science",
        "stack_depth",
        "n_eff_support",
        "support_w1",
        "support_w2",
        "valid_fraction",
        "owner",
        "hole_mask",
    )
    out = {}
    for name in names:
        array = np.asarray(getattr(assembled, name))
        contiguous = np.ascontiguousarray(array)
        out[name] = {
            "shape": [int(v) for v in array.shape],
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "sha256": _sha256_bytes(contiguous.tobytes(order="C")),
        }
    return out


def store_commit_snapshot(root: Path) -> dict:
    return {
        "commit_digest": commit_digest(root),
        "tree_stat_digest": tree_stat_digest(root),
        "content_digest": None,
        "files": int(sum(1 for p in Path(root).rglob("*") if p.is_file())),
        "bytes": int(dir_bytes(root)),
    }


def _timeline_ordered(result: dict) -> tuple[bool, list[str]]:
    """Nested temporal proof: snapshot < assembly < close < post-close RSS."""
    timeline = result.get("timeline") or {}
    keys = (
        "store_snapshot_before_monotonic_s",
        "preassembly_monotonic_s",
        "assembly_start_monotonic_s",
        "assembly_end_monotonic_s",
        "close_start_monotonic_s",
        "close_end_monotonic_s",
        "post_close_rss_monotonic_s",
        "done_monotonic_s",
    )
    missing = [key for key in keys if not _finite_nonneg(timeline.get(key))]
    if missing:
        return False, [f"timeline missing {missing}"]
    values = [float(timeline[key]) for key in keys]
    if values != sorted(values):
        return False, ["timeline is not monotonic in phase order"]
    if not (
        float(timeline["assembly_start_monotonic_s"])
        <= float(timeline["assembly_end_monotonic_s"])
    ):
        return False, ["assembly_start is after assembly_end"]
    return True, []


def _validate_refusal(result: dict) -> tuple[bool, list[str]]:
    """A REFUSED result is NOT_RUN only when it proves a typed pre-payload stop."""
    reasons: list[str] = []
    if str(result.get("status")) != "REFUSED":
        reasons.append("status is not REFUSED")
    refusal = result.get("refusal") or {}
    if not isinstance(refusal, dict) or not refusal:
        reasons.append("refusal evidence block missing")
        return False, reasons
    if refusal.get("typed") is not True:
        reasons.append("refusal is not typed")
    if str(refusal.get("type")) != "AssemblyCapacityError":
        reasons.append("refusal type is not AssemblyCapacityError")
    if str(refusal.get("phase")) != "preflight":
        reasons.append("refusal phase is not preflight")
    if refusal.get("assembly_progressed") is not False:
        reasons.append("refusal claims assembly progress or omits the flag")
    for key in ("payloads_loaded", "verifications", "live_mappings"):
        value = refusal.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            reasons.append(f"refusal {key} is not exactly 0")
    plan = refusal.get("plan")
    if not isinstance(plan, dict) or not plan:
        reasons.append("refusal plan is missing/empty")
    elif not _finite_nonneg(plan.get("required_bytes")) or not _finite_nonneg(
        plan.get("available_bytes")
    ):
        reasons.append("refusal plan lacks finite required/available bytes")
    assembly = result.get("assembly") or {}
    if assembly.get("completed") is not False:
        reasons.append("refusal says assembly completed")
    if assembly.get("progressed") is not False:
        reasons.append("refusal says assembly progressed or omits the flag")
    stats = result.get("stats") or {}
    for key in ("loads", "verifications", "live_mappings"):
        value = stats.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            reasons.append(f"refusal stats.{key} is not exactly 0")
    checks = result.get("checks") or {}
    if checks.get("refused_before_payload") is not True:
        reasons.append("refusal check refused_before_payload is not true")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Child verbs: seed / predict / measure / parity
# ---------------------------------------------------------------------------


def _progress(phase: str, **extra) -> None:
    payload = {"phase": str(phase), "unix_ns": time.time_ns()}
    payload.update(extra)
    print(PROGRESS_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _write_result(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _resolve_identity(spec: dict) -> dict:
    h, w = spec["canvas"]
    nx, ny = spec["grid"]
    return m5a.make_identity(nx=nx, ny=ny, canvas_hw=(h, w), halo=spec["halo"])


def cmd_seed(args: argparse.Namespace) -> int:
    from zemosaic.core.zegrid import run_store as zrs

    spec = validate_case(
        {
            "name": "seed",
            "canvas": parse_canvas(args.canvas),
            "grid": parse_grid(args.grid),
            "halo": int(args.halo),
        }
    )
    root = Path(args.root).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"seed root {root} is not empty; refusing to overwrite")
    if not root.exists():
        root.mkdir(parents=True)
    h, w = spec["canvas"]
    nx, ny = spec["grid"]
    identity = _resolve_identity(spec)
    budget = zrs.DiskBudget(
        durable_cell_bytes=MAX_CASE_DISK_BYTES,
        transactional_margin_bytes=16 * 2 ** 20,
    )
    store = zrs.RunCellStore.open(root, identity=identity, budget=budget)
    started = time.monotonic()
    cells = []
    max_cell_bytes = 0
    total_cell_bytes = 0
    try:
        for row in range(ny):
            for col in range(nx):
                seed = seed_for(row, col, nx)
                result = m5a.complete_result(
                    row,
                    col,
                    seed=seed,
                    nx=nx,
                    ny=ny,
                    h=h,
                    w=w,
                    halo=spec["halo"],
                )
                cell_id = str(result["cell_id"])
                receipt = store.commit_cell(
                    result,
                    attempt_id=f"a-{row}-{col}",
                    backend="cpu",
                )
                cell_bytes = int(receipt.logical_bytes)
                max_cell_bytes = max(max_cell_bytes, cell_bytes)
                total_cell_bytes += cell_bytes
                cells.append(
                    {
                        "cell_id": cell_id,
                        "row": int(row),
                        "col": int(col),
                        "seed": int(seed),
                        "attempt_id": f"a-{row}-{col}",
                        "payload_sha256": str(receipt.payload_sha256),
                        "logical_bytes": cell_bytes,
                        "commit_file": str(receipt.commit_file),
                    }
                )
                # Producer process isolation: only ONE produced payload exists at
                # a time. Drop it before building the next cell.
                del result
                del receipt
                if (row * nx + col) % 8 == 7:
                    gc.collect()
        _progress("seed_done", cells=len(cells))
    finally:
        store.close()
    payload = {
        "schema": SCHEMA_SEED,
        "status": "SEEDED",
        "qualified_measurement": False,
        "case": {
            "canvas": [h, w],
            "grid": [nx, ny],
            "halo": int(spec["halo"]),
            "cells": int(nx * ny),
            "seed_formula": "seed = row * nx + col",
        },
        "store": {
            "root": str(root),
            "commit_snapshot": store_commit_snapshot(root),
            "disk_bytes": int(dir_bytes(root)),
            "max_cell_logical_bytes": int(max_cell_bytes),
            "total_cell_logical_bytes": int(total_cell_bytes),
        },
        "cells": cells,
        "identity": {
            "run_id": identity["run_id"],
            "execution_context": identity.get("execution_context"),
            "note": (
                "execution_context.available_ram_bytes is the SYNTHETIC test "
                "fixture value (4 MiB) inherited from the M5-A helper; it is "
                "NOT the measured host RAM and does not drive this bench"
            ),
        },
        "producer": {
            "elapsed_s": float(time.monotonic() - started),
            "retained_payloads": 0,
            "note": "one produced cell at a time; no result list is retained",
        },
    }
    _write_result(Path(args.result), payload)
    print(MARKER_PREFIX + json.dumps({"status": "SEEDED", "result": str(args.result)}))
    return 0


def _open_read_only(root: Path):
    from zemosaic.core.zegrid import run_store as zrs

    return zrs.RunCellStore.open(root, identity=None, read_only=True, create=False)


def _access_and_ids(root: Path, *, max_resident_cells: int = 2):
    from zemosaic.core.zegrid import bounded_assembly as zbas

    store = _open_read_only(root)
    access = zbas.StoreAssemblyAccess(store, max_resident_cells=int(max_resident_cells))
    ids = list(access.session.cell_ids())
    return store, access, ids


def cmd_predict(args: argparse.Namespace) -> int:
    """Metadata-only predicted plan (never a measured budget claim)."""
    from zemosaic.core.zegrid import bounded_assembly as zbas
    from zemosaic.core.zegrid import ram_contract as rcontract
    from zemosaic import zemosaic_zegrid_mode as zz

    spec = validate_case(
        {
            "name": "predict",
            "canvas": parse_canvas(args.canvas),
            "grid": parse_grid(args.grid),
            "halo": int(args.halo),
        }
    )
    root = Path(args.root).resolve()
    h, w = spec["canvas"]
    nx, ny = spec["grid"]
    raw = int(zz.available_memory_bytes())
    budget = int(rcontract.admit_budget_bytes(raw))
    store, access, ids = _access_and_ids(root)
    try:
        plans = {}
        refusals = {}
        for gauge_mode in ("per_cell", "global"):
            try:
                plans[gauge_mode] = zbas.preflight_assembly(
                    access.session,
                    ids,
                    canvas_hw=(h, w),
                    available_bytes=budget,
                    require_budget=True,
                    purpose=f"rss_prediction_{gauge_mode}",
                    gauge_mode=gauge_mode,
                    grid=(nx, ny),
                )
            except zbas.AssemblyCapacityError as exc:
                refusals[gauge_mode] = dict(exc.plan)
                refusals[gauge_mode]["error"] = str(exc)
        plan = plans.get("per_cell")
        stats = access.stats()
    finally:
        access.close()
        store.close()
    if plan is None:
        payload = {
            "schema": SCHEMA_PREDICT,
            "status": "PREDICTED_REFUSED",
            "qualified_measurement": False,
            "case": {
                "canvas": [h, w],
                "grid": [nx, ny],
                "halo": int(spec["halo"]),
                "cells": int(nx * ny),
            },
            "host": {
                "raw_available_bytes": raw,
                "admit_budget_bytes": budget,
                "budget_source": "ram_contract.admit_budget_bytes",
                "observed_unix_ns": time.time_ns(),
            },
            "plan": None,
            "refusals": refusals,
            "session": stats,
            "note": "metadata-only prediction refused by the real budget",
        }
        _write_result(Path(args.result), payload)
        print(MARKER_PREFIX + json.dumps({"status": "PREDICTED_REFUSED"}))
        return 0
    payload = {
        "schema": SCHEMA_PREDICT,
        "status": "PREDICTED",
        "qualified_measurement": False,
        "case": {
            "canvas": [h, w],
            "grid": [nx, ny],
            "halo": int(spec["halo"]),
            "cells": int(nx * ny),
        },
        "host": {
            "raw_available_bytes": raw,
            "admit_budget_bytes": budget,
            "budget_source": "ram_contract.admit_budget_bytes",
            "observed_unix_ns": time.time_ns(),
        },
        "plan": plan,
        "plans": plans,
        "refusals": refusals,
        "session": stats,
        "note": (
            "pure metadata-only plans (per_cell and global); no payload was "
            "loaded and no assembly ran. Shown separately, NOT measurement "
            "evidence"
        ),
    }
    _write_result(Path(args.result), payload)
    print(MARKER_PREFIX + json.dumps({"status": "PREDICTED"}))
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Fresh measured child: real store read + REAL guarded assembly.

    Only a typed ``AssemblyCapacityError`` proven to occur BEFORE any payload
    load/verification/mapping yields ``REFUSED`` (parent: NOT_RUN). Any other
    exception, or a capacity error after progress, is an ERROR result with a
    nonzero exit status (parent: FAIL). Timeouts and guard breaches are also
    FAIL by parent precedence.
    """
    import traceback

    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import bounded_assembly as zbas
    from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: F401

    spec = validate_case(
        {
            "name": "measure",
            "canvas": parse_canvas(args.canvas),
            "grid": parse_grid(args.grid),
            "halo": int(args.halo),
        }
    )
    root = Path(args.root).resolve()
    h, w = spec["canvas"]
    nx, ny = spec["grid"]
    result_path = Path(args.result).resolve()
    timeline: dict[str, float] = {}

    baseline = _self_snapshot()
    _progress("imports_done")
    host_before = _host_meminfo()
    timeline["imports_done_monotonic_s"] = time.monotonic()
    store, access, ids = _access_and_ids(root)

    def _case_block() -> dict:
        return {
            "canvas": [h, w],
            "grid": [nx, ny],
            "halo": int(spec["halo"]),
            "cells": len(ids),
            "cell_ids": ids,
            "seed_formula": "seed = row * nx + col",
        }

    def _admission_block() -> dict:
        return {
            "budget_source": (
                "caller_explicit_diagnostic"
                if args.available_bytes is not None
                else "ram_contract.admit_budget_bytes"
            ),
            "caller_explicit_available_bytes": (
                None if args.available_bytes is None else int(args.available_bytes)
            ),
        }

    def _write_error(
        *,
        error: dict,
        stats: dict,
        before: dict,
        elapsed: float,
        progressed: bool,
        phase: str,
        returncode: int,
    ) -> int:
        payload = {
            "schema": SCHEMA_MEASURE,
            "status": "ERROR",
            "qualified_measurement": False,
            "case": _case_block(),
            "host": {"before": host_before, "after": _host_meminfo()},
            "baseline_imports": baseline,
            "preassembly": before,
            "assembly": {
                "completed": False,
                "progressed": bool(progressed),
                "elapsed_s": float(elapsed),
                "error": error,
            },
            "admission": _admission_block(),
            "stats": stats,
            "stats_after_close": stats,
            "timeline": timeline,
            "checks": {},
            "note": (
                "ERROR result: a genuine exception or a capacity refusal after "
                "payload/assembly progress is NOT a qualified refusal"
            ),
        }
        _write_result(result_path, payload)
        _progress("done", status="ERROR", error_phase=phase)
        print(MARKER_PREFIX + json.dumps({"status": "ERROR", "phase": phase}))
        return int(returncode)

    try:
        store_before = store_commit_snapshot(root)
        timeline["store_snapshot_before_monotonic_s"] = time.monotonic()
        canvas = m5a.make_canvas(h, w)
        bounded = access.results(ids)
        before = _self_snapshot()
        timeline["preassembly_monotonic_s"] = time.monotonic()
        _progress("preassembly", current_rss_bytes=before["current_rss_bytes"])
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        wall_before = time.perf_counter()
        timeline["assembly_start_monotonic_s"] = time.monotonic()
        _progress("assembly_start", current_rss_bytes=before["current_rss_bytes"])
        admitted_available = None
        assembled = None
        info = None
        preflight_plan = None
        try:
            if args.available_bytes is not None:
                # DIAGNOSTIC ONLY (guard-path test): an explicit caller budget
                # bypasses the fresh observation. Never used by the matrix.
                admitted_available = int(args.available_bytes)
            if args.inject_error == "generic":
                raise RuntimeError("injected generic failure before guarded assembly")
            if args.inject_capacity_after_load:
                # Deliberate post-progress refusal: one REAL payload load first.
                access.session.load(ids[0])
                raise zbas.AssemblyCapacityError(
                    "injected capacity refusal after payload progress",
                    plan={
                        "required_bytes": 1,
                        "available_bytes": 0,
                        "injected": True,
                    },
                    required_bytes=1,
                    available_bytes=0,
                )
            assembled, info, preflight_plan = zz._assemble_cells_guarded(
                canvas,
                nx,
                ny,
                "per_cell",
                access.cores(ids),
                access.descriptors(ids),
                bounded_results=bounded,
                available_bytes=admitted_available,
                purpose="rss_bench_measure",
                halo_px=int(spec["halo"]),
                patches=access.patches(ids),
            )
        except zbas.AssemblyCapacityError as exc:
            wall_after = time.perf_counter()
            after = _self_snapshot()
            timeline["assembly_end_monotonic_s"] = time.monotonic()
            stats = access.stats()
            progressed = bool(
                int(stats.get("loads", 0))
                or int(stats.get("verifications", 0))
                or int(stats.get("live_mappings", 0))
            )
            elapsed = float(wall_after - wall_before)
            if progressed:
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "phase": "post_progress_capacity",
                    "traceback": traceback.format_exc(),
                    "plan": dict(exc.plan),
                    "required_bytes": int(exc.required_bytes or 0),
                    "available_bytes": int(exc.available_bytes or 0),
                }
                return _write_error(
                    error=error,
                    stats=stats,
                    before=before,
                    elapsed=elapsed,
                    progressed=True,
                    phase="post_progress_capacity",
                    returncode=4,
                )
            payload = {
                "schema": SCHEMA_MEASURE,
                "status": "REFUSED",
                "qualified_measurement": False,
                "case": _case_block(),
                "host": {"before": host_before, "after": _host_meminfo()},
                "baseline_imports": baseline,
                "preassembly": before,
                "assembly": {
                    "completed": False,
                    "progressed": False,
                    "elapsed_s": elapsed,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "phase": "preflight",
                    },
                },
                "refusal": {
                    "typed": True,
                    "type": "AssemblyCapacityError",
                    "phase": "preflight",
                    "assembly_progressed": False,
                    "payloads_loaded": int(stats.get("loads", 0)),
                    "verifications": int(stats.get("verifications", 0)),
                    "live_mappings": int(stats.get("live_mappings", 0)),
                    "plan": dict(exc.plan),
                    "message": str(exc),
                },
                "admission": {
                    **_admission_block(),
                    "required_bytes": int(exc.required_bytes or 0),
                    "available_bytes": int(exc.available_bytes or 0),
                },
                "stats": stats,
                "stats_after_close": stats,
                "timeline": timeline,
                "checks": {
                    "refused_before_payload": (
                        int(stats.get("loads", 0)) == 0
                        and int(stats.get("verifications", 0)) == 0
                        and int(stats.get("live_mappings", 0)) == 0
                    )
                },
            }
            _write_result(result_path, payload)
            _progress("done", status="REFUSED")
            print(MARKER_PREFIX + json.dumps({"status": "REFUSED"}))
            return 0
        except Exception as exc:
            wall_after = time.perf_counter()
            after = _self_snapshot()
            timeline["assembly_end_monotonic_s"] = time.monotonic()
            stats = access.stats()
            progressed = bool(
                int(stats.get("loads", 0))
                or int(stats.get("verifications", 0))
                or int(stats.get("live_mappings", 0))
            )
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "phase": "assembly",
                "traceback": traceback.format_exc(),
                "payloads_loaded": int(stats.get("loads", 0)),
                "verifications": int(stats.get("verifications", 0)),
                "live_mappings": int(stats.get("live_mappings", 0)),
            }
            return _write_error(
                error=error,
                stats=stats,
                before=before,
                elapsed=float(wall_after - wall_before),
                progressed=progressed,
                phase="assembly",
                returncode=3,
            )
        wall_after = time.perf_counter()
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        after = _self_snapshot()
        timeline["assembly_end_monotonic_s"] = time.monotonic()
        _progress(
            "assembly_end",
            current_rss_bytes=after["current_rss_bytes"],
            error=None,
        )
        stats_after_assembly = access.stats()

        # Digest while the returned arrays are alive. The hashing time/peak is
        # deliberately OUTSIDE the assembly window (see assembly.elapsed_s and
        # timeline), never silently relabelled as assembly work.
        digest_before_close = canvas_output_digest(assembled)
        timeline["output_digest_before_close_monotonic_s"] = time.monotonic()
        timeline["close_start_monotonic_s"] = time.monotonic()
        _progress("close_start")
        access.close()
        gc_started = time.monotonic()
        gc.collect()
        timeline["close_end_monotonic_s"] = time.monotonic()
        stats_after_close = access.stats()
        post_close = _self_snapshot()
        timeline["post_close_rss_monotonic_s"] = time.monotonic()
        digest_after_close = canvas_output_digest(assembled)
        store_after = store_commit_snapshot(root)
        timeline["done_monotonic_s"] = time.monotonic()
        _progress("closed", current_rss_bytes=post_close["current_rss_bytes"])

        elapsed = float(wall_after - wall_before)
        cpu_s = float(
            (cpu_after.ru_utime - cpu_before.ru_utime)
            + (cpu_after.ru_stime - cpu_before.ru_stime)
        )
        peak_live_cells = int(stats_after_assembly.get("peak_live_cells", 0))
        cap = int(stats_after_assembly.get("max_resident_cells", 0))
        max_live_bytes = stats_after_assembly.get("max_live_bytes")
        peak_live_bytes = int(stats_after_assembly.get("peak_live_mapped_bytes", 0))
        plan = preflight_plan or {}
        graph = plan.get("graph") or {}
        required = int(plan.get("required_bytes", 0))
        available = int(plan.get("available_bytes", 0))
        host_admission = plan.get("host") or {}
        raw_available = host_admission.get("raw_available_bytes")
        budget_source = host_admission.get("budget_source")
        output_planes_ok = all(
            name in digest_before_close
            and isinstance(digest_before_close[name].get("sha256"), str)
            and len(digest_before_close[name]["sha256"]) == 64
            for name in OUTPUT_PLANES
        )

        timeline_ok, timeline_reasons = _timeline_ordered(
            {"timeline": timeline}
        )
        store_metadata_unchanged = bool(
            store_before.get("commit_digest") == store_after.get("commit_digest")
            and store_before.get("tree_stat_digest") == store_after.get("tree_stat_digest")
        )
        checks = {
            "assembly_completed": assembled is not None,
            "offsets_applied": bool(info and info.get("offsets")),
            "cap_two": cap == 2,
            "peak_live_cells_within_cap": 1 <= peak_live_cells <= cap and cap == 2,
            "live_budget_armed": (
                isinstance(max_live_bytes, int)
                and max_live_bytes > 0
                and peak_live_bytes > 0
                and peak_live_bytes <= max_live_bytes
            ),
            "preflight_metadata_only": bool(
                (plan.get("preflight") or {}).get("payloads_loaded_before") == 0
                and (plan.get("preflight") or {}).get("verifications_before") == 0
            ),
            "real_budget_observed": bool(budget_source == "ram_contract.admit_budget_bytes"),
            "model_fits_admission": bool(required > 0 and available > 0 and required <= available),
            "returned_arrays_alive_after_close": digest_before_close == digest_after_close,
            "session_closed": bool(stats_after_close.get("closed")),
            "close_released_mappings": (
                int(stats_after_close.get("live_cells", -1)) == 0
                and int(stats_after_close.get("live_mapped_bytes", -1)) == 0
            ),
            "finite_measurements": all(
                math.isfinite(float(value)) and float(value) >= 0
                for value in (
                    baseline["current_rss_bytes"],
                    before["current_rss_bytes"],
                    after["current_rss_bytes"],
                    post_close["current_rss_bytes"],
                    elapsed,
                    cpu_s,
                    max_live_bytes or 0,
                )
            ),
            "store_metadata_unchanged": store_metadata_unchanged,
            "qualified_path": bool(args.available_bytes is None),
            "output_planes_present": output_planes_ok,
            "timeline_ordered": bool(timeline_ok),
        }
        status = "PASS" if all(checks.values()) else "CHECK_FAILED"
        payload = {
            "schema": SCHEMA_MEASURE,
            "status": status,
            "qualified_measurement": bool(args.available_bytes is None),
            "case": _case_block(),
            "platform": {
                "system": _platform.system(),
                "release": _platform.release(),
                "machine": _platform.machine(),
                "python": _platform.python_version(),
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "cpu_count": os.cpu_count(),
                "page_size": resource.getpagesize(),
            },
            "provenance": {
                "repo_root": str(REPO_ROOT),
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "module_sha256": {
                    "zemosaic_zegrid_mode.py": _sha256_file(
                        REPO_ROOT / "src/zemosaic/zemosaic_zegrid_mode.py"
                    ),
                    "bounded_assembly.py": _sha256_file(
                        REPO_ROOT / "src/zemosaic/core/zegrid/bounded_assembly.py"
                    ),
                    "cell_feather.py": _sha256_file(
                        REPO_ROOT / "src/zemosaic/core/zegrid/cell_feather.py"
                    ),
                    "run_store.py": _sha256_file(
                        REPO_ROOT / "src/zemosaic/core/zegrid/run_store.py"
                    ),
                    "m5a_fixture.py": _sha256_file(
                        REPO_ROOT / "tests/test_zegrid_bounded_assembly_m5.py"
                    ),
                },
                "corpus": (
                    "synthetic deterministic producer (M5-A helper geometry) "
                    "committed through the REAL durable store; not FITS, not M31"
                ),
                "identity_execution_context": (
                    "synthetic fixture (available_ram_bytes=4 MiB), NOT host RAM"
                ),
            },
            "host": {
                "before": host_before,
                "after": _host_meminfo(),
                "sample_available_unix_ns": time.time_ns(),
            },
            "baseline_imports": baseline,
            "preassembly": before,
            "assembly": {
                "completed": True,
                "progressed": True,
                "elapsed_s": elapsed,
                "cpu_user_s": float(cpu_after.ru_utime - cpu_before.ru_utime),
                "cpu_system_s": float(cpu_after.ru_stime - cpu_before.ru_stime),
                "cpu_total_s": cpu_s,
                "post_current_rss_bytes": int(after["current_rss_bytes"]),
                "post_hwm_rss_bytes": int(after["hwm_rss_bytes"]),
                "post_virtual_bytes": int(after["virtual_bytes"]),
                "post_virtual_peak_bytes": int(after["virtual_peak_bytes"]),
                "post_close_current_rss_bytes": int(post_close["current_rss_bytes"]),
                "delta_pre_to_post_current_bytes": int(
                    after["current_rss_bytes"] - before["current_rss_bytes"]
                ),
                "hwm_minus_baseline_hwm_bytes": int(
                    after["hwm_rss_bytes"] - baseline["hwm_rss_bytes"]
                ),
                "post_close_snapshot": post_close,
                "notes": [
                    "ru_maxrss/VmHWM include imports and touched mmap pages; "
                    "hwm_minus_baseline_hwm is a process HWM subtraction, NOT "
                    "the incremental assembly peak",
                    "the parent's sampled peak during the assembly window is the "
                    "reported observed peak; the lifetime sampled peak includes "
                    "post-assembly digest/close work and is labelled separately",
                ],
            },
            "admission": {
                "budget_source": budget_source,
                "raw_available_bytes": raw_available,
                "available_bytes": available,
                "required_bytes": required,
                "fits": bool(plan.get("fits")),
                "preflight": plan.get("preflight"),
            },
            "model": {
                "required_bytes": required,
                "canvas_bytes": int((plan.get("canvas") or {}).get("total_bytes", 0)),
                "canvas_detail": plan.get("canvas"),
                "cells": plan.get("cells"),
                "resident_mapped_bytes": int(
                    (plan.get("cells") or {}).get("resident_mapped_bytes", 0)
                ),
                "store_total_bytes": int(
                    (plan.get("cells") or {}).get("store_total_bytes", 0)
                ),
                "transient_bytes": int((plan.get("transient") or {}).get("total_bytes", 0)),
                "dense_solver_bytes": int(graph.get("total_bytes", 0)),
                "dense_solver": graph,
                "scratch_bytes": int(plan.get("scratch_bytes", 0)),
                "assumptions": plan.get("notes"),
            },
            "stats": stats_after_assembly,
            "stats_after_close": stats_after_close,
            "counters": {
                "max_resident_cells": cap,
                "peak_live_cells": peak_live_cells,
                "peak_live_mappings": int(stats_after_assembly.get("peak_live_mappings", 0)),
                "peak_live_mapped_bytes": peak_live_bytes,
                "max_live_bytes": max_live_bytes,
                "loads": int(stats_after_assembly.get("loads", 0)),
                "evictions": int(stats_after_assembly.get("evictions", 0)),
                "verifications": int(stats_after_assembly.get("verifications", 0)),
                "reopen_generations": int(
                    stats_after_assembly.get("reopen_generations", 0)
                ),
            },
            "geometry": {
                "anchor": info.get("anchor") if info else None,
                "edges_used": int(info.get("edges_used", 0)) if info else 0,
                "edges_built": int(info.get("edges_built", 0)) if info else 0,
                "disconnected": list(info.get("disconnected", [])) if info else [],
                "policy": info.get("policy") if info else None,
                "halo_px": int(spec["halo"]),
                "offsets": info.get("offsets") if info else None,
            },
            "digests": {
                "output": digest_before_close,
                "output_after_close": digest_after_close,
                "store_commits_before": store_before.get("commit_digest"),
                "store_commits_after": store_after.get("commit_digest"),
                "store_tree_before": store_before.get("tree_stat_digest"),
                "store_tree_after": store_after.get("tree_stat_digest"),
                "store_snapshot_before": store_before,
                "store_snapshot_after": store_after,
                "scope": (
                    "child snapshots cover commit records + file path/size/mtime "
                    "before assembly and after close; the PARENT additionally "
                    "stream-hashes every store file before spawn/after exit"
                ),
            },
            "timeline": timeline,
            "timeline_reasons": timeline_reasons,
            "timings": {
                "gc_s": float(time.monotonic() - gc_started),
                "digest_before_close_included_in_lifetime_peak": True,
                "assembly_elapsed_s": elapsed,
            },
            "checks": checks,
            "note": (
                "single-host Linux observation of a synthetic deterministic "
                "producer through the real store + real guarded assembly. It is "
                "not a universal memory guarantee and does not cover the full "
                "FITS product, physical M3 waiting/recovery, Windows/NTFS or "
                "BLAS-scale canvases."
            ),
        }
        _write_result(result_path, payload)
        _progress("done", status=status)
        print(MARKER_PREFIX + json.dumps({"status": status, "result": str(result_path)}))
        return 0
    finally:
        try:
            access.close()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass

def cmd_parity(args: argparse.Namespace) -> int:
    """Small-case parity: bounded store path vs eager dense reference.

    Runs in its OWN process so the retained dense payload list can never
    influence a measured baseline.
    """
    from zemosaic import zemosaic_zegrid_mode as zz

    spec = validate_case(
        {
            "name": "parity",
            "canvas": parse_canvas(args.canvas),
            "grid": parse_grid(args.grid),
            "halo": int(args.halo),
        }
    )
    root = Path(args.root).resolve()
    h, w = spec["canvas"]
    nx, ny = spec["grid"]
    store, access, ids = _access_and_ids(root)
    try:
        canvas_bounded = m5a.make_canvas(h, w)
        bounded = access.results(ids)
        assembled_bounded, info_bounded, plan = zz._assemble_cells_guarded(
            canvas_bounded,
            nx,
            ny,
            "per_cell",
            access.cores(ids),
            access.descriptors(ids),
            bounded_results=bounded,
            purpose="rss_bench_parity",
            halo_px=int(spec["halo"]),
            patches=access.patches(ids),
        )
        digest_bounded = canvas_output_digest(assembled_bounded)
        stats = access.stats()
    finally:
        try:
            access.close()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass

    # Dense reference in this separate process: rebuild the SAME deterministic
    # payloads in memory (no store read), retain the eager list, raw consumer.
    results = []
    for row in range(ny):
        for col in range(nx):
            results.append(
                m5a.complete_result(
                    row,
                    col,
                    seed=seed_for(row, col, nx),
                    nx=nx,
                    ny=ny,
                    h=h,
                    w=w,
                    halo=spec["halo"],
                )
            )
    canvas_dense = m5a.make_canvas(h, w)
    assembled_dense, info_dense = zz._assemble_cells(
        canvas_dense,
        nx,
        ny,
        "per_cell",
        m5a.eager_cores(results),
        {r["cell_id"]: r for r in results},
        halo_px=int(spec["halo"]),
        patches=m5a.eager_patches(results),
    )
    digest_dense = canvas_output_digest(assembled_dense)

    mismatches = []

    def _cmp(name: str, left, right) -> None:
        try:
            if left.dtype.kind == "f":
                same = np.array_equal(left, right, equal_nan=True)
            else:
                same = np.array_equal(left, right)
        except Exception as exc:
            mismatches.append(f"{name}: comparison error {type(exc).__name__}: {exc}")
            return
        if not same:
            mismatches.append(f"{name}: pixel mismatch")

    for name in (
        "science",
        "stack_depth",
        "n_eff_support",
        "support_w1",
        "support_w2",
        "valid_fraction",
        "owner",
        "hole_mask",
    ):
        _cmp(name, getattr(assembled_bounded, name), getattr(assembled_dense, name))
    if list(assembled_bounded.complete_cells) != list(assembled_dense.complete_cells):
        mismatches.append("complete_cells mismatch")
    if list(assembled_bounded.incomplete_cells) != list(assembled_dense.incomplete_cells):
        mismatches.append("incomplete_cells mismatch")
    if info_bounded.get("offsets") != info_dense.get("offsets"):
        mismatches.append("offset graph mismatch")
    for key in digest_bounded:
        if digest_bounded[key]["sha256"] != digest_dense[key]["sha256"]:
            mismatches.append(f"digest mismatch: {key}")
    status = "PARITY_PASS" if not mismatches else "PARITY_FAIL"
    payload = {
        "schema": SCHEMA_PARITY,
        "status": status,
        "qualified_measurement": False,
        "case": {
            "canvas": [h, w],
            "grid": [nx, ny],
            "halo": int(spec["halo"]),
            "cells": int(nx * ny),
        },
        "mismatches": mismatches,
        "bounded": {
            "digests": digest_bounded,
            "stats": stats,
            "offsets": info_bounded.get("offsets"),
            "plan_required_bytes": int((plan or {}).get("required_bytes", 0)),
        },
        "dense": {
            "digests": digest_dense,
            "retained_payloads": len(results),
            "offsets": info_dense.get("offsets"),
        },
        "note": (
            "separate process; the dense payload list is intentionally retained "
            "HERE only and never in a measured child"
        ),
    }
    _write_result(Path(args.result), payload)
    print(MARKER_PREFIX + json.dumps({"status": status}))
    return 0 if status == "PARITY_PASS" else 1


# ---------------------------------------------------------------------------
# Parent supervision
# ---------------------------------------------------------------------------


def monitor_decision(
    *,
    rss_bytes: int | None,
    available_bytes: int | None,
    rss_cap_bytes: int = MAX_CHILD_RSS_BYTES,
    reserve_bytes: int = MIN_RESERVE_BYTES,
) -> str | None:
    """Pure guard: return a breach reason or None.

    Unknown telemetry is NOT equivalent to safety. A living child with no RSS
    or no host availability is an explicit abort reason.
    """
    if rss_bytes is None:
        return "TELEMETRY_UNAVAILABLE (child RSS missing)"
    if available_bytes is None:
        return "TELEMETRY_UNAVAILABLE (host MemAvailable missing)"
    if int(rss_bytes) > int(rss_cap_bytes):
        return f"CHILD_RSS_CAP ({int(rss_bytes)} > {int(rss_cap_bytes)} B)"
    if int(available_bytes) < int(reserve_bytes):
        return f"RESERVE_BREACH ({int(available_bytes)} < {int(reserve_bytes)} B)"
    return None


def _kill_owned(proc: subprocess.Popen) -> None:
    """Kill ONLY the child process group this bench started."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


def run_child(
    argv: list[str],
    *,
    log_path: Path,
    timeout_s: float,
    rss_cap_bytes: int = MAX_CHILD_RSS_BYTES,
    reserve_bytes: int = MIN_RESERVE_BYTES,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
) -> dict:
    """Run one owned child, capture bounded output, sample RSS and host RAM."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        cwd=str(REPO_ROOT),
    )
    phases: list[dict] = []
    lines: list[str] = []
    reader_done = threading.Event()
    truncated = {"value": False, "bytes": 0}

    def _reader() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                now = time.monotonic()
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        payload = json.loads(line[len(PROGRESS_PREFIX):])
                        payload["_monotonic_s"] = float(now - started)
                        phases.append(payload)
                    except Exception:
                        pass
                encoded = line.encode("utf-8", "replace")
                if truncated["bytes"] + len(encoded) <= STDOUT_CAP_BYTES:
                    lines.append(line)
                    truncated["bytes"] += len(encoded)
                else:
                    truncated["value"] = True
        finally:
            reader_done.set()

    reader = threading.Thread(target=_reader, name="m5b-rss-reader", daemon=True)
    reader.start()

    samples = 0
    valid_rss_samples = 0
    valid_host_samples = 0
    valid_paired_samples = 0
    valid_assembly_samples = 0
    peak_rss = 0
    peak_virtual = 0
    min_available = None
    assembly_peak = None
    breach = None
    timed_out = False
    current_phase = "start"
    in_assembly = False
    assembly_start_seen = False
    assembly_end_seen = False
    seen_phase_count = 0
    telemetry_unavailable = 0
    telemetry_reasons: list[str] = []
    telemetry_fail_streak = 0
    phase_timeline: list[dict] = []
    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > float(timeout_s):
                timed_out = True
                _kill_owned(proc)
                break
            while seen_phase_count < len(phases):
                phase = phases[seen_phase_count]
                seen_phase_count += 1
                name = str(phase.get("phase", ""))
                current_phase = name
                phase_timeline.append(
                    {
                        "phase": name,
                        "monotonic_s": float(phase.get("_monotonic_s", 0.0)),
                        "unix_ns": int(phase.get("unix_ns", 0)),
                    }
                )
                if name == "assembly_start":
                    in_assembly = True
                    assembly_start_seen = True
                elif name == "assembly_end":
                    in_assembly = False
                    assembly_end_seen = True
                elif name in ("closed", "done"):
                    in_assembly = False
            mem = _read_proc_mem(proc.pid)
            host = _host_meminfo()
            available = host.get("mem_available_bytes")
            samples += 1
            rss_ok = bool(mem.get("available")) and isinstance(
                mem.get("rss_bytes"), int
            ) and int(mem["rss_bytes"]) > 0
            host_ok = isinstance(available, int) and int(available) > 0
            if rss_ok:
                valid_rss_samples += 1
                rss = int(mem["rss_bytes"])
                peak_rss = max(peak_rss, rss)
                peak_virtual = max(peak_virtual, int(mem["virtual_bytes"] or 0))
                if in_assembly:
                    valid_assembly_samples += 1
                    assembly_peak = rss if assembly_peak is None else max(assembly_peak, rss)
            if host_ok:
                valid_host_samples += 1
                min_available = (
                    int(available) if min_available is None else min(min_available, int(available))
                )
            if rss_ok and host_ok:
                valid_paired_samples += 1
            if not rss_ok or not host_ok:
                telemetry_unavailable += 1
                reason_parts = []
                if not rss_ok:
                    reason_parts.append("child RSS")
                if not host_ok:
                    reason_parts.append("host MemAvailable")
                telemetry_reasons.append("missing " + " and ".join(reason_parts))
                # Normal process-exit race: the child may disappear between
                # poll() and the /proc read. Tolerate a SHORT bounded streak so
                # an already-exiting child is not falsely failed.
                if proc.poll() is not None:
                    break
                telemetry_fail_streak += 1
                if telemetry_fail_streak >= 5:
                    breach = monitor_decision(
                        rss_bytes=(int(mem["rss_bytes"]) if rss_ok else None),
                        available_bytes=(int(available) if host_ok else None),
                        rss_cap_bytes=int(rss_cap_bytes),
                        reserve_bytes=int(reserve_bytes),
                    )
                    if breach is not None:
                        _kill_owned(proc)
                        break
                time.sleep(float(sample_interval_s))
                continue
            telemetry_fail_streak = 0
            breach = monitor_decision(
                rss_bytes=int(mem["rss_bytes"]),
                available_bytes=int(available),
                rss_cap_bytes=int(rss_cap_bytes),
                reserve_bytes=int(reserve_bytes),
            )
            if breach is not None:
                _kill_owned(proc)
                break
            time.sleep(float(sample_interval_s))
    finally:
        try:
            returncode = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            _kill_owned(proc)
            returncode = proc.wait(timeout=15)
        reader.join(timeout=10)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        elapsed = time.monotonic() - started
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.writelines(lines)
            if truncated["value"]:
                handle.write(
                    f"\n[M5B] stdout truncated at {STDOUT_CAP_BYTES} bytes\n"
                )
    return {
        "argv": list(argv),
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "guard_breach": breach,
        "elapsed_s": float(elapsed),
        "samples": int(samples),
        "valid_rss_samples": int(valid_rss_samples),
        "valid_host_samples": int(valid_host_samples),
        "valid_paired_samples": int(valid_paired_samples),
        "valid_assembly_samples": int(valid_assembly_samples),
        "telemetry_unavailable_count": int(telemetry_unavailable),
        "telemetry_unavailable_reasons": list(telemetry_reasons[:20]),
        "assembly_window_observed": bool(assembly_start_seen and assembly_end_seen),
        "sample_interval_s": float(sample_interval_s),
        "sampled_peak_rss_bytes": int(peak_rss),
        "sampled_peak_virtual_bytes": int(peak_virtual),
        "sampled_min_available_bytes": (None if min_available is None else int(min_available)),
        "sampled_assembly_peak_rss_bytes": (
            None if valid_assembly_samples == 0 else int(assembly_peak or 0)
        ),
        "phases": phases,
        "phase_timeline": phase_timeline,
        "log": str(log_path),
        "stdout_truncated": bool(truncated["value"]),
        "last_phase": current_phase,
    }


_REQUIRED_MEASURE = (
    "schema",
    "status",
    "qualified_measurement",
    "case",
    "platform",
    "provenance",
    "host",
    "baseline_imports",
    "preassembly",
    "assembly",
    "admission",
    "model",
    "stats",
    "stats_after_close",
    "counters",
    "digests",
    "geometry",
    "timeline",
    "checks",
)


def _finite_nonneg(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def classify_measurement(
    *,
    run: dict,
    result: dict | None,
    store_commit_before: str | None,
    store_commit_after: str | None,
    store_content_before: str | None = None,
    store_content_after: str | None = None,
    require_content_digest: bool = False,
    expected_case: dict | None = None,
) -> dict:
    """Fail-closed verdict for ONE measured child. JSON alone is never success.

    Precedence: process timeout/guard/nonzero -> FAIL. A REFUSED result is
    NOT_RUN only when its typed pre-payload evidence validates. Missing safety
    telemetry or too few samples -> NOT_RUN (nonqualified), never PASS.
    """
    process_failures: list[str] = []
    if run.get("timed_out"):
        process_failures.append("child timed out")
    if run.get("guard_breach"):
        process_failures.append(f"guard breach: {run['guard_breach']}")
    if int(run.get("returncode", -1)) != 0:
        process_failures.append(f"child returncode {run.get('returncode')}")
    if result is None:
        return {
            "status": "FAIL",
            "reasons": process_failures + ["result JSON missing or unreadable"],
            "result": None,
        }
    status_value = str(result.get("status", ""))
    if process_failures:
        return {"status": "FAIL", "reasons": process_failures, "result": result}
    if status_value == "REFUSED":
        valid, refusal_reasons = _validate_refusal(result)
        if valid:
            refusal_reasons = []
            if store_commit_before is not None and store_commit_after is not None:
                if store_commit_before != store_commit_after:
                    refusal_reasons.append(
                        "durable commit digest changed during a refused child"
                    )
            if require_content_digest:
                if not store_content_before or not store_content_after:
                    refusal_reasons.append("parent store content digest is missing")
                elif store_content_before != store_content_after:
                    refusal_reasons.append(
                        "parent store CONTENT digest changed during a refused child"
                    )
            elif (
                store_content_before is not None
                and store_content_after is not None
                and store_content_before != store_content_after
            ):
                refusal_reasons.append(
                    "parent store content digest changed during a refused child"
                )
            if refusal_reasons:
                return {
                    "status": "FAIL",
                    "reasons": refusal_reasons,
                    "result": result,
                }
            return {
                "status": "NOT_RUN",
                "reasons": ["typed pre-payload capacity refusal"],
                "result": result,
                "store_content_unchanged": (
                    None
                    if store_content_before is None or store_content_after is None
                    else store_content_before == store_content_after
                ),
            }
        return {
            "status": "FAIL",
            "reasons": [f"invalid/spoofed refusal: {reason}" for reason in refusal_reasons],
            "result": result,
        }

    reasons: list[str] = []
    missing = [key for key in _REQUIRED_MEASURE if key not in result]
    if missing:
        reasons.append(f"partial result: missing {missing}")
    if status_value != "PASS":
        reasons.append(f"result status {status_value!r} is not PASS")
    if not bool(result.get("qualified_measurement")):
        reasons.append("not a qualified measurement (explicit diagnostic budget)")

    assembly = result.get("assembly") or {}
    if assembly.get("completed") is not True:
        reasons.append("assembly did not complete")
    if assembly.get("progressed") is not True:
        reasons.append("assembly progress flag missing/false")

    checks = result.get("checks")
    if not isinstance(checks, dict) or not checks:
        reasons.append("checks are missing/empty (vacuous success refused)")
    else:
        missing_checks = [key for key in REQUIRED_CHECKS if key not in checks]
        if missing_checks:
            reasons.append(f"missing required checks: {missing_checks}")
        failed_checks = sorted(key for key, value in checks.items() if value is not True)
        if failed_checks:
            reasons.append(f"failed checks: {failed_checks}")

    counters = result.get("counters") or {}
    if int(counters.get("max_resident_cells", -1)) != 2:
        reasons.append("resident cap is not 2")
    peak_cells = int(counters.get("peak_live_cells", -1))
    if not (1 <= peak_cells <= 2):
        reasons.append("peak live cells is not within 1..2")
    max_live = counters.get("max_live_bytes")
    peak_mapped = counters.get("peak_live_mapped_bytes")
    if not isinstance(max_live, int) or max_live <= 0:
        reasons.append("live byte envelope is missing/non-positive")
    if not isinstance(peak_mapped, int) or peak_mapped <= 0 or (
        isinstance(max_live, int) and peak_mapped > max_live
    ):
        reasons.append("peak mapped bytes is missing/zero/exceeds the envelope")

    model = result.get("model") or {}
    admission = result.get("admission") or {}
    required = model.get("required_bytes")
    available = admission.get("available_bytes")
    if not _finite_nonneg(required) or float(required) <= 0:
        reasons.append("model required_bytes is missing/zero/non-finite")
    if not _finite_nonneg(available) or float(available) <= 0:
        reasons.append("admission available_bytes is missing/zero/non-finite")
    if _finite_nonneg(required) and _finite_nonneg(available) and not (
        float(required) <= float(available)
    ):
        reasons.append("model required_bytes exceeds available_bytes")
    if not bool(admission.get("fits")):
        reasons.append("model does not fit the observed admission budget")
    if str(admission.get("budget_source")) != "ram_contract.admit_budget_bytes":
        reasons.append("admission did not use the real ram_contract observation")

    stats_close = result.get("stats_after_close") or {}
    if stats_close.get("closed") is not True:
        reasons.append("bounded session is not closed")
    if int(stats_close.get("live_cells", -1)) != 0:
        reasons.append("live cells remain after close")
    if int(stats_close.get("live_mapped_bytes", -1)) != 0:
        reasons.append("live mapped bytes remain after close")

    digests = result.get("digests") or {}
    if store_commit_before is not None and store_commit_after is not None:
        if store_commit_before != store_commit_after:
            reasons.append("durable commit digest changed during measurement")
    if digests.get("store_commits_before") != digests.get("store_commits_after"):
        reasons.append("child observed a commit-digest change")
    if digests.get("store_tree_before") != digests.get("store_tree_after"):
        reasons.append("child observed a store metadata/tree change")
    if require_content_digest:
        if not store_content_before or not store_content_after:
            reasons.append("parent store content digest is missing")
        elif store_content_before != store_content_after:
            reasons.append("parent store CONTENT digest changed during measurement")
    elif (
        store_content_before is not None
        and store_content_after is not None
        and store_content_before != store_content_after
    ):
        reasons.append("parent store content digest changed")

    output = digests.get("output") or {}
    missing_planes = [name for name in OUTPUT_PLANES if name not in output]
    if missing_planes:
        reasons.append(f"output planes missing: {missing_planes}")
    if any(
        not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
        for entry in output.values()
    ):
        reasons.append("output digests missing/invalid")
    if output and digests.get("output") != digests.get("output_after_close"):
        reasons.append("returned arrays did not survive session close")

    timeline_ok, timeline_reasons = _timeline_ordered(result)
    if not timeline_ok:
        reasons.extend(timeline_reasons)

    baseline = result.get("baseline_imports") or {}
    pre = result.get("preassembly") or {}
    after = result.get("assembly") or {}
    for label, value in (
        ("baseline current RSS", baseline.get("current_rss_bytes")),
        ("preassembly current RSS", pre.get("current_rss_bytes")),
        ("post-assembly current RSS", after.get("post_current_rss_bytes")),
        ("post-close current RSS", after.get("post_close_current_rss_bytes")),
        ("elapsed_s", after.get("elapsed_s")),
        ("sampled peak RSS", run.get("sampled_peak_rss_bytes")),
    ):
        if not _finite_nonneg(value):
            reasons.append(f"{label} is not finite/non-negative")

    if expected_case is not None:
        case = result.get("case") or {}
        if [int(v) for v in case.get("canvas", [])] != list(expected_case["canvas"]):
            reasons.append("result canvas does not match the requested case")
        if [int(v) for v in case.get("grid", [])] != list(expected_case["grid"]):
            reasons.append("result grid does not match the requested case")
        if int(case.get("cells", -1)) != int(expected_case["cells"]):
            reasons.append("result cell count does not match the requested case")

    sampled_peak = run.get("sampled_peak_rss_bytes")
    if sampled_peak and int(sampled_peak) > MAX_CHILD_RSS_BYTES:
        reasons.append("sampled child RSS exceeded the per-child cap")

    # Safety/monitoring evidence is a QUALIFICATION prerequisite. Too little
    # evidence is NOT_RUN, never PASS; a process/guard failure above is FAIL.
    monitoring_reasons: list[str] = []
    for key in ("valid_rss_samples", "valid_host_samples", "valid_paired_samples"):
        if int(run.get(key, 0)) < MIN_VALID_SAMPLES:
            monitoring_reasons.append(
                f"{key}={run.get(key, 0)} < {MIN_VALID_SAMPLES}"
            )
    if not _finite_nonneg(sampled_peak) or int(sampled_peak or 0) <= 0:
        monitoring_reasons.append("sampled peak RSS is missing/zero")
    min_avail = run.get("sampled_min_available_bytes")
    if not isinstance(min_avail, int) or min_avail <= 0:
        monitoring_reasons.append("sampled host MemAvailable is missing/invalid")
    if run.get("assembly_window_observed") is not True:
        monitoring_reasons.append("assembly phase window was not observed")
    if int(run.get("valid_assembly_samples", 0)) < MIN_VALID_SAMPLES:
        monitoring_reasons.append(
            f"valid_assembly_samples={run.get('valid_assembly_samples', 0)} < {MIN_VALID_SAMPLES}"
        )
    assembly_peak = run.get("sampled_assembly_peak_rss_bytes")
    if not isinstance(assembly_peak, int) or assembly_peak <= 0:
        monitoring_reasons.append("assembly-window sampled peak is missing/zero")

    if reasons:
        return {
            "status": "FAIL",
            "reasons": reasons,
            "result": result,
            "store_content_unchanged": (
                None
                if store_content_before is None or store_content_after is None
                else store_content_before == store_content_after
            ),
        }
    if monitoring_reasons:
        return {
            "status": "NOT_RUN",
            "reasons": [
                "insufficient safety/monitoring evidence: " + "; ".join(monitoring_reasons)
            ],
            "result": result,
            "store_content_unchanged": True,
        }
    return {
        "status": "PASS",
        "reasons": [],
        "result": result,
        "store_content_unchanged": True,
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def process_failure_reasons(run: dict | None) -> list[str]:
    """Process-level failures that cannot be erased by an exit-0 JSON result."""
    if not isinstance(run, dict):
        return ["child run record missing"]
    reasons: list[str] = []
    if run.get("timed_out"):
        reasons.append("child timed out")
    if run.get("guard_breach"):
        reasons.append(f"guard breach: {run['guard_breach']}")
    if int(run.get("returncode", -1)) != 0:
        reasons.append(f"child returncode {run.get('returncode')}")
    return reasons


def _git_state() -> dict:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT)] + args,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return (out.stdout or "").strip()
        except Exception:
            return ""

    return {
        "head": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_porcelain": _run(["status", "--porcelain"]),
    }


def cmd_matrix(args: argparse.Namespace) -> int:
    base = _ensure_under_workspace(Path(args.base))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = base / f"run-{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"run-{stamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + float(MAX_TOTAL_SECONDS)

    fs_base = assert_real_disk(run_dir)
    avail_start = _fresh_available_bytes()
    if avail_start < MIN_RESERVE_BYTES + 128 * 2 ** 20:
        print(
            json.dumps(
                {
                    "schema": SCHEMA_MATRIX,
                    "status": "NOT_RUN",
                    "reason": (
                        "host MemAvailable is below the 1 GiB reserve plus a "
                        "minimal working margin"
                    ),
                    "mem_available_bytes": avail_start,
                },
                indent=2,
            )
        )
        return 2

    names = [part.strip() for part in str(args.cases).split(",") if part.strip()]
    all_cases = {case["name"]: case for case in DEFAULT_CASES}
    if args.include_8x8 and "c1024-g8x8" not in names:
        names.append("c1024-g8x8")
    if not names:
        names = ["c512-g4x4", "c1024-g4x4"]
    specs = []
    for name in names:
        if name not in all_cases:
            raise SystemExit(f"unknown case {name!r}; known: {sorted(all_cases)}")
        specs.append(validate_case(all_cases[name]))

    provenance = {
        "schema": SCHEMA_MATRIX,
        "started_unix_ns": time.time_ns(),
        "run_dir": str(run_dir),
        "workspace_root": str(WORKSPACE_ROOT),
        "git_before": _git_state(),
        "fs": fs_base,
        "platform": {
            "system": _platform.system(),
            "release": _platform.release(),
            "machine": _platform.machine(),
            "python": _platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "budgets": {
            "max_canvas": MAX_CANVAS,
            "max_cells": MAX_CELLS,
            "max_child_seconds": MAX_CHILD_SECONDS,
            "max_total_seconds": MAX_TOTAL_SECONDS,
            "max_disk_bytes": MAX_DISK_BYTES,
            "min_reserve_bytes": MIN_RESERVE_BYTES,
            "max_child_rss_bytes": MAX_CHILD_RSS_BYTES,
        },
        "mem_available_start_bytes": int(avail_start),
        "limits": [
            "synthetic deterministic producer + real durable store + real "
            "guarded assembly consumer",
            "NOT the full FITS product, NOT physical M3 waiting/recovery, "
            "NOT Windows/NTFS, NOT a BLAS-scale universal bound",
            "stored identity execution_context is a synthetic 4 MiB fixture",
        ],
    }
    _write_result(run_dir / "provenance.json", provenance)

    case_reports = []
    overall = "PASS"
    stop_reason = None
    safety_stop = False

    def _degrade(status: str) -> None:
        nonlocal overall
        if status == "FAIL":
            overall = "FAIL"
        elif status == "NOT_RUN" and overall == "PASS":
            overall = "PARTIAL"

    for spec in specs:
        case_dir = run_dir / spec["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        store_dir = case_dir / "store"
        report = {
            "schema": SCHEMA_CASE,
            "case": {
                "name": spec["name"],
                "canvas": list(spec["canvas"]),
                "grid": list(spec["grid"]),
                "halo": int(spec["halo"]),
                "cells": int(spec["cells"]),
            },
            "status": "NOT_RUN",
            "reasons": [],
        }
        try:
            remaining = deadline - time.monotonic()
            if remaining < 10:
                report["reasons"].append(
                    "whole-benchmark deadline reached before seeding"
                )
                stop_reason = "whole-benchmark deadline reached; not run"
                _degrade("NOT_RUN")
                continue
            if stop_reason:
                report["reasons"].append(stop_reason)
                _degrade("NOT_RUN")
                continue

            assert_real_disk(case_dir)
            if store_dir.exists():
                raise RuntimeError(f"store directory already exists: {store_dir}")
            timeout = min(MAX_CHILD_SECONDS, max(1.0, remaining))
            seed_run = run_child(
                _case_argv(
                    "seed",
                    spec,
                    store_dir,
                    result=str(case_dir / "seed.json"),
                ),
                log_path=case_dir / "seed.log",
                timeout_s=timeout,
            )
            seed = _read_json(case_dir / "seed.json")
            report["seed_run"] = seed_run
            seed_failures = process_failure_reasons(seed_run)
            if seed_failures or seed is None or seed.get("status") != "SEEDED":
                report["status"] = "FAIL"
                report["reasons"].extend(
                    seed_failures or ["seed child did not produce a valid SEEDED result"]
                )
                stop_reason = "stopped after a failed seed"
                safety_stop = True
                _degrade("FAIL")
                continue
            report["seed"] = {
                "store": seed.get("store"),
                "producer": seed.get("producer"),
            }
            store_bytes = int(dir_bytes(store_dir))
            report["store_bytes"] = store_bytes
            if store_bytes > MAX_CASE_DISK_BYTES:
                raise RuntimeError(
                    f"case store {store_bytes} B exceeds {MAX_CASE_DISK_BYTES} B"
                )
            if dir_bytes(run_dir) > MAX_DISK_BYTES:
                raise RuntimeError(
                    f"artifact set exceeds total disk budget {MAX_DISK_BYTES} B"
                )

            predict_run = run_child(
                _case_argv(
                    "predict",
                    spec,
                    store_dir,
                    result=str(case_dir / "predict.json"),
                ),
                log_path=case_dir / "predict.log",
                timeout_s=min(MAX_CHILD_SECONDS, max(1.0, deadline - time.monotonic())),
            )
            predict = _read_json(case_dir / "predict.json")
            report["predict_run"] = predict_run
            report["predicted"] = predict
            predict_failures = process_failure_reasons(predict_run)
            if predict_failures or predict is None:
                report["status"] = "FAIL"
                report["reasons"].extend(
                    predict_failures or ["predict child produced no result JSON"]
                )
                stop_reason = "stopped after a failed prediction"
                safety_stop = True
                _degrade("FAIL")
                continue
            if str(predict.get("status")) == "PREDICTED_REFUSED":
                report["status"] = "NOT_RUN"
                report["reasons"].append(
                    "metadata-only prediction refused by the real admission budget"
                )
                stop_reason = "stopped: predicted model did not fit the real budget"
                _degrade("NOT_RUN")
                shutil.rmtree(store_dir, ignore_errors=True)
                report["store_removed"] = True
                continue
            required = int((predict.get("plan") or {}).get("required_bytes", 0))
            avail_now = _fresh_available_bytes()
            headroom = int(avail_now) - MIN_RESERVE_BYTES
            report["headroom_before_measure_bytes"] = headroom
            if required <= 0 or headroom <= 0 or required > headroom // 2:
                report["status"] = "NOT_RUN"
                report["reasons"].append(
                    "insufficient real headroom for the predicted model "
                    f"(required={required} B, headroom={headroom} B)"
                )
                stop_reason = "stopped: insufficient real headroom for larger cases"
                _degrade("NOT_RUN")
                shutil.rmtree(store_dir, ignore_errors=True)
                report["store_removed"] = True
                continue

            commit_before = commit_digest(store_dir)
            content_before_time = time.monotonic()
            content_before = store_content_digest(store_dir)
            timeout = min(MAX_CHILD_SECONDS, max(1.0, deadline - time.monotonic()))
            measure_run = run_child(
                _case_argv(
                    "measure",
                    spec,
                    store_dir,
                    result=str(case_dir / "measure.json"),
                ),
                log_path=case_dir / "measure.log",
                timeout_s=timeout,
            )
            content_after_time = time.monotonic()
            commit_after = commit_digest(store_dir)
            content_after = store_content_digest(store_dir)
            result = _read_json(case_dir / "measure.json")
            verdict = classify_measurement(
                run=measure_run,
                result=result,
                store_commit_before=commit_before,
                store_commit_after=commit_after,
                store_content_before=content_before,
                store_content_after=content_after,
                require_content_digest=True,
                expected_case=spec,
            )
            report["measure_run"] = measure_run
            report["verdict"] = verdict
            report["store_commit_digest_before"] = commit_before
            report["store_commit_digest_after"] = commit_after
            report["store_content_digest_before"] = content_before
            report["store_content_digest_after"] = content_after
            report["snapshot_timeline"] = {
                "content_before_monotonic_s": content_before_time,
                "measure_spawn_to_exit_monotonic_s": measure_run.get("elapsed_s"),
                "content_after_monotonic_s": content_after_time,
                "scope": (
                    "parent stream-hashed every store file before spawn and "
                    "after child exit; hashing pages are outside the measured "
                    "child RSS and warm the page cache"
                ),
            }
            report["status"] = verdict["status"]
            if verdict["reasons"]:
                report["reasons"].extend(verdict["reasons"])
            if verdict["status"] == "PASS":
                result = verdict["result"] or {}
                sampled_peak = measure_run.get("sampled_peak_rss_bytes")
                assembly_peak = measure_run.get("sampled_assembly_peak_rss_bytes")
                pre_rss = (result.get("preassembly") or {}).get(
                    "current_rss_bytes"
                )
                report["observed"] = {
                    "sampled_peak_rss_bytes": sampled_peak,
                    "sampled_peak_scope": (
                        "whole child lifetime at 20 ms; includes post-assembly "
                        "digest and close work, so it is NOT the assembly peak"
                    ),
                    "sampled_assembly_peak_rss_bytes": assembly_peak,
                    "sampled_assembly_peak_scope": (
                        "between assembly_start and assembly_end progress phases"
                    ),
                    "sampled_peak_virtual_bytes": measure_run.get(
                        "sampled_peak_virtual_bytes"
                    ),
                    "sampled_min_available_bytes": measure_run.get(
                        "sampled_min_available_bytes"
                    ),
                    "sampled_peak_delta_vs_preassembly_bytes": (
                        None
                        if sampled_peak is None or pre_rss is None
                        else int(sampled_peak) - int(pre_rss)
                    ),
                    "sampled_assembly_delta_vs_preassembly_bytes": (
                        None
                        if assembly_peak is None or pre_rss is None
                        else int(assembly_peak) - int(pre_rss)
                    ),
                    "valid_rss_samples": measure_run.get("valid_rss_samples"),
                    "valid_host_samples": measure_run.get("valid_host_samples"),
                    "valid_paired_samples": measure_run.get("valid_paired_samples"),
                    "valid_assembly_samples": measure_run.get("valid_assembly_samples"),
                    "telemetry_unavailable_count": measure_run.get(
                        "telemetry_unavailable_count"
                    ),
                    "baseline_current_rss_bytes": (
                        result.get("baseline_imports") or {}
                    ).get("current_rss_bytes"),
                    "preassembly_current_rss_bytes": pre_rss,
                    "post_assembly_current_rss_bytes": (
                        result.get("assembly") or {}
                    ).get("post_current_rss_bytes"),
                    "post_assembly_hwm_rss_bytes": (
                        result.get("assembly") or {}
                    ).get("post_hwm_rss_bytes"),
                    "post_close_current_rss_bytes": (
                        result.get("assembly") or {}
                    ).get("post_close_current_rss_bytes"),
                    "hwm_minus_baseline_hwm_bytes": (
                        result.get("assembly") or {}
                    ).get("hwm_minus_baseline_hwm_bytes"),
                    "virtual_bytes_before": (
                        result.get("preassembly") or {}
                    ).get("virtual_bytes"),
                    "virtual_bytes_after": (
                        result.get("assembly") or {}
                    ).get("post_virtual_bytes"),
                    "elapsed_s": (result.get("assembly") or {}).get("elapsed_s"),
                    "cpu_total_s": (result.get("assembly") or {}).get("cpu_total_s"),
                    "model_required_bytes": (result.get("model") or {}).get(
                        "required_bytes"
                    ),
                    "model_breakdown": {
                        "canvas_bytes": (result.get("model") or {}).get(
                            "canvas_bytes"
                        ),
                        "resident_mapped_bytes": (result.get("model") or {}).get(
                            "resident_mapped_bytes"
                        ),
                        "transient_bytes": (result.get("model") or {}).get(
                            "transient_bytes"
                        ),
                        "dense_solver_bytes": (result.get("model") or {}).get(
                            "dense_solver_bytes"
                        ),
                    },
                    "counters": result.get("counters"),
                    "timeline": result.get("timeline"),
                    "store_content_digest_unchanged": (
                        content_before == content_after
                    ),
                }
            else:
                _degrade(verdict["status"])
                if verdict["status"] == "FAIL":
                    stop_reason = "stopped after a failed measured case"
                    safety_stop = True
                else:
                    stop_reason = "stopped: a measured case was honestly NOT_RUN"
                    safety_stop = True
                shutil.rmtree(store_dir, ignore_errors=True)
                report["store_removed"] = True
        except Exception as exc:
            report["status"] = "FAIL"
            report["reasons"].append(f"{type(exc).__name__}: {exc}")
            stop_reason = "stopped after a failed case"
            safety_stop = True
            _degrade("FAIL")
            shutil.rmtree(store_dir, ignore_errors=True)
            report["store_removed"] = True
        finally:
            report["disk_bytes_after_case"] = int(dir_bytes(run_dir))
            _write_result(case_dir / "case.json", report)
            case_reports.append(report)

    parity_report = {
        "schema": SCHEMA_CASE,
        "case": PARITY_CASE,
        "status": "NOT_RUN",
        "reasons": [],
    }
    remaining = deadline - time.monotonic()
    if not args.parity:
        parity_report["reasons"].append("parity disabled")
        if overall == "PASS":
            overall = "PARTIAL"
    elif overall == "FAIL":
        parity_report["reasons"].append(
            "parity NOT launched after a prior failure; FAIL preserved"
        )
    elif safety_stop:
        parity_report["reasons"].append(
            "parity NOT launched after a safety/evidence stop"
        )
        if overall == "PASS":
            overall = "PARTIAL"
    elif remaining <= 15:
        parity_report["reasons"].append(
            "parity skipped: whole-benchmark deadline reached"
        )
        if overall == "PASS":
            overall = "PARTIAL"
    else:
        parity_dir = run_dir / PARITY_CASE["name"]
        parity_dir.mkdir(parents=True, exist_ok=True)
        store_dir = parity_dir / "store"
        parity_report["parity_launched"] = False
        try:
            assert_real_disk(parity_dir)
            spec = validate_case(PARITY_CASE)
            remaining = deadline - time.monotonic()
            if remaining <= 15:
                parity_report["status"] = "NOT_RUN"
                parity_report["reasons"].append(
                    "parity NOT launched: whole-benchmark deadline reached before seed"
                )
                if overall == "PASS":
                    overall = "PARTIAL"
            else:
                seed_run = run_child(
                    _case_argv(
                        "seed", spec, store_dir, result=str(parity_dir / "seed.json")
                    ),
                    log_path=parity_dir / "seed.log",
                    timeout_s=min(MAX_CHILD_SECONDS, max(1.0, remaining)),
                )
                seed = _read_json(parity_dir / "seed.json")
                parity_report["seed_run"] = seed_run
                parity_report["seed"] = seed
                seed_failures = process_failure_reasons(seed_run)
                if seed_failures:
                    parity_report["status"] = "FAIL"
                    parity_report["reasons"].extend(seed_failures)
                    parity_report["reasons"].append(
                        "parity seed process failed; parity child NOT launched"
                    )
                    overall = "FAIL"
                elif seed is None or seed.get("status") != "SEEDED":
                    parity_report["status"] = "FAIL"
                    parity_report["reasons"].append(
                        "parity seed produced missing/invalid SEEDED result; "
                        "parity child NOT launched"
                    )
                    overall = "FAIL"
                else:
                    remaining_after_seed = deadline - time.monotonic()
                    parity_report["remaining_after_seed_s"] = float(
                        remaining_after_seed
                    )
                    if remaining_after_seed <= 10:
                        parity_report["status"] = "NOT_RUN"
                        parity_report["reasons"].append(
                            "whole-benchmark deadline exhausted after parity seed; "
                            "parity child NOT launched"
                        )
                        if overall == "PASS":
                            overall = "PARTIAL"
                    else:
                        parity_report["parity_launched"] = True
                        parity_run = run_child(
                            _case_argv(
                                "parity",
                                spec,
                                store_dir,
                                result=str(parity_dir / "parity.json"),
                            ),
                            log_path=parity_dir / "parity.log",
                            timeout_s=min(
                                MAX_CHILD_SECONDS,
                                max(1.0, deadline - time.monotonic()),
                            ),
                        )
                        parity = _read_json(parity_dir / "parity.json")
                        parity_report["parity_run"] = parity_run
                        parity_report["parity"] = parity
                        parity_failures = process_failure_reasons(parity_run)
                        if parity_failures:
                            parity_report["status"] = "FAIL"
                            parity_report["reasons"].extend(parity_failures)
                            overall = "FAIL"
                        elif (
                            parity is not None
                            and parity.get("status") == "PARITY_PASS"
                            and not parity.get("mismatches")
                        ):
                            parity_report["status"] = "PASS"
                        else:
                            parity_report["status"] = "FAIL"
                            parity_report["reasons"].append(
                                "parity child did not return a valid PARITY_PASS result"
                            )
                            overall = "FAIL"
        except Exception as exc:
            parity_report["status"] = "FAIL"
            parity_report["reasons"].append(f"{type(exc).__name__}: {exc}")
            overall = "FAIL"
        finally:
            _write_result(parity_dir / "case.json", parity_report)
    _write_result(run_dir / "parity-case.json", parity_report)
    case_reports.append(parity_report)

    # Independent end-state checks.
    disk_total = int(dir_bytes(run_dir))
    if disk_total > MAX_DISK_BYTES:
        overall = "FAIL"
    git_after = _git_state()
    summary = {
        "schema": SCHEMA_MATRIX,
        "status": overall,
        "run_dir": str(run_dir),
        "started_unix_ns": provenance["started_unix_ns"],
        "finished_unix_ns": time.time_ns(),
        "elapsed_s": float(time.monotonic() - started),
        "git_before": provenance["git_before"],
        "git_after": git_after,
        "repo_unchanged": git_after["head"] == provenance["git_before"]["head"],
        "fs": fs_base,
        "disk_bytes_total": disk_total,
        "disk_budget_bytes": MAX_DISK_BYTES,
        "cases": case_reports,
        "budgets": provenance["budgets"],
        "limits": provenance["limits"],
        "note": (
            "PASS means every selected synthetic case was admitted with a fresh "
            "real budget, observed under the reserve/RSS caps, kept cap=2 and "
            "stable commits, and parity vs a separate dense reference passed. "
            "It does not qualify the full product, Windows, M3 pressure or "
            "BLAS-scale canvases."
        ),
    }
    _write_result(run_dir / "matrix.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if overall == "PASS" else (2 if overall in ("PARTIAL", "NOT_RUN") else 1)


# ---------------------------------------------------------------------------
# Light self-test verbs (guard tests need real processes, not benchmarks)
# ---------------------------------------------------------------------------


def cmd_selftest_fail(args: argparse.Namespace) -> int:
    _write_result(
        Path(args.result),
        {"schema": SCHEMA_MEASURE, "status": "PASS", "partial": True},
    )
    print(MARKER_PREFIX + json.dumps({"status": "PASS"}))
    return 2


def cmd_selftest_sleep(args: argparse.Namespace) -> int:
    time.sleep(float(args.seconds))
    return 0


def cmd_selftest_grow(args: argparse.Namespace) -> int:
    size = int(args.bytes)
    payload = bytearray(size)
    for offset in range(0, size, 4096):
        payload[offset] = 1
    print(f"grown {size} B", flush=True)
    time.sleep(float(args.seconds))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="verb", required=True)

    def _case_args(p: argparse.ArgumentParser, *, result: bool = True) -> None:
        p.add_argument("--root", required=True, help="durable store root")
        p.add_argument("--canvas", required=True, help="HxW")
        p.add_argument("--grid", required=True, help="NxM cells")
        p.add_argument("--halo", type=int, default=8)
        if result:
            p.add_argument("--result", required=True, help="JSON result path")

    p_seed = sub.add_parser("seed")
    _case_args(p_seed)
    p_seed.set_defaults(func=cmd_seed)

    p_predict = sub.add_parser("predict")
    _case_args(p_predict)
    p_predict.set_defaults(func=cmd_predict)

    p_measure = sub.add_parser("measure")
    _case_args(p_measure)
    p_measure.add_argument(
        "--available-bytes",
        type=int,
        default=None,
        help="DIAGNOSTIC ONLY: explicit caller budget (never used by the matrix)",
    )
    p_measure.add_argument(
        "--inject-error",
        choices=("none", "generic"),
        default="none",
        help="TEST ONLY: raise a generic exception before guarded assembly",
    )
    p_measure.add_argument(
        "--inject-capacity-after-load",
        action="store_true",
        help="TEST ONLY: load one payload then raise AssemblyCapacityError",
    )
    p_measure.set_defaults(func=cmd_measure)

    p_parity = sub.add_parser("parity")
    _case_args(p_parity)
    p_parity.set_defaults(func=cmd_parity)

    p_matrix = sub.add_parser("matrix")
    p_matrix.add_argument("--base", default=str(DEFAULT_BASE))
    p_matrix.add_argument(
        "--cases",
        default="c512-g4x4,c1024-g4x4",
        help="comma-separated case names",
    )
    p_matrix.add_argument("--include-8x8", action="store_true")
    p_matrix.add_argument("--parity", action="store_true", default=True)
    p_matrix.add_argument("--no-parity", dest="parity", action="store_false")
    p_matrix.set_defaults(func=cmd_matrix)

    p_fail = sub.add_parser("selftest-fail")
    p_fail.add_argument("--result", required=True)
    p_fail.set_defaults(func=cmd_selftest_fail)

    p_sleep = sub.add_parser("selftest-sleep")
    p_sleep.add_argument("--seconds", type=float, default=5.0)
    p_sleep.set_defaults(func=cmd_selftest_sleep)

    p_grow = sub.add_parser("selftest-grow")
    p_grow.add_argument("--bytes", type=int, required=True)
    p_grow.add_argument("--seconds", type=float, default=2.0)
    p_grow.set_defaults(func=cmd_selftest_grow)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
