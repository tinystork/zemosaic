#!/usr/bin/env python3
"""ZM-ZEGRID M3 — owned Linux pressure runner (reference + gated experiment).

Two bricks live here:

``reference``
    Runs the REAL ZeGrid product on the deterministic frozen corpus in a fresh
    child process with NO pressure, either ``dense`` or ``bounded``. The bounded
    reference is the provenance of ``X_b`` (the bounded plan's
    ``aggregate_bytes`` actually selected by the engine). The parent samples host
    ``MemAvailable`` and the product tree RSS, enforces the hard ceilings, and
    classifies honestly (``NOT_RUN`` on headroom exhaustion, never a forced
    completion).

``experiment``
    The physical-pressure scenario. It is GATED OFF for this mission: without
    ``--i-am-reviewed`` *and* ``M3_PRESSURE_EXPERIMENT_REVIEWED=1`` it refuses
    before spawning anything and exits ``NOT_RUN``. The supervision/guard/IPC
    machinery exists and is unit-tested against simulated children, but this
    mission never applies physical pressure.

Hard ceilings (never widened):
    * whole pressure process RSS ``<= 512 MiB`` (payload + interpreter overhead),
    * live host ``MemAvailable`` never below the 1 GiB reserve,
    * product tree RSS never above the calibrated reference ceiling,
    * ack deadline ``<= 2 s``; guard sampling ``0.25 s``,
    * total lifetime ``<= 900 s``; pressure phase ``<= 300 s``.

Exit codes: 0 PASS/OK, 2 NOT_RUN, 3 FAIL, 4 usage/tool failure.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

MIB = 1 << 20
GIB = 1 << 30

SCHEMA = "zemosaic-zegrid-m3-pressure-runner-v1"

HARD_PRESSURE_CAP_BYTES = 512 * MIB
MIN_RESERVE_BYTES = 1 * GIB
CONTRACT_MAX_ACK_DEADLINE_S = 2.0
GUARD_SAMPLE_INTERVAL_S = 0.25
MAX_LIFETIME_S = 900.0
MAX_PRESSURE_PHASE_S = 300.0
PRODUCT_CEILING_ALLOWANCE_BYTES = 64 * MIB
PRESSURE_STEP_BYTES = 64 * MIB
DEFAULT_PAYLOAD_BYTES = 480 * MIB

EXIT_PASS = 0
EXIT_NOT_RUN = 2
EXIT_FAIL = 3
EXIT_USAGE = 4

CORPUS_FRAMES = 8
CORPUS_SIZE = 512
CORPUS_SHIFTS = (
    (0, 0), (96, 96), (192, 192), (288, 288),
    (0, 288), (288, 0), (192, 96), (96, 192),
)
LAYOUT = "4x4"
HALO_PX = 64


# ---------------------------------------------------------------------------
# Host observation helpers (no cgroups, no sysctl, no cache drops)
# ---------------------------------------------------------------------------

def read_mem_available_bytes(meminfo_path="/proc/meminfo"):
    with open(meminfo_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                _, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    break
                amount = int(parts[0])
                unit = parts[1].lower() if len(parts) > 1 else "kb"
                if unit not in ("kb", "kib"):
                    raise ValueError(f"unexpected MemAvailable unit {unit!r}")
                return amount * 1024
    raise ValueError("MemAvailable missing from meminfo")


def _read_proc_status_kb(pid, field):
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def read_process_rss_bytes(pid):
    return _read_proc_status_kb(pid, "VmRSS")


def read_children(pid):
    try:
        with open(f"/proc/{int(pid)}/task/{int(pid)}/children", "r") as handle:
            return [int(tok) for tok in handle.read().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def read_tree_rss_bytes(pid, *, max_depth=6):
    """Sum VmRSS over the owned tree by walking ``children`` (bounded depth)."""
    total = 0
    seen = set()
    frontier = [(int(pid), 0)]
    while frontier:
        current, depth = frontier.pop()
        if current in seen or depth > max_depth:
            continue
        seen.add(current)
        rss = read_process_rss_bytes(current)
        if rss:
            total += rss
        if depth < max_depth:
            frontier.extend((child, depth + 1) for child in read_children(current))
    return total


def _now():
    return time.monotonic()


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.errors = 0

    def emit(self, kind, **fields):
        record = {"kind": str(kind), "monotonic": _now()}
        record.update(fields)
        line = (json.dumps(record, sort_keys=True, default=str) + "\n").encode()
        try:
            with self._lock:
                with self.path.open("ab") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
        except Exception:  # noqa: BLE001 - telemetry is never fatal
            self.errors += 1


# ---------------------------------------------------------------------------
# Guard / supervision (pure enough to simulate with fake children)
# ---------------------------------------------------------------------------

class GuardDecision:
    def __init__(self, ok, code=None, detail=""):
        self.ok = bool(ok)
        self.code = code
        self.detail = str(detail)

    def __bool__(self):
        return self.ok

    def as_dict(self):
        return {"ok": self.ok, "code": self.code, "detail": self.detail}


OK = GuardDecision(True)

#: Guard order is fixed and explicit: the pressure process is always the FIRST
#: thing killed. The product is only touched after a cooperative stop.
GUARD_ORDER = (
    "1_pressure_rss_or_reserve_breach__kill_owned_pressure_pid",
    "2_wait_pressure_exit__rss_zero",
    "3_cooperative_product_stop",
    "4_last_resort_owned_tree_kill__classify_FAIL",
)


def evaluate_guard(
    *,
    pressure_rss_bytes,
    mem_available_bytes,
    pressure_cap_bytes=HARD_PRESSURE_CAP_BYTES,
    reserve_bytes=MIN_RESERVE_BYTES,
    product_tree_rss_bytes=None,
    product_ceiling_bytes=None,
):
    """Pure guard evaluation. Missing telemetry is NEVER treated as healthy."""
    if mem_available_bytes is None:
        return GuardDecision(False, "MISSING_TELEMETRY", "MemAvailable unavailable")
    if reserve_bytes < MIN_RESERVE_BYTES:
        return GuardDecision(
            False, "RESERVE_BELOW_CONTRACT",
            f"reserve {reserve_bytes} < {MIN_RESERVE_BYTES}",
        )
    if pressure_cap_bytes > HARD_PRESSURE_CAP_BYTES:
        return GuardDecision(
            False, "CAP_ABOVE_CONTRACT",
            f"cap {pressure_cap_bytes} > {HARD_PRESSURE_CAP_BYTES}",
        )
    if pressure_rss_bytes is None:
        return GuardDecision(
            False, "MISSING_TELEMETRY", "pressure RSS unavailable"
        )
    if int(pressure_rss_bytes) > int(pressure_cap_bytes):
        return GuardDecision(
            False, "PRESSURE_RSS_BREACH",
            f"{pressure_rss_bytes} > {pressure_cap_bytes}",
        )
    if int(mem_available_bytes) < int(reserve_bytes):
        return GuardDecision(
            False, "RESERVE_BREACH",
            f"MemAvailable {mem_available_bytes} < {reserve_bytes}",
        )
    if product_ceiling_bytes is not None:
        if product_tree_rss_bytes is None:
            return GuardDecision(
                False, "MISSING_TELEMETRY", "product tree RSS unavailable"
            )
        if int(product_tree_rss_bytes) > int(product_ceiling_bytes):
            return GuardDecision(
                False, "PRODUCT_CEILING_BREACH",
                f"{product_tree_rss_bytes} > {product_ceiling_bytes}",
            )
    return OK


def calibrate_product_ceiling(reference_peak_tree_rss_bytes):
    if reference_peak_tree_rss_bytes is None:
        raise ValueError("a reference peak tree RSS is required for a ceiling")
    return int(reference_peak_tree_rss_bytes) + PRODUCT_CEILING_ALLOWANCE_BYTES


class PressureSupervisor:
    """Parent-owned supervision of ONE owned pressure child.

    All host reads, the clock and the child handle are injectable so the guard
    order and classification can be tested with simulated children — no
    physical pressure is required (or permitted) in this mission.
    """

    def __init__(
        self,
        *,
        pressure_pid,
        mem_available_fn=read_mem_available_bytes,
        pressure_rss_fn=read_process_rss_bytes,
        product_tree_rss_fn=read_tree_rss_bytes,
        clock=_now,
        event_sink=None,
        pressure_cap_bytes=HARD_PRESSURE_CAP_BYTES,
        reserve_bytes=MIN_RESERVE_BYTES,
        product_ceiling_bytes=None,
        kill_fn=None,
        sample_interval_s=GUARD_SAMPLE_INTERVAL_S,
    ):
        self.pressure_pid = int(pressure_pid)
        self.mem_available_fn = mem_available_fn
        self.pressure_rss_fn = pressure_rss_fn
        self.product_tree_rss_fn = product_tree_rss_fn
        self.clock = clock
        self.events = event_sink
        self.pressure_cap_bytes = int(pressure_cap_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.product_ceiling_bytes = product_ceiling_bytes
        self.sample_interval_s = float(sample_interval_s)
        self._kill = kill_fn or _default_kill
        self.peak_pressure_rss = 0
        self.release_reason = None

    def sample_once(self, *, product_pid=None):
        mem = self.mem_available_fn()
        rss = self.pressure_rss_fn(self.pressure_pid)
        if rss is not None:
            self.peak_pressure_rss = max(self.peak_pressure_rss, int(rss))
        tree = (
            None
            if product_pid is None
            else self.product_tree_rss_fn(product_pid)
        )
        decision = evaluate_guard(
            pressure_rss_bytes=rss,
            mem_available_bytes=mem,
            pressure_cap_bytes=self.pressure_cap_bytes,
            reserve_bytes=self.reserve_bytes,
            product_tree_rss_bytes=tree,
            product_ceiling_bytes=self.product_ceiling_bytes,
        )
        if self.events is not None:
            self.events.emit(
                "guard_sample",
                mem_available_bytes=mem,
                pressure_pid=self.pressure_pid,
                pressure_rss_bytes=rss,
                product_pid=product_pid,
                product_tree_rss_bytes=tree,
                product_ceiling_bytes=self.product_ceiling_bytes,
                decision=decision.as_dict(),
            )
        return decision

    def release_and_wait(self, *, timeout_s=5.0):
        """Step 1+2: kill ONLY the owned pressure PID, wait for RSS to reach 0."""
        self.release_reason = self.release_reason or "guard_release"
        self._kill(self.pressure_pid)
        deadline = self.clock() + float(timeout_s)
        while self.clock() < deadline:
            rss = self.pressure_rss_fn(self.pressure_pid)
            if rss is None or int(rss) == 0:
                if self.events is not None:
                    self.events.emit(
                        "pressure_released",
                        pressure_pid=self.pressure_pid,
                        reason=self.release_reason,
                        final_rss_bytes=rss,
                    )
                return True
            time.sleep(min(self.sample_interval_s, 0.05))
        return False


def _default_kill(pid):
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False


def _kill_then_kill(pid, *, grace_s=2.0, clock=_now):
    """SIGTERM then SIGKILL, only the owned PID."""
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = clock() + float(grace_s)
    while clock() < deadline:
        if read_process_rss_bytes(pid) is None:
            return True
        time.sleep(0.05)
    try:
        os.kill(int(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


# ---------------------------------------------------------------------------
# Pressure child (allocator) — implemented, never executed in this mission
# ---------------------------------------------------------------------------

def pressure_child_main(spec_path):
    """Owned child: anonymous mmap + page touch in bounded steps.

    Writes ONE readiness line to stdout, then allocates ``step_bytes`` per
    command read from stdin (``+``/``-``/``q``). Never forks, never changes
    cgroup/sysctl/swap, never touches another process.
    """
    import mmap

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    step = int(spec["step_bytes"])
    cap = int(spec["cap_bytes"])
    touched = 0
    blocks = []

    def grow():
        nonlocal touched
        if touched + step > cap:
            return False
        block = mmap.mmap(-1, step)
        block.write(b"\x01" * step)
        blocks.append(block)
        touched += step
        return True

    def shrink():
        nonlocal touched
        if not blocks:
            return False
        block = blocks.pop()
        touched -= len(block)
        block.close()
        return True

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(
        json.dumps(
            {"ready": True, "pid": os.getpid(), "step_bytes": step, "cap_bytes": cap}
        ),
        flush=True,
    )
    for line in sys.stdin:
        cmd = line.strip()
        if cmd in ("+", "grow"):
            ok = grow()
        elif cmd in ("-", "shrink"):
            ok = shrink()
        elif cmd in ("q", "quit"):
            break
        else:
            ok = False
        print(
            json.dumps({"ok": ok, "touched_bytes": touched, "rss_bytes": read_process_rss_bytes(os.getpid())}),
            flush=True,
        )
    for block in blocks:
        block.close()
    return 0


def _spawn_pressure_child(*, step_bytes, cap_bytes, workdir):
    spec_path = Path(workdir) / "pressure-child-spec.json"
    spec_path.write_text(
        json.dumps({"step_bytes": int(step_bytes), "cap_bytes": int(cap_bytes)}),
        encoding="utf-8",
    )
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_pressure-child", str(spec_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


# ---------------------------------------------------------------------------
# One-shot bounded IPC
# ---------------------------------------------------------------------------

class HookServer:
    """Parent side of the one-shot ack socket.

    The product-side barrier hook connects, sends one JSON request line and waits
    at most ``ack_deadline_s`` for ONE JSON ack. Timeout/refusal => release and
    NOT_RUN; the product never proceeds as if pressure governed.
    """

    def __init__(self, socket_path, *, ack_deadline_s=CONTRACT_MAX_ACK_DEADLINE_S):
        self.socket_path = Path(socket_path)
        self.ack_deadline_s = float(ack_deadline_s)
        if self.ack_deadline_s > CONTRACT_MAX_ACK_DEADLINE_S:
            raise ValueError("ack deadline exceeds the 2 s contract")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.socket_path))
        self._sock.listen(1)
        self._sock.settimeout(self.ack_deadline_s)

    def serve_once(self, apply_fn):
        try:
            conn, _ = self._sock.accept()
        except socket.timeout:
            return {"pressure_applied": False, "status": "NOT_RUN", "reason": "ack_timeout_no_request"}
        with conn:
            conn.settimeout(self.ack_deadline_s)
            try:
                request = json.loads(conn.makefile("r", encoding="utf-8").readline() or "{}")
            except Exception as exc:  # noqa: BLE001
                request = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                response = dict(apply_fn(request) or {})
            except Exception as exc:  # noqa: BLE001
                response = {
                    "pressure_applied": False,
                    "status": "NOT_RUN",
                    "reason": f"apply_error:{type(exc).__name__}:{exc}",
                }
            response.setdefault("ack_deadline_s", self.ack_deadline_s)
            conn.sendall((json.dumps(response, sort_keys=True) + "\n").encode())
            return response

    def close(self):
        try:
            self._sock.close()
        finally:
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def write_corpus(root, *, frame_size=CORPUS_SIZE, shift_scale=1):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    size = int(frame_size)
    scale_px = float(shift_scale)
    inp = Path(root) / "in"
    inp.mkdir(parents=True, exist_ok=True)
    scale = 0.0012
    rng = np.random.default_rng(20261027)
    for i in range(CORPUS_FRAMES):
        dx, dy = CORPUS_SHIFTS[i % len(CORPUS_SHIFTS)]
        dx, dy = dx * scale_px, dy * scale_px
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 - dx * scale, 30.0 + dy * scale]
        wcs.wcs.crpix = [size / 2.0, size / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (size, size)
        yy, xx = np.mgrid[0:size, 0:size]
        data = (
            100.0
            + 20.0 * np.exp(
                -((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
                / (2 * (size / 12.8) ** 2)
            )
            + 3.0 * i
            + rng.normal(0.0, 0.5, (size, size))
        ).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = 10.0
        hdu.writeto(inp / f"frame_{i:04d}.fits", overwrite=True)
    return inp


# ---------------------------------------------------------------------------
# Product child (real run, fresh process)
# ---------------------------------------------------------------------------

def _published_path(out, name):
    """Resolve a canonical member inside the run-owned publication generation."""
    out = Path(out)
    pubs = out / ".zegrid_publications"
    if pubs.is_dir():
        for pointer in sorted(pubs.glob("*/CURRENT.json")):
            try:
                data = json.loads(pointer.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            generation = str(data.get("generation") or "")
            if not generation:
                continue
            candidate = pointer.parent / generation / name
            if candidate.is_file():
                return candidate
    return out / name


def _digest_fits(path):
    import hashlib

    import numpy as np
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        arr = np.ascontiguousarray(data)
        digest = hashlib.sha256()
        digest.update(arr.dtype.str.encode("ascii"))
        digest.update(json.dumps([int(v) for v in arr.shape]).encode("ascii"))
        digest.update(arr.tobytes())
    return {
        "file": Path(path).name,
        "shape": [int(v) for v in data.shape],
        "dtype": str(data.dtype),
        "sha256": digest.hexdigest(),
    }


def product_child_main(payload_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    result_path = Path(payload["result_path"])
    repo = Path(payload["repo"]).resolve()
    sys.path.insert(0, str(repo / "src"))

    result = {
        "ok": False,
        "pid": os.getpid(),
        "stage": "import",
        "storage_profile": payload.get("storage_profile"),
    }
    try:
        import zemosaic.zemosaic_zegrid_mode as zz
        from types import SimpleNamespace

        import m3_observers as obs

        bundle = obs.ObserverBundle()
        if payload.get("commit_trace"):
            sink = obs.TraceSink(payload["commit_trace"], role="product")
            bundle.sinks["commit"] = sink
            bundle.commit = obs.CommitObserver(sink).install(
                __import__(
                    "zemosaic.core.zegrid.run_store", fromlist=["RunCellStore"]
                ).RunCellStore
            )
        if payload.get("execution_trace"):
            sink = obs.TraceSink(payload["execution_trace"], role="product")
            bundle.sinks["execution"] = sink
            bundle.execution = obs.ExecutionObserver(sink).install(zz)
        if payload.get("wait_trace"):
            sink = obs.TraceSink(payload["wait_trace"], role="product")
            bundle.sinks["wait"] = sink
            bundle.wait = obs.WaitObserver(sink).install(zz)
        if payload.get("barrier_trace"):
            sink = obs.TraceSink(payload["barrier_trace"], role="product")
            bundle.sinks["barrier"] = sink
            bundle.barrier = obs.BarrierObserver(
                sink,
                mode=str(payload.get("barrier_mode", "observe")),
                hook=None,
                minimum_commits=int(payload.get("minimum_commits", 0)),
                source_path=Path(zz.__file__),
            ).install(zz)

        zconfig = SimpleNamespace(
            zegrid_scheduler_mode="active",
            zegrid_layout=payload.get("layout", LAYOUT),
            zegrid_resume_mode="never",
            zegrid_durable_store_dir=payload["store"],
            zegrid_resource_recovery_enabled=bool(
                payload.get("recovery_enabled", False)
            ),
            zegrid_resource_recovery_deadline_s=int(
                payload.get("recovery_deadline_s", 900)
            ),
        )
        result["stage"] = "run"
        result["observers"] = bundle.as_dict()
        zz.run_zegrid_mode(
            str(payload["corpus"]),
            str(payload["out"]),
            zconfig=zconfig,
            use_gpu=False,
            workers=int(payload.get("workers", 1)),
            cell_gauge_mode="per_cell",
            zegrid_halo_px=int(payload.get("halo_px", HALO_PX)),
            storage_profile=payload.get("storage_profile", "dense"),
        )
        result["stage"] = "digest"
        out = Path(payload["out"])
        result["science"] = _digest_fits(_published_path(out, "mosaic_grid.fits"))
        result["coverage"] = _digest_fits(_published_path(out, "mosaic_grid_coverage.fits"))
        result["manifest_path"] = str(_published_path(out, "zegrid_manifest.json"))
        result["published_paths"] = {
            "science": str(_published_path(out, "mosaic_grid.fits")),
            "coverage": str(_published_path(out, "mosaic_grid_coverage.fits")),
            "manifest": str(_published_path(out, "zegrid_manifest.json")),
        }
        result["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the harness records the truth
        result.update(
            {
                "ok": False,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result.get("ok") else 1


# ---------------------------------------------------------------------------
# Manifest extraction (X_b provenance)
# ---------------------------------------------------------------------------

AGGREGATE_PATHS = (
    ("r55_scheduler", "execution", "aggregate_bytes"),
    ("per_cell_diagnostics", "ram_contract", "aggregate_bytes"),
    ("ram_contract", "aggregate_bytes"),
    ("r55_scheduler", "final_envelope", "aggregate_bytes"),
)


def _get_path(document, path):
    node = document
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def extract_aggregate(document):
    """Return ``{value, json_path, budget_bytes, raw_available_bytes, ...}``."""
    for path in AGGREGATE_PATHS:
        value = _get_path(document, path)
        if isinstance(value, int) and value > 0:
            observation = _get_path(
                document, ("r55_scheduler", "plan_observation")
            ) or {}
            ram = _get_path(document, ("per_cell_diagnostics", "ram_contract")) or {}
            return {
                "aggregate_bytes": int(value),
                "json_path": ".".join(path),
                "plan_budget_bytes": observation.get("budget_bytes"),
                "plan_raw_available_bytes": observation.get("raw_available_bytes"),
                "final_layout": observation.get("layout"),
                "ram_contract_aggregate_bytes": ram.get("aggregate_bytes"),
            }
    return {"aggregate_bytes": None, "json_path": None}


# ---------------------------------------------------------------------------
# Reference command
# ---------------------------------------------------------------------------

def _write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _find_git_root(path):
    """Return the nearest ancestor (inclusive) containing ``.git``, else None."""
    current = Path(path).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def ensure_outside_git(path, *, what="run root", allow_inside_git=False):
    """Refuse to create artifacts inside ANY git checkout.

    A tracked worktree must stay clean; run roots and stores are evidence, not
    source. This is checked BEFORE any directory is created.
    """
    parent = Path(path).resolve()
    root = _find_git_root(parent)
    if root is not None and not allow_inside_git:
        raise SystemExit(
            f"refusing to use {what} {parent} inside the git checkout {root}; "
            "pass --artifact-base or an absolute path outside any repository"
        )
    return parent


def resolve_artifact_path(value, base):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(base).expanduser() / path
    return path.resolve()


def run_reference(args):
    repo = Path(args.repo).resolve()
    base = Path(args.artifact_base).expanduser().resolve()
    allow_git = bool(getattr(args, "allow_inside_git", False))
    root = ensure_outside_git(
        resolve_artifact_path(args.out, base), what="--out",
        allow_inside_git=allow_git,
    )
    root.mkdir(parents=True, exist_ok=True)
    store = ensure_outside_git(
        resolve_artifact_path(args.store, base), what="--store",
        allow_inside_git=allow_git,
    )
    store.mkdir(parents=True, exist_ok=True)
    events_path = root / "events.jsonl"
    result_path = root / "result.json"
    payload_path = root / "payload.json"

    if not (repo / "src" / "zemosaic" / "zemosaic_zegrid_mode.py").is_file():
        _write_json(args.json_out, {"schema": SCHEMA, "status": "FAIL", "reason": "repo_not_found"})
        return EXIT_USAGE

    if args.corpus and Path(args.corpus).is_dir() and (Path(args.corpus) / "in").is_dir():
        corpus = Path(args.corpus) / "in"
    else:
        corpus = write_corpus(
            Path(args.corpus) if args.corpus else root,
            frame_size=int(args.frame_size),
            shift_scale=float(args.shift_scale),
        )

    payload = {
        "repo": str(repo),
        "corpus": str(corpus),
        "out": str(root / "product"),
        "store": str(store),
        "storage_profile": str(args.storage_profile),
        "workers": int(args.workers),
        "layout": str(args.layout),
        "halo_px": HALO_PX,
        "recovery_enabled": False,
        "recovery_deadline_s": 900,
        "result_path": str(result_path),
        "commit_trace": str(root / "commit-trace.jsonl"),
        "execution_trace": str(root / "execution-trace.jsonl"),
        "barrier_trace": str(root / "barrier-trace.jsonl"),
        "barrier_mode": "observe",
        "minimum_commits": 0,
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    events = JsonlWriter(events_path)
    product_ceiling = (
        None if args.product_ceiling_bytes is None else int(args.product_ceiling_bytes)
    )
    events.emit(
        "reference_start",
        storage_profile=str(args.storage_profile),
        repo=str(repo),
        artifact_base=str(base),
        reserve_bytes=MIN_RESERVE_BYTES,
        product_ceiling_bytes=product_ceiling,
        mem_available_bytes=read_mem_available_bytes(),
    )
    started = _now()
    peak_tree = 0
    peak_mem_available = None
    min_mem_available = None
    guard_breach = None
    stop_reason = None

    # File-backed streams (never PIPE): the product writes a long progress log
    # and a full pipe buffer would block it in pipe_write while the parent is
    # sampling. Redirecting to owned files keeps the observation loop safe.
    stdout_handle = (root / "stdout.log").open("w", encoding="utf-8")
    stderr_handle = (root / "stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_product-run", str(payload_path)],
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    try:
        while True:
            now = _now()
            elapsed = now - started
            mem = read_mem_available_bytes()
            tree = read_tree_rss_bytes(proc.pid)
            peak_tree = max(peak_tree, int(tree))
            peak_mem_available = int(mem) if peak_mem_available is None else max(peak_mem_available, int(mem))
            min_mem_available = int(mem) if min_mem_available is None else min(min_mem_available, int(mem))
            events.emit(
                "reference_sample",
                elapsed_s=round(elapsed, 3),
                mem_available_bytes=mem,
                product_tree_rss_bytes=tree,
                pid=proc.pid,
            )
            if mem < MIN_RESERVE_BYTES:
                guard_breach = "RESERVE_BREACH"
                stop_reason = f"MemAvailable {mem} < reserve {MIN_RESERVE_BYTES}"
                break
            if product_ceiling is not None and tree > product_ceiling:
                guard_breach = "PRODUCT_CEILING_BREACH"
                stop_reason = (
                    f"product tree RSS {tree} > ceiling {product_ceiling}"
                )
                break
            if elapsed > float(args.max_lifetime_s):
                guard_breach = "LIFETIME_BREACH"
                stop_reason = f"reference exceeded {args.max_lifetime_s} s"
                break
            if proc.poll() is not None:
                break
            time.sleep(GUARD_SAMPLE_INTERVAL_S)
    finally:
        if proc.poll() is None and guard_breach:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        stdout_handle.close()
        stderr_handle.close()

    duration = _now() - started
    result = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))

    manifest = {}
    manifest_path = _published_path(root / "product", "zegrid_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    document = {
        "schema": SCHEMA,
        "mode": "reference",
        "storage_profile": str(args.storage_profile),
        "corpus": {
            "path": str(corpus),
            "frames": CORPUS_FRAMES,
            "frame_size_px": int(args.frame_size),
            "shift_scale": float(args.shift_scale),
            "layout": str(args.layout),
            "halo_px": HALO_PX,
        },
        "status": None,
        "exit_code": proc.returncode,
        "duration_s": round(duration, 3),
        "peak_tree_rss_bytes": int(peak_tree),
        "peak_mem_available_bytes": peak_mem_available,
        "min_mem_available_bytes": min_mem_available,
        "reserve_bytes": MIN_RESERVE_BYTES,
        "product_ceiling_bytes": product_ceiling,
        "guard_breach": guard_breach,
        "stop_reason": stop_reason,
        "product_result": result,
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "aggregate": extract_aggregate(manifest) if manifest else {},
        "manifest_scheduler_status": _get_path(manifest, ("r55_scheduler", "status")),
        "durable_commits_observed": _count_kind(root / "commit-trace.jsonl", "commit"),
        "execution_invocations": _count_kind(root / "execution-trace.jsonl", "execution_end"),
    }

    if guard_breach:
        document["status"] = "NOT_RUN"
    elif result.get("ok") and proc.returncode == 0:
        document["status"] = "PASS"
    elif result.get("exception_type") in (
        "AdmissionExhausted", "LayoutInfeasible", "ModeInfeasible",
        "RamContractError", "RuntimeControlError",
    ):
        document["status"] = "NOT_RUN"
    else:
        document["status"] = "FAIL"

    _write_json(root / "exit.json", {"status": document["status"], "exit_code": proc.returncode})
    _write_json(args.json_out, document)
    print(json.dumps({"status": document["status"], "json_out": str(args.json_out), "aggregate": document.get("aggregate")}))
    return EXIT_PASS if document["status"] == "PASS" else (
        EXIT_NOT_RUN if document["status"] == "NOT_RUN" else EXIT_FAIL
    )


def _count_kind(path, kind):
    total = 0
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    if json.loads(line).get("kind") == kind:
                        total += 1
                except Exception:  # noqa: BLE001
                    continue
    except FileNotFoundError:
        return 0
    return total


# ---------------------------------------------------------------------------
# Experiment command (gated off)
# ---------------------------------------------------------------------------

def _experiment_authorized(args):
    return bool(getattr(args, "i_am_reviewed", False)) and os.environ.get(
        "M3_PRESSURE_EXPERIMENT_REVIEWED"
    ) == "1"


def run_experiment(args):
    document = {
        "schema": SCHEMA,
        "mode": "experiment",
        "status": "NOT_RUN",
        "pressure_applied": False,
        "guard_order": list(GUARD_ORDER),
        "hard_pressure_cap_bytes": HARD_PRESSURE_CAP_BYTES,
        "reserve_bytes": int(args.reserve_bytes),
        "ack_deadline_s": float(args.ack_deadline_s),
        "pressure_bytes": int(args.pressure_bytes),
        "k_commits": int(args.k_commits),
        "next_step": (
            "obtain explicit review of the physical-pressure scenario, then "
            "re-run with --i-am-reviewed and M3_PRESSURE_EXPERIMENT_REVIEWED=1"
        ),
    }
    if not _experiment_authorized(args):
        document["reason"] = "PRESSURE_EXPERIMENT_NOT_AUTHORIZED"
        document["detail"] = (
            "experiment mode is gated OFF for this mission: no review grant, so "
            "no child process, socket, mmap or pressure step was created"
        )
        _write_json(args.json_out, document)
        print(json.dumps(document, indent=2, sort_keys=True))
        return EXIT_NOT_RUN

    # Contract validation happens BEFORE any allocation.
    if int(args.pressure_cap_bytes) > HARD_PRESSURE_CAP_BYTES:
        document.update(status="FAIL", reason="CAP_ABOVE_CONTRACT")
        _write_json(args.json_out, document)
        return EXIT_USAGE
    if int(args.pressure_bytes) > int(args.pressure_cap_bytes):
        document.update(status="FAIL", reason="DOSE_ABOVE_CAP")
        _write_json(args.json_out, document)
        return EXIT_USAGE
    if int(args.reserve_bytes) < MIN_RESERVE_BYTES:
        document.update(status="FAIL", reason="RESERVE_BELOW_CONTRACT")
        _write_json(args.json_out, document)
        return EXIT_USAGE
    if float(args.ack_deadline_s) > CONTRACT_MAX_ACK_DEADLINE_S:
        document.update(status="FAIL", reason="ACK_DEADLINE_ABOVE_CONTRACT")
        _write_json(args.json_out, document)
        return EXIT_USAGE

    # The reviewed flow is implemented but intentionally NOT executed here.
    document.update(
        status="NOT_RUN",
        reason="REVIEWED_FLOW_NOT_EXECUTED_IN_THIS_MISSION",
        detail=(
            "guard/IPC machinery is implemented and unit-tested with simulated "
            "children; this mission must not apply physical pressure"
        ),
    )
    _write_json(args.json_out, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return EXIT_NOT_RUN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ref = sub.add_parser("reference", help="real no-pressure product reference")
    ref.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ref.add_argument("--corpus", default=None)
    ref.add_argument("--out", required=True)
    ref.add_argument("--store", required=True)
    ref.add_argument(
        "--artifact-base",
        default=str(Path.home() / ".openclaw/workspace/.m3-linux-runner-artifacts"),
        help="artifact base outside any git checkout (relative --out/--store land here)",
    )
    ref.add_argument("--product-ceiling-bytes", type=int, default=None)
    ref.add_argument(
        "--allow-inside-git", action="store_true",
        help="explicit override: allow an artifact root inside a git checkout",
    )
    ref.add_argument("--json-out", required=True)
    ref.add_argument("--storage-profile", choices=("dense", "bounded"), default="dense")
    ref.add_argument("--workers", type=int, default=1)
    ref.add_argument("--frame-size", type=int, default=CORPUS_SIZE)
    ref.add_argument("--shift-scale", type=float, default=1.0)
    ref.add_argument("--layout", default=LAYOUT)
    ref.add_argument("--max-lifetime-s", type=float, default=MAX_LIFETIME_S)

    exp = sub.add_parser("experiment", help="gated physical-pressure scenario")
    exp.add_argument("--corpus", default=None)
    exp.add_argument("--reference", default=None)
    exp.add_argument("--out", required=True)
    exp.add_argument("--store", required=True)
    exp.add_argument("--json-out", required=True)
    exp.add_argument("--workers", type=int, default=1)
    exp.add_argument("--k-commits", type=int, default=2)
    exp.add_argument("--reserve-bytes", type=int, default=MIN_RESERVE_BYTES)
    exp.add_argument("--pressure-bytes", type=int, default=DEFAULT_PAYLOAD_BYTES)
    exp.add_argument("--pressure-cap-bytes", type=int, default=HARD_PRESSURE_CAP_BYTES)
    exp.add_argument("--ack-deadline-s", type=float, default=CONTRACT_MAX_ACK_DEADLINE_S)
    exp.add_argument("--max-lifetime-s", type=float, default=MAX_LIFETIME_S)
    exp.add_argument("--i-am-reviewed", action="store_true")

    sub.add_parser("_product-run", help=argparse.SUPPRESS)
    sub.add_parser("_pressure-child", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_product-run":
        return product_child_main(argv[1])
    if argv and argv[0] == "_pressure-child":
        return pressure_child_main(argv[1])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "reference":
        return run_reference(args)
    if args.command == "experiment":
        return run_experiment(args)
    parser.error("unknown command")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
