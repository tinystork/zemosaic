"""Product-level BATCH TRUTH witness for one external Cell.

ZM-ZEGRID-BATCH-TRUTH-AND-USAGE (2026-11-12)
============================================

WHAT THIS IS
------------
A reproducible measurement of ONE Cell in the ``external_spool`` lane, driven
through the REAL product entry (``zemosaic.zegrid_mode.run_zegrid_mode``) on a
synthetic corpus.  It emits, as one machine-readable tuple:

* ``recorded_admission_column_batch`` — the durable working-set value;
* ``requested_column_batch`` / ``effective_column_batch`` / ``clamped``;
* ``rows_per_batch`` / ``cols_per_batch`` / ``batch_positions``;
* the model's priced ``peak_bytes`` and the ``peak_phase``;
* ``reduction_batch_bytes``;
* ``executed_*`` — what the REDUCER itself reports it formed (block geometry,
  measured inside ``reduce_spooled_tile``), not a re-derivation;
* ``reducer`` — the TRUE peak RSS of the process that executed the Cell, with
  its pid, role and sampling source, plus the coordinator's own peak as a
  SEPARATE value so the two can never be confused.

WHY IT EXISTS
-------------
The real M33 run recorded ``column_batch = 6528`` and priced it at 12.74 GiB,
while the host's real used RAM matched ~4600 positions and the published
"per-process peak" was 1.06 GiB.  That combination is only possible if either
(a) a smaller resolved batch reached the executor, or (b) the published peak
described the wrong process / a non-peak sample.  This witness settles it with
numbers on a product-level Cell.

HOW TO RUN
----------
::

    python tests/support/external_batch_truth.py --json /tmp/batch-truth.json

The default profile is a few seconds (4 frames, 256x256, one Cell, tile 128).
``--profile heavy`` needs ``ZEMOSAIC_MEASURE_SLOW=1``.

NOT: no science constant, no admission ceiling, no model coefficient, no
dataset is touched.  The synthetic corpus is built in a temporary directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "tests", REPO / "tools" / "zegrid_m6"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from make_tiny_corpus import make_tiny_corpus  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402

SCHEMA = "zemosaic-external-batch-truth-v1"
MIB = 2 ** 20
GIB = 2 ** 30
_SLOW_ENV = "ZEMOSAIC_MEASURE_SLOW"


class _HugeModel:
    """Force the planner past every RAM tile so AUTO selects the external lane."""

    def predict_bound_bytes(self, n, area):
        return 1 << 62

    def predict_peak_bytes(self, n, area):
        return 1 << 62


def _status_kib(field: str) -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


class _TreeSampler:
    """Passive sampler of THIS process and every descendant (psutil).

    The self sample is authoritative for the reducer when the Cell executes in
    this process (the product's external class is single-owner/serial, so that
    is the normal case).  Descendants are sampled too, so a worker-child
    reduction is never invisible; every per-pid peak carries its role.
    """

    def __init__(self, interval_s: float = 0.025) -> None:
        self.interval_s = float(interval_s)
        self.self_peak_bytes = 0
        self.per_pid: dict[int, dict] = {}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        import psutil

        me = os.getpid()
        while not self._stop.is_set():
            try:
                proc = psutil.Process(me)
                self._record(me, "coordinator", proc)
                for child in proc.children(recursive=True):
                    self._record(int(child.pid), "cell_worker", child)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _record(self, pid: int, role: str, proc) -> None:
        try:
            rss = int(proc.memory_info().rss)
        except Exception:
            return
        entry = self.per_pid.setdefault(
            pid, {"pid": int(pid), "role": role, "peak_rss_bytes": 0}
        )
        if rss > entry["peak_rss_bytes"]:
            entry["peak_rss_bytes"] = rss
        if pid == os.getpid() and rss > self.self_peak_bytes:
            self.self_peak_bytes = rss
        self.samples += 1

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="batch-truth-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _external_cell_records(manifest: dict) -> list[dict]:
    """Every per-Cell record that claims the external working set."""
    out = []
    for key in ("per_cell_records", "cell_records", "cells"):
        rows = manifest.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and (
                    str(row.get("mode")) == "external"
                    or str(row.get("working_set")) == "external_spool"
                ):
                    out.append(row)
    return out


def measure_one_cell(
    *,
    layout: str = "1x1",
    available_bytes: int = 8 * GIB,
    stream_estimate_bytes: int = 10 * GIB,
    halo_px: int = 8,
    sample_interval_s: float = 0.025,
    workdir: str | os.PathLike | None = None,
) -> dict:
    """Run ONE product-level external Cell and return the truth tuple."""
    tmp_ctx = None
    if workdir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="zemosaic-batch-truth-")
        workdir = tmp_ctx.name
    workdir = Path(workdir)
    corpus = make_tiny_corpus(workdir / "in")
    out_dir = workdir / "ext"

    cfg = SimpleNamespace(zegrid_layout=str(layout), zegrid_scheduler_mode="legacy")
    saved = (
        zz.available_memory_bytes,
        zz._estimate_streaming_bytes,
        zal.FITTED_MEMORY_MODEL,
    )
    hwm_before = _status_kib("VmHWM")
    sampler = _TreeSampler(sample_interval_s)
    t0 = time.perf_counter()
    try:
        zz.available_memory_bytes = lambda: int(available_bytes)
        zz._estimate_streaming_bytes = lambda *a, **k: int(stream_estimate_bytes)
        zal.FITTED_MEMORY_MODEL = _HugeModel()
        sampler.start()
        zz.run_zegrid_mode(
            str(corpus), str(out_dir),
            zegrid_halo_px=int(halo_px),
            zconfig=cfg,
            storage_profile="bounded",
        )
        # Read BEFORE any temporary-directory cleanup below.
        manifest = json.loads((out_dir / "zegrid_manifest.json").read_text())
    finally:
        elapsed = time.perf_counter() - t0
        sampler.stop()
        zz.available_memory_bytes, zz._estimate_streaming_bytes, \
            zal.FITTED_MEMORY_MODEL = saved
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    ws = dict(manifest.get("working_set") or {})
    host = dict(ws.get("host_components") or {})
    priced_peak = int(ws.get("host_peak_bytes") or host.get("peak_bytes") or 0)
    reductions = int(
        host.get("reduction_batch_bytes") or host.get("batch_total_bytes") or 0
    )
    phases = dict(host.get("phases") or {})
    peak_phase = max(phases, key=lambda k: phases[k]) if phases else None

    records = _external_cell_records(manifest)
    record = records[0] if records else {}
    ext = dict(record.get("external") or {})
    mem = dict(record.get("memory_peak") or {})
    executed_measured = dict(ext.get("executed_batch") or {})

    self_hwm_after = _status_kib("VmHWM")
    self_peak_kib = max(
        int(sampler.self_peak_bytes) // 1024,
        int(self_hwm_after or 0),
    )
    per_pid = sorted(sampler.per_pid.values(), key=lambda e: -e["peak_rss_bytes"])

    recorded = ws.get("column_batch")
    requested = ws.get("requested_column_batch") or ext.get("requested_column_batch")
    effective = ws.get("effective_column_batch") or ext.get("effective_column_batch")
    executed = ext.get("executed_column_batch")
    executed_positions = (
        executed_measured.get("max_block_positions")
        if executed_measured else None
    )

    tuple_block = {
        "recorded_admission_column_batch": (
            None if recorded is None else int(recorded)
        ),
        "requested_column_batch": (
            None if requested is None else int(requested)
        ),
        "effective_column_batch": (
            None if effective is None else int(effective)
        ),
        "column_batch_clamped": bool(ws.get("column_batch_clamped", False)),
        "rows_per_batch": (
            None if ws.get("executed_rows_per_batch") is None
            else int(ws["executed_rows_per_batch"])
        ),
        "cols_per_batch": (
            None if ws.get("executed_cols_per_batch") is None
            else int(ws["executed_cols_per_batch"])
        ),
        "batch_positions": (
            None if effective is None else int(effective)
        ),
        "model_priced_peak_bytes": int(priced_peak),
        "peak_phase": None if peak_phase is None else str(peak_phase),
        "reduction_batch_bytes": int(reductions),
        "executed_column_batch": None if executed is None else int(executed),
        "executed_max_block_positions": (
            None if executed_positions is None else int(executed_positions)
        ),
        "executed_batch_measured": executed_measured,
    }

    reducer_peak_kib = int(mem.get("peak_rss_kib") or 0)
    reducer = {
        "peak_rss_bytes": reducer_peak_kib * 1024,
        "source": mem.get("source"),
        "pid": mem.get("pid"),
        "role": mem.get("role"),
        "parent_pid": mem.get("parent_pid"),
        "is_true_peak": mem.get("is_true_peak"),
    }
    coordinator = {
        "peak_rss_bytes": self_peak_kib * 1024,
        "source": "max(sampler VmRSS, VmHWM)",
        "pid": os.getpid(),
        "role": "coordinator",
    }

    verdict = {
        "executed_equals_recorded": (
            tuple_block["executed_column_batch"] is not None
            and tuple_block["recorded_admission_column_batch"] is not None
            and int(tuple_block["executed_column_batch"])
            == int(tuple_block["recorded_admission_column_batch"])
        ),
        "executed_equals_effective_priced": (
            tuple_block["executed_column_batch"] is not None
            and tuple_block["effective_column_batch"] is not None
            and int(tuple_block["executed_column_batch"])
            == int(tuple_block["effective_column_batch"])
        ),
        "executed_measured_equals_effective": (
            executed_positions is not None
            and tuple_block["effective_column_batch"] is not None
            and int(executed_positions)
            == int(tuple_block["effective_column_batch"])
        ),
        "reducer_is_coordinator": mem.get("pid") == os.getpid(),
        "reducer_peak_is_true_peak": bool(mem.get("is_true_peak")),
        "recorded_exceeds_one_tile_interior": bool(
            tuple_block["recorded_admission_column_batch"] is not None
            and tuple_block["effective_column_batch"] is not None
            and int(tuple_block["recorded_admission_column_batch"])
            > int(tuple_block["effective_column_batch"])
        ),
    }

    return {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "profile": {
            "layout": layout,
            "frames": 4,
            "frame_hw": [256, 256],
            "halo_px": int(halo_px),
            "available_bytes": int(available_bytes),
            "stream_estimate_bytes": int(stream_estimate_bytes),
        },
        "cell": {
            "cell_id": record.get("cell_id"),
            "mode": record.get("mode"),
            "working_set": record.get("working_set"),
            "tile_size": ws.get("tile_size"),
            "bulk": dict(record.get("storage", {})).get("external"),
        },
        "tuple": tuple_block,
        "reducer": reducer,
        "coordinator": coordinator,
        "sampled_processes": per_pid,
        "verdict": verdict,
        "elapsed_s": round(float(elapsed), 3),
        "uncertainty": {
            "reducer_peak": (
                "True peak from the product's own per-cell telemetry "
                "(resource ru_maxrss / psutil peak_wset / VmHWM), labelled "
                "with pid+role; the sampler cross-checks the same process."
            ),
            "executed_batch": (
                "Reported by reduce_spooled_tile per tile (blocks it actually "
                "formed), aggregated by run_cell_streaming_external."
            ),
            "sampling": "25 ms tree sampler; VmHWM closes self-process gaps.",
            "baseline": (
                "absolute RSS includes the interpreter/import baseline; the "
                "model prices owned arrays only, so absolute > model by design."
            ),
        },
    }


def _summary(artefact: dict) -> str:
    t = artefact["tuple"]
    v = artefact["verdict"]
    r = artefact["reducer"]
    return "\n".join([
        "EXTERNAL CELL BATCH TRUTH",
        f"  recorded admission column_batch : {t['recorded_admission_column_batch']}",
        f"  requested / effective / clamped : {t['requested_column_batch']} / "
        f"{t['effective_column_batch']} / {t['column_batch_clamped']}",
        f"  rows x cols x positions         : {t['rows_per_batch']} x "
        f"{t['cols_per_batch']} = {t['batch_positions']}",
        f"  model priced peak / phase       : "
        f"{t['model_priced_peak_bytes'] / MIB:.1f} MiB / {t['peak_phase']}",
        f"  reduction_batch_bytes           : {t['reduction_batch_bytes'] / MIB:.1f} MiB",
        f"  EXECUTED column_batch           : {t['executed_column_batch']}",
        f"  executed max block (measured)   : {t['executed_max_block_positions']}",
        f"  reducer peak RSS                : {r['peak_rss_bytes'] / MIB:.1f} MiB "
        f"[{r['role']} pid={r['pid']} source={r['source']}]",
        f"  coordinator peak RSS            : "
        f"{artefact['coordinator']['peak_rss_bytes'] / MIB:.1f} MiB",
        f"  executed == recorded            : {v['executed_equals_recorded']}",
        f"  executed == effective(priced)   : {v['executed_equals_effective_priced']}",
        f"  reducer == coordinator          : {v['reducer_is_coordinator']}",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("--layout", default="1x1")
    ap.add_argument("--profile", default="standard", choices=("standard", "heavy"))
    ap.add_argument("--sample-interval-ms", type=float, default=25.0)
    args = ap.parse_args(argv)
    if args.profile == "heavy" and not os.environ.get(_SLOW_ENV):
        print(f"profile 'heavy' is gated: set {_SLOW_ENV}=1", file=sys.stderr)
        return 2
    art = measure_one_cell(
        layout=args.layout,
        sample_interval_s=args.sample_interval_ms / 1000.0,
    )
    print(_summary(art))
    if args.json:
        Path(args.json).write_text(json.dumps(art, indent=2), encoding="utf-8")
        print(f"\nartefact: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
