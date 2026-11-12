"""Measurement-only harness for the external working-set HOST model.

ZM-ZEGRID-EXTERNAL-MODEL-MEASURE (2026-11-12)
=============================================

WHAT THIS IS
------------
A *witness*, not a fix.  It runs the real production executor
(:func:`zemosaic.core.zegrid.external_stream.run_cell_streaming_external`) on a
synthetic cell for several ``column_batch`` values and compares

* the model's prediction — ``estimate_external_working_set(...)["host"]
  ["peak_bytes"]`` plus its explicit term breakdown, against
* the ACTUAL peak resident memory of the process during that run, sampled from
  ``/proc/self/status`` (``VmRSS``) every ``sample_interval_s`` (default 25 ms),
  cross-checked with the kernel high-water mark (``VmHWM``, reset before each
  run through ``/proc/self/clear_refs``).

Every configuration runs in a FRESH child process (one Cell in flight, exactly
like production), so allocator high-water marks from a previous configuration
cannot leak into the next measurement.

WHAT THIS IS NOT
----------------
* No product code is imported by ``src/`` — this module lives under ``tests/``
  and is only ever imported by the bounded regression and by the CLI.
* The executor is NOT perturbed: no monkeypatching, no instrumented copy of the
  reduction, no constant overridden.  Sampling is passive (a reader thread that
  only reads ``/proc/self/status``).
* Model constants, admission thresholds, planner, science and payload are
  untouched.

HOW TO RUN
----------
Fast default (used by the bounded regression) is a few seconds::

    python tests/support/external_model_measure.py --profile quick --json /tmp/quick.json

The realistic shape (N=300, patch 256x256, tile 128, four batches) takes
minutes and is opt-in::

    python tests/support/external_model_measure.py --profile standard --json /tmp/standard.json

Large-N / real-corpus variants are gated behind ``--profile heavy`` or
``ZEMOSAIC_MEASURE_SLOW=1`` (see ``run_matrix``); the default pytest run never
touches them.

MEASUREMENT UNCERTAINTY (stated, not hidden)
--------------------------------------------
* **Sampling**: a 25 ms reader can miss a transient spike shorter than the
  interval.  ``VmHWM`` is the kernel's own high-water mark for the whole
  process and closes that gap, so the reported peak is
  ``max(sampled_peak, VmHWM_after)``.
* **Baseline**: RSS includes the interpreter + NumPy + import-time state.
  ``delta_peak = peak - baseline`` is the quantity comparable to the model's
  Cell working set; ``peak`` (absolute) is reported as an upper bound.
* **Allocator retention**: glibc may keep freed arenas, so a later phase can
  inherit a high-water mark from an earlier one.  The model is a *peak over
  sequential phases*, so this biases the observed value UP (never down).
* **Page cache / mmap**: the reduction reads each spooled tile through
  ``np.load(mmap_mode="r")``; resident file-backed pages count in RSS.  This
  also biases observed UP, and is invisible to the model (which prices owned
  arrays only).
* **Spool writes**: ``np.lib.format.open_memmap`` writes go to the page cache
  via the mapping, again a potential (small) RSS contributor not in the model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:  # the pytest conftest does this too; CLI needs it.
    sys.path.insert(0, str(_SRC))

from zemosaic.core.canonical_engine import CanonicalStackRequest  # noqa: E402
from zemosaic.core.zegrid import external_stream as xstream  # noqa: E402
from zemosaic.core.zegrid import file_provider as fp  # noqa: E402
from zemosaic.core.zegrid.rejection_sink import (  # noqa: E402
    BoundedCountingRejectionSink,
)

SCHEMA = "zemosaic-external-model-measurement-v1"
MIB = 2 ** 20
GIB = 2 ** 30

DEFAULT_SAMPLE_INTERVAL_S = 0.025
_SLOW_ENV = "ZEMOSAIC_MEASURE_SLOW"


# ---------------------------------------------------------------------------
# RSS sampling
# ---------------------------------------------------------------------------

def _read_status_field(field: str) -> int | None:
    """Read one kB-valued field from ``/proc/self/status`` (Linux only)."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _reset_peak_rss() -> bool:
    """Reset the kernel high-water mark for this process (Linux, best effort)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


class RssSampler:
    """Passive reader thread: samples ``VmRSS`` until stopped."""

    def __init__(self, interval_s: float = DEFAULT_SAMPLE_INTERVAL_S) -> None:
        self.interval_s = float(interval_s)
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _read_status_field("VmRSS")
            if value is not None:
                self.samples += 1
                if value > self.peak_bytes:
                    self.peak_bytes = value
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="rss-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Synthetic corpus + request
# ---------------------------------------------------------------------------

def _geometry(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 6.0 * np.sin(xx * 0.37) + 3.0 * np.cos(yy * 0.29)
    return base, float(yy[0, 0]), xx, yy


def make_prepare(n: int, h: int, w: int, channels: int = 3, seed: int = 20261112):
    """Deterministic, varied per-frame generator (never all-identical).

    Cheap by construction (trig, no RNG per pixel) because the spool pass
    prepares every frame of every tile: the point of the measurement is the
    reduction's live set, not the generator's throughput.
    """
    base = 100.0 + 6.0 * np.sin(
        np.mgrid[0:h, 0:w][1] * 0.37
    ) + 3.0 * np.cos(np.mgrid[0:h, 0:w][0] * 0.29)
    yy, xx = np.mgrid[0:h, 0:w]
    ch_gain = np.linspace(0.97, 1.03, channels).astype(np.float32)

    def prepare(i: int):
        phase = 0.11 * (int(i) % 19)
        gain = 0.96 + 0.08 * ((int(i) % 13) / 12.0)
        field = base * gain + 1.4 * np.sin(
            xx * 0.713 + yy * 0.311 + phase
        )
        frame = np.empty((h, w, channels), dtype=np.float32)
        for ch in range(channels):
            frame[:, :, ch] = (
                field * float(ch_gain[ch])
                + 0.9 * np.sin(xx * 0.19 + phase + ch)
            )
        mask = np.ones((h, w), dtype=bool)
        if i % 13 == 0:
            mask[:, :2] = False  # low-N column
        if i % 7 == 0:
            frame[3:6, 3:6, :] = np.nan
        if i % 97 == 96:
            frame[5, 5, 0] = 1e5  # rare outlier (exercises rejection)
        return np.ascontiguousarray(frame), mask

    return prepare


def make_request(n: int, *, rejection: str = "winsorized_sigma_clip",
                 combine: str = "mean", taper: str = "footprint"):
    return CanonicalStackRequest(
        images=[None] * n,
        geometric_support=[None] * n,
        normalization="sky_mean",
        weighting="noise_variance",
        rejection=rejection,
        combine=combine,
        taper=taper,
        taper_px=8.0,
    )


def model_price(n: int, h: int, w: int, tile: int, batch: int,
                channels: int = 3, request=None) -> dict:
    """Price the EXACT configuration the executor will run (same halo source)."""
    request = request or make_request(n)
    _k, _px, _floor, halo = xstream._resolve_taper(request)
    return xstream.estimate_external_working_set(
        n_frames=n,
        height=h,
        width=w,
        channels=channels,
        tile_size=tile,
        halo_px=halo,
        column_batch=int(batch),
        equalize_rgb=bool(request.equalize_rgb),
        rejection=str(request.rejection),
        combine=str(request.combine),
    )


# ---------------------------------------------------------------------------
# Term breakdown / verdict helpers
# ---------------------------------------------------------------------------

def _terms(host: dict) -> dict:
    """Named host-model contributors, in the model's own vocabulary."""
    return {
        "reduction_batch_bytes": int(host["batch_total_bytes"]),
        "persistent_result_bytes": int(host["persistent_result_bytes"]),
        "reduce_tile_outputs_bytes": int(host["reduce_tile_outputs_bytes"]),
        "provider_window_bytes": int(host["provider_window_bytes"]),
        "write_tile_bytes": int(host["write_tile_bytes"]),
        "finalize_total_bytes": int(host["finalize_total_bytes"]),
        "dense_rejection_mask_bytes": int(host["dense_rejection_mask_bytes"]),
    }


def _dominant_term(host: dict) -> tuple[str, int]:
    terms = _terms(host)
    name = max(terms, key=lambda k: terms[k])
    return name, terms[name]


def _peak_phase(host: dict) -> str:
    phases = host["phases"]
    return max(phases, key=lambda k: phases[k])


def classify_ratio(ratio: float, *, tolerance: float = 0.5) -> str:
    """Verdict for ``observed / predicted`` (>1 means the model UNDER-estimates)."""
    if ratio >= 1.0 + tolerance:
        return "model_under_estimates"
    if ratio <= 1.0 - tolerance:
        return "model_over_estimates"
    return "model_tracks_reality"


# ---------------------------------------------------------------------------
# One measurement (runs in its own process)
# ---------------------------------------------------------------------------

def measure_one(
    *,
    n: int,
    patch: int,
    tile: int,
    batch: int,
    channels: int = 3,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    spool_root: str | os.PathLike | None = None,
    rejection: str = "winsorized_sigma_clip",
    combine: str = "mean",
    taper: str = "footprint",
    seed: int = 20261112,
    label: str | None = None,
) -> dict:
    """Execute ONE configuration and return predicted vs observed peak memory."""
    request = make_request(
        n, rejection=rejection, combine=combine, taper=taper
    )
    pricing = model_price(n, patch, patch, tile, batch, channels, request)
    host = pricing["host"]

    provider = fp.WindowedCanonicalProvider(
        make_prepare(n, patch, patch, channels, seed),
        n,
        expected_hw=(patch, patch),
        expected_channels=channels,
    )
    sink = BoundedCountingRejectionSink((n, patch, patch, channels))

    tmp_ctx = None
    if spool_root is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="zemosaic-measure-")
        spool_root = tmp_ctx.name
    else:
        spool_root = str(spool_root)
        Path(spool_root).mkdir(parents=True, exist_ok=True)

    try:
        gc.collect()
        baseline = _read_status_field("VmRSS") or 0
        hwm_reset = _reset_peak_rss()
        sampler = RssSampler(sample_interval_s)
        t0 = time.perf_counter()
        sampler.start()
        try:
            xstream.run_cell_streaming_external(
                provider,
                request,
                tile_size=tile,
                column_batch=int(batch),
                spool_root=spool_root,
                rejection_sink=sink,
                cell_id=f"measure-n{n}-t{tile}-b{batch}",
                attempt_id="a0",
            )
        finally:
            elapsed = time.perf_counter() - t0
            sampler.stop()
        hwm_after = _read_status_field("VmHWM")
        after = _read_status_field("VmRSS") or 0
    finally:
        provider.close()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    observed_peak = max(int(sampler.peak_bytes), int(hwm_after or 0))
    delta_peak = max(0, observed_peak - int(baseline))
    predicted = int(host["peak_bytes"])
    dominant, dominant_bytes = _dominant_term(host)
    ratio_delta = (delta_peak / predicted) if predicted else float("nan")
    ratio_abs = (observed_peak / predicted) if predicted else float("nan")

    return {
        "label": label or f"n{n}_patch{patch}_tile{tile}_batch{batch}",
        "configuration": {
            "n_frames": int(n),
            "patch_hw": [int(patch), int(patch)],
            "tile_size": int(tile),
            "channels": int(channels),
            "requested_column_batch": int(batch),
            "effective_column_batch": int(pricing["ram"]["effective_column_batch"]),
            "column_batch_clamped": bool(pricing["ram"]["column_batch_clamped"]),
            "batch_positions": int(pricing["ram"]["batch_positions"]),
            "rows_per_batch": int(pricing["ram"]["rows_per_batch"]),
            "cols_per_batch": int(pricing["ram"]["cols_per_batch"]),
            "taper": taper,
            "rejection": rejection,
            "combine": combine,
        },
        "model": {
            "peak_bytes": predicted,
            "peak_phase": _peak_phase(host),
            "phases": {k: int(v) for k, v in host["phases"].items()},
            "terms": _terms(host),
            "dominant_term": dominant,
            "dominant_term_bytes": int(dominant_bytes),
            "reduce_fixed_bytes": int(host["reduce_fixed_bytes"]),
            "batch_bytes_per_position_frame": int(
                host["batch_bytes_per_position_frame"]
            ),
            "batch_live_bound_bytes_per_position_frame": int(
                host["batch_live_bound_bytes_per_position_frame"]
            ),
        },
        "observed": {
            "peak_rss_bytes": int(observed_peak),
            "peak_rss_source": "max(sampler VmRSS, VmHWM)",
            "baseline_rss_bytes": int(baseline),
            "delta_peak_bytes": int(delta_peak),
            "post_run_rss_bytes": int(after),
            "sampler_samples": int(sampler.samples),
            "sample_interval_s": float(sample_interval_s),
            "hwm_reset_supported": bool(hwm_reset),
            "elapsed_s": round(float(elapsed), 3),
        },
        "comparison": {
            "ratio_observed_delta_over_predicted": float(ratio_delta),
            "ratio_observed_absolute_over_predicted": float(ratio_abs),
            "verdict": classify_ratio(ratio_delta),
        },
    }


# ---------------------------------------------------------------------------
# Profiles / matrix
# ---------------------------------------------------------------------------

PROFILES = {
    # fast smoke shape: geometry is real, N is tiny (a few seconds total)
    "quick": {
        "n_frames": 12,
        "patch": 64,
        "tile": 32,
        "batches": [24, 96, 384, 1536],
    },
    # the shape the brief asks for: large N, realistic patch, tile 128
    "standard": {
        "n_frames": 300,
        "patch": 256,
        "tile": 128,
        "batches": [384, 1536, 4608, 9216],
    },
    # heavier: doubles N (and is memory-hungrier) to expose the N-scaling of
    # the batch term. Batches stay below 9216 so the run fits a small host.
    "heavy": {
        "n_frames": 600,
        "patch": 256,
        "tile": 128,
        "batches": [1536, 4608],
    },
}


def run_matrix(
    profile: str = "standard",
    *,
    json_path: str | os.PathLike,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    keep_spool_root: str | os.PathLike | None = None,
    profiles: dict | None = None,
) -> dict:
    """Run every configuration of ``profile`` in a FRESH child process."""
    cfg = (profiles or PROFILES)[profile]
    results = []
    with tempfile.TemporaryDirectory(prefix="zemosaic-measure-matrix-") as root:
        spool_root = keep_spool_root or os.path.join(root, "spool")
        for batch in cfg["batches"]:
            spec = {
                "n": cfg["n_frames"],
                "patch": cfg["patch"],
                "tile": cfg["tile"],
                "batch": batch,
                "sample_interval_s": sample_interval_s,
                "spool_root": spool_root,
            }
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--one",
                 "--spec", json.dumps(spec)],
                capture_output=True,
                text=True,
                check=False,
            )
            line = next(
                (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
                None,
            )
            if proc.returncode != 0 or line is None:
                raise RuntimeError(
                    f"measurement child failed for spec {spec}: "
                    f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )
            results.append(json.loads(line[len("RESULT "):]))

    artefact = {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sample_interval_s": float(sample_interval_s),
        "uncertainty": {
            "sampling": "VmRSS sampled every interval; VmHWM closes the gaps",
            "baseline": "delta_peak = peak - baseline RSS (null_workload)",
            "allocator_retention": "biases observed UP",
            "page_cache_mmap": "spool reads via mmap count in RSS; biases observed UP",
            "spool_writes": "memmap writes may add page-cache RSS; biases observed UP",
        },
        "configurations": results,
        "verdict": summarise(results),
        "extrapolation": extrapolate(results, target_bytes=1060 * MIB),
    }
    Path(json_path).write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    return artefact


def summarise(results: list[dict]) -> dict:
    """Aggregate verdict over configurations (delta-based ratios)."""
    ratios = [
        r["comparison"]["ratio_observed_delta_over_predicted"] for r in results
    ]
    worst = max(results, key=lambda r: r["comparison"]["ratio_observed_delta_over_predicted"])
    best = min(results, key=lambda r: r["comparison"]["ratio_observed_delta_over_predicted"])
    dominant_terms = sorted({r["model"]["dominant_term"] for r in results})
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    verdict = classify_ratio(mean_ratio) if ratios else "no_data"
    return {
        "mean_ratio": mean_ratio,
        "min_ratio": float(best["comparison"]["ratio_observed_delta_over_predicted"]),
        "max_ratio": float(worst["comparison"]["ratio_observed_delta_over_predicted"]),
        "verdict": verdict,
        "dominant_terms": dominant_terms,
        "per_configuration_verdicts": {
            r["label"]: r["comparison"]["verdict"] for r in results
        },
    }


def extrapolate(results: list[dict], *, target_bytes: int) -> dict:
    """Fit observed delta peak vs batch_positions; locate the production peak.

    Coefficients are ordinary least squares on the measured points
    ``(batch_positions, delta_peak_bytes)``.  The intercept groups the
    batch-independent fixed terms (persistent planes, tile outputs, baseline
    churn); the slope is the observed bytes per position.
    """
    xs = np.array([r["configuration"]["batch_positions"] for r in results], float)
    ys = np.array([r["observed"]["delta_peak_bytes"] for r in results], float)
    if len(xs) < 2 or float(np.ptp(xs)) == 0.0:
        return {"status": "insufficient_points"}
    slope, intercept = np.polyfit(xs, ys, 1)
    implied_positions = (
        (target_bytes - intercept) / slope if slope > 0 else float("nan")
    )
    model_slope = float(np.mean([
        r["model"]["terms"]["reduction_batch_bytes"]
        / max(1, r["configuration"]["batch_positions"])
        for r in results
    ]))
    return {
        "status": "ok",
        "observed_bytes_per_position": float(slope),
        "observed_fixed_bytes": float(intercept),
        "model_bytes_per_position": model_slope,
        "observed_over_model_slope": float(slope / model_slope) if model_slope else float("nan"),
        "target_bytes": int(target_bytes),
        "target_label": "production M33 observed peak (~1.06 GiB RSS)",
        "implied_batch_positions_at_target": float(implied_positions),
        "model_predicted_at_that_batch": float(
            implied_positions * model_slope if slope > 0 else float("nan")
        ),
        "note": (
            "A production run with a 1.06 GiB peak can only have executed a "
            "batch near this implied position count; if the frozen batch was "
            "13440, the peak was not measured on the process that reduced."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(artefact: dict) -> None:
    print(
        f"{'configuration':<38}{'positions':>10}{'predicted':>12}"
        f"{'observedΔ':>12}{'ratio':>8}  dominant term"
    )
    for r in artefact["configurations"]:
        c, m, o, cmp = (
            r["configuration"], r["model"], r["observed"], r["comparison"]
        )
        print(
            f"{r['label']:<38}{c['batch_positions']:>10}"
            f"{m['peak_bytes'] / GIB:>11.3f}G{o['delta_peak_bytes'] / GIB:>11.3f}G"
            f"{cmp['ratio_observed_delta_over_predicted']:>8.2f}  {m['dominant_term']}"
        )
    v = artefact["verdict"]
    print(
        f"\nverdict: {v['verdict']}  (observed/predicted mean {v['mean_ratio']:.2f}, "
        f"range {v['min_ratio']:.2f}..{v['max_ratio']:.2f})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    ap.add_argument("--json", default=None, help="write the artefact here")
    ap.add_argument("--sample-interval-ms", type=float,
                    default=DEFAULT_SAMPLE_INTERVAL_S * 1000.0)
    ap.add_argument("--keep-spool-root", default=None)
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--spec", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.one:
        spec = json.loads(args.spec)
        result = measure_one(
            n=spec["n"], patch=spec["patch"], tile=spec["tile"],
            batch=spec["batch"],
            sample_interval_s=float(spec.get("sample_interval_s", DEFAULT_SAMPLE_INTERVAL_S)),
            spool_root=spec.get("spool_root"),
        )
        print("RESULT " + json.dumps(result))
        return 0

    if args.profile in ("heavy",) and not os.environ.get(_SLOW_ENV):
        print(
            f"profile {args.profile!r} is gated: set {_SLOW_ENV}=1 to run it",
            file=sys.stderr,
        )
        return 2

    json_path = args.json or f"external-model-measure-{args.profile}.json"
    artefact = run_matrix(
        args.profile,
        json_path=json_path,
        sample_interval_s=args.sample_interval_ms / 1000.0,
        keep_spool_root=args.keep_spool_root,
    )
    _print_table(artefact)
    print(f"\nartefact: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
