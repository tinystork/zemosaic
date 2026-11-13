"""Product-level PEAK ATTRIBUTION harness for one external_spool Cell.

ZM-ZEGRID-PEAK-ATTRIBUTION-20261113
===================================

WHAT THIS IS
------------
A *witness* that runs ONE real ``external_spool`` Cell through the production
executor (:func:`zemosaic.core.zegrid.external_stream.run_cell_streaming_external`)
and decomposes the observed peak, PER PHASE, into

* **owned anonymous** — resident private memory the executor owns (the arrays
  the model prices, plus whatever the allocator retains),
* **file-backed mapped** — spool tiles written through ``open_memmap`` and read
  back through ``np.load(mmap_mode="r")``; resident file pages count in the
  working set but are NOT owned memory,
* **allocator retention / transient overlap** — owned bytes above the priced
  live set (derived, never guessed),
* **unexplained** — the part of the total peak not attributable to the measured
  components (reported as ``None`` when the platform cannot separate them).

The production M33 shape (n = 21 818, tile 128, column_batch 6528) prices a
13.7 GB batch term and cannot be executed on a 7.5 GiB host. The harness
therefore uses a shape with the SAME structural ratios, which are what the
attribution depends on:

* ``tile = 128`` and one tile per Cell, so the reduction reads exactly the same
  interior surface per frame;
* ``column_batch = 6528`` positions = 17 interior rows x 128 columns x 3 ch, the
  same block geometry the production batch resolves to;
* the batch and the mapped spool surface BOTH scale linearly in ``n``, and their
  ratio is n-independent, so the measured ``owned/model`` and
  ``mapped/model`` ratios transfer to production while the absolute bytes do not.

Both profiles are bounded and reproducible; the larger one is opt-in (slow).

HOW TO RUN
----------
::

    python tests/support/external_peak_attribution.py --profile bounded \\
        --json /tmp/peak-attribution-bounded.json

    ZEMOSAIC_MEASURE_SLOW=1 python tests/support/external_peak_attribution.py \\
        --profile shaped --json /tmp/peak-attribution-shaped.json

NOTHING here is imported by ``src/``: it is a witness, not product code.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from zemosaic.core.zegrid import external_stream as xstream  # noqa: E402
from zemosaic.core.zegrid import file_provider as fp  # noqa: E402
from zemosaic.core.zegrid import memory_peak as zmem  # noqa: E402
from zemosaic.core.zegrid.rejection_sink import (  # noqa: E402
    BoundedCountingRejectionSink,
)

from support import external_model_measure as em  # noqa: E402

SCHEMA = "zemosaic-external-peak-attribution-v1"
MIB = 2 ** 20
_SLOW_ENV = em._SLOW_ENV

PROFILES = {
    # bounded tier: same tile / batch geometry, small n (seconds)
    "bounded": {
        "n": 60,
        "patch": 128,
        "tile": 128,
        "column_batch": 6528,
        "sample_interval_s": 0.02,
    },
    # shaped tier: production-ratio shape on a small host (minutes)
    "shaped": {
        "n": 300,
        "patch": 128,
        "tile": 128,
        "column_batch": 6528,
        "sample_interval_s": 0.01,
    },
}


def _mib(value) -> float | None:
    return None if value is None else round(float(value) / MIB, 3)


def measure_cell(
    *,
    n: int,
    patch: int,
    tile: int,
    column_batch: int,
    channels: int = 3,
    sample_interval_s: float = 0.02,
    spool_root: str | os.PathLike | None = None,
) -> dict:
    """Run ONE external Cell and return the attributed decomposition."""
    request = em.make_request(n)
    pricing = em.model_price(
        n, patch, patch, tile, column_batch, channels, request
    )
    host = pricing["host"]
    provider = fp.WindowedCanonicalProvider(
        em.make_prepare(n, patch, patch, channels), n,
        expected_hw=(patch, patch), expected_channels=channels,
    )
    sink = BoundedCountingRejectionSink((n, patch, patch, channels))

    tmp_ctx = None
    if spool_root is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="zemosaic-attribution-")
        spool_root = tmp_ctx.name
    else:
        Path(spool_root).mkdir(parents=True, exist_ok=True)

    sampler = zmem.PeakMemorySampler(interval_s=sample_interval_s)
    gc.collect()
    # The model prices the CELL's owned working set, not the interpreter
    # baseline (numpy + imports), so every component is also reported as a
    # DELTA from a sample taken immediately before the executor starts.
    baseline = zmem.sample_memory_components()
    t0 = time.perf_counter()
    sampler.start()
    try:
        try:
            xstream.run_cell_streaming_external(
                provider, request,
                tile_size=tile,
                column_batch=int(column_batch),
                spool_root=spool_root,
                rejection_sink=sink,
                cell_id=f"attrib-n{n}-t{tile}-b{column_batch}",
                attempt_id="a0",
                memory_probe=sampler.mark,
            )
        finally:
            elapsed = time.perf_counter() - t0
            sampler.stop()
        observed = sampler.result()
    finally:
        provider.close()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    predicted = int(host["peak_bytes"])
    observed = dict(observed)
    observed["baseline"] = baseline
    owned_peak = observed.get("owned_peak_bytes")
    file_peak = observed.get("file_backed_peak_bytes")
    at_peak = dict(observed.get("at_total_peak") or {})

    def _delta(value, key):
        if value is None or baseline.get(key) is None:
            return None
        return int(value) - int(baseline[key])

    total_at_peak = _delta(at_peak.get("total_bytes"), "total_bytes")
    owned_at_peak = _delta(at_peak.get("owned_bytes"), "owned_bytes")
    file_at_peak = _delta(at_peak.get("file_backed_bytes"), "file_backed_bytes")
    deltas = {
        "total_delta_bytes": _delta(observed.get("total_peak_bytes"), "total_bytes"),
        "owned_delta_bytes": _delta(owned_peak, "owned_bytes"),
        "file_backed_delta_bytes": _delta(file_peak, "file_backed_bytes"),
    }
    capabilities = dict(observed.get("capability") or {})

    # Priced live set attained at the REDUCE phase (the peak phase):
    # persistent planes + tile outputs + the column batch. That is the quantity
    # the model claims to bound.
    reduce_price = int(host["phases"]["reduce"])

    # Retention / transient overlap: owned bytes above the priced live set, only
    # meaningful where the owned component is measurably resident.
    retention = None
    if owned_at_peak is not None:
        retention = int(owned_at_peak) - reduce_price

    # Mapped residue above what the model already names as the spool surface:
    # the model now NAMES the mapped surface, so the honest comparison is
    # observed_file_backed vs mapped_spool_bytes_per_tile_read.
    named_mapped = int(host.get("mapped_spool_bytes_per_tile_read", 0))
    mapped_residual = None
    if file_at_peak is not None:
        mapped_residual = int(file_at_peak) - named_mapped

    unexplained = at_peak.get("unaccounted_bytes")

    attribution = {
        "total_peak_bytes": total_at_peak,
        "owned_bytes": owned_at_peak,
        "file_backed_bytes": file_at_peak,
        "shared_bytes": at_peak.get("shared_bytes"),
        "unexplained_bytes": unexplained,
        "priced_live_bytes_at_reduce": reduce_price,
        "allocator_retention_bytes": retention,
        "named_mapped_spool_bytes": named_mapped,
        "mapped_residual_bytes": mapped_residual,
        "absolute_at_peak_bytes": {
            "total": at_peak.get("total_bytes"),
            "owned": at_peak.get("owned_bytes"),
            "file_backed": at_peak.get("file_backed_bytes"),
        },
    }

    shares = {}
    if total_at_peak:
        for name in ("owned_bytes", "file_backed_bytes", "unexplained_bytes"):
            value = attribution[name]
            shares[name] = None if value is None else round(
                float(value) / float(total_at_peak), 4
            )
    return {
        "label": f"n{n}_patch{patch}_tile{tile}_batch{column_batch}",
        "configuration": {
            "n_frames": int(n),
            "patch_hw": [int(patch), int(patch)],
            "tile_size": int(tile),
            "channels": int(channels),
            "requested_column_batch": int(column_batch),
            "effective_column_batch": int(pricing["ram"]["effective_column_batch"]),
            "batch_positions": int(pricing["ram"]["batch_positions"]),
            "rows_per_batch": int(pricing["ram"]["rows_per_batch"]),
            "cols_per_batch": int(pricing["ram"]["cols_per_batch"]),
            "interior_hw": [
                int(host["interior_hw"][0]),
                int(host["interior_hw"][1]),
            ],
            "sample_interval_s": float(sample_interval_s),
        },
        "model": {
            "peak_bytes": predicted,
            "peak_phase": max(host["phases"], key=lambda k: host["phases"][k]),
            "phases": {k: int(v) for k, v in host["phases"].items()},
            "terms": em._terms(host),
            "priced_domain": host.get("priced_domain"),
            "named_mapped_spool_bytes_per_tile_read": named_mapped,
            "named_mapped_spool_bytes_per_tile_write": int(
                host.get("mapped_spool_bytes_per_tile_write", 0)
            ),
            "mapped_pages_in_host_peak": int(
                host.get("mapped_pages_in_host_peak", 0)
            ),
        },
        "observed": observed,
        "observed_deltas": deltas,
        "attribution": attribution,
        "shares_of_total_peak": shares,
        "comparison": {
            "ratio_total_peak_over_model": (
                None if total_at_peak is None
                else float(total_at_peak) / predicted
            ),
            "ratio_owned_peak_over_model": (
                None if owned_at_peak is None
                else float(owned_at_peak) / predicted
            ),
            "ratio_owned_peak_over_reduce_price": (
                None if owned_at_peak is None or reduce_price == 0
                else float(owned_at_peak) / float(reduce_price)
            ),
            "owned_mib": _mib(owned_at_peak),
            "file_backed_mib": _mib(file_at_peak),
            "total_mib": _mib(total_at_peak),
            "model_mib": _mib(predicted),
        },
        "elapsed_s": round(float(elapsed), 3),
        "missing_instruments": list(observed.get("missing_instruments") or []),
        "capability": capabilities,
    }


def run_profile(profile: str, *, json_path=None) -> dict:
    config = dict(PROFILES[profile])
    if profile == "shaped" and os.environ.get(_SLOW_ENV) != "1":
        raise SystemExit(
            f"profile {profile!r} is gated: set {_SLOW_ENV}=1 to run it"
        )
    result = measure_cell(**config)
    artefact = {
        "schema": SCHEMA,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "result": result,
    }
    if json_path is not None:
        Path(json_path).write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    return artefact


def _print(result: dict) -> None:
    a = result["attribution"]
    m = result["model"]
    o = result["observed"]
    print(f"configuration: {result['configuration']}")
    print(
        f"model: peak {m['peak_bytes'] / MIB:.1f} MiB ({m['peak_phase']}), "
        f"priced domain {m['priced_domain']}"
    )
    print(
        f"  named mapped spool surface: read "
        f"{m['named_mapped_spool_bytes_per_tile_read'] / MIB:.1f} MiB, write "
        f"{m['named_mapped_spool_bytes_per_tile_write'] / MIB:.1f} MiB "
        f"(in host peak: {m['mapped_pages_in_host_peak']})"
    )
    print(
        f"observed: total {_mib(a['total_peak_bytes'])} MiB = owned "
        f"{_mib(a['owned_bytes'])} + file-backed {_mib(a['file_backed_bytes'])}"
        f" + unexplained {_mib(a['unexplained_bytes'])}"
    )
    print(
        f"  owned peak (independent max, Δ) "
        f"{_mib(result['observed_deltas']['owned_delta_bytes'])} MiB, "
        f"file-backed peak Δ "
        f"{_mib(result['observed_deltas']['file_backed_delta_bytes'])} MiB"
    )
    print(
        f"  priced live set {a['priced_live_bytes_at_reduce'] / MIB:.1f} MiB, "
        f"retention/transient {_mib(a['allocator_retention_bytes'])} MiB, "
        f"mapped residual {_mib(a['mapped_residual_bytes'])} MiB"
    )
    print("phases:")
    for label, slot in (o.get("phases") or {}).items():
        print(
            f"  {label:<10} elapsed {slot.get('elapsed_s')}s "
            f"total {_mib(slot.get('total_peak_bytes'))} "
            f"owned {_mib(slot.get('owned_peak_bytes'))} "
            f"file {_mib(slot.get('file_backed_peak_bytes'))} MiB"
        )
    print(f"shares: {result['shares_of_total_peak']}")
    print(f"missing instruments: {result['missing_instruments']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default="bounded", choices=sorted(PROFILES))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    artefact = run_profile(args.profile, json_path=args.json)
    _print(artefact["result"])
    if args.json:
        print(f"\nartefact: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
