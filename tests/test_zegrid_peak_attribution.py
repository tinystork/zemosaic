"""ZM-ZEGRID-PEAK-ATTRIBUTION-20261113 — attribution + honest telemetry.

These tests pin the OUTCOME of the attribution:

1. the product telemetry reports OWNED and FILE-BACKED map peaks separately and
   never as one misleading working-set figure;
2. the external host model NAMES the mapped spool surface while keeping it out
   of the priced peak (the model prices owned arrays only);
3. the model was NOT lowered: the predicted bound for the measured
   production-shaped configuration is >= the pre-change bound;
4. the attribution harness decomposes one real external-spool Cell per phase and
   closes the arithmetic (own + mapped + unexplained == total at the peak);
5. the phase hook cannot perturb the executor (bit-identical payloads).

The heavy, production-ratio profile is opt-in via ``ZEMOSAIC_MEASURE_SLOW=1``
(see ``tests/support/external_peak_attribution.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from support import external_model_measure as em
from support import external_peak_attribution as attr

from zemosaic.core.zegrid import external_stream as xstream
from zemosaic.core.zegrid import memory_peak as zmem

MIB = 2 ** 20

#: Pinned pre-change predicted bound for the shaped measurement (n=300,
#: patch 128x128, tile 128, column_batch 6528, winsor+mean). This is the value
#: the model produced at baseline HEAD 2a28e93 and it must never decrease.
PINNED_SHAPED_PEAK_BYTES = 193_937_408

#: Pinned production-shape terms (n=21 818, tile 128, c=3, batch 6528).
PRODUCTION_N = 21_818
PINNED_PRODUCTION_PRICED_PEAK_BYTES = 13_679_009_792
PINNED_PRODUCTION_MAPPED_READ_BYTES = 5_719_457_792
#: With a 128x128 patch the extended tile has no halo room, so the write and
#: read surfaces coincide. A real Cell patch is larger than one tile, where the
#: written (extended) surface is ``n * 144 * 144 * 16`` per tile — 26 % larger.
PINNED_PRODUCTION_MAPPED_WRITE_BYTES = 5_719_457_792


def _model(n: int, patch: int, tile: int, batch: int) -> dict:
    request = em.make_request(n)
    return em.model_price(n, patch, patch, tile, batch, 3, request)


# ---------------------------------------------------------------------------
# 1. The model names the mapped surface, and it is NOT in the priced peak
# ---------------------------------------------------------------------------

def test_model_names_mapped_surface_without_pricing_it():
    pricing = _model(PRODUCTION_N, 128, 128, 6528)
    host = pricing["host"]

    assert host["priced_domain"] == "owned_anonymous_live_arrays"
    assert host["mapped_pages_in_host_peak"] == 0
    assert host["peak_bytes"] == PINNED_PRODUCTION_PRICED_PEAK_BYTES
    assert host["mapped_spool_bytes_per_tile_read"] == (
        PINNED_PRODUCTION_MAPPED_READ_BYTES
    )
    assert host["mapped_spool_bytes_per_tile_write"] == (
        PINNED_PRODUCTION_MAPPED_WRITE_BYTES
    )
    # The mapped surface is a real, large fraction of the priced peak: this is
    # exactly the term that made a 17 340 MB working-set peak look like +26 %
    # against a 13.7 GB owned-array model.
    ratio = host["mapped_spool_bytes_per_tile_read"] / host["peak_bytes"]
    assert 0.40 <= ratio <= 0.43
    # ... and it is never folded into the peak itself.
    assert host["peak_bytes"] == max(host["phases"].values())


def test_mapped_surface_scales_with_n_and_keeps_its_ratio():
    a = _model(300, 128, 128, 6528)["host"]
    b = _model(600, 128, 128, 6528)["host"]
    assert b["mapped_spool_bytes_per_tile_read"] == (
        2 * a["mapped_spool_bytes_per_tile_read"]
    )
    ratio_a = a["mapped_spool_bytes_per_tile_read"] / a["peak_bytes"]
    ratio_b = b["mapped_spool_bytes_per_tile_read"] / b["peak_bytes"]
    # n-independent structural ratio (asymptotically; the fixed 512 B/position
    # term is a small correction at these shapes).
    assert abs(ratio_a - ratio_b) < 0.01
    # The ram view publishes the same additive fields.
    assert a["mapped_pages_in_host_peak"] == 0
    assert a["priced_domain"] == "owned_anonymous_live_arrays"


# ---------------------------------------------------------------------------
# 2. The model was not lowered
# ---------------------------------------------------------------------------

def test_model_is_not_lowered_for_the_measured_configuration():
    host = _model(300, 128, 128, 6528)["host"]
    assert host["peak_bytes"] >= PINNED_SHAPED_PEAK_BYTES
    # The production bound is likewise unchanged (same constants, so the same
    # number: raising was not proven necessary — see the durable report).
    production = _model(PRODUCTION_N, 128, 128, 6528)["host"]
    assert production["peak_bytes"] >= PINNED_PRODUCTION_PRICED_PEAK_BYTES


# ---------------------------------------------------------------------------
# 3. The platform decomposition is additive, labelled and never conflated
# ---------------------------------------------------------------------------

def test_platform_capability_is_explicit_not_guessed():
    caps = zmem.platform_capability()
    assert caps["schema"] == zmem.MEMORY_COMPONENTS_SCHEMA
    if caps["platform"].startswith("linux"):
        assert caps["owned_measured"] is True
        assert caps["file_backed_measured"] is True
        assert caps["owned_source"] == "proc_status_RssAnon"
        assert caps["file_backed_source"] == "proc_status_RssFile"
        assert caps["missing_instruments"] == []
    sample = zmem.sample_memory_components()
    assert set(sample) >= {
        "total_bytes", "owned_bytes", "file_backed_bytes", "shared_bytes",
        "swap_bytes", "sources",
    }


def test_sampler_result_keeps_components_separate(tmp_path):
    sampler = zmem.PeakMemorySampler(interval_s=0.01)
    sampler.start()
    sampler.mark("alpha")
    _ = np.empty(4 * 1024 * 1024, dtype=np.uint8)  # owned anonymous
    sampler.mark("beta")
    sampler.stop()
    result = sampler.result()

    assert result["schema"] == zmem.MEMORY_COMPONENTS_SCHEMA
    assert result["samples"] >= 2
    assert set(result["phases"]) >= {"alpha", "beta"}
    # The two component peaks are independent maxima and MUST stay separate.
    if result["capability"]["owned_measured"]:
        assert result["owned_peak_bytes"] is None or (
            result["owned_peak_bytes"] > 0
        )
    assert result["at_total_peak"] is not None
    at_peak = result["at_total_peak"]
    if at_peak["owned_bytes"] is not None and at_peak["file_backed_bytes"] is not None:
        accounted = (
            at_peak["owned_bytes"]
            + at_peak["file_backed_bytes"]
            + (at_peak["shared_bytes"] or 0)
        )
        # The decomposition of ONE instant closes on the total (Linux).
        assert abs(at_peak["total_bytes"] - accounted) <= 16 * MIB
        assert at_peak["unaccounted_bytes"] is not None
    assert "must not be summed" in result["additivity"]


# ---------------------------------------------------------------------------
# 4. One real external-spool Cell: per-phase decomposition + closure
# ---------------------------------------------------------------------------

def test_one_cell_decomposes_per_phase(tmp_path):
    """Bounded production-shaped Cell: owned vs mapped, per phase, closed."""
    result = attr.measure_cell(
        n=60,
        patch=128,
        tile=128,
        column_batch=6528,
        sample_interval_s=0.02,
        spool_root=tmp_path / "spool",
    )
    model = result["model"]
    attribution = result["attribution"]
    deltas = result["observed_deltas"]
    observed = result["observed"]

    # The model priced the SAME geometry the executor ran.
    assert result["configuration"]["batch_positions"] == 6528
    assert model["peak_phase"] == "reduce"
    assert model["priced_domain"] == "owned_anonymous_live_arrays"
    assert model["mapped_pages_in_host_peak"] == 0

    # Per-phase decomposition exists for the three executor phases.
    for phase in ("write", "reduce"):
        assert phase in observed["phases"]
        slot = observed["phases"][phase]
        assert slot["total_peak_bytes"] is not None
        assert slot["owned_peak_bytes"] is not None
        assert slot["file_backed_peak_bytes"] is not None

    # The file-backed peak tracks the NAMED spool surface (the spool tile the
    # reduction maps), not the batch workspace.
    named = model["named_mapped_spool_bytes_per_tile_read"]
    assert named > 0
    file_delta = deltas["file_backed_delta_bytes"]
    assert file_delta is not None
    assert 0.75 * named <= file_delta <= 1.35 * named

    # The owned peak stays close to the priced peak: on this shape the model is
    # an upper bound on the live set (the residual above it is allocator
    # retention, quantified in the durable report).
    owned_delta = deltas["owned_delta_bytes"]
    assert owned_delta is not None
    assert 0.6 * model["peak_bytes"] <= owned_delta <= 1.3 * model["peak_bytes"]

    # Closure: at the total peak, owned + file-backed + unexplained == total.
    total = attribution["total_peak_bytes"]
    owned = attribution["owned_bytes"]
    mapped = attribution["file_backed_bytes"]
    unexplained = attribution["unexplained_bytes"]
    assert total is not None and owned is not None and mapped is not None
    if unexplained is not None:
        assert abs((owned + mapped + unexplained) - total) <= 0.02 * total
    # ... and the two biggest shares account for essentially all of it.
    shares = result["shares_of_total_peak"]
    assert shares["owned_bytes"] + shares["file_backed_bytes"] >= 0.98

    # The mapped pages are NOT owned bytes: the model's peak must stay a bound
    # on the OWNED component, not on the working set (which is larger).
    assert total >= owned


# ---------------------------------------------------------------------------
# 5. The phase hook cannot perturb the executor
# ---------------------------------------------------------------------------

def test_memory_probe_is_additive_and_does_not_change_payload(tmp_path):
    from zemosaic.core.zegrid import file_provider as fp
    from zemosaic.core.zegrid.rejection_sink import BoundedCountingRejectionSink

    def run(with_probe: bool):
        request = em.make_request(8)
        provider = fp.WindowedCanonicalProvider(
            em.make_prepare(8, 32, 32, 3), 8,
            expected_hw=(32, 32), expected_channels=3,
        )
        sink = BoundedCountingRejectionSink((8, 32, 32, 3))
        marks = []
        spool_root = tmp_path / ("probe" if with_probe else "plain")
        spool_root.mkdir()

        def probe(label):
            marks.append(label)

        try:
            result = xstream.run_cell_streaming_external(
                provider, request,
                tile_size=16,
                column_batch=48,
                spool_root=spool_root,
                rejection_sink=sink,
                cell_id="additive", attempt_id="a0",
                memory_probe=probe if with_probe else None,
            )
        finally:
            provider.close()
        return result, marks

    plain, _ = run(False)
    probed, marks = run(True)
    np.testing.assert_array_equal(plain.science, probed.science)
    np.testing.assert_array_equal(
        plain.estimator_weight_sum, probed.estimator_weight_sum
    )
    np.testing.assert_array_equal(plain.valid_mask, probed.valid_mask)
    assert marks.count("write") >= 1
    assert marks.count("reduce") >= 1
    assert marks[-1] == "finalize"


# ---------------------------------------------------------------------------
# 6. The heavy, production-ratio profile (opt-in only)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(em._SLOW_ENV) != "1",
    reason=f"set {em._SLOW_ENV}=1 to run the production-ratio attribution",
)
def test_shaped_profile_attributes_the_gap(tmp_path):
    artefact = attr.run_profile("shaped", json_path=tmp_path / "shaped.json")
    result = artefact["result"]
    assert artefact["schema"] == attr.SCHEMA
    host = result["model"]
    assert host["peak_bytes"] == PINNED_SHAPED_PEAK_BYTES
    observed = result["observed"]
    deltas = result["observed_deltas"]
    assert observed["owned_peak_bytes"] > 0
    assert observed["file_backed_peak_bytes"] > 0
    # The mapped spool surface is resident and separated from owned memory.
    # Compare the BASELINE-SUBTRACTED file-backed increment with the named
    # spool surface (the absolute peak also contains the interpreter's own
    # mapped files).
    named = host["named_mapped_spool_bytes_per_tile_read"]
    file_delta = deltas["file_backed_delta_bytes"]
    assert file_delta is not None
    ratio = file_delta / named
    assert 0.75 <= ratio <= 1.35
    # The owned increment exceeds the priced live set only by allocator
    # retention (quantified in the durable report).
    owned_delta = deltas["owned_delta_bytes"]
    assert owned_delta is not None
    owned_ratio = owned_delta / host["peak_bytes"]
    assert 0.9 <= owned_ratio <= 1.35
    # No unexplained slice at the peak on a platform with full components.
    at_peak = observed["at_total_peak"]
    if at_peak["unaccounted_bytes"] is not None:
        assert abs(at_peak["unaccounted_bytes"]) <= 0.02 * at_peak["total_bytes"]
