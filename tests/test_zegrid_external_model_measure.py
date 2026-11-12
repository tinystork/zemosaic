"""Bounded regression for the external-model MEASUREMENT harness.

ZM-ZEGRID-EXTERNAL-MODEL-MEASURE (2026-11-12).  These tests prove the
measurement PATH works (predicted and observed are both recorded, the artefact
is well-formed); they are NOT a claim about the model's accuracy — that lives
in the durable report and its JSON artefact.

The heavy, realistic profiles are opt-in:

* ``pytest tests/test_zegrid_external_model_measure.py -q`` runs the fast tier;
* ``ZEMOSAIC_MEASURE_SLOW=1 pytest -m slow tests/test_zegrid_external_model_measure.py``
  runs the standard profile (N=300, patch 256x256, tile 128, 4 batches;
  several minutes);
* the harness CLI runs any profile directly, e.g.::

      python tests/support/external_model_measure.py --profile standard \\
          --json /tmp/external-model-standard.json

Nothing here is imported by ``src/``: the harness is a witness, not product code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from support import external_model_measure as em

REPO = Path(__file__).resolve().parents[1]


def test_measure_one_records_prediction_and_peak(tmp_path):
    """Bounded configuration: both sides recorded, comparison well-formed."""
    result = em.measure_one(
        n=12,
        patch=32,
        tile=16,
        batch=96,
        spool_root=tmp_path / "spool",
        sample_interval_s=0.02,
    )

    assert result["configuration"]["n_frames"] == 12
    assert result["configuration"]["tile_size"] == 16
    assert result["configuration"]["batch_positions"] >= 1
    # The executor really resolved the requested batch (no silent clamp here).
    assert result["configuration"]["column_batch_clamped"] is False

    model = result["model"]
    assert model["peak_bytes"] > 0
    assert model["peak_phase"] in model["phases"]
    assert model["dominant_term"] in model["terms"]
    # The batch term is the largest priced contributor for this shape.
    assert model["peak_bytes"] >= model["terms"]["reduction_batch_bytes"]

    observed = result["observed"]
    assert observed["peak_rss_bytes"] > 0
    assert observed["baseline_rss_bytes"] > 0
    assert observed["delta_peak_bytes"] >= 0
    assert observed["sampler_samples"] >= 1
    assert observed["sample_interval_s"] == 0.02

    comparison = result["comparison"]
    assert comparison["ratio_observed_delta_over_predicted"] == pytest.approx(
        observed["delta_peak_bytes"] / model["peak_bytes"], rel=1e-9
    )
    assert comparison["verdict"] in {
        "model_over_estimates",
        "model_tracks_reality",
        "model_under_estimates",
    }
    # JSON round-trip: the artefact is machine-readable end to end.
    assert json.loads(json.dumps(result)) == result


def test_matrix_artefact_is_well_formed(tmp_path):
    """The child-process matrix path records every configuration."""
    out = tmp_path / "quick.json"
    artefact = em.run_matrix(
        "quick",
        json_path=out,
        sample_interval_s=0.05,
        profiles={
            "quick": {
                "n_frames": 8,
                "patch": 32,
                "tile": 16,
                "batches": [24, 96, 576, 1536],
            }
        },
    )

    assert out.is_file()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["schema"] == em.SCHEMA
    assert saved["verdict"]["verdict"] in {
        "model_over_estimates",
        "model_tracks_reality",
        "model_under_estimates",
    }
    assert len(saved["configurations"]) == 4
    positions = [c["configuration"]["batch_positions"] for c in saved["configurations"]]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)  # spanning distinct batches
    for cfg in saved["configurations"]:
        assert cfg["model"]["peak_bytes"] > 0
        assert cfg["observed"]["peak_rss_bytes"] > 0
        assert cfg["observed"]["sampler_samples"] >= 1
        assert cfg["comparison"]["verdict"] in saved["verdict"]["per_configuration_verdicts"].values()
    # Linear fit over >= 2 points must be available.
    assert saved["extrapolation"]["status"] == "ok"


def test_classify_ratio_bands():
    assert em.classify_ratio(0.4) == "model_over_estimates"
    assert em.classify_ratio(0.5) == "model_over_estimates"
    assert em.classify_ratio(0.9) == "model_tracks_reality"
    assert em.classify_ratio(1.0) == "model_tracks_reality"
    assert em.classify_ratio(1.45) == "model_tracks_reality"
    assert em.classify_ratio(1.5) == "model_under_estimates"
    assert em.classify_ratio(1.51) == "model_under_estimates"
    assert em.classify_ratio(6.2) == "model_under_estimates"


def test_extrapolate_recovers_known_line():
    results = [
        {
            "label": f"p{pos}",
            "configuration": {"batch_positions": pos},
            "observed": {"delta_peak_bytes": 10 * 2 ** 20 + 1_000 * pos},
            "model": {"terms": {"reduction_batch_bytes": 1_200 * pos}},
        }
        for pos in (1_000, 4_000)
    ]
    out = em.extrapolate(results, target_bytes=10 * 2 ** 20 + 1_000 * 9_000)
    assert out["status"] == "ok"
    assert out["observed_bytes_per_position"] == pytest.approx(1_000.0, rel=1e-9)
    assert out["observed_fixed_bytes"] == pytest.approx(10 * 2 ** 20, rel=1e-6)
    assert out["implied_batch_positions_at_target"] == pytest.approx(9_000.0, rel=1e-3)
    # A single point cannot be fitted.
    assert em.extrapolate(results[:1], target_bytes=1)["status"] == "insufficient_points"


def test_harness_is_not_product_code():
    """The measurement module must never be imported from ``src/``."""
    hits = [
        str(p)
        for p in (REPO / "src").rglob("*.py")
        if "external_model_measure" in p.read_text(encoding="utf-8")
    ]
    assert hits == []


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(em._SLOW_ENV) != "1",
    reason=f"set {em._SLOW_ENV}=1 to run the realistic measurement profile",
)
def test_standard_profile_records_all_configurations(tmp_path):
    """N=300, patch 256x256, tile 128, four batches — minutes, opt-in only."""
    out = tmp_path / "standard.json"
    artefact = em.run_matrix("standard", json_path=out)
    assert len(artefact["configurations"]) == 4
    for cfg in artefact["configurations"]:
        assert cfg["configuration"]["batch_positions"] >= 300
        assert cfg["model"]["peak_bytes"] > 0
        assert cfg["observed"]["delta_peak_bytes"] > 0
        assert cfg["observed"]["elapsed_s"] > 0
