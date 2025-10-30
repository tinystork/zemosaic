"""Unit tests for dynamic footprint budget calculations in the filter preview."""

from zemosaic_filter_gui import _compute_dynamic_footprint_budget


def test_budget_within_small_dataset_uses_preview_cap():
    budget = _compute_dynamic_footprint_budget(100, 200, max_footprints=3000)
    assert budget == 200


def test_budget_medium_dataset_restricts_to_400():
    budget = _compute_dynamic_footprint_budget(1500, 2500, max_footprints=3000)
    assert budget == 400


def test_budget_large_dataset_forces_centroids_only():
    budget = _compute_dynamic_footprint_budget(5000, 5000, max_footprints=3000)
    assert budget == 0


def test_budget_handles_none_preview_cap():
    budget = _compute_dynamic_footprint_budget(800, None, max_footprints=3000)
    assert budget == 1500


def test_budget_negative_preview_cap_falls_back_to_zero():
    budget = _compute_dynamic_footprint_budget(800, -50, max_footprints=3000)
    assert budget == 0
