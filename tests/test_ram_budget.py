import math
import pathlib
import sys

import pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from zemosaic_worker import _apply_ram_budget_to_groups


def _make_info(idx: int, height: int = 100, width: int = 100) -> dict:
    return {
        "path_raw": f"frame_{idx}.fits",
        "preprocessed_shape": (height, width, 3),
    }


def _stub_cluster(groups_to_return):
    """Return a clustering stub that yields ``groups_to_return`` when threshold is low enough."""

    def _cluster(group, threshold, _callback, orientation_split_threshold_deg=0.0):
        return groups_to_return(threshold, group)

    return _cluster


def test_ram_budget_triggers_temporal_split():
    group = [_make_info(i) for i in range(10)]
    per_frame_bytes = 100 * 100 * 4
    budget_bytes = per_frame_bytes * 3  # Force split into batches of 3 frames

    new_groups, adjustments = _apply_ram_budget_to_groups(
        [group],
        budget_bytes,
        base_threshold_deg=0.5,
        orientation_split_threshold_deg=0.0,
        cluster_func=_stub_cluster(lambda _thr, g: [g]),
    )

    # Expect four groups: 3 + 3 + 3 + 1 frames
    sizes = sorted(len(g) for g in new_groups)
    assert sizes == [1, 3, 3, 3]
    split_adjustments = [adj for adj in adjustments if adj.get("method") == "split"]
    assert split_adjustments, "Temporal split adjustment should be recorded"
    assert split_adjustments[0]["segment_size"] == 3


def test_ram_budget_recluster_before_split():
    group = [_make_info(i) for i in range(4)]
    per_frame_bytes = 100 * 100 * 4
    budget_bytes = per_frame_bytes * 2  # Enough for two frames, not four

    def _groups_for_threshold(threshold, g):
        if threshold < 0.2:
            mid = len(g) // 2
            return [g[:mid], g[mid:]]
        return [g]

    new_groups, adjustments = _apply_ram_budget_to_groups(
        [group],
        budget_bytes,
        base_threshold_deg=0.5,
        orientation_split_threshold_deg=0.0,
        cluster_func=_stub_cluster(_groups_for_threshold),
    )

    assert [len(g) for g in new_groups] == [2, 2]
    recluster_events = [adj for adj in adjustments if adj.get("method") == "recluster"]
    assert recluster_events, "Recluster adjustment should be recorded"
    assert float(recluster_events[0]["new_threshold_deg"]) < 0.2


def test_ram_budget_single_frame_over_budget():
    group = [_make_info(0, height=1000, width=1000)]
    total_bytes = 1000 * 1000 * 4
    budget_bytes = total_bytes // 4  # Smaller than per-frame requirement

    new_groups, adjustments = _apply_ram_budget_to_groups(
        [group],
        budget_bytes,
        base_threshold_deg=0.5,
        orientation_split_threshold_deg=0.0,
        cluster_func=_stub_cluster(lambda _thr, g: [g]),
    )

    assert len(new_groups) == 1
    single_over = [adj for adj in adjustments if adj.get("method") == "single_over_budget"]
    assert single_over, "Single-frame over-budget warning should be recorded"
    assert math.isclose(single_over[0]["estimated_mb"], total_bytes / (1024 ** 2), rel_tol=1e-6)
