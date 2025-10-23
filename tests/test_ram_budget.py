import pathlib
import sys

import pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from zemosaic_worker import _auto_split_groups, _compute_auto_tile_caps


def _make_header(index: int) -> dict:
    return {"DATE-OBS": f"2023-01-01T00:00:{index:02d}"}


def test_auto_caps_uses_ram_budget_and_minimum():
    resource = {"usable_ram_mb": 1200.0, "ram_available_mb": 1500.0, "disk_free_mb": 2000.0}
    per_frame = {"per_frame_mb": 50.0}

    caps = _compute_auto_tile_caps(resource, per_frame, policy_max=50, policy_min=8)

    assert caps["cap"] == 24  # floor(1200 / 50)
    assert caps["frames_by_ram"] == 24
    assert not caps["memmap"]


def test_auto_caps_enables_memmap_when_ram_tight():
    resource = {"usable_ram_mb": 200.0, "ram_available_mb": 250.0, "disk_free_mb": 20000.0}
    per_frame = {"per_frame_mb": 80.0}

    caps = _compute_auto_tile_caps(resource, per_frame, policy_max=50, policy_min=8)

    assert caps["cap"] == 8  # respects minimum when frames_by_ram < 8
    assert caps["memmap"] is True


def test_auto_split_groups_temporal_chunks():
    group = [{"header": _make_header(i), "phase0_index": i} for i in range(10)]

    result = _auto_split_groups([group], cap=3, min_cap=2)
    sizes = [len(g) for g in result]
    assert sizes == [3, 3, 3, 1]


def test_auto_split_groups_spatial_prefers_clusters():
    group = []
    for i in range(4):
        group.append({
            "header": _make_header(i),
            "phase0_center": (0.0, 0.0),
            "phase0_fov_deg": 1.0,
            "phase0_index": i,
        })
    for i in range(4, 8):
        group.append({
            "header": _make_header(i),
            "phase0_center": (10.0, 0.0),
            "phase0_fov_deg": 1.0,
            "phase0_index": i,
        })

    result = _auto_split_groups([group], cap=4, min_cap=2)
    assert [len(g) for g in result] == [4, 4]
    # Ensure chronological order preserved within subgroups
    assert [g[0]["phase0_index"] for g in result] == [0, 4]
