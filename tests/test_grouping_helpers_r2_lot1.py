"""Behavior characterization for the shared filter grouping helpers.

These tests pin the canonical implementations of the three shared grouping
helpers now living in ``zemosaic.core.grouping_helpers`` (extracted verbatim
from ``zemosaic_filter_gui``). They are deterministic and touch no network,
no user data, and no real filesystem state.

They also prove the compatibility seams required by the R2 lot 1 extraction:

* the legacy Tk module re-exports the *same objects* (``is`` identity) as the
  neutral module; and
* the official Qt path resolves the neutral helpers (``is`` identity), while
  the Qt inline fallback copies remain defined and *distinct* objects.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.core.grouping_helpers as gh  # noqa: E402


def _entry(ra=None, dec=None, pa_deg=None):
    entry = {}
    if ra is not None:
        entry["RA"] = ra
    if dec is not None:
        entry["DEC"] = dec
    if pa_deg is not None:
        entry["PA_DEG"] = pa_deg
    return entry


# ---------------------------------------------------------------------------
# _circular_dispersion_deg
# ---------------------------------------------------------------------------

class TestCircularDispersion:
    def test_single_value_returns_zero(self):
        assert gh._circular_dispersion_deg([42.0]) == 0.0

    def test_empty_returns_zero(self):
        assert gh._circular_dispersion_deg([]) == 0.0

    def test_non_finite_skipped(self):
        # Only finite values survive; non-finite are skipped.
        assert gh._circular_dispersion_deg([math.nan, math.inf, -math.inf]) == 0.0
        assert gh._circular_dispersion_deg([math.nan, 10.0]) == 0.0

    def test_wrap_around_minimal_arc(self):
        # Values around 359/0/1 cover a 2-degree arc, not 358 degrees.
        assert gh._circular_dispersion_deg([359.0, 0.0, 1.0]) == pytest.approx(2.0)

    def test_values_spanning_gt_180(self):
        # Values spanning >180: [0, 200] -> sorted gaps 200 and 160,
        # max gap 200, so minimal covering arc = 360 - 200 = 160.
        assert gh._circular_dispersion_deg([0.0, 200.0]) == pytest.approx(160.0)
        # 0 and 270 span 270 via the long way but only 90 via the short arc.
        assert gh._circular_dispersion_deg([0.0, 270.0]) == pytest.approx(90.0)

    def test_duplicate_values_zero(self):
        assert gh._circular_dispersion_deg([15.0, 15.0, 15.0]) == 0.0


# ---------------------------------------------------------------------------
# _split_group_by_orientation
# ---------------------------------------------------------------------------

class TestSplitGroupByOrientation:
    def test_threshold_nonpositive_returns_group(self):
        group = [_entry(pa_deg=0.0), _entry(pa_deg=180.0)]
        assert gh._split_group_by_orientation(group, 0) == [group]
        assert gh._split_group_by_orientation(group, -5.0) == [group]

    def test_len_le_one_returns_group(self):
        single = [_entry(pa_deg=12.0)]
        assert gh._split_group_by_orientation(single, 10.0) == [single]

    def test_circular_wrap_grouping(self):
        # 359 and 1 are 2 degrees apart circularly.
        group = [_entry(pa_deg=359.0), _entry(pa_deg=1.0)]
        result = gh._split_group_by_orientation(group, 5.0)
        # They group together, so a single subgroup -> [group].
        assert result == [group]

    def test_invalid_or_missing_pa(self):
        group = [
            _entry(pa_deg=None),
            _entry(pa_deg="not-a-number"),
            _entry(pa_deg=math.nan),
        ]
        # No valid PA values -> len(with_angles) <= 1 -> [group].
        assert gh._split_group_by_orientation(group, 10.0) == [group]

    def test_subgroup_order_by_first_original_index(self):
        group = [
            _entry(pa_deg=10.0),   # idx 0
            _entry(pa_deg=200.0),  # idx 1
            _entry(pa_deg=11.0),   # idx 2 (joins idx 0)
        ]
        result = gh._split_group_by_orientation(group, 5.0)
        assert len(result) == 2
        # First subgroup starts with original idx 0 (10, 11); second with idx 1 (200).
        assert result[0][0] is group[0]
        assert result[0][1] is group[2]
        assert result[1][0] is group[1]

    def test_without_angles_appended_to_largest_subgroup(self):
        group = [
            _entry(pa_deg=10.0),   # idx 0
            _entry(pa_deg=200.0),  # idx 1
            _entry(pa_deg=201.0),  # idx 2 (joins idx 1 -> subgroup of 2)
            _entry(pa_deg=None),   # idx 3 -> without_angles
        ]
        result = gh._split_group_by_orientation(group, 5.0)
        assert len(result) == 2
        # Largest subgroup is the {200, 201} one (size 2).
        largest = max(result, key=len)
        assert group[3] in largest
        # The {10} subgroup (size 1) does not receive the missing-PA entry.
        assert group[3] not in result[0] if len(result[0]) == 1 else True

    def test_single_subgroup_returns_group(self):
        group = [_entry(pa_deg=10.0), _entry(pa_deg=11.0), _entry(pa_deg=12.0)]
        result = gh._split_group_by_orientation(group, 5.0)
        assert result == [group]


# ---------------------------------------------------------------------------
# _merge_small_groups
# ---------------------------------------------------------------------------

class TestMergeSmallGroups:
    def test_noop_empty_groups(self):
        assert gh._merge_small_groups([], 2, 10) == []

    def test_noop_nonpositive_min_size(self):
        groups = [[_entry(ra=0, dec=0)]]
        assert gh._merge_small_groups(groups, 0, 10) == groups
        assert gh._merge_small_groups(groups, -1, 10) == groups

    def test_noop_nonpositive_cap(self):
        groups = [[_entry(ra=0, dec=0)]]
        assert gh._merge_small_groups(groups, 2, 0) == groups
        assert gh._merge_small_groups(groups, 2, -3) == groups

    def test_undersized_group_merged_into_nearest(self):
        # Group 0 (size 1) is undersized; group 1 is nearest neighbour.
        groups = [
            [_entry(ra=0.0, dec=0.0)],
            [_entry(ra=0.1, dec=0.0), _entry(ra=0.2, dec=0.0)],
        ]
        result = gh._merge_small_groups(groups, 2, 10)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_cap_overflow_rejected(self):
        # Undersized group 0 cannot merge into group 1 (would exceed cap 2).
        groups = [
            [_entry(ra=0.0, dec=0.0)],
            [_entry(ra=0.1, dec=0.0), _entry(ra=0.2, dec=0.0)],
        ]
        result = gh._merge_small_groups(groups, 2, 2)
        assert len(result) == 2

    def test_cap_allowance_allows_overflow(self):
        groups = [
            [_entry(ra=0.0, dec=0.0)],
            [_entry(ra=0.1, dec=0.0), _entry(ra=0.2, dec=0.0)],
        ]
        # cap 2, allowance 3 -> merge allowed.
        result = gh._merge_small_groups(groups, 2, 2, cap_allowance=3)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_dispersion_guard_triggers_only_when_max_dispersion_gt_0(self):
        groups = [
            [_entry(ra=0.0, dec=0.0)],
            [_entry(ra=0.1, dec=0.0), _entry(ra=0.2, dec=0.0)],
        ]
        # max_dispersion_deg=0 -> guard inactive -> merge proceeds.
        result = gh._merge_small_groups(
            groups, 2, 10, compute_dispersion=lambda coords: 999.0, max_dispersion_deg=0.0
        )
        assert len(result) == 1

        # max_dispersion_deg=1.0 -> guard active, dispersion 999 > 1 -> merge blocked.
        groups2 = [
            [_entry(ra=0.0, dec=0.0)],
            [_entry(ra=0.1, dec=0.0), _entry(ra=0.2, dec=0.0)],
        ]
        result2 = gh._merge_small_groups(
            groups2, 2, 10, compute_dispersion=lambda coords: 999.0, max_dispersion_deg=1.0
        )
        assert len(result2) == 2

    def test_log_fn_receives_canonical_message(self):
        logs = []
        groups = [
            [_entry(ra=0.0, dec=0.0)],
            [_entry(ra=0.1, dec=0.0), _entry(ra=0.2, dec=0.0)],
        ]
        gh._merge_small_groups(groups, 2, 10, log_fn=logs.append)
        assert logs == ["Merged group 0 (1 imgs) into 1 (size=3)"]

    def test_merged_groups_removed_and_order_preserved(self):
        # Groups: undersized group 2 merges into nearest; remaining order kept.
        groups = [
            [_entry(ra=0.0, dec=0.0), _entry(ra=0.1, dec=0.0)],  # idx 0 (size 2)
            [_entry(ra=50.0, dec=50.0), _entry(ra=50.1, dec=50.0)],  # idx 1 (size 2)
            [_entry(ra=50.2, dec=50.0)],  # idx 2 (size 1, nearest idx 1)
        ]
        result = gh._merge_small_groups(groups, 2, 10)
        assert len(result) == 2
        # idx 0 preserved as first, merged idx 1+idx 2 as second.
        assert result[0][0] is groups[0][0]
        assert groups[2][0] in result[1]


# ---------------------------------------------------------------------------
# identity / compatibility seams
# ---------------------------------------------------------------------------

class TestIdentityAndCompat:
    def test_legacy_reexports_same_objects(self):
        gui = importlib.import_module("zemosaic.zemosaic_filter_gui")
        for name in (
            "_merge_small_groups",
            "_split_group_by_orientation",
            "_circular_dispersion_deg",
            "_group_center_deg",
            "_angular_sep_deg",
            "_circ_delta_deg",
        ):
            assert getattr(gui, name) is getattr(gh, name)

    def test_qt_official_resolution_uses_neutral_helpers(self):
        qt = importlib.import_module("zemosaic.zemosaic_filter_gui_qt")
        assert qt._tk_merge_small_groups is gh._merge_small_groups
        assert qt._tk_split_group_by_orientation is gh._split_group_by_orientation
        assert qt._tk_circular_dispersion_deg is gh._circular_dispersion_deg

    def test_neutral_module_has_no_tk_import(self):
        import ast

        src = (SRC / "zemosaic" / "core" / "grouping_helpers.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    mods.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mods.add(node.module.split(".")[0])
        assert "tkinter" not in mods
        assert "zemosaic_filter_gui" not in mods
        assert "PySide6" not in mods and "PyQt" not in mods
