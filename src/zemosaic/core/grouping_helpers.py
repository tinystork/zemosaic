"""Shared filter grouping helpers (neutral module).

This module contains the canonical implementations of the shared filter
grouping helpers, extracted verbatim from ``zemosaic_filter_gui`` so that the
official Qt filter can obtain them without importing the legacy Tk module.

This module must remain free of any tkinter, Qt, or ``zemosaic_filter_gui``
import so it can be imported by either filter front-end.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Callable, Optional


def _group_center_deg(group: list[dict]) -> Optional[tuple[float, float]]:
    """Return the average RA/DEC (degrees) of a group if available."""

    ras: list[float] = []
    decs: list[float] = []
    for info in group:
        ra = info.get("RA")
        dec = info.get("DEC")
        if ra is not None and dec is not None:
            try:
                ras.append(float(ra))
                decs.append(float(dec))
            except Exception:
                continue
    if not ras:
        return None
    return (sum(ras) / len(ras), sum(decs) / len(decs))


def _angular_sep_deg(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> float:
    """Approximate angular separation in degrees between two (RA, DEC) tuples."""

    if not a or not b:
        return 9999.0
    dra = abs(a[0] - b[0])
    ddec = abs(a[1] - b[1])
    return (dra ** 2 + ddec ** 2) ** 0.5


def _merge_small_groups(
    groups: list[list[dict]],
    min_size: int,
    cap: int,
    *,
    cap_allowance: Optional[int] = None,
    compute_dispersion: Optional[Callable[[list[tuple[float, float]]], float]] = None,
    max_dispersion_deg: Optional[float] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[list[dict]]:
    """Merge undersized groups with their nearest neighbour when safe.

    Parameters
    ----------
    groups : list[list[dict]]
        Groups to examine.
    min_size : int
        Minimum size below which a group becomes a merge candidate.
    cap : int
        Hard cap (without allowance) used as base reference.
    cap_allowance : Optional[int]
        Optional absolute cap allowing temporary overflows.
    compute_dispersion : Optional[Callable]
        Callable returning the maximum angular separation (deg) for coordinates.
    max_dispersion_deg : Optional[float]
        Reject merges that would push dispersion beyond this threshold.
    log_fn : Optional[Callable[[str], None]]
        Optional logging callback invoked for each successful merge.
    """

    if not groups or min_size <= 0 or cap <= 0:
        return groups

    cap_limit = int(cap_allowance) if cap_allowance and cap_allowance > 0 else int(cap)
    cap_limit = max(cap_limit, int(cap))

    merged_flags = [False] * len(groups)
    centers = [_group_center_deg(g) for g in groups]

    def _collect_coords(payload: list[dict]) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for info in payload:
            ra = info.get("RA")
            dec = info.get("DEC")
            if ra is None or dec is None:
                continue
            try:
                coords.append((float(ra), float(dec)))
            except Exception:
                continue
        return coords

    for i, group in enumerate(groups):
        if merged_flags[i] or len(group) >= min_size:
            continue

        best_j: Optional[int] = None
        best_dist = float("inf")
        for j, neighbour in enumerate(groups):
            if i == j or merged_flags[j]:
                continue
            dist = _angular_sep_deg(centers[i], centers[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is None:
            continue

        candidate_size = len(groups[best_j]) + len(group)
        if candidate_size > cap_limit:
            continue

        if compute_dispersion is not None and max_dispersion_deg is not None and max_dispersion_deg > 0:
            coords_combined = _collect_coords(groups[best_j]) + _collect_coords(group)
            if coords_combined:
                try:
                    dispersion_val = float(compute_dispersion(coords_combined))
                except Exception:
                    dispersion_val = None
                if dispersion_val is not None and dispersion_val > max_dispersion_deg:
                    continue

        groups[best_j].extend(group)
        merged_flags[i] = True
        centers[best_j] = _group_center_deg(groups[best_j])
        if log_fn is not None:
            try:
                log_fn(
                    f"Merged group {i} ({len(group)} imgs) into {best_j} (size={len(groups[best_j])})"
                )
            except Exception:
                pass

    return [grp for idx, grp in enumerate(groups) if not merged_flags[idx]]


def _circ_delta_deg(a: float, b: float) -> float:
    """Return the absolute circular delta (degrees) between two angles."""

    try:
        delta = abs(float(a) - float(b)) % 360.0
    except Exception:
        return float("inf")
    if delta > 180.0:
        delta = 360.0 - delta
    return delta


def _circular_dispersion_deg(values: Iterable[float]) -> float:
    """Estimate the minimal arc covering ``values`` on the unit circle (deg)."""

    sanitized = []
    for val in values:
        try:
            coerced = float(val)
        except Exception:
            continue
        if math.isnan(coerced) or math.isinf(coerced):
            continue
        sanitized.append(coerced % 360.0)

    if len(sanitized) <= 1:
        return 0.0

    sanitized.sort()
    gaps: list[float] = []
    for i in range(len(sanitized) - 1):
        gaps.append(sanitized[i + 1] - sanitized[i])
    gaps.append((sanitized[0] + 360.0) - sanitized[-1])
    max_gap = max(gaps) if gaps else 0.0
    dispersion = 360.0 - max_gap
    if dispersion < 0.0:
        dispersion = 0.0
    return dispersion


def _split_group_by_orientation(group: list[dict], threshold_deg: float) -> list[list[dict]]:
    """Split ``group`` using circular proximity of ``PA_DEG`` values."""

    if threshold_deg <= 0 or len(group) <= 1:
        return [group]

    with_angles: list[tuple[int, float]] = []
    without_angles: list[int] = []
    for idx, info in enumerate(group):
        try:
            pa_val = info.get("PA_DEG")
        except Exception:
            pa_val = None
        try:
            pa_deg = float(pa_val)
        except Exception:
            pa_deg = None
        if pa_deg is None or math.isnan(pa_deg):
            without_angles.append(idx)
            continue
        with_angles.append((idx, pa_deg % 360.0))

    if len(with_angles) <= 1:
        return [group]

    with_angles.sort(key=lambda pair: pair[1])
    parent = list(range(len(with_angles)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(with_angles) - 1):
        if _circ_delta_deg(with_angles[i][1], with_angles[i + 1][1]) <= threshold_deg:
            _union(i, i + 1)
    if _circ_delta_deg(with_angles[0][1], with_angles[-1][1]) <= threshold_deg:
        _union(0, len(with_angles) - 1)

    buckets: dict[int, list[int]] = {}
    for idx_valid, (original_idx, _) in enumerate(with_angles):
        root = _find(idx_valid)
        buckets.setdefault(root, []).append(original_idx)

    subgroups: list[list[dict]] = []
    subgroup_meta: list[tuple[int, list[dict]]] = []
    for indices in buckets.values():
        ordered_indices = sorted(indices)
        subgroup_meta.append((ordered_indices[0], [group[i] for i in ordered_indices]))

    if not subgroup_meta:
        return [group]

    subgroup_meta.sort(key=lambda pair: pair[0])
    subgroups = [payload for _, payload in subgroup_meta]

    if without_angles:
        fallback_target = max(subgroups, key=len, default=subgroups[0])
        for idx in without_angles:
            fallback_target.append(group[idx])

    if len(subgroups) == 1:
        return [group]
    return subgroups
