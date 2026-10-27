"""ZM-ZEGRID-RESILIENT-RUN-RESUME M4 REWORK-1 — resumable gauge + geometry.

Plan §6.1 / §6.6: the photometric gauge is a FIRST-CLASS resumable artifact
(reference(s), weights, exclusions, coefficients), not a hash to re-derive; the
frozen geometry (canvas/WCS, layout/grid, cell ids, patch geometry, memberships,
halo) must be CONSUMED from the store on restart, never re-planned.

This module serialises both artifacts as strict, versioned, atomic JSON files
inside the run directory:

* ``gauge.json``     — full ``FixedNormalization`` + reference + frame order.
* ``geometry.json``  — canvas, layout dict, per-cell core/patch/membership specs.

No pickle. Every artifact carries a self digest (``artifact_sha256``) AND its
scientific fingerprint, which is re-computed on load and compared with the
frozen run identity. A mismatch is an explicit refusal, never a re-derivation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import run_store as zrs
from .run_resume import ResumeCorruptGlobal, ResumeIncompatible, ResumeStateError

GAUGE_SCHEMA = "zegrid-run-gauge-v1"
GEOMETRY_SCHEMA = "zegrid-run-geometry-v1"
ARTIFACT_VERSION = "1"
GAUGE_FILENAME = "gauge.json"
GEOMETRY_FILENAME = "geometry.json"


# ---------------------------------------------------------------------------
# Deterministic digest helpers (same contract as the run identity builder)
# ---------------------------------------------------------------------------


def _array_digest(value) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(zrs.canonical_json([int(v) for v in arr.shape]))
    h.update(arr.tobytes())
    return h.hexdigest()


def _digest_tree(value):
    """Faithful copy of the identity builder's digest tree.

    In particular a dataclass (e.g. ``FrameExclusion``) is represented ONLY by
    its qualified type name, exactly as the frozen identity does. Duplicating
    that behaviour is required to reproduce the stored gauge fingerprint.
    """
    if isinstance(value, np.ndarray):
        return {
            "__array_sha256__": _array_digest(value),
            "dtype": value.dtype.str,
            "shape": [int(v) for v in value.shape],
        }
    if isinstance(value, Mapping):
        return {
            str(key): _digest_tree(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_digest_tree(item) for item in value]
    if isinstance(value, np.generic):
        return _digest_tree(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return {"__non_finite__": repr(value)}
        return value
    return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}"}


def sha256_json(value) -> str:
    return hashlib.sha256(zrs.canonical_json(_digest_tree(value))).hexdigest()


def _self_digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(zrs.canonical_json(body)).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], run_root: Path) -> None:
    tmp = Path(run_root) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    zrs.atomic_write_bytes(path, zrs.canonical_json(payload), tmp_dir=tmp)


def _read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ResumeStateError(f"{label} is missing at {path}")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except Exception as exc:
        raise ResumeCorruptGlobal(f"{label} at {path} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ResumeStateError(f"{label} at {path} must be an object")
    return value


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------


def gauge_fingerprint(
    *,
    gauge,
    mode: str,
    frame_ids: Sequence[str],
    reference_frame_id: str | None,
) -> str:
    """Reproduce the identity builder's frozen gauge section hash."""
    fields = {
        key: value
        for key, value in vars(gauge).items()
        if key != "reference_index"
    }
    return sha256_json(
        {
            "gauge": _digest_tree(fields),
            "reference_index": int(gauge.reference_index),
            "reference_frame_id": (
                None if reference_frame_id is None else str(reference_frame_id)
            ),
            "frame_ids": [str(v) for v in frame_ids],
            "mode": str(mode),
        }
    )


def write_gauge_artifact(
    run_root: str | os.PathLike[str],
    *,
    run_id: str,
    identity: Mapping[str, Any],
    gauge,
    mode: str,
    frame_ids: Sequence[str],
    reference_frame_id: str | None,
) -> dict:
    """Persist the COMPLETE global gauge before any cell is admitted."""
    root = Path(run_root)
    frame_ids = tuple(str(v) for v in frame_ids)
    if isinstance(gauge.coefficients, np.ndarray):
        coefficients = gauge.coefficients.astype(np.float64, copy=False).tolist()
    else:  # pragma: no cover - defensive
        coefficients = np.asarray(gauge.coefficients, dtype=np.float64).tolist()
    body = {
        "schema": GAUGE_SCHEMA,
        "schema_version": ARTIFACT_VERSION,
        "run_id": str(run_id),
        "identity_sha256": zrs.identity_sha256(identity),
        "mode": str(mode),
        "frame_ids": list(frame_ids),
        "frame_count": int(len(frame_ids)),
        "reference_index": int(gauge.reference_index),
        "reference_frame_id": (
            None if reference_frame_id is None else str(reference_frame_id)
        ),
        "coefficients": coefficients,
        "norm_active": [bool(v) for v in np.asarray(gauge.norm_active)],
        "weights": [float(v) for v in np.asarray(gauge.weights)],
        "weight_active": [bool(v) for v in np.asarray(gauge.weight_active)],
        "exclusions": [
            {
                "index": int(item.index),
                "stage": str(item.stage),
                "reason": str(item.reason),
                "detail": None if item.detail is None else str(item.detail),
            }
            for item in gauge.exclusions
        ],
        "gauge_sha256": gauge_fingerprint(
            gauge=gauge,
            mode=str(mode),
            frame_ids=frame_ids,
            reference_frame_id=reference_frame_id,
        ),
        "note": (
            "M4: complete resumable global gauge. Never pickle; arrays are "
            "inlined as strict JSON numbers. The per-Cell photometric reference "
            "pin is stored WITH each CELL commit (plan §6.1), not re-derived."
        ),
    }
    payload = dict(body)
    payload["artifact_sha256"] = _self_digest(body)
    _atomic_json(root / GAUGE_FILENAME, payload, root)
    return payload


def read_gauge_artifact(
    run_root: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict:
    root = Path(run_root)
    raw = _read_json(root / GAUGE_FILENAME, "gauge artifact")
    artifact_sha = raw.pop("artifact_sha256", None)
    if not artifact_sha or artifact_sha != _self_digest(raw):
        raise ResumeCorruptGlobal(
            "gauge artifact self-digest mismatch; refusing a corrupt gauge "
            "(never a best-effort recovery)"
        )
    if raw.get("schema") != GAUGE_SCHEMA:
        raise ResumeStateError(
            f"gauge artifact schema {raw.get('schema')!r} != {GAUGE_SCHEMA!r}"
        )
    if str(raw.get("schema_version")) != ARTIFACT_VERSION:
        raise ResumeStateError(
            f"gauge artifact version {raw.get('schema_version')!r} is not "
            f"supported ({ARTIFACT_VERSION})"
        )
    if identity is not None:
        if str(raw.get("run_id")) != str(identity["run_id"]):
            raise ResumeCorruptGlobal("gauge artifact belongs to another run_id")
        if str(raw.get("identity_sha256")) != zrs.identity_sha256(identity):
            raise ResumeCorruptGlobal("gauge artifact belongs to another identity")
    return raw


def restore_global_gauge(
    run_root: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any],
) -> tuple[Any, tuple[str, ...], str | None]:
    """Load and VALIDATE the full gauge; returns ``(gauge, frame_ids, ref_id)``."""
    from zemosaic.core.canonical_stacking import FrameExclusion
    from zemosaic.core.canonical_streaming import FixedNormalization

    raw = read_gauge_artifact(run_root, identity=identity)
    frame_ids = tuple(str(v) for v in raw["frame_ids"])
    if int(raw["frame_count"]) != len(frame_ids):
        raise ResumeCorruptGlobal("gauge frame_count disagrees with frame_ids")
    n = len(frame_ids)
    coefficients = np.asarray(raw["coefficients"], dtype=np.float64)
    norm_active = np.asarray(raw["norm_active"], dtype=bool)
    weights = np.asarray(raw["weights"], dtype=np.float64)
    weight_active = np.asarray(raw["weight_active"], dtype=bool)
    if coefficients.shape != (n, 3, 2) and coefficients.ndim != 3:
        raise ResumeCorruptGlobal(
            f"gauge coefficients shape {coefficients.shape} is invalid for {n} frames"
        )
    for arr, expected, label in (
        (norm_active, (n,), "norm_active"),
        (weights, (n,), "weights"),
        (weight_active, (n,), "weight_active"),
    ):
        if arr.shape != expected:
            raise ResumeCorruptGlobal(
                f"gauge {label} shape {arr.shape} != {expected}"
            )
    exclusions = tuple(
        FrameExclusion(
            index=int(item["index"]),
            stage=str(item["stage"]),
            reason=str(item["reason"]),
            detail=None if item.get("detail") is None else str(item["detail"]),
        )
        for item in raw["exclusions"]
    )
    gauge = FixedNormalization(
        reference_index=int(raw["reference_index"]),
        coefficients=np.ascontiguousarray(coefficients),
        norm_active=np.ascontiguousarray(norm_active),
        weights=np.ascontiguousarray(weights),
        weight_active=np.ascontiguousarray(weight_active),
        exclusions=exclusions,
    )
    stored_sha = str(
        identity["science_contract"]["gauge"]["sha256"]
    )
    actual_sha = gauge_fingerprint(
        gauge=gauge,
        mode=str(raw["mode"]),
        frame_ids=frame_ids,
        reference_frame_id=raw.get("reference_frame_id"),
    )
    if actual_sha != stored_sha:
        raise ResumeIncompatible(
            "resume refused: the restored gauge does not match the frozen "
            f"identity gauge hash ({actual_sha} != {stored_sha}); never "
            "re-derive or repair a global gauge",
            reasons=[
                {
                    "section": "gauge",
                    "restored_sha256": actual_sha,
                    "stored_sha256": stored_sha,
                    "detail": "corrupt/mismatched gauge artifact",
                }
            ],
        )
    return gauge, frame_ids, raw.get("reference_frame_id")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _bounds(value: Mapping[str, Any], label: str) -> list[int]:
    out = [int(value[k]) for k in ("x0", "y0", "x1", "y1")]
    if out[2] <= out[0] or out[3] <= out[1]:
        raise ResumeCorruptGlobal(f"{label} is empty: {out}")
    return out


def geometry_fingerprints(
    *,
    canvas,
    layout: Mapping[str, Any],
    cell_ctxs: Sequence[Any],
    halo_px: int,
) -> dict:
    """Reproduce every geometry section of the frozen identity."""
    bounds = [
        [
            str(cell.cell_id),
            int(cell.core.x0),
            int(cell.core.y0),
            int(cell.core.x1),
            int(cell.core.y1),
        ]
        for (_row, _col, cell, _patch, _mem) in cell_ctxs
    ]
    memberships = [
        [
            str(cell.cell_id),
            [str(v) for v in mem.core_ids],
            [str(v) for v in mem.patch_ids],
        ]
        for (_row, _col, cell, _patch, mem) in cell_ctxs
    ]
    canvas_shape = [int(canvas.height), int(canvas.width)]
    return {
        "grid_geometry_sha256": sha256_json(bounds),
        "grid_cell_ids_sha256": sha256_json(
            sorted(str(cell.cell_id) for (_r, _c, cell, _p, _m) in cell_ctxs)
        ),
        "memberships_sha256": sha256_json(memberships),
        "memberships_n_cells": int(len(memberships)),
        "canvas_wcs_sha256": hashlib.sha256(
            str(canvas.wcs_header).encode("utf-8")
        ).hexdigest(),
        "canvas_geometry_sha256": sha256_json(
            {
                "canvas_id": str(getattr(canvas, "canvas_id", "")),
                "shape": canvas_shape,
                "resolution_deg": float(getattr(canvas, "resolution_deg", 0.0)),
            }
        ),
        "nx": int(layout["nx"]),
        "ny": int(layout["ny"]),
        "cell_ids": [
            str(cell.cell_id) for (_r, _c, cell, _p, _m) in cell_ctxs
        ],
        "halo_px": int(halo_px),
    }


def write_geometry_artifact(
    run_root: str | os.PathLike[str],
    *,
    run_id: str,
    identity: Mapping[str, Any],
    canvas,
    layout: Mapping[str, Any],
    cell_ctxs: Sequence[Any],
    halo_px: int,
) -> dict:
    root = Path(run_root)
    fingerprints = geometry_fingerprints(
        canvas=canvas, layout=layout, cell_ctxs=cell_ctxs, halo_px=halo_px
    )
    cells = []
    for row, col, cell, patch, mem in cell_ctxs:
        cells.append(
            {
                "row": int(row),
                "col": int(col),
                "cell_id": str(cell.cell_id),
                "core": [
                    int(cell.core.x0), int(cell.core.y0),
                    int(cell.core.x1), int(cell.core.y1),
                ],
                "layout_id": str(getattr(cell, "layout_id", "")),
                "canvas_id": str(getattr(cell, "canvas_id", "")),
                "patch": [
                    int(patch.patch.x0), int(patch.patch.y0),
                    int(patch.patch.x1), int(patch.patch.y1),
                ],
                "core_slice": [
                    int(patch.core_slice.x0), int(patch.core_slice.y0),
                    int(patch.core_slice.x1), int(patch.core_slice.y1),
                ],
                "halo_px": int(patch.halo_px),
                "patch_wcs_header": str(patch.patch_wcs_header),
                "core_ids": [str(v) for v in mem.core_ids],
                "patch_ids": [str(v) for v in mem.patch_ids],
            }
        )
    body = {
        "schema": GEOMETRY_SCHEMA,
        "schema_version": ARTIFACT_VERSION,
        "run_id": str(run_id),
        "identity_sha256": zrs.identity_sha256(identity),
        "canvas": {
            "canvas_id": str(canvas.canvas_id),
            "wcs_header": str(canvas.wcs_header),
            "width": int(canvas.width),
            "height": int(canvas.height),
            "resolution_deg": float(canvas.resolution_deg),
        },
        "layout": zrs.json_safe(dict(layout), "restored layout"),
        "tile_px": int(identity["science_contract"]["grid"]["tile_px"]),
        "halo_px": int(halo_px),
        "cell_count": int(len(cells)),
        "cells": cells,
        "geometry_sha256": sha256_json(
            {
                "canvas": {
                    "canvas_id": str(canvas.canvas_id),
                    "shape": [int(canvas.height), int(canvas.width)],
                    "resolution_deg": float(canvas.resolution_deg),
                },
                "layout": zrs.json_safe(dict(layout), "layout fingerprint"),
                "cells": cells,
                "halo_px": int(halo_px),
            }
        ),
        "fingerprints": fingerprints,
        "note": (
            "M4: frozen execution geometry. A restart consumes these exact "
            "cells/memberships and never runs the adaptive layout search."
        ),
    }
    payload = dict(body)
    payload["artifact_sha256"] = _self_digest(body)
    _atomic_json(root / GEOMETRY_FILENAME, payload, root)
    return payload


def read_geometry_artifact(
    run_root: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any] | None = None,
) -> dict:
    root = Path(run_root)
    raw = _read_json(root / GEOMETRY_FILENAME, "geometry artifact")
    artifact_sha = raw.pop("artifact_sha256", None)
    if not artifact_sha or artifact_sha != _self_digest(raw):
        raise ResumeCorruptGlobal(
            "geometry artifact self-digest mismatch; refusing a corrupt "
            "geometry (never a re-derivation)"
        )
    if raw.get("schema") != GEOMETRY_SCHEMA:
        raise ResumeStateError(
            f"geometry artifact schema {raw.get('schema')!r} != {GEOMETRY_SCHEMA!r}"
        )
    if str(raw.get("schema_version")) != ARTIFACT_VERSION:
        raise ResumeStateError(
            f"geometry artifact version {raw.get('schema_version')!r} is not "
            f"supported ({ARTIFACT_VERSION})"
        )
    if identity is not None:
        if str(raw.get("run_id")) != str(identity["run_id"]):
            raise ResumeCorruptGlobal("geometry artifact belongs to another run_id")
        if str(raw.get("identity_sha256")) != zrs.identity_sha256(identity):
            raise ResumeCorruptGlobal("geometry artifact belongs to another identity")
    return raw


def restore_geometry(
    run_root: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any],
) -> tuple[Any, dict, list]:
    """Rebuild canvas/layout/cell contexts from the STORED geometry.

    Pure object reconstruction from frozen integers/strings: no WCS projection,
    no layout candidate search, no membership recomputation.
    """
    from . import geometry as zg

    raw = read_geometry_artifact(run_root, identity=identity)
    canvas_data = raw["canvas"]
    canvas = zg.GlobalCanvas(
        canvas_id=str(canvas_data["canvas_id"]),
        wcs_header=str(canvas_data["wcs_header"]),
        width=int(canvas_data["width"]),
        height=int(canvas_data["height"]),
        resolution_deg=float(canvas_data["resolution_deg"]),
    )
    layout = dict(raw["layout"])
    halo_px = int(raw["halo_px"])
    cell_ctxs = []
    for entry in raw["cells"]:
        cid = str(entry["cell_id"])
        core = _bounds(
            {k: entry["core"][i] for i, k in enumerate(("x0", "y0", "x1", "y1"))},
            f"cell {cid} core",
        )
        patch_raw = entry["patch"]
        core_slice_raw = entry["core_slice"]
        cell = zg.ZeGridCell(
            cell_id=cid,
            canvas_id=str(entry["canvas_id"]),
            layout_id=str(entry["layout_id"]),
            row=int(entry["row"]),
            col=int(entry["col"]),
            core=zg.GlobalBounds(*core),
        )
        patch = zg.ProcessingPatch(
            cell_id=cid,
            patch=zg.GlobalBounds(
                int(patch_raw[0]), int(patch_raw[1]),
                int(patch_raw[2]), int(patch_raw[3]),
            ),
            core_slice=zg.PatchBounds(
                int(core_slice_raw[0]), int(core_slice_raw[1]),
                int(core_slice_raw[2]), int(core_slice_raw[3]),
            ),
            halo_px=int(entry.get("halo_px", halo_px)),
            patch_wcs_header=str(entry["patch_wcs_header"]),
        )
        mem = zg.CellMembership(
            cell_id=cid,
            core_ids=tuple(str(v) for v in entry["core_ids"]),
            patch_ids=tuple(str(v) for v in entry["patch_ids"]),
        )
        cell_ctxs.append((int(entry["row"]), int(entry["col"]), cell, patch, mem))

    # Validate every stored geometry value against the frozen identity BEFORE
    # returning anything executable. A mismatch is a refusal, never a repair.
    actual = geometry_fingerprints(
        canvas=canvas, layout=layout, cell_ctxs=cell_ctxs, halo_px=halo_px
    )
    contract = identity["science_contract"]
    checks = (
        ("grid.geometry_sha256", contract["grid"]["geometry_sha256"],
         actual["grid_geometry_sha256"]),
        ("grid.cell_ids_sha256", contract["grid"]["cell_ids_sha256"],
         actual["grid_cell_ids_sha256"]),
        ("memberships.sha256", contract["memberships"]["sha256"],
         actual["memberships_sha256"]),
        ("memberships.n_cells", contract["memberships"]["n_cells"],
         actual["memberships_n_cells"]),
        ("canvas.wcs_sha256", contract["canvas"]["wcs_sha256"],
         actual["canvas_wcs_sha256"]),
        ("canvas.geometry_sha256", contract["canvas"]["geometry_sha256"],
         actual["canvas_geometry_sha256"]),
        ("grid.nx", contract["grid"]["nx"], actual["nx"]),
        ("grid.ny", contract["grid"]["ny"], actual["ny"]),
        ("halo.halo_px", contract["halo"]["halo_px"], actual["halo_px"]),
    )
    reasons = []
    for label, stored, loaded in checks:
        if stored != loaded:
            reasons.append(
                {"section": label, "stored": stored, "restored": loaded}
            )
    if reasons:
        raise ResumeIncompatible(
            "resume refused: the stored geometry does not match the frozen "
            f"identity ({len(reasons)} mismatch(es))",
            reasons=reasons,
        )
    return canvas, layout, cell_ctxs
