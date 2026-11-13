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
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import run_store as zrs
from .run_resume import ResumeCorruptGlobal, ResumeIncompatible, ResumeStateError

LEGACY_GAUGE_SCHEMA = "zegrid-run-gauge-v1"
GAUGE_SCHEMA = "zegrid-run-gauge-v2"
GEOMETRY_SCHEMA = "zegrid-run-geometry-v1"
LEGACY_GAUGE_ARTIFACT_VERSION = "1"
GAUGE_ARTIFACT_VERSION = "2"
GEOMETRY_ARTIFACT_VERSION = "1"
# Backwards-compatible name for the geometry version (the only artifact that
# stayed at v1). Kept so external importers do not silently compare a bumped
# gauge version against geometry.
ARTIFACT_VERSION = GEOMETRY_ARTIFACT_VERSION
GAUGE_FILENAME = "gauge.json"
GEOMETRY_FILENAME = "geometry.json"

GAUGE_COEFFICIENTS_ENCODING = "explicit-nonfinite-v1"
_GAUGE_COEFFICIENT_DTYPES = frozenset({"<f8", ">f8", "=f8"})
_GAUGE_EXCLUSION_STAGES = frozenset({"normalization", "weighting"})


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


def _reject_json_constant(token: str):
    raise ValueError(f"non-finite JSON number token {token!r} is forbidden")


def _read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise ResumeStateError(f"{label} is missing at {path}")
    try:
        value = json.loads(
            path.read_bytes().decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
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


def _require_json_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResumeCorruptGlobal(f"{label} must be an integer, got {value!r}")
    return int(value)


def _check_coefficient_shape(arr: np.ndarray, frame_count: int, *, error) -> None:
    if (
        arr.ndim != 3
        or arr.shape[0] != frame_count
        or arr.shape[1] < 1
        or arr.shape[2] != 2
    ):
        raise error(
            f"gauge coefficients shape {arr.shape} is invalid for "
            f"{frame_count} frames; expected (N, C, 2)"
        )


def _check_gauge_vectors(
    norm_active: np.ndarray,
    weights: np.ndarray,
    weight_active: np.ndarray,
    *,
    frame_count: int,
    error,
) -> None:
    """Shared strict validation of the gauge's active flags and weights."""
    for arr, label in (
        (norm_active, "norm_active"),
        (weight_active, "weight_active"),
        (weights, "weights"),
    ):
        if arr.shape != (frame_count,):
            raise error(f"gauge {label} shape {arr.shape} != ({frame_count},)")
    if not np.all(np.isfinite(weights)):
        raise error("gauge weights must be strictly finite")
    if np.any(weights < 0.0):
        raise error("gauge weights must be non-negative")
    if np.any(weight_active & ~norm_active):
        raise error("gauge weight_active must be a subset of norm_active")
    if np.any(weights[weight_active] <= 0.0):
        raise error("gauge weights of weight-active frames must be positive")
    if np.any(weights[~weight_active] != 0.0):
        raise error("gauge weights of weight-inactive frames must be exactly 0")


def _check_coefficients_match_activity(
    coefficients: np.ndarray, norm_active: np.ndarray, *, frame_count: int, error
) -> None:
    """NaN is legal ONLY on frames excluded from normalization."""
    if np.any(np.isinf(coefficients)):
        index = tuple(int(v) for v in np.argwhere(np.isinf(coefficients))[0])
        raise error(
            f"gauge coefficients contain an infinite value at index {index}; "
            "infinity is never a valid photometric coefficient"
        )
    nan_mask = np.isnan(coefficients)
    bad = nan_mask & np.asarray(norm_active, dtype=bool).reshape((frame_count, 1, 1))
    if bool(bad.any()):
        index = tuple(int(v) for v in np.argwhere(bad)[0])
        raise error(
            "gauge coefficients have a non-finite value on a norm-active frame "
            f"at index {index}"
        )


def _validate_exclusions(
    items,
    *,
    frame_count: int,
    norm_active: np.ndarray,
    weight_active: np.ndarray,
    error,
) -> list[dict]:
    if not isinstance(items, (list, tuple)):
        raise error("gauge exclusions must be a list")
    out: list[dict] = []
    for position, item in enumerate(items):
        if isinstance(item, Mapping):
            if set(item) != {"index", "stage", "reason", "detail"}:
                raise error(
                    f"gauge exclusion[{position}] must have exactly "
                    "index/stage/reason/detail"
                )
            raw_index = item.get("index")
            stage = item.get("stage")
            reason = item.get("reason")
            detail = item.get("detail")
        else:
            raw_index = getattr(item, "index", None)
            stage = getattr(item, "stage", None)
            reason = getattr(item, "reason", None)
            detail = getattr(item, "detail", None)
        if isinstance(raw_index, bool) or not isinstance(
            raw_index, (int, np.integer)
        ):
            raise error(f"gauge exclusion[{position}].index must be an integer")
        index = int(raw_index)
        if not (0 <= index < frame_count):
            raise error(
                f"gauge exclusion[{position}].index {index} is out of range "
                f"[0, {frame_count})"
            )
        if not isinstance(stage, str) or stage not in _GAUGE_EXCLUSION_STAGES:
            raise error(
                f"gauge exclusion[{position}].stage {stage!r} is not one of "
                f"{sorted(_GAUGE_EXCLUSION_STAGES)}"
            )
        if not isinstance(reason, str) or not reason:
            raise error(f"gauge exclusion[{position}].reason must be a non-empty string")
        if detail is not None and not isinstance(detail, str):
            raise error(f"gauge exclusion[{position}].detail must be null or a string")
        if stage == "normalization" and bool(norm_active[index]):
            raise error(
                f"gauge normalization exclusion at frame {index} contradicts "
                "norm_active=True"
            )
        if stage == "weighting" and bool(weight_active[index]):
            raise error(
                f"gauge weighting exclusion at frame {index} contradicts "
                "weight_active=True"
            )
        out.append(
            {"index": index, "stage": stage, "reason": reason, "detail": detail}
        )
    return out


def _encode_gauge_coefficients(
    coefficients: np.ndarray, norm_active: np.ndarray, *, frame_count: int
) -> dict:
    """Encode coefficients as strict JSON, preserving NaN positions and bits."""
    arr = np.array(coefficients, dtype="<f8", copy=True)
    _check_coefficient_shape(arr, frame_count, error=ResumeStateError)
    active = np.asarray(norm_active, dtype=bool)
    if active.shape != (frame_count,):
        raise ResumeStateError(
            f"gauge norm_active shape {active.shape} != ({frame_count},)"
        )
    _check_coefficients_match_activity(
        arr, active, frame_count=frame_count, error=ResumeStateError
    )
    nan_mask = np.isnan(arr)
    bits = arr.view(np.uint64)
    non_finite = [
        {
            "index": [int(i), int(c), int(k)],
            "kind": "nan",
            "float64_bits": f"0x{int(bits[i, c, k]):016x}",
        }
        for i, c, k in np.argwhere(nan_mask)
    ]
    return {
        "encoding": GAUGE_COEFFICIENTS_ENCODING,
        "dtype": "<f8",
        "shape": [int(v) for v in arr.shape],
        "values": np.where(nan_mask, None, arr).tolist(),
        "non_finite": non_finite,
    }


def _flatten_coefficient_values(
    node: Any, shape: tuple[int, ...], expected_total: int
) -> tuple[list[float], set[tuple]]:
    """Validate the nested JSON ``values`` against ``shape`` and flatten them.

    The walk is proportional to the ALREADY-MATERIALISED JSON content, never to
    the declared shape: a compact document claiming a huge dimension fails on
    the child list length before any array proportional to that dimension is
    allocated. ``expected_total`` also bounds the flat list for a small shape
    fed with oversized content.
    """
    flat: list[float] = []
    nulls: set[tuple] = set()

    def walk(value: Any, position: tuple) -> None:
        depth = len(position)
        if depth == len(shape):
            if len(flat) >= expected_total:
                raise ResumeCorruptGlobal(
                    "gauge coefficients contain more values than the declared shape"
                )
            if value is None:
                nulls.add(position)
                flat.append(np.nan)
                return
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ResumeCorruptGlobal(
                    f"gauge coefficient at {position} must be a number or null, "
                    f"got {type(value).__name__}"
                )
            try:
                number = float(value)
            except (OverflowError, ValueError) as exc:
                raise ResumeCorruptGlobal(
                    f"gauge coefficient at {position} is not representable: {exc}"
                ) from exc
            if not math.isfinite(number):
                raise ResumeCorruptGlobal(
                    f"gauge coefficient at {position} is a non-finite JSON number"
                )
            flat.append(number)
            return
        expected = shape[depth]
        if not isinstance(value, list) or len(value) != expected:
            raise ResumeCorruptGlobal(
                f"gauge coefficients at {position} must be a list of length {expected}"
            )
        for index, child in enumerate(value):
            walk(child, position + (index,))

    walk(node, ())
    return flat, nulls


def _decode_gauge_coefficients(
    block: Any, *, frame_count: int, schema_version: int
) -> np.ndarray:
    if schema_version == 1:
        if not isinstance(block, list):
            raise ResumeCorruptGlobal("legacy gauge v1 coefficients must be a list")
        try:
            arr = np.asarray(block, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ResumeCorruptGlobal(
                f"legacy gauge v1 coefficients are not numeric: {exc}"
            ) from exc
        _check_coefficient_shape(arr, frame_count, error=ResumeCorruptGlobal)
        if not np.all(np.isfinite(arr)):
            raise ResumeCorruptGlobal(
                "legacy gauge v1 carries non-finite coefficients; gauge v1 only "
                "supports strictly finite coefficients (refusing, never repaired)"
            )
        return np.ascontiguousarray(arr)

    if not isinstance(block, Mapping):
        raise ResumeCorruptGlobal("gauge coefficients must be an object")
    if set(block) != {"encoding", "dtype", "shape", "values", "non_finite"}:
        raise ResumeCorruptGlobal(
            "gauge coefficients must have exactly "
            "encoding/dtype/shape/values/non_finite"
        )
    if block.get("encoding") != GAUGE_COEFFICIENTS_ENCODING:
        raise ResumeCorruptGlobal(
            f"gauge coefficients encoding {block.get('encoding')!r} is not "
            f"supported ({GAUGE_COEFFICIENTS_ENCODING})"
        )
    if block.get("dtype") not in _GAUGE_COEFFICIENT_DTYPES:
        raise ResumeCorruptGlobal(
            f"gauge coefficients dtype {block.get('dtype')!r} is not float64"
        )
    shape_raw = block.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 3:
        raise ResumeCorruptGlobal("gauge coefficients shape must be a list of 3 integers")
    shape = tuple(
        _require_json_int(value, "gauge coefficients shape") for value in shape_raw
    )
    if shape[0] != frame_count or shape[1] < 1 or shape[2] != 2:
        raise ResumeCorruptGlobal(
            f"gauge coefficients shape {shape} is invalid for {frame_count} frames; "
            "expected (N, C, 2)"
        )

    expected_total = math.prod(shape)
    flat, nulls = _flatten_coefficient_values(
        block.get("values"), shape, expected_total
    )
    if len(flat) != expected_total:
        raise ResumeCorruptGlobal(
            f"gauge coefficients have {len(flat)} values but the shape {shape} "
            f"declares {expected_total}"
        )

    entries = block.get("non_finite")
    if not isinstance(entries, list):
        raise ResumeCorruptGlobal("gauge coefficients non_finite must be a list")
    if len(entries) > len(flat):
        raise ResumeCorruptGlobal(
            "gauge coefficients non_finite has more entries than the value grid"
        )
    declared: dict[tuple, int] = {}
    for position, item in enumerate(entries):
        if not isinstance(item, Mapping) or set(item) != {
            "index", "kind", "float64_bits"
        }:
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}] must have exactly "
                "index/kind/float64_bits"
            )
        index_raw = item.get("index")
        if not isinstance(index_raw, list) or len(index_raw) != 3:
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}].index must be a list of 3 integers"
            )
        index = tuple(
            _require_json_int(value, f"gauge non_finite[{position}].index")
            for value in index_raw
        )
        if any(not (0 <= index[axis] < shape[axis]) for axis in range(3)):
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}].index {index} is out of range {shape}"
            )
        if item.get("kind") != "nan":
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}].kind must be 'nan'"
            )
        bits_text = item.get("float64_bits")
        if (
            not isinstance(bits_text, str)
            or not bits_text.lower().startswith("0x")
            or len(bits_text) != 18
        ):
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}].float64_bits must be a 0x-prefixed "
                "64-bit hex string"
            )
        try:
            bits = int(bits_text, 16)
        except ValueError as exc:
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}].float64_bits is not hexadecimal"
            ) from exc
        value = np.array([bits], dtype=np.uint64).view(np.float64)[0]
        if not np.isnan(value):
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}] declares a bit pattern that is "
                "not NaN"
            )
        if index in declared:
            raise ResumeCorruptGlobal(
                f"gauge non_finite[{position}] duplicates index {index}"
            )
        declared[index] = bits
    if set(declared) != nulls:
        raise ResumeCorruptGlobal(
            "gauge coefficients non_finite table does not exactly match the "
            "null sentinels"
        )
    # Allocate ONLY after the content proved the declared shape and the
    # non-finite table was validated. The size here is the real JSON content,
    # so a compact corrupt document never triggers a shape-sized allocation.
    arr = np.asarray(flat, dtype=np.float64).reshape(shape)
    bits_view = arr.view(np.uint64)
    for index, bits in declared.items():
        bits_view[index] = np.uint64(bits)
    return np.ascontiguousarray(arr)


def _read_bool_vector(raw: Mapping[str, Any], label: str, frame_count: int) -> np.ndarray:
    value = raw.get(label)
    if not isinstance(value, list) or len(value) != frame_count:
        raise ResumeCorruptGlobal(
            f"gauge {label} must be a list of {frame_count} booleans"
        )
    for index, item in enumerate(value):
        if not isinstance(item, bool):
            raise ResumeCorruptGlobal(f"gauge {label}[{index}] must be a boolean")
    return np.asarray(value, dtype=bool)


def _read_weight_vector(raw: Mapping[str, Any], label: str, frame_count: int) -> np.ndarray:
    value = raw.get(label)
    if not isinstance(value, list) or len(value) != frame_count:
        raise ResumeCorruptGlobal(
            f"gauge {label} must be a list of {frame_count} numbers"
        )
    out = np.empty((frame_count,), dtype=np.float64)
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ResumeCorruptGlobal(f"gauge {label}[{index}] must be a number")
        try:
            number = float(item)
        except (OverflowError, ValueError) as exc:
            raise ResumeCorruptGlobal(
                f"gauge {label}[{index}] is not representable: {exc}"
            ) from exc
        if not math.isfinite(number):
            raise ResumeCorruptGlobal(
                f"gauge {label}[{index}] must be strictly finite"
            )
        out[index] = number
    return out


def _validate_gauge_document(raw: Mapping[str, Any]) -> dict:
    """Validate and decode a gauge document; never repairs it."""
    schema = raw.get("schema")
    version_raw = raw.get("schema_version")
    if schema == LEGACY_GAUGE_SCHEMA:
        if str(version_raw) != LEGACY_GAUGE_ARTIFACT_VERSION:
            raise ResumeStateError(
                f"legacy gauge artifact version {version_raw!r} is not supported "
                f"({LEGACY_GAUGE_ARTIFACT_VERSION})"
            )
        schema_version = 1
    elif schema == GAUGE_SCHEMA:
        if str(version_raw) != GAUGE_ARTIFACT_VERSION:
            raise ResumeStateError(
                f"gauge artifact version {version_raw!r} is not supported "
                f"({GAUGE_ARTIFACT_VERSION})"
            )
        schema_version = 2
    else:
        raise ResumeStateError(
            f"gauge artifact schema {schema!r} is not supported"
        )

    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ResumeCorruptGlobal("gauge run_id must be a non-empty string")
    identity_sha = raw.get("identity_sha256")
    if not isinstance(identity_sha, str) or not identity_sha:
        raise ResumeCorruptGlobal("gauge identity_sha256 must be a non-empty string")
    gauge_sha = raw.get("gauge_sha256")
    if (
        not isinstance(gauge_sha, str)
        or len(gauge_sha) != 64
        or any(char not in "0123456789abcdef" for char in gauge_sha)
    ):
        raise ResumeCorruptGlobal(
            "gauge_sha256 must be a lowercase 64-character hex digest"
        )
    mode = raw.get("mode")
    if not isinstance(mode, str) or not mode:
        raise ResumeCorruptGlobal("gauge mode must be a non-empty string")

    frame_ids_raw = raw.get("frame_ids")
    if not isinstance(frame_ids_raw, list) or not all(
        isinstance(value, str) and value for value in frame_ids_raw
    ):
        raise ResumeCorruptGlobal(
            "gauge frame_ids must be a list of non-empty strings"
        )
    frame_ids = tuple(frame_ids_raw)
    frame_count = len(frame_ids)
    if frame_count < 1:
        raise ResumeCorruptGlobal("gauge artifact must cover at least one frame")
    stored_count = _require_json_int(raw.get("frame_count"), "gauge frame_count")
    if stored_count != frame_count:
        raise ResumeCorruptGlobal("gauge frame_count disagrees with frame_ids")

    reference_index = _require_json_int(
        raw.get("reference_index"), "gauge reference_index"
    )
    if not (0 <= reference_index < frame_count):
        raise ResumeCorruptGlobal(
            f"gauge reference_index {reference_index} is out of range "
            f"[0, {frame_count})"
        )
    reference_frame_id = raw.get("reference_frame_id")
    if reference_frame_id is not None:
        if not isinstance(reference_frame_id, str):
            raise ResumeCorruptGlobal("gauge reference_frame_id must be null or a string")
        if reference_frame_id != frame_ids[reference_index]:
            raise ResumeCorruptGlobal(
                "gauge reference_frame_id does not match "
                f"frame_ids[{reference_index}]"
            )

    norm_active = _read_bool_vector(raw, "norm_active", frame_count)
    weight_active = _read_bool_vector(raw, "weight_active", frame_count)
    weights = _read_weight_vector(raw, "weights", frame_count)
    _check_gauge_vectors(
        norm_active,
        weights,
        weight_active,
        frame_count=frame_count,
        error=ResumeCorruptGlobal,
    )
    coefficients = _decode_gauge_coefficients(
        raw.get("coefficients"),
        frame_count=frame_count,
        schema_version=schema_version,
    )
    _check_coefficients_match_activity(
        coefficients,
        norm_active,
        frame_count=frame_count,
        error=ResumeCorruptGlobal,
    )
    exclusions = _validate_exclusions(
        raw.get("exclusions"),
        frame_count=frame_count,
        norm_active=norm_active,
        weight_active=weight_active,
        error=ResumeCorruptGlobal,
    )
    from zemosaic.core.canonical_stacking import FrameExclusion

    return {
        "schema_version": schema_version,
        "frame_ids": frame_ids,
        "coefficients": coefficients,
        "norm_active": np.ascontiguousarray(norm_active),
        "weights": np.ascontiguousarray(weights),
        "weight_active": np.ascontiguousarray(weight_active),
        "exclusions": tuple(
            FrameExclusion(
                index=item["index"],
                stage=item["stage"],
                reason=item["reason"],
                detail=item["detail"],
            )
            for item in exclusions
        ),
        "exclusions_json": exclusions,
        "reference_index": reference_index,
        "reference_frame_id": reference_frame_id,
        "mode": mode,
    }


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
    """Persist the COMPLETE global gauge before any cell is admitted.

    Fail-closed: all validation runs BEFORE the atomic rename, so an invalid
    in-memory gauge can never replace a previous valid ``gauge.json``.
    """
    root = Path(run_root)
    frame_ids = tuple(str(v) for v in frame_ids)
    frame_count = len(frame_ids)
    if frame_count < 1:
        raise ResumeStateError("gauge artifact requires at least one frame id")
    if any(not frame_id for frame_id in frame_ids):
        raise ResumeStateError("gauge frame_ids must be non-empty strings")

    raw_reference_index = gauge.reference_index
    if isinstance(raw_reference_index, bool) or not isinstance(
        raw_reference_index, (int, np.integer)
    ):
        raise ResumeStateError("gauge reference_index must be an integer")
    reference_index = int(raw_reference_index)
    if not (0 <= reference_index < frame_count):
        raise ResumeStateError(
            f"gauge reference_index {reference_index} is out of range "
            f"[0, {frame_count})"
        )
    reference_frame_id_value = (
        None if reference_frame_id is None else str(reference_frame_id)
    )
    if (
        reference_frame_id_value is not None
        and reference_frame_id_value != frame_ids[reference_index]
    ):
        raise ResumeStateError(
            f"gauge reference_frame_id {reference_frame_id_value!r} does not "
            f"match frame_ids[{reference_index}]={frame_ids[reference_index]!r}"
        )

    norm_active = np.asarray(gauge.norm_active)
    weights = np.asarray(gauge.weights)
    weight_active = np.asarray(gauge.weight_active)
    coefficients = np.asarray(gauge.coefficients)
    if norm_active.dtype != np.dtype(bool):
        raise ResumeStateError(
            f"gauge norm_active must be bool (got {norm_active.dtype.name})"
        )
    if weight_active.dtype != np.dtype(bool):
        raise ResumeStateError(
            f"gauge weight_active must be bool (got {weight_active.dtype.name})"
        )
    if weights.dtype != np.dtype(np.float64):
        raise ResumeStateError(
            f"gauge weights must be float64 (got {weights.dtype.name})"
        )
    if coefficients.dtype != np.dtype(np.float64):
        raise ResumeStateError(
            f"gauge coefficients must be float64 (got {coefficients.dtype.name})"
        )
    _check_gauge_vectors(
        norm_active,
        weights,
        weight_active,
        frame_count=frame_count,
        error=ResumeStateError,
    )
    encoded_coefficients = _encode_gauge_coefficients(
        coefficients, norm_active, frame_count=frame_count
    )
    exclusions = _validate_exclusions(
        gauge.exclusions,
        frame_count=frame_count,
        norm_active=norm_active,
        weight_active=weight_active,
        error=ResumeStateError,
    )

    body = {
        "schema": GAUGE_SCHEMA,
        "schema_version": GAUGE_ARTIFACT_VERSION,
        "run_id": str(run_id),
        "identity_sha256": zrs.identity_sha256(identity),
        "mode": str(mode),
        "frame_ids": list(frame_ids),
        "frame_count": int(frame_count),
        "reference_index": int(reference_index),
        "reference_frame_id": reference_frame_id_value,
        "coefficients": encoded_coefficients,
        "norm_active": [bool(v) for v in norm_active],
        "weights": [float(v) for v in weights],
        "weight_active": [bool(v) for v in weight_active],
        "exclusions": exclusions,
        "gauge_sha256": gauge_fingerprint(
            gauge=gauge,
            mode=str(mode),
            frame_ids=frame_ids,
            reference_frame_id=reference_frame_id_value,
        ),
        "note": (
            "M4: complete resumable global gauge. Never pickle. Coefficients use "
            f"{GAUGE_COEFFICIENTS_ENCODING}: finite values inline, NaN positions "
            "as null plus an exact float64-bit table; NaN is legal only on "
            "norm_active=False frames. The per-Cell photometric reference pin is "
            "stored WITH each CELL commit (plan §6.1), not re-derived."
        ),
    }
    # Validate the exact document that will be written (strict JSON, sentinels,
    # decoded semantics) before touching the filesystem.
    _validate_gauge_document(body)
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
    _validate_gauge_document(raw)
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
    from zemosaic.core.canonical_streaming import FixedNormalization

    raw = read_gauge_artifact(run_root, identity=identity)
    parsed = _validate_gauge_document(raw)
    gauge = FixedNormalization(
        reference_index=parsed["reference_index"],
        coefficients=parsed["coefficients"],
        norm_active=parsed["norm_active"],
        weights=parsed["weights"],
        weight_active=parsed["weight_active"],
        exclusions=parsed["exclusions"],
    )
    stored_sha = str(identity["science_contract"]["gauge"]["sha256"])
    actual_sha = gauge_fingerprint(
        gauge=gauge,
        mode=parsed["mode"],
        frame_ids=parsed["frame_ids"],
        reference_frame_id=parsed["reference_frame_id"],
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
    return gauge, parsed["frame_ids"], parsed["reference_frame_id"]


# ---------------------------------------------------------------------------
# Gauge IMPORT (ZM-ZEGRID-GAUGE-IMPORT)
# ---------------------------------------------------------------------------


def _gauge_import_refusal(section: str, detail: str, **facts):
    """One structured refusal for a gauge import, naming the failing section."""
    from .run_resume import ResumeIncompatible

    reason = {"section": str(section), "detail": str(detail)}
    reason.update(facts)
    return ResumeIncompatible(
        f"gauge import refused ({section}): {detail}",
        reasons=[reason],
    )


def _verify_gauge_import(
    gauge_path: str | os.PathLike[str],
    *,
    origin_identity: Mapping[str, Any],
    origin_run_json_path: str | None,
    expected_frame_rows: Sequence[Mapping[str, Any]] | None,
    expected_frame_ids: Sequence[str],
    expected_canvas: Mapping[str, Any],
    expected_science_params: Mapping[str, Any],
    expected_mode: str,
) -> tuple[Any, tuple[str, ...], str | None, dict]:
    """ONE verification body shared by the pre-check and the full import.

    ``expected_frame_rows=None`` means the caller cannot hash the corpus yet:
    the frame COUNT, every ``frame_id``/``order``/``role`` and the id order are
    still verified, but ``content_sha256``/``nbytes`` are NOT (unavailable).
    :func:`import_gauge_artifact` ALWAYS passes the rows, so the content digests
    are always verified before any gauge is imported — the pre-check is never
    the only gate.

    Opt-in only: the caller passes an explicit artifact path and the origin
    ``run.json`` identity. Every check below is the SAME validator the resume
    path already trusts:

    * strict JSON + ``artifact_sha256`` self-digest -> :func:`_read_json`,
      :func:`_self_digest` (identical to :func:`read_gauge_artifact`);
    * schema/version/coefficient encoding/vectors/exclusions ->
      :func:`_validate_gauge_document` (identical to the restore path);
    * origin identity shape/schema -> :func:`run_store.validate_run_identity`;
    * artifact binding (``run_id`` + ``identity_sha256``) -> the same two checks
      as :func:`read_gauge_artifact`;
    * frame set: same ids, order, content digests and sizes -> the origin
      identity's ``science_contract.inputs`` section;
    * canvas -> the origin identity's ``science_contract.canvas`` section;
    * photometric parameters (normalization/weighting/gauge mode/reference) ->
      the origin identity's ``science_contract.science.params`` (the execution
      ``backend`` field is EXCLUDED: it is not a photometric parameter and is
      normalised to the run's effective backend, which a fresh import cannot
      know here);
    * gauge coherence: reference frame id/index, mode and the EXACT
      :func:`gauge_fingerprint` of the decoded gauge must equal BOTH the origin
      identity's ``gauge.sha256`` and the artifact's own ``gauge_sha256``
      (identical to :func:`restore_global_gauge`).

    The artifact is only READ (no write, no repair, no normalisation). Returns
    ``(gauge, frame_ids, reference_frame_id, provenance)``.
    """
    from .run_store import StoreFormatError, validate_run_identity
    from .run_store import identity_sha256 as _identity_sha256

    path = Path(gauge_path)
    try:
        raw = _read_json(path, "imported gauge artifact")
    except Exception as exc:  # unreadable / non-strict JSON / duplicate keys
        raise _gauge_import_refusal(
            "artifact",
            f"the artifact at {str(path)!r} is not a readable strict-JSON "
            f"gauge document ({type(exc).__name__}: {exc})",
            gauge_path=str(path),
        ) from exc
    artifact_sha = raw.pop("artifact_sha256", None)
    if not artifact_sha or artifact_sha != _self_digest(raw):
        raise _gauge_import_refusal(
            "artifact",
            "the artifact self-digest does not verify; refusing a corrupt "
            "gauge (never a best-effort import)",
            gauge_path=str(path),
        )
    try:
        parsed = _validate_gauge_document(raw)
    except Exception as exc:
        raise _gauge_import_refusal(
            "artifact",
            f"the artifact does not satisfy {GAUGE_SCHEMA}: "
            f"{type(exc).__name__}: {exc}",
            gauge_path=str(path),
        ) from exc

    try:
        origin = validate_run_identity(origin_identity)
    except StoreFormatError as exc:
        raise _gauge_import_refusal(
            "identity",
            f"the origin run.json is not a valid run identity: {exc}",
        ) from exc
    origin_sha = _identity_sha256(origin)
    if str(raw.get("run_id")) != str(origin["run_id"]):
        raise _gauge_import_refusal(
            "identity",
            "the artifact belongs to another run_id "
            f"({raw.get('run_id')!r} != {origin['run_id']!r})",
        )
    if str(raw.get("identity_sha256")) != origin_sha:
        raise _gauge_import_refusal(
            "identity",
            "the artifact belongs to another identity "
            f"({raw.get('identity_sha256')!r} != {origin_sha!r})",
        )

    # ---- frame set: ids, order, content digests, sizes -------------------
    origin_inputs = origin["science_contract"]["inputs"]
    origin_rows = origin_inputs.get("frames")
    if not isinstance(origin_rows, list):
        raise _gauge_import_refusal(
            "inputs", "the origin identity carries no frames inputs section"
        )
    if expected_frame_rows is None:
        # Fail-fast path: no per-frame content digests are available yet.
        # Verify every field that does NOT require reading the corpus.
        expected = [
            {"frame_id": str(frame_id), "order": int(index), "role": "light"}
            for index, frame_id in enumerate(expected_frame_ids)
        ]
        compared_fields = ("frame_id", "order", "role")
    else:
        expected = [dict(row) for row in expected_frame_rows]
        compared_fields = (
            "frame_id", "content_sha256", "nbytes", "order", "role"
        )
    if len(origin_rows) != len(expected):
        raise _gauge_import_refusal(
            "inputs",
            "the frame count differs "
            f"(origin {len(origin_rows)} != current {len(expected)})",
            origin_frames=len(origin_rows),
            current_frames=len(expected),
        )
    for index, (stored, current) in enumerate(zip(origin_rows, expected)):
        for field in compared_fields:
            if stored.get(field) != current.get(field):
                raise _gauge_import_refusal(
                    "inputs",
                    f"frame {index} ({current.get('frame_id')!r}) differs on "
                    f"{field!r}: origin {stored.get(field)!r} != current "
                    f"{current.get(field)!r}",
                    frame_index=int(index),
                    frame_id=str(current.get("frame_id")),
                    field=str(field),
                )
    current_ids = [str(v) for v in expected_frame_ids]
    origin_ids = [str(row.get("frame_id")) for row in origin_rows]
    if origin_ids != current_ids:
        # Never inline thousands of ids in a refusal: name the first difference.
        first = next(
            index for index, (a, b) in enumerate(zip(origin_ids, current_ids))
            if a != b
        ) if len(origin_ids) == len(current_ids) else 0
        raise _gauge_import_refusal(
            "inputs",
            "the frame ORDER differs from the current descriptor order at "
            f"index {first} (origin {origin_ids[first:first + 1]!r} != current "
            f"{current_ids[first:first + 1]!r})",
            frame_index=int(first),
        )

    # ---- canvas ----------------------------------------------------------
    if dict(origin["science_contract"]["canvas"]) != dict(expected_canvas):
        stored_canvas = dict(origin["science_contract"]["canvas"])
        detail = ", ".join(
            f"{key}: origin {stored_canvas.get(key)!r} != current "
            f"{dict(expected_canvas).get(key)!r}"
            for key in sorted(set(stored_canvas) | set(expected_canvas))
            if stored_canvas.get(key) != dict(expected_canvas).get(key)
        )
        raise _gauge_import_refusal(
            "canvas", f"the canvas differs ({detail})"
        )

    # ---- photometric parameters (backend excluded: execution field) ------
    origin_params = dict(origin["science_contract"]["science"].get("params") or {})
    current_params = dict(expected_science_params)
    origin_params.pop("backend", None)
    current_params.pop("backend", None)
    if origin_params != current_params:
        differing = sorted(
            key for key in set(origin_params) | set(current_params)
            if origin_params.get(key) != current_params.get(key)
        )
        # REWORK-1: an UNDECIDED cell-gauge request is not "a photometric
        # parameter that differs": the engine default (auto) is resolved only
        # after the gauge and membership exist and may switch the run to
        # per_cell, so a global-gauge artifact must not be imported blindly.
        # Keep the section, make the detail actionable (and never coerce).
        if differing == ["cell_gauge_mode"]:
            _origin_mode = origin_params.get("cell_gauge_mode")
            _current_mode = current_params.get("cell_gauge_mode")
            if _current_mode in (None, "auto") and _origin_mode not in (
                None, "auto"
            ):
                raise _gauge_import_refusal(
                    "science",
                    "the origin run used cell_gauge_mode="
                    f"{_origin_mode!r}; this run leaves it at the engine "
                    "default (auto), which may resolve to per_cell. Select "
                    "'Global (one gauge)' explicitly to reuse this global "
                    "gauge artifact.",
                    fields=differing,
                    origin_mode=_origin_mode,
                    current_mode=_current_mode,
                )
        raise _gauge_import_refusal(
            "science",
            "the photometric parameters differ on "
            + ", ".join(repr(key) for key in differing),
            fields=differing,
        )

    # ---- gauge coherence (same check as restore_global_gauge) ------------
    origin_gauge = dict(origin["science_contract"]["gauge"])
    frame_ids = tuple(str(v) for v in parsed["frame_ids"])
    if frame_ids != tuple(current_ids):
        raise _gauge_import_refusal(
            "gauge",
            "the artifact frame order differs from the origin identity's",
        )
    if str(parsed["mode"]) != str(expected_mode):
        raise _gauge_import_refusal(
            "gauge",
            f"the artifact gauge mode {parsed['mode']!r} != this run's "
            f"{expected_mode!r}",
        )
    if str(origin_gauge.get("mode")) != str(parsed["mode"]):
        raise _gauge_import_refusal(
            "gauge",
            "the origin identity's gauge mode "
            f"{origin_gauge.get('mode')!r} != the artifact's "
            f"{parsed['mode']!r}",
        )
    reference_frame_id = parsed["reference_frame_id"]
    reference_index = int(parsed["reference_index"])
    if reference_frame_id is not None and str(reference_frame_id) != frame_ids[
        reference_index
    ]:
        raise _gauge_import_refusal(
            "gauge",
            "the artifact's reference_frame_id does not match its own "
            "frame_ids[reference_index]",
        )
    if reference_frame_id != origin_gauge.get("reference_frame_id"):
        raise _gauge_import_refusal(
            "gauge",
            "the artifact's reference frame "
            f"{reference_frame_id!r} != the origin identity's "
            f"{origin_gauge.get('reference_frame_id')!r}",
        )

    from zemosaic.core.canonical_streaming import FixedNormalization

    gauge = FixedNormalization(
        reference_index=reference_index,
        coefficients=parsed["coefficients"],
        norm_active=parsed["norm_active"],
        weights=parsed["weights"],
        weight_active=parsed["weight_active"],
        exclusions=parsed["exclusions"],
    )
    actual_sha = gauge_fingerprint(
        gauge=gauge,
        mode=str(parsed["mode"]),
        frame_ids=frame_ids,
        reference_frame_id=reference_frame_id,
    )
    stored_sha = str(origin_gauge.get("sha256"))
    artifact_sha_value = str(raw.get("gauge_sha256"))
    if actual_sha != stored_sha:
        raise _gauge_import_refusal(
            "gauge",
            "the imported gauge does not reproduce the origin identity's "
            f"gauge hash ({actual_sha} != {stored_sha}); never re-derive or "
            "repair a global gauge",
            restored_sha256=actual_sha,
            stored_sha256=stored_sha,
        )
    if actual_sha != artifact_sha_value:
        raise _gauge_import_refusal(
            "gauge",
            "the artifact's own gauge_sha256 does not match its decoded gauge "
            f"({actual_sha} != {artifact_sha_value})",
            restored_sha256=actual_sha,
            artifact_sha256=artifact_sha_value,
        )

    provenance = {
        "source": "imported_artifact",
        "gauge_path": str(path),
        "origin_run_json_path": (
            None if origin_run_json_path is None else str(origin_run_json_path)
        ),
        "artifact_sha256": str(artifact_sha),
        "gauge_sha256": actual_sha,
        "origin_run_id": str(origin["run_id"]),
        "origin_identity_sha256": origin_sha,
        "frame_count": int(len(frame_ids)),
        "reference_frame_id": reference_frame_id,
        "mode": str(parsed["mode"]),
    }
    return gauge, frame_ids, reference_frame_id, provenance


def import_gauge_artifact(
    gauge_path: str | os.PathLike[str],
    *,
    origin_identity: Mapping[str, Any],
    origin_run_json_path: str | None,
    expected_frame_rows: Sequence[Mapping[str, Any]] | None,
    expected_frame_ids: Sequence[str],
    expected_canvas: Mapping[str, Any],
    expected_science_params: Mapping[str, Any],
    expected_mode: str,
) -> tuple[Any, tuple[str, ...], str | None, dict]:
    """Verify and import a global gauge artifact produced by ANOTHER run.

    Opt-in only: the caller passes an explicit artifact path and the origin
    ``run.json`` identity. Every check is the SAME validator the resume path
    already trusts (artifact self-digest/schema, origin identity + binding,
    frame ids/order/content digests/sizes, canvas, photometry with ``backend``
    excluded, gauge fingerprint).

    The artifact is only READ (no write, no repair, no normalisation). Returns
    ``(gauge, frame_ids, reference_frame_id, provenance)``.

    This is the AUTHORITY: the per-frame content rows are mandatory and their
    ``content_sha256``/``nbytes`` are always verified. The fail-fast
    :func:`precheck_gauge_import` exists only to refuse cheap mismatches before
    the corpus is hashed; it never replaces this check.
    """
    if expected_frame_rows is None:
        raise _gauge_import_refusal(
            "inputs",
            "the full import requires the per-frame content rows; refusing "
            "without them (the content digests are the authority, never the "
            "pre-check alone)",
        )
    return _verify_gauge_import(
        gauge_path,
        origin_identity=origin_identity,
        origin_run_json_path=origin_run_json_path,
        expected_frame_rows=expected_frame_rows,
        expected_frame_ids=expected_frame_ids,
        expected_canvas=expected_canvas,
        expected_science_params=expected_science_params,
        expected_mode=expected_mode,
    )


def precheck_gauge_import(
    gauge_path: str | os.PathLike[str],
    *,
    origin_identity: Mapping[str, Any],
    origin_run_json_path: str | None,
    expected_frame_ids: Sequence[str],
    expected_canvas: Mapping[str, Any],
    expected_science_params: Mapping[str, Any],
    expected_mode: str,
) -> dict:
    """FAIL-FAST pre-check: every cheap contract, ZERO frame hashing.

    Runs the artifact/identity/frame-ids/canvas/photometry/gauge checks before
    the caller reads and hashes the corpus, so a policy-token mismatch refuses
    in seconds instead of after a full-corpus read. Returns the provenance dict
    (identical to the full import's). It does NOT verify content digests — the
    caller must still call :func:`import_gauge_artifact` with the rows, which
    re-verifies everything and is the only gate that authorises the import.
    """
    _gauge, _frame_ids, _reference, provenance = _verify_gauge_import(
        gauge_path,
        origin_identity=origin_identity,
        origin_run_json_path=origin_run_json_path,
        expected_frame_rows=None,
        expected_frame_ids=expected_frame_ids,
        expected_canvas=expected_canvas,
        expected_science_params=expected_science_params,
        expected_mode=expected_mode,
    )
    return provenance


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
        "schema_version": GEOMETRY_ARTIFACT_VERSION,
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
    if str(raw.get("schema_version")) != GEOMETRY_ARTIFACT_VERSION:
        raise ResumeStateError(
            f"geometry artifact version {raw.get('schema_version')!r} is not "
            f"supported ({GEOMETRY_ARTIFACT_VERSION})"
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
