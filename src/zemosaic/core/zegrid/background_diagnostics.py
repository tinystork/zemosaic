"""ZM-ZEGRID-R37-C/D — pre-feather Cell-pair diagnostic artifact (PURE math/artifact code;
OPT-IN production capture seam).

Mission objective
-----------------
The accepted R37-B prototype (``background_model.py``) is scientifically limited
because the canonical final R34F product persists no *pre-feather* Cell-pair
evidence: no source FITS can be re-measured for the exact relation set used at
assembly time. This module is the smallest PURE, deterministic, OFFLINE layer
that captures that evidence from in-memory pre-feather ``PlanCell`` / ``CellPatch``
objects and round-trips it back into the accepted M0/M1 machinery.

Discipline
----------
* **Pure math/artifact code, lazily imported by the explicit OPT-IN capture seam.**
  This module is NOT imported at production module-import time and is NEVER touched
  by the default (capture-disabled) run path. Since ZM-ZEGRID-R37-D it IS imported
  *lazily inside the explicit, default-OFF* ``run_zegrid_mode(
  r37_diagnostic_capture=True)`` capture branch, which publishes the artifact into
  that run's actual output directory. The functions here remain pure/offline for
  standalone use; the capture seam REPLACES the pure default provenance with
  truthful production-capture values (see :func:`capture_provenance`). No default
  change, no corrected mosaic, no correction applied anywhere.
* **Same scientific math.** Pair differences and the coordinate convention are
  produced by the UNCHANGED ``background_model.observe_edge`` via
  ``build_observations``; no duplicate pixel/difference math lives here.
* **Sufficient statistics + capped evidence.** Per relation/channel the payload
  stores theta/information/rhs (sufficient to rerun M1 exactly) and a
  deterministic, stratified, fixed-cap sample of the held-out points. Storage is
  ``O(relations * cap)`` — never ``O(canvas)`` or ``O(frames)``. No global dense
  matrix is ever formed.
* **Strict JSON.** ``json.dumps(payload, allow_nan=False)`` always succeeds.
  Scientific arrays are FINITE by construction here and are REJECTED by the
  loader if they contain ``null``/non-finite values; non-finite is never
  degraded to a fabricated ``0``.
* **Reuses the accepted decision path.** The offline rerun delegates to
  ``background_model.compare_models_from_observations`` (a behaviour-preserving
  extraction of the tail of ``compare_models``); gates, thresholds, solver and
  provenance semantics are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .background_model import (
    BLOCK_PX,
    DEFAULT_FEATHER_PX,
    DEFAULT_MEASUREMENT_PX,
    MODEL_OFFSET_ONLY,
    STATE_INSUFFICIENT_EVIDENCE,
    STATE_ROBUST_PLAN,
    EdgeObservation,
    PlanCell,
    build_observations,
    compare_models_from_observations,
)
from .cell_offset import select_central_cell

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "HOLDOUT_CAP",
    "STRATA_TANGENT",
    "STRATA_NORMAL",
    "COEF_ROUNDTRIP_ATOL",
    "HELDOUT_RMS_REFERENCE_REL_TOL",
    "HELDOUT_RMS_REFERENCE_CONFIG",
    "HELDOUT_RMS_REL_TOL",
    "SYMMETRY_RTOL",
    "PSD_RTOL",
    "RHS_THETA_RTOL",
    "QUALIFICATION_EXACT",
    "QUALIFICATION_RESEARCH_ONLY_CAPPED",
    "CAPTURE_ARTIFACT_BASENAME",
    "CAPTURE_STAGE",
    "CAPTURE_MISSION",
    "CAPTURE_PROVENANCE_STATUS",
    "SOURCE_SET_DIGEST_ALGO",
    "DIAGNOSTIC_PROVENANCE",
    "DiagnosticSchemaError",
    "DiagnosticArtifact",
    "select_capped_holdout",
    "serialise_observation",
    "build_payload",
    "source_set_digest",
    "capture_run_id",
    "capture_provenance",
    "invalidate_capture_target",
    "manifest_links_artifact",
    "load_artifact",
    "dumps",
    "loads",
    "dump_file",
    "load_file",
    "publish_artifact",
    "recompute_decision",
    "estimate_footprint",
    "canonical_witness_estimates",
]

# ---------------------------------------------------------------------------
# Versioning / declared tolerances
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.background_diagnostics"
SCHEMA_VERSION = "r37c-1"

#: Documented FIXED cap of held-out points persisted per relation/channel.
HOLDOUT_CAP = 64
#: Fixed stratification grid (tangent x normal quantile bands) for the cap.
STRATA_TANGENT = 4
STRATA_NORMAL = 4
#: Declared round-trip tolerance for model coefficients (float64).
COEF_ROUNDTRIP_ATOL = 1e-10
#: MEASURED REFERENCE tolerance for held-out RMS when the cap bites, for the ONE
#: declared configuration below. It is NOT an invariant bound: measured values
#: exceed 10% at several supported caps (e.g. 2x2 cap 4 -> ~0.24, cap 8 -> ~0.10,
#: cap 32 -> ~0.11) and are non-monotone in the cap. Never cite it as a global
#: guarantee; a capped held-out metric is approximate, full stop.
HELDOUT_RMS_REFERENCE_REL_TOL = 0.10
#: The exact configuration the reference tolerance was measured on.
HELDOUT_RMS_REFERENCE_CONFIG = "2x2 and 2x3 layouts, seed=7, measurement band 16 px, cap=16"
#: Backwards-compatible alias. Use ONLY as the reference tolerance for
#: ``HELDOUT_RMS_REFERENCE_CONFIG`` -- never as a universal bound.
HELDOUT_RMS_REL_TOL = HELDOUT_RMS_REFERENCE_REL_TOL

#: Documented float64 tolerances for the loader's scientific-consistency gates.
#: Relative to the matrix scale max(1, max|information|) (symmetry/PSD) or to
#: max(1, max|information|) (rhs/theta residual).
SYMMETRY_RTOL = 1e-9
PSD_RTOL = 1e-9
RHS_THETA_RTOL = 1e-6

#: Qualification markers surfaced on the offline decision result.
QUALIFICATION_EXACT = "EXACT"
QUALIFICATION_RESEARCH_ONLY_CAPPED = "RESEARCH_ONLY_HELDOUT_CAPPED"

#: ZM-ZEGRID-R37-D: the FIXED basename of the published capture artifact inside a
#: run's actual output directory. It is never caller-controlled (``publish_artifact``
#: exposes NO basename override), so no path traversal is possible.
CAPTURE_ARTIFACT_BASENAME = "zegrid_r37_pre_feather.json"
#: The exact provenance stage the captured evidence corresponds to: the payload is
#: built AFTER the complete per-Cell patches coexist but BEFORE the inter-cell
#: offset graph, additive offsets and feather assembly run.
CAPTURE_STAGE = "pre_offset_pre_feather"
#: ZM-ZEGRID-R37-D production-capture provenance facts.
CAPTURE_MISSION = "ZM-ZEGRID-R37-D"
CAPTURE_PROVENANCE_STATUS = "production_capture_uncommitted_until_manifest"
#: Declared algorithm for the deterministic source-frame-set digest.
SOURCE_SET_DIGEST_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"
#: Keys that must MATCH for a manifest to "commit" an artifact (M2 cross-link).
CAPTURE_LINKAGE_KEYS = ("capture_id", "filename", "sha256", "bytes")
#: Declared average byte width of one serialised JSON number, used ONLY by the
#: analytic size estimate (a documented model, not a measurement).
_JSON_SCALAR_WIDTH = 20

DIAGNOSTIC_PROVENANCE = {
    "module": "zemosaic.core.zegrid.background_diagnostics",
    "mission": "ZM-ZEGRID-R37-C",
    "status": "research_only_offline_diagnostics",
    "production_wired": False,
    "correction_applied": False,
    "coordinate_system": "global_canvas_normalized_[-1,1]",
    "coordinate_convention": "pixel_centre_+0.5_global_x_from_col_y_from_row",
    "reuses": "zemosaic.core.zegrid.background_model.observe_edge/build_observations",
    "decision_reuse": "zemosaic.core.zegrid.background_model.compare_models_from_observations",
    "holdout_evidence": "deterministic_stratified_quantile_cap_plus_forced_extremes",
    "json_non_finite_policy": "scientific arrays are finite by construction and rejected if null/non-finite",
    "global_dense_matrix": False,
    "no_third_party_code_copied": True,
    "references": [
        "Montage (Caltech/IPAC) — Background Modeling and Rectification / mBgModel: "
        "overlap-difference planar (offset+slope) fitting with iterated global "
        "consistency. https://montage.ipac.caltech.edu/docs/algorithms.html ; "
        "https://montage.ipac.caltech.edu/docs/mBgModel.html",
        "Montage — Jacob et al. 2010, 'Montage: a grid portal and software toolkit "
        "for astronomical image mosaicking', arXiv:1005.4454",
        "PhotometricMosaic (John Murphy) — photometric scale matching + surface "
        "spline relative-gradient model. https://www.photometricmosaic.com/",
        "SDSS ubercalibration (Padmanabhan et al. 2008) — IRLS-based large-scale "
        "relative photometric recalibration over overlap graphs, arXiv:astro-ph/0703454",
        "SWarp (Bertin 2010) — overlap background/scale equalisation: arXiv:1010.0688",
    ],
    "reference_access_note": (
        "Gate-A read-only snapshots archived under "
        ".a2a-reports/ZM-ZEGRID-R37-gateA/references/ (montage_algorithms.html, "
        "montage_bgmodel.html, photometricmosaic.html). Live network to "
        "montage.ipac.caltech.edu / photometricmosaic.com / export.arxiv.org was "
        "partially blocked (connection refused / HTTP 429) at run time."
    ),
}

# NOTE (ZM-ZEGRID-R37-D): the dict above is the **pure/standalone default** shipped
# by ``build_payload`` when it is used on its own (it truthfully describes an
# offline prototype artifact). The explicit production capture seam does NOT ship
# it: it replaces the top-level provenance with :func:`capture_provenance`
# (``production_wired=True``, mission R37-D, capture stage), so a published
# production artifact is never internally contradictory about wiring.


def source_set_digest(identifiers) -> str:
    """Deterministic, order-invariant digest of a source FRAME SET.

    WHAT IS HASHED: the :data:`SOURCE_SET_DIGEST_ALGO` tag followed by the
    SORTED, DE-DUPLICATED set of canonical *relative logical* identifiers (frame
    ``logical_path`` strings — never absolute private paths), each NUL-free and
    newline-terminated. Because the set is sorted and de-duplicated, the digest is
    stable across path object types (``str()``) and across input ORDER, and it
    changes iff the underlying set of identifiers changes.
    """
    uniq = sorted({str(x) for x in identifiers if str(x)})
    h = hashlib.sha256()
    h.update((SOURCE_SET_DIGEST_ALGO + "\n").encode("utf-8"))
    for item in uniq:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def capture_run_id(*, start_ts, pass_label, n_frames, source_set_digest_value, nonce=0) -> str:
    """Stable opaque capture identity (deterministic hash; provenance-only nonce).

    A sha256 over a version tag, the run start timestamp, the pass label
    (``single`` / ``EQ`` / ``ALTZ``), the source frame count, the frame-set digest
    and a high-resolution ``nonce`` (F3: ``time.time_ns()`` generated once per
    ``_run_single`` invocation). Same inputs+nonce -> same id; distinct passes OR
    distinct invocations -> distinct ids even within the same wall-clock second.
    The nonce is provenance only, NEVER scientific RNG. Returned as a plain hex
    string (strict-JSON friendly).
    """
    h = hashlib.sha256()
    for part in (
        "zemosaic.zegrid.r37d.capture.v2",
        str(start_ts),
        str(pass_label),
        str(int(n_frames)),
        str(source_set_digest_value),
        str(nonce),
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def invalidate_capture_target(output_dir) -> bool:
    """Fail-closed invalidation of the FIXED capture target inside ``output_dir``.

    ZM-ZEGRID-R37-D (F1): removes ONLY :data:`CAPTURE_ARTIFACT_BASENAME` (never any
    caller-supplied filename), is IDEMPOTENT (missing target -> no-op, returns
    False), fsyncs the parent directory best-effort, and raises a clear
    :class:`RuntimeError` if an EXISTING fixed target cannot be invalidated. Used
    at requested-run START so a pre-publication abort cannot leave a stale, still
    committed-looking artifact. Returns True iff an existing target was removed.
    """
    out = Path(output_dir).expanduser()
    target = out / CAPTURE_ARTIFACT_BASENAME
    if not target.exists():
        return False
    try:
        os.unlink(target)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(
            f"capture artifact: cannot invalidate existing fixed target {target}: {exc}"
        ) from exc
    _fsync_dir_best_effort(out)
    return True


def _readback_digest(path):
    """SHA-256 + byte count of the on-disk published file (post-publication)."""
    digest = hashlib.sha256()
    n_bytes = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
            n_bytes += len(chunk)
    return digest.hexdigest(), int(n_bytes)


def capture_provenance(*, capture_id, run_label, n_source_frames, source_set_digest_value,
                       source_set_digest_algo=SOURCE_SET_DIGEST_ALGO,
                       halo_px, measurement_px, feather_px,
                       effective_cell_gauge_mode="per_cell") -> dict:
    """Truthful, SINGLE-VALUED provenance for a PUBLISHED production capture.

    Reuses the reference/coordinate context of the pure default but overrides the
    wiring facts so the artifact never simultaneously claims ``production_wired``
    both false and true. The nested ``capture`` block agrees with the top level
    (it carries the run identity + measured widths, not a second wiring claim).
    """
    prov = dict(DIAGNOSTIC_PROVENANCE)
    prov.update({
        "mission": CAPTURE_MISSION,
        "status": CAPTURE_PROVENANCE_STATUS,
        "artifact_kind": "production_capture",
        "production_wired": True,
        "capture_stage": CAPTURE_STAGE,
        "correction_applied": False,
        "committed_requirement": (
            "UNCOMMITTED unless a completed zegrid_manifest.json cross-links the SAME "
            "capture_id + filename + sha256 + bytes; a missing or mismatching manifest "
            "is non-qualifying ({})".format(", ".join(CAPTURE_LINKAGE_KEYS))
        ),
        "capture": {
            "capture_stage": CAPTURE_STAGE,
            "capture_id": str(capture_id),
            "run_label": str(run_label),
            "n_source_frames": int(n_source_frames),
            "source_set_digest": str(source_set_digest_value),
            "source_set_digest_algo": str(source_set_digest_algo),
            "effective_cell_gauge_mode": str(effective_cell_gauge_mode),
            "halo_px": int(halo_px),
            "measurement_px": int(measurement_px),
            "feather_px": int(feather_px),
        },
    })
    return prov


def manifest_links_artifact(capture_meta, manifest_capture_meta) -> bool:
    """True iff a COMPLETED manifest commits this artifact (M2 cross-link rule).

    An artifact is *uncommitted* unless a manifest's capture metadata matches it on
    every :data:`CAPTURE_LINKAGE_KEYS` field (``capture_id``, ``filename``,
    ``sha256``, ``bytes``). A missing manifest block -> False (non-qualifying).
    """
    if not isinstance(manifest_capture_meta, Mapping):
        return False
    if not isinstance(capture_meta, Mapping):
        return False
    for key in CAPTURE_LINKAGE_KEYS:
        if capture_meta.get(key) != manifest_capture_meta.get(key):
            return False
    return True


class DiagnosticSchemaError(ValueError):
    """The diagnostic payload is malformed / violates the declared schema."""


# ---------------------------------------------------------------------------
# Adapters (PlanCell / CellPatch-equivalent, no array copies)
# ---------------------------------------------------------------------------

def _as_plan_cell(obj) -> PlanCell:
    """Accept a pre-feather ``PlanCell`` or ``CellPatch``-equivalent object.

    Both carry ``cell_id``, global patch ``bounds`` (y0,y1,x0,x1), ``science``
    (2-D mono or 3-D) and an optional ``valid_mask``; the arrays are reused by
    reference (never copied), so a ``CellPatch`` works directly.
    """
    if isinstance(obj, PlanCell):
        return obj
    for attr in ("cell_id", "bounds", "science"):
        if not hasattr(obj, attr):
            raise TypeError(
                f"expected PlanCell/CellPatch-equivalent, missing {attr!r} on "
                f"{type(obj).__name__}"
            )
    bounds = tuple(int(v) for v in obj.bounds)
    if len(bounds) != 4:
        raise ValueError(f"cell {obj.cell_id!r}: bounds must have 4 entries")
    return PlanCell(
        cell_id=str(obj.cell_id),
        bounds=bounds,
        science=obj.science,
        valid_mask=getattr(obj, "valid_mask", None),
    )


# ---------------------------------------------------------------------------
# Deterministic stratified capped held-out selection
# ---------------------------------------------------------------------------

def _bin_index(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or values.size == 0:
        return np.zeros(values.shape, dtype=np.int64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, k + 1)[1:-1])
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def _systematic_positions(m: int, q: int) -> list[int]:
    """Deterministic evenly-spaced positions inside ``range(m)`` (``q`` picks)."""
    if q <= 0 or m <= 0:
        return []
    if q >= m:
        return list(range(m))
    out: list[int] = []
    last = -1
    for i in range(q):
        p = int(np.floor((i + 0.5) * m / q))
        if p <= last:
            p = last + 1
        if p >= m:
            p = m - 1
        if p > last:
            out.append(p)
            last = p
    return out


def select_capped_holdout(xy, diff, orientation, *, cap: int = HOLDOUT_CAP,
                          strata_t: int = STRATA_TANGENT,
                          strata_n: int = STRATA_NORMAL):
    """Deterministically subsample held-out ``(xy, diff)`` down to ``cap`` points.

    Strategy (documented, order-independent given the deterministic input order):

    1. **Forced boundary/extreme points** — the min/max each of the tangent and
       normal coordinate plus the four diagonal extremes. This guarantees exact
       tangent/normal range coverage of the full held-out set.
    2. **Quantile stratification** — the remaining points are binned into a fixed
       ``strata_t x strata_n`` grid of (tangent, normal) quantile bands.
    3. **Systematic within-stratum picks** — the leftover budget is allocated
       across strata (with at least one pick per occupied stratum when the budget
       allows) and filled by evenly-spaced positions inside each stratum.

    Never an order-biased prefix. Returns ``(xy_sel, diff_sel, capped)`` where
    ``capped`` is False iff every point was kept.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    diff = np.asarray(diff, dtype=np.float64).ravel()
    if xy.shape[0] != diff.shape[0]:
        raise ValueError(f"holdout_xy/diff length mismatch: {xy.shape[0]} != {diff.shape[0]}")
    if orientation not in ("h", "v"):
        raise ValueError(f"orientation must be 'h' or 'v', got {orientation!r}")
    if int(cap) < 4:
        raise ValueError("holdout cap must be >= 4")
    n = diff.shape[0]
    if n <= cap:
        return xy, diff, False

    # Tangent runs along the shared boundary; normal is perpendicular to it.
    if orientation == "h":
        tangent, normal = xy[:, 1], xy[:, 0]
    else:
        tangent, normal = xy[:, 0], xy[:, 1]

    forced = sorted({
        int(np.argmin(tangent)), int(np.argmax(tangent)),
        int(np.argmin(normal)), int(np.argmax(normal)),
        int(np.argmin(tangent + normal)), int(np.argmax(tangent + normal)),
        int(np.argmin(tangent - normal)), int(np.argmax(tangent - normal)),
    })[:cap]
    forced_set = set(forced)
    pool = [i for i in range(n) if i not in forced_set]
    budget = cap - len(forced)

    selected = list(forced)
    if budget > 0 and pool:
        pt = tangent[pool]
        pn = normal[pool]
        tb = _bin_index(pt, int(strata_t))
        nb = _bin_index(pn, int(strata_n))
        strata: dict[tuple[int, int], list[int]] = {}
        for j, idx in enumerate(pool):
            strata.setdefault((int(tb[j]), int(nb[j])), []).append(idx)
        keys = sorted(strata)
        sizes = {k: len(strata[k]) for k in keys}

        quota = {k: 0 for k in keys}
        if budget >= len(keys):
            # Guarantee every occupied stratum is represented.
            for k in keys:
                quota[k] = 1
        else:
            for k in sorted(keys, key=lambda k: (-sizes[k], k))[:budget]:
                quota[k] = 1

        extra = budget - sum(quota.values())
        if extra > 0:
            exact = {k: extra * sizes[k] / float(len(pool)) for k in keys}
            for k in keys:
                quota[k] = min(sizes[k], quota[k] + int(np.floor(exact[k])))
            order = sorted(keys, key=lambda k: (-(exact[k] - np.floor(exact[k])), k))
            assigned = sum(quota.values())
            i = 0
            guard = 0
            limit = 4 * len(order) * max(budget, 1) + 4
            while assigned < budget and i < limit:
                k = order[i % len(order)]
                if quota[k] < sizes[k]:
                    quota[k] += 1
                    assigned += 1
                i += 1
                guard += 1
                if guard >= limit:
                    break

        for k in keys:
            members = strata[k]
            for p in _systematic_positions(sizes[k], quota[k]):
                selected.append(members[p])

    sel = sorted(set(selected))
    if len(sel) > cap:
        keep = forced + [i for i in sel if i not in forced_set]
        sel = sorted(keep[:cap])
    return xy[sel], diff[sel], True


# ---------------------------------------------------------------------------
# Serialisation of one EdgeObservation (finite by construction)
# ---------------------------------------------------------------------------

def _finite(x, where: str) -> float:
    f = float(x)
    if not np.isfinite(f):
        raise ValueError(f"{where}: expected a finite scientific value, got {x!r}")
    return f


def serialise_observation(obs: EdgeObservation, *, holdout_cap: int = HOLDOUT_CAP,
                          strata_t: int = STRATA_TANGENT,
                          strata_n: int = STRATA_NORMAL) -> dict:
    """Compact, STRICT-JSON record reconstructing ``obs`` offline.

    ``n_holdout`` is the FULL held-out support count (so the M0 support weights
    reproduce exactly); ``holdout_xy``/``holdout_diff`` carry the capped evidence
    subset and ``holdout_capped`` flags when that subset is a subsample.
    """
    theta = np.asarray(obs.theta_global, dtype=np.float64).ravel()
    info = np.asarray(obs.information, dtype=np.float64)
    rhs = np.asarray(obs.rhs, dtype=np.float64).ravel()
    if theta.shape != (3,) or rhs.shape != (3,) or info.shape != (3, 3):
        raise ValueError(f"relation {obs.key()}: unexpected sufficient-statistic shapes")
    xy, diff, capped = select_capped_holdout(
        obs.holdout_xy, obs.holdout_diff, obs.orientation,
        cap=holdout_cap, strata_t=strata_t, strata_n=strata_n,
    )
    return {
        "u": str(obs.u),
        "v": str(obs.v),
        "channel": int(obs.channel),
        "orientation": str(obs.orientation),
        "n_train": int(obs.n_train),
        "n_holdout": int(obs.n_holdout),
        "n_holdout_kept": int(xy.shape[0]),
        "holdout_capped": bool(capped),
        "effective_train": _finite(obs.effective_train, f"{obs.key()} effective_train"),
        "tangent_blocks": int(obs.tangent_blocks),
        "normal_blocks": int(obs.normal_blocks),
        "robust_sigma": _finite(obs.robust_sigma, f"{obs.key()} robust_sigma"),
        "rejection_fraction": _finite(obs.rejection_fraction, f"{obs.key()} rejection_fraction"),
        "theta_global": [_finite(v, f"{obs.key()} theta_global") for v in theta],
        "information": [[_finite(v, f"{obs.key()} information") for v in row] for row in info],
        "rhs": [_finite(v, f"{obs.key()} rhs") for v in rhs],
        "level_z": _finite(obs.level_z, f"{obs.key()} level_z"),
        "curvature_z": _finite(obs.curvature_z, f"{obs.key()} curvature_z"),
        "level_mean": _finite(obs.level_mean, f"{obs.key()} level_mean"),
        "local_offset_train": _finite(obs.local_offset_train, f"{obs.key()} local_offset_train"),
        "measurement_px": (None if obs.measurement_px is None else int(obs.measurement_px)),
        "overlap_px": int(obs.overlap_px),
        "band_used_px": int(obs.band_used_px),
        "band_clamped": bool(obs.band_clamped),
        "holdout_xy": [[_finite(v, f"{obs.key()} holdout_xy") for v in row] for row in xy],
        "holdout_diff": [_finite(v, f"{obs.key()} holdout_diff") for v in diff],
    }


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def build_payload(cells, nx, ny, canvas_hw, *, channels=None, anchor=None,
                  measurement_px=DEFAULT_MEASUREMENT_PX,
                  feather_px=DEFAULT_FEATHER_PX, block_px=BLOCK_PX,
                  holdout_cap: int = HOLDOUT_CAP,
                  strata_t: int = STRATA_TANGENT,
                  strata_n: int = STRATA_NORMAL) -> dict:
    """Capture the pre-feather Cell-pair evidence as a strict-JSON payload.

    Pair differences / coordinate convention come from the UNCHANGED
    ``background_model.build_observations`` (same global-pixel math and
    measurement band as the accepted prototype). No FITS, no rasterisation, no
    dense canvas object is formed.
    """
    plan_cells = {cid: _as_plan_cell(obj) for cid, obj in cells.items()}
    if not plan_cells:
        raise ValueError("no cells provided")
    if channels is None:
        channels = plan_cells[next(iter(plan_cells))].channels()
    channels = int(channels)
    cell_ids = tuple(f"r{r:04d}c{c:04d}" for r in range(ny) for c in range(nx)
                     if f"r{r:04d}c{c:04d}" in plan_cells)
    if not cell_ids:
        raise ValueError("no cells provided")
    if anchor is None:
        anchor = select_central_cell(nx, ny)
        if anchor not in plan_cells:
            anchor = cell_ids[0]
    if anchor not in cell_ids:
        raise ValueError(f"anchor {anchor!r} is not among the layout cells")

    obs_raw, dropped = build_observations(
        plan_cells, nx, ny, canvas_hw, channels=channels,
        block_px=block_px, measurement_px=measurement_px,
    )
    meta = {
        "nx": int(nx),
        "ny": int(ny),
        "canvas_hw": [int(canvas_hw[0]), int(canvas_hw[1])],
        "channels": channels,
        "anchor": str(anchor),
        "cell_ids": [str(c) for c in cell_ids],
        "measurement_px": (None if measurement_px is None else int(measurement_px)),
        "block_px": int(block_px),
        "feather_px": int(feather_px),
        "holdout_cap": int(holdout_cap),
        "holdout_strata": [int(strata_t), int(strata_n)],
        "normalization": "global_canvas_normalized_[-1,1]",
        "coordinate_convention": "pixel_centre_+0.5_global_x_from_col_y_from_row",
        "n_relations": len(obs_raw),
        "n_dropped": len(dropped),
    }
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(DIAGNOSTIC_PROVENANCE),
        "meta": meta,
        "dropped": [
            {
                "u": str(d["u"]),
                "v": str(d["v"]),
                "channel": (None if d.get("channel") is None else int(d["channel"])),
                "reason": str(d["reason"]),
            }
            for d in dropped
        ],
        "relations": [
            serialise_observation(o, holdout_cap=holdout_cap,
                                  strata_t=strata_t, strata_n=strata_n)
            for o in obs_raw
        ],
    }


# ---------------------------------------------------------------------------
# Strict loader / validation
# ---------------------------------------------------------------------------

_TOP_KEYS = {"schema", "schema_version", "provenance", "meta", "dropped", "relations"}
_META_KEYS = {
    "nx", "ny", "canvas_hw", "channels", "anchor", "cell_ids", "measurement_px",
    "block_px", "feather_px", "holdout_cap", "holdout_strata", "normalization",
    "coordinate_convention", "n_relations", "n_dropped",
}
_DROP_KEYS = {"u", "v", "channel", "reason"}
_REL_KEYS = {
    "u", "v", "channel", "orientation", "n_train", "n_holdout", "n_holdout_kept",
    "holdout_capped", "effective_train", "tangent_blocks", "normal_blocks",
    "robust_sigma", "rejection_fraction", "theta_global", "information", "rhs",
    "level_z", "curvature_z", "level_mean", "local_offset_train", "measurement_px",
    "overlap_px", "band_used_px", "band_clamped", "holdout_xy", "holdout_diff",
}


def _req_mapping(x, where: str) -> Mapping:
    if not isinstance(x, Mapping):
        raise DiagnosticSchemaError(f"{where}: expected an object/mapping")
    return x


def _exact_keys(m: Mapping, allowed: set, where: str):
    extra = set(m) - allowed
    missing = allowed - set(m)
    if extra:
        raise DiagnosticSchemaError(f"{where}: unexpected key(s) {sorted(extra)}")
    if missing:
        raise DiagnosticSchemaError(f"{where}: missing key(s) {sorted(missing)}")


def _int(x, where: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise DiagnosticSchemaError(f"{where}: expected an integer, got {type(x).__name__}")
    return int(x)


def _num(x, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise DiagnosticSchemaError(f"{where}: expected a finite number, got {type(x).__name__}")
    f = float(x)
    if not np.isfinite(f):
        raise DiagnosticSchemaError(f"{where}: non-finite scientific value is not allowed")
    return f


def _bool(x, where: str) -> bool:
    if not isinstance(x, bool):
        raise DiagnosticSchemaError(f"{where}: expected a boolean")
    return x


def _str(x, where: str) -> str:
    if not isinstance(x, str) or not x:
        raise DiagnosticSchemaError(f"{where}: expected a non-empty string")
    return x


def _finite_vec(x, n: int, where: str) -> list[float]:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise DiagnosticSchemaError(f"{where}: expected a list of length {n}")
    return [_num(v, f"{where}[{i}]") for i, v in enumerate(x)]


def _optional_int(x, where: str):
    if x is None:
        return None
    return _int(x, where)


def _finite_vec_list(x, where: str) -> list[float]:
    if not isinstance(x, (list, tuple)):
        raise DiagnosticSchemaError(f"{where}: expected a list")
    return [_num(v, f"{where}[{i}]") for i, v in enumerate(x)]


_CELL_ID_RE = re.compile(r"^r(\d{4})c(\d{4})$")


def _parse_cell_id(cid: str, where: str):
    m = _CELL_ID_RE.match(cid)
    if m is None:
        raise DiagnosticSchemaError(f"{where}: cell id {cid!r} is not of the form rRRRRcCCCC")
    return int(m.group(1)), int(m.group(2))


def _grid_member(cid: str, nx: int, ny: int, where: str):
    r, c = _parse_cell_id(cid, where)
    if not (0 <= r < ny and 0 <= c < nx):
        raise DiagnosticSchemaError(f"{where}: cell id {cid!r} is outside the {nx}x{ny} grid")
    return r, c


def _canonical_edge(u: str, v: str, orientation: str, nx: int, ny: int, where: str):
    """Enforce a canonical declared layout relation (u -> right OR u -> down).

    ``build_observations`` only ever emits ``grid_adjacency`` edges: the right
    neighbour (same row, ``c+1``, orientation ``h``) or the down neighbour
    (``r+1``, same column, orientation ``v``). Reversed / diagonal / non-adjacent
    identities are rejected (r37c-1 prefers rejection over canonicalisation).
    """
    ur, uc = _grid_member(u, nx, ny, f"{where}.u")
    vr, vc = _grid_member(v, nx, ny, f"{where}.v")
    if (ur, uc) == (vr, vc):
        raise DiagnosticSchemaError(f"{where}: self relation {u!r}")
    dr, dc = vr - ur, vc - uc
    if (dr, dc) == (0, 1):
        expected = "h"
    elif (dr, dc) == (1, 0):
        expected = "v"
    else:
        raise DiagnosticSchemaError(
            f"{where}: relation {u}->{v} is not a canonical 4-neighbour edge "
            "(expected the right or down neighbour)"
        )
    if orientation is not None and orientation != expected:
        raise DiagnosticSchemaError(
            f"{where}: orientation {orientation!r} inconsistent with {u}->{v} "
            f"(expected {expected!r})"
        )
    return expected


def _validate_information(info: np.ndarray, where: str):
    """Finite + symmetric + PSD (within documented float64 tolerances).

    Zero information is rejected: a relation carrying no constraint must be
    dropped, never stored with fabricated confidence.
    """
    scale = max(1.0, float(np.max(np.abs(info))))
    asym = float(np.max(np.abs(info - info.T)))
    if asym > SYMMETRY_RTOL * scale:
        raise DiagnosticSchemaError(
            f"{where}.information: not symmetric (asym={asym:.3e}, tol={SYMMETRY_RTOL * scale:.3e})"
        )
    evals = np.linalg.eigvalsh(0.5 * (info + info.T))
    if float(np.max(evals)) <= SYMMETRY_RTOL * scale:
        raise DiagnosticSchemaError(f"{where}.information: zero/absent information is not a valid relation")
    if float(np.min(evals)) < -PSD_RTOL * scale:
        raise DiagnosticSchemaError(
            f"{where}.information: not positive semi-definite (min eig={evals.min():.3e})"
        )


def _reconstruct_observation(rel: Mapping, i: int, *, channels: int, nx: int, ny: int,
                             cell_id_set: set, meta_measurement_px, cap: int,
                             ) -> EdgeObservation:
    where = f"relations[{i}]"
    _req_mapping(rel, where)
    _exact_keys(rel, _REL_KEYS, where)
    u = _str(rel["u"], f"{where}.u")
    v = _str(rel["v"], f"{where}.v")
    if u not in cell_id_set or v not in cell_id_set:
        raise DiagnosticSchemaError(f"{where}: endpoints {u!r}/{v!r} are not in meta.cell_ids")
    orientation = _str(rel["orientation"], f"{where}.orientation")
    if orientation not in ("h", "v"):
        raise DiagnosticSchemaError(f"{where}.orientation: expected 'h' or 'v'")
    _canonical_edge(u, v, orientation, nx, ny, where)

    channel = _int(rel["channel"], f"{where}.channel")
    if not (0 <= channel < channels):
        raise DiagnosticSchemaError(f"{where}.channel: {channel} outside [0,{channels})")
    measurement_px = _optional_int(rel["measurement_px"], f"{where}.measurement_px")
    if measurement_px is not None and measurement_px < 1:
        raise DiagnosticSchemaError(f"{where}.measurement_px: must be positive when set")
    if measurement_px != meta_measurement_px:
        raise DiagnosticSchemaError(
            f"{where}.measurement_px: {measurement_px} inconsistent with meta.measurement_px "
            f"{meta_measurement_px}"
        )

    info_raw = rel["information"]
    if not isinstance(info_raw, (list, tuple)) or len(info_raw) != 3:
        raise DiagnosticSchemaError(f"{where}.information: expected 3 rows")
    information = np.array(
        [_finite_vec(row, 3, f"{where}.information[{r}]") for r, row in enumerate(info_raw)],
        dtype=np.float64,
    )
    theta = np.asarray(_finite_vec(rel["theta_global"], 3, f"{where}.theta_global"),
                       dtype=np.float64)
    rhs = np.asarray(_finite_vec(rel["rhs"], 3, f"{where}.rhs"), dtype=np.float64)
    _validate_information(information, where)
    # rhs == information @ theta_global holds exactly for a least-squares solution
    # (verified to ~1e-13 relative on honest payloads, incl. zero-noise).
    prod = information @ theta
    vscale = max(1.0, float(np.max(np.abs(prod))), float(np.max(np.abs(rhs))))
    if float(np.max(np.abs(prod - rhs))) > RHS_THETA_RTOL * vscale:
        raise DiagnosticSchemaError(
            f"{where}: rhs is inconsistent with information@theta_global"
        )

    holdout_xy = rel["holdout_xy"]
    if not isinstance(holdout_xy, (list, tuple)):
        raise DiagnosticSchemaError(f"{where}.holdout_xy: expected a list of [x, y] pairs")
    xy = np.array([_finite_vec(p, 2, f"{where}.holdout_xy[{k}]")
                   for k, p in enumerate(holdout_xy)], dtype=np.float64).reshape(-1, 2)
    diff = np.asarray(_finite_vec_list(rel["holdout_diff"], f"{where}.holdout_diff"),
                      dtype=np.float64)

    n_kept = _int(rel["n_holdout_kept"], f"{where}.n_holdout_kept")
    n_holdout = _int(rel["n_holdout"], f"{where}.n_holdout")
    n_train = _int(rel["n_train"], f"{where}.n_train")
    if n_kept < 0 or n_holdout < 0 or n_train < 0:
        raise DiagnosticSchemaError(f"{where}: negative support count")
    if n_holdout > 0 and n_kept == 0:
        raise DiagnosticSchemaError(
            f"{where}: n_holdout_kept==0 while n_holdout>0 (evidence silently dropped)"
        )
    if n_holdout == 0 and n_kept != 0:
        raise DiagnosticSchemaError(f"{where}: n_holdout_kept>0 while n_holdout==0")
    if n_kept > n_holdout:
        raise DiagnosticSchemaError(f"{where}: kept {n_kept} > full holdout {n_holdout}")
    if n_kept > cap:
        raise DiagnosticSchemaError(f"{where}: kept {n_kept} exceeds meta.holdout_cap {cap}")
    capped_flag = _bool(rel["holdout_capped"], f"{where}.holdout_capped")
    if capped_flag != (n_kept < n_holdout):
        raise DiagnosticSchemaError(
            f"{where}.holdout_capped: {capped_flag} inconsistent with "
            f"n_holdout_kept={n_kept} / n_holdout={n_holdout}"
        )
    if xy.shape[0] != diff.shape[0] or xy.shape[0] != n_kept:
        raise DiagnosticSchemaError(
            f"{where}: holdout length mismatch (xy={xy.shape[0]}, diff={diff.shape[0]}, "
            f"n_holdout_kept={n_kept})"
        )

    effective_train = _num(rel["effective_train"], f"{where}.effective_train")
    if effective_train < 0.0:
        raise DiagnosticSchemaError(f"{where}.effective_train: must be >= 0")
    tangent_blocks = _int(rel["tangent_blocks"], f"{where}.tangent_blocks")
    normal_blocks = _int(rel["normal_blocks"], f"{where}.normal_blocks")
    if tangent_blocks < 0 or normal_blocks < 0:
        raise DiagnosticSchemaError(f"{where}: negative block occupancy")
    robust_sigma = _num(rel["robust_sigma"], f"{where}.robust_sigma")
    if robust_sigma < 0.0:
        raise DiagnosticSchemaError(f"{where}.robust_sigma: must be >= 0")
    rejection_fraction = _num(rel["rejection_fraction"], f"{where}.rejection_fraction")
    if not (0.0 <= rejection_fraction <= 1.0):
        raise DiagnosticSchemaError(f"{where}.rejection_fraction: outside [0,1]")
    level_z = _num(rel["level_z"], f"{where}.level_z")
    curvature_z = _num(rel["curvature_z"], f"{where}.curvature_z")
    if level_z < 0.0 or curvature_z < 0.0:
        raise DiagnosticSchemaError(f"{where}: level_z/curvature_z must be >= 0 (absolute z)")
    overlap_px = _int(rel["overlap_px"], f"{where}.overlap_px")
    band_used_px = _int(rel["band_used_px"], f"{where}.band_used_px")
    if overlap_px <= 0 or band_used_px < 0:
        raise DiagnosticSchemaError(f"{where}: overlap/band widths must be positive/nonnegative")
    if band_used_px > overlap_px:
        raise DiagnosticSchemaError(
            f"{where}: band_used_px {band_used_px} > overlap_px {overlap_px}"
        )

    return EdgeObservation(
        u=u, v=v, channel=channel, orientation=orientation,
        n_train=n_train,
        n_holdout=n_holdout,
        effective_train=effective_train,
        tangent_blocks=tangent_blocks,
        normal_blocks=normal_blocks,
        robust_sigma=robust_sigma,
        rejection_fraction=rejection_fraction,
        theta_global=theta,
        information=information,
        rhs=rhs,
        level_z=level_z,
        curvature_z=curvature_z,
        level_mean=_num(rel["level_mean"], f"{where}.level_mean"),
        local_offset_train=_num(rel["local_offset_train"], f"{where}.local_offset_train"),
        measurement_px=measurement_px,
        overlap_px=overlap_px,
        band_used_px=band_used_px,
        band_clamped=_bool(rel["band_clamped"], f"{where}.band_clamped"),
        holdout_xy=xy,
        holdout_diff=diff,
    )


@dataclass(frozen=True)
class DiagnosticArtifact:
    """Validated offline artifact: reconstructed observations + provenance.

    ``heldout_capped`` / ``n_capped_relations`` preserve the capping fact so no
    downstream consumer has to re-derive it from the raw relations.
    """

    payload: dict
    meta: dict
    provenance: dict
    dropped: tuple
    observations: tuple
    heldout_capped: bool = False
    n_capped_relations: int = 0

    def cell_ids(self) -> tuple:
        return tuple(self.meta["cell_ids"])

    def anchor(self) -> str:
        return self.meta["anchor"]

    def channels(self) -> int:
        return int(self.meta["channels"])

    def canvas_hw(self) -> tuple:
        return (int(self.meta["canvas_hw"][0]), int(self.meta["canvas_hw"][1]))

    def heldout_exact(self) -> bool:
        return not self.heldout_capped


def load_artifact(payload) -> DiagnosticArtifact:
    """Validate a payload and reconstruct the ``EdgeObservation`` tuple."""
    obj = _req_mapping(payload, "payload")
    _exact_keys(obj, _TOP_KEYS, "payload")
    if obj["schema"] != SCHEMA_NAME:
        raise DiagnosticSchemaError(f"payload.schema: unexpected {obj['schema']!r}")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise DiagnosticSchemaError(
            f"payload.schema_version: unexpected {obj['schema_version']!r}"
        )
    meta = dict(_req_mapping(obj["meta"], "meta"))
    _exact_keys(meta, _META_KEYS, "meta")
    channels = _int(meta["channels"], "meta.channels")
    if channels < 1:
        raise DiagnosticSchemaError("meta.channels: must be >= 1")
    nx = _int(meta["nx"], "meta.nx")
    ny = _int(meta["ny"], "meta.ny")
    if nx < 1 or ny < 1:
        raise DiagnosticSchemaError("meta.nx/meta.ny: must be >= 1")

    canvas_hw = meta["canvas_hw"]
    if not isinstance(canvas_hw, (list, tuple)) or len(canvas_hw) != 2:
        raise DiagnosticSchemaError("meta.canvas_hw: expected [H, W]")
    canvas_h = _int(canvas_hw[0], "meta.canvas_hw[0]")
    canvas_w = _int(canvas_hw[1], "meta.canvas_hw[1]")
    if canvas_h <= 0 or canvas_w <= 0:
        raise DiagnosticSchemaError("meta.canvas_hw: dimensions must be positive integers")

    cell_ids_raw = meta["cell_ids"]
    if not isinstance(cell_ids_raw, (list, tuple)) or not cell_ids_raw:
        raise DiagnosticSchemaError("meta.cell_ids: expected a non-empty list")
    cell_ids = [_str(c, f"meta.cell_ids[{i}]") for i, c in enumerate(cell_ids_raw)]
    if len(set(cell_ids)) != len(cell_ids):
        raise DiagnosticSchemaError("meta.cell_ids: duplicates are not allowed")
    order = []
    for i, cid in enumerate(cell_ids):
        order.append(_grid_member(cid, nx, ny, f"meta.cell_ids[{i}]"))
    if order != sorted(order):
        raise DiagnosticSchemaError("meta.cell_ids: must be in canonical row-major order")
    cell_id_set = set(cell_ids)
    anchor = _str(meta["anchor"], "meta.anchor")
    if anchor not in cell_id_set:
        raise DiagnosticSchemaError("meta.anchor: not among meta.cell_ids")

    cap = _int(meta["holdout_cap"], "meta.holdout_cap")
    if cap < 4:
        raise DiagnosticSchemaError("meta.holdout_cap: must be >= 4")
    strata_raw = meta["holdout_strata"]
    if not isinstance(strata_raw, (list, tuple)) or len(strata_raw) != 2:
        raise DiagnosticSchemaError("meta.holdout_strata: expected [tangent, normal]")
    strata_t = _int(strata_raw[0], "meta.holdout_strata[0]")
    strata_n = _int(strata_raw[1], "meta.holdout_strata[1]")
    if strata_t < 1 or strata_n < 1:
        raise DiagnosticSchemaError("meta.holdout_strata: must be integers >= 1")

    block_px = _int(meta["block_px"], "meta.block_px")
    if block_px < 1:
        raise DiagnosticSchemaError("meta.block_px: must be positive")
    feather_px = _int(meta["feather_px"], "meta.feather_px")
    if feather_px < 0:
        raise DiagnosticSchemaError("meta.feather_px: must be nonnegative")
    meta_measurement_px = _optional_int(meta["measurement_px"], "meta.measurement_px")
    if meta_measurement_px is not None and meta_measurement_px < 1:
        raise DiagnosticSchemaError("meta.measurement_px: must be positive when set")
    _str(meta["normalization"], "meta.normalization")
    _str(meta["coordinate_convention"], "meta.coordinate_convention")
    _int(meta["n_relations"], "meta.n_relations")
    _int(meta["n_dropped"], "meta.n_dropped")

    dropped_raw = obj["dropped"]
    if not isinstance(dropped_raw, (list, tuple)):
        raise DiagnosticSchemaError("dropped: expected a list")
    dropped = []
    for i, d in enumerate(dropped_raw):
        where = f"dropped[{i}]"
        _req_mapping(d, where)
        _exact_keys(d, _DROP_KEYS, where)
        u = _str(d["u"], f"{where}.u")
        v = _str(d["v"], f"{where}.v")
        _canonical_edge(u, v, None, nx, ny, where)
        ch = _optional_int(d["channel"], f"{where}.channel")
        if ch is not None and not (0 <= ch < channels):
            raise DiagnosticSchemaError(f"{where}.channel: {ch} outside [0,{channels})")
        dropped.append({"u": u, "v": v, "channel": ch, "reason": _str(d["reason"], f"{where}.reason")})

    rel_raw = obj["relations"]
    if not isinstance(rel_raw, (list, tuple)):
        raise DiagnosticSchemaError("relations: expected a list")
    observations = []
    seen_keys: set = set()
    n_capped = 0
    for i, rel in enumerate(rel_raw):
        obs = _reconstruct_observation(rel, i, channels=channels, nx=nx, ny=ny,
                                       cell_id_set=cell_id_set,
                                       meta_measurement_px=meta_measurement_px, cap=cap)
        if obs.key() in seen_keys:
            raise DiagnosticSchemaError(
                f"relations[{i}]: duplicate relation key {obs.key()}"
            )
        seen_keys.add(obs.key())
        if bool(rel["holdout_capped"]):
            n_capped += 1
        observations.append(obs)
    if _int(meta["n_relations"], "meta.n_relations") != len(observations):
        raise DiagnosticSchemaError("meta.n_relations: does not match len(relations)")
    if _int(meta["n_dropped"], "meta.n_dropped") != len(dropped):
        raise DiagnosticSchemaError("meta.n_dropped: does not match len(dropped)")
    return DiagnosticArtifact(
        payload=obj,
        meta=meta,
        provenance=dict(_req_mapping(obj["provenance"], "provenance")),
        dropped=tuple(dropped),
        observations=tuple(observations),
        heldout_capped=bool(n_capped > 0),
        n_capped_relations=int(n_capped),
    )


# ---------------------------------------------------------------------------
# Strict JSON I/O
# ---------------------------------------------------------------------------

def dumps(payload, *, indent=None) -> str:
    """Canonical strict JSON: ``allow_nan=False`` + sorted keys (byte-stable)."""
    if indent is None:
        return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(payload, allow_nan=False, sort_keys=True, indent=indent)


def _reject_constant(name: str):
    raise DiagnosticSchemaError(f"strict JSON: literal {name!r} is not allowed")


def loads(text: str) -> DiagnosticArtifact:
    """Parse + validate strict JSON text back into an artifact."""
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except DiagnosticSchemaError:
        raise
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise DiagnosticSchemaError(f"invalid JSON: {exc}") from exc
    return load_artifact(payload)


def dump_file(path, payload) -> int:
    """Write canonical strict JSON to ``path``; returns bytes written."""
    text = dumps(payload)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return len(text.encode("utf-8"))


def load_file(path) -> DiagnosticArtifact:
    with open(path, "r", encoding="utf-8") as fh:
        return loads(fh.read())


def publish_artifact(output_dir, payload) -> dict:
    """Validate + atomically publish a capture artifact into ``output_dir``.

    ZM-ZEGRID-R37-D: the ONLY supported way to persist a capture artifact. The
    basename is FIXED (:data:`CAPTURE_ARTIFACT_BASENAME`) and there is NO caller
    override, so there is no path-traversal surface (only the directory is chosen
    by the caller). It

    0. FAIL-CLOSED INVALIDATION: removes any pre-existing fixed target BEFORE
       validating/writing, so a stale artifact from a previous run can never be
       left looking like the current requested capture;
    1. runs the STRICT loader (``load_artifact``) on the in-memory payload BEFORE
       writing, so a malformed payload is never published;
    2. writes the canonical strict JSON to a SAME-DIRECTORY temp file, flushes and
       fsyncs it, then atomically renames it onto the fixed target (the temp is
       always removed on failure);
    3. re-reads the published bytes and computes their SHA-256 AFTER publication.

    On ANY failure the fixed target is ABSENT and no temp residue remains.
    """
    out = Path(output_dir).expanduser()
    name = CAPTURE_ARTIFACT_BASENAME
    out.mkdir(parents=True, exist_ok=True)
    target = out / name

    # 0. Fail-closed invalidation of any pre-existing target FIRST.
    invalidate_capture_target(out)

    # 1. STRICT validation AFTER invalidation (a malformed payload leaves no target).
    artifact = load_artifact(payload)
    data = dumps(payload).encode("utf-8")

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=str(out))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(tmp_path, target)
        tmp_path = None
        _fsync_dir_best_effort(out)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    # 3. SHA-256 + byte count AFTER publication (from the on-disk bytes). F2: if the
    # readback fails, remove the just-published target so "ANY failure => target
    # absent" stays truthful.
    try:
        digest_hex, n_bytes = _readback_digest(target)
    except BaseException:
        try:
            os.unlink(target)
        except OSError:
            pass
        _fsync_dir_best_effort(out)
        raise
    return {
        "status": "written",
        "filename": name,
        "capture_stage": CAPTURE_STAGE,
        "schema": artifact.payload["schema"],
        "schema_version": artifact.payload["schema_version"],
        "bytes": int(n_bytes),
        "sha256": digest_hex,
        "n_cells": int(len(artifact.meta["cell_ids"])),
        "n_relations": int(len(artifact.observations)),
        "n_dropped": int(len(artifact.dropped)),
        "holdout_cap": int(artifact.meta["holdout_cap"]),
        "holdout_capped": bool(artifact.heldout_capped),
        "holdout_exact": bool(artifact.heldout_exact()),
        "qualification": (
            QUALIFICATION_RESEARCH_ONLY_CAPPED if artifact.heldout_capped else QUALIFICATION_EXACT
        ),
    }


def _fsync_dir_best_effort(path) -> None:
    """Best-effort directory fsync (durability of create/unlink/rename)."""
    try:
        dfd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------
# Offline decision rerun (delegates to the accepted comparison path)
# ---------------------------------------------------------------------------

def recompute_decision(artifact: DiagnosticArtifact, *, cfg=None):
    """Rerun the accepted M0/M1 comparison OFFLINE from the artifact.

    Delegates to ``background_model.compare_models_from_observations`` (the
    behaviour-preserving extraction of ``compare_models``) so gates, thresholds
    and the solver are exactly the accepted ones. Held-out metrics use the capped
    evidence subset and are therefore approximate.

    Qualification semantics (diagnostic layer ONLY; the accepted ``background_model``
    decision logic is untouched):

    * The aggregate capping fact is surfaced on BOTH ``result.gates`` and
      ``result.provenance`` as ``diagnostic_heldout_capped`` / ``diagnostic_heldout_exact``
      plus a machine-readable ``diagnostic_qualification`` marker
      (``EXACT`` | ``RESEARCH_ONLY_HELDOUT_CAPPED``).
    * A CAPPED artifact can never qualify ``ROBUST_PLAN``: if the capped rerun would
      have returned ``ROBUST_PLAN`` it is withheld and downgraded to
      ``INSUFFICIENT_EVIDENCE`` (with ``diagnostic_downgraded_from`` recorded and
      an explicit refusal reason). Other non-qualifying decisions are returned
      unchanged. This never silently presents capped held-out evidence as exact.
    """
    meta = artifact.meta
    result = compare_models_from_observations(
        list(artifact.observations), list(artifact.dropped),
        cell_ids=tuple(meta["cell_ids"]), nx=int(meta["nx"]), ny=int(meta["ny"]),
        canvas_hw=(int(meta["canvas_hw"][0]), int(meta["canvas_hw"][1])),
        channels=int(meta["channels"]), anchor=meta["anchor"],
        measurement_px=meta["measurement_px"], feather_px=int(meta["feather_px"]),
        block_px=int(meta["block_px"]), cfg=cfg,
    )
    capped = bool(artifact.heldout_capped)
    qualification = QUALIFICATION_RESEARCH_ONLY_CAPPED if capped else QUALIFICATION_EXACT
    accepted_decision = str(result.decision)
    accepted_model = str(result.model)
    downgraded_from = None
    downgrade_reason = None
    # LOW-1 cleanup (ZM-ZEGRID-R37-D): the accepted scientific ``gates`` mapping is
    # left STRICTLY intact (only its own PASS/FAIL/... values). Diagnostic
    # qualification is surfaced in the dedicated, namespaced
    # ``result.provenance["diagnostic"]`` block, so:
    #   * ``all(v == "PASS" for v in gates.values())`` keeps its ACCEPTED meaning
    #     (no bool/str diagnostic entry is injected into the gate map), and
    #   * ``decision`` / ``qualification`` are internally unambiguous: the block
    #     records the accepted decision, the effective (conservatively downgraded)
    #     decision, the exact qualification and the downgrade reason.
    # The conservative policy is NOT weakened: a capped artifact still cannot
    # qualify ``ROBUST_PLAN`` (it is downgraded, with an explicit refusal reason).
    if capped and result.decision == STATE_ROBUST_PLAN:
        downgraded_from = STATE_ROBUST_PLAN
        downgrade_reason = "heldout_capped_not_exact"
        result.decision = STATE_INSUFFICIENT_EVIDENCE
        result.model = MODEL_OFFSET_ONLY
        result.refusal_reasons.append(
            "diagnostic_heldout_capped: capped held-out evidence cannot qualify ROBUST_PLAN"
        )
    result.provenance["diagnostic"] = {
        "qualified": qualification == QUALIFICATION_EXACT,
        "qualification": qualification,
        "heldout_capped": capped,
        "heldout_exact": (not capped),
        "n_capped_relations": int(artifact.n_capped_relations),
        "accepted_decision": accepted_decision,
        "accepted_model": accepted_model,
        "effective_decision": str(result.decision),
        "effective_model": str(result.model),
        "downgraded_from": downgraded_from,
        "downgrade_reason": downgrade_reason,
        "downgrade_policy": "heldout_capped_cannot_qualify_robust_plan",
        "note": (
            "Diagnostic-only qualification: 'accepted_decision' is what the accepted "
            "M0/M1 logic returned from the reconstructed evidence; 'effective_decision' "
            "is the conservative decision this offline layer surfaces (capped held-out "
            "evidence is research-only and cannot qualify ROBUST_PLAN). The accepted "
            "scientific 'gates' mapping is left untouched by this layer."
        ),
    }
    return result


# ---------------------------------------------------------------------------
# Analytic footprint estimates (no dense object is ever built)
# ---------------------------------------------------------------------------

def estimate_footprint(*, n_cells: int, n_relations: int,
                       holdout_cap: int = HOLDOUT_CAP,
                       canvas_hw=None) -> dict:
    """Analytic size/count estimate for an artifact of this shape.

    Storage is ``O(n_relations * holdout_cap)``: per relation the payload holds
    the 3x3 information, the 3-vector theta/rhs and at most ``holdout_cap``
    ``(x, y, diff)`` triples. This function only counts scalars; it never builds
    a dense canvas or a dense design matrix.
    """
    if n_cells < 0 or n_relations < 0 or holdout_cap < 0:
        raise ValueError("counts must be non-negative")
    holdout_scalars = 2 * holdout_cap + holdout_cap  # (x, y) pairs + diff
    stats_scalars = 3 + 9 + 3                        # theta + information + rhs
    bookkeeping_scalars = 12                         # ids, ints, bools, flags
    per_relation = holdout_scalars + stats_scalars + bookkeeping_scalars
    total_scalars = n_relations * per_relation
    out = {
        "n_cells": int(n_cells),
        "n_relations": int(n_relations),
        "holdout_cap": int(holdout_cap),
        "holdout_points_total": int(n_relations * holdout_cap),
        "json_scalars_total": int(total_scalars),
        "bytes_per_relation_estimate": int(per_relation * _JSON_SCALAR_WIDTH),
        "estimated_bytes": int(total_scalars * _JSON_SCALAR_WIDTH),
        "scaling": "O(relations*cap)",
        "global_dense_matrix": False,
        "estimate_method": "declared_scalar_width_model",
        "assumed_bytes_per_json_number": _JSON_SCALAR_WIDTH,
    }
    if canvas_hw is not None:
        px = int(canvas_hw[0]) * int(canvas_hw[1])
        out["canvas_pixels"] = px
        out["dense_canvas_float64_bytes"] = px * 8
    return out


def canonical_witness_estimates(holdout_cap: int = HOLDOUT_CAP) -> dict:
    """The two mandated witnesses: canonical 6x4x3 and 10,000-Cell sparse."""
    nx6, ny6, ch6, core = 6, 4, 3, 64
    rel6 = (nx6 * (ny6 - 1) + ny6 * (nx6 - 1)) * ch6
    est6 = estimate_footprint(n_cells=nx6 * ny6, n_relations=rel6,
                              holdout_cap=holdout_cap,
                              canvas_hw=(ny6 * core, nx6 * core))
    side = 100
    cells = side * side
    edges = 2 * side * (side - 1)
    est10k = estimate_footprint(n_cells=cells, n_relations=edges, holdout_cap=holdout_cap)
    return {
        "canonical_6x4x3": {**est6, "layout": "6x4", "channels": ch6, "core_px": core},
        "witness_10000_cells": {**est10k, "layout": "100x100", "channels": 1},
    }
