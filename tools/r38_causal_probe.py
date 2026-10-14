#!/usr/bin/env python3
"""ZM-ZEGRID-R38-A — bounded, read-only, offline causal diagnostic harness.

MISSION
-------
Decide (falsify or confirm) whether *per-Cell gauge choices* (frame set,
reference, coefficients, weights/exclusions) are a dominant cause of the
inter-Cell disagreement observed on the exact Caldwell Windows-816 pre-feather
capture. This is a DIAGNOSTIC. It applies NO correction, wires NOTHING into the
production path, and writes NOTHING under the read-only Caldwell tree.

WHAT IT DOES (three explicit, opt-in modes)
-------------------------------------------
``inventory`` (default, writes only the strict-JSON report)
    1. strictly validates the exact capture + manifest identity/linkage
       (816 frames / 6x4 / EXACT / expected capture id + artifact sha256 +
       source-set digest), independently recomputing the digest and the file
       hashes;
    2. resolves every manifest frame id to EXACTLY one read-only source file;
    3. rebuilds the accepted canvas / layout / Cell membership by REUSING the
       production geometry + layout primitives (no duplicated pixel math);
    4. freezes a deterministic, BIASED DEV relation panel (six geometric
       relations, both orientations, the R37-declared worst edge, median/low
       controls, at least one canvas corner) BEFORE any factor result, and
       labels it NOT QUALIFICATION;
    5. estimates the bounded resource cost of the probe (frames / disk / time).

``gauge`` (needs ``--execute-real``)
    Additionally reproduces the LOCAL per-Cell gauge facts on each selected
    Cell's FULL accepted domain: it streams the Cell's aligned patch from the
    read-only FITS (production decoder / crop / reproject primitives, one frame
    resident) into a scratch aligned cache, then runs the FROZEN estimator
    (``photometric.compute_cell_gauge``) on that provider and records the
    ordered frame ids, reference id/support, coefficients, weights, active
    flags and exclusions. Cross-side comparisons are then computed on the
    common active frames.

``factors`` (needs ``--execute-real`` and an existing ``gauge`` run)
    Additionally runs the one-variable-at-a-time factor probe A/B/C/D on the
    bounded shared-pixel ROI of each probed relation, on a FIXED CPU backend,
    reusing the canonical streaming executor with a fixed gauge and a read-only
    ROI provider (no dense N-frame x canvas cube, no duplicated science).

DISCIPLINE
----------
* Module-scope imports are stdlib + numpy only; every ``zemosaic`` import is
  lazy, so importing this tool never activates the engine (import audit below).
* Read-only w.r.t. the source tree: the tool refuses any output/scratch path
  inside the capture dir, the source dir, or the Caldwell tree.
* Strict JSON: ``allow_nan=False``; a non-finite value is emitted as ``null`` /
  an explicit missing marker, never as a fabricated ``0``.
* No new dependency, no network, no third-party code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Schema + frozen expectations for the CHECKED run
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.r38a.causal_probe"
SCHEMA_VERSION = "r38a-1"
MISSION = "ZM-ZEGRID-R38-A-CAUSAL"
CAPTURE_ARTIFACT_BASENAME = "zegrid_r37_pre_feather.json"
MANIFEST_BASENAME = "zegrid_manifest.json"
CAPTURE_STAGE = "pre_offset_pre_feather"
SOURCE_SET_DIGEST_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"

# The exact accepted Windows-816 capture this checked run is recorded against.
CHECKED_IDENTITY = {
    "n_source_frames": 816,
    "canvas_width": 4117,
    "canvas_height": 4377,
    "canvas_id": "e8f1877f16f8cca4",
    "nx": 6,
    "ny": 4,
    "capture_id": "6c311af729425e69c030f87969fe6369dd9e9fc4ab9c133379662d70b65a8e00",
    "artifact_sha256": "a260b3309c39fbfc5264cacc60ba3048c678f83cf8bf6bf9f39c94a134a84032",
    "artifact_bytes": 23252804,
    "source_set_digest": "e3f4c16af30fca1a4c660ba5f76ea6f3d62868dd825bc09a9d83802bfccee3ae",
    "qualification": "EXACT",
    "effective_cell_gauge_mode": "per_cell",
    "halo_px": 8,
    "measurement_px": 16,
    "feather_px": 8,
}

# R37 declared the per-edge p90 worst relation of the Windows-816 witness (from
# the accepted scientific result report). Used ONLY as a named panel input.
DECLARED_WORST_EDGE = ("r0003c0002", "r0003c0003")

# Panel composition rule (frozen BEFORE any factor result).
PANEL_SLOTS = (
    "declared_worst_edge",
    "highest_burden_vertical",
    "highest_burden_overall",
    "median_burden",
    "minimum_burden",
    "corner_relation",
)

# Pre-registered conclusion thresholds (frozen; DEV diagnostic only).
GAUGE_STEP_DOMINANCE_FRAC = 0.80     # |step| >= 80% of observed -> dominant candidate
GAUGE_STEP_PARTIAL_FRAC = 0.50       # |step| >= 50% of observed -> partial contributor
GAUGE_STEP_NEGLIGIBLE_FRAC = 0.05    # |step| <= 5% of observed -> falsifies gauge step
GAUGE_STEP_CONSISTENCY_MAX = 0.25    # mad(|step|) / |step| <= 25% -> stable step
MIN_PROBED_RELATIONS = 2             # fewer probed relations -> INCONCLUSIVE

REPORT_DIR_DEFAULT = Path(
    "/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-R38-A-CAUSAL"
)
SCRATCH_DIR_DEFAULT = Path("/var/tmp/openclaw-tristan/r38a")

# Read-only trees the tool must never write into (defence in depth).
FORBIDDEN_WRITE_ROOTS = (
    Path("/home/tristan/caldwell11"),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ProbeError(RuntimeError):
    """Deterministic, user-facing harness failure (never a traceback dump)."""


class IdentityError(ProbeError):
    pass


class ResolutionError(ProbeError):
    pass


class BudgetError(ProbeError):
    pass


# ---------------------------------------------------------------------------
# Import audit (module scope must never pull ``zemosaic``)
# ---------------------------------------------------------------------------

def module_scope_imports() -> list[str]:
    """Return the fully-qualified module names imported at MODULE scope.

    Parses this file's own AST: the returned list is the static import
    surface of the tool, used by the tests to prove the tool never imports a
    ``zemosaic`` module at import time (all engine access is lazy).
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def import_audit() -> dict:
    """Bounded, strict-JSON import audit (part of every report)."""
    names = module_scope_imports()
    zem = sorted(n for n in names if n == "zemosaic" or n.startswith("zemosaic."))
    return {
        "module_scope_imports": sorted(names),
        "zemosaic_module_scope_imports": zem,
        "engine_access_is_lazy": not zem,
    }


# ---------------------------------------------------------------------------
# Strict JSON helpers
# ---------------------------------------------------------------------------

def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON literal {name!r} is not allowed")


def load_json_strict(path: Path) -> Any:
    """Load JSON refusing NaN/Infinity literals (strict, non-tautological)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh, parse_constant=_reject_constant)


def _fin(x) -> float | None:
    """Finite float or ``None`` (never a fabricated zero)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def dumps_strict(payload: Any) -> str:
    """Canonical strict JSON (sorted keys, allow_nan=False)."""
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_set_digest(identifiers) -> str:
    """Re-implementation of the accepted ``background_diagnostics`` digest.

    The canonical algorithm tag is hashed first, then each SORTED, unique,
    newline-terminated relative logical id. Kept byte-compatible on purpose so
    an independent recomputation can be compared to the artifact/manifest.
    """
    uniq = sorted({str(x) for x in identifiers if str(x)})
    h = hashlib.sha256()
    h.update((SOURCE_SET_DIGEST_ALGO + "\n").encode("utf-8"))
    for item in uniq:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lazy engine access
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _lazy_zemosaic():
    """Import the engine surface lazily; never at module scope."""
    _ensure_src_on_path()
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import auto_layout as zal
    from zemosaic.core.zegrid import background_model as zbm
    from zemosaic.core.zegrid import file_provider as zfp
    from zemosaic.core.zegrid import geometry as zg
    from zemosaic.core.zegrid import photometric as zphot
    from zemosaic.core.zegrid import streaming as zstream
    from zemosaic.core.zegrid import sweep as zsw
    from zemosaic.core.zegrid.science_adapter import MiniTileScienceConfig

    return {
        "zz": zz,
        "zal": zal,
        "zbm": zbm,
        "zfp": zfp,
        "zg": zg,
        "zphot": zphot,
        "zstream": zstream,
        "zsw": zsw,
        "cfg_cls": MiniTileScienceConfig,
    }


def _quiet() -> None:
    """Silence the product's verbose loader logging (stdout noise only)."""
    logging.disable(logging.INFO)


# ---------------------------------------------------------------------------
# 1. Identity validation
# ---------------------------------------------------------------------------

def _require(cond: bool, message: str, exc=IdentityError) -> None:
    if not cond:
        raise exc(message)


def validate_identity(
    capture_dir: Path,
    *,
    expect: str = "checked",
) -> dict:
    """Strictly validate the exact capture + manifest identity/linkage.

    ``expect="checked"`` additionally requires the recorded CHECKED identity
    (816 / 6x4 / EXACT / expected capture id + artifact sha256 + source digest).
    ``expect="any"`` accepts another valid artifact but records the computed
    identity. Both paths recompute the source-set digest, the on-disk artifact
    sha256/bytes and cross-link manifest <-> artifact.
    """
    cap_json = capture_dir / CAPTURE_ARTIFACT_BASENAME
    man_json = capture_dir / MANIFEST_BASENAME
    _require(cap_json.is_file(), f"capture artifact missing: {cap_json.name}")
    _require(man_json.is_file(), f"manifest missing: {man_json.name}")

    artifact = load_json_strict(cap_json)
    manifest = load_json_strict(man_json)
    _require(isinstance(artifact, dict), "capture artifact is not a JSON object")
    _require(isinstance(manifest, dict), "manifest is not a JSON object")

    art_sha = sha256_file(cap_json)
    art_bytes = int(cap_json.stat().st_size)

    # --- artifact provenance + shape facts --------------------------------
    _require(artifact.get("schema") == "zemosaic.zegrid.background_diagnostics",
             "artifact schema mismatch")
    _require(artifact.get("schema_version") == "r37c-1", "artifact schema_version mismatch")
    prov = artifact.get("provenance")
    _require(isinstance(prov, Mapping), "artifact provenance missing")
    cap = prov.get("capture")
    _require(isinstance(cap, Mapping), "artifact provenance.capture missing")
    _require(cap.get("capture_stage") == CAPTURE_STAGE, "artifact capture_stage mismatch")
    _require(bool(prov.get("correction_applied")) is False, "artifact applied a correction")
    _require(bool(prov.get("production_wired")) is True, "artifact not production-wired")

    meta = artifact.get("meta")
    _require(isinstance(meta, Mapping), "artifact meta missing")

    art_facts = {
        "capture_id": str(cap.get("capture_id", "")),
        "n_source_frames": int(cap.get("n_source_frames", -1)),
        "source_set_digest": str(cap.get("source_set_digest", "")),
        "source_set_digest_algo": str(cap.get("source_set_digest_algo", "")),
        "effective_cell_gauge_mode": str(cap.get("effective_cell_gauge_mode", "")),
        "halo_px": int(cap.get("halo_px", -1)),
        "measurement_px": int(cap.get("measurement_px", -1)),
        "feather_px": int(cap.get("feather_px", -1)),
        "n_relations": int(meta.get("n_relations", -1)),
        "n_dropped": int(meta.get("n_dropped", -1)),
        "nx": int(meta.get("nx", -1)),
        "ny": int(meta.get("ny", -1)),
        "canvas_hw": [int(v) for v in meta.get("canvas_hw", [])],
    }

    # --- manifest facts ----------------------------------------------------
    _require(manifest.get("engine") == "zegrid", "manifest engine != zegrid")
    fids = manifest.get("frame_ids")
    _require(isinstance(fids, list) and fids, "manifest frame_ids missing")
    _require(all(isinstance(x, str) and x for x in fids), "manifest frame_ids malformed")
    uniq = sorted(set(fids))
    _require(len(uniq) == len(fids), "manifest frame_ids contain duplicates")
    n_frames = int(manifest.get("n_frames", -1))
    _require(n_frames == len(fids), "manifest n_frames != len(frame_ids)")

    gauge = manifest.get("photometric_gauge")
    _require(isinstance(gauge, Mapping), "manifest photometric_gauge missing")
    effective_mode = str(gauge.get("mode", ""))

    feather = manifest.get("cell_feather")
    _require(isinstance(feather, Mapping), "manifest cell_feather missing")
    _require(bool(feather.get("enabled")) is True, "manifest cell_feather disabled")
    capblk = feather.get("r37_diagnostic_capture")
    _require(isinstance(capblk, Mapping), "manifest r37 capture block missing")

    layout = manifest.get("layout")
    canvas = manifest.get("canvas")
    _require(isinstance(layout, Mapping) and isinstance(canvas, Mapping),
             "manifest canvas/layout missing")

    computed_digest = source_set_digest(uniq)
    _require(
        computed_digest == art_facts["source_set_digest"],
        f"recomputed source-set digest {computed_digest} != artifact "
        f"{art_facts['source_set_digest']}",
    )
    _require(art_facts["source_set_digest_algo"] == SOURCE_SET_DIGEST_ALGO,
             "artifact source_set_digest_algo mismatch")
    _require(str(capblk.get("source_set_digest", "")) == computed_digest,
             "manifest capture block source-set digest mismatch")
    _require(str(capblk.get("source_set_digest_algo", "")) == SOURCE_SET_DIGEST_ALGO,
             "manifest capture block digest algo mismatch")
    _require(int(capblk.get("n_source_frames", -1)) == len(uniq),
             "manifest capture block source count mismatch")

    # --- manifest <-> artifact linkage (non-tautological: disk-derived) ----
    _require(str(capblk.get("capture_id", "")) == art_facts["capture_id"],
             "manifest capture_id != artifact capture_id")
    _require(str(capblk.get("filename", "")) == CAPTURE_ARTIFACT_BASENAME,
             "manifest capture filename mismatch")
    _require(str(capblk.get("sha256", "")) == art_sha,
             "manifest capture sha256 != on-disk artifact sha256")
    _require(int(capblk.get("bytes", -1)) == art_bytes,
             "manifest capture bytes != on-disk artifact bytes")

    # --- EXACT qualification ----------------------------------------------
    qualification = str(capblk.get("qualification", ""))
    holdout_exact = bool(capblk.get("holdout_exact"))
    holdout_capped = bool(capblk.get("holdout_capped"))
    _require(holdout_capped is False, "capture holdout is capped (not EXACT)")
    _require(holdout_exact is True, "capture holdout_exact is not true")
    _require(art_facts["n_dropped"] == 0, "artifact dropped relations")

    computed = {
        "artifact_sha256": art_sha,
        "artifact_bytes": art_bytes,
        "source_set_digest": computed_digest,
        "n_source_frames": len(uniq),
        "canvas_width": int(canvas.get("width", -1)),
        "canvas_height": int(canvas.get("height", -1)),
        "canvas_id": str(canvas.get("id", "")),
        "nx": int(layout.get("nx", -1)),
        "ny": int(layout.get("ny", -1)),
        "qualification": qualification,
        "effective_cell_gauge_mode": effective_mode,
        "capture_id": art_facts["capture_id"],
        "halo_px": art_facts["halo_px"],
        "measurement_px": art_facts["measurement_px"],
        "feather_px": art_facts["feather_px"],
        "n_relations": art_facts["n_relations"],
        "canvas_hw_artifact": art_facts["canvas_hw"],
    }

    # artifact canvas_hw must equal the manifest canvas (row, col order)
    _require(
        art_facts["canvas_hw"] == [computed["canvas_height"], computed["canvas_width"]],
        "artifact canvas_hw != manifest canvas",
    )
    _require(int(meta.get("nx", -1)) == computed["nx"] and int(meta.get("ny", -1)) == computed["ny"],
             "artifact nx/ny != manifest layout")
    _require(effective_mode == "per_cell", "effective cell gauge mode != per_cell")

    if expect == "checked":
        for key, want in CHECKED_IDENTITY.items():
            got = computed.get(key)
            _require(got == want, f"checked identity mismatch for {key!r}: got {got!r} want {want!r}")
    else:
        _require(expect == "any", f"unknown expect {expect!r}")

    checks = {
        "expect": expect,
        "checked_identity_matched": bool(expect == "checked"),
        "linkage": {
            "capture_id": True,
            "artifact_sha256": True,
            "artifact_bytes": True,
            "source_set_digest_recomputed": True,
            "manifest_qualification": qualification,
            "holdout_exact": holdout_exact,
            "holdout_capped": holdout_capped,
            "n_dropped": art_facts["n_dropped"],
        },
    }
    return {
        "capture_dir": str(capture_dir),
        "artifact": {
            "basename": CAPTURE_ARTIFACT_BASENAME,
            "sha256": art_sha,
            "bytes": art_bytes,
            "capture_id": art_facts["capture_id"],
            "schema": artifact.get("schema"),
            "schema_version": artifact.get("schema_version"),
            "relations": artifact.get("relations"),
            "meta": dict(meta),
        },
        "manifest": {
            "basename": MANIFEST_BASENAME,
            "n_frames": n_frames,
            "unique_frame_ids": len(uniq),
            "canvas": {"width": computed["canvas_width"], "height": computed["canvas_height"],
                       "id": computed["canvas_id"]},
            "layout": {"nx": computed["nx"], "ny": computed["ny"]},
            "effective_cell_gauge_mode": effective_mode,
            "qualification": qualification,
        },
        "computed": computed,
        "checks": checks,
        "frame_ids": uniq,
    }


# ---------------------------------------------------------------------------
# 2. Source resolution (read-only)
# ---------------------------------------------------------------------------

def _scan_fits(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in (".fit", ".fits", ".fts") and p.is_file() and not p.is_symlink()
    )


def resolve_sources(source_dir: Path, frame_ids: Sequence[str]) -> dict:
    """Resolve each manifest id to exactly one file under ``source_dir``.

    Deterministic; relative-path match first, then unique basename. Rejects
    absolute ids, traversal, missing/ambiguous basenames and duplicate
    resolution. Never writes anything.
    """
    _require(source_dir.is_dir(), f"source dir is not a directory: {source_dir}", ResolutionError)
    root = source_dir
    root_res = root.resolve()
    files = _scan_fits(root)
    by_rel: dict[str, Path] = {}
    by_base: dict[str, list[Path]] = {}
    for f in files:
        by_rel.setdefault(f.relative_to(root).as_posix(), f)
        by_base.setdefault(f.name, []).append(f)

    resolved: list[Path] = []
    seen: set[Path] = set()
    problems: list[str] = []
    for idx, fid in enumerate(frame_ids):
        raw = str(fid)
        norm = raw.replace("\\", "/")
        pp = PurePosixPath(norm)
        if norm == "" or norm.startswith("/") or norm.startswith("//") or pp.is_absolute():
            problems.append(f"frame_ids[{idx}]: absolute/empty id {raw!r}")
            continue
        if any(part == ".." for part in pp.parts):
            problems.append(f"frame_ids[{idx}]: traversal in {raw!r}")
            continue
        chosen = by_rel.get(norm)
        if chosen is None:
            matches = by_base.get(pp.name, [])
            if len(matches) == 1:
                chosen = matches[0]
            elif len(matches) > 1:
                problems.append(f"frame_ids[{idx}]: ambiguous basename {pp.name!r}")
                continue
            else:
                problems.append(f"frame_ids[{idx}]: not found {raw!r}")
                continue
        try:
            chosen.resolve().relative_to(root_res)
        except ValueError:
            problems.append(f"frame_ids[{idx}]: escapes source dir")
            continue
        if chosen in seen:
            problems.append(f"frame_ids[{idx}]: duplicate resolution")
            continue
        seen.add(chosen)
        resolved.append(chosen)
    if problems:
        raise ResolutionError(
            f"could not resolve {len(problems)} of {len(frame_ids)} ids: " + "; ".join(problems[:10])
        )
    _require(len(resolved) == len(frame_ids), "resolution count mismatch", ResolutionError)
    return {
        "source_dir": str(root),
        "n_files_on_disk": len(files),
        "n_resolved": len(resolved),
        "paths": tuple(resolved),
    }


# ---------------------------------------------------------------------------
# 3. Geometry reconstruction (reuses production primitives)
# ---------------------------------------------------------------------------

def reconstruct_geometry(source_paths: Sequence[Path], source_dir: Path) -> dict:
    """Build descriptors + canvas + layout + per-Cell membership (production APIs)."""
    z = _lazy_zemosaic()
    zg, zal, zsw, zz = z["zg"], z["zal"], z["zsw"], z["zz"]
    from zemosaic.zemosaic_stack_plan import FrameInfo

    frames_info = [FrameInfo(path=Path(p), order=i) for i, p in enumerate(source_paths)]
    descs, rejected = zz._build_frame_descriptors(frames_info, str(source_dir), None, sip_mode="keep")
    _require(len(descs) == len(source_paths), f"descriptors built {len(descs)} != {len(source_paths)}")
    _require(not rejected, f"{len(rejected)} frames rejected at WCS qualification")
    canvas = zg.build_canvas(descs)
    footprints = zal.project_footprints(descs, canvas)
    contexts = zsw.build_cell_contexts_from_footprints(footprints, canvas, 6, 4, halo_px=8)

    cells: dict[str, dict] = {}
    cell_ctx: dict[str, tuple] = {}
    for (row, col, cell, patch, mem) in contexts:
        cells[cell.cell_id] = {
            "row": int(row),
            "col": int(col),
            "core": [int(patch.patch.x0 - (cell.core.x0 - patch.patch.x0)), 0, 0, 0],  # replaced below
            "patch": [int(patch.patch.x0), int(patch.patch.y0), int(patch.patch.x1), int(patch.patch.y1)],
            "patch_hw": [int(v) for v in patch.patch_shape_hw],
            "core_ids": list(mem.core_ids),
            "patch_ids": list(mem.patch_ids),
            "n_core": len(mem.core_ids),
            "n_patch": len(mem.patch_ids),
        }
        # canonical half-open core bounds (x0,y0,x1,y1)
        cells[cell.cell_id]["core"] = [
            int(cell.core.x0), int(cell.core.y0), int(cell.core.x1), int(cell.core.y1),
        ]
        cell_ctx[cell.cell_id] = (cell, patch, mem)

    total_memberships = sum(c["n_patch"] for c in cells.values())
    return {
        "engine": {
            "sip_mode": "keep",
            "reuse": [
                "zemosaic.zemosaic_zegrid_mode._build_frame_descriptors",
                "zemosaic.core.zegrid.geometry.build_canvas",
                "zemosaic.core.zegrid.auto_layout.project_footprints",
                "zemosaic.core.zegrid.sweep.build_cell_contexts_from_footprints",
            ],
        },
        "canvas": {
            "width": int(canvas.width),
            "height": int(canvas.height),
            "id": str(canvas.canvas_id),
            "resolution_deg": _fin(canvas.resolution_deg),
        },
        "layout": {"nx": 6, "ny": 4, "halo_px": 8, "cells": 24},
        "median_projected_footprint_wh": [_fin(footprints.median_w), _fin(footprints.median_h)],
        "total_patch_memberships": int(total_memberships),
        "cells": {k: cells[k] for k in sorted(cells)},
        "frames": [d.frame_id.logical_path for d in descs],
        "_cell_ctx": cell_ctx,
        "_descs": descs,
        "_canvas": canvas,
    }


# ---------------------------------------------------------------------------
# 4. Panel freeze (BIASED DEV; NOT QUALIFICATION)
# ---------------------------------------------------------------------------

def _relation_orientation(u: str, v: str) -> str:
    ur, uc = int(u[1:5]), int(u[6:10])
    vr, vc = int(v[1:5]), int(v[6:10])
    return "h" if ur == vr else "v"


def _canvas_corner_cells(nx: int, ny: int) -> set[str]:
    return {
        f"r{0:04d}c{0:04d}", f"r{0:04d}c{nx - 1:04d}",
        f"r{ny - 1:04d}c{0:04d}", f"r{ny - 1:04d}c{nx - 1:04d}",
    }


def relation_burdens(artifact: Mapping) -> dict:
    """Per-relation seam burden = max over channels |median pair difference|.

    A transparent, deliberately BIASED level-step proxy computed ONLY from the
    accepted capture artifact, used to freeze the DEV panel. It is NOT a
    qualification statistic.
    """
    per: dict[tuple, dict] = {}
    for rel in artifact.get("relations", []):
        u, v, ch = str(rel["u"]), str(rel["v"]), int(rel["channel"])
        key = (u, v)
        rec = per.setdefault(key, {"orientation": str(rel.get("orientation", _relation_orientation(u, v))),
                                   "channels": {}})
        rec["channels"][ch] = {
            "local_offset_train": _fin(rel.get("local_offset_train")),
            "robust_sigma": _fin(rel.get("robust_sigma")),
            "n_train": int(rel.get("n_train", 0)),
            "rejection_fraction": _fin(rel.get("rejection_fraction")),
            "level_mean": _fin(rel.get("level_mean")),
        }
    for key, rec in per.items():
        vals = [abs(c["local_offset_train"]) for c in rec["channels"].values()
                if c["local_offset_train"] is not None]
        rec["burden_adu"] = max(vals) if vals else None
    return per


def freeze_panel(artifact: Mapping, cells: Mapping[str, Mapping], nx: int, ny: int) -> dict:
    """Freeze the deterministic six-relation DEV panel (BEFORE factor results)."""
    per = relation_burdens(artifact)
    keys = sorted(per)
    for k in keys:
        per[k]["orientation"] = _relation_orientation(k[0], k[1])
    burden = {k: (per[k]["burden_adu"] if per[k]["burden_adu"] is not None else -1.0) for k in keys}
    ranked = sorted(keys, key=lambda k: (-burden[k], k))
    vertical = [k for k in ranked if per[k]["orientation"] == "v"]
    corner_cells = _canvas_corner_cells(nx, ny)
    corner_rels = [k for k in ranked if k[0] in corner_cells or k[1] in corner_cells]
    med_burden = float(np.median([burden[k] for k in keys]))

    slot_pick: dict[str, tuple | None] = {
        "declared_worst_edge": DECLARED_WORST_EDGE if DECLARED_WORST_EDGE in per else None,
        "highest_burden_vertical": vertical[0] if vertical else None,
        "highest_burden_overall": ranked[0] if ranked else None,
        "median_burden": None,
        "minimum_burden": min(keys, key=lambda k: (burden[k], k)) if keys else None,
        "corner_relation": None,
    }
    # median-burden relation: closest to the median burden (deterministic tie-break)
    if keys:
        slot_pick["median_burden"] = min(keys, key=lambda k: (abs(burden[k] - med_burden), k))
    # corner relation: cheapest (fewest contributors) corner relation NOT already
    # used, so a canvas corner gets an explicit panel slot (deterministic)
    if corner_rels:
        def _cost(k):
            return (cells[k[0]]["n_patch"] + cells[k[1]]["n_patch"], k)
        free_corner = [k for k in corner_rels if
                       k not in (slot_pick["declared_worst_edge"],
                                 slot_pick["highest_burden_vertical"],
                                 slot_pick["highest_burden_overall"],
                                 slot_pick["median_burden"],
                                 slot_pick["minimum_burden"])]
        slot_pick["corner_relation"] = min(free_corner or corner_rels, key=_cost)

    chosen: list[dict] = []
    used: set[tuple] = set()
    for slot in PANEL_SLOTS:
        k = slot_pick.get(slot)
        if k is None or k in used:
            continue
        used.add(k)
        chosen.append({
            "slot": slot,
            "u": k[0], "v": k[1],
            "orientation": per[k]["orientation"],
            "burden_adu": _fin(burden[k]),
            "n_patch_cells": [int(cells[k[0]]["n_patch"]), int(cells[k[1]]["n_patch"])],
            "membership_cost": int(cells[k[0]]["n_patch"] + cells[k[1]]["n_patch"]),
            "channels": {str(c): per[k]["channels"][c] for c in sorted(per[k]["channels"])},
        })
    # fill any remaining slots from the ranked list (determinism preserved)
    for k in ranked:
        if len(chosen) >= len(PANEL_SLOTS):
            break
        if k in used:
            continue
        used.add(k)
        chosen.append({
            "slot": "fill", "u": k[0], "v": k[1],
            "orientation": per[k]["orientation"], "burden_adu": _fin(burden[k]),
            "n_patch_cells": [int(cells[k[0]]["n_patch"]), int(cells[k[1]]["n_patch"])],
            "membership_cost": int(cells[k[0]]["n_patch"] + cells[k[1]]["n_patch"]),
            "channels": {str(c): per[k]["channels"][c] for c in sorted(per[k]["channels"])},
        })

    orientations = sorted({p["orientation"] for p in chosen})
    corners_present = sorted({
        c for p in chosen for c in (p["u"], p["v"]) if c in corner_cells
    })
    return {
        "label": "NOT_QUALIFICATION",
        "label_reason": (
            "biased DEV panel: the selection deliberately favours the declared worst edge "
            "and the highest-burden relations; it cannot qualify any model and no metric "
            "computed on it is an independent validation"
        ),
        "selection_rule": {
            "burden_definition": "max over channels |local_offset_train| (ADU) from the accepted capture",
            "slots": list(PANEL_SLOTS),
            "declared_worst_edge": list(DECLARED_WORST_EDGE),
            "median_burden_adu": _fin(med_burden),
            "rule": (
                "declared_worst_edge; highest-burden vertical; highest-burden overall; "
                "median-burden; minimum-burden; cheapest corner relation; dedup in slot order, "
                "then fill from the ranked list"
            ),
        },
        "inputs": {
            "n_relations_in_capture": len(keys),
            "canvas_corner_cells": sorted(corner_cells),
        },
        "relations": chosen,
        "orientations_covered": orientations,
        "corner_cells_covered": corners_present,
        "frozen_before_factor_results": True,
    }


# ---------------------------------------------------------------------------
# 5. Bounded aligned-cache build + full-domain gauge facts
# ---------------------------------------------------------------------------

def _subset_fixed_by_ids(fixed, ids_all: Sequence[str], keep: Sequence[str]):
    """Subset a FixedNormalization to ``keep`` (in the given order) by identity."""
    z = _lazy_zemosaic()
    from zemosaic.core.canonical_streaming import FixedNormalization

    gidx = {fid: i for i, fid in enumerate(ids_all)}
    idx = [gidx[f] for f in keep]
    ref = int(fixed.reference_index)
    keep_ref = idx.index(ref) if ref in idx else -1
    if keep_ref < 0:
        # No silent re-anchor: refuse (the caller must supply a gauge that covers
        # the reference, e.g. the shared gauge anchored on side u).
        raise ProbeError("gauge subset does not contain its reference frame")
    return FixedNormalization(
        reference_index=keep_ref,
        coefficients=np.asarray(fixed.coefficients)[idx],
        norm_active=np.asarray(fixed.norm_active)[idx],
        weights=np.asarray(fixed.weights)[idx],
        weight_active=np.asarray(fixed.weight_active)[idx],
        exclusions=tuple(
            e for e in fixed.exclusions if idx.count(int(e.index)) == 0
        ),
        frame_ids=tuple(keep),
    )


def build_cell_cache(cell_id: str, ctx: Mapping, source_paths_index: Mapping[str, Path],
                     scratch_dir: Path, *, max_frames: int) -> dict:
    """Stream one Cell's FULL accepted patch domain to a scratch aligned cache.

    Reuses the production primitives verbatim (product decoder, source-ROI plan,
    ``slice_wcs``, ``reproject_cropped``, ``AlignedCacheBuilder``) so the aligned
    values are bit-identical to the production per-Cell cache. One frame resident.

    RESUMABLE: a scratch cache that is already COMPLETE for the SAME ordered
    frame-id list (the accepted R6 ``cache_is_complete`` predicate) is reused
    instead of rebuilt, so a long bounded probe can resume after an interruption.
    """
    z = _lazy_zemosaic()
    zg, zfp, zz, zsw = z["zg"], z["zfp"], z["zz"], z["zsw"]
    cell, patch, mem = ctx
    ids = list(mem.patch_ids)
    _require(len(ids) <= max_frames,
             f"cell {cell_id}: {len(ids)} contributors exceed cap {max_frames}", BudgetError)
    cache_dir = scratch_dir / "cells" / cell_id
    if zfp.cache_is_complete(cache_dir, ids):
        return {
            "cell_id": cell_id,
            "cache_dir": str(cache_dir),
            "n_contributors": len(ids),
            "reused": True,
            "bytes": 0,
            "seconds": 0.0,
        }
    builder = zfp.AlignedCacheBuilder(cache_dir)
    t0 = time.perf_counter()
    bytes_written = 0
    for fid in ids:
        desc = _all_descs_by_id[fid]
        plan = zg.plan_source_roi(desc, _canvas_ref[0], patch)
        _require(plan is not None, f"cell {cell_id}: no source ROI for {fid}", ResolutionError)
        sb = plan.source_bounds
        hwc = zz._decode_frame_hwc(desc, None)
        crop = hwc[sb.y0:sb.y1, sb.x0:sb.x1]
        crop_chw = np.ascontiguousarray(np.moveaxis(crop, -1, 0))
        cropped_wcs = zz.zxe.slice_wcs(desc.wcs(), sb)
        rgb, geom = zz.zxe.reproject_cropped(
            crop_chw, cropped_wcs, patch.patch_wcs(), patch.patch_shape_hw
        )
        builder.add(fid, rgb, geom)
        del hwc, crop, crop_chw, rgb, geom
    manifest = builder.finish()
    bytes_written = int(manifest["total_bytes"])
    return {
        "cell_id": cell_id,
        "cache_dir": str(cache_dir),
        "n_contributors": len(ids),
        "reused": False,
        "bytes": bytes_written,
        "seconds": round(time.perf_counter() - t0, 3),
    }


_all_descs_by_id: dict[str, object] = {}
_canvas_ref: list = []


def probe_cell_gauge(cell_id: str, cache_dir: Path, cfg) -> dict:
    """Reproduce ONE Cell's local gauge facts on its FULL accepted domain."""
    z = _lazy_zemosaic()
    zfp, zphot, zstream = z["zfp"], z["zphot"], z["zstream"]
    provider = zfp.MemmapCanonicalProvider(cache_dir)
    try:
        request = zstream.build_streaming_request(cfg, provider.n_frames)
        cres = zphot.compute_cell_gauge(provider, request)
        ids = list(provider.frame_ids)
    finally:
        provider.close()
    if cres.status != zphot.CELL_GAUGE_STATUS_OK:
        return {"cell_id": cell_id, "status": cres.status, "detail": cres.detail,
                "n_contributors": len(ids)}
    g = cres.gauge
    coef = np.asarray(g.coefficients)
    weights = np.asarray(g.weights)
    active = np.asarray(g.weight_active, dtype=bool)
    norm_active = np.asarray(g.norm_active, dtype=bool)
    a_vals = sorted({round(float(v), 12) for v in coef[active, :, 0].ravel()})
    exclusions = [
        {"frame_id": ids[int(e.index)], "stage": str(e.stage), "reason": str(e.reason)}
        for e in g.exclusions
    ]
    return {
        "cell_id": cell_id,
        "status": "ok",
        "n_contributors": len(ids),
        "reference_index": int(g.reference_index),
        "reference_frame_id": ids[int(g.reference_index)],
        "active_frames": int(np.count_nonzero(active)),
        "norm_active_frames": int(np.count_nonzero(norm_active)),
        "excluded": exclusions,
        "n_excluded": len(exclusions),
        "weights_summary": _vec_summary(weights[active]),
        "coefficients_a_values": a_vals,
        "coefficients": coef.tolist(),
        "weights": weights.tolist(),
        "norm_active": norm_active.astype(bool).tolist(),
        "weight_active": active.astype(bool).tolist(),
        "frame_ids": ids,
    }


def _vec_summary(v) -> dict:
    a = np.asarray(v, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": _fin(a.min()), "p10": _fin(np.percentile(a, 10)),
        "median": _fin(np.median(a)), "p90": _fin(np.percentile(a, 90)),
        "max": _fin(a.max()),
    }


def compare_gauges(gauge_u: Mapping, gauge_v: Mapping, observed: Mapping) -> dict:
    """Cross-side comparison of two Cells' local gauges on common active frames.

    For ``sky_mean`` the applied transform is ``a=1``, ``b`` additive, so for the
    SAME frame at the SAME pixel the two sides differ by exactly ``b_u - b_v``.
    The median of that difference over common active frames is the gauge-induced
    per-Cell level step; its MAD measures how frame/mask dependent it is.
    """
    if gauge_u.get("status") != "ok" or gauge_v.get("status") != "ok":
        return {"comparable": False, "reason": "gauge_not_ok"}
    ids_u, ids_v = list(gauge_u["frame_ids"]), list(gauge_v["frame_ids"])
    common = [f for f in ids_u if f in set(ids_v) and
              gauge_u["weight_active"][ids_u.index(f)] and gauge_v["weight_active"][ids_v.index(f)]]
    au = sorted(gauge_u["coefficients_a_values"])
    av = sorted(gauge_v["coefficients_a_values"])
    comparable = (au == [1.0] and av == [1.0])
    out = {
        "comparable": bool(comparable),
        "reason": None if comparable else "non_additive_coefficients",
        "u": gauge_u["cell_id"], "v": gauge_v["cell_id"],
        "n_common_active_frames": len(common),
        "reference_u": gauge_u["reference_frame_id"],
        "reference_v": gauge_v["reference_frame_id"],
        "references_agree": gauge_u["reference_frame_id"] == gauge_v["reference_frame_id"],
        "frame_set_asymmetry": {
            "active_only_u": sorted(set(gauge_u["frame_ids"]) - set(gauge_v["frame_ids"])),
            "active_only_v": sorted(set(gauge_v["frame_ids"]) - set(gauge_u["frame_ids"])),
            "excluded_u": sorted(e["frame_id"] for e in gauge_u["excluded"]),
            "excluded_v": sorted(e["frame_id"] for e in gauge_v["excluded"]),
        },
        "per_channel": {},
    }
    if not comparable or not common:
        return out
    cu = np.asarray(gauge_u["coefficients"], dtype=np.float64)
    cv = np.asarray(gauge_v["coefficients"], dtype=np.float64)
    iu = {f: i for i, f in enumerate(ids_u)}
    iv = {f: i for i, f in enumerate(ids_v)}
    for ch in range(cu.shape[1]):
        d = []
        obs_vals = []
        for f in common:
            d.append(float(cu[iu[f], ch, 1] - cv[iv[f], ch, 1]))
            ob = observed.get(str(ch), {}).get("local_offset_train")
            if ob is not None:
                obs_vals.append(float(ob))
        d = np.asarray(d, dtype=np.float64)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        obs = float(np.median(obs_vals)) if obs_vals else None
        ratio = None
        if obs is not None and abs(obs) > 1e-12:
            ratio = med / obs
        out["per_channel"][str(ch)] = {
            "n": int(d.size),
            "gauge_level_step_median_adu": _fin(med),
            "gauge_level_step_mad_adu": _fin(mad),
            "gauge_level_step_p10_adu": _fin(np.percentile(d, 10)),
            "gauge_level_step_p90_adu": _fin(np.percentile(d, 90)),
            "consistency_mad_over_abs_median": _fin(mad / abs(med)) if abs(med) > 1e-12 else None,
            "observed_pair_diff_adu": _fin(obs),
            "gauge_step_over_observed": _fin(ratio),
        }
    return out


# ---------------------------------------------------------------------------
# 6. Bounded factor probe (A/B/C/D) on a shared-pixel ROI
# ---------------------------------------------------------------------------

class _RoiProvider:
    """Read-only view of one Cell's aligned cache restricted to a patch-local ROI.

    Serves the canonical streaming executor's ``get_raw_frame`` / ``get_tile``
    contract for a sub-window, so the executor can restack a BOUNDED ROI with a
    fixed gauge without ever materialising N x patch.
    """

    def __init__(self, cache_dir: Path, frame_ids: Sequence[str], roi: tuple[int, int, int, int]):
        z = _lazy_zemosaic()
        zfp = z["zfp"]
        cache_dir = Path(cache_dir)
        man = zfp.load_cache_manifest(cache_dir)
        all_ids = list(man["frame_ids"])
        idx = {fid: i for i, fid in enumerate(all_ids)}
        self._cache = cache_dir
        self._man = man
        self._sel = [idx[f] for f in frame_ids]
        self.frame_ids = tuple(frame_ids)
        y0, y1, x0, x1 = (int(v) for v in roi)
        self._roi = (y0, y1, x0, x1)
        self.n_frames = len(self._sel)
        self.height = int(y1 - y0)
        self.width = int(x1 - x0)
        self.channels = int(man["channels"])
        self.original_mono = bool(man["original_mono"])
        self.original_ndim = int(man["original_ndim"])
        self.original_shape = tuple(man["original_shape"])
        self._open: dict[int, tuple] = {}

    def _frame(self, k: int):
        if k not in self._open:
            ent = self._man["frames"][k]
            rgb = np.load(self._cache / ent["rgb"], mmap_mode="r")
            sup = np.load(self._cache / ent["support"], mmap_mode="r")
            self._open[k] = (rgb, sup)
        return self._open[k]

    def close(self):
        for rgb, sup in self._open.values():
            for arr in (rgb, sup):
                try:
                    arr._mmap.close()
                except Exception:
                    pass
        self._open = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def shape(self):
        return (self.height, self.width)

    def get_raw_frame(self, i: int):
        k = self._sel[i]
        rgb, sup = self._frame(k)
        y0, y1, x0, x1 = self._roi
        return np.asarray(rgb[y0:y1, x0:x1]), np.asarray(sup[y0:y1, x0:x1])

    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        k = self._sel[i]
        rgb, sup = self._frame(k)
        ry0, _ry1, rx0, _rx1 = self._roi
        tile = np.asarray(rgb[ry0 + y0:ry0 + y1, rx0 + x0:rx0 + x1])
        sup_tile = np.asarray(sup[ry0 + y0:ry0 + y1, rx0 + x0:rx0 + x1])
        from zemosaic.core.canonical_stacking import prepare_canonical_inputs
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


def _restack_roi(cache_dir: Path, frame_ids: Sequence[str], roi, fixed, cfg, rejection: str):
    z = _lazy_zemosaic()
    from zemosaic.core.canonical_streaming import run_canonical_stack_streaming
    zstream = z["zstream"]
    prov = _RoiProvider(cache_dir, frame_ids, roi)
    try:
        req = zstream.build_streaming_request(cfg, prov.n_frames)
        req = _replace_request(req, rejection=rejection)
        res = run_canonical_stack_streaming(prov, req, tile_size=None, fixed=fixed)
        # contributors BEFORE rejection: frames with any geometric support in the ROI
        pre = 0
        sups = []
        for i in range(prov.n_frames):
            _raw, sup = prov.get_raw_frame(i)
            sups.append(np.asarray(sup, dtype=bool))
            if bool(np.any(sup)):
                pre += 1
        valid = np.asarray(res.valid_mask, dtype=bool)
        if valid.ndim == 2:
            valid = valid[:, :, None]
        # contributors AFTER rejection: frames with support that retain at least one
        # non-rejected supported sample (the acceptance criterion: a frame that is
        # present AND not fully rejected for this pixel block)
        post = 0
        rm = getattr(res, "rejection_mask", None)
        if rm is not None:
            rm = np.asarray(rm, dtype=bool)
            if rm.ndim == 4:
                for i in range(prov.n_frames):
                    if not sups[i].any():
                        continue
                    if bool(np.any(sups[i][:, :, None] & ~rm[i])):
                        post += 1
        return {
            "science": np.asarray(res.science, dtype=np.float64),
            "valid": np.asarray(res.valid_mask, dtype=bool),
            "rejected_fraction": _fin(res.rejected_fraction),
            "rejected_sample_count": int(res.rejected_sample_count),
            "n_frames": int(prov.n_frames),
            "contributors_pre_rejection": int(pre),
            "contributors_post_rejection": int(post),
            "contributors_definition": (
                "pre = frames with any geometric support in the ROI; "
                "post = frames with support retaining >=1 non-rejected supported sample"
            ),
        }
    finally:
        prov.close()


def _replace_request(req, *, rejection: str):
    from zemosaic.core.canonical_engine import CanonicalStackRequest
    return CanonicalStackRequest(
        images=req.images, geometric_support=req.geometric_support,
        normalization=req.normalization, weighting=req.weighting,
        rejection=rejection, combine=req.combine, reference_index=req.reference_index,
        taper=req.taper, taper_px=req.taper_px, taper_floor=req.taper_floor,
        sigma_low=req.sigma_low, sigma_high=req.sigma_high, max_iters=req.max_iters,
        winsor_limit_low=req.winsor_limit_low, winsor_limit_high=req.winsor_limit_high,
        backend=req.backend, equalize_rgb=req.equalize_rgb,
    )


def _roi_metrics(du: Mapping, dv: Mapping) -> dict:
    """Robust, bounded metrics of a same-global-pixel pair difference."""
    su, sv = np.asarray(du["science"], dtype=np.float64), np.asarray(dv["science"], dtype=np.float64)
    if su.ndim == 2:
        su = su[:, :, None]
        sv = sv[:, :, None]
    vu, vv = np.asarray(du["valid"]), np.asarray(dv["valid"])
    # the canonical result's valid_mask is per-channel (H, W, C) for RGB: reduce it
    # to the shared 2-D support before intersecting (never silently ignore channels).
    if vu.ndim == 3:
        vu = vu.all(axis=-1)
    if vv.ndim == 3:
        vv = vv.all(axis=-1)
    vmask = vu & vv
    mask = vmask & np.isfinite(su).all(axis=-1) & np.isfinite(sv).all(axis=-1)
    h, w = mask.shape
    if not mask.any():
        return {"n": 0, "reason": "no_shared_valid_pixels"}
    out: dict = {"n": int(mask.sum()), "roi_hw": [int(h), int(w)]}
    ys, xs = np.nonzero(mask)
    gx = xs.astype(np.float64) + 0.5
    gy = ys.astype(np.float64) + 0.5
    per_ch = {}
    for ch in range(su.shape[2]):
        d = (su[:, :, ch] - sv[:, :, ch])[mask]
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        tangent = gx if w >= h else gy
        normal = gy if w >= h else gx
        per_ch[str(ch)] = {
            "median_adu": _fin(med),
            "rms_adu": _fin(math.sqrt(float(np.mean(d * d)))),
            "p90_abs_adu": _fin(np.percentile(np.abs(d), 90)),
            "mad_adu": _fin(mad),
            "tangent_slope_adu_per_px": _fin(_slope(tangent, d)),
            "normal_slope_adu_per_px": _fin(_slope(normal, d)),
            "level_corr": _fin(_corr(0.5 * (su[:, :, ch] + sv[:, :, ch])[mask], d)),
            "block_median_spread_adu": _fin(_block_spread(gx, gy, d)),
        }
    out["per_channel"] = per_ch
    out["rejected_fraction_u"] = du.get("rejected_fraction")
    out["rejected_fraction_v"] = dv.get("rejected_fraction")
    out["contributors_pre_rejection"] = [du.get("contributors_pre_rejection"),
                                         dv.get("contributors_pre_rejection")]
    out["contributors_post_rejection"] = [du.get("contributors_post_rejection"),
                                          dv.get("contributors_post_rejection")]
    out["contributors_definition"] = du.get("contributors_definition")
    return out


def _slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    if den <= 0:
        return None
    return float(np.sum(xc * (y - y.mean())) / den)


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    xc = x - x.mean()
    yc = y - y.mean()
    den = math.sqrt(float(np.sum(xc * xc)) * float(np.sum(yc * yc)))
    if den <= 0:
        return None
    return float(np.sum(xc * yc) / den)


def _block_spread(gx, gy, d, block=32):
    bx = (gx // block).astype(np.int64)
    by = (gy // block).astype(np.int64)
    key = bx * 100000 + by
    meds = []
    for k in np.unique(key):
        vals = d[key == k]
        if vals.size >= 8:
            meds.append(float(np.median(vals)))
    if len(meds) < 2:
        return None
    return float(np.percentile(meds, 90) - np.percentile(meds, 10))


def run_factor_probe(relation, gauges, caches, roi_cfg, cfg) -> dict:
    """One-variable-at-a-time factor evidence for ONE probed relation.

    Chain (each step changes exactly ONE variable from its predecessor):
      BASE  own frames + own local gauge + own weights + declared rejection
      A     FRAME SET only  -> common intersection, each side keeps its OWN gauge
      G     GAUGE only      -> both sides use side u's gauge over the common frames
      W     WEIGHTS only    -> coefficients kept, weights set uniform
      D     REJECTION only  -> none (DIAGNOSTIC ONLY)
      B0    control         -> identity gauge (a=1,b=0) + uniform weights
    """
    u, v = relation["u"], relation["v"]
    gu, gv = gauges.get(u), gauges.get(v)
    if not gu or not gv or gu.get("status") != "ok" or gv.get("status") != "ok":
        return {"status": "NOT_RUN", "reason": "gauge_facts_unavailable"}
    cu, cv = caches.get(u), caches.get(v)
    if not cu or not cv:
        return {"status": "NOT_RUN", "reason": "cell_cache_unavailable"}
    cap = int(roi_cfg["max_cell_contributors"])
    if gu["n_contributors"] > cap or gv["n_contributors"] > cap:
        return {"status": "NOT_RUN",
                "reason": f"contributor_cap_exceeded ({gu['n_contributors']},{gv['n_contributors']} > {cap})"}

    # Patch-local ROI per side relative to the shared GLOBAL band.
    cell_u, patch_u, _memu = _ctx_ref[u]
    cell_v, patch_v, _memv = _ctx_ref[v]
    y0 = max(patch_u.patch.y0, patch_v.patch.y0)
    y1 = min(patch_u.patch.y1, patch_v.patch.y1)
    x0 = max(patch_u.patch.x0, patch_v.patch.x0)
    x1 = min(patch_u.patch.x1, patch_v.patch.x1)
    if y1 <= y0 or x1 <= x0:
        return {"status": "NOT_RUN", "reason": "no_patch_overlap"}
    span = int(roi_cfg["max_tangent_px"])
    if relation["orientation"] == "h":
        boundary = min(cell_u.core.x1, cell_v.core.x1)  # shared vertical edge x
        bx0 = max(x0, boundary - int(roi_cfg["band_px"]))
        bx1 = min(x1, boundary + int(roi_cfg["band_px"]))
        cy = (y0 + y1) // 2
        by0 = max(y0, cy - span // 2)
        by1 = min(y1, by0 + span)
        roi = (by0, by1, bx0, bx1)
    else:
        boundary = min(cell_u.core.y1, cell_v.core.y1)
        by0 = max(y0, boundary - int(roi_cfg["band_px"]))
        by1 = min(y1, boundary + int(roi_cfg["band_px"]))
        cx = (x0 + x1) // 2
        bx0 = max(x0, cx - span // 2)
        bx1 = min(x1, bx0 + span)
        roi = (by0, by1, bx0, bx1)
    roi_u = (roi[0] - patch_u.patch.y0, roi[1] - patch_u.patch.y0,
             roi[2] - patch_u.patch.x0, roi[3] - patch_u.patch.x0)
    roi_v = (roi[0] - patch_v.patch.y0, roi[1] - patch_v.patch.y0,
             roi[2] - patch_v.patch.x0, roi[3] - patch_v.patch.x0)

    ids_u, ids_v = list(gu["frame_ids"]), list(gv["frame_ids"])
    common = [f for f in ids_u if f in set(ids_v) and gu["weight_active"][ids_u.index(f)]
              and gv["weight_active"][ids_v.index(f)]]
    if not common or gu["reference_frame_id"] not in common:
        return {"status": "NOT_RUN", "reason": "no_common_active_frames_covering_reference_u"}

    from zemosaic.core.canonical_streaming import FixedNormalization
    full_u = FixedNormalization(
        reference_index=ids_u.index(gu["reference_frame_id"]),
        coefficients=np.asarray(gu["coefficients"], dtype=np.float64),
        norm_active=np.asarray(gu["norm_active"], dtype=bool),
        weights=np.asarray(gu["weights"], dtype=np.float64),
        weight_active=np.asarray(gu["weight_active"], dtype=bool),
        exclusions=(), frame_ids=tuple(ids_u),
    )
    shared = _subset_fixed_by_ids(full_u, ids_u, common)

    def _sub_or_none(full, ids, name):
        try:
            return _subset_fixed_by_ids(full, ids, common)
        except ProbeError as exc:
            return None

    full_v = FixedNormalization(
        reference_index=ids_v.index(gv["reference_frame_id"]),
        coefficients=np.asarray(gv["coefficients"], dtype=np.float64),
        norm_active=np.asarray(gv["norm_active"], dtype=bool),
        weights=np.asarray(gv["weights"], dtype=np.float64),
        weight_active=np.asarray(gv["weight_active"], dtype=bool),
        exclusions=(), frame_ids=tuple(ids_v),
    )
    own_u = shared                       # side u's OWN gauge restricted to common
    own_v = _sub_or_none(full_v, ids_v, "own_v")   # side v's OWN gauge restricted to common
    uniform = FixedNormalization(
        reference_index=0,
        coefficients=np.asarray(shared.coefficients, dtype=np.float64).copy(),
        norm_active=np.ones(len(common), dtype=bool),
        weights=np.ones(len(common), dtype=np.float64),
        weight_active=np.ones(len(common), dtype=bool),
        exclusions=(), frame_ids=tuple(common),
    )
    identity = FixedNormalization(
        reference_index=0,
        coefficients=np.zeros((len(common), 3, 2), dtype=np.float64) + np.array([1.0, 0.0]),
        norm_active=np.ones(len(common), dtype=bool),
        weights=np.ones(len(common), dtype=np.float64),
        weight_active=np.ones(len(common), dtype=bool),
        exclusions=(), frame_ids=tuple(common),
    )

    results: dict = {}

    def _run(setting):
        if setting == "BASE":  # own frames + own gauge + own weights + declared rejection
            du = _restack_roi(cu["cache_dir"], ids_u, roi_u, full_u, cfg, cfg.rejection)
            dv = _restack_roi(cv["cache_dir"], ids_v, roi_v, full_v, cfg, cfg.rejection)
        elif setting == "A":  # variable: FRAME SET -> common intersection, own gauges kept
            if own_v is None:
                raise ProbeError("side-v reference not in the common frame set")
            du = _restack_roi(cu["cache_dir"], common, roi_u, own_u, cfg, cfg.rejection)
            dv = _restack_roi(cv["cache_dir"], common, roi_v, own_v, cfg, cfg.rejection)
        elif setting == "G":  # variable: GAUGE -> side v aligned to side u's gauge
            du = _restack_roi(cu["cache_dir"], common, roi_u, shared, cfg, cfg.rejection)
            dv = _restack_roi(cv["cache_dir"], common, roi_v, shared, cfg, cfg.rejection)
        elif setting == "W":  # variable: WEIGHTS -> uniform weights (coefficients unchanged)
            du = _restack_roi(cu["cache_dir"], common, roi_u, uniform, cfg, cfg.rejection)
            dv = _restack_roi(cv["cache_dir"], common, roi_v, uniform, cfg, cfg.rejection)
        elif setting == "D":  # variable: REJECTION -> none (DIAGNOSTIC ONLY)
            du = _restack_roi(cu["cache_dir"], common, roi_u, shared, cfg, "none")
            dv = _restack_roi(cv["cache_dir"], common, roi_v, shared, cfg, "none")
        elif setting == "B0":  # control: layout-independent identity gauge (not presumed right)
            du = _restack_roi(cu["cache_dir"], common, roi_u, identity, cfg, cfg.rejection)
            dv = _restack_roi(cv["cache_dir"], common, roi_v, identity, cfg, cfg.rejection)
        else:
            raise ProbeError(setting)
        return du, dv

    for setting in ("BASE", "A", "G", "W", "D", "B0"):
        try:
            du, dv = _run(setting)
            results[setting] = {"status": "RUN", "metrics": _roi_metrics(du, dv),
                                "frames_u": du["n_frames"], "frames_v": dv["n_frames"]}
        except Exception as exc:  # never hide a failing factor, never fake a zero
            results[setting] = {"status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "RUN",
        "contract": {
            "baseline": "BASE: own frames + own local gauge + own weights + declared rejection",
            "A": "BASE -> FRAME SET only (common intersection; each side keeps its OWN gauge, subset)",
            "G": "A -> GAUGE only (both sides use side u's gauge over the common frames)",
            "W": "G -> WEIGHTS only (coefficients kept, weights set uniform 1)",
            "D": "G -> REJECTION only (none; DIAGNOSTIC ONLY, never a correction)",
            "B0": "A -> GAUGE choice (identity a=1,b=0) + uniform weights; layout-independent CONTROL, not presumed right",
            "chain": "BASE -(frame set)-> A -(gauge)-> G -(weights)-> W ; G -(rejection)-> D ; B0 is an extra control",
            "single_variable": "every chain step changes exactly ONE variable from its predecessor",
            "backend": "cpu (fixed)",
            "non_additivity": "the frame set, gauge and weights interact: per-setting metrics are reported, never summed into percentages",
        },
        "limitations": [
            "the ROI is restacked with the accepted streaming executor on a bounded sub-window; "
            "the footprint taper is therefore computed within the ROI (edge effect) while the "
            "gauge coefficients/weights come from the FULL accepted domain",
            "interactions are reported per setting and never summed into an additive attribution",
        ],
        "roi_global": [int(v) for v in roi],
        "common_frames": len(common),
        "settings": results,
    }


_ctx_ref: dict[str, tuple] = {}


# ---------------------------------------------------------------------------
# 7. Conclusion (pre-registered decision rule)
# ---------------------------------------------------------------------------

def conclude(gauge_results: Sequence[Mapping], factor_results: Sequence[Mapping]) -> dict:
    """Apply the frozen conclusion rule; never overclaim."""
    steps = []
    n_comparable = 0
    for g in gauge_results:
        cmp = g.get("comparison") or g
        if not cmp.get("comparable"):
            continue
        n_comparable += 1
        for ch, vals in (cmp.get("per_channel") or {}).items():
            ratio = vals.get("gauge_step_over_observed")
            cons = vals.get("consistency_mad_over_abs_median")
            obs = vals.get("observed_pair_diff_adu")
            if ratio is None or obs is None:
                continue
            steps.append({
                "relation": f"{cmp['u']}|{cmp['v']}", "channel": int(ch),
                "ratio": abs(float(ratio)), "consistency": cons,
            })
    n_probed = n_comparable
    factors_executed = [f for f in factor_results if f.get("status") == "RUN"]

    partial = [s for s in steps if s["ratio"] >= GAUGE_STEP_PARTIAL_FRAC
               and (s["consistency"] is None or s["consistency"] <= GAUGE_STEP_CONSISTENCY_MAX)]
    negligible = [s for s in steps if s["ratio"] <= GAUGE_STEP_NEGLIGIBLE_FRAC]

    if n_probed < MIN_PROBED_RELATIONS:
        state, why = "INCONCLUSIVE", f"only {n_probed} comparable probed relation(s) < {MIN_PROBED_RELATIONS}"
    elif not steps:
        state, why = "INCONCLUSIVE", "no comparable gauge-step observation"
    elif len(negligible) == len(steps):
        state, why = "CAUSE_FALSIFIED", "gauge-induced level step is negligible on every probed relation/channel"
    elif partial and len(partial) >= max(1, len(steps) // 2):
        state = "PARTIAL_CAUSE"
        why = (f"{len(partial)}/{len(steps)} probed relation/channel gauge steps explain >= "
               f"{GAUGE_STEP_PARTIAL_FRAC:.0%} of the observed pair difference with stable sign; "
               "no restack-based dominance proof was executed on this host")
        if factors_executed:
            why += "; factor probe executed (see factor_evidence), dominance not established"
    else:
        state, why = "INCONCLUSIVE", "gauge-step evidence is mixed across probed relation/channel"

    return {
        "state": state,
        "reason": why,
        "thresholds": {
            "dominance_frac": GAUGE_STEP_DOMINANCE_FRAC,
            "partial_frac": GAUGE_STEP_PARTIAL_FRAC,
            "negligible_frac": GAUGE_STEP_NEGLIGIBLE_FRAC,
            "consistency_max": GAUGE_STEP_CONSISTENCY_MAX,
            "min_probed_relations": MIN_PROBED_RELATIONS,
        },
        "n_probed_relations": n_probed,
        "n_ratio_observations": len(steps),
        "ratio_observations": steps,
        "dominant_cause_confirmed": False,
        "claim_limits": [
            "no correction is applied anywhere",
            "a small-band or single-worst-edge result alone is never treated as causal",
            "factor interactions are reported, never summed into an additive attribution",
        ],
    }


# ---------------------------------------------------------------------------
# 8. Budgets + CLI
# ---------------------------------------------------------------------------

def _assert_write_allowed(path: Path, *, label: str, extra_forbidden: Sequence[Path] = ()) -> Path:
    p = Path(path).expanduser()
    rp = p.resolve()
    for root in tuple(FORBIDDEN_WRITE_ROOTS) + tuple(extra_forbidden):
        rr = Path(root).expanduser().resolve()
        if rp == rr or rr in rp.parents:
            raise BudgetError(f"{label} {p} is inside a read-only tree ({root}); refusing")
    return p


def estimate_costs(panel, cells, *, max_cell_contributors, max_frames_total) -> dict:
    """Bounded execution plan: orientation coverage first, then ascending cost."""
    order = sorted(panel["relations"], key=lambda r: (r["membership_cost"], r["u"], r["v"]))
    cells_needed: list[str] = []
    frames = 0
    selected: list[dict] = []
    selected_keys: set[tuple] = set()

    def _fits(rel):
        cu, cv = rel["u"], rel["v"]
        if cells[cu]["n_patch"] > max_cell_contributors or cells[cv]["n_patch"] > max_cell_contributors:
            return None
        add = [c for c in (cu, cv) if c not in cells_needed]
        add_frames = sum(cells[c]["n_patch"] for c in add)
        if frames + add_frames > max_frames_total:
            return None
        return add, add_frames

    def _take(rel):
        got = _fits(rel)
        if got is None:
            return False
        add, add_frames = got
        nonlocal frames
        cells_needed.extend(add)
        frames += add_frames
        selected.append({"u": rel["u"], "v": rel["v"], "orientation": rel["orientation"],
                         "membership_cost": rel["membership_cost"]})
        selected_keys.add((rel["u"], rel["v"]))
        return True

    # pass 1: guarantee one executed relation per orientation when feasible
    for want in ("h", "v"):
        if any(s["orientation"] == want for s in selected):
            continue
        for rel in order:
            if rel["orientation"] == want and _take(rel):
                break
    # pass 2: ascending cost within the remaining budget
    for rel in order:
        if (rel["u"], rel["v"]) in selected_keys:
            continue
        _take(rel)

    return {
        "execution_order_rule": (
            "pass 1 ensures one executed relation per orientation (cheapest first); "
            "pass 2 adds the remaining panel relations by ascending membership cost"
        ),
        "selected_relations": selected,
        "selected_cells": cells_needed,
        "estimated_frames": int(frames),
        "estimated_disk_mib": _fin(frames * 10.1),
        "estimated_seconds_at_2_9s_per_frame": _fin(frames * 2.9),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r38_causal_probe.py",
        description="ZM-ZEGRID-R38-A bounded read-only causal diagnostic harness",
    )
    p.add_argument("--capture-dir", required=True,
                   help="directory holding the exact R37 pre-feather capture + manifest (READ-ONLY)")
    p.add_argument("--source-dir", required=True, help="read-only source FITS directory")
    p.add_argument("--mode", choices=("inventory", "gauge", "factors"), default="inventory")
    p.add_argument("--out", default=None, help="strict-JSON report path")
    p.add_argument("--report-dir", default=str(REPORT_DIR_DEFAULT))
    p.add_argument("--scratch-dir", default=str(SCRATCH_DIR_DEFAULT))
    p.add_argument("--expect-identity", choices=("checked", "any"), default="checked")
    p.add_argument("--execute-real", action="store_true",
                   help="required for the real-data gauge/factor probe (never for inventory)")
    p.add_argument("--gauge-cache", default=None,
                   help="reuse an existing gauge-mode scratch dir instead of rebuilding")
    p.add_argument("--max-cell-contributors", type=int, default=220)
    p.add_argument("--max-frames-total", type=int, default=460)
    p.add_argument("--max-seconds", type=float, default=1200.0)
    p.add_argument("--factor-max-contributors", type=int, default=200)
    p.add_argument("--factor-band-px", type=int, default=8)
    p.add_argument("--factor-max-tangent-px", type=int, default=256)
    p.add_argument("--determinism-check", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    global expect_checked_hint
    args = build_arg_parser().parse_args(argv)
    _quiet()
    t_start = time.perf_counter()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    source_dir = Path(args.source_dir).expanduser().resolve()
    scratch = _assert_write_allowed(args.scratch_dir, label="scratch dir",
                                    extra_forbidden=(capture_dir, source_dir))
    report_dir = _assert_write_allowed(args.report_dir, label="report dir",
                                       extra_forbidden=(capture_dir, source_dir))
    out_path = Path(args.out).expanduser() if args.out else (report_dir / f"r38a_{args.mode}.json")
    _assert_write_allowed(out_path, label="report file", extra_forbidden=(capture_dir, source_dir))

    expect_checked_hint = args.expect_identity == "checked"
    identity = validate_identity(capture_dir, expect=args.expect_identity)
    resolution = resolve_sources(source_dir, identity["frame_ids"])
    geometry = reconstruct_geometry(resolution["paths"], source_dir)

    report: dict = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "mode": args.mode,
        "identity": {k: v for k, v in identity.items() if k != "artifact"},
        "resolution": {"n_resolved": resolution["n_resolved"],
                       "n_files_on_disk": resolution["n_files_on_disk"]},
        "geometry": {k: v for k, v in geometry.items() if not k.startswith("_")},
        "import_audit": import_audit(),
        "budgets": {
            "max_cell_contributors": int(args.max_cell_contributors),
            "max_frames_total": int(args.max_frames_total),
            "max_seconds": _fin(args.max_seconds),
            "factor_max_contributors": int(args.factor_max_contributors),
        },
        "not_run": [],
    }

    # global refs used by the (re)factor helpers
    _all_descs_by_id.clear()
    _canvas_ref.clear()
    _canvas_ref.append(geometry["_canvas"])
    for d in geometry["_descs"]:
        _all_descs_by_id[d.frame_id.logical_path] = d
    for cid, ctx in geometry["_cell_ctx"].items():
        _ctx_ref[cid] = ctx

    panel = freeze_panel(identity["artifact"], geometry["cells"], 6, 4)
    panel["inputs"]["artifact_sha256"] = identity["artifact"]["sha256"]
    report["panel"] = panel

    costs = estimate_costs(panel, geometry["cells"],
                           max_cell_contributors=args.max_cell_contributors,
                           max_frames_total=args.max_frames_total)
    report["probe_plan"] = costs
    report["not_run"].append({
        "item": "panel_relations_outside_execution_budget",
        "reason": "bounded host: only relations whose cells fit the per-cell/global caps are executed",
        "relations": [
            {"u": r["u"], "v": r["v"], "orientation": r["orientation"],
             "membership_cost": r["membership_cost"]}
            for r in panel["relations"]
            if not any(s["u"] == r["u"] and s["v"] == r["v"] for s in costs["selected_relations"])
        ],
    })

    comparisons = []
    executed = []
    factor_results: list = []
    if args.mode in ("gauge", "factors"):
        if not args.execute_real:
            report["not_run"].append({"item": "real_gauge_probe",
                                      "reason": "--execute-real not given"})
        else:
            comparisons, executed, factor_results = _run_real_probe(
                args, report, geometry, panel, costs, scratch, t_start
            )

    if args.mode in ("gauge", "factors") and args.execute_real:
        report["conclusion"] = conclude(comparisons, factor_results)

    if args.determinism_check:
        # Real repeat: rebuild the identity checks and the panel from the raw
        # artifact/capture independently and compare the canonical JSON.
        identity2 = validate_identity(capture_dir, expect=args.expect_identity)
        panel2 = freeze_panel(identity2["artifact"], geometry["cells"], 6, 4)
        panel2["inputs"]["artifact_sha256"] = identity2["artifact"]["sha256"]
        a = dumps_strict({"identity": {k: v for k, v in identity.items() if k != "artifact"},
                          "panel": panel})
        b = dumps_strict({"identity": {k: v for k, v in identity2.items() if k != "artifact"},
                          "panel": panel2})
        report["determinism"] = {"canonical_sha256": hashlib.sha256(a.encode()).hexdigest(),
                                 "stable": a == b}

    report["runtime"] = {
        "seconds": round(time.perf_counter() - t_start, 3),
        "peak_rss_kib": _peak_rss_kib(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "repo_head": _git_head(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_strict(report)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"R38A {args.mode}: wrote {out_path} ({len(text)} bytes)")
    print(json.dumps({"status": "ok", "mode": args.mode, "conclusion": report.get("conclusion", {}).get("state"),
                      "out": str(out_path)}, sort_keys=True))
    return 0


def _peak_rss_kib() -> int:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0


def _git_head() -> str | None:
    try:
        import subprocess
        repo = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _run_real_probe(args, report, geometry, panel, costs, scratch, t_start):
    """Execute the bounded real-data probe; returns (comparisons, executed, factors)."""
    z = _lazy_zemosaic()
    cfg = z["cfg_cls"](
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="mean",
        taper="footprint", taper_px=8.0, backend="cpu",
    )
    cells = geometry["cells"]
    selected_cells = list(costs["selected_cells"])
    report["probe_plan"]["executed_cells"] = selected_cells
    gauge_results = []
    caches: dict[str, dict] = {}
    gauge_reuse = Path(args.gauge_cache).expanduser() if args.gauge_cache else None
    for cid in selected_cells:
        if time.perf_counter() - t_start > args.max_seconds:
            report["not_run"].append({"item": f"cell_gauge:{cid}", "reason": "wall-clock budget exhausted"})
            continue
        try:
            if gauge_reuse is not None:
                cdir = gauge_reuse / "cells" / cid
                if not (cdir / "manifest.json").is_file():
                    raise ProbeError(f"reused gauge cache missing for {cid}")
                caches[cid] = {"cell_id": cid, "cache_dir": str(cdir), "reused": True}
                report.setdefault("reused_caches", []).append(cid)
            else:
                info = build_cell_cache(cid, geometry["_cell_ctx"][cid], None, scratch,
                                        max_frames=args.max_cell_contributors)
                caches[cid] = info
                cdir = Path(info["cache_dir"])
            facts = probe_cell_gauge(cid, cdir, cfg)
            facts["cache"] = caches.get(cid)
            gauge_results.append(facts)
        except BudgetError as exc:
            report["not_run"].append({"item": f"cell_gauge:{cid}", "reason": str(exc)})
        except Exception as exc:  # honest failure, never a fabricated result
            report["not_run"].append({"item": f"cell_gauge:{cid}",
                                      "reason": f"{type(exc).__name__}: {exc}"})
    by_id = {g["cell_id"]: g for g in gauge_results}

    report["gauge_facts"] = [
        {k: v for k, v in g.items() if k not in ("coefficients", "weights", "norm_active", "weight_active")}
        | {"coefficients_sha256": hashlib.sha256(
            np.asarray(g["coefficients"]).tobytes()).hexdigest()} if g.get("status") == "ok" else g
        for g in gauge_results
    ]

    comparisons = []
    executed = []
    for rel in costs["selected_relations"]:
        u, v = rel["u"], rel["v"]
        if u in by_id and v in by_id:
            obs = _observed_channels(panel, u, v)
            cmp = compare_gauges(by_id[u], by_id[v], obs)
            cmp["relation"] = {"u": u, "v": v, "orientation": rel["orientation"]}
            comparisons.append(cmp)
            executed.append(rel)
    report["gauge_comparison"] = comparisons
    report["executed_relations"] = executed

    factor_results: list = []
    if args.mode == "factors":
        roi_cfg = {"max_cell_contributors": args.factor_max_contributors,
                   "band_px": args.factor_band_px, "max_tangent_px": args.factor_max_tangent_px}
        for rel in executed:
            factor_results.append({
                "relation": rel, **run_factor_probe(rel, by_id, caches, roi_cfg, cfg),
            })
        report["factor_evidence"] = factor_results

    return comparisons, executed, factor_results


def _observed_channels(panel, u, v):
    for rel in panel.get("relations", []):
        if rel["u"] == u and rel["v"] == v:
            return rel.get("channels", {})
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
