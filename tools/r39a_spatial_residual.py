"""ZM-ZEGRID-R39-A — read-only post-offset spatial residual diagnostic.

Standalone, offline, READ-ONLY tool implementing the frozen contract
``docs/refactor/R39_POST_OFFSET_SPATIAL_RESIDUAL_PLAN.md`` (R39-A). It consumes
ONLY the compact R37E capture (``zegrid_r37_pre_feather.json``) plus its
``zegrid_manifest.json`` cross-link. It never reads image pixels / FITS, never
imports the product at module scope, and never writes under a dataset root.

Non-normative mechanics (implementation decisions, NOT threshold changes)
------------------------------------------------------------------------
The frozen plan leaves a few *mechanical* choices to the implementation; they are
recorded here so a reviewer can check them. None of them changes a frozen
threshold, a frozen model family, a frozen state, or a frozen conclusion rule.

* **Whitened condition number (§8; reworked normatively before any outcome).**
  ``cond(A) <= 1e4`` is evaluated on the Huber-weighted design whose non-constant
  columns are centred and scaled to unit population standard deviation (a genuine
  *whitening*, so the number measures collinearity and not the arbitrary unit
  ratio between the tangent coordinate ``t`` (O(0.5)) and the normal coordinate
  ``n`` (O(band/W)). The intercept column stays all-ones. ``cond`` is evaluated on
  ``A`` (never only on ``A^T A``). This definition is NEW/RE-DERIVED for this
  coordinate system, not the inherited R37 raw weighted cond (§8, §19).
* **Support gates (§8).** ``effective_train``/``tangent_blocks``/``normal_blocks`` are
  the artifact's production-computed relation scalars (relation pre-gates, §5); they are
  NOT per-fold counts. The per-fold admission requirement is the ``>= 3/4`` gate-passing
  folds condition.
* **Bootstrap (§5.1/§8; corrected R39-A rework-1).** LEAK-FREE: each replicate performs
  four separate fold fits — for every fold ``k`` the model is fit on a whole-tangent-block
  resample of folds ``!= k`` and scored on a whole-tangent-block resample of fold ``k``
  only; the four fold-local OOF vectors are concatenated. No fit ever contains one of its
  own scored test blocks. Coefficient draws use a documented deterministic fold aggregate
  (mean of finite fold coefficients). Seed 0, B=2000 frozen; memory bounded
  (no ``B x samples`` allocation); estimates uncertainty ONLY and never selects a model.
* **PREDICTED feather imprint (§7).** The imprint is computed ONLY when the exact
  accepted production ``feather_weight_field`` ramp can be reconstructed from the
  frozen layout alone (no pixels); multi-Cell corner samples are
  ``NOT_APPLICABLE``; otherwise the block is ``NOT_RUN``. It never drives any
  state, flag or panel outcome. The contextual final-FITS control is always
  ``NOT_RUN`` (no image read).

Normative revision history (recorded in the plan §19)
-----------------------------------------------------
The R39 plan's §3.4/§5.1/§8/§9.1 were revised **normatively, before any real R39
outcome**, to match this implementation: the tangent block runs ALONG the boundary
(``row // 4`` for ``h``, ``col // 4`` for ``v``, matching §3.2 and production
``tangent_blocks``); folds are assigned by the ORDINAL of the sorted unique tangent block
(the raw ``block_id % K`` rule is degenerate under the odd/odd hold-out predicate, where
every held-out block id is odd); the bootstrap is leak-free per fold; ``red_sigma >= 0.20``
is a mandatory admission condition for C1/C2/C3 (H1) and is re-checked fail-closed before
``SPATIAL_STABLE``/``CURVATURE``; ``>= 3/4`` gate-passing folds with complete OOF coverage;
C4 additionally requires C3 admitted/valid plus overall and beyond-C3 reductions; the full
detail is in the plan.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Schema + frozen expectations (R39 plan §1.1 / §15)
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.r39a.spatial_residual"
SCHEMA_VERSION = "r39a-1"
MISSION = "ZM-ZEGRID-R39-A-SPATIAL-DIAG"
CAPTURE_ARTIFACT_BASENAME = "zegrid_r37_pre_feather.json"
MANIFEST_BASENAME = "zegrid_manifest.json"
CAPTURE_STAGE = "pre_offset_pre_feather"
SOURCE_SET_DIGEST_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"
ARTIFACT_SCHEMA = "zemosaic.zegrid.background_diagnostics"
ARTIFACT_SCHEMA_VERSION = "r37c-1"

#: The exact accepted Windows-816 capture this ``checked`` run is recorded against.
CHECKED_IDENTITY = {
    "n_source_frames": 816,
    "canvas_width": 4117,
    "canvas_height": 4377,
    "canvas_id": "e8f1877f16f8cca4",
    "nx": 6,
    "ny": 4,
    "n_cells": 24,
    "channels": 3,
    "capture_id": "6c311af729425e69c030f87969fe6369dd9e9fc4ab9c133379662d70b65a8e00",
    "artifact_sha256": "a260b3309c39fbfc5264cacc60ba3048c678f83cf8bf6bf9f39c94a134a84032",
    "artifact_bytes": 23252804,
    "source_set_digest": "e3f4c16af30fca1a4c660ba5f76ea6f3d62868dd825bc09a9d83802bfccee3ae",
    "qualification": "EXACT",
    "effective_cell_gauge_mode": "per_cell",
    "halo_px": 8,
    "measurement_px": 16,
    "feather_px": 8,
    "block_px": 4,
    "n_relations_geometric": 38,
    "n_records": 114,
    "n_dropped": 0,
    "orientation_h": 20,
    "orientation_v": 18,
    "anchor": "r0001c0002",
    "normalization": "global_canvas_normalized_[-1,1]",
    "coordinate_convention": "pixel_centre_+0.5_global_x_from_col_y_from_row",
    "manifest_edges_used": 38,
    "manifest_edges_built": 38,
}

# Frozen widths / tolerances (R39 plan §1.1 / §3.4 / §6 / §8 / §9).
BLOCK_PX = 4
MEASUREMENT_PX = 16
FEATHER_PX = 8
HALO_PX = 8
INV_TOL = 1e-9
MAD_SCALE = 1.4826
HUBER_DELTA_FACTOR = 1.345
RMS0_SIGMA_GATE = 1.25            # INH: R37 absolute RMS-around-zero gate
COND_MAX = 1.0e4                  # INH: R37 contract §6.1
RED_SIGMA_MIN = 0.20              # NEW: pooled OOF relative robust-sigma reduction
MIN_EFFECTIVE_TRAIN = 256         # INH: R37 contract §6.1
MIN_TANGENT_BLOCKS = 8            # INH: R37 contract §6.1
MIN_NORMAL_BLOCKS = 3             # INH: R37 contract §6.1
LEVERAGE_SIGMA_FRAC = 0.5         # INH: band-edge predicted uncertainty <= 0.5 sigma_ref
COEF_SIGMA_MULT = 3.0             # NEW: |coef| >= 3 x bootstrap SE
N_FOLDS = 4                       # frozen K
MIN_ADMITTED_FOLDS = 3            # M2: >= 3/4 gate-passing folds required for admission
BOOTSTRAP_SEED = 0                # frozen seed
BOOTSTRAP_B_DEFAULT = 2000        # frozen replicate count (no CLI override)
CURVATURE_RED_MIN = 0.20          # NEW: C4 relative reduction beyond C3
MODEL_NAMES = ("C0", "C1", "C2", "C3", "C4")
DIST_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))
STATE_PRECEDENCE = (
    "INCONCLUSIVE",
    "UNEXPLAINED_STRUCTURE",
    "CURVATURE",
    "SPATIAL_STABLE",
    "CONSTANT_ONLY",
    "NEGLIGIBLE",
)
FLAG_SUPPORT_NOT_AVAILABLE = "SUPPORT_NOT_AVAILABLE"
FLAG_SPATIAL_STRUCTURE_PARTIAL = "SPATIAL_STRUCTURE_PARTIAL"
FLAG_UNEXPLAINED_CHANNEL = "UNEXPLAINED_CHANNEL"
QUALIFICATION_DEV = "DEV_NOT_QUALIFICATION"

REPORT_DIR_DEFAULT = Path(
    os.environ.get("R39A_REPORT_DIR")
    or (Path.home() / ".openclaw" / "workspace" / ".a2a-reports" / "ZM-ZEGRID-R39-A-SPATIAL-DIAG")
)

# Resumable-execution contract (checkpoint binding identity).
TOOL_CONTRACT_VERSION = "r39a-1"
CKPT_SCHEMA = "zemosaic.zegrid.r39a.checkpoint"
CKPT_VERSION = "r39a-ckpt-1"
SCRATCH_DIR_DEFAULT = Path(
    os.environ.get("R39A_SCRATCH_DIR") or (Path(tempfile.gettempdir()) / "openclaw-r39a")
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class R39Error(RuntimeError):
    """Deterministic, user-facing harness failure (never a traceback dump)."""


class IdentityError(R39Error):
    """Exact identity / linkage / structural validation refused."""


class ResolutionError(R39Error):
    """Path / write-policy / argument resolution refused."""


class BudgetError(R39Error):
    """Memory / write-budget safety refused."""


class FixtureError(R39Error):
    """A mandatory sign/order, coordinate or analytic fixture failed (hard refuse)."""


# ---------------------------------------------------------------------------
# Import audit (module scope must never pull ``zemosaic``)
# ---------------------------------------------------------------------------

def module_scope_imports() -> list[str]:
    """Fully-qualified module names imported at MODULE scope (self AST parse)."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def import_audit() -> dict:
    mods = module_scope_imports()
    bad = [m for m in mods if m == "zemosaic" or m.startswith("zemosaic.")]
    return {
        "module_scope_imports": sorted(mods),
        "zemosaic_module_scope_imports": sorted(bad),
        "engine_access_is_lazy": not bad,
    }


# ---------------------------------------------------------------------------
# Strict JSON helpers
# ---------------------------------------------------------------------------

def _reject_constant(name: str):
    raise R39Error(f"strict JSON refused a non-finite literal: {name}")


def load_json_strict(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh, parse_constant=_reject_constant)


def _fin(x: Any) -> float | None:
    """Float or ``None`` for a non-finite / non-numeric value (reporting only)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def dumps_strict(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False, indent=2, ensure_ascii=True)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def source_set_digest(identifiers) -> str:
    """Byte-compatible with the accepted ``background_diagnostics`` algorithm."""
    uniq = sorted({str(x) for x in identifiers if str(x)})
    h = hashlib.sha256()
    h.update((SOURCE_SET_DIGEST_ALGO + "\n").encode("utf-8"))
    for item in uniq:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Path privacy: logical labels + recursive fail-closed scan/scrub/redact
# ---------------------------------------------------------------------------

_WIN_DRIVE_RE = re.compile(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"(?:^|[^\\])\\\\[A-Za-z0-9._$-]+[\\]")
_FILE_URI_RE = re.compile(r"file://", re.IGNORECASE)
_POSIX_ABS_EMBEDDED_RE = re.compile(
    r"(?<![\w:/])(?:/(?:[A-Za-z0-9._~+\-]+/)+[A-Za-z0-9._~+\-]*|/[A-Za-z0-9._~+\-]+)"
)
REDACTED_PATH = "<redacted:absolute-path>"


def path_leak_category(value: str) -> str | None:
    """Return a leak category when ``value`` carries an absolute/private path."""
    if not isinstance(value, str) or not value:
        return None
    if _FILE_URI_RE.search(value):
        return "file_uri"
    if _WIN_DRIVE_RE.search(value):
        return "windows_drive_absolute"
    if _UNC_RE.search(value) or value.startswith("\\\\"):
        return "unc"
    if value.startswith("/"):
        return "posix_absolute"
    if _POSIX_ABS_EMBEDDED_RE.search(value):
        return "posix_absolute_embedded"
    return None


def scan_path_leaks(payload: Any, prefix: str = "$") -> list[dict]:
    found: list[dict] = []
    if isinstance(payload, Mapping):
        for key in sorted(payload):
            found.extend(scan_path_leaks(payload[key], f"{prefix}.{key}"))
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            found.extend(scan_path_leaks(item, f"{prefix}[{i}]"))
    elif isinstance(payload, str):
        cat = path_leak_category(payload)
        if cat is not None:
            found.append({"json_path": prefix, "category": cat})
    return found


def scrub_paths(payload: Any, roots: Sequence[Path]) -> Any:
    pairs = []
    for root in roots:
        if root is None:
            continue
        label = f"<root:{Path(root).name or 'root'}>"
        for raw in {str(root), str(Path(root).expanduser())}:
            if raw:
                pairs.append((raw, label))
        try:
            pairs.append((str(Path(root).expanduser().resolve()), label))
        except Exception:
            pass
    pairs.sort(key=lambda kv: -len(kv[0]))

    def _walk(obj):
        if isinstance(obj, Mapping):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v) for v in obj)
        if isinstance(obj, str):
            out = obj
            for raw, label in pairs:
                if raw and raw in out:
                    out = out.replace(raw, label)
            return out
        return obj

    return _walk(payload)


def redact_path_leaks(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {k: redact_path_leaks(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact_path_leaks(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(redact_path_leaks(v) for v in payload)
    if isinstance(payload, str) and path_leak_category(payload) is not None:
        return REDACTED_PATH
    return payload


def minimal_failure_record(*, stage: str, leak_locations: Sequence[Mapping]) -> dict:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "status": "FAILED",
        "status_reason": "durable report failed the privacy boundary",
        "qualification": QUALIFICATION_DEV,
        "scientific_acceptance": False,
        "correction_applied": False,
        "r39b_authorized": False,
        "r39b_track": "NONE",
        "failed": {
            "stage": str(stage),
            "leak_count": int(len(leak_locations)),
            "categories": sorted({str(l.get("category")) for l in leak_locations}),
            "json_paths": sorted({str(l.get("json_path")) for l in leak_locations}),
        },
    }


def enforce_privacy_boundary(payload: Any, roots: Sequence[Path]) -> tuple[Any, list[dict]]:
    scrubbed = scrub_paths(payload, roots)
    redacted = redact_path_leaks(scrubbed)
    return redacted, scan_path_leaks(redacted)


def one_way_path_identity(tag: str, path) -> str:
    try:
        norm = str(Path(path).expanduser().resolve())
    except Exception:
        norm = str(path)
    h = hashlib.sha256()
    h.update((str(tag) + "\n").encode("utf-8"))
    h.update(norm.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Write policy (capture/dataset roots are NEVER output parents)
# ---------------------------------------------------------------------------

_WIN_DRIVE_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def is_drive_absolute(raw: str) -> bool:
    return bool(_WIN_DRIVE_ABS_RE.match(str(raw).strip()))


def has_traversal_component(raw: str) -> bool:
    parts = re.split(r"[\\/]+", str(raw))
    return any(p == ".." for p in parts)


def raw_path_is_ambiguous(raw: str) -> str | None:
    """Fail-closed classification of a RAW output path string."""
    if not isinstance(raw, str) or not raw.strip():
        return "empty"
    s = raw.strip()
    if s.lower().startswith("file://"):
        return "file_uri"
    if is_drive_absolute(s):
        return "windows_drive_absolute"
    if s.startswith("\\\\") or _UNC_RE.search(s):
        return "unc"
    if has_traversal_component(s):
        return "traversal"
    return None


def symlink_component(path: Path, root: Path) -> bool:
    """True when any component of ``path`` relative to ``root`` is a symlink."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    cur = root
    for part in rel.parts:
        cur = cur / part
        try:
            if cur.is_symlink():
                return True
        except OSError:
            return True
    return False


def derive_dataset_root(capture_dir: Path, extra_roots: Sequence[Path] = ()) -> Path:
    """Nearest common ancestor of the capture dir and any explicit extra roots.

    Refuses a too-broad derivation (filesystem/drive root, home or an ancestor of
    home, or a single-component path) instead of silently weakening the
    no-writes-under-the-dataset invariant.
    """
    parts = [str(Path(capture_dir).resolve())] + [str(Path(r).resolve()) for r in extra_roots]
    try:
        common = Path(os.path.commonpath(parts))
    except ValueError as exc:
        raise ResolutionError(f"cannot derive a dataset root: {type(exc).__name__}") from exc
    try:
        common = common.resolve()
    except OSError as exc:
        raise ResolutionError(f"cannot resolve the derived dataset root: {type(exc).__name__}") from exc
    if common == Path(common.anchor):
        raise ResolutionError("derived dataset root is a filesystem/drive root; refusing (too broad)")
    if len(common.parts) < 2:
        raise ResolutionError("derived dataset root is too shallow; refusing (too broad)")
    home = Path.home().resolve()
    if common == home or common in home.parents:
        raise ResolutionError("derived dataset root is home or an ancestor of home; refusing (too broad)")
    return common


def build_write_policy(*, capture_dir: Path, extra_roots: Sequence[Path] = ()) -> dict:
    capture_dir = Path(capture_dir).resolve()
    extras = [Path(p).expanduser().resolve() for p in extra_roots]
    dataset_root = derive_dataset_root(capture_dir, extras)
    protected = (capture_dir,) + tuple(extras) + (dataset_root,)
    return {
        "mode": "capture_plus_derived_dataset_root",
        "protected_roots": protected,
        "dataset_root_identity_sha256": one_way_path_identity(
            "zemosaic.zegrid.r39a.dataset_root.v1", dataset_root
        ),
        "extra_roots_count": len(extras),
    }


def _assert_write_allowed(path: Path, *, label: str, policy: Mapping | None = None) -> Path:
    p = Path(path).expanduser()
    rp = p.resolve()
    roots = []
    if policy is not None:
        roots.extend(policy.get("protected_roots") or ())
    for root in roots:
        rr = Path(root).expanduser().resolve()
        if rp == rr or rr in rp.parents:
            raise BudgetError(f"{label} is inside a protected read-only tree; refusing")
    return p


# ---------------------------------------------------------------------------
# Lazy production access (never at module scope)
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _lazy_cell_feather():
    """Lazy import of the accepted additive-offset / feather production surface."""
    _ensure_src_on_path()
    from zemosaic.core.zegrid import cell_feather as zcf

    return zcf


def _quiet() -> None:
    logging.disable(logging.INFO)


# ---------------------------------------------------------------------------
# Frozen geometry / coordinates (R39 plan §3)
# ---------------------------------------------------------------------------

def forward_coord(index, dim):
    arr = np.asarray(index, dtype=np.float64)
    out = 2.0 * (arr + 0.5) / float(dim) - 1.0
    if out.ndim == 0:
        return float(out)
    return out


def inverse_index(coord: float, dim: int) -> int:
    """Frozen normalized->integer inverse (§3.4); raises on any ambiguity."""
    p_float = ((float(coord) + 1.0) * float(dim) / 2.0) - 0.5
    if not math.isfinite(p_float):
        raise IdentityError("inverse_index: non-finite recovered coordinate")
    nearest = math.floor(p_float + 0.5)
    if abs(p_float - nearest) > INV_TOL:
        raise IdentityError(
            f"inverse_index: recovered index {p_float!r} is not within INV_TOL of an integer"
        )
    p = int(nearest)
    if not (0 <= p < int(dim)):
        raise IdentityError(f"inverse_index: recovered index {p} out of range [0,{dim})")
    return p


def _cell_rc(cell_id: str) -> tuple[int, int]:
    return int(cell_id[1:5]), int(cell_id[6:10])


def relation_boundary(u: str, v: str, orientation: str, W: int, H: int, nx: int, ny: int) -> float:
    """Frozen normal-boundary in GLOBAL PIXEL index units (§3.2).

    ``b = (col(u)+1) * W/nx`` for ``h`` and ``(row(u)+1) * H/ny`` for ``v`` (the
    float cell width, exactly the accepted ``background_model`` boundary).
    """
    ur, uc = _cell_rc(u)
    vr, vc = _cell_rc(v)
    if orientation == "h":
        return float((uc + 1) * (float(W) / float(nx)))
    if orientation == "v":
        return float((ur + 1) * (float(H) / float(ny)))
    raise IdentityError(f"unknown orientation {orientation!r}")


def relation_normal_tangent(col, row, orientation, W, H, nx, ny, u, v):
    """Return (n, t, s) with ``n`` the frozen normal, ``t`` the tangent and ``s``
    the physical normal distance in pixels (§3.2)."""
    b = relation_boundary(u, v, orientation, W, H, nx, ny)
    if orientation == "h":
        X = 2.0 * (col + 0.5) / float(W) - 1.0
        Y = 2.0 * (row + 0.5) / float(H) - 1.0
        X_b = 2.0 * b / float(W) - 1.0
        n = X - X_b
        s = n * (float(W) / 2.0)
        t = Y
    else:
        X = 2.0 * (col + 0.5) / float(W) - 1.0
        Y = 2.0 * (row + 0.5) / float(H) - 1.0
        Y_b = 2.0 * b / float(H) - 1.0
        n = Y - Y_b
        s = n * (float(H) / 2.0)
        t = X
    return n, t, s


def tangent_block_index(col, row, orientation):
    """Whole-block index along the BOUNDARY (the tangent direction of §3.2).

    NOTE (documented ambiguity, see report RISKS): the frozen plan §3.4/§5.1
    parenthetical says ``col // 4`` for ``h`` and ``row // 4`` for ``v``, which is
    the transpose of §3.2's tangent coordinate (``t = Y`` for ``h``) and of the
    accepted production ``tangent_blocks`` (``gyp`` for ``h``). The §3.4
    parenthetical also makes the K=4 fold split degenerate for BOTH orientations
    (only ~2 odd normal-direction blocks lie inside the 16 px band), contradicting
    §8's ``K=4`` folds and the ``tangent_blocks >= 8`` gate. This implementation
    follows §3.2 / production (tangent runs along the boundary): rows for ``h``,
    columns for ``v``.
    """
    return (row // BLOCK_PX) if orientation == "h" else (col // BLOCK_PX)


def normal_block_index(col, row, orientation):
    return (col // BLOCK_PX) if orientation == "h" else (row // BLOCK_PX)


def post_offset_residual(holdout_diff, off_u, off_v):
    """Frozen ``d_post = holdout_diff + off_u - off_v`` (per channel, §3.3)."""
    d = np.asarray(holdout_diff, dtype=np.float64)
    return d + float(off_u) - float(off_v)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(MAD_SCALE * np.median(np.abs(x - med)))


def _rms0(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(x * x))))


def _p90_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), 90.0))


def robust_fit(A: np.ndarray, y: np.ndarray, *, huber_factor: float = HUBER_DELTA_FACTOR,
               max_iter: int = 50, tol: float = 1e-12):
    """Deterministic Huber IRLS fit; returns ``(coef, weights, sigma)``."""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0 or A.shape[0] != n:
        raise R39Error("robust_fit: empty/degenerate input")
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        resid = y - A @ coef
        sigma = _mad_sigma(resid)
        if not math.isfinite(sigma) or sigma <= 0.0:
            break
        cut = huber_factor * sigma
        absr = np.abs(resid)
        w = np.where(absr <= cut, 1.0, cut / np.maximum(absr, 1e-300))
        sw = np.sqrt(w)
        new, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        if float(np.max(np.abs(new - coef))) <= tol * max(1.0, float(np.max(np.abs(coef)))):
            coef = new
            break
        coef = new
    resid = y - A @ coef
    sigma = _mad_sigma(resid)
    return coef, w, float(sigma)


def model_design(model: str, n: np.ndarray, t: np.ndarray) -> np.ndarray:
    ones = np.ones_like(n, dtype=np.float64)
    if model == "C0":
        return np.column_stack([ones])
    if model == "C1":
        return np.column_stack([ones, t])
    if model == "C2":
        return np.column_stack([ones, n])
    if model == "C3":
        return np.column_stack([ones, t, n])
    if model == "C4":
        return np.column_stack([ones, t, n, n * n])
    raise R39Error(f"unknown model {model!r}")


def model_coef_names(model: str) -> tuple[str, ...]:
    return {
        "C0": ("c0",),
        "C1": ("c0", "b_t"),
        "C2": ("c0", "b_n"),
        "C3": ("c0", "b_t", "b_n"),
        "C4": ("c0", "b_t", "b_n", "q_n"),
    }[model]


def _standardise(A: np.ndarray) -> np.ndarray:
    B = np.asarray(A, dtype=np.float64).copy()
    for j in range(1, B.shape[1]):
        col = B[:, j]
        s = float(np.std(col))
        mu = float(np.mean(col))
        if not math.isfinite(s) or s <= 0.0:
            raise IdentityError("design column has zero/non-finite spread")
        B[:, j] = (col - mu) / s
    return B


def whitened_cond(A: np.ndarray, w: np.ndarray) -> float:
    """Whitened condition number of the weighted design (see module docstring)."""
    B = _standardise(A)
    sw = np.sqrt(np.asarray(w, dtype=np.float64))
    return float(np.linalg.cond(B * sw[:, None]))


def design_rank_full(A: np.ndarray) -> bool:
    try:
        B = _standardise(A)
    except IdentityError:
        return False
    return int(np.linalg.matrix_rank(B)) == int(A.shape[1])


# ---------------------------------------------------------------------------
# Identity validation
# ---------------------------------------------------------------------------

def _require(cond: bool, message: str, exc=R39Error) -> None:
    if not cond:
        raise exc(message)


def validate_identity(capture_dir: Path, *, expect: str = "checked") -> dict:
    """Strictly validate the exact capture + manifest identity/linkage (§1.1/§15)."""
    cap_json = capture_dir / CAPTURE_ARTIFACT_BASENAME
    man_json = capture_dir / MANIFEST_BASENAME
    _require(cap_json.is_file(), f"capture artifact missing: {cap_json.name}", IdentityError)
    _require(man_json.is_file(), f"manifest missing: {man_json.name}", IdentityError)

    artifact = load_json_strict(cap_json)
    manifest = load_json_strict(man_json)
    _require(isinstance(artifact, dict), "capture artifact is not a JSON object", IdentityError)
    _require(isinstance(manifest, dict), "manifest is not a JSON object", IdentityError)

    art_sha = sha256_file(cap_json)
    art_bytes = int(cap_json.stat().st_size)

    # --- artifact provenance + shape facts --------------------------------
    _require(artifact.get("schema") == ARTIFACT_SCHEMA, "artifact schema mismatch", IdentityError)
    _require(artifact.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
             "artifact schema_version mismatch", IdentityError)
    prov = artifact.get("provenance")
    _require(isinstance(prov, Mapping), "artifact provenance missing", IdentityError)
    _require(bool(prov.get("production_wired")) is True, "artifact not production-wired", IdentityError)
    _require(bool(prov.get("correction_applied")) is False, "artifact applied a correction", IdentityError)
    _require(prov.get("capture_stage") == CAPTURE_STAGE, "artifact capture_stage mismatch", IdentityError)
    cap = prov.get("capture")
    _require(isinstance(cap, Mapping), "artifact provenance.capture missing", IdentityError)

    meta = artifact.get("meta")
    _require(isinstance(meta, Mapping), "artifact meta missing", IdentityError)

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
        "channels": int(meta.get("channels", -1)),
        "canvas_hw": [int(v) for v in meta.get("canvas_hw", [])],
        "block_px": int(meta.get("block_px", -1)),
        "anchor": str(meta.get("anchor", "")),
        "measurement_px_meta": int(meta.get("measurement_px", -1)) if meta.get("measurement_px") is not None else -1,
        "feather_px_meta": int(meta.get("feather_px", -1)),
        "normalization": str(meta.get("normalization", "")),
        "coordinate_convention": str(meta.get("coordinate_convention", "")),
    }
    _require(art_facts["halo_px"] == HALO_PX, "artifact halo_px mismatch", IdentityError)
    _require(art_facts["measurement_px"] == MEASUREMENT_PX, "artifact measurement_px mismatch", IdentityError)
    _require(art_facts["feather_px"] == FEATHER_PX, "artifact feather_px mismatch", IdentityError)
    _require(art_facts["block_px"] == BLOCK_PX, "artifact block_px mismatch", IdentityError)
    _require(art_facts["measurement_px_meta"] in (MEASUREMENT_PX, -1),
             "artifact meta.measurement_px mismatch", IdentityError)
    _require(art_facts["feather_px_meta"] == FEATHER_PX, "artifact meta.feather_px mismatch", IdentityError)
    _require(art_facts["normalization"] == "global_canvas_normalized_[-1,1]",
             "artifact normalization mismatch", IdentityError)
    _require(art_facts["coordinate_convention"] == "pixel_centre_+0.5_global_x_from_col_y_from_row",
             "artifact coordinate_convention mismatch", IdentityError)
    _require(art_facts["effective_cell_gauge_mode"] == "per_cell",
             "artifact effective_cell_gauge_mode mismatch", IdentityError)

    relations = artifact.get("relations")
    _require(isinstance(relations, list) and relations, "artifact relations missing", IdentityError)
    dropped = artifact.get("dropped")
    _require(isinstance(dropped, list), "artifact dropped missing", IdentityError)
    _require(len(dropped) == 0, "artifact dropped relations present", IdentityError)
    _require(art_facts["n_dropped"] == 0, "artifact n_dropped != 0", IdentityError)

    # geometric vs record decomposition
    geom = {}
    channels_seen = {}
    for rel in relations:
        _require(isinstance(rel, Mapping), "artifact relation is not an object", IdentityError)
        for key in ("u", "v", "channel", "orientation", "n_holdout", "n_holdout_kept",
                    "holdout_capped", "robust_sigma", "holdout_xy", "holdout_diff"):
            _require(key in rel, f"artifact relation missing key {key!r}", IdentityError)
        key = (str(rel["u"]), str(rel["v"]), str(rel["orientation"]))
        geom[key] = geom.get(key, 0) + 1
        channels_seen.setdefault(key, set()).add(int(rel["channel"]))
        _require(bool(rel["holdout_capped"]) is False,
                 f"relation {key} holdout is capped (not EXACT)", IdentityError)
        _require(int(rel["n_holdout_kept"]) == int(rel["n_holdout"]),
                 f"relation {key} holdout_kept != n_holdout", IdentityError)
        if rel.get("measurement_px") is not None:
            _require(int(rel["measurement_px"]) == MEASUREMENT_PX,
                     f"relation {key} measurement_px mismatch", IdentityError)

    n_geom = len(geom)
    orient_counts = {"h": 0, "v": 0}
    for (_u, _v, o) in geom:
        _require(o in ("h", "v"), f"unknown orientation {o!r}", IdentityError)
        orient_counts[o] += 1
    channels = art_facts["channels"]
    for key, chs in channels_seen.items():
        _require(chs == set(range(channels)),
                 f"relation {key} does not carry exactly channels {sorted(range(channels))}",
                 IdentityError)
    _require(art_facts["n_relations"] == len(relations),
             "artifact meta.n_relations != number of records", IdentityError)

    # --- manifest facts ----------------------------------------------------
    _require(manifest.get("engine") == "zegrid", "manifest engine != zegrid", IdentityError)
    fids = manifest.get("frame_ids")
    _require(isinstance(fids, list) and fids, "manifest frame_ids missing", IdentityError)
    _require(all(isinstance(x, str) and x for x in fids), "manifest frame_ids malformed", IdentityError)
    uniq = sorted(set(fids))
    _require(len(uniq) == len(fids), "manifest frame_ids contain duplicates", IdentityError)
    n_frames = int(manifest.get("n_frames", -1))
    _require(n_frames == len(fids), "manifest n_frames != len(frame_ids)", IdentityError)

    canvas = manifest.get("canvas")
    layout = manifest.get("layout")
    gauge = manifest.get("photometric_gauge")
    feather = manifest.get("cell_feather")
    _require(isinstance(canvas, Mapping), "manifest canvas missing", IdentityError)
    _require(isinstance(layout, Mapping), "manifest layout missing", IdentityError)
    _require(isinstance(gauge, Mapping), "manifest photometric_gauge missing", IdentityError)
    _require(isinstance(feather, Mapping), "manifest cell_feather missing", IdentityError)
    _require(bool(feather.get("enabled")) is True, "manifest cell_feather disabled", IdentityError)

    offsets_raw = feather.get("offsets")
    _require(isinstance(offsets_raw, Mapping) and offsets_raw, "manifest cell_feather.offsets missing",
             IdentityError)
    offsets = {}
    for cid, vec in offsets_raw.items():
        _require(isinstance(vec, (list, tuple)) and len(vec) == channels,
                 f"offsets[{cid}] must be a {channels}-vector", IdentityError)
        vals = [float(v) for v in vec]
        _require(all(math.isfinite(v) for v in vals),
                 f"offsets[{cid}] has a non-finite entry", IdentityError)
        offsets[str(cid)] = vals

    capblk = feather.get("r37_diagnostic_capture")
    _require(isinstance(capblk, Mapping), "manifest r37_diagnostic_capture block missing", IdentityError)

    computed_digest = source_set_digest(uniq)
    _require(computed_digest == art_facts["source_set_digest"],
             "recomputed source-set digest != artifact digest", IdentityError)
    _require(art_facts["source_set_digest_algo"] == SOURCE_SET_DIGEST_ALGO,
             "artifact source_set_digest_algo mismatch", IdentityError)
    _require(str(capblk.get("source_set_digest", "")) == computed_digest,
             "manifest capture block source-set digest mismatch", IdentityError)
    _require(str(capblk.get("source_set_digest_algo", "")) == SOURCE_SET_DIGEST_ALGO,
             "manifest capture block digest algo mismatch", IdentityError)
    _require(int(capblk.get("n_source_frames", -1)) == len(uniq),
             "manifest capture block source count mismatch", IdentityError)
    _require(art_facts["n_source_frames"] == len(uniq),
             "artifact n_source_frames != manifest frame count", IdentityError)

    # --- manifest <-> artifact linkage (disk-derived, non-tautological) ----
    _require(str(capblk.get("capture_id", "")) == art_facts["capture_id"],
             "manifest capture_id != artifact capture_id", IdentityError)
    _require(str(capblk.get("filename", "")) == CAPTURE_ARTIFACT_BASENAME,
             "manifest capture filename mismatch", IdentityError)
    _require(str(capblk.get("sha256", "")) == art_sha,
             "manifest capture sha256 != on-disk artifact sha256", IdentityError)
    _require(int(capblk.get("bytes", -1)) == art_bytes,
             "manifest capture bytes != on-disk artifact bytes", IdentityError)

    qualification = str(capblk.get("qualification", ""))
    _require(qualification in ("EXACT", "RESEARCH_ONLY_HELDOUT_CAPPED"),
             f"manifest qualification unexpected: {qualification!r}", IdentityError)
    holdout_exact = bool(capblk.get("holdout_exact"))
    holdout_capped = bool(capblk.get("holdout_capped"))
    _require(holdout_capped is False, "capture holdout is capped (not EXACT)", IdentityError)
    _require(holdout_exact is True, "capture holdout_exact is not true", IdentityError)
    _require(qualification == "EXACT", "capture qualification != EXACT", IdentityError)
    _require(str(capblk.get("capture_stage", "")) == CAPTURE_STAGE,
             "manifest capture_stage mismatch", IdentityError)
    _require(str(capblk.get("effective_cell_gauge_mode", "")) == "per_cell",
             "manifest capture gauge mode mismatch", IdentityError)
    _require(int(capblk.get("halo_px", -1)) == HALO_PX, "manifest capture halo_px mismatch", IdentityError)
    _require(int(capblk.get("measurement_px", -1)) == MEASUREMENT_PX,
             "manifest capture measurement_px mismatch", IdentityError)
    _require(int(capblk.get("feather_px", -1)) == FEATHER_PX,
             "manifest capture feather_px mismatch", IdentityError)
    _require(int(capblk.get("n_cells", -1)) == int(layout.get("nx", -1)) * int(layout.get("ny", -1)),
             "manifest capture n_cells != layout", IdentityError)
    _require(int(capblk.get("n_relations", -1)) == len(relations),
             "manifest capture n_relations != artifact records", IdentityError)
    _require(int(capblk.get("n_dropped", -1)) == 0, "manifest capture n_dropped != 0", IdentityError)
    _require(feather.get("anchor_cell") is not None, "manifest anchor_cell missing", IdentityError)
    _require(int(feather.get("channels", -1)) == channels, "manifest cell_feather.channels mismatch",
             IdentityError)
    _require(int(feather.get("halo_px", -1)) == HALO_PX, "manifest cell_feather.halo_px mismatch",
             IdentityError)
    _require(int(feather.get("edges_used", -1)) == n_geom, "manifest edges_used != geometric relations",
             IdentityError)
    _require(int(feather.get("edges_built", -1)) == n_geom, "manifest edges_built != geometric relations",
             IdentityError)
    _require(str(gauge.get("mode", "")) == "per_cell", "manifest gauge mode != per_cell", IdentityError)
    _require(art_facts["canvas_hw"] == [int(canvas.get("height", -1)), int(canvas.get("width", -1))],
             "artifact canvas_hw != manifest canvas", IdentityError)
    _require(art_facts["nx"] == int(layout.get("nx", -1)) and art_facts["ny"] == int(layout.get("ny", -1)),
             "artifact nx/ny != manifest layout", IdentityError)

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
        "n_cells": int(layout.get("nx", -1)) * int(layout.get("ny", -1)),
        "channels": channels,
        "capture_id": art_facts["capture_id"],
        "qualification": qualification,
        "effective_cell_gauge_mode": "per_cell",
        "halo_px": HALO_PX,
        "measurement_px": MEASUREMENT_PX,
        "feather_px": FEATHER_PX,
        "block_px": BLOCK_PX,
        "anchor": art_facts["anchor"],
        "n_relations_geometric": n_geom,
        "n_records": len(relations),
        "n_dropped": 0,
        "orientation_h": orient_counts["h"],
        "orientation_v": orient_counts["v"],
        "normalization": art_facts["normalization"],
        "coordinate_convention": art_facts["coordinate_convention"],
        "manifest_edges_used": int(feather.get("edges_used", -1)),
        "manifest_edges_built": int(feather.get("edges_built", -1)),
    }

    if expect == "checked":
        for key, want in CHECKED_IDENTITY.items():
            got = computed.get(key)
            _require(got == want,
                     f"checked identity mismatch for {key!r}: got {got!r} want {want!r}",
                     IdentityError)
    else:
        _require(expect == "any", f"unknown expect {expect!r}", ResolutionError)

    return {
        "artifact": artifact,
        "manifest": manifest,
        "relations": relations,
        "offsets": offsets,
        "meta": {
            "nx": computed["nx"],
            "ny": computed["ny"],
            "canvas_w": computed["canvas_width"],
            "canvas_h": computed["canvas_height"],
            "channels": computed["channels"],
            "block_px": BLOCK_PX,
            "measurement_px": MEASUREMENT_PX,
            "halo_px": HALO_PX,
        },
        "computed": computed,
        "frame_ids": uniq,
    }


# ---------------------------------------------------------------------------
# Mandatory fixtures (hard refuse before metrics)
# ---------------------------------------------------------------------------

def coordinate_fixtures(W: int, H: int) -> dict:
    """Forward/inverse round-trip on every integer boundary index (§3.4 fixture 1)."""
    checks = 0
    for dim in (W, H):
        idxs = set()
        for base in (0, 1, BLOCK_PX, 2 * BLOCK_PX, BLOCK_PX + 1, dim // 2, dim - 2, dim - 1):
            for d in (-1, 0, 1):
                v = base + d
                if 0 <= v < dim:
                    idxs.add(int(v))
        for i in sorted(idxs):
            X = forward_coord(i, dim)
            back = inverse_index(X, dim)
            if back != i:
                raise FixtureError(f"coordinate round-trip failed at index {i} (dim={dim})")
            checks += 1
    return {"checks": checks, "inv_tol": INV_TOL, "passed": True}


def _sign_order_production_fixture() -> dict:
    """Exercise the accepted production application path (§3.3 mandatory 1/2)."""
    zcf = _lazy_cell_feather()
    rng = np.random.default_rng(1234)
    su = rng.normal(size=(6, 6, 3)) * 10.0 + 100.0
    sv = rng.normal(size=(6, 6, 3)) * 10.0 + 100.0
    off_u = np.array([1.5, -2.25, 0.75])
    off_v = np.array([-0.5, 3.125, -1.0])
    au = zcf.apply_additive_offsets(su, off_u)
    av = zcf.apply_additive_offsets(sv, off_v)
    lhs = au - av
    holdout_diff = su - sv
    rhs = holdout_diff + off_u[None, None, :] - off_v[None, None, :]
    if not np.allclose(lhs, rhs, rtol=0.0, atol=1e-12):
        raise FixtureError("production apply_additive_offsets identity failed")
    # swapped endpoints flip the sign exactly
    lhs_sw = av - au
    if not np.allclose(lhs_sw, -lhs, rtol=0.0, atol=1e-12):
        raise FixtureError("swapped-endpoint sign flip failed")
    return {"production_identity": True, "swapped_endpoint_flip": True}


def sign_order_fixtures() -> dict:
    """Mandatory sign/order + analytic fixtures (§3.3 / requirement 4)."""
    out = _sign_order_production_fixture()
    rng = np.random.default_rng(99)
    holdout_diff = rng.normal(size=500) * 4.0
    off_u, off_v = 2.5, -1.25
    d_post = post_offset_residual(holdout_diff, off_u, off_v)
    analytic = holdout_diff + off_u - off_v
    if float(np.max(np.abs(d_post - analytic))) > 1e-10:
        raise FixtureError("analytic d_post fixture exceeded 1e-10")
    swapped = post_offset_residual(-holdout_diff, off_v, off_u)
    if float(np.max(np.abs(swapped + d_post))) > 1e-10:
        raise FixtureError("analytic swapped d_post fixture exceeded 1e-10")
    out["analytic_d_post_max_abs_err"] = float(np.max(np.abs(d_post - analytic)))
    out["analytic_swap_max_abs_err"] = float(np.max(np.abs(swapped + d_post)))
    return out


def _recover_relation_samples(rel: Mapping, meta: Mapping, offsets: Mapping) -> dict:
    """Recover integers/blocks/parity/n,t,d_post for one record (fail-closed)."""
    W = meta["canvas_w"]
    H = meta["canvas_h"]
    nx, ny = meta["nx"], meta["ny"]
    u = str(rel["u"])
    v = str(rel["v"])
    orientation = str(rel["orientation"])
    channel = int(rel["channel"])
    xy = np.asarray(rel["holdout_xy"], dtype=np.float64).reshape(-1, 2)
    diff = np.asarray(rel["holdout_diff"], dtype=np.float64).ravel()
    if xy.shape[0] != diff.size:
        raise IdentityError(f"relation {u}/{v}/{channel}: xy/diff length mismatch")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(diff)):
        raise IdentityError(f"relation {u}/{v}/{channel}: non-finite held-out sample")

    cols = np.empty(xy.shape[0], dtype=np.int64)
    rows = np.empty(xy.shape[0], dtype=np.int64)
    for i in range(xy.shape[0]):
        cols[i] = inverse_index(xy[i, 0], W)
        rows[i] = inverse_index(xy[i, 1], H)
        # X/Y reproduction within INV_TOL
        if abs(forward_coord(cols[i], W) - xy[i, 0]) > INV_TOL or \
           abs(forward_coord(rows[i], H) - xy[i, 1]) > INV_TOL:
            raise IdentityError("held-out sample does not reproduce its stored X/Y within INV_TOL")
        # captured block parity predicate: odd/odd on every sample
        if (cols[i] // BLOCK_PX) % 2 != 1 or (rows[i] // BLOCK_PX) % 2 != 1:
            raise IdentityError("held-out sample does not satisfy the odd/odd block-parity predicate")
    n, t, s = relation_normal_tangent(cols, rows, orientation, W, H, nx, ny, u, v)
    if float(np.max(np.abs(s))) > (MEASUREMENT_PX / 2.0) + 1e-6:
        raise IdentityError("held-out sample lies outside the measurement band")
    off_u = offsets[u][channel]
    off_v = offsets[v][channel]
    d_post = post_offset_residual(diff, off_u, off_v)
    return {
        "u": u, "v": v, "orientation": orientation, "channel": channel,
        "cols": cols, "rows": rows, "n": n, "t": t, "s": s,
        "tangent_block": tangent_block_index(cols, rows, orientation),
        "normal_block": normal_block_index(cols, rows, orientation),
        "e_pre": diff, "e_post": d_post, "sigma_ref": float(rel["robust_sigma"]),
        "n_train": int(rel.get("n_train", -1)),
        "effective_train": float(rel.get("effective_train", float("nan"))),
        "artifact_tangent_blocks": int(rel.get("tangent_blocks", -1)),
        "artifact_normal_blocks": int(rel.get("normal_blocks", -1)),
    }


def structural_validation(samples_by_rel: Sequence[Mapping]) -> dict:
    """Real-artifact structural validation summary (§3.4 mandatory fixture 2)."""
    n_samples = 0
    parity_ok = 0
    for s in samples_by_rel:
        n_samples += int(s["cols"].size)
        parity_ok += int(s["cols"].size)
    return {"samples_validated": n_samples, "parity_odd_odd": parity_ok, "passed": True}


# ---------------------------------------------------------------------------
# Model family, folds, OOF, bootstrap (§8)
# ---------------------------------------------------------------------------

def _fold_ids(tangent_block: np.ndarray) -> np.ndarray:
    """K deterministic folds over whole tangent BLOCKS.

    NOTE (documented ambiguity, see report RISKS): the frozen §3.4/§8 rule
    ``fold = (tangent_block_index) % K`` is degenerate under the frozen odd/odd
    hold-out predicate — every held-out tangent block id is ODD, so the raw-block
    rule yields only the folds ``{1, 3}`` and folds 0/2 are always empty (no model
    could ever be admitted). Folds are therefore assigned by the ORDINAL of the
    sorted unique tangent block (0,1,2,... in sorted block order) modulo K, which
    is deterministic, whole-block, and spreads contiguous blocks across folds.
    """
    tb = np.asarray(tangent_block, dtype=np.int64)
    uniq = np.unique(tb)
    ordinal = {int(b): i for i, b in enumerate(uniq)}
    return np.array([ordinal[int(b)] % N_FOLDS for b in tb], dtype=np.int64)


def _fold_gate(A: np.ndarray, w: np.ndarray) -> tuple[bool, str | None]:
    """Design-level identifiability gates (§8) on a fold's TRAINING design.

    Support-count gates (``effective_train``/``tangent_blocks``/``normal_blocks``)
    are checked once per relation from the artifact's production-computed edge
    scalars (see :func:`classify_channel`), because the compact artifact persists
    only the odd/odd held-out subsample from which the production support counts
    cannot be recovered.
    """
    if not design_rank_full(A):
        return False, "rank_deficient"
    try:
        cond = whitened_cond(A, w)
    except IdentityError as exc:
        return False, f"cond_refused:{exc}"
    if not math.isfinite(cond) or cond > COND_MAX:
        return False, "ill_conditioned"
    return True, None


def _fold_blocks(tangent_block: np.ndarray, fold: np.ndarray) -> dict:
    """Deterministic per-fold whole-block partition.

    ``{k: (train_blocks, test_blocks)}`` with ``train_blocks`` the sorted unique
    tangent blocks whose samples have ``fold != k`` and ``test_blocks`` the sorted
    unique tangent blocks with ``fold == k``. Whole tangent blocks are never split.
    """
    tb = np.asarray(tangent_block, dtype=np.int64)
    fl = np.asarray(fold, dtype=np.int64)
    return {
        k: (sorted(int(b) for b in np.unique(tb[fl != k])),
            sorted(int(b) for b in np.unique(tb[fl == k])))
        for k in range(N_FOLDS)
    }


def bootstrap_replicate_draws(tangent_block: np.ndarray, fold: np.ndarray, rng) -> list[dict]:
    """One LEAK-FREE whole-block bootstrap draw (M1).

    For EVERY fold ``k`` the training blocks are drawn (with replacement) ONLY from
    folds ``!= k`` and the scored test blocks ONLY from fold ``k`` itself, so no fit
    ever contains one of its own scored test blocks. Exposed for the structural
    leakage test.
    """
    tb = np.asarray(tangent_block, dtype=np.int64)
    fl = np.asarray(fold, dtype=np.int64)
    block_samples = {int(b): np.nonzero(tb == b)[0] for b in np.unique(tb)}
    parts = _fold_blocks(tb, fl)
    out = []
    for k in range(N_FOLDS):
        tr_blocks, te_blocks = parts[k]
        tr_pick = ([tr_blocks[p] for p in rng.integers(0, len(tr_blocks), size=len(tr_blocks))]
                   if tr_blocks else [])
        te_pick = ([te_blocks[p] for p in rng.integers(0, len(te_blocks), size=len(te_blocks))]
                   if te_blocks else [])
        tr_idx = np.concatenate([block_samples[b] for b in tr_pick]) if tr_pick \
            else np.array([], dtype=np.int64)
        te_idx = np.concatenate([block_samples[b] for b in te_pick]) if te_pick \
            else np.array([], dtype=np.int64)
        out.append({"k": k, "train_blocks": tr_pick, "test_blocks": te_pick,
                    "train_idx": tr_idx, "test_idx": te_idx})
    return out


def _fold_fits(model: str, n: np.ndarray, t: np.ndarray, y: np.ndarray,
               fold: np.ndarray, tangent_block: np.ndarray, normal_block: np.ndarray) -> dict:
    """Per-fold deterministic fits + honest OOF predictions (§8, M2).

    Returns a per-fold status list (gate ok / exact reason / finite coefficient /
    finite predictions), the OOF prediction vector (NaN where a fold produced no
    finite prediction), the coverage fraction, the coefficients of the GATE-PASSING
    finite folds only, and the number of gate-passing folds. A model is admitted only
    with ``>= 3/4`` gate-passing folds AND complete OOF coverage.
    """
    n = np.asarray(n, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fold_arr = np.asarray(fold, dtype=np.int64)
    A_all = model_design(model, n, t)
    oof = np.full(y.size, np.nan, dtype=np.float64)
    folds = []
    coefs_gate_ok: list[np.ndarray] = []
    n_gate_ok = 0
    for k in range(N_FOLDS):
        tr = fold_arr != k
        te = fold_arr == k
        entry = {"k": k, "n_train": int(np.count_nonzero(tr)), "n_test": int(np.count_nonzero(te)),
                 "gate_ok": False, "gate_reason": None, "coef_finite": False, "pred_finite": False}
        if not np.any(tr) or not np.any(te):
            entry["gate_reason"] = "empty_fold"
            folds.append(entry)
            continue
        coef, w, _ = robust_fit(A_all[tr], y[tr])
        entry["coef_finite"] = bool(np.all(np.isfinite(coef)))
        gate_ok, reason = _fold_gate(A_all[tr], w)
        entry["gate_ok"] = bool(gate_ok)
        entry["gate_reason"] = reason
        pred = A_all[te] @ coef
        entry["pred_finite"] = bool(np.all(np.isfinite(pred)))
        if entry["pred_finite"]:
            oof[te] = y[te] - pred
        if gate_ok and entry["coef_finite"]:
            n_gate_ok += 1
            coefs_gate_ok.append(np.asarray(coef, dtype=np.float64))
        folds.append(entry)
    coverage = float(np.count_nonzero(np.isfinite(oof)) / max(1, y.size))
    return {"folds": folds, "oof": oof, "coverage": coverage,
            "oof_finite": bool(coverage == 1.0), "n_gate_ok": int(n_gate_ok),
            "coefs_gate_ok": coefs_gate_ok}


def bootstrap_relation(n: np.ndarray, t: np.ndarray, y: np.ndarray,
                       tangent_block: np.ndarray, fold: np.ndarray,
                       n_boot: int = BOOTSTRAP_B_DEFAULT,
                       seed: int = BOOTSTRAP_SEED) -> dict:
    """LEAK-FREE whole-tangent-block bootstrap (§5.1, M1/M2).

    Each replicate performs FOUR separate fold fits: for every fold ``k`` the model
    is fit on a whole-block resample of the folds ``!= k`` and scored on a
    whole-block resample of fold ``k`` only, then the four fold-local OOF vectors are
    concatenated. No fit ever contains one of its own scored test blocks.

    Coefficient draws are aggregated across folds by a documented deterministic FOLD
    AGGREGATE (the MEAN of the finite fold coefficients) before the SE is taken. Memory
    bounded: per-replicate index arrays only, no ``n_boot x samples`` allocation. The
    bootstrap estimates uncertainty ONLY and never selects a model.
    """
    n = np.asarray(n, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    blocks = np.asarray(tangent_block, dtype=np.int64)
    fold = np.asarray(fold, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    uniq_blocks = np.unique(blocks)
    block_samples = {int(b): np.nonzero(blocks == b)[0] for b in uniq_blocks}

    coef_draws = {m: [] for m in MODEL_NAMES}
    c0_draws: list[float] = []
    sigma_oof_draws = {m: [] for m in MODEL_NAMES}
    red_sigma_draws = {m: [] for m in MODEL_NAMES}

    for _ in range(int(n_boot)):
        draws = bootstrap_replicate_draws(blocks, fold, rng)
        # c0 = robust median over a whole-block resample of ALL samples (a summary
        # statistic, NOT a model fit; documented).
        pick_all = rng.integers(0, len(uniq_blocks), size=len(uniq_blocks))
        res_idx = np.concatenate([block_samples[int(uniq_blocks[p])] for p in pick_all])
        c0_draws.append(float(np.median(y[res_idx])))

        oof_parts = {m: [] for m in MODEL_NAMES}
        fold_coefs = {m: [] for m in MODEL_NAMES}
        for d in draws:
            tri, tei = d["train_idx"], d["test_idx"]
            if tri.size == 0 or tei.size == 0:
                continue
            for m in MODEL_NAMES:
                A_tr = model_design(m, n[tri], t[tri])
                coef, *_ = robust_fit(A_tr, y[tri])
                if not np.all(np.isfinite(coef)):
                    continue
                A_te = model_design(m, n[tei], t[tei])
                resid = y[tei] - A_te @ coef
                if np.all(np.isfinite(resid)):
                    oof_parts[m].append(resid)
                    fold_coefs[m].append(np.asarray(coef, dtype=np.float64))
        for m in MODEL_NAMES:
            if fold_coefs[m]:
                coef_draws[m].append(np.mean(np.vstack(fold_coefs[m]), axis=0))
            if oof_parts[m]:
                sigma_oof_draws[m].append(_mad_sigma(np.concatenate(oof_parts[m])))
            else:
                sigma_oof_draws[m].append(float("nan"))
        s0 = sigma_oof_draws["C0"][-1]
        for m in MODEL_NAMES:
            sd = sigma_oof_draws[m][-1]
            if math.isfinite(s0) and s0 > 0.0 and math.isfinite(sd):
                red_sigma_draws[m].append((s0 - sd) / s0)
            else:
                red_sigma_draws[m].append(float("nan"))

    def _se(vals):
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return None
        return float(np.std(arr, ddof=1))

    coef_se = {}
    for m in MODEL_NAMES:
        draws = np.asarray(coef_draws[m], dtype=np.float64)
        if draws.ndim != 2 or draws.shape[0] < 2:
            coef_se[m] = [None] * len(model_coef_names(m))
        else:
            coef_se[m] = [float(np.std(draws[:, j], ddof=1)) if np.all(np.isfinite(draws[:, j]))
                          else None for j in range(draws.shape[1])]
    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resampling": "whole_tangent_block_leak_free_per_fold",
        "fold_aggregate": "mean_of_finite_fold_coefficients",
        "c0_se": _se(c0_draws),
        "coef_se": coef_se,
        "sigma_oof_se": {m: _se(sigma_oof_draws[m]) for m in MODEL_NAMES},
        "red_sigma_se": {m: _se(red_sigma_draws[m]) for m in MODEL_NAMES},
        "_coef_draws": coef_draws,
    }


def _band_edge_design(model: str, n: np.ndarray, t: np.ndarray, W, H, orientation) -> np.ndarray:
    half = (MEASUREMENT_PX / 2.0) / ((W / 2.0) if orientation == "h" else (H / 2.0))
    t_mean = float(np.mean(t)) if t.size else 0.0
    n_e = np.array([-half, half], dtype=np.float64)
    t_e = np.array([t_mean, t_mean], dtype=np.float64)
    return model_design(model, n_e, t_e)


def _model_report(model: str, n, t, y, fold, tangent_block, normal_block,
                  sigma_ref, W, H, orientation, boot, sigma_oof_c0) -> dict:
    A_all = model_design(model, n, t)
    coef_full, w_full, sigma_full = robust_fit(A_all, y)
    names = model_coef_names(model)
    ff = _fold_fits(model, n, t, y, fold, tangent_block, normal_block)
    oof = ff["oof"]
    oof_ok = bool(ff["oof_finite"])
    sigma_oof = _mad_sigma(oof) if oof_ok else None
    red_sigma = None
    if (oof_ok and sigma_oof is not None and math.isfinite(sigma_oof)
            and math.isfinite(sigma_oof_c0) and sigma_oof_c0 > 0.0):
        red_sigma = float((sigma_oof_c0 - sigma_oof) / sigma_oof_c0)

    se_list = boot["coef_se"].get(model) or [None] * len(names)
    coef_full_vals = [float(v) for v in coef_full]

    # per-fold sign evidence: ONLY gate-passing, finite folds may contribute
    signs = {}
    for j, name in enumerate(names):
        vals = [float(c[j]) for c in ff["coefs_gate_ok"] if np.all(np.isfinite(c))]
        pos = sum(1 for v in vals if v > 0.0)
        neg = sum(1 for v in vals if v < 0.0)
        signs[name] = {"n_folds": len(vals), "pos": int(pos), "neg": int(neg)}

    def _axis_ok(name: str) -> tuple[bool, float | None]:
        j = names.index(name)
        se = se_list[j]
        val = coef_full_vals[j]
        if not math.isfinite(val) or se is None or not math.isfinite(se) or se <= 0.0:
            return False, None
        sg = signs[name]
        if sg["pos"] >= MIN_ADMITTED_FOLDS and sg["neg"] == 0:
            fold_sign = 1.0
        elif sg["neg"] >= MIN_ADMITTED_FOLDS and sg["pos"] == 0:
            fold_sign = -1.0
        else:
            return False, None
        if math.copysign(1.0, val) != fold_sign:
            return False, None
        if abs(val) >= COEF_SIGMA_MULT * se:
            return True, fold_sign
        return False, None

    # band-edge predicted uncertainty from LEAK-FREE bootstrap AGGREGATED coefficient draws
    edge_se = None
    draws = np.asarray(boot["_coef_draws"].get(model) or [], dtype=np.float64)
    if draws.ndim == 2 and draws.shape[0] >= 2:
        Xe = _band_edge_design(model, n, t, W, H, orientation)
        preds = Xe @ draws.T              # (2, n_boot)
        if np.all(np.isfinite(preds)):
            edge_se = float(np.max(np.std(preds, axis=1, ddof=1)))
    leverage_ok = (edge_se is not None and math.isfinite(edge_se)
                   and edge_se <= LEVERAGE_SIGMA_FRAC * float(sigma_ref))

    result = {
        "model": model,
        "coef_names": list(names),
        "coef_full": {k: coef_full_vals[i] for i, k in enumerate(names)},
        "coef_se": {k: (None if se_list[i] is None else float(se_list[i])) for i, k in enumerate(names)},
        "resid_sigma_full": _fin(sigma_full),
        "n_folds_gate_ok": int(ff["n_gate_ok"]),
        "folds_required": MIN_ADMITTED_FOLDS,
        "fold_status": [dict(f) for f in ff["folds"]],
        "oof_coverage": float(ff["coverage"]),
        "oof_complete": bool(ff["oof_finite"]),
        "fold_signs": signs,
        "oof": {
            "n": (int(oof.size) if oof_ok else 0),
            "sigma_oof": _fin(sigma_oof),
            "rms0_oof": (_fin(_rms0(oof)) if oof_ok else None),
            "red_sigma": _fin(red_sigma),
        },
        "band_edge_pred_se": _fin(edge_se),
        "leverage_ok": bool(leverage_ok),
        "_oof": (oof if oof_ok else None),
    }

    axes = {}
    if model in ("C1", "C2", "C3"):
        for name in ("b_t", "b_n"):
            if name in names:
                ok, sgn = _axis_ok(name)
                axes[name] = {"ok": bool(ok), "sign": sgn}
    elif model == "C4":
        ok, sgn = _axis_ok("q_n")
        axes["q_n"] = {"ok": bool(ok), "sign": sgn}
    result["axes"] = axes

    folds_ok = ff["n_gate_ok"] >= MIN_ADMITTED_FOLDS
    base = bool(folds_ok and oof_ok and leverage_ok and all(math.isfinite(v) for v in coef_full_vals))
    if model == "C0":
        result["admitted"] = bool(oof_ok)   # baseline: honest OOF coverage gives a valid denominator
    elif model in ("C1", "C2", "C3"):
        # H1: a spatial C1-C3 admission additionally REQUIRES OOF red_sigma >= 0.20
        red_ok = red_sigma is not None and math.isfinite(red_sigma) and red_sigma >= RED_SIGMA_MIN
        result["admitted"] = bool(base and any(axes[k]["ok"] for k in axes) and red_ok)
    else:  # C4 raw gate only; the M3 relative/overall rule is applied by _apply_c4_rule
        result["admitted"] = bool(base and axes["q_n"]["ok"])
    result["c4_relative_reduction"] = None
    result["q_n_state"] = None
    result["c4_reason"] = None
    return result


def _apply_c4_rule(models: Mapping) -> None:
    """C4 admission (§8, M3): requires C3 admitted/valid, the C4 base gates, the
    OVERALL OOF robust-sigma reduction vs C0 >= 0.20, AND the relative reduction
    beyond a valid C3 >= 0.20; otherwise NOT_IDENTIFIABLE / NOT_ADMITTED."""
    c3rep = models.get("C3", {})
    c4rep = models.get("C4", {})
    q_ok = bool((c4rep.get("axes") or {}).get("q_n", {}).get("ok"))
    if not c4rep.get("admitted"):
        c4rep["q_n_state"] = "NOT_ADMITTED"
        c4rep["c4_reason"] = "c4_base_gates"
        return
    if not c3rep.get("admitted"):
        c4rep["admitted"] = False
        c4rep["q_n_state"] = "NOT_IDENTIFIABLE"
        c4rep["c4_reason"] = "C3_not_admitted"
        return
    over = c4rep.get("oof", {}).get("red_sigma")
    if over is None or not math.isfinite(float(over)) or float(over) < RED_SIGMA_MIN:
        c4rep["admitted"] = False
        c4rep["q_n_state"] = "NOT_IDENTIFIABLE"
        c4rep["c4_reason"] = "overall_reduction_below_0.20"
        return
    c3 = c3rep.get("oof", {}).get("sigma_oof")
    c4 = c4rep.get("oof", {}).get("sigma_oof")
    if c3 is None or not math.isfinite(float(c3)) or float(c3) <= 0.0 or \
       c4 is None or not math.isfinite(float(c4)):
        c4rep["admitted"] = False
        c4rep["q_n_state"] = "NOT_IDENTIFIABLE"
        c4rep["c4_reason"] = "c3_denominator"
        return
    rel = (float(c3) - float(c4)) / float(c3)
    c4rep["c4_relative_reduction"] = float(rel)
    if q_ok and rel >= CURVATURE_RED_MIN:
        c4rep["q_n_state"] = "ADMITTED"
    else:
        c4rep["admitted"] = False
        c4rep["q_n_state"] = "NOT_IDENTIFIABLE"
        c4rep["c4_reason"] = "relative_beyond_c3_below_0.20"


# ---------------------------------------------------------------------------
# Channel classification (§9.1)
# ---------------------------------------------------------------------------

def classify_channel(values: Mapping) -> dict:
    """Return the per-channel state + flags from the prepared analysis values."""
    sigma_ref = float(values["sigma_ref"])
    e_post = values["e_post"]
    c0 = float(np.median(e_post))
    c0_se = values.get("c0_se")
    sig_const = bool(c0_se is not None and math.isfinite(c0_se) and c0_se > 0.0
                     and abs(c0) >= COEF_SIGMA_MULT * c0_se)

    support_ok = bool(
        values["effective_train"] is not None
        and float(values["effective_train"]) >= MIN_EFFECTIVE_TRAIN
        and int(values["tangent_blocks_artifact"]) >= MIN_TANGENT_BLOCKS
        and int(values["normal_blocks_artifact"]) >= MIN_NORMAL_BLOCKS
    )
    sigma_oof_c0 = values["models"]["C0"]["oof"]["sigma_oof"]
    denom_ok = bool(sigma_oof_c0 is not None and math.isfinite(sigma_oof_c0) and sigma_oof_c0 > 0.0)

    flags: list[str] = [FLAG_SUPPORT_NOT_AVAILABLE]
    if not support_ok or not denom_ok:
        return {"state": "INCONCLUSIVE", "flags": flags,
                "reason": "support_gate_fail" if not support_ok else "sigma_oof_C0_denominator"}

    best_model = "C0"
    if values["models"]["C4"]["admitted"]:
        best_model = "C4"
    else:
        spatial = [m for m in ("C1", "C2", "C3") if values["models"][m]["admitted"]]
        if spatial:
            def _key(m):
                s = values["models"][m]["oof"]["sigma_oof"]
                return float(s) if s is not None and math.isfinite(float(s)) else float("inf")
            best_model = min(spatial, key=_key)

    rms0_resid = values["models"][best_model]["oof"]["rms0_oof"]
    red_sigma = values["models"][best_model]["oof"]["red_sigma"] or 0.0

    if rms0_resid is None or not math.isfinite(float(rms0_resid)):
        return {"state": "INCONCLUSIVE", "flags": flags, "reason": "nonfinite_oof_rms0"}

    if float(rms0_resid) > RMS0_SIGMA_GATE * sigma_ref:
        if best_model != "C0":
            flags.append(FLAG_SPATIAL_STRUCTURE_PARTIAL)
        return {"state": "UNEXPLAINED_STRUCTURE", "flags": flags,
                "best_model": best_model,
                "rms0_resid_model": float(rms0_resid), "red_sigma": float(red_sigma)}

    if values["models"]["C4"]["admitted"] or any(
            values["models"][m]["admitted"] for m in ("C1", "C2", "C3")):
        # H1 fail-closed recheck: the promotion-relevant states require the winning
        # model's own pooled OOF red_sigma >= 0.20, independently of admission.
        red_ok = (red_sigma is not None and math.isfinite(float(red_sigma))
                  and float(red_sigma) >= RED_SIGMA_MIN)
        if not red_ok:
            red_sigma = 0.0
    if values["models"]["C4"]["admitted"] and float(red_sigma) >= RED_SIGMA_MIN:
        return {"state": "CURVATURE", "flags": flags, "best_model": "C4",
                "rms0_resid_model": float(rms0_resid), "red_sigma": float(red_sigma)}

    if any(values["models"][m]["admitted"] for m in ("C1", "C2", "C3")) and \
            float(red_sigma) >= RED_SIGMA_MIN:
        return {"state": "SPATIAL_STABLE", "flags": flags, "best_model": best_model,
                "rms0_resid_model": float(rms0_resid), "red_sigma": float(red_sigma)}

    # no spatial model admitted
    rms0_const = _rms0(e_post - c0)
    if float(red_sigma) < RED_SIGMA_MIN and sig_const and rms0_const <= RMS0_SIGMA_GATE * sigma_ref:
        return {"state": "CONSTANT_ONLY", "flags": flags, "best_model": "C0",
                "rms0_resid_model": float(rms0_resid), "red_sigma": float(red_sigma),
                "c0": c0, "c0_se": c0_se, "sig_const": True}
    if float(red_sigma) < RED_SIGMA_MIN and (not sig_const) and rms0_const <= RMS0_SIGMA_GATE * sigma_ref:
        return {"state": "NEGLIGIBLE", "flags": flags, "best_model": "C0",
                "rms0_resid_model": float(rms0_resid), "red_sigma": float(red_sigma),
                "c0": c0, "c0_se": c0_se, "sig_const": False}
    return {"state": "INCONCLUSIVE", "flags": flags, "reason": "no_state_reachable"}


# ---------------------------------------------------------------------------
# Relation aggregate / orientation / panel (§9.2-§9.4)
# ---------------------------------------------------------------------------

def _channel_signature(ch_state: Mapping, model: Mapping) -> tuple | None:
    if ch_state["state"] == "CURVATURE":
        q = model["axes"].get("q_n") or {}
        if q.get("ok") and q.get("sign") is not None:
            return (("q",), (int(q["sign"]),))
        return None
    if ch_state["state"] == "SPATIAL_STABLE":
        axes = model.get("axes") or {}
        fam = []
        sign = []
        for name, key in (("b_n", "n"), ("b_t", "t")):
            a = axes.get(name)
            if a and a.get("ok") and a.get("sign") is not None:
                fam.append(key)
                sign.append(int(a["sign"]))
        if fam:
            return (tuple(fam), tuple(sign))
    return None


def aggregate_relation(channel_states: Sequence[Mapping], channel_models: Sequence[Mapping]) -> dict:
    """Relation state from the 2/3 channel vote + hard R1 override (§9.2)."""
    states = [c["state"] for c in channel_states]
    flags: list[str] = []
    best = None
    best_count = 0
    for st in STATE_PRECEDENCE:
        cnt = sum(1 for s in states if s == st)
        if cnt >= 2 and cnt > best_count:
            best = st
            best_count = cnt
    state = best if best is not None else "INCONCLUSIVE"
    if any(s == "UNEXPLAINED_STRUCTURE" for s in states):
        state = "UNEXPLAINED_STRUCTURE"
        flags.append(FLAG_UNEXPLAINED_CHANNEL)
    # relation spatial signature: shared by >= 2 channels
    sig_counts: dict = {}
    for cs, cm in zip(channel_states, channel_models):
        sig = _channel_signature(cs, cm)
        if sig is not None:
            sig_counts[sig] = sig_counts.get(sig, 0) + 1
    signature = None
    for sig, cnt in sorted(sig_counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if cnt >= 2:
            signature = sig
            break
    return {"state": state, "flags": flags, "spatial_signature": signature}


def orientation_qualification(relation_records: Sequence[Mapping]) -> dict:
    out = {}
    for orient in ("h", "v"):
        sig_counts: dict = {}
        for r in relation_records:
            if r["orientation"] != orient:
                continue
            if r["state"] in ("SPATIAL_STABLE", "CURVATURE") and r["spatial_signature"] is not None:
                key = json.dumps(r["spatial_signature"], sort_keys=True)
                sig_counts[key] = sig_counts.get(key, 0) + 1
        qualifying = [k for k, c in sig_counts.items() if c >= 2]
        out[orient] = {
            "n_relations": sum(1 for r in relation_records if r["orientation"] == orient),
            "signature_counts": {k: v for k, v in sorted(sig_counts.items())},
            "qualifying_signatures": sorted(qualifying),
            "qualifies_spatially": bool(qualifying),
        }
    return out


def panel_state(relation_records: Sequence[Mapping], *, not_run: Sequence[str] = ()) -> dict:
    """Exclusive panel precedence with the single A->B authorization source (§9.4)."""
    states = [r["state"] for r in relation_records]
    n_rel = len(relation_records)

    def _count(orient, want):
        return sum(1 for r in relation_records
                   if r["orientation"] == orient and r["state"] in want)

    has_not_run = bool(not_run)
    has_inconclusive = any(s == "INCONCLUSIVE" for s in states)
    has_unexplained = any(s == "UNEXPLAINED_STRUCTURE" for s in states)

    spatial = {"h": _count("h", ("SPATIAL_STABLE", "CURVATURE")),
               "v": _count("v", ("SPATIAL_STABLE", "CURVATURE"))}
    constant = {"h": _count("h", ("CONSTANT_ONLY",)), "v": _count("v", ("CONSTANT_ONLY",))}
    negligible_all = all(s == "NEGLIGIBLE" for s in states)

    orient = orientation_qualification(relation_records)
    h_qual = orient["h"]["qualifies_spatially"]
    v_qual = orient["v"]["qualifies_spatially"]
    distinct_spatial = spatial["h"] + spatial["v"]

    if has_not_run or has_inconclusive:
        return {"panel_state": "INCONCLUSIVE", "r39b_authorized": False, "r39b_track": "NONE",
                "reason": "incomplete_panel_or_inconclusive_relation"}
    if has_unexplained:
        return {"panel_state": "UNEXPLAINED_STRUCTURE", "r39b_authorized": False, "r39b_track": "NONE",
                "reason": "unexplained_relation_or_channel"}
    allowed_mix = all(s in ("SPATIAL_STABLE", "CURVATURE", "CONSTANT_ONLY", "NEGLIGIBLE") for s in states)
    if allowed_mix and h_qual and v_qual and distinct_spatial >= 4:
        return {"panel_state": "SPATIAL_CONFIRMED", "r39b_authorized": True, "r39b_track": "B1",
                "reason": "M_spatial"}
    no_sux = all(s not in ("SPATIAL_STABLE", "CURVATURE", "UNEXPLAINED_STRUCTURE", "INCONCLUSIVE")
                 for s in states)
    if no_sux and constant["h"] >= 2 and constant["v"] >= 2 and \
            all(s in ("CONSTANT_ONLY", "NEGLIGIBLE") for s in states):
        return {"panel_state": "CONSTANT_ONLY", "r39b_authorized": True, "r39b_track": "B0",
                "reason": "M_constant"}
    if negligible_all:
        return {"panel_state": "NEGLIGIBLE", "r39b_authorized": False, "r39b_track": "NONE",
                "reason": "M_negligible"}
    return {"panel_state": "INCONCLUSIVE", "r39b_authorized": False, "r39b_track": "NONE",
            "reason": "mixed_relation_mix", "n_relations": n_rel}


# ---------------------------------------------------------------------------
# PREDICTED feather imprint (§7) — never drives the conclusion
# ---------------------------------------------------------------------------

def predicted_feather_imprint(samples: Sequence[Mapping], meta: Mapping) -> dict:
    """Best-effort PREDICTED imprint via the accepted production ramp.

    Returns ``{"state": "PREDICTED_OK"|"NOT_APPLICABLE"|"NOT_RUN", ...}``. Corner
    samples (a perpendicular boundary within the halo) are NOT_APPLICABLE. Any
    missing prerequisite yields NOT_RUN; it never affects a state or the panel.
    """
    try:
        zcf = _lazy_cell_feather()
        ramp_fn = zcf.feather_weight_field
    except Exception as exc:  # production surface unavailable
        return {"state": "NOT_RUN", "reason": f"production_ramp_unavailable:{type(exc).__name__}"}
    W = int(meta["canvas_w"])
    H = int(meta["canvas_h"])
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    halo = int(meta["halo_px"])

    def _cell_bounds(row, col):
        x0 = col * W // nx
        x1 = (col + 1) * W // nx
        y0 = row * H // ny
        y1 = (row + 1) * H // ny
        return x0, y0, x1, y1

    def _cell_patch(cid):
        ur, uc = _cell_rc(cid)
        cx0, cy0, cx1, cy1 = _cell_bounds(ur, uc)
        px0 = max(0, cx0 - halo)
        py0 = max(0, cy0 - halo)
        px1 = min(W, cx1 + halo)
        py1 = min(H, cy1 + halo)
        bounds = (py0, py1, px0, px1)
        shape = (py1 - py0, px1 - px0)
        ramp = ramp_fn(shape, (cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0), halo,
                       bounds=bounds, canvas_hw=(H, W))
        return ramp, py0, px0, shape

    patch_cache: dict = {}

    def _ramp_vec(cols, rows, cid):
        if cid not in patch_cache:
            patch_cache[cid] = _cell_patch(cid)
        ramp, py0, px0, shape = patch_cache[cid]
        ly = rows - py0
        lx = cols - px0
        if np.any(ly < 0) or np.any(ly >= shape[0]) or np.any(lx < 0) or np.any(lx >= shape[1]):
            raise R39Error("imprint: local coordinate outside patch")
        return ramp[ly, lx]

    # perpendicular (corner) boundaries for the whole canvas, vectorised
    if ny > 1:
        yb = np.array([rb * H // ny for rb in range(1, ny)], dtype=np.int64)
    else:
        yb = np.array([], dtype=np.int64)
    if nx > 1:
        xb = np.array([cb * W // nx for cb in range(1, nx)], dtype=np.int64)
    else:
        xb = np.array([], dtype=np.int64)

    per_relation = []
    total_ok = 0
    total_corner = 0
    for s in samples:
        cols = s["cols"]
        rows = s["rows"]
        orientation = s["orientation"]
        if orientation == "h":
            corner = np.zeros(cols.size, dtype=bool)
            if yb.size:
                corner = np.any(np.abs(rows[:, None] - yb[None, :]) <= halo, axis=1)
        else:
            corner = np.zeros(cols.size, dtype=bool)
            if xb.size:
                corner = np.any(np.abs(cols[:, None] - xb[None, :]) <= halo, axis=1)
        try:
            ru = _ramp_vec(cols, rows, s["u"])
            rv = _ramp_vec(cols, rows, s["v"])
        except Exception as exc:
            return {"state": "NOT_RUN", "reason": f"ramp_reconstruction_failed:{type(exc).__name__}"}
        denom = ru + rv
        wu = np.divide(ru, denom, out=np.full_like(ru, np.nan), where=denom > 0.0)
        imprint = wu * s["e_post"]
        ok = ~corner & np.isfinite(imprint)
        total_ok += int(np.count_nonzero(ok))
        total_corner += int(np.count_nonzero(corner))
        vals = imprint[ok]
        per_relation.append({
            "u": s["u"], "v": s["v"], "channel": s["channel"],
            "n_ok": int(vals.size), "n_corner_not_applicable": int(np.count_nonzero(corner)),
            "imprint_median": _fin(np.median(vals)) if vals.size else None,
            "imprint_p90_abs": _fin(np.percentile(np.abs(vals), 90.0)) if vals.size else None,
        })
    return {"state": "PREDICTED_OK", "label": "PREDICTED", "relations": per_relation,
            "n_samples_ok": total_ok, "n_samples_corner_not_applicable": total_corner,
            "note": "PREDICTED only; never observed post-feather evidence; never drives a state"}


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def analyse_one_relation(s: Mapping, meta: Mapping, *,
                         n_boot: int = BOOTSTRAP_B_DEFAULT) -> dict:
    """Analyse a single recovered relation x channel sample set (§5-§9.1)."""
    n = s["n"]
    t = s["t"]
    y = s["e_post"]
    sigma_ref = s["sigma_ref"]
    fold = _fold_ids(s["tangent_block"])
    tangent_block = s["tangent_block"]
    normal_block = s["normal_block"]

    boot = bootstrap_relation(n, t, y, tangent_block, fold, n_boot=n_boot)
    ff0 = _fold_fits("C0", n, t, y, fold, tangent_block, normal_block)
    sigma_oof_c0 = _mad_sigma(ff0["oof"]) if ff0["oof_finite"] else float("nan")

    models = {}
    for m in MODEL_NAMES:
        models[m] = _model_report(m, n, t, y, fold, tangent_block, normal_block,
                                  sigma_ref, meta["canvas_w"], meta["canvas_h"], s["orientation"],
                                  boot, sigma_oof_c0)
    _apply_c4_rule(models)

    values = {
        "sigma_ref": sigma_ref,
        "e_post": y,
        "e_pre": s["e_pre"],
        "n_holdout": int(y.size),
        "effective_train": s["effective_train"],
        "tangent_blocks": int(np.unique(tangent_block).size),
        "normal_blocks": int(np.unique(normal_block).size),
        "tangent_blocks_artifact": s["artifact_tangent_blocks"],
        "normal_blocks_artifact": s["artifact_normal_blocks"],
        "c0_se": boot["c0_se"],
        "models": models,
    }
    cls = classify_channel(values)

    med = float(np.median(y))
    mad_sigma = _mad_sigma(y)
    rms0 = _rms0(y)
    p90 = _p90_abs(y)
    pre = s["e_pre"]
    red = {}
    for name, fn in (("sigma", _mad_sigma), ("rms", _rms0), ("p90", _p90_abs)):
        denom = fn(pre)
        num = fn(y)
        if math.isfinite(denom) and denom > 0.0 and math.isfinite(num):
            red[name] = float(1.0 - num / denom)
        else:
            red[name] = None

    return {
        "u": s["u"], "v": s["v"], "orientation": s["orientation"], "channel": s["channel"],
        "n_holdout": int(y.size),
        "n_train": s["n_train"],
        "effective_train": _fin(s["effective_train"]),
        "tangent_blocks": values["tangent_blocks"],
        "normal_blocks": values["normal_blocks"],
        "artifact_tangent_blocks": s["artifact_tangent_blocks"],
        "artifact_normal_blocks": s["artifact_normal_blocks"],
        "sigma_ref": _fin(sigma_ref),
        "metrics": {
            "median": _fin(med),
            "mad_sigma": _fin(mad_sigma),
            "rms0": _fin(rms0),
            "p90_abs": _fin(p90),
            "p90_centred_NEW_contextual": _fin(np.percentile(np.abs(y - med), 90.0)),
        },
        "reduction_vs_pre_offset": red,
        "distance_profile": _distance_profile(s),
        "models": {m: {k: v for k, v in models[m].items() if not k.startswith("_")}
                   for m in MODEL_NAMES},
        "state": cls["state"],
        "state_reason": cls.get("reason"),
        "best_model": cls.get("best_model", "C0"),
        "rms0_resid_model": cls.get("rms0_resid_model"),
        "red_sigma": cls.get("red_sigma"),
        "sig_const": cls.get("sig_const"),
        "flags": sorted(set(cls["flags"])),
        "nuisance_covariates": "NOT_AVAILABLE",
        "SUPPORT_NOT_AVAILABLE": True,
        "causal_attribution_state": "NOT_AVAILABLE",
    }


def _record_key(rel: Mapping) -> str:
    return f"{rel['orientation']}|{rel['u']}|{rel['v']}|{int(rel['channel'])}"


def _collect_records(identity: Mapping, *, n_boot: int, scratch=None,
                     max_seconds: float = 0.0, rebuild: bool = False) -> dict:
    """Deterministically analyse every relation×channel record, optionally
    checkpointing between FULLY completed records (resumable, M-feasibility)."""
    meta = identity["meta"]
    offsets = identity["offsets"]
    relations = identity["relations"]
    total = len(relations)
    binding = _checkpoint_binding(identity, effective_b=int(n_boot)) if scratch is not None else None
    cached: dict = {}
    if scratch is not None:
        cached = _load_checkpoint(scratch, binding, rebuild=rebuild)
    reports: list = [None] * total
    t0 = time.perf_counter()
    n_done = 0
    n_new = 0
    n_resumed = 0
    for i, rel in enumerate(relations):
        key = _record_key(rel)
        if key in cached:
            reports[i] = cached[key]
            n_done += 1
            n_resumed += 1
            continue
        s = _recover_relation_samples(rel, meta, offsets)
        rep = analyse_one_relation(s, meta, n_boot=n_boot)
        reports[i] = rep
        cached[key] = rep
        n_new += 1
        n_done += 1
        if scratch is not None:
            _write_checkpoint(scratch, binding, cached, total=total)
        if max_seconds and max_seconds > 0.0 and (time.perf_counter() - t0) > max_seconds \
                and n_done < total:
            return {"complete": False, "n_done": n_done, "n_total": total,
                    "n_resumed": n_resumed, "n_new": n_new}
    return {"complete": True, "records": reports, "n_done": total, "n_total": total,
            "n_resumed": n_resumed, "n_new": n_new}


def _assemble_analysis(identity: Mapping, channel_reports: Sequence[Mapping]) -> dict:
    meta = identity["meta"]
    offsets = identity["offsets"]
    relations = identity["relations"]
    samples_by_rel = [_recover_relation_samples(rel, meta, offsets) for rel in relations]
    struct = structural_validation(samples_by_rel)

    by_rel: dict[tuple, list[int]] = {}
    for idx, rep in enumerate(channel_reports):
        by_rel.setdefault((rep["u"], rep["v"], rep["orientation"]), []).append(idx)

    relation_records = []
    for (u, v, orient), idxs in sorted(by_rel.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        idxs = sorted(idxs, key=lambda i: channel_reports[i]["channel"])
        ch_states = [{"state": channel_reports[i]["state"]} for i in idxs]
        ch_models = [_model_for_signature(channel_reports[i]) for i in idxs]
        agg = aggregate_relation(ch_states, ch_models)
        relation_records.append({
            "u": u, "v": v, "orientation": orient,
            "channels": [channel_reports[i]["channel"] for i in idxs],
            "channel_states": [channel_reports[i]["state"] for i in idxs],
            "state": agg["state"],
            "flags": sorted(set(agg["flags"]) | {FLAG_SUPPORT_NOT_AVAILABLE}),
            "spatial_signature": (None if agg["spatial_signature"] is None
                                  else [list(agg["spatial_signature"][0]), list(agg["spatial_signature"][1])]),
            "nuisance_covariates": "NOT_AVAILABLE",
            "SUPPORT_NOT_AVAILABLE": True,
            "causal_attribution_state": "NOT_AVAILABLE",
        })

    orient = orientation_qualification(relation_records)
    panel = panel_state(relation_records)
    imprint = predicted_feather_imprint(samples_by_rel, meta)
    return {
        "structural_validation": struct,
        "channels": list(channel_reports),
        "relations": relation_records,
        "orientation": orient,
        "panel": panel,
        "predicted_feather_imprint": imprint,
    }


def analyse_records(identity: Mapping, *, n_boot: int = BOOTSTRAP_B_DEFAULT,
                    scratch=None, max_seconds: float = 0.0, rebuild: bool = False) -> dict:
    """Compute every per-relation/channel analysis + aggregates (metrics).

    Returns a dict with ``complete`` False (checkpoint/partial) or True (with the
    assembled analysis and the ``n_resumed`` count).
    """
    res = _collect_records(identity, n_boot=n_boot, scratch=scratch,
                           max_seconds=max_seconds, rebuild=rebuild)
    if not res["complete"]:
        return {"complete": False, "n_done": int(res["n_done"]),
                "n_total": int(res["n_total"]), "n_resumed": int(res["n_resumed"])}
    analysis = _assemble_analysis(identity, res["records"])
    analysis["complete"] = True
    analysis["n_resumed"] = int(res["n_resumed"])
    analysis["n_new"] = int(res["n_new"])
    return analysis


def _model_for_signature(rep: Mapping) -> Mapping:
    """The channel's best model report (used to derive the orientation signature)."""
    best = rep.get("best_model", "C0")
    return rep["models"].get(best, {})


def _distance_profile(s: Mapping) -> dict:
    s_abs = np.abs(s["s"])
    out = []
    for lo, hi in DIST_BINS:
        for side, sel in (("n<0", s["n"] < 0.0), ("n>0", s["n"] > 0.0)):
            m = sel & (s_abs >= lo) & (s_abs < hi)
            vals = s["e_post"][m]
            out.append({
                "abs_s_bin": [lo, hi], "side": side, "count": int(vals.size),
                "robust_median": _fin(np.median(vals)) if vals.size else None,
            })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r39a_spatial_residual.py",
        description="ZM-ZEGRID-R39-A read-only post-offset spatial residual diagnostic",
    )
    p.add_argument("--capture-dir", required=True,
                   help="directory holding the exact R37 capture + manifest (READ-ONLY)")
    p.add_argument("--out", default=None, help="strict-JSON report path (outside the dataset root)")
    p.add_argument("--dataset-root", action="append", default=None,
                   help="extra protected read-only root (repeatable)")
    p.add_argument("--expect-identity", choices=("checked", "any"), default="checked")
    p.add_argument("--report-dir", default=str(REPORT_DIR_DEFAULT))
    p.add_argument("--determinism-check", action="store_true")
    p.add_argument("--scratch-dir", default="",
                   help="checkpoint scratch dir (opt-in; outside dataset/protected roots); '' disables resumability"
                        " (suggested: " + str(SCRATCH_DIR_DEFAULT) + ")")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="soft wall-clock budget; <=0 means no budget")
    p.add_argument("--rebuild-checkpoint", action="store_true",
                   help="discard a stale/mismatched checkpoint explicitly")
    return p


def _write_report(out_path: Path, report: Mapping, policy: Mapping) -> int:
    """Atomic strict-JSON write under the validated write policy (§15)."""
    out_path = _assert_write_allowed(out_path, label="report file", policy=policy)
    raw = str(out_path)
    cat = raw_path_is_ambiguous(raw)
    if cat is not None and cat != "traversal":
        raise BudgetError(f"report path is ambiguous ({cat}); refusing")
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise BudgetError("report parent is a symlink; refusing")
    if symlink_component(out_path, parent if parent.exists() else parent.parent):
        raise BudgetError("report path has a symlinked component; refusing")
    out_path = _assert_write_allowed(out_path, label="report file", policy=policy)
    text = dumps_strict(report)
    fd, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp", dir=str(parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _assert_write_allowed(out_path, label="report file", policy=policy)
        if tmp.is_symlink() or out_path.is_symlink():
            raise BudgetError("report destination or temp is a symlink; refusing")
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return len(text)


def _peak_rss_kib() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def build_report(identity: Mapping, analysis: Mapping, *, args, policy: Mapping,
                 component_checks: Mapping, t_start: float) -> dict:
    computed = identity["computed"]
    panel = analysis["panel"]
    channels = analysis["channels"]
    relations = analysis["relations"]
    expect = str(getattr(args, "expect_identity", "checked"))
    any_mode = expect != "checked"
    executed = len(relations)
    planned = computed["n_relations_geometric"]
    status = "PASS"
    status_reason = "planned compact relations executed"
    if executed < planned:
        status = "PARTIAL"
        status_reason = f"executed {executed}/{planned} planned geometric relations"
    report = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "status": status,
        "status_reason": status_reason,
        "qualification": "DEV_GENERIC" if any_mode else QUALIFICATION_DEV,
        "expect_identity": expect,
        "scientific_acceptance": False,
        "correction_applied": False,
        "evidence_scope": {
            "kind": "diagnostic_dev_bounded",
            "corpus": "R37E Windows816 capture (DEV/diagnostic; held-out already consumed)",
            "expect_identity": expect,
            "conclusion_aggregation": "relation first, channel second (never independent)",
            "relation_level": True,
            "n_relations_planned": planned,
            "n_relations_executed": executed,
            "n_records": len(channels),
            "correction_applied": False,
            "image_pixels_read": False,
            "fits_read": False,
        },
        "write_policy": {
            "dataset_root_protection": policy["mode"],
            "dataset_root_identity_sha256": policy["dataset_root_identity_sha256"],
            "protected_roots_count": len(policy["protected_roots"]),
            "note": "the capture dir and its derived dataset root are NEVER output parents",
        },
        "identity": {
            "expect_identity": expect,
            "computed": computed,
            "artifact": {k: v for k, v in identity["artifact"].items() if k not in ("relations",)},
        },
        "component_checks": component_checks,
        "structural_validation": analysis["structural_validation"],
        "panel": panel,
        "panel_state": panel["panel_state"],
        "orientation": analysis["orientation"],
        # Authorization is disabled outside the exact `checked` identity: a generic
        # DEV_GENERIC run may NEVER open R39-B / claim scientific acceptance.
        "r39b_authorized": bool(panel["r39b_authorized"]) and not any_mode,
        "r39b_track": ("NONE" if any_mode else panel["r39b_track"]),
        "flags": {
            FLAG_SUPPORT_NOT_AVAILABLE: True,
            "nuisance_covariates": "NOT_AVAILABLE",
        },
        "causal_attribution_state": "NOT_AVAILABLE",
        "predicted_feather_imprint": analysis["predicted_feather_imprint"],
        "contextual_final_fits_controls": {
            "state": "NOT_RUN",
            "reason": "contextual FITS controls are non-causal and remain NOT_RUN by default (no image read)",
            "label": "CONTEXTUAL_NON_CAUSAL",
        },
        "import_audit": import_audit(),
        "determinism": {},
        "runtime": {
            "seconds": _fin(time.perf_counter() - t_start),
            "peak_rss_kib": int(_peak_rss_kib()),
            "bootstrap": {"seed": BOOTSTRAP_SEED, "configured_B": BOOTSTRAP_B_DEFAULT,
                          "effective_B": int(args._effective_b) if hasattr(args, "_effective_b") else BOOTSTRAP_B_DEFAULT,
                          "resampling": "whole_tangent_block_leak_free_per_fold",
                          "fold_aggregate": "mean_of_finite_fold_coefficients",
                          "note": "B is frozen (no CLI override); each replicate fits only \
non-test folds and scores untouched folds"},
            "n_records_resumed": int(analysis.get("n_resumed", 0)),
            "n_records_fresh": int(analysis.get("n_new", len(channels))),
            "checkpoint": {
                "enabled": bool(getattr(args, "_scratch_enabled", False)),
                "scratch_identity_sha256": getattr(args, "_scratch_identity", None),
                "note": "scratch path is never persisted; only a one-way identity hash",
            },
        },
        "relations": relations,
        "channels": channels,
    }
    return report


def _not_run_exceptions() -> tuple:
    return (R39Error, IdentityError, ResolutionError, BudgetError, FixtureError,
            FileNotFoundError, MemoryError)


# ---------------------------------------------------------------------------
# Resumable checkpoint (deterministic; binds to artifact + tool contract)
# ---------------------------------------------------------------------------

def _git_head() -> str | None:
    """Best-effort repository HEAD read straight from ``.git`` (no subprocess)."""
    try:
        repo = Path(__file__).resolve().parents[1]
        head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = repo / ".git" / head.split(":", 1)[1].strip()
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip()
        return head or None
    except Exception:
        return None


def _model_constants() -> dict:
    return {
        "models": list(MODEL_NAMES), "K": N_FOLDS, "min_admitted_folds": MIN_ADMITTED_FOLDS,
        "bootstrap_seed": BOOTSTRAP_SEED, "rms0_gate": RMS0_SIGMA_GATE,
        "red_sigma_min": RED_SIGMA_MIN, "cond_max": COND_MAX,
        "coef_sigma_mult": COEF_SIGMA_MULT, "leverage_sigma_frac": LEVERAGE_SIGMA_FRAC,
        "curvature_red_min": CURVATURE_RED_MIN, "inv_tol": INV_TOL,
        "measurement_px": MEASUREMENT_PX, "feather_px": FEATHER_PX,
        "halo_px": HALO_PX, "block_px": BLOCK_PX,
    }


def _checkpoint_binding(identity: Mapping, *, effective_b: int) -> dict:
    computed = identity["computed"]
    return {
        "tool_contract": TOOL_CONTRACT_VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_sha256": computed["artifact_sha256"],
        "artifact_bytes": computed["artifact_bytes"],
        "capture_id": computed["capture_id"],
        "source_set_digest": computed["source_set_digest"],
        "bootstrap_B": int(effective_b),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_constants_sha256": hashlib.sha256(
            dumps_strict(_model_constants()).encode("utf-8")).hexdigest(),
        "repo_head": _git_head(),
    }


def _ckpt_path(scratch: Path, binding: Mapping) -> Path:
    h = hashlib.sha256(dumps_strict(binding).encode("utf-8")).hexdigest()[:32]
    return Path(scratch) / f"r39a_ckpt_{h}.json"


def _load_checkpoint(scratch: Path, binding: Mapping, *, rebuild: bool = False) -> dict:
    """Load a matching checkpoint; STALE/MALFORMED/SYMLINK refuses (fail-closed)."""
    scratch = Path(scratch)
    if rebuild and scratch.exists():
        for p in sorted(scratch.glob("r39a_ckpt_*.json")):
            try:
                p.unlink()
            except OSError:
                pass
    existing = sorted(scratch.glob("r39a_ckpt_*.json")) if scratch.exists() else []
    path = _ckpt_path(scratch, binding)
    if not path.exists():
        if existing:
            raise ResolutionError(
                "a stale checkpoint bound to a different identity exists; refusing "
                "(use --rebuild-checkpoint to discard it)")
        return {}
    if path.is_symlink():
        raise ResolutionError("checkpoint path is a symlink; refusing")
    try:
        data = load_json_strict(path)
    except Exception as exc:
        raise ResolutionError(f"checkpoint is malformed: {type(exc).__name__}") from exc
    if not isinstance(data, dict) or data.get("schema") != CKPT_SCHEMA \
            or data.get("version") != CKPT_VERSION:
        raise ResolutionError("checkpoint schema/version mismatch; refusing")
    if data.get("binding") != binding:
        raise ResolutionError("checkpoint binding mismatch; refusing (use --rebuild-checkpoint)")
    records = data.get("records")
    if not isinstance(records, dict):
        raise ResolutionError("checkpoint records are malformed; refusing")
    return {str(k): v for k, v in records.items()}


def _write_checkpoint(scratch: Path, binding: Mapping, records: Mapping, *, total: int) -> None:
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(scratch, binding)
    payload = {"schema": CKPT_SCHEMA, "version": CKPT_VERSION, "binding": binding,
               "n_total": int(total), "n_completed": int(len(records)), "records": dict(records)}
    text = dumps_strict(payload)
    fd, tmp_name = tempfile.mkstemp(prefix=".ckpt.", suffix=".tmp", dir=str(scratch), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def run(capture_dir: Path, *, expect: str, policy: Mapping, n_boot: int = BOOTSTRAP_B_DEFAULT,
        scratch=None, max_seconds: float = 0.0, rebuild: bool = False) -> dict:
    """Run identity + fixtures + metrics; raises on any hard refusal.

    ``analysis.complete`` may be False (checkpoint/partial). ``scratch``/``max_seconds``
    enable deterministic resumable execution without changing any science.
    """
    identity = validate_identity(capture_dir, expect=expect)
    W = identity["computed"]["canvas_width"]
    H = identity["computed"]["canvas_height"]
    component_checks = {
        "coordinate_fixtures": coordinate_fixtures(W, H),
        "sign_order_fixtures": sign_order_fixtures(),
    }
    analysis = analyse_records(identity, n_boot=n_boot, scratch=scratch,
                               max_seconds=max_seconds, rebuild=rebuild)
    return {"identity": identity, "analysis": analysis, "component_checks": component_checks}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _quiet()
    t_start = time.perf_counter()
    effective_b = int(BOOTSTRAP_B_DEFAULT)   # frozen; read at call time (no CLI override)
    args._effective_b = effective_b

    raw_capture = Path(args.capture_dir).expanduser()
    if raw_capture.is_symlink():
        print(json.dumps({"status": "FAILED", "stage": "prepare", "reason": "capture_dir_symlink",
                          "wrote_report": False}, sort_keys=True))
        return 1
    scratch = None
    try:
        capture_dir = raw_capture.resolve()
        extra = [Path(p).expanduser() for p in (args.dataset_root or [])]
        policy = build_write_policy(capture_dir=capture_dir, extra_roots=extra)
        report_dir = _assert_write_allowed(args.report_dir, label="report dir", policy=policy)
        if args.out is not None:
            raw_out_cat = raw_path_is_ambiguous(str(args.out))
            if raw_out_cat is not None:
                raise ResolutionError(f"output path is ambiguous ({raw_out_cat}); refusing")
        out_path = (Path(args.out).expanduser() if args.out
                    else (report_dir / "r39a_spatial_residual.json"))
        raw_out_cat = raw_path_is_ambiguous(str(out_path))
        if raw_out_cat is not None:
            raise ResolutionError(f"output path is ambiguous ({raw_out_cat}); refusing")
        _assert_write_allowed(out_path, label="report file", policy=policy)
        if str(args.scratch_dir or ""):
            raw_scratch = Path(args.scratch_dir).expanduser()
            if raw_scratch.is_symlink():
                raise ResolutionError("scratch dir is a symlink; refusing")
            scratch = _assert_write_allowed(raw_scratch, label="scratch dir", policy=policy)
            if scratch.exists() and scratch.is_symlink():
                raise ResolutionError("scratch dir is a symlink; refusing")
    except _not_run_exceptions() as exc:
        print(json.dumps({"status": "FAILED", "stage": "prepare", "reason": type(exc).__name__,
                          "wrote_report": False}, sort_keys=True))
        return 1

    args._scratch_enabled = scratch is not None
    args._scratch_identity = (one_way_path_identity("zemosaic.zegrid.r39a.scratch.v1", scratch)
                              if scratch is not None else None)

    try:
        result = run(capture_dir, expect=args.expect_identity, policy=policy, n_boot=effective_b,
                     scratch=scratch, max_seconds=float(args.max_seconds),
                     rebuild=bool(args.rebuild_checkpoint))
    except _not_run_exceptions() as exc:
        print(json.dumps({"status": "FAILED", "stage": "prepare_or_metrics",
                          "reason": type(exc).__name__, "wrote_report": False}, sort_keys=True))
        return 1
    except Exception as exc:  # programming error -> FAILED, never a scientific NOT_RUN
        print(json.dumps({"status": "FAILED", "stage": "unexpected",
                          "reason": type(exc).__name__, "wrote_report": False}, sort_keys=True))
        return 1

    identity = result["identity"]
    analysis = result["analysis"]
    if not analysis.get("complete", False):
        # Partial/checkpointed run: a distinct non-success status and NO final --out.
        print(json.dumps({"status": "PARTIAL_CHECKPOINT",
                          "n_done": int(analysis.get("n_done", 0)),
                          "n_total": int(analysis.get("n_total", 0)),
                          "wrote_report": False}, sort_keys=True))
        return 3

    report = build_report(identity, analysis, args=args, policy=policy,
                          component_checks=result["component_checks"], t_start=t_start)

    roots = [capture_dir] + [Path(p) for p in policy["protected_roots"]]
    safe, leaks = enforce_privacy_boundary(report, roots)
    if leaks:
        safe = minimal_failure_record(stage="privacy", leak_locations=leaks)
        safe["runtime"] = report.get("runtime", {})
    try:
        if args.determinism_check:
            r2 = run(capture_dir, expect=args.expect_identity, policy=policy, n_boot=effective_b,
                     scratch=scratch, max_seconds=0.0, rebuild=False)
            if not r2["analysis"].get("complete", False):
                raise ResolutionError("determinism rerun did not complete")
            a = dumps_strict(_canonical_metrics(analysis))
            b = dumps_strict(_canonical_metrics(r2["analysis"]))
            safe["determinism"] = {
                "kind": "same_process_pure_recompute",
                "stable": a == b,
                "canonical_sha256": hashlib.sha256(a.encode()).hexdigest(),
                "note": "same-process pure recomputation; NOT an independent real-data rerun",
            }
        else:
            a = dumps_strict(_canonical_metrics(analysis))
            safe["determinism"] = {
                "kind": "single_run",
                "canonical_sha256": hashlib.sha256(a.encode()).hexdigest(),
            }
        _write_report(out_path, safe, policy)
    except _not_run_exceptions() as exc:
        print(json.dumps({"status": "FAILED", "stage": "write", "reason": type(exc).__name__,
                          "wrote_report": False}, sort_keys=True))
        return 1
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "stage": "write_unexpected",
                          "reason": type(exc).__name__, "wrote_report": False}, sort_keys=True))
        return 1

    print(json.dumps({"status": str(safe.get("status")), "panel_state": safe.get("panel_state"),
                      "r39b_authorized": safe.get("r39b_authorized"),
                      "r39b_track": safe.get("r39b_track"),
                      "wrote_report": True}, sort_keys=True))
    return 0


def _canonical_metrics(analysis: Mapping) -> dict:
    return {
        "structural_validation": analysis["structural_validation"],
        "channels": analysis["channels"],
        "relations": analysis["relations"],
        "orientation": analysis["orientation"],
        "panel": analysis["panel"],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
