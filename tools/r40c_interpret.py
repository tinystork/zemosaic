#!/usr/bin/env python3
"""ZM-ZEGRID-R40-C-INTERPRET — offline interpretation of the R40-C DEV diagnostic capture.

READ-ONLY, offline, default-safe. This tool consumes **only** the committed R40-C
capture generation (``<capture_root>/<generation_dir>/index.json`` + its shards),
its ``zegrid_manifest.json`` cross-link block and the runner's
``r40_capture_verification.json`` report. It **never** opens the final product FITS
(``mosaic_grid*.fits`` / aesthetic / uint16 render), never opens a source frame,
never runs a new capture, and never touches the reserved LP341 corpus.

It implements the frozen R40 plan contract
(``docs/refactor/R40_UNEXPLAINED_STRUCTURE_OBSERVABILITY_PLAN.md``):

* **§3** frozen sample set / record unit / ordering (relation × channel × global pixel),
* **§4** controlled tests T1–T5 and the causal-attribution ladder,
* **§6.4 / §15.5** frozen thresholds (relation-first, 2-of-3, ≥2 relations/orientation,
  whole-tangent-block leak-free bootstrap, ``B = 2000``, seed ``0``),
* **§9** exact STOP outcomes.

Design decisions (NOT threshold changes)
----------------------------------------
The frozen plan leaves a few purely *mechanical* choices to the implementation; they
are recorded here so a reviewer can check them. None changes a frozen threshold, a
frozen aggregation rule or a frozen STOP outcome.

* **Relation-first, channel-second.** A channel is never an independent replicate.
  Per relation × channel the tool computes a signed observational contrast with a
  whole-tangent-block (leak-free) bootstrap band; a relation is *promoted* only when
  **2 of its 3 channels** agree on a band excluding zero with the **same sign**
  (the frozen §15.5-5 / R39 §9.2 2-of-3 rule). An orientation qualifies only with
  **≥2 distinct geometric relations** sharing one explicit axis family + sign tuple
  (frozen §6.4). A test reaches ``DETECTED_ASSOCIATION`` only when **both**
  orientations qualify.
* **Group floors (implementation, minimal).** A contrast is computed only when each
  side has ≥ ``MIN_GROUP_SAMPLES`` samples and ≥ ``MIN_GROUP_BLOCKS`` whole tangent
  blocks (a block bootstrap is otherwise undefined). These are minimal *computability*
  floors, not scientific gates, and they are far below the frozen support gates.
* **T2 stratification.** The weight contrast is taken **within the both-Cells-same-depth
  stratum** (``surviving_sample_count_u == surviving_sample_count_v``) and the depth
  contrast **within the both-Cells-same-weight-class stratum** (median split of the
  estimator-weight covariate, computed before the outcome) — the frozen §4.2
  "contrast within the both-Cells-same-class stratum" rule.
* **T3/T5 are context only.** Their reach is capped at association/context and they can
  never drive the panel STOP. T3 uses the PREDICTED arithmetic imprint (no post-feather
  pixel exists by contract). T5 only reports the distance/tangent profile (context).
* **T4 is not fabricated.** The capture artifact is the *core association* capture
  (M1–M11). It carries **no** §5 same-input-intervention evidence, so T4 is reported
  ``NOT_AVAILABLE`` / H4 ``NOT_IDENTIFIABLE`` — never proxied, never faked.
* **L1 — T1 scope label (explicit).** On this capture T1 is a **SUPPORT-conditioning** test
  only: every captured sample is ``valid_u ∧ valid_v`` (the capture seam samples only the
  shared-valid band), so the M4 valid-mask contrast is **degenerate/empty** (0 discordant
  pairs) and T1 tests the **M7 ``n_eff_support``** hypothesis and **cannot** test the M4
  validity hypothesis here. The durable report states this explicitly
  (``tests.T1.scope`` + a report-level ``t1_scope``). No scientific outcome changes.
* **L2 — co-existing lower state.** When the STOP is ``NUISANCE_LINKED`` the stop block also
  surfaces that the lower ``ASSOCIATION_ONLY`` state **also holds**
  (``coexisting_lower_states``), so the classification is not misread as "no association".
* **L3 — stable content identity.** The durable report carries a ``content_sha256`` computed
  over the strict-JSON report with the informational ``/runtime`` block (and the
  ``content_sha256`` field itself, to avoid self-reference) **excluded**, so the durable
  interpretation has a run-to-run **stable identity** while the file sha256 may differ on
  ``/runtime`` alone. ``/runtime`` is kept for information and **never** enters the content
  identity.
* **L4 — frozen STOP precedence.** The §9 precedence ``IDENTITY_MISMATCH →
  EVIDENCE_UNAVAILABLE → INCOMPLETE_PANEL → UNSUPPORTED_CONFOUNDER → NUISANCE_LINKED →
  ASSOCIATION_ONLY → NO_EFFECT`` was a documented interpretation decision and is now
  **FROZEN** in the plan (§9 / §15.17). It changes **no** §6.4/§15.5 threshold.
* **DRY.** Relation-first aggregation, leak-free whole-block resampling, strict-JSON,
  the path-privacy boundary and the post-offset difference maths are **reused** from the
  accepted ``tools/r39a_spatial_residual.py`` (loaded by path); no pixel or difference
  maths is re-implemented here.

This is a **DEV diagnostic interpretation**: ``DEV_NOT_QUALIFICATION``. It makes no
causality claim beyond the frozen ladder, applies no correction and changes no product.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Schema + frozen expectations
# ---------------------------------------------------------------------------

SCHEMA_NAME = "zemosaic.zegrid.r40c.interpretation"
SCHEMA_VERSION = "r40c-1"
MISSION = "ZM-ZEGRID-R40-C-INTERPRET"
QUALIFICATION_DEV = "DEV_NOT_QUALIFICATION"

BLOCK_PX = 4
MEASUREMENT_PX = 16
FEATHER_PX = 8
HALO_PX = 8
N_CHANNELS = 3
CANVAS_LAYOUT = (6, 4)  # nx, ny

# Frozen §6.4 / §15.5-4 aggregation constants.
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
MIN_RELATIONS_PER_ORIENTATION = 2
CHANNEL_PROMOTE_MIN = 2
CROSS_RELATION_COHORT_MIN = 8

# Minimal computability floors (implementation decision, NOT a scientific gate).
MIN_GROUP_SAMPLES = 32
MIN_GROUP_BLOCKS = 2

# Expected artifact identity of record (frozen at the reviewed capture).
EXPECTED_CAPTURE_ID = "5dcef7b90a2eb109685e7f81fcc714c8ceb07e1ab562ab6747fd31b295848ec5"
EXPECTED_INDEX_SHA256 = "d4681ae937b3c8b0e1785010af669024496c7758106d15fe6f11a9fcb4799f89"
EXPECTED_INDEX_BYTES = 89913
EXPECTED_N_SHARDS = 38
EXPECTED_N_RECORDS = 388887
EXPECTED_N_FRAMES = 816
EXPECTED_ORIENTATIONS = {"h": 20, "v": 18}
EXPECTED_RELATION_CHANNELS = 114
EXPECTED_QUALIFICATION = "EXACT"
EXPECTED_PRODUCTION_WIRED = True
EXPECTED_CORRECTION_APPLIED = False
CAPTURE_STAGE = "pre_offset_pre_feather"
GENERATION_INDEX_BASENAME = "index.json"
GENERATION_COMMIT_BASENAME = "COMMIT"
MANIFEST_BASENAME = "zegrid_manifest.json"
VERIFICATION_BASENAME = "r40_capture_verification.json"
MANIFEST_BLOCK_KEY = "r40_diagnostic_capture"
MANIFEST_BLOCK_FIELDS = ("capture_id", "filename", "sha256", "bytes")

REPORT_DIR_DEFAULT = Path(
    "/home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-R40-C-INTERPRET"
)

# L3 — the informational /runtime block and the content digest field itself are excluded
# from the STABLE content identity (no self-reference, no wall-timing dependence).
CONTENT_IDENTITY_EXCLUDES = ("runtime", "content_sha256")

# Ladder / STOP vocabulary (frozen §4.3 / §9).
STATE_NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
STATE_DETECTED_ASSOCIATION = "DETECTED_ASSOCIATION"
STATE_CONTROLLED_EFFECT = "CONTROLLED_EFFECT"
STATE_CONTEXT = "CONTEXT_ONLY"

STOP_EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
STOP_INCOMPLETE_PANEL = "INCOMPLETE_PANEL"
STOP_IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
STOP_UNSUPPORTED_CONFOUNDER = "UNSUPPORTED_CONFOUNDER"
STOP_ASSOCIATION_ONLY = "ASSOCIATION_ONLY"
STOP_NUISANCE_LINKED = "NUISANCE_LINKED"
STOP_NO_EFFECT = "NO_EFFECT"

HIP_LABELS = ("H1", "H2", "H3", "H4", "H5")

_R39A_MODULE = None


class R40CError(RuntimeError):
    """Base interpretation error."""


class Refusal(R40CError):
    """A fail-closed refusal carrying an exact §9 STOP kind."""

    def __init__(self, stop: str, reason: str):
        super().__init__(f"{stop}: {reason}")
        self.stop = str(stop)
        self.reason = str(reason)


# ---------------------------------------------------------------------------
# Module-scope import audit
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
# DRY: reuse the accepted R39-A machinery (loaded lazily, by path)
# ---------------------------------------------------------------------------

def _load_r39a():
    global _R39A_MODULE
    if _R39A_MODULE is None:
        if "zemosaic" not in sys.modules:
            src = str(Path(__file__).resolve().parents[1] / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
        path = Path(__file__).resolve().parents[1] / "tools" / "r39a_spatial_residual.py"
        spec = importlib.util.spec_from_file_location("_r40c_r39a_reuse", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_r40c_r39a_reuse"] = mod
        spec.loader.exec_module(mod)
        _R39A_MODULE = mod
    return _R39A_MODULE


def _load_r40_capture():
    """Lazy access to the accepted capture-core (identity + verify machinery)."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from zemosaic.core.zegrid import r40_capture as rc  # noqa: WPS433 (lazy by design)
    return rc


# ---------------------------------------------------------------------------
# Strict JSON helpers (reused from R39-A)
# ---------------------------------------------------------------------------

def dumps_strict(payload: Any) -> str:
    return _load_r39a().dumps_strict(payload)


def load_json_strict(path: Path) -> Any:
    return _load_r39a().load_json_strict(Path(path))


def sha256_file(path: Path) -> str:
    return _load_r39a().sha256_file(Path(path))


def _fin(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Identity-first, fail-closed
# ---------------------------------------------------------------------------

def _load_capture_files(capture_root: Path) -> dict:
    capture_root = Path(capture_root).expanduser()
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise Refusal(STOP_IDENTITY_MISMATCH, "capture root missing or symlinked")
    manifest_path = capture_root / MANIFEST_BASENAME
    verification_path = capture_root / VERIFICATION_BASENAME
    if not manifest_path.is_file():
        raise Refusal(STOP_IDENTITY_MISMATCH, "manifest missing")
    if not verification_path.is_file():
        raise Refusal(STOP_IDENTITY_MISMATCH, "capture verification report missing")
    manifest = load_json_strict(manifest_path)
    verification = load_json_strict(verification_path)
    if not isinstance(manifest, Mapping) or not isinstance(verification, Mapping):
        raise Refusal(STOP_IDENTITY_MISMATCH, "manifest/verification not JSON objects")

    block = manifest.get(MANIFEST_BLOCK_KEY)
    if not isinstance(block, Mapping):
        raise Refusal(STOP_IDENTITY_MISMATCH, "manifest r40_diagnostic_capture block missing")
    if set(block) != set(MANIFEST_BLOCK_FIELDS):
        raise Refusal(
            STOP_IDENTITY_MISMATCH,
            f"manifest block field set mismatch: {sorted(block)}",
        )
    filename = str(block.get("filename") or "")
    if not filename or "/" in filename or "\\" in filename:
        raise Refusal(STOP_IDENTITY_MISMATCH, "manifest generation dirname is not a bare basename")
    gen_dir = capture_root / filename
    if not gen_dir.is_dir() or gen_dir.is_symlink():
        raise Refusal(STOP_IDENTITY_MISMATCH, "generation directory missing or symlinked")
    index_path = gen_dir / GENERATION_INDEX_BASENAME
    commit_path = gen_dir / GENERATION_COMMIT_BASENAME
    if not index_path.is_file() or not commit_path.is_file():
        raise Refusal(STOP_IDENTITY_MISMATCH, "generation COMMIT/index missing")
    index = load_json_strict(index_path)
    commit = load_json_strict(commit_path)
    if not isinstance(index, Mapping) or not isinstance(commit, Mapping):
        raise Refusal(STOP_IDENTITY_MISMATCH, "index/COMMIT not JSON objects")
    index_sha = sha256_file(index_path)
    index_bytes = int(index_path.stat().st_size)
    return {
        "capture_root": capture_root,
        "manifest": manifest,
        "verification": verification,
        "block": block,
        "gen_dir": gen_dir,
        "index": index,
        "commit": commit,
        "index_sha256": index_sha,
        "index_bytes": index_bytes,
    }


def verify_capture_identity(capture_root: Path) -> dict:
    """Re-verify every frozen identity binding; any mismatch refuses (§6.2/§9)."""
    files = _load_capture_files(capture_root)
    manifest = files["manifest"]
    verification = files["verification"]
    block = files["block"]
    index = files["index"]
    commit = files["commit"]
    index_sha = files["index_sha256"]
    index_bytes = files["index_bytes"]

    def _need(cond: bool, reason: str) -> None:
        if not cond:
            raise Refusal(STOP_IDENTITY_MISMATCH, reason)

    # --- manifest cross-link == index bytes -----------------------------------
    _need(index_sha == EXPECTED_INDEX_SHA256, "index sha256 != frozen expected")
    _need(index_bytes == EXPECTED_INDEX_BYTES, "index bytes != frozen expected")
    _need(str(block.get("sha256")) == index_sha, "manifest block sha256 != on-disk index sha256")
    _need(int(block.get("bytes", -1)) == index_bytes, "manifest block bytes != on-disk index bytes")
    _need(str(block.get("capture_id")) == EXPECTED_CAPTURE_ID, "manifest capture_id != expected")

    # --- runner verification report -------------------------------------------
    cap = verification.get("capture")
    _need(isinstance(cap, Mapping), "verification.capture missing")
    _need(str(verification.get("schema")) == "zemosaic.zegrid.r40c.capture_verification",
          "verification schema mismatch")
    _need(str(verification.get("status")) == "VERIFIED", "verification status != VERIFIED")
    _need(str(cap.get("capture_id")) == EXPECTED_CAPTURE_ID, "verification capture_id != expected")
    _need(str(cap.get("generation_sha256")) == index_sha, "verification generation_sha256 != index sha256")
    _need(str(cap.get("index_sha256")) == index_sha, "verification index_sha256 != index sha256")
    _need(int(cap.get("generation_bytes", -1)) == index_bytes, "verification generation_bytes mismatch")
    _need(int(cap.get("n_shards", -1)) == EXPECTED_N_SHARDS, "verification n_shards mismatch")
    _need(int(cap.get("n_records", -1)) == EXPECTED_N_RECORDS, "verification n_records mismatch")
    vrep = verification.get("verification")
    _need(isinstance(vrep, Mapping), "verification.verification block missing")
    _need(bool(vrep.get("ok")) is True, "verification.ok is not True")
    _need(list(vrep.get("problems") or []) == [], "verification reports problems")
    _need(bool(vrep.get("expected_production_wired")) is True,
          "verification did not expect production_wired")
    _need(int(vrep.get("checked_records", -1)) == EXPECTED_N_RECORDS, "verification checked_records mismatch")

    # --- COMMIT cross-link ------------------------------------------------------
    _need(str(commit.get("capture_id")) == EXPECTED_CAPTURE_ID, "COMMIT capture_id != expected")
    _need(str(commit.get("index_sha256")) == index_sha, "COMMIT index_sha256 != index sha256")
    _need(int(commit.get("n_shards", -1)) == EXPECTED_N_SHARDS, "COMMIT n_shards mismatch")
    _need(int(commit.get("n_records", -1)) == EXPECTED_N_RECORDS, "COMMIT n_records mismatch")

    # --- index identity (repo_head, dropped, capped, qualification, wiring) -----
    ident = index.get("identity")
    if not isinstance(ident, Mapping):
        raise Refusal(STOP_IDENTITY_MISMATCH, "index identity missing")
    rc = _load_r40_capture()
    try:
        checked = rc.check_capture_identity(ident, expected_production_wired=EXPECTED_PRODUCTION_WIRED)
    except Exception as exc:  # capture identity refusal
        raise Refusal(STOP_IDENTITY_MISMATCH, f"index identity refused: {type(exc).__name__}") from exc
    _need(str(checked["capture_id"]) == EXPECTED_CAPTURE_ID, "index capture_id != expected")
    _need(str(checked["capture_stage"]) == CAPTURE_STAGE, "index capture_stage mismatch")
    _need(bool(checked["correction_applied"]) is False, "index correction_applied must be False")
    _need(bool(checked["production_wired"]) is True, "index production_wired must be True")
    _need(str(checked["qualification"]) == EXPECTED_QUALIFICATION, "index qualification != EXACT")
    _need(int(checked["n_dropped"]) == 0, "index n_dropped != 0")
    _need(bool(checked["holdout_capped"]) is False, "index holdout_capped != False")
    _need(int(checked["holdout_capped"]) == 0, "index holdout_capped inconsistent")
    _need(int(checked["block_px"]) == BLOCK_PX, "index block_px mismatch")
    _need(int(checked["measurement_px"]) == MEASUREMENT_PX, "index measurement_px mismatch")
    _need(int(checked["feather_px"]) == FEATHER_PX, "index feather_px mismatch")
    _need(int(checked["halo_px"]) == HALO_PX, "index halo_px mismatch")
    _need(str(checked["gauge_mode"]) == "per_cell", "index gauge_mode != per_cell")
    _need(int(checked["n_frames"]) == EXPECTED_N_FRAMES,
          f"index n_frames != {EXPECTED_N_FRAMES}")
    repo_head = str(checked["repo_head"])
    _need(bool(repo_head), "index repo_head missing")
    local_head = _git_head()

    # --- generation re-verification (independent, byte-level) -------------------
    try:
        verify = rc.verify_generation(files["gen_dir"], expected_production_wired=True)
    except Exception as exc:
        raise Refusal(STOP_IDENTITY_MISMATCH, f"verify_generation refused: {type(exc).__name__}") from exc
    _need(bool(verify.get("ok")) is True, "verify_generation not ok")
    _need(list(verify.get("problems") or []) == [], "verify_generation reported problems")
    _need(int(verify.get("n_shards", -1)) == EXPECTED_N_SHARDS, "verify n_shards mismatch")
    _need(int(verify.get("n_records", -1)) == EXPECTED_N_RECORDS, "verify n_records mismatch")
    _need(int(verify.get("checked_records", -1)) == EXPECTED_N_RECORDS, "verify checked_records mismatch")
    _need(str(verify.get("index_sha256")) == index_sha, "verify index_sha256 mismatch")

    # --- panel structure: 38 shards / 114 relation-channels ---------------------
    shards = index.get("shards")
    _need(isinstance(shards, list) and len(shards) == EXPECTED_N_SHARDS, "index shards != 38")
    orient_counts = {"h": 0, "v": 0}
    seen_rel = {}
    for entry in shards:
        _need(isinstance(entry, Mapping), "shard entry not an object")
        orient = str(entry.get("orientation"))
        _need(orient in orient_counts, f"unknown orientation {orient!r}")
        orient_counts[orient] += 1
        rel = str(entry.get("relation_id"))
        _need(rel not in seen_rel, f"duplicate relation {rel}")
        seen_rel[rel] = entry
    _need(orient_counts["h"] == EXPECTED_ORIENTATIONS["h"],
          f"orientation h count {orient_counts['h']} != {EXPECTED_ORIENTATIONS['h']}")
    _need(orient_counts["v"] == EXPECTED_ORIENTATIONS["v"],
          f"orientation v count {orient_counts['v']} != {EXPECTED_ORIENTATIONS['v']}")
    layout = checked["layout"]
    _need(int(layout["nx"]) == CANVAS_LAYOUT[0] and int(layout["ny"]) == CANVAS_LAYOUT[1],
          "index layout != frozen 6x4")
    canvas = checked["canvas"]
    if int(canvas["width"]) <= 0 or int(canvas["height"]) <= 0:
        raise Refusal(STOP_IDENTITY_MISMATCH, "index canvas invalid")

    # --- record dtype must equal the frozen core dtype --------------------------
    frozen_dtype = rc.R40_RECORD_DTYPE
    index_dtype = np.dtype([(f["name"], f["dtype"]) for f in index["record_dtype"]])
    _need(index_dtype == frozen_dtype, "index record dtype != frozen capture-core dtype")
    _need(int(index["record_itemsize"]) == int(rc.R40_RECORD_ITEMSIZE), "record itemsize mismatch")

    return {
        "capture_root": files["capture_root"],
        "gen_dir": files["gen_dir"],
        "index": index,
        "manifest": manifest,
        "verification": verification,
        "shards": shards,
        "index_sha256": index_sha,
        "index_bytes": index_bytes,
        "capture_id": EXPECTED_CAPTURE_ID,
        "repo_head": repo_head,
        "local_repo_head": local_head,
        "repo_head_matches_local": bool(local_head is not None and local_head == repo_head),
        "n_shards": len(shards),
        "n_records": int(index["n_records"]),
        "n_relation_channels": EXPECTED_RELATION_CHANNELS,
        "orientation_counts": orient_counts,
        "canvas_width": int(canvas["width"]),
        "canvas_height": int(canvas["height"]),
        "nx": int(layout["nx"]),
        "ny": int(layout["ny"]),
        "dtype": frozen_dtype,
        "offsets": _load_offsets(manifest, checked["canvas"]),
        "manifest_block": dict(block),
        "verification_ok": True,
        "production_wired": True,
        "correction_applied": False,
        "qualification": EXPECTED_QUALIFICATION,
        "holdout_capped": False,
        "n_dropped": 0,
    }


def _load_offsets(manifest: Mapping, canvas: Mapping) -> dict:
    """Frozen R34c additive offsets (per cell x channel) from the manifest."""
    feather = manifest.get("cell_feather")
    if not isinstance(feather, Mapping) or bool(feather.get("enabled")) is not True:
        raise Refusal(STOP_IDENTITY_MISMATCH, "manifest cell_feather missing/disabled")
    raw = feather.get("offsets")
    if not isinstance(raw, Mapping) or not raw:
        raise Refusal(STOP_IDENTITY_MISMATCH, "manifest cell_feather.offsets missing")
    out: dict[str, np.ndarray] = {}
    for cid, vec in raw.items():
        if not isinstance(vec, (list, tuple)) or len(vec) != N_CHANNELS:
            raise Refusal(STOP_IDENTITY_MISMATCH, f"offsets[{cid}] must be a {N_CHANNELS}-vector")
        vals = [float(x) for x in vec]
        if not all(math.isfinite(x) for x in vals):
            raise Refusal(STOP_IDENTITY_MISMATCH, f"offsets[{cid}] has a non-finite entry")
        out[str(cid)] = np.asarray(vals, dtype=np.float64)
    return out


def _git_head() -> str | None:
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


# ---------------------------------------------------------------------------
# Sample recovery (M1–M8) — the exact frozen sample set, relation × channel
# ---------------------------------------------------------------------------

def _channel_samples(index, gen_dir: Path, entry: Mapping, dtype, identity: Mapping) -> list[dict]:
    """Decode one shard into per-channel sample dicts (fail-closed structural checks)."""
    r39 = _load_r39a()
    rc = _load_r40_capture()
    array = np.fromfile(gen_dir / str(entry["filename"]), dtype=dtype)
    if array.size != int(entry["count"]):
        raise Refusal(STOP_EVIDENCE_UNAVAILABLE, f"shard {entry['filename']} count mismatch")
    orientation = str(entry["orientation"])
    u = str(entry["u"])
    v = str(entry["v"])
    relation_id = str(entry["relation_id"])
    W = int(identity["canvas_width"])
    H = int(identity["canvas_height"])
    nx = int(identity["nx"])
    ny = int(identity["ny"])
    block_px = int(identity["index"]["identity"]["block_px"])
    out: list[dict] = []
    for ch in range(N_CHANNELS):
        sel = array[array["channel"] == ch]
        if sel.size == 0:
            raise Refusal(STOP_EVIDENCE_UNAVAILABLE, f"relation {relation_id} channel {ch} empty")
        order = np.lexsort((sel["gx"], sel["gy"]))
        s = sel[order]
        gx = s["gx"].astype(np.int64)
        gy = s["gy"].astype(np.int64)
        # structural: frozen sample key + odd/odd block parity (no assumption)
        keys = s["sample_key"]
        for i in range(keys.size):
            want = rc.sample_key(relation_id, ch, int(gx[i]), int(gy[i]))
            if keys[i].decode("ascii") != want:
                raise Refusal(STOP_EVIDENCE_UNAVAILABLE, "captured sample_key != frozen key formula")
            try:
                rc.assert_block_parity(int(gx[i]), int(gy[i]), block_px)
            except Exception as exc:
                raise Refusal(STOP_EVIDENCE_UNAVAILABLE, "captured key violates frozen block parity") from exc
        # M3 identity: post == f64(pre) + offset, bit-identical (tolerance exactly 0)
        su = s["sci_u_pre"].astype(np.float64)
        sv = s["sci_v_pre"].astype(np.float64)
        offsets = identity["offsets"]
        try:
            off_u = float(offsets[u][ch])
            off_v = float(offsets[v][ch])
        except (KeyError, IndexError) as exc:
            raise Refusal(STOP_EVIDENCE_UNAVAILABLE,
                          "manifest offsets missing for a captured cell") from exc
        pu = s["sci_u_post"]
        pv = s["sci_v_post"]
        if not np.array_equal(pu, su + off_u):
            raise Refusal(STOP_EVIDENCE_UNAVAILABLE, "M3 pre/post bit identity violated (u)")
        if not np.array_equal(pv, sv + off_v):
            raise Refusal(STOP_EVIDENCE_UNAVAILABLE, "M3 pre/post bit identity violated (v)")
        if not (np.all(np.isfinite(pu)) and np.all(np.isfinite(pv))):
            raise Refusal(STOP_EVIDENCE_UNAVAILABLE, "M3 non-finite post-offset value")
        d_post = pu - pv  # reused difference maths (post-offset residual basis)
        n, t, _s = r39.relation_normal_tangent(gx, gy, orientation, W, H, nx, ny, u, v)
        tb = r39.tangent_block_index(gx, gy, orientation)
        nb = r39.normal_block_index(gx, gy, orientation)
        out.append({
            "relation_id": relation_id,
            "u": u, "v": v, "orientation": orientation, "channel": ch,
            "gx": gx, "gy": gy, "n": n, "t": t, "tb": tb, "nb": nb,
            "d_post": d_post,
            "valid_u": s["valid_u"], "valid_v": s["valid_v"],
            "neff_u": s["n_eff_support_u"].astype(np.float64),
            "neff_v": s["n_eff_support_v"].astype(np.float64),
            "sw1_u": s["support_w1_u"].astype(np.float64),
            "sw1_v": s["support_w1_v"].astype(np.float64),
            "sw2_u": s["support_w2_u"].astype(np.float64),
            "sw2_v": s["support_w2_v"].astype(np.float64),
            "w_u": s["estimator_weight_sum_u"].astype(np.float64),
            "w_v": s["estimator_weight_sum_v"].astype(np.float64),
            "surv_u": s["surviving_sample_count_u"].astype(np.int64),
            "surv_v": s["surviving_sample_count_v"].astype(np.int64),
            "ramp_u": s["ramp_u"].astype(np.float64),
            "ramp_v": s["ramp_v"].astype(np.float64),
            "n_samples": int(s.size),
        })
    return out


# ---------------------------------------------------------------------------
# Leak-free whole-tangent-block bootstrap of a median contrast
# ---------------------------------------------------------------------------

def _block_index_map(blocks: np.ndarray) -> dict:
    return {int(b): np.nonzero(blocks == b)[0] for b in np.unique(blocks)}


def _resample_median(values: np.ndarray, uniq: np.ndarray, bmap: dict, rng) -> float:
    picks = rng.integers(0, uniq.size, size=uniq.size)
    idx = np.concatenate([bmap[int(uniq[p])] for p in picks]) if uniq.size else np.empty(0, int)
    return float(np.median(values[idx]))


def block_contrast(y: np.ndarray, indicator: np.ndarray, tb: np.ndarray, *,
                   n_boot: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict | None:
    """Signed median contrast between the two indicator sides + whole-block band.

    ``indicator`` is ``+1`` / ``-1`` (0 = tie, excluded). Each side is resampled by
    whole tangent blocks with replacement; the band is the ``2.5``–``97.5`` percentile
    of the resampled median difference. Leak-free by construction (no fitting, no
    train/test split). Returns ``None`` when the computability floors are not met.
    """
    y = np.asarray(y, dtype=np.float64)
    ind = np.asarray(indicator, dtype=np.int64)
    blocks = np.asarray(tb, dtype=np.int64)
    pos = ind > 0
    neg = ind < 0
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(np.count_nonzero(neg))
    if n_pos < MIN_GROUP_SAMPLES or n_neg < MIN_GROUP_SAMPLES:
        return None
    up = np.unique(blocks[pos])
    un = np.unique(blocks[neg])
    if up.size < MIN_GROUP_BLOCKS or un.size < MIN_GROUP_BLOCKS:
        return None
    eff = float(np.median(y[pos]) - np.median(y[neg]))
    mp = _block_index_map(blocks[pos])
    mn = _block_index_map(blocks[neg])
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        a = _resample_median(y[pos], up, mp, rng)
        b = _resample_median(y[neg], un, mn, rng)
        draws[i] = a - b
    lo = float(np.percentile(draws, 2.5))
    hi = float(np.percentile(draws, 97.5))
    return {
        "effect": eff, "lo": lo, "hi": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "sign": int(math.copysign(1, eff)) if eff != 0.0 else 0,
        "n_pos": n_pos, "n_neg": n_neg,
        "n_blocks_pos": int(up.size), "n_blocks_neg": int(un.size),
        "n_boot": int(n_boot), "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Relation-first aggregation (frozen §6.4 / §15.5-4)
# ---------------------------------------------------------------------------

def _channel_state(result: dict | None) -> tuple[str, int]:
    if result is None:
        return STATE_NOT_IDENTIFIABLE, 0
    if result["excludes_zero"]:
        return STATE_DETECTED_ASSOCIATION, int(result["sign"])
    return "NO_EFFECT", 0


def aggregate_relation_first(channel_results: Sequence[tuple[str, int]]) -> dict:
    """2-of-3 same-sign channel vote -> relation state (frozen channel rule)."""
    signs: dict[int, int] = {}
    n_detected = 0
    for state, sign in channel_results:
        if state == STATE_DETECTED_ASSOCIATION and sign != 0:
            n_detected += 1
            signs[sign] = signs.get(sign, 0) + 1
    promote_sign = 0
    for sign, cnt in sorted(signs.items(), key=lambda kv: (-kv[1], kv[0])):
        if cnt >= CHANNEL_PROMOTE_MIN:
            promote_sign = sign
            break
    if promote_sign != 0:
        return {"state": STATE_DETECTED_ASSOCIATION, "sign": promote_sign,
                "n_detected_channels": n_detected, "n_same_sign": signs.get(promote_sign, 0)}
    return {"state": "NO_EFFECT", "sign": 0, "n_detected_channels": n_detected,
            "n_same_sign": 0}


def aggregate_orientation(relation_records: Sequence[Mapping]) -> dict:
    """≥2 distinct relations per orientation with one explicit sign tuple (frozen)."""
    out = {}
    for orient in ("h", "v"):
        signs: dict[int, list] = {}
        for r in relation_records:
            if r["orientation"] != orient:
                continue
            if r["state"] == STATE_DETECTED_ASSOCIATION and r["sign"] != 0:
                signs.setdefault(int(r["sign"]), []).append(r["relation_id"])
        qualified = {s: sorted(ids) for s, ids in signs.items()
                     if len(ids) >= MIN_RELATIONS_PER_ORIENTATION}
        out[orient] = {
            "n_relations": sum(1 for r in relation_records if r["orientation"] == orient),
            "detected_signs": {str(s): sorted(ids) for s, ids in sorted(signs.items())},
            "qualified_signs": {str(s): ids for s, ids in sorted(qualified.items())},
            "qualifies": bool(qualified),
        }
    return out


def summarise_test(per_relation_channel: Sequence[Mapping], *, test: str,
                   axis_family: str, cap: str) -> dict:
    """Relation-first summary of one controlled test from channel-level contrasts."""
    by_rel: dict[tuple, list] = {}
    for rec in per_relation_channel:
        key = (rec["orientation"], rec["u"], rec["v"])
        by_rel.setdefault(key, []).append(rec)
    relation_records = []
    for (orient, u, v), recs in sorted(by_rel.items()):
        recs = sorted(recs, key=lambda r: r["channel"])
        ch_states = [(_channel_state(r["result"])) for r in recs]
        agg = aggregate_relation_first(ch_states)
        relation_records.append({
            "relation_id": recs[0]["relation_id"],
            "u": u, "v": v, "orientation": orient,
            "channel_states": [{"channel": r["channel"], "state": s, "sign": g,
                                "effect": (None if r["result"] is None else _fin(r["result"]["effect"])),
                                "lo": (None if r["result"] is None else _fin(r["result"]["lo"])),
                                "hi": (None if r["result"] is None else _fin(r["result"]["hi"])),
                                "excludes_zero": (None if r["result"] is None else r["result"]["excludes_zero"])}
                               for r, (s, g) in zip(recs, ch_states)],
            "state": agg["state"],
            "sign": agg["sign"],
            "n_detected_channels": agg["n_detected_channels"],
            "n_same_sign": agg["n_same_sign"],
        })
    orientation = aggregate_orientation(relation_records)
    h_ok = orientation["h"]["qualifies"]
    v_ok = orientation["v"]["qualifies"]
    if cap == STATE_CONTEXT:
        state = STATE_CONTEXT
    elif h_ok and v_ok:
        state = STATE_DETECTED_ASSOCIATION
    else:
        state = "NO_EFFECT"
    return {
        "test": test,
        "axis_family": axis_family,
        "max_reach": cap,
        "state": state,
        "relation_first": True,
        "channel_rule": f"{CHANNEL_PROMOTE_MIN}-of-3 same-sign",
        "orientations_qualify": {"h": bool(h_ok), "v": bool(v_ok)},
        "n_relations_detected": sum(1 for r in relation_records
                                    if r["state"] == STATE_DETECTED_ASSOCIATION),
        "cohort_min_relations_met": (
            sum(1 for r in relation_records if r["state"] == STATE_DETECTED_ASSOCIATION)
            >= CROSS_RELATION_COHORT_MIN),
        "relations": relation_records,
    }


# ---------------------------------------------------------------------------
# Controlled tests T1..T5 (frozen §4.2)
# ---------------------------------------------------------------------------

def _sign_ne(vu, vv):
    return np.where(vu > vv, 1, np.where(vu < vv, -1, 0)).astype(np.int64)


def _t1_support(sample: Mapping, *, n_boot: int, seed: int) -> dict:
    """T1 (H1) support/validity conditioning on M3/M4/M7 (reach DETECTED_ASSOCIATION)."""
    d = sample["d_post"]
    tb = sample["tb"]
    both_valid = sample["valid_u"] & sample["valid_v"]
    discordant = int(np.count_nonzero(sample["valid_u"] != sample["valid_v"]))
    # validity indicator contrast (degenerate when no discordant pair exists)
    if discordant >= MIN_GROUP_SAMPLES:
        ind_v = np.where(sample["valid_u"] & ~sample["valid_v"], 1,
                         np.where(~sample["valid_u"] & sample["valid_v"], -1, 0))
        validity = block_contrast(d[both_valid | (ind_v != 0)],
                                  ind_v[both_valid | (ind_v != 0)],
                                  tb[both_valid | (ind_v != 0)], n_boot=n_boot, seed=seed)
    else:
        validity = None
    # M7 effective-support conditioning on the both-valid restriction
    d_bv = d[both_valid]
    tb_bv = tb[both_valid]
    neff_contrast = block_contrast(d_bv, _sign_ne(sample["neff_u"][both_valid],
                                                  sample["neff_v"][both_valid]),
                                   tb_bv, n_boot=n_boot, seed=seed)
    return {
        "validity_indicator": validity,
        "validity_discordant_samples": discordant,
        "validity_applicable": bool(discordant >= MIN_GROUP_SAMPLES),
        "neff_support": neff_contrast,
    }


def _t1_scope(t1_test: Mapping) -> dict:
    """L1 — explicit T1 scope label for THIS capture (support-conditioning only).

    The R40-C capture seam samples only the shared-valid band, so every captured sample is
    ``valid_u ∧ valid_v`` and the M4 valid-mask contrast is empty (0 discordant pairs).
    T1 therefore tests ONLY the M7 ``n_eff_support`` hypothesis and **cannot** test the M4
    validity hypothesis here. This is a structural consequence of the frozen capture
    contract; it is labelled, never hidden, and changes no outcome, threshold or gate.
    """
    detail = t1_test.get("detail") or {}
    discordant = int(detail.get("validity_discordant_total", 0))
    applicable = int(detail.get("validity_applicable_relations", 0))
    degenerate = bool(discordant == 0 and applicable == 0)
    return {
        "label": "SUPPORT_CONDITIONING_ONLY",
        "statement": (
            "On THIS capture T1 is a SUPPORT-conditioning test ONLY: every captured sample "
            "is valid_u AND valid_v, so the M4 valid-mask contrast is degenerate/empty "
            "(0 discordant pairs). T1 tests the M7 n_eff_support hypothesis and CANNOT "
            "test the M4 validity hypothesis here."
        ),
        "m4_validity_contrast": "DEGENERATE_EMPTY" if degenerate else "AVAILABLE",
        "m4_validity_discordant_pairs": discordant,
        "m4_validity_applicable_relations": applicable,
        "tests_observable": "M7 n_eff_support",
        "cannot_test_observable": "M4 valid-mask discordance (structurally absent here)",
        "structural_reason": (
            "the capture seam samples only the shared-valid band (valid_u AND valid_v), "
            "so no discordant pair can be captured by construction"
        ),
    }


def _t2_weight_depth(sample: Mapping, *, n_boot: int, seed: int) -> dict:
    """T2 (H2) estimator-weight/survivor stratification on M3/M5/M6.

    The weight contrast is taken within the both-Cells-same-depth stratum; the depth
    contrast within the both-Cells-same-weight-class stratum (median split of the
    weight covariate, fixed before the outcome).
    """
    d = sample["d_post"]
    tb = sample["tb"]
    same_depth = sample["surv_u"] == sample["surv_v"]
    weight_ind = _sign_ne(sample["w_u"], sample["w_v"])
    if int(np.count_nonzero(same_depth & (weight_ind > 0))) >= MIN_GROUP_SAMPLES and \
       int(np.count_nonzero(same_depth & (weight_ind < 0))) >= MIN_GROUP_SAMPLES:
        weight = block_contrast(d[same_depth], weight_ind[same_depth], tb[same_depth],
                                n_boot=n_boot, seed=seed)
    else:
        weight = None

    level = np.minimum(sample["w_u"], sample["w_v"])
    med = float(np.median(level)) if level.size else 0.0
    class_u = np.where(sample["w_u"] >= med, 1, 0)
    class_v = np.where(sample["w_v"] >= med, 1, 0)
    same_weight_class = class_u == class_v
    depth_ind = _sign_ne(sample["surv_u"], sample["surv_v"])
    if int(np.count_nonzero(same_weight_class & (depth_ind > 0))) >= MIN_GROUP_SAMPLES and \
       int(np.count_nonzero(same_weight_class & (depth_ind < 0))) >= MIN_GROUP_SAMPLES:
        depth = block_contrast(d[same_weight_class], depth_ind[same_weight_class],
                               tb[same_weight_class], n_boot=n_boot, seed=seed)
    else:
        depth = None
    return {
        "weight_within_same_depth": weight,
        "depth_within_same_weight_class": depth,
        "weight_median_split": _fin(med),
        "n_same_depth": int(np.count_nonzero(same_depth)),
        "n_same_weight_class": int(np.count_nonzero(same_weight_class)),
    }


def _t3_ramp(sample: Mapping, *, n_boot: int, seed: int) -> dict:
    """T3 (H3) PREDICTED feather-ramp arithmetic on M3/M8 (context only)."""
    ru = sample["ramp_u"]
    rv = sample["ramp_v"]
    den = ru + rv
    w_u = np.divide(ru, den, out=np.full_like(ru, np.nan), where=den > 0.0)
    d = sample["d_post"]
    ok = np.isfinite(w_u)
    imprint = w_u * d
    finite_imprint = imprint[ok]
    med_w = float(np.median(w_u[ok])) if np.any(ok) else 0.0
    ind = np.where(w_u > med_w, 1, np.where(w_u < med_w, -1, 0)).astype(np.int64)
    ramp_contrast = block_contrast(d[ok], ind[ok], sample["tb"][ok], n_boot=n_boot, seed=seed)
    return {
        "predicted_imprint": {
            "label": "PREDICTED",
            "n_ok": int(finite_imprint.size),
            "median": _fin(np.median(finite_imprint)) if finite_imprint.size else None,
            "p90_abs": _fin(np.percentile(np.abs(finite_imprint), 90.0)) if finite_imprint.size else None,
        },
        "ramp_split_median_wu": _fin(med_w),
        "ramp_contrast": ramp_contrast,
        "note": "PREDICTED arithmetic only; no post-feather pixel exists by contract; never promotes",
    }


def _t4_intervention() -> dict:
    """T4 (H4) exact same-input intervention — NOT capturable here (§5 / §8.11)."""
    return {
        "state": STATE_NOT_IDENTIFIABLE,
        "reason": "no_same_input_intervention_evidence_in_core_capture",
        "available": False,
        "intervention_families": [],
        "note": ("the R40-C core association capture (M1-M11) carries no §5 I1a/I1b/I2 "
                 "same-input intervention block; H4 is NOT_AVAILABLE and is never proxied"),
        "max_reach": STATE_CONTROLLED_EFFECT,
        "reachable": False,
    }


def _t5_distance(sample: Mapping, *, n_boot: int, seed: int) -> dict:
    """T5 (H5) distance/tangent context profile on M3/C3 (context only)."""
    r39 = _load_r39a()
    n = sample["n"]
    # physical normal distance in pixels (frozen §3.2): s = n * (W/2) for h, (H/2) for v
    W = int(sample["_canvas_width"])
    H = int(sample["_canvas_height"])
    scale = (W / 2.0) if sample["orientation"] == "h" else (H / 2.0)
    s_px = np.abs(n) * scale
    d = sample["d_post"]
    profile = []
    for lo, hi in r39.DIST_BINS:
        for side, sel in (("n<0", n < 0.0), ("n>0", n > 0.0)):
            m = sel & (s_px >= lo) & (s_px < hi)
            vals = d[m]
            profile.append({
                "abs_s_bin": [lo, hi], "side": side, "count": int(vals.size),
                "robust_median": _fin(np.median(vals)) if vals.size else None,
            })
    near = s_px < 1.0
    far = s_px >= 1.0
    ind = np.where(near, 1, np.where(far, -1, 0)).astype(np.int64)
    contrast = block_contrast(d, ind, sample["tb"], n_boot=n_boot, seed=seed)
    return {
        "profile": profile,
        "near_vs_far_contrast": contrast,
        "note": "CONTEXT ONLY; cannot exclude a mechanism (frozen §4.2 T5)",
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyse_capture(capture_root: Path, *, n_boot: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    identity = verify_capture_identity(capture_root)
    gen_dir = identity["gen_dir"]
    index = identity["index"]
    dtype = identity["dtype"]

    per_rc: list[dict] = []
    for entry in identity["shards"]:
        for sample in _channel_samples(index, gen_dir, entry, dtype, identity):
            per_rc.append(sample)
    n_relation_channels = len(per_rc)
    if n_relation_channels != EXPECTED_RELATION_CHANNELS:
        raise Refusal(STOP_INCOMPLETE_PANEL,
                      f"relation-channels {n_relation_channels} != {EXPECTED_RELATION_CHANNELS}")
    for sample in per_rc:
        sample["_canvas_width"] = identity["canvas_width"]
        sample["_canvas_height"] = identity["canvas_height"]

    tests: dict[str, dict] = {}

    # T1
    t1_ch = []
    for s in per_rc:
        res = _t1_support(s, n_boot=n_boot, seed=seed)
        primary = res["neff_support"] if res["neff_support"] is not None else res["validity_indicator"]
        t1_ch.append({"u": s["u"], "v": s["v"], "orientation": s["orientation"],
                      "relation_id": s["relation_id"], "channel": s["channel"], "result": primary,
                      "detail": {k: v for k, v in res.items() if k not in ("neff_support",)}})
    tests["T1"] = summarise_test(t1_ch, test="T1", axis_family="support_neff",
                                 cap=STATE_DETECTED_ASSOCIATION)
    tests["T1"]["hypothesis"] = "H1"
    tests["T1"]["detail"] = {"validity_discordant_total":
                             int(sum(r["detail"]["validity_discordant_samples"] for r in t1_ch)),
                             "validity_applicable_relations":
                             int(sum(1 for r in t1_ch if r["detail"]["validity_applicable"]))}
    tests["T1"]["scope"] = _t1_scope(tests["T1"])

    # T2
    for axis in ("weight_within_same_depth", "depth_within_same_weight_class"):
        t2_ch = []
        for s in per_rc:
            res = _t2_weight_depth(s, n_boot=n_boot, seed=seed)
            t2_ch.append({"u": s["u"], "v": s["v"], "orientation": s["orientation"],
                          "relation_id": s["relation_id"], "channel": s["channel"],
                          "result": res[axis],
                          "detail": {"n_same_depth": res["n_same_depth"],
                                     "n_same_weight_class": res["n_same_weight_class"]}})
        tests["T2_" + axis] = summarise_test(t2_ch, test="T2", axis_family=axis,
                                             cap=STATE_DETECTED_ASSOCIATION)
        tests["T2_" + axis]["hypothesis"] = "H2"

    # T3
    t3_ch = []
    for s in per_rc:
        res = _t3_ramp(s, n_boot=n_boot, seed=seed)
        t3_ch.append({"u": s["u"], "v": s["v"], "orientation": s["orientation"],
                      "relation_id": s["relation_id"], "channel": s["channel"],
                      "result": res["ramp_contrast"],
                      "detail": {"predicted_imprint": res["predicted_imprint"]}})
    tests["T3"] = summarise_test(t3_ch, test="T3", axis_family="feather_ramp",
                                 cap=STATE_CONTEXT)
    tests["T3"]["hypothesis"] = "H3"

    # T4 (not available — never faked)
    tests["T4"] = _t4_intervention()
    tests["T4"]["hypothesis"] = "H4"

    # T5
    t5_ch = []
    for s in per_rc:
        res = _t5_distance(s, n_boot=n_boot, seed=seed)
        t5_ch.append({"u": s["u"], "v": s["v"], "orientation": s["orientation"],
                      "relation_id": s["relation_id"], "channel": s["channel"],
                      "result": res["near_vs_far_contrast"],
                      "detail": {}})
    tests["T5"] = summarise_test(t5_ch, test="T5", axis_family="distance_profile",
                                 cap=STATE_CONTEXT)
    tests["T5"]["hypothesis"] = "H5"

    panel = _panel_summary(identity, per_rc)
    stop = _stop_outcome(tests, panel)
    return {
        "identity": identity,
        "per_relation_channel_n": n_relation_channels,
        "tests": tests,
        "panel": panel,
        "stop": stop,
    }


def _panel_summary(identity: Mapping, per_rc: Sequence[Mapping]) -> dict:
    rels = sorted({(s["orientation"], s["u"], s["v"]) for s in per_rc})
    channels = {}
    for s in per_rc:
        channels.setdefault((s["orientation"], s["u"], s["v"]), set()).add(s["channel"])
    complete = all(chs == {0, 1, 2} for chs in channels.values())
    return {
        "n_relations": len(rels),
        "n_relation_channels": len(per_rc),
        "expected_relations": EXPECTED_N_SHARDS,
        "expected_relation_channels": EXPECTED_RELATION_CHANNELS,
        "orientation_relations": {
            "h": sum(1 for r in rels if r[0] == "h"),
            "v": sum(1 for r in rels if r[0] == "v"),
        },
        "every_relation_has_all_channels": bool(complete),
        "panel_complete": bool(len(rels) == EXPECTED_N_SHARDS and complete
                               and len(per_rc) == EXPECTED_RELATION_CHANNELS),
        "n_dropped": 0,
        "holdout_capped": False,
        "qualification": EXPECTED_QUALIFICATION,
    }


def _stop_outcome(tests: Mapping, panel: Mapping) -> dict:
    """Exact §9 STOP outcome + the dominant reasons (deterministic precedence)."""
    reasons: list[str] = []
    if not panel["panel_complete"]:
        return {"stop": STOP_INCOMPLETE_PANEL,
                "reasons": ["relation panel is not 114/114"], "nuisance_families": [],
                "association_families": []}
    t1 = tests["T1"]
    t2w = tests["T2_weight_within_same_depth"]
    t2d = tests["T2_depth_within_same_weight_class"]
    observational = {
        "T1_support_neff": t1["state"],
        "T2_weight": t2w["state"],
        "T2_depth": t2d["state"],
    }
    association_families = [name for name, st in observational.items()
                            if st == STATE_DETECTED_ASSOCIATION]
    # T1/T2 test nuisance covariates (support / weight / depth) by construction.
    nuisance_families = list(association_families)
    if nuisance_families:
        reasons.append("relation-first associations detected with support/weight/depth nuisances")
        reasons.append("observational linkage only: no identity-proven same-input intervention (T4) exists")
        reasons.append(
            "the co-existing lower state ASSOCIATION_ONLY ALSO holds: the surviving relations "
            "are observational-only (association without an identity-proven same-input "
            "intervention), so this STOP does NOT mean 'no association'")
        return {"stop": STOP_NUISANCE_LINKED, "reasons": reasons,
                "nuisance_families": nuisance_families,
                "association_families": association_families,
                "observational_states": observational,
                "observational_only": True,
                "causal_attribution": False,
                "association_present": True,
                "coexisting_lower_states": [STOP_ASSOCIATION_ONLY],
                "read_as": ("NUISANCE_LINKED does NOT mean 'no association': the lower "
                            "ASSOCIATION_ONLY state also holds"),
                "precedence_note": ("NUISANCE_LINKED takes precedence over ASSOCIATION_ONLY when the "
                                    "surviving relation-first association is with a nuisance "
                                    "covariate family (support/weight/depth); both block any "
                                    "correction and both are strictly below CONTROLLED_EFFECT")}
    if any(tests[t]["state"] == STATE_DETECTED_ASSOCIATION for t in ("T3", "T5")):
        reasons.append("context/predicted association only")
        return {"stop": STOP_ASSOCIATION_ONLY, "reasons": reasons,
                "nuisance_families": [], "association_families": [],
                "observational_states": observational,
                "observational_only": True, "causal_attribution": False}
    reasons.append("no repeatable association in either orientation")
    return {"stop": STOP_NO_EFFECT, "reasons": reasons,
            "nuisance_families": [], "association_families": [],
            "observational_states": observational,
            "observational_only": True, "causal_attribution": False}


# ---------------------------------------------------------------------------
# Report building / writing
# ---------------------------------------------------------------------------

def report_content_sha256(report: Mapping) -> str:
    """L3 — STABLE content identity of the durable report.

    ``sha256`` over the strict-JSON canonicalisation of the report with the informational
    ``/runtime`` block EXCLUDED (and the ``content_sha256`` field itself, to avoid
    self-reference). Any scientific / identity / criterion change changes the digest;
    wall-timing (``seconds``, ``peak_rss_kib``) never does.
    """
    payload = {k: v for k, v in report.items() if k not in CONTENT_IDENTITY_EXCLUDES}
    return hashlib.sha256(dumps_strict(payload).encode("utf-8")).hexdigest()


def attach_content_identity(report: dict) -> dict:
    """Attach the L3 content-identity descriptor + the stable ``content_sha256``.

    MUST be called AFTER the privacy boundary has finalised every durable value, so the
    digest is taken over exactly what is written.
    """
    report["content_identity"] = {
        "algorithm": "sha256",
        "canonicalisation": "strict-json (sort_keys, allow_nan=False, indent=2, ascii)",
        "excludes": list(CONTENT_IDENTITY_EXCLUDES),
        "note": ("the runtime block is informational only and NEVER enters the content "
                 "identity; the content_sha256 field is excluded to avoid self-reference"),
    }
    report["content_sha256"] = report_content_sha256(report)
    return report


def _ladder(analysis: Mapping) -> dict:
    tests = analysis["tests"]
    return {
        "H1": {"state": tests["T1"]["state"], "max_reach": STATE_DETECTED_ASSOCIATION,
               "test": "T1 (observational conditioning)"},
        "H2": {"state": (STATE_DETECTED_ASSOCIATION if
                         tests["T2_weight_within_same_depth"]["state"] == STATE_DETECTED_ASSOCIATION or
                         tests["T2_depth_within_same_weight_class"]["state"] == STATE_DETECTED_ASSOCIATION
                         else "NO_EFFECT"),
               "max_reach": STATE_DETECTED_ASSOCIATION,
               "test": "T2 (observational stratification)"},
        "H3": {"state": tests["T3"]["state"], "max_reach": STATE_CONTEXT,
               "test": "T3 (PREDICTED arithmetic)"},
        "H4": {"state": STATE_NOT_IDENTIFIABLE, "max_reach": STATE_CONTROLLED_EFFECT,
               "test": "T4 (exact same-input intervention) — NOT AVAILABLE"},
        "H5": {"state": tests["T5"]["state"], "max_reach": STATE_CONTEXT,
               "test": "T5 (context proxy)"},
    }


def build_report(analysis: Mapping, *, n_boot: int, seed: int, t_start: float,
                 policy: Mapping) -> dict:
    identity = analysis["identity"]
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "mission": MISSION,
        "status": "PASS",
        "qualification": QUALIFICATION_DEV,
        "scientific_acceptance": False,
        "correction_applied": False,
        "evidence_scope": {
            "kind": "dev_diagnostic_offline_interpretation",
            "corpus": "R40-C Windows816 DEV diagnostic capture (consumed DEV corpus)",
            "device_claim": {
                "windows816_mechanism_established": False,
                "windows816_prototype_established": False,
                "note": ("DEV_NOT_QUALIFICATION: this interpretation establishes NO Windows816 "
                         "mechanism, prototype, correction or product change"),
            },
            "relation_first": True,
            "channels_are_not_independent_replicates": True,
            "fits_read": False,
            "pixels_read": False,
            "new_capture_run": False,
            "lp341_consumed": False,
            "rel_capture_root": _one_way_identity("capture_root", identity["capture_root"]),
        },
        "artifact_identity": {
            "capture_id": identity["capture_id"],
            "generation_dirname": identity["gen_dir"].name,
            "index_basename": GENERATION_INDEX_BASENAME,
            "index_sha256": identity["index_sha256"],
            "index_bytes": identity["index_bytes"],
            "manifest_block": {k: identity["manifest_block"][k] for k in MANIFEST_BLOCK_FIELDS},
            "repo_head": identity["repo_head"],
            "local_repo_head": identity["local_repo_head"],
            "repo_head_matches_local": identity["repo_head_matches_local"],
            "n_shards": identity["n_shards"],
            "n_records": identity["n_records"],
            "n_relation_channels": identity["n_relation_channels"],
            "orientation_counts": identity["orientation_counts"],
            "production_wired": True,
            "correction_applied": False,
            "holdout_capped": False,
            "qualification": EXPECTED_QUALIFICATION,
            "n_dropped": 0,
            "verify_generation_ok": True,
            "manifest_cross_link_sha256_matches": True,
            "verification_report_sha256_matches": True,
        },
        "panel": analysis["panel"],
        "frozen_criteria": {
            "channel_rule": f"{CHANNEL_PROMOTE_MIN}-of-3 same-sign",
            "min_relations_per_orientation": MIN_RELATIONS_PER_ORIENTATION,
            "cross_relation_cohort_min": CROSS_RELATION_COHORT_MIN,
            "bootstrap": {"B": int(n_boot), "seed": int(seed),
                          "resampling": "whole_tangent_block_leak_free"},
            "m3_identity": {"mode": "BIT_IDENTICAL", "tolerance": 0.0, "post_dtype": "float64"},
            "block_px": BLOCK_PX, "measurement_px": MEASUREMENT_PX,
            "feather_px": FEATHER_PX, "halo_px": HALO_PX,
            "stop_precedence": ["IDENTITY_MISMATCH", "EVIDENCE_UNAVAILABLE", "INCOMPLETE_PANEL",
                                "UNSUPPORTED_CONFOUNDER", "NUISANCE_LINKED", "ASSOCIATION_ONLY",
                                "NO_EFFECT"],
            "threshold_changes": "NONE",
        },
        "stop": analysis["stop"],
        "stop_outcome": analysis["stop"]["stop"],
        "ladder": _ladder(analysis),
        "t1_scope": _t1_scope(analysis["tests"]["T1"]),
        "tests": {name: tests for name, tests in analysis["tests"].items()},
        "causal_attribution_state": analysis["stop"]["stop"],
        "component_checks": {
            "m3_pre_post_bit_identity": {"passed": True, "mode": "BIT_IDENTICAL", "tolerance": 0.0},
            "sample_key_frozen_formula": {"passed": True},
            "block_parity_odd_odd": {"passed": True},
            "record_dtype_matches_core": {"passed": True},
            "no_fits_pixel_read": {"passed": True},
        },
        "runtime": {
            "seconds": _fin(time.perf_counter() - t_start),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "peak_rss_kib": _peak_rss_kib(),
            "privacy_policy_mode": policy.get("mode"),
            "informational": True,
            "excluded_from_content_identity": True,
        },
        "import_audit": import_audit(),
    }


def _one_way_identity(tag: str, path) -> str:
    return _load_r39a().one_way_path_identity(f"zemosaic.zegrid.r40c.{tag}.v1", path)


def _peak_rss_kib() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def write_report(out_path: Path, report: Mapping, policy: Mapping) -> int:
    out_path = Path(out_path).expanduser()
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise Refusal(STOP_IDENTITY_MISMATCH, "report parent is a symlink")
    text = dumps_strict(report)
    fd, tmp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".tmp", dir=str(parent), text=True)
    tmp = Path(tmp_name)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if tmp.is_symlink() or out_path.is_symlink():
            raise Refusal(STOP_IDENTITY_MISMATCH, "report destination is a symlink")
        os.replace(tmp, out_path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return len(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r40c_interpret.py",
        description=("ZM-ZEGRID-R40-C offline DEV diagnostic interpretation of the "
                     "committed R40-C capture generation (read-only; no FITS, no pixels)"),
    )
    p.add_argument("--capture-root", required=True,
                   help="R40-C capture output root (READ-ONLY; holds the manifest + generation)")
    p.add_argument("--out", default=None, help="strict-JSON report path")
    p.add_argument("--report-dir", default=str(REPORT_DIR_DEFAULT))
    p.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t_start = time.perf_counter()
    out_path = Path(args.out) if args.out else (Path(args.report_dir) / "r40c_interpretation.json")
    try:
        analysis = analyse_capture(Path(args.capture_root), n_boot=int(args.bootstrap_b),
                                   seed=int(args.seed))
    except Refusal as exc:
        print(json.dumps({"status": "REFUSED", "stop": exc.stop, "reason": exc.reason,
                          "wrote_report": False}, sort_keys=True))
        return 2
    except Exception as exc:  # pragma: no cover - unexpected
        print(json.dumps({"status": "FAILED", "stop": None, "reason": type(exc).__name__,
                          "wrote_report": False}, sort_keys=True))
        return 1

    policy = {"mode": "report_dir_only"}
    report = build_report(analysis, n_boot=int(args.bootstrap_b), seed=int(args.seed),
                          t_start=t_start, policy=policy)
    # privacy boundary: no absolute private path may survive in a durable value
    r39 = _load_r39a()
    roots = [Path(args.capture_root)] + [Path(args.report_dir)]
    safe, leaks = r39.enforce_privacy_boundary(report, roots)
    if leaks:  # pragma: no cover - defensive
        safe = r39.minimal_failure_record(stage="privacy", leak_locations=leaks)
        safe["status"] = "FAILED"
    # L3 — the stable content identity is taken AFTER the privacy boundary finalises values
    safe = attach_content_identity(safe)
    try:
        write_report(out_path, safe, policy)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "stage": "write",
                          "reason": type(exc).__name__, "wrote_report": False}, sort_keys=True))
        return 1
    print(json.dumps({"status": str(safe.get("status")), "stop": safe.get("stop_outcome"),
                      "wrote_report": True, "out_sha256": sha256_file(out_path),
                      "content_sha256": safe.get("content_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
