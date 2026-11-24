"""M33 P0 — RUN-checkpoint refusal witness (metadata only, NO pixels).

Why this file exists
--------------------
The historical M33 RUN (GPU requested, external spool) was refused by
``R55_REFUSE_HOST_RAM`` at the FIRST RUN wave checkpoint, after the global
gauge had already completed. The mission requires that refusal to be
reproduced by the REAL product functions (context -> candidate -> mapping ->
bundle -> materialization -> ``_r55_run_checkpoint``), with only SYSTEM
OBSERVATIONS injected, and that the attribution be falsifiable (which term
fires, with which margin/raw/phase budget), not a parallel re-derivation.

What is real here (product code, unmodified)
--------------------------------------------
* ``external_stream.estimate_external_working_set`` — exact host/disk pricing;
* ``_select_external_fallback`` — the product selector that DERIVES the column
  batch from a PLAN budget, incl. ``shrink_frozen_column_batch``;
* ``geometry.build_layout`` / ``build_patch`` — the real 12x6 cell grid and the
  halo-extended patches (worst patch is produced, not asserted);
* ``_r55_external_contract`` / ``_r55_active_external_decide`` /
  ``_r55_active_admit_external`` / ``_r55_active_apply`` — contract
  validation, ACTIVE admission, executable bundle materialization;
* ``scheduler.materialize_active_candidate`` / ``next_shrink`` /
  ``_r55_run_checkpoint`` — the mandatory fresh RUN checkpoint itself;
* ``runtime_control.collect_run_facts`` — the strict RUN fact collector.

What is INJECTED (observations only, never an admission/model term)
-------------------------------------------------------------------
* ``MemAvailable`` (raw_available_bytes) — the historical RUN value and, for
  the fitting control, the PLAN-observation floor;
* free disk space of the spool volume — the historical RUN value;
* nothing else. ``FITTED_MEMORY_MODEL``, the external host model, the margin,
  the 90 % budget factor, K=1, the tile, the halo, the science tokens and the
  worker ladders are the shipped ones.

What is RECONSTRUCTED (labelled, never presented as historical)
---------------------------------------------------------------
The historical PLAN-time observation and the per-frame content are not
attested on this host, so:

* the PLAN observation is DERIVED by inversion (``plan_raw_floor``): the
  smallest ``MemAvailable`` whose 15 %-margined host peak still fits the 90 %
  budget. The selector is then run AT that observation and must return the
  attested ``column_batch=7296`` / ``host_peak=15363724306`` by itself;
* frame identities are synthetic (count 21818 attested; the worst Cell carries
  the attested max N=21801). Only the WORST Cell is non-empty: the 56
  non-empty Cells, their individual N and their memberships are NOT attested
  and are NOT replayed. This cannot change the decision under test: the
  external class is frozen single-owner (K=1), and the CELL-STACK feasibility
  prices ``host_peak`` per cell — the number of pending cells enters no
  admission term (it is recorded as ``remaining_units`` only);
* ``equalize_rgb`` is not attested; the witness proves the host peak is
  invariant to it (the reduce phase dominates the finalize phase), so the
  refusal cannot be an artifact of that token.

No FITS is read or written, no gauge is computed, no kernel runs, no CUDA
allocation is attempted: only geometry/metadata arithmetic.

CLI contract (see :func:`main`)
-------------------------------
* ``--plan-raw`` / ``--run-raw`` — MemAvailable per boundary;
* ``--disk-free`` — free disk for the PLAN boundary (and RUN unless
  ``--run-disk-free`` overrides it);
* ``--run-disk-free`` — free disk for the RUN boundary ONLY; this is the
  reproducible disk witness (``R55_REFUSE_DISK``) and leaves the PLAN
  observation untouched;
* ``--no-launch``, ``--out``.

Exit codes: ``0`` = ``status=OK``; ``2`` = PARTIAL evidence for an anticipated
non-OK status (e.g. ``PLAN_REFUSED``), with the missing blocks declared in
``evidence["partial"]["absent_blocks"]`` and nothing fabricated; ``4`` =
checkout guard refusal (nothing executed, nothing written); ``1`` = unexpected
internal error.

Labelled hypotheses (never presented as measurements): the PLAN observation is
derived by inversion (``RECONSTRUCTED``), the non-empty Cells other than the
worst one are NOT replayed, and ``channels=3`` is a reconstructed hypothesis
CONFIRMED only by exact concordance (no other channel count reproduces the
attested host peak — see :func:`channels_probe`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# --- attested historical facts (M33 GPU_new, rt-274ffd459a64 runtime) --------
ATTESTED = {
    "run_id_label": "M33 GPU_new (attested by the 2026-11-24 read-only diagnostic)",
    "n_frames": 21818,
    "max_n_frames": 21801,          # worst Cell r0003c0002
    "worst_cell_id": "r0003c0002",
    #: patch shape as the product consumes it: (height, width).
    "worst_patch_hw": (1062, 1053),
    #: the diagnostic wrote the same patch as "1053x1062" (width x height).
    "worst_patch_width": 1053,
    "worst_patch_height": 1062,
    "canvas_hw": (6180, 12249),     # height, width
    "grid_nx_ny": (12, 6),          # 12 columns x 6 rows = 72 cells, 56 non-empty
    "halo_px": 16,
    "tile_px": 128,
    #: RECONSTRUCTED hypothesis, NOT a historical measurement: the diagnostic
    #: never states the channel count. 3 is the product default for colour
    #: frames and is CONFIRMED here only by exact concordance — with c=1 the
    #: very same geometry prices 15 315 681 094 B instead of the attested
    #: 15 363 724 306 B (the persistent result planes differ), so the attested
    #: peak discriminates the channel count.
    "channels": 3,
    "channels_provenance": (
        "reconstructed hypothesis validated by exact model concordance "
        "(c=1 yields 15 315 681 094 B, c=3 yields the attested 15 363 724 306 B)"
    ),
    "column_batch": 7296,
    "reduction_workspace_bytes": 15273504768,
    "host_peak_bytes": 15363724306,
    "run_raw_available_bytes": 18941517824,
    "run_budget_bytes": 17047366041,
    "run_disk_free_bytes": 462903336960,
    "vram_budget_bytes": 6717387571,
    "gpu_min_tile_px": 32,
    "gpu_t32_workspace_bytes": 15700914304,
    "gpu_declared_max_tile_px": 14,
    "attempt_name": "RUN wave 0 (per_cell_stack 0/72)",
}

#: sha256 of the four product files read by the diagnostic in the Windows
#: runtime slot. Equal hashes here mean "same code observed", NOT immutability
#: since the historical launch.
ATTESTED_RUNTIME_SHA256 = {
    "src/zemosaic/zemosaic_zegrid_mode.py":
        "e17cab6a4a6ec901613c3b96263991f5a471cf3495aa7a0b10c487e77d760ef2",
    "src/zemosaic/core/zegrid/scheduler.py":
        "d7d7847267120af095ea204f59fc178a1c513c3d1a666e43a4386ad67b59c376",
    "src/zemosaic/core/zegrid/external_stream.py":
        "e8a9e1c3528257eb42eddc3ffc790e7252a4809ce4b6fbd60e8514e43d126229",
    "src/zemosaic/core/zegrid/gpu.py":
        "19d3cdd8f2ee02181d6dd2186b3c467a82abacb1c602a199b2aa50ebe7f8f0f5",
}

# Exact repository snapshot whose four product files were compared with the
# Windows runtime slot during P0. This must not follow HEAD: once P1 is
# committed, HEAD is intentionally the corrected code, not the historical
# attribution baseline.
ATTESTED_BASELINE_COMMIT = "07133a167efa3ebac754ab539b1fd8fd7d14e597"


# ---------------------------------------------------------------------------
# checkout guard — ALWAYS import THIS checkout, never another
# ---------------------------------------------------------------------------
# The qualification venv is an editable install of another tree
# (``projects/zemosaic/src``). Running a tool from this worktree without a
# guard therefore imported the FOREIGN product code and failed with
# ``ImportError: cannot import name 'external_stream'`` — a silent provenance
# hazard: a witness that proves nothing about this checkout. The guard pins
# ``<repo>/src`` first, drops every other path that exposes a ``zemosaic``
# package, and refuses (exit 4, no evidence) if the imported module is not
# inside this checkout.

CHECKOUT_REFUSAL_RC = 4


def _refuse_foreign_checkout(detail: str):
    sys.stderr.write(
        "zegrid_m33_p0_witness: REFUSED — cannot pin the product code of this "
        f"checkout: {detail}\n"
        "Nothing was executed and no evidence was written. Remedy: run the "
        "tool from the worktree root (the guard then pins ./src), or fix the "
        "virtualenv's editable install. A witness must never run on foreign "
        "product code.\n"
    )
    raise SystemExit(CHECKOUT_REFUSAL_RC)


def _pin_checkout_source():
    """Pin ``sys.path`` to this checkout's ``src`` and prove it was used."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src").resolve()
    marker = src / "zemosaic" / "__init__.py"
    if not marker.is_file():
        _refuse_foreign_checkout(
            f"no local package at {marker} (this file is not inside a ZeMosaic "
            "checkout)"
        )
    removed = []
    for entry in list(sys.path):
        try:
            candidate = (Path(entry) if entry else Path.cwd()).resolve()
        except (OSError, RuntimeError):
            continue
        if candidate == src:
            continue
        if (candidate / "zemosaic" / "__init__.py").is_file():
            removed.append(str(entry))
            sys.path.remove(entry)
    # Drop editable finders that could answer ``zemosaic`` before sys.path.
    for finder in list(sys.meta_path):
        owner = type(finder).__module__ or ""
        if "editable" in owner.lower() and "zemosaic" in owner.lower():
            sys.meta_path.remove(finder)
            removed.append(f"meta_path:{owner}")
    sys.path.insert(0, str(src))
    import zemosaic

    actual = Path(zemosaic.__file__).resolve()
    if src not in actual.parents:
        _refuse_foreign_checkout(
            f"the imported package is {actual}, outside {src}"
        )
    return {
        "repo_root": str(root),
        "src": str(src),
        "zemosaic_file": str(actual),
        "locked": True,
        "foreign_sys_path_entries_removed": removed,
    }


CODE_SOURCE = _pin_checkout_source()


# ---------------------------------------------------------------------------
# lazy product imports (keeps `--help` usable without the science stack)
# ---------------------------------------------------------------------------

def _product():
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import external_stream as zext
    from zemosaic.core.zegrid import geometry as zg
    from zemosaic.core.zegrid import gpu as zgpu
    from zemosaic.core.zegrid import ram_contract as rcontract
    from zemosaic.core.zegrid import resource_envelope as renv
    from zemosaic.core.zegrid import runtime_control as zrt
    from zemosaic.core.zegrid import scheduler as zsched
    from zemosaic.core.zegrid import storage_profile as zsp

    return dict(
        zz=zz, zext=zext, zg=zg, zgpu=zgpu, rcontract=rcontract, renv=renv,
        zrt=zrt, zsched=zsched, zsp=zsp,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_baseline_bytes(root: Path, rel: str) -> bytes | None:
    """Content of ``rel`` at the exact P0 baseline; None if unavailable."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "show", f"{ATTESTED_BASELINE_COMMIT}:{rel}"],
            cwd=str(root),
            capture_output=True,
            timeout=60,
        )
    except Exception:  # noqa: BLE001 - provenance must never crash the tool
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def checkout_provenance(root: Path | None = None) -> dict:
    """Provenance of the observed files, working tree AND exact P0 baseline.

    ZM-M33 P1 modifies ``zemosaic_zegrid_mode.py`` in the SAME checkout, so the
    P0 historical attestation belongs to ``ATTESTED_BASELINE_COMMIT``. Both
    facts are published, never conflated:

    * ``baseline_ref.all_match`` — the pinned P0 commit carries the code
      observed in the Windows runtime slot;
    * ``working_tree.*`` — what THIS process actually imported while replaying.
    """
    root = repo_root() if root is None else Path(root)
    files = {}
    for rel, attested in ATTESTED_RUNTIME_SHA256.items():
        path = root / rel
        working = hashlib.sha256(path.read_bytes()).hexdigest()
        baseline_bytes = _git_baseline_bytes(root, rel)
        baseline = (
            None if baseline_bytes is None
            else hashlib.sha256(baseline_bytes).hexdigest()
        )
        files[rel] = {
            "sha256_working_tree": working,
            "sha256_baseline_ref": baseline,
            "attested_sha256": attested,
            "working_tree_matches": working == attested,
            "baseline_ref_matches": (
                baseline == attested if baseline is not None else None
            ),
        }
    baseline_all = all(
        item["baseline_ref_matches"] for item in files.values()
    )
    working_all = all(item["working_tree_matches"] for item in files.values())
    baseline_block = {
        "commit": ATTESTED_BASELINE_COMMIT,
        "all_match": bool(baseline_all),
        "diverged_from_attested": sorted(
            rel for rel, item in files.items()
            if item["baseline_ref_matches"] is False
        ),
        "unavailable": sorted(
            rel for rel, item in files.items()
            if item["baseline_ref_matches"] is None
        ),
    }
    return {
        "claim": (
            "same code OBSERVED as the Windows runtime slot at the pinned "
            f"baseline commit {ATTESTED_BASELINE_COMMIT}; this is not proof "
            "of immutability since the historical launch, and the working "
            "tree may carry later missions"
        ),
        "contract": (
            "provenance.all_match is the pinned P0 BASELINE attestation; "
            "provenance.working_tree is the code this replay executed"
        ),
        "all_match": bool(baseline_all),
        "baseline_ref": baseline_block,
        "working_tree": {
            "all_match": bool(working_all),
            "diverged_from_attested": sorted(
                rel for rel, item in files.items()
                if not item["working_tree_matches"]
            ),
        },
        "code_source": dict(CODE_SOURCE),
        "files": files,
    }


# ---------------------------------------------------------------------------
# observation injection (the ONLY doubled inputs)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def injected_observations(*, raw_available_bytes: int, disk_free_bytes: int):
    """Inject MemAvailable + free disk; restore everything on exit.

    ``shutil.disk_usage`` is the module-wide probe used by both the selector
    and the ACTIVE facts; ``runtime_control.read_disk_free`` is the RUN fact
    collector's probe. ``zz.available_memory_bytes`` is the LAUNCH admission
    probe. Nothing else is patched.
    """
    from zemosaic import zemosaic_zegrid_mode as zz
    import zemosaic.core.zegrid.runtime_control as zrt

    real_disk_usage = shutil.disk_usage
    real_read_disk_free = zrt.read_disk_free
    real_available = zz.available_memory_bytes
    profile_before = zz._STACK_STORAGE_PROFILE

    def _usage(path):
        return shutil._ntuple_diskusage(  # type: ignore[attr-defined]
            int(disk_free_bytes) + 1, 1, int(disk_free_bytes)
        )

    try:
        shutil.disk_usage = _usage
        zrt.read_disk_free = lambda root: int(disk_free_bytes)
        zz.available_memory_bytes = lambda: int(raw_available_bytes)
        zz._STACK_STORAGE_PROFILE = "bounded"
        yield
    finally:
        shutil.disk_usage = real_disk_usage
        zrt.read_disk_free = real_read_disk_free
        zz.available_memory_bytes = real_available
        zz._STACK_STORAGE_PROFILE = profile_before


# ---------------------------------------------------------------------------
# geometry fixture (real layout/patch code, synthesized identities)
# ---------------------------------------------------------------------------

def build_canvas(zg, *, width: int, height: int, resolution_deg: float = 0.0007):
    """A canvas with the ATTESTED shape (12249 x 6180) and a valid TAN WCS.

    Only the shape is attested. The WCS is a stand-in: no pixel is ever read
    and no projection is needed, because memberships are synthesized (see
    :func:`build_cell_ctxs`).
    """
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [23.4621, 30.6599]   # M33, cosmetic only
    wcs.wcs.crpix = [width / 2.0, height / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * resolution_deg
    wcs.array_shape = (height, width)
    return zg.GlobalCanvas(
        canvas_id=f"m33-p0-reconstructed-{width}x{height}",
        wcs_header=wcs.to_header(relax=True).tostring(),
        width=int(width),
        height=int(height),
        resolution_deg=float(resolution_deg),
    )


def build_cell_ctxs(zg, canvas, *, nx: int, ny: int, halo_px: int,
                    worst_cell_id: str, max_n_frames: int, frame_ids):
    """The real 12x6 layout and its real halo patches.

    Returns ``(cell_ctxs, patches_by_id, worst_patch_hw)``. Only
    ``worst_cell_id`` is NON-EMPTY (attested N); every other Cell carries an
    empty membership because its contributor count is not attested.
    """
    layout = zg.build_layout(canvas, int(nx), int(ny))
    cell_ctxs = []
    patches = {}
    worst = (0, 0)
    for row, col, bounds in layout.iter_cells(canvas):
        cid = zg.cell_id(row, col)
        cell = zg.ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col, bounds)
        patch = zg.build_patch(canvas, cell, int(halo_px))
        hw = tuple(int(v) for v in patch.patch_shape_hw)
        patches[cid] = hw
        worst = max(worst, hw)
        ids = tuple(frame_ids[: int(max_n_frames)]) if cid == worst_cell_id else ()
        mem = zg.CellMembership(cell_id=cid, core_ids=ids, patch_ids=ids)
        cell_ctxs.append((row, col, cell, patch, mem))
    return cell_ctxs, patches, worst


# ---------------------------------------------------------------------------
# bundle / contract, from the real selector
# ---------------------------------------------------------------------------

def select_external(zz, *, plan_raw_bytes: int, tmp_root: Path, frozen_batch=None):
    """Run the REAL external selector under ``plan_raw_bytes``."""
    from zemosaic.core.zegrid import ram_contract as rcontract

    budget = rcontract.plan_budget_bytes(int(plan_raw_bytes))
    selection = zz._select_external_fallback(
        permitted_tiles=(ATTESTED["tile_px"],),
        worst_patch_hw=ATTESTED["worst_patch_hw"],
        max_n_frames=ATTESTED["max_n_frames"],
        channels=int(ATTESTED["channels"]),
        budget_bytes=int(budget),
        halo_px=ATTESTED["halo_px"],
        cache_root=str(tmp_root),
        storage_profile="bounded",
        frozen_batch=None if frozen_batch is None else int(frozen_batch),
        observation_domain=rcontract.OBSERVATION_DOMAIN_NET_AVAILABLE,
    )
    return {
        "plan_raw_available_bytes": int(plan_raw_bytes),
        "plan_budget_bytes": int(budget),
        "refused": bool(selection.get("refused")),
        "reason": selection.get("reason"),
        "detail": selection.get("detail"),
        "column_batch": selection.get("column_batch"),
        "requested_column_batch": selection.get("requested_column_batch"),
        "effective_column_batch": selection.get("effective_column_batch"),
        "executed_rows_per_batch": selection.get("executed_rows_per_batch"),
        "executed_cols_per_batch": selection.get("executed_cols_per_batch"),
        "host_peak_bytes": selection.get("host_peak_bytes"),
        "reduction_workspace_bytes": (
            None if selection.get("refused")
            else int(selection["pricing"]["ram"]["reduction_workspace_bytes"])
        ),
        "reduction_budget_bytes": selection.get("reduction_budget_bytes"),
        "residual_batch_budget_bytes": selection.get("residual_batch_budget_bytes"),
        "host_fixed_bytes": selection.get("host_fixed_bytes"),
        "disk_envelope_bytes": selection.get("disk_envelope_bytes"),
        "free_bytes": selection.get("free_bytes"),
        "batch_transition": selection.get("batch_transition"),
        "host_components": selection.get("host_components"),
        "selection": selection,
    }


def contract_from_selection(selection: dict) -> dict:
    """The frozen external contract, EXACTLY as the product publishes it.

    Field-for-field copy of the ``external_task_info`` block built by
    ``_build_execution_bundle`` (mode.py) from the same selector output; the
    product's own ``_r55_external_contract`` validator is the gate below, and
    ``_r55_active_apply`` re-validates it before any task is produced.
    """
    from zemosaic import zemosaic_zegrid_mode as zz

    sel = selection["selection"]
    return {
        "class": zz.WORKING_SET_EXTERNAL,
        "tile_size": int(sel["tile"]),
        "column_batch": int(sel["column_batch"]),
        "requested_column_batch": int(sel["requested_column_batch"]),
        "effective_column_batch": int(sel["effective_column_batch"]),
        "column_batch_clamped": bool(sel["column_batch_clamped"]),
        "cache_budget_bytes": max(
            0, int(sel["reduction_budget_bytes"]) - int(sel["host_peak_bytes"])
        ),
        "spool_root": str(sel["spool_root"]),
        "disk_envelope_bytes": int(sel["disk_envelope_bytes"]),
        "reduction_workspace_bytes": int(
            sel["pricing"]["ram"]["reduction_workspace_bytes"]
        ),
        "host_peak_bytes": int(sel["host_peak_bytes"]),
        "host_components": dict(sel["host_components"]),
        "budget_domain": str(sel["budget_domain"]),
        "equalize_rgb": bool(sel["equalize_rgb"]),
        "rejection": str(sel["rejection"]),
        "combine": str(sel["combine"]),
        "batch_sample_bytes_per_position_frame": int(
            sel["pricing"]["ram"]["bytes_per_position_frame"]
        ),
        "batch_fixed_position_bytes": int(
            sel["pricing"]["ram"]["fixed_position_bytes"]
        ),
        # ZM-M33 P1: the frozen external geometry carries the EFFECTIVE halo
        # (the tile may be clipped to a small patch, so it is not invertible).
        "halo_px": int(ATTESTED["halo_px"]),
        "phase_budget_bytes": int(sel["phase_budget_bytes"]),
        "reduction_budget_bytes": int(sel["reduction_budget_bytes"]),
        "residual_batch_budget_bytes": int(
            sel["residual_batch_budget_bytes"]
        ),
    }


def build_bundle(zz, rcontract, *, plan_raw_bytes: int, contract: dict,
                 cell_ctxs, canvas, tmp_root: Path):
    """A fixture ``ExecutionBundle`` carrying the attested frozen contract.

    Every admission-relevant value comes from the selector / attested facts.
    Layout, refinement rank, halo, tile, K and the contract are the frozen
    ones; only the per-Cell memberships of the non-attested Cells are empty.
    """
    budget = rcontract.plan_budget_bytes(int(plan_raw_bytes))
    envelope = rcontract.PlanEnvelope(
        phase=rcontract.PHASE_PLAN,
        raw_available_bytes=int(plan_raw_bytes),
        factor_num=rcontract.PLAN_FACTOR_NUM,
        factor_den=rcontract.PLAN_FACTOR_DEN,
        budget_bytes=int(budget),
        layout=(int(canvas.width and ATTESTED["grid_nx_ny"][0]),
                int(ATTESTED["grid_nx_ny"][1])),
        refinement_rank=int(
            ATTESTED["grid_nx_ny"][0] * ATTESTED["grid_nx_ny"][1]
        ),
        cpu_tile=int(contract["tile_size"]),
        gpu_tile=None,
        backend=rcontract.BACKEND_CPU,
        cells_in_flight=1,
        cache_workers_per_cell=1,
        shared_bytes=0,
        stack_total_bytes=int(contract["host_peak_bytes"]),
        cache_total_bytes=0,
        aggregate_bytes=int(contract["host_peak_bytes"]),
        pinned=False,
        reason="M33 P0 reconstructed external contract (plan observation)",
    )
    rcontract.validate_envelope(envelope)
    working_set_block = {
        "class": zz.WORKING_SET_EXTERNAL,
        "tile_size": int(contract["tile_size"]),
        "column_batch": int(contract["column_batch"]),
    }
    cache_root = Path(tmp_root)
    cell_specs = [
        (cell, patch, mem, str(cache_root / str(cell.cell_id)), 0, 0, None)
        for (_row, _col, cell, patch, mem) in cell_ctxs
    ]
    return zz.ExecutionBundle(
        layout={
            "nx": int(ATTESTED["grid_nx_ny"][0]),
            "ny": int(ATTESTED["grid_nx_ny"][1]),
            "halo_px": int(ATTESTED["halo_px"]),
            # The product keeps ``layout["working_set"]`` pointing at the SAME
            # block object the bundle publishes (rework-1 C4).
            "working_set": working_set_block,
        },
        cell_ctxs=list(cell_ctxs),
        footprints=None,
        cpu_streaming={},
        cpu_streaming_tile=int(contract["tile_size"]),
        gpu_tile_size=None,
        gpu_streaming_tile=None,
        gpu_streaming_host_peak_bytes=None,
        gpu_used=False,
        gpu_pre_execution_fallback=None,
        cell_specs=cell_specs,
        cell_tasks=[],
        plan={
            "mode": "external",
            "cells_in_flight": 1,
            "cache_workers_per_cell": 1,
            "stack_total_bytes": int(contract["host_peak_bytes"]),
            "cache_total_bytes": 0,
            "aggregate_bytes": int(contract["host_peak_bytes"]),
        },
        chosen_mode="external",
        cells_in_flight=1,
        cache_workers_per_cell=1,
        per_cell_footprint=int(contract["host_peak_bytes"]),
        cpu_fallback_info={},
        cpu_fallback_plan=None,
        cpu_fallback_tasks=None,
        envelope=envelope,
        shared_bytes=0,
        cache_worker_footprint_bytes=0,
        worst_patch_px=int(
            ATTESTED["worst_patch_hw"][0] * ATTESTED["worst_patch_hw"][1]
        ),
        gauge_workers=1,
        cache_build_workers=1,
        worker_budget_bytes=int(budget),
        attempts=[],
        storage_profile="bounded",
        stack_working_set=zz.WORKING_SET_EXTERNAL,
        working_set_block=working_set_block,
        external_contract=dict(contract),
    )


# ---------------------------------------------------------------------------
# margins / budgets (derived, never hardcoded)
# ---------------------------------------------------------------------------

def margin_applied_bytes(renv, host_peak_bytes: int) -> int:
    return int(
        renv.ceil_margin_div(
            int(host_peak_bytes),
            int(renv.DEFAULT_MARGIN_NUM),
            int(renv.DEFAULT_MARGIN_DEN),
        )
    )


def plan_raw_floor(zz, rcontract, renv, host_peak_bytes: int) -> int:
    """Smallest ``MemAvailable`` whose 90 % budget still covers the margin.

    ``ceil(host_peak x 115/100) <= 9*raw//10``. This is the PLAN observation
    the frozen contract MUST have been admitted under; it is derived here so
    the witness never invents a PLAN-time byte count.
    """
    need = margin_applied_bytes(renv, host_peak_bytes)
    raw = (need * rcontract.PLAN_FACTOR_DEN + rcontract.PLAN_FACTOR_NUM - 1) // (
        rcontract.PLAN_FACTOR_NUM
    )
    while rcontract.plan_budget_bytes(raw) < need:
        raw += 1
    while raw > 1 and rcontract.plan_budget_bytes(raw - 1) >= need:
        raw -= 1
    return int(raw)


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------

def _plan_phase(*, plan_raw_bytes: int, disk_free_bytes: int,
                root: Path) -> dict:
    """PLAN observation: real selector + real ACTIVE decision + real apply."""
    p = _product()
    zz, zg = p["zz"], p["zg"]
    rcontract, renv, zsched = p["rcontract"], p["renv"], p["zsched"]

    with injected_observations(
        raw_available_bytes=int(plan_raw_bytes),
        disk_free_bytes=int(disk_free_bytes),
    ):
        canvas = build_canvas(
            zg,
            width=ATTESTED["canvas_hw"][1],
            height=ATTESTED["canvas_hw"][0],
        )
        frame_ids = tuple("frame%08d.fits" % i
                          for i in range(ATTESTED["n_frames"]))
        cell_ctxs, patches, worst = build_cell_ctxs(
            zg, canvas,
            nx=ATTESTED["grid_nx_ny"][0], ny=ATTESTED["grid_nx_ny"][1],
            halo_px=ATTESTED["halo_px"],
            worst_cell_id=ATTESTED["worst_cell_id"],
            max_n_frames=ATTESTED["max_n_frames"],
            frame_ids=frame_ids,
        )
        geometry = {
            "canvas_hw": [int(canvas.height), int(canvas.width)],
            "grid_nx_ny": [int(ATTESTED["grid_nx_ny"][0]),
                           int(ATTESTED["grid_nx_ny"][1])],
            "cells": len(cell_ctxs),
            "non_empty_cells_replayed": 1,
            "non_empty_cells_attested": 56,
            "patch_hw_convention": "height,width (patch_shape_hw)",
            "worst_patch_hw_observed": [int(worst[0]), int(worst[1])],
            "worst_patch_hw_attested_height_width": [
                int(ATTESTED["worst_patch_height"]),
                int(ATTESTED["worst_patch_width"]),
            ],
            "worst_patch_matches": [
                int(worst[0]), int(worst[1])
            ] == [int(ATTESTED["worst_patch_height"]),
                  int(ATTESTED["worst_patch_width"])],
            "worst_cell_patch_hw": [
                int(patches[ATTESTED["worst_cell_id"]][0]),
                int(patches[ATTESTED["worst_cell_id"]][1]),
            ],
        }
        selection = select_external(zz, plan_raw_bytes=int(plan_raw_bytes),
                                    tmp_root=root)
        if selection["refused"]:
            return {"geometry": geometry, "selection": selection,
                    "refused": True}
        contract = contract_from_selection(selection)
        if int(contract["column_batch"]) != int(ATTESTED["column_batch"]):
            raise AssertionError(
                "the product selector did not derive the attested column batch "
                f"at the derived PLAN observation: {contract['column_batch']} "
                f"!= {ATTESTED['column_batch']}"
            )
        bundle = build_bundle(
            zz, rcontract,
            plan_raw_bytes=int(plan_raw_bytes),
            contract=contract,
            cell_ctxs=cell_ctxs,
            canvas=canvas,
            tmp_root=root,
        )
        decision = zz._r55_active_external_decide(
            bundle=bundle,
            cell_ctxs=cell_ctxs,
            canvas=canvas,
            halo_px=int(ATTESTED["halo_px"]),
            cache_root=str(root),
            output_dir=str(root),
            frame_ids=frame_ids,
        )
        applied = zz._r55_active_apply(
            bundle,
            mapping=decision.mapping,
            facts=decision.facts,
            stage="p0_replay",
        )
    external_tasks = [
        t for t in applied.cell_tasks
        if len(t) > 7 and isinstance(t[7], dict)
        and t[7].get("class") == zz.WORKING_SET_EXTERNAL
    ]
    return {
        "refused": False,
        "geometry": geometry,
        "selection": selection,
        "contract": contract,
        "bundle": bundle,
        "decision": decision,
        "applied": applied,
        "frame_ids": frame_ids,
        "cell_ctxs": cell_ctxs,
        "canvas": canvas,
        "external_tasks": external_tasks,
    }


def replay(*, plan_raw_bytes: int | None = None,
           run_raw_bytes: int | None = None,
           disk_free_bytes: int | None = None,
           run_disk_free_bytes: int | None = None,
           latched_raw_bytes: int | None = None,
           scratch: str | Path | None = None,
           include_launch: bool = True) -> dict:
    """Replay PLAN -> LAUNCH -> RUN-wave-0 with the real product functions.

    Each boundary is evaluated under ITS OWN injected observation: the PLAN
    observation (derived by inversion, RECONSTRUCTED) and the attested RUN
    observation. No boundary ever sees a more favourable value than its own.
    """
    p = _product()
    zz, rcontract, renv, zrt, zsched = (
        p["zz"], p["rcontract"], p["renv"], p["zrt"], p["zsched"]
    )

    run_raw = (
        ATTESTED["run_raw_available_bytes"]
        if run_raw_bytes is None else int(run_raw_bytes)
    )
    disk_free = (
        ATTESTED["run_disk_free_bytes"]
        if disk_free_bytes is None else int(disk_free_bytes)
    )
    run_disk_free = (
        int(disk_free) if run_disk_free_bytes is None
        else int(run_disk_free_bytes)
    )
    latched_fact = None
    if latched_raw_bytes is not None:
        raw_latched = int(latched_raw_bytes)
        latched_fact = {
            "raw_available_bytes": raw_latched,
            "budget_bytes": int(zrt.run_budget_bytes(raw_latched)),
            "monotonic_ns": 0,
        }
    plan_raw = (
        plan_raw_floor(zz, rcontract, renv, ATTESTED["host_peak_bytes"])
        if plan_raw_bytes is None else int(plan_raw_bytes)
    )

    owned_tmp = scratch is None
    root = Path(tempfile.mkdtemp(prefix="m33-p0-")) if owned_tmp else Path(scratch)
    root.mkdir(parents=True, exist_ok=True)
    evidence: dict = {
        "schema": "zemosaic-m33-p0-run-refusal-witness-v1",
        "attested": dict(ATTESTED),
        "provenance": checkout_provenance(),
        "hypotheses": {
            "channels": {
                "value": int(ATTESTED["channels"]),
                "status": "RECONSTRUCTED (not attested)",
                "validated_by": ATTESTED["channels_provenance"],
            },
            "plan_observation": {
                "status": "RECONSTRUCTED by inversion",
                "note": (
                    "the historical PLAN MemAvailable is not part of the "
                    "attested facts; only a lower bound is derivable"
                ),
            },
            "non_empty_cells": {
                "status": "NOT REPLAYED",
                "note": (
                    "only the worst Cell (attested N) is non-empty; the other "
                    "55 non-empty Cells and their N are unattested and cannot "
                    "change the admission (K=1, per-Cell host_peak)"
                ),
            },
        },
        "observations_injected": {
            "plan_raw_available_bytes": int(plan_raw),
            "plan_raw_available_source": (
                "derived inversion of the 90 % x 15 % contract (RECONSTRUCTED)"
                if plan_raw_bytes is None else "caller supplied"
            ),
            "run_raw_available_bytes": int(run_raw),
            "run_raw_available_source": "attested historical RUN observation",
            "disk_free_bytes": int(disk_free),
            "disk_free_source": "attested historical RUN observation",
            "run_disk_free_bytes": int(run_disk_free),
            "latched_raw_available_bytes": (
                None if latched_raw_bytes is None else int(latched_raw_bytes)
            ),
        },
    }
    try:
        plan = _plan_phase(plan_raw_bytes=plan_raw, disk_free_bytes=disk_free,
                           root=root)
        evidence["geometry"] = plan["geometry"]
        selection = plan["selection"]
        evidence["plan_selection"] = {
            key: selection[key] for key in selection if key != "selection"
        }
        if plan["refused"]:
            evidence["status"] = "PLAN_REFUSED"
            return evidence
        contract = plan["contract"]
        decision = plan["decision"]
        applied = plan["applied"]
        bundle = plan["bundle"]
        evidence["frozen_contract"] = dict(contract)
        evidence["pricing_check"] = {
            "host_peak_bytes_observed": int(contract["host_peak_bytes"]),
            "host_peak_bytes_attested": int(ATTESTED["host_peak_bytes"]),
            "reduction_workspace_bytes_observed": int(
                contract["reduction_workspace_bytes"]
            ),
            "reduction_workspace_bytes_attested": int(
                ATTESTED["reduction_workspace_bytes"]
            ),
            "column_batch_observed": int(contract["column_batch"]),
            "column_batch_attested": int(ATTESTED["column_batch"]),
            "margin_applied_host_peak_bytes": margin_applied_bytes(
                renv, int(contract["host_peak_bytes"])
            ),
        }
        evidence["plan_decision"] = {
            "candidate": zsched.active_candidate_tuple(
                decision.plan_set.chosen
            ),
            "mapping": dict(decision.mapping),
            "working_set_class": str(decision.working_set_class),
            "phase": str(decision.phase),
            "plan_raw_available_bytes": int(plan_raw),
            "budget_bytes": int(decision.facts["budget_bytes"]),
            "gauge_workers": int(decision.facts["gauge_workers"]),
            "gauge_phase_bytes": int(decision.facts["gauge_phase_bytes"]),
            "aggregate_host_peak_bytes": int(
                decision.plan_set.aggregate_host_peak_bytes
            ),
            "worker_trail": [dict(x) for x in decision.worker_trail],
        }
        external_tasks = plan["external_tasks"]
        evidence["applied_bundle"] = {
            "aggregate_bytes": int(applied.envelope.aggregate_bytes),
            "budget_bytes": int(applied.envelope.budget_bytes),
            "stack_total_bytes": int(applied.envelope.stack_total_bytes),
            "cache_total_bytes": int(applied.envelope.cache_total_bytes),
            "external_tasks": len(external_tasks),
            "external_task_bound_bytes": (
                None if not external_tasks else int(external_tasks[0][5])
            ),
            "external_task_contract_batch": (
                None if not external_tasks
                else int(external_tasks[0][7]["column_batch"])
            ),
            "cell_task_mode": applied.chosen_mode,
            "cpu_streaming_tile": int(applied.cpu_streaming_tile),
        }

        # ---- LAUNCH + RUN under the attested RUN observation --------------
        with injected_observations(
            raw_available_bytes=run_raw, disk_free_bytes=run_disk_free
        ):
            if include_launch:
                launch_bundle, _lcand, _lmap, outcome, _trail = (
                    zz._r55_active_admit_external(
                        initial_bundle=bundle, decision=decision
                    )
                )
                evidence["launch_admission"] = {
                    "note": (
                        "the LAUNCH gate compares the bundle aggregate (no "
                        "margin term) with 90 % of the fresh observation"
                    ),
                    "plan_aggregate_bytes": int(
                        launch_bundle.envelope.aggregate_bytes
                    ),
                    "admission_observations": [
                        o.to_dict() for o in outcome.observations
                    ],
                    "admitted": True,
                    "replans": int(outcome.replans),
                }
            facts = zrt.collect_run_facts(
                root=root,
                available_fn=lambda: int(run_raw),
                include_diagnostics=False,
            )
            phase_budget = int(
                zz._r55_margin_raw_ceiling(int(facts["host"]["budget_bytes"]))
            )
            evidence["run_facts"] = {
                "host": dict(facts["host"]),
                "disk": {"free_bytes": int(facts["disk"]["free_bytes"])},
            }
            try:
                checkpoint = zz._r55_run_checkpoint(
                    decision=decision,
                    candidate=decision.plan_set.chosen,
                    mapping=decision.mapping,
                    bundle=bundle,
                    remaining_units=56,   # attested non-empty Cells
                    wave_index=0,
                    facts=facts,
                    latched=latched_fact,
                )
            except zz._R55RunRefused as exc:
                record = dict(exc.record)
                evidence["run_checkpoint"] = {
                    "status": "refused",
                    "reason_code": str(exc.reason_code),
                    "detail": str(exc.detail),
                    "classification": dict(record.get("classification") or {}),
                    "host": dict(record.get("host") or {}),
                    "fresh_host": dict(record.get("fresh_host") or {}),
                    "disk": dict(record.get("disk") or {}),
                    "host_source": record.get("host_source"),
                    "candidate": dict(record.get("candidate") or {}),
                    "mapping": dict(record.get("mapping") or {}),
                    "shrink_steps": list(record.get("shrink_steps") or ()),
                    "unavailable_shrinks": list(
                        record.get("unavailable_shrinks") or ()
                    ),
                    "budget_attribution": {
                        "raw_available_bytes": int(
                            facts["host"]["raw_available_bytes"]
                        ),
                        "run_budget_bytes": int(facts["host"]["budget_bytes"]),
                        "phase_budget_bytes": int(phase_budget),
                        "contract_host_peak_bytes": int(
                            contract["host_peak_bytes"]
                        ),
                        "contract_reduction_workspace_bytes": int(
                            contract["reduction_workspace_bytes"]
                        ),
                        "margin_applied_host_peak_bytes": margin_applied_bytes(
                            renv, int(contract["host_peak_bytes"])
                        ),
                        "margin_over_budget_bytes": (
                            margin_applied_bytes(
                                renv, int(contract["host_peak_bytes"])
                            ) - int(facts["host"]["budget_bytes"])
                        ),
                        "host_peak_over_phase_budget_bytes": (
                            int(contract["host_peak_bytes"]) - int(phase_budget)
                        ),
                        "effective_column_batch": int(
                            contract["effective_column_batch"]
                        ),
                        "batch_bytes": int(
                            contract["host_components"]["batch_total_bytes"]
                        ),
                        "fixed_bytes": int(
                            contract["host_components"]["reduce_fixed_bytes"]
                        ),
                    },
                }
            else:
                _record = dict(checkpoint["record"])
                _after = checkpoint["bundle"]
                _contract = _after.external_contract or {}
                _tasks = [
                    task for task in _after.cell_tasks
                    if len(task) > 7 and isinstance(task[7], dict)
                    and task[7].get("class") == zz.WORKING_SET_EXTERNAL
                ]
                evidence["run_checkpoint"] = {
                    "status": "accepted",
                    "record": _record,
                    "candidate": zsched.active_candidate_tuple(
                        checkpoint["candidate"]
                    ),
                    "mapping": dict(checkpoint["mapping"]),
                    # ZM-M33 P1: what the admitted successor really is.
                    "batch_shrink": (
                        None if not _record.get("batch_shrink")
                        else dict(_record["batch_shrink"])
                    ),
                    "bundle_after": {
                        "column_batch": (
                            None if not _contract
                            else int(_contract.get("column_batch") or 0)
                        ),
                        "host_peak_bytes": (
                            None if not _contract
                            else int(_contract.get("host_peak_bytes") or 0)
                        ),
                        "disk_envelope_bytes": (
                            None if not _contract
                            else int(_contract.get("disk_envelope_bytes") or 0)
                        ),
                        "envelope_budget_bytes": int(
                            _after.envelope.budget_bytes
                        ),
                        "envelope_aggregate_bytes": int(
                            _after.envelope.aggregate_bytes
                        ),
                        "external_tasks": len(_tasks),
                        "external_task_bound_bytes": (
                            None if not _tasks else int(_tasks[0][5])
                        ),
                        "external_task_batch": (
                            None if not _tasks
                            else int(_tasks[0][7]["column_batch"])
                        ),
                    },
                }
        evidence["status"] = "OK"
    finally:
        if owned_tmp:
            shutil.rmtree(root, ignore_errors=True)
    return evidence
def selector_sweep(*, raws, frozen_batch=None, disk_free_bytes=None) -> list:
    """Selector outputs across observations (P1 headroom evidence, no pixels)."""
    p = _product()
    zz = p["zz"]
    disk = (
        ATTESTED["run_disk_free_bytes"]
        if disk_free_bytes is None else int(disk_free_bytes)
    )
    with tempfile.TemporaryDirectory(prefix="m33-p0-sweep-") as tmp:
        with injected_observations(raw_available_bytes=max(int(r) for r in raws),
                                   disk_free_bytes=disk):
            out = []
            for raw in raws:
                item = select_external(
                    zz, plan_raw_bytes=int(raw), tmp_root=Path(tmp),
                    frozen_batch=frozen_batch,
                )
                out.append({
                    key: item[key] for key in item if key != "selection"
                })
            return out


def frozen_batch_shrink_probe(*, phase_budget_bytes: int,
                              frozen_batch: int = ATTESTED["column_batch"]) -> dict:
    """The EXISTING external shrink helper, priced under a RUN phase budget."""
    p = _product()
    zext = p["zext"]
    shrink = zext.shrink_frozen_column_batch(
        frozen_batch=int(frozen_batch),
        phase_budget_bytes=int(phase_budget_bytes),
        n_frames=int(ATTESTED["max_n_frames"]),
        height=int(ATTESTED["worst_patch_hw"][0]),
        width=int(ATTESTED["worst_patch_hw"][1]),
        channels=int(ATTESTED["channels"]),
        tile_size=int(ATTESTED["tile_px"]),
        halo_px=int(ATTESTED["halo_px"]),
    )
    return {"phase_budget_bytes": int(phase_budget_bytes),
            "frozen_batch": int(frozen_batch),
            "shrink": shrink}


def pricing_probe(*, equalize_rgb_values=(False, True),
                  batch: int = ATTESTED["column_batch"]) -> dict:
    """Exact host pricing of the attested worst Cell (no equalize dependency)."""
    p = _product()
    zext = p["zext"]
    out = {}
    for equalize in equalize_rgb_values:
        priced = zext.estimate_external_working_set(
            n_frames=int(ATTESTED["max_n_frames"]),
            height=int(ATTESTED["worst_patch_hw"][0]),
            width=int(ATTESTED["worst_patch_hw"][1]),
            channels=int(ATTESTED["channels"]),
            tile_size=int(ATTESTED["tile_px"]),
            halo_px=int(ATTESTED["halo_px"]),
            column_batch=int(batch),
            equalize_rgb=bool(equalize),
            rejection="winsorized_sigma_clip",
            combine="mean",
        )
        host = priced["host"]
        out["equalize_rgb=%s" % bool(equalize)] = {
            "host_peak_bytes": int(host["peak_bytes"]),
            "reduce_fixed_bytes": int(host["reduce_fixed_bytes"]),
            "batch_total_bytes": int(host["batch_total_bytes"]),
            "batch_positions": int(host["batch_positions"]),
            "effective_column_batch": int(host["effective_column_batch"]),
            "rows_per_batch": int(host["rows_per_batch"]),
            "cols_per_batch": int(host["cols_per_batch"]),
            "phases": {k: int(v) for k, v in host["phases"].items()},
            "reduction_workspace_bytes": int(
                priced["ram"]["reduction_workspace_bytes"]
            ),
            "disk_envelope_terms": {
                "spool_peak_bytes_worst_tile": int(
                    priced["disk"]["spool_peak_bytes_worst_tile"]
                ),
                "cleanup_margin_bytes": int(
                    priced["disk"]["cleanup_margin_bytes"]
                ),
            },
        }
    return out


def channels_probe() -> dict:
    """Is ``channels`` a real discriminator, or an unfalsifiable assumption?

    The attested host peak is priced for every plausible channel count with the
    SAME geometry. Only ONE value reproduces the attested bytes, so the
    channel count is a reconstructed hypothesis CONFIRMED by concordance — not
    a silent constant.
    """
    p = _product()
    zext = p["zext"]
    observed = {}
    for channels in (1, 2, 3, 4):
        priced = zext.estimate_external_working_set(
            n_frames=int(ATTESTED["max_n_frames"]),
            height=int(ATTESTED["worst_patch_hw"][0]),
            width=int(ATTESTED["worst_patch_hw"][1]),
            channels=int(channels),
            tile_size=int(ATTESTED["tile_px"]),
            halo_px=int(ATTESTED["halo_px"]),
            column_batch=int(ATTESTED["column_batch"]),
            equalize_rgb=False,
            rejection="winsorized_sigma_clip",
            combine="mean",
        )
        host = priced["host"]
        observed[str(channels)] = {
            "host_peak_bytes": int(host["peak_bytes"]),
            "persistent_result_bytes": int(host["persistent_result_bytes"]),
            "batch_positions": int(host["batch_positions"]),
            "matches_attested_peak": int(host["peak_bytes"])
            == int(ATTESTED["host_peak_bytes"]),
        }
    matching = [
        int(key) for key, block in observed.items()
        if block["matches_attested_peak"]
    ]
    return {
        "channels_used": int(ATTESTED["channels"]),
        "status": "reconstructed hypothesis, validated by concordance",
        "candidates": observed,
        "matching_channel_counts": matching,
        "unique": matching == [int(ATTESTED["channels"])],
        "note": ATTESTED["channels_provenance"],
    }


def vram_probe() -> dict:
    """The GPU tile model refusal (no CUDA allocation, no kernel)."""
    p = _product()
    zsched, zgpu = p["zsched"], p["zgpu"]
    taper = 2 * int(zgpu.GPU_TAPER_HALO_PX)
    return {
        "vram_budget_bytes": int(ATTESTED["vram_budget_bytes"]),
        "taper_px": int(taper),
        "min_tile_px": int(zsched.MIN_TILE_PX),
        "workspace_at_min_tile_bytes": int(
            zgpu.gpu_tile_workspace_bytes(
                int(ATTESTED["max_n_frames"]),
                int(ATTESTED["gpu_min_tile_px"]) + taper,
                int(ATTESTED["gpu_min_tile_px"]) + taper,
                int(ATTESTED["channels"]),
            )
        ),
        "declared_max_tile_px": int(
            zgpu.choose_gpu_tile_size(
                int(ATTESTED["max_n_frames"]), int(ATTESTED["channels"]),
                ATTESTED["worst_patch_hw"],
                int(ATTESTED["vram_budget_bytes"]),
            )
        ),
        "declared_max_tile_px_attested": int(ATTESTED["gpu_declared_max_tile_px"]),
    }


def write_evidence(path, **kwargs) -> dict:
    evidence = replay(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return evidence


def main(argv=None) -> int:
    """CLI contract (documented and tested).

    Return codes
    ------------
    * ``0`` — evidence written with ``status="OK"`` (full replay + probes);
    * ``2`` — evidence written with an ANTICIPATED non-OK status, typically
      ``PLAN_REFUSED`` when the injected PLAN observation cannot derive the
      frozen external contract (e.g. ``--disk-free`` below the disk envelope).
      The evidence is PARTIAL on purpose: the blocks that were never produced
      are listed in ``evidence["partial"]["absent_blocks"]`` and no field is
      fabricated;
    * ``4`` — the checkout guard refused (foreign/absent product code); nothing
      was executed and no evidence was written;
    * ``1`` — unexpected internal error (a bug): the traceback is the evidence.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "M33 P0 — replay the RUN checkpoint refusal with the real product "
            "functions (metadata only: no FITS, no gauge, no kernel). "
            "Exit codes: 0=OK, 2=partial evidence (anticipated non-OK status), "
            "4=checkout guard refusal, 1=unexpected error."
        ),
        epilog=(
            "No PYTHONPATH is needed: the tool pins <checkout>/src itself and "
            "refuses a foreign product tree."
        ),
    )
    parser.add_argument("--out", default=None, help="write the JSON evidence here")
    parser.add_argument(
        "--plan-raw",
        type=int,
        default=None,
        help=(
            "MemAvailable for the PLAN boundary (default: derived inversion of "
            "the 90%% x 15%% contract — RECONSTRUCTED, never attested)"
        ),
    )
    parser.add_argument(
        "--run-raw", type=int, default=None,
        help="MemAvailable for the LAUNCH/RUN boundaries (default: attested)",
    )
    parser.add_argument(
        "--disk-free", type=int, default=None,
        help=(
            "free disk for the PLAN boundary, and for RUN too unless "
            "--run-disk-free overrides it"
        ),
    )
    parser.add_argument(
        "--run-disk-free", type=int, default=None,
        help=(
            "free disk for the RUN boundary only (leaves the PLAN observation "
            "untouched): use this to reproduce the R55_REFUSE_DISK witness"
        ),
    )
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args(argv)
    evidence = replay(
        plan_raw_bytes=args.plan_raw,
        run_raw_bytes=args.run_raw,
        disk_free_bytes=args.disk_free,
        run_disk_free_bytes=args.run_disk_free,
        include_launch=not args.no_launch,
    )
    status = str(evidence.get("status") or "UNKNOWN")
    observations = dict(evidence.get("observations_injected") or {})
    # Probes are attached ONLY from their own explicit inputs; nothing is read
    # out of a block that the replay may never have produced.
    evidence["pricing_probe"] = pricing_probe()
    evidence["channels_probe"] = channels_probe()
    evidence["vram_probe"] = vram_probe()
    evidence["selector_sweep"] = selector_sweep(
        raws=[
            int(observations["plan_raw_available_bytes"]),
            int(observations["run_raw_available_bytes"]),
        ]
    )
    run_budget = (9 * int(observations["run_raw_available_bytes"])) // 10
    evidence["frozen_batch_shrink_probe"] = frozen_batch_shrink_probe(
        phase_budget_bytes=int(_product()["zz"]._r55_margin_raw_ceiling(run_budget))
    )
    evidence["frozen_batch_shrink_probe"]["phase_budget_source"] = (
        "derived from the RUN observation with the SAME inversion the RUN "
        "checkpoint applies (not read from a refusal record)"
    )
    if status != "OK":
        evidence["partial"] = {
            "status": status,
            "absent_blocks": [
                block for block in (
                    "pricing_check",
                    "frozen_contract",
                    "plan_decision",
                    "applied_bundle",
                    "launch_admission",
                    "run_facts",
                    "run_checkpoint",
                )
                if block not in evidence
            ],
            "present_blocks": [
                block for block in (
                    "attested", "provenance", "hypotheses",
                    "observations_injected", "geometry", "plan_selection",
                    "pricing_probe", "channels_probe", "vram_probe",
                    "selector_sweep", "frozen_batch_shrink_probe",
                )
                if block in evidence
            ],
            "note": (
                "the replay stopped before producing the missing blocks; they "
                "are ABSENT, never fabricated. A PLAN_REFUSED evidence means "
                "the injected PLAN observation cannot derive the frozen "
                "external contract at all, so there is no RUN boundary to "
                "evaluate."
            ),
        }
    text = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if status == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
