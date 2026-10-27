"""ZM-ZEGRID-RESILIENT-RUN-RESUME M4 — end-to-end kill/resume proof harness.

Proves plan §6.4/§6.6 on a real durable store, a real M1 registry and the REAL
per-cell feathered assembly, with the REWORK-1 requirement that the gauge and
the geometry are RESTORED from durable artifacts (never re-derived):

    run -> gauge+geometry artifacts -> K cell commits -> SIGKILL (new process)
        -> resume (restore gauge+geometry, no gauge/layout execution)
        -> byte-identical product, every valid cell executed exactly once.

The producer is a deterministic synthetic plane generator (no FITS/GPU); the
store, registry validation, gauge/geometry artifacts, sparse plan and feathered
assembly are the product code. GPU/FITS/Windows are NOT_RUN, explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.canonical_streaming import FixedNormalization  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import run_artifacts as zart  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

H = W = 32
CHANNELS = 3
NX = NY = 2
CORE_H = CORE_W = 16
HALO = 4
FRAME_IDS = ("frame_0001.fits", "frame_0002.fits")
CELL_IDS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")

DTYPES = {
    "planes": {
        "core.science_core": ["<f4"],
        "core.estimator_weight_sum_core": ["<f4"],
        "core.support_w1_core": ["<f4"],
        "core.support_w2_core": ["<f4"],
        "core.n_eff_support_core": ["<f4"],
        "core.valid_mask_core": ["|b1"],
        "core.surviving_sample_count_core": ["<i4"],
        "patch.science": ["<f4"],
        "patch.valid_mask": ["|b1"],
        "patch.surviving": ["<i4"],
    }
}


def log(path: Path, **event) -> None:
    event["pid"] = os.getpid()
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ensure_inputs(input_dir: Path) -> list[dict]:
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, name in enumerate(FRAME_IDS):
        path = input_dir / name
        if not path.exists():
            path.write_bytes((f"m4-frame-{index}-" * 512).encode("utf-8"))
        sha, nbytes = zrs.content_sha256_file(path)
        rows.append(
            {
                "frame_id": name,
                "content_sha256": sha,
                "nbytes": int(nbytes),
                "order": index,
                "role": "light",
            }
        )
    return rows


def input_paths(input_dir: Path) -> dict[str, str]:
    return {name: str(input_dir / name) for name in FRAME_IDS}


def synthetic_gauge() -> FixedNormalization:
    coefficients = np.zeros((len(FRAME_IDS), CHANNELS, 2), dtype=np.float64)
    coefficients[:, :, 0] = 1.0
    coefficients[1, :, 1] = 0.125
    return FixedNormalization(
        reference_index=0,
        coefficients=coefficients,
        norm_active=np.array([True, True]),
        weights=np.array([1.0, 0.5], dtype=np.float64),
        weight_active=np.array([True, True]),
        exclusions=(),
    )


def gauge_sha(gauge=None) -> str:
    gauge = gauge or synthetic_gauge()
    return zart.gauge_fingerprint(
        gauge=gauge, mode="per_cell", frame_ids=FRAME_IDS,
        reference_frame_id=FRAME_IDS[0],
    )


def cell_geometry(index):
    row, col = divmod(index, NX)
    cy0, cx0 = row * CORE_H, col * CORE_W
    cy1, cx1 = cy0 + CORE_H, cx0 + CORE_W
    py0, px0 = max(0, cy0 - HALO), max(0, cx0 - HALO)
    py1, px1 = min(H, cy1 + HALO), min(W, cx1 + HALO)
    return (py0, py1, px0, px1), (cy0 - py0, cy1 - py0, cx0 - px0, cx1 - px0)


def synthetic_geometry():
    canvas = zg.GlobalCanvas(
        canvas_id="m4-etoe", wcs_header="", width=W, height=H,
        resolution_deg=0.001,
    )
    layout = zg.build_layout(canvas, NX, NY)
    layout_dict = {
        "nx": NX,
        "ny": NY,
        "cell_count": NX * NY,
        "layout_source": "harness_frozen",
        "layout_origin": "frozen",
        "predicted_bound_bytes": 0,
        "max_patch_area": 0,
        "cells": [
            {
                "cell_id": cid,
                "core": [
                    (index % NX) * CORE_W,
                    (index // NX) * CORE_H,
                    (index % NX) * CORE_W + CORE_W,
                    (index // NX) * CORE_H + CORE_H,
                ],
            }
            for index, cid in enumerate(CELL_IDS)
        ],
        "cpu_streaming": {"selected_tile_size": CORE_H},
        "floors": {},
    }
    cell_ctxs = []
    for index, cid in enumerate(CELL_IDS):
        row, col = divmod(index, NX)
        (py0, py1, px0, px1), (sy0, sy1, sx0, sx1) = cell_geometry(index)
        core = zg.GlobalBounds(col * CORE_W, row * CORE_H,
                               col * CORE_W + CORE_W, row * CORE_H + CORE_H)
        cell = zg.ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col, core)
        patch = zg.ProcessingPatch(
            cell_id=cid,
            patch=zg.GlobalBounds(px0, py0, px1, py1),
            core_slice=zg.PatchBounds(sx0, sy0, sx1, sy1),
            halo_px=HALO,
            patch_wcs_header="",
        )
        mem = zg.CellMembership(
            cell_id=cid, core_ids=tuple(FRAME_IDS), patch_ids=tuple(FRAME_IDS),
        )
        cell_ctxs.append((row, col, cell, patch, mem))
    return canvas, layout_dict, cell_ctxs


def make_identity(run_id: str, rows: list[dict]) -> dict:
    canvas, layout, cell_ctxs = synthetic_geometry()
    fingerprints = zart.geometry_fingerprints(
        canvas=canvas, layout=layout, cell_ctxs=cell_ctxs, halo_px=HALO
    )
    return zrs.build_run_identity(
        run_id=run_id,
        inputs={"frames": rows},
        canvas={
            "shape": [H, W],
            "wcs_sha256": fingerprints["canvas_wcs_sha256"],
            "geometry_sha256": fingerprints["canvas_geometry_sha256"],
        },
        grid={
            "nx": NX,
            "ny": NY,
            "tile_px": CORE_H,
            "geometry_sha256": fingerprints["grid_geometry_sha256"],
            "cell_ids_sha256": fingerprints["grid_cell_ids_sha256"],
        },
        halo={"halo_px": HALO, "provenance": {"effective_px": HALO}},
        memberships={
            "sha256": fingerprints["memberships_sha256"],
            "n_cells": fingerprints["memberships_n_cells"],
        },
        gauge={
            "mode": "per_cell",
            "sha256": gauge_sha(),
            "reference_frame_id": FRAME_IDS[0],
            "reference_frame_ids": list(FRAME_IDS),
            "n_exclusions": 0,
        },
        science={
            "sha256": "3" * 64,
            "contract_version": "1",
            "params": {"reject": "sigma", "sigma": 3.0},
        },
        dtypes=DTYPES,
        backend={"backend": "cpu", "tile_px": CORE_H, "parity_qualified": False},
        execution_context={
            "available_ram_bytes": 8 * 2 ** 20,
            "workers": 2,
            "prefetch_units": 0,
            "host": "m4-harness",
            "note": "runtime only",
        },
    )


def compute_cell(index: int) -> dict:
    """The deterministic per-cell producer (synthetic planes, real schema)."""
    cid = CELL_IDS[index]
    bounds, core_slice = cell_geometry(index)
    py0, py1, px0, px1 = bounds
    ph, pw = py1 - py0, px1 - px0
    rng = np.random.default_rng(1000 + index)
    science = (
        rng.normal(size=(ph, pw, CHANNELS)).astype(np.float32)
        + np.float32(index)
    )
    valid = np.ones((ph, pw), dtype=bool)
    surviving = np.full((ph, pw), 3, dtype=np.int32)
    y0, y1, x0, x1 = core_slice
    return {
        "cell_id": cid,
        "status": "complete",
        "backend_used": "cpu",
        "n_frames": 2,
        "total_bytes": 1234,
        "core": {
            "science_core": science[y0:y1, x0:x1, :],
            "estimator_weight_sum_core": np.ones((CORE_H, CORE_W), np.float32),
            "support_w1_core": np.ones((CORE_H, CORE_W), np.float32),
            "support_w2_core": np.ones((CORE_H, CORE_W), np.float32),
            "n_eff_support_core": np.full((CORE_H, CORE_W), 3.0, np.float32),
            "valid_mask_core": np.ones((CORE_H, CORE_W), bool),
            "surviving_sample_count_core": np.full(
                (CORE_H, CORE_W), 3, np.int32
            ),
        },
        "patch": {
            "bounds": bounds,
            "core_slice": core_slice,
            "science": science,
            "valid_mask": valid,
            "surviving": surviving,
        },
        "record": {
            "cell_id": cid,
            "patch_bounds": [px0, py0, px1, py1],
            "patch_shape_hw": [ph, pw],
            "cell_gauge_mode": "per_cell",
            "backend_used": "cpu",
            "reference_frame_id": FRAME_IDS[0],
            "reference_frame_role": "cell",
            "bookkeeping_reference_frame_id": FRAME_IDS[0],
            "excluded": [],
        },
    }


def product_bytes(payloads: list[dict], out: Path, canvas) -> dict:
    from zemosaic import zemosaic_zegrid_mode as zz

    assembled, info = zz._assemble_per_cell_feathered(
        canvas, NX, NY, {p["cell_id"]: p for p in payloads}, halo_px=HALO,
    )
    digest = hashlib.sha256()
    arrays = []
    for name in sorted(vars(assembled)):
        value = getattr(assembled, name)
        if isinstance(value, np.ndarray):
            arrays.append(name)
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes())
    with open(out, "wb") as handle:
        for name in arrays:
            value = np.ascontiguousarray(getattr(assembled, name))
            handle.write(name.encode("utf-8") + b"\x00")
            handle.write(str(value.dtype).encode("ascii") + b"\x00")
            handle.write(json.dumps(list(value.shape)).encode("ascii") + b"\x00")
            handle.write(value.tobytes())
    return {
        "product_sha256": digest.hexdigest(),
        "arrays": arrays,
        "cell_count": len(payloads),
    }


def new_run(args) -> int:
    rows = ensure_inputs(Path(args.input_dir))
    identity = make_identity(args.run_id, rows)
    store_root = Path(args.store)
    target = store_root
    if store_root.exists():
        has_run = (store_root / "run.json").exists() or any(
            child.is_dir() and (child / "run.json").exists()
            for child in store_root.iterdir()
        )
        if has_run:
            target = store_root / f"run-{args.run_id}"
    store = zrs.RunCellStore.open(
        target,
        identity=identity,
        budget=zrs.DiskBudget(durable_cell_bytes=64 * 2 ** 20),
        session_id=f"harness-{args.run_id}",
    )
    registry = zrr.RunResultRegistry(
        run_id=args.run_id, expected_cell_ids=list(CELL_IDS),
        gauge_mode="per_cell",
    )
    executed = Path(args.executed_log)
    try:
        zres.write_state(
            target,
            run_id=args.run_id,
            expected_cell_ids=CELL_IDS,
            policy={
                "requested": "auto",
                "effective": args.policy,
                "fallback_count": 1 if args.policy == "bounded" else 0,
                "transitions": (
                    [{"from": "dense", "to": "bounded", "phase": "plan_layout",
                      "reason_code": "REFUSE_HOST_RAM", "detail": "harness"}]
                    if args.policy == "bounded" else []
                ),
                "dense_refusal": None,
            },
            gauge_sha256=identity["science_contract"]["gauge"]["sha256"],
            note="m4 harness state",
        )
        # The two expensive phases run ONCE, before any cell, and are durable.
        gauge = synthetic_gauge()
        log(executed, event="phase", phase="gauge_compute", process="new")
        zart.write_gauge_artifact(
            target, run_id=args.run_id, identity=identity, gauge=gauge,
            mode="per_cell", frame_ids=FRAME_IDS,
            reference_frame_id=FRAME_IDS[0],
        )
        canvas, layout, cell_ctxs = synthetic_geometry()
        log(executed, event="phase", phase="layout_compute", process="new")
        zart.write_geometry_artifact(
            target, run_id=args.run_id, identity=identity, canvas=canvas,
            layout=layout, cell_ctxs=cell_ctxs, halo_px=HALO,
        )
        for index, cid in enumerate(CELL_IDS):
            log(executed, event="execute", cell_id=cid, process="new")
            payload = compute_cell(index)
            receipt = registry.accept(
                payload, backend="cpu", plan_generation=0,
                attempt_id=f"p{os.getpid()}:{cid}",
            )
            store.commit_and_release(
                registry, cid, attempt_id=receipt.attempt_id,
                backend="cpu", plan_generation=0,
            )
            log(executed, event="commit", cell_id=cid, process="new")
            if args.kill_after and index + 1 >= args.kill_after:
                os._exit(9)
        payloads = store.ordered_payloads(registry, list(CELL_IDS))
        result = product_bytes(payloads, Path(args.out), canvas)
        result["policy"] = zres.policy_from_snapshot(zres.read_state(target)["policy"])
        result["store_root"] = str(target)
        _state = zres.read_state(target)
        zres.write_state(
            target,
            run_id=_state["run_id"],
            expected_cell_ids=_state["expected_cell_ids"],
            policy=_state["policy"],
            plan_generation=int(_state["plan_generation"]) + 1,
            finalized=True,
            gauge_sha256=_state["gauge_sha256"],
            outputs={"product_sha256": result["product_sha256"]},
            created_unix_ns=_state["created_unix_ns"],
            note=_state["note"],
        )
        Path(args.summary).write_text(json.dumps(result, indent=2) + "\n")
        return 0
    finally:
        registry.dispose()
        store.close()


def resume_run(args) -> int:
    executed = Path(args.executed_log)
    try:
        boot = zres.ResumeBootstrap.open(
            Path(args.store),
            run_id=args.run_id,
            budget=zrs.DiskBudget(durable_cell_bytes=64 * 2 ** 20),
            session_id=f"harness-resume-{os.getpid()}",
        )
    except zres.ResumeSelectionRequired as exc:
        Path(args.out).write_text(json.dumps({
            "status": "selection_required",
            "candidates": exc.candidates,
        }, indent=2))
        return 4
    except zres.ResumeError as exc:
        Path(args.out).write_text(json.dumps({
            "status": "refused",
            "error": f"{type(exc).__name__}: {exc}",
            "reasons": getattr(exc, "reasons", []),
        }, indent=2))
        return 3
    if boot is None:
        Path(args.out).write_text(json.dumps({"status": "no_run"}))
        return 5
    registry = zrr.RunResultRegistry(
        run_id=boot.store.run_id, expected_cell_ids=list(CELL_IDS),
        gauge_mode="per_cell",
    )
    try:
        try:
            rows = ensure_inputs(Path(args.input_dir))
            boot.verify_identity(make_identity(boot.store.run_id, rows))
            boot.verify_inputs(input_paths(Path(args.input_dir)))
            # REWORK-1: the gauge and the geometry come from the STORE. No
            # compute_global_gauge, no adaptive layout search, no membership
            # recomputation. A mismatch is an explicit refusal.
            gauge, frame_ids, reference_frame_id = boot.restore_gauge()
            canvas, layout, cell_ctxs = boot.restore_geometry()
            log(executed, event="phase", phase="gauge_restored", process="resume")
            log(executed, event="phase", phase="geometry_restored", process="resume")
            plan = boot.plan(list(CELL_IDS))
        except zres.ResumeError as exc:
            Path(args.out).write_text(json.dumps({
                "status": "refused",
                "error": f"{type(exc).__name__}: {exc}",
                "reasons": getattr(exc, "reasons", []),
            }, indent=2))
            return 3
        for cid in plan["pending"]:
            log(executed, event="execute", cell_id=cid, process="resume")
            payload = compute_cell(CELL_IDS.index(cid))
            receipt = registry.accept(
                payload, backend="cpu", plan_generation=0,
                attempt_id=f"p{os.getpid()}:{cid}",
            )
            boot.store.commit_and_release(
                registry, cid, attempt_id=receipt.attempt_id,
                backend="cpu", plan_generation=0,
            )
            log(executed, event="commit", cell_id=cid, process="resume")
        payloads = boot.store.ordered_payloads(registry, list(CELL_IDS))
        result = product_bytes(payloads, Path(args.out), canvas)
        result["policy"] = boot.policy
        result["store_root"] = str(boot.run_root)
        result["artifacts"] = {
            "gauge_restored": bool(len(frame_ids) == len(FRAME_IDS)),
            "gauge_reference_frame_id": reference_frame_id,
            "geometry_restored": bool(len(cell_ctxs) == len(CELL_IDS)),
            "geometry_cell_ids": [cell.cell_id for (_r, _c, cell, _p, _m) in cell_ctxs],
        }
        result["plan"] = {
            "pending": plan["pending"],
            "complete": plan["complete"],
            "empty_validated": plan["empty_validated"],
            "invalid_isolated": plan["invalid_isolated"],
            "is_complete": plan["is_complete"],
        }
        _state = zres.read_state(boot.run_root)
        zres.write_state(
            boot.run_root,
            run_id=_state["run_id"],
            expected_cell_ids=_state["expected_cell_ids"],
            policy=boot.policy,
            plan_generation=int(_state["plan_generation"]) + 1,
            finalized=True,
            gauge_sha256=_state["gauge_sha256"],
            outputs={"product_sha256": result["product_sha256"]},
            created_unix_ns=_state["created_unix_ns"],
            note=_state["note"],
        )
        Path(args.summary).write_text(json.dumps(result, indent=2) + "\n")
        return 0
    finally:
        registry.dispose()
        boot.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("new", "resume"), required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--executed-log", required=True)
    parser.add_argument("--run-id", default="run-m4-etoe")
    parser.add_argument("--kill-after", type=int, default=0)
    parser.add_argument("--policy", choices=("dense", "auto", "bounded"),
                        default="dense")
    args = parser.parse_args(argv)
    Path(args.store).mkdir(parents=True, exist_ok=True)
    return new_run(args) if args.mode == "new" else resume_run(args)


if __name__ == "__main__":
    sys.exit(main())
