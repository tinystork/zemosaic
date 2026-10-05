#!/usr/bin/env python3
"""ZM-ZEGRID-R3 — full-layout deterministic executor + canvas assembly runner.

Three modes:

* Orchestrator (default): build manifest/canvas, iterate EVERY Cell of the
  layout in deterministic ROW-MAJOR Cell-ID order, run each Cell ONE AT A TIME
  in a fresh subprocess (worker mode) under the strict memory gate, and persist
  per-cell artifacts (RESUMABLE). Then assemble the final canvas + coverage +
  manifest (Scope B). If a cell is killed/OOM it is reported explicitly and the
  run CONTINUES with the remaining cells.

* Worker (``--cell``): run ONE cell, persist ``minitile_<cell>.npz`` + a
  ``cell_<cell>.json`` status/adequacy/memory record.

* Assemble-only (``--assemble-only``): assemble the already-persisted per-cell
  artifacts into the final canvas + coverage + manifest (resumable).

Fixtures are prepared INCREMENTALLY via the existing
``tools/zegrid_r1/prepare_rgb_fixture.py`` into a DISK dir (never /tmp).

Usage (M106 full layout):
    PYTHONPATH=src .venv/bin/python tools/zegrid_r3/run_executor.py \
        --lights /home/tristan/M106/lights \
        --fixtures /home/tristan/zegrid_r3_m106_fixtures \
        --out /home/tristan/zegrid_r3_m106_outputs \
        --nx 5 --ny 4

Usage (M16 depth witness, non-mosaic single pointing):
    PYTHONPATH=src .venv/bin/python tools/zegrid_r3/run_executor.py \
        --lights /home/tristan/M16/quick \
        --fixtures /home/tristan/zegrid_r3_m16_fixtures \
        --out /home/tristan/zegrid_r3_m16_outputs \
        --nx 4 --ny 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import executor as zx  # noqa: E402
from zemosaic.core.zegrid import execution as zxe  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import mosaic as zm  # noqa: E402
from zemosaic.core.zegrid import science_adapter as zs  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402

R1_PREP = Path(__file__).resolve().parents[1] / "zegrid_r1" / "prepare_rgb_fixture.py"

GATE_WAIT_SECONDS = 20
GATE_MAX_ATTEMPTS = 6


def prepared_path(fixtures: Path, key: str) -> str:
    return str(fixtures / (Path(key).stem + "_rgb.fits"))


def geometry_cache_path(out: Path) -> Path:
    return out / "geometry_cache.json"


def save_geometry_cache(out: Path, frames, canvas) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "frames": [
            {
                "logical_path": f.frame_id.logical_path,
                "source_path": f.source_path,
                "shape_hw": list(f.shape_hw),
                "wcs_header": f.wcs_header,
                "header_sha256": f.header_sha256,
                "instrument": f.instrument,
            }
            for f in frames
        ],
        "canvas": {
            "canvas_id": canvas.canvas_id,
            "wcs_header": canvas.wcs_header,
            "width": canvas.width,
            "height": canvas.height,
            "resolution_deg": canvas.resolution_deg,
        },
    }
    geometry_cache_path(out).write_text(json.dumps(data) + "\n")


def load_geometry_cache(out: Path):
    cache = geometry_cache_path(out)
    if not cache.exists():
        raise FileNotFoundError(
            f"geometry cache missing: {cache}; run the orchestrator first"
        )
    data = json.loads(cache.read_text())
    frames = [
        zg.FrameDescriptor(
            frame_id=zg.FrameId(r["logical_path"]),
            source_path=r["source_path"],
            shape_hw=tuple(r["shape_hw"]),
            wcs_header=r["wcs_header"],
            header_sha256=r["header_sha256"],
            instrument=r["instrument"],
        )
        for r in data["frames"]
    ]
    c = data["canvas"]
    canvas = zg.GlobalCanvas(
        canvas_id=c["canvas_id"],
        wcs_header=c["wcs_header"],
        width=c["width"],
        height=c["height"],
        resolution_deg=c["resolution_deg"],
    )
    return frames, canvas


def ensure_fixtures(fixtures: Path, lights: Path, keys: list[str]) -> None:
    missing = [k for k in keys if not Path(prepared_path(fixtures, k)).exists()]
    if not missing:
        return
    names = ",".join(Path(k).name for k in missing)
    print(f"[fixtures] preparing {len(missing)} missing: {names[:120]}...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(R1_PREP),
            "--input",
            str(lights),
            "--output",
            str(fixtures),
            "--frames",
            names,
        ],
        check=True,
    )


def _cell_json(out: Path, cell_id: str) -> Path:
    return out / f"cell_{cell_id}.json"


def _worker(args, cell_id, row, col) -> int:
    """Run ONE cell's full pipeline and persist npz + status JSON."""
    out = Path(args.out)
    frames, canvas = load_geometry_cache(out)
    config = zx.ExecutorConfig()

    cell, patch, mem = zsw.build_cell_context(frames, canvas, row, col, args.nx, args.ny)
    by_id = {f.frame_id.logical_path: f for f in frames}
    prepared_paths = {k: prepared_path(Path(args.fixtures), k) for k in mem.patch_ids}

    if len(mem.patch_ids) == 0:
        rec = zx.CellExecutionRecord(
            cell_id=cell_id, row=row, col=col, status=zx.STATUS_EMPTY,
            n_patch_contributors=0, n_core_contributors=0,
            core=[cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1],
            patch=[patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1],
            core_ids=list(mem.core_ids), patch_ids=list(mem.patch_ids),
            empty_roi_frames=[],
            message="empty cell: no patch contributors (no invented data)",
        )
        _cell_json(out, cell_id).write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
        return 0

    gate = zx.check_executor_gate(len(mem.patch_ids))
    print(
        f"[cell {cell_id}] N={len(mem.patch_ids)} core={len(mem.core_ids)} "
        f"available={gate.available/2**30:.2f}GiB required={gate.required/2**30:.2f}GiB "
        f"ok={gate.ok}",
        flush=True,
    )
    if not gate.ok:
        rec = zx.CellExecutionRecord(
            cell_id=cell_id, row=row, col=col, status=zx.STATUS_BLOCKED_MEMORY,
            n_patch_contributors=len(mem.patch_ids),
            n_core_contributors=len(mem.core_ids),
            core=[cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1],
            patch=[patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1],
            core_ids=list(mem.core_ids), patch_ids=list(mem.patch_ids),
            empty_roi_frames=[],
            message=(f"memory gate unsatisfied: available={gate.available/2**30:.2f}GiB "
                     f"< required={gate.required/2**30:.2f}GiB"),
        )
        _cell_json(out, cell_id).write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
        return 2

    tracker = zxe.SectionReadTracker()
    res = zsw.run_cell_stack(
        frames, canvas, prepared_paths, row, col, config.science_config(),
        nx=args.nx, ny=args.ny, tracker=tracker,
    )
    from zemosaic.core.zegrid import assembly as za

    cores = za.crop_all_planes_to_core(res.minitile)

    npz_path = out / f"minitile_{cell_id}.npz"
    np.savez_compressed(
        npz_path,
        science_core=cores["science_core"],
        surviving_sample_count_core=cores["surviving_sample_count_core"],
        n_eff_support_core=cores["n_eff_support_core"],
        support_w1_core=cores["support_w1_core"],
        support_w2_core=cores["support_w2_core"],
        valid_mask_core=cores["valid_mask_core"],
    )

    rec = zx.CellExecutionRecord(
        cell_id=cell_id, row=row, col=col, status=zx.STATUS_COMPLETE,
        n_patch_contributors=res.n_contributors,
        n_core_contributors=res.n_core_contributors,
        core=[cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1],
        patch=[patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1],
        core_ids=list(mem.core_ids), patch_ids=list(mem.patch_ids),
        empty_roi_frames=[],
        adequacy=res.adequacy.to_dict(),
        reference_frame_id=res.reference_frame_id,
        excluded=[list(e) for e in res.excluded],
        section_read_count=len(res.section_reads),
        sum_local_px=sum(r.n_pixels_read for r in res.section_reads),
        peak_rss_kib=res.peak_rss_kib,
        mem_available_before=res.mem_available_before,
        mem_available_after=res.mem_available_after,
    )
    _cell_json(out, cell_id).write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
    print(
        f"[cell {cell_id}] done: effective={res.adequacy.effective_contributor_count} "
        f"max_surv={res.adequacy.max_surviving_sample_count} "
        f"valid_frac={res.adequacy.valid_fraction:.4f} "
        f"peak_rss={res.peak_rss_kib/1024:.1f}MiB "
        f"excluded={len(res.excluded)}",
        flush=True,
    )
    return 0


def _run_cell_subprocess(args, cell_id, row, col) -> dict:
    """Run one cell in a fresh subprocess; return its status record."""
    out = Path(args.out)
    json_path = _cell_json(out, cell_id)
    # Resumable: a persisted COMPLETE/EMPTY record is authoritative.
    if json_path.exists():
        prev = json.loads(json_path.read_text())
        if prev.get("status") in (zx.STATUS_COMPLETE, zx.STATUS_EMPTY):
            return prev

    cmd = [
        sys.executable, str(__file__),
        "--cell", cell_id, "--row", str(row), "--col", str(col),
        "--lights", str(args.lights), "--fixtures", str(args.fixtures),
        "--out", str(args.out), "--nx", str(args.nx), "--ny", str(args.ny),
    ]
    proc = subprocess.run(cmd, cwd=SRC.parent)
    if json_path.exists():
        rec = json.loads(json_path.read_text())
        if proc.returncode != 0 and rec.get("status") == zx.STATUS_COMPLETE:
            pass
        if proc.returncode != 0 and rec.get("status") not in (
            zx.STATUS_COMPLETE, zx.STATUS_EMPTY, zx.STATUS_BLOCKED_MEMORY
        ):
            # Killed / OOM / crash — record explicitly, continue.
            killed = proc.returncode == -9 or proc.returncode == 137
            rec["status"] = zx.STATUS_KILLED if killed else zx.STATUS_INCOMPLETE
            rec["message"] = f"subprocess returncode={proc.returncode} (OOM/kill)" if killed else \
                f"subprocess returncode={proc.returncode}"
            json_path.write_text(json.dumps(rec, indent=2) + "\n")
        return rec
    # No record written at all -> hard failure.
    return {
        "cell_id": cell_id, "row": row, "col": col,
        "status": zx.STATUS_INCOMPLETE,
        "message": f"subprocess returncode={proc.returncode} (no record written)",
    }


def _assemble(args) -> dict:
    """Assemble persisted per-cell core slices into the final canvas."""
    out = Path(args.out)
    frames, canvas = load_geometry_cache(out)
    layout = zg.build_layout(canvas, args.nx, args.ny)

    cores: dict[str, dict | None] = {}
    cell_records: dict[str, dict] = {}
    for row, col, cid in zx.cell_order(canvas, args.nx, args.ny):
        jp = _cell_json(out, cid)
        if not jp.exists():
            cell_records[cid] = {"cell_id": cid, "status": "missing"}
            cores[cid] = None
            continue
        rec = json.loads(jp.read_text())
        cell_records[cid] = rec
        if rec.get("status") != zx.STATUS_COMPLETE:
            cores[cid] = None
            continue
        npz_path = out / f"minitile_{cid}.npz"
        if not npz_path.exists():
            cores[cid] = None
            continue
        z = np.load(npz_path)
        cores[cid] = {
            "science_core": z["science_core"],
            "surviving_sample_count_core": z["surviving_sample_count_core"],
            "n_eff_support_core": z["n_eff_support_core"],
            "support_w1_core": z["support_w1_core"],
            "support_w2_core": z["support_w2_core"],
            "valid_mask_core": z["valid_mask_core"],
        }

    assembled = zm.assemble_canvas(canvas, args.nx, args.ny, cores)

    # Ownership assertion (disjoint + exhaustive).
    try:
        owned = zm.verify_disjoint_exhaustive(assembled)
        ownership_ok = True
        ownership_note = "all pixels owned exactly once"
    except AssertionError as exc:
        ownership_ok = False
        ownership_note = str(exc)

    # --- write final science FITS (float32) ---
    from astropy.io import fits

    sci_chw = np.ascontiguousarray(np.moveaxis(assembled.science, -1, 0).astype(np.float32))
    sci_hdr = fits.Header()
    sci_hdr.update(canvas.wcs().to_header(relax=True))
    sci_hdr["ZEGRIDAX"] = ("CHW", "channels-first float32 RGB; WCS on axes 1-2")
    sci_hdr["MISSION"] = "ZM-ZEGRID-R3"
    sci_path = out / "final_science.fits"
    fits.PrimaryHDU(sci_chw, header=sci_hdr).writeto(sci_path, overwrite=True)

    # --- write coverage/support FITS ---
    cov_hdr = fits.Header()
    cov_hdr.update(canvas.wcs().to_header(relax=True))
    cov_hdr["MISSION"] = "ZM-ZEGRID-R3"
    cov_hdu = fits.PrimaryHDU(assembled.stack_depth.astype(np.int32), header=cov_hdr)
    cov_hdu.header["EXTNAME"] = "STACK_DEPTH"
    hdul = fits.HDUList([
        cov_hdu,
        fits.ImageHDU(assembled.n_eff_support.astype(np.float64), name="N_EFF_SUPPORT"),
        fits.ImageHDU(assembled.support_w1.astype(np.float64), name="SUPPORT_W1"),
        fits.ImageHDU(assembled.support_w2.astype(np.float64), name="SUPPORT_W2"),
        fits.ImageHDU(assembled.valid_fraction.astype(np.float32), name="VALID_FRACTION"),
        fits.ImageHDU(assembled.hole_mask.astype(np.uint8), name="HOLE_MASK"),
    ])
    cov_path = out / "final_coverage.fits"
    hdul.writeto(cov_path, overwrite=True)

    # --- manifest ---
    manifest = {
        "mission": "ZM-ZEGRID-R3",
        "canvas": {
            "canvas_id": canvas.canvas_id,
            "width": canvas.width,
            "height": canvas.height,
            "resolution_deg": canvas.resolution_deg,
            "wcs_header": canvas.wcs_header,
        },
        "layout": {"nx": args.nx, "ny": args.ny, "halo_px": 8},
        "executor_config": zx.ExecutorConfig().to_dict(),
        "ownership": {"exact_one_owner": ownership_ok, "note": ownership_note},
        "holes": {
            "hole_pixels": assembled.hole_pixels,
            "coverage_pixels": assembled.coverage_pixels,
            "incomplete_cells": assembled.incomplete_cells,
        },
        "cells": cell_records,
        "peak_rss_max_kib": max(
            (r.get("peak_rss_kib", 0) for r in cell_records.values()),
            default=0,
        ),
        "science_sha256": hashlib.sha256(sci_path.read_bytes()).hexdigest(),
        "coverage_sha256": hashlib.sha256(cov_path.read_bytes()).hexdigest(),
    }
    (out / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[assemble] holes={assembled.hole_pixels} coverage={assembled.coverage_pixels} "
        f"ownership_exact={ownership_ok} incomplete={assembled.incomplete_cells}",
        flush=True,
    )
    return manifest


def _orchestrator(args) -> int:
    out = Path(args.out)
    frames, _ = zg.read_manifest(args.lights)
    canvas = zg.build_canvas(frames)
    save_geometry_cache(out, frames, canvas)
    print(f"[executor] canvas {canvas.width}x{canvas.height} frames={len(frames)}", flush=True)

    order = zx.cell_order(canvas, args.nx, args.ny)
    print(f"[executor] {len(order)} cells (row-major):", flush=True)
    for row, col, cid in order:
        _c, _p, mem = zsw.build_cell_context(frames, canvas, row, col, args.nx, args.ny)
        print(f"  {cid} N={len(mem.patch_ids)} core={len(mem.core_ids)}", flush=True)

    for row, col, cid in order:
        _c, _p, mem = zsw.build_cell_context(frames, canvas, row, col, args.nx, args.ny)
        n = len(mem.patch_ids)
        # Memory gate with bounded wait before each cell.
        ok = False
        gate = None
        for attempt in range(GATE_MAX_ATTEMPTS):
            gate = zx.check_executor_gate(n)
            if gate.ok:
                ok = True
                break
            if attempt < GATE_MAX_ATTEMPTS - 1:
                print(
                    f"[gate] {cid} N={n} available={gate.available/2**30:.2f}GiB < "
                    f"{gate.required/2**30:.2f}GiB; waiting {GATE_WAIT_SECONDS}s "
                    f"(attempt {attempt+1}/{GATE_MAX_ATTEMPTS})",
                    flush=True,
                )
                time.sleep(GATE_WAIT_SECONDS)
        if not ok:
            rec = zx.CellExecutionRecord(
                cell_id=cid, row=row, col=col, status=zx.STATUS_BLOCKED_MEMORY,
                n_patch_contributors=n, n_core_contributors=len(mem.core_ids),
                core=[_c.core.x0, _c.core.y0, _c.core.x1, _c.core.y1],
                patch=[], core_ids=list(mem.core_ids), patch_ids=list(mem.patch_ids),
                empty_roi_frames=[],
                message=(f"memory gate unsatisfied after {GATE_MAX_ATTEMPTS} attempts "
                         f"(available={gate.available/2**30:.2f}GiB)"),
            )
            _cell_json(out, cid).write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
            continue

        ensure_fixtures(Path(args.fixtures), Path(args.lights), list(mem.patch_ids))
        rec = _run_cell_subprocess(args, cid, row, col)
        eff = (rec.get("adequacy") or {}).get("effective_contributor_count")
        print(
            f"[cell {cid}] status={rec.get('status')} "
            f"eff={eff} "
            f"rss={rec.get('peak_rss_kib', 0)/1024:.1f}MiB",
            flush=True,
        )

    manifest = _assemble(args)
    print(f"[executor] DONE manifest={out / 'assembly_manifest.json'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lights", required=True, type=Path)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--cell", type=str, default=None)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--col", type=int, default=None)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    if args.cell is not None:
        return _worker(args, args.cell, args.row, args.col)
    if args.assemble_only:
        _assemble(args)
        return 0
    return _orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
