#!/usr/bin/env python3
"""ZM-ZEGRID-R2 — multi-frame witness sweep + adjacent-Cell continuity.

Two modes:

* Orchestrator (no ``--cell``): build manifest/canvas, order candidates by
  ascending patch-contributor count (tie-break cell id), sweep up to 6 cells
  ONE AT A TIME under a strict memory gate, and stop at the first cell with
  ``effective_contributor_count >= 3`` (genuine multi-frame). Each cell runs in
  a fresh subprocess (``--cell`` mode) for a clean peak-RSS measurement and
  memory isolation. Then run ONE adjacent neighbour and persist the seam
  diagnostic (no blend).

* Worker (``--cell``): run ONE cell's full R1 local pipeline, measure peak RSS
  (``RUSAGE_SELF``) + memory before/after, compute witness adequacy, and persist
  ``minitile_<cell>.npz`` + ``minitile_<cell>.json``.

Fixtures are prepared INCREMENTALLY (only a candidate's missing patch
contributors) via the EXISTING ``tools/zegrid_r1/prepare_rgb_fixture.py`` into
``/home/tristan/zegrid_r2_fixtures`` (disk-backed, outside git). ``/tmp`` is
NEVER used for fixtures. Fixture decode I/O is separate and not counted as local
execution (R0 §P option (a)).

Memory policy (strict): require ``>= 1.2 GiB`` available before EVERY cell run
(``>= 1.6 GiB`` when ``N > 28``). If the gate cannot be satisfied after a short
bounded wait, the sweep aborts gracefully with an explicit BLOCKED table — never
OOM.

Usage (orchestrator):
    PYTHONPATH=src .venv/bin/python tools/zegrid_r2/sweep_multiframe.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import science_adapter as zs  # noqa: E402
from zemosaic.core.zegrid import seam as zsm  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402

LIGHTS = Path("/home/tristan/M106/lights")
FIXTURES = Path("/home/tristan/zegrid_r2_fixtures")
OUT = Path("/home/tristan/zegrid_r2_outputs")
R1_PREP = Path(__file__).resolve().parents[1] / "zegrid_r1" / "prepare_rgb_fixture.py"

# Memory-gate wait policy (bounded).
GATE_WAIT_SECONDS = 20
GATE_MAX_ATTEMPTS = 6


def prepared_path(key: str) -> str:
    return str(FIXTURES / (Path(key).stem + "_rgb.fits"))


def geometry_cache_path() -> Path:
    return OUT / "geometry_cache.json"


def save_geometry_cache(frames, canvas) -> None:
    """Serialize manifest + canvas so worker subprocesses skip re-derivation."""
    OUT.mkdir(parents=True, exist_ok=True)
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
    geometry_cache_path().write_text(json.dumps(data) + "\n")


def load_geometry_cache():
    """Reconstruct (frames, canvas) from the serialized geometry cache."""
    data = json.loads(geometry_cache_path().read_text())
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


def ensure_fixtures(keys: list[str]) -> None:
    """Prepare any missing patch-contributor RGB fixtures via the R1 tool."""
    missing = [k for k in keys if not Path(prepared_path(k)).exists()]
    if not missing:
        return
    names = ",".join(Path(k).name for k in missing)
    print(f"[fixtures] preparing {len(missing)} missing: {names[:120]}...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(R1_PREP),
            "--input",
            str(LIGHTS),
            "--output",
            str(FIXTURES),
            "--frames",
            names,
        ],
        check=True,
    )


def run_cell_worker(cell_id: str, row: int, col: int) -> dict:
    """Run ONE cell in a fresh subprocess and return its persisted result JSON."""
    out_dir = str(OUT)
    subprocess.run(
        [
            sys.executable,
            str(__file__),
            "--cell",
            cell_id,
            "--row",
            str(row),
            "--col",
            str(col),
        ],
        check=True,
        cwd=SRC.parent,  # repo root so ``zemosaic`` resolves
    )
    return json.loads((OUT / f"minitile_{cell_id}.json").read_text())


def _worker(cell_id: str, row: int, col: int) -> int:
    """Worker mode: run one cell, persist NPZ + JSON, report peak RSS + adequacy."""
    import resource

    frames, canvas = load_geometry_cache()
    cell, patch, mem = zsw.build_cell_context(frames, canvas, row, col)
    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    prepared_paths = {k: prepared_path(k) for k in mem.patch_ids}

    gate = zsw.check_memory_gate(len(mem.patch_ids))
    print(
        f"[cell {cell_id}] N={len(mem.patch_ids)} core={len(mem.core_ids)} "
        f"available={gate.available/2**30:.2f}GiB required={gate.required/2**30:.2f}GiB "
        f"ok={gate.ok}",
        flush=True,
    )
    if not gate.ok:
        # Abort gracefully (explicit BLOCKED), never OOM.
        rec = {
            "cell_id": cell_id,
            "row": row,
            "col": col,
            "n_contributors": len(mem.patch_ids),
            "n_core_contributors": len(mem.core_ids),
            "blocked_memory": True,
            "available_bytes": gate.available,
            "required_bytes": gate.required,
            "core": [cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1],
        }
        (OUT / f"minitile_{cell_id}.json").write_text(json.dumps(rec, indent=2) + "\n")
        return 2

    cfg = zs.MiniTileScienceConfig()
    tracker = zx_section_tracker()
    res = zsw.run_cell_stack(
        frames, canvas, prepared_paths, row, col, cfg, tracker=tracker
    )
    mt = res.minitile
    cores = za_crop(res)

    OUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUT / f"minitile_{cell_id}.npz"
    np.savez_compressed(
        npz_path,
        science=mt.science,
        estimator_weight_sum=mt.estimator_weight_sum,
        support_w1=mt.support_w1,
        support_w2=mt.support_w2,
        n_eff_support=mt.n_eff_support,
        valid_mask=mt.valid_mask,
        surviving_sample_count=mt.surviving_sample_count,
        science_core=cores["science_core"],
        estimator_weight_sum_core=cores["estimator_weight_sum_core"],
        support_w1_core=cores["support_w1_core"],
        support_w2_core=cores["support_w2_core"],
        n_eff_support_core=cores["n_eff_support_core"],
        valid_mask_core=cores["valid_mask_core"],
        surviving_sample_count_core=cores["surviving_sample_count_core"],
    )

    adequacy = res.adequacy
    provenance = {
        "mission": "ZM-ZEGRID-R2",
        "cell_id": cell_id,
        "row": row,
        "col": col,
        "canvas": {
            "width": canvas.width,
            "height": canvas.height,
            "resolution_deg": canvas.resolution_deg,
            "canvas_id": canvas.canvas_id,
        },
        "layout": {"nx": zsw.NX, "ny": zsw.NY, "halo_px": zsw.HALO_PX},
        "core": [cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1],
        "patch": [patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1],
        "core_slice": list(mt.core_slice),
        "patch_shape_hw": list(mt.patch_shape_hw),
        "science_config": {
            "normalization": cfg.normalization,
            "weighting": cfg.weighting,
            "rejection": cfg.rejection,
            "combine": cfg.combine,
            "backend": cfg.backend,
            "taper": cfg.taper,
            "taper_px": cfg.taper_px,
            "taper_floor": cfg.taper_floor,
            "reference_index": cfg.reference_index,
        },
        "frame_order": list(mt.frame_order),
        "reference_frame_id": mt.reference_frame_id,
        "excluded_frames": list(mt.excluded),
        "core_contributors": list(mem.core_ids),
        "patch_contributors": list(mem.patch_ids),
        "adequacy": adequacy.to_dict(),
        "section_reads": [
            {
                "path": Path(r.path).name,
                "x0": r.source_bounds.x0,
                "y0": r.source_bounds.y0,
                "x1": r.source_bounds.x1,
                "y1": r.source_bounds.y1,
                "n_pixels_read": r.n_pixels_read,
                "full_frame_pixels": r.full_frame_pixels,
            }
            for r in res.section_reads
        ],
        "sum_local_px": sum(r.n_pixels_read for r in res.section_reads),
        "peak_rss_kib": res.peak_rss_kib,
        "peak_rss_delta_kib": res.peak_rss_delta_kib,
        "mem_available_before": res.mem_available_before,
        "mem_available_after": res.mem_available_after,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    (OUT / f"minitile_{cell_id}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"[cell {cell_id}] done: effective={adequacy.effective_contributor_count} "
        f"max_surv={adequacy.max_surviving_sample_count} "
        f"valid_frac={adequacy.valid_fraction:.4f} "
        f"peak_rss={res.peak_rss_kib/1024:.1f}MiB "
        f"excluded={len(mt.excluded)}",
        flush=True,
    )
    return 0


def _import_runtime():
    """Import R1 modules once (used by worker); cached at module import."""
    from zemosaic.core.zegrid import assembly as za
    from zemosaic.core.zegrid import execution as zx

    return za, zx


_ZA, _ZX = _import_runtime()


def zx_section_tracker():
    return _ZX.SectionReadTracker()


def za_crop(res):
    return _ZA.crop_all_planes_to_core(res.minitile)


def _orchestrator() -> int:
    frames, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    save_geometry_cache(frames, canvas)

    candidates = zsw.candidate_order(frames, canvas)
    print(f"[sweep] {len(candidates)} candidates (ascending N):")
    for c in candidates:
        print(f"  {c.cell_id} N={c.n_contributors} core={c.n_core_contributors}")

    summary = {
        "mission": "ZM-ZEGRID-R2",
        "canvas": {"width": canvas.width, "height": canvas.height},
        "candidate_order": [c.cell_id for c in candidates],
        "cells_tried": [],
        "chosen_cell": None,
        "blocked": False,
    }

    chosen: dict | None = None
    for cand in candidates[: zsw.DEFAULT_MAX_CELLS]:
        cell, _patch, mem = zsw.build_cell_context(frames, canvas, cand.row, cand.col)
        # Memory gate with bounded wait/retry.
        ok = False
        gate = None
        for attempt in range(GATE_MAX_ATTEMPTS):
            gate = zsw.check_memory_gate(cand.n_contributors)
            if gate.ok:
                ok = True
                break
            if attempt < GATE_MAX_ATTEMPTS - 1:
                print(
                    f"[gate] {cand.cell_id} N={cand.n_contributors} "
                    f"available={gate.available/2**30:.2f}GiB < "
                    f"{gate.required/2**30:.2f}GiB; waiting {GATE_WAIT_SECONDS}s "
                    f"(attempt {attempt+1}/{GATE_MAX_ATTEMPTS})",
                    flush=True,
                )
                time.sleep(GATE_WAIT_SECONDS)
        if not ok:
            print(
                f"[gate] {cand.cell_id} BLOCKED on memory after "
                f"{GATE_MAX_ATTEMPTS} attempts (available={gate.available/2**30:.2f}GiB)",
                flush=True,
            )
            summary["blocked"] = True
            summary["blocked_reason"] = (
                f"memory gate unsatisfied for {cand.cell_id} (N={cand.n_contributors})"
            )
            break

        ensure_fixtures(list(mem.patch_ids))
        rec = run_cell_worker(cand.cell_id, cand.row, cand.col)
        summary["cells_tried"].append(rec)
        if rec.get("blocked_memory"):
            summary["blocked"] = True
            summary["blocked_reason"] = (
                f"worker memory gate failed for {cand.cell_id} (N={cand.n_contributors})"
            )
            break
        if rec["adequacy"]["effective_contributor_count"] >= 3:
            chosen = rec
            summary["chosen_cell"] = cand.cell_id
            print(
                f"[sweep] CHOSEN {cand.cell_id}: effective="
                f"{rec['adequacy']['effective_contributor_count']} >= 3",
                flush=True,
            )
            break

    if chosen is None:
        summary["status"] = "BLOCKED"
        (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("[sweep] no candidate reached effective>=3 within 6 cells / memory cap")
        return 3

    # --- Objective C: one adjacent neighbour + seam diagnostic (no blend) ---
    ch_row, ch_col = chosen["row"], chosen["col"]
    seam_rec = None
    # Pick the LOWEST-N adjacent neighbour (memory-friendly; "as near as memory
    # permits"). Tie-break by direction priority (right, bottom, left, top).
    nbr_choices = []
    for _dir, r, c in zsm.adjacent_neighbours(ch_row, ch_col, zsw.NX, zsw.NY):
        _c, _p, _m = zsw.build_cell_context(frames, canvas, r, c)
        nbr_choices.append((len(_m.patch_ids), _dir, r, c, _c.cell_id))
    nbr_choices.sort(key=lambda t: (t[0], t[1]))
    if nbr_choices:
        n_n, _dir, n_row, n_col, n_id = nbr_choices[0]
        n_cell, _np, n_mem = zsw.build_cell_context(frames, canvas, n_row, n_col)
        gate = zsw.check_memory_gate(len(n_mem.patch_ids))
        if not gate.ok:
            print(
                f"[gate] neighbour {n_cell.cell_id} BLOCKED on memory "
                f"(available={gate.available/2**30:.2f}GiB < {gate.required/2**30:.2f}GiB)",
                flush=True,
            )
            summary["blocked"] = True
            summary["blocked_reason"] = f"neighbour {n_cell.cell_id} memory gate failed"
        else:
            ensure_fixtures(list(n_mem.patch_ids))
            n_rec = run_cell_worker(n_cell.cell_id, n_row, n_col)
            summary["neighbour"] = n_rec["cell_id"]
            # Re-load both MiniTiles and compute the seam residual.
            seam_rec = _compute_and_persist_seam(chosen, n_rec, ch_row, ch_col, n_row, n_col)
            summary["seam"] = seam_rec
    else:
        summary["neighbour"] = None
        summary["seam"] = None

    summary["status"] = "DONE" if not summary["blocked"] else "BLOCKED"
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[sweep] status={summary['status']} summary={OUT / 'sweep_summary.json'}")
    return 0 if not summary["blocked"] else 3


def _compute_and_persist_seam(chosen, n_rec, ch_row, ch_col, n_row, n_col) -> dict:
    """Load the chosen + neighbour MiniTiles and compute the seam diagnostic."""
    frames, canvas = load_geometry_cache()

    def _load_mt(cell_id):
        npz = np.load(OUT / f"minitile_{cell_id}.npz")
        return {
            "science": npz["science"],
            "valid_mask": npz["valid_mask"],
            "n_eff_support": npz["n_eff_support"],
            "surviving_sample_count": npz["surviving_sample_count"],
        }

    ch_cell, ch_patch, _ = zsw.build_cell_context(frames, canvas, ch_row, ch_col)
    nb_cell, nb_patch, _ = zsw.build_cell_context(frames, canvas, n_row, n_col)

    # Rebuild MiniTile-like structs from persisted planes.
    from types import SimpleNamespace

    mt_ch = SimpleNamespace(**{
        k: _load_mt(chosen["cell_id"])[k] for k in
        ("science", "valid_mask", "n_eff_support", "surviving_sample_count")
    })
    mt_nb = SimpleNamespace(**{
        k: _load_mt(n_rec["cell_id"])[k] for k in
        ("science", "valid_mask", "n_eff_support", "surviving_sample_count")
    })

    # Core ownership: disjoint + exhaustive over the whole layout.
    layout = zg.build_layout(canvas, zsw.NX, zsw.NY)
    owned = zsm.assert_core_partition_exact(layout, canvas)
    assert int(owned.min()) == 1 and int(owned.max()) == 1

    seam = zsm.compute_seam_diagnostic(
        canvas, ch_cell, ch_patch, mt_ch, nb_cell, nb_patch, mt_nb,
        nx=zsw.NX, ny=zsw.NY, halo_px=zsm.SEAM_HALF,
    )
    seam["core_partition_exact"] = True
    seam["no_halo_double_count"] = True  # MiniTile = full patch; only core_slice placed
    (OUT / f"seam_{chosen['cell_id']}_{n_rec['cell_id']}.json").write_text(
        json.dumps(seam, indent=2) + "\n"
    )
    return seam


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", type=str, default=None)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--col", type=int, default=None)
    args = ap.parse_args()

    if args.cell is not None:
        return _worker(args.cell, args.row, args.col)
    return _orchestrator()


if __name__ == "__main__":
    raise SystemExit(main())
