#!/usr/bin/env python3
"""ZM-ZEGRID-R6 — streaming cell runner + in-memory parity + peak RSS (opt-in).

Runs ONE real M106 cell (default r0001c0002, N=66) through BOTH:
  * the in-memory R1/R2/R3 path (``sweep.run_cell_stack``), and
  * the streaming path (``streaming.run_cell_streaming``, file-backed provider),
records peak RSS for each (VmHWM), and compares every MiniTile plane BIT-EXACT.

Modes (fresh subprocess per heavy run -> clean peak RSS, one at a time):
  * ``--mode build``    : build the aligned disk cache (resumable) + report size.
  * ``--mode inmem``    : in-memory run; persist minitile npz + record; print peak.
  * ``--mode stream``   : streaming run (tile); persist minitile npz + record.
  * ``--mode parity``   : orchestrator: gate -> inmem -> stream(s) -> bit-exact
                          comparison -> parity report JSON (resumable per artifact).

The aligned cache is a NEW disk artifact (never /tmp). Memory gate is checked
before each heavy run; one heavy run at a time.

Usage (M106 N=66 cell r0001c0002):
    PYTHONPATH=src .venv/bin/python tools/zegrid_r6/run_streaming_cell.py \
        --mode parity --lights /home/tristan/M106/lights \
        --fixtures /home/tristan/zegrid_r2_fixtures \
        --cache /home/tristan/zegrid_r6_cache_r0001c0002 \
        --out /home/tristan/zegrid_r6_out_r0001c0002 \
        --row 1 --col 2 --tile 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import science_adapter as zs  # noqa: E402
from zemosaic.core.zegrid import streaming as zstream  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402
from zemosaic.core.zegrid.executor import ExecutorConfig  # noqa: E402

R1_PREP = Path(__file__).resolve().parents[1] / "zegrid_r1" / "prepare_rgb_fixture.py"


def prepared_path(fixtures: Path, key: str) -> str:
    return str(fixtures / (Path(key).stem + "_rgb.fits"))


def _ensure_fixtures(fixtures: Path, lights: Path, keys: list[str]) -> None:
    missing = [k for k in keys if not Path(prepared_path(fixtures, k)).exists()]
    if not missing:
        return
    names = ",".join(Path(k).name for k in missing)
    print(f"[fixtures] preparing {len(missing)}: {names[:120]}...", flush=True)
    subprocess.run(
        [sys.executable, str(R1_PREP), "--input", str(lights),
         "--output", str(fixtures), "--frames", names],
        check=True,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_geometry(lights: Path):
    frames, _ = zg.read_manifest(lights)
    canvas = zg.build_canvas(frames)
    return frames, canvas


def _save_minitile(mt, npz_path: Path) -> None:
    np.savez_compressed(
        npz_path,
        science=np.asarray(mt.science),
        estimator_weight_sum=np.asarray(mt.estimator_weight_sum),
        support_w1=np.asarray(mt.support_w1),
        support_w2=np.asarray(mt.support_w2),
        n_eff_support=np.asarray(mt.n_eff_support),
        valid_mask=np.asarray(mt.valid_mask),
        surviving_sample_count=np.asarray(mt.surviving_sample_count),
        reference_frame_id=np.asarray(mt.reference_frame_id or ""),
    )


def _run_inmem(args, frames, canvas) -> dict:
    """In-memory R1/R2/R3 path for one cell (fresh subprocess -> clean VmHWM)."""
    out = Path(args.out)
    cell, patch, mem = zsw.build_cell_context(frames, canvas, args.row, args.col, args.nx, args.ny)
    gate = zsw.check_memory_gate(len(mem.patch_ids))
    print(
        f"[inmem gate] N={len(mem.patch_ids)} available={gate.available/2**30:.2f}GiB "
        f"required={gate.required/2**30:.2f}GiB ok={gate.ok}", flush=True,
    )
    if not gate.ok:
        raise zsw.MemoryInsufficient("in-memory gate unsatisfied")

    _ensure_fixtures(Path(args.fixtures), Path(args.lights), list(mem.patch_ids))
    prepared = {k: prepared_path(Path(args.fixtures), k) for k in mem.patch_ids}
    config = ExecutorConfig()
    rss_before = zsw.peak_rss_kib()
    res = zsw.run_cell_stack(
        frames, canvas, prepared, args.row, args.col, config.science_config(),
        nx=args.nx, ny=args.ny,
    )
    rss_after = zsw.peak_rss_kib()

    from zemosaic.core.zegrid import assembly as za

    cores = za.crop_all_planes_to_core(res.minitile)
    _save_minitile(res.minitile, out / "minitile_inmem.npz")

    record = {
        "mode": "inmem",
        "cell_id": res.cell_id,
        "row": args.row,
        "col": args.col,
        "n_contributors": res.n_contributors,
        "n_core_contributors": res.n_core_contributors,
        "reference_frame_id": res.reference_frame_id,
        "excluded": [list(e) for e in res.excluded],
        "adequacy": res.adequacy.to_dict(),
        "peak_rss_kib": rss_after,
        "peak_rss_delta_kib": max(0, rss_after - rss_before),
        "patch_shape_hw": list(res.patch.patch_shape_hw),
        "minitile_sha256": _sha256(out / "minitile_inmem.npz"),
    }
    (out / "record_inmem.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[inmem] peak_rss_kib={rss_after} delta={max(0, rss_after - rss_before)}", flush=True)
    return record


def _run_stream(args, frames, canvas) -> dict:
    out = Path(args.out)
    cell, patch, mem = zsw.build_cell_context(frames, canvas, args.row, args.col, args.nx, args.ny)
    config = ExecutorConfig()
    _ensure_fixtures(Path(args.fixtures), Path(args.lights), list(mem.patch_ids))
    prepared = {k: prepared_path(Path(args.fixtures), k) for k in mem.patch_ids}

    res = zstream.run_cell_streaming(
        frames, canvas, prepared, args.row, args.col, config.science_config(),
        Path(args.cache), tile_size=args.tile, nx=args.nx, ny=args.ny,
        reuse_cache=args.reuse_cache, enforce_gate=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    _save_minitile(res.minitile, out / "minitile_stream.npz")

    record = res.to_dict()
    record["mode"] = "stream"
    record["minitile_sha256"] = _sha256(out / "minitile_stream.npz")
    record["cache_dir"] = str(args.cache)
    record["cache_manifest_sha256"] = _sha256(Path(args.cache) / "manifest.json")
    (out / "record_stream.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"[stream] peak_rss_kib={res.peak_rss_kib} delta={res.peak_rss_delta_kib} "
        f"cache_bytes={res.manifest['total_bytes']} tile={args.tile}", flush=True,
    )
    return record


def _build_only(args, frames, canvas) -> dict:
    cell, patch, mem = zsw.build_cell_context(frames, canvas, args.row, args.col, args.nx, args.ny)
    _ensure_fixtures(Path(args.fixtures), Path(args.lights), list(mem.patch_ids))
    prepared = {k: prepared_path(Path(args.fixtures), k) for k in mem.patch_ids}
    from zemosaic.core.zegrid import file_provider as zfp
    from zemosaic.core.zegrid import execution as zxe

    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    crop_plans = {f.frame_id.logical_path: zg.plan_source_roi(f, canvas, patch) for f in patch_frames}
    tracker = zxe.SectionReadTracker()
    manifest = zfp.build_aligned_cache_from_sources(
        patch_frames, prepared, canvas, patch, crop_plans, args.cache,
        tracker=tracker, reuse_cache=args.reuse_cache,
    )
    print(
        f"[build] n_frames={manifest['n_frames']} total_bytes={manifest['total_bytes']} "
        f"cache={args.cache}", flush=True,
    )
    return manifest


def _parity(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, canvas = _load_geometry(Path(args.lights))

    # 1. In-memory reference (fresh subprocess; resumable).
    inmem_json = out / "record_inmem.json"
    if not (inmem_json.exists() and (out / "minitile_inmem.npz").exists()):
        subprocess.run(
            [sys.executable, str(__file__), "--mode", "inmem",
             "--lights", str(args.lights), "--fixtures", str(args.fixtures),
             "--out", str(args.out), "--cache", str(args.cache),
             "--row", str(args.row), "--col", str(args.col),
             "--nx", str(args.nx), "--ny", str(args.ny)],
            check=True,
        )

    # 2. Streaming run(s) at each requested tile size (fresh subprocess each).
    tiles = [int(t) for t in args.tiles.split(",")]
    for tile in tiles:
        stream_json = out / f"record_stream_t{tile}.json"
        if stream_json.exists() and (out / f"minitile_stream_t{tile}.npz").exists():
            continue
        subprocess.run(
            [sys.executable, str(__file__), "--mode", "stream",
             "--lights", str(args.lights), "--fixtures", str(args.fixtures),
             "--out", str(args.out), "--cache", str(args.cache),
             "--row", str(args.row), "--col", str(args.col),
             "--nx", str(args.nx), "--ny", str(args.ny), "--tile", str(tile)],
            check=True,
        )
        # per-tile artifact naming (stream writes minitile_stream.npz by default)
        (out / f"minitile_stream_t{tile}.npz").write_bytes((out / "minitile_stream.npz").read_bytes())
        stream_json.write_text((out / "record_stream.json").read_text())

    inmem_rec = json.loads(inmem_json.read_text())
    inmem_npz = np.load(out / "minitile_inmem.npz")

    parity = {"mission": "ZM-ZEGRID-R6", "cell": inmem_rec["cell_id"],
              "inmem": {k: inmem_rec[k] for k in ("peak_rss_kib", "peak_rss_delta_kib",
                                                   "n_contributors", "reference_frame_id",
                                                   "patch_shape_hw", "minitile_sha256")},
              "tiles": {}}

    for tile in tiles:
        srec = json.loads((out / f"record_stream_t{tile}.json").read_text())
        snpz = np.load(out / f"minitile_stream_t{tile}.npz")
        planes = {
            "science": ("science", False),
            "estimator_weight_sum": ("estimator_weight_sum", False),
            "support_w1": ("support_w1", False),
            "support_w2": ("support_w2", False),
            "n_eff_support": ("n_eff_support", False),
            "valid_mask": ("valid_mask", False),
            "surviving_sample_count": ("surviving_sample_count", False),
        }
        comp = {}
        all_equal = True
        for key, (nm, _nan_eq) in planes.items():
            a = inmem_npz[nm]
            b = snpz[nm]
            eq = bool(np.array_equal(a, b, equal_nan=True))
            comp[key] = {"equal": eq}
            all_equal = all_equal and eq
        ref_eq = str(inmem_npz["reference_frame_id"]) == str(snpz["reference_frame_id"])
        comp["reference_frame_id"] = {"equal": ref_eq,
                                      "inmem": str(inmem_npz["reference_frame_id"]),
                                      "stream": str(snpz["reference_frame_id"])}
        all_equal = all_equal and ref_eq
        parity["tiles"][str(tile)] = {
            "equal": all_equal,
            "planes": comp,
            "stream": {k: srec[k] for k in ("peak_rss_kib", "peak_rss_delta_kib",
                                             "n_contributors", "reference_frame_id",
                                             "tile_size", "minitile_sha256")},
            "cache_total_bytes": srec["manifest"]["total_bytes"],
            "gate": srec["gate"],
        }

    (out / "parity_report.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(f"[parity] report={out / 'parity_report.json'}", flush=True)
    for tile in tiles:
        t = parity["tiles"][str(tile)]
        print(
            f"[parity] tile={tile} equal={t['equal']} "
            f"inmem_rss={parity['inmem']['peak_rss_kib']}KiB "
            f"stream_rss={t['stream']['peak_rss_kib']}KiB", flush=True,
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["build", "inmem", "stream", "parity"])
    ap.add_argument("--lights", type=Path, default=Path("/home/tristan/M106/lights"))
    ap.add_argument("--fixtures", type=Path, default=Path("/home/tristan/zegrid_r2_fixtures"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--row", type=int, default=1)
    ap.add_argument("--col", type=int, default=2)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--tiles", type=str, default="128")
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--reuse-cache", dest="reuse_cache", action="store_true", default=True)
    args = ap.parse_args()

    frames, canvas = _load_geometry(args.lights)

    if args.mode == "build":
        _build_only(args, frames, canvas)
        return 0
    if args.mode == "inmem":
        _run_inmem(args, frames, canvas)
        return 0
    if args.mode == "stream":
        _run_stream(args, frames, canvas)
        return 0
    return _parity(args)


if __name__ == "__main__":
    raise SystemExit(main())
