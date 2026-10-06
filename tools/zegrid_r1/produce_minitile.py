"""ZM-ZEGRID-R1 — produce the real MiniTile for Cell r0000c0000 and persist artifacts.

Runs the full local pipeline (geometry -> section reads -> local reprojection ->
canonical stack -> core crop) and writes:
  * <out>/minitile_r0000c0000.npz  (science, estimator, W1/W2/N_eff, valid, counts)
  * <out>/minitile_r0000c0000.json (provenance, reference, exclusions, ROIs, hashes)
Full-frame fixture preparation I/O is SEPARATE (see prepare_rgb_fixture.py) and
is NOT counted as local execution.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zemosaic.core.zegrid import assembly as za  # noqa: E402
from zemosaic.core.zegrid import execution as zx  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import science_adapter as zs  # noqa: E402

LIGHTS = Path("/home/tristan/M106/lights")
FIXTURES = Path("/tmp/zegrid_r1_fixtures")
OUT = Path("/tmp/zegrid_r1_outputs")


def main() -> int:
    frames, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    layout = zg.build_layout(canvas, 5, 4)
    cell = zg.ZeGridCell(
        zg.cell_id(0, 0), canvas.canvas_id, layout.layout_id, 0, 0,
        layout.cell_bounds(0, 0, canvas),
    )
    patch = zg.build_patch(canvas, cell, 8)
    mem = zg.compute_membership(frames, canvas, cell, patch)

    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    prepared_paths = {k: str(FIXTURES / (Path(k).stem + "_rgb.fits")) for k in mem.patch_ids}
    crop_plans = {
        f.frame_id.logical_path: zg.plan_source_roi(f, canvas, patch) for f in patch_frames
    }

    tracker = zx.SectionReadTracker()
    contribs = zx.build_patch_contributors(
        patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
    )

    cfg = zs.MiniTileScienceConfig()
    sres = zs.run_minitile_stack(
        [c.rgb for c in contribs],
        [c.geometric_support for c in contribs],
        [c.frame_id for c in contribs],
        cfg,
    )
    mt = za.extract_minitile(patch, sres)
    cores = za.crop_all_planes_to_core(mt)

    OUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUT / "minitile_r0000c0000.npz"
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

    manifest = json.loads((FIXTURES / "prepared_manifest.json").read_text())
    provenance = {
        "mission": "ZM-ZEGRID-R1",
        "cell_id": "r0000c0000",
        "canvas": {"width": canvas.width, "height": canvas.height,
                   "resolution_deg": canvas.resolution_deg, "canvas_id": canvas.canvas_id},
        "layout": {"nx": 5, "ny": 4, "halo_px": 8, "layout_id": layout.layout_id},
        "core": {"x0": 0, "y0": 0, "x1": 480, "y1": 819},
        "patch": {"x0": 0, "y0": 0, "x1": 488, "y1": 827},
        "core_slice": list(mt.core_slice),
        "patch_shape_hw": list(mt.patch_shape_hw),
        "science_config": {
            "normalization": cfg.normalization, "weighting": cfg.weighting,
            "rejection": cfg.rejection, "combine": cfg.combine,
            "backend": cfg.backend, "taper": cfg.taper,
            "taper_px": cfg.taper_px, "taper_floor": cfg.taper_floor,
            "reference_index": cfg.reference_index,
        },
        "frame_order": list(mt.frame_order),
        "reference_frame_id": mt.reference_frame_id,
        "excluded_frames": list(mt.excluded),
        "contributors": len(contribs),
        "core_contributors": list(mem.core_ids),
        "patch_contributors": list(mem.patch_ids),
        "crop_plans": {
            k: {"x0": v.source_bounds.x0, "y0": v.source_bounds.y0,
                "x1": v.source_bounds.x1, "y1": v.source_bounds.y1,
                "margin_px": v.margin_px}
            for k, v in crop_plans.items() if v is not None
        },
        "section_reads": [
            {"path": Path(r.path).name, "x0": r.source_bounds.x0, "y0": r.source_bounds.y0,
             "x1": r.source_bounds.x1, "y1": r.source_bounds.y1,
             "n_pixels_read": r.n_pixels_read, "full_frame_pixels": r.full_frame_pixels}
            for r in tracker.records
        ],
        "fixture_manifest_sha256": hashlib.sha256(
            (FIXTURES / "prepared_manifest.json").read_bytes()
        ).hexdigest(),
        "fixtures": [
            {"raw": f["raw"], "raw_sha256": f["raw_sha256"],
             "prepared_sha256": f["prepared_sha256"]}
            for f in manifest["frames"]
        ],
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    json_path = OUT / "minitile_r0000c0000.json"
    json_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print("npz:", npz_path)
    print("json:", json_path)
    print("reference:", mt.reference_frame_id)
    print("excluded:", mt.excluded)
    print("section reads:", len(tracker.records),
          "sum local px:", sum(r.n_pixels_read for r in tracker.records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
