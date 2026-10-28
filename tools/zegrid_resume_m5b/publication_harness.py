#!/usr/bin/env python3
"""M5-B publication harness — REAL subprocess kill during finalization.

Modes:
  baseline            publish one complete generation (real FITS + manifest)
  kill-before-pointer stage + verify a generation, then kill the process
                      (os._exit(9)) exactly before the atomic pointer flip
  recover             publish a NEW complete generation after the kill

The store is not involved: this harness isolates the finalization/publication
boundary so a real process death can be observed by the test. Windows is
NOT_RUN on this host (the same harness must be replayed on NTFS before any
Windows qualification claim).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402

H, W, CH = 12, 12, 3
SCIENCE = "mosaic_grid.fits"
COVERAGE = "mosaic_grid_coverage.fits"
MANIFEST = "zegrid_manifest.json"
RUN_LOG = "zegrid_run.log"


def members():
    return {
        zpub.ROLE_SCIENCE: SCIENCE,
        zpub.ROLE_COVERAGE: COVERAGE,
        zpub.ROLE_MANIFEST: MANIFEST,
        zpub.ROLE_RUN_LOG: RUN_LOG,
    }


def _science(value: float) -> np.ndarray:
    base = np.linspace(0.0, 10.0, H * W, dtype=np.float32).reshape(H, W)
    cube = np.repeat(base[:, :, None], CH, axis=2) + np.float32(value)
    return np.ascontiguousarray(np.moveaxis(cube, -1, 0))


def _manifest(run_id: str) -> dict:
    return {
        "schema": "zemosaic-zegrid-r7-manifest-v1",
        "engine": "zegrid",
        "canvas": {"width": W, "height": H},
        "outputs": {
            "science": SCIENCE,
            "aesthetic": None,
            "coverage": COVERAGE,
            "uint16": None,
            "run_log": RUN_LOG,
        },
        "science_reference": SCIENCE,
        "science_output_contract": {
            "raw": {"file": SCIENCE, "role": "SCI", "dtype": "float32"},
        },
        "harness": {"run_id": run_id, "note": "M5-B publication subprocess witness"},
    }


def publish(out: Path, run_id: str, generation: str, *, value: float) -> dict:
    pub = zpub.PublicationSession.begin(
        out,
        run_id,
        members=members(),
        shape_hw=(H, W),
        channels=CH,
        finishing_config={
            "dbe_enabled": False,
            "rgb_equalize": False,
            "hole_fill_enabled": False,
            "save_uint16": False,
            "export_aesthetic_fits": False,
        },
        available_ram_bytes=8 * 2 ** 30,
        n_cells=1,
        generation=generation,
    )
    fits.PrimaryHDU(_science(value)).writeto(
        pub.path_for(zpub.ROLE_SCIENCE), overwrite=True
    )
    fits.PrimaryHDU(np.ones((H, W), dtype=np.float32)).writeto(
        pub.path_for(zpub.ROLE_COVERAGE), overwrite=True
    )
    (pub.path_for(zpub.ROLE_MANIFEST)).write_text(
        json.dumps(_manifest(run_id), indent=2) + "\n", encoding="utf-8"
    )
    (out / RUN_LOG).write_text("harness run log\n", encoding="utf-8")
    pub.include_auxiliary(zpub.ROLE_RUN_LOG, out / RUN_LOG)
    gen = pub.commit(
        required_roles=[
            zpub.ROLE_SCIENCE,
            zpub.ROLE_COVERAGE,
            zpub.ROLE_MANIFEST,
            zpub.ROLE_RUN_LOG,
        ]
    )
    return gen.as_report()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode", required=True, choices=("baseline", "kill-before-pointer", "recover")
    )
    parser.add_argument("--generation", required=True)
    parser.add_argument("--value", type=float, default=0.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "kill-before-pointer":
        real_atomic = zpub._atomic_write

        def _kill_at_pointer(path, data, *, tmp_dir):
            if Path(path).name == zpub.POINTER_FILENAME:
                os._exit(9)
            return real_atomic(path, data, tmp_dir=tmp_dir)

        zpub._atomic_write = _kill_at_pointer  # type: ignore[assignment]

    report = publish(
        out, args.run_id, args.generation, value=float(args.value)
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
