"""M1 no-pressure control: compare published science/coverage across source trees.

The M1 registry must not change the no-pressure science path. This script runs
the REAL ZeGrid orchestration on a deterministic small FITS corpus, with the
host memory observation pinned to a valid synthetic budget (the host may have
less live RAM than the PLAN floor; the corpus itself is tiny), then records the
exact SHA-256 of the published science and coverage FITS arrays.

Run it once against the baseline commit and once against the M1 source tree,
then compare the JSON. It is deliberately independent of the product module so
the baseline tree does not need any M1 code.

Usage:
    python3 tools/zegrid_resume_m1/m1_no_pressure_control.py \
        --src-root /path/to/src --output-root /tmp/m1_np_candidate \
        --json-out /tmp/m1_np_candidate.json

No GPU is used (``use_gpu=False``); this is a CPU science/coverage identity
control, not a GPU qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

CORPUS_SIZE = 512
CORPUS_FRAMES = 8
SHIFTS = (
    (0, 0), (96, 96), (192, 192), (288, 288),
    (0, 288), (288, 0), (192, 96), (96, 192),
)
LAYOUT = "2x2"
HALO = 8
#: No synthetic inflation: the run uses the LIVE observation only. If this host
#: cannot admit the PLAN budget, the harness reports NOT_RUN instead of
#: fabricating a favourable budget.
SYNTHETIC_INFLATION = False


def _write_corpus(root: Path):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    inp = root / "in"
    inp.mkdir(parents=True, exist_ok=True)
    scale = 0.0012
    rng = np.random.default_rng(20261027)
    for i in range(CORPUS_FRAMES):
        dx, dy = SHIFTS[i % len(SHIFTS)]
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 - dx * scale, 30.0 + dy * scale]
        wcs.wcs.crpix = [CORPUS_SIZE / 2.0, CORPUS_SIZE / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (CORPUS_SIZE, CORPUS_SIZE)
        yy, xx = np.mgrid[0:CORPUS_SIZE, 0:CORPUS_SIZE]
        data = (
            100.0
            + 20.0 * np.exp(
                -((yy - CORPUS_SIZE / 2) ** 2 + (xx - CORPUS_SIZE / 2) ** 2)
                / (2 * 40.0 ** 2)
            )
            + 3.0 * i
            + rng.normal(0.0, 0.5, (CORPUS_SIZE, CORPUS_SIZE))
        ).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = 10.0
        hdu.writeto(inp / f"frame_{i:04d}.fits", overwrite=True)
    return inp


def _array_sha256(data) -> str:
    import numpy as np

    arr = np.ascontiguousarray(data)
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(json.dumps([int(v) for v in arr.shape]).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _fits_digest(path: Path) -> dict:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        return {
            "file": path.name,
            "shape": [int(v) for v in data.shape],
            "dtype": str(data.dtype),
            "sha256": _array_sha256(data),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    src = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(src))

    import zemosaic.zemosaic_zegrid_mode as zz  # noqa: E402

    # Honest boundary: never raise the available-RAM observation above the live
    # one. The M0 synthetic-control witness is historical evidence, not a
    # physical-memory qualification.
    observed_available = int(zz.available_memory_bytes())

    inp = _write_corpus(out_root)
    out_dir = out_root / "out"
    result = {
        "ok": False,
        "stage": "run",
        "observed_available_bytes": int(observed_available),
        "budget_source": "live_observed",
        "synthetic_inflation": bool(SYNTHETIC_INFLATION),
    }
    try:
        zz.run_zegrid_mode(
            str(inp),
            str(out_dir),
            zconfig=SimpleNamespace(
                zegrid_scheduler_mode="active", zegrid_layout=LAYOUT
            ),
            zegrid_halo_px=HALO,
            use_gpu=False,
            cell_gauge_mode="per_cell",
        )
        result["stage"] = "hash"
        science = out_dir / "mosaic_grid.fits"
        coverage = out_dir / "mosaic_grid_coverage.fits"
        manifest = out_dir / "zegrid_manifest.json"
        result.update(
            {
                "ok": True,
                "science": _fits_digest(science),
                "coverage": _fits_digest(coverage),
                "manifest_exists": manifest.is_file(),
                "publication": {
                    "fits": science.is_file(),
                    "coverage": coverage.is_file(),
                    "manifest": manifest.is_file(),
                },
            }
        )
    except BaseException as exc:  # noqa: BLE001 - the harness records the failure
        result.update(
            {
                "ok": False,
                "stage": "run",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "status": (
                    "NOT_RUN"
                    if type(exc).__name__
                    in (
                        "AdmissionExhausted",
                        "LayoutInfeasible",
                        "ModeInfeasible",
                        "RamContractError",
                    )
                    else "FAIL"
                ),
            }
        )
    result["src_root"] = str(src)
    result["mode"] = "CPU_NO_PRESSURE_PER_CELL"
    result["note"] = (
        "CPU science/coverage identity control using the LIVE memory "
        "observation only; NOT an artificial-high-memory or GPU qualification."
    )
    if result.get("ok"):
        result["status"] = "PASS"
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
