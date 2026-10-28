#!/usr/bin/env python3
"""M5-B REWORK-1 execution probe — the REAL ``_run_single`` path.

Modes:
  new       run the production store-backed path (real descriptors, gauge,
            cells, bounded assembly, REAL finishing, serialization,
            publication) and report the returned product path.
  noop      reopen the same store in a NEW process with resume_mode=auto; the
            expensive seams are replaced by os._exit(77) so the test proves the
            finalized no-op does not rerun layout/gauge/assembly/finishing/
            serialization. Reports the returned product path.
  ram-fail  same production run with the publication RAM probe raising; must
            refuse BEFORE raw science / finishing / generation.

The harness never edits the repo and never activates anything by default.
Windows/GPU are NOT_RUN on this host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astropy.io import fits  # noqa: E402

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import final_mosaic_finishing as zfin  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402
from zemosaic.core.zegrid.executor import ExecutorConfig  # noqa: E402
from zemosaic.zemosaic_stack_plan import FrameInfo  # noqa: E402

MARKER = "M5B_JSON:"


def _store_run_id(store: Path) -> str:
    candidates = zres.discover_runs(store)
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one run, got {candidates}")
    return str(candidates[0]["run_id"])


def _config(store: Path, *, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        zegrid_durable_store_dir=str(store),
        zegrid_durable_store_budget_bytes=512 * 2 ** 20,
        zegrid_resume_mode=("auto" if mode == "noop" else "never"),
        zegrid_halo_px=8,
        export_aesthetic_fits=True,
        final_mosaic_dbe_enabled=True,
        grid_rgb_equalize=True,
        save_final_as_uint16=False,
    )


def _frames(input_dir: Path):
    return [
        FrameInfo(path=p, order=index)
        for index, p in enumerate(sorted(input_dir.glob("*.fits")))
    ]


def _seams_exit(*_args, **_kwargs):
    os._exit(77)


def _dbe_state(science, coverage):
    off = zfin.apply_final_mosaic_finishing(
        science,
        coverage,
        config={
            "dbe_enabled": False,
            "rgb_equalize": False,
            "hole_fill_enabled": False,
            "save_uint16": False,
        },
    )
    return off


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--mode", required=True, choices=("new", "noop", "ram-fail"))
    parser.add_argument(
        "--tamper-state",
        choices=("none", "partial", "wrong", "lost"),
        default="none",
    )
    parser.add_argument("--fail-repair", action="store_true")
    args = parser.parse_args()
    input_dir = Path(args.input)
    out = Path(args.out)
    store = Path(args.store)
    out.mkdir(parents=True, exist_ok=True)

    zconfig = _config(store, mode=args.mode)
    finishing_config = zfin.resolve_finishing_config(zconfig)
    science_config = replace(
        ExecutorConfig().science_config(), cell_gauge_mode="per_cell"
    )
    frames = _frames(input_dir)
    resume_mode = "auto" if args.mode == "noop" else "never"

    # TEST SEAM (documented, SYNTHETIC UPSTREAM RESOURCE): this CI host is
    # memory-constrained; the conservative plan/launch admissions refuse even a
    # tiny corpus. The global ``available_memory_bytes`` observation is replaced
    # by 8 GiB, which affects ALL upstream admissions that use it: plan/layout,
    # bounded-assembly preflight, cell admission and launch. It does NOT affect
    # the PUBLICATION admission, which keeps its REAL observation
    # (``_publication_available_ram_bytes`` -> zsw/psutil). This witness is NOT
    # a budget qualification; every synthetic effect is listed in the JSON.
    zz.available_memory_bytes = lambda: 8 * 2 ** 30

    if args.mode == "noop" and args.tamper_state != "none":
        run_id = _store_run_id(store)
        state = zres.read_state(store)
        generation = zpub.resolve_current(out, run_id)
        outputs = generation.outputs_state()
        if args.tamper_state == "partial":
            outputs = {
                "generation": generation.generation,
                "science": {"sha256": outputs["science"]["sha256"]},
            }
        elif args.tamper_state == "wrong":
            outputs = json.loads(json.dumps(outputs))
            outputs["coverage"]["nbytes"] = int(outputs["coverage"]["nbytes"]) + 1
            outputs["science"]["file"] = "wrong.fits"
        elif args.tamper_state == "lost":
            outputs = {}
        zres.write_state(
            store,
            run_id=state["run_id"],
            expected_cell_ids=state["expected_cell_ids"],
            policy=state["policy"],
            plan_generation=int(state["plan_generation"]) + 1,
            finalized=(args.tamper_state != "lost"),
            gauge_sha256=state["gauge_sha256"],
            outputs=outputs,
            created_unix_ns=state["created_unix_ns"],
            note=state["note"],
        )
    if args.mode == "noop" and args.fail_repair:
        _real_write_state = zres.write_state

        def _fail_repair(*a, **k):
            if k.get("finalized") is True:
                raise OSError("injected state repair failure")
            return _real_write_state(*a, **k)

        zres.write_state = _fail_repair


    if args.mode == "noop":
        # Any expensive rerun in the finalized no-op is a hard failure.
        zz._choose_layout_mode_aware = _seams_exit
        zz.zphot.compute_global_gauge = _seams_exit
        zz._assemble_cells = _seams_exit
        zz._assemble_cells_guarded = _seams_exit
        zz.zfin.apply_final_mosaic_finishing = _seams_exit
        zz._write_raw_science_fits = _seams_exit
        zz._write_outputs = _seams_exit

    if args.mode == "ram-fail":
        def _fail_probe():
            raise RuntimeError("injected RAM probe failure")

        zz._publication_available_ram_bytes = _fail_probe

    try:
        result = zz._run_single(
            frames,
            str(input_dir),
            str(out),
            progress_callback=None,
            science_config=science_config,
            zconfig=zconfig,
            workers=1,
            finishing_config=finishing_config,
            halo_px=8,
            scheduler_mode=zsched.MODE_ACTIVE,
            pinned_layout=zz._parse_pinned_layout("1x1"),
            resume_mode=resume_mode,
        )
    except zpub.PublicationCapacityError as exc:
        print(
            MARKER
            + json.dumps(
                {
                    "mode": args.mode,
                    "refused": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "plan": exc.plan,
                }
            )
        )
        return 0
    except Exception as exc:
        print(
            MARKER
            + json.dumps(
                {
                    "mode": args.mode,
                    "refused": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 2

    run_id = _store_run_id(store)
    generation = zpub.resolve_current(out, run_id)
    payload = json.loads(generation.path(zpub.ROLE_MANIFEST).read_text())
    state_after = zres.read_state(store)
    ack_complete, ack_problems = zpub.ack_matches_generation(
        state_after.get("outputs") or {}, generation
    )
    report = {
        "mode": args.mode,
        "refused": False,
        "synthetic_upstream_ram": True,
        "synthetic_effects": [
            "available_memory_bytes = 8 GiB (synthetic) for plan/layout, "
            "bounded-assembly preflight, cell admission and launch",
            "publication admission keeps its REAL observation",
            "not a budget qualification",
        ],
        "run_id": run_id,
        "generation": generation.generation,
        "return": str(result),
        "return_exists": bool(Path(result).exists()),
        "return_inside_generation": str(
            Path(result).parent.resolve()
        ) == str(generation.generation_dir.resolve()),
        "return_role": (
            zpub.ROLE_AESTHETIC
            if Path(result).name == generation.members.get(zpub.ROLE_AESTHETIC, {}).get("file")
            else zpub.ROLE_SCIENCE
        ),
        "members": sorted(generation.members),
        "manifest_finishing": payload.get("finishing"),
        "manifest_outputs": payload.get("outputs"),
        "publication_sha256": generation.publication_sha256,
        "state_finalized": bool(state_after.get("finalized")),
        "state_ack_complete": bool(ack_complete),
        "state_ack_problems": list(ack_problems),
        "state_outputs": state_after.get("outputs") or {},
    }
    try:
        verdict = zpub.verify_finalized(out, store, run_id)
        report["finalization_reason"] = verdict.reason
    except Exception as exc:
        report["finalization_reason"] = f"{type(exc).__name__}: {exc}"
    raw_path = generation.path(zpub.ROLE_SCIENCE)
    cov_path = generation.path(zpub.ROLE_COVERAGE)
    if args.mode == "noop":
        # Finishing is one of the seams replaced by a hard exit in this mode;
        # do NOT call it here (the no-op itself must not finish).
        print(MARKER + json.dumps(report))
        return 0
    raw = np.moveaxis(np.asarray(fits.getdata(raw_path), dtype=np.float32), 0, -1)
    coverage = np.asarray(fits.getdata(cov_path), dtype=np.float32)
    off = zfin.apply_final_mosaic_finishing(
        raw,
        coverage,
        config={
            "dbe_enabled": False,
            "rgb_equalize": False,
            "hole_fill_enabled": False,
            "save_uint16": False,
        },
    )
    on = zfin.apply_final_mosaic_finishing(raw, coverage, config=finishing_config)
    report["finishing_off_identity"] = bool(
        np.array_equal(off.science, raw, equal_nan=True)
    )
    report["finishing_on_differs"] = bool(
        not np.array_equal(on.science, raw, equal_nan=True)
    )
    report["finishing_on_mask_preserved"] = bool(
        np.array_equal(
            np.isfinite(on.science), np.isfinite(raw), equal_nan=True
        )
    )
    if zpub.ROLE_AESTHETIC in generation.members:
        aest = np.moveaxis(
            np.asarray(
                fits.getdata(generation.path(zpub.ROLE_AESTHETIC)),
                dtype=np.float32,
            ),
            0,
            -1,
        )
        report["aesthetic_matches_real_finishing"] = bool(
            np.array_equal(aest, on.science, equal_nan=True)
        )
    print(MARKER + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
