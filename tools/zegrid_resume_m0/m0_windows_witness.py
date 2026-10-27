#!/usr/bin/env python3
"""ZM-ZEGRID-RESILIENT-RUN-RESUME M0 — Windows REAL-GPU baseline witness.

Purpose
-------
Reproduce, in a few minutes, the recorded baseline failure on the REAL product
orchestration path with a REAL GPU: two or more cells complete, an available-RAM
drop is injected at the product boundary, the ACTIVE GPU pipeline raises
``_R55GpuFallback`` and discards every completed GPU result, then the
whole-batch CPU re-admission fails with ``AdmissionExhausted``.

This script is deliberately NOT a scientific qualification and NOT a resume
implementation. It labels:
  * mode=WINDOWS_REAL_GPU, WINDOWS_NO_GPU or LOCAL_CONTROL_CPU,
  * synthetic_pressure=true (the trajectory is injected at
    ``zemosaic_zegrid_mode.available_memory_bytes``, not physical RAM load),
  * provenance (installed module SHA-256 + version + dependency hashes),
  * the stage reached and whether the failure is the EXPECTED baseline failure
    or an UNEXPECTED failure.

Fail-closed rules (M0 rework-1):
* Parent succeeds ONLY when child returncode == 0 AND witness.status == "PASS"
  AND witness.classification == "EXPECTED_BASELINE_FAILURE" AND the artifact cap
  was not exceeded.
* Provenance equality (installed module hash == the ACTUAL imported module
  file hash) is enforced in BOTH parent and child. A mismatch refuses unless
  ``--allow-unverified-provenance`` is explicit; the witness then records
  ``provenance_verified=false`` and a NON-QUALIFYING label.
* The injected high observation is clamped by LIVE available RAM; both values
  and ``synthetic_observation_capped_by_live_ram`` are recorded.
* ``gpu_qualified`` is true ONLY for a real Windows GPU run that completed at
  least one cell with ``backend_used == "gpu"`` and passed every witness check.

Safety
------
* Refuses a non-fresh output directory and, by default, any output outside
  ``%USERPROFILE%\\Desktop\\zgprobe`` (the approved new subfolder root).
* Hard cap: 300 s per invocation (parent-owned process tree is killed on
  timeout) and 1 GiB of NEW artifacts. The parent watches the tree during the
  run and kills the child if the cap is exceeded; status/exit are coherent.
* Child stdout/stderr is streamed to a file, never buffered unbounded in RAM.
* Never edits the installed product; imports only the supplied ``--src-root``.
* No persistent service/task; the child is a normal owned process.
* Original inputs/outputs are never deleted. Cleanup is confined to the fresh
  job directory created by this script.

Typical Windows invocation:

  py m0_windows_witness.py ^
    --src-root "C:\\path\\to\\worktree\\src" ^
    --installed-module "C:\\path\\to\\installed\\zemosaic_zegrid_mode.py" ^
    --output-root "C:\\Users\\JarvisJunior\\Desktop\\zgprobe\\m0_witness_20261027_01"

or through ``run_m0_windows_witness.ps1``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

SCHEMA = "zemosaic-zegrid-resume-m0-windows-witness-v1"
MISSION = "ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027"
MAX_SECONDS_HARD = 300
MAX_NEW_BYTES_HARD = 1024 ** 3  # 1 GiB
CORPUS_SIZE = 512
CORPUS_FRAMES = 8
CORPUS_SHIFTS = (
    (0, 0), (96, 96), (192, 192), (288, 288),
    (0, 288), (288, 0), (192, 96), (96, 192),
)
LAYOUT = "2x2"
HALO = 8
RAW_HIGH = 16 * 1024 ** 3
RAW_LOW = 700 * 1024 ** 2
MIN_EFFECTIVE_HIGH = 2 * 1024 ** 3

# Key imported dependencies whose identity must be recorded. The actual
# imported files (not the expected paths) are hashed in the child.
KEY_DEPENDENCIES = (
    "core/zegrid/parallel.py",
    "core/zegrid/runtime_control.py",
    "core/zegrid/ram_contract.py",
    "core/zegrid/resource_envelope.py",
    "core/zegrid/scheduler.py",
    "core/zegrid/storage_profile.py",
    "core/zegrid/assembly.py",
    "core/zegrid/cell_feather.py",
    "core/zegrid/cell_offset.py",
    "core/zegrid/photometric.py",
    "core/zegrid/science_adapter.py",
    "core/zegrid/streaming.py",
    "core/canonical_streaming.py",
    "core/canonical_engine.py",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_array(value) -> str:
    import numpy as np

    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(repr(tuple(int(v) for v in arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _hash_core(core: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(core):
        h.update(key.encode("ascii"))
        h.update(_hash_array(core[key]).encode("ascii"))
    return h.hexdigest()


def _tree_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _zgprobe_root() -> Path:
    return Path(os.path.expandvars(r"%USERPROFILE%")) / "Desktop" / "zgprobe"


def _preflight_output(output_root: Path, allow_outside: bool) -> None:
    approved = _zgprobe_root().resolve()
    resolved = output_root.resolve()
    if not allow_outside:
        try:
            resolved.relative_to(approved)
        except ValueError:
            raise SystemExit(
                "REFUSED: --output-root must be a NEW subfolder of "
                f"{approved} (pass --allow-outside-zgprobe to override explicitly)"
            )
    if resolved.exists() and any(resolved.iterdir()):
        raise SystemExit(
            f"REFUSED: output directory is not fresh/empty: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)


def estimate_new_bytes() -> int:
    """Conservative preflight estimate (not a measured peak).

    FITS are float32 (4 B/px); add support planes (1 B/px), a bounded cache
    (2 x float32 frame bytes), canvas/output planes (float64 + uint16), and
    logs/diagnostics. This is deliberately above the tiny witness corpus.
    """
    frame_bytes = CORPUS_SIZE * CORPUS_SIZE * 4
    corpus = CORPUS_FRAMES * frame_bytes
    support = CORPUS_FRAMES * CORPUS_SIZE * CORPUS_SIZE
    cache = 2 * corpus
    canvas_side = 2 * CORPUS_SIZE
    outputs = 2 * canvas_side * canvas_side * (3 * 8 + 3 * 2)
    logs = 32 * 1024 ** 2
    return int(corpus + support + cache + outputs + logs + 32 * 1024 ** 2)


def _write_corpus(root: Path) -> tuple[Path, list[dict]]:
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    inp = root / "corpus"
    inp.mkdir(parents=True, exist_ok=False)
    scale = 0.0012
    rng = np.random.default_rng(20261027)
    files: list[dict] = []
    for i in range(CORPUS_FRAMES):
        dx, dy = CORPUS_SHIFTS[i % len(CORPUS_SHIFTS)]
        ra = 10.0 - dx * scale
        dec = 30.0 + dy * scale
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
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
        name = f"frame_{i:04d}.fits"
        path = inp / name
        hdu.writeto(path, overwrite=True)
        files.append({"name": name, "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    (inp / "stack_plan.csv").write_text(
        "file_path,exposure\n" + "".join(f"{f['name']},10.0\n" for f in files),
        encoding="utf-8",
    )
    return inp, files


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _dependency_hashes(root: Path) -> dict:
    """Hash actual dependency files under ``root/zemosaic``."""
    out = {}
    for rel in KEY_DEPENDENCIES:
        path = root / "zemosaic" / rel
        out[rel] = _sha256_file(path) if path.is_file() else None
    return out


def _provenance(args: argparse.Namespace, *, imported_module=None) -> dict:
    src = Path(args.src_root).resolve()
    source_main = src / "zemosaic" / "zemosaic_zegrid_mode.py"
    version_file = src.parent / "version.txt"
    version = None
    if version_file.is_file():
        version = version_file.read_text(encoding="utf-8", errors="replace").strip()
    src_sha = _sha256_file(source_main) if source_main.is_file() else None

    actual_path = None
    actual_sha = None
    if imported_module is not None and getattr(imported_module, "__file__", None):
        actual_path = Path(imported_module.__file__).resolve()
        if actual_path.is_file():
            actual_sha = _sha256_file(actual_path)
    if actual_sha is None and source_main.is_file():
        actual_path = source_main
        actual_sha = src_sha

    record = {
        "src_root": str(src),
        "src_module": str(source_main),
        "src_module_sha256": src_sha,
        "version_txt": version,
        "actual_imported_module": str(actual_path) if actual_path else None,
        "actual_imported_module_sha256": actual_sha,
        "source_actual_match": None,
        "installed_module": None,
        "installed_module_sha256": None,
        "installed_src_root": None,
        "main_module_match": None,
        "source_dependency_sha256": _dependency_hashes(src),
        "dependencies_match": None,
        "provenance_verified": False,
        "qualification": "NON_QUALIFYING_CONTROL",
    }
    if args.installed_module:
        installed = Path(args.installed_module).resolve()
        record["installed_module"] = str(installed)
        record["installed_module_sha256"] = (
            _sha256_file(installed) if installed.is_file() else None
        )
        record["main_module_match"] = (
            record["installed_module_sha256"] == actual_sha
        )
    if getattr(args, "installed_src_root", None):
        installed_root = Path(args.installed_src_root).resolve()
        record["installed_src_root"] = str(installed_root)
        installed_deps = _dependency_hashes(installed_root)
        record["installed_dependency_sha256"] = installed_deps
        record["dependencies_match"] = all(
            installed_deps.get(key) == record["source_dependency_sha256"].get(key)
            for key in KEY_DEPENDENCIES
        )
    record["source_actual_match"] = bool(
        actual_sha is not None and src_sha is not None and actual_sha == src_sha
    )
    verified = bool(
        record["main_module_match"] is True
        and record["source_actual_match"] is True
        and record["dependencies_match"] is not False
        and actual_sha is not None
    )
    record["provenance_verified"] = verified
    record["qualification"] = (
        "VERIFIED_INSTALLED_RUNTIME" if verified else "NON_QUALIFYING_CONTROL"
    )
    return record


def provenance_gate(prov: dict, *, allow_unverified: bool) -> dict:
    """Raise unless provenance is verified; explicit override is labelled."""
    if prov.get("provenance_verified") is True:
        return {
            "allowed": True,
            "provenance_verified": True,
            "qualification": prov.get("qualification", "VERIFIED_INSTALLED_RUNTIME"),
            "override_used": False,
        }
    if not allow_unverified:
        raise SystemExit(
            "REFUSED: provenance mismatch/unverified "
            f"(installed={prov.get('installed_module_sha256')} "
            f"actual={prov.get('actual_imported_module_sha256')}); pass "
            "--allow-unverified-provenance only for a clearly non-qualifying run"
        )
    return {
        "allowed": True,
        "provenance_verified": False,
        "qualification": "NON_QUALIFYING_CONTROL",
        "override_used": True,
    }


# ---------------------------------------------------------------------------
# Child: the actual witness (real GPU, synthetic memory trajectory)
# ---------------------------------------------------------------------------
def _child(args: argparse.Namespace) -> int:
    job = Path(args.output_root).resolve()
    stage_path = job / "stage.json"
    result_path = job / "witness.json"

    def stage(name: str, **extra) -> None:
        _write_json(
            stage_path,
            {"schema": SCHEMA, "stage": name, "updated_at": time.time(), **extra},
        )

    stage("import")
    sys.path.insert(0, str(Path(args.src_root).resolve()))
    from zemosaic import zemosaic_zegrid_mode as zz  # noqa: PLC0415
    from zemosaic.core.zegrid import gpu_readiness as zgr  # noqa: PLC0415
    from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: PLC0415
    from zemosaic.core.zegrid import resource_envelope as renv  # noqa: PLC0415

    prov = _provenance(args, imported_module=zz)
    gate = provenance_gate(
        prov, allow_unverified=bool(args.allow_unverified_provenance)
    )
    prov["provenance_verified"] = bool(gate["provenance_verified"])
    prov["qualification"] = str(gate["qualification"])
    prov["override_used"] = bool(gate["override_used"])

    stage("gpu_readiness")
    readiness = zgr.probe_gpu_readiness_safe(device_index=0)
    try:
        probe = zz.zgpu.probe_gpu_backend()
    except Exception as exc:  # contained: absent backend is NOT_RUN, not failure
        probe = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    gpu_available = bool(
        getattr(readiness, "available", False) and probe.get("available")
    )
    if os.name == "nt":
        mode = "WINDOWS_REAL_GPU" if gpu_available else "WINDOWS_NO_GPU"
    else:
        mode = "LOCAL_CONTROL_CPU"

    import psutil  # noqa: PLC0415

    live_available = int(psutil.virtual_memory().available)
    effective_high = int(min(RAW_HIGH, live_available))
    effective_low = int(min(RAW_LOW, max(256 * 1024 ** 2, effective_high // 2)))
    capped_by_live = bool(effective_high < RAW_HIGH)
    if effective_high < MIN_EFFECTIVE_HIGH:
        _write_json(
            result_path,
            {
                "schema": SCHEMA,
                "mission": MISSION,
                "mode": mode,
                "status": "NOT_RUN",
                "classification": "NOT_RUN",
                "reason": (
                    "live available RAM below the minimum witness envelope: "
                    f"{effective_high} < {MIN_EFFECTIVE_HIGH} bytes"
                ),
                "gpu_qualified": False,
                "gpu_available": gpu_available,
                "provenance": prov,
                "synthetic_pressure": True,
            },
        )
        return 2

    if not gpu_available:
        _write_json(
            result_path,
            {
                "schema": SCHEMA,
                "mission": MISSION,
                "mode": mode,
                "status": "NOT_RUN",
                "classification": "NOT_RUN",
                "gpu_qualified": False,
                "reason": "GPU/CuPy unavailable in this Python environment",
                "readiness": _readiness_dict(readiness),
                "probe": probe,
                "provenance": prov,
                "synthetic_pressure": True,
            },
        )
        return 2

    stage("corpus")
    inp, corpus_files = _write_corpus(job)

    memory = {"low": False, "high": effective_high, "low_bytes": effective_low}
    sequence: list[dict] = []

    def _available() -> int:
        return int(memory["low"] and memory["low_bytes"] or memory["high"])

    zz.available_memory_bytes = _available
    real_consume = zz._consume_stack_cell

    def _consume(prepared, **kw):
        out = real_consume(prepared, **kw)
        if isinstance(out, dict) and out.get("status") == "complete":
            sequence.append(
                {
                    "unit_index": len(sequence),
                    "cell_id": str(out["cell_id"]),
                    "core_sha256": _hash_core(out["core"]),
                    "n_frames": int(out.get("n_frames", 0)),
                    "backend_used": out.get("backend_used"),
                    "status": "complete",
                }
            )
            stage("gpu_units", completed_cells=len(sequence))
            if len(sequence) >= int(args.pressure_after):
                memory["low"] = True
        return out

    zz._consume_stack_cell = _consume
    zz._R55E2_MEASURE_PREFETCH_CAP = 0

    out_dir = job / "out"
    exc: BaseException | None = None
    stage("run")
    try:
        zz.run_zegrid_mode(
            str(inp),
            str(out_dir),
            zconfig=_Config(),
            zegrid_halo_px=HALO,
            use_gpu=True,
        )
    except BaseException as err:  # noqa: BLE001 - the baseline failure is the witness
        exc = err

    failure = _read_json(out_dir / "zegrid_gpu_failure.json") or {}
    checks = _witness_checks(
        exc=exc,
        failure=failure,
        sequence=sequence,
        pressure_after=int(args.pressure_after),
        provenance_verified=bool(prov["provenance_verified"]),
        out_dir=out_dir,
    )
    expected = all(checks.values())
    if expected:
        classification = "EXPECTED_BASELINE_FAILURE"
    elif isinstance(exc, rcontract.AdmissionExhausted):
        classification = "UNEXPECTED_FAILURE"
    elif exc is None:
        classification = "UNEXPECTED_SUCCESS"
    else:
        classification = "UNEXPECTED_FAILURE"

    gpu_qualified = gpu_qualified_for(
        mode=mode,
        gpu_available=gpu_available,
        expected=expected,
        sequence=sequence,
        provenance_verified=bool(prov["provenance_verified"]),
    )
    pipeline = failure.get("pipeline") or {}
    ledger = pipeline.get("ledger") or {}
    units = (failure.get("tile_state") or {}).get("units") or []
    new_bytes = _tree_bytes(job)
    capped = new_bytes > int(args.max_new_bytes)

    payload = {
        "schema": SCHEMA,
        "mission": MISSION,
        "mode": mode,
        "status": "PASS" if expected else "FAIL",
        "classification": classification,
        "checks": checks,
        "gpu_qualified": gpu_qualified,
        "gpu_qualification_reason": (
            "real Windows GPU execution with backend_used=gpu and all witness "
            "checks passed"
            if gpu_qualified
            else "not a qualified real-GPU success (mode, backend, provenance, "
            "or witness checks did not all pass)"
        ),
        "gpu_available": gpu_available,
        "gpu_readiness": _readiness_dict(readiness),
        "provenance": prov,
        "synthetic_pressure": {
            "enabled": True,
            "boundary": "zemosaic_zegrid_mode.available_memory_bytes",
            "raw_available_high_bytes": RAW_HIGH,
            "live_available_bytes": live_available,
            "effective_high_bytes": effective_high,
            "effective_low_bytes": effective_low,
            "synthetic_observation_capped_by_live_ram": capped_by_live,
            "armed_after_completed_cells": int(args.pressure_after),
            "note": (
                "Valid-shaped synthetic trajectory injected at the product "
                "boundary; not physical RAM load and not a RAM-exhaustion test."
            ),
        },
        "scenario": {
            "layout": LAYOUT,
            "halo_px": HALO,
            "prefetch_cap": 0,
            "storage_profile_requested": "auto",
            "scheduler_mode": "active",
            "scientific_floors": "REAL (not lowered)",
            "tile_note": (
                "The local/small-corpus ladder resolves to tile 409 because the "
                "canvas/patch geometry and the admitted VRAM envelope differ "
                "from M31 (tile 32, prefetch 0). The witness reproduces the "
                "host-RAM discard/admission failure, not M31's exact tile."
            ),
        },
        "stage_reached": "cpu_readmission_failure" if exc is not None else "complete",
        "completed_before_failure": sequence,
        "gpu_fallback": {
            "exception_type": (failure.get("exception") or {}).get("type"),
            "reason_code": failure.get("reason_code"),
            "detail": failure.get("detail"),
            "tile_px": (failure.get("tile_state") or {}).get("tile_px"),
            "units_total": pipeline.get("units_total"),
            "units_produced": pipeline.get("units_produced"),
            "units_consumed": pipeline.get("units_consumed"),
            "units_released": pipeline.get("units_released"),
            "prefetch_units": pipeline.get("prefetch_units"),
            "cleanup_failures": pipeline.get("cleanup_failures"),
            "ledger_balanced": ledger.get("balanced"),
            "host_leases": ledger.get("host_leases"),
            "device_leases": ledger.get("device_leases"),
            "disk_leases": ledger.get("disk_leases"),
            "completed_units": [
                u for u in units if u.get("status") == "completed"
            ],
            "refused_units": [u for u in units if u.get("status") == "refused"],
        },
        "cpu_readmission_failure": {
            "exception_type": type(exc).__name__ if exc is not None else None,
            "message": str(exc) if exc is not None else None,
            "expected": isinstance(exc, rcontract.AdmissionExhausted),
        },
        "publication": {
            "fits": (out_dir / "mosaic_grid.fits").exists(),
            "coverage": (out_dir / "mosaic_grid_coverage.fits").exists(),
            "manifest": (out_dir / "zegrid_manifest.json").exists(),
        },
        "cleanup": {
            "cache_staging_bytes_remaining": _tree_bytes(out_dir / "__zegrid_cache__"),
            "pipeline_cleanup_failures": pipeline.get("cleanup_failures"),
        },
        "no_oom": not any(isinstance(e, MemoryError) for e in _exception_chain(exc)),
        "resources": {
            "new_artifacts_bytes": new_bytes,
            "max_new_bytes": int(args.max_new_bytes),
            "cap_exceeded": capped,
        },
        "expected_failure": expected,
        "unexpected_failure": classification.startswith("UNEXPECTED"),
        "note": (
            "PASS means the BASELINE FAILURE was reproduced for the expected "
            "reason. It never means the product succeeded, resume exists, or "
            "GPU/science qualification was obtained. gpu_qualified is true only "
            "for a real Windows GPU execution proven by backend_used=gpu."
        ),
    }
    _write_json(result_path, payload)
    stage(
        "done",
        classification=classification,
        status=payload["status"],
        completed_cells=len(sequence),
    )
    if capped:
        return 4
    return 0 if expected else 1


def gpu_qualified_for(
    *, mode, gpu_available, expected, sequence, provenance_verified
) -> bool:
    """GPU qualification is a narrow proof, never a default/fallback value."""
    backend_ok = bool(sequence) and all(
        cell.get("backend_used") == "gpu" for cell in sequence
    )
    return bool(
        mode == "WINDOWS_REAL_GPU"
        and gpu_available is True
        and expected is True
        and backend_ok
        and provenance_verified is True
    )


def _witness_checks(
    *, exc, failure, sequence, pressure_after, provenance_verified, out_dir
) -> dict:
    pipeline = failure.get("pipeline") or {}
    ledger = pipeline.get("ledger") or {}
    units = (failure.get("tile_state") or {}).get("units") or []
    completed = [u for u in units if u.get("status") == "completed"]
    refused = [u for u in units if u.get("status") == "refused"]
    return {
        "exception_is_admission_exhausted": type(exc).__name__ == "AdmissionExhausted",
        "reason_is_host_ram": failure.get("reason_code") == "R55_REFUSE_HOST_RAM",
        "at_least_pressure_after_completed_cells": len(sequence) >= int(pressure_after),
        "all_cells_report_backend_used": bool(sequence)
        and all(cell.get("backend_used") for cell in sequence),
        "pipeline_consumed_at_least_pressure_after": int(
            pipeline.get("units_consumed", 0)
        )
        >= int(pressure_after),
        "pipeline_cancelled": pipeline.get("cancelled") is True,
        "completed_units_present": len(completed) >= int(pressure_after),
        "completed_units_discarded": bool(completed)
        and all(u.get("discarded") is True for u in completed),
        "refused_unit_fallback_cpu": bool(refused)
        and all(u.get("action") == "fallback_cpu" for u in refused),
        "zero_publication": not any(
            (out_dir / name).exists()
            for name in (
                "mosaic_grid.fits",
                "mosaic_grid_coverage.fits",
                "zegrid_manifest.json",
            )
        ),
        "cleanup_failures_empty": pipeline.get("cleanup_failures") == [],
        "ledger_balanced": ledger.get("balanced") is True,
        "all_leases_zero": all(
            ledger.get(key) == 0
            for key in ("host_leases", "device_leases", "disk_leases")
        ),
        "no_oom": not any(isinstance(e, MemoryError) for e in _exception_chain(exc)),
        "provenance_verified": provenance_verified is True,
    }


def _readiness_dict(readiness) -> dict:
    try:
        return dataclasses.asdict(readiness)
    except Exception:
        return {"unserializable": repr(readiness)}


def _exception_chain(exc: BaseException | None):
    cursor = exc
    seen = 0
    while cursor is not None and seen < 32:
        yield cursor
        cursor = cursor.__cause__
        seen += 1


class _Config:
    zegrid_scheduler_mode = "active"
    zegrid_layout = LAYOUT


# ---------------------------------------------------------------------------
# Parent: fresh dir + owned-process timeout + live cap + fail-closed result
# ---------------------------------------------------------------------------
def evaluate_child_result(
    *,
    returncode: int,
    timed_out: bool,
    witness: dict | None,
    new_bytes: int,
    max_new_bytes: int,
    stage_record: dict | None,
) -> dict:
    """Pure fail-closed parent verdict. No filesystem, no process."""
    stage = (stage_record or {}).get("stage")
    if timed_out:
        return {
            "status": "TIMEOUT",
            "exit": 3,
            "reason": "child exceeded the owned-process timeout",
            "stage_reached": stage,
        }
    if int(new_bytes) > int(max_new_bytes):
        return {
            "status": "CAP_EXCEEDED",
            "exit": 4,
            "reason": (
                f"new artifacts {int(new_bytes)} > max {int(max_new_bytes)}"
            ),
            "stage_reached": stage,
        }
    if int(returncode) == 2 and witness is not None and witness.get(
        "status"
    ) == "NOT_RUN":
        return {
            "status": "NOT_RUN",
            "exit": 2,
            "reason": witness.get("reason", "not run"),
            "stage_reached": stage,
        }
    if int(returncode) != 0:
        return {
            "status": "CHILD_EXIT_NONZERO",
            "exit": 1,
            "reason": f"child returncode={int(returncode)}; witness never trusted alone",
            "stage_reached": stage,
        }
    if witness is None:
        return {
            "status": "PROCESS_FAILED",
            "exit": 1,
            "reason": "no witness.json produced",
            "stage_reached": stage,
        }
    status = witness.get("status")
    if status == "NOT_RUN":
        return {
            "status": "NOT_RUN",
            "exit": 2,
            "reason": witness.get("reason", "not run"),
            "stage_reached": stage,
        }
    if status != "PASS":
        return {
            "status": "FAIL",
            "exit": 1,
            "reason": f"witness status={status!r}",
            "stage_reached": stage,
        }
    if witness.get("classification") != "EXPECTED_BASELINE_FAILURE":
        return {
            "status": "FAIL",
            "exit": 1,
            "reason": (
                "witness classification is not EXPECTED_BASELINE_FAILURE: "
                f"{witness.get('classification')!r}"
            ),
            "stage_reached": stage,
        }
    if witness.get("resources", {}).get("cap_exceeded") is True:
        return {
            "status": "CAP_EXCEEDED",
            "exit": 4,
            "reason": "witness reports cap_exceeded; PASS is never allowed",
            "stage_reached": stage,
        }
    return {
        "status": "PASS",
        "exit": 0,
        "reason": "returncode==0 and PASS/EXPECTED_BASELINE_FAILURE",
        "stage_reached": stage,
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _parent(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    _preflight_output(output_root, args.allow_outside_zgprobe)
    estimate = estimate_new_bytes()
    if int(args.max_new_bytes) < estimate:
        raise SystemExit(
            "REFUSED: --max-new-bytes is below the conservative preflight "
            f"estimate ({estimate} B); the 1 GiB hard cap is unchanged"
        )
    free = shutil.disk_usage(output_root).free
    if free < int(args.max_new_bytes):
        raise SystemExit(
            f"REFUSED: free disk {free} < requested max-new-bytes {args.max_new_bytes}"
        )

    prov = _provenance(args)
    gate = provenance_gate(
        prov, allow_unverified=bool(args.allow_unverified_provenance)
    )
    prov["provenance_verified"] = bool(gate["provenance_verified"])
    prov["qualification"] = str(gate["qualification"])
    prov["override_used"] = bool(gate["override_used"])

    child_cmd = [
        args.python_exe,
        str(Path(__file__).resolve()),
        "--child",
        "--src-root", str(Path(args.src_root).resolve()),
        "--output-root", str(output_root),
        "--max-new-bytes", str(int(args.max_new_bytes)),
        "--pressure-after", str(int(args.pressure_after)),
    ]
    if args.installed_module:
        child_cmd.extend(
            ["--installed-module", str(Path(args.installed_module).resolve())]
        )
    if args.installed_src_root:
        child_cmd.extend(
            ["--installed-src-root", str(Path(args.installed_src_root).resolve())]
        )
    if args.allow_unverified_provenance:
        child_cmd.append("--allow-unverified-provenance")

    log_path = output_root / "child.log"
    started = time.time()
    cap_hit = threading.Event()
    with log_path.open("wb") as log_fh:
        proc = subprocess.Popen(
            child_cmd,
            cwd=str(output_root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def _watch() -> None:
            while proc.poll() is None:
                if _tree_bytes(output_root) > int(args.max_new_bytes):
                    cap_hit.set()
                    _kill_tree(proc)
                    return
                time.sleep(0.25)

        watcher = threading.Thread(target=_watch, name="m0-cap-watcher", daemon=True)
        watcher.start()
        timed_out = False
        try:
            proc.wait(timeout=int(args.max_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        finally:
            watcher.join(timeout=2)

    witness = _read_json(output_root / "witness.json")
    stage_record = _read_json(output_root / "stage.json") or {}
    new_bytes = _tree_bytes(output_root)
    verdict = evaluate_child_result(
        returncode=int(proc.returncode if proc.returncode is not None else -1),
        timed_out=timed_out or cap_hit.is_set(),
        witness=witness,
        new_bytes=new_bytes,
        max_new_bytes=int(args.max_new_bytes),
        stage_record=stage_record,
    )
    if cap_hit.is_set() and verdict["status"] != "CAP_EXCEEDED":
        verdict = {
            "status": "CAP_EXCEEDED",
            "exit": 4,
            "reason": "live cap watcher killed the child",
            "stage_reached": verdict.get("stage_reached"),
        }
    summary = {
        "schema": SCHEMA,
        "mission": MISSION,
        "mode": (witness or {}).get("mode", "UNKNOWN"),
        "status": verdict["status"],
        "reason": verdict["reason"],
        "timed_out": bool(timed_out),
        "max_seconds": int(args.max_seconds),
        "wall_seconds": round(time.time() - started, 3),
        "exit_code": proc.returncode,
        "stage_reached": verdict.get("stage_reached"),
        "stage_record": stage_record,
        "witness_path": str(output_root / "witness.json"),
        "provenance": prov,
        "provenance_verified": bool(prov.get("provenance_verified")),
        "gpu_qualified": bool((witness or {}).get("gpu_qualified", False)),
        "new_artifacts_bytes": new_bytes,
        "max_new_bytes": int(args.max_new_bytes),
        "estimate_new_bytes": estimate,
        "cap_exceeded": new_bytes > int(args.max_new_bytes) or cap_hit.is_set(),
        "child_log": str(log_path),
        "note": (
            "PASS requires child returncode==0 AND witness PASS AND "
            "classification EXPECTED_BASELINE_FAILURE AND no cap breach. "
            "TIMEOUT/NOT_RUN/CHILD_EXIT_NONZERO/PROCESS_FAILED are never a "
            "baseline reproduction or a product success."
        ),
    }
    _write_json(output_root / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(verdict["exit"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help="internal child mode")
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--installed-module", default=None)
    parser.add_argument("--installed-src-root", default=None)
    parser.add_argument("--allow-unverified-provenance", action="store_true")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allow-outside-zgprobe", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS_HARD)
    parser.add_argument("--max-new-bytes", type=int, default=MAX_NEW_BYTES_HARD)
    parser.add_argument("--pressure-after", type=int, default=2)
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.max_seconds = min(max(1, int(args.max_seconds)), MAX_SECONDS_HARD)
    args.max_new_bytes = min(int(args.max_new_bytes), MAX_NEW_BYTES_HARD)
    if not args.child and not args.installed_module:
        if not args.allow_unverified_provenance:
            raise SystemExit(
                "REFUSED: --installed-module is required; use "
                "--allow-unverified-provenance only for a labelled dry run"
            )
    if not args.child and args.installed_module and not Path(
        args.installed_module
    ).is_file():
        raise SystemExit(f"REFUSED: installed module not found: {args.installed_module}")
    if args.child:
        try:
            return _child(args)
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 - must be recorded, not hidden
            _write_json(
                Path(args.output_root) / "witness.json",
                {
                    "schema": SCHEMA,
                    "mission": MISSION,
                    "mode": "LOCAL_CONTROL_CPU" if os.name != "nt" else "WINDOWS_REAL_GPU",
                    "status": "FAIL",
                    "classification": "UNEXPECTED_FAILURE",
                    "gpu_qualified": False,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "stage": (_read_json(Path(args.output_root) / "stage.json") or {}),
                    "synthetic_pressure": True,
                    "provenance": _provenance(args),
                    "note": "Unexpected harness/product failure; not a baseline reproduction.",
                },
            )
            return 1
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
