"""ZM-ZEGRID-RESILIENT-RUN-RESUME M0 — baseline failure witness (CONTROL ONLY).

This is NOT a product-success test and NOT a GPU qualification.

It reproduces the baseline defect on the REAL product orchestration path
(``run_zegrid_mode`` -> ``_run_single`` -> bounded GPU producer/consumer ->
``_R55GpuFallback`` -> whole-batch CPU re-admission) with a real, small,
floor-respecting FITS corpus and real canonical cell computations. The GPU
device execution itself is REPLACED by the CPU canonical executor and is
therefore labelled ``CONTROL_ONLY_FAKE_GPU``: no CuPy device is used, and the
result cannot certify GPU behaviour or GPU science. The real-GPU witness is the
separate Windows harness under ``tools/zegrid_resume_m0/``.

Witnessed baseline facts (M0 EVIDENCE, preserved unchanged in the M0 report and
`.a2a-reports` artifacts):
* at least two cells are fully stacked and their result payloads exist;
* a synthetic available-memory drop is injected at the REAL product boundary
  (``available_memory_bytes``) only after those two completed cells;
* the next GPU unit is refused for host RAM and the run raises
  ``_R55GpuFallback``;
* in the PRE-M1 baseline every completed GPU result was discarded and the
  whole-batch CPU re-admission then failed with ``AdmissionExhausted``.

M1 CHANGED EXPECTATION (documented, not silently weakened): replaying this
harness on the M1 working code now asserts the OPPOSITE of the old discard:
the two accepted payloads are retained by the run registry through the refusal
and cleanup, the whole-batch CPU rerun is refused (no unqualified backend
mixing), and nothing is published. The historical discard behaviour remains
recorded in the immutable M0 report/Windows witness; it is not re-asserted as
current behaviour here. The dedicated M1 acceptance test
(``test_zegrid_resume_m1_conservation.py``) owns the full payload-fingerprint
(per-cell halo + reference pin) proof.

The test PASS means "the M0 baseline mechanism is still reproduced for the
expected reason AND M1 no longer erases the accepted work". It never means the
product succeeded, that resume exists, or that GPU science is qualified.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import gpu_readiness as zgr  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import resource_envelope as renv  # noqa: E402

GIB = 2 ** 30
MIB = 2 ** 20

# High enough for the initial dense GPU plan/admission; the injected floor is
# just low enough that the CPU whole-batch re-admission cannot fit.
RAW_HIGH = 16 * GIB
RAW_LOW = 700 * MIB
VRAM_FREE = 4 * GIB
VRAM_TOTAL = 8 * GIB

# Real scientific floors are used (nothing is lowered for the corpus).
CORPUS_SIZE = 512
CORPUS_FRAMES = 8
CORPUS_SHIFTS = (
    (0, 0), (96, 96), (192, 192), (288, 288),
    (0, 288), (288, 0), (192, 96), (96, 192),
)
LAYOUT = "2x2"
HALO = 8

ARTIFACT_ENV = "M0_ARTIFACT_DIR"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_array(value) -> str:
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


def _write_corpus(root: Path) -> tuple[Path, list[str], list[str]]:
    """Deterministic 8-frame, 512 px, overlapping TAN corpus.

    The 2x2 pinned layout keeps every Cell well above the real
    ``min_patch_area_px=50_000`` floor and every Cell receives at least three
    legitimate contributors, so no scientific floor is lowered for the witness.
    """
    inp = root / "in"
    inp.mkdir(parents=True)
    scale = 0.0012
    rng = np.random.default_rng(20261027)
    names: list[str] = []
    hashes: list[str] = []
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
        names.append(name)
        hashes.append(_sha256_bytes(path.read_bytes()))
    (inp / "stack_plan.csv").write_text(
        "file_path,exposure\n" + "".join(f"{n},10.0\n" for n in names),
        encoding="utf-8",
    )
    return inp, names, hashes


def _fake_gpu_seams(monkeypatch, memory_state: dict) -> None:
    """Control-only GPU seams: no CuPy import, no CUDA device, deterministic."""
    readiness = zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="control-only-no-device",
        vram_total_bytes=VRAM_TOTAL,
        vram_free_bytes=VRAM_FREE,
    )
    probe = {
        "available": True,
        "device": "control-only-no-device",
        "cupy_version": "14.1.1",
        "vram_total_bytes": VRAM_TOTAL,
        "vram_free_bytes": VRAM_FREE,
        "reason": None,
    }

    def _available() -> int:
        return int(memory_state["low"] and RAW_LOW or RAW_HIGH)

    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", lambda: dict(probe))
    monkeypatch.setattr(zz.zgr, "probe_gpu_readiness_safe", lambda **_kw: readiness)
    monkeypatch.setattr(
        zz.zgr, "read_vram_free_bytes", lambda device_index=0: int(VRAM_FREE)
    )
    monkeypatch.setattr(zz, "available_memory_bytes", _available)


def _run_once(
    *,
    base: Path,
    inp: Path,
    arm_after: int = 2,
) -> dict:
    """Run one REAL orchestration pass and return the bounded witness record."""
    sequence: list[dict] = []
    seen: list[str] = []
    memory = {"low": False}

    with pytest.MonkeyPatch.context() as mp:
        _fake_gpu_seams(mp, memory)

        # CONTROL-ONLY: the GPU device engine is the CPU canonical executor.
        # The orchestration, pipeline, leases, fallback and CPU re-admission are
        # the real product code; only the device execution is substituted.
        real_stream = zz.run_canonical_stack_streaming

        def _control_stream(
            provider, request, tile_size=zz.STREAM_TILE_SIZE, fixed=None,
            rejection_sink=None,
        ):
            return real_stream(
                provider,
                dataclasses.replace(request, backend="cpu"),
                tile_size=tile_size,
                fixed=fixed,
                rejection_sink=rejection_sink,
            )

        mp.setattr(zz, "run_canonical_stack_streaming", _control_stream)

        # The existing R55-E2 measurement seam is SHRINK-ONLY. Cap the admitted
        # prefetch at 0 to mirror the recorded M31-like minimum-prefetch case.
        mp.setattr(zz, "_R55E2_MEASURE_PREFETCH_CAP", 0)

        real_consume = zz._consume_stack_cell

        def _consume(prepared, **kw):
            out = real_consume(prepared, **kw)
            if isinstance(out, dict) and out.get("status") == "complete":
                unit_index = len(sequence)
                sequence.append(
                    {
                        "unit_index": int(unit_index),
                        "cell_id": str(out["cell_id"]),
                        "core_sha256": _hash_core(out["core"]),
                        "n_frames": int(out.get("n_frames", 0)),
                        "stack_s": round(float(out.get("stack_s", 0.0)), 6),
                        "status": "complete",
                    }
                )
                seen.append(str(out["cell_id"]))
                if len(seen) >= int(arm_after):
                    # Synthetic pressure is injected at the REAL product memory
                    # boundary, after the completed cells, not as real RAM load.
                    memory["low"] = True
            return out

        mp.setattr(zz, "_consume_stack_cell", _consume)

        out_dir = base / "out"
        exc: BaseException | None = None
        try:
            zz.run_zegrid_mode(
                str(inp),
                str(out_dir),
                zconfig=SimpleNamespace(
                    zegrid_scheduler_mode="active", zegrid_layout=LAYOUT
                ),
                zegrid_halo_px=HALO,
                use_gpu=True,
            )
        except BaseException as err:  # noqa: BLE001 - the failure IS the witness
            exc = err

    return {
        "sequence": sequence,
        "memory": memory,
        "exception": exc,
        "out_dir": out_dir,
        "cell_ids": seen,
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_dir() -> Path | None:
    raw = os.environ.get(ARTIFACT_ENV)
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _corpus_record(inp: Path, names: list[str], hashes: list[str]) -> dict:
    return {
        "size_px": CORPUS_SIZE,
        "frames": CORPUS_FRAMES,
        "files": [
            {"name": name, "sha256": digest}
            for name, digest in zip(names, hashes)
        ],
        "layout": LAYOUT,
        "halo_px": HALO,
        "scientific_floors": "REAL (auto_layout.ScientificFloors defaults; not lowered)",
        "storage_profile_requested": "auto",
    }


def test_m0_baseline_gpu_fallback_now_conserves_completed_cells(
    tmp_path, monkeypatch,
):
    inp, names, hashes = _write_corpus(tmp_path)
    first = _run_once(base=tmp_path / "run1", inp=inp)
    second = _run_once(base=tmp_path / "run2", inp=inp)

    # --- 1. At least two COMPLETE cells existed before the pressure. --------
    assert len(first["sequence"]) >= 2, first["sequence"]
    assert len(second["sequence"]) == len(first["sequence"])
    assert first["cell_ids"] == second["cell_ids"]
    deterministic_fields = ("unit_index", "cell_id", "core_sha256", "n_frames", "status")
    assert [
        {k: c[k] for k in deterministic_fields} for c in first["sequence"]
    ] == [
        {k: c[k] for k in deterministic_fields} for c in second["sequence"]
    ], (
        "the same deterministic witness must produce identical cell IDs and "
        "core hashes on a second run (wall times are volatile and excluded)"
    )
    assert [c["unit_index"] for c in first["sequence"]] == list(
        range(len(first["sequence"]))
    )
    assert all(c["status"] == "complete" for c in first["sequence"])
    assert all(len(c["core_sha256"]) == 64 for c in first["sequence"])
    assert all(c["n_frames"] >= 1 for c in first["sequence"])

    # --- 2. M1 CHANGED EXPECTATION: conserved refusal, not CPU rerun. -------
    exc = first["exception"]
    exc2 = second["exception"]
    assert isinstance(exc, zz._R55RunConservedResults), repr(exc)
    assert isinstance(exc2, zz._R55RunConservedResults), repr(exc2)
    assert type(exc2) is type(exc)
    assert second["memory"]["low"] is True
    text = str(exc)
    assert exc.reason_code == "R55_RUN_RESULTS_CONSERVED"
    assert "CPU reuse refused" in text
    # The historical AdmissionExhausted path is intentionally NOT reached now
    # (recomputing accepted cells would double the work / mix backends).
    assert not isinstance(exc, rcontract.AdmissionExhausted)
    assert not isinstance(exc, MemoryError)
    cause_chain = []
    cursor = exc
    while cursor is not None:
        cause_chain.append(cursor)
        cursor = cursor.__cause__
    assert not any(isinstance(e, MemoryError) for e in cause_chain)

    # --- 3. The REAL GPU fallback artifact records retention, not discard. --
    failure_path = first["out_dir"] / "zegrid_gpu_failure.json"
    assert failure_path.is_file()
    failure = _load_json(failure_path)
    assert failure["reason_code"] == renv.REASON_REFUSE_HOST_RAM
    pipeline = failure["pipeline"]
    assert pipeline["units_consumed"] == len(first["sequence"])
    assert pipeline["units_total"] >= len(first["sequence"]) + 1
    assert pipeline["units_released"] == pipeline["units_produced"]
    assert pipeline["units_produced"] <= pipeline["units_total"]
    assert pipeline["lifecycle"] == "caller_owned_consumer"
    assert pipeline["cancelled"] is True
    assert pipeline["cancel_reason"].startswith("consumer_error:_R55GpuFallback")
    assert pipeline["cleanup_failures"] == []
    # Minimum-prefetch scenario (the existing shrink-only measurement seam).
    assert pipeline["prefetch_units"] == 0
    assert pipeline["queue_max"] == 1
    # The ledger is balanced and owns nothing after the cancel.
    ledger = pipeline["ledger"]
    assert ledger["balanced"] is True
    assert ledger["host_leases"] == ledger["device_leases"] == ledger["disk_leases"] == 0
    assert ledger["host_used_bytes"] == 0
    assert ledger["device_used_bytes"] == 0
    assert ledger["disk_used_bytes"] == 0

    units = failure["tile_state"]["units"]
    completed = [u for u in units if u["status"] == "completed"]
    assert len(completed) == len(first["sequence"])
    # M1: every accepted completed unit is retained; only unaccepted work is
    # discarded. (The pre-M1 artifact asserted discarded=True for all.)
    assert all(u["retained"] is True for u in completed)
    assert all(u["discarded"] is False for u in completed)
    refused = [u for u in units if u["status"] == "refused"]
    assert len(refused) == 1
    assert refused[0]["action"] == "fallback_cpu"
    assert refused[0]["discarded"] is True
    assert refused[0]["retained"] is False
    # The refused boundary is the exact memory floor injected at the boundary.
    last = failure["revalidations_tail"][-1]
    assert last["host"]["raw_available_bytes"] == RAW_LOW
    assert last["host"]["budget_bytes"] < RAW_LOW
    assert last["vram"]["free_bytes"] == VRAM_FREE
    assert last["vram"]["reason_code"] == "R55_VRAM_CONTINUE"
    attempts = last["shrink_attempts"]
    assert attempts, "the RUN must walk the admitted tile ladder on host pressure"
    assert all(a["device_ok"] is True for a in attempts)
    assert all(a["host_ok"] is False for a in attempts)
    # The whole declared ladder was walked and still failed HOST RAM, exactly
    # the M31 shape (admissible VRAM, no host-admissible rung left).
    assert [a["tile_px"] for a in attempts] == sorted(
        {a["tile_px"] for a in attempts}, reverse=True
    )

    # --- 4. Nothing is published; accepted work lives in the M1 registry. ---
    out_dir = first["out_dir"]
    assert not (out_dir / "mosaic_grid.fits").exists()
    assert not (out_dir / "mosaic_grid_coverage.fits").exists()
    assert not (out_dir / "zegrid_manifest.json").exists()
    # The run cache is empty (directories may remain; no staged payloads do).
    cache_root = out_dir / "__zegrid_cache__"
    if cache_root.exists():
        assert [p.name for p in cache_root.rglob("*")] == []
    for cell in first["sequence"]:
        assert cell["cell_id"] not in {
            p.name for p in out_dir.rglob("*")
        }
    registry = exc.result_registry
    assert registry is not None
    assert registry.counts()["accepted"] == len(first["sequence"])
    assert registry.accepted_cell_ids() == first["cell_ids"]
    assert registry.pending_cell_ids()
    assert registry.retained_bytes > 0
    # Explicit disposal is the one ownership hand-off (M1, in-memory only).
    summary = registry.dispose()
    assert summary["released_cells"] == len(first["sequence"])
    assert summary["already_disposed"] is False
    assert registry.dispose()["already_disposed"] is True
    # Keep the second run's registry explicitly disposed too (deterministic
    # witness comparison happened before this point).
    exc2.result_registry.dispose()

    # --- 5. Record the reproducible witness for the durable M0 report. ------
    record = {
        "schema": "zemosaic-zegrid-resume-m0-baseline-witness-v1",
        "mission": "ZM-ZEGRID-RESILIENT-RUN-RESUME-M0-20261027",
        "mode": "CONTROL_ONLY_FAKE_GPU",
        "gpu_qualified": False,
        "gpu_note": (
            "Device execution substituted by the real CPU canonical executor; "
            "no CuPy device and no GPU science. The real-GPU witness is the "
            "separate Windows harness. This record must never be read as GPU "
            "qualification. The 16 GiB high observation is a synthetic control "
            "budget used only to drive the product decision path; it is NOT a "
            "claim that this host has 16 GiB available. The Windows harness "
            "clamps its high observation by LIVE available RAM."
        ),
        "baseline_failure_reproduced": True,
        "m1_expectation_change": {
            "from": "all completed GPU results discarded + AdmissionExhausted",
            "to": (
                "accepted payloads retained by the run registry; whole-batch "
                "CPU reuse refused (no numeric parity qualification); no "
                "publication"
            ),
            "historical_evidence_preserved_in": (
                "ZEGRID_RESILIENT_RUN_RESUME_M0_20261027.md and the Windows "
                "m0_witness_out artifacts"
            ),
            "full_payload_fingerprint_test": (
                "tests/test_zegrid_resume_m1_conservation.py"
            ),
        },
        "retained_after_refusal": {
            "accepted_cells": len(first["cell_ids"]),
            "cell_ids": list(first["cell_ids"]),
            "pending_cells": len(registry.pending_cell_ids()),
            "retained_bytes": int(summary["released_retained_bytes"]),
            "disposed_after_record": True,
        },
        "product_success": False,
        "corpus": _corpus_record(inp, names, hashes),
        "completed_before_failure": first["sequence"],
        "synthetic_pressure": {
            "boundary": "zemosaic_zegrid_mode.available_memory_bytes",
            "armed_after_completed_cells": 2,
            "raw_available_high_bytes": RAW_HIGH,
            "raw_available_low_bytes": RAW_LOW,
            "note": "valid-shaped synthetic trajectory, not physical RAM pressure",
        },
        "gpu_fallback": {
            "exception_type": failure["exception"]["type"],
            "reason_code": failure["reason_code"],
            "detail": failure["detail"],
            "tile_px": failure["tile_state"]["tile_px"],
            "prefetch_units": pipeline["prefetch_units"],
            "units_total": pipeline["units_total"],
            "units_consumed": pipeline["units_consumed"],
            "units_released": pipeline["units_released"],
        },
        "cpu_reuse_refused": {
            "exception_type": type(exc).__name__,
            "reason_code": exc.reason_code,
            "message": text,
            "registry_snapshot": getattr(exc, "record", {}).get(
                "result_registry"
            ),
        },
        "publication": {
            "fits": False,
            "coverage": False,
            "manifest": False,
            "assembly": False,
        },
        "cleanup": {
            "pipeline_cleanup_failures": pipeline["cleanup_failures"],
            "cache_staging_remaining": False,
        },
        "no_oom": True,
        "test_command": (
            "python3 -m pytest -q "
            "tests/test_zegrid_resume_m0_baseline_witness.py"
        ),
        "m1_note": (
            "This replay is the historical M0 mechanism witness updated for M1; "
            "it is not the M1 acceptance test by itself."
        ),
    }
    artifact = _artifact_dir()
    if artifact is not None:
        (artifact / "baseline_witness_control_only.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
