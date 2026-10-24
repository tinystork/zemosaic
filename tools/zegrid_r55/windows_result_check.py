#!/usr/bin/env python3
"""R55-E3 Windows beta result checker (read-only, fail-closed).

This tool validates a *captured* Windows beta result directory. It never runs
the pipeline, never probes hardware, never writes into the result directory and
never claims hardware qualification. Before a real witness is imported the
verdict is ``WINDOWS_NOT_RUN``.

Expected result directory (produced by the Windows witness guide):

    windows_result.json
    zegrid_manifest.json
    mosaic_grid.fits
    mosaic_grid_coverage.fits
    run.log

``windows_result.json`` contract (``zemosaic-r55e3-windows-result-v1``)::

    {
      "schema": "zemosaic-r55e3-windows-result-v1",
      "status": "DONE",
      "os": {"name": "Windows", "version": "...", "arch": "AMD64"},
      "gpu": {"name": "...", "driver": "...", "runtime": "...",
              "vram_total_bytes": 8589934592,
              "vram_free_bytes_at_start": 7516192768},
      "config": {"zegrid_mode": true, "use_gpu": true,
                 "zegrid_scheduler_mode": "active", "zegrid_halo_px": 64},
      "run": {"corpus_frames": 12, "peak_rss_bytes": 2147483648,
              "ram_budget_bytes": 34359738368},
      "fallback": {"used": false, "reason_code": null},
      "parity_reference": {"mosaic_grid.fits": "<sha256>",
                           "mosaic_grid_coverage.fits": "<sha256>"},
      "artifacts": {"zegrid_manifest.json": "<sha256>", "run.log": "<sha256>"}
    }

Any missing file, absent key, wrong token, bad hash or incoherent manifest is a
hard failure (exit 3). A structurally valid result is reported as
``WINDOWS_RESULT_VALID_UNQUALIFIED`` and still carries ``qualified=false``:
only a human/architect review may turn it into a qualification.

Usage:
    python tools/zegrid_r55/windows_result_check.py --result DIR [--json-out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "zemosaic-r55e3-windows-result-v1"
STATUS_NOT_RUN = "WINDOWS_NOT_RUN"
STATUS_INVALID = "WINDOWS_RESULT_INVALID"
STATUS_VALID_UNQUALIFIED = "WINDOWS_RESULT_VALID_UNQUALIFIED"
VRAM_POLICY_PERCENT = 70
REQUIRED_FILES = (
    "windows_result.json",
    "zegrid_manifest.json",
    "mosaic_grid.fits",
    "mosaic_grid_coverage.fits",
    "run.log",
)
# ``windows_result.json`` cannot contain its own hash; every other required
# artifact must be hashed.
REQUIRED_ARTIFACT_HASHES = tuple(
    name for name in REQUIRED_FILES if name != "windows_result.json"
)


class CheckError(Exception):
    """A closed, actionable validation failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _int(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckError(f"{name} must be a plain integer, got {value!r}")
    if value < minimum:
        raise CheckError(f"{name} must be >= {minimum}, got {value}")
    return value


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckError(f"cannot read {path.name}: {type(exc).__name__}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object")
    return payload


def validate_result(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        return {
            "schema": "zemosaic-r55e3-windows-check-v1",
            "status": STATUS_NOT_RUN,
            "qualified": False,
            "reason": f"result directory not found: {root}",
        }

    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        return {
            "schema": "zemosaic-r55e3-windows-check-v1",
            "status": STATUS_NOT_RUN,
            "qualified": False,
            "reason": "missing required artifact(s): " + ", ".join(missing),
        }

    try:
        result = _load_json(root / "windows_result.json")
        _require(result.get("schema") == SCHEMA, "unsupported windows_result schema")
        _require(result.get("status") == "DONE", "windows_result status must be DONE")

        os_info = result.get("os")
        _require(isinstance(os_info, dict), "os must be an object")
        _require(str(os_info.get("name", "")).lower() == "windows", "os.name must be Windows")
        _require(bool(str(os_info.get("version", "")).strip()), "os.version is required")
        _require(bool(str(os_info.get("arch", "")).strip()), "os.arch is required")

        gpu = result.get("gpu")
        _require(isinstance(gpu, dict), "gpu must be an object")
        for key in ("name", "driver", "runtime"):
            _require(bool(str(gpu.get(key, "")).strip()), f"gpu.{key} is required")
        vram_total = _int(gpu.get("vram_total_bytes"), "gpu.vram_total_bytes", minimum=1)
        vram_free = _int(
            gpu.get("vram_free_bytes_at_start"),
            "gpu.vram_free_bytes_at_start",
            minimum=1,
        )
        _require(vram_free <= vram_total, "gpu.vram_free_bytes_at_start exceeds total")

        config = result.get("config")
        _require(isinstance(config, dict), "config must be an object")
        _require(config.get("zegrid_mode") is True, "config.zegrid_mode must be true")
        _require(config.get("use_gpu") is True, "config.use_gpu must be true")
        _require(
            config.get("zegrid_scheduler_mode") == "active",
            "config.zegrid_scheduler_mode must be the exact token 'active'",
        )
        halo_px = _int(config.get("zegrid_halo_px"), "config.zegrid_halo_px", minimum=1)

        run = result.get("run")
        _require(isinstance(run, dict), "run must be an object")
        _int(run.get("corpus_frames"), "run.corpus_frames", minimum=1)
        peak_rss = _int(run.get("peak_rss_bytes"), "run.peak_rss_bytes", minimum=1)
        if run.get("ram_budget_bytes") is not None:
            ram_budget = _int(run.get("ram_budget_bytes"), "run.ram_budget_bytes", minimum=1)
            _require(peak_rss <= ram_budget, "run.peak_rss_bytes exceeds run.ram_budget_bytes")

        fallback = result.get("fallback")
        _require(isinstance(fallback, dict), "fallback must be an object")
        _require(isinstance(fallback.get("used"), bool), "fallback.used must be a bool")
        fallback_used = bool(fallback["used"])
        if fallback_used:
            _require(bool(str(fallback.get("reason_code", "")).strip()),
                     "fallback.reason_code is required when fallback.used is true")

        parity_reference = result.get("parity_reference")
        _require(isinstance(parity_reference, dict), "parity_reference must be an object")
        for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
            _require(isinstance(parity_reference.get(name), str),
                     f"parity_reference.{name} is required")

        artifacts = result.get("artifacts")
        _require(isinstance(artifacts, dict), "artifacts must be an object")
        for name in REQUIRED_ARTIFACT_HASHES:
            _require(isinstance(artifacts.get(name), str),
                     f"artifacts.{name} is required")

        # Byte-level truth: hashes are recomputed, never trusted.
        observed = {name: _sha256(root / name) for name in REQUIRED_FILES}
        for name in REQUIRED_ARTIFACT_HASHES:
            _require(
                observed[name] == artifacts[name],
                f"artifact hash mismatch for {name}",
            )
        for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
            _require(
                observed[name] == parity_reference[name],
                f"parity mismatch for {name}: active run differs from the legacy reference",
            )

        manifest = _load_json(root / "zegrid_manifest.json")
        block = manifest.get("r55_scheduler")
        _require(isinstance(block, dict), "manifest r55_scheduler block is missing")
        _require(block.get("mode") == "active", "manifest scheduler mode must be active")
        _require(block.get("requested_mode") == "active",
                 "manifest requested_mode must be active")
        _require(block.get("status") == "executed", "manifest scheduler status must be executed")
        identity = block.get("identity")
        _require(isinstance(identity, dict), "manifest scheduler identity is missing")
        _int(identity.get("halo_px"), "identity.halo_px", minimum=1)
        _require(int(identity["halo_px"]) == halo_px, "identity halo differs from config halo")

        mapping = block.get("mapping")
        _require(isinstance(mapping, dict), "manifest scheduler mapping is missing")
        _require(mapping.get("backend") == "gpu", "manifest mapping.backend must be gpu")
        _require(_int(mapping.get("units_in_flight"), "mapping.units_in_flight", minimum=1) == 1,
                 "active GPU mapping must be single-owner (units_in_flight=1)")
        prefetch = _int(mapping.get("prefetch_units"), "mapping.prefetch_units")
        _require(prefetch <= 4, "prefetch_units is outside the bounded E2 ladder")

        gpu_execution = block.get("gpu_execution")
        if gpu_execution is not None or not fallback_used:
            _require(isinstance(gpu_execution, dict), "manifest gpu_execution is missing")
            _require(
                gpu_execution.get("qualified") is True,
                "manifest gpu_execution is not qualified",
            )
            policy_percent = _int(
                gpu_execution.get("percent"), "gpu_execution.percent", minimum=1
            )
            _require(policy_percent <= VRAM_POLICY_PERCENT,
                     "gpu_execution.percent exceeds the 70% VRAM policy")
            device_peak = gpu_execution.get("device_peak_bytes")
            if device_peak is not None:
                device_peak = _int(device_peak, "gpu_execution.device_peak_bytes", minimum=1)
                budget = (vram_free * VRAM_POLICY_PERCENT) // 100
                _require(device_peak <= budget,
                         "device_peak_bytes exceeds the 70% VRAM budget")

        telemetry = manifest.get("r55_run_telemetry")
        _require(isinstance(telemetry, dict), "manifest r55_run_telemetry is missing")
        pipeline = telemetry.get("gpu_pipeline")
        _require(isinstance(pipeline, dict), "manifest gpu_pipeline telemetry is missing")
        ledger = pipeline.get("ledger")
        _require(isinstance(ledger, dict), "gpu_pipeline ledger is missing")
        _require(ledger.get("balanced") is True, "gpu_pipeline ledger is not balanced")
        _require(pipeline.get("threads_joined") is True, "gpu_pipeline threads are not joined")
        _require(
            _int(pipeline.get("queue_max"), "gpu_pipeline.queue_max", minimum=1)
            == prefetch + 1,
            "gpu_pipeline.queue_max must equal prefetch_units + 1",
        )

        gpu_block = manifest.get("gpu")
        _require(isinstance(gpu_block, dict), "manifest gpu block is missing")
        if fallback_used:
            _require(gpu_block.get("final_backend") == "cpu",
                     "fallback result must publish the CPU backend")
            _require(gpu_block.get("actually_used") is False,
                     "fallback result must not claim GPU cell execution")
            _require(pipeline.get("cancelled") is True,
                     "fallback gpu_pipeline must be cancelled")
            gpu_fallback = block.get("gpu_fallback")
            _require(isinstance(gpu_fallback, dict), "manifest gpu_fallback is missing")
            _require(
                str(gpu_fallback.get("reason_code", "")).strip()
                == str(fallback["reason_code"]).strip(),
                "fallback reason differs between result and manifest",
            )
        else:
            _require(gpu_block.get("final_backend") == "gpu",
                     "non-fallback result must publish the GPU backend")
            _require(gpu_block.get("actually_used") is True,
                     "non-fallback result must claim GPU cell execution")
            _require(pipeline.get("cancelled") is False,
                     "non-fallback gpu_pipeline must not be cancelled")

        return {
            "schema": "zemosaic-r55e3-windows-check-v1",
            "status": STATUS_VALID_UNQUALIFIED,
            "qualified": False,
            "reason": "structurally valid Windows beta result; hardware qualification "
                      "remains a separate human/architect decision",
            "scheduler_mode": "active",
            "backend": mapping.get("backend"),
            "prefetch_units": prefetch,
            "queue_max": pipeline.get("queue_max"),
            "fallback_used": fallback_used,
            "vram_total_bytes": vram_total,
            "vram_free_bytes_at_start": vram_free,
            "peak_rss_bytes": peak_rss,
        }
    except CheckError as exc:
        return {
            "schema": "zemosaic-r55e3-windows-check-v1",
            "status": STATUS_INVALID,
            "qualified": False,
            "reason": str(exc),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="captured Windows result directory")
    parser.add_argument("--json-out", default=None, help="optional verdict JSON path")
    args = parser.parse_args(argv)

    verdict = validate_result(Path(args.result))
    text = json.dumps(verdict, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)

    if verdict["status"] == STATUS_VALID_UNQUALIFIED:
        return 0
    if verdict["status"] == STATUS_NOT_RUN:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
