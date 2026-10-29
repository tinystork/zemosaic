"""ZM-ZEGRID-M6 — headless ZeGrid CLI (no Qt import).

This is the product entry point for running ZeGrid without the Qt launcher::

    python -m zemosaic.zegrid_cli --input INPUT --output OUTPUT [...]

Contract
--------
* It calls the REAL :func:`zemosaic.zemosaic_zegrid_mode.run_zegrid_mode`
  path; there is no simulator, no harness and no fake success.
* Exit codes are explicit and stable:
    - ``0`` success: the science FITS + manifest were published;
    - ``2`` refused / NOT_RUN: invalid or contradictory options, missing input,
      a capacity refusal raised by the engine, or an unsupported platform;
    - ``3`` error / cooperative stop: any unexpected failure, or a stop that was
      honoured at a safe boundary (``STOPPED_RESUMABLE``).
* ``SIGTERM``/``SIGINT`` request a COOPERATIVE stop through the run's
  ``RunResultRegistry`` (the CLI owns the stop channel passed to the engine).
  The current safe boundary finishes; durable cell commits stay intact and no
  publication is attempted. A bounded ``--stop-grace-seconds`` watchdog then
  exits non-zero with a clear message if the engine cannot reach a boundary.
  Windows signal delivery is **NOT_RUN** (documented, not claimed): the code
  path is portable, but only the POSIX behaviour is qualified.
* Every option fails closed: unknown or invalid values are refused, never
  silently coerced.

CPU only by design in this mission (``use_gpu=False``); GPU is out of scope.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_ERROR = 3

SUMMARY_SCHEMA = "zegrid-cli-summary-v1"
DEFAULT_STOP_GRACE_SECONDS = 120.0

_LAYOUT_RE = re.compile(r"^(\d+)[x*,](\d+)$")


class RefusedOption(ValueError):
    """An option or option combination that the CLI refuses up front."""


def _positive_int(text: str) -> int:
    try:
        value = int(text, 10)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a positive number, got {text!r}")
    if not (value > 0.0) or value != value or value == float("inf"):
        raise argparse.ArgumentTypeError(
            f"expected a finite positive number, got {text!r}"
        )
    return value


def _layout(text: str) -> str:
    value = str(text).strip().lower()
    match = _LAYOUT_RE.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"expected a pinned layout like '2x2', got {text!r}"
        )
    nx, ny = int(match.group(1)), int(match.group(2))
    if nx < 1 or ny < 1:
        raise argparse.ArgumentTypeError(f"layout cells must be >= 1, got {text!r}")
    return f"{nx}x{ny}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m zemosaic.zegrid_cli",
        description=(
            "Run ZeGrid headlessly (no Qt). CPU only. Exit codes: "
            "0 success, 2 refused/NOT_RUN, 3 error/cooperative stop."
        ),
    )
    parser.add_argument("--input", required=True, help="folder of input FITS frames")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument(
        "--workers", type=_positive_int, default=1,
        help="CPU workers (default 1)",
    )
    parser.add_argument(
        "--storage-profile", choices=("auto", "dense", "bounded"), default="auto",
        help="durable storage profile (product default auto: dense first, one "
             "bounded retry only on a qualified host-RAM refusal)",
    )
    parser.add_argument(
        "--durable-store-dir", default=None,
        help="root of the durable per-cell store (enables durable commits)",
    )
    parser.add_argument(
        "--durable-store-budget-bytes", type=_positive_int, default=None,
        help="positive durable-store byte budget (requires --durable-store-dir)",
    )
    parser.add_argument(
        "--resume-mode", choices=("never", "auto"), default="never",
        help="launch-time resume policy (default never)",
    )
    parser.add_argument(
        "--resume-run-id", default=None,
        help="explicit run generation when --resume-mode=auto",
    )
    parser.add_argument(
        "--resource-recovery", action="store_true", default=False,
        help="opt in to host-resource recovery at wave boundaries (default off)",
    )
    parser.add_argument(
        "--resource-recovery-deadline-s", type=_positive_float, default=None,
        help=(
            "positive recovery deadline in seconds (requires --resource-recovery). "
            "Without it the wait has no deadline (documented)."
        ),
    )
    parser.add_argument(
        "--halo-px", type=int, default=None,
        help="anti-checkerboard halo in output px (8/16/32/64/128/256; default 64)",
    )
    parser.add_argument(
        "--gauge-mode", choices=("auto", "global", "per_cell"), default=None,
        help="per-cell photometric gauge mode (default global)",
    )
    parser.add_argument(
        "--layout", type=_layout, default=None,
        help="pinned layout 'NXxNY' (default: RAM-adaptive)",
    )
    parser.add_argument(
        "--json-summary", default=None,
        help="optional path for a machine-readable JSON summary",
    )
    parser.add_argument(
        "--stop-grace-seconds", type=_positive_float,
        default=DEFAULT_STOP_GRACE_SECONDS,
        help=(
            "bounded grace after SIGTERM/SIGINT before the process exits "
            f"non-zero (default {DEFAULT_STOP_GRACE_SECONDS:.0f}s)"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help=(
            "explicitly allow replacing a previous ZeGrid publication found in "
            "--output (default: refuse). Never deletes foreign files."
        ),
    )
    return parser


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


_ZEGRID_OUTPUT_FILE_RE = re.compile(r"^mosaic_grid.*\.fits$", re.IGNORECASE)


def _publication_pointer(output_dir: Path) -> dict | None:
    """Return the newest durable publication pointer, or None."""
    publications = output_dir / ".zegrid_publications"
    if not publications.is_dir():
        return None
    pointers = sorted(publications.glob("*/CURRENT.json"))
    if not pointers:
        return None
    pointer = pointers[-1]
    payload: dict = {}
    try:
        loaded = json.loads(pointer.read_bytes().decode("utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        pass
    return {
        "path": str(pointer),
        "run_id": payload.get("run_id"),
        "generation": payload.get("generation"),
    }


def _dir_publication_evidence(root: Path) -> list[str]:
    """Publication evidence directly inside one directory (no recursion)."""
    evidence: list[str] = []
    try:
        if (root / "zegrid_manifest.json").is_file():
            evidence.append("zegrid_manifest.json")
        for path in sorted(root.glob("mosaic_grid*.fits")):
            if path.is_file():
                evidence.append(path.name)
        if (root / ".zegrid_publications").is_dir():
            evidence.append(".zegrid_publications/")
    except OSError:
        pass
    return evidence


def _manifest_is_zegrid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
        return isinstance(payload, dict) and payload.get("engine") == "zegrid"
    except Exception:
        return False


def _detect_previous_publication(output_dir: Path) -> dict:
    """Detect a previous ZeGrid publication and foreign content.

    Publication evidence (C1 contract): ``zegrid_manifest.json``,
    ``mosaic_grid*.fits``, or a ``.zegrid_publications/`` root/pointer.

    REWORK-2: mount-segregated runs publish into ``grid_EQ/`` and
    ``grid_ALTZ/`` subdirectories, so direct child directories are inspected
    too. A child that holds a publication is ZeGrid-owned (not foreign) and its
    contents are named in the refusal. ``zegrid_run.log`` and
    ``__zegrid_cache__`` are ours but are NOT publication evidence on their own
    (a failed first attempt must not block a retry).
    """
    evidence: list[str] = []
    foreign: list[str] = []
    zegrid = False
    if not output_dir.is_dir():
        return {"found": False, "zegrid": False, "evidence": [], "foreign": []}

    root_evidence = _dir_publication_evidence(output_dir)
    evidence.extend(root_evidence)
    if "zegrid_manifest.json" in root_evidence:
        zegrid = bool(_manifest_is_zegrid(output_dir / "zegrid_manifest.json"))
    if any(name.startswith("mosaic_grid") for name in root_evidence):
        zegrid = True
    if ".zegrid_publications/" in root_evidence:
        zegrid = True

    try:
        entries = sorted(output_dir.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        name = entry.name
        if name in {
            "zegrid_manifest.json", "zegrid_run.log",
            "__zegrid_cache__", ".zegrid_publications",
        }:
            continue
        if entry.is_file() and _ZEGRID_OUTPUT_FILE_RE.match(name):
            continue
        if entry.is_dir():
            child = _dir_publication_evidence(entry)
            if child:
                evidence.extend(f"{name}/{item}" for item in child)
                zegrid = True
                continue
        foreign.append(name)

    return {
        "found": bool(evidence),
        "zegrid": bool(zegrid),
        "evidence": sorted(set(evidence)),
        "foreign": sorted(foreign),
    }


def validate_options(args) -> dict:
    """Cross-option validation; raises :class:`RefusedOption` on any refusal."""
    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        raise RefusedOption(f"input folder does not exist or is not a folder: {input_dir}")
    if not any(input_dir.glob("*.fits")) and not any(input_dir.glob("*.fit")):
        raise RefusedOption(f"input folder contains no FITS frames: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise RefusedOption(f"output path exists and is not a folder: {output_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise RefusedOption("output folder must differ from the input folder")
    if _is_within(output_dir, input_dir):
        raise RefusedOption("output folder must not be inside the input folder")

    store_dir = None
    if args.durable_store_dir:
        store_dir = Path(args.durable_store_dir).expanduser()
        if _is_within(store_dir, output_dir):
            raise RefusedOption(
                "durable store dir must not be inside the output folder "
                "(publication and store must not nest)"
            )
        if store_dir.resolve() == output_dir.resolve():
            raise RefusedOption("durable store dir must differ from the output folder")
    if args.durable_store_budget_bytes is not None and store_dir is None:
        raise RefusedOption(
            "--durable-store-budget-bytes requires --durable-store-dir"
        )
    if args.resume_mode == "auto" and store_dir is None:
        raise RefusedOption("--resume-mode=auto requires --durable-store-dir")
    if args.resume_run_id and args.resume_mode != "auto":
        raise RefusedOption("--resume-run-id requires --resume-mode=auto")
    if args.resource_recovery_deadline_s is not None and not args.resource_recovery:
        raise RefusedOption(
            "--resource-recovery-deadline-s requires --resource-recovery"
        )

    # ZM-ZEGRID-M6 REWORK-1 (C1): a reused output folder is never overwritten
    # silently. Resuming and overwriting are contradictory by construction.
    if args.overwrite and args.resume_mode == "auto":
        raise RefusedOption(
            "--overwrite refuses --resume-mode=auto: a resume must never "
            "replace a previous publication"
        )
    previous = _detect_previous_publication(output_dir)
    if previous["found"] and not args.overwrite:
        raise RefusedOption(
            "output folder already contains a previous ZeGrid publication "
            f"({', '.join(previous['evidence'])}). Refusing to overwrite it "
            "silently; choose a new --output or pass --overwrite explicitly"
        )
    if args.overwrite and previous["foreign"]:
        raise RefusedOption(
            "--overwrite refused: the output folder holds content that is not "
            f"a ZeGrid publication ({', '.join(previous['foreign'])}); those "
            "files are not ours to destroy"
        )

    halo_px = None
    if args.halo_px is not None:
        from .zegrid_halo import ZeGridHaloError, resolve_zegrid_halo_px

        try:
            halo_px = resolve_zegrid_halo_px(args.halo_px)
        except ZeGridHaloError as exc:
            raise RefusedOption(str(exc)) from exc

    summary_path = None
    if args.json_summary:
        summary_path = Path(args.json_summary).expanduser()
        if summary_path.exists() and summary_path.is_dir():
            raise RefusedOption(
                f"--json-summary points at a folder, not a file: {summary_path}"
            )

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "store_dir": store_dir,
        "summary_path": summary_path,
        "halo_px": halo_px,
        "previous": previous,
        "previous_pointer": _publication_pointer(output_dir),
    }


def _peak_rss_kib() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss // 1024)
        except Exception:
            return None


def _count_durable_commits(store_dir: Path) -> tuple[int, int]:
    """Return (commit manifests found, manifests that parse as JSON)."""
    found = 0
    valid = 0
    try:
        for path in store_dir.rglob("commits/*.json"):
            found += 1
            try:
                json.loads(path.read_bytes().decode("utf-8"))
                valid += 1
            except Exception:
                pass
    except Exception:
        pass
    return found, valid


def _write_summary(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_publication(output_dir: Path):
    """Resolve the published science FITS + manifest from disk evidence.

    ``run_zegrid_mode`` returns ``None`` on the product path, so the CLI reads
    the authoritative artifacts instead of trusting a return value:
    * no durable store: ``<output>/zegrid_manifest.json`` + its declared outputs;
    * durable store: the ``CURRENT.json`` pointer selects the published
      generation under ``<output>/.zegrid_publications/<run>/``.
    """
    base = output_dir
    publications = output_dir / ".zegrid_publications"
    if publications.is_dir():
        pointers = sorted(publications.glob("*/CURRENT.json"))
        if pointers:
            pointer = pointers[-1]
            try:
                payload = json.loads(pointer.read_bytes().decode("utf-8"))
                generation = str(payload.get("generation") or "")
                run_dir = pointer.parent
                if generation and (run_dir / generation).is_dir():
                    base = run_dir / generation
            except Exception:
                pass
    manifest_path = base / "zegrid_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except Exception:
        return None, manifest_path
    outputs = manifest.get("outputs")
    science_name = outputs.get("science") if isinstance(outputs, dict) else None
    if not science_name:
        return None, manifest_path
    science_path = (base / str(science_name)).resolve()
    if not science_path.exists():
        return None, manifest_path
    return science_path, manifest_path


def _publication_is_fresh(manifest_path: Path, started_ns: int) -> bool:
    """True only when this run wrote (or rewrote) the manifest.

    A reused output folder can still contain the manifest of an earlier run.
    Reporting that as this run's success would be a fake COMPLETE. The engine
    writes the manifest inside ``run_zegrid_mode``, i.e. after ``started_ns``,
    so a manifest whose mtime predates the run is stale by construction.
    """
    try:
        stat = manifest_path.stat()
    except OSError:
        return False
    return int(stat.st_mtime_ns) >= int(started_ns)


def _write_set_fingerprint(output_dir: Path) -> dict:
    """Stat fingerprint of the files the engine may write for this run.

    Covers the output root and the mount-segregated children (``grid_EQ``,
    ``grid_ALTZ``). Used only as a defence-in-depth assertion after a run: with
    no ``--overwrite``, NO pre-existing file may change. If one did, the
    run is reported as an error even if publication resolution looked fine.
    """
    roots = [output_dir]
    try:
        roots.extend(
            child for child in output_dir.iterdir()
            if child.is_dir() and child.name in {"grid_EQ", "grid_ALTZ"}
        )
    except OSError:
        pass
    fingerprint: dict[str, tuple[int, int]] = {}
    for root in roots:
        try:
            for path in root.iterdir():
                if not path.is_file():
                    continue
                stat = path.stat()
                fingerprint[str(path.relative_to(output_dir))] = (
                    int(stat.st_mtime_ns), int(stat.st_size)
                )
        except OSError:
            continue
    return fingerprint


def _build_config(args, resolved) -> SimpleNamespace:
    from .zemosaic_config import DEFAULT_CONFIG, validate_zegrid_run_policy

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            # The cooperative-stop path requires the ACTIVE scheduler (the
            # product default). It is set explicitly so a stale config value
            # can never silently disable the registry.
            "zegrid_scheduler_mode": "active",
            "zegrid_halo_px": (
                resolved["halo_px"]
                if resolved["halo_px"] is not None
                else DEFAULT_CONFIG["zegrid_halo_px"]
            ),
            "zegrid_layout": args.layout,
            "zegrid_cell_gauge_mode": args.gauge_mode,
            "zegrid_resume_mode": args.resume_mode,
            "zegrid_resume_run_id": args.resume_run_id,
            "zegrid_durable_store_dir": (
                str(resolved["store_dir"]) if resolved["store_dir"] else None
            ),
            "zegrid_durable_store_budget_bytes": args.durable_store_budget_bytes,
            "zegrid_resource_recovery_enabled": bool(args.resource_recovery),
            "zegrid_resource_recovery_deadline_s": args.resource_recovery_deadline_s,
        }
    )
    # Fail closed on any invalid value before the engine is even imported.
    validate_zegrid_run_policy(cfg)
    return SimpleNamespace(**cfg)


def _install_stop_handlers(channel, grace_seconds: float, state: dict):
    """Install POSIX cooperative-stop handlers.

    Windows signal delivery is NOT_RUN: ``signal.SIGTERM`` exists but is not a
    reliable console-termination channel there, so no Windows claim is made.
    The handler itself is portable (thread-safe ``Event`` + a watchdog thread).
    """

    def _watchdog(signum: int) -> None:
        time.sleep(float(grace_seconds))
        message = (
            f"[zegrid] stop requested ({signum}) but no safe boundary was "
            f"reached within {grace_seconds:g}s; exiting {EXIT_ERROR} "
            "WITHOUT publishing (durable commits stay atomic on disk)\n"
        )
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
        finally:
            os._exit(EXIT_ERROR)

    def _handler(signum, frame):  # noqa: ARG001 - signal signature
        if state.get("stop_requested"):
            sys.stderr.write(
                f"[zegrid] second signal ({signum}); exiting {EXIT_ERROR} now\n"
            )
            sys.stderr.flush()
            os._exit(EXIT_ERROR)
        state["stop_requested"] = True
        state["signal"] = int(signum)
        first = channel.request(f"signal {signum}")
        sys.stderr.write(
            f"[zegrid] cooperative stop requested ({signum}); finishing the "
            f"current safe boundary (grace {grace_seconds:g}s)\n"
        )
        sys.stderr.flush()
        if first:
            threading.Thread(
                target=_watchdog, args=(int(signum),), daemon=True
            ).start()

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Non-main thread or unsupported platform: documented NOT_RUN.
            pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        resolved = validate_options(args)
    except RefusedOption as exc:
        sys.stderr.write(f"[zegrid] REFUSED: {exc}\n")
        return EXIT_REFUSED

    started = time.time()
    state: dict = {"stop_requested": False, "signal": None}
    summary: dict = {
        "schema": SUMMARY_SCHEMA,
        "status": "ERROR",
        "exit_code": EXIT_ERROR,
        "input": str(resolved["input_dir"]),
        "output": str(resolved["output_dir"]),
        "workers": int(args.workers),
        "storage_profile": args.storage_profile,
        "durable_store_dir": (
            str(resolved["store_dir"]) if resolved["store_dir"] else None
        ),
        "durable_store_budget_bytes": args.durable_store_budget_bytes,
        "resume_mode": args.resume_mode,
        "resume_run_id": args.resume_run_id,
        "resource_recovery_enabled": bool(args.resource_recovery),
        "resource_recovery_deadline_s": args.resource_recovery_deadline_s,
        "overwrite": bool(args.overwrite),
        "previous_publication": {
            "found": bool(resolved["previous"]["found"]),
            "zegrid": bool(resolved["previous"]["zegrid"]),
            "evidence": list(resolved["previous"]["evidence"]),
            "foreign": list(resolved["previous"]["foreign"]),
            "pointer": resolved["previous_pointer"],
        },
        "halo_px": resolved["halo_px"],
        "layout": args.layout,
        "gauge_mode": args.gauge_mode,
        "gpu": False,
        "stop": {
            "requested": False,
            "signal": None,
            "status": None,
            "durable": False,
            "durable_commits_reported": 0,
            "durable_commits_found": 0,
            "durable_commits_valid": 0,
            "publication_started": False,
        },
        "publication": {"science": None, "manifest": None},
    }

    # Lazy import: --help and argument refusals never load the engine (and never
    # touch Qt). The import assertion test relies on this.
    from . import zemosaic_zegrid_mode as zz
    from .core.zegrid.run_registry import CooperativeStopChannel

    config = _build_config(args, resolved)
    channel = CooperativeStopChannel()
    _install_stop_handlers(channel, args.stop_grace_seconds, state)

    sci_path = None
    status = "ERROR"
    exit_code = EXIT_ERROR
    error_text = None
    stop_record: dict = {}
    publication_before = (
        resolved["output_dir"] / ".zegrid_publications"
    ).exists()
    # Any artifact this run publishes is written after this instant; an older
    # manifest in a reused output folder is therefore provably not ours.
    run_started_ns = time.time_ns()
    write_set_before = _write_set_fingerprint(resolved["output_dir"])
    summary["overwrite_guard"] = "not_checked"

    try:
        sci_path = zz.run_zegrid_mode(
            str(resolved["input_dir"]),
            str(resolved["output_dir"]),
            progress_callback=lambda message, **_: print(
                f"[zegrid] {message}", flush=True
            ),
            workers=int(args.workers),
            storage_profile=args.storage_profile,
            cell_gauge_mode=args.gauge_mode,
            zegrid_halo_px=resolved["halo_px"],
            use_gpu=False,
            zconfig=config,
            stop_channel=channel,
        )
    except BaseException as exc:  # noqa: BLE001 - classified below, never silent
        stop_cls = getattr(zz, "_R55RunStopRequested", None)
        if stop_cls is not None and isinstance(exc, stop_cls):
            stop_record = dict(getattr(exc, "record", {}) or {})
            status = (
                "STOPPED_RESUMABLE"
                if str(stop_record.get("status")) == "stopped_resumable"
                else "STOPPED_IN_MEMORY"
            )
            exit_code = EXIT_ERROR
            error_text = str(exc)
        else:
            status = "ERROR"
            exit_code = EXIT_ERROR
            error_text = f"{type(exc).__name__}: {exc}"
            import traceback

            traceback.print_exc()
    else:
        science_path, manifest_path = _resolve_publication(resolved["output_dir"])
        if (
            science_path is not None
            and manifest_path is not None
            and not _publication_is_fresh(manifest_path, run_started_ns)
        ):
            status = "ERROR"
            exit_code = EXIT_ERROR
            error_text = (
                "stale publication artifacts in a reused output folder: the "
                f"manifest {manifest_path} predates this run; refusing to "
                "report a success that this run did not publish"
            )
        elif science_path is not None and manifest_path is not None:
            status = "COMPLETE"
            exit_code = EXIT_OK
            summary["publication"] = {
                "science": str(science_path),
                "manifest": str(manifest_path),
            }
            # REWORK-1/2 C1 defence in depth: without --overwrite no file that
            # existed before the run may change. Detection refuses earlier; this
            # assertion makes an unclassified pre-existing publication a loud
            # error instead of a silent success.
            if not args.overwrite:
                write_set_after = _write_set_fingerprint(resolved["output_dir"])
                changed = sorted(
                    name for name, token in write_set_before.items()
                    if write_set_after.get(name) != token
                )
                if changed:
                    status = "ERROR"
                    exit_code = EXIT_ERROR
                    error_text = (
                        "internal overwrite guard triggered: pre-existing "
                        f"output file(s) changed without --overwrite: "
                        f"{', '.join(changed)}"
                    )
                    summary["publication"] = {"science": None, "manifest": None}
                    summary["overwrite_guard"] = "triggered"
                else:
                    summary["overwrite_guard"] = "ok"
            # REWORK-1 C1: with a durable store a new run must publish its OWN
            # generation. A pointer that still resolves to the previous run's
            # generation is a failed replacement, never a success.
            new_pointer = _publication_pointer(resolved["output_dir"])
            summary["publication"]["pointer"] = new_pointer
            previous_pointer = resolved["previous_pointer"]
            if (
                previous_pointer is not None
                and new_pointer is not None
                and (
                    new_pointer["path"] == previous_pointer["path"]
                    or (
                        new_pointer.get("run_id")
                        and new_pointer.get("run_id")
                        == previous_pointer.get("run_id")
                    )
                )
            ):
                status = "ERROR"
                exit_code = EXIT_ERROR
                error_text = (
                    "--overwrite resolved to the PREVIOUS publication "
                    f"generation ({new_pointer['path']}); the new run did not "
                    "publish its own generation"
                )
                summary["publication"] = {
                    "science": None, "manifest": None,
                }
        else:
            status = "ERROR"
            exit_code = EXIT_ERROR
            error_text = (
                "engine returned without a verified science FITS + manifest "
                f"(science={science_path!r}, manifest={manifest_path!r})"
            )

    duration = time.time() - started
    commits_found = commits_valid = 0
    if resolved["store_dir"] is not None:
        commits_found, commits_valid = _count_durable_commits(resolved["store_dir"])

    stop_entry = summary["stop"]
    stop_entry.update(
        {
            "requested": bool(state.get("stop_requested") or channel.requested),
            "signal": state.get("signal"),
            "status": stop_record.get("status"),
            "durable": bool(stop_record.get("durable_stopped_resumable")),
            "durable_commits_reported": int(
                stop_record.get("durable_commits") or 0
            ),
            "durable_commits_found": int(commits_found),
            "durable_commits_valid": int(commits_valid),
            "publication_started": bool(
                publication_before
                or (resolved["output_dir"] / ".zegrid_publications").exists()
            ),
        }
    )
    summary.update(
        {
            "status": status,
            "exit_code": int(exit_code),
            "duration_s": round(float(duration), 3),
            "peak_rss_kib": _peak_rss_kib(),
            "error": error_text,
        }
    )

    try:
        _write_summary(resolved["summary_path"], summary)
    except Exception as exc:
        sys.stderr.write(f"[zegrid] could not write JSON summary: {exc}\n")

    if status == "COMPLETE":
        print(
            f"[zegrid] COMPLETE in {duration:.1f}s — "
            f"science={summary['publication']['science']}"
        )
    elif status.startswith("STOPPED"):
        print(
            f"[zegrid] {status} after {duration:.1f}s — "
            f"durable_commits={commits_found} (valid={commits_valid}), "
            "no publication"
        )
    else:
        sys.stderr.write(f"[zegrid] ERROR: {error_text}\n")
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
