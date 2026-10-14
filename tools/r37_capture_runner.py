#!/usr/bin/env python3
"""ZM-ZEGRID-R37-E — reproducible cross-platform R37 diagnostic-capture runner (CLI).

WHY THIS EXISTS
---------------
The canonical R37 diagnostic capture (ZM-ZEGRID-R37-D seam) must be produced at
the real pre-offset/pre-feather per-Cell assembly point of a *specific* baseline
run (the accepted `R34F`: 653 selected frames out of 816, a 6x4 layout, effective
`per_cell` gauge). Reproducing that run by hand is error-prone: the exact frame
subset, the pinned layout, the gauge mode and the capture opt-in all have to be
reconstructed coherently from a previous run's ``zegrid_manifest.json`` +
``run_config_snapshot.json``.

The originally capable host was a Windows machine (RTX 3070, ~16 GiB). This tool
lets that machine — or any capable machine — reproduce the exact baseline frame
subset/layout and produce an exact-heldout R37 artifact with ONE documented
command, WITHOUT any GUI/config wiring.

SAFETY MODEL (read this)
------------------------
* The tool is **dry-run by default**. Without ``--execute`` it performs the real
  descriptor/canvas/pinned-layout memory preflight (using the SAME accepted
  ZeGrid functions as a production run) and writes **nothing** — not in the input
  dir, not in the baseline dir, not in the output dir.
* ``--execute`` additionally refuses unless the output dir is a **new/empty**
  destination that cannot overwrite the input or the baseline run (no equality,
  no containment either way).
* ``--execute`` REQUIRES a successful real memory preflight. ``--force-infeasible``
  does not exist by design (forcing an infeasible pinned layout only invites
  swap-thrash/OOM), and ``--skip-memory-preflight --execute`` is refused BEFORE
  any engine work. ``--skip-memory-preflight`` is a metadata/selection inspection
  dry-run only: it reports ``preflight SKIPPED`` (never ``PASS``) and never
  executes.
* Multi-pass runs (mount-segregated ``grid_EQ`` / ``grid_ALTZ``) are verified as a
  whole: every pass must be internally consistent, the pass frame sets must be
  disjoint, and their union must equal the exact selected frame set.
* No shell interpolation, no hidden subprocess, no GUI/config activation, no
  correction application. It calls ``run_zegrid_mode`` directly with explicit
  parameters and the explicit, default-OFF ``r37_diagnostic_capture=True`` seam.

EXAMPLES
--------
bash (Linux/macOS)::

    python tools/r37_capture_runner.py \\
        --input-dir "/path/to/lights organized/EQ/IRCUT" \\
        --baseline-run-dir "/path/to/zegrid_out/R34F" \\
        --output-dir "/path/to/zegrid_out/R37E_CAPTURE" \\
        --gpu auto --holdout-cap 1000000            # dry-run / preflight

    python tools/r37_capture_runner.py \\
        --input-dir "/path/to/lights organized/EQ/IRCUT" \\
        --baseline-run-dir "/path/to/zegrid_out/R34F" \\
        --output-dir "/path/to/zegrid_out/R37E_CAPTURE" \\
        --gpu auto --holdout-cap 1000000 --execute  # real capture + verify

PowerShell (Windows; adjust the example drives/paths to your checkout/data)::

    python .\\tools\\r37_capture_runner.py `
        --input-dir "C:\\Users\\TRISTAN\\Desktop\\astrophoto\\caldwell_11\\lights organized\\EQ\\IRCUT" `
        --baseline-run-dir "C:\\Users\\TRISTAN\\Desktop\\astrophoto\\caldwell_11\\OUT\\R34F" `
        --output-dir "C:\\Users\\TRISTAN\\Desktop\\astrophoto\\caldwell_11\\OUT\\R37E_CAPTURE" `
        --gpu auto --holdout-cap 1000000

    python .\\tools\\r37_capture_runner.py `
        --input-dir "C:\\Users\\TRISTAN\\Desktop\\astrophoto\\caldwell_11\\lights organized\\EQ\\IRCUT" `
        --baseline-run-dir "C:\\Users\\TRISTAN\\Desktop\\astrophoto\\caldwell_11\\OUT\\R34F" `
        --output-dir "C:\\Users\\TRISTAN\\Desktop\\astrophoto\\caldwell_11\\OUT\\R37E_CAPTURE" `
        --gpu auto --holdout-cap 1000000 --execute

Exit codes: ``0`` ok, ``1`` validation/other error, ``2`` memory preflight FAIL,
``3`` unsafe output destination.

The import of this module is intentionally side-effect-free: the ZeMosaic engine
modules are imported lazily, inside the functions that need them, so importing
the runner never activates a capture or a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Fixed names / defaults
# ---------------------------------------------------------------------------

MANIFEST_NAME = "zegrid_manifest.json"
SNAPSHOT_NAME = "run_config_snapshot.json"
#: Mirror of ``zemosaic.core.zegrid.background_diagnostics.CAPTURE_ARTIFACT_BASENAME``.
#: Cross-checked against the real constant when the diagnostic module is loaded.
CAPTURE_ARTIFACT_BASENAME = "zegrid_r37_pre_feather.json"
REPORT_NAME = "r37_capture_verification.json"
VERIFICATION_SCHEMA = "zemosaic.zegrid.r37e.capture_verification"
VERIFICATION_VERSION = 1

DEFAULT_HOLDOUT_CAP = 1_000_000

FITS_SUFFIXES = (".fit", ".fits")
#: Known mount-segregated output sub-directories of a ZeGrid run.
MOUNT_PASS_DIRS = ("grid_EQ", "grid_ALTZ")

CAPTURE_MISSION = "ZM-ZEGRID-R37-D"
CAPTURE_STAGE = "pre_offset_pre_feather"
CELL_GAUGE_MODE_PER_CELL = "per_cell"

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_PREFLIGHT_FAIL = 2
_EXIT_UNSAFE_OUTPUT = 3


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RunnerError(RuntimeError):
    """Base class for runner failures (message is user-facing)."""


class BaselineError(RunnerError):
    """The baseline run directory / manifest / snapshot is invalid."""


class SelectionError(RunnerError):
    """A manifest frame_id cannot be resolved unambiguously to a source file."""


class ConfigError(RunnerError):
    """A reconstructed config value could not be parsed strictly."""


class OutputSafetyError(RunnerError):
    """The requested output destination is not a safe new/empty directory."""


# ---------------------------------------------------------------------------
# Lazy engine access (import side-effect freedom)
# ---------------------------------------------------------------------------

_ZEGRID_CACHE: dict[str, Any] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    """Make a src-layout checkout importable when run as a plain script."""
    src = _repo_root() / "src"
    if src.is_dir():
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_zegrid():
    """Import (once) and return ``zemosaic.zemosaic_zegrid_mode``."""
    mod = _ZEGRID_CACHE.get("zz")
    if mod is None:
        _ensure_src_on_path()
        from zemosaic import zemosaic_zegrid_mode as mod  # local import: intentional

        _ZEGRID_CACHE["zz"] = mod
    return mod


def _load_bgd():
    """Import (once) and return the accepted background-diagnostics module."""
    mod = _ZEGRID_CACHE.get("bgd")
    if mod is None:
        _ensure_src_on_path()
        from zemosaic.core.zegrid import background_diagnostics as mod  # local import

        _ZEGRID_CACHE["bgd"] = mod
    return mod


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _reject_json_constant(name: str):
    raise BaselineError(f"strict JSON: literal {name!r} is not allowed")


def load_json_strict(path: Path) -> Any:
    """Load JSON rejecting NaN/Infinity (strict, byte-honest)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh, parse_constant=_reject_json_constant)
    except BaselineError:
        raise
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise BaselineError(f"{path}: cannot read ({exc})") from exc


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), int(n)


def _digest_names(names: Sequence[str]) -> str:
    h = hashlib.sha256()
    for name in sorted({str(n) for n in names}):
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _norm_key(value: Any) -> str:
    """Platform-aware comparison key (forward slashes + os.path.normcase)."""
    try:
        return os.path.normcase(str(value).replace("\\", "/"))
    except Exception:
        return str(value)


def _as_posix(value: str) -> str:
    return str(value).strip().replace("\\", "/")


#: Windows drive-absolute syntax (``C:/`` or ``C:\``) after separator
#: normalisation. A relative name that merely CONTAINS a colon (e.g.
#: ``a:b.fit`` or ``C:foo``) is NOT drive-absolute and stays allowed.
_DRIVE_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_drive_absolute(norm: str) -> bool:
    return bool(_DRIVE_ABSOLUTE_RE.match(norm))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _git_head(repo_root: Path) -> str | None:
    """Best-effort HEAD sha WITHOUT invoking a subprocess."""
    try:
        git = repo_root / ".git"
        if git.is_file():  # worktree / submodule pointer file
            text = git.read_text(encoding="utf-8").strip()
            if text.lower().startswith("gitdir:"):
                target = Path(text.split(":", 1)[1].strip())
                # A relative gitdir is relative to the worktree root (the
                # directory CONTAINING the .git pointer file), not the CWD.
                if not target.is_absolute():
                    target = git.parent / target
                git = target
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            ref_file = git / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip()
            packed = git / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith(("#", "^")):
                        continue
                    sha, _, name = line.partition(" ")
                    if name.strip() == ref:
                        return sha.strip()
            return None
        return head
    except Exception:
        return None


def _package_version() -> str | None:
    try:
        _ensure_src_on_path()
        import zemosaic  # local import

        return str(getattr(zemosaic, "__version__", "")) or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Baseline reconstruction — strict manifest/snapshot load + validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineInfo:
    run_dir: Path
    manifest_path: Path
    snapshot_path: Path
    manifest: Mapping[str, Any]
    snapshot: Mapping[str, Any]
    frame_ids: tuple[str, ...]
    nx: int
    ny: int
    canvas_width: int
    canvas_height: int
    normalization: str
    gauge_mode: str


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise BaselineError(message)


def load_baseline(baseline_run_dir: str | os.PathLike[str]) -> BaselineInfo:
    """Strictly load + validate a baseline run's manifest and config snapshot."""
    base = Path(baseline_run_dir).expanduser()
    _require(base.is_dir(), f"baseline run dir is not a directory: {base}")
    manifest_path = base / MANIFEST_NAME
    snapshot_path = base / SNAPSHOT_NAME
    _require(manifest_path.is_file(), f"baseline manifest missing: {manifest_path}")
    _require(snapshot_path.is_file(), f"baseline config snapshot missing: {snapshot_path}")

    manifest = load_json_strict(manifest_path)
    snapshot = load_json_strict(snapshot_path)
    _require(isinstance(manifest, Mapping), "baseline manifest is not a JSON object")
    _require(isinstance(snapshot, Mapping), "baseline config snapshot is not a JSON object")

    _require(manifest.get("engine") == "zegrid",
             f"baseline engine must be 'zegrid', got {manifest.get('engine')!r}")

    cf = manifest.get("cell_feather")
    _require(isinstance(cf, Mapping) and cf.get("enabled") is True,
             "baseline cell_feather must be enabled (per_cell feather path)")

    fids = manifest.get("frame_ids")
    _require(isinstance(fids, list) and len(fids) > 0, "baseline frame_ids must be a non-empty list")
    for i, fid in enumerate(fids):
        _require(isinstance(fid, str) and fid.strip() != "",
                 f"baseline frame_ids[{i}] must be a non-empty string")
    _require(len(set(fids)) == len(fids), "baseline frame_ids must be unique")
    n_frames = manifest.get("n_frames")
    _require(_is_int(n_frames) and int(n_frames) == len(fids),
             f"baseline n_frames ({n_frames!r}) must equal len(frame_ids) ({len(fids)})")

    canvas = manifest.get("canvas")
    _require(isinstance(canvas, Mapping), "baseline canvas must be an object")
    width = canvas.get("width")
    height = canvas.get("height")
    _require(_is_int(width) and int(width) > 0, f"baseline canvas.width invalid: {width!r}")
    _require(_is_int(height) and int(height) > 0, f"baseline canvas.height invalid: {height!r}")

    layout = manifest.get("layout")
    _require(isinstance(layout, Mapping), "baseline layout must be an object")
    nx = layout.get("nx")
    ny = layout.get("ny")
    _require(_is_int(nx) and int(nx) > 0, f"baseline layout.nx invalid: {nx!r}")
    _require(_is_int(ny) and int(ny) > 0, f"baseline layout.ny invalid: {ny!r}")
    _require(int(nx) <= int(width) and int(ny) <= int(height),
             f"baseline layout {nx}x{ny} exceeds canvas {width}x{height}")

    gauge = manifest.get("photometric_gauge")
    _require(isinstance(gauge, Mapping), "baseline photometric_gauge must be an object")
    gauge_mode = gauge.get("mode")
    _require(gauge_mode == CELL_GAUGE_MODE_PER_CELL,
             f"baseline effective gauge must be 'per_cell', got {gauge_mode!r}")

    normalization = manifest.get("normalization") or "sky_mean"
    _require(isinstance(normalization, str) and normalization.strip() != "",
             f"baseline normalization invalid: {normalization!r}")

    return BaselineInfo(
        run_dir=base,
        manifest_path=manifest_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
        snapshot=snapshot,
        frame_ids=tuple(str(f) for f in fids),
        nx=int(nx),
        ny=int(ny),
        canvas_width=int(width),
        canvas_height=int(height),
        normalization=str(normalization),
        gauge_mode=str(gauge_mode),
    )


# ---------------------------------------------------------------------------
# 2. Exact source-frame resolution
# ---------------------------------------------------------------------------

@dataclass
class SelectionResolution:
    input_dir: Path
    paths: tuple[Path, ...]
    #: Canonical POSIX relative logical ids under ``input_dir`` — the SAME
    #: identity the engine emits as ``FrameId.logical_path`` (NEVER a basename).
    logical_ids: tuple[str, ...]
    digest: str
    matched_by_relative: int
    matched_by_basename: int

    @property
    def count(self) -> int:
        return len(self.paths)

    @property
    def basenames(self) -> tuple[str, ...]:
        return tuple(PurePosixPath(i).name for i in self.logical_ids)

    def to_dict(self) -> dict:
        return {
            "requested": self.count,
            "resolved": self.count,
            "matched_by_relative_path": self.matched_by_relative,
            "matched_by_basename": self.matched_by_basename,
            "logical_ids_digest_sha256": self.digest,
        }


def _scan_fits(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.suffix.lower() in FITS_SUFFIXES:
                out.append(p)
        except OSError:
            continue
    out.sort(key=lambda p: p.as_posix().lower())
    return out


def resolve_selection(input_dir: str | os.PathLike[str],
                      frame_ids: Sequence[str]) -> SelectionResolution:
    """Resolve each manifest frame_id to EXACTLY one source file under input_dir.

    Deterministic; preserves the manifest frame_id order. Rejects path traversal,
    absolute ids, missing ids, ambiguous basenames and duplicate resolution. No
    basename ambiguity is tolerated and no path may escape the input dir.
    """
    root = Path(input_dir).expanduser()
    if not root.is_dir():
        raise SelectionError(f"input dir is not a directory: {root}")
    root_res = root.resolve()
    files = _scan_fits(root)

    by_rel: dict[str, Path] = {}
    by_base: dict[str, list[Path]] = {}
    for f in files:
        rel = f.relative_to(root).as_posix()
        by_rel.setdefault(_norm_key(rel), f)
        by_base.setdefault(_norm_key(f.name), []).append(f)

    resolved: list[Path] = []
    seen: set[Path] = set()
    n_rel = 0
    n_base = 0
    problems: list[str] = []
    fids = list(frame_ids)
    n_frames = len(fids)
    for idx, fid in enumerate(fids):
        raw = str(fid)
        norm = _as_posix(raw)
        if norm == "":
            problems.append(f"frame_ids[{idx}]: empty id")
            continue
        pp = PurePosixPath(norm)
        if _is_drive_absolute(norm):
            problems.append(
                f"frame_id {raw!r}: drive-absolute paths (e.g. 'C:\\path' or "
                "'C:/path') are not allowed"
            )
            continue
        if norm.startswith("/") or norm.startswith("//") or pp.is_absolute():
            problems.append(
                f"frame_id {raw!r}: POSIX-absolute/UNC paths are not allowed"
            )
            continue
        if any(part == ".." for part in pp.parts):
            problems.append(f"frame_id {raw!r}: path traversal ('..') is not allowed")
            continue
        if any(part == "" for part in pp.parts):
            problems.append(f"frame_id {raw!r}: empty path component")
            continue

        chosen = by_rel.get(_norm_key(norm))
        mode = "relative"
        if chosen is None:
            matches = by_base.get(_norm_key(pp.name), [])
            if len(matches) == 1:
                chosen = matches[0]
                mode = "basename"
            elif len(matches) > 1:
                problems.append(
                    f"frame_id {raw!r}: ambiguous basename — "
                    f"{len(matches)} files match ({', '.join(sorted(m.as_posix() for m in matches))})"
                )
                continue
            else:
                problems.append(f"frame_id {raw!r}: not found under input dir")
                continue
        try:
            chosen.resolve().relative_to(root_res)
        except ValueError:
            problems.append(f"frame_id {raw!r}: resolved path escapes the input dir")
            continue
        if chosen in seen:
            problems.append(f"frame_id {raw!r}: resolves to an already-selected file")
            continue
        seen.add(chosen)
        resolved.append(chosen)
        if mode == "relative":
            n_rel += 1
        else:
            n_base += 1

    if problems:
        head = problems[:20]
        more = "" if len(problems) <= 20 else f" (+{len(problems) - 20} more)"
        raise SelectionError(
            f"could not resolve {len(problems)} frame_id(s) of {n_frames}:\n  - "
            + "\n  - ".join(head) + more
        )

    if len(resolved) != n_frames:
        raise SelectionError(
            f"resolved {len(resolved)} frames but manifest declares {n_frames}"
        )
    if not resolved:
        raise SelectionError("selection resolved zero frames")

    logical_ids = [p.relative_to(root).as_posix() for p in resolved]
    return SelectionResolution(
        input_dir=root,
        paths=tuple(resolved),
        logical_ids=tuple(logical_ids),
        digest=_digest_names(logical_ids),
        matched_by_relative=n_rel,
        matched_by_basename=n_base,
    )


# ---------------------------------------------------------------------------
# 3. Strict config parsing + pinned-layout config copy
# ---------------------------------------------------------------------------

def parse_winsor_limits(value: Any) -> tuple[float, float] | None:
    """Strictly parse a winsor-limits config value into ``(low, high)`` or None.

    Accepts the Classic string form ``"0.05,0.05"`` (also ``;``/``x`` separators)
    or a 2-element list/tuple. Each value must be finite with ``0 <= v < 0.5`` and
    ``low + high < 1`` (the canonical engine's validity domain). A non-empty but
    unparseable/out-of-domain value is a LOUD error, never silently defaulted.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [p for p in re.split(r"[,;x]", text) if p.strip() != ""]
        if len(parts) != 2:
            raise ConfigError(f"invalid stacking_winsor_limits {value!r} (expected 'low,high')")
        raw = parts
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        raw = [value[0], value[1]]
    else:
        raise ConfigError(f"invalid stacking_winsor_limits {value!r} (expected 'low,high')")
    try:
        low, high = float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid stacking_winsor_limits {value!r}") from exc
    import math

    if not (math.isfinite(low) and math.isfinite(high)):
        raise ConfigError(f"stacking_winsor_limits must be finite, got {value!r}")
    if not (0.0 <= low < 0.5 and 0.0 <= high < 0.5 and low + high < 1.0):
        raise ConfigError(f"stacking_winsor_limits out of domain [0,0.5)^2: {value!r}")
    return (low, high)


def _parse_sigma(snapshot: Mapping[str, Any], key: str, default: float) -> float:
    import math

    if key not in snapshot or snapshot.get(key) is None:
        return float(default)
    value = snapshot.get(key)
    if not _is_number(value):
        raise ConfigError(f"{key} must be a number, got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f <= 0.0:
        raise ConfigError(f"{key} must be finite and > 0, got {value!r}")
    return f


def _portable_temp_dir(value: Any) -> str:
    """Return a usable temporary-work dir, or "" (fall back to the output dir).

    A baseline snapshot recorded on another OS may carry an absolute temp path
    that does not exist here; using it would create a stray directory. Only an
    existing directory is honoured.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        p = Path(text).expanduser()
        if p.is_dir():
            return str(p)
    except Exception:
        pass
    return ""


def build_pinned_config(snapshot: Mapping[str, Any], nx: int, ny: int) -> SimpleNamespace:
    """Build a COPIED ``SimpleNamespace`` config pinning the baseline layout.

    The source snapshot mapping is never mutated. ``zegrid_layout`` is pinned to
    the baseline ``nx x ny`` so the RAM-adaptive search is bypassed and the exact
    baseline geometry is reproduced (floors are still enforced by the engine).
    """
    cfg = SimpleNamespace(**dict(snapshot))
    cfg.zegrid_layout = f"{int(nx)}x{int(ny)}"
    cfg.temporary_work_dir = _portable_temp_dir(snapshot.get("temporary_work_dir"))
    return cfg


# ---------------------------------------------------------------------------
# 4. Invocation reconstruction (explicit parameters, no hidden defaults)
# ---------------------------------------------------------------------------

@dataclass
class Invocation:
    input_dir: Path
    output_dir: Path
    kwargs: dict
    params_report: dict


def build_invocation(
    baseline: BaselineInfo,
    selection: SelectionResolution,
    snapshot: Mapping[str, Any],
    cfg: SimpleNamespace,
    *,
    gpu_mode: str,
    holdout_cap: int,
) -> Invocation:
    """Reconstruct the exact ``run_zegrid_mode`` keyword arguments.

    ``--gpu auto`` omits ``use_gpu`` so the engine resolves it from the baseline
    config (``use_gpu_grid`` -> ... -> default); ``on``/``off`` pass an explicit
    bool. Everything else is reconstructed from the baseline manifest/snapshot
    with strict parsing.
    """
    if gpu_mode not in ("auto", "on", "off"):
        raise ConfigError(f"invalid --gpu {gpu_mode!r} (expected auto|on|off)")
    if not _is_int(holdout_cap) or int(holdout_cap) < 4:
        raise ConfigError(f"--holdout-cap must be an int >= 4, got {holdout_cap!r}")

    reject_algo = snapshot.get("stacking_rejection_algorithm", "kappa_sigma")
    if not isinstance(reject_algo, str) or reject_algo.strip() == "":
        raise ConfigError(f"stacking_rejection_algorithm invalid: {reject_algo!r}")
    reject_algo = reject_algo.strip().lower()

    kappa_low = _parse_sigma(snapshot, "stacking_kappa_low", 3.0)
    kappa_high = _parse_sigma(snapshot, "stacking_kappa_high", 3.0)
    winsor = parse_winsor_limits(snapshot.get("stacking_winsor_limits"))
    if winsor is None:
        winsor = (0.05, 0.05)

    use_gpu_arg: bool | None = None
    if gpu_mode == "on":
        use_gpu_arg = True
    elif gpu_mode == "off":
        use_gpu_arg = False

    kwargs: dict = {
        "stack_norm_method": baseline.normalization,
        "stack_reject_algo": reject_algo,
        "stack_kappa_low": kappa_low,
        "stack_kappa_high": kappa_high,
        "winsor_limits": (float(winsor[0]), float(winsor[1])),
        "cell_gauge_mode": CELL_GAUGE_MODE_PER_CELL,
        "selection_paths": [str(p) for p in selection.paths],
        "zconfig": cfg,
        "r37_diagnostic_capture": True,
        "r37_diagnostic_holdout_cap": int(holdout_cap),
    }
    if use_gpu_arg is not None:
        kwargs["use_gpu"] = use_gpu_arg

    params_report = {
        "normalization": kwargs["stack_norm_method"],
        "stack_reject_algo": kwargs["stack_reject_algo"],
        "stack_kappa_low": kwargs["stack_kappa_low"],
        "stack_kappa_high": kwargs["stack_kappa_high"],
        "winsor_limits": list(kwargs["winsor_limits"]),
        "cell_gauge_mode": kwargs["cell_gauge_mode"],
        "r37_diagnostic_capture": True,
        "r37_diagnostic_holdout_cap": int(holdout_cap),
        "gpu_mode": gpu_mode,
        "use_gpu_arg": use_gpu_arg,
        "selection_frames": selection.count,
        "selection_basenames_digest_sha256": selection.digest,
        "pinned_layout": f"{baseline.nx}x{baseline.ny}",
        "zegrid_layout_config": cfg.zegrid_layout,
        "temporary_work_dir": cfg.temporary_work_dir,
    }
    return Invocation(input_dir=selection.input_dir, output_dir=Path("."), kwargs=kwargs,
                      params_report=params_report)


# ---------------------------------------------------------------------------
# 5. Memory / capability preflight (real, no stacking, no writes)
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    verdict: str  # "PASS" | "FAIL"
    report: dict
    failure: str | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"


def _cpu_ladder(zz, canvas, footprints, nx: int, ny: int, ram_budget: int | None) -> dict:
    """Compute the R36 CPU streaming-tile ladder report (accepted functions only)."""
    halo = int(zz.zsw.HALO_PX)
    tile = int(zz.STREAM_TILE_SIZE)
    cells = list(zz._iter_cell_bounds_mode_aware(canvas, footprints, nx, ny, halo, tile))
    inmem_worst = max((int(c["inmem_bound_bytes"]) for c in cells), default=0)
    stream_worst_by_tile = {int(t): 0 for t in zz.CPU_STREAMING_TILE_LADDER}
    for c in cells:
        hw = tuple(c["patch_hw"])
        n = int(c["n"])
        for t in zz.CPU_STREAMING_TILE_LADDER:
            b = int(zz._estimate_streaming_bytes(n, hw, int(t)))
            if b > stream_worst_by_tile[int(t)]:
                stream_worst_by_tile[int(t)] = b
    selected, trials, reason = zz._select_cpu_streaming_tile(stream_worst_by_tile, ram_budget)
    min_tile = int(zz.CPU_STREAMING_TILE_MINIMUM)
    return {
        "cell_count": len(cells),
        "max_contributors": max((int(c["n"]) for c in cells), default=0),
        "inmem_worst_bytes": int(inmem_worst),
        "stream_worst_by_tile": {str(k): int(v) for k, v in stream_worst_by_tile.items()},
        "trials": trials,
        "selected_tile_size": selected,
        "minimum_tile_size": min_tile,
        "minimum_tile_worst_bound_bytes": int(stream_worst_by_tile.get(min_tile, 0)),
        "reason": reason,
        "ladder": [int(t) for t in zz.CPU_STREAMING_TILE_LADDER],
    }


def preflight(
    input_dir: str | os.PathLike[str],
    selection: SelectionResolution,
    baseline: BaselineInfo,
    cfg: SimpleNamespace,
    *,
    progress=None,
) -> PreflightResult:
    """Real descriptor/canvas/pinned-layout memory preflight (no stacking/writes)."""
    zz = _load_zegrid()
    zal = zz.zal
    zg = zz.zg
    _ensure_src_on_path()
    from zemosaic.zemosaic_stack_plan import FrameInfo  # local import

    def _p(msg: str) -> None:
        if progress is not None:
            try:
                progress(msg)
            except Exception:
                pass

    sip_mode = "keep"
    try:
        cand = str(getattr(cfg, "zegrid_sip_mode", "keep") or "keep").strip().lower()
        if cand in ("keep", "strip"):
            sip_mode = cand
    except Exception:
        sip_mode = "keep"

    frames_info = [FrameInfo(path=Path(p), order=i) for i, p in enumerate(selection.paths)]
    descs, rejected = zz._build_frame_descriptors(
        frames_info, str(input_dir), None, sip_mode=sip_mode
    )
    canvas = zg.build_canvas(descs)

    available = int(zz.available_memory_bytes())
    ram_budget = available

    footprints_out: list = []
    verdict = "PASS"
    failure = None
    layout = None
    emit = (lambda msg, lvl="INFO": _p(str(msg))) if progress is not None else None
    try:
        layout = zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=ram_budget, available_bytes=available,
            tile_size=int(zz.STREAM_TILE_SIZE), pinned_layout=(baseline.nx, baseline.ny),
            emit=emit, footprints_out=footprints_out,
        )
    except (zal.LayoutInfeasible, ValueError) as exc:
        verdict = "FAIL"
        failure = f"{type(exc).__name__}: {exc}"

    footprints = footprints_out[0] if footprints_out else zal.project_footprints(descs, canvas)
    ladder = _cpu_ladder(zz, canvas, footprints, baseline.nx, baseline.ny, ram_budget)

    gpu = {}
    try:
        gpu = dict(zz.zgpu.probe_gpu_backend())
    except Exception as exc:  # pragma: no cover - defensive
        gpu = {"available": False, "reason": f"probe_failed: {type(exc).__name__}"}

    report = {
        "requested_frames": selection.count,
        "candidate_frames": len(frames_info),
        "descriptors_built": len(descs),
        "rejected_wcs": len(rejected),
        # Basename-only: never persist a private absolute source path.
        "rejected_wcs_detail": [
            {"basename": Path(str(r.get("path", ""))).name, "reason": r.get("reason")}
            for r in list(rejected[:20])
        ],
        "sip_mode": sip_mode,
        "canvas": {
            "width": int(canvas.width),
            "height": int(canvas.height),
            "resolution_deg": float(canvas.resolution_deg),
            "id": str(canvas.canvas_id),
        },
        "baseline_canvas": {"width": baseline.canvas_width, "height": baseline.canvas_height},
        "pinned_layout": {"nx": baseline.nx, "ny": baseline.ny},
        "available_bytes": ram_budget,
        "ram_budget_bytes": ram_budget,
        "cpu_streaming": ladder,
        "gpu": {
            "available": bool(gpu.get("available")),
            "device": gpu.get("device"),
            "vram_total_bytes": gpu.get("vram_total_bytes"),
            "vram_free_bytes": gpu.get("vram_free_bytes"),
            "reason": gpu.get("reason"),
        },
        "verdict": verdict,
        "failure": failure,
    }
    if layout is not None:
        report["layout_decision"] = {
            "nx": int(layout["nx"]),
            "ny": int(layout["ny"]),
            "predicted_bound_bytes": int(layout["predicted_bound_bytes"]),
            "max_contributors": int(layout["max_contributors"]),
            "layout_source": layout.get("layout_source"),
            "budget_bound_choice": bool(layout.get("budget_bound_choice")),
            "warnings": list(layout.get("warnings") or ()),
        }
    return PreflightResult(verdict=verdict, report=report, failure=failure)


# ---------------------------------------------------------------------------
# 6. Output safety
# ---------------------------------------------------------------------------

def _is_same_or_nested(a: Path, b: Path) -> bool:
    try:
        ar = a.expanduser().resolve(strict=False)
        br = b.expanduser().resolve(strict=False)
    except Exception:
        return False
    return ar == br or ar in br.parents or br in ar.parents


def validate_output_safety(
    output_dir: str | os.PathLike[str],
    *,
    baseline_run_dir: str | os.PathLike[str],
    input_dir: str | os.PathLike[str],
) -> Path:
    """Require a NEW/empty output dir that cannot overwrite input or baseline."""
    out = Path(output_dir).expanduser()
    for label, other in (("baseline run dir", baseline_run_dir), ("input dir", input_dir)):
        if _is_same_or_nested(out, Path(other)):
            raise OutputSafetyError(
                f"output dir {out} overlaps the {label} {Path(other)} — refusing to run"
            )
    if out.exists():
        if not out.is_dir():
            raise OutputSafetyError(f"output path exists and is not a directory: {out}")
        try:
            entries = list(out.iterdir())
        except OSError as exc:
            raise OutputSafetyError(f"cannot inspect output dir {out}: {exc}") from exc
        if entries:
            raise OutputSafetyError(
                f"output dir {out} exists and is not empty ({len(entries)} entries); "
                "refusing to run (use a new/empty destination)"
            )
    return out


# ---------------------------------------------------------------------------
# 7. Execute + post-run verification
# ---------------------------------------------------------------------------

def _call_run_zegrid_mode(input_folder: str, output_folder: str, *,
                          progress_callback=None, **kwargs):
    """Single call seam: calls ``run_zegrid_mode`` directly (no shell/subprocess)."""
    zz = _load_zegrid()
    return zz.run_zegrid_mode(
        input_folder, output_folder, progress_callback=progress_callback, **kwargs
    )


def _canonical_manifest_id(fid: Any) -> str:
    """Return a manifest frame id as a canonical POSIX relative logical path.

    Fails CLOSED (raises ``ValueError``) on any id that is not a canonical
    relative logical path: empty, absolute, drive-absolute, UNC, traversal,
    empty/'.' components, or a non-canonical spelling (``a//b``, ``./a``). The
    engine emits POSIX relative logical paths; separators are normalised
    explicitly and anything ambiguous is rejected rather than basename-normalised.
    """
    raw = str(fid)
    if raw.strip() == "":
        raise ValueError("empty frame id")
    norm = raw.strip().replace("\\", "/")
    if _is_drive_absolute(norm):
        raise ValueError(f"drive-absolute frame id {raw!r}")
    if norm.startswith("/"):
        raise ValueError(f"absolute/UNC frame id {raw!r}")
    pp = PurePosixPath(norm)
    if any(part == ".." for part in pp.parts):
        raise ValueError(f"traversal frame id {raw!r}")
    if any(part == "" for part in pp.parts):
        raise ValueError(f"empty path component in frame id {raw!r}")
    if any(part == "." for part in pp.parts):
        raise ValueError(f"non-canonical frame id {raw!r}")
    if pp.as_posix() != norm:
        raise ValueError(f"non-canonical frame id {raw!r}")
    if pp.name in ("", ".", ".."):
        raise ValueError(f"non-canonical frame id {raw!r}")
    return norm


def _discover_unknown_manifests(output_dir: Path) -> list[str]:
    """Any manifest under ``output_dir`` outside the recognized topology dirs.

    Bounded recursion that NEVER follows symlinked directory trees, so a symlink
    to an external manifest tree cannot be used to smuggle a pass in (or to
    traverse outside the output dir). Returns POSIX relative dir names.
    """
    known: set = set()
    try:
        known.add(output_dir.resolve(strict=False))
    except Exception:
        known.add(output_dir)
    for name in MOUNT_PASS_DIRS:
        try:
            known.add((output_dir / name).resolve(strict=False))
        except Exception:
            known.add(output_dir / name)
    unknown: list[str] = []
    try:
        walk = os.walk(output_dir, followlinks=False)
        for root, dirs, files in walk:
            rp = Path(root)
            # never descend through symlinked directories
            dirs[:] = [d for d in dirs if not (rp / d).is_symlink()]
            if MANIFEST_NAME in files:
                try:
                    key = rp.resolve(strict=False)
                except Exception:
                    key = rp
                if key not in known:
                    try:
                        rel = rp.relative_to(output_dir).as_posix()
                    except Exception:
                        rel = rp.name
                    unknown.append(rel or ".")
    except OSError:
        pass
    return sorted(set(unknown))


def _iter_run_dirs(output_dir: Path) -> list[Path]:
    dirs = [output_dir]
    for name in MOUNT_PASS_DIRS:
        cand = output_dir / name
        if cand.is_dir():
            dirs.append(cand)
    return dirs


def _safe_regular_file(path: Path, root_res: Path) -> tuple[bool, str | None]:
    """Fail-closed path-level check for a file we are about to read.

    A consumed manifest/artifact must be a regular NON-symlink file whose
    resolved path stays under the resolved output root. This is path-level
    validation, NOT sandboxing, and makes no TOCTOU guarantee beyond the checked
    path at this moment: a symlinked file, a dangling link, a non-regular file or
    a resolved escape is rejected before strict-load/hash.
    """
    try:
        if path.is_symlink():
            return False, "is a symlink"
        if not path.is_file():
            return False, "not a regular file"
        resolved = path.resolve(strict=True)
    except OSError as exc:
        return False, f"unresolvable ({type(exc).__name__})"
    try:
        resolved.relative_to(root_res)
    except ValueError:
        return False, "resolves outside the output root"
    return True, None


def _pass_dir_is_symlink(output_dir: Path, name: str) -> bool:
    """True when a recognized pass path exists as a symlink (incl. dangling)."""
    try:
        return (output_dir / name).is_symlink()
    except OSError:
        return False


def _safe_manifest(path: Path) -> tuple[Mapping | None, str | None]:
    """Strict-load a manifest, returning ``(manifest, None)`` or ``(None, reason)``.

    Any invalid JSON, non-object root, or wrong-typed ``frame_ids`` (not a list, or
    with non-string items) is a CLEAN verification failure: the caller records a
    basename/category-only reason and never raises a traceback.
    """
    try:
        obj = load_json_strict(path)
    except BaselineError as exc:
        return None, f"invalid JSON ({type(exc).__name__})"
    if not isinstance(obj, Mapping):
        return None, "manifest root is not an object"
    fids = obj.get("frame_ids")
    if not isinstance(fids, (list, tuple)):
        return None, "frame_ids is not a list"
    for i, f in enumerate(fids):
        if not isinstance(f, str):
            return None, f"frame_ids[{i}] is not a string"
    return obj, None


def verify_outputs(
    output_dir: Path,
    *,
    baseline: BaselineInfo,
    selection: SelectionResolution,
) -> dict:
    """Strict-load artifacts+manifests and verify linkage/identity/provenance.

    Topology (fail-closed): exactly ONE of
      (a) a single root ``zegrid_manifest.json``, or
      (b) recognized ``grid_EQ`` + ``grid_ALTZ`` pass manifests with NO root
          manifest.
    Any other manifest under the output dir (any depth, symlink-safe) is unknown;
    root+passes coexistence is stale/ambiguous; a missing mount peer fails.

    Identity/accounting uses canonical POSIX relative logical ids. The
    run-level qualification is the AGGREGATE across every pass (EXACT iff every
    pass is exact), never the first pass. Source-set provenance
    (``source_set_digest`` / algo / counts) is recomputed per pass via the
    accepted ``background_diagnostics`` helper.
    """
    bgd = _load_bgd()
    if CAPTURE_ARTIFACT_BASENAME != bgd.CAPTURE_ARTIFACT_BASENAME:  # pragma: no cover
        raise RunnerError(
            "capture artifact basename drift: runner mirrors "
            f"{CAPTURE_ARTIFACT_BASENAME!r} but the engine declares "
            f"{bgd.CAPTURE_ARTIFACT_BASENAME!r}"
        )
    q_exact = str(bgd.QUALIFICATION_EXACT)
    q_capped = str(bgd.QUALIFICATION_RESEARCH_ONLY_CAPPED)
    digest_algo = str(bgd.SOURCE_SET_DIGEST_ALGO)

    out_res = output_dir.resolve(strict=False)
    failures: list[str] = []
    passes: list[dict] = []

    root_manifest_path = output_dir / MANIFEST_NAME
    root_symlink = root_manifest_path.is_symlink()
    if root_symlink:
        failures.append("root manifest is a symlink (rejected)")
    root_ok, root_reason = (True, None) if root_symlink else _safe_regular_file(
        root_manifest_path, out_res)
    root_has = root_symlink is False and root_ok and root_manifest_path.is_file()
    if not root_symlink and root_manifest_path.exists() and not root_ok:
        failures.append(f"root manifest rejected ({root_reason})")

    pass_state: dict[str, str] = {}
    present: dict[str, bool] = {}
    for name in MOUNT_PASS_DIRS:
        p = output_dir / name
        if _pass_dir_is_symlink(output_dir, name):
            # A recognized pass PATH that is a symlink is rejected BEFORE any
            # manifest/artifact read — even when dangling or without a manifest.
            failures.append(f"recognized pass dir {name} is a symlink (rejected)")
            pass_state[name] = "symlink"
            present[name] = False
            continue
        if p.exists() and not p.is_dir():
            failures.append(f"recognized pass path {name} is not a directory (rejected)")
            pass_state[name] = "notdir"
            present[name] = False
            continue
        if p.is_dir():
            pass_state[name] = "dir"
            mf = p / MANIFEST_NAME
            ok, reason = _safe_regular_file(mf, out_res) if mf.exists() or mf.is_symlink() \
                else (False, None)
            if ok:
                present[name] = True
            else:
                present[name] = False
                if mf.is_symlink() or (mf.exists() and reason):
                    failures.append(f"{name} manifest rejected ({reason})")
        else:
            pass_state[name] = "absent"
            present[name] = False

    eq_has = present.get("grid_EQ", False)
    altz_has = present.get("grid_ALTZ", False)

    for unknown in _discover_unknown_manifests(output_dir):
        failures.append(f"unknown manifest under output dir: {unknown}")

    topology_ok = True
    symlink_fail = any(s == "symlink" for s in pass_state.values())
    if symlink_fail or root_symlink:
        topology_ok = False
    if root_has and (eq_has or altz_has):
        failures.append(
            "ambiguous topology: a root manifest coexists with mount-pass "
            "manifests (stale/ambiguous)"
        )
        topology_ok = False
    if not root_has and not (eq_has or altz_has):
        failures.append("no zegrid_manifest.json found under the output dir")
        topology_ok = False
    if eq_has != altz_has:
        failures.append("incomplete mount pair: grid_EQ and grid_ALTZ must both exist")
        topology_ok = False

    if root_has and not (eq_has or altz_has):
        pass_dirs = [output_dir]
    elif eq_has and altz_has and not root_has:
        pass_dirs = [output_dir / "grid_EQ", output_dir / "grid_ALTZ"]
    else:
        pass_dirs = []

    if not topology_ok or not pass_dirs:
        return {
            "verdict": "FAIL", "passes": [], "failures": failures,
            "multi_pass": False, "aggregate_heldout_exact": False,
            "aggregate_heldout_capped": True,
            "aggregate_qualification": q_capped,
            "selection_accounting": {},
        }

    pass_sets: list[set] = []
    for run_dir in pass_dirs:
        manifest_path = run_dir / MANIFEST_NAME
        artifact_path = run_dir / bgd.CAPTURE_ARTIFACT_BASENAME
        entry: dict = {
            "pass_dir": run_dir.name,
            "artifact_basename": artifact_path.name,
            "artifact_exists": False,
        }
        checks: dict[str, bool] = {}

        # The manifest we are about to read must itself be a regular non-symlink
        # file under the output root.
        m_ok, m_reason = _safe_regular_file(manifest_path, out_res)
        if not m_ok:
            failures.append(f"{run_dir.name}: manifest rejected ({m_reason})")
            checks["manifest_safe"] = False
            entry["checks"] = checks
            entry["pass_ok"] = False
            passes.append(entry)
            pass_sets.append(set())
            continue
        checks["manifest_safe"] = True

        manifest, m_err = _safe_manifest(manifest_path)
        if manifest is None:
            failures.append(f"{run_dir.name}: manifest rejected ({m_err})")
            checks["manifest_valid"] = False
            entry["checks"] = checks
            entry["pass_ok"] = False
            passes.append(entry)
            pass_sets.append(set())
            continue
        checks["manifest_valid"] = True
        entry["manifest_n_frames"] = manifest.get("n_frames")
        frame_ids_raw = [str(f) for f in (manifest.get("frame_ids") or [])]
        layout = manifest.get("layout") if isinstance(manifest.get("layout"), Mapping) else {}
        gauge = (manifest.get("photometric_gauge")
                 if isinstance(manifest.get("photometric_gauge"), Mapping) else {})
        cf = (manifest.get("cell_feather")
              if isinstance(manifest.get("cell_feather"), Mapping) else {})

        # canonical logical ids (fail closed on any malformed id)
        pass_ids: list[str] = []
        malformed: list[str] = []
        for fid in frame_ids_raw:
            try:
                pass_ids.append(_canonical_manifest_id(fid))
            except ValueError as exc:
                malformed.append(str(exc))
        pass_set = set(pass_ids)
        pass_sets.append(pass_set)
        entry["pass_frame_count"] = len(pass_set)
        checks["manifest_ids_canonical"] = not malformed
        if malformed:
            failures.append(
                f"{run_dir.name}: malformed manifest logical frame id(s): "
                + "; ".join(malformed[:5])
            )
        checks["pass_frames_unique"] = len(pass_ids) == len(pass_set)
        checks["pass_n_frames_consistent"] = (
            _is_int(manifest.get("n_frames")) and int(manifest["n_frames"]) == len(pass_set)
        )
        if not checks["pass_frames_unique"]:
            failures.append(f"{run_dir.name}: duplicate logical frame_ids within the pass")
        if not checks["pass_n_frames_consistent"]:
            failures.append(f"{run_dir.name}: manifest n_frames != unique logical frame_ids")

        cap = cf.get("r37_diagnostic_capture") if isinstance(cf, Mapping) else None
        if not isinstance(cap, Mapping):
            cap = None
        if cap is None:
            failures.append(f"{run_dir.name}: manifest has no r37_diagnostic_capture block")
            checks["manifest_capture_block"] = False
        else:
            checks["manifest_capture_block"] = True

        # The artifact must be a regular non-symlink file under the output root.
        if artifact_path.is_symlink():
            failures.append(f"{run_dir.name}: capture artifact is a symlink (rejected)")
            checks["artifact_safe"] = False
            checks["artifact_exists"] = False
            entry["checks"] = checks
            entry["pass_ok"] = False
            passes.append(entry)
            continue
        if not artifact_path.exists():
            failures.append(f"{run_dir.name}: capture artifact missing ({artifact_path.name})")
            checks["artifact_exists"] = False
            entry["checks"] = checks
            entry["pass_ok"] = False
            passes.append(entry)
            continue
        a_ok, a_reason = _safe_regular_file(artifact_path, out_res)
        if not a_ok:
            failures.append(f"{run_dir.name}: capture artifact rejected ({a_reason})")
            checks["artifact_safe"] = False
            entry["checks"] = checks
            entry["pass_ok"] = False
            passes.append(entry)
            continue
        checks["artifact_safe"] = True
        checks["artifact_exists"] = True
        entry["artifact_exists"] = True

        try:
            artifact = bgd.load_file(artifact_path)
        except Exception as exc:
            # category + basename only: never echo raw exception text (it may
            # embed an absolute path).
            failures.append(
                f"{run_dir.name}: artifact failed strict load "
                f"({type(exc).__name__}, {artifact_path.name})"
            )
            checks["artifact_strict_load"] = False
            entry["checks"] = checks
            entry["pass_ok"] = False
            passes.append(entry)
            continue
        checks["artifact_strict_load"] = True

        disk_digest, disk_bytes = _sha256_file(artifact_path)
        prov = dict(artifact.provenance)
        capture_block = dict(prov.get("capture") or {})

        # --- artifact-DERIVED truth (independent of manifest claims) ---------
        art_capped = bool(artifact.heldout_capped)
        art_exact = bool(artifact.heldout_exact())
        art_qualification = q_capped if art_capped else q_exact
        art_source_count = capture_block.get("n_source_frames")
        recomputed_digest = str(bgd.source_set_digest(pass_ids))

        entry.update({
            "capture_id": capture_block.get("capture_id"),
            "run_label": capture_block.get("run_label"),
            "n_source_frames": art_source_count,
            "sha256": disk_digest,
            "bytes": disk_bytes,
            "artifact_qualification": art_qualification,
            "artifact_heldout_capped": art_capped,
            "artifact_heldout_exact": art_exact,
            "manifest_qualification": (cap or {}).get("qualification"),
            "manifest_heldout_capped": (cap or {}).get("holdout_capped"),
            "manifest_heldout_exact": (cap or {}).get("holdout_exact"),
            "recomputed_source_set_digest": recomputed_digest,
            "source_set_digest_algo": digest_algo,
            "n_relations": len(artifact.observations),
            "n_dropped": len(artifact.dropped),
            "artifact_meta_nx": int(artifact.meta["nx"]),
            "artifact_meta_ny": int(artifact.meta["ny"]),
            "capture_stage": prov.get("capture_stage"),
            "mission": prov.get("mission"),
            "production_wired": prov.get("production_wired"),
            "correction_applied": prov.get("correction_applied"),
        })

        # --- linkage: feed ARTIFACT/DISK-derived values, never manifest copies --
        linkage_meta = {
            "capture_id": capture_block.get("capture_id"),
            "filename": artifact_path.name,
            "sha256": disk_digest,
            "bytes": disk_bytes,
        }
        checks["manifest_links_artifact"] = bool(bgd.manifest_links_artifact(linkage_meta, cap))
        checks["capture_id_matches_manifest"] = (
            capture_block.get("capture_id") is not None
            and capture_block.get("capture_id") == (cap or {}).get("capture_id")
        )
        checks["filename_matches_manifest"] = (cap or {}).get("filename") == artifact_path.name
        checks["sha256_matches_disk"] = (cap or {}).get("sha256") == disk_digest
        checks["bytes_matches_disk"] = (cap or {}).get("bytes") == disk_bytes
        checks["mission_truthful"] = prov.get("mission") == CAPTURE_MISSION
        checks["stage_truthful"] = prov.get("capture_stage") == CAPTURE_STAGE
        checks["production_wired_true"] = prov.get("production_wired") is True
        checks["correction_not_applied"] = prov.get("correction_applied") is False
        checks["gauge_mode_per_cell"] = (
            capture_block.get("effective_cell_gauge_mode") == CELL_GAUGE_MODE_PER_CELL
        )
        checks["layout_matches_baseline"] = (
            int(artifact.meta["nx"]) == baseline.nx and int(artifact.meta["ny"]) == baseline.ny
        )
        checks["manifest_layout_matches_baseline"] = (
            int(layout.get("nx", -1)) == baseline.nx
            and int(layout.get("ny", -1)) == baseline.ny
        )
        checks["manifest_gauge_per_cell"] = (
            gauge.get("mode") == CELL_GAUGE_MODE_PER_CELL
        )
        # source-set provenance (independent recomputation)
        checks["artifact_source_set_digest_matches"] = (
            capture_block.get("source_set_digest") == recomputed_digest
        )
        checks["artifact_source_set_digest_algo_matches"] = (
            capture_block.get("source_set_digest_algo") == digest_algo
        )
        checks["manifest_source_set_digest_matches"] = (
            (cap or {}).get("source_set_digest") == recomputed_digest
        )
        checks["manifest_source_set_digest_algo_matches"] = (
            (cap or {}).get("source_set_digest_algo") == digest_algo
        )
        checks["artifact_source_count_matches_pass"] = (
            _is_int(art_source_count) and int(art_source_count) == len(pass_set)
        )
        checks["manifest_source_count_matches_pass"] = (
            (cap or {}).get("n_source_frames") == len(pass_set)
        )
        # artifact-derived qualification vs manifest claims (all three)
        checks["artifact_qualification_matches_manifest"] = (
            (cap or {}).get("qualification") == art_qualification
        )
        checks["artifact_heldout_capped_matches_manifest"] = (
            (cap or {}).get("holdout_capped") == art_capped
        )
        checks["artifact_heldout_exact_matches_manifest"] = (
            (cap or {}).get("holdout_exact") == art_exact
        )
        checks["manifest_qualification_consistent"] = (
            (cap or {}).get("qualification")
            == (q_capped if (cap or {}).get("holdout_capped") else q_exact)
        )

        for key, val in checks.items():
            if not val:
                failures.append(f"{run_dir.name}: check failed: {key}")
        entry["checks"] = checks
        entry["pass_ok"] = all(checks.values())
        passes.append(entry)

    multi_pass = len(pass_dirs) > 1
    union: set = set()
    disjoint = True
    for s in pass_sets:
        if union & s:
            disjoint = False
        union |= s
    if multi_pass:
        accounting = {
            "pass_frame_sets_disjoint": disjoint,
            "union_matches_selection_count": len(union) == selection.count,
            "union_matches_selection_digest": _digest_names(union) == selection.digest,
        }
    else:
        accounting = {
            "single_pass_matches_selection_count": len(union) == selection.count,
            "single_pass_matches_selection_digest": _digest_names(union) == selection.digest,
        }
    for key, val in accounting.items():
        if not val:
            failures.append(f"selection accounting: check failed: {key}")

    # HIGH: run-level qualification is the AGGREGATE across every pass.
    aggregate_exact = bool(passes) and all(bool(p.get("artifact_heldout_exact")) for p in passes)
    aggregate_capped = any(bool(p.get("artifact_heldout_capped")) for p in passes)
    aggregate_qualification = q_exact if aggregate_exact else q_capped

    verdict = "PASS" if (not failures and all(p.get("pass_ok") for p in passes)) else "FAIL"
    return {
        "verdict": verdict,
        "passes": passes,
        "failures": failures,
        "multi_pass": multi_pass,
        "selection_accounting": accounting,
        "union_frame_count": len(union),
        "aggregate_heldout_exact": aggregate_exact,
        "aggregate_heldout_capped": aggregate_capped,
        "aggregate_qualification": aggregate_qualification,
    }


#: Embedded absolute-path tokens (as opposed to start-anchored whole strings).
#: The POSIX form requires >= 2 separators so ordinary prose (``WCS/shape``)
#: is NOT mangled. Known absolute roots are handled separately (they may embed
#: spaces, which regexes cannot capture reliably).
_EMBEDDED_UNC_RE = re.compile(r"\\\\[^\s'\"<>()\[\]]+")
_EMBEDDED_WIN_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>()\[\]]+")
_EMBEDDED_POSIX_RE = re.compile(r"/(?:[^\s'\"<>()\[\]]+/)+[^\s'\"<>()\[\]]*")


def _basename_of_token(token: str) -> str:
    t = str(token).replace("\\", "/").rstrip("/")
    base = t.rsplit("/", 1)[-1]
    return base or "<abs-path>"


def _scrub_string(value: str, roots: Sequence[str]) -> str:
    out = value
    # 1) known absolute roots (robust to embedded spaces), replaced by basename.
    for root in roots:
        if not root:
            continue
        try:
            base = PurePosixPath(str(root).replace("\\", "/")).name or "<root>"
        except Exception:
            base = "<root>"
        out = out.replace(str(root), base)
    # 2) embedded generic absolute tokens anywhere in the text.
    out = _EMBEDDED_UNC_RE.sub(lambda m: _basename_of_token(m.group(0)), out)
    out = _EMBEDDED_WIN_RE.sub(lambda m: _basename_of_token(m.group(0)), out)
    out = _EMBEDDED_POSIX_RE.sub(lambda m: _basename_of_token(m.group(0)), out)
    return out


def _scrub_absolute_paths(obj: Any, roots: Sequence[str] = ()) -> Any:
    """Recursively remove private absolute paths from a durable-report structure.

    Defense-in-depth at the durable-report boundary: the runner may use real
    absolute paths during execution, but the persisted verification report must
    carry only non-sensitive basename/count/digest/boolean facts. Removes BOTH
    whole-value paths AND paths embedded after text (POSIX / Windows / UNC,
    including paths with spaces via the known ``roots``).
    """
    if isinstance(obj, str):
        return _scrub_string(obj, roots)
    if isinstance(obj, list):
        return [_scrub_absolute_paths(v, roots) for v in obj]
    if isinstance(obj, tuple):
        return [_scrub_absolute_paths(v, roots) for v in obj]
    if isinstance(obj, Mapping):
        return {k: _scrub_absolute_paths(v, roots) for k, v in obj.items()}
    return obj


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic same-directory JSON write (temp + fsync + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def build_verification_report(
    *,
    invocation: Invocation,
    baseline: BaselineInfo,
    selection: SelectionResolution,
    verification: Mapping[str, Any],
    preflight_report: Mapping[str, Any] | None,
    pinned_layout: str,
    holdout_cap: int,
    gpu_mode: str,
) -> dict:
    """Compact verification report — basenames/counts/digests only (no private abs paths).

    The exact/capped/qualification facts come from the artifact-DERIVED truth on
    the verification entries, never from manifest claims. Absolute paths are
    scrubbed at this durable-report boundary.
    """
    multi_pass = bool(verification.get("multi_pass"))
    aggregate_exact = bool(verification.get("aggregate_heldout_exact"))
    aggregate_capped = bool(verification.get("aggregate_heldout_capped"))
    aggregate_qualification = verification.get("aggregate_qualification")

    pass_summaries = []
    artifact = None
    for p in verification.get("passes", ()):
        pass_summaries.append({
            "pass_dir": p.get("pass_dir"),
            "frames": p.get("pass_frame_count"),
            "artifact_exists": bool(p.get("artifact_exists")),
            "artifact_sha256": p.get("sha256"),
            "artifact_bytes": p.get("bytes"),
            "capture_id": p.get("capture_id"),
            "artifact_qualification": p.get("artifact_qualification"),
            "artifact_heldout_exact": p.get("artifact_heldout_exact"),
            "artifact_heldout_capped": p.get("artifact_heldout_capped"),
            "recomputed_source_set_digest": p.get("recomputed_source_set_digest"),
            "pass_ok": bool(p.get("pass_ok")),
        })
        # A single pass may present its full artifact; a MULTI-pass run must NOT
        # present one pass's sha/capture id as if it were the whole run.
        if (not multi_pass) and artifact is None and p.get("artifact_exists"):
            artifact = {
                "pass_dir": p.get("pass_dir"),
                "basename": p.get("artifact_basename"),
                "sha256": p.get("sha256"),
                "bytes": p.get("bytes"),
                "capture_id": p.get("capture_id"),
                "run_label": p.get("run_label"),
                "n_source_frames": p.get("n_source_frames"),
                "n_relations": p.get("n_relations"),
                "n_dropped": p.get("n_dropped"),
                # artifact-DERIVED truth:
                "qualification": p.get("artifact_qualification"),
                "heldout_capped": p.get("artifact_heldout_capped"),
                "heldout_exact": p.get("artifact_heldout_exact"),
                "capture_stage": p.get("capture_stage"),
                "mission": p.get("mission"),
                "production_wired": p.get("production_wired"),
                "correction_applied": p.get("correction_applied"),
            }
    if multi_pass:
        artifact = {
            "multi_pass": True,
            "n_passes": len(pass_summaries),
            "aggregate_qualification": aggregate_qualification,
            "aggregate_heldout_exact": aggregate_exact,
            "aggregate_heldout_capped": aggregate_capped,
            "union_frame_count": verification.get("union_frame_count"),
            "per_pass_artifacts": [
                {"pass_dir": s["pass_dir"], "sha256": s["artifact_sha256"],
                 "bytes": s["artifact_bytes"], "capture_id": s["capture_id"],
                 "qualification": s["artifact_qualification"],
                 "heldout_exact": s["artifact_heldout_exact"]}
                for s in pass_summaries
            ],
        }

    temp_dir = str(invocation.params_report.get("temporary_work_dir") or "")
    roots = [str(invocation.input_dir), str(baseline.run_dir),
             str(invocation.output_dir), temp_dir]

    report = {
        "schema": VERIFICATION_SCHEMA,
        "version": VERIFICATION_VERSION,
        "mission": "ZM-ZEGRID-R37-E",
        "runner": "tools/r37_capture_runner.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_version": _package_version(),
        "repo_head": _git_head(_repo_root()),
        "verification_verdict": verification.get("verdict"),
        # run-level AGGREGATE truth (never the first pass):
        "qualification": aggregate_qualification,
        "exact_heldout": aggregate_exact,
        "heldout_capped": aggregate_capped,
        "multi_pass": multi_pass,
        "command_parameters": {
            "input_dir_basename": invocation.input_dir.name,
            "baseline_run_dir_name": baseline.run_dir.name,
            "output_dir_name": invocation.output_dir.name,
            "pinned_layout": pinned_layout,
            "cell_gauge_mode": CELL_GAUGE_MODE_PER_CELL,
            "holdout_cap": int(holdout_cap),
            "gpu_mode": gpu_mode,
            "r37_diagnostic_capture": True,
            "selection": selection.to_dict(),
        },
        "invocation_parameters": dict(invocation.params_report),
        "preflight": dict(preflight_report) if preflight_report is not None else {"skipped": True},
        "artifact": artifact,
        "passes": pass_summaries,
        "selection_accounting": dict(verification.get("selection_accounting") or {}),
        "linkage": {
            "verdict": verification.get("verdict"),
            "failure_count": len(list(verification.get("failures") or [])),
        },
        "failures": list(verification.get("failures") or []),
    }
    # Durable-report boundary: strip private absolute paths (whole-value or
    # embedded after text) via known roots + generic patterns.
    return _scrub_absolute_paths(report, roots)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def _console_progress(*args, **kwargs) -> None:
    parts = [str(a) for a in args if a is not None]
    if parts:
        print("[r37-runner] " + " ".join(parts), flush=True)


def _print_block(title: str, payload: Mapping[str, Any]) -> None:
    print(f"--- {title} ---", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r37_capture_runner",
        description=(
            "Reproducible, dry-run-by-default CLI runner for the R37-D diagnostic "
            "capture seam (ZM-ZEGRID-R37-E)."
        ),
    )
    parser.add_argument("--input-dir", required=True,
                        help="folder containing the source FITS frames")
    parser.add_argument("--baseline-run-dir", required=True,
                        help="run dir containing zegrid_manifest.json + run_config_snapshot.json")
    parser.add_argument("--output-dir", required=True,
                        help="NEW/empty destination for the capture run (never the baseline dir)")
    parser.add_argument("--holdout-cap", type=int, default=DEFAULT_HOLDOUT_CAP,
                        help=f"exact-heldout target (default {DEFAULT_HOLDOUT_CAP})")
    parser.add_argument("--gpu", choices=("auto", "on", "off"), default="auto",
                        help="GPU preference; 'auto' prefers the baseline config (use_gpu_grid)")
    parser.add_argument("--execute", action="store_true",
                        help="perform the real capture (required for any write/full run)")
    parser.add_argument("--skip-memory-preflight", action="store_true",
                        help=("inspection dry-run ONLY: skip the real memory preflight "
                              "and report 'preflight SKIPPED' (never PASS). Refused with "
                              "--execute."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        baseline = load_baseline(args.baseline_run_dir)
    except BaselineError as exc:
        print(f"[r37-runner] BASELINE ERROR: {exc}", file=sys.stderr, flush=True)
        return _EXIT_ERROR

    try:
        selection = resolve_selection(args.input_dir, baseline.frame_ids)
    except SelectionError as exc:
        print(f"[r37-runner] SELECTION ERROR: {exc}", file=sys.stderr, flush=True)
        return _EXIT_ERROR

    try:
        cfg = build_pinned_config(baseline.snapshot, baseline.nx, baseline.ny)
        invocation = build_invocation(
            baseline, selection, baseline.snapshot, cfg,
            gpu_mode=args.gpu, holdout_cap=args.holdout_cap,
        )
    except ConfigError as exc:
        print(f"[r37-runner] CONFIG ERROR: {exc}", file=sys.stderr, flush=True)
        return _EXIT_ERROR

    try:
        out_dir = validate_output_safety(
            args.output_dir,
            baseline_run_dir=baseline.run_dir,
            input_dir=args.input_dir,
        )
    except OutputSafetyError as exc:
        print(f"[r37-runner] UNSAFE OUTPUT: {exc}", file=sys.stderr, flush=True)
        return _EXIT_UNSAFE_OUTPUT
    invocation.output_dir = out_dir

    print(
        f"[r37-runner] baseline {baseline.run_dir.name}: {len(baseline.frame_ids)} frame_ids, "
        f"layout {baseline.nx}x{baseline.ny}, canvas {baseline.canvas_width}x"
        f"{baseline.canvas_height}, gauge {baseline.gauge_mode}",
        flush=True,
    )
    print(
        f"[r37-runner] resolved {selection.count} source frame(s) under "
        f"{selection.input_dir.name} (relative={selection.matched_by_relative}, "
        f"basename={selection.matched_by_basename}), digest "
        f"{selection.digest[:16]}…",
        flush=True,
    )
    _print_block("EFFECTIVE INVOCATION PARAMETERS", invocation.params_report)

    preflight_report: dict | None = None
    preflight_verdict = "SKIPPED"

    # M1 (hard memory-safety contract): --execute REQUIRES a real successful
    # preflight. Skipping it is an inspection-only affordance and is refused
    # before ANY engine work — it is not a force-infeasible escape hatch.
    if args.skip_memory_preflight and args.execute:
        print(
            "[r37-runner] REFUSED: --skip-memory-preflight cannot be combined with "
            "--execute. A real memory preflight is REQUIRED before any engine work "
            "(no --force-infeasible escape hatch by design).",
            file=sys.stderr, flush=True,
        )
        return _EXIT_PREFLIGHT_FAIL

    if args.skip_memory_preflight:
        print(
            "[r37-runner] WARNING: --skip-memory-preflight given — the real memory "
            "preflight is SKIPPED (inspection dry-run only; NO execution). The pinned "
            "layout's feasibility is NOT established on this host.",
            file=sys.stderr, flush=True,
        )
        preflight_report = {"verdict": "SKIPPED", "skipped": True}
        preflight_verdict = "SKIPPED"
    else:
        print("[r37-runner] running memory/capability preflight (no stacking, no writes)…",
              flush=True)
        pf = preflight(args.input_dir, selection, baseline, cfg, progress=_console_progress)
        preflight_report = pf.report
        preflight_verdict = pf.verdict
        _print_block("PREFLIGHT", preflight_report)
        if not pf.ok:
            print(
                f"[r37-runner] PREFLIGHT FAIL: pinned layout {baseline.nx}x{baseline.ny} "
                f"cannot fit on this host. {pf.failure}",
                file=sys.stderr, flush=True,
            )
            if args.execute:
                print(
                    "[r37-runner] refusing --execute on a failed preflight "
                    "(no --force-infeasible by design).",
                    file=sys.stderr, flush=True,
                )
            return _EXIT_PREFLIGHT_FAIL

    if not args.execute:
        print(
            f"[r37-runner] DRY-RUN complete: preflight {preflight_verdict}, nothing "
            "written. Re-run with --execute to perform the capture.",
            flush=True,
        )
        return _EXIT_OK

    # --- execute (a real preflight PASS is guaranteed above) ----------------
    print(f"[r37-runner] EXECUTE → {out_dir}", flush=True)
    try:
        _call_run_zegrid_mode(
            str(selection.input_dir), str(out_dir),
            progress_callback=_console_progress, **invocation.kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - surface the real failure verbatim
        print(f"[r37-runner] CAPTURE RUN FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return _EXIT_ERROR

    print("[r37-runner] run finished — verifying artifact↔manifest linkage…", flush=True)
    try:
        verification = verify_outputs(out_dir, baseline=baseline, selection=selection)
    except Exception as exc:  # noqa: BLE001 - structured failure, no traceback
        print(
            f"[r37-runner] verification error ({type(exc).__name__}); recording a "
            "structured FAIL.",
            file=sys.stderr, flush=True,
        )
        verification = {
            "verdict": "FAIL", "passes": [], "multi_pass": False,
            "failures": [f"verification error ({type(exc).__name__})"],
            "aggregate_heldout_exact": False, "aggregate_heldout_capped": True,
            "aggregate_qualification": "RESEARCH_ONLY_HELDOUT_CAPPED",
            "selection_accounting": {},
        }
    report = build_verification_report(
        invocation=invocation, baseline=baseline, selection=selection,
        verification=verification, preflight_report=preflight_report,
        pinned_layout=f"{baseline.nx}x{baseline.ny}", holdout_cap=args.holdout_cap,
        gpu_mode=args.gpu,
    )
    # M5: the durable verification report is part of the contract — a failed
    # atomic write is a failure, and must never be followed by a PASS claim.
    try:
        _atomic_write_json(out_dir / REPORT_NAME, report)
        print(f"[r37-runner] wrote {REPORT_NAME}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[r37-runner] FAILED to write verification report {REPORT_NAME}: {exc} "
            "— aborting (nonzero).",
            file=sys.stderr, flush=True,
        )
        return _EXIT_ERROR

    if verification.get("verdict") != "PASS":
        print("[r37-runner] POST-RUN VERIFICATION FAILED:", file=sys.stderr, flush=True)
        for item in verification.get("failures", ())[:20]:
            print(f"  - {item}", file=sys.stderr, flush=True)
        return _EXIT_ERROR

    exactly = bool(report.get("exact_heldout"))
    exact_label = "EXACT" if exactly else "CAPPED (not exact)"
    if report.get("multi_pass"):
        detail = f"passes={len(report.get('passes') or [])}"
    else:
        art = report.get("artifact") or {}
        detail = f"sha256={art.get('sha256')} bytes={art.get('bytes')}"
    print(
        f"[r37-runner] VERIFICATION PASS — aggregate qualification "
        f"{report.get('qualification')} ({exact_label}); {detail}",
        flush=True,
    )
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
