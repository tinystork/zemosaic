#!/usr/bin/env python3
"""ZM-ZEGRID-R40-C — bounded, reviewed runner for the R40 DEV diagnostic capture.

WHY THIS EXISTS
---------------
The R40 diagnostic-capture seam (``ZM-ZEGRID-R40-B2`` families A..I, committed at
HEAD ``222a624``) is **default-OFF** and only reachable from the production
``per_cell`` assembly seam through three explicit kwargs of
``run_zegrid_mode``::

    r40_seam_capture=True
    r40_output_root="<fresh capture output path>"
    r40_seam_capture_disk_budget_bytes=<int>

The R40-C HUMAN_GATE authorizes exactly **one** DEV Windows816 diagnostic capture
(``DEV_NOT_QUALIFICATION``) on a capable Windows host. Hand-rolling that call is
error-prone and silently degradable; this tool makes it runnable with EXACT,
safely-validated commands, mirroring the safety/CLI philosophy of the accepted
``tools/r37_capture_runner.py`` (the R37-E runner).

SAFETY MODEL (read this)
------------------------
* **Dry-run by default.** Without ``--execute`` NOTHING is written anywhere. The
  dry run validates every argument, the output-destination safety and the frozen
  core bindings, prints the EXACT ``run_zegrid_mode`` kwargs it *would* use, and
  exits ``0``.
* ``--execute`` calls ``run_zegrid_mode`` **directly** — no shell, no subprocess —
  with the explicit, default-OFF seam above. Every other run stays default-OFF and
  behaviour-neutral.
* ``cell_gauge_mode="per_cell"`` is REQUIRED (the only seam that carries the R40
  capture); any other value is refused BEFORE any work.
* The input dataset is **read-only**: the tool lists source FITS *names* only; it
  never opens a source frame and NEVER reads the final product FITS
  (science/aesthetic/coverage).
* After the run the tool INDEPENDENTLY verifies the published generation with
  ``verify_generation(..., expected_production_wired=True)`` and REFUSES/FAILS
  (non-zero) if ``ok is not True`` or ``problems`` is non-empty.
* Fail-closed: the frozen gates (uncapped EXACT held-out selection per relation,
  a fully wired ``114/114`` relation-channel panel, ``n_dropped == 0``, inherited
  support gates) are the *core's* to enforce. This runner never weakens them and
  never degrades a capped/uncertain run into a "best effort" capture.
* The durable provenance report is strict JSON (``allow_nan=False``) with
  **basenames only** — no absolute private paths — and is refused if the accepted
  core privacy scan finds a path leak.

PATHS AND THE FROZEN CORE (important, and honest)
-------------------------------------------------
The committed R40 core resolves the capture root through
``r40_capture.resolve_fresh_output_root``, which (frozen, not changed here)
REFUSES drive-absolute (``C:\\...``) roots, UNC roots and ``..`` traversal. A
Windows absolute capture root therefore cannot be used as-is. This runner
requests the capture under the same physical directory as ``--output-dir`` while
passing the core a **relative** capture root whenever the drive-absolute form
would be refused — and it REFUSES with an actionable message when no safe
relative form exists (e.g. the output path lies on another drive than the CWD).
On Windows, ``cd`` to an ancestor of your output path (for example your data
root) so the output path is a plain relative path.

HOW TO RUN (Windows PowerShell; dry-run then execute)
-----------------------------------------------------
::

    cd C:\\Users\\TRISTAN\\Desktop\\astrophoto
    python C:\\Users\\TRISTAN\\zemosaic\\tools\\r40_capture_runner.py `
        --input-dir "lights organized\\EQ\\IRCUT" `
        --output-dir "OUT\\R40C_CAPTURE_222a624"

    python C:\\Users\\TRISTAN\\zemosaic\\tools\\r40_capture_runner.py `
        --input-dir "lights organized\\EQ\\IRCUT" `
        --output-dir "OUT\\R40C_CAPTURE_222a624" `
        --execute

Exit codes: ``0`` ok, ``1`` validation/other error, ``2`` post-run verification
FAIL, ``3`` unsafe output destination.

Importing this module is intentionally side-effect-free: the ZeMosaic engine and
the R40 core are imported lazily, inside the functions that need them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Fixed names / defaults
# ---------------------------------------------------------------------------

MISSION = "ZM-ZEGRID-R40-C-PREREG-RUNNER"
CAPTURE_MISSION = "ZM-ZEGRID-R40-C"
CAPTURE_STAGE = "pre_offset_pre_feather"
CELL_GAUGE_MODE_PER_CELL = "per_cell"

#: ``--holdout-cap`` value meaning "uncapped EXACT held-out selection". The R40
#: seam has NO cap parameter by construction (frozen §1.3 exact selection), so any
#: requested cap would be a degradation and is REFUSED (never silently ignored).
UNCAPPED_HOLDOUT_CAP = 0

REPORT_NAME = "r40_capture_verification.json"
REPORT_SCHEMA = "zemosaic.zegrid.r40c.capture_verification"
REPORT_VERSION = 1

#: Mirrors of the frozen ``zemosaic.core.zegrid.r40_capture`` constants. They are
#: asserted against the real core before any dry-run/execute decision (a mismatch
#: is a hard refusal, never a warning).
CAPTURE_CONTRACT_VERSION = "r40-capture-1"
GENERATION_DIR_PREFIX = "r40_capture_"
GENERATION_INDEX_BASENAME = "index.json"
GENERATION_COMMIT_BASENAME = "COMMIT"
SOURCE_SET_DIGEST_ALGO = "sha256_sorted_unique_relative_logical_paths_v1"
CORE_DEFAULT_DISK_BUDGET_BYTES = 256 * 1024 * 1024
B1_PROTOCOL_SHA256 = "d689b2d78cebebad0222f503f804a8d6c9a2bf60316c88067b1851e1a447ffa7"
B1_CRITERIA_SHA256 = "67be63e4423f782646f71bc350be5ff5a33a2007b67f0999a76a841154da92b9"
B1_RESULT_SHA256 = "d597cd1d67b0cd91593e1da777ed2dde4ff0105864cecd63fbfe259d327ab474"
B1_CLAIM_ID = "c965187cd826ade68c556a59e1f9373d5a4dc500"

FITS_SUFFIXES = (".fit", ".fits")

_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_VERIFY_FAIL = 2
_EXIT_UNSAFE_OUTPUT = 3

_TOOL_BASENAME = "r40_capture_runner.py"

#: Import caches (never populated at import time).
_CACHE: dict[str, Any] = {}


class RunnerError(RuntimeError):
    """Base class for every runner refusal."""


class OutputSafetyError(RunnerError):
    """The output destination is unsafe (overlap / non-empty / symlink / traversal)."""


class BindingError(RunnerError):
    """A frozen core binding does not match the recorded constant."""


class VerificationError(RunnerError):
    """The published generation failed independent verification."""


# ---------------------------------------------------------------------------
# Lazy imports (import of this module performs NO engine work)
# ---------------------------------------------------------------------------

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
    mod = _CACHE.get("zz")
    if mod is None:
        _ensure_src_on_path()
        from zemosaic import zemosaic_zegrid_mode as mod  # local import: intentional

        _CACHE["zz"] = mod
    return mod


def _load_core():
    """Import (once) and return ``zemosaic.core.zegrid.r40_capture``."""
    mod = _CACHE.get("r40c")
    if mod is None:
        _ensure_src_on_path()
        from zemosaic.core.zegrid import r40_capture as mod  # local import

        _CACHE["r40c"] = mod
    return mod


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

_DRIVE_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"^\\\\[^\\]+\\")
_FILE_URI_RE = re.compile(r"^file://", re.IGNORECASE)


def _has_traversal(raw: str) -> bool:
    return any(part == ".." for part in re.split(r"[\\/]+", str(raw)))


def _is_drive_absolute(raw: str) -> bool:
    return bool(_DRIVE_ABSOLUTE_RE.match(str(raw).strip()))


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), int(n)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head(repo_root: Path) -> str | None:
    """Best-effort HEAD sha WITHOUT invoking a subprocess (mirrors R37-E)."""
    try:
        git = repo_root / ".git"
        if git.is_file():  # worktree / submodule pointer file
            text = git.read_text(encoding="utf-8").strip()
            if text.lower().startswith("gitdir:"):
                target = Path(text.split(":", 1)[1].strip())
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


def _basename(value: str) -> str:
    text = str(value).replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


def _looks_like_absolute(token: str) -> bool:
    t = str(token)
    return bool(
        _DRIVE_ABSOLUTE_RE.match(t) or _UNC_RE.match(t) or _FILE_URI_RE.match(t)
        or t.startswith("/")
    )


def scrub_command(argv: Sequence[str]) -> list[str]:
    """Reduce any absolute path token in a recorded command line to its basename.

    The durable report must never carry absolute private paths. Flags are kept
    verbatim; only path-ish *values* are reduced (to their basename).
    """
    out: list[str] = []
    for token in argv:
        text = str(token)
        if text.startswith("-"):
            out.append(text)
        elif _looks_like_absolute(text):
            out.append(_basename(text) or "<path>")
        else:
            out.append(text)
    return out


def scan_fits(root: Path) -> list[Path]:
    """Read-only listing of the FITS inventory under ``root`` (names only)."""
    out: list[Path] = []
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.suffix.lower() in FITS_SUFFIXES:
                out.append(p)
        except OSError:  # pragma: no cover - defensive
            continue
    out.sort(key=lambda p: p.as_posix().lower())
    return out


# ---------------------------------------------------------------------------
# Frozen-core bindings + signature contract
# ---------------------------------------------------------------------------

def assert_frozen_bindings(core: Any) -> dict:
    """Assert the recorded frozen constants against the real core (fail closed)."""
    expected = {
        "CAPTURE_CONTRACT_VERSION": CAPTURE_CONTRACT_VERSION,
        "GENERATION_DIR_PREFIX": GENERATION_DIR_PREFIX,
        "GENERATION_INDEX_BASENAME": GENERATION_INDEX_BASENAME,
        "GENERATION_COMMIT_BASENAME": GENERATION_COMMIT_BASENAME,
        "SOURCE_SET_DIGEST_ALGO": SOURCE_SET_DIGEST_ALGO,
        "DEFAULT_DISK_BUDGET_BYTES": CORE_DEFAULT_DISK_BUDGET_BYTES,
        "B1_PROTOCOL_SHA256": B1_PROTOCOL_SHA256,
        "B1_CRITERIA_SHA256": B1_CRITERIA_SHA256,
        "B1_RESULT_SHA256": B1_RESULT_SHA256,
        "B1_CLAIM_ID": B1_CLAIM_ID,
    }
    mismatches = []
    for name, want in expected.items():
        got = getattr(core, name, None)
        if got != want:
            mismatches.append(f"{name}: core={got!r} recorded={want!r}")
    if mismatches:
        raise BindingError(
            "frozen R40 core bindings do not match the recorded constants; refusing: "
            + "; ".join(mismatches)
        )
    return {
        "capture_contract_version": str(core.CAPTURE_CONTRACT_VERSION),
        "b1_protocol_sha256": str(core.B1_PROTOCOL_SHA256),
        "b1_criteria_sha256": str(core.B1_CRITERIA_SHA256),
        "b1_result_sha256": str(core.B1_RESULT_SHA256),
        "b1_claim_id": str(core.B1_CLAIM_ID),
    }


def assert_kwargs_supported(kwargs: Mapping[str, Any], *, zz: Any | None = None) -> dict:
    """Assert every planned kwarg is a REAL, keyword-only parameter of the seam.

    A misspelled/unsupported kwarg must never be silently swallowed by a
    ``**kwargs`` catch-all (that would run a *different* capture than the one
    described in the plan).
    """
    import inspect

    zz = zz or _load_zegrid()
    sig = inspect.signature(zz.run_zegrid_mode)
    problems = []
    for name, value in kwargs.items():
        param = sig.parameters.get(name)
        if param is None:
            problems.append(f"{name}: not a parameter of run_zegrid_mode")
            continue
        if param.kind is not inspect.Parameter.KEYWORD_ONLY:
            problems.append(f"{name}: expected keyword-only, got {param.kind.name}")
        if param.default is inspect.Parameter.empty and value is None:
            problems.append(f"{name}: required parameter has no value")
    if problems:
        raise BindingError(
            "planned kwargs do not match run_zegrid_mode's signature; refusing: "
            + "; ".join(problems)
        )
    return {name: sig.parameters[name].default for name in kwargs}


# ---------------------------------------------------------------------------
# Output-destination safety (mirrors the accepted R37-E refusal semantics)
# ---------------------------------------------------------------------------

def _resolve_or_none(path: Path) -> Path | None:
    try:
        return path.expanduser().resolve(strict=False)
    except Exception:  # pragma: no cover - defensive
        return None


def _is_same_or_nested(a: Path, b: Path) -> bool:
    ar, br = _resolve_or_none(a), _resolve_or_none(b)
    if ar is None or br is None:
        return False
    return ar == br or ar in br.parents or br in ar.parents


def _assert_no_symlink_components(path: Path) -> None:
    """Refuse when ``path`` or any existing ancestor component is a symlink."""
    try:
        parts = Path(path).absolute().parts
    except OSError as exc:  # pragma: no cover - defensive
        raise OutputSafetyError(f"cannot inspect output path components: {exc}") from exc
    acc = Path(parts[0])
    for part in parts[1:]:
        acc = acc / part
        try:
            if acc.is_symlink():
                raise OutputSafetyError(
                    f"symlinked path component refused: {acc.name!r}"
                )
        except OSError as exc:  # pragma: no cover - defensive
            raise OutputSafetyError(f"cannot stat path component {acc.name!r}") from exc


def validate_output_safety(
    output_dir: str | os.PathLike[str],
    *,
    input_dir: str | os.PathLike[str],
    protected_roots: Sequence[str | os.PathLike[str]] = (),
) -> Path:
    """Require a NEW/EMPTY output dir that cannot overwrite input or a baseline."""
    raw = str(output_dir)
    if not raw.strip():
        raise OutputSafetyError("output dir is empty; refusing")
    if _FILE_URI_RE.match(raw.strip()):
        raise OutputSafetyError("file:// output dirs are refused")
    if _has_traversal(raw):
        raise OutputSafetyError("parent-traversal output dir is refused")
    out = Path(raw).expanduser()
    _assert_no_symlink_components(out)
    if out.is_symlink():
        raise OutputSafetyError(f"symlinked output dir is refused: {out.name}")

    for label, other in (("input dir", input_dir),):
        if _is_same_or_nested(out, Path(other)):
            raise OutputSafetyError(
                f"output dir {out.name!r} overlaps the {label} {Path(other).name!r} — "
                "refusing to run"
            )
    for root in protected_roots:
        if _is_same_or_nested(out, Path(root)):
            raise OutputSafetyError(
                f"output dir {out.name!r} overlaps the protected root "
                f"{Path(root).name!r} — refusing to run"
            )

    if out.exists():
        if not out.is_dir():
            raise OutputSafetyError(f"output path exists and is not a directory: {out.name!r}")
        try:
            entries = list(out.iterdir())
        except OSError as exc:
            raise OutputSafetyError(f"cannot inspect output dir {out.name!r}: {exc}") from exc
        if entries:
            raise OutputSafetyError(
                f"output dir {out.name!r} exists and is not empty ({len(entries)} entries); "
                "refusing to run (use a new/empty destination)"
            )
    return out


# ---------------------------------------------------------------------------
# Capture root: the frozen core refuses drive-absolute / UNC / traversal roots
# ---------------------------------------------------------------------------

def resolve_capture_root(
    output_dir: str | os.PathLike[str],
    *,
    core: Any | None = None,
    cwd: str | Path | None = None,
) -> tuple[str, bool]:
    """Return ``(capture_root_argument, relativized)`` for ``r40_output_root``.

    The committed core's ``resolve_fresh_output_root`` refuses drive-absolute/UNC
    roots and ``..`` traversal. The capture lands in the SAME physical directory as
    ``--output-dir``; when only a relative spelling can satisfy the frozen core,
    that spelling is used (and flagged in the report). When no safe spelling
    exists, the runner refuses with an actionable message.
    """
    core = core or _load_core()
    base_cwd = Path(cwd if cwd is not None else os.getcwd())

    def _try(raw: str) -> str | None:
        try:
            core.resolve_fresh_output_root(raw)
        except Exception:
            return None
        return raw

    as_given = str(output_dir)
    if _try(as_given) is not None:
        return as_given, False

    # A drive-absolute / UNC / file:// root is only meaningfully relative-izable on
    # a native Windows host (where ``os.path`` is ``ntpath``). On any other platform
    # such a string is NOT the drive root the frozen core refuses, so accepting a
    # derived spelling would be a silent reinterpretation: refuse instead.
    if (os.name != "nt") and (
        _is_drive_absolute(as_given) or as_given.strip().startswith("\\\\")
        or _FILE_URI_RE.match(as_given.strip())
    ):
        raise OutputSafetyError(
            "the output dir looks like a Windows drive/UNC/file:// root, which this "
            "platform cannot use as a capture root; refusing (this is not a Windows host)."
        )

    try:
        rel = os.path.relpath(os.fspath(output_dir), os.fspath(base_cwd))
    except ValueError as exc:  # e.g. different Windows drives
        raise OutputSafetyError(
            "cannot express the output dir as a relative path from the current "
            "working directory (are they on different drives?); refusing. Run the "
            "runner from an ancestor of the output path."
        ) from exc
    if _has_traversal(rel):
        raise OutputSafetyError(
            f"the output dir is not under the current working directory ({_basename(as_given)} "
            "needs parent traversal); the frozen capture core refuses traversal roots. "
            "cd to an ancestor of the output path and retry."
        )
    if _try(rel) is None:
        raise OutputSafetyError(
            "the frozen capture core refuses this output location; choose a fresh output "
            "path reachable without traversal from the current working directory."
        )
    return rel, True


# ---------------------------------------------------------------------------
# Planned invocation
# ---------------------------------------------------------------------------

def assert_cell_gauge_mode(mode: Any) -> str:
    """Require the ONLY seam that carries the R40 capture."""
    if not isinstance(mode, str):
        raise RunnerError(f"cell gauge mode must be a string, got {type(mode).__name__}")
    if mode != CELL_GAUGE_MODE_PER_CELL:
        raise RunnerError(
            f"the R40 capture exists ONLY at the per_cell seam; cell_gauge_mode="
            f"{mode!r} is refused (required: {CELL_GAUGE_MODE_PER_CELL!r})"
        )
    return mode


def assert_holdout_cap(value: Any) -> int:
    """The R40 gate is the UNCAPPED EXACT held-out selection; a cap is refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerError(f"--holdout-cap must be an int, got {type(value).__name__}")
    if int(value) != UNCAPPED_HOLDOUT_CAP:
        raise RunnerError(
            "--holdout-cap "
            f"{int(value)} would request a CAPPED held-out selection; the R40 gate "
            "requires the UNCAPPED exact selection per relation. Refusing (never "
            "degrade a run into a capped capture)."
        )
    return int(value)


def assert_disk_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerError(
            f"--disk-budget-bytes must be an int, got {type(value).__name__}"
        )
    if int(value) <= 0:
        raise RunnerError(f"--disk-budget-bytes must be positive, got {int(value)}")
    return int(value)


def resolve_invocation(
    *,
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    cell_gauge_mode: str,
    holdout_cap: int,
    disk_budget_bytes: int,
    core: Any | None = None,
    cwd: str | Path | None = None,
) -> dict:
    """Build the exact ``run_zegrid_mode`` kwargs (validated, default-OFF seam)."""
    assert_cell_gauge_mode(cell_gauge_mode)
    assert_holdout_cap(holdout_cap)
    budget = assert_disk_budget(disk_budget_bytes)
    capture_root, relativized = resolve_capture_root(output_dir, core=core, cwd=cwd)
    kwargs = {
        "cell_gauge_mode": CELL_GAUGE_MODE_PER_CELL,
        "r40_seam_capture": True,
        "r40_output_root": capture_root,
        "r40_seam_capture_disk_budget_bytes": budget,
    }
    assert_kwargs_supported(kwargs)
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "capture_root": capture_root,
        "capture_root_relativized": bool(relativized),
        "kwargs": kwargs,
    }


# ---------------------------------------------------------------------------
# Published-generation discovery + independent verification
# ---------------------------------------------------------------------------

def find_published_generation(capture_root: str | os.PathLike[str]) -> Path:
    """Return the ONE committed R40 generation dir under the capture root."""
    root = Path(capture_root).expanduser()
    if not root.is_dir():
        raise VerificationError(f"capture root does not exist: {root.name!r}")
    committed: list[Path] = []
    incomplete: list[str] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not child.name.startswith(GENERATION_DIR_PREFIX):
            continue
        if (child / GENERATION_COMMIT_BASENAME).is_file():
            committed.append(child)
        else:
            incomplete.append(child.name)
    if not committed:
        detail = f" (uncommitted candidates: {incomplete})" if incomplete else ""
        raise VerificationError(
            f"no committed R40 capture generation under {root.name!r}{detail}"
        )
    if len(committed) > 1:
        raise VerificationError(
            f"expected exactly ONE committed generation, found {len(committed)}: "
            + ", ".join(p.name for p in committed)
        )
    return committed[0]


def verify_published_generation(gen_dir: str | os.PathLike[str], *, core: Any | None = None) -> dict:
    """Independently verify a published generation (strict production-wired)."""
    core = core or _load_core()
    try:
        report = core.verify_generation(
            Path(gen_dir), expected_production_wired=True
        )
    except Exception as exc:  # noqa: BLE001 - structured refusal, no traceback
        raise VerificationError(
            f"verify_generation refused ({type(exc).__name__}): {exc}"
        ) from exc
    if report.get("ok") is not True or list(report.get("problems") or ()):
        raise VerificationError(
            "post-run generation verification FAILED: "
            + "; ".join(str(p) for p in (report.get("problems") or ("ok is not True",)))
        )
    return report


def read_generation_index(gen_dir: str | os.PathLike[str]) -> dict:
    path = Path(gen_dir) / GENERATION_INDEX_BASENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"cannot read the generation index: {exc}") from exc


# ---------------------------------------------------------------------------
# Provenance report
# ---------------------------------------------------------------------------

def build_report(
    *,
    invocation: Mapping[str, Any],
    command: Sequence[str],
    gen_dir: Path,
    index: Mapping[str, Any],
    verification: Mapping[str, Any],
    repo_head: str,
    tool_sha: Mapping[str, str],
    frozen: Mapping[str, str],
    input_inventory_fits_count: int,
) -> dict:
    """Strict-JSON provenance report (basenames only; no absolute private paths)."""
    identity = index.get("identity") or {}
    index_path = Path(gen_dir) / GENERATION_INDEX_BASENAME
    index_sha, index_bytes = _sha256_file(index_path)
    commit_path = Path(gen_dir) / GENERATION_COMMIT_BASENAME
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "mission": MISSION,
        "capture_mission": CAPTURE_MISSION,
        "phase": "implementation",
        "status": "VERIFIED",
        "repo_head": str(repo_head),
        "tool_source_sha256": {
            "runner": str(tool_sha["runner"]),
            "zemosaic_zegrid_mode": str(tool_sha["zemosaic_zegrid_mode"]),
        },
        "frozen_bindings": {
            "b1_protocol_sha256": str(frozen["b1_protocol_sha256"]),
            "b1_criteria_sha256": str(frozen["b1_criteria_sha256"]),
            "b1_result_sha256": str(frozen["b1_result_sha256"]),
            "b1_claim_id": str(frozen["b1_claim_id"]),
        },
        "input": {
            "basename": _basename(invocation["input_dir"]),
            "n_frames": int(identity.get("n_frames") or 0),
            "source_set_digest": str(identity.get("source_set_digest") or ""),
            "source_set_digest_algo": str(identity.get("source_set_digest_algo") or ""),
            "source_set_count": int(identity.get("source_set_count") or 0),
            "input_inventory_fits_count": int(input_inventory_fits_count),
        },
        "output": {
            "basename": _basename(invocation["output_dir"]),
            "capture_root_relativized": bool(invocation["capture_root_relativized"]),
        },
        "capture": {
            "capture_contract_version": str(identity.get("capture_contract_version") or ""),
            "capture_stage": str(identity.get("capture_stage") or ""),
            "capture_id": str(identity.get("capture_id") or ""),
            "generation_dirname": Path(gen_dir).name,
            "generation_index_basename": GENERATION_INDEX_BASENAME,
            "generation_sha256": index_sha,
            "generation_bytes": index_bytes,
            "n_shards": int(index.get("n_shards") or 0),
            "n_records": int(index.get("n_records") or 0),
            "shard_bytes": int(index.get("shard_bytes") or 0),
            "commit_sha256": _sha256_text(
                commit_path.read_text(encoding="utf-8")
            ),
            "index_sha256": str(commit.get("index_sha256") or ""),
        },
        "qualification": {
            "holdout_capped": bool(identity.get("holdout_capped")),
            "qualification": str(identity.get("qualification") or ""),
            "n_dropped": int(identity.get("n_dropped") or 0),
        },
        "verification": {
            "expected_production_wired": True,
            "ok": bool(verification.get("ok")),
            "problems": list(verification.get("problems") or ()),
            "checked_records": int(verification.get("checked_records") or 0),
            "checked_bytes": int(verification.get("checked_bytes") or 0),
        },
        "generation_verified": True,
        "command": [str(t) for t in command],
        "default_off_elsewhere": True,
        "dev_not_qualification": True,
    }
    return report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Strict-JSON atomic write (allow_nan=False, no temp residue)."""
    import tempfile

    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".r40c_", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: no cover - defensive
            pass
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _console_progress(*args, **kwargs) -> None:
    parts = [str(a) for a in args if a is not None]
    if parts:
        print("[r40-runner] " + " ".join(parts), flush=True)


def _print_block(title: str, payload: Mapping[str, Any]) -> None:
    print(f"--- {title} ---", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r40_capture_runner",
        description=(
            "Bounded, dry-run-by-default runner for the R40 DEV diagnostic capture "
            "seam (ZM-ZEGRID-R40-C). Calls run_zegrid_mode directly with the "
            "explicit, default-OFF r40_seam_capture seam."
        ),
    )
    parser.add_argument("--input-dir", required=True,
                        help="source dataset folder (read-only; never opened by this tool)")
    parser.add_argument("--output-dir", required=True,
                        help="NEW/empty destination for the capture run (never the input/baseline)")
    parser.add_argument("--disk-budget-bytes", type=int, default=CORE_DEFAULT_DISK_BUDGET_BYTES,
                        help=(f"hard capture disk budget in bytes (positive int; "
                              f"default {CORE_DEFAULT_DISK_BUDGET_BYTES})"))
    parser.add_argument("--holdout-cap", type=int, default=UNCAPPED_HOLDOUT_CAP,
                        help=("UNCAPPED sentinel only (0/omitted). The R40 gate requires the "
                              "uncapped EXACT held-out selection; any positive cap is refused."))
    parser.add_argument("--cell-gauge-mode", default=CELL_GAUGE_MODE_PER_CELL,
                        help=("the effective cell gauge mode; must be 'per_cell' "
                              "(the only seam carrying the R40 capture)"))
    parser.add_argument("--baseline-dir", action="append", default=[], metavar="DIR",
                        help=("optional read-only root the output must not overlap "
                              "(repeatable; e.g. an existing baseline run dir)"))
    parser.add_argument("--execute", action="store_true",
                        help="perform the real capture (required for any write/full run)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_argv = list(argv if argv is not None else sys.argv[1:])

    try:
        core = _load_core()
        frozen = assert_frozen_bindings(core)
        zz = _load_zegrid()
    except Exception as exc:  # noqa: BLE001
        print(f"[r40-runner] BINDING ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return _EXIT_ERROR

    repo_head = _git_head(_repo_root())
    if not repo_head or not re.match(r"^[0-9a-f]{40}$", str(repo_head)):
        print("[r40-runner] cannot resolve a 40-hex repo HEAD; refusing to capture",
              file=sys.stderr, flush=True)
        return _EXIT_ERROR

    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        print(f"[r40-runner] INPUT ERROR: input dir is not a directory: {input_dir.name!r}",
              file=sys.stderr, flush=True)
        return _EXIT_ERROR
    inventory = scan_fits(input_dir)
    if not inventory:
        print("[r40-runner] INPUT ERROR: no FITS frames found under the input dir",
              file=sys.stderr, flush=True)
        return _EXIT_ERROR

    try:
        out_dir = validate_output_safety(
            args.output_dir, input_dir=args.input_dir, protected_roots=args.baseline_dir,
        )
    except OutputSafetyError as exc:
        print(f"[r40-runner] UNSAFE OUTPUT: {exc}", file=sys.stderr, flush=True)
        return _EXIT_UNSAFE_OUTPUT

    try:
        invocation = resolve_invocation(
            input_dir=str(input_dir), output_dir=str(out_dir),
            cell_gauge_mode=args.cell_gauge_mode, holdout_cap=args.holdout_cap,
            disk_budget_bytes=args.disk_budget_bytes, core=core,
        )
    except OutputSafetyError as exc:
        print(f"[r40-runner] UNSAFE OUTPUT: {exc}", file=sys.stderr, flush=True)
        return _EXIT_UNSAFE_OUTPUT
    except RunnerError as exc:
        print(f"[r40-runner] CONFIG ERROR: {exc}", file=sys.stderr, flush=True)
        return _EXIT_ERROR

    plan = {
        "repo_head": repo_head,
        "input_dir_basename": _basename(str(input_dir)),
        "output_dir_basename": _basename(str(out_dir)),
        "capture_root_relativized": invocation["capture_root_relativized"],
        "n_source_fits_in_inventory": len(inventory),
        "frozen_bindings": frozen,
        "run_zegrid_mode_kwargs": invocation["kwargs"],
        "execute": bool(args.execute),
    }
    print(
        "[r40-runner] R40-C DEV diagnostic capture — "
        f"HEAD {repo_head[:12]}, inventory {len(inventory)} FITS, "
        f"output {_basename(str(out_dir))!r}",
        flush=True,
    )
    _print_block("EXACT PLAN (run_zegrid_mode kwargs)", plan)

    if not args.execute:
        print(
            "[r40-runner] DRY-RUN complete: NOTHING written. Re-run with --execute "
            "to perform the capture.",
            flush=True,
        )
        return _EXIT_OK

    # --- execute -----------------------------------------------------------
    print(f"[r40-runner] EXECUTE -> {out_dir}", flush=True)
    try:
        result = zz.run_zegrid_mode(
            str(input_dir), str(out_dir),
            progress_callback=_console_progress, **invocation["kwargs"],
        )
        del result  # run_zegrid_mode returns None; kept explicit for clarity
    except Exception as exc:  # noqa: BLE001 - surface the real failure verbatim
        print(f"[r40-runner] CAPTURE RUN FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return _EXIT_ERROR

    print("[r40-runner] run finished — locating + verifying the published generation…",
          flush=True)
    try:
        gen_dir = find_published_generation(invocation["capture_root"])
        verification = verify_published_generation(gen_dir, core=core)
        index = read_generation_index(gen_dir)
    except VerificationError as exc:
        print(f"[r40-runner] VERIFICATION FAILED: {exc}", file=sys.stderr, flush=True)
        return _EXIT_VERIFY_FAIL

    identity_head = str((index.get("identity") or {}).get("repo_head") or "")
    if identity_head != repo_head:
        print(
            "[r40-runner] VERIFICATION FAILED: the published generation's repo_head "
            f"{identity_head[:12]!r} != the observed HEAD {repo_head[:12]!r}",
            file=sys.stderr, flush=True,
        )
        return _EXIT_VERIFY_FAIL

    runner_path = Path(__file__).resolve()
    zz_path = Path(zz.__file__).resolve()
    tool_sha = {
        "runner": _sha256_file(runner_path)[0],
        "zemosaic_zegrid_mode": _sha256_file(zz_path)[0],
    }
    command = [Path(sys.executable).name or "python", _TOOL_BASENAME] + scrub_command(raw_argv)
    report = build_report(
        invocation=invocation, command=command, gen_dir=gen_dir, index=index,
        verification=verification, repo_head=repo_head, tool_sha=tool_sha,
        frozen=frozen, input_inventory_fits_count=len(inventory),
    )
    try:
        core.assert_no_path_leaks(report)
        core.assert_finite_json(report, "$.report")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[r40-runner] VERIFICATION FAILED: provenance report refused by the "
            f"frozen privacy/finiteness boundary ({type(exc).__name__}): {exc}",
            file=sys.stderr, flush=True,
        )
        return _EXIT_VERIFY_FAIL

    try:
        _atomic_write_json(Path(out_dir) / REPORT_NAME, report)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[r40-runner] FAILED to write verification report {REPORT_NAME}: {exc} "
            "— aborting (nonzero).",
            file=sys.stderr, flush=True,
        )
        return _EXIT_ERROR

    cap = report["capture"]
    qual = report["qualification"]
    print(
        f"[r40-runner] VERIFICATION PASS — capture_id={cap['capture_id']} "
        f"shards={cap['n_shards']} records={cap['n_records']} "
        f"sha256={cap['generation_sha256'][:16]}… "
        f"qualification={qual['qualification']} capped={qual['holdout_capped']} "
        f"n_dropped={qual['n_dropped']}",
        flush=True,
    )
    print(f"[r40-runner] wrote {REPORT_NAME}", flush=True)
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
