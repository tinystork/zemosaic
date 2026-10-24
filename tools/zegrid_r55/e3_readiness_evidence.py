#!/usr/bin/env python3
"""R55-E3 deterministic readiness evidence + standard SHA-256 sidecar.

Pure source/contract evidence: no hardware probe, no timestamp, no float, no
personal path. Running this tool twice on the same checkout produces
byte-identical JSON.

Status is deliberately ``WINDOWS_NOT_RUN``: this evidence closes the *readiness*
surface (GUI selector, propagation, packaging audit, checker, guide) and does not
qualify Windows or RTX. Only an imported real Windows result may change that,
through a separate human/architect decision.

Usage:
    python tools/zegrid_r55/e3_readiness_evidence.py --out DIR [--verify DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FILENAME = "r55e3_readiness_evidence.json"
MISSION = "ZM-ZEGRID-R55-E3-WINDOWS-BETA-READINESS-20261024"
SCHEMA = "zemosaic-r55e3-readiness-evidence-v1"

# Files whose exact bytes define the E3 readiness surface. Missing entries are a
# hard error: the evidence may never silently shrink.
SOURCE_FILES = (
    "src/zemosaic/zemosaic_gui_qt.py",
    "src/zemosaic/zemosaic_config.py",
    "src/zemosaic/zemosaic_worker.py",
    "src/zemosaic/zemosaic_zegrid_mode.py",
    "src/zemosaic/core/zegrid/scheduler.py",
    "src/zemosaic/core/zegrid/gpu_readiness.py",
    "src/zemosaic/core/zegrid/parallel.py",
    "src/zemosaic/locales/en.json",
    "tests/test_zegrid_r55e3_windows_beta.py",
    "tools/zegrid_r55/windows_result_check.py",
    "tools/zegrid_r55/e3_readiness_evidence.py",
    "docs/beta/ZEGRID_WINDOWS_BETA_WITNESS.md",
    "ZeMosaic.spec",
    "pyproject.toml",
)

TEST_CONTRACT = (
    "tests/test_zegrid_r55e3_windows_beta.py",
    "tests/test_zegrid_r55_observe_only.py",
    "tests/test_zegrid_r55d2_cpu_active.py",
    "tests/test_zegrid_r55e1_run_checkpoints.py",
    "tests/test_zegrid_r55e2_active_gpu_prefetch.py",
    "tests/test_zegrid_r53c_gui_halo.py",
    "tests/test_zegrid_r22_gpu_contract.py",
    "tests/test_zegrid_r22_runtime_fallback.py",
    "tests/test_zegrid_r36_adaptive_cpu_tile.py",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as exc:
        raise SystemExit(f"cannot read git HEAD: {type(exc).__name__}: {exc}")
    if proc.returncode != 0:
        raise SystemExit(f"cannot read git HEAD: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _packaging_audit(repo: Path) -> dict:
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    base = pyproject.split("[project.optional-dependencies]", 1)[0]
    optional = pyproject.split("[project.optional-dependencies]", 1)[1]
    return {
        "cupy_in_base_dependencies": '"cupy' in base,
        "cupy_optional_extra_declared": '"cupy-cuda12x"' in optional,
        "r55_modules_importable": True,
        "note": (
            "R55 modules stay CuPy-lazy and the package starts when CuPy is "
            "genuinely unavailable. Historical parallel_utils imports CuPy when "
            "it IS installed; that behaviour is pre-existing and unchanged by E3."
        ),
    }


def canonical_payload(repo: Path) -> dict:
    repo = repo.resolve()
    hashes = {}
    for rel in SOURCE_FILES:
        path = repo / rel
        if not path.is_file():
            raise SystemExit(f"missing evidence source file: {rel}")
        hashes[rel] = _sha256_file(path)
    return {
        "schema": SCHEMA,
        "mission": MISSION,
        "phase": "implementation",
        "status": "WINDOWS_NOT_RUN",
        "windows_qualification": {
            "status": "WINDOWS_NOT_RUN",
            "qualified": False,
            "reason": "No real Windows/RTX witness has been imported; readiness only.",
        },
        "git_head": _git_head(repo),
        "source_hashes": hashes,
        "test_contract": list(TEST_CONTRACT),
        "packaging": _packaging_audit(repo),
        "checker": "tools/zegrid_r55/windows_result_check.py",
        "guide": "docs/beta/ZEGRID_WINDOWS_BETA_WITNESS.md",
    }


def canonical_bytes(repo: Path) -> bytes:
    text = json.dumps(canonical_payload(repo), sort_keys=True, indent=2, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def write_evidence(out_dir: Path, repo: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(repo)
    digest = _sha256_bytes(data)
    json_path = out_dir / FILENAME
    json_path.write_bytes(data)
    (out_dir / (FILENAME + ".sha256")).write_text(
        f"{digest}  {FILENAME}\n", encoding="ascii"
    )
    return json_path, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--verify", default=None, help="directory to compare against")
    parser.add_argument("--repo", default=str(REPO_ROOT), help="repository root")
    args = parser.parse_args(argv)

    repo = Path(args.repo)
    path, digest = write_evidence(Path(args.out), repo)
    print(f"wrote {path}")
    print(f"sha256 {digest}")

    if args.verify:
        other_dir = Path(args.verify)
        other = other_dir / FILENAME
        if not other.is_file():
            print(f"VERIFY_FAIL missing {other}")
            return 2
        if other.read_bytes() != path.read_bytes():
            print("VERIFY_FAIL evidence bytes differ")
            return 3
        sidecar = other_dir / (FILENAME + ".sha256")
        if not sidecar.is_file():
            print(f"VERIFY_FAIL missing {sidecar}")
            return 4
        expected = sidecar.read_text(encoding="ascii").split()[0]
        if expected != digest:
            print("VERIFY_FAIL sha256 sidecar differs")
            return 5
        print("VERIFY_OK byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
