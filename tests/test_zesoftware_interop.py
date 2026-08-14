"""Anti-drift + wheel-inclusion tests for ZeMosaic consumer interop metadata.

The ``zemosaic/zesoftware_interop.json`` file is *deployment metadata* for the
ZeAlfie pre-activation compatibility evaluator.  It is not product behavior, so
these tests pin it to the authoritative ZeMosaic source-of-truth constants in
:mod:`zemosaic.zesolver_adapter` and :mod:`zemosaic.solver_port` so the JSON can
never silently drift from the real consumer contract.

Independence is preserved: nothing here imports ZeAlfie or a real ZeSolver, and
the wheel test reads the JSON straight out of the wheel zip without importing
any product code from the built artifact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.solver_port import (  # noqa: E402
    OPTIONAL_ZESOLVER_CAPABILITIES,
    REQUIRED_ZESOLVER_CAPABILITIES,
    ZESOLVER_SOLVE_BACKEND_CAPABILITIES,
)
from zemosaic.zesolver_adapter import (  # noqa: E402
    _ZESOLVER_API_MODULE,
    _ZESOLVER_SUPPORTED_API_MAJOR,
)

INTEROP_PATH = SRC / "zemosaic" / "zesoftware_interop.json"


def _load_interop() -> dict:
    return json.loads(INTEROP_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Top-level identity and shape
# ---------------------------------------------------------------------------


def test_interop_file_exists_and_parses():
    data = _load_interop()
    assert data["schema"] == "zesoftware.interop.v1"
    assert data["product_id"] == "zemosaic"
    assert data["distribution_name"] == "ZeMosaic"


def test_no_invented_provider_contract():
    # ZeMosaic does not currently publish any public inter-project API, so it
    # must not declare any ``provides`` entry.
    assert _load_interop()["provides"] == []


def test_single_consumer_entry_for_zesolver():
    consumes = _load_interop()["consumes"]
    assert isinstance(consumes, list)
    assert len(consumes) == 1
    entry = consumes[0]
    assert entry["provider_product_id"] == "zesolver"
    assert entry["provider_distribution_name"] == "ZeSolver"


# ---------------------------------------------------------------------------
# Anti-drift: JSON vs. ZeMosaic source-of-truth constants
# ---------------------------------------------------------------------------


def test_api_module_matches_adapter_constant():
    entry = _load_interop()["consumes"][0]
    assert entry["api_module"] == _ZESOLVER_API_MODULE


def test_api_version_specifier_matches_supported_major():
    entry = _load_interop()["consumes"][0]
    major = _ZESOLVER_SUPPORTED_API_MAJOR
    assert major == 1
    # ``_ZESOLVER_SUPPORTED_API_MAJOR == 1`` means "API major 1 only",
    # expressed declaratively as ``>=1,<2``.
    assert entry["api_version"] == f">={major},<{major + 1}"


def test_required_capabilities_match_solver_port():
    entry = _load_interop()["consumes"][0]
    assert entry["required_capabilities"] == list(REQUIRED_ZESOLVER_CAPABILITIES)
    assert REQUIRED_ZESOLVER_CAPABILITIES == ("wcs_write",)


def test_solve_backend_any_of_group_matches_and_is_required():
    entry = _load_interop()["consumes"][0]
    groups = entry["any_of_capabilities"]
    assert isinstance(groups, list)
    assert len(groups) == 1
    group = groups[0]
    assert group["id"] == "solve_backend"
    assert group["capabilities"] == list(ZESOLVER_SOLVE_BACKEND_CAPABILITIES)
    assert ZESOLVER_SOLVE_BACKEND_CAPABILITIES == ("near_solve", "blind_solve")
    assert group["required"] is True


def test_optional_capabilities_match_solver_port():
    entry = _load_interop()["consumes"][0]
    assert entry["optional_capabilities"] == list(OPTIONAL_ZESOLVER_CAPABILITIES)
    assert OPTIONAL_ZESOLVER_CAPABILITIES == ("cancel", "gpu")


def test_zesolver_remains_optional():
    entry = _load_interop()["consumes"][0]
    assert entry["optional"] is True


# ---------------------------------------------------------------------------
# Wheel inclusion: the metadata must ship in built wheels, inspectable without
# importing product code from the wheel.
# ---------------------------------------------------------------------------


def test_built_wheel_contains_exactly_one_interop_json(tmp_path):
    try:
        import build  # noqa: F401
    except ImportError:  # pragma: no cover - dev-only dependency
        pytest.skip("`build` is not installed; cannot build a wheel")

    outdir = tmp_path / "dist"
    outdir.mkdir()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(outdir),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, (
        f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
    )

    wheels = list(outdir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    interop_entries = [n for n in names if n.endswith("zesoftware_interop.json")]
    assert interop_entries == ["zemosaic/zesoftware_interop.json"], interop_entries

    # Parse the JSON straight from the zip: no import of product code.
    with zipfile.ZipFile(wheel) as zf:
        raw = zf.read("zemosaic/zesoftware_interop.json")
    data = json.loads(raw.decode("utf-8"))
    assert data["schema"] == "zesoftware.interop.v1"
    assert data["product_id"] == "zemosaic"
    assert data["distribution_name"] == "ZeMosaic"
