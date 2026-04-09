from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cupy_available_line(path: Path) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("CUPY_AVAILABLE ="):
            return line
    raise AssertionError(f"CUPY_AVAILABLE assignment not found in {path}")


def test_worker_cupy_detection_is_not_windows_only():
    line = _cupy_available_line(REPO_ROOT / "zemosaic_worker.py")
    assert 'find_spec("cupy") is not None' in line
    assert "IS_WINDOWS" not in line


def test_telemetry_cupy_detection_is_not_windows_only():
    line = _cupy_available_line(REPO_ROOT / "zemosaic_resource_telemetry.py")
    assert 'find_spec("cupy") is not None' in line
    assert "IS_WINDOWS" not in line
