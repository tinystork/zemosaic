"""M0 REWORK-1 — fail-closed Windows harness tests (no Windows, no GPU).

These tests exercise the harness's pure gates and predicates. They do not run
Windows, do not use a GPU, and do not claim a real GPU result. The local
control witness remains in ``test_zegrid_resume_m0_baseline_witness.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "tools" / "zegrid_resume_m0" / "m0_windows_witness.py"

spec = importlib.util.spec_from_file_location("m0_windows_witness", HARNESS_PATH)
assert spec and spec.loader
m0 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m0
spec.loader.exec_module(m0)


def _args(tmp_path: Path, *, installed: Path | None, installed_src: Path | None = None):
    return SimpleNamespace(
        src_root=str(tmp_path / "src"),
        installed_module=str(installed) if installed else None,
        installed_src_root=str(installed_src) if installed_src else None,
    )


def _write_module(root: Path, body: str) -> Path:
    pkg = root / "zemosaic"
    pkg.mkdir(parents=True, exist_ok=True)
    path = pkg / "zemosaic_zegrid_mode.py"
    path.write_text(body, encoding="utf-8")
    return path


def _witness(**over):
    base = {
        "schema": m0.SCHEMA,
        "status": "PASS",
        "classification": "EXPECTED_BASELINE_FAILURE",
        "resources": {"cap_exceeded": False},
        "gpu_qualified": True,
    }
    base.update(over)
    return base


def _valid_failure(tmp_path: Path):
    return {
        "reason_code": "R55_REFUSE_HOST_RAM",
        "exception": {"type": "_R55GpuFallback"},
        "pipeline": {
            "units_consumed": 2,
            "cancelled": True,
            "cleanup_failures": [],
            "ledger": {
                "balanced": True,
                "host_leases": 0,
                "device_leases": 0,
                "disk_leases": 0,
            },
        },
        "tile_state": {
            "units": [
                {"status": "completed", "discarded": True},
                {"status": "completed", "discarded": True},
                {"status": "refused", "action": "fallback_cpu", "discarded": True},
            ]
        },
    }


class AdmissionExhausted(RuntimeError):
    pass


def _valid_sequence():
    return [
        {"unit_index": 0, "cell_id": "r0000c0000", "backend_used": "gpu",
         "core_sha256": "a" * 64},
        {"unit_index": 1, "cell_id": "r0000c0001", "backend_used": "gpu",
         "core_sha256": "b" * 64},
    ]


# ---------------------------------------------------------------------------
# F2 — parent fail-closed
# ---------------------------------------------------------------------------
def test_parent_rejects_nonzero_child_exit_even_with_pass_witness(tmp_path):
    verdict = m0.evaluate_child_result(
        returncode=7,
        timed_out=False,
        witness=_witness(),
        new_bytes=100,
        max_new_bytes=1000,
        stage_record={"stage": "done"},
    )
    assert verdict["status"] == "CHILD_EXIT_NONZERO"
    assert verdict["exit"] != 0


def test_parent_rejects_wrong_classification_even_with_zero_exit(tmp_path):
    verdict = m0.evaluate_child_result(
        returncode=0,
        timed_out=False,
        witness=_witness(classification="UNEXPECTED_SUCCESS"),
        new_bytes=100,
        max_new_bytes=1000,
        stage_record={"stage": "run"},
    )
    assert verdict["status"] == "FAIL"
    assert verdict["exit"] != 0


def test_parent_rejects_cap_exceeded_never_pass():
    verdict = m0.evaluate_child_result(
        returncode=0,
        timed_out=False,
        witness=_witness(),
        new_bytes=2000,
        max_new_bytes=1000,
        stage_record={"stage": "done"},
    )
    assert verdict["status"] == "CAP_EXCEEDED"
    assert verdict["exit"] == 4


def test_parent_timeout_is_not_pass():
    verdict = m0.evaluate_child_result(
        returncode=None,
        timed_out=True,
        witness=_witness(),
        new_bytes=100,
        max_new_bytes=1000,
        stage_record={"stage": "gpu_units"},
    )
    assert verdict["status"] == "TIMEOUT"
    assert verdict["exit"] == 3


def test_parent_not_run_is_coherent_only_with_exit_two():
    verdict = m0.evaluate_child_result(
        returncode=2,
        timed_out=False,
        witness={"status": "NOT_RUN", "reason": "no gpu"},
        new_bytes=100,
        max_new_bytes=1000,
        stage_record={"stage": "gpu_readiness"},
    )
    assert verdict["status"] == "NOT_RUN"
    assert verdict["exit"] == 2
    # A mismatched exit code is never trusted, even with a NOT_RUN witness.
    mismatch = m0.evaluate_child_result(
        returncode=7,
        timed_out=False,
        witness={"status": "NOT_RUN", "reason": "no gpu"},
        new_bytes=100,
        max_new_bytes=1000,
        stage_record={"stage": "gpu_readiness"},
    )
    assert mismatch["status"] == "CHILD_EXIT_NONZERO"
    assert mismatch["exit"] != 0


def test_parent_accepts_only_full_coherent_pass():
    verdict = m0.evaluate_child_result(
        returncode=0,
        timed_out=False,
        witness=_witness(),
        new_bytes=100,
        max_new_bytes=1000,
        stage_record={"stage": "done"},
    )
    assert verdict == {
        "status": "PASS",
        "exit": 0,
        "reason": "returncode==0 and PASS/EXPECTED_BASELINE_FAILURE",
        "stage_reached": "done",
    }


# ---------------------------------------------------------------------------
# F1 — provenance equality, not presence
# ---------------------------------------------------------------------------
def test_provenance_mismatch_refuses_by_default(tmp_path):
    src = _write_module(tmp_path / "src", "SOURCE = 1\n")
    installed = tmp_path / "installed.py"
    installed.write_text("DIFFERENT = 2\n", encoding="utf-8")
    prov = m0._provenance(_args(tmp_path, installed=installed))
    assert prov["main_module_match"] is False
    assert prov["provenance_verified"] is False
    with pytest.raises(SystemExit):
        m0.provenance_gate(prov, allow_unverified=False)


def test_provenance_override_is_labelled_non_qualifying(tmp_path):
    _write_module(tmp_path / "src", "SOURCE = 1\n")
    installed = tmp_path / "installed.py"
    installed.write_text("DIFFERENT = 2\n", encoding="utf-8")
    prov = m0._provenance(_args(tmp_path, installed=installed))
    gate = m0.provenance_gate(prov, allow_unverified=True)
    assert gate["allowed"] is True
    assert gate["provenance_verified"] is False
    assert gate["qualification"] == "NON_QUALIFYING_CONTROL"
    assert gate["override_used"] is True


def test_provenance_match_is_verified(tmp_path):
    src = _write_module(tmp_path / "src", "SOURCE = 1\n")
    installed = tmp_path / "installed.py"
    installed.write_bytes(src.read_bytes())
    prov = m0._provenance(_args(tmp_path, installed=installed))
    assert prov["main_module_match"] is True
    assert prov["source_actual_match"] is True
    assert prov["provenance_verified"] is True
    gate = m0.provenance_gate(prov, allow_unverified=False)
    assert gate["provenance_verified"] is True
    assert gate["qualification"] == "VERIFIED_INSTALLED_RUNTIME"


def test_provenance_uses_actual_imported_module_file(tmp_path):
    src = _write_module(tmp_path / "src", "SOURCE = 1\n")
    actual = tmp_path / "actual_imported.py"
    actual.write_text("ACTUAL = 3\n", encoding="utf-8")
    installed = tmp_path / "installed.py"
    installed.write_bytes(actual.read_bytes())
    fake_module = SimpleNamespace(__file__=str(actual))
    prov = m0._provenance(
        _args(tmp_path, installed=installed), imported_module=fake_module
    )
    assert prov["actual_imported_module"] == str(actual.resolve())
    # The actual imported file differs from the source path -> not verified.
    assert prov["main_module_match"] is True
    assert prov["source_actual_match"] is False
    assert prov["provenance_verified"] is False


# ---------------------------------------------------------------------------
# F3 — full witness predicate
# ---------------------------------------------------------------------------
def test_witness_checks_accept_complete_evidence(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    checks = m0._witness_checks(
        exc=AdmissionExhausted("aggregate 1 > B_admit 0"),
        failure=_valid_failure(tmp_path),
        sequence=_valid_sequence(),
        pressure_after=2,
        provenance_verified=True,
        out_dir=out,
    )
    assert all(checks.values()), checks


def test_witness_checks_reject_missing_discard(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    failure = _valid_failure(tmp_path)
    failure["tile_state"]["units"][0]["discarded"] = False
    checks = m0._witness_checks(
        exc=AdmissionExhausted("aggregate 1 > B_admit 0"),
        failure=failure,
        sequence=_valid_sequence(),
        pressure_after=2,
        provenance_verified=True,
        out_dir=out,
    )
    assert checks["completed_units_discarded"] is False
    assert not all(checks.values())


def test_witness_checks_reject_publication_and_cleanup_and_leases(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "mosaic_grid.fits").write_bytes(b"x")
    failure = _valid_failure(tmp_path)
    failure["pipeline"]["cleanup_failures"] = ["leftover"]
    failure["pipeline"]["ledger"]["host_leases"] = 1
    checks = m0._witness_checks(
        exc=AdmissionExhausted("aggregate 1 > B_admit 0"),
        failure=failure,
        sequence=_valid_sequence(),
        pressure_after=2,
        provenance_verified=True,
        out_dir=out,
    )
    assert checks["zero_publication"] is False
    assert checks["cleanup_failures_empty"] is False
    assert checks["all_leases_zero"] is False


def test_witness_checks_require_provenance(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    checks = m0._witness_checks(
        exc=AdmissionExhausted("aggregate 1 > B_admit 0"),
        failure=_valid_failure(tmp_path),
        sequence=_valid_sequence(),
        pressure_after=2,
        provenance_verified=False,
        out_dir=out,
    )
    assert checks["provenance_verified"] is False


def test_gpu_qualified_only_for_real_windows_gpu_proof():
    assert m0.gpu_qualified_for(
        mode="WINDOWS_REAL_GPU",
        gpu_available=True,
        expected=True,
        sequence=_valid_sequence(),
        provenance_verified=True,
    ) is True
    for kwargs in (
        {"mode": "LOCAL_CONTROL_CPU", "gpu_available": True, "expected": True},
        {"mode": "WINDOWS_REAL_GPU", "gpu_available": False, "expected": True},
        {"mode": "WINDOWS_REAL_GPU", "gpu_available": True, "expected": False},
        {"mode": "WINDOWS_REAL_GPU", "gpu_available": True, "expected": True,
         "sequence": [{"backend_used": "cpu"}], "provenance_verified": True},
        {"mode": "WINDOWS_REAL_GPU", "gpu_available": True, "expected": True,
         "sequence": _valid_sequence(), "provenance_verified": False},
    ):
        base = {
            "mode": kwargs["mode"],
            "gpu_available": kwargs["gpu_available"],
            "expected": kwargs["expected"],
            "sequence": kwargs.get("sequence", _valid_sequence()),
            "provenance_verified": kwargs.get("provenance_verified", True),
        }
        assert m0.gpu_qualified_for(**base) is False, kwargs


# ---------------------------------------------------------------------------
# F4/F5 — estimate and coherent cap/status
# ---------------------------------------------------------------------------
def test_preflight_estimate_uses_float32_and_outputs():
    estimate = m0.estimate_new_bytes()
    frame_bytes = m0.CORPUS_FRAMES * m0.CORPUS_SIZE * m0.CORPUS_SIZE * 4
    assert estimate >= frame_bytes
    assert estimate > 64 * 1024 ** 2


def test_evaluate_child_cap_exceeded_is_coherent():
    verdict = m0.evaluate_child_result(
        returncode=0,
        timed_out=False,
        witness=_witness(resources={"cap_exceeded": True}),
        new_bytes=10,
        max_new_bytes=1000,
        stage_record={"stage": "done"},
    )
    assert verdict["status"] == "CAP_EXCEEDED"
    assert verdict["exit"] == 4


# ---------------------------------------------------------------------------
# F2/F1/F6 — parent end-to-end with a mocked child process
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self, *, exit_code=0, timeout=False, witness=None, cwd=None):
        self.pid = 424242
        self.returncode = None
        self._final = int(exit_code)
        self._timeout = bool(timeout)
        self._timed_once = False
        self._killed = False
        self._witness = witness
        if witness is not None and cwd is not None:
            Path(cwd, "witness.json").write_text(
                __import__("json").dumps(witness), encoding="utf-8"
            )

    def poll(self):
        if self._killed:
            self.returncode = self._final
        return self.returncode

    def wait(self, timeout=None):
        if self._timeout and not self._timed_once:
            self._timed_once = True
            import subprocess
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self.returncode = self._final
        return self.returncode

    def kill(self):
        self._killed = True
        self.returncode = self._final


def _parent_args(tmp_path, *, installed, extra=()):
    return m0._parse_args(
        [
            "--src-root", str(tmp_path / "src"),
            "--installed-module", str(installed),
            "--output-root", str(tmp_path / "out"),
            "--allow-outside-zgprobe",
            "--max-seconds", "10",
            *extra,
        ]
    )


def test_parent_end_to_end_rejects_exit7_with_pass_witness(tmp_path, monkeypatch):
    installed = _write_module(tmp_path / "src", "SOURCE = 1\n")
    args = _parent_args(tmp_path, installed=installed)
    monkeypatch.setattr(
        m0.subprocess,
        "Popen",
        lambda *a, **kw: _FakeProc(exit_code=7, witness=_witness(), cwd=kw.get("cwd")),
    )
    monkeypatch.setattr(m0, "_kill_tree", lambda proc: None)
    code = m0._parent(args)
    summary = __import__("json").loads(
        (tmp_path / "out" / "run_summary.json").read_text()
    )
    assert code == 1
    assert summary["status"] == "CHILD_EXIT_NONZERO"
    assert summary["status"] != "PASS"


def test_parent_end_to_end_refuses_provenance_mismatch(tmp_path):
    src = _write_module(tmp_path / "src", "SOURCE = 1\n")
    installed = tmp_path / "installed.py"
    installed.write_text("DIFFERENT = 2\n", encoding="utf-8")
    args = _parent_args(tmp_path, installed=installed)
    with pytest.raises(SystemExit):
        m0._parent(args)


def test_parent_end_to_end_timeout_is_never_pass(tmp_path, monkeypatch):
    installed = _write_module(tmp_path / "src", "SOURCE = 1\n")
    args = _parent_args(tmp_path, installed=installed)
    monkeypatch.setattr(
        m0.subprocess,
        "Popen",
        lambda *a, **kw: _FakeProc(exit_code=0, timeout=True, cwd=kw.get("cwd")),
    )
    monkeypatch.setattr(m0, "_kill_tree", lambda proc: setattr(proc, "_killed", True))
    code = m0._parent(args)
    summary = __import__("json").loads(
        (tmp_path / "out" / "run_summary.json").read_text()
    )
    assert code == 3
    assert summary["status"] == "TIMEOUT"
    assert summary["status"] != "PASS"
