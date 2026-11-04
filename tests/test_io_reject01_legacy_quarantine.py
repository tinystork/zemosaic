"""ZM-IO-REJECT-01 — legacy quarantine contract + no-serial-replay witnesses.

All files are synthetic copies under ``tmp_path``. The real Windows corpus and
any production run are strictly out of scope.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from zemosaic import input_inventory as zin
from zemosaic import io_quarantine as ziq
from zemosaic import zemosaic_stack_plan as zsp
from zemosaic import zemosaic_utils as zu
from zemosaic.core.zegrid import parallel as zpar


# ---------------------------------------------------------------------------
# A. Legacy destination contract
# ---------------------------------------------------------------------------


def _write(path: Path, payload: bytes = b"frame") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_exact_legacy_destination_and_inventory_exclusion(tmp_path):
    root = tmp_path / "input"
    src = _write(root / "M45" / "light.fit")

    result = ziq.move_to_unaligned(src, root)

    expected = root / "unaligned_by_zemosaic" / "light.fit"
    assert result.status == ziq.STATUS_MOVED
    assert Path(result.destination) == expected
    assert expected.is_file()
    assert not src.exists()
    # The historical scanner exclusion still hides the quarantined source.
    assert zin.scan_input_fits(root) == []


def test_existing_destination_is_never_overwritten(tmp_path):
    root = tmp_path / "input"
    src = _write(root / "light.fit", b"new-source")
    existing = _write(root / "unaligned_by_zemosaic" / "light.fit", b"old-dest")

    result = ziq.move_to_unaligned(src, root)

    assert result.status == ziq.STATUS_MOVED_RENAMED
    assert Path(result.destination) == root / "unaligned_by_zemosaic" / "light__1.fit"
    assert existing.read_bytes() == b"old-dest"
    assert (root / "unaligned_by_zemosaic" / "light__1.fit").read_bytes() == b"new-source"
    assert not src.exists()


def test_concurrent_calls_move_exactly_once(tmp_path):
    root = tmp_path / "input"
    src = _write(root / "light.fit")
    results = []
    lock = threading.Lock()

    def _worker():
        res = ziq.move_to_unaligned(src, root)
        with lock:
            results.append(res)

    threads = [threading.Thread(target=_worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    quarantined = sorted((root / "unaligned_by_zemosaic").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name == "light.fit"
    assert not src.exists()
    assert all(r.status in {
        ziq.STATUS_MOVED,
        ziq.STATUS_ALREADY_MOVED,
        ziq.STATUS_MISSING,
    } for r in results)


def test_source_already_in_quarantine_is_skipped(tmp_path):
    root = tmp_path / "input"
    src = _write(root / "unaligned_by_zemosaic" / "old.fit")
    result = ziq.move_to_unaligned(src, root)
    assert result.status == ziq.STATUS_SKIPPED_EXCLUDED
    assert src.exists()


def test_nested_source_quarantines_next_to_itself(tmp_path):
    """Legacy contract: nested frame -> its OWN sibling unaligned folder."""
    root = tmp_path / "input"
    src = _write(root / "M45" / "bad.fit")

    result = ziq.move_to_unaligned(src, src.parent)

    assert result.status == ziq.STATUS_MOVED
    assert Path(result.destination) == (
        root / "M45" / "unaligned_by_zemosaic" / "bad.fit"
    )
    assert not (root / "unaligned_by_zemosaic").exists()
    assert zin.scan_input_fits(root) == []


def test_exdev_fallback_preserves_content_and_removes_source(tmp_path, monkeypatch):
    import errno

    root = tmp_path / "input"
    src = _write(root / "light.fit", b"exdev-content")

    def _no_link(_src, _dst):
        raise OSError(errno.EXDEV, "simulated cross-device link")

    monkeypatch.setattr(ziq.os, "link", _no_link)

    result = ziq.move_to_unaligned(src, root)

    assert result.status == ziq.STATUS_MOVED
    assert Path(result.destination).read_bytes() == b"exdev-content"
    assert not src.exists()
    assert [p.name for p in (root / "unaligned_by_zemosaic").iterdir()] == [
        "light.fit"
    ]


def test_exdev_copy_failure_cleans_reservation_and_keeps_source(
    tmp_path, monkeypatch
):
    import errno

    root = tmp_path / "input"
    src = _write(root / "light.fit", b"must-survive")

    def _no_link(_src, _dst):
        raise OSError(errno.EXDEV, "simulated cross-device link")

    def _fail_move(_src, _dst):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(ziq.os, "link", _no_link)
    monkeypatch.setattr(ziq.shutil, "move", _fail_move)

    result = ziq.move_to_unaligned(src, root)

    assert result.status == ziq.STATUS_FAILED
    assert src.read_bytes() == b"must-survive"
    assert list((root / "unaligned_by_zemosaic").iterdir()) == []


def test_cross_process_same_basename_never_overwrites(tmp_path):
    """Two real processes race for the same destination basename."""
    root = tmp_path / "input"
    src_a = _write(root / "a" / "light.fit", b"AAA")
    src_b = _write(root / "b" / "light.fit", b"BBB")
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "from zemosaic.io_quarantine import move_to_unaligned\n"
        "result = move_to_unaligned(sys.argv[1], sys.argv[2])\n"
        "print(result.status)\n"
    )
    env = dict(os.environ)
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for path in (src_a, src_b)
    ]
    for proc in procs:
        _out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, err[-2000:]

    files = list((root / "unaligned_by_zemosaic").iterdir())
    assert len(files) == 2
    assert {p.read_bytes() for p in files} == {b"AAA", b"BBB"}
    assert not src_a.exists() and not src_b.exists()


def test_quarantine_result_is_picklable(tmp_path):
    root = tmp_path / "input"
    src = _write(root / "light.fit")
    result = ziq.move_to_unaligned(src, root)
    clone = pickle.loads(pickle.dumps(result))
    assert clone == result
    assert clone.status == ziq.STATUS_MOVED
    assert clone.destination == result.destination


def _run_worker_wrapper(source: Path, owner_root: Path) -> tuple[str, str]:
    """Run the legacy worker wrapper in a CLEAN subprocess.

    Importing ``zemosaic_worker`` in the pytest process has a pre-existing
    logging side effect (``ZeMosaicWorker.propagate=False``) that can silence an
    unrelated caplog test. Isolating the witness keeps the combined suite
    order-independent without changing production logging.
    """
    repo_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "from zemosaic import zemosaic_worker as zworker\n"
        "status, destination = zworker._move_to_unaligned_safe(\n"
        "    sys.argv[1], sys.argv[2]\n"
        ")\n"
        "print(status)\n"
        "print(destination)\n"
    )
    env = dict(os.environ)
    src = str(repo_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [sys.executable, "-c", code, str(source), str(owner_root)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.strip().splitlines()
    assert len(lines) >= 2, proc.stdout
    return lines[0], lines[1]


def test_worker_legacy_wrapper_keeps_tuple_contract(tmp_path):
    root = tmp_path / "input"
    src = _write(root / "light.fit")
    status, destination = _run_worker_wrapper(src, root)
    assert status == "moved"
    assert Path(destination) == root / "unaligned_by_zemosaic" / "light.fit"


def test_worker_legacy_wrapper_maps_collision_to_moved(tmp_path):
    """A renamed collision is still the historical ``moved`` status."""
    root = tmp_path / "input"
    src = _write(root / "light.fit", b"new-source")
    existing = _write(root / "unaligned_by_zemosaic" / "light.fit", b"old-dest")

    status, destination = _run_worker_wrapper(src, root)

    assert status == "moved"
    expected = root / "unaligned_by_zemosaic" / "light__1.fit"
    assert Path(destination) == expected
    assert expected.read_bytes() == b"new-source"
    assert existing.read_bytes() == b"old-dest"
    assert not src.exists()


# ---------------------------------------------------------------------------
# B. Descriptor setup/WCS rejects are quarantined
# ---------------------------------------------------------------------------


class _FrameInfo:
    def __init__(self, path, wcs, shape_hw=(10, 10)):
        self.path = Path(path)
        self.wcs = wcs
        self.shape_hw = shape_hw


def _tan_wcs():
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [-0.001, 0.001]
    return w


def _linear_wcs():
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["LINEAR", "LINEAR"]
    return w


def test_build_frame_descriptors_quarantines_wcs_reject(tmp_path, monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    good = _write(root / "good.fit")
    bad = _write(root / "M45" / "bad.fit")
    frames = [
        _FrameInfo(good, _tan_wcs()),
        _FrameInfo(bad, _linear_wcs()),
    ]

    descs, rejected = zz._build_frame_descriptors(frames, root, None)

    assert [Path(d.source_path) for d in descs] == [good]
    assert good.exists()
    assert len(rejected) == 1
    entry = rejected[0]
    assert entry["path"] == str(bad)
    assert entry["quarantine_status"] == "moved"
    assert Path(entry["quarantine_destination"]) == (
        root / "M45" / "unaligned_by_zemosaic" / "bad.fit"
    )
    assert not bad.exists()
    assert (root / "M45" / "unaligned_by_zemosaic" / "bad.fit").is_file()
    # A future recursive inventory never re-ingests the rejected source.
    assert [Path(p) for p in zin.scan_input_fits(root)] == [good]


def _assert_source_intact_and_never_quarantined(src: Path) -> None:
    assert src.exists()
    assert not (src.parent / ziq.UNALIGNED_DIRNAME).exists()


def test_setup_memory_error_never_moves_source(tmp_path, monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write(root / "M45" / "frame.fit")

    def _oom(_path):
        raise MemoryError("simulated allocator failure")

    monkeypatch.setattr(zsp, "_open_fits_safely", _oom)
    with pytest.raises(MemoryError, match="allocator failure"):
        zz._build_frame_descriptors([zsp.FrameInfo(path=src)], root, None)
    _assert_source_intact_and_never_quarantined(src)


def test_setup_programming_error_never_moves_source(tmp_path, monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write(root / "frame.fit")

    def _bug(_path):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(zsp, "_open_fits_safely", _bug)
    with pytest.raises(zsp.FrameSetupRefusal) as info:
        zz._build_frame_descriptors([zsp.FrameInfo(path=src)], root, None)
    assert info.value.kind == zsp.WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL
    _assert_source_intact_and_never_quarantined(src)


def test_setup_permission_error_never_moves_source(tmp_path, monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write(root / "frame.fit")

    def _denied(_path):
        raise PermissionError("access denied")

    monkeypatch.setattr(zsp, "_open_fits_safely", _denied)
    with pytest.raises(zsp.FrameSetupRefusal) as info:
        zz._build_frame_descriptors([zsp.FrameInfo(path=src)], root, None)
    assert info.value.kind == zsp.WCS_FAILURE_MISSING_OR_ACCESS
    _assert_source_intact_and_never_quarantined(src)


def test_setup_missing_file_never_moves_other_source(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    missing = root / "gone.fit"
    other = _write(root / "other.fit")
    with pytest.raises(zsp.FrameSetupRefusal) as info:
        zz._build_frame_descriptors([zsp.FrameInfo(path=missing)], root, None)
    assert info.value.kind == zsp.WCS_FAILURE_MISSING_OR_ACCESS
    assert not missing.exists()
    _assert_source_intact_and_never_quarantined(other)


def test_setup_cancelled_never_moves_source(tmp_path, monkeypatch):
    import concurrent.futures

    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write(root / "frame.fit")

    def _cancelled(_path):
        raise concurrent.futures.CancelledError("cooperative cancel")

    monkeypatch.setattr(zsp, "_open_fits_safely", _cancelled)
    with pytest.raises(concurrent.futures.CancelledError):
        zz._build_frame_descriptors([zsp.FrameInfo(path=src)], root, None)
    _assert_source_intact_and_never_quarantined(src)


def test_real_corrupt_fits_is_terminal_and_moved(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    bad = _write(root / "M45" / "corrupt.fit", b"this is not a FITS file")
    with pytest.raises(RuntimeError):
        # It is a terminal source reject; with no frame left the builder raises.
        zz._build_frame_descriptors([zsp.FrameInfo(path=bad)], root, None)
    destination = root / "M45" / ziq.UNALIGNED_DIRNAME / "corrupt.fit"
    assert destination.is_file()
    assert not bad.exists()


def test_qualify_wcs_exception_is_internal_and_never_moves(tmp_path, monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import geometry as zg

    root = tmp_path / "input"
    src = _write(root / "frame.fit")

    def _boom(_wcs):
        raise RuntimeError("qualify bug")

    monkeypatch.setattr(zg, "qualify_wcs", _boom)
    with pytest.raises(zsp.FrameSetupRefusal) as info:
        zz._build_frame_descriptors([_FrameInfo(src, _tan_wcs())], root, None)
    assert info.value.kind == zsp.WCS_FAILURE_INFRASTRUCTURE_OR_INTERNAL
    _assert_source_intact_and_never_quarantined(src)


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("oom"),
        RuntimeError("bug"),
        zpar.ParallelPoolError("pool failed"),
    ],
)
def test_payload_boundary_ignores_non_payload_errors(tmp_path, exc):
    import concurrent.futures

    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write(root / "frame.fit")
    result = zz._quarantine_payload_failure(exc, root, None, phase="gauge")
    assert result is None
    _assert_source_intact_and_never_quarantined(src)


def test_payload_boundary_ignores_cancelled_error(tmp_path):
    import concurrent.futures

    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write(root / "frame.fit")
    result = zz._quarantine_payload_failure(
        concurrent.futures.CancelledError(), root, None, phase="cell cache"
    )
    assert result is None
    _assert_source_intact_and_never_quarantined(src)


# ---------------------------------------------------------------------------
# R4: no active WCS helper may absorb MemoryError/cancellation
# ---------------------------------------------------------------------------


def test_extract_pixel_scale_primary_memory_error_propagates(monkeypatch):
    def _oom(_wcs):
        raise MemoryError("primary pixel-scale OOM")

    monkeypatch.setattr(zsp, "proj_plane_pixel_scales", _oom)
    with pytest.raises(MemoryError, match="primary pixel-scale OOM"):
        zsp._extract_pixel_scale_deg(object())


def test_extract_pixel_scale_fallback_memory_error_propagates(monkeypatch):
    class _CdeltOOM:
        @property
        def cdelt(self):
            raise MemoryError("fallback cdelt OOM")

    class _WcsWithCdelt:
        wcs = _CdeltOOM()

    def _primary_fails(_wcs):
        raise ValueError("primary scale unavailable")

    monkeypatch.setattr(zsp, "proj_plane_pixel_scales", _primary_fails)
    with pytest.raises(MemoryError, match="fallback cdelt OOM"):
        zsp._extract_pixel_scale_deg(_WcsWithCdelt())


def test_header_int_memory_error_propagates():
    class _HeaderOOM:
        def get(self, _key):
            raise MemoryError("header OOM")

    with pytest.raises(MemoryError, match="header OOM"):
        zsp._header_int(_HeaderOOM(), "NAXIS")


def test_active_helper_cancellation_propagates():
    import concurrent.futures

    class _HeaderCancelled:
        def get(self, _key):
            raise concurrent.futures.CancelledError("header cancelled")

    with pytest.raises(concurrent.futures.CancelledError):
        zsp._header_int(_HeaderCancelled(), "NAXIS")
    with pytest.raises(concurrent.futures.CancelledError):
        zsp._admit_rgb_chw_cube(_HeaderCancelled(), (3, 10, 10))


def test_mount_header_memory_error_propagates():
    class _HeaderOOM:
        def get(self, _key):
            raise MemoryError("mount header OOM")

    with pytest.raises(MemoryError, match="mount header OOM"):
        zin.mount_from_header(_HeaderOOM())


def test_mount_normalize_memory_error_propagates():
    class _BadStr:
        def __str__(self):
            raise MemoryError("mount normalization OOM")

    with pytest.raises(MemoryError, match="mount normalization OOM"):
        zin.normalize_mount(_BadStr())


def _write_minimal_tan_fits(path: Path) -> Path:
    from astropy.io import fits

    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.zeros((16, 16), dtype=np.float32))
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    hdu.header["CRVAL1"] = 10.0
    hdu.header["CRVAL2"] = 20.0
    hdu.header["CRPIX1"] = 8.0
    hdu.header["CRPIX2"] = 8.0
    hdu.header["CDELT1"] = -0.001
    hdu.header["CDELT2"] = 0.001
    hdu.writeto(path, overwrite=True)
    return path


def test_setup_memory_error_from_pixel_scale_never_moves(tmp_path, monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write_minimal_tan_fits(root / "M45" / "frame.fit")

    def _oom(_wcs):
        raise MemoryError("pixel-scale OOM on active path")

    monkeypatch.setattr(zsp, "_extract_pixel_scale_deg", _oom)
    with pytest.raises(MemoryError, match="pixel-scale OOM"):
        zz._build_frame_descriptors([zsp.FrameInfo(path=src)], root, None)
    _assert_source_intact_and_never_quarantined(src)


def test_setup_value_error_from_pixel_scale_stays_terminal_and_moves(
    tmp_path, monkeypatch
):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    src = _write_minimal_tan_fits(root / "M45" / "frame.fit")

    def _bad_scale(_wcs):
        raise ValueError("invalid header scale")

    monkeypatch.setattr(zsp, "_extract_pixel_scale_deg", _bad_scale)
    with pytest.raises(RuntimeError):
        zz._build_frame_descriptors([zsp.FrameInfo(path=src)], root, None)
    assert (root / "M45" / ziq.UNALIGNED_DIRNAME / "frame.fit").is_file()
    assert not src.exists()


# ---------------------------------------------------------------------------
# C. Loader: None return activates raw fallback; both failures stay structured
# ---------------------------------------------------------------------------


class _HDU:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.header = {}


class _FakeHDUL:
    def __init__(self, data):
        self._hdu = _HDU(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getitem__(self, key):
        if key == 0:
            return self._hdu
        raise KeyError(key)

    def __contains__(self, key):
        return False


@contextmanager
def _opened(data):
    yield _FakeHDUL(data)


def test_loader_none_from_primary_activates_raw_fallback(monkeypatch):
    calls = {"raw": 0}

    # ``load_and_validate_fits`` ABSORBS its errors and returns data=None.
    monkeypatch.setattr(
        zu, "load_and_validate_fits", lambda *a, **k: (None, {}, {})
    )

    def _raw(path):
        calls["raw"] += 1
        return _opened(np.ones((4, 4), dtype=np.float32))

    monkeypatch.setattr(zu, "_open_fits_safely", _raw)
    arr, weights = zu.load_image_with_optional_alpha(Path("synthetic.fit"))
    assert calls["raw"] == 1
    assert arr.shape == (4, 4, 1)
    assert weights is None


def test_loader_primary_none_and_raw_failure_preserves_both_causes(monkeypatch):
    monkeypatch.setattr(
        zu, "load_and_validate_fits", lambda *a, **k: (None, {}, {})
    )

    def _raw(path):
        raise OSError("CRC mismatch on read")

    monkeypatch.setattr(zu, "_open_fits_safely", _raw)

    with pytest.raises(ziq.FramePayloadError) as info:
        zu.load_image_with_optional_alpha(Path("broken.fit"))
    exc = info.value
    assert exc.path.endswith("broken.fit")
    assert "returned no image data" in exc.primary_error
    assert "CRC mismatch on read" in exc.raw_error
    assert "CRC mismatch on read" in str(exc)


def test_loader_raw_fallback_success_when_primary_raises(monkeypatch):
    def _primary(*a, **k):
        raise RuntimeError("primary exploded")

    monkeypatch.setattr(zu, "load_and_validate_fits", _primary)
    monkeypatch.setattr(
        zu, "_open_fits_safely", lambda path: _opened(np.ones((3, 3), dtype=np.float32))
    )

    arr, weights = zu.load_image_with_optional_alpha(Path("recovered.fit"))
    assert arr.shape == (3, 3, 1)
    assert weights is None


def test_loader_valid_payload_is_unchanged(monkeypatch):
    expected = np.arange(36, dtype=np.float32).reshape(6, 6)
    monkeypatch.setattr(
        zu, "load_and_validate_fits", lambda *a, **k: (expected, {}, {})
    )

    def _raw(path):  # must never run
        raise AssertionError("raw fallback must not be used for a valid payload")

    monkeypatch.setattr(zu, "_open_fits_safely", _raw)
    arr, weights = zu.load_image_with_optional_alpha(Path("valid.fit"))
    assert np.array_equal(arr[..., 0], expected)
    assert weights is None


# ---------------------------------------------------------------------------
# D. pmap: worker failure propagates, no whole-wave serial replay
# ---------------------------------------------------------------------------


def _pmap_fail_worker(task):
    value, counter_path = task
    with open(counter_path, "a", encoding="utf-8") as fh:
        fh.write(f"{value}\n")
    if int(value) == 3:
        raise RuntimeError("unreadable frame payload")
    return int(value) * 2


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_pmap_process_worker_failure_propagates_without_replay(
    tmp_path, caplog
):
    counter = tmp_path / "calls.txt"
    tasks = [(i, str(counter)) for i in range(6)]
    meta = {}

    with caplog.at_level(logging.WARNING, logger=zpar.log.name):
        with pytest.raises(zpar.ParallelTaskError) as info:
            zpar.pmap(_pmap_fail_worker, tasks, workers=2, meta=meta)

    assert info.value.index == 3
    assert isinstance(info.value.original, RuntimeError)
    assert "unreadable frame payload" in str(info.value)
    assert meta["fallback"] is False
    assert "parallel map unavailable" not in caplog.text
    assert "falling back to SERIAL" not in caplog.text
    calls = _lines(counter)
    # No task was re-executed (a serial replay would add every task a 2nd time).
    assert len(calls) == len(set(calls))
    assert all(calls.count(str(i)) <= 1 for i in range(6))


def test_pmap_thread_path_worker_failure_propagates_without_replay(monkeypatch):
    calls = []

    def _worker(value):
        calls.append(value)
        if value == 2:
            raise RuntimeError("thread payload failure")
        return value

    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    meta = {}
    with pytest.raises(zpar.ParallelTaskError) as info:
        zpar.pmap(_worker, [0, 1, 2, 3], workers=2, meta=meta)

    assert info.value.index == 2
    assert "thread payload failure" in str(info.value)
    assert meta["fallback"] is False
    assert calls.count(2) == 1
    assert all(calls.count(v) <= 1 for v in calls)


def _pmap_payload_worker(task):
    path = str(task)
    raise ziq.FramePayloadError(
        "Failed to load frame: primary and raw failed",
        path=path,
        primary_error="RuntimeError: primary",
        raw_error="OSError: CRC",
    )


def test_pmap_first_completed_cancels_pending_and_drains_running(
    monkeypatch, caplog
):
    """FIRST_COMPLETED detects the error, PENDING tasks are cancelled, and
    already-RUNNING tasks are drained before the exception propagates."""
    release = threading.Event()
    running_one = threading.Event()
    started = []
    started_lock = threading.Lock()

    def _worker(value):
        with started_lock:
            started.append(value)
        if value == 0:
            # Wait until task 1 is genuinely running, then fail. This guarantees
            # the drain has a real already-started task to wait for.
            running_one.wait(2.0)
            raise RuntimeError("early payload failure")
        if value == 1:
            running_one.set()
            release.wait(10.0)
        elif value == 2:
            release.wait(10.0)
        return value

    # Independent bounded releaser: the drain must NOT depend on a finally
    # executed only after pmap returns.
    timer = threading.Timer(0.5, release.set)
    timer.daemon = True
    timer.start()

    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    meta = {}
    started_at = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=zpar.log.name):
        with pytest.raises(zpar.ParallelTaskError) as info:
            zpar.pmap(_worker, [0, 1, 2, 3], workers=2, meta=meta)
    elapsed = time.monotonic() - started_at

    assert info.value.index == 0
    assert "early payload failure" in str(info.value)
    # The error was observed early, but propagation waited for the running
    # blocked task(s) to be released and drained.
    assert 0.4 <= elapsed < 5.0, (
        f"expected a bounded drain around 0.5s, observed {elapsed:.2f}s"
    )
    assert meta["fallback"] is False
    # Task 3 was PENDING when the failure was observed and must never start.
    assert 3 not in started
    # No task ran twice (no serial replay).
    with started_lock:
        snapshot = list(started)
    assert snapshot.count(0) == 1
    assert all(snapshot.count(v) <= 1 for v in snapshot)
    # After propagation the executor is quiescent: no new side effect appears.
    time.sleep(0.2)
    with started_lock:
        assert started == snapshot
    assert "parallel map unavailable" not in caplog.text
    assert "falling back to SERIAL" not in caplog.text


def test_pmap_preserves_structured_payload_error_across_process(tmp_path):
    tasks = [tmp_path / "bad_a.fit", tmp_path / "bad_b.fit"]
    with pytest.raises(zpar.ParallelTaskError) as info:
        zpar.pmap(_pmap_payload_worker, tasks, workers=2)
    original = info.value.original
    assert isinstance(original, ziq.FramePayloadError)
    assert original.path == str(tasks[info.value.index])
    assert original.primary_error == "RuntimeError: primary"
    assert original.raw_error == "OSError: CRC"


def test_pmap_partial_submit_failure_drains_executor(monkeypatch):
    import concurrent.futures

    real_submit = concurrent.futures.ThreadPoolExecutor.submit
    real_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown
    submit_calls = []
    shutdowns = []

    def _flaky_submit(self, fn, *args, **kwargs):
        submit_calls.append(1)
        if len(submit_calls) == 2:
            raise RuntimeError("simulated partial submit failure")
        return real_submit(self, fn, *args, **kwargs)

    def _record_shutdown(self, wait=True, cancel_futures=False):
        shutdowns.append((wait, cancel_futures))
        return real_shutdown(
            self, wait=wait, cancel_futures=cancel_futures
        )

    monkeypatch.setattr(
        concurrent.futures.ThreadPoolExecutor, "submit", _flaky_submit
    )
    monkeypatch.setattr(
        concurrent.futures.ThreadPoolExecutor, "shutdown", _record_shutdown
    )
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)

    executed = []

    def _worker(value):
        executed.append(value)
        return value

    meta = {}
    with pytest.raises(zpar.ParallelPoolError, match="submit failed"):
        zpar.pmap(_worker, [1, 2, 3], workers=2, meta=meta)

    assert shutdowns == [(True, True)]
    assert meta["fallback"] is False
    assert all(executed.count(v) <= 1 for v in executed)


def test_pmap_broken_pool_is_not_a_task_error(monkeypatch):
    import concurrent.futures

    shutdowns = []

    class _BrokenPool:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn, task):
            future = concurrent.futures.Future()
            future.set_exception(
                concurrent.futures.process.BrokenProcessPool("simulated broken pool")
            )
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            shutdowns.append((wait, cancel_futures))

    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", _BrokenPool
    )
    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)

    meta = {}
    with pytest.raises(zpar.ParallelPoolError) as info:
        zpar.pmap(lambda value: value, [1, 2], workers=2, meta=meta)

    assert not isinstance(info.value, zpar.ParallelTaskError)
    assert "BrokenProcessPool" in str(info.value)
    assert shutdowns == [(True, True)]
    assert meta["fallback"] is False


def test_pmap_heartbeat_failure_drains_without_replay(monkeypatch):
    release = threading.Event()
    running_one = threading.Event()
    executed = []
    executed_lock = threading.Lock()
    heartbeat_calls = []

    def _worker(value):
        with executed_lock:
            executed.append(value)
        if value == 0:
            running_one.wait(2.0)
            raise RuntimeError("heartbeat-path failure")
        if value == 1:
            running_one.set()
            release.wait(10.0)
        return value

    def _heartbeat():
        heartbeat_calls.append(time.monotonic())

    timer = threading.Timer(0.4, release.set)
    timer.daemon = True
    timer.start()

    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: True)
    meta = {}
    started_at = time.monotonic()
    with pytest.raises(zpar.ParallelTaskError) as info:
        zpar.pmap(
            _worker,
            [0, 1, 2],
            workers=2,
            heartbeat=_heartbeat,
            heartbeat_interval_s=0.02,
            meta=meta,
        )
    elapsed = time.monotonic() - started_at

    assert info.value.index == 0
    assert heartbeat_calls
    assert meta["heartbeat_calls"] >= 1
    assert meta["fallback"] is False
    assert elapsed >= 0.3
    with executed_lock:
        snapshot = list(executed)
    time.sleep(0.2)
    with executed_lock:
        assert executed == snapshot
    assert all(snapshot.count(v) <= 1 for v in snapshot)


def test_pmap_executor_creation_failure_keeps_bounded_serial_fallback(
    monkeypatch, caplog
):
    import concurrent.futures

    class _BoomExecutor:
        def __init__(self, *a, **k):
            raise OSError("simulated spawn failure")

    def _square(value):
        return value * value

    monkeypatch.setattr(zpar, "_parent_is_daemonic", lambda: False)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _BoomExecutor)

    with caplog.at_level(logging.WARNING, logger=zpar.log.name):
        out = zpar.pmap(_square, [2, 3, 4, 5], workers=4)

    assert out == [4, 9, 16, 25]
    assert "parallel map unavailable" in caplog.text


# ---------------------------------------------------------------------------
# E. Parent boundary: move + terminal refusal, no other source touched
# ---------------------------------------------------------------------------


def test_boundary_quarantines_source_and_refuses_run(tmp_path):
    from zemosaic import zemosaic_zegrid_mode as zz

    root = tmp_path / "input"
    bad = _write(root / "M45" / "broken.fit")
    untouched = _write(root / "M45" / "fine.fit")
    messages = []
    payload = ziq.FramePayloadError(
        "Failed to load frame: primary and raw failed",
        path=str(bad),
        primary_error="RuntimeError: primary",
        raw_error="OSError: CRC",
    )
    wrapped = zpar.ParallelTaskError(3, "task", payload)

    def _emit(message, _item=None, lvl="INFO", **kwargs):
        messages.append((message, lvl))

    with pytest.raises(ziq.FramePayloadQuarantineError) as info:
        zz._quarantine_payload_failure(
            wrapped, root, _emit, phase="gauge"
        )

    exc = info.value
    assert exc.path == str(bad)
    assert exc.move_status == "moved"
    assert Path(exc.destination) == (
        root / "M45" / "unaligned_by_zemosaic" / "broken.fit"
    )
    assert not bad.exists()
    assert untouched.exists()
    assert [p.name for p in (root / "M45" / "unaligned_by_zemosaic").iterdir()] == [
        "broken.fit"
    ]
    text = str(exc)
    assert "new run" in text.lower()
    assert "replan" in text.lower()
    assert any(lvl == "ERROR" and "new run" in msg.lower() for msg, lvl in messages)


def test_run_boundary_refuses_before_science_and_manifest(tmp_path, monkeypatch):
    """A gauge decode failure moves the source and stops before any science."""
    import subprocess
    import sys

    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import photometric as zphot

    repo_root = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "zegrid_routine" / "make_routine_corpus.py"),
            str(corpus),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    inp = tmp_path / "in"
    inp.mkdir()
    for src in sorted(corpus.glob("*.fits")):
        (inp / src.name).write_bytes(src.read_bytes())
    out = tmp_path / "out"
    failing = sorted(inp.glob("*.fits"))[0]

    payload = ziq.FramePayloadError(
        "Failed to load frame: primary and raw failed",
        path=str(failing),
        primary_error="RuntimeError: primary",
        raw_error="OSError: CRC",
    )

    def _boom(*args, **kwargs):
        raise zpar.ParallelTaskError(0, "gauge-task", payload)

    monkeypatch.setattr(zphot, "compute_global_gauge", _boom)

    with pytest.raises(ziq.FramePayloadQuarantineError):
        zz.run_zegrid_mode(str(inp), str(out))

    assert not (out / "mosaic_grid.fits").exists()
    assert not (out / "zegrid_manifest.json").exists()
    assert (inp / "unaligned_by_zemosaic" / failing.name).is_file()
    assert not failing.exists()
