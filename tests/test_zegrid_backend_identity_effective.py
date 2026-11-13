"""ZM-ZEGRID-BACKEND-IDENTITY — identity uses the EFFECTIVE backend.

The defect reproduced on real data: a CPU-frozen durable store resumed on a
host where GPU stacking is enabled was refused at identity with

    science: stored={sha256='475e9eae…'} current={sha256='ec7407b2…'}

even though the product KNOWS to ignore a GPU on a CPU-frozen run (the GUI logs
"When: GPU ignored — CPU-frozen run"). The only differing field was the
``backend`` inside the science parameters: ``cpu`` in the stored contract,
``gpu`` in the current process (the requested one). The normalisation happened
AFTER the identity check.

These tests pin the fix WITHOUT a corpus or a GPU:

* the effective backend is ONE decision (``_effective_run_backend``), consumed
  by the plan gate (``_frozen_planning_gpu``) and by the identity;
* a CPU-frozen identity is byte-identical whether the host requested no GPU or
  requested a GPU (the same science digest, numerically pinned below);
* a GPU-frozen identity keeps refreezing ``gpu`` (the frozen backend governs);
* every genuine science change (normalization/weighting/rejection/combine/
  kappa/winsor/taper/cell_gauge_mode/reference) is STILL refused with the
  historical message, and a real backend switch is still refused.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402
from zemosaic.core.zegrid import run_store as zrs  # noqa: E402

MIB = 2**20
CELL_IDS = ("r0000c0000", "r0000c0001", "r0001c0000", "r0001c0001")

# Numerically pinned digests of the fixture below. They are the CONTRACT: the
# same store resumed with and without a GPU request must produce this exact
# science fingerprint and this exact identity.
PINNED_CPU_SCIENCE_SHA256 = (
    "796675a2a8af9917b1da78dfd612cbc3fcac777324d827dc3fc9d5ffc6ffd341"
)
PINNED_CPU_IDENTITY_SHA256 = (
    "f70dadaac0752237d4d84143569caf3107e6bd53baad1aab0236b0630dd6fe31"
)
PINNED_GPU_SCIENCE_SHA256 = (
    "349d327bd1d1281351aa8f31be4c6cd71931c335dde0a8b5f0806c4b87c3a149"
)
PINNED_GPU_IDENTITY_SHA256 = (
    "c5d99092a26012ac7a7981657027c8e41d417cb13bc3a02afd96a32488b0c084"
)
# Pre-fix behaviour (the defect): the SAME CPU run when the science params were
# digested with the REQUESTED gpu backend.
PINNED_REQUESTED_GPU_ON_CPU_SCIENCE_SHA256 = (
    "349d327bd1d1281351aa8f31be4c6cd71931c335dde0a8b5f0806c4b87c3a149"
)


# ---------------------------------------------------------------------------
# Fixture: a real-shaped run-scope bundle + deterministic inputs
# ---------------------------------------------------------------------------


class _FrameId:
    def __init__(self, logical_path):
        self.logical_path = logical_path


class _Desc:
    def __init__(self, logical_path, source_path):
        self.frame_id = _FrameId(logical_path)
        self.source_path = source_path


class _Bounds:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


class _Cell:
    def __init__(self, cell_id, bounds):
        self.cell_id = cell_id
        self.core = bounds


class _Membership:
    def __init__(self, cell_id, core_ids, patch_ids):
        self.cell_id = cell_id
        self.core_ids = tuple(core_ids)
        self.patch_ids = tuple(patch_ids)


def _scope_bundle(*, cpu_tile=96, gpu_tile=None, parity=False):
    cells, tasks, ctxs = [], [], []
    for index, cid in enumerate(CELL_IDS):
        col, row = index % 2, index // 2
        bounds = _Bounds(col * 40, row * 32, (col + 1) * 40, (row + 1) * 32)
        cell = _Cell(cid, bounds)
        mem = _Membership(cid, ["f0"], ["f0", "f1"])
        cells.append(cell)
        tasks.append((cell, None, None))
        ctxs.append((row, col, cell, None, mem))
    return SimpleNamespace(
        layout={"nx": 2, "ny": 2},
        cpu_streaming_tile=int(cpu_tile),
        gpu_streaming_tile=gpu_tile,
        gpu_used=gpu_tile is not None,
        gpu_parity_qualified=bool(parity),
        cell_tasks=tasks,
        cell_ctxs=ctxs,
        chosen_mode="stream",
        working_set_block={
            "class": zz.WORKING_SET_RAM_TILE,
            "tile_size": int(cpu_tile),
            "column_batch": None,
        },
    )


def _canvas():
    return SimpleNamespace(
        height=64,
        width=80,
        wcs_header="WCSLIB: synthetic header",
        canvas_id="canvas-frozen",
        resolution_deg=0.5,
    )


def _gauge(reference_index=0):
    return SimpleNamespace(
        reference_index=int(reference_index),
        exclusions=[],
        coefficients=[0.5, 1.0, 1.25, 1.5],
        weights=[1.0, 1.0, 1.0, 1.0],
    )


def _science(**overrides):
    base = dict(
        cell_gauge_mode="per_cell",
        contract_version="zegrid-science-v1",
        normalization="sky_mean",
        weighting="noise_variance",
        rejection="kappa_sigma",
        combine="mean",
        taper="footprint",
        taper_px=8.0,
        taper_floor=0.0,
        sigma_low=3.0,
        sigma_high=3.0,
        winsor_limit_low=0.05,
        winsor_limit_high=0.05,
        backend="cpu",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _frames(tmp_path):
    frames = []
    for index in range(2):
        path = tmp_path / f"f{index}.fits"
        path.write_bytes((f"frame-{index}-" * 32).encode("utf-8"))
        frames.append(_Desc(f"f{index}.fits", str(path)))
    return frames


def _identity(
    tmp_path,
    *,
    bundle,
    science,
    effective_backend=None,
    gauge=None,
):
    frames = _frames(tmp_path)
    return zz._r55_build_run_store_identity(
        run_id="run-effective",
        descs=frames,
        canvas=_canvas(),
        bundle=bundle,
        halo_px=4,
        halo_provenance={"effective_px": 4, "source": "run_parameter"},
        global_gauge=gauge or _gauge(),
        global_frame_ids=[d.frame_id.logical_path for d in frames],
        gauge_mode=str(getattr(science, "cell_gauge_mode", "global")),
        science_config=science,
        workers=2,
        prefetch_units=None,
        effective_backend=effective_backend,
    )


# ---------------------------------------------------------------------------
# 1 — ONE decision, in both directions
# ---------------------------------------------------------------------------


def _admission(tile_px=64, *, backend="cpu"):
    return zres.build_admission(
        backend=backend, tile_px=int(tile_px), parity_qualified=False
    )


def test_effective_backend_is_single_sourced_in_both_directions():
    # Fresh run / pre-parity store: the selected backend is the effective one.
    assert zz._effective_run_backend(None, gpu_used=False) == "cpu"
    assert zz._effective_run_backend(None, gpu_used=True) == "gpu"
    assert zz._frozen_planning_gpu(None, gpu_used=True) is True
    assert zz._frozen_planning_gpu(None, gpu_used=False) is False

    # CPU-frozen: a GPU offered by this host is IGNORED (never a switch).
    cpu_frozen = _admission(64, backend="cpu")
    assert zz._effective_run_backend(cpu_frozen, gpu_used=True) == "cpu"
    assert zz._effective_run_backend(cpu_frozen, gpu_used=False) == "cpu"
    assert zz._frozen_planning_gpu(cpu_frozen, gpu_used=True) is False

    # GPU-frozen: the frozen backend governs; with a GPU it stays GPU.
    gpu_frozen = _admission(213, backend="gpu")
    assert zz._effective_run_backend(gpu_frozen, gpu_used=True) == "gpu"
    assert zz._frozen_planning_gpu(gpu_frozen, gpu_used=True) is True

    # GPU-frozen without a GPU: the typed refusal is preserved (no silent CPU).
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable) as excinfo:
        zz._effective_run_backend(gpu_frozen, gpu_used=False)
    assert excinfo.value.frozen_backend == "gpu"
    assert excinfo.value.frozen_tile_px == 213
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._frozen_planning_gpu(gpu_frozen, gpu_used=False)

    # Unknown frozen token: refused, never guessed.
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._effective_run_backend(
            {"backend": "tpu", "tile_px": 64, "parity_qualified": False},
            gpu_used=True,
        )


# ---------------------------------------------------------------------------
# 2 — the CPU-frozen science identity is unchanged by a GPU request
# ---------------------------------------------------------------------------


def test_cpu_frozen_identity_ignores_a_gpu_request(tmp_path):
    bundle = _scope_bundle(cpu_tile=96)
    # The identity an untouched CPU run stores today (requested == effective).
    stored = _identity(tmp_path, bundle=bundle, science=_science())
    stored_science = zrs.identity_fingerprints(stored)["science"]
    stored_identity = zrs.identity_sha256(stored)
    stored_backend = zrs.identity_fingerprints(stored)["backend"]

    # NUMERIC PIN: the stored contract is not silently redefined.
    assert stored_science == PINNED_CPU_SCIENCE_SHA256
    assert stored_identity == PINNED_CPU_IDENTITY_SHA256

    # The DEFECT reproduced: the current process requested a GPU (the science
    # config carries it) while the plan/bundle is frozen CPU. Without the
    # normalisation only the science section differs (exactly the production
    # message), and the resume is refused.
    requested = _identity(
        tmp_path, bundle=bundle, science=_science(backend="gpu")
    )
    assert zrs.identity_fingerprints(requested)["science"] == (
        PINNED_REQUESTED_GPU_ON_CPU_SCIENCE_SHA256
    )
    assert zrs.identity_fingerprints(requested)["science"] != stored_science
    assert zrs.identity_fingerprints(requested)["backend"] == stored_backend
    verdict = zrs.check_identity_compatibility(stored, requested)
    assert verdict["compatible"] is False
    assert [r["section"] for r in verdict["reasons"]] == ["science"]

    # THE FIX: the identity is built with the EFFECTIVE (frozen) backend.
    effective = _identity(
        tmp_path,
        bundle=bundle,
        science=_science(backend="gpu"),
        effective_backend="cpu",
    )
    assert zrs.identity_fingerprints(effective)["science"] == stored_science
    assert zrs.identity_sha256(effective) == stored_identity
    assert zrs.identity_fingerprints(effective)["backend"] == stored_backend
    assert zrs.check_identity_compatibility(stored, effective)["compatible"]


def test_gpu_frozen_identity_refreezes_gpu(tmp_path):
    bundle = _scope_bundle(cpu_tile=0, gpu_tile=213, parity=True)
    gpu_identity = _identity(
        tmp_path,
        bundle=bundle,
        science=_science(backend="gpu"),
        effective_backend="gpu",
    )
    assert gpu_identity["science_contract"]["backend"] == {
        "backend": "gpu",
        "tile_px": 213,
        "parity_qualified": True,
    }
    assert zrs.identity_fingerprints(gpu_identity)["science"] == (
        PINNED_GPU_SCIENCE_SHA256
    )
    assert zrs.identity_sha256(gpu_identity) == PINNED_GPU_IDENTITY_SHA256

    # The effective backend governs even if the config were still CPU-shaped.
    same = _identity(
        tmp_path,
        bundle=bundle,
        science=_science(backend="cpu"),
        effective_backend="gpu",
    )
    assert zrs.identity_sha256(same) == zrs.identity_sha256(gpu_identity)


def test_identity_builder_refuses_a_contradicting_effective_backend(tmp_path):
    with pytest.raises(zz._R55RunStoreIdentityError):
        _identity(
            tmp_path,
            bundle=_scope_bundle(cpu_tile=96),
            science=_science(),
            effective_backend="gpu",
        )
    with pytest.raises(zz._R55RunStoreIdentityError):
        _identity(
            tmp_path,
            bundle=_scope_bundle(cpu_tile=96),
            science=_science(),
            effective_backend="tpu",
        )


# ---------------------------------------------------------------------------
# 3 — genuine science changes still fail closed with the historical message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"normalization": "linear_fit"},
        {"weighting": "uniform"},
        {"rejection": "winsorized_sigma_clip"},
        {"combine": "median"},
        {"sigma_low": 2.5},
        {"sigma_high": 4.0},
        {"winsor_limit_low": 0.10},
        {"winsor_limit_high": 0.20},
        {"taper": "none"},
        {"taper_px": 16.0},
        {"taper_floor": 0.5},
        {"cell_gauge_mode": "global"},
    ],
)
def test_genuine_science_changes_are_still_refused(tmp_path, overrides):
    bundle = _scope_bundle(cpu_tile=96)
    stored = _identity(tmp_path, bundle=bundle, science=_science())
    changed = _identity(
        tmp_path,
        bundle=bundle,
        science=_science(**overrides),
        effective_backend="cpu",
    )
    assert zrs.identity_fingerprints(changed)["science"] != (
        zrs.identity_fingerprints(stored)["science"]
    )
    verdict = zrs.check_identity_compatibility(stored, changed)
    assert verdict["compatible"] is False
    # ``cell_gauge_mode`` legitimately also moves the gauge/dtype blocks; every
    # other override is science-only. The refusal is what matters.
    sections = [r["section"] for r in verdict["reasons"]]
    assert "science" in sections
    if set(overrides) != {"cell_gauge_mode"}:
        assert sections == ["science"]
    message = zrs.describe_identity_mismatch(verdict["reasons"])
    assert "science: stored=" in message and "current=" in message
    assert "science contract or effective parameters changed" in message
    assert "NEW output folder" in message


def test_reference_index_change_is_still_refused(tmp_path):
    """``reference_index`` lives in the gauge identity and stays fail-closed."""
    bundle = _scope_bundle(cpu_tile=96)
    stored = _identity(tmp_path, bundle=bundle, science=_science())
    other_ref = _identity(
        tmp_path,
        bundle=bundle,
        science=_science(),
        effective_backend="cpu",
        gauge=_gauge(reference_index=1),
    )
    verdict = zrs.check_identity_compatibility(stored, other_ref)
    assert verdict["compatible"] is False
    assert [r["section"] for r in verdict["reasons"]] == ["gauge"]


def test_a_real_backend_switch_is_still_refused(tmp_path):
    """A frozen contract never owns a switch to another backend."""
    cpu_bundle = _scope_bundle(cpu_tile=96)
    stored = _identity(tmp_path, bundle=cpu_bundle, science=_science())
    gpu_bundle = _scope_bundle(cpu_tile=0, gpu_tile=213, parity=True)
    switched = _identity(
        tmp_path,
        bundle=gpu_bundle,
        science=_science(backend="gpu"),
        effective_backend="gpu",
    )
    verdict = zrs.check_identity_compatibility(stored, switched)
    assert verdict["compatible"] is False
    sections = [r["section"] for r in verdict["reasons"]]
    assert "backend" in sections
    assert "science" in sections
    backend_reason = next(
        r for r in verdict["reasons"] if r["section"] == "backend"
    )
    assert backend_reason["stored"]["backend"] == "cpu"
    assert backend_reason["current"]["backend"] == "gpu"


# ---------------------------------------------------------------------------
# 4 — the product sites consume the single decision
# ---------------------------------------------------------------------------


def test_run_single_builds_both_identities_with_the_effective_backend():
    source = (
        REPO_ROOT / "src" / "zemosaic" / "zemosaic_zegrid_mode.py"
    ).read_text(encoding="utf-8")
    assert source.count("effective_backend=_effective_backend") == 2
    # ZM-ZEGRID-HARDENING-PASS (H6): TWO documented resolutions, both required:
    # the pre-admission one (host-selected -> plan gate) and the post-admission
    # one (final executed value -> identity). They are equal on every resume
    # path without a pre-execution fallback (proven at runtime below).
    assert '_effective_backend = "gpu" if bool(gpu_used) else "cpu"' not in source
    assert "_frozen_planning_gpu(\n            _frozen_admission, gpu_used=bool(gpu_used)\n        )" in source
    start = source.index("def _run_single(")
    body = source[start:source.index("\ndef ", start + 1)]
    assert body.count("_frozen_planning_gpu(") == 2
    assert "REQUIRED to be a second call" in body
    assert 'if _requested_backend != _effective_backend:' in source
    assert 'science_config = replace(science_config, backend=_effective_backend)' in source


# ---------------------------------------------------------------------------
# ZM-ZEGRID-HARDENING-PASS (H5) — science params must carry `backend`
# ---------------------------------------------------------------------------

def test_identity_builder_refuses_science_params_without_backend(tmp_path):
    science = _science()
    del science.backend
    assert "backend" not in vars(science)
    with pytest.raises(zz._R55RunStoreIdentityError) as exc:
        _identity(
            tmp_path,
            bundle=_scope_bundle(cpu_tile=96),
            science=science,
            effective_backend="cpu",
        )
    message = str(exc.value)
    assert "backend" in message
    assert "science parameters" in message

    # The same refusal applies to a GPU-frozen resume.
    with pytest.raises(zz._R55RunStoreIdentityError):
        _identity(
            tmp_path,
            bundle=_scope_bundle(cpu_tile=96, gpu_tile=128, parity=True),
            science=science,
            effective_backend="gpu",
        )


# ---------------------------------------------------------------------------
# ZM-ZEGRID-HARDENING-PASS (H6) — ONE resolution, reused by the identity
# ---------------------------------------------------------------------------

def test_h6_two_resolutions_agree_on_the_resume_path(tmp_path):
    """H6: both documented resolutions return the SAME token on a resume.

    Without a pre-execution fallback the post-admission ``gpu_used`` is the
    pre-admission value coerced by the first call, so the identity freezes
    exactly the backend the plan and the ignore diagnostics used.
    """
    cpu_frozen = _admission(tile_px=64, backend="cpu")
    gpu_frozen = _admission(tile_px=64, backend="gpu")
    # Pre-admission (host requests GPU) == post-admission (final value) on resume.
    for admission, host in ((cpu_frozen, True), (cpu_frozen, False),
                            (gpu_frozen, True)):
        first = zz._frozen_planning_gpu(admission, gpu_used=bool(host))
        second = zz._frozen_planning_gpu(
            admission, gpu_used=bool(first))
        assert first == second
    # A GPU-frozen resume without a GPU is still the typed refusal (the guard
    # the identity relies on is never relaxed).
    with pytest.raises(zres.ResumeFrozenAdmissionUnavailable):
        zz._frozen_planning_gpu(gpu_frozen, gpu_used=False)

    # The identity freezes the value the resolution returns, on both backends.
    cpu_identity = _identity(
        tmp_path, bundle=_scope_bundle(cpu_tile=96),
        science=_science(backend="gpu"), effective_backend="cpu")
    assert cpu_identity["science_contract"]["science"]["params"]["backend"] == "cpu"
    gpu_identity = _identity(
        tmp_path, bundle=_scope_bundle(cpu_tile=96, gpu_tile=128, parity=True),
        science=_science(backend="cpu"), effective_backend="gpu")
    assert gpu_identity["science_contract"]["science"]["params"]["backend"] == "gpu"
    assert gpu_identity["science_contract"]["backend"]["backend"] == "gpu"
