"""Characterization witness: GUI/config → worker process wrapper → run_hierarchical_mosaic
effective-kwargs contract (dispatch propagation).

Pins the *current* value/rename/parse/drop behavior of the real
``run_hierarchical_mosaic_process`` wrapper before R2 extraction.  This is a
characterization witness, not an endorsement: behavior observed here is the
baseline to preserve (or deliberately change, with a separate decision).

Design notes
------------
* We exercise the real wrapper, not a copied rename map.  The only substitution
  is the heavy downstream callable: ``zemosaic.zemosaic_worker.run_hierarchical_mosaic``
  is monkeypatched to a capture stub whose ``__signature__`` mirrors the real
  function, so the wrapper's ``inspect.signature(run_hierarchical_mosaic)`` still
  resolves the real 104-parameter contract while the stub records the exact
  ``final_kwargs`` actually delivered.
* ``crash_breadcrumbs_mode="off"`` is injected so the wrapper does not start its
  heartbeat thread and does not write breadcrumb files (no files, no threads).
* A minimal queue double replaces the multiprocessing queue: the wrapper only
  needs ``.put()``.
* Falsy values (``False``, ``0``, ``""``) and non-default numerics are asserted to
  survive, because the current wrapper keys off ``in`` membership rather than
  truthiness and the current contract therefore preserves them.

Silent drop of unknown kwargs is characterized and labeled an architectural
risk, not an endorsement.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_worker as zw  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal queue double
# ---------------------------------------------------------------------------


class _RecordingQueue:
    """Minimal stand-in for the worker's multiprocessing queue (needs only .put())."""

    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_signature_params() -> dict[str, inspect.Parameter]:
    return inspect.signature(zw.run_hierarchical_mosaic).parameters


def _capture_run(monkeypatch: pytest.MonkeyPatch, **kwargs) -> tuple[dict, list, int]:
    """Run the real wrapper with ``run_hierarchical_mosaic`` replaced by a capture stub.

    Returns ``(captured_final_kwargs, queue_items, call_count)``.
    """
    real_sig = inspect.signature(zw.run_hierarchical_mosaic)

    state: dict = {"kwargs": None, "calls": 0}

    def capture_stub(**final_kwargs: object) -> None:
        state["calls"] += 1
        state["kwargs"] = final_kwargs

    # Mirror the real signature so the wrapper's signature-based rename/suffix/filter
    # logic behaves exactly as in production.
    capture_stub.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", capture_stub)

    queue = _RecordingQueue()
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("crash_breadcrumbs_mode", "off")

    zw.run_hierarchical_mosaic_process(queue, solver_settings_dict=None, **call_kwargs)

    captured = state["kwargs"] if state["kwargs"] is not None else {}
    return captured, list(queue.items), int(state["calls"])


def _assert_signature_guard(captured: dict) -> None:
    """Every kwarg delivered downstream must be accepted by the real signature."""
    real_params = set(_real_signature_params())
    unexpected = set(captured) - real_params
    assert not unexpected, f"downstream kwargs not in real signature: {sorted(unexpected)}"


@pytest.fixture
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate HOME/XDG and cwd; restore crash-breadcrumb module globals after the run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / ".local" / "share"))

    saved = (
        zw._CRASH_BREADCRUMB_PATH,
        zw._CRASH_STATE_PATH,
        zw._CRASH_BREADCRUMB_MODE,
    )
    yield
    zw._CRASH_BREADCRUMB_PATH, zw._CRASH_STATE_PATH, zw._CRASH_BREADCRUMB_MODE = saved


# ---------------------------------------------------------------------------
# 1. Representative GUI/config names map to effective worker names
# ---------------------------------------------------------------------------


def test_gui_config_names_map_to_worker_names_preserving_values(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None
) -> None:
    captured, _q, calls = _capture_run(
        monkeypatch,
        input_dir="/data/input",
        output_dir="/data/output",
        stacking_normalize_method="linear_fit",
        stacking_weighting_method="noise_variance",
        stacking_rejection_algorithm="winsorized_sigma_clip",
        stacking_final_combine_method="mean",
        stacking_kappa_low=2.5,
        stacking_kappa_high=3.5,
    )
    _assert_signature_guard(captured)
    assert calls == 1

    # GUI config names -> effective worker names, values preserved exactly.
    assert captured["input_folder"] == "/data/input"
    assert captured["output_folder"] == "/data/output"
    assert captured["stack_norm_method"] == "linear_fit"
    assert captured["stack_weight_method"] == "noise_variance"
    assert captured["stack_reject_algo"] == "winsorized_sigma_clip"
    assert captured["stack_final_combine"] == "mean"
    assert captured["stack_kappa_low"] == 2.5
    assert captured["stack_kappa_high"] == 3.5

    # Original GUI keys must NOT leak through to the downstream callable.
    for gui_key in (
        "input_dir",
        "output_dir",
        "stacking_normalize_method",
        "stacking_weighting_method",
        "stacking_rejection_algorithm",
        "stacking_final_combine_method",
        "stacking_kappa_low",
        "stacking_kappa_high",
    ):
        assert gui_key not in captured, f"GUI key leaked downstream: {gui_key}"


# ---------------------------------------------------------------------------
# 2. stacking_winsor_limits parsing (current behavior, not desired behavior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.05,0.05", (0.05, 0.05)),
        ("0.10,0.20", (0.1, 0.2)),
        (" 0.10 , 0.20 ", (0.1, 0.2)),
        # Everything that is not exactly two comma-separated floats falls back to
        # the hard-coded (0.05, 0.05) default — current behavior, pinned as-is.
        ("0.1,0.2,0.3", (0.05, 0.05)),
        ("0.5", (0.05, 0.05)),
        ("", (0.05, 0.05)),
        ("not-a-number", (0.05, 0.05)),
    ],
)
def test_stacking_winsor_limits_forms_parse_to_delivered_tuple(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, raw: str, expected: tuple
) -> None:
    captured, _q, calls = _capture_run(monkeypatch, stacking_winsor_limits=raw)
    _assert_signature_guard(captured)
    assert calls == 1

    # Delivered as a tuple under parsed_winsor_limits (never as a raw string/path).
    assert "stacking_winsor_limits" not in captured
    assert isinstance(captured["parsed_winsor_limits"], tuple)
    assert captured["parsed_winsor_limits"] == expected


# ---------------------------------------------------------------------------
# 3. valid _config suffix promotion / signature-aware handling
# ---------------------------------------------------------------------------


def test_valid_config_suffix_promotion(monkeypatch: pytest.MonkeyPatch, _isolated_env: None) -> None:
    captured, _q, calls = _capture_run(
        monkeypatch,
        poststack_equalize_rgb=False,
        num_base_workers=7,
        stack_ram_budget_gb=3.5,
    )
    _assert_signature_guard(captured)
    assert calls == 1

    # A key whose "_config"-suffixed form is in the signature but whose bare form
    # is not is promoted to the suffixed name.
    assert captured["poststack_equalize_rgb_config"] is False
    assert captured["num_base_workers_config"] == 7
    assert captured["stack_ram_budget_gb_config"] == 3.5
    assert "poststack_equalize_rgb" not in captured
    assert "num_base_workers" not in captured
    assert "stack_ram_budget_gb" not in captured


# ---------------------------------------------------------------------------
# 4. Unknown kwargs are silently dropped today
# ---------------------------------------------------------------------------


def test_unknown_kwargs_silently_dropped(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None
) -> None:
    captured, _q, calls = _capture_run(
        monkeypatch,
        stacking_kappa_low=3.0,
        bogus_unknown_kwarg=123,
        another_made_up_flag=True,
        not_a_real_parameter="hello",
    )
    _assert_signature_guard(captured)
    assert calls == 1

    # Unknown kwargs never reach run_hierarchical_mosaic.
    for bad in ("bogus_unknown_kwarg", "another_made_up_flag", "not_a_real_parameter"):
        assert bad not in captured, f"unknown kwarg leaked downstream: {bad}"
    # Known kwarg still arrives.
    assert captured["stack_kappa_low"] == 3.0


# ---------------------------------------------------------------------------
# 5. Wrapper defaults when absent
# ---------------------------------------------------------------------------


def test_wrapper_defaults_when_absent(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None
) -> None:
    captured, _q, calls = _capture_run(monkeypatch, stacking_kappa_low=3.0)
    _assert_signature_guard(captured)
    assert calls == 1

    assert captured["stack_ram_budget_gb_config"] == 0.0
    assert captured["num_base_workers_config"] == 0


# ---------------------------------------------------------------------------
# 6. Falsy values are not lost by truthiness handling
# ---------------------------------------------------------------------------


def test_falsy_values_preserved(monkeypatch: pytest.MonkeyPatch, _isolated_env: None) -> None:
    captured, _q, calls = _capture_run(
        monkeypatch,
        poststack_equalize_rgb=False,
        stacking_kappa_low=0.0,
        stacking_normalize_method="",
        stacking_kappa_high=4.5,
    )
    _assert_signature_guard(captured)
    assert calls == 1

    assert captured["poststack_equalize_rgb_config"] is False
    assert captured["stack_kappa_low"] == 0.0
    assert captured["stack_norm_method"] == ""
    assert captured["stack_kappa_high"] == 4.5


# ---------------------------------------------------------------------------
# 7. Downstream invoked exactly once; no accidental heavy work
# ---------------------------------------------------------------------------


def test_downstream_invoked_exactly_once_no_side_effects(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None, tmp_path: Path
) -> None:
    captured, queue_items, calls = _capture_run(
        monkeypatch,
        stacking_kappa_low=3.0,
        stacking_kappa_high=3.0,
        stacking_winsor_limits="0.05,0.05",
    )
    _assert_signature_guard(captured)
    assert calls == 1

    # The wrapper completed its normal path: PROCESS_DONE emitted, no PROCESS_ERROR.
    assert any(
        isinstance(item, tuple) and item and item[0] == "PROCESS_DONE" for item in queue_items
    )
    assert not any(
        isinstance(item, tuple) and item and item[0] == "PROCESS_ERROR" for item in queue_items
    )

    # No files were written (breadcrumbs disabled; output dir not created).
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Generic signature guard (every delivered kwarg accepted by the real signature)
# ---------------------------------------------------------------------------


def test_generic_signature_guard_holds_for_full_kwarg_set(
    monkeypatch: pytest.MonkeyPatch, _isolated_env: None
) -> None:
    captured, _q, calls = _capture_run(
        monkeypatch,
        input_dir="/in",
        output_dir="/out",
        stacking_normalize_method="additive",
        stacking_weighting_method="none",
        stacking_rejection_algorithm="kappa_sigma",
        stacking_final_combine_method="median",
        stacking_kappa_low=1.0,
        stacking_kappa_high=2.0,
        stacking_winsor_limits="0.1,0.2",
        poststack_equalize_rgb=True,
        bogus_unknown_kwarg="should be dropped",
    )
    _assert_signature_guard(captured)
    assert calls == 1

    # The wrapper always injects progress_callback and solver_settings; both are
    # real signature parameters and must be present.
    assert "progress_callback" in captured
    assert "solver_settings" in captured
    assert captured["solver_settings"] is None
