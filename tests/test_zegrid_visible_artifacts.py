"""ZM-ZEGRID-VISIBLE-ARTIFACTS-20261031 — visible work/publication directories.

The ZeMosaic product must be transparent about what it writes: the durable store,
the publication container and the scratch cache are VISIBLE names.

    .zegrid-durable     -> zegrid-durable
    .zegrid_publications-> zegrid_publications
    __zegrid_cache__    -> zegrid-cache

Legacy hidden layouts are READ-ONLY fallbacks: nothing is moved, renamed or
deleted, and a legacy run still resumes and verifies. Ownership/precheck lists
recognise BOTH name generations. A visible name already taken by unrelated
content is a fail-closed refusal (nothing is written into foreign data).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (SRC, REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402

MIB = 2 ** 20
TINY_CORPUS_TOOL = REPO_ROOT / "tools" / "zegrid_m6" / "make_tiny_corpus.py"


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "in"
    if not corpus.exists():
        proc = subprocess.run(
            [sys.executable, str(TINY_CORPUS_TOOL), str(corpus)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
    return corpus


def _run(
    tmp_path: Path,
    *,
    resume_mode: str = "never",
    store_dir: Path | None = None,
    out: Path | None = None,
    corpus: Path | None = None,
    lines: list[str] | None = None,
):
    corpus = corpus or _corpus(tmp_path)
    out = out or (tmp_path / "out")
    store = store_dir or (out / "zegrid-durable")
    zconfig = SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_durable_store_dir=str(store),
        zegrid_resume_mode=str(resume_mode),
        zegrid_resume_run_id=None,
        zegrid_durable_store_budget_bytes=64 * MIB,
        zegrid_layout="2x2",
        zegrid_cell_gauge_mode="global",
        zegrid_resource_recovery_enabled=False,
        zegrid_resource_recovery_deadline_s=None,
        zegrid_sip_mode="keep",
        temporary_work_dir="",
    )
    zz.run_zegrid_mode(
        str(corpus),
        str(out),
        (lambda *args, **_kwargs: lines.append(str(args[0])))
        if lines is not None
        else None,
        zconfig=zconfig,
        zegrid_halo_px=8,
        use_gpu=False,
    )
    return out, Path(store), corpus


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files_under(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in names:
        for path in sorted(root.rglob(name)):
            if path.is_file():
                found[str(path.relative_to(root))] = _digest(path)
    return found


def test_fresh_run_writes_only_visible_work_directories(tmp_path):
    out, store, _corpus_dir = _run(tmp_path)
    assert (out / "zegrid_publications").is_dir()
    assert (out / "zegrid-durable").is_dir()
    assert (out / "zegrid-cache").is_dir()
    for legacy in (".zegrid_publications", ".zegrid-durable", "__zegrid_cache__"):
        assert not (out / legacy).exists(), legacy
    # The r4 discovery contract still points at the generation dir.
    payload = json.loads(
        (out / "zemosaic_products.json").read_text(encoding="utf-8")
    )
    gen_dir = Path(payload["generation_dir"])
    assert gen_dir.is_dir()
    assert gen_dir.is_relative_to((out / "zegrid_publications").resolve())
    assert (out / "PRODUCTS.txt").is_file()


def test_legacy_layout_resumes_and_verifies_without_being_moved(tmp_path):
    out, _store, corpus = _run(tmp_path)
    # Convert THIS TEST's own fresh layout into the old hidden one.
    os.rename(out / "zegrid_publications", out / ".zegrid_publications")
    os.rename(out / "zegrid-durable", out / ".zegrid-durable")
    if (out / "zegrid-cache").is_dir():
        os.rename(out / "zegrid-cache", out / "__zegrid_cache__")
    legacy_store = out / ".zegrid-durable"
    legacy_pub = out / ".zegrid_publications"
    before_publication = _files_under(legacy_pub, ("publication.json", "*.fits"))
    before_identity = _files_under(legacy_store, ("run.json", "state.json"))
    assert before_publication and before_identity

    lines: list[str] = []
    _run(
        tmp_path,
        resume_mode="auto",
        store_dir=legacy_store,
        out=out,
        corpus=corpus,
        lines=lines,
    )
    # ONE explicit line per legacy container in use (store + publication).
    publication_lines = [
        line for line in lines if "legacy publication layout in use" in line
    ]
    store_lines = [
        line for line in lines if "legacy checkpoint store in use" in line
    ]
    assert len(publication_lines) == 1, publication_lines
    assert len(store_lines) == 1, store_lines
    assert ".zegrid_publications" in publication_lines[0]
    assert ".zegrid-durable" in store_lines[0]

    # Nothing was moved/renamed/deleted: the hidden directories still exist and
    # the identity/publication bytes are unchanged.
    assert legacy_store.is_dir() and legacy_pub.is_dir()
    assert not (out / "zegrid_publications").exists()
    assert not (out / "zegrid-durable").exists()
    assert _files_under(legacy_pub, ("publication.json", "*.fits")) == before_publication
    assert _files_under(legacy_store, ("run.json", "state.json")) == before_identity

    # Products remain discoverable and still point into the legacy generation.
    payload = json.loads(
        (out / "zemosaic_products.json").read_text(encoding="utf-8")
    )
    gen_dir = Path(payload["generation_dir"])
    assert gen_dir.is_dir()
    assert gen_dir.is_relative_to(legacy_pub.resolve())


def test_publication_name_taken_by_foreign_content_fails_closed(tmp_path):
    out = tmp_path / "out"
    foreign = out / "zegrid_publications"
    foreign.mkdir(parents=True)
    (foreign / "foreign.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(zpub.PublicationPathError) as excinfo:
        _run(tmp_path, out=out)
    assert "unrelated content" in str(excinfo.value)
    assert (foreign / "foreign.txt").read_text(encoding="utf-8") == "user data"
    assert not list(foreign.glob("run-*"))


def test_store_name_taken_by_foreign_content_fails_closed(tmp_path):
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    store.mkdir(parents=True)
    (store / "foreign.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        _run(tmp_path, out=out, store_dir=store)
    message = str(excinfo.value)
    assert "durable store folder" in message and "Nothing was written" in message
    assert (store / "foreign.txt").read_text(encoding="utf-8") == "user data"
    assert not (store / "run.json").exists()


def test_cache_name_taken_by_foreign_content_fails_closed(tmp_path):
    out = tmp_path / "out"
    cache = out / "zegrid-cache"
    cache.mkdir(parents=True)
    (cache / "foreign.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        _run(tmp_path, out=out)
    message = str(excinfo.value)
    assert "cache directory" in message and "unrelated content" in message
    assert (cache / "foreign.txt").read_text(encoding="utf-8") == "user data"


@pytest.mark.parametrize(
    "dirname", ["zegrid_publications", ".zegrid_publications"]
)
def test_publication_ownership_recognises_both_names(tmp_path, dirname):
    out = tmp_path / "out"
    (out / dirname / "run-x" / "gen-1").mkdir(parents=True)
    info = zpub.detect_previous_publication(out)
    assert info["zegrid"] is True
    assert info["foreign"] == []
    assert dirname + "/" in info["evidence"]


@pytest.mark.parametrize(
    "name",
    [
        "zegrid_publications",
        ".zegrid_publications",
        "zegrid-cache",
        "__zegrid_cache__",
    ],
)
def test_gui_owned_names_recognise_both_generations(tmp_path, name, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:  # pragma: no cover - app bootstrap
        QApplication([])
    from test_zegrid_scheduler_gui_advanced import _make_window

    out = tmp_path / "out"
    (out / name).mkdir(parents=True)
    win = _make_window(tmp_path, monkeypatch, {"output_dir": str(out)})
    owned = win._zegrid_owned_output_names(out, None)
    assert name in owned


@pytest.mark.parametrize(
    "container", ["zegrid_publications", ".zegrid_publications"]
)
def test_gui_products_resolution_works_for_both_layouts(
    tmp_path, container, monkeypatch
):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:  # pragma: no cover - app bootstrap
        QApplication([])
    from test_zegrid_scheduler_gui_advanced import _make_window

    out = tmp_path / "out"
    gen = out / container / "run-x" / "gen-1"
    gen.mkdir(parents=True)
    (gen / "mosaic_grid_science.fits").write_bytes(b"science")
    # ZM-ZEGRID-ROOT-DELIVERABLES: the delivered root copy is what the user
    # opens; the generation directory stays the immutable archive.
    (out / "mosaic_grid_science.fits").write_bytes(b"science")
    (out / "zemosaic_products.json").write_text(
        json.dumps(
            {
                "schema": zpub.PRODUCTS_SCHEMA,
                "run_id": "r",
                "generation": "gen-1",
                "generation_dir": str(gen),
                "products": {},
                "root_files": {
                    "science": {
                        "path": str(out / "mosaic_grid_science.fits"),
                        "sha256": hashlib.sha256(b"science").hexdigest(),
                        "nbytes": len(b"science"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    win = _make_window(tmp_path, monkeypatch, {"output_dir": str(out)})
    win._zegrid_active = True
    assert win._zegrid_products_dir().resolve() == out.resolve()
    assert win._completion_open_target().resolve() == out.resolve()
    message = win._completion_message_text()
    assert "mosaic_grid_science.fits" in message
    assert str(out) in message
