"""Initial Qt filter dialog for ZeMosaic."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence


_pyside_spec = importlib.util.find_spec("PySide6")
if _pyside_spec is None:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt filter interface. "
        "Install PySide6 or use the Tk interface instead."
    )

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (  # noqa: E402  - imported after availability check
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


if importlib.util.find_spec("locales.zemosaic_localization") is not None:
    from locales.zemosaic_localization import ZeMosaicLocalization  # type: ignore
else:  # pragma: no cover - optional dependency guard
    ZeMosaicLocalization = None  # type: ignore[assignment]


class _FallbackLocalizer:
    """Very small localization shim when the full helper is unavailable."""

    def __init__(self, language_code: str = "en") -> None:
        self.language_code = language_code

    def get(self, key: str, default_text: str | None = None, **_: Any) -> str:
        return default_text if default_text is not None else key

    def set_language(self, language_code: str) -> None:
        self.language_code = language_code


@dataclass(slots=True)
class _NormalizedItem:
    """Container describing a row displayed by the Qt dialog."""

    original: Any
    display_name: str
    file_path: str
    has_wcs: bool
    instrument: str | None
    include_by_default: bool = True


class FilterQtDialog(QDialog):
    """Small Qt dialog allowing users to review candidate files.

    The dialog intentionally focuses on the essentials for now:

    * present the list of candidate files that will be used for the mosaic;
    * allow end users to include or exclude individual entries;
    * expose OK/Cancel semantics equivalent to the Tk filter interface.

    The dialog stores the input payload so that future iterations can add
    richer analysis (clustering, previews, stream scanning, ...).
    """

    def __init__(
        self,
        raw_files_with_wcs_or_dir: Any,
        initial_overrides: Any = None,
        *,
        stream_scan: bool = False,
        scan_recursive: bool = True,
        batch_size: int = 100,
        preview_cap: int = 200,
        solver_settings_dict: dict | None = None,
        config_overrides: dict | None = None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._input_payload = raw_files_with_wcs_or_dir
        self._initial_overrides = initial_overrides
        self._stream_scan = stream_scan
        self._scan_recursive = scan_recursive
        self._batch_size = batch_size
        self._preview_cap = preview_cap
        self._solver_settings = solver_settings_dict
        self._config_overrides = config_overrides or {}
        self._accepted = False

        self._localizer = self._load_localizer()
        self._normalized_items = self._normalize_items(raw_files_with_wcs_or_dir, initial_overrides)

        self._table = QTableWidget(self)
        self._summary_label = QLabel(self)

        self._build_ui()
        self._populate_table()
        self._update_summary_label()

    # ------------------------------------------------------------------
    # Helpers - localization and normalization
    # ------------------------------------------------------------------
    def _load_localizer(self) -> Any:
        language = None
        try:
            language = str(self._config_overrides.get("language"))
        except Exception:
            language = None

        if ZeMosaicLocalization is not None:
            try:
                loc = ZeMosaicLocalization()
                if language:
                    loc.set_language(language)
                return loc
            except Exception:
                pass
        fallback = _FallbackLocalizer(language or "en")
        return fallback

    def _normalize_items(
        self,
        payload: Any,
        initial_overrides: Any,
    ) -> List[_NormalizedItem]:
        """Materialise the input payload into ``_NormalizedItem`` instances."""

        # Extract optional exclusion hints from overrides
        excluded_paths: Sequence[str] = ()
        if isinstance(initial_overrides, dict):
            candidate = initial_overrides.get("excluded_paths")
            if isinstance(candidate, (list, tuple, set)):
                excluded_paths = [os.fspath(p) for p in candidate]

        items: list[_NormalizedItem] = []

        def _should_exclude(path: str) -> bool:
            return any(os.path.normcase(path) == os.path.normcase(entry) for entry in excluded_paths)

        def _build_from_mapping(obj: dict) -> _NormalizedItem:
            original = obj
            instrument = None
            for key in ("instrument", "INSTRUME", "instrument_name"):
                if key in obj and obj[key]:
                    try:
                        instrument = str(obj[key])
                    except Exception:
                        instrument = None
                    if instrument:
                        break
            file_path = str(
                obj.get("file_path")
                or obj.get("path")
                or obj.get("filepath")
                or obj.get("filename")
                or obj.get("full_path")
                or obj.get("fullpath")
                or obj.get("name")
                or obj.get("src")
                or ""
            )
            if not file_path and "header" in obj:
                try:
                    file_path = str(obj["header"].get("FILENAME", ""))
                except Exception:
                    file_path = ""
            has_wcs = False
            for key in ("has_wcs", "wcs_solved", "has_valid_wcs"):
                if obj.get(key):
                    has_wcs = True
                    break
            if not has_wcs and "wcs" in obj:
                has_wcs = obj["wcs"] is not None
            display_name = file_path or instrument or "Item"
            include = not _should_exclude(file_path)
            return _NormalizedItem(
                original=original,
                display_name=display_name,
                file_path=file_path,
                has_wcs=has_wcs,
                instrument=instrument,
                include_by_default=include,
            )

        def _build_from_path(path_obj: Path) -> _NormalizedItem:
            file_path = str(path_obj)
            display_name = path_obj.name or file_path
            include = not _should_exclude(file_path)
            return _NormalizedItem(
                original=file_path,
                display_name=display_name,
                file_path=file_path,
                has_wcs=False,
                instrument=None,
                include_by_default=include,
            )

        if isinstance(payload, (str, os.PathLike)):
            directory = Path(payload)
            if directory.is_dir():
                pattern = "**/*" if self._scan_recursive else "*"
                for candidate in directory.glob(pattern):
                    if candidate.is_file():
                        items.append(_build_from_path(candidate))
            elif Path(payload).is_file():
                items.append(_build_from_path(Path(payload)))
        elif isinstance(payload, Iterable):
            for element in payload:
                if isinstance(element, dict):
                    items.append(_build_from_mapping(element))
                elif isinstance(element, (str, os.PathLike)):
                    items.append(_build_from_path(Path(element)))
                else:
                    items.append(
                        _NormalizedItem(
                            original=element,
                            display_name=str(element),
                            file_path=str(element),
                            has_wcs=False,
                            instrument=None,
                            include_by_default=True,
                        )
                    )
        else:
            items.append(
                _NormalizedItem(
                    original=payload,
                    display_name=str(payload),
                    file_path=str(payload),
                    has_wcs=False,
                    instrument=None,
                    include_by_default=True,
                )
            )

        return items

    # ------------------------------------------------------------------
    # UI creation
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        title = self._localizer.get("filter.dialog.title", "Filter raw frames")
        self.setWindowTitle(title)
        self.resize(800, 500)

        main_layout = QVBoxLayout(self)

        description = self._localizer.get(
            "filter.dialog.description",
            "Review the frames that will be used before launching the mosaic process.",
        )
        main_layout.addWidget(QLabel(description, self))

        self._table.setColumnCount(3)
        headers = [
            self._localizer.get("filter.column.file", "File"),
            self._localizer.get("filter.column.instrument", "Instrument"),
            self._localizer.get("filter.column.wcs", "WCS"),
        ]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(self._handle_item_changed)
        main_layout.addWidget(self._table)

        control_row = QGridLayout()
        select_all_btn = QPushButton(self._localizer.get("filter.button.select_all", "Select all"), self)
        select_none_btn = QPushButton(self._localizer.get("filter.button.select_none", "Select none"), self)
        select_all_btn.clicked.connect(self._select_all)  # type: ignore[arg-type]
        select_none_btn.clicked.connect(self._select_none)  # type: ignore[arg-type]
        control_row.addWidget(select_all_btn, 0, 0)
        control_row.addWidget(select_none_btn, 0, 1)
        control_row.addWidget(self._summary_label, 0, 2)
        control_row.setColumnStretch(2, 1)
        main_layout.addLayout(control_row)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _populate_table(self) -> None:
        items = self._normalized_items
        self._table.blockSignals(True)
        self._table.setRowCount(len(items))
        for row, entry in enumerate(items):
            name_item = QTableWidgetItem(entry.display_name)
            name_item.setFlags(name_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            name_item.setCheckState(Qt.Checked if entry.include_by_default else Qt.Unchecked)
            self._table.setItem(row, 0, name_item)

            instrument = entry.instrument or self._localizer.get("filter.value.unknown", "Unknown")
            instrument_item = QTableWidgetItem(instrument)
            instrument_item.setFlags(instrument_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 1, instrument_item)

            has_wcs_text = (
                self._localizer.get("filter.value.wcs_present", "Yes")
                if entry.has_wcs
                else self._localizer.get("filter.value.wcs_missing", "No")
            )
            wcs_item = QTableWidgetItem(has_wcs_text)
            wcs_item.setFlags(wcs_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 2, wcs_item)

        self._table.blockSignals(False)

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def _handle_item_changed(self, _: QTableWidgetItem) -> None:
        self._update_summary_label()

    def _select_all(self) -> None:
        self._toggle_all(True)

    def _select_none(self) -> None:
        self._toggle_all(False)

    def _toggle_all(self, enabled: bool) -> None:
        state = Qt.Checked if enabled else Qt.Unchecked
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None:
                item.setCheckState(state)
        self._table.blockSignals(False)
        self._update_summary_label()

    def _update_summary_label(self) -> None:
        total = self._table.rowCount()
        selected = 0
        for row in range(total):
            item = self._table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                selected += 1
        summary_text = self._localizer.get(
            "filter.summary.selected",
            "{selected} of {total} frames selected",
            selected=selected,
            total=total,
        )
        try:
            self._summary_label.setText(summary_text.format(selected=selected, total=total))
        except Exception:
            self._summary_label.setText(f"{selected} / {total}")

    # ------------------------------------------------------------------
    # QDialog API
    # ------------------------------------------------------------------
    def accept(self) -> None:  # noqa: D401 - inherit docstring
        self._accepted = True
        super().accept()

    def reject(self) -> None:  # noqa: D401 - inherit docstring
        self._accepted = False
        super().reject()

    # ------------------------------------------------------------------
    # Public helpers queried by ``launch_filter_interface_qt``
    # ------------------------------------------------------------------
    def selected_items(self) -> List[Any]:
        results: list[Any] = []
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                results.append(entry.original)
        return results

    def was_accepted(self) -> bool:
        return self._accepted

    def overrides(self) -> Any:
        # Placeholder: future iterations will translate UI state into overrides
        return None


def launch_filter_interface_qt(
    raw_files_with_wcs_or_dir,
    initial_overrides=None,
    *,
    stream_scan=False,
    scan_recursive=True,
    batch_size=100,
    preview_cap=200,
    solver_settings_dict=None,
    config_overrides=None,
    **kwargs,
):
    """Launch the Qt-based filter dialog and return the user selection."""

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(["zemosaic-filter-qt"])
        owns_app = True

    dialog = FilterQtDialog(
        raw_files_with_wcs_or_dir,
        initial_overrides,
        stream_scan=stream_scan,
        scan_recursive=scan_recursive,
        batch_size=batch_size,
        preview_cap=preview_cap,
        solver_settings_dict=solver_settings_dict,
        config_overrides=config_overrides,
    )
    dialog.exec()

    try:
        selected = dialog.selected_items()
    finally:
        if owns_app:
            app.quit()

    if dialog.was_accepted():
        return selected, True, dialog.overrides()

    return raw_files_with_wcs_or_dir, False, None


__all__ = ["FilterQtDialog", "launch_filter_interface_qt"]
