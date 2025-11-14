"""Initial Qt filter dialog for ZeMosaic."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import math
from typing import Any, Iterable, Iterator, List, Sequence, Tuple


_pyside_spec = importlib.util.find_spec("PySide6")
if _pyside_spec is None:  # pragma: no cover - import guard
    raise ImportError(
        "PySide6 is required to use the ZeMosaic Qt filter interface. "
        "Install PySide6 or use the Tk interface instead."
    )

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (  # noqa: E402  - imported after availability check
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    QVBoxLayout,
)


if importlib.util.find_spec("locales.zemosaic_localization") is not None:
    from locales.zemosaic_localization import ZeMosaicLocalization  # type: ignore
else:  # pragma: no cover - optional dependency guard
    ZeMosaicLocalization = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for metadata extraction
    from astropy.io import fits
    from astropy.wcs import WCS
except Exception:  # pragma: no cover - optional dependency guard
    fits = None  # type: ignore[assignment]
    WCS = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - matplotlib optional
    FigureCanvasQTAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from zemosaic_astrometry import solve_with_astap, set_astap_max_concurrent_instances
except Exception:  # pragma: no cover - optional dependency guard
    solve_with_astap = None  # type: ignore[assignment]
    set_astap_max_concurrent_instances = None  # type: ignore[assignment]


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
    group_label: str | None
    include_by_default: bool = True
    center_ra_deg: float | None = None
    center_dec_deg: float | None = None
    cluster_index: int | None = None


def _iter_normalized_entries(
    payload: Any,
    initial_overrides: Any,
    *,
    scan_recursive: bool,
) -> Iterator[_NormalizedItem]:
    """Yield ``_NormalizedItem`` instances without eagerly materialising them."""

    excluded_paths: Sequence[str] = ()
    if isinstance(initial_overrides, dict):
        candidate = initial_overrides.get("excluded_paths")
        if isinstance(candidate, (list, tuple, set)):
            excluded_paths = [os.fspath(p) for p in candidate]

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
        group_label: str | None = None
        for key in (
            "group_label",
            "group",
            "group_id",
            "cluster",
            "cluster_id",
            "tile_group",
            "tile_group_id",
            "master_tile_id",
        ):
            if key in obj and obj[key] not in (None, ""):
                try:
                    group_label = str(obj[key])
                except Exception:
                    group_label = None
                if group_label:
                    break
        if group_label is None and "grouping" in obj:
            grouping = obj.get("grouping")
            if isinstance(grouping, dict):
                for key in ("label", "id", "name"):
                    candidate = grouping.get(key)
                    if candidate not in (None, ""):
                        try:
                            group_label = str(candidate)
                        except Exception:
                            group_label = None
                        if group_label:
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
        ra_deg = None
        dec_deg = None
        for key in ("center_ra_deg", "ra_deg", "ra"):
            if key in obj:
                try:
                    ra_deg = float(obj[key])
                except Exception:
                    ra_deg = None
                if ra_deg is not None:
                    break
        for key in ("center_dec_deg", "dec_deg", "dec"):
            if key in obj:
                try:
                    dec_deg = float(obj[key])
                except Exception:
                    dec_deg = None
                if dec_deg is not None:
                    break
        if (ra_deg is None or dec_deg is None) and "header" in obj:
            try:
                header_obj = obj["header"]
            except Exception:
                header_obj = None
            if header_obj is not None:
                ra_from_header, dec_from_header = _extract_center_from_header(header_obj)
                ra_deg = ra_deg if ra_deg is not None else ra_from_header
                dec_deg = dec_deg if dec_deg is not None else dec_from_header
        return _NormalizedItem(
            original=original,
            display_name=display_name,
            file_path=file_path,
            has_wcs=has_wcs,
            instrument=instrument,
            group_label=group_label,
            include_by_default=include,
            center_ra_deg=ra_deg,
            center_dec_deg=dec_deg,
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
            group_label=None,
            include_by_default=include,
        )

    if isinstance(payload, (str, os.PathLike)):
        directory = Path(payload)
        if directory.is_dir():
            pattern = "**/*" if scan_recursive else "*"
            for candidate in directory.glob(pattern):
                if candidate.is_file():
                    yield _build_from_path(candidate)
        elif Path(payload).is_file():
            yield _build_from_path(Path(payload))
    elif isinstance(payload, Iterable):
        for element in payload:
            if isinstance(element, dict):
                yield _build_from_mapping(element)
            elif isinstance(element, (str, os.PathLike)):
                yield _build_from_path(Path(element))
            else:
                yield _NormalizedItem(
                    original=element,
                    display_name=str(element),
                    file_path=str(element),
                    has_wcs=False,
                    instrument=None,
                    group_label=None,
                    include_by_default=True,
                )
    else:
        yield _NormalizedItem(
            original=payload,
            display_name=str(payload),
            file_path=str(payload),
            has_wcs=False,
            instrument=None,
            group_label=None,
            include_by_default=True,
        )


def _header_has_wcs(header: Any) -> bool:
    """Return ``True`` when the FITS header already contains a valid WCS."""

    if header is None:
        return False
    try:
        if WCS is not None:
            wcs_obj = WCS(header)
            if getattr(wcs_obj, "is_celestial", False):
                return True
    except Exception:
        pass
    for key in ("CTYPE1", "CTYPE2", "CD1_1", "CD2_2", "PC1_1", "PC2_2"):
        if header.get(key):
            return True
    return False


def _detect_instrument_from_header(header: Any) -> str | None:
    """Mimic the Tk GUI heuristics to detect the instrument label."""

    if header is None:
        return None
    try:
        creator_raw = str(header.get("CREATOR", "")).strip()
        instrume_raw = str(header.get("INSTRUME", "")).strip()
    except Exception:
        creator_raw = ""
        instrume_raw = ""
    creator = creator_raw.upper()
    instrume = instrume_raw.upper()
    if "SEESTAR S50" in creator:
        return "Seestar S50"
    if "SEESTAR S30" in creator:
        return "Seestar S30"
    if "ASIAIR" in creator:
        return instrume_raw or "ASIAIR"
    if instrume.startswith("ZWO ASI") or "ASI" in instrume:
        return instrume_raw or "ASI (Unknown)"
    if instrume_raw:
        return instrume_raw
    return creator_raw or None


def _sexagesimal_to_degrees(raw: str, is_ra: bool) -> float | None:
    """Convert sexagesimal strings to decimal degrees."""

    text = raw.strip().lower().replace("h", ":").replace("m", ":").replace("s", "")
    parts = [segment for segment in text.replace("°", ":").replace("'", ":").split(":") if segment]
    if not parts:
        return None

    try:
        numbers = [float(part) for part in parts]
    except Exception:
        return None

    if is_ra:
        hours = numbers[0]
        minutes = numbers[1] if len(numbers) > 1 else 0.0
        seconds = numbers[2] if len(numbers) > 2 else 0.0
        return (hours + minutes / 60.0 + seconds / 3600.0) * 15.0

    sign = -1.0 if text.startswith("-") else 1.0
    degrees = abs(numbers[0])
    arcmin = numbers[1] if len(numbers) > 1 else 0.0
    arcsec = numbers[2] if len(numbers) > 2 else 0.0
    return sign * (degrees + arcmin / 60.0 + arcsec / 3600.0)


def _extract_center_from_header(header: Any) -> Tuple[float | None, float | None]:
    """Attempt to extract the approximate pointing centre from a FITS header."""

    if header is None:
        return None, None

    def _coerce(value: Any, *, is_ra: bool) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            text = str(value).strip()
        except Exception:
            return None
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            pass
        if ":" in text or any(sep in text for sep in ("h", "m", "°", "'")):
            return _sexagesimal_to_degrees(text, is_ra)
        return None

    ra_keys = ("CRVAL1", "OBJCTRA", "RA", "RA_DEG", "OBJRA", "OBJRA_DEG")
    dec_keys = ("CRVAL2", "OBJCTDEC", "DEC", "DEC_DEG", "OBJDEC", "OBJDEC_DEG")

    ra_value = None
    for key in ra_keys:
        if key in header:
            ra_value = _coerce(header.get(key), is_ra=True)
            if ra_value is not None:
                break

    dec_value = None
    for key in dec_keys:
        if key in header:
            dec_value = _coerce(header.get(key), is_ra=False)
            if dec_value is not None:
                break

    return ra_value, dec_value


class _DirectoryScanWorker(QObject):
    """Background worker resolving missing WCS entries with ASTAP."""

    progress_changed = Signal(int, str)
    row_updated = Signal(int, dict)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        items: List[_NormalizedItem],
        solver_settings: dict | None,
        localizer: Any,
        astap_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self._items = items
        self._solver_settings = solver_settings or {}
        self._localizer = localizer
        self._overrides = astap_overrides or {}
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        total = len(self._items)
        if total == 0:
            self.progress_changed.emit(0, self._localizer.get("filter.scan.empty", "No files to scan."))
            self.finished.emit()
            return

        astap_cfg = self._prepare_astap_configuration()
        if astap_cfg is None:
            message = self._localizer.get(
                "filter.scan.astap_missing",
                "ASTAP configuration is incomplete; skipping WCS solving.",
            )
            self.error.emit(message)

        for index, item in enumerate(self._items):
            if self._stop_requested:
                break
            path = item.file_path
            message = self._localizer.get(
                "filter.scan.inspecting",
                "Inspecting {name}…",
                name=os.path.basename(path) if path else item.display_name,
            )
            self.progress_changed.emit(self._progress_percent(index, total), message)
            row_update: dict[str, Any] = {}
            if not path or not os.path.isfile(path):
                row_update["error"] = self._localizer.get(
                    "filter.scan.missing",
                    "File missing or not accessible.",
                )
                self.row_updated.emit(index, row_update)
                continue

            header = None
            if fits is not None:
                try:
                    header = fits.getheader(path, ignore_missing_end=True)
                except Exception as exc:  # pragma: no cover - passthrough IO error
                    row_update["error"] = str(exc)
            else:
                row_update["error"] = self._localizer.get(
                    "filter.scan.astropy_missing",
                    "Astropy is not installed; cannot inspect headers.",
                )

            has_wcs = _header_has_wcs(header) if header is not None else False
            instrument = _detect_instrument_from_header(header)

            if astap_cfg is not None and not has_wcs and solve_with_astap is not None and header is not None:
                solve_message = self._localizer.get(
                    "filter.scan.solving",
                    "Solving WCS with ASTAP…",
                )
                self.progress_changed.emit(self._progress_percent(index, total), solve_message)

                def _solver_callback(text: str, *_: Any) -> None:
                    try:
                        message_text = str(text)
                    except Exception:
                        message_text = ""
                    if message_text:
                        self.progress_changed.emit(
                            self._progress_percent(index, total),
                            message_text,
                        )

                try:
                    wcs_result = solve_with_astap(
                        path,
                        header,
                        astap_cfg["exe"],
                        astap_cfg["data"],
                        search_radius_deg=astap_cfg["radius"],
                        downsample_factor=astap_cfg["downsample"],
                        sensitivity=astap_cfg["sensitivity"],
                        timeout_sec=astap_cfg["timeout"],
                        update_original_header_in_place=False,
                        progress_callback=_solver_callback,
                    )
                except Exception as exc:  # pragma: no cover - solver failure path
                    row_update["error"] = str(exc)
                else:
                    if wcs_result is not None and getattr(wcs_result, "is_celestial", False):
                        has_wcs = True
                        row_update["solver"] = "ASTAP"

            row_update["has_wcs"] = has_wcs
            if instrument:
                row_update["instrument"] = instrument
            if header is not None:
                ra_deg, dec_deg = _extract_center_from_header(header)
                if ra_deg is not None and dec_deg is not None:
                    row_update["center_ra_deg"] = ra_deg
                    row_update["center_dec_deg"] = dec_deg
            self.row_updated.emit(index, row_update)
            completed_message = self._localizer.get(
                "filter.scan.progress",
                "Processed {done}/{total} files.",
                done=index + 1,
                total=total,
            )
            self.progress_changed.emit(self._progress_percent(index + 1, total), completed_message)

        self.finished.emit()

    def request_stop(self) -> None:
        self._stop_requested = True

    def _prepare_astap_configuration(self) -> dict[str, Any] | None:
        exe_candidates = [
            self._solver_settings.get("astap_executable_path"),
            self._solver_settings.get("astap_executable"),
            self._overrides.get("astap_executable_path"),
        ]
        data_candidates = [
            self._solver_settings.get("astap_data_directory_path"),
            self._solver_settings.get("astap_data_dir"),
            self._overrides.get("astap_data_directory_path"),
        ]
        exe_path = next((str(p) for p in exe_candidates if p), "")
        data_dir = next((str(p) for p in data_candidates if p), "")
        if not exe_path or not os.path.isfile(exe_path):
            return None

        radius = self._coerce_float(
            self._solver_settings.get("astap_search_radius_deg"),
            self._overrides.get("astap_search_radius_deg"),
            default=None,
        )
        downsample = self._coerce_int(
            self._solver_settings.get("astap_downsample"),
            self._overrides.get("astap_downsample"),
            default=None,
        )
        sensitivity = self._coerce_int(
            self._solver_settings.get("astap_sensitivity"),
            self._overrides.get("astap_sensitivity"),
            default=None,
        )
        timeout = self._coerce_int(
            self._solver_settings.get("astap_timeout_sec"),
            self._overrides.get("astap_timeout_sec"),
            default=180,
        )

        if set_astap_max_concurrent_instances is not None:
            concurrency = self._coerce_int(
                self._solver_settings.get("astap_max_instances"),
                self._overrides.get("astap_max_instances"),
                default=None,
            )
            if concurrency:
                try:
                    set_astap_max_concurrent_instances(concurrency)
                except Exception:  # pragma: no cover - runtime guard
                    pass

        return {
            "exe": exe_path,
            "data": data_dir,
            "radius": radius,
            "downsample": downsample,
            "sensitivity": sensitivity,
            "timeout": timeout,
        }

    @staticmethod
    def _progress_percent(current: int, total: int) -> int:
        if total <= 0:
            return 0
        try:
            return min(100, max(0, int((current / total) * 100)))
        except Exception:  # pragma: no cover - defensive
            return 0

    @staticmethod
    def _coerce_float(*values: Any, default: float | None = None) -> float | None:
        for value in values:
            if value in (None, "", False):
                continue
            try:
                return float(value)
            except Exception:
                continue
        return default

    @staticmethod
    def _coerce_int(*values: Any, default: int | None = None) -> int | None:
        for value in values:
            if value in (None, "", False):
                continue
            try:
                return int(value)
            except Exception:
                continue
        return default


class _StreamIngestWorker(QObject):
    """Background worker that yields normalized entries in small batches."""

    batch_ready = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        payload: Any,
        initial_overrides: Any,
        *,
        scan_recursive: bool,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._payload = payload
        self._initial_overrides = initial_overrides
        self._scan_recursive = bool(scan_recursive)
        self._batch_size = max(1, int(batch_size))
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        try:
            iterator = _iter_normalized_entries(
                self._payload,
                self._initial_overrides,
                scan_recursive=self._scan_recursive,
            )
            batch: list[_NormalizedItem] = []
            for entry in iterator:
                if self._stop_requested:
                    break
                batch.append(entry)
                if len(batch) >= self._batch_size:
                    self.batch_ready.emit(batch)
                    batch = []
            if batch:
                self.batch_ready.emit(batch)
        except Exception as exc:  # pragma: no cover - defensive guard
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def request_stop(self) -> None:
        self._stop_requested = True


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
        self._normalized_items: list[_NormalizedItem] = []
        self._stream_thread: QThread | None = None
        self._stream_worker: _StreamIngestWorker | None = None
        self._streaming_active = False
        self._streaming_completed = not self._stream_scan
        self._stream_loaded_count = 0
        try:
            self._batch_size = max(1, int(batch_size))
        except Exception:
            self._batch_size = 100
        self._dialog_button_box: QDialogButtonBox | None = None

        if self._stream_scan:
            self._normalized_items = []
        else:
            self._normalized_items = self._normalize_items(raw_files_with_wcs_or_dir, initial_overrides)

        overrides = config_overrides or {}
        self._auto_group_requested = bool(overrides.get("filter_auto_group", False))
        self._seestar_priority = bool(overrides.get("filter_seestar_priority", False))

        self._table = QTableWidget(self)
        self._summary_label = QLabel(self)
        self._status_label = QLabel(self)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._run_analysis_btn: QPushButton | None = None
        self._scan_thread: QThread | None = None
        self._scan_worker: _DirectoryScanWorker | None = None
        self._auto_group_checkbox = None
        self._seestar_checkbox = None
        self._astap_instances_combo: QComboBox | None = None
        self._astap_instances_value = self._resolve_initial_astap_instances()
        self._preview_canvas: FigureCanvasQTAgg | None = None
        self._preview_axes = None
        self._preview_hint_label = QLabel(self)
        self._preview_default_hint = ""
        self._preview_refresh_pending = False
        self._cluster_groups: list[list[_NormalizedItem]] = []
        self._cluster_threshold_used: float | None = None
        self._cluster_refresh_pending = False
        self._preview_color_cycle: tuple[str, ...] = (
            "#3f7ad6",
            "#d64b3f",
            "#3fd65d",
            "#d6a63f",
            "#8a3fd6",
            "#3fc6d6",
            "#d63fb8",
            "#6ed63f",
        )

        self._preview_canvas = self._create_preview_canvas()
        self._build_ui()
        if self._stream_scan:
            self._prepare_streaming_mode(raw_files_with_wcs_or_dir, initial_overrides)
        else:
            self._populate_table()
            self._update_summary_label()
            self._schedule_preview_refresh()
            self._schedule_cluster_refresh()

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

        return list(
            _iter_normalized_entries(
                payload,
                initial_overrides,
                scan_recursive=self._scan_recursive,
            )
        )

    # ------------------------------------------------------------------
    # UI creation
    # ------------------------------------------------------------------
    def _create_preview_canvas(self) -> FigureCanvasQTAgg | None:
        """Initialise the Matplotlib preview canvas if Matplotlib is available."""

        self._preview_hint_label.setWordWrap(True)
        unavailable_text = self._localizer.get(
            "filter.preview.unavailable",
            "Matplotlib is not available; preview disabled.",
        )
        if Figure is None or FigureCanvasQTAgg is None:
            self._preview_default_hint = unavailable_text
            self._preview_hint_label.setText(unavailable_text)
            return None

        figure = Figure(figsize=(5, 3))
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.add_subplot(111)
        self._preview_axes = axes
        axes.set_xlabel(self._localizer.get("filter.preview.ra", "Right Ascension (°)"))
        axes.set_ylabel(self._localizer.get("filter.preview.dec", "Declination (°)"))
        axes.set_title(self._localizer.get("filter.preview.title", "Sky preview"))
        axes.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        self._preview_default_hint = self._localizer.get(
            "filter.preview.hint",
            "Preview shows the approximate pointing of selected frames (limited by preview cap).",
        )
        self._preview_hint_label.setText(self._preview_default_hint)
        return canvas

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

        self._table.setColumnCount(4)
        headers = [
            self._localizer.get("filter.column.file", "File"),
            self._localizer.get("filter.column.wcs", "WCS"),
            self._localizer.get("filter.column.group", "Group"),
            self._localizer.get("filter.column.instrument", "Instrument"),
        ]
        self._table.setHorizontalHeaderLabels(headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(self._handle_item_changed)

        splitter = QSplitter(Qt.Vertical, self)
        splitter.setChildrenCollapsible(False)

        table_container = QWidget(self)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self._table)
        splitter.addWidget(table_container)

        preview_container = QWidget(self)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_title = QLabel(self._localizer.get("filter.preview.header", "Sky preview"), self)
        preview_layout.addWidget(preview_title)
        if self._preview_canvas is not None:
            preview_layout.addWidget(self._preview_canvas, stretch=1)
        preview_layout.addWidget(self._preview_hint_label)
        splitter.addWidget(preview_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        control_group = QGridLayout()
        self._run_analysis_btn = QPushButton(self._localizer.get("filter.button.run_scan", "Run analysis"), self)
        self._run_analysis_btn.clicked.connect(self._on_run_analysis)  # type: ignore[arg-type]
        control_group.addWidget(self._run_analysis_btn, 0, 0)

        control_row = QGridLayout()
        select_all_btn = QPushButton(self._localizer.get("filter.button.select_all", "Select all"), self)
        select_none_btn = QPushButton(self._localizer.get("filter.button.select_none", "Select none"), self)
        select_all_btn.clicked.connect(self._select_all)  # type: ignore[arg-type]
        select_none_btn.clicked.connect(self._select_none)  # type: ignore[arg-type]
        control_row.addWidget(select_all_btn, 0, 0)
        control_row.addWidget(select_none_btn, 0, 1)
        control_row.addWidget(self._summary_label, 0, 2)
        control_row.setColumnStretch(2, 1)
        control_group.addLayout(control_row, 0, 1, 2, 1)
        progress_row = QGridLayout()
        progress_row.addWidget(self._status_label, 0, 0)
        progress_row.addWidget(self._progress_bar, 0, 1)
        progress_row.setColumnStretch(1, 1)
        control_group.addLayout(progress_row, 1, 0, 1, 2)
        main_layout.addLayout(control_group)

        options_box = self._create_options_box()
        main_layout.addWidget(options_box)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        self._dialog_button_box = button_box

    def _prepare_streaming_mode(self, payload: Any, initial_overrides: Any) -> None:
        """Enable incremental ingestion when the dialog is opened in stream mode."""

        self._stream_loaded_count = 0
        self._streaming_active = True
        self._streaming_completed = False
        self._table.setRowCount(0)
        self._update_summary_label()
        loading_text = self._localizer.get("filter.stream.starting", "Discovering frames…")
        self._status_label.setText(loading_text)
        self._progress_bar.setRange(0, 0)
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(False)
        ok_button = None
        if self._dialog_button_box is not None:
            ok_button = self._dialog_button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setEnabled(False)
        self._start_stream_worker(payload, initial_overrides)

    def _start_stream_worker(self, payload: Any, initial_overrides: Any) -> None:
        self._stop_stream_worker()
        self._stream_worker = _StreamIngestWorker(
            payload,
            initial_overrides,
            scan_recursive=self._scan_recursive,
            batch_size=self._batch_size,
        )
        self._stream_thread = QThread(self)
        self._stream_worker.moveToThread(self._stream_thread)
        self._stream_thread.started.connect(self._stream_worker.run)
        self._stream_worker.batch_ready.connect(self._on_stream_batch)
        self._stream_worker.error.connect(self._on_stream_error)
        self._stream_worker.finished.connect(self._on_stream_finished)
        self._stream_worker.finished.connect(self._stream_thread.quit)
        self._stream_worker.finished.connect(self._stream_worker.deleteLater)
        self._stream_thread.finished.connect(self._on_stream_thread_finished)
        self._stream_thread.finished.connect(self._stream_thread.deleteLater)
        self._stream_thread.start()

    def _on_stream_batch(self, entries: list[_NormalizedItem]) -> None:
        if not entries:
            return
        self._normalized_items.extend(entries)
        self._append_rows(entries)
        self._stream_loaded_count += len(entries)
        message_template = self._localizer.get(
            "filter.stream.loading",
            "Discovered {count} frame(s)…",
            count=self._stream_loaded_count,
        )
        try:
            self._status_label.setText(message_template.format(count=self._stream_loaded_count))
        except Exception:
            self._status_label.setText(f"Discovered {self._stream_loaded_count} frame(s)…")

    def _on_stream_finished(self) -> None:
        self._streaming_active = False
        self._streaming_completed = True
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        total_loaded = len(self._normalized_items)
        message_template = self._localizer.get(
            "filter.stream.completed",
            "Finished loading {count} frame(s).",
            count=total_loaded,
        )
        try:
            self._status_label.setText(message_template.format(count=total_loaded))
        except Exception:
            self._status_label.setText(f"Finished loading {total_loaded} frame(s).")
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(True)
        if self._dialog_button_box is not None:
            ok_button = self._dialog_button_box.button(QDialogButtonBox.Ok)
            if ok_button is not None:
                ok_button.setEnabled(True)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _on_stream_error(self, message: str) -> None:
        if not message:
            return
        try:
            self._status_label.setText(message)
        except Exception:
            pass

    def _on_stream_thread_finished(self) -> None:
        self._stream_thread = None
        self._stream_worker = None

    def _stop_stream_worker(self) -> None:
        worker = self._stream_worker
        thread = self._stream_thread
        was_running = False
        if worker is not None:
            worker.request_stop()
        if thread is not None:
            was_running = thread.isRunning()
            thread.quit()
            thread.wait(2000)
        self._stream_worker = None
        self._stream_thread = None
        self._streaming_active = False
        if was_running:
            self._streaming_completed = False

    def _populate_table(self) -> None:
        items = self._normalized_items
        self._table.blockSignals(True)
        self._table.setRowCount(len(items))
        for row, entry in enumerate(items):
            self._populate_row(row, entry)
        self._table.blockSignals(False)

    def _populate_row(self, row: int, entry: _NormalizedItem) -> None:
        name_item = QTableWidgetItem(entry.display_name)
        name_item.setFlags(name_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        name_item.setCheckState(Qt.Checked if entry.include_by_default else Qt.Unchecked)
        self._table.setItem(row, 0, name_item)

        has_wcs_text = (
            self._localizer.get("filter.value.wcs_present", "Yes")
            if entry.has_wcs
            else self._localizer.get("filter.value.wcs_missing", "No")
        )
        wcs_item = QTableWidgetItem(has_wcs_text)
        wcs_item.setFlags(wcs_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 1, wcs_item)

        group_label = entry.group_label or self._localizer.get("filter.value.group_unassigned", "Unassigned")
        group_item = QTableWidgetItem(group_label)
        group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 2, group_item)

        instrument = entry.instrument or self._localizer.get("filter.value.unknown", "Unknown")
        instrument_item = QTableWidgetItem(instrument)
        instrument_item.setFlags(instrument_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 3, instrument_item)

    def _append_rows(self, entries: Sequence[_NormalizedItem]) -> None:
        if not entries:
            return
        start_row = self._table.rowCount()
        self._table.blockSignals(True)
        self._table.setRowCount(start_row + len(entries))
        for offset, entry in enumerate(entries):
            self._populate_row(start_row + offset, entry)
        self._table.blockSignals(False)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------
    def _handle_item_changed(self, _: QTableWidgetItem) -> None:
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

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
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

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
            summary_formatted = summary_text.format(selected=selected, total=total)
        except Exception:
            summary_formatted = f"{selected} / {total}"

        cluster_count = sum(1 for group in self._cluster_groups if group)
        if cluster_count:
            cluster_fragment = self._localizer.get("filter.summary.groups", "{groups} group(s)")
            try:
                cluster_formatted = cluster_fragment.format(groups=cluster_count)
            except Exception:
                cluster_formatted = f"{cluster_count} group(s)"
            summary_display = f"{summary_formatted} – {cluster_formatted}"
        else:
            summary_display = summary_formatted

        self._summary_label.setText(summary_display)

    def _create_options_box(self):
        from PySide6.QtWidgets import QCheckBox, QGroupBox

        box = QGroupBox(self._localizer.get("filter.group.options", "Filter options"), self)
        layout = QHBoxLayout(box)

        self._auto_group_checkbox = QCheckBox(
            self._localizer.get("filter.option.auto_group", "Auto-group master tiles"),
            box,
        )
        self._auto_group_checkbox.setChecked(self._auto_group_requested)
        layout.addWidget(self._auto_group_checkbox)

        self._seestar_checkbox = QCheckBox(
            self._localizer.get("filter.option.seestar_mode", "Apply Seestar heuristics"),
            box,
        )
        self._seestar_checkbox.setChecked(self._seestar_priority)
        layout.addWidget(self._seestar_checkbox)

        concurrency_label = QLabel(
            self._localizer.get("filter_label_astap_instances", "Max ASTAP instances"),
            box,
        )
        layout.addWidget(concurrency_label)

        self._astap_instances_combo = QComboBox(box)
        self._populate_astap_instances_combo()
        layout.addWidget(self._astap_instances_combo)

        layout.addStretch(1)
        return box

    def _on_run_analysis(self) -> None:
        if self._streaming_active and self._stream_thread is not None and self._stream_thread.isRunning():
            wait_text = self._localizer.get(
                "filter.stream.wait",
                "Please wait for frame discovery to finish before running analysis.",
            )
            self._status_label.setText(wait_text)
            return

        if self._scan_thread is not None and self._scan_thread.isRunning():
            return

        self._status_label.setText(
            self._localizer.get("filter.scan.starting", "Starting analysis…")
        )
        self._progress_bar.setValue(0)
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(False)

        solver_settings = self._solver_settings or {}
        self._scan_worker = _DirectoryScanWorker(
            self._normalized_items,
            solver_settings,
            self._localizer,
            astap_overrides=self._config_overrides,
        )
        self._scan_thread = QThread(self)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.progress_changed.connect(self._on_scan_progress)
        self._scan_worker.row_updated.connect(self._on_scan_row_update)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.start()

    def _resolve_preview_cap(self) -> int | None:
        """Return the integer cap applied to preview points or ``None`` for unlimited."""

        try:
            if self._preview_cap in (None, ""):
                return 200
            cap = int(self._preview_cap)
            if cap <= 0:
                return None
            return cap
        except Exception:
            return 200

    def _ensure_entry_coordinates(self, entry: _NormalizedItem) -> Tuple[float | None, float | None]:
        if entry.center_ra_deg is not None and entry.center_dec_deg is not None:
            return entry.center_ra_deg, entry.center_dec_deg
        if fits is None:
            return None, None
        path = entry.file_path
        if not path or not os.path.isfile(path):
            return None, None
        try:
            header = fits.getheader(path, ignore_missing_end=True)
        except Exception:
            return None, None
        ra_deg, dec_deg = _extract_center_from_header(header)
        entry.center_ra_deg = ra_deg
        entry.center_dec_deg = dec_deg
        return ra_deg, dec_deg

    def _collect_preview_points(self) -> List[Tuple[float, float, int | None]]:
        if self._preview_canvas is None:
            return []
        limit = self._resolve_preview_cap()
        points: list[Tuple[float, float, int | None]] = []
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue
            cluster_idx = entry.cluster_index if isinstance(entry.cluster_index, int) else None
            points.append((ra_deg, dec_deg, cluster_idx))
            if limit is not None and len(points) >= limit:
                break
        return points

    def _schedule_preview_refresh(self) -> None:
        if self._preview_canvas is None or self._preview_refresh_pending:
            return
        self._preview_refresh_pending = True
        QTimer.singleShot(0, self._update_preview_plot)

    def _schedule_cluster_refresh(self) -> None:
        if self._cluster_refresh_pending:
            return
        self._cluster_refresh_pending = True
        QTimer.singleShot(0, self._update_cluster_assignments)

    def _update_cluster_assignments(self) -> None:
        self._cluster_refresh_pending = False
        groups, threshold = self._compute_cluster_groups()
        self._cluster_groups = groups
        self._cluster_threshold_used = threshold

        assigned_ids: set[int] = set()
        label_template = self._localizer.get("filter.group.auto_label", "Group {index}")
        for idx, group in enumerate(groups, start=1):
            try:
                label = label_template.format(index=idx)
            except Exception:
                label = f"Group {idx}"
            for entry in group:
                entry.group_label = label
                entry.cluster_index = idx - 1
                assigned_ids.add(id(entry))

        for entry in self._normalized_items:
            if id(entry) not in assigned_ids:
                entry.group_label = None
                entry.cluster_index = None

        unassigned_text = self._localizer.get("filter.value.group_unassigned", "Unassigned")
        self._table.blockSignals(True)
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 2)
            if item is None:
                continue
            label = entry.group_label or unassigned_text
            item.setText(label)
        self._table.blockSignals(False)

        if groups and (self._scan_thread is None or not self._scan_thread.isRunning()):
            threshold_value = threshold if isinstance(threshold, (int, float)) else 0.0
            summary_template = self._localizer.get(
                "filter.cluster.summary",
                "Auto-grouped {groups} cluster(s) (threshold {threshold:.2f}°)",
            )
            try:
                summary_text = summary_template.format(
                    groups=len(groups),
                    threshold=threshold_value,
                )
            except Exception:
                summary_text = f"Auto-grouped {len(groups)} clusters"
            self._status_label.setText(summary_text)

        self._update_summary_label()
        self._schedule_preview_refresh()

    def _compute_cluster_groups(self) -> Tuple[List[List[_NormalizedItem]], float | None]:
        threshold = self._resolve_cluster_threshold()
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            return [], threshold

        groups: list[list[_NormalizedItem]] = []
        centroids: list[Tuple[float, float]] = []
        for row, entry in enumerate(self._normalized_items):
            item = self._table.item(row, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            ra_deg, dec_deg = entry.center_ra_deg, entry.center_dec_deg
            if ra_deg is None or dec_deg is None:
                ra_deg, dec_deg = self._ensure_entry_coordinates(entry)
            if ra_deg is None or dec_deg is None:
                continue

            assigned_idx = None
            for idx, (cen_ra, cen_dec) in enumerate(centroids):
                distance = self._angular_distance_deg(ra_deg, dec_deg, cen_ra, cen_dec)
                if distance <= threshold:
                    assigned_idx = idx
                    break

            if assigned_idx is None:
                groups.append([entry])
                centroids.append((ra_deg, dec_deg))
            else:
                groups[assigned_idx].append(entry)
                count = len(groups[assigned_idx])
                cen_ra, cen_dec = centroids[assigned_idx]
                new_ra = cen_ra + (ra_deg - cen_ra) / count
                new_dec = cen_dec + (dec_deg - cen_dec) / count
                centroids[assigned_idx] = (new_ra, new_dec)

        non_empty = [group for group in groups if group]
        return non_empty, threshold

    def _resolve_cluster_threshold(self) -> float:
        candidates: tuple[Any, ...] = (
            self._safe_lookup(self._config_overrides, "cluster_panel_threshold"),
            self._safe_lookup(self._initial_overrides, "cluster_panel_threshold"),
            self._safe_lookup(self._solver_settings, "cluster_panel_threshold"),
            self._safe_lookup(self._config_overrides, "panel_clustering_threshold_deg"),
            self._safe_lookup(self._solver_settings, "panel_clustering_threshold_deg"),
        )
        for candidate in candidates:
            try:
                value = float(candidate)
            except Exception:
                continue
            if value > 0:
                return max(0.02, min(5.0, value))
        return 0.25

    @staticmethod
    def _angular_distance_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        dra = abs(float(ra1) - float(ra2)) % 360.0
        if dra > 180.0:
            dra = 360.0 - dra
        ddec = abs(float(dec1) - float(dec2))
        return math.hypot(dra, ddec)

    def _update_preview_plot(self) -> None:
        self._preview_refresh_pending = False
        if self._preview_axes is None or self._preview_canvas is None:
            return

        points = self._collect_preview_points()
        axes = self._preview_axes
        axes.clear()
        axes.set_xlabel(self._localizer.get("filter.preview.ra", "Right Ascension (°)"))
        axes.set_ylabel(self._localizer.get("filter.preview.dec", "Declination (°)"))
        axes.set_title(self._localizer.get("filter.preview.title", "Sky preview"))
        axes.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        if not points:
            message = self._localizer.get(
                "filter.preview.empty",
                "No WCS information available for the current selection.",
            )
            axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
            self._preview_hint_label.setText(self._preview_default_hint)
            self._preview_canvas.draw_idle()
            return

        grouped_points: dict[int | None, list[Tuple[float, float]]] = {}
        for ra_deg, dec_deg, group_idx in points:
            grouped_points.setdefault(group_idx, []).append((ra_deg, dec_deg))

        legend_needed = False
        for group_idx, coords in grouped_points.items():
            ra_coords = [coord[0] for coord in coords]
            dec_coords = [coord[1] for coord in coords]
            if isinstance(group_idx, int):
                color = self._preview_color_cycle[group_idx % len(self._preview_color_cycle)]
                label_template = self._localizer.get("filter.preview.group_label", "Group {index}")
                try:
                    label = label_template.format(index=group_idx + 1)
                except Exception:
                    label = f"Group {group_idx + 1}"
                legend_needed = True
            else:
                color = "#3f7ad6"
                label = None
                if len(grouped_points) > 1:
                    label = self._localizer.get("filter.preview.group_unassigned", "Unassigned")
                    legend_needed = True

            scatter_kwargs = dict(c=color, s=24, alpha=0.85, edgecolors="none")
            if label:
                scatter_kwargs["label"] = label
            axes.scatter(ra_coords, dec_coords, **scatter_kwargs)

        if legend_needed and len(grouped_points) > 1:
            axes.legend(loc="upper right", fontsize="small")

        ra_values = [pt[0] for pt in points]
        dec_values = [pt[1] for pt in points]
        ra_min, ra_max = min(ra_values), max(ra_values)
        dec_min, dec_max = min(dec_values), max(dec_values)
        if ra_min == ra_max:
            ra_margin = max(1.0, abs(ra_min) * 0.01 + 0.5)
        else:
            ra_margin = (ra_max - ra_min) * 0.05
        if dec_min == dec_max:
            dec_margin = max(1.0, abs(dec_min) * 0.01 + 0.5)
        else:
            dec_margin = (dec_max - dec_min) * 0.05
        axes.set_xlim(ra_max + ra_margin, ra_min - ra_margin)
        axes.set_ylim(dec_min - dec_margin, dec_max + dec_margin)
        axes.set_aspect("equal", adjustable="datalim")
        preview_cap_value = self._resolve_preview_cap()
        summary_text = self._localizer.get(
            "filter.preview.summary",
            "Showing {count} frame(s) in preview (cap {cap}).",
            count=len(points),
            cap=preview_cap_value or "∞",
        )
        clusters_present = sum(1 for key in grouped_points.keys() if isinstance(key, int))
        if clusters_present:
            cluster_fragment = self._localizer.get("filter.preview.groups_hint", "{groups} cluster(s)")
            try:
                cluster_formatted = cluster_fragment.format(groups=clusters_present)
            except Exception:
                cluster_formatted = f"{clusters_present} cluster(s)"
            summary_text = f"{summary_text} – {cluster_formatted}"
        try:
            self._preview_hint_label.setText(
                summary_text.format(count=len(points), cap=preview_cap_value or "∞")
            )
        except Exception:
            self._preview_hint_label.setText(summary_text)
        self._preview_canvas.draw_idle()

    # ------------------------------------------------------------------
    # QDialog API
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_stream_worker()
        if self._scan_worker is not None:
            self._scan_worker.request_stop()
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait(2000)
            self._scan_thread = None
        self._scan_worker = None
        super().closeEvent(event)

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

    def _resolved_wcs_count(self) -> int:
        return sum(1 for entry in self._normalized_items if entry.has_wcs)

    def overrides(self) -> Any:
        if self._cluster_refresh_pending:
            self._update_cluster_assignments()

        overrides: dict[str, Any] = {}
        if self._auto_group_checkbox is not None:
            overrides["filter_auto_group"] = bool(self._auto_group_checkbox.isChecked())
        if self._seestar_checkbox is not None:
            overrides["filter_seestar_priority"] = bool(self._seestar_checkbox.isChecked())
        if self._astap_instances_value:
            overrides["astap_max_instances"] = int(self._astap_instances_value)
        if self._cluster_groups and self._auto_group_checkbox is not None and self._auto_group_checkbox.isChecked():
            serialized_groups: list[list[dict[str, Any]]] = []
            for group in self._cluster_groups:
                entries_payload: list[dict[str, Any]] = []
                for entry in group:
                    if not entry.file_path:
                        continue
                    payload = {
                        "path": entry.file_path,
                        "has_wcs": bool(entry.has_wcs),
                    }
                    if entry.instrument:
                        payload["instrument"] = entry.instrument
                    if entry.center_ra_deg is not None and entry.center_dec_deg is not None:
                        payload["RA"] = entry.center_ra_deg
                        payload["DEC"] = entry.center_dec_deg
                    entries_payload.append(payload)
                if entries_payload:
                    serialized_groups.append(entries_payload)
            if serialized_groups:
                overrides["preplan_master_groups"] = serialized_groups
        if isinstance(self._cluster_threshold_used, (int, float)) and self._cluster_threshold_used > 0:
            overrides["cluster_panel_threshold"] = float(self._cluster_threshold_used)
        resolved_count = self._resolved_wcs_count()
        if resolved_count:
            overrides["resolved_wcs_count"] = int(resolved_count)
        return overrides or None

    def input_items(self) -> List[Any]:
        """Return a shallow copy of the original payload list."""

        if self._stream_scan and not self._streaming_completed:
            return [
                entry.original
                for entry in _iter_normalized_entries(
                    self._input_payload,
                    self._initial_overrides,
                    scan_recursive=self._scan_recursive,
                )
            ]
        return [entry.original for entry in self._normalized_items]

    # ------------------------------------------------------------------
    # Scan worker slots
    # ------------------------------------------------------------------
    def _on_scan_progress(self, percent: int, message: str) -> None:
        try:
            self._progress_bar.setValue(percent)
        except Exception:
            pass
        if message:
            try:
                self._status_label.setText(message)
            except Exception:
                pass

    def _on_scan_row_update(self, row: int, payload: dict) -> None:
        if not (0 <= row < len(self._normalized_items)):
            return
        entry = self._normalized_items[row]
        has_wcs = payload.get("has_wcs")
        if isinstance(has_wcs, bool):
            entry.has_wcs = has_wcs
            item = self._table.item(row, 1)
            if item is not None:
                text = (
                    self._localizer.get("filter.value.wcs_present", "Yes")
                    if has_wcs
                    else self._localizer.get("filter.value.wcs_missing", "No")
                )
                item.setText(text)
        instrument = payload.get("instrument")
        if instrument:
            entry.instrument = instrument
            item = self._table.item(row, 3)
            if item is not None:
                item.setText(str(instrument))
        ra_deg = payload.get("center_ra_deg")
        dec_deg = payload.get("center_dec_deg")
        if isinstance(ra_deg, (int, float)) and isinstance(dec_deg, (int, float)):
            entry.center_ra_deg = float(ra_deg)
            entry.center_dec_deg = float(dec_deg)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _on_scan_error(self, message: str) -> None:
        if not message:
            return
        try:
            self._status_label.setText(message)
        except Exception:
            pass

    def _on_scan_finished(self) -> None:
        if self._run_analysis_btn is not None:
            self._run_analysis_btn.setEnabled(True)
        self._scan_thread = None
        self._scan_worker = None
        self._status_label.setText(
            self._localizer.get("filter.scan.completed", "Analysis completed.")
        )
        self._progress_bar.setValue(100)
        self._update_summary_label()
        self._schedule_preview_refresh()
        self._schedule_cluster_refresh()

    def _resolve_initial_astap_instances(self) -> int:
        sources: tuple[Any, ...] = (
            self._safe_lookup(self._config_overrides, "astap_max_instances"),
            self._safe_lookup(self._solver_settings, "astap_max_instances"),
            os.environ.get("ZEMOSAIC_ASTAP_MAX_PROCS"),
        )
        for candidate in sources:
            try:
                if candidate in (None, "", False):
                    continue
                value = int(candidate)
                if value > 0:
                    return max(1, value)
            except Exception:
                continue
        return 1

    @staticmethod
    def _safe_lookup(mapping: Any, key: str) -> Any:
        if isinstance(mapping, dict):
            return mapping.get(key)
        try:
            return getattr(mapping, key)
        except Exception:
            return None

    def _populate_astap_instances_combo(self) -> None:
        if self._astap_instances_combo is None:
            return
        combo = self._astap_instances_combo
        combo.blockSignals(True)
        combo.clear()
        cpu_count = os.cpu_count() or 2
        cap = max(1, cpu_count // 2)
        options = {str(i): i for i in range(1, cap + 1)}
        current = self._astap_instances_value
        options[str(current)] = current
        for text, value in sorted(options.items(), key=lambda item: int(item[0])):
            combo.addItem(text, value)
        index = combo.findData(current)
        if index < 0:
            index = combo.findText(str(current))
        if index < 0:
            combo.addItem(str(current), current)
            index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)
        combo.currentIndexChanged.connect(self._on_astap_instances_changed)  # type: ignore[arg-type]
        self._apply_astap_instances_choice(current, warn=False)

    def _on_astap_instances_changed(self, index: int) -> None:
        if self._astap_instances_combo is None:
            return
        value = self._astap_instances_combo.itemData(index)
        if value is None:
            try:
                value = int(self._astap_instances_combo.currentText())
            except Exception:
                value = self._astap_instances_value
        if not self._apply_astap_instances_choice(int(value), warn=True):
            self._astap_instances_combo.blockSignals(True)
            target_idx = self._astap_instances_combo.findData(self._astap_instances_value)
            if target_idx >= 0:
                self._astap_instances_combo.setCurrentIndex(target_idx)
            self._astap_instances_combo.blockSignals(False)

    def _apply_astap_instances_choice(self, value: int, *, warn: bool) -> bool:
        parsed = max(1, int(value))
        if warn and parsed > 1 and parsed != self._astap_instances_value:
            title = self._localizer.get("filter_astap_multi_warning_title", "ASTAP Concurrency Warning")
            message = self._localizer.get(
                "filter_astap_multi_warning_message",
                (
                    "Running more than one ASTAP instance can trigger the \"Access violation\" "
                    "popup you saw earlier. Only continue if you are ready to dismiss "
                    "those warnings and understand this mode is not officially supported."
                ),
            )
            continue_label = self._localizer.get(
                "filter_astap_multi_warning_continue", "Continue anyway"
            )
            cancel_label = self._localizer.get(
                "filter_astap_multi_warning_cancel", "Cancel"
            )
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(title)
            box.setText(message)
            continue_button = box.addButton(continue_label, QMessageBox.AcceptRole)
            cancel_button = box.addButton(cancel_label, QMessageBox.RejectRole)
            box.setDefaultButton(cancel_button)
            box.exec()
            if box.clickedButton() is not continue_button:
                return False

        self._astap_instances_value = parsed
        if isinstance(self._solver_settings, dict):
            self._solver_settings["astap_max_instances"] = parsed
        if isinstance(self._config_overrides, dict):
            self._config_overrides["astap_max_instances"] = parsed
        self._set_astap_concurrency_runtime(parsed)
        return True

    def _set_astap_concurrency_runtime(self, value: int) -> None:
        try:
            clamped = max(1, int(value))
        except Exception:
            clamped = 1
        os.environ["ZEMOSAIC_ASTAP_MAX_PROCS"] = str(clamped)
        if set_astap_max_concurrent_instances is not None:
            try:
                set_astap_max_concurrent_instances(clamped)
            except Exception:
                pass


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
    """Launch the Qt-based filter dialog and return the user selection.

    Returns a tuple ``(filtered_list, accepted, overrides)`` mirroring the
    semantics of the Tk filter dialog. When the user validates the selection,
    ``filtered_list`` contains the chosen entries and ``overrides`` includes
    any optional tweaks gathered in the dialog. Cancelling the dialog returns
    the original input list, ``accepted`` set to ``False`` and ``overrides``
    set to ``None``.
    """

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
        overrides = dialog.overrides()
        all_items = dialog.input_items()
        accepted = dialog.was_accepted()
    finally:
        if owns_app:
            app.quit()

    if accepted:
        return selected, True, overrides

    return all_items, False, None


__all__ = ["FilterQtDialog", "launch_filter_interface_qt"]
