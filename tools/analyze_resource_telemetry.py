from __future__ import annotations

"""
Quick, dependency-light telemetry analyzer for resource_telemetry.csv.

Summarizes CPU/GPU load and chunking knobs per phase so we can calibrate the
parallel heuristics without pulling in heavy deps like pandas.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping


WARN_CPU_MAX = 90.0  # percent
WARN_RAM_FRACTION = 0.90
WARN_GPU_FRACTION = 0.90
NUMERIC_FIELDS = (
    "cpu_workers",
    "rows_per_chunk",
    "gpu_rows_per_chunk",
    "max_chunk_bytes",
    "gpu_max_chunk_bytes",
)


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _default_phase_row() -> Dict[str, object]:
    return {
        "cpu_sum": 0.0,
        "cpu_count": 0,
        "cpu_max": 0.0,
        "gpu_sum": 0.0,
        "gpu_count": 0,
        "gpu_max": 0.0,
        "ram_ratio_max": 0.0,
        "field_sum": {field: 0.0 for field in NUMERIC_FIELDS},
        "field_count": {field: 0 for field in NUMERIC_FIELDS},
    }


def _accumulate_phase(stats: Dict[str, Dict[str, object]], row: Mapping[str, object]) -> None:
    name = (row.get("phase_name") or "unknown").strip() or "unknown"
    phase = stats[name]

    cpu_percent = _safe_float(row.get("cpu_percent"))
    if cpu_percent is not None:
        phase["cpu_sum"] += cpu_percent
        phase["cpu_count"] += 1
        phase["cpu_max"] = max(phase["cpu_max"], cpu_percent)

    ram_used = _safe_float(row.get("ram_used_mb"))
    ram_total = _safe_float(row.get("ram_total_mb"))
    if ram_used is not None and ram_total and ram_total > 0:
        ram_fraction = ram_used / ram_total
        phase["ram_ratio_max"] = max(phase["ram_ratio_max"], ram_fraction)

    gpu_used = _safe_float(row.get("gpu_used_mb"))
    gpu_total = _safe_float(row.get("gpu_total_mb"))
    if gpu_total and gpu_total > 0 and gpu_used is not None and gpu_used >= 0:
        gpu_fraction = gpu_used / gpu_total
        phase["gpu_sum"] += gpu_fraction
        phase["gpu_count"] += 1
        phase["gpu_max"] = max(phase["gpu_max"], gpu_fraction)

    for field in NUMERIC_FIELDS:
        value = _safe_float(row.get(field))
        if value is None or value <= 0:
            continue
        phase["field_sum"][field] += value
        phase["field_count"][field] += 1


def _format_mean(total: float, count: int) -> str:
    if count <= 0:
        return "n/a"
    return f"{total / count:.2f}"


def _format_field(field: str, total: float, count: int) -> str:
    if count <= 0:
        return "-"
    mean = total / count
    if "bytes" in field:
        return f"{mean / (1024 ** 2):.1f} MB"
    if mean.is_integer():
        return f"{int(mean)}"
    return f"{mean:.2f}"


def _iter_csv_rows(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def analyze(path: Path) -> None:
    stats: Dict[str, Dict[str, object]] = defaultdict(_default_phase_row)
    for row in _iter_csv_rows(path):
        _accumulate_phase(stats, row)

    warnings: list[str] = []
    for phase_name in sorted(stats.keys()):
        data = stats[phase_name]
        cpu_count = int(data["cpu_count"])
        gpu_count = int(data["gpu_count"])
        print(f"\nPhase: {phase_name}")
        print(f"  CPU% mean={_format_mean(data['cpu_sum'], cpu_count)} max={data['cpu_max']:.1f} (n={cpu_count})")
        if gpu_count > 0:
            gpu_mean_pct = float(data["gpu_sum"]) / gpu_count * 100.0
            gpu_max_pct = float(data["gpu_max"]) * 100.0
            print(f"  GPU used/total: mean={gpu_mean_pct:.1f}% max={gpu_max_pct:.1f}% (n={gpu_count})")
        else:
            print("  GPU used/total: n/a")

        ram_fraction = float(data["ram_ratio_max"])
        print(f"  RAM fraction max: {ram_fraction*100.0:.1f}%")

        field_sum: Dict[str, float] = data["field_sum"]  # type: ignore[assignment]
        field_count: Dict[str, int] = data["field_count"]  # type: ignore[assignment]
        field_lines = []
        for field in NUMERIC_FIELDS:
            field_lines.append(f"{field}={_format_field(field, field_sum[field], field_count[field])}")
        print(f"  Means (non-zero): {', '.join(field_lines)}")

        if data["cpu_max"] > WARN_CPU_MAX:
            warnings.append(f"{phase_name}: CPU max {data['cpu_max']:.1f}% exceeds {WARN_CPU_MAX}%")
        if ram_fraction > WARN_RAM_FRACTION:
            warnings.append(f"{phase_name}: RAM max {ram_fraction*100.0:.1f}% exceeds {WARN_RAM_FRACTION*100:.0f}%")
        if data["gpu_max"] > WARN_GPU_FRACTION:
            warnings.append(
                f"{phase_name}: GPU VRAM max {data['gpu_max']*100.0:.1f}% exceeds {WARN_GPU_FRACTION*100:.0f}%"
            )

    if warnings:
        print("\nWarnings:")
        for line in warnings:
            print(f"- {line}")
    else:
        print("\nNo saturation warnings detected.")


def _resolve_csv_path(argv: list[str]) -> Path:
    if argv:
        candidate = Path(argv[0]).expanduser()
        if candidate.is_file():
            return candidate
        print(f"Provided path not found: {candidate}", file=sys.stderr)
        sys.exit(1)

    default = Path(__file__).resolve().parents[1] / "resource_telemetry.csv"
    if default.is_file():
        return default
    print("resource_telemetry.csv not found; provide a path explicitly.", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    csv_path = _resolve_csv_path(argv)
    print(f"Analyzing telemetry from: {csv_path}")
    analyze(csv_path)


if __name__ == "__main__":
    main()
