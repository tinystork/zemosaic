from __future__ import annotations

from pathlib import Path

import grid_mode


def test_load_stack_plan_resolves_windows_paths_against_input_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    file_a = input_dir / "Light_A.fit"
    file_b = input_dir / "Light_B.fit"
    file_a.write_bytes(b"dummy")
    file_b.write_bytes(b"dummy")

    csv_path = input_dir / "stack_plan.csv"
    csv_path.write_text(
        "file_path,exposure\n"
        "D:\\ASTRO\\project\\Light_A.fit,10\n"
        "D:\\ASTRO\\project\\Light_B.fit,20\n",
        encoding="utf-8",
    )

    frames = grid_mode.load_stack_plan(csv_path)

    assert len(frames) == 2
    assert frames[0].path == file_a
    assert frames[1].path == file_b
    assert all(frame.path.is_file() for frame in frames)
