"""ZM-ZEGRID-M6 Part D — close-stats diagnostic key contract.

``BoundedAssemblyAccess.close()`` exposes ``peak_live_cells`` /
``peak_live_mapped_bytes``. The historical close log read ``peak_resident_*``
and raised KeyError inside the best-effort handler, printing a warning after a
successful close. This focused test pins the formatter to the real counters.
"""

from __future__ import annotations

import pytest

from zemosaic import zemosaic_zegrid_mode as zz


def test_close_message_reports_real_live_counters():
    message = zz.format_bounded_assembly_close(
        {
            "peak_live_cells": 2,
            "peak_live_mappings": 3,
            "peak_live_mapped_bytes": 4096,
        }
    )
    assert message == (
        "ZeGrid: bounded assembly access closed "
        "(peak 2 mapped cell(s), 4096 B)"
    )


def test_close_message_uses_live_keys_not_the_historical_resident_keys():
    # A dict carrying ONLY the old names must not silently produce a message:
    # the mismatch is what made every successful close emit a warning.
    with pytest.raises(KeyError):
        zz.format_bounded_assembly_close(
            {
                "peak_resident_cells": 2,
                "peak_resident_mapped_bytes": 4096,
            }
        )


def test_close_message_never_leaks_a_counter_object_repr():
    message = zz.format_bounded_assembly_close(
        {"peak_live_cells": 1, "peak_live_mapped_bytes": 0}
    )
    assert "peak 1 mapped cell(s), 0 B" in message
    assert "peak_resident" not in message
