import logging

import grid_mode


def test_emit_uses_worker_logger(caplog):
    message = "Logging smoke test"
    with caplog.at_level(logging.INFO, logger="ZeMosaicWorker"):
        grid_mode._emit(message)

    records = [rec for rec in caplog.records if rec.name.startswith("ZeMosaicWorker")]
    assert records, "No logs captured on ZeMosaicWorker logger"
    assert any("[GRID]" in rec.getMessage() and message in rec.getMessage() for rec in records)
