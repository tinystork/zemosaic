    try:
        _best_effort_anchor_photometry()
    except Exception as anchor_exc:
        logger.info(
            "existing_master_tiles_mode: best-effort anchor failed; continuing without anchor",
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
