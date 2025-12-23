def _maybe_bump_phase5_gpu_rows_per_chunk(
    plan: Any,
    gpu_ctx: Any,
    output_shape_hw: tuple[int, int] | None,
    tiles_total: int,
    logger: logging.Logger | None,
) -> None:
    """Conservatively increase Phase 5 GPU chunk rows when safe mode allows it."""

    try:
        safe_mode = int(getattr(gpu_ctx, "safe_mode", 0) or 0)
    except Exception:
        return

    if safe_mode != 1:
        return

    on_battery = getattr(gpu_ctx, "on_battery", None)
    power_plugged = getattr(gpu_ctx, "power_plugged", None)
    if not ((on_battery is False) or (power_plugged is True)):
        return

    try:
        current_rows = getattr(plan, "gpu_rows_per_chunk", None)
        max_bytes = getattr(plan, "gpu_max_chunk_bytes", None)
    except Exception:
        return

    try:
        out_w = int(output_shape_hw[1]) if output_shape_hw and len(output_shape_hw) > 1 else None
    except Exception:
        out_w = None

    if not current_rows or not max_bytes or not out_w or out_w <= 0:
        return

    bytes_per_row = max(1, out_w * 4 * 2)
    rows_budget = int(max_bytes) // (bytes_per_row * max(1, int(tiles_total or 0)))
    candidate = int(rows_budget)
    if candidate <= current_rows or candidate <= 0:
        return

    new_rows = min(256, max(current_rows, candidate))
    new_rows = max(new_rows, min(96, 256))
    if new_rows <= current_rows:
        return

    try:
        plan.gpu_rows_per_chunk = new_rows
    except Exception:
        return

    if logger:
        try:
            logger.info(
                "Phase5 GPU: bump rows_per_chunk %s -> %s (plugged), max_chunk=%.1fMB, out_w=%s, n_tiles=%s",
                current_rows,
                new_rows,
                float(max_bytes) / (1024.0 ** 2),
                out_w,
                max(1, int(tiles_total or 0)),
            )
        except Exception:
            pass


                _maybe_bump_phase5_gpu_rows_per_chunk(
                    parallel_plan_phase5,
                    gpu_safety_ctx_phase5,
                    final_output_shape_hw,
                    tiles_total_phase5,
                    logger,
                )
