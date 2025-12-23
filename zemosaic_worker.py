) -> tuple[list[np.ndarray], list[Any], list[np.ndarray | None], list[float]]:
    tile_weights: list[float] = []
    ) -> tuple[np.ndarray | None, Any | None, np.ndarray | None, float]:
                try:
                    tw_val = float(tile_entry[3]) if len(tile_entry) >= 4 else 1.0
                except Exception:
                    tw_val = 1.0
                return arr, twcs, cov, tw_val
        return None, None, None, 1.0
            arr, twcs, cov, tile_weight = _coerce_tile_payload(entry)
            try:
                tw_val = float(tile_weight)
            except Exception:
                tw_val = 1.0
            if not math.isfinite(tw_val) or tw_val <= 0.0:
                tw_val = 1.0
            tile_weights.append(tw_val)
        tile_weights_from_loader: list[float] = []
            if len(loader_payload) >= 4:
                tile_weights_from_loader = list(loader_payload[3] or [])
            try:
                tw_val = float(tile_weights_from_loader[idx]) if idx < len(tile_weights_from_loader) else 1.0
            except Exception:
                tw_val = 1.0
            if not math.isfinite(tw_val) or tw_val <= 0.0:
                tw_val = 1.0
            tile_weights.append(tw_val)
    if len(tile_weights) < len(tiles):
        tile_weights.extend([1.0] * (len(tiles) - len(tile_weights)))
    return tiles, wcs_list, coverage_list, tile_weights
    (
        tiles_for_second_pass,
        wcs_for_second_pass,
        coverage_for_second_pass,
        tile_weights_for_second_pass,
    ) = _prepare_tiles_for_two_pass(
    fallback_used: bool = not bool(collected_tiles) and fallback_tile_loader is not None
            tile_weights=tile_weights_for_second_pass,
            tile_weight_val = 1.0
            if isinstance(entry, dict):
                try:
                    tile_weight_val = float(entry.get("tile_weight", 1.0))
                except Exception:
                    tile_weight_val = 1.0
                if not math.isfinite(tile_weight_val) or tile_weight_val <= 0.0:
                    tile_weight_val = 1.0
            collect_tile_data.append((arr_copy, tile_wcs, cov_copy, tile_weight_val))
        tile_weight_summary_logged = False
            tile_weights_for_entries: list[float] = []
            input_weight_max = 0.0
                    tile_weights_for_entries.append(tw_value)
                            weight_source_base,
                try:
                    local_max = float(np.nanmax(weight2d)) if isinstance(weight2d, np.ndarray) and weight2d.size else 0.0
                except Exception:
                    local_max = 0.0
                input_weight_max = max(input_weight_max, local_max)

            tile_weights_arg: list[float] | None = None
            if tile_weighting_applied and tile_weights_for_entries:
                tile_weights_arg = tile_weights_for_entries
                if not tile_weight_summary_logged:
                    try:
                        weights_arr = np.asarray(tile_weights_for_entries, dtype=np.float64)
                        finite_weights = weights_arr[np.isfinite(weights_arr)]
                        if finite_weights.size:
                            tw_min = float(np.min(finite_weights))
                            tw_med = float(np.median(finite_weights))
                            tw_max = float(np.max(finite_weights))
                            ratio = float(tw_max / tw_min) if tw_min > 0 else float("inf")
                            logger.debug(
                                "[Phase5] tile_weights summary: min=%.4f median=%.4f max=%.4f ratio=%.4f entries=%d",
                                tw_min,
                                tw_med,
                                tw_max,
                                ratio,
                                len(tile_weights_for_entries),
                            )
                        tile_weight_summary_logged = True
                    except Exception:
                        pass
            if tile_weights_arg is not None and input_weight_max > 1.5:
                logger.warning(
                    "[Phase5] possible double weighting: input_weights max=%.3f with tile_weights enabled",
                    input_weight_max,
                )
                if tile_weights_arg is not None:
                    invoke_kwargs["tile_weights"] = tile_weights_arg
    tile_weights: list[float] | None = None,
    def _normalize_tile_weights(weights_obj: list[float] | None, expected: int) -> list[float] | None:
        if weights_obj is None:
            return None
        normalized: list[float] = []
        try:
            iterable = list(weights_obj)
        except Exception:
            iterable = [weights_obj]
        for idx in range(expected):
            try:
                raw = iterable[idx]
            except Exception:
                raw = 1.0
            try:
                val = float(raw)
            except Exception:
                val = 1.0
            if not math.isfinite(val) or val <= 0.0:
                val = 1.0
            normalized.append(val)
        if len(normalized) < expected:
            normalized.extend([1.0] * (expected - len(normalized)))
        return normalized

    tile_weights_norm = _normalize_tile_weights(tile_weights, len(corrected_tiles))
    if tile_weights_norm and logger and logger.isEnabledFor(logging.DEBUG):
        try:
            weights_arr = np.asarray(tile_weights_norm, dtype=np.float64)
            finite_weights = weights_arr[np.isfinite(weights_arr)]
            if finite_weights.size:
                tw_min = float(np.min(finite_weights))
                tw_med = float(np.median(finite_weights))
                tw_max = float(np.max(finite_weights))
                ratio = float(tw_max / tw_min) if tw_min > 0 else float("inf")
                logger.debug(
                    "[TwoPass] tile_weights summary: min=%.4f median=%.4f max=%.4f ratio=%.4f entries=%d",
                    tw_min,
                    tw_med,
                    tw_max,
                    ratio,
                    len(tile_weights_norm),
                )
        except Exception:
            pass

        input_weight_max = 0.0
        if input_weights_list is not None:
            try:
                for weight_entry in input_weights_list:
                    if weight_entry is None:
                        continue
                    weight_arr = np.asarray(weight_entry, dtype=np.float32)
                    local_max = float(np.nanmax(weight_arr)) if weight_arr.size else 0.0
                    input_weight_max = max(input_weight_max, local_max)
            except Exception:
                input_weight_max = 0.0
        if tile_weights_norm is not None:
            local_kwargs["tile_weights"] = tile_weights_norm
        if tile_weights_norm is not None and input_weight_max > 1.5 and logger:
            logger.warning(
                "[TwoPass] possible double weighting: input_weights max=%.3f with tile_weights enabled",
                input_weight_max,
            )
