        weight_array_scaled_ids: set[int] = set()
        tile_weight_log_ids: set[str] = set()
        weights_embedded_with_tile = False
            for idx_entry, entry in enumerate(valid_entries):
                weight_source_base = "ones"
                    weight_source_base = "alpha_weight2d"
                                weight_source_base = "coverage_mask"
                weight_source = weight_source_base
                if tile_weighting_applied:
                    try:
                        tw_raw = entry.get("tile_weight", 1.0) if isinstance(entry, dict) else 1.0
                        tw_value = float(tw_raw)
                    except Exception:
                        tw_value = 1.0
                    if not math.isfinite(tw_value) or tw_value <= 0.0:
                        tw_value = 1.0
                    weights_embedded_with_tile = True
                    if isinstance(weight2d, np.ndarray):
                        weight_arr = np.asarray(weight2d, dtype=np.float32, order="C", copy=False)
                        weight_id = id(weight_arr)
                        if weight_id not in weight_array_scaled_ids:
                            weight_array_scaled_ids.add(weight_id)
                            np.multiply(weight_arr, tw_value, out=weight_arr, casting="unsafe")
                        weight2d = weight_arr
                        if isinstance(entry, dict) and weight_source_base == "alpha_weight2d":
                            entry["alpha_weight2d"] = weight_arr
                    weight_source = f"{weight_source_base}*tile_weight"
                    tile_label = (
                        entry.get("tile_id")
                        or entry.get("path")
                        or f"tile_{idx_entry}"
                    )
                    if tile_label not in tile_weight_log_ids:
                        logger.info(
                            "[Phase5] tile_weight applied: tile=%s tw=%.3f weights_source=%s",
                            tile_label,
                            tw_value,
                            weight_source,
                        )
                        tile_weight_log_ids.add(str(tile_label))
                if (
                    tile_weighting_applied
                    and weights_for_entries is not None
                    and not weights_embedded_with_tile
                ):
                return zemosaic_utils.reproject_and_coadd_wrapper(
