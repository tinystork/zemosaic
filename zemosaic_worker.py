    tile_weights: list[float] | None = None,
            tile_weights=tile_weights,
        tile_weights_for_sources: list[float] = []
            weight_val = 1.0
                try:
                    weight_val = float(entry.get("tile_weight", 1.0))
                except Exception:
                    weight_val = 1.0
                if not math.isfinite(weight_val) or weight_val <= 0:
                    weight_val = 1.0
            tile_weights_for_sources.append(weight_val)
                tile_weights=tile_weights_for_sources,
