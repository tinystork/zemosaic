    alpha_debug_logged = False
        alpha_weight2d: np.ndarray | None = None
                valid2d = alpha_mask_arr > ALPHA_OPACITY_THRESHOLD
                alpha_weight2d = valid2d.astype(np.float32, copy=False)
                data = np.asarray(data, dtype=np.float32, order="C", copy=True)
                if data.ndim == 3:
                    data[~valid2d, :] = 0.0
                else:
                    data[~valid2d] = 0.0
                if not alpha_debug_logged:
                    try:
                        alpha_min = float(np.nanmin(alpha_mask_arr)) if alpha_mask_arr.size else 0.0
                        alpha_max = float(np.nanmax(alpha_mask_arr)) if alpha_mask_arr.size else 0.0
                        valid_frac = float(np.mean(valid2d)) if valid2d.size else 0.0
                        _pcb(
                            "[Alpha] mask stats",
                            prog=None,
                            lvl="INFO_DETAIL",
                            alpha_min=f"{alpha_min:.3f}",
                            alpha_max=f"{alpha_max:.3f}",
                            valid_frac=f"{valid_frac:.3f}",
                            weight_shape=str(alpha_weight2d.shape),
                            data_shape=str(data.shape[:2]),
                        )
                    except Exception:
                        pass
                    alpha_debug_logged = True
        else:
            alpha_weight2d = None
        if alpha_weight2d is not None:
            coverage_mask_entry = np.asarray(alpha_weight2d, dtype=np.float32, order="C")
        if alpha_weight2d is not None:
            tile_entry["alpha_weight2d"] = alpha_weight2d
    alpha_weights_present = any(
        isinstance(entry, dict) and entry.get("alpha_weight2d") is not None for entry in effective_tiles
    )
    if use_gpu and alpha_weights_present:
        use_gpu = False
        try:
            _pcb(
                "[Alpha] Per-pixel alpha weights require CPU coadd; forcing CPU for Phase 5",
                prog=None,
                lvl="INFO_DETAIL",
            )
        except Exception:
            logger.info("[Alpha] Per-pixel alpha weights require CPU coadd; forcing CPU for Phase 5")

            data_list = []
            wcs_list = []
            input_weights_list = []
            for entry in valid_entries:
                entry_data = entry.get("data")
                data_plane = entry_data[..., ch]
                data_list.append(data_plane)
                wcs_list.append(entry.get("wcs"))
                weight2d = entry.get("alpha_weight2d") if isinstance(entry, dict) else None
                if weight2d is not None:
                    input_weights_list.append(weight2d)
                else:
                    input_weights_list.append(np.ones_like(data_plane, dtype=np.float32))
            reproj_call_kwargs["input_weights"] = input_weights_list
