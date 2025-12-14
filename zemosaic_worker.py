            corrected_tiles = 0
            gain_min = None
            gain_max = None
            offset_min = None
            offset_max = None
            for tile_entry in effective_tiles:
                tile_id = tile_entry.get("tile_id")
                if not tile_id or tile_entry.get("data") is None:
                try:
                    gain_to_apply = float(gain_val)
                    offset_to_apply = float(offset_val)
                    if match_bg:
                        gain_before = gain_to_apply
                        offset_before = offset_to_apply
                        if gain_to_apply < gain_limit_min:
                            gain_to_apply = gain_limit_min
                        elif gain_to_apply > gain_limit_max:
                            gain_to_apply = gain_limit_max
                        if affine_offset_limit_adu > 0.0:
                            if abs(offset_to_apply) > affine_offset_limit_adu:
                                offset_to_apply = 0.0
                            else:
                                offset_to_apply = max(-affine_offset_limit_adu, min(offset_to_apply, affine_offset_limit_adu))
                        if gain_to_apply != gain_before or offset_to_apply != offset_before:
                            try:
                                _pcb(
                                    "assemble_warn_affine_clamped",
                                    prog=None,
                                    lvl="INFO_DETAIL",
                                    tile_id=tile_id,
                                    gain_before=gain_before,
                                    gain_after=gain_to_apply,
                                    offset_before=offset_before,
                                    offset_after=offset_to_apply,
                                )
                            except Exception:
                                pass
                    arr_obj = tile_entry["data"]
                    try:
                        import cupy as cp  # type: ignore

                        if isinstance(arr_obj, cp.ndarray):
                            arr_cp = cp.asarray(arr_obj, dtype=cp.float32)
                            if gain_to_apply != 1.0:
                                cp.multiply(arr_cp, gain_to_apply, out=arr_cp, casting="unsafe")
                            if offset_to_apply != 0.0:
                                cp.add(arr_cp, offset_to_apply, out=arr_cp, casting="unsafe")
                            tile_entry["data"] = arr_cp
                        else:
                            arr_np = np.asarray(arr_obj, dtype=np.float32, order="C")
                            if gain_to_apply != 1.0:
                                np.multiply(arr_np, gain_to_apply, out=arr_np, casting="unsafe")
                            if offset_to_apply != 0.0:
                                np.add(arr_np, offset_to_apply, out=arr_np, casting="unsafe")
                            tile_entry["data"] = arr_np
                    except Exception:
                        arr_np = np.asarray(arr_obj, dtype=np.float32, order="C")
                        if gain_to_apply != 1.0:
                            np.multiply(arr_np, gain_to_apply, out=arr_np, casting="unsafe")
                        if offset_to_apply != 0.0:
                            np.add(arr_np, offset_to_apply, out=arr_np, casting="unsafe")
                        tile_entry["data"] = arr_np
                    corrected_tiles += 1
                    gain_min = gain_to_apply if gain_min is None else min(gain_min, gain_to_apply)
                    gain_max = gain_to_apply if gain_max is None else max(gain_max, gain_to_apply)
                    offset_min = offset_to_apply if offset_min is None else min(offset_min, offset_to_apply)
                    offset_max = offset_to_apply if offset_max is None else max(offset_max, offset_to_apply)
                    logger.info(
                        "apply_photometric: tile=%s gain=%.5f offset=%.5f",
                        tile_id,
                        gain_to_apply,
                        offset_to_apply,
                    )
                except Exception:
                    continue
            if corrected_tiles:
                try:
                    _pcb(
                        "assemble_info_intertile_photometric_applied",
                        prog=None,
                        lvl="INFO_DETAIL",
                        num_tiles=corrected_tiles,
                    )
                except Exception:
                    pass
                    "apply_photometric summary: corrected_tiles=%d gain[min=%.5f max=%.5f] offset[min=%.5f max=%.5f]",
                    corrected_tiles,
                    gain_min if gain_min is not None else 0.0,
                    gain_max if gain_max is not None else 0.0,
                    offset_min if offset_min is not None else 0.0,
                    offset_max if offset_max is not None else 0.0,
            else:
                nontrivial_affine = False
                pending_affine_list = None
