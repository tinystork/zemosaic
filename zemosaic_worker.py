class _TwoPassDiagnostics:
    def __init__(self, logger: logging.Logger | None, *, downsample_factor: int = 8):
        self.logger = logger
        self.enabled = logger is not None and logger.isEnabledFor(logging.DEBUG)
        self.downsample_factor = max(1, int(downsample_factor))
        self.base_ready = False
        self.base_mask: np.ndarray | None = None
        self.base_mosaic_lr: np.ndarray | None = None
        self.base_cov_lr: np.ndarray | None = None
        self.lowres_wcs: Any | None = None
        self.total_pixels: int = 0
        self.coverage_threshold: float = 0.0
        self.coverage_reject_warned = False
        self.mask_mismatch_warned = False
        self.nan_reproj_warned = False
        self.overlap_pre: dict[int, dict[str, float]] = {}
        self.overlap_post: dict[int, dict[str, float]] = {}
        self.tile_stats: dict[int, dict[str, float]] = {}

    def _log(self, level: int, msg: str, *args) -> None:
        if not self.enabled or self.logger is None:
        try:
            self.logger.log(level, msg, *args)
        except Exception:
            pass
    def _to_luminance(self, arr: Any, gain: float = 1.0) -> np.ndarray | None:
        try:
            data = np.asarray(arr, dtype=np.float32) * float(gain)
        except Exception:
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            channels_last = data
            if data.shape[-1] not in (1, 3) and data.shape[0] in (1, 3, 4):
                try:
                    channels_last = np.moveaxis(data, 0, -1)
                except Exception:
                    channels_last = data
            channel_count = channels_last.shape[-1]
            if channel_count == 1:
                return channels_last[..., 0]
            if channel_count >= 3:
                return (
                    0.25 * channels_last[..., 0]
                    + 0.5 * channels_last[..., 1]
                    + 0.25 * channels_last[..., 2]
                )
        return None
    @staticmethod
    def _downsample_2d(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray | None:
        try:
            arr2 = np.asarray(arr, dtype=np.float32)
        except Exception:
            return None
        arr2 = np.squeeze(arr2)
        if arr2.ndim != 2:
            return None
        if arr2.shape == target_hw:
            return arr2
        try:
            import cv2  # type: ignore
            return cv2.resize(
                np.ascontiguousarray(arr2),
                (target_hw[1], target_hw[0]),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            src_h, src_w = arr2.shape
            y_coords = np.linspace(0, max(src_h - 1, 0), target_hw[0])
            x_coords = np.linspace(0, max(src_w - 1, 0), target_hw[1])
            y_idx = np.clip(y_coords.astype(np.int64), 0, src_h - 1)
            x_idx = np.clip(x_coords.astype(np.int64), 0, src_w - 1)
            return arr2[np.ix_(y_idx, x_idx)]

    @staticmethod
    def _make_lowres_wcs(src_wcs: Any, factor: float) -> Any:
        if src_wcs is None:
            return None
        try:
            lowres = copy.deepcopy(src_wcs)
        except Exception:
            return None
        try:
            wcs_obj = getattr(lowres, "wcs", None)
            if wcs_obj is None:
            if getattr(wcs_obj, "cd", None) is not None:
                cd = np.array(wcs_obj.cd, copy=True)
                if cd.shape[0] >= 2 and cd.shape[1] >= 2:
                    cd[:2, :2] *= factor
                    wcs_obj.cd = cd
            elif getattr(wcs_obj, "cdelt", None) is not None:
                cdelt = np.array(wcs_obj.cdelt, copy=True)
                if cdelt.size >= 2:
                    cdelt[:2] *= factor
                    wcs_obj.cdelt = cdelt
            crpix = getattr(wcs_obj, "crpix", None)
            if crpix is not None:
                crpix_arr = np.array(crpix, dtype=np.float64, copy=True)
                if crpix_arr.size >= 2:
                    crpix_arr[:2] = (crpix_arr[:2] + (factor - 1.0) / 2.0) / factor
                    wcs_obj.crpix = crpix_arr
            return lowres
        except Exception:
            return None
    def prepare_base(self, mosaic_data: np.ndarray | None, coverage_hw: np.ndarray | None, output_wcs: Any) -> None:
        if not self.enabled:
            return
        if mosaic_data is None or coverage_hw is None or output_wcs is None:
            return
            coverage_arr = np.asarray(coverage_hw, dtype=np.float32)
        mosaic_lum = self._to_luminance(mosaic_data)
        target_h = max(1, int(coverage_arr.shape[0] // self.downsample_factor))
        target_w = max(1, int(coverage_arr.shape[1] // self.downsample_factor))
        mosaic_lr = self._downsample_2d(mosaic_lum, target_hw)
        coverage_lr = self._downsample_2d(coverage_arr, target_hw)
        base_mask = np.isfinite(mosaic_lr) & np.isfinite(coverage_lr) & (coverage_lr > self.coverage_threshold)
        lowres_wcs = self._make_lowres_wcs(output_wcs, float(self.downsample_factor)) or output_wcs
        self.base_ready = True
        self.base_mask = base_mask
        self.base_mosaic_lr = mosaic_lr
        self.base_cov_lr = coverage_lr
        self.lowres_wcs = lowres_wcs
        self.total_pixels = total_pixels

        coverage_positive = np.count_nonzero(coverage_lr > self.coverage_threshold)
        coverage_reject_frac = 1.0 - (coverage_positive / float(total_pixels)) if total_pixels else 0.0
        if coverage_reject_frac > 0.2 and not self.coverage_reject_warned:
            self.coverage_reject_warned = True
            self._log(
                logging.WARNING,
                "[TwoPassSanity] coverage mask rejects %.2f%% of pixels (threshold>%.1e)",
                coverage_reject_frac * 100.0,
                self.coverage_threshold,
            )

    def log_global_context(
        self,
        *,
        sigma_px: int,
        gain_clip: tuple[float, float],
        tile_count: int,
        output_shape: tuple[int, int] | None,
        mosaic_dtype: Any,
        fallback_used: bool,
    ) -> None:
        if not self.enabled:
            return
        self._log(
            logging.DEBUG,
            "[TwoPassCfg] sigma=%s clip_min=%.3f clip_max=%.3f tiles=%d output_shape=%s dtype=%s fallback_used=%s DS=%d",
            sigma_px,
            gain_clip[0],
            gain_clip[1],
            tile_count,
            output_shape,
            mosaic_dtype,
            fallback_used,
            self.downsample_factor,
        )

    def log_coverage_stats(self, coverage_hw: np.ndarray | None) -> None:
        if not self.enabled or coverage_hw is None:
            return
        try:
            cov_arr = np.asarray(coverage_hw, dtype=np.float32)
        except Exception:
            return
        cov_arr = np.squeeze(cov_arr)
        if cov_arr.ndim != 2:
            return
        finite_cov = np.nan_to_num(cov_arr, nan=0.0, posinf=0.0, neginf=0.0)
        positive_mask = finite_cov > self.coverage_threshold
        positive_coords = np.argwhere(positive_mask)
        bbox = None
        if positive_coords.size:
            y_min, x_min = np.min(positive_coords, axis=0)
            y_max, x_max = np.max(positive_coords, axis=0)
            bbox = (int(y_min), int(x_min), int(y_max), int(x_max))
        frac_positive = positive_mask.sum() / float(finite_cov.size) if finite_cov.size else 0.0
        self._log(
            logging.DEBUG,
            "[TwoPassCoverage] min=%.4f mean=%.4f median=%.4f max=%.4f frac_positive=%.4f bbox=%s threshold=%.1e",
            float(np.nanmin(finite_cov)) if finite_cov.size else float("nan"),
            float(np.nanmean(finite_cov)) if finite_cov.size else float("nan"),
            float(np.nanmedian(finite_cov)) if finite_cov.size else float("nan"),
            float(np.nanmax(finite_cov)) if finite_cov.size else float("nan"),
            frac_positive,
            bbox,
            self.coverage_threshold,
        )

    def log_tile_stats(
        self,
        tiles: list[np.ndarray],
        coverages: list[np.ndarray | None] | None,
    ) -> None:
        if not self.enabled:
            return
        coverages = coverages or []
        for idx, tile in enumerate(tiles):
            if tile is None:
            tile_arr = np.asarray(tile, dtype=np.float32)
            if tile_arr.ndim == 2:
                tile_arr = tile_arr[..., np.newaxis]
            cov_arr = coverages[idx] if idx < len(coverages) else None
            coverage_mask = None
            if cov_arr is not None:
                try:
                    coverage_mask = np.asarray(cov_arr, dtype=np.float32)
                    coverage_mask = np.squeeze(coverage_mask)
                    if coverage_mask.shape != tile_arr.shape[:2]:
                        coverage_mask = coverage_mask.reshape(tile_arr.shape[:2])
                except Exception:
                    coverage_mask = None
                if coverage_mask is None and not self.mask_mismatch_warned:
                    self.mask_mismatch_warned = True
                    self._log(logging.WARNING, "[TwoPassSanity] coverage mask shape mismatch for tile %d", idx)
            if coverage_mask is not None:
                coverage_mask = coverage_mask > self.coverage_threshold
            rgb_channels = tile_arr[..., : min(3, tile_arr.shape[-1])]
            finite_mask = np.all(np.isfinite(rgb_channels), axis=-1)
            if tile_arr.shape[-1] >= 4:
                alpha_mask = tile_arr[..., 3] > 0
                finite_mask &= alpha_mask
            if coverage_mask is not None:
                finite_mask &= coverage_mask
            total_px = int(finite_mask.size)
            valid_px = int(np.count_nonzero(finite_mask))
            valid_frac = valid_px / float(total_px) if total_px else 0.0
            if valid_px <= 0:
            flat = rgb_channels[finite_mask]
            med = np.median(flat, axis=0)
            mad = np.median(np.abs(flat - med), axis=0)
            mean = np.mean(flat, axis=0)
            stats = {
                "valid_frac": float(valid_frac),
                "median_r": float(med[0]) if med.size >= 1 else float("nan"),
                "median_g": float(med[1]) if med.size >= 2 else float("nan"),
                "median_b": float(med[2]) if med.size >= 3 else float("nan"),
                "mad_r": float(mad[0]) if mad.size >= 1 else float("nan"),
                "mad_g": float(mad[1]) if mad.size >= 2 else float("nan"),
                "mad_b": float(mad[2]) if mad.size >= 3 else float("nan"),
                "mean_r": float(mean[0]) if mean.size >= 1 else float("nan"),
                "mean_g": float(mean[1]) if mean.size >= 2 else float("nan"),
                "mean_b": float(mean[2]) if mean.size >= 3 else float("nan"),
            }
            self.tile_stats[idx] = stats
            self._log(
                logging.DEBUG,
                "[TwoPassTileStats] idx=%d valid_frac=%.4f median=(%.4f,%.4f,%.4f) mad=(%.4f,%.4f,%.4f) mean=(%.4f,%.4f,%.4f)",
                idx,
                stats["valid_frac"],
                stats["median_r"],
                stats["median_g"],
                stats["median_b"],
                stats["mad_r"],
                stats["mad_g"],
                stats["mad_b"],
                stats["mean_r"],
                stats["mean_g"],
                stats["mean_b"],
            )

    def _regression(self, ref_vals: np.ndarray, tile_vals: np.ndarray) -> tuple[float, float, float]:
        slope = float("nan")
        intercept = float("nan")
        corr = float("nan")
        try:
            x = np.asarray(ref_vals, dtype=np.float64)
            y = np.asarray(tile_vals, dtype=np.float64)
            if x.size >= 2 and y.size >= 2:
                x_mean = float(np.nanmean(x))
                y_mean = float(np.nanmean(y))
                x_centered = x - x_mean
                y_centered = y - y_mean
                var_x = float(np.nanmean(x_centered * x_centered))
                var_y = float(np.nanmean(y_centered * y_centered))
                cov_xy = float(np.nanmean(x_centered * y_centered))
                if var_x > 0:
                    slope = cov_xy / var_x
                    intercept = y_mean - slope * x_mean
                if var_x > 0 and var_y > 0:
                    corr = cov_xy / (math.sqrt(var_x) * math.sqrt(var_y) + 1e-12)
        except Exception:
            pass
        return slope, intercept, corr

    def log_overlap(
        self,
        *,
        idx: int,
        tile_data: np.ndarray,
        tile_wcs: Any,
        gain: float = 1.0,
        stage: str = "pre",
    ) -> None:
        if not self.enabled or not self.base_ready or self.base_mosaic_lr is None or self.base_mask is None:
            return
        if not (REPROJECT_AVAILABLE and reproject_interp):
            return
        if tile_data is None or tile_wcs is None:
            return
        tile_lum = self._to_luminance(tile_data, gain)
        if tile_lum is None:
            return
        try:
            tile_proj, footprint = reproject_interp(
                (tile_lum, tile_wcs),
                self.lowres_wcs,
                shape_out=self.base_mosaic_lr.shape,
                return_footprint=True,
            )
        except Exception:
            self._log(logging.DEBUG, "[TwoPassOverlap] idx=%d reprojection failed", idx)
            return
        tile_proj_arr = np.asarray(tile_proj, dtype=np.float32)
        overlap_mask = self.base_mask & np.isfinite(tile_proj_arr)
        if footprint is not None:
                footprint_arr = np.asarray(footprint, dtype=np.float32)
                overlap_mask &= footprint_arr > 0
            except Exception:
                pass
        valid_count = int(np.count_nonzero(overlap_mask))
        if self.total_pixels > 0:
            nan_frac = 1.0 - (np.count_nonzero(np.isfinite(tile_proj_arr)) / float(self.total_pixels))
            if nan_frac > 0.05 and not self.nan_reproj_warned:
                self.nan_reproj_warned = True
                self._log(logging.WARNING, "[TwoPassSanity] reprojection NaN fraction %.2f%%", nan_frac * 100.0)
        overlap_frac = valid_count / float(self.total_pixels) if self.total_pixels else 0.0
        if valid_count <= 0:
            return
        ref_vals = self.base_mosaic_lr[overlap_mask]
        tile_vals = tile_proj_arr[overlap_mask]
        delta = tile_vals - ref_vals
        delta_med = float(np.nanmedian(delta))
        abs_delta_med = float(np.nanmedian(np.abs(delta)))
        delta_mad = float(np.nanmedian(np.abs(delta - delta_med)))
        slope, intercept, corr = self._regression(ref_vals, tile_vals)
        record = {
            "overlap_frac": float(overlap_frac),
            "delta_med": delta_med,
            "abs_delta_med": abs_delta_med,
            "delta_mad": delta_mad,
            "slope": slope,
            "intercept": intercept,
            "corr": corr,
        }
        if stage == "pre":
            self.overlap_pre[idx] = record
        else:
            self.overlap_post[idx] = record
        self._log(
            logging.DEBUG,
            "[TwoPassOverlap] idx=%d overlap_frac=%.4f delta_med=%.5f abs_delta_med=%.5f delta_mad=%.5f slope=%.5f intercept=%.5f corr=%.5f stage=%s",
            idx,
            overlap_frac,
            delta_med,
            abs_delta_med,
            delta_mad,
            slope,
            intercept,
            corr,
            stage,
        )
        if stage == "post":
            pre = self.overlap_pre.get(idx)
            if pre:
                self._log(
                    logging.DEBUG,
                    "[TwoPassApply] idx=%d delta_med_pre=%.5f delta_med_post=%.5f",
                    idx,
                    pre.get("delta_med", float("nan")),
                    delta_med,

    def emit_summary(self) -> None:
        if not self.enabled:
            return
        source = self.overlap_post or self.overlap_pre
        if not source:
            return
        top = sorted(source.items(), key=lambda kv: abs(kv[1].get("delta_med", 0.0)), reverse=True)[:5]
        for idx, stats in top:
            self._log(
                logging.DEBUG,
                "[TwoPassWorst] idx=%d overlap_frac=%.4f abs_delta_med=%.5f delta_med=%.5f slope=%.5f intercept=%.5f",
                stats.get("overlap_frac", float("nan")),
                stats.get("abs_delta_med", float("nan")),
                stats.get("delta_med", float("nan")),
                stats.get("slope", float("nan")),
                stats.get("intercept", float("nan")),
        overlaps = [v.get("overlap_frac", 0.0) for v in source.values()]
        delta_abs = [v.get("abs_delta_med", float("nan")) for v in source.values()]
        weights = np.array(overlaps, dtype=np.float64)
        deltas = np.array(delta_abs, dtype=np.float64)
        valid_mask = np.isfinite(deltas) & np.isfinite(weights)
        weighted = float(np.sum(deltas[valid_mask] * weights[valid_mask]) / max(np.sum(weights[valid_mask]), 1e-12)) if valid_mask.any() else float("nan")
        global_overlap_mean = float(np.nanmean(overlaps)) if overlaps else float("nan")
        self._log(
            logging.DEBUG,
            "[TwoPassScore] global_overlap_mean=%.5f global_abs_delta_med=%.5f",
            global_overlap_mean,
            weighted,
        )


def _apply_two_pass_coverage_renorm_if_requested(
    final_mosaic_data: np.ndarray | None,
    final_mosaic_coverage: np.ndarray | None,
    *,
    two_pass_enabled: bool,
    two_pass_sigma_px: int,
    gain_clip_tuple: tuple[float, float],
    final_output_wcs: Any,
    final_output_shape_hw: tuple[int, int] | None,
    use_gpu_two_pass: bool,
    logger: logging.Logger | None,
    collected_tiles: list[tuple[np.ndarray, Any]] | None = None,
    fallback_tile_loader: Callable[[], tuple[list[np.ndarray], list[Any]]] | None = None,
    parallel_plan: ParallelPlan | None = None,
    telemetry_ctrl: ResourceTelemetryController | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run the coverage renormalization second pass if configured."""

    diagnostics = _TwoPassDiagnostics(logger)
    fallback_used = not bool(collected_tiles) and fallback_tile_loader is not None
    if diagnostics.enabled:
        diagnostics.prepare_base(final_mosaic_data, final_mosaic_coverage, final_output_wcs)

    if diagnostics.enabled:
        diagnostics.log_global_context(
            sigma_px=two_pass_sigma_px,
            gain_clip=gain_clip_tuple,
            tile_count=len(tiles_for_second_pass),
            output_shape=final_output_shape_hw,
            mosaic_dtype=getattr(final_mosaic_data, "dtype", None),
            fallback_used=fallback_used,
        )
        diagnostics.log_coverage_stats(final_mosaic_coverage)
        diagnostics.log_tile_stats(tiles_for_second_pass, coverage_for_second_pass)
        try:
            for idx, (tile, twcs) in enumerate(zip(tiles_for_second_pass, wcs_for_second_pass)):
                diagnostics.log_overlap(idx=idx, tile_data=tile, tile_wcs=twcs, stage="pre")
        except Exception:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[TwoPassOverlap] diagnostic failed", exc_info=True)
            diag_recorder=diagnostics,
    if diagnostics.enabled:
        diagnostics.emit_summary()
    diag_recorder: _TwoPassDiagnostics | None = None,
    if diag_recorder is not None and diag_recorder.enabled:
        try:
            for idx, (tile, twcs) in enumerate(zip(tiles, tiles_wcs)):
                gain_val = gains[idx] if idx < len(gains) else 1.0
                diag_recorder.log_overlap(
                    idx=idx,
                    tile_data=tile,
                    tile_wcs=twcs,
                    gain=gain_val,
                    stage="post",
                )
        except Exception:
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[TwoPassApply] post-overlap diagnostics failed", exc_info=True)
