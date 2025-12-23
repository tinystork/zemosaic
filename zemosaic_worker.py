                phase5_chunk_auto_cfg = worker_config_cache.get("phase5_chunk_auto", True)
                phase5_chunk_auto_flag = _coerce_bool_flag(phase5_chunk_auto_cfg)
                if phase5_chunk_auto_flag is None:
                    phase5_chunk_auto_flag = True
                if not phase5_chunk_auto_flag:
                    try:
                        forced_mb = int(worker_config_cache.get("phase5_chunk_mb", 128))
                    except Exception:
                        forced_mb = 128
                    forced_mb = max(64, min(1024, forced_mb))
                    forced_bytes = int(forced_mb) * 1024 * 1024
                    applied_override = False
                    try:
                        setattr(parallel_plan_phase5, "gpu_max_chunk_bytes", forced_bytes)
                        applied_override = True
                    except Exception:
                        pass
                    if isinstance(parallel_plan_phase5, dict):
                        parallel_plan_phase5["gpu_max_chunk_bytes"] = forced_bytes
                        applied_override = True
                    if applied_override:
                        logger.info(
                            "Phase5 GPU chunk override: %d MB (%d bytes) for global_reproject",
                            forced_mb,
                            forced_bytes,
                        )
                    else:
                        logger.debug(
                            "Phase5 GPU chunk override skipped: gpu_max_chunk_bytes not present"
                        )
