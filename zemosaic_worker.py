            winsor_max_workers_val = 0 if worker_limit_val <= 0 else max(1, worker_limit_val)
                "winsor_max_workers": winsor_max_workers_val,
                "[P4.5][G%03d] Stack params: reject=%s, combine=%s, kappa=%.2f, winsor_limits=%s, workers=%s, weight=%s",
                "AUTO(0)" if winsor_max_workers_val <= 0 else str(stack_kwargs["winsor_max_workers"]),
            )
        try:
            winsor_worker_limit_cfg = int(winsor_worker_limit_config)
        except Exception:
            winsor_worker_limit_cfg = 1
        winsor_auto = winsor_worker_limit_cfg <= 0
        if winsor_auto:
            candidate = 0
            if global_parallel_plan and getattr(global_parallel_plan, "cpu_workers", 0) > 0:
                candidate = int(getattr(global_parallel_plan, "cpu_workers", 0))
            elif effective_base_workers and effective_base_workers > 0:
                candidate = int(effective_base_workers)
            else:
                candidate = cpu_total
        else:
            candidate = winsor_worker_limit_cfg
        winsor_worker_limit = max(1, min(int(candidate), cpu_total))
        if winsor_auto:
            pcb(
                f"Winsor worker limit: AUTO (cfg={winsor_worker_limit_cfg}) -> resolved={winsor_worker_limit} (cpu_total={cpu_total})",
                prog=None,
                lvl="INFO_DETAIL",
            )
    try:
        winsor_worker_limit_cfg = int(winsor_worker_limit_config)
    except Exception:
        winsor_worker_limit_cfg = 1
    winsor_auto = winsor_worker_limit_cfg <= 0
    if winsor_auto:
        candidate = 0
        if global_parallel_plan and getattr(global_parallel_plan, "cpu_workers", 0) > 0:
            candidate = int(getattr(global_parallel_plan, "cpu_workers", 0))
        elif effective_base_workers and effective_base_workers > 0:
            candidate = int(effective_base_workers)
        else:
            candidate = cpu_total
    else:
        candidate = winsor_worker_limit_cfg
    winsor_worker_limit = max(1, min(int(candidate), cpu_total))
    if winsor_auto:
        pcb(
            f"Winsor worker limit: AUTO (cfg={winsor_worker_limit_cfg}) -> resolved={winsor_worker_limit} (cpu_total={cpu_total})",
            prog=None,
            lvl="INFO_DETAIL",
        )
