    cpu_workers: int | None = None,
    try:
        cpu_workers = int(cpu_workers) if cpu_workers is not None else None
    except Exception:
        cpu_workers = None
    if cpu_workers is not None and cpu_workers < 1:
        cpu_workers = 1

            cpu_workers=cpu_workers,
                cpu_workers=(
                    int(processing_threads)
                    if processing_threads and int(processing_threads) > 0
                    else min(os.cpu_count() or 1, 8)
                ),
                cpu_workers=(
                    int(assembly_process_workers)
                    if assembly_process_workers and int(assembly_process_workers) > 0
                    else min(os.cpu_count() or 1, 8)
                ),
