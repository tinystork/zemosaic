# Bench matrix 50 images (ZeMosaic)

Input folder prepared:
- `/home/tristan/zemosaic/example/bench50_new_auto`
- 50 symlinks from `/home/tristan/near_auto_compare100/new_auto_proxy`
- manifest: `/home/tristan/zemosaic/example/bench50_new_auto/bench50_manifest.json`

Profiles:
- `profiles/01_cpu_safe.json`
- `profiles/02_hybrid_balanced.json`
- `profiles/03_gpu_push.json`

Run all profiles:
```bash
bash /home/tristan/zemosaic/benchmarks/bench50_matrix/scripts/run_bench50_matrix.sh
```

Output root:
- `/home/tristan/zemosaic/example/out_bench50/<timestamp>/...`
