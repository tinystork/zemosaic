#!/usr/bin/env bash
set -euo pipefail

ZEMOSAIC_ROOT="/home/tristan/zemosaic"
PY="$ZEMOSAIC_ROOT/.venv/bin/python"
WORKER="$ZEMOSAIC_ROOT/zemosaic_worker.py"
INPUT_DIR="/home/tristan/zemosaic/example/bench50_new_auto"
PROFILES_DIR="/home/tristan/zemosaic/benchmarks/bench50_matrix/profiles"
OUT_ROOT_BASE="/home/tristan/zemosaic/example/out_bench50"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="$OUT_ROOT_BASE/$STAMP"

mkdir -p "$OUT_ROOT"

run_case() {
  local profile_name="$1"
  local case_name="$2"
  local profile_path="$PROFILES_DIR/$profile_name"
  local out_dir="$OUT_ROOT/$case_name"
  mkdir -p "$out_dir"

  echo "=== CASE: $case_name ($profile_name) ==="
  echo "Output: $out_dir"
  (
    cd "$ZEMOSAIC_ROOT"
    /usr/bin/time -v "$PY" "$WORKER" "$INPUT_DIR" "$out_dir" --config "$profile_path"
  ) >"$out_dir/console.log" 2>&1
  echo "DONE: $case_name"
}

run_case "01_cpu_safe.json" "cpu_safe"
run_case "02_hybrid_balanced.json" "hybrid_balanced"
run_case "03_gpu_push.json" "gpu_push"

echo "All bench50 runs completed. Root: $OUT_ROOT"
