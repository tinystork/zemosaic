#!/usr/bin/env bash
# Cross-platform PyInstaller build helper for macOS/Linux hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
else
  echo "[build_zemosaic_posix] Virtual environment '.venv' introuvable."
  echo "Créez-la avec 'python3 -m venv .venv' puis relancez le script."
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pyinstaller

pyinstaller ZeMosaic.spec
