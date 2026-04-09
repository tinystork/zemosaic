#!/usr/bin/env bash
# Cross-platform PyInstaller build helper for macOS/Linux hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REQUIREMENTS_FILE="${ZEMOSAIC_REQUIREMENTS_FILE:-requirements.txt}"

if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
else
  echo "[build_zemosaic_posix] Virtual environment '.venv' introuvable."
  echo "Créez-la avec 'python3 -m venv .venv' puis relancez le script."
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "[build_zemosaic_posix] Fichier d'exigences introuvable: ${REQUIREMENTS_FILE}"
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r "${REQUIREMENTS_FILE}"
python3 -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib

if [[ -n "${ZEMOSAIC_CUPY_PKG:-}" ]]; then
  echo "[build_zemosaic_posix] Installing CuPy package: ${ZEMOSAIC_CUPY_PKG}"
  python3 -m pip install "${ZEMOSAIC_CUPY_PKG}"
else
  echo "[build_zemosaic_posix] No extra CuPy package specified; GPU support depends on ${REQUIREMENTS_FILE}."
fi

rm -rf build dist
pyinstaller --noconfirm --clean ZeMosaic.spec
