@echo off
REM === Script de compilation ZeMosaic (.exe) ===
REM === ZeMosaic Build Script (.exe) ===

REM Ce script doit être lancé depuis le dossier racine du projet
REM This script must be run from the project root folder

setlocal EnableExtensions EnableDelayedExpansion

REM Usage:
REM   compile\compile_zemosaic._win.bat [onedir|onefile] [debug|release]
REM Examples:
REM   compile\compile_zemosaic._win.bat onedir
REM   compile\compile_zemosaic._win.bat onefile debug

REM Default build mode: onedir (more reliable than onefile for large native deps).
if "%ZEMOSAIC_BUILD_MODE%"=="" set "ZEMOSAIC_BUILD_MODE=onedir"

REM Parse args (optional)
for %%A in (%*) do (
  if /I "%%~A"=="onefile" set "ZEMOSAIC_BUILD_MODE=onefile"
  if /I "%%~A"=="onedir"  set "ZEMOSAIC_BUILD_MODE=onedir"
  if /I "%%~A"=="debug"   set "ZEMOSAIC_DEBUG_BUILD=1"
  if /I "%%~A"=="release" set "ZEMOSAIC_DEBUG_BUILD="
)

REM Onefile workaround: keep extraction path short to avoid WinError 206 (path too long)
REM e.g. Shapely's `shapely.libs` directory under _MEIxxxx can exceed MAX_PATH.
if /I "%ZEMOSAIC_BUILD_MODE%"=="onefile" (
  if "%ZEMOSAIC_RUNTIME_TMPDIR%"=="" set "ZEMOSAIC_RUNTIME_TMPDIR=C:\Temp"
  if not exist "%ZEMOSAIC_RUNTIME_TMPDIR%" mkdir "%ZEMOSAIC_RUNTIME_TMPDIR%" >NUL 2>NUL
)

REM Ensure local venv exists
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create it with: python -m venv .venv
  exit /b 1
)

REM Activate the local virtual environment
call .venv\Scripts\activate

REM Install/update deps (reproducible builds benefit from a clean venv)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib

REM Clean build outputs
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Launch PyInstaller using the .spec file
pyinstaller --noconfirm --clean ZeMosaic.spec

echo.
echo Build mode: %ZEMOSAIC_BUILD_MODE%
if /I "%ZEMOSAIC_BUILD_MODE%"=="onefile" (
  echo Output: dist\ZeMosaic.exe
) else (
  echo Output: dist\ZeMosaic\ZeMosaic.exe
)

REM Pause pour voir les messages en fin de compilation
REM Pause to view final messages after build
pause
