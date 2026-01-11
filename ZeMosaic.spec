# -*- mode: python ; coding: utf-8 -*-

import os
import importlib.util
from pathlib import Path

block_cipher = None

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()
DEBUG_BUILD = os.environ.get("ZEMOSAIC_DEBUG_BUILD") == "1"
BUILD_MODE = os.environ.get("ZEMOSAIC_BUILD_MODE", "onedir").strip().lower()
ONEFILE = BUILD_MODE == "onefile"

datas = [
    (str(PROJECT_ROOT / 'locales' / '*.json'), 'locales'),
    (str(PROJECT_ROOT / 'icon' / '*.ico'), 'icon'),
    (str(PROJECT_ROOT / 'icon' / '*.png'), 'icon'),
    (str(PROJECT_ROOT / 'gif' / '*.gif'), 'gif'),
]
binaries = []

try:
    import photutils
except ModuleNotFoundError:
    photutils = None

if photutils:
    _photutils_path = Path(photutils.__file__).resolve().parent / 'CITATION.rst'
    if _photutils_path.exists():
        datas.append((str(_photutils_path), 'photutils'))

hiddenimports = [
    'reproject',
    'locales.zemosaic_localization',
    'solver_settings',
    'zemosaic_config',
    'lecropper',
    'zemosaic_worker',
    'zemosaic_astrometry',
    'zemosaic_filter_gui',
    'zemosaic_filter_gui_qt',
    'zemosaic_gui_qt',
]

# Optional preview plots in the Qt filter UI
if importlib.util.find_spec("matplotlib") is not None:
    hiddenimports.extend(
        [
            "matplotlib",
            "matplotlib.pyplot",
            "matplotlib.figure",
            "matplotlib.backends.backend_agg",
            "matplotlib.backends.backend_qtagg",
            "matplotlib.backends.backend_qt",
            "matplotlib.backends.qt_compat",
            "matplotlib.backends.backend_tkagg",
            "matplotlib.backends._backend_tk",
            "matplotlib.backends._tkagg",
        ]
    )
    try:
        from PyInstaller.utils.hooks import collect_data_files
    except Exception:
        collect_data_files = None  # type: ignore[assignment]
    if collect_data_files is not None:
        datas += collect_data_files("matplotlib")

# Optional GPU acceleration via CuPy (cupy-cuda12x installs the `cupy` module)
if importlib.util.find_spec("cupy") is not None:
    hiddenimports.extend(["cupy", "cupy_backends", "cupyx"])
    try:
        from PyInstaller.utils.hooks import collect_all
    except Exception:
        collect_all = None  # type: ignore[assignment]
    if collect_all is not None:
        for pkg in ("cupy", "cupy_backends", "cupyx"):
            try:
                pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
                datas += pkg_datas
                binaries += pkg_binaries
                hiddenimports += pkg_hiddenimports
            except Exception:
                # Keep CuPy optional: if collection fails (missing driver on build host,
                # partial install, etc.), the app will still build and fall back to CPU.
                pass

# Shapely bundles native DLLs under `shapely.libs` (delvewheel). Collect them
# explicitly to avoid missing runtime binaries (especially in onedir builds).
if importlib.util.find_spec("shapely") is not None:
    try:
        from PyInstaller.utils.hooks import collect_dynamic_libs
    except Exception:
        collect_dynamic_libs = None  # type: ignore[assignment]
    if collect_dynamic_libs is not None:
        binaries += collect_dynamic_libs("shapely")

a = Analysis(
    ['run_zemosaic.py'],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT.parent)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PROJECT_ROOT / "pyinstaller_hooks")],
    runtime_hooks=[str(PROJECT_ROOT / "pyinstaller_hooks" / "rthook_zemosaic_sys_path.py")],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='ZeMosaic',
        debug=DEBUG_BUILD,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=os.environ.get("ZEMOSAIC_RUNTIME_TMPDIR") or None,
        console=DEBUG_BUILD,
        icon=str(PROJECT_ROOT / 'icon' / 'zemosaic.ico')
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='ZeMosaic',
        debug=DEBUG_BUILD,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=DEBUG_BUILD,
        icon=str(PROJECT_ROOT / 'icon' / 'zemosaic.ico')
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='ZeMosaic',
    )
