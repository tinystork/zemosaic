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
CUPY_REQUIRED = os.environ.get("ZEMOSAIC_REQUIRE_CUPY") == "1"

datas = [
    (str(PROJECT_ROOT / 'locales' / '*.json'), 'locales'),
    (str(PROJECT_ROOT / 'icon' / '*.ico'), 'icon'),
    (str(PROJECT_ROOT / 'icon' / '*.png'), 'icon'),
    (str(PROJECT_ROOT / 'gif' / '*.gif'), 'gif'),
]
binaries = []

try:
    import photutils
except Exception:
    photutils = None

if photutils:
    _photutils_path = Path(photutils.__file__).resolve().parent / 'CITATION.rst'
    if _photutils_path.exists():
        datas.append((str(_photutils_path), 'photutils'))

_version_stub = PROJECT_ROOT / "_version.py"
if _version_stub.exists():
    datas.append((str(_version_stub), '.'))

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
_cupy_spec = importlib.util.find_spec("cupy")
if _cupy_spec is None:
    if CUPY_REQUIRED:
        raise SystemExit(
            "CuPy not installed in build environment. Install the correct package "
            "(e.g. cupy-cuda12x) or unset ZEMOSAIC_REQUIRE_CUPY."
        )
    print("[WARN] CuPy not installed in build env; GPU support will be disabled in the build.")
else:
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

# Optional dependency used by astroalign (star detection). The sep package ships
# a compiled extension (sep_pjw) that imports a top-level _version module at runtime.
# Ensure both the pure-Python wrapper and _version are bundled when available.
_sep_spec = importlib.util.find_spec("sep")
if _sep_spec is not None:
    hiddenimports.extend(["sep", "sep_pjw"])
    _sep_version_spec = importlib.util.find_spec("_version")
    if _sep_version_spec is not None:
        hiddenimports.append("_version")
    try:
        from PyInstaller.utils.hooks import collect_all
    except Exception:
        collect_all = None  # type: ignore[assignment]
    if collect_all is not None:
        try:
            pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all("sep")
            datas += pkg_datas
            binaries += pkg_binaries
            hiddenimports += pkg_hiddenimports
        except Exception:
            pass
    # CuPy depends on fastrlock (C-extension). Ensure it is bundled explicitly.
    _fastrlock_spec = importlib.util.find_spec("fastrlock")
    if _fastrlock_spec is None:
        if CUPY_REQUIRED:
            raise SystemExit(
                "CuPy requires 'fastrlock' but it is missing. Reinstall CuPy or "
                "install fastrlock in the build environment."
            )
        print("[WARN] fastrlock missing; CuPy import may fail at runtime.")
    else:
        hiddenimports.append("fastrlock")
        hiddenimports.append("fastrlock._fastrlock")
        if collect_all is not None:
            try:
                pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all("fastrlock")
                datas += pkg_datas
                binaries += pkg_binaries
                hiddenimports += pkg_hiddenimports
            except Exception:
                pass

# Shapely native DLL collection is handled by `pyinstaller_hooks/hook-shapely.py`.

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

# Windows: Shapely wheels ship GEOS DLLs under `shapely.libs/` and call
# `os.add_dll_directory()` at import time (delvewheel patch). In frozen apps this
# can crash with WinError 206 depending on path resolution (subst drives / long paths).
#
# Workaround: relocate the DLLs next to Shapely's extension modules (`shapely/`).
# The Windows loader will resolve them from the extension module directory.
if os.name == "nt":
    relocated = []
    for entry in list(getattr(a, "binaries", []) or []):
        try:
            dest, src, kind = entry
        except Exception:
            relocated.append(entry)
            continue
        normalized_dest = str(dest).replace("/", "\\")
        if normalized_dest.lower().startswith("shapely.libs\\"):
            relocated.append((str(Path("shapely") / Path(normalized_dest).name), src, kind))
        else:
            relocated.append(entry)
    a.binaries = relocated

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
