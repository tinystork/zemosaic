"""Packaging-readiness tests for the ZeMosaic src-layout milestone (ZM-PKG-IMPL-01).

These tests are stdlib-only (``unittest``) so they run without pytest and only
require a Python interpreter with the package importable via ``src/``.  They
cover metadata/entry point, namespace hygiene, CWD independence, resource
resolution, config migration, log placement, and CPU-only import behavior.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_python(code: str, *, cwd: Path, env_extra: dict | None = None, timeout: int = 180):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class MetadataTests(unittest.TestCase):
    def test_pyproject_exists_and_parses(self):
        if tomllib is None:
            self.skipTest("tomllib unavailable")
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIn("project", data)
        self.assertEqual(data["project"]["name"], "ZeMosaic")
        self.assertIn("gui-scripts", data["project"])
        self.assertEqual(
            data["project"]["gui-scripts"]["zemosaic"], "zemosaic._app:main"
        )

    def test_entry_point_callable_exists(self):
        from zemosaic._app import main

        self.assertTrue(callable(main))

    def test_version_single_source(self):
        if tomllib is None:
            self.skipTest("tomllib unavailable")
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        dynamic = data["project"].get("dynamic", [])
        self.assertIn("version", dynamic)
        attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        self.assertEqual(attr, "zemosaic.__version__")

        import zemosaic

        self.assertIsInstance(zemosaic.__version__, str)
        self.assertTrue(zemosaic.__version__)

    def test_cupy_not_a_base_dependency(self):
        if tomllib is None:
            self.skipTest("tomllib unavailable")
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        base = data["project"]["dependencies"]
        for dep in base:
            self.assertNotIn("cupy", dep.lower())
        extras = data["project"]["optional-dependencies"]
        self.assertTrue(any("cupy" in d.lower() for d in extras.get("gpu", [])))

    def test_package_data_declared(self):
        if tomllib is None:
            self.skipTest("tomllib unavailable")
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        pkg_data = data["tool"]["setuptools"]["package-data"]["zemosaic"]
        joined = "\n".join(pkg_data)
        self.assertIn("locales", joined)
        self.assertIn("icon", joined)
        self.assertIn("gif", joined)


class NamespaceTests(unittest.TestCase):
    def test_no_flat_global_aliases(self):
        import zemosaic  # noqa: F401

        # The legacy flat module names must not leak into sys.modules as
        # top-level aliases when the package is imported.
        for legacy in ("zemosaic_utils", "zemosaic_config", "zemosaic_worker"):
            self.assertNotIn(legacy, sys.modules)

    def test_import_of_flat_name_fails(self):
        import importlib

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("zemosaic_utils")

    def test_package_lives_under_src(self):
        import zemosaic

        pkg_file = Path(zemosaic.__file__).resolve()
        self.assertTrue(
            str(pkg_file).startswith(str(SRC.resolve())),
            f"package resolved outside src: {pkg_file}",
        )


class ResourceTests(unittest.TestCase):
    def test_resources_accessible(self):
        from zemosaic._resources import resource_path

        locales = resource_path("locales", "en.json")
        icon = resource_path("icon", "zemosaic.ico")
        gif = resource_path("gif", "opening.gif")
        self.assertTrue(locales.is_file(), locales)
        self.assertTrue(icon.is_file(), icon)
        self.assertTrue(gif.is_file(), gif)
        # Must resolve inside the package tree (not CWD-dependent).
        self.assertTrue(str(locales.resolve()).startswith(str(SRC.resolve())))

    def test_cwd_independent_import_and_resource(self):
        with tempfile.TemporaryDirectory() as td:
            code = (
                "import os\n"
                "from pathlib import Path\n"
                "import zemosaic\n"
                "from zemosaic._resources import resource_path\n"
                "p = resource_path('locales', 'en.json')\n"
                "assert p.is_file(), p\n"
                "assert 'zemosaic' in str(Path(zemosaic.__file__).resolve()), zemosaic.__file__\n"
                "print('OK')\n"
            )
            proc = run_python(code, cwd=Path(td))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("OK", proc.stdout)


class ConfigTests(unittest.TestCase):
    def test_config_outside_package_and_migration(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cwd = td_path / "checkout"
            xdg = td_path / "xdg"
            cwd.mkdir()
            xdg.mkdir()
            legacy = cwd / "zemosaic_config.json"
            legacy.write_text(
                json.dumps({"language": "fr", "output_dir": "/legacy/out"}), encoding="utf-8"
            )

            code = (
                "import json\n"
                "from pathlib import Path\n"
                "import zemosaic.zemosaic_config as zc\n"
                "p = Path(zc.get_config_path())\n"
                "print('CONFIG=' + str(p))\n"
                "data = json.loads(p.read_text(encoding='utf-8'))\n"
                "assert data.get('language') == 'fr', data\n"
                "assert '/legacy/out' in data.get('output_dir', ''), data\n"
                "print('MIGRATED_OK')\n"
            )
            env = {"XDG_CONFIG_HOME": str(xdg)}
            proc = run_python(code, cwd=cwd, env_extra=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("MIGRATED_OK", proc.stdout)

            config_line = next(
                (ln for ln in proc.stdout.splitlines() if ln.startswith("CONFIG=")), ""
            )
            config_path = Path(config_line.split("=", 1)[1])
            self.assertTrue(str(config_path).startswith(str(xdg.resolve())), config_path)
            # Must not live inside the installed/source package tree.
            self.assertNotIn(str(SRC.resolve()), str(config_path.resolve()))

    def test_migration_never_overwrites_user_config(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cwd = td_path / "checkout"
            xdg = td_path / "xdg"
            cwd.mkdir()
            xdg.mkdir()
            (cwd / "zemosaic_config.json").write_text(
                json.dumps({"language": "fr"}), encoding="utf-8"
            )
            # Pre-create the user config with a different value.
            user_config = xdg / "ZeMosaic" / "zemosaic_config.json"
            user_config.parent.mkdir(parents=True, exist_ok=True)
            user_config.write_text(json.dumps({"language": "de"}), encoding="utf-8")

            code = (
                "import json\n"
                "from pathlib import Path\n"
                "import zemosaic.zemosaic_config as zc\n"
                "p = Path(zc.get_config_path())\n"
                "data = json.loads(p.read_text(encoding='utf-8'))\n"
                "assert data.get('language') == 'de', data\n"
                "print('NO_OVERWRITE_OK')\n"
            )
            proc = run_python(code, cwd=cwd, env_extra={"XDG_CONFIG_HOME": str(xdg)})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("NO_OVERWRITE_OK", proc.stdout)


class LogTests(unittest.TestCase):
    def test_worker_log_outside_package(self):
        with tempfile.TemporaryDirectory() as td:
            xdg = Path(td) / "xdg"
            xdg.mkdir()
            code = (
                "from pathlib import Path\n"
                "import sys\n"
                "import zemosaic.zemosaic_worker as zw\n"
                "p = Path(zw._resolve_worker_log_file_path())\n"
                "print('LOG=' + str(p))\n"
                "src = Path(zw.__file__).resolve().parent\n"
                "assert not str(p.resolve()).startswith(str(src)), p\n"
                "assert 'site-packages' not in str(p.resolve()), p\n"
                "print('WORKER_LOG_OK')\n"
            )
            proc = run_python(code, cwd=Path(td), env_extra={"XDG_CONFIG_HOME": str(xdg)})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("WORKER_LOG_OK", proc.stdout)

    def test_filter_log_path_is_user_scoped_structural(self):
        src_text = (SRC / "zemosaic" / "zemosaic_filter_gui_qt.py").read_text(
            encoding="utf-8"
        )
        # The filter log must no longer be written next to the module file.
        self.assertNotIn('with_name("zemosaic_filter.log")', src_text)
        self.assertIn("_filter_log_path()", src_text)


class CpuOnlyImportTests(unittest.TestCase):
    def test_import_does_not_pull_cupy(self):
        code = (
            "import importlib.util\n"
            "import sys\n"
            "import zemosaic\n"
            "import zemosaic.cuda_utils as cu\n"
            "assert 'cupy' not in sys.modules, sorted(k for k in sys.modules if k.startswith('cupy'))\n"
            "assert isinstance(cu.CUPY_AVAILABLE, bool), cu.CUPY_AVAILABLE\n"
            "assert cu.CUPY_AVAILABLE == (importlib.util.find_spec('cupy') is not None)\n"
            "print('CUPY_IMPORT_OK')\n"
        )
        proc = run_python(code, cwd=REPO)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("CUPY_IMPORT_OK", proc.stdout)


class LauncherTests(unittest.TestCase):
    def test_launcher_is_thin_delegation(self):
        text = (REPO / "run_zemosaic.py").read_text(encoding="utf-8")
        self.assertIn("from zemosaic._app import main", text)
        self.assertIn("main()", text)
        # No bootstrap duplication: must not import the Qt module directly.
        self.assertNotIn("zemosaic_gui_qt", text)
        # Must not sys.path-hack the parent directory.
        self.assertNotIn("_parent_dir", text)
        self.assertNotIn("(_SRC, _ROOT)", text)

    def test_launcher_delegates(self):
        code = (
            "import run_zemosaic\n"
            "from zemosaic._app import main as real_main\n"
            "assert run_zemosaic.main is real_main\n"
            "print('LAUNCHER_OK')\n"
        )
        proc = run_python(code, cwd=REPO)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LAUNCHER_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
