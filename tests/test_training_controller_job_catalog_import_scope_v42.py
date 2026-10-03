"""Repository job-catalog import scope regression for the shared controller."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "tools" / "universal_training_controller_job_catalog_v2.py"


def _load_loader():
    current = types.ModuleType("universal_training_controller_current")
    current._ORIGINAL_JOB_RECORDS = lambda root, profile: []
    previous = sys.modules.get("universal_training_controller_current")
    sys.modules["universal_training_controller_current"] = current
    try:
        spec = importlib.util.spec_from_file_location("_catalog_loader_v42_test", LOADER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("universal_training_controller_current", None)
        else:
            sys.modules["universal_training_controller_current"] = previous


class CatalogImportScopeV42Tests(unittest.TestCase):
    def test_sibling_repository_package_imports_work_and_sys_path_is_restored(self) -> None:
        loader = _load_loader()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "science"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "authority.py").write_text(
                "VALUE = 'from-sibling-package'\n", encoding="utf-8"
            )
            catalog = root / "training" / "catalog.py"
            catalog.parent.mkdir()
            catalog.write_text(
                "from science.authority import VALUE\n"
                "def iter_jobs():\n"
                "    return [{'id': VALUE}]\n",
                encoding="utf-8",
            )
            root_text = str(root.resolve())
            while root_text in sys.path:
                sys.path.remove(root_text)
            jobs = loader._load_catalog(root, {
                "path": "training/catalog.py",
                "function": "iter_jobs",
            })
            self.assertEqual([{"id": "from-sibling-package"}], jobs)
            self.assertNotIn(root_text, sys.path)

    def test_failed_catalog_import_restores_path_and_module_cache(self) -> None:
        loader = _load_loader()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = root / "training" / "broken.py"
            catalog.parent.mkdir()
            catalog.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
            root_text = str(root.resolve())
            while root_text in sys.path:
                sys.path.remove(root_text)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                loader._load_catalog(root, {"path": "training/broken.py"})
            self.assertNotIn(root_text, sys.path)
            self.assertNotIn("_training_control_catalog_broken", sys.modules)


if __name__ == "__main__":
    unittest.main()
