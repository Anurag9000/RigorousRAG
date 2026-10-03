"""Regression for immutable-controller loading of the physical training catalog."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "training" / "authoritative_training_suite_catalog_v2.py"


class PhysicalTrainingCatalogImportTest(unittest.TestCase):
    def test_catalog_imports_from_isolated_non_repository_cwd(self) -> None:
        code = (
            "import importlib.util, pathlib; "
            f"path=pathlib.Path({str(CATALOG)!r}); "
            "spec=importlib.util.spec_from_file_location('_isolated_catalog_v2', path); "
            "module=importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "assert module.SCHEMA == 'rigorousrag-authoritative-training-suite/v2-dataset-cohorts'"
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=temporary,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
