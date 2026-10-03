"""Regression tests for v42 scientific/workload scanner scope.

These tests exercise only static controller logic. They do not execute training
or claim runtime/CUDA parity.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

_current = types.ModuleType("universal_training_controller_current")
_current._enhanced_coverage_report = lambda root, profile, jobs: {"coverage_ok": True}
_current._reachability = lambda root, jobs: {
    "reachable_sources": [],
    "executed_sources": [],
}
sys.modules.setdefault("universal_training_controller_current", _current)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

registry_members = importlib.import_module("universal_training_controller_registry_member_closure")
workload = importlib.import_module("universal_training_controller_workload_closure")
scientific = importlib.import_module("universal_training_controller_scientific_surface_v25")
selectors = importlib.import_module("universal_training_controller_selector_closure_v26")
declarations = importlib.import_module("universal_training_controller_declaration_closure_v30")


class ScannerScopeV42Tests(unittest.TestCase):
    def test_module_registries_remain_contractual_but_runtime_locals_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "training" / "fixture.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "DATASETS = {'alpha': 1, 'beta': 2}\n"
                "MODELS = build_models()\n"
                "def train_one():\n"
                "    dataset = load_dataset()\n"
                "    policy = {'runtime': True}\n"
                "    encoder = make_encoder()\n"
                "    return dataset, policy, encoder\n",
                encoding="utf-8",
            )
            v25 = scientific._scientific_registry_findings(source, "training/fixture.py")
            self.assertEqual({"DATASETS", "MODELS"}, {row["symbol"] for row in v25})
            models = next(row for row in v25 if row["symbol"] == "MODELS")
            self.assertFalse(models["enumerable"])
            self.assertEqual([], models["members"])

            v30 = declarations._python_findings(source, "training/fixture.py")
            symbols = {row["symbol"] for row in v30}
            self.assertIn("DATASETS", symbols)
            self.assertIn("MODELS", symbols)
            self.assertNotIn("dataset", symbols)
            self.assertNotIn("policy", symbols)
            self.assertNotIn("encoder", symbols)

    def test_module_control_flow_registry_is_still_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "training" / "conditional.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "if True:\n"
                "    DATASETS = {'alpha': 1}\n"
                "else:\n"
                "    DATASETS = {'beta': 2}\n",
                encoding="utf-8",
            )
            rows = declarations._python_findings(source, "training/conditional.py")
            datasets = next(row for row in rows if row["symbol"] == "DATASETS")
            self.assertEqual(["alpha", "beta"], datasets["members"])

    def test_ci_workflows_and_archived_controller_bundles_are_not_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = root / ".github" / "workflows" / "model-training.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text(
                "name: training\n"
                "jobs:\n"
                "  train:\n"
                "    strategy:\n"
                "      matrix:\n"
                "        family: [a, b]\n"
                "    steps:\n"
                "      - run: echo test\n",
                encoding="utf-8",
            )
            archived = root / "controller_bundle_v99" / "controller.py"
            archived.parent.mkdir(parents=True)
            archived.write_text("MODELS = {'controller': object()}\n", encoding="utf-8")

            inventory = workload._inventory(root, [])
            self.assertEqual([], inventory["training_config_surfaces"])
            self.assertEqual([], inventory["registry_surfaces"])
            excluded = {row["path"]: row["reason"]
                        for row in inventory["excluded_workload_infrastructure"]}
            self.assertEqual("ci_workflow", excluded[".github/workflows/model-training.yml"])
            self.assertEqual("archived_controller_bundle",
                             excluded["controller_bundle_v99/controller.py"])
            self.assertEqual([], selectors._iter_component_configs(root))
            self.assertEqual([], declarations._structured_choices(root))

    def test_real_training_recipe_remains_a_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "config" / "model_training.yaml"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(
                "model: dense-retriever\n"
                "epochs: 3\n"
                "batch_size: 8\n",
                encoding="utf-8",
            )
            inventory = workload._inventory(root, [])
            rows = inventory["training_config_surfaces"]
            self.assertEqual(["config/model_training.yaml"], [row["path"] for row in rows])
            self.assertEqual(["model"], rows[0]["identity_keys"])
            self.assertIn("epochs", rows[0]["training_keys"])
            self.assertIn("config/model_training.yaml", selectors._iter_component_configs(root))


if __name__ == "__main__":
    unittest.main()
