#!/usr/bin/env python3
"""One-command exhaustive RigorousRAG scientific workload orchestration.

This root launcher owns no resource scheduler. It selects the repository-owned
closed-world scientific catalog and invokes the canonical controller bootstrap;
that bootstrap delegates admission, GPU selection, RAM/VRAM/swap pressure gates,
concurrency, pause/resume, retries, CUDA-OOM fallback and persistent process state
to the literal byte-pinned OPF_ADP scheduler.

The v2 scientific authority preserves the complete logical model/data/task/recipe
inventory while compiling same-dataset learned retrieval recipes into one
synchronized shared-batch physical cohort. Generic lifecycle discovery remains
disabled: production import/evaluation/release CLIs require content-bound
arguments and are admitted only through explicit governed lifecycle jobs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "Anurag9000/RigorousRAG"
ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "tools" / "universal_training_controller_entry.py"
CATALOG = "training/authoritative_training_suite_catalog_v2.py"

TRAINING_CONSOLE_ALIASES = [
    "rigorousrag-advanced-training",
    "rigorousrag-classical-training",
    "rigorousrag-retrieval-training",
]
TRAINING_SUBCOMMAND_ALIASES = [f"{name}:train" for name in TRAINING_CONSOLE_ALIASES]

PROFILE = {
    "repository": REPOSITORY,
    "scientific_authority": CATALOG,
    "scientific_authority_version": 39,
    "job_catalog": {"path": CATALOG, "function": "iter_jobs", "args": ["exhaustive"]},
    "disable_lifecycle_orchestration": True,
    "auto_lifecycle_discovery": False,
    "preferred_training_entrypoints": [],
    "preferred_dataset_entrypoints": [],
    "setup_commands": [],
    "ignore_entrypoints": ["run_all_training.py"],
    "dynamic_registry_covers": [],
    "auto_console_training_jobs": False,
    "auto_console_subcommand_jobs": False,
    "ignore_console_scripts": TRAINING_CONSOLE_ALIASES,
    "ignore_console_subcommands": TRAINING_SUBCOMMAND_ALIASES,
    "require_registered_training_entrypoints": True,
    "require_registered_training_scheduling": False,
    "require_registered_training_subcommands": True,
    "require_registered_training_subcommand_scheduling": False,
    "strict_coverage": True,
    "require_native_resume": True,
    "require_exact_resume": True,
    "require_training_exact_resume": True,
    "require_training_early_stopping": True,
    "require_well_formed_training_exemptions": True,
    "require_dag_enforcement": True,
    "require_model_surface_accounting": True,
    "require_workload_surface_accounting": True,
    "require_registry_member_accounting": True,
    "require_dynamic_registry_accounting": True,
    "require_scientific_component_config_accounting": True,
    "require_declared_combination_accounting": True,
    "require_scientific_ontology_accounting": True,
    "require_declarative_scientific_source_accounting": True,
    "require_extended_scientific_component_accounting": True,
    "require_full_scientific_choice_accounting": True,
    "require_role_paradigm_protocol_accounting": True,
    "require_existing_job_targets": True,
    "require_source_proven_training_exact_resume": True,
    "require_source_proven_training_early_stopping": True,
    "require_all_retained_trainable_source_reachability": True,
    "require_literal_opf_mechanism_parity": True,
    "require_dataset_cohort_execution": True,
    "require_cpu_gpu_backend_variants": True,
    "require_shared_batch_views": True,
    "require_uniform_cohort_batch_size": True,
    "require_cohort_exact_resume": True,
}


def main() -> int:
    if not CONTROLLER.is_file():
        raise RuntimeError(f"Canonical controller bootstrap is missing: {CONTROLLER}")
    if not (ROOT / CATALOG).is_file():
        raise RuntimeError(f"Authoritative RigorousRAG suite catalog is missing: {ROOT / CATALOG}")
    env = os.environ.copy()
    env["TRAINING_CONTROL_PROFILE"] = json.dumps(PROFILE, separators=(",", ":"), sort_keys=True)
    env["TRAINING_CONTROL_REPO_ROOT"] = str(ROOT)
    env.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC", "30")
    return subprocess.call([sys.executable, str(CONTROLLER), *sys.argv[1:]], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
