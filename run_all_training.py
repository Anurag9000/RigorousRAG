#!/usr/bin/env python3
"""One-command OPF-v20 training launcher for every RigorousRAG training recipe."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "Anurag9000/RigorousRAG"
ROOT = Path(__file__).resolve().parent
CONTROLLER = ROOT / "tools" / "universal_training_controller_entry.py"
CONTROLLER_BLOB = "b1dd61a2288a8140b327173317b132e1b2527d1f"

# The installed training authority exposes one argparse `train` subcommand, but
# the repository ships two independent curricula.  The console-subcommand job
# materializes the grounded-generation recipe and the explicit second job
# materializes the dynamic-RAG-policy recipe through the exact same authoritative
# CLI.  This avoids duplicate parent/subcommand scheduling while keeping both
# source paths in the closed-world training catalog.
PROFILE = {
    "repository": REPOSITORY,
    "preferred_training_entrypoints": [],
    "preferred_dataset_entrypoints": [],
    "dynamic_registry_covers": [],
    "ignore_entrypoints": ["run_all_training.py"],
    "extra_jobs": [
        {
            "id": "rigorousrag-advanced-training:train:dynamic-rag-policy",
            "command": [
                "training/authoritative_advanced_training_cli.py",
                "train",
                "--config",
                "config/advanced_dynamic_rag_training.example.json",
            ],
            "entrypoint_source": "training/authoritative_advanced_training_cli.py",
            "device_capable": True,
            "phase": "training",
            "family": "advanced-rag-dynamic-policy",
            "repeat_index": 0,
            "recipe_config": "config/advanced_dynamic_rag_training.example.json",
        }
    ],
    "auto_console_training_jobs": False,
    "require_registered_training_entrypoints": True,
    "require_registered_training_scheduling": True,
    "auto_console_subcommand_jobs": True,
    "require_registered_training_subcommands": True,
    "require_registered_training_subcommand_scheduling": True,
    "console_subcommand_args": {
        "rigorousrag-advanced-training:train": [
            "--config",
            "config/advanced_grounded_training.example.json",
        ]
    },
    "console_subcommand_metadata": {
        "rigorousrag-advanced-training:train": {
            "recipe_config": "config/advanced_grounded_training.example.json",
            "family": "advanced-rag-grounded-generation",
        }
    },
    "strict_coverage": True,
    "require_native_resume": True,
    "require_exact_resume": True,
    "require_training_exact_resume": True,
    "require_training_early_stopping": True,
    "require_dag_enforcement": True,
    "require_model_surface_accounting": True,
    "require_literal_opf_mechanism_parity": True,
    "require_well_formed_training_exemptions": True,
}


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def main() -> int:
    if not CONTROLLER.is_file():
        raise RuntimeError(f"Pinned training controller is missing: {CONTROLLER}")
    actual = _git_blob_sha(CONTROLLER.read_bytes())
    if actual != CONTROLLER_BLOB:
        raise RuntimeError(
            "Pinned local training controller bootstrap checksum mismatch: "
            f"expected {CONTROLLER_BLOB}, got {actual}"
        )

    env = os.environ.copy()
    env["TRAINING_CONTROL_PROFILE"] = json.dumps(PROFILE, separators=(",", ":"))
    env["TRAINING_CONTROL_REPO_ROOT"] = str(ROOT)
    env.setdefault("TRAINING_CONTROL_TERMINATION_GRACE_SEC", "30")
    return subprocess.call(
        [sys.executable, str(CONTROLLER), *sys.argv[1:]],
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
