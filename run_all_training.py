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
CONTROLLER_BLOB = "16db0d4c6fe886fc204d23b54b61bb7e6590dc13"


def _job(
    job_id: str,
    source: str,
    config: str,
    family: str,
    *,
    device_capable: bool,
) -> dict[str, object]:
    return {
        "id": job_id,
        "command": [source, "train", "--config", config],
        "entrypoint_source": source,
        "device_capable": device_capable,
        "phase": "training",
        "family": family,
        "repeat_index": 0,
        "recipe_config": config,
    }


CLASSICAL_SOURCE = "training/authoritative_classical_training_cli_v3.py"
RETRIEVAL_SOURCE = "training/authoritative_retrieval_training_cli.py"

# Registered console subcommands materialize one canonical recipe per training
# authority. Additional curricula using the same authority are explicit jobs so
# the pressure scheduler can admit/pause/resume each recipe independently without
# duplicating parent console programs.
EXTRA_JOBS = [
    _job(
        "rigorousrag-advanced-training:train:dynamic-rag-policy",
        "training/authoritative_advanced_training_cli.py",
        "config/advanced_dynamic_rag_training.example.json",
        "advanced-rag-dynamic-policy",
        device_capable=True,
    ),
    _job(
        "rigorousrag-classical-training:train:listwise-fusion",
        CLASSICAL_SOURCE,
        "config/classical_listwise_fusion_training.example.json",
        "classical-listwise-fusion",
        device_capable=False,
    ),
    _job(
        "rigorousrag-classical-training:train:domain-classifier",
        CLASSICAL_SOURCE,
        "config/classical_domain_training.example.json",
        "classical-domain-classifier",
        device_capable=False,
    ),
    _job(
        "rigorousrag-classical-training:train:plan-ranker",
        CLASSICAL_SOURCE,
        "config/classical_plan_ranker_training.example.json",
        "classical-plan-ranker",
        device_capable=False,
    ),
    _job(
        "rigorousrag-retrieval-training:train:dense-distilled",
        RETRIEVAL_SOURCE,
        "config/retrieval_dense_distilled_training.example.json",
        "retrieval-dense-distilled",
        device_capable=True,
    ),
    _job(
        "rigorousrag-retrieval-training:train:splade-base",
        RETRIEVAL_SOURCE,
        "config/retrieval_splade_base_training.example.json",
        "retrieval-splade-base",
        device_capable=True,
    ),
    _job(
        "rigorousrag-retrieval-training:train:splade-distilled",
        RETRIEVAL_SOURCE,
        "config/retrieval_splade_distilled_training.example.json",
        "retrieval-splade-distilled",
        device_capable=True,
    ),
    _job(
        "rigorousrag-retrieval-training:train:unicoil",
        RETRIEVAL_SOURCE,
        "config/retrieval_unicoil_training.example.json",
        "retrieval-unicoil",
        device_capable=True,
    ),
    _job(
        "rigorousrag-retrieval-training:train:colbert-base",
        RETRIEVAL_SOURCE,
        "config/retrieval_colbert_base_training.example.json",
        "retrieval-colbert-base",
        device_capable=True,
    ),
    _job(
        "rigorousrag-retrieval-training:train:colbert-distilled",
        RETRIEVAL_SOURCE,
        "config/retrieval_colbert_distilled_training.example.json",
        "retrieval-colbert-distilled",
        device_capable=True,
    ),
    _job(
        "rigorousrag-retrieval-training:train:cross-encoder-listwise",
        RETRIEVAL_SOURCE,
        "config/retrieval_cross_encoder_training.example.json",
        "retrieval-cross-encoder-listwise",
        device_capable=True,
    ),
]

PROFILE = {
    "repository": REPOSITORY,
    "preferred_training_entrypoints": [],
    "preferred_dataset_entrypoints": [],
    "dynamic_registry_covers": [],
    "ignore_entrypoints": ["run_all_training.py"],
    "extra_jobs": EXTRA_JOBS,
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
        ],
        "rigorousrag-classical-training:train": [
            "--config",
            "config/classical_fusion_training.example.json",
        ],
        "rigorousrag-retrieval-training:train": [
            "--config",
            "config/retrieval_dense_base_training.example.json",
        ],
    },
    "console_subcommand_metadata": {
        "rigorousrag-advanced-training:train": {
            "recipe_config": "config/advanced_grounded_training.example.json",
            "family": "advanced-rag-grounded-generation",
        },
        "rigorousrag-classical-training:train": {
            "recipe_config": "config/classical_fusion_training.example.json",
            "family": "classical-fusion-weight",
            "device_capable": False,
        },
        "rigorousrag-retrieval-training:train": {
            "recipe_config": "config/retrieval_dense_base_training.example.json",
            "family": "retrieval-dense-base",
            "device_capable": True,
        },
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
