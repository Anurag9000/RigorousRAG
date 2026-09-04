"""Source-revision-bound v2 authority for RigorousRAG classical learners.

v1 introduced the exact-resumable execution path.  This layer preserves that tested
implementation while binding every learner to the exact Git source revision. Fusion and
ListNet already include ``source_revision`` in their immutable training specs, so v2
resolves ``auto`` before delegating. Domain classification and plan ranking use the
existing exact ``ResumeStateStore`` but strengthen their training-manifest identity with
source revision in addition to train/validation bytes. A code revision change therefore
cannot silently resume an old fitting state.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import authoritative_classical_training_cli as v1
from training.query_plan_fitting import FittingConfig
from training.query_plan_resume import (
    ResumeStateStore,
    fit_domain_classifier_resumable,
    fit_plan_ranker_resumable,
)

SCHEMA = v1.SCHEMA
RESULT_SCHEMA = "rigorousrag-authoritative-classical-training-result/v2"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HEX = frozenset("0123456789abcdef")


def _source_revision(value: Any) -> str:
    requested = v1._identifier(value, "source_revision", 64).lower()
    if requested == "auto" or requested in {"0" * 40, "0" * 64}:
        try:
            requested = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.STDOUT,
                text=True,
            ).strip().lower()
        except Exception as exc:
            raise RuntimeError("source_revision=auto requires an exact Git checkout") from exc
    if len(requested) not in {40, 64} or any(ch not in _HEX for ch in requested):
        raise ValueError("source_revision must be auto or a 40/64-character hexadecimal Git object id")
    return requested


def _run_domain(
    config: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    training = v1._domain_examples(train_path)
    validation = v1._domain_examples(validation_path)
    source_revision = str(config["source_revision"])
    manifest = v1._digest(
        {
            "schema": "rigorousrag-domain-training-identity/v2",
            "source_revision": source_revision,
            "train_sha256": v1._sha_file(train_path),
            "validation_sha256": v1._sha_file(validation_path),
        }
    )
    fitting = FittingConfig(**dict(config.get("training", {})))
    store = ResumeStateStore(output_dir / "state")
    pointer = store.root / "domain-classifier-latest.json"
    artifact, result = fit_domain_classifier_resumable(
        training,
        labels=tuple(str(value) for value in config.get("labels", ())),
        fallback_label=config["fallback_label"],
        training_manifest_digest=manifest,
        validation=validation,
        minimum_confidence=float(config.get("minimum_confidence", 0.55)),
        config=fitting,
        state_store=store,
        resume=pointer.is_file(),
        checkpoint_every_batches=1,
    )
    return {
        "kind": "domain_classifier",
        "source_revision": source_revision,
        "training_manifest_digest": manifest,
        "artifact": asdict(artifact),
        "fit_result": asdict(result),
    }


def _run_plan(
    config: Mapping[str, Any],
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    training = v1._plan_examples(train_path)
    validation = v1._plan_examples(validation_path)
    source_revision = str(config["source_revision"])
    manifest = v1._digest(
        {
            "schema": "rigorousrag-plan-training-identity/v2",
            "source_revision": source_revision,
            "train_sha256": v1._sha_file(train_path),
            "validation_sha256": v1._sha_file(validation_path),
        }
    )
    fitting = FittingConfig(**dict(config.get("training", {})))
    store = ResumeStateStore(output_dir / "state")
    pointer = store.root / "plan-ranker-latest.json"
    artifact, result = fit_plan_ranker_resumable(
        training,
        training_manifest_digest=manifest,
        validation=validation,
        config=fitting,
        latency_penalty=float(config.get("latency_penalty", 0.0)),
        cost_penalty=float(config.get("cost_penalty", 0.0)),
        risk_penalty=float(config.get("risk_penalty", 0.0)),
        state_store=store,
        resume=pointer.is_file(),
        checkpoint_every_batches=1,
    )
    return {
        "kind": "plan_ranker",
        "source_revision": source_revision,
        "training_manifest_digest": manifest,
        "artifact": asdict(artifact),
        "fit_result": asdict(result),
    }


def run_config(config_path: str | Path) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    _, raw_config, kind, train_path, validation_path, output_dir = v1._common(selected)
    config = dict(raw_config)
    config["source_revision"] = _source_revision(config.get("source_revision", "auto"))

    if kind == "fusion_weight":
        result = v1._run_fusion(config, train_path, validation_path, output_dir)
    elif kind == "listwise_fusion":
        result = v1._run_listwise(config, train_path, validation_path, output_dir)
    elif kind == "domain_classifier":
        result = _run_domain(config, train_path, validation_path, output_dir)
    elif kind == "plan_ranker":
        result = _run_plan(config, train_path, validation_path, output_dir)
    else:  # defensive closed-world guard; v1._common already validates.
        raise ValueError(f"unsupported classical training kind {kind!r}")

    manifest = {
        "schema": RESULT_SCHEMA,
        "source_revision": config["source_revision"],
        "config_sha256": v1._sha_file(selected),
        "train_data_sha256": v1._sha_file(train_path),
        "validation_data_sha256": v1._sha_file(validation_path),
        "result": result,
    }
    manifest["result_sha256"] = v1._digest(manifest)
    v1._atomic_json(output_dir / "training_result.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run or exactly resume one source-bound classical recipe")
    train.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        print(json.dumps(run_config(args.config), sort_keys=True, separators=(",", ":")))
        return 0
    raise RuntimeError(f"unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RESULT_SCHEMA", "SCHEMA", "main", "run_config"]
