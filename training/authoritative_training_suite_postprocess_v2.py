#!/usr/bin/env python3
"""Cohort-aware wrapper around the authoritative training-suite postprocess.

The scientific validation/test/metrics logic remains in v1.  This wrapper changes
only retrieval artifact root resolution while the synchronized cohort adapter is
active: a recipe result/checkpoint under ``output_dir/dataset_cohort_v1`` is
preferred over a legacy individually-trained artifact under ``output_dir``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import authoritative_training_suite_postprocess as base

COHORT_SUBDIR = "dataset_cohort_v1"


def _retrieval_root(config: Path, raw: Mapping[str, Any]) -> Path:
    output = raw.get("output_dir")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("retrieval config must declare output_dir")
    candidate = Path(output).expanduser()
    standard = candidate if candidate.is_absolute() else config.parent / candidate
    standard = standard.resolve(strict=False)
    cohort = standard / COHORT_SUBDIR
    if (cohort / "training_result.json").is_file():
        return cohort
    return standard


def _config_and_result(family: str, config_path: str | Path) -> tuple[Path, Mapping[str, Any], Path, Mapping[str, Any]]:
    config = base._regular(config_path, "training config")
    raw = base._read_json(config, "training config")
    if family == "retrieval":
        root = _retrieval_root(config, raw)
        result_path = root / "training_result.json"
        result = base._read_json(base._regular(result_path, "training result"), "training result")
        return config, raw, result_path.resolve(strict=True), result
    return base._config_and_result(family, config_path)


def _retrieval_checkpoint(config: Path, raw: Mapping[str, Any]) -> Mapping[str, Any]:
    from training.checkpointing import CheckpointManager

    root = _retrieval_root(config, raw)
    manager = CheckpointManager(root / "checkpoints")
    pointer = "best" if (manager.root / "best.json").is_file() else "latest"
    if not (manager.root / f"{pointer}.json").is_file():
        raise ValueError("retrieval cohort training has neither a best nor latest checkpoint pointer")
    digest = manager.resolve_pointer(pointer)
    path, manifest = manager.verify(digest)
    return {
        "checkpoint_digest": digest,
        "checkpoint_path": str(path),
        "checkpoint_run_id": manifest.run_id,
        "checkpoint_source_commit": manifest.source_commit,
        "checkpoint_training_config_digest": manifest.training_config_digest,
        "checkpoint_dataset_manifest_digest": manifest.dataset_manifest_digest,
        "checkpoint_metric_snapshot": dict(manifest.metric_snapshot),
        "dataset_cohort_artifact": root.name == COHORT_SUBDIR,
    }


def validate_result(family: str, config_path: str | Path, training_job_id: str) -> Mapping[str, Any]:
    original = base._config_and_result
    try:
        base._config_and_result = _config_and_result
        return base.validate_result(family, config_path, training_job_id)
    finally:
        base._config_and_result = original


def test_artifact(family: str, config_path: str | Path, training_job_id: str) -> Mapping[str, Any]:
    original_config = base._config_and_result
    original_checkpoint = base._retrieval_checkpoint
    try:
        base._config_and_result = _config_and_result
        base._retrieval_checkpoint = _retrieval_checkpoint
        return base.test_artifact(family, config_path, training_job_id)
    finally:
        base._config_and_result = original_config
        base._retrieval_checkpoint = original_checkpoint


def metrics_result(family: str, config_path: str | Path, training_job_id: str) -> Mapping[str, Any]:
    original = base._config_and_result
    try:
        base._config_and_result = _config_and_result
        return base.metrics_result(family, config_path, training_job_id)
    finally:
        base._config_and_result = original


def _parser() -> argparse.ArgumentParser:
    return base._parser()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "validate":
        payload = validate_result(args.family, args.config, args.training_job_id)
    elif args.operation == "test":
        payload = test_artifact(args.family, args.config, args.training_job_id)
    elif args.operation == "metrics":
        payload = metrics_result(args.family, args.config, args.training_job_id)
    else:
        raise RuntimeError(f"unsupported postprocess operation {args.operation}")
    base._atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["COHORT_SUBDIR", "main", "metrics_result", "test_artifact", "validate_result"]
