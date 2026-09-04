"""Fail-closed artifact-bound authority for RigorousRAG classical learners.

Version 3 preserves the source-revision and exact-resume guarantees introduced by
``authoritative_classical_training_cli_v2`` and closes the remaining fusion/ListNet
artifact-identity gap.  Fusion learners must now name the real calibration contract
file and one real calibration artifact per profile.  Their SHA-256 digests are derived
from admitted bytes at launch and become part of the immutable training specification;
optional expected digests are verification constraints, never substitutes for files.

Consequently a missing artifact, malformed/placeholder digest, digest mismatch, profile
coverage mismatch, source change, train/validation mutation, or calibration mutation
fails closed instead of silently resuming an incompatible optimization state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import authoritative_classical_training_cli as v1
from training import authoritative_classical_training_cli_v2 as v2

SCHEMA = v1.SCHEMA
RESULT_SCHEMA = "rigorousrag-authoritative-classical-training-result/v3"
_HEX = frozenset("0123456789abcdef")


def _expected_sha256(value: Any, label: str) -> str:
    selected = v1._identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    if selected == "0" * 64:
        raise ValueError(f"{label} may not use the all-zero placeholder digest")
    return selected


def _reject_placeholder_source_revision(value: Any) -> None:
    if not isinstance(value, str):
        return
    selected = value.strip().lower()
    if selected in {"0" * 40, "0" * 64}:
        raise ValueError("source_revision may not use an all-zero placeholder; use 'auto' for the checked-out Git revision")


def _artifact_path(root: Path, value: Any, label: str) -> Path:
    path = v1._path(root, value, label, must_exist=True)
    if not path.is_file():
        raise FileNotFoundError(f"{label} must resolve to a regular file: {path}")
    return path


def _bind_calibration_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    bound = dict(config)
    profiles = tuple(v1._identifier(value, "profile_id", 200) for value in config.get("profile_ids", ()))
    if not profiles:
        raise ValueError("profile_ids must contain at least one profile")
    if len(set(profiles)) != len(profiles):
        raise ValueError("profile_ids must be unique")

    contract_path = _artifact_path(root, config.get("calibration_contract"), "calibration_contract")
    raw_artifacts = config.get("calibration_artifacts")
    if not isinstance(raw_artifacts, Mapping):
        raise ValueError("calibration_artifacts must be a profile->file object")
    artifact_keys = {str(key) for key in raw_artifacts}
    if artifact_keys != set(profiles):
        missing = sorted(set(profiles) - artifact_keys)
        unexpected = sorted(artifact_keys - set(profiles))
        raise ValueError(
            "calibration_artifacts must cover profile_ids exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )

    contract_sha = v1._sha_file(contract_path)
    artifact_paths: dict[str, Path] = {}
    artifact_shas: dict[str, str] = {}
    for profile in profiles:
        artifact = _artifact_path(root, raw_artifacts[profile], f"calibration_artifacts[{profile!r}]")
        artifact_paths[profile] = artifact
        artifact_shas[profile] = v1._sha_file(artifact)

    expected_contract = config.get("calibration_contract_sha256")
    if expected_contract is not None:
        expected = _expected_sha256(expected_contract, "calibration_contract_sha256")
        if expected != contract_sha:
            raise ValueError(
                "calibration_contract_sha256 does not match calibration_contract bytes: "
                f"expected {expected}, actual {contract_sha}"
            )

    expected_artifacts = config.get("calibration_artifact_sha256s")
    if expected_artifacts is not None:
        if not isinstance(expected_artifacts, Mapping):
            raise ValueError("calibration_artifact_sha256s must be a profile->SHA256 object")
        expected_keys = {str(key) for key in expected_artifacts}
        if expected_keys != set(profiles):
            raise ValueError("calibration_artifact_sha256s must cover profile_ids exactly when supplied")
        for profile in profiles:
            expected = _expected_sha256(expected_artifacts[profile], f"calibration_artifact_sha256s[{profile!r}]")
            if expected != artifact_shas[profile]:
                raise ValueError(
                    f"calibration artifact digest mismatch for profile {profile!r}: "
                    f"expected {expected}, actual {artifact_shas[profile]}"
                )

    # v1's tested training implementations consume immutable digests.  Replace any
    # caller-supplied values with digests computed from the admitted files.
    bound["profile_ids"] = list(profiles)
    bound["calibration_contract_sha256"] = contract_sha
    bound["calibration_artifact_sha256s"] = artifact_shas
    identity = {
        "schema": "rigorousrag-calibration-input-identity/v1",
        "contract": {
            "configured_path": str(config["calibration_contract"]),
            "sha256": contract_sha,
        },
        "artifacts": {
            profile: {
                "configured_path": str(raw_artifacts[profile]),
                "sha256": artifact_shas[profile],
            }
            for profile in profiles
        },
    }
    identity = {**identity, "identity_sha256": v1._digest(identity)}
    return bound, identity


def run_config(config_path: str | Path) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    root, raw_config, kind, train_path, validation_path, output_dir = v1._common(selected)
    _reject_placeholder_source_revision(raw_config.get("source_revision"))
    config = dict(raw_config)
    config["source_revision"] = v2._source_revision(config.get("source_revision", "auto"))

    input_artifacts: Mapping[str, Any] | None = None
    if kind in {"fusion_weight", "listwise_fusion"}:
        config, input_artifacts = _bind_calibration_inputs(root, config)

    if kind == "fusion_weight":
        result = v1._run_fusion(config, train_path, validation_path, output_dir)
    elif kind == "listwise_fusion":
        result = v1._run_listwise(config, train_path, validation_path, output_dir)
    elif kind == "domain_classifier":
        result = v2._run_domain(config, train_path, validation_path, output_dir)
    elif kind == "plan_ranker":
        result = v2._run_plan(config, train_path, validation_path, output_dir)
    else:  # v1._common is closed-world, retain a defensive guard here as well.
        raise ValueError(f"unsupported classical training kind {kind!r}")

    manifest: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "source_revision": config["source_revision"],
        "config_sha256": v1._sha_file(selected),
        "train_data_sha256": v1._sha_file(train_path),
        "validation_data_sha256": v1._sha_file(validation_path),
        "result": result,
    }
    if input_artifacts is not None:
        manifest["input_artifacts"] = input_artifacts
    manifest["result_sha256"] = v1._digest(manifest)
    v1._atomic_json(output_dir / "training_result.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run or exactly resume one artifact-bound classical recipe")
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
