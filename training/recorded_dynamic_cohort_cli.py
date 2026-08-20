"""Config-driven publication/verification of recorded dynamic-runtime cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import DynamicRewardConfig
from training.authoritative_canonical_materialization import _dynamic_governance, _split_policy
from training.recorded_dynamic_cohort_authority import (
    publish_recorded_dynamic_cohort,
    verify_recorded_dynamic_cohort,
)

_MAX_CONFIG_BYTES = 64 * 1024 * 1024
_MAX_EPISODES = 100_000


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="recorded dynamic cohort config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("recorded dynamic cohort config exceeds byte safety bound")
    try:
        value = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("recorded dynamic cohort config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("recorded dynamic cohort config must contain an object")
    return value


def _reward(raw: Any) -> DynamicRewardConfig:
    if raw is None:
        return DynamicRewardConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("reward_config must be an object")
    allowed = {"discount", "gae_lambda", "retrieval_cost", "verification_cost", "abstention_cost"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"reward_config contains unsupported fields: {sorted(unknown)}")
    return DynamicRewardConfig(**dict(raw))


def run_recorded_dynamic_cohort_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    allowed = {
        "schema", "episode_receipt_paths", "governance", "split_policy", "source_commit",
        "reward_config", "dataset_output_dir", "cohort_output_dir",
    }
    required = allowed - {"reward_config"}
    if set(raw) - allowed or required - set(raw) or raw.get("schema") != "rigorousrag-recorded-dynamic-cohort-config/v1":
        raise ValueError("recorded dynamic cohort config has unsupported/missing fields or schema")
    paths = raw["episode_receipt_paths"]
    if not isinstance(paths, list) or not paths or len(paths) > _MAX_EPISODES or any(not isinstance(item, str) or not item.strip() for item in paths):
        raise ValueError(f"episode_receipt_paths must contain 1..{_MAX_EPISODES} non-empty path strings")
    verified = publish_recorded_dynamic_cohort(
        tuple(paths),
        governance=_dynamic_governance(raw["governance"]),
        split_policy=_split_policy(raw["split_policy"]),
        source_commit=raw["source_commit"],
        reward_config=_reward(raw.get("reward_config")),
        dataset_output_dir=raw["dataset_output_dir"],
        cohort_output_dir=raw["cohort_output_dir"],
    )
    receipt_path = Path(verified.root) / "cohort_receipt.json"
    return {
        "cohort_receipt_path": str(receipt_path),
        "cohort_receipt_sha256": verified.receipt.receipt_sha256,
        "dataset_manifest_sha256": verified.dataset.manifest.manifest_digest,
        "dataset_source_set_sha256": verified.dataset.receipt.source_set_sha256,
        "episode_count": verified.receipt.episode_count,
        "record_count": verified.receipt.record_count,
        "runtime_lineage_sha256": verified.receipt.runtime_lineage_sha256,
        "split_names": [item.name for item in verified.dataset.receipt.splits],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rigorousrag-dynamic-recordings")
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("config")
    verify = sub.add_parser("verify")
    verify.add_argument("receipt")
    args = parser.parse_args(argv)
    if args.command == "publish":
        result = run_recorded_dynamic_cohort_config(args.config)
    else:
        verified = verify_recorded_dynamic_cohort(args.receipt)
        result = {
            "cohort_receipt_path": str(Path(verified.root) / "cohort_receipt.json"),
            "cohort_receipt_sha256": verified.receipt.receipt_sha256,
            "dataset_manifest_sha256": verified.dataset.manifest.manifest_digest,
            "dataset_source_set_sha256": verified.dataset.receipt.source_set_sha256,
            "episode_count": verified.receipt.episode_count,
            "record_count": verified.receipt.record_count,
            "runtime_lineage_sha256": verified.receipt.runtime_lineage_sha256,
            "split_names": [item.name for item in verified.dataset.receipt.splits],
        }
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_recorded_dynamic_cohort_config"]
