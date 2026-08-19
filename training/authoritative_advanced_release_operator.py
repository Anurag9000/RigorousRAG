"""Local-only production release operator for advanced RAG artifacts.

The general advanced-training operator retains research/reporting commands.  This module owns
the stricter production evidence path:

verified checkpoint -> verified v2 result artifacts -> checkpoint-bound authoritative
evaluation evidence -> exact exported artifact -> metric policy -> authoritative promotion
evidence.

It performs no benchmark, training, inference, model download or network access.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.authoritative_advanced_evaluation import (
    build_authoritative_advanced_evaluation_evidence,
    write_authoritative_advanced_evaluation_evidence,
)
from evaluation.authoritative_advanced_evaluation_verification import (
    verify_authoritative_advanced_evaluation_evidence,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_artifacts import MetricQualificationPolicy
from training.advanced_rag_operator import verify_checkpoint_from_config
from training.advanced_rag_runtime_loading import read_advanced_artifact_manifest
from training.authoritative_advanced_promotion import (
    assert_authoritative_advanced_promotion,
    build_authoritative_advanced_promotion_evidence,
    read_authoritative_advanced_promotion_evidence,
    write_authoritative_advanced_promotion_evidence,
)

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path, label: str) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must contain an object")
    return raw


def _promotion_policy(path: str | Path) -> MetricQualificationPolicy:
    raw = _read(path, "promotion policy")
    if (
        set(raw) != {"schema", "minimum", "maximum"}
        or raw.get("schema") != "rigorousrag-advanced-promotion-policy-config/v1"
    ):
        raise ValueError(
            "promotion policy must be rigorousrag-advanced-promotion-policy-config/v1"
        )
    if not isinstance(raw["minimum"], Mapping) or not isinstance(raw["maximum"], Mapping):
        raise ValueError("promotion policy minimum/maximum must be objects")
    return MetricQualificationPolicy(minimum=raw["minimum"], maximum=raw["maximum"])


def build_evaluation_evidence_from_config(
    config_path: str | Path,
    *,
    checkpoint_digest: str,
    result_receipt_paths: Sequence[str | Path],
    aggregation: str,
    evaluation_receipt_output: str | Path,
    evidence_output: str | Path,
) -> Mapping[str, Any]:
    binding = verify_checkpoint_from_config(
        config_path,
        checkpoint_digest=checkpoint_digest,
    )
    evaluation, evidence = build_authoritative_advanced_evaluation_evidence(
        binding,
        result_receipt_paths=result_receipt_paths,
        aggregation=aggregation,
        evaluation_receipt_path=evaluation_receipt_output,
    )
    write_authoritative_advanced_evaluation_evidence(evidence_output, evidence)
    verified_evaluation, verified_evidence = verify_authoritative_advanced_evaluation_evidence(
        evidence_output
    )
    if (
        verified_evaluation.receipt_sha256 != evaluation.receipt_sha256
        or verified_evidence.evidence_sha256 != evidence.evidence_sha256
    ):
        raise RuntimeError("authoritative evaluation changed during release publication")
    return {
        "evaluation_receipt_path": str(
            safe_advanced_path(
                evaluation_receipt_output,
                label="advanced evaluation receipt",
                must_exist=True,
                require_file=True,
            )
        ),
        "evaluation_receipt_sha256": evaluation.receipt_sha256,
        "authoritative_evaluation_evidence_path": str(
            safe_advanced_path(
                evidence_output,
                label="authoritative evaluation evidence",
                must_exist=True,
                require_file=True,
            )
        ),
        "authoritative_evaluation_evidence_sha256": evidence.evidence_sha256,
        "benchmark_id": evidence.benchmark_id,
        "benchmark_manifest_sha256": evidence.benchmark_manifest_sha256,
        "evaluator_contract_sha256": evidence.evaluator_contract_sha256,
        "run_count": len(evidence.runs),
        "sample_count": evidence.sample_count,
    }


def qualify_artifact_authoritatively(
    artifact_directory: str | Path,
    *,
    authoritative_evaluation_evidence_path: str | Path,
    policy_path: str | Path,
    promotion_output: str | Path,
) -> Mapping[str, Any]:
    directory = safe_advanced_path(
        artifact_directory,
        label="advanced artifact directory",
        must_exist=True,
        require_directory=True,
    )
    manifest = read_advanced_artifact_manifest(directory)
    policy = _promotion_policy(policy_path)
    promotion = build_authoritative_advanced_promotion_evidence(
        manifest,
        authoritative_evaluation_evidence_path=authoritative_evaluation_evidence_path,
        policy=policy,
    )
    write_authoritative_advanced_promotion_evidence(promotion_output, promotion)
    parsed = read_authoritative_advanced_promotion_evidence(promotion_output)
    assert_authoritative_advanced_promotion(manifest, parsed)
    if parsed.evidence_sha256 != promotion.evidence_sha256:
        raise RuntimeError("authoritative promotion evidence changed during publication")
    return {
        "artifact_sha256": manifest.artifact_sha256,
        "promoted": parsed.promoted,
        "reason_codes": list(parsed.reason_codes),
        "policy_sha256": parsed.policy_sha256,
        "metrics_sha256": parsed.metrics_sha256,
        "evaluation_receipt_sha256": parsed.evaluation_receipt_sha256,
        "authoritative_evaluation_evidence_sha256": parsed.authoritative_evaluation_evidence_sha256,
        "authoritative_promotion_evidence_sha256": parsed.evidence_sha256,
        "promotion_output": str(
            safe_advanced_path(
                promotion_output,
                label="authoritative promotion evidence",
                must_exist=True,
                require_file=True,
            )
        ),
    }


def verify_authoritative_promotion(
    artifact_directory: str | Path,
    promotion_evidence_path: str | Path,
) -> Mapping[str, Any]:
    directory = safe_advanced_path(
        artifact_directory,
        label="advanced artifact directory",
        must_exist=True,
        require_directory=True,
    )
    manifest = read_advanced_artifact_manifest(directory)
    evidence = read_authoritative_advanced_promotion_evidence(promotion_evidence_path)
    assert_authoritative_advanced_promotion(manifest, evidence)
    return {
        "artifact_sha256": manifest.artifact_sha256,
        "promoted": evidence.promoted,
        "reason_codes": list(evidence.reason_codes),
        "evaluation_receipt_sha256": evidence.evaluation_receipt_sha256,
        "authoritative_evaluation_evidence_sha256": evidence.authoritative_evaluation_evidence_sha256,
        "authoritative_promotion_evidence_sha256": evidence.evidence_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-authoritative-release",
        description="Authoritative local-only advanced RAG release evidence operator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    evaluation = sub.add_parser("evaluation-evidence")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--checkpoint-digest", required=True)
    evaluation.add_argument("--result-receipt", action="append", required=True)
    evaluation.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    evaluation.add_argument("--evaluation-receipt-output", required=True)
    evaluation.add_argument("--evidence-output", required=True)

    qualify = sub.add_parser("qualify")
    qualify.add_argument("--artifact", required=True)
    qualify.add_argument("--evaluation-evidence", required=True)
    qualify.add_argument("--policy", required=True)
    qualify.add_argument("--output", required=True)

    verify = sub.add_parser("verify-promotion")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--evidence", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluation-evidence":
        result = build_evaluation_evidence_from_config(
            args.config,
            checkpoint_digest=args.checkpoint_digest,
            result_receipt_paths=tuple(args.result_receipt),
            aggregation=args.aggregation,
            evaluation_receipt_output=args.evaluation_receipt_output,
            evidence_output=args.evidence_output,
        )
    elif args.command == "qualify":
        result = qualify_artifact_authoritatively(
            args.artifact,
            authoritative_evaluation_evidence_path=args.evaluation_evidence,
            policy_path=args.policy,
            promotion_output=args.output,
        )
    elif args.command == "verify-promotion":
        result = verify_authoritative_promotion(args.artifact, args.evidence)
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_evaluation_evidence_from_config",
    "main",
    "qualify_artifact_authoritatively",
    "verify_authoritative_promotion",
]
