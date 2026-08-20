"""Strict production qualification and verification helpers for advanced releases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_artifacts import MetricQualificationPolicy
from training.advanced_rag_runtime_loading import read_advanced_artifact_manifest
from training.authoritative_advanced_promotion import (
    read_authoritative_advanced_promotion_evidence,
    write_authoritative_advanced_promotion_evidence,
)
from training.strict_authoritative_advanced_promotion import (
    assert_strict_authoritative_advanced_promotion,
    build_strict_authoritative_advanced_promotion_evidence,
)

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read_policy(path: str | Path) -> MetricQualificationPolicy:
    source = safe_advanced_path(
        path,
        label="promotion policy",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("promotion policy exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("promotion policy is not strict JSON") from exc
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"schema", "minimum", "maximum"}
        or raw.get("schema") != "rigorousrag-advanced-promotion-policy-config/v1"
        or not isinstance(raw.get("minimum"), Mapping)
        or not isinstance(raw.get("maximum"), Mapping)
    ):
        raise ValueError(
            "promotion policy must be rigorousrag-advanced-promotion-policy-config/v1"
        )
    return MetricQualificationPolicy(minimum=raw["minimum"], maximum=raw["maximum"])


def qualify_artifact_strictly(
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
    policy = _read_policy(policy_path)
    promotion = build_strict_authoritative_advanced_promotion_evidence(
        manifest,
        authoritative_evaluation_evidence_path=authoritative_evaluation_evidence_path,
        policy=policy,
    )
    write_authoritative_advanced_promotion_evidence(promotion_output, promotion)
    parsed = read_authoritative_advanced_promotion_evidence(promotion_output)
    assert_strict_authoritative_advanced_promotion(manifest, parsed)
    if parsed.evidence_sha256 != promotion.evidence_sha256:
        raise RuntimeError("strict promotion evidence changed during publication")
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


def verify_promotion_strictly(
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
    assert_strict_authoritative_advanced_promotion(manifest, evidence)
    return {
        "artifact_sha256": manifest.artifact_sha256,
        "promoted": evidence.promoted,
        "reason_codes": list(evidence.reason_codes),
        "policy_sha256": evidence.policy_sha256,
        "metrics_sha256": evidence.metrics_sha256,
        "evaluation_receipt_sha256": evidence.evaluation_receipt_sha256,
        "authoritative_evaluation_evidence_sha256": evidence.authoritative_evaluation_evidence_sha256,
        "authoritative_promotion_evidence_sha256": evidence.evidence_sha256,
    }


__all__ = ["qualify_artifact_strictly", "verify_promotion_strictly"]
