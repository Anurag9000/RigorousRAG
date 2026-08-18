"""Privacy-safe quality observations for residency, evidence trust and runtime authority."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from evaluation.quality_observability import MetricObservation
from orchestration.runtime_stack_authority import RuntimeAuthorityRecord, RuntimePromotionDecision
from security.data_residency import ResidencyDecision
from security.retrieved_content_trust import RetrievedContentTrustDecision


def observations_from_residency(
    decisions: Sequence[ResidencyDecision],
    *,
    service_id: str,
    policy_sha256: str,
) -> tuple[MetricObservation, ...]:
    rows = tuple(decisions)
    if not rows or any(not isinstance(value, ResidencyDecision) for value in rows):
        raise ValueError("decisions must be a non-empty ResidencyDecision sequence")
    if any(value.policy_sha256 != policy_sha256 for value in rows):
        raise ValueError("residency observations must share the supplied policy digest")
    eligible = sum(value.eligible for value in rows)
    tags = (("service_id", service_id), ("policy_sha256", policy_sha256), ("metric_family", "residency"))
    count = len(rows)
    return (
        MetricObservation("residency.eligible_rate", eligible / count, "higher", "ratio", count, "security.data_residency", tags),
        MetricObservation("residency.ineligible_rate", (count - eligible) / count, "lower", "ratio", count, "security.data_residency", tags),
    )


def observations_from_retrieved_content_trust(
    decisions: Sequence[RetrievedContentTrustDecision],
    *,
    policy_sha256: str,
) -> tuple[MetricObservation, ...]:
    rows = tuple(decisions)
    if not rows or any(not isinstance(value, RetrievedContentTrustDecision) for value in rows):
        raise ValueError("decisions must be a non-empty RetrievedContentTrustDecision sequence")
    if any(value.policy_sha256 != policy_sha256 for value in rows):
        raise ValueError("trust observations must share the supplied policy digest")
    counts = Counter(value.action for value in rows)
    count = len(rows)
    signaled = sum(bool(value.signal_sha256s) for value in rows)
    tags = (("policy_sha256", policy_sha256), ("metric_family", "retrieved_content_trust"))
    return (
        MetricObservation("retrieved_content.allow_rate", (counts["allow_as_evidence"] + counts["allow_with_warning"]) / count, "higher", "ratio", count, "security.retrieved_content_trust", tags),
        MetricObservation("retrieved_content.review_rate", counts["review"] / count, "lower", "ratio", count, "security.retrieved_content_trust", tags),
        MetricObservation("retrieved_content.quarantine_rate", counts["quarantine"] / count, "lower", "ratio", count, "security.retrieved_content_trust", tags),
        MetricObservation("retrieved_content.instruction_signal_rate", signaled / count, "lower", "ratio", count, "security.retrieved_content_trust", tags),
    )


def observations_from_runtime_promotion(
    decision: RuntimePromotionDecision,
    *,
    service_id: str,
    domain: str,
) -> tuple[MetricObservation, ...]:
    if not isinstance(decision, RuntimePromotionDecision):
        raise ValueError("decision must be RuntimePromotionDecision")
    tags = (
        ("service_id", service_id),
        ("domain", domain),
        ("stack_sha256", decision.stack_sha256),
        ("policy_sha256", decision.policy_sha256),
        ("metric_family", "runtime_promotion"),
    )
    return (
        MetricObservation("runtime.promotion_eligible", float(decision.eligible), "higher", "boolean", 1, "orchestration.runtime_stack_authority", tags),
        MetricObservation("runtime.promotion_failure_reason_count", float(len(decision.reason_codes)), "lower", "count", 1, "orchestration.runtime_stack_authority", tags),
    )


def observations_from_runtime_authority(record: RuntimeAuthorityRecord) -> tuple[MetricObservation, ...]:
    if not isinstance(record, RuntimeAuthorityRecord):
        raise ValueError("record must be RuntimeAuthorityRecord")
    tags = (
        ("service_id", record.service_id),
        ("domain", record.domain_id),
        ("stack_sha256", record.stack_sha256),
        ("variant", record.action),
        ("metric_family", "runtime_authority"),
    )
    return (
        MetricObservation("runtime.authority_revision", float(record.authority_revision), "neutral", "revision", 1, "orchestration.runtime_stack_authority", tags),
        MetricObservation("runtime.fencing_token", float(record.fencing_token), "neutral", "token", 1, "orchestration.runtime_stack_authority", tags),
        MetricObservation("runtime.rollback_active", float(record.action == "rollback"), "neutral", "boolean", 1, "orchestration.runtime_stack_authority", tags),
    )


__all__ = [
    "observations_from_residency",
    "observations_from_retrieved_content_trust",
    "observations_from_runtime_authority",
    "observations_from_runtime_promotion",
]
