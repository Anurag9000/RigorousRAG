"""Privacy-safe quality/efficiency observations for evidence context construction."""

from __future__ import annotations

from evaluation.quality_observability import MetricObservation
from tools.evidence_context_materialization import MaterializedContext
from tools.evidence_context_packing import ContextPackingPolicy, ContextPackingReceipt


def observations_from_context_packing(
    receipt: ContextPackingReceipt,
    *,
    policy: ContextPackingPolicy,
) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, ContextPackingReceipt) or not isinstance(policy, ContextPackingPolicy):
        raise ValueError("receipt/policy types are invalid")
    if receipt.policy_sha256 != policy.policy_sha256:
        raise ValueError("context packing receipt does not belong to supplied policy")
    count = len(receipt.selected)
    utilization = receipt.total_tokens / policy.max_context_tokens
    redundancies = [row.max_redundancy for row in receipt.selected]
    mean_redundancy = sum(redundancies) / count if count else 0.0
    max_redundancy = max(redundancies) if redundancies else 0.0
    counter_fraction = receipt.counterevidence_count / count if count else 0.0
    mandatory_fraction = receipt.mandatory_count / count if count else 0.0
    tags = (("metric_family", "context_packing"), ("policy_digest", policy.policy_sha256), ("variant", "selected"))
    source = "evaluation.context_packing_observability"
    return (
        MetricObservation("context_packing.selected_count", float(count), "neutral", "count", count, source, tags),
        MetricObservation("context_packing.total_tokens", float(receipt.total_tokens), "lower", "tokens", count, source, tags),
        MetricObservation("context_packing.token_budget_utilization", utilization, "neutral", "ratio", count, source, tags),
        MetricObservation("context_packing.mean_max_redundancy", mean_redundancy, "lower", "ratio", count, source, tags),
        MetricObservation("context_packing.max_redundancy", max_redundancy, "lower", "ratio", count, source, tags),
        MetricObservation("context_packing.counterevidence_fraction", counter_fraction, "higher", "ratio", count, source, tags),
        MetricObservation("context_packing.mandatory_fraction", mandatory_fraction, "neutral", "ratio", count, source, tags),
    )


def observations_from_materialized_context(context: MaterializedContext) -> tuple[MetricObservation, ...]:
    if not isinstance(context, MaterializedContext):
        raise ValueError("context must be MaterializedContext")
    count = len(context.evidence)
    tags = (("metric_family", "context_packing"), ("tokenizer_digest", context.tokenizer_sha256), ("variant", "materialized"))
    source = "evaluation.context_packing_observability"
    return (
        MetricObservation("context_packing.materialized_count", float(count), "neutral", "count", count, source, tags),
        MetricObservation("context_packing.materialized_tokens", float(context.total_tokens), "lower", "tokens", count, source, tags),
    )


__all__ = ["observations_from_context_packing", "observations_from_materialized_context"]
