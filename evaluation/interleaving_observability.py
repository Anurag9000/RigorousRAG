"""Privacy-safe quality observations for retrieval interleaving promotion."""

from __future__ import annotations

from evaluation.quality_observability import MetricObservation
from evaluation.retrieval_interleaving_promotion import InterleavingPromotionReceipt


def observations_from_interleaving_promotion(
    receipt: InterleavingPromotionReceipt,
) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, InterleavingPromotionReceipt):
        raise ValueError("receipt must be InterleavingPromotionReceipt")
    tie_fraction = receipt.ties / receipt.impression_count if receipt.impression_count else 1.0
    decisive_fraction = receipt.decisive_count / receipt.impression_count if receipt.impression_count else 0.0
    tags = (
        ("policy_digest", receipt.candidate_policy_sha256),
        ("metric_family", "retrieval_interleaving"),
        ("variant", f"candidate_{receipt.candidate_team}"),
    )
    source = "evaluation.retrieval_interleaving_promotion"
    return (
        MetricObservation("retrieval.interleaving.preference_rate", receipt.candidate_preference_rate, "higher", "ratio", receipt.impression_count, source, tags),
        MetricObservation("retrieval.interleaving.wilson_low", receipt.candidate_wilson_low, "higher", "ratio", receipt.decisive_count, source, tags),
        MetricObservation("retrieval.interleaving.sign_test_p_value", receipt.sign_test_p_value, "lower", "probability", receipt.decisive_count, source, tags),
        MetricObservation("retrieval.interleaving.tie_fraction", tie_fraction, "lower", "ratio", receipt.impression_count, source, tags),
        MetricObservation("retrieval.interleaving.decisive_fraction", decisive_fraction, "higher", "ratio", receipt.impression_count, source, tags),
        MetricObservation("retrieval.interleaving.eligible", float(receipt.eligible), "higher", "ratio", receipt.impression_count, source, tags),
    )


__all__ = ["observations_from_interleaving_promotion"]
