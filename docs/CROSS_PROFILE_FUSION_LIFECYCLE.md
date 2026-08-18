# Governed cross-profile retrieval fusion

This document defines the source-level authority chain for combining retrieval signals from heterogeneous score spaces in RigorousRAG.

## Core safety invariant

BM25, learned-sparse, dense-vector, distance-based, late-interaction, reranker, and other retrieval scores are **not numerically comparable merely because they are floating-point values**. Raw scores from different score profiles must never be averaged, summed, or fed into learned cross-profile weights directly.

RigorousRAG therefore exposes two legitimate cross-profile paths:

1. **Rank-only fallback** — weighted reciprocal-rank fusion (`tools/corpus_fusion.py`) when compatible calibration evidence is unavailable or stale.
2. **Calibrated fusion** — map each profile's raw score independently to a held-out relevance probability under an immutable calibration contract, then fuse those calibrated probabilities.

The `AUTO` path in `tools/cross_profile_fusion.py` falls back to RRF if complete compatible calibration is not available. Strict calibrated modes fail closed instead of mixing incomparable scores.

## 1. Score-profile and calibration identity

`tools/cross_profile_fusion.py` binds every score-producing profile to:

- profile id and model/retriever family;
- exact scoring-contract digest;
- exact model-profile digest;
- score direction (`higher_is_better` or `lower_is_better`).

A `CalibrationContract` additionally binds the meaning of calibrated relevance probability to the exact:

- dataset manifest;
- split;
- relevance-label contract;
- candidate universe;
- domain/cohort.

A calibrator cannot silently move between score functions, model revisions, datasets, candidate populations, or relevance semantics.

## 2. Isotonic score calibration

`fit_isotonic_calibrator` implements dependency-free weighted pool-adjacent-violators isotonic regression. It supports either score direction and emits a content-addressed `IsotonicCalibrationArtifact`.

`evaluate_isotonic_calibrator` reports held-out Brier score and expected calibration error using the repository's shared calibration metrics.

The fitted artifact is a mathematical calibration model, **not automatically an approved production artifact**.

## 3. Calibration qualification

`evaluation/cross_profile_calibration.py` separates fitting from qualification.

A `CalibrationQualificationPolicy` can require:

- minimum total evaluation examples;
- minimum positive and negative support;
- maximum held-out Brier score;
- maximum ECE.

The qualification receipt binds the exact profile, calibrator artifact, calibration contract, evaluation-example digest, and policy digest. Strict multi-profile calibrated fusion requires one compatible calibration contract and one compatible qualification policy across participating profiles.

## 4. Governed fusion receipts

`tools/cross_profile_fusion_governance.py` binds fusion to:

- complete ranked input identity;
- all profile identities;
- all rank/raw-score values used as calibrator inputs;
- the complete fusion policy;
- calibration artifacts;
- the resulting fused ranking.

The governed wrapper also applies the same per-list candidate ceiling as the underlying RRF implementation. Durable receipts contain identifiers and digests, not raw query/document text.

Within calibrated fusion, duplicate shards from the same profile contribute at most once to a candidate. This prevents one retriever profile from gaining extra weight merely by being partitioned into multiple lists.

## 5. Pointwise learned profile weighting

`training/cross_profile_fusion_fitting.py` learns non-negative softmax-constrained profile weights from **calibrated probabilities only**.

Source includes:

- weighted binary log-loss;
- positive-class weighting;
- L2 regularization;
- gradient clipping;
- deterministic epoch permutations;
- deterministic minibatches;
- exact minibatch resume state;
- validation and early stopping;
- immutable calibration/data/config/source-revision lineage;
- content-addressed learned-weight artifacts.

This objective learns marginal relevance-probability weighting.

## 6. Listwise learned profile weighting

`training/cross_profile_listwise_fusion.py` complements the pointwise learner with a query-grouped ListNet-style objective.

It supports:

- graded relevance per candidate;
- query grouping by privacy-safe query digest;
- calibrated profile logits as features;
- softmax-constrained profile weights;
- target and prediction temperatures;
- query weights;
- L2 regularization and gradient clipping;
- deterministic query shuffles;
- exact minibatch resume;
- validation/early stopping;
- immutable training lineage.

This objective optimizes ordering rather than only marginal relevance probability.

## 7. Learned-weight promotion

Training success does not imply deployment eligibility.

`evaluation/fusion_weight_promotion.py` qualifies pointwise learned weights on a distinct held-out set using:

- minimum class support;
- absolute log-loss and Brier limits;
- improvement over uniform profile weighting;
- anti-collapse maximum single-profile weight.

`evaluation/listwise_fusion_promotion.py` qualifies listwise weights using:

- minimum evaluation queries;
- mean nDCG / nDCG@k;
- practical improvement over uniform weights;
- per-query regression fraction;
- anti-collapse limits.

Promotion receipts are content-addressed and tied to exact learned artifacts and evaluation splits.

## 8. Train-to-runtime lineage

`tools/learned_cross_profile_policy.py`, `tools/promoted_learned_cross_profile_policy.py`, and `tools/promoted_listwise_cross_profile_policy.py` make the source lineage explicit.

A promoted runtime path rejects:

- a different profile population;
- a replacement calibrator;
- a different calibration contract;
- an unqualified calibrator;
- a promotion receipt for another learned artifact.

Execution receipts bind the trained artifact, calibration artifacts, qualification/promotion evidence, and governed fusion result.

## 9. Calibration drift and fail-safe fallback

A previously qualified calibrator can become stale.

`evaluation/cross_profile_calibration_drift.py` binds a calibration-drift reference to the exact calibrator and qualification receipt, then checks:

- qualification age;
- minimum live-score evidence;
- population stability index over calibrated-probability histograms;
- Jensen-Shannon divergence;
- labeled Brier and ECE when current labels are available.

Blocking evidence produces `requalify_rrf_only`. `tools/current_cross_profile_policy.py` is the highest-authority learned-fusion serving path and requires an exact current `calibrated_ok` drift decision for every participating calibrator. Otherwise it refuses calibrated fusion so the caller can use RRF-only fallback.

## 10. Canonical persistence

`training/cross_profile_artifact_io.py` persists the small non-PyTorch artifacts used by this stack:

- isotonic calibrators;
- calibration qualification receipts;
- pointwise/listwise training states;
- learned weight artifacts;
- pointwise/listwise promotion receipts.

Persistence uses:

- schema-discriminated canonical JSON;
- envelope SHA-256;
- bounded reads;
- atomic same-directory replacement;
- fsync where supported;
- symlink/reparse-point rejection;
- type-specific reconstruction through existing validators.

It does not deserialize arbitrary Python objects.

## 11. Quality observability

`evaluation/cross_profile_observability.py` maps calibration and fusion evidence into the shared `MetricObservation` contract. Exported dimensions use profile/artifact digests and metric-family identifiers; raw queries, candidate text, and document content are not observability dimensions.

These metrics can participate in normal `QualitySnapshot`, SLO, comparison, and dashboard workflows.

## 12. Randomized policy comparison

Offline metric changes and descriptive counterfactual retrieval diffs do not by themselves identify a user preference effect.

`evaluation/retrieval_interleaving.py` therefore provides deterministic-replayable randomized team-draft interleaving for same-query policy comparison. It records contributor ownership per displayed position and can aggregate engagement credit using preference rates, Wilson intervals, and an exact sign test.

`orchestration/retrieval_interleaving_journal.py` provides owner-scoped durable experiment registration, immutable impressions/outcomes, idempotent replay, cross-owner rejection, and complete-evidence export.

`evaluation/retrieval_interleaving_promotion.py` gates experiments on minimum traffic, decisive comparisons, tie fraction, candidate preference, confidence bounds, and sign-test evidence. `evaluation/retrieval_interleaving_governance.py` seals the exact impression/outcome digest pairs into the promotion authority receipt.

`evaluation/interleaving_observability.py` exports aggregate experiment metrics into the shared quality dashboard without promoting query/source identities to observability dimensions.

Randomized interleaving is an experiment design, not an unconditional causal claim. Its interpretation still depends on correct traffic eligibility, stable measurement, experiment integrity, and the stated randomization assumptions.

## Source versus execution boundary

All components above are source implementations. This work does **not** imply that:

- real calibration labels were collected;
- real calibrators or fusion weights were fit;
- datasets or model artifacts were downloaded;
- tests or benchmarks were executed;
- online traffic was interleaved;
- a production policy was promoted;
- measured nDCG, Brier, ECE, PSI, click preference, or latency values exist.

Those remain execution/artifact activities and must produce real evidence before the corresponding promotion or serving paths can succeed in production.
