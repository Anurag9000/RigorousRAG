# RigorousRAG advanced capability expansion — 2026-08-09

This document records the capability expansion implemented directly on `main` after the exact-head compatibility baseline was restored. It is an implementation/audit record, **not** a release-readiness declaration. Release readiness still requires the repository's exact-head verification matrix to pass on the final unchanged SHA.

## Proven compatibility baseline

Before this expansion, exact-head run `31297259618` passed all 16 registered jobs on commit `812e4d358530dc3bbd8a8c4f95051f0565153a4e`:

- workflow registration smoke;
- Linux full test/coverage lanes on Python 3.10, 3.11 and 3.12;
- Windows classic-storage regressions on Python 3.10 and 3.12;
- Compose/container build;
- release-lock generation, verification and hash-only installation on Ubuntu, macOS and Windows for Python 3.10, 3.11 and 3.12.

The expansion below must receive its own exact-head proof; the earlier green baseline is provenance, not evidence for later commits.

## P1 — durable compaction-recovery evidence

Added a dedicated durable compaction-recovery journal in the graph-compaction SQLite database. Recovery now records an exact, operator-bound intent before repairing interrupted compaction receipts, supports crash/retry accounting, records bounded failure types instead of exception text, accounts for completed versus already-completed jobs, and seals a deterministic terminal receipt digest. Exact reconciliation-report and recoverable-job confirmation remains mandatory. CLI status/list/recover surfaces are privacy-safe and do not echo arbitrary operator reason text.

## P2 — migration cutover fencing

The existing cutover preparation journal already had expiring leases and takeover after expiry. The missing guarantee was monotonic fencing. Each claim now increments a persistent fencing token, terminal transitions require the exact live token in addition to worker identity and lease validity, runtime orchestration propagates that token, and existing unfenced SQLite journals are migrated in place. A same-named stale worker cannot complete a newer claim with its older token.

## Retrieval architecture expansion

Added model-agnostic primitives and a bounded advanced retrieval pipeline for:

- per-component logit temperature/bias calibration;
- calibrated weighted fusion;
- SPLADE-style sparse expansion similarity;
- ColBERT-style MaxSim late interaction;
- multi-vector max/mean/top-mean aggregation;
- query/budget/uncertainty-aware Matryoshka dimension selection;
- optional SPLADE and late-interaction adapter contracts;
- dense + BM25 + optional sparse expansion + optional late interaction + optional reranker fusion;
- MMR/source diversity after calibrated fusion;
- safe fallback to base signals when optional advanced adapters fail.

Existing embedding profile support, including BGE-M3 dense/sparse/multi-vector modes and explicit model-adapter registration, is retained rather than duplicated.

## Adaptive routing governance

The existing route benchmark already measured selected versus oracle route outcomes, regret, cost and latency. Added the governance layer needed to use those results operationally:

- exact paired-case policy comparisons;
- selected-route distribution measurement;
- Jensen-Shannon route-distribution shift;
- promotion gates for success, route accuracy, regret, cost, latency and shift;
- deterministic comparison/decision digests;
- explicit hold and rollback recommendations.

Added a separate uncertainty-aware stopping policy that can stop, continue, escalate or abstain based on evidence sufficiency, answer confidence, agent disagreement, contradiction risk, uncertainty, marginal improvement, currentness requirements and remaining budget.

## Scientific graph depth

Kept the authoritative evidence-graph schema stable and added derived scientific semantics:

- explicit `valid_from`, `valid_to` and `retracted_at` interpretation;
- active/not-yet-valid/expired/retracted statuses;
- conservative retraction-dependency risk propagation over explicit dependency relations;
- deterministic scientific hyperedges covering claims, evidence, methods, datasets and results;
- degree-normalized hypergraph projection for downstream GNN experiments;
- deterministic temporal replay across strictly increasing `as_of` points with state-transition evidence.

These are derived signals. They do not rewrite authoritative graph identities or graph payloads.

## Multimodal evidence lineage

Added immutable page-coordinate evidence regions for text, tables, figures, captions, charts and equations. Region identity binds owner/document/source digest/page/kind/normalized bounding box/content digest/extractor. Raw extracted OCR/layout text is not stored in the derived region object. Added deterministic coordinate citation IDs, overlap deduplication, extractor contracts, bounded extractor iteration and fail-closed type validation.

## Evaluation and experiments

Added a governed dataset registry covering the requested core evaluation families: SciFact, NFCorpus, FiQA, TREC-COVID, ArguAna, CQADupStack, HotpotQA, MuSiQue, plus legal/financial/biomedical adapter targets CUAD, FinQA and PubMedQA.

Added deterministic evaluation tools for:

- paired bootstrap confidence intervals;
- paired sign-flip permutation significance tests;
- Brier score and expected calibration error;
- selective-risk/coverage curves;
- counterfactual citation decoys;
- poisoned-metadata ranking overlap and recall drop;
- long-context position bias;
- duplicate/stale document-version retrieval;
- expert-review agreement, normalized disagreement entropy, adjudication confidence and pairwise Cohen's kappa;
- deterministic lexical hard-negative mining.

## Context packing and deployment economics

Added risk-aware evidence packing under explicit token and per-source budgets. Marginal utility combines relevance, evidence strength, retraction risk, source-diversity bonus and redundancy penalty. Added quality/cost/latency Pareto-frontier construction and budget-constrained deployment selection so a retrieval policy cannot win only by spending more.

## Freshness, multilingual routing and continual adaptation

Added:

- exponential half-life freshness scoring;
- separate stale-generation penalties;
- current-generation/freshness summaries for adaptive routing;
- Unicode-script signals for multilingual and code-switched queries without pretending to perform language identification;
- multilingual-model and lexical-fallback routing signals;
- population-stability-index drift measurement;
- stable/shadow-rebuild/urgent-rebuild decisions driven by distribution shift, retrieval-quality drop, stale-document fraction and update failures.

## Migration compatibility and cache correctness

The existing migration benchmark remains the authoritative paired quality/resource benchmark. Added complementary compatibility guards for:

- paired nearest-neighbor overlap and rank displacement when embedding dimensions/models change;
- dimension-change reporting without assuming equal vector dimensionality;
- owner/generation/profile/retrieval-config-bound cache keys;
- fail-closed cache reuse after any cutover-relevant identity change.

## Distributed production hardening

The ingestion job store already had durable retry timing/backoff and terminal failure states. Added missing production primitives:

- digest-only owner-scoped dead-letter records, never raw payload storage;
- expiring dead-letter replay leases with monotonic fencing;
- replay receipt digests and exact-confirm abandonment;
- deterministic admit/defer/shed backpressure decisions;
- closed/open/half-open circuit-breaker transitions;
- availability, error-rate, p50/p95/p99 latency, latency-SLO compliance and error-budget burn evidence.

## Intentionally remaining boundaries

This expansion deliberately does not claim that optional external model weights, third-party benchmark corpora, production OCR engines, a production cutover adapter, disaster-recovery infrastructure or deployment-specific SLO dashboards exist merely because interfaces/evaluators now exist. Those require environment-specific artifacts and execution evidence.

The next audit should prioritize: production migration execution/cutover adapter proof; concrete SPLADE/ColBERT/multilingual model adapters and model-card/version governance; image-text embedding and chart entailment adapters; contradiction/retraction-aware retrieval integration; online policy shadow traffic and rollback state; tenant quota/admission persistence; DR/failover exercises; retrieval cache implementation using the new cutover-safe key contract; continual embedding/index adaptation experiments; expert adjudication workflows; benchmark acquisition/version manifests; and full final exact-head CI evidence.
