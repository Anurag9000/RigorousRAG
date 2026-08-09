# RigorousRAG advanced capability expansion — 2026-08-09

This document records the capability expansion implemented directly on `main` after the exact-head compatibility baseline was restored. It is an implementation/audit record, **not** a release-readiness declaration. Release readiness still requires the repository's exact-head verification matrix to pass on the final unchanged SHA.

## Proven compatibility baseline

Before this expansion, exact-head run `31297259618` passed all 16 registered jobs on commit `812e4d358530dc3bbd8a8c4f95051f0565153a4e`:

- workflow registration smoke;
- Linux full test/coverage lanes on Python 3.10, 3.11 and 3.12;
- Windows classic-storage regressions on Python 3.10 and 3.12;
- Compose/container build;
- release-lock generation, verification and hash-only installation on Ubuntu, macOS and Windows for Python 3.10, 3.11 and 3.12.

After the expansion and correction of the calibration-regression fixture, exact-head run `31302042911` passed the same 16/16 matrix on commit `01e5008909cf0e8b435e974ca9f83533d6f69856`. All three Linux lanes passed the complete 1,930-test suite.

The cache/governed-retrieval head `3edf570eaa11cf8a7558a219b459c68009d0e9ce` subsequently passed exact-head run `31302611641` 16/16. The adaptive-policy/tenant-quota head `2ed570bbeebc8d1e29f1e66c0cc3ae88bfaa4f3e` subsequently passed exact-head run `31302915963` 16/16. These proofs cover the expansion through `2ed570b`; subsequent capability commits require their own unchanged-head proof.

## P1 — durable compaction-recovery evidence

Added a dedicated durable compaction-recovery journal in the graph-compaction SQLite database. Recovery now records an exact, operator-bound intent before repairing interrupted compaction receipts, supports crash/retry accounting, records bounded failure types instead of exception text, accounts for completed versus already-completed jobs, and seals a deterministic terminal receipt digest. Exact reconciliation-report and recoverable-job confirmation remains mandatory. CLI status/list/recover surfaces are privacy-safe and do not echo arbitrary operator reason text.

## P2 — migration cutover fencing

The existing cutover preparation journal already had expiring leases and takeover after expiry. The missing guarantee was monotonic fencing. Each claim now increments a persistent fencing token, terminal transitions require the exact live token in addition to worker identity and lease validity, runtime orchestration propagates that token, and existing unfenced SQLite journals are migrated in place. A same-named stale worker cannot complete a newer claim with its older token.

## Concrete local cutover execution

Added a concrete single-host cutover adapter over the existing authoritative vector, sparse and generation stores. It consumes the already validated migration shadow rather than rebuilding target rows during cutover, binds the exact task/profile/content/artifact identity, captures source vector embeddings before mutation, verifies source identity again immediately before visibility, publishes the validated precomputed target vectors, replaces the sparse generation, and advances the append-only generation pointer only after both stores succeed.

The adapter performs local compensation inside the visibility commit if vector, sparse or generation publication fails. For post-visibility saga faults it restores the captured source embeddings, sparse snapshot and authoritative generation, then verifies the prepared source snapshot digests and captured embeddings. Tests cover successful publication, post-visibility rollback, source drift, and dimensional incompatibility.

This adapter intentionally fails **before visibility** when target vector dimensionality differs from the current physical Chroma collection. The current repository still needs a blue/green physical-collection registry and atomic collection-pointer cutover before dimension-changing embedding migrations can be called production-ready. The adapter is therefore a concrete single-host same-dimension production path, not distributed or dimension-changing cutover proof.

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

## Adaptive routing governance and operational state

The existing route benchmark already measured selected versus oracle route outcomes, regret, cost and latency. Added the governance layer needed to use those results operationally:

- exact paired-case policy comparisons;
- selected-route distribution measurement;
- Jensen-Shannon route-distribution shift;
- promotion gates for success, route accuracy, regret, cost, latency and shift;
- deterministic comparison/decision digests;
- explicit hold and rollback recommendations.

Added a separate uncertainty-aware stopping policy that can stop, continue, escalate or abstain based on evidence sufficiency, answer confidence, agent disagreement, contradiction risk, uncertainty, marginal improvement, currentness requirements and remaining budget.

Added a durable SQLite policy-state journal. Each owner has at most one promoted policy; candidate revisions enter shadow state against the exact promoted baseline, shadow evidence is immutable within a revision and bound to comparison/metrics digests, promotion requires the exact eligible decision digest, and rollback requires the exact promoted candidate and restores the superseded baseline revision. Policy revisions are monotonic and the store persists only bounded policy IDs/digests and governance evidence—not prompts or query text.

## Scientific graph depth

Kept the authoritative evidence-graph schema stable and added derived scientific semantics:

- explicit `valid_from`, `valid_to` and `retracted_at` interpretation;
- active/not-yet-valid/expired/retracted statuses;
- conservative retraction-dependency risk propagation over explicit dependency relations;
- deterministic scientific hyperedges covering claims, evidence, methods, datasets and results;
- degree-normalized hypergraph projection for downstream GNN experiments;
- deterministic temporal replay across strictly increasing `as_of` points with state-transition evidence.

These are derived signals. They do not rewrite authoritative graph identities or graph payloads.

Added a governed graph-search surface alongside the legacy lexical search. It requires an explicit `as_of`, always excludes evidence that is not temporally active, and either excludes active evidence above an explicit conservative retraction-risk threshold or deterministically penalizes its lexical score. Retraction source IDs remain derived provenance; the authoritative graph is not mutated.

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

Added the corresponding durable SQLite retrieval cache. Cache rows are keyed by the full cutover-safe identity and persist only bounded result handles, scores and content digests—not snippets or retrieved document text. Reads verify canonical payload schema, timing identity and result digest; live-key collisions fail closed; expiry, owner invalidation and pre-generation invalidation are explicit and deterministic.

## Distributed production hardening

The ingestion job store already had durable retry timing/backoff and terminal failure states. Added missing production primitives:

- digest-only owner-scoped dead-letter records, never raw payload storage;
- expiring dead-letter replay leases with monotonic fencing;
- replay receipt digests and exact-confirm abandonment;
- deterministic admit/defer/shed backpressure decisions;
- closed/open/half-open circuit-breaker transitions;
- availability, error-rate, p50/p95/p99 latency, latency-SLO compliance and error-budget burn evidence.

Added durable tenant quota/admission accounting. Per-owner quota configuration controls requests, work units, concurrent inflight reservations, window duration and lease duration. Reservations are atomic, have monotonic fencing tokens, expire automatically to release abandoned capacity, and require the exact live token for renewal/commit/release. Committed usage is windowed and owner-isolated; reservation records carry IDs and numeric accounting only, never request payloads.

## Intentionally remaining boundaries

This expansion deliberately does not claim that optional external model weights, third-party benchmark corpora, production OCR engines, blue/green dimension-changing vector cutover, distributed disaster-recovery infrastructure or deployment-specific SLO dashboards exist merely because interfaces/evaluators now exist. Those require environment-specific artifacts and execution evidence.

The next audit should prioritize: blue/green physical vector-collection registry and atomic collection-pointer cutover; concrete SPLADE/ColBERT/multilingual model adapters and model-card/version governance; image-text embedding and chart entailment adapters; DR/failover snapshot/restore exercises; continual embedding/index adaptation experiments; expert adjudication workflows; benchmark acquisition/version manifests; and full final exact-head CI evidence.
