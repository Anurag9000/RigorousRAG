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

The cache/governed-retrieval head `3edf570eaa11cf8a7558a219b459c68009d0e9ce` passed exact-head run `31302611641` 16/16. The adaptive-policy/tenant-quota head `2ed570bbeebc8d1e29f1e66c0cc3ae88bfaa4f3e` passed run `31302915963` 16/16. The concrete local cutover head `d40f4f838310cebedc9e82c4b567b8768c52ea72` passed run `31303310249` 16/16. The blue/green route-registry head `516c20645448ef73386dc6a913b518a84042f3b0` passed run `31303607270` on attempt 2: the isolated Windows classic-storage Python 3.12 lane was rerun and passed, while Linux 3.10/3.11/3.12 full suites, Windows 3.10 storage, container, registration and all nine release-lock combinations were green. These proofs cover the expansion through `516c206`; subsequent capability commits require their own unchanged-head proof.

## P1 — durable compaction-recovery evidence

Added a dedicated durable compaction-recovery journal in the graph-compaction SQLite database. Recovery records an exact operator-bound intent before repairing interrupted compaction receipts, supports crash/retry accounting, records bounded failure types instead of exception text, accounts for completed versus already-completed jobs, and seals a deterministic terminal receipt digest. Exact reconciliation-report and recoverable-job confirmation remain mandatory. CLI status/list/recover surfaces are privacy-safe and do not echo arbitrary operator reason text.

## P2 — migration cutover fencing

The existing cutover preparation journal already had expiring leases and takeover after expiry. The missing guarantee was monotonic fencing. Each claim now increments a persistent fencing token, terminal transitions require the exact live token in addition to worker identity and lease validity, runtime orchestration propagates that token, and existing unfenced SQLite journals are migrated in place. A same-named stale worker cannot complete a newer claim with its older token.

## Concrete same-dimension cutover execution

Added a concrete single-host cutover adapter over the existing authoritative vector, sparse and generation stores. It consumes the validated migration shadow rather than rebuilding target rows during cutover, binds exact task/profile/content/artifact identity, captures source vector embeddings before mutation, verifies source identity immediately before visibility, publishes precomputed target vectors, replaces the sparse generation, and advances the append-only generation pointer only after both stores succeed.

The adapter performs local compensation if vector, sparse or generation publication fails. For post-visibility saga faults it restores captured source embeddings, sparse snapshot and authoritative generation, then verifies prepared source snapshot digests and captured embeddings. Tests cover successful publication, post-visibility rollback, source drift and dimensional incompatibility.

The same-dimension adapter intentionally fails before visibility when target vector dimensionality differs from the current physical collection. Dimension-changing migrations use the separate blue/green path below.

## Blue/green physical vector routing and dimension-changing cutover

Added a durable physical vector-collection registry and append-only per-document route journal:

- immutable physical collection specifications derive from the complete embedding-profile fingerprint, model name and explicit dimensionality;
- deterministic Chroma-safe physical names replace caller-selected collection names;
- one profile fingerprint maps to one physical collection identity;
- per-owner/document routing history is append-only with monotonic revisions and authoritative generation-sequence binding;
- route heads advance through atomic compare-and-swap over expected revision, collection, profile and generation identity;
- rollback is another audited route revision and advances generation history rather than rewinding it;
- `generation_advance` records semantic source recovery on the same physical collection after append-only generation recovery;
- immediately repeated identical route operations are idempotent only while still current; superseded retries fail closed;
- physical collections cannot be retired while any current route references them, and retirement requires exact collection-ID confirmation;
- owner/document route isolation and bounded current-route/history reads are enforced;
- `VectorCollectionRouter` resolves each document through its current physical collection and caches layers by immutable collection ID.

Added a concrete dimension-changing blue/green cutover adapter. It resolves the registered target collection, loads the already validated shadow artifact, verifies exact task/profile/content/artifact identity and target dimensions, writes precomputed target embeddings to the separate physical collection, reads them back and verifies them before visibility, then advances sparse state, the append-only generation store and finally the route CAS. The source physical collection is never overwritten.

If sparse/generation publication or route CAS fails before visibility, the adapter restores source sparse/generation semantics and, when append-only recovery advanced the generation sequence, records a same-collection `generation_advance` route revision. The cutover saga supports an optional `validate_aborted_source` proof so adapters can verify exact source semantics without falsely requiring append-only sequence numbers to rewind. If a post-visibility fault occurs, the adapter appends a restored source generation and a route rollback to the source physical collection, then verifies source vector/sparse digests and route/generation coherence.

A concrete `ChromaPhysicalCollectionProvider` creates/opens deterministic dimension-specific Chroma collections. Focused tests cover a 2D→3D cutover, source-collection preservation, hidden-target discard, target unavailability, post-visibility rollback and route-CAS failure after generation publication.

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

Existing embedding-profile support, including BGE-M3 dense/sparse/multi-vector modes and explicit model-adapter registration, is retained rather than duplicated.

## Adaptive routing governance and operational state

The route benchmark already measured selected versus oracle outcomes, regret, cost and latency. Added exact paired-case comparisons, route-distribution measurement, Jensen-Shannon shift, promotion gates for success/accuracy/regret/cost/latency/shift, deterministic comparison/decision digests, and explicit hold/rollback recommendations.

Added an uncertainty-aware stopping policy that can stop, continue, escalate or abstain from evidence sufficiency, answer confidence, agent disagreement, contradiction risk, uncertainty, marginal improvement, currentness requirements and remaining budget.

Added a durable SQLite policy-state journal. Each owner has at most one promoted policy; candidate revisions enter shadow state against the exact promoted baseline, shadow evidence is immutable within a revision and bound to comparison/metrics digests, promotion requires the exact eligible decision digest, and rollback restores the superseded baseline revision. Policy revisions are monotonic and store only bounded policy IDs/digests and governance evidence—not prompts or query text.

## Scientific graph depth

Kept the authoritative evidence-graph schema stable and added derived scientific semantics:

- `valid_from`, `valid_to` and `retracted_at` interpretation;
- active/not-yet-valid/expired/retracted statuses;
- conservative retraction-dependency risk propagation over explicit dependency relations;
- deterministic scientific hyperedges covering claims, evidence, methods, datasets and results;
- degree-normalized hypergraph projection for downstream GNN experiments;
- deterministic temporal replay across strictly increasing `as_of` points with state-transition evidence.

Added governed graph search alongside lexical search. It requires explicit `as_of`, excludes temporally inactive evidence, and either excludes evidence above an explicit retraction-risk threshold or deterministically penalizes its lexical score. These are derived signals and do not rewrite authoritative graph identities.

## Multimodal evidence lineage

Added immutable page-coordinate evidence regions for text, tables, figures, captions, charts and equations. Region identity binds owner/document/source digest/page/kind/normalized bounding box/content digest/extractor. Raw extracted OCR/layout text is not stored in the derived region object. Added deterministic coordinate citation IDs, overlap deduplication, extractor contracts, bounded extractor iteration and fail-closed type validation.

## Evaluation and experiments

Added a governed dataset registry covering SciFact, NFCorpus, FiQA, TREC-COVID, ArguAna, CQADupStack, HotpotQA, MuSiQue, plus CUAD, FinQA and PubMedQA.

Added deterministic paired bootstrap confidence intervals, paired sign-flip permutation tests, Brier/ECE calibration, selective-risk curves, counterfactual citation decoys, poisoned-metadata ranking stability/recall drop, long-context position bias, duplicate/stale-version retrieval, expert-review disagreement/adjudication metrics, pairwise Cohen's kappa, and deterministic lexical hard-negative mining.

## Context packing and deployment economics

Added risk-aware evidence packing under token and per-source budgets. Marginal utility combines relevance, evidence strength, retraction risk, source-diversity bonus and redundancy penalty. Added quality/cost/latency Pareto-frontier construction and budget-constrained deployment selection so a policy cannot win only by spending more.

## Freshness, multilingual routing and continual adaptation

Added exponential half-life freshness scoring, stale-generation penalties, current-generation/freshness summaries, Unicode-script signals for multilingual/code-switched queries, multilingual-model and lexical-fallback routing signals, population-stability-index drift measurement, and stable/shadow-rebuild/urgent-rebuild decisions from distribution shift, retrieval-quality drop, stale-document fraction and update failures.

## Migration compatibility and cache correctness

The migration benchmark remains the authoritative paired quality/resource benchmark. Added paired nearest-neighbor overlap/rank displacement for embedding migrations, dimension-change reporting, owner/generation/profile/retrieval-config-bound cache keys, and fail-closed cache reuse after cutover-relevant identity changes.

Added a durable SQLite retrieval cache. Rows are keyed by full cutover-safe identity and persist only bounded result handles, scores and content digests—not snippets or retrieved document text. Reads verify canonical schema, timing identity and result digest; live-key collisions fail closed; expiry, owner invalidation and pre-generation invalidation are explicit.

## Distributed production hardening

Added digest-only owner-scoped dead letters, expiring replay leases with monotonic fencing, replay receipt digests, deterministic admit/defer/shed backpressure, circuit-breaker state transitions, and availability/error-rate/p50/p95/p99/SLO/error-budget evidence.

Added durable tenant quota/admission accounting. Per-owner configuration controls requests, work units, concurrent inflight reservations, window duration and lease duration. Reservations are atomic, have monotonic fencing tokens, expire to release abandoned capacity, and require the exact live token for renewal/commit/release. Committed usage is windowed and owner-isolated; reservation rows never store request payloads.

## Intentionally remaining boundaries

This expansion deliberately does not claim that optional external model weights, third-party benchmark corpora, production OCR/model engines, distributed disaster-recovery infrastructure or deployment-specific SLO dashboards exist merely because interfaces/evaluators exist. Those require environment-specific artifacts and execution evidence.

The next audit should prioritize: durable crash-resumable target-population intents/receipts and reconciliation for physical collections; distributed/multi-process blue/green cutover coordination; DR snapshot/export/restore/failover rehearsal; cross-profile corpus retrieval and calibrated fusion; concrete SPLADE/ColBERT/multilingual adapters with model-card/revision/fingerprint governance; image-text and chart/table entailment adapters; benchmark acquisition/version/license/hash manifests; continual drift→shadow-build→benchmark→promotion execution; durable expert adjudication workflows; and final exact-head CI plus branch/PR hygiene verification.
