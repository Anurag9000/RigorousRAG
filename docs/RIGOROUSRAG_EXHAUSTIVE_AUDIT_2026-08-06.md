# RigorousRAG exhaustive mission, implementation audit and forward plan

Audit date: 2026-08-06

Audited branch: `main`

Repository policy: direct commits to `main`; no active feature branches or new pull requests.

## 1. Mission reconstructed from the original audit and continuation work

The project mission is not merely to add more retrievers. It is to turn the repository into a reproducible, tenant-safe, provenance-complete, scientifically defensible RAG research and production platform. The complete requested program is grouped below so that implementation claims can be audited against explicit objectives.

### 1.1 Repository-wide engineering and correctness

- Read and audit every source, test, configuration, workflow and documentation surface.
- Correct numerical, ranking, parsing, pagination, identity, concurrency and persistence defects.
- Replace hidden or ambiguous behavior with typed, bounded and testable contracts.
- Preserve backward-compatible defaults where safety and correctness allow it.
- Make optional dependencies fail explicitly and locally instead of corrupting unrelated paths.
- Keep code, tests, examples, configuration, documentation and status ledgers synchronized.
- Require deterministic identifiers, canonical JSON, finite numeric values and bounded inputs/outputs.
- Add adversarial, corruption, retry, crash-recovery, path-identity and cross-platform tests.
- Make exact-head CI, clean installation, lock generation, packaging and container behavior reproducible.

### 1.2 Security, tenancy, privacy and retained evidence

- Derive tenant identity only from server-controlled authentication state.
- Scope every source, vector, sparse posting, graph, job, trace and operator action by owner.
- Reject symlinks, reparse points, path replacement and database identity changes.
- Bound uploads, archives, pages, pixels, OCR, metadata, provider payloads and generated output.
- Treat retrieved/provider content as untrusted evidence rather than instructions.
- Harden SSRF, redirects, DNS/peer validation, proxies, MIME, response sizes and deadlines.
- Mask credentials, contact data, paths, control characters and privacy-sensitive text before persistence or display.
- Keep structural provenance distinct from semantic support and scientific truth.
- Provide privacy-safe operator views that expose hashes, counts, states and reason codes rather than raw evidence.

### 1.3 Authoritative ingestion and lifecycle

- Parse PDF, DOCX, text and OCR inputs through one bounded privacy-finalized boundary.
- Assign stable owner/document/content identities.
- Coordinate retained source, vector, sparse and generation stores.
- Use durable journals, leases, retries, compensation and startup reconciliation.
- Support replacement, deletion, restoration, drift detection, repair and exact-confirmation cleanup.
- Build isolated migration/shadow indexes, validate them, benchmark them and gate cutover.
- Preserve rollback identities and prevent publication of mixed or unvalidated generations.

### 1.4 Retrieval architectures and ranking

- Dense retrieval with explicit embedding-profile governance.
- Fielded BM25 and persistent sparse retrieval.
- Hybrid fusion using weighted scoring and reciprocal-rank fusion.
- MMR/diversity selection, source caps, filters and complete component traces.
- Cross-encoder, listwise, late-interaction and neural sparse reranking interfaces.
- Query expansion, HyDE, step-back, decomposition and multi-query policies.
- Parent-child, sentence-window, contextual compression and metadata-aware retrieval.
- Multi-stage cascades governed by quality, latency, memory and cost budgets.
- Abstention and explicit evidence-gap reporting.

### 1.5 Adaptive, corrective and agentic RAG

- Deterministic query intent/complexity analysis and route selection.
- Evidence sufficiency/confidence estimation.
- Bounded corrective retrieval attempts with failure containment.
- Multi-hop subquestion DAGs with loop and branching limits.
- Heterogeneous uploaded/web/scholarly routes.
- Critique/revise behavior without hidden provenance mutation.
- Versioned route policies, calibration, promotion and rollback.
- Privacy-safe trace storage, retention, export and diagnostics.

### 1.6 GraphRAG and evidence graphs

- Typed document, section, entity, claim, method, dataset, result and citation nodes.
- Typed provenance, support, contradiction, agreement, lineage and reference edges.
- Generation-bound graph construction and fail-closed authoritative reads.
- Cross-document graph sets and bounded multi-hop path search.
- Derived community summaries with explicit versioning and provenance.
- Reconciliation jobs, leases, retries, audit classifications and retention plans.
- Safe compaction that cannot delete current or authoritative evidence.

### 1.7 Scientific and domain evidence intelligence

- Scientific claim, method, intervention, outcome, dataset, limitation and result extraction.
- PICO/PECO and study-design schemas.
- Numerical results, units, intervals and effect sizes.
- Claim-evidence-entailment records with uncertainty and reviewer correction.
- Contradiction clusters, replication links, citation-context analysis and lineage.
- Risk-of-bias/evidence-quality checklists.
- Governed extractor registry, benchmark promotion, rollback and audit trails.
- LegalRAG/domain-specific schemas without presenting structural extraction as legal or scientific truth.

### 1.8 Evaluation, datasets and experiments

- BEIR-style normalized corpus/query/qrels contracts.
- Retrieval, citation, support, abstention, calibration, route and multi-hop metrics.
- Deterministic manifests, resumable matrices, repeat seeds and immutable results.
- Paired confidence intervals, non-inferiority gates and practical-gain thresholds.
- Dataset cards, licenses, checksums, leakage checks and split manifests.
- Domain datasets including SciFact, NFCorpus, TREC-COVID, FiQA, ArguAna, HotpotQA, 2WikiMultiHopQA, MuSiQue, PubMedQA, BioASQ and structure-aware benchmarks where licensing permits.
- Synthetic corruption, prompt-injection, tenant-isolation and provenance adversarial suites.
- Latency, throughput, memory, storage, provider cost and energy measurements.

### 1.9 Operations, observability and distributed production

- Privacy-conscious traces, metrics, dashboards, SLOs and failure taxonomies.
- Backup/restore, migration, disaster-recovery and exact fault-injection drills.
- Durable queues, distributed leases, idempotency keys and transactional outbox/saga patterns.
- Encrypted/versioned object storage, key management and secure deletion.
- Dedicated model serving, batching, GPU scheduling, circuit breakers and resource budgets.
- Parser isolation, malware scanning, egress firewall, TLS ingress and secret management as deployment controls.

### 1.10 Git and delivery policy

- Preserve all useful progress and history on `main`.
- Commit new work directly to `main` rather than opening new PRs.
- Keep only `main` as an active branch.
- Do not force-push or rewrite preserved remediation history.
- Treat exact-head workflow success on one unchanged commit as the release gate.

## 2. Pull-request and branch audit

### 2.1 Historical PRs

- PR #1, `Harden RigorousRAG and rebuild the evidence pipeline`, is the substantive 751-commit remediation history.
- PRs #2, #3 and #4 were disposable verification branches used to trigger exact-head checks while PR #1 stopped producing useful synchronization runs.
- All four PRs are closed and their commits are preserved in `main` history.
- Closed PR records are audit metadata, not active branches. Deleting them would remove useful provenance and is not required to satisfy the single-active-branch policy.

### 2.2 Active branches

- `main` is the only active branch observed on 2026-08-06.
- No branch cleanup was required in this continuation.
- All changes in this continuation were committed directly to `main`.

## 3. What is implemented on current `main`

The implementation ledger dated 2026-08-02 is useful but stale. Later direct-to-main commits completed several items it still labels open. The following status reflects repository code and commit history observed through 2026-08-06.

### 3.1 Baseline hardening from PR #1 — implemented

- Tenant-safe API and bounded execution contracts.
- Safe uploads, retained-source identity and durable ingestion.
- Bounded parsing, OCR, privacy masking and visual-evidence checks.
- Owner-scoped retrieval and canonical citation provenance.
- Provider/network/browser/CLI/deployment hardening.
- Deterministic lock and exact-head workflow infrastructure.
- Extensive unit, integration, storage, release and platform tests.

### 3.2 Retrieval and evaluation foundation — implemented

- Typed bounded candidates and component traces.
- BM25, dense retrieval, weighted fusion, RRF and MMR.
- Heuristic and optional cross-encoder reranking.
- BEIR-style loading and retrieval/citation metrics.
- Deterministic experiment matrices and resumable immutable results.
- Offline benchmark entrypoints.

### 3.3 Persistent hybrid lifecycle — substantially implemented

- Embedding profile registry and owner-scoped persistent sparse index.
- Vector/sparse/generation snapshots, compensation and reconciliation.
- Privacy-finalized document service and retained-source lifecycle outbox.
- Leases, retries, startup reconciliation, operator status/retry/reset tooling.
- Isolated migration/shadow construction, benchmark evidence, promotion policies and non-mutating cutover preflight.
- Corpus dense/sparse/hybrid retrieval with generation validation and provenance traces.

### 3.4 Adaptive and multi-hop RAG — implemented foundation

- Query analysis, route selection, evidence-sufficiency decisions and bounded corrective retrieval.
- Query decomposition, subquestion DAGs and heterogeneous route execution.
- Privacy-safe traces and offline route/multi-hop benchmark contracts.
- Calibration, risk-coverage and abstention analysis primitives.

### 3.5 GraphRAG/evidence graph — implemented foundation

- Typed graph schemas and append-only generation-scoped graph store.
- Exact graph digests and current pointers.
- Durable graph reconciliation job journal with leases/retries.
- Fail-closed authority assessment against the authoritative generation store.
- Authority-aware graph CLI reads.
- Cross-document graph sets, bounded search and derived graph operations.
- Privacy-safe graph-job audit and conservative retention planning.

### 3.6 Scientific-claim governance — implemented foundation

- Scientific claim extraction and graph integration.
- Reviewer correction/custody records and append-only governance surfaces.
- Extractor registry, evaluation, benchmark promotion and rollback contracts.
- Operator/runtime/CLI tests and documentation for governed promotion.

## 4. New implementation completed in this continuation

### 4.1 Durable evidence-graph payload compaction

New modules and contracts:

- `tools/evidence_graph_compaction.py`
- `tools/evidence_graph_compaction_runtime.py`
- `tests/unit/test_evidence_graph_compaction.py`
- governed commands in `tools/evidence_graph_operations_cli.py`
- `EVIDENCE_GRAPH_COMPACTION_DB_PATH` in `config/evidence_graph.env.example`

Safety and behavior:

- Requires an exact retention-plan digest.
- Requires the exact complete set of candidate job IDs.
- Revalidates job identity and candidate age against the confirmed plan.
- Refuses authoritative-current graph jobs.
- Refuses the current graph-store generation.
- Verifies the historical graph digest before deletion.
- Persists a durable intent before deleting a graph payload.
- Resumes safely after a crash between payload deletion and receipt completion.
- Is idempotent after successful completion.
- Refuses unexplained missing historical graphs without a prior durable intent.
- Preserves authoritative vector, sparse and generation state.
- Preserves the smaller graph-job journal rows as the audit trail.
- Records cancelled terminal jobs without attempting graph deletion.
- Supports owner/phase-filtered privacy-safe receipt inspection.

Focused tests committed:

1. verified historical payload deletion and completed receipt;
2. cancelled-job audit-only behavior;
3. exact plan and job confirmations;
4. crash-resume after deletion-before-receipt;
5. authoritative-current refusal;
6. completed-operation idempotency;
7. unexplained missing-payload refusal;
8. owner/phase receipt filtering.

## 5. Remaining work, ordered by dependency and risk

Items below are not complete merely because adjacent interfaces exist. Each requires source, adversarial tests, benchmarks, configuration, operator docs, migration/rollback behavior and exact-head verification.

### Priority 0 — exact-head proof and release integrity

- Run the complete authoritative workflow on one unchanged `main` SHA.
- Pass Linux Python 3.10–3.12 full suites and coverage.
- Pass Windows storage suites.
- Pass Compose/container validation.
- Pass Linux/Windows/macOS lock generation and hash-required dry installation.
- Re-audit generated artifacts and documentation after the final green SHA.

### Priority 1 — lifecycle and retention completion

- Retention/compaction for completed ingestion/lifecycle operations.
- Privacy-safe lifecycle audit exports and cross-journal correlation.
- Bounded retention for adaptive traces, migration shadows, benchmark reports, preflight records and rollback artifacts.
- Protected complete rollback-artifact storage with encryption/key rotation and secure deletion.
- Fault injection across every vector/sparse/manifest/registry/cleanup/graph-compaction phase.
- Multi-process leader election or periodic reconciliation.

### Priority 2 — governed migration execution

- Durable cutover journal with exclusive leases and idempotency keys.
- Atomic or compensating vector+sparse+generation publication.
- Validation of the newly published authoritative generation before commit.
- Automatic rollback on every failed publication/validation phase.
- Exact rollback identity verification and old-state retention until verification succeeds.
- Worker pause/resume/cancel semantics.

### Priority 3 — retrieval quality and filtering

- Benchmark-calibrated dense/sparse fusion weights.
- Independent-corpus RRF and explicit per-source/document caps.
- Date, MIME, field, section, authority and provenance filters.
- Multi-stage reranker cascades with latency/cost budgets.
- SPLADE-compatible sparse neural retrieval.
- ColBERT/late-interaction retrieval.
- Cross-encoder and listwise reranker registry with promotion/rollback.
- Connected dense/sparse/hybrid benchmark reports on representative corpora.

### Priority 4 — adaptive-policy promotion

- Representative calibration of evidence thresholds and route coefficients.
- Versioned runtime calibrator installation by domain/corpus/profile.
- Validated domain classifier and domain-specialized policies.
- Connected uploaded/web/scholarly route experiments.
- Repeated-seed policy ablations, confidence intervals and promotion gates.
- Trace retention, export, privacy budgets and dashboards.

### Priority 5 — graph and scientific depth

- Production entity linking and relation extraction adapters.
- Temporal/versioned contradiction and agreement clusters.
- Community detection/summary promotion with benchmark evidence.
- PICO/PECO, study design, numerical result, unit and effect-size extraction.
- Risk-of-bias/evidence-quality schemas with human review.
- Replication links, protocol-to-result traceability and citation-context analysis.
- Domain benchmark promotion for scientific extractors and graph builders.
- Explicit uncertainty calibration and abstention for extraction and entailment.

### Priority 6 — multimodal and structure-aware ingestion

- Layout-preserving reading order and coordinate provenance.
- Table cell grids and table-aware retrieval.
- Formula/LaTeX extraction and equation references.
- Figure/caption/panel/in-text mention linking.
- OCR coordinate/confidence propagation.
- Visual embeddings and cross-modal late interaction.
- Modality-aware chunking and cross-modal citations.
- External parser sandbox and malware scanning integration.

### Priority 7 — datasets and experiment science

- Licensed adapters and cards for the roadmap datasets.
- Structure-aware table/figure/formula benchmarks.
- Leakage and contamination checks.
- Repeated-seed ablations for chunking, fusion, routing, reranking and graph policies.
- Paired/bootstrap/permutation procedures and multiple-comparison controls.
- Wall-clock latency, memory, storage, billing and energy measurements.
- Regression histories and machine-readable model/index/system cards.

### Priority 8 — distributed production

- Shared SQL registries and distributed queues/leases.
- Transactional outbox/saga across external stores.
- Encrypted versioned object storage.
- Distributed rate limits and idempotency.
- Dedicated model serving and GPU-aware batching/scheduling.
- OpenTelemetry/Prometheus integration and privacy-conscious SLO dashboards.
- Backup/restore and disaster-recovery drills.
- Egress firewall, TLS ingress, secret manager and deployment isolation.

## 6. Additional recommended architectures and experiments

These additions are valuable only when introduced behind governed interfaces and promoted by benchmark evidence.

### Retrieval models

- Modern multilingual dense profiles for Indic and cross-lingual retrieval.
- Domain adapters for scientific, biomedical and legal corpora.
- SPLADE-family learned sparse retrieval.
- ColBERT-family late interaction.
- Matryoshka embeddings for dimension/latency ablations.
- Multi-vector document representations for tables, figures and long sections.

### Reranking and context construction

- Lightweight cross-encoder, large cross-encoder and LLM-listwise cascade policies.
- Contextual retrieval/document-prefix augmentation.
- Learned source-cap and diversity policies.
- Evidence packing as a constrained optimization problem over support, novelty and token cost.
- Citation-aware sentence selection and claim-conditioned compression.

### Graph and reasoning

- Graph expansion policies trained or calibrated on multi-hop benchmarks.
- Temporal knowledge-graph snapshots and contradiction evolution.
- Hyperedges for claims supported jointly by multiple evidence units.
- Causal/methodological relation types kept separate from textual entailment.
- Graph neural reranking only as an optional benchmarked derived layer, never an authority source.

### Evaluation

- Retrieval robustness under OCR noise, section reordering, duplicate evidence and stale generations.
- Counterfactual citation tests where fluent unsupported answers must abstain.
- Calibration by domain, language, document age and evidence modality.
- Long-context lost-in-the-middle and citation-position studies.
- Adversarial source-conflict, prompt-injection and poisoned-metadata benchmarks.
- Human expert review protocols with inter-rater agreement and adjudication.

## 7. Verification state and non-claims

- Focused local compaction harnesses passed before commit, and eight focused repository tests are now committed.
- This document does not claim that the complete repository suite or exact-head matrix is green.
- This document does not claim distributed exactly-once behavior.
- This document does not claim certified de-identification, malware resistance or parser sandboxing.
- This document does not claim that graph structure or extracted claims establish scientific or legal truth.
- Current release readiness remains conditional on one complete exact-head workflow success on an unchanged `main` commit.

## 8. Continuation rule

Future implementation should consume the ordered backlog above, commit coherent code+tests+configuration+documentation units directly to `main`, inspect the resulting exact-head checks, fix failures on `main`, and update this audit rather than creating parallel branches or unsupported completion claims.
