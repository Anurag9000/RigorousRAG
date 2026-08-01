# Capability implementation status

Last updated: 2026-08-01

This is the authoritative capability-expansion ledger for `main`. A checked implementation item means source and focused tests/contracts were committed. It does not mean the complete exact-head release matrix passed.

## Repository policy

- Development is committed directly to `main`.
- No feature branches or pull requests are used.
- Implementation, tests, documentation, configuration and TODO state must remain aligned.
- Release claims require the authoritative exact-head workflow on one unchanged `main` SHA.

## Wave 1 — retrieval and evaluation foundation

### Implemented

- [x] Typed bounded retrieval candidates and component traces.
- [x] Candidate-pool BM25 with document-frequency counting.
- [x] Reciprocal-rank and weighted-score fusion.
- [x] MMR relevance/diversity selection.
- [x] Dense, lexical and candidate-pool hybrid modes.
- [x] Heuristic and optional cross-encoder rerankers.
- [x] Backward-compatible dense defaults.
- [x] BEIR-style corpus/query/qrels loader.
- [x] Retrieval and citation metrics.
- [x] Deterministic experiment matrices and immutable resumable results.
- [x] Offline BM25 benchmark CLI.

### Focused verification evidence

- 12 Wave 1 tests passed in the local constrained harness.
- Compilation and AST parsing passed for the Wave 1 changed surfaces.
- Ruff and the complete repository matrix were not established in that local environment.

## Wave 2A — embedding governance and persistent sparse index

### Implemented

- [x] Declarative embedding-profile registry with seven built-in families.
- [x] Compatibility aliases and stable profile fingerprints.
- [x] Strict operator-defined profiles.
- [x] Persistent owner-scoped fielded sparse index.
- [x] Transactional replacement, optimistic generations and exact restore/delete.
- [x] Page, section, field, frequency and token-position provenance.
- [x] Strict metadata JSON, corruption refusal and path/database identity checks.

### Focused verification evidence

- 21 Wave 2A tests and 33 combined Wave 1+2A tests passed in the local constrained harness.

## Wave 2B — authoritative durable generations

### Implemented

- [x] Complete bounded vector-generation snapshots and exact restoration.
- [x] Deterministic sparse-field extraction.
- [x] Vector+sparse replacement and deletion compensation.
- [x] Append-only generation history and optimistic current pointers.
- [x] One reentrant owner/document lock across vector, sparse and manifest operations.
- [x] Active, deleted and restored durable generation states.
- [x] Profile fingerprint, content hash, vector row and sparse generation manifests.
- [x] Privacy-finalized authoritative document commit boundary.
- [x] API/durable-worker ingestion through the shared document service.
- [x] Reload-idempotent public authoritative deletion.
- [x] Raw internal vector deletion for compensation without lifecycle recursion.
- [x] Batch ingestion snapshot/restore across vector, sparse and manifest state.
- [x] Drift reconciliation for store/manifest categories.
- [x] Bounded operator scan/plan CLI.
- [x] Exact-confirmation cleanup for deleted-generation residue.
- [x] Independent runtime paths for vector, sparse and generation databases.

### Tests/contracts committed

- generation-store lifecycle, sequence, corruption and identity tests;
- vector snapshot/restore tests;
- deterministic sparse-field tests;
- three-store replacement/deletion/compensation tests;
- authoritative public/raw deletion tests;
- document-service integration tests;
- batch registry-failure restoration tests;
- reconciliation CLI tests.

### Still open in Wave 2B

- [ ] Include the retained-document registry as a fourth coordinated participant or durable outbox consumer.
- [ ] Startup reconciliation and resumable repair execution.
- [ ] Adoption/reindex workflows for existing pre-manifest documents.
- [ ] Fault-injection execution on the exact current repository head.

## Wave 2C — corpus-level hybrid retrieval

### Implemented

- [x] Independent dense and persistent sparse corpus candidate generation.
- [x] Document-level weighted fusion.
- [x] Durable current-generation validation before evidence publication.
- [x] Dense owner/content-hash/profile validation.
- [x] Sparse generation/profile validation.
- [x] Dense chunk and sparse field evidence materialization.
- [x] Page, section, field, frequency and position traces.
- [x] Bounded MMR evidence selection.
- [x] Optional second-stage reranking.
- [x] Explicit `corpus-sparse` and `corpus-hybrid` agent-tool modes.
- [x] Protected citation metadata precedence.
- [x] Partial expanded-query failure containment.

### Tests/contracts committed

- sparse-only recall beyond the dense pool;
- stale generation rejection;
- cross-owner and hash/profile mismatch rejection;
- document filter propagation;
- deleted/missing manifest rejection;
- corpus-mode routing and provenance;
- protected metadata behavior;
- partial and total expanded-query failures.

### Still open in Wave 2C

- [ ] Benchmark-calibrated fusion weights.
- [ ] Independent-corpus reciprocal-rank fusion.
- [ ] Explicit per-document/source caps.
- [ ] Date, MIME, field, section and provenance filters.
- [ ] Multi-stage reranker cascades with latency/cost budgets.
- [ ] Dense/sparse/hybrid comparative benchmark reports.

## Wave 2D — profile migration control plane

### Implemented

- [x] Profile-drift inventory from durable current generations.
- [x] Retained-source eligibility classification without path disclosure.
- [x] Deterministic migration task IDs.
- [x] Validated migration candidate and task schemas.
- [x] Durable SQLite migration journal with path/database identity checks.
- [x] Idempotent task seeding.
- [x] Expiring worker leases and renewal.
- [x] Retry ceilings and generic failure types.
- [x] Validation digests required before committed state.
- [x] Expired running and validated task recovery contracts.
- [x] Planned/failed cancellation only.
- [x] Owner verification before operator cancellation.
- [x] Inventory, seed, status and cancel CLI commands.
- [x] No execution or cutover command before shadow validation exists.

### Tests/contracts committed

- real embedding-profile alias use;
- profile-drift reason classification;
- private-path absence;
- stable task IDs;
- journal seed/claim/renew/validate/commit/fail/cancel behavior;
- lease-expiry recovery;
- retry ceilings;
- database identity replacement;
- cross-owner cancellation refusal before mutation;
- generic operator error output.

### Still open in Wave 2D

- [ ] Shadow vector and sparse stores isolated by task.
- [ ] Retained-source execution through the privacy-finalized pipeline.
- [ ] Target-profile encoder adapter construction.
- [ ] Shadow quality, provenance, count and resource validation.
- [ ] Durable shadow artifact identity.
- [ ] Atomic current-generation cutover and rollback references.
- [ ] Bounded shadow retention and cleanup.
- [ ] Crash/fault injection at every migration phase.
- [ ] Active-worker pause/resume/cancel semantics.

## Wave 3 — adaptive and corrective RAG

### Implemented

- [x] Deterministic query intent and complexity analysis.
- [x] Retrieval-mode selection for exact, comparative, temporal, quantitative, method, evidence and explanatory questions.
- [x] Evidence-count, document-diversity, score, provenance and generation signals.
- [x] Explicit sufficient, weak, insufficient and empty evidence decisions.
- [x] Corrective retrieval plans with bounded attempts and estimated cost.
- [x] Bounded execution over dense, corpus-sparse and corpus-hybrid modes.
- [x] Accumulated-evidence deduplication and total ceilings.
- [x] Per-attempt error containment and trace records.
- [x] Public adaptive uploaded-document tool and JSON-safe trace payload.
- [x] Conservative abstention after exhausted insufficient retrieval.
- [x] Reliability reports with Brier score, ECE and maximum calibration gap.
- [x] Dependency-free isotonic confidence calibration.
- [x] Risk-coverage curves and abstention-threshold selection.

### Tests/contracts committed

- exact/comparison/method/temporal routing behavior;
- evidence-sufficiency decision boundaries;
- plan attempt/cost ceilings;
- hostile input and iterator containment;
- adaptive execution success, retries and total failures;
- evidence deduplication and trace payloads;
- reliability bins and calibration metrics;
- monotonic isotonic calibration;
- tied-threshold risk-coverage behavior;
- feasible and infeasible abstention thresholds.

### Still open in Wave 3

- [ ] Representative benchmark calibration of evidence coefficients and decision thresholds.
- [ ] Versioned runtime calibrator installation and corpus/profile selection.
- [ ] Explicit domain classifier and domain-specialized policies.
- [ ] Uploaded/web/scholarly routing experiments.
- [ ] Adaptive-policy ablations, confidence intervals and promotion gates.

## Wave 4 — decomposition and multi-hop retrieval foundation

### Implemented

- [x] Validated bounded subquestion model.
- [x] Deterministic heuristic decomposition and explicit proposed-plan support.
- [x] Entity and temporal constraint extraction.
- [x] Duplicate, dangling and cyclic graph refusal.
- [x] Stable SHA-256 plan fingerprints.
- [x] Topological parallel batches and terminal-node detection.
- [x] Bounded parallel independent hops and serial dependent batches.
- [x] Worker, timeout, per-hop, dependency-evidence and total-evidence ceilings.
- [x] Per-hop failure and timeout containment.
- [x] Missing-prerequisite evidence skip policy.
- [x] Immutable hop/source/document/page lineage.
- [x] Cross-hop document/source grouping without synthetic citation creation.
- [x] Bounded lexical constraint propagation from prerequisite evidence.
- [x] Public adaptive uploaded-document multi-hop tool.
- [x] Citation and lineage separation in serialized output.
- [x] Terminal-evidence abstention.

### Focused verification evidence

- 12 focused decomposition, executor and public-tool tests passed locally.
- Python compilation passed for the three new modules and focused tests.
- Ruff was unavailable in the constrained local environment.

### Still open in Wave 4

- [ ] Strict-schema model-assisted decomposition with deterministic fallback.
- [ ] Learned plan ranking and plan-quality scoring.
- [ ] Entity resolution and normalized temporal ranges.
- [ ] Heterogeneous uploaded/web/scholarly multi-hop execution.
- [ ] Global backend/latency/cost allocation across hops.
- [ ] HotpotQA, 2WikiMultiHopQA, MuSiQue and scientific multi-document adapters.
- [ ] Decomposition, hop-recall, path-support, answer-support and citation metrics.
- [ ] Full agent/API/browser registration and integration tests.

## Verification status

The main-only exact-head workflow is configured for:

- Linux Python 3.10, 3.11 and 3.12 full suites;
- Windows classic-storage regressions on Python 3.10 and 3.12;
- Compose validation and container build;
- nine platform/Python release-lock jobs.

The repository currently has only `main`, and the historical pull requests are closed. A green complete result is not observable for the latest exact `main` head through the available connector, and the constrained execution environment cannot clone GitHub because DNS resolution fails. Release readiness is therefore not claimed.

## Permanent non-claims

- Retrieval rank is not proof of factual correctness.
- Citation presence is not proof of entailment.
- Storage-generation alignment is provenance, not truth.
- A decomposition graph is not proof that the question was decomposed optimally.
- Cross-hop evidence grouping is not proof of a shared claim.
- Learned rerankers may encode bias and require benchmark validation.
- Regex masking is not certified de-identification.
- SQLite/process-local transactions are not distributed exactly-once infrastructure.
- Scientific conclusions require source inspection, expert review and replication.
