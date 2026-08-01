# Capability implementation status

Last updated: 2026-08-01

This is the authoritative capability-expansion ledger for `main`. A checked implementation item means source and focused tests/contracts were committed. It does not mean the complete exact-head release matrix passed.

## Repository policy

- Development is committed directly to `main`.
- No feature branches or pull requests are used.
- Implementation, tests, documentation, configuration and TODO state remain aligned.
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
- Compilation and AST parsing passed for changed Wave 1 surfaces.

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

## Wave 2B — authoritative four-store lifecycle foundation

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
- [x] Durable retained-registry replacement/deletion outbox.
- [x] Deterministic owner/document lifecycle operation IDs.
- [x] Planned, index-committed, registry-committed, completed and failed phases.
- [x] Worker leases, renewal, release, retry ceilings and generic failure types.
- [x] Exact generation sequence/content-hash reconciliation.
- [x] Crash recovery after index commit without duplicate vector/sparse writes.
- [x] Pending deletion identity reuse after the generation sequence changes.
- [x] Private cleanup-intent journal persisted before registry mutation.
- [x] Idempotent retained-source cleanup after process crashes.
- [x] Startup reconciliation before the first RAG retrieval layer is served.
- [x] API/durable-job retained-source lifecycle coordination.
- [x] Batch retained-source copy bridge with owner/byte identity validation.
- [x] One-use context-local batch source intent and temporary private source binding.
- [x] Redundant batch registration short-circuit before a second SQLite write.
- [x] Privacy-safe lifecycle pending/status/reconcile/retry CLI.
- [x] Exact-confirmation reset of failed lifecycle operations.
- [x] Configurable lifecycle outbox, cleanup journal, claim limit and lease duration.

### Tests/contracts committed

- generation-store lifecycle, sequence, corruption and identity tests;
- vector snapshot/restore and deterministic sparse-field tests;
- three-store replacement/deletion/compensation tests;
- authoritative public/raw deletion tests;
- document-service and batch restoration tests;
- lifecycle operation ID, phase and immutable-field tests;
- lease, renewal, release, retry and generic failure tests;
- exact replacement/deletion generation checks;
- registry failure replay without reindexing;
- cleanup intent ordering and already-absent cleanup replay;
- deletion operation-ID reuse and duplicate-delete refusal;
- database/parent identity and symlink/reparse refusal;
- startup reconciliation before RAG construction;
- root-scoped idempotent source removal;
- operator CLI privacy, bounds and confirmation tests;
- batch owner/byte identity and one-use context tests;
- idempotent batch registration short-circuit tests;
- import-hook behavior for modules that replace themselves in `sys.modules`.

### Focused verification evidence

- The partial local lifecycle core/runtime/boundary/import suite passed 22 tests.
- The batch-source bridge and lifecycle operator CLI passed 10 additional focused local tests.
- Python compilation passed for the locally exercised lifecycle modules and tests.
- Full exact-head, Windows and multi-process fault injection remains required.

### Still open in Wave 2B

- [ ] Retention/compaction policy for completed lifecycle operations.
- [ ] Privacy-safe operator audit export and job/lifecycle correlation reports.
- [ ] Periodic reconciliation or distributed leadership for multi-process deployment.
- [ ] Retained-source reindex for vector-only, sparse-only and manifest-only states.
- [ ] Explicitly reviewed adoption for verified aligned pre-manifest stores.
- [ ] Exact-head fault injection at every vector/sparse/manifest/registry/cleanup phase.

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

## Wave 3 — adaptive, corrective and route-experiment foundation

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
- [x] Privacy-safe SQLite adaptive trace store with owner/run isolation.
- [x] Optional runtime trace database configuration.
- [x] Query hashing and aggregate trace persistence without raw query/evidence storage.
- [x] Strict private-key filtering and provider-failure containment in public adaptive payloads.
- [x] Offline route adapters and benchmark harness for dense, corpus-sparse, corpus-hybrid, web and scholarly routes.
- [x] Strict local route fixtures with duplicate-key, nonstandard-number and symlink/reparse refusal.
- [x] Query-free/evidence-free JSON reports.
- [x] Router/oracle success, route accuracy, cost/latency utility, route aggregates and regret.
- [x] Reliability reports with Brier score, ECE and maximum calibration gap.
- [x] Dependency-free isotonic confidence calibration.
- [x] Risk-coverage curves and abstention-threshold selection.

### Still open in Wave 3

- [ ] Representative benchmark calibration of evidence coefficients and decision thresholds.
- [ ] Versioned runtime calibrator installation and corpus/profile selection.
- [ ] Validated domain classifier and domain-specialized policies.
- [ ] Connected production-scale uploaded/web/scholarly route experiments.
- [ ] Repeated-seed adaptive-policy ablations, confidence intervals and promotion gates.
- [ ] Trace retention, compaction, export and dashboards.

## Wave 4 — decomposition, heterogeneous multi-hop and benchmark foundation

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
- [x] Strict closed-schema model-assisted decomposition with deterministic fallback.
- [x] Provider-response digest without retaining model-authored evidence.
- [x] Structural plan-quality diagnostics.
- [x] Hard global estimated-cost ceiling across uploaded-document hops.
- [x] Minimum viable attempt reservation and impossible-budget refusal.
- [x] DAG-aware weighted allocation and unused-budget accounting.
- [x] Heterogeneous uploaded/web/scholarly multi-hop routing.
- [x] Production adapters for uploaded dense/sparse/hybrid, web and scholarly routes.
- [x] Global estimated-workload, latency and monetary planning budgets.
- [x] Privacy-safe multi-hop trace persistence.
- [x] One authoritative agent tool with `single`, `adaptive`, `multihop` and `heterogeneous` strategies.
- [x] Closed strategy schema, server owner injection and bounded argument validation.
- [x] Citation-only strategy publication through the existing evidence registry.
- [x] Adaptive and multi-hop abstention preventing weak citation publication.
- [x] Agent evidence deduplication/relabeling, API serialization and browser safe-DOM propagation.
- [x] Unicode answer normalization, answer exact match and token F1.
- [x] Document and support precision, recall and F1.
- [x] Page, section, field, source, sentence and paragraph support locators.
- [x] Complete support-path, hop-coverage and citation-lineage validity metrics.
- [x] Abstention-aware macro aggregation and explicitly heuristic answer-support score.
- [x] Strict HotpotQA, 2WikiMultiHopQA and MuSiQue local adapters.
- [x] Dataset byte fingerprints, bounded UTF-8 reads, duplicate-key/NaN refusal and path defenses.

### Focused verification evidence

- 35 focused decomposition, model-boundary, budget, executor, public-tool, evaluation and dataset-adapter tests passed locally.
- Strategy harnesses exercised all four retrieval strategies, import/reload behavior, abstention, lineage and live dispatcher validation.
- Python compilation passed for locally exercised strategy and multi-hop surfaces.

### Still open in Wave 4

- [ ] Learned plan ranking and benchmark-calibrated plan selection.
- [ ] Entity resolution and normalized temporal ranges.
- [ ] Measured rather than estimated cross-backend token/latency/monetary allocation.
- [ ] Custom scientific multi-document adapters and governed dataset cards.
- [ ] Semantic claim-support and entailment metrics per hop and final answer.
- [ ] Multi-hop ablation reports and historical regression thresholds.
- [ ] Exact-head end-to-end API/browser tests with real authoritative generations and connected providers.

## Verification status

The main-only exact-head workflow is configured for:

- Linux Python 3.10, 3.11 and 3.12 full suites;
- Windows classic-storage regressions on Python 3.10 and 3.12;
- Compose validation and container build;
- nine platform/Python release-lock jobs.

The repository currently has only `main`, and historical pull requests are closed. A complete green result is not observable for the latest exact `main` head through the available connector, and the constrained execution environment cannot clone GitHub because DNS resolution fails. Release readiness is therefore not claimed.

## Permanent non-claims

- Retrieval rank is not proof of factual correctness.
- Citation presence is not proof of entailment.
- Storage-generation alignment is provenance, not truth.
- A lifecycle outbox is not a distributed atomic transaction or consensus system.
- A decomposition graph or structural quality score is not proof of optimal decomposition.
- Cross-hop evidence grouping is not proof of shared claim support.
- The answer-support score is token/support-recall multiplication, not semantic entailment.
- Offline route fixtures prove harness behavior, not calibrated production routing.
- Estimated resource values are planning proxies, not measured token, latency or monetary cost.
- Learned rerankers may encode bias and require benchmark validation.
- Regex masking is not certified de-identification.
- Scientific conclusions require source inspection, expert review and replication.
