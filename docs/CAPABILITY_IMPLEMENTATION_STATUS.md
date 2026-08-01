# Capability implementation status

Last updated: 2026-08-02

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

## Wave 2D — isolated migration, paired promotion and cutover preflight

### Planning and journal implemented

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
- [x] Planned/failed cancellation only and owner verification before cancellation.
- [x] Inventory, seed, status and cancel CLI commands.

### Shadow construction implemented

- [x] Task-isolated manifest-last vector and sparse shadow artifacts.
- [x] Retained-source reparse through the current privacy-finalized ingestion pipeline.
- [x] Owner/source-byte identity verification before shadow construction.
- [x] Explicit embedding encoder interface and process-local adapter registry.
- [x] Default bounded SentenceTransformer-compatible adapter.
- [x] Fail-closed behavior for adapter-required profiles without a registered adapter.
- [x] Deterministic authoritative sparse fields used for one-to-one shadow vector rows.
- [x] Field, page, section, source-sequence, content-hash and profile provenance.
- [x] Independent row-count, vector-dimension, boolean, NaN and infinity refusal.
- [x] Source-generation sequence/profile/content checks before and after building.
- [x] Immutable shadow vector/sparse digests, sizes, counts and parser fingerprint.
- [x] Idempotent shadow reuse across retry timestamps and changed-artifact refusal.
- [x] Tamper, symlink/reparse, root-identity and strict JSON defenses.
- [x] Atomic shadow-build claimant that excludes already-validated tasks.
- [x] Build, validate and failed/cancelled cleanup CLI with no cutover command.

### Repository-owned paired benchmark implemented

- [x] Strict query-ID-only paired current/shadow fixture schema.
- [x] Exact ordered benchmark contract across repeated runs.
- [x] Contract-only benchmark fingerprint independent of ranked outputs/resources.
- [x] Recall@k, nDCG@k, MRR, support recall, citation precision and abstention accuracy.
- [x] Conservative p95/max resource aggregation and mean estimated cost.
- [x] Repeated-run and distinct-seed accounting.
- [x] Signed paired 95% confidence intervals.
- [x] Strict duplicate-key, nonstandard-number, unknown-field, symlink and file-identity refusal.
- [x] Atomic aggregate evidence and optional detailed interval-report output.
- [x] Contract inspection and benchmark script entrypoint.

### Aggregate promotion implemented

- [x] Versioned conservative aggregate promotion policy.
- [x] Exact task/journal/manifest/evidence/source-generation alignment gates.
- [x] Quality floors and maximum point-estimate regressions for six quality metrics.
- [x] Resource ceilings for p95 latency, peak memory, storage and estimated cost.
- [x] Minimum query-count, repeated-run, seed-count and confidence-level gates.
- [x] Deterministic evidence, policy and report digests with bounded reason codes.
- [x] Append-only promotion-report history and atomic per-task current pointer.
- [x] Strict aggregate evidence/policy JSON and privacy-safe evaluate/status/history/remove CLI.

### Paired statistical promotion implemented

- [x] Versioned `paired-noninferiority-v1` policy.
- [x] Minimum paired-run, seed-count and confidence-level requirements.
- [x] Lower-confidence-bound non-inferiority for recall, nDCG, MRR, support recall, citation precision and abstention accuracy.
- [x] Optional lower-confidence-bound practical-gain thresholds.
- [x] Deterministic per-metric assessments and statistical assessment digest.
- [x] Composite evidence and policy digests attached without changing the stored promotion-report schema.
- [x] Direct in-process `evaluate-fixture` benchmark plus aggregate/statistical promotion flow.
- [x] Final blocking whenever either aggregate or statistical gates fail.

### Non-mutating cutover preflight implemented

- [x] Require migration task state `validated`.
- [x] Require current promotion report `eligible` under `paired-promotion-v1`.
- [x] Bind exact task/shadow/report owner, document, source sequence, source profile, target profile and validation identities.
- [x] Re-capture the complete current authoritative vector/sparse/generation snapshot.
- [x] Require the source generation to remain active/restored with the exact source sequence, profile and content hash.
- [x] Require the shadow content hash to equal the current authoritative content hash.
- [x] Hash complete current vector rows and metadata under owner/document scope.
- [x] Require vector rollback row count equal to the generation record.
- [x] Hash complete current sparse fields and metadata.
- [x] Require sparse rollback generation/profile equal to the generation record/task.
- [x] Derive rollback and target artifact identity digests.
- [x] Persist only hashes, counts, sequences and fingerprints; do not persist rollback text.
- [x] Append-only preflight history and atomic per-task current pointer.
- [x] Plan/status/history and failed/cancelled cleanup CLI.
- [x] Explicit `mutation_performed: false` in successful operator payloads.
- [x] No approval, execute, pointer-swap, task-commit or rollback command.

### Focused verification evidence

- Isolated shadow store/executor contracts cover manifest-last writes, idempotent replay, tamper detection, source-generation races and generic failures.
- Encoder and retained-source builder contracts cover passage prefixes, adapter-required refusal/registration, exact owner/document identity, one-to-one vector/sparse provenance and malformed custom encoders.
- Shadow runtime/CLI contracts cover build-only claiming, validated-task exclusion, attempt ceilings, privacy-safe output and no-cutover behavior.
- The promotion/benchmark/statistical constrained local harness passed **42 tests**.
- These contracts cover aggregate decisions, paired metric computation, contract fingerprints, signed intervals, non-inferiority, practical gains, direct fixture evaluation, composite digests, append-only history, strict JSON, path defenses and exact-confirmation cleanup.
- The cutover-preflight constrained local harness passed **15 tests**.
- Preflight contracts cover source/shadow/report identity binding, eligible-paired-report requirement, complete vector/sparse snapshot hashing, scope/count/generation checks, timestamp-stable digests, append-only history, tamper/path defenses and explicit non-mutation output.
- Full exact-head repository execution, real model loading, real current/shadow benchmark execution, Windows and cutover/rollback fault injection remain required.

### Still open in Wave 2D

- [ ] Execute governed fixtures against the actual current and shadow retrieval stacks rather than consuming collected ranked identifiers.
- [ ] Measure wall-clock latency, process/device memory, artifact storage and provider billing.
- [ ] Add reviewed bootstrap/permutation procedures and multiple-comparison controls where scientifically appropriate.
- [ ] Governed production adapters for Instructor, SPECTER2, BGE-M3 and future adapter-required profiles.
- [ ] Protected durable rollback-artifact store containing complete privacy-finalized vector and sparse snapshots.
- [ ] Explicit rollback encryption/key-management, retention and secure-deletion policy.
- [ ] Durable cutover journal with exclusive leases and idempotency keys.
- [ ] Atomic or compensating vector+sparse+generation publication with no exposed unvalidated mixed state.
- [ ] Validate the new authoritative generation before marking the migration committed.
- [ ] Automatic rollback after every failed publication/validation phase.
- [ ] Exact rollback-identity verification and old-state retention until verification passes.
- [ ] Bounded shadow/report/preflight/rollback retention and compaction.
- [ ] Active-worker pause/resume/cancel semantics.
- [ ] Crash/fault injection at every build, report, preflight, publication and rollback phase.

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
- A migration shadow is not live state.
- An `eligible` promotion report does not authorize or perform cutover.
- Paired non-inferiority under one governed fixture does not prove universal non-inferiority.
- A cutover preflight is not approval or reservation.
- A rollback identity digest is not a restorable rollback artifact.
- Collected paired ranked identifiers are not the same as repository-orchestrated real-stack benchmark execution.
- Estimated or supplied resource values are not measured deployment accounting.
- A decomposition graph or structural quality score is not proof of optimal decomposition.
- Cross-hop evidence grouping is not proof of shared claim support.
- The answer-support score is token/support-recall multiplication, not semantic entailment.
- Offline route fixtures prove harness behavior, not calibrated production routing.
- Learned rerankers or embedding adapters may encode bias and require benchmark validation.
- Regex masking is not certified de-identification.
- Scientific conclusions require source inspection, expert review and replication.
