# Exhaustive implementation TODO

This backlog is ordered by dependency and risk. Checked items have committed implementation and focused contracts; release verification remains separate.

## Wave 1 — retrieval and evaluation foundation

- [x] Hybrid candidate ranking and component traces.
- [x] Candidate-pool BM25.
- [x] Reciprocal-rank and weighted fusion.
- [x] MMR and source caps.
- [x] Heuristic and optional cross-encoder rerankers.
- [x] Dense compatibility defaults.
- [x] BEIR loader and retrieval/citation metrics.
- [x] Deterministic manifests and resumable experiment store.
- [x] Offline baseline CLI.

## Wave 2A — model registry and persistent sparse index

- [x] Canonical embedding profiles, aliases and fingerprints.
- [x] MiniLM, E5, BGE, GTE, Instructor, SPECTER2 and BGE-M3 profiles.
- [x] Strict operator profile JSON.
- [x] Owner/document/generation-isolated fielded sparse schema.
- [x] Page, section, field, frequency and position provenance.
- [x] Transactional replace, snapshot, restore and delete.
- [x] Symlink/reparse/database identity defenses.
- [x] Field-weighted BM25 and exact document filtering.

## Wave 2B — authoritative four-store lifecycle

- [x] Public bounded vector generation snapshots and exact restore.
- [x] Deterministic sparse-field generation.
- [x] Compensating vector+sparse replacement coordinator.
- [x] Coordinated raw vector+sparse deletion.
- [x] Append-only durable generation history and current pointers.
- [x] One reentrant owner/document lock across vector, sparse and generation stores.
- [x] Content hash, profile fingerprint, vector count and sparse generation manifests.
- [x] Privacy-finalized document-service integration.
- [x] API/durable-worker ingestion integration.
- [x] Public authoritative RAG deletion with raw internal compensation seam.
- [x] Batch ingestion authoritative snapshot/restore.
- [x] Drift scan and dry-run repair planning.
- [x] Bounded reconciliation CLI.
- [x] Exact-confirmation cleanup of deleted-generation residue.
- [x] Durable retained-document registry replacement/deletion outbox.
- [x] Planned/index-committed/registry-committed/completed/failed state machine.
- [x] Deterministic owner/document replacement and deletion operation IDs.
- [x] Worker leases, renewal, release, retry ceilings and generic errors.
- [x] Exact generation/content-hash reconciliation.
- [x] Registry replay after index commit without duplicate indexing.
- [x] Pending deletion identity reuse after generation sequence changes.
- [x] Private cleanup-intent journal persisted before registry mutation.
- [x] Idempotent cleanup replay after process crashes or already-absent files.
- [x] Startup reconciliation before first retrieval-layer construction.
- [x] API/durable-job retained-source coordination.
- [x] Batch retained-source copy bridge with owner and byte-identity validation.
- [x] One-use context-local batch source intent and temporary source binding.
- [x] Idempotent redundant batch registry short-circuit.
- [x] Privacy-safe pending/status/reconcile/retry lifecycle CLI.
- [x] Exact-confirmation retry of failed lifecycle operations.
- [x] Document lifecycle configuration and runbook.
- [ ] Add completed-operation retention and compaction policy.
- [ ] Add privacy-safe operator audit export and job/lifecycle correlation.
- [ ] Add periodic reconciliation or distributed leadership for multi-process deployment.
- [ ] Add retained-source reindex for vector-only, sparse-only and manifest-only states.
- [ ] Add explicitly reviewed adoption for verified aligned pre-manifest stores.
- [ ] Run exact-head fault injection at every four-store and cleanup phase.

## Wave 2C — corpus-level hybrid retrieval

- [x] Generate dense and sparse candidates independently.
- [x] Fuse candidates at the document level.
- [x] Validate durable generations before publication.
- [x] Validate dense owner/content hash/profile metadata.
- [x] Validate sparse generation/profile metadata.
- [x] Materialize dense chunks and sparse fields with provenance.
- [x] Add explicit `corpus-sparse` and `corpus-hybrid` modes.
- [x] Return dense, sparse, fused, generation and profile traces.
- [x] Contain partial multi-query failures.
- [x] Protect ranking/generation citation metadata from evidence overrides.
- [ ] Benchmark and calibrate dense/sparse fusion weights.
- [ ] Add independent-corpus reciprocal-rank fusion.
- [ ] Add explicit per-document and source caps.
- [ ] Add MIME, date, field, page, section and provenance filters.
- [ ] Add reranker cascades with latency, memory and cost budgets.
- [ ] Add dense/sparse/hybrid ablation reports and regression thresholds.

## Wave 2D — profile migration, promotion and cutover preparation

- [x] Inventory current manifests by target profile and durable source sequence.
- [x] Classify retained-source eligibility without exposing source paths.
- [x] Generate immutable deterministic migration task IDs.
- [x] Persist idempotent resumable migration tasks.
- [x] Add worker leases, renewal, retry ceilings and generic failure types.
- [x] Require validation digests before committed state.
- [x] Reclaim expired running and validated tasks.
- [x] Add inventory, seed, status and owner-verified cancel commands.
- [x] Refuse live cutover until isolated validation, promotion and rollback prerequisites exist.
- [x] Write task-isolated manifest-last vector and sparse shadow artifacts without replacing current state.
- [x] Reparse retained sources through the current privacy-finalized ingestion pipeline.
- [x] Revalidate owner/source-byte document identity before shadow construction.
- [x] Construct explicit target-profile encoder adapters.
- [x] Refuse adapter-required profiles until a named adapter factory is registered.
- [x] Build one-to-one vector and sparse rows with field/page/section provenance.
- [x] Validate finite vectors, dimensions, counts, content hashes and target profile fingerprints.
- [x] Recheck the source generation before and after shadow construction.
- [x] Persist immutable shadow artifact identities, digests, sizes and validation manifests.
- [x] Add atomic build-only claiming that excludes already-validated tasks.
- [x] Add no-cutover shadow build/validate/remove CLI commands.
- [x] Add conservative aggregate quality and resource promotion policy.
- [x] Require exact task/manifest/journal/evidence/source-generation alignment for promotion reports.
- [x] Evaluate recall, nDCG, MRR, support recall, citation precision and abstention accuracy.
- [x] Evaluate p95 latency, peak memory, storage and estimated-cost ratios.
- [x] Persist append-only promotion reports with immutable history and an atomic current pointer.
- [x] Add strict evidence/policy JSON, deterministic reason codes and no-cutover promotion CLI.
- [x] Add a repository-owned paired query-ID-only benchmark producer.
- [x] Enforce the same ordered benchmark contract across repeated runs.
- [x] Compute contract-only benchmark fingerprints independent of model outputs/resources.
- [x] Compute signed paired 95% confidence intervals across repeated runs.
- [x] Add paired lower-bound non-inferiority gates for all six quality metrics.
- [x] Add optional lower-bound practical-gain thresholds.
- [x] Bind statistical assessment/policy digests into the final promotion report without changing its schema.
- [x] Add direct in-process `evaluate-fixture` benchmark plus aggregate/statistical promotion flow.
- [x] Add hashes/counts-only cutover preflight identities.
- [x] Require an eligible paired-statistical report before preflight.
- [x] Re-capture and hash complete current vector/sparse rollback snapshots during preflight.
- [x] Require exact current generation sequence/profile/content and snapshot count/generation alignment.
- [x] Persist append-only preflight history and an atomic current pointer.
- [x] Add plan/status/history and failed/cancelled cleanup CLI with explicit `mutation_performed: false`.
- [x] Add AES-256-GCM encrypted rollback payloads containing complete privacy-finalized vector, sparse and generation snapshots.
- [x] Require explicit operator key ID and canonical base64 32-byte key material with no plaintext/default fallback.
- [x] Add manifest-last encrypted artifact publication, authenticated metadata and tamper/wrong-key refusal.
- [x] Add status/verify/capture and double-confirmed failed/cancelled cleanup commands without restore.
- [x] Reconstruct encrypted payloads into public immutable vector, sparse and generation snapshot types in memory.
- [x] Verify a write/read/re-snapshot cycle in a bounded non-authoritative staging store.
- [x] Add a deterministic preparation operation ID and leased ready-only preparation journal.
- [x] Bind validated task, eligible report, preflight, rollback artifact, staging verification and unchanged source generation before `ready`.
- [x] Add an adapter-only hidden-write/validate/visibility/validate compensation saga with fault-injection contracts.
- [ ] Execute governed benchmark fixtures against the actual current and shadow retrieval stacks.
- [ ] Measure wall-clock latency, process/device memory, storage and provider billing rather than accepting supplied resource observations.
- [ ] Add reviewed bootstrap/permutation tests and multiple-comparison controls where appropriate.
- [ ] Add governed production adapters for Instructor, SPECTER2, BGE-M3 and future adapter-required profiles.
- [ ] Integrate KMS/HSM or a production secret manager and governed key rotation/re-encryption.
- [ ] Define encrypted rollback retention, legal hold and secure-deletion policy.
- [ ] Implement a production vector+sparse+generation cutover adapter.
- [ ] Connect ready preparation operations to a durable executing/committed/rolled-back saga journal.
- [ ] Atomically or compensatingly publish vector, sparse and durable current-generation state without exposing an unvalidated mixed generation.
- [ ] Validate the new authoritative generation before marking the migration committed.
- [ ] Add automatic rollback and exact rollback-identity verification after every failed publication phase.
- [ ] Keep old state until rollback and new-generation verification pass.
- [ ] Add bounded shadow/report/preflight/rollback retention and compaction.
- [ ] Add active-worker pause/resume/cancel semantics.
- [ ] Test crash recovery at every build, report, preflight, encrypted artifact, cutover and rollback phase.

## Wave 3 — adaptive and corrective RAG

- [x] Evidence-sufficiency and retrieval-quality signals.
- [x] Query intent, complexity and retrieval-mode routing policy.
- [x] Corrective retrieval plans with strict attempt and estimated-cost ceilings.
- [x] Bounded corrective-plan execution with accumulated-evidence limits.
- [x] Public adaptive uploaded-document retrieval tool and bounded trace payload.
- [x] Route traces and contained per-attempt failure diagnostics.
- [x] Privacy-safe durable adaptive trace store with owner/run isolation.
- [x] Optional runtime trace-store configuration and bounded trace persistence.
- [x] Private-key filtering and strict JSON-safe adaptive payloads.
- [x] Offline dense/sparse/hybrid/web/scholarly route experiment harness.
- [x] Strict reproducible route fixtures and query/evidence-free reports.
- [x] Router/oracle success, route accuracy, cost/latency utility and regret metrics.
- [x] Brier score, reliability bins, ECE and maximum calibration-gap reports.
- [x] Dependency-free isotonic confidence calibration.
- [x] Risk-coverage curves and abstention-threshold selection.
- [x] Conservative abstention after insufficient terminal evidence.
- [ ] Calibrate evidence-sufficiency coefficients on representative datasets.
- [ ] Install a versioned runtime calibrator selected by benchmark and corpus profile.
- [ ] Add a validated domain classifier and domain-specific policies.
- [ ] Run representative connected dense/sparse/web/scholarly route experiments.
- [ ] Add repeated-seed ablations, confidence intervals and promotion thresholds.
- [ ] Add trace retention, compaction, export and operational dashboards.

## Wave 4 — decomposition and heterogeneous multi-hop retrieval

- [x] Bounded deterministic query decomposition.
- [x] Validated acyclic dependency graph for subquestions.
- [x] Stable plan fingerprints, topological batches and terminal-node detection.
- [x] Parallel independent hops and serial dependent batches.
- [x] Bounded worker, timeout, per-hop, dependency and total-evidence budgets.
- [x] Entity and temporal constraint extraction and propagation.
- [x] Bounded lexical hint propagation from prerequisite evidence.
- [x] Evidence joining without citation laundering or source-identity collapse.
- [x] Per-hop source/document/page lineage and contained failures.
- [x] Public adaptive uploaded-document multi-hop tool and lineage payload.
- [x] Abstention when terminal hops provide no evidence.
- [x] Strict-schema model-assisted decomposition with deterministic fallback.
- [x] Provider-response digesting without retaining model-authored evidence.
- [x] Structural token/entity/time/redundancy/parallelism/depth diagnostics.
- [x] Hard global uploaded-document estimated-cost allocation.
- [x] Minimum-attempt reservation and fail-fast impossible-budget checks.
- [x] DAG-aware weighted allocation, per-hop caps and unused-budget reporting.
- [x] Heterogeneous uploaded/web/scholarly multi-hop routing.
- [x] Production uploaded dense/sparse/hybrid, web and scholarly adapters.
- [x] Global workload, latency and monetary planning budgets.
- [x] Privacy-safe multi-hop trace persistence.
- [x] One authoritative agent tool with single/adaptive/multihop/heterogeneous strategies.
- [x] Closed strategy schema and bounded dispatcher argument validation.
- [x] Server-owned Citation-only publication for every strategy.
- [x] Adaptive/multi-hop abstention before evidence registry publication.
- [x] Agent deduplication/relabeling, API serialization and safe browser propagation.
- [x] Answer exact match and Unicode token-F1 metrics.
- [x] Document and support precision/recall/F1 metrics.
- [x] Complete support-path, hop-coverage and citation-lineage metrics.
- [x] Sentence, paragraph, page, section, field and source support locators.
- [x] Abstention-aware macro aggregation and heuristic answer-support score.
- [x] Strict local HotpotQA and 2WikiMultiHopQA adapters.
- [x] Strict local MuSiQue JSON/JSONL adapter.
- [x] Dataset SHA-256, UTF-8/size limits, duplicate-key/NaN refusal and path defenses.
- [ ] Add learned decomposition selection and plan-quality ranking.
- [ ] Add entity resolution and normalized temporal ranges.
- [ ] Replace planning proxies with measured cross-backend resource models.
- [ ] Add custom scientific multi-document adapters and governed dataset cards.
- [ ] Add semantic claim-support and entailment metrics per hop and final answer.
- [ ] Add multi-hop ablation reports and historical regression thresholds.
- [ ] Run exact-head end-to-end API/browser/provider integration tests.

## Wave 5 — provenance evidence graph

- [x] Typed document, section, claim, entity, method, dataset and citation nodes.
- [x] Deterministic provenance-preserving graph edges and generation-scoped graph digests.
- [x] Explicit-only graph construction from finalized documents and reviewed annotations/relations.
- [x] Transactional immutable graph generations and optimistic current pointers.
- [x] Deterministic lexical node retrieval and type filters.
- [x] Directed cycle-safe path explanations with edge/node filters.
- [x] Explicit-edge-only support and contradiction clustering.
- [x] Read-only privacy-conscious graph status/history/search/path/analysis CLI.
- [x] Durable exact-authoritative-generation derived graph job identities and SQLite journal.
- [x] Exclusive leases, expiry/reclaim, attempt ceilings, generic failures, retry and cancellation.
- [x] Structural document/section graph rebuild from authoritative sparse snapshots.
- [x] Deleted-generation tombstone graphs.
- [x] Exact generation/content/profile/sparse identity checks before and after graph publication.
- [x] Idempotent exact graph publication and backward-pointer refusal.
- [x] Fail-closed current graph reader requiring exact authoritative sequence/content/profile identity.
- [x] Deleted current graph tombstone validation.
- [x] Historical graph inspection explicitly marked non-current.
- [ ] Add startup/periodic job scheduling after multi-process leadership is implemented.
- [ ] Add distributed or database-backed leadership instead of process-local striped locks.
- [ ] Add privacy-safe job audit export, dead-letter reporting, retention and compaction.
- [ ] Add crash/disk/concurrency injection around claim, graph insert, pointer publication and job completion.
- [ ] Add cross-document graph-set types without collapsing owner/document/generation provenance.
- [ ] Add reviewed cross-document citation and entity-resolution workflows.
- [ ] Add bounded GraphRAG retrieval and summaries with source lineage.
- [ ] Add path-aware evidence selection and server-owned citation conversion.
- [ ] Add graph retrieval/path-completeness benchmarks and historical regression thresholds.
- [ ] Add reviewed closed-schema scientific extraction adapters and human correction lineage.
- [ ] Add graph database backup/restore and disaster-recovery policy.

## Wave 6 — multimodal scientific ingestion

- [ ] Reading-order and layout models.
- [ ] Table structure extraction and cell provenance.
- [ ] Formula detection, OCR and normalization.
- [ ] Figure, caption and panel association.
- [ ] OCR/layout/table/formula quality gates.
- [ ] Multimodal retrieval and evidence citations.

## Wave 7 — scientific evidence intelligence

- [ ] Structured research-question and PICO/PECO schemas.
- [ ] Methods, population, intervention, outcome and result extraction.
- [ ] Effect-size and uncertainty normalization.
- [ ] Risk-of-bias and evidence-quality fields.
- [ ] Claim/evidence/limitation/conflict linking.
- [ ] Human-review queues and correction lineage.

## Wave 8 — models and datasets

- [x] Explicit dense embedding adapter interface and process-local adapter registry.
- [ ] Governed production adapters for Instructor, SPECTER2, BGE-M3 and future adapter-required profiles.
- [ ] Learned sparse retrieval adapter.
- [ ] Late-interaction/ColBERT-style adapter.
- [ ] Cross-encoder and listwise reranker interfaces.
- [ ] Additional multilingual and scientific-domain model profiles.
- [ ] Public, scientific, multilingual, multimodal and adversarial dataset cards.
- [ ] License, version, checksum and split governance.

## Wave 9 — experimentation and observability

- [x] Query-contract paired fixture aggregation across repeated runs and seeds.
- [x] Signed paired 95% confidence-interval foundation.
- [x] Versioned non-inferiority and optional practical-gain policies.
- [x] Versioned conservative aggregate promotion policy and append-only report foundation.
- [ ] Repository-owned orchestration that executes repeated current/shadow runs rather than consuming collected outputs.
- [ ] Bootstrap/permutation tests, multiple-comparison controls and reviewed practical-effect governance.
- [ ] Measured latency, throughput, memory, storage and monetary cost.
- [ ] Retrieval, citation, entailment and abstention dashboards.
- [ ] Per-stage traces and bounded failure artifacts.
- [ ] Historical regression baselines and automatic promotion gates from real-stack benchmark output.

## Wave 10 — distributed production architecture

- [ ] Durable distributed queue and worker leases.
- [ ] SQL registry and outbox/saga coordination.
- [ ] Object storage for retained files and artifacts.
- [ ] Distributed rate limiting and admission control.
- [ ] Idempotency keys and exactly-once-effect design.
- [ ] TLS, egress policy, secret manager, malware scanning and parser sandbox.

## Required verification before release claims

- [ ] Dependency installation and `pip check`.
- [ ] Whitespace and generated-artifact checks.
- [ ] Python compilation.
- [ ] Fatal Ruff checks and configured lint policy.
- [ ] Full pytest and measured branch coverage on Python 3.10, 3.11 and 3.12.
- [ ] Windows compatibility tests.
- [ ] Docker Compose validation.
- [ ] Docker image build and readiness smoke tests.
- [ ] Clean-clone CLI/API/batch ingestion, deletion, reconciliation and retrieval smoke tests.
- [ ] Concurrency and fault injection across vector, sparse, manifest, registry and cleanup journals.
- [ ] Adaptive and heterogeneous multi-hop integration tests against authoritative generations.
- [ ] Migration shadow, promotion, statistical and preflight tests against real authoritative current/shadow retrieval runs.
- [ ] Protected rollback-artifact, cutover and rollback fault injection.
- [ ] Evidence-graph reconciliation, stale-read refusal, corruption and multi-process concurrency tests.
- [ ] Final line-by-line regression audit of one unchanged exact `main` SHA.
