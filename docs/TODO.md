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

## Wave 2D — profile migration and reindex

- [x] Inventory current manifests by target profile and durable source sequence.
- [x] Classify retained-source eligibility without exposing source paths.
- [x] Generate immutable deterministic migration task IDs.
- [x] Persist idempotent resumable migration tasks.
- [x] Add worker leases, renewal, retry ceilings and generic failure types.
- [x] Require validation digests before committed state.
- [x] Reclaim expired running and validated tasks.
- [x] Add inventory, seed, status and owner-verified cancel commands.
- [x] Refuse live cutover until isolated validation and promotion gates exist.
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
- [ ] Add a repository-owned benchmark runner that produces promotion evidence from governed query fixtures.
- [ ] Add confidence intervals, statistical tests and practical-effect thresholds produced by that runner.
- [ ] Replace estimated resource values with measured latency, memory, storage and monetary accounting where available.
- [ ] Atomically cut over vector, sparse and durable current-generation state.
- [ ] Keep exact rollback references and verify rollback before releasing old state.
- [ ] Add bounded shadow and promotion-report retention/compaction.
- [ ] Add active-worker pause/resume/cancel semantics.
- [ ] Test crash recovery at every build, report, cutover and rollback phase.

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

## Wave 5 — evidence graph

- [ ] Typed document, section, claim, entity, method, dataset and citation nodes.
- [ ] Provenance-preserving graph edges.
- [ ] Graph construction from authoritative generations.
- [ ] Graph retrieval and path explanations.
- [ ] Contradiction/support clustering.
- [ ] Bounded GraphRAG summaries with source lineage.

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

- [ ] Repository-owned repeated runs, seeds and confidence intervals.
- [ ] Statistical tests and practical-effect thresholds.
- [ ] Measured latency, throughput, memory, storage and monetary cost.
- [ ] Retrieval, citation, entailment and abstention dashboards.
- [ ] Per-stage traces and bounded failure artifacts.
- [x] Versioned conservative promotion-policy and append-only report foundation.
- [ ] Historical regression baselines and automatic promotion gates from repository-owned benchmark output.

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
- [ ] Migration shadow, promotion-report, cutover and rollback fault injection against authoritative generations.
- [ ] Final line-by-line regression audit of one unchanged exact `main` SHA.
