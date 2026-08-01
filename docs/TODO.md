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

## Wave 2B — authoritative durable generations

- [x] Public bounded vector generation snapshots and exact restore.
- [x] Deterministic sparse-field generation.
- [x] Compensating vector+sparse replacement coordinator.
- [x] Coordinated raw vector+sparse deletion.
- [x] Append-only durable generation history and current pointers.
- [x] One reentrant owner/document lock across all three stores.
- [x] Content hash, profile fingerprint, vector count and sparse generation manifests.
- [x] Privacy-finalized document-service integration.
- [x] API/durable-worker ingestion integration.
- [x] Public authoritative RAG deletion with raw internal compensation seam.
- [x] Batch ingestion authoritative snapshot/restore.
- [x] Drift scan and dry-run repair planning.
- [x] Bounded reconciliation CLI.
- [x] Exact-confirmation cleanup of deleted-generation residue.
- [ ] Add retained-document registry as a fourth transaction participant or durable outbox consumer.
- [ ] Run startup reconciliation before serving retrieval.
- [ ] Add resumable repair journal and operator audit records.
- [ ] Add retained-source reindex for vector-only, sparse-only and manifest-only states.
- [ ] Add explicitly reviewed adoption for verified aligned pre-manifest stores.

## Wave 2C — corpus-level hybrid retrieval

- [x] Generate dense and sparse candidates independently.
- [x] Fuse candidates at the document level.
- [x] Validate durable generations before publication.
- [x] Validate dense owner/content hash/profile metadata.
- [x] Validate sparse generation/profile metadata.
- [x] Materialize dense chunks and sparse fields with provenance.
- [x] Add explicit `corpus-sparse` and `corpus-hybrid` tool modes.
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
- [x] Refuse execution/cutover until shadow validation is available.
- [ ] Write shadow vector and sparse generations without replacing current state.
- [ ] Reparse retained sources through the privacy-finalized ingestion pipeline.
- [ ] Construct target-profile encoder adapters explicitly.
- [ ] Validate counts, hashes, provenance, quality and resource budgets.
- [ ] Persist shadow artifact identities and experiment metadata.
- [ ] Atomically cut over the durable current pointer.
- [ ] Keep rollback references and bounded shadow retention.
- [ ] Add active-worker pause/resume/cancel semantics.
- [ ] Test crash recovery at every migration phase.

## Wave 3 — adaptive and corrective RAG

- [x] Evidence-sufficiency and retrieval-quality signals.
- [x] Query intent, complexity and retrieval-mode routing policy.
- [x] Corrective retrieval plans with strict attempt and estimated-cost ceilings.
- [x] Bounded corrective-plan execution with accumulated-evidence limits.
- [x] Public adaptive uploaded-document retrieval tool and bounded trace payload.
- [x] Route traces and contained per-attempt failure diagnostics.
- [x] Privacy-safe durable adaptive trace store with owner/run isolation.
- [x] Optional runtime trace-store configuration and bounded trace persistence.
- [x] Private-key filtering and strict JSON-safe adaptive API payloads.
- [x] Offline dense/sparse/hybrid/web/scholarly route experiment harness.
- [x] Strict reproducible route fixtures and query/evidence-free benchmark reports.
- [x] Router/oracle success, route accuracy, cost/latency utility and regret metrics.
- [x] Brier score, reliability bins, ECE and maximum calibration-gap reports.
- [x] Dependency-free isotonic confidence calibration.
- [x] Risk-coverage curves and abstention-threshold selection.
- [x] Conservative abstention after insufficient terminal evidence.
- [ ] Calibrate evidence-sufficiency coefficients on representative datasets.
- [ ] Install a versioned runtime calibrator selected by benchmark and corpus profile.
- [ ] Add an explicit learned or validated domain classifier and domain-specific policies.
- [ ] Run representative connected dense/sparse/web/scholarly route experiments.
- [ ] Add adaptive-policy ablations, repeated seeds, confidence intervals and promotion thresholds.
- [ ] Add trace retention, compaction, export and operational dashboards.

## Wave 4 — decomposition and multi-hop retrieval

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
- [x] Deterministic token/entity/time/redundancy/parallelism/depth plan-quality diagnostics.
- [x] Hard global estimated-cost allocation across uploaded-document adaptive hops.
- [x] Minimum-attempt reservation and fail-fast impossible-budget checks.
- [x] Weighted DAG-aware remainder allocation, per-hop caps and unused-budget reporting.
- [x] Answer exact-match and Unicode token-F1 metrics.
- [x] Document and support precision/recall/F1 metrics.
- [x] Complete-support-path, hop-coverage and citation-lineage metrics.
- [x] Sentence, paragraph, page, section, field and source support locators.
- [x] Abstention-aware macro aggregation and explicitly heuristic answer-support score.
- [x] Strict local HotpotQA adapter with sentence-support preservation.
- [x] Strict local 2WikiMultiHopQA adapter with sentence-support preservation.
- [x] Strict local MuSiQue JSON/JSONL adapter with decomposition and paragraph-support preservation.
- [x] Dataset SHA-256 fingerprints, UTF-8/size limits, duplicate-key/NaN refusal and symlink/reparse protection.
- [ ] Add learned decomposition selection and plan-quality ranking.
- [ ] Add entity resolution and normalized temporal ranges.
- [ ] Add uploaded/web/scholarly heterogeneous multi-hop routing.
- [ ] Add global budget allocation across heterogeneous retrieval backends, latency and monetary cost.
- [ ] Add custom scientific multi-document benchmark adapters and dataset cards.
- [ ] Add semantic claim-support and entailment metrics per hop and final answer.
- [ ] Add multi-hop ablation reports and historical regression thresholds.
- [ ] Register the public tool with the full agent/API/browser surfaces after integration tests.

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

- [ ] Dense embedding adapter interface.
- [ ] Learned sparse retrieval adapter.
- [ ] Late-interaction/ColBERT-style adapter.
- [ ] Cross-encoder and listwise reranker interfaces.
- [ ] Multilingual and scientific-domain model profiles.
- [ ] Public, scientific, multilingual, multimodal and adversarial dataset cards.
- [ ] License, version, checksum and split governance.

## Wave 9 — experimentation and observability

- [ ] Repeated runs, seeds and confidence intervals.
- [ ] Statistical tests and practical-effect thresholds.
- [ ] Latency, throughput, memory and cost measurements.
- [ ] Retrieval, citation, entailment and abstention dashboards.
- [ ] Per-stage traces and bounded failure artifacts.
- [ ] Historical regression baselines and promotion gates.

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
- [ ] Clean-clone CLI/API ingestion, deletion, reconciliation and retrieval smoke tests.
- [ ] Concurrency and fault injection across vector, sparse, manifest and registry boundaries.
- [ ] Adaptive and multi-hop integration tests against authoritative corpus generations.
- [ ] Final line-by-line regression audit of one unchanged exact `main` SHA.
