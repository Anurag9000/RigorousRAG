# RigorousRAG exhaustive mission, implementation audit, gap analysis, and forward architecture

Audit date: 2026-08-06  
Audited branch: `main`  
Audited implementation baseline: `ffbb3b14ec29ed3822f0f9f5c7fc5a7eefad1335`  
Repository policy: direct commits to `main`; no new feature branches or pull requests; preserve historical merged-PR provenance.

## 0. Authoritative audit artifacts

This document is the human-readable architecture and roadmap. The machine-readable source of truth is:

- `docs/rigorousrag_capability_ledger.json`
- validator: `scripts/validate_capability_ledger.py`
- regression tests: `tests/unit/test_capability_ledger.py`

The ledger deliberately separates three dimensions that old checklists conflated:

1. **Implementation** — `not_started`, `partial`, or `implemented`.
2. **Validation** — `not_validated`, `unit_validated`, `integration_validated`, or `experimentally_validated`.
3. **Release** — `not_verified`, `release_verified`, or `blocked`.

Therefore:

- code existence does not imply production completeness;
- unit tests do not imply experimental superiority;
- integration validation does not imply release certification;
- no capability is release-certified until the complete exact-head workflow passes on one unchanged `main` SHA.

The ledger currently aggregates **32 architecture-level capabilities**:

- 14 implemented foundations;
- 12 partial capabilities;
- 6 not-started capability families;
- 0 release-verified capabilities;
- 1 explicit release blocker: exact-head Linux full-suite failure.

## 1. Executive verdict

RigorousRAG is no longer the small prototype it began as. It now contains a broad, security-conscious foundation spanning deterministic ranking, tenant-safe storage, bounded ingestion, authoritative generation management, dense and sparse retrieval, hybrid fusion, adaptive routing, multi-hop decomposition, server-owned citations, evidence graphs, scientific claim governance, durable graph operations, experiment primitives, and cross-platform release infrastructure.

However, the project is **not yet complete or release-certified**. Its strongest area is correctness and governance infrastructure. Its largest remaining gaps are:

1. exact-head release proof;
2. full lifecycle retention, rollback custody, and crash-phase testing;
3. governed migration execution rather than preflight alone;
4. benchmark-calibrated retrieval and adaptive policies;
5. complete verification-aware generation and evidence packing;
6. deeper scientific schemas and temporal evidence reasoning;
7. licensed dataset adapters and rigorous statistical experiment programs;
8. multimodal document structure;
9. distributed production, observability, key management, backup, and disaster recovery;
10. domain-specialized LegalRAG, biomedical, and financial workflows.

The correct next strategy is not indiscriminate feature accumulation. It is to finish the dependency chain in order: release integrity, lifecycle, migration, retrieval quality, adaptive verification, scientific depth, multimodality, experiments, then distributed production.

## 2. What we originally wanted to build

### 2.1 Starting point

The repository began as three loosely connected ideas:

- a randomized Ritter-style bounding-sphere algorithm in `RRAlgorithm.py` history;
- PageRank plus TF-IDF crawling and search in the classic modules;
- a FastAPI, vector-database, sentence-transformer, and provider-backed RAG application.

The audit identified a recurring problem: useful ideas existed, but correctness, provenance, tenancy, persistence, evaluation, and operational boundaries were not unified.

### 2.2 Defects that motivated the rebuild

The initial program of work included:

- correct PageRank dangling-mass handling and convergence;
- replace simplistic score mixing with traceable fusion;
- repair imports, stale tests, API wiring, and incompatible legacy paths;
- prevent model-authored or unbound citations;
- consolidate or strictly coordinate retained source, vector, sparse, and generation state;
- bound parsing, uploads, crawling, search, provider output, and generated output;
- make optional dependencies fail locally and explicitly;
- eliminate unsafe path, symlink, reparse-point, archive, and database identity behavior;
- add deterministic identifiers, canonical JSON, finite numeric contracts, and bounded collections;
- create reproducible dependency locks, containers, clean installs, and exact-head CI;
- replace informal evaluation with immutable datasets, manifests, metrics, ablations, and statistical gates.

### 2.3 Target end state

The requested end state is a **reproducible, tenant-safe, provenance-complete, scientifically defensible RAG research and production platform** with the following properties.

#### Correctness and reproducibility

- deterministic algorithms and stable tie-breaking;
- typed schemas and explicit versioning;
- canonical identifiers and serialized records;
- clean dependency installation and lock authority;
- adversarial, corruption, retry, concurrency, and cross-platform tests;
- exact-head release proof on one unchanged commit.

#### Security, privacy, and tenancy

- tenant identity derived only from trusted server authentication;
- every source, chunk, posting, vector, graph, job, trace, and operator action owner-scoped;
- fail-closed filesystem and database identity checks;
- SSRF, redirect, peer, proxy, MIME, response-size, timeout, and egress controls;
- privacy masking before persistence or display;
- operator views based on hashes, counts, states, and reason codes rather than unnecessary raw evidence;
- clear separation between provenance, semantic support, scientific truth, legal interpretation, and medical advice.

#### Authoritative ingestion and lifecycle

- one bounded privacy-finalized parsing boundary;
- stable owner, document, content, generation, and source-region identities;
- coordinated retained-source, vector, sparse, and generation stores;
- durable journals, leases, retries, compensation, and startup reconciliation;
- replacement, deletion, restoration, drift repair, retention, compaction, and rollback;
- isolated migration shadows and exact-confirmed cutovers;
- no mixed or unvalidated authoritative generations.

#### Retrieval and ranking

- TF-IDF and fielded BM25 baselines;
- governed dense embeddings;
- weighted fusion, reciprocal-rank fusion, MMR, source caps, and filters;
- cross-encoder, listwise, neural sparse, and late-interaction reranking;
- HyDE, step-back, multi-query, decomposition, parent-child, sentence-window, and compression strategies;
- adaptive cascades governed by quality, latency, memory, and cost;
- multilingual, Indic, scientific, biomedical, legal, and financial profiles;
- explicit abstention and evidence-gap reporting.

#### Adaptive, corrective, and agentic RAG

- deterministic query analysis and route selection;
- evidence sufficiency and calibrated confidence;
- bounded corrective retrieval;
- loop-safe multi-hop subquestion DAGs;
- uploaded, web, and scholarly routes with common provenance;
- critique and revision without hidden evidence mutation;
- versioned policies, calibration, promotion, and rollback;
- privacy-safe traces and diagnostics.

#### Answer verification and citations

- server-owned citation identifiers;
- exact claim-to-evidence binding;
- support, contradiction, uncertainty, and reviewer correction records;
- evidence-aware abstention;
- counterfactual citation tests;
- constrained evidence packing and claim-conditioned compression;
- no answer considered successful merely because it is fluent.

#### GraphRAG and scientific evidence intelligence

- typed document, section, entity, claim, method, dataset, intervention, outcome, result, limitation, and citation nodes;
- provenance, support, contradiction, agreement, replication, lineage, and reference edges;
- generation-bound authoritative graph publication;
- cross-document bounded search;
- derived communities and summaries with explicit provenance;
- durable graph jobs, authority checks, retention, and safe compaction;
- scientific claim extraction, review custody, benchmark promotion, and rollback;
- PICO or PECO, study design, numerical results, units, intervals, effect sizes, risk of bias, retractions, and temporal contradiction evolution.

#### Evaluation and research science

- normalized corpus, query, and qrels contracts;
- Recall, Precision, MRR, MAP, nDCG, citation precision and recall, claim support, faithfulness, abstention, calibration, route, and multi-hop metrics;
- deterministic manifests and resumable immutable matrices;
- repeated seeds, paired confidence intervals, non-inferiority, practical-gain thresholds, and multiple-comparison handling;
- dataset cards, licenses, checksums, leakage checks, and split manifests;
- SciFact, NFCorpus, TREC-COVID, FiQA, ArguAna, CQADupStack, HotpotQA, 2WikiMultiHopQA, MuSiQue, PubMedQA, BioASQ, and structure-aware benchmarks where licensing permits;
- prompt-injection, OCR-noise, duplicate-evidence, stale-generation, tenant-isolation, corruption, and unsupported-citation suites;
- latency, throughput, memory, storage, provider cost, and energy measurement.

#### Distributed production

- privacy-conscious observability, SLOs, and failure taxonomies;
- shared durable queues, leases, idempotency keys, and transactional outbox or saga patterns;
- encrypted versioned object storage and KMS rotation;
- backup, restore, disaster-recovery, RPO, and RTO drills;
- dedicated model serving, batching, GPU scheduling, and circuit breakers;
- parser isolation, malware scanning, egress firewall, TLS ingress, secrets, quotas, and distributed rate limits.

## 3. What has already been implemented

### 3.1 Correctness and bounded contracts — implemented foundation

Implemented evidence includes `Pagerank.py`, `Searching.py`, `tools/config.py`, `server.py`, and their unit and integration tests.

Completed foundation:

- corrected probability-mass behavior and deterministic ranking contracts;
- finite numeric and bounded input handling;
- canonical owner, source, candidate, generation, and citation identities;
- explicit failures rather than silent optional-dependency corruption;
- compatibility layers for legacy callers where safe;
- integration-tested server response contracts.

Still required for completion:

- generated property tests across random graph topologies;
- schema-version compatibility matrices;
- exact-head release proof.

### 3.2 Storage, tenancy, and ingestion — implemented foundation

Implemented evidence includes `storage.py`, `tools/document_store.py`, `tools/document_service.py`, `ingest_docs.py`, and extensive boundary tests.

Completed foundation:

- owner-scoped retained-source storage;
- root and member identity validation;
- traversal, link, reparse-point, and unsafe-member rejection;
- bounded uploads, parsing, OCR, metadata, and chunks;
- privacy finalization before durable retrieval publication;
- recoverable records for partial ingestion failures;
- Windows fallback behavior that preserves root-integrity failures.

Still required:

- encrypted external object storage;
- KMS rotation and secure deletion;
- complete structure-aware parsing;
- malware scanning and parser sandbox integration;
- macOS identity tests and recurring corruption drills.

### 3.3 Four-store lifecycle — substantially implemented

Implemented evidence includes `tools/authoritative_document_index.py`, `tools/corpus_hybrid_retrieval.py`, `tools/due_scheduler.py`, and document reconciliation tests.

Completed foundation:

- coordinated retained-source, vector, sparse, and generation identities;
- durable jobs, leases, retries, compensation, and startup reconciliation;
- authoritative generation validation;
- owner-scoped replacement and deletion foundations;
- rejection of mixed or stale vector and sparse generations;
- operator-visible status and recovery contracts.

Remaining:

- one unified durable state machine for replacement, delete, restore, and cleanup;
- full crash injection at every phase;
- lifecycle journal retention and compaction;
- encrypted complete rollback artifacts;
- multi-process leader election or periodic reconciliation.

### 3.4 Retrieval foundation — implemented, not fully calibrated

Implemented evidence includes classic index/search modules plus `tools/embedding_*` and `tools/corpus_hybrid_retrieval.py`.

Completed foundation:

- TF-IDF and persistent sparse/BM25 paths;
- governed dense embedding profiles and registry;
- dense and sparse generation compatibility;
- weighted hybrid fusion and RRF;
- MMR, diversity, source caps, and component traces;
- heuristic and optional reranking interfaces;
- owner and provenance constraints.

Partial or missing:

- complete typed filter grammar;
- benchmark-calibrated fusion and field weights;
- governed reranker registry and cascades;
- SPLADE-family neural sparse retrieval;
- ColBERT-family late interaction;
- multilingual and domain promotion evidence;
- Matryoshka and multi-vector aggregation;
- retrieval-policy promotion and rollback.

### 3.5 Adaptive and multi-hop RAG — implemented foundation

Implemented evidence includes `tools/adaptive_retrieval.py`, `tools/adaptive_rag_tool.py`, `tools/decomposition_model.py`, `tools/adaptive_retrieval_runner.py`, `tools/confidence_calibration.py`, and trace stores.

Completed foundation:

- deterministic query features and route selection;
- evidence sufficiency decisions;
- bounded corrective attempts;
- bounded decomposition and subquestion DAGs;
- privacy-safe route traces;
- confidence, risk-coverage, and threshold primitives;
- offline adaptive-route fixtures and experiment contracts.

Partial or missing:

- explicit HyDE, step-back, and multi-query implementations;
- representative route and hop calibration;
- connected uploaded, web, and scholarly route evidence;
- domain-specific calibrator installation;
- adaptive policy registry, promotion, and rollback;
- bounded trace retention and cross-journal operator views.

### 3.6 Citations and answer verification — strong foundation, incomplete verifier

Implemented evidence includes `server.py`, `server_app.py`, `tools/evidence_graph_citations.py`, calibration modules, and server integration tests.

Completed foundation:

- server-owned citation IDs;
- owner and generation validation;
- exact source-region lineage;
- evidence-sufficiency and abstention primitives;
- critique and corrective-retrieval components.

Partial or missing:

- a single typed claim verifier contract;
- supported-revision end-to-end tests;
- constrained evidence packing;
- claim-conditioned compression that preserves citation spans;
- counterfactual citation and unsupported-fluency benchmark gates.

### 3.7 Evidence graph and GraphRAG — implemented foundation

Implemented evidence includes `tools/evidence_graph_builder.py`, `tools/evidence_graph_analysis.py`, `tools/evidence_graph_authority.py`, agent integration, operations CLI, and graph API integration tests.

Completed foundation:

- typed owner- and generation-bound graphs;
- exact digests and current pointers;
- fail-closed authority assessment;
- durable graph jobs, leases, retries, and audit classifications;
- cross-document graph sets and bounded search;
- derived graph analyses;
- privacy-safe operator reads.

Partial or missing:

- large-corpus construction and query benchmarks;
- schema migration tests;
- production entity linking and relation extraction adapters;
- benchmark-governed community summaries;
- temporal contradiction, agreement, replication, and retraction evolution;
- hyperedges and optional graph-neural derived reranking.

### 3.8 Scientific claim governance — implemented foundation

Implemented evidence includes scientific claim contracts, extraction, evaluation, reviewer custody, extractor benchmark, promotion, rollback, runtime, and CLI modules.

Completed foundation:

- exact claim source spans;
- extractor and schema identity;
- append-only reviewer corrections and custody;
- benchmark-governed promotion and rollback;
- graph integration;
- explicit distinction between structural provenance and scientific truth.

Missing depth:

- PICO or PECO;
- study design and risk-of-bias schemas;
- methods, datasets, outcomes, and limitations;
- numerical results, units, confidence intervals, denominators, and effect sizes;
- calibrated entailment and uncertainty;
- retraction feeds, replication links, and protocol-to-result traceability;
- expert-review workflows and domain promotion datasets.

### 3.9 Durable graph compaction — newly implemented

Implemented modules:

- `tools/evidence_graph_compaction.py`
- `tools/evidence_graph_compaction_runtime.py`
- `tools/evidence_graph_operations_cli.py`
- `tests/unit/test_evidence_graph_compaction.py`
- `config/evidence_graph.env.example`

Implemented safety properties:

- exact retention-plan digest confirmation;
- exact complete candidate job confirmation;
- revalidation of job identity and age;
- refusal of authoritative-current graphs;
- refusal of the current graph generation;
- historical graph digest verification;
- durable intent before deletion;
- crash resume between deletion and receipt completion;
- idempotency after completion;
- refusal of unexplained missing payloads;
- preservation of vector, sparse, generation, and graph-job audit state;
- audit-only treatment of cancelled terminal jobs;
- owner- and phase-filtered privacy-safe receipts.

This implementation is the reference design for lifecycle, trace, migration, and rollback-artifact retention.

### 3.10 Evaluation and experiment foundation — implemented but incomplete

Implemented evidence includes `evaluation/__init__.py`, `experiments/__init__.py`, adaptive route experiments, and extractor benchmarks.

Completed foundation:

- normalized evaluation contracts;
- retrieval and citation metrics;
- deterministic experiment identities;
- resumable immutable result concepts;
- calibration and route experiment primitives;
- some benchmark promotion logic.

Missing:

- complete licensed dataset adapters and cards;
- repository-wide experiment registry;
- leakage and contamination checks;
- standard repeated-seed statistical library;
- structure-aware and multimodal datasets;
- unified adversarial manifest;
- online shadow and controlled A/B evaluation;
- cost, memory, storage, and energy histories.

### 3.11 Release infrastructure — mostly operational, exact-head blocked

The authoritative workflow contains 16 job families for registration, Linux Python versions, Windows storage, container validation, and platform lock generation.

For run `31076973893` on baseline `ffbb3b14ec29ed3822f0f9f5c7fc5a7eefad1335`:

- registration smoke passed;
- all nine lock jobs passed;
- both Windows storage jobs passed;
- container validation passed;
- Linux Python 3.10, 3.11, and 3.12 reached the full test suite and failed on one shared deterministic behavioral regression.

Result: infrastructure repair is proven, but release certification remains **blocked** until the shared Linux failure is corrected and all jobs pass on one unchanged final SHA.

## 4. What remains, in implementation order

### P0 — exact-head integrity

Deliverables:

- isolate the shared Linux assertion or exception;
- patch source and add focused regression coverage;
- make the workflow reporter read-only so certification never mutates `main`;
- validate the capability ledger in CI;
- rerun the complete 16-job matrix on one unchanged SHA;
- record the exact successful run and SHA.

Definition of done: every authoritative job is green on the same head and the ledger may change `OPS-001.release` to `release_verified`.

### P1 — lifecycle completion

Deliverables:

- lifecycle retention and exact-confirmed compaction;
- rollback artifact schema, encryption, KMS rotation, and secure deletion;
- privacy-safe audit exports and cross-journal correlation;
- full fault injection for source, vector, sparse, generation, registry, cleanup, and graph phases;
- multi-process reconciliation and leader election.

Definition of done: every operation can crash at every mutation boundary and either resume idempotently or restore an exact authoritative state.

### P2 — governed migration execution

Deliverables:

- exclusive cutover lease and idempotency key;
- durable cutover journal;
- atomic or compensating vector, sparse, and generation publication;
- post-publication validation;
- automatic exact rollback;
- old-state retention until rollback verification;
- pause, resume, cancel, retry, and operator receipt semantics.

Definition of done: no failed phase can expose mixed state or destroy the last verified rollback generation.

### P3 — retrieval quality

Deliverables:

- complete typed filters;
- calibrated BM25 fields and dense/sparse fusion;
- reranker registry and budgeted cascades;
- SPLADE and ColBERT adapters;
- multilingual and domain profiles;
- Matryoshka and multi-vector experiments;
- retrieval policy promotion and rollback.

Definition of done: each promoted policy beats or is non-inferior to baselines under quality, latency, memory, and cost gates with repeated-seed evidence.

### P4 — adaptive verification

Deliverables:

- explicit query transforms;
- route and multi-hop calibration;
- verifier contract;
- constrained evidence packing;
- counterfactual citation suite;
- adaptive policy registry;
- connected route experiments;
- trace retention and dashboards.

Definition of done: the system can explain why it chose a route, why evidence is sufficient, which claims are supported, and why it abstained.

### P5 — graph and domain evidence depth

Deliverables:

- production entity and relation adapters;
- PICO or PECO and study-design schemas;
- numerical result and unit extraction;
- evidence quality and risk-of-bias review;
- temporal contradictions, retractions, and replication;
- governed community summaries;
- LegalRAG, biomedical, scientific, and financial schemas.

Definition of done: extracted structure is versioned, reviewable, benchmarked, uncertainty-aware, and never misrepresented as authoritative truth.

### P6 — multimodal structure

Deliverables:

- layout-preserving reading order;
- table cell grids;
- formulas and equation references;
- figure, panel, caption, and in-text links;
- OCR coordinates and confidence;
- visual embeddings and cross-modal late interaction;
- modality-aware chunks and citations;
- sandboxed parser execution.

Definition of done: every answer citation can point to exact text, table cell, formula, figure panel, or caption coordinates.

### P7 — experiment science

Deliverables:

- licensed adapters and data cards;
- immutable dataset and system manifests;
- leakage checks;
- repeated-seed ablations;
- paired bootstrap or permutation intervals;
- multiple-comparison controls;
- adversarial and corruption suites;
- cost, latency, memory, storage, and energy histories;
- online shadow evaluation.

Definition of done: every promoted architecture or policy has reproducible evidence and practical operational trade-offs.

### P8 — distributed production

Deliverables:

- shared SQL registries, queues, and leases;
- outbox or saga coordination;
- encrypted object storage and KMS;
- backup and disaster-recovery drills;
- model serving and GPU-aware batching;
- distributed quotas and idempotency;
- OpenTelemetry, Prometheus, SLO dashboards, and privacy budgets;
- TLS, secrets, egress firewall, parser isolation, and malware scanning.

Definition of done: node loss, duplicate delivery, partial store failure, key rotation, and restore drills preserve tenant isolation and exact authoritative identity.

## 5. Additional high-value capabilities beyond the original plan

These are justified additions, not immediate P0 work.

### 5.1 Retrieval architecture extensions

- learned sparse retrieval using SPLADE-family profiles;
- late interaction using ColBERT-family profiles;
- multilingual Indic and cross-lingual retrieval;
- domain adapters for science, biomedicine, law, and finance;
- Matryoshka dimension and latency ablations;
- multi-vector documents for sections, tables, figures, and long documents;
- tiered indexes with exact, compressed, and cold-storage layers;
- vector quantization and index-compression experiments;
- learned source-cap and diversity policies.

### 5.2 Verification and generation extensions

- constrained evidence packing as an optimization problem;
- claim-conditioned sentence selection;
- counterfactual citation tests;
- answerability and evidence-gap prediction;
- verifier ensembles with disagreement-aware abstention;
- support-preserving long-context reordering;
- position-bias and lost-in-the-middle evaluation;
- semantic cache entries bound to corpus generation, policy, and evidence digest;
- cache invalidation that fails closed on source changes.

### 5.3 Graph and scientific extensions

- temporal knowledge-graph snapshots;
- hyperedges for claims jointly supported by multiple evidence units;
- retraction and correction propagation;
- contradiction evolution over time;
- protocol, method, dataset, result, and limitation lineage;
- causal or methodological edges kept separate from textual entailment;
- active-learning queues for high-uncertainty reviewer cases;
- optional graph-neural reranking only as a derived, benchmarked feature;
- expert disagreement and adjudication records.

### 5.4 Evaluation extensions

- OCR corruption, section reordering, duplicate evidence, stale generations, and poisoned metadata;
- unsupported but fluent answers;
- swapped and cross-owner citations;
- selective prediction and calibration by language, domain, route, and document type;
- long-context position and distractor tests;
- expert review agreement and adjudication metrics;
- model, index, policy, prompt, parser, and dataset cards;
- historical regression dashboards.

### 5.5 Production and governance extensions

- provenance-bound semantic caching;
- privacy-preserving telemetry aggregation;
- policy-as-code for retention, authority, and promotion;
- cryptographic artifact manifests and optional signing;
- data residency and per-tenant encryption domains;
- right-to-erasure verification across every store and derived artifact;
- canary and shadow index publication;
- chaos and disaster-recovery test environments;
- capacity planning for CPU, GPU, vector memory, disk, queue depth, and provider budgets.

## 6. Recommended execution sequence

### Gate A — certify the repository

1. fix the Linux deterministic failure;
2. make exact-head reporting non-mutating;
3. validate the ledger in CI;
4. obtain one full green run.

### Milestone B — make data lifecycle complete

1. lifecycle compaction;
2. rollback custody;
3. fault injection;
4. multiprocess reconciliation.

### Milestone C — execute migrations safely

1. cutover journal;
2. publication state machine;
3. post-publication validation;
4. automatic rollback.

### Milestone D — prove retrieval quality

1. filters and calibrated fusion;
2. reranker registry;
3. SPLADE and ColBERT;
4. multilingual and domain profiles;
5. policy promotion.

### Milestone E — prove answer support

1. verifier contract;
2. evidence packing;
3. counterfactual citations;
4. calibrated adaptive policies.

### Milestone F — deepen scientific and multimodal evidence

1. PICO or PECO and numerical results;
2. temporal contradiction and replication;
3. structure-aware ingestion;
4. cross-modal retrieval and citations;
5. domain-specialized schemas.

### Milestone G — scale and operate

1. full dataset and experiment program;
2. shared queues and registries;
3. object custody and KMS;
4. observability and SLOs;
5. backup and disaster recovery;
6. isolated production deployment.

## 7. Definition of done for every future capability

A capability may be marked `implemented` only when it has:

1. a typed or otherwise explicit source contract;
2. bounded inputs and outputs;
3. owner, provenance, generation, and identity semantics where applicable;
4. deterministic behavior or documented nondeterministic controls;
5. unit tests including adversarial boundaries;
6. integration tests across every affected store or service;
7. configuration and secure defaults;
8. operator documentation and reason codes;
9. migration, rollback, retention, and failure behavior;
10. observability without privacy leakage.

It may be marked `experimentally_validated` only when it additionally has:

1. representative datasets and immutable manifests;
2. baseline comparisons;
3. repeated seeds or justified deterministic evaluation;
4. uncertainty intervals and practical-gain thresholds;
5. latency, memory, storage, and cost measurements;
6. leakage and contamination checks;
7. reproducible result artifacts.

It may be marked `release_verified` only when the complete authoritative workflow passes on the exact unchanged `main` SHA containing the capability and its audit update.

## 8. Repository and delivery policy

- All future work is committed directly to `main`.
- No new feature branches or pull requests are created.
- Historical merged and closed pull requests remain as audit provenance.
- Useful history is never force-pushed away.
- Only `main` should remain active.
- Documentation checkboxes are not authoritative; the validated capability ledger is.
- Every final status report must name the exact commit and workflow run it describes.

## 9. Immediate next implementation slice

The immediate next slice is unambiguous:

1. validate this ledger and its evidence paths in CI;
2. replace the mutating exact-head reporter with a read-only workflow summary;
3. extract and fix the common Linux full-suite regression;
4. rerun all 16 authoritative jobs on the resulting unchanged head;
5. only after that, begin `LIFE-002` lifecycle retention and compaction using `GRAPH-004` as the reference safety protocol.
