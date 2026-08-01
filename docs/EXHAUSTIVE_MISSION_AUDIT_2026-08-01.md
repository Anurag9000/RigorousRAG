# RigorousRAG exhaustive mission and live gap audit

Date: 2026-08-01  
Authoritative branch: `main`

This document reconstructs the project mission from the historical remediation records, closed pull requests, commit history, architecture documents, implementation ledgers and live repository tree. It is an implementation audit, not a verbatim export of a previous chat.

## 1. Requested repository policy

The requested policy was to:

1. preserve every valid change from earlier branches and pull requests;
2. consolidate the complete surviving project on `main`;
3. commit all future work directly to `main`;
4. create no new feature branches or pull requests;
5. avoid force-pushing and history rewriting;
6. keep code, tests, configuration, documentation, specifications and experiment records synchronized;
7. remove obsolete branches only after preserving their work;
8. retain only `main` as the live branch;
9. verify one unchanged exact `main` head before claiming release readiness.

### Live audit

- Default and authoritative branch: `main`.
- Branch inventory: only `main`.
- Open pull requests: none.
- Historical PRs #1–#4: closed.
- Current continuation: direct coherent commits to `main`.
- No force update or history rewrite was used.

The structural main-only mission is complete. Exact-head release certification remains open.

## 2. Original forensic-audit mission

The user asked for substantially more than a superficial code review. The requested mission included:

- read and understand every repository file, class, function, method, script, configuration and test surface;
- reconstruct the intended product, research methodology and trust model;
- trace data flow, control flow, persistence, state transitions, retries, cancellation and failure recovery;
- identify incomplete, placeholder, duplicated, dead, misleading, unsafe or mathematically incorrect behavior;
- audit retrieval, ranking, reranking, fusion, calibration, evaluation and statistical methodology;
- audit privacy, tenant isolation, provenance, citation authority, filesystem and network boundaries;
- audit parser/OCR complexity, upload handling, retained evidence and visual rendering;
- audit concurrency, queues, timeouts, resource ceilings, cleanup and crash behavior;
- audit agent, API, CLI, browser, container, dependency-lock and release surfaces;
- fix defects rather than only list them;
- add regression tests for every newly enforced invariant;
- document limitations and non-claims explicitly;
- propose and implement additional useful models, architectures, pipelines, datasets, experiments, features and tasks;
- rerun the audit after implementation and keep the status ledger synchronized.

No finite review can prove the absence of all defects. The repository therefore separates committed contracts, focused test evidence and complete exact-head release certification.

## 3. Product mission reconstructed

RigorousRAG is intended to become an evidence-oriented scientific and academic research platform with the following major systems.

### 3.1 Classic academic search

- allowed-domain resumable crawling;
- robots-policy support;
- bounded peer-validated remote requests and redirects;
- persisted pages, graph, TF-IDF and PageRank;
- manifest-last immutable generations with hashes/counts;
- scientific/Unicode tokenization;
- title-weighted sparse ranking and authority fusion;
- offline search without compulsory recrawling.

### 3.2 Uploaded-document RAG

- owner-scoped PDF, DOCX, Markdown and text ingestion;
- stable owner+source-byte document identity;
- deterministic parent/child chunks and semantic sections;
- dense and fielded-sparse retrieval;
- hybrid corpus fusion and reranking;
- generation/profile/content-hash validation before publication;
- owner-scoped list, replace, retrieve and delete;
- compensating restoration across persistent stores.

### 3.3 Parsing, OCR and retained evidence

- strict file signature, extension, size and path checks;
- symlink/reparse refusal;
- bounded PDF/DOCX/archive complexity;
- optional bounded per-page OCR with partial provenance;
- privacy masking over text, titles, metadata, filenames and summaries;
- private retained-source registry;
- byte-identity validation before visual rendering;
- page/pixel/payload limits for figures.

### 3.4 Durable lifecycle

- durable queued/processing/finalizing/success/failed jobs;
- atomic claims and duplicate-worker exclusion;
- retry ceilings and persisted backoff;
- centralized scheduling;
- bounded executor admission;
- startup replay and invalid-state refusal;
- owner-scoped public job status without private paths;
- deletion and orphan reconciliation.

### 3.5 Agent and citation authority

- immutable request-scoped context;
- credential-derived owner identity;
- server model allowlist;
- bounded request/tool/turn/time/evidence/response budgets;
- schema validation of model-authored arguments;
- retrieved text treated as untrusted data;
- server-owned evidence registry and deterministic citations;
- no model authority to invent citation objects;
- explicit no-match/outage/fallback behavior.

### 3.6 Scientific evidence intelligence

- figure/caption visual support;
- protocol extraction;
- advocate/skeptic/judge analysis retaining original evidence;
- cross-paper comparison and conflict analysis;
- limitation extraction;
- deterministic escaped BibTeX;
- explicit separation of structural provenance from semantic truth.

### 3.7 Service, browser, CLI and deployment

- bounded FastAPI request framing;
- authenticated multi-user and isolated single-user operation;
- local browser assets and safe DOM rendering;
- session-only credentials/history;
- bounded terminal-safe CLI output;
- non-root read-only containers with dropped capabilities and loopback defaults;
- readiness checks over HTTP, stores and writable volumes.

### 3.8 Reproducibility and release

- immutable platform/Python requirement snapshots;
- hash-pinned lock generation/verification;
- no alternate-index or local-path release authority;
- immutable workflow pins;
- Linux/Windows/macOS and Python-version matrices;
- dependency, compile, lint, test, coverage, Compose, image and smoke gates;
- one unchanged exact-head result before release claims.

## 4. Work already completed before this continuation

### Security and reliability remediation

Substantial contracts already existed for tenant identity, request/body/tool/result ceilings, safe upload anchoring, durable ingestion, parser/OCR limits, masking, retained-source identity, citation authority, SSRF/redirect defenses, classic-generation integrity, frontend safety, readiness, hardened deployment and release-lock generation.

### Retrieval and evaluation foundation

- typed candidates and component traces;
- candidate-pool BM25;
- weighted and reciprocal-rank fusion;
- MMR and source diversity;
- heuristic/optional cross-encoder reranking;
- BEIR-style loaders and retrieval/citation metrics;
- deterministic experiment matrices and resumable records;
- offline baseline CLI.

### Embedding and sparse governance

- MiniLM, E5, BGE, GTE, Instructor, SPECTER2 and BGE-M3 profiles;
- aliases and stable fingerprints;
- strict operator profiles;
- persistent owner/document/generation-isolated fielded sparse index;
- page/section/field/frequency/position provenance;
- transactional replace/snapshot/restore/delete;
- path/database identity defenses.

### Authoritative generations

- bounded vector snapshots and exact restoration;
- deterministic sparse fields;
- vector+sparse compensation;
- append-only generation history and optimistic current pointers;
- one owner/document lock across vector, sparse and manifest stores;
- authoritative ingestion/deletion and batch rollback;
- drift scans and bounded reconciliation CLI;
- refusal to publish stale/deleted/misaligned generations.

### Corpus hybrid retrieval

- independent dense and sparse candidates;
- document-level fusion;
- generation/profile/hash validation;
- dense chunk and sparse field materialization;
- MMR and optional reranking;
- `corpus-sparse` and `corpus-hybrid` modes;
- expanded-query failure containment;
- protected citation metadata.

### Migration control plane

- profile-drift inventory;
- retained-source eligibility;
- deterministic task IDs;
- durable migration journal;
- idempotent seeding;
- leases, renewal, retries and recovery;
- validation digests;
- owner-verified cancellation;
- inventory/seed/status/cancel commands;
- deliberate refusal to cut over before shadow validation exists.

### Adaptive/corrective retrieval and calibration

- query intent/complexity analysis;
- evidence sufficiency signals;
- bounded corrective plans/execution;
- evidence deduplication and trace records;
- public adaptive uploaded-document tool;
- conservative abstention;
- Brier score, reliability bins, ECE and maximum gap;
- isotonic calibration;
- risk-coverage and abstention threshold selection.

## 5. New work completed in this continuation

### 5.1 Documentation and status repair

The old README/remediation language still described a historical draft PR, and the TODO incorrectly marked implemented adaptive work as unfinished. The live README, remediation record, capability ledger, TODO and multi-hop design document were replaced/synchronized with the current `main` state.

### 5.2 Deterministic query decomposition

- bounded subquestion schema;
- explicit-plan and deterministic heuristic planning;
- entity and temporal constraint extraction;
- duplicate, dangling and cyclic dependency refusal;
- stable SHA-256 plan fingerprints;
- topological parallel batches and terminal nodes.

### 5.3 Strict model-assisted decomposition

- one bounded OpenAI-compatible planning call;
- closed planning-only JSON schema;
- unsupported answer/citation/URL fields rejected;
- provider response retained only as a SHA-256 digest;
- deterministic fallback on provider/schema failure;
- generic failure types without provider detail leakage;
- token/entity/time/redundancy/parallelism/depth diagnostics.

### 5.4 Provenance-preserving multi-hop executor

- parallel independent hops and serial dependent batches;
- bounded workers, timeout, results, dependency evidence and total evidence;
- dependency evidence supplied separately from search text;
- failure/timeout containment;
- prerequisite-evidence skip policy;
- immutable hop/source/document/page lineage;
- cross-hop document/source grouping without synthetic citations;
- terminal-path abstention.

### 5.5 Public adaptive multi-hop tool

- deterministic or strict model-assisted planning;
- adaptive uploaded-document retrieval per hop;
- explicit entity/time constraints;
- bounded dependency-derived lexical hints;
- no raw prerequisite passage concatenation;
- separate citation and lineage payloads;
- plan, quality, budget, trace, join and terminal status output.

### 5.6 Hard global estimated-cost allocation

A concrete bug class was removed: a per-hop ceiling can silently multiply by the number of subquestions. The new allocator:

- computes every hop's minimum viable adaptive attempt;
- rejects per-hop limits below a minimum;
- rejects global totals below the sum of all minima before retrieval;
- reserves all minima;
- allocates remaining capacity deterministically using DAG/relation complexity;
- enforces per-hop caps;
- reports unused capacity;
- records exact accounting in the public result.

The estimate remains a workload proxy, not measured tokens, latency or money.

### 5.7 Multi-hop evaluation

- Unicode answer normalization;
- exact match and token F1;
- document precision/recall/F1;
- support precision/recall/F1;
- page, section, field, source, sentence and paragraph locators;
- complete support-path rate;
- hop coverage;
- citation-lineage validity;
- abstention rate and bounded macro aggregation;
- clearly labeled heuristic answer-support score.

### 5.8 Strict benchmark adapters

Local-only adapters now exist for:

- HotpotQA;
- 2WikiMultiHopQA;
- MuSiQue JSON and JSONL.

They preserve answer aliases and sentence/paragraph support facts, record exact-byte SHA-256 fingerprints, bound bytes/examples/nesting, require UTF-8, reject duplicate JSON keys and NaN/Infinity, validate support references and duplicate IDs, and reject symlink/reparse/non-regular paths. They do not download datasets or establish licensing.

### 5.9 Adaptive trace persistence and route experiments

Concurrent `main` commits were preserved and incorporated:

- privacy-safe owner/run-isolated adaptive SQLite trace storage;
- optional `ADAPTIVE_TRACE_DB_PATH` configuration;
- query hashing and aggregate-only trace persistence;
- strict private-key filtering in public adaptive payloads;
- reproducible offline dense/sparse/hybrid/web/scholarly route fixtures;
- query/evidence-free reports;
- router/oracle success, route accuracy, utility, cost/latency proxy and regret metrics.

These establish a harness, not calibrated production routing.

## 6. Verification performed

- confirmed only `main` is live;
- confirmed no open pull requests;
- preserved concurrent `main` commits rather than overwriting them;
- reviewed current architecture, recent commits, TODO/status and key agent/retrieval surfaces;
- added seven new multi-hop modules and corresponding focused tests;
- ran **35 focused tests: all passed**;
- ran Python compilation for the seven new modules/tests: passed;
- Ruff was unavailable in the constrained local environment;
- clean-clone/full-suite/container/cross-platform exact-head verification was impossible because the execution environment cannot resolve GitHub hosts;
- no release-ready claim is made.

## 7. Remaining work, dependency ordered

### 7.1 Exact-head release gate

Run dependency installation, `pip check`, whitespace/artifact checks, compilation, Ruff, full pytest/coverage on Python 3.10–3.12, Windows storage regressions, Compose validation, image build/readiness smoke, clean-clone ingestion/deletion/reconciliation/retrieval tests, concurrency/fault injection and one final line-by-line audit on the same unchanged SHA.

### 7.2 Full adaptive/multi-hop agent integration

The live legacy dispatcher still exposes classic uploaded-document search but does not yet safely register and dispatch the adaptive and multi-hop schemas through the entire agent/API/browser citation boundary. This large security-sensitive integration requires full-file tests and should not be performed as an unverified partial rewrite.

### 7.3 Fourth transaction participant and repairs

- coordinate retained-document registry through a transaction participant or durable outbox;
- startup reconciliation before retrieval;
- resumable repair journal and operator audit trail;
- retained-source reindex and carefully reviewed pre-manifest adoption.

### 7.4 Profile migration execution

- isolated shadow vector/sparse generations;
- privacy-finalized retained-source reparse;
- explicit target encoder construction;
- count/hash/provenance/quality/resource validation;
- durable shadow identities;
- atomic cutover and rollback references;
- retention cleanup and phase-by-phase fault injection.

### 7.5 Retrieval calibration and cascades

- benchmark fusion weights and independent-corpus RRF;
- date/MIME/field/page/section/provenance filters;
- per-document/source caps;
- heuristic → compact cross-encoder → expensive judge cascades;
- repeated ablations, confidence intervals and practical promotion thresholds.

### 7.6 Adaptive policy completion

- representative coefficient/threshold calibration;
- versioned runtime calibrators;
- validated domain classifier;
- connected production-scale route experiments;
- repeated seeds and promotion gates;
- trace retention/compaction/export/dashboard policy.

### 7.7 Multi-hop completion

- learned plan ranking;
- entity resolution and normalized temporal ranges;
- uploaded/web/scholarly heterogeneous hops;
- measured latency/token/monetary allocation across backends;
- custom scientific multi-document datasets/cards;
- semantic claim-support/entailment metrics;
- historical regression thresholds.

### 7.8 Evidence graph

- typed document, section, claim, entity, method, dataset and citation nodes;
- provenance-preserving edges;
- graph construction from authoritative generations;
- path retrieval/explanations;
- support/contradiction clusters;
- bounded GraphRAG summaries.

### 7.9 Multimodal scientific ingestion

- reading order/layout models;
- table detection/structure/cell provenance;
- formula detection/OCR/normalization;
- figure-caption-panel association;
- quality gates and multimodal retrieval/citations.

### 7.10 Scientific evidence intelligence

- PICO/PECO and structured research-question schemas;
- methods/population/intervention/outcome/result extraction;
- effect-size and uncertainty normalization;
- risk-of-bias/evidence-quality fields;
- claim/evidence/limitation/conflict linking;
- human review and correction lineage.

### 7.11 Production distribution

- durable distributed queue/leases;
- SQL registry and outbox/saga coordination;
- object storage;
- distributed admission/rate limits;
- idempotency and exactly-once-effect design;
- TLS, egress policy, secret manager, malware scanning and parser sandboxing.

## 8. Recommended additional models and architectures

Only add these behind dataset, license, budget and promotion gates:

- learned sparse: SPLADE, uniCOIL;
- late interaction: ColBERTv2, PLAID;
- multilingual/scientific embeddings: multilingual E5, BGE-M3, SPECTER2 variants and governed GritLM/NV-Embed-class profiles;
- cross-encoder/listwise reranker interfaces and uncertainty-triggered cascades;
- schema-constrained multi-query, HyDE, step-back, acronym/identifier, multilingual and temporal transformations;
- RAPTOR-style hierarchy, parent-document/sentence-window retrieval, contextual chunk labels, claim-level retrieval and contradiction-aware diversification;
- provenance-preserving GraphRAG;
- LayoutLMv3/DiT-like layout models, Table Transformer, Nougat/formula OCR and CLIP/SigLIP figure retrieval;
- SciSpacy/SciBERT extraction, PICO/PECO, effect-size parsing, citation-intent classification and mandatory-review risk-of-bias assistance.

## 9. Recommended datasets and experiments

### Retrieval/scientific

BEIR, MS MARCO, Natural Questions, TriviaQA, LoTTE, SciFact, SCIDOCS, TREC-COVID, NFCorpus, ArguAna and Touché where licensing and task fit permit.

### Multi-hop

HotpotQA, 2WikiMultiHopQA and MuSiQue adapters now exist. Future work should add QASC, carefully audited StrategyQA and custom cross-paper scientific comparison/contradiction sets.

### Multilingual/Indic

MIRACL, Mr.TyDi, mMARCO and native Indic question/evidence corpora with language-specific tokenization and translated-versus-native ablations.

### Multimodal/document

DocVQA, InfographicVQA, PubTables-1M, PubLayNet, ChartQA and governed formula/scientific-figure datasets.

### Security/robustness

Malformed PDF/DOCX/archive corpora, decompression bombs, parser-complexity fixtures, tenant-isolation attacks, hostile provider payloads, SSRF/redirect/proxy cases, prompt injection, citation spoofing and data-exfiltration tests.

Every promotion should record exact dataset/version/license/checksum/splits, seeds, retrieval/citation/support/abstention metrics, latency/throughput/memory/storage/cost, confidence intervals, paired tests, practical-effect thresholds, failure categories, model/profile fingerprints, commit SHA and rollback decision.

## 10. Canonical status sources

- Machine-actionable backlog: `docs/TODO.md`
- Capability ledger: `docs/CAPABILITY_IMPLEMENTATION_STATUS.md`
- Multi-hop design: `docs/MULTIHOP_RETRIEVAL.md`
- Live remediation/release status: `docs/REMEDIATION_STATUS.md`

The next safest dependency-ordered implementation is full adaptive/multi-hop agent/API integration under focused dispatcher/citation tests, followed by fourth-store repair coordination and shadow migration execution.
