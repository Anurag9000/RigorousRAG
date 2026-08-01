# RigorousRAG exhaustive mission and live gap audit

Date: 2026-08-01  
Authoritative branch: `main`

This ledger reconstructs the project mission represented by the repository audit records, historical pull-request description, commit history, architecture documents and capability ledgers. It is an implementation audit, not a verbatim export of the prior chat transcript.

## 1. Repository and execution policy requested

The requested engineering policy is:

1. Preserve all valid work already completed in historical branches and pull requests.
2. Put the complete surviving implementation on `main`.
3. Commit all future work directly to `main` in coherent, documented commits.
4. Do not create feature branches or pull requests.
5. Do not force-push, rewrite or discard history.
6. Keep source, tests, configuration, documentation, status ledgers and experiment records synchronized.
7. Remove obsolete branches after their work is preserved.
8. Keep only `main` as the live branch.
9. Verify the exact final `main` head rather than extrapolating from an older successful run.

### Live policy audit

- The repository default branch is `main`.
- The branch inventory returned only `main`.
- Historical PRs #1–#4 are closed; their work is represented in `main` history.
- Current implementation work in this continuation is committed directly to `main`.
- No history rewrite or force update was used.

The structural main-only goal is therefore complete. Exact-head release verification remains open.

## 2. Original forensic-audit mission

The requested audit was broader than checking that the application starts. It required:

- reading every repository file, class, function, method, script, loop and configuration surface;
- reconstructing the intended product and experimental methodology;
- tracing data flow, control flow, state transitions, trust boundaries and failure paths;
- identifying missing, partial, placeholder, duplicated, dead, inconsistent or misleading implementation;
- checking mathematical, retrieval, ranking, evaluation and statistical logic;
- reviewing security, privacy, tenant isolation, provenance, citation authority and filesystem/network boundaries;
- checking concurrency, retries, timeouts, cancellation, crash recovery, persistence and cleanup;
- auditing parser, OCR, vector, sparse, graph, browser, API, CLI, container and release surfaces;
- correcting defects rather than merely listing them;
- adding regression tests for every corrected invariant;
- preserving reproducibility records and explicit limitations;
- rerunning the audit after implementation to catch regressions and documentation drift.

No finite audit can prove the absence of every defect. The repository therefore distinguishes committed contracts, focused test evidence and complete exact-head release certification.

## 3. Product mission reconstructed

RigorousRAG is intended to be an evidence-oriented scientific and academic research platform with these major product goals.

### 3.1 Classic academic retrieval

- resumable allowed-domain crawling;
- robots-policy support;
- bounded peer-validated remote requests and redirects;
- persisted pages, graph, TF-IDF index and PageRank;
- manifest-last generation commits with hashes and counts;
- Unicode/scientific tokenization;
- title-weighted sparse ranking and authority fusion;
- offline search without compulsory recrawling.

### 3.2 Uploaded-document RAG

- owner-scoped PDF, DOCX, Markdown and text ingestion;
- stable owner+source-byte document identity;
- deterministic parent/child chunks and semantic sections;
- dense retrieval, query expansion and document filtering;
- persistent fielded sparse retrieval;
- dense/sparse corpus fusion;
- generation/profile/content-hash validation before evidence publication;
- owner-scoped list, retrieval, replacement and deletion;
- compensating restoration when multi-store writes fail.

### 3.3 Parsing, OCR and retained evidence

- strict file, signature, extension, symlink/reparse and size validation;
- bounded PDF page/text/render complexity;
- bounded DOCX members, expansion and compression ratios;
- optional per-page OCR with partial provenance;
- text, title, metadata, filename and summary masking;
- private retained-source registry;
- current-byte identity validation before figure rendering;
- bounded visual page, pixel and encoded-payload handling.

### 3.4 Durable ingestion and lifecycle

- durable queued/processing/finalizing/success/failed states;
- atomic claims and duplicate-worker exclusion;
- retry ceilings and persisted exponential deadlines;
- one centralized scheduler rather than one timer per job;
- bounded executor admission;
- startup replay and explicit failure of invalid recovery state;
- owner-scoped public job status without private paths;
- source/vector/registry deletion and orphan reconciliation.

### 3.5 Agent and citation authority

- immutable request-scoped agent context;
- credential-derived owner identity;
- server model allowlist;
- bounded request, tool, turn, timeout, evidence and response budgets;
- schema validation of model-authored tool arguments;
- retrieved text treated as untrusted evidence;
- server-owned evidence registry and deterministic citation labels;
- no model authority to invent citation objects;
- retrieval-only fallback and explicit outage/no-match distinction.

### 3.6 Scientific evidence tools

- figure/caption visual entailment support;
- conservative protocol extraction;
- advocate/skeptic/judge analysis with original evidence retained;
- cross-paper comparison matrices;
- contradiction-versus-context-difference analysis;
- limitation extraction;
- deterministic escaped BibTeX;
- structural provenance explicitly separated from semantic truth.

### 3.7 Service, browser, CLI and deployment

- FastAPI API and bounded request framing;
- authenticated multi-user and isolated single-user modes;
- browser UI without third-party runtime scripts/fonts;
- safe DOM rendering rather than untrusted `innerHTML`;
- session-only credentials/history;
- bounded CLI arguments and terminal-safe output;
- non-root read-only container, dropped capabilities and loopback default;
- readiness checks for HTTP, SQLite and writable state volumes.

### 3.8 Reproducibility and release

- immutable platform/Python requirements snapshots;
- hash-pinned lock generation and verification;
- no alternate package-index or local-path authority in release inputs;
- immutable workflow-action pins;
- Linux/Windows/macOS and Python-version matrices;
- dependency, compile, lint, test, coverage, Compose, image-build and smoke gates;
- exact-head evidence before any release-ready claim.

## 4. Completed implementation audit

### 4.1 Security and reliability remediation — substantially implemented

Historical remediation and continuation passes implemented contracts across:

- tenant identity and owner isolation;
- request/body/executor/tool/result ceilings;
- upload anchoring and link/reparse refusal;
- durable ingestion and retry scheduling;
- parser/OCR/archive complexity limits;
- privacy masking and control-character handling;
- source registry and immutable visual evidence;
- citation URL/page authority;
- connected-peer SSRF and redirect controls;
- classic-index generation integrity;
- frontend asset and DOM safety;
- readiness and hardened container deployment;
- release-lock generation and verification code.

Residual limitations are explicitly documented rather than falsely marked complete.

### 4.2 Retrieval and evaluation foundation — implemented

- typed candidates and component traces;
- BM25 candidate ranking;
- weighted and reciprocal-rank fusion;
- MMR and source diversity;
- heuristic and optional cross-encoder reranking;
- BEIR-style datasets and retrieval/citation metrics;
- deterministic experiment matrices and resumable result storage;
- offline benchmark CLI.

### 4.3 Embedding and sparse-index governance — implemented

- built-in profiles for MiniLM, E5, BGE, GTE, Instructor, SPECTER2 and BGE-M3;
- aliases and stable fingerprints;
- strict operator profiles;
- persistent owner/document/generation-isolated sparse index;
- field/page/section/frequency/position provenance;
- transactional snapshot, restore, replacement and deletion;
- path/database identity checks.

### 4.4 Authoritative multi-store generations — implemented foundation

- vector snapshots and exact restoration;
- deterministic sparse fields;
- vector+sparse compensation;
- append-only durable generation history;
- optimistic current pointers;
- one owner/document lock across vector, sparse and generation stores;
- authoritative ingestion/deletion/batch rollback;
- drift scans and bounded reconciliation planning/CLI;
- public retrieval refusal of stale/deleted/misaligned generations.

### 4.5 Corpus-level hybrid retrieval — implemented foundation

- independent dense and sparse candidate generation;
- document-level fusion;
- generation/profile/content-hash validation;
- dense chunk and sparse-field materialization;
- MMR, optional reranking and component traces;
- `corpus-sparse` and `corpus-hybrid` modes;
- expanded-query failure containment;
- protected citation metadata.

### 4.6 Migration control plane — implemented planning/journal layer

- profile-drift inventory;
- retained-source eligibility classification;
- deterministic task IDs;
- durable migration journal;
- idempotent seeding;
- leases, renewals, retry limits and recovery;
- validation digests;
- owner-verified operator cancellation;
- inventory/seed/status/cancel commands;
- deliberate refusal to cut over before shadow validation exists.

### 4.7 Adaptive and corrective RAG — implemented foundation

- query intent and complexity analysis;
- exact/comparison/temporal/quantitative/method/evidence/explanation routing;
- evidence sufficiency based on count, document diversity, scores, provenance and generations;
- bounded corrective plans and execution;
- accumulated-evidence deduplication;
- per-attempt traces and contained errors;
- public adaptive uploaded-document tool;
- conservative abstention;
- Brier/ECE/reliability reports;
- isotonic calibration;
- risk-coverage and abstention-threshold selection.

### 4.8 Decomposition, budgeted multi-hop retrieval and evaluation — implemented in this continuation

- bounded subquestions;
- explicit or deterministic heuristic decomposition;
- strict closed-schema model-assisted decomposition;
- provider-response digesting without retaining model-authored evidence;
- deterministic fallback on malformed or unavailable model output;
- token/entity/time/redundancy/parallelism/depth plan diagnostics;
- entity and temporal constraints;
- validated DAGs with cycle/dangling/duplicate refusal;
- stable fingerprints and topological batches;
- parallel independent and serial dependent execution;
- bounded workers, deadlines and evidence;
- dependency-evidence separation;
- per-hop source/document/page lineage;
- joins without citation laundering;
- bounded dependency-derived lexical hints;
- hard global estimated-cost allocation across all uploaded-document hops;
- minimum-attempt reservation and impossible-budget refusal before retrieval;
- weighted remainder allocation, per-hop caps and unused-budget accounting;
- public adaptive multi-hop result with plan, quality, budget, traces and lineage;
- terminal-evidence abstention;
- answer exact match and Unicode token F1;
- document and support precision, recall and F1;
- complete support-path, hop-coverage and citation-lineage metrics;
- abstention-aware macro aggregation;
- explicitly heuristic answer-support scoring;
- 30 focused local tests passing.

## 5. Partial or open implementation

### 5.1 Exact-head verification

This is the highest-priority unresolved gate. The complete configured workflow has not been observed green on the latest exact `main` SHA through the available connector. The local constrained environment cannot resolve GitHub hosts, so it cannot clone the live repository or execute a clean-clone matrix. No release-ready claim is made.

### 5.2 Four-store transaction completion

The retained-document registry is not yet a fully coordinated fourth participant or durable outbox consumer alongside vector, sparse and generation stores. Startup reconciliation, resumable repair execution and adoption/reindex of pre-manifest state remain open.

### 5.3 Corpus retrieval calibration

Fusion weights, independent-corpus RRF, filters, source caps, reranker cascades and ablation thresholds require representative benchmark evidence.

### 5.4 Profile migration execution

The journal/control plane exists, but shadow vector+sparse creation, privacy-finalized retained-source reparsing, target encoder construction, artifact validation, atomic cutover, rollback retention and phase-by-phase fault injection remain open.

### 5.5 Adaptive policy calibration

The deterministic policy and calibration tools exist, but the coefficients and thresholds have not been promoted from representative repeated benchmark runs into a versioned runtime calibrator. Web/scholarly routing and explicit domain policies remain open.

### 5.6 Multi-hop integration and benchmark completion

The uploaded-document planning, cost-allocation, execution and structural evaluation foundation exists. Remaining work includes learned plan ranking, entity resolution, normalized temporal ranges, heterogeneous uploaded/web/scholarly backends, measured cross-backend latency/token/monetary budgets, HotpotQA/2WikiMultiHopQA/MuSiQue/scientific adapters, semantic support/entailment metrics, ablation thresholds and full agent/API/browser registration.

### 5.7 Evidence graph

Typed claim/entity/method/dataset/citation graphs, graph construction, graph retrieval, path explanations, support/contradiction clustering and bounded GraphRAG summaries remain open.

### 5.8 Multimodal scientific ingestion

Advanced reading order, layout parsing, table cells, formulas, figure-panel association and multimodal retrieval remain open. Existing PDF/OCR/visual support is bounded but heuristic.

### 5.9 Scientific evidence intelligence

PICO/PECO schemas, structured methods/populations/interventions/outcomes/results, effect-size normalization, risk-of-bias fields and human correction lineage remain open.

### 5.10 Model and dataset expansion

Adapter interfaces for learned sparse models, late-interaction retrieval, listwise reranking and multilingual/scientific models remain open, as do dataset cards and license/version/checksum governance.

### 5.11 Experimentation and observability

Repeated seeds, confidence intervals, statistical tests, practical-effect thresholds, resource/cost measurements, dashboards and historical promotion gates remain open.

### 5.12 Distributed production architecture

The current architecture is deliberately single-host/process-local. Distributed queueing, SQL registry/outbox, object storage, distributed rate limiting, idempotency/effect coordination, TLS/egress/secret-manager integration, malware scanning and parser sandboxing remain open.

## 6. Additional models, architectures and pipelines recommended

These additions are useful only when governed by datasets, budgets and promotion gates rather than added as unmeasured names.

### Retrieval models

- SPLADE and uniCOIL learned sparse adapters;
- ColBERTv2 and PLAID late-interaction retrieval;
- multilingual E5/BGE-M3 and domain-specialized scientific embeddings;
- SPECTER2 variants for citation/scientific neighborhood retrieval;
- GritLM/NV-Embed-class profiles where license and hardware permit;
- query/document asymmetric adapters with explicit normalization contracts.

### Reranking

- cross-encoder MiniLM and BGE rerankers;
- scientific-domain rerankers;
- listwise LLM reranking only behind strict cost, latency and citation-preservation limits;
- cascade policies: heuristic → compact cross-encoder → expensive judge only when uncertainty warrants it.

### Query transformation

- schema-constrained multi-query expansion;
- HyDE ablations by domain and query class;
- step-back prompting as retrieval planning, not evidence;
- acronym/identifier normalization;
- multilingual query translation with original-language provenance;
- temporal and entity-aware expansion.

### Evidence architectures

- RAPTOR-style hierarchical summaries with source lineage;
- parent-document and sentence-window retrieval;
- contextual chunk enrichment with explicit generated-context labels;
- GraphRAG over typed, provenance-preserving nodes;
- claim-level retrieval and support-set construction;
- contradiction-aware diversified retrieval.

### Multimodal models

- Docling/Marker-style layout pipelines where licensing permits;
- LayoutLMv3/DiT-like layout representations;
- Table Transformer/DETR-style table detection and structure recognition;
- Nougat/LaTeX OCR evaluation for formulas;
- figure-caption-panel association models;
- CLIP/SigLIP-style figure retrieval with page/caption lineage.

### Scientific extraction

- SciSpacy/SciBERT entity and relation adapters;
- structured PICO/PECO extraction;
- effect-size and uncertainty parsers;
- citation-context intent classification;
- method/dataset/metric/result schemas;
- risk-of-bias support with mandatory human review.

## 7. Datasets and benchmark categories recommended

### General retrieval

- BEIR tasks appropriate to license and availability;
- MS MARCO passage/document subsets;
- Natural Questions and TriviaQA for retrieval diagnostics;
- LoTTE for domain transfer.

### Scientific retrieval

- SciFact;
- SCIDOCS;
- TREC-COVID;
- NFCorpus;
- ArguAna and Touché where argumentative retrieval is relevant;
- citation-context and paper-recommendation datasets.

### Multi-hop

- HotpotQA;
- 2WikiMultiHopQA;
- MuSiQue;
- QASC;
- StrategyQA only with careful evidence auditing;
- custom cross-paper scientific comparison and contradiction sets.

### Citation and faithfulness

- citation precision/recall datasets;
- claim-evidence entailment sets;
- adversarial unsupported-citation tests;
- source-mutation and stale-generation regressions;
- citation-laundering and prompt-injection suites.

### Multilingual and Indic

- MIRACL;
- Mr.TyDi;
- mMARCO;
- Indic question/evidence corpora with language-specific tokenization and evaluation;
- translated-versus-native retrieval ablations.

### Multimodal/document

- DocVQA;
- InfographicVQA;
- PubTables-1M;
- PubLayNet;
- ChartQA;
- formula and scientific-figure datasets with explicit license records.

### Robustness and security

- malformed PDF/DOCX/archive corpora;
- decompression-bomb and parser-complexity fixtures;
- cross-owner metadata attacks;
- hostile provider payloads;
- SSRF/redirect/proxy test fixtures;
- prompt-injection, citation-spoofing and data-exfiltration suites.

## 8. Required experimental methodology

Every model or pipeline promotion should record:

- immutable dataset card, version, license and checksums;
- train/dev/test split identities;
- repeated seeds where stochasticity exists;
- recall@k, precision@k, MRR, nDCG and MAP as applicable;
- citation precision, citation recall, source coverage and provenance validity;
- answer support/entailment and abstention risk-coverage;
- latency percentiles, throughput, memory, storage and monetary cost;
- confidence intervals and paired statistical tests;
- practical-effect thresholds, not only p-values;
- failure categories and bounded artifacts;
- exact model/profile fingerprints and software commit;
- promotion/rejection decision and rollback reference.

## 9. Verification performed in this continuation

- Confirmed through the GitHub connector that only `main` is present.
- Confirmed historical PRs are closed.
- Audited current architecture, TODO, status, recent commits and key retrieval/adaptive surfaces.
- Identified stale Wave 3 status and historical PR-state documentation.
- Added deterministic and strict model-assisted decomposition.
- Added global estimated-cost allocation across multi-hop adaptive retrieval.
- Added provenance-preserving parallel/serial multi-hop execution and public result payloads.
- Added answer, document, support-path, hop, citation-lineage and abstention metrics.
- Added focused tests.
- Ran local focused tests: 30 passed.
- Ran Python compilation for the six new modules and focused tests: passed.
- Ruff was unavailable locally.
- Clean-clone and complete repository verification were not possible because GitHub DNS resolution fails in the execution environment.

## 10. Next dependency-ordered implementation sequence

1. Register and integration-test adaptive/multi-hop tools across the agent/API response boundary.
2. Complete the retained-registry/outbox and startup repair path.
3. Implement shadow migration execution and atomic cutover.
4. Build benchmark adapters and calibrated hybrid/adaptive/multi-hop experiment suites.
5. Add filters, source caps and reranker cascades.
6. Add heterogeneous multi-hop routing and measured cross-backend budgets.
7. Add semantic support/entailment evaluation and promotion thresholds.
8. Build typed evidence graph and scientific extraction schemas.
9. Expand multimodal scientific parsing.
10. Add full observability and promotion gates.
11. Execute the unchanged exact-head release matrix and only then declare release readiness.

The canonical machine-actionable backlog is `docs/TODO.md`; the implementation ledger is `docs/CAPABILITY_IMPLEMENTATION_STATUS.md`.
