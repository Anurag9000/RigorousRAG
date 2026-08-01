# Current remediation and implementation status

Last updated: 2026-08-01

This document replaces the historical draft-PR status. It reports the live repository state and distinguishes committed implementation from exact-head release certification.

## Repository state

- Authoritative branch: `main`
- Default branch: `main`
- Live branch inventory: only `main`
- Historical pull requests: #1–#4 closed
- Development policy: coherent direct commits to `main`; no new feature branches or pull requests
- History policy: no force-push or history rewriting

The previous `agent/exhaustive-remediation` branch and draft-PR workflow are historical. Their surviving work is represented in `main` history.

## Current product surfaces

The live repository covers:

- classic allowed-domain crawling, TF-IDF, PageRank and generation persistence;
- PDF/DOCX/Markdown/text ingestion, bounded OCR and privacy masking;
- private retained-source registry and visual evidence validation;
- durable jobs, retries, centralized scheduling and startup replay contracts;
- owner-scoped vector retrieval and document lifecycle;
- persistent fielded sparse retrieval;
- authoritative vector+sparse+generation coordination;
- generation-validated corpus hybrid retrieval;
- embedding profiles and migration-control planning;
- adaptive/corrective retrieval, evidence sufficiency and abstention;
- privacy-safe adaptive trace persistence;
- offline dense/sparse/hybrid/web/scholarly route experiments;
- confidence calibration and risk-coverage analysis;
- bounded deterministic and strict model-assisted query decomposition;
- provenance-preserving multi-hop retrieval with a hard global estimated-cost ceiling;
- answer, document, support-path, hop, lineage and abstention evaluation metrics;
- strict HotpotQA, 2WikiMultiHopQA and MuSiQue local adapters;
- scholarly/web/page/handbook and scientific-analysis tools;
- request-scoped agent, FastAPI, browser, CLI and container surfaces;
- release-lock generation and exact-head workflow configuration.

## Implemented critical controls

| Area | Current contract |
|---|---|
| Tenant isolation | API-key mapping or server-owned single-user identity controls vector, sparse, generation, registry, document and visual operations. Caller owner headers cannot choose another tenant. |
| Request admission | Request bodies, work queues, timeouts, identifiers, models, evidence, citations, metadata and responses are bounded. |
| Upload/retention | Random owner-scoped names, exact byte limits, anchored/no-follow operations, stable roots, private modes, fsync and symlink/reparse refusal. |
| Durable ingestion | SQLite state machine, atomic claims, bounded attempts, durable backoff, centralized scheduling, startup replay, immutable parser snapshots and compensation. |
| Parsing/OCR | PDF/DOCX/text complexity ceilings, bounded OCR, partial-page provenance and control-character refusal. |
| Privacy | Native/OCR text, metadata, titles, filenames, summaries, jobs, paths, credentials, contacts and scientific outputs receive best-effort masking. |
| Retrieval provenance | Uploaded evidence requires owner/document/chunk metadata; authoritative corpus modes validate current generations, content hashes and profiles. |
| Citation authority | Citations are selected from actual tool evidence; credential-bearing or ambiguous URLs and invalid page provenance are rejected. |
| Network/provider boundary | Public DNS and connected-peer validation, redirect revalidation, proxy suppression, credential stripping, deadlines, MIME/header/body limits and strict JSON. |
| Classic persistence | Manifest-last immutable generations, hashes, counts, locks, strict JSON and identity-bound storage roots. |
| Multi-store generations | Vector and sparse snapshots, append-only generation history, compensation, drift scans and bounded reconciliation planning. |
| Adaptive RAG | Query routing, evidence signals, bounded corrective attempts, privacy-safe traces, failure containment and abstention. |
| Route experiments | Strict offline route fixtures and query/evidence-free reports compare router/oracle success, accuracy, utility, cost/latency proxies and regret. |
| Multi-hop RAG | Closed-schema plans, validated DAGs, hard global estimated-cost allocation, bounded parallel/serial execution, constraint propagation and immutable per-hop citation lineage. |
| Evaluation | Answer EM/token-F1, document/support P/R/F1, sentence/paragraph/page support, path completeness, hop coverage, lineage validity, abstention and bounded macro aggregation. |
| Dataset adapters | Local-only bounded HotpotQA, 2WikiMultiHopQA and MuSiQue normalization with exact-byte fingerprints and malformed/path-hostile input refusal. |
| Frontend/deployment | Safe DOM rendering, session-only credentials, local assets, readiness probes, non-root read-only container and loopback default. |
| Reproducibility | Immutable requirements snapshots, hash locks, authority stripping, atomic publication and immutable workflow pins. |

## Capability status

### Completed foundations

- Wave 1: hybrid retrieval and evaluation foundation.
- Wave 2A: embedding governance and persistent sparse index.
- Wave 2B: authoritative vector+sparse+generation foundation.
- Wave 2C: generation-validated corpus hybrid retrieval foundation.
- Wave 2D: migration inventory, planning and durable journal/control plane.
- Wave 3: adaptive/corrective retrieval, privacy-safe tracing, offline route experiments and calibration foundation.
- Wave 4: bounded decomposition, global estimated-cost allocation, uploaded-document multi-hop execution, structural evaluation and three benchmark-format adapters.

Detailed checked/open items are in:

- `docs/CAPABILITY_IMPLEMENTATION_STATUS.md`
- `docs/TODO.md`
- `docs/EXHAUSTIVE_MISSION_AUDIT_2026-08-01.md`
- `docs/MULTIHOP_RETRIEVAL.md`

## Current focused verification evidence

Historical remediation runs established substantial test and lock-matrix evidence, but they do not certify the current head.

For the decomposition/multi-hop continuation:

- 35 focused tests passed locally;
- Python compilation passed for the seven new modules and focused tests;
- Ruff was unavailable in the constrained local environment.

The concurrent adaptive trace and route-experiment additions include committed focused tests, but they were not executed in the partial local patch workspace because that workspace does not contain the full repository dependency graph.

## Exact-head release status

Release readiness is **not claimed**.

The configured exact-head matrix includes:

- Linux Python 3.10, 3.11 and 3.12 dependency, compile, lint, pytest and coverage jobs;
- Windows Python 3.10 and 3.12 storage regressions;
- Docker Compose validation and image build;
- Linux/Windows/macOS Python 3.10–3.12 lock generation and hash-required dry installation.

No complete green result is currently observable through the available connector for the latest exact `main` SHA. The constrained execution environment cannot clone or download from GitHub because DNS resolution fails. Every release claim must wait for one unchanged `main` head to pass the complete matrix.

## Highest-priority remaining work

1. Register and integration-test adaptive and multi-hop tools through the full agent/API/browser response path.
2. Coordinate the retained-document registry as a fourth transaction participant or durable outbox consumer.
3. Run startup reconciliation and implement resumable repair/adoption workflows.
4. Execute shadow profile migrations, validate artifacts and implement atomic cutover/rollback.
5. Benchmark and calibrate dense/sparse fusion, adaptive policy, decomposition quality and abstention thresholds.
6. Run representative connected route experiments with repeated seeds and promotion gates.
7. Add filters, independent-corpus fusion, source caps and reranker cascades.
8. Add heterogeneous uploaded/web/scholarly multi-hop routing with measured cross-backend latency/token/cost budgets.
9. Add custom scientific benchmark adapters and semantic support/entailment evaluation.
10. Build evidence graph, multimodal scientific ingestion and structured evidence intelligence.
11. Add repeated-run statistical/resource observability and promotion gates.
12. Complete the exact-head release matrix and final regression audit.

## Residual architectural and scientific limitations

These are disclosed rather than falsely marked complete:

- provider code already running in a Python thread cannot be forcibly terminated safely;
- application SSRF controls still require deployment DNS/egress policy;
- filesystem anchoring is not host isolation or encryption at rest;
- parser checks are not malware scanning or sandboxing;
- retained sources are not application-encrypted;
- process-local admission, scheduling, rate limiting, SQLite state and compensation are not distributed exactly-once infrastructure;
- OCR, reading order, tables, formulas, scanned captions and multi-panel interpretation remain heuristic;
- regex masking is not certified de-identification;
- retrieval rank, generation alignment and citation presence do not prove semantic support;
- a valid decomposition plan or structural score does not prove optimal decomposition;
- the estimated-cost ceiling is a deterministic workload proxy, not measured token, latency or monetary cost;
- offline route fixtures prove harness behavior, not calibrated production routing;
- cross-hop grouping does not prove a shared claim;
- the heuristic answer-support score does not prove entailment;
- dataset-format validation does not establish quality, representativeness or license suitability;
- scientific outputs require source inspection, expert review and replication.
