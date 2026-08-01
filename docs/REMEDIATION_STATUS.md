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
- confidence calibration and risk-coverage analysis;
- bounded query decomposition and provenance-preserving multi-hop retrieval;
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
| Adaptive RAG | Query routing, evidence signals, bounded corrective attempts, trace records, failure containment and abstention. |
| Multi-hop RAG | Validated DAGs, topological batches, bounded parallel/serial execution, constraint propagation and immutable per-hop citation lineage. |
| Frontend/deployment | Safe DOM rendering, session-only credentials, local assets, readiness probes, non-root read-only container and loopback default. |
| Reproducibility | Immutable requirements snapshots, hash locks, authority stripping, atomic publication and immutable workflow pins. |

## Capability status

### Completed foundations

- Wave 1: hybrid retrieval and evaluation foundation.
- Wave 2A: embedding governance and persistent sparse index.
- Wave 2B: authoritative vector+sparse+generation foundation.
- Wave 2C: generation-validated corpus hybrid retrieval foundation.
- Wave 2D: migration inventory, planning and durable journal/control plane.
- Wave 3: adaptive/corrective retrieval and calibration foundation.
- Wave 4: bounded decomposition and uploaded-document multi-hop foundation.

Detailed checked/open items are in:

- `docs/CAPABILITY_IMPLEMENTATION_STATUS.md`
- `docs/TODO.md`
- `docs/EXHAUSTIVE_MISSION_AUDIT_2026-08-01.md`
- `docs/MULTIHOP_RETRIEVAL.md`

## Current focused verification evidence

Historical remediation runs established substantial test and lock-matrix evidence, but they do not certify the current head.

For the new decomposition/multi-hop continuation:

- 12 focused tests passed locally;
- Python compilation passed for the three new modules and focused tests;
- Ruff was unavailable in the constrained local environment.

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
5. Benchmark and calibrate dense/sparse fusion, adaptive policy and abstention thresholds.
6. Add filters, independent-corpus fusion, source caps and reranker cascades.
7. Add heterogeneous uploaded/web/scholarly multi-hop routing and benchmarks.
8. Build evidence graph, multimodal scientific ingestion and structured evidence intelligence.
9. Add repeated-run statistical/resource observability and promotion gates.
10. Complete the exact-head release matrix and final regression audit.

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
- a valid decomposition plan does not prove optimal decomposition;
- cross-hop grouping does not prove a shared claim;
- scientific outputs require source inspection, expert review and replication.
