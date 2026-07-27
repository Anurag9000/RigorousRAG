# Exhaustive Remediation Status

This document records the remediation performed after the repository-wide static audit of 2026-07-27. It does not claim that software can be proven free of every defect. It distinguishes implemented controls, executable verification, and residual limitations.

## Scope

The remediation branch audits and changes every product surface identified in the repository inventory:

- classic crawler, sparse index, PageRank, storage, and CLIs;
- uploaded-document parsing, masking, summaries, and vector retrieval;
- agent orchestration, provenance, scientific-analysis tools, and web tools;
- FastAPI identity, uploads, jobs, models, throttling, and direct tool routes;
- browser rendering, authentication, uploads, document management, and responsive tools;
- tests, CI, dependencies, containers, runtime artifacts, licensing, and documentation.

## Critical findings

| Finding | Status | Implemented remediation |
|---|---|---|
| Default-user cross-tenant vector retrieval | Resolved in code | Every vector query, list, delete, comparison, limitation lookup, and document path lookup requires an owner filter. The default owner no longer disables filtering. |
| Spoofable `X-Owner-ID` | Resolved in code | Tenant identity is derived from `API_KEY_OWNERS_JSON`, legacy server-derived key IDs, or a server-controlled single-user owner. Client owner headers are ignored. |
| Shared mutable global agent | Resolved in code | A new immutable owner/model agent is created for each request; blocking runs are moved to a worker thread. |
| Arbitrary URL SSRF | Resolved in code | HTTP(S)-only validation, DNS resolution, private/local/link-local/reserved blocking, redirect revalidation, streaming limits, timeouts, and proxy isolation are implemented. |
| Persistent/reflected browser XSS | Resolved in code | External Markdown runtime removed; no dynamic `innerHTML`; constrained DOM renderer, safe links, CSP, and session-only storage. |
| Unsafe upload names and unbounded bodies | Resolved in code | Random owner-scoped storage names, suffix and content-signature checks, streaming byte limits, collision-safe creation, and default post-index deletion. |
| Authentication incompatible with frontend | Resolved in code | Browser sends `X-API-Key`, receives model/auth configuration from `/config`, and never sends owner identity. |
| False redaction/privacy guarantee | Corrected and partially mitigated | Full text and every section use the same masking pass; local paths are excluded from serialization; originals are deleted by default. Documentation now states masking is best effort, not guaranteed anonymization. |

## High-severity findings

| Area | Status | Notes |
|---|---|---|
| Stale tests and absent CI | Remediated | Tests for removed internals were removed/replaced. CI compiles, performs fatal lint checks, runs coverage tests on Python 3.10–3.12, and builds the container. |
| Model-authored citations | Resolved in code | Models return answer prose only. A server evidence registry chooses and relabels actual tool citations. |
| Prompt injection from retrieved text | Mitigated | Retrieved text is explicitly untrusted; tenant scope and provenance are enforced outside the model. Model-prose injection risk cannot be eliminated completely. |
| Provider/model mismatch | Resolved in code | Server model allowlist and request-scoped provider configuration; frontend options come from the server. |
| CLI/API ingestion divergence | Resolved | One document service handles summaries and vector indexing for both entrypoints. |
| Semantic sections discarded by RAG | Resolved | Redacted semantic sections and page data are passed directly into deterministic vector chunks. |
| Duplicate/non-idempotent indexing | Resolved | Stable owner-content IDs and deterministic upserts; new vectors are written before stale IDs are pruned. |
| Debate judge lacks original evidence | Resolved | Original evidence is included in advocate, skeptic, and judge prompts; generated arguments are identified as analysis. |
| Wrong figure image/regex/MIME | Substantially remediated | Exact caption text is searched and a caption-adjacent region is rendered as PNG. Missing labels fail closed. Complex multi-panel localization remains heuristic. |
| Empty-evidence comparisons | Resolved | Comparisons and matrices stop and report evidence gaps whenever a required document returns no evidence. |
| Unbounded matrix cost | Resolved | Document and metric caps plus one bounded synthesis call replace per-cell model calls. |
| Web domain-filter bypass and missing timeout | Resolved | Parsed host boundary checks, request timeout, HTTP status validation, structured provider errors. |
| Crawler redirect escape and oversized downloads | Resolved | Safe bounded downloader and final-host revalidation. |
| Non-atomic classic-index persistence | Resolved | fsync plus atomic replace, locks, schema versions, and corrupt-file quarantine. |
| Tracked vector database and generated artifacts | Resolved in branch | Runtime Chroma DB, fake root research fixture, destructive frontend generator, stale self-audits, and mislabeled image were removed. |
| Sensitive raw-query logging | Resolved | Query SHA-256 and length replace raw text; logging is locked and failure-isolated. |
| Missing request budgets | Substantially remediated | Query, upload, download, redirect, turn, tool-call, model, execution-time, and rate limits are implemented. Running Python tool threads cannot be forcibly killed safely; distributed hard cancellation remains an infrastructure concern. |
| Job privacy and restart loss | Resolved for single service | Owner-scoped SQLite job store survives restart and expires old jobs. Durable distributed execution remains a worker-queue concern. |
| Document-specific UI was only prompt text | Resolved | Tool schema and RAG query support exact `doc_id`; document cards prefill a real supported filter. |
| Incorrect BibTeX | Resolved for supported fields | Escaped values, deterministic keys, venue fields, common entry types, and duplicate-key handling. |

## Medium and low findings

The branch additionally implements:

- clean index rebuilds without stale postings;
- Unicode, numeric, and scientific-identifier tokenization;
- PageRank validation, convergence, fetched-page graph filtering, and normalized authority prior;
- stronger URL canonicalization and tracking-parameter removal;
- fail-closed robots policy by default and a real contact URL;
- query-centered snippets and context alignment by URL;
- optional crawling rather than mandatory recrawling in CLIs;
- clear separation of vector distance and similarity score;
- relevant handbook passage retrieval;
- structured web/tool errors rather than error citations;
- nullable-safe source rendering and mobile tool access;
- no external frontend runtime dependencies;
- non-root, read-only container with dropped Linux capabilities and health checks;
- separated runtime/development dependencies, unified project tooling, environment template, and an actual MIT license;
- honest replacement documentation instead of arithmetic completion scorecards.

## New justified functionality

Functionality added because it directly closes audited gaps:

1. Credential-to-owner identity mapping.
2. Owner-scoped document deletion.
3. Crash-persistent ingestion jobs.
4. Per-principal rate limiting.
5. HTTP security headers and request IDs.
6. Safe public URL downloader shared by crawler/page tools.
7. Stable owner-content document identities.
8. Shared API/CLI document service.
9. Server-side evidence registry.
10. Structured evidence location fields: document, chunk, page, quote, source ID.
11. Config endpoint for secure browser/provider integration.
12. Clean-clone CI and container build.
13. Security and architecture documentation.

No unrelated feature was added merely to expand the feature count.

## Residual limitations

These are intentionally disclosed rather than falsely marked complete:

- Scanned/image-only PDFs require an external OCR pipeline.
- PDF reading order, tables, formulas, and multi-panel figure localization remain heuristic.
- Regex masking is not certified de-identification.
- File validation is not malware scanning or document sandboxing.
- A process-local rate limiter is insufficient for multiple replicas.
- In-process background execution should become a durable worker queue for high-scale deployments.
- Python threads executing third-party calls cannot be safely force-terminated; network/client deadlines are the primary control.
- Application-level SSRF controls should be combined with egress firewall rules.
- Citation provenance is structural; semantic support still requires evidence inspection or a separately validated entailment system.
- Scientific-analysis outputs remain model analyses and do not substitute for expert review or replication.
- Release deployments should generate platform-specific dependency lock files with hashes.

## Verification status

### Statically verified in the remediation work

- Every current changed file and symbol was re-read through the GitHub connector.
- The branch diff was reconciled against the complete repository inventory.
- Deterministic stale tests and obsolete runtime artifacts were removed.
- Contract tests were added for security, privacy, identity, vector scope, provenance, scientific fail-closed behavior, storage, frontend rendering, operations, and deployment configuration.

### Executable verification

The branch defines GitHub Actions checks for:

- Python 3.10, 3.11, and 3.12;
- `compileall`;
- fatal Ruff syntax/name checks;
- pytest with branch coverage and an honest initial 50% floor;
- Docker image build.

The remediation is not considered merge-ready until those checks run against the exact pull-request head. The coverage floor should rise only from measured results, not from a fabricated target.
