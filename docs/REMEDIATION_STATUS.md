# Exhaustive Remediation Status

This document records the remediation performed after the repository-wide static audit of 2026-07-27 and the subsequent regression audit of the remediation branch. It does not claim that software can be proven free of every defect. It distinguishes implemented controls, executable verification, and residual limitations.

## Scope

The remediation branch audits and changes every product surface identified in the repository inventory:

- classic crawler, sparse index, PageRank, storage, and CLIs;
- uploaded-document parsing, optional OCR, masking, summaries, source retention, and vector retrieval;
- durable ingestion queue, bounded workers, startup recovery, and document registry;
- agent orchestration, provenance, scientific-analysis tools, and web tools;
- FastAPI identity, uploads, jobs, models, throttling, and direct tool routes;
- browser rendering, authentication, uploads, document management, and responsive tools;
- tests, CI, dependencies, containers, runtime artifacts, licensing, and documentation.

## Critical findings

| Finding | Status | Implemented remediation |
|---|---|---|
| Default-user cross-tenant vector retrieval | Resolved in code | Every vector query, list, delete, comparison, limitation lookup, and figure operation requires an owner filter. The default owner no longer disables filtering. |
| Spoofable `X-Owner-ID` | Resolved in code | Tenant identity is derived from `API_KEY_OWNERS_JSON`, legacy server-derived key IDs, or a server-controlled single-user owner. Client owner headers are ignored. |
| Shared mutable global agent | Resolved in code | A new immutable owner/model agent is created for each request; blocking runs are moved to a worker thread. |
| Arbitrary URL SSRF | Resolved in code | HTTP(S)-only validation, DNS resolution, private/local/link-local/reserved blocking, redirect revalidation, streaming limits, timeouts, and proxy isolation are implemented. |
| Persistent/reflected browser XSS | Resolved in code | External Markdown runtime removed; no dynamic `innerHTML`; constrained DOM renderer, safe links, CSP, and session-only storage. |
| Unsafe upload names and unbounded bodies | Resolved in code | Random owner-scoped storage names, suffix/content-signature checks, streaming byte limits, and collision-safe creation. |
| Authentication incompatible with frontend | Resolved in code | Browser sends `X-API-Key`, receives model/auth configuration from `/config`, and never sends owner identity. |
| False redaction/privacy guarantee | Corrected and substantially mitigated | Full text, OCR output, every section, titles, filenames, metadata, summaries, and job-facing strings use masking. Local paths are excluded from serialization, vectors, citations, and public job status. Documentation states masking is best effort. |
| Source path leaked into vector metadata | Resolved in continuation | Filesystem paths moved to a private owner-scoped SQLite document registry. Scientific visual tools resolve owner/document sources through that registry. |
| Default ingestion broke visual tools | Resolved in continuation | Source retention is explicit through `RETAIN_SOURCE_FILES`; retained sources are registered, exposed only as a Boolean capability, and deleted with the document. |

## High-severity findings

| Area | Status | Notes |
|---|---|---|
| Stale tests and absent CI | Remediated in code/config | Tests for removed internals were removed/replaced. CI is configured to compile, perform fatal lint checks, run coverage tests on Python 3.10–3.12, and build the container. Repository Actions still must execute for the exact head. |
| Model-authored citations | Resolved in code | Models return answer prose only. A server evidence registry chooses and relabels actual tool citations. |
| Prompt injection from retrieved text | Mitigated | Retrieved and OCR text are explicitly untrusted; tenant scope and provenance are enforced outside the model. Model-prose injection risk cannot be eliminated completely. |
| Provider/model mismatch | Resolved in code | Server model allowlist and request-scoped provider configuration; frontend options come from the server. |
| CLI/API ingestion divergence | Resolved | One document service handles summaries and vector indexing for both entrypoints. |
| Semantic sections discarded by RAG | Resolved | Redacted semantic sections and page data are passed directly into deterministic vector chunks. |
| Duplicate/non-idempotent indexing | Resolved | Stable owner-content IDs and deterministic upserts; new vectors are written before stale IDs are pruned. |
| Debate judge lacks original evidence | Resolved | Original evidence is included in advocate, skeptic, and judge prompts; generated arguments are identified as analysis. |
| Wrong figure image/regex/MIME | Substantially remediated | Exact caption text is searched and a caption-adjacent region is rendered as PNG. Missing labels fail closed. Complex multi-panel and scanned-caption localization remains heuristic. |
| Empty-evidence comparisons | Resolved | Comparisons and matrices stop and report evidence gaps whenever a required document returns no evidence. |
| Unbounded matrix cost | Resolved | Document and metric caps plus one bounded synthesis call replace per-cell model calls. |
| Web domain-filter bypass and missing timeout | Resolved | Parsed host boundary checks, request timeout, HTTP status validation, and structured provider errors. |
| Crawler redirect escape and oversized downloads | Resolved | Safe bounded downloader and final-host revalidation. |
| Non-atomic classic-index persistence | Resolved | fsync plus atomic replace, locks, schema versions, and corrupt-file quarantine. |
| Tracked vector database and generated artifacts | Resolved in branch | Runtime Chroma DB, fake root research fixture, destructive frontend generator, stale self-audits, and mislabeled image were removed. |
| Sensitive raw-query logging | Resolved | Query SHA-256 and length replace raw text; logging is locked and failure-isolated. |
| Missing request budgets | Substantially remediated | Query, upload, download, redirect, turn, tool-call, model, execution-time, OCR, worker, retry, and rate limits are implemented. Running Python threads cannot be forcibly killed safely. |
| Job privacy and restart loss | Resolved for single-host deployment | Private source paths remain in SQLite only; public status is owner-scoped. Queued/interrupted jobs are startup-reconciled and atomically claimed with a retry ceiling. |
| Multi-worker duplicate job replay | Resolved in continuation | SQLite atomic claim changes only one queued row to processing and increments its attempt count. Other workers cannot process the same claim. |
| Image-only PDF rejection | Substantially remediated | Optional bounded Tesseract OCR is available for low-text pages. Disabled, missing, exhausted, and empty OCR paths return explicit diagnostics. |
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
- non-root, read-only container with dropped Linux capabilities, Tesseract, and health checks;
- separated runtime/development dependencies, unified project tooling, environment template, and an actual MIT license;
- honest replacement documentation instead of arithmetic completion scorecards.

## New justified functionality

Functionality added because it directly closes audited gaps:

1. Credential-to-owner identity mapping.
2. Owner-scoped document deletion.
3. SQLite ingestion queue with crash recovery.
4. Atomic worker claims and retry ceiling.
5. Private owner-scoped document/source registry.
6. Configurable source retention and complete deletion lifecycle.
7. Optional bounded OCR for scanned and mixed PDFs.
8. Per-principal rate limiting.
9. HTTP security headers and request IDs.
10. Safe public URL downloader shared by crawler/page tools.
11. Stable owner-content document identities.
12. Shared API/CLI document service.
13. Server-side evidence registry.
14. Structured evidence location fields: document, chunk, page, quote, source ID.
15. Config endpoint for secure browser/provider integration.
16. Clean-clone CI and container build configuration.
17. Security, architecture, and remediation documentation.

No unrelated feature was added merely to expand the feature count.

## Residual limitations

These are intentionally disclosed rather than falsely marked complete:

- OCR quality depends on Tesseract, scan resolution, language packs, orientation, and layout. OCR output requires review.
- Scanned figure-caption localization does not yet use OCR coordinates; exact selectable caption text is still required for visual cropping.
- PDF reading order, tables, formulas, headings, and multi-panel figure localization remain heuristic.
- Regex masking is not certified de-identification.
- Retained source files are not application-encrypted; deployment storage must provide encryption at rest where required.
- File validation is not malware scanning or document sandboxing.
- A process-local rate limiter is insufficient for multiple replicas.
- SQLite plus the bounded executor provides single-host recovery, not distributed exactly-once execution.
- Python threads executing third-party calls cannot be safely force-terminated; network/client deadlines are the primary control.
- Application-level SSRF controls should be combined with egress firewall rules.
- Citation provenance is structural; semantic support still requires evidence inspection or a separately validated entailment system.
- Scientific-analysis outputs remain model analyses and do not substitute for expert review or replication.
- Release deployments should generate platform-specific dependency lock files with hashes.

## Verification status

### Statically verified in the remediation work

- Every current changed file and public contract was re-read through the GitHub connector.
- The continuation pass re-audited source retention, ingestion recovery, scientific-tool file access, OCR, and their tests.
- The branch diff was reconciled against the complete repository inventory.
- Deterministic stale tests and obsolete runtime artifacts were removed.
- Contract tests cover security, privacy, identity, vector scope, atomic job claims, source-registry isolation, OCR, provenance, scientific fail-closed behavior, storage, frontend rendering, operations, and deployment configuration.

### Executable verification

The branch defines GitHub Actions checks for:

- Python 3.10, 3.11, and 3.12;
- `compileall`;
- fatal Ruff syntax/name checks;
- pytest with branch coverage and an honest initial 50% floor;
- Docker image build including Tesseract.

The available work environment could not clone GitHub or execute the branch locally, and GitHub reported no workflow runs for the previous PR heads. The remediation is therefore still draft and not considered merge-ready until checks run against the exact current head. The coverage floor should rise only from measured results, not from a fabricated target.
