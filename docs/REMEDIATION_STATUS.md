# Exhaustive Remediation Status

This document records the implementation performed after the repository-wide static audit of 2026-07-27 and subsequent regression audits of the remediation branch. It does not claim that the software is proven defect-free. It separates controls implemented in source, tests written for those controls, executable verification, and residual limitations.

## Scope

The remediation branch audits and changes every product surface identified in the repository inventory:

- classic crawler, sparse index, PageRank, persistence, and CLIs;
- PDF/DOCX/text parsing, optional OCR, masking, source identity, retention, and vector retrieval;
- durable ingestion queue, centralized delayed scheduling, bounded admission, startup recovery, and document registry;
- agent orchestration, query/tool concurrency, provenance, scientific-analysis tools, page/web search, and BibTeX;
- FastAPI identity, body limits, uploads, jobs, models, throttling, request deadlines, and direct tool routes;
- browser rendering, authentication, upload status, document management, source capability, and responsive tools;
- tests, CI, dependencies, container readiness, runtime artifacts, telemetry, licensing, and documentation.

## Critical findings

| Finding | Status | Implemented remediation |
|---|---|---|
| Default-user cross-tenant retrieval | Resolved in code | Every vector query, list, replace, delete, comparison, limitation lookup, and figure operation requires the server-owned tenant ID. |
| Spoofable `X-Owner-ID` | Resolved in code | Tenant identity comes from configured API-key mappings or the server-controlled single-user identity. Client owner headers are ignored. |
| Shared mutable global agent | Resolved in code | Every query gets an immutable owner/model agent and enters a dedicated bounded request executor. |
| Arbitrary URL SSRF and DNS rebinding | Substantially resolved in code | The shared downloader validates schemes, DNS results, and the actual connected peer; blocks private/local/reserved networks; disables proxies; revalidates redirects; strips cross-origin credentials; prevents POST-body replay; and enforces byte/end-to-end time budgets. Network egress policy remains required defense in depth. |
| Persistent/reflected browser XSS | Resolved in code | External Markdown runtime removed; no untrusted `innerHTML`; constrained DOM renderer, safe links, CSP, and session-only storage. |
| Unsafe uploads and parser resource exhaustion | Substantially resolved in code | Pre-parser body ceiling, random owner-scoped names, byte ceilings, fsync, signatures, symlink rejection, DOCX archive expansion/path checks, PDF/OCR/text complexity budgets, mutation detection, and explicit errors. This is not malware scanning or sandboxing. |
| Authentication incompatible with frontend | Resolved in code | Browser sends `X-API-Key`, reads `/config`, and never supplies tenant identity. |
| False redaction/privacy guarantee | Corrected and mitigated | Full text, OCR output, sections, titles, filenames, metadata, summaries, and job-facing strings are masked. Diagnostic paths, URI credentials, and common secret parameters are also removed. Documentation states masking is best effort. |
| Source paths leaked into vectors | Resolved | Paths exist only in the private owner-scoped registry/queue and are excluded from Chroma, citations, manifests, and public job payloads. |
| Source lifecycle broke visual tools | Resolved with bounded capability | Retention is explicit; actual file availability is validated dynamically; visual lookup verifies current bytes against the owner/content document ID and enforces PDF page/render budgets; deletion removes vectors, registry, and source. |

## High-severity findings

| Area | Status | Notes |
|---|---|---|
| Stale tests and absent CI | Remediated in source/config | Tests were replaced with contract/regression tests. CI is configured for Python 3.10–3.12 compile, fatal Ruff checks, pytest/coverage, and Docker build. Exact-head execution is still absent. |
| Model-authored citations | Resolved | The model returns prose. A bounded server evidence registry selects, deduplicates, relabels, and serializes actual tool citations. |
| Prompt injection through evidence | Mitigated | Retrieved and OCR text are explicitly untrusted; tenant/provenance controls live outside the model. Model-prose risk remains. |
| Tool argument and context abuse | Resolved in continuation | Runtime JSON-schema validation, bounded argument/result/evidence/answer sizes, generic exception messages, and process-wide bounded running-plus-queued tool admission. |
| False tool timeout | Resolved in continuation | Single and parallel tool calls return after the configured deadline instead of waiting for executor shutdown. Running third-party Python threads still cannot be killed. |
| Unbounded timed-out tool threads/queue | Resolved in continuation | Per-request executors were replaced with one process-wide executor; `MAX_PENDING_TOOL_TASKS` prevents an unbounded submission queue and timed-out running work retains capacity until completion. |
| Unbounded HTTP query work | Resolved in continuation | `/query` uses a dedicated `BoundedExecutor`, explicit running-plus-pending limits, generic overload responses, and a whole-request timeout. |
| Provider/model mismatch | Resolved | Server model allowlist and request-scoped provider configuration; frontend options come from `/config`. |
| CLI/API ingestion divergence | Resolved | Shared parsing/indexing services and source registry; CLI can explicitly retain bounded random private copies. |
| Redaction-induced document-ID collisions | Resolved in continuation | Stable IDs use owner plus source-file SHA-256, while only the redacted-text hash is exported. Distinct files that mask to the same text no longer overwrite each other. |
| Source changes after parsing | Resolved in continuation | Identity is recomputed immediately before summary/vector writes; retained visual bytes are re-hashed again before figure access. |
| Semantic sections discarded | Resolved | Redacted sections/page provenance are passed into deterministic parent/child chunks. |
| Partial vector generations after failed batches | Substantially resolved in continuation | Previous chunks are captured; new-only chunks are removed and old chunks restored on failed upsert/stale-delete sequences. Incomplete compensation is explicit. |
| One large document hides the library | Resolved in continuation | Document listing paginates chunks until the requested distinct-document count or a configured scan ceiling. |
| Retrieval outage represented as no evidence | Resolved in continuation | Total Chroma failure raises unavailable rather than returning an empty result. Fallback warnings distinguish outage from no match. |
| Debate judge lacks original evidence | Resolved | Original evidence is included for advocate, skeptic, and judge. |
| Incorrect/unbounded figure selection | Substantially remediated | Exact selectable caption text is located; only owner/content-matching retained PDFs under configured page/render limits reach the caption-adjacent renderer. Scanned captions, exact encoded-image byte limiting, and complex multi-panel localization remain limitations. |
| Empty-evidence comparisons/matrices | Resolved | Required-document evidence gaps stop synthesis. Matrix work is capped and consolidated into one bounded call. |
| Web domain bypass/unbounded provider call | Resolved | Parsed hostname boundaries; Serper uses the shared peer-validated bounded downloader; provider errors are generic. |
| Crawler redirect escape/oversized download | Resolved | Shared downloader, final-host trust check, response-byte and total-time ceilings. |
| Non-atomic classic persistence | Resolved | Deterministic serialization, fsync, atomic replacement, generation manifest hashes/counts, schema versions, and mixed/corrupt-generation quarantine. |
| Sensitive query/owner logging and unbounded logs | Resolved in continuation | Query/owner SHA-256 replaces plaintext; recursive event bounds and configured JSONL rotation/backups. |
| Job restart loss and duplicate replay | Resolved for one host | SQLite queue, atomic due claims, attempt ceilings, finalizing reconciliation, source ownership, and retry deadlines. |
| Immediate retries, timer explosion, worker starvation | Resolved in continuation | SQLite stores bounded exponential deadlines; one lazy heap/condition scheduler manages all delayed jobs; `INGEST_MAX_PENDING` prevents unbounded executor submission. |
| Image-only PDF rejection | Substantially remediated | Optional bounded Tesseract OCR with actual-attempt accounting and page-local failure provenance. |
| Incorrect BibTeX | Resolved for supported fields | Escaped values, deterministic unique keys, venue fields, and common entry types. |
| Liveness mislabeled as readiness | Resolved in continuation | Container probe verifies HTTP, both SQLite stores, and create/fsync/delete access to upload/vector volumes without initializing the embedding model. |
| Oversized body before parser controls | Resolved in continuation | Pure ASGI middleware rejects declared or streamed oversize before multipart/JSON parsing and explicitly completes partial responses. |

## Medium and low findings

The branch additionally implements:

- Unicode, numeric, and scientific-identifier tokenization;
- clean sparse-index rebuilds without stale postings;
- PageRank validation, convergence, fetched-page graph filtering, and normalized authority blending;
- stronger URL canonicalization and tracking-parameter removal;
- fail-closed robots behavior and a real crawler contact URL;
- query-centered snippets and URL-aligned contexts;
- persisted-index search without mandatory recrawling;
- relevant handbook passage retrieval;
- structured failures rather than error citations;
- nullable-safe source rendering and mobile tool access;
- lifecycle-aware upload status and retained-PDF eligibility/verification fields;
- request-ID allowlisting and model/job/document/response identifier limits;
- non-root read-only container, dropped capabilities, Tesseract, named state volumes, and dependency-aware readiness;
- separated runtime/development dependencies, environment template, package metadata, and MIT license;
- removal of committed runtime databases, fabricated fixtures, destructive generators, mislabeled assets, and stale self-certification documents.

## New justified functionality

Functionality added because it directly closes audited gaps:

1. Credential-to-owner identity mapping.
2. Owner-scoped document deletion.
3. Private owner-scoped document/source registry.
4. SQLite ingestion queue and explicit state machine.
5. Atomic due claims, attempt ceilings, durable exponential deadlines, and startup reconciliation.
6. One lazily started keyed deadline scheduler for all delayed jobs.
7. Bounded ingestion running-plus-queued admission.
8. Complete retained-source replacement/deletion/orphan lifecycle.
9. Optional bounded OCR for scanned and mixed PDFs.
10. DOCX/PDF/text complexity guards and source-mutation detection.
11. Stable owner/source document identity and retained-byte visual verification.
12. Shared API/CLI parsing and indexing services.
13. Compensating vector replacement and paginated library listing.
14. Credential-derived per-principal throttling.
15. Pre-parser request-body enforcement.
16. Dedicated bounded HTTP query executor and whole-request deadline.
17. Server-side evidence registry and bounded response models.
18. Runtime tool-schema validation and bounded tool execution/admission.
19. Connected-peer SSRF validation and safe cross-origin redirect semantics.
20. Dependency-aware container readiness.
21. Bounded pseudonymous rotating telemetry.
22. Security, architecture, remediation, and deployment documentation.
23. Clean-clone CI and Docker build configuration.

No unrelated feature was added merely to increase the feature count.

## Residual limitations

These are disclosed rather than falsely marked complete:

- OCR quality depends on Tesseract, scan resolution, language packs, orientation, and layout.
- Scanned figure-caption localization does not use OCR coordinates; exact selectable caption text is required.
- PDF reading order, tables, formulas, headings, and multi-panel figure localization remain heuristic.
- Visual page/render geometry is bounded, but the renderer does not yet apply a separate exact post-PNG/base64 byte ceiling.
- Direct scientific HTTP routes are rate-limited but do not yet share `/query`'s dedicated whole-route executor/deadline.
- Regex masking is not certified de-identification.
- File/archive validation is not malware scanning or parser sandboxing.
- Retained source files are not application-encrypted.
- Process-local limiters, schedulers, executors, and SQLite stores do not support multiple replicas.
- SQLite plus compensating vector writes is not a formal cross-store transaction or distributed exactly-once system.
- Python threads and operating-system DNS resolution cannot be force-terminated safely; bounded admission and network/provider deadlines limit impact but do not provide hard cancellation.
- Application SSRF controls should be combined with DNS/egress firewall rules.
- The readiness probe tests stores/volumes, not embedding-model download or representative semantic retrieval.
- Citation provenance is structural; semantic support still requires source inspection or a separately validated entailment system.
- Scientific-analysis outputs require expert review and replication.
- Release deployments should generate platform-specific dependency locks with hashes.

## Verification status

### Statically inspected and contract-tested in source

- The remediation and continuation passes re-read changed public contracts and exact commit diffs through the GitHub connector.
- Regression tests now target tenant isolation, request-body ceilings, query overload/deadline behavior, tool admission/timeouts, peer-validated networking, redirect secrets, total download deadlines, parser/archive complexity, source identity, retained-source mutation, visual page/render limits, vector rollback, paginated listing, retrieval outages, queue migration/backoff, centralized delayed scheduling, ingestion admission, source reconciliation, bounded models, telemetry rotation, OCR, scientific fail-closed behavior, frontend safety, and deployment configuration.
- Runtime artifacts remain ignored and committed stale artifacts were removed.

### Executable verification still required

The branch defines checks for:

- Python 3.10, 3.11, and 3.12;
- `compileall`;
- fatal Ruff syntax/name checks;
- pytest with branch coverage and an initial 50% floor;
- Docker image build including Tesseract.

The available remediation environment could not clone/download and execute the branch, and GitHub Actions had not produced a workflow run for earlier exact heads through the available connector. Therefore this PR remains draft and is not merge-ready until checks run against the exact current head and every failure is corrected. Coverage targets must rise from measured results, not fabricated claims.
