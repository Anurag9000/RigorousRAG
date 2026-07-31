# Exhaustive Remediation Status

This document records implementation performed after the repository-wide static audit of 2026-07-27 and seven subsequent regression audits of the remediation branch. It does not claim that the software is proven defect-free. It separates controls implemented in source, tests written for those controls, observed executable evidence, current release gates, and residual limitations.

## Scope

The remediation branch audits and changes every product surface identified in the repository inventory:

- classic crawler, sparse index, PageRank, generation persistence, scholarly adapters, and CLIs;
- PDF/DOCX/text parsing, optional OCR, masking, source identity, retention, and vector retrieval;
- durable ingestion queue, centralized delayed scheduling, bounded admission, startup recovery, document registry, and operator repair;
- agent orchestration, query/tool concurrency, provenance, scientific-analysis tools, page/web/scholarly search, legacy summarization, and BibTeX;
- FastAPI identity, body limits, uploads, jobs, models, throttling, request deadlines, direct tool routes, and portable static assets;
- browser rendering, authentication, upload status, document management, source capability, and responsive tools;
- tests, cross-platform CI, release-lock generation, dependencies, container readiness, runtime artifacts, telemetry, licensing, and documentation.

## Critical findings

| Finding | Status | Implemented remediation |
|---|---|---|
| Default-user cross-tenant retrieval | Resolved in code | Every vector query, list, replace, delete, comparison, limitation lookup, and figure operation requires the server-owned tenant ID. |
| Spoofable `X-Owner-ID` | Resolved in code | Tenant identity comes from configured API-key mappings or the server-controlled single-user identity. Client owner headers are ignored. |
| Shared mutable global agent | Resolved in code | Every query gets an immutable owner/model agent and enters a dedicated bounded research executor. |
| Arbitrary URL SSRF and DNS rebinding | Substantially resolved in code | The shared downloader validates schemes, DNS results, and the actual connected peer; blocks private/local/reserved networks; disables proxies; revalidates redirects; strips cross-origin credentials; prevents POST-body replay; and enforces byte/end-to-end time budgets. Network egress policy remains required defense in depth. |
| Persistent/reflected browser XSS | Resolved in code | External Markdown runtime removed; no untrusted `innerHTML`; constrained DOM renderer, safe links, CSP, and session-only storage. |
| Unsafe uploads and parser resource exhaustion | Substantially resolved in code | Pre-parser body ceiling, random owner-scoped names, byte ceilings, fsync, signatures, symlink rejection, DOCX archive expansion/path checks, PDF/OCR/text complexity budgets, mutation detection, and explicit errors. This is not malware scanning or sandboxing. |
| Authentication incompatible with frontend | Resolved in code | Browser sends `X-API-Key`, reads `/config`, and never supplies tenant identity. |
| False redaction/privacy guarantee | Corrected and mitigated | Full text, OCR output, sections, titles, filenames, metadata, summaries, job-facing strings, CLI output, provider results, and scientific result mappings are masked. Sentence-final email punctuation is handled correctly. Documentation states masking is best effort. |
| Source paths leaked into vectors | Resolved | Paths exist only in the private owner-scoped registry/queue and are excluded from Chroma, citations, manifests, and public job payloads. |
| Source lifecycle broke visual tools | Resolved with bounded capability | Retention is explicit; actual file availability is validated dynamically; visual lookup verifies current bytes against the owner/content document ID and enforces PDF page, true preallocation geometry, actual rendered-pixel, and exact encoded-payload limits; deletion removes vectors, registry, and source. |

## High-severity findings

| Area | Status | Notes |
|---|---|---|
| Stale tests and absent CI | Remediated and partially executed | Tests were replaced with contract/regression tests. One consolidated 16-job workflow now checks Linux Python 3.10–3.12, focused Windows storage, Compose/container, and nine platform lock combinations. The first full Linux run found two privacy failures after 711 passes and 76.25% coverage; the shared root cause is corrected, but the final head still requires a complete rerun. |
| Model-authored citations | Resolved | The model returns prose. A bounded server evidence registry selects, deduplicates, relabels, and serializes actual tool citations. |
| Prompt injection through evidence | Mitigated | Retrieved and OCR text is explicitly untrusted; tenant/provenance controls live outside the model. Model-prose risk remains. |
| Tool argument and context abuse | Resolved in continuation | Runtime JSON-schema validation, bounded argument/result/evidence/answer sizes, generic exception messages, and process-wide bounded running-plus-queued tool admission. |
| False tool timeout | Resolved in continuation | Tool calls return after the configured deadline instead of waiting for executor shutdown. Running third-party Python threads still cannot be killed. |
| Unbounded timed-out tool threads/queue | Resolved in continuation | Per-request executors were replaced with one process-wide executor; `MAX_PENDING_TOOL_TASKS` prevents an unbounded submission queue and timed-out running work retains capacity until completion. |
| Unbounded HTTP research work | Resolved in continuation | `/query`, direct visual entailment, direct protocol extraction, document listing, and document deletion share bounded research execution, explicit running-plus-pending limits, generic overload responses, and whole-operation timeouts. |
| Provider/model mismatch and malformed provider objects | Resolved | Server model allowlist, strict request-scoped provider configuration, bounded strict JSON, safe provider field extraction, and frontend options from `/config`. |
| CLI/API ingestion divergence | Resolved | Shared parsing/indexing services and source registry; CLI can explicitly retain bounded random private copies. |
| Batch CLI partial finalization | Resolved in continuation | The CLI refuses symlinked inputs, captures the prior owner/document vector generation before replacement, restores it when registry finalization fails, cleans new retained copies, and publishes manifests atomically. |
| Redaction-induced document-ID collisions | Resolved in continuation | Stable IDs use owner plus source-file SHA-256, while only the redacted-text hash is exported. Distinct files that mask to the same text no longer overwrite each other. |
| Sentence-final emails survived masking | Resolved in pass seven | The shared email regex now ends domain labels on alphanumeric boundaries, preserving ordinary punctuation while masking OCR text, semantic sections, metadata, telemetry, and scientific output consistently. |
| Source changes after parsing | Resolved in continuation | Identity is recomputed immediately before summary/vector writes; retained visual bytes are re-hashed again before figure access. |
| Semantic sections discarded | Resolved | Redacted sections/page provenance are passed into deterministic parent/child chunks. |
| Partial vector generations after failed batches | Substantially resolved in continuation | Previous chunks are captured; new-only chunks are removed and old chunks restored on failed upsert/stale-delete sequences. Incomplete compensation is explicit. |
| One large document hides the library | Resolved in continuation | Document listing paginates chunks until the requested distinct-document count or a configured scan ceiling. |
| Retrieval outage represented as no evidence | Resolved in continuation | Total Chroma failure raises unavailable rather than returning an empty result. Fallback warnings distinguish outage from no match. |
| Legacy summarizer loses generator evidence | Resolved in continuation | Hits and contexts are aligned once into a bounded snapshot; OpenAI/Ollama failure reuses that snapshot instead of re-reading exhausted iterators. Provider response fields and direct output are strictly bounded and privacy-masked. |
| Debate judge lacks original evidence | Resolved | Original evidence is included for advocate, skeptic, and judge. |
| Incorrect/unbounded figure selection | Substantially remediated | Exact selectable caption text is located; only owner/content-matching retained PDFs under page/geometry budgets reach rendering; actual rendered pixels and exact base64 bytes are capped. Scanned captions and complex multi-panel localization remain limitations. |
| Scientific output leaks diagnostic metadata | Resolved in continuation | Every scientific result passes through recursive value/key sanitization before JSON serialization; key collisions are preserved with deterministic bounded suffixes. |
| Empty-evidence comparisons/matrices | Resolved | Required-document evidence gaps stop synthesis. Matrix work is capped and consolidated into one bounded call. |
| Web/scholarly provider bypass or unbounded result parsing | Resolved | Parsed hostname boundaries; shared peer-validated downloader; strict provider JSON; bounded candidate/result/author/metadata inspection; generic failures. |
| Crawler redirect escape/oversized download | Resolved | Shared downloader, final-host trust check, response-byte and total-time ceilings. |
| Non-atomic or redirectable classic persistence | Resolved | Deterministic serialization, fsync, atomic generation publication, manifest hashes/counts, strict JSON, cross-process locks, root ancestry checks, and root device/inode binding. POSIX member I/O is descriptor-relative; Windows fallback performs strict bounded identity-checked parsing and rejects reparse points. |
| Sensitive query/owner logging and unbounded logs | Resolved in continuation | Query/owner SHA-256 replaces plaintext; recursive event bounds, process serialization, configured JSONL rotation/backups, and symlink ancestry refusal. |
| Job restart loss and duplicate replay | Resolved for one host | SQLite queue, atomic due claims, attempt ceilings, finalizing reconciliation, source ownership, and retry deadlines. |
| Immediate retries, timer explosion, worker starvation | Resolved in continuation | SQLite stores bounded exponential deadlines; one lazy heap/condition scheduler manages delayed jobs; `INGEST_MAX_PENDING` prevents unbounded executor submission. |
| Corruption hidden behind valid job rows | Resolved in pass seven | Operator listing uses bounded rowid pagination and applies the result limit to corrupt rows rather than the initial raw-row prefix. |
| Durable public text could self-create corrupt rows | Resolved in pass seven | Public job filenames/messages are privacy-masked, control-normalized, single-line, and bounded at the shared `JobStore` write boundary. |
| Image-only PDF rejection | Substantially remediated | Optional bounded Tesseract OCR with actual-attempt accounting and page-local failure provenance. |
| Incorrect BibTeX | Resolved for supported fields | Escaped and privacy-masked values, deterministic unique keys, venue completeness fallback, scalar schema enforcement, iterator/output ceilings, and common entry types. |
| Liveness mislabeled as readiness | Resolved in continuation | Container probe verifies HTTP, both SQLite stores, and create/fsync/delete access to upload/vector volumes without initializing the embedding model. |
| Oversized body before parser controls | Resolved in continuation | Pure ASGI middleware rejects declared or streamed oversize before multipart/JSON parsing and explicitly completes partial responses. |
| Unsafe default container exposure | Resolved in continuation | Compose publishes to `127.0.0.1` by default; non-loopback exposure is documented as requiring API-key mode, TLS ingress, firewalling, and deliberate proxy trust. Uvicorn proxy-header trust is not enabled by default. |
| Release lock verification followed path redirection | Resolved in pass seven | Lock verification now rejects linked/reparse ancestry and performs bounded no-follow identity-stable reads with strict UTF-8. |

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
- immutable validated trusted-source catalogues;
- launch-directory-independent bundled frontend mounting through one validated production resolver;
- explicit audited retirement of corrupt durable rows without implicit source deletion;
- non-root read-only container, dropped capabilities, no-new-privileges, PID limit, Tesseract, named state volumes, and dependency-aware readiness;
- separated runtime/development dependencies, environment template, package metadata, MIT license, and platform lock tooling;
- one unconditional exact-head workflow for pull requests, pushes, tags, merge queues, and manual dispatch;
- removal of committed runtime databases, fabricated fixtures, destructive generators, mislabeled assets, and stale self-certification documents.

## New justified functionality

Functionality added because it directly closes audited gaps:

1. Credential-to-owner identity mapping.
2. Owner-scoped document deletion.
3. Private owner-scoped document/source registry.
4. SQLite ingestion queue and explicit state machine.
5. Atomic due claims, attempt ceilings, durable exponential deadlines, and startup reconciliation.
6. One lazily started keyed deadline scheduler for delayed jobs.
7. Bounded ingestion running-plus-queued admission.
8. Complete retained-source replacement/deletion/orphan lifecycle.
9. Optional bounded OCR for scanned and mixed PDFs.
10. DOCX/PDF/text complexity guards and source-mutation detection.
11. Stable owner/source document identity and retained-byte visual verification.
12. Shared API/CLI parsing and indexing services.
13. Transactional batch-CLI vector restoration and atomic manifests.
14. Compensating vector replacement and paginated library listing.
15. Credential-derived per-principal throttling.
16. Pre-parser request-body enforcement.
17. Dedicated bounded HTTP research executor and whole-operation deadline.
18. Server-side evidence registry and bounded response models.
19. Runtime tool-schema validation and bounded tool execution/admission.
20. Connected-peer SSRF validation and safe cross-origin redirect semantics.
21. Strict bounded web and scholarly provider adapters.
22. Exact visual page, fixed preallocation-geometry, actual-pixel, and encoded-payload ceilings.
23. Recursive scientific-result value/key sanitization.
24. Identity-bound classic state root with descriptor-relative POSIX persistence.
25. Dependency-aware container readiness and loopback-only default publishing.
26. Bounded pseudonymous rotating telemetry.
27. Security, architecture, remediation, and deployment documentation.
28. Clean-clone CI, Compose validation, and Docker build configuration.
29. Strict Windows classic JSON fallback and focused Windows CI.
30. Verified module-relative frontend static mounting.
31. Sanitized, fingerprint-bound, audited corrupt-row operator retirement.
32. Linux/Windows/macOS Python 3.10–3.12 hashed release-lock generation and verification.
33. Identity-stable no-follow release-lock verification.
34. Bounded corrupt-result pagination for operator inspection.
35. Single-line durable public job-state normalization.
36. One consolidated 16-job exact-head release gate.

No unrelated feature was added merely to increase the feature count.

## Residual limitations

These are disclosed rather than falsely marked complete:

- OCR quality depends on Tesseract, scan resolution, language packs, orientation, and layout.
- Scanned figure-caption localization does not use OCR coordinates; exact selectable caption text is required.
- PDF reading order, tables, formulas, headings, and multi-panel figure localization remain heuristic.
- Regex masking is not certified de-identification.
- File/archive validation is not malware scanning or parser sandboxing.
- Retained source files are not application-encrypted.
- Process-local limiters, schedulers, executors, and SQLite stores do not support multiple replicas.
- SQLite plus compensating vector writes is not a formal cross-store transaction or distributed exactly-once system.
- Python threads and operating-system DNS resolution cannot be force-terminated safely; bounded admission and network/provider deadlines limit impact but do not provide hard cancellation.
- Application SSRF controls should be combined with DNS/egress firewall rules.
- Final-path robots policy is applied before indexing/link expansion but after a redirect response has already been fetched.
- The readiness probe tests stores/volumes, not embedding-model download or representative semantic retrieval.
- Citation provenance is structural; semantic support still requires source inspection or a separately validated entailment system.
- Scientific-analysis outputs require expert review and replication.
- Platform-specific hashed locks must be produced by a successful target-platform resolver run; source tooling alone is not a release artifact.

## Verification status

### Source and regression coverage

Seven remediation/continuation passes re-read changed public contracts and exact commit diffs. Regression tests target tenant isolation, request-body ceilings, query/direct-route overload and deadlines, tool admission/timeouts, peer-validated networking, strict provider JSON, redirect secrets, parser/archive complexity, source identity, retained-source mutation, visual limits, scientific output sanitization, vector/CLI rollback, paginated listing, retrieval outages, queue migration/backoff, delayed scheduling, source reconciliation, corrupt-row retirement and scanning, durable public text, bounded models, privacy-safe telemetry, OCR, punctuation-aware email masking, frontend safety/portability, classic storage identity/Windows fallback, release-lock path identity, and deployment configuration.

### Observed executable evidence

- A nine-job Linux/Windows/macOS Python 3.10–3.12 release-lock matrix succeeded, including hash-contract verification, `--require-hashes --no-deps --dry-run`, and artifact upload.
- A later superseded run again completed all nine lock jobs successfully.
- The first complete Linux Python 3.12 suite passed dependency installation, `pip check`, compilation, and fatal Ruff checks.
- That run collected 713 tests, passed 711, failed two, and measured 76.25% branch coverage.
- Both failures were one sentence-final email-regex defect affecting OCR text and semantic sections; pass seven corrected the shared primitive and retained the failing contracts unchanged.

### Final exact-head verification still required

One consolidated workflow now defines:

- one exact-checkout registration job;
- Linux Python 3.10, 3.11, and 3.12 dependency checks, whitespace, compilation, fatal Ruff, pytest, and branch coverage;
- focused Windows Python 3.10 and 3.12 classic-storage compilation and regressions;
- Docker Compose configuration parsing and Docker image build including Tesseract;
- nine hashed lock jobs across Linux, Windows, and macOS for Python 3.10–3.12.

The available remediation environment cannot clone/download the branch because `github.com` DNS resolution fails. The final branch head must therefore be certified through GitHub Actions or another clean environment. PR #1 remains draft and is not merge-ready until all 16 jobs succeed on the same final head, every failure is corrected, and the resulting diff and documentation are re-audited. Coverage targets must rise from measured results, not fabricated claims.
