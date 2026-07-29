# Continuation Audit — Pass 4

Date: 2026-07-29
Branch: `agent/exhaustive-remediation`
Draft PR: #1

## Scope

This pass continued the remediation audit over the public compatibility wrappers, shared security boundaries, direct-call APIs, persistence helpers, concurrency controls, response models, and adversarial tests. The focus was not feature expansion; it was finding places where Python coercion, hostile iterables, malformed backend/provider objects, symbolic links, non-standard JSON, non-finite numbers, or unbounded streams could bypass the intended API and OpenAPI contracts.

## Classic search and persistence

- Replaced classic JSON reads with bounded descriptor-based no-follow regular-file reads.
- Rejected non-standard JSON constants, symlinked members, FIFOs, and oversized members.
- Validated and bounded crawler timing, domains, seeds, frontier state, links, URLs, headers, and persisted page fields.
- Added strict DNS/IP hostname validation and canonical IPv6 rendering.
- Recomputed sparse-index norms from accepted finite postings instead of trusting persisted norms.
- Bounded documents, terms, postings, tokenization, and snippets.
- Bounded PageRank nodes, edges, target iterables, iterations, damping, and tolerance.
- Validated classic search construction, score math, queries, context gathering, and storage paths.
- Ensured legacy summarizer prompts include every listed source and divide prompt space fairly across sources.
- Bounded provider settings, prompts, outputs, citation markers, source strings, and CLI limits.

## Vector RAG and document registry

- Hardened the public RAG subclass independently of Chroma behavior.
- Validated identifiers, metadata, filters, sections, chunk sizes, overlap, booleans, query expansion, and backend result structures.
- Enforced owner and document provenance again after vector results return.
- Added a bounded, owner-safe document-list scan.
- Revalidated Chroma path ancestry for direct construction and singleton reuse.
- Revalidated document-registry database and upload-root ancestry on every connection and singleton reuse.
- Made visual verification flags, document IDs, cleanup clocks, and public registry fields strict and bounded.

## Scientific tools and agent provider boundary

- Applied direct field limits uniformly to visual entailment, paper comparison, comparison matrices, protocol extraction, debate, conflict detection, and limitation extraction.
- Replaced raw visual extraction exceptions with controlled evidence-unavailable messages.
- Sanitized model-provider tool calls before they enter conversation history.
- Bounded tool-call IDs, names, arguments, content, citations, durations, choices, and provider fields.
- Prevented infinite provider tool streams and hostile attribute objects from escaping the boundary.
- Preserved one final synthesis turn after exactly consuming the configured tool-call budget.
- Made malformed embedded scientific JSON generic unavailable evidence rather than echoing raw malformed content.
- Bounded citation registration and preserved existing evidence labels after the evidence cap.

## Uploads, ingestion, source identity, and queue state

- Revalidated all service state paths before `server_app` import, including upload, SQLite, Chroma, and classic storage ancestors.
- Hardened durable job-store database ancestry, IDs, statuses, timestamps, retry counts, state transitions, and public text.
- Preserved lexical source-path recording while preventing job-database path redirection.
- Hardened descriptor-anchored upload storage against symlinked ancestors, non-byte streams, invalid limits, and nonregular files.
- Ensured immutable ingestion snapshots preserve strict byte-limit semantics and private regular-file permissions.
- Made standalone `ingest_file` consume a bounded no-follow byte snapshot instead of reopening the caller path during parsing.
- Re-hashed parser source identity through one bounded descriptor primitive before indexing.
- Prevented document-controlled metadata from overwriting owner ID, filename, MIME type, creation time, summary, or job ID.
- Bounded and masked document-summary provider output.

## Shared network, web, page, readiness, handbook, and telemetry boundaries

- Completed the connected-peer downloader boundary with bounded API-key configuration, URL/hostname validation, request headers, request bodies, content types, redirects, response headers, response bytes, and end-to-end deadlines.
- Charged empty streaming chunks against the total deadline.
- Treated the validated requested/redirect URL—not provider-controlled response metadata—as the authoritative final URL.
- Validated web-search query, domains, result limit, provider key, and provider JSON before result processing.
- Restricted domain filters to hostnames only; ports, credentials, paths, queries, fragments, and unsupported schemes are rejected.
- Validated direct page-extraction arguments before networking and retained structured redacted failure objects.
- Rebuilt readiness checks around strict loopback JSON, lexical path validation, SQLite identity checks, and descriptor-relative write probes.
- Hardened handbook reads against symlink ancestry, FIFOs, invalid UTF-8, oversized input, non-finite ranking state, and malformed direct query/top-k values.
- Made the shared privacy sanitizer fully no-throw and JSON-safe for hostile objects, hostile container subclasses, cycles, deep nesting, non-finite numbers, and oversized integers.
- Rebuilt telemetry around bounded single-event appends, advisory process locks, safe rotation, symlink/FIFO refusal, and no-throw public helpers.

## Concurrency and request boundaries

- Made the sliding-window rate limiter strict about integer settings, keys, finite clocks, and backwards per-key time.
- Made the bounded executor strict about limits, thread names, shutdown booleans, submit failures, cancellation, and exactly-once slot release.
- Made the keyed due scheduler strict about deadlines, keys, capacities, names, shutdown flags, invalid wall clocks, thread-start failures, and callback `BaseException`s.
- Hardened ASGI request limiting against malformed bodies, non-byte payloads, malformed `more_body`, malformed response messages, and arbitrary `ValueError` stringification.
- Preserved exact visual-document 404 translation while leaving unrelated application errors to the normal server boundary.

## Public models, verification, ingestion models, configuration, and BibTeX

- Restricted citation URLs to `http`, `https`, or `local`; stripped userinfo and rejected localhost/private literal hosts.
- Preserved bounded nested metadata, rejected boolean page numbers, enabled assignment validation, and bounded citation/warning iterables before Pydantic materialization.
- Deduplicated duplicate citation labels and duplicate evidence identities in public answers.
- Bounded citation verification to 100 citations, 1,000 markers, and 500 issues.
- Replaced the misleading zero-source pass with an explicit no-evidence warning.
- Made ingestion models strict about booleans, MIME types, IDs, source paths, timezone-aware timestamps, section iterables, aggregate section text, extras, and assignment.
- Validated configuration-helper names, defaults, bounds, ordering, and `write_back` while preserving malformed-environment fallback and clamping.
- Hardened BibTeX generation against hostile truthiness/string conversion, unsupported scalar values, credential/PII leakage, iterator failure, text-as-collection input, and unbounded total output.

## Regression coverage

This pass added or expanded focused tests for:

- classic storage symlinks, FIFOs, strict JSON, and byte ceilings;
- crawler credentials, IPv6, infinite generators, malformed persisted pages, headers, and link caps;
- sparse-index corruption, PageRank graph bombs, search math, legacy prompt completeness, and CLI cleanup;
- RAG direct parameters, malformed vector responses, owner/document filtering, listings, and symlinked paths;
- document-registry ancestor swaps and direct flags;
- all scientific-tool direct limits;
- provider-call pre-conversation sanitization and exact-budget synthesis;
- queue state, retry overflow, upload ancestry, non-byte streams, immutable snapshots, and protected metadata;
- shared downloader headers, bodies, MIME, peers, redirects, deadlines, and final URLs;
- web/page provider JSON, domains, redaction, and no-network validation ordering;
- readiness malformed values, non-standard JSON, symlinks, and descriptor probes;
- handbook FIFOs, UTF-8, replacement identity, and non-finite ranking;
- hostile privacy objects/containers, telemetry rotation/locks/truncation, and JSON validity;
- rate limiting, executor slot lifecycle, scheduler lifecycle, and malformed ASGI bodies;
- citation URL/model assignment, infinite public-model iterables, verification marker storms, ingestion-model aggregates, configuration helpers, and BibTeX output ceilings.

## Verification status

The exact branch head has **not** executed in this environment. The runtime still cannot resolve `github.com`, so it cannot download or clone the branch archive. The GitHub connector reports no workflow run for the exact head checked during this pass.

Therefore the following remain unverified against the exact current head:

- `python -m compileall -q .`
- fatal Ruff checks (`E9`, `F63`, `F7`, `F82`)
- pytest and coverage
- Docker build
- cross-platform behavior outside the statically reviewed POSIX/Windows branches

PR #1 must remain draft until a clean environment executes those gates and every resulting failure is fixed.

## Residual non-claims

- Structural provenance and lexical diagnostics do not prove scientific entailment.
- Regex privacy masking is not certified de-identification.
- Parser and archive bounds are not malware scanning or sandboxing.
- Process-local locks, schedulers, rate limits, executors, SQLite queues, and vector compensation are not distributed exactly-once infrastructure.
- Connected-peer SSRF defenses still require deployment egress controls as defense in depth.
- Python cannot forcibly terminate arbitrary provider code already running in a thread.
