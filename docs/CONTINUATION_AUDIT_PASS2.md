# Continuation Audit — Pass Two

## Scope

This record covers the second regression pass over the remediation branch. It documents defects found in the already-remediated implementation, the source changes made to address them, the regression contracts added, and the verification boundary that still prevents merge readiness.

This is not a claim that the repository is defect-free. No executable test, lint, coverage, or container result may be inferred from source inspection alone.

## Findings and corrections

### 1. Missing visual documents escaped as internal errors

**Finding:** direct visual entailment performed owner-scoped vector metadata lookup before constructing its fail-closed result. A missing or other-owner document could raise a `ValueError` that surfaced as an internal server error.

**Correction:**

- translate only the exact owner-scoped missing-document sentinel on `/tool/visual-entailment`;
- return a generic no-store `404` body;
- preserve unrelated `ValueError` exceptions for the normal server error boundary;
- add middleware-level and full FastAPI route regressions.

### 2. Non-replacement vector rollback could delete existing evidence

**Finding:** `RAGLayer.add_document(..., replace=False)` did not snapshot prior rows. Deterministic IDs could overwrite existing chunks, and a later batch failure could delete or fail to restore those pre-existing chunks.

**Correction:**

- snapshot the complete owner/document generation in both replacement modes;
- restore overwritten deterministic rows on failure;
- delete stale rows only when `replace=True`;
- add success and partial-failure regressions for non-replacement mode.

### 3. Shared injected HTTP sessions had a proxy-state race

**Finding:** `safe_download` temporarily set an injected `requests.Session.trust_env=False`, then restored it. Two concurrent calls sharing the same injected session could interleave, allowing one call to restore proxy inheritance while the other was still following redirects.

**Correction:**

- serialize mutable `trust_env` handling for injected shared sessions through a process-wide re-entrant lock;
- keep owned per-call sessions fully concurrent;
- preserve the caller’s original `trust_env` setting after completion;
- add a two-thread regression proving proxy inheritance cannot be restored mid-download.

### 4. Redirected crawler targets bypassed canonical state checks

**Finding:** the crawler validated the requested URL before fetch but did not fully reapply crawl-state policy to the final canonical URL. Redirect targets could be fetched again, counted against the wrong host quota, or indexed despite final-path robots denial.

**Correction:**

- mark the canonical target visited;
- refuse duplicate canonical pages;
- recheck final-host domain quota;
- recheck robots policy for a changed canonical URL before indexing or link expansion;
- add duplicate, final-quota, and final-robots regressions.

The shared downloader still obtains the redirect response before the crawler can apply final-path robots policy. This remains a disclosed crawler limitation rather than being overstated as pre-fetch robots enforcement.

### 5. Classic crawl/index/PageRank files had no cross-file commit point

**Finding:** individually atomic JSON files could still represent different builds after a crash between crawl-state, sparse-index, and PageRank writes. Startup loaded those files independently despite documentation claiming generation consistency.

**Correction:**

- write immutable generation-specific crawl, index, and PageRank files;
- publish `snapshot_manifest.json` last as the atomic cross-file commit point;
- record exact filenames, SHA-256 digests, byte lengths, and record counts;
- validate crawl-page, index-document, and PageRank key consistency;
- fail the entire generation closed on missing, malformed, oversized, tampered, or inconsistent members;
- preserve the previous manifest when a new generation write is interrupted;
- reject partial or mixed legacy fixed-file state before first migration;
- add `CLASSIC_STORAGE_DIR` and `CLASSIC_MAX_SNAPSHOT_FILE_BYTES` configuration;
- add round-trip, tamper, interrupted-save, component-mismatch, migration, and engine-reload regressions.

### 6. Classic internal search cached an old generation indefinitely

**Finding:** the internal academic-search adapter created one engine for the process lifetime. A crawler rebuild performed by another process was invisible until API restart.

**Correction:**

- track manifest and legacy-file device, inode, ctime, mtime, and size identity;
- rebuild the cached engine whenever the committed storage signature changes;
- detect atomic same-size manifest replacement;
- enforce the schema’s 2,000-character query limit for direct Python callers;
- add reload, same-size replacement, query-bound, and citation-mapping regressions.

### 7. Frontend visual actions used retention rather than capability

**Finding:** document cards enabled the PDF figure tool whenever any source was retained. Retained DOCX, Markdown, and text documents therefore displayed a nonfunctional visual action.

**Correction:**

- use `visual_source_available`, not `source_retained`, as the action capability;
- disable the figure action for ineligible retained sources;
- distinguish text-only, retained-but-ineligible, and eligible-but-verified-on-use states;
- add static frontend contract tests.

### 8. Public response and telemetry metadata admitted non-finite numbers

**Finding:** nested metadata could contain `NaN` or infinities, causing non-standard JSON or response serialization failure. Generic telemetry details could also include local paths, credentials, or secret query parameters, and the telemetry append path could follow a symlink.

**Correction:**

- normalize non-finite response metadata to `null`;
- serialize telemetry with `allow_nan=False`;
- normalize invalid durations and counts;
- recursively sanitize telemetry keys and values;
- mask local paths, URI credentials, and common secret parameters;
- refuse symlinked telemetry paths and use no-follow append semantics where supported;
- add JSON-safety, masking, rotation, and symlink-target regressions.

### 9. Container readiness trusted redirects, proxies, and symlinked state

**Finding:** the readiness client followed redirect/proxy behavior, read unbounded JSON, and resolved symlinked database or storage paths.

**Correction:**

- restrict readiness HTTP to loopback hosts;
- disable proxy inheritance and redirects;
- cap response bytes;
- require a small JSON object with `status=ok`;
- reject symlink components for SQLite and writable directories;
- retain create/fsync/delete persistence probes;
- add remote, oversized, non-200, symlink, and aggregate readiness regressions.

### 10. BibTeX types could claim missing required venue metadata

**Finding:** records defaulted to `@article` even without a journal, and several common LaTeX-special characters were not escaped. Direct callers could also emit an unbounded number of records.

**Correction:**

- require venue/institution/school fields for typed records;
- fall back to `@misc` when required fields are absent;
- escape backslash, braces, percent, hash, dollar, ampersand, underscore, tilde, and caret once;
- cap output at 100 valid emitted records;
- skip malformed non-mapping iterable elements;
- add completeness, escaping, and direct-call cap regressions.

### 11. Web provider results could trigger excessive DNS validation

**Finding:** a provider payload could cause many sequential public-address resolutions before the requested result limit was reached.

**Correction:**

- add `WEB_SEARCH_MAX_RESULT_CANDIDATES`;
- inspect only the bounded candidate prefix;
- apply caller hostname restrictions before DNS/public-address validation;
- add candidate-cap and pre-DNS domain-filter regressions.

### 12. Direct page results exposed sensitive URL components

**Finding:** success and failure results returned URL strings verbatim. Credentials embedded in a URL or secret query parameters could enter citations, model context, or browser output.

**Correction:**

- preserve the original URL only for the validated network request;
- mask credentials and common secret parameters in every returned URL;
- apply the same boundary to successful final URLs and failed input URLs;
- add success and failure regressions.

### 13. Very short citation evidence silently passed diagnostics

**Finding:** lexical citation checking skipped evidence with fewer than eight meaningful tokens and could then report that the citation-structure check passed.

**Correction:**

- emit an explicit `weak_evidence_text` diagnostic;
- include the meaningful-token count;
- require manual source inspection in the user-facing audit message;
- retain separate serious structural checks for missing, duplicate, or absent markers.

### 14. Handbook retrieval had unbounded file and paragraph behavior

**Finding:** handbook retrieval read the whole file, direct calls did not enforce schema query length, and one pathological paragraph could bypass the intended 1,200-character chunk size.

**Correction:**

- require a regular non-symlink UTF-8 handbook;
- add `HANDBOOK_MAX_BYTES` and `HANDBOOK_MAX_CHUNKS`;
- enforce a hard 1,200-character chunk ceiling;
- enforce the 2,000-character direct-call query limit before file access;
- rebuild the cached sparse index when file identity changes;
- add long-paragraph, chunk-limit, oversized, symlink, query, and reload regressions.

### 15. Uploaded-document retrieval could emit unattributed citations

**Finding:** the RAG tool initialized embeddings before empty-query validation, had weaker direct-call length checks than intended, and could produce `local://unknown` citations from malformed metadata.

**Correction:**

- validate empty, oversized query, owner ID, and document ID before vector-store initialization;
- add bounded query/document lengths to the tool schema;
- require returned chunk metadata to match the authenticated owner;
- require a non-empty bounded document ID;
- enforce requested document equality before citation construction;
- skip malformed or cross-owner chunks;
- add no-initialization, length, owner, and provenance regressions.

### 16. Legacy classic-index summarization was independently unbounded

**Finding:** the legacy LLM adapter accepted arbitrary query/context counts, built large prompts, did not bound Ollama-returned content after receipt, and returned unsupported numeric citation markers without warning. The AI CLI performed lexical retrieval before the adapter’s query validation.

**Correction:**

- cap query, aligned sources, per-source context, total prompt, summary, source strings, and model identifiers;
- add `LEGACY_LLM_TIMEOUT_SECONDS` for the OpenAI-compatible client;
- warn on missing or unsupported numeric markers;
- validate the AI CLI query before lexical retrieval;
- report invalid interactive queries without ending the session;
- add prompt/output/source/query/citation regressions.

A running Ollama client call still depends on the provider’s own transport behavior; post-response output is bounded, but Python cannot force-terminate a provider call safely.

### 17. Final source identity hashing could read an unbounded replacement

**Finding:** the post-parser identity recheck streamed the source without an independent byte ceiling. A host replacement between parsing and indexing could force large hashing I/O before being rejected.

**Correction:**

- hash only a regular no-follow file;
- enforce the upload byte ceiling independently during the final check;
- compare current SHA-256 and derived owner/document UUID before summary or vector writes;
- add digest, oversized replacement, symlink swap, and invalid-limit regressions.

## Configuration added or clarified

- `CLASSIC_STORAGE_DIR`
- `CLASSIC_MAX_SNAPSHOT_FILE_BYTES`
- `WEB_SEARCH_MAX_RESULT_CANDIDATES`
- `HANDBOOK_MAX_BYTES`
- `HANDBOOK_MAX_CHUNKS`
- `LEGACY_LLM_TIMEOUT_SECONDS`

These are present in `.env.example` and Docker Compose.

## Verification boundary

The branch defines clean-clone checks for Python 3.10–3.12 compilation, fatal Ruff checks, pytest with coverage, and Docker build. The connected remediation environment still cannot clone or download the branch because `github.com` DNS resolution fails, and connector-authored commits have produced no exact-head GitHub Actions run or commit status through the available interface.

The pull request must remain draft until the exact release head executes successfully and every failure is corrected.
