# RigorousRAG continuation audit — Pass 9

Date: 2026-07-31  
Branch: `agent/exhaustive-remediation`  
Draft PR: #1

## Scope

Pass nine continued from the pass-eight exact source head and audited public model,
provider-adapter, retrieval, scientific-analysis, CLI, export, and classic-engine reload
boundaries. The recurring themes were silent numeric coercion, incomplete ASCII-control
handling, credential-bearing authority, malformed provider structures, identity drift,
and tests that still encoded superseded permissive behavior.

This record describes implemented source contracts and committed regressions. It does not
claim that the current head has passed the configured workflow. GitHub still exposes no
pull-request workflow run for connector-authored heads, and the available execution
container cannot clone the repository because `github.com` DNS resolution fails.

## 1. Citation URLs and page provenance fail closed

### Finding

Canonical citation construction silently discarded embedded URL credentials. It also
permitted browser-ambiguous backslashes and allowed Pydantic to coerce booleans or
fractional numeric objects into page numbers.

### Correction

- public and `local://` citation URLs reject username/password authority;
- citation URLs reject ASCII controls, backslashes, malformed ports, private/local public
  targets, invalid host labels, unsupported schemes, and empty local source identities;
- query secrets and private path fragments remain masked for otherwise valid URLs;
- page numbers use the exact integer/index protocol and remain bounded to 1–1,000,000;
- assignment validation applies the same rules after construction.

### Regression contract

Tests cover credentialed URLs, browser-ambiguous backslashes, public IPv6, exact index
objects, booleans, floats, `Decimal`, `Fraction`, assignment bypass attempts, and query
secret masking.

## 2. Ingestion text and semantic page numbers are exact

### Finding

Document and section text rejected NUL but allowed other non-text C0 controls and DEL.
Semantic page numbers could be coerced from booleans or fractional numeric objects.

### Correction

- document and section content reject all non-text ASCII controls and DEL;
- tab, carriage return, and newline remain valid document-layout characters;
- page numbers require exact bounded integers on creation and assignment;
- aggregate section-text, metadata, private-path, timezone, and serialization controls
  remain enforced.

## 3. Citation verification reports issue truncation truthfully

### Finding

Exactly 500 genuine citation issues were incorrectly relabeled as a truncation marker,
even when no 501st issue had been dropped.

### Correction

The verifier now distinguishes capacity from overflow. Exactly 500 real issues remain
intact. The final item becomes `issue_limit_reached` only after an additional issue is
actually attempted and cannot be recorded.

## 4. BibTeX generation does not copy arbitrary candidate mappings

### Finding

Each candidate dictionary was copied wholesale before a small known field set was used.
Hostile or extremely large mappings could therefore trigger unbounded work. ASCII
controls could also enter BibTeX fields, and long year strings could disproportionately
inflate citation keys.

### Correction

- read only known scalar fields through bounded direct lookups;
- never iterate or copy the complete candidate mapping;
- replace every ASCII control with whitespace before BibTeX escaping;
- cap year digits used in citation keys;
- preserve candidate count, output count, field, total-output, duplicate-key, privacy, and
  unsupported-type bounds.

## 5. Handbook reads bind one immutable filesystem identity

### Finding

The handbook reader used high-level path checks and a single read call. Windows reparse
points, same-path replacement, and in-read mutation were not fully represented in the
cache signature. `top_k` also silently truncated fractional numeric objects.

### Correction

- reject lexical symbolic-link and reparse-point components;
- reject every ASCII control-bearing query and path;
- bind pre-open, opened-descriptor, post-read descriptor, and post-read path identity;
- require stable device, inode, ctime, mtime, and size;
- perform bounded incremental no-follow reads with strict UTF-8;
- use exact bounded `top_k` values;
- rebuild the cache after same-size/same-mtime replacement because inode/ctime changes;
- reject in-place mutation observed during a read.

## 6. Single-page and scholarly/web provider adapters are canonical

### Single-page extraction

- byte limits use exact integer semantics;
- all ASCII controls in user-agent strings become whitespace;
- user agents remain bounded before reaching the downloader;
- downloaded HTML/text, final URLs, content bytes, charset fallback, and public errors
  remain bounded and privacy-masked.

### Scholarly search

- queries reject all ASCII controls;
- years and result limits use exact integer semantics;
- provider keys must already be canonical—no leading/trailing whitespace or controls;
- non-byte, malformed, non-standard, or structurally invalid provider JSON fails with a
  generic provider error;
- authors, external identifiers, candidates, years, URLs, and citation construction remain
  bounded.

### Public web search

- queries and allowed domains reject all controls and backslash ambiguity;
- domains remain IDNA-canonical hostnames without paths, ports, credentials, query, or
  fragments;
- result limits use exact integer semantics;
- provider keys must be canonical and control-free;
- non-byte and malformed provider JSON is distinguished from request failure;
- candidate iteration and citation construction remain bounded and per-result fail-closed.

## 7. Uploaded-document retrieval requires canonical provenance

### Finding

`n_results` silently truncated fractional numeric objects. Direct queries and HyDE output
rejected only NUL. Backend owner metadata was stripped before comparison, so noncanonical
owner strings could be accepted. Oversized page numbers reached citation validation.

### Correction

- exact bounded result counts;
- all-control rejection for direct query and HyDE text;
- strict document/model identifiers and boolean flags before vector initialization;
- exact canonical owner metadata equality;
- bounded backend chunks, IDs, parent/child text, scores, section titles, and page numbers;
- malformed or cross-owner chunks are dropped individually;
- empty queries and empty HyDE expansions remain no-work results.

## 8. Scientific-integrity direct calls validate before retrieval

### Finding

Document IDs, metrics, and queries were converted with truthiness and `str()`. Hostile
objects could execute `__bool__` or `__str__`, leak private diagnostics, or become
unexpected identifiers before scientific retrieval or provider work.

### Correction

- comparison document IDs and metrics must be bounded strings;
- queries, claims, figure labels, document IDs, model names, and owner IDs are validated
  before retrieval, PDF rendering, or model calls;
- hostile iterable items are rejected without truth-testing or stringification;
- existing immutable PDF-byte rendering, exact pixel/encoded-byte ceilings, owner-scoped
  registry lookup, and conservative result semantics remain unchanged.

## 9. Internal classic search reloads only from stable state

### Finding

The internal-search adapter lacked Windows reparse detection, used a non-identity-stable
manifest read, silently truncated fractional limits, and could publish an engine loaded
while the committed storage generation changed. One malformed hit could also abort later
valid citations.

### Correction

- storage paths and manifest entries reject symbolic links and reparse points;
- manifest bytes are read through one stable pre/open/post identity with strict bounded
  JSON;
- nonregular generation members receive an invalid signature;
- query controls and fractional limits fail before engine initialization;
- malformed hits and invalid citations are skipped individually;
- replacement engines are published only when before/after storage signatures match;
- unstable candidates are closed, the previous engine is preserved, and repeated churn
  fails explicitly after a bounded number of attempts.

## 10. CLI validation precedes provider initialization

### Finding

One-shot queries were validated only after provider construction. CLI argument strings
were count-bounded but not individually length-bounded and rejected only NUL. Model output
could include terminal escape, carriage-return, DEL, or other control sequences.

### Correction

- every argument is a bounded control-free string;
- one-shot queries are validated before provider construction;
- model and owner identifiers are canonical before `SearchAgent` initialization;
- terminal output masks private data and removes terminal-control sequences while
  preserving ordinary answer line breaks and tabs;
- provider and request failures remain generic;
- model names and queries are not echoed in status output.

## Verification status

Historical executable evidence remains unchanged:

- two superseded nine-job release-lock matrices succeeded;
- one superseded Linux Python 3.12 suite collected 713 tests, passed 711, measured 76.25%
  branch coverage, and exposed the subsequently corrected sentence-final email masking
  defect.

No current-head success is claimed. Final required evidence remains:

- Linux Python 3.10–3.12 full suites;
- Windows Python 3.10 and 3.12 storage suites;
- Compose validation and container build;
- Linux/Windows/macOS Python 3.10–3.12 release-lock generation, verification,
  hash-required dry installation, and artifact publication;
- correction and complete rerun after every failure.

## Residual boundaries

Pass nine does not change these non-claims:

- application SSRF protection still requires deployment DNS and egress controls;
- provider code already running in a Python thread cannot be forcibly terminated safely;
- filesystem anchoring is not host isolation or encryption at rest;
- parser checks are not malware scanning or sandboxing;
- process-local admission, scheduling, rate limiting, SQLite stores, and vector
  compensation are not distributed exactly-once infrastructure;
- OCR, reading order, tables, formulas, scanned captions, and multi-panel interpretation
  remain heuristic;
- regex masking is not certified de-identification;
- structural provenance does not establish semantic entailment;
- scientific output requires source inspection, experts, and replication.

## Merge gate

PR #1 must remain draft until the authoritative 16-job workflow succeeds on one final
exact head and the resulting diff, documentation, and generated lock artifacts are
re-audited.
