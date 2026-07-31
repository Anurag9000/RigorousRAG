# Exhaustive Remediation Status

This document is the current status index for the repository-wide remediation begun on
2026-07-27 and continued through **nine** regression/audit passes. It does not claim that
the software is defect-free or merge-ready. Detailed findings and changes are recorded in
`CONTINUATION_AUDIT.md` and `CONTINUATION_AUDIT_PASS2.md` through
`CONTINUATION_AUDIT_PASS9.md`.

## Current repository state

- Branch: `agent/exhaustive-remediation`
- Draft pull request: #1
- Pull-request state: open, mergeable, draft
- Authoritative workflow: `.github/workflows/release-locks.yml`
- Configured gate: 16 jobs
- Final-head executable result: **not yet established**

The branch now covers every product surface in the original inventory:

- classic crawler, lexical index, PageRank, generation persistence, internal search, and
  command-line interfaces;
- PDF/DOCX/text parsing, bounded OCR, privacy masking, stable source identity, retained
  files, visual evidence, and vector retrieval;
- durable ingestion state, retries, scheduling, startup recovery, document registry, and
  corrupt-row operator recovery;
- agent orchestration, provider calls, tool admission, evidence provenance, scientific
  analysis, scholarly/web/page/handbook retrieval, and BibTeX export;
- FastAPI identity, request framing/body limits, uploads, jobs, models, throttling,
  deadlines, and frontend assets;
- browser rendering, session-only credentials, document lifecycle, readiness, telemetry,
  container deployment, release locks, tests, and documentation.

## Implemented critical controls

| Area | Current source contract |
|---|---|
| Tenant isolation | Server-owned API-key mapping or single-user identity controls every vector, registry, document, comparison, limitation, and visual operation. Caller owner headers cannot select another tenant. |
| Request admission | Request bodies, malformed framing, running-plus-pending work, timeouts, identifiers, models, evidence, citations, warnings, metadata, and output are bounded. |
| Upload and retention | Random owner-scoped names, exact byte limits, descriptor-relative POSIX member operations, stable root identities, private modes, `fsync`, and symbolic-link/reparse refusal. |
| Durable ingestion | SQLite state machine, atomic claims, bounded attempts, durable backoff, one keyed scheduler, startup replay, source re-hashing, immutable parser snapshots, and compensating vector restoration. |
| Corrupt durable rows | Normal reads fail closed. Fingerprint-bound operator tooling provides sanitized, high-water-marked, keyset-paginated inspection and explicit retirement without implicit source/vector/registry deletion. |
| Parsing and OCR | PDF page/text/pixel ceilings, DOCX archive/member/ratio ceilings, text limits, bounded OCR attempts and rendering, partial-page provenance, strict public models, and non-text control refusal. |
| Privacy | Native/OCR text, titles, filenames, metadata, summaries, jobs, scientific results, CLI output, paths, credentials, secrets, contact data, IPs, and non-finite values are best-effort masked. Sentence-final email punctuation is handled correctly. |
| Retrieval provenance | Every uploaded-document result requires canonical owner/document/chunk metadata. Malformed, cross-owner, non-finite, incomplete, or invalid backend rows are dropped. |
| Citation authority | Evidence is selected from actual tool output. Credential-bearing or browser-ambiguous citation URLs are rejected; public hosts and exact page numbers are validated. |
| Provider/network boundary | Public DNS and connected-peer validation, redirect revalidation, proxy suppression, cross-origin secret stripping, safe POST semantics, strict MIME/header/body/deadline limits, and strict provider JSON. |
| Classic persistence | Immutable generation files, manifest-last commit point, hashes, byte lengths, counts, cross-component checks, process locks, strict JSON, identity-bound roots, and Windows fallback checks. |
| Frontend/browser | No untrusted `innerHTML`, no third-party runtime scripts/fonts, session-only API key/history, bounded rendering, portable lexical asset resolution, and symlink/reparse refusal for package and assets. |
| Readiness/deployment | Bounded loopback JSON, identity-stable SQLite checks, safe volume write/fsync/delete probes, non-root read-only container, dropped capabilities, named state volumes, and loopback publishing by default. |
| Release reproducibility | Immutable requirements snapshots, public-PyPI authority, ambient package/proxy/Python/certificate authority removal, exact pins and hashes, identity-stable lock verification, atomic publication, and immutable action pins. |

## Pass-eight additions

Pass eight concentrated on runtime and filesystem boundaries:

- lexical frontend module/package/asset identity;
- exact integer configuration, rate-limit, executor, scheduler, request, and upload limits;
- boolean clock rejection and bounded scheduler wall-clock rechecks;
- identity-stable readiness probes and SQLite URI handling;
- strict malformed/conflicting HTTP framing;
- reparse-aware owner upload roots and files.

See `CONTINUATION_AUDIT_PASS8.md`.

## Pass-nine additions

Pass nine concentrated on public models, providers, retrieval, and output:

- credential-free citation authority and exact page provenance;
- document/section control-character boundaries;
- truthful citation-verification overflow reporting;
- bounded BibTeX candidate lookup and citation keys;
- immutable handbook reads and mutation-aware caching;
- strict single-page, scholarly, web, internal, and uploaded-document retrieval inputs;
- canonical provider keys and strict non-byte/malformed provider JSON;
- hostile scientific-integrity objects rejected before retrieval;
- stable before/after classic-engine signatures during reload;
- CLI validation before provider initialization and terminal-control removal.

See `CONTINUATION_AUDIT_PASS9.md`.

## Observed executable evidence

Historical evidence only:

1. Workflow run `30547701731` completed all nine Linux/Windows/macOS Python
   3.10–3.12 release-lock jobs successfully.
2. A later superseded lock matrix again completed all nine combinations.
3. A superseded Linux Python 3.12 full suite:
   - collected 713 tests;
   - passed 711;
   - measured 76.25% branch coverage;
   - passed dependency consistency, compilation, and fatal Ruff checks;
   - failed two tests caused by one shared sentence-final email-masking bug, which was
     subsequently corrected.

These runs predate later pass-seven, pass-eight, and pass-nine changes. They are not final
release certification. See `EXECUTABLE_VERIFICATION.md`.

## Required final gate

PR #1 must remain draft until one final exact head completes:

- Linux Python 3.10, 3.11, and 3.12 full dependency, whitespace, compilation, fatal Ruff,
  pytest, and measured coverage checks;
- Windows Python 3.10 and 3.12 focused classic-storage checks;
- Docker Compose validation and container build;
- Linux, Windows, and macOS Python 3.10–3.12 lock generation, verification,
  `--require-hashes --no-deps --dry-run`, and artifact publication.

Every failure must be corrected and the entire 16-job workflow rerun. The final diff,
documentation, and generated artifacts must then be re-audited before the PR leaves draft.

No current-head success is claimed. GitHub has not exposed a pull-request workflow run for
connector-authored heads, and the available execution container cannot clone the branch
because DNS resolution for `github.com` fails.

## Residual architectural and scientific limitations

These remain disclosed rather than falsely marked complete:

- final-path robots policy cannot prevent the redirect response itself from being fetched;
- provider code already running in a Python thread cannot be forcibly terminated safely;
- application SSRF controls still require deployment DNS and egress policy;
- filesystem anchoring is not host isolation or encryption at rest;
- parser checks are not malware scanning or sandboxing;
- retained sources are not application-encrypted;
- process-local admission, scheduling, rate limiting, SQLite stores, and compensating
  vector writes are not distributed exactly-once infrastructure;
- OCR quality, reading order, tables, formulas, scanned captions, and multi-panel figure
  interpretation remain heuristic;
- regex masking is not certified de-identification;
- readiness does not prove model availability or representative semantic retrieval;
- structural provenance does not prove semantic entailment;
- scientific outputs require source inspection, expert review, and replication.
