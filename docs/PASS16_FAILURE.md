# Pass sixteen diagnostic

```text
ERROR: file or directory not found: tests/unit/test_documentation_consistency.py


no tests ran in 0.00s

```

```diff
diff --git a/.github/workflows/release-locks.yml b/.github/workflows/release-locks.yml
index 4d0f9a5..a159f3a 100644
--- a/.github/workflows/release-locks.yml
+++ b/.github/workflows/release-locks.yml
@@ -5,7 +5,7 @@ on:
   pull_request:
   merge_group:
   push:
-    branches: [main, "agent/**"]
+    branches: [main]
     tags: ["v*"]
 
 permissions:
diff --git a/docs/EXECUTABLE_VERIFICATION.md b/docs/EXECUTABLE_VERIFICATION.md
index 7f7c29e..4b988d8 100644
--- a/docs/EXECUTABLE_VERIFICATION.md
+++ b/docs/EXECUTABLE_VERIFICATION.md
@@ -1,129 +1,79 @@
 # Executable verification ledger
 
-This ledger records observed workflow execution. It distinguishes a source-level test
-contract from a successful run against a concrete pull-request head or merge ref.
-
-## Release-lock matrix
-
-The first release-lock runs exposed and corrected three independent workflow defects:
-
-1. the isolated lock environment lacked `typing-extensions` required by pip-tools;
-2. pip-tools 7.6 was incompatible with pip 26 through its sync-module import boundary;
-3. a `shell: python` Actions step executed from a temporary directory and could not import
-   the repository's `scripts` package.
-
-The lock generator subsequently gained additional source-level controls:
-
-- resolution from an immutable bounded requirements snapshot;
-- rejection of resolver options, nested files, URLs, alternate indexes, and local paths;
-- removal of ambient pip, proxy, Python-path, certificate, keyring, and cache authority;
-- public-PyPI authority selected explicitly;
-- identity-stable generated-output reads;
-- atomic verified publication;
-- identity-stable bounded `GITHUB_OUTPUT` append;
-- strict no-follow verifier reads for every existing path component.
-
-Observed successful run:
-
-- workflow run: `30547701731`;
-- source head: `5268f9168dbb184be0b09e41af6f8931f2444aaf`;
-- result: all nine jobs succeeded;
-- platforms: Linux, Windows, macOS;
-- Python: 3.10, 3.11, 3.12;
-- every job passed generation, lock verification, hash-required installation dry run, and
-  artifact upload.
-
-A later superseded run, `30603463220`, again completed all nine lock jobs successfully.
-Passes seven through nine hardened the generator, verifier, workflow, and surrounding
-runtime boundaries after those runs, so these results are historical evidence rather than
-final release certification.
-
-## First full Linux suite
-
-Observed superseded pull-request run:
-
-- workflow run: `30603463220`;
-- tested pull-request head: `f95ecd29190d4a0fcbed772590894afaf2cadcdc` through its merge ref;
-- platform/Python: Linux, Python 3.12;
-- dependency installation: passed;
-- `pip check`: passed;
-- `python -m compileall -q .`: passed;
-- fatal Ruff checks (`E9`, `F63`, `F7`, `F82`): passed;
-- collected tests: 713;
-- passed tests: 711;
-- failed tests: 2;
-- measured branch coverage: 76.25%, above the configured 50% floor.
-
-Both failed tests exposed the same privacy-boundary bug:
-
-- OCR-derived text retained `alice@example.com.`;
-- semantic sections retained `alice@example.com.`.
-
-The shared email pattern treated the sentence-final period as a possible continuation and
-therefore failed to match the address. Pass seven corrected the shared privacy primitive
-and added punctuation-specific regressions. The failed run is evidence of a discovered and
-corrected defect, not a passing release certificate.
-
-## Consolidated exact-head workflow
-
-All release gates now live in one unconditional workflow,
-`.github/workflows/release-locks.yml`, named `Exact-head verification and release locks`.
-It contains 16 jobs:
-
-- one exact-checkout registration smoke job;
-- Linux dependency consistency, whitespace comparison, compilation, fatal Ruff checks,
-  pytest, and measured branch coverage on Python 3.10–3.12;
-- focused Windows classic-storage compilation and regressions on Python 3.10 and 3.12;
+This ledger records observed execution separately from committed source/test contracts.
+A focused successful suite certifies only its selected components. A complete release
+certificate requires all authoritative jobs to succeed for the exact current `main` SHA.
+
+## Historical broad evidence
+
+### Release-lock matrices
+
+- Run `30547701731`: all nine Linux/Windows/macOS, Python 3.10–3.12 lock jobs passed.
+- Run `30603463220`: all nine platform/Python lock jobs passed again.
+
+Each passed lock generation, verification, hash-required dry installation, and artifact
+publication. Later generator/verifier hardening means these are historical evidence, not a
+certificate for current `main`.
+
+### First complete Linux Python 3.12 suite
+
+A superseded run:
+
+- passed dependency installation, `pip check`, compilation, and fatal Ruff checks;
+- collected 713 tests;
+- passed 711 and failed 2;
+- measured 76.25% branch coverage, above the configured 50% floor.
+
+Both failures were the same sentence-final email masking defect in OCR and semantic
+sections. The shared privacy primitive and direct punctuation regressions were corrected
+after that run.
+
+## Observed focused continuation execution
+
+| Pass | Published commit | Observed gate |
+|---|---|---|
+| 10 | `8d81a1a9778f5a1224517ad5bcfa7956596e9f9e` | Operator repair, server path configuration, and trusted-source suites compiled and passed. |
+| 11 | `522ed5eb9e709a2cb8f4093d7cb083bdaa607bfc` | All 22 selected telemetry identity, append, lock, and rotation tests passed. |
+| 13 | `45fb4c26d52675539156d4d9c0a841914fdcc93c` | Security-boundary activation, reload, authentication, URL/domain/header, safe-download, peer, redirect, and response tests passed. |
+| 14 | `d6970f2598cfee4668e57d9a0262d26160e1f9b3` | All 56 selected scientific-integrity, RAG, reimport, exact-integer, and reparse tests passed. |
+| 15 | `3130208e7957fda91e1e480672d02ef332778237` | Selected classic storage, registry, search-agent, RAG, and stateful compatibility suites passed after correcting one stale workflow test path. |
+
+Pass twelve's strict security contracts were exercised by the pass-thirteen suite.
+Temporary one-shot workflows, patch scripts, and failure diagnostics were removed from
+each successful published head.
+
+## Authoritative exact-head workflow
+
+`.github/workflows/release-locks.yml`, named `Exact-head verification and release locks`,
+contains 16 jobs:
+
+- exact-checkout registration smoke;
+- Linux Python 3.10–3.12 dependency consistency, whitespace, compilation, fatal Ruff,
+  complete pytest, and measured branch coverage;
+- Windows Python 3.10 and 3.12 classic-storage compilation and regressions;
 - Docker Compose validation and container build;
-- nine Linux/Windows/macOS Python 3.10–3.12 release-lock jobs.
-
-The workflow runs for every pull request, branch push, version tag, merge queue, and manual
-dispatch. Third-party actions are pinned to immutable official release commits and
-checkout credential persistence is disabled. A single concurrency group cancels
-superseded runs. The older duplicate CI and exact-head workflows were removed so one check
-suite is authoritative.
-
-## Pass-eight source-level regressions
-
-Pass eight added or expanded tests for:
-
-- lexical frontend module/package/asset symlink and reparse-point refusal;
-- exact integer semantics across configuration, rate limiting, bounded execution,
-  scheduling, request-body limits, and upload byte ceilings;
-- boolean clock rejection;
-- periodic scheduler wall-clock rechecks;
-- readiness SQLite URI escaping, database/parent identity, reparse refusal, short writes,
-  and safe probe cleanup;
-- malformed/conflicting/excessive HTTP framing rejected before application execution;
-- owner upload root/owner/file redirection and identity boundaries.
-
-## Pass-nine source-level regressions
-
-Pass nine added or expanded tests for:
-
-- credential-free citation authority, backslash refusal, exact page numbers, and assignment
-  validation;
-- non-text document-control refusal while preserving layout whitespace;
-- truthful citation-issue overflow reporting;
-- bounded BibTeX candidate-field lookup, control removal, and citation-key construction;
-- immutable handbook reads, reparse refusal, mutation detection, and exact result counts;
-- exact single-page, scholarly-search, public-web-search, internal-search, and uploaded-RAG
-  result limits;
-- canonical provider keys and complete query-control refusal;
-- strict non-byte/malformed provider JSON handling;
-- canonical owner/document provenance for uploaded chunks;
-- hostile scientific-integrity iterable values rejected before retrieval;
-- stable before/after classic-engine signatures during reload;
-- CLI argument, query, model, owner, and terminal-output boundaries.
-
-Pass-eight and pass-nine regressions are committed source contracts, not observed passes.
-No current-head pull-request workflow run is exposed through the available connector. The
-execution container also cannot clone the branch because DNS resolution for `github.com`
-fails. No current-head success is therefore asserted here.
+- Linux/Windows/macOS Python 3.10–3.12 release-lock generation, verification,
+  hash-required dry installation, and artifact publication.
+
+It runs for `main`, version tags, manual dispatch, pull requests, and merge queues. The
+repository development policy nevertheless uses direct commits to its only branch,
+`main`.
+
+`.github/workflows/exact-head-report.yml` publishes
+`docs/LATEST_EXACT_HEAD_RESULT.md` only when:
+
+1. the completed run's branch is `main`;
+2. its SHA equals the checked-out current `main` SHA;
+3. `origin/main` still equals that SHA immediately before publication.
+
+Stale, branch, PR, and merge-queue results cannot overwrite the ledger.
 
 ## Current release boundary
 
-No merge-readiness claim is permitted until the consolidated workflow succeeds against
-one final pull-request head after all pass-seven, pass-eight, and pass-nine changes. Every
-failure must be fixed and the entire 16-job workflow rerun. The final diff and
-documentation must then be re-audited before PR #1 is moved out of draft.
+No current-head success is claimed unless `docs/LATEST_EXACT_HEAD_RESULT.md` exists and
+records `success` for the exact current `main` commit. The available local execution
+container cannot clone GitHub because DNS resolution for `github.com` fails; GitHub
+Actions is therefore the executable source of truth.
+
+Every authoritative failure remains blocking and must be corrected directly on `main`,
+then rerun across the complete matrix.
diff --git a/docs/REMEDIATION_STATUS.md b/docs/REMEDIATION_STATUS.md
index b777924..65a506b 100644
--- a/docs/REMEDIATION_STATUS.md
+++ b/docs/REMEDIATION_STATUS.md
@@ -1,135 +1,95 @@
-# Exhaustive Remediation Status
+# Exhaustive remediation status
 
-This document is the current status index for the repository-wide remediation begun on
-2026-07-27 and continued through **nine** regression/audit passes. It does not claim that
-the software is defect-free or merge-ready. Detailed findings and changes are recorded in
-`CONTINUATION_AUDIT.md` and `CONTINUATION_AUDIT_PASS2.md` through
-`CONTINUATION_AUDIT_PASS9.md`.
+This is the authoritative status index for the RigorousRAG repository-wide remediation
+started on 2026-07-27 and continued through **fifteen** audit/regression passes.
 
-## Current repository state
+## Repository state
 
-- Branch: `agent/exhaustive-remediation`
-- Draft pull request: #1
-- Pull-request state: open, mergeable, draft
-- Authoritative workflow: `.github/workflows/release-locks.yml`
+- Default and only branch: `main`
+- Open pull requests: none
+- Development policy: coherent commits directly to `main`; no feature branches or PRs
+- Authoritative verification workflow: `.github/workflows/release-locks.yml`
 - Configured gate: 16 jobs
-- Final-head executable result: **not yet established**
-
-The branch now covers every product surface in the original inventory:
-
-- classic crawler, lexical index, PageRank, generation persistence, internal search, and
-  command-line interfaces;
-- PDF/DOCX/text parsing, bounded OCR, privacy masking, stable source identity, retained
-  files, visual evidence, and vector retrieval;
-- durable ingestion state, retries, scheduling, startup recovery, document registry, and
-  corrupt-row operator recovery;
-- agent orchestration, provider calls, tool admission, evidence provenance, scientific
-  analysis, scholarly/web/page/handbook retrieval, and BibTeX export;
-- FastAPI identity, request framing/body limits, uploads, jobs, models, throttling,
-  deadlines, and frontend assets;
-- browser rendering, session-only credentials, document lifecycle, readiness, telemetry,
-  container deployment, release locks, tests, and documentation.
-
-## Implemented critical controls
-
-| Area | Current source contract |
+- Freshness-bound result reporter: `.github/workflows/exact-head-report.yml`
+- Current final-head certificate: not established unless
+  `docs/LATEST_EXACT_HEAD_RESULT.md` exists for the exact current `main` SHA with a
+  `success` conclusion
+
+Prior PRs #1–#4 are preserved only as closed/merged history. All previous implementation
+commits and every continuation pass are contained in `main`.
+
+## Implemented product and safety surfaces
+
+The current source covers:
+
+- classic crawling, lexical indexing, PageRank, immutable generation persistence, and
+  internal search;
+- PDF/DOCX/text ingestion, bounded OCR, semantic sections, stable owner-scoped document
+  identity, retained sources, visual evidence, and vector retrieval;
+- durable jobs, retries, scheduling, startup recovery, registry lifecycle, and
+  fingerprint-bound corrupt-row operator recovery;
+- scholarly, web, page, handbook, uploaded-document, comparison, limitation, debate,
+  protocol, conflict, figure, and BibTeX tools;
+- request-scoped agent orchestration, bounded tool execution, authoritative citations,
+  provider/network controls, and privacy-safe telemetry;
+- FastAPI authentication, uploads, request framing/body limits, throttling, deadlines,
+  readiness, frontend lifecycle, container deployment, release locks, and test contracts.
+
+## High-value enforced contracts
+
+| Boundary | Current contract |
 |---|---|
-| Tenant isolation | Server-owned API-key mapping or single-user identity controls every vector, registry, document, comparison, limitation, and visual operation. Caller owner headers cannot select another tenant. |
-| Request admission | Request bodies, malformed framing, running-plus-pending work, timeouts, identifiers, models, evidence, citations, warnings, metadata, and output are bounded. |
-| Upload and retention | Random owner-scoped names, exact byte limits, descriptor-relative POSIX member operations, stable root identities, private modes, `fsync`, and symbolic-link/reparse refusal. |
-| Durable ingestion | SQLite state machine, atomic claims, bounded attempts, durable backoff, one keyed scheduler, startup replay, source re-hashing, immutable parser snapshots, and compensating vector restoration. |
-| Corrupt durable rows | Normal reads fail closed. Fingerprint-bound operator tooling provides sanitized, high-water-marked, keyset-paginated inspection and explicit retirement without implicit source/vector/registry deletion. |
-| Parsing and OCR | PDF page/text/pixel ceilings, DOCX archive/member/ratio ceilings, text limits, bounded OCR attempts and rendering, partial-page provenance, strict public models, and non-text control refusal. |
-| Privacy | Native/OCR text, titles, filenames, metadata, summaries, jobs, scientific results, CLI output, paths, credentials, secrets, contact data, IPs, and non-finite values are best-effort masked. Sentence-final email punctuation is handled correctly. |
-| Retrieval provenance | Every uploaded-document result requires canonical owner/document/chunk metadata. Malformed, cross-owner, non-finite, incomplete, or invalid backend rows are dropped. |
-| Citation authority | Evidence is selected from actual tool output. Credential-bearing or browser-ambiguous citation URLs are rejected; public hosts and exact page numbers are validated. |
-| Provider/network boundary | Public DNS and connected-peer validation, redirect revalidation, proxy suppression, cross-origin secret stripping, safe POST semantics, strict MIME/header/body/deadline limits, and strict provider JSON. |
-| Classic persistence | Immutable generation files, manifest-last commit point, hashes, byte lengths, counts, cross-component checks, process locks, strict JSON, identity-bound roots, and Windows fallback checks. |
-| Frontend/browser | No untrusted `innerHTML`, no third-party runtime scripts/fonts, session-only API key/history, bounded rendering, portable lexical asset resolution, and symlink/reparse refusal for package and assets. |
-| Readiness/deployment | Bounded loopback JSON, identity-stable SQLite checks, safe volume write/fsync/delete probes, non-root read-only container, dropped capabilities, named state volumes, and loopback publishing by default. |
-| Release reproducibility | Immutable requirements snapshots, public-PyPI authority, ambient package/proxy/Python/certificate authority removal, exact pins and hashes, identity-stable lock verification, atomic publication, and immutable action pins. |
-
-## Pass-eight additions
-
-Pass eight concentrated on runtime and filesystem boundaries:
-
-- lexical frontend module/package/asset identity;
-- exact integer configuration, rate-limit, executor, scheduler, request, and upload limits;
-- boolean clock rejection and bounded scheduler wall-clock rechecks;
-- identity-stable readiness probes and SQLite URI handling;
-- strict malformed/conflicting HTTP framing;
-- reparse-aware owner upload roots and files.
-
-See `CONTINUATION_AUDIT_PASS8.md`.
-
-## Pass-nine additions
-
-Pass nine concentrated on public models, providers, retrieval, and output:
-
-- credential-free citation authority and exact page provenance;
-- document/section control-character boundaries;
-- truthful citation-verification overflow reporting;
-- bounded BibTeX candidate lookup and citation keys;
-- immutable handbook reads and mutation-aware caching;
-- strict single-page, scholarly, web, internal, and uploaded-document retrieval inputs;
-- canonical provider keys and strict non-byte/malformed provider JSON;
-- hostile scientific-integrity objects rejected before retrieval;
-- stable before/after classic-engine signatures during reload;
-- CLI validation before provider initialization and terminal-control removal.
-
-See `CONTINUATION_AUDIT_PASS9.md`.
-
-## Observed executable evidence
-
-Historical evidence only:
-
-1. Workflow run `30547701731` completed all nine Linux/Windows/macOS Python
-   3.10–3.12 release-lock jobs successfully.
-2. A later superseded lock matrix again completed all nine combinations.
-3. A superseded Linux Python 3.12 full suite:
-   - collected 713 tests;
-   - passed 711;
-   - measured 76.25% branch coverage;
-   - passed dependency consistency, compilation, and fatal Ruff checks;
-   - failed two tests caused by one shared sentence-final email-masking bug, which was
-     subsequently corrected.
-
-These runs predate later pass-seven, pass-eight, and pass-nine changes. They are not final
-release certification. See `EXECUTABLE_VERIFICATION.md`.
+| Tenant identity | Server-owned API-key mapping or configured single-user identity controls vector, registry, scientific, and lifecycle operations. Caller owner headers cannot select another tenant. |
+| Request/work admission | HTTP framing, bodies, identifiers, models, running-plus-pending work, deadlines, tool calls, evidence, citations, metadata, warnings, and responses are bounded. |
+| Upload/retained files | Random owner-scoped names, exact byte limits, descriptor-relative POSIX operations, root/file identity binding, private modes, `fsync`, and symbolic-link/reparse refusal. |
+| Durable ingestion | SQLite state machine, atomic claims, bounded attempts, durable backoff, keyed scheduling, startup replay, immutable parser snapshots, source re-hashing, and compensating vector restoration. |
+| Corrupt rows | Normal reads fail closed. Operator scans are sanitized, bounded, keyset-paginated, high-water-marked, fingerprint-bound, and never implicitly delete retained sources/vectors/registry records. |
+| Parsing/OCR | PDF page/text/pixel ceilings, DOCX member/expansion/ratio ceilings, text limits, bounded OCR and rendering, strict models, and non-text-control refusal. |
+| Privacy | Best-effort masking covers text, OCR, filenames, metadata, summaries, jobs, scientific objects, CLI/telemetry output, paths, credentials, contact data, IPs, and non-finite values. |
+| Retrieval/provenance | Uploaded results require canonical owner/document/chunk metadata and exact page provenance; malformed, cross-owner, incomplete, or non-finite rows are discarded. |
+| Network/provider | Canonical credentials/configuration, duplicate-key rejection, public DNS plus connected-peer validation, redirect revalidation, proxy suppression, cross-origin secret stripping, strict headers/MIME/body/deadline limits, and strict provider JSON. |
+| Stateful compatibility | Security, scientific, RAG, classic storage, document registry, and search-agent wrappers preserve original/public identities across reimports instead of stacking wrappers or resetting singleton state. |
+| Browser/deployment | No untrusted `innerHTML` or third-party runtime assets; session-only credentials/history; lexical asset identity; non-root read-only container; dropped capabilities; named volumes; loopback publishing by default. |
+| Release reproducibility | Immutable requirements snapshots, public-PyPI authority, ambient resolver-authority removal, exact pins/hashes, no-follow identity-stable verification, atomic publication, and immutable action pins. |
+
+## Observed execution
+
+Focused successful workflow execution is recorded for:
+
+- pass 10: operator repair, service paths, trusted sources;
+- pass 11: telemetry publication and rotation;
+- pass 13: strict security/network boundary, including pass-twelve regressions;
+- pass 14: scientific-integrity and RAG compatibility/exact-input suites;
+- pass 15: classic storage, document registry, search agent, RAG, and stateful reimport
+  suites.
+
+Historical full-suite evidence remains nonfinal: an earlier Linux Python 3.12 run passed
+711/713 tests with 76.25% branch coverage and exposed the subsequently corrected
+sentence-final email-masking defect. Two historical nine-platform lock matrices passed.
 
 ## Required final gate
 
-PR #1 must remain draft until one final exact head completes:
-
-- Linux Python 3.10, 3.11, and 3.12 full dependency, whitespace, compilation, fatal Ruff,
-  pytest, and measured coverage checks;
-- Windows Python 3.10 and 3.12 focused classic-storage checks;
-- Docker Compose validation and container build;
-- Linux, Windows, and macOS Python 3.10–3.12 lock generation, verification,
-  `--require-hashes --no-deps --dry-run`, and artifact publication.
-
-Every failure must be corrected and the entire 16-job workflow rerun. The final diff,
-documentation, and generated artifacts must then be re-audited before the PR leaves draft.
-
-No current-head success is claimed. GitHub has not exposed a pull-request workflow run for
-connector-authored heads, and the available execution container cannot clone the branch
-because DNS resolution for `github.com` fails.
+One unchanged final `main` SHA must complete all 16 authoritative jobs. Every failure must
+be corrected directly on `main`, followed by a complete rerun. A success record for an
+older SHA, another branch, pull request, or merge queue is rejected by the reporter and is
+not a release certificate.
 
 ## Residual architectural and scientific limitations
 
-These remain disclosed rather than falsely marked complete:
-
-- final-path robots policy cannot prevent the redirect response itself from being fetched;
-- provider code already running in a Python thread cannot be forcibly terminated safely;
-- application SSRF controls still require deployment DNS and egress policy;
-- filesystem anchoring is not host isolation or encryption at rest;
-- parser checks are not malware scanning or sandboxing;
-- retained sources are not application-encrypted;
-- process-local admission, scheduling, rate limiting, SQLite stores, and compensating
-  vector writes are not distributed exactly-once infrastructure;
-- OCR quality, reading order, tables, formulas, scanned captions, and multi-panel figure
-  interpretation remain heuristic;
-- regex masking is not certified de-identification;
-- readiness does not prove model availability or representative semantic retrieval;
-- structural provenance does not prove semantic entailment;
-- scientific outputs require source inspection, expert review, and replication.
+- Redirect-target robots policy cannot prevent the redirect response itself from being
+  fetched.
+- Python thread work already executing cannot be forcibly killed safely.
+- Application SSRF controls still require deployment DNS and egress policy.
+- Filesystem identity checks are not host isolation or encryption at rest.
+- Parser limits are not malware scanning or sandboxing.
+- Process-local admission, scheduling, rate limiting, SQLite, locks, and vector
+  compensation are not distributed exactly-once infrastructure.
+- OCR, reading order, formulas, tables, scanned captions, and multi-panel interpretation
+  remain heuristic.
+- Regex masking is not certified de-identification.
+- Readiness does not prove model availability or representative semantic retrieval.
+- Structural provenance does not prove semantic support or scientific truth; source
+  inspection, expert assessment, and replication remain required.
+
+Detailed records are in `CONTINUATION_AUDIT.md`, passes 2–9, passes 10–12, and passes
+13–15.
diff --git a/tests/unit/test_deployment_parity.py b/tests/unit/test_deployment_parity.py
index 7d3abd8..efa01fa 100644
--- a/tests/unit/test_deployment_parity.py
+++ b/tests/unit/test_deployment_parity.py
@@ -63,6 +63,8 @@ def test_exact_head_workflow_is_unconditional_and_complete():
     assert "name: Exact-head verification and release locks" in workflow
     assert "  pull_request:\n" in workflow
     assert "  merge_group:\n" in workflow
+    assert "    branches: [main]\n" in workflow
+    assert "agent/**" not in workflow
     assert "paths:" not in workflow.split("permissions:", 1)[0]
     assert "python-version: [\"3.10\", \"3.11\", \"3.12\"]" in workflow
     assert "runs-on: windows-latest" in workflow

```
