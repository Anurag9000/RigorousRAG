# Executable verification ledger

This ledger records observed workflow execution. It distinguishes a source-level test
contract from a successful run against a concrete pull-request head or merge ref.

## Release-lock matrix

The first release-lock runs exposed and corrected three independent workflow defects:

1. the isolated lock environment lacked `typing-extensions` required by pip-tools;
2. pip-tools 7.6 was incompatible with pip 26 through its sync-module import boundary;
3. a `shell: python` Actions step executed from a temporary directory and could not import
   the repository's `scripts` package.

The lock generator subsequently gained additional source-level controls:

- resolution from an immutable bounded requirements snapshot;
- rejection of resolver options, nested files, URLs, alternate indexes, and local paths;
- removal of ambient pip, proxy, Python-path, certificate, keyring, and cache authority;
- public-PyPI authority selected explicitly;
- identity-stable generated-output reads;
- atomic verified publication;
- identity-stable bounded `GITHUB_OUTPUT` append;
- strict no-follow verifier reads for every existing path component.

Observed successful run:

- workflow run: `30547701731`;
- source head: `5268f9168dbb184be0b09e41af6f8931f2444aaf`;
- result: all nine jobs succeeded;
- platforms: Linux, Windows, macOS;
- Python: 3.10, 3.11, 3.12;
- every job passed generation, lock verification, hash-required installation dry run, and
  artifact upload.

A later superseded run, `30603463220`, again completed all nine lock jobs successfully.
Passes seven through nine hardened the generator, verifier, workflow, and surrounding
runtime boundaries after those runs, so these results are historical evidence rather than
final release certification.

## First full Linux suite

Observed superseded pull-request run:

- workflow run: `30603463220`;
- tested pull-request head: `f95ecd29190d4a0fcbed772590894afaf2cadcdc` through its merge ref;
- platform/Python: Linux, Python 3.12;
- dependency installation: passed;
- `pip check`: passed;
- `python -m compileall -q .`: passed;
- fatal Ruff checks (`E9`, `F63`, `F7`, `F82`): passed;
- collected tests: 713;
- passed tests: 711;
- failed tests: 2;
- measured branch coverage: 76.25%, above the configured 50% floor.

Both failed tests exposed the same privacy-boundary bug:

- OCR-derived text retained `alice@example.com.`;
- semantic sections retained `alice@example.com.`.

The shared email pattern treated the sentence-final period as a possible continuation and
therefore failed to match the address. Pass seven corrected the shared privacy primitive
and added punctuation-specific regressions. The failed run is evidence of a discovered and
corrected defect, not a passing release certificate.

## Consolidated exact-head workflow

All release gates now live in one unconditional workflow,
`.github/workflows/release-locks.yml`, named `Exact-head verification and release locks`.
It contains 16 jobs:

- one exact-checkout registration smoke job;
- Linux dependency consistency, whitespace comparison, compilation, fatal Ruff checks,
  pytest, and measured branch coverage on Python 3.10–3.12;
- focused Windows classic-storage compilation and regressions on Python 3.10 and 3.12;
- Docker Compose validation and container build;
- nine Linux/Windows/macOS Python 3.10–3.12 release-lock jobs.

The workflow runs for every pull request, branch push, version tag, merge queue, and manual
dispatch. Third-party actions are pinned to immutable official release commits and
checkout credential persistence is disabled. A single concurrency group cancels
superseded runs. The older duplicate CI and exact-head workflows were removed so one check
suite is authoritative.

## Pass-eight source-level regressions

Pass eight added or expanded tests for:

- lexical frontend module/package/asset symlink and reparse-point refusal;
- exact integer semantics across configuration, rate limiting, bounded execution,
  scheduling, request-body limits, and upload byte ceilings;
- boolean clock rejection;
- periodic scheduler wall-clock rechecks;
- readiness SQLite URI escaping, database/parent identity, reparse refusal, short writes,
  and safe probe cleanup;
- malformed/conflicting/excessive HTTP framing rejected before application execution;
- owner upload root/owner/file redirection and identity boundaries.

## Pass-nine source-level regressions

Pass nine added or expanded tests for:

- credential-free citation authority, backslash refusal, exact page numbers, and assignment
  validation;
- non-text document-control refusal while preserving layout whitespace;
- truthful citation-issue overflow reporting;
- bounded BibTeX candidate-field lookup, control removal, and citation-key construction;
- immutable handbook reads, reparse refusal, mutation detection, and exact result counts;
- exact single-page, scholarly-search, public-web-search, internal-search, and uploaded-RAG
  result limits;
- canonical provider keys and complete query-control refusal;
- strict non-byte/malformed provider JSON handling;
- canonical owner/document provenance for uploaded chunks;
- hostile scientific-integrity iterable values rejected before retrieval;
- stable before/after classic-engine signatures during reload;
- CLI argument, query, model, owner, and terminal-output boundaries.

Pass-eight and pass-nine regressions are committed source contracts, not observed passes.
No current-head pull-request workflow run is exposed through the available connector. The
execution container also cannot clone the branch because DNS resolution for `github.com`
fails. No current-head success is therefore asserted here.

## Current release boundary

No merge-readiness claim is permitted until the consolidated workflow succeeds against
one final pull-request head after all pass-seven, pass-eight, and pass-nine changes. Every
failure must be fixed and the entire 16-job workflow rerun. The final diff and
documentation must then be re-audited before PR #1 is moved out of draft.
