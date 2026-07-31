# Executable verification ledger

This ledger records observed workflow execution. It distinguishes a source-level test
contract from a successful run against a concrete pull-request head or merge ref.

## Release-lock matrix

The first release-lock runs exposed and corrected three independent workflow defects:

1. the isolated lock environment lacked `typing-extensions` required by pip-tools;
2. pip-tools 7.6 was incompatible with pip 26 through its sync-module import boundary;
3. a `shell: python` Actions step executed from a temporary directory and could not import
   the repository's `scripts` package.

The final lock generator:

- installs an isolated `pip>=25,<26` and pip-tools 7.6 toolchain;
- resolves with backtracking;
- pins unsafe bootstrap packages rather than leaving warnings;
- emits exact SHA-256 hashes;
- publishes its verified absolute output path directly through `GITHUB_OUTPUT`;
- validates every requirement is exactly pinned and hashed;
- performs a `pip --require-hashes --no-deps --dry-run` installation check;
- uploads the generated lock artifact.

Observed successful run:

- workflow run: `30547701731`;
- source head: `5268f9168dbb184be0b09e41af6f8931f2444aaf`;
- result: all nine jobs succeeded;
- platforms: Linux, Windows, macOS;
- Python: 3.10, 3.11, 3.12;
- every job passed generation, lock verification, hash-required installation dry run, and
  artifact upload.

A later superseded run, `30603463220`, again completed all nine lock jobs successfully.
Pass seven subsequently hardened verifier path/identity handling, so every final lock job
must regenerate and revalidate its artifact on the final head.

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

All release gates now live in one unconditional pull-request workflow,
`.github/workflows/release-locks.yml`, named `Exact-head verification and release locks`.
It contains 16 jobs:

- one exact-checkout registration smoke job;
- Linux dependency consistency, whitespace comparison, compilation, fatal Ruff checks,
  pytest, and measured branch coverage on Python 3.10–3.12;
- focused Windows classic-storage compilation and regressions on Python 3.10 and 3.12;
- Docker Compose validation and container build;
- nine Linux/Windows/macOS Python 3.10–3.12 release-lock jobs.

The workflow runs for every pull request, branch pushes, version tags, merge queues, and
manual dispatches. A single concurrency group cancels superseded runs. The older duplicate
CI and exact-head workflows were removed so one check suite is authoritative.

## Current release boundary

No merge-readiness claim is permitted until the consolidated workflow succeeds against
the final pull-request head after the privacy, lock-verifier, frontend, repair-scan, job
text, and workflow corrections. Every failure must be fixed and the entire 16-job workflow
rerun. The final diff and documentation must then be re-audited before PR #1 is moved out
of draft.
