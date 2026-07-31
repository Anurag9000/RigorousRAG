# RigorousRAG continuation audit — Pass 7

Date: 2026-07-31  
Branch: `agent/exhaustive-remediation`  
Draft PR: #1

## Scope

This pass re-read the concrete implementation boundaries carried forward after pass six
and compared them against the actual branch head rather than the stale pull-request
summary. It then followed the new frontend, release-lock, operator-repair, durable-text,
and verification surfaces for adjacent correctness failures.

This record describes source changes and regression contracts. It is not a declaration
that the final exact head has passed every executable release gate. PR #1 remains draft
until the consolidated workflow succeeds after the final corrective commit.

## 1. The production frontend mount now uses the validated resolver directly

### Finding

Pass six added `tools.frontend_static.frontend_directory()` and compatibility tests, but
the real service still mounted a separately implemented path. The production path checked
the directory but did not reuse the helper's complete required-asset checks. This created
two frontend-root implementations that could drift.

### Correction

- `server_app.py` now imports and calls `frontend_directory()` directly;
- the mount is independent of process working directory;
- the resolver is anchored to the resolved module file;
- the resolver module, frontend directory, and required `index.html`, `app.js`,
  `lifecycle.js`, and `preload.js` assets must be regular non-symlink entries;
- the optional legacy `StaticFiles(directory="frontend")` adapter remains narrow and
  idempotent, but is no longer the production service's security boundary;
- the helper documentation now accurately states that adapter installation is explicit,
  not an import-time side effect.

### Regression contract

The existing portability suite imports the real service from an unrelated working
directory, rejects missing/symlinked assets, and proves ordinary non-sentinel static
mounts retain framework semantics.

## 2. Release-lock verification no longer follows redirected paths

### Finding

The lock generator rejected linked inputs and outputs, but the verifier checked only the
final path with high-level `Path` methods and then reopened it with `read_text()`. A
symlinked/reparse-point ancestor or a validation-to-read replacement could redirect the
verified bytes.

### Correction

- reject every existing lexical symbolic-link or Windows reparse-point component;
- require a bounded regular final file;
- open with no-follow/non-inheritable/binary flags where available;
- compare device/inode identity before open, after open, and after reading;
- read incrementally under the 20 MB ceiling;
- require strict UTF-8;
- preserve all existing exact-pin, SHA-256 hash, and no-index-authority checks.

### Regression contract

New tests reject a linked ancestor, reject a final symlink while preserving its target,
and reject malformed UTF-8 lock bytes.

## 3. Corrupt-row listing scans for corrupt results rather than truncating raw rows

### Finding

`list_corrupt_jobs(limit=N)` selected only the first `N` database rows and filtered valid
rows afterward. A valid prefix could therefore conceal later corruption and return an
empty list even though malformed durable state existed.

### Correction

- use bounded keyset pagination by `rowid`;
- scan in 500-row batches;
- apply `limit` to corrupt results, not raw rows;
- retain a separate 100,000-row scan ceiling to prevent unbounded operator queries;
- preserve sanitized output and exact full-row fingerprints.

### Regression contract

New tests place corruption behind a valid prefix and require it to be returned with
`limit=1`; they also prove the result limit remains a corrupt-record limit.

## 4. Durable public job text can no longer create self-corrupting rows

### Finding

The repair tool correctly treated ASCII controls in persisted public filenames/messages
as malformed, but `JobStore._safe_public_text()` masked PII without removing those
controls. A normal write could therefore create a row that recovery or operator tooling
later classified as corrupt.

### Correction

- public job filenames and messages are now normalized to bounded single-line values at
  the durable write boundary;
- every ASCII control, including DEL, becomes a space after privacy masking;
- trimming and field ceilings are applied after normalization;
- the invariant applies to API, recovery, CLI, tests, and future direct `JobStore` callers.

### Regression contract

New tests persist newline, tab, carriage-return, and DEL-bearing fields, verify the exact
normalized public values, require the repair scan to consider the row valid, and verify
500/2,000-character ceilings after normalization.

## 5. Exact-head verification is one unconditional PR-observable workflow

### Finding

Three partially overlapping workflows existed:

- the historical CI workflow;
- the pass-six exact-head workflow;
- the release-lock workflow, which also contained a Linux test job.

The connector could observe only pull-request-triggered runs, while some exact-head work
was duplicated in push-triggered workflows. The release workflow also had path filters
that could skip configuration-only changes.

### Correction

- consolidated Linux Python 3.10–3.12 tests, Windows Python 3.10/3.12 classic-storage
  tests, Compose validation, container build, and the nine platform/Python lock jobs into
  `.github/workflows/release-locks.yml`;
- renamed it `Exact-head verification and release locks`;
- made it unconditional for every pull request;
- retained branch pushes, version tags, and manual dispatch;
- kept full-history, event-aware whitespace comparison;
- retained per-job dependency checks, compilation, fatal Ruff checks, pytest/branch
  coverage, hash-only lock installation dry runs, and artifacts;
- removed the two duplicate workflows instead of consuming redundant runners.

The repository now has one authoritative 15-job merge gate. No lightweight compatibility
check is substituted for the actual verification work.

## Verification status

Observed before the final pass-seven commits:

- all nine Linux/Windows/macOS Python 3.10–3.12 release-lock jobs succeeded on prior
  exact heads;
- dependency installation, `pip check`, compilation, and fatal Ruff checks completed
  successfully in an earlier Linux exact-head job before later commits superseded it.

Still required on the final pass-seven head:

- all three Linux full test/coverage jobs;
- both Windows classic-storage jobs;
- Compose validation and container build;
- all nine regenerated release-lock jobs;
- correction and complete rerun after any failure.

The connected execution environment still cannot clone the repository because DNS
resolution for `github.com` fails. Connector-backed GitHub Actions therefore remains the
executable source of truth.

## Residual architectural and scientific boundaries

Pass seven does not change the following honest non-claims:

- final-path robots policy can stop indexing/link expansion only after a redirect response
  has already been fetched;
- arbitrary provider code already running in a Python thread cannot be forcibly killed;
- application SSRF controls require deployment DNS/egress policy;
- filesystem anchoring is not host isolation or encryption at rest;
- parser checks are not malware scanning or sandboxing;
- process-local admission, rate limiting, scheduling, SQLite stores, and vector
  compensation are not distributed exactly-once infrastructure;
- OCR, reading order, tables, formulas, scanned-caption coordinates, and multi-panel
  interpretation remain heuristic;
- regex masking is not certified de-identification;
- structural provenance does not prove semantic entailment;
- scientific-analysis output requires source inspection, domain experts, and replication.

## Merge gate

PR #1 must remain draft until the one authoritative exact-head workflow succeeds on the
final commit and the resulting diff, documentation, and generated release artifacts have
been re-audited.
