# RigorousRAG continuation audit — Pass 6

Date: 2026-07-30  
Branch: `agent/exhaustive-remediation`  
Draft PR: #1

## Scope

This pass continued from the explicit implementation boundaries recorded after pass five.
It addressed Windows classic-storage parity, frontend launch-directory portability,
operator handling of corrupt durable rows, and reproducible platform-specific release
locks. It also added focused regression and CI contracts for each boundary.

This document records source changes and tests. It is not an exact-head execution
certificate. The pull request remains draft until all configured checks run against the
final commit and every failure is corrected.

## 1. Windows classic JSON fallback no longer delegates to permissive legacy parsing

### Finding

POSIX classic snapshot reads already used descriptor-relative, bounded, strict UTF-8/JSON
parsing. The Windows fallback still called the legacy pathname reader, so strict rejection
of `NaN`, `Infinity`, and `-Infinity` and equivalent member-identity checks were not
implemented at parity.

### Correction

- added strict bounded pathname-fallback parsing with `parse_constant` rejection;
- require a regular file before and after open;
- compare file identity before open, after open, and after reading;
- reject symbolic links and Windows reparse points, including junction-like entries;
- enforce the configured byte ceiling before and during reads;
- quarantine only the exact invalid regular-file identity that was inspected;
- preserve the target of a refused link or reparse point;
- retain root-identity validation before and after fallback operations.

### Verification contract

Focused tests call the fallback directly and require rejection/quarantine for non-standard
JSON and oversized members while proving that symbolic-link targets remain untouched.
The main CI workflow now includes a Windows Python 3.10/3.12 job covering classic storage,
snapshot locking, and internal-search reload behavior.

## 2. Frontend mounting is independent of the process working directory

### Finding

`server_app.py` retained the historical `StaticFiles(directory="frontend")` declaration.
That declaration worked in the container's `/app` working directory but failed when the
same module was imported or launched from an arbitrary directory.

### Correction

- added a narrow, idempotent FastAPI `StaticFiles` adapter;
- only the exact legacy sentinel `"frontend"` is rebound;
- the replacement is derived from the installed module location, not `Path.cwd()`;
- the bundled directory must be a real directory rather than a symlink;
- `index.html`, `app.js`, `lifecycle.js`, and `preload.js` must be regular non-symlink files;
- every non-sentinel `StaticFiles` caller keeps the framework's normal path semantics;
- no process-wide `chdir` is performed.

### Verification contract

Tests instantiate the sentinel after changing to an unrelated directory, verify that
ordinary custom static directories are unaffected, reject missing bundled assets, and
import the real `server` module in a subprocess whose working directory is outside the
repository.

## 3. Corrupt durable job rows have explicit, non-destructive operator tooling

### Finding

Recovery correctly skipped malformed durable rows rather than replaying them. Operators
had no supported mechanism to inspect or retire those rows, leaving potentially referenced
source files protected indefinitely and encouraging unsafe direct SQL edits.

### Correction

Added `python -m tools.operator_repair` with two conservative commands:

- `list` returns only sanitized corruption reasons, row IDs, exact SHA-256 fingerprints,
  bounded public status/filename fields, source-presence booleans, and valid timestamps;
- `retire` requires the exact row ID, complete-row fingerprint, explicit confirmation
  token, and an operator reason.

Retirement:

- obtains an immediate SQLite write transaction;
- re-reads and re-fingerprints the complete row;
- refuses a row that changed after inspection;
- refuses any row that currently satisfies the durable schema;
- deletes only the selected corrupt database row;
- never deletes source files, vectors, or document-registry records;
- records the action, fingerprint, reason, timestamp, and source-preservation state in an
  append-only `operator_repairs` audit table;
- never prints the private source path or raw private database values.

### Verification contract

Tests cover output secrecy, exact confirmation, stale-fingerprint refusal, valid-row
protection, source preservation, audit-table insertion, and CLI JSON output.

## 4. Release locks are generated rather than fabricated

### Finding

Runtime dependency ranges were appropriate for development but not reproducible release
artifacts. A trustworthy lock cannot be authored statically without resolving transitive
packages for the actual operating system and Python minor version.

### Correction

- added `pip-tools` to development tooling;
- added a bounded platform-aware lock generator;
- generate names of the form `locks/runtime-<platform>-py<minor>.txt`;
- use backtracking resolution and SHA-256 hashes;
- omit embedded index URLs and trusted-host authority;
- refuse symlinked inputs/outputs and unsafe paths;
- added a verifier that requires exact `==` pins and at least one SHA-256 hash per package;
- added a GitHub Actions matrix for Linux, Windows, and macOS on Python 3.10–3.12;
- validate every generated file and perform a `--require-hashes` installation dry run;
- publish each platform/Python lock as a workflow artifact;
- run the matrix when lock inputs change, on manual dispatch, and for version tags.

The repository does not check in invented lock contents. Release artifacts must come from
a successful resolver run on the target platform and interpreter.

### Verification contract

Unit tests cover valid hashed locks, ranges or missing hashes, embedded package-index
authority, generator arguments, and symbolic-link output refusal.

## Remaining implementation and verification boundaries

The four concrete implementation items carried from pass five are now addressed in source.
The following boundaries remain:

- no clean exact-head import, compile, fatal lint, full unit/integration, coverage, Compose,
  container-build, Windows, or release-lock workflow result has yet been observed;
- any failure from those workflows must be fixed and all checks rerun on the new exact head;
- final-path robots policy can stop indexing/link expansion only after a redirect response
  has already been fetched;
- arbitrary provider code already executing in a Python thread cannot be forcibly killed;
- application SSRF protections still require deployment DNS/egress policy;
- upload/parser checks are not malware scanning or parser sandboxing;
- retained sources require deployment-provided encryption at rest where appropriate;
- process-local executors, limiters, schedulers, SQLite stores, and vector compensation are
  not distributed exactly-once infrastructure;
- OCR, document reading order, tables, formulas, scanned-caption coordinates, and
  multi-panel figure interpretation remain heuristic;
- structural provenance does not prove semantic entailment;
- scientific-analysis output still requires source inspection, domain experts, and
  replication.

## Merge gate

PR #1 must remain draft. Source inspection, added tests, commit creation, and workflow
configuration are not substitutes for execution. Merge readiness requires all final-head
checks to run successfully, all discovered failures to be corrected, and a final diff and
documentation audit after the last corrective commit.
