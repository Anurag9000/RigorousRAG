# RigorousRAG continuation audit — Pass 8

Date: 2026-07-31  
Branch: `agent/exhaustive-remediation`  
Draft PR: #1

## Scope

Pass eight continued from the actual branch head after pass seven and compared newly added
regression files against the production code they purported to protect. It then audited
adjacent configuration, admission, scheduling, readiness, request-framing, and upload
boundaries for the same classes of truncation, path-redirection, identity, and malformed
input errors.

This record describes implemented source changes and regression contracts. It does not
claim that the final exact head has passed the configured workflow. No pull-request run is
currently exposed for the connector-authored head, and the available execution container
cannot clone the repository because DNS resolution for `github.com` fails.

## 1. Frontend assets are validated lexically before resolution

### Finding

The frontend resolver still called `Path.resolve()` before checking its own module path.
A symbolic-link package root or resolver module could therefore redirect validation to a
different tree. A committed regression also referenced a Windows reparse-point constant
that the production module did not define, making that test deterministically red.

### Correction

- validate the lexical `__file__` path without resolving through links;
- reject symbolic-link and Windows reparse-point final entries and ancestors;
- validate the resolver module, `tools` directory, package root, frontend directory, and
  every required asset;
- preserve precise diagnostics for a redirected final file versus a redirected ancestor;
- keep the compatibility `StaticFiles(directory="frontend")` adapter explicit and narrow;
- keep the real production mount on `frontend_directory()`.

### Regression contract

Tests now cover a linked resolver module, linked package ancestor, reparse-flagged asset,
reparse-flagged package root, missing assets, arbitrary launch directories, and ordinary
non-sentinel `StaticFiles` behaviour.

## 2. Programmatic integer limits are exact rather than truncating

### Finding

Several public constructors used `int(value)` and rejected only fractional built-in
`float` values. Fractional `Decimal`, `Fraction`, and other numeric objects could therefore
be silently truncated into configuration or admission limits.

### Correction

The following boundaries now use the exact integer/index protocol and reject booleans:

- bounded environment helper parameters;
- rate-limit request and key capacities;
- bounded-executor worker and pending capacities;
- due-scheduler pending-key capacity;
- request-body maximum bytes;
- owner upload/read/copy byte limits.

Portable environment variable names are also limited to ASCII letters, digits, and
underscores and may not begin with a digit.

### Regression contract

Tests reject fractional `Decimal` and `Fraction` inputs while accepting objects that
implement an exact `__index__` value.

## 3. Boolean clocks cannot masquerade as numeric timestamps

### Finding

Python converts booleans to `0.0` or `1.0`. Rate-limit and due-scheduler clock parsers
therefore accepted `True` and `False` as real timestamps.

### Correction

- rate-limit timestamps reject booleans explicitly;
- due-scheduler deadlines reject booleans before a thread is started;
- readiness HTTP timeouts treat booleans as malformed and use the bounded default.

### Regression contract

Tests require boolean clock values to fail without mutating queues or starting scheduler
threads.

## 4. Long scheduler waits recheck wall-clock state

### Finding

The durable scheduler could sleep for up to 86,400 seconds for a far-future entry. A large
forward wall-clock adjustment would not wake that condition wait, delaying already-due
work until the original timeout expired.

### Correction

Long condition waits are capped at 60 seconds. The durable database remains authoritative,
but the scheduler now periodically re-evaluates wall-clock deadlines and replacement
entries.

## 5. Readiness probes are identity-stable and reparse-aware

### Finding

The readiness probe:

- detected POSIX symlinks but not Windows reparse points;
- rejected only NUL rather than every ASCII path control;
- embedded SQLite paths directly into a URI, allowing `?` and `#` to change URI parsing;
- did not bind the SQLite parent identity across the probe;
- used one `os.write()` for directory probes;
- could close a probe and then unlink a replacement entry during cleanup.

### Correction

- reject symbolic-link and reparse-point components;
- reject every ASCII control-bearing state path;
- percent-encode SQLite URI paths and open them in existing read/write mode;
- enable SQLite `query_only` for the readiness query;
- verify database and parent identities before and after the query;
- handle short writes explicitly;
- on POSIX, unlink the private probe name while its original descriptor remains open,
  then write and `fsync` the descriptor and directory;
- on portable platforms, use an anonymous temporary-file lifecycle where supported;
- preserve bounded loopback HTTP, strict JSON, no proxies, and no redirects.

### Regression contract

Tests cover URI metacharacters, all ASCII path controls, simulated reparse flags, short
writes, boolean timeout fallback, and absence of leftover probe files.

## 6. Malformed HTTP framing fails before application execution

### Finding

Malformed, conflicting, excessive, or non-byte `Content-Length` headers were treated as
though no length had been supplied. Stream counting still prevented an oversized body,
but ambiguous request framing reached application code.

### Correction

- distinguish an absent `Content-Length` from malformed framing;
- reject conflicting duplicate values, signs, whitespace, non-ASCII bytes, excessive
  digits, malformed fields, too many fields, and oversized names/values;
- return a generic connection-closing HTTP 400 before invoking the application;
- continue accepting identical duplicate lengths;
- continue stream-counting genuinely lengthless/chunked bodies;
- retain the 413 body ceiling and partial-response completion behaviour;
- remove the unrelated import-time frontend monkeypatch from request middleware.

### Regression contract

Tests cover exact limit objects, malformed/conflicting framing, excessive header
structures, identical duplicate lengths, chunked overflow, malformed ASGI body messages,
and response completion after an early response start.

## 7. Owner upload paths and limits are reparse-aware

### Finding

Owner-scoped upload storage used exact descriptor-relative POSIX final operations, but its
lexical checks did not detect Windows reparse points and its byte-limit parser silently
truncated fractional non-float numeric objects. The POSIX root descriptor was also not
compared with the root identity inspected before opening.

### Correction

- use exact integer byte limits;
- reject every ASCII control-bearing path;
- reject symbolic-link or reparse-point root, owner, source, and final entries;
- bind the POSIX root descriptor to the inspected root device/inode;
- repeat root and owner identity checks in the portable fallback;
- verify portable source and destination identities across reads and writes;
- preserve owner-scoped lexical path structure and descriptor-relative POSIX member I/O.

### Regression contract

Tests cover fractional numeric limits, exact index values, linked roots and owner
directories, simulated reparse roots, non-byte streams, oversized cleanup, owner scoping,
and owner-directory replacement.

## Verification status

Observed executable evidence remains the earlier superseded runs documented in
`EXECUTABLE_VERIFICATION.md`:

- two nine-job platform lock matrices succeeded on older heads;
- one Linux Python 3.12 full suite collected 713 tests, passed 711, measured 76.25% branch
  coverage, and exposed the subsequently corrected sentence-final email masking defect.

No executable result is claimed for the current pass-eight head. Required final evidence:

- Linux Python 3.10–3.12 full suites;
- Windows Python 3.10 and 3.12 storage suites;
- Compose validation and container build;
- Linux/Windows/macOS Python 3.10–3.12 release-lock generation, verification, hash-only
  installation dry run, and artifact publication;
- correction and complete rerun after every failure.

## Residual architectural and scientific boundaries

Pass eight does not change these non-claims:

- final-path robots policy cannot prevent the redirect response itself from being fetched;
- arbitrary provider code already running in a Python thread cannot be forcibly killed;
- application SSRF controls still require deployment DNS and egress policy;
- filesystem anchoring is not host isolation or encryption at rest;
- parser checks are not malware scanning or sandboxing;
- process-local schedulers, executors, limiters, SQLite stores, and vector compensation are
  not distributed exactly-once infrastructure;
- OCR, reading order, tables, formulas, scanned-caption coordinates, and multi-panel
  interpretation remain heuristic;
- regex masking is not certified de-identification;
- structural provenance does not establish semantic entailment;
- scientific outputs require source inspection, experts, and replication.

## Merge gate

PR #1 must remain draft until the authoritative 16-job workflow succeeds on one final
exact head and the resulting diff, documentation, and generated lock artifacts are
re-audited.
