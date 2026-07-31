# RigorousRAG continuation audit — Passes 10–12

Date: 2026-07-31  
Branch: `main`  
Development policy: direct, fast-forward commits to `main`; no feature branches or pull requests.

## Scope

These passes continued the repository-wide remediation after consolidation of every prior
branch and pull request into `main`. They audited durable operator repair, service path
configuration, the trusted-source catalogue, telemetry publication and rotation, and the
shared authentication/network security boundary.

This record separates observed focused execution from source-level contracts awaiting the
restored exact-head matrix.

## Pass 10 — durable repair, service paths, and trusted sources

### Operator repair audit integrity

The append-only repair ledger now:

- bounds raw reason input before masking;
- masks credentials, paths, and contact data;
- replaces every ASCII control and DEL with whitespace;
- collapses the result to one bounded public line;
- rejects an empty normalized reason;
- validates that the repair clock is numeric, finite, and nonnegative before opening the
  write transaction;
- leaves the corrupt durable row untouched when the clock is invalid.

Regressions cover control-bearing/secret-bearing reasons and a non-finite system clock.

### Service state paths

The server wrapper now inspects every existing lexical state-path component with `lstat`
and rejects both POSIX symbolic links and Windows reparse points before importing the
application. Inspection errors fail closed rather than being converted into a resolved
pathname.

### Trusted-source catalogue

Catalogue category text and HTTPS seeds now require canonical, bounded, control-free
input. Seeds reject surrounding whitespace, DEL/C0 controls, credentials, backslashes,
query strings, fragments, non-default authority, IP literals, and malformed IDNA host
labels. Category names/descriptions reject embedded controls.

### Observed verification

The focused pass-ten workflow compiled the modified modules and passed the operator,
server-configuration, and trusted-source regression suites before commit
`8d81a1a9778f5a1224517ad5bcfa7956596e9f9e` was published to `main`.

## Pass 11 — telemetry identity and destructive rotation

Telemetry publication now distinguishes two identity contracts:

1. live open-descriptor binding uses device/inode identity, which remains valid across a
   legitimate append;
2. unlink and rename/rotation decisions use a richer snapshot containing device, inode,
   ctime, mtime, size, and mode, so a same-path replacement cannot be deleted or renamed.

Additional controls include:

- exact integer semantics for token/count metrics;
- boolean rejection for numeric durations;
- lexical symlink and Windows reparse-point refusal;
- parent-directory descriptor/path identity checks;
- lock-file identity checks before, during, and after the critical section;
- destination identity checks before and after append;
- source and destination snapshot checks around every destructive rotation step;
- fail-closed handling for redirected/non-regular lock, log, and backup entries.

The first focused run exposed inode reuse in a replacement regression. The correction kept
live descriptor identity stable while applying richer snapshots only to destructive
operations. The rerun passed all 22 focused telemetry tests before commit
`522ed5eb9e709a2cb8f4093d7cb083bdaa607bfc` was published. Temporary scripts, workflows,
and failure diagnostics were removed.

## Pass 12 — authentication and network configuration boundary

A dedicated `tools.security_boundary` compatibility layer is imported by `tools` before
public security helpers are consumed. It preserves the existing connected-peer-validated
transport while replacing permissive configuration/direct-call parsing with fail-closed
contracts.

### Authentication configuration

- JSON mappings reject duplicate keys through an object-pairs hook.
- API keys must already be canonical, bounded, and free of every ASCII control.
- Configured owner IDs must already equal their normalized representation.
- Legacy comma-separated keys reject padded segments, controls, and duplicates.
- Configuration byte and entry-count ceilings remain enforced.

### URLs, domains, uploads, and headers

- upload filenames reject every ASCII control and DEL;
- public URLs reject surrounding whitespace, backslashes, controls, and oversized input
  before DNS resolution;
- hostname allowlists accept only canonical hostname authority, with no credentials,
  explicit ports, path, parameters, query, fragment, controls, or backslashes;
- request header names/values must already be canonical strings and values reject every
  ASCII control;
- response metadata does not stringify arbitrary objects and drops invalid, sensitive, or
  control-bearing fields;
- remote byte limits require exact integer/index values;
- boolean request timeouts are rejected.

A dedicated regression module covers duplicate/noncanonical key configuration, upload
names, URLs, domain allowlists, request/response headers, exact numeric limits, and
boundary activation.

## Verification status

Passes ten and eleven have observed focused successful execution as described above.
Pass twelve has committed source and regression contracts but has not yet been certified
by an observed current-head run. The temporary focused runner did not register reliably,
so it was removed instead of being represented as evidence.

The authoritative `Exact-head verification and release locks` workflow has been restored
unchanged. The next exact-head run must establish:

- Linux Python 3.10–3.12 dependency consistency, compilation, fatal Ruff checks, complete
  pytest execution, and measured branch coverage;
- Windows Python 3.10/3.12 classic-storage compilation and regressions;
- Compose validation and container build;
- Linux, Windows, and macOS Python 3.10–3.12 release-lock generation, verification,
  hash-required dry installation, and artifact publication.

Any failure remains a blocking defect and must be corrected on `main`, followed by a full
rerun on the resulting exact head.

## Residual non-claims

These passes do not change the repository's explicit architectural and scientific limits:

- application SSRF controls still require deployment DNS and egress enforcement;
- arbitrary provider code already executing in a Python thread cannot be forcibly killed
  safely;
- filesystem identity checks are not host isolation or encryption at rest;
- parser limits are not malware scanning or sandboxing;
- process-local admission, scheduling, rate limiting, SQLite stores, and compensating
  vector writes are not distributed exactly-once infrastructure;
- OCR, reading order, tables, equations, scanned captions, and multi-panel interpretation
  remain heuristic;
- regex masking is not certified de-identification;
- structural provenance does not establish semantic entailment or scientific truth.
