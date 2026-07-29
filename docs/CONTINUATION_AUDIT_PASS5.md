# RigorousRAG continuation audit — Pass 5

This pass continued the repository-wide remediation on `agent/exhaustive-remediation` after the fourth continuation audit. It concentrated on orchestration, deployment parity, ambient HTTP authority, browser lifecycle behavior, public identifiers, and filesystem paths.

## Verification status

This is a static remediation record, not an execution certificate.

The available environment still cannot clone or download the branch because `github.com` DNS resolution fails. No exact-head `compileall`, Ruff, pytest, coverage, Compose validation, container build, or GitHub Actions run was observed during this pass. The pull request must remain draft until a clean environment runs all release checks against the exact head.

## Completed changes

### Trusted-source catalogue

- Validate DNS labels using IDNA-aware hostname rules.
- Reject IP-literal seeds, malformed/empty labels, credentials, fragments, controls, and non-HTTPS schemes.
- Permit only the default HTTPS port and normalize explicit `:443` away.
- Require non-empty bounded source-category descriptions.
- Bound seed/category/domain derivation and reject infinite iterables.

### Classic and agent command-line interfaces

- Bound direct CLI argument streams before `argparse` materialization.
- Surface agent evidence, truncation, and reasoning-budget warnings instead of silently discarding them.
- Mask all CLI-displayed answer, warning, citation, URL, title, and snippet fields.
- Stream batch directory traversal rather than materializing an entire directory tree.
- Bound inspected directory entries, supported files, prior vector rows, prior vector text, metadata, and identifiers.
- Request at most one vector row beyond the rollback ceiling so oversized generations fail before unbounded materialization.
- Reject control-bearing input/output paths, provider fields, and rollback vector IDs.

### CI and deployment parity

- Fetch complete Git history before merge-base whitespace checks.
- Select the whitespace comparison base from the pull-request base, actual push `before` SHA, parent commit, or root commit as appropriate.
- Expose active summary, embedding, vector-metadata, semantic-section, RAG-query, outbound-body, executor, and response budgets in both `.env.example` and Compose.
- Add static tests that keep critical runtime settings synchronized across operator-facing deployment files.

### Shared downloader and injected sessions

- Continue using public-DNS and actual-connected-peer validation, redirect revalidation, byte limits, MIME checks, and end-to-end deadlines.
- Neutralize ambient authority on injected `requests.Session` instances under a process-wide lock:
  - environment proxy inheritance;
  - explicit proxies;
  - session authentication;
  - default headers;
  - cookies;
  - default query parameters;
  - response hooks;
  - disabled TLS verification;
  - client certificates.
- Retain only caller-explicit bounded request headers/body and the session transport adapters.
- Restore the exact prior session state on success and failure.
- Add concurrency tests proving one shared-session request cannot restore proxies or credentials while another request is in flight.

### Public citation and evidence models

- Validate citation DNS hostnames with IDNA-aware label rules.
- Reject whitespace-bearing, empty-label, malformed, single-label, localhost, private, and reserved public hosts.
- Preserve valid global IPv4/IPv6 and `local://` evidence identifiers.
- Continue masking URI credentials, sensitive query values, local paths, contact details, and non-finite metadata.

### Browser lifecycle

- Add a preload fetch boundary so the first `/config` request receives a client-side deadline before application boot.
- Add `AbortController` deadlines for subsequent API operations.
- Bound drag/drop and file-input enumeration to 100 files without `Array.from` materialization.
- Give uploads, status polls, document deletion, and research requests explicit operation-appropriate deadlines.
- Bound rendered document cards to 5,000 rows.
- Keep API keys/history in session storage only and retain DOM-only rendering without `innerHTML`.
- Align browser field `maxlength` values with server contracts for API keys, queries, scientific text, identifiers, and BibTeX metadata.

### Uniform identifier and filesystem policy

Scientific/document prose remains multiline-capable, but identifiers and filesystem paths now reject every ASCII control character (`0x00`–`0x1f` and `0x7f`). The policy is enforced across:

- ingested document IDs and private source paths;
- retained-document registry IDs, filenames, SQLite path, and upload root;
- durable job IDs, document IDs, queue source paths, stored rows, and recoverable rows;
- summary models, indexing job IDs, and source-identity paths;
- RAG collection/model/document identifiers, metadata keys, Chroma paths, backend chunk IDs, and listed document IDs;
- scientific figure/document/model identifiers and comparison item identifiers;
- uploaded-document retrieval model/document/backend identifiers;
- descriptor-anchored upload roots, retained source paths, and owner-relative filenames;
- standalone parser paths;
- batch paths, provider fields, and vector rollback IDs;
- classic-index storage roots and classic snapshot member names;
- server state paths before `server_app` import.

Corrupted durable rows or malformed vector-backend rows are skipped/fail closed rather than replayed or emitted as evidence.

## Regression coverage added or expanded

Coverage was added for:

- trusted seed hostname, port, fragment, and IP-literal rejection;
- bounded CLI argument streams and surfaced warnings;
- bounded batch traversal and backend-request limits;
- deployment environment parity and event-aware CI whitespace checks;
- injected-session ambient authority on success, failure, and concurrent use;
- malformed citation DNS names;
- browser preload order, request deadlines, upload enumeration, source capability, and field ceilings;
- control-bearing identifiers across ingestion models, registry, durable jobs, document service, RAG, scientific tools, and uploaded-document retrieval;
- corrupt SQLite recovery rows and malformed Chroma identifiers;
- control-bearing server state paths, upload roots/sources, parser paths, classic roots, internal-index roots, batch paths, and rollback vector IDs.

## Static compatibility review

The large compatibility wrappers were re-read after full-file reconstruction:

- `server.py` retained pre-import configuration, bounded scheduling, replay-only recovery, immutable ingestion, owner-scoped deletion, error handlers, and final module handoff.
- `tools/rag.py` retained ingestion, deletion, HyDE/multi-query generation, owner-scoped query, document listing, singleton caching, and module handoff.
- `tools/job_store.py` retained SQLite initialization/migration, ping, prune, retry calculation, state transitions, atomic claim, owner-scoped reads, recovery, and active-source protection.
- `storage.py` retained root identity binding, advisory snapshot locking, strict POSIX JSON reads, quarantine, descriptor-relative writes, and module handoff.

No missing public method or alias assignment was found in those wrappers during the static review.

## Remaining implementation and verification boundaries

- **Windows classic JSON fallback:** POSIX classic reads reject non-standard JSON constants through descriptor-relative parsing. The Windows compatibility fallback still delegates to the legacy pathname parser and should be replaced or validated on Windows before release.
- **Frontend static mount portability:** `server_app.py` still mounts `frontend` relative to the process working directory. The container uses `/app`, but arbitrary launch directories are not yet supported safely.
- **Release execution:** no exact-head clean import, syntax, lint, unit/integration, coverage, Compose, container, or Windows test result is available.
- **Dependency locks:** development ranges remain intentionally flexible; release deployments still need platform-specific lock files with hashes.
- **Distributed semantics:** executors, admission, scheduler, SQLite stores, and vector compensation remain process-local/single-host mechanisms.
- **Scientific limitations:** OCR, reading order, tables, formulas, scanned captions, and multi-panel visual localization remain heuristic. Provenance does not prove semantic entailment.
- **Operational corruption repair:** corrupted durable rows are skipped conservatively; operator repair/retirement tooling remains preferable to silently deleting their potentially referenced files.
