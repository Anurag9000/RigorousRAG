# Continuation Audit Record

## Purpose

This record covers the regression-audit and implementation continuation performed after the initial exhaustive RigorousRAG remediation. It is intentionally narrower than the repository-wide goals document: it records defects found in the remediation itself, the controls added to correct them, the tests written for those controls, and the remaining verification boundary.

This document is not a declaration that the branch is defect-free or production-ready. Source inspection and test creation are complete only for the items listed here; executable verification remains required for the exact release commit.

## Continuation findings and implemented corrections

### 1. Delayed ingestion retries consumed one timer thread per job

**Finding:** persisted retry deadlines were correct, but the server created one `threading.Timer` for every delayed job. High retry cardinality could therefore create excessive daemon threads.

**Correction:**

- added a keyed heap/condition `DueScheduler`;
- starts one daemon thread lazily only after the first delayed job;
- replacing a key invalidates the older heap entry without an O(n) removal;
- cancellation and shutdown clear pending work;
- callback exceptions cannot terminate scheduler liveness;
- SQLite `next_attempt_at` remains the durable source of truth.

### 2. Ingestion workers had a bounded count but an unbounded executor queue

**Finding:** `ThreadPoolExecutor(max_workers=N)` alone does not bound submitted-but-not-running futures.

**Correction:**

- added `INGEST_MAX_PENDING` as a running-plus-queued admission ceiling;
- saturated durable jobs remain `queued` in SQLite;
- one short scheduler admission retry is created rather than submitting more futures;
- admission is released only when the submitted future completes;
- submit failure releases admission before rescheduling.

### 3. Agent tools had bounded workers but unbounded pending submission

**Finding:** repeated requests and timed-out tools could accumulate indefinitely in the process-wide executor queue.

**Correction:**

- added `MAX_PENDING_TOOL_TASKS`;
- tool submission requires a process-wide admission slot;
- saturation returns `ExecutorUnavailable` without queueing;
- a timed-out running tool retains capacity until it actually completes;
- submit failure releases capacity synchronously;
- tool exception text remains excluded from model context and public warnings.

### 4. HTTP research operations lacked whole-route admission and deadlines

**Finding:** `/query` originally used a generic shared threadpool, and direct visual/protocol routes had only per-principal rate limiting. Slow or timed-out operations could accumulate without a dedicated running-plus-pending ceiling.

**Correction:**

- added reusable `BoundedExecutor` infrastructure;
- `/query`, direct visual entailment, and direct protocol extraction share a dedicated research executor;
- `QUERY_WORKERS` controls running threads;
- `QUERY_MAX_PENDING` controls running plus queued operations;
- saturation returns generic `503` with `Retry-After`;
- `QUERY_TIMEOUT_SECONDS` is a whole-operation HTTP deadline;
- timed-out running work retains its slot until underlying completion;
- lifecycle shutdown cancels queued work and closes the executor.

### 5. Retained visual sources could differ from indexed evidence

**Finding:** the registry proved owner/path containment but did not prove that retained bytes still represented the document identified by `doc_id`. A host-side mutation after indexing could make visual analysis inspect different evidence.

**Correction:**

- stable document IDs remain owner ID plus source-file SHA-256;
- visual lookup re-hashes the current retained file;
- the registry derives the expected UUID again and compares it to `doc_id`;
- mutation makes the source visually unavailable;
- the file remains retained, orphan-protected, manageable, and deletable.

### 6. Retained-PDF visual analysis lacked complete complexity ceilings

**Finding:** visual access needed independent protection from pathological retained PDFs and large rendered image payloads.

**Correction:**

- `VISUAL_MAX_PDF_PAGES` bounds pages inspected;
- `VISUAL_CLIP_HEIGHT_POINTS` bounds the caption-adjacent region;
- `VISUAL_MAX_RENDER_PIXELS` bounds actual rendered pixels;
- `VISUAL_MAX_ENCODED_BYTES` bounds the exact base64 payload before it reaches a vision model;
- ordinary document listing performs only cheap retained-PDF eligibility checks;
- full identity/page/geometry verification runs on visual access;
- unsafe PDFs remain retained and deletable rather than being mistaken for orphans.

### 7. Scientific result objects could expose diagnostics through values or keys

**Finding:** direct parser/model error strings could contain filesystem paths, URI credentials, or secret query parameters. Mapping keys were also previously unsanitized.

**Correction:**

- expanded metadata masking for POSIX, Windows, home, and `file://` paths;
- masks credentials embedded in URIs;
- masks common API-key, access-token, password, and secret query parameters;
- recursively sanitizes mapping values and keys;
- bounds key length;
- preserves colliding sanitized keys using deterministic `#2`, `#3`, and later suffixes;
- every scientific-tool JSON result passes through this boundary.

### 8. The ASGI body limiter could leave a partial response unfinished

**Finding:** if downstream code started a response and then read enough request data to cross the body ceiling, the status could no longer be replaced with `413`, and the response body could remain open.

**Correction:**

- the middleware tracks response start and completion;
- an unstarted response receives the bounded no-store `413` response;
- an already-started partial response is explicitly completed with `more_body=False`;
- conflicting content lengths fall back to streamed byte counting.

### 9. Documentation and deployment controls had drifted from code

**Finding:** the README, architecture, security model, remediation ledger, example environment, and Compose file still described earlier queue, visual, and route behavior.

**Correction:**

- synchronized query, tool, ingestion, request-body, visual page/pixel/encoded-byte, OCR, retrieval, and telemetry controls;
- documented eligible-versus-verified retained-PDF state;
- removed limitations that are now implemented;
- retained only genuine residual limitations.

## Regression contracts added or expanded

The continuation adds focused tests for:

- lazy scheduler startup, keyed replacement, cancellation, shutdown, and callback failure isolation;
- delayed-ingestion deduplication and centralized release;
- executor saturation without unbounded ingestion submission;
- bounded-executor overload, queued-work ceilings, release, and shutdown;
- query success, overload, timeout, and capacity retention after timeout;
- direct protocol execution through the same bounded research executor;
- tool-executor saturation, submit failure, shared-pool reuse, and timeout-slot retention;
- retained-PDF page and render-pixel ceilings;
- owner/content byte verification before visual access;
- refusal of host-mutated retained evidence;
- exact encoded visual payload limits;
- scientific JSON path/credential sanitization;
- metadata-key masking, length bounds, and collision preservation;
- ASGI declared, chunked, conflicting-length, and partial-response completion behavior.

## Current residual limitations

The continuation does not remove these architectural or scientific limits:

- OCR quality depends on Tesseract, scan quality, language packs, orientation, and layout.
- Scanned-caption coordinate localization is not implemented; visual localization requires selectable caption text.
- PDF reading order, tables, formulas, headings, and complex multi-panel figures remain heuristic.
- Regex masking is not certified de-identification.
- File/archive checks are not malware scanning or parser sandboxing.
- Retained files require deployment-provided encryption at rest where necessary.
- Executors, schedulers, rate limiting, SQLite stores, and vector compensation are process-local/single-host mechanisms.
- Python threads and operating-system DNS resolution cannot be forcibly terminated safely.
- Application SSRF controls require network egress policy as defense in depth.
- Chroma plus SQLite compensation is not a formal cross-store transaction.
- Citation provenance is structural and does not prove semantic entailment.
- Scientific outputs still require source inspection, expert review, and replication.
- Release deployments should produce platform-specific dependency locks with hashes.

## Verification boundary

The branch contains CI configuration for:

- Python 3.10, 3.11, and 3.12;
- `python -m compileall -q .`;
- fatal Ruff syntax/name checks;
- pytest with branch coverage and the configured baseline;
- Docker image build.

The remediation environment still cannot resolve `github.com` for clone/archive access and therefore cannot execute the branch locally. No passing status may be inferred from source inspection or from the presence of test files. The pull request must remain draft until GitHub Actions or another clean environment runs the exact head and every failure is corrected.
