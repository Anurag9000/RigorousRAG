# Security Model

## Deployment modes

### Single-user mode

When neither `API_KEY_OWNERS_JSON` nor `ALLOWED_API_KEYS` is set, every request is assigned the server-controlled `SINGLE_USER_OWNER_ID`. This mode is suitable only for a trusted local workstation or otherwise isolated service.

### Authenticated multi-user mode

Set `API_KEY_OWNERS_JSON` to a JSON object mapping opaque API keys to owner IDs:

```json
{"key-for-alice":"alice","key-for-lab-b":"lab-b"}
```

The server derives tenant identity from this mapping. `X-Owner-ID` is intentionally ignored. API keys should be randomly generated, distributed through a secret manager, rotated, and never committed.

`ALLOWED_API_KEYS` remains a compatibility option. Each key receives a distinct server-derived owner ID based on a one-way digest; explicit mappings are preferred.

## HTTP request and execution admission

- `MAX_REQUEST_BODY_BYTES` is enforced by pure ASGI middleware before FastAPI parses JSON or multipart data, including chunked bodies without a usable `Content-Length`.
- Conflicting or malformed length declarations fall back to streamed counting; oversized requests receive a bounded no-store `413` response and the connection is closed.
- If an application has already started a response before a streamed limit is crossed, the middleware explicitly terminates that response instead of leaving the connection hanging.
- Per-principal request throttling is controlled by `REQUESTS_PER_MINUTE`.
- `/query`, direct visual entailment, and direct protocol extraction share a process-wide `BoundedExecutor`. `QUERY_WORKERS` controls running threads, `QUERY_MAX_PENDING` controls running plus queued work, and excess requests fail closed with `503` rather than entering an unbounded executor queue.
- `QUERY_TIMEOUT_SECONDS` is a whole-operation HTTP deadline. Timed-out running work continues to hold admission until its underlying Python thread actually exits; it cannot create unlimited replacement threads.
- Python cannot safely force-terminate a running third-party thread. Provider/network timeouts and bounded admission therefore remain required together.

## Upload, parsing, retention, and document identity

- Maximum source bytes are controlled by `MAX_UPLOAD_BYTES`.
- Only `.pdf`, `.docx`, `.md`, and `.txt` are accepted.
- Uploaded names are display metadata only; storage names are random and owner-scoped.
- Uploads are streamed, flushed, and `fsync`ed before the durable job row is created.
- If durable queue creation fails, the newly written upload is removed immediately and the API returns a generic unavailable response.
- Symlinked source files are rejected.
- PDF and DOCX signatures are verified; text files containing NUL bytes are rejected as binary.
- DOCX packages are checked for required members, duplicate names, unsafe paths, symlinks, encryption, member count, total expansion, and compression ratio.
- PDF page count, total extracted characters, OCR attempts, OCR render pixels, and vector chunk count are bounded.
- The source is checked for size, timestamp, and inode changes during parsing, then its SHA-256-derived identity is recomputed immediately before summarization/vector writes.
- Stable document identity is derived from owner ID plus source-file SHA-256. The source hash is not exported; public metadata contains only a hash of redacted text. Distinct documents that redact to identical text therefore do not replace one another.
- `RETAIN_SOURCE_FILES=true` retains sources for figure/visual tools; `false` deletes them after indexing.
- Retained filesystem paths are stored only in the owner-scoped SQLite document registry (`DOCUMENT_DB_PATH`). They are not written to Chroma metadata, citations, manifests, or API responses.
- Registry reads dynamically validate that a retained path still names a regular non-symlink file inside `UPLOAD_DIR`; missing or invalid files downgrade to text-only capability.
- Ordinary document listing performs only cheap retained-file/PDF eligibility checks. Full visual verification runs on demand.
- Before a retained PDF is returned to a visual tool, the registry re-hashes its current bytes and verifies that owner plus SHA-256 still derives the registered `doc_id`. Host-side mutation therefore makes the source visually unavailable while keeping it retained, protected, and deletable.
- `VISUAL_MAX_PDF_PAGES`, `VISUAL_MAX_RENDER_PIXELS`, `VISUAL_MAX_ENCODED_BYTES`, and `VISUAL_CLIP_HEIGHT_POINTS` fail closed on excessive page count, caption-region geometry, actual rendered pixels, or exact base64 image payload length.
- Re-ingestion registers the new source before deleting the previous retained file.
- `DELETE /docs/{doc_id}` removes vectors, registry state, and the retained file.
- Old unreferenced regular uploads are removed only after a grace period. Active-job, retained-document, recent, and symlink paths are protected. Reconciliation fails closed if either reference store cannot be read.

Figure localization still requires selectable caption text; scanned-caption coordinate localization remains unimplemented.

These checks are not malware analysis or parser sandboxing. Untrusted deployments should add external scanning and isolation.

Retained files are plaintext unless the deployment volume provides encryption at rest. Highly sensitive deployments should set `RETAIN_SOURCE_FILES=false` or use encrypted storage with an explicit retention policy.

## Privacy boundary

Full text, OCR output, sections, titles, filenames, summaries, metadata, and owner-facing job strings pass through masking before indexing or serialization. The metadata sanitizer additionally removes:

- common email, phone, address, payment-card, identity, and IP patterns;
- POSIX, Windows, home-directory, and `file://` paths;
- credentials embedded in URIs;
- common API-key, token, password, and secret query parameters;
- sensitive content in mapping keys as well as values, while preserving colliding sanitized keys with deterministic suffixes.

Every serialized scientific-tool result passes through this recursive sanitizer. Keys are length-bounded before JSON serialization.

Masking uses regular expressions and a Luhn check for several common identifiers. It can miss identifiers and occasionally mask benign text. It is not certified de-identification. For regulated or highly sensitive data, use a dedicated data-loss-prevention pipeline, encryption at rest, retention controls, audit access, and jurisdiction-appropriate review.

## Durable ingestion boundary

The single-host ingestion state machine is:

```text
queued -> processing -> finalizing -> success
                  \---- retry ----/    \-> failed
```

- SQLite stores owner, source path, attempts, state, and `next_attempt_at`.
- A worker atomically claims only a due `queued` row and increments its attempt count.
- Retry deadlines use bounded exponential backoff and survive restart.
- One lazily started heap/condition scheduler thread manages all delayed jobs; there is no `threading.Timer` per job.
- `INGEST_MAX_PENDING` bounds running plus executor-queued ingestion futures. When saturated, durable jobs remain `queued` and receive a short scheduler admission retry instead of entering an unbounded in-memory queue.
- Startup reconciliation reschedules interrupted jobs, fails invalid/exhausted jobs, and promotes already-registered `finalizing` jobs without re-indexing.
- A duplicate worker that loses the atomic claim does not own source cleanup.
- Failed jobs clear document IDs that never committed.
- Internal provider, database, credential, and filesystem exception details are not copied verbatim into public job status.

This is crash recovery for one shared host/filesystem/database. It is not a distributed exactly-once queue. Multi-host deployments require a dedicated queue, shared transactional database, worker leases, and idempotency controls.

## Vector-store boundary

- Every query, list, replace, and delete includes the server-owned tenant ID.
- Deterministic chunk IDs make normal re-ingestion idempotent.
- Replacement captures the previous chunks before writing the new generation.
- If any upsert or stale-delete batch fails, the vector layer removes new-only chunks and restores the previous generation. Incomplete compensation becomes an explicit error.
- Total retrieval backend failure raises an unavailable error rather than being represented as an empty evidence result.
- Document listing paginates chunks until it reaches the requested number of distinct documents or a configured scan ceiling.

Chroma is not a cross-database transaction participant with SQLite. Compensating restoration substantially reduces mixed-generation states but cannot provide a formal distributed transaction guarantee.

## OCR boundary

OCR is disabled by default. When enabled:

- only low-native-text pages are OCR candidates;
- `OCR_MAX_PAGES`, `OCR_DPI`, `OCR_TIMEOUT_SECONDS`, and `MAX_PDF_RENDER_PIXELS` bound cost;
- the limit counts actual OCR attempts, not absolute page positions;
- page-local timeout/failure does not discard other usable pages;
- attempted, successful, empty, failed, and limit-skipped pages are recorded;
- OCR text enters the same masking and indexing pipeline as native text;
- OCR output is untrusted extracted data.

Tesseract is an external executable and should be patched and constrained like any other parser. OCR is not a security sanitizer.

## Remote fetch security

The crawler, direct-page tool, and configured Serper provider use the shared downloader. It:

- allows only GET, HEAD, and POST over HTTP or HTTPS;
- rejects embedded credentials, invalid ports, fragments, caller-controlled `Host`, `Content-Length`, transfer framing, proxy, and hop-by-hop headers;
- rejects CR/LF in caller-supplied headers;
- resolves and rejects private, loopback, link-local, multicast, reserved, unspecified, localhost, and metadata-network destinations;
- disables environment-proxy inheritance, including for injected sessions;
- verifies the actual connected socket peer address after connection, closing the pre-resolution DNS-rebinding time-of-check/time-of-use gap;
- revalidates every redirect;
- strips authorization, cookie, API-key, and token headers on cross-origin redirects;
- refuses cross-origin 307/308 replay of POST bodies and converts 301/302/303 POST redirects to bodyless GET;
- closes every response on success or failure;
- enforces response-byte and end-to-end wall-clock limits across redirects and streamed chunks.

DNS resolution is still an operating-system call that cannot be forcibly cancelled safely from a Python thread. Production deployments should pair application checks with an egress firewall, blocked metadata routes, controlled DNS, and provider allowlists where appropriate.

## Agent, tool, and prompt-injection boundary

Documents, webpages, snippets, figures, OCR text, and tool outputs are evidence data, not instructions. Tenant scope and authoritative citation objects are enforced by server code rather than model text.

- Tool arguments are parsed as bounded JSON objects and validated against the declared schemas at runtime.
- Tool results, citations, evidence count, model output tokens, and serialized final answers are bounded.
- `MAX_CONCURRENT_TOOL_WORKERS` bounds running tool threads and `MAX_PENDING_TOOL_TASKS` bounds running plus queued tool futures process-wide.
- When tool capacity is saturated, calls fail closed as unavailable rather than accumulating in an unbounded executor queue.
- A tool deadline returns without waiting for unfinished work. A timed-out running tool retains its admission slot until it actually finishes.
- Raw tool exception text is not placed in model context or user warnings.
- The model returns answer prose only. The server registers, deduplicates, relabels, and selects actual evidence objects.
- Unsupported citation labels are diagnosed structurally.

A model can still produce incorrect prose. Users must inspect evidence for semantic support; structural citation provenance is not an entailment proof.

## Browser security

The browser application:

- uses no third-party JavaScript or font CDN;
- does not assign untrusted data to `innerHTML`;
- renders a constrained subset of Markdown through DOM nodes;
- validates external link protocols;
- stores API keys and conversation history in `sessionStorage`, not persistent local storage;
- never sends an owner header;
- represents `queued`, `processing`, `finalizing`, `success`, and `failed` states;
- distinguishes retained PDF eligibility from verification performed when a visual action is invoked.

Deploy behind HTTPS. Add HSTS and environment-specific ingress controls at the reverse proxy.

## API and operational boundaries

- Request IDs are reflected only when they match a strict bounded identifier pattern; otherwise the server generates a new ID.
- Model names, job IDs, document IDs, response fields, warnings, citations, and metadata strings are bounded.
- The container readiness probe verifies HTTP liveness, both SQLite registries, and create/fsync/delete access to upload and vector volumes without initializing the embedding model.
- Telemetry stores query SHA-256/length and owner SHA-256, not raw query text or plaintext owner ID.
- Telemetry events are recursively bounded, JSONL files rotate at a configured size, and logging failure never fails a user request.

The readiness probe does not prove that the embedding model can download or that every Chroma query will succeed. Operational monitoring should exercise representative retrieval separately.

## Reporting vulnerabilities

Do not include secrets or private documents in a public issue. Report the affected commit, reproduction steps using synthetic data, impact, and expected boundary.
