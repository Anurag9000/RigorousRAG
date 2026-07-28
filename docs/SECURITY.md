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

## Upload, parsing, retention, and document identity

- Maximum source bytes are controlled by `MAX_UPLOAD_BYTES`.
- Only `.pdf`, `.docx`, `.md`, and `.txt` are accepted.
- Uploaded names are display metadata only; storage names are random and owner-scoped.
- Uploads are streamed, flushed, and `fsync`ed before the durable job row is created.
- Symlinked source files are rejected.
- PDF and DOCX signatures are verified; text files containing NUL bytes are rejected as binary.
- DOCX packages are checked for required members, duplicate names, unsafe paths, symlinks, encryption, member count, total expansion, and compression ratio.
- PDF page count, total extracted characters, OCR attempts, OCR render pixels, and vector chunk count are bounded.
- The source is checked for size, timestamp, and inode changes during parsing.
- Stable document identity is derived from owner ID plus source-file SHA-256. The source hash is not exported; public metadata contains only a hash of redacted text. This prevents distinct documents that redact to identical text from replacing one another.
- `RETAIN_SOURCE_FILES=true` retains sources for figure/visual tools; `false` deletes them after indexing.
- Retained filesystem paths are stored only in the owner-scoped SQLite document registry (`DOCUMENT_DB_PATH`). They are not written to Chroma metadata, citations, manifests, or API responses.
- Registry reads dynamically validate that a retained path still names a regular non-symlink file inside `UPLOAD_DIR`; missing or invalid files downgrade to text-only capability.
- Re-ingestion registers the new source before deleting the previous retained file.
- `DELETE /docs/{doc_id}` removes vectors, registry state, and the retained file.
- Old unreferenced regular uploads are removed only after a grace period. Active-job, retained-document, recent, and symlink paths are protected. Reconciliation fails closed if either reference store cannot be read.

These checks are not malware analysis or parser sandboxing. Untrusted deployments should add external scanning and isolation.

Retained files are plaintext unless the deployment volume provides encryption at rest. Highly sensitive deployments should set `RETAIN_SOURCE_FILES=false` or use encrypted storage with an explicit retention policy.

## Durable ingestion boundary

The single-host ingestion state machine is:

```text
queued -> processing -> finalizing -> success
                  \---- retry ----/    \-> failed
```

- SQLite stores owner, source path, attempts, state, and `next_attempt_at`.
- A worker atomically claims only a due `queued` row and increments its attempt count.
- Delayed retries wait in a deduplicated daemon scheduler rather than occupying ingestion workers.
- Retry deadlines use bounded exponential backoff and survive restart.
- Startup reconciliation reschedules interrupted jobs, fails invalid/exhausted jobs, and promotes already-registered `finalizing` jobs without re-indexing.
- A duplicate worker that loses the atomic claim does not own source cleanup.
- Internal provider, database, and filesystem exception messages are not copied verbatim into public job status.

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
- One process-wide tool executor limits live tool threads. A request timeout returns without waiting for unfinished work; Python cannot safely kill a running third-party thread, so provider/network deadlines remain essential.
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
- enables figure actions only when the server reports an actually available retained source.

Deploy behind HTTPS. Add HSTS and environment-specific ingress controls at the reverse proxy.

## API and operational boundaries

- Request IDs are reflected only when they match a strict bounded identifier pattern; otherwise the server generates a new ID.
- Model names, job IDs, and document IDs are length-bounded at the request boundary.
- The container readiness probe verifies HTTP liveness, both SQLite registries, and create/fsync/delete access to upload and vector volumes without initializing the embedding model.
- Telemetry stores query SHA-256/length and owner SHA-256, not raw query text or plaintext owner ID.
- Telemetry events are recursively bounded, JSONL files rotate at a configured size, and logging failure never fails a user request.

The readiness probe does not prove that the embedding model can download or that every Chroma query will succeed. Operational monitoring should exercise representative retrieval separately.

## Privacy limits

Masking uses regular expressions and a Luhn check for several common identifiers. It can miss identifiers and occasionally mask benign text. It is not certified de-identification.

For regulated or highly sensitive data, use a dedicated data-loss-prevention pipeline, encryption at rest, retention controls, audit access, and jurisdiction-appropriate review.

## Reporting vulnerabilities

Do not include secrets or private documents in a public issue. Report the affected commit, reproduction steps using synthetic data, impact, and expected boundary.
