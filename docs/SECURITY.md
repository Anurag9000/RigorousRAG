# Security Model

## Deployment modes

### Single-user mode

When neither `API_KEY_OWNERS_JSON` nor `ALLOWED_API_KEYS` is set, every request is assigned the server-controlled `SINGLE_USER_OWNER_ID`. This mode is suitable only for a trusted local workstation or otherwise isolated service.

### Authenticated multi-user mode

Set `API_KEY_OWNERS_JSON` to a JSON object mapping opaque API keys to owner IDs:

```json
{"key-for-alice":"alice","key-for-lab-b":"lab-b"}
```

The server derives tenant identity from this mapping. `X-Owner-ID` is intentionally ignored. API keys should be generated randomly, distributed through a secret manager, rotated, and never committed.

`ALLOWED_API_KEYS` remains a compatibility option. Each key receives a distinct server-derived owner ID based on a one-way digest; explicit mappings are preferred.

## Upload, retention, and document registry

- Maximum bytes are controlled by `MAX_UPLOAD_BYTES`.
- Only `.pdf`, `.docx`, `.md`, and `.txt` are accepted.
- PDF and DOCX content signatures are verified.
- Uploaded names are display metadata only; storage names are random and owner-scoped.
- Text files containing NUL bytes are rejected as binary.
- `RETAIN_SOURCE_FILES=true` retains sources for figure/visual tools; `false` deletes them after indexing.
- Retained filesystem paths are stored only in the owner-scoped SQLite document registry (`DOCUMENT_DB_PATH`). They are not written to Chroma metadata, citations, manifests, or API responses.
- Re-ingesting the same owner/content identity registers the new source before deleting the previous retained file.
- `DELETE /docs/{doc_id}` removes vectors, registry state, and the retained file.
- The registry rejects paths outside `UPLOAD_DIR` and revalidates existence/ownership at lookup time.
- Parsing and extension checks are not malware analysis. Untrusted deployments should add an external scanning and sandbox layer.

Retained files are plaintext unless the deployment volume provides encryption at rest. Highly sensitive deployments should set `RETAIN_SOURCE_FILES=false` or use encrypted storage with an explicit retention policy.

## Durable ingestion boundary

Uploads are written before a durable SQLite job record is created. Jobs enter `queued`, and workers atomically claim them before changing to `processing`. Only one thread/process can claim a queued job. On restart, queued/interrupted jobs are reconciled against their owner-scoped source path and retry count. Missing, out-of-root, or exhausted jobs fail closed.

This is crash recovery for a single shared filesystem/database. It is not a distributed exactly-once queue. Multi-host deployments require a dedicated queue, shared transactional database, worker leases, and idempotency controls.

## OCR boundary

OCR is disabled by default. When enabled:

- only low-native-text pages are OCR candidates;
- `OCR_MAX_PAGES`, `OCR_DPI`, and `OCR_TIMEOUT_SECONDS` bound cost;
- OCR text enters the same masking and indexing pipeline as native text;
- OCR output is untrusted extracted data;
- scan quality, page language, orientation, and layout can cause omissions or substitutions.

Tesseract is an external executable and should be patched and constrained like any other parser. OCR is not a security sanitizer.

## Remote fetch security

Public-page fetches:

- allow only HTTP and HTTPS;
- reject embedded credentials;
- resolve DNS before each request;
- reject private, loopback, link-local, multicast, reserved, unspecified, and localhost destinations;
- disable environment-proxy inheritance for owned sessions;
- revalidate every redirect;
- stream responses under a byte limit;
- enforce timeouts and redirect limits.

DNS rebinding cannot be eliminated solely at application level. Production networks should also apply egress firewall policy and block cloud metadata routes.

## Browser security

The browser application:

- uses no third-party JavaScript or font CDN;
- does not assign untrusted data to `innerHTML`;
- renders a constrained subset of Markdown through DOM nodes;
- validates external link protocols;
- stores API keys and conversation history in `sessionStorage`, not persistent local storage;
- never sends an owner header.

Deploy behind HTTPS. Add HSTS and environment-specific ingress controls at the reverse proxy.

## Model and prompt-injection boundary

Documents, webpages, snippets, figures, OCR text, and tool outputs are evidence data, not instructions. The system prompt states this explicitly. Tenant scope and authoritative citation objects are enforced by server code rather than model text.

A model can still produce incorrect prose. The server guarantees only that returned citation objects came from actual tool results and that labels map structurally. Users must inspect evidence for semantic support.

## Privacy limits

Masking uses regular expressions and a Luhn check for several common identifiers. It can miss identifiers and can occasionally mask benign text. It is not a de-identification certification.

For regulated or highly sensitive data, use a dedicated data-loss-prevention pipeline, encryption at rest, retention controls, audit access, and jurisdiction-appropriate review.

## Telemetry

Agent telemetry stores query length and SHA-256, not raw query text. It remains possible to correlate repeated identical queries. Disable or relocate telemetry when this is unacceptable.

## Reporting vulnerabilities

Do not include secrets or private documents in a public issue. Report the affected commit, reproduction steps using synthetic data, impact, and expected boundary.
