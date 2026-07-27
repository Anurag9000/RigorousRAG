# Security Model

## Deployment modes

### Single-user mode

When neither `API_KEY_OWNERS_JSON` nor `ALLOWED_API_KEYS` is set, every request is assigned the server-controlled `SINGLE_USER_OWNER_ID`. This mode is suitable only for a trusted local workstation or an otherwise isolated service.

### Authenticated multi-user mode

Set `API_KEY_OWNERS_JSON` to a JSON object mapping opaque API keys to owner IDs:

```json
{"key-for-alice":"alice","key-for-lab-b":"lab-b"}
```

The server derives tenant identity from this mapping. `X-Owner-ID` is intentionally ignored. API keys should be generated randomly, distributed through a secret manager, rotated, and never committed.

`ALLOWED_API_KEYS` remains a compatibility option. Each key receives a distinct server-derived owner ID based on a one-way digest; explicit mappings are preferred.

## Upload security

- Maximum bytes are controlled by `MAX_UPLOAD_BYTES`.
- Only `.pdf`, `.docx`, `.md`, and `.txt` are accepted.
- PDF and DOCX content signatures are verified.
- Uploaded names are display metadata only; storage names are random and owner-scoped.
- Text files containing NUL bytes are rejected as binary.
- Originals are deleted after indexing unless `RETAIN_UPLOADS=true`.
- Parsing is not malware analysis. Untrusted deployments should add an external scanning/sandbox layer.

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

Deploy behind HTTPS. Add platform-specific CSP, HSTS, framing, and referrer headers at the reverse proxy or ingress layer.

## Model and prompt-injection boundary

Documents, webpages, snippets, figures, and tool outputs are evidence data, not instructions. The system prompt states this explicitly. More importantly, tenant scope and authoritative citation objects are enforced by server code rather than model text.

A model can still produce incorrect prose. The server guarantees only that returned citation objects came from actual tool results and that labels map structurally. Users must inspect evidence for semantic support.

## Privacy limits

Masking uses regular expressions and a Luhn check for several common identifiers. It can miss identifiers and can occasionally mask benign numeric text. It is not a de-identification certification.

For regulated or highly sensitive data, use a dedicated data-loss-prevention pipeline, encryption at rest, retention controls, audit access, and jurisdiction-appropriate review.

## Telemetry

Agent telemetry stores query length and SHA-256, not raw query text. It remains possible to correlate repeated identical queries. Disable or relocate telemetry when this is unacceptable.

## Reporting vulnerabilities

Do not include secrets or private documents in a public issue. Report the affected commit, reproduction steps using synthetic data, impact, and expected boundary.
