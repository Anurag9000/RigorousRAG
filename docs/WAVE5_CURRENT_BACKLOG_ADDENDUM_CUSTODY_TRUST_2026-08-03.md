# Wave 5 current-backlog addendum — custody trust

Last updated: 2026-08-03

This addendum supersedes the custody-signature, timestamp, and key-rotation checkboxes in `WAVE5_CURRENT_BACKLOG_2026-08-02.md`. The older file remains historical planning context.

## Completed custody trust layers

- [x] Deterministic external restore chain-of-custody export.
- [x] Protected-key HMAC-SHA256 authentication envelope.
- [x] Publicly verifiable Ed25519 custody-signature envelope.
- [x] Durable owner-scoped custody signer public-key registry.
- [x] Active/retired signer lifecycle with historical signature verification.
- [x] Direct process-owned signer administration.
- [x] One-operation expiring signed signer-administration credentials.
- [x] Crash recovery for signer registration after reservation.
- [x] Signer-governance compliance audit.
- [x] Enforcement-aware signing that refuses noncompliant active keys.
- [x] Signer rotation assessment with overlap and retirement planning.
- [x] Governed Ed25519 custody timestamp-authority attestations.
- [x] Exact custody-envelope, manifest, chain, nonce, and asserted-time binding.
- [x] Durable timestamp-authority public-key registry.
- [x] Active-key issuance and retired-key historical verification windows.
- [x] Query-only timestamp authority inspection and verification.
- [x] Timestamp-authority rotation assessment with explicit policy digest.
- [x] Durable one-serial timestamp issuance journal.
- [x] Exact signed public attestation persistence before output publication.
- [x] Unique serial reservation per owner/authority/key.
- [x] Crash recovery after output creation before phase persistence.
- [x] Crash recovery after output phase before completion.
- [x] Query-only issuance status/list and private-key-free recovery execution.
- [x] Issuance queue-health audit and conservative retention planning.
- [x] Integrity-backed issuance legal holds with monotonic release.
- [x] Automatic active-hold integration into retention plans.
- [x] Independent-process exact serial replay and cross-output serial exclusion.
- [x] Independent-process same-output publication contention with one winner.
- [x] Independent-process exact and conflicting legal-hold placement.
- [x] Abrupt process-death recovery at both timestamp output phases.
- [x] Controlled atomic-publication failure with no partial output.
- [x] Real zero-wait SQLite lock refusal with no partial issuance or hold row.
- [x] Missing private-key failure before intent or output creation.
- [x] Offline RFC 3161 SHA-256 request bundles with nonce and optional policy.
- [x] Atomic no-overwrite DER request emission without implicit network transport.
- [x] Strict granted/rejected `TimeStampResp` parsing and TSTInfo validation.
- [x] Exact message-imprint, nonce, requested-policy, serial and generation-time validation.
- [x] CMS content-type/message-digest and ESSCertID/ESSCertIDv2 signer binding.
- [x] Critical timeStamping-only TSA EKU and signer-certificate validity checks.
- [x] Pinned OpenSSL certificate-chain verification with optional intermediates and CRLs.
- [x] Digest-only RFC 3161 verification receipts with explicit non-claims.
- [x] Real local OpenSSL TSA round-trip, wrong-nonce, wrong-anchor and rejection tests.
- [x] Durable owner-scoped external TSA trust-profile registry.
- [x] Process-owned trust-profile registration and retirement.
- [x] Exact policy, root, intermediate, CRL and signer-fingerprint pinning.
- [x] Explicit token-generation validity windows and retired-profile historical verification.
- [x] Governed profile enforcement before RFC 3161 receipt publication.
- [x] Trust-profile collision, replacement and path-substitution refusal.

## Remaining trusted-time and hardware work

- [ ] Integrate an externally trusted timestamp service or governed institutional time source.
- [ ] Establish governed out-of-band TSA identity vetting and trust-anchor distribution.
- [ ] Add automated rotation assessment and overlap reporting across external TSA profiles.
- [ ] Add live or archived OCSP evidence and broader revocation-policy handling.
- [ ] Add hardware-backed authority signing through HSM/KMS/PKCS#11.
- [ ] Add hardware-backed or independently attested clock evidence.
- [ ] Add governed TSA network transport, endpoint allowlists and credential handling if online submission is required.
- [ ] Add Windows `spawn`, POSIX `spawn`, and multi-container contention matrices.
- [ ] Inject production-timeout SQLite busy expiry, WAL corruption, `SQLITE_IOERR`, and `SQLITE_FULL`.
- [ ] Inject directory-fsync, quota exhaustion, and non-root key-permission failures.
- [ ] Add destructive-retention authorization, deletion journal, and secure compaction policy.
- [ ] Run exact-current complete pytest, coverage, Ruff, Windows, and container matrices.

## Exact terminology

Two distinct timestamp evidence types now exist.

### Governed Ed25519 asserted-time attestation

The repository-owned timestamp authority signs an asserted time and exact custody scope. It is not represented as:

- an RFC 3161 timestamp token;
- a proof that the authority clock was accurate;
- a hardware-clock attestation;
- a scientific-correctness guarantee;
- independent proof of institutional identity.

### RFC 3161 verification receipt

The RFC 3161 boundary verifies an actual nonce-bearing timestamp token against the exact request and supplied certificate evidence. A successful receipt proves request/token binding and certificate-path validation under the supplied trust anchors.

A governed external TSA profile additionally proves that the exact policy, root/intermediate/CRL digests, signer allowlist and token-time window were registered by a process-owned actor. It still does not independently prove:

- that the registered trust anchors belong to the intended institution;
- that the TSA clock was accurate or externally audited;
- that a hardware-backed clock or signing device was used;
- current revocation status when no governed CRL evidence is supplied;
- authenticity of the transport channel outside the signed token.

The receipt therefore retains:

```text
independently_trusted_clock_proven=false
hardware_clock_proven=false
```

The two evidence types are never silently converted or relabeled into each other.

## Permanent non-claims

- Public-key signatures prove possession of a matching private key, not correctness of the underlying evidence.
- Key registries and TSA trust profiles require governed out-of-band distribution to establish external identity.
- Historical verification does not independently prove signing or wall-clock accuracy beyond the recorded cryptographic scope.
- RFC 3161 token and trust-profile validation are not institutional TSA accreditation or hardware-clock proof.
- Rotation reports are planning information, not mutation authorization.
- Legal holds and retention candidates are not deletion authorization.
- Durable issuance recovery is not distributed consensus.
- POSIX `fork` tests do not establish Windows or multi-host behavior.
- Focused reconstructed checks are not a complete release matrix.
- Release readiness is not claimed.
