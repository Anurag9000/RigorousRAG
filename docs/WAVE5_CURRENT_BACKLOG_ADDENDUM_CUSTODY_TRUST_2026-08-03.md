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

## Remaining trusted-time and hardware work

- [ ] Integrate RFC 3161 timestamp request/response verification where required.
- [ ] Integrate an externally trusted timestamp service or governed institutional time source.
- [ ] Record and verify trusted timestamp-service certificate chains and revocation status.
- [ ] Add hardware-backed authority signing through HSM/KMS/PKCS#11.
- [ ] Add hardware-backed or independently attested clock evidence.
- [ ] Add key-rotation overlap across external timestamp authority certificate chains.
- [ ] Add independent-process duplicate-serial, hold-placement, and output-path contention tests.
- [ ] Add process-kill, filesystem-full, fsync, SQLite busy/locked, and key-access fault injection.
- [ ] Add destructive-retention authorization, deletion journal, and secure compaction policy.
- [ ] Run exact-current complete pytest, coverage, Ruff, Windows, and container matrices.

## Exact terminology

The implemented timestamp receipt is an **Ed25519 authority attestation of an asserted time**.

It is not represented as:

- an RFC 3161 timestamp token;
- a proof that the authority clock was accurate;
- a hardware-clock attestation;
- a scientific-correctness guarantee;
- independent proof of institutional identity.

Its evidence value comes from exact custody binding, Ed25519 verification, governed public-key registration, registration/retirement chronology, one-serial durable publication recovery, retention audit, and integrity-backed legal holds.

## Permanent non-claims

- Public-key signatures prove possession of a matching private key, not correctness of the underlying evidence.
- Key registries require governed out-of-band public-key distribution to establish external identity.
- Historical verification does not independently prove signing or asserted wall-clock time beyond the recorded cryptographic scope.
- Rotation reports are planning information, not mutation authorization.
- Legal holds and retention candidates are not deletion authorization.
- Durable issuance recovery is not distributed consensus.
- Focused reconstructed checks are not a complete release matrix.
- Release readiness is not claimed.
