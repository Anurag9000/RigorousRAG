# Wave 5 status addendum — custody signer public-key registry

Last updated: 2026-08-03

## Implemented

### Public signer registry

- [x] Owner-scoped Ed25519 public signer records.
- [x] Explicit key ID and issuer.
- [x] SHA-256 fingerprint of the raw 32-byte Ed25519 public key.
- [x] Active and retired states.
- [x] Multiple active keys for deliberate rotation overlap.
- [x] Unique owner/key-ID and owner/fingerprint constraints.
- [x] Idempotent exact registration.
- [x] Key-ID and fingerprint collision refusal.
- [x] Monotonic exact-confirmation retirement.
- [x] Immutable registration and retirement actor-binding provenance.
- [x] Deterministic record digest and database-row tamper refusal.
- [x] No private-key bytes, public-key bytes, PEM files, or paths in the registry.
- [x] No key deletion or automatic rotation command.

### Governed signing and verification

- [x] New signatures require an active registry record.
- [x] Manifest owner must equal registry owner.
- [x] Supplied private key must derive the registered public fingerprint.
- [x] Registered key ID is used for the signed envelope.
- [x] Retired keys cannot create new governed signatures.
- [x] Active or retired records may verify historical signatures.
- [x] Public key, envelope fingerprint, key ID, and owner must all agree.
- [x] Query-only status/list boundary.
- [x] Actor IDs, key paths, and private material omitted from summaries.

### Signer-administration replay boundary

The canonical administration entrypoint is:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers_governed.py ...
```

It permits only these process-owned actor-binding methods:

```text
process_environment
descriptor_file
```

Every other actor-binding method fails closed, including signed/HMAC assertions, OIDC assertions, command-line identities, and unknown future methods.

This restriction prevents one short-lived signed reviewer assertion from being replayed across multiple signer registrations or retirements. Signed signer administration remains disabled until it receives a dedicated durable one-operation reservation journal.

The older non-suffixed signer script remains a compatibility surface and must not be used for new signer administration. New operator documentation and deployment procedures must use the governed entrypoint.

## Focused verification

A reconstructed signer-registry core passed:

```text
3 focused checks passed
```

Covered:

- idempotent registration and unique fingerprints;
- multiple active-key overlap;
- monotonic retirement and replay rules;
- record/database tamper refusal.

Three repository-native actor-boundary contracts are committed for:

- acceptance of direct process environment/file actors;
- refusal of every non-direct and future actor method;
- canonical boundary installation before command delegation.

The complete exact-current repository suite has not been executed.

## Still open

- [ ] Dedicated one-operation reservation for signed signer-administration assertions.
- [ ] Trusted timestamp authority integration.
- [ ] Issuer/key overlap-window policy enforcement.
- [ ] Hardware-backed private-key custody.
- [ ] External public-key transparency publication.
- [ ] Independent-process registration/retirement contention.
- [ ] Complete exact-current pytest, coverage, Ruff, Windows, and container matrices.

## Permanent non-claims

- Registry activation proves governance state, not real-world identity by itself.
- Retirement prevents new governed signatures but cannot timestamp old signatures.
- Historical verification does not prove a signature predates retirement.
- Software file permissions are not hardware-backed key custody.
- Focused reconstructed tests are not the complete release matrix.
- Release readiness is not claimed.
