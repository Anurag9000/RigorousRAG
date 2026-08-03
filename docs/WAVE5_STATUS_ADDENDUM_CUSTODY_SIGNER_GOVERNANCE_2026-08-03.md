# Wave 5 status addendum — custody signer governance

Last updated: 2026-08-03

This addendum supersedes narrower signer-status notes and records the current complete signer-governance boundary.

## Implemented

### Signature capability

- [x] Ed25519 public-key envelopes over complete external custody manifests.
- [x] Explicit key ID and raw-public-key SHA-256 fingerprint.
- [x] Protected PEM private-key loading.
- [x] Atomic no-overwrite signed-envelope publication.
- [x] Offline public-key verification and fingerprint pinning.
- [x] Shared-secret HMAC envelope retained as a distinct lower-assurance option.

### Public signer registry

- [x] Owner/key/issuer/public-fingerprint records only.
- [x] Active and retired states.
- [x] Multiple active keys for overlap.
- [x] Unique owner/key-ID and owner/fingerprint constraints.
- [x] Exact idempotent registration.
- [x] Monotonic retirement with immutable actor-binding provenance.
- [x] Retired-key historical verification.
- [x] Active-only governed signing.
- [x] Query-only status/list.
- [x] No private-key storage or key deletion.

### Direct signer administration

Canonical entrypoint:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signers_governed.py ...
```

- [x] Process environment and descriptor-file actor bindings only.
- [x] Every signed, command-line, OIDC, and unknown method refused.
- [x] Prevents replay of a signed reviewer credential across signer changes.

### Signed one-operation administration

Canonical entrypoint:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py ...
```

- [x] Deterministic owner/action/key/action-digest use identity.
- [x] Issuer and expiry provenance required.
- [x] Reservation before registry mutation.
- [x] One binding digest may authorize only one action.
- [x] Exact `reserved -> committed` transition.
- [x] Crash recovery before or after registry mutation.
- [x] Retroactive attachment to an unreserved existing action refused.
- [x] Direct and command-line identities refused on the signed path.
- [x] Query-only use status.
- [x] No assertion body/signature or actor ID returned.

### Rotation audit

- [x] Explicit deterministic rotation policy.
- [x] Allowed issuer enforcement.
- [x] Maximum active-key count.
- [x] Maximum age and warning window.
- [x] Minimum successor overlap.
- [x] Missing-active-key global action without synthetic key records.
- [x] Expired/due/current/unapproved/retired classifications.
- [x] Register-successor, maintain-overlap, retirement-eligible, count-reduction, and issuer-investigation actions.
- [x] Query-only report with no automatic mutation.

## Focused execution evidence

Executed reconstructed evidence remains separately bounded:

```text
3/3 Ed25519 signature checks passed
3/3 signer-registry core checks passed
```

Repository-native contracts are additionally committed for direct actor restriction, signer CLI behavior, signed one-operation reservations/recovery, query-only status, and rotation assessment. Those newest contracts have not been executed together from a complete unchanged current checkout.

## Compatibility boundaries

The historical signer and signed-admin scripts remain compatibility surfaces. New deployment procedures must use the `*_governed.py` entrypoints.

Compatibility surfaces do not override the canonical governance requirements documented here.

## Still open

- [ ] Complete exact-current repository pytest, coverage, Ruff, and full-tree compilation.
- [ ] Execute all signer contracts together against the current head.
- [ ] Independent-process registration, retirement, and assertion-use contention.
- [ ] Process-kill and SQLite I/O/disk-full injection across reservation and registry commits.
- [ ] Trusted timestamp authority integration.
- [ ] Issuer/key overlap-window enforcement as a durable organization policy.
- [ ] Hardware-backed private-key custody.
- [ ] External public-key transparency publication and monitoring.
- [ ] Windows and container signer matrices.

## Permanent non-claims

- Ed25519 proves possession of a private key, not scientific correctness.
- Registry state does not establish real-world identity without external governance.
- Retirement cannot prove whether an existing signature predates retirement without trusted time.
- Signed administration proves one credential/action binding, not multi-party approval.
- Software key-file permissions are not HSM custody.
- Focused reconstructed checks are not the complete release matrix.
- Release readiness is not claimed.
