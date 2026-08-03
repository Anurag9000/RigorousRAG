# Wave 5 custody timestamp verification boundary

Last updated: 2026-08-03

This ledger records the exact evidence boundary for the custody timestamp-authority attestation, authority registry, operator path, and rotation assessment.

## Implemented

### Authority attestation

- [x] Ed25519 signature over canonical timestamp scope.
- [x] Exact signed custody-envelope SHA-256 binding.
- [x] Custody-manifest and chain-digest binding.
- [x] Owner, authority ID, key ID, and public fingerprint binding.
- [x] Asserted timestamp, nonce digest, and deterministic serial binding.
- [x] Custody-manifest chronology requirement.
- [x] Configurable future-time verification ceiling.
- [x] Atomic no-overwrite attestation publication.
- [x] Explicit `rfc3161_token: false` and `hardware_clock_proven: false` flags.

### Authority-key governance

- [x] Durable owner/authority/key-scoped public-key registry.
- [x] Process-owned or signed actor provenance on registration and retirement.
- [x] Exact idempotent registration replay preserving the original timestamp.
- [x] Per-owner public-fingerprint uniqueness.
- [x] Monotonic active-to-retired lifecycle.
- [x] Active-only new attestation issuance.
- [x] Historical verification only inside the registered active interval.
- [x] Record-digest, database-payload, path-redirection, and file-identity defenses.
- [x] Query-only registry for inspection, issuance checks, and historical verification.
- [x] Dedicated database runtime with cross-journal alias refusal.

### Operator surface

- [x] Exact fingerprint confirmation before actor/store access.
- [x] Exact retirement key-ID confirmation.
- [x] Register, retire, status, list, issue-governed, and verify-governed commands.
- [x] Actor-ID, raw-path, and private-key-free summaries.
- [x] Generic fail-closed operator errors.
- [x] Dedicated configuration and runbook.

### Rotation assessment

- [x] Initial-key gap classification.
- [x] Healthy single-active-key classification.
- [x] Aged-key/no-successor classification.
- [x] Active overlap-window classification.
- [x] Oldest-key retirement-readiness classification.
- [x] Excessive active-key count classification.
- [x] Explicit maximum age, minimum overlap, and active-key ceiling policy.
- [x] Deterministic policy and report digests.
- [x] Query-only path-free CLI.
- [x] No registration, retirement, signing, or deletion command.

## Executed reconstructed evidence

The focused workspace used the exact new timestamp modules and API-faithful stubs only for older repository dependencies.

Executed command family:

```text
PYTHONPATH=. python -m compileall -q tools tests scripts
PYTHONPATH=. python -m pytest -q
```

Result:

```text
13 passed
```

The checks cover:

1. authority-attestation round trip;
2. atomic no-overwrite output;
3. exact custody-envelope, manifest, and chain binding;
4. wrong-key, signature/field tamper, future-time, and chronology refusal;
5. deterministic nonce digest and serial;
6. exact authority registration replay and collision refusal;
7. active issuance and retired historical verification;
8. database payload and file-identity tamper refusal;
9. query-only write refusal;
10. runtime path/hard-link alias refusal and caching;
11. confirmation before actor/store resolution;
12. privacy-safe end-to-end operator behavior;
13. every rotation classification, report reconstruction, bounds, chronology, and query-only CLI behavior.

This is not a complete unchanged checkout of the current repository.

## Committed repository-native contracts

- timestamp attestation and authority registry: 4 test functions;
- query-only runtime and operator boundaries: 4 test functions;
- rotation assessment and CLI: 5 test functions.

Total:

```text
13 contracts
```

These contracts have not been executed together from a complete exact-current checkout.

## Exact-current verification still required

- complete repository pytest and coverage;
- Ruff and full-tree compilation;
- production package-import and CLI smoke tests;
- exact `cryptography==49.0.0` environment verification;
- independent-process registration/retirement/issuance contention;
- process-kill recovery around registry and output publication;
- duplicate serial and output-path races;
- SQLite busy/locked, WAL, I/O, and disk-full injection;
- private-key access and filesystem-full failures;
- Windows permissions and reparse-point behavior;
- Docker/Compose persistence and restart.

A fresh clone was attempted but the execution container could not resolve `github.com`; this is a source-acquisition limitation, not passing or failing test evidence.

## Remaining trust work

- RFC 3161 integration where required;
- external timestamp-service certificate-chain governance;
- trusted time-source identity and revocation handling;
- HSM/KMS/PKCS#11 authority signing;
- hardware-backed or independently attested clock evidence;
- durable authority-attestation issuance/serial journal and recovery.

## Non-claims

- No full exact-current pytest or CI success is claimed.
- The custom attestation is not an RFC 3161 token.
- The custom attestation does not prove hardware-clock accuracy.
- Registry governance does not independently establish institutional identity.
- Public-key possession does not prove scientific correctness.
- Rotation actions are planning information, not mutation authorization.
- Release readiness is not claimed.
