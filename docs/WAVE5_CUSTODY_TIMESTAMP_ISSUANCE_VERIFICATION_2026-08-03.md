# Wave 5 custody timestamp issuance verification boundary

Last updated: 2026-08-03

This ledger supersedes the issuance-related open items in `WAVE5_CUSTODY_TIMESTAMP_VERIFICATION_BOUNDARY_2026-08-03.md`.

## Implemented issuance path

### One-serial durable issuance

- [x] Deterministic issuance identity over owner, authority, key, serial, and output-path digest.
- [x] Unique serial reservation per owner/authority/key.
- [x] Exact signed public attestation persisted before output publication.
- [x] Private authority key used only during seed and never stored.
- [x] Expiring leases, reclaim, attempt ceilings, retry, and pre-output cancellation.
- [x] Atomic no-overwrite output publication.
- [x] Exact existing-output replay after a crash.
- [x] Divergent output and missing previously published output refusal.
- [x] Recovery after output creation before phase persistence.
- [x] Recovery after `output_published` before completion.
- [x] Retired-key historical completion only within registration/retirement chronology.
- [x] Query-only status and listing.
- [x] Dedicated non-aliasing database runtime.

### Operations and retention

- [x] Planned, active-running, expired-reclaimable, expired-exhausted, retryable-failed, exhausted-failed, completed, and cancelled classifications.
- [x] Deterministic audit and retention digests.
- [x] Bounded-completeness and duplicate-ID refusal.
- [x] Nonterminal and failed history excluded from retention candidates.
- [x] Latest terminal issuance protected per authority/key.
- [x] Completed issuances retained by default.
- [x] No retry, cancellation, signing, publication, deletion, or compaction verb in the operations CLI.

### Durable issuance legal holds

- [x] Deterministic owner/issuance/hold-key identity.
- [x] Referenced issuance and owner validation before placement.
- [x] Process-owned or signed actor provenance.
- [x] Actor-expiry refusal.
- [x] Complete active/released record digest.
- [x] Monotonic active-to-released history.
- [x] Exact release confirmation before actor/store resolution and inside the transaction.
- [x] Query-only status/list and active issuance-ID discovery.
- [x] Automatic durable active-hold integration into retention plans.
- [x] No deletion capability.

### Contention and fault containment

- [x] Independent-process exact serial reservation replay.
- [x] Independent-process same-serial/different-output exclusion.
- [x] Independent-process same-output publication with exactly one winner.
- [x] Independent-process exact legal-hold placement replay.
- [x] Independent-process conflicting legal-hold scope refusal.
- [x] Abrupt process death after output creation before phase persistence.
- [x] Abrupt process death after output-phase persistence before completion.
- [x] Lease-expiry reclaim and recovery after both abrupt-death windows.
- [x] Controlled atomic-publication `OSError` with no partial output.
- [x] Real zero-wait SQLite lock refusal with no partial issuance row.
- [x] Real zero-wait SQLite lock refusal with no partial hold row.
- [x] Missing private-key failure before intent or output creation.

## Executed reconstructed evidence

The focused reconstructed workspace contains the exact newly implemented timestamp authority, issuance, operations, holds, contention, and fault modules. Older custody dependencies are represented by API-faithful local implementations used throughout the timestamp work.

Executed command family:

```text
PYTHONPATH=.:tests python -m compileall -q tools tests scripts
PYTHONPATH=.:tests python -m pytest -q
```

Result:

```text
38 passed
```

Composition:

- 13 custody timestamp authority, registry, CLI, and rotation checks;
- 8 one-serial issuance and operator checks;
- 4 issuance operational audit and retention checks;
- 4 issuance legal-hold and retention-integration checks;
- 3 independent-process contention checks;
- 3 abrupt-process and atomic-publication failure checks;
- 3 SQLite-lock and private-key-access failure checks.

Focused compilation passed.

Python 3.13 emitted 12 advisory warnings about `fork()` from a multithreaded pytest process. All child processes exited with the expected codes and every contract passed. Windows `spawn` behavior is not inferred from these results.

## Committed repository-native contracts

New test files committed for this issuance phase:

- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance.py` — 4 functions;
- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_cli.py` — 4 functions;
- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_operations.py` — 4 functions;
- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds.py` — 4 functions;
- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contention.py` — 3 functions;
- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_process_kill.py` — 3 parameterized/function contracts;
- `test_evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_storage_faults.py` — 3 functions.

The complete custody timestamp family also includes the earlier 13 authority/rotation repository-native contracts.

These tests have not been executed together from a complete unchanged checkout of the current repository.

## Exact-current verification limitation

A clean clone of the live repository was attempted, but the execution container could not resolve `github.com`. This prevented reconstructing a complete exact-current source tree for full repository pytest, coverage, and Ruff.

This is a source-acquisition limitation, not evidence that the full suite passed or failed.

## Still open

- Windows `spawn` multiprocessing behavior;
- POSIX `spawn` and `forkserver` behavior;
- multi-container and multi-host contention;
- production 30-second SQLite busy-timeout expiry;
- WAL corruption and recovery;
- `SQLITE_IOERR`, `SQLITE_FULL`, and journal-mode fault injection;
- real filesystem quota exhaustion;
- output-directory `fsync` failure;
- non-root private-key permission denial;
- HSM/KMS/PKCS#11 access, timeout, and failover;
- destructive-retention authorization and deletion journal;
- secure deletion and database compaction policy;
- RFC 3161 and external trusted-time integration;
- complete exact-current pytest, coverage, Ruff, Windows, Docker, and live package-import matrices.

## Non-claims

- No complete exact-current pytest or CI success is claimed.
- POSIX process tests do not prove Windows or distributed behavior.
- Controlled storage faults are not full disk-failure qualification.
- The timestamp receipt remains a custom Ed25519 authority attestation, not an RFC 3161 token.
- The attestation does not prove hardware-clock accuracy.
- Legal holds and retention plans are not deletion authorization.
- Durable journal recovery is not distributed consensus.
- Release readiness is not claimed.
