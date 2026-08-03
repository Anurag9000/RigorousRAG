# Custody timestamp issuance fault and contention matrix

Last updated: 2026-08-03

This document records the committed fault contracts for durable custody timestamp issuance and issuance legal holds.

The contracts exercise actual SQLite files, independent POSIX processes, atomic output publication, abrupt process death, and controlled filesystem/storage failures. They do not claim complete platform or disaster-recovery coverage.

## Independent-process serial reservation

Two processes seed the same authority assertion with the same timestamp, nonce, and output path.

Expected result:

- both return the same deterministic issuance ID;
- the serial is represented by one durable row;
- no duplicate payload or serial row is created.

Two processes then seed the same deterministic authority assertion and serial for different output paths.

Expected result:

- exactly one serial reservation succeeds;
- the other fails because the serial is already reserved;
- no cross-output reuse is permitted.

This verifies both exact idempotence and exclusive serial ownership.

## Independent-process output contention

Two different timestamp attestations are seeded for the same output path. Their serials and issuance IDs differ, but their canonical output-path digest is identical.

Both processes execute concurrently.

Expected result:

- exactly one atomic output publication succeeds;
- the matching issuance completes;
- the competing issuance detects divergent existing output and fails;
- the output file remains a complete canonical attestation;
- no overwrite or merge occurs.

## Independent-process hold placement

Two processes place the same hold identity with the same reason and actor binding.

Expected result:

- both return the same deterministic hold ID;
- one durable hold record exists;
- the original creation time and provenance are preserved.

Two processes place the same hold identity with different reason codes.

Expected result:

- exactly one scope wins;
- the other fails with an identity collision;
- no ambiguous or merged legal-hold record is created.

## Abrupt process death after output creation

A worker terminates with `os._exit` immediately after the attestation output is atomically created but before `output_published` is persisted.

Observed durable state:

```text
state: running
phase: planned
output: present
```

After the lease expires, another worker reclaims the intent, verifies that the existing output bytes exactly equal the persisted attestation, records the missing phase, and completes.

The recovery attempt count increments to two. The authority private key is not required.

## Abrupt process death after phase persistence

A worker terminates after `output_published` is persisted but before final verification and completion.

Observed durable state:

```text
state: running
phase: output_published
output: present
```

After lease expiry, another worker reclaims the intent. Recovery requires the output to remain present and byte-identical. The exact output is verified and the issuance completes.

Missing or divergent output in this phase fails closed.

## Atomic publication failure

The atomic output publisher is forced to raise an `OSError` before publication succeeds.

Expected result:

- no partial output file exists;
- the issuance records `failed` while preserving phase `planned`;
- exact retry is permitted within the attempt ceiling;
- restoring the publisher allows successful completion.

The contract represents a controlled filesystem-publication failure. It does not fully emulate every disk-full or directory-fsync failure mode.

## Real SQLite lock refusal

The issuance database is locked with a real `BEGIN IMMEDIATE` transaction. A zero-wait journal attempts to seed a new issuance.

Expected result:

- SQLite returns `database is locked`;
- the transaction inserts no partial issuance row;
- after the external lock is released, the issuance remains absent.

The same contract is exercised for legal-hold placement:

- SQLite lock refusal is propagated;
- no partial hold row exists after rollback.

Production journals retain their configured busy timeout; the zero-wait subclass is a fault-injection seam used only by the test.

## Private-key access failure

Issuance seed receives a missing authority private-key path.

Expected result:

- key loading fails before signing;
- no issuance intent is created;
- no attestation output is created;
- no private-key material is stored.

## Executed focused evidence

The complete reconstructed custody timestamp suite passed:

```text
38 passed
```

The suite contains:

- 13 authority-attestation, registry, operator, and rotation checks;
- 8 durable one-serial issuance and operator checks;
- 4 issuance operations and retention checks;
- 4 issuance legal-hold and retention-integration checks;
- 3 independent-process contention checks;
- 3 abrupt-process/filesystem-publication checks;
- 3 SQLite-lock and key-access fault checks.

Focused compilation also passed.

The multiprocessing tests use POSIX `fork`. Python 3.13 emitted advisory deprecation warnings because the pytest process is multithreaded; all child processes exited with their expected codes and all assertions passed.

## Still open

- Windows `spawn`-based independent-process testing;
- POSIX `spawn` and `forkserver` testing;
- real multi-host or container-to-container contention;
- SQLite busy-timeout expiry at production timeout values;
- WAL corruption and recovery;
- injected `SQLITE_IOERR`, `SQLITE_FULL`, and journal-mode failures;
- output-directory `fsync` failure;
- full filesystem and quota exhaustion;
- private-key permission denial under non-root container users;
- HSM/KMS/PKCS#11 availability and timeout faults;
- exact-current complete repository pytest, coverage, Ruff, Windows, and Docker matrices.

## Non-claims

- These contracts do not establish distributed consensus.
- A POSIX process test does not prove Windows behavior.
- Controlled `OSError` injection is not complete disk-full testing.
- Successful recovery does not prove the authority clock was accurate.
- Legal holds and retention plans do not authorize deletion.
- Focused reconstructed tests are not a complete unchanged checkout of the current repository.
- Release readiness is not claimed.
