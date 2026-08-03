# Crash-recoverable custody timestamp issuance

Last updated: 2026-08-03

This subsystem strengthens the governed custody timestamp path with a durable one-serial issuance journal.

The compatibility `issue-governed` command signs and publishes in one process. The stronger issuance path separates those responsibilities:

1. **seed** verifies governance, signs once, and durably stores the exact public attestation payload;
2. **execute** publishes that persisted payload atomically without requiring the private key;
3. exact replay recovers process failure after output creation or after phase persistence.

Private authority keys are never stored in the issuance journal.

## Configuration

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH=data/evidence_graph_set_signed_retirement_custody_timestamp_authorities.sqlite3
EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH=data/evidence_graph_set_signed_retirement_custody_timestamp_issuances.sqlite3
```

The issuance database must not equal or hard-link to the authority registry, custody signer databases, custody/artifact/hold databases, restore/retirement journals, or publication journals.

See:

```text
config/evidence_graph_custody_timestamp_authority.env.example
```

## Durable identity

Each issuance ID is deterministic over:

- owner ID;
- timestamp authority ID;
- authority key ID;
- attestation serial;
- canonical output-path digest.

The output path itself is never stored.

Each journal row also binds:

- canonical public attestation digest;
- exact signed attestation JSON;
- state and recovery phase;
- attempt count and ceiling;
- expiring worker lease;
- verification digest;
- generic failure type;
- monotonic timestamps.

The database enforces one serial per owner/authority/key. The same serial cannot be reserved for another output identity.

## Seed

Calculate the canonical output-path digest using the repository helper or an operator preflight, then run:

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuances.py seed \
  --owner-id alice \
  --authority-id institutional-tsa \
  --key-id institutional-tsa-2026-01 \
  --authority-private-key-path institutional-tsa-2026-01.private.pem \
  --signed-envelope-path evidence/restore-chain.ed25519.json \
  --custody-signer-public-key-path custody-signer.public.pem \
  --output-path evidence/restore-chain.timestamp.json \
  --confirm-output-path-digest OUTPUT_PATH_SHA256
```

Seed performs no output publication. It requires:

- exact output-path digest confirmation before store access;
- a query-only initialized authority registry;
- an active exact authority key;
- private-key fingerprint equality with the registry;
- a valid signed custody envelope;
- matching envelope and registry owner;
- asserted time not earlier than custody-manifest generation;
- asserted time not earlier than authority registration;
- an output path that does not already exist.

Seed generates a random nonce, signs the canonical timestamp scope, validates the resulting attestation, and stores only the public payload and signature. The private key is not retained.

## Execute

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuances.py execute \
  ISSUANCE_ID \
  --output-path evidence/restore-chain.timestamp.json \
  --worker-id timestamp-worker-1 \
  --lease-seconds 60
```

Execution:

1. claims or reclaims the exact issuance with an expiring lease;
2. verifies the supplied output-path digest;
3. reconstructs and validates the stored attestation;
4. rechecks the authority public fingerprint and registration/retirement chronology;
5. atomically creates the exact canonical output when absent;
6. reuses an already existing output only when its bytes exactly equal the persisted payload;
7. records `output_published`;
8. rereads and verifies the exact output;
9. records a deterministic verification digest and completes the issuance.

No output overwrite or merge operation exists.

## Recovery phases

```text
planned
→ output_published
→ verified
```

### Crash after output creation before phase persistence

The journal fails while preserving `planned`. After exact retry, execution finds the existing byte-identical output, records `output_published`, verifies it, and completes.

### Crash after phase persistence before completion

The journal fails while preserving `output_published`. After exact retry, the output must still exist and exactly match the stored payload. Missing or divergent output fails closed.

### Authority retirement during recovery

A prepared attestation can finish after key retirement only when:

```text
registered_at <= attestation.asserted_at <= retired_at
```

This preserves a valid pre-retirement issuance intent without allowing post-retirement assertions.

## Status and listing

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuances.py status ISSUANCE_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuances.py list \
  --owner-id alice \
  --state failed
```

These commands use SQLite `mode=ro` and `query_only=ON`. They do not initialize a missing journal.

Output includes only IDs, digests, states, phases, lease presence/expiry, counts, timestamps, verification digest, and generic failure type. It excludes:

- attestation signature;
- nonce material;
- authority private key;
- raw output, envelope, public-key, or database paths;
- custody source text.

## Retry and cancellation

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuances.py retry \
  ISSUANCE_ID \
  --owner-id alice \
  --confirm-issuance-id ISSUANCE_ID
```

Retry preserves the recovery phase and obeys the attempt ceiling.

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuances.py cancel \
  ISSUANCE_ID \
  --owner-id alice \
  --confirm-issuance-id ISSUANCE_ID
```

Cancellation is allowed only for `planned` or `failed` work that remains in phase `planned`. Once output work has been durably recorded, cancellation is refused.

Both commands require exact confirmation before opening the mutable journal and revalidate confirmation inside the journal.

## Integrity defenses

The journal refuses:

- issuance ID/scope divergence;
- attestation digest, serial, owner, authority, or key mismatch;
- duplicate JSON keys and non-finite JSON constants in stored payloads;
- duplicate serial reservation;
- invalid state/phase/lease combinations;
- non-monotonic timestamps;
- attempt-ceiling bypass;
- wrong worker or expired lease mutation;
- symlink/reparse path traversal;
- database or parent identity replacement;
- output path mismatch;
- divergent pre-existing output;
- missing output after the `output_published` phase;
- authority fingerprint or chronology drift.

## Verification boundary

Focused reconstructed execution passed:

```text
8 passed
```

Covered:

- normal issuance and terminal replay;
- unique serial reservation;
- crash after output creation before phase persistence;
- crash after output phase before completion;
- divergent output collision;
- missing previously published output;
- retired-key historical completion;
- database payload tamper refusal;
- query-only journal write refusal;
- runtime database-alias refusal and caching;
- output confirmation before store resolution;
- seed/execute/status/list privacy-safe CLI behavior.

Combined custody timestamp execution passed:

```text
21 passed
```

This includes the 13 attestation, authority-lifecycle, operator, and rotation checks plus the 8 issuance checks.

The repository-native issuance suite contains eight test functions. It has not been executed from a complete unchanged checkout of the current repository.

## Remaining work

- independent-process serial-reservation contention;
- independent-process output-path races;
- real process-kill recovery at both output phases;
- SQLite busy/locked, WAL, I/O, and disk-full injection;
- output directory fsync failure injection;
- issuance operational audit and conservative retention planning;
- durable legal holds over issuance records;
- RFC 3161 or external trusted-time integration;
- exact-current full pytest, coverage, Ruff, Windows, and container matrices.

## Non-claims

- Persisting a signed attestation does not prove an external clock was accurate.
- The receipt remains a custom Ed25519 authority attestation, not an RFC 3161 token.
- A completed issuance does not prove scientific correctness.
- Journal recovery is not distributed consensus.
- Focused reconstructed checks are not the complete release matrix.
- Release readiness is not claimed.
