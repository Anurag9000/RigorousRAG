# Custody timestamp issuance legal holds

Last updated: 2026-08-03

This subsystem adds durable, integrity-backed legal holds over custody timestamp issuance records.

A hold affects retention planning only. It does not retry, cancel, complete, republish, overwrite, delete, or otherwise mutate the referenced issuance.

## Configuration

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH=data/evidence_graph_set_signed_retirement_custody_timestamp_issuances.sqlite3
EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_HOLD_DB_PATH=data/evidence_graph_set_signed_retirement_custody_timestamp_issuance_holds.sqlite3
```

The hold database must not equal or hard-link to the issuance, authority, signer, custody, artifact, restore, retirement, publication, or administration databases.

Hold placement and release use the existing process-owned or signed actor binding.

## Immutable hold identity

A hold ID is deterministic over:

- owner ID;
- exact issuance ID;
- operator-defined hold key.

The hold record additionally binds:

- reason code;
- active or released status;
- creation actor ID, binding method, binding digest, and timestamp;
- release actor ID, binding method, binding digest, and timestamp when released;
- schema version;
- complete hold-record digest.

The complete mutable state is committed by `hold_digest` in the same SQLite row. Modifying status, reason, actor provenance, timestamps, or scope without recomputing the exact record is detected during reconstruction.

## Place a hold

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds.py place \
  --owner-id alice \
  --issuance-id ISSUANCE_ID \
  --hold-key litigation-2026-01 \
  --reason-code legal_matter
```

Placement requires:

1. a configured process-owned or signed actor binding;
2. an initialized query-only issuance journal;
3. the referenced issuance to exist;
4. the issuance owner to equal the requested owner;
5. a non-expired actor binding;
6. an exact deterministic hold identity.

Exact replay with the same issuance, hold key, reason code, and actor binding returns the original record and preserves the original creation timestamp.

Changing any governed scope field produces an identity collision and fails closed.

## Release a hold

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds.py release \
  HOLD_ID \
  --owner-id alice \
  --confirm-hold-id HOLD_ID
```

Release requires exact hold-ID confirmation before the actor or mutable store is opened. The store repeats confirmation and owner validation inside the transaction.

Release is monotonic:

```text
active → released
```

A released hold cannot become active again. Exact replay returns the stored released record and preserves the original release provenance.

## Query-only inspection

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds.py status HOLD_ID
```

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds.py list \
  --owner-id alice \
  --issuance-id ISSUANCE_ID \
  --status active
```

Status and list use SQLite `mode=ro` with `query_only=ON`. A missing or uninitialized hold database is not created by inspection.

Operator summaries expose:

- hold, owner, and issuance IDs;
- hold key and reason code;
- active/released status;
- binding methods and binding digests;
- creation/release timestamps;
- hold digest.

They omit raw actor IDs and all database paths.

## Retention integration

The timestamp issuance retention planner accepts:

```bash
--hold-db-path PATH
```

or reads:

```bash
EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_HOLD_DB_PATH
```

When configured, the planner reads all active hold records through the query-only hold view and merges their issuance IDs with any explicit temporary `--hold-issuance-id` values.

An active durable hold forces:

```text
held: true
retention_candidate: false
reason: legal_hold
```

After governed release, that hold no longer protects later retention plans. Earlier plans remain immutable evidence of their own generation time.

The planner reports only hold counts, not hold database paths or actor identities.

## Integrity and path defenses

The hold store refuses:

- issuance/owner scope mismatch;
- unsupported status;
- active rows containing release fields;
- released rows missing release fields;
- release chronology before creation;
- deterministic hold-ID divergence;
- hold-digest divergence;
- expired actor binding;
- duplicate scope collisions;
- missing or invalid issuance read boundary;
- wrong owner or exact-confirmation mismatch;
- symlink/reparse traversal;
- database or parent inode replacement;
- query-only writes;
- database aliasing with protected journals.

## Verification boundary

Focused reconstructed execution passed:

```text
29 passed
```

The four issuance-hold checks cover:

- deterministic placement and replay;
- owner-scope validation;
- exact release confirmation and monotonic release;
- actor-expiry refusal;
- database-row and file-identity tamper detection;
- query-only write refusal;
- cross-journal alias refusal;
- confirmation before actor/store resolution;
- active durable-hold retention protection;
- released-hold removal from subsequent retention plans;
- path-free and actor-ID-free output.

The 29-test combined timestamp stack also includes authority attestations, authority lifecycle, rotation planning, one-serial issuance recovery, and issuance operations.

These are focused reconstructed tests, not a complete unchanged checkout of the current repository.

## Permanent non-claims

- A legal hold is not deletion authorization.
- Hold release is not deletion or compaction authorization.
- Legal holds do not prove custody evidence is scientifically correct.
- Process-owned actor binding is not external institutional IAM unless configured through a governed identity source.
- Retention planning does not perform deletion.
- Focused reconstructed checks are not the complete release matrix.
- Release readiness is not claimed.
