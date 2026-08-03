# Custody artifact operational audit and retention planning

Last updated: 2026-08-03

This runbook covers read-only inspection of durable pre-restore backup/receipt publication attempts.

The artifact journal records intent before publication and terminally distinguishes verified pairs, explicit orphan evidence, failures, and cancelled operations. Operational tooling does not create journals, retry work, alter files, release holds, or delete history.

## 1. Read-only audit

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifact_operations.py audit \
  --owner-id alice \
  --artifact-db-path data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3 \
  --limit 1000
```

Optional filters:

```bash
--restore-id RESTORE_ID
--state planned|running|completed|orphaned|failed|cancelled
```

The restore ID is deterministically derived from the immutable owner, snapshot digest, and target-path digest already bound into the artifact attempt. No journal migration or redundant restore-ID column is required.

Classifications are:

- `planned_ready`;
- `running_active`;
- `running_expired_reclaimable`;
- `running_expired_exhausted`;
- `failed_retryable`;
- `failed_exhausted`;
- `completed_pair`;
- `orphan_backup_without_receipt`;
- `orphan_receipt_without_backup`;
- `orphan_artifact_collision`;
- `cancelled`.

Reports contain only IDs, digests, byte counts, state/phase, lease presence, generic failures, and timestamps. They never return raw target, snapshot, backup, receipt, or database paths, and never return receipt actor IDs.

Reports revalidate item counts, classification counts, unique IDs, safety flags, and their deterministic digest when reconstructed.

## 2. Conservative retention plan

```bash
python scripts/evidence_graph_set_signed_retirement_restore_custody_artifact_operations.py retention-plan \
  --owner-id alice \
  --artifact-db-path data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3 \
  --minimum-age-seconds 31536000 \
  --retain-latest-per-target 1 \
  --durable-hold-db-path data/evidence_graph_set_signed_retirement_holds.sqlite3
```

Default policy:

- planned, running, and failed attempts are excluded;
- every orphan record is permanently excluded because it is unresolved chain-of-custody evidence;
- completed pairs are retained by default;
- cancelled attempts may become planning candidates only when old and not the newest terminal record for the target;
- the newest configured number of terminal records for each target-path digest are protected;
- active durable restore holds protect every artifact attempt deriving the held restore ID.

To consider old duplicate completed-pair records, the operator must explicitly add:

```bash
--include-completed
```

Temporary planning-only protection may be added with:

```bash
--hold-restore-id RESTORE_ID
```

The durable hold database remains the authoritative long-lived hold source. Temporary command-line holds do not create legal-hold records.

## 3. No destructive action

The operations script exposes only:

```text
audit
retention-plan
```

It has no delete, unlink, overwrite, compact, retry, cancel, publish, or hold-release command.

Every report states:

```text
journal_mutation_performed: false
hold_store_mutation_performed: false
artifact_mutation_performed: false
artifact_deletion_performed: false
artifact_overwrite_performed: false
source_text_returned: false
raw_path_returned: false
```

A retention candidate is planning information, not deletion authorization.

## 4. Verification boundary

The durable artifact publication slice was executed in a reconstructed dependency workspace using the exact committed artifact contracts, governed journal, recovery executor, runtime, and query-only SQLite view with minimal stubs only for older unrelated services:

```text
19 focused checks passed
```

Executed coverage includes:

- deterministic identity;
- SQLite lifecycle and phase guards;
- lease reclaim and attempt ceilings;
- normal pair publication;
- crash-after-publication recovery;
- all three orphan dispositions;
- completed-pair live revalidation and tamper refusal;
- scope mismatch before claim;
- canonical and hard-link runtime alias refusal;
- query-only read/write refusal.

The artifact audit/retention CLI and four repository-native operational contracts are committed. They have not yet been executed together from a complete exact-current checkout.

Full repository pytest, coverage, Ruff, platform matrices, independent-process contention, process-kill injection, SQLite I/O/disk-full injection, and exact-current integration remain open. Release readiness is not claimed.
