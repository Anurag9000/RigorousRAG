# Hold-permit recovery operations and retention planning

Last updated: 2026-08-04

This operator surface audits immutable stale-permit recovery receipts and produces conservative retention plans. It is read-only: there is no command to delete a receipt, release a quarantine hold, change a permit, or mutate a restore intent.

## Audit recovery evidence

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery_operations.py \
  audit \
  --owner-id OWNER \
  --limit 1000
```

For every integrity-verified recovery receipt, the audit revalidates:

- the live permit row still exists;
- owner and restore scope remain exact;
- the permit is `released`;
- the live released-permit digest matches the receipt;
- a referenced quarantine hold still exists;
- quarantine owner, restore, hold digest and status remain exact.

The resulting classifications are:

- `quarantine_active`: recovery created a quarantine hold and it remains active;
- `quarantine_released`: the quarantine hold was subsequently released through the normal governed hold path;
- `released_hold_cleanup`: recovery cleaned a stale permit whose original hold was already released, so no quarantine was created.

Missing or mismatched permits, quarantine holds, integrity records or receipt rows fail closed rather than becoming retention candidates.

Audit output contains IDs, digests, states, timestamps and binding methods only. It returns no source text or raw filesystem paths.

## Plan conservative retention

```bash
python scripts/evidence_graph_set_signed_retirement_restore_hold_permit_recovery_operations.py \
  retention-plan \
  --owner-id OWNER \
  --minimum-age-seconds 31536000 \
  --retain-latest-per-restore 1 \
  --hold-recovery-id RECOVERY_ID \
  --limit 10000
```

A recovery receipt is never a candidate while its quarantine hold remains active.

The plan also protects:

- every explicitly held recovery ID;
- the configured number of newest receipts for each restore intent;
- receipts younger than the minimum age;
- any evidence whose live permit or quarantine revalidation fails.

Only old, resolved evidence whose quarantine is released or was never required may appear as a planning candidate.

A planning candidate is not deletion authorization. No deletion implementation is exposed by this command family.

## Safety flags

Audit and retention responses explicitly report:

- `mutation_performed: false`;
- `hold_mutation_performed: false`;
- `permit_mutation_performed: false`;
- `deletion_performed: false`;
- `source_text_returned: false`;
- `raw_paths_returned: false`.

## Verification boundary

Four repository-native contracts are committed for:

1. active, released and cleanup classification;
2. live permit and quarantine drift refusal;
3. active-quarantine, latest-per-restore and explicit-hold retention protection;
4. CLI non-mutation and absence of delete/release commands.

These four contracts have not been executed inside a fresh exact-current complete repository checkout. The separately executed governed recovery harness remains **12/12** with focused compilation passing. Release readiness is not claimed.
