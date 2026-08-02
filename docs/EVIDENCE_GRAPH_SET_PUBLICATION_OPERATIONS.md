# Evidence-graph-set publication operations

Last updated: 2026-08-02

## Scope

This layer provides privacy-safe operational visibility for the durable graph-set publication journal. It does not modify publication attempts, graph-set pointers, graph-set history, reviewed proposals or authoritative document generations.

## Audit classifications

`audit_publication_attempts` classifies each owner-scoped operation as one of:

- `planned`;
- `running_active`;
- `expired_reclaimable`;
- `expired_exhausted`;
- `failed_retryable`;
- `failed_exhausted`;
- `compensation_failed`;
- `completed`;
- `compensated`;
- `cancelled`.

The report contains only operation IDs, graph-set keys, states, phases, attempt/lease counts, candidate/previous IDs, generic failure types, compensation-error counts and timestamps. It contains no graph text, proposal evidence, query text, source path or provider response.

A deterministic report digest binds the owner, optional graph-set key, generation time, classifications and ordered item identities.

## Retention planning

`plan_publication_retention` is planning-only. It never deletes data and always reports:

```json
{
  "deletion_performed": false
}
```

An attempt can be proposed as `old_terminal_noncurrent` only when all of the following hold:

1. its state is `completed`, `compensated` or `cancelled`;
2. it has a completion timestamp older than the selected cutoff;
3. neither its candidate nor previous graph-set ID is the current pointer for its graph-set key;
4. it has no compensation-error record.

The planner retains records with these reasons:

- `nonterminal`;
- `failure_record`;
- `missing_completion_time`;
- `recent_terminal`;
- `references_current_pointer`;
- `compensation_errors`.

Eligibility is not authorization to delete. Legal hold, backup, export, audit requirements and a separately reviewed destructive protocol remain prerequisites.

## Commands

```bash
python -m tools.evidence_graph_set_publish_operations_cli audit \
  --owner-id alice \
  --graph-set-key review \
  --limit 10000

python -m tools.evidence_graph_set_publish_operations_cli retention-plan \
  --owner-id alice \
  --graph-set-key review \
  --minimum-age-seconds 2592000 \
  --limit 10000
```

Script wrapper:

```bash
python scripts/evidence_graph_set_publication_operations.py ...
```

Both commands are read-only and return no source text.

## Focused verification

Four focused contracts passed for:

- expired/reclaimable/exhausted and retryable/exhausted failure classification;
- deterministic audit digests;
- exclusion of recent, current-referenced and failed records from retention candidates;
- explicit non-deletion behavior in the API and CLI.

Together with the journal/recovery suites, the local publication control-plane harness passed **32/32 tests**, and compilation passed for all publication modules and scripts.

This is focused local verification, not a replacement for an unchanged-head full repository/platform matrix.

## Remaining operations work

- Add durable privacy-safe audit export with signed manifests.
- Add dashboards and alert thresholds for expired leases and compensation failures.
- Add legal-hold and retention-policy records.
- Add backup/restore drills for the journal and graph-set store.
- Add a separately reviewed exact-confirmation deletion/compaction protocol only after those controls exist.
- Add distributed worker leadership and coordination with callers of the legacy synchronous publisher.

## Permanent non-claims

- A retention candidate is not deletion authorization.
- An expired lease does not prove a worker process is dead.
- A completed publication does not establish scientific truth.
- Release readiness is not claimed.
