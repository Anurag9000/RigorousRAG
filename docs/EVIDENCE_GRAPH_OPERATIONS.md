# Evidence graph operational audit and retention planning

Last updated: 2026-08-02

## Scope

The evidence-graph operations layer provides privacy-safe queue health, dead-letter, artifact-integrity and retention-planning reports for derived graph jobs. It does not delete journal rows, graph generations or authoritative data.

## Components

- `tools/evidence_graph_operations.py`
  - deterministic owner-scoped operational reports;
  - expired-lease, retryable-failure and exhausted-dead-letter classification;
  - superseded nonterminal job detection;
  - current, stale and missing/mismatched completed-artifact classification;
  - conservative terminal-job retention planning;
  - timestamp-independent report and plan digests.
- `tools/evidence_graph_operations_cli.py`
  - `audit` and `retention-plan` commands;
  - JSON-only bounded output;
  - no graph text, sparse text, retained paths or provider responses;
  - no destructive command.
- `scripts/evidence_graph_operations.py`
  - operator entrypoint.

## Audit command

```bash
python -m tools.evidence_graph_operations_cli audit \
  --owner-id alice --limit 10000
```

The report contains:

- counts for planned, running, completed, failed and cancelled jobs;
- expired running leases;
- retryable failed jobs;
- exhausted dead-letter jobs;
- nonterminal jobs whose exact authoritative generation is no longer current;
- completed jobs that still exactly match both authoritative and graph current pointers;
- completed jobs whose graph is valid but no longer current;
- completed jobs whose historical graph is missing or has the wrong digest;
- a deterministic report digest excluding generation time.

The report does not expose graph node text, sparse fields, retained-source paths or exception messages.

## Conservative retention planning

```bash
python -m tools.evidence_graph_operations_cli retention-plan \
  --owner-id alice --min-age-seconds 2592000 --limit 10000
```

A job becomes a candidate only when all applicable conditions hold:

1. it is terminal (`completed` or `cancelled`);
2. it is at least the reviewed minimum age;
3. its authoritative source generation is no longer current;
4. its generation is not the current evidence-graph pointer;
5. for completed jobs, the historical graph still exists and its digest equals the recorded completion digest.

The planner explicitly retains:

- current or recent terminal jobs;
- planned, running and failed jobs;
- exhausted dead letters;
- completed jobs with missing or mismatched graph artifacts.

The response always contains:

```json
{
  "mutation_performed": false,
  "deletion_authorized": false
}
```

A retention candidate is an input to human review and future compaction policy, not permission to remove data.

## Focused verification

The focused operations harness passed **6 tests** covering:

- deterministic queue-state and artifact-health classification;
- expired leases, retryable failures and exhausted dead letters;
- superseded nonterminal jobs;
- current, stale and missing/mismatched completed jobs;
- conservative candidate selection;
- explicit retention of current, recent, active, failed and corrupt jobs;
- privacy-safe CLI output;
- planning-only deletion semantics;
- NaN/invalid-age refusal.

The combined new Wave 5 journal, reconciliation, authority and operations slice passed **26 tests** locally, and all changed modules compiled. This is focused local evidence, not the exact-head Linux/Windows/container release matrix.

## Remaining before destructive compaction

- Append-only or signed archival export before journal-row removal.
- Legal-hold and minimum-retention policy.
- Exact double confirmation for any future destructive command.
- Coordinated ordering between historical graph deletion and job-row deletion.
- Crash and disk-failure injection around archive and deletion boundaries.
- Multi-process leadership and operator authorization controls.
- Backup/restore validation and audit-log retention.

## Permanent non-claims

- A retention candidate is not deletion authorization.
- A dead-letter classification is not proof that a job can never succeed.
- Missing graph artifacts are integrity findings and are deliberately retained.
- Operational reports contain generation identity, not scientific truth.
- Release readiness is not claimed.
