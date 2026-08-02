# Durable evidence-graph-set publication journal

Last updated: 2026-08-02

## Purpose

Reviewed cross-document relations were already publishable through a compare-and-swap and compensating pointer protocol. That synchronous path protected ordinary exceptions, but a process could still terminate after one durable step and before the next in-memory step was recorded.

The publication journal adds a resumable phase record around that existing protocol. It does not weaken or replace the established `publish_approved_graph_set` API. It provides a separate operator path for publications that require durable crash recovery.

## Immutable operation identity

`EvidenceGraphSetPublicationAttempt` binds:

- owner ID;
- logical graph-set key;
- a sorted, unique list of reviewed proposal IDs;
- the exact expected current graph-set ID, or an explicit no-current expectation;
- the maximum attempt count.

The deterministic operation ID excludes timestamps, leases, attempts and outcomes. Re-seeding the same immutable intent reuses the same operation; a changed proposal list or pointer expectation creates a different operation.

The journal stores no graph text, proposal evidence, queries, source paths, provider responses or automatic semantic output.

## State and phase model

States:

```text
planned
running
completed
compensated
failed
cancelled
```

Durable phases:

```text
planned
candidate_stored
pointer_activated
verified
compensated
```

A worker must hold an unexpired exclusive lease for every phase transition. Expired running attempts are reclaimable while their attempt ceiling remains available.

## Publication protocol

For a new or resumed operation, the executor:

1. loads every immutable proposal and verifies owner/key scope;
2. acquires all member document locks in deterministic order;
3. verifies the actual pointer matches the operation’s immutable expectation;
4. resolves the exact current authoritative member graphs;
5. converts only terminally approved proposals into explicit relations;
6. deterministically rebuilds the candidate graph set;
7. stores the candidate as an immutable non-current version;
8. records previous and candidate IDs, digests and counts;
9. checks candidate authority;
10. compare-and-swap activates the candidate pointer;
11. records pointer activation;
12. rechecks candidate authority and pointer identity;
13. records a completed outcome digest.

No step mutates vector rows, sparse rows, retained sources or authoritative generation records.

## Crash recovery

Recovery uses the graph-set pointer as the durable source of truth rather than trusting the journal phase alone.

### Before candidate identity is journaled

If the candidate was stored but the process stopped before `candidate_stored` was persisted, the next claimant deterministically rebuilds the same candidate and records its exact identity.

### After pointer commit but before phase persistence

If the actual pointer already equals the candidate while the journal still says `candidate_stored`, the next claimant records activation and proceeds to verification. An ordinary exception in this window triggers compensation immediately even though the phase record lagged behind the pointer.

### After activation

If the candidate remains authoritative, recovery records `completed`.

If a member generation or graph changed, recovery restores the exact previous pointer or clears the pointer for a failed first publication, verifies that compensation, and records `compensated`.

### After compensation but before terminal persistence

If the pointer already equals the previous set—or is absent for a first publication—the next claimant records the compensated terminal outcome without republishing the candidate.

### External pointer changes

A pointer that is neither the immutable previous set nor the operation’s candidate is treated as external concurrent work. Recovery fails closed and never overwrites that pointer.

## Compensation

For first publication, compensation clears the pointer only when it still equals the exact candidate ID.

For replacement, compensation restores the complete immutable previous graph set only when the pointer still equals the exact candidate ID.

After either action, recovery re-reads the pointer. Incomplete compensation is recorded as a failed attempt with bounded generic error codes rather than private exception text.

## Operator commands

Configuration:

```dotenv
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH=data/evidence_graph_set_publications.sqlite3
```

Commands:

```bash
python -m tools.evidence_graph_set_publish_journal_cli seed \
  --owner-id alice \
  --graph-set-key review \
  --proposal-id <approved-proposal-sha256> \
  --expect-no-current

python -m tools.evidence_graph_set_publish_journal_cli seed \
  --owner-id alice \
  --graph-set-key review \
  --proposal-id <approved-proposal-sha256> \
  --expected-current-set-id <current-set-sha256>

python -m tools.evidence_graph_set_publish_journal_cli execute \
  <operation-id> --worker-id publisher-1 --lease-seconds 60

python -m tools.evidence_graph_set_publish_journal_cli reconcile-one \
  --owner-id alice --worker-id publisher-1 --lease-seconds 60

python -m tools.evidence_graph_set_publish_journal_cli status <operation-id>

python -m tools.evidence_graph_set_publish_journal_cli list \
  --owner-id alice --graph-set-key review --limit 100

python -m tools.evidence_graph_set_publish_journal_cli retry \
  <operation-id> --owner-id alice \
  --confirm-operation-id <same-operation-id>

python -m tools.evidence_graph_set_publish_journal_cli cancel \
  <operation-id> --owner-id alice \
  --confirm-operation-id <same-operation-id>
```

The script wrapper is:

```bash
python scripts/evidence_graph_set_publications.py ...
```

Retry is explicit and limited to failed or successfully compensated attempts below their ceiling. Cancellation is limited to planned/failed attempts and cannot cancel an activated pointer phase.

Operator output contains only IDs, digests, counts, states, phases, attempts, leases and generic failure/compensation codes. It explicitly reports that authoritative mutation, semantic inference and automatic approval were not performed.

## Focused verification

The initial isolated recovery harness passed 19 contracts while the design was being hardened. A repository-compatible committed suite then passed 9 focused contracts covering:

- deterministic identity and idempotent seeding;
- exclusive leases, expiry reclaim and attempt ceilings;
- successful publication and terminal replay;
- crash after pointer commit before phase persistence;
- exception after unjournaled activation with exact compensation;
- crash after compensation before terminal journal persistence;
- replacement failure restoring the previous pointer;
- row tamper and database replacement detection;
- retry, cancellation, owner scope and confirmation boundaries;
- privacy-safe seed/status/list/cancel/idle CLI behavior.

These are focused local contracts. Exact-current full-repository, Windows, Docker and distributed multi-process verification must be recorded separately after execution.

## Deliberate boundaries

- The journal is not a distributed consensus system.
- SQLite leases coordinate cooperating workers that use this journal; they do not prevent an unrelated process from calling the legacy synchronous publisher directly.
- Shared document locks remain process-local.
- Reviewer authorization and separation of duties are not added by this slice.
- Candidate graph sets remain immutable history after compensation; only the current pointer is restored or cleared.
- Automatic relation approval remains impossible.
- Release readiness is not claimed.
