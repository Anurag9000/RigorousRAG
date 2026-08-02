# Crash-recoverable signed publication retirement

Last updated: 2026-08-02

This runbook covers the exact retirement of an expired authorization-only publication attempt when the matching signed-provenance publication already completed.

The retirement saga exists to close a narrow cross-journal recovery risk:

- the authorization-only and signed publication paths intentionally use separate journals;
- both may contain the same logical publication operation;
- the signed operation may complete while the weaker authorization-only operation remains `running` under an expired lease;
- the weaker candidate may still be current or may be reclaimable by another worker;
- direct database edits or one-shot cancellation would create pointer and crash-consistency hazards.

The retirement saga never relabels an authorization-only graph set as signed. It never restores the weaker candidate as compensation.

## 1. Isolated durable stores

Configure three distinct SQLite files:

```bash
EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH=data/evidence_graph_set_publications.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH=data/evidence_graph_set_signed_publications.sqlite3
EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH=data/evidence_graph_set_signed_retirements.sqlite3
```

The retirement runtime rejects:

- equal canonical paths;
- relative and absolute paths resolving to the same location;
- existing hard links referencing either publication journal;
- path redirection through symlinks or reparse points;
- parent-directory or database-inode replacement after startup.

The retirement database stores IDs, phases, lease metadata, authority digests, pointer observations and verification digests. It stores no source text, relation evidence, query text, assertion body, signature or key material.

## 2. Prerequisites

A retirement may be seeded only when a fresh read-only preflight reports one of:

- `retire_expired_journal_only`; or
- `restore_signed_pointer_then_retire`.

The preflight proves:

- the authorization-only and signed rows have the same immutable logical operation scope;
- the weaker row is `running` with an expired lease;
- the signed row is `completed` and `verified`;
- the signed candidate ID and digest match the immutable graph-set store;
- the signed candidate is authoritative against current generations and evidence graphs;
- the current pointer is either the signed candidate or the weaker candidate.

Run the preflight directly when inspecting an operation:

```bash
python scripts/evidence_graph_set_signed_retirement.py preflight OPERATION_ID \
  --owner-id alice
```

The preflight is read-only.

## 3. Seed a retirement saga

Seeding reruns the preflight and requires exact operation-ID confirmation:

```bash
python scripts/evidence_graph_set_signed_retirements.py seed OPERATION_ID \
  --owner-id alice \
  --confirm-operation-id OPERATION_ID \
  --max-attempts 3
```

Seeding writes only the retirement journal. It does not:

- claim either publication attempt;
- change the graph-set pointer;
- cancel the weaker attempt;
- alter an immutable graph-set version;
- publish a new graph set.

The deterministic retirement identity commits:

- owner ID;
- logical publication operation ID;
- graph-set key;
- signed candidate ID and digest;
- weaker candidate ID, when present;
- signed candidate authority digest.

An exact repeated seed returns the existing retirement. A different immutable scope produces a different retirement ID or fails as an identity collision.

## 4. Execute or reconcile

Execute one exact retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py execute RETIREMENT_ID \
  --worker-id retirement-worker-1 \
  --lease-seconds 60
```

Reconcile the next claimable retirement for one owner:

```bash
python scripts/evidence_graph_set_signed_retirements.py reconcile-one \
  --owner-id alice \
  --worker-id retirement-worker-1 \
  --lease-seconds 60
```

`execute` uses a renewable wall-clock lease. `reconcile-one` captures one bounded timestamp for queue selection and the selected execution, matching the repository's other one-item reconciliation boundaries.

## 5. Durable phases

The saga advances monotonically:

```text
planned
  → pointer_restore_intent
  → pointer_safe
  → authorization_retired
  → verified
```

### `planned`

The engine reloads both publication rows, reloads the signed candidate and revalidates signed authority.

Before durable intent, the current pointer must be either:

- the completed signed candidate; or
- the weaker authorization-only candidate.

Any other pointer causes refusal with no pointer or weaker-journal mutation.

### `pointer_restore_intent`

The engine has durably accepted only the signed/weaker candidate pointer states.

It then takes over the weaker attempt's expired lease under:

```text
signed-retirement:<RETIREMENT_ID>
```

The takeover:

- is allowed only when the previous lease expired, or when the same retirement already owns it;
- does not increment the weaker publication attempt count;
- is renewed through later phases;
- blocks ordinary authorization-only workers from reclaiming the weaker row.

If the weaker candidate is current, the engine restores the signed candidate through compare-and-swap using the exact weaker candidate ID as the expected pointer.

If the signed candidate is already current, no pointer write occurs.

If a different pointer appears after durable intent, the engine preserves that newer pointer and continues only with retirement of the weaker duplicate. It never overwrites a post-intent external publication.

### `pointer_safe`

The engine renews the exact weaker-row retirement lease.

It refuses cancellation if the weaker candidate became current again after pointer safety.

Only the exact retirement lease may transition the weaker row to `cancelled`. The operation is idempotent when the weaker row was already cancelled by the same recovery history.

### `authorization_retired`

The weaker publication row is durably cancelled and cannot be reclaimed.

The engine rechecks:

- weaker state is `cancelled`;
- the weaker candidate is not current;
- signed candidate identity remains intact;
- signed candidate remains authoritative;
- the stored signed authority digest still matches.

A newer external pointer may remain current. Retirement correctness requires only that the weaker candidate is not current and cannot resume.

### `verified`

The terminal verification digest commits:

- retirement ID;
- logical publication operation ID;
- signed and weaker candidate IDs;
- final pointer observation;
- signed authority digest;
- terminal weaker state.

The retirement lease is cleared and the result is immutable.

## 6. Crash recovery matrix

### Crash after retirement claim

The retirement journal remains `running`. Another worker may reclaim it only after lease expiry. The attempt count increases and the recorded phase is preserved.

### Crash after `pointer_restore_intent`

No weaker cancellation is assumed. Recovery reacquires or renews the exact weaker-row retirement lease and examines the actual pointer.

### Crash after signed pointer commit but before `pointer_safe`

The actual pointer is the source of truth. Recovery observes the signed candidate already current, avoids a second pointer write and records `pointer_safe`.

### Crash after weaker cancellation but before `authorization_retired`

Recovery observes the weaker row already cancelled, treats cancellation as idempotent and records `authorization_retired`.

### Crash after `authorization_retired` but before completion

Recovery performs final pointer, weaker-state and signed-authority verification, then writes the terminal digest.

### Lease expiry during any phase

The retirement journal can be reclaimed after expiry without losing phase. The weaker publication lease is separately reacquired or renewed before any weaker-row cancellation.

## 7. External pointer behavior

The pointer policy deliberately differs before and after durable intent.

Before `pointer_restore_intent`:

- only the signed or weaker candidate pointer is accepted;
- an unrelated pointer causes refusal;
- no pointer or weaker-journal mutation occurs.

After `pointer_restore_intent`:

- the weaker candidate may be compare-and-swap replaced by the signed candidate;
- the signed candidate may already be current;
- any newer unrelated pointer is preserved;
- the weaker candidate is never restored;
- the weaker row may be retired only after proving it is not current.

This ordering prevents the saga from overwriting a newer publication while also preventing the expired weaker worker from resurrecting its candidate.

## 8. Failure, retry and cancellation

Inspect one retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py status RETIREMENT_ID
```

List retirements:

```bash
python scripts/evidence_graph_set_signed_retirements.py list \
  --owner-id alice \
  --publication-operation-id OPERATION_ID
```

Retry a failed retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py retry RETIREMENT_ID \
  --owner-id alice \
  --confirm-retirement-id RETIREMENT_ID
```

Retry preserves the durable recovery phase. It does not restart from `planned` after pointer intent has been recorded.

Cancel only an unstarted retirement:

```bash
python scripts/evidence_graph_set_signed_retirements.py cancel RETIREMENT_ID \
  --owner-id alice \
  --confirm-retirement-id RETIREMENT_ID
```

Cancellation is permitted only in phase `planned`, before pointer intent. Once pointer work may have begun, the saga must be recovered or inspected rather than discarded.

`status` and `list` load only the retirement journal. They do not require graph, generation or publication stores. `retry` and `cancel` also act only on the retirement journal and require exact confirmation.

## 9. Safety invariants

Implemented invariants:

- no weaker-pointer compensation path exists;
- no direct database edits are exposed through the CLI;
- no active unrelated weaker lease can be stolen;
- no weaker cancellation occurs without the exact saga-owned live lease;
- no pointer change occurs before durable intent;
- no pre-intent external pointer is overwritten;
- no post-intent external pointer is overwritten;
- no stale signed candidate may justify retirement completion;
- no weaker candidate may remain current at completion;
- no assurance-level journal files may alias;
- no raw source or signed assertion secret enters retirement state or output.

## 10. Verification evidence

Focused reconstructed execution using the committed saga logic and API-faithful stubs for unrelated repository services currently includes:

- deterministic retirement identity and journal lifecycle;
- lease reclaim with phase preservation;
- retry phase preservation and attempt ceilings;
- exact weaker lease takeover without retry-count inflation;
- exact lease-owner cancellation and idempotent cancellation replay;
- signed pointer restoration through compare-and-swap;
- already-signed-pointer no-op behavior;
- crash after pointer commit;
- crash after weaker cancellation;
- pre-intent external-pointer refusal;
- post-intent external-pointer preservation;
- stale signed-authority refusal;
- live competing weaker-lease refusal;
- weaker-pointer reactivation refusal;
- late signed-authority drift after weaker retirement;
- weaker-lease renewal before cancellation;
- raw post-claim failure normalization;
- third-journal path isolation.

The latest focused saga harness passes **12/12** checks and compiles the new core modules.

This is not an exact-current complete repository run. Full pytest, coverage, Ruff, Windows, containers, true multi-process kill injection, SQLite I/O failure injection and disk-full testing remain open. Release readiness is not claimed.
