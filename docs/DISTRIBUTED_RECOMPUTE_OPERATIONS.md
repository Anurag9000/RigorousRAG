# Distributed research recomputation operations

RigorousRAG keeps research-artifact invalidation and recomputation authority in the
owner-scoped dependency invalidation ledger. The durable queue is transport only.
Queue messages contain exactly one opaque `task_id`; they do not contain owner IDs,
queries, evidence text, citations, report bodies, or artifact payloads.

## Execution modes

`scripts/recompute_research.py` exposes three bounded operator actions:

- `local` directly claims and executes work from the authoritative ledger. This is the
  historical/default behavior and does not require a queue.
- `publish` scans currently queued authoritative tasks for one owner and idempotently
  publishes opaque handoffs to the configured durable queue.
- `worker` consumes at most `--max-tasks` queue records, re-validates and exact-claims
  every task in the authoritative owner-scoped ledger, executes it, and settles the
  transport record.

The CLI deliberately does not daemonize and does not poll forever. A scheduler,
supervisor, Kubernetes Job/CronJob, systemd timer, or other deployment controller may
invoke bounded publish/worker cycles explicitly.

## Same-host durable SQLite transport

The built-in `SQLiteDurableQueue` is intended for independent producer and worker
processes on the same host/container-volume domain. It uses WAL journaling,
`BEGIN IMMEDIATE` claim fencing, visibility receipts, delayed negative acknowledgements,
bounded transport attempts, dead-lettering, and namespace-scoped idempotency.

Do **not** place this SQLite file on an unsupported NFS/SMB/network filesystem and call
that a multi-host queue. Multi-host high availability requires a networked durable queue
provider implementing `tools.durable_queue.DurableQueue`; that remains a separate
production-distribution capability.

Enable the built-in provider explicitly:

```text
RECOMPUTE_QUEUE_BACKEND=sqlite
RECOMPUTE_QUEUE_DB_PATH=data/governance/research_recompute_queue.sqlite3
RECOMPUTE_QUEUE_NAMESPACE=research-recompute-v1
RECOMPUTE_QUEUE_MAX_ATTEMPTS=5
RECOMPUTE_QUEUE_MAX_PAYLOAD_BYTES=4096
RECOMPUTE_LEDGER_MAX_ATTEMPTS=5
RECOMPUTE_CLAIM_TIMEOUT_SECONDS=900
RECOMPUTE_VISIBILITY_TIMEOUT_SECONDS=1800
RECOMPUTE_BUSY_RETRY_SECONDS=30
```

`RECOMPUTE_QUEUE_BACKEND=disabled` is the default. `publish` and `worker` fail closed
when transport is disabled. If `RECOMPUTE_QUEUE_DB_PATH` is blank in a non-Compose
deployment, the runtime derives
`<CLASSIC_STORAGE_DIR>/governance/research_recompute_queue.sqlite3`.

The visibility timeout must be greater than or equal to the authoritative ledger claim
timeout. This prevents routine transport redelivery from repeatedly encountering a
still-fresh ledger claim and consuming the transport retry budget.

A queue namespace persists its configured maximum transport-attempt count. Changing
that setting for an existing namespace fails closed; use an intentionally new namespace
for a changed transport policy.

## Publish and consume

Publish up to 100 queued tasks for one owner:

```bash
python scripts/recompute_research.py \
  --mode publish \
  --owner-id alice \
  --max-tasks 100
```

Consume up to 100 handoffs using an explicit worker identity:

```bash
python scripts/recompute_research.py \
  --mode worker \
  --owner-id alice \
  --worker-id recompute-worker-01 \
  --max-tasks 100
```

The worker exits when no handoff is immediately available or the bound is reached.
Worker output is metadata-only: task IDs, terminal/coordination state, success, and
error type. It does not emit private replay queries or evidence content.

## Explicit failed-task retry

A recompute handler failure is durable in the authoritative ledger and the transport
occurrence is acknowledged. Retrying is an explicit operator decision:

```bash
python scripts/recompute_research.py \
  --mode publish \
  --owner-id alice \
  --retry-task <task-id> \
  --max-tasks 100
```

The ledger attempt counter is preserved when a failed task is requeued. Handoff
idempotency is bound to `(owner, task_id, ledger_attempt_epoch)`, so repeated publication
of the same queued epoch is idempotent while a deliberate retry can create the next
transport record. A worker never auto-publishes a retried task.

To requeue without publishing or executing anything:

```bash
python scripts/recompute_research.py \
  --owner-id alice \
  --retry-task <task-id> \
  --retry-only
```

## Compose usage

The main Compose service carries the recompute transport environment and already mounts
`/app/data` on the persistent `rigorousrag_crawl` volume. With
`RECOMPUTE_QUEUE_BACKEND=sqlite`, the default Compose queue path is
`/app/data/governance/research_recompute_queue.sqlite3`.

Use one-shot operator containers so they share the exact deployment image, environment,
and persistent stores:

```bash
docker compose run --rm api python scripts/recompute_research.py \
  --mode publish --owner-id alice --max-tasks 100

docker compose run --rm api python scripts/recompute_research.py \
  --mode worker --owner-id alice --worker-id compose-worker-01 --max-tasks 100
```

For multi-tenant deployments, invoke owner-bound cycles deliberately. The bridge is
constructed for exactly one normalized owner; the transport message itself never grants
tenant authority.

## Recovery semantics

- A crashed transport worker leaves a visibility lease. After expiry the queue record is
  reclaimable or dead-lettered if its transport-attempt budget is exhausted.
- A crashed application worker also leaves an authoritative ledger claim. A replacement
  worker may recover that claim only after `RECOMPUTE_CLAIM_TIMEOUT_SECONDS`.
- A stale receipt cannot acknowledge or nack a record after its lease expires.
- Duplicate delivery of a completed/failed/cancelled authoritative task is acknowledged
  without re-executing the handler.
- A malformed handoff or a task missing from the owner-scoped ledger never reaches a
  recompute handler.
- Handler success/failure is committed to the authoritative ledger before the transport
  occurrence is acknowledged.

The invalidation ledger therefore remains the source of truth; queue durability improves
handoff availability without converting queue payloads into authorization or evidence.
