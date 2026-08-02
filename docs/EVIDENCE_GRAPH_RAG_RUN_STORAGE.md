# Resumable evidence-graph benchmark run storage

Last updated: 2026-08-02

## Purpose

Large repeated-run GraphRAG benchmarks can be interrupted after some seeds complete. The run store persists each completed seed independently as a text-free immutable record, allowing a retry of the exact plan to reuse completed runs without resolving their queries or invoking their selectors again.

## Identity model

Two fingerprints are intentionally distinct:

- **benchmark fingerprint**: benchmark ID, ordered gold-case contract, run seeds and schema; selection configuration is excluded so baseline/candidate systems remain comparable;
- **plan fingerprint**: benchmark contract plus closed selector configuration; changing selector settings creates a different resumable execution plan.

Each run additionally binds:

- deterministic run ID;
- seed;
- exact gold-case digests;
- run-contract digest;
- run-result digest;
- run-report digest;
- case count;
- completion timestamp.

A completed `(plan fingerprint, run ID)` cannot be overwritten with a different result.

## Durable store

`GraphRAGBenchmarkRunStore` uses SQLite and stores:

- plan and benchmark fingerprints;
- benchmark/run identity;
- run aggregate metrics;
- per-case evaluation digests;
- contract/result/report/stored-run digests;
- seed, case count and timestamp.

It does not store:

- raw query text;
- evidence/node text;
- answer text;
- provider responses;
- source paths;
- citations.

Every read reconstructs validated public dataclasses and verifies row, payload and digest consistency. The store enforces parent/database identity and rejects symlink/reparse paths.

Default configuration:

```dotenv
EVIDENCE_GRAPH_RAG_RUN_DB_PATH=data/evidence_graph_rag_runs.sqlite3
```

## Resumable execution

`execute_resumable_live_graph_rag_benchmark(...)` processes runs in deterministic seed order:

1. derive exact run ID and contract digest from the plan;
2. look for an immutable completed run;
3. reuse it only when benchmark, plan, contract and case count match;
4. otherwise resolve each governed query, verify its digest and execute the selector;
5. reduce selections immediately to text-free observations;
6. compute and atomically persist one completed run report;
7. continue to the next seed;
8. assemble the final benchmark report only after the exact planned run set exists.

If a later run fails, earlier completed runs remain reusable. A retry does not call the query resolver or selector for reused runs.

Changing the seed list changes both benchmark and plan fingerprints. Changing selector configuration changes the plan fingerprint but preserves the benchmark fingerprint, allowing governed historical comparison without unsafe resume mixing.

## Operator CLI

The CLI is intentionally limited to text-free inspection and exact plan cleanup:

```bash
python -m tools.evidence_graph_rag_run_cli status <plan-fingerprint>

python -m tools.evidence_graph_rag_run_cli remove-plan <plan-fingerprint> \
  --confirm-plan-fingerprint <same-plan-fingerprint>
```

`status` is read-only. `remove-plan` requires exact fingerprint confirmation. Because query/evidence text is never stored, cleanup explicitly reports that it did not remove such text.

The CLI does not execute benchmarks because a production query provider must be injected programmatically and separately governed.

## Focused verification

The resumable storage stack passed **9 focused tests** covering:

- complete execution followed by full run reuse;
- interruption after one completed run and execution of missing runs only;
- no query resolution or selector call for reused runs;
- selector-configuration plan isolation with comparable benchmark fingerprints;
- absence of query/evidence text from SQLite payloads;
- row/payload tamper detection;
- database replacement detection;
- exact-confirmation plan removal;
- privacy-safe status and bounded not-found behavior.

## Remaining work

- Add per-plan retention, legal-hold and archival policy.
- Add process/distributed leases before concurrent writers are supported.
- Add per-case timeout/cancellation and bounded failure artifacts.
- Add measured latency, memory and backend-I/O observations.
- Add backup/restore validation and SQLite corruption drills.
- Add an explicit release-baseline registry separate from resumable runs.

## Permanent non-claims

- SQLite run storage is not a distributed scheduler.
- A completed run report does not prove the external resolver/backend avoided logging.
- Resume identity does not guarantee deterministic behavior from an ungoverned stochastic backend.
- Text-free metric storage is not a substitute for storage encryption or access control.
- Release readiness is not claimed.
