# Governed migration promotion gates

Last updated: 2026-08-02

## Scope

Validated migration shadows remain isolated from the live vector, sparse and generation stores. This control plane evaluates aggregate benchmark evidence and writes an append-only `eligible` or `blocked` report. It does **not** mutate a live generation, change a current pointer, delete rollback state or provide a cutover command.

## Components

- `tools/migration_promotion.py`
  - strict retrieval-quality and resource observations;
  - a conservative versioned default policy;
  - exact task, shadow-manifest and source-generation alignment;
  - deterministic evidence, policy and report digests;
  - sorted bounded reason codes instead of private exception text.
- `tools/migration_promotion_store.py`
  - append-only immutable reports addressed by report digest;
  - one atomic per-task `current.json` pointer;
  - strict duplicate-key and non-standard-number refusal;
  - symlink/reparse, root-identity, member-type and byte-size defenses;
  - no retained-source paths, raw queries or evidence passages.
- `tools/migration_promotion_runtime.py`
  - path-scoped process-local report-store factory.
- `tools/migration_promotion_cli.py` and `scripts/migration_promotions.py`
  - evaluate strict aggregate evidence;
  - inspect the current or a historical report;
  - inspect bounded report history;
  - remove reports only for failed or cancelled tasks under exact confirmation;
  - no live commit or cutover action.

## Required evidence

Evidence files are strict JSON objects containing only aggregate metrics and immutable identities:

```json
{
  "task_id": "<64-character task id>",
  "validation_digest": "<shadow validation SHA-256>",
  "benchmark_fingerprint": "<dataset/configuration SHA-256>",
  "source_sequence": 12,
  "source_content_sha256": "<finalized content SHA-256>",
  "vector_count": 42,
  "sparse_count": 42,
  "repeated_runs": 5,
  "seed_count": 5,
  "confidence_interval_level": 0.95,
  "current_quality": {
    "query_count": 100,
    "recall_at_k": 0.80,
    "ndcg_at_k": 0.74,
    "mrr": 0.72,
    "support_recall": 0.82,
    "citation_precision": 0.96,
    "abstention_accuracy": 0.90
  },
  "shadow_quality": {
    "query_count": 100,
    "recall_at_k": 0.82,
    "ndcg_at_k": 0.76,
    "mrr": 0.73,
    "support_recall": 0.83,
    "citation_precision": 0.96,
    "abstention_accuracy": 0.91
  },
  "current_resources": {
    "p95_latency_ms": 100.0,
    "peak_memory_bytes": 1000000000,
    "index_bytes": 500000000,
    "estimated_cost_units": 10.0
  },
  "shadow_resources": {
    "p95_latency_ms": 120.0,
    "peak_memory_bytes": 1100000000,
    "index_bytes": 650000000,
    "estimated_cost_units": 11.0
  }
}
```

Unknown fields are rejected. Raw queries, retrieved passages, source paths and provider responses do not belong in promotion evidence.

## Default conservative policy

The built-in `conservative-v1` policy requires:

- at least 50 benchmark queries;
- at least three repeated runs and three seeds;
- confidence level of at least 0.95;
- equal shadow vector and sparse row counts;
- shadow quality floors for recall, nDCG, MRR, support recall, citation precision and abstention accuracy;
- no quality regression beyond the metric-specific allowance;
- p95 latency, peak memory, storage and estimated-cost ratios within configured ceilings;
- the live source generation to remain active/restored with the exact source sequence, profile fingerprint and content hash used to build the shadow.

An optional strict policy JSON file may override named policy fields. Unknown policy fields, invalid probabilities, non-positive ratios, duplicate JSON keys and NaN/Infinity values are rejected.

## Operator commands

```bash
python -m tools.migration_promotion_cli evaluate <task-id> \
  --evidence-file migration_evidence.json

python -m tools.migration_promotion_cli evaluate <task-id> \
  --evidence-file migration_evidence.json \
  --policy-file reviewed_policy.json

python -m tools.migration_promotion_cli status <task-id>
python -m tools.migration_promotion_cli status <task-id> \
  --report-digest <report-sha256>
python -m tools.migration_promotion_cli history <task-id> --limit 100

python -m tools.migration_promotion_cli remove-task <task-id> \
  --confirm-task-id <same-task-id>
```

Exit status is `0` for an eligible report, `1` for a valid blocked report or bounded not-found result, and `2` for invalid/unavailable input or state.

## Report semantics

A report records:

- task, owner, document and source-generation identity;
- source and target profile fingerprints;
- shadow validation, benchmark, evidence and policy digests;
- `eligible` or `blocked`;
- deterministic reason codes;
- quality deltas and resource ratios;
- the evaluation timestamp and schema version.

The report digest deliberately excludes only the evaluation timestamp. Re-evaluating identical evidence under the same policy reuses the immutable first report instead of creating timestamp-only audit churn.

## Focused verification

The constrained local harness passed 21 tests covering:

- eligible and blocked decisions;
- manifest, journal, evidence and source-generation alignment;
- benchmark minimums and confidence level;
- quality floors and regression ceilings;
- latency, memory, storage and estimated-cost ratios;
- zero-baseline fail-closed behavior;
- deterministic evidence, policy and report digests;
- closed-schema strict JSON input;
- append-only history and current-pointer behavior;
- report and pointer tamper detection;
- symlink and replaced-root refusal;
- exact-confirmation cleanup and task-state restrictions;
- path-free CLI and persisted reports.

## Remaining before cutover can exist

An `eligible` report is necessary but not sufficient for live promotion. The repository still requires:

1. a repository-owned benchmark runner that produces the aggregate evidence from governed query fixtures rather than accepting only externally generated aggregate JSON;
2. confidence intervals and statistical/practical-effect tests produced by that runner;
3. an atomic vector+sparse+generation cutover transaction;
4. durable rollback references and exact rollback verification;
5. cutover and rollback leases, idempotency keys and crash recovery;
6. bounded shadow/report retention and cleanup;
7. fault injection at every pre-cutover, pointer-swap, registry and rollback boundary;
8. a clean exact-head Linux, Windows and container verification matrix.

## Permanent non-claims

- `eligible` does not mean the target model is scientifically superior.
- Aggregate metrics do not prove claim-level entailment or factual correctness.
- Estimated cost is not measured monetary cost unless the benchmark producer supplies measured accounting.
- A promotion report does not authorize or perform cutover.
- Release readiness is not claimed without the exact-head repository verification matrix.
