# Governed migration promotion gates

Last updated: 2026-08-02

## Scope

Validated migration shadows remain isolated from the live vector, sparse and generation stores. The promotion control plane can now:

1. consume a strict, query-ID-only paired benchmark fixture;
2. derive aggregate retrieval, support, citation, abstention and resource evidence;
3. compute paired confidence intervals over repeated runs;
4. apply aggregate quality/resource gates and paired statistical non-inferiority gates;
5. persist an append-only `eligible` or `blocked` report.

It does **not** mutate a live generation, replace vector or sparse state, change a durable current pointer, delete rollback state or expose a cutover command.

## Components

### Aggregate promotion policy

`tools/migration_promotion.py` provides:

- strict retrieval-quality and resource observations;
- the versioned `conservative-v1` policy;
- exact task, shadow-manifest, evidence and live source-generation alignment;
- quality floors and maximum point-estimate regressions;
- p95 latency, peak-memory, storage and estimated-cost ratios;
- deterministic evidence, policy and report digests;
- sorted bounded reason codes instead of private exception text.

### Repository-owned paired benchmark producer

`tools/migration_benchmark.py` and `tools/migration_benchmark_cli.py` provide:

- strict paired current/shadow runs over the same ordered query contract;
- query identifiers, relevance identifiers and aggregate support/citation outcomes only;
- recall@k, nDCG@k, MRR, support recall, citation precision and abstention accuracy;
- conservative p95/max resource aggregation and mean estimated cost;
- repeated-run and distinct-seed accounting;
- signed 95% paired-delta confidence intervals;
- a benchmark fingerprint derived from the governed query contract, not model outputs;
- strict duplicate-key, NaN/Infinity, symlink/reparse and file-identity refusal;
- atomic evidence and optional detailed interval-report publication.

`tools/migration_promotion_cli.py evaluate-fixture` invokes this producer in-process, so the recommended promotion path no longer trusts a manually authored aggregate evidence file.

### Paired statistical gate

`tools/migration_statistical_gate.py` provides:

- the versioned `paired-noninferiority-v1` policy;
- minimum repeated-run, seed-count and confidence-level requirements;
- lower-confidence-bound non-inferiority checks for all six quality metrics;
- optional lower-confidence-bound practical-gain thresholds;
- deterministic per-metric assessment records and assessment digest;
- composite evidence and policy digests attached to the existing promotion-report schema;
- final blocking whenever either the aggregate or statistical gate fails.

The default non-inferiority margins are:

- recall@k: 0.01;
- nDCG@k: 0.01;
- MRR: 0.01;
- support recall: 0.01;
- citation precision: 0.00;
- abstention accuracy: 0.01.

Practical-gain thresholds are disabled by default and must be deliberately configured per metric.

### Append-only audit storage

`tools/migration_promotion_store.py` and `tools/migration_promotion_runtime.py` provide:

- immutable reports addressed by report digest;
- one atomic per-task `current.json` pointer;
- idempotent reuse of an identical report without timestamp-only churn;
- strict duplicate-key and non-standard-number refusal;
- symlink/reparse, root-identity, member-type and byte-size defenses;
- no retained-source paths, raw queries, passages or provider responses.

## Paired fixture contract

A benchmark fixture contains immutable migration identities and one or more paired runs. Each run uses the same ordered query contract but may use a different seed.

```json
{
  "task_id": "<64-character task id>",
  "validation_digest": "<shadow validation SHA-256>",
  "source_sequence": 12,
  "source_content_sha256": "<finalized content SHA-256>",
  "vector_count": 42,
  "sparse_count": 42,
  "rank_cutoff": 10,
  "runs": [
    {
      "seed": 1,
      "cases": [
        {
          "query_id": "case-001",
          "relevant_ids": ["doc-a", "doc-b"],
          "current_ranked_ids": ["doc-a", "doc-c"],
          "shadow_ranked_ids": ["doc-a", "doc-b"],
          "support_total": 2,
          "current_support_found": 1,
          "shadow_support_found": 2,
          "current_citation_count": 2,
          "current_valid_citation_count": 2,
          "shadow_citation_count": 2,
          "shadow_valid_citation_count": 2,
          "should_abstain": false,
          "current_abstained": false,
          "shadow_abstained": false
        }
      ],
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
  ]
}
```

The fixture deliberately does not accept raw query text, answer text, retrieved passages, retained paths or provider responses. Unknown fields are rejected.

All runs must use the same ordered sequence of:

- `query_id`;
- relevant identifiers;
- total support-fact count;
- expected abstention behavior.

Ranked outputs, observed support/citation outcomes and resource observations may differ between current and shadow systems and across seeds.

## Aggregate evidence compatibility path

The strict aggregate evidence schema remains available for compatibility and independent pipelines:

```bash
python -m tools.migration_promotion_cli evaluate <task-id> \
  --evidence-file migration_evidence.json \
  --policy-file reviewed_policy.json
```

This path applies aggregate point-estimate gates only because it has no paired per-run intervals. The preferred path is `evaluate-fixture`.

## Recommended operator workflow

Inspect and validate the benchmark contract:

```bash
python -m tools.migration_benchmark_cli inspect \
  --fixture-file paired_fixture.json
```

Optionally materialize aggregate evidence and the detailed interval report:

```bash
python -m tools.migration_benchmark_cli run \
  --fixture-file paired_fixture.json \
  --evidence-output migration_evidence.json \
  --report-output migration_benchmark_report.json
```

Generate evidence, apply aggregate and statistical gates, and persist the final report in one process:

```bash
python -m tools.migration_promotion_cli evaluate-fixture <task-id> \
  --fixture-file paired_fixture.json \
  --policy-file reviewed_promotion_policy.json \
  --statistical-policy-file reviewed_statistical_policy.json
```

Inspect current or historical reports:

```bash
python -m tools.migration_promotion_cli status <task-id>
python -m tools.migration_promotion_cli status <task-id> \
  --report-digest <report-sha256>
python -m tools.migration_promotion_cli history <task-id> --limit 100
```

Remove reports only after the migration task is failed or cancelled and the task ID is repeated exactly:

```bash
python -m tools.migration_promotion_cli remove-task <task-id> \
  --confirm-task-id <same-task-id>
```

Equivalent script entrypoints are:

```bash
python scripts/migration_benchmarks.py ...
python scripts/migration_promotions.py ...
```

Exit status is `0` for an eligible final report, `1` for a valid blocked report or bounded not-found result, and `2` for invalid/unavailable input or state.

## Final report semantics

A persisted report records:

- task, owner, document and source-generation identity;
- source and target profile fingerprints;
- shadow validation and benchmark fingerprints;
- evidence and policy digests;
- `eligible` or `blocked`;
- deterministic aggregate/statistical reason codes;
- quality deltas and resource ratios;
- evaluation timestamp and schema version.

For `evaluate-fixture`, the final evidence digest commits to both the aggregate benchmark evidence and the statistical-assessment digest. The final policy digest commits to both aggregate and statistical policies. The report schema remains backward-compatible.

The report digest deliberately excludes only the evaluation timestamp. Re-evaluating identical evidence under identical policies reuses the immutable first report.

## Focused verification

The constrained local promotion/benchmark/statistical harness passed **42 tests** covering:

- eligible and blocked aggregate decisions;
- manifest, journal, evidence and source-generation alignment;
- benchmark minimums and confidence level;
- quality floors and point-regression ceilings;
- latency, memory, storage and estimated-cost ratios;
- zero-baseline fail-closed behavior;
- strict paired query-contract enforcement;
- recall@k, nDCG@k, MRR, support, citation and abstention calculations;
- contract-only benchmark fingerprints;
- repeated runs, distinct seeds and signed paired confidence intervals;
- paired lower-bound non-inferiority;
- optional practical-gain blocking;
- deterministic evidence, aggregate-policy, statistical-policy, assessment and report digests;
- direct in-process fixture evaluation;
- append-only history and current-pointer behavior;
- report and pointer tamper detection;
- strict JSON, unknown-field, symlink and replaced-root refusal;
- exact-confirmation cleanup and task-state restrictions;
- path-free CLI and persisted reports.

This is focused verification in an isolated harness. It is not the complete exact-head repository matrix.

## Remaining before cutover can exist

An `eligible` report is necessary but not sufficient for live promotion. The repository still requires:

1. adapters that execute the governed fixture against the actual current and shadow retrieval stacks rather than consuming already collected ranked identifiers;
2. measured wall-clock latency, process/device memory, artifact storage and provider billing rather than operator-supplied resource observations;
3. additional reviewed statistical procedures where appropriate, such as bootstrap intervals, permutation tests, multiple-comparison correction and practical-effect governance;
4. an atomic vector+sparse+generation cutover transaction;
5. durable rollback references and exact rollback verification;
6. cutover and rollback leases, idempotency keys and crash recovery;
7. bounded shadow/report retention and cleanup;
8. fault injection at every pre-cutover, pointer-swap, registry and rollback boundary;
9. a clean exact-head Linux, Windows and container verification matrix.

## Permanent non-claims

- `eligible` does not mean the target model is scientifically superior.
- Non-inferiority under this benchmark does not prove universal non-inferiority.
- Confidence intervals over repeated configured runs do not automatically represent every deployment source of uncertainty.
- Aggregate metrics do not prove claim-level entailment or factual correctness.
- Estimated cost is not measured monetary cost unless the benchmark producer supplies measured accounting.
- A promotion report does not authorize or perform cutover.
- Release readiness is not claimed without the exact-head repository verification matrix.
