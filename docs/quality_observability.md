# Unified retrieval-quality observability

`evaluation.quality_observability` is the repository-wide aggregation contract for
retrieval/RAG quality telemetry. It deliberately does **not** reimplement metric
algorithms. Retrieval metrics, semantic support, calibration, conformal selective risk,
latency/resource measurement, and drift stay in their existing owning modules; this
layer normalizes their outputs into one deterministic, auditable representation.

## What the contract records

Every `QualitySnapshot` binds:

- a bounded observation window (`started_at_unix`, `ended_at_unix`,
  `generated_at_unix`);
- run, system, and domain identifiers;
- SHA-256 identities for the dataset manifest, evaluation split, and evaluation
  contract;
- the source-code revision;
- optional retrieval-stack, model, and environment SHA-256 identities; and
- a sorted set of scalar `MetricObservation` values.

Each metric has an explicit direction (`higher`, `lower`, or `neutral`), unit,
sample count, source module, and optional non-content dimensions. The snapshot rejects
duplicate metric identities instead of silently averaging values whose aggregation
semantics may be invalid (for example, averaging independently computed p95 values).

The canonical JSON payload is SHA-256 digested. Input metric order therefore cannot
change snapshot identity.

## Existing metric adapters

The module has adapters for the repository's existing producers:

- `observations_from_benchmark_suite`
- `observations_from_retrieval_metrics`
- `observations_from_generation_metrics`
- `observations_from_semantic_metrics`
- `observations_from_selective_risk`
- `observations_from_latency_summary`
- `observations_from_resource_usage`
- `observations_from_drift_report`

`observations_from_mapping` is the escape hatch for another scalar metric producer.
It requires an explicit direction for every metric; the observability layer never
guesses whether an unknown metric should increase or decrease.

The benchmark-suite adapter exports aggregate scores and latency only. It does not
copy benchmark queries, generated answers, documents, evidence passages, or row text
into observability artifacts.

## Privacy boundary

Metric dimensions are intentionally narrow. Tag keys that imply raw query/prompt/
answer/text/content/evidence/document payloads are rejected. Arbitrary metadata keys
are also rejected; dimensions must be from the approved non-content set or use an
identity/version/digest suffix such as `_id`, `_version`, `_digest`, or `_sha256`.

This does not replace upstream data-governance controls. It creates a fail-closed
observability schema whose standard adapters have no field in which to serialize raw
evaluation text.

## SLO evaluation

`QualitySLO` defines a scalar threshold against one metric identity. An SLO can select
dimension values with `tag_match`.

`evaluate_quality_slos` and `build_quality_dashboard` behave as follows:

- exactly one matching metric: evaluate the comparator;
- no matching required metric: `missing`, failed;
- no matching optional metric: `optional_missing`, passed;
- more than one matching metric: `ambiguous`, failed.

This prevents a dashboard from silently selecting one of several route-, language-, or
cohort-specific measurements.

`QualityDashboard` includes the snapshot digest, all normalized scalar metrics, SLO
results, failed-SLO names, health status, and its own canonical digest.

## Comparison and trend building

`compare_quality_snapshots` requires the same system, domain, dataset-manifest digest,
split digest, and evaluation-contract digest. Model, retrieval stack, environment,
code revision, and run identity may change because those are common experiment
variables.

For matched metrics it reports:

- raw absolute delta;
- relative delta when the baseline is non-zero;
- direction-normalized delta;
- `improved`, `regressed`, `unchanged`, or (for neutral metrics) `changed`.

Metrics appearing only in one snapshot are marked `new` or `missing`. Direction or
unit changes for the same metric identity fail closed because they represent a contract
change, not a comparable trend.

A sequence of snapshots can therefore be compared pairwise without changing metric
semantics or losing the immutable source snapshot identities.

## Durable machine-readable export

`write_quality_snapshot` and `write_quality_dashboard` emit canonical JSON with a
trailing newline. They write to a same-directory temporary file, flush and `fsync`,
replace atomically, reject redirecting output paths, and request owner-only permissions
where the platform supports them.

A consumer should verify `snapshot_digest` or `dashboard_digest` before loading an
artifact into a long-term dashboard, experiment registry, or release gate.

## Minimal example

```python
from evaluation.quality_observability import (
    QualityProvenance,
    QualitySLO,
    QualitySnapshot,
    QualityWindow,
    build_quality_dashboard,
    observations_from_retrieval_metrics,
)

metrics = observations_from_retrieval_metrics(
    {"ndcg@10": 0.81, "recall@10": 0.92},
    sample_count=500,
    tags={"route": "hybrid"},
)

snapshot = QualitySnapshot(
    QualityWindow(1_700_000_000, 1_700_003_600, 1_700_003_601),
    QualityProvenance(
        run_id="eval-42",
        system_id="rigorous-rag",
        domain_id="research",
        dataset_manifest_digest="a" * 64,
        split_digest="b" * 64,
        evaluation_contract_digest="c" * 64,
        code_revision="deadbeef",
        retrieval_stack_digest="d" * 64,
        model_digest="e" * 64,
    ),
    metrics,
)

dashboard = build_quality_dashboard(
    snapshot,
    (
        QualitySLO(
            "hybrid ndcg",
            "retrieval.ndcg@10",
            ">=",
            0.80,
            tag_match={"route": "hybrid"},
        ),
    ),
)
assert dashboard.healthy
```

The example uses placeholder digests only to illustrate the API. Production callers
must bind the actual immutable artifact identities used for that evaluation run.
