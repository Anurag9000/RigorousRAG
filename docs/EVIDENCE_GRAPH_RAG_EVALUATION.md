# Evidence-graph retrieval evaluation, benchmarks and regression gates

Last updated: 2026-08-02

## Scope

This layer evaluates bounded evidence-graph selections without retaining raw query text, evidence text, source paths or provider responses. It measures exact generation-scoped retrieval and stored-path coverage, produces reproducible repeated-run reports, and compares historical reports under a versioned non-inferiority policy.

It does not change runtime retrieval policy or publish citations automatically.

## Per-case metrics

`tools/evidence_graph_rag_evaluation.py` defines strict gold cases using:

- query ID;
- graph-set ID and digest;
- query SHA-256;
- required `(document ID, generation, node ID)` locators;
- required traversal edge IDs;
- expected abstention.

Per-case evaluation reports:

- node precision, recall and F1;
- document precision, recall and F1;
- traversal-edge precision, recall and F1;
- complete required-path success;
- expanded-evidence lineage completeness;
- abstention correctness;
- retained evidence/traversal counts;
- estimated work units;
- deterministic case, selection and evaluation digests.

An abstention gold case cannot contain required nodes or edges.

Metric conventions are explicit:

- empty prediction and empty gold receive perfect precision/recall/F1;
- non-empty prediction against empty gold receives zero precision/recall/F1;
- empty prediction against non-empty gold receives precision 1 and recall/F1 0.

These are retrieval conventions, not semantic-entailment measures.

## Query-digest-only benchmark fixtures

`tools/evidence_graph_rag_benchmark.py` separates gold identity from a text-free selection observation.

A selection observation contains only:

- graph-set/query/selection digests;
- selected generation-scoped node locators;
- traversal edge IDs;
- expanded-lineage validity booleans;
- abstention;
- evidence, traversal and work counts.

It does not contain query text, node text, answer text or citations.

A fixture contains repeated runs with explicit run IDs and seeds. Every run must use the same ordered gold-case contract. The benchmark fingerprint binds:

- benchmark ID;
- ordered gold-case digests;
- run seeds;
- schema version.

It deliberately excludes selection outputs and resource observations, allowing baseline and candidate reports to remain comparable under one governed contract.

Reports include per-run and macro aggregates, run/result digests, run/seed/case counts, a benchmark fingerprint and deterministic report digest.

## Benchmark CLI

```bash
python -m tools.evidence_graph_rag_benchmark_cli inspect graph_rag_fixture.json

python -m tools.evidence_graph_rag_benchmark_cli run graph_rag_fixture.json \
  --output-file graph_rag_report.json
```

The CLI provides:

- strict JSON with duplicate-key and NaN/Infinity refusal;
- bounded byte reads;
- file identity checks during reads;
- symlink/reparse refusal for input/output paths;
- atomic same-filesystem report replacement;
- contract-only inspection;
- explicit `contains_raw_query=false` and `contains_evidence_text=false` output.

## Historical regression policy

`tools/evidence_graph_rag_regression.py` compares two benchmark reports only when they share:

- benchmark fingerprint;
- run and case dimensions;
- exact `(run ID, seed)` identities;
- exact per-run gold-contract digests.

The default `graph-rag-conservative-v1` policy requires minimum run, seed and case counts; aggregate floors for node/document/edge F1, complete-path rate, lineage completeness and abstention accuracy; metric-specific non-inferiority margins; and a bounded mean estimated-work ratio.

For each metric, paired deltas are computed between aligned run aggregates. The current dependency-free interval is a normal approximation:

```text
mean delta ± z(confidence) × sample standard error
```

The lower bound must remain above the negative non-inferiority margin. This approximation is explicitly reported and is not presented as a bootstrap or exact small-sample interval.

A report is `eligible` only when no blocking reason exists. Eligibility does not alter runtime configuration.

## Regression CLI

```bash
python -m tools.evidence_graph_rag_regression_cli compare \
  baseline_report.json candidate_report.json \
  --policy-file reviewed_graph_rag_policy.json \
  --output-file regression_report.json
```

Exit status:

- `0`: valid eligible comparison;
- `1`: valid blocked comparison;
- `2`: malformed, forged, misaligned or unavailable input.

Output includes aggregate deltas, paired intervals, work ratio, policy/report digests and deterministic reason codes. It explicitly reports:

```json
{
  "paired_interval_method": "normal_approximation_over_run_deltas",
  "contains_raw_query": false,
  "contains_evidence_text": false,
  "runtime_policy_changed": false
}
```

## Focused verification

Focused contracts passed:

- **6** evaluation tests;
- **10** benchmark producer/CLI tests;
- **8** regression engine/CLI tests.

They cover perfect and partial retrieval, complete-path and lineage metrics, abstention, identity mismatch, macro aggregation, strict schemas, ordered run contracts, contract-only fingerprints, exact count invariants, duplicate-key/NaN refusal, path defenses, atomic report writes, floors, paired non-inferiority, work ratios, forged digest refusal and valid blocked/eligible exit semantics.

A fresh exact-current `main` archive passed the complete Wave 5 focused family at **90/90 tests**, and every `tools/evidence_graph*.py` and `scripts/evidence_graph*.py` file compiled against the repository’s real public modules. This is focused exact-current verification, not the full repository release matrix.

## Remaining work

- Execute fixtures from live selectors rather than consuming pre-collected observations.
- Add bootstrap or permutation intervals and multiple-comparison controls for promotion decisions.
- Add measured wall-clock, memory and backend-I/O accounting alongside estimated work.
- Add governed benchmark dataset cards, checksums, split/version/license metadata and annotation guidance.
- Add relation precision, stale-set behavior and adversarial corruption cases.
- Store historical baselines under explicit release/version governance.
- Add dashboards only after retention and privacy review.
- Add authoritative citation-conversion and agent/API integration tests.

## Permanent non-claims

- Node/path recall does not prove factual correctness or entailment.
- Complete stored-path coverage does not prove causality.
- The normal interval is an approximation, especially with few runs.
- Estimated work units are not measured latency, memory or monetary cost.
- An eligible historical comparison does not change runtime behavior.
- Release readiness is not claimed.
