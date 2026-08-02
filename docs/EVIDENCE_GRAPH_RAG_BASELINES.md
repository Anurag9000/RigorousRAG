# Governed GraphRAG historical baselines

Last updated: 2026-08-02

## Purpose

The baseline registry records which text-free GraphRAG benchmark report is the governed historical comparison point for one exact benchmark contract and one regression policy. It is append-only and changes only a baseline pointer; it does not install a selector, alter runtime retrieval, publish citations, or modify an evidence graph.

## Scope identity

A current pointer is scoped by:

```text
benchmark fingerprint + policy ID
```

The benchmark fingerprint binds the ordered gold contract and run seeds. The policy ID and policy digest bind the quality floors, paired non-inferiority margins, confidence level, minimum run/seed/case counts, and estimated-work ceiling.

Different policy IDs maintain independent baseline histories and pointers.

## Baseline record

`GraphRAGBaselineRecord` stores:

- benchmark fingerprint and benchmark ID;
- policy ID and policy digest;
- the complete validated text-free benchmark report;
- previous baseline digest for replacements;
- eligible regression-report digest authorizing a replacement;
- activation timestamp and schema version;
- a deterministic baseline digest excluding only activation time.

An initial baseline contains neither a previous-baseline digest nor a regression digest. A replacement contains both.

## First baseline activation

First activation requires an explicit no-current expectation. It fails if a current baseline already exists.

```bash
python -m tools.evidence_graph_rag_baseline_cli initialize \
  candidate_report.json \
  --policy-file graph_rag_policy.json \
  --expect-no-current
```

The candidate report is reconstructed through the strict benchmark-report schema and its optional report digest is verified.

## Replacement activation

Replacement requires:

1. the exact current baseline digest;
2. a candidate report under the same benchmark fingerprint;
3. an eligible regression report;
4. regression baseline-report digest equal to the current baseline report;
5. regression candidate-report digest equal to the candidate;
6. regression policy ID and digest equal to the selected policy;
7. no blocking reason codes.

```bash
python -m tools.evidence_graph_rag_baseline_cli promote \
  candidate_report.json regression_report.json \
  --policy-file graph_rag_policy.json \
  --expected-current-baseline-digest <current-baseline-sha256>
```

Blocked, forged, misaligned, stale-current, or policy-mismatched replacements fail closed.

## Durable store

`GraphRAGBaselineStore` uses SQLite with:

- append-only immutable baseline rows;
- one current pointer per benchmark fingerprint and policy ID;
- explicit expected-current comparison inside one immediate transaction;
- idempotent same-identity insertion;
- strict nested report reconstruction;
- row, payload, report, policy, lineage, and pointer digest checks;
- bounded payload size;
- symlink/reparse, parent identity, and database identity defenses.

Suggested configuration:

```dotenv
EVIDENCE_GRAPH_RAG_BASELINE_DB_PATH=data/evidence_graph_rag_baselines.sqlite3
```

The runtime factory is path-scoped and defaults to that location.

## Read-only inspection

```bash
python -m tools.evidence_graph_rag_baseline_cli status \
  <benchmark-fingerprint> --policy-id graph-rag-conservative-v1

python -m tools.evidence_graph_rag_baseline_cli history \
  <benchmark-fingerprint> --policy-id graph-rag-conservative-v1 \
  --limit 100
```

Status and history expose only benchmark, report, policy, baseline, lineage, and regression digests plus timestamps. They explicitly report:

```json
{
  "contains_raw_query": false,
  "contains_evidence_text": false,
  "runtime_policy_changed": false
}
```

## Focused verification

The baseline registry stack passed **9 focused tests** covering:

- explicit first-baseline activation;
- stale/no-current expectation refusal;
- exact eligible-regression replacement;
- blocked and identity-mismatched regression refusal;
- policy-scoped independent pointers;
- append-only history and replacement lineage;
- payload/database replacement tamper detection;
- path-scoped runtime caching;
- privacy-safe initialize/promote/status/history CLI behavior;
- bounded missing-baseline behavior.

## Remaining work

- Add signed or append-only baseline audit export.
- Add baseline retention, legal-hold, backup, and restore policy.
- Add reviewer authorization and separation of duties for baseline replacement.
- Add exact-current CI release labels pointing to baseline digests.
- Add multiple-comparison-aware statistical methods before broad model/policy promotion.
- Add measured runtime resource observations to baseline reports.
- Add distributed pointer coordination before concurrent multi-process promotion.

## Permanent non-claims

- A baseline is a governed comparison reference, not proof of scientific truth.
- An eligible regression report is only as reliable as its benchmark contract, labels, runs, and statistical assumptions.
- Baseline activation does not change runtime retrieval behavior.
- SQLite pointer atomicity is not distributed consensus.
- Release readiness is not claimed.
