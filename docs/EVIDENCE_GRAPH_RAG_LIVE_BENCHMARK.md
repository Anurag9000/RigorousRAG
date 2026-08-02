# Live authoritative evidence-graph benchmark execution

Last updated: 2026-08-02

## Scope

The live benchmark bridge executes governed graph-retrieval cases through the actual current-set selector while preventing query and evidence text from entering benchmark artifacts.

A plan contains only benchmark identity, run seeds, strict gold cases, query SHA-256 values and a closed selector configuration. Query text is supplied just in time by an injected resolver, verified against the governed digest, used for one selector call, reduced immediately to a text-free observation and not returned in the result.

## Components

`tools/evidence_graph_rag_live_benchmark.py` provides:

- `GraphRAGLiveBenchmarkPlan`;
- `GraphRAGLiveBenchmarkResult`;
- conversion from a validated `GraphEvidenceSelection` to a text-free `GraphRAGSelectionObservation`;
- callback-based execution for controlled test or adapter environments;
- direct authoritative execution through `select_current_graph_set_evidence` with injected graph-set, generation and evidence-graph stores;
- strict mapping support for plan files without query text.

## Query handling

For each gold case:

1. call `query_resolver(query_id)`;
2. require non-empty bounded text without NUL characters;
3. calculate SHA-256 over the normalized query text;
4. require exact equality with the gold case `query_digest`;
5. invoke the selector only after that equality check;
6. convert the returned selection to identities, counts and lineage booleans;
7. delete local query/selection references before continuing.

A digest mismatch fails before the selection runner is called.

The bridge cannot guarantee that an arbitrary external query resolver or selector implementation does not log inputs internally. Production use must therefore inject reviewed implementations. The bridge itself does not persist or return those values.

## Closed selector configuration

Allowed selector fields are limited to:

```text
node_types
within_edge_types
cross_edge_types
per_document_hits
max_lexical_seeds
max_within_per_seed
max_cross_depth
max_cross_per_seed
max_total_items
```

Unknown fields are rejected. Type-filter lists must contain non-empty text values. Numeric fields must be bounded integers.

The current deterministic selector does not consume random seeds. Seeds remain part of the benchmark contract so future stochastic adapters can be compared under the same explicit repeated-run structure.

## Observation reduction

The reduced observation retains only:

- graph-set and query digests;
- selection digest;
- selected generation-scoped node locators;
- traversal edge IDs;
- validity of expanded-item lineage references;
- abstention;
- evidence, traversal and estimated-work counts.

Evidence node text, labels, source paths, raw queries and provider outputs are discarded from the benchmark representation.

## Direct authoritative execution

`execute_authoritative_graph_rag_benchmark(...)` injects:

- owner and logical graph-set key;
- query resolver;
- graph-set store;
- authoritative generation store;
- evidence-graph store.

It routes every case through `select_current_graph_set_evidence`, inheriting the selector’s member locks, before/after authority rechecks, graph-generation verification, explicit-edge traversal and abstention semantics.

## Focused verification

The focused live bridge passed **6 tests** covering:

- repeated live execution and immediate text reduction;
- absence of query/evidence text from returned reports;
- query-digest mismatch before selector invocation;
- graph-set identity mismatch refusal;
- strict plan schema and closed selector fields;
- direct authoritative selector invocation with injected real-store interfaces.

These focused contracts do not substitute for running representative annotated datasets against deployed vector/sparse/graph backends.

## Remaining work

- Add a governed query provider backed by reviewed local benchmark assets.
- Add timeout, cancellation and per-case failure-report policy for large live suites.
- Add measured latency, memory and backend-I/O collection without query/evidence leakage.
- Add resumable append-only run storage with exact plan/report digests.
- Add distributed execution only after deterministic partition and duplicate-effect contracts exist.
- Add authoritative citation conversion and end-to-end API/browser tests.

## Permanent non-claims

- Digest verification does not make the query non-sensitive while it is in memory.
- External resolvers or backends may log unless separately governed.
- Repeated seeds do not create stochastic variation in the current deterministic baseline.
- A live benchmark report does not change runtime policy.
- Release readiness is not claimed.
