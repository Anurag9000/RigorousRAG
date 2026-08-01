# Capability Wave 5 — provenance evidence graph foundation

Last updated: 2026-08-02

## Scope

Wave 5 now has a generation-scoped, provenance-preserving graph foundation for:

- documents;
- sections;
- claims;
- entities;
- methods;
- datasets;
- citations;
- explicit containment, mention, support, contradiction, citation, method, dataset, derivation and equivalence relations.

The implementation is deliberately explicit-only. It does not infer support, contradiction, entity equivalence, citation intent, methods or datasets from text. Those relations enter the graph only through validated `GraphAnnotation` and `ExplicitGraphRelation` inputs supplied by a reviewed extractor or operator.

## Typed deterministic graph contracts

`tools/evidence_graph_types.py` provides:

- validated `EvidenceNode` and `EvidenceEdge` values;
- owner/document/generation isolation;
- deterministic SHA-256 node IDs from scope, type and natural key;
- deterministic SHA-256 edge IDs from scope, endpoints, type and relation key;
- bounded labels, privacy-finalized text, page/section provenance and metadata;
- finite edge weights;
- self-loop refusal;
- exact endpoint validation;
- exactly one document node per graph batch;
- immutable `EvidenceGraphBatch` graph digests excluding creation time;
- ordered `EvidencePath` values with exact adjacency validation.

Supported node types are:

```text
document, section, claim, entity, method, dataset, citation
```

Supported edge types are:

```text
contains, mentions, supports, contradicts, cites,
uses_method, uses_dataset, derived_from, same_as
```

## Explicit graph construction

`tools/evidence_graph_builder.py` builds one graph for one finalized authoritative document generation.

The builder:

1. verifies owner and document identity;
2. verifies finalized document text and any declared content hash;
3. creates one deterministic document node;
4. creates deterministic section nodes with page, title and section metadata;
5. creates only explicitly supplied claim/entity/method/dataset/citation nodes;
6. attaches each annotation to its declared section or document container;
7. adds only explicitly supplied semantic relations;
8. rejects duplicate keys, unknown endpoints and out-of-range section references;
9. refuses caller-supplied `contains` relations because containment is generated structurally;
10. returns a fully revalidated generation-scoped batch.

Textual similarity or disagreement alone never creates a support or contradiction edge.

## Transactional generation store

`tools/evidence_graph_store.py` provides a path-safe SQLite store with:

- immutable owner/document/generation rows;
- complete strictly encoded graph batches;
- idempotent same-generation/same-digest replay;
- collision refusal when the same generation has different graph content;
- one optimistic current-generation pointer per owner/document;
- exact expected-current checks for commit and activation;
- bounded generation history;
- exact-digest deletion of non-current generations only;
- strict duplicate-key and NaN/Infinity refusal on read;
- graph, row and pointer digest revalidation;
- owner isolation;
- symlink/reparse and database/parent identity defenses.

The store is append-only except for the current pointer and exact-confirmed deletion of a non-current generation.

## Retrieval and path explanations

`tools/evidence_graph_retrieval.py` provides:

- bounded deterministic lexical node search;
- node-type filters;
- matched-term and provenance-digest reporting;
- directed breadth-first simple paths;
- edge-type and intermediate-node-type filters;
- cycle prevention;
- depth, path-count and visited-state ceilings;
- deterministic outgoing-neighbor inspection.

These functions traverse only stored graph edges. They do not synthesize missing relations or claim that a lexical match entails semantic support.

## Explicit support and contradiction analysis

`tools/evidence_graph_analysis.py` provides:

- node and edge counts;
- incoming explicit support clusters for claim nodes;
- incoming explicit contradiction clusters for claim nodes;
- conflict flags only when a claim has both stored support and contradiction edges;
- deterministic cluster and analysis digests.

A support or contradiction edge targeting a non-claim node is counted structurally but is not misrepresented as a claim-evidence cluster. Unlinked contradictory-looking text creates no cluster.

## Runtime and operator surface

`tools/evidence_graph_runtime.py` provides a path-scoped process-local store factory using:

```dotenv
EVIDENCE_GRAPH_DB_PATH=data/evidence_graph.sqlite3
```

`tools/evidence_graph_cli.py` and `scripts/evidence_graphs.py` provide read-only commands:

```bash
python -m tools.evidence_graph_cli status \
  --owner-id alice --doc-id <document-id>

python -m tools.evidence_graph_cli history \
  --owner-id alice --doc-id <document-id> --limit 100

python -m tools.evidence_graph_cli search "randomized protocol" \
  --owner-id alice --doc-id <document-id> --node-type method

python -m tools.evidence_graph_cli paths <source-node-id> <target-node-id> \
  --owner-id alice --doc-id <document-id> --max-depth 6

python -m tools.evidence_graph_cli analyze \
  --owner-id alice --doc-id <document-id>
```

The CLI returns IDs, bounded labels, page/section locators, scores, relation types, counts and digests. It does not return node text. Every successful payload reports `mutation_performed: false`; analysis additionally reports `semantic_inference_performed: false`.

## Focused verification

The clean repository archive passed **28 focused evidence-graph tests** covering:

- deterministic node/edge identities and provenance digests;
- graph scope, exact document-node count and endpoint invariants;
- self-loop, non-finite weight and identity refusal;
- deterministic explicit graph construction across timestamps and iterators;
- content-hash, duplicate-key, unknown-endpoint and section-bound failures;
- absence of inferred support/contradiction relations;
- lexical ranking and type filters;
- directed path ordering, type filters and cycle prevention;
- deterministic neighbor inspection;
- explicit incoming support/contradiction conflict clusters;
- transactional commit, history and current pointers;
- optimistic concurrency and generation collision refusal;
- exact activation and non-current deletion;
- strict stored-JSON and pointer tamper detection;
- owner isolation and database/path identity defenses;
- path-scoped runtime caching;
- read-only privacy-conscious CLI behavior.

All evidence-graph modules and the operator script compiled in the clean archive. These are focused archive tests, not the complete exact-head Linux, Windows and container matrix.

## Deliberate integration boundary

The evidence graph is not yet written automatically during authoritative ingestion. Automatic graph generation would create a fifth persisted participant in the document lifecycle. Adding it as an uncoordinated post-commit callback could leave vector/sparse/generation/registry state current while the graph is absent or stale.

Before automatic integration, the project must choose and implement one of these reviewed designs:

1. a durable derived-index outbox consumer keyed by the authoritative generation;
2. a fifth participant in lifecycle coordination with explicit compensation;
3. a fully rebuildable graph cache with authoritative-generation reconciliation and no correctness dependency on graph availability.

Until then, graph batches are committed only through the validated programmatic `EvidenceGraphStore` boundary. The read-only CLI cannot import or mutate graph state.

## Remaining Wave 5 work

- Add durable derived-graph jobs/outbox and startup/periodic reconciliation.
- Verify graph content hash, profile and generation against the authoritative generation before current-pointer publication.
- Add exact rebuild/delete behavior when an authoritative generation changes or is removed.
- Add cross-document graph-set types without collapsing owner/document/generation provenance.
- Add explicit cross-document citation and entity-resolution review workflows.
- Add GraphRAG retrieval over bounded cross-document paths.
- Add path-aware evidence selection and citation conversion through the authoritative agent registry.
- Add graph retrieval benchmarks, path-completeness metrics and historical regression thresholds.
- Add reviewed scientific extraction adapters; model output must remain closed-schema, provenance-linked and human-auditable.
- Add retention/compaction and graph database backup/restore policy.
- Run exact-head concurrency, corruption, Windows and container verification.

## Permanent non-claims

- A stored `supports` edge is an explicit annotation, not independently verified entailment.
- A stored `contradicts` edge is an explicit annotation, not proof of scientific falsification.
- Lexical graph rank is not semantic truth.
- A graph path is a provenance explanation of stored edges, not causal proof.
- `same_as` is an explicit asserted relation, not automatic entity-resolution certainty.
- The graph is not yet complete for every authoritative document.
- Release readiness is not claimed.
