# Bounded authoritative evidence-graph selection

Last updated: 2026-08-02

## Scope

The first GraphRAG slice is a read-only evidence selector. It resolves one current authoritative cross-document graph set, searches each exact member graph, expands only stored explicit edges, and returns privacy-finalized evidence nodes with complete generation and traversal lineage.

It does not generate an answer, summarize a path, infer new relations or convert evidence into citations.

## Authority boundary

`select_current_graph_set_evidence(...)`:

1. resolves the logical current graph set through member authority checks;
2. acquires every member owner/document lock in deterministic document order;
3. resolves the current set again after acquiring locks;
4. rejects any pointer change before selection;
5. loads every exact member graph generation;
6. verifies graph digest, content hash and profile fingerprint;
7. performs bounded selection;
8. resolves the current set and member authority again before returning;
9. rejects any set or authority-digest change during selection.

The locks coordinate only one process. Read-time authority checks remain required because another process may advance member generations.

## Lexical seeds

Each exact member graph is searched independently through the existing deterministic lexical graph search.

Budgets include:

- maximum member graphs;
- per-document lexical hits;
- global lexical seeds;
- node-type filters;
- total retained evidence.

Unsupported node types fail closed rather than silently producing empty output.

## Explicit traversal

Two bounded expansion modes are supported:

- within-document outgoing edges already stored in the member graph;
- reviewed cross-document edges already stored in the current graph set.

No edge is synthesized from textual similarity.

Cross-document traversal validates the target member generation, graph digest, node ID and node provenance digest before materializing evidence. Traversal supports bounded depth, per-seed expansion limits, edge-type filters and cycle prevention.

Unsupported edge filters fail closed.

## Evidence values

Each `GraphEvidenceItem` records:

- owner, document and exact graph generation;
- graph and node identities;
- node type, label and privacy-finalized text;
- page and section locator;
- deterministic score and matched query terms;
- node provenance digest;
- lexical, within-document or cross-document origin;
- traversal-step digests for expanded evidence.

The item digest contains a text SHA-256 rather than embedding raw text into the digest material.

`GraphTraversalStep` records exact source/target document generations, edge identity/type/provenance, depth and weight.

## Deduplication and ordering

Evidence is unique by `(document, generation, node ID)`.

When the same node is both a lexical hit and a traversal target, lexical evidence wins at equal or better score. Final ordering is deterministic by score, origin priority, document, type and node ID.

Only traversal steps referenced by retained evidence are returned.

## Selection output

`GraphEvidenceSelection` contains:

- graph-set and authority identities;
- query SHA-256 rather than raw query text;
- selected evidence and traversals;
- retained lexical and expanded counts;
- bounded estimated work units;
- exact abstention state;
- deterministic selection digest.

The type permanently enforces:

```json
{
  "citation_conversion_performed": false,
  "answer_generated": false
}
```

An empty lexical seed set produces conservative abstention and no traversal.

## Focused verification

The focused selector harness passed **9 tests** covering:

- bounded multi-document lexical selection;
- absence of answer generation and citation conversion;
- reviewed cross-document expansion;
- stored within-document edge expansion;
- lexical precedence over duplicate traversal evidence;
- stale set and member graph drift refusal;
- cross-edge target provenance mismatch refusal;
- empty-evidence abstention;
- unsupported node/edge filter refusal;
- before/after current-set re-resolution.

The complete current-main archive focused Wave 5 suite passed after adding this selector, and all evidence-graph modules/scripts compiled against the repository’s real public types. This remains focused verification, not the full Linux/Windows/container release matrix.

## Remaining work

- Add graph-specific retrieval and path-completeness evaluation metrics.
- Add strict local benchmark fixtures and historical regression reports.
- Add server-owned conversion from selected graph evidence to the existing authoritative citation/evidence registry.
- Add an agent tool only after citation conversion and abstention tests pass.
- Add API/browser propagation tests.
- Add learned or semantic retrieval only behind benchmarked adapters; lexical and explicit-edge selection remains the deterministic baseline.
- Add distributed read/write coordination for multi-process graph publication.

## Permanent non-claims

- Lexical graph score is not semantic truth.
- A stored path explains reviewed graph assertions; it is not causal proof.
- Traversal expansion does not verify entailment.
- Selected evidence is not automatically citation-ready.
- No answer is produced by this layer.
- Release readiness is not claimed.
